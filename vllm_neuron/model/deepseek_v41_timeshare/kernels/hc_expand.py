"""The NKI kernel for hc_expand.

One partition module of hf:deepseek-ai/DeepSeek-V4.1-Flash: `hc_expand`, a sequential group
over 1 submodule(s) (hc_expand) — `HyperConnectExpand`, the head of the mHC rank-4 residual
stream: the point where one hidden state becomes `hc_mult` copies of itself, plus the one-hot
mix that makes the first layer's `hc_pre` return copy 0 unchanged.

What this computes, from the frozen spec in `reference_torch.py`
(`HyperConnectExpand.forward`) and `make_identity_pre_mix` in `vendor/model.py`, as
pseudocode:

    h       = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)
    pre_mix = zeros(batch, tok, hc_mult) in float32 ; pre_mix[:, :, 0] = 1.0
    return h, pre_mix

so, written out per element with `x` bfloat16 (1, tok, dim):

    out0[0, t, j, d] = x[0, t, d]                    for every copy j
    out1[0, t, j]    = 1.0 if j == 0 else 0.0        float32

There is no arithmetic here at all. `repeat` is pure data movement — every one of the four
copies is the input's own bfloat16 word, bit for bit — and `make_identity_pre_mix` builds a
constant that reads nothing of `h` but its first two sizes. So the kernel's whole job is
traffic: read 1.25 MiB of `x` out of HBM, write 5 MiB of `out0` and 2 KiB of `out1` back, and
add nothing on the way. That is also why both outputs come out *bit-identical* to the
recording (max_abs_err exactly 0.0 over 2,621,440 + 512 elements): the reference has no
rounding step here for a kernel to reproduce differently, so anything short of bit-identity
would be a real bug rather than numerics.

`hc_mult` is not an argument — the module was constructed with it, `config.json` gives it as
4, and the recorded golden's shape (1, 128, 4, 5120) confirms it — so it is the constant HC
below. `make_identity_pre_mix` names float32 explicitly in `vendor/model.py` regardless of
`h`'s dtype, which is what the recorded fp32 (1, 128, 4) golden shows.

Shape of the computation on the device
-------------------------------------
Nothing is computed, so nothing belongs on an engine: the four copies are produced by DMA
alone, straight from HBM to HBM, with a **broadcast access pattern** on the source. `x` viewed
as (tok, 1, dim) with `.broadcast(1, HC)` is a legal `nisa.dma_copy` source, and one such
transfer per column chunk fills all four copy slots of the destination. The input is never
staged in SBUF at all; the only SBUF tile in the kernel is the 2 KiB `pre_mix`.

`pre_mix` itself is two `nisa.memset`s per token tile — zero the (128, 4) tile, then write 1.0
into its first column — and one store. A memset takes a compile-time constant, which is
exactly what the reference's `zeros` and `[:, :, 0] = 1.0` are.

Where the time goes
-------------------
0.0100 ms for 6.5 MB moved, i.e. ~650 GB/s of DMA against a kernel with zero instructions on
the PE, Vector, Scalar or GpSimd engines. Measured on device, every one of these bit-identical
to the recording:

| variant (CHUNK=1024 unless noted)                       | ms      |
|---------------------------------------------------------|---------|
| HBM->HBM, broadcast source AP  **(this kernel)**        | 0.0100  |
| HBM->HBM, broadcast, 64-token tiles                     | 0.0103  |
| HBM->HBM, broadcast, 32-token tiles                     | 0.0118  |
| SBUF staged, broadcast store from the SBUF tile         | 0.0125  |
| HBM->HBM, one DMA per copy (no broadcast, 4x the DMAs)  | 0.0180  |
| SBUF staged, one store per copy                         | 0.0185  |

Two things worth carrying to the next kernel of this shape. First, the broadcast access
pattern is worth 1.8x over issuing one transfer per copy: the replication happens inside the
one descriptor set, and four transfers of the same bytes do not overlap enough to make up for
it. Second, staging through SBUF costs 25% even though the SBUF tile is written once and read
once — a pure HBM->HBM `dma_copy` needs no engine to hand the data over, and `nisa.dma_copy`
accepts HBM on *both* sides.

CHUNK sweep for the winning variant, every one bit-identical:

    256 -> 0.0126   512 -> 0.0105   640 -> 0.0205   1024 -> 0.0100
    1280 -> 0.0182  2560 -> 0.0182  5120 -> 0.0195      (5120 = one DMA for the whole tensor)

which is not monotonic in size at all: what the DMA wants is a **power-of-two** transfer
width. 640, 1280, 2560 and 5120 all land near 0.019 ms regardless of how big they are, while
256, 512 and 1024 fall on a clean curve where bigger is better. `dim` is 5120 = 2^10 * 5, so
1024 is the largest power-of-two chunk available and it is the optimum — which also means this
constant should be re-derived, not carried over, for any `dim` with a different factorisation.

Nothing here touches act_quant/fp8_gemm/fp4_*/sparse_attn, and the `compat/hyper_connections.py`
that defines this module rebinds no primitive it uses: the slice is pure movement plus a
constant.

NKI and the Python standard library only; the movement happens on the Trainium device.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Tokens per tile. 128 is the partition dimension of SBUF, which is what `pre_mix` is built
#: on; the copies themselves never enter SBUF, so for them this is only how the DMAs are cut
#: up. Smaller tiles measured slower (0.0103 at 64, 0.0118 at 32).
TILE = 128

#: `hc_mult` from `config.json` — the number of hyper-connection copies of the residual stream
#: this model carries, and so the size of the axis `HyperConnectExpand` inserts. The module
#: takes it as a constructor argument, not a forward argument, so it cannot be read off any
#: tensor the kernel receives; the recorded golden's shape (1, 128, 4, 5120) is the cross-check.
HC = 4

#: Columns of `dim` moved per transfer. `dim` is read once and written HC times whatever this
#: is, so it does not change the traffic — it decides the DMA's transfer width, which wants to
#: be a power of two (see the sweep in the module docstring: 640 and 1280 are ~2x worse than
#: 512 and 1024). 1024 is the largest power of two that divides 5120, and the measured
#: optimum. Must divide `dim`.
CHUNK = 1024


@nki.jit
def kernel(hidden_states):
    """Compute hc_expand (hc_expand) in NKI.

    Inputs, in order — `HyperConnectExpand.forward(h)`:
      hidden_states: bfloat16[1, 128, 5120]   h, the embedding output entering the stack

    Returns the group's output, both tensors of the spec's `return h, pre_mix`:
      bfloat16[1, 128, 4, 5120]   the input, one copy per hyper-connection
      float32[1, 128, 4]          the identity pre-mix: 1.0 on copy 0, 0.0 elsewhere
    """
    batch, seqlen, dim = hidden_states.shape
    rows = batch * seqlen
    assert dim % CHUNK == 0 and rows % TILE == 0

    copies_hbm = nl.ndarray((batch, seqlen, HC, dim), dtype=hidden_states.dtype,
                            buffer=nl.shared_hbm)
    mix_hbm = nl.ndarray((batch, seqlen, HC), dtype=nl.float32, buffer=nl.shared_hbm)

    # (rows, ...) views: batch and sequence are both just tokens of an independent per-token
    # copy, so flattening them keeps one tiling loop and every transfer row-major. `x3` carries
    # the length-1 copy axis the broadcast below expands, in the position the destination has
    # it.
    x3 = hidden_states.reshape((rows, 1, dim))
    copies = copies_hbm.reshape((rows, HC, dim))
    mix = mix_hbm.reshape((rows, HC))

    n_tiles = rows // TILE
    n_chunks = dim // CHUNK

    for tile_idx in nl.affine_range(n_tiles):
        row0 = tile_idx * TILE

        # pre_mix: zeros with 1.0 on copy 0. Both are compile-time constants, which is what
        # `zeros(...)` and `pre_mix[:, :, 0] = 1.0` are in the reference — nothing here is read
        # off an input.
        pm = nl.ndarray((TILE, HC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pm, 0.0)
        nisa.memset(pm[:, 0:1], 1.0)
        nisa.dma_copy(dst=mix[nl.ds(row0, TILE), 0:HC], src=pm,
                      dge_mode=nisa.dge_mode.none)

        for chunk_idx in nl.affine_range(n_chunks):
            col0 = chunk_idx * CHUNK

            # The whole of `repeat`, in one transfer per chunk: `.broadcast(1, HC)` on the
            # source's length-1 copy axis makes the DMA read each token's CHUNK bfloat16 words
            # once and write them into all HC copy slots, HBM to HBM, with no SBUF staging and
            # no engine instruction. `repeat` duplicates the words unchanged, so the bytes that
            # arrive are the recording's own.
            #
            # `dge_mode.none` asks the Neuron Runtime to build this transfer's descriptors into
            # HBM before the NEFF runs, rather than having the GpSimd Engine build them during
            # it (which is what the software DGE behind `nl.load` does, at ~0.8 us of engine
            # time per transfer). For a kernel that is nothing but transfers that is the only
            # engine time there is to remove.
            nisa.dma_copy(dst=copies[nl.ds(row0, TILE), 0:HC, nl.ds(col0, CHUNK)],
                          src=x3[nl.ds(row0, TILE), 0:1, nl.ds(col0, CHUNK)].broadcast(1, HC),
                          dge_mode=nisa.dge_mode.none)

    return copies_hbm, mix_hbm
