"""The NKI kernel for hc_collapse.

One partition module of hf:deepseek-ai/DeepSeek-V4.1-Flash: `hc_collapse`, a sequential
group over 1 submodule(s) (hc_collapse) — `HyperConnectCollapse`, the tail of the mHC
rank-4 residual stream: the point where the four hyper-connection copies are mixed back
down into the single hidden state that the output norm reads.

What this computes, from the frozen spec in `reference_torch.py`
(`HyperConnectCollapse.forward`), whose body is two lines:

    y = sum over dim 2 of ( pre_mix.unsqueeze(-1) * x.float() )
    return y.to(x.dtype)

so, per element, with `x` bfloat16 (1, tok, hc, dim) and `pre_mix` float32 (1, tok, hc):

    y[0, t, d] = bfloat16( sum over j of  pre_mix[0, t, j] * float32(x[0, t, j, d]) )

One weighted sum over the copy axis, the weights one float32 per (token, copy), the
products and the accumulation in float32, the single rounding to bfloat16 at the very end.
There is no weight tensor, no norm, no quantization and no masking in this slice —
`pre_mix` arrives as an argument because the sublayer upstream of this one produced it, and
`compat/hyper_connections.py` (which `reference_torch.py` is copied from verbatim) rebinds
nothing this module touches. Nothing here goes near act_quant / fp8_gemm / fp4_* /
sparse_attn.

`hc` and `dim` are read off `input_0.shape`, so no constant from `config.json` is needed;
`hc_mult = 4` there is the cross-check, not the source.

Shape of the computation on the device
-------------------------------------
Arithmetic intensity is as low as it gets: 5 MiB in, 1.25 MiB out, and two float32
operations per input element. So this is a streaming kernel and the only thing that matters
is keeping the DMAs busy and spending exactly one engine instruction per loaded tile.

All four copies of one column chunk arrive in **one** DMA. `x` viewed as (tok, hc, dim) and
sliced `x3[t0:t0+128, 0:hc, f0:f0+CHUNK]` is a two-level strided access pattern — CHUNK
contiguous bfloat16 words, `hc` of them `dim` apart, per token — which lands in an SBUF tile
shaped (128, hc, CHUNK) whose `[:, j, :]` is copy `j`. That is 5 transfers for the whole
kernel instead of 20 and it is the largest win measured here: 0.0302 -> 0.0260 ms, with the
profile's `dma_active_time` falling 22.1 -> 15.7 us. The bytes moved are identical; only the
descriptor count changes.

Out of that one tile the four copies fold into one float32 accumulator in SBUF:

    j == 0:  acc = x_j * pre_mix[:, j]            (one Vector `tensor_scalar`)
    j  > 0:  acc = x_j * pre_mix[:, j] + acc      (one Vector `scalar_tensor_tensor`)

`pre_mix[:, j]` is one value per token, i.e. a *per-partition scalar*, which is the operand
shape the engines multiply a tile by for free — no broadcast instruction and no second tile.
The accumulator is float32 and the store's destination is bfloat16, so the mandatory
rounding is the **DMA's**: a converting `dma_copy` costs no compute instruction and is bit
for bit what a Vector Engine cast into a staging tile produces (both measured, both 0.0
error, and the two spellings within run-to-run noise of each other).

The result is bit-identical to the recording (max_abs_err exactly 0.0 over 655,360
elements): the spec's one rounding step is the only one this kernel has, and it does the
four products in the same order (`j` ascending) that the spec's sum over dim 2 does.

What was measured and rejected, all at CHUNK 1024 (see `.autohelix/notes/iter-1.md`):

| variant                                                     | ms     | error   |
|-------------------------------------------------------------|--------|---------|
| one strided DMA per chunk, converting store  **(this one)** | 0.0260 | 0.0     |
| ... with the last accumulate writing a bfloat16 tile        | 0.0261 | 0.0     |
| ... with the j == 0 product moved to the Scalar Engine      | 0.0265 | 0.0     |
| one DMA per copy (4x the transfers)                         | 0.0300 | 0.0     |
| `dge_mode.hwdge` on the big load                            | 0.0293 | 0.0     |
| `nl.load` / `nl.store` instead of `nisa.dma_copy`           | 0.0297 | 0.0     |
| two accumulators, one on Vector and one on GpSimd           | 0.1083 | 0.03125 |

Splitting the accumulation across engines is the one badly wrong idea: the GpSimd Engine
takes 94.8 us for half the elements the Vector Engine does in 21 us, so any float32
element-wise work handed to it dominates everything. It also re-associates the sum, which is
why those rows are not bit-exact (still far inside the bar, but a loss for nothing).

NKI and the Python standard library only; the arithmetic happens on the Trainium device.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Tokens per tile. 128 is SBUF's partition count, and `tok` is 128 here, so the token loop
#: runs once; it is still written as a loop so a longer sample needs no change.
PMAX = 128

#: Columns of `dim` folded per pass: one DMA for all `hc` copies, one Vector instruction per
#: copy, and one converting store, all over the same CHUNK columns. Must divide `dim`.
#:
#: This is the DMA's transfer width, and the memory notes from `hc_expand` (same model, same
#: `dim`) measured that width wanting to be a *power of two* — every non-power-of-two width
#: there was ~2x worse whatever its size. `dim` is 5120 = 2^10 * 5, so 1024 is the largest
#: power of two available, and it is also what the `hc_attn_in` kernel's own sweep settled on
#: for its pass over the same tensor.
#:
#: Swept again here on top of the one-DMA-per-chunk load, and 1024 is the optimum by a clear
#: margin: 512 -> 0.0285 ms, **1024 -> 0.0260**, 2560 -> 0.0335, 5120 (one pass, no chunking)
#: -> 0.0353. Too small and the descriptor count costs; too large and there are not enough
#: chunks left to overlap one chunk's load with the previous chunk's arithmetic (at 5120 there
#: is exactly one, and nothing overlaps at all).
CHUNK = 1024


@nki.jit
def kernel(input_0, input_1):
    """Compute hc_collapse (hc_collapse) in NKI.

    Inputs, in order — `HyperConnectCollapse.forward(x, pre_mix)`:
      input_0: bfloat16[1, 128, 4, 5120]   `x`, the rank-4 residual stream
      input_1: float32[1, 128, 4]          `pre_mix`, one mixing weight per token per copy

    Returns the group's output: bfloat16[1, 128, 5120].
    """
    batch, n_tok, hc, dim = input_0.shape
    rows = batch * n_tok

    assert dim % CHUNK == 0, "CHUNK must divide dim"
    assert rows % PMAX == 0, "PMAX must divide batch*tokens"

    y_out = nl.ndarray((batch, n_tok, dim), dtype=input_0.dtype, buffer=nl.shared_hbm)

    # Flat (token, ...) views: batch and sequence are both just tokens of an independent
    # per-token reduction, so flattening them keeps one tiling loop. `x3[t, j, f]` is
    # `x[0, t, j, f]`, with the copy axis kept so one access pattern can span all of it.
    x3 = input_0.reshape((rows, hc, dim))
    pm2 = input_1.reshape((rows, hc))
    y2 = y_out.reshape((rows, dim))

    for t in nl.affine_range(rows // PMAX):
        t0 = t * PMAX

        # The mixing weights for this token tile: (128, 4) float32, 2 KiB, and the first
        # thing every instruction below waits on. A plain `nl.load` rather than a
        # `dma_copy(dge_mode=none)` — the `hc_attn_in` notes measured runtime-built
        # descriptors *losing* on exactly this transfer (a pre-built descriptor has to be
        # fetched from HBM before its transfer can start, which is the wrong trade at the
        # head of a dependency chain), while winning on the big streaming ones below.
        pm = nl.load(pm2[t0:t0 + PMAX, :])

        for c in range(dim // CHUNK):
            f0 = c * CHUNK

            # One float32 row of CHUNK columns per token, plus the bfloat16 tile holding
            # every copy of those columns. Both are declared per chunk so the scheduler is
            # free to overlap one chunk's load with the previous chunk's arithmetic instead
            # of serialising on a single reused buffer.
            acc = nl.ndarray((PMAX, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            xg = nl.ndarray((PMAX, hc, CHUNK), dtype=input_0.dtype, buffer=nl.sbuf)

            # All `hc` copies of this column chunk in one transfer: the source access
            # pattern is CHUNK contiguous words, `hc` of them `dim` apart, per token, and
            # the destination tile's `[:, j, :]` is copy `j`. One DMA instead of `hc`, for
            # the same bytes (0.0300 -> 0.0260 ms; `dma_active_time` 22.1 -> 15.7 us).
            #
            # `dge_mode.none` asks the Neuron Runtime to build this transfer's descriptors
            # into HBM before the NEFF runs, instead of having the GpSimd Engine build them
            # during it (which is what the software DGE behind `nl.load` does, at ~0.8 us of
            # engine time per transfer). This kernel is nothing but transfers plus one
            # instruction each, so that bookkeeping is the only engine time there is to
            # remove; `hwdge` and plain `nl.load` both measured slower here (0.0293, 0.0297).
            nisa.dma_copy(dst=xg, src=x3[t0:t0 + PMAX, 0:hc, f0:f0 + CHUNK],
                          dge_mode=nisa.dge_mode.none)

            for j in range(hc):
                # acc = sum_j pre_mix[:, j] * x[:, j, f0:f0+CHUNK], one fused Vector
                # instruction per copy: the products and the running sum are float32, as
                # `x.float()` and the spec's sum are, and `pre_mix[:, j:j+1]` is a
                # per-partition scalar so no broadcast is materialised. `j` ascends, so
                # copy 0 seeds the accumulator before any later copy adds onto it and the
                # summation order is the reference's own.
                #
                # These are left exactly as written, and the profile says why. The Vector
                # Engine is this kernel's busiest (19.2 us of a 26 us kernel) so handing a
                # copy to another engine looks free, but there is nothing left to hand over:
                # the `tensor_scalar` below is already compiled to an `ACTIVATE` on the
                # **Scalar** Engine (5 instructions, 6.2 us), leaving Vector with only the
                # three `SCALAR_TENSOR_TENSOR`s (15 instructions, 19.2 us). Spelling the
                # `j == 0` product as an explicit `nisa.activation(op=nl.copy,
                # scale=pre_mix[:, 0:1])` therefore measures 0.0265 against 0.0260 — it
                # moves work that had already moved. Three Vector two-tensor ops per chunk is
                # the floor for this dataflow: an accumulate can only run on Vector or
                # GpSimd (`scalar_tensor_tensor` takes no `engine=`, and `activation`'s
                # `bias` is a per-partition scalar, not a tile), and GpSimd needs 94.8 us for
                # two of them against Vector's 21 us for three.
                if j == 0:
                    nisa.tensor_scalar(dst=acc, data=xg[:, j, :], op0=nl.multiply,
                                       operand0=pm[:, j:j + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,
                                              operand0=pm[:, j:j + 1],
                                              op1=nl.add, operand1=acc)

            # `y.to(x.dtype)`: `acc` is float32 and `y2` is bfloat16, so this one converting
            # DMA is the spec's single rounding step, with no compute instruction spent on
            # it.
            nisa.dma_copy(dst=y2[t0:t0 + PMAX, f0:f0 + CHUNK], src=acc,
                          dge_mode=nisa.dge_mode.none)

    return y_out
