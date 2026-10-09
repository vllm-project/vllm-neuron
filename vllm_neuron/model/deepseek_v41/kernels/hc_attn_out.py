"""The NKI kernel for layers.0.hc_attn_out.

One partition module of hf:deepseek-ai/DeepSeek-V4.1-Flash: `layers.0.hc_attn_out`, a
sequential group over 1 submodule(s) (layers.0.hc_attn_out) — `HyperConnectSublayerOut`,
the mHC `hc_post`: expand a sublayer's output back to `hc` copies of the residual stream,
with the residual stream mixed back in.

What this computes, from the frozen spec in `reference_torch.py`
(`HyperConnectSublayerOut.forward`):

    y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), 2)
    return y.type_as(x)

with `x` bfloat16 (1, tok, dim), `residual` bfloat16 (1, tok, hc, dim), `post` float32
(1, tok, hc) and `comb` float32 (1, tok, hc, hc). Written out per element, which is what
this kernel does directly:

    y[t, j, d] = post[t, j] * x[t, d] + sum_i comb[t, i, j] * residual[t, i, d]

Two things in that line are worth stating because the recorded sample nearly hides both,
and `scratch/`-side host checks against `tensors/reference.bin` are what settled them:

* The sum runs over `comb`'s **first** index — `dim=2` of the rank-5 broadcast product is
  `comb`'s `i`, not its `j`, so the per-token 4x4 mixing matrix enters *transposed*
  relative to the obvious reading. `comb` is not symmetric here. Summing over the wrong
  index reproduces the recorded reference to 0.0469, which is *under* this module's
  0.0512501 ceiling, so the recording cannot fail it — the spec is the only thing that
  can, and against the spec the index order below is bit-exact (worst element 0.0 over
  2,621,440 elements, zero mismatching bfloat16 words).
* The sum runs over **every** copy `i`. In the recorded sample all four copies of
  `residual` are bit-identical (layer 0's residual stream comes straight from
  `hc_expand`, which replicates the embedding), so on that one input the sum collapses to
  `(sum_i comb[t, i, j]) * residual[t, 0, :]`. Dropping the `i > 0` terms moves the
  recorded output by 0.546 and moves a copy-perturbed input by 3.47, so the shortcut is
  wrong and is not taken.

Everything is per token, and there are 128 of them, so the whole kernel is one tile: tokens
on the 128 SBUF partitions, `dim` along the free axis, and every coefficient a (tok, 1)
column — exactly the operand shape `nisa.tensor_scalar` and `nisa.scalar_tensor_tensor`
broadcast along the free axis for free. There is no contraction over a long axis, so
nothing here wants the PE array: the whole computation is `hc * (hc + 1)` = 20 fused
multiply-accumulates over `dim` columns, and the accumulation happens in float32 because
`post` and `comb` are float32: the reference's dtype promotion makes every product and the
sum float32 before `.type_as(x)` rounds the result back to bfloat16.

Where the time goes
-------------------
0.1038 ms, and it is a Vector Engine instruction count: the profile shows 80
`SCALAR_TENSOR_TENSOR` at 1276 ns each = 102.1 us of Vector against a 103.8 us wall, with
11.5 us of gaps in the whole span and DMA active only 14.8% of it. So the kernel is within a
couple of percent of the floor its instruction mix allows, and the floor is the 16
accumulating Vector passes per output element (4 copies x 4 residual terms) — everything
that could leave Vector already has.

That 1276 ns is ~1 element per cycle per lane and does not move with dtype: making the
accumulator bfloat16 instead of float32 measured 0.1036 ms, i.e. nothing, while giving up
bit-identity (max_abs_err 0 -> 0.00293). Vector's cost here is elements passed over, not
bytes moved and not ALU operations — a 1-op Scalar `TENSOR_SCALAR` over the same 1024
columns is 1134 ns, only 11% less than the 2-op fused Vector instruction, which is why
fusing the multiply into the accumulate is right and splitting products out is not.

Nothing here touches act_quant/fp8_gemm/fp4_*/sparse_attn, and no `compat/*.py` rebinds
anything this module reads: the slice is plain float32/bfloat16 arithmetic.

NKI and the Python standard library only; the arithmetic happens on the Trainium device.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition dimension of SBUF: tokens are laid out across it, 128 at a time.
PMAX = 128

#: Columns of `dim` handled per pipeline stage: one `x` load, one `residual` load, the 20
#: fused multiply-accumulates over those columns, and one output store. `dim` is read and
#: written once either way, so this does not change the traffic — it decides whether the
#: transfers overlap the arithmetic or sit in front of it. Must divide `dim`.
#:
#: Measured on device (ms), all of them bit-identical to the recording:
#:   640 -> 0.1123   1024 -> 0.1038   1280 -> 0.1085   2560 -> 0.1093   5120 -> 0.1207
#: 5120 is the untiled kernel, where the 6.6 MB of loads all land before any arithmetic;
#: below 1024 the per-instruction issue cost stops being amortised. 1024 also beats 1280,
#: which divides `dim` just as evenly, so this is a width the engines like and not just a
#: chunk count.
CHUNK = 1024


@nki.jit
def kernel(input_0, input_1, input_2, input_3):
    """Compute layers.0.hc_attn_out (hc_attn_out) in NKI.

    Inputs, in order — the argument order of `HyperConnectSublayerOut.forward` at its call
    site in `compat/hyper_connections.py` (`self.hc_attn_out(x, residual, attn_post,
    attn_comb)`):
      input_0: bfloat16[1, 128, 5120]      x        — the attention sublayer's output
      input_1: bfloat16[1, 128, 4, 5120]   residual — the rank-4 stream, saved before hc_attn_in
      input_2: float32[1, 128, 4]          post     — per-token gate on x, one per copy
      input_3: float32[1, 128, 4, 4]       comb     — per-token 4x4 mix of the residual copies

    Returns the group's output: bfloat16[1, 128, 4, 5120].
    """
    batch, seqlen, dim = input_0.shape
    hc = input_2.shape[-1]
    rows = batch * seqlen
    assert dim % CHUNK == 0

    out_hbm = nl.ndarray((batch, seqlen, hc, dim), dtype=input_0.dtype,
                         buffer=nl.shared_hbm)

    # (rows, ...) views of the 3D/4D arguments: batch and sequence are both just tokens of
    # an independent per-token computation, so flattening them keeps one tiling loop.
    # `comb` flattens all the way to (rows, hc*hc), with comb[t, i, j] at column i*hc + j.
    x_hbm = input_0.reshape((rows, dim))
    res_hbm = input_1.reshape((rows, hc, dim))
    post_hbm = input_2.reshape((rows, hc))
    comb_hbm = input_3.reshape((rows, hc * hc))
    y_hbm = out_hbm.reshape((rows, hc, dim))

    n_tiles = (rows + PMAX - 1) // PMAX
    n_chunks = dim // CHUNK

    for tile_idx in nl.affine_range(n_tiles):
        row0 = tile_idx * PMAX
        nr = min(PMAX, rows - row0)

        # The coefficients are 16 and 64 bytes per partition, are reused by every chunk, and
        # gate everything behind them, so they are hoisted out of the chunk loop and stay
        # plain loads: a runtime-built descriptor lives in HBM and has to be fetched from
        # there before its transfer can start, which is the wrong trade at the head of a
        # dependency chain.
        post = nl.load(post_hbm[nl.ds(row0, nr), 0:hc])
        comb = nl.load(comb_hbm[nl.ds(row0, nr), 0:hc * hc])

        for chunk_idx in nl.affine_range(n_chunks):
            col0 = chunk_idx * CHUNK

            # The three large transfers get `dge_mode.none`: the Neuron Runtime builds their
            # DMA descriptors into HBM before the NEFF runs, instead of the GpSimd Engine
            # building them during it (which is what the software DGE behind `nl.load` does,
            # at ~0.8 us of engine time per transfer).
            xg = nl.ndarray((nr, CHUNK), dtype=input_0.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xg, src=x_hbm[nl.ds(row0, nr), nl.ds(col0, CHUNK)],
                          dge_mode=nisa.dge_mode.none)

            # All `hc` copies of this column range in one strided transfer.
            rg = nl.ndarray((nr, hc, CHUNK), dtype=input_1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rg,
                          src=res_hbm[nl.ds(row0, nr), 0:hc, nl.ds(col0, CHUNK)],
                          dge_mode=nisa.dge_mode.none)

            og = nl.ndarray((nr, hc, CHUNK), dtype=input_0.dtype, buffer=nl.sbuf)
            # One accumulator per output copy, so the four `j` chains are independent and
            # the scheduler can interleave them. (It measures the same as a single shared
            # accumulator, 0.1038 either way — the kernel is not dependency-stalled — but
            # independent chains are the cheaper thing to be wrong about.)
            acc = nl.ndarray((nr, hc, CHUNK), dtype=nl.float32, buffer=nl.sbuf)

            for j in nl.static_range(hc):
                # acc[j] = sum_i comb[t, i, j] * residual[t, i, cols] in float32, then
                # y[:, j, cols] = acc[j] + post[t, j] * x[t, cols]. That is the reference's
                # order — it sums the residual terms before adding the `post * x` term — so
                # the float32 rounding is the reference's too, and the result is bit-identical
                # to the recording.
                #
                # `i == 0` seeds the accumulator with a `tensor_scalar`, which needs no
                # accumulate and so no zeroing pass, and every later term folds its multiply
                # and its add into one `scalar_tensor_tensor`.
                #
                # Which engine issues each of these is the whole performance story, because
                # the kernel is nothing but elementwise passes. `scalar_tensor_tensor` takes
                # no `engine=` and is Vector-only; `tensor_scalar` takes one, so the four
                # seeds are pinned to the Scalar (Activation) Engine and Vector is left with
                # the 16 accumulating passes — which is the floor. Materialising `k` of the
                # five products per copy on Scalar leaves `5 - k` for a fused Vector
                # `scalar_tensor_tensor` and `k - 1` materialised tiles for a Vector
                # `tensor_tensor`, i.e. 4 Vector passes per copy however `k` is chosen, and
                # `k > 1` measured slightly worse (0.1056) because the extra materialised
                # tiles cost SBUF traffic. The adds cannot leave Vector: `tensor_tensor`'s
                # `engine=` rejects the Scalar Engine outright (`tensor_tensor engine must be
                # one of [vector, gpsimd, unknown]`), and the GpSimd Engine, which it does
                # accept, loses twice over — a 1024-column GpSimd `tensor_tensor` measured
                # 7898 ns against Vector's 1276 ns, and routing 5 of the 80 accumulates
                # through it left Vector's busy time *unchanged* because every remaining
                # Vector instruction slowed to 1357 ns. Its idle time is not parallel
                # capacity; it contends for SBUF bandwidth (0.1090 ms for 5 passes moved,
                # 0.1134 for 10, against 0.1038 for none).
                nisa.tensor_scalar(dst=acc[:, j, :], data=rg[:, 0, :],
                                   op0=nl.multiply, operand0=comb[:, j:j + 1],
                                   engine=nisa.scalar_engine)
                for i in nl.static_range(1, hc):
                    col = i * hc + j
                    nisa.scalar_tensor_tensor(dst=acc[:, j, :], data=rg[:, i, :],
                                              op0=nl.multiply,
                                              operand0=comb[:, col:col + 1],
                                              op1=nl.add, operand1=acc[:, j, :])

                # The `dst` dtype rounds the float32 accumulator to bfloat16, which is what
                # `.type_as(x)` does: the engine's internal value is float32, so this is the
                # same round-to-nearest.
                nisa.scalar_tensor_tensor(dst=og[:, j, :], data=xg,
                                          op0=nl.multiply, operand0=post[:, j:j + 1],
                                          op1=nl.add, operand1=acc[:, j, :])

            nisa.dma_copy(dst=y_hbm[nl.ds(row0, nr), 0:hc, nl.ds(col0, CHUNK)], src=og,
                          dge_mode=nisa.dge_mode.none)

    return out_hbm
