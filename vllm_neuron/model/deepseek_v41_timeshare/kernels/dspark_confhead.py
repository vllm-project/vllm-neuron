"""The NKI kernel for mtp.2.confidence_head.

One partition module of hf:deepseek-ai/DeepSeek-V4.1-Flash: `mtp.2.confidence_head`, a
sequential group over 1 submodule(s) (mtp.2.confidence_head).

What this computes, from the frozen reference implementation beside this file
(`DSparkConfidenceHead.forward`), in words:

    hidden = cat([hidden, markov_embed], dim=-1)   # (1, 5, 5120) + (1, 5, 256)
    return self.proj(hidden.float()).squeeze(-1)   # -> float32 (1, 5)

`self.proj` is `Linear(5376, 1, dtype=float32)`, so `bias is None` (the constructor only
creates `self.bias` when `bias=True`, and the head does not pass it) and
`register_parameter("scale", None)` runs -- there is no quantization scale. `linear()` in
`vendor/kernel.py` dispatches on the *weight* dtype: the recorded weight is a plain
bfloat16 `(1, 5376)` tensor, neither `float4_e2m1fn_x2` nor `float8_e4m3fn`, so the branch
taken is the last one, the plain unquantized `linear(x, weight)`. No `act_quant`, no
`fp8_gemm`, no `fp4_gemm`, and nothing in `compat/` rebinds `linear`/`Linear`, so
`vendor/kernel.py` governs unmodified. There is no quantization and no masking on this path.

So the whole module is five independent 5376-term dot products:

    out[0, t] = sum_k  cat(input_0, input_1)[0, t, k] * weight[0, k]

accumulated in float32. The recorded activations and the recorded weight both hold
bfloat16 values, and the product of two bfloat16 numbers is *exact* in float32 (two 8-bit
mantissas multiply into 16 bits), so accumulating those products in float32 reproduces the
reference's float32 parameter exactly; the only numerical freedom left is the order of the
5376-term sum. Checked on the host before writing any NKI: the float32
`cat -> linear -> squeeze` path reproduces `tensors/reference.bin` to 3.6e-07, against a
0.0368176 ceiling.

Shape of the computation on the device
-------------------------------------
This is the smallest kernel in the partition -- 26,880 multiply-accumulates in total, every
tensor resident in SBUF at once, nothing to tile or stream -- so it is bounded by *fixed
per-instruction and per-DMA overheads*, not by arithmetic or bandwidth. That is what the
shape below is built around: it uses as few instructions and as few DMA descriptors as the
layout allows, and it hands the arithmetic to the one engine that does a whole 128-deep
contraction in a single instruction.

The contraction runs down the **partition** axis, cut into 42 slices of 128, and each slice
is one `nisa.nc_matmul`: the weight slice is the (128, 1) stationary operand, the five
tokens' activations are the (128, 5) moving operand, and the 42 results accumulate in one
float32 PSUM bank. `dst = stationary.T @ moving` is (1, 5) -- the module's output shape,
directly. No product tile, no reduction, and no transpose.

Two details make the loads cheap, which is where the time actually goes:

* **The `cat` is never materialised.** The two halves of the concatenation are contracted
  separately and accumulate into the same PSUM bank, which is the same arithmetic. That
  matters because it lets each input keep its own k-slicing: a partition only has to see the
  *same* k on the activation side and the weight side, while different partitions -- and
  different halves -- may cover k in any order, since the dot product is a sum over all of
  k. So `input_0` is sliced as k = q * 40 + i and `input_1` as k = 5120 + q * 2 + i, and
  between them the 128 partitions cover all 5376 values of k exactly once.
* **128 is the only partition count that works.** The split needs `d_hidden == P * C0` and
  `d_embed == P * C1` with the same `P`, so `P` must divide gcd(5120, 256) = 256; the
  largest such `P` that is a legal partition count is 128, giving C0 = 40 and C1 = 2. This
  is asserted below rather than assumed.

Both activation tiles are fetched by **one DMA each** with an explicit access pattern that
folds the token axis in, under `dge_mode.none` so the runtime builds the descriptors ahead
of execution instead of GpSimd building them during it. That one change is the single
biggest win in this kernel: the obvious spelling -- a Python loop doing
`nl.load(input_0.reshape((n_tok, 128, 40))[t])` per token, which is a plain contiguous
sub-tensor load and looks strictly simpler -- ran 13.0 us against this version's 7.1 us,
with 10.5 us of GpSimd time doing software descriptor generation for ten small strided DMAs.

The weight is the one tensor with no such pattern, because 5376 is not 128 * 40 and the two
halves start at different offsets. It is spelled as a slice of the flattened HBM tensor
reshaped into place, which `nl.load` accepts and turns into a static DMA. It is 10 KiB, so
it does not matter either way.

Accuracy: this is also the most accurate of the arrangements measured, at **4.17e-07**
against the 0.0368176 ceiling -- 88,000x under. The Tensor Engine multiplies the bfloat16
operands exactly and PSUM accumulates in float32, and summing 42 slices of 128 is a
shallower reduction tree than one 5376-term sequential sum: the token-on-partition layout
this replaced measured 1.28e-06. So the fast arrangement gave up nothing; it is strictly
better on both axes.

Measured on trn2, `total_exec_time` from neuron-explorer, all bit-identical in verdict:

  token on partitions, Vector multiply + reduce        25.3 us   err 1.28e-06
  k on partitions, per-token loads, Vector + transpose 13.0 us   err 5.07e-07
  k on partitions, batched static-DGE loads            7.1 us    err 5.07e-07
  ... + cross-partition sum as one matmul vs ones      6.4 us    err 4.47e-07
  this kernel: whole contraction on the PE array       5.4 us    err 4.17e-07

NKI and the standard library only. Reference:
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
"""

import nki
import nki.language as nl
import nki.isa as nisa

#: SBUF partitions, and so the depth of each of the 42 accumulating matmuls. Not a tuning
#: knob: see the module docstring -- it is the only value that divides both 5120 and 256
#: and is a legal partition count, and it is also the PE array's contraction depth.
PMAX = 128


@nki.jit
def kernel(input_0, input_1, mtp_2_confidence_head_proj_weight):
    """Compute mtp.2.confidence_head (confidence_head) in NKI.

    Inputs, in order:
      input_0: bfloat16[1, 5, 5120]    the hidden state
      input_1: bfloat16[1, 5, 256]     the markov head's embedding
      mtp_2_confidence_head_proj_weight: bfloat16[1, 5376]   out_features == 1, no bias

    Returns the group's output: float32[1, 5].
    """
    weight = mtp_2_confidence_head_proj_weight

    n_tok = input_0.shape[1]
    d_hidden = input_0.shape[2]
    d_embed = input_1.shape[2]
    out_features, d_cat = weight.shape

    # The two halves of the concatenation have to account for the whole weight row, or part
    # of the dot product would silently be dropped.
    assert d_hidden + d_embed == d_cat, (
        f"cat({d_hidden}, {d_embed}) != weight in_features {d_cat}"
    )
    # `squeeze(-1)` collapses out_features, so the module only makes sense with one output.
    assert out_features == 1, f"expected out_features == 1, got {out_features}"
    # Every k must land on exactly one partition. If PMAX did not divide both halves evenly
    # the trailing elements would fall outside the slicing and be dropped from the sum --
    # a silent wrong answer, not a compile error.
    assert d_hidden % PMAX == 0 and d_embed % PMAX == 0, (
        f"PMAX ({PMAX}) must divide both {d_hidden} and {d_embed}"
    )

    #: Contraction elements per partition, from each of the two inputs, and their total --
    #: i.e. the number of 128-deep matmuls the contraction becomes.
    c_hidden = d_hidden // PMAX     # 40
    c_embed = d_embed // PMAX       # 2
    c_total = c_hidden + c_embed    # 42

    out = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.shared_hbm)

    # --- the activations, contraction on the partition axis -----------------------------
    # xt[q, t, i] is the activation for token t at k = q * C + i (C = c_hidden for the
    # first half, c_embed for the second). Both halves sit side by side on the free axis so
    # that a single `i` index walks the whole 42-deep slice list below.
    #
    # `pattern` is [[stride, num], ...] in elements with the partition axis first, so this
    # reads partition q's C contiguous values for each of the n_tok tokens: 128 * 5
    # descriptors of C * 2 bytes, in one instruction per input. `dge_mode.none` has the
    # Neuron Runtime materialise those descriptors into HBM before the NEFF runs; the
    # default lets GpSimd generate them during execution, which at this size costs more
    # than all the arithmetic put together (see the module docstring).
    #
    # bfloat16 is kept: the recorded values already are bfloat16, so `.float()` would change
    # no bit, and the Tensor Engine multiplies bfloat16 operands exactly into float32 PSUM.
    xt = nl.ndarray((PMAX, n_tok, c_total), dtype=input_0.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=xt[:, :, 0:c_hidden],
        src=input_0.ap(
            pattern=[[c_hidden, PMAX], [d_hidden, n_tok], [1, c_hidden]], offset=0
        ),
        dge_mode=nisa.dge_mode.none,
    )
    nisa.dma_copy(
        dst=xt[:, :, c_hidden:c_total],
        src=input_1.ap(
            pattern=[[c_embed, PMAX], [d_embed, n_tok], [1, c_embed]], offset=0
        ),
        dge_mode=nisa.dge_mode.none,
    )

    # --- the weight, in the same k-slicing ---------------------------------------------
    # Neither half is a reshape of the whole tensor (5376 != PMAX * c_hidden, and the second
    # half starts at element 5120), so each is spelled as a slice of the flattened HBM
    # tensor reshaped into the (PMAX, C) layout its activations already have. `nl.load`
    # accepts that and emits a static DMA.
    w_flat = weight.reshape((d_cat,))
    wt = nl.ndarray((PMAX, c_total), dtype=weight.dtype, buffer=nl.sbuf)
    wt[:, 0:c_hidden] = nl.load(w_flat[0:d_hidden].reshape((PMAX, c_hidden)))
    wt[:, c_hidden:c_total] = nl.load(w_flat[d_hidden:d_cat].reshape((PMAX, c_embed)))

    # --- the contraction, on the PE array ----------------------------------------------
    # Slice i contracts 128 values of k in one instruction: `dst = stationary.T @ moving`
    # with stationary (PMAX, 1) the weight slice and moving (PMAX, n_tok) the tokens'
    # activations for that slice, giving (1, n_tok). `accumulate` keeps slice i's result in
    # the same PSUM bank as the previous ones, so all 42 partial sums add in float32
    # hardware accumulation -- exactly, and with no instruction of their own. The first
    # slice must *not* accumulate, or it would add to whatever the bank last held.
    acc = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.psum)
    for i in range(c_total):
        nisa.nc_matmul(dst=acc, stationary=wt[:, i:i + 1], moving=xt[:, :, i],
                       accumulate=(i > 0))

    # `nl.store` will not read PSUM, so the result is copied to SBUF first. `acc` is already
    # (1, n_tok) -- the `.squeeze(-1)` of the specification is implicit in the stationary
    # operand being one column wide -- so nothing needs reshaping on the way out.
    res = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.sbuf)
    res[...] = nl.copy(acc)
    nl.store(out, value=res)

    return out
