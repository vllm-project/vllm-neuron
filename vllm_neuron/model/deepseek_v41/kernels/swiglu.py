"""The experts' clamped SwiGLU — and the routing weight — as one NKI kernel.

**Not currently wired, and the reason is now a settled measurement rather than
a suspicion.** This kernel was wired into `model._swiglu_expert`, passed the
gate's verifier at four ranks (`/tmp/iter13/v13b.json`, 4/4 `REASONABLE`), and
was taken back out because it moves tokens: it differs from the torch
expression below by one bf16 step on about 1 value in 100,000, and over 280
expert calls a forward that flipped a near-tied EOS on the verifier's needle
case and broke the token-for-token agreement between speculation on and off
that `uses-dspark-speculation` is judged on (2 of 3 short cases still matched;
`smoke-test` diverged after 80 characters).

Two sources were found and only one is fixable:

1. *multiply order* — `nisa.scalar_tensor_tensor` is `(data op0 scalar) op1
   tensor`, so the one-instruction form computes `(silu * w) * u` where torch
   computes `w * (silu * u)`, and fp32 multiplication is not associative. This
   is **fixed** below, in two instructions, and it makes the 1-row and 6-row
   cases bitwise identical to torch.
2. *`silu` itself, which cannot be fixed* — two equally valid fp32
   formulations of `x * sigmoid(x)` disagree in the last fp32 bit. Measured
   directly against torch, with no kernel involved: `F.silu(x)` and
   `x * torch.sigmoid(x)` differ on **22% of 2,000,000 fp32 values**, and after
   a bf16 cast on **4.5e-06** of them. The kernel's residual device mismatch
   rate against torch is 2.5e-05 (29 of 1,179,648 at 512 rows, 0 at 6 and 1
   rows) — the same order. So **no SwiGLU kernel can be bit-identical to
   `F.silu` short of reproducing torch's internal formulation**, and the choice
   is not "fix the kernel" but "decide whether a 1e-05 perturbation is
   acceptable".

Which reframes the problem: the fragile thing is not this kernel but the fact
that speculation's token-for-token identity is *empirical* — a verify step
reduces over `1 + k` rows where a plain step reduces over one, and nothing
guarantees they agree bit for bit, so any value perturbation anywhere re-rolls
it. `.autohelix/notes/iter-13.md` §3 has the full argument and what to do about
it.


`model._swiglu_expert` runs an expert's FFN as three GEMMs (`w1`, `w3`, then
`w2`, all `kernels/blockquant`) with this between them::

    gate = nki_linear(x, w1, ...).float()
    up   = nki_linear(x, w3, ...).float()
    up   = torch.clamp(up, min=-limit, max=limit)
    gate = torch.clamp(gate, max=limit)
    y    = F.silu(gate) * up
    y    = weights * y                 # routing weight, routed experts only
    return nki_linear(y.to(dtype), w2, ...)

The middle five lines were the last arithmetic inside an expert expressed as a
torch subgraph. They are elementwise over `[rows, moe_intermediate_size]` =
`[rows, 2304]`, and they run 7 times a layer (6 routed experts plus the shared
one) on 43 layers, so they are the most frequently executed arithmetic in the
model after the GEMMs themselves.

Derived from `legacy/moe_short.py`'s `_swiglu` (lines 158-169), which is the
same four instructions for the routed experts of `layers.1.ffn`::

    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.scalar_tensor_tensor(dst=dst, data=sg, op0=nl.multiply, operand0=wcol,
                              op1=nl.multiply, operand1=u)

Five things change:

0. the two multiplies are reassociated to torch's order — see point 1 of the
   note above, and the comment on the two instructions that do it.
1. the clamps are staged in fp32 rather than in bf16 — `_swiglu8` in the same
   legacy file (the shared expert's variant, line 642) does it this way and
   `_swiglu` does not, and `_swiglu_expert` widens before clamping too. **The
   two orders are exactly equal here and that was checked rather than assumed**
   (`/tmp/iter13/ktest_swiglu.py`: the bf16-staged form is bitwise identical to
   the torch reference on every shape tested): a bf16 operand clamped against a
   bf16-representable limit is unchanged by widening first. So this is a
   dtype-staging choice, not a correction — stated because the obvious
   assumption that it *is* a correction is wrong.
2. `limit` is an operand, not the module constant `SWIGLU_LIMIT = 10.0`, so one
   NEFF serves the routed experts and the shared expert even if a config gives
   them different limits (this checkpoint gives both `swiglu_limit = 10.0`).
3. the row count is ragged rather than a baked 128, and the free axis is
   chunked by a divisor of the operand's width rather than fixed.
4. the routing weight is an operand column for a routed expert and a column of
   ones for the shared expert, rather than two separate kernels. `1.0 * y` is
   exact in fp32, so the shared expert's result is unchanged by going through
   the same instruction.

Interface
---------
    gate    [rows, inter]  bf16 or fp32   `w1 @ x`, the gate branch
    up      [rows, inter]  same           `w3 @ x`, the up branch
    weights [rows, 1]      fp32           the routing weight, or ones
    limit   [1, 1]         fp32           `swiglu_limit`
    ->      [rows, inter]  gate's dtype   `weights * silu(min(gate, limit))
                                           * clamp(up, -limit, limit)`

The interior is fp32 and only the store narrows, which is the reference's
single rounding on the way into the `w2` GEMM.

Row-count invariance holds trivially and for the same reason it matters in
`rmsnorm.py`: every value here depends on its own row and column only — there
is no reduction — so a row's output does not depend on how many rows the call
carried, and a speculative verify step's `1 + k` rows give bit-identical
results to the `1`-row plain step.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width; one row per partition, 128 rows per tile.
PT = 128


def _chunk(width):
    """Free-axis chunk: a divisor of `width` small enough that four fp32 copies
    of it are a few KiB per partition. `rmsnorm._chunk`'s rule, widened at the
    bottom because `moe_intermediate_size = 2304` has no divisor above 256 in
    that list.
    """
    for cand in (2048, 1024, 512, 256, 128, 64, 32):
        if width % cand == 0:
            return cand
    return width


@nki.jit
def swiglu(gate, up, weights, limit):
    """The clamped SwiGLU with its routing weight. See the module docstring."""
    rows, inter = gate.shape
    assert up.shape == (rows, inter), f"up {up.shape} wants ({rows}, {inter})"
    assert weights.shape == (rows, 1), (
        f"weights {weights.shape} wants ({rows}, 1) — one routing weight a row"
    )
    assert limit.shape == (1, 1), f"limit {limit.shape} wants (1, 1)"

    ch = _chunk(inter)
    nchunks = inter // ch

    out = nl.ndarray((rows, inter), dtype=gate.dtype, buffer=nl.shared_hbm)

    def body(r0, nr):
        # A freshly loaded `(1, n)` tile broadcasts; a slice of a staged one
        # does not — see `hc_mixes`' module docstring. Both of these are
        # per-partition scalars, which is what `tensor_scalar`'s operands and
        # `scalar_tensor_tensor`'s `operand0` want.
        lim = nl.broadcast_to(nl.load(limit[0:1, 0:1], dtype=nl.float32), (nr, 1))
        nlim = nl.multiply(lim, -1.0)
        w = nl.load(weights[r0 : r0 + nr, 0:1], dtype=nl.float32)

        for j in range(nchunks):
            c0 = j * ch
            # `min(gate, limit)` — the gate branch is clamped from above only,
            # which is the training-time asymmetry `swiglu_limit` encodes.
            g = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=g,
                data=nl.load(gate[r0 : r0 + nr, c0 : c0 + ch], dtype=nl.float32),
                op0=nl.minimum,
                operand0=lim,
            )
            # `clamp(up, -limit, limit)` — both sides, in one instruction.
            u = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=u,
                data=nl.load(up[r0 : r0 + nr, c0 : c0 + ch], dtype=nl.float32),
                op0=nl.maximum,
                operand0=nlim,
                op1=nl.minimum,
                operand1=lim,
            )
            sg = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.silu, data=g)
            # `weight * (silu(gate) * up)`, in **two** instructions, which is
            # the reference's association and not `legacy/moe_short._swiglu`'s.
            # `nisa.scalar_tensor_tensor` computes `(data op0 scalar) op1
            # tensor`, so the one-instruction form is `(silu * weight) * up`;
            # fp32 multiplication is not associative and that cost one bf16 step
            # on about 1 value in 100,000, which is enough to move a token (see
            # the module docstring). The routing weight multiplies *before*
            # `w2`, which is what makes a zero routing weight contribute exactly
            # zero (`w2` has no bias) and is why the pad rows of a routed
            # expert's row list need no mask.
            y = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y, data1=sg, op=nl.multiply, data2=u)
            nisa.tensor_scalar(dst=y, data=y, op0=nl.multiply, operand0=w)
            nl.store(
                out[r0 : r0 + nr, c0 : c0 + ch], value=nl.copy(y, dtype=gate.dtype)
            )

    # The ragged-row pattern every kernel in this directory uses: full tiles in
    # `affine_range`, then one more call with a Python row count for the tail.
    full, tail = rows // PT, rows % PT
    for t in nl.affine_range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return out
