"""`RMSNorm` as one NKI kernel, with a **row-count-independent** reduction.

`model.RMSNorm.forward` was four torch ops — `x.float()`, `square().mean(-1)`,
`rsqrt`, and the weighted multiply — and the mean is a reduction over `dim`
fp32 values *per row*. That is the one shape in this model where the compiler
is free to pick a different accumulation tree for a 1-row tensor than for a
6-row or 512-row one, and a different tree is a different last bit in the
reciprocal square root, which an argmax over 129,280 near-tied logits can turn
into a different token.

So this kernel exists for a numerical reason rather than a performance one:
**the reduction blocking here is a function of `dim` alone.** Each row lives in
its own partition, its `dim` values are reduced in `dim // _chunk(dim)` chunks
in index order, and the chunk width does not depend on how many rows the step
carried. One row of a 1-row call and the same row of a 6-row call therefore
accumulate the same values in the same order and come out bit-identical, which
is what makes speculative verification (`1 + k` rows where a plain decode step
has 1) exactness-preserving rather than approximately so.

The structure is `engram_gate.gated_add`'s, including its `_rsqrt` Newton
correction — `nl.rsqrt` alone is a hardware approximation and the reference is
a torch `rsqrt`.

Interface
---------
    x      [rows, dim]  bf16 or fp32   the rows to normalize
    weight [1, dim]      any           the learned gain
    eps    [1, 1]        fp32          `rms_norm_eps`, as a tensor because the
                                       kernel entry point takes tensors only
    ->     [rows, dim]   x's dtype     `x * rsqrt(mean(x^2) + eps) * weight`

Both reductions run in fp32 whatever `x`'s storage is, which is the reference's
fp32 interior: `rms_norm_eps` is 1e-20 for this model, small enough that a bf16
variance would make the epsilon a no-op.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width; one row per partition, 128 rows per tile.
PT = 128


def _chunk(dim):
    """Free-axis chunk for the reduction: a divisor of `dim`, small enough that
    one fp32 copy is a few KiB per partition. `engram_gate._chunk`'s rule, and
    the reason this kernel is row-count invariant — it reads `dim` only.
    """
    for cand in (2048, 1024, 512, 256, 128):
        if dim % cand == 0:
            return cand
    return dim


def _rsqrt(m):
    """`1 / sqrt(m)` with one Newton step: `y * (1.5 - 0.5 * m * y^2)`."""
    y = nl.rsqrt(m)
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    return nl.multiply(y, corr)


@nki.jit
def rms_norm(x, weight, eps):
    """`x * rsqrt(mean(x^2, -1) + eps) * weight`. See the module docstring."""
    rows, dim = x.shape
    assert weight.shape == (1, dim), f"weight {weight.shape} wants (1, {dim})"
    assert eps.shape == (1, 1), f"eps {eps.shape} wants (1, 1)"

    ch = _chunk(dim)
    nchunks = dim // ch
    inv_dim = 1.0 / dim

    out = nl.ndarray((rows, dim), dtype=x.dtype, buffer=nl.shared_hbm)

    def body(r0, nr):
        # The rows in fp32, kept because the weighted multiply below needs them
        # again once the variance — which depends on every chunk — is known. At
        # `dim = 5120` that is 20 KiB per partition.
        xf = nl.ndarray((nr, dim), dtype=nl.float32, buffer=nl.sbuf)
        acc = None
        for j in range(nchunks):
            xj = nl.load(x[r0 : r0 + nr, j * ch : (j + 1) * ch], dtype=nl.float32)
            nisa.tensor_copy(dst=xf[0:nr, j * ch : (j + 1) * ch], src=xj)
            sq = nl.sum(nl.square(xj), axis=1, keepdims=True)
            acc = sq if acc is None else nl.add(acc, sq)

        rstd = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd[...] = _rsqrt(nl.add(nl.multiply(acc, inv_dim), nl.broadcast_to(
            nl.load(eps[0:1, 0:1], dtype=nl.float32), (nr, 1)
        )))

        for j in range(nchunks):
            # A freshly loaded `(1, n)` tile broadcasts; a slice of a staged one
            # does not — see `hc_mixes`' module docstring.
            wj = nl.broadcast_to(
                nl.load(weight[0:1, j * ch : (j + 1) * ch], dtype=nl.float32),
                (nr, ch),
            )
            y = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
            # `x * rstd` rides the activation's per-partition scale, which is
            # exactly what the per-row reciprocal is.
            nisa.activation(
                dst=y, op=nl.copy, data=xf[0:nr, j * ch : (j + 1) * ch], scale=rstd
            )
            nisa.tensor_tensor(dst=y, data1=y, op=nl.multiply, data2=wj)
            nl.store(
                out[r0 : r0 + nr, j * ch : (j + 1) * ch],
                value=nl.copy(y, dtype=x.dtype),
            )

    # The ragged-row pattern every kernel in this directory uses: full tiles in
    # `affine_range`, then one more call with a Python row count for the tail.
    # Each row is its own partition either way, so the tail call's arithmetic is
    # the full tile's arithmetic — which is the invariance this kernel is for.
    full, tail = rows // PT, rows % PT
    for t in nl.affine_range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return out
