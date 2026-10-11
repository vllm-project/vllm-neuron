"""The indexer's score — per-head `q . k`, rectified, and combined by the
learned per-head weights — as one NKI kernel.

This is the arithmetic behind `model.Indexer.forward`, which is the model's
*second* attention: 32 heads of 128 dims score every query against one shared
key per compressed position, the per-head scores are rectified and then summed
with the weights `weights_proj` produces. Its output decides which compressed
positions `sparse_attn` is allowed to read, so it runs on all eight
`index_source_layer_ids` of every forward pass.

Why it is a kernel rather than an einsum
--------------------------------------------------------------------------
The torch form was

    index_score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
    index_score = (index_score.relu() * weights.float().unsqueeze(-1)).sum(2)

and the intermediate the first line names is `[s, heads, n]` fp32 — at the
verifier's 9,328-token prompt on a ratio-1 index layer (`n = 9,344`) that is
**11.2 GB per layer**, for a value whose only use is to be rectified and
reduced away on the next line. Fusing the three steps keeps the live set at one
`[heads, ntile]` psum tile and makes the only `[s, n]` tensor the result itself.

Measured, so that the next reader does not have to: fusing it did **not** lower
HBM use (`.autohelix/notes/iter-5.md` §4 has the two memory tables), which means
neuronx-cc was already fusing the einsum into the reduce that follows it. The
reason to do it here anyway is that the fusion stops being something the
compiler has to find, and that the arithmetic — including which parts are fp32 —
is stated in one place.

Interface
---------
    q  [tokens * heads, dim]  bf16   row (t, h) is token t's head h
    k  [n, dim]               bf16   one shared index key per compressed group
    w  [heads, tokens]        fp32   `weights_proj(x)` already scaled, head-major
    -> [tokens, n]            fp32   `sum_h relu(q[t,h].k[j]) * w[h,t]`

`dim % 128 == 0`, `n % 128 == 0` (`model._compress_read_len` rounds to 128 for
exactly this reason) and `heads <= 128`. `w` arrives transposed because a
`[heads, 1]` column is what the weighted sum contracts over, and taking it from
a `[heads, tokens]` tile is a slice where taking it from `[tokens, heads]` would
be a per-token transpose.

Where the precision is deliberate
---------------------------------
  * the QK product takes bf16 operands into an fp32 psum accumulator. Both
    operands have been through `fp4_quant_dequant` with ue8m0 (power-of-two)
    block scales by the time they arrive, so bf16 holds them exactly and this
    matches the reference's fp32 einsum bit for bit on the inputs it is given;
  * the rectified scores stay **fp32** into the weighted sum, which is an fp32
    `nc_matmul` against the weight column. Rounding them to bf16 first would be
    cheaper and is what the attention kernel does with its probabilities — but
    there the reference itself stages in bf16, and here it does not: the sum
    runs in fp32 and its result picks the top-`index_topk` positions, where a
    rounding can reorder neighbours.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width of the tensor engine.
PT = 128
#: A psum bank holds 2 KiB per partition = 512 fp32, which is also `nc_matmul`'s
#: widest moving free axis.
PSUM_FP32 = 512


def _ntile(n):
    """Free-axis width for the score tile: as wide as a psum bank allows, and a
    divisor of `n` so no compressed position is dropped.

    `n` is `model._compress_read_len`'s output, so it is a multiple of 128 and
    this returns 512 whenever it is also a multiple of 512 — which the 128-tile
    rounding makes the common case at any real context length.
    """
    if n <= PSUM_FP32:
        return n
    for cand in (PSUM_FP32, 256, PT):
        if n % cand == 0:
            return cand
    return PT


def _align16(n):
    """Round an SBUF free extent up to 16 elements = 32 bytes, because a
    `dma_transpose` destination has to be 32-byte aligned. See
    `sparse_attn._align16` for the failure it avoids — a NEFF *load* error
    rather than a compile error.
    """
    return -(-n // 16) * 16


@nki.jit
def weighted_scores(q, k, w):
    """Rectified per-head QK scores combined by the indexer's head weights.
    See the module docstring for the layout and the precision notes."""
    tq, d = q.shape
    n, dk = k.shape
    heads, tokens = w.shape

    assert dk == d, f"k dim {dk} does not match q dim {d}"
    assert tq == tokens * heads, (
        f"q rows {tq} is not tokens({tokens}) * heads({heads})"
    )
    assert d % PT == 0, f"dim={d} must be a multiple of {PT}"
    assert n % PT == 0, f"n={n} must be a multiple of {PT} — the caller rounds"
    assert heads <= PT, f"heads={heads} exceeds the partition width {PT}"

    dchunks = d // PT
    ktiles = n // PT
    nt = _ntile(n)
    ntiles = n // nt
    hp = _align16(heads)

    out = nl.ndarray((tokens, n), dtype=nl.float32, buffer=nl.shared_hbm)

    # The index keys are shared by every query, so their transpose is hoisted
    # out of the token loop: both products below contract on `dim`, which has to
    # be the partition axis.
    kt = nl.ndarray((PT, dchunks, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in nl.static_range(dchunks):
        for kb in nl.static_range(ktiles):
            nisa.dma_transpose(
                dst=kt[0:PT, c, kb * PT : (kb + 1) * PT],
                src=k[kb * PT : (kb + 1) * PT, c * PT : (c + 1) * PT],
            )

    for t in nl.affine_range(tokens):
        # This token's heads, transposed so the QK product contracts on `dim`.
        qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in nl.static_range(dchunks):
            nisa.dma_transpose(
                dst=qt[0:PT, c, 0:heads],
                src=q[t * heads : t * heads + heads, c * PT : (c + 1) * PT],
            )
        # The weight column for this token: `[heads, 1]`, the contraction
        # operand of the weighted sum.
        wt = nl.load(w[0:heads, t : t + 1], dtype=nl.float32)

        row = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
        for nb in nl.static_range(ntiles):
            acc = nl.ndarray((heads, nt), dtype=nl.float32, buffer=nl.psum)
            for c in nl.static_range(dchunks):
                nisa.nc_matmul(
                    dst=acc,
                    stationary=qt[0:PT, c, 0:heads],
                    moving=kt[0:PT, c, nb * nt : (nb + 1) * nt],
                    accumulate=(c > 0),
                )
            # `relu`, in fp32, out of the psum accumulator.
            rect = nl.ndarray((heads, nt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=rect, data=acc, op0=nl.maximum, operand0=0.0
            )
            # The head sum, weighted: contracts on `heads`, so the one-wide
            # stationary operand is the weight column and the result is this
            # token's `[1, nt]` slice of the score row.
            wsum = nl.ndarray((1, nt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=wsum, stationary=wt[0:heads, 0:1], moving=rect[0:heads, 0:nt]
            )
            nisa.tensor_copy(dst=row[0:1, nb * nt : (nb + 1) * nt], src=wsum)
        nl.store(out[t : t + 1, 0:n], value=row)
    return out
