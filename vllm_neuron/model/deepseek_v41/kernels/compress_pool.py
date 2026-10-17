"""The KV compressor's learned softmax pooling, as one NKI kernel.

`model.Compressor.forward` pools `compress_ratio` consecutive tokens into one
KV latent with a *learned gate*: a second projection (`wgate`) produces one
logit per channel per member, the logits are softmaxed **across the members**,
and the members' `wkv` outputs are summed against those weights. 18 of the 40
decoder layers compress at ratio 2 (`config.compress_ratios`), so this runs on
every one of them on every forward.

It was the last piece of named heavy arithmetic still computed by torch on the
live path — `(kv * score.softmax(dim=1)).sum(dim=1)` over fp32 widenings of the
two paged buffers — which is to say it ran *on the device*, inside
`Block.attn_half`'s graph, as an XLA subgraph rather than as a kernel.

Derived from `legacy/l2_attn.py`, lines 1044-1062, which is the same pooling
fused into that partition's whole attention kernel::

    g0 = nl.load(cgt[r0:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
    g1 = nl.load(cgt[r0 + 1:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
    v0 = nl.load(ckv[r0:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
    v1 = nl.load(ckv[r0 + 1:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
    mx = nl.maximum(g0, g1)
    e0 = nl.exp(nl.subtract(g0, mx))
    e1 = nl.exp(nl.subtract(g1, mx))
    nisa.reciprocal(dst=rz, data=nl.add(e0, e1))
    p0 = nl.multiply(e0, rz)
    p1 = nl.multiply(e1, rz)
    acc = nl.add(nl.multiply(v0, p0), nl.multiply(v1, p1))

Three things change, and they are the same three every kernel in the parent
directory changed on the way up out of `legacy/`: the member count comes from
the operand shape rather than being the module constant `RATIO`, the group
count is ragged rather than a baked multiple of 128, and the group tile is a
`(groups, ratio, dim)` operand the caller gathers rather than this partition's
own `private_hbm` scratch. The arithmetic — including the strided member load,
the max subtraction and the `reciprocal`-then-multiply form of the division —
is line for line the above.

Interface
---------
    kv     [groups, ratio, dim]  bf16 or fp32   `wkv`'s per-member output
    score  [groups, ratio, dim]  same           `wgate`'s per-member logits
    ->     [groups, dim]         fp32           the pooled latent

`dim` is `config.head_dim` (512) and `ratio` is 2 for every layer that
compresses in this checkpoint, but neither is baked: `ratio` is read off the
operand and any `ratio >= 2` traces. The output is **fp32**, not the operands'
dtype, because the caller's next step is `Compressor.norm` and the reference
pools in fp32 and rounds once, on the way into that norm.

Why the reduction is row-count invariant, which matters for speculation
----------------------------------------------------------------------
A group's output depends on that group's `ratio` members and nothing else: the
softmax is over the member axis, which sits in the *free* axis of one partition,
and the group axis is the partition axis. So a group pooled in a 1-group call
and the same group pooled in a 256-group call accumulate the same values in the
same order. `kernels/rmsnorm.py`'s module docstring explains why that property
is load-bearing here — a speculative verify step carries `1 + k` rows where a
plain decode step carries one, and an argmax over 129,280 near-tied logits can
turn a last-bit difference into a different token.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width; one *group* per partition, 128 groups per tile.
PT = 128


def _chunk(dim):
    """Free-axis chunk for the pooling: a divisor of `dim` small enough that
    `ratio + 3` fp32 copies of it fit a partition comfortably.

    `rmsnorm._chunk`'s rule. At this model's `head_dim = 512` it returns 512 and
    the loop runs once; the smaller candidates are for a config that widens the
    latent past what one pass can stage.
    """
    for cand in (2048, 1024, 512, 256, 128):
        if dim % cand == 0:
            return cand
    return dim


def _recip(d, shape):
    """`1 / d` with one Newton step: `y * (2 - d * y)`.

    `nisa.reciprocal` is a hardware approximation, like `nl.rsqrt`, which
    `rmsnorm._rsqrt` and `hc_mixes._rms_recip` both correct the same way and for
    the same reason: the reference these kernels are checked against is a torch
    division. `legacy/l2_attn.py`'s pooling uses the bare instruction, and
    correcting it is the one place this kernel knowingly departs from its
    ancestor.

    **The correction is invisible under `nki.simulate`** — the simulator
    evaluates `reciprocal` in numpy, exactly — so it cannot be credited with any
    of the accuracy measured there, and it was not: with and without this step,
    simulation gives the same 9.5e-07 worst absolute error against torch on
    values of RMS 1.8 (`/tmp/iter13/errcheck.py`). It is here for the device,
    where the instruction alone is good to about 12 bits, which is the same
    order as the bf16 rounding the caller applies next.
    """
    y = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=y, data=d)
    return nl.multiply(y, nl.add(nl.multiply(nl.multiply(d, y), -1.0), 2.0))


@nki.jit
def softmax_pool(kv, score):
    """`(kv * score.softmax(dim=1)).sum(dim=1)`. See the module docstring."""
    groups, ratio, dim = kv.shape
    assert score.shape == (groups, ratio, dim), (
        f"score {score.shape} wants ({groups}, {ratio}, {dim})"
    )
    assert ratio >= 2, f"ratio={ratio} — a single-member group needs no pooling"

    # The members of one group are `ratio` consecutive rows, which is what the
    # caller's gather produces and what `legacy/l2_attn` reads with a strided
    # load. Flattening here makes that stride expressible: member `m` of the
    # groups in a tile is `[r0 + m : r0 + rn : ratio]`.
    kv2 = kv.reshape((groups * ratio, dim))
    sc2 = score.reshape((groups * ratio, dim))

    ch = _chunk(dim)
    nchunks = dim // ch

    out = nl.ndarray((groups, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    def body(g0, ng):
        r0, rn = g0 * ratio, ng * ratio
        for j in range(nchunks):
            c0 = j * ch
            g = [
                nl.load(
                    sc2[r0 + m : r0 + rn : ratio, c0 : c0 + ch], dtype=nl.float32
                )
                for m in range(ratio)
            ]
            # Max subtraction, as `torch.softmax` does it: the shift cancels in
            # the quotient, so this is exact rather than an approximation, and
            # it is what keeps a large gate logit from overflowing the exp.
            mx = g[0]
            for m in range(1, ratio):
                mx = nl.maximum(mx, g[m])
            e = [nl.exp(nl.subtract(g[m], mx)) for m in range(ratio)]
            den = e[0]
            for m in range(1, ratio):
                den = nl.add(den, e[m])
            rz = _recip(den, (ng, ch))

            # In member order, so the fp32 accumulation order is the reference's
            # `sum(dim=1)`. The probability is formed first and multiplies the
            # value second — `legacy/l2_attn`'s `p0`/`p1` then `v0 * p0`.
            acc = None
            for m in range(ratio):
                v = nl.load(
                    kv2[r0 + m : r0 + rn : ratio, c0 : c0 + ch], dtype=nl.float32
                )
                term = nl.multiply(v, nl.multiply(e[m], rz))
                acc = term if acc is None else nl.add(acc, term)
            nl.store(out[g0 : g0 + ng, c0 : c0 + ch], value=acc)

    # The ragged-row pattern every kernel in this directory uses: full tiles in
    # `affine_range`, then one more call with a Python row count for the tail.
    # A decode step that completes a group has `groups == 1` and takes the tail
    # path alone.
    full, tail = groups // PT, groups % PT
    for t in nl.affine_range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return out
