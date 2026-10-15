"""The attention arithmetic — QK score, masked softmax with the learned sink,
and the probability-weighted sum — as one NKI kernel.

This is the kernel behind `model.sparse_attn`, which is the only multi-head
attention this architecture has: every backbone layer and the DSpark draft
block attend over a *gathered* slot list (the sliding window, plus the
compressed latents the indexer chose) rather than over a contiguous KV range.

Why this shape, and not the one the shipped partition kernels use
--------------------------------------------------------------------------
`l3_attn._sparse_attention_impl` scores **every** live compressed slot and
multiplies by a scatter-built mask. That is exact, and at the 128-token trace it
was generated for it is faster than a gather. It is also quadratic in the
compressed length: at the verifier's 8192-token prompt it would score
8192 x 4096 slots per head instead of 8192 x 640, which is more arithmetic than
the rest of the model put together. This kernel takes the slots already
gathered, so its cost is linear in `topk`. Do not "upgrade" it to the masked
form.

**The gather is inside the kernel** (`gathered_attn_idx`, which is what
`model.sparse_attn` calls). `gathered_attn` is the same arithmetic over slots
the caller gathered, kept as the equality reference the fused one is held to;
`gathered_attn_idx`'s docstring carries the fp32 argument for why the two agree
bit for bit and what the staged form cost. What stays outside either kernel is
the comparison that builds the mask and the clamp of the empty sentinel — two
elementwise ops on a `[tokens, topk]` int32 tensor.

Interface
---------
    q     [tokens * heads, head_dim]  bf16   row (t, h) is token t's head h
    kv    [n, head_dim]               bf16   the slot table (`gathered_attn_idx`)
    idx   [tokens, topk]              int32  slot per (token, slot), clamped
    g     [tokens * topk,  head_dim]  bf16   gathered slots (`gathered_attn`)
    bias  [tokens, BIAS_PAD + topk]   fp32   `softmax_scale` in column 0, then
                                             0 for a live slot, -1e30 for empty
    sink  [heads, 1]                  fp32   the per-head learned logit bias
    ->    [tokens * heads, head_dim]  bf16

`head_dim % 128 == 0`, `topk % 128 == 0` and `heads <= 128`. Padding the top-k
list with empty slots is free, because an empty slot's bias removes it from the
softmax, so the caller pads rather than this kernel.

Where the precision is deliberate
---------------------------------
Transcribed from `vendor/kernel.py`'s `sparse_attn` and the torch equivalent in
`compat/sparse_attn_sm89.py`:

  * the QK product takes bf16 operands into an fp32 psum accumulator, and the
    scale multiplies the *logits* rather than `q`, so it never passes through a
    bf16 rounding;
  * the denominator sums **fp32** probabilities while the PV product consumes
    **bf16** ones — that is the reference's own split (its staging buffer is
    bf16), which is why `_softmax` writes the exponentials twice;
  * the sink joins the denominator after the row max is final.

A row whose every slot is empty comes out **exactly zero**, not NaN: the row max
is -1e30, every probability is `exp(0) = 1`, and `exp(sink + 1e30)` overflows to
`+inf`, so the reciprocal of the denominator is 0. That is the reference
kernel's convention and is why `model._NEG_BOUND` is finite rather than `-inf`.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width of the tensor engine.
PT = 128
#: A psum bank holds 2 KiB per partition = 512 fp32, which is also `nc_matmul`'s
#: widest moving free axis.
PSUM_FP32 = 512

#: `gathered_attn_idx` takes the softmax scale in column 0 of its bias tensor
#: rather than as its own operand, and this is how much of the row comes before
#: the bias proper. 16 fp32 = 64 bytes, so the bias slice the score product
#: reads is 64-byte aligned where a 1-column prefix would put it at byte 4.
#:
#: Why they share a tensor at all: `nisa.scalar_tensor_tensor` takes the scale
#: as a per-partition vector and the bias as a tile, both from SBUF, and the
#: hardware requires two SBUF inputs of one instruction to start at the same
#: partition. As two separate `nl.broadcast_to(nl.load(...))` tiles that holds
#: only if the allocator happens to agree, and at some shapes it does not —
#:
#:     [NCC_IBIR297] For specific instructions, base partition for access is
#:     expected to be equal if both inputs are in SB
#:
#: which **both** kernels here hit when compiled standalone: `gathered_attn_idx`
#: at `tokens=8, topk=256` and `gathered_attn` at `tokens=512, topk=128`. It has
#: never fired inside the model, where these kernels are compiled as part of a
#: much larger graph, so this is a latent fragility rather than a live bug — but
#: it does mean the shipped form cannot be compiled on its own at every live
#: shape, which is exactly what an equality harness needs to do. Slices of one
#: broadcast tile cannot disagree about their base partition.
BIAS_PAD = 16


def _ktile(k):
    """Free-axis width for the score tile: as wide as a psum bank allows, and a
    divisor of `k` so no slots are dropped.

    `topk` is 128 (a window-only layer) or 640 (window + `index_topk`), so this
    returns 128 for the layers that read compressed KV and 128 for the ones that
    do not; 512 and 256 are here for the shapes a different config would give.
    """
    if k <= PSUM_FP32:
        return k
    for cand in (PSUM_FP32, 256, PT):
        if k % cand == 0:
            return cand
    return PT


def _align16(n):
    """Round an SBUF free extent up to 16 elements = 32 bytes.

    A `dma_transpose` destination must be 32-byte aligned, and the destination
    here is `qt[:, c, :]` at byte offset `c * extent * 2`. An unaligned extent
    does not fail the compile — it fails the NEFF *load*, with the cause only
    visible under `NEURON_RT_LOG_LEVEL=INFO`:

        TDRV:dma_util_expand_desc_transpose  transpose dest offset ... must be
        32B aligned
    """
    return -(-n // 16) * 16


@nki.jit
def gathered_attn(q, g, bias, sink):
    """Multi-head attention over per-query *pre-gathered* slots.

    **This is the reference, not the live path.** `model.sparse_attn` calls
    `gathered_attn_idx` below, which gathers the slots itself; this one stays
    because it is the golden output that version is held to — same inputs,
    identical bytes out, checked element by element on the device
    (`/tmp/it4/devcmp.py`). Keeping it costs nothing (it is a separate
    `nki.jit` object, compiled only when called) and it is the only way to
    prove the fused version did not move a logit, which on this port decides
    DSpark acceptance.

    Two things differ from the version iterations 0-3 shipped, neither of them
    arithmetic:

      * `softmax_scale` arrives in column 0 of `bias` rather than as its own
        `(1, 1)` operand — see `BIAS_PAD`. As two separate broadcast tiles the
        scale and the bias hit `[NCC_IBIR297]` (two SBUF inputs of one
        instruction at different base partitions) at `tokens=512, topk=128`,
        in *this* kernel too, so the shipped form cannot be compiled standalone
        at every live shape and could not serve as the reference unchanged. The
        instruction and both operand *values* are untouched, and the tiny-shape
        run of `devcmp.py` compared the shipped form against the combined one
        and got identical bytes;
      * nothing else.

    See the module docstring for the layout and the precision notes."""
    tq, d = q.shape
    tokens, kb_ = bias.shape
    k = kb_ - BIAS_PAD
    heads = tq // tokens

    assert tq == tokens * heads, f"q rows {tq} is not tokens({tokens}) * heads"
    assert g.shape == (tokens * k, d), (
        f"g {g.shape} does not match tokens({tokens}) * topk({k}) x head_dim({d})"
    )
    assert d % PT == 0, f"head_dim={d} must be a multiple of {PT}"
    assert k % PT == 0, f"topk={k} must be a multiple of {PT} — the caller pads"
    assert heads <= PT, f"heads={heads} exceeds the partition width {PT}"
    assert sink.shape == (heads, 1), f"sink {sink.shape} wants ({heads}, 1)"

    dchunks = d // PT
    ktiles = k // PT
    kt = _ktile(k)
    ntiles = k // kt
    hp = _align16(heads)

    out = nl.ndarray((tq, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Loop-invariant, and both have to be *loaded*: `nisa.activation` and
    # `nisa.tensor_scalar` take SBUF/PSUM operands, and handing either one an
    # HBM tensor fails at trace time with
    #
    #     AssertionError: activation data must be in ['sbuf', 'psum'], got
    #     shared_hbm
    #
    # The scale is a tensor rather than a kernel constant so that one NEFF
    # serves every layer. A freshly loaded `(1, n)` tile broadcasts to
    # `(rows, n)`; a *slice* of an SBUF tile does not — see `hc_mixes._spread`.
    sink_t = nl.load(sink[0:heads, 0:1], dtype=nl.float32)

    # One token's working set, reused by every iteration: the transposed query
    # (for the score product, which contracts on head_dim), the transposed
    # slots (same), and the slots in their natural layout (for the value
    # product, which contracts on the slot axis). Both products contract on the
    # partition axis, which is what forces the two copies of `g`.
    for t in nl.affine_range(tokens):
        qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        gt = nl.ndarray((PT, dchunks, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in nl.static_range(dchunks):
            nisa.dma_transpose(
                dst=qt[0:PT, c, 0:heads],
                src=q[t * heads : t * heads + heads, c * PT : (c + 1) * PT],
            )
            nisa.dma_transpose(
                dst=gt[0:PT, c, 0:k],
                src=g[t * k : (t + 1) * k, c * PT : (c + 1) * PT],
            )
        for kb in nl.static_range(ktiles):
            nisa.tensor_copy(
                dst=gn[0:PT, kb, 0:d],
                src=nl.load(
                    g[t * k + kb * PT : t * k + (kb + 1) * PT, 0:d],
                    dtype=nl.bfloat16,
                ),
            )

        # ── logits ────────────────────────────────────────────────────────
        bias_b = nl.broadcast_to(
            nl.load(bias[t : t + 1, 0 : BIAS_PAD + k]), (heads, BIAS_PAD + k)
        )
        scale_p = bias_b[0:heads, 0:1]
        sc = nl.ndarray((heads, k), dtype=nl.float32, buffer=nl.sbuf)
        for nb in nl.static_range(ntiles):
            acc = nl.ndarray((heads, kt), dtype=nl.float32, buffer=nl.psum)
            for c in nl.static_range(dchunks):
                nisa.nc_matmul(
                    dst=acc,
                    stationary=qt[0:PT, c, 0:heads],
                    moving=gt[0:PT, c, nb * kt : (nb + 1) * kt],
                    accumulate=(c > 0),
                )
            # `(score * scale) + bias` in one instruction. The empty slots'
            # -1e30 dominates any finite logit in fp32, so a masked slot's
            # logit is exactly -1e30 and the row max below is exact.
            nisa.scalar_tensor_tensor(
                dst=sc[0:heads, nb * kt : (nb + 1) * kt],
                data=acc,
                op0=nl.multiply,
                operand0=scale_p,
                op1=nl.add,
                operand1=bias_b[
                    0:heads, BIAS_PAD + nb * kt : BIAS_PAD + (nb + 1) * kt
                ],
            )

        # ── softmax, with the sink in the denominator ─────────────────────
        neg_max = nl.negative(nl.max(sc[0:heads, 0:k], axis=1, keepdims=True))
        p32 = nl.ndarray((heads, k), dtype=nl.float32, buffer=nl.sbuf)
        den = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=p32,
            op=nl.exp,
            data=sc[0:heads, 0:k],
            bias=neg_max,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=den,
        )
        sk = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
        rcp = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
        # The value product takes bf16 probabilities — the reference's staging
        # buffer is bf16 — while `den` above summed the fp32 ones.
        p16 = nl.copy(p32, dtype=nl.bfloat16)

        # ── probability-weighted sum ──────────────────────────────────────
        # The product contracts on the slot axis, so the probabilities have to
        # be transposed; 128 slots at a time, which is the PE array's width.
        pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in nl.static_range(ktiles):
            # bf16 in, bf16 out: `nc_transpose` in transpose mode insists the
            # destination dtype match the source on gen3+ —
            #
            #     AssertionError: nc_matmul (transpose mode) dst dtype must
            #     match input dtype on gen3+, got dst=float32 but
            #     input=bfloat16
            #
            # which costs nothing here: the probabilities are already bf16, so
            # the transpose is exact.
            ptps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(
                dst=ptps[0:PT, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT]
            )
            nisa.tensor_copy(
                dst=pt[0:PT, kb, 0:heads], src=ptps[0:PT, 0:heads]
            )
        acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
        for kb in nl.static_range(ktiles):
            nisa.nc_matmul(
                dst=acc_o,
                stationary=pt[0:PT, kb, 0:heads],
                moving=gn[0:PT, kb, 0:d],
                accumulate=(kb > 0),
            )
        o16 = nl.ndarray((heads, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=o16, data=acc_o, op0=nl.multiply, operand0=rcp
        )
        nl.store(out[t * heads : t * heads + heads, 0:d], value=o16)
    return out


@nki.jit
def gathered_attn_idx(q, kv, idx, bias, sink):
    """`gathered_attn` with the gather inside it. **This is the live path.**

    Interface
    ---------
        q     [tokens * heads, head_dim]   bf16   row (t, h) is token t's head h
        kv    [n, head_dim]                bf16   the slot table, *not* gathered
        idx   [tokens, topk]               int32  slot id per (token, slot),
                                                  already clamped to [0, n)
        bias  [tokens, BIAS_PAD + topk]    fp32   column 0 is `softmax_scale`;
                                                  from `BIAS_PAD` on, 0 for a
                                                  live slot and -1e30 for empty
        sink  [heads, 1]                   fp32   the per-head learned logit bias
        ->    [tokens * heads, head_dim]   bf16

    Why this exists
    ---------------
    `gathered_attn` takes `g = kv[idx]` as an operand, so the caller has to
    materialize it — and `g` is `[tokens, topk, head_dim]`, which at the live
    prefill shape (512 tokens, 640 slots, head_dim 512) is **320 MiB of bf16
    per layer per forward** expanded out of a 1.7 MiB table. Measured on
    iteration 3's profile, the torch expression that built it was the single
    largest item on the prefill path: 73% of the instruction time of one
    per-layer graph (`.autohelix/notes/iter-3.md` §0.4), because that tensor is
    written to HBM and then read back *twice* — once transposed for the score
    product, once in natural layout for the value product — and because the
    `masked_fill` that zeroes the empty slots is two more passes over it.

    Gathering here instead leaves one read: 320 MiB of indirect DMA straight
    into SBUF, against ~1.9 GiB of HBM traffic for the staged version.

    Bit-exactness, which is the point
    ---------------------------------
    Every value this computes is the same **bit pattern** `gathered_attn`
    produces, and the one difference is deliberate: an *empty* slot's row is
    read from `kv[idx]` (the caller clamps `-1` to `0`) rather than zeroed.
    Both reach the same output exactly, in fp32:

      * the logit of an empty slot is `q·kv[0] * scale + (-1e30)`. `|q·kv|` is
        bounded by a few times `head_dim` and one ulp of 1e30 is ~1.2e23, so
        the sum is **exactly** -1e30, which is what the zeroed row gives. The
        row max, and so every probability, is therefore identical;
      * the value product weights that row by `p = exp(-1e30 - max) = 0`, and
        `0 * finite` is exactly 0. `kv` is fp8/fp4-rounded activation, never
        inf or NaN, so the product cannot be NaN;
      * in the all-empty case (`p = 1` for every slot) the accumulator holds
        `sum(kv[0])` instead of 0, and the reciprocal of the denominator is
        exactly 0 because `exp(sink + 1e30)` is `+inf` — so the row is exactly
        zero either way, which is the reference's convention.

    Nothing else changes: the same products are summed in the same sequence,
    the slot axis keeps `idx`'s order, `den` is still one hardware reduction
    over the same `(heads, topk)` fp32 tile, and the value product still
    accumulates over slot tiles in ascending order.

    What it costs
    -------------
    `gathered_attn` gets `gt` (head_dim on the partition axis, for the score
    product) and `gn` (slots on the partition axis, for the value product) from
    two separate HBM reads of `g`. Here one indirect DMA fills `gn` and the
    tensor engine transposes it into `gt`, which is `dchunks` transposes and
    one PSUM-to-SBUF copy per slot tile. That is on-chip work traded for HBM
    traffic, and the trade came out free: **measured, this graph's tensor-engine
    time went 856.8 -> 840.4 us an execution** while its Vector, GpSimd and
    Scalar time each fell ~43% (`.autohelix/notes/iter-4.md` §3b). 10 240
    128x128 bf16 transposes an execution cost the PE array nothing, because on
    this model it is busy ~22% of a step and nothing else wants it.
    """
    tq, d = q.shape
    tokens, kb_ = bias.shape
    k = kb_ - BIAS_PAD
    heads = tq // tokens
    nrows, kvd = kv.shape

    assert tq == tokens * heads, f"q rows {tq} is not tokens({tokens}) * heads"
    assert kvd == d, f"kv head_dim {kvd} does not match q head_dim {d}"
    assert idx.shape == (tokens, k), (
        f"idx {idx.shape} does not match tokens({tokens}) x topk({k})"
    )
    assert nrows > 0, "kv is empty"
    assert d % PT == 0, f"head_dim={d} must be a multiple of {PT}"
    assert k % PT == 0, f"topk={k} must be a multiple of {PT} — the caller pads"
    assert heads <= PT, f"heads={heads} exceeds the partition width {PT}"
    assert sink.shape == (heads, 1), f"sink {sink.shape} wants ({heads}, 1)"

    dchunks = d // PT
    ktiles = k // PT
    kt = _ktile(k)
    ntiles = k // kt
    hp = _align16(heads)

    out = nl.ndarray((tq, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sink_t = nl.load(sink[0:heads, 0:1], dtype=nl.float32)

    for t in nl.affine_range(tokens):
        qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        gt = nl.ndarray((PT, dchunks, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)

        # This token's slot ids, laid out so that tile `kb`'s ids sit in one
        # column: `ids[p, kb] = idx[t, kb * PT + p]`. An indirect DMA takes its
        # per-partition offsets from an SBUF column, so this is the layout the
        # gather below needs, and one strided read fills every tile's column at
        # once. The dynamic `t` rides in on the slice's own storage offset —
        # `ap(offset=...)` wants a Python int.
        ids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ids, src=idx[t : t + 1, 0:k].ap(pattern=[[1, PT], [PT, ktiles]])
        )

        for c in nl.static_range(dchunks):
            nisa.dma_transpose(
                dst=qt[0:PT, c, 0:heads],
                src=q[t * heads : t * heads + heads, c * PT : (c + 1) * PT],
            )

        for kb in nl.static_range(ktiles):
            # The gather. `vector_offset` gives every partition its own row of
            # `kv`, scaled by `kv`'s own row stride, so partition `p` reads
            # `kv[ids[p, kb]]`. `indirect_dim=0` is the only form the hardware
            # has (`addr = vec_off[p] * coef + f`), which is why `ids` had to be
            # a column.
            nisa.dma_copy(
                dst=gn[0:PT, kb, 0:d],
                src=kv.ap(
                    pattern=[[d, PT], [1, d]],
                    offset=0,
                    vector_offset=ids[0:PT, kb : kb + 1],
                    indirect_dim=0,
                ),
            )
            # And the second layout, on chip. One PSUM tile per slot tile holds
            # all `dchunks` transposed blocks, so the copy out is one
            # instruction over a strided destination rather than `dchunks` of
            # them. bf16 in, bf16 out: `nc_transpose` insists the destination
            # dtype match the source on gen3+, which costs nothing here.
            tps = nl.ndarray((PT, dchunks, PT), dtype=nl.bfloat16, buffer=nl.psum)
            for c in nl.static_range(dchunks):
                nisa.nc_transpose(
                    dst=tps[0:PT, c, 0:PT],
                    data=gn[0:PT, kb, c * PT : (c + 1) * PT],
                )
            nisa.tensor_copy(
                dst=gt[0:PT, 0:dchunks, kb * PT : (kb + 1) * PT], src=tps
            )

        # ── logits ────────────────────────────────────────────────────────
        # One broadcast load carries both the scale (column 0) and this token's
        # bias row, so the two SBUF operands of `scalar_tensor_tensor` below are
        # slices of one tile — see `BIAS_PAD`.
        bias_b = nl.broadcast_to(
            nl.load(bias[t : t + 1, 0 : BIAS_PAD + k]), (heads, BIAS_PAD + k)
        )
        scale_p = bias_b[0:heads, 0:1]
        sc = nl.ndarray((heads, k), dtype=nl.float32, buffer=nl.sbuf)
        for nb in nl.static_range(ntiles):
            acc = nl.ndarray((heads, kt), dtype=nl.float32, buffer=nl.psum)
            for c in nl.static_range(dchunks):
                nisa.nc_matmul(
                    dst=acc,
                    stationary=qt[0:PT, c, 0:heads],
                    moving=gt[0:PT, c, nb * kt : (nb + 1) * kt],
                    accumulate=(c > 0),
                )
            nisa.scalar_tensor_tensor(
                dst=sc[0:heads, nb * kt : (nb + 1) * kt],
                data=acc,
                op0=nl.multiply,
                operand0=scale_p,
                op1=nl.add,
                operand1=bias_b[
                    0:heads, BIAS_PAD + nb * kt : BIAS_PAD + (nb + 1) * kt
                ],
            )

        # ── softmax, with the sink in the denominator ─────────────────────
        neg_max = nl.negative(nl.max(sc[0:heads, 0:k], axis=1, keepdims=True))
        p32 = nl.ndarray((heads, k), dtype=nl.float32, buffer=nl.sbuf)
        den = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=p32,
            op=nl.exp,
            data=sc[0:heads, 0:k],
            bias=neg_max,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=den,
        )
        sk = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
        rcp = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
        p16 = nl.copy(p32, dtype=nl.bfloat16)

        # ── probability-weighted sum ──────────────────────────────────────
        pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in nl.static_range(ktiles):
            ptps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(
                dst=ptps[0:PT, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT]
            )
            nisa.tensor_copy(
                dst=pt[0:PT, kb, 0:heads], src=ptps[0:PT, 0:heads]
            )
        acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
        for kb in nl.static_range(ktiles):
            nisa.nc_matmul(
                dst=acc_o,
                stationary=pt[0:PT, kb, 0:heads],
                moving=gn[0:PT, kb, 0:d],
                accumulate=(kb > 0),
            )
        o16 = nl.ndarray((heads, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=o16, data=acc_o, op0=nl.multiply, operand0=rcp
        )
        nl.store(out[t * heads : t * heads + heads, 0:d], value=o16)
    return out
