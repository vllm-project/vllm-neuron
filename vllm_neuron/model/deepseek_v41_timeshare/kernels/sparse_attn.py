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

What stays outside the kernel is `kv[batch, idxs]` and the comparison that
builds the mask — data movement and a compare, both inside the caller's
enclosing compiled graph. The arithmetic is all here.

Interface
---------
    q     [tokens * heads, head_dim]  bf16   row (t, h) is token t's head h
    g     [tokens * topk,  head_dim]  bf16   gathered slots, zeroed where empty
    bias  [tokens, topk]              fp32   0 for a live slot, -1e30 for empty
    sink  [heads, 1]                  fp32   the per-head learned logit bias
    scale [1, 1]                      fp32   `softmax_scale`
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
def gathered_attn(q, g, bias, sink, scale):
    """Multi-head attention over per-query gathered slots. See the module
    docstring for the layout and the precision notes."""
    tq, d = q.shape
    tokens, k = bias.shape
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
    # `scale` is a tensor rather than a kernel constant so that one NEFF serves
    # every layer. A freshly loaded `(1, n)` tile broadcasts to `(rows, n)`; a
    # *slice* of an SBUF tile does not — see `hc_mixes._spread`.
    scale_p = nl.broadcast_to(nl.load(scale[0:1, 0:1]), (heads, 1))
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
        bias_b = nl.broadcast_to(nl.load(bias[t : t + 1, 0:k]), (heads, k))
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
                operand1=bias_b[0:heads, nb * kt : (nb + 1) * kt],
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
