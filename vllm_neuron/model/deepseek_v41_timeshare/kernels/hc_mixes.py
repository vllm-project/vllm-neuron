"""The hyper-connection mixing coefficients — the stream statistic, the three
nonlinearities and the 20-iteration Sinkhorn — as one NKI kernel.

`model.HyperConnections.mixes` runs twice per layer (once for the attention
sublayer, once for the FFN), so 86 times per token per forward. The GEMM that
produces the raw coefficients is already `kernels/blockquant.fp32w_linear`;
everything after it used to be torch, and this kernel is that remainder:

    rms      one statistic per token over the whole flattened `hc * dim` stream
    pre      sigmoid(raw * scale0 + base0) + eps
    post     2 * sigmoid(raw * scale1 + base1)
    comb     softmax(raw * scale2 + base2) + eps, then Sinkhorn

Interface
---------
    raw   [rows, width]      bf16   the coefficient GEMM's output, unscaled
    flat  [rows, hc * dim]   bf16   the flattened residual stream
    base  [1, width]         fp32
    scale [1, 3]             fp32
    ->    pre  [rows, hc] fp32, post [rows, hc] fp32, comb [rows, hc*hc] fp32

`hc` is recovered from `width = (2 + hc) * hc`, and `rows` may be ragged — the
token count is whatever the scheduler handed the step.

Baked constants
---------------
`SINKHORN_ITERS` cannot be a runtime argument (it is a loop bound in the
instruction stream) and `HC_EPS` / `NORM_EPS` are folded into the instructions
beside it. `DeepseekV41ForCausalLM._check_kernel_constants` asserts all three
against the config, next to the same assertion for `hc_expand.HC`.

Two NKI facts this file depends on
----------------------------------
  * **`nl.rsqrt` is an approximation.** `l3_attn._rmsnorm` follows it with one
    Newton step and so does `_rms_recip` here; without it the statistic is
    visibly off against the torch `rsqrt`.
  * **`nl.broadcast_to` will not broadcast a *slice*.** Its access pattern's
    partition step is the source tile's row stride, and it requires that to
    equal the free extent being broadcast from, so `scale_tile[:, 0:1]` over 4
    columns fails:

        AssertionError: ap() pattern has invalid partition stride. Partition
        step 1 must equal tensor free dimension size 3.

    A freshly `nl.load`ed `(1, n)` tile broadcast to `(rows, n)` is fine, which
    is why every coefficient here is loaded from its own HBM slice rather than
    sliced out of one staged tile.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width.
PT = 128
#: `hc_sinkhorn_iters`. Not a convergence tolerance — the reference runs exactly
#: this many passes, so matching the count matters bit-for-bit.
SINKHORN_ITERS = 20
#: `hc_eps`.
HC_EPS = 1e-06
#: `rms_norm_eps`.
NORM_EPS = 1e-20


def _hc_from_width(width):
    """`width = (2 + hc) * hc` inverted. 24 -> 4."""
    hc = 1
    while (2 + hc) * hc < width:
        hc += 1
    assert (2 + hc) * hc == width, f"width={width} is not (2 + hc) * hc"
    return hc


def _sqchunk(fd):
    """Free-axis chunk for the sum of squares: a divisor of the stream width,
    small enough that one fp32 copy of it is a few KiB per partition."""
    for cand in (2048, 1024, 512, 256, 128):
        if fd % cand == 0:
            return cand
    return fd


def _rms_recip(flat, r0, nr, fd):
    """`1 / sqrt(mean(flat^2) + eps)` for each of `nr` rows, `[nr, 1]` fp32.

    Accumulated chunk by chunk rather than over one `[nr, fd]` fp32 tile: at
    `hc * dim = 20480` that tile would be 80 KiB per partition, and the
    chunked sum is the same arithmetic modulo summation order (~1e-7).
    """
    ch = _sqchunk(fd)
    total = None
    for c in range(fd // ch):
        x = nl.load(flat[r0 : r0 + nr, c * ch : (c + 1) * ch], dtype=nl.float32)
        part = nl.sum(nl.square(x), axis=1, keepdims=True)
        total = part if total is None else nl.add(total, part)
    m = nl.add(nl.multiply(total, 1.0 / fd), NORM_EPS)
    y = nl.rsqrt(m)
    # One Newton step on the reciprocal square root: y * (1.5 - 0.5 * m * y^2).
    # `nl.rsqrt` alone is a hardware approximation; `l3_attn._rmsnorm` corrects
    # it the same way.
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    return nl.multiply(y, corr)


@nki.jit
def kernel(raw, flat, base, scale):
    """See the module docstring. Returns `(pre, post, comb)`."""
    rows, width = raw.shape
    frows, fd = flat.shape
    hc = _hc_from_width(width)

    assert frows == rows, f"flat has {frows} rows, raw has {rows}"
    assert fd % hc == 0, f"stream width {fd} is not a multiple of hc={hc}"
    assert base.shape == (1, width), f"base {base.shape} wants (1, {width})"
    assert scale.shape == (1, 3), f"scale {scale.shape} wants (1, 3)"

    pre = nl.ndarray((rows, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    post = nl.ndarray((rows, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb = nl.ndarray((rows, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    def body(r0, nr):
        rs = _rms_recip(flat, r0, nr, fd)
        rawt = nl.load(raw[r0 : r0 + nr, 0:width], dtype=nl.float32)
        mix = nl.ndarray((nr, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mix, data=rawt, op0=nl.multiply, operand0=rs)

        # Each coefficient group gets its own scale and its own bias, and a
        # different nonlinearity — transcribed from `hc_split_sinkhorn_kernel`
        # in `vendor/kernel.py`, not inferred. `pre` is a sigmoid plus eps,
        # `post` is *twice* a sigmoid, `comb` is a softmax plus eps.
        s0 = nl.broadcast_to(nl.load(scale[0:1, 0:1]), (nr, 1))
        s1 = nl.broadcast_to(nl.load(scale[0:1, 1:2]), (nr, 1))
        s2 = nl.broadcast_to(nl.load(scale[0:1, 2:3]), (nr, 1))
        b0 = nl.broadcast_to(nl.load(base[0:1, 0:hc]), (nr, hc))
        b1 = nl.broadcast_to(nl.load(base[0:1, hc : 2 * hc]), (nr, hc))
        b2 = nl.broadcast_to(
            nl.load(base[0:1, 2 * hc : 2 * hc + hc * hc]), (nr, hc * hc)
        )

        t0 = nl.ndarray((nr, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=t0, data=mix[0:nr, 0:hc], op0=nl.multiply, operand0=s0,
            op1=nl.add, operand1=b0,
        )
        nl.store(pre[r0 : r0 + nr, 0:hc], value=nl.add(nl.sigmoid(t0), HC_EPS))

        t1 = nl.ndarray((nr, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=t1, data=mix[0:nr, hc : 2 * hc], op0=nl.multiply, operand0=s1,
            op1=nl.add, operand1=b1,
        )
        nl.store(post[r0 : r0 + nr, 0:hc], value=nl.multiply(nl.sigmoid(t1), 2.0))

        t2 = nl.ndarray((nr, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=t2,
            data=mix[0:nr, 2 * hc : 2 * hc + hc * hc],
            op0=nl.multiply, operand0=s2, op1=nl.add, operand1=b2,
        )
        # `comb[i, j]` lives at `i * hc + j`, so the reference's `dim=-1` (the
        # row of the hc x hc block) is the contiguous run `i*hc .. i*hc+hc` and
        # its `dim=-2` (the column) is strided.
        #
        # Everything below stays **two-dimensional** and reduces only over a
        # contiguous run, because this hardware will not reduce over anything
        # else:
        #
        #     AssertionError: tensor_reduce axis must be the last contiguous
        #     dim(s) of the tile. For a 3D tensor, expected axis=(2,) (or a
        #     shorter trailing subset). Got axis=(1,)
        #
        # so a `[nr, hc, hc]` tile cannot have its axis-1 summed. The column
        # sums are written as `hc - 1` adds of whole `[nr, hc]` row blocks
        # instead — which also needs no broadcast at all, since the sum of the
        # blocks *is* the per-column divisor laid out the way each block wants
        # it.
        # Both passes multiply by a reciprocal where the reference divides.
        # That is not a choice: a tensor-tensor `divide` is not an ALU
        # operation on this engine and the *compiler* rejects it, not the
        # tracer —
        #
        #     ISA validation failed: TensorTensorArithOp; inst failed
        #     assertion check: 's3s3d3_tt_valid_op'
        #     FATAL: neuronx-cc compilation failed with 70.
        #
        # 39 reciprocals over the Sinkhorn cost ~2e-06 of relative error
        # against torch's division, which is two orders below the bf16 rounding
        # already in the stream.
        def rowsum_div(src, dst):
            """Divide each `hc x hc` row by its own sum (the reference's
            `dim=-1` normalization). The divisor is one value per token, so it
            rides the per-partition operand and needs no broadcast."""
            for i in range(hc):
                lo, hi = i * hc, (i + 1) * hc
                s = nl.sum(src[0:nr, lo:hi], axis=1, keepdims=True)
                nisa.tensor_scalar(
                    dst=dst[0:nr, lo:hi],
                    data=src[0:nr, lo:hi],
                    op0=nl.multiply,
                    operand0=nl.reciprocal(nl.add(s, HC_EPS)),
                )

        def colsum_div(src, dst):
            """Divide each `hc x hc` column by its own sum (the reference's
            `dim=-2` normalization). The sum of the row blocks *is* the divisor
            laid out the way each block wants it."""
            s = nl.add(src[0:nr, 0:hc], src[0:nr, hc : 2 * hc])
            for i in range(2, hc):
                s = nl.add(s, src[0:nr, i * hc : (i + 1) * hc])
            d = nl.reciprocal(nl.add(s, HC_EPS))
            for i in range(hc):
                lo, hi = i * hc, (i + 1) * hc
                nisa.tensor_tensor(
                    dst=dst[0:nr, lo:hi], data1=src[0:nr, lo:hi], data2=d,
                    op=nl.multiply,
                )

        a = nl.ndarray((nr, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((nr, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(hc):
            lo, hi = i * hc, (i + 1) * hc
            mx = nl.max(t2[0:nr, lo:hi], axis=1, keepdims=True)
            e = nl.exp(
                nl.subtract(t2[0:nr, lo:hi], nl.broadcast_to(mx, (nr, hc)))
            )
            s = nl.sum(e, axis=1, keepdims=True)
            nisa.tensor_scalar(
                dst=a[0:nr, lo:hi],
                data=e,
                op0=nl.multiply,
                operand0=nl.reciprocal(s),
                op1=nl.add,
                operand1=HC_EPS,
            )

        # The Sinkhorn is not a symmetric loop: the first column normalization
        # is outside it, so there are `iters` column passes and `iters - 1` row
        # passes. Getting either count wrong leaves a model that runs and
        # drifts.
        colsum_div(a, b)
        for _ in range(SINKHORN_ITERS - 1):
            rowsum_div(b, a)
            colsum_div(a, b)
        nl.store(comb[r0 : r0 + nr, 0 : hc * hc], value=b[0:nr, 0 : hc * hc])

    # The ragged-row pattern every kernel in this directory uses: full tiles in
    # `affine_range`, where the row offset may be symbolic, then one more call
    # with a Python row count for the tail. A `min(PT, rows - row0)` inside
    # `affine_range` cannot size a tile, because `row0` is symbolic there.
    full, tail = rows // PT, rows % PT
    for t in nl.affine_range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return pre, post, comb
