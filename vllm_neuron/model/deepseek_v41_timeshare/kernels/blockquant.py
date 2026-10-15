"""Shape-general block-quantized linear kernels — the arithmetic every other
module in this model is mostly made of.

The 16 partition-specific kernels beside this file (`l2_attn`, `moe_short`, …)
are faster but each was generated against one trace: token count, window length
and rank count are module-level constants they assert on entry. A timeshared
port needs the same arithmetic at whatever shape the scheduler hands it, so
this module factors the three storage layouts the checkpoint uses into kernels
that take their shapes from the tensors:

    fp8  `[out, in]` e4m3 + `[ceil(out/32), ceil(in/32)]` e8m0   attention, shared experts, lm head path
    fp4  `[out, in//2]` two e2m1 per byte + `[out, in/32]` e8m0  routed experts
    bf16 `[out, in]`, no scale                                   compressor, indexer wk, router

Every weight here is **n-major**, as the checkpoint has it, and `_gemm` turns
each tile with a `dma_transpose` on the way into SBUF because `nc_matmul` wants
the contracted axis on partitions. That transpose is the largest single DMA
term in both measured paths (104-115 GB/s against ~464 for the same bytes read
contiguously), and iteration 1 of the optimizing run tried the obvious fix —
store the fp8 pair k-major, decode straight into the matmul's orientation — and
measured it **slower end to end**. `_dequant_fp8_k` and `fp8_linear_kmajor`
keep that arm, unreachable, with the numbers; `legacy/moe_short.py:623` shows
the form that does win, which decodes in SBUF and never stages a dequantized
weight through HBM at all. Read both before trying again.

The fp4 experts can be made k-major as cheaply — `legacy/moe_short.py` stores
them `[k/2, n]`, a plain transpose of the *byte* matrix (4.4 ms a matrix on the
host, ~170 µs on device), keeping the checkpoint's two-k-per-byte packing. What
that costs is that a byte's two nibbles land on one partition, so the
contraction splits into even-k and odd-k halves — two `nc_matmul`s per weight
tile against an activation pre-split to match (`moe_short.py:499-510`) — the
same products in a different fp32 order, so not bit-exact. See
`.autohelix/notes/iter-1.md` §5.

The bit-level decoders, the activation round trip and the GEMM pipeline are
lifted from `l3_attn.py` (`_decode_e4m3`, `_pow2_ceil_log2`, `_qdq_fp8`,
`_dequant_weight`, `_gemm`), which the previous project profiled and validated
against the reference — this file changes their shape handling, not their
arithmetic. Two consequences of that provenance worth knowing:

  * The activation round trip divides by `FP8_PEAK = 224`, not by the
    reference's 448, because Trn2's `nl.float8_e4m3` saturates at 240. It
    costs one mantissa bit against `model.act_quant` and is what the
    compiler's `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag is about.
  * The weight decoders leave the exponent bias out of the mantissa decode and
    fold it into the scale, so a decoded value is a *very* small bf16 (an e4m3
    `eeee = 0` lands on a bf16 denormal) and only the product with the scale is
    in range. Do not "simplify" either half on its own.

Every kernel here requires `m % 128 == 0` and `k % 128 == 0`; callers pad.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width of the tensor engine; every tile is 128 rows.
PT = 128
#: Widest free-axis tile. A psum bank holds 2 KiB per partition = 512 fp32.
NT = 512
#: Quantization block along the contracted axis, both for the activation and
#: for the stored scales.
BLK = 32

FP8_MAX = 448.0
#: Half of `FP8_MAX`: see the module docstring.
FP8_PEAK = FP8_MAX / 2.0
FP8_AMAX_FLOOR = 1e-4

MGROUP_BYTES = 64 * 1024
WLOOK = 8
XLOOK = 2

BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)
#: fp8: a decoded e4m3 comes out 2**-120 too small, and the stored scale means
#: 2**(code-127), so one multiply by 2**(code-7) closes both.
SC_EXP_ADD = 120

#: fp4: 0x1040. Multiplying a 4-bit code by this lands the sign in bit 15, the
#: two exponent bits in 7..8 and the mantissa bit in 6, which is a bf16 whose
#: exponent field *is* the e2m1 exponent.
FP4_NIB_MUL = 4160.0
#: sign | exponent(2) | mantissa(1) of the bf16 built above; everything else
#: the multiply produced is spill.
FP4_FIELD = 0x81C0
#: fp4: a decoded e2m1 is 2**-126 too small and the stored scale is
#: 2**(code-127), so the closing factor is 2**(code-1). Split as
#: 2**(code-1-FP4_SPLIT) on the bf16 weight and 2**FP4_SPLIT on the fp32
#: accumulator, so neither factor leaves bf16's exponent range.
FP4_SPLIT = 64
FP4_POST = float(2.0**FP4_SPLIT)
FP4_EXP_BIAS = 127 - 1 - FP4_SPLIT


def _mgroup(mtiles, ktiles):
    """How many m-tiles to stage at once: the largest power of two that both
    divides `mtiles` and keeps the staged activation under `MGROUP_BYTES`.

    Dividing matters — `_gemm` iterates `mtiles // mg`, so a non-divisor would
    silently drop the remainder.
    """
    mg = 1
    while (
        mg * 2 <= mtiles
        and mtiles % (mg * 2) == 0
        and (mg * 2) * ktiles * PT * 2 <= MGROUP_BYTES
    ):
        mg *= 2
    return mg


#: Free-axis tile widths to try, widest first. Any divisor of `n` not larger
#: than `NT` is correct; wider is fewer tiles, so fewer instructions.
#:
#: The short list this used to be — `(512, 256, 128)` and then `1` — held a
#: cliff that cost iterations 8 and 9 their 4-rank runs, and it is worth
#: spelling out because nothing about it is visible from the call site. `n` for
#: the unembedding is the vocabulary this rank owns:
#:
#:     TP=1   n = 129280   % 256 == 0   -> nt = 256,  505 tiles
#:     TP=2   n =  64640   % 128 == 0   -> nt = 128,  505 tiles
#:     TP=4   n =  32320   divides none -> nt =   1, **32320 tiles**
#:
#: 32320 is 64 x 505, so it is divisible by 64 and by nothing larger that was on
#: the list. Falling to `nt = 1` multiplies the tile count by 64 and the inner
#: `static_range(ktiles)` emits 40 `nc_matmul`s per tile, so the compiler was
#: asked to schedule 1.29M matmuls for one GEMM. What that produced, measured:
#:
#:     TP=1 head graph   NEFF 3-4 MB,   compile seconds
#:     TP=4 head graph   NEFF **132 MB**, compile **10.5 min**
#:
#: and then the NEFF could not be *loaded* at four ranks: the runtime builds a
#: per-NEFF debug-info graph from the NEFF's own sections
#: (`NDBG:ndebug_info_graph_build`), 132 MB of NEFF came to ~13 GiB of host
#: memory, four ranks did it within 200 ms of each other, and `Committed_AS`
#: hit 112.9 GiB of a 114.8 GiB limit — `std::bad_alloc`, every 4-rank attempt,
#: always just past layer 39. That is the whole failure that eleven device runs
#: in iteration 9 attributed to the collectives' host buffer: it was never the
#: collectives, it was one GEMM's tile width. It is also exactly why TP=2
#: worked untuned — 64640 happens to be 128-aligned and TP=4's shard is not.
#:
#: So the list continues down in powers of two. Adding the smaller candidates
#: cannot change any `n` that already matched a wider one, which is why this is
#: inert at TP=1 and TP=2.
_NTILES = (NT, 256, PT, 64, 32, 16, 8, 4, 2)


def _ntile(n):
    """Free-axis tile width: as wide as a psum bank allows, and a divisor."""
    if n <= NT:
        return n
    for cand in _NTILES:
        if n % cand == 0:
            return cand
    return 1


def _pow2_ceil_log2(x):
    """`2 ** ceil(log2 x)` and its reciprocal, by exponent arithmetic.

    `fast_round_scale` in the reference — this is what `ue8m0` means: the
    activation scale is rounded *up* to a power of two so dequantization is
    exact.
    """
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv


def _qdq_fp8(tile, rows, n):
    """Round `tile` through blockwise fp8 and return it in bf16.

    `vendor/kernel.py`'s `act_quant` with `scale_fmt="ue8m0"`: one scale per 32
    consecutive values, from that block's absolute maximum floored at 1e-4 and
    rounded up to a power of two. The reference runs this before *both* the fp8
    and the fp4 GEMM, so it is part of the arithmetic rather than a storage
    detail — skipping it costs 1.5e-2 of max relative error on a 384-expert MoE
    layer.
    """
    nb = n // BLK
    blocked = tile.reshape((rows, nb, BLK))
    ab = nl.ndarray((rows, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked, bias=0.0, scale=1.0)
    amax = nl.maximum(
        nl.copy(nl.max(ab, axis=2, keepdims=True), dtype=nl.float32), FP8_AMAX_FLOOR
    )
    scale, rcp = _pow2_ceil_log2(nl.multiply(amax, 1.0 / FP8_PEAK))
    rcp_b = nl.copy(rcp, dtype=nl.bfloat16)
    scale_b = nl.copy(scale, dtype=nl.bfloat16)
    q8 = nl.ndarray((rows, nb, BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=q8, data1=blocked, op=nl.multiply, data2=nl.broadcast_to(rcp_b, (rows, nb, BLK))
    )
    deq = nl.ndarray((rows, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=deq, data1=q8, op=nl.multiply, data2=nl.broadcast_to(scale_b, (rows, nb, BLK))
    )
    return deq.reshape((rows, n))


def _decode_e4m3(byte_tile, rows, k):
    """e4m3 bytes to bf16, 2**-120 too small (see `SC_EXP_ADD`)."""
    t = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=t, data=byte_tile, op0=nl.multiply, operand0=BYTE_HI, engine=nisa.engine.scalar
    )
    hi = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=hi.view(nl.uint32),
        data=t.view(nl.uint32),
        op0=nl.bitwise_and,
        operand0=E4M3_MAG2,
        op1=nl.right_shift,
        operand1=4,
        engine=nisa.engine.vector,
    )
    bits = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=bits.view(nl.uint32),
        data=t.view(nl.uint32),
        op0=nl.bitwise_and,
        operand0=E4M3_SGN2,
        op1=nl.bitwise_or,
        operand1=hi.view(nl.uint32),
    )
    return bits.view(nl.bfloat16)


def _dequant_fp8(w_u8, s_u8, dst_hbm, n, k):
    """fp8 `[n, k]` with a `32 x 32`-tile e8m0 scale, into a bf16 HBM scratch.

    The scale grid is blocked on *both* axes, so one scale row covers 32 weight
    rows — hence the inner broadcast. `_gemm` then reads the scratch with
    `dma_transpose`, which needs an HBM source; that is the only reason this
    materializes instead of staying in SBUF.
    """
    kb = k // BLK
    rows = PT if n % PT == 0 else BLK
    nsb = rows // BLK
    for nb in nl.affine_range(n // rows):
        wq = _decode_e4m3(
            nl.load(w_u8[nb * rows : (nb + 1) * rows, 0:k], dtype=nl.uint8), rows, k
        )
        sf = nl.ndarray((rows, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in nl.static_range(nsb):
            sb = nl.load(s_u8[nb * nsb + i : nb * nsb + i + 1, 0:kb], dtype=nl.int32)
            one = nl.copy(nl.left_shift(nl.add(sb, SC_EXP_ADD), 23)).view(nl.float32)
            nisa.tensor_copy(
                dst=sf[i * BLK : (i + 1) * BLK, 0:kb],
                src=nl.broadcast_to(one, (BLK, kb)),
            )
        rep = nl.broadcast_to(sf.reshape((rows, kb, 1)), (rows, kb, BLK))
        prod = nl.ndarray((rows, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=prod, data1=wq, data2=rep.reshape((rows, k)), op=nl.multiply
        )
        nl.store(dst_hbm[nb * rows : (nb + 1) * rows, 0:k], value=prod)


#: Widest n-slab one `_dequant_fp8_k` iteration decodes at once, in elements.
#: The slab is the instruction width of the whole decode, and wide is what
#: matters: every NKI instruction carries a few hundred ns of fixed overhead, so
#: decoding a `[128, 256]` tile per matmul tile instead of a `[128, n]` slab
#: costs the same arithmetic 1.4-2.0x more (measured in the optimization project
#: that preceded this port). The cap is SBUF: `_decode_e4m3` holds three uint16
#: tiles of the slab plus the product, so a slab of 8192 is ~80 KiB of the
#: 192 KiB a partition has, and the engram `wkv`'s n = 25600 would be 250 KiB.
_DQ_SLAB = 8192


def _nslab(n):
    """Widest divisor of `n` at most `_DQ_SLAB` and a multiple of `BLK`.

    A divisor so the loop has no tail, a `BLK` multiple so each slab holds whole
    scale blocks along n.
    """
    if n <= _DQ_SLAB:
        return n
    best = BLK
    for cand in range(BLK, _DQ_SLAB + 1, BLK):
        if n % cand == 0:
            best = cand
    return best


def _dequant_fp8_k(w_u8, s_u8, dst_hbm, k, n):
    """fp8 `[k, n]` **k-major** with a `32 x 32`-tile e8m0 scale, into bf16 HBM.

    **Unreachable: measured end to end and slower.** Kept with its numbers
    because the idea is right and the next attempt should start from what went
    wrong rather than from the idea.

    The same arithmetic as `_dequant_fp8` with the axes swapped: the partition
    axis is the contracted one, so a decoded tile already has the orientation
    `nc_matmul` wants for the moving operand, and `_gemm(..., w_kmajor=True)`
    reads this scratch with a plain `dma_copy` instead of a `dma_transpose`.
    Bit-identical to `_dequant_fp8` + `_gemm` — same bytes, same scale per
    `(n, k-block)`, same `k -> partition` map, so the same products in the same
    order; checked on device at five shapes, 0 of 4,784,128 elements differ.

    **What a four-rank verifier run measured** (long phase, against the
    baseline's own profile, both at 40 layers and 512-row prefill segments):

    | | baseline | k-major |
    |---|---|---|
    | prefill pass, fully traced | 1200.4 / 1171.5 ms | **1141.2 / 1111.5 ms** |
    | prefill pass, the TTFT metric's | 300.4 ms mean | **297.0 ms mean** |
    | decode pass | 394.7 ms | **414.2 ms** |
    | transposed-read volume, decode window | 18.90 GiB | **12.95 GiB** |
    | `sync DMA_DIRECT2D` | 17,462 instr, 7.8 ms | **93,832 instr, 53.1 ms** |

    So the transposing read really does disappear — a third of the window's
    transposed volume is the fp8 weights — and the strided `dma_copy` that
    replaces it costs more in descriptor generation than the transpose cost in
    bandwidth. The arithmetic was not the problem (the dequant opcodes moved
    +1-3% despite 13-17% more instructions from the narrower slabs).

    **What would fix it**, and why it is not a variant of this function: the
    staging DMA exists only because the dequantized weight goes through HBM at
    all. `legacy/moe_short.py:623` (`_gemm_fp8`) decodes a k-major tile **in
    SBUF** and hands it straight to `nc_matmul`, so there is no scratch, no
    staging copy and no transpose anywhere. That needs `_gemm`'s loop nest
    turned inside out (k outer, PSUM banks per n-tile), which is the next piece
    of work; this layout is the half of it that is already proven.

    One scale row covers 32 weight rows on *both* axes, so `nsb = 4` scale rows
    serve one 128-partition k tile, and each of them is replicated across 32
    partitions exactly as `_dequant_fp8` replicates along its own partitions.
    """
    assert n % BLK == 0, f"n={n} must be a multiple of {BLK} for a k-major scale"
    slab = _nslab(n)
    nb = slab // BLK
    nsb = PT // BLK
    for kt in nl.affine_range(k // PT):
        for ns in nl.affine_range(n // slab):
            wq = _decode_e4m3(
                nl.load(
                    w_u8[kt * PT : (kt + 1) * PT, ns * slab : (ns + 1) * slab],
                    dtype=nl.uint8,
                ),
                PT,
                slab,
            )
            sf = nl.ndarray((PT, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i in nl.static_range(nsb):
                sb = nl.load(
                    s_u8[
                        kt * nsb + i : kt * nsb + i + 1,
                        ns * nb : (ns + 1) * nb,
                    ],
                    dtype=nl.int32,
                )
                one = nl.copy(nl.left_shift(nl.add(sb, SC_EXP_ADD), 23)).view(
                    nl.float32
                )
                nisa.tensor_copy(
                    dst=sf[i * BLK : (i + 1) * BLK, 0:nb],
                    src=nl.broadcast_to(one, (BLK, nb)),
                )
            rep = nl.broadcast_to(sf.reshape((PT, nb, 1)), (PT, nb, BLK))
            prod = nl.ndarray((PT, slab), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=prod, data1=wq, data2=rep.reshape((PT, slab)), op=nl.multiply
            )
            nl.store(
                dst_hbm[kt * PT : (kt + 1) * PT, ns * slab : (ns + 1) * slab],
                value=prod,
            )


def _dequant_fp4(w_u8, s_u8, dst_hbm, n, k):
    """fp4 `[n, k//2]` with a per-row, per-32-column e8m0 scale, into bf16 HBM.

    The low nibble of a byte is the lower k index (`weight_loader`'s
    `dequant_block_fp4`, matching `vendor/kernel.py`'s `fp4_gemm`), so the
    nibbles land on a strided destination rather than being de-interleaved
    later. The scale is per weight row here, unlike fp8's 32-row tiles.
    """
    kb = k // BLK
    half = BLK // 2
    rows = PT if n % PT == 0 else BLK
    for nb in nl.affine_range(n // rows):
        pk = nl.load(
            w_u8[nb * rows : (nb + 1) * rows, 0 : k // 2], dtype=nl.uint8
        ).reshape((rows, kb, half))
        # Widened before any bitwise op: `nisa.tensor_scalar_bitvec` requires
        # dst and src to share a dtype, so a uint8 source into a uint16
        # destination fails MLIR verification rather than widening implicitly.
        pk16 = nl.ndarray((rows, kb, half), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=pk16, data=pk, op0=nl.multiply, operand0=1.0, engine=nisa.engine.scalar
        )
        nib = nl.ndarray((rows, kb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=nib[:, :, 0:BLK:2], data=pk16, op0=nl.bitwise_and, operand0=0x0F
        )
        nisa.tensor_scalar(
            dst=nib[:, :, 1:BLK:2],
            data=pk16,
            op0=nl.right_shift,
            operand0=4,
            op1=nl.bitwise_and,
            operand1=0x0F,
        )
        spread = nl.ndarray((rows, kb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=spread,
            data=nib,
            op0=nl.multiply,
            operand0=FP4_NIB_MUL,
            engine=nisa.engine.scalar,
        )
        bits = nl.ndarray((rows, kb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=bits, data=spread, op0=nl.bitwise_and, operand0=FP4_FIELD
        )
        code = nl.load(s_u8[nb * rows : (nb + 1) * rows, 0:kb], dtype=nl.uint8)
        sf = nl.ndarray((rows, kb), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sf,
            data=code,
            op0=nl.multiply,
            operand0=128.0,
            op1=nl.add,
            operand1=float(FP4_EXP_BIAS * 128),
            engine=nisa.engine.scalar,
        )
        rep = nl.broadcast_to(
            sf.view(nl.bfloat16).reshape((rows, kb, 1)), (rows, kb, BLK)
        )
        prod = nl.ndarray((rows, kb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=prod, data1=bits.view(nl.bfloat16), data2=rep, op=nl.multiply
        )
        nl.store(
            dst_hbm[nb * rows : (nb + 1) * rows, 0:k], value=prod.reshape((rows, k))
        )


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_scale=1.0, qdq=False, w_kmajor=False):
    """`dst[m, n] = x[m, k] @ w[n, k].T`, both operands bf16 in HBM.

    `nc_matmul` wants the contracted axis on partitions for both operands, so
    both sides arrive by `dma_transpose`. The staging is double-buffered along
    k (`WLOOK`) and m (`XLOOK`) exactly as `l3_attn._gemm` does; `out_scale`
    folds the fp4 decoder's split exponent back in on the way out.

    `w_kmajor` says the weight is already `[k, n]` — what `_dequant_fp8_k`
    writes — so its tiles need no transpose and arrive by `dma_copy`. Same
    tiles, same `k -> partition` map, same accumulation order: only the DMA
    pattern differs, and a contiguous read of these bytes measured ~4x the
    throughput of the transposing one.
    """
    ktiles = k // PT
    nt = _ntile(n)
    mg = _mgroup(m // PT, ktiles)
    ntiles = n // nt
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=xt[0:PT, mi, kb, 0:PT],
                    src=x_hbm[
                        (mgb * mg + mi) * PT : (mgb * mg + mi + 1) * PT,
                        kb * PT : (kb + 1) * PT,
                    ],
                )

        def ntile(nb, jit):
            # The staged weight's free axis is padded to a 32-byte multiple.
            # `wt[:, kb, :]` sits at byte offset `kb * ntp * 2` inside each
            # partition, and a DMA transpose destination has to be 32B-aligned:
            # an unpadded nt = 24 (the hyper-connection coefficients) or nt = 1
            # (the DSpark confidence head) gives offsets of 48 and 2 bytes and
            # the NEFF fails to *load*, with the cause only in the runtime log —
            #
            #     TDRV:dma_util_expand_desc_transpose  transpose dest offset
            #     139056 must be 32B aligned
            #     RuntimeError: Could not load the model status=1 message=Unknown
            #     Failure
            ntp = -(-nt // 16) * 16
            wt = nl.ndarray((PT, ktiles, ntp), dtype=nl.bfloat16, buffer=nl.sbuf)

            def stage_w(kb):
                if w_kmajor:
                    nisa.dma_copy(
                        dst=wt[0:PT, kb, 0:nt],
                        src=w_hbm[kb * PT : (kb + 1) * PT, nb * nt : (nb + 1) * nt],
                    )
                else:
                    nisa.dma_transpose(
                        dst=wt[0:PT, kb, 0:nt],
                        src=w_hbm[nb * nt : (nb + 1) * nt, kb * PT : (kb + 1) * PT],
                    )

            for kb in nl.static_range(min(WLOOK, ktiles)):
                stage_w(kb)
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    if mi == 0 and kb + WLOOK < ktiles:
                        stage_w(kb + WLOOK)
                    nisa.nc_matmul(
                        dst=acc,
                        stationary=xt[0:PT, mi, kb, 0:PT],
                        moving=wt[0:PT, kb, 0:nt],
                        accumulate=(kb > 0),
                    )
                if out_scale == 1.0:
                    row = nl.copy(acc, dtype=nl.bfloat16)
                else:
                    row = nl.ndarray((PT, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=row, data=acc, op0=nl.multiply, operand0=out_scale
                    )
                nl.store(
                    dst_hbm[
                        (mgb * mg + mi) * PT : (mgb * mg + mi + 1) * PT,
                        nb * nt : (nb + 1) * nt,
                    ],
                    value=_qdq_fp8(row, PT, nt) if qdq else row,
                )

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


def _check(k, n):
    assert k % PT == 0, f"k={k} must be a multiple of {PT}"
    assert k % BLK == 0, f"k={k} must be a multiple of {BLK}"
    assert n % _ntile(n) == 0, f"n={n} has no usable free-axis tile"


def _stage_rows(x, m, k, dst_hbm, quantize):
    """Copy `x[m, k]` into a bf16 `[ceil(m/128)*128, k]` HBM scratch.

    Row padding happens *here*, inside the kernel, rather than in the caller.
    That is not a style choice: a `torch.nn.functional.pad` on the caller's
    side would be an eager aten op on a Neuron tensor, which this backend
    serves through a CPU fallback — a host round trip of the whole activation
    per linear. Everything in this file has to take the shape it is given.

    The pad rows are zeroed and then run through the same quantization as the
    real ones, so they contribute nothing and need no masking downstream; the
    caller slices the first `m` output rows, which is a view.
    """
    full = m // PT
    tail = m - full * PT
    for mb in nl.affine_range(full):
        tile = nl.load(x[mb * PT : (mb + 1) * PT, 0:k], dtype=nl.bfloat16)
        nl.store(
            dst_hbm[mb * PT : (mb + 1) * PT, 0:k],
            value=_qdq_fp8(tile, PT, k) if quantize else tile,
        )
    if tail:
        padded = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padded, 0.0)
        nisa.tensor_copy(
            dst=padded[0:tail, 0:k],
            src=nl.load(x[full * PT : full * PT + tail, 0:k], dtype=nl.bfloat16),
        )
        nl.store(
            dst_hbm[full * PT : (full + 1) * PT, 0:k],
            value=_qdq_fp8(padded, PT, k) if quantize else padded,
        )


# =============================================================================
# Kernels
# =============================================================================


@nki.jit
def fp8_linear(x, w, scale):
    """`x[m, k]` (+ fp8 activation round trip) against fp8 `w[n, k]`.

    `w` is the stored e4m3 bytes viewed as uint8 and `scale` the e8m0 exponent
    bytes; the activation is rounded through fp8 first, which is what
    `vendor/model.py`'s `linear()` does. The returned rows are
    `ceil(m/128)*128`; the caller keeps the first `m`.

    **This is the n-major form, and it is here because the k-major one was
    tried end to end and measured slower.** See `_dequant_fp8_k` for the
    numbers; the short version is that storing the weight `[k, n]` really does
    delete the transposing read (the decode window's transposed-read volume
    fell 18.90 -> 12.95 GiB), and the strided `dma_copy` that replaces it costs
    more in descriptor generation than the transpose cost in bandwidth
    (`sync DMA_DIRECT2D` 17,462 -> 93,832 instructions, 7.8 -> 53.1 ms). The
    layout is only worth having together with the in-SBUF fusion that removes
    the staging DMA altogether — `legacy/moe_short.py:623` — which is the next
    piece of work, not a variant of this one.
    """
    m, k = x.shape
    n = w.shape[0]
    _check(k, n)
    mpad = -(-m // PT) * PT

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xq = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    wd = nl.ndarray((n, k), dtype=nl.bfloat16, buffer=nl.private_hbm)

    _stage_rows(x, m, k, xq, True)
    _dequant_fp8(w, scale, wd, n, k)
    _gemm(xq, wd, out, mpad, k, n)
    return out


@nki.jit
def fp8_linear_raw(x, w, scale):
    """`fp8_linear` without the activation round trip.

    One caller: the block-diagonal `wo_a`. The reference runs that projection
    against the dequantized weight with the activation left alone, so quantizing
    it here would add rounding the reference does not have.
    """
    m, k = x.shape
    n = w.shape[0]
    _check(k, n)
    mpad = -(-m // PT) * PT

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xs = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    wd = nl.ndarray((n, k), dtype=nl.bfloat16, buffer=nl.private_hbm)

    _stage_rows(x, m, k, xs, False)
    _dequant_fp8(w, scale, wd, n, k)
    _gemm(xs, wd, out, mpad, k, n)
    return out


#: SBUF bytes a partition may spend on the staged weight in the fused GEMM.
#: 96 KiB of the 192 KiB a partition has, which leaves the activation tiles
#: (`mg * ktiles * 128 * 2`, up to 40 KiB at the 512-row prefill width) and the
#: decoder's three uint16 temporaries room beside it. Measured footprints at the
#: live shapes are 40-80 KiB.
_WT_BYTES = 96 * 1024


def _nchunk(n, nt, ktiles):
    """Widest multiple of `nt` that divides `n` and fits `_WT_BYTES`.

    This is the fused GEMM's one tuning decision, and it exists because the two
    costs pull opposite ways: the staged weight has to fit SBUF, and the decode
    wants to run on as few, as wide instructions as possible (a marginal NKI
    instruction measures at 130-190 ns on this port, so decoding per *matmul*
    tile — `[128, 256]` — the way `legacy/moe_short.py` does would add ~200 µs
    to a `[128, 5120] x [5120, 2304]` GEMM).

    When nothing wider than one matmul tile fits — the MTP input projection,
    `k = 15360`, is the case: 120 k-tiles leave 409 bytes a partition for the
    chunk — this returns `nt`, and the staged weight is then exactly the size
    `_gemm` already allocates for the same shape. So the floor is the current
    footprint rather than an overflow.
    """
    cap = max(nt, _WT_BYTES // (ktiles * 2))
    best, cand = nt, nt
    while cand <= cap:
        if n % cand == 0:
            best = cand
        cand += nt
    return best


def _decode_fp8_tile(w, s, dst, kb, c0, nc):
    """Decode the k-major fp8 tile `w[kb*128:(kb+1)*128, c0:c0+nc]` into `dst`.

    `_dequant_fp8_k`'s body for one tile, writing SBUF instead of HBM: the
    partition axis is k, so what lands in `dst` is already `nc_matmul`'s moving
    operand and nothing transposes anything.
    """
    nbk = nc // BLK
    nsb = PT // BLK
    byte = nl.load(w[kb * PT : (kb + 1) * PT, c0 : c0 + nc], dtype=nl.uint8)
    wq = _decode_e4m3(byte, PT, nc)
    sf = nl.ndarray((PT, nbk), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in nl.static_range(nsb):
        sb = nl.load(
            s[kb * nsb + i : kb * nsb + i + 1, c0 // BLK : c0 // BLK + nbk],
            dtype=nl.int32,
        )
        one = nl.copy(nl.left_shift(nl.add(sb, SC_EXP_ADD), 23)).view(nl.float32)
        nisa.tensor_copy(
            dst=sf[i * BLK : (i + 1) * BLK, 0:nbk],
            src=nl.broadcast_to(one, (BLK, nbk)),
        )
    rep = nl.broadcast_to(sf.reshape((PT, nbk, 1)), (PT, nbk, BLK))
    nisa.tensor_tensor(
        dst=dst, data1=wq, data2=rep.reshape((PT, nc)), op=nl.multiply
    )


def _stage_x_tile(x_hbm, xt, mgb, mg, ktiles, mi):
    """One m-tile of the activation, transposed into SBUF."""
    for kb in nl.static_range(ktiles):
        nisa.dma_transpose(
            dst=xt[0:PT, mi, kb, 0:PT],
            src=x_hbm[
                (mgb * mg + mi) * PT : (mgb * mg + mi + 1) * PT,
                kb * PT : (kb + 1) * PT,
            ],
        )


def _gemm_fp8_fused(x_hbm, w, s, dst_hbm, m, k, n):
    """`dst[m, n] = x[m, k] @ w[k, n]` with the weight decoded in SBUF.

    The fp8 weight never becomes a bf16 tensor in HBM: each `(k-tile,
    n-chunk)` tile is loaded as bytes, decoded in place and consumed by the
    matmuls of that chunk. So against `_dequant_fp8` + `_gemm` this drops, per
    GEMM, one `n*k*2`-byte write and one `n*k*2`-byte **transposing** read
    (measured at 104-115 GB/s, against ~464 for a contiguous read) and adds one
    `n*k`-byte contiguous read. `legacy/moe_short.py:623` is the same shape.

    Accumulation is unchanged — ascending k into one PSUM bank per `(m-tile,
    n-tile)`, same `k -> partition` map — so the result is bit-identical to the
    two-phase kernels, which is checked rather than argued (§3 of
    `.autohelix/notes/iter-1.md`): 0 of 4,587,520 output elements differ over
    six shapes on device, including the decode width `m=6`, the prefill width
    `m=512` (`mg=4`), `n=512` where the chunk is one matmul tile, and an
    `m=130` row-pad tail.

    **Unreachable, and unmeasured for speed.** It needs the k-major weight that
    commit `bfaf9b1` stored and this iteration reverted, and the one thing
    nobody has yet is a device *timing* of it: the standalone NKI entry point
    cannot be profiled (its runtime is not the one
    `torch.classes.neuron.Runtime().start_profiling` initializes — "Neuron
    runtime must be initialized before starting profiling"), so the only
    instrument is an end-to-end verifier run. Wire it, run it once, and compare
    `device_tpot_ms_long` against 109.63 and `device_ttft_ms_long` against
    1201.67.
    """
    ktiles = k // PT
    nt = _ntile(n)
    mg = _mgroup(m // PT, ktiles)
    nc = _nchunk(n, nt, ktiles)
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for mi in nl.static_range(mg):
            _stage_x_tile(x_hbm, xt, mgb, mg, ktiles, mi)
        for cb in nl.affine_range(n // nc):
            wt = nl.ndarray((PT, ktiles, nc), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in nl.static_range(ktiles):
                _decode_fp8_tile(w, s, wt[0:PT, kb, 0:nc], kb, cb * nc, nc)
            for nb in nl.static_range(nc // nt):
                for mi in nl.static_range(mg):
                    acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(
                            dst=acc,
                            stationary=xt[0:PT, mi, kb, 0:PT],
                            moving=wt[0:PT, kb, nb * nt : (nb + 1) * nt],
                            accumulate=(kb > 0),
                        )
                    nl.store(
                        dst_hbm[
                            (mgb * mg + mi) * PT : (mgb * mg + mi + 1) * PT,
                            cb * nc + nb * nt : cb * nc + (nb + 1) * nt,
                        ],
                        value=nl.copy(acc, dtype=nl.bfloat16),
                    )


@nki.jit
def fp8_linear_fused(x, w, scale):
    """`fp8_linear` with the weight decoded in SBUF — see `_gemm_fp8_fused`.

    `w` is `[k, n]` and `scale` `[k/32, n/32]`: the k-major pair, which is the
    half of commit `bfaf9b1` worth keeping. **Unreachable** until a loader
    stores that layout again.
    """
    m, k = x.shape
    n = w.shape[1]
    assert w.shape[0] == k, f"fp8 weight {w.shape} is not k-major for k={k}"
    _check(k, n)
    mpad = -(-m // PT) * PT
    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xq = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _stage_rows(x, m, k, xq, True)
    _gemm_fp8_fused(xq, w, scale, out, mpad, k, n)
    return out


@nki.jit
def fp8_linear_kmajor(x, w, scale):
    """**Unreachable**, and kept as the measured k-major arm — see
    `_dequant_fp8_k`. Bit-identical to `fp8_linear` on the same weight
    transposed, and slower end to end.

    Takes `w` as `[k, n]` and `scale` as `[k/32, n/32]`, i.e. the transpose of
    what the checkpoint stores. `/tmp/it1/ab8.py` is the harness that proved the
    identity (0 of 4,784,128 elements differ over five shapes); the model-side
    half of the experiment — `BlockQuantLinear._specs`, `shard_parameter`'s
    one-line transpose, `verify_shard_plan`'s reversed expectation and
    `grouped()`'s second-axis slice — is in commit `bfaf9b1` if it is wanted
    back.
    """
    m, k = x.shape
    n = w.shape[1]
    assert w.shape[0] == k, f"fp8 weight {w.shape} is not k-major for k={k}"
    _check(k, n)
    mpad = -(-m // PT) * PT

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xq = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    wd = nl.ndarray((k, n), dtype=nl.bfloat16, buffer=nl.private_hbm)

    _stage_rows(x, m, k, xq, True)
    _dequant_fp8_k(w, scale, wd, k, n)
    _gemm(xq, wd, out, mpad, k, n, w_kmajor=True)
    return out


@nki.jit
def fp4_linear(x, w, scale):
    """`x[m, k]` (+ fp8 activation round trip) against fp4 `w[n, k//2]`.

    The routed experts' GEMM. Same activation handling as `fp8_linear` — the
    reference quantizes the activation for the fp4 path too.
    """
    m, k = x.shape
    n = w.shape[0]
    assert w.shape[1] * 2 == k, f"fp4 weight {w.shape} does not pack k={k}"
    _check(k, n)
    mpad = -(-m // PT) * PT

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xq = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    wd = nl.ndarray((n, k), dtype=nl.bfloat16, buffer=nl.private_hbm)

    _stage_rows(x, m, k, xq, True)
    _dequant_fp4(w, scale, wd, n, k)
    _gemm(xq, wd, out, mpad, k, n, out_scale=FP4_POST)
    return out


@nki.jit
def fp32w_linear(x, w):
    """`x[m, k] @ w[n, k].T` where `w` is stored **fp32**, narrowed in-kernel.

    One caller: the hyper-connection coefficient projection, the only fp32
    weight in the checkpoint. It cannot be narrowed on the caller's side — a
    `Tensor.to(torch.bfloat16)` on a Neuron tensor is served by the backend's
    CPU fallback, which does not implement dtype conversion at all
    (`RuntimeError: Expected self.dtype() == dst.dtype()`), and narrowing it
    inside the traced `forward` would be a graph side effect that makes the
    first call trace differently from every later one.

    `w` is staged through a bf16 scratch rather than fed to `_gemm` directly
    because `dma_transpose` cannot convert. That costs one copy of the weight,
    which is fine at `[24, 20480]` and is why `bf16_linear` — whose callers
    include the 1.3 GiB unembedding — does not do it.
    """
    m, k = x.shape
    n = w.shape[0]
    _check(k, n)
    mpad = -(-m // PT) * PT
    npad = -(-n // PT) * PT
    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xs = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ws = nl.ndarray((npad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _stage_rows(x, m, k, xs, False)
    _stage_rows(w, n, k, ws, False)
    _gemm(xs, ws, out, mpad, k, n)
    return out


@nki.jit
def bf16_linear(x, w):
    """`x[m, k] @ w[n, k].T`, both bf16, no quantization anywhere.

    The compressor's `wkv`/`wgate`, the indexer's `wk` and `weights_proj`, the
    router and the unembedding: the checkpoint stores these unquantized and the
    reference runs them without an activation round trip, so neither happens
    here. `x` may arrive fp32 (the compressor pools in fp32) and is narrowed on
    the way into the staging scratch, since `dma_transpose` cannot convert.
    """
    m, k = x.shape
    n = w.shape[0]
    _check(k, n)
    mpad = -(-m // PT) * PT
    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xs = nl.ndarray((mpad, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _stage_rows(x, m, k, xs, False)
    _gemm(xs, w, out, mpad, k, n)
    return out
