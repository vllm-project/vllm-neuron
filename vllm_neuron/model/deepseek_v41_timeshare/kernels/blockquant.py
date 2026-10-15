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


# =============================================================================
# fp4, fused — the routed experts' GEMM with no dequantized weight in HBM
# =============================================================================
#
# `fp4_linear` above writes the whole bf16 weight to a `private_hbm` scratch and
# `_gemm` reads it back through a `dma_transpose`. Measured in the baseline
# gate's own decode profile, per routed-expert execution (three GEMMs,
# 35.4M weight values): **engine 917 µs, device DMA 904 µs, union 1117 µs**, and
# the DMA is three transposing reads of the dequantized bf16 weight, 22.5 MiB
# each at 104-115 GB/s = 655 µs. Each weight value costs 4.5 bytes of traffic
# (0.5 stored + 2 written + 2 read back) where 0.5 is what it is worth.
#
# The kernels below move the transpose to the *packed* bytes, viewed as uint16
# and transposed by DMA straight into SBUF (16.9 MiB instead of 67.5), decode
# there and hand `nc_matmul` the result. No scratch, no staging, no bf16 weight
# in HBM. What that buys, from the same profile's per-opcode table:
#
#   | per execution                        | `fp4_linear` | fused |
#   |--------------------------------------|---|---|
#   | transposing weight DMA               | 67.5 MiB | 16.9 MiB |
#   | dequantized-weight writes            | 67.5 MiB | none |
#   | nibble extract (Vector)              | 180 µs, strided dst, 2x | 4 ops, 32-bit, de-interleaved |
#   | uint8 -> uint16 widen (Scalar)       | 129 µs | gone: the extract is already uint16 |
#   | 32-fold scale broadcast COPY (Vector)| 156 µs | gone: the scale lands per 8 partitions |
#
# The price is the accumulation order: a `[128, nt]` moving tile now carries k
# values `4r + c` for one `c`, so the contraction is four interleaved matmul
# chains instead of one over 128 consecutive k. Same products, different fp32
# order, so this is **not bit-exact** against `fp4_linear` — it is the one
# deliberate numerical change. `_dequant_fp4`'s own arithmetic is reproduced
# value for value (same nibble, same `FP4_NIB_MUL`, same `FP4_FIELD`, same
# `2**(code-65)`), and that is checked rather than argued: under `nki.simulate`
# against the shipped `fp4_linear`, **1 of 32,768 output elements differs at
# `m=128 k=512 n=256`, 4 of 32,768 at `k=1152` (which exercises the partial last
# k-tile) and 0 of 2,304 at the decode width `m=6`** — one bf16 step each, from
# the accumulation order and nothing else. On device at the six live expert
# shapes (`k=5120 n=2304` and `k=2304 n=5120`, `m` 128/130/256/512) the result
# agrees with a float64 emulation to **2.1-3.3e-3 relative**, which is the bf16
# output quantum (2**-8 = 3.9e-3), i.e. as close as the dtype allows.

#: The low nibble of each uint16 lane of a uint32, i.e. two nibbles a lane-op.
#: Byte 0 of a packed pair holds k = 4r (low nibble) and 4r+1 (high), byte 1
#: holds 4r+2 and 4r+3, so nibble `c` of the pair is k = 4r + c at bits 4c..4c+3
#: — and a right shift inside the uint32 brings lane 1's nibble to bits 16..19
#: while this mask drops what crossed the lane boundary.
FP4_NIB_MASK2 = 0x000F000F
#: `FP4_FIELD` doubled and signed: NKI takes a 32-bit bitwise operand as int32,
#: the same way `E4M3_SGN2` is written.
FP4_FIELD2 = 0x81C081C0 - (1 << 32)

#: PSUM holds 8 banks of 2 KiB a partition. An accumulation chain has to stay
#: in one place for the whole k loop, so the chunking below is sized to keep
#: `mtiles * (nc // nt)` live accumulators inside both of these.
_PSUM_TILES = 6
_PSUM_FP32 = 6 * 2048 // 4


def _fp4_plan(n, mtiles):
    """`(nt, nc)`: the PSUM tile width and the n-chunk the k loop runs over.

    `nt` is the widest divisor of `n` a PSUM bank holds (512 fp32), which is
    not `_ntile`'s power-of-two list: the MoE intermediate 2304 divides by 384
    and not by 512, and taking 384 over 256 is what turns a decode step's `w1`
    from three chunks into one.
    `nc` is how much of `n` one pass over k covers, and it is as wide as PSUM
    allows: widening it does not change any volume — every per-chunk cost here
    is proportional to bytes — but it does spread the ~150 ns fixed cost of a
    NKI instruction over a wider decode. At the live shapes: `nc = n` for `w1`
    and `w3` on a decode step (one chunk, six 384-wide accumulators), `n/2` for
    `w2`, and `nc = nt` at a 512-row prefill segment, where four m-tiles leave
    room for one accumulator each.
    """
    nt = 1
    for cand in range(1, NT + 1):
        if n % cand == 0:
            nt = cand
    room = min(max(1, _PSUM_TILES // mtiles), max(1, _PSUM_FP32 // (mtiles * nt)))
    best = 1
    for cand in range(1, room + 1):
        if n % (nt * cand) == 0:
            best = cand
    return nt, nt * best


def _stage_rows_fp4(x, m, k, dst_hbm):
    """`_stage_rows(..., quantize=True)` with the k axis de-interleaved.

    Writes `dst[j, c, r] = xq[j, 4r + c]`, so the activation tile that pairs
    with nibble `c` of a transposed weight pair is a *contiguous* `[128, 128]`
    block of HBM and `dma_transpose` can take it as it stands. The split has to
    happen after `_qdq_fp8`, because the quantization block is 32 consecutive
    k of the original order; doing it in SBUF against a stride-4 view costs four
    `tensor_copy`s over `k/4` values each, ~3 µs at the live shapes.
    """
    kq = k // 4
    full = m // PT
    tail = m - full * PT
    for mb in nl.affine_range(full):
        tile = _qdq_fp8(nl.load(x[mb * PT : (mb + 1) * PT, 0:k], dtype=nl.bfloat16), PT, k)
        blocked = tile.reshape((PT, kq, 4))
        half = nl.ndarray((PT, kq), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(4):
            nisa.tensor_copy(dst=half, src=blocked[0:PT, 0:kq, c])
            nl.store(dst_hbm[mb * PT : (mb + 1) * PT, c, 0:kq], value=half)
    if tail:
        padded = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padded, 0.0)
        nisa.tensor_copy(
            dst=padded[0:tail, 0:k],
            src=nl.load(x[full * PT : full * PT + tail, 0:k], dtype=nl.bfloat16),
        )
        tile = _qdq_fp8(padded, PT, k)
        blocked = tile.reshape((PT, kq, 4))
        half = nl.ndarray((PT, kq), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(4):
            nisa.tensor_copy(dst=half, src=blocked[0:PT, 0:kq, c])
            nl.store(dst_hbm[full * PT : (full + 1) * PT, c, 0:kq], value=half)


def _fp4_decode_tile(wb, scb, dst, nc, c):
    """Nibble `c` of every packed pair in `wb[*, nc]`, dequantized into `dst`.

    `wb` is a transposed tile of the stored bytes viewed as uint16, so a
    partition is one pair — four k values — and `dst` comes out as the moving
    operand `nc_matmul` wants for the k values `4r + c`.

    Four instructions, against `_dequant_fp4`'s seven for the same values:
    the extract is 32-bit and its destination contiguous (where `_dequant_fp4`
    writes `nib[:, :, c::2]` and pays 2x instead of 4x), the multiply by
    `FP4_NIB_MUL` widens nothing because the extract already landed in uint16
    (`_dequant_fp4` pays a separate 129 µs Scalar pass to widen uint8 first),
    and the scale multiply reads `scb` element for element, where
    `_dequant_fp4` has to materialize a 32-fold broadcast first — a 156 µs
    Vector COPY an execution.
    """
    nib = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    if c == 0:
        nisa.tensor_scalar(
            dst=nib.view(nl.uint32),
            data=wb.view(nl.uint32),
            op0=nl.bitwise_and,
            operand0=FP4_NIB_MASK2,
            engine=nisa.engine.vector,
        )
    else:
        nisa.tensor_scalar(
            dst=nib.view(nl.uint32),
            data=wb.view(nl.uint32),
            op0=nl.right_shift,
            operand0=4 * c,
            op1=nl.bitwise_and,
            operand1=FP4_NIB_MASK2,
            engine=nisa.engine.vector,
        )
    pre = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=pre, data=nib, op0=nl.multiply, operand0=FP4_NIB_MUL,
        engine=nisa.engine.scalar,
    )
    # A separate destination for the field mask, not `dst` in place: a
    # `tensor_tensor` whose `dst` and `data1` are two different dtype views of
    # one tile reads as zeros (checked under `nki.simulate`), and iteration 4 of
    # the previous run measured an in-place uint32 mask at 26.8 -> 390.5 µs.
    bits = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bits.view(nl.uint32),
        data=pre.view(nl.uint32),
        op0=nl.bitwise_and,
        operand0=FP4_FIELD2,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=dst,
        data1=bits.view(nl.bfloat16),
        data2=scb.view(nl.bfloat16),
        op=nl.multiply,
    )


def _fp4_scale_rows(s16, srep, n, k):
    """`srep[r, j] = s[j, r // 8]` — the e8m0 code of the 32-value block that
    k-pair `r` belongs to, k-major, one byte a pair.

    This exists because of a hardware rule rather than a preference. A
    transposed `[128 pairs, nc]` weight tile has one scale block spanning eight
    partitions, so the natural thing is to broadcast each code over its eight
    partitions in SBUF — and a compute instruction's partition base has to be
    0, 32, 64 or 96, so writing eight partitions at base 8 fails BIR
    verification outright:

        Invalid access of 8 partitions starting at partition 8

    A DMA has no such rule, so the replication is done once per GEMM through
    HBM: `(k/4) * n` bytes written and read back, 2.95 MB and ~15 µs at the
    live expert shapes, against the 22.5 MiB of transposing weight reads this
    whole kernel exists to delete. Per `(k-tile, n-chunk)` the k loop then
    spends one contiguous load and one `tensor_scalar` where the SBUF route
    would have spent nineteen instructions.

    `srep` is `[k//64, 2, 8, n]`: scale pair, which of its two bytes, and the
    eight pairs that share the code. Flattened that is exactly the `[k//4, n]`
    the k loop indexes, because pair `r` carries block `r // 8`.
    """
    npair = k // 64
    tw = _ntile(n)
    for tb in range(-(-npair // 64)):
        p0 = tb * 64
        pn = min(64, npair - p0)
        sp = nl.ndarray((64, n), dtype=nl.uint16, buffer=nl.sbuf)
        for q in range(n // tw):
            nisa.dma_transpose(
                dst=sp[0:pn, q * tw : (q + 1) * tw],
                src=s16[q * tw : (q + 1) * tw, p0 : p0 + pn],
            )
        spu = sp.view(nl.uint8)
        for h in range(2):
            half = nl.ndarray((64, n), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_copy(dst=half[0:pn, 0:n], src=spu[0:pn, h : 2 * n : 2])
            for i in range(8):
                nl.store(srep[p0 : p0 + pn, h, i, 0:n], value=half[0:pn, 0:n])


def _gemm_fp4_fused(x4_hbm, w16, srep, dst_hbm, m, k, n):
    """`dst[m, n] = x[m, k] @ w[n, k].T` with the fp4 weight decoded in SBUF.

    `w16` is the stored `[n, k//2]` bytes viewed as uint16 and `s16` the
    `[n, k//32]` e8m0 bytes viewed the same way; both are read by
    `dma_transpose`, which is why they have to be two bytes wide (its only hard
    dtype constraint). A transposed weight tile is `[128 pairs, nc]`, so a
    partition carries four k values and one 32-value scale block spans eight
    partitions — which is where the scale broadcast disappears to.

    The k loop is outer and PSUM holds every accumulator across it, so each
    weight tile is read, decoded and consumed once. `k // 4` need not be a
    multiple of 128: the last tile takes the remainder (`k = 2304`, the MoE
    intermediate, leaves 64 of 576), and `k % 64 == 0` is all this needs.
    """
    kq = k // 4
    rfull = kq // PT
    rrem = kq - rfull * PT
    rtiles = rfull
    if rrem:
        rtiles = rfull + 1
    mtiles = m // PT
    nt, nc = _fp4_plan(n, mtiles)
    ntiles = nc // nt

    at = nl.ndarray((PT, 4, rtiles, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    for mi in range(mtiles):
        for c in range(4):
            for rb in range(rtiles):
                part = PT
                if rb >= rfull:
                    part = rrem
                nisa.dma_transpose(
                    dst=at[0:part, c, rb, mi * PT : (mi + 1) * PT],
                    src=x4_hbm[
                        mi * PT : (mi + 1) * PT, c, rb * PT : rb * PT + part
                    ],
                )

    for cb in nl.affine_range(n // nc):
        accs = []
        for mi in range(mtiles):
            row = []
            for nb in range(ntiles):
                row.append(nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum))
            accs.append(row)
        for rb in range(rtiles):
            part = PT
            if rb >= rfull:
                part = rrem
            wb = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
            for q in range(ntiles):
                lo = cb * nc + q * nt
                nisa.dma_transpose(
                    dst=wb[0:part, q * nt : (q + 1) * nt],
                    src=w16[lo : lo + nt, rb * PT : rb * PT + part],
                )
            # `srep` already carries one e8m0 code per k-pair, so the scale
            # arrives as a plain contiguous load and the whole 32-fold broadcast
            # `_dequant_fp4` pays for (a 156 µs Vector COPY an execution) is
            # gone. See `_fp4_scale_rows` for why it has to come from HBM.
            cod = nl.load(
                srep[rb * PT : rb * PT + part, cb * nc : (cb + 1) * nc],
                dtype=nl.uint8,
            )
            scb = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scb[0:part, 0:nc],
                data=cod,
                op0=nl.multiply,
                operand0=128.0,
                op1=nl.add,
                operand1=float(FP4_EXP_BIAS * 128),
                engine=nisa.engine.scalar,
            )
            for c in range(4):
                dqf = nl.ndarray((PT, nc), dtype=nl.bfloat16, buffer=nl.sbuf)
                _fp4_decode_tile(wb, scb, dqf, nc, c)
                for nb in range(ntiles):
                    for mi in range(mtiles):
                        nisa.nc_matmul(
                            dst=accs[mi][nb],
                            stationary=at[0:part, c, rb, mi * PT : (mi + 1) * PT],
                            moving=dqf[0:part, nb * nt : (nb + 1) * nt],
                            accumulate=(rb > 0 or c > 0),
                        )
        for mi in range(mtiles):
            for nb in range(ntiles):
                row = nl.ndarray((PT, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=row, data=accs[mi][nb], op0=nl.multiply, operand0=FP4_POST
                )
                nl.store(
                    dst_hbm[
                        mi * PT : (mi + 1) * PT,
                        cb * nc + nb * nt : cb * nc + (nb + 1) * nt,
                    ],
                    value=row,
                )


@nki.jit
def fp4_linear_fused(x, w, scale):
    """`fp4_linear` with the weight transposed as bytes and decoded in SBUF.

    Same stored form, same call signature: `w` is `[n, k//2]` uint8 and `scale`
    `[n, k//32]` uint8, exactly as the checkpoint has them and exactly as
    `fp4_linear` takes them, so nothing in the loader or the shard plan moves.
    The only difference a caller can see is the last bit or two of the result —
    see the comment above `FP4_NIB_MASK2` for why that is unavoidable here.

    **Faster, measured, and unreachable — and the three facts are not in
    tension.** Iteration 2 wired this into `nki_linear`, ran the gate's own
    verifier at four ranks, and got:

    | | `fp4_linear` | this |
    |---|---|---|
    | routed-expert engine, **per execution** | 917.2 µs | **623.7** |
    | — Vector / Scalar / Tensor / GpSimd | 619 / 456 / 293 / 234 | **420 / 412 / 212 / 95** |
    | `device_step_ms_long` | 394.653 | **354.984** (−10.05%) |
    | `device_step_ms_short` | 437.103 | **380.864** (−12.87%) |
    | `device_tpot_ms_short` | 149.129 | **125.461** (−15.87%) |
    | `device_ttft_ms_long` | 1183 | 1192.560 (+0.81%) |
    | `device_tpot_ms_long` | 109.626 | 125.962 (**+14.90%**) |
    | `device_ttft_ms_short` | 779.179 | 830.653 (**+6.61%**) |

    The verifier *passed* with it in (4/4 `REASONABLE`, the needle's access code
    correct). The two numbers that moved the wrong way did so for reasons that
    are not speed, both diagnosed in `.autohelix/notes/iter-2.md` §4:

      * `tpot = step / tokens_per_step`, and the divisor fell 3.6 → 2.818
        because the needle case emitted 32 tokens where the record-holding run
        emitted 19. The first five decode steps accepted 12 of 25 drafts — the
        baseline's own rate — so this is EOS placement, not a worse drafter.
        **To land this kernel the decode pass has to reach 324 ms**
        (= 115.108 × 2.818), from 355.
      * `ttft_short` is half a *fully traced* prefill pass, and this kernel
        emits ~1 261 activity rows an execution where `fp4_linear` emits 2 230.
        Freeing that much NRT trace budget makes **untouched** graphs get priced
        more completely: in the same window `69d801c7`'s worst-traced execution
        went 5 044 → 40 533 rows at an unchanged 15 executions and its measured
        engine time rose 28%, while `544192c5`, whose coverage did not move,
        did not move in time either (0.08%).

    So wire it together with the fp8 kernel in the same shape (notes §5.2,
    worth ~39 ms of scale plumbing a decode step), not on its own.
    """
    m, k = x.shape
    n = w.shape[0]
    assert w.shape[1] * 2 == k, f"fp4 weight {w.shape} does not pack k={k}"
    assert k % 64 == 0, f"k={k} must be a multiple of 64 for the uint16 view"
    _check(k, n)
    mpad = -(-m // PT) * PT

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    x4 = nl.ndarray((mpad, 4, k // 4), dtype=nl.bfloat16, buffer=nl.private_hbm)
    srep = nl.ndarray((k // 64, 2, 8, n), dtype=nl.uint8, buffer=nl.private_hbm)

    _stage_rows_fp4(x, m, k, x4)
    _fp4_scale_rows(scale.view(nl.uint16), srep, n, k)
    _gemm_fp4_fused(
        x4, w.view(nl.uint16), srep.reshape((k // 4, n)), out, mpad, k, n
    )
    return out


# =============================================================================
# fp8, fused — the same trick one size up, and for a different reason
# =============================================================================
#
# `fp8_linear`'s problem is not its DMA. Measured on the baseline decode window
# (`an3.py`, shared-expert graph `9539a540`, 40 executions, engine 1059.8 µs
# against a 1124 µs union — so only 64 µs of DMA is even exposed), **601 µs of
# that engine budget is scale plumbing**, for 11,520 distinct scale values:
#
#   | per execution                                           | engine | µs |
#   |---|---|---|
#   | `MATMUL` + `LDWEIGHTS` (the floor)                      | Tensor | 326 + 97 |
#   | the `32 x 32` grid -> a `[128, k]` replica: `left_shift/add` (304 instr), the `view`/`CAST`, and the 32-partition `broadcast_to` that lowers to `STREAM_SHUFFLE` | Sc+Ve+Gp | **445** |
#   | the 32-fold free-axis `broadcast_to` materializing as a `COPY` | Vector | **156** |
#   | the scale multiply itself                               | Vector | 156 |
#   | `_decode_e4m3`: the uint8->uint16 widen, the magnitude shift, the sign `or` | Sc+Ve | 245 + 86 + 156 |
#
# `Vector` at 818 µs is what binds. The kernels below delete the 601 µs: the
# scale arrives as one contiguous byte a partition out of an HBM replica
# (`_fp8_scale_rows`), and the e4m3 byte is selected by one uint32-wide mask
# instead of a Scalar multiply by 256 — which lands it exactly where
# `E4M3_MAG2`/`E4M3_SGN2` already expect it, so `_decode_e4m3`'s other two
# instructions are reused verbatim.
#
# Two k values a partition, so the contraction splits in two (against fp4's
# four), and as in the fp4 case that makes the result **not bit-exact** against
# `fp8_linear` — same products, different fp32 order.
#
# **Unreachable.** `nki_linear` still dispatches `"fp8"` to `fp8_linear`, for
# the same reason the fp4 one is unreachable: see `fp4_linear_fused`'s docstring
# and `.autohelix/notes/iter-2.md` §4. Wire the two together.

#: Select byte 1 of each uint16 lane, in place (signed, as `E4M3_SGN2` is).
FP8_HI_MASK2 = 0xFF00FF00 - (1 << 32)
#: Select byte 0 of each uint16 lane; a following `<< 8` moves it to byte 1,
#: which is where `_decode_e4m3`'s masks expect the e4m3 byte to sit.
FP8_LO_MASK2 = 0x00FF00FF


def _fp8_scale_rows(s16, srep, n, k, ns):
    """`srep[r, j] = scale[j // 32, r // 16]`, the e8m0 code of the `32 x 32`
    block that k-pair `r` and output `j` belong to, k-major, one byte a pair.

    The fp8 grid is blocked on *both* axes, which is the only thing harder here
    than in `_fp4_scale_rows`: a code has to be replicated 16x along k (one
    scale block is 32 k = 16 pairs) and 32x along n. The k replication goes
    through HBM because a compute instruction's SBUF partition base must be 0,
    32, 64 or 96 and 16-partition groups at base 16 are illegal; the n
    replication is a free stride-0 `broadcast_to` on the free axis.

    `s16` is `scale.view(nl.uint16)`, i.e. `[ns, kb/2]` with two k-blocks a
    uint16, and `srep` is `[kb/2, 2, 16, n]` — scale pair, which of its two
    bytes, and the 16 k-pairs that share the code — which flattens to exactly
    the `[k/2, n]` the k loop indexes.

    **Validated on device**, bit-exact against
    `scale[j // 32, r // 16]` at `(n, k)` = (2304, 5120), (5120, 2304) and
    (8192, 1280): 0 of 5.9M / 5.9M / 5.2M bytes differ (`/tmp/it2/fp8scale.py`).
    """
    npair = (k // 32) // 2
    tw = _ntile(ns)
    for tb in range(-(-npair // 64)):
        p0 = tb * 64
        pn = min(64, npair - p0)
        sp = nl.ndarray((64, ns), dtype=nl.uint16, buffer=nl.sbuf)
        for q in range(ns // tw):
            nisa.dma_transpose(
                dst=sp[0:pn, q * tw : (q + 1) * tw],
                src=s16[q * tw : (q + 1) * tw, p0 : p0 + pn],
            )
        spu = sp.view(nl.uint8)
        dei = nl.ndarray((PT, ns), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dei[0:pn, 0:ns], src=spu[0:pn, 0 : 2 * ns : 2])
        nisa.tensor_copy(dst=dei[64 : 64 + pn, 0:ns], src=spu[0:pn, 1 : 2 * ns : 2])
        for h in range(2):
            rep = nl.ndarray((PT, ns, 32), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=rep[0:pn, 0:ns, 0:32],
                src=nl.broadcast_to(
                    dei[h * 64 : h * 64 + pn, 0:ns].reshape((pn, ns, 1)),
                    (pn, ns, 32),
                ),
            )
            flat = rep.reshape((PT, ns * 32))
            for i in range(16):
                nl.store(srep[p0 : p0 + pn, h, i, 0:n], value=flat[0:pn, 0:n])


def _stage_rows_fp8k(x, m, k, dst_hbm, quantize):
    """`_stage_rows` with the k axis split by `k % 2`.

    Writes `dst[j, c, r] = xq[j, 2r + c]`, so the activation tile that pairs
    with byte `c` of a transposed uint16 weight pair is a contiguous
    `[128, 128]` block of HBM.
    """
    kh = k // 2
    full = m // PT
    tail = m - full * PT
    for mb in nl.affine_range(full):
        tile = nl.load(x[mb * PT : (mb + 1) * PT, 0:k], dtype=nl.bfloat16)
        if quantize:
            tile = _qdq_fp8(tile, PT, k)
        blocked = tile.reshape((PT, kh, 2))
        half = nl.ndarray((PT, kh), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(2):
            nisa.tensor_copy(dst=half, src=blocked[0:PT, 0:kh, c])
            nl.store(dst_hbm[mb * PT : (mb + 1) * PT, c, 0:kh], value=half)
    if tail:
        padded = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padded, 0.0)
        nisa.tensor_copy(
            dst=padded[0:tail, 0:k],
            src=nl.load(x[full * PT : full * PT + tail, 0:k], dtype=nl.bfloat16),
        )
        tile = _qdq_fp8(padded, PT, k) if quantize else padded
        blocked = tile.reshape((PT, kh, 2))
        half = nl.ndarray((PT, kh), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(2):
            nisa.tensor_copy(dst=half, src=blocked[0:PT, 0:kh, c])
            nl.store(dst_hbm[full * PT : (full + 1) * PT, c, 0:kh], value=half)


def _fp8_decode_tile(wb, scb, dst, nc, c):
    """Byte `c` of every packed pair in `wb[*, nc]`, dequantized into `dst`.

    Four instructions, and the middle two are `_decode_e4m3`'s own: the mask
    that selects the byte also leaves it in bits 8..15 of its uint16 lane,
    which is precisely the position `_decode_e4m3` gets by multiplying a uint8
    by 256 on the Scalar engine — 245 µs an execution that this replaces with a
    32-bit Vector op.
    """
    t = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    if c:
        nisa.tensor_scalar(
            dst=t.view(nl.uint32), data=wb.view(nl.uint32),
            op0=nl.bitwise_and, operand0=FP8_HI_MASK2, engine=nisa.engine.vector,
        )
    else:
        nisa.tensor_scalar(
            dst=t.view(nl.uint32), data=wb.view(nl.uint32),
            op0=nl.bitwise_and, operand0=FP8_LO_MASK2,
            op1=nl.left_shift, operand1=8, engine=nisa.engine.vector,
        )
    hi = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=hi.view(nl.uint32), data=t.view(nl.uint32),
        op0=nl.bitwise_and, operand0=E4M3_MAG2,
        op1=nl.right_shift, operand1=4, engine=nisa.engine.vector,
    )
    bits = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=bits.view(nl.uint32), data=t.view(nl.uint32),
        op0=nl.bitwise_and, operand0=E4M3_SGN2,
        op1=nl.bitwise_or, operand1=hi.view(nl.uint32),
    )
    nisa.tensor_tensor(
        dst=dst, data1=bits.view(nl.bfloat16), data2=scb.view(nl.bfloat16),
        op=nl.multiply,
    )


#: SBUF a partition may spend on the resident activation tiles of the fused fp8
#: GEMM (`2 * rtiles * m * 2` bytes). The MTP input projection, `k = 15360`, is
#: 60 k-pair tiles and so needs 30 KiB at `m = 128` and 120 KiB at `m = 512` —
#: the second does not fit beside the decode tiles, and the fix is an outer
#: m-group loop that re-reads the weight per group, which is not written. Until
#: it is, `fp8_linear_fused` asserts and the caller dispatches by shape.
_FP8_AT_BYTES = 64 * 1024


def _gemm_fp8_tfused(x2_hbm, w16, srep, dst_hbm, m, k, n):
    """`dst[m, n] = x[m, k] @ w[n, k].T` with the fp8 weight decoded in SBUF.

    `_gemm_fp4_fused` with two lanes instead of four: `w16` is the stored
    `[n, k]` e4m3 bytes viewed as uint16, so a transposed `[128, nc]` tile puts
    one *pair* of k values on each partition and the contraction is two
    interleaved chains. `srep` is `_fp8_scale_rows`' `[k/2, n]` replica.
    """
    kh = k // 2
    rfull = kh // PT
    rrem = kh - rfull * PT
    rtiles = rfull
    if rrem:
        rtiles = rfull + 1
    mtiles = m // PT
    nt, nc = _fp4_plan(n, mtiles)
    ntiles = nc // nt
    assert 2 * rtiles * m * 2 <= _FP8_AT_BYTES, (
        f"activation tiles are {2 * rtiles * m * 2} bytes a partition for "
        f"k={k} m={m}; see _FP8_AT_BYTES"
    )

    at = nl.ndarray((PT, 2, rtiles, m), dtype=nl.bfloat16, buffer=nl.sbuf)
    for mi in range(mtiles):
        for c in range(2):
            for rb in range(rtiles):
                part = PT
                if rb >= rfull:
                    part = rrem
                nisa.dma_transpose(
                    dst=at[0:part, c, rb, mi * PT : (mi + 1) * PT],
                    src=x2_hbm[mi * PT : (mi + 1) * PT, c, rb * PT : rb * PT + part],
                )

    for cb in nl.affine_range(n // nc):
        accs = []
        for mi in range(mtiles):
            row = []
            for nb in range(ntiles):
                row.append(nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum))
            accs.append(row)
        for rb in range(rtiles):
            part = PT
            if rb >= rfull:
                part = rrem
            wb = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
            for q in range(ntiles):
                lo = cb * nc + q * nt
                nisa.dma_transpose(
                    dst=wb[0:part, q * nt : (q + 1) * nt],
                    src=w16[lo : lo + nt, rb * PT : rb * PT + part],
                )
            cod = nl.load(
                srep[rb * PT : rb * PT + part, cb * nc : (cb + 1) * nc],
                dtype=nl.uint8,
            )
            scb = nl.ndarray((PT, nc), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scb[0:part, 0:nc], data=cod,
                op0=nl.multiply, operand0=128.0,
                op1=nl.add, operand1=float(SC_EXP_ADD * 128),
                engine=nisa.engine.scalar,
            )
            for c in range(2):
                dqf = nl.ndarray((PT, nc), dtype=nl.bfloat16, buffer=nl.sbuf)
                _fp8_decode_tile(wb, scb, dqf, nc, c)
                for nb in range(ntiles):
                    for mi in range(mtiles):
                        nisa.nc_matmul(
                            dst=accs[mi][nb],
                            stationary=at[0:part, c, rb, mi * PT : (mi + 1) * PT],
                            moving=dqf[0:part, nb * nt : (nb + 1) * nt],
                            accumulate=(rb > 0 or c > 0),
                        )
        for mi in range(mtiles):
            for nb in range(ntiles):
                nl.store(
                    dst_hbm[
                        mi * PT : (mi + 1) * PT,
                        cb * nc + nb * nt : cb * nc + (nb + 1) * nt,
                    ],
                    value=nl.copy(accs[mi][nb], dtype=nl.bfloat16),
                )


@nki.jit
def fp8_linear_tfused(x, w, scale, quantize=True):
    """`fp8_linear` with the weight transposed as bytes and decoded in SBUF.

    Same stored form and same signature as `fp8_linear` / `fp8_linear_raw`:
    `w` is `[n, k]` uint8 e4m3 and `scale` the `[ceil(n/32), ceil(k/32)]` e8m0
    grid, so nothing in the loader or the shard plan moves. `quantize=False` is
    `fp8_linear_raw`'s behaviour, for the block-diagonal `wo_a`.

    **Unreachable, and validated for correctness but never timed end to end.**
    The model has **355 fp8 linears in 10 distinct `(k, n)`**
    (`/tmp/it2/fp8shapes.py` enumerates them off-device from a `meta` build),
    and on device against a float64 emulation of `_qdq_fp8` + the e4m3 decode
    (`/tmp/it2/dev8*.py`) **all ten agree to 1.9-3.4e-3 relative**:

    | k | n | count | what | max relative |
    |---|---|---|---|---|
    | 5120 | 2304 | 86 | shared expert `w1`/`w3` | 1.9e-3 |
    | 1280 | 8192 | 43 | `wq_b` | 2.7e-3 |
    | 2048 | 5120 | 43 | `wo_b` | 2.1e-3 (at m=512) |
    | 2304 | 5120 | 43 | shared expert `w2` | 2.3e-3 |
    | 4096 | 2048 | 43 | | 2.0e-3 |
    | 5120 | 512 | 43 | | 3.4e-3 |
    | 5120 | 1280 | 43 | (`nt = 320`) | 2.0e-3 |
    | 1280 | 1024 | 8 | | 3.1e-3 |
    | 6144 | 25600 | 2 | the engram `wkv`, widest `n` | 3.0e-3 |
    | 15360 | 5120 | 1 | the MTP input projection | 2.2e-3 (m=128); **asserts at m=512** |

    plus `m = 130` (a row-pad tail), `m = 256` (two m-tiles) and
    `quantize=False` (the `wo_a` path, 3.3e-3).

    3.4e-3 is the bf16 output quantum (2**-8 = 3.9e-3), so this is as close to
    `fp8_linear` as the dtype allows. The *exact* comparison cannot be made
    either way, and it is worth knowing why: `nki.simulate` cannot run
    `_decode_e4m3` at all (`OverflowError: Python integer -2147450880 out of
    bounds for uint32` on the signed `E4M3_SGN2` operand inside
    `scalar_tensor_tensor_bitvec`), and the standalone device entry point
    rejects `_gemm`'s closures, so neither instrument can host the reference.

    What is **not** established: any speed number. And `k = 15360` at a 512-row
    prefill width asserts (122 880 bytes of activation tiles a partition), which
    is **one** linear of 355 — dispatch that call site to `fp8_linear` rather
    than writing the outer m-group loop, because a loop there would re-read and
    re-decode the weight once per m-group and 4x the decode is worse than the
    scratch it would be avoiding.
    """
    m, k = x.shape
    n = w.shape[0]
    assert w.shape[1] == k, f"fp8 weight {w.shape} is not [n, k] for k={k}"
    assert k % 64 == 0, f"k={k} must be a multiple of 64"
    assert n % BLK == 0, f"n={n} must be a multiple of {BLK}"
    _check(k, n)
    mpad = -(-m // PT) * PT
    ns = -(-n // BLK)

    out = nl.ndarray((mpad, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    x2 = nl.ndarray((mpad, 2, k // 2), dtype=nl.bfloat16, buffer=nl.private_hbm)
    srep = nl.ndarray((k // 64, 2, 16, n), dtype=nl.uint8, buffer=nl.private_hbm)

    _stage_rows_fp8k(x, m, k, x2, quantize)
    _fp8_scale_rows(scale.view(nl.uint16), srep, n, k, ns)
    _gemm_fp8_tfused(
        x2, w.view(nl.uint16), srep.reshape((k // 2, n)), out, mpad, k, n
    )
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
