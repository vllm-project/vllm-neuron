# SPDX-License-Identifier: Apache-2.0
"""Block-scaled fp8 / fp4 GEMMs on the Trainium3 tensor engine (`nc_matmul_mx`).

The checkpoint's weights are microscaled already: fp8 `[n, k]` e4m3 with one e8m0
scale per `32 x 32` tile, fp4 `[n, k//2]` e2m1 pairs with one e8m0 scale per row
and 32 columns, and the model quantizes activations per row in 32-wide blocks with
power-of-two scales. That is the OCP MX format the NeuronCore-v4 tensor engine
multiplies natively, dequantizing inside the array: one instruction contracts
K = 512 (128 partitions x 4 packed elements) at four times the bf16 rate, so the
weights are no longer decoded to bf16 scratch in HBM and read back.

Layout (`k` = contraction index inside a 512-wide chunk, `p` = partition):

    data   x4 element at [p, f] packs k = 4p + j, j = 0..3 (byte / nibble j)
    scale  group g = partitions 8g..8g+7 = k in [32g, 32g + 32), one e8m0 code,
           stored at partition 32 * (g // 4) + g % 4, column f (the quadrant layout
           `quantize_mx` produces and `nc_matmul_mx` reads)

so a hardware scaling group is exactly one checkpoint block. Weights arrive from
the host in that layout (`mk_model.mx_fp8` / `mx_fp4`): `[K/4, N]` words (int32 for
fp8, int16 for fp4) and `[K/32, N]` scale bytes, streamed into SBUF with plain
DMAs. Activations are staged on chip: rows `[128, K]` bf16, four strided PE
transposes per chunk into the j-minor `[p, m, j]` tile, then `quantize_mx`
(scale 2^(floor(log2 amax) - 7), e4m3fn, the OCP rounding).

Every helper runs on one physical core; callers split the output columns (or the
rows) between the two cores as before.
"""

import nki.isa as nisa
import nki.language as nl

PT = 128
#: Contraction elements per matmul instruction and per 128-partition chunk.
KC = 512
#: Widest output tile (fp32 PSUM columns) streamed per instruction.
NT = 512
FP8_X4 = nl.float8_e4m3fn_x4
FP4_X4 = nl.float4_e2m1fn_x4


def cdiv(n, d):
    return (n + d - 1) // d


def chunks_of(k):
    """`[(k0, kk)]` 512-wide chunks of a contraction of `k` (the last may be shorter;
    `kk` is a multiple of 128 so the chunk's partitions are 32, 64, 96 or 128)."""
    out = []
    k0 = 0
    while k0 < k:
        kk = min(KC, k - k0)
        assert kk % PT == 0, f"contraction {k} is not a multiple of 128"
        out.append((k0, kk))
        k0 += kk
    return out


def ntiles_of(n, nt=NT):
    """`[(n0, w)]` output tiles of at most `nt` columns."""
    out = []
    n0 = 0
    while n0 < n:
        w = min(nt, n - n0)
        out.append((n0, w))
        n0 += w
    return out


# --------------------------------------------------------------------------- activations


def interleave_rows(rows, k0, kk, xT):
    """`xT[p, m, j] = rows[m, k0 + 4p + j]` for the chunk `[k0, k0 + kk)`: four PE
    transposes of the stride-4 column sets, each copied into its j slot."""
    pk = kk // 4
    k = rows.shape[1]
    for j in range(4):
        tp = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:pk, :], data=rows.ap(pattern=[[k, PT], [4, pk]], offset=k0 + j))
        nisa.tensor_copy(dst=xT[0:pk, :, j], src=tp[0:pk, :])


def alloc_x(nmt, k):
    """SBUF for `nmt` quantized activation tiles of contraction `k`: the x4 data
    `[PT, nmt, nkc, PT]` and the scale tile `[PT, nmt, nkc, PT]` (uint8)."""
    nkc = len(chunks_of(k))
    xq = nl.ndarray((PT, nmt, nkc, PT), dtype=FP8_X4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, nmt, nkc, PT), dtype=nl.uint8, buffer=nl.sbuf)
    # Defined in full before the quadrant rows are written: a tile that is only
    # partly written inside a dynamic loop reads as a loop-carried value to the
    # compiler (NCC_IGCA108).
    nisa.memset(xs, 0)
    return xq, xs


def stage_rows(rows, k, xq, xs, mi):
    """Quantize the bf16 SBUF rows `[128, k]` into slot `mi` of `(xq, xs)`."""
    for kc, (k0, kk) in enumerate(chunks_of(k)):
        pk = kk // 4
        xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        interleave_rows(rows, k0, kk, xT)
        nisa.quantize_mx(
            dst=xq[0:pk, mi, kc, :], src=xT[0:pk, :, :].reshape((pk, PT * 4)),
            dst_scale=xs[0:pk, mi, kc, :],
        )


def stage_x(x_hbm, r0, k, xq, xs, mi, col0=0):
    """Load rows `r0 .. r0+128` (columns `col0 .. col0+k`) of the bf16 HBM `x_hbm`
    and quantize them into slot `mi`."""
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=x_hbm[r0 : r0 + PT, col0 : col0 + k])
    stage_rows(rows, k, xq, xs, mi)


def stage_x_gathered(x_hbm, gidx, k, xq, xs, mi):
    """As `stage_x` for the rows `x_hbm[gidx[p]]` (`gidx` an int32 `(128, 1)` SBUF
    column: an indirect DMA gather)."""
    kfull = x_hbm.shape[1]
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rows,
        src=x_hbm.ap(pattern=[[kfull, PT], [1, k]], vector_offset=gidx[0:PT, 0:1], indirect_dim=0),
    )
    stage_rows(rows, k, xq, xs, mi)


# --------------------------------------------------------------------------- weights


#: Weight chunks staged together in one SBUF tile (x4 fp8: 2 KiB a chunk a partition).
WG = 8


def load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4):
    """Chunks `g0 .. g0+ng` of the host-packed weight for output columns `n0 .. n0+w`:
    one `[PT, ng, w]` moving tile and its `[PT, ng, w]` scale tile (quadrant layout,
    zeroed first so every partition is defined inside a dynamic loop)."""
    wt = nl.ndarray((PT, ng, w), dtype=FP4_X4 if fp4 else FP8_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, ng, w), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(ws, 0)
    for j in range(ng):
        k0, kk = chunks[g0 + j]
        pk = kk // 4
        row0 = krow0 + k0 // 4
        nisa.dma_copy(dst=wt[0:pk, j, :], src=w_mx[row0 : row0 + pk, n0 : n0 + w])
        grp0 = row0 // 8
        for q in range(pk // 32):
            nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, j, :],
                          src=s_mx[grp0 + 4 * q : grp0 + 4 * q + 4, n0 : n0 + w])
    return wt, ws


def gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, r0, lo, hi, fp4=False, out_dtype=nl.bfloat16,
         out_scale=1.0, krow0=0):
    """`out[r0 + 128*mi + m, n] = out_scale * sum_k x[m, k] w[n, k]` for the `nmt` staged
    tiles and the output columns `[lo, hi)`.

    `w_mx` is the `[K/4, N]` word view (already the x4 dtype) and `s_mx` the `[K/32, N]`
    scale bytes; `krow0` offsets both for a contraction slice (in units of k/4 rows).
    """
    chunks = chunks_of(k)
    nkc = len(chunks)
    for n0, w in ntiles_of(hi - lo):
        n0 += lo
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                for mi in range(nmt):
                    nisa.nc_matmul_mx(
                        dst=accs[mi], stationary=xq[0:pk, mi, kc, :], moving=wt[0:pk, j, :],
                        stationary_scale=xs[0:pk, mi, kc, :], moving_scale=ws[0:pk, j, :],
                        accumulate=(kc > 0),
                    )
        for mi in range(nmt):
            o = nl.ndarray((PT, w), dtype=out_dtype, buffer=nl.sbuf)
            if out_scale == 1.0:
                nisa.tensor_copy(dst=o, src=accs[mi])
            else:
                nisa.tensor_scalar(dst=o, data=accs[mi], op0=nl.multiply, operand0=out_scale)
            nisa.dma_copy(dst=out_hbm[r0 + mi * PT : r0 + (mi + 1) * PT, n0 : n0 + w], src=o)


def gemm_sb(xq, xs, k, w_mx, s_mx, out_sb, n, fp4=False, krow0=0, n0=0):
    """One staged tile against weight columns `[n0, n0 + n)`, into the SBUF tile
    `out_sb[128, n]` (any dtype: the PSUM copy casts)."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    for c0, w in ntiles_of(n):
        acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0 + c0, w, fp4)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                nisa.nc_matmul_mx(
                    dst=acc, stationary=xq[0:pk, 0, kc, :], moving=wt[0:pk, j, :],
                    stationary_scale=xs[0:pk, 0, kc, :], moving_scale=ws[0:pk, j, :],
                    accumulate=(kc > 0),
                )
        nisa.tensor_copy(dst=out_sb[:, c0 : c0 + w], src=acc)


# --------------------------------------------------------------------------- whole GEMMs

#: Row tiles staged together (their accumulators share PSUM: 4 x 512 fp32 = 8 KiB).
MG = 4


def gemm_rows(x_hbm, m, k, w_mx, s_mx, out_hbm, lo, hi, fp4=False, out_dtype=nl.bfloat16,
              out_scale=1.0, x_col0=0, krow0=0):
    """`out[0:m, lo:hi] = out_scale * x[0:m, x_col0 : x_col0 + k] @ w[lo:hi, krow].T` for
    bf16 HBM rows (`m` a multiple of 128), the activation quantized on chip."""
    mt = m // PT
    for g0 in range(0, mt, MG):
        nmt = min(MG, mt - g0)
        xq, xs = alloc_x(nmt, k)
        for mi in range(nmt):
            stage_x(x_hbm, (g0 + mi) * PT, k, xq, xs, mi, col0=x_col0)
        gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, g0 * PT, lo, hi, fp4=fp4, out_dtype=out_dtype,
             out_scale=out_scale, krow0=krow0)
