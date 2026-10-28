# SPDX-License-Identifier: Apache-2.0
"""The megakernels' GEMMs on the Trainium2 tensor engine.

Trainium3's tensor engine multiplies the checkpoint's block-scaled fp8 / fp4
weights natively (`nc_matmul_mx`, the trn3 port's `mk_mx`); NeuronCore-v3 has no
block-scaled matmul (`nc_matmul_mx` and `quantize_mx` are v4-only), and its fp8
type saturates at 240, so a checkpoint e4m3fn weight (peak 448) cannot even be
loaded as-is. What the per-op path does instead -- decode the weight to bf16 and
run a bf16 matmul -- is what these GEMMs do too, with one change of *where*:

  * **fp8 linears are decoded once, at load time** (`mk_model.kmajor_bf16`):
    `dequant(w).T` as a bf16 `[K, N]` buffer, the contraction on the partition
    axis, so a weight tile DMAs straight into SBUF as the matmul's moving
    operand. The per-op kernels decode the same bytes to the same bf16 values on
    every call (`blockquant._decode_e4m3` times the `2**(code-7)` row scale), so
    the arithmetic is unchanged; what goes is the vector-engine decode a weight
    element a call, at twice the weight bytes read.
  * **activations** are round-tripped through fp8 as they are staged
    (`blockquant._qdq_fp8`, peak 224 -- see that module), then transposed on the
    PE into the stationary `[128, k/128, rows]` layout.
  * the fp4 routed experts stay fp4 and are decoded on chip (`mk_moe`).

Two partitionings, one per phase, because the two phases are bound by different
things:

  * **decode** (`gemm_cols`): a step is at most 128 rows, one tile, and the GEMM
    is a weight read. Each physical core stages the tile and computes its half
    of the output columns, so the two cores read disjoint halves of the weight.
  * **prefill** (`gemm_rows`): many row tiles; each core takes its own row tiles
    (`tiles_of`) and all the columns, staging up to `MG` tiles at a time so each
    weight tile loaded is reused `MG` times.

Every helper runs on one physical core and is told its rows / columns.
"""

import nki.isa as nisa
import nki.language as nl

from ...blockquant import _qdq_fp8

PT = 128
#: Widest PSUM tile: one 2 KiB bank of fp32 a partition.
NT = 512
#: Row tiles staged together in `gemm_rows` (their accumulators share PSUM).
MG = 4
#: Weight k-tiles loaded together: `WG * NT * 2` bytes a partition (8 KiB at 512).
WG = 8
#: Weight k-tile groups load as one sync-queue hardware-DGE DMA.
W_HWDGE = True


def cdiv(n, d):
    return (n + d - 1) // d


def ntiles_of(lo, hi, nt=NT):
    """`[(n0, w)]` output tiles covering `[lo, hi)`, at most `nt` wide, equal widths
    where they divide (a 128-wide matmul tile runs at half the rate of a 384)."""
    n = hi - lo
    cnt = cdiv(n, nt)
    out = []
    if n % cnt == 0 and (n // cnt) <= nt:
        w = n // cnt
        for i in range(cnt):
            out.append((lo + i * w, w))
        return out
    n0 = lo
    while n0 < hi:
        w = min(nt, hi - n0)
        out.append((n0, w))
        n0 += w
    return out


def half_cols(n, pid):
    """`(lo, hi)` of core `pid`'s share of `n` output columns, in whole 128-tiles
    (the larger half to core 0)."""
    tiles = n // PT
    a = (tiles + 1) // 2
    if pid == 0:
        return 0, a * PT
    return a * PT, n


def stage_rows_T(x_hbm, r0, nr, k, qdq, col0=0, rows=None):
    """Rows `r0 .. r0+nr` (columns `col0 .. col0+k`) of the bf16 HBM `x_hbm` as the
    stationary operand `xT[p, kb, m] = x[r0 + m, col0 + kb*128 + p]` (bf16), with
    the fp8 round trip first when `qdq`. Rows past `nr` are zero. `rows`, when
    given, is a bf16 SBUF tile already holding the rows (used instead of a load)."""
    nk = k // PT
    if rows is None:
        rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        if nr < PT:
            nisa.memset(rows, 0.0)
        nisa.dma_copy(dst=rows[0:nr, :], src=x_hbm[r0 : r0 + nr, col0 : col0 + k])
    src = _qdq_fp8(rows, PT, k) if qdq else rows
    xT = nl.ndarray((PT, nk, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(nk):
        tp = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=src[:, kb * PT : (kb + 1) * PT])
        nisa.tensor_copy(dst=xT[:, kb, :], src=tp)
    return xT


def _drain(acc, nr, w, out_hbm, r0, n0, out_dtype, out_scale, add_hbm=None):
    o = nl.ndarray((PT, w), dtype=out_dtype, buffer=nl.sbuf)
    if out_scale == 1.0:
        nisa.tensor_copy(dst=o[0:nr, :], src=acc[0:nr, :])
    else:
        nisa.tensor_scalar(dst=o[0:nr, :], data=acc[0:nr, :], op0=nl.multiply, operand0=out_scale)
    if add_hbm is not None:
        # out = add + out (the fp32 sum `_add_into` would form from a stored `out`)
        ad = nl.ndarray((PT, w), dtype=out_dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ad[0:nr, :], src=add_hbm[r0 : r0 + nr, n0 : n0 + w])
        nisa.tensor_tensor(dst=o[0:nr, :], data1=ad[0:nr, :], data2=o[0:nr, :], op=nl.add)
    nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, n0 : n0 + w], src=o[0:nr, :])


def gemm_staged(xTs, nrs, r0s, k, wk_hbm, out_hbm, lo, hi, out_dtype=nl.bfloat16,
                out_scale=1.0, krow0=0, out_col0=None, add_hbm=None):
    """`out[r0s[i] + m, n] = out_scale * sum_k xT_i[k, m] * wk[krow0 + k, n]` (`+ add[..]`
    with `add_hbm`) for the
    staged tiles `xTs` (row counts `nrs`) and the columns `[lo, hi)` of the k-major
    bf16 weight `wk_hbm[K, N]`. The output columns land at `out_col0 + (n - lo)`
    (default: `n` itself)."""
    nk = k // PT
    if out_col0 is None:
        out_col0 = lo
    for n0, w in ntiles_of(lo, hi):
        accs = []
        for _ in range(len(xTs)):
            accs.append(nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum))
        for g0 in range(0, nk, WG):
            ng = min(WG, nk - g0)
            wt = nl.ndarray((PT, ng, w), dtype=nl.bfloat16, buffer=nl.sbuf)
            if W_HWDGE:
                # the group's k-tiles in one hardware-DGE DMA (no gpsimd descriptors)
                N = wk_hbm.shape[1]
                kr = krow0 + g0 * PT
                nisa.dma_copy(dst=wt[:, 0:ng, :],
                              src=wk_hbm.ap(pattern=[[N, PT], [PT * N, ng], [1, w]],
                                            offset=kr * N + n0),
                              dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            else:
                for j in range(ng):
                    kr = krow0 + (g0 + j) * PT
                    nisa.dma_copy(dst=wt[:, j, :], src=wk_hbm[kr : kr + PT, n0 : n0 + w])
            for j in range(ng):
                kb = g0 + j
                for i in range(len(xTs)):
                    nisa.nc_matmul(dst=accs[i][0 : nrs[i], :], stationary=xTs[i][:, kb, 0 : nrs[i]],
                                   moving=wt[:, j, :], accumulate=(kb > 0))
        for i in range(len(xTs)):
            _drain(accs[i], nrs[i], w, out_hbm, r0s[i], out_col0 + (n0 - lo), out_dtype, out_scale,
                   add_hbm=add_hbm)


def gemm_staged_sb(xTs, nrs, k, wk_hbm, out_sbs, lo, hi):
    """`gemm_staged` into SBUF: `out_sbs[i][m, n - lo] = sum_k xT_i[k, m] * wk[k, n]`
    (the PSUM copy casts to the tile's dtype)."""
    nk = k // PT
    N = wk_hbm.shape[1]
    for n0, w in ntiles_of(lo, hi):
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in xTs]
        for g0 in range(0, nk, WG):
            ng = min(WG, nk - g0)
            wt = nl.ndarray((PT, ng, w), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt[:, 0:ng, :],
                          src=wk_hbm.ap(pattern=[[N, PT], [PT * N, ng], [1, w]],
                                        offset=g0 * PT * N + n0),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            for j in range(ng):
                kb = g0 + j
                for i in range(len(xTs)):
                    nisa.nc_matmul(dst=accs[i][0 : nrs[i], :], stationary=xTs[i][:, kb, 0 : nrs[i]],
                                   moving=wt[:, j, :], accumulate=(kb > 0))
        for i in range(len(xTs)):
            nisa.tensor_copy(dst=out_sbs[i][0 : nrs[i], n0 - lo : n0 - lo + w],
                             src=accs[i][0 : nrs[i], :])


def gemm_cols(x_hbm, nr, k, wk_hbm, out_hbm, lo, hi, qdq=True, out_dtype=nl.bfloat16,
              out_scale=1.0, x_col0=0, krow0=0, r0=0):
    """Decode: the one row tile `x[r0 : r0+nr]` (nr <= 128) against output columns
    `[lo, hi)` -- the calling core's half."""
    if hi <= lo:
        return
    xT = stage_rows_T(x_hbm, r0, nr, k, qdq, col0=x_col0)
    gemm_staged([xT], [nr], [r0], k, wk_hbm, out_hbm, lo, hi, out_dtype=out_dtype,
                out_scale=out_scale, krow0=krow0)


def gemm_rows(x_hbm, tiles, k, wk_hbm, out_hbm, n, qdq=True, out_dtype=nl.bfloat16,
              out_scale=1.0, x_col0=0, krow0=0, lo=0, hi=None, more=(), add_hbm=None):
    """Prefill: the row tiles `tiles` (`[(r0, nr)]`, the calling core's) against
    output columns `[lo, hi)` (default all `n`), `MG` tiles staged at a time.
    `more` = `[(wk_hbm, out_hbm, n)]`: further GEMMs of the same rows (all their
    columns), run on the same staged tiles."""
    if hi is None:
        hi = n
    for g0 in range(0, len(tiles), MG):
        grp = tiles[g0 : g0 + MG]
        xTs, nrs, r0s = [], [], []
        for r0, nr in grp:
            xTs.append(stage_rows_T(x_hbm, r0, nr, k, qdq, col0=x_col0))
            nrs.append(nr)
            r0s.append(r0)
        gemm_staged(xTs, nrs, r0s, k, wk_hbm, out_hbm, lo, hi, out_dtype=out_dtype,
                    out_scale=out_scale, krow0=krow0, add_hbm=add_hbm)
        for wk2, out2, n2 in more:
            gemm_staged(xTs, nrs, r0s, k, wk2, out2, 0, n2, out_dtype=out_dtype,
                        out_scale=out_scale, krow0=krow0)


def gemm_sb(xT, nr, k, wk_hbm, out_sb, lo, hi, krow0=0):
    """One staged tile against weight columns `[lo, hi)` into the SBUF tile
    `out_sb[128, hi - lo]` (any dtype: the PSUM copy casts)."""
    nk = k // PT
    for n0, w in ntiles_of(lo, hi):
        acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
        for g0 in range(0, nk, WG):
            ng = min(WG, nk - g0)
            wt = nl.ndarray((PT, ng, w), dtype=nl.bfloat16, buffer=nl.sbuf)
            if W_HWDGE:
                # the group's k-tiles in one hardware-DGE DMA (no gpsimd descriptors)
                N = wk_hbm.shape[1]
                kr = krow0 + g0 * PT
                nisa.dma_copy(dst=wt[:, 0:ng, :],
                              src=wk_hbm.ap(pattern=[[N, PT], [PT * N, ng], [1, w]],
                                            offset=kr * N + n0),
                              dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            else:
                for j in range(ng):
                    kr = krow0 + (g0 + j) * PT
                    nisa.dma_copy(dst=wt[:, j, :], src=wk_hbm[kr : kr + PT, n0 : n0 + w])
            for j in range(ng):
                kb = g0 + j
                nisa.nc_matmul(dst=acc[0:nr, :], stationary=xT[:, kb, 0:nr], moving=wt[:, j, :],
                               accumulate=(kb > 0))
        nisa.tensor_copy(dst=out_sb[0:nr, n0 - lo : n0 - lo + w], src=acc[0:nr, :])
