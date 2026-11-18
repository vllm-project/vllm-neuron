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


#: `interleave_rows`: the four transposes of a chunk land in one PSUM tile and one
#: strided copy moves them into the j-minor layout (instead of four 128-column copies).
STAGE_ONE_COPY = True


def interleave_rows(rows, k0, kk, xT):
    """`xT[p, m, j] = rows[m, k0 + 4p + j]` for the chunk `[k0, k0 + kk)`: four PE
    transposes of the stride-4 column sets, each copied into its j slot."""
    pk = kk // 4
    k = rows.shape[1]
    if STAGE_ONE_COPY:
        tp = nl.ndarray((PT, 4, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(4):
            nisa.nc_transpose(dst=tp[0:pk, j, :],
                              data=rows.ap(pattern=[[k, PT], [4, pk]], offset=k0 + j))
        nisa.tensor_copy(dst=xT[0:pk, :, :], src=tp.ap(pattern=[[4 * PT, pk], [1, PT], [PT, 4]]))
        return
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


#: Staging with `nv` (<= 42) valid rows (decode): `G = 128 // nv` 512-wide chunks share
#: one set of four PE transposes -- chunk c's rows sit on partitions `c*nv .. c*nv+nv` of
#: a packed `[128, 512]` tile (or `m*nkc + c` when one group holds every chunk: a single
#: contiguous DMA) -- and only the nv valid columns are copied / quantized; the other
#: columns of `xq` / `xs` are zero (rows past nv give exact zeros). Every valid row's
#: values and 32-element scale groups are unchanged, so its quantization is identical.
STAGE_PACK = True
#: ... only when at least this many chunks share a transpose: at G = 2 (T = 60) the per-chunk
#: row DMAs cost more than the halved transposes save (in8192_bs10 exec-cc 6.40 -> 6.45 ms).
PACK_MIN_G = 3


def pack_ok(nv, k):
    return STAGE_PACK and nv is not None and 0 < nv and PT // nv >= PACK_MIN_G and k % PT == 0


def stage_x_packed(x_hbm, r0, k, xq, xs, mi, nv, col0=0, shifts=1):
    """`stage_x` for the first `nv` rows only (`pack_ok(nv, k)`; the rest zero).
    `shifts`: slots `mi .. mi + shifts` get the same rows, slot `mi + s` at columns
    `s*nv .. s*nv + nv` (for `gemm_sb_packed`)."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    G = PT // nv
    kfull = x_hbm.shape[1]
    nisa.memset(xq[:, mi : mi + shifts, :, :], 0)
    for c0 in range(0, nkc, G):
        g = min(G, nkc - c0)
        q = g * nv
        full = all(chunks[c0 + c][1] == KC for c in range(g))
        xp = nl.ndarray((PT, KC), dtype=nl.bfloat16, buffer=nl.sbuf)
        mmajor = (g == nkc and full and kfull == k and col0 == 0)
        if mmajor:
            # partition m * nkc + c = the rows' flat 512-element runs: one DMA
            nisa.dma_copy(dst=xp[0:q, :], src=x_hbm.ap(pattern=[[KC, q], [1, KC]], offset=r0 * kfull))
        else:
            if not full:
                # (a short chunk's columns past its end feed only unused partitions)
                nisa.memset(xp, 0.0, engine=nisa.engine.gpsimd)
            for c in range(g):
                k0, kk = chunks[c0 + c]
                nisa.dma_copy(dst=xp[c * nv : (c + 1) * nv, 0:kk],
                              src=x_hbm[r0 : r0 + nv, col0 + k0 : col0 + k0 + kk])
        _stage_packed_tile(xp, nv, chunks, c0, g, xq, xs, mi, mmajor, shifts)


def _stage_packed_tile(xp, nv, chunks, c0, g, xq, xs, mi, mmajor=False, shifts=1):
    """Chunks `c0 .. c0+g` from the packed bf16 SBUF tile `xp` (`[128, xw]`, chunk c's
    row m on partition `c*nv + m`, or `m*g + c` when `mmajor`) into slot `mi`."""
    q = g * nv
    xw = xp.shape[1]
    tp = nl.ndarray((PT, 4, q), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(4):
        nisa.nc_transpose(dst=tp[:, j, :], data=xp.ap(pattern=[[xw, q], [4, PT]], offset=j))
    # xT[p, c, m, j] = tp[p, j, (c, m)]
    xT = nl.ndarray((PT, g, nv * 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    if mmajor:
        src = tp.ap(pattern=[[4 * q, PT], [1, g], [g, nv], [q, 4]])
        dst = xT.ap(pattern=[[g * nv * 4, PT], [nv * 4, g], [4, nv], [1, 4]])
    else:
        src = tp.ap(pattern=[[4 * q, PT], [1, q], [q, 4]])
        dst = xT.ap(pattern=[[g * nv * 4, PT], [4, q], [1, 4]])
    nisa.tensor_copy(dst=dst, src=src)
    for c in range(g):
        pk = chunks[c0 + c][1] // 4
        for sh in range(shifts):
            nisa.quantize_mx(dst=xq[0:pk, mi + sh, c0 + c, sh * nv : (sh + 1) * nv], src=xT[0:pk, c, :],
                             dst_scale=xs[0:pk, mi + sh, c0 + c, sh * nv : (sh + 1) * nv])


def stage_rows_packed(xp, nv, k, c0, g, xq, xs, mi):
    """Quantize chunks `c0 .. c0+g` of a contraction of `k` from the packed bf16 SBUF
    tile `xp` (`[128, 512]`: chunk c0 + c's row m on partition `c*nv + m`) into slot `mi`
    (whose other columns the caller zeroed)."""
    chunks = chunks_of(k)
    assert g * nv <= PT
    _stage_packed_tile(xp, nv, chunks, c0, g, xq, xs, mi)


def stage_x(x_hbm, r0, k, xq, xs, mi, col0=0, nv=None, shifts=1):
    """Load rows `r0 .. r0+128` (columns `col0 .. col0+k`) of the bf16 HBM `x_hbm`
    and quantize them into slot `mi`. `nv`: only the first `nv` rows are real (the
    packed staging when `pack_ok`, `shifts` as `stage_x_packed`)."""
    if pack_ok(nv, k):
        stage_x_packed(x_hbm, r0, k, xq, xs, mi, nv, col0=col0, shifts=shifts)
        return
    assert shifts == 1
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rows, src=x_hbm[r0 : r0 + PT, col0 : col0 + k])
    stage_rows(rows, k, xq, xs, mi)


def stage_x_gathered(x_hbm, gidx, k, xq, xs, mi, nrows=PT):
    """As `stage_x` for the rows `x_hbm[gidx[p]]` (`gidx` an int32 `(128, 1)` SBUF
    column: an indirect DMA gather). `nrows < 128`: only the first `nrows` rows are
    gathered, the others are zero (what the zero rows the remaining slots point at hold)."""
    kfull = x_hbm.shape[1]
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    if nrows < PT:
        nisa.memset(rows, 0.0, engine=nisa.engine.gpsimd)
    nisa.dma_copy(
        dst=rows[0:nrows, :],
        src=x_hbm.ap(pattern=[[kfull, nrows], [1, k]], vector_offset=gidx[0:nrows, 0:1], indirect_dim=0),
    )
    stage_rows(rows, k, xq, xs, mi)


# --------------------------------------------------------------------------- weights


#: Weight chunks staged together in one SBUF tile (x4 fp8: 2 KiB a chunk a partition);
#: a group's full chunks load in one data DMA + 4 scale DMAs (`load_w_group`).
WG = 16


#: `gemm_sb` (the routed experts, inside a dynamic loop) issues its weight / scale DMAs
#: from the hardware DGE on the sync engine instead of GpSimd's software DGE (they have
#: static addresses; GpSimd took ~0.75 us to issue each scale DMA).
GEMM_SB_HWDGE = True


#: The hardware-DGE weight DMAs alternate between the sync and the scalar engine's
#: queues (one queue alone streams the routed experts' weights at ~160 GB/s).
GEMM_SB_HW_ENGINES = (nisa.engine.sync, nisa.engine.scalar)
_wdma_n = [0]
#: The static (dense) GEMMs' weight / scale DMAs on the same hardware DGE queues
#: (`GEMM_SB_HW_ENGINES`) instead of the default DMA path.
DENSE_W_HWDGE = True
#: `gemm`'s result stores by statically generated descriptors (their own DMA ring) instead
#: of the scalar engine's hardware-DGE queue, where they waited behind the next GEMM's
#: prefetched weight DMAs (core 1's q/kv halves landed ~10 us late at in1024_bs1).
GEMM_STORE_STATIC = True


#: The routed experts' (dynamic-address) weight DMAs rotate over these queues ("sync" /
#: "scalar": hardware DGE; "swdge": GpSimd's software DGE -- adding it as a third queue
#: did not change the bs10 expert loop, which is tensor-engine-bound: in1024_bs10 172 -> 170 us).
EXPERT_W_QUEUES = ("sync", "scalar")
_edma_n = [0]
_edma_s = [0]


#: `prefetch_w`: the instruction its weight DMAs are schedule-pinned after (else None).
_W_AFTER = [None]


def _pin(ins):
    after = _W_AFTER[0]
    if after is not None and ins is not None and hasattr(ins, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)


def _wdma(dst, src, hw, eoff=None, scale=False):
    if eoff is not None and EXPERT_W_QUEUES:
        # the small scale DMAs stay on the hardware-DGE queues (GpSimd's software DGE
        # spends ~0.75 us generating each one's descriptors)
        qs = [q for q in EXPERT_W_QUEUES if q != "swdge"] if scale else EXPERT_W_QUEUES
        cnt = _edma_s if scale else _edma_n
        q = qs[cnt[0] % len(qs)]
        cnt[0] += 1
        if q == "swdge":
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)
        else:
            eng = nisa.engine.sync if q == "sync" else nisa.engine.scalar
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
        return
    if hw:
        eng = GEMM_SB_HW_ENGINES[_wdma_n[0] % len(GEMM_SB_HW_ENGINES)]
        _wdma_n[0] += 1
        _pin(nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng))
    else:
        _pin(nisa.dma_copy(dst=dst, src=src))


def load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4, hw=False, dense=False, eoff=None):
    """Chunks `g0 .. g0+ng` of the host-packed weight for output columns `n0 .. n0+w`:
    one `[PT, ng, w]` moving tile and its `[PT, ng, w]` scale tile (quadrant layout,
    zeroed first so every partition is defined inside a dynamic loop). `dense`: a static
    GEMM's (`gemm`) group, its DMAs on the hardware DGE queues when `DENSE_W_HWDGE`."""
    wt = nl.ndarray((PT, ng, w), dtype=FP4_X4 if fp4 else FP8_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, ng, w), dtype=nl.uint8, buffer=nl.sbuf)
    # static GEMMs: GpSimd (idle there) zeroes it, off the DVE that stages activations
    _pin(nisa.memset(ws, 0, engine=nisa.engine.unknown if hw else nisa.engine.gpsimd))
    _load_w_into(wt, ws, 0, w_mx, s_mx, chunks, g0, ng, krow0, n0, w, hw or (dense and DENSE_W_HWDGE),
                 eoff=eoff)
    return wt, ws


def _wap(t, pattern, offset, eoff):
    """`t.ap(pattern, offset)`; with `eoff` (a device scalar), `t` is a stacked `[E, ..]`
    weight and the pattern addresses its slice `eoff` (a dynamic start address)."""
    if eoff is None:
        return t.ap(pattern=pattern, offset=offset)
    return t.ap(pattern=pattern, offset=offset, scalar_offset=eoff, indirect_dim=0)


def _load_w_into(wt, ws, cb, w_mx, s_mx, chunks, g0, ng, krow0, n0, w, hw, eoff=None):
    """`load_w_group`'s DMAs into columns `cb .. cb+w` of the (zeroed) tiles `wt` / `ws`.
    `eoff`: `w_mx` / `s_mx` are stacked `[E, K/4, N]` / `[E, K/32, N]`, slice `eoff`."""
    # The group's full 512-wide chunks are consecutive 128-row blocks of the packed
    # weight (and 16-row blocks of the scales): one DMA for the data and one per
    # quadrant for the scales; a short last chunk on its own.
    nf = 0
    while nf < ng and chunks[g0 + nf][1] == KC:
        nf += 1
    ncol = w_mx.shape[-1]
    scol = s_mx.shape[-1]
    if nf > 0:
        r0 = krow0 + chunks[g0][0] // 4
        _wdma(wt[0:PT, 0:nf, cb : cb + w],
              _wap(w_mx, [[ncol, PT], [PT * ncol, nf], [1, w]], r0 * ncol + n0, eoff), hw, eoff)
        for q in range(4):
            _wdma(ws[32 * q : 32 * q + 4, 0:nf, cb : cb + w],
                  _wap(s_mx, [[scol, 4], [16 * scol, nf], [1, w]], (r0 // 8 + 4 * q) * scol + n0, eoff),
                  hw, eoff, scale=True)
    for j in range(nf, ng):
        k0, kk = chunks[g0 + j]
        pk = kk // 4
        row0 = krow0 + k0 // 4
        if eoff is None:
            _wdma(wt[0:pk, j, cb : cb + w], w_mx[row0 : row0 + pk, n0 : n0 + w], hw)
        else:
            _wdma(wt[0:pk, j, cb : cb + w], _wap(w_mx, [[ncol, pk], [1, w]], row0 * ncol + n0, eoff), hw, eoff)
        grp0 = row0 // 8
        for q in range(pk // 32):
            if eoff is None:
                _wdma(ws[32 * q : 32 * q + 4, j, cb : cb + w],
                      s_mx[grp0 + 4 * q : grp0 + 4 * q + 4, n0 : n0 + w], hw)
            else:
                _wdma(ws[32 * q : 32 * q + 4, j, cb : cb + w],
                      _wap(s_mx, [[scol, 4], [1, w]], (grp0 + 4 * q) * scol + n0, eoff), hw, eoff,
                      scale=True)


def alloc_w_full(k, n, fp4):
    """SBUF for a whole weight slice (every chunk of a contraction of `k`, `n` output
    columns): `(wt, ws)`, filled by `load_w_full`, read by `gemm_sb_pre`."""
    nkc = len(chunks_of(k))
    wt = nl.ndarray((PT, nkc, n), dtype=FP4_X4 if fp4 else FP8_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, nkc, n), dtype=nl.uint8, buffer=nl.sbuf)
    return wt, ws


def load_w_full(w_mx, s_mx, k, n, wt, ws, krow0=0, n0=0):
    """Weight columns `[n0, n0 + n)` (every chunk) into `alloc_w_full`'s tiles, with
    `gemm_sb`'s DMAs (hardware DGE), one group per output tile of `NT` columns."""
    chunks = chunks_of(k)
    nisa.memset(ws, 0)
    for c0, w in ntiles_of(n):
        _load_w_into(wt, ws, c0, w_mx, s_mx, chunks, 0, len(chunks), krow0, n0 + c0, w,
                     GEMM_SB_HWDGE)


def load_w_cols(w_mx, s_mx, k, n, fp4=False, krow0=0, n0=0, eoff=None, nblk=2):
    """Every chunk of the weight columns `[n0, n0 + n)` into one SBUF slice
    (`alloc_w_full`'s tiles) by `nblk` column blocks of `_load_w_into` DMAs (a few large
    DMAs instead of one group per 512-column output tile), for `gemm_sb_pre_multi`.
    `eoff`: as `gemm_sb_multi`'s."""
    chunks = chunks_of(k)
    wt, ws = alloc_w_full(k, n, fp4)
    nisa.memset(ws, 0)
    nb = cdiv(n // PT, nblk) * PT
    for cb in range(0, n, nb):
        wb = min(nb, n - cb)
        _load_w_into(wt, ws, cb, w_mx, s_mx, chunks, 0, len(chunks), krow0, n0 + cb, wb,
                     GEMM_SB_HWDGE, eoff=eoff)
    return wt, ws


def gemm_sb_pre_multi(xq, xs, k, wt, ws, n, emit):
    """`gemm_sb_multi` (one staged tile) against a slice already in SBUF (`load_w_cols`):
    the same matmuls in the same order, `emit(0, c0, w, acc)` per output tile."""
    chunks = chunks_of(k)
    for c0, w in ntiles_of(n):
        acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
        for kc in range(len(chunks)):
            pk = chunks[kc][1] // 4
            nisa.nc_matmul_mx(
                dst=acc, stationary=xq[0:pk, 0, kc, :], moving=wt[0:pk, kc, c0 : c0 + w],
                stationary_scale=xs[0:pk, 0, kc, :], moving_scale=ws[0:pk, kc, c0 : c0 + w],
                accumulate=(kc > 0),
            )
        emit(0, c0, w, acc)


def gemm_sb_pre(xq, xs, k, wt, ws, out_sb, n):
    """`gemm_sb` against a weight slice already in SBUF (`load_w_full`): the same
    matmuls in the same order."""
    chunks = chunks_of(k)
    for c0, w in ntiles_of(n):
        acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
        for kc in range(len(chunks)):
            pk = chunks[kc][1] // 4
            nisa.nc_matmul_mx(
                dst=acc, stationary=xq[0:pk, 0, kc, :], moving=wt[0:pk, kc, c0 : c0 + w],
                stationary_scale=xs[0:pk, 0, kc, :], moving_scale=ws[0:pk, kc, c0 : c0 + w],
                accumulate=(kc > 0),
            )
        nisa.tensor_copy(dst=out_sb[:, c0 : c0 + w], src=acc)


def prefetch_w(k, w_mx, s_mx, lo, hi, fp4=False, krow0=0, after=None):
    """`gemm`'s weight groups for the output columns `[lo, hi)` loaded ahead (the same
    tiles and DMAs), for `gemm(pre=...)`: `{(n0, g0): (wt, ws)}`. `after`: the DMAs are
    schedule-pinned after this instruction."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    pre = {}
    _W_AFTER[0] = after
    try:
        for n0, w in ntiles_of(hi - lo):
            n0 += lo
            for g0 in range(0, nkc, WG):
                ng = min(WG, nkc - g0)
                pre[(n0, g0)] = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4, dense=True)
    finally:
        _W_AFTER[0] = None
    return pre


def gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, r0, lo, hi, fp4=False, out_dtype=nl.bfloat16,
         out_scale=1.0, krow0=0, epilogue=None, out_rows=None, pre=None, kc0=0):
    """`out[r0 + 128*mi + m, n] = out_scale * sum_k x[m, k] w[n, k]` for the `nmt` staged
    tiles and the output columns `[lo, hi)`.

    `w_mx` is the `[K/4, N]` word view (already the x4 dtype) and `s_mx` the `[K/32, N]`
    scale bytes; `krow0` offsets both for a contraction slice (in units of k/4 rows).
    `epilogue(row0, n0, w, o)`, when given, takes each `(128, w)` SBUF result tile `o`
    (rows `row0 ..`, columns `n0 ..`) instead of the store to `out_hbm`.
    `out_rows`: only the rows below it are stored (`out_hbm` may have fewer rows).
    `kc0`: the activation's chunks start at chunk `kc0` of `(xq, xs)`.
    """
    chunks = chunks_of(k)
    nkc = len(chunks)
    for n0, w in ntiles_of(hi - lo):
        n0 += lo
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            if pre is not None:
                wt, ws = pre[(n0, g0)]
            else:
                wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4, dense=True)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                for mi in range(nmt):
                    nisa.nc_matmul_mx(
                        dst=accs[mi], stationary=xq[0:pk, mi, kc0 + kc, :], moving=wt[0:pk, j, :],
                        stationary_scale=xs[0:pk, mi, kc0 + kc, :], moving_scale=ws[0:pk, j, :],
                        accumulate=(kc > 0),
                    )
        for mi in range(nmt):
            o = nl.ndarray((PT, w), dtype=out_dtype, buffer=nl.sbuf)
            if out_scale == 1.0:
                nisa.tensor_copy(dst=o, src=accs[mi])
            else:
                nisa.tensor_scalar(dst=o, data=accs[mi], op0=nl.multiply, operand0=out_scale)
            if epilogue is not None:
                epilogue(r0 + mi * PT, n0, w, o)
                continue
            row0 = r0 + mi * PT
            nr = PT if out_rows is None else min(PT, out_rows - row0)
            if nr <= 0:
                continue
            if GEMM_STORE_STATIC:
                nisa.dma_copy(dst=out_hbm[row0 : row0 + nr, n0 : n0 + w], src=o[0:nr, :],
                              dge_mode=nisa.dge_mode.none)
            else:
                nisa.dma_copy(dst=out_hbm[row0 : row0 + nr, n0 : n0 + w], src=o[0:nr, :])


def gemm_sb(xq, xs, k, w_mx, s_mx, out_sb, n, fp4=False, krow0=0, n0=0, epilogue=None):
    """One staged tile against weight columns `[n0, n0 + n)`, into the SBUF tile
    `out_sb[128, n]` (any dtype: the PSUM copy casts). `epilogue(c0, w)`, when given,
    runs after each output tile's copy (columns `c0 .. c0 + w` of `out_sb`)."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    for c0, w in ntiles_of(n):
        acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0 + c0, w, fp4,
                                  hw=GEMM_SB_HWDGE)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                nisa.nc_matmul_mx(
                    dst=acc, stationary=xq[0:pk, 0, kc, :], moving=wt[0:pk, j, :],
                    stationary_scale=xs[0:pk, 0, kc, :], moving_scale=ws[0:pk, j, :],
                    accumulate=(kc > 0),
                )
        nisa.tensor_copy(dst=out_sb[:, c0 : c0 + w], src=acc)
        if epilogue is not None:
            epilogue(c0, w)


def gemm_sb_multi(xq, xs, nmt, k, w_mx, s_mx, n, emit, fp4=False, krow0=0, n0=0, eoff=None):
    """`gemm_sb` for the `nmt` staged tiles of `(xq, xs)` sharing each weight group's
    load: `emit(t, c0, w, acc)` takes tile `t`'s fp32 PSUM result for the output
    columns `c0 .. c0 + w`. Each tile's matmuls are `gemm_sb`'s, in the same order.
    `eoff` (a device scalar): `w_mx` / `s_mx` are stacked per expert, use slice `eoff`."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    for c0, w in ntiles_of(n):
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0 + c0, w, fp4,
                                  hw=GEMM_SB_HWDGE, eoff=eoff)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                for t in range(nmt):
                    nisa.nc_matmul_mx(
                        dst=accs[t], stationary=xq[0:pk, t, kc, :], moving=wt[0:pk, j, :],
                        stationary_scale=xs[0:pk, t, kc, :], moving_scale=ws[0:pk, j, :],
                        accumulate=(kc > 0),
                    )
        for t in range(nmt):
            emit(t, c0, w, accs[t])


def gemm_sb_packed(xq, xs, k, w_mx, s_mx, n, acc, fp4=False, n0=0, eoff=None, t0=0, t1=None):
    """`gemm_sb` for the output tiles `t0 .. t1` with tile `t0 + c`'s rows on partitions
    `c*nv ..` of the one PSUM tile `acc` (`[128, 512]` fp32): its stationary is slot c of
    a `stage_x(shifts=)` staging (the rows at columns `c*nv ..`, zero elsewhere), and
    every matmul after the first accumulates (the other tiles' rows add exact zeros).
    Each output element's products and accumulation order are `gemm_sb`'s."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    first = True
    for c, (c0, w) in enumerate(ntiles_of(n)[t0:t1]):
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, 0, n0 + c0, w, fp4,
                                  hw=GEMM_SB_HWDGE, eoff=eoff)
            for j in range(ng):
                kc = g0 + j
                pk = chunks[kc][1] // 4
                nisa.nc_matmul_mx(
                    dst=acc[:, 0:w], stationary=xq[0:pk, c, kc, :], moving=wt[0:pk, j, :],
                    stationary_scale=xs[0:pk, c, kc, :], moving_scale=ws[0:pk, j, :],
                    accumulate=not first,
                )
                first = False


# --------------------------------------------------------------------------- whole GEMMs

#: Row tiles staged together (their accumulators share PSUM: 4 x 512 fp32 = 8 KiB).
MG = 4


def gemm_rows(x_hbm, m, k, w_mx, s_mx, out_hbm, lo, hi, fp4=False, out_dtype=nl.bfloat16,
              out_scale=1.0, x_col0=0, krow0=0, epilogue=None, out_rows=None, nv=None):
    """`out[0:m, lo:hi] = out_scale * x[0:m, x_col0 : x_col0 + k] @ w[lo:hi, krow].T` for
    bf16 HBM rows (`m` a multiple of 128), the activation quantized on chip. `nv`: only
    the first `nv` rows are real (`m == 128`; the result's other rows are zero)."""
    mt = m // PT
    for g0 in range(0, mt, MG):
        nmt = min(MG, mt - g0)
        xq, xs = alloc_x(nmt, k)
        for mi in range(nmt):
            stage_x(x_hbm, (g0 + mi) * PT, k, xq, xs, mi, col0=x_col0, nv=nv if mt == 1 else None)
        gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, g0 * PT, lo, hi, fp4=fp4, out_dtype=out_dtype,
             out_scale=out_scale, krow0=krow0, epilogue=epilogue, out_rows=out_rows)


def gemm_rows_multi(x_hbm, m, k, specs, x_col0=0, nv=None):
    """`gemm_rows` for several weights against the same activation rows, staging (and
    quantizing) each row group once: `specs` is a list of `(w_mx, s_mx, out_hbm, lo, hi)`
    (bf16 outputs). Same arithmetic as one `gemm_rows` per weight."""
    mt = m // PT
    for g0 in range(0, mt, MG):
        nmt = min(MG, mt - g0)
        xq, xs = alloc_x(nmt, k)
        for mi in range(nmt):
            stage_x(x_hbm, (g0 + mi) * PT, k, xq, xs, mi, col0=x_col0, nv=nv if mt == 1 else None)
        for spec in specs:
            # (an optional 6th element: the weight's `prefetch_w` groups)
            w_mx, s_mx, out_hbm, lo, hi = spec[:5]
            gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, g0 * PT, lo, hi,
                 pre=spec[5] if len(spec) > 5 else None)
