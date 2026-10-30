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


def stage_x(x_hbm, r0, k, xq, xs, mi, col0=0, nvalid=PT):
    """Load rows `r0 .. r0+128` (columns `col0 .. col0+k`) of the bf16 HBM `x_hbm`
    and quantize them into slot `mi`. Only the first `nvalid` rows are read: the rest
    are zero rows made on chip (the HBM pad rows are never written nor read)."""
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    if nvalid < PT:
        nisa.memset(rows, 0.0)
        if nvalid > 0:
            nisa.dma_copy(dst=rows[0:nvalid, :], src=x_hbm[r0 : r0 + nvalid, col0 : col0 + k])
    else:
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


#: Weight chunks staged together in one SBUF tile (x4 fp8: 2 KiB a chunk a partition);
#: a group's full chunks load in one data DMA + 4 scale DMAs (`load_w_group`).
WG = 16


#: `gemm_sb` (the routed experts, inside a dynamic loop) issues its weight / scale DMAs
#: from the hardware DGE on the sync engine instead of GpSimd's software DGE (they have
#: static addresses; GpSimd took ~0.75 us to issue each scale DMA).
GEMM_SB_HWDGE = True
#: The static GEMMs' (`gemm`: projections, wo, shared expert) weight / scale DMAs on the
#: hardware DGE too, off GpSimd's in-order stream (whose software-DGE gathers and core
#: barriers queued behind them at decode sizes).
GEMM_STATIC_HWDGE = False


#: The hardware-DGE weight DMAs alternate between the sync and the scalar engine's
#: queues (one queue alone streams the routed experts' weights at ~160 GB/s).
#: (iter-4 retried (sync, scalar), judged by the routed loops' span at in8192_bs10 instead of
#: the noisy total: unchanged, 185 us either way -- not bound by the issuing queue.)
#: (fused launch, the dense experts' one dynamic loop) both queues: each dynamically
#: addressed DMA costs its issuing engine ~1 us (the expert-index register reload + address
#: ALU ops), ~34 a dense expert. (Sync alone with fewer, 32 KB groups measured no better: the
#: weight stream is not issue-bound; ACT's issues may queue behind its SwiGLU work.)
GEMM_SB_HW_ENGINES = (nisa.engine.sync, nisa.engine.scalar)
_wdma_n = [0]


def _wdma(dst, src, hw):
    if hw:
        eng = GEMM_SB_HW_ENGINES[_wdma_n[0] % len(GEMM_SB_HW_ENGINES)]
        _wdma_n[0] += 1
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src)


def load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4, hw=False, eid=None):
    """Chunks `g0 .. g0+ng` of the host-packed weight for output columns `n0 .. n0+w`:
    one `[PT, ng, w]` moving tile and its `[PT, ng, w]` scale tile (quadrant layout,
    zeroed first so every partition is defined inside a dynamic loop). `eid`: a device
    index (a register) into the stacked `[E, K/4, N]` / `[E, K/32, N]` `w_mx` / `s_mx`."""
    wt = nl.ndarray((PT, ng, w), dtype=FP4_X4 if fp4 else FP8_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, ng, w), dtype=nl.uint8, buffer=nl.sbuf)
    # static GEMMs: GpSimd (idle there) zeroes it, off the DVE that stages activations
    nisa.memset(ws, 0, engine=nisa.engine.unknown if hw else nisa.engine.gpsimd)
    _load_w_into(wt, ws, 0, w_mx, s_mx, chunks, g0, ng, krow0, n0, w, hw, eid=eid)
    return wt, ws


def _load_w_into(wt, ws, cb, w_mx, s_mx, chunks, g0, ng, krow0, n0, w, hw, eid=None):
    """`load_w_group`'s DMAs into columns `cb .. cb+w` of the (zeroed) tiles `wt` / `ws`."""
    # The group's full 512-wide chunks are consecutive 128-row blocks of the packed
    # weight (and 16-row blocks of the scales): one DMA for the data and one per
    # quadrant for the scales; a short last chunk on its own.

    def src(t, pattern, offset):
        if eid is None:
            return t.ap(pattern=pattern, offset=offset)
        # the expert's slice of the stacked weight (offset by eid * its stride)
        return t.ap(pattern=pattern, offset=offset, scalar_offset=eid, indirect_dim=0)

    ncol = w_mx.shape[-1]
    scol = s_mx.shape[-1]
    nf = 0
    while nf < ng and chunks[g0 + nf][1] == KC:
        nf += 1
    if nf > 0:
        r0 = krow0 + chunks[g0][0] // 4
        _wdma(wt[0:PT, 0:nf, cb : cb + w],
              src(w_mx, [[ncol, PT], [PT * ncol, nf], [1, w]], r0 * ncol + n0), hw)
        for q in range(4):
            _wdma(ws[32 * q : 32 * q + 4, 0:nf, cb : cb + w],
                  src(s_mx, [[scol, 4], [16 * scol, nf], [1, w]], (r0 // 8 + 4 * q) * scol + n0), hw)
    for j in range(nf, ng):
        k0, kk = chunks[g0 + j]
        pk = kk // 4
        row0 = krow0 + k0 // 4
        grp0 = row0 // 8
        if eid is None:
            _wdma(wt[0:pk, j, cb : cb + w], w_mx[row0 : row0 + pk, n0 : n0 + w], hw)
            for q in range(pk // 32):
                _wdma(ws[32 * q : 32 * q + 4, j, cb : cb + w],
                      s_mx[grp0 + 4 * q : grp0 + 4 * q + 4, n0 : n0 + w], hw)
            continue
        _wdma(wt[0:pk, j, cb : cb + w], src(w_mx, [[ncol, pk], [1, w]], row0 * ncol + n0), hw)
        for q in range(pk // 32):
            _wdma(ws[32 * q : 32 * q + 4, j, cb : cb + w],
                  src(s_mx, [[scol, 4], [1, w]], (grp0 + 4 * q) * scol + n0), hw)


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


def gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, r0, lo, hi, fp4=False, out_dtype=nl.bfloat16,
         out_scale=1.0, krow0=0, epilogue=None):
    """`out[r0 + 128*mi + m, n] = out_scale * sum_k x[m, k] w[n, k]` for the `nmt` staged
    tiles and the output columns `[lo, hi)`.

    `w_mx` is the `[K/4, N]` word view (already the x4 dtype) and `s_mx` the `[K/32, N]`
    scale bytes; `krow0` offsets both for a contraction slice (in units of k/4 rows).
    `epilogue(row0, n0, w, o)`, when given, takes each `(128, w)` SBUF result tile `o`
    (rows `row0 ..`, columns `n0 ..`) instead of the store to `out_hbm`.
    """
    chunks = chunks_of(k)
    nkc = len(chunks)
    for n0, w in ntiles_of(hi - lo):
        n0 += lo
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, fp4,
                                  hw=GEMM_STATIC_HWDGE)
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
            if epilogue is not None:
                epilogue(r0 + mi * PT, n0, w, o)
                continue
            nisa.dma_copy(dst=out_hbm[r0 + mi * PT : r0 + (mi + 1) * PT, n0 : n0 + w], src=o)


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


#: `gemm_sb_multi` (the routed experts) loads the weights of several consecutive output
#: tiles with one DMA group (when the whole contraction is one group, nkc <= WG): up to
#: this many bytes of moving data a partition, so each packed weight row is read in longer
#: contiguous runs (512 fp4 columns are only 1 KB) by fewer DMAs. The matmuls (operands,
#: order) are unchanged. 0: one group per 512-column output tile.
WIDE_W_BYTES_SB = 24 * 1024


def _wide_groups(n, nkc, fp4, limit):
    """`[(c0, w, tiles)]`: `ntiles_of(n)` grouped into weight loads of at most `limit`
    bytes a partition (`tiles` = the group's `[(t0, tw)]`, relative to its `c0`)."""
    eb = 2 if fp4 else 4
    tiles = ntiles_of(n)
    # the fewest groups within the limit, the tiles spread evenly over them
    ngr = max(1, min(len(tiles), cdiv(n * nkc * eb, limit)))
    while ngr < len(tiles) and cdiv(len(tiles), ngr) * NT * nkc * eb > limit:
        ngr += 1
    per = cdiv(len(tiles), ngr)
    out = []
    for i, (c0, w) in enumerate(tiles):
        if i % per:
            out[-1][2].append((out[-1][1], w))
            out[-1][1] += w
        else:
            out.append([c0, w, [(0, w)]])
    return out


def alloc_w_wide(k, n, fp4):
    """`load_w_wide`'s tiles allocated (and the scale tiles zeroed) once, for a caller that
    loads several experts' weights into the same tiles in turn (`load_w_wide(into=...)`):
    every load writes the same partitions, the others stay zero. The next expert's
    loads then wait only for the matmuls reading the tiles (no per-expert memset)."""
    nkc = len(chunks_of(k))
    out = []
    for gc0, gw, tl in _wide_groups(n, nkc, fp4, WIDE_W_BYTES_SB):
        wt = nl.ndarray((PT, nkc, gw), dtype=FP4_X4 if fp4 else FP8_X4, buffer=nl.sbuf)
        ws = nl.ndarray((PT, nkc, gw), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(ws, 0)
        out.append((wt, ws))
    return out


def load_w_wide(w_mx, s_mx, k, n, fp4, krow0=0, n0=0, eid=None, into=None):
    """`gemm_sb_multi`'s wide weight groups (`nkc <= WG`), loaded now: a caller issues
    several GEMMs' weight DMAs ahead of their matmuls (`gemm_sb_multi_pre`). `into`:
    `alloc_w_wide`'s tiles to load into (else new ones)."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    assert WIDE_W_BYTES_SB > 0 and nkc <= WG
    out = []
    for i, (gc0, gw, tl) in enumerate(_wide_groups(n, nkc, fp4, WIDE_W_BYTES_SB)):
        if into is None:
            wt, ws = load_w_group(w_mx, s_mx, chunks, 0, nkc, krow0, n0 + gc0, gw, fp4,
                                  hw=GEMM_SB_HWDGE, eid=eid)
        else:
            wt, ws = into[i]
            _load_w_into(wt, ws, 0, w_mx, s_mx, chunks, 0, nkc, krow0, n0 + gc0, gw,
                         GEMM_SB_HWDGE, eid=eid)
        out.append((gc0, tl, wt, ws))
    return out


def gemm_sb_multi_pre(xq, xs, nmt, k, groups, emit):
    """`gemm_sb_multi`'s matmuls (the same, in the same order) against `load_w_wide`'s
    groups."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    for gc0, tl, wt, ws in groups:
        for t0, w in tl:
            accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
            for kc in range(nkc):
                pk = chunks[kc][1] // 4
                for t in range(nmt):
                    nisa.nc_matmul_mx(
                        dst=accs[t], stationary=xq[0:pk, t, kc, :], moving=wt[0:pk, kc, t0 : t0 + w],
                        stationary_scale=xs[0:pk, t, kc, :], moving_scale=ws[0:pk, kc, t0 : t0 + w],
                        accumulate=(kc > 0),
                    )
            for t in range(nmt):
                emit(t, gc0 + t0, w, accs[t])


def gemm_sb_multi(xq, xs, nmt, k, w_mx, s_mx, n, emit, fp4=False, krow0=0, n0=0, eid=None):
    """`gemm_sb` for the `nmt` staged tiles of `(xq, xs)` sharing each weight group's
    load: `emit(t, c0, w, acc)` takes tile `t`'s fp32 PSUM result for the output
    columns `c0 .. c0 + w`. Each tile's matmuls are `gemm_sb`'s, in the same order.
    `eid`: see `load_w_group` (the wide-group path only)."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    assert eid is None or (WIDE_W_BYTES_SB > 0 and nkc <= WG)
    if WIDE_W_BYTES_SB > 0 and nkc <= WG:
        for gc0, gw, tl in _wide_groups(n, nkc, fp4, WIDE_W_BYTES_SB):
            wt, ws = load_w_group(w_mx, s_mx, chunks, 0, nkc, krow0, n0 + gc0, gw, fp4,
                                  hw=GEMM_SB_HWDGE, eid=eid)
            for t0, w in tl:
                accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
                for kc in range(nkc):
                    pk = chunks[kc][1] // 4
                    for t in range(nmt):
                        nisa.nc_matmul_mx(
                            dst=accs[t], stationary=xq[0:pk, t, kc, :], moving=wt[0:pk, kc, t0 : t0 + w],
                            stationary_scale=xs[0:pk, t, kc, :], moving_scale=ws[0:pk, kc, t0 : t0 + w],
                            accumulate=(kc > 0),
                        )
                for t in range(nmt):
                    emit(t, gc0 + t0, w, accs[t])
        return
    for c0, w in ntiles_of(n):
        accs = [nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum) for _ in range(nmt)]
        for g0 in range(0, nkc, WG):
            ng = min(WG, nkc - g0)
            wt, ws = load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0 + c0, w, fp4,
                                  hw=GEMM_SB_HWDGE)
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


# --------------------------------------------------------------------------- whole GEMMs

#: Row tiles staged together (their accumulators share PSUM: 4 x 512 fp32 = 8 KiB).
MG = 4


def _nvalid(m_valid, r0):
    return PT if m_valid is None else max(0, min(PT, m_valid - r0))


def gemm_rows(x_hbm, m, k, w_mx, s_mx, out_hbm, lo, hi, fp4=False, out_dtype=nl.bfloat16,
              out_scale=1.0, x_col0=0, krow0=0, epilogue=None, m_valid=None):
    """`out[0:m, lo:hi] = out_scale * x[0:m, x_col0 : x_col0 + k] @ w[lo:hi, krow].T` for
    bf16 HBM rows (`m` a multiple of 128), the activation quantized on chip.
    `m_valid`: only rows `< m_valid` are read from `x_hbm`, the others staged as zeros."""
    mt = m // PT
    for g0 in range(0, mt, MG):
        nmt = min(MG, mt - g0)
        xq, xs = alloc_x(nmt, k)
        for mi in range(nmt):
            r0 = (g0 + mi) * PT
            stage_x(x_hbm, r0, k, xq, xs, mi, col0=x_col0, nvalid=_nvalid(m_valid, r0))
        gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, g0 * PT, lo, hi, fp4=fp4, out_dtype=out_dtype,
             out_scale=out_scale, krow0=krow0, epilogue=epilogue)


def gemm_rows_multi(x_hbm, m, k, specs, x_col0=0, m_valid=None):
    """`gemm_rows` for several weights against the same activation rows, staging (and
    quantizing) each row group once: `specs` is a list of `(w_mx, s_mx, out_hbm, lo, hi)`
    (bf16 outputs). Same arithmetic as one `gemm_rows` per weight."""
    mt = m // PT
    for g0 in range(0, mt, MG):
        nmt = min(MG, mt - g0)
        xq, xs = alloc_x(nmt, k)
        for mi in range(nmt):
            r0 = (g0 + mi) * PT
            stage_x(x_hbm, r0, k, xq, xs, mi, col0=x_col0, nvalid=_nvalid(m_valid, r0))
        for w_mx, s_mx, out_hbm, lo, hi in specs:
            gemm(xq, xs, nmt, k, w_mx, s_mx, out_hbm, g0 * PT, lo, hi)
