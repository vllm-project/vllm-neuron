# SPDX-License-Identifier: Apache-2.0
"""The MoE of a decode block in the TP x EP layout (`moe_tp_layout.py`), one row tile.

The deployment spreads the 384 routed experts over the 64 ranks by id, so a decode step's few
rows (6 a sequence) leave most ranks without work and two or three with one or more whole
experts (~45 us of weight streaming each), and the world all-reduce after the MoE waits for
the busiest. Here the experts are split 8 ways over the 8 ranks of a group (`TP`), each group
holding 48 of them (`EP`): a rank runs every active expert of its group, 288 hidden columns of
each, so the work a step's routing gives a group is shared by its 8 ranks, and the shared
expert is split over all 64 (one or two 32-column blocks a rank) instead of replicated.

Per core (LNC=2, both cores run this; `pid` = program id):

  1. the shared expert's block of this core: gate|up GEMM (one 64-column tile), SwiGLU, staged
     for its down GEMM (contraction 32, zero-padded);
  2. the router: bf16 logits GEMM (each core half of the experts; the operands are the
     checkpoint's bf16 values: the same products as the fp32 GEMM), the halves swapped SBUF to
     SBUF (`ROUTER_SR`), then both cores the top-k of every row, the group's dense routing
     weights `[T, 48]` and the active experts' ids (compacted on chip, padded with an
     out-of-bounds id whose DMAs are skipped);
  3. the routed experts in `SLOTS` statically unrolled slot pairs (core c the active expert
     2p + c of pair p; a dynamic loop over the rest, rarely entered): its gate|up GEMM, SwiGLU x
     routing weight, staged; with `MOE_CS` the two cores swap the staged activations and each
     runs every slot's (and the shared expert's two blocks') down GEMM for its half of the
     output columns, accumulated in four PSUM banks, and stores that half of `cc2_src` (else
     each core all columns of its slots, then the column halves combined across the cores).

The values are the deployment's (the same bf16 GEMM outputs, SwiGLU and quantization points of
every hidden value); only the fp32 partial sums of the down projections are split differently
across ranks and cores before the world all-reduce adds them.

The module also holds the hyper-connection pieces the members call for few rows (the
block-diagonal mix GEMM, the spread-layout norm, the eps-padded Sinkhorn and its schedule
anchors) and the engram projection's split over the attention group (`EngramKV`).
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

PT = 128
KC = 512
NT = 512
FP8_X4 = nl.float8_e4m3fn_x4
FP4_X4 = nl.float4_e2m1fn_x4

#: hidden columns of a routed expert a rank holds, and its gate|up tile split (moe_tp_layout)
HID = 288
T0 = 256
#: the down GEMM's contraction: the 288 hidden values on 72 partitions, padded to 128 (512:
#: `nc_matmul_mx` takes 32, 64 or 128 partitions)
HID_KP = 128
#: the shared expert's hidden block a core holds; its down contraction padded to 32 partitions
SH = 32
SH_KP = 32
#: the expert weight DMAs (dynamic addresses) rotate over these hardware-DGE queues
W_QUEUES = (nisa.engine.sync, nisa.engine.scalar)
_q = [0]
#: statically unrolled routed-expert slots a core (core c: active slots c, c + 2, ..): a dynamic
#: loop costs ~10 us an iteration on this hardware (every engine drains its DMA queues and
#: handshakes at the back-edge) and ~20 us to enter and leave, a padding slot ~10 us of PE time
SLOTS = 4
#: the slots past `2 * SLOTS` in a dynamic loop (off: wrong when more experts are active)
OVERFLOW = True
#: the static slots' expert ids read from the compacted list in SBUF (no HBM round trip)
SLOT_IDS_SBUF = True
#: the static slots' routing-weight columns computed on chip (a one-hot matmul) instead of a
#: dynamic-offset DMA each from the HBM copy of the routing weights
SLOT_COLS = True
#: a static slot's weights in 4 DMAs (gate|up words, down words: one dynamic-offset DMA each on
#: the sync queue; the gate|up and the down scale rows: one SWDGE gather each, partition p of
#: the scale tile <- row 16 e + g(p) of `s13`) instead of 10 (each dynamic DMA costs its issuing
#: engine a register load and address arithmetic: ~0.7 us on sync, ~1.3 us on ACT); the padding
#: slots name `PAD_ID`: -1 the zero expert `ng` (in bounds: its weights are loaded and multiply a
#: zero routing weight), None the first id past the stacked experts, any other value that id --
#: out of bounds, the DMAs skipped (`oob_mode.skip`). Out-of-bounds dynamic DMAs pass every
#: launch workspace but hang the full model's decode (a sequencer fault on core 1 once three
#: r1-family launches run the TP x EP MoE; even one expert past the end, even with the bytes
#: there allocated), so the padding slots load the zero expert (+~100 us a launch at in1024_bs1).
SLOT_DMA3 = True
PAD_ID = -1
#: (the scale gathers' rows off the scale partitions: out of bounds, clamped to the first row
#: past the scale tensor)
BIG_ROW = 1 << 20
#: The scale gathers read in-bounds rows on every partition (no out-of-bounds SWDGE gather
#: descriptors: with them the full-model decode hung -- a sequencer fault on core 1 -- once three
#: r1-family launches ran the TP x EP MoE, while every launch workspace passed)
GATHER_INBOUNDS = True
#: `MOE_CS`'s slot pairs from this one on load their weights by SWDGE row gathers (`vector_offset`)
#: whose padding slots' rows are out of bounds and skipped -- the skip the HWDGE dynamic-offset
#: DMAs cannot do in the model (`PAD_ID`; out-of-bounds SWDGE gathers are fine there) -- instead
#: of loading the zero expert; pair 0 (nearly always active) keeps the hardware DGE. None: every
#: slot on the hardware DGE. (L24 launch in1024_bs1: zero-expert loads 2.937 ms, SWDGE from
#: pair 2 2.914, from pair 1 2.883, from pair 0 2.921; out-of-bounds HWDGE 2.842 but unsafe.)
SLOT_SW_FROM = 1
#: the group's dense routing weights by one broadcast compare / multiply / reduce instead of a
#: compare-multiply and an add per top-k pick
ROUTE_VEC = True
#: the engines the down GEMM's accumulator adds alternate over
ADD_ENGINES = (nisa.engine.vector,)


#: Shape-only SBUF constants (block-diagonal masks, spread selectors) shared by every member of
#: one fused launch; the launch clears it at its start (`reset()`): the tiles belong to one trace.
CACHE = {}


def reset():
    CACHE.clear()


def is_tp(w1, w3) -> bool:
    """Whether a block's routed weights are in the decode TP x EP layout."""
    return len(w3.shape) == 2 and tuple(w3.shape) == (1, 1) and len(w1.shape) == 3 and w1.shape[1] == PT


def cdiv(n, d):
    return (n + d - 1) // d


# --------------------------------------------------------------------------- staging


def chunks_of(k):
    out = []
    k0 = 0
    while k0 < k:
        kk = min(KC, k - k0)
        assert kk % PT == 0, f"contraction {k} is not a multiple of 128"
        out.append((k0, kk))
        k0 += kk
    return out


def alloc_x(k):
    nkc = len(chunks_of(k))
    xq = nl.ndarray((PT, 1, nkc, PT), dtype=FP8_X4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, 1, nkc, PT), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(xs, 0)
    return xq, xs


#: `stage_rows(nr=T)` transposes / copies / quantizes only the `nr` real rows: the staged
#: activation's columns past them are left undefined (the down GEMMs read columns 0 .. T of their
#: stationary, or rows < T of their result; a column feeds only its own result row)
STAGE_LIVE = True


def stage_rows(rows, k, xq, xs, nr=PT):
    """Quantize the bf16 SBUF rows `[128, k]` (rows past the real ones zero) into `(xq, xs)`:
    four strided PE transposes a 512-chunk into the j-minor layout, then `quantize_mx`.
    `nr`: the real rows (`STAGE_LIVE`; columns `nr ..` of `xq` are then undefined)."""
    if not STAGE_LIVE:
        nr = PT
    for kc, (k0, kk) in enumerate(chunks_of(k)):
        pk = kk // 4
        tp = nl.ndarray((PT, 4, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(4):
            nisa.nc_transpose(dst=tp[0:pk, j, 0:nr], data=rows.ap(pattern=[[k, nr], [4, pk]], offset=k0 + j))
        xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xT[0:pk, 0:nr, :], src=tp.ap(pattern=[[4 * PT, pk], [1, nr], [PT, 4]]))
        nisa.quantize_mx(dst=xq[0:pk, 0, kc, 0:nr], src=xT.ap(pattern=[[PT * 4, pk], [1, nr * 4]], offset=0),
                         dst_scale=xs[0:pk, 0, kc, 0:nr])


#: `stage_x` packs few rows: G = 128 // T 512-wide chunks share one set of four PE transposes
#: (chunk c's rows on partitions c*T ..; the rows' values and 32-element scale groups unchanged)
STAGE_PACK = True


def _stage_packed(x_hbm, T, k, xq, xs):
    """`mk_mx.stage_x_packed` (shifts 1): rows `0 .. T` into `(xq, xs)`, the other columns zero."""
    chunks = chunks_of(k)
    nkc = len(chunks)
    G = PT // T
    nisa.memset(xq, 0)
    for c0 in range(0, nkc, G):
        g = min(G, nkc - c0)
        q = g * T
        xp = nl.ndarray((PT, KC), dtype=nl.bfloat16, buffer=nl.sbuf)
        mmajor = g == nkc and all(kk == KC for _, kk in chunks) and x_hbm.shape[1] == k
        if mmajor:
            # partition m * nkc + c = the rows' flat 512-element runs: one DMA
            nisa.dma_copy(dst=xp[0:q, :], src=x_hbm.ap(pattern=[[KC, q], [1, KC]], offset=0))
        else:
            for c in range(g):
                k0, kk = chunks[c0 + c]
                nisa.dma_copy(dst=xp[c * T : (c + 1) * T, 0:kk], src=x_hbm[0:T, k0 : k0 + kk])
        tp = nl.ndarray((PT, 4, q), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(4):
            nisa.nc_transpose(dst=tp[:, j, :], data=xp.ap(pattern=[[KC, q], [4, PT]], offset=j))
        xT = nl.ndarray((PT, g, T * 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        if mmajor:
            src = tp.ap(pattern=[[4 * q, PT], [1, g], [g, T], [q, 4]])
            dst = xT.ap(pattern=[[g * T * 4, PT], [T * 4, g], [4, T], [1, 4]])
        else:
            src = tp.ap(pattern=[[4 * q, PT], [1, q], [q, 4]])
            dst = xT.ap(pattern=[[g * T * 4, PT], [4, q], [1, 4]])
        nisa.tensor_copy(dst=dst, src=src)
        for c in range(g):
            pk = chunks[c0 + c][1] // 4
            nisa.quantize_mx(dst=xq[0:pk, 0, c0 + c, 0:T], src=xT[0:pk, c, :], dst_scale=xs[0:pk, 0, c0 + c, 0:T])


def stage_x(x_hbm, T, k):
    """Rows `0 .. T` of the bf16 HBM `x_hbm` (`[>= 128, k]`), quantized for the MX GEMMs."""
    xq, xs = alloc_x(k)
    if STAGE_PACK and PT // T >= 3 and k % PT == 0:
        _stage_packed(x_hbm, T, k, xq, xs)
        return xq, xs
    rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    if T < PT:
        nisa.memset(rows, 0.0, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=rows[0:T, :], src=x_hbm[0:T, 0:k])
    stage_rows(rows, k, xq, xs)
    return xq, xs


# --------------------------------------------------------------------------- pieces


def _wdma(dst, src):
    eng = W_QUEUES[_q[0] % len(W_QUEUES)]
    _q[0] += 1
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)


def _swiglu(g16, u16, wcol, y16, n, limit):
    """`y16[:, 0:n] = bf16(silu(min(g, limit)) * clamp(u, -limit, limit) [* w])` (g16 / u16
    bf16 `[128, n]`, wcol fp32 `[128, 1]` or None): `mk_moe.swiglu_tile`'s arithmetic."""
    g = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g16, op0=nl.minimum, operand0=limit)
    u = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u16, op0=nl.maximum, operand0=-limit, op1=nl.minimum, operand1=limit)
    sg = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    if wcol is not None:
        y = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=sg, data2=u, op=nl.multiply)
        nisa.tensor_scalar(dst=y16[:, 0:n], data=y, op0=nl.multiply, operand0=wcol[0:PT, 0:1])
    else:
        nisa.tensor_tensor(dst=y16[:, 0:n], data1=sg, data2=u, op=nl.multiply)


def _down(yq, ys, kp, w2t, w2s, dim, acc, nr, first):
    """`acc[0:nr] (+)= y @ w2` (one staged chunk of `kp` partitions, fp32 PSUM tiles; the adds
    alternate between the vector and the GpSimd engine)."""
    for j, n0 in enumerate(range(0, dim, NT)):
        w = min(NT, dim - n0)
        a = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul_mx(dst=a[:, 0:w], stationary=yq[0:kp, 0, 0, :], moving=w2t[0:kp, n0 : n0 + w],
                          stationary_scale=ys[0:kp, 0, 0, :], moving_scale=w2s[0:kp, n0 : n0 + w],
                          accumulate=False)
        if first:
            nisa.tensor_copy(dst=acc[0:nr, n0 : n0 + w], src=a[0:nr, 0:w])
        else:
            nisa.tensor_tensor(dst=acc[0:nr, n0 : n0 + w], data1=acc[0:nr, n0 : n0 + w], data2=a[0:nr, 0:w],
                               op=nl.add, engine=ADD_ENGINES[j % len(ADD_ENGINES)])


def shared_up(pid, T, xq, xs, nkc, sh13, shs13, sh2, shs2, dim, limit):
    """This core's shared-expert block: its gate|up GEMM and SwiGLU, staged for the down GEMM
    (`(yq, ys, w2t, w2s)`: a 32-partition contraction chunk and the down weights)."""
    W = 2 * SH
    key = ("shared_bufs", nkc, dim)
    if not ROUTED_PERSIST or key not in CACHE:
        wt = nl.ndarray((PT, nkc, W), dtype=FP8_X4, buffer=nl.sbuf)
        ws = nl.ndarray((PT, nkc, W), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(ws, 0, engine=nisa.engine.gpsimd)
        w2t = nl.ndarray((PT, dim), dtype=FP8_X4, buffer=nl.sbuf)
        nisa.memset(w2t[0:SH_KP, :], 0, engine=nisa.engine.gpsimd)
        w2s = nl.ndarray((PT, dim), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(w2s[0:SH_KP, :], 0, engine=nisa.engine.gpsimd)
        y16 = nl.ndarray((PT, 4 * SH_KP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(y16, 0.0)
        if ROUTED_PERSIST:
            CACHE[key] = (wt, ws, w2t, w2s, y16)
    if ROUTED_PERSIST:
        wt, ws, w2t, w2s, y16 = CACHE[key]
    nisa.dma_copy(dst=wt, src=sh13.view(FP8_X4).ap(pattern=[[nkc * W, PT], [W, nkc], [1, W]],
                                                    offset=pid * PT * nkc * W),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    for q in range(4):
        nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, :, :],
                      src=shs13.ap(pattern=[[nkc * W, 4], [W, nkc], [1, W]], offset=(pid * 16 + 4 * q) * nkc * W),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    nisa.dma_copy(dst=w2t[0 : SH // 4, :], src=sh2.view(FP8_X4)[pid, :, :],
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    nisa.dma_copy(dst=w2s[0:1, :], src=shs2[pid, :, :], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    a = nl.ndarray((PT, W), dtype=nl.float32, buffer=nl.psum)
    for kc in range(nkc):
        nisa.nc_matmul_mx(dst=a, stationary=xq[0:PT, 0, kc, :], moving=wt[0:PT, kc, :],
                          stationary_scale=xs[0:PT, 0, kc, :], moving_scale=ws[0:PT, kc, :],
                          accumulate=(kc > 0))
    g16 = nl.ndarray((PT, SH), dtype=nl.bfloat16, buffer=nl.sbuf)
    u16 = nl.ndarray((PT, SH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g16, src=a[:, 0:SH])
    nisa.tensor_copy(dst=u16, src=a[:, SH:W])
    _swiglu(g16, u16, None, y16, SH, limit)
    yq, ys = alloc_x(4 * SH_KP)
    stage_rows(y16, 4 * SH_KP, yq, ys, nr=T)
    return yq, ys, w2t, w2s, SH_KP


def down_sum(terms, dim, acc, nr, first):
    """`acc[0:nr] (+)= sum_t y_t @ w2_t` over the `terms` `(yq, ys, w2t, w2s, kp)` (one staged
    chunk each), the terms of each 512-column tile accumulated in PSUM: one add a tile."""
    for n0 in range(0, dim, NT):
        w = min(NT, dim - n0)
        a = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
        for j, (yq, ys, w2t, w2s, kp) in enumerate(terms):
            nisa.nc_matmul_mx(dst=a[:, 0:w], stationary=yq[0:kp, 0, 0, :], moving=w2t[0:kp, n0 : n0 + w],
                              stationary_scale=ys[0:kp, 0, 0, :], moving_scale=w2s[0:kp, n0 : n0 + w],
                              accumulate=(j > 0))
        if first:
            nisa.tensor_copy(dst=acc[0:nr, n0 : n0 + w], src=a[0:nr, 0:w])
        else:
            nisa.tensor_tensor(dst=acc[0:nr, n0 : n0 + w], data1=acc[0:nr, n0 : n0 + w], data2=a[0:nr, 0:w],
                               op=nl.add)


#: the two cores' router-logit halves swapped SBUF to SBUF (`nisa.sendrecv`) into one `[T, n_experts]`
#: tile instead of through the HBM `logits` (a store, a core barrier and a load)
ROUTER_SR = True
#: DMA QoS priority of the routed slots' weight loads (on the MoE's critical path; the next block's
#: weight prefetches run alongside them); None: the default
W_PRIORITY = 0


def _wprio():
    return {} if W_PRIORITY is None else dict(priority=W_PRIORITY)


def router_logits(pid, T, xn2, gate_w, logits, dim, n_experts, lg_sb=None):
    """`logits[0:T, half] = xn2[0:T] @ gate_w[:, half]` for this core's half of the experts
    (bf16 operands, fp32 PSUM: the products of the fp32 router GEMM, exactly)."""
    nh = n_experts // 2
    n0 = pid * nh
    nk = dim // PT
    wT = nl.ndarray((PT, nk, nh), dtype=gate_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wT, src=gate_w.ap(pattern=[[n_experts, PT], [PT * n_experts, nk], [1, nh]], offset=n0))
    x = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x[0:T, :], src=xn2[0:T, 0:dim])
    acc = nl.ndarray((PT, nh), dtype=nl.float32, buffer=nl.psum)
    stat_dt = nl.bfloat16 if gate_w.dtype == nl.bfloat16 else nl.float32
    if T <= 16:
        t16 = nl.ndarray((PT, nk, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_transpose(dst=t16, src=x.ap(pattern=[[dim, 16], [PT, nk], [1, PT]], offset=0), axes=(2, 1, 0))
        if stat_dt == nl.float32:
            t32 = nl.ndarray((PT, nk, 16), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t32[0:PT, :, 0:T], src=t16[0:PT, :, 0:T])
            t16 = t32
        for kb in range(nk):
            nisa.nc_matmul(dst=acc[0:T, 0:nh], stationary=t16[0:PT, kb, 0:T], moving=wT[0:PT, kb, 0:nh],
                           accumulate=(kb > 0))
    else:
        for kb in range(nk):
            if kb % 4 == 0:
                nb_ = min(4, nk - kb)
                tps = nl.ndarray((PT, 4, PT), dtype=nl.bfloat16, buffer=nl.psum)
                for j in range(nb_):
                    nisa.nc_transpose(dst=tps[0:PT, j, 0:T], data=x[0:T, (kb + j) * PT : (kb + j + 1) * PT])
                xT4 = nl.ndarray((PT, 4, PT), dtype=stat_dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=xT4[0:PT, 0:nb_, 0:T], src=tps[0:PT, 0:nb_, 0:T])
            nisa.nc_matmul(dst=acc[0:T, 0:nh], stationary=xT4[0:PT, kb % 4, 0:T], moving=wT[0:PT, kb, 0:nh],
                           accumulate=(kb > 0))
    if lg_sb is not None:
        # this core's half into the shared tile, the other core's from it
        nisa.tensor_copy(dst=lg_sb[0:T, n0 : n0 + nh], src=acc[0:T, :])
        m0 = (1 - pid) * nh
        nisa.sendrecv(src=lg_sb[0:T, n0 : n0 + nh], dst=lg_sb[0:T, m0 : m0 + nh], send_to_rank=1 - pid,
                      recv_from_rank=1 - pid, pipe_id=12)
        return
    o = nl.ndarray((PT, nh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o[0:T, :], src=acc[0:T, :])
    nisa.dma_copy(dst=logits[0:T, n0 : n0 + nh], src=o[0:T, :])


#: `route_group`'s weight normalization reciprocal on the vector engine (else `nl.reciprocal`, an
#: ACT table switch between the router's softplus / sqrt and the SwiGLUs' silu)
ROUTE_RCP_DVE = True


def route_group(logits, gate_b, glo, T, n_experts, topk, ng, route_scale, lg_sb=None, keep=False, sqrt_dve=False,
                bias_sb=None, lo_sb=None, gather_inbounds=None):
    """The router's selection (`mk_common.route_topk`'s arithmetic) and the group's dense
    routing weights: `(wexp_hbm [128, ng + 1] fp32 private HBM (rows past T and column ng
    zero), alist [1, ng + 1] int32 private HBM (the active local experts ascending, padded with
    ng, the count at column ng), nact (a device scalar))`."""
    nr = T
    if lg_sb is not None:
        lg = lg_sb
    else:
        lg = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lg[0:nr, :], src=logits[0:nr, 0:n_experts])
    sp = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sp[0:nr, :], op=nl.softplus, data=lg[0:nr, :])
    scores = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    if sqrt_dve:
        # (the square root on the vector engine: no ACT table switch after the softplus)
        nisa.tensor_scalar(dst=scores[0:nr, :], data=sp[0:nr, :], op0=nl.power, operand0=0.5)
    else:
        nisa.activation(dst=scores[0:nr, :], op=nl.sqrt, data=sp[0:nr, :])
    # (`bias_sb`: the bias already on the rows' partitions, fp32 -- loaded early by the caller)
    bb = bias_sb[0:nr, 0:n_experts] if bias_sb is not None else nl.broadcast_to(
        nl.load(gate_b[0:1, 0:n_experts], dtype=nl.float32), (nr, n_experts))
    sel = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sel[0:nr, :], data1=scores[0:nr, :], data2=bb, op=nl.add)
    best = nl.ndarray((PT, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=best[0:nr, :], src=sel[0:nr, :])
    idx8 = nl.ndarray((PT, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=idx8[0:nr, :], data=sel[0:nr, :], vals=best[0:nr, :])
    picked = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=picked[0:nr, :], data=scores[0:nr, :], indices=idx8[0:nr, 0:topk])
    tot = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tot[0:nr, :], op=nl.add, data=picked[0:nr, :], axis=(1,), keepdims=True)
    inv = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    if ROUTE_RCP_DVE:
        # (the reciprocal on the vector engine: no ACT table switch in the routing's chain)
        ts = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ts[0:nr, :], data=tot[0:nr, :], op0=nl.add, operand0=1e-20)
        rc = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rc[0:nr, :], data=ts[0:nr, :])
        nisa.tensor_scalar(dst=inv[0:nr, :], data=rc[0:nr, :], op0=nl.multiply, operand0=route_scale)
    else:
        inv[0:nr, :] = nl.multiply(nl.reciprocal(nl.add(tot[0:nr, :], 1e-20)), route_scale)
    wts = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=wts[0:nr, :], data=picked[0:nr, :], op0=nl.multiply, operand0=inv[0:nr, 0:1])
    idf = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idf[0:nr, :], src=idx8[0:nr, 0:topk])
    # the group's expert ids glo + e as fp32 (exact: < 2^24)
    ei = nl.ndarray((PT, ng), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ei, pattern=[[1, ng]], offset=0, channel_multiplier=0)
    ef = nl.ndarray((PT, ng), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ef, src=ei)
    # (`lo_sb`: glo on the rows' partitions, fp32 -- loaded early by the caller)
    lo = lo_sb[0:nr, 0:1] if lo_sb is not None else nl.broadcast_to(nl.load(glo[0:1, 0:1], dtype=nl.float32), (nr, 1))
    nisa.tensor_scalar(dst=ef[0:nr, :], data=ef[0:nr, :], op0=nl.add, operand0=lo)
    # wsb[t, e] = sum_k (idf[t, k] == glo + e) * wts[t, k] (at most one term is nonzero)
    # (columns ng, ng + 1 stay zero: the zero expert's and the padding id's)
    wsb = nl.ndarray((PT, ng + 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(wsb, 0.0)
    if ROUTE_VEC:
        # [row, e, k]: (glo + e == id_k) * w_k, summed over k (broadcast operands, 3 instructions)
        oh = nl.ndarray((PT, ng, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=oh[0:nr, :, :], data1=ef.ap(pattern=[[ng, nr], [1, ng], [0, topk]], offset=0),
                           data2=idf.ap(pattern=[[topk, nr], [0, ng], [1, topk]], offset=0), op=nl.equal)
        nisa.tensor_tensor(dst=oh[0:nr, :, :], data1=oh[0:nr, :, :],
                           data2=wts.ap(pattern=[[topk, nr], [0, ng], [1, topk]], offset=0), op=nl.multiply)
        nisa.tensor_reduce(dst=wsb[0:nr, 0:ng], op=nl.add, data=oh[0:nr, :, :], axis=(2,))
    else:
        for k in range(topk):
            m = nl.ndarray((PT, ng), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m[0:nr, :], data=ef[0:nr, :], op0=nl.equal, operand0=idf[0:nr, k : k + 1],
                               op1=nl.multiply, operand1=wts[0:nr, k : k + 1])
            nisa.tensor_tensor(dst=wsb[0:nr, 0:ng], data1=wsb[0:nr, 0:ng], data2=m[0:nr, :], op=nl.add)
    wexp = nl.ndarray((PT, ng + 2), dtype=nl.float32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=wexp, src=wsb)
    # the active experts: any row's weight > 0 (rows past T are zero)
    m16 = nl.ndarray((PT, ng), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m16, data=wsb[:, 0:ng], op0=nl.greater, operand0=0.0)
    ones = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    cnt = nl.ndarray((1, ng), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt, stationary=ones, moving=m16)
    flag = nl.ndarray((PT, ng), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(flag, 0.0)
    nisa.tensor_scalar(dst=flag[0:1, :], data=cnt, op0=nl.greater, operand0=0.0)
    nz = nl.ndarray((PT, ng + 3), dtype=nl.int32, buffer=nl.sbuf)
    # (the padding id: the first one past the stacked experts -- the zero expert `ng` is the last --
    # so the skipped DMAs' addresses stay just past their tensors; an id far out (1 << 12) put them
    # gigabytes past, beyond the device memory for a tensor allocated high, where the DMA faulted
    # instead of being skipped: the decode hung once enough blocks' TP weights were allocated)
    pad = ({None: ng + 1, -1: ng}.get(PAD_ID, PAD_ID)) if (SLOT_DMA3 and SLOT_COLS) else ng
    nisa.memset(nz[0:1, ng + 1 : ng + 3], pad)
    nisa.nonzero_with_count(dst=nz[:, 0 : ng + 1], src=flag, index_offset=0, padding_val=pad)
    # (the count at column ng; slots ng + 1, ng + 2: padding)
    alist = nl.ndarray((1, ng + 3), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=alist, src=nz[0:1, :])
    nact = nisa.scalar_load(nz[0:1, ng : ng + 1])
    slot = None
    if SLOT_COLS:
        # the static slots' routing-weight columns on chip: one-hot [e, k] = (alist[k] == e) from
        # the compacted ids broadcast over the experts' partitions, then wsb @ one-hot
        S2 = 2 * SLOTS
        nzf = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nzf[0:1, :], src=nz[0:1, 0:S2])
        nb = PT if SLOT_DMA3 else ng + 1
        onesr = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(onesr[0:1, :], 1.0)
        bc = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=bc[0:nb, :], stationary=onesr[0:1, 0:nb], moving=nzf[0:1, :])
        pi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=pi, pattern=[[1, 1]], offset=0, channel_multiplier=1)
        pf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pf, src=pi)
        oh = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oh[0 : ng + 1, :], data=bc[0 : ng + 1, :], op0=nl.equal, operand0=pf[0 : ng + 1, 0:1])
        wT = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=wT[0 : ng + 1, 0:PT], data=wsb[0:PT, 0 : ng + 1])
        wTs = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wTs[0 : ng + 1, :], src=wT[0 : ng + 1, :])
        sp = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sp[0:PT, :], stationary=wTs[0 : ng + 1, 0:PT], moving=oh[0 : ng + 1, :])
        slot = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slot, src=sp)
        if SLOT_DMA3:
            # the scale gathers' row indices: [p, j] = 16 * id_j + g(p) (out of bounds off the
            # scale partitions and for the padding ids)
            sf = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
            if SLOT_NOZERO:
                # (the padding slots' scale rows: the zero expert's, finite codes for the
                # uninitialized weight tiles their skipped weight DMAs leave)
                ce = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ce, data=bc, op0=nl.minimum, operand0=float(ng))
                src_bc = ce
            else:
                src_bc = bc
            nisa.tensor_scalar(dst=sf, data=src_bc, op0=nl.multiply, operand0=16.0, op1=nl.add,
                               operand1=_scale_rows_base(gather_inbounds)[:, 0:1])
            # (the out-of-bounds rows clamped to the first row past the scale tensor -- its ng + 1
            # experts' 16 rows -- for the same reason as the padding id)
            nisa.tensor_scalar(dst=sf, data=sf, op0=nl.minimum, operand0=float((ng + 1) * 16))
            sidx = nl.ndarray((PT, S2), dtype=nl.int32, buffer=nl.sbuf)
            ROUTE_END[0] = nisa.tensor_copy(dst=sidx, src=sf)
            slot = (slot, sidx)
            if SLOT_SW_FROM is not None:
                # the SWDGE slots' weight rows: [p, j] = PT * id_j + p (gate|up) and HID / 4 * id_j + p
                # (down), pushed out of bounds for the padding id (the zero expert ng)
                off = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=off, data=bc, op0=nl.greater_equal, operand0=float(ng), op1=nl.multiply,
                                   operand1=float(4 * (ng + 1) * PT))
                rows = []
                for rpe in (PT, HID // 4):
                    wf = nl.ndarray((PT, S2), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=wf, data=bc, op0=nl.multiply, operand0=float(rpe), op1=nl.add,
                                       operand1=pf[0:PT, 0:1])
                    nisa.tensor_tensor(dst=wf, data1=wf, data2=off, op=nl.add)
                    wi = nl.ndarray((PT, S2), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=wi, src=wf)
                    rows.append(wi)
                slot = (slot[0], sidx, rows[0], rows[1])
                if keep:
                    slot = slot + (wsb,)
    return wexp, alist, nact, nz, slot


#: the routing's last vector-engine instruction (the scale rows' index tile), for callers' schedule pins
ROUTE_END = [None]


def _scale_rows_base(inbounds=None):
    """`[p] = 4 q + r` for the scale partitions p = 32 q + r (r < 4), `BIG_ROW` elsewhere (fp32).
    `inbounds`: `GATHER_INBOUNDS` for this caller (None: the module's)."""
    inb = GATHER_INBOUNDS if inbounds is None else inbounds
    key = ("scale_rows_base", inb)
    if key in CACHE:
        return CACHE[key]
    pi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pi, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    r = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=pi, op0=nl.bitwise_and, operand0=31)
    q = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q, data=pi, op0=nl.right_shift, operand0=5)
    rf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rf, src=r)
    qf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qf, src=q)
    v = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    if inb:
        # every partition a valid row of the slot's expert: the off-quadrant partitions (r >= 4)
        # re-read row 4 q + (r & 3), scale codes the MX matmuls never read
        r3 = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r3, data=pi, op0=nl.bitwise_and, operand0=3)
        r3f = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r3f, src=r3)
        base = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=base, data=qf, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=r3f)
        CACHE[key] = base
        return base
    nisa.scalar_tensor_tensor(dst=v, data=qf, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=rf)
    # off: BIG_ROW where r >= 4
    off = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=off, data=rf, op0=nl.greater_equal, operand0=4.0, op1=nl.multiply,
                       operand1=float(BIG_ROW))
    base = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=base, data1=v, data2=off, op=nl.add)
    CACHE[key] = base
    return base


class _ExpertBufs:
    """The routed loop's SBUF weight tiles, allocated once outside the loop: each iteration
    computes from them and refills them for its successor (the parts no DMA writes stay zero --
    scale codes of unused groups, w2 partitions past the 72 hidden rows, the zero-padded
    hidden columns)."""

    def __init__(self, nkc, dim):
        W = 2 * HID
        self.wt = nl.ndarray((PT, nkc, W), dtype=FP4_X4, buffer=nl.sbuf)
        self.ws = nl.ndarray((PT, nkc * W), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(self.ws, 0)
        self.w2t = nl.ndarray((PT, dim), dtype=FP4_X4, buffer=nl.sbuf)
        nisa.memset(self.w2t, 0, engine=nisa.engine.gpsimd)
        self.w2s = nl.ndarray((PT, dim), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(self.w2s, 0, engine=nisa.engine.gpsimd)
        self.y16 = nl.ndarray((PT, 4 * HID_KP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(self.y16, 0.0)


#: the routed slots' scale DMAs on the scalar engine's hardware-DGE queue (else GpSimd's software
#: DGE, ~0.7 us to issue each)
SCALES_HWDGE = True


def _scale_q():
    if SCALES_HWDGE:
        return dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    return dict(dge_mode=nisa.dge_mode.swdge)


#: the first slot pair's gate|up weights (the routing's critical path: nothing to overlap them
#: with) in two DMAs, halves of the contraction on the sync and the ACT queues
SLOT0_SPLIT = True


def _load_up(e, nkc, dim, w13v, s13, bufs, sidx=None, split=False, wx=None):
    """Expert `e`'s gate|up weights and their scales into `bufs` (sync / GpSimd queues).
    `sidx`: the slot's scale-row column (`SLOT_DMA3`). `wx`: the slot's weight rows (`SLOT_SW_FROM`:
    one SWDGE row gather, skipped for the padding slots, instead of the dynamic-offset DMA)."""
    W = 2 * HID
    nks = nkc * W + dim
    nkh = nkc // 2
    if sidx is not None:
        if wx is not None:
            w13r = w13v.reshape((w13v.shape[0] * PT, nkc * W))
            nisa.dma_copy(dst=bufs.wt[0:PT, 0:nkc, :],
                          src=w13r.ap(pattern=[[nkc * W, PT], [W, nkc], [1, W]], offset=0, vector_offset=wx,
                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
        else:
            parts = (((0, nkh), nisa.engine.sync), ((nkh, nkc), nisa.engine.scalar)) if split else (((0, nkc), nisa.engine.sync),)
            for (k0, k1), q in parts:
                nisa.dma_copy(dst=bufs.wt[0:PT, k0:k1, :],
                              src=w13v.ap(pattern=[[nkc * W, PT], [W, k1 - k0], [1, W]], offset=k0 * W,
                                          scalar_offset=e, indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=q,
                              oob_mode=nisa.oob_mode.skip, **_wprio())
        s13r = s13.reshape((s13.shape[0] * 16, nks))
        nisa.dma_copy(dst=bufs.ws[0:PT, :],
                      src=s13r.ap(pattern=[[nks, PT], [1, nkc * W]], offset=0, vector_offset=sidx, indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
        return
    for k0, k1, q in ((0, nkh, nisa.engine.sync), (nkh, nkc, nisa.engine.sync)):
        nisa.dma_copy(dst=bufs.wt[0:PT, k0:k1, :],
                      src=w13v.ap(pattern=[[nkc * W, PT], [W, k1 - k0], [1, W]], offset=k0 * W, scalar_offset=e,
                                  indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=q, oob_mode=nisa.oob_mode.skip)
    for q in range(4):
        nisa.dma_copy(dst=bufs.ws[32 * q : 32 * q + 4, :],
                      src=s13.ap(pattern=[[nks, 4], [1, nkc * W]], offset=4 * q * nks, scalar_offset=e, indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip, **_scale_q())


def _load_down(e, nkc, dim, w2v, s13, bufs, sidx=None):
    """Expert `e`'s down weights and scales into `bufs`."""
    W = 2 * HID
    kp = HID // 4
    nks = nkc * W + dim
    if sidx is not None:
        nisa.dma_copy(dst=bufs.w2t[0:kp, :], src=w2v.ap(pattern=[[dim, kp], [1, dim]], offset=0, scalar_offset=e,
                                                        indirect_dim=0), dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip)
        s13r = s13.reshape((s13.shape[0] * 16, nks))
        nisa.dma_copy(dst=bufs.w2s[0:PT, :],
                      src=s13r.ap(pattern=[[nks, PT], [1, dim]], offset=nkc * W, vector_offset=sidx, indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
        return
    nisa.dma_copy(dst=bufs.w2t[0:kp, :], src=w2v.ap(pattern=[[dim, kp], [1, dim]], offset=0, scalar_offset=e,
                                                    indirect_dim=0), dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip)
    for q, n in ((0, 4), (1, 4), (2, 1)):
        nisa.dma_copy(dst=bufs.w2s[32 * q : 32 * q + n, :],
                      src=s13.ap(pattern=[[nks, n], [1, dim]], offset=4 * q * nks + nkc * W, scalar_offset=e,
                                 indirect_dim=0), oob_mode=nisa.oob_mode.skip, **_scale_q())


def _slot_ids(alist, ng, i, offs):
    """The active-expert ids at slots `i + o` (device scalars), from one DMA."""
    n = max(offs) + 1
    ec = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.sbuf)
    if isinstance(i, int):
        nisa.dma_copy(dst=ec, src=alist[0:1, i : i + n])
    else:
        nisa.dma_copy(dst=ec, src=alist.ap(pattern=[[ng + 3, 1], [1, n]], offset=0, scalar_offset=i, indirect_dim=1))
    return [nisa.scalar_load(ec[0:1, o : o + 1]) for o in offs]


#: the routed slots' SBUF weight / activation buffers allocated (and their never-written parts
#: zeroed) once a launch, shared by its blocks (`CACHE`): the per-block memsets (~8 us of GpSimd
#: a core) sat in GpSimd's queue ahead of the block's core barriers and gathers
ROUTED_PERSIST = False


#: (`SLOT_DMA3`) no zeroing of the routed weight / scale tiles: the scale gathers write all 16
#: scale partitions of every slot (the padding slots the zero expert's codes), the MX matmuls read
#: no others, and the down weights' padding partitions 72.. (finite fp4 codes, whatever they
#: hold) meet the activation's zero contraction padding
SLOT_NOZERO = True


def _nozero():
    return SLOT_NOZERO and SLOT_DMA3 and SLOT_COLS


class _UpBufs:
    def __init__(self, nkc):
        W = 2 * HID
        self.wt = nl.ndarray((PT, nkc, W), dtype=FP4_X4, buffer=nl.sbuf)
        self.ws = nl.ndarray((PT, nkc * W), dtype=nl.uint8, buffer=nl.sbuf)
        if not _nozero():
            nisa.memset(self.ws, 0)


class _DownBufs:
    def __init__(self, dim):
        self.w2t = nl.ndarray((PT, dim), dtype=FP4_X4, buffer=nl.sbuf)
        self.w2s = nl.ndarray((PT, dim), dtype=nl.uint8, buffer=nl.sbuf)
        if not _nozero():
            nisa.memset(self.w2t, 0, engine=nisa.engine.gpsimd)
            nisa.memset(self.w2s, 0, engine=nisa.engine.gpsimd)


#: `_up`'s gate|up GEMM as two equal 288-column matmuls a chunk instead of 512 + 64 (each
#: instruction's fixed cost paid on half the columns; the same products and accumulation)
UP_EVEN = False


def _up(T, xq, xs, nkc, limit, ub, wcol, y16):
    """A loaded expert's gate|up GEMM and SwiGLU x routing weight, staged: `(yq, ys)`."""
    W = 2 * HID
    # (the W columns: gate 0 .. T0 | up 0 .. T0 | gate T0 .. HID | up T0 .. HID)
    S0 = W // 2 if UP_EVEN else 2 * T0
    a0 = nl.ndarray((PT, S0), dtype=nl.float32, buffer=nl.psum)
    a1 = nl.ndarray((PT, W - S0), dtype=nl.float32, buffer=nl.psum)
    for kc in range(nkc):
        nisa.nc_matmul_mx(dst=a0, stationary=xq[0:PT, 0, kc, :], moving=ub.wt[0:PT, kc, 0:S0],
                          stationary_scale=xs[0:PT, 0, kc, :], moving_scale=ub.ws[0:PT, kc * W : kc * W + S0],
                          accumulate=(kc > 0))
        nisa.nc_matmul_mx(dst=a1, stationary=xq[0:PT, 0, kc, :], moving=ub.wt[0:PT, kc, S0:W],
                          stationary_scale=xs[0:PT, 0, kc, :],
                          moving_scale=ub.ws[0:PT, kc * W + S0 : (kc + 1) * W], accumulate=(kc > 0))
    g16 = nl.ndarray((PT, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
    u16 = nl.ndarray((PT, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
    r = HID - T0
    if UP_EVEN:
        # a0 = gate 0 .. T0 | up 0 .. S0 - T0; a1 = up S0 - T0 .. T0 | gate T0 .. HID | up T0 .. HID
        e = S0 - T0
        q = 2 * T0 - S0
        nisa.tensor_copy(dst=g16[:, 0:T0], src=a0[:, 0:T0])
        nisa.tensor_copy(dst=u16[:, 0:e], src=a0[:, T0:S0])
        nisa.tensor_copy(dst=u16[:, e:T0], src=a1[:, 0:q])
        nisa.tensor_copy(dst=g16[:, T0:HID], src=a1[:, q : q + r])
        nisa.tensor_copy(dst=u16[:, T0:HID], src=a1[:, q + r : q + 2 * r])
    else:
        nisa.tensor_copy(dst=g16[:, 0:T0], src=a0[:, 0:T0])
        nisa.tensor_copy(dst=u16[:, 0:T0], src=a0[:, T0 : 2 * T0])
        nisa.tensor_copy(dst=g16[:, T0:HID], src=a1[:, 0:r])
        nisa.tensor_copy(dst=u16[:, T0:HID], src=a1[:, r : 2 * r])
    _swiglu(g16, u16, wcol, y16, HID, limit)
    yq, ys = alloc_x(4 * HID_KP)
    stage_rows(y16, 4 * HID_KP, yq, ys, nr=T)
    return yq, ys


def _wcol(e, wexp, ng):
    wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    if SLOT_DMA3 and SLOT_COLS:
        # (the padding id is out of bounds: its column stays zero)
        nisa.memset(wcol, 0.0)
    nisa.dma_copy(dst=wcol, src=wexp.ap(pattern=[[ng + 2, PT], [1, 1]], offset=0, scalar_offset=e, indirect_dim=1),
                  dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
    return wcol


def routed(pid, T, xq, xs, nkc, w13, s13, w2, wexp, alist, nact, ng, dim, limit, acc, sh_term, nz=None, slot=None):
    """The group's active experts into `acc` (which this initializes with the shared expert's
    term `sh_term`): `SLOTS` statically unrolled slots a core (core c: active slots c, c + 2,
    ..; those past the count name the zero expert `ng`), two at a time with their down GEMMs
    accumulated in PSUM; the slots past `2 * SLOTS` (when that many experts are active) in a
    dynamic loop, two an iteration."""
    w13v = w13.view(FP4_X4)
    w2v = w2.view(FP4_X4)
    S = SLOTS
    if nz is not None:
        # (the ids straight from the compacted SBUF list)
        ids = [nisa.scalar_load(nz[0:1, pid + 2 * k : pid + 2 * k + 1]) for k in range(S)]
    else:
        ids = _slot_ids(alist, ng, 0, tuple(pid + 2 * k for k in range(S)))
    key = ("routed_bufs", nkc, dim)
    if not ROUTED_PERSIST or key not in CACHE:
        ups = [_UpBufs(nkc) for _ in range(2)]
        downs = [_DownBufs(dim) for _ in range(2)]
        y16s = []
        for _ in range(2):
            y = nl.ndarray((PT, 4 * HID_KP), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(y, 0.0)
            y16s.append(y)
        if ROUTED_PERSIST:
            CACHE[key] = (ups, downs, y16s)
    if ROUTED_PERSIST:
        ups, downs, y16s = CACHE[key]
    sidx = None
    if isinstance(slot, tuple):
        slot, sidx = slot[0], slot[1]

    def wcol_of(k):
        return slot[:, pid + 2 * k : pid + 2 * k + 1] if slot is not None else _wcol(ids[k], wexp, ng)

    def sx(k):
        return sidx[0:PT, pid + 2 * k : pid + 2 * k + 1] if sidx is not None else None

    # software-pipelined: slot k's gate|up weights stream into its up-buffer as soon as slot
    # k - 2's gate|up GEMM has read it, its down weights once pair k // 2 - 1's down GEMMs have
    for k in range(min(S, 2)):
        _load_up(ids[k], nkc, dim, w13v, s13, ups[k % 2], sx(k), split=SLOT0_SPLIT)
        _load_down(ids[k], nkc, dim, w2v, s13, downs[k % 2], sx(k))
    first = True
    for k0 in range(0, S, 2):
        terms = [] if not first or sh_term is None else [sh_term]
        ks = list(range(k0, min(S, k0 + 2)))
        for k in ks:
            b = k % 2
            yq, ys = _up(T, xq, xs, nkc, limit, ups[b], wcol_of(k), y16s[b])
            terms.append((yq, ys, downs[b].w2t, downs[b].w2s, HID_KP))
            if k + 2 < S:
                _load_up(ids[k + 2], nkc, dim, w13v, s13, ups[b], sx(k + 2))
        down_sum(terms, dim, acc, T, first)
        first = False
        for k in ks:
            if k + 2 < S:
                _load_down(ids[k + 2], nkc, dim, w2v, s13, downs[k % 2], sx(k + 2))

    def body(i):
        (e,) = _slot_ids(alist, ng, i, (pid,))
        _load_up(e, nkc, dim, w13v, s13, ups[0])
        _load_down(e, nkc, dim, w2v, s13, downs[0])
        wc = _wcol(e, wexp, ng)
        yq, ys = _up(T, xq, xs, nkc, limit, ups[0], wc, y16s[0])
        down_sum([(yq, ys, downs[0].w2t, downs[0].w2s, HID_KP)], dim, acc, T, False)

    if OVERFLOW:
        n2 = ((nact + 1) // 2) * 2
        nl.fori_loop(2 * S, n2, body, step=2)


#: `combine` SBUF to SBUF: each core sends the other its column half of the partial sums
#: (`nisa.sendrecv`) and the DMA engine adds the two halves as it writes `cc2_src`
#: (`nisa.dma_compute`), instead of an HBM round trip through `part`, a core barrier and a
#: DVE add on T partitions
COMBINE_SR = True


def combine(pid, T, acc, part, cc2_src, dim):
    """`cc2_src[0:T] = core 0's acc + core 1's`, each core its column half."""
    dh = dim // 2
    own = (pid * dh, (pid + 1) * dh)
    oth = ((1 - pid) * dh, (2 - pid) * dh)
    if COMBINE_SR:
        bv = nl.ndarray((PT, dh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=acc[0:T, oth[0] : oth[1]], dst=bv[0:T, :], send_to_rank=1 - pid, recv_from_rank=1 - pid,
                      pipe_id=0)
        a_ = acc[0:T, own[0] : own[1]]
        srcs = [a_, bv[0:T, :]] if pid == 0 else [bv[0:T, :], a_]
        nisa.dma_compute(dst=cc2_src[0:T, own[0] : own[1]], srcs=srcs, reduce_op=nl.add)
        return
    nisa.dma_copy(dst=part[pid, 0:T, oth[0] : oth[1]], src=acc[0:T, oth[0] : oth[1]])
    nisa.core_barrier(data=part, cores=(0, 1))
    bv = nl.ndarray((PT, dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bv[0:T, :], src=part[1 - pid, 0:T, own[0] : own[1]])
    a_ = acc[0:T, own[0] : own[1]]
    p0, p1 = (a_, bv[0:T, :]) if pid == 0 else (bv[0:T, :], a_)
    sv = nl.ndarray((PT, dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sv[0:T, :], data1=p0, data2=p1, op=nl.add)
    nisa.dma_copy(dst=cc2_src[0:T, own[0] : own[1]], src=sv[0:T, :])


# --------------------------------------------------------------------------- column split

#: The down projections split by output columns between the cores instead of by slot: each core
#: runs its slots' gate|up GEMMs and SwiGLU, the cores swap the staged activations (`nisa.sendrecv`,
#: the T live columns), and each core runs every slot's down GEMM (the shared expert's two blocks,
#: the 2 * SLOTS static slots) for its half of the output columns, all of them accumulated in four
#: PSUM banks: no SBUF accumulator copies / adds (8 DVE instructions on T partitions a slot pair) and
#: no combine (each core stores its half of `cc2_src`). Same products and fp32 terms; only their
#: summation order changes.
MOE_CS = True


def _swap(pid, yq, ys, T, pipe):
    """The other core's staged activation columns `0 .. T` (`[PT, T]` tiles), ours sent: the
    live columns of `stage_rows`' `[PT, 1, nkc, PT]` tiles (chunk 0)."""
    dq = nl.ndarray((PT, T), dtype=FP8_X4, buffer=nl.sbuf)
    ds = nl.ndarray((PT, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.sendrecv(src=yq[0:PT, 0, 0, 0:T], dst=dq, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=pipe)
    nisa.sendrecv(src=ys[0:PT, 0, 0, 0:T], dst=ds, send_to_rank=1 - pid, recv_from_rank=1 - pid,
                  pipe_id=pipe + 1)
    return dq, ds


class _HalfDown:
    """A slot's down weights / scales for this core's output columns `[PT, dim / 2]`."""

    def __init__(self, dh):
        self.w2t = nl.ndarray((PT, dh), dtype=FP4_X4, buffer=nl.sbuf)
        self.w2s = nl.ndarray((PT, dh), dtype=nl.uint8, buffer=nl.sbuf)


def _load_down_half(e, nkc, dim, c0, w2v, s13, bufs, sidx, wx=None):
    W = 2 * HID
    kp = HID // 4
    nks = nkc * W + dim
    dh = dim // 2
    if wx is not None:
        # (`SLOT_SW_FROM`: one SWDGE row gather, skipped for the padding slots)
        w2r = w2v.reshape((w2v.shape[0] * kp, dim))
        nisa.dma_copy(dst=bufs.w2t[0:kp, :], src=w2r.ap(pattern=[[dim, kp], [1, dh]], offset=c0,
                                                        vector_offset=wx[0:kp, :], indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
    else:
        nisa.dma_copy(dst=bufs.w2t[0:kp, :], src=w2v.ap(pattern=[[dim, kp], [1, dh]], offset=c0, scalar_offset=e,
                                                        indirect_dim=0), dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip, **_wprio())
    s13r = s13.reshape((s13.shape[0] * 16, nks))
    nisa.dma_copy(dst=bufs.w2s[0:PT, :],
                  src=s13r.ap(pattern=[[nks, PT], [1, dh]], offset=nkc * W + c0, vector_offset=sidx, indirect_dim=0),
                  dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)


def _down_cs(accs, yq, ys, staged, kp, w2t, w2s, T, first):
    """`accs[j][0:T] (+)= y @ w2[:, j*512 ..]` for this core's column tiles; y staged
    (`stage_rows`' tiles, `staged`) or swapped (`[PT, T]`)."""
    sq = yq[0:kp, 0, 0, 0:T] if staged else yq[0:kp, 0:T]
    ss = ys[0:kp, 0, 0, 0:T] if staged else ys[0:kp, 0:T]
    for j, a in enumerate(accs):
        nisa.nc_matmul_mx(dst=a[0:T, :], stationary=sq, moving=w2t[0:kp, j * NT : (j + 1) * NT],
                          stationary_scale=ss, moving_scale=w2s[0:kp, j * NT : (j + 1) * NT],
                          accumulate=not first)


def shared_cs(pid, T, xq, xs, nkc, sh13, shs13, sh2, shs2, dim, limit):
    """The shared expert for `MOE_CS`: this core's block's gate|up GEMM and SwiGLU, staged and
    compacted, and both blocks' down weights for this core's columns: `(cq, cs, w2s)` with
    `w2s` = [(w2t, w2s) of block 0, of block 1]."""
    W = 2 * SH
    dh = dim // 2
    c0 = pid * dh
    wt = nl.ndarray((PT, nkc, W), dtype=FP8_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, nkc, W), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(ws, 0, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=wt, src=sh13.view(FP8_X4).ap(pattern=[[nkc * W, PT], [W, nkc], [1, W]],
                                                    offset=pid * PT * nkc * W),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    for q in range(4):
        nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, :, :],
                      src=shs13.ap(pattern=[[nkc * W, 4], [W, nkc], [1, W]], offset=(pid * 16 + 4 * q) * nkc * W),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    downs = []
    for b in range(2):
        w2t = nl.ndarray((PT, dh), dtype=FP8_X4, buffer=nl.sbuf)
        nisa.memset(w2t[0:SH_KP, :], 0, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=w2t[0 : SH // 4, :], src=sh2.view(FP8_X4)[b, :, c0 : c0 + dh],
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        w2s = nl.ndarray((PT, dh), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(w2s[0:SH_KP, :], 0, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=w2s[0:1, :], src=shs2[b, :, c0 : c0 + dh], dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.scalar)
        downs.append((w2t, w2s))
    a = nl.ndarray((PT, W), dtype=nl.float32, buffer=nl.psum)
    for kc in range(nkc):
        nisa.nc_matmul_mx(dst=a, stationary=xq[0:PT, 0, kc, :], moving=wt[0:PT, kc, :],
                          stationary_scale=xs[0:PT, 0, kc, :], moving_scale=ws[0:PT, kc, :],
                          accumulate=(kc > 0))
    g16 = nl.ndarray((PT, SH), dtype=nl.bfloat16, buffer=nl.sbuf)
    u16 = nl.ndarray((PT, SH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g16, src=a[:, 0:SH])
    nisa.tensor_copy(dst=u16, src=a[:, SH:W])
    y16 = nl.ndarray((PT, 4 * SH_KP), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(y16, 0.0)
    _swiglu(g16, u16, None, y16, SH, limit)
    yq, ys = alloc_x(4 * SH_KP)
    stage_rows(y16, 4 * SH_KP, yq, ys, nr=T)
    return yq, ys, downs


def routed_cs(pid, T, xq, xs, nkc, w13, s13, w2, wexp, alist, nact, ng, dim, limit, sh, nz, slot,
              cc2_src, t_out=None):
    """`MOE_CS`'s routed experts + the shared expert's down GEMMs, stored to this core's half of
    `cc2_src[0:T]`. Pair p = slots (2p, 2p + 1): core c the gate|up of slot 2p + c, both cores
    the down GEMMs of both slots (their columns)."""
    w13v = w13.view(FP4_X4)
    w2v = w2.view(FP4_X4)
    S = SLOTS
    dh = dim // 2
    c0 = pid * dh
    st = slot
    slot, sidx = st[0], st[1]
    # (`SLOT_SW_FROM`: the SWDGE slots' weight rows)
    wr13, wr2 = (st[2], st[3]) if len(st) > 2 else (None, None)
    # the slot ids of both cores (slot 2k + c of core c)
    ids = [nisa.scalar_load(nz[0:1, j : j + 1]) for j in range(2 * S)]

    def sx(j):
        return sidx[0:PT, j : j + 1]

    def wcol(j):
        return slot[:, j : j + 1]

    def wx(rows, j, p):
        # pair p's SWDGE weight rows of slot j, else None (the hardware DGE)
        if rows is None or SLOT_SW_FROM is None or p < SLOT_SW_FROM:
            return None
        return rows[0:PT, j : j + 1]

    nt = dh // NT
    accs = [nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum) for _ in range(nt)]
    ups = [_UpBufs(nkc) for _ in range(2)]
    downs = [[_HalfDown(dh) for _ in range(2)] for _ in range(2)]
    y16s = []
    for _ in range(2):
        y = nl.ndarray((PT, 4 * HID_KP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(y, 0.0)
        y16s.append(y)
    # the shared expert: this core's block, then the other core's (swapped with pair 0's slots)
    cq, cs, shd = sh
    # loads of pairs 0 and 1 (own gate|up of slots pid, 2 + pid; both cores' down halves)
    for p in range(min(S, 2)):
        _load_up(ids[2 * p + pid], nkc, dim, w13v, s13, ups[p % 2], sx(2 * p + pid), split=SLOT0_SPLIT,
                 wx=wx(wr13, 2 * p + pid, p))
        for c in range(2):
            _load_down_half(ids[2 * p + c], nkc, dim, c0, w2v, s13, downs[p % 2][c], sx(2 * p + c),
                            wx=wx(wr2, 2 * p + c, p))
    first = True
    for p in range(S):
        b = p % 2
        j = 2 * p + pid
        yq, ys = _up(T, xq, xs, nkc, limit, ups[b], wcol(j), y16s[b])
        if p + 2 < S:
            _load_up(ids[2 * (p + 2) + pid], nkc, dim, w13v, s13, ups[b], sx(2 * (p + 2) + pid),
                     wx=wx(wr13, 2 * (p + 2) + pid, p + 2))
        if p == 0:
            # (the shared expert's columns swapped along with pair 0's)
            sq, ss_ = _swap(pid, cq, cs, T, 8)
            terms = [(cq, cs, True, shd[pid], SH_KP), (sq, ss_, False, shd[1 - pid], SH_KP)]
        else:
            terms = []
        oq, os_ = _swap(pid, yq, ys, T, 2 * (p % 4))
        mine = (yq, ys, True, (downs[b][pid].w2t, downs[b][pid].w2s), HID_KP)
        other = (oq, os_, False, (downs[b][1 - pid].w2t, downs[b][1 - pid].w2s), HID_KP)
        # (slot order 2p, 2p + 1 on both cores)
        terms += [mine, other] if pid == 0 else [other, mine]
        for q_, s_, st, (w2t, w2s), kp in terms:
            _down_cs(accs, q_, s_, st, kp, w2t, w2s, T, first)
            first = False
        if p + 2 < S:
            for c in range(2):
                _load_down_half(ids[2 * (p + 2) + c], nkc, dim, c0, w2v, s13, downs[b][c], sx(2 * (p + 2) + c),
                                wx=wx(wr2, 2 * (p + 2) + c, p + 2))

    # this core's columns of the rank's partial, stored ahead of the dynamic loop (rarely entered:
    # its terms are added to the stored columns by the DMA engine)
    o = nl.ndarray((PT, dh), dtype=nl.float32, buffer=nl.sbuf)
    for j, a in enumerate(accs):
        nisa.tensor_copy(dst=o[0:T, j * NT : (j + 1) * NT], src=a[0:T, :],
                         engine=nisa.engine.scalar if j % 2 else nisa.engine.vector)
    # (`t_out`: the rows stored -- T is rounded up to even for the MX matmuls' stationaries)
    to = T if t_out is None else t_out
    dst = cc2_src[0:to, c0 : c0 + dh]
    nisa.dma_copy(dst=dst, src=o[0:to, :], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    def body(i):
        e01 = _slot_ids(alist, ng, i, (0, 1))
        e = e01[pid]
        _load_up(e, nkc, dim, w13v, s13, ups[0])
        for c, ec in enumerate(e01):
            _load_down_half_dyn(ec, nkc, dim, c0, w2v, s13, downs[0][c])
        wc = _wcol(e, wexp, ng)
        yq, ys = _up(T, xq, xs, nkc, limit, ups[0], wc, y16s[0])
        oq, os_ = _swap(pid, yq, ys, T, 10)
        pair = [(yq, ys, True), (oq, os_, False)] if pid == 0 else [(oq, os_, False), (yq, ys, True)]
        tmp = [nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum) for _ in range(nt)]
        for c, (q_, s_, st) in enumerate(pair):
            _down_cs(tmp, q_, s_, st, HID_KP, downs[0][c].w2t, downs[0][c].w2s, T, c == 0)
        ot = nl.ndarray((PT, dh), dtype=nl.float32, buffer=nl.sbuf)
        for j, a in enumerate(tmp):
            nisa.tensor_copy(dst=ot[0:T, j * NT : (j + 1) * NT], src=a[0:T, :])
        nisa.dma_compute(dst=dst, srcs=[dst, ot[0:to, :]], reduce_op=nl.add)

    if OVERFLOW:
        n2 = ((nact + 1) // 2) * 2
        nl.fori_loop(2 * S, n2, body, step=2)


def _load_down_half_dyn(e, nkc, dim, c0, w2v, s13, bufs):
    """`_load_down_half` for a dynamic slot (no precomputed scale-row column): the scale rows by
    quadrant DMAs (`_load_down`'s)."""
    W = 2 * HID
    kp = HID // 4
    nks = nkc * W + dim
    dh = dim // 2
    nisa.dma_copy(dst=bufs.w2t[0:kp, :], src=w2v.ap(pattern=[[dim, kp], [1, dh]], offset=c0, scalar_offset=e,
                                                    indirect_dim=0), dge_mode=nisa.dge_mode.hwdge,
                  engine=nisa.engine.sync, oob_mode=nisa.oob_mode.skip)
    for q, n in ((0, 4), (1, 4), (2, 1)):
        nisa.dma_copy(dst=bufs.w2s[32 * q : 32 * q + n, :],
                      src=s13.ap(pattern=[[nks, n], [1, dh]], offset=4 * q * nks + nkc * W + c0, scalar_offset=e,
                                 indirect_dim=0), oob_mode=nisa.oob_mode.skip, **_scale_q())


def moe(pid, T, xn2, gate_w_t, gate_b, expert_lo, w13, s13, w2, s2, sh13, shs13, sh2, shs2,
        logits, part, cc2_src, dim, limit, route_scale, topk, xstage=None, mid=None):
    """The block's MoE for rows `0 .. T` of `xn2` (bf16 HBM, rows past T zero) into
    `cc2_src[0:T]` (this rank's fp32 partial; the caller barriers and all-reduces it).
    `xstage`: `xn2`'s rows already staged (`X.stage_x`: slot 0 is used). `mid(pid)`: called
    after the routing, before the routed experts (the caller's hc work for the core)."""
    assert T <= PT
    n_experts = gate_w_t.shape[1]
    ng = w13.shape[0] - 1
    nkc = dim // KC
    # the MX matmuls' stationaries need an even column count: an odd T runs the expert GEMMs on
    # T + 1 rows (xn2's next row is zero and its routing weights are zero) and stores T
    Tm = T + (T % 2)
    if xstage is not None:
        xq, xs = xstage[0][:, 0:1, :, :], xstage[1][:, 0:1, :, :]
    else:
        xq, xs = stage_x(xn2, Tm, dim)
    cs_ok = MOE_CS and SLOT_DMA3 and SLOT_COLS and SLOT_IDS_SBUF and dim % (2 * NT) == 0
    assert cs_ok or Tm == T, "an odd row count needs MOE_CS"
    if cs_ok:
        sh = shared_cs(pid, Tm, xq, xs, nkc, sh13, shs13, sh2, shs2, dim, limit)
    else:
        acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        sh_term = shared_up(pid, T, xq, xs, nkc, sh13, shs13, sh2, shs2, dim, limit)
    lg_sb = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf) if ROUTER_SR else None
    router_logits(pid, T, xn2, gate_w_t, logits, dim, n_experts, lg_sb=lg_sb)
    if lg_sb is None:
        nisa.core_barrier(data=logits, cores=(0, 1))
    wexp, alist, nact, nz, slot = route_group(logits, gate_b, expert_lo, T, n_experts, topk, ng, route_scale,
                                              lg_sb=lg_sb)
    if mid is not None:
        mid(pid)
    if cs_ok:
        routed_cs(pid, Tm, xq, xs, nkc, w13, s13, w2, wexp, alist, nact, ng, dim, limit, sh, nz, slot, cc2_src,
                  t_out=T)
        return
    routed(pid, T, xq, xs, nkc, w13, s13, w2, wexp, alist, nact, ng, dim, limit, acc, sh_term,
           nz=nz if SLOT_IDS_SBUF else None, slot=slot)
    combine(pid, T, acc, part, cc2_src, dim)


# --------------------------------------------------------------------------- hc mix GEMM

#: hc_in's mix GEMM (`x @ fn`, K = hc * dim = 20480, N = mix = 24) for few rows as a
#: block-diagonal GEMM: J = 128 // rows K-tiles' transposed rows side by side on the stationary
#: and their fn tiles side by side on the moving operand (J * 24 <= 512 columns), so one
#: matmul covers J K-tiles (8 instead of 160 at T = 6); the diagonal blocks are the products,
#: masked, summed over the J blocks and the J row groups. The sum of squares from the same
#: transposed rows (one square, one reduction, one ones-matmul) instead of 20 ACT passes over
#: the row layout.
MIX_BD = True
MIX_BD_MAX_ROWS = 16
#: (members) the stacked mix's rows of a group in one DMA (`hc_in_tile`, r1 families). (OFF: not
#: expressible -- a DMA maps only its source's outer dimension onto the partitions)
HC_STACK_ONE_DMA = False


def mix_bd_ok(nr, mix):
    return MIX_BD and nr <= MIX_BD_MAX_ROWS and PT // nr >= 2 and mix <= 256


def _onehot(dst, k, pattern):
    """`dst[0:k, :] = (iota(pattern) - partition == 0)` as fp32."""
    n = dst.shape[1]
    v = nl.ndarray((PT, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=v[0:k, :], pattern=pattern, offset=0, channel_multiplier=-1)
    vf = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vf[0:k, :], src=v[0:k, :])
    nisa.tensor_scalar(dst=dst[0:k, :], data=vf[0:k, :], op0=nl.equal, operand0=0.0)


def bd_consts(nr, mix):
    """The block-diagonal mask `[J*nr, J*mix]`, the row selector `[J*nr, nr]` and a ones column."""
    J = min(PT // nr, NT // mix)
    R = nl.ndarray((PT, J * nr), dtype=nl.float32, buffer=nl.sbuf)
    _onehot(R, J, [[1, J], [0, nr]])
    C = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.sbuf)
    _onehot(C, J, [[1, J], [0, mix]])
    mp = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mp[0 : J * nr, :], stationary=R[0:J, :], moving=C[0:J, :])
    mask = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mask[0 : J * nr, :], src=mp[0 : J * nr, :])
    A = nl.ndarray((PT, J * nr), dtype=nl.float32, buffer=nl.sbuf)
    _onehot(A, nr, [[0, J], [1, nr]])
    I = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    _onehot(I, nr, [[1, nr]])
    sp = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=sp[0 : J * nr, :], stationary=A[0:nr, :], moving=I[0:nr, :])
    sel = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sel[0 : J * nr, :], src=sp[0 : J * nr, :])
    ones = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    return J, mask, sel, ones


def mix_bd(t32c, fnT, nr, nk, mix, consts, ss_out=None):
    """`x @ fn` (an `[nr, mix]` fp32 PSUM tile) from `t32c` (`[PT, nk * nr]` fp32: x^T, K-tile
    kb's rows at columns kb*nr ..) and `fnT` (`[PT, nk, mix]` fp32); with `ss_out` (`[PT, 1]`
    fp32 SBUF) also `ss_out[0:nr] = sum_k x^2`."""
    J, mask, sel, ones = consts
    P = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.psum)
    for g in range(cdiv(nk, J)):
        j0 = g * J
        jn = min(J, nk - j0)
        nisa.nc_matmul(dst=P[0 : jn * nr, 0 : jn * mix], stationary=t32c[0:PT, j0 * nr : (j0 + jn) * nr],
                       moving=fnT.ap(pattern=[[nk * mix, PT], [1, jn * mix]], offset=j0 * mix),
                       accumulate=(g > 0), is_stationary_onezero=True)
    if ss_out is not None:
        sq = nl.ndarray((PT, nk * nr), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=t32c[0:PT, 0 : nk * nr], data2=t32c[0:PT, 0 : nk * nr], op=nl.multiply)
        sqr = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sqr, op=nl.add, data=sq.ap(pattern=[[nk * nr, PT], [1, nr], [nr, nk]]), axis=(2,))
        ssp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ssp[0:nr, :], stationary=sqr[0:PT, 0:nr], moving=ones[0:PT, 0:1])
        nisa.tensor_copy(dst=ss_out[0:nr, :], src=ssp[0:nr, :])
    Pm = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Pm[0 : J * nr, :], data1=P[0 : J * nr, :], data2=mask[0 : J * nr, :], op=nl.multiply)
    Z = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=Z[0 : J * nr, :], op=nl.add,
                       data=Pm.ap(pattern=[[J * mix, J * nr], [1, mix], [mix, J]]), axis=(2,))
    out = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out[0:nr, :], stationary=sel[0 : J * nr, 0:nr], moving=Z[0 : J * nr, :])
    return out


def hc_mix(xg, fnT, nr, hc, dim, mix, cache, ss_out):
    """hc_in's mix GEMM and sum of squares for `nr` (<= 16) rows `xg` (hc bf16 SBUF tiles
    `[PT, dim]`): DMA-transposed, cast to fp32 compactly, then `mix_bd`. `cache`: a dict the
    block-diagonal constants are kept in (the caller's per-call constants)."""
    kpg = dim // PT
    nk = hc * kpg
    t32c = nl.ndarray((PT, nk * nr), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(hc):
        t16 = nl.ndarray((PT, kpg, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_transpose(dst=t16, src=xg[g].ap(pattern=[[dim, 16], [PT, kpg], [1, PT]], offset=0), axes=(2, 1, 0))
        nisa.tensor_copy(dst=t32c.ap(pattern=[[nk * nr, PT], [nr, kpg], [1, nr]], offset=g * kpg * nr),
                         src=t16[0:PT, :, 0:nr])
    key = ("bd", nr, mix)
    if key not in CACHE:
        CACHE[key] = bd_consts(nr, mix)
    return mix_bd(t32c, fnT, nr, nk, mix, CACHE[key], ss_out=ss_out)


#: (members, r2 families) hc_in's mix for few rows read straight from HBM in the stacked layout
#: (partition j * nr + r = row r's 128-column chunk k0 + j: one DMA a group of J chunks) and
#: transposed by the PE (one transpose a group, the bf16 -> fp32 copy straight into `mix_bd`'s
#: K-tile layout) instead of the rows' load and hc DMA transposes (~11 us on core 1)
HC_MIX_STACK = True


def hc_mix_hbm(x2, r0, nr, fnT, hc, dim, mix, ss_out, after=None):
    """`hc_mix` for rows `r0 .. r0 + nr` of `x2` (`[T, hc * dim]` bf16 HBM)."""
    width = hc * dim
    nk = width // PT
    key = ("bd", nr, mix)
    if key not in CACHE:
        CACHE[key] = bd_consts(nr, mix)
    J = CACHE[key][0]
    t32c = nl.ndarray((PT, nk * nr), dtype=nl.float32, buffer=nl.sbuf)
    x2c = x2.reshape((x2.shape[0] * nk, PT))
    for g in range(cdiv(nk, J)):
        k0 = g * J
        js = min(J, nk - k0)
        n = js * nr
        xs = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for r in range(nr):
            # (partition r * js + j = row r's chunk k0 + j: one DMA a row -- a DMA maps only its
            # source's outer dimension onto the partitions)
            q = nisa.engine.sync if (g * nr + r) % 2 == 0 else nisa.engine.scalar
            k = (r0 + r) * nk + k0
            _sched_after(nisa.dma_copy(dst=xs[r * js : (r + 1) * js, 0:PT], src=x2c[k : k + js, 0:PT],
                                       dge_mode=nisa.dge_mode.hwdge, engine=q), after)
        tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tps[0:PT, 0:n], data=xs[0:n, 0:PT])
        # t32c[:, (k0 + j) * nr + r] = tps[:, r * js + j]
        nisa.tensor_copy(dst=t32c.ap(pattern=[[nk * nr, PT], [nr, js], [1, nr]], offset=k0 * nr),
                         src=tps.ap(pattern=[[PT, PT], [1, js], [js, nr]]))
    return mix_bd(t32c, fnT, nr, nk, mix, CACHE[key], ss_out=ss_out)


class BDMix:
    """hc_in's mix GEMM accumulated block-diagonally from stacked transposed rows (`add`: J
    K-tiles' x^T side by side, one matmul against their fn tiles side by side), the sum of
    squares from the same tiles, and the reduction to `[nr, mix]` (`finish`); see `MIX_BD`."""

    def __init__(self, nr, mix, cache, want_ss=True):
        key = ("bd", nr, mix)
        if key not in CACHE:
            CACHE[key] = bd_consts(nr, mix)
        self.J, self.mask, self.sel, self.ones = CACHE[key]
        self.nr, self.mix = nr, mix
        self.P = nl.ndarray((PT, self.J * mix), dtype=nl.float32, buffer=nl.psum)
        self.first = True
        self.sq = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf) if want_ss else None
        self.sq_first = True

    def add(self, stat, jn, kb0, fnT):
        """`stat`: `[PT, jn * nr]` fp32, K-tiles kb0 .. kb0 + jn of x^T (tile j's rows at columns
        j * nr ..); `fnT`: `[PT, nk, mix]` fp32."""
        J, nr, mix = self.J, self.nr, self.mix
        nk = fnT.shape[1]
        assert jn <= J and (not self.first or jn == J)
        nisa.nc_matmul(dst=self.P[0 : jn * nr, 0 : jn * mix], stationary=stat,
                       moving=fnT.ap(pattern=[[nk * mix, PT], [1, jn * mix]], offset=kb0 * mix),
                       accumulate=not self.first, is_stationary_onezero=True)
        self.first = False
        if self.sq is not None:
            q = nl.ndarray((PT, J * nr), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q[:, 0 : jn * nr], data1=stat, data2=stat, op=nl.multiply)
            r = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=r, op=nl.add, data=q.ap(pattern=[[J * nr, PT], [1, nr], [nr, jn]]), axis=(2,))
            if self.sq_first:
                nisa.tensor_copy(dst=self.sq, src=r)
            else:
                nisa.tensor_tensor(dst=self.sq, data1=self.sq, data2=r, op=nl.add)
            self.sq_first = False

    def finish(self, ss_out=None):
        """The `[nr, mix]` fp32 PSUM product; `ss_out[0:nr, 0:1]` the sum of squares."""
        J, nr, mix = self.J, self.nr, self.mix
        Pm = nl.ndarray((PT, J * mix), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=Pm[0 : J * nr, :], data1=self.P[0 : J * nr, :], data2=self.mask[0 : J * nr, :],
                           op=nl.multiply)
        Z = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=Z[0 : J * nr, :], op=nl.add,
                           data=Pm.ap(pattern=[[J * mix, J * nr], [1, mix], [mix, J]]), axis=(2,))
        out = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=out[0:nr, :], stationary=self.sel[0 : J * nr, 0:nr], moving=Z[0 : J * nr, :])
        if ss_out is not None and self.sq is not None:
            ssp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ssp[0:nr, :], stationary=self.sq[0:PT, 0:nr], moving=self.ones[0:PT, 0:1])
            nisa.tensor_copy(dst=ss_out[0:nr, :], src=ssp[0:nr, :])
        return out


def bd_stack_ok(nr, mix, J):
    """Whether a stacked-transpose hc_in (J K-tiles of nr rows on partitions) takes `BDMix`."""
    return mix_bd_ok(nr, mix) and J == min(PT // nr, NT // mix) and PT // J == nr


# --------------------------------------------------------------------------- spread RMS norm

#: hc_in's RMS norm of few mixed rows in the spread layout (partition t*S + s: row t, columns
#: s*W ..: ~120 partitions instead of the rows' 6): the sum of squares a partition, the rows'
#: sums by a selector matmul, rstd broadcast back by the transposed selector, the scaling and
#: the norm weight elementwise, and the normed rows stored from that layout -- instead of the
#: mixed rows stored, reloaded as rows and normed on 6 partitions (~16 us).
NORM_SPREAD = True


def _range_onehot(dst, k, n, pattern, cm, lo, hi):
    """`dst[0:k, 0:n] = (lo <= iota(pattern, channel_multiplier=cm) < hi)` as fp32."""
    v = nl.ndarray((PT, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=v[0:k, :], pattern=pattern, offset=0, channel_multiplier=cm)
    vf = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vf[0:k, :], src=v[0:k, :])
    a = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a[0:k, :], data=vf[0:k, :], op0=nl.greater_equal, operand0=float(lo))
    nisa.tensor_scalar(dst=vf[0:k, :], data=vf[0:k, :], op0=nl.less, operand0=float(hi))
    nisa.tensor_tensor(dst=dst[0:k, 0:n], data1=a[0:k, :], data2=vf[0:k, :], op=nl.multiply)


def spread_consts(nr, S, mode, cache):
    """`(sel [P, nr], selT [nr, P])` between the spread layout's partitions and the rows:
    mode "t" partition p = t*S + s, mode "s" p = s*nr + t (row t)."""
    key = ("sp", nr, S, mode)
    if key in CACHE:
        return CACHE[key]
    P = nr * S
    sel = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    selT = nl.ndarray((PT, P), dtype=nl.float32, buffer=nl.sbuf)
    if mode == "t":
        # v[p, r] = p - S*r in [0, S); v[r, p] the same
        _range_onehot(sel, P, nr, [[-S, nr]], 1, 0, S)
        _range_onehot(selT, nr, P, [[1, P]], -S, 0, S)
    else:
        # selT[r, (s, t)] = (t == r); sel = selT^T (one matmul with the identity)
        _onehot(selT, nr, [[0, S], [1, nr]])
        I = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
        _onehot(I, nr, [[1, nr]])
        sp = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sp[0:P, :], stationary=selT[0:nr, 0:P], moving=I[0:nr, :])
        nisa.tensor_copy(dst=sel[0:P, :], src=sp[0:P, :])
    c = (sel, selT)
    CACHE[key] = c
    return c


def spread_rstd(ys, nr, S, mode, W, eps, cache):
    """`1 / sqrt(mean(row^2) + eps)` (one Newton step, `rsqrt_newton`) of the rows of the bf16
    spread tile `ys` (`[PT, W]`, `nr * S` partitions), broadcast back to the spread layout
    (`[PT, 1]` fp32 SBUF)."""
    P = nr * S
    dim = S * W
    sel, selT = spread_consts(nr, S, mode, cache)
    part = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    junk = nl.ndarray((PT, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=junk[0:P, :], op=nl.square, data=ys[0:P, 0:W], reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=part[0:P, 0:1])
    ssp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp[0:nr, :], stationary=sel[0:P, 0:nr], moving=part[0:P, 0:1])
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ss[0:nr, :], src=ssp[0:nr, :])
    m = nl.add(nl.multiply(ss[0:nr, 0:1], 1.0 / dim), eps)
    y = nl.rsqrt(m)
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd[0:nr, :] = nl.multiply(y, corr)
    rbp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rbp[0:P, :], stationary=selT[0:nr, 0:P], moving=rstd[0:nr, 0:1])
    rb = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rb[0:P, :], src=rbp[0:P, :])
    return rb


def norm_spread_t(ys, nr, S, W, normed, norm_w, eps, cache):
    """RMS norm of the mixed bf16 rows `ys` (SBUF, spread layout "t": partition t*S + s = row t,
    columns s*W .. s*W + W) times `norm_w` (`[1, S*W]` HBM) into `normed[0:nr]` (bf16 HBM): the
    arithmetic of `hc_norm_tail` (fp32 sum of squares, rsqrt + Newton step, scaling, weight)."""
    P = nr * S
    wkey = ("wsp", nr, S, id(norm_w))
    wsp = cache.get(wkey) if cache is not None else None
    if wsp is None:
        wsp = nl.ndarray((PT, W), dtype=nl.float32, buffer=nl.sbuf)
        for t in range(nr):
            nisa.dma_copy(dst=wsp[t * S : (t + 1) * S, :], src=norm_w.ap(pattern=[[W, S], [1, W]], offset=0))
        if cache is not None:
            cache[wkey] = wsp
    rb = spread_rstd(ys, nr, S, "t", W, eps, cache)
    yv = nl.ndarray((PT, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=yv[0:P, :], op=nl.copy, data=ys[0:P, 0:W], scale=rb[0:P, 0:1])
    o16 = nl.ndarray((PT, W), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o16[0:P, :], data1=yv[0:P, :], data2=wsp[0:P, :], op=nl.multiply)
    nisa.dma_copy(dst=normed.ap(pattern=[[W, P], [1, W]], offset=0), src=o16[0:P, 0:W],
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


# --------------------------------------------------------------------------- window rows early

#: Decode: a block's slot table gets its window rows of the positions before the step from
#: the window cache at the block's start (they do not depend on this step), and the step's own
#: rows straight from its kv rows -- instead of the whole window gathered after this step's
#: rows were scattered into the cache (scatter -> barrier -> gather on the attention's path).
WIN_EARLY = True


def put_rows(src_hbm, r0, nr, dst_hbm, d0, n, pair):
    """`dst[d0 + r] = src[r0 + r]` for `r < nr` (bf16, `n` columns); `pair`: into the
    pair-interleaved column order (`kv_pair_layout`), as `gather_rows` stores its rows."""
    t = nl.ndarray((PT, n), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:n])
    if pair:
        npair = n // (2 * PT)
        tp = nl.ndarray((PT, npair, PT, 2), dtype=src_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tp[0:nr, :, :, :], src=t.ap(pattern=[[n, nr], [2 * PT, npair], [1, PT], [PT, 2]]))
        t = tp.reshape((PT, n))
    nisa.dma_copy(dst=dst_hbm[d0 : d0 + nr, 0:n], src=t[0:nr, :])


# --------------------------------------------------------------------------- sinkhorn

#: hc_in's comb sinkhorn (softmax over each row's hc columns + eps, then 20 alternating row /
#: column normalizations, each dividing by its sum + eps) with every normalization on the
#: vector engine -- a reduce, an eps add and a divide -- instead of DVE reduces / multiplies
#: interleaved with ACT reciprocals (each DVE <-> ACT hop of the 200-instruction chain costs
#: a semaphore round trip: ~25-45 us a call on few rows)
SINKHORN_DVE = True
#: (members) r2 / r0 blocks run hc_in(attn)'s core-1 sinkhorn after the attention (see the
#: members' `SINKHORN_A_LATE` sites) instead of alongside core 1's q/kv work
SINKHORN_A_LATE = True
HC_EPS = 1e-06
SINKHORN_ITERS = 20


#: (members, r0 / r2) core 1's hc_in(attn) mix half run after the q / kv projections' barriers
HC_A_MIX_LATE = True


#: (members, r1a) core 1's hc_in(ffn) mix half run in the MoE's mid (after the routing) instead of
#: ahead of the MoE (its outputs are first read at the world all-reduce)
R1_MIX_MID = True


#: (members, index sources) the index-score all-reduce over the T real rows only
IDX_SC_T = True


#: (members) DMA QoS priority of the blocks' attention / world all-reduces (0 = the highest: they
#: are on the critical path, the next block's weight prefetches in their shadow are not; None:
#: the default)
AR_PRIORITY = None


#: (members, r1 families) hc_in's sinkhorns at the all-reduces schedule-pinned after an anchor
#: instruction (`pin_next_sinkhorn`) instead of gated by a DMA of one element of the all-reduce's
#: source into the logits' pad (the DMA waited ~17 us in the queues; the 20-step chain then ran
#: ~20 us past the world all-reduce into the next block): the FFN one after the world
#: all-reduce's source barrier, the attention one after the attention's output barrier
SINK_F_PIN = True
SINK_A_PIN = True
#: (r1 members) the attention one issued right after the attention's output barrier
SINK_A_AT_O = True


def pin_next_sinkhorn(ins):
    """The next `sinkhorn_dve` call without its own `after` starts after `ins`."""
    if ins is not None and hasattr(ins, "depends_on"):
        CACHE["sink_after"] = ins


def _sched_after(ins, after):
    if after is not None and ins is not None and hasattr(ins, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)


def sinkhorn_dve(t2all, nt, hc, after=None):
    """`mk_common.sinkhorn_batched` (`t2all`: `(PT, nt, hc*hc)` fp32; the result likewise)."""
    if after is None:
        after = CACHE.pop("sink_after", None)
    n = nt * hc
    hh = hc * hc
    f32 = nl.float32
    x3 = t2all.reshape((PT, n, hc))
    a = nl.ndarray((PT, nt, hh), dtype=f32, buffer=nl.sbuf)
    b = nl.ndarray((PT, nt, hh), dtype=f32, buffer=nl.sbuf)
    mx = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    _sched_after(nisa.tensor_reduce(dst=mx, op=nl.max, data=x3, axis=(2,), keepdims=True), after)
    dif = nl.ndarray((PT, n, hc), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dif, data1=x3, data2=nl.broadcast_to(mx, (PT, n, hc)), op=nl.subtract)
    e = nl.ndarray((PT, n, hc), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=e, op=nl.exp, data=dif)
    se = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=se, op=nl.add, data=e, axis=(2,), keepdims=True)
    a3 = a.reshape((PT, n, hc))
    re = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.reciprocal(dst=re, data=se)
    if SINKHORN_PAD:
        return _sinkhorn_pad(e, re, nt, hc)
    nisa.tensor_tensor(dst=a3, data1=e, data2=nl.broadcast_to(re, (PT, n, hc)), op=nl.multiply)
    nisa.tensor_scalar(dst=a3, data=a3, op0=nl.add, operand0=HC_EPS)

    def rowdiv(src, dst):
        s3 = src.reshape((PT, n, hc))
        sm = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.add, data=s3, axis=(2,), keepdims=True)
        nisa.tensor_scalar(dst=sm, data=sm, op0=nl.add, operand0=HC_EPS)
        r = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=r, data=sm)
        nisa.tensor_tensor(dst=dst.reshape((PT, n, hc)), data1=s3, data2=nl.broadcast_to(r, (PT, n, hc)),
                           op=nl.multiply)

    def coldiv(src, dst):
        cs = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        # [p, t, j, i] = src[p, t, i*hc + j]: the column sums over the rows i
        nisa.tensor_reduce(dst=cs, op=nl.add, data=src.ap(pattern=[[nt * hh, PT], [hh, nt], [1, hc], [hc, hc]],
                                                          offset=0), axis=(3,))
        nisa.tensor_scalar(dst=cs, data=cs, op0=nl.add, operand0=HC_EPS)
        d = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=d, data=cs)
        nisa.tensor_tensor(dst=dst.reshape((PT, nt, hc, hc)), data1=src.reshape((PT, nt, hc, hc)),
                           data2=d.ap(pattern=[[nt * hc, PT], [hc, nt], [0, hc], [1, hc]], offset=0), op=nl.multiply)

    coldiv(a, b)
    for _ in range(SINKHORN_ITERS - 1):
        rowdiv(b, a)
        coldiv(a, b)
    return b


#: `sinkhorn_dve` with each matrix stored `(hc + 1) x (hc + 1)`: an eps column after every row
#: and an eps row after the last, so a row / column reduce over hc + 1 entries is `sum + eps`
#: (a reduce, a reciprocal and a multiply a normalization instead of four instructions)
SINKHORN_PAD = True


def _sinkhorn_pad(e, re, nt, hc):
    f32 = nl.float32
    H = hc + 1
    G = H * H
    a = nl.ndarray((PT, nt, H, H), dtype=f32, buffer=nl.sbuf)
    b = nl.ndarray((PT, nt, H, H), dtype=f32, buffer=nl.sbuf)
    nisa.memset(a, HC_EPS)
    nisa.memset(b, HC_EPS)

    def mat(x):
        # the hc x hc entries of every matrix: [p, t, i, j]
        return x.ap(pattern=[[nt * G, PT], [G, nt], [H, hc], [1, hc]], offset=0)

    # softmax rows + eps
    nisa.tensor_tensor(dst=mat(a), data1=e.reshape((PT, nt, hc, hc)),
                       data2=re.ap(pattern=[[nt * hc, PT], [hc, nt], [1, hc], [0, hc]], offset=0), op=nl.multiply)
    nisa.tensor_scalar(dst=mat(a), data=mat(a), op0=nl.add, operand0=HC_EPS)

    def rowdiv(src, dst):
        sm = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.add, data=src.ap(pattern=[[nt * G, PT], [G, nt], [H, hc], [1, H]], offset=0),
                           axis=(3,))
        r = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=r, data=sm)
        nisa.tensor_tensor(dst=mat(dst), data1=mat(src), data2=r.ap(pattern=[[nt * hc, PT], [hc, nt], [1, hc], [0, hc]],
                                                                     offset=0), op=nl.multiply)

    def coldiv(src, dst):
        cs = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        # [p, t, j, i]: column j's hc entries and the eps row
        nisa.tensor_reduce(dst=cs, op=nl.add, data=src.ap(pattern=[[nt * G, PT], [G, nt], [1, hc], [H, H]], offset=0),
                           axis=(3,))
        d = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=d, data=cs)
        nisa.tensor_tensor(dst=dst, data1=mat(src), data2=d.ap(pattern=[[nt * hc, PT], [hc, nt], [0, hc], [1, hc]],
                                                                offset=0), op=nl.multiply)

    out = nl.ndarray((PT, nt, hc * hc), dtype=f32, buffer=nl.sbuf)
    o4 = out.reshape((PT, nt, hc, hc))
    if SINKHORN_ITERS == 1:
        coldiv(a, o4)
        return out
    coldiv(a, mat(b))
    for it in range(SINKHORN_ITERS - 1):
        rowdiv(b, a)
        coldiv(a, o4 if it == SINKHORN_ITERS - 2 else mat(b))
    return out


#: (index sources, one row tile) the index weights' projection on core 1 and the index queries'
#: rope / fp4 round trip split between the cores by head halves (`tools/patch_idxq_split.py`)
IDXQ_SPLIT = True


# --------------------------------------------------------------------------- engram read-out

#: (members with an engram read-out) its fp8 `wkv` projection (`[6144, (hc + 1) * dim]`, ~157 MB a
#: rank, every rank streaming all of it) split over the 8 ranks of the attention group: 1024-column
#: blocks b dealt to the group's slot b % 8 (`mk_model.engram_tp_map`, 4 blocks a rank, zero
#: padded), each rank's `[T, 4096]` part all-gathered over the group; the gate reads the blocks
#: back through `EngramKV`
ENGRAM_TP = True
ENGRAM_GROUP = 8
ENGRAM_BLK = 1024


class EngramKV:
    """The all-gathered engram projection `[ENGRAM_GROUP * T, nb * ENGRAM_BLK]` (slot-major):
    global column block b is at slot b % G, local block b // G."""

    def __init__(self, gath, T, width):
        self.gath, self.T, self.width = gath, T, width

    def cols(self, r0, nr, c0, n):
        b = c0 // ENGRAM_BLK
        assert (c0 % ENGRAM_BLK) + n <= ENGRAM_BLK
        off = ((b % ENGRAM_GROUP) * self.T + r0) * self.width + (b // ENGRAM_GROUP) * ENGRAM_BLK + c0 % ENGRAM_BLK
        return self.gath.ap(pattern=[[self.width, nr], [1, n]], offset=off)


def engram_kv_cols(kv, r0, nr, c0, n):
    """Rows `r0 .. r0 + nr`, columns `c0 .. c0 + n` of the engram projection (`EngramKV` or HBM)."""
    if isinstance(kv, EngramKV):
        return kv.cols(r0, nr, c0, n)
    return kv[r0 : r0 + nr, c0 : c0 + n]


def engram_sharded(e_wkv4, e_full):
    """Whether the host passed this rank's blocks (`e_full`: the projection's full width)."""
    return ENGRAM_TP and e_wkv4.shape[1] < e_full


def engram_proj_tp(pid, emb_sum, Tp, T, e_in, e_wkv4, e_wkv_s, rg_attn, gemm_rows):
    """This rank's blocks of the projection (each core half of its columns), all-gathered over the
    attention group: an `EngramKV`."""
    width = e_wkv4.shape[1]
    part = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ekv_part")
    h = width // 2
    gemm_rows(emb_sum, Tp, e_in, e_wkv4, e_wkv_s, part, pid * h, (pid + 1) * h, out_rows=T, nv=T)
    nisa.core_barrier(data=part, cores=(0, 1))
    gath = nl.ndarray((ENGRAM_GROUP * T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ekv_gath")
    ncc.all_gather(srcs=[part], dsts=[gath], replica_group=rg_attn, collective_dim=0,
                   **({} if AR_PRIORITY is None else dict(priority=AR_PRIORITY)))
    return EngramKV(gath, T, width)
