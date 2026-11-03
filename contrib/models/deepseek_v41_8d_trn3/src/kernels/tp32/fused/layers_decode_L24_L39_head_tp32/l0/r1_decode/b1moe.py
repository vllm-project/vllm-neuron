# SPDX-License-Identifier: Apache-2.0
"""The MoE of a bs1 decode block on chip (round 5): `moe_tp`'s TP x EP layout (8 groups of 8 ranks, a
group's 48 experts split 8 ways by hidden columns), its routing (`moe_tp.route_group`), with

  * the input staged from the packed SBUF tile (no HBM round trip);
  * the router as 40 bf16 matmuls per core (its half of the experts) straight from the staging's
    transposes (the weight rows read in the transposes' order), the halves swapped SBUF to SBUF;
  * the gate|up GEMM of a core's 4 static slots as one matmul per contraction chunk and 512-column
    tile over the 4 slots' columns side by side (20 -> 5 matmuls a chunk);
  * each core every slot's down GEMM for its half of the output columns (the staged activations
    swapped), accumulated in PSUM (the shared expert's two blocks too), stored to `cc2_src`;
  * the slots past 8 (more than 8 active experts in the group) in `moe_tp`'s dynamic loop.

Same products, SwiGLU and quantization points as `moe_tp` (the deployment's values); the fp32 sums of
the down projections run in a different order.
"""

import nki.isa as nisa
import nki.language as nl

from .. import moe_tp as M
from .mk_common import PT

KC = 512
NT = 512
HID = 288
W13 = 2 * HID
SH = 32
#: the shared down GEMM's contraction tile: 32 partitions (the matmul's least), the block's
#: SH_C // 4 rows of x4 words first, zeros after
SH_KP = 32
FP4_X4 = nl.float4_e2m1fn_x4
FP8_X4 = nl.float8_e4m3fn_x4
F32 = nl.float32
BF = nl.bfloat16
#: static slot pairs (core c: slots 2p + c)
SLOTS = 5
#: the router weight as hi + lo bf16 (two matmuls a chunk: the fp32 GEMM's accuracy; else bf16)
ROUTER_HL = False
#: the core computing the routing (sent to the other), unless ROUTE_BOTH
ROUTE_CORE = 1
#: both cores compute the routing (no hand-off: each core's slot-id registers local)
ROUTE_BOTH = True
#: the routing's square root on the vector engine (no ACT table switch; unsupported: no DVE pow)
SQRT_DVE = False


def sh_cols(sh13, nkc):
    """The shared hidden columns a core holds (`moe_tp_layout.shared`: `sh13` `[2, 128, nkc * 2 SH_C]`,
    whole 32-column blocks: 32 at 64 ranks, 64 at 32)."""
    sh_c = sh13.shape[-1] // (2 * nkc)
    assert sh_c % SH == 0 and sh_c // 4 <= SH_KP and sh_c // SH <= 4, f"shared block width {sh_c}"
    return sh_c


#: (8d) the routed down GEMM's contraction packed: a core's SLOTS slots' 288 hidden values side by side
#: (`NKT` 128-partition tiles instead of one 72-of-128-partition tile a slot), every tile one matmul a
#: 512-column output tile: 3 down terms for a core's 4 slots instead of 4 (the PE streams the moving
#: weights a column a cycle whatever the contraction height); the scale rows by one SWDGE gather a
#: (core, tile) whose row index per partition picks the slot owning that packed scale group
PACK_DOWN = True
#: (8d) the two cores' shared-expert blocks' down products as one term: one 64-partition contraction tile
#: (core b's block at partitions 32 b ..), the halves combined after the exchange
SHARE_PACK = True
SH_KP2 = 64
#: x4-partitions of a slot's down contraction, tiles of the packed one
KPS = HID // 4
NKT = -(-SLOTS * KPS // PT)


def _tile_kp(t):
    """Contraction partitions of packed tile t (the MX matmul takes 32, 64 or 128)."""
    used = min(PT, SLOTS * KPS - PT * t)
    return 32 if used <= 32 else (64 if used <= 64 else PT)


def _last_slot(t):
    """The slot whose SwiGLU completes packed tile t (its values [512 t, 512 t + 512))."""
    return min(SLOTS - 1, ((t + 1) * KC - 1) // HID)


def _pack_consts(ng):
    """Shape-only constants of the packed scale gathers (once a launch, `moe_tp.CACHE`): `OH` [PT, NKT,
    SLOTS] fp32 = 1 where partition p is a scale partition (p = 32 q + r, r < 4: group g = 4 q + r) of
    tile t whose packed group G = 16 t + g belongs to slot k = G // 9, and `CK` [PT, NKT] fp32: added to
    that slot's `route_group` scale row index (16 id + g) it gives 16 id + G % 9; 16 ng (the zero
    expert's row) on scale partitions past the slots; (ng + 1) 16 (out of bounds: skipped) elsewhere."""
    key = ("b1_pack_down", ng, SLOTS, NKT)
    if key in M.CACHE:
        return M.CACHE[key]
    pi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pi, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    ri = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ri, data=pi, op0=nl.bitwise_and, operand0=31)
    qi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qi, data=pi, op0=nl.right_shift, operand0=5)
    rf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rf, src=ri)
    qf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qf, src=qi)
    g = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=g, data=qf, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=rf)
    isc = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=isc, data=rf, op0=nl.less, operand0=4.0)
    oob = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oob, data=isc, op0=nl.multiply, operand0=-float((ng + 1) * 16), op1=nl.add,
                       operand1=float((ng + 1) * 16))
    ti = nl.ndarray((PT, NKT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ti, pattern=[[16, NKT]], offset=0, channel_multiplier=0)
    G = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=G, src=ti)
    nisa.tensor_scalar(dst=G, data=G, op0=nl.add, operand0=g[:, 0:1])
    s = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.memset(s, 0.0)
    for k in range(1, SLOTS + 1):
        nisa.scalar_tensor_tensor(dst=s, data=G, op0=nl.greater_equal, operand0=float(9 * k), op1=nl.add, operand1=s)
    OH = nl.ndarray((PT, NKT, SLOTS), dtype=F32, buffer=nl.sbuf)
    for k in range(SLOTS):
        nisa.tensor_scalar(dst=OH[:, :, k], data=s, op0=nl.equal, operand0=float(k), op1=nl.multiply,
                           operand1=isc[:, 0:1])
    ins = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ins, data=s, op0=nl.less, operand0=float(SLOTS), op1=nl.multiply, operand1=isc[:, 0:1])
    # gl - g = G - 9 s - g
    a = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=a, data=s, op0=nl.multiply, operand0=-9.0, op1=nl.add, operand1=G)
    nisa.tensor_scalar(dst=a, data=a, op0=nl.subtract, operand0=g[:, 0:1])
    CK = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=CK, data1=a, data2=ins, op=nl.multiply)
    # (scale partitions past the slots: the zero expert's row; elsewhere out of bounds)
    bey = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bey, data=ins, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=isc[:, 0:1])
    nisa.scalar_tensor_tensor(dst=CK, data=bey, op0=nl.multiply, operand0=float(16 * ng), op1=nl.add, operand1=CK)
    nisa.tensor_scalar(dst=CK, data=CK, op0=nl.add, operand0=oob[:, 0:1])
    M.CACHE[key] = (OH, CK)
    return OH, CK


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def n_experts_of(gate_w, nkc):
    """The router's expert count (`gate_w` `[dim, E]` or in `layout.router_b1`'s layout)."""
    return 2 * gate_w.shape[2] // (4 * nkc) if len(gate_w.shape) == 3 else gate_w.shape[1]


def _oob_rows(sidx, nrows, S2):
    """`sidx` with its off-quadrant partitions' rows out of bounds (`nrows`: the first row past the
    scale rows, which the skip-mode gathers drop)."""
    pi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pi, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    r = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=pi, op0=nl.bitwise_and, operand0=31)
    rf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rf, src=r)
    off = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=off, data=rf, op0=nl.greater_equal, operand0=4.0, op1=nl.multiply, operand1=float(nrows))
    sf = nl.ndarray((PT, S2), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sf, src=sidx)
    nisa.tensor_scalar(dst=sf, data=sf, op0=nl.add, operand0=off[:, 0:1], op1=nl.minimum, operand1=float(nrows))
    so = nl.ndarray((PT, S2), dtype=nl.int32, buffer=nl.sbuf)
    _ROUTE_END[0] = nisa.tensor_copy(dst=so, src=sf)
    return so


#: the routing's last vector-engine instruction (`_oob_rows`), the pin of the MoE's first hc steps
_ROUTE_END = [None]


def moe_static(pid, nkc, gate_w, sh13, shs13, sh2, shs2, eng=nisa.engine.scalar, bufs=None, parts="rs"):
    """The block's static MoE weights into SBUF (issued early, e.g. during the attention side): this
    core's shared-expert gate|up block, the shared down blocks (its columns), its router half.
    `bufs`: a launch-level dict for the persistent buffers."""
    dim = nkc * KC
    dh = dim // 2
    c0 = pid * dh
    nh = n_experts_of(gate_w, nkc) // 2
    out = {}
    if "r" in parts:
        # the router weight, this core's half of the experts, rows in the staging's order:
        # wr[p, c, j, e] = gate_w[512 c + 4 p + j, nh * pid + e]
        wr = nl.ndarray((PT, nkc, 4, nh), dtype=BF, buffer=nl.sbuf)
        if ROUTER_HL and len(gate_w.shape) == 2:
            # the fp32 weight as hi + lo bf16 (the router's products of the bf16 inputs to ~2^-17, as
            # the fp32 GEMM): hi by a casting DMA, lo = w - hi from an fp32 copy
            ne = 2 * nh
            pat = dict(pattern=[[4 * ne, PT], [KC * ne, nkc], [ne, 4], [1, nh]], offset=pid * nh)
            nisa.dma_copy(dst=wr, src=gate_w.ap(**pat), **_hw(eng))
            wf = nl.ndarray((PT, nkc, 4, nh), dtype=F32, buffer=nl.sbuf)
            # (static descriptors: the fp32 copy is rejected by the hardware DGE, and the software DGE
            # would hold GpSimd, the all-reduces' trigger engine)
            nisa.dma_copy(dst=wf, src=gate_w.ap(**pat), dge_mode=nisa.dge_mode.none)
            wlo = nl.ndarray((PT, nkc, 4, nh), dtype=BF, buffer=nl.sbuf)
            # (on the vector engine: GpSimd triggers the all-reduces, a long GpSimd op ahead of the
            # trigger holds the collective)
            nisa.tensor_tensor(dst=wlo, data1=wf, data2=wr, op=nl.subtract)
            out["r"] = (wr, wlo)
        elif len(gate_w.shape) == 3:
            # (`layout.router_b1`: bf16, this core's rows contiguous)
            nisa.dma_copy(dst=wr.reshape((PT, nkc * 4 * nh)), src=gate_w[pid], **_hw(eng))
        else:
            # rows in the staging's order: wr[p, c, j, e] = gate_w[512 c + 4 p + j, nh * pid + e]
            ne = 2 * nh
            nisa.dma_copy(dst=wr, src=gate_w.ap(pattern=[[4 * ne, PT], [KC * ne, nkc], [ne, 4], [1, nh]],
                                                offset=pid * nh), **_hw(eng))
        out.setdefault("r", wr)
    if "s" not in parts:
        return out
    SH_C = sh_cols(sh13, nkc)
    W = 2 * SH_C
    swt = nl.ndarray((PT, nkc, W), dtype=FP8_X4, buffer=nl.sbuf)
    sws = nl.ndarray((PT, nkc, W), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=swt, src=sh13.view(FP8_X4).ap(pattern=[[nkc * W, PT], [W, nkc], [1, W]], offset=pid * PT * nkc * W),
                  **_hw(eng))
    for q in range(4):
        nisa.dma_copy(dst=sws[32 * q : 32 * q + 4, :, :],
                      src=shs13.ap(pattern=[[nkc * W, 4], [W, nkc], [1, W]], offset=(pid * 16 + 4 * q) * nkc * W),
                      **_hw(eng))
    # the shared down blocks: 32 hidden rows = 8 partitions, one scale row, in a 32-partition tile
    # (the matmul's least contraction) whose other rows are zero: one buffer a launch, zeroed once
    if bufs is not None and "sd" in bufs:
        sdt, sds = bufs["sd"]
    else:
        if SHARE_PACK:
            # (one tile: block b's rows at partitions 32 b .., its scale rows at 32 b + l)
            sdt = nl.ndarray((SH_KP2, dh), dtype=FP8_X4, buffer=nl.sbuf)
            sds = nl.ndarray((SH_KP2, dh), dtype=nl.uint8, buffer=nl.sbuf)
        else:
            sdt = nl.ndarray((SH_KP, 2, dh), dtype=FP8_X4, buffer=nl.sbuf)
            sds = nl.ndarray((SH_KP, 2, dh), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(sdt, 0, engine=nisa.engine.gpsimd)
        nisa.memset(sds, 0, engine=nisa.engine.gpsimd)
        if bufs is not None:
            bufs["sd"] = (sdt, sds)
    for b in range(2):
        if SHARE_PACK:
            nisa.dma_copy(dst=sdt[32 * b : 32 * b + SH_C // 4, :], src=sh2.view(FP8_X4)[b, :, c0 : c0 + dh],
                          **_hw(eng))
            nisa.dma_copy(dst=sds[32 * b : 32 * b + SH_C // SH, :], src=shs2[b, 0 : SH_C // SH, c0 : c0 + dh],
                          **_hw(eng))
            continue
        nisa.dma_copy(dst=sdt[0 : SH_C // 4, b, :], src=sh2.view(FP8_X4)[b, :, c0 : c0 + dh], **_hw(eng))
        # (one scale row a 32-column block: partitions 8 g .. 8 g + 7 = scale partition g, g < 4)
        nisa.dma_copy(dst=sds[0 : SH_C // SH, b, :], src=shs2[b, 0 : SH_C // SH, c0 : c0 + dh], **_hw(eng))
    out["s"] = (swt, sws, sdt, sds)
    return out


#: hc_in tail steps (`b1.Ticker`) at the MoE's waits: once the expert weights are issued, behind each
#: slot's gate|up GEMM
TICKS = dict(wait=4, slot=2)
#: the expert scale rows by SWDGE row gathers (one a slot, GpSimd) instead of 4 HWDGE DMAs a slot
#: (the dynamic-offset HWDGE DMAs cost ~0.4 us of sync / ACT issue each)
SCALES_SW = True
#: the scale gathers' off-quadrant partitions (p % 32 >= 4: codes the MX matmuls never read) pushed out
#: of bounds and skipped: 16 rows a gather instead of 128 (the in-bounds re-reads were ~40 % of the
#: MoE's HBM bytes)
SCALES_OOB = True
#: the overflow loop before the static slots' compute (else after their down GEMMs)
OVERFLOW_EARLY = False
#: the overflow loop inside a dynamic branch on "more than 2 * SLOTS active experts" (`nki.experimental.if_else`):
#: the zero-trip loop's all-engine synchronization, semaphore reset and descriptor setup only when taken
OVERFLOW_IF = False
#: (late) the overflow loop's down products added into the SBUF output tile before its store, "sbuf"
#: (no HBM read-modify-write after the store: the loop's entry not behind the store and its barrier),
#: else "hbm" (accumulated into the stored columns by DMA)
OVERFLOW_ACC = "hbm"
#: the slots' 64-column gate|up tails as one strided 256-column matmul a chunk (else 4; off: the MX
#: moving operand must be one contiguous run)
TAILS_MERGED = False
#: expert weight DMA queues: "sync" (all on the sync engine) or "split" (sync / ACT alternating)
EXPERT_Q = "fifo"
#: the expert weights' DMA priorities (0 highest): the gate|up slots in order, the down weights last
UP_PRIO = (0, 1, 2, 2, 2)
DOWN_PRIO = 3
#: the scale gathers' DMA priority (0 highest; their few rows ahead of the weights: slot 0's GEMM waits
#: for its scales), None for the default
SCALES_PRIO = 0
#: this core's first SLOTS - 1 slots' down products issued right after the last gate|up GEMM (the PE
#: busy while the last slot's SwiGLU / staging run), the rest after the exchange; the down weights of
#: this core's slots loaded first
DOWN_EARLY = False
#: the gate|up weight / scale tiles: GU_BUFS slot buffers (slot s in buffer s % GU_BUFS; slots past the
#: first GU_BUFS load once the GEMM before them in that buffer is done, after the down weights' DMAs):
#: SLOTS (every slot its own) or 2 (~35 KB a partition less live SBUF in the MoE -- with the attention's
#: persistent first-GEMM tile the compiler spilled that tile every block)
GU_BUFS = 2
#: the MoE's first hc_in tail steps (`TICKS["wait"]`) scheduled after the routing (`_ROUTE_END`)
PIN_TICKS = True
#: the down weights' DMA queue: "sync" (every slot's on the sync engine) or "split" (sync / ACT by slot:
#: the ACT queue's in-order stream holds the odd slots' behind the gate|up evacuations and SwiGLUs)
DOWN_Q = "sync"
#: the scale gathers (GpSimd) pinned ahead of the activation exchange (else the scheduler sinks the
#: other core's slots' gathers behind the exchange, next to their late consumers)
PIN_SCALES = True
#: ("fifo": the gate|up weights in slot order, then the down weights, on the sync queue: each slot's
#: arriving whole before the next's instead of all sharing the bandwidth. A queue holds at most ~16
#: DMAs awaited by distinct consumers: more and the LNC scheduler deadlocks, NCC_ISCH907)


def _route_one(pid, T, lg, gate_b, expert_lo, n_experts, topk, ng, route_scale, S2):
    """The routing on core ROUTE_CORE only, its results sent to the other core."""
    # the routing on core ROUTE_CORE only (the other core's vector engine runs hc_in(ffn)'s Sinkhorn),
    # its results sent over: [nz (ids, count) | slot weights | scale rows | gate|up rows | down rows |
    # the group's dense weights], int32 words
    NZ = ng + 3
    o_sw, o_sx, o_r13, o_r2, o_w = NZ, NZ + S2, NZ + 2 * S2, NZ + 3 * S2, NZ + 4 * S2
    NB = o_w + ng + 2
    rbuf = nl.ndarray((PT, NB), dtype=nl.int32, buffer=nl.sbuf)
    if pid == ROUTE_CORE:
        wexp, alist, nact, nz, slot = M.route_group(None, gate_b, expert_lo, T, n_experts, topk, ng, route_scale,
                                                    lg_sb=lg, keep=True)
        slotw, sidx, wr13, wr2, wsb = slot
        nisa.tensor_copy(dst=rbuf[:, 0:NZ], src=nz)
        nisa.tensor_copy(dst=rbuf[:, o_sw : o_sw + S2], src=slotw.view(nl.int32))
        nisa.tensor_copy(dst=rbuf[:, o_sx : o_sx + S2], src=sidx)
        nisa.tensor_copy(dst=rbuf[:, o_r13 : o_r13 + S2], src=wr13)
        nisa.tensor_copy(dst=rbuf[:, o_r2 : o_r2 + S2], src=wr2)
        nisa.tensor_copy(dst=rbuf[:, o_w : o_w + ng + 2], src=wsb.view(nl.int32))
    else:
        nisa.memset(rbuf, 0)
    rgot = nl.ndarray((PT, NB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.sendrecv(src=rbuf, dst=rgot, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=17,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    if pid != ROUTE_CORE:
        nz = rgot[:, 0:NZ]
        slotw = nl.ndarray((PT, S2), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slotw.view(nl.int32), src=rgot[:, o_sw : o_sw + S2])
        sidx = rgot[:, o_sx : o_sx + S2]
        wr13 = rgot[:, o_r13 : o_r13 + S2]
        wr2 = rgot[:, o_r2 : o_r2 + S2]
        wexp = nl.ndarray((PT, ng + 2), dtype=F32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=wexp, src=rgot[:, o_w : o_w + ng + 2].view(F32))
        alist = nl.ndarray((1, NZ), dtype=nl.int32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=alist, src=rgot[0:1, 0:NZ])
        nact = nisa.scalar_load(rgot[0:1, ng : ng + 1])
    return wexp, alist, nact, nz, slotw, sidx


def _overflow(pid, T, xq, xs, nkc, dim, c0, dh, w13v, s13, w2v, wexp, alist, nact, ng, limit, acc=None, dst=None):
    """The active slots past the static ones (more than 2 * SLOTS active experts in the group) in a
    dynamic loop (zero trips almost always), each pair's down products added into `acc` (`[T, dh]`
    fp32 SBUF, this core's columns) or `dst` (HBM, accumulated by DMA)."""
    S2 = 2 * SLOTS
    ntl = dh // NT
    T0 = 256
    r = HID - T0
    ub = M._UpBufs(nkc)
    db = [M._HalfDown(dh) for _ in range(2)]
    y1 = nl.ndarray((T, KC), dtype=BF, buffer=nl.sbuf)
    nisa.memset(y1, 0.0)

    def body(i):
        e01 = M._slot_ids(alist, ng, i, (0, 1))
        e = e01[pid]
        M._load_up(e, nkc, dim, w13v, s13, ub)
        for c, ec in enumerate(e01):
            M._load_down_half_dyn(ec, nkc, dim, c0, w2v, s13, db[c])
        wc = M._wcol(e, wexp, ng)
        a0 = nl.ndarray((T, NT), dtype=F32, buffer=nl.psum)
        a1 = nl.ndarray((T, W13 - NT), dtype=F32, buffer=nl.psum)
        for kc in range(nkc):
            nisa.nc_matmul_mx(dst=a0, stationary=xq[:, kc * T : kc * T + T], moving=ub.wt[:, kc, 0:NT],
                              stationary_scale=xs[:, kc * T : kc * T + T], moving_scale=ub.ws[:, kc * W13 : kc * W13 + NT],
                              accumulate=(kc > 0))
            nisa.nc_matmul_mx(dst=a1, stationary=xq[:, kc * T : kc * T + T], moving=ub.wt[:, kc, NT:W13],
                              stationary_scale=xs[:, kc * T : kc * T + T],
                              moving_scale=ub.ws[:, kc * W13 + NT : (kc + 1) * W13], accumulate=(kc > 0))
        gb = nl.ndarray((T, HID), dtype=BF, buffer=nl.sbuf)
        ubf = nl.ndarray((T, HID), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gb[:, 0:T0], src=a0[:, 0:T0])
        nisa.tensor_copy(dst=ubf[:, 0:T0], src=a0[:, T0:NT])
        nisa.tensor_copy(dst=gb[:, T0:HID], src=a1[:, 0:r])
        nisa.tensor_copy(dst=ubf[:, T0:HID], src=a1[:, r : 2 * r])
        g1 = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g1, data=gb, op0=nl.minimum, operand0=limit)
        u1 = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=u1, data=ubf, op0=nl.maximum, operand0=-limit, op1=nl.minimum, operand1=limit)
        s1_ = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=s1_, op=nl.silu, data=g1)
        v1 = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v1, data1=s1_, data2=u1, op=nl.multiply)
        nisa.tensor_scalar(dst=y1[:, 0:HID], data=v1, op0=nl.multiply, operand0=wc[0:T, 0:1])
        tq = nl.ndarray((PT, 4, T), dtype=BF, buffer=nl.psum)
        for j in range(4):
            nisa.nc_transpose(dst=tq[:, j, :], data=y1.ap(pattern=[[KC, T], [4, PT]], offset=j))
        yT1 = nl.ndarray((PT, T, 4), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yT1, src=tq.ap(pattern=[[4 * T, PT], [1, T], [T, 4]]))
        yq1 = nl.ndarray((PT, T), dtype=FP8_X4, buffer=nl.sbuf)
        ys1 = nl.ndarray((PT, T), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.quantize_mx(dst=yq1, src=yT1.reshape((PT, T * 4)), dst_scale=ys1)
        oq1 = nl.ndarray((PT, T), dtype=nl.int32, buffer=nl.sbuf)
        nisa.sendrecv(src=yq1.view(nl.int32), dst=oq1, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=15,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
        os1 = nl.ndarray((PT, T), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.sendrecv(src=ys1, dst=os1, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=16,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
        pr = [(yq1, ys1), (oq1.view(FP8_X4), os1)]
        if pid == 1:
            pr = pr[::-1]
        tmp = [nl.ndarray((T, NT), dtype=F32, buffer=nl.psum) for _ in range(ntl)]
        for c, (q_, s_) in enumerate(pr):
            for t, a in enumerate(tmp):
                nisa.nc_matmul_mx(dst=a, stationary=q_, moving=db[c].w2t[:, t * NT : (t + 1) * NT],
                                  stationary_scale=s_, moving_scale=db[c].w2s[:, t * NT : (t + 1) * NT],
                                  accumulate=(c > 0))
        ot = nl.ndarray((T, dh), dtype=F32, buffer=nl.sbuf)
        for t, a in enumerate(tmp):
            nisa.tensor_copy(dst=ot[:, t * NT : (t + 1) * NT], src=a)
        if dst is not None:
            nisa.dma_compute(dst=dst, srcs=[dst, ot], reduce_op=nl.add)
        else:
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ot, op=nl.add)

    n2 = ((nact + 1) // 2) * 2
    nl.fori_loop(S2, n2, body, step=2)


def moe(pid, T, xq, xs, xT, gate_w, gate_b, expert_lo, w13, s13, w2, sh13, shs13, sh2, shs2, cc2_src,
        limit, route_scale, topk, after_route=None, static=None, after_logits=None, before_down=None, tick=None,
        bufs=None):
    """The MoE of the 6 rows staged in `(xq, xs)` (MX, 6 columns a chunk) / `xT` (their bf16
    transposes `[128, nkc, 6, 4]`) into this core's columns of `cc2_src[0:T]` (fp32).
    `static`: `moe_static`'s tiles (else loaded here). Hooks (program order): `after_logits()` once
    the router's matmuls are issued, `after_route()` once the slot ids are known, `before_down()`
    before the down GEMMs; `tick(n)` steps a background chain at the waits (`TICKS`)."""
    tk = tick if tick is not None else (lambda n=1, after=None: None)
    _ROUTE_END[0] = None
    # (zeroed at the MoE's start on GpSimd: on the vector engine it sat in the routing's chain)
    y16_pre = nl.ndarray((T, (NKT if PACK_DOWN else SLOTS) + 1, KC), dtype=BF, buffer=nl.sbuf)
    nisa.memset(y16_pre, 0.0, engine=nisa.engine.gpsimd)
    nkc = xT.shape[1]
    dim = nkc * KC
    dh = dim // 2
    c0 = pid * dh
    n_experts = n_experts_of(gate_w, nkc)
    nh = n_experts // 2
    ng = w13.shape[0] - 1
    w13v = w13.view(FP4_X4)
    w2v = w2.view(FP4_X4)
    S2 = 2 * SLOTS
    q0, q1 = nisa.engine.sync, nisa.engine.scalar
    SH_C = sh_cols(sh13, nkc)
    W = 2 * SH_C
    # the router bias on the rows' partitions (fp32), a sync-queue DMA now (`nl.load` in the routing put
    # it on GpSimd's queue, ahead of the attention's exchanges)
    bias = nl.ndarray((PT, n_experts), dtype=F32, buffer=nl.sbuf)
    bsrc = gate_b.ap(pattern=[[0, T], [1, n_experts]], offset=0)
    if gate_b.dtype == F32:
        nisa.dma_copy(dst=bias[0:T], src=bsrc, **_hw(q0))
    else:
        b16 = nl.ndarray((PT, n_experts), dtype=gate_b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b16[0:T], src=bsrc, **_hw(q0))
        nisa.tensor_copy(dst=bias[0:T], src=b16[0:T])
    # the group's first expert id (`route_group`'s glo) likewise
    loi = nl.ndarray((PT, 1), dtype=expert_lo.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=loi[0:T], src=expert_lo.ap(pattern=[[0, T], [1, 1]], offset=0), **_hw(q0))
    lof = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lof[0:T], src=loi[0:T])
    static = dict(static or {})
    miss = "".join(k for k in "rs" if k not in static)
    if miss:
        static.update(moe_static(pid, nkc, gate_w, sh13, shs13, sh2, shs2, q1, parts=miss))
    wr = static["r"]
    wr_parts = wr if isinstance(wr, tuple) else (wr,)
    swt, sws, sdt, sds = static["s"]

    # ---------------------------------------------------------------- router
    lp = nl.ndarray((T, nh), dtype=F32, buffer=nl.psum)
    i = 0
    for c in range(nkc):
        for j in range(4):
            for w_ in wr_parts:
                last_r = nisa.nc_matmul(dst=lp, stationary=xT[:, c, :, j], moving=w_[:, c, j, :], accumulate=(i > 0))
                i += 1
    lg = nl.ndarray((PT, n_experts), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lg[0:T, pid * nh : pid * nh + nh], src=lp)
    nisa.sendrecv(src=lg[0:T, pid * nh : pid * nh + nh], dst=lg[0:T, (1 - pid) * nh : (2 - pid) * nh],
                  send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=12, dma_engine=nisa.dma_engine.gpsimd_dma)
    if after_logits is not None:
        after_logits(last_r)
    if ROUTE_BOTH:
        # the routing on both cores (the same inputs, the same results): the slot ids each core's DMAs
        # need without a cross-core hand-off
        wexp, alist, nact, nz, slot = M.route_group(None, gate_b, expert_lo, T, n_experts, topk, ng, route_scale,
                                                    lg_sb=lg, keep=True, sqrt_dve=SQRT_DVE, bias_sb=bias, lo_sb=lof,
                                                    gather_inbounds=not (SCALES_SW is True and SCALES_OOB))
        slotw, sidx, wr13, wr2, wsb = slot
    else:
        wexp, alist, nact, nz, slotw, sidx = _route_one(pid, T, lg, gate_b, expert_lo, n_experts, topk, ng,
                                                         route_scale, S2)
    ids = [nisa.scalar_load(nz[0:1, j : j + 1]) for j in range(S2)]
    if SCALES_SW == "hw4":
        # the scale loads' expert ids: the padding slots the zero expert's (finite codes for the stale
        # weight tiles their skipped weight DMAs leave; the routing weight zeroes them)
        nzc = nl.ndarray((1, S2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nzc, data=nz[0:1, 0:S2], op0=nl.minimum, operand0=ng)
        ids_s = [nisa.scalar_load(nzc[0:1, j : j + 1]) for j in range(S2)]

    # ---------------------------------------------------------------- routed weights
    # gate|up of this core's slots j = 2 s + pid side by side: wt[:, kc, s, :]; the down weights of
    # every slot (this core's columns): w2t[:, j, :]
    NB = min(GU_BUFS, SLOTS)
    wt = nl.ndarray((PT, NB, nkc, W13), dtype=FP4_X4, buffer=nl.sbuf)
    ws = nl.ndarray((PT, NB, nkc, W13), dtype=nl.uint8, buffer=nl.sbuf)
    # (PACK_DOWN: [core c, packed tile t] at c * NKT + t; else slot j)
    w2t = nl.ndarray((PT, 2 * NKT if PACK_DOWN else S2, dh), dtype=FP4_X4, buffer=nl.sbuf)
    w2s = nl.ndarray((PT, 2 * NKT if PACK_DOWN else S2, dh), dtype=nl.uint8, buffer=nl.sbuf)
    nks = nkc * W13 + dim
    kp = HID // 4
    s13r = s13.reshape((s13.shape[0] * 16, nks))

    nkh = nkc // 2
    qs = (q0, q0) if EXPERT_Q == "sync" else (q0, q1)
    if SCALES_SW is True and SCALES_OOB:
        # (`route_group(gather_inbounds=False)`: the off-quadrant rows already past the scale rows)
        _ROUTE_END[0] = M.ROUTE_END[0]
    hw = dict(dge_mode=nisa.dge_mode.hwdge, oob_mode=nisa.oob_mode.skip, priority=0)
    dq = (q0, q0) if DOWN_Q == "sync" else qs
    sprio = {} if SCALES_PRIO is None else dict(priority=SCALES_PRIO)
    gathers = []

    if SCALES_SW == "quad":
        # the quadrants' row indices on partitions 0..3 (a gather's index tile starts at partition 0):
        # partition 32 q + r's row 16 id + 4 q + r = sidx[r] + 4 q
        sidx_q = [sidx]
        for q4 in range(1, 4):
            t_ = nl.ndarray((4, S2), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=t_, data=sidx[0:4, :], op0=nl.add, operand0=4 * q4)
            sidx_q.append(t_)

    def load_up(s):
        # expert ids[j]'s gate|up rows (slot 0: the contraction halves on both queues) and its scale
        # rows into the quadrant layout (partitions 32 q .. 32 q + 3: scale rows 4 q .. 4 q + 3)
        j = 2 * s + pid
        if EXPERT_Q == "fifo":
            parts = ((0, nkc, q0),)
        else:
            parts = ((0, nkh, qs[0]), (nkh, nkc, qs[1])) if s == 0 else ((0, nkc, qs[s % 2]),)
        for k0, k1, q in parts:
            nisa.dma_copy(dst=wt[:, s % NB, k0:k1, :], src=w13v.ap(pattern=[[nkc * W13, PT], [W13, k1 - k0], [1, W13]],
                                                              offset=k0 * W13, scalar_offset=ids[j], indirect_dim=0),
                          engine=q, **dict(hw, priority=UP_PRIO[s]))
        if SCALES_SW == "quad":
            # four SWDGE row gathers onto the scale partitions only (32 q .. 32 q + 3: `sidx`'s rows,
            # in bounds; the zero expert's for padding ids): 16 rows, not 128 rows' worth of bytes
            for q4 in range(4):
                nisa.dma_copy(dst=ws[32 * q4 : 32 * q4 + 4, s % NB, :, :],
                              src=s13r.ap(pattern=[[nks, 4], [W13, nkc], [1, W13]], offset=0,
                                          vector_offset=sidx_q[q4][0:4, j : j + 1], indirect_dim=0),
                              dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
            return
        if SCALES_SW == "hw4":
            # the expert's 16 scale rows onto the quadrant partitions 32 q .. 32 q + 3: one HWDGE DMA a
            # quadrant (16 rows' bytes, not 128 partitions' as the SWDGE gather)
            for q4 in range(4):
                nisa.dma_copy(dst=ws[32 * q4 : 32 * q4 + 4, s % NB, :, :],
                              src=s13.ap(pattern=[[nks, 4], [W13, nkc], [1, W13]], offset=4 * q4 * nks,
                                         scalar_offset=ids_s[j], indirect_dim=0), engine=qs[(s + q4) % 2], **hw)
            return
        if SCALES_SW:
            # one SWDGE row gather (`sidx`: partition p's scale row, the zero expert's for padding ids)
            gathers.append(nisa.dma_copy(dst=ws[:, s % NB, :, :],
                                         src=s13r.ap(pattern=[[nks, PT], [W13, nkc], [1, W13]], offset=0,
                                                     vector_offset=sidx[0:PT, j : j + 1], indirect_dim=0),
                                         dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip, **sprio))
            return
        for q4 in range(4):
            nisa.dma_copy(dst=ws[32 * q4 : 32 * q4 + 4, s % NB, :, :],
                          src=s13.ap(pattern=[[nks, 4], [W13, nkc], [1, W13]], offset=4 * q4 * nks,
                                     scalar_offset=ids[j], indirect_dim=0), engine=qs[(s + 1) % 2], **hw)

    def load_down(j):
        if PACK_DOWN:
            # slot j's 72 rows onto its packed partitions [72 p, 72 p + 72) of core c's tiles (the
            # scale rows: one gather a tile, `pack_scales`)
            c, p = j % 2, j // 2
            lo = KPS * p
            for t in range(lo // PT, (lo + KPS - 1) // PT + 1):
                a, b = max(lo, PT * t), min(lo + KPS, PT * (t + 1))
                nisa.dma_copy(dst=w2t[a - PT * t : b - PT * t, c * NKT + t, :],
                              src=w2v.ap(pattern=[[dim, b - a], [1, dh]], offset=(a - lo) * dim + c0,
                                         scalar_offset=ids[j], indirect_dim=0),
                              **dict(hw, engine=dq[j % 2], priority=DOWN_PRIO))
            return
        nisa.dma_copy(dst=w2t[0:kp, j, :], src=w2v.ap(pattern=[[dim, kp], [1, dh]], offset=c0, scalar_offset=ids[j],
                                                      indirect_dim=0),
                      **dict(hw, engine=dq[j % 2], priority=DOWN_PRIO))
        if SCALES_SW == "quad":
            for q4 in range(4):
                nisa.dma_copy(dst=w2s[32 * q4 : 32 * q4 + 4, j, :],
                              src=s13r.ap(pattern=[[nks, 4], [1, dh]], offset=nkc * W13 + c0,
                                          vector_offset=sidx_q[q4][0:4, j : j + 1], indirect_dim=0),
                              dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
            return
        if SCALES_SW == "hw4":
            for q4 in range(4):
                nisa.dma_copy(dst=w2s[32 * q4 : 32 * q4 + 4, j, :],
                              src=s13.ap(pattern=[[nks, 4], [1, dh]], offset=4 * q4 * nks + nkc * W13 + c0,
                                         scalar_offset=ids_s[j], indirect_dim=0), engine=qs[(j + q4) % 2], **hw)
            return
        if SCALES_SW:
            gathers.append(nisa.dma_copy(dst=w2s[:, j, :],
                                         src=s13r.ap(pattern=[[nks, PT], [1, dh]], offset=nkc * W13 + c0,
                                                     vector_offset=sidx[0:PT, j : j + 1], indirect_dim=0),
                                         dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip, **sprio))
            return
        # the 16 scale rows (9 real, zero padding: the chunk's unused partitions read zero scales)
        for q4 in range(4):
            nisa.dma_copy(dst=w2s[32 * q4 : 32 * q4 + 4, j, :],
                          src=s13.ap(pattern=[[nks, 4], [1, dh]], offset=4 * q4 * nks + nkc * W13 + c0,
                                     scalar_offset=ids[j], indirect_dim=0), engine=qs[(j + 1) % 2], **hw)

    def pack_scales():
        # the packed tiles' scale rows: partition p of tile (c, t) <- row 16 id_j + G % 9 of slot j = 2 k + c
        # owning its packed group G (`_pack_consts`), one SWDGE gather a tile
        OH, CK = _pack_consts(ng)
        sf = nl.ndarray((PT, S2), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sf, src=sidx)
        pidx = nl.ndarray((PT, 2 * NKT), dtype=nl.int32, buffer=nl.sbuf)
        for c in range(2):
            pr = nl.ndarray((PT, NKT, SLOTS), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=OH, data2=sf.ap(pattern=[[S2, PT], [0, NKT], [2, SLOTS]], offset=c),
                               op=nl.multiply)
            ix = nl.ndarray((PT, NKT), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ix, op=nl.add, data=pr, axis=(2,))
            nisa.tensor_tensor(dst=ix, data1=ix, data2=CK, op=nl.add)
            nisa.tensor_copy(dst=pidx[:, c * NKT : (c + 1) * NKT], src=ix)
        for k in range(2 * NKT):
            gathers.append(nisa.dma_copy(dst=w2s[:, k, :],
                                         src=s13r.ap(pattern=[[nks, PT], [1, dh]], offset=nkc * W13 + c0,
                                                     vector_offset=pidx[0:PT, k : k + 1], indirect_dim=0),
                                         dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip, **sprio))

    if PACK_DOWN:
        assert SCALES_SW is True and SCALES_OOB and not DOWN_EARLY, "PACK_DOWN: SWDGE scale gathers only"
    for s in range(NB):
        load_up(s)
    for j in ([2 * p + pid for p in range(SLOTS)] + [2 * p + 1 - pid for p in range(SLOTS)] if DOWN_EARLY
              else range(S2)):
        load_down(j)
    if PACK_DOWN:
        pack_scales()
    if after_route is not None:
        after_route()
    acc = None
    if M.OVERFLOW and OVERFLOW_EARLY:
        # (the overflow slots' loop while the static slots' weights are in flight: slower, the loop's
        # all-engine synchronization stops the overlap of the routing, the mix and the static slots)
        acc = nl.ndarray((T, dh), dtype=F32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)
        _overflow(pid, T, xq, xs, nkc, dim, c0, dh, w13v, s13, w2v, wexp, alist, nact, ng, limit, acc=acc)
    # (pinned behind the routing's last vector-engine instruction: else the scheduler hoists the hc
    # steps into the routing's chain)
    tk(TICKS["wait"], after=_ROUTE_END[0] if PIN_TICKS else None)

    # ---------------------------------------------------------------- shared expert gate|up
    sa = nl.ndarray((T, W), dtype=F32, buffer=nl.psum)
    for kc in range(nkc):
        nisa.nc_matmul_mx(dst=sa, stationary=xq[:, kc * T : kc * T + T], moving=swt[:, kc, :],
                          stationary_scale=xs[:, kc * T : kc * T + T], moving_scale=sws[:, kc, :], accumulate=(kc > 0))

    # ---------------------------------------------------------------- routed gate|up (4 slots)
    # per slot: one 512- and one 64-column matmul a contraction chunk; bf16 gate / up
    ncol = SLOTS * W13
    gu = nl.ndarray((T, ncol), dtype=BF, buffer=nl.sbuf)
    nt1 = W13 - NT
    # (tile order per slot: g 0:256 | u 0:256 | g 256:288 | u 256:288)
    T0 = 256
    r = HID - T0
    nq = (NKT if PACK_DOWN else SLOTS) + 1
    # y16[t, s, 0:288] = bf16(SwiGLU(g, u) * w[t, slot 2 s + pid]); columns 288 .. 512 zero (the
    # 128-partition chunk); chunk SLOTS: the shared block's SwiGLU (no routing weight, 32 values)
    y16 = y16_pre
    y16f = y16.reshape((T, nq * KC))
    yq = nl.ndarray((PT, nq * T), dtype=FP8_X4, buffer=nl.sbuf)
    ys = nl.ndarray((PT, nq * T), dtype=nl.uint8, buffer=nl.sbuf)

    def stage_chunk(q):
        # chunk q of y16 staged (MX) into yq / ys columns q * T ..
        tpq = nl.ndarray((PT, 4, T), dtype=BF, buffer=nl.psum)
        for j in range(4):
            nisa.nc_transpose(dst=tpq[:, j, :], data=y16f.ap(pattern=[[nq * KC, T], [4, PT]], offset=q * KC + j))
        yTq = nl.ndarray((PT, T, 4), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=yTq, src=tpq.ap(pattern=[[4 * T, PT], [1, T], [T, 4]]))
        nisa.quantize_mx(dst=yq[:, q * T : (q + 1) * T], src=yTq.reshape((PT, T * 4)), dst_scale=ys[:, q * T : (q + 1) * T])

    def swiglu_slot(s_):
        # the slot's SwiGLU x routing weight into y16[:, s_] (overlapping the next slot's GEMM)
        o = s_ * W13
        g16 = nl.ndarray((T, HID), dtype=BF, buffer=nl.sbuf)
        u16 = nl.ndarray((T, HID), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=g16[:, 0:T0], src=gu[:, o : o + T0])
        nisa.tensor_copy(dst=u16[:, 0:T0], src=gu[:, o + T0 : o + 2 * T0])
        nisa.tensor_copy(dst=g16[:, T0:HID], src=gu[:, o + 2 * T0 : o + 2 * T0 + r])
        nisa.tensor_copy(dst=u16[:, T0:HID], src=gu[:, o + 2 * T0 + r : o + W13])
        gf = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gf, data=g16, op0=nl.minimum, operand0=limit)
        uf = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=uf, data=u16, op0=nl.maximum, operand0=-limit, op1=nl.minimum, operand1=limit)
        sg = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=gf)
        y = nl.ndarray((T, HID), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=sg, data2=uf, op=nl.multiply)
        j = 2 * s_ + pid
        if PACK_DOWN:
            # (the slot's values at [288 s, 288 s + 288) of the packed vector; a packed tile staged once
            # its last slot is in)
            nisa.tensor_scalar(dst=y16f[:, HID * s_ : HID * (s_ + 1)], data=y, op0=nl.multiply,
                               operand0=slotw[0:T, j : j + 1])
            for t in range(NKT):
                if _last_slot(t) == s_:
                    stage_chunk(t)
            return
        nisa.tensor_scalar(dst=y16[:, s_, 0:HID], data=y, op0=nl.multiply, operand0=slotw[0:T, j : j + 1])
        stage_chunk(s_)

    ntl = dh // NT
    dn = dict(acc=None, n=0, pin=None)

    def down_term(sq, ss, wq_, ws_, k):
        # one slot's (or shared block's) down products into this core's columns (PSUM accumulation)
        if dn["acc"] is None:
            dn["acc"] = [nl.ndarray((T, NT), dtype=F32, buffer=nl.psum) for _ in range(ntl)]
        for t, a in enumerate(dn["acc"]):
            mm = nisa.nc_matmul_mx(dst=a, stationary=sq[0:k, :], moving=wq_[0:k, t * NT : (t + 1) * NT],
                                   stationary_scale=ss[0:k, :], moving_scale=ws_[0:k, t * NT : (t + 1) * NT],
                                   accumulate=(dn["n"] > 0))
            pin_down = dn["pin"]
            if dn["n"] == 0 and t == 0 and pin_down is not None and hasattr(mm, "depends_on"):
                # (the next member's weight prefetch issued before the down GEMM: else scheduled late)
                for pd in (pin_down if isinstance(pin_down, list) else [pin_down]):
                    if pd is not None and hasattr(pd, "depends_on"):
                        mm.depends_on(pd, kind=nisa.dependency.schedule)
        dn["n"] += 1

    def own_term(p):
        j = 2 * p + pid
        return (yq[:, p * T : p * T + T], ys[:, p * T : p * T + T], w2t[:, j, :], w2s[:, j, :], PT)

    for s_ in range(SLOTS):
        a0 = nl.ndarray((T, NT), dtype=F32, buffer=nl.psum)
        a1 = nl.ndarray((T, nt1), dtype=F32, buffer=nl.psum)
        for kc in range(nkc):
            nisa.nc_matmul_mx(dst=a0, stationary=xq[:, kc * T : kc * T + T], moving=wt[:, s_ % NB, kc, 0:NT],
                              stationary_scale=xs[:, kc * T : kc * T + T], moving_scale=ws[:, s_ % NB, kc, 0:NT],
                              accumulate=(kc > 0))
            nisa.nc_matmul_mx(dst=a1, stationary=xq[:, kc * T : kc * T + T], moving=wt[:, s_ % NB, kc, NT:W13],
                              stationary_scale=xs[:, kc * T : kc * T + T], moving_scale=ws[:, s_ % NB, kc, NT:W13],
                              accumulate=(kc > 0))
        nisa.tensor_copy(dst=gu[:, s_ * W13 : s_ * W13 + NT], src=a0, engine=nisa.engine.scalar)
        nisa.tensor_copy(dst=gu[:, s_ * W13 + NT : (s_ + 1) * W13], src=a1, engine=nisa.engine.scalar)
        if s_ + NB < SLOTS:
            # the slot after this one in its buffer (program order: behind this slot's GEMM)
            load_up(s_ + NB)
        tk(TICKS["slot"])
        if DOWN_EARLY and s_ == SLOTS - 1:
            # this core's earlier slots' down products while the last slot's SwiGLU / staging run
            dn["pin"] = before_down() if before_down is not None else None
            for p_ in range(SLOTS - 1):
                down_term(*own_term(p_))
        swiglu_slot(s_)
    # the shared block's SwiGLU (no routing weight) as chunk SLOTS (32 hidden values)
    sg16 = nl.ndarray((T, W), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sg16, src=sa, engine=nisa.engine.scalar)
    sgf = nl.ndarray((T, SH_C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sgf, data=sg16[:, 0:SH_C], op0=nl.minimum, operand0=limit)
    suf = nl.ndarray((T, SH_C), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=suf, data=sg16[:, SH_C:W], op0=nl.maximum, operand0=-limit, op1=nl.minimum, operand1=limit)
    ssg = nl.ndarray((T, SH_C), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=ssg, op=nl.silu, data=sgf)
    nqs = nq - 1
    so = 128 * pid if SHARE_PACK else 0
    nisa.tensor_tensor(dst=y16[:, nqs, so : so + SH_C], data1=ssg, data2=suf, op=nl.multiply)
    stage_chunk(nqs)
    # swap with the other core (its slots and its shared block): int32 / uint8 views
    # (`bufs`: persistent receive tiles, `b1attn.SWAP_PERSIST`: no false dependency on a transient tile)
    pk = ("moe_x", nq * T, pid)
    if bufs is not None and pk in bufs:
        oq, os_ = bufs[pk]
    else:
        oq = nl.ndarray((PT, nq * T), dtype=nl.int32, buffer=nl.sbuf, name=f"mxq_{pid}" if bufs is not None else "")
        os_ = nl.ndarray((PT, nq * T), dtype=nl.uint8, buffer=nl.sbuf, name=f"mxs_{pid}" if bufs is not None else "")
        if bufs is not None:
            bufs[pk] = (oq, os_)
    xq_sr = nisa.sendrecv(src=yq.view(nl.int32), dst=oq, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=13,
                          dma_engine=nisa.dma_engine.gpsimd_dma)
    if PIN_SCALES and hasattr(xq_sr, "depends_on"):
        for g_ in gathers:
            if g_ is not None and hasattr(g_, "depends_on"):
                xq_sr.depends_on(g_, kind=nisa.dependency.schedule)
    nisa.sendrecv(src=ys, dst=os_, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=14,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    oqv = oq.view(FP8_X4)
    if not DOWN_EARLY:
        dn["pin"] = before_down() if before_down is not None else None

    # ---------------------------------------------------------------- down: this core's columns
    terms = []
    # (this core's slots first: the other core's arrive by the exchange meanwhile)
    for c in (pid, 1 - pid):
        mine = c == pid
        if PACK_DOWN:
            for t in range(NKT):
                sq = (yq if mine else oqv)[:, t * T : t * T + T]
                ss = (ys if mine else os_)[:, t * T : t * T + T]
                terms.append((sq, ss, w2t[:, c * NKT + t, :], w2s[:, c * NKT + t, :], _tile_kp(t)))
        for p in range(SLOTS if not PACK_DOWN else 0):
            if mine and DOWN_EARLY and p < SLOTS - 1:
                continue
            j = 2 * p + c
            sq = (yq if mine else oqv)[:, p * T : p * T + T]
            ss = (ys if mine else os_)[:, p * T : p * T + T]
            terms.append((sq, ss, w2t[:, j, :], w2s[:, j, :], PT))
        if SHARE_PACK:
            continue
        sq = (yq if mine else oqv)[:, nqs * T : nqs * T + T]
        ss = (ys if mine else os_)[:, nqs * T : nqs * T + T]
        terms.append((sq, ss, sdt[:, c, :], sds[:, c, :], SH_KP))
    if SHARE_PACK:
        # both blocks in one stationary: this core's partitions 32 pid .. from its own staging, the
        # other's 32 (1 - pid) .. from the exchange
        cq = nl.ndarray((SH_KP2, T), dtype=nl.int32, buffer=nl.sbuf)
        cs = nl.ndarray((SH_KP2, T), dtype=nl.uint8, buffer=nl.sbuf)
        for b, (q_, s_) in ((pid, (yq.view(nl.int32), ys)), (1 - pid, (oq, os_))):
            nisa.tensor_copy(dst=cq[32 * b : 32 * b + 32, :], src=q_[32 * b : 32 * b + 32, nqs * T : nqs * T + T])
            nisa.tensor_copy(dst=cs[32 * b : 32 * b + 32, :], src=s_[32 * b : 32 * b + 32, nqs * T : nqs * T + T])
        terms.append((cq.view(FP8_X4), cs, sdt, sds, SH_KP2))
    for term in terms:
        down_term(*term)
    dacc = dn["acc"]
    o = nl.ndarray((T, dh), dtype=F32, buffer=nl.sbuf)
    for t, a in enumerate(dacc):
        if acc is None:
            nisa.tensor_copy(dst=o[:, t * NT : (t + 1) * NT], src=a,
                             engine=nisa.engine.scalar if t % 2 else nisa.engine.vector)
        else:
            nisa.tensor_tensor(dst=o[:, t * NT : (t + 1) * NT], data1=a, data2=acc[:, t * NT : (t + 1) * NT], op=nl.add)
    dst = cc2_src[0:T, c0 : c0 + dh]
    if M.OVERFLOW and not OVERFLOW_EARLY and OVERFLOW_ACC == "sbuf":
        # the slots past the static ones (more than 2 * SLOTS active experts), added in SBUF
        _overflow(pid, T, xq, xs, nkc, dim, c0, dh, w13v, s13, w2v, wexp, alist, nact, ng, limit, acc=o)
    nisa.dma_copy(dst=dst, src=o, **_hw(q0))
    if M.OVERFLOW and not OVERFLOW_EARLY and OVERFLOW_ACC != "sbuf":
        # the slots past the static ones (more than 2 * SLOTS active experts), added in HBM
        run_ovf = lambda: _overflow(pid, T, xq, xs, nkc, dim, c0, dh, w13v, s13, w2v, wexp, alist, nact, ng, limit,
                                    dst=dst)
        if OVERFLOW_IF:
            import nki.experimental as ne
            fl = nl.ndarray((1, 1), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fl, src=nz[0:1, ng : ng + 1])
            nisa.tensor_scalar(dst=fl, data=fl, op0=nl.greater, operand0=float(S2))
            fi = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fi, src=fl)
            reg = nisa.register_alloc()
            nisa.register_load(reg, fi)
            ne.if_else(reg, run_ovf)
        else:
            run_ovf()


