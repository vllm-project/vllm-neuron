# SPDX-License-Identifier: Apache-2.0
"""Hy3 decode MoE block in one NKI kernel (``HY3_DECODE_MOE_FUSED=1``).

Per layer and rank, for T <= 128 decode tokens:
  post-attention RMSNorm
  -> router: BF16 logits, sigmoid, + expert bias, top-k, renormalize, x router scaling factor
  -> the rank's routed experts (all local experts, weighted by their affinities)
  -> + the shared expert's decode shard (HYV3StaticFP8MLP, num_shards=8), weighted by
     ``shared_decode_weight``.
The output is the rank's partial [T, H]. The caller all-reduces it (``HYV3MoE._combine``).

Why it is faster than torch ops + ``moe_tkg``:
  * Static FP8 activations (W8A8) with the checkpoint's calibrated input scales, doubled at
    load time like every other input scale here. The matmuls contract 256 rows per pass
    (``matmul_perf_mode.double_row``); BF16 activations allow only 128.
  * The weights are read in HYV3MoE's own layouts ([E, H, 2, I], [E, I, H]) with long
    contiguous DMA rows. The two cores of the logical NeuronCore (LNC=2) each take one half
    of H for gate/up, so a row is a full 2 * I bytes. They exchange the halves of their
    partial sums (``sendrecv``), and each keeps one half of I for the activation and the
    down projection. A last exchange sums the two halves of the output.
  * No torch ops between the attention all-reduce and the MoE all-reduce.

Inputs:
  hidden       [T, H] bf16: the residual stream after attention, before the norm
  ln_w         [1, H] bf16: post_attention_layernorm weight
  router_wT    [H, E_all] bf16: the router weight transposed. The checkpoint router is BF16.
  expert_bias  [1, E_all] fp32: selection-only bias
  ebase        [1, 1] fp32: this rank's first local expert id
  gate_up_w    [E, H, 2, I] fp8, down_w [E, I, H] fp8: the local experts
  sh_gate_w, sh_up_w [H, IS] fp8, sh_down_w [IS, H] fp8: the shared decode shard
  scales       [1, 4 * (E + 1)] fp32. Per expert, the shared one last:
               gate dequant (w * in), up dequant / down in, down dequant (w * in, times
               shared_decode_weight for the shared shard), 1 / gate-up in
  eps, route_scale, top_k: compile-time constants

Top-k by threshold: the experts whose biased score is >= the k-th largest (``nisa.max8``)
are selected. An exact tie at the threshold would select more than k; this was not seen
with fp32 scores, and the tokens matched the torch router 19/19 in serving.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import os

# HY3_MOE_STATIC_DGE=1: static DMA descriptors for the weight streams. On one core the
# layer-to-layer latency drops from 0.193 to 0.152 ms, but the descriptors live in HBM: about
# 1.1 GB more "DMA rings IO" per core with the decode buckets [1, 4, 8]. At KV_CAP 0.61 that
# overflows HBM in serving. Off by default.
_WKW = dict(dge_mode=nisa.dge_mode.none) if os.environ.get("HY3_MOE_STATIC_DGE", "0") == "1" else {}

P = 128
FP8_MAX = 240.0
RING = 6
DN_COLS = 1024


@nki.jit
def moe_tkg_fused(hidden, ln_w, router_wT, expert_bias, ebase, gate_up_w, down_w, sh_gate_w, sh_up_w,
                sh_down_w, scales, eps=1e-5, route_scale=1.0, top_k=8):
    T, H = hidden.shape
    E, _, _, I = gate_up_w.shape
    EA = router_wT.shape[1]
    IS = sh_gate_w.shape[1]
    NE = E + 1
    H1 = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    n_k = H1 // 2 // n_prg
    HH = H // n_prg
    h0 = pid * HH

    out = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.shared_hbm)

    sc_t = nl.ndarray((P, 4 * NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc_t[0:T, 0 : 4 * NE], src=scales.ap(pattern=[[0, T], [1, 4 * NE]], offset=0))
    sc_p = nl.ndarray((P, 4 * NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc_p[0:P, 0 : 4 * NE], src=scales.ap(pattern=[[0, P], [1, 4 * NE]], offset=0))

    # ---- RMSNorm (fp32 math, bf16 result like the torch path) ----
    hf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hf[0:T, 0:H], src=hidden.ap(pattern=[[H, T], [1, H]], offset=0))
    gam = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:H], src=ln_w.ap(pattern=[[0, T], [1, H]], offset=0))
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:H], op=nl.square, data=hf[0:T, 0:H], reduce_op=nl.add,
                    reduce_res=ss[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / H, bias=eps)
    nrm = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=nrm[0:T, 0:H], data=hf[0:T, 0:H], op0=nl.multiply,
                              operand0=rr[0:T, 0:1], op1=nl.multiply, operand1=gam[0:T, 0:H])
    nT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:P, 0:T], data=nrm[0:T, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=nT[0:P, h1, 0:T], src=tp[0:P, 0:T])

    # ---- router: logits [T, EA] = normed @ W^T (bf16 x bf16, fp32 accumulate) ----
    rw = nl.ndarray((P, H1, EA), dtype=router_wT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[0:P, 0:H1, 0:EA], src=router_wT.ap(pattern=[[EA, P], [P * EA, H1], [1, EA]], offset=0))
    lg = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.psum)
    for h1 in range(H1):
        nisa.nc_matmul(dst=lg[0:T, 0:EA], stationary=nT[0:P, h1, 0:T], moving=rw[0:P, h1, 0:EA])
    scr = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scr[0:T, 0:EA], op=nl.sigmoid, data=lg[0:T, 0:EA])
    bia = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bia[0:T, 0:EA], src=expert_bias.ap(pattern=[[0, T], [1, EA]], offset=0))
    sb = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(sb[0:T, 0:EA], scr[0:T, 0:EA], bia[0:T, 0:EA], nl.add)
    top8 = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top8[0:T, 0:8], src=sb[0:T, 0:EA])
    sel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sel[0:T, 0:EA], data=sb[0:T, 0:EA], op0=nl.greater_equal,
                       operand0=top8[0:T, top_k - 1 : top_k])
    wsel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    wsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(wsel[0:T, 0:EA], scr[0:T, 0:EA], sel[0:T, 0:EA], nl.multiply)
    nisa.tensor_reduce(wsum[0:T, 0:1], op=nl.add, data=wsel[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=wsum[0:T, 0:1], data=wsum[0:T, 0:1], op0=nl.add, operand0=1e-20)
    winv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=winv[0:T, 0:1], data=wsum[0:T, 0:1])
    nisa.tensor_scalar(dst=winv[0:T, 0:1], data=winv[0:T, 0:1], op0=nl.multiply, operand0=route_scale)
    # local experts' affinities: one-hot columns ebase + j picked by an iota compare
    eid = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(eid[0:T, 0:EA], pattern=[[1, EA]], offset=0, channel_multiplier=0)
    eb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb[0:T, 0:1], src=ebase.ap(pattern=[[0, T], [1, 1]], offset=0))
    rel = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rel[0:T, 0:EA], data=eid[0:T, 0:EA], op0=nl.subtract, operand0=eb[0:T, 0:1])
    aff_sb = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(E):
        oh = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oh[0:T, 0:EA], data=rel[0:T, 0:EA], op0=nl.equal, operand0=float(j))
        pk = nl.ndarray((P, EA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(pk[0:T, 0:EA], wsel[0:T, 0:EA], oh[0:T, 0:EA], nl.multiply)
        nisa.tensor_reduce(aff_sb[0:T, j : j + 1], op=nl.add, data=pk[0:T, 0:EA], axis=1, keepdims=True)
    nisa.tensor_scalar(dst=aff_sb[0:T, 0:E], data=aff_sb[0:T, 0:E], op0=nl.multiply, operand0=winv[0:T, 0:1])

    col_d = nl.ndarray((P, NE), dtype=nl.float32, buffer=nl.sbuf)
    for e in range(E):
        nisa.tensor_tensor(
            col_d[0:T, e : e + 1], aff_sb[0:T, e : e + 1], sc_t[0:T, 4 * e + 2 : 4 * e + 3], nl.multiply
        )
    nisa.tensor_copy(dst=col_d[0:T, E : E + 1], src=sc_t[0:T, 4 * E + 2 : 4 * E + 3])

    gu_ring = [nl.ndarray((P, 2, 2 * I), dtype=gate_up_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    dn_ring = [nl.ndarray((P, 2, DN_COLS), dtype=down_w.dtype, buffer=nl.sbuf) for _ in range(RING)]
    acc = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    slots = [0, 0]

    for e in range(NE):
        shared = e == E
        Ie = IS if shared else I
        ILe = Ie // n_prg
        n_ic = ILe // P

        xb = nl.ndarray((P, 2 * n_k, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=xb[0:P, 0 : 2 * n_k, 0:T], data=nT[0:P, pid * 2 * n_k : (pid + 1) * 2 * n_k, 0:T],
            op0=nl.multiply, operand0=sc_p[0:P, 4 * e + 3 : 4 * e + 4],
            op1=nl.minimum, operand1=FP8_MAX,
        )
        xq = nl.ndarray((P, 2 * n_k, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xq[0:P, 0 : 2 * n_k, 0:T], data=xb[0:P, 0 : 2 * n_k, 0:T],
                           op0=nl.maximum, operand0=-FP8_MAX)

        chunks = [(g, c0, min(512, Ie - c0)) for g in range(2) for c0 in range(0, Ie, 512)]
        ps = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in chunks]
        for k in range(n_k):
            buf = gu_ring[slots[0] % RING]
            slots[0] += 1
            row0 = h0 + 2 * k * P
            if shared:
                for g, w in ((0, sh_gate_w), (1, sh_up_w)):
                    nisa.dma_copy(
                        dst=buf[0:P, 0:2, g * IS : (g + 1) * IS],
                        src=w.ap(pattern=[[IS, P], [P * IS, 2], [1, IS]], offset=row0 * IS),
                        **_WKW,
                    )
            else:
                nisa.dma_copy(
                    dst=buf[0:P, 0:2, 0 : 2 * I],
                    src=gate_up_w.ap(pattern=[[2 * I, P], [P * 2 * I, 2], [1, 2 * I]],
                                     offset=e * (H * 2 * I) + row0 * 2 * I),
                    **_WKW,
                )
            for ci, (g, c0, cs) in enumerate(chunks):
                nisa.nc_matmul(
                    dst=ps[ci][0:T, 0:cs],
                    stationary=xq[0:P, 2 * k : 2 * k + 2, 0:T],
                    moving=buf[0:P, 0:2, g * Ie + c0 : g * Ie + c0 + cs],
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )
        part = nl.ndarray((P, 2, Ie), dtype=nl.float32, buffer=nl.sbuf)
        for ci, (g, c0, cs) in enumerate(chunks):
            nisa.tensor_copy(dst=part[0:T, g, c0 : c0 + cs], src=ps[ci][0:T, 0:cs])
        send = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        recv = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=send[0:T, 0:2, 0:ILe], src=part[0:T, 0:2, other * ILe : (other + 1) * ILe])
        nisa.sendrecv(src=send[0:T, 0:2, 0:ILe], dst=recv[0:T, 0:2, 0:ILe],
                      send_to_rank=other, recv_from_rank=other, pipe_id=e % 2)
        gu = nl.ndarray((P, 2, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(gu[0:T, 0:2, 0:ILe], recv[0:T, 0:2, 0:ILe],
                           part[0:T, 0:2, pid * ILe : (pid + 1) * ILe], nl.add)

        g_sb = nl.ndarray((P, ILe), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sb[0:T, 0:ILe], op=nl.silu, data=gu[0:T, 0, 0:ILe],
                        scale=sc_t[0:T, 4 * e : 4 * e + 1])
        it = nl.ndarray((P, ILe), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=it[0:T, 0:ILe], data=gu[0:T, 1, 0:ILe],
            op0=nl.multiply, operand0=sc_t[0:T, 4 * e + 1 : 4 * e + 2],
            op1=nl.multiply, operand1=g_sb[0:T, 0:ILe],
        )
        nisa.tensor_scalar(dst=it[0:T, 0:ILe], data=it[0:T, 0:ILe],
                           op0=nl.minimum, operand0=FP8_MAX, op1=nl.maximum, operand1=-FP8_MAX)
        itT = nl.ndarray((P, n_ic, P), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for c in range(n_ic):
            tp2 = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp2[0:P, 0:T], data=it[0:T, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=itT[0:P, c, 0:T], src=tp2[0:P, 0:T])

        dw = sh_down_w if shared else down_w
        base = 0 if shared else e * (I * H)
        dr = n_ic % 2 == 0  # double row needs pairs of 128-row chunks
        n_pass = n_ic // 2 if dr else n_ic
        for hq in range(H // DN_COLS):
            pd = [nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(DN_COLS // 512)]
            for j in range(n_pass):
                db = dn_ring[slots[1] % RING]
                slots[1] += 1
                rows = 2 if dr else 1
                r0 = pid * ILe + rows * j * P
                nisa.dma_copy(
                    dst=db[0:P, 0:rows, 0:DN_COLS],
                    src=dw.ap(pattern=[[H, P], [P * H, rows], [1, DN_COLS]],
                              offset=base + r0 * H + hq * DN_COLS),
                    **_WKW,
                )
                for b in range(DN_COLS // 512):
                    if dr:
                        nisa.nc_matmul(
                            dst=pd[b][0:T, 0:512],
                            stationary=itT[0:P, 2 * j : 2 * j + 2, 0:T],
                            moving=db[0:P, 0:2, b * 512 : (b + 1) * 512],
                            perf_mode=nisa.matmul_perf_mode.double_row,
                        )
                    else:
                        nisa.nc_matmul(
                            dst=pd[b][0:T, 0:512],
                            stationary=itT[0:P, j, 0:T],
                            moving=db[0:P, 0, b * 512 : (b + 1) * 512],
                        )
            for b in range(DN_COLS // 512):
                c0 = hq * DN_COLS + b * 512
                if e == 0:
                    nisa.tensor_scalar(dst=acc[0:T, c0 : c0 + 512], data=pd[b][0:T, 0:512],
                                       op0=nl.multiply, operand0=col_d[0:T, e : e + 1])
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:T, c0 : c0 + 512], data=pd[b][0:T, 0:512],
                        op0=nl.multiply, operand0=col_d[0:T, e : e + 1],
                        op1=nl.add, operand1=acc[0:T, c0 : c0 + 512],
                    )

    # Sum the two programs' partials: each owns one half of H.
    send_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
    recv_o = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=send_o[0:T, 0:HH], src=acc[0:T, other * HH : (other + 1) * HH])
    nisa.sendrecv(src=send_o[0:T, 0:HH], dst=recv_o[0:T, 0:HH],
                  send_to_rank=other, recv_from_rank=other, pipe_id=2)
    o_sb = nl.ndarray((P, HH), dtype=hidden.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(o_sb[0:T, 0:HH], recv_o[0:T, 0:HH], acc[0:T, h0 : h0 + HH], nl.add)
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=o_sb[0:T, 0:HH])
    return out
