# SPDX-License-Identifier: Apache-2.0
"""The compressor of a kv-source block for one 6-row span, on chip (round 5): this step's compressed
latents and index keys written into the caches before the block's indexer (`b1idx`) and attention
read them.

    ratio 2: pkv = xn @ cm_wkv_t, psc = xn @ cm_wgate_t (fp32; core 0 / core 1) -> pool caches
             lat_raw[g] = sum_m pkv[m] softmax_m(psc[m]) over the group's member slots (both cores)
    lat = rmsnorm(lat_raw) -> rope, fp4 (e4m3 scales, blocks of 16) -> compress_kv_cache   (core 0)
    k   = rmsnorm(lat @ ix_wk.T) -> rope, fp4 (pow2, blocks of 32)  -> index_k_cache       (core 1)

Same arithmetic as `mk_compress` (`fp32_proj_rows`, `pool_groups`, `norm_rows`, `rope_fp4_put`,
`bf16_gemm_rows`); the projections' fp32 sums run in a different order."""

import nki.isa as nisa
import nki.language as nl

from . import b1attn as A
from .mk_common import PT, ROPE_DIM
from .mk_compress import fp4_qdq_tile

F32 = nl.float32
BF = nl.bfloat16
T6 = A.T6
KC = A.KC


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def usable(cs):
    """The shapes this path handles: ratio 1 or 2, bf16 projection weights, 512-wide latents."""
    r = cs["ratio"]
    ok = (r in (1, 2) and cs["wkv_t"].dtype == BF and len(cs["wkv_t"].shape) == 2 and cs["wkv_t"].shape[1] == KC
          and tuple(cs["ix_wk"].shape) == (PT, KC) and cs["wslots"].shape[0] <= PT)
    if r == 2:
        ok = ok and cs["wgate_t"].dtype == BF and cs["mslots"].shape[0] == r * cs["wslots"].shape[0]
    else:
        ok = ok and cs["wslots"].shape[0] == T6
    return ok


def prep(pid, cs, dim, eng):
    """Block-start loads: this core's projection weight (core 0 kv, core 1 gate) in the staged xn's row
    order `[128, C, 4, 512]` (row 512 c + 4 p + j), the norm weights, the group tables."""
    nkc = dim // KC
    if cs["ratio"] == 2:
        w = cs["wkv_t"] if pid == 0 else cs["wgate_t"]
        n = w.shape[1]
        wt = nl.ndarray((PT, nkc, 4, n), dtype=BF, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w.ap(pattern=[[4 * n, PT], [KC * n, nkc], [1, 4 * n]], offset=0), **_hw(eng))
    else:
        # (ratio 1: the latent projection's column halves, one a core)
        w = cs["wkv_t"]
        nf = w.shape[1]
        n = nf // 2
        wt = nl.ndarray((PT, nkc, 4, n), dtype=BF, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w.ap(pattern=[[4 * nf, PT], [KC * nf, nkc], [nf, 4], [1, n]], offset=pid * n),
                      **_hw(eng))
    st = dict(cs)
    st["wt"] = wt
    ng = cs["wslots"].shape[0]
    st["ng"] = ng
    sync = nisa.engine.sync
    ws = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws[0:T6, :], src=cs["write_slots"][0:T6, 0:1], **_hw(sync))
    st["ws"] = ws
    gw = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gw[0:ng, :], src=cs["wslots"][0:ng, 0:1], **_hw(sync))
    st["gw"] = gw
    ms = []
    for m in range(cs["ratio"] if cs["ratio"] > 1 else 0):
        t = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:ng, :], src=cs["mslots"].ap(pattern=[[cs["ratio"], ng], [1, 1]], offset=m), **_hw(sync))
        ms.append(t)
    st["ms"] = ms
    st["nw_lat"] = A.bcast_rows(cs["norm_w"], cs["wkv_t"].shape[1], ng, BF, eng=sync)
    kd = cs["ix_wk"].shape[0]
    st["nw_k"] = A.bcast_rows(cs["knorm_w"], kd, ng, BF, eng=sync)
    # ix_wk [kd, 512] (n on partitions): transposed on chip once a block (4 PE transposes)
    wk = nl.ndarray((PT, KC), dtype=BF, buffer=nl.sbuf)
    nisa.dma_copy(dst=wk[0:kd, :], src=cs["ix_wk"][0:kd, 0:KC], **_hw(sync))
    st["wk_rows"] = wk
    # the groups' (cos | sin) rows: positions grp_pos
    pos = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos[0:ng, :], src=cs["pos"][0:ng, 0:1], **_hw(sync))
    csg = nl.ndarray((PT, ROPE_DIM), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=csg[0:ng, :], src=cs["freqs"].ap(pattern=[[ROPE_DIM, ng], [1, ROPE_DIM]],
                                                       vector_offset=pos[0:ng, 0:1], indirect_dim=0))
    st["csg"] = csg
    return st


def _norm(x16, ng, n, w_b):
    """`mk_common.norm_rows` of the bf16 rows `x16[0:ng, 0:n]`: bf16 out."""
    ss = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
    junk = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=junk[0:ng], op=nl.square, data=x16[0:ng, 0:n], reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=ss[0:ng, 0:1])
    rs = A.rstd_of(ss, ng, n)
    out = nl.ndarray((PT, n), dtype=BF, buffer=nl.sbuf)
    yv = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.activation(dst=yv[0:ng], op=nl.copy, data=x16[0:ng, 0:n], scale=rs[0:ng, 0:1])
    nisa.tensor_tensor(dst=out[0:ng], data1=yv[0:ng], data2=w_b[0:ng, 0:n], op=nl.multiply)
    return out


def _rope_fp4_store(t16, ng, n, cs, blk, e4m3, cache, gw):
    """`rope_fp4_put` of the rows `t16[0:ng, 0:n]` (one head of width n: rope on its last 64 columns,
    the groups' positions), stored to `cache[grp_wslots[g]]`."""
    A.rope_cols(t16, ng, n - ROPE_DIM, n, 1, cs, inverse=False)
    tf = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tf[0:ng], src=t16[0:ng, 0:n])
    q = fp4_qdq_tile(tf, ng, n, blk, e4m3)
    nisa.dma_copy(dst=cache.ap(pattern=[[n, ng], [1, n]], vector_offset=gw[0:ng, 0:1], indirect_dim=0), src=q[0:ng, :])


def compress(pid, st, xT, barrier):
    """The step's compressor (see the module docstring), from the staged xn (`stage_xn(want_t=True)`'s
    `xT [128, C, 6, 4]` bf16); the caches written and core-barriered on return."""
    wt = st["wt"]
    nkc, n = wt.shape[1], wt.shape[3]
    ng = st["ng"]
    ratio = st["ratio"]
    # 1. this core's projection (fp32, exact bf16 products) -> its pool cache rows (ratio 2) / its half of
    #    the latent (ratio 1)
    ps = nl.ndarray((T6, n), dtype=F32, buffer=nl.psum)
    i = 0
    for c in range(nkc):
        for j in range(4):
            nisa.nc_matmul(dst=ps, stationary=xT.ap(pattern=[[nkc * T6 * 4, PT], [4, T6]], offset=c * T6 * 4 + j),
                           moving=wt[:, c, j, :], accumulate=(i > 0))
            i += 1
    if ratio == 1:
        # bf16 (`bf16_gemm_rows`), the halves swapped: lat_raw [6, 512] on both cores
        h16 = nl.ndarray((T6, n), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=h16, src=ps)
        o16 = A.swap(pid, h16, 35)
        n = 2 * n
        lat_raw = nl.ndarray((PT, n), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lat_raw[0:T6, pid * (n // 2) : (pid + 1) * (n // 2)], src=h16)
        nisa.tensor_copy(dst=lat_raw[0:T6, (1 - pid) * (n // 2) : (2 - pid) * (n // 2)], src=o16)
        return _finish(pid, st, lat_raw, ng, n, barrier)
    pr = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pr[0:T6], src=ps)
    pool = st["pool_kv"] if pid == 0 else st["pool_score"]
    nisa.dma_copy(dst=pool.ap(pattern=[[n, T6], [1, n]], vector_offset=st["ws"][0:T6, 0:1], indirect_dim=0),
                  src=pr[0:T6, :])
    barrier(st["pool_kv"])
    barrier(st["pool_score"])
    # 2. the groups' pooled latents (both cores): members' kv / score rows, softmax over members
    kvm, scm = [], []
    for m in range(ratio):
        kv = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=kv[0:ng, :], src=st["pool_kv"].ap(pattern=[[n, ng], [1, n]], vector_offset=st["ms"][m][0:ng, 0:1],
                                                             indirect_dim=0))
        sc = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc[0:ng, :], src=st["pool_score"].ap(pattern=[[n, ng], [1, n]],
                                                                vector_offset=st["ms"][m][0:ng, 0:1], indirect_dim=0))
        kvm.append(kv)
        scm.append(sc)
    mx = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mx[0:ng], data1=scm[0][0:ng], data2=scm[1][0:ng], op=nl.maximum)
    for m in range(2, ratio):
        nisa.tensor_tensor(dst=mx[0:ng], data1=mx[0:ng], data2=scm[m][0:ng], op=nl.maximum)
    es = []
    for m in range(ratio):
        d = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d[0:ng], data1=scm[m][0:ng], data2=mx[0:ng], op=nl.subtract)
        e = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=e[0:ng], op=nl.exp, data=d[0:ng])
        es.append(e)
    den = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=den[0:ng], data1=es[0][0:ng], data2=es[1][0:ng], op=nl.add)
    for m in range(2, ratio):
        nisa.tensor_tensor(dst=den[0:ng], data1=den[0:ng], data2=es[m][0:ng], op=nl.add)
    rz = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rz[0:ng], data=den[0:ng])
    # rz2 = rz (2 - den rz)
    t = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t[0:ng], data1=den[0:ng], data2=rz[0:ng], op=nl.multiply)
    nisa.tensor_scalar(dst=t[0:ng], data=t[0:ng], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=2.0)
    rz2 = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rz2[0:ng], data1=rz[0:ng], data2=t[0:ng], op=nl.multiply)
    acc = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
    for m in range(ratio):
        w = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=w[0:ng], data1=es[m][0:ng], data2=rz2[0:ng], op=nl.multiply)
        term = nl.ndarray((PT, n), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=term[0:ng] if m else acc[0:ng], data1=kvm[m][0:ng], data2=w[0:ng], op=nl.multiply)
        if m:
            nisa.tensor_tensor(dst=acc[0:ng], data1=acc[0:ng], data2=term[0:ng], op=nl.add)
    lat_raw = nl.ndarray((PT, n), dtype=BF, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lat_raw[0:ng], src=acc[0:ng])
    return _finish(pid, st, lat_raw, ng, n, barrier)


def _finish(pid, st, lat_raw, ng, n, barrier):
    # 3. the latent: rmsnorm, then (core 0) rope + fp4 into the compressed cache
    lat = _norm(lat_raw, ng, n, st["nw_lat"])
    kd = st["ix_wk"].shape[0]
    if pid == 0:
        lat_c = nl.ndarray((PT, n), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lat_c[0:ng], src=lat[0:ng])
        _rope_fp4_store(lat_c, ng, n, st["csg"], 16, True, st["ckv"], st["gw"])
    else:
        # 4. (core 1) the index keys: rmsnorm(lat @ ix_wk.T) (bf16 GEMM), rope + fp4 into the key cache
        nb = n // PT
        tl = nl.ndarray((PT, nb, PT), dtype=BF, buffer=nl.psum)
        tw = nl.ndarray((PT, nb, PT), dtype=BF, buffer=nl.psum)
        for b in range(nb):
            nisa.nc_transpose(dst=tl[:, b, 0:ng], data=lat[0:ng, b * PT : (b + 1) * PT])
            nisa.nc_transpose(dst=tw[:, b, 0:kd], data=st["wk_rows"][0:kd, b * PT : (b + 1) * PT])
        latT = nl.ndarray((PT, nb, PT), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=latT[:, :, 0:ng], src=tl[:, :, 0:ng])
        wkT = nl.ndarray((PT, nb, PT), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wkT[:, :, 0:kd], src=tw[:, :, 0:kd])
        kp = nl.ndarray((PT, kd), dtype=F32, buffer=nl.psum)
        for b in range(nb):
            nisa.nc_matmul(dst=kp[0:ng, :], stationary=latT[:, b, 0:ng], moving=wkT[:, b, 0:kd], accumulate=(b > 0))
        k16 = nl.ndarray((PT, kd), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k16[0:ng], src=kp[0:ng])
        kn = _norm(k16, ng, kd, st["nw_k"])
        _rope_fp4_store(kn, ng, kd, st["csg"], 32, False, st["ikc"], st["gw"])
    barrier(st["ckv"])
    barrier(st["ikc"])
