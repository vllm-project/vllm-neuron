# SPDX-License-Identifier: Apache-2.0
"""The stage boundary of the pipeline scheme (`DEEPSEEK_V41_SCHEME=tp8pp8`).

vLLM runs 8 pipeline stages of 8 tensor-parallel ranks; a stage owns 5 blocks and
reduces over its own ranks. `boundary_body` is the seam between consecutive stages,
an in-kernel collective like everything else in the port: ranks `8s + t` and
`8(s+1) + t` form a 2-rank replica group (32 such pairs cover the world, one
layout for the even-numbered sending stages and one for the odd), the sending
stage puts the hyper-connection stream, `pre_mix`, the current index top-k and the
rows this step wrote into the current compressed-KV / index-key caches into the
pair's sum, the receiving stage contributes zeros and takes the sum: its stream
for its own blocks, and the cache rows scattered into its replica of the source
layer's caches at the same slots.

**One NEFF for both sides.** The role is a runtime input (`role`: 1.0 on the
sender, 0.0 on the receiver) that scales the payload, not a trace-time constant:
the sender's and the receiver's kernels are the same compiled program, so the
collective they meet in is identical by construction. Role-specialized kernels
(sender / receiver / bystander, 2026-10-24) compiled to collectives with
different signatures at some shapes ("replica group signature mismatch for group
0: likely caused by mismatched collectives between peers" at the first 512-row
prefill), and the runtime refused the execution. The sender writes the sum (its
own payload) back into its outputs and its own cache rows, which is a no-op.

`flags = (kver,)`.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .mk_common import PT, bcast_row, scatter_rows, tiles_of


def _copy_rows(src_hbm, dst_hbm, tiles, n, dtype, chunk=4096):
    """`dst[rows] = src[rows]` for the row tiles, through SBUF."""
    chunk = min(chunk, n)
    for r0, nr in tiles:
        for c0 in range(0, n, chunk):
            w = min(chunk, n - c0)
            t = nl.ndarray((PT, w), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, c0 : c0 + w])
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, c0 : c0 + w], src=t[0:nr, :])


def _zero_rows(dst_hbm, tiles, n, dtype, chunk=4096):
    chunk = min(chunk, n)
    z = nl.ndarray((PT, chunk), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(z, 0)
    for r0, nr in tiles:
        for c0 in range(0, n, chunk):
            w = min(chunk, n - c0)
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, c0 : c0 + w], src=z[0:nr, 0:w])


def _scaled_rows(src_hbm, dst_hbm, tiles, n, dtype, w, chunk=4096):
    """`dst[rows] = src[rows] * w` (`w` a `[PT, 1]` fp32 column of 0 / 1) for the row tiles."""
    chunk = min(chunk, n)
    for r0, nr in tiles:
        for c0 in range(0, n, chunk):
            wd = min(chunk, n - c0)
            t = nl.ndarray((PT, wd), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, c0 : c0 + wd])
            nisa.tensor_scalar(dst=t[0:nr, :], data=t[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, c0 : c0 + wd], src=t[0:nr, :])


def _gather_scaled_rows(cache_hbm, slots_hbm, dst_hbm, tiles, n, w):
    """`dst[r] = cache[slots[r]] * w` (bf16) for the row tiles."""
    for r0, nr in tiles:
        ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
        t = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=cache_hbm.ap(pattern=[[n, nr], [1, n]],
                                                        vector_offset=ix[0:nr, 0:1], indirect_dim=0))
        nisa.tensor_scalar(dst=t[0:nr, :], data=t[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=t[0:nr, :])


def boundary_body(stream, pre_mix, topk, cmp_cache, idx_cache, slots, role, groups, flags):
    """One stage boundary; see the module doc. `cmp_cache[rows, hd]` / `idx_cache[rows,
    kd]` are this rank's caches of the current source layer (owned on the sender,
    the replica on the receiver; aliased and written through), `slots[R, 1]` the
    cache slots this step wrote (a 1-row dummy when there is no compressed source
    yet), `role[1, 1]` fp32 1.0 on the sending stage and 0.0 on the receiving one."""
    (kver,) = flags
    T, hc, dim = stream.shape
    width = hc * dim
    nt = topk.shape[1]
    R = slots.shape[0]
    hd = cmp_cache.shape[1]
    kd = idx_cache.shape[1]
    have_rows = R > 1 or hd > 1
    pid = nl.program_id(0)
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    rows_t = tiles_of(T, pid)
    rows_r = tiles_of(R, pid)
    w = bcast_row(role[0:1, 0:1], 1, PT)

    out_stream = nl.ndarray((T, hc, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_mix = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    out_topk = nl.ndarray((T, nt), dtype=nl.int32, buffer=nl.shared_hbm)
    s2 = stream.reshape((T, width))
    o2 = out_stream.reshape((T, width))
    # payloads: bf16 stream, fp32 [pre_mix | topk], bf16 cache rows [cmp | idx]
    cs_src = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cs_src")
    cs_dst = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cs_dst")
    cf_src = nl.ndarray((T, hc + nt), dtype=nl.float32, buffer=nl.shared_hbm, name="cf_src")
    cf_dst = nl.ndarray((T, hc + nt), dtype=nl.float32, buffer=nl.shared_hbm, name="cf_dst")
    cr_src = nl.ndarray((R, hd + kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cr_src")
    cr_dst = nl.ndarray((R, hd + kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cr_dst")

    _scaled_rows(s2, cs_src, rows_t, width, nl.bfloat16, w)
    for r0, nr in rows_t:
        m = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])
        nisa.tensor_scalar(dst=m[0:nr, :], data=m[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
        nisa.dma_copy(dst=cf_src[r0 : r0 + nr, 0:hc], src=m[0:nr, :])
        ti = nl.ndarray((PT, nt), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti[0:nr, :], src=topk[r0 : r0 + nr, 0:nt])
        tf = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf[0:nr, :], src=ti[0:nr, :])
        nisa.tensor_scalar(dst=tf[0:nr, :], data=tf[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
        nisa.dma_copy(dst=cf_src[r0 : r0 + nr, hc : hc + nt], src=tf[0:nr, :])
    if have_rows:
        _gather_scaled_rows(cmp_cache, slots, cr_src[0:R, 0:hd], rows_r, hd, w)
        _gather_scaled_rows(idx_cache, slots, cr_src[0:R, hd : hd + kd], rows_r, kd, w)
    else:
        _zero_rows(cr_src, rows_r, hd + kd, nl.bfloat16)
    nisa.core_barrier(data=cs_src, cores=(0, 1))
    nisa.core_barrier(data=cf_src, cores=(0, 1))
    nisa.core_barrier(data=cr_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cs_src], dsts=[cs_dst], replica_group=rg, op=nl.add)
    ncc.all_reduce(srcs=[cf_src], dsts=[cf_dst], replica_group=rg, op=nl.add)
    ncc.all_reduce(srcs=[cr_src], dsts=[cr_dst], replica_group=rg, op=nl.add)

    # Both sides take the sum: the sender gets its own payload back.
    _copy_rows(cs_dst, o2, rows_t, width, nl.bfloat16)
    for r0, nr in rows_t:
        m = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m[0:nr, :], src=cf_dst[r0 : r0 + nr, 0:hc])
        nisa.dma_copy(dst=out_mix[r0 : r0 + nr, 0:hc], src=m[0:nr, :])
        tf = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tf[0:nr, :], src=cf_dst[r0 : r0 + nr, hc : hc + nt])
        ti = nl.ndarray((PT, nt), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ti[0:nr, :], src=tf[0:nr, :])
        nisa.dma_copy(dst=out_topk[r0 : r0 + nr, 0:nt], src=ti[0:nr, :])
    if have_rows:
        scatter_rows(cr_dst[0:R, 0:hd], slots, cmp_cache, rows_r, hd)
        scatter_rows(cr_dst[0:R, hd : hd + kd], slots, idx_cache, rows_r, kd)
    nisa.core_barrier(data=out_stream, cores=(0, 1))
    nisa.core_barrier(data=out_mix, cores=(0, 1))
    nisa.core_barrier(data=out_topk, cores=(0, 1))
    nisa.core_barrier(data=cmp_cache, cores=(0, 1))
    nisa.core_barrier(data=idx_cache, cores=(0, 1))
    return out_stream, out_mix, out_topk, cmp_cache, idx_cache
