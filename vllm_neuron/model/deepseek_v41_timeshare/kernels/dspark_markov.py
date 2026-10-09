"""`mtp.2.markov_head` on 4 ranks, cut along hidden (`FLOORPLAN.md`: `hidden x4 (allreduce)`). Rank
`r` holds hidden columns `[64r, 64r+64)` of both `(129280, 256)` tables and computes a partial
sum over its own quarter of the contraction for every vocabulary row; two pipelined
`ncc.all_reduce` calls rejoin them.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

PMAX = 128

WORLD_SIZE = 4

NCORE = 2


SCALAR_NUM = 20
SCALAR_DEN = 101

XWIDTH = 48
XWIDTH0 = 4

MMROWS = 8

RAMP0 = 8
RAMP = 2.0
RAMP_MAX = 160

CUTS = (248,)

EARLY_DMA = False

CC_HBM = False

HWDGE_W = True

HWDGE_OUT = True
HWDGE_IDX = True

HWDGE_ENGINE = "sync"


_hwdge_engine = nisa.engine.scalar if HWDGE_ENGINE == "scalar" else nisa.engine.sync


def _out_dma(dst, src):
    if HWDGE_OUT:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
    else:
        nisa.dma_copy(dst=dst, src=src)


def _blocks(rows_per_part):
    out = []
    left = rows_per_part
    size = RAMP0
    while left > 0:
        take = min(int(size), RAMP_MAX, left)
        if left - take < RAMP0:
            take = left
        out.append(take)
        left -= take
        size = size * RAMP
    return tuple(out)


def _chunk_widths(blocks):
    out = []
    i = 0
    for bw in blocks:
        cap = XWIDTH0 if i == 0 else XWIDTH
        n = (bw + cap - 1) // cap
        out.append((bw + n - 1) // n)
        i += 1
    return tuple(out)


def _needed_xb(widths):
    out = []
    high = 0
    for cw in widths:
        high = max(high, cw)
        out.append(high)
    return tuple(out)


def _impl(input_ids, embed_weight_shard, head_weight_shard, replica_group):
    embed_weight = embed_weight_shard
    head_weight = head_weight_shard

    vocab, dim = head_weight.shape
    n_tokens = 1
    for size in input_ids.shape:
        n_tokens *= size

    assert embed_weight.shape == head_weight.shape, (
        f"embed {embed_weight.shape} and head {head_weight.shape} must be the same shard"
    )
    assert vocab % (PMAX * NCORE) == 0, (
        f"PMAX * NCORE ({PMAX * NCORE}) must divide vocab ({vocab})"
    )
    assert dim % NCORE == 0, f"NCORE ({NCORE}) must divide the shard width ({dim})"
    assert 0 <= SCALAR_NUM <= SCALAR_DEN, "the Scalar share must be a fraction in [0, 1]"

    logits = nl.ndarray((n_tokens, vocab), dtype=nl.float32, buffer=nl.shared_hbm)
    embed = nl.ndarray((n_tokens, dim), dtype=embed_weight.dtype, buffer=nl.shared_hbm)

    core = nl.program_id(0)

    if HWDGE_IDX:
        idx = nl.ndarray((n_tokens, 1), dtype=input_ids.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=input_ids.reshape((n_tokens, 1)),
                      dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
    else:
        idx = nl.load(input_ids.reshape((n_tokens, 1)))

    rows_per_core = vocab // NCORE
    rows_per_part = rows_per_core // PMAX

    blocks = _blocks(rows_per_part)
    widths = _chunk_widths(blocks)
    need = _needed_xb(widths)

    WFULL = need[-1]
    rowr = nl.ndarray((n_tokens, WFULL, dim), dtype=embed_weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rowr,
        src=embed_weight.ap(
            pattern=[[dim, n_tokens], [0, WFULL], [1, dim]],
            offset=0,
            vector_offset=idx,
            indirect_dim=0,
        ),
    )

    ones = nl.ndarray((1, PMAX), dtype=embed_weight.dtype, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)

    bc0 = nl.ndarray((PMAX, need[0], dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bc0, stationary=ones, moving=rowr[:, 0:need[0], :],
                   is_stationary_onezero=True)

    bc = nl.ndarray((PMAX, WFULL, dim), dtype=nl.float32, buffer=nl.psum)
    k = 0
    while k < WFULL:
        m = min(k + MMROWS, WFULL)
        nisa.nc_matmul(dst=bc[:, k:m, :], stationary=ones, moving=rowr[:, k:m, :],
                       is_stationary_onezero=True)
        k = m

    w4 = head_weight.reshape((NCORE, PMAX, rows_per_part, dim))
    out3 = logits.reshape((NCORE, PMAX, rows_per_part))

    bounds = (0,) + tuple(CUTS) + (rows_per_part,)
    for _i in range(len(bounds) - 1):
        assert bounds[_i] < bounds[_i + 1], f"CUTS must be strictly increasing within the sweep"
    nchunk = len(bounds) - 1

    acc_sb = []
    cc_src = []
    cc_dst = []
    for _j in range(nchunk):
        _n = bounds[_j + 1] - bounds[_j]
        acc_sb.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"acc_{_j}"))
        if CC_HBM:
            cc_src.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.private_hbm))
            cc_dst.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.private_hbm))
        else:
            cc_src.append(acc_sb[_j])
            cc_dst.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"cc_dst_{_j}"))

    bi = 0
    r0 = 0
    while bi < len(blocks):
        bw = blocks[bi]
        cw = widths[bi]
        r1 = r0 + bw
        if HWDGE_W:
            w = nl.ndarray((PMAX, bw, dim), dtype=head_weight.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=w4[core, :, r0:r1, :],
                          dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
        else:
            w = nl.load(w4[core, :, r0:r1, :])

        nscalar = (bw * SCALAR_NUM) // SCALAR_DEN
        split = bw - nscalar

        sink = nl.ndarray((PMAX, dim), dtype=nl.float8_e4m3, buffer=nl.sbuf)

        c0 = 0
        while c0 < bw:
            j = 0
            while bounds[j + 1] <= r0 + c0:
                j += 1
            base = bounds[j]
            acc = acc_sb[j]
            c1 = min(c0 + cw, bw, bounds[j + 1] - r0)

            prod = nl.ndarray((PMAX, c1 - c0, dim), dtype=nl.float32, buffer=nl.sbuf)
            if bi == 0:
                nisa.tensor_tensor(dst=prod, data1=w[:, c0:c1, :],
                                   data2=bc0[:, 0:c1 - c0, :], op=nl.multiply)
            else:
                nisa.tensor_tensor(dst=prod, data1=w[:, c0:c1, :],
                                   data2=bc[:, 0:c1 - c0, :], op=nl.multiply)

            v1 = min(c1, split)
            if v1 > c0:
                nisa.tensor_reduce(dst=acc[:, r0 + c0 - base:r0 + v1 - base], op=nl.add,
                                   data=prod[:, 0:v1 - c0, :], axis=(2,))

            b = max(c0, split)
            while b < c1:
                nisa.activation(dst=sink, op=nl.copy, data=prod[:, b - c0, :],
                                reduce_op=nl.add, reduce_res=acc[:, r0 + b - base:r0 + b - base + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                b += 1

            if CC_HBM:
                nisa.dma_copy(dst=cc_src[j][:, r0 + c0 - base:r0 + c1 - base],
                              src=acc[:, r0 + c0 - base:r0 + c1 - base])

            c0 = c1

            if j + 1 < nchunk and r0 + c1 == bounds[j + 1]:
                ncc.all_reduce(srcs=[cc_src[j]], dsts=[cc_dst[j]],
                               replica_group=replica_group, op=nl.add)
                if EARLY_DMA:
                    _out_dma(dst=out3[core, :, base:bounds[j + 1]], src=cc_dst[j])

        r0 = r1
        bi += 1

    ncc.all_reduce(srcs=[cc_src[nchunk - 1]], dsts=[cc_dst[nchunk - 1]],
                   replica_group=replica_group, op=nl.add)

    for _j in range(nchunk):
        if _j + 1 < nchunk and EARLY_DMA:
            continue
        _out_dma(dst=out3[core, :, bounds[_j]:bounds[_j + 1]], src=cc_dst[_j])

    half = dim // NCORE
    nl.store(embed[0:n_tokens, core * half:core * half + half],
             value=rowr[:, 0, core * half:core * half + half])

    return logits, embed


kernel = nki.jit(_impl)
