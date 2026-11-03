# SPDX-License-Identifier: Apache-2.0
"""A decode step's prologue on the device: every per-step input the fused decode launches
read that the host used to compute (`mk_model.StepIndices`), from the step's block tables,
its (optimistic) span starts and the previous step's rejection-sampler output.

    step_prologue(pmeta, prev, ids, history, tokmap, prod, hcol, hinv, <outputs...>, flags)

What it writes, in place, into the step's persistent buffers (`mk_model.StepPrologue`):

- per `(ratio, window)` combination the model's layers use (`flags` ratios, in order): the
  `mk_host.AttnIndices` columns `row_pos`, `write_slots`, `win_slots`, `wv`, and for a
  compressed ratio also `cmp_slots`, `clen`, `grp_pos`, `grp_mslots`, `grp_wslots` -- the
  values `mk_host.build_attn_indices` defines (`twin` / `cbase` / ratio 0's `cmp_slots` are
  shape constants the host writes once);
- the two engram layers' `[T, 24]` n-gram row ids, local to this rank's shard (what
  `NgramHashState.hash_spans` and `StepIndices.hash_cols` compute), from a per-slot history
  of compressed token ids kept on the device (`history`, updated here with the step's own
  tokens before any is read, as `hash_spans` does);
- the step's `[T, 1]` int32 token column the fused embedding reads, and the spans'
  corrected start positions `[B, 1]` (the drafter's `pos`).

Positions: span `b` starts at `pmeta` start `- rejected_b`, `rejected_b = K + 1 -
#{j : 0 <= prev[b, j] < vocab}` (vLLM's async-scheduling correction of an optimistic
start, `vllm_neuron.functional.correct_spec_decode_positions_and_slot_mapping`); a
synchronous step passes `prev` all zeros, which corrects nothing.

The n-gram hash is 64-bit integer arithmetic (`NgramHashState`): a compressed id times an
odd multiplier, XOR-rolled over the n-gram, reduced modulo a prime below 2^24. The
products come from a table (`prod[c]`: every layer's and shift's `c * multiplier` as
(hi, lo) int32 words), the XOR is bitwise on the words, and the reduction is Horner's rule
over 7-bit chunks in exact GpSimd int32 arithmetic (`r = (r << 7 | chunk) mod p`, the
quotient from an fp32 estimate, corrected by one `p` either way).

Indices are int32 throughout (positions < 2^14, slots < 2^17). Core 0 does all of it.

`pmeta` is `[B * (H + Wp), 1]` int32: span `b`'s row is `H` header columns (`C_*`) then its
block table, clamped at 0 and zero-padded to `Wp` entries (a padded span's table is all
zeros: block 0, which no sequence owns)."""

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128
GP = nisa.engine.unknown
#: `pmeta` header columns of a span
C_START, C_LIVE, C_TBASE, C_BL, C_BG2, C_BNJ2, C_BNJ1 = range(7)
H = 8
#: the n-gram hash: layers x n-gram sizes x heads columns, 7-bit Horner chunks
HASH_LAYERS, NGRAMS, HEADS = 2, 3, 8
HCOLS = HASH_LAYERS * NGRAMS * HEADS
SHIFTS = NGRAMS + 1
DEAD = -1


def _log2(n):
    k = 0
    while (1 << k) < n:
        k += 1
    assert (1 << k) == n, f"{n} is not a power of two"
    return k


def _i(shape):
    return nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)


def _f(shape):
    return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)


def _ts(dst, data, op0, a0, op1=None, a1=None):
    nisa.tensor_scalar(dst=dst, data=data, op0=op0, operand0=a0, op1=op1, operand1=a1, engine=GP)


def _tt(dst, a, b, op):
    nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=GP)


def _colf(col, np_):
    """An int32 `(np_, 1)` column as fp32: a per-partition operand of an arithmetic
    `tensor_scalar` must be fp32 (exact here, every value is below 2^24)."""
    f = _f((PT, 1))
    nisa.tensor_copy(dst=f[0:np_, :], src=col[0:np_, :])
    return f


def _tsc(dst, data, op, col, np_):
    """`data <op> col` with the `(np_, 1)` int32 column broadcast along the free axis."""
    nisa.tensor_scalar(dst=dst, data=data, op0=op, operand0=_colf(col, np_)[0:np_, 0:1])


def _span_cols(pm, prev, B, n, q0, np_, Hw, K1):
    """Rows `q0 .. q0 + np_` of a layout with `n` rows per span (row `q` is span `q // n`'s):
    each row's span header `(np_, H)` and previous-step rejection row `(np_, K1)`."""
    hdr = _i((PT, H))
    pv = _i((PT, K1))
    for b in range(B):
        lo, hi = max(b * n, q0), min((b + 1) * n, q0 + np_)
        if lo < hi:
            nisa.dma_copy(dst=hdr[lo - q0 : hi - q0, 0:H], src=pm.ap(pattern=[[0, hi - lo], [1, H]], offset=b * Hw))
            nisa.dma_copy(dst=pv[lo - q0 : hi - q0, 0:K1],
                          src=prev.ap(pattern=[[0, hi - lo], [1, K1]], offset=b * K1))
    return hdr, pv


def _starts(hdr, pv, np_, K1, vocab):
    """`(np_, 1)` corrected start of each row's span: the header's start minus the previous
    step's rejected drafts."""
    ge = _i((PT, K1))
    _ts(ge[0:np_, :], pv[0:np_, :], nl.greater_equal, 0)
    lt = _i((PT, K1))
    _ts(lt[0:np_, :], pv[0:np_, :], nl.less, vocab)
    ok = _i((PT, K1))
    _tt(ok[0:np_, :], ge[0:np_, :], lt[0:np_, :], nl.multiply)
    cnt = _i((PT, 1))
    nisa.tensor_reduce(dst=cnt[0:np_, :], op=nl.add, data=ok[0:np_, :], axis=(1,), keepdims=True)
    st = _i((PT, 1))
    _tt(st[0:np_, :], hdr[0:np_, C_START : C_START + 1], cnt[0:np_, :], nl.add)
    _ts(st[0:np_, :], st[0:np_, :], nl.add, -K1)
    return st


def _gather1(src, idx, np_):
    """`src_flat[idx[p]]` for the `np_` partitions (int32 `(np_, 1)`)."""
    out = _i((PT, 1))
    nisa.dma_copy(dst=out[0:np_, 0:1],
                  src=src.ap(pattern=[[1, np_], [1, 1]], vector_offset=idx[0:np_, 0:1], indirect_dim=0))
    return out


def _slot(pm, tbase, pos, np_, bs, bs_log2):
    """The cache slot of position `pos` (`(np_, 1)`, >= 0) through the row's span table
    (`tbase` its offset in `pm`)."""
    blk = _i((PT, 1))
    _ts(blk[0:np_, :], pos[0:np_, :], nl.right_shift, bs_log2)
    idx = _i((PT, 1))
    _tt(idx[0:np_, :], blk[0:np_, :], tbase[0:np_, :], nl.add)
    tab = _gather1(pm, idx, np_)
    off = _i((PT, 1))
    _ts(off[0:np_, :], pos[0:np_, :], nl.bitwise_and, bs - 1)
    s = _i((PT, 1))
    _ts(s[0:np_, :], tab[0:np_, :], nl.multiply, bs)
    _tt(s[0:np_, :], s[0:np_, :], off[0:np_, :], nl.add)
    return s


def _col_out(dst, tile, np_, r0=0):
    nisa.dma_copy(dst=dst[r0 : r0 + np_, 0:1], src=tile[0:np_, 0:1])


def _hash_rows(T, row_pos, ws, hdr, ids_t, pm, history, tokmap, prod, hcol, hinv, pad_id, bs, bs_log2):
    """`(T, HCOLS)` int32: the rows' n-gram ids (layer-major, then n-gram size, then head),
    plus each column's offset minus this rank's shard start. Writes the rows' compressed
    ids into `history` at their write slots first."""
    c = _gather1(tokmap, ids_t, T)
    nisa.dma_copy(dst=history.ap(pattern=[[1, T], [1, 1]], vector_offset=ws[0:T, 0:1], indirect_dim=0),
                  src=c[0:T, 0:1])
    tbase = hdr[0:T, C_TBASE : C_TBASE + 1]
    toks = [c]
    blocked = _i((PT, 1))
    nisa.memset(blocked[0:T, :], 0)
    for s in range(1, SHIFTS):
        ps = _i((PT, 1))
        _ts(ps[0:T, :], row_pos[0:T, :], nl.add, -s)
        neg = _i((PT, 1))
        _ts(neg[0:T, :], ps[0:T, :], nl.less, 0)
        _ts(ps[0:T, :], ps[0:T, :], nl.maximum, 0)
        sl = _slot(pm, tbase, ps, T, bs, bs_log2)
        src = _gather1(history, sl, T)
        dead = _i((PT, 1))
        _ts(dead[0:T, :], src[0:T, :], nl.equal, DEAD)
        _tt(blocked[0:T, :], blocked[0:T, :], neg[0:T, :], nl.maximum)
        _tt(blocked[0:T, :], blocked[0:T, :], dead[0:T, :], nl.maximum)
        # tok = blocked ? pad : src  ==  src + blocked * (pad - src)
        d = _i((PT, 1))
        _ts(d[0:T, :], src[0:T, :], nl.multiply, -1, nl.add, pad_id)
        _tt(d[0:T, :], d[0:T, :], blocked[0:T, :], nl.multiply)
        tok = _i((PT, 1))
        _tt(tok[0:T, :], src[0:T, :], d[0:T, :], nl.add)
        toks.append(tok)
    # the products of every shift's token: prod[c] = [layer][shift][hi, lo]
    pw = 2 * SHIFTS
    prods = []
    for s in range(SHIFTS):
        pr = _i((PT, HASH_LAYERS * pw))
        nisa.dma_copy(dst=pr[0:T, :], src=prod.ap(pattern=[[HASH_LAYERS * pw, T], [1, HASH_LAYERS * pw]],
                                                   vector_offset=toks[s][0:T, 0:1], indirect_dim=0))
        prods.append(pr)
    # the rolled (hi, lo) of every (layer, n-gram) broadcast over the heads: (T, HCOLS)
    vhi = _i((PT, HCOLS))
    vlo = _i((PT, HCOLS))
    for l in range(HASH_LAYERS):
        rh = _i((PT, 1))
        rl = _i((PT, 1))
        b0 = l * pw
        nisa.tensor_copy(dst=rh[0:T, :], src=prods[0][0:T, b0 : b0 + 1], engine=GP)
        nisa.tensor_copy(dst=rl[0:T, :], src=prods[0][0:T, b0 + 1 : b0 + 2], engine=GP)
        for n in range(NGRAMS):
            s = n + 1
            _tt(rh[0:T, :], rh[0:T, :], prods[s][0:T, b0 + 2 * s : b0 + 2 * s + 1], nl.bitwise_xor)
            _tt(rl[0:T, :], rl[0:T, :], prods[s][0:T, b0 + 2 * s + 1 : b0 + 2 * s + 2], nl.bitwise_xor)
            c0 = (l * NGRAMS + n) * HEADS
            for h in range(HEADS):
                nisa.tensor_copy(dst=vhi[0:T, c0 + h : c0 + h + 1], src=rh[0:T, :], engine=GP)
                nisa.tensor_copy(dst=vlo[0:T, c0 + h : c0 + h + 1], src=rl[0:T, :], engine=GP)
    primes = nl.ndarray((PT, HCOLS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=primes[0:T, :], src=hcol.ap(pattern=[[0, T], [1, HCOLS]], offset=0))
    hoff = nl.ndarray((PT, HCOLS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hoff[0:T, :], src=hcol.ap(pattern=[[0, T], [1, HCOLS]], offset=HCOLS))
    inv = _f((PT, HCOLS))
    nisa.dma_copy(dst=inv[0:T, :], src=hinv.ap(pattern=[[0, T], [1, HCOLS]], offset=0))

    def chunk(k):
        """bits [62 - 7k .. 56 - 7k] of the 63-bit value."""
        ch = _i((PT, HCOLS))
        hi_bit = 62 - 7 * k
        lo_bit = hi_bit - 6
        if lo_bit >= 32:
            _ts(ch[0:T, :], vhi[0:T, :], nl.right_shift, lo_bit - 32, nl.bitwise_and, 127)
        elif hi_bit < 32:
            _ts(ch[0:T, :], vlo[0:T, :], nl.right_shift, lo_bit, nl.bitwise_and, 127)
        else:            # straddles the words: hi's low bits above lo's top bits
            nhi = hi_bit - 31
            a = _i((PT, HCOLS))
            _ts(a[0:T, :], vhi[0:T, :], nl.bitwise_and, (1 << nhi) - 1, nl.left_shift, 32 - lo_bit)
            _ts(ch[0:T, :], vlo[0:T, :], nl.right_shift, lo_bit, nl.bitwise_and, (1 << (32 - lo_bit)) - 1)
            _tt(ch[0:T, :], ch[0:T, :], a[0:T, :], nl.bitwise_or)
        return ch

    r = chunk(0)
    for k in range(1, 9):
        ch = chunk(k)
        x = _i((PT, HCOLS))
        _ts(x[0:T, :], r[0:T, :], nl.left_shift, 7)
        _tt(x[0:T, :], x[0:T, :], ch[0:T, :], nl.bitwise_or)
        xf = _f((PT, HCOLS))
        nisa.tensor_copy(dst=xf[0:T, :], src=x[0:T, :])
        nisa.tensor_tensor(dst=xf[0:T, :], data1=xf[0:T, :], data2=inv[0:T, :], op=nl.multiply)
        q = _i((PT, HCOLS))
        nisa.tensor_copy(dst=q[0:T, :], src=xf[0:T, :])
        _tt(q[0:T, :], q[0:T, :], primes[0:T, :], nl.multiply)
        r = _i((PT, HCOLS))
        _tt(r[0:T, :], x[0:T, :], q[0:T, :], nl.subtract)
        # r in [-p, 2p): one p up or down
        m = _i((PT, HCOLS))
        _ts(m[0:T, :], r[0:T, :], nl.less, 0)
        _tt(m[0:T, :], m[0:T, :], primes[0:T, :], nl.multiply)
        _tt(r[0:T, :], r[0:T, :], m[0:T, :], nl.add)
        m2 = _i((PT, HCOLS))
        _tt(m2[0:T, :], r[0:T, :], primes[0:T, :], nl.greater_equal)
        _tt(m2[0:T, :], m2[0:T, :], primes[0:T, :], nl.multiply)
        _tt(r[0:T, :], r[0:T, :], m2[0:T, :], nl.subtract)
    _tt(r[0:T, :], r[0:T, :], hoff[0:T, :], nl.add)
    return r


def step_prologue_body(
    pmeta, prev, ids, history, tokmap, prod, hcol, hinv,
    a_row_pos, a_write_slots, a_win_slots, a_wv,
    b_row_pos, b_write_slots, b_win_slots, b_wv, b_cmp_slots, b_clen, b_grp_pos, b_grp_mslots, b_grp_wslots,
    c_row_pos, c_write_slots, c_win_slots, c_wv, c_cmp_slots, c_clen, c_grp_pos, c_grp_mslots, c_grp_wslots,
    hash0, hash1, embed_ids, starts, flags,
):
    """See the module docstring. `flags` = (B, L, K, Wp, block_size, window, vocab, pad_id,
    ratio_a, ratio_b, ratio_c, read_len_b, read_len_c, kernel_version): combination `a` is
    uncompressed (ratio 0), `b` / `c` compressed."""
    B, L, K, Wp, bs, win, vocab, pad_id, ra, rb, rc, rlen_b, rlen_c, _kver = flags
    assert ra == 0 and rb in (1, 2) and rc in (1, 2)
    T = B * L
    assert T <= PT, f"a decode step of {T} rows"
    K1 = K + 1
    Hw = H + Wp
    bs_log2 = _log2(bs)
    nwin = win + L - 1
    pid = nl.program_id(0)
    done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if pid == 0:
        # ---------------------------------------------------------------- the rows
        hdr, pv = _span_cols(pmeta, prev, B, L, 0, T, Hw, K1)
        st = _starts(hdr, pv, T, K1, vocab)
        rr = _i((PT, 1))
        nisa.iota(dst=rr[0:T, :], pattern=[[1, 1]], offset=0, channel_multiplier=1)
        tt_ = _i((PT, 1))
        _tt(tt_[0:T, :], rr[0:T, :], hdr[0:T, C_BL : C_BL + 1], nl.subtract)
        row_pos = _i((PT, 1))
        _tt(row_pos[0:T, :], tt_[0:T, :], st[0:T, :], nl.add)
        tbase = hdr[0:T, C_TBASE : C_TBASE + 1]
        slot_r = _slot(pmeta, tbase, row_pos, T, bs, bs_log2)
        ws = _i((PT, 1))
        _tt(ws[0:T, :], slot_r[0:T, :], hdr[0:T, C_LIVE : C_LIVE + 1], nl.multiply)
        wv = _i((PT, 1))
        _ts(wv[0:T, :], row_pos[0:T, :], nl.add, -(win - 1))
        for dst_rp, dst_ws, dst_wv in ((c_row_pos, c_write_slots, c_wv), (b_row_pos, b_write_slots, b_wv),
                                       (a_row_pos, a_write_slots, a_wv)):
            _col_out(dst_rp, row_pos, T)
            _col_out(dst_ws, ws, T)
            _col_out(dst_wv, wv, T)
        for ratio, dst_clen in ((rb, b_clen), (rc, c_clen)):
            cl = _i((PT, 1))
            _ts(cl[0:T, :], row_pos[0:T, :], nl.add, 1)
            if ratio == 2:
                _ts(cl[0:T, :], cl[0:T, :], nl.right_shift, 1)
            _col_out(dst_clen, cl, T)
        # ratio 1's groups are the rows (`grp_pos` the positions, `grp_wslots` the write
        # slots, `grp_mslots` the rows' own slots); ratio 2's below
        for ratio, d_gp, d_gm, d_gw in ((rb, b_grp_pos, b_grp_mslots, b_grp_wslots),
                                        (rc, c_grp_pos, c_grp_mslots, c_grp_wslots)):
            if ratio == 1:
                _col_out(d_gp, row_pos, T)
                _col_out(d_gm, slot_r, T)
                _col_out(d_gw, ws, T)
        # the token column the fused embedding reads, and the n-gram ids
        ids_raw = _i((PT, 1))
        nisa.dma_copy(dst=ids_raw[0:T, 0:1], src=ids.ap(pattern=[[1, T], [1, 1]], offset=0))
        # (a placeholder id -- the runner's -1 for a draft it has not got -- would be an
        # out-of-bounds gather here and in the embedding: token 0 instead)
        ids_t = _i((PT, 1))
        _ts(ids_t[0:T, :], ids_raw[0:T, :], nl.maximum, 0)
        _col_out(embed_ids, ids_t, T)
        hv = _hash_rows(T, row_pos, ws, hdr, ids_t, pmeta, history, tokmap, prod, hcol, hinv, pad_id, bs, bs_log2)
        half = HCOLS // 2
        nisa.dma_copy(dst=hash0[0:T, 0:half], src=hv[0:T, 0:half])
        nisa.dma_copy(dst=hash1[0:T, 0:half], src=hv[0:T, half:HCOLS])

        # ---------------------------------------------------------------- the spans' starts
        h1, p1 = _span_cols(pmeta, prev, B, 1, 0, B, Hw, K1)
        s1 = _starts(h1, p1, B, K1, vocab)
        _col_out(starts, s1, B)

        # ---------------------------------------------------------------- ratio 2's groups
        for ratio, d_gp, d_gm, d_gw in ((rb, b_grp_pos, b_grp_mslots, b_grp_wslots),
                                        (rc, c_grp_pos, c_grp_mslots, c_grp_wslots)):
            if ratio != 2:
                continue
            G = L // 2 + 1
            ng = B * G
            hg, pg = _span_cols(pmeta, prev, B, G, 0, ng, Hw, K1)
            sg = _starts(hg, pg, ng, K1, vocab)
            kk = _i((PT, 1))
            nisa.iota(dst=kk[0:ng, :], pattern=[[1, 1]], offset=0, channel_multiplier=1)
            _tt(kk[0:ng, :], kk[0:ng, :], hg[0:ng, C_BG2 : C_BG2 + 1], nl.subtract)
            g0 = _i((PT, 1))
            _ts(g0[0:ng, :], sg[0:ng, :], nl.right_shift, 1)
            g1 = _i((PT, 1))
            _ts(g1[0:ng, :], sg[0:ng, :], nl.add, L)
            _ts(g1[0:ng, :], g1[0:ng, :], nl.right_shift, 1)
            cnt = _i((PT, 1))
            _tt(cnt[0:ng, :], g1[0:ng, :], g0[0:ng, :], nl.subtract)
            valid = _i((PT, 1))
            _tt(valid[0:ng, :], kk[0:ng, :], cnt[0:ng, :], nl.less)
            first = _i((PT, 1))
            _tt(first[0:ng, :], g0[0:ng, :], kk[0:ng, :], nl.add)
            _ts(first[0:ng, :], first[0:ng, :], nl.multiply, 2)
            _tt(first[0:ng, :], first[0:ng, :], valid[0:ng, :], nl.multiply)
            tb = hg[0:ng, C_TBASE : C_TBASE + 1]
            s0 = _slot(pmeta, tb, first, ng, bs, bs_log2)
            _tt(s0[0:ng, :], s0[0:ng, :], valid[0:ng, :], nl.multiply)
            f1 = _i((PT, 1))
            _ts(f1[0:ng, :], first[0:ng, :], nl.add, 1)
            s1g = _slot(pmeta, tb, f1, ng, bs, bs_log2)
            _tt(s1g[0:ng, :], s1g[0:ng, :], valid[0:ng, :], nl.multiply)
            _col_out(d_gp, first, ng)
            _col_out(d_gw, s0, ng)
            gm = _i((PT, 2))
            nisa.tensor_copy(dst=gm[0:ng, 0:1], src=s0[0:ng, :], engine=GP)
            nisa.tensor_copy(dst=gm[0:ng, 1:2], src=s1g[0:ng, :], engine=GP)
            nisa.dma_copy(dst=d_gm.ap(pattern=[[2, ng], [1, 2]], offset=0), src=gm[0:ng, 0:2])

        # ---------------------------------------------------------------- the window slots
        # partition = window row i, free = span: slot(max(start - (win - 1) + i, 0))
        for i0 in range(0, nwin, PT):
            ni = min(PT, nwin - i0)
            sb = _i((PT, B))
            nisa.dma_copy(dst=sb[0:ni, :], src=pmeta.ap(pattern=[[0, ni], [Hw, B]], offset=C_START))
            tb = _i((PT, B))
            nisa.dma_copy(dst=tb[0:ni, :], src=pmeta.ap(pattern=[[0, ni], [Hw, B]], offset=C_TBASE))
            pvb = _i((PT, B, K1))
            nisa.dma_copy(dst=pvb[0:ni, :, :], src=prev.ap(pattern=[[0, ni], [K1, B], [1, K1]], offset=0))
            ge = _i((PT, B, K1))
            _ts(ge[0:ni, :, :], pvb[0:ni, :, :], nl.greater_equal, 0)
            lt = _i((PT, B, K1))
            _ts(lt[0:ni, :, :], pvb[0:ni, :, :], nl.less, vocab)
            _tt(ge[0:ni, :, :], ge[0:ni, :, :], lt[0:ni, :, :], nl.multiply)
            cnt = _i((PT, B))
            nisa.tensor_reduce(dst=cnt[0:ni, :], op=nl.add, data=ge[0:ni, :, :], axis=(2,))
            ip = _i((PT, B))
            nisa.iota(dst=ip[0:ni, :], pattern=[[0, B]], offset=i0 - (win - 1) - K1, channel_multiplier=1)
            wp = _i((PT, B))
            _tt(wp[0:ni, :], sb[0:ni, :], cnt[0:ni, :], nl.add)
            _tt(wp[0:ni, :], wp[0:ni, :], ip[0:ni, :], nl.add)
            _ts(wp[0:ni, :], wp[0:ni, :], nl.maximum, 0)
            blk = _i((PT, B))
            _ts(blk[0:ni, :], wp[0:ni, :], nl.right_shift, bs_log2)
            _tt(blk[0:ni, :], blk[0:ni, :], tb[0:ni, :], nl.add)
            tab = _i((PT, B))
            for b in range(B):
                nisa.dma_copy(dst=tab[0:ni, b : b + 1],
                              src=pmeta.ap(pattern=[[1, ni], [1, 1]], vector_offset=blk[0:ni, b : b + 1],
                                           indirect_dim=0))
            off = _i((PT, B))
            _ts(off[0:ni, :], wp[0:ni, :], nl.bitwise_and, bs - 1)
            sl = _i((PT, B))
            _ts(sl[0:ni, :], tab[0:ni, :], nl.multiply, bs)
            _tt(sl[0:ni, :], sl[0:ni, :], off[0:ni, :], nl.add)
            for dst in (a_win_slots, b_win_slots, c_win_slots):
                nisa.dma_copy(dst=dst.ap(pattern=[[1, ni], [nwin, B]], offset=i0), src=sl[0:ni, :])

        # ---------------------------------------------------------------- the compressed slots
        # rows q = b * nj + j: block j of span b's first `read_len * ratio` positions; entry
        # (q, k) is compressed row i = j * F + k at position i * ratio (F = bs / ratio),
        # or the span's last compressed row's slot from i = clen on. Row-major, the tile
        # is `cmp_slots[q * F + k]`.
        for ratio, rlen, dst, c_bnj in ((rb, rlen_b, b_cmp_slots, C_BNJ2 if rb == 2 else C_BNJ1),
                                        (rc, rlen_c, c_cmp_slots, C_BNJ2 if rc == 2 else C_BNJ1)):
            F = bs // ratio
            nj = rlen // F
            assert nj * F == rlen and nj <= Wp
            rows = B * nj
            for q0 in range(0, rows, PT):
                np_ = min(PT, rows - q0)
                hq, pq = _span_cols(pmeta, prev, B, nj, q0, np_, Hw, K1)
                sq = _starts(hq, pq, np_, K1, vocab)
                tabj = _i((PT, 1))
                for b in range(B):
                    lo, hi = max(b * nj, q0), min((b + 1) * nj, q0 + np_)
                    if lo < hi:
                        nisa.dma_copy(dst=tabj[lo - q0 : hi - q0, 0:1],
                                      src=pmeta.ap(pattern=[[1, hi - lo], [1, 1]], offset=b * Hw + H + lo - b * nj))
                cl = _i((PT, 1))
                _ts(cl[0:np_, :], sq[0:np_, :], nl.add, L)
                if ratio == 2:
                    _ts(cl[0:np_, :], cl[0:np_, :], nl.right_shift, 1)
                clm1 = _i((PT, 1))
                _ts(clm1[0:np_, :], cl[0:np_, :], nl.add, -1, nl.maximum, 0)
                jq = _i((PT, 1))
                nisa.iota(dst=jq[0:np_, :], pattern=[[1, 1]], offset=q0, channel_multiplier=1)
                _tt(jq[0:np_, :], jq[0:np_, :], hq[0:np_, c_bnj : c_bnj + 1], nl.subtract)
                _ts(jq[0:np_, :], jq[0:np_, :], nl.multiply, F)
                iv = _i((PT, F))
                nisa.iota(dst=iv[0:np_, :], pattern=[[1, F]], offset=0, channel_multiplier=0)
                _tsc(iv[0:np_, :], iv[0:np_, :], nl.add, jq, np_)
                mask = _i((PT, F))
                _tsc(mask[0:np_, :], iv[0:np_, :], nl.less, cl, np_)
                tb32 = _i((PT, 1))
                _ts(tb32[0:np_, :], tabj[0:np_, :], nl.multiply, bs)
                val = _i((PT, F))
                nisa.iota(dst=val[0:np_, :], pattern=[[ratio, F]], offset=0, channel_multiplier=0)
                _tsc(val[0:np_, :], val[0:np_, :], nl.add, tb32, np_)
                lp = _i((PT, 1))
                _ts(lp[0:np_, :], clm1[0:np_, :], nl.multiply, ratio)
                last = _slot(pmeta, hq[0:np_, C_TBASE : C_TBASE + 1], lp, np_, bs, bs_log2)
                # out = last + mask * (val - last)
                _tsc(val[0:np_, :], val[0:np_, :], nl.subtract, last, np_)
                _tt(val[0:np_, :], val[0:np_, :], mask[0:np_, :], nl.multiply)
                _tsc(val[0:np_, :], val[0:np_, :], nl.add, last, np_)
                nisa.dma_copy(dst=dst.ap(pattern=[[F, np_], [1, F]], offset=q0 * F), src=val[0:np_, :])

        z = _i((1, 1))
        nisa.memset(z, 0)
        nisa.dma_copy(dst=done[0:1, 0:1], src=z)
    return done


@nki.jit
def step_prologue(
    pmeta, prev, ids, history, tokmap, prod, hcol, hinv,
    a_row_pos, a_write_slots, a_win_slots, a_wv,
    b_row_pos, b_write_slots, b_win_slots, b_wv, b_cmp_slots, b_clen, b_grp_pos, b_grp_mslots, b_grp_wslots,
    c_row_pos, c_write_slots, c_win_slots, c_wv, c_cmp_slots, c_clen, c_grp_pos, c_grp_mslots, c_grp_wslots,
    hash0, hash1, embed_ids, starts, flags,
):
    return step_prologue_body(
        pmeta, prev, ids, history, tokmap, prod, hcol, hinv,
        a_row_pos, a_write_slots, a_win_slots, a_wv,
        b_row_pos, b_write_slots, b_win_slots, b_wv, b_cmp_slots, b_clen, b_grp_pos, b_grp_mslots, b_grp_wslots,
        c_row_pos, c_write_slots, c_win_slots, c_wv, c_cmp_slots, c_clen, c_grp_pos, c_grp_mslots, c_grp_wslots,
        hash0, hash1, embed_ids, starts, flags,
    )


def history_put_body(history, slots, cids, flags):
    """`history[slots[r]] = cids[r]` for the `[T, 1]` rows: a step whose n-gram ids the host
    computed (a prefill chunk) puts its tokens into the device history the prologue reads.
    `flags` = (kernel_version,)."""
    T = slots.shape[0]
    pid = nl.program_id(0)
    done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if pid == 0:
        for r0 in range(0, T, PT):
            nr = min(PT, T - r0)
            sl = _i((PT, 1))
            nisa.dma_copy(dst=sl[0:nr, 0:1], src=slots[r0 : r0 + nr, 0:1])
            cv = _i((PT, 1))
            nisa.dma_copy(dst=cv[0:nr, 0:1], src=cids[r0 : r0 + nr, 0:1])
            nisa.dma_copy(dst=history.ap(pattern=[[1, nr], [1, 1]], vector_offset=sl[0:nr, 0:1], indirect_dim=0),
                          src=cv[0:nr, 0:1])
        z = _i((1, 1))
        nisa.memset(z, 0)
        nisa.dma_copy(dst=done[0:1, 0:1], src=z)
    return done


@nki.jit
def history_put(history, slots, cids, flags):
    return history_put_body(history, slots, cids, flags)
