# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENTAL block-sparse decode (HY3_SPARSE_DECODE) helpers, on CPU."""

import torch
from hy_v3_neuron.model import (
    _SUMMARY_EMPTY,
    sparse_budget_blocks,
    sparse_select_blocks,
    update_k_summary_decode,
    update_k_summary_prefill,
)

BS, GROUP, D, KVH, NBLK = 32, 4, 16, 1, 128
L = BS * GROUP  # tokens per group


def _empty():
    k_sum = torch.empty(NBLK, 2, KVH, D, dtype=torch.bfloat16)
    k_sum[:, 0] = _SUMMARY_EMPTY
    k_sum[:, 1] = -_SUMMARY_EMPTY
    return k_sum


def _table(n_blocks, g):
    return torch.randperm(NBLK - 1, generator=g)[:n_blocks] + 1  # block 0 is the null block


def _slots(table_row, positions):
    return torch.tensor([int(table_row[p // BS]) * BS + p % BS for p in positions])


def _expected(keys, table_row, n):
    """Per-group min / max of the keys of positions < n, by the group's first block."""
    out = {}
    for gi in range(-(-n // L)):
        ks = torch.stack([keys[p] for p in range(gi * L, min(n, (gi + 1) * L))])
        out[int(table_row[gi * GROUP])] = (ks.amin(0), ks.amax(0))
    return out


def _check(k_sum, exp):
    for blk, (mn, mx) in exp.items():
        torch.testing.assert_close(k_sum[blk, 0].float(), mn.to(torch.bfloat16).float())
        torch.testing.assert_close(k_sum[blk, 1].float(), mx.to(torch.bfloat16).float())


def test_budget_is_a_multiple_of_256_and_capped():
    assert sparse_budget_blocks(2688, 0.125, 4, 8) == 512  # 336 + 12 -> 512
    assert sparse_budget_blocks(1024, 0.1, 4, 8) == 256
    assert sparse_budget_blocks(200, 0.5, 4, 8) == 200  # not smaller: dense


def _prefill(k_sum, table, keys, start, n, T, g):
    k = torch.randn(KVH, T, D, generator=g)
    pos = list(range(start, start + n))
    for i, p in enumerate(pos):  # a re-computed token has the same K
        if p in keys:
            k[:, i] = keys[p]
    slots = torch.full((T,), -1)
    slots[:n] = _slots(table, pos)
    for i, p in enumerate(pos):
        keys[p] = k[:, i]
    update_k_summary_prefill(k_sum, k, slots, table, BS, GROUP)


def test_prefill_summaries_match_the_written_keys():
    g = torch.Generator().manual_seed(0)
    table = _table(12, g)
    keys = {}
    k_sum = _empty()
    # A stale summary on a reused block must not survive a fresh write.
    k_sum[table[4], 0] = -100.0
    k_sum[table[4], 1] = 100.0
    _prefill(k_sum, table, keys, 0, 160, 256, g)  # group 0 full, group 1 partial, padding after
    _prefill(k_sum, table, keys, 160, 200, 256, g)  # starts inside group 1 (a 5-block prefix hit)
    _prefill(k_sum, table, keys, 359, 1, 128, g)  # the re-computed last token
    _check(k_sum, _expected(keys, table, 360))


def test_decode_steps_fold_into_the_summaries():
    g = torch.Generator().manual_seed(1)
    table = _table(12, g).view(1, 12)
    keys = {}
    k_sum = _empty()
    k_sum[table[0, 8], 0] = -100.0  # stale: group 2 starts in decode, fresh
    k_sum[table[0, 8], 1] = 100.0
    _prefill(k_sum, table[0], keys, 0, 250, 256, g)
    n, written, cache = 250, [], {}
    while n < 270:  # 4-token steps, 2 accepted each: crosses into group 2 at 256
        k = torch.randn(1, 4, KVH, D, generator=g)
        pos = torch.arange(n, n + 4)
        for i, p in enumerate(pos.tolist()):
            written.append((p, k[0, i]))
            cache[p] = k[0, i]  # the slot's current key
        slots = _slots(table[0], pos.tolist())
        update_k_summary_decode(k_sum, k, pos, slots, table, BS, GROUP)
        n += 2
    # Group 1 (no step starts it) holds every key written into it, rejected ones included.
    ks = torch.stack([keys[p] for p in range(L, 250)] + [x for p, x in written if p < 2 * L])
    _check(k_sum, {int(table[0, GROUP]): (ks.amin(0), ks.amax(0))})
    # Group 2 is restarted by the step at 256: a bound of the keys in the cache, no stale values.
    s = k_sum[int(table[0, 2 * GROUP])].float()
    now = torch.stack([x for p, x in cache.items() if p >= 2 * L]).to(torch.bfloat16).float()
    assert (s[0] <= now.amin(0)).all() and (s[1] >= now.amax(0)).all()
    assert s.abs().max() < 100


def test_padding_rows_write_only_the_null_block():
    g = torch.Generator().manual_seed(2)
    table = _table(8, g).view(1, 8)
    k_sum = _empty()
    before = k_sum.clone()
    k = torch.randn(1, 4, KVH, D, generator=g)
    update_k_summary_decode(k_sum, k, torch.zeros(4, dtype=torch.long), torch.full((4,), -1), table, BS, GROUP)
    assert torch.equal(k_sum[1:], before[1:])


def _summaries(n, n_blocks, g, hot=None):
    table = _table(n_blocks, g).view(1, n_blocks)
    k = torch.randn(KVH, n_blocks * BS, D, generator=g)
    if hot is not None:
        k[0, hot[0]] = hot[1]
    k_sum = _empty()
    slots = torch.full((n_blocks * BS,), -1)
    slots[:n] = _slots(table[0], range(n))
    update_k_summary_prefill(k_sum, k, slots, table[0], BS, GROUP)
    return table, k, k_sum


def _groups(compact, table):
    """The table's groups as group indices of the request's table."""
    first = compact.view(-1, GROUP)[:, 0].tolist()
    row = table[0].tolist()
    return [row.index(b) // GROUP for b in first]


def test_bound_is_an_upper_bound_and_the_hot_group_is_kept():
    g = torch.Generator().manual_seed(3)
    n, nb = 1000, 40  # 8 history groups, the last one partial; 10 in the table
    q = torch.randn(2, 1, 1, D, generator=g)  # 2 query heads on the KV head
    table, k, k_sum = _summaries(n, nb, g, hot=(3 * L + 5, q[:, 0, 0].mean(0) * 20))
    for gi in range(-(-n // L)):
        toks = k[0, gi * L : min(n, (gi + 1) * L)]
        true = (q[:, 0, 0] @ toks.T).max()
        s = k_sum[int(table[0, gi * GROUP])].float()
        bound = (q[:, 0, 0].clamp(min=0) @ s[1, 0] + q[:, 0, 0].clamp(max=0) @ s[0, 0]).max()
        assert bound >= true - 1e-2
    # 5 groups in the table: sink, hot, the recent one before the last (any order),
    # then the last history group and the one after it
    compact, prior = sparse_select_blocks(q, k_sum, table, torch.tensor([n]), BS, GROUP, 20, 1, 2)
    nsv = -(-n // L)
    gs = _groups(compact[0], table)
    assert sorted(gs[:3]) == [0, 3, nsv - 2] and gs[3:] == [nsv - 1, nsv]
    # every kept group's blocks, in order within the group
    for i, gi in enumerate(gs):
        assert compact[0, i * GROUP : (i + 1) * GROUP].tolist() == table[0, gi * GROUP : (gi + 1) * GROUP].tolist()
    assert int(prior[0]) == 3 * L + (n - L * (nsv - 1))


def test_budget_covering_the_history_is_dense():
    g = torch.Generator().manual_seed(4)
    n = 300  # 3 groups
    table, _, k_sum = _summaries(n, 32, g)
    q = torch.randn(1, 1, 4, D, generator=g)  # one request, 4 verify tokens
    compact, prior = sparse_select_blocks(q, k_sum, table, torch.tensor([n]), BS, GROUP, 16, 1, 1)
    assert compact[0].tolist() == table[0, :16].tolist()  # the table, incl. the next group
    assert int(prior[0]) == n
    # 3 history groups also fit exactly in 4 table groups minus one
    compact, prior = sparse_select_blocks(q, k_sum, table, torch.tensor([3 * L]), BS, GROUP, 16, 1, 1)
    assert compact[0].tolist() == table[0, :16].tolist() and int(prior[0]) == 3 * L


def test_the_group_after_the_history_stays_in_the_table():
    """MTP verify tokens crossing a group boundary are read back through the table."""
    g = torch.Generator().manual_seed(5)
    n = 6 * L  # history ends exactly at a group boundary
    table, _, k_sum = _summaries(n, 32, g)
    q = torch.randn(1, 1, 4, D, generator=g)
    compact, prior = sparse_select_blocks(q, k_sum, table, torch.tensor([n]), BS, GROUP, 12, 1, 1)
    assert compact[0, 4:8].tolist() == table[0, 20:24].tolist()  # the last history group, kept
    assert compact[0, 8:].tolist() == table[0, 24:28].tolist()  # the next one
    assert int(prior[0]) == 2 * L


def test_empty_summaries_are_not_nan_and_rank_last():
    g = torch.Generator().manual_seed(7)
    n = 10 * L
    table, _, k_sum = _summaries(n, 48, g)
    # group 4 lost its summary (never happens with decode updates; must stay sane)
    k_sum[int(table[0, 4 * GROUP]), 0] = _SUMMARY_EMPTY
    k_sum[int(table[0, 4 * GROUP]), 1] = -_SUMMARY_EMPTY
    q = torch.randn(1, 1, 4, D, generator=g) * 100
    compact, _ = sparse_select_blocks(q, k_sum, table, torch.tensor([n]), BS, GROUP, 32, 1, 1)
    gs = _groups(compact[0], table)
    assert 4 not in gs[:6] and gs[6:] == [9, 10] and len(set(gs)) == 8


def test_padding_requests_get_no_history():
    g = torch.Generator().manual_seed(6)
    table, _, k_sum = _summaries(500, 32, g)
    table = torch.cat([table, torch.full((1, 32), -1)])
    q = torch.randn(1, 2, 1, D, generator=g)
    _, prior = sparse_select_blocks(q, k_sum, table, torch.tensor([500, 0]), BS, GROUP, 12, 1, 1)
    assert int(prior[1]) == 0
