# SPDX-License-Identifier: Apache-2.0
"""NeuronModelRunner recurrent-state bookkeeping (CPU): Mamba "align" prefix-caching state
copies and table, the dedicated state-pool slots, and per-group kernel block sizes.

The methods are called on a bare ``object.__new__(NeuronModelRunner)`` with only the
attributes each one reads set by hand; the block tables are vLLM's real BlockTable.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.block_table import BlockTable  # noqa: E402


def _nmr():
    # Imported at test time, not at collection: importing the real vllm_neuron.functional
    # before test/unit/model/qwen3_5/test_model_smoke.py runs defeats that test's
    # sys.modules stubs (``import vllm_neuron.functional as NF`` resolves through the
    # already-imported parent package).
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner

BS = 4  # align-mode state block size (tokens per block)
MAX_BLOCKS = 8


# --------------------------------------------------------------------------- helpers


def _runner(**attrs):
    runner = object.__new__(_nmr().NeuronModelRunner)
    runner.device = torch.device("cpu")
    for k, v in attrs.items():
        setattr(runner, k, v)
    return runner


def _req(num_computed_tokens, *block_ids):
    """CachedRequestState stand-in: block_ids[g] are the KV-manager blocks of group g."""
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens, block_ids=tuple(block_ids)
    )


def _sched_out(num_scheduled, finished=(), preempted=None, resumed=()):
    return SimpleNamespace(
        num_scheduled_tokens=dict(num_scheduled),
        finished_req_ids=set(finished),
        preempted_req_ids=None if preempted is None else set(preempted),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
    )


def _align_runner(requests, req_ids, groups=(1,), state_idx=None):
    """Runner with align-mode groups ``groups`` (group 0 is attention); ``copies`` is the
    argument of the last _copy_state_blocks call."""
    runner = _runner(
        _mamba_align_groups=list(groups),
        _mamba_align_block_size=BS,
        _mamba_state_idx=dict(state_idx or {}),
        requests=requests,
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
    )
    runner.copies = []
    runner._copy_state_blocks = lambda c: setattr(runner, "copies", c)
    return runner


# ------------------------------------------------------------ _preprocess_mamba_align


def test_first_chunk_of_new_request_copies_nothing():
    # No computed tokens: no previous state (-1) to carry over. 8 tokens end in block 1.
    r = _align_runner({"a": _req(0, [10, 11, 12], [20, 21, 22])}, ["a"])
    r._preprocess_mamba_align(_sched_out({"a": 8}))
    assert r.copies == []
    assert r._mamba_state_idx == {"a": 1}


def test_prefill_chunk_crossing_blocks_copies_previous_running_block():
    # Block-aligned chunks (the scheduler aligns them in align mode): after tokens 0..7
    # the state lives in block 1 (= state at the 8-token boundary). The next chunk of 8
    # ends in block 3, so block 1 is copied to block 3 and block 1 keeps the
    # boundary state the KV manager caches.
    ids = [20, 21, 22, 23, 24]
    r = _align_runner({"a": _req(8, [0], ids)}, ["a"], state_idx={"a": 1})
    r._preprocess_mamba_align(_sched_out({"a": 8}))
    assert r.copies == [[(21, 23)]]
    assert r._mamba_state_idx == {"a": 3}

    # Unaligned last chunk: 16 + 3 tokens -> block 4.
    r.requests["a"].num_computed_tokens = 16
    r._preprocess_mamba_align(_sched_out({"a": 3}))
    assert r.copies[-1] == [(23, 24)]
    assert r._mamba_state_idx == {"a": 4}


def test_decode_copies_only_when_crossing_a_block_boundary():
    ids = [30, 31, 32]
    r = _align_runner({"a": _req(5, [0], ids)}, ["a"], state_idx={"a": 1})
    # positions 5, 6, 7 are still in block 1 (positions 4..7)
    for nc in (5, 6, 7):
        r.requests["a"].num_computed_tokens = nc
        r._preprocess_mamba_align(_sched_out({"a": 1}))
        assert r.copies == []
        assert r._mamba_state_idx["a"] == 1
    # position 8 opens block 2: block 1 keeps the state at the 8-token boundary.
    r.requests["a"].num_computed_tokens = 8
    r._preprocess_mamba_align(_sched_out({"a": 1}))
    assert r.copies == [[(31, 32)]]
    assert r._mamba_state_idx["a"] == 2


def test_new_request_resuming_from_cached_prefix_copies_cached_state():
    # 8 tokens hit the prefix cache: the cached state is in block 1 (blocks before it are
    # the null block in align mode). 5 new tokens end in block 3.
    r = _align_runner({"b": _req(8, [1, 2, 3, 4], [0, 41, 42, 43])}, ["b"])
    r._preprocess_mamba_align(_sched_out({"b": 5}))
    assert r.copies == [[(41, 43)]]
    assert r._mamba_state_idx == {"b": 3}


def test_cached_prefix_of_one_block_copies_into_next_block():
    # 4-token hit: the cached state is block 0's; 4 new tokens end in block 1.
    r = _align_runner({"b": _req(4, [1, 2], [51, 52])}, ["b"])
    r._preprocess_mamba_align(_sched_out({"b": 4}))
    assert r.copies == [[(51, 52)]]
    assert r._mamba_state_idx == {"b": 1}


def test_unscheduled_request_keeps_its_state_index():
    # A running request hidden by the Neuron scheduler this step (e.g. a decode held
    # back during another request's prefill) is not scheduled: no copy, index unchanged.
    reqs = {"a": _req(7, [0], [60, 61, 62]), "b": _req(0, [0], [70, 71])}
    r = _align_runner(reqs, ["a", "b"], state_idx={"a": 1})
    r._preprocess_mamba_align(_sched_out({"b": 4}))
    assert r.copies == []
    assert r._mamba_state_idx == {"a": 1, "b": 0}
    # Next step a decodes from position 7 -> still block 1 (its index is reused, not
    # recomputed from num_computed_tokens).
    r._preprocess_mamba_align(_sched_out({"a": 1}))
    assert r._mamba_state_idx["a"] == 1 and r.copies == []


def test_finished_preempted_resumed_drop_state_index():
    reqs = {"c": _req(12, [0], [80, 81, 82, 83, 84])}
    r = _align_runner(
        reqs, ["c"], state_idx={"c": 0, "gone": 2, "pre": 1}
    )  # "c" stale: wrong on purpose
    # "c" was resumed: its stale index must not be used; prev = (12-1)//4 = 2, and
    # 12 + 4 tokens end in block 3.
    r._preprocess_mamba_align(
        _sched_out({"c": 4}, finished={"gone"}, preempted={"pre"}, resumed={"c"})
    )
    assert r._mamba_state_idx == {"c": 3}
    assert r.copies == [[(82, 83)]]


def test_preempted_none_is_tolerated():
    r = _align_runner({"a": _req(0, [0], [1, 2])}, ["a"])
    r._preprocess_mamba_align(_sched_out({"a": 1}, preempted=None))
    assert r._mamba_state_idx == {"a": 0}


def test_copies_batched_per_group_in_batch_order():
    # Two align-mode groups (1 and 2) with different block ids for the same request.
    reqs = {
        "a": _req(4, [0], [100, 101], [200, 201]),
        "b": _req(8, [0], [110, 111, 112], [210, 211, 212]),
        "c": _req(2, [0], [120, 121], [220, 221]),  # stays in block 0
    }
    r = _align_runner(
        reqs, ["b", "c", "a"], groups=(1, 2), state_idx={"a": 0, "b": 1, "c": 0}
    )
    r._preprocess_mamba_align(_sched_out({"a": 1, "b": 1, "c": 1}))
    assert r.copies == [[(111, 112), (100, 101)], [(211, 212), (200, 201)]]


def test_no_align_groups_is_a_noop():
    r = _runner(_mamba_align_groups=[])
    r._preprocess_mamba_align(_sched_out({"a": 1}))  # reads nothing else
    r = _runner()
    r._preprocess_mamba_align(_sched_out({"a": 1}))


def test_copy_state_blocks_chunks_and_pads_with_null_block():
    # _copy_state_blocks runs the copy graph in fixed chunks of _STATE_COPY_CHUNK pairs,
    # padding with (0, 0) (null block onto itself). Run it eagerly on CPU tensors.
    chunk = _nmr()._STATE_COPY_CHUNK
    n_pages = 2 * chunk + 4
    g0 = torch.arange(n_pages, dtype=torch.float32).repeat_interleave(3).view(n_pages, 3)
    g1 = g0.clone() + 1000
    r = _runner(
        _state_copy_pages=[g0, g1],
        _state_copy_page_groups=[0, 1],
        _state_copy_fn=_nmr()._copy_state_pages,
    )
    pairs0 = [(i, chunk + 2 + i) for i in range(1, chunk + 2)]  # chunk + 1 pairs
    pairs1 = [(i + 1, chunk + 2 + i) for i in range(1, chunk + 2)]
    calls = []
    real = r._state_copy_fn

    def _fn(pages, groups, src, dst):
        calls.append((src.clone(), dst.clone()))
        return real(pages, groups, src, dst)

    r._state_copy_fn = _fn
    r._copy_state_blocks([pairs0, pairs1])
    assert len(calls) == 2
    assert all(s.shape == (2, chunk) for s, _ in calls)
    assert calls[1][0][:, 1:].eq(0).all() and calls[1][1][:, 1:].eq(0).all()
    for s, d in pairs0:
        assert g0[d].tolist() == [float(s)] * 3
    for s, d in pairs1:
        assert g1[d].tolist() == [1000.0 + s] * 3
    assert g0[0].tolist() == [0.0] * 3 and g1[0].tolist() == [1000.0] * 3


# ------------------------------------------------------------ _mamba_align_state_table


def _block_table(rows):
    bt = BlockTable(
        block_size=BS,
        max_num_reqs=4,
        max_num_blocks_per_req=MAX_BLOCKS,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=BS,
        cp_kv_cache_interleave_size=1,
    )
    for i, ids in enumerate(rows):
        bt.add_row(ids, i)
    return bt


def test_align_state_table_is_running_block_per_row():
    tables = [None, _block_table([[20, 21, 22], [30, 31], [40, 41, 42, 43]])]
    r = _runner(
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"], block_table=tables),
        _mamba_state_idx={"a": 2, "b": 0},  # c: not scheduled yet -> no running state
    )
    t = r._mamba_align_state_table(4, 1)
    assert t.shape == (4, 1)
    assert t.dtype == torch.int32
    # a: block 2 -> 22; b: block 0 -> 30; c: no index -> null block; row 3: padding.
    assert t[:, 0].tolist() == [22, 30, 0, 0]


def test_align_state_table_follows_preprocess():
    # End-to-end with _preprocess_mamba_align on the same block ids.
    ids = [[0], [50, 51, 52]]
    tables = [None, _block_table([ids[1]])]
    reqs = {"a": _req(4, *ids)}
    r = _align_runner(reqs, ["a"], state_idx={"a": 0})
    r.input_batch.block_table = tables
    r._preprocess_mamba_align(_sched_out({"a": 5}))
    assert r.copies == [[(50, 52)]]
    assert r._mamba_align_state_table(1, 1).tolist() == [[52]]


# ------------------------------------------------------------ dedicated state pool


def _pool_runner(max_num_reqs, req_ids, live=None, slot_pages=1):
    return _runner(
        _state_pool_slots={},
        _state_pool_free=list(range(max_num_reqs)),
        _state_pool_slot_pages=slot_pages,
        requests={r: object() for r in (live if live is not None else req_ids)},
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
    )


def test_state_pool_table_pages_per_slot():
    # Slot s owns pages 1 + s*n .. 1 + s*n + n - 1, n = 1 + num_speculative_blocks (page 0
    # is the null page).
    r = _pool_runner(4, ["a", "b"], slot_pages=2)
    t = r._state_pool_table(4, 2)
    assert t.dtype == torch.int32
    assert t.tolist() == [[1, 2], [3, 4], [-1, -1], [-1, -1]]
    assert r._state_pool_slots == {"a": 0, "b": 1}
    assert r._state_pool_free == [2, 3]


def test_state_pool_slot_stable_across_batch_reorder_and_absence():
    r = _pool_runner(4, ["a", "b", "c"])
    r._state_pool_table(4, 1)
    assert r._state_pool_slots == {"a": 0, "b": 1, "c": 2}
    # b is out of the persistent batch for a step (still live): keeps its slot.
    r.input_batch.req_ids = ["c", "a"]
    assert r._state_pool_table(2, 1)[:, 0].tolist() == [3, 1]
    r.input_batch.req_ids = ["b", "c", "a"]
    assert r._state_pool_table(4, 1)[:, 0].tolist() == [2, 3, 1, -1]
    assert r._state_pool_slots == {"a": 0, "b": 1, "c": 2}


def test_state_pool_freed_slots_reused_fifo():
    r = _pool_runner(3, ["a", "b", "c"])
    r._state_pool_table(3, 1)
    # a finishes (released via the scheduler output) ...
    r._release_state_pool_slots(_sched_out({}, finished={"a"}))
    del r.requests["a"]
    # ... and b disappears from self.requests without a release (lazy cleanup).
    del r.requests["b"]
    assert r._state_pool_free == [0]
    r.requests.update(d=object(), e=object())
    r.input_batch.req_ids = ["c", "d", "e"]
    t = r._state_pool_table(3, 1)
    assert r._state_pool_slots == {"c": 2, "d": 0, "e": 1}
    assert t[:, 0].tolist() == [3, 1, 2]
    assert r._state_pool_free == []


def test_state_pool_release_preempted_and_idempotent():
    r = _pool_runner(3, ["a", "b"])
    r._state_pool_table(3, 1)
    r._release_state_pool_slots(
        _sched_out({}, finished={"zzz"}, preempted={"b"})
    )  # unknown id ignored
    assert r._state_pool_slots == {"a": 0} and r._state_pool_free == [2, 1]
    r._release_state_pool_slots(_sched_out({}, preempted={"b"}))
    assert r._state_pool_free == [2, 1]  # no double free
    # Preempted request comes back: fresh slot (recomputed from zero state).
    r.input_batch.req_ids = ["b", "a"]
    r._state_pool_table(2, 1)
    assert r._state_pool_slots == {"a": 0, "b": 2}


def test_state_pool_release_without_pool_is_noop():
    _runner()._release_state_pool_slots(_sched_out({}, finished={"a"}))


def test_state_pool_exhausted_raises():
    r = _pool_runner(2, ["a", "b", "c"])
    with pytest.raises(RuntimeError, match="pool exhausted"):
        r._state_pool_table(4, 1)


def test_state_pool_pages_fit_pool_size():
    # With n = 1 + num_speculative_blocks (the state block-table width), every page the
    # table hands out is inside the pool sized by _state_pool_pages.
    for k in (0, 1, 2):
        max_reqs = 3
        r = _pool_runner(max_reqs, ["a", "b", "c"], slot_pages=1 + k)
        r.max_num_reqs = max_reqs
        r._dedicated_state_pool = lambda: True
        spec = MambaSpec(
            block_size=1024,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="none",
            num_speculative_blocks=k,
        )
        n_pages = r._state_pool_pages(spec)
        assert n_pages == 1 + max_reqs * (1 + k)
        t = r._state_pool_table(max_reqs, 1 + k)
        assert int(t.max()) == n_pages - 1
        assert sorted(t.flatten().tolist()) == list(range(1, n_pages))


def test_state_pool_table_wider_than_slot_stays_in_pool():
    # vLLM pads the state block table for alignment (max_model_len not a multiple of 128
    # gives a table several blocks wide in "none" mode). Only the first 1 + k columns are a
    # request's state blocks; the rest map to the null page, never past the pool.
    max_reqs, k, width = 3, 2, 16
    r = _pool_runner(max_reqs, ["a", "b", "c"], slot_pages=1 + k)
    t = r._state_pool_table(max_reqs, width)
    n_pages = 1 + max_reqs * (1 + k)
    assert t[:, : 1 + k].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert int(t[:, 1 + k :].abs().sum()) == 0
    assert int(t.max()) == n_pages - 1


def test_state_pool_pages_zero_for_align_or_shared():
    r = _runner(max_num_reqs=4)
    spec_align = MambaSpec(
        block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"
    )
    spec_none = MambaSpec(
        block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="none"
    )
    r._dedicated_state_pool = lambda: True
    assert r._state_pool_pages(spec_align) == 0
    r._dedicated_state_pool = lambda: False
    assert r._state_pool_pages(spec_none) == 0


# ------------------------------------------------------------ _kernel_block_sizes


def _attn(bs):
    return FullAttentionSpec(block_size=bs, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)


def _swa(bs):
    return SlidingWindowSpec(
        block_size=bs, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=64
    )


def _mamba(bs, mode="align"):
    return MambaSpec(
        block_size=bs, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode=mode
    )


def _kv_config(*specs):
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"l{i}"], s) for i, s in enumerate(specs)],
    )


def test_kernel_block_sizes_hybrid_attention_keeps_plugin_block_size():
    # Attention blocks enlarged to the state page (256) split into 32-token kernel blocks;
    # the state group keeps its own block size.
    r = _runner(_attn_kernel_block_size=32)
    cfg = _kv_config(_attn(256), _mamba(256), _swa(256))
    assert r._kernel_block_sizes(cfg) == [32, 256, 32]


def test_kernel_block_sizes_without_state_are_group_block_sizes():
    r = _runner(_attn_kernel_block_size=32)
    assert r._kernel_block_sizes(_kv_config(_attn(256), _swa(128))) == [256, 128]


def test_kernel_block_sizes_non_divisible_or_unset_fall_back():
    r = _runner(_attn_kernel_block_size=96)
    assert r._kernel_block_sizes(_kv_config(_attn(256), _mamba(256))) == [256, 256]
    r = _runner()  # get_kv_cache_spec not run yet: no _attn_kernel_block_size
    assert r._kernel_block_sizes(_kv_config(_attn(256), _mamba(256))) == [256, 256]


def test_kernel_block_sizes_mamba_none_mode():
    # Without prefix caching the state block spans max_model_len.
    r = _runner(_attn_kernel_block_size=32)
    assert r._kernel_block_sizes(_kv_config(_attn(64), _mamba(4096, "none"))) == [32, 4096]
