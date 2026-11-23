# SPDX-License-Identifier: Apache-2.0
"""Prefix-affinity routing across data-parallel engines (CPU only)."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.patches import dp_load_balancer_patch as lb
from vllm_neuron.vllm.patches.dp_prefix_affinity import PrefixAffinityIndex

CHUNK = 128


def tokens(seed: int, n: int) -> list[int]:
    return [(seed * 7919 + i * 31) % 50000 for i in range(n)]


# ---------------------------------------------------------------- the index


def test_prefix_hashes_full_chunks_and_keeps_the_rest():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    prefix = index.prefix(tokens(1, 300))
    assert len(prefix.hashes) == 2
    assert prefix.pending == tokens(1, 300)[256:]
    assert prefix.last_hash == prefix.hashes[-1]
    assert index.prefix(tokens(1, 300)).hashes == prefix.hashes


def test_a_hash_covers_every_chunk_before_it():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    base = tokens(1, 3 * CHUNK)
    changed = [base[0] + 1] + base[1:]
    a, b = index.prefix(base).hashes, index.prefix(changed).hashes
    assert all(x != y for x, y in zip(a, b))


def test_lora_and_cache_salt_start_another_chain():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    ids = tokens(1, 2 * CHUNK)
    plain = index.prefix(ids).hashes
    assert index.prefix(ids, lora_name="a").hashes[0] != plain[0]
    assert index.prefix(ids, cache_salt="s").hashes[0] != plain[0]


def test_match_finds_the_engines_with_the_longest_prefix():
    index = PrefixAffinityIndex(num_engines=3, slack=4)
    long_prompt = tokens(1, 4 * CHUNK)
    index.admit("a", 0, index.prefix(long_prompt[: 2 * CHUNK]))
    index.admit("b", 2, index.prefix(long_prompt))
    assert index.match(index.prefix(long_prompt).hashes) == ([2], 4)
    assert index.match(index.prefix(long_prompt[: 2 * CHUNK]).hashes) == ([0, 2], 2)
    assert index.match(index.prefix(tokens(9, 4 * CHUNK)).hashes) == ([0, 1, 2], 0)


def test_choose_follows_the_prefix_within_the_slack():
    index = PrefixAffinityIndex(num_engines=2, slack=2)
    prompt = index.prefix(tokens(1, 4 * CHUNK))
    index.admit("a", 1, prompt)
    order = [0, 1]
    assert index.choose(prompt, [(0.0, 0.0), (2.0, 0.0)], 0, order) == 1
    assert index.choose(prompt, [(0.0, 0.0), (3.0, 0.0)], 0, order) == 0
    s = index.stats
    assert (s.requests, s.matched, s.followed, s.overridden) == (2, 2, 1, 1)
    assert s.matched_tokens == 4 * CHUNK


def test_a_prefix_on_every_engine_leaves_the_load_balanced_choice():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    system = index.prefix(tokens(1, 2 * CHUNK))
    index.admit("a", 0, system)
    index.admit("b", 1, system)
    assert index.choose(system, [(3.0,), (0.0,)], 1, [0, 1]) == 1
    assert index.stats.matched == 0


def test_generated_tokens_keep_a_rollout_on_its_engine():
    """Rollouts of one prompt differ only in what they generated."""
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    task = tokens(1, 2 * CHUNK)
    index.admit("r0", 0, index.prefix(task))
    index.admit("r1", 1, index.prefix(task))
    gen0, gen1 = tokens(2, CHUNK + 5), tokens(3, CHUNK + 5)
    index.extend("r0", gen0[:100])
    index.extend("r0", gen0[100:])
    index.extend("r1", gen1)
    index.finish("r0")
    index.finish("r1")
    turn2 = index.prefix(task + gen1 + tokens(4, 300))
    assert index.match(turn2.hashes) == ([1], 3)
    assert index.choose(turn2, [(0.0,), (1.0,)], 0, [0, 1]) == 1


def test_extend_of_an_unknown_request_does_nothing():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    index.extend("missing", tokens(1, 300))
    assert index.match(index.prefix(tokens(1, 300)).hashes)[1] == 0


def test_eviction_drops_the_end_of_a_prefix_first():
    index = PrefixAffinityIndex(num_engines=1, slack=4, max_chunks_per_engine=4)
    first = tokens(1, 3 * CHUNK)
    index.admit("a", 0, index.prefix(first))
    index.admit("b", 0, index.prefix(tokens(2, 2 * CHUNK)))
    # 5 chunks for a capacity of 4: the last chunk of the first prompt goes.
    assert index.match(index.prefix(first).hashes) == ([0], 2)


def test_resize_forgets_the_engines():
    index = PrefixAffinityIndex(num_engines=2, slack=4)
    prompt = index.prefix(tokens(1, 2 * CHUNK))
    index.admit("a", 1, prompt)
    index.resize(3)
    assert index.num_engines == 3
    assert index.match(prompt.hashes)[1] == 0


@pytest.mark.parametrize(
    "kwargs", [{"num_engines": 0}, {"max_chunks_per_engine": 0}, {"chunk_tokens": 0}]
)
def test_invalid_index_sizes_raise(kwargs):
    args = {"num_engines": 2, "slack": 1, **kwargs}
    with pytest.raises(ValueError):
        PrefixAffinityIndex(**args)


# ------------------------------------------------- the patched DP client


class FakeClient:
    """The DPLBAsyncMPClient state that the routing reads and writes."""

    def __init__(self, num_engines: int, affinity=None):
        self.core_engines = [bytes([i]) for i in range(num_engines)]
        self.lb_engines = [[0, 0, 0.0] for _ in range(num_engines)]
        self.engine_inflight: Counter = Counter()
        self.reqs_in_flight: dict = {}
        self.engine_indices = {e: i for i, e in enumerate(self.core_engines)}
        self.client_count = 1
        self.eng_start_index = 0
        self.shared_engine_inflight = None
        setattr(self, lb._ROUTING_ENABLED_ATTR, True)
        setattr(self, lb._AFFINITY_ATTR, affinity)

    def route(self, request_id: str, prompt: list[int], **fields) -> int:
        request = SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=prompt,
            sampling_params=object(),
            pooling_params=None,
            mm_features=None,
            prompt_embeds=None,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        for key, value in fields.items():
            setattr(request, key, value)
        engine = lb._get_core_engine_for_request(self, request)
        return self.core_engines.index(engine)

    def finish(self, request_id: str, generated: list[int]) -> None:
        output = SimpleNamespace(request_id=request_id, new_token_ids=generated)
        outputs = SimpleNamespace(outputs=[output], finished_requests={request_id})
        asyncio.run(lb._process_engine_outputs(self, outputs))

    def set_running(self, engine: int, running: int) -> None:
        self.lb_engines[engine] = [0, running, 0.0]


def run_session(client: FakeClient) -> list[int]:
    """Three turns of one agent session while two other engines stay busy."""
    prompt = tokens(1, 4 * CHUNK)
    engines = [client.route("t0", prompt)]
    for turn in (1, 2):
        generated = tokens(10 + turn, 200)
        client.finish(f"t{turn - 1}", generated)
        prompt = prompt + generated + tokens(20 + turn, 300)
        # Another session now runs on the engine of this session.
        client.set_running(engines[-1], client.lb_engines[engines[-1]][1] + 1)
        engines.append(client.route(f"t{turn}", prompt))
    return engines


def test_a_session_stays_on_its_engine():
    client = FakeClient(4, PrefixAffinityIndex(num_engines=4, slack=4))
    engines = run_session(client)
    assert engines == [engines[0]] * 3
    stats = getattr(client, lb._AFFINITY_ATTR).stats
    assert (stats.followed, stats.overridden) == (2, 0)


def test_without_affinity_a_session_moves_to_the_least_loaded_engine():
    engines = run_session(FakeClient(4))
    assert engines[1] != engines[0] and engines[2] != engines[1]


def test_the_load_guard_moves_a_session_off_an_overloaded_engine():
    client = FakeClient(2, PrefixAffinityIndex(num_engines=2, slack=2))
    prompt = tokens(1, 4 * CHUNK)
    first = client.route("t0", prompt)
    client.finish("t0", tokens(2, 200))
    client.set_running(first, 3)
    assert client.route("t1", prompt + tokens(2, 200)) == 1 - first
    assert getattr(client, lb._AFFINITY_ATTR).stats.overridden == 1


def test_an_explicit_rank_wins_and_records_the_prefix():
    client = FakeClient(2, PrefixAffinityIndex(num_engines=2, slack=4))
    prompt = tokens(1, 4 * CHUNK)
    assert client.route("a", prompt, data_parallel_rank=1) == 1
    client.set_running(1, 1)
    assert client.route("b", prompt) == 1


def test_a_multimodal_request_is_load_balanced():
    affinity = PrefixAffinityIndex(num_engines=2, slack=4)
    client = FakeClient(2, affinity)
    prompt = tokens(1, 4 * CHUNK)
    client.route("a", prompt, data_parallel_rank=1)
    client.set_running(1, 1)
    assert client.route("b", prompt, mm_features=[object()]) == 0
    assert affinity.stats.requests == 0


def test_routing_keeps_the_in_flight_accounting():
    client = FakeClient(2, PrefixAffinityIndex(num_engines=2, slack=4))
    engine = client.route("a", tokens(1, 4 * CHUNK))
    assert client.reqs_in_flight["a"] == client.core_engines[engine]
    client.finish("a", tokens(2, 10))
    assert "a" not in client.reqs_in_flight
    assert client.engine_inflight[client.core_engines[engine]] == 0


# ------------------------------------------------------------ configuration


def vllm_config(prefix_caching=True, max_num_seqs=8, num_gpu_blocks=5200):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_caching,
            num_gpu_blocks=num_gpu_blocks,
            block_size=64,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_affinity_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_DP_PREFIX_AFFINITY", raising=False)
    assert lb._make_prefix_affinity(vllm_config(), 2, 1) is None


def test_affinity_needs_prefix_caching(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_DP_PREFIX_AFFINITY", "1")
    assert lb._make_prefix_affinity(vllm_config(prefix_caching=False), 2, 1) is None


def test_affinity_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_DP_PREFIX_AFFINITY", "1")
    monkeypatch.delenv("VLLM_NEURON_DP_PREFIX_AFFINITY_SLACK", raising=False)
    index = lb._make_prefix_affinity(vllm_config(), 2, 1)
    assert index.num_engines == 2
    assert index.slack == 4
    # 5200 blocks of 64 tokens over 2 engines, in 128-token chunks.
    assert index.max_chunks_per_engine == 5200 * 64 // 2 // CHUNK


def test_affinity_slack_from_the_environment(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_DP_PREFIX_AFFINITY", "1")
    monkeypatch.setenv("VLLM_NEURON_DP_PREFIX_AFFINITY_SLACK", "0")
    assert lb._make_prefix_affinity(vllm_config(), 2, 1).slack == 0
