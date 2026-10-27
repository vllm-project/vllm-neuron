# SPDX-License-Identifier: Apache-2.0
"""NeuronScheduler: decode steps between the chunks of a segmented prefill (CPU).

Drives a real NeuronScheduler (tiny local Qwen3 config, no weights, no tokenizer, a
plain full-attention KV cache config) through schedule() / update_from_output() with
fake model outputs and records which requests each step schedules.

decode_steps_between_prefill_chunks = N:
  N > 0: while a multi-chunk prefill runs and requests are decoding, each prefill
         chunk is followed by N decode-only steps (prefill hidden), then the next chunk.
  N = 0: the prefill runs uninterrupted; decodes wait until its last chunk.
"""

import json

import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

# vllm_neuron is imported inside the fixture, not at collection: importing the real
# vllm_neuron.functional before test/unit/model/qwen3_5/test_model_smoke.py runs defeats
# that test's sys.modules stubs (``import vllm_neuron.functional as NF`` resolves through
# the already-imported parent package).

BLOCK_SIZE = 16
CHUNK = 64  # max_num_batched_tokens: one prefill chunk
NUM_BLOCKS = 1000
LONG_PROMPT = 3 * CHUNK + 8  # 4 chunks: 64, 64, 64, 8
SHORT_PROMPT = 10  # 1 chunk

_TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 1,
    "vocab_size": 1000,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_qwen3")
    (d / "config.json").write_text(json.dumps(_TINY_QWEN3))
    return str(d)


@pytest.fixture
def make_scheduler(model_dir):
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    def _make(decode_steps: int, max_num_seqs: int = 4):
        model_config = ModelConfig(
            model=model_dir,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=1024,
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=CHUNK,
            max_model_len=1024,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            async_scheduling=False,
        )
        cache_config = CacheConfig(
            block_size=BLOCK_SIZE, cache_dtype="auto", enable_prefix_caching=False
        )
        cache_config.num_gpu_blocks = NUM_BLOCKS
        vllm_config = VllmConfig(
            model_config=model_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            additional_config={
                "neuron_config": {"decode_steps_between_prefill_chunks": decode_steps}
            },
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        sched = NeuronScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK_SIZE,
        )
        assert sched._decode_steps_between_chunks == decode_steps
        assert sched.num_batched_tokens_buckets == [CHUNK]
        return sched

    return _make


def _request(req_id: str, prompt_len: int, max_tokens: int = 100) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=[(i % 900) + 1 for i in range(prompt_len)],
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
    )


def _step(sched) -> dict[str, int]:
    """One engine step: schedule, run a fake model, feed the output back.

    Returns {req_id: num_scheduled_tokens}. A request whose prompt is complete after this
    step samples one token; a partial prefill chunk samples none.
    """
    out = sched.schedule()
    req_ids = list(out.num_scheduled_tokens)
    sampled = [
        [7] if sched.requests[r].num_computed_tokens >= sched.requests[r].num_prompt_tokens
        else []
        for r in req_ids
    ]
    sched.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        ),
    )
    return dict(out.num_scheduled_tokens)


def _start_decoders(sched, ids=("a", "b"), max_tokens: int = 100):
    """Bring short requests into decode (one prefill step each)."""
    for r in ids:
        sched.add_request(_request(r, SHORT_PROMPT, max_tokens))
        assert _step(sched) == {r: SHORT_PROMPT}
    # They now decode together (branch 3.3).
    assert _step(sched) == {r: 1 for r in ids}
    assert sched._interleave_decodes_left == 0


DECODE_AB = {"a": 1, "b": 1}


def _chunk(n: int) -> dict[str, int]:
    return {"c": n}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_decode_steps_after_each_prefill_chunk(make_scheduler, n):
    sched = make_scheduler(n)
    _start_decoders(sched)
    sched.add_request(_request("c", LONG_PROMPT))

    steps = [_step(sched) for _ in range(4 + 3 * n + 1)]
    expected = []
    for i, size in enumerate((CHUNK, CHUNK, CHUNK, 8)):
        expected.append(_chunk(size))
        if i < 3:
            expected += [DECODE_AB] * n
    # After the last chunk C is decoding too and joins the batch.
    expected.append({"a": 1, "b": 1, "c": 1})
    assert steps == expected
    assert sched._interleave_decodes_left == 0
    # Every token of C was computed exactly once, a/b produced one token per decode step.
    assert sched.requests["c"].num_computed_tokens == LONG_PROMPT + 1
    assert sched.requests["a"].num_output_tokens == 1 + 1 + 3 * n + 1


def test_zero_keeps_prefill_uninterrupted(make_scheduler):
    sched = make_scheduler(0)
    _start_decoders(sched)
    sched.add_request(_request("c", LONG_PROMPT))
    steps = [_step(sched) for _ in range(5)]
    assert steps == [
        _chunk(CHUNK),
        _chunk(CHUNK),
        _chunk(CHUNK),
        _chunk(8),
        {"a": 1, "b": 1, "c": 1},
    ]
    assert sched._interleave_decodes_left == 0


def test_no_decoders_prefill_runs_back_to_back(make_scheduler):
    # Nothing to interleave: the counter must not arm without hidden decodes.
    sched = make_scheduler(2)
    sched.add_request(_request("c", LONG_PROMPT))
    steps = [_step(sched) for _ in range(5)]
    assert steps == [_chunk(CHUNK), _chunk(CHUNK), _chunk(CHUNK), _chunk(8), {"c": 1}]
    assert sched._interleave_decodes_left == 0


def test_single_chunk_prefill_not_followed_by_extra_decode_only_steps(make_scheduler):
    # A one-chunk prefill arms the counter in 3.2 but finishes in that chunk, so the next
    # step is a normal decode step with every request (3.3 resets the counter).
    sched = make_scheduler(2)
    _start_decoders(sched)
    sched.add_request(_request("c", SHORT_PROMPT))
    assert _step(sched) == {"c": SHORT_PROMPT}
    assert sched._interleave_decodes_left == 2  # armed, but no prefill left to hide
    assert _step(sched) == {"a": 1, "b": 1, "c": 1}
    assert sched._interleave_decodes_left == 0


def test_counter_rearmed_when_new_prefill_starts(make_scheduler):
    # Branch 3.2 resets the counter: a prefill aborted mid-interleave leaves a stale count,
    # and the next prefill must still get its full N decode steps after its first chunk.
    sched = make_scheduler(2)
    _start_decoders(sched)
    sched.add_request(_request("c", LONG_PROMPT))
    assert _step(sched) == _chunk(CHUNK)
    assert _step(sched) == DECODE_AB
    assert sched._interleave_decodes_left == 1
    sched.finish_requests("c", RequestStatus.FINISHED_ABORTED)

    sched.add_request(_request("d", 2 * CHUNK))
    # No prefill in running and a waiter is admissible -> 3.2 (decodes hidden).
    assert _step(sched) == {"d": CHUNK}
    assert sched._interleave_decodes_left == 2
    assert _step(sched) == DECODE_AB
    assert _step(sched) == DECODE_AB
    assert _step(sched) == {"d": CHUNK}
    assert _step(sched) == {"a": 1, "b": 1, "d": 1}


def test_decoders_finishing_mid_interleave_resume_prefill(make_scheduler):
    # If every decoding request finishes during the interleave, the remaining decode
    # steps are skipped and the prefill continues at once (3.0 needs decodes).
    sched = make_scheduler(3)
    # a/b: prefill samples token 1, first decode step token 2, so max_tokens=3 means they
    # finish on the first interleaved decode step.
    _start_decoders(sched, max_tokens=3)
    sched.add_request(_request("c", LONG_PROMPT))
    assert _step(sched) == _chunk(CHUNK)
    assert _step(sched) == DECODE_AB  # a, b finish here
    assert "a" not in sched.requests and "b" not in sched.requests
    assert sched._interleave_decodes_left == 2
    assert _step(sched) == _chunk(CHUNK)
    # No decodes hidden in 3.1 -> the counter is not re-armed.
    assert _step(sched) == _chunk(CHUNK)
    assert _step(sched) == _chunk(8)


def test_waiting_request_not_admitted_during_interleave(make_scheduler):
    # During the decode steps the prefill is hidden from the base scheduler; a new waiter
    # must not sneak in (can_schedule holds it back while a prefill is in running).
    sched = make_scheduler(2)
    _start_decoders(sched)
    sched.add_request(_request("c", 2 * CHUNK))
    assert _step(sched) == _chunk(CHUNK)
    sched.add_request(_request("d", SHORT_PROMPT))
    assert _step(sched) == DECODE_AB
    assert _step(sched) == DECODE_AB
    assert _step(sched) == _chunk(CHUNK)
    # c finished its prefill -> d can now prefill alone (3.2), then everyone decodes.
    assert _step(sched) == {"d": SHORT_PROMPT}
    assert _step(sched) == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_async_kv_pull_branch_clears_counter(make_scheduler, monkeypatch):
    # The async-KV-pull branch (pure KV consumer, every waiter a remote-prefill pull)
    # keeps the decodes running and must clear any armed counter, otherwise the pulled
    # request's last-token recompute would be hidden behind stale decode steps.
    # A real KV connector is out of scope on CPU, so flip the role flag and stub the base
    # schedule call (it would otherwise run the pull as a local prefill).
    sched = make_scheduler(2)
    _start_decoders(sched)
    sched._skip_decode_holdback = True
    sched._interleave_decodes_left = 2  # e.g. armed by a previous 3.2 step
    pull = _request("p", 40)
    pull.kv_transfer_params = {"do_remote_prefill": True}
    sched.add_request(pull)

    seen = {}

    def _base(throttle_prefills=False):
        seen["running"] = sorted(r.request_id for r in sched.running)
        seen["waiting"] = [r.request_id for r in sched.waiting]
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(sched, "_call_base_schedule", _base)
    sched.schedule()
    assert seen == {"running": ["a", "b"], "waiting": ["p"]}  # no holdback
    assert sched._interleave_decodes_left == 0
