# SPDX-License-Identifier: Apache-2.0
"""The resolved Qwen3.5 defaults: a plain ``vllm serve`` gets the measured configuration.

Covers NeuronPlatform._apply_qwen3_5_defaults (batch limits, buckets, snapshot rows,
language-model-only), NeuronPlatform._runs_vision_encoder,
NeuronModelRunner._resolve_async_readback_skip and NeuronWorker._get_kv_cap_fraction.
CPU only.
"""

import sys
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

_ENV = (
    "VLLM_NEURON_BACKEND_DELTANET_SNAPSHOT_ROWS",
    "VLLM_NEURON_BACKEND_ASYNC_SKIP_PREFILL_READBACK",
    "VLLM_NEURON_BACKEND_ASYNC_DEVICE_TOKEN_MERGE",
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "/models/Qwen3.8-27B"])


def _vllm_config(
    max_model_len=16384, max_num_seqs=256, max_num_batched_tokens=2048, apc=True
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5ForConditionalGeneration"]
            ),
            max_model_len=max_model_len,
            multimodal_config=SimpleNamespace(language_model_only=False),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=apc),
    )


def test_zero_config_gets_the_measured_serving_config():
    cfg, nc = _vllm_config(), {}
    assert NeuronPlatform._is_qwen3_5(cfg)
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert cfg.scheduler_config.max_num_seqs == 32
    assert cfg.scheduler_config.max_num_batched_tokens == 8192
    assert nc["num_seqs_buckets"] == [32]
    assert nc["num_batched_tokens_buckets"] == [256, 2048, 4096, 8192]
    assert nc["kv_segment_size_buckets"] == [8192]
    assert nc["decode_context_length_buckets"] == [16256]
    assert nc["deltanet_snapshot_rows"] == 129
    assert cfg.model_config.multimodal_config.language_model_only is True


def test_explicit_values_win(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "m", "--max-num-seqs", "16", "--max-num-batched-tokens=4096"],
    )
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_SNAPSHOT_ROWS", "33")
    cfg = _vllm_config(max_num_seqs=16, max_num_batched_tokens=4096)
    nc = {"num_seqs_buckets": [8, 16], "decode_context_length_buckets": [4096]}
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert cfg.scheduler_config.max_num_seqs == 16
    assert cfg.scheduler_config.max_num_batched_tokens == 4096
    # The explicit bucket keys win, and each key that is not set gets its default.
    assert nc == {
        "num_seqs_buckets": [8, 16],
        "decode_context_length_buckets": [4096],
        "num_batched_tokens_buckets": [256, 2048, 4096],
        "kv_segment_size_buckets": [4096],
        "_qwen3_5_defaults_applied": True,
        "deltanet_snapshot_rows": 33,
    }


@pytest.mark.parametrize(
    "max_model_len, expected",
    [
        (4096, [3968]),
        (16384, [16256]),
        (16385, [16384]),
        (32768, [16384, 32640]),
        (65536, [16384, 32768, 65408]),
        (262144, [16384, 32768, 65536, 131072, 262016]),
    ],
)
def test_decode_context_buckets_double_up_to_max_model_len(max_model_len, expected):
    cfg, nc = _vllm_config(max_model_len=max_model_len), {}
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert nc["decode_context_length_buckets"] == expected


def test_an_art_rollout_profile_keeps_its_buckets_and_gets_the_rest(monkeypatch):
    # ART's trn2.48xlarge Qwen3.6-27B profile: 64K, 8 sequences, 2048-token chunks.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm",
            "serve",
            "m",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "2048",
        ],
    )
    cfg = _vllm_config(max_model_len=65536, max_num_seqs=8, max_num_batched_tokens=2048)
    nc = {
        "num_batched_tokens_buckets": [2048],
        "num_seqs_buckets": [1, 4, 8],
        "decode_context_length_buckets": [4096, 8192, 16384, 32768],
    }
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert nc["num_batched_tokens_buckets"] == [2048]
    assert nc["num_seqs_buckets"] == [1, 4, 8]
    assert nc["decode_context_length_buckets"] == [4096, 8192, 16384, 32768]
    assert nc["kv_segment_size_buckets"] == [2048]


def test_no_prefix_caching_needs_no_snapshot_rows():
    cfg, nc = _vllm_config(apc=False), {}
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert nc["deltanet_snapshot_rows"] == 0


def test_a_short_context_clips_the_buckets():
    cfg, nc = _vllm_config(max_model_len=4096), {}
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert cfg.scheduler_config.max_num_batched_tokens == 4096
    assert nc["num_batched_tokens_buckets"] == [256, 2048, 4096]
    # Prefill is single-shot, so there is no KV segment to bucket.
    assert "kv_segment_size_buckets" not in nc
    assert nc["decode_context_length_buckets"] == [3968]


def test_a_later_pass_keeps_the_first_result():
    cfg, nc = _vllm_config(), {}
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    cfg.scheduler_config.max_num_seqs = 64
    NeuronPlatform._apply_qwen3_5_defaults(cfg, nc)
    assert cfg.scheduler_config.max_num_seqs == 64
    assert nc["num_seqs_buckets"] == [32]


def test_other_models_are_not_qwen3_5():
    cfg = _vllm_config()
    cfg.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    assert not NeuronPlatform._is_qwen3_5(cfg)


def _runner(model, reorder=False, spec=None):
    return SimpleNamespace(
        model=model,
        speculative_config=spec,
        _use_async_reorder=lambda: reorder,
    )


@pytest.mark.parametrize(
    "model_opt_in, env, reorder, spec, expected",
    [
        (True, None, False, None, True),  # Qwen3.5 default
        (False, None, False, None, False),  # other models keep the old behavior
        (False, "1", False, None, True),  # forced on
        (True, "0", False, None, False),  # kill switch
        (True, None, True, None, False),  # the in-graph reorder owns the decode ids
        (True, "1", False, object(), False),  # never with speculative decoding
    ],
)
def test_async_readback_skip_resolution(
    monkeypatch, model_opt_in, env, reorder, spec, expected
):
    if env is not None:
        monkeypatch.setenv("VLLM_NEURON_BACKEND_ASYNC_SKIP_PREFILL_READBACK", env)
        monkeypatch.setenv("VLLM_NEURON_BACKEND_ASYNC_DEVICE_TOKEN_MERGE", env)
    model = SimpleNamespace(supports_async_readback_skip=model_opt_in)
    runner = _runner(model, reorder=reorder, spec=spec)
    NeuronModelRunner._resolve_async_readback_skip(runner)
    assert runner._skip_prefill_readback is expected
    assert runner._device_token_merge is expected


def _worker(model):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def test_kv_cap_comes_from_the_model_then_the_global_default(monkeypatch):
    qwen = SimpleNamespace(default_kv_cap_fraction=0.23)
    assert NeuronWorker._get_kv_cap_fraction(_worker(qwen)) == 0.23
    assert NeuronWorker._get_kv_cap_fraction(_worker(SimpleNamespace())) == 0.30
    monkeypatch.setenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.18")
    assert NeuronWorker._get_kv_cap_fraction(_worker(qwen)) == 0.18


def test_the_qwen3_5_model_declares_its_defaults():
    from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5ForCausalLM

    assert Qwen3_5ForCausalLM.supports_async_readback_skip is True
    assert Qwen3_5ForCausalLM.default_kv_cap_fraction == 0.23
    assert getattr(Qwen3_5ForCausalLM, "supports_async_reorder", False) is False


def test_the_factory_takes_the_multimodal_call_of_the_runner(monkeypatch):
    # The checkpoint has a vision_config, so the runner calls
    # from_configs(hf_config=..., text_neuron_config=..., encoder_neuron_config=...).
    from vllm_neuron.model.qwen3_5 import factory

    seen = {}

    def select(cls, hf_config, neuron_config):
        seen["neuron_config"] = neuron_config
        return "model"

    monkeypatch.setattr(
        factory.Qwen3_5ForCausalLM, "_select_implementation", classmethod(select)
    )
    text = object()
    out = factory.Qwen3_5ForConditionalGeneration.from_configs(
        hf_config=None, text_neuron_config=text, encoder_neuron_config=object()
    )
    assert out == "model" and seen["neuron_config"] is text


def _model_config(vision=True, language_model_only=False, mm_embeds=False):
    hf_config = SimpleNamespace()
    if vision:
        hf_config.vision_config = SimpleNamespace()
    return SimpleNamespace(
        hf_config=hf_config,
        multimodal_config=SimpleNamespace(
            language_model_only=language_model_only, enable_mm_embeds=mm_embeds
        ),
    )


def test_a_language_model_only_server_gets_no_encoder_settings():
    """The checkpoint has a vision encoder, but the server takes text only."""
    assert NeuronPlatform._runs_vision_encoder(_model_config()) is True
    assert not NeuronPlatform._runs_vision_encoder(
        _model_config(language_model_only=True)
    )
    # Precomputed embeddings still go through the encoder settings.
    assert NeuronPlatform._runs_vision_encoder(
        _model_config(language_model_only=True, mm_embeds=True)
    )
    assert not NeuronPlatform._runs_vision_encoder(_model_config(vision=False))


def test_qwen3_5_defaults_leave_no_encoder_to_configure():
    cfg = _vllm_config()
    cfg.model_config.hf_config.vision_config = SimpleNamespace()
    assert NeuronPlatform._runs_vision_encoder(cfg.model_config)
    NeuronPlatform._apply_qwen3_5_defaults(cfg, {})
    assert not NeuronPlatform._runs_vision_encoder(cfg.model_config)
