# SPDX-License-Identifier: Apache-2.0
"""The opt-in NKI stochastic sampler runs on the V1 model runner only. CPU only."""

from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.model_runner_v2_policy import get_neuron_model_runner_v2_policy


def _config(neuron_config):
    model = SimpleNamespace(
        runner_type="generate",
        is_moe=False,
        is_hybrid=False,
        is_attention_free=False,
        is_encoder_decoder=False,
        multimodal_config=None,
        enforce_eager=False,
        get_sliding_window=lambda: None,
        architectures=["LlamaForCausalLM"],
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
    )
    return SimpleNamespace(
        model_config=model,
        parallel_config=None,
        cache_config=None,
        scheduler_config=None,
        speculative_config=None,
        additional_config={"neuron_config": neuron_config},
    )


@pytest.mark.parametrize(
    "backend, routed", [("nki", True), ("cpu", False), (None, False)]
)
def test_the_nki_backend_selects_the_v1_runner(backend, routed):
    neuron_config = {} if backend is None else {"stochastic_sampling_backend": backend}
    policy = get_neuron_model_runner_v2_policy(_config(neuron_config))
    assert ("NKI stochastic sampling" in policy.unsupported_features) is routed
