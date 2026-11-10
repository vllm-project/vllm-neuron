# SPDX-License-Identifier: Apache-2.0
"""Reasoning-parser servers and `thinking_token_budget` requests (CPU)."""

import pytest
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.exceptions import VLLMValidationError

platform = pytest.importorskip("vllm_neuron.vllm.platform")

MODEL = "Qwen/Qwen3-0.6B"


def test_reasoning_parser_server_uses_model_runner_v2(monkeypatch):
    """A reasoning parser only changes how the API server parses the output,
    so it does not move a dense model off Model Runner V2."""
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    config = EngineArgs(model=MODEL, reasoning_parser="qwen3").create_engine_config()

    assert config.reasoning_config is not None
    assert config.use_v2_model_runner


def test_thinking_token_budget_is_rejected():
    """Neither Neuron model runner enforces the budget, so a request that sets
    it is rejected instead of being served without it. `AsyncLLM` returns a
    `VLLMValidationError` to the client as a 400; it turns any other exception
    into an empty 500."""
    with pytest.raises(VLLMValidationError, match="thinking_token_budget") as exc:
        platform.NeuronPlatform.validate_request(
            {}, SamplingParams(thinking_token_budget=8)
        )

    assert exc.value.parameter == "thinking_token_budget"
