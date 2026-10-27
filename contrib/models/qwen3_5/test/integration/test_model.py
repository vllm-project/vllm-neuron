# SPDX-License-Identifier: Apache-2.0
"""Integration test for Qwen3.5, served through vLLM on Neuron.

Adapted from ``contrib/models/template/test/integration/test_model.py``. It
serves the model through vLLM (with the ``vllm-neuron`` plugin) and compares the
per-token logits against a HuggingFace reference.

Requirements
------------
* A ``vllm-neuron`` build with recurrent-state support (``KVSpec.state_layers``
  in ``vllm_neuron/model/kv_cache.py``). Without it the model fails at
  ``bind_state_cache``.
* Neuron hardware. The defaults (Qwen3.5-2B, TP=4) fit a trn2.3xlarge.

Registering the model
---------------------
The test registers both architectures itself, from this process, before it
constructs ``LLM(...)`` (Option 1 in ``contrib/README.md``). It imports the
installed ``qwen3_5`` package if ``pip install -e contrib/models/qwen3_5`` was
run, and otherwise loads this folder's ``src/`` under that name.

Running
-------
    pytest contrib/models/qwen3_5/test/integration/test_model.py \
        -v -s --timeout=3600 > /tmp/test_output.txt 2>&1

Override the checkpoint or the tensor-parallel size with
``QWEN35_MODEL_PATH`` and ``QWEN35_TP``.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

# A local path or a Hugging Face Hub model id. See the README for checkpoints.
MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-2B")
TENSOR_PARALLEL_SIZE = int(os.environ.get("QWEN35_TP", "4"))

PROMPTS = [
    "The color of the sky is",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The argmax (divergence) lower bound stays strict at 1e-5: a nonzero divergence
# is a real correctness bug. The per-K upper bounds are the template's bf16
# dense-golden convention and have not been tuned for Qwen3.5 yet.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Deterministic results across runs.
torch.manual_seed(42)


def _import_qwen3_5():
    """Return the ``qwen3_5`` package: the installed one, or this folder's ``src/``."""
    try:
        import qwen3_5
    except ImportError:
        src = Path(__file__).resolve().parents[2] / "src"
        spec = importlib.util.spec_from_file_location(
            "qwen3_5", src / "__init__.py", submodule_search_locations=[str(src)]
        )
        qwen3_5 = importlib.util.module_from_spec(spec)
        sys.modules["qwen3_5"] = qwen3_5
        spec.loader.exec_module(qwen3_5)
    return qwen3_5


def _register_qwen3_5() -> None:
    """Register both Qwen3.5 architectures with vLLM (contrib/README.md, Option 1).

    The engine and worker processes inherit the registration when vLLM forks
    them. If they are spawned instead, install the package, whose
    ``vllm.general_plugins`` entry point registers the architectures in every
    process.
    """
    from vllm import ModelRegistry

    qwen3_5 = _import_qwen3_5()
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration", qwen3_5.Qwen3_5ForConditionalGeneration
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        qwen3_5.Qwen3_5MoeForConditionalGeneration,
    )


# Each pair compiles a separate graph, so keep the list short. seq_len is also
# the prefill bucket: the model does not support chunked prefill.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 1024),
        (4, 1024),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve Qwen3.5 through vLLM and validate its logits.

    Args:
        batch_size: Number of prompts to run in a single batch.
        seq_len: Maximum model (context + generation) length to configure.
    """

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    _register_qwen3_5()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
    )

    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits.
    #   * async_scheduling=False keeps logprobs available.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy.
    #
    # Two are required by this model:
    #   * enable_prefix_caching=False: prefix caching over recurrent state is
    #     not supported.
    #   * no optimization_level: an explicit level disables the model's
    #     compiler_args_for, which the dense decode graph needs.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        generate_fn = create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH)
        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
        ]
        result = multi_prompt_logit_validation(
            prompts_input_ids=prompts_input_ids,
            generate_fn=generate_fn,
            prompts_expected_logits=cached_goldens["dtype_logits"],
            prompts_baseline_logits=cached_goldens["fp32_logits"],
            colorize=True,
            replicate_to_batch_size=batch_size,
            tol_map=TOL_MAP,
        )
        assert result.passed, (
            f"Logit validation failed for parameters: {batch_size=}, {seq_len=}"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
