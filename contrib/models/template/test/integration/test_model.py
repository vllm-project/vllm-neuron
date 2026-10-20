# SPDX-License-Identifier: Apache-2.0
"""Sample integration test for a contrib model, served through vLLM on Neuron.

This template shows how to validate a contributed model's accuracy by serving
it through vLLM (with the ``vllm-neuron`` plugin) and comparing the per-token
logits against a HuggingFace reference. Copy this file to
``contrib/models/<your_model_name>/test/integration/test_model.py`` and fill in
the model-specific pieces marked with ``TODO``.

Registering your model
----------------------
Contrib models are NOT wired into ``vllm_neuron/model/registry.py``. Before this
test can serve your model, register its architecture with vLLM using one of the
approaches documented in ``contrib/README.md`` ("How do I use a custom
model with vLLM?"):

  * Copy the modeling code into ``vllm_neuron/model/<model_name>/`` and add the
    architecture to ``get_models()`` in ``vllm_neuron/model/registry.py``, or
  * Register your architecture out-of-tree with vLLM's model registry from your
    own package.

Running
-------
Serving a model requires Neuron hardware (or ``VLLM_NEURON_CPU_MODE=1`` for a
CPU-mode smoke run). Run it manually per the model README:

    pytest contrib/models/template/test/integration/test_model.py \
        -v -s --timeout=3600 > /tmp/test_output.txt 2>&1

Logit validation
----------------
Instead of matching generated text, this template compares the model's raw
per-token logits against a HuggingFace golden reference. Helpers used
from the ``vllm_neuron.accuracy`` API:

  * ``get_cached_reference_goldens`` computes (and caches) the HF reference
    logits. It returns ``fp32_logits`` (the FP32 baseline) and ``dtype_logits``
    (a same-dtype pass teacher-forced along the FP32 greedy sequence), which
    together enable three-way validation.
  * ``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so it returns
    dense per-token logits.
  * ``multi_prompt_logit_validation`` compares the two, flagging any argmax
    divergence and any top-K logit error outside the tolerance map.

See also:
  * vLLM offline inference: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
  * Registering a model with vLLM:
    https://docs.vllm.ai/en/latest/contributing/model/registration.html
"""

import pytest
import torch

# TODO: Set this to a checkpoint compatible with your model (see the model
# README's "Example Checkpoints" section). A local path or a Hugging Face Hub
# model id both work.
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

# TODO: Provide the prompts to validate. The HF reference logits are computed
# from these same prompts, so no expected text/logits are hard-coded here.
PROMPTS = [
    "The color of the sky is",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# TODO: Tune the tolerance map for your model. The argmax (divergence) lower
# bound stays strict at 1e-5 — a nonzero divergence is a real correctness bug
# and must not be widened to hide it. The per-K upper bounds bound the allowed
# top-K logit drift (here the common bf16 dense-golden convention).
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Deterministic results across runs.
torch.manual_seed(42)


# TODO: Set the (batch_size, seq_len) pairs to validate for your model.
# Each pair compiles a separate graph, so keep the list short.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
        (2, 8192),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve the contributed model through vLLM and validate its logits.

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

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key; ``model_checkpoint`` is what
    # the reference model loads from. They are the same value here because
    # MODEL_PATH already resolves for HuggingFace. Split them if you resolve the
    # checkpoint to a local directory yourself.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
    )

    # TODO: Adjust engine arguments (tensor_parallel_size, dtype, quantization,
    # neuron_config, etc.) to match your model and target instance type.
    #
    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits. Without it vLLM returns
    #     normalized log-probabilities, which do not match the HF reference.
    #   * async_scheduling=False keeps logprobs available. The async path returns
    #     before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy
    #     and deterministic, so a rerun compares the same token sequence.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    # Without them the defaults compile a power-of-2 ladder up to each limit
    # (for example batch 4 compiles decode graphs for 1, 2 and 4), which costs
    # compile time that the test never uses.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=32,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
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
