# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Gemma3 (rnj-1) contrib model, served through vLLM on Neuron.

The test serves ``EssentialAI/rnj-1-instruct`` with the implementation in
``contrib/models/gemma3/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its
own ``Gemma3ForCausalLM``; the registration replaces it.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test.
With ``pip install -e contrib/models/gemma3`` the plugin entry point registers
the architecture in every process instead, and the test still works.

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    GEMMA3_CHECKPOINT_PATH=/path/to/rnj-1-instruct \
    pytest contrib/models/gemma3/test/integration/test_model.py \
        -v -s --timeout=10800 > /tmp/test_output.txt 2>&1

Without ``GEMMA3_CHECKPOINT_PATH`` the test downloads the pinned revision
(about 33 GB of FP32 weights) from the Hugging Face Hub.

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence.
``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so that it returns
dense per-token logits, and ``multi_prompt_logit_validation`` compares the two.
It flags any argmax divergence and any top-K logit error outside ``TOL_MAP``.
"""

import os

import pytest
import torch

from _local_src import src_package

MODEL_ID = "EssentialAI/rnj-1-instruct"
# The HuggingFace revision of the recorded runs.
HF_REVISION = "5ef96d7a82b2bff8009192979432c9bc64e870cd"
CHECKPOINT_ENV = "GEMMA3_CHECKPOINT_PATH"

# Prompts to validate. The HF reference logits come from these same prompts,
# so no expected text or logits are hard-coded here.
PROMPTS = [
    "The color of the sky is",
    "A short summary of the history of the printing press:",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map of the template. The argmax (divergence)
# lower bound stays strict at 1e-5 — a nonzero divergence is a real
# correctness bug and must not be widened to hide it. The per-K upper bounds
# bound the allowed top-K logit drift.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# The tensor parallel degree of the recorded runs: 32 Q heads and 8 KV heads
# give 4 Q heads and one KV head on each rank.
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)

ARCHITECTURE = "Gemma3ForCausalLM"


def _checkpoint() -> str:
    """Return a local directory with the pinned checkpoint."""
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=HF_REVISION)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


# Batch 1 and batch 4 at a 2048-token model length. Each pair compiles its own
# graphs.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
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

    _register_model()
    checkpoint = _checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint stores FP32 weights and the engine serves BF16, so the
    # same-dtype reference must be BF16. The checkpoint has no remote code;
    # the built-in transformers Gemma3 class loads it.
    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=f"rev-{HF_REVISION}",
    )

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
    # ``dtype="bfloat16"`` is model-specific: the checkpoint stores FP32
    # weights, the implementation supports only the BF16 served dtype, and
    # its weight load casts each tensor to it.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        block_size=32,
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
