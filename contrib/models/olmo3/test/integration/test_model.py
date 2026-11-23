# SPDX-License-Identifier: Apache-2.0
"""Integration test for the OLMo-3 contrib model, served through vLLM on Neuron.

The test serves ``allenai/Olmo-3-7B-Instruct`` with the implementation in
``contrib/models/olmo3/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its
own ``Olmo3ForCausalLM`` (the Transformers backend); the registration replaces
it.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test.
With ``pip install -e contrib/models/olmo3`` the plugin entry point registers
the architecture in every process instead, and the test still works.

Running
-------
The configuration below needs 16 NeuronCores on one trn2.48xlarge node:

    OLMO3_CHECKPOINT_PATH=/path/to/Olmo-3-7B-Instruct \
    pytest contrib/models/olmo3/test/integration/test_model.py \
        -v -s --timeout=14400 > /tmp/test_output.txt 2>&1

Without ``OLMO3_CHECKPOINT_PATH`` the test downloads the pinned revision from
the Hugging Face Hub.

Model-specific settings
-----------------------
* ``optimization_level=3``: the implementation is validated at neuronx-cc
  ``-O3``. At the default ``-O2`` the compiled full model showed a large error
  at layers 20, 21 and 24, although each of those layers is correct when it is
  compiled alone. The vLLM optimization level maps 1:1 onto the neuronx-cc
  level, and an explicit level applies to every graph.
* ``max_model_len=2048``: the length the recorded device run used. The
  sliding window (4096 tokens, 24 of the 32 layers) is longer than this
  context, so here every layer attends over the full context, the same as the
  HF reference. The window is applied by the attention mask; the CPU tests in
  ``test/unit`` check it with a reduced window.

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

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
# The HuggingFace revision of the recorded runs.
HF_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
CHECKPOINT_ENV = "OLMO3_CHECKPOINT_PATH"

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

# The tensor parallel degree of the recorded runs. 32 Q/KV heads: TP 2, 4, 8
# and 16 are legal (the implementation needs an even number of Q heads on
# each rank).
TENSOR_PARALLEL_SIZE = 16

# neuronx-cc -O3 for every graph (see the module docstring).
OPTIMIZATION_LEVEL = 3

# Deterministic results across runs.
torch.manual_seed(42)

ARCHITECTURE = "Olmo3ForCausalLM"


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


# Batch 1 is the recorded shape; batch 4 also runs the batched decode
# attention. Each pair compiles its own graphs.
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
    # The checkpoint stores BF16 weights, so the same-dtype reference is BF16.
    # The checkpoint has no remote code; the built-in transformers class loads it.
    model_config = AutoConfig.from_pretrained(checkpoint)
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
    # ``optimization_level=3`` is model-specific (see the module docstring).
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        optimization_level=OPTIMIZATION_LEVEL,
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
