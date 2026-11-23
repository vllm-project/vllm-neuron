# SPDX-License-Identifier: Apache-2.0
"""Integration test for the StarCoder2 contrib model, served through vLLM on Neuron.

The test serves ``bigcode/starcoder2-3b`` with the implementation in
``contrib/models/starcoder2/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM maps
``Starcoder2ForCausalLM`` to its own Transformers backend, and the call
replaces that entry.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test:
a spawned child re-imports from scratch and never runs the registration call.
It then resolves vLLM's Transformers backend class, which has no
``from_configs``, and the Neuron model runner fails with an ``AttributeError``.

Checkpoint
----------
Set ``STARCODER2_CHECKPOINT_PATH`` to a local copy of the checkpoint. Without
it, the test downloads the pinned revision (about 12 GB, FP32) from the
HuggingFace Hub. Both the reference and vLLM load the same directory.

Running
-------
The configuration below is the topology of the recorded logit run, TP 4, on
one trn2.48xlarge node (the recorded run spread the 4 ranks over 2 nodes):

    VLLM_NEURON_BACKEND=neuron_native \\
    pytest contrib/models/starcoder2/test/integration/test_model.py \\
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

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

from starcoder2_test_utils import (
    ARCHITECTURE,
    CHECKPOINT_ENV,
    HF_REVISION,
    MODEL_ID,
    TP_SIZE,
    src_package,
)

# The two prompts of the recorded logit run.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt, as in the recorded run.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map. The argmax (divergence) lower bound
# stays strict at 1e-5 — a nonzero divergence is a real correctness bug and
# must not be widened to hide it. The per-K upper bounds bound the allowed
# top-K logit drift.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# The tensor parallel size of the recorded run. 24 query heads and 2 KV heads
# give 6 query heads and 1 KV head per rank.
TENSOR_PARALLEL_SIZE = TP_SIZE

# The engine shape of the recorded run: a 2,048-token model length, one
# sequence, 1,024 batched tokens and prefill buckets of 256 and 1,024 tokens.
# The window of the checkpoint is 4,096 tokens, so this shape stays inside it.
MAX_NUM_BATCHED_TOKENS = 1024
NUM_BATCHED_TOKENS_BUCKETS = [256, 1024]

# Deterministic results across runs.
torch.manual_seed(42)


def _checkpoint() -> str:
    """Return the local checkpoint directory at the pinned revision."""
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=HF_REVISION)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM.

    The factory is registered as a class object, not as a ``"module:Class"``
    string: vLLM resolves a string registration by import, and a module that
    was loaded by file location cannot be imported by name.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


# The (batch_size, seq_len) pair of the recorded logit run.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
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
    # same-dtype reference must be BF16.
    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
    model_config.dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # The StarCoder2 tokenizer has no pad token, and the golden generator asks
    # for padding. Each prompt is a batch of one, so no pad token gets added.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=f"starcoder2-rev-{HF_REVISION}",
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
    # ``quantization: bf16`` is model-specific: the StarCoder2-3B checkpoint
    # stores FP32 weights and this implementation computes in BF16.
    #
    # The two bucket lists pin compilation to the shapes of the recorded run.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": NUM_BATCHED_TOKENS_BUCKETS,
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
