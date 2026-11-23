# SPDX-License-Identifier: Apache-2.0
"""Integration test for the GLM-4-0414 contrib model, served through vLLM on Neuron.

The test serves ``zai-org/GLM-4-32B-0414`` with the implementation in
``contrib/models/glm4/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its
own GPU ``Glm4ForCausalLM``; the registration replaces it.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test:
a spawned child re-imports from scratch and resolves vLLM's own
``Glm4ForCausalLM`` instead of this folder's.

Checkpoint
----------
``GLM4_0414_CHECKPOINT_PATH`` points at a local copy of the pinned revision.
Without it, the test downloads the pinned revision (about 65 GB) from the
HuggingFace hub.

Running
-------
The configuration below needs 8 NeuronCores (LNC 2) on one trn2.48xlarge
node:

    pytest contrib/models/glm4/test/integration/test_model.py \
        -v -s --timeout=14400 > /tmp/test_output.txt 2>&1

A first run compiles the 61-layer graphs, which takes longer than the vLLM
and compiler default timeouts. The test raises those timeouts (see
``_COMPILE_TIMEOUTS``) unless they are already set in the environment.

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence. ``_full_vocab_generate_fn``
wraps the vLLM ``LLM`` so that it returns dense per-token logits over the whole
model vocabulary, and ``multi_prompt_logit_validation`` compares the two. It
flags any argmax divergence and any top-K logit error outside ``TOL_MAP``.
"""

import os

import pytest
import torch

from _local_src import src_package
from glm4_test_utils import (
    ARCHITECTURE,
    GOLDEN_VARIANT,
    MODEL_ID,
    resolve_checkpoint,
)

# The two prompts of the port's logit test.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt (the port's logit test
# used 16).
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

# Per-prompt checks of the three-way validation. Each one compares the
# largest error of one prompt against 1.5x the largest error of the CPU BF16
# reference for the same prompt.
PER_PROMPT_CHECKS = ("pp_max_linf_1.5x", "pp_max_l2_1.5x")

# 48 query heads and 2 KV heads: TP 8 gives each rank 6 query heads and one KV
# head (each KV head replicated on 4 ranks). TP 8 is the degree of every
# recorded device run of this model.
TENSOR_PARALLEL_SIZE = 8

# The prefill size of the recorded serving runs. It is smaller than the model
# length, so prefill runs as segmented (chunked) prefill with one 1024-token
# graph.
MAX_NUM_BATCHED_TOKENS = 1024

# The recorded runs raised these timeouts for the 61-layer model: the engine
# start (which includes the graph compilation), the compile of one graph, and
# one model execution. vLLM's defaults are 600 s, 600 s and 300 s.
_COMPILE_TIMEOUTS = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "5400",
    "NEURON_LIBTORCH_COMPILATION_TIMEOUT": "2400",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600",
}

# Deterministic results across runs.
torch.manual_seed(42)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _full_vocab_generate_fn(llm, output_length: int, vocab_size: int):
    """Return a ``generate_fn`` that packs the logits of every vocabulary id.

    This is the plugin's ``create_offline_vllm_generate_fn`` with one change.
    That helper requests ``tokenizer.vocab_size`` logprobs and sizes each row
    from the largest id it gets back. The GLM-4 tokenizer has 151,329 base
    tokens (151,343 with the special tokens), and the model scores 151,552 ids,
    so each step would return a different top-k subset and rows of different
    lengths. This function requests all ``vocab_size`` ids and builds each row
    with that length, the width of the HF reference. The padding ids past the
    tokenizer decode to an empty string, so vLLM's logprob detokenizer accepts
    them. ``multi_prompt_logit_validation`` and ``TOL_MAP`` do not change.

    Args:
        llm: The running vLLM ``LLM``.
        output_length: The number of tokens to decode for each prompt.
        vocab_size: The vocabulary size of the model (``config.vocab_size``).

    Returns:
        ``generate_fn(input_ids) -> (logits, sampled_token_ids)``. ``logits``
        has the shape ``[output_length, num_prompts, vocab_size]``.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def generate_fn(input_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
        sampling_params = SamplingParams(
            temperature=0, max_tokens=output_length, logprobs=vocab_size
        )
        outputs = llm.generate(prompts, sampling_params)

        batch_logits = []
        batch_tokens = []
        for output in outputs:
            completion = output.outputs[0]
            rows = []
            for step, logprob_dict in enumerate(completion.logprobs):
                assert len(logprob_dict) == vocab_size, (
                    f"Step {step} returned {len(logprob_dict)} logprobs, expected "
                    f"all {vocab_size}. Check max_logprobs=-1."
                )
                row = torch.full((vocab_size,), -float("inf"))
                row[torch.tensor(list(logprob_dict.keys()))] = torch.tensor(
                    [lp.logprob for lp in logprob_dict.values()]
                )
                rows.append(row)
            batch_logits.append(torch.stack(rows, dim=0))
            batch_tokens.append(
                torch.tensor(completion.token_ids[:output_length], dtype=torch.long)
            )
        return torch.stack(batch_logits, dim=1), torch.stack(batch_tokens, dim=0)

    return generate_fn


# The recorded runs served batch 1 at a 2048-token model length. Batch 4 at the
# same length checks the batched decode graph. Each pair compiles its own graphs.
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
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    for name, value in _COMPILE_TIMEOUTS.items():
        os.environ.setdefault(name, value)

    _register_model()
    checkpoint = resolve_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint stores BF16 weights, so the same-dtype reference is BF16.
    model_config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key; ``model_checkpoint`` is what
    # the reference model loads from. ``golden_variant`` ties the cached goldens
    # to the pinned checkpoint revision.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=GOLDEN_VARIANT,
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
    # The two bucket lists pin compilation to the one prefill shape and the one
    # decode batch this case uses.
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
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [MAX_NUM_BATCHED_TOKENS],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        generate_fn = _full_vocab_generate_fn(
            llm, OUTPUT_LENGTH, model_config.vocab_size
        )
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
        n_prompts = len(result.per_prompt_results)
        for check in PER_PROMPT_CHECKS:
            value = result.aggregate_metrics.get(check)
            assert value == f"{n_prompts}/{n_prompts}", (
                f"{check} is {value} for parameters: {batch_size=}, {seq_len=}"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
