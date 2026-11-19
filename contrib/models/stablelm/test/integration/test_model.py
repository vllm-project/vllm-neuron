# SPDX-License-Identifier: Apache-2.0
"""Integration test for the StableLM contrib model, served through vLLM on Neuron.

The test serves ``stabilityai/stablelm-zephyr-3b`` with the implementation in
``contrib/models/stablelm/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its own
``StableLmForCausalLM``, so the registration replaces the built-in entry.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test.
If this folder is also installed (``pip install -e contrib/models/stablelm``),
its plugin entry point registers the same implementation in every process.

Checkpoint
----------
``STABLELM_CHECKPOINT_PATH`` points at a local copy of the checkpoint. Without
it, the test downloads the pinned revision (``HF_REVISION``) from the
HuggingFace hub, about 5.6 GB.

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    pytest contrib/models/stablelm/test/integration/test_model.py \
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence. ``_full_vocab_generate_fn``
wraps the vLLM ``LLM`` so that it returns dense per-token logits, and
``multi_prompt_logit_validation`` compares the two. It flags any argmax
divergence and any top-K logit error outside ``TOL_MAP``.
"""

import pytest
import torch

from _local_src import src_package
from stablelm_test_utils import ARCHITECTURE, HF_REVISION, MODEL_ID, resolve_checkpoint

# The two prompts of the port's logit test.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt, as in the port's logit
# test.
OUTPUT_LENGTH = 16

# The port's logit test passed with the default tolerance map of
# ``multi_prompt_logit_validation``
# (``vllm_neuron.accuracy.constants.DEFAULT_TOLERANCE_MAP``). This map keeps
# those bounds. They are tighter than the template's bf16 dense-golden map.
# The argmax (divergence) lower bound stays strict at 1e-5 — a nonzero
# divergence is a real correctness bug and must not be widened to hide it.
TOL_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

# 32 heads and 32 KV heads: TP 8 gives 4 of each per rank, the degree of the
# port's device tests.
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM.

    Pass the class object, not vLLM's ``"module:Class"`` string form: vLLM
    inspects a string registration by importing it in a fresh interpreter,
    which cannot import a package that ``_local_src`` loaded by file location.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _map_missing_token_strings(llm) -> None:
    """Map the vocabulary ids without a token string to "".

    The model scores 50,304 ids. The tokenizer has 50,277 tokens, so
    ``convert_ids_to_tokens`` returns None for the padding ids past it, and
    vLLM 0.29 crashes when it decodes the logprobs of those ids. The logit
    comparison reads the logits, never the strings.
    """
    tokenizer = llm.get_tokenizer()
    convert_ids_to_tokens = tokenizer.convert_ids_to_tokens

    def _convert_ids_to_tokens(ids, *args, **kwargs):
        tokens = convert_ids_to_tokens(ids, *args, **kwargs)
        if isinstance(tokens, list):
            return ["" if token is None else token for token in tokens]
        return "" if tokens is None else tokens

    tokenizer.convert_ids_to_tokens = _convert_ids_to_tokens


def _full_vocab_generate_fn(llm, output_length: int, vocab_size: int):
    """Return a ``generate_fn`` that packs the logits of every vocabulary id.

    This is the plugin's ``create_offline_vllm_generate_fn`` with one change.
    That helper requests ``tokenizer.vocab_size`` logprobs (50,254 here) and
    sizes each row from the largest id it gets back, so a row can miss ids and
    two rows can have different lengths. This function requests all
    ``vocab_size`` ids (``config.vocab_size``, 50,304) and gives every row
    that length, the width of the HF reference.

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


# The (batch_size, seq_len) pairs: the port's logit test ran one sequence at a
# 2048-token model length. Batch 4 runs the same prompts as four sequences.
# Each pair compiles its own graph.
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

    _register_model()
    checkpoint = resolve_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint stores BF16 weights, so the same-dtype reference is BF16.
    model_config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key; ``model_checkpoint`` is what
    # the reference model loads from. ``golden_variant`` keys the cache on the
    # pinned revision.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=f"rev-{HF_REVISION[:12]}",
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
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
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
    _map_missing_token_strings(llm)
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
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
