# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Bloom contrib model, served through vLLM on Neuron.

The test serves ``bigscience/bloomz-560m`` (pinned revision) with the
implementation in ``contrib/models/bloom/src/`` and compares the raw per-token
logits with a HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``.

The registration reaches vLLM's engine-core and worker processes by
inheritance: the offline ``LLM(...)`` path starts those processes with the
``fork`` start method, so they inherit this process's ``sys.modules`` and the
populated ``ModelRegistry``. vLLM ships its own GPU ``BloomForCausalLM``, so a
process that misses the registration does not report an unknown architecture:
it resolves vLLM's class and then fails on ``from_configs``. The test therefore
skips under ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` unless this folder is
installed with ``pip install -e contrib/models/bloom``, whose entry point
registers the model in every process.

Running
-------
The configuration below needs 2 NeuronCores on one Trn2 node:

    pytest contrib/models/bloom/test/integration/test_model.py \
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

Set ``BLOOM_CHECKPOINT_PATH`` to a local copy of the pinned revision to avoid
the Hub download.

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence. ``multi_prompt_logit_validation``
compares them with the served logits. It flags any argmax divergence and any
top-K logit error outside ``TOL_MAP``.
"""

import os
from importlib.metadata import entry_points

import pytest
import torch

import bloom_test_utils as U
from _local_src import src_package

# The ten prompts of the port's logit test (``PROMPTS_10`` of the plugin's
# shared test prompts), copied here so the folder is self-contained.
PROMPTS = [
    "The capital of France is Paris, which is known for its beautiful architecture, museums, and the iconic Eiffel Tower that attracts millions of tourists every year.",
    "In machine learning, a neural network is a computational model inspired by the human brain that consists of interconnected nodes organized in layers to process information.",
    "The theory of relativity was developed by Albert Einstein in the early twentieth century and fundamentally changed our understanding of space, time, and gravity in the universe.",
    "Python is a high-level programming language known for its simplicity and readability, making it popular among beginners and experienced developers alike for various applications.",
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning across nine countries and containing approximately ten percent of all species on Earth.",
    "Quantum computing leverages the principles of quantum mechanics to perform calculations that would be impossible for classical computers to solve in reasonable time.",
    "The Great Wall of China is one of the most impressive architectural feats in human history, stretching over thirteen thousand miles across northern China.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities such as burning fossil fuels.",
    "The human genome contains approximately three billion base pairs of DNA, encoding the genetic instructions for building and maintaining a human organism.",
    "Artificial intelligence has made remarkable progress in recent years, with applications ranging from natural language processing to autonomous vehicles and medical diagnosis.",
]

# Number of tokens to decode and validate per prompt (the port's logit test
# used 16).
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

# The tensor-parallel degree of the port's logit run (there split over two
# nodes, one rank each). 16 heads / 2 ranks gives 8 heads per rank.
TENSOR_PARALLEL_SIZE = 2

# The HF Bloom classes do not support SDPA, so the reference loads with eager
# attention. The variant keeps those goldens apart from other cache entries.
GOLDEN_VARIANT = f"eager-rev-{U.HF_REVISION[:7]}"

# Deterministic results across runs.
torch.manual_seed(42)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(U.ARCHITECTURE, getattr(src_package(), U.ARCHITECTURE))


def _require_registration_reaches_engine() -> None:
    """Skip unless vLLM's child processes see this folder's registration."""
    method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if method == "fork":
        return
    installed = any(
        ep.name == "bloom_neuron" for ep in entry_points(group="vllm.general_plugins")
    )
    if not installed:
        pytest.skip(
            f"VLLM_WORKER_MULTIPROC_METHOD={method!r}: a spawned child does not "
            "inherit the in-process registration and would resolve vLLM's own "
            "BloomForCausalLM. Use the fork start method, or install this "
            "folder with `pip install -e contrib/models/bloom`."
        )


def _load_hf_reference(model_checkpoint: str, dtype: torch.dtype, config=None):
    """Load the HF reference with eager attention (HF Bloom has no SDPA)."""
    from vllm_neuron.accuracy.goldens.reference_model import init_hf_model

    return init_hf_model(
        model_checkpoint, dtype, config=config, attn_implementation="eager"
    )


def _full_vocab_generate_fn(llm, output_length: int, vocab_size: int):
    """Return a ``generate_fn`` that packs the logits of every vocabulary id.

    This replaces the plugin's ``create_offline_vllm_generate_fn`` for one
    reason: the model scores 250,880 ids, but the tokenizer has 250,680. The
    plugin helper requests ``tokenizer.vocab_size`` logprobs and sizes each row
    from the largest id it gets back, so the rows get different lengths and
    ``torch.stack`` fails. vLLM 0.29 also crashes when it detokenizes one of
    the 200 padding ids, because the tokenizer has no string for them. This
    function requests all ``vocab_size`` ids with ``detokenize=False`` (the
    comparison reads logits, never strings) and builds each row with that
    length, the width of the HF reference. ``ignore_eos=True`` decodes exactly
    ``output_length`` tokens, as the HF reference does: bloomz often emits EOS
    early.

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
            temperature=0,
            max_tokens=output_length,
            logprobs=vocab_size,
            detokenize=False,
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, sampling_params)

        batch_logits = []
        batch_tokens = []
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.logprobs) == output_length, (
                f"{len(completion.logprobs)} decode steps, expected {output_length}."
            )
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


# The (batch_size, seq_len) pairs: batch 1 at the 2048-token model length of
# the port's logit run, and batch 4 at the same length. Each pair compiles its
# own graph.
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

    _require_registration_reaches_engine()
    _register_model()

    checkpoint = U.get_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint stores FP16 weights and the engine computes in BF16, so
    # the same-dtype reference must be BF16.
    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key; ``model_checkpoint`` is
    # what the reference model loads from.
    cached_goldens = get_cached_reference_goldens(
        model_id=U.MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        model_loader=_load_hf_reference,
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
    # ``quantization: bf16`` matches the port's serving runs: the checkpoint
    # stores FP16 weights and this implementation computes in BF16.
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
                "quantization": "bf16",
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
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
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
