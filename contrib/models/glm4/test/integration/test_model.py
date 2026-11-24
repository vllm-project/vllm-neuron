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

Long prompts
------------
The test runs at a 9216-token model length (8192 prompt tokens plus 1024
generated tokens) with two prompts of 8000 to 8192 tokens each. The prompts
are built in this file from fixed text: a prose document whose sections repeat
in a fixed order, and a Python module of small generated functions. Each one
is cut at a sentence or function boundary to the longest prefix that the
model's own tokenizer encodes in at most 8192 tokens. The test checks the
token counts of the ids that the reference and vLLM both get.

Prefill runs in 1024-token chunks (segmented prefill), so an 8192-token prompt
takes 8 prefill steps, and every chunk after the first attends to the K/V of
the earlier chunks in the paged KV cache.

Running
-------
The configuration below needs 8 NeuronCores (LNC 2) on one trn2.48xlarge
node:

    pytest contrib/models/glm4/test/integration/test_model.py \
        -v -s --timeout=14400 > /tmp/test_output.txt 2>&1

A first run computes the HuggingFace references on the host CPU and compiles
the 61-layer graphs, which takes longer than the vLLM and compiler default
timeouts. The test raises those timeouts (see ``_COMPILE_TIMEOUTS``) unless
they are already set in the environment.

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

# Model length of both cases: room for an 8192-token prompt and 1024 generated
# tokens. It is a multiple of the 1024-token prefill chunk and of the KV block
# size (32).
MAX_MODEL_LEN = 9216

# Token count range of each long prompt, as the reference tokenizes it.
MIN_PROMPT_TOKENS = 8000
MAX_PROMPT_TOKENS = 8192

# Number of tokens to decode and validate per prompt.
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
# head (each KV head replicated on 4 ranks). The serve example in README.md
# uses the same TP 8.
TENSOR_PARALLEL_SIZE = 8

# The prefill chunk. It is smaller than the model length and is a supported KV
# segment size, so vLLM Neuron turns segmented (chunked) prefill on with one
# 1024-token prefill graph and a 1024-token KV segment.
MAX_NUM_BATCHED_TOKENS = 1024

# The first start compiles the 61-layer graphs, and vLLM's defaults (600 s for
# the engine start, 600 s for the compile of one graph, 300 s for one model
# execution) are too short for that. These values cover the 9216-token
# configuration.
_COMPILE_TIMEOUTS = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "5400",
    "NEURON_LIBTORCH_COMPILATION_TIMEOUT": "2400",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600",
}

# Deterministic results across runs.
torch.manual_seed(42)


# =============================================================================
# Long prompts
# =============================================================================

# The sections of the prose prompt. The document repeats them in this order
# with increasing section numbers, so the text at the cut point continues a
# passage that appears earlier in the prompt, about 700 tokens back.
_SECTIONS = (
    (
        "Tropical rainforests",
        "Tropical rainforests grow in a belt near the equator and receive heavy "
        "rain all year.",
        "They cover a small part of the land but hold more than half of the "
        "known plant and animal species.",
        "Their trees release water vapor that forms clouds and carries rain far "
        "inland.",
        "Scientists study these forests to understand how carbon moves between "
        "the land and the air.",
        "Rivers that wind through the forests carry sediment, fish and seeds "
        "over thousands of kilometers.",
    ),
    (
        "The human genome",
        "The human genome contains about three billion base pairs of DNA.",
        "These pairs encode the instructions for building and maintaining a "
        "human body.",
        "Only a small part of the sequence codes for proteins, while the rest "
        "helps control when genes are active.",
        "Researchers compare genomes to trace ancestry and to find variants "
        "linked to disease.",
        "Sequencing a genome once took years, and today it can be done in about a day.",
    ),
    (
        "Ocean currents",
        "Ocean currents move heat from the equator toward the poles.",
        "Wind, the rotation of the Earth and differences in salt and "
        "temperature drive them.",
        "The Gulf Stream carries warm water along the coast of North America "
        "and across the Atlantic.",
        "Deep currents sink near the poles and rise again in other basins over "
        "centuries.",
        "Changes in these currents can shift rainfall patterns on several continents.",
    ),
    (
        "Volcanoes",
        "A volcano forms where molten rock reaches the surface of the Earth.",
        "Many volcanoes sit along the edges of tectonic plates, where one plate "
        "slides beneath another.",
        "Some eruptions release slow rivers of lava, while others throw ash "
        "high into the sky.",
        "Volcanic soil is rich in minerals, so farms often grow on the slopes "
        "of old volcanoes.",
        "Monitoring stations measure ground movement and gas to warn nearby "
        "towns before an eruption.",
    ),
    (
        "Bridges",
        "Engineers choose the shape of a bridge from the distance it must span "
        "and the load it must carry.",
        "Arch bridges push their weight outward into the ground at each end.",
        "Suspension bridges hang the deck from cables that run over tall towers.",
        "Wind and temperature change the length of the steel, so joints let "
        "the deck expand and contract.",
        "Regular inspections look for cracks, corrosion and worn bearings.",
    ),
    (
        "The printing press",
        "The printing press made it possible to copy text faster than any scribe.",
        "Movable metal type let printers arrange letters, print a page and "
        "reuse the letters for the next page.",
        "Books became cheaper, and more people learned to read.",
        "Printed pamphlets spread news and ideas across cities within weeks.",
        "Libraries grew as printers produced the same work in hundreds of "
        "identical copies.",
    ),
    (
        "Honeybees",
        "A honeybee colony can hold tens of thousands of workers and a single queen.",
        "Workers collect nectar and pollen and turn the nectar into honey "
        "stored in wax cells.",
        "A forager that finds flowers performs a dance that tells other bees "
        "the direction and distance.",
        "Bees pollinate many crops, including apples, almonds and berries.",
        "Beekeepers check their hives for disease and make sure the colony has "
        "enough food for winter.",
    ),
    (
        "Glaciers",
        "A glacier is a slow river of ice that forms where more snow falls than melts.",
        "The weight of the ice makes it flow downhill a few meters or more each day.",
        "As it moves, the glacier grinds rock into fine powder and carves deep "
        "valleys.",
        "Meltwater from glaciers feeds rivers that supply water to farms and cities.",
        "Scientists measure the thickness of glaciers to track how the climate "
        "changes.",
    ),
)

# The four function shapes of the code prompt, used in turn. ``{n}`` is the
# running function number.
_FUNCTIONS = (
    'def scale_{n:03d}(values):\n    """Return each value multiplied by {n}."""\n'
    "    return [value * {n} for value in values]",
    'def shift_{n:03d}(values):\n    """Return each value increased by {n}."""\n'
    "    return [value + {n} for value in values]",
    'def clip_{n:03d}(values):\n    """Return each value limited to at most {n}."""\n'
    "    return [min(value, {n}) for value in values]",
    "def count_above_{n:03d}(values):\n"
    '    """Return how many values are greater than {n}."""\n'
    "    return sum(1 for value in values if value > {n})",
)


def _prose_pieces(num_sections: int) -> list[str]:
    """Return the sentences of the prose prompt, one piece per sentence.

    The first sentence of each section carries the section heading, so every
    prefix of the pieces ends at the end of a sentence.
    """
    pieces = []
    for index in range(num_sections):
        title, first, *rest = _SECTIONS[index % len(_SECTIONS)]
        heading = f"Section {index + 1}. {title}\n\n"
        pieces.append(("\n\n" if index else "") + heading + first)
        pieces.extend(" " + sentence for sentence in rest)
    return pieces


def _code_pieces(num_functions: int) -> list[str]:
    """Return the code prompt, one piece per function after a module docstring.

    Every prefix of the pieces ends at the last line of a function.
    """
    pieces = [
        '"""Helpers for lists of numbers.\n\n'
        "Each function takes a list of numbers and applies one fixed "
        'operation.\n"""'
    ]
    for n in range(1, num_functions + 1):
        pieces.append("\n\n\n" + _FUNCTIONS[n % len(_FUNCTIONS)].format(n=n))
    return pieces


def _num_prompt_tokens(tokenizer, prompt: str) -> int:
    """Return the number of ids the golden reference gets for ``prompt``.

    ``get_cached_reference_goldens`` tokenizes each prompt with this call, and
    the vLLM side gets the same ids.
    """
    return len(tokenizer([prompt], padding=True, truncation=True)["input_ids"][0])


def _fit_prompt(tokenizer, pieces: list[str]) -> str:
    """Join the longest prefix of ``pieces`` with at most MAX_PROMPT_TOKENS tokens."""
    assert _num_prompt_tokens(tokenizer, "".join(pieces)) > MAX_PROMPT_TOKENS, (
        "The prompt text is too short to fill the token budget."
    )
    low, high = 1, len(pieces)
    while low < high:
        mid = (low + high + 1) // 2
        if _num_prompt_tokens(tokenizer, "".join(pieces[:mid])) <= MAX_PROMPT_TOKENS:
            low = mid
        else:
            high = mid - 1
    return "".join(pieces[:low])


def build_long_prompts(tokenizer) -> list[str]:
    """Return the two long prompts, each 8000 to 8192 tokens.

    Args:
        tokenizer: The tokenizer of the checkpoint.

    Returns:
        The prose prompt and the code prompt.
    """
    prompts = [
        _fit_prompt(tokenizer, _prose_pieces(num_sections=120)),
        _fit_prompt(tokenizer, _code_pieces(num_functions=400)),
    ]
    for prompt in prompts:
        num_tokens = _num_prompt_tokens(tokenizer, prompt)
        assert MIN_PROMPT_TOKENS <= num_tokens <= MAX_PROMPT_TOKENS, (
            f"A long prompt has {num_tokens} tokens, expected "
            f"{MIN_PROMPT_TOKENS} to {MAX_PROMPT_TOKENS}."
        )
    return prompts


# =============================================================================
# Test
# =============================================================================


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _full_vocab_generate_fn(llm, output_length: int, vocab_size: int):
    """Return a ``generate_fn`` that packs the logits of every vocabulary id.

    This is the plugin's ``create_offline_vllm_generate_fn`` with two changes.
    That helper requests ``tokenizer.vocab_size`` logprobs and sizes each row
    from the largest id it gets back. The GLM-4 tokenizer has 151,329 base
    tokens (151,343 with the special tokens), and the model scores 151,552 ids,
    so each step would return a different top-k subset and rows of different
    lengths. This function requests all ``vocab_size`` ids and builds each row
    with that length, the width of the HF reference. The padding ids past the
    tokenizer decode to an empty string, so vLLM's logprob detokenizer accepts
    them. It also sets ``ignore_eos``: the HF reference always decodes
    ``output_length`` tokens, so vLLM must not stop early at an end-of-sequence
    id. ``multi_prompt_logit_validation`` and ``TOL_MAP`` do not change.

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
            ignore_eos=True,
            logprobs=vocab_size,
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


# Batch 1 and batch 4 at the 9216-token model length. Batch 4 runs four copies
# of each prompt and checks the batched decode graph. Each pair compiles its
# own graphs.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, MAX_MODEL_LEN),
        (4, MAX_MODEL_LEN),
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

    # Compute (or load from cache) the HuggingFace reference logits for the
    # long prompts. The checkpoint stores BF16 weights, so the same-dtype
    # reference is BF16. The reference runs on the host CPU: the FP32 model
    # (about 130 GB) and then the BF16 model (about 65 GB), one at a time, with
    # SDPA attention, one prompt at a time.
    model_config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = build_long_prompts(tokenizer)
    # ``model_id`` only builds the golden cache key; ``model_checkpoint`` is what
    # the reference model loads from. ``golden_variant`` ties the cached goldens
    # to the pinned checkpoint revision.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=prompts,
        output_length=OUTPUT_LENGTH,
        golden_variant=GOLDEN_VARIANT,
    )
    prompts_input_ids = [
        [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
    ]
    for prompt_ids in prompts_input_ids:
        num_tokens = len(prompt_ids[0])
        assert MIN_PROMPT_TOKENS <= num_tokens <= MAX_PROMPT_TOKENS, (
            f"A reference prompt has {num_tokens} tokens, expected "
            f"{MIN_PROMPT_TOKENS} to {MAX_PROMPT_TOKENS}."
        )
        assert num_tokens + OUTPUT_LENGTH <= seq_len

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
