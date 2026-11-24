# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Qwen3-MoE contrib model, served through vLLM on Neuron.

The test serves ``Qwen/Qwen3-30B-A3B`` with the implementation in
``contrib/models/qwen3_moe/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its
own GPU ``Qwen3MoeForCausalLM``; the registration replaces it.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test:
a spawned child re-imports from scratch and resolves vLLM's own
``Qwen3MoeForCausalLM`` instead of this folder's. If this folder is also
installed (``pip install -e contrib/models/qwen3_moe``), its plugin entry point
registers the same implementation in every process.

Checkpoint
----------
``QWEN3_MOE_CHECKPOINT_PATH`` points at a local copy of the pinned revision.
Without it, the test downloads the pinned revision (about 61 GB) from the
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

Prefill runs in 2048-token chunks (segmented prefill with a 2048-token KV
segment), so an 8192-token prompt takes 4 prefill steps, and every chunk after
the first attends to the K/V of the earlier chunks in the paged KV cache.

Running
-------
The configuration below needs 16 NeuronCores (LNC 2) on one trn2.48xlarge
node:

    pytest contrib/models/qwen3_moe/test/integration/test_model.py \\
        -v -s --timeout=18000 > /tmp/test_output.txt 2>&1

A first run computes the HuggingFace references on the host CPU and compiles
the 48-layer graphs, which takes longer than the vLLM and compiler default
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
from qwen3_moe_test_utils import (
    ARCHITECTURE,
    GOLDEN_VARIANT,
    MODEL_ID,
    resolve_checkpoint,
)

# Model length of both cases: room for an 8192-token prompt and 1024 generated
# tokens. It is a multiple of the KV block size.
MAX_MODEL_LEN = 9216

# Token count range of each long prompt, as the reference tokenizes it.
MIN_PROMPT_TOKENS = 8000
MAX_PROMPT_TOKENS = 8192

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map of the contrib template. The argmax
# (divergence) lower bound stays strict at 1e-5 — a nonzero divergence is a
# real correctness bug and must not be widened to hide it. The per-K upper
# bounds bound the allowed top-K logit drift.
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

# 32 query heads and 4 KV heads: TP 16 gives each rank 2 query heads and one
# KV head (each KV head replicated on 4 ranks). Expert parallelism over the
# same 16 ranks gives each rank 8 complete experts of the 128. The serve
# example in README.md uses the same TP 16 with --enable-expert-parallel.
TENSOR_PARALLEL_SIZE = 16

# The prefill chunk: the batched-token limit, the one prefill bucket and the
# one KV segment size. It is smaller than the model length and is a supported
# KV segment size, so vLLM Neuron runs segmented (chunked) prefill.
MAX_NUM_BATCHED_TOKENS = 2048
KV_SEGMENT_SIZE = 2048

# The first start compiles the 48-layer graphs, and vLLM's defaults (600 s for
# the engine start, 600 s for the compile of one graph, 300 s for one model
# execution) are too short for that.
_COMPILE_TIMEOUTS = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
    "NEURON_LIBTORCH_COMPILATION_TIMEOUT": "3600",
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
        "Coral reefs",
        "Coral reefs grow in warm, shallow seas where sunlight reaches the sea floor.",
        "Each reef is built by colonies of tiny animals that lay down a "
        "skeleton of limestone.",
        "Algae that live inside the corals turn sunlight into food and give "
        "the reef its colors.",
        "Thousands of kinds of fish, crabs and sponges shelter among the "
        "branches of a healthy reef.",
        "When the water stays too warm for weeks, the corals expel their "
        "algae and turn white.",
    ),
    (
        "Railways",
        "The first public railways carried coal from mines to rivers and ports.",
        "Steam engines soon pulled passenger carriages between growing "
        "industrial towns.",
        "A standard distance between the rails let trains of different "
        "companies share the same track.",
        "Signals and timetables kept trains apart on lines that carried "
        "traffic in both directions.",
        "Electric trains later replaced steam on many busy routes because "
        "they accelerate faster.",
    ),
    (
        "Deserts",
        "A desert is a region that receives very little rain in a typical year.",
        "Some deserts are hot and sandy, while others are cold and covered "
        "with stones.",
        "Plants in a desert store water in thick stems or send long roots "
        "deep into the ground.",
        "Many desert animals stay underground during the day and come out "
        "to feed at night.",
        "Strong winds shape the sand into dunes that move slowly across the land.",
    ),
    (
        "Telescopes",
        "A telescope collects light from distant objects and brings it to a focus.",
        "Lens telescopes bend light with glass, and mirror telescopes "
        "reflect it with curved surfaces.",
        "A larger opening gathers more light, so faint galaxies become visible.",
        "Telescopes in orbit avoid the blurring caused by the moving air "
        "above the ground.",
        "Radio telescopes use large dishes to detect waves that the eye cannot see.",
    ),
    (
        "Bread",
        "Bread is made from flour, water, salt and a leavening agent such as yeast.",
        "Kneading the dough builds a network of gluten that traps the gas "
        "released by the yeast.",
        "The dough rises for an hour or more before it is shaped into loaves.",
        "In a hot oven the crust browns while the inside sets into a soft crumb.",
        "Bakers in different regions use rye, wheat, barley or corn to make "
        "their own kinds of bread.",
    ),
    (
        "Rivers",
        "A river begins at a spring, a lake or a melting glacier high in the hills.",
        "Small streams join together as the water flows downhill toward the sea.",
        "The moving water carries sand and mud and drops it where the current slows.",
        "Floods spread this soil across the valley floor and make the land fertile.",
        "Near the coast many rivers split into channels that form a wide delta.",
    ),
    (
        "Clocks",
        "Early clocks used falling water or burning candles to mark the hours.",
        "Mechanical clocks with gears and weights appeared in the towers of "
        "medieval towns.",
        "A swinging pendulum made clocks far more accurate, because each "
        "swing takes the same time.",
        "Quartz crystals vibrate at a steady rate and keep the time in most "
        "modern watches.",
        "Atomic clocks measure the vibrations of atoms and lose less than a "
        "second in millions of years.",
    ),
    (
        "Forests in winter",
        "In cold climates many trees drop their leaves before winter arrives.",
        "The trees slow their growth and store sugar in their roots and stems.",
        "Evergreen trees keep narrow needles that lose little water in the "
        "frozen months.",
        "Snow covers the ground and protects small plants and animals from "
        "the coldest air.",
        "In spring the days grow longer, the sap rises and new buds open on "
        "the branches.",
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
    """Load the factory from ``src/`` and register it with vLLM.

    Pass the class object, not vLLM's ``"module:Class"`` string form: vLLM
    inspects a string registration by importing it in a fresh interpreter,
    which cannot import a package that ``_local_src`` loaded by file location.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _map_missing_token_strings(llm) -> None:
    """Map the vocabulary ids without a token string to "".

    The model scores 151,936 ids. The tokenizer has 151,669 tokens (151,643
    plus 26 special tokens), so ``convert_ids_to_tokens`` returns None for the
    padding ids past it, and vLLM 0.29 fails when it decodes the logprobs of
    those ids. The logit comparison reads the logits, never the strings.
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

    This is the plugin's ``create_offline_vllm_generate_fn`` with two changes.
    That helper requests ``tokenizer.vocab_size`` logprobs (151,643 here) and
    sizes each row from the largest id it gets back, so a row can miss ids and
    two rows can have different lengths. This function requests all
    ``vocab_size`` ids (``config.vocab_size``, 151,936) and gives every row
    that length, the width of the HF reference. It also sets ``ignore_eos``:
    the HF reference always decodes ``output_length`` tokens, so vLLM must not
    stop early at an end-of-sequence id. ``multi_prompt_logit_validation`` and
    ``TOL_MAP`` do not change.

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
    # (about 122 GB) and then the BF16 model (about 61 GB), one at a time,
    # with SDPA attention, one prompt at a time.
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
    # enable_expert_parallel=True places 8 complete experts on each of the 16
    # ranks. The three bucket lists pin compilation to the one prefill shape (a
    # 2048-token chunk with 2048-token KV segments) and the one decode batch of
    # this case.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_expert_parallel=True,
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
                "kv_segment_size_buckets": [KV_SEGMENT_SIZE],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    _map_missing_token_strings(llm)
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
