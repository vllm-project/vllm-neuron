# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Ouro contrib model, served through vLLM on Neuron.

The test serves ``ByteDance/Ouro-1.4B`` with the implementation in
``contrib/models/ouro/src/`` and compares the raw per-token logits with a
reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. Loading ``src/``
also registers the ``"ouro"`` model type with transformers' ``AutoConfig``
(``src/hf_config.py``), so vLLM and the tokenizer parse the checkpoint config
without ``trust_remote_code``.

The registrations have to reach vLLM's engine-core and worker processes, and
they reach them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules``, the populated ``ModelRegistry`` and the ``AutoConfig``
registration. That is the default for a Python script. Do not set
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test. With
``pip install -e contrib/models/ouro`` the plugin entry point registers both
in every process instead, and the test still works.

Long prompts and chunked prefill
--------------------------------
Each case runs at a maximum model length of 9216 tokens (8192 prompt tokens
plus 1024 generated tokens). The two prompts are built in this file from
fixed text and cut to ``PROMPT_TOKENS`` tokens with the checkpoint's own
tokenizer, so they are the same on every run; the test checks that each one
re-tokenizes to 8000-8192 tokens. The first prompt is a story, ledger lines
and the story again, and ends inside the second copy, so its continuation
repeats text about 7800 tokens back. The second prompt is ledger lines with
a short repeating pattern.

Prefill runs in chunks of ``MAX_NUM_BATCHED_TOKENS`` tokens. With
``kv_segment_size_buckets`` set, every chunk runs ``NF.segmented_attention``,
and each chunk after the first attends to the KV cache that the earlier
chunks wrote. This is the path that ``vllm serve`` takes for a long prompt
with the flags in the README.

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
    pytest contrib/models/ouro/test/integration/test_model.py \
        -v -s --timeout=14400 > /tmp/test_output.txt 2>&1

Without ``OURO_CHECKPOINT_PATH`` the test downloads the pinned revision from
the Hugging Face Hub. The reference logits run on the host CPU, once in FP32
and once in BF16 for each of the two 8100-token prompts, which takes tens of
minutes; the first case computes them and the second reads them from the
golden cache.

Logit validation
----------------
transformers does not include the Ouro architecture, and the checkpoint's
remote modeling code is written for an older transformers. The reference is
therefore ``test/ouro_reference.py``: an independent torch implementation of
the HF ``modeling_ouro.py`` rounding order that reads the safetensors
weights directly. ``get_cached_reference_goldens`` runs it through
``model_loader=`` and caches the logits: ``fp32_logits`` is the FP32
baseline, and ``dtype_logits`` is a BF16 pass teacher-forced along the FP32
greedy sequence. ``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM``
so that it returns dense per-token logits, and
``multi_prompt_logit_validation`` compares the two. It flags any argmax
divergence and any top-K logit error outside ``TOL_MAP``.
"""

import os

import pytest
import torch

from _local_src import src_package
from ouro_reference import load_ouro_reference
from ouro_test_utils import (
    ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    resolve_ouro_checkpoint,
)

# Maximum model length of both cases: 8192 prompt tokens plus 1024 generated
# tokens, the configuration of the serve command in the README.
MAX_MODEL_LEN = 9216

# Each prompt is cut to PROMPT_TOKENS tokens, and must re-tokenize to
# MIN_PROMPT_TOKENS-MAX_PROMPT_TOKENS tokens. 8100 is not a multiple of the
# 1024-token prefill chunk, so the last chunk of each prompt is a partial one.
PROMPT_TOKENS = 8100
MIN_PROMPT_TOKENS = 8000
MAX_PROMPT_TOKENS = 8192

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# Fixed text of the two prompts: a story and numbered ledger lines.
_STORY = (
    "The lighthouse on the northern cape was built from grey stone that the "
    "people of the harbour town carried up the cliff path one cart at a time. "
    "It took four summers to raise the tower, because the winter storms closed "
    "the path every year from the first frost until the thaw. When the lamp "
    "was lit for the first time, the whole town climbed the hill to watch the "
    "beam sweep across the water, and the fishing boats that were still out at "
    "sea turned toward it and came home safely before dark.\n\n"
    "The first keeper was a carpenter named Tomas, who had helped to build the "
    "stairs inside the tower. He kept a logbook on a shelf beside the lamp and "
    "wrote in it every evening: the direction of the wind, the height of the "
    "waves, the ships he had seen and the hour at which he had trimmed the "
    "wick. Over thirty years he filled eleven books, and the town still keeps "
    "them in a glass case in the library, next to the old map of the coast.\n\n"
    "The lamp burned whale oil at first, and later paraffin. Every morning "
    "Tomas polished the brass and cleaned the soot from the glass with a soft "
    "cloth, because a dirty lens could halve the reach of the light. On clear "
    "nights the beam could be seen from twenty miles away, and sailors said "
    "that they could set their course by it as surely as by the north star.\n\n"
    "In the great storm of the ninth winter, the wind broke two of the tower "
    "windows and the rain put out the lamp twice before midnight. Each time, "
    "Tomas climbed the stairs in the dark, dried the wick and lit it again. In "
    "the morning the townspeople found him asleep at the top of the tower with "
    "the logbook open on his knees. The last line read: light kept through the "
    "night, no ships lost.\n\n"
    "Today the lamp is electric and turns on by itself at sunset, but the town "
    "still chooses a keeper every spring. The keeper walks up the cliff path "
    "once a week, checks the windows and the stairs, and writes a short note "
    "in a new logbook, so that the record that Tomas began has never been "
    "broken."
)

# The ledger items cycle with period 7 and the shelf numbers with period 12.
_LEDGER_ITEMS = ("lamp", "rope", "barrel", "ladder", "kettle", "anchor", "lantern")

# Ledger lines between the two copies of the story in the first prompt. With
# 385 lines the second copy starts at token 7795, so the 8100-token prompt
# ends 305 tokens into it.
_RECALL_LEDGER_LINES = 385


def _ledger_lines(count: int) -> str:
    return "\n".join(
        f"Entry {i}: moved the {_LEDGER_ITEMS[i % len(_LEDGER_ITEMS)]} from "
        f"shelf {i % 12} to shelf {(i + 1) % 12}."
        for i in range(1, count + 1)
    )


def _recall_text() -> str:
    """The story, ledger lines, then the story again.

    The prompt ends inside the second copy of the story, so the next tokens
    repeat text about 7800 tokens back, in the first prefill chunk.
    """
    return (
        f"Chapter 1\n\n{_STORY}\n\nChapter 2\n\nStoreroom ledger\n\n"
        f"{_ledger_lines(_RECALL_LEDGER_LINES)}\n\nChapter 3\n\n{_STORY}"
    )


def _ledger_text() -> str:
    """600 ledger lines. The next tokens follow the pattern of the last lines."""
    return "Storeroom ledger\n\n" + _ledger_lines(600)


def _long_prompt(tokenizer, text: str) -> str:
    """Return the first ``PROMPT_TOKENS`` tokens of ``text`` as a string."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > PROMPT_TOKENS, (
        f"The source text has {len(ids)} tokens, fewer than {PROMPT_TOKENS}."
    )
    return tokenizer.decode(ids[:PROMPT_TOKENS])


def _check_prompt_lengths(lengths: list[int], what: str) -> None:
    for i, length in enumerate(lengths):
        assert MIN_PROMPT_TOKENS <= length <= MAX_PROMPT_TOKENS, (
            f"Prompt {i} {what} has {length} tokens, not "
            f"{MIN_PROMPT_TOKENS}-{MAX_PROMPT_TOKENS}."
        )


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

# Per-prompt checks of the three-way validation. Each one compares the
# largest error of one prompt against 1.5x the largest error of the CPU BF16
# reference for the same prompt.
PER_PROMPT_CHECKS = ("pp_max_linf_1.5x", "pp_max_l2_1.5x")

# TP 8 on one trn2.48xlarge. 16 Q/KV heads give 2 of each to each rank.
TENSOR_PARALLEL_SIZE = 8

# The prefill chunk: the batched-token limit, the one prefill bucket and the
# one KV segment size. An 8100-token prompt prefills in 8 chunks.
MAX_NUM_BATCHED_TOKENS = 1024
KV_SEGMENT_SIZE = 1024

# The reference is not the HF remote code, so its goldens get their own cache
# key.
GOLDEN_VARIANT = f"ouro-independent-ref-{MODEL_REVISION[:8]}"

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


# The (batch_size, seq_len) pairs: batch 1 and batch 4 (the max_num_seqs of
# the serve command in the README) at a 9216-token model length. Each pair
# compiles its own graphs.
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
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    # Option-1 registration reaches the engine core only through fork
    # inheritance. Decline to run rather than fail obscurely if the start
    # method is not fork.
    start_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if start_method != "fork":
        pytest.skip(
            f"VLLM_WORKER_MULTIPROC_METHOD={start_method!r}: a spawned engine "
            "core does not inherit this process's registrations."
        )

    _register_model()
    checkpoint = resolve_ouro_checkpoint()

    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # get_cached_reference_goldens tokenizes the prompt strings again, with
    # this call. Check the lengths before the reference runs.
    prompts = [
        _long_prompt(tokenizer, text) for text in (_recall_text(), _ledger_text())
    ]
    _check_prompt_lengths(
        [
            len(tokenizer([p], padding=True, truncation=True)["input_ids"][0])
            for p in prompts
        ],
        "string",
    )

    # Compute (or load from cache) the reference logits for the prompts. The
    # checkpoint stores BF16 weights, so the same-dtype reference is BF16.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=prompts,
        output_length=OUTPUT_LENGTH,
        golden_variant=GOLDEN_VARIANT,
        model_loader=load_ouro_reference,
    )
    prompts_input_ids = [
        [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
    ]
    # The token ids that the reference ran and that vLLM receives.
    _check_prompt_lengths([len(p[0]) for p in prompts_input_ids], "token ids")

    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits. Without it vLLM returns
    #     normalized log-probabilities, which do not match the reference.
    #   * async_scheduling=False keeps logprobs available. The async path returns
    #     before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy
    #     and deterministic, so a rerun compares the same token sequence.
    #
    # The three bucket lists pin compilation to the one prefill shape (a
    # 1024-token chunk with 1024-token KV segments) and the one decode shape
    # of this case.
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
                "kv_segment_size_buckets": [KV_SEGMENT_SIZE],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        generate_fn = create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH)
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
