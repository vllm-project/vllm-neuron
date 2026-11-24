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
        -v -s --timeout=14400 > /tmp/test_output.txt 2>&1

Without ``GEMMA3_CHECKPOINT_PATH`` the test downloads the pinned revision
(about 33 GB of FP32 weights) from the Hugging Face Hub. The HF reference runs
the 8B model on the host CPU in FP32 and in BF16 over two prompts of about
8,200 tokens, so the first case takes much longer than the second, which
reads the cached reference logits.

Long prompts
------------
Each case serves a 9216-token model length (an 8192-token prompt plus 1024
generated tokens) and validates two prompts of 8000 to 8192 tokens. The
prompts are synthetic documents built in this file from fixed word lists and a
fixed linear congruential generator, so every run sends the same text. Each
prompt keeps the largest number of records that the checkpoint's tokenizer
encodes in at most 8192 tokens, and the test asserts the token count of every
prompt as ``get_cached_reference_goldens`` tokenizes it.

``max_num_batched_tokens`` (8192) is below the model length, so vLLM runs
chunked prefill, and ``kv_segment_size_buckets`` turns on the plugin's
segmented prefill: every prefill step goes through ``NF.segmented_attention``,
the ``kv_segment_size`` branch of ``Gemma3Attention.forward_prefill``. Each
prompt fits in one 8192-token step, so it prefills in a single segmented step
with no prior KV. Decode then runs over a 9216-token KV context.

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
# The pinned HuggingFace revision of the checkpoint.
HF_REVISION = "5ef96d7a82b2bff8009192979432c9bc64e870cd"
CHECKPOINT_ENV = "GEMMA3_CHECKPOINT_PATH"

# Model length of both cases: an 8192-token prompt plus 1024 generated
# tokens. It is a multiple of the 32-token KV block.
MAX_MODEL_LEN = 9216

# Tokens per prefill step and the KV segment size of segmented prefill. 8192
# is the largest KV segment size the segmented attention kernel supports.
PREFILL_TOKENS = 8192

# Token count range of each prompt, BOS included.
MIN_PROMPT_TOKENS = 8000
MAX_PROMPT_TOKENS = 8192

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

# Per-prompt checks of the three-way validation. Each one compares the
# largest error of one prompt against 1.5x the largest error of the CPU BF16
# reference for the same prompt.
PER_PROMPT_CHECKS = ("pp_max_linf_1.5x", "pp_max_l2_1.5x")

# The tensor parallel degree of the test: 32 Q heads and 8 KV heads give 4 Q
# heads and one KV head on each rank.
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)

ARCHITECTURE = "Gemma3ForCausalLM"

# Word lists of the two synthetic documents.
_CITIES = (
    "Lisbon,Osaka,Nairobi,Quito,Tromso,Hobart,Tbilisi,Windhoek,"
    "Cusco,Tallinn,Hanoi,Perth"
)
_GOODS = (
    "barley,copper wire,salt,wool,tea,glass beads,timber,olive oil,indigo,cotton,"
    "pepper,tin"
)
_ACTIONS = "shipped,received,stored,inspected,weighed,sold,loaded,counted"
_LEDGER_NOTES = (
    "The weather was dry.,Two crates were damp.,The clerk signed late.,"
    "A bridge was closed.,The harbour was busy.,Prices rose slightly."
)
_STATIONS = (
    "Ridge Point,Cedar Bay,North Mill,Salt Flats,Pine Hollow,Granite Pass,"
    "Lake Vista,Old Harbour"
)
_WINDS = "north,northeast,east,southeast,south,southwest,west,northwest"
_SKIES = (
    "The sky was clear.,Light rain fell.,Fog lifted at noon.,"
    "Clouds built up in the afternoon.,A thunderstorm passed nearby.,"
    "Snow dusted the hills."
)
# Records generated per document, more than an 8192-token prompt holds.
_NUM_RECORDS = 600


def _lcg(x: int) -> int:
    """Return the next state of a fixed linear congruential generator."""
    return (1103515245 * x + 12345) % 2**31


def _pick(x: int, words: str) -> tuple[str, int]:
    """Advance the generator and pick one entry of a comma-separated list."""
    x = _lcg(x)
    choices = words.split(",")
    return choices[x % len(choices)], x


def _ledger_record(index: int, x: int) -> tuple[str, int]:
    city, x = _pick(x, _CITIES)
    goods, x = _pick(x, _GOODS)
    action, x = _pick(x, _ACTIONS)
    note, x = _pick(x, _LEDGER_NOTES)
    x = _lcg(x)
    return (
        f"Record {index}: On day {x % 365 + 1}, the depot in {city} {action} "
        f"{x % 900 + 100} units of {goods}. {note}\n"
    ), x


def _station_record(index: int, x: int) -> tuple[str, int]:
    station, x = _pick(x, _STATIONS)
    wind, x = _pick(x, _WINDS)
    sky, x = _pick(x, _SKIES)
    x = _lcg(x)
    return (
        f"Reading {index}: At station {station} on day {x % 365 + 1}, the "
        f"temperature was {x % 41 - 5} degrees, the humidity {x % 71 + 20} "
        f"percent and the wind {x % 60 + 2} km/h from the {wind}. {sky}\n"
    ), x


# (head, record generator, generator seed, tail) of each prompt.
_DOCUMENTS = (
    (
        "The following is a ledger of trade records from a network of depots.\n\n",
        _ledger_record,
        2026,
        "\nQuestion: Which depot appears most often in the ledger above, and "
        "what did it handle?\nAnswer:",
    ),
    (
        "The following is a log of readings from a network of weather stations.\n\n",
        _station_record,
        1871,
        "\nQuestion: Summarize the readings above, station by station.\nAnswer:",
    ),
)


def _num_prompt_tokens(tokenizer, prompt: str) -> int:
    """Count the tokens of ``prompt`` as ``get_cached_reference_goldens`` does."""
    encoded = tokenizer([prompt], padding=True, truncation=True)
    return len(encoded["input_ids"][0])


def _fit_prompt(tokenizer, head: str, records: list[str], tail: str) -> str:
    """Return the head, the most records that fit and the tail as one prompt.

    A binary search with the checkpoint's tokenizer finds the largest ``n``
    for which the head, the first ``n`` records and the tail have at most
    MAX_PROMPT_TOKENS tokens.
    """

    def build(n: int) -> str:
        return head + "".join(records[:n]) + tail

    assert _num_prompt_tokens(tokenizer, build(len(records))) > MAX_PROMPT_TOKENS
    lo, hi = 0, len(records)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _num_prompt_tokens(tokenizer, build(mid)) <= MAX_PROMPT_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return build(lo)


def _long_prompts(tokenizer) -> list[str]:
    """Build the two long prompts, truncated to MAX_PROMPT_TOKENS tokens."""
    prompts = []
    for head, make_record, seed, tail in _DOCUMENTS:
        records, x = [], seed
        for index in range(1, _NUM_RECORDS + 1):
            record, x = make_record(index, x)
            records.append(record)
        prompts.append(_fit_prompt(tokenizer, head, records, tail))
    return prompts


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


# Batch 1 and batch 4 at a 9216-token model length. Each pair compiles its own
# graphs.
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

    _register_model()
    checkpoint = _checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for the
    # long prompts. The checkpoint stores FP32 weights and the engine serves
    # BF16, so the same-dtype reference must be BF16. The checkpoint has no
    # remote code; the built-in transformers Gemma3 class loads it.
    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = _long_prompts(tokenizer)
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=prompts,
        output_length=OUTPUT_LENGTH,
        golden_variant=f"rev-{HF_REVISION}",
    )
    # The reference re-tokenizes the prompt strings; check what it ran.
    for index, prompt_ids in enumerate(cached_goldens["input_ids"]):
        num_tokens = len(prompt_ids[0])
        print(f"Prompt {index + 1}: {num_tokens} tokens")
        assert MIN_PROMPT_TOKENS <= num_tokens <= MAX_PROMPT_TOKENS, (
            f"Prompt {index + 1} has {num_tokens} tokens, outside "
            f"[{MIN_PROMPT_TOKENS}, {MAX_PROMPT_TOKENS}]."
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
    # The bucket lists pin compilation to one prefill graph (an 8192-token
    # step with an 8192-token KV segment) and one decode graph (this case's
    # batch size over a 9216-token context). ``kv_segment_size_buckets``
    # turns on segmented prefill; see "Long prompts" above.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=PREFILL_TOKENS,
        max_num_seqs=batch_size,
        block_size=32,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [PREFILL_TOKENS],
                "kv_segment_size_buckets": [PREFILL_TOKENS],
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
        n_prompts = len(result.per_prompt_results)
        for check in PER_PROMPT_CHECKS:
            value = result.aggregate_metrics.get(check)
            assert value == f"{n_prompts}/{n_prompts}", (
                f"{check} is {value} for parameters: {batch_size=}, {seq_len=}"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
