# SPDX-License-Identifier: Apache-2.0
"""Integration test for Nanbeige4.2-3B served through vLLM on Neuron.

The test serves the contributed model with the offline ``LLM`` API and
compares the raw per-token logits against a HuggingFace reference, using the
three-way rule of ``vllm_neuron.accuracy`` (FP32 baseline, same-dtype teacher
forced pass, Neuron output). It is adapted from
``contrib/models/template/test/integration/test_model.py``.

Registration
------------
A contrib model is not in ``vllm_neuron/model/registry.py``, so this test
registers the architecture itself from the local ``src`` package, before it
builds the engine (contrib/README.md, "How do I register a custom model with
vLLM?", option 1).

That works here because the offline ``LLM(...)`` path forks: the plugin pins
``distributed_executor_backend="mp"`` (``vllm_neuron/vllm/platform.py``), the
start method comes from ``VLLM_WORKER_MULTIPROC_METHOD``, which the plugin
leaves at vLLM's ``fork`` default for every architecture except Qwen3-ASR, and
a forked engine-core or worker process inherits this process's populated
``ModelRegistry``. ``NeuronWorker.__init__`` then registers the plugin's own
``get_models()`` entries, which only overwrite their own architecture keys, so
the entry registered here survives and
``ModelRegistry.resolve_model_cls(...)`` in the model runner finds it.

The fallback, if the start method is ``spawn`` (a spawned process re-executes
from scratch and never runs the call below, and a class object cannot be
carried across ``spawn`` at all): use option 2 of contrib/README.md — move
``src/`` into ``vllm_neuron/model/nanbeige/`` and add the architecture to
``_MODELS`` in ``vllm_neuron/model/registry.py``. That edit is the user's, not
this PR's. The same applies to ``vllm serve``: the CLI starts the engine
itself, so option 1 has no script to run. These are code-path readings, not
observations from a run.

Checkpoint
----------
``MODEL_PATH`` defaults to the Hub id. Set ``NANBEIGE4_CHECKPOINT_PATH`` to a
local directory to run without a Hub download. The checkpoint ships its model
code in the repository, so both the reference and vLLM load it with
``trust_remote_code=True``.

Scope
-----
The two cases below match the shapes that the port's own two-node device test
exercised: one sequence with a 2,048-token context (the logit case) and four
sequences at the same context length (the 1,024-in / 512-out serving case).
This single-node offline configuration was not itself part of a CI run; see
the "Validation report" section of the model README.

Run with:
    pytest contrib/models/nanbeige/test/integration/test_model.py \\
        -v -s --timeout=3600
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

# Load the model implementation that ships in this contrib folder
# (``contrib/models/nanbeige/src/``) by file location, so this test keeps
# working when the folder is copied out of the repository. See
# ``contrib/models/nanbeige/test/_local_src.py``.
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "contrib_nanbeige_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_local_src)
nb = _local_src.load_test_helper("nanbeige_utils")

# The architecture string in the checkpoint's ``config.json``.
ARCHITECTURE = "NanbeigeForCausalLM"

# ``MODEL_ID`` only builds the golden cache key; ``MODEL_PATH`` is what the
# reference model and vLLM load from.
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
MODEL_PATH = os.environ.get("NANBEIGE4_CHECKPOINT_PATH") or MODEL_ID

# The two prompts the port's logit tests used (``PROMPTS_2`` in
# ``test/vllm_neuron/utils/test_prompts.py``, copied here per contrib/README.md).
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt. The port's device logit
# case used 32.
OUTPUT_LENGTH = 32

# The port's device logit case ran against the default tolerance map of
# ``multi_prompt_logit_validation``
# (``vllm_neuron.accuracy.constants.DEFAULT_TOLERANCE_MAP``), so this test uses
# the same bounds instead of the looser dense-BF16 convention in the template.
# The argmax (divergence) lower bound stays strict at 1e-5.
TOL_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

# 48 Q heads and 8 KV heads. TP8 gives 6 Q heads and 1 KV head per rank, the
# degree the port's device test used (there split over two nodes).
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve Nanbeige4.2-3B through vLLM and validate its logits.

    Args:
        batch_size: Number of prompts to run in a single batch.
        seq_len: Maximum model (context + generation) length to configure.
    """

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, ModelRegistry

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    # Register the contrib architecture before the engine resolves it.
    ModelRegistry.register_model(ARCHITECTURE, _local_src.load().NanbeigeForCausalLM)

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        # The checkpoint's remote code builds its KV cache with
        # ``DynamicCache.from_legacy_cache``, which transformers 5.17 (the
        # version vLLM Neuron 0.29.0.1.2.0 ships with) no longer has, so it
        # cannot generate. The independent upstream-Llama reference in
        # ``nanbeige_utils`` computes the goldens instead;
        # ``test_nanbeige_reference_matches_remote_code`` checks it against
        # the remote code.
        model_loader=nb.load_loop_reference,
        golden_variant=nb.REFERENCE_IMPL,
    )

    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits.
    #   * async_scheduling=False keeps logprobs available.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy
    #     and deterministic.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
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

    # vLLM 0.29.0 decodes every logprob id to a token string. The Nanbeige
    # tokenizer has no string for the padding ids past its vocabulary
    # (``convert_ids_to_tokens`` returns None for them), which crashes the
    # detokenizer. Map those ids to "" in this test's tokenizer: the logit
    # comparison reads the logits, never the strings.
    tokenizer_in_engine = llm.get_tokenizer()
    convert_ids_to_tokens = tokenizer_in_engine.convert_ids_to_tokens

    def _convert_ids_to_tokens(ids, *args, **kwargs):
        tokens = convert_ids_to_tokens(ids, *args, **kwargs)
        if isinstance(tokens, list):
            return ["" if token is None else token for token in tokens]
        return "" if tokens is None else tokens

    tokenizer_in_engine.convert_ids_to_tokens = _convert_ids_to_tokens
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
