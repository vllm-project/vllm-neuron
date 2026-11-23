# SPDX-License-Identifier: Apache-2.0
"""Integration test for EXAONE-4.0-32B served through vLLM on Neuron.

The test serves the contributed model with the offline ``LLM`` API and
compares the raw per-token logits against a HuggingFace reference, using the
three-way rule of ``vllm_neuron.accuracy`` (FP32 baseline, same-dtype teacher
forced pass, Neuron output). It is adapted from
``contrib/models/template/test/integration/test_model.py``.

Registering your model
----------------------
A contrib model is not in ``vllm_neuron/model/registry.py``, and vLLM ships
its own ``Exaone4ForCausalLM``, so this test registers the architecture from
the local ``src`` package before it builds the engine — which replaces vLLM's
class (contrib/README.md, "How do I register a custom model with vLLM?",
option 1).

That registration reaches the engine-core and worker processes by ``fork``
inheritance: the plugin runs them as separate processes, and the default start
method copies this process's registry into them. No fallback is needed here,
but the test does depend on that default — running it with
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` would resolve vLLM's own EXAONE 4.0
class instead of this one, because vLLM's entry is never removed, and the run
would then fail in the worker with an ``AttributeError`` on ``from_configs``.
See the model README's Usage section.

Checkpoint
----------
``MODEL_PATH`` defaults to the Hub id. Set ``EXAONE4_CHECKPOINT_PATH`` to a
local directory to run without a Hub download. The BF16 weights are about
64 GB, and the reference pass loads the same checkpoint on CPU.

Scope
-----
One case only. The model is 32B, so each ``(batch_size, seq_len)`` pair costs
a full graph compile, and `(4, 2048)` is the shape the port's recorded logit
run used (TP8 per engine, a 2,048-token context, four sequences). That run was
a two-node 1P1D disaggregated deployment; this single-node offline
configuration was not itself part of a CI run. The 2,048-token context stays
below the 4,096-token sliding window, so this case does not reach the window
boundary. See the "Validation report" section of the model README.

Run with:
    pytest contrib/models/exaone4/test/integration/test_model.py \\
        -v -s --timeout=7200
"""

import os
import sys

import pytest
import torch


def _load_local_src():
    """Load this model's ``test/_local_src.py`` helper by file location.

    Loading the helper by location, under a name unique to this model, keeps
    two contrib models collected in one pytest run from resolving each other's
    helper (a plain ``import _local_src`` would return whichever copy Python
    cached first). Nothing is added to ``sys.path``.
    """
    import importlib.util
    from pathlib import Path

    name = "contrib_exaone4_local_src"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_local_src = _load_local_src()

# The architecture string in the checkpoint's ``config.json``.
ARCHITECTURE = "Exaone4ForCausalLM"

# ``MODEL_ID`` only builds the golden cache key; ``MODEL_PATH`` is what the
# reference model and vLLM load from.
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-32B"
MODEL_PATH = os.environ.get("EXAONE4_CHECKPOINT_PATH") or MODEL_ID

# The two prompts the port's DI logit test used (``PROMPTS_2`` in
# ``test/vllm_neuron/utils/test_prompts.py``, copied here per contrib/README.md).
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt, as in the port's DI
# logit run.
OUTPUT_LENGTH = 16

# The tolerance map of the port's DI logit run (the same map as the Qwen3-32B
# dense on-device-sampling DI test). The argmax (divergence) lower bound stays
# strict at 1e-5 — a nonzero divergence is a real correctness bug and must not
# be widened to hide it.
TOL_MAP = {
    "5": (1e-5, 0.025),
    "50": (1e-5, 0.04),
    "1000": (1e-5, 0.06),
    "all": (1e-5, 0.065),
}

# 40 Q heads and 8 KV heads. TP8 gives 5 Q heads and 1 KV head per rank, the
# degree each engine of the port's DI logit run used, and also the degree of
# its single-node performance run.
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve EXAONE-4.0-32B through vLLM and validate its logits.

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

    # The modeling code that ships in this contrib folder (``../../src/``),
    # loaded by file location so that this test keeps working after the model
    # folder is copied out of this repository.
    Exaone4ForCausalLM = _local_src.load().Exaone4ForCausalLM

    # Register the contrib architecture, replacing vLLM's own EXAONE 4.0
    # class, before the engine resolves it.
    ModelRegistry.register_model(ARCHITECTURE, Exaone4ForCausalLM)

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
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
