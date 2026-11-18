# SPDX-License-Identifier: Apache-2.0
"""Integration test for the contrib ERNIE 4.5 dense model on Neuron.

The test serves ``baidu/ERNIE-4.5-0.3B-PT`` through vLLM with the
``vllm-neuron`` plugin and compares the raw per-token logits against a
HuggingFace golden reference, as the contrib template does.

Scope
-----
The shapes, the tolerance map and the tensor-parallel degree come from the
configuration that the model's CI evidence was produced with (see the model
README's "Validation report"): TP 1, a 2,048-token context, batch 1 and batch
4, 16 decoded tokens and the two shared logit-validation prompts. The logit
evidence itself came from a two-node 1P1D disaggregated run, so this
single-node offline arrangement is an adaptation of it, not a rerun of it.

The ERNIE 4.5 port supports the ``neuron_native`` route only.
``test/conftest.py`` forces ``VLLM_NEURON_BACKEND`` before pytest imports this
module, which is before anything in ``vllm_neuron`` reads it. The helper
modules come from ``test/_local_src.py``, by file location.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``. The test loads the
factory from this folder's ``src/`` and registers it with vLLM's
``ModelRegistry`` before it builds the engine, which is Option 1 of
``contrib/README.md`` ("How do I register a custom model with vLLM?"). That
works because the engine core and its workers are forks of this process, so
they inherit both the registry entry and the loaded ``src/`` package.

Upstream vLLM ships its own ``Ernie4_5ForCausalLM``, so this registration
overwrites a built-in entry rather than adding a new one. A process that does
not inherit it therefore does not report an unsupported architecture, as it
would for a contrib name vLLM does not know: it resolves vLLM's class and then
fails on ``from_configs``, a plugin convention vLLM's classes do not define.
The resulting ``AttributeError`` says nothing about registration, which is why
this test refuses to run unless the start method is ``fork``. If your engine
runs its workers in processes that do not inherit this process's memory, use
Option 2 instead.

Running
-------
Serving the model needs Neuron hardware::

    pytest contrib/models/ernie4_5/test/integration/test_model.py \
        -v -s --timeout=3600 > /tmp/test_output.txt 2>&1

Set ``ERNIE4_5_CHECKPOINT_DIR`` to a local snapshot of the pinned revision to
validate exactly the checkpoint the evidence was produced with.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The test must keep importing after the folder is copied out of the
    repository, so it resolves ``src/`` and the sibling test helpers from their
    own paths instead of through a ``contrib.models.ernie4_5...`` namespace
    path that needs the repository root on ``sys.path``. A plain
    ``import _local_src`` would need this directory on ``sys.path`` and would
    resolve to whichever contrib model got imported first in the process. See
    ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the local-src helper at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_local_src = _load_local_src()
_checkpoint = _local_src.load_helper("checkpoint")

# The architecture string in the checkpoint's config.json; vLLM resolves the
# model class through it. The pinned HF revision that every CI result for this
# model was produced at is ``_checkpoint.REVISION``.
ARCHITECTURE = _checkpoint.ARCHITECTURE

# A local snapshot of the pinned revision, or the Hub id (which resolves to the
# Hub's current revision for this model id).
MODEL_PATH = os.environ.get(_checkpoint.CHECKPOINT_DIR_ENV, _checkpoint.MODEL_ID)
MODEL_ID = _checkpoint.MODEL_ID

# The two prompts the plugin's logit-validation tests use for this model.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning across nine countries and containing approximately ten percent of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, encoding the genetic instructions for building and maintaining a human organism.",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The tolerance map of the ERNIE 4.5 logit run that passed in CI (the
# disaggregated-inference logit test). The argmax (divergence) lower bound
# stays at the strict 1e-5 of the template: a nonzero divergence is a real
# correctness bug. The per-K upper bounds are the ones that run used.
TOL_MAP = {
    "5": (1e-5, 0.015),
    "50": (1e-5, 0.04),
    "1000": (1e-5, 0.06),
    "all": (1e-5, 0.065),
}

# Deterministic results across runs.
torch.manual_seed(42)


# TP 1 with a 2,048-token context, at the two batch sizes the evidence covers.
def _require_fork_start_method() -> None:
    """Skip unless vLLM's child processes will inherit this process's memory.

    The Option 1 registration below reaches the engine core and its workers
    only because they are ``fork``s of this process. This matters more for
    ERNIE 4.5 than for a typical contrib model: upstream vLLM already ships its
    own ``Ernie4_5ForCausalLM`` in the built-in model table, so a child process
    that did not inherit this registration does **not** report an unsupported
    architecture. It resolves vLLM's class and then fails on ``from_configs``
    (``neuron_model_runner.py``), which is a plugin convention that vLLM's own
    model classes do not define, so the failure is an ``AttributeError`` that
    names nothing about registration. Decline to run rather than turn a
    registration problem into that.
    """
    method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if method != "fork":
        pytest.skip(
            f"This test registers the contrib architecture in-process, which "
            f"reaches vLLM's children only under the fork start method. "
            f"VLLM_WORKER_MULTIPROC_METHOD={method!r}: a spawned child would "
            f"resolve {ARCHITECTURE!r} to vLLM's own built-in implementation "
            f"instead of this folder's, then fail on from_configs. Use the "
            f"guide's Option 2 (the architecture in the plugin's registry) "
            f"for a spawned engine."
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve ERNIE 4.5 through vLLM on Neuron and validate its logits.

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

    _require_fork_start_method()

    # Register the contrib architecture before the engine builds the model.
    # This overwrites vLLM's own built-in entry for the same architecture
    # string, so vLLM logs "already registered ... will be overwritten". That
    # warning is the point: without it, the engine resolves vLLM's ERNIE 4.5
    # rather than this folder's and then fails on from_configs.
    #
    # Pass the class object, not vLLM's ``"module:Class"`` string form: vLLM
    # inspects a string registration by importing it in a fresh interpreter,
    # which cannot import a module that ``_local_src`` loaded from a file
    # location. A class object is inspected in this process instead.
    ModelRegistry.register_model(ARCHITECTURE, getattr(_local_src.load(), ARCHITECTURE))

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key, so it stays the Hub id
    # even when MODEL_PATH points at a local snapshot.
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
    #   * logprobs_mode="raw_logits" returns raw logits. Without it vLLM returns
    #     normalized log-probabilities, which do not match the HF reference.
    #   * async_scheduling=False keeps logprobs available. The async path returns
    #     before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy
    #     and deterministic, so a rerun compares the same token sequence.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
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
