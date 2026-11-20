# SPDX-License-Identifier: Apache-2.0
"""Integration test for the contrib BailingMoeV2 model (inclusionAI/Ling-mini-2.0).

The test serves the model through vLLM on Neuron with tensor parallelism 8 and
expert parallelism 8, decodes 16 greedy tokens for each of 2 prompts, and
compares the raw per-token logits with an independent reference.

Reference goldens
-----------------
The goldens do not come from the HuggingFace remote modeling code of the
checkpoint: that code does not import with the transformers version of the
plugin's test image. ``test/module/reference_model.py`` is an independent
torch transcription of it, and it has the ``model_loader`` signature that
``get_cached_reference_goldens`` expects. ``GOLDEN_VARIANT`` namespaces the
golden cache, so these goldens never collide with HF-derived ones.

Geometry
--------
The recorded runs of this port drove TP 8 / EP 8 as 4 ranks on each of 2
trn2.48xlarge nodes, through the repository's
``test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py`` launcher.
This test builds one offline engine in one process, so it needs 8 Neuron cores
reachable from that process. The model README's "Validation report" states what
the recorded evidence does and does not cover; no recorded test id names logit
validation for this model, so this test has not been run in the shape it ships
in.

Registration
------------
A contrib model is not in ``vllm_neuron/model/registry.py``. This test
registers the architecture with vLLM itself, which is the first option in
contrib/README.md ("How do I register a custom model with vLLM?").

That option reaches the engine's child processes by ``fork`` inheritance, and
by nothing else. ``LLM(...)`` runs the engine core in a child process and the
8 workers in children of that one, because the plugin pins
``distributed_executor_backend="mp"`` (``vllm_neuron/vllm/platform.py``).
Those children come from ``VLLM_WORKER_MULTIPROC_METHOD``, which defaults to
``fork``; a forked child inherits this process's ``ModelRegistry``, so the
architecture is already present when ``NeuronWorker.__init__`` adds the
plugin's own models (``register_model`` only replaces same-architecture keys)
and when the runner calls ``ModelRegistry.resolve_model_cls``. Under ``spawn``
a child re-executes from scratch and no registration of this process survives,
which is why ``vllm serve`` needs option 2 of contrib/README.md.
``_require_fork_start_method`` below fails the test with that explanation
rather than letting a worker fail with an unsupported architecture.

Running
-------
    pytest contrib/models/bailing_moe/test/integration/test_model.py \
        -v -s --timeout=10800
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

utils = _local_src.load_test_module("utils")
_reference = _local_src.load_test_module("reference_model")
load_bailing_reference = _reference.load_bailing_reference

# The architecture string of the checkpoint's ``config.json`` and the factory
# that implements it. ``ModelRegistry`` also takes a lazy "module:Class"
# string, but that form needs a module name that a fresh interpreter can
# import, and the loader above gives the factory a module name derived from
# ``_local_src.SRC_PACKAGE`` (``contrib_bailing_moe_src.factory``) that exists
# only in this process and the processes it forks. So the test registers the
# class object, which ``register_model`` inspects in this process and never
# imports again.
ARCHITECTURE = "BailingMoeV2ForCausalLM"
FACTORY = _local_src.load("factory").BailingMoeV2ForCausalLM

# inclusionAI/Ling-mini-2.0 at the revision the port pins.
MODEL_ID = utils.MODEL_ID
REVISION = utils.REVISION

# The two prompts of the plugin's shared logit-validation list, which the
# port's full-model tests use.
PROMPTS = utils.PROMPTS_2

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The tolerance map of ``vllm_neuron.accuracy`` (``DEFAULT_TOLERANCE_MAP``),
# written out here. The port's own full-model logit test passed
# ``tol_map=None``, which selects exactly this map, so the shipped test keeps
# the same bounds instead of the looser contrib-template values. The argmax
# (divergence) bound stays at 1e-5: a nonzero divergence is a real correctness
# bug and must not be widened to hide it.
TOL_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

# Per-prompt checks of the three-way validation. Each one compares the
# largest error of one prompt against 1.5x the largest error of the CPU BF16
# reference for the same prompt.
PER_PROMPT_CHECKS = ("pp_max_linf_1.5x", "pp_max_l2_1.5x")

# Deterministic results across runs.
torch.manual_seed(42)


def _require_fork_start_method() -> None:
    """Fail unless the engine's child processes will inherit this registry.

    The registration of this test reaches the engine core and the workers only
    because they are forked from this process (see the "Registration" section
    of the module docstring). Under ``spawn`` the architecture is absent in
    those processes and the run fails inside a worker. Serving a contrib model
    without this inheritance is option 2 of contrib/README.md.
    """
    method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if method != "fork":
        pytest.fail(
            f"VLLM_WORKER_MULTIPROC_METHOD={method!r}. This test registers "
            "BailingMoeV2ForCausalLM in the test process, which only reaches "
            "the engine core and the workers when they are forked from it. "
            "Unset the variable (the default is fork), or register the "
            "architecture with the plugin itself (option 2 in "
            "contrib/README.md)."
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        # The geometry of the port's full-model tests: one sequence in a
        # 256-token context. Each pair compiles its own graphs, so keep the
        # list short.
        (1, 256),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve BailingMoeV2 through vLLM and validate its logits.

    Args:
        batch_size: Number of prompts to run in a single batch.
        seq_len: Maximum model (context + generation) length to configure.
    """

    from transformers import AutoTokenizer
    from vllm import LLM, ModelRegistry

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    _require_fork_start_method()
    ModelRegistry.register_model(ARCHITECTURE, FACTORY)

    # A local snapshot of the pinned revision.
    checkpoint = utils.resolve_checkpoint()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The reference reads ``config.json`` of the checkpoint itself, so it needs
    # the target dtype only. Ling-mini-2.0 is a BF16 checkpoint, and the
    # factory accepts bfloat16 only.
    model_config = SimpleNamespace(torch_dtype=torch.bfloat16)
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=utils.GOLDEN_VARIANT,
        model_loader=load_bailing_reference,
    )

    # The neuron_config of the port's logit server, with the buckets of this
    # case. The two bucket lists pin compilation to the one shape this case
    # uses; without them the defaults compile a power-of-2 ladder.
    neuron_config = {
        **utils.LOGIT_SERVER_NEURON_CONFIG,
        "num_batched_tokens_buckets": [seq_len],
        "num_seqs_buckets": [batch_size],
    }

    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits. Without it vLLM returns
    #     normalized log-probabilities, which do not match the reference.
    #   * async_scheduling=False keeps logprobs available. The async path
    #     returns before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} (in neuron_config)
    #     makes sampling greedy and deterministic.
    # The remaining arguments mirror the port's full-model server: expert
    # parallelism over the 8 ranks, the 32-token KV block size, and
    # trust_remote_code for the checkpoint's own config and tokenizer code.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=utils.WORLD_SIZE,
        enable_expert_parallel=True,
        trust_remote_code=True,
        block_size=utils.BLOCK_SIZE,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={"neuron_config": neuron_config},
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
