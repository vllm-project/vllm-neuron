# SPDX-License-Identifier: Apache-2.0
"""Integration test for the contrib HRM-Text model (sapientinc/HRM-Text-1B).

The test serves the model through vLLM on Neuron with tensor parallelism 2,
decodes 16 greedy tokens for each of 2 prompts, and compares the raw per-token
logits with the HuggingFace reference through the ``vllm_neuron.accuracy``
golden cache (FP32 baseline plus a same-dtype teacher-forced pass).

Geometry
--------
The recorded run of the port's own logit test drove TP 2 as one rank on each of
2 trn2.48xlarge nodes, through an online ``vllm serve`` pair started by the
plugin's shared logit-test flow (``test/vllm_neuron/utils/logit_test_flow.py``).
That flow lives outside this folder, so the contrib test re-expresses the same
check with one offline engine: same checkpoint, same 2 prompts, same 16 output
tokens, same 256-token context, same tolerance map. TP 2 then needs 2 Neuron
cores reachable from this process. See the model README's "Validation report"
for what the recorded evidence covers.

Registration
------------
A contrib model is not in ``vllm_neuron/model/registry.py``. This test
registers the architecture with vLLM itself, which is the first option in
contrib/README.md ("How do I register a custom model with vLLM?"). It loads the
factory class out of this folder's ``src/`` by file location
(``test/_local_src.py``), so the folder works when it is copied into an
application and nothing needs the repository root on ``sys.path``.

A registered class object lives in this process only. vLLM resolves the
architecture here (``ModelConfig`` inspects the registered class in process)
and again in the engine-core and worker processes, which the plugin runs with
the ``mp`` executor. Under the default start method those are forks of this
process, so they inherit both the registration and the loaded ``src/`` modules.
With ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` they do not, and the test skips:
copy ``src/`` into an importable package of your own and register the
``"module:Class"`` string form instead, or take option 2 of contrib/README.md.

Running
-------
    pytest contrib/models/hrm_text/test/integration/test_model.py \
        -v -s --timeout=10800
"""

import os

import pytest
import torch


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` by file location.

    That helper loads this folder's ``src/`` (and the shared
    ``test/module/common.py``) under module names unique to this contrib model.
    Loading the helper itself by location, under its own unique name, keeps
    every name out of ``sys.path``: two contrib models collected in one pytest
    run cannot resolve to each other's code, and nothing needs the repository
    root.
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "contrib_hrm_text_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()

common = _local_src.load_common()

# The architecture string of the checkpoint's ``config.json``.
ARCHITECTURE = "HrmTextForCausalLM"

# sapientinc/HRM-Text-1B at the revision the port pins.
MODEL_ID = common.MODEL_ID
REVISION = common.MODEL_REVISION

# The two prompts of the plugin's shared logit-validation list, which the
# port's own logit test uses. They are repeated here because that list lives
# outside this folder.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning across nine countries and containing approximately ten percent of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, encoding the genetic instructions for building and maintaining a human organism.",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# Tensor parallelism. HRM-Text has 12 attention heads and the decode kernel
# needs an even number of heads on each rank, so the port supports TP 1 and
# TP 2 only; TP 2 is the degree the recorded runs used.
TP_SIZE = 2

# The tolerance map of ``vllm_neuron.accuracy`` (``DEFAULT_TOLERANCE_MAP``),
# written out here. The port's logit test passed ``tol_map=None``, which
# selects exactly this map, so the recorded PASS was measured against these
# bounds and the shipped test keeps them instead of the looser contrib-template
# values. The argmax (divergence) bound stays at 1e-5: a nonzero divergence is
# a real correctness bug and must not be widened to hide it.
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        # The geometry of the port's logit test: one sequence in a 256-token
        # context. Each pair compiles its own graphs, so keep the list short.
        (1, 256),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve HRM-Text through vLLM and validate its logits.

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

    # The engine's child processes inherit this registration only by fork
    # (see "Registration" above).
    start_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if start_method != "fork":
        pytest.skip(
            f"VLLM_WORKER_MULTIPROC_METHOD={start_method!r}: a spawned engine "
            "process does not inherit the ModelRegistry of this process. "
            "Register an importable package instead; see the docstring."
        )

    ModelRegistry.register_model(ARCHITECTURE, _local_src.load().HrmTextForCausalLM)

    # A local snapshot of the pinned revision (HRM_TEXT_MODEL_PATH overrides).
    checkpoint = common.resolve_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=checkpoint,
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
    #   * async_scheduling=False keeps logprobs available. The async path
    #     returns before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} (in neuron_config)
    #     makes sampling greedy and deterministic.
    # The two bucket lists pin compilation to the one shape this case uses.
    # The whole block mirrors the engine arguments of the port's logit test.
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TP_SIZE,
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
                "max_logprobs": -1,
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
        n_prompts = len(result.per_prompt_results)
        for check in PER_PROMPT_CHECKS:
            value = result.aggregate_metrics.get(check)
            assert value == f"{n_prompts}/{n_prompts}", (
                f"{check} is {value} for parameters: {batch_size=}, {seq_len=}"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
