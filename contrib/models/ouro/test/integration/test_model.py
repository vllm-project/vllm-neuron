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

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
    pytest contrib/models/ouro/test/integration/test_model.py \
        -v -s --timeout=10800 > /tmp/test_output.txt 2>&1

Without ``OURO_CHECKPOINT_PATH`` the test downloads the pinned revision from
the Hugging Face Hub.

Logit validation
----------------
transformers does not include the Ouro architecture, and the checkpoint's
remote modeling code is written for an older transformers. The reference is
therefore ``test/ouro_reference.py``: an independent torch implementation of
the HF ``modeling_ouro.py`` rounding order that reads the safetensors
weights directly. ``get_cached_reference_goldens`` runs it through
``model_loader=`` (the same reference the port's device logit test used) and
caches the logits: ``fp32_logits`` is the FP32 baseline, and
``dtype_logits`` is a BF16 pass teacher-forced along the FP32 greedy sequence.
``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so that it returns
dense per-token logits, and ``multi_prompt_logit_validation`` compares the
two. It flags any argmax divergence and any top-K logit error outside
``TOL_MAP``.
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

# Prompts to validate. The reference logits come from these same prompts, so
# no expected text or logits are hard-coded here.
PROMPTS = [
    "The color of the sky is",
    "A short summary of the history of the printing press:",
]

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

# The tensor parallel degree of the port's device tests. 16 Q/KV heads give 2
# of each to each rank.
TENSOR_PARALLEL_SIZE = 8

# The batched-token limit and prefill bucket of the port's device tests.
MAX_NUM_BATCHED_TOKENS = 1024

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


# The (batch_size, seq_len) pairs: batch 1 and batch 4 (the port's
# max_num_seqs) at a 2048-token model length. Each pair compiles its own
# graphs.
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

    # Compute (or load from cache) the reference logits for PROMPTS. The
    # checkpoint stores BF16 weights, so the same-dtype reference is BF16.
    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.torch_dtype = torch.bfloat16
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
        golden_variant=GOLDEN_VARIANT,
        model_loader=load_ouro_reference,
    )

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
    # The two bucket lists pin compilation to the one shape this case uses.
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
