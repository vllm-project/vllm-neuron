# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Phi3 contrib model, served through vLLM on Neuron.

The test serves ``microsoft/Phi-4-mini-instruct`` with the implementation in
``contrib/models/phi3/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, by file location, so the test needs neither the
repository root nor any ``sys.path`` entry. The engine core and the workers are
forked from this process and inherit the populated registry.

Running
-------
The configuration below is the single-node topology of the recorded accuracy
and performance runs: TP 8 on one trn2.48xlarge node with a 4,096-token model
length.

    pytest contrib/models/phi3/test/integration/test_model.py \
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

The engine arguments below set ``disable_sliding_window=True``, the offline
form of ``--disable-sliding-window``, as the recorded runs did. The setting is
recommended, not required. This implementation has no sliding-window attention
path. The factory accepts a window that is not smaller than
``max_position_embeddings``. Phi-4-mini declares a 262,144-token window, so the
window has no effect for this checkpoint.

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence.
``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so that it returns
dense per-token logits, and ``multi_prompt_logit_validation`` compares the two.
It flags any argmax divergence and any top-K logit error outside ``TOL_MAP``.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

# The checkpoint that the recorded runs used.
MODEL_PATH = "microsoft/Phi-4-mini-instruct"

# Prompts to validate. The HF reference logits come from these same prompts,
# so no expected text or logits are hard-coded here.
PROMPTS = [
    "The color of the sky is",
    "A short summary of the history of the printing press:",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map, the same map the port's own logit test
# used. The argmax (divergence) lower bound stays strict at 1e-5 — a nonzero
# divergence is a real correctness bug and must not be widened to hide it. The
# per-K upper bounds bound the allowed top-K logit drift.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# The tensor parallel size of the recorded single-node runs. The port's own
# multi-node tests ran TP 4 (see this model's README).
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)


def _builtin_phi3_loader(model_checkpoint, dtype, config=None):
    """Load the reference with the ``Phi3ForCausalLM`` built into transformers.

    The Phi-4-mini checkpoint ships remote modeling code that imports
    ``transformers.utils.LossKwargs``, which transformers 5.17 (the version
    vLLM Neuron 0.29.0.1.2.0 ships with) no longer has, so the default
    ``trust_remote_code`` loader cannot import it. The built-in class implements
    the same architecture and reads the same checkpoint.
    """
    from transformers import Phi3ForCausalLM

    kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
    if config is not None:
        kwargs["config"] = config
    model = Phi3ForCausalLM.from_pretrained(model_checkpoint, **kwargs)
    model.eval()
    return model


def _load_local_src():
    """Return ``test/_local_src.py``, loaded by file location.

    A plain ``import _local_src`` would resolve to whichever contrib model
    pytest collected first, so load the helper under a name that is unique to
    this model. Nothing goes on ``sys.path``, so this test file still imports
    the model's own code after the folder is copied out of the repository.
    """
    name = "contrib_phi3_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _register_model() -> None:
    """Load the factory out of ``src/`` and register it with vLLM.

    The engine core and its workers are forked from this process, so they
    inherit the registry this call populates (see the model README, "Offline
    inference with the Python API"). A spawned engine core would not. This is
    also why serving needs option 2 in ``contrib/README.md``: ``vllm serve``
    gives the user no script in which to call ``register_model`` at all.

    The factory is registered as a class object, not as a ``"module:Class"``
    string: vLLM resolves a string registration in a fresh interpreter, which
    cannot import a module that was loaded by file location.
    """
    from vllm import ModelRegistry

    package = _load_local_src().load()
    ModelRegistry.register_model("Phi3ForCausalLM", package.Phi3ForCausalLM)


# The (batch_size, seq_len) pairs of the recorded single-node runs: batch 1 and
# batch 4 at a 4,096-token model length. Each pair compiles its own graph.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 4096),
        (4, 4096),
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
    # method is not fork; serving under spawn needs option 2 instead.
    start_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    if start_method != "fork":
        pytest.skip(
            f"VLLM_WORKER_MULTIPROC_METHOD={start_method!r}: a spawned engine "
            "core does not inherit this process's ModelRegistry, so option-1 "
            "registration cannot reach it."
        )

    _register_model()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        model_loader=_builtin_phi3_loader,
        golden_variant="transformers_builtin_phi3",
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
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        disable_sliding_window=True,
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
