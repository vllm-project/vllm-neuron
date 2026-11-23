# SPDX-License-Identifier: Apache-2.0
"""Integration test for the contrib Granite model on Neuron.

The test serves ``ibm/PowerLM-3b`` through vLLM with the ``vllm-neuron``
plugin and compares the raw per-token logits against a HuggingFace golden
reference, as the contrib template does.

Scope
-----
The shape, the tolerance map and the tensor-parallel degree come from the
configuration that produced this model's logit evidence (see the model README's
"Validation report"): TP 4, a 2,048-token context, batch 4, 16 decoded tokens,
the two shared logit-validation prompts and the plugin's default tolerance map.
That evidence came from an online two-node run, so this single-node offline
arrangement is an adaptation of it, not a rerun of it. TP 4 fits on one
trn2.48xlarge, so this test does not need a second node.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``. The test loads the
factory from this folder's ``src/`` through the shared ``test/_local_src.py``
helper, which loads that folder by file location, and registers the class
with vLLM's ``ModelRegistry`` before it builds the engine. That is Option 1 of
``contrib/README.md`` ("How do I register a custom model with vLLM?"). It
reaches the engine because its subprocesses are forked from this one and
inherit this registry; it would not reach a spawned engine core. Making
``src/`` importable by name is not enough there — the ``register_model`` call
itself has to run in that process.

Running
-------
Serving the model needs Neuron hardware::

    pytest contrib/models/granite/test/integration/test_model.py \
        -v -s --timeout=3600 > /tmp/test_output.txt 2>&1

Set ``GRANITE_CHECKPOINT_DIR`` to a local snapshot of the pinned revision to
validate exactly the checkpoint the evidence was produced with.
"""

import importlib.util
import os
import pathlib
import sys

import pytest
import torch


def _load_local_src():
    """Load this model folder's ``test/_local_src.py`` by file location.

    Every contrib model ships a file with this name, so a plain
    ``import _local_src`` would bind whichever copy entered ``sys.modules``
    first. The per-model name used here cannot collide.
    """
    name = "contrib_granite_local_src"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()

# The architecture string in the checkpoint's config.json. vLLM resolves the
# model class through this string.
ARCHITECTURE = "GraniteForCausalLM"

MODEL_ID = "ibm/PowerLM-3b"
# The HF revision that every result in the model README was produced at.
REVISION = "38a624db4c8c9dd15d8bfb51845b7b177348e104"
# A local snapshot of REVISION, or the Hub id.
MODEL_PATH = os.environ.get("GRANITE_CHECKPOINT_DIR", MODEL_ID)
# Pin the Hub id to REVISION. A local snapshot has no Hub revision, so use None.
_REVISION = REVISION if MODEL_PATH == MODEL_ID else None

# The two prompts the plugin's logit-validation tests use for this model.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning across nine countries and containing approximately ten percent of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, encoding the genetic instructions for building and maintaining a human organism.",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The plugin's default tolerance map, which is what the Granite logit run used
# (it passed no tolerance map of its own). The argmax (divergence) lower bound
# is the strict 1e-5 of the template: a nonzero divergence is a real
# correctness bug and must not be widened to hide it.
TOL_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

# Deterministic results across runs.
torch.manual_seed(42)


# TP 4 with a 2,048-token context at batch 4: the shape the logit evidence
# covers. Each pair compiles a separate graph, so the list stays short.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve Granite through vLLM on Neuron and validate its logits.

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

    # Register the contrib architecture before the engine builds the model.
    # ``_local_src.load()`` loads this folder's ``src/`` by file location,
    # under a module name no other contrib model can take.
    #
    # Pass the class, never a ``"module:Class"`` string: vLLM inspects a
    # string entry by importing that module path in a fresh ``sys.executable``
    # subprocess, which cannot import a module loaded from a file location. A
    # class object is inspected in this process instead.
    ModelRegistry.register_model(ARCHITECTURE, getattr(_local_src.load(), ARCHITECTURE))

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    model_config = AutoConfig.from_pretrained(MODEL_PATH, revision=_REVISION)
    # The PowerLM-3b checkpoint holds FP32 weights and sets no dtype, while the
    # served model runs in bfloat16. The goldens take their target dtype from
    # this config, so it has to say bfloat16.
    if getattr(model_config, "torch_dtype", None) in (None, torch.float32):
        model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=_REVISION)
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
        revision=_REVISION,
        tokenizer_revision=_REVISION,
        tensor_parallel_size=4,
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
