# SPDX-License-Identifier: Apache-2.0
"""Integration test for the MiMo-V2.5 contrib model, served through vLLM on Neuron.

Adapted from ``contrib/models/template/test/integration/test_model.py``. It serves
``MiMoV2ForCausalLM`` through vLLM (with the ``vllm-neuron`` plugin) at TP=64 /
EP=64 on one trn2.48xlarge and compares the per-token logits against a
HuggingFace reference.

Registering the model
---------------------
Contrib models are NOT wired into ``vllm_neuron/model/registry.py``. This test
registers the architecture itself before it constructs ``LLM`` (Option 1 in
``contrib/README.md``): it uses the installed ``mimo_v2_5`` package when
``pip install -e contrib/models/mimo_v2_5`` has been run, and otherwise loads
this folder's ``src/`` under that name. The factory CLASS is registered (not a
lazy ``"module:Class"`` string) because vLLM inspects lazy entries in a fresh
subprocess that would not see a source-tree import; the engine and worker
processes inherit the registration because vLLM forks them on Neuron.

Running
-------
Needs a trn2.48xlarge (64 logical cores). Run it manually per the model README:

    MIMO_V2_5_CHECKPOINT=/path/to/MiMo-V2.5 \\
    pytest contrib/models/mimo_v2_5/test/integration/test_model.py \\
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

Reference logits
----------------
``get_cached_reference_goldens`` runs the HF model on the host (FP32 and BF16
passes) and caches the result, so only the first run pays for it. MiMo-V2.5 is
~310B parameters: the FP32 pass alone needs ~1.25 TB of host RAM, which a
trn2.48xlarge (2 TiB) has, but it is slow. The released checkpoint is FP8
blockwise; if the HF FP8 path refuses to dequantize on CPU in your transformers
version, point ``MIMO_V2_5_REFERENCE_CHECKPOINT`` at a BF16 re-export (the
Neuron side loads either; its loaders probe for ``*_scale_inv`` keys).

Logit validation
----------------
Helpers from the ``vllm_neuron.accuracy`` API:

  * ``get_cached_reference_goldens`` computes (and caches) the HF reference
    logits: ``fp32_logits`` (the FP32 baseline) and ``dtype_logits`` (a
    same-dtype pass teacher-forced along the FP32 greedy sequence).
  * ``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so it returns
    dense per-token logits.
  * ``multi_prompt_logit_validation`` compares the two, flagging any argmax
    divergence and any top-K logit error outside the tolerance map.
"""

import importlib.util
import os
import sys

import pytest
import torch

# A local path or a Hugging Face Hub id. The hub repo is ~294 GiB; see the README
# for an allow-listed download of only what the text decoder reads.
MODEL_PATH = os.environ.get("MIMO_V2_5_CHECKPOINT", "XiaomiMiMo/MiMo-V2.5")
REFERENCE_PATH = os.environ.get("MIMO_V2_5_REFERENCE_CHECKPOINT", MODEL_PATH)

# The HF reference logits are computed from these same prompts.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# Starting point: the template's bf16 dense-golden convention. The argmax
# (divergence) lower bound stays strict at 1e-5 -- a nonzero divergence is a
# real correctness bug. The upper bounds have NOT been tuned on 0.29 yet; the
# eager attention runs in fp32 (tighter than a fused bf16 kernel) but the 47
# bf16 MoE layers dominate the drift. Widen only with evidence.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

TENSOR_PARALLEL_SIZE = 64
EP_DEGREE = 64

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")

# Deterministic results across runs.
torch.manual_seed(42)


def register_mimo_v2_5() -> None:
    """Register MiMoV2ForCausalLM with vLLM before ``LLM(...)`` is built."""
    if "mimo_v2_5" not in sys.modules:
        try:
            import mimo_v2_5  # noqa: F401  (installed package)
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                "mimo_v2_5",
                os.path.join(_SRC_DIR, "__init__.py"),
                submodule_search_locations=[_SRC_DIR],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["mimo_v2_5"] = module
            spec.loader.exec_module(module)

    from vllm import ModelRegistry

    from mimo_v2_5 import ARCHITECTURE, MiMoV2ForCausalLM

    ModelRegistry.register_model(ARCHITECTURE, MiMoV2ForCausalLM)


def _text_only_overrides(config):
    """Delete the config keys that route this text port elsewhere.

    ``quantization_config`` would make vLLM derive quantization="fp8", which
    the Neuron platform rejects; ``vision_config`` makes vLLM rewrite the
    architecture to ``MiMoV2OmniForCausalLM``. See ``scripts/run.py``.
    """
    for key in ("quantization_config", "vision_config"):
        if hasattr(config, key):
            delattr(config, key)
    return config


def _reference_loader(model_checkpoint, dtype, config=None):
    """HF reference loader for MiMo-V2.5's remote modeling code.

    MiMo's attention adds a learned sink column, which SDPA cannot express, so
    let the stock loader fall back to eager attention when SDPA is rejected.
    """
    from vllm_neuron.accuracy import init_hf_model

    return init_hf_model(
        model_checkpoint, dtype, config=config, eager_attn_fallback=True
    )


# Each (batch_size, seq_len) pair compiles its own prefill and decode graphs
# (~5-10 min each at TP=64), so keep the list short. seq_len is both
# max_model_len and max_num_batched_tokens: this port prefills single-shot.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 1024),
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve MiMo-V2.5 through vLLM and validate its logits.

    Args:
        batch_size: Number of prompts to run in a single batch.
        seq_len: Maximum model (context + generation) length to configure.
    """
    register_mimo_v2_5()

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing MiMo-V2.5 with parameters: {batch_size=}, {seq_len=}")

    model_config = AutoConfig.from_pretrained(REFERENCE_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # model_id only builds the golden cache key; model_checkpoint is what the
    # reference model loads from.
    cached_goldens = get_cached_reference_goldens(
        model_id="XiaomiMiMo/MiMo-V2.5",
        model_checkpoint=REFERENCE_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        model_loader=_reference_loader,
    )

    # Four arguments are required by logit validation, not optional tuning:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits, not log-probabilities.
    #   * async_scheduling=False keeps logprobs available.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy.
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_expert_parallel=True,
        trust_remote_code=True,
        hf_overrides=_text_only_overrides,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        # One KV buffer per layer: on the native route a buffer shared by several
        # layers (vLLM's hybrid KV manager) turns every KV write into a whole-pool
        # copy (see MiMoV2ForCausalLM.bind_kv_cache).
        disable_hybrid_kv_cache_manager=True,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "ep_degree": EP_DEGREE,
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
