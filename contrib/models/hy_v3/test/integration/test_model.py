# SPDX-License-Identifier: Apache-2.0
"""Integration accuracy test for Hy3 (tencent/Hy3-FP8), served through vLLM on Neuron.

The full model is 295B parameters, too large for an FP32 HuggingFace reference.
The test therefore builds a reduced checkpoint (the first ``NUM_LAYERS`` decoder
layers of the real checkpoint, with real weights) and compares the per-token
logits of the Neuron FP8 model with the HuggingFace reference of the same
layers dequantized to BF16 (``scripts/make_mini.py``):

  * ``<mini>/fp8``  - original FP8 weights and scales, served by vLLM Neuron
  * ``<mini>/bf16`` - the same weights dequantized to BF16, loaded by HF

Registration: install the contrib package once (``pip install -e
contrib/models/hy_v3``). It registers ``HYV3ForCausalLM`` in every vLLM process
(the engine core and Neuron workers are spawned) through a
``vllm.general_plugins`` entry point.

Environment:
  HY3_CKPT       Full tencent/Hy3-FP8 checkpoint directory (required unless
                 HY3_MINI_CKPT already holds a reduced checkpoint).
  HY3_MINI_CKPT  Where the reduced checkpoint is built / cached
                 (default: /tmp/hy3_mini<NUM_LAYERS>).
  HY3_KV_SCALE_PROFILE  KV scale profile for the FP8-KV cases (the checkpoint
                 ships no KV scales; scripts/calibrate_kv_scales.py makes one, a
                 full-model profile also covers the reduced model). Without it
                 the FP8-KV cases are skipped.

Run on a trn2.48xlarge (TP=64, full EP):

    HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/integration/test_model.py \
        -v -s --timeout=7200
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

NUM_LAYERS = 4
MINI_CKPT = Path(os.environ.get("HY3_MINI_CKPT", f"/tmp/hy3_mini{NUM_LAYERS}"))
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "人工智能的未来发展趋势是",
    "In 1969, Neil Armstrong became the first person to",
]

OUTPUT_LENGTH = 16

# The model runs static per-tensor FP8 weights and activations against a BF16
# (dequantized-weight) golden, so the top-K logit bounds are wider than the
# bf16 dense-golden defaults. Measured on the 4-layer checkpoint: top-K
# relative error <= 0.06; the bounds keep ~2x headroom.
TOL_MAP = {
    "5": (1e-5, 0.1),
    "50": (1e-5, 0.12),
    "1000": (1e-5, 0.15),
    "all": (1e-5, 0.15),
}

# Argmax divergence tolerance, in logits. FP8 activation quantization moves
# logits by a few hundredths, so at near-ties of the (very flat) 4-layer
# distribution the argmax can flip against the BF16 golden: measured flips have
# a golden top-1 vs top-2 gap of 0.06-0.125. The template's 1-BF16-ULP bound
# is meant for BF16 models; 0.25 still fails any real defect, which shows up
# as gaps of several logits.
DIVERGENCE_TOL = 0.25

torch.manual_seed(42)


@pytest.fixture(scope="module")
def mini_ckpt() -> Path:
    """Build (once) the reduced FP8 + BF16 checkpoint pair."""
    if not (MINI_CKPT / "fp8" / "config.json").exists():
        full = os.environ.get("HY3_CKPT")
        if not full:
            pytest.skip("set HY3_CKPT to the tencent/Hy3-FP8 checkpoint directory")
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPTS / "make_mini.py"),
                full,
                str(MINI_CKPT),
                str(NUM_LAYERS),
            ],
            check=True,
        )
    return MINI_CKPT


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len, prefix_caching, kv_cache_dtype",
    [
        (1, 2048, False, "auto"),  # flash-attention prefill
        (4, 2048, True, "auto"),  # segmented prefill (prefix caching), batch-4 decode
        # FP8 KV cache (packed K, as served): the same two prefill paths
        (1, 2048, False, "fp8"),
        (4, 2048, True, "fp8"),
    ],
)
def test_model_accuracy(
    mini_ckpt: Path,
    batch_size: int,
    seq_len: int,
    prefix_caching: bool,
    kv_cache_dtype: str,
) -> None:
    """Serve the reduced Hy3 checkpoint through vLLM and validate its logits."""
    if kv_cache_dtype == "fp8" and not os.environ.get("HY3_KV_SCALE_PROFILE"):
        pytest.skip("set HY3_KV_SCALE_PROFILE to a KV scale profile for the FP8-KV cases")
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    try:
        from hy_v3_neuron import register
    except ImportError:
        pytest.fail(
            "install the contrib package first: pip install -e contrib/models/hy_v3"
        )
    register()

    fp8_path, bf16_path = str(mini_ckpt / "fp8"), str(mini_ckpt / "bf16")
    case = f"{batch_size=}, {seq_len=}, {prefix_caching=}, {kv_cache_dtype=}"
    print(f"Testing Hy3 with {case}")

    model_config = AutoConfig.from_pretrained(bf16_path)
    tokenizer = AutoTokenizer.from_pretrained(bf16_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=f"tencent/Hy3-FP8-mini{NUM_LAYERS}-bf16",
        model_checkpoint=bf16_path,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
    )

    neuron_config = {
        "on_device_sampling_config": {"all_greedy": True},
        "num_batched_tokens_buckets": [seq_len],
        "num_seqs_buckets": [batch_size],
    }
    if prefix_caching:
        neuron_config["kv_segment_size_buckets"] = [seq_len]
    if kv_cache_dtype == "fp8":
        neuron_config["fp8_packed_kv"] = True

    # max_logprobs / logprobs_mode / async_scheduling / all_greedy are required
    # by the logit validation (see contrib/models/template).
    llm = LLM(
        model=fp8_path,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        block_size=32,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=prefix_caching,
        kv_cache_dtype=kv_cache_dtype,
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
            divergence_n_ulps=None,
            divergence_difference_tol=DIVERGENCE_TOL,
        )
        assert result.passed, f"Logit validation failed for {case}"
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for {case}")
