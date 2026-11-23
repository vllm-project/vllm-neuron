# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test for OLMo in CPU mode.

The test runs the full vLLM pipeline on CPU at TP 2 with a random-weight tiny
checkpoint: model build, weight loading (FP32 checkpoint, BF16 compute),
prefill, decode and on-device greedy sampling. HF ``OlmoForCausalLM`` gives the
independent reference.

The tiny config keeps the OLMo-specific paths of ``allenai/OLMo-1B-hf``:
non-parametric LayerNorm, default RoPE, tied embeddings, an FP32 checkpoint,
and more than one KV head on each rank (4 at TP 2, 2 at TP 8 for the 1B
model), so decode takes the per-head block table path.

``test_tp2_logits_match_hf`` runs twice: once through the NKI kernels in the
simulator (``NKI_SIMULATOR=1``, the device code path) and once through the
torch fallback of the plugin (``NKI_SIMULATOR=0``).

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in BF16
(teacher forcing). The error of vLLM against FP32 must be at most 5x the BF16
reference error, the default ``max_linf_ratio`` of
``vllm_neuron.accuracy.testing.assert_close_three_way``. The BF16 reference
error has a floor of 1% of the largest FP32 logit.

The test registers the contrib architecture with vLLM's ``ModelRegistry``
before it builds the engine (contrib/README.md, option 1). The engine core runs
in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are
forked from it, so the registration is in effect where the model is built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo/test/unit/test_tiny_e2e.py -v
"""

import contextlib
import os

import pytest
import torch
from transformers import OlmoConfig as HFOlmoConfig
from transformers import OlmoForCausalLM as HFOlmoForCausalLM

from _local_src import src_package

ARCHITECTURE = "OlmoForCausalLM"

# - VLLM_ENABLE_V1_MULTIPROCESSING=0: run the engine core in this process; the
#   engine-core subprocess hangs in CPU mode. TP workers still start safely.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM.
# - NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker needs a platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

TP_SIZE = 2
PROMPT = list(range(3, 40))
MAX_TOKENS = 4
TOP_LOGPROBS = 5
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01

# 8 Q heads and 8 KV heads: 4 of each on a rank at TP 2. The hidden size is a
# multiple of 128, so the decode kernel accepts the geometry.
TINY_OLMO_CONFIG = HFOlmoConfig(
    vocab_size=256,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    max_position_embeddings=256,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    attention_bias=False,
    clip_qkv=None,
    tie_word_embeddings=True,
    pad_token_id=1,
    bos_token_id=None,
    eos_token_id=2,
)


@contextlib.contextmanager
def _env(overrides: dict[str, str]):
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def tiny_olmo_dir(tmp_path_factory):
    """Make a random-weight tiny OLMo checkpoint in FP32, one time."""
    model_dir = str(tmp_path_factory.mktemp("tiny_olmo"))
    torch.manual_seed(42)
    model = HFOlmoForCausalLM(TINY_OLMO_CONFIG)
    # A larger weight scale than the HF init gives logits with clear margins.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.08)
    model.to(torch.float32).save_pretrained(model_dir)
    return model_dir


@contextlib.contextmanager
def _llm(model_dir: str):
    """Yield an LLM. Shut down its engine core at exit, also after a failure."""
    from vllm import LLM, ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=256,
        block_size=128,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=128,
        max_logprobs=TOP_LOGPROBS,
        logprobs_mode="raw_logits",
        # The async sampler output has no logprobs. Use sync scheduling.
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [128, 256],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def _hf_teacher_forced_logits(model_dir: str, tokens: list[int], dtype):
    model = HFOlmoForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


@pytest.mark.parametrize("nki_simulator", ["1", "0"], ids=["nki", "torch"])
def test_tp2_logits_match_hf(tiny_olmo_dir, nki_simulator):
    """TP 2: the vLLM raw logits must match the HF reference (three-way)."""
    from vllm import SamplingParams

    with _env({**_ENV_OVERRIDES, "NKI_SIMULATOR": nki_simulator}):
        with _llm(tiny_olmo_dir) as llm:
            outputs = llm.generate(
                [{"prompt_token_ids": PROMPT}],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                    ignore_eos=True,
                    logprobs=TOP_LOGPROBS,
                ),
            )
    completion = outputs[0].outputs[0]
    tokens = list(completion.token_ids)
    assert len(tokens) == MAX_TOKENS
    assert completion.logprobs is not None and len(completion.logprobs) == len(
        tokens
    ), "vLLM did not return logprobs for each generated token"

    ref_fp32 = _hf_teacher_forced_logits(tiny_olmo_dir, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_olmo_dir, tokens, torch.bfloat16)

    base_floor = BASE_FLOOR_FRACTION * ref_fp32.abs().max().item()
    for step, (token, step_logprobs) in enumerate(zip(tokens, completion.logprobs)):
        ids = sorted(step_logprobs)
        assert token in ids
        got = torch.tensor([step_logprobs[i].logprob for i in ids])
        fp32 = ref_fp32[step, ids]
        bf16 = ref_bf16[step, ids]
        tgt_err = (got - fp32).abs().max().item()
        base_err = max((bf16 - fp32).abs().max().item(), base_floor)
        print(
            f"step {step}: token={token} tgt_linf={tgt_err:.5f} "
            f"base_linf={base_err:.5f} ratio={tgt_err / base_err:.2f}"
        )
        assert tgt_err <= LINF_RATIO * base_err, (
            f"Step {step}: the vLLM logit error {tgt_err:.5f} is more than "
            f"{LINF_RATIO}x the BF16 reference error {base_err:.5f}."
        )

        # Greedy token: compare with the FP32 argmax when the margin is larger
        # than the error bound. A smaller margin is a valid tie.
        top2 = torch.topk(ref_fp32[step], 2)
        margin = (top2.values[0] - top2.values[1]).item()
        if margin > 2 * LINF_RATIO * base_err:
            assert token == top2.indices[0].item(), (
                f"Step {step}: vLLM selected token {token}, but the FP32 argmax "
                f"is {top2.indices[0].item()} (margin {margin:.4f})."
            )
