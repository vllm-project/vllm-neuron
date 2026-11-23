# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test of OLMo-3 through vLLM in CPU mode (TP 2).

The test runs the full vLLM pipeline on CPU with a random-weight tiny
checkpoint: registration, model construction on the meta device, weight
loading, prefill, decode and on-device greedy sampling, with the raw logits of
the whole vocabulary returned through ``logprobs_mode="raw_logits"``. The HF
``Olmo3ForCausalLM`` (eager attention) gives the independent reference.

The tiny config (``olmo3_test_utils.TINY_CONFIG``) keeps the OLMo-3 paths:

* three sliding-window layers and one full-attention layer, with a window of
  16 tokens. The 40-token prompt and the 8 decode steps are longer than the
  window, so the window mask is active in the prefill and in the decode
  attention;
* YaRN RoPE with an explicit attention factor on the full-attention layer and
  the default RoPE on the sliding-window layers;
* the full-width QK-norm across the two TP ranks and the post-norm layer;
* an untied LM head and a vocabulary (251) that does not divide by 2, so the
  embedding and the LM head pad one row and the sampler masks it.

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in
BF16 (teacher forcing). The L-inf error of the vLLM logits against FP32 must
be at most 5x the HF BF16 error (the default ``max_linf_ratio`` of
``assert_close_three_way``). The BF16 error has a floor of 1% of the largest
FP32 logit, which is about one BF16 ULP at the logit scale.

The test registers the architecture with vLLM's ``ModelRegistry`` before it
builds the engine (option 1 in ``contrib/README.md``). vLLM ships its own
``Olmo3ForCausalLM`` (the Transformers backend), so the registration replaces
it. The engine core runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
and the TP workers are forked from it, so the registration is in effect
everywhere the model is built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/olmo3/test/unit/test_olmo3_tiny_e2e.py -v -s
"""

import os

import pytest
import torch

from vllm_neuron import envs

from _local_src import src_package
from olmo3_test_utils import ARCHITECTURE, TINY_WINDOW, make_tiny_checkpoint

# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the engine core subprocess hangs in CPU
#   mode. In-process mode prevents this. The TP workers still start.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM.
# - NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker
#   needs a concrete platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NKI_SIMULATOR": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

TP_SIZE = 2
PROMPT = list(range(3, 43))
MAX_TOKENS = 8
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01

assert len(PROMPT) + MAX_TOKENS > 2 * TINY_WINDOW


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set the env vars for the tiny E2E test. Restore them at teardown."""
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_olmo3_e2e")))


def _generate(model_dir: str):
    from vllm import LLM, ModelRegistry, SamplingParams

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=TP_SIZE,
        dtype="bfloat16",
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        block_size=32,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=32,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        # The async path returns before it computes logprobs.
        async_scheduling=False,
        # The device integration test compiles at -O3. In CPU mode the level
        # has no effect, but the engine must accept it.
        optimization_level=3,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [128],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )
    try:
        outputs = llm.generate(
            [{"prompt_token_ids": PROMPT}],
            SamplingParams(
                temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, logprobs=-1
            ),
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert len(outputs) == 1
    return outputs[0].outputs[0]


def _hf_teacher_forced_logits(model_dir: str, tokens: list[int], dtype):
    from transformers import Olmo3ForCausalLM as HFOlmo3ForCausalLM

    model = HFOlmo3ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


@pytest.mark.usefixtures("dist_cleanup")
def test_tp2_logits_match_hf(tiny_checkpoint):
    """TP 2: the vLLM raw logits of the full vocabulary match HF (three-way)."""
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("This test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")

    completion = _generate(tiny_checkpoint)
    tokens = list(completion.token_ids)
    assert len(tokens) == MAX_TOKENS
    assert completion.logprobs is not None and len(completion.logprobs) == len(
        tokens
    ), "vLLM did not return logprobs for each generated token."

    ref_fp32 = _hf_teacher_forced_logits(tiny_checkpoint, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_checkpoint, tokens, torch.bfloat16)
    vocab_size = ref_fp32.shape[-1]

    base_floor = BASE_FLOOR_FRACTION * ref_fp32.abs().max().item()
    for step, (token, step_logprobs) in enumerate(zip(tokens, completion.logprobs)):
        ids = sorted(step_logprobs)
        # The full vocabulary, without the padded LM-head row.
        assert ids == list(range(vocab_size)), (
            f"Step {step}: vLLM returned {len(ids)} logits, not {vocab_size}."
        )
        got = torch.tensor([step_logprobs[i].logprob for i in ids])
        assert torch.isfinite(got).all(), f"Step {step}: non-finite logits."
        tgt_err = (got - ref_fp32[step]).abs().max().item()
        base_err = max((ref_bf16[step] - ref_fp32[step]).abs().max().item(), base_floor)
        print(
            f"step {step}: token={token} tgt_linf={tgt_err:.5f} "
            f"base_linf={base_err:.5f} ratio={tgt_err / base_err:.2f}"
        )
        assert tgt_err <= LINF_RATIO * base_err, (
            f"Step {step}: the vLLM logit error {tgt_err:.5f} is more than "
            f"{LINF_RATIO}x the BF16 reference error {base_err:.5f}."
        )

        # Greedy token: it must be the FP32 argmax when the margin is larger
        # than the error bound. A smaller margin is a valid tie.
        top2 = torch.topk(ref_fp32[step], 2)
        margin = (top2.values[0] - top2.values[1]).item()
        if margin > 2 * LINF_RATIO * base_err:
            assert token == top2.indices[0].item(), (
                f"Step {step}: vLLM selected token {token}, but the FP32 "
                f"argmax is {top2.indices[0].item()} (margin {margin:.4f})."
            )
