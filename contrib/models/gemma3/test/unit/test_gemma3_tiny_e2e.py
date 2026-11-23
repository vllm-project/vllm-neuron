# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test of Gemma3 (rnj-1 layout) through vLLM in CPU mode.

The test runs the full vLLM pipeline on CPU with a random-weight tiny
checkpoint: registration, model construction on the meta device, weight
loading (FP32 checkpoint, BF16 served dtype, tied LM head), prefill, decode
and on-device greedy sampling, with the raw logits of the whole vocabulary
returned through ``logprobs_mode="raw_logits"``. The HF ``Gemma3ForCausalLM``
(eager attention) gives the independent reference.

The tiny config (``gemma3_test_utils.TINY_CONFIG``) keeps the rnj-1 paths:
the ``1 + w`` RMSNorms (four per layer, the final norm and the per-head
QK-norm), the ``sqrt(hidden_size)`` embedding scale, YaRN RoPE with its
attention factor, the ``query_pre_attn_scalar`` softmax scale, the GELU-tanh
MLP, the final logit soft cap and the tied embeddings. It has 4 Q heads and
2 KV heads, so TP 2 gives each rank one KV head: the decode branch that rnj-1
takes at TP 8 and TP 16 on device (a 3D cache and a 2D block table).

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in
BF16 (teacher forcing). The L-inf error of the vLLM logits against FP32 must
be at most 5x the HF BF16 error (the default ``max_linf_ratio`` of
``assert_close_three_way``). The BF16 error has a floor of 1% of the largest
FP32 logit, which is about one BF16 ULP at the logit scale.

The test registers the architecture with vLLM's ``ModelRegistry`` before it
builds the engine (option 1 in ``contrib/README.md``). vLLM ships its own
``Gemma3ForCausalLM``, so the registration replaces it. The engine core runs
in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are
forked from it, so the registration is in effect everywhere the model is
built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/gemma3/test/unit/test_gemma3_tiny_e2e.py -v -s
"""

import os
import shutil

import pytest
import torch

from vllm_neuron import envs

from _local_src import src_package
from gemma3_test_utils import ARCHITECTURE, make_tiny_checkpoint

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

# 60 prompt tokens and 8 decode steps (positions 60 to 67): the decode steps
# cross a KV block boundary (block size 32) and pass the tiny YaRN original
# context of 64 positions.
PROMPT = list(range(3, 63))
MAX_TOKENS = 8
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01

# vLLM's worker message queues reserve this much /dev/shm (sparse) at startup
# and refuse to start with less free space.
MIN_SHM_BYTES = 512 * 1024 * 1024


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
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_gemma3_e2e")))


def _generate(model_dir: str, tp_size: int):
    from vllm import LLM, ModelRegistry, SamplingParams

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
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
    from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM

    model = HFGemma3ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("tp_size", [2])
def test_logits_match_hf(tiny_checkpoint, tp_size):
    """The vLLM raw logits of the full vocabulary match HF (three-way)."""
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("This test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    if os.path.isdir("/dev/shm") and shutil.disk_usage("/dev/shm").free < MIN_SHM_BYTES:
        pytest.skip(
            "vLLM worker queues need at least 512 MiB of free /dev/shm "
            "(for example docker run --shm-size=1g)."
        )

    completion = _generate(tiny_checkpoint, tp_size)
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
        assert ids == list(range(vocab_size)), (
            f"Step {step}: vLLM returned {len(ids)} logits, not {vocab_size}."
        )
        got = torch.tensor([step_logprobs[i].logprob for i in ids])
        assert torch.isfinite(got).all(), f"Step {step}: non-finite logits."
        # The final soft cap bounds every logit to (-30, 30).
        assert got.abs().max().item() <= 30.0, f"Step {step}: logit over the cap."
        tgt_err = (got - ref_fp32[step]).abs().max().item()
        base_err = max((ref_bf16[step] - ref_fp32[step]).abs().max().item(), base_floor)
        print(
            f"TP {tp_size} step {step}: token={token} tgt_linf={tgt_err:.5f} "
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
