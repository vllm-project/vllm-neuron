# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test of the GLM-4-0414 implementation through vLLM, in CPU mode.

The test runs the whole vLLM pipeline on CPU with a random-weight tiny
checkpoint: model build, weight loading (with the RoPE column permutation and
the fused gate/up split), prefill, decode with the in-kernel KV cache update,
and on-device greedy sampling. The HF ``Glm4ForCausalLM`` from transformers is
the independent reference.

The tiny config keeps the GLM-4-0414 specific paths of GLM-4-32B-0414:

- Four RMSNorms per decoder layer (sandwich norms), with random weights so a
  norm in the wrong place changes the logits.
- Partial interleaved RoPE (``partial_rotary_factor`` 0.5 of head_dim 128).
- The fused ``gate_up_proj`` checkpoint tensor and an untied LM head.
- GQA with 8 query heads and 2 KV heads. At TP 4 each rank has 2 query heads
  and 1 KV head, and each KV head is replicated on 2 ranks. The real model at
  TP 8 also has 1 KV head per rank (6 query heads, each KV head on 4 ranks).

Two prefill paths are covered:

- ``flash``: ``max_num_batched_tokens`` equals the model length, so the prompt
  runs in one step through ``NF.flash_attention``.
- ``segmented``: ``max_num_batched_tokens`` 512 is below the model length 1024,
  as in the integration test and the serve example in README.md (1024 below
  9216). vLLM Neuron turns segmented prefill on by default only for a
  supported segment size (512 and up), so the chunk is 512 tokens. A
  600-token prompt runs as two chunks
  through ``NF.qkv_proj`` with the in-kernel KV write and
  ``NF.segmented_attention``; the second chunk attends to the first chunk's
  K/V in the paged cache.

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in BF16
(teacher forcing). The error of vLLM against FP32 must be at most 5x the BF16
reference error, with a floor of 1% of the largest FP32 logit for the BF16
error (about one BF16 ULP at the logit scale).

The engine core runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
and the TP workers are forked from it, so the ``ModelRegistry`` registration
below is in effect everywhere the model is built. vLLM ships its own
``Glm4ForCausalLM``, and the registration replaces it.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm4/test/unit/test_tiny_glm4_e2e.py -v
"""

import contextlib
import os
import shutil

import pytest
import torch

from _local_src import src_package
from glm4_test_utils import ARCHITECTURE

# The CPU-mode environment of this test:
# - VLLM_ENABLE_V1_MULTIPROCESSING=0: run the engine core in this process, so
#   the registration below is in effect there. The TP workers are still
#   separate processes, forked from this one.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM.
# - NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker
#   needs a concrete platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NKI_SIMULATOR": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

# vLLM's multiprocess executor gives each TP worker a 240 MiB shared-memory
# response queue in /dev/shm and refuses to start when /dev/shm has less free
# space. A container started with the docker default (64 MiB) cannot run the
# TP 4 engine; start it with --shm-size or --ipc=host.
_MIN_SHM_FREE_BYTES = 256 << 20

NUM_LAYERS = 2
MAX_TOKENS = 4
TOP_LOGPROBS = 5
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01
QK_WEIGHT_SCALE = 4.0

# name -> (prompt, max_model_len, max_num_batched_tokens, prefill buckets)
_PREFILL_CASES = {
    "flash": (list(range(3, 40)), 256, 256, [128, 256]),
    "segmented": ([3 + (7 * i) % 500 for i in range(600)], 1024, 512, [512]),
}


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set the env vars for the tiny E2E tests. Restore them at teardown."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.skip("The tiny E2E test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    if os.path.isdir("/dev/shm"):
        free = shutil.disk_usage("/dev/shm").free
        if free < _MIN_SHM_FREE_BYTES:
            pytest.skip(
                f"/dev/shm has {free >> 20} MiB free; vLLM's TP 4 engine needs "
                f"{_MIN_SHM_FREE_BYTES >> 20} MiB. Start the container with "
                "--shm-size or --ipc=host."
            )
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _tiny_config():
    from transformers import Glm4Config

    return Glm4Config(
        vocab_size=512,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture(scope="module")
def tiny_glm4_dir(tmp_path_factory):
    """Make a random-weight tiny GLM-4 checkpoint in BF16, one time."""
    from transformers import Glm4ForCausalLM

    model_dir = str(tmp_path_factory.mktemp("tiny_glm4"))
    torch.manual_seed(42)
    model = Glm4ForCausalLM(_tiny_config())
    # HF initializes norm weights to 1. Random values make a norm in the wrong
    # place visible. The default weight scale (std 0.02) gives near-uniform
    # attention, which hides a RoPE error; larger Q/K weights make the scores
    # position sensitive.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
            elif name.endswith(("q_proj.weight", "k_proj.weight")):
                param.mul_(QK_WEIGHT_SCALE)
    model.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir


@contextlib.contextmanager
def _llm(
    model_dir: str,
    tp_size: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    buckets: list[int],
):
    """Yield an LLM. Shut down its engine core at exit, also after a failure."""
    from vllm import LLM, ModelRegistry

    # vLLM has its own Glm4ForCausalLM, so the contrib class must replace it in
    # the registry before the engine resolves the architecture.
    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=128,
        tensor_parallel_size=tp_size,
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
                "num_batched_tokens_buckets": buckets,
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def _hf_teacher_forced_logits(model_dir: str, prompt: list[int], tokens, dtype):
    from transformers import Glm4ForCausalLM

    model = Glm4ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([prompt + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(prompt) - 1
    return logits[start : start + len(tokens)]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("prefill", list(_PREFILL_CASES))
@pytest.mark.parametrize("tp_size", [4])
def test_tiny_glm4_logits_match_hf(tiny_glm4_dir, tp_size: int, prefill: str):
    """The vLLM raw logits of prefill and decode must match the HF reference."""
    from vllm import SamplingParams

    from vllm_neuron.utils.bucket_utils import get_default_kv_segment_size_buckets

    prompt, max_model_len, max_num_batched_tokens, buckets = _PREFILL_CASES[prefill]
    # The runner turns segmented prefill on exactly when this default is not
    # None. Check that each case takes the prefill path it is named for.
    segment_buckets = get_default_kv_segment_size_buckets(
        max_num_batched_tokens, max_model_len
    )
    if prefill == "segmented":
        assert segment_buckets == [max_num_batched_tokens]
        assert len(prompt) > max_num_batched_tokens
    else:
        assert segment_buckets is None
    with _llm(
        tiny_glm4_dir, tp_size, max_model_len, max_num_batched_tokens, buckets
    ) as llm:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                logprobs=TOP_LOGPROBS,
            ),
        )
    assert len(outputs) == 1
    completion = outputs[0].outputs[0]
    tokens = list(completion.token_ids)
    assert len(tokens) == MAX_TOKENS
    assert completion.logprobs is not None and len(completion.logprobs) == len(
        tokens
    ), "vLLM did not return logprobs for each generated token"

    ref_fp32 = _hf_teacher_forced_logits(tiny_glm4_dir, prompt, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_glm4_dir, prompt, tokens, torch.bfloat16)

    base_floor = BASE_FLOOR_FRACTION * ref_fp32.abs().max().item()
    for step, (token, step_logprobs) in enumerate(zip(tokens, completion.logprobs)):
        ids = sorted(step_logprobs)
        assert token in ids
        got = torch.tensor([step_logprobs[i].logprob for i in ids])
        assert torch.isfinite(got).all(), f"Step {step}: non-finite logits {got}"
        fp32 = ref_fp32[step, ids]
        bf16 = ref_bf16[step, ids]
        tgt_err = (got - fp32).abs().max().item()
        base_err = max((bf16 - fp32).abs().max().item(), base_floor)
        print(
            f"{prefill} step {step}: token={token} tgt_linf={tgt_err:.5f} "
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
                f"Step {step}: vLLM selected token {token}, but the FP32 "
                f"argmax is {top2.indices[0].item()} (margin {margin:.4f})."
            )
