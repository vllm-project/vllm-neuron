# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test for StarCoder2 in CPU mode.

The test runs the full vLLM pipeline on CPU with a random-weight tiny
checkpoint: model load (FP32 checkpoint cast to BF16), prefill, decode and
sampling, at TP 4. The HF ``Starcoder2ForCausalLM`` built into transformers is
the independent reference.

The tiny config keeps the StarCoder2-specific paths and the TP 4 layout of the
real checkpoint:

- LayerNorm with a bias, a bias on Q, K, V, O, ``c_fc`` and ``c_proj``, and a
  non-gated GELU tanh MLP. The test gives every bias and every norm a random
  value, so a dropped bias or a wrong norm changes the logits.
- 8 query heads and 2 KV heads: TP 4 gives 2 query heads and 1 KV head per
  rank, with 2 ranks for each KV head (the real model has 6 query heads per
  rank, and the same KV layout).
- Tied word embeddings, and the RoPE theta of the real checkpoint.

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in BF16
(teacher forcing). The error of vLLM against FP32 must be at most 5x the BF16
reference error, the default ``max_linf_ratio`` of ``assert_close_three_way``.
The BF16 reference error has a floor of 1% of the largest FP32 logit, about one
BF16 ULP at the logit scale.

The test registers the contrib architecture with vLLM's ``ModelRegistry``
before it builds the engine, because a contrib model is not in the plugin's
registry and vLLM maps ``Starcoder2ForCausalLM`` to its own Transformers
backend. The engine core runs in this process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are forked from it,
so the registration is in effect everywhere the model is built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/starcoder2/test/unit/test_tiny_e2e.py -v
"""

import contextlib
import os

import pytest
import torch
from transformers import Starcoder2Config, Starcoder2ForCausalLM

from starcoder2_test_utils import ARCHITECTURE, TP_SIZE, src_package

# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the V1 engine core subprocess hangs in
#   CPU mode. In-process mode prevents this. TP workers still start safely.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM.
# - NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker needs a concrete
#   platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

NUM_LAYERS = 2
PROMPT = list(range(3, 40))
MAX_TOKENS = 4
TOP_LOGPROBS = 5
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01

TINY_STARCODER2_CONFIG = Starcoder2Config(
    vocab_size=512,
    hidden_size=1024,
    intermediate_size=2048,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=512,
    norm_epsilon=1e-5,
    hidden_act="gelu_pytorch_tanh",
    use_bias=True,
    rope_parameters={"rope_type": "default", "rope_theta": 999999.4420358813},
    sliding_window=4096,
    tie_word_embeddings=True,
    attention_dropout=0.0,
    residual_dropout=0.0,
    embedding_dropout=0.0,
    bos_token_id=0,
    eos_token_id=0,
)


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
def tiny_starcoder2_dir(tmp_path_factory):
    """Make a random-weight tiny StarCoder2 checkpoint in FP32, one time."""
    model_dir = str(tmp_path_factory.mktemp("tiny_starcoder2"))
    torch.manual_seed(42)
    model = Starcoder2ForCausalLM(TINY_STARCODER2_CONFIG)
    # HF initializes the biases to zero and the norms to one. Random values
    # make the test sensitive to a dropped bias or a wrong norm.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
            elif name.endswith("bias"):
                param.copy_(0.05 * torch.randn_like(param))
    model.to(torch.float32).save_pretrained(model_dir)
    return model_dir


@contextlib.contextmanager
def _llm(model_dir: str, tp_size: int):
    """Yield an LLM. Shut down its engine core at exit, also after a failure."""
    from vllm import LLM, ModelRegistry

    # vLLM maps this architecture to its Transformers backend, so the contrib
    # class must replace that entry before the engine resolves it.
    ModelRegistry.register_model(ARCHITECTURE, src_package().Starcoder2ForCausalLM)
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=256,
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
                "quantization": "bf16",
                "num_batched_tokens_buckets": [128, 256],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def _hf_teacher_forced_logits(model_dir: str, tokens: list[int], dtype):
    model = Starcoder2ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


@pytest.mark.usefixtures("dist_cleanup")
def test_tp4_logits_match_hf(tiny_starcoder2_dir):
    """TP 4: the vLLM raw logits must match the HF reference (three-way)."""
    from vllm import SamplingParams

    with _llm(tiny_starcoder2_dir, tp_size=TP_SIZE) as llm:
        outputs = llm.generate(
            [{"prompt_token_ids": PROMPT}],
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

    ref_fp32 = _hf_teacher_forced_logits(tiny_starcoder2_dir, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_starcoder2_dir, tokens, torch.bfloat16)

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

        # Greedy token: compare with the FP32 argmax when the margin is
        # larger than the error bound. A smaller margin is a valid tie.
        top2 = torch.topk(ref_fp32[step], 2)
        margin = (top2.values[0] - top2.values[1]).item()
        if margin > 2 * LINF_RATIO * base_err:
            assert token == top2.indices[0].item(), (
                f"Step {step}: vLLM selected token {token}, but the FP32 "
                f"argmax is {top2.indices[0].item()} (margin {margin:.4f})."
            )
