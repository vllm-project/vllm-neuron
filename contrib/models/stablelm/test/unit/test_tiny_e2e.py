# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end test for StableLM in CPU mode.

The test runs the full vLLM pipeline on CPU with a random-weight tiny
checkpoint: registration, the vLLM model protocol checks, model build, weight
loading, prefill, decode, on-device greedy sampling and the raw-logit return.
The HF ``StableLmForCausalLM`` gives the independent reference.

The tiny checkpoint keeps head_dim 80, the 0.25 partial rotary factor and the
LayerNorm with a bias (see ``stablelm_test_utils.TINY_OVERRIDES``). The test
sets ``NKI_SIMULATOR=1``, so the NKI kernels that accept the shapes run in the
simulator instead of their torch fallbacks.

Accuracy rule: the test feeds the vLLM tokens back into HF in FP32 and in BF16
(teacher forcing). On every vocabulary id, the error of vLLM against FP32 must
be at most 5x the BF16 reference error. This is the default
``max_linf_ratio`` of ``assert_close_three_way``. The BF16 reference error has
a floor of 1% of the largest FP32 logit, which is about one BF16 ULP at the
logit scale.

The test registers the contrib architecture with vLLM's ``ModelRegistry``
before it builds the engine (contrib/README.md, "How do I register a custom
model with vLLM?"). vLLM ships its own ``StableLmForCausalLM``, so this
replaces the built-in entry. The engine core runs in this process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are forked from it,
so the registration is in effect everywhere the model is built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/stablelm/test/unit/test_tiny_e2e.py -v
"""

import contextlib
import os

import pytest
import torch

from vllm_neuron import envs

from _local_src import src_package
from stablelm_test_utils import ARCHITECTURE, TINY_OVERRIDES, make_tiny_checkpoint

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Set VLLM_NEURON_CPU_MODE=1."
)

# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the V1 engine core subprocess hangs in
#   CPU mode. In-process mode prevents this. TP workers still start safely.
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
VOCAB_SIZE = TINY_OVERRIDES["vocab_size"]
PROMPT = list(range(3, 40))
MAX_TOKENS = 4
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01


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
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_stablelm_e2e")))


@contextlib.contextmanager
def _llm(model_dir: str):
    """Yield an LLM. Shut down its engine core at exit, also after a failure."""
    from vllm import LLM, ModelRegistry

    # vLLM has its own StableLmForCausalLM, so the contrib class must replace
    # it in the registry before the engine resolves the architecture. Pass the
    # class object: vLLM inspects a "module:Class" string in a fresh
    # interpreter, which cannot import the package that _local_src loaded.
    ModelRegistry.register_model(ARCHITECTURE, src_package().StableLmForCausalLM)
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=256,
        max_num_batched_tokens=128,
        block_size=32,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=64,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        # The async sampler output has no logprobs. Use sync scheduling.
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
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def _hf_teacher_forced_logits(model_dir: str, tokens: list[int], dtype):
    from transformers import StableLmForCausalLM as HFStableLmForCausalLM

    model = HFStableLmForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


def test_tp2_logits_match_hf(tiny_checkpoint):
    """TP 2: the vLLM raw logits of every vocabulary id match the HF reference."""
    from vllm import SamplingParams

    with _llm(tiny_checkpoint) as llm:
        outputs = llm.generate(
            [{"prompt_token_ids": PROMPT}],
            SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                logprobs=VOCAB_SIZE,
            ),
        )
    assert len(outputs) == 1
    completion = outputs[0].outputs[0]
    tokens = list(completion.token_ids)
    assert len(tokens) == MAX_TOKENS
    assert completion.logprobs is not None and len(completion.logprobs) == len(
        tokens
    ), "vLLM did not return logprobs for each generated token"

    ref_fp32 = _hf_teacher_forced_logits(tiny_checkpoint, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_checkpoint, tokens, torch.bfloat16)

    base_floor = BASE_FLOOR_FRACTION * ref_fp32.abs().max().item()
    for step, (token, step_logprobs) in enumerate(zip(tokens, completion.logprobs)):
        assert sorted(step_logprobs) == list(range(VOCAB_SIZE)), (
            f"Step {step}: vLLM returned {len(step_logprobs)} logits, not all "
            f"{VOCAB_SIZE}. Check max_logprobs=-1."
        )
        got = torch.tensor([step_logprobs[i].logprob for i in range(VOCAB_SIZE)])
        fp32 = ref_fp32[step]
        bf16 = ref_bf16[step]
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
        top2 = torch.topk(fp32, 2)
        margin = (top2.values[0] - top2.values[1]).item()
        if margin > 2 * LINF_RATIO * base_err:
            assert token == top2.indices[0].item(), (
                f"Step {step}: vLLM selected token {token}, but the FP32 "
                f"argmax is {top2.indices[0].item()} (margin {margin:.4f})."
            )
