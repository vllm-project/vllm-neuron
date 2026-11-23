# SPDX-License-Identifier: Apache-2.0
"""Tiny E2E inference test for MiniCPM3 through the full vLLM pipeline (CPU).

Run the vLLM ``LLM(...)`` pipeline (model build on meta, weight load, forward,
sampling, output) in CPU mode. The test uses no Neuron hardware and no real
weights. It builds a 2-layer random-weight MiniCPM3 checkpoint and saves it as
``pytorch_model.bin``, the format of the published MiniCPM3-4B checkpoint, so
vLLM's ``load_weights`` call goes through the ``.bin`` fallback of
``src/checkpoints.py``.

These MiniCPM3-specific paths run here:
  - MLA mapped onto the fused multi-head kernels (latent projection, fused
    per-head QKV, padded head dim 128), in prefill and in decode.
  - LongRoPE from the ``rope_scaling`` config.
  - muP scaling of the embeddings, the residual branches and the logits.
  - Tied embeddings: the lm_head loads ``model.embed_tokens.weight``.
  - At TP 2, 5 attention heads on each rank, the per-rank head count of the
    TP 8 serving configuration of MiniCPM3-4B.

The test registers this folder's factory with vLLM's ``ModelRegistry`` before
it builds the engine (option 1 of ``contrib/README.md``). vLLM ships its own
``MiniCPM3ForCausalLM``, so the call replaces that entry. The engine core runs
in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are
forked from it, so the registration is in effect everywhere the model is built.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/minicpm3/test/unit/test_tiny_e2e.py -v
"""

import os
import shutil

import pytest
import torch
from vllm import LLM, ModelRegistry, SamplingParams

from vllm_neuron import envs

from _local_src import src_package
import minicpm3_test_utils as U

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE,
    reason="CPU-mode test. Set VLLM_NEURON_CPU_MODE=1.",
)

# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the V1 engine core subprocess hangs in
#   CPU mode. In-process mode avoids this. TP workers still start safely.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM, so the computed
#   budget is always near zero.
# - NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker
#   runs the NKI CPU simulator against a concrete platform target. Without
#   them the worker cannot resolve a device.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NKI_SIMULATOR": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

# vLLM's TP workers talk through shared-memory queues in /dev/shm: a 160 MiB
# broadcast queue, plus a 240 MiB response queue for each worker. A container
# with Docker's default 64 MiB /dev/shm cannot start a TP 2 engine.
_TP2_SHM_BYTES = (160 + 2 * 240) * 2**20

TINY_VOCAB = 512
PROMPT = list(range(3, 13))
OUTPUT_TOKENS = 8
_FACTORS = [1.0 + 0.5 * i for i in range(16)]

# The MiniCPM3-4B geometry per head (qk_nope 64, qk_rope 32, v 128), with 10
# heads (5 on each TP 2 rank), 2 layers and a small hidden size. The weights
# use the checkpoint's initializer_range (0.1), so the attention and MLP
# branches are not negligible against the scaled embedding in the residual
# stream, and the first generated token depends on them.
TINY_CONFIG = {
    "vocab_size": TINY_VOCAB,
    "hidden_size": 1280,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "q_lora_rank": 256,
    "kv_lora_rank": 128,
    "max_position_embeddings": 512,
    "rope_scaling": {
        "type": "longrope",
        "long_factor": _FACTORS,
        "short_factor": _FACTORS,
        "original_max_position_embeddings": 512,
    },
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set the env vars for the tiny E2E tests. Restore them at teardown."""
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def tiny_minicpm3_dir(tmp_path_factory):
    """Create the random-weight tiny MiniCPM3 checkpoint once per module."""
    path = str(tmp_path_factory.mktemp("tiny_minicpm3"))
    U.write_random_checkpoint(path, U.hf_config(TINY_CONFIG), seed=42)
    assert not [n for n in os.listdir(path) if n.endswith(".safetensors")]
    return path


def _run_inference(model_dir: str, tp_size: int) -> list[int]:
    ModelRegistry.register_model(U.ARCHITECTURE, getattr(src_package(), U.ARCHITECTURE))
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
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [128, 256]}},
    )
    try:
        outputs = llm.generate(
            [{"prompt_token_ids": PROMPT}],
            SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True),
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert len(outputs) == 1
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert len(token_ids) == OUTPUT_TOKENS
    assert all(0 <= t < TINY_VOCAB for t in token_ids), token_ids
    return token_ids


@pytest.fixture(scope="module")
def hf_reference_tokens(tiny_minicpm3_dir):
    """Greedy-decode OUTPUT_TOKENS tokens of PROMPT with the HF reference in FP32."""
    model = U.native_hf_model_loader(tiny_minicpm3_dir, torch.float32)
    with torch.no_grad():
        logits = model(torch.tensor([PROMPT])).logits[0, -1]
        out = model.generate(
            torch.tensor([PROMPT]),
            max_new_tokens=OUTPUT_TOKENS,
            min_new_tokens=OUTPUT_TOKENS,
            do_sample=False,
        )
    top2 = logits.topk(2).values
    print(
        f"HF FP32 first-token top-2 logit margin: {(top2[0] - top2[1]).item():.4f} "
        f"(logit std {logits.std().item():.4f})"
    )
    return out[0, len(PROMPT) :].tolist()


def _check_first_token(tokens: list[int], reference: list[int]) -> None:
    """The first generated token must match the FP32 HF reference.

    Later tokens of a random-weight model can flip on a near tie, so only the
    first token is a gate.
    """
    print(f"vLLM tokens {tokens}, HF FP32 tokens {reference}")
    assert tokens[0] == reference[0]


# test_tp2 runs before test_tp1. The TP 1 engine core initializes vLLM's
# distributed state in this process, and the workers of a TP 2 engine forked
# after it would inherit a world size of 1 and refuse to start.
def test_tp2(tiny_minicpm3_dir, hf_reference_tokens):
    """TP 2: 5 heads per rank, sharded vocab, multi-worker dispatch."""
    free = shutil.disk_usage("/dev/shm").free if os.path.isdir("/dev/shm") else None
    if free is not None and free < _TP2_SHM_BYTES:
        pytest.skip(
            f"/dev/shm has {free // 2**20} MiB free; a TP 2 vLLM engine needs about "
            f"{_TP2_SHM_BYTES // 2**20} MiB (run the container with --shm-size=1g "
            f"or --ipc=host)."
        )
    tokens = _run_inference(tiny_minicpm3_dir, tp_size=2)
    _check_first_token(tokens, hf_reference_tokens)


def test_tp1(tiny_minicpm3_dir, hf_reference_tokens):
    """TP 1: single-worker full pipeline."""
    tokens = _run_inference(tiny_minicpm3_dir, tp_size=1)
    _check_first_token(tokens, hf_reference_tokens)
