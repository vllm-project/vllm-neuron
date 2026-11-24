# SPDX-License-Identifier: Apache-2.0
"""MiniCPM3 component and full-model tests (CPU mode, MPExecutor).

Layer-0 attention and MLP run at TP 4, and the full model runs at the TP size
of the integration test (8 ranks, 5 attention heads on each rank). The tests
use random-weight checkpoints with the real MiniCPM3-4B geometry, so nothing
is downloaded. The independent reference is the MiniCPM3 implementation built
into ``transformers``; each check is three-way (FP32 HF baseline, BF16 HF
reference, this folder's implementation).

Coverage:
  - Attention prefill of 128 tokens, then 8 decode steps that read the paged
    KV cache that the prefill and the earlier steps wrote (TP 4).
  - Attention chunked prefill: 1,536 tokens in three 512-token chunks through
    the segmented-prefill branch (``NF.segmented_attention``, its torch
    fallback on CPU). Each chunk attends to the KV cache that the earlier
    chunks wrote (TP 4). The integration test and the serving configuration
    prefill long prompts this way, in 2,048-token chunks.
  - MLP prefill (sequence-parallel reduce-scatter) and decode (all-reduce)
    (TP 4).
  - Full model with 2 layers at TP 8: ``load_weights`` (the vLLM path, which
    reads the ``pytorch_model.bin`` through ``src/checkpoints.py``), then
    prefill and decode logits.
  - With ``MINICPM3_CHECKPOINT_PATH`` set, the same full-model check on the
    real 62-layer checkpoint (skipped otherwise; with an FP32 HF reference of
    the full model it needs several tens of GB of host memory).

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/minicpm3/test/unit/test_components.py -v
"""

import math
from functools import partial

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor

import minicpm3_test_utils as U

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE,
    reason="CPU-mode test. Set VLLM_NEURON_CPU_MODE=1.",
)

DEVICE = "cpu"
# The full model runs at the integration TP size (8 ranks, 5 heads on each).
WORLD_SIZE = U.WORLD_SIZE
# The layer-0 attention and MLP checks run at TP 4. On CPU each rank's partial
# output is rounded to BF16 and
# the gloo all-reduce / reduce-scatter adds the partials in BF16, so this
# rounding grows with the rank count; with 8 ranks it alone takes a single
# layer's error past the three-way bound (the BF16 HF reference does one FP32
# accumulation). The full-model check at TP 8 compares gathered logits and
# passes the same bound.
COMPONENT_WORLD_SIZE = 4
SEED = 42
# Attention geometry: a prefill of 128 tokens, then 8 decode steps at positions
# 128 to 135.
ATTN_DECODE_STEPS = 8
ATTN_SEQ_LEN = 128 + ATTN_DECODE_STEPS
BLOCK_SIZE = 32
ATTN_NUM_BLOCKS = 8
# Chunked prefill: three chunks of 512 tokens, the smallest KV segment size of
# the segmented-prefill kernel. 48 blocks of 32 tokens hold the 1,536 tokens.
CHUNK_TOKENS = 512
CHUNKED_SEQ_LEN = 3 * CHUNK_TOKENS
CHUNKED_NUM_BLOCKS = CHUNKED_SEQ_LEN // BLOCK_SIZE
MLP_TOKENS = 128
MLP_DECODE_TOKENS = 4
# Full model: a prefill of 32 tokens, then a decode at position 32.
FULL_INPUT_LEN = 33
FULL_NUM_BLOCKS = 4
# MiniCPM3-4B geometry with 2 layers and a vocabulary that TP 8 divides.
TWO_LAYERS = {"num_hidden_layers": 2, "vocab_size": 4096, "initializer_range": 0.02}


@pytest.fixture(scope="module")
def two_layer_checkpoint(tmp_path_factory):
    """A two-layer random-weight checkpoint with the MiniCPM3-4B geometry."""
    path = str(tmp_path_factory.mktemp("minicpm3_two_layers"))
    return U.write_random_checkpoint(path, U.hf_config(TWO_LAYERS), seed=SEED)


def test_attention_prefill_decode(two_layer_checkpoint):
    """Attention prefill of 128 tokens, then 8 decode steps that read the cache."""
    world_size = COMPONENT_WORLD_SIZE
    checkpoint = two_layer_checkpoint
    goldens = U.hf_attention_reference(checkpoint, ATTN_SEQ_LEN, SEED)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            U.attention_model_load, checkpoint, ATTN_NUM_BLOCKS, BLOCK_SIZE, DEVICE
        ),
        device=DEVICE,
    )
    prefill, decode = U.run_attention_prefill_decode(
        executor,
        checkpoint,
        goldens["input"],
        ATTN_NUM_BLOCKS,
        BLOCK_SIZE,
        decode_steps=ATTN_DECODE_STEPS,
    )
    n = ATTN_SEQ_LEN - ATTN_DECODE_STEPS
    assert_close_three_way(
        baseline=goldens["fp32"][:n],
        expected=goldens["bf16"][:n],
        actual=torch.cat(prefill, dim=0),
        name=f"minicpm3/attn_prefill/seq{n}/tp{world_size}",
    )
    # One three-way check over the 8 decode steps of each rank, so that the
    # error distributions have 8 rows of samples, as the prefill check has 128.
    steps = range(n, ATTN_SEQ_LEN)
    for rank, outputs in enumerate(decode):
        assert_close_three_way(
            baseline=[goldens["fp32"][i : i + 1] for i in steps],
            expected=[goldens["bf16"][i : i + 1] for i in steps],
            actual=outputs,
            name=f"minicpm3/attn_decode/pos{n}-{ATTN_SEQ_LEN - 1}/tp{world_size}/rank{rank}",
        )


def test_attention_chunked_prefill(two_layer_checkpoint):
    """Attention prefill of 1,536 tokens in three 512-token segmented chunks."""
    world_size = COMPONENT_WORLD_SIZE
    checkpoint = two_layer_checkpoint
    goldens = U.hf_attention_reference(checkpoint, CHUNKED_SEQ_LEN, SEED)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            U.attention_model_load, checkpoint, CHUNKED_NUM_BLOCKS, BLOCK_SIZE, DEVICE
        ),
        device=DEVICE,
    )
    chunks = U.run_attention_chunked_prefill(
        executor,
        checkpoint,
        goldens["input"],
        CHUNK_TOKENS,
        CHUNKED_NUM_BLOCKS,
        BLOCK_SIZE,
    )
    # One three-way check per chunk: the rows of chunk c attend to the
    # c * 512 rows that the earlier chunks put in the KV cache.
    for c, per_rank in enumerate(chunks):
        rows = slice(c * CHUNK_TOKENS, (c + 1) * CHUNK_TOKENS)
        assert_close_three_way(
            baseline=goldens["fp32"][rows],
            expected=goldens["bf16"][rows],
            actual=torch.cat(per_rank, dim=0),
            name=f"minicpm3/attn_chunked_prefill/chunk{c}/tp{world_size}",
        )


def test_mlp_prefill_decode(two_layer_checkpoint):
    """MLP prefill (SP reduce-scatter) and decode (all-reduce)."""
    world_size = COMPONENT_WORLD_SIZE
    checkpoint = two_layer_checkpoint
    goldens = U.hf_mlp_reference(checkpoint, MLP_TOKENS, SEED)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(U.mlp_model_load, checkpoint, DEVICE),
        device=DEVICE,
    )
    prefill, decode = U.run_mlp_prefill_decode(
        executor, goldens["input"], MLP_DECODE_TOKENS
    )
    assert_close_three_way(
        baseline=goldens["fp32"],
        expected=goldens["bf16"],
        actual=torch.cat(prefill, dim=0),
        name=f"minicpm3/mlp_prefill/t{MLP_TOKENS}/tp{world_size}",
    )
    for rank, out in enumerate(decode):
        assert_close_three_way(
            baseline=goldens["fp32"][:MLP_DECODE_TOKENS],
            expected=goldens["bf16"][:MLP_DECODE_TOKENS],
            actual=out,
            name=f"minicpm3/mlp_decode/t{MLP_DECODE_TOKENS}/tp{world_size}/rank{rank}",
        )


def _check_full_model_logits(checkpoint: str, input_ids: list[int]) -> None:
    """Load the full model on WORLD_SIZE ranks, then compare prefill/decode logits.

    The worker load stops with an error if a parameter has no mapping, stays on
    the meta device, is not BF16, or holds a value that is not finite.
    """
    goldens = U.hf_full_model_logits(checkpoint, input_ids)
    hf = U.load_hf_config(checkpoint)
    executor = MPExecutor(
        world_size=WORLD_SIZE,
        model_load=partial(
            U.full_model_load, checkpoint, FULL_NUM_BLOCKS, BLOCK_SIZE, DEVICE
        ),
        device=DEVICE,
    )
    ids = torch.tensor(input_ids, dtype=torch.int64)
    n = len(input_ids) - 1
    executor.dispatch(
        ids[:n],
        torch.arange(n, dtype=torch.int64),
        U.full_model_metadata(hf.num_hidden_layers, 0, n, FULL_NUM_BLOCKS, BLOCK_SIZE),
        torch.tensor([n - 1], dtype=torch.int64),
    )
    prefill = executor.collect(timeout=U.COLLECT_TIMEOUT_S)
    executor.dispatch(
        ids[n:],
        torch.tensor([n], dtype=torch.int64),
        U.full_model_metadata(hf.num_hidden_layers, n, 1, FULL_NUM_BLOCKS, BLOCK_SIZE),
        torch.tensor([0], dtype=torch.int64),
    )
    decode = executor.collect(timeout=U.COLLECT_TIMEOUT_S)

    for rank in range(WORLD_SIZE):
        for step, outputs, row in (("prefill", prefill, 0), ("decode", decode, 1)):
            logits = outputs[rank].float()
            assert logits.shape == (1, hf.vocab_size)
            assert torch.isfinite(logits).all()
            assert_close_three_way(
                baseline=goldens["fp32"][row : row + 1],
                expected=goldens["bf16"][row : row + 1],
                actual=logits,
                name=f"minicpm3/full_{step}_logits/tp{WORLD_SIZE}/rank{rank}",
            )


def test_full_model_load_and_logits(two_layer_checkpoint):
    """Full 2-layer model: the vLLM load path, then prefill and decode logits."""
    torch.manual_seed(SEED)
    hf = U.load_hf_config(two_layer_checkpoint)
    # The prefill length (32) divides by the TP size, as sequence parallelism
    # needs.
    assert (FULL_INPUT_LEN - 1) % WORLD_SIZE == 0
    input_ids = torch.randint(3, hf.vocab_size, (FULL_INPUT_LEN,)).tolist()
    _check_full_model_logits(two_layer_checkpoint, input_ids)


def test_full_model_load_and_logits_real_checkpoint():
    """The real 62-layer checkpoint: complete load, then reference logits.

    Skipped unless ``MINICPM3_CHECKPOINT_PATH`` is set.
    """
    checkpoint = U.local_checkpoint_or_skip()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text = (
        "The human genome contains approximately three billion base pairs of "
        "DNA, encoding the genetic instructions for building and maintaining a "
        "human organism. The Amazon rainforest is the largest tropical "
        "rainforest in the world."
    )
    input_ids = tokenizer(text)["input_ids"]
    length = math.floor((len(input_ids) - 1) / WORLD_SIZE) * WORLD_SIZE + 1
    assert length > WORLD_SIZE
    _check_full_model_logits(checkpoint, input_ids[:length])
