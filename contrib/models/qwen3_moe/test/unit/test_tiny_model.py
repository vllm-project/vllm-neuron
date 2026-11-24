# SPDX-License-Identifier: Apache-2.0
"""Tiny full-model test of Qwen3-MoE on CPU: prefill and decode logits against HF.

The test builds the whole ``Qwen3MoeForCausalLM`` of ``src/`` through the
factory on 2 ``MPExecutor`` ranks, loads the tiny random-weight checkpoint
with ``load_weights``, binds a paged KV cache and runs the model's own
``forward`` the way the vLLM Neuron runner calls it: a prefill, then 3 decode
steps of one token each. It compares the logits of the last prompt token and
of each decode step with the HF ``Qwen3MoeForCausalLM`` of transformers in
FP32 and BF16 (``assert_close_three_way``), teacher-forced on the same tokens.

Two prefill paths are covered, both on a 160-token prompt:

- ``flash``: one 256-token prefill step (96 padding tokens) through
  ``NF.flash_attention``.
- ``segmented``: two 128-token prefill chunks with a 512-token KV segment,
  through ``NF.segmented_attention``. The second chunk (32 real tokens and 96
  padding tokens) attends to the K/V that the first chunk wrote to the paged
  cache. This is the long-prompt path of the integration test and of the
  serve example (2048-token chunks).

The cases run with EP 2 (the shipped layout), and the segmented case also
without EP. In CPU mode (without ``NKI_SIMULATOR=1``) every NKI op runs its torch
fallback. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_moe/test/unit/test_tiny_model.py -v
"""

from functools import partial

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_package
from qwen3_moe_test_utils import (
    COLLECT_TIMEOUT_S,
    TINY_OVERRIDES,
    TINY_TP,
    load_hf_config,
    load_hf_reference,
    make_tiny_checkpoint,
    set_mock_vllm_config,
)

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Set VLLM_NEURON_CPU_MODE=1."
)

BLOCK_SIZE = 32
# 8 blocks of 32 tokens per sequence: a 256-token decode context, a multiple
# of 2 * 128 tokens as the decode mask needs. Block 0 is the null block, so
# the cache has one more block.
BLOCKS_PER_SEQ = 8
NUM_BLOCKS = BLOCKS_PER_SEQ + 1
BLOCK_TABLE = torch.arange(1, BLOCKS_PER_SEQ + 1, dtype=torch.int32)[None]
PAD_SLOT_ID = -1

PROMPT_LEN = 160
DECODE_STEPS = 3
FLASH_BUCKET = 256
SEGMENT_CHUNK = 128
KV_SEGMENT_SIZE = 512
SEED = 7


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_qwen3_moe_model")))


def _tokens() -> tuple[list[int], list[int]]:
    """The prompt and the teacher-forced decode inputs."""
    generator = torch.Generator().manual_seed(SEED)
    vocab = TINY_OVERRIDES["vocab_size"] - 2  # leave out the bos/eos ids
    ids = torch.randint(0, vocab, (PROMPT_LEN + DECODE_STEPS,), generator=generator)
    ids = ids.tolist()
    return ids[:PROMPT_LEN], ids[PROMPT_LEN:]


def _slot(position: int) -> int:
    block = int(BLOCK_TABLE[0, position // BLOCK_SIZE])
    return block * BLOCK_SIZE + position % BLOCK_SIZE


class _Runner(torch.nn.Module):
    """Hold the model and its KV cache. Run one step per call."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        max_query_len: int,
        cached_seq_len: int,
        kv_segment_size: int,
        sampling_positions: torch.Tensor,
    ) -> torch.Tensor:
        metadata = {
            "block_table_tensor": BLOCK_TABLE,
            "slot_mapping": slot_mapping,
            "max_query_len": max_query_len,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": BLOCKS_PER_SEQ,
            "decode_token_threshold": 1,
            "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
            "kv_segment_size": kv_segment_size,
        }
        attn_metadata = {
            f"layers.{i}.self_attn": metadata
            for i in range(len(self.model.model.layers))
        }
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                positions=positions,
                attn_metadata=attn_metadata,
                sampling_positions=sampling_positions,
            )
        return logits.float()


def _model_load(checkpoint: str, ep: bool) -> torch.nn.Module:
    """Top-level picklable model load: the full model for this rank."""
    set_mock_vllm_config(ep_enabled=ep)
    hf_config = load_hf_config(checkpoint)
    # No on-device sampling: the forward returns the logits.
    neuron_config = NeuronConfig(on_device_sampling_config=None)
    with torch.device("meta"):
        model = src_package().Qwen3MoeForCausalLM.from_configs(hf_config, neuron_config)
    model.load_weights(checkpoint, torch.device("cpu"), None)
    caches = {}
    for layer in model.get_kv_spec().layers:
        shape = (NUM_BLOCKS, layer.num_kv_heads, BLOCK_SIZE, layer.head_size)
        caches[layer.name] = [
            torch.zeros(shape, dtype=layer.dtype),
            torch.zeros(shape, dtype=layer.dtype),
        ]
    model.bind_kv_cache(caches)
    return _Runner(model.eval())


def _prefill_steps(prompt: list[int], path: str) -> list[tuple]:
    """Return the prefill step inputs and the row of the last prompt token."""
    if path == "flash":
        chunks = [(0, PROMPT_LEN, FLASH_BUCKET, 0)]
    else:
        chunks = [
            (
                start,
                min(start + SEGMENT_CHUNK, PROMPT_LEN),
                SEGMENT_CHUNK,
                KV_SEGMENT_SIZE,
            )
            for start in range(0, PROMPT_LEN, SEGMENT_CHUNK)
        ]
    steps = []
    for start, end, bucket, segment in chunks:
        num_real = end - start
        pad = bucket - num_real
        ids = torch.tensor(prompt[start:end] + [0] * pad, dtype=torch.int32)
        # The runner pads a prefill at the last real position.
        positions = torch.tensor(
            list(range(start, end)) + [end - 1] * pad, dtype=torch.int64
        )
        slots = torch.tensor(
            [_slot(p) for p in range(start, end)] + [PAD_SLOT_ID] * pad,
            dtype=torch.int64,
        )
        sampling = torch.tensor([num_real - 1], dtype=torch.int64)
        steps.append((ids, positions, slots, bucket, start, segment, sampling))
    return steps


def _hf_logits(checkpoint: str, prompt: list[int], decode: list[int]):
    """HF logits of the last prompt token and of each decode input, FP32 and BF16."""
    ids = torch.tensor([prompt + decode], dtype=torch.long)
    rows = slice(PROMPT_LEN - 1, PROMPT_LEN - 1 + 1 + DECODE_STEPS)
    out = []
    for dtype in (torch.float32, torch.bfloat16):
        model = load_hf_reference(checkpoint, dtype)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, rows].float()
        out.append([row for row in logits])
    return out


@pytest.mark.parametrize(
    "path, ep",
    [
        pytest.param("flash", True, id="flash-ep2"),
        pytest.param("segmented", True, id="segmented-ep2"),
        pytest.param("segmented", False, id="segmented-tp2"),
    ],
)
def test_tiny_model_logits_vs_hf(tiny_checkpoint, path: str, ep: bool):
    """Three-way: the prefill and decode logits of the full model against HF."""
    from vllm_neuron.utils.executor import MPExecutor

    prompt, decode = _tokens()
    executor = MPExecutor(
        world_size=TINY_TP,
        model_load=partial(_model_load, checkpoint=tiny_checkpoint, ep=ep),
        device="cpu",
        ep_degree=TINY_TP if ep else 1,
    )

    def run(*args) -> torch.Tensor:
        executor.dispatch(*args)
        outputs = [torch.as_tensor(o) for o in executor.collect(COLLECT_TIMEOUT_S)]
        # The LM head gathers the vocabulary, so every rank has the same logits.
        for rank, out in enumerate(outputs[1:], start=1):
            torch.testing.assert_close(
                out, outputs[0], rtol=0, atol=0, msg=f"rank {rank}"
            )
        return outputs[0]

    actual = []
    for step in _prefill_steps(prompt, path):
        logits = run(*step)
    actual.append(logits[0])

    for i, token in enumerate(decode):
        position = PROMPT_LEN + i
        logits = run(
            torch.tensor([token], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int64),
            torch.tensor([_slot(position)], dtype=torch.int64),
            1,
            0,
            0,
            torch.tensor([0], dtype=torch.int64),
        )
        actual.append(logits[0])

    vocab = load_hf_config(tiny_checkpoint).vocab_size
    assert all(row.shape == (vocab,) for row in actual)
    baseline, expected = _hf_logits(tiny_checkpoint, prompt, decode)
    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"qwen3_moe/tiny_model_cpu/{path}/{'ep' if ep else 'tp'}{TINY_TP}",
    )
