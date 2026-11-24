# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE sparse MoE block on CPU, against the HF block.

The test runs layer 0's ``Qwen3MoeSparseMoeBlock`` of the tiny random-weight
checkpoint on 2 ``MPExecutor`` ranks, with EP 2 (4 complete experts on each
rank, the layout that ``--enable-expert-parallel`` gives) and without EP (a
128-column shard of all 8 experts on each rank). It compares the output with
the HF ``Qwen3MoeSparseMoeBlock`` of transformers in FP32 and BF16
(``assert_close_three_way``):

- Prefill: each rank gets its sequence-parallel shard of the normalized
  hidden states. The block runs ``NF.router``, all-gathers, builds the
  blockwise expert mapping, runs ``NF.moe_cte`` and reduce-scatters. The last
  16 tokens are bucket padding at the last real position, as the vLLM Neuron
  runner pads a prefill; only the real rows are compared.
- Decode (batch 4): each rank gets the residual stream and the RMSNorm
  weight. ``NF.moe_block_tkg`` applies the RMSNorm, the router and the
  experts, and the ranks all-reduce.

In CPU mode (without ``NKI_SIMULATOR=1``) every NKI op runs its torch
fallback. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_moe/test/unit/test_moe_block.py -v
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_module
from qwen3_moe_test_utils import (
    COLLECT_TIMEOUT_S,
    TINY_TP,
    load_hf_config,
    load_hf_reference,
    make_tiny_checkpoint,
    read_checkpoint_tensors,
    set_mock_vllm_config,
)

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Set VLLM_NEURON_CPU_MODE=1."
)

LAYER_IDX = 0
PREFILL_TOKENS = 256
PREFILL_PADDING = 16
DECODE_BATCH = 4
SEEDS = (0, 1)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_qwen3_moe_block")))


def _block_mappings(num_experts: int) -> dict:
    """The checkpoint mappings of layer ``LAYER_IDX``'s MoE block, by block name."""
    full = src_module("model").qwen3_moe_checkpoint_mappings(LAYER_IDX + 1, num_experts)
    prefix = f"model.layers.{LAYER_IDX}.mlp."
    return {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}


def _moe_block_load(checkpoint: str, ep: bool) -> torch.nn.Module:
    """Top-level picklable model load: layer 0's MoE block for this rank."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    set_mock_vllm_config(ep_enabled=ep)
    model_mod = src_module("model")
    config = src_module("config").Qwen3MoeConfig.from_configs(
        load_hf_config(checkpoint), NeuronConfig()
    )
    block = model_mod.Qwen3MoeSparseMoeBlock(config)
    state = (
        SafetensorsCheckpoint(checkpoint)
        .load_sharded(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            model=block,
            mappings=_block_mappings(config.num_experts),
            device=torch.device("cpu"),
        )
        .state_dict
    )
    block.load_state_dict(state, strict=True, assign=True)
    return block.eval()


def _sample(seed: int, num_tokens: int, hidden: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(num_tokens, hidden, generator=generator) * 2 - 1).to(
        torch.bfloat16
    )


def _hf_outputs(checkpoint: str, inputs: list[torch.Tensor], decode: bool):
    """Return the HF block outputs in FP32 and BF16 for each input."""
    out = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        layer = load_hf_reference(checkpoint, dtype).model.layers[LAYER_IDX]
        rows = []
        with torch.no_grad():
            for x in inputs:
                h = x.to(dtype).unsqueeze(0)
                if decode:
                    h = layer.post_attention_layernorm(h)
                rows.append(layer.mlp(h).squeeze(0).float())
        out[name] = rows
    return out["fp32"], out["bf16"]


@pytest.mark.parametrize("ep", [True, False], ids=["ep2", "tp2"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_moe_block_vs_hf(tiny_checkpoint, ep: bool, phase: str):
    """Three-way: the MoE block at TP 2 (with and without EP) against HF."""
    from vllm_neuron.utils.executor import MPExecutor

    hidden = load_hf_config(tiny_checkpoint).hidden_size
    executor = MPExecutor(
        world_size=TINY_TP,
        model_load=partial(_moe_block_load, checkpoint=tiny_checkpoint, ep=ep),
        device="cpu",
        ep_degree=TINY_TP if ep else 1,
    )

    inputs, actual = [], []
    if phase == "prefill":
        num_real = PREFILL_TOKENS - PREFILL_PADDING
        positions = torch.arange(PREFILL_TOKENS, dtype=torch.int32)
        positions[num_real:] = num_real - 1
        for seed in SEEDS:
            x = _sample(seed, PREFILL_TOKENS, hidden)
            executor.dispatch_per_rank(
                [(shard, positions, False) for shard in x.chunk(TINY_TP, dim=0)]
            )
            outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
            out = torch.cat([torch.as_tensor(o) for o in outputs], dim=0)
            assert out.shape == (PREFILL_TOKENS, hidden)
            inputs.append(x[:num_real])
            actual.append(out[:num_real].float())
    else:
        gamma = read_checkpoint_tensors(
            tiny_checkpoint,
            [f"model.layers.{LAYER_IDX}.post_attention_layernorm.weight"],
        ).popitem()[1]
        for seed in SEEDS:
            x = _sample(seed, DECODE_BATCH, hidden)
            executor.dispatch_per_rank([(x, None, True, gamma)] * TINY_TP)
            outputs = [torch.as_tensor(o) for o in executor.collect(COLLECT_TIMEOUT_S)]
            # The all-reduce gives every rank the same full output.
            for rank, out in enumerate(outputs[1:], start=1):
                torch.testing.assert_close(
                    out, outputs[0], rtol=0, atol=0, msg=f"rank {rank}"
                )
            assert outputs[0].shape == (DECODE_BATCH, hidden)
            inputs.append(x)
            actual.append(outputs[0].float())

    baseline, expected = _hf_outputs(tiny_checkpoint, inputs, phase == "decode")
    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"qwen3_moe/moe_block_cpu/{phase}/{'ep' if ep else 'tp'}{TINY_TP}",
    )
