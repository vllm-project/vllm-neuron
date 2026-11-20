# SPDX-License-Identifier: Apache-2.0
"""OLMo2 layer-0 component tests at TP 8.

The tests compare the port modules with independent HF references:

- ``test_attention_prefill``: 1024-token prefill attention. This test runs in
  CPU mode (one node) and on the device (two nodes).
- ``test_attention_decode``: a 1024-token prefill fills the KV cache. Then 4
  decode steps run at the positions 1024..1027. Device only.
- ``test_mlp_prefill``: the MLP on 1024 prefill tokens. Device only.
- ``test_mlp_decode``: the MLP on a decode batch of 4 tokens. Device only.

Each test uses ``assert_close_three_way``. The FP32 HF output is the baseline.
The BF16 HF output gives the expected dtype error.

On two nodes, each node checks the outputs of its local ranks. A prefill
output is in the SP layout, so global rank g holds the rows
g * 128 .. (g + 1) * 128. A decode output is all-reduced, so each rank holds
all rows.

The tests need 8 NeuronCores. Run the whole file on a trn2.48xlarge node:
    pytest contrib/models/olmo2/test/module/test_components.py -v

Only ``test_attention_prefill`` has a CPU path:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/olmo2/test/module/test_components.py::test_attention_prefill -v

``OLMO2_NNODES`` selects a multi-node topology (see ``olmo2_test_utils.py``).
The default is one node, which holds all 8 ranks.
"""

from functools import partial

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way

from olmo2_test_utils import (
    COLLECT_TIMEOUT_S,
    DECODE_STEPS,
    PREFILL_LEN,
    WORLD_SIZE,
    attention_model_load,
    component_device,
    decode_metadata,
    get_layer_goldens,
    local_global_ranks,
    make_executor,
    mlp_model_load,
    prefill_metadata,
    resolve_checkpoint,
)

TOKENS_PER_RANK = PREFILL_LEN // WORLD_SIZE
MAX_LINF_RATIO = 5.0


def _require_device() -> None:
    """Stop a device-only test in CPU mode with a clear error."""
    if envs.VLLM_NEURON_CPU_MODE:
        raise RuntimeError(
            "This test needs a Neuron device. The CPU decode attention fallback "
            "does not support a per-head block table. Unset VLLM_NEURON_CPU_MODE."
        )


def _local_prefill_rows(reference: torch.Tensor) -> torch.Tensor:
    """Return the SP rows of the local ranks, in collect order."""
    rows = [
        reference[g * TOKENS_PER_RANK : (g + 1) * TOKENS_PER_RANK]
        for g in local_global_ranks()
    ]
    return torch.cat(rows, dim=0)


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).contiguous()


def _fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32).contiguous()


def _prefill_attention_inputs(goldens: dict) -> tuple:
    # The model gives FP32 cos/sin to the attention, as HF OLMo2 does.
    return (
        _bf16(goldens["attn_input"][:PREFILL_LEN]),
        torch.arange(PREFILL_LEN, dtype=torch.long),
        _fp32(goldens["cos"][:PREFILL_LEN]),
        _fp32(goldens["sin"][:PREFILL_LEN]),
        prefill_metadata(PREFILL_LEN),
    )


def _check_prefill(outputs: list, goldens: dict, key: str, name: str) -> None:
    ranks = local_global_ranks()
    assert len(outputs) == len(ranks)
    for rank, out in zip(ranks, outputs):
        assert tuple(out.shape) == (
            TOKENS_PER_RANK,
            goldens[f"{key}_fp32"].shape[-1],
        ), f"rank {rank}: output shape {tuple(out.shape)}"
    actual = torch.cat([out.detach().cpu() for out in outputs], dim=0).float()
    assert torch.isfinite(actual).all(), f"{name}: the output has a non-finite value"
    assert_close_three_way(
        _local_prefill_rows(goldens[f"{key}_fp32"]).float(),
        _local_prefill_rows(goldens[f"{key}_bf16"]).float(),
        actual,
        name=f"{name}/nodes_ranks{ranks[0]}-{ranks[-1]}",
        max_linf_ratio=MAX_LINF_RATIO,
    )


def _check_all_reduced(outputs: list, name: str) -> torch.Tensor:
    """Check that each local rank has the same all-reduced output."""
    first = outputs[0].detach().cpu()
    for rank, out in zip(local_global_ranks(), outputs):
        assert torch.allclose(out.detach().cpu(), first, atol=1e-3, rtol=1e-3), (
            f"{name}: rank {rank} output is different from the first local rank"
        )
    assert torch.isfinite(first).all(), f"{name}: the output has a non-finite value"
    return first.float()


# =============================================================================
# Attention
# =============================================================================


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
def test_attention_prefill(world_size: int):
    """Compare a 1024-token prefill attention with the HF reference."""
    torch.manual_seed(42)
    checkpoint_path = resolve_checkpoint()
    goldens = get_layer_goldens(checkpoint_path)

    executor = make_executor(
        partial(
            attention_model_load,
            checkpoint_path=checkpoint_path,
            device=component_device(),
        )
    )
    executor.dispatch(*_prefill_attention_inputs(goldens))
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    _check_prefill(outputs, goldens, "attn", "olmo2_attention_prefill/tp8/layer0")


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
def test_attention_decode(world_size: int):
    """Fill the cache with a 1024-token prefill. Then compare 4 decode steps."""
    _require_device()
    torch.manual_seed(42)
    checkpoint_path = resolve_checkpoint()
    goldens = get_layer_goldens(checkpoint_path)

    executor = make_executor(
        partial(
            attention_model_load, checkpoint_path=checkpoint_path, device="neuron:0"
        )
    )
    executor.dispatch(*_prefill_attention_inputs(goldens))
    executor.collect(timeout=COLLECT_TIMEOUT_S)

    actual = []
    for step in range(DECODE_STEPS):
        position = PREFILL_LEN + step
        executor.dispatch(
            _bf16(goldens["attn_input"][position : position + 1]),
            torch.tensor([position], dtype=torch.long),
            _fp32(goldens["cos"][position : position + 1]),
            _fp32(goldens["sin"][position : position + 1]),
            decode_metadata(position),
        )
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        out = _check_all_reduced(outputs, f"attention decode step {step}")
        assert tuple(out.shape) == (1, goldens["attn_fp32"].shape[-1])
        actual.append(out)

    decode_rows = slice(PREFILL_LEN, PREFILL_LEN + DECODE_STEPS)
    assert_close_three_way(
        goldens["attn_fp32"][decode_rows].float(),
        goldens["attn_bf16"][decode_rows].float(),
        torch.cat(actual, dim=0),
        name="olmo2_attention_decode/tp8/layer0/prefill1024_steps4",
        max_linf_ratio=MAX_LINF_RATIO,
    )


# =============================================================================
# MLP
# =============================================================================


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
def test_mlp_prefill(world_size: int):
    """Compare the MLP on 1024 prefill tokens with the HF reference."""
    _require_device()
    torch.manual_seed(42)
    checkpoint_path = resolve_checkpoint()
    goldens = get_layer_goldens(checkpoint_path)

    executor = make_executor(
        partial(mlp_model_load, checkpoint_path=checkpoint_path, device="neuron:0")
    )
    executor.dispatch(_bf16(goldens["mlp_input"][:PREFILL_LEN]), True)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    _check_prefill(outputs, goldens, "mlp", "olmo2_mlp_prefill/tp8/layer0")


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
def test_mlp_decode(world_size: int):
    """Compare the MLP on a decode batch of 4 tokens with the HF reference."""
    _require_device()
    torch.manual_seed(42)
    checkpoint_path = resolve_checkpoint()
    goldens = get_layer_goldens(checkpoint_path)
    decode_rows = slice(PREFILL_LEN, PREFILL_LEN + DECODE_STEPS)

    executor = make_executor(
        partial(mlp_model_load, checkpoint_path=checkpoint_path, device="neuron:0")
    )
    executor.dispatch(_bf16(goldens["mlp_input"][decode_rows]), False)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

    actual = _check_all_reduced(outputs, "mlp decode")
    assert tuple(actual.shape) == tuple(goldens["mlp_fp32"][decode_rows].shape)
    assert_close_three_way(
        goldens["mlp_fp32"][decode_rows].float(),
        goldens["mlp_bf16"][decode_rows].float(),
        actual,
        name="olmo2_mlp_decode/tp8/layer0/bs4",
        max_linf_ratio=MAX_LINF_RATIO,
    )
