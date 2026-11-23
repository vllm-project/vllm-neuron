# SPDX-License-Identifier: Apache-2.0
"""ModernBERT device tests: TP 2 across 2 Trn2 nodes (1 rank on each node).

Model: ibm-granite/granite-embedding-small-english-r2 (BF16, 12 layers).

Each node runs this file. MPExecutor starts 1 worker on each node, and the
2 workers form one TP group. The attention and MLP all-reduces go across the
nodes, so each worker takes part in each forward. A failure on a worker
raises in ``collect`` on its node.

Coverage:
  - Attention of a full layer (0) and a sliding layer (1), 1024 tokens.
  - GeGLU MLP of layer 0, 1024 tokens.
  - Full model, 1 sequence of 1024 tokens: hidden states and CLS embedding.
  - Full model, 2 packed sequences (700 + 300) and a padding tail.

The references are the HF ``ModernBertModel`` in FP32 and BF16.

Run with (on each node):
    python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \
        -m pytest -v contrib/models/modernbert/test/module/test_device.py
"""

import importlib.util
import os
import sys
from functools import partial
from pathlib import Path

import pytest
import torch

from vllm_neuron.accuracy.testing import (
    assert_close_three_way,
    assert_pooling_close_three_way,
)
from vllm_neuron.utils.executor import MPExecutor

# Load this folder's own code by file location, not through a package path
# rooted at the repository: a contrib model has to keep working when the folder
# is copied into an application. ``test/_local_src.py`` is the loader, and it is
# loaded the same way, under a name no other contrib model can collide with.
_LOADER_NAME = "contrib_modernbert_local_src"
if _LOADER_NAME in sys.modules:
    _local_src = sys.modules[_LOADER_NAME]
else:
    _loader_spec = importlib.util.spec_from_file_location(
        _LOADER_NAME, Path(__file__).resolve().parents[1] / "_local_src.py"
    )
    _local_src = importlib.util.module_from_spec(_loader_spec)
    sys.modules[_LOADER_NAME] = _local_src
    _loader_spec.loader.exec_module(_local_src)

common = _local_src.load_test_module("module/common.py")

TP_SIZE = common.TP_SIZE
cls_embedding = common.cls_embedding
hf_component_references = common.hf_component_references
hf_hidden_states = common.hf_hidden_states
load_hf_models = common.load_hf_models
make_token_ids = common.make_token_ids
model_load = common.model_load
pack = common.pack
resolve_checkpoint = common.resolve_checkpoint

DEVICE = "neuron:0"
NNODES = 2
SEQ_LEN = 1024
MIN_COS = 0.99
COLLECT_TIMEOUT_S = 3600.0


def _node_rank() -> int:
    return int(os.environ.get("NODE_RANK", "0"))


@pytest.fixture(scope="module")
def checkpoint_path():
    return resolve_checkpoint()


def _run(checkpoint_path, world_size, component, layer_idx, *inputs):
    """Run one component on the TP group and return the outputs of the local ranks."""
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(model_load, checkpoint_path, component, layer_idx, DEVICE),
        device=DEVICE,
        nnodes=NNODES,
        node_rank=_node_rank(),
    )
    executor.dispatch(*inputs)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
    assert len(outputs) == world_size // NNODES
    return outputs


def _global_rank(local_rank: int, world_size: int) -> int:
    return _node_rank() * (world_size // NNODES) + local_rank


@pytest.mark.parametrize(
    "layer_idx", [0, 1], ids=["full_attention", "sliding_attention"]
)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_attention_device(checkpoint_path, world_size, layer_idx):
    """TP 2 attention on Neuron matches the HF attention (three-way)."""
    (input_ids,) = make_token_ids(checkpoint_path, [SEQ_LEN])
    hidden, baseline, expected = hf_component_references(
        checkpoint_path, f"layers.{layer_idx}.attn", input_ids
    )
    positions = torch.arange(SEQ_LEN, dtype=torch.int64)

    outputs = _run(
        checkpoint_path, world_size, "attention", layer_idx, hidden, positions
    )
    for local_rank, actual in enumerate(outputs):
        rank = _global_rank(local_rank, world_size)
        result = assert_close_three_way(
            baseline,
            expected,
            actual.float(),
            name=f"modernbert/device/attn{layer_idx}/rank{rank}",
        )
        assert result.passed


@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_mlp_device(checkpoint_path, world_size):
    """TP 2 GeGLU MLP on Neuron matches the HF MLP (three-way)."""
    (input_ids,) = make_token_ids(checkpoint_path, [SEQ_LEN])
    hidden, baseline, expected = hf_component_references(
        checkpoint_path, "layers.0.mlp", input_ids
    )

    outputs = _run(checkpoint_path, world_size, "mlp", 0, hidden)
    for local_rank, actual in enumerate(outputs):
        rank = _global_rank(local_rank, world_size)
        result = assert_close_three_way(
            baseline,
            expected,
            actual.float(),
            name=f"modernbert/device/mlp0/rank{rank}",
        )
        assert result.passed


@pytest.mark.parametrize(
    "lengths", [[SEQ_LEN], [700, 300]], ids=["single_1024", "packed_700_300"]
)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_full_model_device(checkpoint_path, world_size, lengths):
    """The TP 2 model on Neuron matches the HF hidden states and the CLS embeddings."""
    sequences = make_token_ids(checkpoint_path, lengths)
    input_ids, positions = pack(sequences)

    hf_fp32, hf_bf16 = load_hf_models(checkpoint_path)
    baseline = [hf_hidden_states(hf_fp32, s).float() for s in sequences]
    expected = [hf_hidden_states(hf_bf16, s).float() for s in sequences]

    outputs = _run(checkpoint_path, world_size, "full", 0, input_ids, positions)
    for local_rank, out in enumerate(outputs):
        rank = _global_rank(local_rank, world_size)
        assert out.shape == (sum(lengths), baseline[0].shape[-1])
        actual = list(out.float().split(lengths))
        tag = "x".join(str(n) for n in lengths)

        result = assert_close_three_way(
            torch.cat(baseline),
            torch.cat(expected),
            torch.cat(actual),
            name=f"modernbert/device/hidden_{tag}/rank{rank}",
        )
        pooled = assert_pooling_close_three_way(
            baseline=torch.stack([cls_embedding(h) for h in baseline]),
            expected=torch.stack([cls_embedding(h) for h in expected]),
            actual=torch.stack([cls_embedding(h) for h in actual]),
            task="embed",
            name=f"modernbert/device/embed_{tag}/rank{rank}",
            min_cos=MIN_COS,
        )
        print(
            f"rank {rank} {tag}: worst cos {pooled.min_tgt_cos:.6f} "
            f"(floor {MIN_COS}), mean cos {pooled.mean_tgt_cos:.6f}, "
            f"BF16-HF worst cos {pooled.min_base_cos:.6f}"
        )
        assert result.passed
        assert pooled.passed
