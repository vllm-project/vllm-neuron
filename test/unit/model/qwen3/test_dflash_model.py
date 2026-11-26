# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the DFlash drafter's token embedding under tensor parallelism.

The drafter embeds its query block with a vocab-sharded embedding. Each TP rank
must trace the same graph, so the native backend compiles one draft NEFF that
all ranks share. Run with ``VLLM_NEURON_CPU_MODE=1``; no device is needed.
"""

import difflib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.model.qwen3.dflash_model as dflash_model
from vllm_neuron.model.qwen3.dflash_model import DFlashQwen3Model
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

VOCAB_SIZE = 32
EMBED_DIM = 4
# Stands in for the TP device group; no collective runs in these tests.
_FAKE_DEVICE_GROUP = object()


def _identity_all_reduce(tensor, reduceOp, group):
    # The tests check each rank's partial embedding, so skip the TP sum.
    return tensor


def _draft_model_for_rank(
    tp_rank: int, tp_size: int, full_weight: torch.Tensor
) -> DFlashQwen3Model:
    """Return a drafter whose embedding holds ``tp_rank``'s vocab shard."""
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=tp_size),
        patch.object(torch.distributed, "get_rank", return_value=tp_rank),
        patch("vllm_neuron.nn.embedding.is_native_backend", return_value=True),
    ):
        embed_tokens = VocabDimShardedEmbedding(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            dtype=torch.float32,
            tp_group=_FAKE_DEVICE_GROUP,
        )
    start = embed_tokens.vocab_start_idx
    with torch.no_grad():
        embed_tokens.weight.copy_(
            full_weight[start : start + embed_tokens.vocab_size_per_rank]
        )

    # Only the embedding is under test, so skip building the decoder layers.
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.embed_tokens = embed_tokens
    return model


def _trace_embed_input_ids(
    model: DFlashQwen3Model, tp_rank: int, input_ids: torch.Tensor
) -> tuple[list[str], torch.Tensor]:
    """Compile ``embed_input_ids`` as rank ``tp_rank``; return graphs and output."""
    # The stamped rank tensor that stamp_spmd_rank_tensors puts on each group.
    tp_group = SimpleNamespace(
        rank_in_group_tensor=torch.tensor([tp_rank], dtype=torch.int32),
        device_group=_FAKE_DEVICE_GROUP,
    )
    graphs: list[str] = []

    def record_graph(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward

    torch._dynamo.reset()
    with (
        patch.object(dflash_model, "get_tp_group", lambda: tp_group),
        patch("vllm_neuron.nn.embedding.all_reduce", _identity_all_reduce),
    ):
        compiled = torch.compile(
            model.embed_input_ids, backend=record_graph, fullgraph=True
        )
        output = compiled(input_ids)
    torch._dynamo.reset()
    return graphs, output


@pytest.mark.parametrize("tp_size", [2, 4])
def test_draft_embedding_traces_one_graph_for_all_tp_ranks(tp_size: int) -> None:
    torch.manual_seed(0)
    full_weight = torch.randn(VOCAB_SIZE, EMBED_DIM)
    # Token ids from every rank's vocab shard.
    input_ids = torch.tensor([0, 3, 7, 8, 12, 15, 16, 21, 24, 31])

    graphs_per_rank = []
    partial_sum = torch.zeros(input_ids.numel(), EMBED_DIM)
    for tp_rank in range(tp_size):
        model = _draft_model_for_rank(tp_rank, tp_size, full_weight)
        graphs, output = _trace_embed_input_ids(model, tp_rank, input_ids)
        assert len(graphs) == 1
        graphs_per_rank.append(graphs[0])

        # Values are unchanged: each rank embeds only the tokens it owns.
        shard = VOCAB_SIZE // tp_size
        owned = (input_ids >= tp_rank * shard) & (input_ids < (tp_rank + 1) * shard)
        expected = torch.where(
            owned[:, None],
            F.embedding(input_ids, full_weight),
            torch.zeros(()),
        )
        torch.testing.assert_close(output, expected)
        partial_sum += output

    # One graph for all ranks: the vocab range comes from the stamped rank
    # tensor, not from a per-rank Python int baked into the graph.
    for tp_rank, graph in enumerate(graphs_per_rank[1:], start=1):
        diff = "\n".join(
            difflib.unified_diff(
                graphs_per_rank[0].splitlines(),
                graph.splitlines(),
                "rank 0",
                f"rank {tp_rank}",
                lineterm="",
            )
        )
        assert not diff, f"DFlash draft embedding graph differs per TP rank:\n{diff}"
    torch.testing.assert_close(partial_sum, F.embedding(input_ids, full_weight))
