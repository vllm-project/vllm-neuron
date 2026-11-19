# SPDX-License-Identifier: Apache-2.0
"""Jina Embeddings v5 text nano: device component and model tests.

Topology: TP=2. By default both ranks run on one node:

    pytest contrib/models/jina_embeddings_v5/test/module/ -v -s --timeout=3600

The recorded runs of the port placed 1 rank on each of 2 trn2.48xlarge nodes,
so that every TP collective crossed the nodes. To run that topology, set
``JINA_V5_NNODES=2`` and ``NODE_RANK`` on both nodes, together with the
rendezvous address of node 0 and ``NEURON_RT_ROOT_COMM_ID`` that the plugin's
multi-node executor reads (the multi-node launcher of the recorded runs is not
part of this folder). Each node checks the outputs of its own ranks against
the HF ``EuroBertModel`` reference.

Coverage, in the 1024-token serving bucket:

- Layer-0 attention prefill (fused QKV + RoPE, non-causal ``attention_cte``,
  o_proj), with bucket padding.
- Layer-0 MLP prefill.
- Full 12-layer encoder prefill: hidden states and pooled embeddings.
- Full 12-layer encoder on a decode-classified batch of 1-token sequences.
"""

import functools
import os

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import (
    assert_close_three_way,
    assert_pooling_close_three_way,
)
from vllm_neuron.utils.executor import MPExecutor

import jina_v5_test_utils as utils
from jina_v5_test_utils import (
    DEVICE_BUCKET,
    LAYER_IDX,
    build_attn_metadata,
    get_checkpoint,
    hf_last_hidden,
    hf_layer_attention,
    hf_layer_mlp,
    load_config_dict,
    load_hf_encoder,
    padded_positions,
    padded_token_ids,
    pool_last_normalize,
    random_token_ids,
)

NNODES = int(os.environ.get("JINA_V5_NNODES", "1"))
MIN_COS = 0.99
N_SAMPLES = 3
ATTN_REAL_LEN = 1000
MODEL_REAL_LENS = (1024, 700)
DECODE_NUM_SEQS = 4
COLLECT_TIMEOUT_S = 3600


@pytest.fixture(scope="module")
def node_rank():
    if envs.VLLM_NEURON_CPU_MODE:
        pytest.skip(
            "The device tests need Neuron devices. Unset VLLM_NEURON_CPU_MODE "
            "to run them."
        )
    if NNODES > 1:
        assert "NODE_RANK" in os.environ, (
            f"JINA_V5_NNODES={NNODES}, but NODE_RANK is not set. Set NODE_RANK "
            "on each node."
        )
    return int(os.environ.get("NODE_RANK", "0"))


@pytest.fixture(scope="module")
def checkpoint():
    return get_checkpoint()


@pytest.fixture(scope="module")
def config_dict(checkpoint):
    return load_config_dict(checkpoint)


@pytest.fixture(scope="module")
def hf_models(checkpoint):
    return {
        "fp32": load_hf_encoder(checkpoint, torch.float32),
        "bf16": load_hf_encoder(checkpoint, torch.bfloat16),
    }


def _executor(load_fn, checkpoint, config_dict, world_size, node_rank):
    return MPExecutor(
        world_size=world_size,
        model_load=functools.partial(load_fn, checkpoint, config_dict, "neuron:0"),
        device="neuron:0",
        nnodes=NNODES,
        node_rank=node_rank,
    )


def _local_rows(world_size: int, node_rank: int) -> slice:
    """Return the SP rows that the ranks of this node own."""
    ranks_per_node = world_size // NNODES
    first_rank = node_rank * ranks_per_node
    per_rank = DEVICE_BUCKET // world_size
    return slice(first_rank * per_rank, (first_rank + ranks_per_node) * per_rank)


def _collect_sp(executor) -> torch.Tensor:
    """Return the SP slices of the ranks of this node, in rank order."""
    return torch.cat(executor.collect(timeout=COLLECT_TIMEOUT_S), dim=0)


@pytest.mark.parametrize("world_size", [2])
def test_device_attention_prefill(
    world_size, node_rank, checkpoint, config_dict, hf_models
):
    torch.manual_seed(42)
    hidden = config_dict["hidden_size"]
    positions = padded_positions(ATTN_REAL_LEN, DEVICE_BUCKET)
    metadata = build_attn_metadata(
        config_dict["num_hidden_layers"], DEVICE_BUCKET, DEVICE_BUCKET
    )
    executor = _executor(
        utils.load_attention, checkpoint, config_dict, world_size, node_rank
    )

    rows = _local_rows(world_size, node_rank)
    real_stop = min(rows.stop, ATTN_REAL_LEN) - rows.start
    baseline, expected, actual = [], [], []
    for sample in range(N_SAMPLES):
        torch.manual_seed(sample)
        h = torch.randn(DEVICE_BUCKET, hidden, dtype=torch.bfloat16)
        executor.dispatch(h, positions, metadata)
        out = _collect_sp(executor)  # the SP slices of this node
        assert out.shape == (rows.stop - rows.start, hidden)
        real = h[:ATTN_REAL_LEN]
        ref32 = hf_layer_attention(hf_models["fp32"], real.float(), LAYER_IDX)
        ref16 = hf_layer_attention(hf_models["bf16"], real, LAYER_IDX)
        baseline.append(ref32[rows][:real_stop])
        expected.append(ref16[rows][:real_stop])
        actual.append(out[:real_stop])

    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"jina_v5/device_attention/tp{world_size}/node{node_rank}",
    )


@pytest.mark.parametrize("world_size", [2])
def test_device_mlp_prefill(world_size, node_rank, checkpoint, config_dict, hf_models):
    torch.manual_seed(42)
    hidden = config_dict["hidden_size"]
    executor = _executor(utils.load_mlp, checkpoint, config_dict, world_size, node_rank)

    rows = _local_rows(world_size, node_rank)
    baseline, expected, actual = [], [], []
    for sample in range(N_SAMPLES):
        torch.manual_seed(sample)
        h = torch.randn(DEVICE_BUCKET, hidden, dtype=torch.bfloat16)
        executor.dispatch(h)
        out = _collect_sp(executor)  # the SP slices of this node
        baseline.append(hf_layer_mlp(hf_models["fp32"], h[rows].float(), LAYER_IDX))
        expected.append(hf_layer_mlp(hf_models["bf16"], h[rows], LAYER_IDX))
        actual.append(out)

    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"jina_v5/device_mlp/tp{world_size}/node{node_rank}",
    )


@pytest.mark.parametrize("world_size", [2])
def test_device_full_model_prefill(
    world_size, node_rank, checkpoint, config_dict, hf_models
):
    torch.manual_seed(42)
    metadata = build_attn_metadata(
        config_dict["num_hidden_layers"], DEVICE_BUCKET, DEVICE_BUCKET
    )
    executor = _executor(
        utils.load_encoder, checkpoint, config_dict, world_size, node_rank
    )

    baseline, expected, actual = [], [], []
    pooled = {"fp32": [], "bf16": [], "port": []}
    for seed, real_len in enumerate(MODEL_REAL_LENS):
        token_ids = random_token_ids(seed, real_len)
        executor.dispatch(
            padded_token_ids(token_ids, DEVICE_BUCKET),
            padded_positions(real_len, DEVICE_BUCKET),
            metadata,
        )
        # The encoder all-gathers the sequence, so each rank has [T, H].
        out = executor.collect(timeout=COLLECT_TIMEOUT_S)[0]
        assert out.shape == (DEVICE_BUCKET, config_dict["hidden_size"])
        ref32 = hf_last_hidden(hf_models["fp32"], token_ids)
        ref16 = hf_last_hidden(hf_models["bf16"], token_ids)
        baseline.append(ref32)
        expected.append(ref16)
        actual.append(out[:real_len])
        pooled["fp32"].append(pool_last_normalize(ref32, real_len))
        pooled["bf16"].append(pool_last_normalize(ref16, real_len))
        pooled["port"].append(pool_last_normalize(out, real_len))

    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"jina_v5/device_full_model/tp{world_size}/node{node_rank}",
    )
    assert_pooling_close_three_way(
        baseline=torch.stack(pooled["fp32"]),
        expected=torch.stack(pooled["bf16"]),
        actual=torch.stack(pooled["port"]),
        task="embed",
        name=f"jina_v5/device_pooled/tp{world_size}/node{node_rank}",
        min_cos=MIN_COS,
    )


@pytest.mark.parametrize("world_size", [2])
def test_device_full_model_single_token_batch(
    world_size, node_rank, checkpoint, config_dict, hf_models
):
    """A batch of 1-token sequences takes the decode-classified path.

    The runner warmup compiles this path, and a 1-token request selects it.
    """
    torch.manual_seed(42)
    metadata = build_attn_metadata(
        config_dict["num_hidden_layers"],
        DECODE_NUM_SEQS,
        max_query_len=1,
        num_seqs=DECODE_NUM_SEQS,
    )
    executor = _executor(
        utils.load_encoder, checkpoint, config_dict, world_size, node_rank
    )

    token_ids = random_token_ids(7, DECODE_NUM_SEQS)
    executor.dispatch(
        torch.tensor(token_ids, dtype=torch.int32),
        torch.zeros(DECODE_NUM_SEQS, dtype=torch.int32),
        metadata,
    )
    out = executor.collect(timeout=COLLECT_TIMEOUT_S)[0]
    assert out.shape == (DECODE_NUM_SEQS, config_dict["hidden_size"])

    ref32 = torch.cat([hf_last_hidden(hf_models["fp32"], [t]) for t in token_ids])
    ref16 = torch.cat([hf_last_hidden(hf_models["bf16"], [t]) for t in token_ids])
    assert_close_three_way(
        baseline=ref32,
        expected=ref16,
        actual=out,
        name=f"jina_v5/device_single_token/tp{world_size}/node{node_rank}",
    )
    assert_pooling_close_three_way(
        baseline=torch.nn.functional.normalize(ref32.float(), dim=-1),
        expected=torch.nn.functional.normalize(ref16.float(), dim=-1),
        actual=torch.nn.functional.normalize(out.float(), dim=-1),
        task="embed",
        name=f"jina_v5/device_single_token_pooled/tp{world_size}/node{node_rank}",
        min_cos=MIN_COS,
    )
