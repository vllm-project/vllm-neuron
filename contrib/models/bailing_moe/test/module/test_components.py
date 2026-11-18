# SPDX-License-Identifier: Apache-2.0
"""Component tests for BailingMoeV2 (inclusionAI/Ling-mini-2.0).

Each test loads one real layer of the pinned checkpoint on 8 ranks (TP 8,
EP 8) and compares the output with the independent torch reference in
``reference_model.py``. The comparison is three-way: FP32 reference, BF16
reference and the Neuron module.

  - Attention of layer 1: one prefill of 256 tokens, then 4 decode steps
    that read the KV cache that the prefill wrote.
  - MoE block of layer 1 (256 experts, top 8, shared expert): prefill with
    256 tokens and decode with 4 tokens.
  - Dense MLP of layer 0 (TP shard 640): prefill with 256 tokens, then
    decode with 4 tokens.

The serving-geometry cases use the geometry of the logit test. They also
assert that each output is finite:

  - Attention of layer 1: 13 real tokens in a 256-token prefill bucket, then
    4 decode steps with a block table of 8 entries (7 null entries).
  - MoE block of layer 1: 13 real tokens in a 256-token prefill bucket.
  - MoE block of layer 1: 27 real tokens and 229 identical pad rows from the
    embedding of pad token 0, for each pad position mode of the runner.
  - MoE block of layer 1 and dense MLP of layer 0: decode with 1 token.

CPU mode runs the 8 ranks on one host and uses the torch paths of
``vllm_neuron.functional``. A device run spans 2 trn2.48xlarge nodes with 4
ranks on each node, and uses the NKI kernels. Start the device run through
the repository's ``test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py``.

On a device, each rank compiles through ``utils.RouteRecordingBackend``. The
backend records the NKI call sites of each compiled graph. Each test asserts
that the prefill graph and the decode graph of each local rank contain the
required NKI kernels. A torch fallback of a required operation fails the test.

Run (CPU):
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
        pytest contrib/models/bailing_moe/test/module/test_components.py -v
"""

import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

utils = _local_src.load_test_module("utils")
_reference = _local_src.load_test_module("reference_model")
_tensor_index = _reference._tensor_index
load_reference_layer = _reference.load_reference_layer
read_reference_config = _reference.read_reference_config
reference_attention = _reference.reference_attention
reference_mlp = _reference.reference_mlp
reference_moe = _reference.reference_moe
reference_rms_norm = _reference.reference_rms_norm

RANDOM_SEED = 42
PREFILL_TOKENS = 256
DECODE_STEPS = 4
MOE_DECODE_TOKENS = 4

# Serving geometry of the logit test. A short prompt fills the 256-token
# prefill bucket with pad tokens. The KV cache of the sequence starts at
# block 1, because block 0 is the vLLM null block. Pad tokens write to
# slot 0. The decode block table has MAX_MODEL_LEN / BLOCK_SIZE entries, and
# the runner changes each null entry to -1.
PADDED_REAL_TOKENS = 13
# An earlier port revision gave NaN prefill logits for the first logit prompt. That
# prompt has 27 tokens. The runner fills the bucket with pad token 0, so all
# 229 pad rows are the same row.
LOGIT_PROMPT_REAL_TOKENS = 27
PAD_TOKEN_ID = 0
SERVING_BLOCKS_PER_SEQ = PREFILL_TOKENS // utils.BLOCK_SIZE
SERVING_FIRST_BLOCK = 1
NULL_SLOT = 0
NULL_BLOCK_SENTINEL = -1


@pytest.fixture(autouse=True)
def set_constant_seed():
    torch.manual_seed(RANDOM_SEED)


@pytest.fixture(scope="module")
def checkpoint():
    return utils.resolve_checkpoint()


def _device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def _route_dir(tmp_path) -> str | None:
    """Return the directory of the dispatch records, or None in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE:
        return None
    route_dir = tmp_path / "nki_routes"
    route_dir.mkdir()
    return str(route_dir)


def _check_routes(route_dir, global_ranks, start, phase):
    """Assert the NKI sites of ``phase`` on each local rank. Skip in CPU mode."""
    if route_dir is None:
        return start
    records = utils.read_route_records(route_dir, global_ranks)
    return utils.assert_nki_sites(records, start, phase)


def _normed_input(num_tokens: int, hidden: int, gamma: torch.Tensor, eps: float):
    """Return a BF16 input with the scale that the layer sees after its norm."""
    x = torch.randn(num_tokens, hidden, dtype=torch.float32).to(torch.bfloat16)
    return reference_rms_norm(x, gamma.to(torch.bfloat16), eps)


def _local_rows(full: torch.Tensor, global_ranks: list[int], world_size: int):
    """Return the SP rows of ``global_ranks`` from a full ``[T, H]`` tensor."""
    shard = full.shape[0] // world_size
    return torch.cat([full[r * shard : (r + 1) * shard] for r in global_ranks], dim=0)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that ``tensor`` has no NaN or inf value."""
    bad = (~torch.isfinite(tensor.float())).sum().item()
    assert bad == 0, (
        f"{name} has {bad} NaN or inf values in {tensor.numel()} values. "
        "Read the output of this component at the serving geometry."
    )


def _padded_positions(num_real: int, num_tokens: int) -> torch.Tensor:
    """Return the prefill positions of the runner: pads repeat the last position."""
    positions = torch.full((num_tokens,), num_real - 1, dtype=torch.int32)
    positions[:num_real] = torch.arange(num_real, dtype=torch.int32)
    return positions


def _zero_pad_positions(num_real: int, num_tokens: int) -> torch.Tensor:
    """Return the prefill positions of the other runner path: pads are at 0."""
    positions = torch.zeros((num_tokens,), dtype=torch.int32)
    positions[:num_real] = torch.arange(num_real, dtype=torch.int32)
    return positions


PAD_POSITION_MODES = {
    "repeat_last": _padded_positions,
    "zero": _zero_pad_positions,
}


def _embedding_row(checkpoint: str, token_id: int) -> torch.Tensor:
    """Read one row of the input embedding from the checkpoint as ``[1, H]`` BF16."""
    from safetensors import safe_open

    key = "model.word_embeddings.weight"
    path = _tensor_index(checkpoint)[key]
    with safe_open(path, framework="pt") as f:
        row = f.get_slice(key)[token_id : token_id + 1]
    return row.to(torch.bfloat16)


def _serving_slot(position: int) -> int:
    return SERVING_FIRST_BLOCK * utils.BLOCK_SIZE + position


def _serving_block_table() -> torch.Tensor:
    """Return the decode block table of the runner for one short sequence."""
    table = torch.full(
        (1, SERVING_BLOCKS_PER_SEQ), NULL_BLOCK_SENTINEL, dtype=torch.int32
    )
    table[0, 0] = SERVING_FIRST_BLOCK
    return table


def _padded_prefill_metadata(layer_idx: int, num_real: int, num_tokens: int) -> dict:
    """Return the prefill metadata of a short sequence in a padded bucket."""
    slots = torch.full((num_tokens,), NULL_SLOT, dtype=torch.int64)
    slots[:num_real] = torch.tensor([_serving_slot(i) for i in range(num_real)])
    return {
        f"layers.{layer_idx}.self_attn": {
            "max_query_len": num_tokens,
            "decode_token_threshold": 1,
            "slot_mapping": slots,
            "block_size": utils.BLOCK_SIZE,
            "block_table_tensor": _serving_block_table(),
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def _serving_decode_metadata(layer_idx: int, position: int) -> dict:
    """Return the decode metadata of one token of a short sequence."""
    return {
        f"layers.{layer_idx}.self_attn": {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": torch.tensor([_serving_slot(position)], dtype=torch.int64),
            "block_size": utils.BLOCK_SIZE,
            "block_table_tensor": _serving_block_table(),
            "max_blocks_per_seq": SERVING_BLOCKS_PER_SEQ,
        }
    }


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_attention_prefill_then_decode(checkpoint, world_size, request, tmp_path):
    """Prefill 256 tokens, then decode 4 tokens on the same KV cache."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cpu_mode = envs.VLLM_NEURON_CPU_MODE
    device = _device()
    route_dir = _route_dir(tmp_path)
    layer_idx = utils.MOE_LAYER

    cfg = read_reference_config(checkpoint)
    ref32 = load_reference_layer(
        checkpoint, cfg, layer_idx, torch.float32, include_mlp=False
    )
    ref16 = load_reference_layer(
        checkpoint, cfg, layer_idx, torch.bfloat16, include_mlp=False
    )

    num_inputs = PREFILL_TOKENS + DECODE_STEPS
    hidden = _normed_input(
        num_inputs, cfg.hidden_size, ref32["input_layernorm"], cfg.rms_norm_eps
    )
    positions = torch.arange(num_inputs)

    # Reference: one prefill, then one decode step at a time with the KV cache.
    baseline, expected = [], []
    with torch.no_grad():
        out32, cache32 = reference_attention(
            cfg,
            ref32,
            hidden[None, :PREFILL_TOKENS].float(),
            positions[:PREFILL_TOKENS],
            None,
        )
        out16, cache16 = reference_attention(
            cfg, ref16, hidden[None, :PREFILL_TOKENS], positions[:PREFILL_TOKENS], None
        )
        baseline.append(_local_rows(out32[0], global_ranks, world_size))
        expected.append(_local_rows(out16[0], global_ranks, world_size))
        for step in range(DECODE_STEPS):
            idx = PREFILL_TOKENS + step
            pos = positions[idx : idx + 1]
            out32, cache32 = reference_attention(
                cfg, ref32, hidden[None, idx : idx + 1].float(), pos, cache32
            )
            out16, cache16 = reference_attention(
                cfg, ref16, hidden[None, idx : idx + 1], pos, cache16
            )
            baseline.append(out32[0])
            expected.append(out16[0])

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.attention_model_load,
            checkpoint=checkpoint,
            layer_idx=layer_idx,
            cpu_mode=cpu_mode,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )

    actual = []
    executor.dispatch(
        hidden[:PREFILL_TOKENS],
        torch.arange(PREFILL_TOKENS, dtype=torch.int32),
        utils.prefill_metadata(layer_idx, PREFILL_TOKENS),
        True,
    )
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    actual.append(torch.cat([o.to(torch.bfloat16) for o in outputs], dim=0))
    counts = _check_routes(route_dir, global_ranks, {}, "attention prefill")

    for step in range(DECODE_STEPS):
        idx = PREFILL_TOKENS + step
        executor.dispatch(
            hidden[idx : idx + 1],
            torch.tensor([idx], dtype=torch.int32),
            utils.decode_metadata(layer_idx, idx),
            False,
        )
        outputs = executor.collect(timeout=3600)
        # The all-reduce gives the same decode output on each rank.
        for rank_out in outputs[1:]:
            assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"
        actual.append(outputs[0].to(torch.bfloat16))
    _check_routes(route_dir, global_ranks, counts, "attention decode")

    assert_close_three_way(
        baseline,
        expected,
        actual,
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


def _moe_references(checkpoint: str, layer_idx: int):
    cfg = read_reference_config(checkpoint)
    ref32 = load_reference_layer(checkpoint, cfg, layer_idx, torch.float32)
    ref16 = load_reference_layer(checkpoint, cfg, layer_idx, torch.bfloat16)
    return cfg, ref32, ref16


def _moe_executor(
    checkpoint: str, world_size: int, nnodes: int, node_rank: int, route_dir
):
    device = _device()
    return MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.moe_model_load,
            checkpoint=checkpoint,
            layer_idx=utils.MOE_LAYER,
            cpu_mode=envs.VLLM_NEURON_CPU_MODE,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_moe_prefill(checkpoint, world_size, request, tmp_path):
    """Blockwise prefill MoE with 256 tokens across EP 8."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg, ref32, ref16 = _moe_references(checkpoint, utils.MOE_LAYER)

    hidden = _normed_input(
        PREFILL_TOKENS,
        cfg.hidden_size,
        ref32["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    with torch.no_grad():
        baseline = _local_rows(
            reference_moe(cfg, ref32, hidden.float()), global_ranks, world_size
        )
        expected = _local_rows(
            reference_moe(cfg, ref16, hidden), global_ranks, world_size
        )

    route_dir = _route_dir(tmp_path)
    executor = _moe_executor(checkpoint, world_size, nnodes, node_rank, route_dir)
    executor.dispatch(hidden, torch.arange(PREFILL_TOKENS, dtype=torch.int32), True)
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "moe prefill")
    actual = torch.cat([o.to(torch.bfloat16) for o in outputs], dim=0)

    assert_close_three_way(
        [baseline],
        [expected],
        [actual],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_moe_decode(checkpoint, world_size, request, tmp_path):
    """Decode MoE with 4 replicated tokens across EP 8."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg, ref32, ref16 = _moe_references(checkpoint, utils.MOE_LAYER)

    hidden = _normed_input(
        MOE_DECODE_TOKENS,
        cfg.hidden_size,
        ref32["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    with torch.no_grad():
        baseline = reference_moe(cfg, ref32, hidden.float())
        expected = reference_moe(cfg, ref16, hidden)

    route_dir = _route_dir(tmp_path)
    executor = _moe_executor(checkpoint, world_size, nnodes, node_rank, route_dir)
    executor.dispatch(
        hidden, torch.full((MOE_DECODE_TOKENS,), 300, dtype=torch.int32), False
    )
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "moe decode")
    for rank_out in outputs[1:]:
        assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"

    assert_close_three_way(
        [baseline],
        [expected],
        [outputs[0].to(torch.bfloat16)],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_dense_mlp_prefill_then_decode(checkpoint, world_size, request, tmp_path):
    """Dense MLP of layer 0 (TP shard 640): prefill 256 tokens, then decode 4."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg = read_reference_config(checkpoint)
    ref32 = load_reference_layer(checkpoint, cfg, 0, torch.float32)
    ref16 = load_reference_layer(checkpoint, cfg, 0, torch.bfloat16)
    assert not ref32["is_moe"], "layer 0 must be the dense MLP layer"

    hidden = _normed_input(
        PREFILL_TOKENS + MOE_DECODE_TOKENS,
        cfg.hidden_size,
        ref32["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    prefill, decode = hidden[:PREFILL_TOKENS], hidden[PREFILL_TOKENS:]

    def reference(layer, x):
        return reference_mlp(
            x, layer["gate_proj"], layer["up_proj"], layer["down_proj"]
        )

    with torch.no_grad():
        baseline = [
            _local_rows(reference(ref32, prefill.float()), global_ranks, world_size),
            reference(ref32, decode.float()),
        ]
        expected = [
            _local_rows(reference(ref16, prefill), global_ranks, world_size),
            reference(ref16, decode),
        ]

    device = _device()
    route_dir = _route_dir(tmp_path)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.dense_mlp_model_load,
            checkpoint=checkpoint,
            cpu_mode=envs.VLLM_NEURON_CPU_MODE,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )

    executor.dispatch(prefill, torch.arange(PREFILL_TOKENS, dtype=torch.int32), True)
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    actual = [torch.cat([o.to(torch.bfloat16) for o in outputs], dim=0)]
    counts = _check_routes(route_dir, global_ranks, {}, "dense mlp prefill")

    executor.dispatch(
        decode, torch.full((MOE_DECODE_TOKENS,), 300, dtype=torch.int32), False
    )
    outputs = executor.collect(timeout=3600)
    for rank_out in outputs[1:]:
        assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"
    actual.append(outputs[0].to(torch.bfloat16))
    _check_routes(route_dir, global_ranks, counts, "dense mlp decode")

    assert_close_three_way(
        baseline,
        expected,
        actual,
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


# =============================================================================
# Serving-geometry regression cases
#
# An earlier port revision gave NaN logits on the first request of the
# full-model logit test. The cases above did not cover the
# geometry of that request: a short prompt in a padded 256-token prefill
# bucket, a decode with one token, and a decode block table with null
# entries. The cases below use that geometry, and assert finite outputs and
# the three-way comparison. They do not replace the cases above.
# =============================================================================


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_attention_padded_prefill_then_decode(
    checkpoint, world_size, request, tmp_path
):
    """Prefill 13 tokens in a 256-token bucket, then decode 4 tokens.

    The metadata is the same as the metadata of the runner. Pad tokens repeat
    the last position and write to the null slot. The decode block table has
    8 entries, and 7 of them are -1.
    """
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cpu_mode = envs.VLLM_NEURON_CPU_MODE
    device = _device()
    route_dir = _route_dir(tmp_path)
    layer_idx = utils.MOE_LAYER
    num_real = PADDED_REAL_TOKENS

    cfg = read_reference_config(checkpoint)
    ref32 = load_reference_layer(
        checkpoint, cfg, layer_idx, torch.float32, include_mlp=False
    )
    ref16 = load_reference_layer(
        checkpoint, cfg, layer_idx, torch.bfloat16, include_mlp=False
    )

    hidden = _normed_input(
        PREFILL_TOKENS + DECODE_STEPS,
        cfg.hidden_size,
        ref32["input_layernorm"],
        cfg.rms_norm_eps,
    )
    prefill, decode = hidden[:PREFILL_TOKENS], hidden[PREFILL_TOKENS:]
    real_positions = torch.arange(num_real)

    # Reference: a prefill of the real tokens only, then one decode step at a
    # time with the KV cache.
    baseline, expected = [], []
    # The real rows are in the SP shard of global rank 0.
    has_real_rows = 0 in global_ranks
    with torch.no_grad():
        out32, cache32 = reference_attention(
            cfg, ref32, prefill[None, :num_real].float(), real_positions, None
        )
        out16, cache16 = reference_attention(
            cfg, ref16, prefill[None, :num_real], real_positions, None
        )
        if has_real_rows:
            baseline.append(out32[0])
            expected.append(out16[0])
        for step in range(DECODE_STEPS):
            pos = torch.tensor([num_real + step])
            out32, cache32 = reference_attention(
                cfg, ref32, decode[None, step : step + 1].float(), pos, cache32
            )
            out16, cache16 = reference_attention(
                cfg, ref16, decode[None, step : step + 1], pos, cache16
            )
            baseline.append(out32[0])
            expected.append(out16[0])

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.attention_model_load,
            checkpoint=checkpoint,
            layer_idx=layer_idx,
            cpu_mode=cpu_mode,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )

    actual = []
    executor.dispatch(
        prefill,
        _padded_positions(num_real, PREFILL_TOKENS),
        _padded_prefill_metadata(layer_idx, num_real, PREFILL_TOKENS),
        True,
    )
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    for rank, rank_out in zip(global_ranks, outputs):
        _assert_finite(f"padded attention prefill rank {rank}", rank_out)
    if has_real_rows:
        rank0_out = outputs[global_ranks.index(0)]
        actual.append(rank0_out[:num_real].to(torch.bfloat16))
    counts = _check_routes(route_dir, global_ranks, {}, "attention prefill")

    for step in range(DECODE_STEPS):
        position = num_real + step
        executor.dispatch(
            decode[step : step + 1],
            torch.tensor([position], dtype=torch.int32),
            _serving_decode_metadata(layer_idx, position),
            False,
        )
        outputs = executor.collect(timeout=3600)
        _assert_finite(f"attention decode step {step}", outputs[0])
        for rank_out in outputs[1:]:
            assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"
        actual.append(outputs[0].to(torch.bfloat16))
    _check_routes(route_dir, global_ranks, counts, "attention decode")

    assert_close_three_way(
        baseline,
        expected,
        actual,
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_moe_padded_prefill(checkpoint, world_size, request, tmp_path):
    """Blockwise prefill MoE with 13 real tokens in a 256-token bucket.

    Pad tokens repeat the last position, so the padding mask removes them
    from the routed experts. All output rows must be finite. The real rows
    must agree with the reference of the real tokens.
    """
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg, ref32, ref16 = _moe_references(checkpoint, utils.MOE_LAYER)
    num_real = PADDED_REAL_TOKENS

    hidden = _normed_input(
        PREFILL_TOKENS,
        cfg.hidden_size,
        ref32["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    with torch.no_grad():
        baseline = reference_moe(cfg, ref32, hidden[:num_real].float())
        expected = reference_moe(cfg, ref16, hidden[:num_real])

    route_dir = _route_dir(tmp_path)
    executor = _moe_executor(checkpoint, world_size, nnodes, node_rank, route_dir)
    executor.dispatch(hidden, _padded_positions(num_real, PREFILL_TOKENS), True)
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "moe prefill")
    for rank, rank_out in zip(global_ranks, outputs):
        _assert_finite(f"padded moe prefill rank {rank}", rank_out)

    # The real rows are in the SP shard of global rank 0. The other ranks
    # hold pad rows only, and the finite check above covers them.
    if 0 not in global_ranks:
        return
    actual = outputs[global_ranks.index(0)][:num_real].to(torch.bfloat16)
    assert_close_three_way(
        [baseline],
        [expected],
        [actual],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("pad_positions", sorted(PAD_POSITION_MODES))
@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_moe_padded_prefill_identical_pad_rows(
    checkpoint, world_size, pad_positions, request, tmp_path
):
    """Blockwise prefill MoE with 27 real tokens and 229 identical pad rows.

    The runner fills a short prompt with pad token 0, so each pad row is the
    same row. The router then sends all pad rows to the same 8 experts, and
    these experts get full blocks of pad tokens. Random pad rows do not make
    this load. Each pad row here is the normed embedding row of token 0.

    Both pad position modes of the runner apply: the pads repeat the last
    position, or the pads are at position 0. All output rows must be finite.
    The real rows must agree with the reference of the real tokens.

    Earlier port revisions: the case with pads at position 0 stopped
    in an inter-node collective. The harness then traced a different graph on
    each rank. See test_component_harness_prefill_graph_is_rank_invariant.
    """
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg, ref32, ref16 = _moe_references(checkpoint, utils.MOE_LAYER)
    num_real = LOGIT_PROMPT_REAL_TOKENS
    gamma = ref32["post_attention_layernorm"].to(torch.bfloat16)

    real = _normed_input(
        num_real,
        cfg.hidden_size,
        ref32["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    pad_row = reference_rms_norm(
        _embedding_row(checkpoint, PAD_TOKEN_ID), gamma, cfg.rms_norm_eps
    )
    _assert_finite("normed pad row", pad_row)
    hidden = torch.cat(
        [real, pad_row.expand(PREFILL_TOKENS - num_real, -1)], dim=0
    ).contiguous()
    with torch.no_grad():
        baseline = reference_moe(cfg, ref32, real.float())
        expected = reference_moe(cfg, ref16, real)

    route_dir = _route_dir(tmp_path)
    executor = _moe_executor(checkpoint, world_size, nnodes, node_rank, route_dir)
    positions = PAD_POSITION_MODES[pad_positions](num_real, PREFILL_TOKENS)
    executor.dispatch(hidden, positions, True)
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "moe prefill")
    for rank, rank_out in zip(global_ranks, outputs):
        _assert_finite(
            f"identical-pad moe prefill ({pad_positions}) rank {rank}", rank_out
        )

    # The real rows are in the SP shard of global rank 0.
    if 0 not in global_ranks:
        return
    actual = outputs[global_ranks.index(0)][:num_real].to(torch.bfloat16)
    assert_close_three_way(
        [baseline],
        [expected],
        [actual],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_moe_decode_single_token(checkpoint, world_size, request, tmp_path):
    """Decode MoE with 1 token at position 13 across EP 8 (batch 1 serving)."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg, ref32, ref16 = _moe_references(checkpoint, utils.MOE_LAYER)

    hidden = _normed_input(
        1, cfg.hidden_size, ref32["post_attention_layernorm"], cfg.rms_norm_eps
    )
    with torch.no_grad():
        baseline = reference_moe(cfg, ref32, hidden.float())
        expected = reference_moe(cfg, ref16, hidden)

    route_dir = _route_dir(tmp_path)
    executor = _moe_executor(checkpoint, world_size, nnodes, node_rank, route_dir)
    executor.dispatch(
        hidden, torch.tensor([PADDED_REAL_TOKENS], dtype=torch.int32), False
    )
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "moe decode")
    _assert_finite("single-token moe decode", outputs[0])
    for rank_out in outputs[1:]:
        assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"

    assert_close_three_way(
        [baseline],
        [expected],
        [outputs[0].to(torch.bfloat16)],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_dense_mlp_single_token_decode(checkpoint, world_size, request, tmp_path):
    """Dense MLP of layer 0: decode 1 token at position 13 (batch 1 serving)."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    cfg = read_reference_config(checkpoint)
    ref32 = load_reference_layer(checkpoint, cfg, 0, torch.float32)
    ref16 = load_reference_layer(checkpoint, cfg, 0, torch.bfloat16)
    assert not ref32["is_moe"], "layer 0 must be the dense MLP layer"

    hidden = _normed_input(
        1, cfg.hidden_size, ref32["post_attention_layernorm"], cfg.rms_norm_eps
    )

    def reference(layer, x):
        return reference_mlp(
            x, layer["gate_proj"], layer["up_proj"], layer["down_proj"]
        )

    with torch.no_grad():
        baseline = reference(ref32, hidden.float())
        expected = reference(ref16, hidden)

    device = _device()
    route_dir = _route_dir(tmp_path)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.dense_mlp_model_load,
            checkpoint=checkpoint,
            cpu_mode=envs.VLLM_NEURON_CPU_MODE,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=3600,
    )

    executor.dispatch(
        hidden, torch.tensor([PADDED_REAL_TOKENS], dtype=torch.int32), False
    )
    outputs = executor.collect(timeout=3600)
    assert len(outputs) == len(global_ranks)
    _check_routes(route_dir, global_ranks, {}, "dense mlp decode")
    _assert_finite("single-token dense mlp decode", outputs[0])
    for rank_out in outputs[1:]:
        assert torch.equal(rank_out, outputs[0]), "decode ranks disagree"

    assert_close_three_way(
        [baseline],
        [expected],
        [outputs[0].to(torch.bfloat16)],
        name=f"{request.node.name}/node{node_rank}",
        plot_on_failure=True,
        output_dir="accuracy_debug",
    )
