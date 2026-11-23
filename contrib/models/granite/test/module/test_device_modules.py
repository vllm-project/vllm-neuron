# SPDX-License-Identifier: Apache-2.0
"""Granite (ibm/PowerLM-3b) device module tests: TP4 across 2 Trn2 nodes.

Each node runs this file at the same time. The launcher sets NODE_RANK,
MASTER_ADDR and NEURON_RT_ROOT_COMM_ID. MPExecutor starts 2 local ranks on
each node, so the 4 TP ranks span both nodes and every collective crosses the
node boundary.

The modules use the real checkpoint weights of the pinned revision in
bfloat16. The reference is the HF module in FP32 and in BF16 on the same
input. Each node checks the output rows of its local ranks.

Checks:
1. Attention prefill (1024 tokens, SP) and one decode step from the paged
   KV cache that the prefill wrote (layer 0).
2. MLP prefill (1024 tokens, prefill kernel) and decode (4 tokens, decode
   kernel) (layer 0).
3. Decoder layer prefill and one decode step (layer 39, the last layer).

Run with (on each of the 2 nodes):
    python3 contrib/models/granite/test/module/launch_multinode.py \
        -m pytest -v contrib/models/granite/test/module/test_device_modules.py
"""

import importlib.util
import os
import pathlib
import sys
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Load this model folder's ``test/_local_src.py`` by file location.

    Every contrib model ships a file with this name, so a plain
    ``import _local_src`` would bind whichever copy entered ``sys.modules``
    first. The per-model name used here cannot collide.
    """
    name = "contrib_granite_local_src"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()
_reference = _local_src.helper("granite_reference")

TP_SIZE = _reference.TP_SIZE
attn_metadata = _reference.attn_metadata
hf_attention = _reference.hf_attention
hf_decoder_layer = _reference.hf_decoder_layer
hf_mlp = _reference.hf_mlp
hf_rope = _reference.hf_rope
load_hf_config = _reference.load_hf_config
module_mappings = _reference.module_mappings
node_rank = _reference.node_rank
resolve_checkpoint = _reference.resolve_checkpoint
run_hf_attention = _reference.run_hf_attention
run_hf_decoder_layer = _reference.run_hf_decoder_layer

NNODES = 2
DEVICE = "neuron:0"
PREFILL_LEN = 1024
MLP_DECODE_TOKENS = 4
BLOCK_SIZE = 32
# Align the block count to LNC2 * 128, as the model runner does.
BLOCKS_ALIGNMENT = 2 * 128
COLLECT_TIMEOUT_S = 3000

pytestmark = [
    pytest.mark.skipif(
        envs.VLLM_NEURON_CPU_MODE,
        reason="Granite device module tests need Neuron hardware.",
    ),
    pytest.mark.skipif(
        "MASTER_ADDR" not in os.environ,
        reason="Run with launch_multinode.py on 2 nodes (MASTER_ADDR is not set).",
    ),
]


def _num_blocks(total_tokens: int) -> int:
    min_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    aligned = (
        (min_blocks + BLOCKS_ALIGNMENT - 1) // BLOCKS_ALIGNMENT
    ) * BLOCKS_ALIGNMENT
    return max(BLOCKS_ALIGNMENT, aligned)


NUM_BLOCKS = _num_blocks(PREFILL_LEN + 1)


@pytest.fixture(scope="module")
def checkpoint():
    return resolve_checkpoint()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


# =============================================================================
# Harnesses (built inside each worker)
# =============================================================================


class _ModuleHarness(torch.nn.Module):
    """Hold one Granite module. Give the SP input slice to it in prefill.

    kind: "attention", "mlp" or "layer".
    """

    def __init__(self, checkpoint_path: str, kind: str, layer_idx: int):
        super().__init__()
        from vllm_neuron.model.neuron_config import NeuronConfig

        _src_model = _local_src.load("model")
        GraniteConfig = _local_src.load("config").GraniteConfig
        GraniteAttention = _src_model.GraniteAttention
        GraniteDecoderLayer = _src_model.GraniteDecoderLayer
        GraniteMLP = _src_model.GraniteMLP

        cfg = GraniteConfig.from_configs(
            load_hf_config(checkpoint_path), NeuronConfig()
        )
        self.kind = kind
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if kind == "attention":
            self.module = GraniteAttention(cfg, layer_idx=layer_idx)
        elif kind == "mlp":
            self.module = GraniteMLP(cfg)
        else:
            self.module = GraniteDecoderLayer(cfg, layer_idx=layer_idx)
        attn = self._attention()
        self._cache_shape = (
            None
            if attn is None
            else (
                NUM_BLOCKS,
                attn.num_key_value_heads_per_rank,
                BLOCK_SIZE,
                cfg.head_dim,
            )
        )

    def _attention(self):
        # Find the attention module through self.module. Do not keep a second
        # reference as an attribute: nn.Module registers that reference as a
        # second submodule, and its parameters then have no mapping.
        if self.kind == "attention":
            return self.module
        if self.kind == "layer":
            return self.module.self_attn
        return None

    def bind_cache(self, device) -> None:
        attn = self._attention()
        if attn is not None:
            attn.k_cache = torch.zeros(
                self._cache_shape, dtype=torch.bfloat16, device=device
            )
            attn.v_cache = torch.zeros(
                self._cache_shape, dtype=torch.bfloat16, device=device
            )

    def _sp_slice(self, hidden_states):
        per_rank = hidden_states.shape[0] // self.world_size
        return hidden_states[self.rank * per_rank : (self.rank + 1) * per_rank]

    def forward(self, hidden_states, positions, cos, sin, metadata, phase):
        is_prefill = phase == "prefill"
        if is_prefill:
            hidden_states = self._sp_slice(hidden_states)
        if self.kind == "mlp":
            return self.module(hidden_states, is_prefill=is_prefill)
        return self.module(hidden_states, positions, (cos, sin), metadata)


def _load_harness(checkpoint_path: str, kind: str, layer_idx: int):
    """Build the harness, load the TP shards of the real weights, compile it."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    harness = _ModuleHarness(checkpoint_path, kind, layer_idx)
    module_prefix = {
        "attention": f"model.layers.{layer_idx}.self_attn.",
        "mlp": f"model.layers.{layer_idx}.mlp.",
        "layer": f"model.layers.{layer_idx}.",
    }[kind]
    mappings = module_mappings(
        load_hf_config(checkpoint_path), module_prefix, "module."
    )
    result = SafetensorsCheckpoint(checkpoint_path).load_sharded_pipelined(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        model=harness,
        mappings=mappings,
        device=DEVICE,
    )
    assert not result.missing_keys, result.missing_keys
    # The checkpoint is FP32. The modules run in bfloat16.
    state = {k: v.to(torch.bfloat16) for k, v in result.state_dict.items()}
    missing, unexpected = harness.load_state_dict(state, strict=False, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    harness.bind_cache(DEVICE)
    return torch.compile(
        harness, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def _make_executor(checkpoint_path: str, kind: str, layer_idx: int, world_size: int):
    return MPExecutor(
        world_size=world_size,
        model_load=partial(_load_harness, checkpoint_path, kind, layer_idx),
        device=DEVICE,
        nnodes=NNODES,
        node_rank=node_rank(),
        load_timeout=COLLECT_TIMEOUT_S,
    )


def _local_global_ranks(world_size: int) -> list[int]:
    local = world_size // NNODES
    return [node_rank() * local + i for i in range(local)]


def _check_sp_rows(outputs, ref_fp32, ref_bf16, world_size, name):
    """Compare the SP output slice of each local rank with the reference rows."""
    per_rank = ref_fp32.shape[0] // world_size
    for out, g in zip(outputs, _local_global_ranks(world_size)):
        rows = slice(g * per_rank, (g + 1) * per_rank)
        assert out.shape == ref_fp32[rows].shape, (out.shape, ref_fp32[rows].shape)
        assert_close_three_way(
            ref_fp32[rows],
            ref_bf16[rows],
            out.detach().cpu(),
            name=f"{name}/rank{g}",
            plot_on_failure=True,
            output_dir="accuracy_debug",
        )


def _check_replicated(outputs, ref_fp32, ref_bf16, name):
    """Compare the full output of each local rank with the reference."""
    for i, out in enumerate(outputs):
        assert_close_three_way(
            ref_fp32,
            ref_bf16,
            out.detach().cpu(),
            name=f"{name}/local{i}",
            plot_on_failure=True,
            output_dir="accuracy_debug",
        )


def _run_prefill_then_decode(executor, hf_config, hidden, layer_name):
    """Prefill PREFILL_LEN tokens, then decode the token at PREFILL_LEN."""
    total = PREFILL_LEN + 1
    cos, sin = hf_rope(hf_config, total, torch.bfloat16)
    cos, sin = cos[0], sin[0]

    executor.dispatch(
        hidden[:PREFILL_LEN],
        torch.arange(PREFILL_LEN, dtype=torch.long),
        cos[:PREFILL_LEN],
        sin[:PREFILL_LEN],
        attn_metadata([layer_name], PREFILL_LEN, BLOCK_SIZE, NUM_BLOCKS, True),
        "prefill",
    )
    prefill_out = executor.collect(timeout=COLLECT_TIMEOUT_S)

    executor.dispatch(
        hidden[PREFILL_LEN:],
        torch.tensor([PREFILL_LEN], dtype=torch.long),
        cos[PREFILL_LEN:],
        sin[PREFILL_LEN:],
        attn_metadata([layer_name], total, BLOCK_SIZE, NUM_BLOCKS, False),
        "decode",
    )
    decode_out = executor.collect(timeout=COLLECT_TIMEOUT_S)
    return prefill_out, decode_out


# =============================================================================
# 1. Attention prefill -> decode (layer 0)
# =============================================================================


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_attention_prefill_then_decode(checkpoint, world_size):
    """Attention prefill and decode from the written KV cache match HF."""
    layer_idx = 0
    hf_config = load_hf_config(checkpoint)
    hidden = torch.randn(PREFILL_LEN + 1, hf_config.hidden_size, dtype=torch.bfloat16)

    executor = _make_executor(checkpoint, "attention", layer_idx, world_size)
    prefill_out, decode_out = _run_prefill_then_decode(
        executor, hf_config, hidden, f"layers.{layer_idx}.self_attn"
    )

    ref_fp32 = run_hf_attention(
        hf_attention(checkpoint, hf_config, layer_idx, torch.float32),
        hidden.float(),
        hf_config,
    )
    ref_bf16 = run_hf_attention(
        hf_attention(checkpoint, hf_config, layer_idx, torch.bfloat16),
        hidden,
        hf_config,
    )
    _check_sp_rows(
        prefill_out,
        ref_fp32[:PREFILL_LEN],
        ref_bf16[:PREFILL_LEN],
        world_size,
        "granite_attention/prefill",
    )
    _check_replicated(
        decode_out,
        ref_fp32[PREFILL_LEN:],
        ref_bf16[PREFILL_LEN:],
        "granite_attention/decode",
    )


# =============================================================================
# 2. MLP prefill (prefill kernel) and decode (decode kernel) (layer 0)
# =============================================================================


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_mlp_prefill_and_decode(checkpoint, world_size):
    """The MLP output for 1024 prefill tokens and 4 decode tokens matches HF."""
    layer_idx = 0
    hf_config = load_hf_config(checkpoint)
    prefill_hidden = torch.randn(
        PREFILL_LEN, hf_config.hidden_size, dtype=torch.bfloat16
    )
    decode_hidden = torch.randn(
        MLP_DECODE_TOKENS, hf_config.hidden_size, dtype=torch.bfloat16
    )

    executor = _make_executor(checkpoint, "mlp", layer_idx, world_size)
    # The MLP harness does not use the positions, the RoPE or the metadata.
    executor.dispatch(prefill_hidden, None, None, None, None, "prefill")
    prefill_out = executor.collect(timeout=COLLECT_TIMEOUT_S)
    executor.dispatch(decode_hidden, None, None, None, None, "decode")
    decode_out = executor.collect(timeout=COLLECT_TIMEOUT_S)

    mlp_fp32 = hf_mlp(checkpoint, hf_config, layer_idx, torch.float32)
    mlp_bf16 = hf_mlp(checkpoint, hf_config, layer_idx, torch.bfloat16)
    with torch.no_grad():
        prefill_fp32 = mlp_fp32(prefill_hidden.float())
        prefill_bf16 = mlp_bf16(prefill_hidden)
        decode_fp32 = mlp_fp32(decode_hidden.float())
        decode_bf16 = mlp_bf16(decode_hidden)

    _check_sp_rows(
        prefill_out, prefill_fp32, prefill_bf16, world_size, "granite_mlp/prefill"
    )
    _check_replicated(decode_out, decode_fp32, decode_bf16, "granite_mlp/decode")


# =============================================================================
# 3. Decoder layer prefill -> decode (layer 39, the last layer)
# =============================================================================


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_decoder_layer_prefill_then_decode(checkpoint, world_size):
    """The full last layer (norms, residual scale, attention, MLP) matches HF."""
    layer_idx = 39
    hf_config = load_hf_config(checkpoint)
    hidden = torch.randn(PREFILL_LEN + 1, hf_config.hidden_size, dtype=torch.bfloat16)

    executor = _make_executor(checkpoint, "layer", layer_idx, world_size)
    prefill_out, decode_out = _run_prefill_then_decode(
        executor, hf_config, hidden, f"layers.{layer_idx}.self_attn"
    )

    ref_fp32 = run_hf_decoder_layer(
        hf_decoder_layer(checkpoint, hf_config, layer_idx, torch.float32),
        hidden.float(),
        hf_config,
    )
    ref_bf16 = run_hf_decoder_layer(
        hf_decoder_layer(checkpoint, hf_config, layer_idx, torch.bfloat16),
        hidden,
        hf_config,
    )
    _check_sp_rows(
        prefill_out,
        ref_fp32[:PREFILL_LEN],
        ref_bf16[:PREFILL_LEN],
        world_size,
        "granite_layer39/prefill",
    )
    _check_replicated(
        decode_out,
        ref_fp32[PREFILL_LEN:],
        ref_bf16[PREFILL_LEN:],
        "granite_layer39/decode",
    )
