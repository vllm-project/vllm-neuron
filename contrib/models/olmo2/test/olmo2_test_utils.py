# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OLMo2 (allenai/OLMo-2-0425-1B) contrib tests.

This file travels with the model: it is the port's own test helper, vendored
into the contrib folder so that nothing outside
``contrib/models/olmo2/`` is needed. It sits at the ``test/`` root because
``test/unit`` and ``test/module`` both import it, so neither of those
directories owns it (``test/conftest.py`` puts this directory on ``sys.path``).
The implementation under ``src/`` is reached through ``_local_src``.

The helpers give:
- The checkpoint path and the topology (TP 8, on 1 node or on 2 nodes).
- Independent HuggingFace references for layer 0 (attention and MLP). The
  references come from the HF ``Olmo2`` modules only. They do not use the
  port code.
- Picklable ``model_load`` functions and harness modules for ``MPExecutor``.
- Attention metadata builders for prefill and decode.

The topology and the checkpoint come from the environment:
- ``OLMO2_NNODES`` is the number of nodes (default 1, all 8 ranks on one
  node). The recorded 2-node runs used a launcher in the plugin test tree
  that is not part of this contribution; to run on 2 nodes, set
  ``OLMO2_NNODES=2``, ``NODE_RANK`` and ``NEURON_RT_ROOT_COMM_ID`` yourself.
- ``NODE_RANK`` is the rank of this node (default 0).
- ``OLMO2_CHECKPOINT_PATH`` points at a checkpoint that is already on disk.
  Without it, the pinned revision comes from the HuggingFace hub.
"""

from __future__ import annotations

import copy
import json
import os

import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron import envs
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from _local_src import src_module, src_package

MODEL_ID = "allenai/OLMo-2-0425-1B"
HF_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"

# TP 8 is the registered topology. 16 heads / 8 ranks gives 2 heads per rank
# (an even count for the decode megakernel). Each rank owns 2 whole KV heads.
WORLD_SIZE = 8
SEED = 42
LAYER_IDX = 0
LAYER_NAME = f"layers.{LAYER_IDX}.self_attn"

# One reference sequence gives all component goldens. The rows 0..1023 are the
# prefill rows. The rows 1024..1027 are the decode rows. Causal attention makes
# the prefill rows equal to a 1024-token forward.
PREFILL_LEN = 1024
DECODE_STEPS = 4
REFERENCE_LEN = PREFILL_LEN + DECODE_STEPS

BLOCK_SIZE = 32
# The decode kernel needs a block count that is a multiple of LNC * 128.
BLOCKS_ALIGNMENT = 2 * 128

COLLECT_TIMEOUT_S = 3600.0


# =============================================================================
# Checkpoint and topology
# =============================================================================


def resolve_checkpoint() -> str:
    """Return the local checkpoint directory of the pinned revision.

    ``OLMO2_CHECKPOINT_PATH`` points at a checkpoint that is already on disk.
    Without it, download the pinned revision from the HuggingFace hub.
    """
    path = os.environ.get("OLMO2_CHECKPOINT_PATH")
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"OLMO2_CHECKPOINT_PATH={path!r} has no config.json. "
                "Set OLMO2_CHECKPOINT_PATH to a directory with the checkpoint."
            )
        return path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )


def read_config_json(checkpoint_path: str) -> dict:
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        return json.load(f)


def num_nodes() -> int:
    return int(os.environ.get("OLMO2_NNODES", "1"))


def node_rank() -> int:
    return int(os.environ.get("NODE_RANK", "0"))


def ranks_per_node() -> int:
    nnodes = num_nodes()
    if WORLD_SIZE % nnodes != 0:
        raise ValueError(
            f"OLMO2_NNODES={nnodes} does not divide the world size {WORLD_SIZE}. "
            "Set OLMO2_NNODES to 1, 2, 4 or 8."
        )
    return WORLD_SIZE // nnodes


def local_global_ranks() -> list[int]:
    """Return the global ranks of the local workers, in collect order."""
    per_node = ranks_per_node()
    first = node_rank() * per_node
    return list(range(first, first + per_node))


def component_device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def _check_ccom_root() -> None:
    """Make sure that all nodes use the same CCOM root address.

    A loopback root address on each node starts one CCOM root on each node.
    Then the first collective across the nodes does not complete.
    """
    if num_nodes() == 1 or envs.VLLM_NEURON_CPU_MODE:
        return
    root = os.environ.get("NEURON_RT_ROOT_COMM_ID", "")
    host = root.rsplit(":", 1)[0]
    if host in ("", "localhost", "127.0.0.1"):
        raise RuntimeError(
            f"NEURON_RT_ROOT_COMM_ID={root!r} is not an address of node 0, but "
            f"the test uses {num_nodes()} nodes. Set NEURON_RT_ROOT_COMM_ID to "
            "<node-0-address>:<port> before pytest starts, or set OLMO2_NNODES=1 "
            "to put all ranks on one node."
        )


def make_executor(model_load):
    """Make the MPExecutor for the registered topology."""
    from vllm_neuron.utils.executor import MPExecutor

    _check_ccom_root()
    return MPExecutor(
        world_size=WORLD_SIZE,
        model_load=model_load,
        device=component_device(),
        nnodes=num_nodes(),
        node_rank=node_rank(),
    )


# =============================================================================
# Independent HuggingFace references
# =============================================================================


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return mask.triu(diagonal=1)[None, None]


def _hf_attention(attn, rotary_emb, hidden: torch.Tensor) -> torch.Tensor:
    """Run one HF attention module on [1, T, H]. Return [T, H]."""
    seq_len = hidden.shape[1]
    positions = torch.arange(seq_len, dtype=torch.long)[None]
    position_embeddings = rotary_emb(hidden, positions)
    out = attn(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=_causal_mask(seq_len, hidden.dtype),
    )[0]
    return out[0]


def compute_layer_goldens(checkpoint_path: str) -> dict:
    """Compute the FP32 and BF16 HF references for layer 0.

    The inputs come from one FP32 history:
    - attention input = FP32 embeddings of seeded random token IDs
    - MLP input = attention input + post_attention_layernorm(FP32 attention)

    The FP32 reference and the BF16 reference use these same inputs. The
    Neuron run uses the same inputs, cast to BF16.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    layer = model.model.layers[LAYER_IDX]
    layer_bf16 = copy.deepcopy(layer).to(torch.bfloat16)
    rotary_emb = model.model.rotary_emb

    generator = torch.Generator().manual_seed(SEED)
    input_ids = torch.randint(
        0, model.config.vocab_size, (1, REFERENCE_LEN), generator=generator
    )
    positions = torch.arange(REFERENCE_LEN, dtype=torch.long)[None]

    with torch.no_grad():
        attn_input = model.model.embed_tokens(input_ids)  # [1, T, H] FP32
        attn_fp32 = _hf_attention(layer.self_attn, rotary_emb, attn_input)
        attn_bf16 = _hf_attention(
            layer_bf16.self_attn, rotary_emb, attn_input.to(torch.bfloat16)
        )

        mlp_input = attn_input + layer.post_attention_layernorm(attn_fp32[None])
        mlp_fp32 = layer.mlp(mlp_input)[0]
        mlp_bf16 = layer_bf16.mlp(mlp_input.to(torch.bfloat16))[0]

        cos, sin = rotary_emb(attn_input, positions)

    return {
        "input_ids": input_ids[0],
        "attn_input": attn_input[0].contiguous(),
        "attn_fp32": attn_fp32.contiguous(),
        "attn_bf16": attn_bf16.contiguous(),
        "mlp_input": mlp_input[0].contiguous(),
        "mlp_fp32": mlp_fp32.contiguous(),
        "mlp_bf16": mlp_bf16.contiguous(),
        "cos": cos[0].contiguous(),
        "sin": sin[0].contiguous(),
    }


def get_layer_goldens(checkpoint_path: str) -> dict:
    """Load the layer-0 goldens from the cache, or compute them."""
    import transformers

    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    key_config = {
        "model_id": MODEL_ID,
        "hf_revision": HF_REVISION,
        "test": "olmo2_layer_components",
        "reference": "hf_olmo2_eager_fp32_bf16",
        "layer_idx": LAYER_IDX,
        "reference_len": REFERENCE_LEN,
        "seed": SEED,
    }
    return get_or_compute_goldens(
        key_config=key_config,
        compute_fn=lambda: compute_layer_goldens(checkpoint_path),
        metadata={"transformers_version": transformers.__version__},
    )


# =============================================================================
# Attention metadata
# =============================================================================


def num_cache_blocks() -> int:
    min_blocks = (REFERENCE_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    aligned = (min_blocks + BLOCKS_ALIGNMENT - 1) // BLOCKS_ALIGNMENT
    return max(BLOCKS_ALIGNMENT, aligned * BLOCKS_ALIGNMENT)


def prefill_metadata(seq_len: int) -> dict:
    num_blocks = num_cache_blocks()
    return {
        LAYER_NAME: {
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "slot_mapping": torch.arange(seq_len, dtype=torch.long),
            "max_query_len": seq_len,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": num_blocks,
            "decode_token_threshold": 1,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def decode_metadata(position: int) -> dict:
    num_blocks = num_cache_blocks()
    return {
        LAYER_NAME: {
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "slot_mapping": torch.tensor([position], dtype=torch.long),
            "max_query_len": 1,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": num_blocks,
            "decode_token_threshold": 1,
        }
    }


# =============================================================================
# Module loads (top-level functions, so that functools.partial can pickle them)
# =============================================================================


def attention_checkpoint_mappings() -> dict:
    """Map the attention parameters to the HF keys. Written for the test only."""
    prefix = f"model.layers.{LAYER_IDX}.self_attn"
    return {
        "qkv_proj_weight": [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ],
        "o_proj_weight": f"{prefix}.o_proj.weight",
        "q_norm_weight": f"{prefix}.q_norm.weight",
        "k_norm_weight": f"{prefix}.k_norm.weight",
    }


def mlp_checkpoint_mappings() -> dict:
    """Map the MLP parameters to the HF keys. Written for the test only."""
    prefix = f"model.layers.{LAYER_IDX}.mlp"
    return {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }


def _load_module_weights(
    module: nn.Module, checkpoint_path: str, mappings: dict, device: str
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    checkpoint = SafetensorsCheckpoint(checkpoint_path, None)
    if envs.VLLM_NEURON_CPU_MODE:
        result = checkpoint.load_sharded(rank, world_size, module, mappings, device)
    else:
        result = checkpoint.load_sharded_pipelined(
            rank, world_size, module, mappings, device
        )
    # The checkpoint stores FP32 weights. The port computes in BF16.
    state_dict = {k: v.to(torch.bfloat16) for k, v in result.state_dict.items()}
    module.load_state_dict(state_dict, strict=True, assign=True)


def _maybe_compile(module: nn.Module) -> nn.Module:
    if envs.VLLM_NEURON_CPU_MODE:
        return module
    return torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )


class AttentionHarness(nn.Module):
    """Run Olmo2Attention. Give each rank its SP slice of a prefill input."""

    def __init__(self, attn: nn.Module, rank: int, tokens_per_rank: int):
        super().__init__()
        self.attn = attn
        self.rank = rank
        self.tokens_per_rank = tokens_per_rank

    def forward(self, hidden_states, positions, cos, sin, attn_metadata):
        meta = attn_metadata[LAYER_NAME]
        if meta["max_query_len"] > meta["decode_token_threshold"]:
            start = self.rank * self.tokens_per_rank
            hidden_states = hidden_states[start : start + self.tokens_per_rank]
        return self.attn(hidden_states, positions, (cos, sin), attn_metadata)


def attention_model_load(checkpoint_path: str, device: str) -> nn.Module:
    """Build layer-0 Olmo2Attention with the real weights and a zero KV cache."""
    Olmo2Config = src_module("config").Olmo2Config
    Olmo2Attention = src_module("model").Olmo2Attention

    config = Olmo2Config.from_configs(os.path.join(checkpoint_path, "config.json"))
    attn = Olmo2Attention(config, layer_idx=LAYER_IDX)
    _load_module_weights(attn, checkpoint_path, attention_checkpoint_mappings(), device)

    cache_shape = (
        num_cache_blocks(),
        attn.num_key_value_heads_per_rank,
        BLOCK_SIZE,
        attn.head_dim,
    )
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    harness = AttentionHarness(
        attn, rank=dist.get_rank(), tokens_per_rank=PREFILL_LEN // WORLD_SIZE
    )
    return _maybe_compile(harness)


class MlpHarness(nn.Module):
    """Run Olmo2MLP. Give each rank its SP slice of a prefill input."""

    def __init__(self, mlp: nn.Module, rank: int, tokens_per_rank: int):
        super().__init__()
        self.mlp = mlp
        self.rank = rank
        self.tokens_per_rank = tokens_per_rank

    def forward(self, hidden_states, is_prefill: bool):
        if is_prefill:
            start = self.rank * self.tokens_per_rank
            hidden_states = hidden_states[start : start + self.tokens_per_rank]
        return self.mlp(hidden_states, is_prefill=is_prefill)


def mlp_model_load(checkpoint_path: str, device: str) -> nn.Module:
    """Build layer-0 Olmo2MLP with the real weights."""
    Olmo2Config = src_module("config").Olmo2Config
    Olmo2MLP = src_module("model").Olmo2MLP

    config = Olmo2Config.from_configs(os.path.join(checkpoint_path, "config.json"))
    mlp = Olmo2MLP(config)
    _load_module_weights(mlp, checkpoint_path, mlp_checkpoint_mappings(), device)
    harness = MlpHarness(
        mlp, rank=dist.get_rank(), tokens_per_rank=PREFILL_LEN // WORLD_SIZE
    )
    return _maybe_compile(harness)


# =============================================================================
# Full-model load report (CPU)
# =============================================================================

# Each entry: (parameter name, row slice, column slice). A slice keeps the
# report small. The test builds the expected values from the safetensors files.
REPORT_LAYERS = (0, 15)


def report_spec() -> list[tuple[str, slice, slice]]:
    full = slice(None)
    first_rows = slice(0, 8)
    spec = [
        ("model.embed_tokens.weight", first_rows, full),
        ("model.embed_tokens.weight", slice(-8, None), full),
        ("lm_head.weight", first_rows, full),
        ("lm_head.weight", slice(-8, None), full),
        ("model.norm.weight", full, full),
    ]
    for layer in REPORT_LAYERS:
        prefix = f"model.layers.{layer}"
        spec += [
            (f"{prefix}.self_attn.qkv_proj_weight", full, full),
            (f"{prefix}.self_attn.o_proj_weight", full, full),
            (f"{prefix}.self_attn.q_norm_weight", full, full),
            (f"{prefix}.self_attn.k_norm_weight", full, full),
            (f"{prefix}.mlp.gate_proj_weight", full, slice(0, 8)),
            (f"{prefix}.mlp.up_proj_weight", full, slice(0, 8)),
            (f"{prefix}.mlp.down_proj_weight", first_rows, full),
            (f"{prefix}.post_attention_layernorm.weight", full, full),
            (f"{prefix}.post_feedforward_layernorm.weight", full, full),
        ]
    return spec


class LoadReportHarness(nn.Module):
    """Report the load status and the selected weight slices of one rank.

    The output is a tuple:
    - [0]: int64 [5] = parameter count, meta tensors, non-finite parameters,
      parameters with a dtype other than BF16, and buffers on meta.
    - [1:]: the tensors of ``report_spec()``, in the same order.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self):
        params = dict(self.model.named_parameters())
        num_meta = sum(1 for p in params.values() if p.device.type == "meta")
        num_nonfinite = sum(
            1
            for p in params.values()
            if p.device.type != "meta" and not torch.isfinite(p).all()
        )
        num_wrong_dtype = sum(1 for p in params.values() if p.dtype != torch.bfloat16)
        num_meta_buffers = sum(
            1 for b in self.model.buffers() if b.device.type == "meta"
        )
        status = torch.tensor(
            [len(params), num_meta, num_nonfinite, num_wrong_dtype, num_meta_buffers],
            dtype=torch.int64,
        )
        selected = []
        for name, rows, cols in report_spec():
            tensor = params[name].detach()
            if tensor.dim() == 1:
                selected.append(tensor[rows].clone())
            else:
                selected.append(tensor[rows, cols].clone())
        return (status, *selected)


def full_model_load(checkpoint_path: str, device: str) -> nn.Module:
    """Build the full model on meta, then load the TP shards of this rank."""
    from transformers import AutoConfig

    Olmo2ForCausalLM = src_package().Olmo2ForCausalLM

    hf_config = AutoConfig.from_pretrained(checkpoint_path)
    with torch.device("meta"):
        model = Olmo2ForCausalLM.from_configs(hf_config, None)
    model.load_weights(checkpoint_path, torch.device(device), None)
    return LoadReportHarness(model)
