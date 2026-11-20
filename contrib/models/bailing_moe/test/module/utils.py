# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the BailingMoeV2 (inclusionAI/Ling-mini-2.0) tests.

The component tests run on 8 ranks with expert parallelism 8. In CPU mode the
8 ranks run on one host. On a device the 8 ranks span 2 trn2.48xlarge nodes
with 4 ranks on each node. Run the device tests through
``test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py``. The
launcher sets NODE_RANK, MASTER_ADDR and NEURON_RT_ROOT_COMM_ID.

Each ``*_model_load`` function is a top-level function, so that
``functools.partial`` of it is picklable for MPExecutor.
"""

import importlib.util
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron import envs

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

MODEL_ID = "inclusionAI/Ling-mini-2.0"
REVISION = "a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab"

# Native topology: TP 8 with EP 8. Each rank holds 32 experts with the full
# intermediate size 512, so the prefill MoE kernel applies.
WORLD_SIZE = 8
EP_DEGREE = 8
DEVICE_NNODES = 2

# Layer 1 is the first MoE layer. Layer 0 has the dense MLP.
MOE_LAYER = 1

# KV cache of the component tests: 64 blocks of 32 tokens (2048 tokens).
BLOCK_SIZE = 32
NUM_BLOCKS = 64

VOCAB_SIZE = 157184

# Namespace of the full-model goldens. The goldens come from the independent
# torch reference, not from the HF remote modeling code.
GOLDEN_VARIANT = "bailing_torch_reference_a810f64"


def resolve_checkpoint() -> str:
    """Return a local snapshot of the pinned Ling-mini-2.0 revision.

    Example:
        >>> path = resolve_checkpoint()
        >>> os.path.exists(os.path.join(path, "config.json"))
        True
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=REVISION)


def read_hf_config(checkpoint: str) -> dict:
    """Read ``config.json`` of the checkpoint as a dict."""
    with open(os.path.join(checkpoint, "config.json")) as f:
        return json.load(f)


def device_topology() -> tuple[int, int]:
    """Return ``(nnodes, node_rank)`` for the MPExecutor of this host.

    CPU mode runs all ranks on one host. A device run spans 2 nodes.

    Example:
        >>> nnodes, node_rank = device_topology()
    """
    if envs.VLLM_NEURON_CPU_MODE:
        return 1, 0
    node_rank = os.environ.get("NODE_RANK")
    if node_rank is None:
        raise RuntimeError(
            "NODE_RANK is not set. Run the BailingMoeV2 device tests through "
            "test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py."
        )
    return DEVICE_NNODES, int(node_rank)


def local_global_ranks(nnodes: int, node_rank: int) -> list[int]:
    """Return the global ranks of the MPExecutor workers on this host."""
    per_node = WORLD_SIZE // nnodes
    return list(range(node_rank * per_node, (node_rank + 1) * per_node))


def set_mock_vllm_config(ep_enabled: bool = True) -> None:
    """Set a mock vLLM config for module construction outside a vLLM engine.

    The MoE block reads ``enable_expert_parallel`` and ``data_parallel_size``.
    The full model also reads ``cache_config.cache_dtype``.
    """
    import vllm.config as vllm_config_module

    mock_config = MagicMock()
    mock_config.parallel_config.enable_expert_parallel = ep_enabled
    mock_config.parallel_config.data_parallel_size = 1
    mock_config.cache_config.cache_dtype = "auto"
    vllm_config_module.get_current_vllm_config = lambda: mock_config


def neuron_config_for(checkpoint: str):
    """Return the BailingMoeV2Config of the checkpoint with a default NeuronConfig."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    BailingMoeV2Config = _local_src.load("config").BailingMoeV2Config

    return BailingMoeV2Config.from_configs(read_hf_config(checkpoint), NeuronConfig())


def attention_mappings(layer_idx: int) -> dict[str, str]:
    """Return the module-local weight mappings of one attention block."""
    p = f"model.layers.{layer_idx}.attention"
    return {
        "attn.qkv_proj_weight": f"{p}.query_key_value.weight",
        "attn.o_proj_weight": f"{p}.dense.weight",
        "attn.query_layernorm.weight": f"{p}.query_layernorm.weight",
        "attn.key_layernorm.weight": f"{p}.key_layernorm.weight",
    }


def moe_mappings(layer_idx: int, num_experts: int) -> dict:
    """Return the module-local weight mappings of one MoE block."""
    p = f"model.layers.{layer_idx}.mlp"
    mappings = {
        "block.gate.weight": f"{p}.gate.weight",
        "block.gate.expert_bias": f"{p}.gate.expert_bias",
        "block.gate_up_proj_weight": [
            f"{p}.experts.{e}.gate_proj.weight" for e in range(num_experts)
        ]
        + [f"{p}.experts.{e}.up_proj.weight" for e in range(num_experts)],
        "block.down_proj_weight": [
            f"{p}.experts.{e}.down_proj.weight" for e in range(num_experts)
        ],
    }
    for proj in ("gate_proj", "up_proj", "down_proj"):
        mappings[f"block.shared_experts.{proj}_weight"] = (
            f"{p}.shared_experts.{proj}.weight"
        )
    return mappings


def load_module_weights(
    module: nn.Module, checkpoint: str, mappings: dict, cpu_mode: bool, device: str
) -> None:
    """Load the rank shard of ``mappings`` into ``module``.

    Each loaded tensor gets the dtype of its target parameter. Thus the router
    stays in FP32 and the other weights stay in BF16.
    """
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    rank = dist.get_rank()
    ws = dist.get_world_size()
    ckpt = SafetensorsCheckpoint(checkpoint)
    if cpu_mode:
        loaded = ckpt.load_sharded(
            rank=rank, world_size=ws, model=module, mappings=mappings, device=device
        )
    else:
        loaded = ckpt.load_sharded_pipelined(
            rank=rank, world_size=ws, model=module, mappings=mappings, device=device
        )
    params = dict(module.named_parameters())
    missing = sorted(set(mappings) - set(loaded.state_dict))
    if missing:
        raise RuntimeError(f"The checkpoint load did not give {missing}.")
    state = {
        name: tensor.to(params[name].dtype)
        for name, tensor in loaded.state_dict.items()
    }
    module.load_state_dict(state, strict=False, assign=True)


# =============================================================================
# Kernel dispatch recorder
# =============================================================================

# The NKI call sites that each component phase must contain. Each site is the
# innermost ``vllm_neuron/functional`` frame (file, function) of an NKI node in
# the compiled graph. Each listed function calls its kernel through
# ``wrap_nki`` directly, so a torch fallback gives no node with this frame.
REQUIRED_NKI_SITES = {
    "attention prefill": {
        ("qkv.py", "qkv_proj"),
        ("attention_cte.py", "flash_attention"),
        ("o_proj.py", "o_proj"),
    },
    "attention decode": {
        ("attention_decode_mask.py", "_gen_mask_via_kernel"),
        ("attention_decode.py", "attention_decode"),
    },
    "moe prefill": {("moe_cte.py", "moe_cte"), ("mlp.py", "mlp")},
    "moe decode": {("moe_tkg.py", "moe_tkg"), ("mlp.py", "mlp")},
    "dense mlp prefill": {("mlp.py", "mlp")},
    "dense mlp decode": {("mlp.py", "mlp")},
}

_FRAME_RE = re.compile(r'File "([^"]+)", line \d+, in (\S+)')
_NO_STACK_TRACE = ("<no stack trace>", "<unknown>")


def _is_nki_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.HigherOrderOperator):
        return True
    return "nki" in str(getattr(target, "__name__", target)).lower()


def nki_call_sites(gm: torch.fx.GraphModule) -> list[list[str]]:
    """Return the ``[file, function]`` call site of each NKI node in ``gm``.

    The call site is the innermost ``vllm_neuron/functional`` frame of the
    ``stack_trace`` metadata that Dynamo writes on each node.

    Example:
        >>> sites = nki_call_sites(gm)
        >>> ["mlp.py", "mlp"] in sites
        True
    """
    sites = set()
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if not _is_nki_node(node):
                continue
            trace = node.meta.get("stack_trace")
            if not trace:
                sites.add(_NO_STACK_TRACE)
                continue
            frames = [
                (os.path.basename(path), func)
                for path, func in _FRAME_RE.findall(trace)
                if "/vllm_neuron/functional/" in path.replace("\\", "/")
            ]
            sites.add(
                frames[-1] if frames else ("<outside functional>", str(node.target))
            )
    return [list(site) for site in sorted(sites)]


class RouteRecordingBackend:
    """Record the NKI call sites of each compiled graph, then compile it.

    Each graph that Dynamo gives to the backend adds one JSON line to
    ``record_path``. The line holds the NKI call sites of the graph. Then the
    real Neuron backend compiles the graph without change.
    """

    def __init__(self, backend_name: str, record_path: str):
        self.backend_name = backend_name
        self.record_path = record_path

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        with open(self.record_path, "a") as f:
            f.write(json.dumps({"nki_sites": nki_call_sites(gm)}) + "\n")
        return torch._dynamo.lookup_backend(self.backend_name)(gm, example_inputs)


def route_record_path(route_dir: str, global_rank: int) -> str:
    """Return the record file of one global rank."""
    return os.path.join(route_dir, f"rank{global_rank}.jsonl")


def read_route_records(route_dir: str, global_ranks: list[int]) -> dict:
    """Return the list of graph records of each rank in ``global_ranks``."""
    records = {}
    for rank in global_ranks:
        path = route_record_path(route_dir, rank)
        if not os.path.exists(path):
            records[rank] = []
            continue
        with open(path) as f:
            records[rank] = [json.loads(line) for line in f if line.strip()]
    return records


def assert_nki_sites(
    records: dict, start: dict, phase: str, required: set | None = None
) -> dict:
    """Assert that the graphs after ``start`` contain each required NKI site.

    ``start`` gives the number of records of each rank before the phase.
    The function returns the record counts after the phase.

    Example:
        >>> counts = assert_nki_sites(records, {0: 0}, "moe prefill")
    """
    required = REQUIRED_NKI_SITES[phase] if required is None else required
    counts = {}
    for rank, rank_records in records.items():
        new = rank_records[start.get(rank, 0) :]
        if not new:
            raise AssertionError(
                f"Rank {rank} compiled no graph for the {phase} phase. "
                "The dispatch recorder has no evidence."
            )
        seen = {tuple(site) for record in new for site in record["nki_sites"]}
        print(f"[bailing dispatch] rank {rank} {phase}: NKI sites {sorted(seen)}")
        missing = required - seen
        if missing:
            raise AssertionError(
                f"Rank {rank} {phase} graph has no NKI node for {sorted(missing)}. "
                f"The graph has {sorted(seen)}. The phase uses a torch fallback."
            )
        counts[rank] = len(rank_records)
    return counts


def _tp_group():
    from vllm.distributed import get_tp_group

    return get_tp_group()


def sp_shard(hidden_states: torch.Tensor, group) -> torch.Tensor:
    """Return the SP rows of this rank from the full ``[T, H]`` sequence.

    The row offset comes from the stamped rank tensor of ``group``, as in the
    runner. Thus all ranks trace the same graph and share one NEFF. A Python
    int rank puts a different constant into the graph of each rank, and each
    rank then runs a different NEFF with the same collectives.
    """
    from vllm_neuron.parallel.neuron_parallel_state import shard_select

    return shard_select(hidden_states, 0, group)


def _compile_or_eager(module: nn.Module, cpu_mode: bool, route_dir: str | None):
    if cpu_mode:
        return module
    backend = envs.get_compile_backend_name()
    if route_dir is not None:
        backend = RouteRecordingBackend(
            backend, route_record_path(route_dir, dist.get_rank())
        )
    return torch.compile(module.to("neuron:0"), fullgraph=True, backend=backend)


# =============================================================================
# Attention harness
# =============================================================================


class AttentionHarness(nn.Module):
    """Run one BailingMoeV2Attention with the model RoPE tables.

    Prefill takes the full sequence and gives this rank its SP shard, as the
    model does. Decode takes the replicated tokens.
    """

    def __init__(self, attn: nn.Module, rotary: nn.Module, group):
        super().__init__()
        self.attn = attn
        self.rotary = rotary
        self.group = group

    def forward(self, hidden_states, positions, attn_metadata, is_prefill: bool):
        cos, sin = self.rotary(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        if is_prefill:
            hidden_states = sp_shard(hidden_states, self.group)
        return self.attn(hidden_states, positions, (cos, sin), attn_metadata)


def attention_model_load(
    checkpoint: str,
    layer_idx: int,
    cpu_mode: bool,
    device: str,
    route_dir: str | None = None,
) -> nn.Module:
    """Top-level picklable model load of the attention component test."""
    BailingMoeV2Attention = _local_src.load("model").BailingMoeV2Attention
    BailingMoeV2RotaryEmbedding = _local_src.load("model").BailingMoeV2RotaryEmbedding

    set_mock_vllm_config(ep_enabled=True)
    config = neuron_config_for(checkpoint)
    attn = BailingMoeV2Attention(config, layer_idx=layer_idx)
    harness = AttentionHarness(attn, BailingMoeV2RotaryEmbedding(config), _tp_group())
    load_module_weights(
        harness, checkpoint, attention_mappings(layer_idx), cpu_mode, device
    )

    cache_shape = (
        NUM_BLOCKS,
        attn.num_key_value_heads_per_rank,
        BLOCK_SIZE,
        config.head_dim,
    )
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    return _compile_or_eager(harness, cpu_mode, route_dir)


def prefill_metadata(layer_idx: int, num_tokens: int) -> dict:
    """Return the attention metadata of one prefill sequence at positions 0..T-1."""
    return {
        f"layers.{layer_idx}.self_attn": {
            "max_query_len": num_tokens,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(num_tokens, dtype=torch.int64),
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(NUM_BLOCKS, dtype=torch.int32)[None],
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def decode_metadata(layer_idx: int, position: int) -> dict:
    """Return the attention metadata of one decode token of one sequence."""
    return {
        f"layers.{layer_idx}.self_attn": {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": torch.tensor([position], dtype=torch.int64),
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(NUM_BLOCKS, dtype=torch.int32)[None],
            "max_blocks_per_seq": NUM_BLOCKS,
        }
    }


# =============================================================================
# MoE harness
# =============================================================================


class MoEHarness(nn.Module):
    """Run one BailingMoeV2SparseMoeBlock.

    Prefill takes the full sequence and gives this rank its SP shard. Decode
    takes the replicated tokens.
    """

    def __init__(self, block: nn.Module, group):
        super().__init__()
        self.block = block
        self.group = group

    def forward(self, hidden_states, positions, is_prefill: bool):
        if is_prefill:
            hidden_states = sp_shard(hidden_states, self.group)
        return self.block(hidden_states, is_prefill, positions)


def moe_model_load(
    checkpoint: str,
    layer_idx: int,
    cpu_mode: bool,
    device: str,
    route_dir: str | None = None,
) -> nn.Module:
    """Top-level picklable model load of the MoE component test."""
    BailingMoeV2SparseMoeBlock = _local_src.load("model").BailingMoeV2SparseMoeBlock

    set_mock_vllm_config(ep_enabled=True)
    config = neuron_config_for(checkpoint)
    block = BailingMoeV2SparseMoeBlock(config)
    if block.ep_degree != EP_DEGREE or block.intermediate_size_per_rank != 512:
        raise RuntimeError(
            f"The MoE placement is ep_degree={block.ep_degree} with intermediate "
            f"shard {block.intermediate_size_per_rank}. The test needs EP {EP_DEGREE} "
            "with the full intermediate size 512."
        )
    harness = MoEHarness(block, _tp_group())
    load_module_weights(
        harness,
        checkpoint,
        moe_mappings(layer_idx, config.num_experts),
        cpu_mode,
        device,
    )
    return _compile_or_eager(harness, cpu_mode, route_dir)


def dense_mlp_mappings() -> dict[str, str]:
    """Return the module-local weight mappings of the dense MLP of layer 0."""
    return {
        f"block.{proj}_weight": f"model.layers.0.mlp.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj")
    }


def dense_mlp_model_load(
    checkpoint: str, cpu_mode: bool, device: str, route_dir: str | None = None
) -> nn.Module:
    """Top-level picklable model load of the dense MLP component test (layer 0)."""
    BailingMoeV2MLP = _local_src.load("model").BailingMoeV2MLP

    set_mock_vllm_config(ep_enabled=True)
    config = neuron_config_for(checkpoint)
    if config.is_moe_layer(0):
        raise RuntimeError("Layer 0 of the checkpoint is not a dense MLP layer.")
    block = BailingMoeV2MLP(config, config.intermediate_size)
    # The dense MLP has the same call contract as the MoE block.
    harness = MoEHarness(block, _tp_group())
    load_module_weights(harness, checkpoint, dense_mlp_mappings(), cpu_mode, device)
    return _compile_or_eager(harness, cpu_mode, route_dir)


# =============================================================================
# Full-model load harness
# =============================================================================


class LoadedModelProbe(nn.Module):
    """Load the full model with ``load_weights`` and return spot values.

    The probe fills each parameter with NaN before the load. After the load,
    a NaN marks a parameter that the load did not write.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, marker: torch.Tensor):
        nan_params = torch.zeros(1, dtype=torch.int64)
        for _, param in self.model.named_parameters():
            if param.is_floating_point() and torch.isnan(param).any():
                nan_params += 1
        layer = self.model.model.layers[MOE_LAYER]
        return (
            nan_params,
            torch.tensor([sum(p.numel() for p in self.model.parameters())]),
            layer.attention.qkv_proj_weight.detach().clone(),
            layer.mlp.gate.weight.detach().clone(),
            layer.mlp.gate_up_proj_weight[0].detach().clone(),
            layer.mlp.down_proj_weight[0].detach().clone(),
            self.model.lm_head.weight[:4].detach().clone(),
            self.model.model.layers[0].mlp.gate_proj_weight[:, :4].detach().clone(),
        )


def _row_nonfinite(hidden: torch.Tensor) -> torch.Tensor:
    """Return ``[rows]`` int32: 1 for a row with a NaN or inf value, else 0."""
    return (~torch.isfinite(hidden.float())).any(dim=-1).to(torch.int32)


def _row_peak(hidden: torch.Tensor) -> torch.Tensor:
    """Return ``[rows]`` FP32: the largest absolute value of each row."""
    return hidden.float().abs().amax(dim=-1)


def prefill_stage_names(num_layers: int) -> list[str]:
    """Return the stage names of the rows of ``FullModelPrefillProbe`` records."""
    names = ["embedding"]
    for layer_idx in range(num_layers):
        names += [f"layer {layer_idx} attention", f"layer {layer_idx} mlp"]
    return names + ["final norm"]


class FullModelPrefillProbe(nn.Module):
    """Run the full-model prefill and record each stage of the SP shard.

    The probe runs the modules of ``BailingMoeV2ForCausalLM`` in the order of
    the model forward. After the embedding, after each attention residual,
    after each MLP residual and after the final norm, it records two values
    for each row of the SP shard of this rank: a non-finite flag and the
    largest absolute value. The logits come from the same LM head and the
    same all-gather as the logits that the server returns.

    The probe does not run the sampler. The logit test reads the raw logits,
    and the sampler does not change them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, positions, attn_metadata, sampling_positions):
        backbone = self.model.model
        positions = positions.to(torch.int32)
        # The runner gives the stamped TP rank tensor, so all ranks share one
        # graph. MPExecutor stamps the same tensor.
        rank = getattr(backbone.tp_group, "rank_in_group_tensor", None)
        hidden = backbone.word_embeddings(input_ids, scatter_tokens=True, rank=rank)
        position_embeddings = backbone.rotary_emb(
            positions, device=hidden.device, dtype=hidden.dtype
        )
        flags = [_row_nonfinite(hidden)]
        peaks = [_row_peak(hidden)]

        # The same steps as BailingMoeV2DecoderLayer.forward for a prefill.
        for layer in backbone.layers:
            residual = hidden
            out = layer.attention(
                hidden_states=layer.input_layernorm(hidden),
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
            hidden = residual + out
            flags.append(_row_nonfinite(hidden))
            peaks.append(_row_peak(hidden))

            residual = hidden
            out = layer.mlp(
                layer.post_attention_layernorm(hidden),
                is_prefill=True,
                positions=positions,
            )
            hidden = residual + out
            flags.append(_row_nonfinite(hidden))
            peaks.append(_row_peak(hidden))

        hidden = backbone.norm(hidden)
        flags.append(_row_nonfinite(hidden))
        peaks.append(_row_peak(hidden))

        if backbone.world_size > 1:
            hidden = backbone.tp_group.all_gather(hidden, dim=0)
        hidden = torch.index_select(hidden, dim=0, index=sampling_positions)
        logits = self.model.lm_head(hidden)
        if self.model.world_size > 1:
            logits = self.model.tp_group.all_gather(logits, dim=1)
        return torch.stack(flags), torch.stack(peaks), logits.float()


def full_model_prefill_load(
    checkpoint: str,
    cpu_mode: bool,
    device: str,
    route_dir: str | None = None,
) -> nn.Module:
    """Top-level picklable load of the full-model prefill probe.

    Each layer gets a KV cache of ``NUM_BLOCKS`` blocks with the layout of the
    attention harness.
    """
    from vllm_neuron.model.neuron_config import NeuronConfig

    BailingMoeV2ForCausalLM = _local_src.load("factory").BailingMoeV2ForCausalLM

    set_mock_vllm_config(ep_enabled=True)
    model = BailingMoeV2ForCausalLM.from_configs(
        read_hf_config(checkpoint), NeuronConfig()
    )
    model.load_weights(checkpoint, torch.device(device), None)
    _bind_zero_kv_caches(model, device)
    return _compile_or_eager(FullModelPrefillProbe(model), cpu_mode, route_dir)


def _bind_zero_kv_caches(model: nn.Module, device: str) -> None:
    """Bind a zero KV cache of ``NUM_BLOCKS`` blocks to each attention layer."""
    kv_caches = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.attention
        cache_shape = (
            NUM_BLOCKS,
            attn.num_key_value_heads_per_rank,
            BLOCK_SIZE,
            attn.head_dim,
        )
        kv_caches[f"layers.{layer_idx}.self_attn"] = [
            torch.zeros(cache_shape, dtype=torch.bfloat16, device=device),
            torch.zeros(cache_shape, dtype=torch.bfloat16, device=device),
        ]
    model.bind_kv_cache(kv_caches)


# The neuron_config of the logit server that the integration test builds.
# The runner
# copies ``max_logprobs=-1`` into ``max_logprobs``, so the model gathers the
# logits of all ranks.
LOGIT_SERVER_NEURON_CONFIG = {
    "quantization": "bf16",
    "on_device_sampling_config": {"all_greedy": True},
    "num_batched_tokens_buckets": [256],
    "num_seqs_buckets": [1],
}
LOGIT_SERVER_MAX_LOGPROBS = -1


def logit_server_neuron_config():
    """Return the NeuronConfig that the logit server builds for the model.

    Example:
        >>> config = logit_server_neuron_config()
        >>> config.on_device_sampling_config.all_greedy
        True
    """
    from vllm_neuron.model.neuron_config import NeuronConfig

    return NeuronConfig.from_dict(
        {**LOGIT_SERVER_NEURON_CONFIG, "max_logprobs": LOGIT_SERVER_MAX_LOGPROBS}
    )


def load_pretrained_hf_config(checkpoint: str):
    """Load the HF config of the checkpoint as vLLM does (remote code on).

    Example:
        >>> hf_config = load_pretrained_hf_config(resolve_checkpoint())
    """
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)


# The config fields that the model reads. ``rope_scaling`` is not in the list,
# because the model does not read it.
MODEL_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "partial_rotary_factor",
    "use_qk_norm",
    "use_bias",
    "use_qkv_bias",
    "hidden_act",
    "tie_word_embeddings",
    "torch_dtype",
    "first_k_dense_replace",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "num_shared_experts",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "norm_topk_prob",
    "score_function",
    "moe_router_enable_expert_bias",
    "num_nextn_predict_layers",
)


def config_path_differences(checkpoint: str) -> list[str]:
    """Compare the model config of the server path with the raw JSON path.

    The server builds the model from the PretrainedConfig of vLLM. The other
    full-model tests build it from ``config.json``. Return one line for each
    field in ``MODEL_CONFIG_FIELDS`` that has a different value.

    Example:
        >>> assert not config_path_differences(resolve_checkpoint())
    """
    BailingMoeV2Config = _local_src.load("config").BailingMoeV2Config

    server = BailingMoeV2Config.from_configs(load_pretrained_hf_config(checkpoint))
    raw = BailingMoeV2Config.from_configs(read_hf_config(checkpoint))
    return [
        f"{name}: server path {getattr(server, name)!r}, "
        f"config.json path {getattr(raw, name)!r}"
        for name in MODEL_CONFIG_FIELDS
        if getattr(server, name) != getattr(raw, name)
    ]


class ServerForwardProbe(nn.Module):
    """Run ``BailingMoeV2ForCausalLM.forward`` with the inputs of the runner.

    ``FullModelPrefillProbe`` runs the modules one by one, and it does not run
    the sampler. This probe calls the model forward of the server graph: the
    backbone, the LM head, the on-device sampler and the logit all-gather. It
    returns the sampled tokens and the gathered logits in FP32.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, positions, attn_metadata, sampling_positions, sampling_params
    ):
        # The runner gives the stamped TP rank tensor. MPExecutor stamps the
        # same tensor.
        rank = getattr(self.model.tp_group, "rank_in_group_tensor", None)
        sampled_tokens, logits = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
            sampling_params=sampling_params,
            rank=rank,
        )
        return sampled_tokens, logits.float()


def meta_tensor_names(model: nn.Module) -> list[str]:
    """Return the names of the parameters and buffers that are meta tensors.

    Example:
        >>> assert not meta_tensor_names(model)
    """
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return [name for name, tensor in tensors if tensor.is_meta]


def load_like_runner(model_factory, checkpoint: str, device: str) -> nn.Module:
    """Build and load the model with the steps of ``NeuronModelRunner.load_model``.

    The runner calls the factory in a ``torch.device("meta")`` context. Then it
    calls ``load_weights`` and moves the model to the device. A tensor that
    ``__init__`` makes in that context is a meta tensor. The other full-model
    helpers build the model on CPU, so they do not see this condition.

    Example:
        >>> model = load_like_runner(lambda: build(), resolve_checkpoint(), "cpu")
    """
    with torch.device("meta"):
        model = model_factory()
    model.load_weights(checkpoint, torch.device(device), None)
    remaining = meta_tensor_names(model)
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} parameters or buffers are meta tensors after "
            f"load_weights: {remaining[:8]}. Load or build each of them in "
            "load_weights."
        )
    return model.to(device)


def full_model_server_forward_load(
    checkpoint: str,
    cpu_mode: bool,
    device: str,
    route_dir: str | None = None,
    meta_init: bool = False,
) -> nn.Module:
    """Top-level picklable load of the model with the config of the logit server.

    The load uses the same configs as the server: the PretrainedConfig with
    remote code, and ``logit_server_neuron_config``. The model goes through
    the factory, as in the runner. With ``meta_init``, the load also uses the
    meta construction of the runner (see :func:`load_like_runner`).
    """
    BailingMoeV2ForCausalLM = _local_src.load("factory").BailingMoeV2ForCausalLM

    set_mock_vllm_config(ep_enabled=True)

    def build() -> nn.Module:
        return BailingMoeV2ForCausalLM.from_configs(
            load_pretrained_hf_config(checkpoint), logit_server_neuron_config()
        )

    if meta_init:
        model = load_like_runner(build, checkpoint, device)
    else:
        model = build()
        model.load_weights(checkpoint, torch.device(device), None)
    if model.on_device_sampling_config is None or not model._gather_logits:
        raise RuntimeError(
            "The model has no on-device sampler or does not gather the logits. "
            "The probe needs the configuration of the logit server."
        )
    _bind_zero_kv_caches(model, device)
    return _compile_or_eager(ServerForwardProbe(model), cpu_mode, route_dir)


class MetaInitModelProbe(LoadedModelProbe):
    """Return the spot values of ``LoadedModelProbe`` for a runner-style load.

    The first value counts the non-finite floating-point parameters. The
    probe also returns the QK-norm gammas of the MoE layer, because the
    loader applies the RoPE head permutation to them.
    """

    def forward(self, marker: torch.Tensor):
        nonfinite = torch.zeros(1, dtype=torch.int64)
        for _, param in self.model.named_parameters():
            if param.is_floating_point() and not torch.isfinite(param).all():
                nonfinite += 1
        attn = self.model.model.layers[MOE_LAYER].attention
        return (nonfinite, *super().forward(marker)[1:]) + (
            attn.query_layernorm.weight.detach().clone(),
            attn.key_layernorm.weight.detach().clone(),
        )


def full_model_meta_load(checkpoint: str, device: str) -> nn.Module:
    """Top-level picklable load of the full model with the steps of the runner."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    BailingMoeV2ForCausalLM = _local_src.load("factory").BailingMoeV2ForCausalLM

    set_mock_vllm_config(ep_enabled=True)
    model = load_like_runner(
        lambda: BailingMoeV2ForCausalLM.from_configs(
            read_hf_config(checkpoint), NeuronConfig()
        ),
        checkpoint,
        device,
    )
    return MetaInitModelProbe(model)


def full_model_load(checkpoint: str, device: str) -> nn.Module:
    """Top-level picklable load of the full model through ``load_weights``."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    BailingMoeV2ForCausalLM = _local_src.load("factory").BailingMoeV2ForCausalLM

    set_mock_vllm_config(ep_enabled=True)
    model = BailingMoeV2ForCausalLM.from_configs(
        read_hf_config(checkpoint), NeuronConfig()
    )
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.fill_(float("nan"))
    model.load_weights(checkpoint, torch.device(device), None)
    return LoadedModelProbe(model)


# =============================================================================
# Vendored test helpers
# =============================================================================
# The plugin's own test tree holds these two helpers outside this folder
# (``test/vllm_neuron/model/utils.py`` and ``test/vllm_neuron/utils/test_prompts.py``).
# A contrib model must be self-contained, so they are copied here verbatim, as
# contrib/README.md allows for a shared test helper.


class FakeSafeSlice:
    """Minimal stand-in for a safetensors PySafeSlice backed by a torch.Tensor.

    Lets the weight-loader transforms run without a checkpoint file. It
    supports slicing and the shape and dtype queries only.
    """

    __slots__ = ("_t",)

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor

    def __getitem__(self, idx):
        return self._t[idx]

    def get_shape(self):
        return list(self._t.shape)

    def get_dtype(self):
        return str(self._t.dtype).replace("torch.", "")


# The two prompts of the plugin's shared logit-validation prompt list
# (``PROMPTS_10[4]`` and ``PROMPTS_10[8]`` of
# ``test/vllm_neuron/utils/test_prompts.py``). The full-model tests and the
# integration test use the same two prompts.
PROMPTS_2 = [
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning across nine countries and containing approximately ten percent of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, encoding the genetic instructions for building and maintaining a human organism.",
]
