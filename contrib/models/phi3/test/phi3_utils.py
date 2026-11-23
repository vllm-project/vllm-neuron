# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Phi3 (Phi-4-mini-instruct) contrib tests.

This file travels with the model: it is the port's own test helper, vendored
into the contrib folder so that nothing outside ``contrib/models/phi3/`` is
needed. ``test/unit`` and ``test/module`` both use it, and they load it the way
this file loads ``src/``: by file location, under a module name that is unique
to this model. Nothing in this folder needs a ``sys.path`` entry or the
repository root. ``test/_local_src.py`` holds both loaders; ``src_package`` and
``src_module`` below wrap the ``src/`` one.

The helpers in this file have three groups:

- Checkpoint and topology: resolve the pinned checkpoint revision, and get the
  node topology for MPExecutor.
- Worker side: top-level harness classes and load functions. MPExecutor pickles
  each load function through ``functools.partial``, so no closure is used.
- Reference side: HF Phi3 modules and the HF full model. They read the
  checkpoint directly and do not use the port code.

All references use the HF ``transformers`` Phi3 implementation in eager mode.
Each reference runs in FP32 and in BF16 on the same inputs, for a three-way
comparison against the Neuron result.
"""

import functools
import gc
import json
import os

import torch
import torch.nn as nn

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Return ``test/_local_src.py``, loaded by file location.

    A plain ``import _local_src`` would resolve to whichever contrib model
    pytest collected first, so this module (and every test module) loads the
    helper under a name that is unique to this model.
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "contrib_phi3_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_LOCAL_SRC = _load_local_src()

#: The module name that ``src/`` is loaded under; build any string module path
#: (patch target, logger name) from this.
SRC_PACKAGE = _LOCAL_SRC.SRC_PACKAGE


def src_package():
    """Return the ``src`` package of this contrib model."""
    return _LOCAL_SRC.load()


def src_module(name: str):
    """Return a module of the ``src`` package, for example ``"model"``."""
    return _LOCAL_SRC.load(name)


MODEL_ID = "microsoft/Phi-4-mini-instruct"
HF_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"

# TP=4: the 24 Q heads give 6 on each rank and the 8 KV heads give 2 on each
# rank, so each rank holds whole GQA groups, and the 200064 vocabulary divides
# into 4 shards of 50016. This is the size the recorded unit and module runs
# used; see src/README.md, "Parallelism", for the sizes the model supports.
WORLD_SIZE = 4
# The device group uses 2 nodes with 2 TP ranks on each node.
DEVICE_NNODES = 2

BLOCK_SIZE = 32
# The decode kernel needs a block count that is a multiple of LNC * 128.
NUM_BLOCKS = 256
LAYER_IDX = 0
HIDDEN_SEED = 42

# Change this value when the semantics of a reference change. The value is
# part of each golden cache key.
REFERENCE_VERSION = "hf-phi3-eager-v1"

_EXPECTED_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "vocab_size": 200064,
}

# The port read its prompt text from the plugin's shared LongBench prompt
# corpus (``test/vllm_neuron/utils/data/longbench_prompts.json``). That file is
# outside the model folder and is several megabytes, so the text is local here
# and repeats to the requested length. Only the token IDs matter: every
# reference is computed from the same IDs.
_PROMPT_TEXT = (
    "The printing press reached Europe in the middle of the fifteenth "
    "century, and within fifty years a trade that had copied books one page "
    "at a time was producing them by the thousand. Printers settled in the "
    "university towns first, because that is where the readers were, then "
    "followed the rivers and the fairs. A compositor set type from a "
    "manuscript, a pressman inked the forme and pulled the bar, and a boy "
    "hung the damp sheets to dry. The work was loud, slow and precise. What "
    "changed was not the reading but the copying: a scribe produced one book "
    "in a season, a shop produced a shelf of them in a week, and the price "
    "of a page fell until a merchant could own a library. Printers argued "
    "over spelling, invented the title page, numbered their leaves, and "
    "quarrelled with their patrons about money. Censors followed soon after, "
    "which is its own kind of compliment to the machine.\n\n"
)


# =============================================================================
# Checkpoint and topology
# =============================================================================


@functools.lru_cache(maxsize=1)
def resolve_checkpoint() -> str:
    """Return a local directory that holds the Phi-4-mini checkpoint.

    ``PHI3_CHECKPOINT_PATH`` points at a checkpoint that is already on disk.
    Without it, the function downloads the pinned revision from the
    HuggingFace hub.
    """
    local = os.environ.get("PHI3_CHECKPOINT_PATH")
    if local:
        path, provenance = local, f"PHI3_CHECKPOINT_PATH={local}"
    else:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL_ID,
            revision=HF_REVISION,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "tokenizer*",
            ],
        )
        provenance = f"huggingface {MODEL_ID}@{HF_REVISION}"

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    problems = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in _EXPECTED_CONFIG.items()
        if config.get(key) != value
    ]
    if "Phi3ForCausalLM" not in (config.get("architectures") or []):
        problems.append(f"architectures={config.get('architectures')!r}")
    if problems:
        raise RuntimeError(
            f"Checkpoint at {path} is not Phi-4-mini-instruct: {', '.join(problems)}. "
            f"Supply the {MODEL_ID} checkpoint."
        )
    print(f"[phi3] checkpoint: {provenance}, path={path}")
    return path


def topology() -> tuple[int, int, str]:
    """Return (nnodes, node_rank, device) for MPExecutor.

    CPU mode uses one node. The device group uses DEVICE_NNODES nodes. The
    multi-node launcher sets NODE_RANK on each node.
    """
    if envs.VLLM_NEURON_CPU_MODE:
        return 1, 0, "cpu"
    return DEVICE_NNODES, int(os.environ.get("NODE_RANK", "0")), "neuron:0"


def local_global_ranks(nnodes: int, node_rank: int) -> list[int]:
    """Return the global TP ranks of the local MPExecutor workers."""
    per_node = WORLD_SIZE // nnodes
    return [node_rank * per_node + i for i in range(per_node)]


def make_executor(model_load, load_timeout: float = 1800.0) -> MPExecutor:
    """Make an MPExecutor with the Phi3 test topology."""
    nnodes, node_rank, device = topology()
    return MPExecutor(
        world_size=WORLD_SIZE,
        model_load=model_load,
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        load_timeout=load_timeout,
    )


def collect_timeout() -> float:
    """Return the collect timeout. A device call can include a compilation."""
    return 600.0 if envs.VLLM_NEURON_CPU_MODE else 3600.0


def layer_names(num_layers: int) -> list[str]:
    return [f"layers.{i}.self_attn" for i in range(num_layers)]


def prefill_metadata(
    names: list[str],
    start: int,
    length: int,
    kv_segment_size: int | None = None,
) -> dict:
    """Return prefill attention metadata for one sequence.

    Set ``kv_segment_size`` to select segmented prefill. The chunk then attends
    over ``start`` prior tokens in the paged KV cache.
    """
    entry = {
        "max_query_len": length,
        "decode_token_threshold": 1,
        "slot_mapping": torch.arange(start, start + length, dtype=torch.int64),
        "block_size": BLOCK_SIZE,
        "block_table_tensor": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0),
        "max_blocks_per_seq": NUM_BLOCKS,
        "cached_seq_len": (
            torch.tensor([[start]], dtype=torch.int32) if kv_segment_size else None
        ),
        "kv_segment_size": kv_segment_size,
    }
    return {name: dict(entry) for name in names}


def decode_metadata(names: list[str], position: int) -> dict:
    """Return decode attention metadata for one sequence and one token."""
    entry = {
        "max_query_len": 1,
        "decode_token_threshold": 1,
        "slot_mapping": torch.tensor([position], dtype=torch.int64),
        "block_size": BLOCK_SIZE,
        "block_table_tensor": torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0),
        "max_blocks_per_seq": NUM_BLOCKS,
    }
    return {name: dict(entry) for name in names}


def random_hidden(num_tokens: int, hidden_size: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_tokens, hidden_size, dtype=torch.float32).to(torch.bfloat16)


def natural_prompt_ids(tokenizer, num_tokens: int, offset: int = 0) -> list[int]:
    """Return ``num_tokens`` token IDs of natural text, from ``offset``."""
    ids = tokenizer(_PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("The tokenizer returned no tokens for the prompt text.")
    while len(ids) < offset + num_tokens:
        ids = ids + ids
    return list(ids[offset : offset + num_tokens])


# =============================================================================
# Worker side: harnesses and load functions
# =============================================================================


def _worker_device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def _maybe_compile(module: nn.Module) -> nn.Module:
    if envs.VLLM_NEURON_CPU_MODE:
        return module
    return torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def _port_config(checkpoint: str, neuron_config=None):
    from vllm_neuron.model.neuron_config import NeuronConfig

    Phi3Config = src_module("config").Phi3Config

    return Phi3Config.from_configs(
        os.path.join(checkpoint, "config.json"), neuron_config or NeuronConfig()
    )


def _load_sharded(module: nn.Module, checkpoint: str, mappings: dict, device: str):
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    ckpt = SafetensorsCheckpoint(checkpoint)
    if envs.VLLM_NEURON_CPU_MODE:
        result = ckpt.load_sharded(
            tp_group.rank_in_group, tp_group.world_size, module, mappings, device
        )
    else:
        result = ckpt.load_sharded_pipelined(
            tp_group.rank_in_group, tp_group.world_size, module, mappings, device
        )
    module.load_state_dict(result.state_dict, strict=False, assign=True)
    return result


def _empty_kv_cache(num_kv_heads: int, head_dim: int, device: str) -> list:
    shape = (NUM_BLOCKS, num_kv_heads, BLOCK_SIZE, head_dim)
    return [
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
    ]


class AttentionHarness(nn.Module):
    """Run one Phi3Attention layer with the model RoPE and the SP row split.

    In prefill each TP rank receives its own 1 / world_size of the rows, as
    after the embedding scatter in the full model.
    """

    def __init__(self, attn: nn.Module, rotary: nn.Module):
        super().__init__()
        self.attn = attn
        self.rotary = rotary

    def forward(self, hidden_states, positions, attn_metadata):
        layer_name = f"layers.{self.attn.layer_idx}.self_attn"
        is_prefill = (
            attn_metadata[layer_name]["max_query_len"]
            > attn_metadata[layer_name]["decode_token_threshold"]
        )
        positions = positions.to(torch.int32)
        cos, sin = self.rotary(
            positions, device=hidden_states.device, dtype=self.attn.dtype
        )
        if is_prefill and self.attn.world_size > 1:
            rows = hidden_states.shape[0] // self.attn.world_size
            start = self.attn.rank * rows
            hidden_states = hidden_states[start : start + rows]
        return self.attn(hidden_states, positions, (cos, sin), attn_metadata)


class MLPHarness(nn.Module):
    """Run one Phi3MLP layer with the SP row split in prefill."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states, is_prefill: bool):
        if is_prefill and self.mlp.world_size > 1:
            rows = hidden_states.shape[0] // self.mlp.world_size
            start = self.mlp.rank * rows
            hidden_states = hidden_states[start : start + rows]
        return self.mlp(hidden_states, is_prefill=is_prefill)


def load_attention(checkpoint: str, layer_idx: int) -> nn.Module:
    """MPExecutor load function for one attention layer."""
    model_module = src_module("model")
    Phi3Attention = model_module.Phi3Attention
    Phi3RotaryEmbedding = model_module.Phi3RotaryEmbedding

    device = _worker_device()
    config = _port_config(checkpoint)
    attn = Phi3Attention(config, layer_idx=layer_idx)
    prefix = f"model.layers.{layer_idx}.self_attn"
    _load_sharded(
        attn,
        checkpoint,
        {
            "qkv_proj_weight": f"{prefix}.qkv_proj.weight",
            "o_proj_weight": f"{prefix}.o_proj.weight",
        },
        device,
    )
    attn.k_cache, attn.v_cache = _empty_kv_cache(
        attn.num_key_value_heads_per_rank, attn.head_dim, device
    )
    # The rotary module holds its frequencies as CPU buffers. Move them to the
    # device, as the serving runner does.
    rotary = Phi3RotaryEmbedding(config).to(device)
    return _maybe_compile(AttentionHarness(attn, rotary))


def load_mlp(checkpoint: str, layer_idx: int) -> nn.Module:
    """MPExecutor load function for one MLP layer."""
    Phi3MLP = src_module("model").Phi3MLP

    device = _worker_device()
    config = _port_config(checkpoint)
    mlp = Phi3MLP(config)
    prefix = f"model.layers.{layer_idx}.mlp"
    _load_sharded(
        mlp,
        checkpoint,
        {
            "gate_proj_weight": f"{prefix}.gate_up_proj.weight",
            "up_proj_weight": f"{prefix}.gate_up_proj.weight",
            "down_proj_weight": f"{prefix}.down_proj.weight",
        },
        device,
    )
    return _maybe_compile(MLPHarness(mlp))


def load_full_model(checkpoint: str) -> nn.Module:
    """MPExecutor load function for the full model with CPU sampling.

    The model comes from the registered factory, so the factory validation
    also runs. All 32 layers and the original BF16 weights load.
    """
    return _maybe_compile(_build_full_model(checkpoint))


def _build_full_model(checkpoint: str) -> nn.Module:
    from vllm_neuron.model.neuron_config import NeuronConfig

    Phi3ForCausalLM = src_package().Phi3ForCausalLM

    device = _worker_device()
    model = Phi3ForCausalLM.from_configs(
        os.path.join(checkpoint, "config.json"),
        NeuronConfig(on_device_sampling_config=None),
    )
    if envs.VLLM_NEURON_CPU_MODE:
        _load_sharded(model, checkpoint, model.checkpoint_mappings(), device)
    else:
        model.load_weights(checkpoint, torch.device(device), None)
    # Move the buffers (the RoPE frequencies) to the device, as the serving
    # runner does after load_weights.
    model = model.to(device)

    caches = {}
    for name, layer in zip(layer_names(len(model.model.layers)), model.model.layers):
        caches[name] = _empty_kv_cache(
            layer.self_attn.num_key_value_heads_per_rank,
            layer.self_attn.head_dim,
            device,
        )
    model.bind_kv_cache(caches)
    return model


def probe_stage_names(num_layers: int) -> list[str]:
    """Return the stage names of FullModelProbe, in output row order."""
    names = ["embed_tokens"]
    for i in range(num_layers):
        names += [f"layers.{i}.attn_residual", f"layers.{i}.mlp_residual"]
    return names + ["norm", "lm_head"]


def _stage_stats(x: torch.Tensor) -> torch.Tensor:
    # Return [count of values that are not finite, absmax of the finite values].
    xf = x.float()
    finite = torch.isfinite(xf)
    absmax = torch.where(finite, xf.abs(), torch.zeros_like(xf)).amax()
    return torch.stack([(~finite).sum().float(), absmax])


class FullModelProbe(nn.Module):
    """Run the Phi3ForCausalLM prefill and record statistics after each stage.

    The probe calls the submodules of the model in the order of
    ``Phi3Model.forward`` and ``Phi3DecoderLayer.forward``. It uses the same
    inputs as the serving runner, which includes the ``rank`` tensor. The
    output is (stats, logits). Row i of stats is
    [count of values that are not finite, finite absmax] for stage i of
    ``probe_stage_names``. The stats of the decoder stages are for the SP
    rows of this rank.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        backbone = self.model.model
        positions = positions.to(torch.int32)
        stats = []

        hidden = backbone.embed_tokens(input_ids, scatter_tokens=True, rank=rank)
        stats.append(_stage_stats(hidden))
        position_embeddings = backbone.rotary_emb(
            positions, device=hidden.device, dtype=hidden.dtype
        )

        for layer in backbone.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden = layer.self_attn(
                hidden_states=hidden,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
            hidden = residual + hidden
            stats.append(_stage_stats(hidden))

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden, is_prefill=True)
            hidden = residual + hidden
            stats.append(_stage_stats(hidden))

        hidden = backbone.norm(hidden)
        stats.append(_stage_stats(hidden))
        if backbone.world_size > 1:
            hidden = backbone.tp_group.all_gather(hidden, dim=0)

        hidden = torch.index_select(hidden, dim=0, index=sampling_positions)
        logits = self.model.lm_head(hidden)
        stats.append(_stage_stats(logits))
        return torch.stack(stats), logits


def load_full_model_probe(checkpoint: str) -> nn.Module:
    """MPExecutor load function for FullModelProbe on the full model."""
    return _maybe_compile(FullModelProbe(_build_full_model(checkpoint)))


# The serving decode bucket is 2048 tokens, so a request owns at most 64 blocks.
SERVING_BLOCK_TABLE_WIDTH = 2048 // BLOCK_SIZE
# The serving runner never gives block 0 (the null block) to a request.
FIRST_REQUEST_BLOCK = 1


def serving_prefill_inputs(
    names: list[str], prompt_ids: list[int], padded_len: int, kv_segment_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (input_ids, positions, metadata) as the serving runner makes them.

    The runner pads the prompt to the prefill bucket. A pad token has ID 0, the
    position of the last real token, and slot 0 (the null block). The device
    block table holds the request blocks, then -1 (PAD_SLOT_ID) in each entry
    that the request does not own. The prefill uses the segmented branch, as
    with prefix caching on.
    """
    num_real = len(prompt_ids)
    assert 0 < num_real <= padded_len
    num_pad = padded_len - num_real
    input_ids = torch.tensor(list(prompt_ids) + [0] * num_pad, dtype=torch.int64)
    positions = torch.cat(
        (
            torch.arange(num_real, dtype=torch.int64),
            torch.full((num_pad,), num_real - 1, dtype=torch.int64),
        )
    )
    first_slot = FIRST_REQUEST_BLOCK * BLOCK_SIZE
    slot_mapping = torch.cat(
        (
            torch.arange(first_slot, first_slot + num_real, dtype=torch.int64),
            torch.zeros(num_pad, dtype=torch.int64),
        )
    )
    num_owned = -(-num_real // BLOCK_SIZE)
    assert FIRST_REQUEST_BLOCK + num_owned <= NUM_BLOCKS
    block_table = torch.full((1, SERVING_BLOCK_TABLE_WIDTH), -1, dtype=torch.int32)
    block_table[0, :num_owned] = torch.arange(
        FIRST_REQUEST_BLOCK, FIRST_REQUEST_BLOCK + num_owned, dtype=torch.int32
    )
    entry = {
        "max_query_len": padded_len,
        "decode_token_threshold": 1,
        "slot_mapping": slot_mapping,
        "block_size": BLOCK_SIZE,
        "block_table_tensor": block_table,
        "max_blocks_per_seq": SERVING_BLOCK_TABLE_WIDTH,
        "cached_seq_len": torch.tensor([[0]], dtype=torch.int32),
        "kv_segment_size": kv_segment_size,
    }
    return input_ids, positions, {name: dict(entry) for name in names}


class CoverageReport(nn.Module):
    """Return the weight-coverage statistics of one rank."""

    def __init__(self, stats: torch.Tensor):
        super().__init__()
        self.register_buffer("stats", stats)

    def forward(self, _unused):
        return self.stats.clone()


def load_weight_coverage(checkpoint: str) -> nn.Module:
    """MPExecutor load function that checks the full-model weight coverage.

    The function raises with the parameter names when a check fails:

    - Each model parameter has a checkpoint tensor (strict load).
    - The model references each checkpoint tensor (no unexpected key).
    - Each loaded tensor has the parameter shape and the BF16 dtype.
    - Each loaded tensor is finite.
    - No parameter or buffer stays on the meta device.
    - Each buffer (the RoPE frequencies) is finite.
    - The tied LM head shard is equal to the embedding shard.

    The returned stats are [parameters, loaded tensors, layers, tied equal, rank].
    """
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_neuron.model.neuron_config import NeuronConfig

    Phi3ForCausalLM = src_package().Phi3ForCausalLM

    # Build the model on the meta device, as the serving runner does. A tensor
    # that the load does not replace then stays on the meta device.
    with torch.device("meta"):
        model = Phi3ForCausalLM.from_configs(
            os.path.join(checkpoint, "config.json"),
            NeuronConfig(on_device_sampling_config=None),
        )
    params = dict(model.named_parameters())
    result = _load_sharded(model, checkpoint, model.checkpoint_mappings(), "cpu")

    problems = []
    if result.missing_keys:
        problems.append(f"missing checkpoint keys for {sorted(result.missing_keys)}")
    if result.unexpected_keys:
        problems.append(f"unexpected checkpoint keys {sorted(result.unexpected_keys)}")
    not_loaded = sorted(set(params) - set(result.state_dict))
    if not_loaded:
        problems.append(f"parameters without a loaded tensor {not_loaded}")
    for name, tensor in result.state_dict.items():
        if tuple(tensor.shape) != tuple(params[name].shape):
            problems.append(
                f"{name} shape {tuple(tensor.shape)} != {tuple(params[name].shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            problems.append(f"{name} dtype {tensor.dtype} is not bfloat16")
        if not torch.isfinite(tensor.float()).all():
            problems.append(f"{name} has a value that is not finite")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            problems.append(f"{name} is on the meta device")
    for name, tensor in model.named_buffers():
        if not tensor.is_meta and not torch.isfinite(tensor.float()).all():
            problems.append(f"buffer {name} has a value that is not finite")

    tied = model.config.tie_word_embeddings and torch.equal(
        model.lm_head.weight, model.model.embed_tokens.weight
    )
    if model.config.tie_word_embeddings and not tied:
        problems.append("lm_head.weight shard is not equal to the embedding shard")
    if problems:
        raise RuntimeError(
            "Phi3 weight coverage check failed: " + "; ".join(problems[:50])
        )

    stats = torch.tensor(
        [
            len(params),
            len(result.state_dict),
            len(model.model.layers),
            int(tied),
            get_tp_group().rank_in_group,
        ],
        dtype=torch.int64,
    )
    return CoverageReport(stats)


# =============================================================================
# Reference side: HF Phi3 (independent of the port code)
# =============================================================================


def read_checkpoint_tensors(checkpoint: str, keys: list[str]) -> dict:
    """Read checkpoint tensors by name with safetensors."""
    from safetensors import safe_open

    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {key: "model.safetensors" for key in keys}

    by_file = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for file_name, file_keys in by_file.items():
        with safe_open(os.path.join(checkpoint, file_name), framework="pt") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key).clone()
    return tensors


def hf_config(checkpoint: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    config._attn_implementation = "eager"
    return config


def _causal_mask(num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.triu(
        torch.full((num_tokens, num_tokens), torch.finfo(torch.float32).min),
        diagonal=1,
    )
    return mask[None, None].to(dtype)


def hf_attention_reference(
    checkpoint: str, layer_idx: int, hidden: torch.Tensor
) -> dict:
    """Run HF Phi3Attention causally over all rows of ``hidden``.

    Returns {"fp32": [T, H], "bf16": [T, H]}. Row t is the output of the token
    at position t. This equals a prefill of rows [0, P) and then a decode of
    each later row.
    """
    from transformers.models.phi3.modeling_phi3 import (
        Phi3Attention,
        Phi3RotaryEmbedding,
    )

    config = hf_config(checkpoint)
    prefix = f"model.layers.{layer_idx}.self_attn"
    weights = read_checkpoint_tensors(
        checkpoint, [f"{prefix}.qkv_proj.weight", f"{prefix}.o_proj.weight"]
    )
    num_tokens = hidden.shape[0]
    position_ids = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(0)

    out = {}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        attn = Phi3Attention(config, layer_idx=layer_idx)
        attn.load_state_dict(
            {
                "qkv_proj.weight": weights[f"{prefix}.qkv_proj.weight"],
                "o_proj.weight": weights[f"{prefix}.o_proj.weight"],
            },
            strict=True,
        )
        attn = attn.to(dtype).eval()
        rotary = Phi3RotaryEmbedding(config)
        x = hidden.to(dtype).unsqueeze(0)
        with torch.no_grad():
            cos, sin = rotary(x, position_ids)
            result, _ = attn(
                x,
                position_embeddings=(cos, sin),
                attention_mask=_causal_mask(num_tokens, dtype),
            )
        out[label] = result[0].detach()
    return out


def hf_mlp_reference(checkpoint: str, layer_idx: int, hidden: torch.Tensor) -> dict:
    """Run HF Phi3MLP. Returns {"fp32": [T, H], "bf16": [T, H]}."""
    from transformers.models.phi3.modeling_phi3 import Phi3MLP

    config = hf_config(checkpoint)
    prefix = f"model.layers.{layer_idx}.mlp"
    weights = read_checkpoint_tensors(
        checkpoint, [f"{prefix}.gate_up_proj.weight", f"{prefix}.down_proj.weight"]
    )
    out = {}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        mlp = Phi3MLP(config)
        mlp.load_state_dict(
            {
                "gate_up_proj.weight": weights[f"{prefix}.gate_up_proj.weight"],
                "down_proj.weight": weights[f"{prefix}.down_proj.weight"],
            },
            strict=True,
        )
        mlp = mlp.to(dtype).eval()
        with torch.no_grad():
            out[label] = mlp(hidden.to(dtype)).detach()
    return out


def hf_full_model_reference(
    checkpoint: str, prompt_ids: list[int], num_decode: int
) -> dict:
    """Compute the HF full-model logits on one FP32 teacher-forced history.

    The FP32 model makes ``num_decode`` greedy tokens after the prompt. Then
    the FP32 model and the BF16 model each run once over the prompt and these
    tokens. Row t of each logits tensor predicts token t + 1.

    Returns {"token_ids": [P + D], "fp32": [P + D, V], "bf16": [P + D, V]}.
    """
    from transformers import AutoModelForCausalLM

    ids = list(prompt_ids)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        for _ in range(num_decode):
            logits = model(torch.tensor([ids])).logits[0, -1]
            ids.append(int(logits.argmax()))
        fp32_logits = model(torch.tensor([ids])).logits[0].detach()
    del model
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        bf16_logits = model(torch.tensor([ids])).logits[0].detach()
    del model
    gc.collect()
    return {
        "token_ids": torch.tensor(ids, dtype=torch.int64),
        "fp32": fp32_logits,
        "bf16": bf16_logits,
    }


def hf_full_model_rows(checkpoint: str, prompt_ids: list[int], rows: list[int]) -> dict:
    """Return the HF prefill logits of the prompt at ``rows`` only.

    Returns {"fp32": [len(rows), V], "bf16": [len(rows), V]}. The golden cache
    keeps only these rows, because all rows of a 1024-token prompt are large.
    """
    full = hf_full_model_reference(checkpoint, prompt_ids, num_decode=0)
    index = torch.tensor(rows, dtype=torch.int64)
    return {
        "fp32": full["fp32"].index_select(0, index).contiguous(),
        "bf16": full["bf16"].index_select(0, index).contiguous(),
    }


def golden_key(test: str, **fields) -> dict:
    """Return a golden cache key with the model, revision and reference version."""
    return {
        "test": test,
        "model_id": MODEL_ID,
        "revision": HF_REVISION,
        "reference": REFERENCE_VERSION,
        **fields,
    }


# =============================================================================
# Shared component checks (CPU group and device group)
# =============================================================================


def check_attention(
    checkpoint: str,
    chunk_len: int,
    num_chunks: int,
    num_decode: int,
    segmented: bool,
    name: str,
) -> None:
    """Compare Phi3Attention prefill and decode with HF FP32 and BF16.

    The test runs ``num_chunks`` prefill chunks of ``chunk_len`` tokens, then
    ``num_decode`` decode tokens, on one paged KV cache. Set ``segmented`` to
    use segmented prefill (the APC path) in place of flash attention.
    """
    from transformers import AutoConfig
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    hidden_size = AutoConfig.from_pretrained(checkpoint).hidden_size
    prefill_len = chunk_len * num_chunks
    total = prefill_len + num_decode
    assert total <= NUM_BLOCKS * BLOCK_SIZE
    assert chunk_len % WORLD_SIZE == 0

    hidden = random_hidden(total, hidden_size, HIDDEN_SEED)
    positions = torch.arange(total, dtype=torch.int64)
    goldens = get_or_compute_goldens(
        key_config=golden_key(
            "attention", layer_idx=LAYER_IDX, num_tokens=total, seed=HIDDEN_SEED
        ),
        compute_fn=lambda: hf_attention_reference(checkpoint, LAYER_IDX, hidden),
    )

    nnodes, node_rank, _ = topology()
    ranks = local_global_ranks(nnodes, node_rank)
    names = layer_names(LAYER_IDX + 1)[LAYER_IDX:]
    executor = make_executor(functools.partial(load_attention, checkpoint, LAYER_IDX))

    rows = chunk_len // WORLD_SIZE
    pre_actual, pre_fp32, pre_bf16 = [], [], []
    for chunk in range(num_chunks):
        start = chunk * chunk_len
        metadata = prefill_metadata(
            names, start, chunk_len, kv_segment_size=chunk_len if segmented else None
        )
        executor.dispatch(
            hidden[start : start + chunk_len],
            positions[start : start + chunk_len],
            metadata,
        )
        outputs = executor.collect(timeout=collect_timeout())
        for out, rank in zip(outputs, ranks):
            lo = start + rank * rows
            assert tuple(out.shape) == (rows, hidden_size), out.shape
            pre_actual.append(out)
            pre_fp32.append(goldens["fp32"][lo : lo + rows])
            pre_bf16.append(goldens["bf16"][lo : lo + rows])
    assert_close_three_way(
        pre_fp32, pre_bf16, pre_actual, name=f"{name}/prefill/node{node_rank}"
    )

    dec_actual, dec_fp32, dec_bf16 = [], [], []
    for step in range(num_decode):
        pos = prefill_len + step
        executor.dispatch(
            hidden[pos : pos + 1], positions[pos : pos + 1], decode_metadata(names, pos)
        )
        outputs = executor.collect(timeout=collect_timeout())
        for out in outputs:
            assert tuple(out.shape) == (1, hidden_size), out.shape
            dec_actual.append(out)
            dec_fp32.append(goldens["fp32"][pos : pos + 1])
            dec_bf16.append(goldens["bf16"][pos : pos + 1])
    if num_decode:
        assert_close_three_way(
            dec_fp32, dec_bf16, dec_actual, name=f"{name}/decode/node{node_rank}"
        )


def check_mlp(checkpoint: str, prefill_len: int, decode_tokens: int, name: str):
    """Compare Phi3MLP prefill and decode with HF FP32 and BF16."""
    from transformers import AutoConfig
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    hidden_size = AutoConfig.from_pretrained(checkpoint).hidden_size
    hidden_pre = random_hidden(prefill_len, hidden_size, HIDDEN_SEED + 1)
    hidden_dec = random_hidden(decode_tokens, hidden_size, HIDDEN_SEED + 2)
    gold_pre = get_or_compute_goldens(
        key_config=golden_key(
            "mlp", layer_idx=LAYER_IDX, num_tokens=prefill_len, seed=HIDDEN_SEED + 1
        ),
        compute_fn=lambda: hf_mlp_reference(checkpoint, LAYER_IDX, hidden_pre),
    )
    gold_dec = get_or_compute_goldens(
        key_config=golden_key(
            "mlp", layer_idx=LAYER_IDX, num_tokens=decode_tokens, seed=HIDDEN_SEED + 2
        ),
        compute_fn=lambda: hf_mlp_reference(checkpoint, LAYER_IDX, hidden_dec),
    )

    nnodes, node_rank, _ = topology()
    ranks = local_global_ranks(nnodes, node_rank)
    executor = make_executor(functools.partial(load_mlp, checkpoint, LAYER_IDX))

    rows = prefill_len // WORLD_SIZE
    executor.dispatch(hidden_pre, True)
    outputs = executor.collect(timeout=collect_timeout())
    actual, fp32, bf16 = [], [], []
    for out, rank in zip(outputs, ranks):
        assert tuple(out.shape) == (rows, hidden_size), out.shape
        actual.append(out)
        fp32.append(gold_pre["fp32"][rank * rows : (rank + 1) * rows])
        bf16.append(gold_pre["bf16"][rank * rows : (rank + 1) * rows])
    assert_close_three_way(fp32, bf16, actual, name=f"{name}/prefill/node{node_rank}")

    executor.dispatch(hidden_dec, False)
    outputs = executor.collect(timeout=collect_timeout())
    for out in outputs:
        assert tuple(out.shape) == (decode_tokens, hidden_size), out.shape
    assert_close_three_way(
        [gold_dec["fp32"]] * len(outputs),
        [gold_dec["bf16"]] * len(outputs),
        list(outputs),
        name=f"{name}/decode/node{node_rank}",
    )


def _first_bad_stage(stats: torch.Tensor, stage_names: list[str]) -> str | None:
    for i, (nonfinite, _) in enumerate(stats.tolist()):
        if nonfinite > 0:
            previous = (
                f"{stage_names[i - 1]} absmax {stats[i - 1, 1].item():.4g}"
                if i
                else "no previous stage"
            )
            return (
                f"{stage_names[i]} has {int(nonfinite)} values that are not finite "
                f"({previous})"
            )
    return None


def check_full_model_prefill(
    checkpoint: str,
    cases: dict[str, tuple[list[int], list[int]]],
    padded_len: int,
    name: str,
) -> None:
    """Compare the full-model device prefill with HF FP32 and BF16.

    Each case is (prompt_ids, sampled_rows). The probe pads each prompt to
    ``padded_len`` with the serving runner layout (``serving_prefill_inputs``)
    and runs all cases on one MPExecutor. Then the function makes these
    checks for each case:

    - Each stage of FullModelProbe is finite on each rank. The message gives
      the first stage that is not finite.
    - The logits at the sampled rows are close to HF (three-way).

    All cases run before the first assertion, so one failure does not hide
    the result of a different case.
    """
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    num_layers = _EXPECTED_CONFIG["num_hidden_layers"]
    names = layer_names(num_layers)
    stage_names = probe_stage_names(num_layers)
    nnodes, node_rank, _ = topology()
    ranks = local_global_ranks(nnodes, node_rank)

    goldens = {}
    for case, (prompt_ids, rows) in cases.items():
        goldens[case] = get_or_compute_goldens(
            key_config=golden_key("full_model_rows", prompt_ids=prompt_ids, rows=rows),
            compute_fn=functools.partial(
                hf_full_model_rows, checkpoint, prompt_ids, rows
            ),
        )

    executor = make_executor(
        functools.partial(load_full_model_probe, checkpoint), load_timeout=1800.0
    )
    results = {}
    for case, (prompt_ids, rows) in cases.items():
        input_ids, positions, metadata = serving_prefill_inputs(
            names, prompt_ids, padded_len, kv_segment_size=padded_len
        )
        sampling_positions = torch.tensor(rows, dtype=torch.int64)
        executor.dispatch_per_rank(
            [
                (
                    input_ids,
                    positions,
                    metadata,
                    sampling_positions,
                    torch.tensor(rank, dtype=torch.int32),
                )
                for rank in ranks
            ]
        )
        results[case] = executor.collect(timeout=collect_timeout())

    problems = []
    for case, outputs in results.items():
        for (stats, logits), rank in zip(outputs, ranks):
            print(
                f"[phi3] {name}/{case} rank {rank} stage absmax: "
                + ", ".join(
                    f"{stage}={value:.4g}"
                    for stage, value in zip(stage_names, stats[:, 1].tolist())
                )
            )
            bad = _first_bad_stage(stats, stage_names)
            if bad:
                problems.append(f"{case} rank {rank}: {bad}")
    assert not problems, (
        "Phi3 full-model prefill has values that are not finite: " + "; ".join(problems)
    )

    for case, outputs in results.items():
        rows = cases[case][1]
        for _, logits in outputs:
            assert tuple(logits.shape) == tuple(goldens[case]["fp32"].shape), (
                logits.shape
            )
        assert_close_three_way(
            [goldens[case]["fp32"]] * len(outputs),
            [goldens[case]["bf16"]] * len(outputs),
            [logits for _, logits in outputs],
            name=f"{name}/{case}/rows{rows}/node{node_rank}",
        )
