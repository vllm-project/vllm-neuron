# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Ouro tests: checkpoints, harnesses and metadata.

Checkpoints
-----------
* The pinned ``ByteDance/Ouro-1.4B`` revision. ``OURO_CHECKPOINT_PATH`` points
  at a local copy. ``resolve_ouro_checkpoint()`` downloads the pinned revision
  when the variable is not set (2.9 GB). ``real_checkpoint_or_skip()`` skips
  the test instead, so that the CPU unit tests do not download the weights.
  Each resolved checkpoint gets an identity check, so that a different
  revision or a different model cannot give a false pass.
* ``make_tiny_checkpoint()`` writes a random-weight Ouro checkpoint with the
  real architecture (sandwich norms, 4 UT steps, early-exit gate) at a small
  size. The CPU unit tests use it.

Harnesses
---------
Each ``load_*`` function is a top-level function. A test binds its arguments
with ``functools.partial``, so the worker pool can pickle it and reuse its
workers. The functions run inside an MPExecutor worker, where the TP group
exists.

Each harness wraps one real Ouro module. The harness keeps the KV cache of
each UT step, and returns the tensors that the test compares with the
independent reference in ``ouro_reference.py``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron import envs

from _local_src import src_module, src_package

# The early-exit gate key prefix in the checkpoint.
GATE_PREFIX = "model.early_exit_gate"

NORM_NAMES = (
    "input_layernorm",
    "input_layernorm_2",
    "post_attention_layernorm",
    "post_attention_layernorm_2",
)

ARCHITECTURE = "OuroForCausalLM"


# -----------------------------------------------------------------------------
# The pinned checkpoint
# -----------------------------------------------------------------------------

MODEL_ID = "ByteDance/Ouro-1.4B"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
CHECKPOINT_ENV = "OURO_CHECKPOINT_PATH"

# Identity of the pinned revision.
SAFETENSORS_FILE = "model.safetensors"
SAFETENSORS_SIZE = 2869336434
EXPECTED_CONFIG = {
    "model_type": "ouro",
    "architectures": [ARCHITECTURE],
    "num_hidden_layers": 24,
    "total_ut_steps": 4,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "vocab_size": 49152,
    "tie_word_embeddings": False,
    "early_exit_threshold": 1.0,
}


def verify_checkpoint_identity(path: str, check_weights: bool = True) -> dict:
    """Check that ``path`` holds the pinned Ouro-1.4B revision.

    Returns the raw ``config.json`` as a dict. Raises AssertionError with the
    first difference. ``check_weights=False`` checks only ``config.json``.
    """
    root = Path(path)
    config_path = root / "config.json"
    assert config_path.is_file(), (
        f"Checkpoint {path} has no config.json. Set {CHECKPOINT_ENV} to a "
        f"local copy of {MODEL_ID} at revision {MODEL_REVISION}."
    )
    with open(config_path) as f:
        config = json.load(f)
    for key, expected in EXPECTED_CONFIG.items():
        assert config.get(key) == expected, (
            f"config.json field {key} is {config.get(key)!r}, not {expected!r}. "
            f"Use {MODEL_ID} at revision {MODEL_REVISION}."
        )
    if check_weights:
        weights = root / SAFETENSORS_FILE
        assert weights.is_file(), f"Checkpoint {path} has no {SAFETENSORS_FILE}."
        size = weights.stat().st_size
        assert size == SAFETENSORS_SIZE, (
            f"{SAFETENSORS_FILE} has {size} bytes, not {SAFETENSORS_SIZE}. "
            f"Use {MODEL_ID} at revision {MODEL_REVISION}."
        )
    return config


def resolve_ouro_checkpoint() -> str:
    """Return the local directory of the pinned Ouro-1.4B checkpoint.

    Use ``OURO_CHECKPOINT_PATH`` first. If it is not set, download the pinned
    revision from the Hugging Face Hub. Check the identity in both cases.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    verify_checkpoint_identity(path)
    return path


def real_checkpoint_or_skip() -> str:
    """Return ``OURO_CHECKPOINT_PATH`` after the identity check, or skip."""
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        pytest.skip(
            f"Set {CHECKPOINT_ENV} to a local copy of {MODEL_ID} at revision "
            f"{MODEL_REVISION} to run the tests with the real weights."
        )
    verify_checkpoint_identity(path)
    return path


def pinned_config_dir() -> str:
    """Return a directory with the ``config.json`` of the pinned revision.

    Use ``OURO_CHECKPOINT_PATH`` if it is set. Otherwise download only
    ``config.json`` (1.5 kB), not the weights.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        from huggingface_hub import hf_hub_download

        path = os.path.dirname(
            hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
        )
    verify_checkpoint_identity(path, check_weights=False)
    return path


# -----------------------------------------------------------------------------
# Tiny random-weight checkpoint
# -----------------------------------------------------------------------------

# The Ouro-1.4B architecture at a small size. 4 Q/KV heads and an
# intermediate size of 512 divide by TP 2. The vocabulary of 512 divides by
# TP 2, as the model requires.
TINY_CONFIG = {
    "architectures": [ARCHITECTURE],
    "model_type": "ouro",
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "sliding_window": None,
    "layer_types": ["full_attention", "full_attention"],
    "torch_dtype": "bfloat16",
    "total_ut_steps": 4,
    "early_exit_threshold": 1.0,
    "bos_token_id": 0,
    "eos_token_id": 0,
}


def make_tiny_checkpoint(model_dir: str, seed: int = 42) -> str:
    """Write a random-weight BF16 checkpoint with ``TINY_CONFIG`` to ``model_dir``.

    The norm weights are ``1 + 0.1 * randn`` instead of ones, so that a norm
    that reads the wrong weight changes the output. The weight scales keep the
    activations and the logits at a scale where a wrong result shows.
    """
    from safetensors.torch import save_file

    cfg = TINY_CONFIG
    H, inter = cfg["hidden_size"], cfg["intermediate_size"]
    D, Nh, Nkv = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, std):
        return (torch.randn(*shape, generator=gen) * std).to(torch.bfloat16)

    def norm_weight():
        return (1.0 + 0.1 * torch.randn(H, generator=gen)).to(torch.bfloat16)

    tensors = {
        "model.embed_tokens.weight": randn(V, H, std=1.0),
        "model.norm.weight": norm_weight(),
        "lm_head.weight": randn(V, H, std=0.1),
        f"{GATE_PREFIX}.weight": randn(1, H, std=0.02),
        f"{GATE_PREFIX}.bias": randn(1, std=0.1),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = randn(Nh * D, H, std=H**-0.5)
        tensors[f"{p}.self_attn.k_proj.weight"] = randn(Nkv * D, H, std=H**-0.5)
        tensors[f"{p}.self_attn.v_proj.weight"] = randn(Nkv * D, H, std=H**-0.5)
        tensors[f"{p}.self_attn.o_proj.weight"] = randn(H, Nh * D, std=(Nh * D) ** -0.5)
        tensors[f"{p}.mlp.gate_proj.weight"] = randn(inter, H, std=H**-0.5)
        tensors[f"{p}.mlp.up_proj.weight"] = randn(inter, H, std=H**-0.5)
        tensors[f"{p}.mlp.down_proj.weight"] = randn(H, inter, std=inter**-0.5)
        for norm in NORM_NAMES:
            tensors[f"{p}.{norm}.weight"] = norm_weight()

    os.makedirs(model_dir, exist_ok=True)
    save_file(
        tensors, os.path.join(model_dir, SAFETENSORS_FILE), metadata={"format": "pt"}
    )
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return model_dir


# -----------------------------------------------------------------------------
# Configuration and weight mappings
# -----------------------------------------------------------------------------


def ouro_config(model_checkpoint: str):
    """Return the Neuron ``OuroConfig`` of the checkpoint."""
    from transformers import AutoConfig

    from vllm_neuron.model.neuron_config import NeuronConfig

    # Importing src/ registers the "ouro" model type with AutoConfig.
    src_package()
    hf_config = AutoConfig.from_pretrained(model_checkpoint)
    return src_module("config").OuroConfig.from_configs(hf_config, NeuronConfig())


def attention_mappings(layer_idx: int, prefix: str = "") -> dict:
    """Return the checkpoint mappings of one OuroAttention module."""
    p = f"model.layers.{layer_idx}.self_attn"
    return {
        f"{prefix}qkv_proj_weight": [
            f"{p}.q_proj.weight",
            f"{p}.k_proj.weight",
            f"{p}.v_proj.weight",
        ],
        f"{prefix}o_proj_weight": f"{p}.o_proj.weight",
    }


def mlp_mappings(layer_idx: int, prefix: str = "") -> dict:
    """Return the checkpoint mappings of one OuroMLP module."""
    p = f"model.layers.{layer_idx}.mlp"
    return {
        f"{prefix}{proj}_weight": f"{p}.{proj}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj")
    }


def decoder_mappings(layer_idx: int) -> dict:
    """Return the checkpoint mappings of one OuroDecoderLayer module."""
    mappings = attention_mappings(layer_idx, prefix="self_attn.")
    mappings.update(mlp_mappings(layer_idx, prefix="mlp."))
    for norm in NORM_NAMES:
        mappings[f"{norm}.weight"] = f"model.layers.{layer_idx}.{norm}.weight"
    return mappings


def _load_module(module: nn.Module, model_checkpoint: str, mappings: dict, device):
    """Load the TP shards of ``module``. Raise if a mapped key is missing."""
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    checkpoint = SafetensorsCheckpoint(model_checkpoint, None)
    if envs.VLLM_NEURON_CPU_MODE:
        result = checkpoint.load_sharded(rank, world_size, module, mappings, device)
    else:
        result = checkpoint.load_sharded_pipelined(
            rank, world_size, module, mappings, device
        )
    if result.missing_keys:
        raise RuntimeError(
            f"The checkpoint does not have the keys of {result.missing_keys}. "
            "Use the pinned Ouro-1.4B checkpoint."
        )
    module.load_state_dict(result.state_dict, strict=False, assign=True)
    return module


def _maybe_compile(module: nn.Module, device):
    """Compile the module for Neuron. Keep it eager in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE:
        return module
    return torch.compile(
        module.to(device), fullgraph=True, backend=envs.get_compile_backend_name()
    )


def _allocate_slots(attn: nn.Module, cache_shape: tuple, device) -> None:
    """Give each UT-step KV slot of ``attn`` a zero cache."""
    for slot in attn.kv_slots:
        slot.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
        slot.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)


def _other_slot_sums(attn: nn.Module, ut_step: int) -> torch.Tensor:
    """Return the absolute sum of each KV slot except the slot of ``ut_step``."""
    sums = [
        slot.k_cache.float().abs().sum() + slot.v_cache.float().abs().sum()
        for u, slot in enumerate(attn.kv_slots)
        if u != ut_step
    ]
    return torch.stack(sums)


def write_context_cache(
    cache: torch.Tensor,
    context: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> None:
    """Write a context K or V into a paged cache.

    Args:
        cache: [num_blocks, nkh, block_size, D] cache of this rank.
        context: [B, nkh, C, D] K or V after RoPE. C divides by block_size.
        block_table: [B, blocks_per_seq] cache blocks of each sequence.
        block_size: Tokens in one block.
    """
    B, nkh, C, D = context.shape
    num_ctx_blocks = C // block_size
    blocks = context.view(B, nkh, num_ctx_blocks, block_size, D).permute(0, 2, 1, 3, 4)
    for b in range(B):
        cache[block_table[b, :num_ctx_blocks].long()] = blocks[b].to(cache.dtype)


def prefill_metadata(
    layer_name: str, num_tokens: int, block_size: int, first_block: int = 1
) -> dict:
    """Return prefill metadata for one sequence that starts at ``first_block``.

    Block 0 is the null block of the scheduler, so the sequence starts at
    block 1.
    """
    num_seq_blocks = (num_tokens + block_size - 1) // block_size
    return {
        layer_name: {
            "max_query_len": num_tokens,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(num_tokens, dtype=torch.int64)
            + first_block * block_size,
            "block_size": block_size,
            "block_table_tensor": torch.arange(
                first_block, first_block + num_seq_blocks, dtype=torch.int32
            ).unsqueeze(0),
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def gather_cache_tokens(
    caches: list[torch.Tensor], slots: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Read the cache rows of ``slots`` from each rank. Return [Nkv, T, D]."""
    blocks = slots // block_size
    positions = slots % block_size
    per_rank = [c[blocks, :, positions].permute(1, 0, 2) for c in caches]
    return torch.cat(per_rank, dim=0)


# -----------------------------------------------------------------------------
# Module harnesses
# -----------------------------------------------------------------------------


class AttentionHarness(nn.Module):
    """Run OuroAttention for one UT step. Return the output and the caches.

    The input is the SP shard of this rank in prefill, and all tokens in
    decode. The output is
    (attention output, K cache of the step, V cache of the step, the absolute
    sum of each other slot).
    """

    def __init__(self, attn: nn.Module, ut_step: int):
        super().__init__()
        self.attn = attn
        self.ut_step = ut_step

    def forward(self, hidden_states, positions, cos, sin, attn_metadata):
        out = self.attn(
            hidden_states,
            positions,
            (cos, sin),
            attn_metadata,
            ut_step=self.ut_step,
        )
        slot = self.attn.kv_slots[self.ut_step]
        return (
            out,
            slot.k_cache.clone(),
            slot.v_cache.clone(),
            _other_slot_sums(self.attn, self.ut_step),
        )


class DecoderHarness(nn.Module):
    """Run OuroDecoderLayer for one UT step in prefill.

    Return (layer output, K cache of the step, V cache of the step, the
    absolute sum of each other slot).
    """

    def __init__(self, layer: nn.Module, ut_step: int):
        super().__init__()
        self.layer = layer
        self.ut_step = ut_step

    def forward(self, hidden_states, positions, cos, sin, attn_metadata):
        out = self.layer(
            hidden_states,
            positions,
            (cos, sin),
            attn_metadata,
            ut_step=self.ut_step,
            is_prefill=True,
        )
        attn = self.layer.self_attn
        slot = attn.kv_slots[self.ut_step]
        return (
            out,
            slot.k_cache.clone(),
            slot.v_cache.clone(),
            _other_slot_sums(attn, self.ut_step),
        )


def load_attention_harness(
    model_checkpoint: str,
    layer_idx: int,
    ut_step: int,
    num_blocks: int,
    block_size: int,
    device: str,
    context_k: torch.Tensor | None = None,
    context_v: torch.Tensor | None = None,
    context_block_table: torch.Tensor | None = None,
):
    """Load an AttentionHarness for one layer.

    Give ``context_k``, ``context_v`` and ``context_block_table`` to write a
    decode context into the cache of ``ut_step`` before the first call. The
    context tensors hold all KV heads, [B, Nkv, C, D]. Each rank writes its
    own heads.
    """
    OuroAttention = src_module("model").OuroAttention

    config = ouro_config(model_checkpoint)
    attn = OuroAttention(config, layer_idx=layer_idx)
    _load_module(attn, model_checkpoint, attention_mappings(layer_idx), device)

    nkh = attn.num_key_value_heads_per_rank
    cache_shape = (num_blocks, nkh, block_size, config.head_dim)
    _allocate_slots(attn, cache_shape, "cpu")

    if context_k is not None:
        heads = slice(attn.rank * nkh, (attn.rank + 1) * nkh)
        slot = attn.kv_slots[ut_step]
        write_context_cache(
            slot.k_cache, context_k[:, heads], context_block_table, block_size
        )
        write_context_cache(
            slot.v_cache, context_v[:, heads], context_block_table, block_size
        )

    for slot in attn.kv_slots:
        slot.k_cache = slot.k_cache.to(device)
        slot.v_cache = slot.v_cache.to(device)

    return _maybe_compile(AttentionHarness(attn, ut_step), device)


def load_mlp_harness(model_checkpoint: str, layer_idx: int, device: str):
    """Load the OuroMLP of one layer. Call it with (hidden_states, is_prefill)."""
    OuroMLP = src_module("model").OuroMLP

    config = ouro_config(model_checkpoint)
    mlp = OuroMLP(config)
    _load_module(mlp, model_checkpoint, mlp_mappings(layer_idx), device)
    return _maybe_compile(mlp, device)


def load_decoder_harness(
    model_checkpoint: str,
    layer_idx: int,
    ut_step: int,
    num_blocks: int,
    block_size: int,
    device: str,
):
    """Load a DecoderHarness for one layer."""
    OuroDecoderLayer = src_module("model").OuroDecoderLayer

    config = ouro_config(model_checkpoint)
    layer = OuroDecoderLayer(config, layer_idx=layer_idx)
    _load_module(layer, model_checkpoint, decoder_mappings(layer_idx), device)
    nkh = layer.self_attn.num_key_value_heads_per_rank
    _allocate_slots(
        layer.self_attn, (num_blocks, nkh, block_size, config.head_dim), device
    )
    return _maybe_compile(DecoderHarness(layer, ut_step), device)


# -----------------------------------------------------------------------------
# Full-model checks (CPU mode)
# -----------------------------------------------------------------------------


def _shard(t: torch.Tensor, dim: int, rank: int, world_size: int) -> torch.Tensor:
    size = t.shape[dim] // world_size
    return t.narrow(dim, rank * size, size)


class WeightCheckHarness(nn.Module):
    """Compare the loaded TP shards of the full model with the checkpoint.

    The forward raises AssertionError at the first difference. It returns the
    number of checkpoint tensors that it read and compared. A fused parameter
    (for example ``qkv_proj_weight``) counts each checkpoint tensor in it.
    """

    def __init__(self, model: nn.Module, model_checkpoint: str):
        super().__init__()
        self.model = model
        self.model_checkpoint = model_checkpoint

    def _expected_shards(self, rank: int, world_size: int) -> dict:
        """Slice the checkpoint tensors by hand, in the parameter layout."""
        import os

        from safetensors import safe_open

        cfg = self.model.config
        last = cfg.num_hidden_layers - 1
        expected = {}
        read_keys = set()
        path = os.path.join(self.model_checkpoint, "model.safetensors")
        with safe_open(path, framework="pt", device="cpu") as f:

            def get(key):
                read_keys.add(key)
                return f.get_tensor(key).to(cfg.torch_dtype)

            for i in (0, last):
                p = f"model.layers.{i}"
                q = _shard(get(f"{p}.self_attn.q_proj.weight"), 0, rank, world_size)
                k = _shard(get(f"{p}.self_attn.k_proj.weight"), 0, rank, world_size)
                v = _shard(get(f"{p}.self_attn.v_proj.weight"), 0, rank, world_size)
                expected[f"{p}.self_attn.qkv_proj_weight"] = torch.cat(
                    [q.t(), k.t(), v.t()], dim=1
                )
                expected[f"{p}.self_attn.o_proj_weight"] = _shard(
                    get(f"{p}.self_attn.o_proj.weight"), 1, rank, world_size
                ).t()
                for proj in ("gate_proj", "up_proj"):
                    expected[f"{p}.mlp.{proj}_weight"] = _shard(
                        get(f"{p}.mlp.{proj}.weight"), 0, rank, world_size
                    ).t()
                expected[f"{p}.mlp.down_proj_weight"] = _shard(
                    get(f"{p}.mlp.down_proj.weight"), 1, rank, world_size
                ).t()
                for norm in NORM_NAMES:
                    expected[f"{p}.{norm}.weight"] = get(f"{p}.{norm}.weight")

            expected["model.embed_tokens.weight"] = _shard(
                get("model.embed_tokens.weight"), 0, rank, world_size
            )
            expected["lm_head.weight"] = _shard(
                get("lm_head.weight"), 0, rank, world_size
            )
            expected["model.norm.weight"] = get("model.norm.weight")
            expected["model.early_exit_gate_weight"] = get(f"{GATE_PREFIX}.weight").t()
            expected["model.early_exit_gate_bias"] = get(f"{GATE_PREFIX}.bias")
        return expected, read_keys

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        rank = self.model.rank
        world_size = self.model.world_size
        state = dict(self.model.named_parameters())

        # Each parameter must hold real, finite values in the model dtype.
        for name, param in state.items():
            assert not param.is_meta, f"Parameter {name} is on the meta device."
            assert param.dtype == self.model.config.torch_dtype, (
                f"Parameter {name} has dtype {param.dtype}, not "
                f"{self.model.config.torch_dtype}."
            )
            assert torch.isfinite(param.float()).all(), (
                f"Parameter {name} has a value that is not finite."
            )

        expected, read_keys = self._expected_shards(rank, world_size)
        for name, ref in expected.items():
            assert name in state, f"The model has no parameter {name}."
            got = state[name]
            assert tuple(got.shape) == tuple(ref.shape), (
                f"Parameter {name} has the shape {tuple(got.shape)} on rank "
                f"{rank}, not {tuple(ref.shape)}."
            )
            assert torch.equal(got.cpu(), ref), (
                f"Parameter {name} on rank {rank} is not equal to the checkpoint shard."
            )
        return torch.tensor([len(read_keys)], dtype=torch.int64) + dummy * 0


def load_weight_check_harness(model_checkpoint: str, device: str):
    """Load the full model with ``load_weights`` and wrap it for the check."""
    OuroForCausalLM = src_module("model").OuroForCausalLM

    config = ouro_config(model_checkpoint)
    model = OuroForCausalLM(config)
    model.load_weights(model_checkpoint, torch.device(device), None)
    return WeightCheckHarness(model, model_checkpoint)


class KVSpecHarness(nn.Module):
    """Check the KV spec, the cache binding and the cache owners.

    The forward raises AssertionError at the first difference. It returns
    the number of KV layers.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        cfg = self.model.config
        L, U = cfg.num_hidden_layers, cfg.total_ut_steps
        world_size = self.model.world_size
        expected_heads = max(cfg.num_key_value_heads // world_size, 1)

        names = self.model.kv_layer_names()
        assert names == [f"layers.{n}.self_attn" for n in range(L * U)], (
            f"The KV layer names are not layers.0..{L * U - 1}.self_attn in "
            f"UT-step order. The first names are {names[:3]}."
        )

        spec = self.model.get_kv_spec()
        assert [layer.name for layer in spec.layers] == names, (
            "The KV spec layer names are not the same as kv_layer_names()."
        )
        for layer in spec.layers:
            assert layer.num_kv_heads == expected_heads, (
                f"{layer.name} has {layer.num_kv_heads} KV heads, not {expected_heads}."
            )
            assert layer.head_size == cfg.head_dim, (
                f"{layer.name} has head size {layer.head_size}, not {cfg.head_dim}."
            )
            assert layer.dtype == cfg.torch_dtype, (
                f"{layer.name} has dtype {layer.dtype}, not {cfg.torch_dtype}."
            )
            assert layer.sliding_window_size is None and not layer.key_only, (
                f"{layer.name} is not a full-attention K/V layer."
            )

        caches = {
            layer.name: [
                torch.zeros(2, layer.num_kv_heads, 4, layer.head_size),
                torch.zeros(2, layer.num_kv_heads, 4, layer.head_size),
            ]
            for layer in spec.layers
        }
        self.model.bind_kv_cache(caches)
        for u in range(U):
            for i, decoder_layer in enumerate(self.model.model.layers):
                slot = decoder_layer.self_attn.kv_slots[u]
                name = f"layers.{u * L + i}.self_attn"
                assert slot.k_cache is caches[name][0], (
                    f"Slot {u} of layer {i} does not hold the K cache of {name}."
                )
                assert slot.v_cache is caches[name][1], (
                    f"Slot {u} of layer {i} does not hold the V cache of {name}."
                )

        # Each bound cache has exactly one owner module. The torch.compile
        # trace reads the caches from these module attributes.
        owners = [
            m for m in self.model.modules() if getattr(m, "k_cache", None) is not None
        ]
        assert len(owners) == L * U, (
            f"{len(owners)} modules hold a KV cache, not {L * U}."
        )
        owner_ids = {id(owner.k_cache) for owner in owners}
        assert owner_ids == {id(kv[0]) for kv in caches.values()}, (
            "The KV cache owners do not hold the bound caches."
        )

        incomplete = dict(caches)
        incomplete.pop(names[-1])
        try:
            self.model.bind_kv_cache(incomplete)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"bind_kv_cache accepts a cache dict without {names[-1]}."
            )
        return torch.tensor([len(names)], dtype=torch.int64) + dummy * 0


def load_kv_spec_harness(model_checkpoint: str, device: str):
    """Build the full model without weights, and wrap it for the KV check."""
    OuroForCausalLM = src_module("model").OuroForCausalLM

    model = OuroForCausalLM(ouro_config(model_checkpoint))
    return KVSpecHarness(model)
