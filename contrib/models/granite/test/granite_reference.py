# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Granite (ibm/PowerLM-3b) functional tests.

This copy is local to the model folder, so the tests stay self-contained. It
lives once, here in ``test/``, and every test module reaches it by file
location through ``_local_src.helper("granite_reference")``.

This module gives:

- The pinned model ID, the pinned revision and the metadata fingerprints.
- ``resolve_checkpoint()``: find a local checkpoint for the pinned revision.
- Raw tensor reads from the checkpoint shards (no port code).
- Independent HuggingFace reference modules (attention, MLP, decoder layer)
  that load the raw FP32 checkpoint tensors.
- ``module_mappings()``: the production checkpoint mappings, restricted to one
  module, for a test harness that holds that module.

The tests use the HF modules as the reference. They never use port outputs
as goldens.
"""

import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import sys
from functools import lru_cache
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


def _load_local_src():
    """Load this model folder's ``test/_local_src.py`` by file location.

    Every contrib model ships a file with this name, so a plain
    ``import _local_src`` would bind whichever copy entered ``sys.modules``
    first. The per-model name used here cannot collide.
    """
    name = "contrib_granite_local_src"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parents[0] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()

MODEL_ID = "ibm/PowerLM-3b"
HF_REVISION = "38a624db4c8c9dd15d8bfb51845b7b177348e104"

# sha256 of the metadata files at HF_REVISION.
CONFIG_SHA256 = "7a46cefc1ce42466242bdbe466c9af52c42ae32761501602c607e5e11227e879"
INDEX_SHA256 = "9eefcc67bcd63c6e629b64ce5f5ecb106cbe0f6b33b86de4d1675aef7e829874"

NUM_CHECKPOINT_TENSORS = 642
TP_SIZE = 4

# The safetensors index of PowerLM-3b has no lm_head key (tied embeddings).
INDEX_FILE = "model.safetensors.index.json"

# Set this variable to a local snapshot of the pinned revision to skip the
# download.
CHECKPOINT_DIR_ENV = "GRANITE_CHECKPOINT_DIR"


def node_rank() -> int:
    """Return the node rank that the multi-node launcher sets (0 by default)."""
    return int(os.environ.get("NODE_RANK", "0"))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_mismatches(path: str) -> list[str]:
    """Return the list of revision problems for a checkpoint directory.

    An empty list means that the directory holds the pinned revision.
    """
    problems = []
    for name, expected in (("config.json", CONFIG_SHA256), (INDEX_FILE, INDEX_SHA256)):
        file_path = os.path.join(path, name)
        if not os.path.isfile(file_path):
            problems.append(f"{name} is missing")
            continue
        actual = _sha256(file_path)
        if actual != expected:
            problems.append(f"{name} sha256 is {actual}, expected {expected}")
    # A HF hub snapshot directory has the revision as its name.
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent == "snapshots":
        snapshot = os.path.basename(os.path.abspath(path))
        if snapshot != HF_REVISION:
            problems.append(f"HF snapshot {snapshot} is not revision {HF_REVISION}")
    return problems


@lru_cache(maxsize=1)
def resolve_checkpoint() -> str:
    """Return a local checkpoint directory for MODEL_ID at HF_REVISION.

    First use the directory in ``GRANITE_CHECKPOINT_DIR``. If that copy is not
    the pinned revision, download the pinned revision from the HF hub.
    """
    path = os.environ.get(CHECKPOINT_DIR_ENV) or None
    if path is not None:
        problems = fingerprint_mismatches(path)
        if not problems:
            return path
        logger.warning(
            "Checkpoint %s is not revision %s (%s). Download the pinned revision.",
            path,
            HF_REVISION,
            "; ".join(problems),
        )

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    problems = fingerprint_mismatches(path)
    if problems:
        raise RuntimeError(
            f"Downloaded checkpoint {path} does not match revision {HF_REVISION}: "
            f"{'; '.join(problems)}. Check the pinned fingerprints."
        )
    return path


def load_hf_config(path: str):
    """Load the HF config with eager attention for the reference modules."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "eager"
    return cfg


def goldens_config(path: str):
    """Return the HF config for the full-model goldens.

    The checkpoint sets no dtype. The served model runs in bfloat16, so the
    target dtype of the goldens is bfloat16.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    if getattr(cfg, "torch_dtype", None) in (None, torch.float32):
        cfg.torch_dtype = torch.bfloat16
    return cfg


# =============================================================================
# Raw checkpoint reads
# =============================================================================


def read_tensors(path: str, keys: list[str]) -> dict[str, torch.Tensor]:
    """Read raw tensors (checkpoint dtype) from the safetensors shards."""
    from safetensors import safe_open

    index_path = os.path.join(path, INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        # A small checkpoint has one file and no index.
        weight_map = {key: "model.safetensors" for key in keys}
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    out = {}
    for file_name, file_keys in by_file.items():
        with safe_open(os.path.join(path, file_name), framework="pt") as f:
            for key in file_keys:
                out[key] = f.get_tensor(key)
    return out


def index_keys(path: str) -> list[str]:
    """Return every tensor key in the safetensors index."""
    with open(os.path.join(path, INDEX_FILE)) as f:
        return sorted(json.load(f)["weight_map"].keys())


# =============================================================================
# Production checkpoint mappings, restricted to one module
# =============================================================================


def full_mappings(hf_config) -> dict:
    """Return the production checkpoint mappings for the full model.

    The mapping method reads only the layer count and the bias flags, so a
    lightweight stand-in replaces the model object.
    """
    GraniteConfig = _local_src.load("config").GraniteConfig
    GraniteForCausalLM = _local_src.load("model").GraniteForCausalLM

    cfg = GraniteConfig.from_configs(hf_config)
    stand_in = SimpleNamespace(
        model=SimpleNamespace(layers=[None] * cfg.num_hidden_layers), config=cfg
    )
    return GraniteForCausalLM.checkpoint_mappings(stand_in)


def module_mappings(hf_config, module_prefix: str, harness_prefix: str = "") -> dict:
    """Return the mappings for one module, renamed for a test harness.

    Example: ``module_prefix="model.layers.0.mlp."`` and
    ``harness_prefix="mlp."`` map ``mlp.gate_proj_weight`` to
    ``model.layers.0.mlp.gate_proj.weight``.
    """
    out = {}
    for name, keys in full_mappings(hf_config).items():
        if name.startswith(module_prefix):
            out[harness_prefix + name[len(module_prefix) :]] = keys
    if not out:
        raise ValueError(f"No production mapping starts with {module_prefix!r}.")
    return out


# =============================================================================
# Independent HF reference modules
# =============================================================================


def _load_prefixed(module: torch.nn.Module, path: str, prefix: str) -> None:
    names = [prefix + n for n, _ in module.named_parameters()]
    raw = read_tensors(path, names)
    state = {n[len(prefix) :]: t for n, t in raw.items()}
    module.load_state_dict(state, strict=True)


def hf_attention(path: str, hf_config, layer_idx: int, dtype: torch.dtype):
    """Return the HF GraniteAttention for a layer with the checkpoint weights."""
    from transformers.models.granite.modeling_granite import GraniteAttention

    attn = GraniteAttention(hf_config, layer_idx=layer_idx).float()
    _load_prefixed(attn, path, f"model.layers.{layer_idx}.self_attn.")
    return attn.to(dtype).eval()


def hf_mlp(path: str, hf_config, layer_idx: int, dtype: torch.dtype):
    """Return the HF GraniteMLP for a layer with the checkpoint weights."""
    from transformers.models.granite.modeling_granite import GraniteMLP

    mlp = GraniteMLP(hf_config).float()
    _load_prefixed(mlp, path, f"model.layers.{layer_idx}.mlp.")
    return mlp.to(dtype).eval()


def hf_decoder_layer(path: str, hf_config, layer_idx: int, dtype: torch.dtype):
    """Return the HF GraniteDecoderLayer for a layer with the checkpoint weights."""
    from transformers.models.granite.modeling_granite import GraniteDecoderLayer

    layer = GraniteDecoderLayer(hf_config, layer_idx=layer_idx).float()
    _load_prefixed(layer, path, f"model.layers.{layer_idx}.")
    return layer.to(dtype).eval()


def hf_rope(hf_config, seq_len: int, dtype: torch.dtype):
    """Return the HF (cos, sin) for positions 0..seq_len-1, shape [1, T, Dh]."""
    from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding

    rope = GraniteRotaryEmbedding(config=hf_config)
    dummy = torch.zeros(1, seq_len, hf_config.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    return rope(dummy, position_ids)


def causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Return an additive causal mask [1, 1, T, T]."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None]


def _first(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run_hf_attention(attn, hidden: torch.Tensor, hf_config) -> torch.Tensor:
    """Run HF attention on the full sequence [T, H]. Return [T, H]."""
    dtype = hidden.dtype
    seq_len = hidden.shape[0]
    cos, sin = hf_rope(hf_config, seq_len, dtype)
    out = attn(
        hidden.unsqueeze(0),
        position_embeddings=(cos, sin),
        attention_mask=causal_mask(seq_len, dtype),
    )
    return _first(out)[0]


@torch.no_grad()
def run_hf_decoder_layer(layer, hidden: torch.Tensor, hf_config) -> torch.Tensor:
    """Run the HF decoder layer on the full sequence [T, H]. Return [T, H]."""
    dtype = hidden.dtype
    seq_len = hidden.shape[0]
    cos, sin = hf_rope(hf_config, seq_len, dtype)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    out = layer(
        hidden.unsqueeze(0),
        attention_mask=causal_mask(seq_len, dtype),
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )
    return _first(out)[0]


def reference_pair(build_fn, run_fn, path, hf_config, layer_idx, hidden_bf16):
    """Return the (FP32, BF16) HF outputs for the same BF16 input.

    The FP32 pass runs the FP32 module on the input that is cast to FP32.
    Each module loads the checkpoint weights separately, so the FP32 module
    keeps the original FP32 weights.
    """
    module_bf16 = build_fn(path, hf_config, layer_idx, torch.bfloat16)
    module_fp32 = build_fn(path, hf_config, layer_idx, torch.float32)
    out_bf16 = run_fn(module_bf16, hidden_bf16)
    out_fp32 = run_fn(module_fp32, hidden_bf16.float())
    return out_fp32, out_bf16


# =============================================================================
# Paged KV-cache metadata
# =============================================================================


def attn_metadata(
    layer_names: list[str],
    seq_len: int,
    block_size: int,
    num_blocks: int,
    is_prefill: bool,
    batch: int = 1,
) -> dict:
    """Build the per-layer attention metadata that the model runner gives.

    Prefill: one sequence of ``seq_len`` tokens in blocks 0..num_blocks-1.
    Decode: one new token at position ``seq_len - 1``.
    """
    if is_prefill:
        slot_mapping = torch.arange(seq_len, dtype=torch.long)
        max_query_len = seq_len
    else:
        slot_mapping = torch.tensor([seq_len - 1], dtype=torch.long)
        max_query_len = 1
    entry = {
        "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch, 1),
        "slot_mapping": slot_mapping,
        "max_query_len": max_query_len,
        "block_size": block_size,
        "max_blocks_per_seq": num_blocks,
        "decode_token_threshold": 1,
    }
    if is_prefill:
        entry["cached_seq_len"] = None
        entry["kv_segment_size"] = None
    return {name: dict(entry) for name in layer_names}
