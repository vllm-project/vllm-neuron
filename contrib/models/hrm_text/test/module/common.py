# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the HRM-Text tests.

This module has two parts:

- Reference math. These functions use only torch and the raw checkpoint
  tensors. They do not import the port code. They follow the HF HRM-Text
  semantics: the HF conversion splits ``gqkv_proj`` into gate, q, k and v with
  ``chunk(4, dim=0)``, and ``gate_up_proj`` into gate and up with
  ``chunk(2, dim=0)``.
- Worker harnesses. MPExecutor workers load these top-level functions through
  ``functools.partial``. The harnesses run the port modules, which they reach
  through ``test/_local_src.py``: it loads this folder's ``src/`` by file
  location, so no import here depends on the repository root.

The unit, module and integration tests all load this module through
``_local_src.load_common()``.
"""

import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` by file location.

    That helper loads this folder's ``src/`` (and the shared
    ``test/module/common.py``) under module names unique to this contrib model.
    Loading the helper itself by location, under its own unique name, keeps
    every name out of ``sys.path``: two contrib models collected in one pytest
    run cannot resolve to each other's code, and nothing needs the repository
    root.
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "contrib_hrm_text_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()

MODEL_ID = "sapientinc/HRM-Text-1B"
MODEL_REVISION = "22097cbcecdd1301afe30a19a3ee61b96a9863e5"
# Set HRM_TEXT_MODEL_PATH to a local copy of the checkpoint at MODEL_REVISION.
MODEL_PATH = os.environ.get("HRM_TEXT_MODEL_PATH", MODEL_ID)

NUM_HEADS = 12
HEAD_DIM = 128
HIDDEN = 1536
INTERMEDIATE = 4096
VOCAB = 65536
RMS_EPS = 1e-6
ROPE_THETA = 10000.0

ATTN_LAYER_PREFIX = "model.L_module.layers.0"
MLP_LAYER_PREFIX = "model.H_module.layers.15"
KV_SLOT = 0
BLOCK_SIZE = 32


def resolve_checkpoint() -> str:
    """Return the local checkpoint directory of HRM-Text.

    ``HRM_TEXT_MODEL_PATH`` selects a local copy of the checkpoint. Without
    it, this downloads the pinned revision from the Hugging Face Hub. The
    plugin's shared checkpoint resolver lives outside this folder, so the
    contrib tests resolve the checkpoint themselves.
    """
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


def read_raw_config(checkpoint_path: str) -> dict:
    """Return the raw config.json of the checkpoint."""
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        return json.load(f)


# =============================================================================
# Raw checkpoint access (independent of the port loaders)
# =============================================================================


def _tensor_files(checkpoint_path: str) -> dict[str, str]:
    """Map each checkpoint tensor name to its safetensors file."""
    from safetensors import safe_open

    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: os.path.join(checkpoint_path, v) for k, v in weight_map.items()}
    path = os.path.join(checkpoint_path, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        return {k: path for k in f.keys()}


def checkpoint_keys(checkpoint_path: str) -> set[str]:
    """Return the set of tensor names in the checkpoint."""
    return set(_tensor_files(checkpoint_path))


def load_raw_tensors(checkpoint_path: str, names: list[str]) -> dict:
    """Read the named tensors from the checkpoint without a transform."""
    from safetensors import safe_open

    files = _tensor_files(checkpoint_path)
    out = {}
    for name in names:
        with safe_open(files[name], framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


def expected_rank_shards(raw: dict, prefix: str, rank: int, world_size: int) -> dict:
    """Return the expected per-rank parameters of one decoder layer.

    Use the HF split semantics, then take the head-contiguous rows of each rank.
    The port stores each projection transposed as [in, out].
    """
    gqkv = raw[f"{prefix}.attn.gqkv_proj.weight"]
    gate, q, k, v = gqkv.chunk(4, dim=0)
    rows = NUM_HEADS * HEAD_DIM // world_size
    sl = slice(rank * rows, (rank + 1) * rows)
    o = raw[f"{prefix}.attn.o_proj.weight"]

    gate_up = raw[f"{prefix}.mlp.gate_up_proj.weight"]
    mlp_gate, mlp_up = gate_up.chunk(2, dim=0)
    inter = INTERMEDIATE // world_size
    isl = slice(rank * inter, (rank + 1) * inter)
    down = raw[f"{prefix}.mlp.down_proj.weight"]

    return {
        f"{prefix}.self_attn.qkv_proj_weight": torch.cat([q[sl], k[sl], v[sl]]).T,
        f"{prefix}.self_attn.gate_proj_weight": gate[sl].T,
        f"{prefix}.self_attn.o_proj_weight": o[:, sl].T,
        f"{prefix}.mlp.gate_proj_weight": mlp_gate[isl].T,
        f"{prefix}.mlp.up_proj_weight": mlp_up[isl].T,
        f"{prefix}.mlp.down_proj_weight": down[:, isl].T,
    }


# =============================================================================
# Reference math (HF HRM-Text semantics, eager attention)
# =============================================================================


def ref_rms_norm(x: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Parameterless RMSNorm. Compute in FP32, then cast to the input dtype."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return x32.to(x.dtype)


def ref_rope(positions: torch.Tensor, dtype: torch.dtype):
    """Default RoPE. Return cos and sin of shape [T, head_dim]."""
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def ref_gated_attention(
    x: torch.Tensor, w_gqkv: torch.Tensor, w_o: torch.Tensor
) -> torch.Tensor:
    """Causal gated attention of one sequence. x: [T, H]. Return [T, H].

    The dtype of x is the compute dtype. Softmax runs in FP32, the same as
    the HF eager attention.
    """
    dtype = x.dtype
    tokens = x.shape[0]
    w_gate, w_q, w_k, w_v = w_gqkv.to(dtype).chunk(4, dim=0)

    def heads(w):
        return (x @ w.T).view(tokens, NUM_HEADS, HEAD_DIM).transpose(0, 1)

    q, k, v = heads(w_q), heads(w_k), heads(w_v)
    cos, sin = ref_rope(torch.arange(tokens), dtype)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    scores = (q @ k.transpose(-1, -2)) * HEAD_DIM**-0.5
    mask = torch.full((tokens, tokens), float("-inf")).triu(1)
    probs = torch.softmax(scores.float() + mask, dim=-1).to(dtype)
    attn = (probs @ v).transpose(0, 1)  # [T, Nh, Dh]

    gate = torch.sigmoid((x @ w_gate.T).view(tokens, NUM_HEADS, HEAD_DIM))
    out = (gate * attn).reshape(tokens, NUM_HEADS * HEAD_DIM)
    return out @ w_o.to(dtype).T


def ref_mlp(
    x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """SwiGLU MLP. x: [T, H]. Return [T, H]."""
    dtype = x.dtype
    w_gate, w_up = w_gate_up.to(dtype).chunk(2, dim=0)
    return (F.silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.to(dtype).T


def ref_decoder_layer(x: torch.Tensor, raw: dict, prefix: str) -> torch.Tensor:
    """Pre-norm decoder layer of one sequence. x: [T, H]. Return [T, H]."""
    h = x + ref_gated_attention(
        ref_rms_norm(x),
        raw[f"{prefix}.attn.gqkv_proj.weight"],
        raw[f"{prefix}.attn.o_proj.weight"],
    )
    return h + ref_mlp(
        ref_rms_norm(h),
        raw[f"{prefix}.mlp.gate_up_proj.weight"],
        raw[f"{prefix}.mlp.down_proj.weight"],
    )


def layer_raw_names(prefix: str) -> list[str]:
    """Return the checkpoint tensor names of one decoder layer."""
    return [
        f"{prefix}.attn.gqkv_proj.weight",
        f"{prefix}.attn.o_proj.weight",
        f"{prefix}.mlp.gate_up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]


def three_way_refs(fn, x_bf16: torch.Tensor, *weights) -> tuple:
    """Return (FP32 baseline, BF16 expected) for one reference function."""
    with torch.no_grad():
        baseline = fn(x_bf16.float(), *[w.float() for w in weights])
        expected = fn(x_bf16, *[w.to(torch.bfloat16) for w in weights])
    return baseline, expected


# =============================================================================
# Port harnesses (run in MPExecutor workers)
# =============================================================================


def _hrm_config(checkpoint_path: str):
    from vllm_neuron.model.neuron_config import NeuronConfig

    HrmTextConfig = _local_src.load("config").HrmTextConfig
    return HrmTextConfig.from_configs(read_raw_config(checkpoint_path), NeuronConfig())


def _load_module(module: nn.Module, checkpoint_path: str, mappings: dict, device):
    from vllm_neuron import envs
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    rank = get_tp_group().rank_in_group
    world_size = get_tp_group().world_size
    checkpoint = SafetensorsCheckpoint(checkpoint_path, None)
    if envs.VLLM_NEURON_CPU_MODE:
        result = checkpoint.load_sharded(rank, world_size, module, mappings, device)
    else:
        result = checkpoint.load_sharded_pipelined(
            rank, world_size, module, mappings, device
        )
    module.load_state_dict(result.state_dict, strict=False, assign=True)


# =============================================================================
# NKI kernel dispatch evidence
# =============================================================================

# Kernel name parts that each phase must show in its compiled FX graph. The
# names are the functions that the NF APIs give to ``wrap_nki``.
PREFILL_ATTN_KERNELS = ("qkv", "attention_cte", "output_projection")
DECODE_ATTN_KERNELS = ("attention_block_tkg", "qkv", "output_projection")
MLP_KERNELS = ("mlp",)


def dispatch_log_path(tag: str, rank: int) -> str:
    """Return the dispatch record file of one test on one rank.

    The file goes in the per-test log directory that CI collects. Without that
    directory, the file goes in the temp directory.
    """
    import tempfile

    log_dir = os.environ.get("VLLM_NEURON_TEST_LOG_DIR") or tempfile.gettempdir()
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"hrm_text_nki_dispatch_{tag}_rank{rank}.json")


def _kernel_name(kernel_idx) -> str:
    from libtorch_neuronx_lite.nki.nki_hop import kernel_registry

    if not isinstance(kernel_idx, int):
        return "unknown"
    func = kernel_registry.get_func(kernel_idx)
    func = getattr(func, "func", func)
    return getattr(func, "__name__", None) or repr(func)


def _nki_kernels_in_graph(gm: torch.fx.GraphModule) -> list[str]:
    """Return the NKI kernel name of each ``nki_kernel_wrapper`` node."""
    names = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_name = getattr(node.target, "_name", None) or str(node.target)
        if "nki_kernel_wrapper" not in target_name:
            continue
        names.append(_kernel_name(node.kwargs.get("kernel_idx")))
    return names


def _recording_backend(log_path: str):
    """Wrap the Neuron compile backend. Record the NKI kernels of each graph.

    The record file holds one entry for each graph that the backend compiles.
    The worker writes the file. The test process reads it after the run.
    """
    import torch._dynamo

    from vllm_neuron import envs

    inner = torch._dynamo.lookup_backend(envs.get_compile_backend_name())
    if os.path.exists(log_path):
        os.remove(log_path)

    def backend(gm: torch.fx.GraphModule, example_inputs):
        record = {"graphs": []}
        if os.path.exists(log_path):
            with open(log_path) as f:
                record = json.load(f)
        record["graphs"].append(
            {
                "num_inputs": len(example_inputs),
                "nki_kernels": _nki_kernels_in_graph(gm),
            }
        )
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2)
        return inner(gm, example_inputs)

    return backend


def assert_nki_dispatch(log_path: str, expected: tuple[str, ...]) -> list[str]:
    """Check that one compiled graph calls an NKI kernel for each expected name.

    Each item of ``expected`` is a part of a kernel function name. Return the
    kernel names of the graph that matched.
    """
    assert os.path.exists(log_path), (
        f"The NKI dispatch record {log_path} does not exist. The worker did not "
        f"compile the module with the recording backend."
    )
    with open(log_path) as f:
        graphs = json.load(f)["graphs"]
    for graph in graphs:
        kernels = graph["nki_kernels"]
        if all(any(part in name for name in kernels) for part in expected):
            return kernels
    raise AssertionError(
        f"No compiled graph calls an NKI kernel for each of {expected}. "
        f"Recorded graphs: {graphs}. Check the kernel eligibility predicates."
    )


def _maybe_compile(module: nn.Module, dispatch_log: str | None = None) -> nn.Module:
    from vllm_neuron import envs

    if envs.VLLM_NEURON_CPU_MODE:
        return module
    backend = (
        _recording_backend(dispatch_log)
        if dispatch_log
        else envs.get_compile_backend_name()
    )
    return torch.compile(module, fullgraph=True, backend=backend)


class KVLayerHarness(nn.Module):
    """Run an HRM-Text attention module or decoder layer on one KV slot.

    The harness owns the paged KV cache of the slot. A call with more tokens
    than block-table rows is a prefill: the harness gives this rank its SP
    shard of the full input. Otherwise the call is a decode step with one
    token for each block-table row.
    """

    def __init__(
        self, module: nn.Module, attn: nn.Module, config, num_blocks: int, device
    ):
        super().__init__()
        HrmTextRotaryEmbedding = _local_src.load("model").HrmTextRotaryEmbedding

        self.module = module
        self.rank = get_tp_group().rank_in_group
        self.world_size = get_tp_group().world_size
        self.rotary = HrmTextRotaryEmbedding(config)
        cache_shape = (
            num_blocks,
            attn.num_key_value_heads_per_rank,
            BLOCK_SIZE,
            attn.head_dim,
        )
        attn.k_caches[KV_SLOT] = torch.zeros(
            cache_shape, dtype=torch.bfloat16, device=device
        )
        attn.v_caches[KV_SLOT] = torch.zeros(
            cache_shape, dtype=torch.bfloat16, device=device
        )

    def forward(self, hidden, positions, slot_mapping, block_table):
        kv_slot_name = _local_src.load("model").kv_slot_name

        tokens = hidden.shape[0]
        is_prefill = tokens > block_table.shape[0]
        cos, sin = self.rotary(positions, device=hidden.device, dtype=hidden.dtype)
        attn_metadata = {
            kv_slot_name(KV_SLOT): {
                "max_query_len": tokens if is_prefill else 1,
                "decode_token_threshold": 1,
                "slot_mapping": slot_mapping,
                "block_size": BLOCK_SIZE,
                "block_table_tensor": block_table,
                "cached_seq_len": None,
                "kv_segment_size": None,
            }
        }
        if is_prefill:
            per_rank = tokens // self.world_size
            hidden = hidden[self.rank * per_rank : (self.rank + 1) * per_rank]
        return self.module(
            hidden,
            positions,
            (cos, sin),
            attn_metadata,
            kv_slot=KV_SLOT,
            is_prefill=is_prefill,
        )


class MLPHarness(nn.Module):
    """Run the HRM-Text MLP. Prefill gives this rank its SP shard of the input."""

    def __init__(self, mlp: nn.Module, is_prefill: bool):
        super().__init__()
        self.mlp = mlp
        self.is_prefill = is_prefill
        self.rank = get_tp_group().rank_in_group
        self.world_size = get_tp_group().world_size

    def forward(self, hidden):
        if self.is_prefill:
            per_rank = hidden.shape[0] // self.world_size
            hidden = hidden[self.rank * per_rank : (self.rank + 1) * per_rank]
        return self.mlp(hidden, is_prefill=self.is_prefill)


def load_attention_harness(
    checkpoint_path: str,
    num_blocks: int,
    device: str,
    dispatch_log: str | None = None,
):
    """Worker model_load: the gated attention of ATTN_LAYER_PREFIX."""
    HrmTextAttention = _local_src.load("model").HrmTextAttention

    config = _hrm_config(checkpoint_path)
    attn = HrmTextAttention(config, kv_slots=[KV_SLOT])
    gqkv = f"{ATTN_LAYER_PREFIX}.attn.gqkv_proj.weight"
    mappings = {
        "qkv_proj_weight": gqkv,
        "gate_proj_weight": gqkv,
        "o_proj_weight": f"{ATTN_LAYER_PREFIX}.attn.o_proj.weight",
    }
    _load_module(attn, checkpoint_path, mappings, device)
    return _maybe_compile(
        KVLayerHarness(attn, attn, config, num_blocks, device), dispatch_log
    )


def load_decoder_layer_harness(
    checkpoint_path: str,
    num_blocks: int,
    device: str,
    dispatch_log: str | None = None,
):
    """Worker model_load: the full decoder layer of ATTN_LAYER_PREFIX."""
    HrmTextDecoderLayer = _local_src.load("model").HrmTextDecoderLayer

    config = _hrm_config(checkpoint_path)
    layer = HrmTextDecoderLayer(config, kv_slots=[KV_SLOT])
    gqkv = f"{ATTN_LAYER_PREFIX}.attn.gqkv_proj.weight"
    gate_up = f"{ATTN_LAYER_PREFIX}.mlp.gate_up_proj.weight"
    mappings = {
        "self_attn.qkv_proj_weight": gqkv,
        "self_attn.gate_proj_weight": gqkv,
        "self_attn.o_proj_weight": f"{ATTN_LAYER_PREFIX}.attn.o_proj.weight",
        "mlp.gate_proj_weight": gate_up,
        "mlp.up_proj_weight": gate_up,
        "mlp.down_proj_weight": f"{ATTN_LAYER_PREFIX}.mlp.down_proj.weight",
    }
    _load_module(layer, checkpoint_path, mappings, device)
    return _maybe_compile(
        KVLayerHarness(layer, layer.self_attn, config, num_blocks, device),
        dispatch_log,
    )


def load_mlp_harness(
    checkpoint_path: str,
    is_prefill: bool,
    device: str,
    dispatch_log: str | None = None,
):
    """Worker model_load: the MLP of MLP_LAYER_PREFIX."""
    HrmTextMLP = _local_src.load("model").HrmTextMLP

    mlp = HrmTextMLP(_hrm_config(checkpoint_path))
    gate_up = f"{MLP_LAYER_PREFIX}.mlp.gate_up_proj.weight"
    mappings = {
        "gate_proj_weight": gate_up,
        "up_proj_weight": gate_up,
        "down_proj_weight": f"{MLP_LAYER_PREFIX}.mlp.down_proj.weight",
    }
    _load_module(mlp, checkpoint_path, mappings, device)
    return _maybe_compile(MLPHarness(mlp, is_prefill), dispatch_log)


# Parameters that the full-checkpoint load test returns for comparison.
SAMPLE_LAYER_PREFIXES = ("model.L_module.layers.0", "model.H_module.layers.15")
SAMPLE_GLOBAL_PARAMS = ("model.embed_tokens.weight", "lm_head.weight", "model.z_L_init")
SAMPLE_LAYER_SUFFIXES = (
    "self_attn.qkv_proj_weight",
    "self_attn.gate_proj_weight",
    "self_attn.o_proj_weight",
    "mlp.gate_proj_weight",
    "mlp.up_proj_weight",
    "mlp.down_proj_weight",
)


def sample_param_names() -> list[str]:
    """Return the names of the sampled parameters, in the returned order."""
    names = [
        f"{prefix}.{suffix}"
        for prefix in SAMPLE_LAYER_PREFIXES
        for suffix in SAMPLE_LAYER_SUFFIXES
    ]
    return names + list(SAMPLE_GLOBAL_PARAMS)


class _LoadReport(nn.Module):
    """Return the load report tensors that the worker computed at load time."""

    def __init__(self, tensors: tuple):
        super().__init__()
        self._tensors = tensors

    def forward(self, _unused):
        return self._tensors


def load_full_model_report(checkpoint_path: str, device: str):
    """Worker model_load: load the full checkpoint into the full model.

    The returned callable gives a tuple. Element 0 is a counter tensor:
    [params not loaded, meta tensors, non-finite params, shape mismatches,
    unused checkpoint keys, KV spec layers, unique KV names, bound KV slots,
    parameter count]. The other elements are the sampled parameters, in the
    order of ``sample_param_names()``.
    """
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    HrmTextForCausalLM = _local_src.load().HrmTextForCausalLM

    rank = get_tp_group().rank_in_group
    world_size = get_tp_group().world_size
    # The factory validates the raw config, then builds the implementation.
    model = HrmTextForCausalLM.from_configs(
        read_raw_config(checkpoint_path), NeuronConfig()
    )
    param_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}

    mappings = model.checkpoint_mappings()
    checkpoint = SafetensorsCheckpoint(checkpoint_path, None)
    state = checkpoint.load_sharded(
        rank, world_size, model, mappings, device
    ).state_dict
    model.load_state_dict(state, strict=False, assign=True)

    not_loaded = sum(1 for n in param_shapes if n not in state)
    shape_mismatch = sum(
        1 for n, s in param_shapes.items() if n in state and tuple(state[n].shape) != s
    )
    tensors = list(model.parameters()) + list(model.buffers())
    meta = sum(1 for t in tensors if t.is_meta)
    nonfinite = sum(
        1
        for p in model.parameters()
        if p.is_floating_point() and not bool(torch.isfinite(p).all())
    )

    referenced = set()
    for name in param_shapes:
        value = mappings.get(name, name)
        referenced.update([value] if isinstance(value, str) else value)
    unused = len(checkpoint_keys(checkpoint_path) - referenced)

    spec = model.get_kv_spec()
    names = [layer.name for layer in spec.layers]
    fake = {name: [torch.zeros(1, dtype=torch.bfloat16)] * 2 for name in names}
    model.bind_kv_cache(fake)
    bound = sum(len(attn.k_caches) for attn in model._attention_layers())

    counters = torch.tensor(
        [
            not_loaded,
            meta,
            nonfinite,
            shape_mismatch,
            unused,
            len(names),
            len(set(names)),
            bound,
            len(param_shapes),
        ],
        dtype=torch.int64,
    )
    params = dict(model.named_parameters())
    samples = tuple(params[n].detach().clone() for n in sample_param_names())
    return _LoadReport((counters,) + samples)
