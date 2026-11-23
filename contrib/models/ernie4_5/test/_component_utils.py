# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ERNIE 4.5 component tests (CPU and device).

The helpers compute the HF goldens, seed the paged KV cache from the HF cache,
and give top-level picklable ``model_load`` functions for ``MPExecutor``.

HF stores K in the interleaved RoPE layout. The port stores K after the
load-time reorder (the even dimensions first, then the odd dimensions of each
head). ``hf_kv_to_paged`` applies the same reorder to the HF K cache. The
reorder does not change Q.K, so the attention output stays the same.

The module sits at the root of the model's ``test/`` folder, so the three test
directories share one copy. A test module reaches it with
``_local_src.load_helper("component_utils")``, which registers it in
``sys.modules`` as ``contrib_ernie4_5_test_component_utils``. The
``model_load`` functions below pickle by reference under that module name,
which the forked ``MPExecutor`` workers inherit.
"""

import copy
import importlib.util
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from vllm_neuron import envs
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint


def _load_local_src():
    """Load the sibling ``_local_src.py`` helper by file location.

    See ``_local_src.py``: a plain ``import _local_src`` would need this
    directory on ``sys.path`` and would collide with the other contrib models.
    """
    path = Path(__file__).resolve().parent / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the local-src helper at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_local_src = _load_local_src()
_checkpoint = _local_src.load_helper("checkpoint")
MODEL_ID = _checkpoint.MODEL_ID
REVISION = _checkpoint.REVISION
kv_heads_for_rank = _checkpoint.kv_heads_for_rank

_src_model = _local_src.load("model")
Ernie4_5Config = _local_src.load("config").Ernie4_5Config
Ernie4_5Attention = _src_model.Ernie4_5Attention
Ernie4_5MLP = _src_model.Ernie4_5MLP
Ernie4_5RotaryEmbedding = _src_model.Ernie4_5RotaryEmbedding

LAYER_IDX = 0
PREFILL_BLOCK_SIZE = 16
# Decode megakernel block geometry. Align the block count to LNC_SIZE * 128.
DECODE_BLOCK_SIZE = 32
BLOCKS_ALIGNMENT = 2 * 128
PREFILL_KV_SEED = 999


def layer_name(layer_idx: int = LAYER_IDX) -> str:
    return f"layers.{layer_idx}.self_attn"


def attention_mappings(layer_idx: int = LAYER_IDX) -> dict:
    prefix = f"model.layers.{layer_idx}.self_attn"
    return {
        "qkv_proj_weight": [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ],
        "o_proj_weight": f"{prefix}.o_proj.weight",
    }


def mlp_mappings(layer_idx: int = LAYER_IDX) -> dict:
    prefix = f"model.layers.{layer_idx}.mlp"
    return {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }


def golden_key(test: str, **fields) -> dict:
    """Return a golden cache key. The key includes the pinned revision."""
    return {
        "test": test,
        "model": MODEL_ID.replace("/", "_"),
        "revision": REVISION,
        "layer_idx": LAYER_IDX,
        **fields,
    }


def golden_metadata() -> dict:
    import transformers

    return {"framework_version": f"transformers-{transformers.__version__}"}


def decode_num_blocks(total_seq_len: int) -> int:
    min_blocks = (total_seq_len + DECODE_BLOCK_SIZE - 1) // DECODE_BLOCK_SIZE
    aligned = (min_blocks + BLOCKS_ALIGNMENT - 1) // BLOCKS_ALIGNMENT
    return max(BLOCKS_ALIGNMENT, aligned * BLOCKS_ALIGNMENT)


def prefill_hidden(seed: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)


def decode_hidden(seed: int, num_steps: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_steps, hidden_size, dtype=torch.bfloat16)


def mlp_hidden(seed: int, tokens: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.rand(tokens, hidden_size) * 2 - 1).to(torch.bfloat16)


def rope(positions: torch.Tensor, hf_config_dict: dict):
    """Return the port cos/sin in BF16, as ``Ernie4_5Model.forward`` gives them."""
    config = Ernie4_5Config.from_configs(hf_config_dict, NeuronConfig())
    return Ernie4_5RotaryEmbedding(config)(
        positions, device=torch.device("cpu"), dtype=torch.bfloat16
    )


# -----------------------------------------------------------------------------
# HF goldens
# -----------------------------------------------------------------------------


def _load_hf_model(checkpoint: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(torch.float32).min), diagonal=1
    )
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


def compute_attention_prefill_goldens(
    checkpoint: str, seq_len: int, seeds: list[int]
) -> dict:
    """Compute the HF prefill attention outputs (FP32 and BF16) of layer 0."""
    model = _load_hf_model(checkpoint)
    hf_attn = model.model.layers[LAYER_IDX].self_attn
    hf_attn_fp32 = copy.deepcopy(hf_attn).float()
    pos = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

    fp32, bf16 = [], []
    for seed in seeds:
        h = prefill_hidden(seed, seq_len, model.config.hidden_size).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h, pos)
        with torch.no_grad():
            o32, _ = hf_attn_fp32(
                h.float(),
                position_embeddings=(cos.float(), sin.float()),
                attention_mask=_causal_mask(seq_len, torch.float32),
            )
            o16, _ = hf_attn(
                h,
                position_embeddings=(cos, sin),
                attention_mask=_causal_mask(seq_len, torch.bfloat16),
            )
        fp32.append(o32[0].detach())
        bf16.append(o16[0].detach())
    return {"fp32": fp32, "bf16": bf16}


def compute_attention_decode_goldens(
    checkpoint: str, prefill_len: int, num_steps: int, seeds: list[int]
) -> dict:
    """Compute the HF decode attention outputs of layer 0 after one prefill.

    The prefill input is the same for each sample. The function also returns
    the BF16 K/V of the prefill in the HF layout ``[1, kv_heads, S, Hd]``.
    """
    from transformers import DynamicCache

    model = _load_hf_model(checkpoint)
    hidden_size = model.config.hidden_size
    hf_attn = model.model.layers[LAYER_IDX].self_attn
    hf_attn_fp32 = copy.deepcopy(hf_attn).float()

    total = prefill_len + num_steps
    pos_full = torch.arange(total, dtype=torch.int64).unsqueeze(0)
    cos_full, sin_full = model.model.rotary_emb(
        torch.zeros(1, total, hidden_size, dtype=torch.bfloat16), pos_full
    )
    h_prefill = prefill_hidden(PREFILL_KV_SEED, prefill_len, hidden_size).unsqueeze(0)

    def run(attn, dtype):
        per_step = [[] for _ in range(num_steps)]
        prefill_kv = None
        for seed in seeds:
            cache = DynamicCache()
            h_decode = decode_hidden(seed, num_steps, hidden_size)
            with torch.no_grad():
                attn(
                    h_prefill.to(dtype),
                    position_embeddings=(
                        cos_full[:, :prefill_len],
                        sin_full[:, :prefill_len],
                    ),
                    attention_mask=_causal_mask(prefill_len, dtype),
                    past_key_values=cache,
                )
                if prefill_kv is None:
                    layer = cache.layers[LAYER_IDX]
                    prefill_kv = (
                        layer.keys[:, :, :prefill_len].detach().clone(),
                        layer.values[:, :, :prefill_len].detach().clone(),
                    )
                for step in range(num_steps):
                    p = prefill_len + step
                    out, _ = attn(
                        h_decode[step : step + 1].unsqueeze(0).to(dtype),
                        position_embeddings=(
                            cos_full[:, p : p + 1],
                            sin_full[:, p : p + 1],
                        ),
                        attention_mask=None,
                        past_key_values=cache,
                    )
                    per_step[step].append(out[0].detach())
        return per_step, prefill_kv

    # The HF model gives FP32 cos/sin to each layer, also in BF16. Do the same.
    per_step_fp32, _ = run(hf_attn_fp32, torch.float32)
    per_step_bf16, (k_bf16, v_bf16) = run(hf_attn, torch.bfloat16)
    return {
        "per_step_fp32": per_step_fp32,
        "per_step_bf16": per_step_bf16,
        "prefill_k_bf16": k_bf16,
        "prefill_v_bf16": v_bf16,
    }


def compute_mlp_goldens(checkpoint: str, tokens: int, seeds: list[int]) -> dict:
    """Compute the HF MLP outputs (FP32 and BF16) of layer 0."""
    model = _load_hf_model(checkpoint)
    hf_mlp = model.model.layers[LAYER_IDX].mlp
    hf_mlp_fp32 = copy.deepcopy(hf_mlp).float()
    fp32, bf16 = [], []
    for seed in seeds:
        h = mlp_hidden(seed, tokens, model.config.hidden_size)
        with torch.no_grad():
            fp32.append(hf_mlp_fp32(h.float()).detach())
            bf16.append(hf_mlp(h).detach())
    return {"fp32": fp32, "bf16": bf16}


# -----------------------------------------------------------------------------
# Paged KV cache
# -----------------------------------------------------------------------------


def hf_kv_to_paged(
    k_hf: torch.Tensor, v_hf: torch.Tensor, block_size: int, num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the HF KV ``[1, H, S, D]`` to the paged port layout ``[N, H, B, D]``.

    Token t goes to block t // block_size, offset t % block_size. K gets the
    port head-dimension order: the even dimensions first, then the odd
    dimensions.
    """
    k_port = torch.cat([k_hf[..., 0::2], k_hf[..., 1::2]], dim=-1)
    _, heads, seq_len, head_dim = k_hf.shape
    shape = (num_blocks, heads, block_size, head_dim)
    k_paged = torch.zeros(shape, dtype=k_hf.dtype)
    v_paged = torch.zeros(shape, dtype=v_hf.dtype)
    for t in range(seq_len):
        k_paged[t // block_size, :, t % block_size] = k_port[0, :, t]
        v_paged[t // block_size, :, t % block_size] = v_hf[0, :, t]
    return k_paged, v_paged


def shard_paged_kv(
    k_paged: torch.Tensor, v_paged: torch.Tensor, world_size: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Give each TP rank the KV heads that ``Ernie4_5Attention`` assigns to it."""
    num_kv_heads = k_paged.shape[1]
    ks, vs = [], []
    for rank in range(world_size):
        heads = kv_heads_for_rank(rank, world_size, num_kv_heads)
        ks.append(k_paged[:, heads].contiguous())
        vs.append(v_paged[:, heads].contiguous())
    return ks, vs


def decode_metadata(position: int, num_blocks: int) -> dict:
    return {
        layer_name(): {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": torch.tensor([position], dtype=torch.int64),
            "block_size": DECODE_BLOCK_SIZE,
            # The decode kernel needs an int32 block table. This table uses all
            # blocks, so it has no -1 padding.
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).unsqueeze(
                0
            ),
            "max_blocks_per_seq": num_blocks,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def prefill_metadata(seq_len: int) -> dict:
    num_blocks = (seq_len + PREFILL_BLOCK_SIZE - 1) // PREFILL_BLOCK_SIZE
    return {
        layer_name(): {
            "max_query_len": seq_len,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(seq_len, dtype=torch.int64),
            "block_size": PREFILL_BLOCK_SIZE,
            "block_table_tensor": torch.zeros(1, num_blocks, dtype=torch.int32),
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


# -----------------------------------------------------------------------------
# Top-level picklable model loads (run in the MPExecutor workers)
# -----------------------------------------------------------------------------


class SPSlicePrefill(nn.Module):
    """Give the SP slice of this rank to the wrapped prefill module."""

    def __init__(self, module: nn.Module, rank: int, per_rank: int):
        super().__init__()
        self.module = module
        self.rank = rank
        self.per_rank = per_rank

    def forward(self, hidden_states, *args):
        start = self.rank * self.per_rank
        return self.module(hidden_states[start : start + self.per_rank], *args)


def _load_module(module, checkpoint, mappings, device, cpu_mode):
    rank, ws = dist.get_rank(), dist.get_world_size()
    ckpt = SafetensorsCheckpoint(checkpoint, None)
    if cpu_mode:
        result = ckpt.load_sharded(rank, ws, module, mappings, device)
    else:
        result = ckpt.load_sharded_pipelined(rank, ws, module, mappings, device)
    module.load_state_dict(result.state_dict, strict=False, assign=True)


# -----------------------------------------------------------------------------
# Kernel dispatch probe (device runs only)
# -----------------------------------------------------------------------------

# Op name -> (NF module, eligibility predicate, Torch fallbacks). The predicate
# result selects the NKI kernel or the Torch fallback during the trace.
_PROBED_OPS = {
    "qkv_proj": (
        "vllm_neuron.functional.attention.qkv",
        "_can_use_qkv_kernel",
        ("_torch_qkv_impl", "_torch_qkv_in_kernel_cache_write_impl"),
    ),
    "flash_attention": (
        "vllm_neuron.functional.attention.attention_cte",
        "_can_use_flash_attention_kernel",
        ("_torch_attention_impl",),
    ),
    "o_proj": (
        "vllm_neuron.functional.attention.o_proj",
        "_can_use_kernel",
        ("_torch_o_proj_impl",),
    ),
    "attention_decode": (
        "vllm_neuron.functional.attention.attention_decode",
        "_can_use_attention_block_kernel",
        ("_torch_attention_decode_impl",),
    ),
    "mlp": ("vllm_neuron.functional.mlp", "_can_use_kernel", ("_torch_mlp_impl",)),
}
_DISPATCH_RECORDS: dict[str, set] = {}


@torch._dynamo.assume_constant_result
def _note_dispatch(op: str, value: bool) -> None:
    # Dynamo folds this call to a constant. Thus it runs once at trace time,
    # outside the compiled graph.
    _DISPATCH_RECORDS.setdefault(op, set()).add(bool(value))


def _wrap_predicate(op, orig):
    def predicate(*args, **kwargs):
        result = orig(*args, **kwargs)
        _note_dispatch(op, result)
        return result

    return predicate


def _wrap_fallback(op, orig):
    def fallback(*args, **kwargs):
        _note_dispatch(f"{op}:torch", True)
        return orig(*args, **kwargs)

    return fallback


def install_dispatch_probe() -> None:
    """Record the kernel choice of each probed NF op during the next trace.

    The worker pool reuses processes, so wrap each function one time only.
    Clear the records on each call, because each model load traces again.
    """
    import importlib

    for op, (mod_name, pred_name, fallback_names) in _PROBED_OPS.items():
        mod = importlib.import_module(mod_name)
        if not getattr(mod, "_ernie_dispatch_probe", False):
            setattr(mod, pred_name, _wrap_predicate(op, getattr(mod, pred_name)))
            for name in fallback_names:
                setattr(mod, name, _wrap_fallback(op, getattr(mod, name)))
            mod._ernie_dispatch_probe = True
    _DISPATCH_RECORDS.clear()


class DispatchProbe(nn.Module):
    """Run the compiled module. Then assert that each required op used its kernel.

    The check fails if a required op has no record, if its predicate returned
    False, or if a Torch fallback of the op ran during the trace.
    """

    def __init__(self, module, required_ops: tuple[str, ...]):
        super().__init__()
        self.module = module
        self.required_ops = required_ops

    def forward(self, *args):
        out = self.module(*args)
        problems = []
        for op in self.required_ops:
            if _DISPATCH_RECORDS.get(op) != {True}:
                problems.append(f"{op}: predicate results {_DISPATCH_RECORDS.get(op)}")
            if f"{op}:torch" in _DISPATCH_RECORDS:
                problems.append(f"{op}: the Torch fallback ran")
        if problems:
            raise AssertionError(
                f"Rank {dist.get_rank()}: a required op did not use its NKI kernel. "
                "Fix the kernel eligibility for this geometry: " + "; ".join(problems)
            )
        return out


def _finish(module, cpu_mode, required_ops: tuple[str, ...] = ()):
    if cpu_mode:
        return module
    install_dispatch_probe()
    compiled = torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )
    return DispatchProbe(compiled, required_ops)


def load_attention_prefill(
    hf_config_dict: dict, checkpoint: str, seq_len: int, device: str, cpu_mode: bool
):
    rank, ws = dist.get_rank(), dist.get_world_size()
    config = Ernie4_5Config.from_configs(hf_config_dict, NeuronConfig())
    attn = Ernie4_5Attention(config, layer_idx=LAYER_IDX)
    _load_module(attn, checkpoint, attention_mappings(), device, cpu_mode)
    num_blocks = (seq_len + PREFILL_BLOCK_SIZE - 1) // PREFILL_BLOCK_SIZE
    shape = (
        num_blocks,
        attn.num_key_value_heads_per_rank,
        PREFILL_BLOCK_SIZE,
        config.head_dim,
    )
    attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return _finish(
        SPSlicePrefill(attn, rank, seq_len // ws),
        cpu_mode,
        ("qkv_proj", "flash_attention", "o_proj"),
    )


def load_attention_decode(
    hf_config_dict: dict,
    checkpoint: str,
    k_paged_per_rank: list,
    v_paged_per_rank: list,
    device: str,
    cpu_mode: bool,
):
    rank = dist.get_rank()
    config = Ernie4_5Config.from_configs(hf_config_dict, NeuronConfig())
    attn = Ernie4_5Attention(config, layer_idx=LAYER_IDX)
    _load_module(attn, checkpoint, attention_mappings(), device, cpu_mode)
    attn.k_cache = k_paged_per_rank[rank].to(device)
    attn.v_cache = v_paged_per_rank[rank].to(device)
    return _finish(attn, cpu_mode, ("attention_decode",))


def load_mlp(hf_config_dict: dict, checkpoint: str, device: str, cpu_mode: bool):
    config = Ernie4_5Config.from_configs(hf_config_dict, NeuronConfig())
    mlp = Ernie4_5MLP(config)
    _load_module(mlp, checkpoint, mlp_mappings(), device, cpu_mode)
    return _finish(mlp, cpu_mode, ("mlp",))
