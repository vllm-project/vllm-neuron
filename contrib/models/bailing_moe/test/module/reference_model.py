# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference of the HF BailingMoeV2 forward pass.

The HF remote modeling code of inclusionAI/Ling-mini-2.0 does not import with
the transformers version of the test image. This module is an independent
transcription of that code (revision a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab).
It uses the HF checkpoint layout directly. It does not use the Neuron model
code, the Neuron weight loaders or the RoPE head permutation.

The transcription keeps the HF numerics:
  - RMSNorm in FP32, then a cast to the input dtype, then the multiply by the weight.
  - Fused query_key_value split into Q heads, K heads and V heads.
  - Per-head QK RMSNorm before RoPE.
  - Partial RoPE on the first ``rotary_dim`` head dims with rotate_half.
  - Eager attention with the softmax in FP32.
  - Sigmoid router in FP32 with the expert bias for the selection only and
    group-limited top-k.
  - Expert outputs cast to FP32, multiplied by the routing weights, summed and
    cast back.

The router weight and the expert bias stay in FP32 for each dtype. The Neuron
model also keeps them in FP32.

``load_bailing_reference`` has the ``model_loader`` signature of
``vllm_neuron.accuracy.goldens.generate_three_way_reference_logits``.
"""

import gc
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

FP32_KEY_SUFFIXES = ("mlp.gate.weight", "mlp.gate.expert_bias")


# =============================================================================
# Config and weights
# =============================================================================


def read_reference_config(checkpoint: str, config=None) -> SimpleNamespace:
    """Read the BailingMoeV2 fields that the reference needs.

    ``config`` is an optional PretrainedConfig or dict. If it is None, the
    function reads ``config.json`` from ``checkpoint``.
    """
    if isinstance(config, dict):
        raw = dict(config)
    elif config is not None and hasattr(config, "to_dict"):
        raw = config.to_dict()
    else:
        # A config without to_dict (for example a dtype-only namespace) does
        # not hold the architecture. Read config.json from the checkpoint.
        with open(os.path.join(checkpoint, "config.json")) as f:
            raw = json.load(f)

    rope_parameters = raw.get("rope_parameters") or {}
    head_dim = raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
    partial = raw.get(
        "partial_rotary_factor", rope_parameters.get("partial_rotary_factor", 1.0)
    )
    rope_theta = raw.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
    return SimpleNamespace(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        rope_theta=float(rope_theta),
        rms_norm_eps=raw["rms_norm_eps"],
        use_qk_norm=raw.get("use_qk_norm", True),
        first_k_dense_replace=raw.get("first_k_dense_replace", 0),
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        n_group=raw.get("n_group", 1),
        topk_group=raw.get("topk_group", 1),
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.0),
        norm_topk_prob=raw.get("norm_topk_prob", True),
        num_shared_experts=raw.get("num_shared_experts"),
    )


def _tensor_index(checkpoint: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: os.path.join(checkpoint, v) for k, v in weight_map.items()}
    single = os.path.join(checkpoint, "model.safetensors")
    from safetensors import safe_open

    with safe_open(single, framework="pt") as f:
        return {k: single for k in f.keys()}


def _target_dtype(key: str, dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if key.endswith(FP32_KEY_SUFFIXES) else dtype


def load_reference_tensors(
    checkpoint: str, keys: list[str], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Read ``keys`` from the checkpoint and cast them for the reference.

    The router tensors stay in FP32. The other tensors use ``dtype``.
    """
    from safetensors import safe_open

    index = _tensor_index(checkpoint)
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"Checkpoint keys are missing: {missing[:5]}")
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(index[key], []).append(key)
    out = {}
    for path, file_keys in by_file.items():
        with safe_open(path, framework="pt") as f:
            for key in file_keys:
                out[key] = f.get_tensor(key).to(_target_dtype(key, dtype))
    return out


def load_reference_layer(
    checkpoint: str,
    cfg: SimpleNamespace,
    layer_idx: int,
    dtype: torch.dtype,
    include_mlp: bool = True,
) -> dict[str, torch.Tensor]:
    """Load the weights of one decoder layer with the experts stacked.

    Returns a dict with the short names that :func:`reference_decoder_layer` reads.
    Set ``include_mlp=False`` to load only the norms and the attention weights.
    """
    p = f"model.layers.{layer_idx}"
    keys = {
        "input_layernorm": f"{p}.input_layernorm.weight",
        "post_attention_layernorm": f"{p}.post_attention_layernorm.weight",
        "qkv": f"{p}.attention.query_key_value.weight",
        "dense": f"{p}.attention.dense.weight",
    }
    if cfg.use_qk_norm:
        keys["query_layernorm"] = f"{p}.attention.query_layernorm.weight"
        keys["key_layernorm"] = f"{p}.attention.key_layernorm.weight"
    is_moe = layer_idx >= cfg.first_k_dense_replace
    if not include_mlp:
        pass
    elif not is_moe:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys[proj] = f"{p}.mlp.{proj}.weight"
    else:
        keys["router"] = f"{p}.mlp.gate.weight"
        keys["expert_bias"] = f"{p}.mlp.gate.expert_bias"
        if cfg.num_shared_experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys[f"shared_{proj}"] = f"{p}.mlp.shared_experts.{proj}.weight"

    tensors = load_reference_tensors(checkpoint, list(keys.values()), dtype)
    layer = {name: tensors[key] for name, key in keys.items()}
    layer["is_moe"] = is_moe

    if is_moe and include_mlp:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expert_keys = [
                f"{p}.mlp.experts.{e}.{proj}.weight" for e in range(cfg.num_experts)
            ]
            expert_tensors = load_reference_tensors(checkpoint, expert_keys, dtype)
            layer[f"experts_{proj}"] = torch.stack(
                [expert_tensors.pop(k) for k in expert_keys], dim=0
            )
            del expert_tensors
    return layer


# =============================================================================
# Building blocks
# =============================================================================


def reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """HF BailingMoeV2RMSNorm."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


def reference_rotary(
    cfg: SimpleNamespace, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF BailingMoeV2RotaryEmbedding. Returns cos and sin of shape [S, rotary_dim]."""
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, cfg.rotary_dim, 2, dtype=torch.int64).float()
            / cfg.rotary_dim
        )
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reference_apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """HF partial RoPE. ``x`` is [B, N, S, Dh]. ``cos`` and ``sin`` are [S, R]."""
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


def reference_attention(
    cfg: SimpleNamespace,
    layer: dict,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """HF BailingMoeV2Attention with eager attention.

    ``x`` is [B, S, H] and ``positions`` is [S]. ``cache`` holds the K and V of
    the earlier tokens, each [B, Nkv, S_prior, Dh], or None.
    """
    bsz, q_len, _ = x.shape
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qkv = F.linear(x, layer["qkv"]).view(bsz, q_len, nh + 2 * nkv, hd)
    q, k, v = qkv.split([nh, nkv, nkv], dim=-2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if cfg.use_qk_norm:
        q = reference_rms_norm(q, layer["query_layernorm"], cfg.rms_norm_eps)
        k = reference_rms_norm(k, layer["key_layernorm"], cfg.rms_norm_eps)
    cos, sin = reference_rotary(cfg, positions, x.dtype)
    q = reference_apply_partial_rope(q, cos, sin)
    k = reference_apply_partial_rope(k, cos, sin)

    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)

    groups = nh // nkv
    k_rep = k.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * (hd**-0.5)
    kv_len = k_rep.shape[2]
    key_pos = torch.arange(kv_len, device=x.device)
    allowed = key_pos[None, :] <= positions[:, None].to(key_pos.dtype)
    mask = torch.zeros(q_len, kv_len, dtype=scores.dtype, device=x.device)
    mask = mask.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    scores = scores + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_rep).transpose(1, 2).reshape(bsz, q_len, nh * hd)
    return F.linear(out, layer["dense"]), new_cache


def reference_mlp(x, gate_w, up_w, down_w):
    """HF BailingMoeV2MLP with SiLU."""
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def reference_gate(
    cfg: SimpleNamespace, x: torch.Tensor, router_w: torch.Tensor, expert_bias
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF BailingMoeV2Gate. ``x`` is [T, H]. Returns (topk_idx, topk_weight)."""
    num_tokens = x.shape[0]
    logits = F.linear(x.to(torch.float32), router_w.to(torch.float32))
    scores = torch.sigmoid(logits.float()).type_as(logits)
    scores_for_routing = scores + expert_bias.to(torch.float32)

    group_scores = (
        scores_for_routing.view(num_tokens, cfg.n_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, cfg.n_group, cfg.num_experts // cfg.n_group)
        .reshape(num_tokens, -1)
    )
    masked = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(masked, k=cfg.num_experts_per_tok, dim=-1)

    weights = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
    if cfg.num_experts_per_tok > 1 and cfg.norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return topk_idx, weights


def reference_moe(cfg: SimpleNamespace, layer: dict, x: torch.Tensor) -> torch.Tensor:
    """HF BailingMoeV2SparseMoeBlock (inference path). ``x`` is [T, H]."""
    topk_idx, topk_weight = reference_gate(
        cfg, x, layer["router"], layer["expert_bias"]
    )
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    for e in torch.unique(topk_idx).tolist():
        token_ids, slot = (topk_idx == e).nonzero(as_tuple=True)
        y = reference_mlp(
            x[token_ids],
            layer["experts_gate_proj"][e],
            layer["experts_up_proj"][e],
            layer["experts_down_proj"][e],
        )
        out.index_add_(
            0, token_ids, y.to(torch.float32) * topk_weight[token_ids, slot, None]
        )
    out = out.to(x.dtype)
    if cfg.num_shared_experts:
        out = out + reference_mlp(
            x,
            layer["shared_gate_proj"],
            layer["shared_up_proj"],
            layer["shared_down_proj"],
        )
    return out


def reference_decoder_layer(
    cfg: SimpleNamespace,
    layer: dict,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache,
):
    """HF BailingMoeV2DecoderLayer. ``x`` is [B, S, H]."""
    residual = x
    h = reference_rms_norm(x, layer["input_layernorm"], cfg.rms_norm_eps)
    h, new_cache = reference_attention(cfg, layer, h, positions, cache)
    h = residual + h
    residual = h
    h = reference_rms_norm(h, layer["post_attention_layernorm"], cfg.rms_norm_eps)
    bsz, seq, hidden = h.shape
    flat = h.reshape(bsz * seq, hidden)
    if layer["is_moe"]:
        flat = reference_moe(cfg, layer, flat)
    else:
        flat = reference_mlp(
            flat, layer["gate_proj"], layer["up_proj"], layer["down_proj"]
        )
    return residual + flat.view(bsz, seq, hidden), new_cache


# =============================================================================
# Full causal LM
# =============================================================================


class BailingReferenceForCausalLM:
    """Full-model reference with a list KV cache.

    The call contract matches what
    ``vllm_neuron.accuracy.goldens.generate_reference_logits`` and
    ``generate_teacher_forced_logits`` use.
    """

    def __init__(self, cfg: SimpleNamespace, weights: dict, dtype: torch.dtype):
        self.cfg = cfg
        self.config = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers)
        self.dtype = dtype
        self.embed = weights["embed"]
        self.final_norm = weights["norm"]
        self.lm_head = weights["lm_head"]
        self.layers = weights["layers"]

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        past_key_values=None,
        return_dict: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        seq = input_ids.shape[1]
        past = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if cache_position is None:
            cache_position = torch.arange(past, past + seq)
        positions = cache_position.to(torch.int64)

        h = self.embed[input_ids]
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = None if past_key_values is None else past_key_values[i]
            h, c = reference_decoder_layer(self.cfg, layer, h, positions, layer_cache)
            new_cache.append(c)
        h = reference_rms_norm(h, self.final_norm, self.cfg.rms_norm_eps)
        logits = F.linear(h, self.lm_head).float()
        return SimpleNamespace(logits=logits, past_key_values=new_cache)


def load_bailing_reference(
    model_checkpoint: str, dtype: torch.dtype, config=None
) -> BailingReferenceForCausalLM:
    """Load the full reference model with all layers and all experts.

    This is the ``model_loader`` for ``get_goldens_with_cache``.

    Example:
        >>> ref = load_bailing_reference("/path/to/Ling-mini-2.0", torch.float32)
        >>> out = ref(torch.tensor([[1, 2, 3]]), cache_position=torch.arange(3))
        >>> out.logits.shape[-1]
        157184
    """
    cfg = read_reference_config(model_checkpoint, config)
    top = load_reference_tensors(
        model_checkpoint,
        ["model.word_embeddings.weight", "model.norm.weight", "lm_head.weight"],
        dtype,
    )
    layers = []
    for layer_idx in range(cfg.num_hidden_layers):
        layers.append(load_reference_layer(model_checkpoint, cfg, layer_idx, dtype))
        gc.collect()
    weights = {
        "embed": top["model.word_embeddings.weight"],
        "norm": top["model.norm.weight"],
        "lm_head": top["lm_head.weight"],
        "layers": layers,
    }
    return BailingReferenceForCausalLM(cfg, weights, dtype)
