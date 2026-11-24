# SPDX-License-Identifier: Apache-2.0
"""Independent torch reference for Ouro.

Transformers does not include the Ouro architecture, and the checkpoint code
needs ``trust_remote_code``. This module gives a reference that reads the
safetensors weights directly. It does not import the Neuron implementation.

The reference follows the rounding order of the HF ``modeling_ouro.py`` of
``ByteDance/Ouro-1.4B`` (revision 574fa66c):

- RMSNorm normalizes in fp32, casts to the input dtype, then multiplies by
  the weight.
- Attention is eager attention with an fp32 softmax.
- The UT loop applies the final norm after each step.
- The exit gate uses ``torch.cumsum`` on the stacked exit probabilities and
  selects the first step at or above the threshold.

``OuroReferenceModel`` has the call signature that the golden helpers use:
``model(input_ids, past_key_values=, cache_position=, return_dict=True,
use_cache=True)``. The result has ``.logits`` and ``.past_key_values``.
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

SAFETENSORS_FILE = "model.safetensors"


def _cfg(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def read_config(model_checkpoint: str, config: Any = None) -> dict:
    """Read ``config.json``. Apply the fields of ``config`` if it is given."""
    with open(os.path.join(model_checkpoint, "config.json")) as f:
        raw = json.load(f)
    if config is not None:
        for key in (
            "num_hidden_layers",
            "total_ut_steps",
            "early_exit_threshold",
            "rms_norm_eps",
            "rope_theta",
        ):
            sentinel = object()
            value = _cfg(config, key, sentinel)
            if value is not sentinel:
                raw[key] = value
    return raw


def load_tensors(
    model_checkpoint: str, keys: Optional[list[str]], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Load the named tensors (or all tensors) from the checkpoint in ``dtype``."""
    from safetensors import safe_open

    path = os.path.join(model_checkpoint, SAFETENSORS_FILE)
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        names = list(f.keys()) if keys is None else keys
        for name in names:
            out[name] = f.get_tensor(name).to(dtype)
    return out


def layer_keys(layer_idx: int) -> list[str]:
    """Return the checkpoint keys of one decoder layer."""
    p = f"model.layers.{layer_idx}"
    return [
        f"{p}.self_attn.q_proj.weight",
        f"{p}.self_attn.k_proj.weight",
        f"{p}.self_attn.v_proj.weight",
        f"{p}.self_attn.o_proj.weight",
        f"{p}.mlp.gate_proj.weight",
        f"{p}.mlp.up_proj.weight",
        f"{p}.mlp.down_proj.weight",
        f"{p}.input_layernorm.weight",
        f"{p}.input_layernorm_2.weight",
        f"{p}.post_attention_layernorm.weight",
        f"{p}.post_attention_layernorm_2.weight",
    ]


# -----------------------------------------------------------------------------
# Functional pieces
# -----------------------------------------------------------------------------


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(input_dtype)


def ref_rope(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos and sin with the shape [S, head_dim]."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_attention(
    h: torch.Tensor,
    positions: torch.Tensor,
    w: dict[str, torch.Tensor],
    prefix: str,
    config: dict,
    past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Eager causal attention for one layer.

    Args:
        h: [B, S, H] input after input_layernorm.
        positions: [S] absolute positions of the queries.
        w: Weights keyed by checkpoint name.
        prefix: ``model.layers.{i}.self_attn``.
        config: Raw config dict.
        past_kv: Optional cached (K, V), each [B, Nkv, S_past, D], after RoPE.

    Returns:
        (output [B, S, H], (K, V) for the full sequence).
    """
    B, S, _ = h.shape
    D = config["head_dim"]
    Nh = config["num_attention_heads"]
    Nkv = config["num_key_value_heads"]
    q = F.linear(h, w[f"{prefix}.q_proj.weight"]).view(B, S, Nh, D).transpose(1, 2)
    k = F.linear(h, w[f"{prefix}.k_proj.weight"]).view(B, S, Nkv, D).transpose(1, 2)
    v = F.linear(h, w[f"{prefix}.v_proj.weight"]).view(B, S, Nkv, D).transpose(1, 2)

    cos, sin = ref_rope(positions, D, config["rope_theta"], h.dtype)
    cos, sin = cos[None, None], sin[None, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    if past_kv is not None:
        k = torch.cat([past_kv[0], k], dim=2)
        v = torch.cat([past_kv[1], v], dim=2)
    new_kv = (k, v)

    groups = Nh // Nkv
    k_rep = k.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)

    key_positions = torch.arange(k.shape[2])
    mask = torch.zeros(S, k.shape[2], dtype=h.dtype)
    mask = mask.masked_fill(
        key_positions[None, :] > positions[:, None], torch.finfo(h.dtype).min
    )
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * (D**-0.5) + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_rep).transpose(1, 2).reshape(B, S, Nh * D)
    return F.linear(out, w[f"{prefix}.o_proj.weight"]), new_kv


def ref_mlp(h: torch.Tensor, w: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    gate = F.linear(h, w[f"{prefix}.gate_proj.weight"])
    up = F.linear(h, w[f"{prefix}.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w[f"{prefix}.down_proj.weight"])


def ref_decoder_layer(
    x: torch.Tensor,
    positions: torch.Tensor,
    w: dict[str, torch.Tensor],
    layer_idx: int,
    config: dict,
    past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One Ouro decoder layer with sandwich norms."""
    p = f"model.layers.{layer_idx}"
    eps = config["rms_norm_eps"]
    residual = x
    h = ref_rms_norm(x, w[f"{p}.input_layernorm.weight"], eps)
    h, new_kv = ref_attention(h, positions, w, f"{p}.self_attn", config, past_kv)
    h = ref_rms_norm(h, w[f"{p}.input_layernorm_2.weight"], eps)
    h = residual + h
    residual = h
    h = ref_rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], eps)
    h = ref_mlp(h, w, f"{p}.mlp")
    h = ref_rms_norm(h, w[f"{p}.post_attention_layernorm_2.weight"], eps)
    return residual + h, new_kv


def ref_exit_select(
    step_hidden: list[torch.Tensor],
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    threshold: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the exit hidden state as the HF model does.

    Args:
        step_hidden: U tensors, each [B, S, H], the normed output of each step.
        gate_weight: [1, H] checkpoint weight of ``model.early_exit_gate``.
        gate_bias: [1] checkpoint bias of ``model.early_exit_gate``.
        threshold: Exit threshold. None selects the last step.

    Returns:
        (selected hidden [B, S, H], exit step [B, S]).
    """
    U = len(step_hidden)
    if threshold is None:
        last = step_hidden[-1]
        return last, torch.full(last.shape[:2], U - 1, dtype=torch.long)
    # The HF model gives each step as a contiguous tensor. For that input,
    # F.linear adds the bias in the fused addmm and rounds one time. A
    # non-contiguous input uses matmul and then a bf16 bias add, which rounds
    # two times and can change the exit step of a token.
    gates = [F.linear(hs.contiguous(), gate_weight, gate_bias) for hs in step_hidden]
    pdf = []
    remaining = torch.ones_like(gates[0].squeeze(-1))
    for idx, g in enumerate(gates):
        lam = torch.sigmoid(g.squeeze(-1))
        if idx < U - 1:
            p = lam * remaining
            remaining = remaining * (1.0 - lam)
        else:
            p = remaining
        pdf.append(p)
    stacked_pdf = torch.stack(pdf, dim=2)
    cumulative = torch.cumsum(stacked_pdf, dim=2)
    mask = cumulative >= threshold
    exit_steps = torch.argmax(mask.float(), dim=2)
    exit_steps[~mask.any(dim=2)] = U - 1
    stacked = torch.stack(step_hidden, dim=2)
    index = exit_steps[..., None, None].expand(-1, -1, 1, stacked.size(-1))
    return torch.gather(stacked, 2, index).squeeze(2), exit_steps


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------


class OuroReferenceModel(nn.Module):
    """Full Ouro causal LM in plain torch, with a list KV cache.

    The cache is a list of ``num_hidden_layers * total_ut_steps`` (K, V)
    pairs. Entry ``u * num_hidden_layers + i`` is the cache of layer ``i`` at
    UT step ``u``, as in the HF model.
    """

    def __init__(
        self, model_checkpoint: str, dtype: torch.dtype, config: Any = None
    ) -> None:
        super().__init__()
        self.config = read_config(model_checkpoint, config)
        self.dtype = dtype
        self.weights = load_tensors(model_checkpoint, None, dtype)
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        expected |= {"model.early_exit_gate.weight", "model.early_exit_gate.bias"}
        for i in range(self.config["num_hidden_layers"]):
            expected |= set(layer_keys(i))
        missing = expected - set(self.weights)
        if missing:
            raise KeyError(
                f"The checkpoint does not have {len(missing)} expected keys, "
                f"for example {sorted(missing)[:5]}."
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        cache_position: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        cfg = self.config
        w = self.weights
        L = cfg["num_hidden_layers"]
        U = cfg["total_ut_steps"]
        if cache_position is None:
            past = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            cache_position = torch.arange(past, past + input_ids.shape[1])
        positions = cache_position.to(torch.long)

        h = F.embedding(input_ids, w["model.embed_tokens.weight"])
        new_cache = []
        step_hidden = []
        for u in range(U):
            for i in range(L):
                idx = u * L + i
                past_kv = past_key_values[idx] if past_key_values else None
                h, kv = ref_decoder_layer(h, positions, w, i, cfg, past_kv)
                new_cache.append(kv)
            h = ref_rms_norm(h, w["model.norm.weight"], cfg["rms_norm_eps"])
            step_hidden.append(h)

        selected, exit_steps = ref_exit_select(
            step_hidden,
            w["model.early_exit_gate.weight"],
            w["model.early_exit_gate.bias"],
            cfg.get("early_exit_threshold"),
        )
        logits = F.linear(selected, w["lm_head.weight"])
        return SimpleNamespace(
            logits=logits,
            past_key_values=new_cache if use_cache else None,
            exit_steps=exit_steps,
        )


def load_ouro_reference(
    model_checkpoint: str, dtype: torch.dtype, config: Any = None, **kwargs
) -> OuroReferenceModel:
    """Model loader for ``get_goldens_with_cache(model_loader=...)``."""
    return OuroReferenceModel(model_checkpoint, dtype, config=config)
