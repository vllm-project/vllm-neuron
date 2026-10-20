# SPDX-License-Identifier: Apache-2.0
"""Public wrapper for the fused SWA attention kernel.

Fuses, for a single SWA layer of a TP-sharded GPT-OSS model, the chain
``QKV projection -> RoPE + Q-scale -> sliding-window attention (per-head sink +
block KV cache prior) -> output projection`` into one NKI kernel.

The kernel writes the post-RoPE K and plain V into the paged KV cache in place
at the active-token positions and returns ``(out, k_cache, v_cache)``; the FX
aliasing pass threads the cache writes back to the model's
``self.k_cache``/``self.v_cache``.

The kernel accepts both bf16 and packed-FP8 KV cache layouts (nki_library >=
1.0.14244, which ships ``nkilib.experimental.attention.swa_fused_cte`` with the
5D per-head packed layout and ``k_scale``/``v_scale`` dequant args):
  - bf16:  k_cache/v_cache ``[num_blocks, num_kv_heads, block_size, d_head]``
  - FP8:   packed ``[num_blocks, num_kv_heads, block_size // 2, d_head, 2]``,
           with per-tensor ``k_scale``/``v_scale`` (shape ``[1, 1]`` fp32).

The model only routes the packed-FP8 configuration here (the gate in
``model_mxfp4.forward_prefill`` requires ``fp8_packed``): that is the path
validated end-to-end on device. bf16-KV serving uses the qkv+segmented
fallback.

When the kernel is unavailable (CPU-mode or device without NKI), this wrapper
falls back to an inline torch implementation (below). The fallback is inlined
rather than imported from ``nkilib...swa_fused_cte_torch`` — like the qkv
fallback, the nkilib reference is numpy + ``neuron_dtypes`` + python per-slot
iteration (its FP8 outputs are not even torch tensors), so it neither traces
nor composes with this framework.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

_FP8_MAX_E4M3 = 448.0
_FP8_MAX_E5M2 = 240.0

# Initialize the SWA-fused kernel at module import, from the INSTALLED nkilib
# (no vendored copy). Guard the import so CPU-only / older-wheel environments
# still load the module and fall back to the inline torch implementation.
_wrapped_swa_fused_cte = None
try:
    import nki
    import nki.language as nl

    from nkilib.experimental.attention.swa_fused_cte import swa_fused_cte

    _swa_fused_cte_jit = nki.jit()(swa_fused_cte)
    _wrapped_swa_fused_cte = wrap_nki(_swa_fused_cte_jit)
except Exception as e:  # noqa: BLE001
    logger.debug("swa_fused_cte kernel unavailable, will fall back to torch: %s", e)


def _can_use_swa_fused_kernel(hidden: Tensor) -> bool:
    """Return True iff the NKI kernel can serve this call.

    FP8 (packed) KV cache IS supported by this kernel version, so unlike the
    v1 wrapper there is no dtype rejection here — the caller passes
    ``k_scale``/``v_scale`` for the FP8 path.
    """
    if _wrapped_swa_fused_cte is None:
        return False
    if not can_run_kernel(hidden):
        return False
    return True


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-split (non-interleaved) RoPE; cos/sin pre-duplicated across halves.

    ``x``: [..., S, d_head]; ``rotate_half([x1, x2]) = [-x2, x1]``.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _quantize_fp8(x: Tensor, scale: float, fp8_dtype: torch.dtype) -> Tensor:
    """clamp(bf16(x) / scale, ±fp8_max) cast to ``fp8_dtype``.

    Rounds through bf16 first so the quantization input matches the kernel's
    bf16 compute precision (same convention as the kernel's in-place write).
    """
    fp8_max = _FP8_MAX_E4M3 if fp8_dtype == torch.float8_e4m3fn else _FP8_MAX_E5M2
    q = (x.to(torch.bfloat16).float() / scale).clamp(-fp8_max, fp8_max)
    return q.to(fp8_dtype)


def _torch_swa_fused_impl(
    hidden_states: Tensor,
    qkv_weight: Tensor,
    op_weight: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    sink: Tensor,
    prior_tokens: Tensor,
    qkv_bias: Tensor,
    op_bias: Tensor,
    scale: float,
    sliding_window: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    d_head: int,
    k_scale: Optional[Tensor],
    v_scale: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Inline torch fallback mirroring ``swa_fused_cte``'s semantics and the
    in-place ``(out, k_cache, v_cache)`` contract, so downstream cache readers
    see the write on CPU too.

    Semantics (matching the kernel / its executable spec):
      1. ``qkv = hidden @ qkv_weight + qkv_bias``; split heads.
      2. Half-split RoPE on Q and K; Q pre-scaled by ``scale``.
      3. Active post-RoPE K / plain V scattered into the paged cache: token
         ``s`` -> logical block ``n_prior_blocks + s // block_size`` ->
         physical ``block_tables[b, .]``, slot ``s % block_size``. FP8 caches
         store ``clamp(bf16(x)/scale, ±fp8_max)``; K packed (token 2i ->
         [..., 0], 2i+1 -> [..., 1]), V token-major. Prior blocks are never
         rewritten (their bytes stay verbatim).
      4. Sliding-window causal attention over prior+active with the per-head
         sink as an extra softmax denominator column (dropped before PV).
         Attention uses the full-precision active K/V and the dequantized
         prior (as the kernel does).
      5. ``out = attn @ op_weight + op_bias``.
    """
    fp8_packed = k_scale is not None and k_cache.dim() == 5 and k_cache.shape[-1] == 2
    k_s = float(k_scale.flatten()[0].item()) if k_scale is not None else 1.0
    v_s = float(v_scale.flatten()[0].item()) if v_scale is not None else 1.0

    hidden_f = hidden_states.float()
    B, S, H = hidden_f.shape
    q_dim = num_q_heads * d_head
    kv_dim = num_kv_heads * d_head
    group_size = num_q_heads // num_kv_heads
    prior_len = int(prior_tokens.flatten()[0].item())
    n_prior_blocks = (prior_len + block_size - 1) // block_size

    # 1./2. QKV projection + RoPE (+ Q scale), all in fp32 like the kernel spec.
    qkv = torch.matmul(hidden_f, qkv_weight.float()) + qkv_bias.float()
    q = qkv[:, :, :q_dim].reshape(B, S, num_q_heads, d_head)
    k = qkv[:, :, q_dim : q_dim + kv_dim].reshape(B, S, num_kv_heads, d_head)
    v = qkv[:, :, q_dim + kv_dim :].reshape(B, S, num_kv_heads, d_head)
    cos = cos_cache.float().unsqueeze(2)  # [B, S, 1, d]
    sin = sin_cache.float().unsqueeze(2)
    q = _apply_rope(q, cos, sin) * scale
    k = _apply_rope(k, cos, sin)

    # Per-token physical block / slot for the ACTIVE region.
    s_idx = torch.arange(S, device=block_tables.device)
    active_lblk = n_prior_blocks + s_idx // block_size  # [S] logical
    active_off = s_idx % block_size  # [S]

    out = torch.zeros((B, S, H), dtype=torch.float32)
    for b in range(B):
        phys = block_tables[b, active_lblk].long()  # [S]

        # 4a. Gather the (dequantized) prior BEFORE writing the active region.
        if prior_len > 0:
            p_idx = torch.arange(prior_len, device=block_tables.device)
            pblk = block_tables[b, p_idx // block_size].long()
            poff = p_idx % block_size
            if fp8_packed:
                # fp8 advanced indexing isn't implemented on CPU: gather via a
                # uint8 bit-view, then reinterpret back to fp8.
                k_prior_u8 = k_cache.view(torch.uint8)[pblk, :, poff // 2, :, poff % 2]
                v_prior_u8 = v_cache.view(torch.uint8)[pblk, :, poff, :]
                k_prior = k_prior_u8.view(k_cache.dtype).float() * k_s
                v_prior = v_prior_u8.view(v_cache.dtype).float() * v_s
            else:
                k_prior = k_cache[pblk, :, poff, :].float()
                v_prior = v_cache[pblk, :, poff, :].float()
        else:
            k_prior = k.new_zeros((0, num_kv_heads, d_head))
            v_prior = v.new_zeros((0, num_kv_heads, d_head))

        # 3. Scatter active K/V into the cache in place (prior bytes untouched).
        if fp8_packed:
            kq = _quantize_fp8(k[b], k_s, k_cache.dtype)  # [S, kv, d]
            vq = _quantize_fp8(v[b], v_s, v_cache.dtype)
            k_cache.view(torch.uint8)[phys, :, active_off // 2, :, active_off % 2] = (
                kq.view(torch.uint8)
            )
            v_cache.view(torch.uint8)[phys, :, active_off, :] = vq.view(torch.uint8)
        else:
            k_cache[phys, :, active_off, :] = k[b].to(k_cache.dtype)
            v_cache[phys, :, active_off, :] = v[b].to(v_cache.dtype)

        # 4b. SWA causal attention over prior+active, sink as extra column.
        k_seq = torch.cat([k_prior, k[b]], dim=0)  # [T, kv, d]
        v_seq = torch.cat([v_prior, v[b]], dim=0)
        T = prior_len + S
        k_full = k_seq.repeat_interleave(group_size, dim=1)  # [T, nq, d]
        v_full = v_seq.repeat_interleave(group_size, dim=1)

        q_pos = (torch.arange(S) + prior_len).unsqueeze(1)  # [S, 1]
        k_pos = torch.arange(T).unsqueeze(0)  # [1, T]
        mask = (k_pos > q_pos) | (k_pos < (q_pos - (sliding_window - 1)))

        # scores[h, s, t] (Q already carries the softmax scale)
        scores = torch.einsum("shd,thd->hst", q[b], k_full)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        sink_col = sink[b].float().reshape(num_q_heads, 1, 1).expand(-1, S, 1)
        weights = F.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[..., :-1]
        attn = torch.einsum("hst,thd->shd", weights, v_full)  # [S, nq, d]

        # 5. Output projection.
        out[b] = torch.matmul(attn.reshape(S, q_dim), op_weight.float()) + (
            op_bias.float()
        )

    return out.to(hidden_states.dtype), k_cache, v_cache


def swa_fused_attention(
    hidden_states: Tensor,
    qkv_weight: Tensor,
    op_weight: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    sink: Tensor,
    prior_tokens: Tensor,
    qkv_bias: Tensor,
    op_bias: Tensor,
    scale: float,
    sliding_window: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    d_head: int,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused SWA attention for one layer (QKV proj + RoPE + SWA attn + O proj +
    in-kernel KV cache write).

    Layout conventions (must match ``swa_fused_cte``):
      hidden_states : [B, S, H]
      qkv_weight    : [H, (num_q_heads + 2*num_kv_heads) * d_head], cols [Q|K|V]
      op_weight     : [num_q_heads*d_head, H]
      k_cache/v_cache:
        - bf16:  [num_blocks, num_kv_heads, block_size, d_head]
        - FP8:   [num_blocks, num_kv_heads, block_size // 2, d_head, 2] (packed)
      block_tables  : [B, max_blocks_per_seq] int32 (logical->physical)
      cos_cache     : [B, S, d_head] (RoPE cos for active tokens, halves duplicated)
      sin_cache     : [B, S, d_head]
      sink          : [B, num_q_heads] per-head sink logit
      prior_tokens  : [1, 1] int32 runtime valid-prior length
      qkv_bias      : [1, (num_q_heads + 2*num_kv_heads) * d_head]
      op_bias       : [1, H]
      k_scale/v_scale: [1, 1] fp32 per-tensor dequant scale (broadcast to the
                       kernel's [128, 1] per-partition layout internally);
                       required iff the cache is FP8 (packed), else None.

    Constraints (from swa_fused_cte):
      - ``sliding_window <= block_size``.
      - ``S`` is a multiple of 128.
      - ``d_head <= 128``.
      - Caller pre-duplicates RoPE cos/sin halves and passes the active-token
        slice (length ``S``).

    Returns ``(out, k_cache, v_cache)`` — ``out`` is ``[B, S, H]``, caches
    updated in place at the active-token positions.
    """
    if not _can_use_swa_fused_kernel(hidden_states):
        return _torch_swa_fused_impl(
            hidden_states=hidden_states,
            qkv_weight=qkv_weight,
            op_weight=op_weight,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            sink=sink,
            prior_tokens=prior_tokens,
            qkv_bias=qkv_bias,
            op_bias=op_bias,
            scale=scale,
            sliding_window=sliding_window,
            block_size=block_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            d_head=d_head,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # Packed-FP8 swa_fused_cte loads the KV quant scales with a fixed
    # ``dma_copy(src=k_scale[0:pmax, 0:1])`` (swa_fused_cte.py), so the scale
    # must be HBM shape ``[pmax, 1]`` (per-partition, broadcast) — a scalar
    # ``[1, 1]`` triggers an out-of-bounds on dim 0. Broadcast the per-tensor
    # scalar the caller passes to ``[pmax, 1]`` (same convention as qkv_cte;
    # kernel versions that read only ``[0:1, 0:1]`` accept this shape too).
    _PMAX = nl.tile_size.pmax
    if k_scale is not None and k_scale.shape[0] != _PMAX:
        k_scale = k_scale.reshape(1, 1).expand(_PMAX, 1).contiguous()
    if v_scale is not None and v_scale.shape[0] != _PMAX:
        v_scale = v_scale.reshape(1, 1).expand(_PMAX, 1).contiguous()

    # NKI kernel uses LNC=2 sharding (head-parallel across two cores).
    return _wrapped_swa_fused_cte[2](
        hidden_states=hidden_states,
        qkv_weight=qkv_weight,
        op_weight=op_weight,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        sink=sink,
        prior_tokens=prior_tokens,
        qkv_bias=qkv_bias,
        op_bias=op_bias,
        scale=scale,
        sliding_window=sliding_window,
        block_size=block_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_head=d_head,
        k_scale=k_scale,
        v_scale=v_scale,
    )
