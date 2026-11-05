# SPDX-License-Identifier: Apache-2.0
"""Eager PyTorch reference kernels for the Qwen3.5 GatedDeltaNet (GDN).

These are the correctness baseline for the Qwen3.5 hybrid layers. They are
pure PyTorch (no NKI, no CUDA), transcribed from the transformers reference
``transformers/models/qwen3_5/modeling_qwen3_5.py``. They run on any backend
including ``neuron``, so they unblock correctness first; an NKI megakernel in
``vllm_neuron/nki/`` will replace them for speed.

Layout conventions (matching the transformers reference):

* ``x`` / ``conv_state``: ``[B, C, L]`` (channels = ``2 * key_dim + value_dim``).
* ``query`` / ``key``: ``[B, T, H, k_head_dim]``, ``value``: ``[B, T, H, v_head_dim]``.
* ``g`` (log-space decay) and ``beta`` (input gate): ``[B, T, H]``.
"""

import torch
import torch.nn.functional as F

CHUNK_SIZE = 64


def gdn_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along ``dim``, matching the FLA ``l2norm`` implementation."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the GDN decay gate ``g`` and input gate ``beta``.

    ``g`` is returned in log space (negative); the recurrence applies ``exp()``.
    ``A_log`` and ``dt_bias`` are per-head parameters ``[num_v_heads]``.
    ``a`` and ``b`` are ``[..., num_v_heads]``.
    """
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return g, beta


def gdn_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Causal depthwise conv1d for prefill.

    ``weight`` is ``[C, kernel_size]``; the conv is grouped (``groups=C``).
    Left-padding by ``kernel_size - 1`` makes the conv causal.
    """
    _, channels, length = x.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=padding, groups=channels)
    if seq_len is not None:
        out = out[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out


def gdn_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Single-step causal depthwise conv1d for decode.

    ``conv_state`` is ``[B, C, state_len]``; it is updated in place.
    """
    _, channels, length = x.shape
    state_len = conv_state.shape[-1]
    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -length:]
    if activation is not None:
        out = F.silu(out)
    return out.to(x.dtype)


def gdn_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked Gated DeltaNet recurrence for prefill.

    Transcribed from ``torch_chunk_gated_delta_rule`` in transformers. Inputs
    are ``[B, T, H, D]``; ``g`` is log-space and ``beta`` is in ``[0, 1]``.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = gdn_l2norm(query, dim=-1)
        key = gdn_l2norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def gdn_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-step Gated DeltaNet recurrence for decode.

    Transcribed from ``torch_recurrent_gated_delta_rule`` in transformers.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = gdn_l2norm(query, dim=-1)
        key = gdn_l2norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
