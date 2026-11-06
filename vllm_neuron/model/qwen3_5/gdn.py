# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.5 GatedDeltaNet (linear-attention) module for Neuron.

This is the linear-attention half of the Qwen3.5 hybrid stack. It mirrors the
transformers ``Qwen3_5GatedDeltaNet`` but is written in the vLLM-neuron native
style: raw ``nn.Parameter`` weights in checkpoint layout, eager PyTorch math
from ``vllm_neuron.functional.attention.gdn_attention``, and no NKI yet.

Scope of this landing (Phase 1): correct, eager, single-core. TP sharding of
the GDN projections and the recurrent-state paged cache are follow-ups. The
weights are therefore stored replicated (full), which is correct for TP=1.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

import vllm_neuron.functional as NF
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3_5Config

logger = logging.getLogger(__name__)


class Qwen3_5RMSNormGated(nn.Module):
    """RMSNorm with a sigmoid/silu gate on a separate ``gate`` tensor.

    Used to gate the GDN core output with the ``z`` projection.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear-attention layer.

    Anatomy (checkpoint keys under ``model.layers.{i}.linear_attn.*``):

    * ``in_proj_qkv``  -> ``[2 * key_dim + value_dim, hidden]`` fused Q/K/V.
    * ``in_proj_z``    -> ``[value_dim, hidden]`` gating projection.
    * ``in_proj_b``    -> ``[num_v_heads, hidden]`` input gate projection.
    * ``in_proj_a``    -> ``[num_v_heads, hidden]`` decay gate projection.
    * ``conv1d``       -> ``[conv_dim, 1, kernel_size]`` depthwise causal conv.
    * ``dt_bias`` / ``A_log`` -> ``[num_v_heads]`` per-head discretization.
    * ``norm``         -> ``[head_v_dim]`` gated RMSNorm weight.
    * ``out_proj``     -> ``[hidden, value_dim]`` output projection.

    The recurrent state is ``(conv_state, ssm_state)``: a rolling conv buffer of
    width ``kernel_size - 1`` and the delta-rule state ``[B, H, k_dim, v_dim]``.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.dtype = config.torch_dtype
        self.layer_norm_epsilon = config.rms_norm_eps
        self.activation = "silu"

        # conv_dim = 2 * key_dim + value_dim (Q, K, V).
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections in checkpoint layout ([out, in]).
        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(self.conv_dim, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(self.value_dim, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(self.num_v_heads, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(self.num_v_heads, self.hidden_size, dtype=self.dtype)
        )

        # Depthwise causal conv. Checkpoint stores [conv_dim, 1, kernel_size].
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, self.conv_kernel_size, dtype=self.dtype)
        )

        # Discretization parameters (float32 for numeric stability; the gate
        # math does -exp(A_log) * softplus(a + dt_bias)).
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))

        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, self.layer_norm_epsilon, self.dtype
        )
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.value_dim, dtype=self.dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Replicated (TP=1) weight loaders.

        Phase 1 loads full weights; the projections are not TP-sharded and the
        forward performs no collectives. Multi-core TP sharding of ``in_proj_*``
        (num_k_heads // tp, num_v_heads // tp) is the next increment.
        """
        # Identity/default loaders suffice for full weights; these are explicit
        # no-op sharding loaders documenting the TP=1 intent.
        full_loaders = {
            "in_proj_qkv_weight": sharding_weight_loader(
                shard_dim=0, shard_size=self.conv_dim, num_shards=1
            ),
            "in_proj_z_weight": sharding_weight_loader(
                shard_dim=0, shard_size=self.value_dim, num_shards=1
            ),
            "in_proj_b_weight": sharding_weight_loader(
                shard_dim=0, shard_size=self.num_v_heads, num_shards=1
            ),
            "in_proj_a_weight": sharding_weight_loader(
                shard_dim=0, shard_size=self.num_v_heads, num_shards=1
            ),
            "out_proj_weight": sharding_weight_loader(
                shard_dim=0, shard_size=self.hidden_size, num_shards=1
            ),
        }
        for name, loader in full_loaders.items():
            set_weight_loader(getattr(self, name), loader)

    def _project(self, hidden_states: torch.Tensor):
        """Run the four input projections.

        Returns ``(mixed_qkv, z, b, a)`` with mixed_qkv in ``[B, T, conv_dim]``.
        """
        mixed_qkv = F.linear(hidden_states, self.in_proj_qkv_weight)
        z = F.linear(hidden_states, self.in_proj_z_weight)
        b = F.linear(hidden_states, self.in_proj_b_weight)
        a = F.linear(hidden_states, self.in_proj_a_weight)
        return mixed_qkv, z, b, a

    def _split_qkv(self, mixed_qkv: torch.Tensor):
        """Split the convolved ``[B, T, conv_dim]`` into q/k/v head tensors."""
        batch, seq, _ = mixed_qkv.shape
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch, seq, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq, self.num_v_heads, self.head_v_dim)
        return query, key, value

    def forward(
        self, hidden_states: torch.Tensor, is_prefill: bool = True
    ) -> torch.Tensor:
        """Dispatch to the stateless prefill or the cached decode path.

        Decode with a cached recurrent state is correct in ``forward_decode``
        but not yet wired to the runner's paged conv+SSM state, so the top-level
        ``forward`` only supports prefill in this phase.
        """
        if is_prefill:
            return self._forward_prefill(hidden_states)
        raise NotImplementedError(
            "GDN decode requires the recurrent-state runner plumbing (paged "
            "conv+SSM cache). Call forward_decode(hidden_states, conv_state, "
            "ssm_state) directly for unit testing."
        )

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Stateless prefill pass (chunked delta rule).

        ``hidden_states`` is the vLLM-neuron flattened token tensor
        ``[tokens, hidden]``. Phase 1 handles a single sequence, so it is viewed
        as ``[1, tokens, hidden]`` and the output is squeezed back to 2D.
        """
        tokens, _ = hidden_states.shape
        hidden_states = hidden_states.unsqueeze(0)  # [1, tokens, hidden]
        batch, seq, _ = hidden_states.shape

        mixed_qkv, z, b, a = self._project(hidden_states)

        # Depthwise causal conv over the channel dim.
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, T]
        mixed_qkv = NF.gdn_causal_conv1d_fn(
            mixed_qkv,
            self.conv1d_weight.squeeze(1),
            bias=None,
            activation=self.activation,
            seq_len=seq,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, conv_dim]

        query, key, value = self._split_qkv(mixed_qkv)
        z = z.reshape(batch, seq, self.num_v_heads, self.head_v_dim)

        g, beta = NF.gdn_gating(self.A_log, a, b, self.dt_bias)

        # GQA: replicate q/k to the number of value heads.
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = NF.gdn_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch, seq, self.value_dim)
        output = F.linear(core_attn_out, self.out_proj_weight)
        return output.squeeze(0)  # [tokens, hidden]

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        """Single-step decode with a cached recurrent state.

        ``hidden_states`` is the vLLM-neuron flattened token tensor
        ``[tokens=1, hidden]``. ``conv_state`` is ``[B, conv_dim,
        kernel_size - 1]``; ``ssm_state`` is ``[B, H, k_head_dim, v_head_dim]``.
        Returns ``(output, conv_state, ssm_state)`` with ``output`` back in 2D.
        """
        tokens, _ = hidden_states.shape
        assert tokens == 1, "forward_decode expects a single token per sequence"
        hidden_states = hidden_states.unsqueeze(0)  # [1, 1, hidden]
        batch, seq, _ = hidden_states.shape

        mixed_qkv, z, b, a = self._project(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = NF.gdn_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d_weight.squeeze(1),
            bias=None,
            activation=self.activation,
        ).transpose(1, 2)

        query, key, value = self._split_qkv(mixed_qkv)
        z = z.reshape(batch, seq, self.num_v_heads, self.head_v_dim)

        g, beta = NF.gdn_gating(self.A_log, a, b, self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, ssm_state = NF.gdn_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch, seq, self.value_dim)
        output = F.linear(core_attn_out, self.out_proj_weight)
        return output.squeeze(0), conv_state, ssm_state
