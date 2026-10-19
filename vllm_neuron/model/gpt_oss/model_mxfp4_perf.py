# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS MXFP4 giga-kernel (GK) DECODE model
===========================================

DECODE-ONLY model for GPT-OSS-120B, selected by the factory when
``neuron_config.gk_decode`` is set.

The fused ``gpt_oss_decode_mega_kernel`` runs the WHOLE decoder stack in one
NEFF, with its own TP/EP collectives. So the per-layer modules here
(``GptOssAttention`` / ``GptOssExperts`` / ...) declare NO weight params and
have NO forward. Each one exposes its weights' loaders instead, and those
loaders build the [L, ...] stacks the kernel reads. Prefill runs on the native per-layer
model (``model_mxfp4``, ``gk_decode`` unset) over disaggregated inference;
speculative decoding / Eagle3 are not supported on this path.

MXFP4 specifics retained (needed to LOAD + stack the weights the kernel reads):
- Expert weights stored in MXFP4 tiled format (uint16 blocks + uint8 scales).
- Hidden dimension shuffled for hardware memory layout; weight loaders shuffle.
- Weight loaders handle TP / attention-DP / EP sharding.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Sharding used to LOAD/place weights.
  # <-- MODEL-SPECIFIC: ...    GPT-OSS specific.
  # <-- MXFP4: ...             MXFP4 tiled-format specific.
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group


from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    ShardSpec,
    set_weight_loader,
    with_rank_override,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.functional.gpt_oss_tail import (
    TailMode,
    chunked_tail_gather,
    gpt_oss_chunked_tail,
    gpt_oss_tail_precheck,
    gpt_oss_tail_reject_reason,
    resolve_tail_mode,
    tail_glue_anchor_zero,
)

import vllm_neuron.envs as envs
import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import EmbedDimShardedEmbedding, VocabDimShardedEmbedding

# <-- MXFP4: Weight loaders for the MXFP4 tiled format with shuffling. This
# decode-only model declares no weight params; the weights are stacked and
# loaded by these helpers (the MoE/attention compute lives on the prefill model).
from vllm_neuron.model.gpt_oss.weight_loaders_mxfp4 import (
    Q_HEIGHT,
    _get_i_tiling_shard_i,
    expert_parallel_mxfp4_down_bias_loader,
    expert_parallel_mxfp4_down_scale_loader,
    expert_parallel_mxfp4_down_weight_loader,
    expert_parallel_mxfp4_gate_up_bias_loader,
    expert_parallel_mxfp4_gate_up_blocks_loader,
    expert_parallel_mxfp4_gate_up_scale_loader,
    fused_qkv_bias_loader as mxfp4_fused_qkv_bias_loader,
    fused_qkv_shuffling_weight_loader,
    o_proj_weight_loader as mxfp4_o_proj_weight_loader,
    shuffling_weight_loader,
)

from .config import GptOssConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: GPT-OSS uses RMSNorm with variance on unpadded portion
# <-- MXFP4: Hidden dim is shuffled, so norm weights are also shuffled
# =============================================================================


class GptOssRMSNorm(nn.Module):
    """RMS Normalization with support for padded hidden dimensions.

    <-- MODEL-SPECIFIC: GPT-OSS pads hidden_size to 3072 for hardware alignment.
    The variance is computed only on the unpadded portion, and the padded positions
    are zeroed after normalization.
    """

    def __init__(self, config: GptOssConfig, has_weight: bool = True):
        super().__init__()
        # has_weight=False for the per-layer norms whose weight is a stacked
        # weight; the final model norm keeps its weight and runs forward().
        if has_weight:
            self.weight = nn.Parameter(
                torch.ones(config.hidden_size, dtype=config.torch_dtype)
            )
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.hidden_size = config.hidden_size
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # <-- MXFP4: Hidden dim is shuffled, so padded zeros are interleaved.
        # Adjust mean by (H_padded / H_actual) to compensate for zero-padded positions.
        scaling_factor = self.hidden_size / self.unpadded_hidden_size
        variance = hidden_states.pow(2).mean(-1, keepdim=True) * scaling_factor
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states
        # <-- MXFP4: Zero shuffled padded positions via reshape to [4, H//4] layout
        output_shape = output.shape
        output = output.view(*output_shape[:-1], 4, self.hidden_size // 4)
        output[..., self.unpadded_hidden_size // 4 :] = 0.0
        output = output.view(output_shape)
        return output.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: GPT-OSS uses YaRN scaling for RoPE
# =============================================================================


class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling.

    <-- MODEL-SPECIFIC: YaRN interpolation/extrapolation is GPT-OSS specific.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_parameters["rope_theta"]
        rope_parameters = config.rope_parameters or {}
        self.scaling_factor = rope_parameters.get("factor", 1.0)
        self.beta_slow = rope_parameters.get("beta_slow", 1.0)
        self.beta_fast = rope_parameters.get("beta_fast", 32.0)
        self.initial_context_length = rope_parameters.get(
            "original_max_position_embeddings", 8192
        )

        inv_freq, concentration = self._compute_inv_freq_and_concentration("cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("concentration", concentration, persistent=False)
        # Python-float copy of the concentration scalar, for the GK fused-RoPE path. A float() on
        # the buffer at trace time is data-dependent and breaks the graph.
        self.concentration_float = float(concentration)

    def _compute_inv_freq_and_concentration(self, device):
        """<-- MODEL-SPECIFIC: YaRN frequency computation."""
        freq = self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = (
            d_half
            * math.log(self.initial_context_length / (self.beta_fast * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        high = (
            d_half
            * math.log(self.initial_context_length / (self.beta_slow * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (
            high - low
        )
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        concentration = torch.tensor(concentration, device=device)
        return inv_freq, concentration

    def inv_freq_on(self, device):
        """Return the RoPE ``inv_freq`` table on ``device``, recomputed from the YaRN parameters.
        The non-persistent buffer reads as unmaterialized memory inside the trace.
        """
        inv_freq, _ = self._compute_inv_freq_and_concentration(device)
        return inv_freq

    def forward(self, position_ids, device, dtype):
        inv_freq_expanded = self.inv_freq[None, :].float()
        position_ids_expanded = position_ids[:, None].float()
        freqs = position_ids_expanded @ inv_freq_expanded
        cos = freqs.cos() * self.concentration
        sin = freqs.sin() * self.concentration
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (sinks, sliding window, GQA, RoPE application)
# <-- MXFP4: Attention weights use shuffling weight loaders
# =============================================================================


# NOTE: RoPE is fused into NF.qkv_proj for prefill; decode handles RoPE in
# its own dedicated kernel call. The standalone `_apply_rotary_emb` /
# `apply_rotary_pos_emb` helpers are no longer used by either path.


def _replicated_sinks_loader(
    shard_size: int,
    num_shards: int,
    attention_dp_size: int,
    tp_rank: int,
):
    """Load sinks replicated across attention DP ranks.

    Each DP rank within the same TP position has a different shard of sinks.
    This loader concatenates all DP shards for the given TP position so the
    decode path doesn't need a runtime collective.

    Returns a tensor of size [shard_size * attention_dp_size].
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    ddp = attention_dp_size

    def transform(slices, rank):
        assert len(slices) == 1
        slice_obj = slices[0]
        shards = []
        for dp_rank in range(ddp):
            effective_rank = dp_rank + tp_rank * ddp
            start = (effective_rank % num_shards) * shard_size
            end = start + shard_size
            shards.append(slice_obj[start:end])
        return torch.cat(shards, dim=0).to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


class GptOssAttention(nn.Module):
    """Decode-only attention weight holder: no forward, no weight params.
    Holds the TP/attention-DP geometry, weight loaders, and KV scales that the
    stacking loaders and the giga-kernel read (GQA, sinks, sliding window).
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # <-- MODEL-SPECIFIC: Sliding window on alternating (even) layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Attention DP setup (read by mega_decode; the effective
        # shard count and the interleaved effective rank drive weight loading). <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_rank = get_neuron_attention_dp_rank()

        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        # KV heads-per-rank in the QKV projection output. With kv_needs_a2a,
        # attention DP shards KV further than world_size, so the projection emits
        # a smaller K/V block than num_key_value_heads_per_rank. Used by the
        # weight loaders to size the fused-QKV K/V blocks.
        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        num_kv_heads_for_weight = self.num_kv_heads_for_weight  # local alias

        # Sinks are per-head; the count spans this rank's Q heads across attn DP.
        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )

        # >>> PARALLELISM: QKV/O weight shapes for TP * attention DP <<<
        # No weight params here: the attention weights are stacked. This holds
        # only the geometry the stacking loaders and the giga-kernel read.
        self.effective_q_shards = effective_q_shards
        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = num_kv_heads_for_weight * self.head_dim
        self.qkv_size = self.q_size + 2 * self.kv_size
        self.o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        self.k_cache = None
        self.v_cache = None

        # When True, the K cache uses the swizzled packed FP8 layout
        # ([num_blocks, (kv_heads,) block_len // 2, d_head, 2]); see model_bf16.
        self.fp8_packed = False

        # KV cache quantization scales are set during weight loading.
        # We also need floats since tensor.item() causes a graph break,
        # and kernels currently use floats for perf reasons.
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

    def gk_weight_loaders(self) -> dict:
        """Return {stack_key: (loader, per_layer_shape, dtype)} for this module's
        stacked weights, from the shared MXFP4 helpers + TP/attention-DP geometry.
        """
        ddp = self.attention_dp_size
        eqs = self.effective_q_shards
        eqr = self.attention_dp_rank + self.tp_group.rank_in_group * ddp

        qkv_loader = with_rank_override(
            fused_qkv_shuffling_weight_loader(
                self.q_size,
                self.kv_size,
                (self.hidden_size, self.qkv_size),
                self.num_kv_replicas,
                attention_dp_size=ddp,
                kv_needs_a2a=self.kv_needs_a2a,
            ),
            rank=eqr,
        )
        qkv_bias_loader = with_rank_override(
            mxfp4_fused_qkv_bias_loader(
                self.q_size,
                self.kv_size,
                self.num_kv_replicas,
                attention_dp_size=ddp,
                kv_needs_a2a=self.kv_needs_a2a,
            ),
            rank=eqr,
        )
        o_loader = with_rank_override(
            mxfp4_o_proj_weight_loader(
                self.o_proj_in_features,
                (self.o_proj_in_features, self.hidden_size),
                eqs,
            ),
            rank=eqr,
        )
        o_bias_loader = with_rank_override(
            shuffling_weight_loader(0, (self.hidden_size,), None, eqs), rank=eqr
        )
        # Sinks are per-head biases — pre-gathered across attention DP at load
        # time so no runtime all_gather is needed during decode.
        sinks_loader = _replicated_sinks_loader(
            shard_size=self.num_attention_heads_per_rank,
            num_shards=eqs,
            attention_dp_size=self.attention_dp_size,
            tp_rank=self.tp_group.rank_in_group,
        )
        dt = self.dtype
        return {
            "W_qkv": (qkv_loader, (self.hidden_size, self.qkv_size), dt),
            "bias_qkv": (qkv_bias_loader, (self.qkv_size,), dt),
            "W_out": (o_loader, (self.o_proj_in_features, self.hidden_size), dt),
            "bias_out": (o_bias_loader, (self.hidden_size,), dt),
            "sink": (sinks_loader, (self.num_q_heads_after_a2a,), torch.float32),
        }


# =============================================================================
# Section 4: MoE Experts (MXFP4) — DECODE weight holder
# =============================================================================


class GptOssExperts(nn.Module):
    """Decode-only MXFP4 MoE weight holder (experts + router + post-attn norm):
    no forward, no weight params. Holds the EP/TP geometry and weight loaders
    the stacking loaders and the giga-kernel read (tiled uint16/uint8 format).
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()

        # >>> PARALLELISM: TP + EP degrees + placement <<<
        # These derive the local expert count, intermediate-dim shard, and EP rank
        # that the weight loaders + param shapes need, and that mega_decode reads
        # (ep_degree, ep_rank, mlp_dp_size, num_experts_per_token, limit) to build
        # the giga-kernel args. The MoE compute + its runtime collectives live on
        # the native prefill model; this decode-only holder keeps only degrees.
        self.tp_group = get_tp_group()
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel

        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
        )

        self.mlp_tp_rank = get_neuron_mlp_tp_group().rank_in_group

        if self.ep_enabled:
            # EP enabled: read degree/rank/tp-sub-degree from neuron parallel state.
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.tp_degree = get_neuron_ep_tp_group().world_size
        else:
            # No EP: all experts on all ranks, intermediate dim sharded across
            # TP * mlp_dp_size via the MLP TP group.
            self.ep_degree = 1
            self.ep_rank = 0
            self.tp_degree = get_neuron_mlp_tp_group().world_size

        self.total_num_experts = config.num_local_experts
        self.num_local_experts = config.num_local_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: Intermediate dim sharded across TP degree <<<
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree

        # <-- MODEL-SPECIFIC: SwiGLU clamp (mega_decode reads .limit) <--
        self.limit = config.swiglu_limit

        # <-- MXFP4: Derived sizes for tiled format
        self.num_mx_groups = self.hidden_size // 32

        # Pre-MLP RMSNorm; its weight is stacked, so no local weight param.
        self.post_attention_layernorm = GptOssRMSNorm(config, has_weight=False)

        # No router/expert weight params here: they are stacked. Keep only the
        # geometry the stacking loaders and the giga-kernel read.
        self.param_dtype = config.torch_dtype
        self.num_I_tiles, self.q_blocks_per_I_tile = _get_i_tiling_shard_i(
            self.intermediate_size_per_rank
        )

    def gk_weight_loaders(self) -> dict:
        """Return {stack_key: (loader, per_layer_shape, dtype)} for this module's
        stacked weights, from the shared EP-aware MXFP4 helpers + EP/TP geometry.
        """
        # Linear EP placement: rank k owns experts [k*L, (k+1)*L)
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )
        # The EP loaders pre-slice each input to this rank's experts, so the
        # expert dim of each expected_shape is the local expert count (equals
        # total_num_experts when ep_degree == 1).
        num_local_experts = len(local_expert_indices)
        i2 = self.intermediate_size_per_rank * 2
        i32 = self.intermediate_size_per_rank // 32
        # with_rank_override ensures the MLP TP group rank is used for loading.
        _ro = self.mlp_tp_rank

        router_loader = shuffling_weight_loader(
            shuffle_dim=1,
            expected_shard_shape=(self.total_num_experts, self.hidden_size),
            shard_spec=None,
        )
        # router_bias: identity load (replicated, no shard/shuffle).
        router_bias_loader = SafetensorsWeightLoader()
        post_ln_loader = shuffling_weight_loader(
            shuffle_dim=0,
            expected_shard_shape=(self.hidden_size,),
            shard_spec=None,
        )

        gate_up_blocks = with_rank_override(
            expert_parallel_mxfp4_gate_up_blocks_loader(
                local_expert_indices=local_expert_indices,
                shard_spec=ShardSpec(dim=1, size=i2, num_shards=self.tp_degree),
                expected_shape=(num_local_experts, i2, self.num_mx_groups, 16),
            ),
            rank=_ro,
        )
        gate_up_scale = with_rank_override(
            expert_parallel_mxfp4_gate_up_scale_loader(
                local_expert_indices=local_expert_indices,
                shard_spec=ShardSpec(dim=1, size=i2, num_shards=self.tp_degree),
                expected_shape=(num_local_experts, i2, self.num_mx_groups),
            ),
            rank=_ro,
        )
        gate_up_bias = with_rank_override(
            expert_parallel_mxfp4_gate_up_bias_loader(
                local_expert_indices=local_expert_indices,
                shard_spec=ShardSpec(dim=1, size=i2, num_shards=self.tp_degree),
                expected_shape=(num_local_experts, i2),
                hidden_act_bias=1.0,
            ),
            rank=_ro,
        )
        down_weight = with_rank_override(
            expert_parallel_mxfp4_down_weight_loader(
                local_expert_indices=local_expert_indices,
                shard_spec=ShardSpec(dim=2, size=i32, num_shards=self.tp_degree),
                expected_shape=(num_local_experts, self.hidden_size, i32, 16),
            ),
            rank=_ro,
        )
        down_scale = with_rank_override(
            expert_parallel_mxfp4_down_scale_loader(
                local_expert_indices=local_expert_indices,
                shard_spec=ShardSpec(dim=2, size=i32, num_shards=self.tp_degree),
                expected_shape=(num_local_experts, self.hidden_size, i32),
            ),
            rank=_ro,
        )
        down_bias = with_rank_override(
            expert_parallel_mxfp4_down_bias_loader(
                local_expert_indices=local_expert_indices,
                expected_shape=(num_local_experts, self.hidden_size),
                scale=self.tp_degree,
            ),
            rank=_ro,
        )

        H = self.hidden_size
        E = num_local_experts
        return {
            "post_attn_layernorm_weight": (post_ln_loader, (H,), self.param_dtype),
            "router_weights": (
                router_loader,
                (self.total_num_experts, H),
                torch.float16,
            ),
            "router_bias": (
                router_bias_loader,
                (self.total_num_experts,),
                torch.float16,
            ),
            "expert_gate_up_weights": (
                gate_up_blocks,
                (E, 128, 2, H // 512, self.intermediate_size_per_rank),
                torch.uint16,
            ),
            "expert_gate_up_weights_scale": (
                gate_up_scale,
                (E, 16, 2, H // 512, self.intermediate_size_per_rank),
                torch.uint8,
            ),
            "expert_gate_up_bias": (gate_up_bias, (E, i2), self.param_dtype),
            "expert_down_weights": (
                down_weight,
                (E, self.q_blocks_per_I_tile * Q_HEIGHT, self.num_I_tiles, H),
                torch.uint16,
            ),
            "expert_down_weights_scale": (
                down_scale,
                (E, self.q_blocks_per_I_tile, self.num_I_tiles, H),
                torch.uint8,
            ),
            "expert_down_bias": (down_bias, (E, H), self.param_dtype),
        }


# =============================================================================
# Section 5: MLP Wrapper
# =============================================================================


class GptOssMLP(nn.Module):
    """Decode-only holder for the experts submodule (no forward)."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.experts = GptOssExperts(config)
        self.dtype = config.torch_dtype


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


def _dp_transition(
    x: torch.Tensor,
    current_group,
    target_group,
    dim: int = 0,
) -> torch.Tensor:
    """Transition tensor between DP gathered states.

    SPMD: rank offsets come from the stamped ``rank_in_group_tensor`` and go through
    ``index_select``, which takes the 1-D stamped tensor. ``narrow`` does not take it, and a
    host-int start bakes a per-rank constant.
    """
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x
    # 1-D rank tensor (shape (1,)); do NOT reshape to 0-D, which dynamo constant-folds.
    rank_tensor = current_group.rank_in_group_tensor.to(torch.int64)
    if current_dp > target_dp:
        per_dp = x.shape[dim] // current_dp
        length = target_dp * per_dp
        start = (rank_tensor // target_dp) * target_dp * per_dp
        idx = torch.arange(length, device=x.device, dtype=torch.int64) + start
        return x.index_select(dim, idx)
    per_dp = x.shape[dim] // current_dp
    start = rank_tensor * per_dp
    idx = torch.arange(per_dp, device=x.device, dtype=torch.int64) + start
    x = x.index_select(dim, idx)
    return target_group.all_gather(x, dim=dim)


class GptOssDecoderLayer(nn.Module):
    """Decode-only weight holder: input RMSNorm + attention/MLP holders (no forward)."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        # input_layernorm weight is stacked, so no local weight param.
        self.input_layernorm = GptOssRMSNorm(config, has_weight=False)
        self.self_attn = GptOssAttention(config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.layer_idx = layer_idx
        self._hidden_size = config.hidden_size
        self._param_dtype = config.torch_dtype

    def gk_weight_loaders(self) -> dict:
        """Return {stack_key: (loader, per_layer_shape, dtype)} for the input RMSNorm."""
        return {
            "input_layernorm_weight": (
                shuffling_weight_loader(
                    shuffle_dim=0,
                    expected_shard_shape=(self._hidden_size,),
                    shard_spec=None,
                ),
                (self._hidden_size,),
                self._param_dtype,
            )
        }


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Embedding TP group (TP * embedding_dp_size) + DP column <<<
        self.embedding_dp_size = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_embedding_dp_group,
            get_neuron_mlp_dp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        emb_device_group = emb_tp_group.device_group
        self.embedding_tp_group = emb_tp_group
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # >>> PARALLELISM: token embedding <<<
        # GK fused embedding, HIDDEN-dim shard. Each rank holds [vocab, H/tp]. Others shard vocab.
        from vllm_neuron.model.gpt_oss.mega_decode import (
            _gk_fuse_embed_enabled,
            _gk_fuse_embed_hidden_enabled,
            _gk_fuse_rope_enabled,
        )

        # Resolve each fusion switch ONCE, here. forward() and the GK runner both run inside the
        # trace, where an environment read is re-evaluated or breaks the graph.
        self._gk_fuse_embed = _gk_fuse_embed_enabled()
        self._gk_fuse_rope = _gk_fuse_rope_enabled()
        self._gk_fuse_embed_hidden = (
            self._gk_fuse_embed and _gk_fuse_embed_hidden_enabled()
        )
        # One greppable banner for the fusion decision. Dynamo traces forward() and refuses a
        # logging.Logger method inside the graph, so print it here.
        logger.info(
            "[GK] fusions: embed=%s embed_hidden_shard=%s rope=%s embedding_dp_size=%d "
            "flat_row_gather=%s",
            self._gk_fuse_embed,
            self._gk_fuse_embed_hidden,
            self._gk_fuse_rope,
            self.embedding_dp_size,
            envs.VLLM_NEURON_TAIL_FLAT_ROW_GATHER,
        )
        if self._gk_fuse_embed_hidden:
            self.embed_tokens = EmbedDimShardedEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.hidden_size,
                dtype=config.torch_dtype,
                tp_group=emb_device_group,
            )
        else:
            self.embed_tokens = VocabDimShardedEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.hidden_size,
                dtype=config.torch_dtype,
                tp_group=emb_device_group,
            )

        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = GptOssRMSNorm(config)
        self.rotary_emb = GptOssRotaryEmbedding(config)

        # <-- MXFP4: Shuffled weight loaders for embedding and final norm.
        # Shuffle the hidden dim in both cases; the hidden-dim shard shards that same dim.
        if self._gk_fuse_embed_hidden:
            emb_loader = shuffling_weight_loader(
                shuffle_dim=1,
                expected_shard_shape=(
                    config.vocab_size,
                    self.embed_tokens.embed_dim_per_rank,
                ),
                shard_spec=ShardSpec(
                    dim=1,
                    size=self.embed_tokens.embed_dim_per_rank,
                    num_shards=self.embed_tokens.tp_size,
                ),
            )
        else:
            emb_loader = shuffling_weight_loader(
                shuffle_dim=1,
                expected_shard_shape=(
                    self.embed_tokens.vocab_size_per_rank,
                    config.hidden_size,
                ),
                shard_spec=ShardSpec(
                    dim=0,
                    size=self.embed_tokens.vocab_size_per_rank,
                    num_shards=self.embed_tokens.tp_size,
                ),
            )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)
        set_weight_loader(
            self.norm.weight,
            shuffling_weight_loader(
                shuffle_dim=0,
                expected_shard_shape=(config.hidden_size,),
                shard_spec=None,
            ),
        )

        # Decode-only giga-kernel: the fused gpt_oss_decode_mega_kernel runs the
        # whole decoder stack in one lnc1 NEFF, replacing the per-layer decode loop.
        # Prefill runs on the native per-layer model over DI. The runner keys off
        # use_stacked_kv_cache to allocate the per-group KV buffers.
        self.use_gk_decode = bool(
            config.neuron_config and config.neuron_config.gk_decode
        )
        # Count of per-group KV buffers set as K_cache_buf_i / V_cache_buf_i attrs
        # by set_kv_cache_buffers.
        self._gk_kv_buf_count = 0
        from vllm_neuron.model.gpt_oss.mega_decode import MegaDecodeRunner

        self._mega_decode = MegaDecodeRunner(self)
        logger.info(
            "[GK] GPT-OSS decode giga-kernel ENABLED (neuron_config.gk_decode) "
            "on model_mxfp4_perf: decode-only model, fuses all %d layers into one "
            "NEFF; prefill is served by the native per-layer model over DI.",
            config.num_hidden_layers,
        )

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata=None,
        rank=None,
        skip_final_norm=False,
        return_sp_shard=False,
    ):
        # GK FUSED RoPE: the giga-kernel computes cos/sin on-chip, so the host rotary_emb does
        # not run and passes no cos/sin.
        fuse_rope = self._gk_fuse_rope

        if self._gk_fuse_embed:
            # >>> GK FUSED EMBEDDING: the giga-kernel builds layer 0's input, so the host does
            # not compute the hidden state. It supports embedding_dp_size == 1 only.
            assert self.embedding_dp_size == 1, (
                "VLLM_NEURON_MEGA_FUSE_EMBED needs embedding_dp_size == 1; got "
                f"{self.embedding_dp_size}. Set embedding_dp_size to 1, or turn the "
                f"fused embedding off."
            )
            # No log call here: Dynamo refuses a logging.Logger method inside the graph.
            position_embeddings = (
                None
                if fuse_rope
                else self.rotary_emb(
                    positions, device=positions.device, dtype=self.config.torch_dtype
                )
            )
            hidden_states = self._mega_decode.decode(
                None,
                positions,
                position_embeddings,
                attn_metadata,
                input_ids=input_ids,
                rank=rank,
                return_sp_shard=return_sp_shard,
            )
            # Respect skip_final_norm, because the fused sampling tail applies this norm
            # itself, the same as the path below.
            if not skip_final_norm:
                hidden_states = self.norm(hidden_states)
            return hidden_states

        # >>> PARALLELISM: Embedding DP — gather input_ids (tiny, just token IDs) <<<
        if self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        emb_rank = rank
        if rank is not None and self.embedding_dp_size > 1:
            # SPMD: the stamped embedding TP group rank tensor IS the shard index the embedding
            # module expects, because that group spans TP * embedding_dp_size. A Python-int rank
            # bakes a per-rank constant, which splinters the compile cache into one NEFF per rank.
            emb_rank = self.embedding_tp_group.rank_in_group_tensor.to(rank.dtype)
        # Decode is TP-replicated (scatter_tokens=False): the GK kernel wants the
        # full [B_local*S_decode, H] tensor and does its own TP/EP collectives.
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=False, rank=emb_rank
        )

        # >>> PARALLELISM: Transition emb_dp → mlp_dp for decoder layers <<<
        hidden_states = _dp_transition(
            hidden_states, self.embedding_dp_group, self.mlp_dp_group
        )

        position_embeddings = (
            None
            if fuse_rope
            else self.rotary_emb(
                positions, device=hidden_states.device, dtype=hidden_states.dtype
            )
        )

        # >>> GK: whole-stack decode giga-kernel — replaces the per-layer loop.
        hidden_states = self._mega_decode.decode(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            return_sp_shard=return_sp_shard,
        )

        # The fused sampling tail applies this norm itself, on the far side of the row slice,
        # index_select and lm_head-DP gather the caller runs. RMSNorm is row-wise, so moving it
        # past those is arithmetically identical. Skipping it here lets the kernel keep the
        # normed tile in SBUF, instead of round-tripping [T, H] through HBM.
        if not skip_final_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class GptOssForCausalLM(nn.Module):
    """DECODE-ONLY GPT-OSS giga-kernel model (selected when neuron_config.gk_decode
    is set). Prefill is served by the native per-layer model over disaggregated
    inference; speculative decoding / Eagle3 are not supported on the GK path."""

    # The forward assembles the decode input_ids on the device, so the runner may send the
    # in-graph reorder arguments. A model without this attribute keeps the host swap path.
    supports_async_reorder = True

    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config, batch_size)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: LM Head TP group + DP column <<<
        self.lm_head_dp_size = (
            config.neuron_config.lm_head_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_tp_group,
            get_neuron_lm_head_dp_group,
            get_neuron_mlp_dp_group,
        )

        lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = lm_head_tp_group
        lm_head_device_group = lm_head_tp_group.device_group
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()
        lm_head_tp_rank = lm_head_tp_group.rank_in_group
        self.lm_head_tp_rank = lm_head_tp_rank

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # >>> PARALLELISM: Column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=lm_head_device_group,
        )

        # <-- MXFP4: Shuffled weight loader for LM head
        lm_head_loader = shuffling_weight_loader(
            shuffle_dim=1,
            expected_shard_shape=(
                self.lm_head.out_features_per_rank,
                config.hidden_size,
            ),
            shard_spec=ShardSpec(
                dim=0,
                size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
            ),
        )
        lm_head_loader = with_rank_override(lm_head_loader, rank=lm_head_tp_rank)
        set_weight_loader(self.lm_head.weight, lm_head_loader)

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
            )

        # >>> FUSED SAMPLING TAIL: norm + lm_head + top-k + sampler in NKI <<<
        # Everything the eligibility check needs, resolved once here. Only the LENGTH of the ranks
        # tuple reaches the kernel. The kernel derives each candidate's global vocab id from the
        # position in the hop mesh, so the shard order is a contract of the process group itself.
        self.tail_kernel_tp_ranks = tuple(lm_head_tp_group.ranks)
        self.tail_kernel_v_s = self.lm_head.out_features_per_rank
        self.tail_kernel_k = (
            self.on_device_sampling_config.max_top_k
            if self.on_device_sampling_config is not None
            else 0
        )
        # The parallelism degrees the gate reads. The lm_head TP group is BUILT as base TP x
        # lm_head_dp_size, so its size alone cannot tell TP8 x DP16 from TP4 x DP32 -- the two
        # also agree on the rank list, v_s and the row count.
        self.tail_kernel_tp = self.tp_group.world_size
        self.tail_kernel_embedding_dp = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        # Read the RESOLVED degree, not the requested one. `resolve_ep_degree` rewrites a requested
        # 1 to TP*DP for the process groups when --enable-expert-parallel is set, but
        # NeuronConfig keeps the raw field. The documented way to run the target sets no explicit
        # ep_degree, so the raw field is 1 there while the effective degree is 128.
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_degree

        self.tail_kernel_ep = get_neuron_ep_degree()
        # The all-to-all decomposition, whose hop sizes must multiply to the lm_head TP size. A
        # deployment selects it with neuron_config.gpt_oss_tail_hops. The (8, 16) default is the only
        # tuple the gate admits and the measured one. (128,) is refused by the gate and falls back to torch.
        self.tail_kernel_hops = tuple(config.neuron_config.gpt_oss_tail_hops)
        self.tail_identity_rows = self._tail_identity_rows_ok()
        # [H, V_s] copy of the lm_head shard. The chunked kernel takes lm_head_weight_t as [H, V_s],
        # so the copy is REQUIRED, and only on-device sampling gates it. Reading that layout from the [V_s, H]
        # shard instead is a transposing load whose contiguous run is one element, paid on every
        # decode step. It costs V_s * H * 2 bytes, measured as 9.205 MiB for each rank.
        self.register_buffer("tail_kernel_weight_hv", None, persistent=False)

    def _tail_identity_rows_ok(self) -> bool:
        """Host-side mesh check for the replica-local tail forms.

        All of these must hold, and each is a structural property of the standard mesh, so the
        answer is identical on every rank (a per-rank difference would trace two graphs):
          * ``tail_kernel_hops == (world,)``: one hop, ``owned_row_block`` is the identity, so lm_head
            TP rank ``r`` owns global rows ``[r*R, (r+1)*R)``;
          * the lm_head TP group is rank-ordered ``0..world-1`` (group rank == global rank);
          * this rank's TP group is a contiguous, aligned block ``[tp*k, tp*(k+1))`` with this rank at
            position ``s`` (so global row ``tp*k + s`` is local row ``s`` of replica ``k``);
          * the lm_head DP column is ``{s + tp*d}`` with this rank at position ``k`` (the row order the
            gathered path relies on), and ``lm_head_dp_size == world // tp``.
        Logged on every rank so a serve log witnesses the decision.
        """
        try:
            lm_ranks = tuple(int(r) for r in self.lm_head_tp_group.ranks)
            tp_ranks = tuple(int(r) for r in self.tp_group.ranks)
            dp_ranks = tuple(int(r) for r in self.lm_head_dp_group.ranks)
            lm_rank = int(self.lm_head_tp_group.rank_in_group)
            tp_rank = int(self.tp_group.rank_in_group)
            dp_rank = int(self.lm_head_dp_group.rank_in_group)
        except (AttributeError, TypeError):
            logger.info(
                "[gpt_oss_tail] identity rows=False (groups expose no rank lists)"
            )
            return False
        world, tp = len(lm_ranks), len(tp_ranks)
        reason = None
        if world == 0 or tp == 0 or world % tp:
            reason = f"world={world} not a multiple of tp={tp}"
        elif tuple(self.tail_kernel_hops) != (world,):
            reason = (
                f"hops={tuple(self.tail_kernel_hops)} is not the one-hop map ({world},)"
            )
        elif lm_ranks != tuple(range(world)):
            reason = "lm_head TP group is not rank-ordered 0..world-1"
        elif self.lm_head_dp_size != world // tp:
            reason = f"lm_head_dp_size={self.lm_head_dp_size} != world/tp={world // tp}"
        else:
            k, s = lm_rank // tp, lm_rank % tp
            if tp_ranks != tuple(range(tp * k, tp * (k + 1))) or tp_rank != s:
                reason = f"TP group {tp_ranks[:2]}.. is not the aligned block of rank {lm_rank}"
            elif (
                dp_ranks != tuple(s + tp * d for d in range(world // tp))
                or dp_rank != k
            ):
                reason = (
                    f"lm_head DP column {dp_ranks[:2]}.. is not the TP-position column"
                )
        logger.info(
            "[gpt_oss_tail] identity rows=%s hops=%s world=%d tp=%d lm_head_dp=%d%s",
            reason is None,
            tuple(self.tail_kernel_hops),
            world,
            tp,
            self.lm_head_dp_size,
            "" if reason is None else f" reason={reason}",
        )
        return reason is None

    # >>> GK: whole-stack decode giga-kernel capability flag <<<
    # The runner reads this to compile with --logical-nc-config=1 and to allocate the per-group
    # KV buffers. Delegates to the inner GptOssModel.
    @property
    def use_stacked_kv_cache(self) -> bool:
        return getattr(self.model, "use_gk_decode", False)

    def set_kv_cache_buffers(self, k_bufs, v_bufs):
        """Store the per-group KV buffers as individual ``K_cache_buf_i`` / ``V_cache_buf_i`` attributes.
        The fork-child meta-swap walks tensor attributes and does not recurse into a list.
        """
        self.model._gk_kv_buf_count = len(k_bufs)
        for i, (k, v) in enumerate(zip(k_bufs, v_bufs)):
            setattr(self.model, f"K_cache_buf_{i}", k)
            setattr(self.model, f"V_cache_buf_{i}", v)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        logit_mask=None,
        rank=None,
        **kwargs,  # absorb runner-passed args this decode-only path ignores
        # (spec_decode_metadata=None, inputs_embeds/is_token_ids/rotary_* — no
        # prefill, prompt-embeds, or speculative decoding on the GK decode path).
    ):
        # Assemble the decode input_ids on the device.
        #
        # The runner gives the device future of the previous step, plus a permutation and a
        # mask that it builds on the host from the request ids. A request that continues takes
        # the reordered future. A new request, a resumed request, a row that still prefills,
        # and a padded slot take the host input_ids. The host therefore does not read the
        # sampled tokens of the previous step back.
        #
        # The arguments are absent on a step that does not use the reorder, and this block is
        # then a no-op. Remove all three arguments in every case, so a later consumer of
        # ``**kwargs`` never sees a partial set.
        #
        # Keep this block identical in every model that declares supports_async_reorder.
        # test_async_reorder.py compares them and fails on a divergence.
        async_reorder_perm = kwargs.pop("async_reorder_perm", None)
        async_reorder_future = kwargs.pop("async_reorder_future", None)
        async_reorder_mask = kwargs.pop("async_reorder_mask", None)
        if async_reorder_perm is not None:
            input_ids = torch.where(
                async_reorder_mask,
                input_ids,
                torch.index_select(async_reorder_future, 0, async_reorder_perm),
            )

        positions = positions.to(torch.int32)

        # >>> FUSED SAMPLING TAIL: decide before the stack runs <<<
        # The kernel norms its own input, so "skip the final norm" and "call the kernel" are one
        # decision. ONE variable, tail_mode, carries it to the three sites that must agree:
        # skip_final_norm, which entry point runs, and whether the framework still samples.

        # is_prefill=False and spec_decode_metadata=None are structural here, not runtime facts.
        # This module is the decode-only giga-kernel model, and its forward absorbs and ignores
        # both. So the guards the shared model needs are already met.
        tail_mode = resolve_tail_mode()
        if (
            gpt_oss_tail_precheck(
                is_prefill=False,
                sampler=getattr(self, "sampler", None),
                on_device_sampling_config=self.on_device_sampling_config,
                sampling_params=sampling_params,
                logit_mask=logit_mask,
                spec_decode_metadata=None,
                rank=rank,
                mode=tail_mode,
                gather_logits=self._gather_logits,
            )
            is not None
        ):
            tail_mode = TailMode.OFF
        if tail_mode is not TailMode.OFF and (
            gpt_oss_tail_reject_reason(
                probe=self.lm_head.weight,
                n_rows=sampling_positions.shape[0] * self.lm_head_dp_size,
                hidden=self.config.hidden_size,
                v_s=self.tail_kernel_v_s,
                k=self.tail_kernel_k,
                world=len(self.tail_kernel_tp_ranks),
                hidden_actual=self.config.unpadded_hidden_size,
                tp=self.tail_kernel_tp,
                lm_head_dp=self.lm_head_dp_size,
                embedding_dp=self.tail_kernel_embedding_dp,
                ep=self.tail_kernel_ep,
                hops=self.tail_kernel_hops,
                lm_head_tp_ranks=self.tail_kernel_tp_ranks,
                mode=tail_mode,
            )
            is not None
        ):
            tail_mode = TailMode.OFF

        # >>> FLAT ROW GATHER: one all_gather over the lm_head TP group <<<
        # Needs mlp_dp_size == 1 and sampling_positions identity over the local rows.
        flat_row_gather = (
            envs.VLLM_NEURON_TAIL_FLAT_ROW_GATHER
            and tail_mode is TailMode.CHUNKED
            and self.mlp_dp_group.world_size == 1
            and self.lm_head_dp_size > 1
        )
        tail_local_rows = (
            flat_row_gather
            and getattr(self, "tail_identity_rows", False)
            and self.on_device_sampling_config is not None
            and sampling_params is not None
            and int(sampling_params.shape[0]) == int(sampling_positions.shape[0])
        )
        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            skip_final_norm=tail_mode.skips_model_final_norm,
            return_sp_shard=flat_row_gather,
        )
        _glue_zero = tail_glue_anchor_zero(hidden_states) if tail_local_rows else None
        if _glue_zero is not None:
            sampling_params = sampling_params + _glue_zero.to(sampling_params.dtype)

        if flat_row_gather:
            # hidden_states is this rank's [gbs/world, H] SP shard. ONE gather gives [gbs, H],
            # already in batch order, so the mlp-DP slice and the index_select are both skipped.
            gbs = sampling_positions.shape[0] * self.lm_head_dp_size
            world = self.lm_head_tp_group.world_size
            assert hidden_states.shape[0] * world == gbs, (
                f"Flat row gather needs the SP shard to hold gbs/world = {gbs // world} rows, "
                f"got {hidden_states.shape[0]}. Set VLLM_NEURON_TAIL_FLAT_ROW_GATHER=0."
            )
            if tail_local_rows:
                assert (
                    gbs // world
                ) * self.tp_group.world_size == sampling_positions.shape[0], (
                    f"identity rows need tp * gbs/world == local rows, got "
                    f"{self.tp_group.world_size} * {gbs // world} != {sampling_positions.shape[0]}"
                )
            hidden_states_for_logits = self.lm_head_tp_group.all_gather(
                hidden_states, dim=0
            )
            # The metadata still needs the lm_head-DP gather unless the kernel only reads this replica's
            # rows: it is rank-local request state, so no rank can build the other replicas' rows without it.
            if self.on_device_sampling_config is not None:
                if logit_mask is not None:
                    logit_mask = self.lm_head_dp_group.all_gather(logit_mask, dim=0)
                if sampling_params is not None and not tail_local_rows:
                    sampling_params = self.lm_head_dp_group.all_gather(
                        sampling_params, dim=0
                    )
        else:
            # >>> PARALLELISM: Slice to local before sampling position selection <<<
            mlp_dp = self.mlp_dp_group.world_size
            if mlp_dp > 1:
                local_size = hidden_states.shape[0] // mlp_dp
                dp_rank = self.mlp_dp_group.rank_in_group
                hidden_states = hidden_states[
                    dp_rank * local_size : (dp_rank + 1) * local_size
                ]

            hidden_states_for_logits = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

            # >>> PARALLELISM: Transition local → lm_head_dp <<<
            if self.lm_head_dp_size > 1:
                hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                    hidden_states_for_logits, dim=0
                )
                if self.on_device_sampling_config is not None:
                    # ODS samples on the gathered lm_head_dp row layout. Keep
                    # per-row inputs aligned with logits until sampling finishes.
                    if logit_mask is not None:
                        if logit_mask.shape[0] == sampling_positions.shape[0]:
                            logit_mask = self.lm_head_dp_group.all_gather(
                                logit_mask, dim=0
                            )
                        elif logit_mask.shape[0] != hidden_states_for_logits.shape[0]:
                            raise ValueError(
                                "logit_mask rows must match either local or "
                                "lm_head_dp-gathered sampling rows. "
                                f"Got logit_mask={logit_mask.shape}, "
                                f"local_rows={sampling_positions.shape[0]}, "
                                f"gathered_rows={hidden_states_for_logits.shape[0]}."
                            )
                    if sampling_params is not None:
                        sampling_params = self.lm_head_dp_group.all_gather(
                            sampling_params, dim=0
                        )

        # On the fused path hidden_states_for_logits stays un-normed and uncast: the kernel norms in
        # fp32, so the norm INPUT keeps the precision it arrived with.

        # ONE rank value for every branch below. Read the stamped rank tensor of the lm_head TP
        # group. This tensor is the shard index. A Python-int delta bakes a per-rank constant, which
        # SPLINTERS the compile cache into one NEFF for each rank instead of collapsing to one.
        # RankConstantCheckPass raises on such a delta if NEURON_LIBTORCH_SPMD_STRICT=1.
        lm_head_rank = (
            None
            if rank is None
            else self.lm_head_tp_group.rank_in_group_tensor.to(rank.dtype)
        )
        if _glue_zero is not None and lm_head_rank is not None:
            lm_head_rank = lm_head_rank + _glue_zero.to(lm_head_rank.dtype)
        _local_rows_rank = None
        if tail_local_rows and rank is not None:
            _local_rows_rank = self.tp_group.rank_in_group_tensor.to(rank.dtype)
            if _glue_zero is not None:
                _local_rows_rank = _local_rows_rank + _glue_zero.to(
                    _local_rows_rank.dtype
                )
        if tail_mode is TailMode.CHUNKED:
            # The kernel shards batch ROWS, so this rank samples only the rows it owns and returns
            # no logits. lm_head_weight_hv is the [H, V_s] mirror the kernel requires, and
            # lm_head_rank is a TENSOR: one NEFF runs on every rank, so the row slice inside the
            # kernel wrapper must read the rank at runtime.
            owned_tokens = gpt_oss_chunked_tail(
                # Read the sampler's own flag rather than promising a draw separately: a
                # promise can stay true after the append is removed or bypassed.
                deterministic=self.sampler.deterministic,
                hidden_states=hidden_states_for_logits,
                gamma=self.model.norm.weight,
                lm_head_weight_hv=self.tail_kernel_weight_hv,
                sampling_params=sampling_params,
                rank=lm_head_rank,
                lm_head_tp_ranks=self.tail_kernel_tp_ranks,
                hops=self.tail_kernel_hops,
                k=self.tail_kernel_k,
                eps=self.config.rms_norm_eps,
                hidden_actual=self.config.unpadded_hidden_size,
                zero_interleaved_pad=True,
                local_rows_rank=_local_rows_rank,
            )
            if tail_local_rows:
                sampled_tokens = self.tp_group.all_gather(owned_tokens, dim=0)
            else:
                # The exchange puts the batch back together. all_gather concatenates in RANK order,
                # which is NOT batch order for a multi-hop mesh, so index_select de-permutes it. The
                # index depends on (VTP, gbs, hops) only, so it is the same constant on every rank.
                gathered = self.lm_head_tp_group.all_gather(owned_tokens, dim=0)
                sampled_tokens = chunked_tail_gather(
                    gathered,
                    len(self.tail_kernel_tp_ranks),
                    int(sampling_params.shape[0]),
                    self.tail_kernel_hops,
                )
            # No logits exist on this path. The precheck refuses the tail when the caller needs
            # them (needs-logits) or when on-device sampling is off, so nothing below reads them.
            logits = None
        else:
            hidden_states_for_logits = hidden_states_for_logits.to(
                self.config.torch_dtype
            )
            logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        # >>> PARALLELISM: Slice back to local batch <<<
        if self.lm_head_dp_size > 1:
            B_local = sampling_positions.shape[0]
            # SPMD: index_select off the stamped rank tensor, for the same reason as the sampled
            # token slice below. The fused tail needs on-device sampling, so these two branches run
            # only with the tail off, and a host-int start there still bakes one replica's rows.
            # Build the index per branch: on the fused path logits is None and has no device.
            _dp_off = (
                self.lm_head_dp_group.rank_in_group_tensor.to(torch.int64) * B_local
            )
            if self.on_device_sampling_config is None:
                logits = logits.index_select(
                    0,
                    torch.arange(B_local, device=logits.device, dtype=torch.int64)
                    + _dp_off,
                )
            if gathered_logits is not None:
                gathered_logits = gathered_logits.index_select(
                    0,
                    torch.arange(
                        B_local, device=gathered_logits.device, dtype=torch.int64
                    )
                    + _dp_off,
                )

        if self.on_device_sampling_config is None:
            return logits

        if tail_mode.framework_samples:
            sampled_tokens = self.sampler(
                logits, sampling_params, logit_mask=logit_mask, tp_rank=lm_head_rank
            )
        if self.lm_head_dp_size > 1 and not tail_local_rows:
            # Slice sampled tokens after sampler; slicing logits earlier would
            # desynchronize SO masks from lm_head_dp-gathered logits.
            # SPMD: index_select with the stamped rank tensor. A Python-int start bakes a per-rank
            # constant, which splinters the compile cache into one NEFF per rank. A slice start
            # cannot take the 1-D stamped tensor; index_select can, so it is the form used here.
            B_local = sampling_positions.shape[0]
            # 1-D rank tensor (shape (1,)); do NOT reshape to 0-D, which dynamo constant-folds.
            _dp_offset_s = (
                self.lm_head_dp_group.rank_in_group_tensor.to(torch.int64) * B_local
            )
            _idx_s = (
                torch.arange(B_local, device=sampled_tokens.device, dtype=torch.int64)
                + _dp_offset_s
            )
            sampled_tokens = sampled_tokens.index_select(0, _idx_s)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = GptOssConfig.from_configs(hf_config, neuron_config)
        return cls(config, batch_size=1)

    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<
    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=layer.self_attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Packed FP8 K cache is one rank higher than V (never packed); see
            # model_bf16.bind_kv_cache. The hasattr guard keeps this safe for any
            # attention class that predates the flag.
            if hasattr(layer.self_attn, "fp8_packed"):
                layer.self_attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1

            # Active-KV FP8 invariant: with an FP8 K cache the kernel quantizes the
            # active K/V only if k_scale/v_scale are non-None. The GK path opts into
            # fp8 independently of cache_config (which loads scales), so backfill the
            # identity scale (exact for direct_cast; never overrides loaded scales).
            attn = layer.self_attn
            if (
                getattr(attn, "fp8_packed", False)
                and k_cache.dtype == torch.float8_e4m3fn
            ):
                if getattr(attn, "k_scale", None) is None:
                    attn.k_scale = torch.ones(
                        1, dtype=torch.bfloat16, device=k_cache.device
                    ).reshape(1, 1)
                    attn.k_scale_float = 1.0
                if getattr(attn, "v_scale", None) is None:
                    attn.v_scale = torch.ones(
                        1, dtype=torch.bfloat16, device=k_cache.device
                    ).reshape(1, 1)
                    attn.v_scale_float = 1.0

    def load_weights(self, checkpoint_path, device, cache_dir):
        """Load MXFP4 weights from checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        <-- MODEL-SPECIFIC: HF checkpoint key → model parameter mappings
        <-- MXFP4: Expert weights map to separate blocks/scales checkpoint keys
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = dict()
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # Attention
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.qkv_proj_bias"] = [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{prefix}.self_attn.o_proj_bias"] = (
                f"{prefix}.self_attn.o_proj.bias"
            )
            mappings[f"{prefix}.self_attn.sinks"] = f"{prefix}.self_attn.sinks"
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )

            # MoE
            mappings[f"{prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.experts.router_weight"] = (
                f"{prefix}.mlp.router.weight"
            )
            mappings[f"{prefix}.mlp.experts.router_bias"] = f"{prefix}.mlp.router.bias"

            # <-- MXFP4: Separate blocks and scales checkpoint keys
            mappings[f"{prefix}.mlp.experts.gate_up_proj_weight"] = (
                f"{prefix}.mlp.experts.gate_up_proj_blocks"
            )
            mappings[f"{prefix}.mlp.experts.gate_up_proj_scale"] = (
                f"{prefix}.mlp.experts.gate_up_proj_scales"
            )
            mappings[f"{prefix}.mlp.experts.gate_up_proj_bias"] = (
                f"{prefix}.mlp.experts.gate_up_proj_bias"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_weight"] = (
                f"{prefix}.mlp.experts.down_proj_blocks"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_scale"] = (
                f"{prefix}.mlp.experts.down_proj_scales"
            )
            mappings[f"{prefix}.mlp.experts.down_proj_bias"] = (
                f"{prefix}.mlp.experts.down_proj_bias"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)

        # Load the tiny KV-cache scales FIRST. The stacked W_out loader folds
        # 1/v_scale per layer, so v_scale_float must be set before the stacks
        # load. _load_kv_cache_scales reads slices directly, so index first.
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

        # Declare each [L, ...] stacked weight as a temporary parameter with its
        # own loader (load_sharded_pipelined only loads named parameters), so the
        # framework reads the checkpoint keys and copies each stack to device once.
        specs = self.model._mega_decode.build_stacked_weight_specs(mappings)
        stacked_mappings = dict()
        for spec in specs:
            param = nn.Parameter(
                torch.empty(spec.shape, dtype=spec.dtype, device="meta"),
                requires_grad=False,
            )
            set_weight_loader(param, spec.loader)
            setattr(self.model, spec.attr, param)
            stacked_mappings[f"model.{spec.attr}"] = spec.ckpt_keys

        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, {**mappings, **stacked_mappings}, device
        ).state_dict

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Republish each loaded stack as a BUFFER (counted by model.buffers() for
        # the KV budget). Use a buffer, NOT nn.Parameter: the trace fork's meta-swap
        # binds a Parameter stack wrong (device out-of-bounds), a buffer correctly.
        for spec in specs:
            loaded = getattr(self.model, spec.attr).detach()
            delattr(self.model, spec.attr)
            self.model.register_buffer(spec.attr, loaded)

        self._maybe_materialize_tail_weight_hv()

    def _maybe_materialize_tail_weight_hv(self) -> None:
        """Hold the [H, V_s] copy of the lm_head shard that the chunked tail's GEMM needs.

        The chunked kernel takes ``lm_head_weight_t`` as ``[H, V_s]``, so this copy is REQUIRED
        rather than an optimisation, and no LAYOUT predicate and no flag gate it. It costs
        ``V_s * H * 2`` bytes for each rank on top of the shard, measured as 9.205 MiB. Take it
        once here, because the shard is constant across decode steps -- which is the whole reason
        the transpose is worth hoisting out of the per-step load.

        ONE load-time condition skips it: on-device sampling is off, which the precheck refuses
        too, so the tail cannot run. The kill switch is deliberately NOT a condition. It is an env
        var, and this method reads it at LOAD while the gate reads it again at TRACE, so a 0 at
        load and a 1 at trace would leave this copy None and the wrapper would fail on ``.shape``.
        The geometry checks stay in the gate, so at an unsupported shape the first forward refuses
        the tail and this copy goes unread.
        """
        if self.on_device_sampling_config is None:
            return
        w = self.lm_head.weight.detach()
        if w.is_meta:
            # The CPU-compile path builds the model on the meta device and load_weights_lite
            # never populates lm_head.weight, so a host round-trip raises "Cannot copy out of
            # meta tensor". Tracing needs the shape and the dtype, never the values.
            self.tail_kernel_weight_hv = torch.empty(
                w.shape[1], w.shape[0], dtype=torch.bfloat16, device=w.device
            )
        else:
            # Transposed on the HOST and copied back, not in place on the device: the neuron
            # device's copy_ rejects this, and .t() alone is a stride view that moves no bytes,
            # so it would leave the load exactly as strided. .contiguous() is what makes V_s the
            # contiguous axis. The kernel reads the weight as bf16, so cast here.
            self.tail_kernel_weight_hv = (
                w.cpu().t().contiguous().to(torch.bfloat16).to(w.device)
            )
        # A strided copy is numerically identical and slower to load: the DMA descriptor is built
        # from strides. test_gpt_oss_tail_chunked_wiring.py pins that the kill switch does not gate
        # this copy, because the switch is read here at load and again by the gate at trace.
        assert self.tail_kernel_weight_hv.is_contiguous(), (
            "the [H, V_s] lm_head copy must be contiguous; a .t() view would be strided"
        )
        logger.info(
            "[gpt_oss_tail] hv lm_head copy materialized: %s on %s "
            "(%.1f MiB/rank on top of the shard)",
            tuple(self.tail_kernel_weight_hv.shape),
            self.tail_kernel_weight_hv.device,
            self.tail_kernel_weight_hv.numel()
            * self.tail_kernel_weight_hv.element_size()
            / 2**20,
        )

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile.

        Loads ONLY the tiny KV-cache quantization scales (not the full MXFP4
        weights) so the traced graph bakes the correct ``softmax_scale`` /
        ``W_out`` scale constants instead of the 1.0 defaults. Expert weights
        and their scales enter the graph as parameters (not Python-float
        constants), so they are not needed at trace time.
        """
        # The layout is baked in at TRACE time, so the [H, V_s] copy must exist on this path too.
        # Without it the graph is traced against the [V_s, H] shard and the flag is silently
        # inert for the whole run. ABOVE the early return below, which fires whenever the KV
        # cache is unquantized and would otherwise skip it.
        self._maybe_materialize_tail_weight_hv()

        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        # Scales only exist for quantized KV caches; nothing to load otherwise.
        # Skipping the checkpoint I/O here lets CPU compile run with
        # skip_weights (no safetensors on disk) when the KV cache is unquantized.
        if (
            get_current_vllm_config().cache_config.cache_dtype
            not in QUANTIZED_KV_CACHE_DTYPES
        ):
            return
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # load_weights() populates the tensor index via load_sharded_pipelined;
        # the lite path skips that, so index explicitly or every key lookup
        # below misses and silently leaves the scales at 1.0.
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    # Invert scales: checkpoint stores scales for (tensor / scale),
                    # but the kernel quantizes via (tensor * scale), so invert once here.
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
