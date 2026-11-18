# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 BF16 Implementation
==============================

Implementation of EXAONE 4.0 for the Neuron backend. The structure comes
from the Qwen3 dense model (GQA, per-head QK-norm, SiLU MLP).

Supported parallelism: TP, SP (prefill).
Reference model: LGAI-EXAONE/EXAONE-4.0-32B (dense, 40Q/8KV GQA, head_dim=128).

Differences from Qwen3:
  - Post-norm decoder layer. There is no input norm. The layer normalizes the
    attention output and the MLP output before each residual add.
  - Hybrid attention. A sliding-window layer attends to the last 4096 keys.
    A full layer attends to all keys.
  - RoPE applies only on the sliding-window layers. The full layers use no
    position embedding (NoPE). This port gives the identity rotation
    (cos=1, sin=0) to the same fused kernels, so both layer types use one path.
  - Llama3 RoPE scaling.
  - The per-rank MLP intermediate size (27392 / 8 = 3424) is zero-padded to a
    multiple of 128 (3456). The zero columns and rows add exactly zero.

Dispatch evidence:
  Each layer writes one ``[exaone4_dispatch]`` log line for each main
  component at trace time. The line gives the result of the same eligibility
  predicate that the NF wrapper uses, with the same arguments. The functional
  device test reads these lines from each server log.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    EXAONE 4.0 specific. Change when porting.
"""

import importlib
import logging
import math

import torch
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import NormType, QKVOutputLayout
from vllm_neuron.functional.attention import attention_cte as _attn_cte
from vllm_neuron.functional.attention import attention_decode as _attn_decode
from vllm_neuron.functional.attention import attention_decode_mask as _attn_mask
from vllm_neuron.functional.attention import attention_segmented_cte as _attn_seg
from vllm_neuron.functional.attention import o_proj as _o_proj
from vllm_neuron.functional.attention import qkv as _qkv
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import (
    build_per_head_block_table,
    paged_slot_indices,
    scatter_new_kv,
)
from .config import FULL_ATTENTION, SLIDING_ATTENTION, Exaone4Config

logger = logging.getLogger(__name__)

# vllm_neuron.functional exports the function ``mlp``, and the function hides
# the module of the same name. Get the module from the import system.
_mlp = importlib.import_module("vllm_neuron.functional.mlp")


def _wait_collective(tensor: torch.Tensor) -> torch.Tensor:
    """Return a plain tensor for a functional collective output.

    In CPU mode, a collective gives a lazy AsyncCollectiveTensor. A view op
    keeps the subclass, and the NKI kernel HOP has no rule for it. Wait for the
    value before a kernel call. On device, the collective gives a plain tensor,
    and this function then does nothing.

    Example:
        >>> X = _wait_collective(self.tp_group.all_gather(X, dim=0))
    """
    if isinstance(tensor, AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor


# Pad the per-rank MLP intermediate size to this multiple.
_MLP_INTERMEDIATE_ALIGN = 128

DISPATCH_LOG_TAG = "[exaone4_dispatch]"


# =============================================================================
# Section 0: Dispatch evidence
# =============================================================================


# Keep this function behind assume_constant_result. A bare logger.info in a
# traced region is a hard Dynamo failure. The decorator runs the call off-graph
# one time at trace time.
@torch._dynamo.assume_constant_result
def _log_dispatch(
    phase: str,
    component: str,
    layer: int,
    layer_type: str,
    rank: int,
    kernel: bool,
    detail: str,
) -> None:
    """Write one dispatch line for a main component at trace time.

    Example:
        >>> _log_dispatch("decode", "mlp", 0, "sliding_attention", 0, True, "T=4")
    """
    logger.info(
        "%s phase=%s component=%s layer=%d layer_type=%s rank=%d kernel=%s detail=%s",
        DISPATCH_LOG_TAG,
        phase,
        component,
        layer,
        layer_type,
        rank,
        kernel,
        detail,
    )


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class Exaone4RMSNorm(nn.Module):
    """RMS normalization in FP32, with a learned scale."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding
# <-- MODEL-SPECIFIC: Llama3 piecewise scaling (factor 16, original 8192).
# =============================================================================


class Exaone4RotaryEmbedding(nn.Module):
    """RoPE with optional Llama3 frequency scaling."""

    def __init__(self, config: Exaone4Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        scaling = config.rope_scaling or {}
        self.rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if self.rope_type == "llama3":
            self.scaling_factor = scaling.get("factor", 1.0)
            self.low_freq_factor = scaling.get("low_freq_factor", 1.0)
            self.high_freq_factor = scaling.get("high_freq_factor", 1.0)
            self.orig_max_position = scaling.get(
                "original_max_position_embeddings", 8192
            )
        elif self.rope_type not in (None, "default"):
            raise NotImplementedError(
                f"rope_type '{self.rope_type}' is not supported for EXAONE 4.0. "
                "Use 'default' or 'llama3'."
            )

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        if self.rope_type == "llama3":
            inv_freq = self._apply_llama3_scaling(inv_freq)
        return inv_freq

    def _apply_llama3_scaling(self, inv_freq: torch.Tensor) -> torch.Tensor:
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor
        wave_len = 2 * math.pi / inv_freq

        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = torch.zeros_like(wave_len)

        return torch.where(
            wave_len < high_freq_wavelen,
            inv_freq,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freq / self.scaling_factor,
                (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
            ),
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class Exaone4Attention(nn.Module):
    """Multi-head attention with TP head sharding and per-head QK-norm.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather → QKV → norm → RoPE → attn → O proj → reduce-scatter
    - Decode: fused megakernel

    <-- MODEL-SPECIFIC:
    - GQA: 40 Q heads / 8 KV heads (32B)
    - Per-head QK-norm before RoPE on every layer
    - Sliding-window layers use RoPE and a window of ``sliding_window`` keys
    - Full layers use NoPE. The caller gives the identity rotation.
    """

    def __init__(self, config: Exaone4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # <-- MODEL-SPECIFIC: hybrid attention type of this layer
        self.is_sliding = config.is_sliding(layer_idx)
        self.layer_type = SLIDING_ATTENTION if self.is_sliding else FULL_ATTENTION
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Head sharding <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = self.num_attention_heads * self.head_dim // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # <-- MODEL-SPECIFIC: Per-head QK-norm (on head_dim)
        self.q_norm = Exaone4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Exaone4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of QKV and O-proj weights. <<<"""
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_attention_heads * self.head_dim // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def _log(self, phase: str, component: str, kernel: bool, detail: str) -> None:
        _log_dispatch(
            phase,
            component,
            self.layer_idx,
            self.layer_type,
            self.rank,
            kernel,
            detail,
        )

    # ── Forward dispatch ──────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """>>> PARALLELISM: Dispatch to prefill or decode. <<<"""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            # Layer 0 receives the embedding all-reduce output directly.
            return self.forward_decode(
                _wait_collective(hidden_states),
                positions,
                position_embeddings,
                attn_metadata,
                attention_mask=attention_mask,
            )
        # >>> PARALLELISM: All-gather from SP <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        hidden_states = _wait_collective(hidden_states)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Prefill path ──────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: Fused QKV projection + pre-RoPE QK-norm + RoPE.
        # <-- MODEL-SPECIFIC: a full layer receives cos=1, sin=0 (NoPE).
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)  # [1, T, Dh]
        sin_cache = sin.unsqueeze(0)  # [1, T, Dh]
        hidden_3d = hidden_states.unsqueeze(0)

        self._log(
            "prefill",
            "qkv_proj",
            _qkv._can_use_qkv_kernel(
                hidden=hidden_3d,
                qkv_weights=self.qkv_proj_weight,
                mlp_prev=None,
                attn_prev=None,
                d_head=self.head_dim,
                output_layout=QKVOutputLayout.BSD,
                norm_type=NormType.NO_NORM,
                norm_b=None,
                fused_rope=True,
            ),
            f"T={tokens} H={hidden} qkv={tuple(self.qkv_proj_weight.shape)}",
        )
        qkv = NF.qkv_proj(
            hidden=hidden_3d,
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, -1),
            qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, -1),
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Reshape to per-head layout: [T, Nh, Dh] → [Nh, T, Dh]
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2: Update KV cache
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        # FP8 KV cache quantize-on-write. Only kv_cache_dtype=fp8 uses it.
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.v_cache.dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # Step 3: Attention.
        # <-- MODEL-SPECIFIC: sliding window on sliding layers, causal on full layers.
        if kv_segment_size:
            self._log(
                "prefill",
                "segmented_attention",
                _attn_seg._can_use_segmented_kernel(
                    q,
                    kv_segment_size=kv_segment_size,
                    block_size=block_size,
                    tp_q=True,
                    sliding_window=self.sliding_window,
                ),
                f"q={tuple(q.shape)} kv_segment_size={kv_segment_size} "
                f"block_size={block_size} window={self.sliding_window}",
            )
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=True,
                tp_out=True,
            )
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v

            self._log(
                "prefill",
                "flash_attention",
                _attn_cte._can_use_flash_attention_kernel(v=v_flash),
                f"v={tuple(v_flash.shape)} window={self.sliding_window}",
            )
            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=False,
                tp_out=True,
            )

        # Step 4: Output projection
        attn_output = attn_output.unsqueeze(0)
        self._log(
            "prefill",
            "o_proj",
            _o_proj._can_use_kernel(attn_output, self.o_proj_weight),
            f"active={tuple(attn_output.shape)} w={tuple(self.o_proj_weight.shape)}",
        )
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ───────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        attention_mask: torch.Tensor | None = None,
    ):
        """>>> PARALLELISM: Fused megakernel with TP. <<<

        The kernel fuses the QKV projection, the QK-norm, RoPE, attention and
        the output projection. The model gives a precomputed mask: a window
        mask for a sliding layer, or a causal mask for a full layer.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"Decode token count ({tokens}) is not a multiple of the batch size "
            f"({B}). Make the decode bucket a multiple of the batch size."
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # <-- MODEL-SPECIFIC: a full layer receives cos=1, sin=0 (NoPE).
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        if attention_mask is None:
            # Build a causal mask, with the window start on a sliding layer.
            assert positions is not None
            pos_ids = positions.view(1, B * S_decode)
            start_pos = None
            if self.sliding_window is not None:
                start_pos = torch.clamp(pos_ids - self.sliding_window + 1, min=0).to(
                    torch.float32
                )
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B,
                q_head=self.num_attention_heads_per_rank,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=start_pos,
                block_len=block_size,
            )

        if nkh > 1:
            # GQA (nkh > 1) reaches the fused decode megakernel only with a
            # per-head 3D block table and the 4D cache.
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            k_cache = self.k_cache
            v_cache = self.v_cache
        else:
            # nkh == 1 (TP >= num_kv_heads): squeeze the head axis for the
            # kernel's 3D cache and 2D table.
            k_cache = (
                self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
            )
            v_cache = (
                self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
            )
            active_blocks_table = block_table.to(torch.int32)

        self._log(
            "decode",
            "attention_decode",
            _attn_decode._can_use_attention_block_kernel(
                X=X,
                V_cache=v_cache,
                active_blocks_table=active_blocks_table,
                attention_dp=1,
            ),
            f"X={tuple(X.shape)} v_cache={tuple(v_cache.shape)} "
            f"table={tuple(active_blocks_table.shape)} S_ctx={S_ctx} "
            f"window={self.sliding_window} fused=qkv_proj,qk_norm,rope,o_proj",
        )
        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            # Post-norm model: the attention input has no norm.
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=False,
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        if nkh > 1:
            scatter_new_kv(
                self.k_cache,
                self.v_cache,
                K_new,
                V_new,
                slot_mapping,
                block_size,
                nkh,
                self.head_dim,
            )
        else:
            # Manual KV cache update
            block_indices, position_indices = paged_slot_indices(
                slot_mapping, block_size, self.k_cache.shape[0]
            )
            num_tokens = slot_mapping.shape[0]

            k_new = (
                K_new.permute(1, 2, 0)
                .reshape(B, nkh, S_decode, self.head_dim)
                .transpose(0, 1)
                .reshape(nkh, B * S_decode, self.head_dim)
            )
            k_new_flat = k_new.reshape(-1, self.head_dim)
            v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

            head_indices_for_put = torch.arange(
                nkh, dtype=torch.long, device=hidden_states.device
            ).repeat_interleave(num_tokens)
            block_indices_for_put = block_indices.repeat(nkh)
            position_indices_for_put = position_indices.repeat(nkh)

            self.k_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                k_new_flat.to(self.k_cache.dtype),
            )
            self.v_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                v_new_flat.to(self.v_cache.dtype),
            )

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: Dense MLP
# =============================================================================


def padded_intermediate_size(intermediate_size: int, world_size: int) -> int:
    """Return the per-rank MLP intermediate size, padded to a multiple of 128.

    Example:
        >>> padded_intermediate_size(27392, 8)
        3456
    """
    per_rank = intermediate_size // world_size
    return (
        (per_rank + _MLP_INTERMEDIATE_ALIGN - 1)
        // _MLP_INTERMEDIATE_ALIGN
        * _MLP_INTERMEDIATE_ALIGN
    )


class Exaone4MLP(nn.Module):
    """Dense SiLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden → intermediate/TP per rank
    - down_proj: intermediate/TP → hidden per rank
    - Prefill: all-gather → compute → reduce-scatter (SP)
    - Decode: compute → all-reduce

    <-- MODEL-SPECIFIC: the per-rank intermediate size is zero-padded to a
    multiple of 128. silu(0) * 0 = 0, and the zero rows of down_proj add 0.
    """

    def __init__(self, config: Exaone4Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = (
            SLIDING_ATTENTION if config.is_sliding(layer_idx) else FULL_ATTENTION
        )

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size {config.intermediate_size} is not divisible by "
                f"the tensor-parallel size {self.world_size}. Use a "
                "tensor-parallel size that divides intermediate_size."
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.padded_intermediate_size_per_rank = padded_intermediate_size(
            config.intermediate_size, self.world_size
        )
        padded = self.padded_intermediate_size_per_rank

        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, padded, dtype=config.torch_dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, padded, dtype=config.torch_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(padded, config.hidden_size, dtype=config.torch_dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of MLP weights, with zero padding. <<<"""
        gate_up_loader = sharding_weight_loader_with_padding(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
            pad_dim=1,
            padded_size=self.padded_intermediate_size_per_rank,
            unpadded_size=self.intermediate_size_per_rank,
        )
        down_loader = sharding_weight_loader_with_padding(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
            pad_dim=0,
            padded_size=self.padded_intermediate_size_per_rank,
            unpadded_size=self.intermediate_size_per_rank,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<"""
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        hidden_states = _wait_collective(hidden_states)

        _log_dispatch(
            "prefill" if is_prefill else "decode",
            "mlp",
            self.layer_idx,
            self.layer_type,
            self.rank,
            _mlp._can_use_kernel(hidden_states, self.gate_proj_weight),
            f"hidden={tuple(hidden_states.shape)} "
            f"gate={tuple(self.gate_proj_weight.shape)}",
        )
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: post-norm residual structure
# =============================================================================


class Exaone4DecoderLayer(nn.Module):
    """Single EXAONE 4.0 decoder layer.

    Architecture:
        h = h + post_attention_layernorm(attn(h))
        h = h + post_feedforward_layernorm(mlp(h))
    """

    def __init__(self, config: Exaone4Config, layer_idx: int):
        super().__init__()
        self.self_attn = Exaone4Attention(config, layer_idx=layer_idx)
        self.mlp = Exaone4MLP(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Exaone4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Exaone4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.layer_idx = layer_idx

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class Exaone4Model(nn.Module):
    """EXAONE 4.0 transformer backbone.

    >>> PARALLELISM: SP during prefill — embed → chunk → layers → all-gather. <<<
    """

    def __init__(self, config: Exaone4Config):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                Exaone4DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Exaone4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Exaone4RotaryEmbedding(config)

        # <-- MODEL-SPECIFIC: per-layer attention type, RoPE and mask sources
        self.uses_rope = [config.uses_rope(i) for i in range(config.num_hidden_layers)]
        self.is_sliding = [
            config.is_sliding(i) for i in range(config.num_hidden_layers)
        ]
        self.first_sliding_layer = next(
            (i for i, s in enumerate(self.is_sliding) if s), None
        )
        self.first_full_layer = next(
            (i for i, s in enumerate(self.is_sliding) if not s), None
        )
        self.num_attention_heads_per_rank = (
            config.num_attention_heads // self.world_size
        )

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def _decode_mask(
        self,
        source_layer: int,
        attn_metadata: dict,
        positions: torch.Tensor,
        sliding_window: int | None,
    ) -> torch.Tensor:
        """Build the decode mask for all layers of one attention type.

        The mask uses the block table frame of ``source_layer``. The runner can
        trim the block table of a sliding layer to the window. It then gives
        ``swa_kv_pos_offset``, and the mask end and the window start move into
        the trimmed frame (same rule as GPT-OSS).
        """
        layer_name = f"layers.{source_layer}.self_attn"
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        max_blocks = attn_metadata[layer_name]["max_blocks_per_seq"]
        B = block_table.shape[0]
        S_decode = positions.shape[0] // B
        S_ctx = max_blocks * block_size
        pos_ids = positions.reshape(1, B * S_decode)

        start_pos = None
        if sliding_window is not None:
            swa_kv_pos_offset = attn_metadata[layer_name].get("swa_kv_pos_offset")
            if swa_kv_pos_offset is not None:
                pos_ids = pos_ids - swa_kv_pos_offset.reshape(B, 1).expand(
                    B, S_decode
                ).reshape(1, B * S_decode)
                # A padding row has position 0 and a positive offset. Clamp it
                # to 0, so that its window is empty (see GPT-OSS).
                pos_ids = torch.clamp(pos_ids, min=0)
            start_pos = torch.clamp(pos_ids - sliding_window + 1, min=0)

        pos_ids_f = pos_ids.to(torch.float32)
        _log_dispatch(
            "decode",
            "decode_mask_sliding" if sliding_window is not None else "decode_mask_full",
            source_layer,
            SLIDING_ATTENTION if sliding_window is not None else FULL_ATTENTION,
            self.rank,
            _attn_mask._can_use_kernel(pos_ids_f, B, S_decode, S_ctx),
            f"bs={B} s_active={S_decode} s_prior={S_ctx} window={sliding_window}",
        )
        return NF.gen_attention_decode_mask(
            pos_ids=pos_ids_f,
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=start_pos.to(torch.float32) if start_pos is not None else None,
            block_len=block_size,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold
        sp = is_prefill

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)

        if (
            sp
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            inputs_embeds = shard_select(inputs_embeds, 0, self.tp_group)
            is_token_ids = shard_select(is_token_ids, 0, self.tp_group)

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        rope_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        # <-- MODEL-SPECIFIC: identity rotation for the NoPE layers
        nope_embeddings = (
            torch.ones_like(rope_embeddings[0]),
            torch.zeros_like(rope_embeddings[1]),
        )

        # Build each decode mask one time for each step.
        swa_mask = None
        full_mask = None
        if not is_prefill:
            if self.first_sliding_layer is not None:
                swa_mask = self._decode_mask(
                    self.first_sliding_layer,
                    attn_metadata,
                    positions,
                    self.config.sliding_window,
                )
            if self.first_full_layer is not None:
                full_mask = self._decode_mask(
                    self.first_full_layer, attn_metadata, positions, None
                )

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=(
                    rope_embeddings if self.uses_rope[idx] else nope_embeddings
                ),
                attn_metadata=attn_metadata,
                attention_mask=swa_mask if self.is_sliding[idx] else full_mask,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather to the full sequence <<<
        if sp and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: LM Head + Weight Loading
# =============================================================================


def get_weight_mappings(num_layers: int) -> dict[str, str | list[str]]:
    """Return the map from model parameter names to HF checkpoint keys.

    Example:
        >>> m = get_weight_mappings(1)
        >>> m["model.layers.0.mlp.down_proj_weight"]
        'model.layers.0.mlp.down_proj.weight'
    """
    mappings: dict[str, str | list[str]] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "model.norm.weight",
    }
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        mappings[f"{prefix}.self_attn.o_proj_weight"] = (
            f"{prefix}.self_attn.o_proj.weight"
        )
        mappings[f"{prefix}.self_attn.q_norm.weight"] = (
            f"{prefix}.self_attn.q_norm.weight"
        )
        mappings[f"{prefix}.self_attn.k_norm.weight"] = (
            f"{prefix}.self_attn.k_norm.weight"
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = (
            f"{prefix}.post_attention_layernorm.weight"
        )
        mappings[f"{prefix}.post_feedforward_layernorm.weight"] = (
            f"{prefix}.post_feedforward_layernorm.weight"
        )
        mappings[f"{prefix}.mlp.gate_proj_weight"] = f"{prefix}.mlp.gate_proj.weight"
        mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
        mappings[f"{prefix}.mlp.down_proj_weight"] = f"{prefix}.mlp.down_proj.weight"
    return mappings


class Exaone4ForCausalLM(nn.Module):
    """EXAONE 4.0 model with LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    <-- MODEL-SPECIFIC: untied lm_head (tie_word_embeddings=False).
    """

    def __init__(self, config: Exaone4Config):
        super().__init__()
        self.config = config
        self.model = Exaone4Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        lm_head_key = (
            "model.embed_tokens.weight" if config.tie_word_embeddings else None
        )
        self._lm_head_key = lm_head_key
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prefill length {T} is not a multiple of world_size "
                f"{self.world_size} that is larger than world_size. Use a "
                "prefill bucket that meets this rule for SP."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        # One-token decode is row-aligned by construction. Skip the gather.
        if (
            not is_prefill
            and max_query_len == 1
            and spec_decode_metadata is None
            and hidden_states.shape[0] == sampling_positions.shape[0]
        ):
            hidden_states_for_logits = hidden_states
        else:
            hidden_states_for_logits = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Exaone4Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    # <-- MODEL-SPECIFIC: hybrid sliding/full KV groups
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    "Allocate a KV cache for each layer in get_kv_spec."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the EXAONE 4.0 HF checkpoint."""
        mappings = get_weight_mappings(len(self.model.layers))
        if self._lm_head_key is not None:
            mappings["lm_head.weight"] = self._lm_head_key

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load only the KV cache scales for the CPU compile."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn
            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
