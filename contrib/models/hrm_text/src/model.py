# SPDX-License-Identifier: Apache-2.0
"""
HRM-Text BF16 Implementation
============================

Implementation of HRM-Text (Hierarchical Reasoning Model) for the Neuron
native backend. Adapted from the SmolLM3 BF16 model.

HRM-Text has two transformer stacks of the same design, H (slow) and L (fast).
The forward runs this recurrence over the same token embeddings::

    z_H = embed(input_ids) * embedding_scale
    z_L = z_L_init
    for h in range(H_cycles):
        for l in range(L_cycles):
            z_L = L_module(z_L + z_H)      # KV slot offset (h * (L_cycles + 1) + l) * N
        z_H = H_module(z_H + z_L)          # KV slot offset (h * (L_cycles + 1) + L_cycles) * N
    logits = lm_head(z_H)

N is ``num_layers_per_stack``. Each stack pass of a layer writes a different
KV cache slot. Thus one layer object owns more than one KV cache.

Differences from Llama:
  - Parameterless RMSNorm (no weight), and a parameterless final norm in each stack.
  - Sigmoid output gate on attention: ``o_proj(sigmoid(gate_proj(x)) * attn)``.
  - Fused checkpoint tensors: ``gqkv_proj`` (gate, q, k, v) and ``gate_up_proj``.
  - Scaled embedding and the H/L recurrence.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    HRM-Text-specific. Change when porting.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from transformers import PretrainedConfig
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_block_table,
    paged_slot_indices,
    scatter_new_kv,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import HrmTextConfig

logger = logging.getLogger(__name__)


def kv_slot_name(slot: int) -> str:
    """Return the KV cache layer name of a KV slot."""
    return f"layers.{slot}.self_attn"


# All KV slots are in one KV cache group. Thus they share one block table, one
# slot mapping and one decode position set. The first decode layer stores the
# decode block table under this key. The other 127 layer passes of the step use
# it again.
_DECODE_INPUTS_CACHE_KEY = "_hrm_text_decode_inputs"


# =============================================================================
# Section 1: Parameterless RMS Normalization
# <-- MODEL-SPECIFIC: HRM-Text RMSNorm has no weight. It computes in FP32 and
# casts back to the input dtype, the same as the HF reference.
# =============================================================================


class HrmTextRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(hidden_states.dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (standard RoPE, rotate_half)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. q: [Nh, T, Dh], k: [Nkv, T, Dh], cos/sin: [T, Dh]."""
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class HrmTextRotaryEmbedding(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin of shape [T, head_dim]."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        freqs = (inv_freq[:, None] @ position_ids[None, :].float()).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Gated Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (MHA, sigmoid output gate, several KV slots per layer)
# =============================================================================


class HrmTextAttention(nn.Module):
    """Multi-head attention with a sigmoid output gate and TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V/gate heads are sharded across TP ranks (MHA, so KV heads shard too)
    - Prefill: all-gather input -> attention -> gate -> O proj -> reduce-scatter
    - Decode: fused NKI attention without O proj -> gate -> O proj -> all-reduce

    NKI kernels (Neuron device):
    - Prefill: QKV projection, flash or segmented attention, gate and O proj.
    - Decode: ``attention_block_tkg`` (QKV, RoPE, cache write, attention), gate
      and O proj.
    CPU mode uses the torch fallback of each NF API.

    <-- MODEL-SPECIFIC:
    - The gate multiplies the attention output elementwise before o_proj. The
      fused decode kernel cannot apply the gate, so the caller applies the gate
      and o_proj after the kernel (W_out=None).
    - The layer runs once in each stack pass. Each pass uses its own KV slot.
    """

    def __init__(self, config: HrmTextConfig, kv_slots: list[int]):
        super().__init__()
        self.kv_slots = list(kv_slots)
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"HRM-Text attention supports only MHA, but the config has "
                f"{self.num_attention_heads} query heads and "
                f"{self.num_key_value_heads} KV heads."
            )
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not "
                f"divisible by the TP size ({self.world_size}). Use a TP size "
                f"that divides the head count."
            )

        # >>> PARALLELISM: Head sharding (MHA: same count for Q and KV) <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_attention_heads_per_rank

        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = self.q_size
        self.qkv_split_indices = [self.q_size, self.q_size + self.kv_size]

        # Transposed storage [in, out], the same as the other native models.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, 3 * self.q_size, dtype=self.dtype)
        )
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.q_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )

        # KV caches bound externally via bind_kv_cache(): slot -> tensor.
        self.k_caches: dict[int, torch.Tensor] = {}
        self.v_caches: dict[int, torch.Tensor] = {}

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the loaders that split the fused ``gqkv_proj`` checkpoint tensor.

        <-- MODEL-SPECIFIC: The checkpoint row order is [gate | q | k | v]. Each
        part has ``num_attention_heads * head_dim`` rows. Each rank takes the
        rows of its heads from each part.
        """
        part = self.num_attention_heads * self.head_dim
        rows = self.q_size

        def _rows(slice_obj, part_idx: int, rank: int) -> torch.Tensor:
            start = part_idx * part + rank * rows
            return slice_obj[start : start + rows]

        def qkv_transform(slices, rank):
            assert len(slices) == 1
            q = _rows(slices[0], 1, rank)
            k = _rows(slices[0], 2, rank)
            v = _rows(slices[0], 3, rank)
            return torch.cat([q, k, v], dim=0).T

        def gate_transform(slices, rank):
            assert len(slices) == 1
            return _rows(slices[0], 0, rank).T

        set_weight_loader(
            self.qkv_proj_weight, SafetensorsWeightLoader(transform=qkv_transform)
        )
        set_weight_loader(
            self.gate_proj_weight, SafetensorsWeightLoader(transform=gate_transform)
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        kv_slot: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        if not is_prefill:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata, kv_slot
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata, kv_slot
        )

    def _gate_and_o_proj(
        self, hidden_states: torch.Tensor, attn_bnds: torch.Tensor
    ) -> torch.Tensor:
        """Apply ``o_proj(sigmoid(gate_proj(x)) * attn)`` with NKI projections.

        hidden_states: [T, H]. attn_bnds: [1, Nh_rank, Dh, T] (BNdS layout).
        Returns the partial [T, H] output of this rank.

        <-- MODEL-SPECIFIC: The gate projection uses the NKI QKV projection
        kernel (``NF.qkv_proj``) as a plain projection, because the gate weight
        has the same [H, Nh_rank * Dh] layout as a fused QKV weight. The gated
        attention then goes to the NKI output projection kernel
        (``NF.o_proj``) in the BNdS layout that both attention kernels give.
        """
        tokens = hidden_states.shape[0]
        nh = self.num_attention_heads_per_rank
        gate = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.gate_proj_weight,
            bias=None,
        )  # [1, T, Nh * Dh]
        gate = gate.view(1, tokens, nh, self.head_dim).permute(0, 2, 3, 1)
        # Calculate the sigmoid and the gate product in FP32. Then round once
        # to the weight dtype. Two BF16 roundings here move the prefill logits
        # outside the logit tolerance.
        gated = (
            (torch.sigmoid(gate.float()) * attn_bnds.float())
            .to(self.dtype)
            .contiguous()
        )  # [1, Nh, Dh, T]
        output = NF.o_proj(gated, self.o_proj_weight, None)  # [1, T, H]
        return output.view(tokens, self.hidden_size)

    def _write_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write K/V ``[Nkh, T, Dh]`` to the paged cache ``[blocks, Nkh, bs, Dh]``."""
        num_heads, num_tokens, _ = k.shape
        # A pad slot (-1) goes to the null block. It must not become an
        # out-of-range scatter address on the device.
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, k_cache.shape[0]
        )
        k_flat = k.reshape(-1, self.head_dim).to(k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(v_cache.dtype)
        head_indices_for_put = torch.arange(
            num_heads, dtype=torch.long, device=k.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(num_heads)
        position_indices_for_put = position_indices.repeat(num_heads)
        k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        kv_slot: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape
        nh = self.num_attention_heads_per_rank

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, nh, self.head_dim).transpose(0, 1)
        k = k.view(tokens, nh, self.head_dim).transpose(0, 1)
        v = v.view(tokens, nh, self.head_dim).transpose(0, 1)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # >>> PARALLELISM: Per-rank cache (TP sharded KV heads) <<<
        meta = attn_metadata[kv_slot_name(kv_slot)]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        cached_seq_len = meta.get("cached_seq_len")
        kv_segment_size = meta.get("kv_segment_size")
        k_cache = self.k_caches[kv_slot]
        v_cache = self.v_caches[kv_slot]
        self._write_kv(k_cache, v_cache, k, v, slot_mapping, block_size)

        # <-- MODEL-SPECIFIC: Causal attention. The PrefixLM mask is not used.
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )  # [Nh, Dh, T]
        else:
            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # [Nh, Dh, T] -> [1, Nh, Dh, T] (BNdS layout of NF.o_proj)
        output = self._gate_and_o_proj(hidden_states, attn_output.unsqueeze(0))

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        # Reduce the partial outputs in FP32. The decoder layer rounds once
        # after the residual add.
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output.float(), dim=0)

        return output.contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        kv_slot: int,
    ) -> torch.Tensor:
        """Decode: fused NKI attention without o_proj, then the gate and o_proj.

        <-- MODEL-SPECIFIC: HRM applies the sigmoid gate before o_proj. Thus the
        fused decode kernel runs with ``W_out=None`` and gives the attention
        output in the BNdS layout [B, Nh, Dh, S]. The kernel does the QKV
        projection, RoPE and the attention. It also gives the new K/V tokens
        (``update_cache=False``). ``scatter_new_kv`` then writes these tokens
        into the paged KV cache.

        The kernel cannot write the cache in place when ``W_out`` is None. Its
        K/V outputs are then the new tokens and not the aliased caches, so the
        write-back to the cache input fails at trace time.

        The kernel folds the KV heads into the batch. Each rank has 6 KV heads
        (MHA, TP 2), so the kernel needs the per-head block table
        [B, Nkh, blocks]. A shared 2D table is not correct for the kernel at
        Nkh > 1.

        The kernel makes the prior attention mask from ``pos_ids``. Thus the
        mask layout always agrees with the kernel sharding of the folded batch.
        """
        meta = attn_metadata[kv_slot_name(kv_slot)]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"Decode token count ({tokens}) is not a multiple of the batch size "
            f"({B}). Give the same number of tokens to each request."
        )

        hidden_states = hidden_states.to(self.dtype)
        nh = self.num_attention_heads_per_rank
        nkh = self.num_key_value_heads_per_rank
        k_cache = self.k_caches[kv_slot]
        v_cache = self.v_caches[kv_slot]
        X = hidden_states.view(B, S_decode, hidden)

        # Kernel RoPE layout: [Dh // 2, B, S] (rotate_half, contiguous halves).
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

        # <-- MODEL-SPECIFIC: Give the positions to the kernel (fused mask
        # generation). The kernel then makes the prior mask on chip in the
        # layout of its folded batch [B * Nkh]. An external mask from
        # NF.gen_attention_decode_mask uses bs=B and q_head=Nh. At B=1 its
        # block-length resize is different from the kernel sharding, and all
        # decode steps then diverge.
        pos_ids = positions.view(B, S_decode).to(torch.float32)

        active_blocks_table = attn_metadata.get(_DECODE_INPUTS_CACHE_KEY)
        if active_blocks_table is None:
            # >>> PARALLELISM: Per-head table for the KV heads of this rank <<<
            # The kernel path needs the per-head table at Nkh > 1. The torch
            # fallback of NF.attention_decode (CPU mode without the NKI
            # simulator) reads the 2D table instead.
            if can_run_kernel(X) and nkh > 1:
                active_blocks_table = build_per_head_block_table(
                    block_table.to(torch.int32), nkh
                )
            else:
                active_blocks_table = block_table.to(torch.int32)
            attn_metadata[_DECODE_INPUTS_CACHE_KEY] = active_blocks_table

        attn_output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=None,
            pos_ids=pos_ids,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=False,
            W_out=None,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )  # [B, Nh, Dh, S]

        # The kernel gives K_new [Dh, B, Nkh, S]. The torch fallback gives
        # [Dh, B * Nkh, S] in batch-major order. Use the 4D layout for both.
        K_new = K_new.reshape(self.head_dim, B, nkh, S_decode)
        V_new = V_new.reshape(B, nkh, S_decode, self.head_dim)
        scatter_new_kv(
            k_cache,
            v_cache,
            K_new,
            V_new,
            slot_mapping,
            block_size,
            nkh,
            self.head_dim,
        )

        # [B, Nh, Dh, S] -> [1, Nh, Dh, B * S]. Token b * S + s is row
        # b * S + s of hidden_states.
        attn_bnds = attn_output.permute(1, 2, 0, 3).reshape(
            1, nh, self.head_dim, tokens
        )
        output = self._gate_and_o_proj(hidden_states, attn_bnds)

        # >>> PARALLELISM: TP all-reduce <<<
        # Reduce the partial outputs in FP32. The decoder layer rounds once
        # after the residual add.
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output.float())

        return output


# =============================================================================
# Section 4: MLP (SwiGLU)
# =============================================================================


class HrmTextMLP(nn.Module):
    """SwiGLU MLP with TP intermediate sharding: down(silu(gate(x)) * up(x)).

    <-- MODEL-SPECIFIC: The checkpoint fuses gate and up in ``gate_up_proj``
    with the row order [gate | up].
    """

    def __init__(self, config: HrmTextConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        rows = self.intermediate_size_per_rank
        part = self.intermediate_size

        def _part_transform(part_idx: int):
            def transform(slices, rank):
                assert len(slices) == 1
                start = part_idx * part + rank * rows
                return slices[0][start : start + rows].T

            return transform

        set_weight_loader(
            self.gate_proj_weight, SafetensorsWeightLoader(transform=_part_transform(0))
        )
        set_weight_loader(
            self.up_proj_weight, SafetensorsWeightLoader(transform=_part_transform(1))
        )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=rows,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        # Reduce the partial outputs in FP32. The decoder layer rounds once
        # after the residual add.
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output.float(), dim=0)
            else:
                output = self.tp_group.all_reduce(output.float())

        return output


# =============================================================================
# Section 5: Decoder Layer and Stack
# =============================================================================


class HrmTextDecoderLayer(nn.Module):
    """Pre-norm decoder layer: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: HrmTextConfig, kv_slots: list[int]):
        super().__init__()
        self.input_layernorm = HrmTextRMSNorm(config.rms_norm_eps)
        self.post_attention_layernorm = HrmTextRMSNorm(config.rms_norm_eps)
        self.self_attn = HrmTextAttention(config, kv_slots)
        self.mlp = HrmTextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        kv_slot: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            kv_slot=kv_slot,
            is_prefill=is_prefill,
        )
        # Add the residual in FP32 and round once to the residual dtype. At
        # TP > 1 the sublayer output is an FP32 sum of the rank partials.
        hidden_states = (residual.float() + hidden_states.float()).to(residual.dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        return (residual.float() + hidden_states.float()).to(residual.dtype)


class HrmTextStack(nn.Module):
    """One transformer stack (H or L) with a parameterless final norm.

    ``kv_slot_offsets`` lists the slot offset of each pass of this stack. Layer
    ``i`` in the pass with offset ``o`` uses KV slot ``o + i``.
    """

    def __init__(self, config: HrmTextConfig, kv_slot_offsets: list[int]):
        super().__init__()
        self.kv_slot_offsets = list(kv_slot_offsets)
        self.layers = nn.ModuleList(
            [
                HrmTextDecoderLayer(
                    config, [offset + i for offset in self.kv_slot_offsets]
                )
                for i in range(config.num_layers_per_stack)
            ]
        )
        self.final_norm = HrmTextRMSNorm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        cycle_offset: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                kv_slot=cycle_offset + i,
                is_prefill=is_prefill,
            )
        return self.final_norm(hidden_states)


# =============================================================================
# Section 6: Model Backbone (H/L recurrence)
# =============================================================================


class HrmTextModel(nn.Module):
    """HRM-Text backbone.

    >>> PARALLELISM: SP <<<
    During prefill each rank holds T / world_size tokens between collectives.
    The final hidden state is all-gathered on prefill.
    """

    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        # <-- MODEL-SPECIFIC: Initial L state (replicated on each rank).
        self.z_L_init = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=config.torch_dtype),
            requires_grad=False,
        )

        h_offsets = [
            config.cycle_offset(hc, config.L_cycles) for hc in range(config.H_cycles)
        ]
        l_offsets = [
            config.cycle_offset(hc, lc)
            for hc in range(config.H_cycles)
            for lc in range(config.L_cycles)
        ]
        self.H_module = HrmTextStack(config, h_offsets)
        self.L_module = HrmTextStack(config, l_offsets)
        self.rotary_emb = HrmTextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = attn_metadata[kv_slot_name(0)]
        is_prefill = first["max_query_len"] > first["decode_token_threshold"]

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        z_H = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        # <-- MODEL-SPECIFIC: Scaled embedding.
        z_H = z_H * self.config.embedding_scale
        z_L = self.z_L_init.to(z_H.dtype).expand_as(z_H)

        # SP slices hold the matching positions, but RoPE in attention runs on
        # the full gathered sequence, so compute it for all positions.
        position_embeddings = self.rotary_emb(
            positions, device=z_H.device, dtype=z_H.dtype
        )

        cfg = self.config
        for hc in range(cfg.H_cycles):
            for lc in range(cfg.L_cycles):
                z_L = self.L_module(
                    z_L + z_H,
                    positions,
                    position_embeddings,
                    attn_metadata,
                    cycle_offset=cfg.cycle_offset(hc, lc),
                    is_prefill=is_prefill,
                )
            z_H = self.H_module(
                z_H + z_L,
                positions,
                position_embeddings,
                attn_metadata,
                cycle_offset=cfg.cycle_offset(hc, cfg.L_cycles),
                is_prefill=is_prefill,
            )

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            z_H = self.tp_group.all_gather(z_H, dim=0)

        return z_H


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class HrmTextForCausalLM(nn.Module):
    """HRM-Text model with an untied, column-parallel language modeling head."""

    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.config = config
        self.model = HrmTextModel(config)

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

    def _attention_layers(self) -> list[HrmTextAttention]:
        return [
            layer.self_attn
            for stack in (self.model.H_module, self.model.L_module)
            for layer in stack.layers
        ]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("HRM-Text does not support inputs_embeds. Pass input_ids.")

        positions = positions.to(torch.int32)

        first = attn_metadata[kv_slot_name(0)]
        is_prefill = first["max_query_len"] > first["decode_token_threshold"]

        T = input_ids.shape[0]

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) must be larger than world_size "
                f"({self.world_size}) and divisible by it for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

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
        config = HrmTextConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------
    # <-- MODEL-SPECIFIC: One KV cache for each (layer, stack pass) slot.

    def get_kv_spec(self) -> KVSpec:
        layers = []
        attn = self.model.H_module.layers[0].self_attn
        for slot in range(self.config.num_kv_slots):
            layers.append(
                LayerSpec(
                    name=kv_slot_name(slot),
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for attn in self._attention_layers():
            for slot in attn.kv_slots:
                name = kv_slot_name(slot)
                if name not in kv_caches:
                    raise RuntimeError(
                        f"KV cache for layer {name} is not initialized. "
                        f"Allocate one cache for each of the "
                        f"{self.config.num_kv_slots} KV slots."
                    )
                attn.k_caches[slot] = kv_caches[name][0]
                attn.v_caches[slot] = kv_caches[name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str]:
        """Return the parameter name -> checkpoint key mappings.

        ``model.embed_tokens.weight``, ``model.z_L_init`` and ``lm_head.weight``
        have the same name in the checkpoint, so they need no mapping.
        """
        mappings = {}
        for stack in ("H_module", "L_module"):
            for i in range(self.config.num_layers_per_stack):
                prefix = f"model.{stack}.layers.{i}"
                gqkv = f"{prefix}.attn.gqkv_proj.weight"
                mappings[f"{prefix}.self_attn.qkv_proj_weight"] = gqkv
                mappings[f"{prefix}.self_attn.gate_proj_weight"] = gqkv
                mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                    f"{prefix}.attn.o_proj.weight"
                )
                gate_up = f"{prefix}.mlp.gate_up_proj.weight"
                mappings[f"{prefix}.mlp.gate_proj_weight"] = gate_up
                mappings[f"{prefix}.mlp.up_proj_weight"] = gate_up
                mappings[f"{prefix}.mlp.down_proj_weight"] = (
                    f"{prefix}.mlp.down_proj.weight"
                )
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.checkpoint_mappings(), device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
