# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR audio encoder (BF16).

Complete audio transformer implementation: PatchEmbed → Blocks × depth → Merger.
Produces main embeddings.

All tensors flow as 3D ``[num_blocks, block_size, hidden_size]`` through the
audio encoder (block packing layout). Each block is processed independently
in attention via bounds masking. After the merger, the tensor is written directly
into the on-device encoder cache buffer via index_put_.

Supported parallelism: TP (audio-specific TP group, decoupled from text model).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3-ASR audio-specific. Change when porting.

PARALLELISM SHARDING:
  TP:  Attention heads sharded, MLP intermediate sharded, merger MLP sharded.
       The audio encoder uses the encoder TP group
       (get_neuron_encoder_tp_group), decoupled from the text model TP group.
       No SP — audio encoder processes full sequence on all TP ranks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import NormType
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_encoder_tp_group,
    shard_select,
)
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader
from vllm_neuron.utils.dynamo_graph import allow_in_graph_cacheable

from .config import Qwen3ASRAudioConfig
from .utils.audio_preprocessing import resolve_block_geometry
from .weight_loader import (
    audio_conv1_weight_loader,
    audio_qkv_bias_loader,
    audio_qkv_weight_loader,
)


@allow_in_graph_cacheable
def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU via erf — traceable on Neuron (libtorch_neuronx_lite patches F.gelu
    with a C extension that torch.compile/Dynamo cannot trace through). Matches
    the audio encoder's activation_function == "gelu" (exact, not tanh)."""
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


# =============================================================================
# Section 2: Sinusoids Position Embedding
# =============================================================================


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        position_embedding = self.compute_default_singular_positional_embedding()
        self.positional_embedding = nn.Buffer(position_embedding, persistent=False)

    def compute_default_singular_positional_embedding(self):
        # Build the table on CPU. The model init runs under a meta device
        # context. A meta tensor has no data, so a later .to(device) fails.
        # This buffer is a fixed sinusoid table, not a checkpoint weight.
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment
            * torch.arange(self.channels // 2, device="cpu").float()
        )
        scaled_time = (
            torch.arange(self.length, device="cpu")[:, np.newaxis]
            * inv_timescales[np.newaxis, :]
        )
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# =============================================================================
# Section 3: Dense MLP
# =============================================================================


class Qwen3ASRAudioMLP(nn.Module):
    """Audio transformer MLP block with tensor-parallel sharding.

    Architecture: ``up_proj(x) → GELU → down_proj(x)`` (non-gated FFN).

    Parallelism (column-parallel up, row-parallel down):
    - up_proj weight is sharded on the intermediate dim (param dim=1) across TP ranks.
    - down_proj weight is sharded on the intermediate dim (param dim=0) across TP ranks.
    - TP all-reduce after down_proj matmul, then the down_proj bias is added.

    Args:
        config: Qwen3-ASR audio config (uses d_model, encoder_ffn_dim, torch_dtype).
    """

    def __init__(self, config: Qwen3ASRAudioConfig):
        super().__init__()

        # >>> PARALLELISM: encoder TP group (decoupled from text TP) <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size

        self.d_model = config.d_model
        # >>> PARALLELISM: Intermediate dim sharded across audio TP <<<
        self.intermediate_size_per_rank = config.encoder_ffn_dim // self.world_size

        self.up_proj_weight = nn.Parameter(
            torch.empty(
                self.d_model, self.intermediate_size_per_rank, dtype=config.torch_dtype
            )
        )
        self.up_proj_bias = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank, self.d_model, dtype=config.torch_dtype
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(
                self.d_model,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """>>> PARALLELISM: TP sharding of MLP weights. <<<

        Checkpoint stores nn.Linear weights as [out, in]; our params are the
        transpose [in, out], so is_storage_transposed=True. down_proj_bias is
        replicated ([d_model]) and needs no loader. At world_size == 1 these
        reduce to a plain transpose (num_shards=1, no real sharding).
        """
        set_weight_loader(
            self.up_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.up_proj_bias,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP forward: up → GELU → down → all-reduce → + bias.

        Args:
            hidden_states: ``[..., d_model]`` — arbitrary leading dims.

        Returns:
            ``[..., d_model]`` — same leading dims as input.
        """

        up = (
            hidden_states @ self.up_proj_weight + self.up_proj_bias
        )  # column-parallel [T, I/tp]
        output = gelu(up) @ self.down_proj_weight  # row-parallel partials [T, H]

        # >>> PARALLELISM: row-parallel down_proj -> reduce partial sums <<<
        # Required whenever the intermediate dim is sharded (world_size > 1).
        # The replicated down_proj bias is added AFTER the reduce; adding it
        # before would accumulate world_size * bias.
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output + self.down_proj_bias


# =============================================================================
# Section 4: Attention
# =============================================================================


class Qwen3ASRAudioAttention(nn.Module):
    """audio transformer self-attention with bounds masking.

    Architecture:
    - The QKV projection fuses LayerNorm and bias.
    - Bidirectional attention with bounds masking (no causal mask)
    - Output projection (with bias) via raw weight parameters

    Parallelism:
    - QKV weight fused from separate q/k/v checkpoints via audio_qkv_weight_loader
    - Output projection weight sharded on input dim
    - No SP — audio encoder processes full sequence
    - No GQA — num_heads Q = num_heads K = num_heads V

    Args:
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.
        hidden_size: Total hidden dimension (num_heads * head_dim).
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        config: Qwen3ASRAudioConfig,
    ) -> None:
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.num_heads  # derived again for correctness
        self.scale = self.head_dim**-0.5
        self.dtype = config.torch_dtype

        # >>> PARALLELISM: encoder TP group (decoupled from text TP) <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Head sharding calculation <<<
        assert self.num_heads % self.world_size == 0, (
            f"num_heads {self.num_heads} must be divisible by tp world_size {self.world_size}"
        )
        self.num_heads_per_rank: int = self.num_heads // self.world_size

        self.norm = nn.LayerNorm(self.d_model, dtype=self.dtype)

        # >>> PARALLELISM: QKV weight shapes (per-rank sharded heads) <<<
        qkv_size_per_rank = 3 * self.num_heads_per_rank * self.head_dim
        self.qkv_weight = nn.Parameter(
            torch.empty(self.d_model, qkv_size_per_rank, dtype=self.dtype)
        )
        self.qkv_bias = nn.Parameter(torch.empty(qkv_size_per_rank, dtype=self.dtype))

        # >>> PARALLELISM: O-proj input sharded across TP ranks <<<
        o_proj_in = self.num_heads_per_rank * self.head_dim
        self.proj_weight = nn.Parameter(
            torch.empty(o_proj_in, self.d_model, dtype=self.dtype)
        )
        self.proj_bias = nn.Parameter(torch.empty(self.d_model, dtype=self.dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """>>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors. <<<"""
        # The norm parameters are replicated. The default loader copies each tensor.
        # Qwen3-ASR stores q_proj/k_proj/v_proj separately, so the mapping must
        # point qkv_weight -> [q_proj.weight, k_proj.weight, v_proj.weight] and
        # qkv_bias -> [q_proj.bias, k_proj.bias, v_proj.bias]; these loaders fuse
        # the three slices with per-rank head sharding.
        set_weight_loader(
            self.qkv_weight,
            audio_qkv_weight_loader(self.num_heads_per_rank, self.head_dim),
        )
        set_weight_loader(
            self.qkv_bias,
            audio_qkv_bias_loader(self.num_heads_per_rank, self.head_dim),
        )
        set_weight_loader(
            self.proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward with per-block bounds masking.

        Uses block packing layout: each block is processed independently via
        bounds masking. The batch dimension (dim 0) is num_blocks — blocks are
        treated as independent sequences for attention.

        Args:
            hidden_states: ``[num_blocks, block_size, d_model]``.
            bound_min: ``[num_blocks, block_size, 1]`` int32 — inclusive lower bound.
            bound_max: ``[num_blocks, block_size, 1]`` int32 — exclusive upper bound.

        Returns:
            ``[num_blocks, block_size, d_model]``
        """
        num_blocks, block_size, _ = hidden_states.shape
        hidden_states = hidden_states.to(self.dtype)

        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
        # qkv: [num_blocks, block_size, 3 * heads_per_rank * head_dim]

        qkv = NF.qkv_proj(
            hidden=hidden_states,
            qkv_weights=self.qkv_weight,
            norm_weights=self.norm.weight.unsqueeze(0),
            eps=self.norm.eps,
            norm_type=NormType.LAYER_NORM,
            bias=self.qkv_bias.unsqueeze(0),
            norm_b=self.norm.bias.unsqueeze(0),
        )

        hd = self.num_heads_per_rank * self.head_dim
        q = qkv[:, :, :hd]
        k = qkv[:, :, hd : 2 * hd]
        v = qkv[:, :, 2 * hd :]

        # [num_blocks, block_size, heads_per_rank * head_dim]
        # → [num_blocks, block_size, heads_per_rank, head_dim]
        q = q.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)
        k = k.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)
        v = v.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)

        # [num_blocks, block_size, heads_per_rank, head_dim]
        # → [num_blocks * heads_per_rank, block_size, head_dim]
        q = q.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )
        k = k.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )
        v = v.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )

        # <-- MODEL-SPECIFIC: Bidirectional attention with per-block bounds masking
        # bound_min/max: [num_blocks, block_size, 1]
        # → [num_blocks * heads_per_rank, block_size, 1] (repeat per head)
        attn_bound_min = bound_min.repeat_interleave(self.num_heads_per_rank, dim=0)
        attn_bound_max = bound_max.repeat_interleave(self.num_heads_per_rank, dim=0)

        attn_output = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,
            bound_min=attn_bound_min,
            bound_max=attn_bound_max,
        )

        # >>> PARALLELISM: O-proj + all-reduce across audio TP ranks <<<
        # [num_blocks * heads_per_rank, head_dim, block_size]
        # → [num_blocks, heads_per_rank, head_dim, block_size] for NF.o_proj
        attn_output = attn_output.view(
            num_blocks, self.num_heads_per_rank, self.head_dim, block_size
        )

        output = NF.o_proj(attn_output, self.proj_weight)  # [num_blocks, S, H]

        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output + self.proj_bias


# =============================================================================
# Section 5: Encoder Layer
# =============================================================================


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Single audio transformer block with two pre-norm residual paths.

    The attention module fuses the first LayerNorm into the QKV projection.
    The MLP path applies norm2 before the MLP.
    """

    def __init__(self, config: Qwen3ASRAudioConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.dtype = config.torch_dtype

        self.norm2 = nn.LayerNorm(self.d_model, dtype=self.dtype)  # final_layer_norm
        self.attn = Qwen3ASRAudioAttention(config)
        self.mlp = Qwen3ASRAudioMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Block forward with pre-norm residual connections.

        Args:
            hidden_states: ``[num_blocks, block_size, d_model]``
            bound_min: ``[num_blocks, block_size, 1]`` int32
            bound_max: ``[num_blocks, block_size, 1]`` int32

        Returns:
            ``[num_blocks, block_size, d_model]``
        """
        hidden_states = hidden_states + self.attn(hidden_states, bound_min, bound_max)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        # skip fp16 clamp as not needed for bf16

        return hidden_states


# =============================================================================
# Section 6: Audio Model (top-level)
# =============================================================================


class _Qwen3ASRPackedConv1(nn.Module):
    def __init__(self, out_channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_channels = 1
        self.out_channels = out_channels
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1
        self.weight = nn.Parameter(torch.empty(9, out_channels, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels, dtype=dtype))
        set_weight_loader(self.weight, audio_conv1_weight_loader())


class Qwen3ASREncoder(nn.Module):
    """Complete Qwen3-ASR audio encoder.

    Args:
        config: audio config object. It reads d_model, encoder_ffn_dim,
            encoder_attention_heads, head_dim, encoder_layers, num_mel_bins,
            downsample_hidden_size, output_dim, n_window, n_window_infer,
            window_tokens, max_position_embeddings,
            encoder_attention_block_size, torch_dtype, and neuron_config.
    """

    def __init__(self, config: Qwen3ASRAudioConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim

        # >>> PARALLELISM: encoder DP group <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_encoder_dp_group,
        )

        self.dp_group = get_neuron_encoder_dp_group()
        self.dp_size = self.dp_group.world_size if self.dp_group else 1

        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(
            config.max_position_embeddings,  # 13
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        # dtype must match the (bf16) activations: nn.Conv2d/Linear/LayerNorm
        # default to fp32, and the weight loader would then down-cast the bf16
        # checkpoint tensors to fp32 params, breaking conv on a bf16 input
        # ("Input type BFloat16 and bias type float should be the same").
        dtype = config.torch_dtype
        self.ln_post = nn.LayerNorm(config.d_model, dtype=dtype)
        self.conv2d1 = _Qwen3ASRPackedConv1(
            config.downsample_hidden_size,
            dtype,
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
            dtype=dtype,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
            dtype=dtype,
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
            dtype=dtype,
        )
        self.n_window_infer = config.n_window_infer
        # Post-CNN tokens in one inference window (104)
        self.window_tokens = config.window_tokens

        # Output projector (== HF Qwen3ASRMultiModalProjector, folded into encoder).
        self.proj1 = nn.Linear(config.d_model, config.d_model, dtype=dtype)
        self.proj2 = nn.Linear(config.d_model, config.output_dim, dtype=dtype)

    def _conv1_addmm(self, x: torch.Tensor) -> torch.Tensor:
        """Run the first convolution with patch extraction and addmm."""
        conv1 = self.conv2d1
        assert (
            conv1.in_channels == 1
            and conv1.kernel_size == (3, 3)
            and conv1.stride == (2, 2)
            and conv1.padding == (1, 1)
            and conv1.dilation == (1, 1)
            and conv1.groups == 1
            and conv1.bias is not None
            and conv1.weight.shape == (9, conv1.out_channels)
        ), (
            "conv2d1 has unsupported parameters. Use one input channel, a 3x3 "
            "kernel, stride 2, padding 1, dilation 1, one group, a bias, and a "
            "packed weight."
        )
        batch_size, channels, height, width = x.shape
        assert channels == 1

        output_height = (height + 1) // 2
        output_width = (width + 1) // 2
        padded = F.pad(x, (1, 1, 1, 1))
        patches = torch.stack(
            [
                padded[
                    :,
                    0,
                    kernel_row : kernel_row + 2 * output_height : 2,
                    kernel_col : kernel_col + 2 * output_width : 2,
                ]
                for kernel_row in range(3)
                for kernel_col in range(3)
            ],
            dim=-1,
        )
        output = torch.addmm(
            conv1.bias,
            patches.reshape(-1, 9),
            conv1.weight,
        ).view(
            batch_size,
            output_height,
            output_width,
            conv1.out_channels,
        )
        return output.permute(0, 3, 1, 2)

    def forward(
        self,
        input_features: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        chunk_slots: torch.Tensor,
        encoder_cache_buffer: torch.Tensor | None = None,
        write_block_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full audio encoder on one packed feature stream.

        The CPU preprocessing packs the mel features into fixed-size attention
        blocks. The slot unit is one base chunk; a block holds
        ``slots_per_block = block_size // num_tokens_per_chunk`` chunk slots. The
        preprocessing also makes ``chunk_slots``, which says where each real
        chunk goes, and the block-packed ``bound_min`` / ``bound_max``, which
        mark the valid and the padding tokens (per attention segment). The graph
        needs no ``nonzero`` (its output shape would be data-dependent); the
        ``index_select`` below keeps a static shape, so it compiles.

        Args:
            input_features: ``[1, num_mel_bins, F]`` mel features for one packed
                stream. ``F`` is a multiple of ``num_mel_frames_per_chunk``. The CPU and the
                top-level model do this preprocessing.
            bound_min, bound_max: ``[num_blocks, block_size, 1]`` int32. Token ``i``
                in block ``b`` attends to ``[bound_min[b, i], bound_max[b, i])``
                in block-local index space.
            chunk_slots: ``[num_blocks * slots_per_block]`` int64, the gather
                index from ``AudioBlockAssignment.chunk_slots()`` used by the
                ``index_select`` below. Entry ``s`` is the conv-output chunk
                index that fills block slot ``s``. A real slot points at its base
                chunk; a padding slot points at chunk 0 (a harmless duplicate),
                and the bounds give it an empty span so it does not attend.
            encoder_cache_buffer: optional ``[num_cache_blocks, cache_block_size, output_dim]``.
                Pre-allocated device cache (input-output alias). The last dim is
                output_dim (post-projector), not d_model.
            write_block_ids: ``[num_ve_blocks]`` int64. It maps encoder output
                block ``i`` to a cache buffer block index.

        Returns:
            The updated ``encoder_cache_buffer`` (same tensor, aliased). It returns
            the encoder features ``[num_blocks, block_size, output_dim]`` when the
            call gives no cache.

        Variables:
            num_mel_frames_per_chunk = mel frames in one base chunk    [100 = 2 * n_window = 1s]
            time_steps               = post-CNN tokens in one base chunk    [13 = num_tokens_per_chunk]
            window_tokens            = post-CNN tokens in one window        [104 = 8 * 13]
            num_chunks               = base chunks in this input            [feature_length / num_mel_frames_per_chunk]
            block_size               = post-CNN tokens per block, kernel-aligned, from the bounds
            slots_per_block          = whole chunk slots in one block       [block_size // num_tokens_per_chunk]
            num_blocks               = attention blocks, from the bounds

        The input is one packed feature stream, so the leading batch dim is
        always 1. The code expresses batching across num_blocks.
        """
        assert input_features.shape[0] == 1, (
            "The audio encoder takes one packed feature stream. "
            f"The leading batch dim must be 1, but it is {input_features.shape[0]}."
        )

        _, num_mel_bins, feature_length = input_features.shape
        num_mel_frames_per_chunk = (
            self.n_window * 2
        )  # mel frames in one base chunk [100 = 2 * n_window = 1s]

        if feature_length % num_mel_frames_per_chunk != 0:
            raise ValueError(
                f"Qwen3ASREncoder expects the padded feature length to be a "
                f"multiple of num_mel_frames_per_chunk ({num_mel_frames_per_chunk}), but got "
                f"{feature_length}. Pad the mel to whole base chunks."
            )

        # The layout comes from the bounds, so the encoder is a pure function of
        # its input shapes and stays compile-stable. Attention derives the same
        # two values from hidden_states.shape.
        num_blocks, block_size, _ = bound_min.shape
        slots_per_block, _, _ = resolve_block_geometry(
            block_size, self.window_tokens, self.n_window, self.n_window_infer
        )

        # Production keeps one source of truth: the neuron config. Module tests
        # pass no neuron config and drive the layout from the bounds alone.
        nc = self.config.neuron_config
        if nc is not None and block_size != nc.encoder_attention_block_size:
            raise ValueError(
                f"The bounds block size ({block_size}) does not match "
                f"neuron_config.encoder_attention_block_size "
                f"({nc.encoder_attention_block_size}). Fix the preprocessing or "
                f"the bucket block size."
            )

        num_chunks = feature_length // num_mel_frames_per_chunk

        expected_slots = num_blocks * slots_per_block
        if chunk_slots.shape[0] != expected_slots:
            raise ValueError(
                f"chunk_slots has {chunk_slots.shape[0]} entries but the layout "
                f"has {expected_slots} slots (num_blocks={num_blocks} * "
                f"slots_per_block={slots_per_block})."
            )

        # Step 1: Chunk mel into per-base-chunk conv inputs: [num_chunks, 1, mel, num_mel_frames_per_chunk].
        chunked = (
            input_features.view(
                1, num_mel_bins, num_chunks, num_mel_frames_per_chunk
            ).permute(  # splits F into (num_chunks, num_mel_frames_per_chunk)
                2, 0, 1, 3
            )  # moves num_chunks to the batch dim
        )  # this single permute is valid only because the batch dim is 1

        # Step 2: Conv frontend (3x s2p1 -> 8x downsample): [num_chunks, C, freq, time_steps].
        x = gelu(self._conv1_addmm(chunked))
        x = gelu(self.conv2d2(x))
        x = gelu(self.conv2d3(x))
        _, conv_channels, freq_bins, time_steps = x.shape
        x = self.conv_out(
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(num_chunks, time_steps, conv_channels * freq_bins)
        )  # [num_chunks, time_steps, d_model]
        # Sinusoidal position embedding, added per base chunk (no RoPE).
        x = x + self.positional_embedding(time_steps).to(x.dtype)

        assert time_steps == self.config.max_position_embeddings, (
            f"The conv frontend gave {time_steps} time steps per base chunk but "
            f"config.max_position_embeddings is "
            f"{self.config.max_position_embeddings}."
        )

        # Step 3: Gather base chunks into their block slots.
        # [num_chunks, t, H] -> [num_slots, t, H]
        x = x.index_select(0, chunk_slots)
        # -> [num_blocks, slots_per_block * t, H]
        x = x.reshape(num_blocks, slots_per_block * time_steps, -1)
        # Pad the block tail up to the kernel-aligned block size, e.g.
        # 128 - 9 * 13 = 11 at slots_per_block == 9.
        # -> [num_blocks, block_size, H]
        seq_pad = block_size - x.shape[1]
        if seq_pad > 0:
            x = F.pad(x, (0, 0, 0, seq_pad))

        # >>> PARALLELISM: DP scatter — each rank takes its slice of blocks <<<
        # Blocks are independent, so each rank only needs its own. shard_select
        # keeps the offset a runtime graph input, so all ranks share one graph.
        if self.dp_size > 1:
            assert num_blocks % self.dp_size == 0, (
                f"num_blocks ({num_blocks}) must be a multiple of dp_size "
                f"({self.dp_size}). Fix the dp-aligned block padding in "
                f"plan_audio_blocks."
            )
            x = shard_select(x, 0, self.dp_group)
            bound_min = shard_select(bound_min, 0, self.dp_group)
            bound_max = shard_select(bound_max, 0, self.dp_group)

        # Step 4: Transformer layers
        for layer in self.layers:
            x = layer(x, bound_min, bound_max)

        # Step 5: Final norm, then output projector. ln_post keeps d_model (1024).
        # The projector then maps d_model -> output_dim (2048 == text hidden_size).
        x = self.ln_post(x)
        x = self.proj2(gelu(self.proj1(x)))

        # >>> PARALLELISM: DP gather — combine blocks from all DP ranks <<<
        # The cache write below needs all blocks on each rank.
        if self.dp_size > 1:
            x = self.dp_group.all_gather(x, dim=0)

        # Step 6:  Write to the encoder cache buffer.
        if encoder_cache_buffer is not None:
            encoder_cache_buffer.index_put_((write_block_ids,), x)
            return encoder_cache_buffer
        # Module tests pass no cache, use the returned embeddings directly.
        return x

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    # HF checkpoint prefix for the audio tower weights.
    _AUDIO_TOWER_PREFIX = "thinker.audio_tower"

    @classmethod
    def build_weight_mappings(cls, encoder_layers: int) -> dict[str, str | list[str]]:
        """Build parameter name → checkpoint key(s) mapping for the audio encoder.

        The value is a single checkpoint key for most params, or a list of keys
        for the fused QKV weight/bias — Qwen3-ASR stores q/k/v separately, so the
        list ``[q, k, v]`` is what ``audio_qkv_weight_loader`` fuses.
        """
        p = cls._AUDIO_TOWER_PREFIX
        mappings: dict[str, str | list[str]] = {
            "conv2d1.weight": f"{p}.conv2d1.weight",
            "conv2d1.bias": f"{p}.conv2d1.bias",
            "conv2d2.weight": f"{p}.conv2d2.weight",
            "conv2d2.bias": f"{p}.conv2d2.bias",
            "conv2d3.weight": f"{p}.conv2d3.weight",
            "conv2d3.bias": f"{p}.conv2d3.bias",
            "conv_out.weight": f"{p}.conv_out.weight",  # bias=False
            "ln_post.weight": f"{p}.ln_post.weight",
            "ln_post.bias": f"{p}.ln_post.bias",
            # Output projector (proj1 -> GELU -> proj2).
            "proj1.weight": f"{p}.proj1.weight",
            "proj1.bias": f"{p}.proj1.bias",
            "proj2.weight": f"{p}.proj2.weight",
            "proj2.bias": f"{p}.proj2.bias",
        }

        for i in range(encoder_layers):
            lp, cp = f"layers.{i}", f"{p}.layers.{i}"

            mappings[f"{lp}.norm2.weight"] = f"{cp}.final_layer_norm.weight"
            mappings[f"{lp}.norm2.bias"] = f"{cp}.final_layer_norm.bias"

            mappings[f"{lp}.attn.norm.weight"] = f"{cp}.self_attn_layer_norm.weight"
            mappings[f"{lp}.attn.norm.bias"] = f"{cp}.self_attn_layer_norm.bias"
            mappings[f"{lp}.attn.qkv_weight"] = [
                f"{cp}.self_attn.q_proj.weight",
                f"{cp}.self_attn.k_proj.weight",
                f"{cp}.self_attn.v_proj.weight",
            ]
            mappings[f"{lp}.attn.qkv_bias"] = [
                f"{cp}.self_attn.q_proj.bias",
                f"{cp}.self_attn.k_proj.bias",
                f"{cp}.self_attn.v_proj.bias",
            ]
            mappings[f"{lp}.attn.proj_weight"] = f"{cp}.self_attn.out_proj.weight"
            mappings[f"{lp}.attn.proj_bias"] = f"{cp}.self_attn.out_proj.bias"

            mappings[f"{lp}.mlp.up_proj_weight"] = f"{cp}.fc1.weight"
            mappings[f"{lp}.mlp.up_proj_bias"] = f"{cp}.fc1.bias"
            mappings[f"{lp}.mlp.down_proj_weight"] = f"{cp}.fc2.weight"
            mappings[f"{lp}.mlp.down_proj_bias"] = f"{cp}.fc2.bias"

        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        cpu_mode: bool = True,
    ) -> None:
        """Load weights from a HuggingFace safetensors checkpoint."""
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_encoder_tp_group,
        )
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_group = get_neuron_encoder_tp_group()
        tp_rank = tp_group.rank_in_group
        tp_size = tp_group.world_size

        mappings = self.build_weight_mappings(self.config.encoder_layers)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)

        if cpu_mode:
            sd = checkpoint.load_sharded(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict
        else:
            sd = checkpoint.load_sharded_pipelined(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict

        self.load_state_dict(sd, strict=False, assign=True)
