# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (HYV3) Static-FP8 Implementation
====================================

Implementation of tencent/Hy3-FP8 for the Neuron backend.

Checkpoint format: every linear layer (attention, dense MLP, shared expert and
each routed expert) stores ``float8_e4m3fn`` weights with a scalar BF16
``weight_scale`` and a scalar static ``input_scale``. ``lm_head`` and
``embed_tokens`` are BF16. The MTP (next-n) layer stored after the decoder
stack is not loaded.

Supported parallelism: TP (attention heads / dense MLP), SP (prefill), EP for
the routed experts (``--enable-expert-parallel``; full EP by default), and DCP
(``--decode-context-parallel-size``) to shard the KV cache across the ranks
that hold a replicated KV head.

Kernel mapping:
  - Attention prefill: ``NF.qkv_proj`` (STATIC) -> torch QK-norm + RoPE ->
    flash / segmented attention -> ``NF.o_proj`` (STATIC).
  - Attention decode: ``NF.attention_decode`` megakernel (STATIC QKV/O, fused
    input RMSNorm, pre-RoPE QK-norm, in-kernel KV cache update).
  - Dense MLP / shared expert: ``NF.rmsnorm_quant`` + ``NF.mlp`` (STATIC).
  - Routed experts: torch router (sigmoid + selection-only bias), then
    ``NF.moe_cte`` (``shard_on_i``, per-channel FP8 weight dequant) for
    prefill and ``NF.moe_tkg`` (STATIC FP8 weights) for decode. Both run
    BF16 activations x FP8 weights.

FP8 range handling
------------------
The checkpoint is calibrated for OCP ``float8_e4m3fn`` (max 448). The STATIC
kernels use the 240-capped ``float8_e4m3`` on TRN2 (and ``moe_tkg`` defaults
to it everywhere). Both formats share exponent bias 7, so every value with
``|x| <= 240`` has the same encoding. We therefore halve every FP8 weight at
load time (an exact exponent shift for normal values) and double the matching
weight scale. Input (activation) scales are doubled as well so calibrated
activations quantize to ``|x/s| <= 224`` instead of clipping at 240.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    Hy3 specific.
  # <-- STATIC-FP8: ...        FP8 weight / scale handling.
"""

import logging

import nki.language as nl
import torch
from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    QuantizationType,
)
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_tp_group,
    get_world_group,
)

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.model import (
    _DECODE_MASK_CACHE_KEY,
    _validate_dcp_decode_sequence_length,
)
from vllm_neuron.model.llama3.model_static_fp8 import _read_scalar_from_slice
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import rank_to_row
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import HYV3Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 0: Static-FP8 helpers
# =============================================================================

_FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension the STATIC kernels expect for pre-broadcast scales.
_PMAX = 128

# <-- STATIC-FP8: see module docstring. Halving is exact for normal fp8 values
# and moves the 448-range checkpoint into the 240-range kernel dtype.
_FP8_WEIGHT_DOWNSCALE = 0.5
_FP8_SCALE_COMPENSATION = 1.0 / _FP8_WEIGHT_DOWNSCALE

# CTE MLP kernel produces wrong results when the per-rank intermediate size is
# not a multiple of 128, so the dense MLP intermediate is zero-padded per rank.
_MLP_INTERMEDIATE_ALIGN = 128

# Blockwise MoE prefill block size (shard_on_i requires a multiple of 256).
_MOE_BLOCK_SIZE = 256


def _halve_fp8(w: torch.Tensor) -> torch.Tensor:
    assert w.dtype == _FP8_DTYPE, f"expected a float8_e4m3fn weight, got {w.dtype}"
    return (w.float() * _FP8_WEIGHT_DOWNSCALE).to(_FP8_DTYPE)


def _with_fp8_halving(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    base = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return _halve_fp8(base(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def _scalar_scale_loader(num_slices: int = 1, cols: int = 1) -> SafetensorsWeightLoader:
    """Scalar scale(s) -> ``[128, cols]`` fp32, multiplied by the FP8 compensation.

    ``num_slices == cols``: one column per checkpoint scalar (fused QKV weight
    scale). ``cols == 1`` with several slices: the scalars must agree (shared
    activation scale of a fused projection) and a single column is returned.
    """

    def transform(slices, rank):
        assert len(slices) == num_slices, (
            f"expected {num_slices} slices, got {len(slices)}"
        )
        scalars = [_read_scalar_from_slice(s) for s in slices]
        if cols == 1:
            for s in scalars[1:]:
                assert torch.equal(scalars[0], s), (
                    "fused projections must share one static input scale, got "
                    f"{[v.item() for v in scalars]}"
                )
            scalars = scalars[:1]
        stacked = torch.cat(scalars).reshape(1, cols) * _FP8_SCALE_COMPENSATION
        return stacked.expand(_PMAX, cols).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _row_shard(slice_obj, start: int, size: int, dim: int) -> torch.Tensor:
    """Read ``[start, start+size)`` along ``dim`` of a 2-D slice, zero-padding past the end."""
    total = slice_obj.get_shape()[dim]
    end = min(start + size, total)
    if dim == 0:
        t = slice_obj[start:end, :]
    else:
        t = slice_obj[:, start:end]
    if end - start < size:
        pad_shape = list(t.shape)
        pad_shape[dim] = size - (end - start)
        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=dim)
    return t


def _padded_mlp_loader(
    is_down: bool, shard_size: int, padded_size: int, tp_rank: int
) -> SafetensorsWeightLoader:
    """TP-shard a dense gate/up ([I, H]) or down ([H, I]) weight and zero-pad I.

    Returns ``[H, padded]`` for gate/up and ``[padded, H]`` for down, halved
    into the kernel fp8 range.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        w = _row_shard(slices[0], tp_rank * shard_size, shard_size, 1 if is_down else 0)
        w = w.T  # gate/up: [H, shard], down: [shard, H]
        if shard_size < padded_size:
            pad_dim = 0 if is_down else 1
            pad_shape = list(w.shape)
            pad_shape[pad_dim] = padded_size - shard_size
            w = torch.cat([w, torch.zeros(pad_shape, dtype=w.dtype)], dim=pad_dim)
        return _halve_fp8(w.contiguous())

    return SafetensorsWeightLoader(transform=transform)


def _expert_gate_up_loader(
    num_experts: int, i_shard: int, i_rank: int
) -> SafetensorsWeightLoader:
    """``[gate_0..gate_{L-1}, up_0..up_{L-1}]`` ([I, H] each) -> ``[L, H, 2, I_tp]``."""

    def transform(slices, rank):
        assert len(slices) == 2 * num_experts
        experts = []
        for e in range(num_experts):
            g = slices[e][i_rank * i_shard : (i_rank + 1) * i_shard, :]
            u = slices[num_experts + e][i_rank * i_shard : (i_rank + 1) * i_shard, :]
            experts.append(torch.stack([g.T, u.T], dim=1))  # [H, 2, I_tp]
        return _halve_fp8(torch.stack(experts).contiguous())

    return SafetensorsWeightLoader(transform=transform)


def _expert_down_loader(
    num_experts: int, i_shard: int, i_rank: int
) -> SafetensorsWeightLoader:
    """``[down_0..down_{L-1}]`` ([H, I] each) -> ``[L, I_tp, H]``."""

    def transform(slices, rank):
        assert len(slices) == num_experts
        experts = [s[:, i_rank * i_shard : (i_rank + 1) * i_shard].T for s in slices]
        return _halve_fp8(torch.stack(experts).contiguous())

    return SafetensorsWeightLoader(transform=transform)


def _expert_scale_loader(
    num_experts: int, num_groups: int, shape_fn
) -> SafetensorsWeightLoader:
    """Per-expert scalar weight scales -> kernel layout.

    ``slices`` is grouped by projection (``[gate_0..gate_{L-1}, up_0..]``).
    ``shape_fn`` maps the ``[L, num_groups]`` fp32 scale matrix to the layout
    the kernel wants.
    """

    def transform(slices, rank):
        assert len(slices) == num_experts * num_groups
        vals = (
            torch.stack([_read_scalar_from_slice(s)[0] for s in slices])
            .view(num_groups, num_experts)
            .T
        )  # [L, groups]
        return shape_fn(vals * _FP8_SCALE_COMPENSATION).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _shard_kv_by_dcp_rank(
    kv: torch.Tensor, dcp_size: int, dcp_rank: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Slice a [n_heads, tokens, head_dim] K/V to this DCP rank's interleaved
    block shard ((pos // block_size) % dcp_size == dcp_rank).

    ``dcp_rank`` is a runtime tensor so the traced graph is identical on every
    rank (a Python int would compile one NEFF per rank).
    """
    n_heads, tokens, head_dim = kv.shape
    blocks = kv.reshape(
        n_heads, tokens // (dcp_size * block_size), dcp_size, block_size, head_dim
    )
    local = torch.index_select(blocks, 2, dcp_rank.reshape(1).long())
    return local.reshape(n_heads, -1, head_dim).contiguous()


def _fp32_param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


# =============================================================================
# Section 1: RMSNorm + RoPE
# <-- MODEL-SPECIFIC: standard RMSNorm, default RoPE (rotate_half)
# =============================================================================


class HYV3RMSNorm(nn.Module):
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HYV3RotaryEmbedding(nn.Module):
    def __init__(self, config: HYV3Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        freqs = position_ids[:, None].float() * inv_freq[None, :]  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 2: Attention (static FP8, QK-norm before RoPE)
# =============================================================================


class HYV3Attention(nn.Module):
    """GQA attention with TP head sharding and static-FP8 projections.

    >>> PARALLELISM: TP <<<
    - Q heads sharded across TP; KV heads replicated when TP > num KV heads.
    - Prefill: RMSNorm (SP) -> all-gather -> attention -> reduce-scatter.
    - Decode: fused megakernel (input RMSNorm fused) -> all-reduce.

    >>> PARALLELISM: DCP (decode context parallelism) <<<
    With TP > num_kv_heads every KV head is replicated on TP / num_kv_heads
    ranks. DCP turns those replicas into a sequence-sharded group: each rank
    stores only the KV blocks it owns (interleaved by block_size), and
    attention combines the per-rank partial results with an LSE correction
    (``segmented_attention_cp`` in prefill, the DCP path of
    ``attention_decode`` in decode). KV capacity grows by ``dcp_size``.

    <-- MODEL-SPECIFIC: per-head q_norm / k_norm applied before RoPE.
    """

    def __init__(self, config: HYV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: TP group / head sharding <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
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

        # >>> PARALLELISM: DCP group = ranks sharing one replicated KV head <<<
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None
        if self.dcp_size > 1 and self.num_kv_replicas % self.dcp_size != 0:
            raise ValueError(
                f"decode_context_parallel_size={self.dcp_size} must divide the "
                f"KV replication factor {self.num_kv_replicas} "
                f"(tensor_parallel_size / num_key_value_heads)."
            )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # <-- STATIC-FP8: fp8 weights + [128, 3] / [128, 1] fp32 scales.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.qkv_weight_scale = _fp32_param(_PMAX, 3)
        self.qkv_input_scale = _fp32_param(_PMAX, 1)
        self.o_weight_scale = _fp32_param(_PMAX, 1)
        self.o_input_scale = _fp32_param(_PMAX, 1)

        # <-- MODEL-SPECIFIC: per-head QK-norm (head_dim RMSNorm, replicated)
        self.q_norm = HYV3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = HYV3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally via bind_kv_cache() (BF16 only).
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.qkv_proj_weight,
            _with_fp8_halving(
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=1,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=self.num_kv_replicas,
                )
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            _with_fp8_halving(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.q_size,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                )
            ),
        )
        set_weight_loader(self.qkv_weight_scale, _scalar_scale_loader(3, cols=3))
        set_weight_loader(self.qkv_input_scale, _scalar_scale_loader(3, cols=1))
        set_weight_loader(self.o_weight_scale, _scalar_scale_loader())
        set_weight_loader(self.o_input_scale, _scalar_scale_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        ln_w: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        # >>> PARALLELISM: DCP groups are consecutive TP ranks <<<
        dcp_rank = rank % self.dcp_size if self.dcp_size > 1 else None
        if is_decode:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                ln_w,
                dcp_rank,
            )

        # Pre-attention RMSNorm in the SP region; the STATIC qkv kernel needs bf16 input.
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        hidden_states = (x * ln_w.view(1, -1)).to(self.dtype)
        # >>> PARALLELISM: All-gather from SP <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, position_embeddings, attn_metadata, dcp_rank
        )

    def _qk_norm(self, t: torch.Tensor, norm: HYV3RMSNorm) -> torch.Tensor:
        tf = t.to(torch.float32)
        tf = tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return tf * norm.weight.to(torch.float32)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        dcp_rank: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        meta = attn_metadata[self.layer_name]

        # Step 1: QKV projection (STATIC fp8)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            quantization_type=QuantizationType.STATIC,
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2: <-- MODEL-SPECIFIC: QK-norm then rotate_half RoPE (fp32)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(0).to(torch.float32)
        sin = sin.unsqueeze(0).to(torch.float32)
        q = self._qk_norm(q, self.q_norm)
        k = self._qk_norm(k, self.k_norm)
        q = (q * cos + _rotate_half(q) * sin).to(self.dtype)
        k = (k * cos + _rotate_half(k) * sin).to(self.dtype)

        # Step 3: KV cache write (padding slots are -1 and land outside the cache).
        # >>> PARALLELISM: with DCP each rank keeps only its interleaved
        # block shard; the runner's slot_mapping already covers just that shard.
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        if self.dcp_size > 1:
            k_write = _shard_kv_by_dcp_rank(k, self.dcp_size, dcp_rank, block_size)
            v_write = _shard_kv_by_dcp_rank(v, self.dcp_size, dcp_rank, block_size)
        else:
            k_write, v_write = k, v
        nkh = self.num_key_value_heads_per_rank
        index = (
            (slot_mapping // block_size).repeat(nkh),
            torch.arange(nkh, dtype=torch.long, device=q.device).repeat_interleave(
                slot_mapping.shape[0]
            ),
            (slot_mapping % block_size).repeat(nkh),
        )
        self.k_cache.index_put_(
            index, k_write.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        )
        self.v_cache.index_put_(
            index, v_write.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        )

        # Step 4: Attention
        kv_segment_size = meta.get("kv_segment_size")
        if self.dcp_size > 1:
            # Full Q against this rank's local KV (cached prior + current
            # shard), LSE-combined across the DCP group.
            cached_seq_len = meta.get("cached_seq_len")
            attn_output = NF.segmented_attention_cp(
                q=q,
                k_local=k_write,
                v_local=v_write,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=(
                    cached_seq_len // self.dcp_size
                    if cached_seq_len is not None
                    else None
                ),
                block_size=block_size,
                cp_rank=dcp_rank.reshape(1, 1).to(torch.int32),
                cp_world_size=self.dcp_size,
                cp_kv_cache_interleave_size=block_size,
                cp_group=self.dcp_group,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        elif kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=meta.get("cached_seq_len"),
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Step 5: Output projection (STATIC fp8)
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0),
            self.o_proj_weight,
            None,
            quantization_type=QuantizationType.STATIC,
            weight_scales=self.o_weight_scale,
            input_scales=self.o_input_scale,
        ).squeeze(0)

        # >>> PARALLELISM: Reduce-scatter back to SP <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        ln_w: torch.Tensor,
        dcp_rank: torch.Tensor | None,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        S_ctx = meta["max_blocks_per_seq"] * block_size

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        _validate_dcp_decode_sequence_length(self.dcp_size, S_decode)
        X = hidden_states.to(self.dtype).view(B, S_decode, hidden)

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

        # >>> PARALLELISM: DCP — the block table is already this rank's shard
        # (S_ctx is local). Mask the prior tokens this rank owns and let only
        # the owning rank count the active token, so it is combined once.
        local_filled = None
        dcp_active_mask = None
        if self.dcp_size > 1:
            W = self.dcp_size
            R = dcp_rank.to(torch.float32)
            n = positions.view(B, S_decode)[:, 0:1].to(torch.float32)
            vblock = block_size * W
            full_vblocks = torch.div(n, vblock, rounding_mode="trunc")
            remaining = n - full_vblocks * vblock
            local_filled = full_vblocks * block_size + torch.clamp(
                remaining - R * block_size, min=0, max=block_size
            )
            owner = torch.div(remaining, block_size, rounding_mode="trunc") % W
            dcp_active_mask = (owner == R).float()

        # The decode mask is identical for every layer: build it once per step.
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=positions.view(1, B * S_decode).to(torch.float32),
                bs=B,
                q_head=self.num_attention_heads_per_rank * self.dcp_size,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
                local_filled_slots=local_filled,
                dcp_active_mask=dcp_active_mask,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=self.rms_norm_eps,
            rmsnorm_X_gamma=ln_w.view(1, -1),
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=self.qkv_input_scale,
            # <-- MODEL-SPECIFIC: pre-RoPE QK-norm with learned gammas
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
            W_out=self.o_proj_weight,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=self.o_weight_scale,
            input_dequant_scale_out=self.o_input_scale,
            transposed_out=False,
            out_in_sb=False,
            k_scale=None,
            v_scale=None,
            dcp_size=self.dcp_size,
            dcp_group=self.dcp_group,
            dcp_active_owner=dcp_active_mask,
        )

        # >>> PARALLELISM: TP all-reduce of the O projection (after the DCP combine) <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.view(tokens, hidden)


# =============================================================================
# Section 3: Static-FP8 gated MLP (dense layer + shared expert)
# =============================================================================


class HYV3StaticFP8MLP(nn.Module):
    """SiLU-gated MLP with fused RMSNorm + static FP8 activation quant.

    The caller owns the collectives: ``forward`` returns this rank's
    (possibly partial) result for the tokens it was given.

    ``tp_sharded=True``: intermediate dim sharded across TP (dense layer).
    ``tp_sharded=False``: full weights on every rank (shared expert); used on
    SP-local tokens in prefill and on all tokens in decode, so no reduction
    is needed.
    """

    def __init__(self, config: HYV3Config, intermediate_size: int, tp_sharded: bool):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.tp_sharded = tp_sharded and self.world_size > 1
        self.hidden_size = config.hidden_size
        self.act_dtype = config.torch_dtype

        if self.tp_sharded:
            shard = -(-intermediate_size // self.world_size)
        else:
            shard = intermediate_size
        self.intermediate_shard = shard
        self.intermediate_size_per_rank = (
            -(-shard // _MLP_INTERMEDIATE_ALIGN) * _MLP_INTERMEDIATE_ALIGN
        )
        I = self.intermediate_size_per_rank

        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, I, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, I, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(I, self.hidden_size, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.gate_weight_scale = _fp32_param(_PMAX, 1)
        self.up_weight_scale = _fp32_param(_PMAX, 1)
        self.down_weight_scale = _fp32_param(_PMAX, 1)
        self.gate_up_input_scale = _fp32_param(_PMAX, 1)
        self.down_input_scale = _fp32_param(_PMAX, 1)

        tp_rank = self.tp_group.rank_in_group if self.tp_sharded else 0
        set_weight_loader(
            self.gate_proj_weight, _padded_mlp_loader(False, shard, I, tp_rank)
        )
        set_weight_loader(
            self.up_proj_weight, _padded_mlp_loader(False, shard, I, tp_rank)
        )
        set_weight_loader(
            self.down_proj_weight, _padded_mlp_loader(True, shard, I, tp_rank)
        )
        for name in ("gate_weight_scale", "up_weight_scale", "down_weight_scale"):
            set_weight_loader(getattr(self, name), _scalar_scale_loader())
        set_weight_loader(self.gate_up_input_scale, _scalar_scale_loader(2, cols=1))
        set_weight_loader(self.down_input_scale, _scalar_scale_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        ln_w: torch.Tensor,
        eps: float,
        is_prefill: bool,
    ) -> torch.Tensor:
        # >>> PARALLELISM: SP all-gather (TP-sharded MLP only) <<<
        gather = is_prefill and self.tp_sharded
        num_tokens = hidden_states.shape[0] * (self.world_size if gather else 1)

        # The kernel picks CTE vs TKG from its token count. CTE needs fp8 input
        # (explicit fused RMSNorm + static quant); TKG needs bf16 input and
        # fuses the RMSNorm itself (it fails to compile on fp8 input).
        if num_tokens > TKG_BS_SEQLEN_THRESHOLD:
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=QuantizationType.STATIC,
            )
            norm_type, mlp_ln_w = NormType.NO_NORM, None
        else:
            norm_type, mlp_ln_w = NormType.RMS_NORM, ln_w.view(1, -1)
        if gather:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=norm_type,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_weight_scale,
            up_w_scale=self.up_weight_scale,
            down_w_scale=self.down_weight_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype="bfloat16",
        )


class HYV3DenseMLP(nn.Module):
    """Dense MLP layer (first_k_dense_replace layers), TP-sharded."""

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.mlp = HYV3StaticFP8MLP(config, config.intermediate_size, tp_sharded=True)
        self.tp_group = self.mlp.tp_group
        self.world_size = self.mlp.world_size

    def forward(self, hidden_states, ln_w, eps, is_prefill, rank):
        output = self.mlp(hidden_states, ln_w, eps, is_prefill)
        # >>> PARALLELISM: combine TP partial sums <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 4: MoE (sigmoid router + shared expert + FP8 routed experts)
# =============================================================================


class HYV3MoE(nn.Module):
    """Hy3 sparse MoE block.

    >>> PARALLELISM: EP (+ optional TP within an EP group) for routed experts <<<
    - Experts are placed linearly: EP rank k owns ``[k*L, (k+1)*L)``.
    - Router and shared expert are replicated on every rank.
    - Prefill: router + shared expert on SP-local tokens; tokens and
      affinities are all-gathered for the routed experts, whose partial
      outputs are reduce-scattered back to SP.
    - Decode: tokens are replicated; routed partial outputs are all-reduced.

    <-- MODEL-SPECIFIC routing:
        scores = sigmoid(x @ W_r^T)                      (fp32)
        top-k selection on scores + expert_bias          (bias selects only)
        weights = scores[topk] / sum(scores[topk]) * router_scaling_factor
    """

    def __init__(self, config: HYV3Config):
        super().__init__()
        from vllm.config import get_current_vllm_config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.hidden_size = config.hidden_size
        self.total_num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.router_scaling_factor = config.router_scaling_factor
        self.route_norm = config.route_norm

        # >>> PARALLELISM: EP configuration <<<
        vllm_config = get_current_vllm_config()
        if vllm_config.parallel_config.enable_expert_parallel:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            self.ep_tp_rank = self.ep_tp_group.rank_in_group
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.tp_degree = self.world_size
            self.ep_tp_rank = self.tp_group.rank_in_group
        self.global_world_size = get_world_group().world_size

        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree
        L, I_tp, H = (
            self.num_local_experts,
            self.intermediate_size_per_rank,
            self.hidden_size,
        )

        # <-- MODEL-SPECIFIC: router (fp32, replicated) + selection-only bias
        self.router_weight = _fp32_param(self.total_num_experts, H)
        self.expert_bias = _fp32_param(self.total_num_experts)

        # <-- STATIC-FP8: routed expert weights (local experts, I sharded by tp_degree)
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(L, H, 2, I_tp, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(L, I_tp, H, dtype=_FP8_DTYPE), requires_grad=False
        )
        # Prefill (moe_cte shard_on_i): per-channel dequant scales.
        self.gate_up_proj_scale = _fp32_param(L, 1, 2 * I_tp)
        self.down_proj_scale = _fp32_param(L, 1, H)
        # Decode (moe_tkg STATIC): per-expert scalar dequant scales.
        self.gate_up_proj_scale_tkg = _fp32_param(L, 2, 1)
        self.down_proj_scale_tkg = _fp32_param(L, 1)

        # <-- MODEL-SPECIFIC: shared expert(s), full weights on every rank
        self.shared_mlp = HYV3StaticFP8MLP(
            config,
            config.moe_intermediate_size * config.num_shared_experts,
            tp_sharded=False,
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        L, I_tp, H = (
            self.num_local_experts,
            self.intermediate_size_per_rank,
            self.hidden_size,
        )
        local_expert_indices = list(range(self.ep_rank * L, (self.ep_rank + 1) * L))

        def ep(loader):
            return expert_parallel_grouped_loader(
                local_expert_indices, loader, total_num_experts=self.total_num_experts
            )

        set_weight_loader(
            self.gate_up_proj_weight,
            ep(_expert_gate_up_loader(L, I_tp, self.ep_tp_rank)),
        )
        set_weight_loader(
            self.down_proj_weight, ep(_expert_down_loader(L, I_tp, self.ep_tp_rank))
        )
        set_weight_loader(
            self.gate_up_proj_scale,
            ep(
                _expert_scale_loader(
                    L,
                    2,
                    lambda s: (
                        s.view(L, 2, 1).expand(L, 2, I_tp).reshape(L, 1, 2 * I_tp)
                    ),
                )
            ),
        )
        set_weight_loader(
            self.down_proj_scale,
            ep(_expert_scale_loader(L, 1, lambda s: s.view(L, 1, 1).expand(L, 1, H))),
        )
        set_weight_loader(
            self.gate_up_proj_scale_tkg,
            ep(_expert_scale_loader(L, 2, lambda s: s.view(L, 2, 1))),
        )
        set_weight_loader(
            self.down_proj_scale_tkg,
            ep(_expert_scale_loader(L, 1, lambda s: s.view(L, 1))),
        )

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense ``[T, E]`` fp32 affinities and ``[T, K]`` expert indices."""
        logits = hidden_states.to(torch.float32) @ self.router_weight.T
        scores = torch.sigmoid(logits)
        _, top_k_index = torch.topk(
            scores + self.expert_bias, self.num_experts_per_token, dim=-1
        )
        top_k_weights = scores.gather(1, top_k_index)
        if self.route_norm:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        top_k_weights = top_k_weights * self.router_scaling_factor
        affinities = torch.zeros_like(scores).scatter(1, top_k_index, top_k_weights)
        return affinities, top_k_index

    def _ep_row(self, rank: torch.Tensor) -> torch.Tensor:
        """EP rank of this worker derived from the runtime rank tensor.

        Using the tensor (not the Python int) keeps the graph rank-agnostic.
        """
        return rank_to_row(rank, self.tp_degree, self.global_world_size)

    def forward(self, hidden_states, ln_w, eps, is_prefill, rank):
        # <-- MODEL-SPECIFIC: post-attention RMSNorm feeds router + routed experts
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        normed = (x * ln_w.to(torch.float32)).to(hidden_states.dtype)
        affinities, top_k_index = self._route(normed)

        # Shared expert: full weights; SP-local tokens in prefill, all tokens
        # in decode. Its output is complete on every rank.
        shared_output = self.shared_mlp(hidden_states, ln_w, eps, is_prefill)

        if is_prefill:
            routed = self._forward_prefill(normed, affinities, rank)
        else:
            routed = self._forward_decode(normed, affinities, top_k_index, rank)
        return routed + shared_output

    def _forward_prefill(self, normed, affinities, rank):
        # >>> PARALLELISM: All-gather tokens + affinities from SP <<<
        if self.world_size > 1:
            affinities = self.tp_group.all_gather(affinities, dim=0)
            normed = self.tp_group.all_gather(normed, dim=0)
        T = normed.shape[0]

        # >>> PARALLELISM: keep only this rank's experts <<<
        if self.ep_degree > 1:
            local_expert_indices = self._ep_row(
                rank
            ).long() * self.num_local_experts + torch.arange(
                self.num_local_experts, device=normed.device, dtype=torch.long
            )
            affinities = NF.get_local_expert_affinities(
                affinities, local_expert_indices
            )

        masked, token_position_to_id, block_to_expert, conditions = (
            NF.build_blockwise_mapping(
                expert_affinities=affinities,
                num_local_experts=self.num_local_experts,
                num_experts_per_token=self.num_experts_per_token,
                block_size=_MOE_BLOCK_SIZE,
                moe_group=self.ep_tp_group,
                tp_degree=self.tp_degree,
                rank=rank,
            )
        )

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i,
            conditions=conditions,
            hidden_states=normed,
            expert_affinities_masked=masked,
            gate_up_proj_weight=self.gate_up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            gate_up_proj_scale=self.gate_up_proj_scale,
            down_proj_scale=self.down_proj_scale,
            activation_function=ActFnType.SiLU,
            block_size=_MOE_BLOCK_SIZE,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )[:T]

        # >>> PARALLELISM: combine expert partials and return to SP <<<
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output

    def _forward_decode(self, normed, affinities, top_k_index, rank):
        rank_id = self._ep_row(rank).to(torch.int32).view(1, 1)
        # moe_tkg selects STATIC from the presence of input scales; on TRN2 the
        # activations stay BF16 so the values are unused.
        dummy_in_scale = torch.ones(
            (self.num_local_experts, 1), dtype=torch.float32, device=normed.device
        )
        output = NF.moe_tkg(
            hidden_input=normed,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            expert_affinities=affinities,
            expert_index=top_k_index.to(torch.int32),
            is_all_expert=True,
            rank_id=rank_id,
            expert_gate_up_weights_scale=self.gate_up_proj_scale_tkg,
            expert_down_weights_scale=self.down_proj_scale_tkg,
            expert_gate_up_input_scale=dummy_in_scale,
            expert_down_input_scale=dummy_in_scale,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )
        # >>> PARALLELISM: combine expert partials <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 5: Decoder layer / backbone
# =============================================================================


class HYV3DecoderLayer(nn.Module):
    def __init__(self, config: HYV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = HYV3Attention(config, layer_idx)
        self.mlp = (
            HYV3MoE(config) if config.is_moe_layer(layer_idx) else HYV3DenseMLP(config)
        )

    def forward(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata,
        is_decode,
        rank,
    ):
        # The norms are fused into the attention / MLP modules.
        attn_out = self.self_attn(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            ln_w=self.input_layernorm.weight,
            is_decode=is_decode,
            rank=rank,
        )
        hidden_states = hidden_states + attn_out
        mlp_out = self.mlp(
            hidden_states,
            ln_w=self.post_attention_layernorm.weight,
            eps=self.rms_norm_eps,
            is_prefill=not is_decode,
            rank=rank,
        )
        return hidden_states + mlp_out


class HYV3Model(nn.Module):
    """>>> PARALLELISM: SP during prefill — embed -> chunk -> layers -> all-gather. <<<"""

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.config = config
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
                pad_shard=True,
            ),
        )
        self.layers = nn.ModuleList(
            [HYV3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = HYV3RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata,
        is_prefill: bool,
        rank=None,
        inputs_embeds=None,
        is_token_ids=None,
    ):
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                is_decode=not is_prefill,
                rank=rank,
            )
        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# =============================================================================
# Section 6: Causal LM + weight loading
# =============================================================================


class HYV3ForCausalLM(nn.Module):
    """Hy3 model with column-parallel LM head."""

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.config = config
        self.model = HYV3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
        )

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
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        first = attn_metadata["layers.0.self_attn"]
        is_prefill = first["max_query_len"] > first["decode_token_threshold"]

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata,
            is_prefill=is_prefill,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        hidden_states = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered_logits = (
            self.tp_group.all_gather(logits, dim=1) if self._gather_logits else None
        )
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(HYV3Config.from_configs(hf_config, neuron_config))

    # ── KV cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
                for i, layer in enumerate(self.model.layers)
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    # ── Weight loading ────────────────────────────────────────────────────

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        """<-- MODEL-SPECIFIC: model parameter name -> checkpoint key(s)."""
        cfg = self.config
        E = cfg.num_experts
        m: dict[str, str | list[str]] = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": (
                "model.embed_tokens.weight"
                if cfg.tie_word_embeddings
                else "lm_head.weight"
            ),
            "model.norm.weight": "model.norm.weight",
        }

        def fp8_linear(dst: str, src: str):
            m[f"{dst}_weight"] = f"{src}.weight"

        for i, layer in enumerate(self.model.layers):
            p = f"model.layers.{i}"
            a = f"{p}.self_attn"
            m[f"{a}.qkv_proj_weight"] = [f"{a}.{n}_proj.weight" for n in "qkv"]
            m[f"{a}.qkv_weight_scale"] = [f"{a}.{n}_proj.weight_scale" for n in "qkv"]
            m[f"{a}.qkv_input_scale"] = [f"{a}.{n}_proj.input_scale" for n in "qkv"]
            m[f"{a}.o_proj_weight"] = f"{a}.o_proj.weight"
            m[f"{a}.o_weight_scale"] = f"{a}.o_proj.weight_scale"
            m[f"{a}.o_input_scale"] = f"{a}.o_proj.input_scale"
            m[f"{a}.q_norm.weight"] = f"{a}.q_norm.weight"
            m[f"{a}.k_norm.weight"] = f"{a}.k_norm.weight"
            m[f"{p}.input_layernorm.weight"] = f"{p}.input_layernorm.weight"
            m[f"{p}.post_attention_layernorm.weight"] = (
                f"{p}.post_attention_layernorm.weight"
            )

            def gated_mlp(dst: str, src: str):
                for n in ("gate", "up", "down"):
                    m[f"{dst}.{n}_proj_weight"] = f"{src}.{n}_proj.weight"
                    m[f"{dst}.{n}_weight_scale"] = f"{src}.{n}_proj.weight_scale"
                m[f"{dst}.gate_up_input_scale"] = [
                    f"{src}.gate_proj.input_scale",
                    f"{src}.up_proj.input_scale",
                ]
                m[f"{dst}.down_input_scale"] = f"{src}.down_proj.input_scale"

            if isinstance(layer.mlp, HYV3DenseMLP):
                gated_mlp(f"{p}.mlp.mlp", f"{p}.mlp")
                continue

            mp = f"{p}.mlp"
            m[f"{mp}.router_weight"] = f"{mp}.router.gate.weight"
            m[f"{mp}.expert_bias"] = f"{mp}.expert_bias"

            def experts(proj: str, suffix: str, mp: str = mp) -> list[str]:
                return [f"{mp}.experts.{e}.{proj}.{suffix}" for e in range(E)]

            gate_up_w = experts("gate_proj", "weight") + experts("up_proj", "weight")
            gate_up_s = experts("gate_proj", "weight_scale") + experts(
                "up_proj", "weight_scale"
            )
            m[f"{mp}.gate_up_proj_weight"] = gate_up_w
            m[f"{mp}.down_proj_weight"] = experts("down_proj", "weight")
            m[f"{mp}.gate_up_proj_scale"] = gate_up_s
            m[f"{mp}.gate_up_proj_scale_tkg"] = gate_up_s
            m[f"{mp}.down_proj_scale"] = experts("down_proj", "weight_scale")
            m[f"{mp}.down_proj_scale_tkg"] = experts("down_proj", "weight_scale")
            gated_mlp(f"{mp}.shared_mlp", f"{mp}.shared_mlp")
        return m

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self._weight_mappings(), device
        ).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)
        for name, param in self.named_parameters():
            if param.is_meta:
                raise RuntimeError(
                    f"Parameter {name} was not loaded from the checkpoint"
                )

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """CPU compile: no scale-derived compile-time constants to load."""
