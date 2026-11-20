# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT encoder for embedding (pooling runner)
=================================================

ModernBERT is a bidirectional encoder. This module serves it with the pooling
runner (CLS pooling, then L2 normalize). It has no LM head and no decode phase.

    input_ids -> tok_embeddings -> LayerNorm
              -> N x [ h = h + Attn(attn_norm(h)); h = h + GeGLU(mlp_norm(h)) ]
              -> final_norm -> [T, H] (the pooler reads the CLS token)

>>> PARALLELISM: Tensor parallel. Each rank holds ``num_heads / tp`` heads and
    ``intermediate_size / tp`` MLP channels. The hidden state stays replicated.
    One all-reduce follows the attention output projection, and one all-reduce
    follows the MLP down projection. The vocabulary and the norms are
    replicated.

<-- MODEL-SPECIFIC:
  - Bidirectional attention through ``NF.flash_attention`` with per-query
    KV bounds (``causal_mask=False``). A "full_attention" layer attends to the
    full sequence. A "sliding_attention" layer attends to keys at a distance
    of ``local_attention // 2`` or less.
  - Each layer type has its own RoPE theta.
  - LayerNorm without bias. Layer 0 has no attention norm (HF uses Identity).
  - GeGLU MLP: ``Wi`` holds ``[input; gate]`` and the output is
    ``Wo(gelu(input) * gate)``.

The model gets the sequence boundaries from ``positions``. The pooling runner
packs complete prompts (vLLM turns off chunked prefill and prefix caching for
an encoder-only pooling model), so the token at flat index ``i`` with position
``p`` belongs to the sequence that starts at ``i - p``.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import FULL_ATTENTION, ModernBertConfig

logger = logging.getLogger(__name__)

# The attention_cte kernel reads the per-query bounds in tiles of 128 rows.
# Pad the token dimension to a multiple of this value.
ATTN_SEQ_ALIGN = 128


# =============================================================================
# Section 1: Helpers (the tests use them as the port side of the comparison)
# =============================================================================


def modernbert_rotary(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RoPE ``(cos, sin)`` tables ``[T, head_dim]`` for ``positions``.

    The calculation is in FP32. The result is cast to ``dtype``, as in HF
    ``ModernBertRotaryEmbedding``.
    """
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=positions.device).to(
                torch.float32
            )
            / head_dim
        )
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def modernbert_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to ``x`` ``[T, N, D]`` in FP32, as in HF ``apply_rotary_pos_emb``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x32 = x.float()
    return ((x32 * cos) + (_rotate_half(x32) * sin)).to(x.dtype)


def modernbert_attention_bounds(
    positions: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the KV bounds ``[T, 1]`` int32 for the full and sliding layers.

    The result is ``(full_min, full_max, sliding_min, sliding_max)``. A minimum
    is inclusive and a maximum is exclusive. Each range contains the query
    token, so no row is empty.

    The token at flat index ``i`` with position ``p`` belongs to the sequence
    that starts at ``i - p``. Tokens with the same start form one sequence.
    A padding token gets its own start, so it never joins a real sequence.
    """
    num_tokens = positions.shape[0]
    idx = torch.arange(num_tokens, dtype=torch.int32, device=positions.device)
    start = idx - positions.to(torch.int32)
    start = torch.minimum(torch.clamp(start, min=0), idx)
    same_seq = start[:, None] == start[None, :]
    end = (same_seq.to(torch.int32) * (idx + 1)[None, :]).amax(dim=1)

    sliding_min = torch.maximum(start, idx - window)
    sliding_max = torch.minimum(end, idx + window + 1)
    return (
        start.unsqueeze(-1),
        end.unsqueeze(-1),
        sliding_min.unsqueeze(-1),
        sliding_max.unsqueeze(-1),
    )


def modernbert_qkv_weight_loader(
    hidden_size: int, shard_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """Shard the fused HF ``Wqkv`` ``[3*H, H]`` into a ``[H, 3*shard]`` parameter.

    HF views the ``Wqkv`` output as ``(3, heads, head_dim)``. Thus the rows are
    all Q heads, then all K heads, then all V heads. Each rank takes its head
    rows from each of the three blocks.
    """

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "modernbert_qkv_weight_loader takes one tensor"
        r = rank % num_shards
        parts = [
            slices[0][
                j * hidden_size + r * shard_size : j * hidden_size
                + (r + 1) * shard_size,
                :,
            ]
            for j in range(3)
        ]
        return torch.cat(parts, dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def modernbert_geglu_weight_loader(
    intermediate_size: int, shard_size: int, num_shards: int, half: int
) -> SafetensorsWeightLoader:
    """Shard one half of the HF ``Wi`` ``[2*I, H]`` into a ``[H, shard]`` parameter.

    ``half=0`` selects the "input" half (the activation input). ``half=1``
    selects the "gate" half (the multiplier).
    """

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "modernbert_geglu_weight_loader takes one tensor"
        r = rank % num_shards
        lo = half * intermediate_size + r * shard_size
        return slices[0][lo : lo + shard_size, :].T

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Section 2: Norm and embeddings
# =============================================================================


class ModernBertLayerNorm(nn.Module):
    """LayerNorm without bias. The calculation is in FP32, as in HF ``nn.LayerNorm``."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), (x.shape[-1],), self.weight.float(), None, self.eps
        ).to(x.dtype)


class ModernBertEmbeddings(nn.Module):
    """<-- MODEL-SPECIFIC: Token embedding, then LayerNorm (replicated on each rank)."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=config.torch_dtype
        )
        self.norm = ModernBertLayerNorm(
            config.hidden_size, config.norm_eps, config.torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = NF.embedding(input_ids, self.tok_embeddings.weight)
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )
        return self.norm(hidden_states.to(self.dtype))


# =============================================================================
# Section 3: Attention
# =============================================================================


class ModernBertAttention(nn.Module):
    """Bidirectional multi-head attention with per-query KV bounds.

    >>> PARALLELISM: heads are sharded across TP. All-reduce after o_proj. <<<
    <-- MODEL-SPECIFIC: fused Wqkv, no bias, per-layer-type RoPE theta.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] != FULL_ATTENTION
        self.rope_theta = (
            config.local_rope_theta if self.is_sliding else config.global_rope_theta
        )

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not a "
                f"multiple of tensor_parallel_size ({self.world_size}). "
                "Use a tensor_parallel_size that divides the number of heads."
            )
        self.num_heads_per_rank = self.num_attention_heads // self.world_size
        # MHA: the number of KV heads equals the number of query heads.
        self.num_key_value_heads_per_rank = self.num_heads_per_rank
        self.shard_size = self.num_heads_per_rank * self.head_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, 3 * self.shard_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.shard_size, self.hidden_size, dtype=self.dtype)
        )

        # The runner binds a KV cache to each layer. The encoder does not read
        # or write it.
        self.k_cache = None
        self.v_cache = None

        set_weight_loader(
            self.qkv_proj_weight,
            modernbert_qkv_weight_loader(
                self.hidden_size, self.shard_size, self.world_size
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.shard_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Return the attention output ``[T, H]`` (reduced across TP)."""
        tokens = hidden_states.shape[0]
        hidden_states = hidden_states.to(self.dtype)
        n, d = self.num_heads_per_rank, self.head_dim

        # Step 1: Fused QKV projection -> [T, 3 * n * d]
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q = qkv[:, : self.shard_size].reshape(tokens, n, d)
        k = qkv[:, self.shard_size : 2 * self.shard_size].reshape(tokens, n, d)
        v = qkv[:, 2 * self.shard_size :].reshape(tokens, n, d)

        # Step 2: <-- MODEL-SPECIFIC: RoPE with the theta of this layer type
        cos, sin = modernbert_rotary(positions, d, self.rope_theta, self.dtype)
        q = modernbert_apply_rotary(q, cos, sin)
        k = modernbert_apply_rotary(k, cos, sin)

        # Step 3: Bidirectional attention. [T, n, d] -> [n, T, d]
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        attn_bound_min = bound_min.unsqueeze(0).expand(n, tokens, 1).contiguous()
        attn_bound_max = bound_max.unsqueeze(0).expand(n, tokens, 1).contiguous()

        attn_output = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scaling,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=attn_bound_min,
            bound_max=attn_bound_max,
        )

        # Step 4: Output projection. [n, T, d] -> [1, n, d, T] for NF.o_proj
        attn_output = attn_output.view(1, n, tokens, d).permute(0, 1, 3, 2)
        output = NF.o_proj(attn_output, self.o_proj_weight, None).squeeze(0)

        # >>> PARALLELISM: Sum the partial outputs of the head shards <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 4: MLP
# =============================================================================


class ModernBertMLP(nn.Module):
    """GeGLU MLP: ``Wo(gelu(input) * gate)``.

    >>> PARALLELISM: intermediate dim sharded across TP. All-reduce after down. <<<
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not a "
                f"multiple of tensor_parallel_size ({self.world_size}). "
                "Use a tensor_parallel_size that divides intermediate_size."
            )
        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        dtype = config.torch_dtype

        # gate_proj = HF "input" half (activation input). up_proj = HF "gate" half.
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, self.hidden_size, dtype=dtype)
        )

        set_weight_loader(
            self.gate_proj_weight,
            modernbert_geglu_weight_loader(
                config.intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                half=0,
            ),
        )
        set_weight_loader(
            self.up_proj_weight,
            modernbert_geglu_weight_loader(
                config.intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                half=1,
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
        # <-- MODEL-SPECIFIC: exact (erf) GELU gated MLP, no bias.
        # vLLM Neuron 0.29's torch path for NF.mlp applies torch.nn.functional.gelu,
        # a C builtin its graph tracer refuses ("function marked as skipped"), so
        # the GeGLU runs here with the ATen GELU op instead.
        gate = hidden_states @ self.gate_proj_weight
        up = hidden_states @ self.up_proj_weight
        output = (torch.ops.aten.gelu.default(gate) * up) @ self.down_proj_weight
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 5: Encoder layer and backbone
# =============================================================================


class ModernBertLayer(nn.Module):
    """Pre-norm encoder layer. Layer 0 has no attention norm."""

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = ModernBertLayerNorm(
                config.hidden_size, config.norm_eps, config.torch_dtype
            )
        self.attn = ModernBertAttention(config, layer_idx)
        self.mlp_norm = ModernBertLayerNorm(
            config.hidden_size, config.norm_eps, config.torch_dtype
        )
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), positions, bound_min, bound_max
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertEncoderModel(nn.Module):
    """ModernBERT backbone. It returns the post-final-norm hidden states ``[T, H]``."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = ModernBertLayerNorm(
            config.hidden_size, config.norm_eps, config.torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = positions.shape[0]

        # Pad to the kernel row tile. A padding token has position 0, so it
        # gets its own sequence start and does not join a real sequence.
        pad = (-num_tokens) % ATTN_SEQ_ALIGN
        if pad:
            input_ids = F.pad(input_ids, (0, pad))
            positions = F.pad(positions, (0, pad))
            if inputs_embeds is not None:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad))
            if is_token_ids is not None:
                is_token_ids = F.pad(is_token_ids, (0, pad), value=True)

        hidden_states = self.embeddings(input_ids, inputs_embeds, is_token_ids)

        full_min, full_max, sliding_min, sliding_max = modernbert_attention_bounds(
            positions, self.config.sliding_window
        )

        for layer in self.layers:
            if layer.attn.is_sliding:
                bound_min, bound_max = sliding_min, sliding_max
            else:
                bound_min, bound_max = full_min, full_max
            hidden_states = layer(hidden_states, positions, bound_min, bound_max)

        hidden_states = self.final_norm(hidden_states)
        return hidden_states[:num_tokens]

    def checkpoint_mappings(
        self, checkpoint_names: set[str], param_prefix: str = ""
    ) -> dict[str, str]:
        """Map the parameter names to the HF checkpoint keys.

        A checkpoint saved from ``ModernBertModel`` has root-level keys. A
        checkpoint saved from a task head has the ``model.`` prefix. Find the
        prefix from the checkpoint key names.
        """
        probe = "embeddings.tok_embeddings.weight"
        if probe in checkpoint_names:
            ckpt_prefix = ""
        elif f"model.{probe}" in checkpoint_names:
            ckpt_prefix = "model."
        else:
            raise KeyError(
                f"The checkpoint has no '{probe}' key and no 'model.{probe}' key. "
                "Use a ModernBERT checkpoint in the HF transformers format."
            )

        p, c = param_prefix, ckpt_prefix
        mappings = {
            f"{p}embeddings.tok_embeddings.weight": f"{c}embeddings.tok_embeddings.weight",
            f"{p}embeddings.norm.weight": f"{c}embeddings.norm.weight",
            f"{p}final_norm.weight": f"{c}final_norm.weight",
        }
        for i, layer in enumerate(self.layers):
            lp, lc = f"{p}layers.{i}", f"{c}layers.{i}"
            if not isinstance(layer.attn_norm, nn.Identity):
                mappings[f"{lp}.attn_norm.weight"] = f"{lc}.attn_norm.weight"
            mappings[f"{lp}.attn.qkv_proj_weight"] = f"{lc}.attn.Wqkv.weight"
            mappings[f"{lp}.attn.o_proj_weight"] = f"{lc}.attn.Wo.weight"
            mappings[f"{lp}.mlp_norm.weight"] = f"{lc}.mlp_norm.weight"
            mappings[f"{lp}.mlp.gate_proj_weight"] = f"{lc}.mlp.Wi.weight"
            mappings[f"{lp}.mlp.up_proj_weight"] = f"{lc}.mlp.Wi.weight"
            mappings[f"{lp}.mlp.down_proj_weight"] = f"{lc}.mlp.Wo.weight"
        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None,
        rank: int,
        world_size: int,
        pipelined: bool = True,
    ) -> None:
        """Load the TP shard of ``rank`` from the HF checkpoint.

        ``pipelined=True`` needs an initialized process group.
        ``pipelined=False`` loads each tensor in sequence.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        mappings = self.checkpoint_mappings(checkpoint.get_tensor_names())
        if pipelined:
            load_result = checkpoint.load_sharded_pipelined(
                rank, world_size, self, mappings, device
            )
        else:
            load_result = checkpoint.load_sharded(
                rank, world_size, self, mappings, device
            )
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "The ModernBERT checkpoint does not agree with the model. "
                f"Missing parameters: {load_result.missing_keys}. "
                f"Unused checkpoint keys: {load_result.unexpected_keys}."
            )
        rank_sharded = load_result.state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=True, assign=True)


# =============================================================================
# Section 6: Pooling model
# =============================================================================


class ModernBertForEmbedding(nn.Module):
    """ModernBERT backbone with the upstream vLLM pooler (CLS, then L2 normalize).

    >>> PARALLELISM: TP backbone with a replicated hidden state. <<<
    <-- MODEL-SPECIFIC: encoder-only. No LM head and no decode phase.
    """

    is_pooling_model = True
    default_seq_pooling_type = "CLS"
    attn_type = "encoder_only"

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.model = ModernBertEncoderModel(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.pooler import DispatchPooler

        vllm_config = get_current_vllm_config()
        self._validate_vllm_config(vllm_config)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, (
            "pooler_config is None. ModernBertForEmbedding needs the pooling "
            "runner. Launch with --runner pooling."
        )
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    @staticmethod
    def _validate_vllm_config(vllm_config) -> None:
        model_config = vllm_config.model_config
        if model_config is not None and model_config.runner_type != "pooling":
            raise ValueError(
                f"runner_type={model_config.runner_type!r} is not supported for "
                "ModernBERT. ModernBERT is an encoder. Launch with --runner pooling."
            )
        cache_config = vllm_config.cache_config
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if (cache_dtype or "").startswith("fp8"):
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported for ModernBERT. "
                "The encoder does not use the KV cache. Use the default (auto)."
            )
        if getattr(cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "Prefix caching is not supported for ModernBERT, because the "
                "attention is bidirectional. Launch with --no-enable-prefix-caching."
            )
        scheduler_config = vllm_config.scheduler_config
        if getattr(scheduler_config, "enable_chunked_prefill", False) and (
            model_config is not None
            and scheduler_config.max_num_batched_tokens < model_config.max_model_len
        ):
            raise ValueError(
                "Chunked prefill is not supported for ModernBERT, because the "
                "attention is bidirectional. Set --max-num-batched-tokens to "
                "--max-model-len or larger."
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return the flattened ``[T, H]`` post-norm hidden states."""
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = ModernBertConfig.from_configs(hf_config, neuron_config)
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config is not None and isinstance(model_config.dtype, torch.dtype):
            config.torch_dtype = model_config.dtype
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────
    # The encoder never reads the KV cache. These hooks exist because the
    # NeuronModelRunner calls get_kv_spec and bind_kv_cache on each model.

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.attn",
                    num_kv_heads=layer.attn.num_key_value_heads_per_rank,
                    head_size=layer.attn.head_dim,
                    dtype=layer.attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} is not initialized")
            layer.attn.k_cache = kv_caches[layer_name][0]
            layer.attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the ModernBERT weights from the HF checkpoint.

        The checkpoint has no pooling weights. The pooler is weightless.
        """
        self.model.load_weights(
            checkpoint_path, device, cache_dir, self.rank, self.world_size
        )
