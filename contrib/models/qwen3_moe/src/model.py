# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-MoE (Qwen3-30B-A3B) for the native vLLM Neuron route.

The attention, the norms, the RoPE, the embedding and the LM head are the
Qwen3 dense modules of the vLLM Neuron plugin (``vllm_neuron.model.qwen3``).
This file adds the sparse MoE block and the layer stack that uses it.

>>> PARALLELISM <<<
- Attention: TP head sharding, as in Qwen3 dense.
- MoE: expert parallelism inside the TP group (``--enable-expert-parallel``).
  Each rank holds ``num_experts / ep_degree`` complete experts. Without EP,
  each rank holds all experts with a TP shard of the expert intermediate dim.
- EP is the declared topology. Without EP, a small expert shard (< 128)
  makes the prefill experts use torch. The model logs a warning in that case.
- Prefill: NF.router on the SP shard -> all-gather -> NF.moe_cte
  -> reduce-scatter. In vLLM Neuron 0.29.0.1.2.0, NF.router always runs its
  PyTorch path (FP32 matmul, softmax, top-k and renorm).
- Decode: NF.moe_block_tkg (RMSNorm + router + experts in one kernel)
  -> all-reduce.

<-- MODEL-SPECIFIC:
- Router: softmax over all experts in FP32 -> top-8 -> renormalize
  (``norm_topk_prob=True``). No router bias.
- Experts: ``down(silu(gate(x)) * up(x))``. No bias, no clamp.
- Each decoder layer has a sparse MoE block (``decoder_sparse_step=1``,
  ``mlp_only_layers=[]``). The factory rejects other layouts.
"""

import functools
import logging

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.model import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3MoeConfig

logger = logging.getLogger(__name__)

# Minimum expert intermediate shard of NF.moe_cte (_can_use_moe_cte_kernel).
_MOE_CTE_MIN_INTERMEDIATE = 128


@functools.lru_cache(maxsize=None)
def _warn_small_expert_shard(intermediate_size_per_rank: int) -> None:
    # Log one time, not one time for each layer.
    logger.warning(
        "The expert intermediate shard (%d) is smaller than the minimum of the "
        "prefill MoE kernel (%d), so prefill experts use the torch path. "
        "Enable --enable-expert-parallel.",
        intermediate_size_per_rank,
        _MOE_CTE_MIN_INTERMEDIATE,
    )


# =============================================================================
# Section 4: Sparse MoE block
# Mixed: PARALLELISM (EP/TP expert placement) + MODEL-SPECIFIC (routing, SiLU)
# (Sections 1-3, RMSNorm, RoPE and attention, are the Qwen3 dense modules
# imported above.)
# =============================================================================


def _shape(s) -> tuple:
    # A safetensors slice has get_shape(). A torch tensor has shape.
    return tuple(s.get_shape()) if hasattr(s, "get_shape") else tuple(s.shape)


def expert_gate_up_loader(tp_degree: int) -> SafetensorsWeightLoader:
    """Return a loader that fuses per-expert gate and up weights.

    The loader receives ``[gate_0..gate_{E-1}, up_0..up_{E-1}]`` for the local
    experts. Each HF slice has the shape ``[I, H]``. The result has the shape
    ``[E, H, 2 * I / tp_degree]``, with the gate columns first.

    Args:
        tp_degree: The number of shards of the expert intermediate dim.

    Example:
        >>> loader = expert_gate_up_loader(tp_degree=1)
        >>> set_weight_loader(param, loader)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        n = len(slices) // 2
        shard = rank % tp_degree
        fused = []
        for i in range(n):
            size = _shape(slices[i])[0] // tp_degree
            rows = slice(shard * size, (shard + 1) * size)
            gate = slices[i][rows, :]
            up = slices[n + i][rows, :]
            fused.append(torch.cat([gate.T, up.T], dim=-1))
        return torch.stack(fused, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def expert_down_loader(tp_degree: int) -> SafetensorsWeightLoader:
    """Return a loader that stacks per-expert down weights.

    Each HF slice has the shape ``[H, I]``. The result has the shape
    ``[E, I / tp_degree, H]``.

    Args:
        tp_degree: The number of shards of the expert intermediate dim.

    Example:
        >>> loader = expert_down_loader(tp_degree=1)
        >>> set_weight_loader(param, loader)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        shard = rank % tp_degree
        stacked = []
        for s in slices:
            size = _shape(s)[1] // tp_degree
            cols = slice(shard * size, (shard + 1) * size)
            stacked.append(s[:, cols].T)
        return torch.stack(stacked, dim=0)

    return SafetensorsWeightLoader(transform=transform)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with top-k softmax routing and SiLU experts.

    >>> PARALLELISM: EP inside the TP group <<<
    - With EP, ``ep_degree`` must equal the TP world size. Each rank holds
      ``num_experts / ep_degree`` complete experts (linear placement).
    - Without EP, each rank holds all experts with a TP shard of the
      expert intermediate dim.
    - ``moe_group`` is the full TP group for each outer collective.

    <-- MODEL-SPECIFIC: router weight [E, H] replicated on all ranks.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        from vllm.config import get_current_vllm_config

        self.tp_group = get_tp_group()
        self.moe_group = self.tp_group

        parallel_config = get_current_vllm_config().parallel_config
        self.ep_enabled = bool(parallel_config.enable_expert_parallel)
        if getattr(parallel_config, "data_parallel_size", 1) > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3-MoE. "
                "Set --data-parallel-size to 1."
            )

        if self.ep_enabled:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            if self.ep_degree != self.tp_group.world_size:
                raise ValueError(
                    f"ep_degree ({self.ep_degree}) is not equal to the "
                    f"tensor-parallel size ({self.tp_group.world_size}). "
                    "Qwen3-MoE supports EP over the full TP group only. "
                    "Remove ep_degree from neuron_config."
                )
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.tp_degree = self.tp_group.world_size

        self.total_num_experts = config.num_experts
        if self.total_num_experts % self.ep_degree != 0:
            raise ValueError(
                f"num_experts ({self.total_num_experts}) is not divisible by "
                f"ep_degree ({self.ep_degree}). Use a different "
                "tensor-parallel size."
            )
        if config.moe_intermediate_size % self.tp_degree != 0:
            raise ValueError(
                f"moe_intermediate_size ({config.moe_intermediate_size}) is not "
                f"divisible by the expert TP degree ({self.tp_degree}). Enable "
                "--enable-expert-parallel, or use a different tensor-parallel size."
            )

        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree
        # EP is the declared topology. Without EP, the expert shard can be
        # smaller than the minimum of NF.moe_cte, and prefill experts then use torch.
        if self.intermediate_size_per_rank < _MOE_CTE_MIN_INTERMEDIATE:
            _warn_small_expert_shard(self.intermediate_size_per_rank)
        self.rms_norm_eps = config.rms_norm_eps
        self.block_size = 256

        # >>> PARALLELISM: Router replicated on all ranks (NOT EP-sharded) <<<
        # The HF router is a BF16 nn.Linear. Keep the checkpoint dtype, and
        # compute the softmax in FP32.
        self.router_weight = nn.Parameter(
            torch.empty(
                self.total_num_experts, self.hidden_size, dtype=config.torch_dtype
            )
        )
        # [E_L, H, 2 * I_r]: gate columns first, then up columns.
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                2 * self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        # [E_L, I_r, H]
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: EP filters the experts, TP shards the intermediate dim. <<<"""
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            if self.ep_degree > 1:
                return expert_parallel_grouped_loader(
                    local_expert_indices, loader, self.total_num_experts
                )
            return loader

        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(expert_gate_up_loader(self.tp_degree)),
        )
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(expert_down_loader(self.tp_degree)),
        )

    def _gate_up_4d(self) -> torch.Tensor:
        return self.gate_up_proj_weight.reshape(
            self.num_local_experts,
            self.hidden_size,
            2,
            self.intermediate_size_per_rank,
        )

    def _ep_rank_tensor(self, device: torch.device) -> torch.Tensor:
        """Return the EP rank as a ``[1, 1]`` int32 runtime tensor.

        The rank comes from the stamped ``rank_in_group_tensor``, so all ranks
        trace one graph. A Python-int rank makes one NEFF for each rank.
        """
        if self.ep_degree <= 1:
            return torch.zeros((1, 1), dtype=torch.int32, device=device)
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

        return get_neuron_ep_group().rank_in_group_tensor.reshape(1, 1).to(torch.int32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None,
        is_decode: bool,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the MoE block.

        Prefill gets the normalized SP shard ``[T / TP, H]`` and returns the SP
        shard. Decode gets the residual stream ``[B, H]`` and the RMSNorm
        weight ``gamma``. The decode kernel applies the RMSNorm.
        """
        if is_decode:
            return self.forward_decode(hidden_states, gamma)
        return self.forward_prefill(hidden_states, positions)

    def forward_decode(
        self, hidden_states: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """Decode: fused RMSNorm + router + experts kernel, then all-reduce."""
        output = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=gamma.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self._gate_up_4d(),
            expert_down_weights=self.down_proj_weight,
            rank_id=self._ep_rank_tensor(hidden_states.device),
            top_k=self.num_experts_per_token,
            eps=self.rms_norm_eps,
            # <-- MODEL-SPECIFIC: softmax over all experts, then top-k, then renorm
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=self.norm_topk_prob,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
            is_all_expert=True,
            skip_router_logits=True,
        )
        output = output.reshape(hidden_states.shape)

        # >>> PARALLELISM: Sum the local-expert partials over all ranks <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)
        return output

    def forward_prefill(
        self, hidden_states: torch.Tensor, positions: torch.Tensor | None
    ) -> torch.Tensor:
        """Prefill: router -> all-gather -> blockwise MoE kernel -> reduce-scatter."""
        # Router on the SP shard: [T / TP, H] -> [T / TP, E]. The router of
        # vLLM Neuron 0.29.0.1.2.0 runs its PyTorch path: it casts the hidden
        # states and the router weight to FP32, then computes the softmax,
        # the top-k and the renorm in FP32, as the HF FP32 router does.
        expert_affinities = NF.router(
            hidden_states=hidden_states,
            router_weights=self.router_weight.T,
            top_k=self.num_experts_per_token,
            # <-- MODEL-SPECIFIC: softmax over all experts, then top-k, then
            # renorm (norm_topk_prob=True; the factory refuses False, which has
            # no matching NF.router computation order).
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=(
                RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
            ),
        )

        # >>> PARALLELISM: All-gather the full sequence <<<
        if self.tp_group.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # True for a real token, False for a padding token.
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        # >>> PARALLELISM: With EP, keep the affinities of the local experts <<<
        if self.ep_degree > 1:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_group,
            )

            ep_rank = get_neuron_ep_group().rank_in_group_tensor.to(torch.int32)
            local_expert_indices = (
                torch.arange(
                    self.num_local_experts,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                + ep_rank * self.num_local_experts
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self._gate_up_4d(),
            down_proj_weight=self.down_proj_weight,
            # <-- MODEL-SPECIFIC: SiLU gated experts, no bias, no clamp
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

        # >>> PARALLELISM: Sum the partials and return to the SP layout <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)
        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: Qwen3 pre-norm layout with a sparse MoE block
# =============================================================================


class Qwen3MoeDecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer with a sparse MoE block in place of the dense MLP.

    Architecture: hidden -> RMSNorm -> Attention -> residual -> RMSNorm -> MoE
    -> residual. In decode, the MoE kernel applies the second RMSNorm.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if is_decode:
            hidden_states = self.mlp(
                hidden_states,
                positions,
                is_decode=True,
                gamma=self.post_attention_layernorm.weight,
            )
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, positions, is_decode=False)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# >>> PARALLELISM: same SP layout as Qwen3 dense <<<
# =============================================================================


class Qwen3MoeModel(Qwen3Model):
    """Qwen3-MoE backbone. The forward is the Qwen3 dense forward."""

    def __init__(self, config: Qwen3MoeConfig):
        nn.Module.__init__(self)
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

        # <-- MODEL-SPECIFIC: 48 MoE decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

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

        self.aux_hidden_state_layers = []
        # The MoE layer does not support sequence-parallel decode.
        self.decode_sp_enabled = False


# =============================================================================
# Section 7: LM Head + Weight Loading
# =============================================================================


def _expert_keys(prefix: str, name: str, num_experts: int) -> list[str]:
    return [f"{prefix}.mlp.experts.{e}.{name}.weight" for e in range(num_experts)]


def qwen3_moe_checkpoint_mappings(
    num_layers: int, num_experts: int
) -> dict[str, str | list[str]]:
    """Return the map from each model parameter to its HF checkpoint key(s).

    The map depends only on the layer count and the expert count, so a test
    can check it against the checkpoint index without a model instance.
    """
    mappings = {
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
        for norm in ("q_norm", "k_norm"):
            mappings[f"{prefix}.self_attn.{norm}.weight"] = (
                f"{prefix}.self_attn.{norm}.weight"
            )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"

        # <-- MODEL-SPECIFIC: router and per-expert weights
        mappings[f"{prefix}.mlp.router_weight"] = f"{prefix}.mlp.gate.weight"
        mappings[f"{prefix}.mlp.gate_up_proj_weight"] = _expert_keys(
            prefix, "gate_proj", num_experts
        ) + _expert_keys(prefix, "up_proj", num_experts)
        mappings[f"{prefix}.mlp.down_proj_weight"] = _expert_keys(
            prefix, "down_proj", num_experts
        )
    return mappings


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    """Qwen3-MoE model with LM head.

    The forward, the KV cache binding and the sampler are the Qwen3 dense
    implementation. The base class is already wrapped by
    ``@async_speculative_decoding``, so this class does not wrap it again.
    """

    def __init__(self, config: Qwen3MoeConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3MoeModel(config)
        self._init_head(config)

    def _init_head(self, config: Qwen3MoeConfig) -> None:
        import vllm_neuron.nn as neuron_nn
        from vllm_neuron.nn.sampler import Sampler

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

        # <-- MODEL-SPECIFIC: untied lm_head
        # >>> PARALLELISM: Column-parallel LM head <<<
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

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3MoeConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from each model parameter to its HF checkpoint key(s)."""
        return qwen3_moe_checkpoint_mappings(
            len(self.model.layers), self.config.num_experts
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from a Qwen3-MoE HF checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding and EP filtering. <<<
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.checkpoint_mappings(), device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(rank_sharded, strict=False, assign=True)
