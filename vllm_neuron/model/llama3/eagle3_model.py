# SPDX-License-Identifier: Apache-2.0
import logging
import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from libtorch_neuronx_lite.compile.platform import get_platform_target
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron import envs
import vllm_neuron.nn as neuron_nn
from vllm_neuron import functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.model import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader_with_padding,
    with_rank_override,
)

logger = logging.getLogger(__name__)


def extract_accepted_tokens(
    input_ids: torch.Tensor,
    sampling_positions: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    vocab_size: int,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device extract_next_token_ids + compute_token_indices_to_sample.

    1. Finds the last valid token per row in raw_sampled_token_ids (valid = not -1
       and < vocab_size). Scatters it into input_ids at sampling_positions.
    2. Adjusts sampling_positions by subtracting rejected token count per request.

    Args:
        input_ids: Shifted target token IDs [T].
        sampling_positions: Indices into input_ids for each request [bs].
        raw_sampled_token_ids: [bs, max_spec_len+1] with -1 for rejected,
            or [bs, 1] for prefill.
        vocab_size: Model vocabulary size.
        num_speculative_tokens: Number of speculative tokens per step.

    Returns:
        (patched_input_ids, adjusted_sampling_positions, next_token_ids)
        where next_token_ids is the bonus (last valid) token per request
        [bs] — exposed so the draft NEFF output can include it as the
        first column, allowing the next target NEFF to consume a single
        device future without any intermediate Python-side ops.
    """
    valid_mask = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1)  # [bs]
    last_valid_idx = torch.clamp(valid_count - 1, min=0)  # [bs]
    next_token_ids = (
        raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
        .squeeze(1)
        .to(torch.int32)
    )  # [bs]
    # Guard: if all entries in a row are invalid, use 0 instead of garbage
    next_token_ids = torch.where(
        valid_count > 0, next_token_ids, torch.zeros_like(next_token_ids)
    )

    # Adjust sampling_positions for rejected tokens BEFORE scatter.
    # Prefill: raw_sampled has 1 column → no adjustment needed.
    # Steady-state: raw_sampled has num_spec+1 columns → adjust for rejections.
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = torch.clamp(num_speculative_tokens + 1 - valid_count, min=0)
        sampling_positions = torch.clamp(
            sampling_positions - num_rejected.to(sampling_positions.dtype), min=0
        )

    # Patch input_ids at adjusted sampling_positions with next_token_ids
    input_ids = input_ids.scatter(0, sampling_positions.to(torch.long), next_token_ids)

    return input_ids, sampling_positions, next_token_ids


def compute_slot_mapping(
    positions: torch.Tensor,
    block_table_tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute slot_mapping on-device from positions and block table.

    Maps each position to its KV cache slot index:
        slot = block_id * block_size + (position % block_size)

    where block_id is looked up from block_table_tensor using the block
    number derived from the position.

    Args:
        positions: Token positions [bs].
        block_table_tensor: Block table [bs, max_blocks_per_seq].
        block_size: Number of slots per block.

    Returns:
        Slot indices [bs].
    """
    block_numbers = positions // block_size
    block_ids = block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(
        -1
    )
    slot = block_ids * block_size + (positions % block_size)
    # If the gathered block id is the inactive-block sentinel (-1), propagate
    # PAD_SLOT_ID (-1) to slot mapping
    return torch.where(block_ids < 0, torch.full_like(slot, -1), slot)


def _slice_dp_to_local(
    x: torch.Tensor, dp_group, rank: torch.Tensor | None = None
) -> torch.Tensor:
    """Slice a DP-gathered tensor along dim 0 down to this rank's local rows.

    The draft runs on the local batch, but some inputs arrive gathered across a
    DP column group. Keep this rank's own contiguous block of rows. A DP size of
    1 returns the input unchanged.

    Args:
        x: Tensor gathered across ``dp_group`` along dim 0. The row count is a
            multiple of ``dp_group.world_size``.
        dp_group: The DP column group that ``x`` is gathered over.
        rank: Optional rank tensor within ``dp_group``. Pass this value to
            derive the slice from the model rank input.

    Returns:
        The local slice of ``x`` along dim 0.

    Example:
        >>> local = _slice_dp_to_local(gathered, embedding_dp_group)
    """
    if dp_group.world_size <= 1:
        return x
    # shard_select is the SPMD-safe slice. It keeps the rank tensor 1-D so
    # torch.compile does not bake a per-rank constant into the graph.
    from vllm_neuron.parallel.neuron_parallel_state import shard_select

    return shard_select(x, 0, dp_group, rank=rank)


def _select_spmd_sequence_shard(
    x: torch.Tensor, rank: torch.Tensor, n_shards: int
) -> torch.Tensor:
    """Select one contiguous row shard without an indirect index operation."""
    total_rows = x.shape[0]
    if total_rows % n_shards != 0:
        raise ValueError(
            f"Sequence row count {total_rows} is not divisible by {n_shards} shards."
        )
    rows_per_shard = total_rows // n_shards
    shards = x.reshape(n_shards, rows_per_shard, *x.shape[1:])
    shard_ids = torch.arange(n_shards, device=x.device, dtype=torch.int64)
    rank_mask = (shard_ids == rank.reshape(1).to(torch.int64)).reshape(
        n_shards, *((1,) * x.ndim)
    )
    return (shards * rank_mask.to(x.dtype)).sum(dim=0)


def to_target_hidden_layout(
    weight: torch.Tensor, unpadded_hidden: int, padded_hidden: int, pad_groups: int
) -> torch.Tensor:
    """Move a natural-order hidden axis into the target's padded layout.

    The target pads its hidden dimension for hardware alignment. Two layouts
    exist, and ``pad_groups`` selects between them:

    * ``pad_groups == 1`` — the real values stay contiguous and the padding sits
      at the tail. Llama and BF16 GPT-OSS use this layout.
    * ``pad_groups > 1`` — the target pads the tail and then deinterleaves the
      axis by ``pad_groups``, so natural index ``n`` moves to slot
      ``(n % pad_groups) * (padded_hidden // pad_groups) + n // pad_groups``.
      MXFP4 GPT-OSS uses this layout with 4 groups (see ``_shuffle_hidden_dim``
      in ``model/gpt_oss/weight_loaders_mxfp4.py``). The padding then lands in
      the last ``(padded_hidden - unpadded_hidden) // pad_groups`` slots of each
      group, which is where ``GptOssRMSNorm`` zeroes it.

    The draft reads target hidden states only through ``fc`` and its optional
    ``input_norm``, so only those two weights need this transform. Every other
    draft weight stays in the draft's own contiguous layout.

    Args:
        weight: Weight whose LAST dimension is a natural-order hidden axis of
            size ``unpadded_hidden``.
        unpadded_hidden: Real hidden size, for example 2880.
        padded_hidden: Padded hidden size, for example 3072.
        pad_groups: Number of deinterleave groups. Use 1 for a tail pad.

    Returns:
        The weight with its last dimension in the target's layout, of size
        ``padded_hidden``.

    Example:
        >>> w = torch.arange(4.0).reshape(1, 4)
        >>> to_target_hidden_layout(w, 4, 8, 2).tolist()
        [[0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0]]
    """
    if pad_groups < 1:
        raise ValueError(f"pad_groups must be positive, got {pad_groups}")
    if padded_hidden < unpadded_hidden:
        raise ValueError(
            f"padded_hidden ({padded_hidden}) must be >= "
            f"unpadded_hidden ({unpadded_hidden})"
        )
    if weight.shape[-1] != unpadded_hidden:
        raise ValueError(f"Expected last dim {unpadded_hidden}, got {weight.shape[-1]}")
    if padded_hidden % pad_groups or unpadded_hidden % pad_groups:
        raise ValueError(
            f"unpadded_hidden ({unpadded_hidden}) and padded_hidden "
            f"({padded_hidden}) must both be divisible by pad_groups "
            f"({pad_groups})"
        )

    if padded_hidden == unpadded_hidden:
        return weight
    weight = torch.nn.functional.pad(weight, (0, padded_hidden - unpadded_hidden))
    if pad_groups == 1:
        return weight
    # Deinterleave the padded axis: [.., H] -> [.., H / g, g] -> [.., g, H / g].
    lead = weight.shape[:-1]
    return (
        weight.reshape(*lead, padded_hidden // pad_groups, pad_groups)
        .transpose(-2, -1)
        .reshape(*lead, padded_hidden)
    )


def _make_rmsnorm(config: LlamaConfig) -> LlamaRMSNorm:
    """Create RMSNorm, padding-aware if hidden_size != unpadded_hidden_size.

    When the draft model uses padded hidden dimensions (e.g. GPT-OSS 2880→3072),
    variance must be computed only on the unpadded portion to match GPU behavior.
    """
    if config.hidden_size != config.unpadded_hidden_size:
        return _PaddedRMSNorm(config)
    return LlamaRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)


class _PaddedRMSNorm(nn.Module):
    """Padding-aware RMSNorm over one or more concatenated hidden states.

    Each of the ``n_chunks`` hidden-state chunks is interleaved-padded from
    ``unpadded_hidden_size`` to ``hidden_size`` (e.g. GPT-OSS 2880→3072). A
    single joint variance is computed over the real (unpadded) elements of every
    chunk, so padded positions never contribute; those positions are then zeroed
    in the output. The default ``n_chunks=1`` is the regular padded RMSNorm used
    by decoder layers. ``n_chunks=3`` matches upstream vLLM's dense
    ``RMSNorm(n_chunks * unpadded_hidden)`` applied to the concatenated aux
    hidden states before ``fc`` (``norm_before_fc``, e.g. RedHatAI gpt-oss
    eagle3 heads).
    """

    def __init__(self, config: LlamaConfig, n_chunks: int = 1, pad_groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(n_chunks * config.hidden_size, dtype=config.torch_dtype)
        )
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.padded_hidden_size = config.hidden_size
        self.n_chunks = n_chunks
        # Padding position within a chunk. With pad_groups == 1 the padding is
        # one tail block; with pad_groups > 1 it is the tail of each group (the
        # MXFP4 GPT-OSS layout). See to_target_hidden_layout.
        self.pad_groups = pad_groups
        self.group_size = config.hidden_size // pad_groups
        self.real_per_group = config.unpadded_hidden_size // pad_groups
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        leading_shape = hidden_states.shape[:-1]
        # Group the padding positions of every chunk into one trailing block, so
        # both layouts use the same slice. The order inside a group does not
        # matter here: the variance sums the real elements and the write zeroes
        # the padded ones.
        grouped_shape = (
            *leading_shape,
            self.n_chunks * self.pad_groups,
            self.group_size,
        )
        chunks = hidden_states.reshape(*grouped_shape)
        # Joint variance over the real elements of every chunk (padding excluded)
        # so the numerics match a dense RMSNorm over the unpadded concat.
        real_values = chunks[..., : self.real_per_group]
        variance = real_values.reshape(*leading_shape, -1).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = (self.weight * hidden_states).reshape(*grouped_shape)
        output[..., self.real_per_group :] = 0.0
        return output.reshape(*leading_shape, -1).to(input_dtype)


def fc_weight_layout_for_target(target: str) -> str:
    """Return the fc weight layout for a platform target.

    trn3 stores the fc weight ``[in, out]``. The draft profile and the accuracy
    runs validated this layout on trn3. Other targets keep the ``nn.Linear``
    ``[out, in]`` layout. The match is a prefix match, because
    ``get_platform_target()`` can return a target with a revision suffix.

    Args:
        target: The platform target, for example ``"trn3"`` or ``"trn2"``.

    Returns:
        ``"in_out"`` on trn3 targets, else ``"out_in"``.

    Example:
        >>> fc_weight_layout_for_target("trn3")
        'in_out'
        >>> fc_weight_layout_for_target("trn2")
        'out_in'
    """
    return "in_out" if target.startswith("trn3") else "out_in"


def fc_interleaved_padding_weight_loader(
    unpadded_hidden: int,
    padded_hidden: int,
    pad_groups: int = 1,
    num_shards: int = 1,
    weight_layout: str = "out_in",
) -> SafetensorsWeightLoader:
    """Pad, shard and lay out the fc weight that combines 3 aux hidden states.

    The fc layer in Eagle3 combines 3 auxiliary hidden states from the target model.
    When the target model uses padded hidden dimensions, the fc weight needs to be
    padded to match.

    Checkpoint weight: [unpadded_hidden, unpadded_hidden * 3]
    Padded weight:     [padded_hidden, padded_hidden * 3]

    The input dimension has 3 concatenated aux hidden states. Each one is moved
    into the target's hidden layout, which ``pad_groups`` selects. A wrong
    ``pad_groups`` puts every input feature in the wrong column, so the draft
    reads scrambled hidden states and proposes useless tokens.

    ``num_shards`` shards the output dim by ``rank % num_shards``.
    ``weight_layout="in_out"`` returns the transposed shard, the parameter
    layout of ``ColumnParallelLinear(weight_layout="in_out")``.

    Args:
        unpadded_hidden: Original hidden size (e.g., 2880)
        padded_hidden: Target hidden size after padding (e.g., 3072)
        pad_groups: Deinterleave groups of the target's hidden layout. Use 1 for
            a tail pad (Llama, BF16 GPT-OSS) and 4 for MXFP4 GPT-OSS. See
            :func:`to_target_hidden_layout`.
        num_shards: Ways to shard the output dim (the fc TP group size).
        weight_layout: ``"out_in"`` (nn.Linear order) or ``"in_out"`` (transposed).

    Returns:
        SafetensorsWeightLoader that pads, shards and lays out the fc weight.
    """
    if weight_layout not in ("out_in", "in_out"):
        raise ValueError(
            f"weight_layout must be 'out_in' or 'in_out', got {weight_layout!r}"
        )
    if num_shards < 1 or padded_hidden % num_shards:
        raise ValueError(
            f"padded_hidden ({padded_hidden}) must be a positive multiple of "
            f"num_shards ({num_shards})"
        )

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, f"Expected 1 slice for fc weight, got {len(slices)}"
        assert padded_hidden >= unpadded_hidden, (
            f"padded_hidden ({padded_hidden}) must be >= unpadded_hidden ({unpadded_hidden})"
        )

        weight = slices[0][:]  # [out=unpadded_hidden, in=unpadded_hidden * 3]

        if padded_hidden != unpadded_hidden:
            # Split the input dim into 3 aux parts and lay each out like the target.
            aux_weights = torch.split(weight, unpadded_hidden, dim=1)
            padded_aux = [
                to_target_hidden_layout(w, unpadded_hidden, padded_hidden, pad_groups)
                for w in aux_weights
            ]
            weight = torch.cat(
                padded_aux, dim=1
            )  # [unpadded_hidden, padded_hidden * 3]

            # Pad output dim
            out_pad = padded_hidden - weight.shape[0]
            if out_pad > 0:
                weight = torch.nn.functional.pad(weight, (0, 0, 0, out_pad))
        # weight: [padded_hidden, padded_hidden * 3]

        if num_shards > 1:
            shard = padded_hidden // num_shards
            start = (rank % num_shards) * shard
            weight = weight[start : start + shard, :]
        if weight_layout == "in_out":
            weight = weight.T.contiguous()
        return weight

    return SafetensorsWeightLoader(transform=transform)


def concat_norm_interleaved_padding_weight_loader(
    unpadded_hidden: int, padded_hidden: int, n_chunks: int = 3, pad_groups: int = 1
) -> SafetensorsWeightLoader:
    """Pad a 1-D norm weight over an ``n_chunks``-way concat with interleaved zeros.

    Companion to :func:`fc_interleaved_padding_weight_loader` for the
    ``input_norm`` gamma (``norm_before_fc``). Checkpoint weight is
    ``[n_chunks * unpadded_hidden]``; each chunk is moved into the target's
    hidden layout, which ``pad_groups`` selects, so the gamma columns line up
    with the fc input columns.
    """

    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    if padded_hidden < unpadded_hidden:
        raise ValueError(
            f"padded_hidden ({padded_hidden}) must be >= "
            f"unpadded_hidden ({unpadded_hidden})"
        )

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, f"Expected 1 slice for input_norm, got {len(slices)}"
        weight = slices[0][:]  # [n_chunks * unpadded_hidden]
        expected_shape = (n_chunks * unpadded_hidden,)
        if weight.ndim != 1 or tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"Expected input_norm weight shape {expected_shape}, "
                f"got {tuple(weight.shape)}"
            )
        if padded_hidden == unpadded_hidden:
            return weight
        chunks = torch.split(weight, unpadded_hidden)
        padded = [
            to_target_hidden_layout(c, unpadded_hidden, padded_hidden, pad_groups)
            for c in chunks
        ]
        return torch.cat(padded)  # [n_chunks * padded_hidden]

    return SafetensorsWeightLoader(transform=transform)


def fused_rmsnorm_padding_weight_loader(
    unpadded_hidden: int, padded_hidden: int
) -> SafetensorsWeightLoader:
    """Load gamma corrected for RMSNorm over zero-padded hidden states.

    The fused MLP kernel divides the sum of squares by ``padded_hidden``.
    Scaling gamma by ``sqrt(unpadded_hidden / padded_hidden)`` and epsilon by
    the corresponding ratio makes that calculation equivalent to RMSNorm over
    only the real hidden elements. The epsilon correction is applied by the
    decoder layer.
    """
    if unpadded_hidden <= 0 or padded_hidden < unpadded_hidden:
        raise ValueError(
            f"Expected 0 < unpadded_hidden <= padded_hidden, got "
            f"{unpadded_hidden} and {padded_hidden}"
        )

    weight_scale = math.sqrt(unpadded_hidden / padded_hidden)

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            f"Expected 1 slice for fused RMSNorm, got {len(slices)}"
        )
        weight = slices[0][:]
        if weight.ndim != 1 or weight.shape[0] != unpadded_hidden:
            raise ValueError(
                f"Expected fused RMSNorm weight shape ({unpadded_hidden},), "
                f"got {tuple(weight.shape)}"
            )
        weight = weight * weight_scale
        return torch.nn.functional.pad(weight, (0, padded_hidden - unpadded_hidden))

    return SafetensorsWeightLoader(transform=transform)


def embedding_sharding_padding_weight_loader(
    vocab_size_per_rank: int,
    num_shards: int,
    padded_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Shard embedding along vocab dim (0) and pad along hidden dim (1).

    This loader handles both:
    1. Vocabulary dimension sharding for tensor parallelism
    2. Hidden dimension padding for alignment requirements

    Args:
        vocab_size_per_rank: Size of each vocabulary shard per rank.
        num_shards: Total number of shards (TP size).
        padded_hidden_size: Target hidden dimension after padding.

    Returns:
        SafetensorsWeightLoader that shards and pads the embedding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            f"Expected 1 slice for embedding weight, got {len(slices)}"
        )
        # Get the full embedding weight
        full_weight = slices[0][:]  # [vocab_size, hidden_size]

        # Shard along vocab dimension
        start_idx = rank * vocab_size_per_rank
        end_idx = start_idx + vocab_size_per_rank
        # Handle case where last shard might be smaller
        actual_end = min(end_idx, full_weight.shape[0])
        sharded_weight = full_weight[start_idx:actual_end]

        # Pad to full shard size if needed (for last rank)
        if sharded_weight.shape[0] < vocab_size_per_rank:
            pad_rows = vocab_size_per_rank - sharded_weight.shape[0]
            sharded_weight = torch.nn.functional.pad(
                sharded_weight, (0, 0, 0, pad_rows)
            )

        # Pad hidden dimension
        pad_amount = padded_hidden_size - sharded_weight.shape[-1]
        if pad_amount > 0:
            sharded_weight = torch.nn.functional.pad(sharded_weight, (0, pad_amount))

        return sharded_weight  # [shard_size, padded_hidden_size]

    return SafetensorsWeightLoader(transform=transform)


def _eagle3_qkv_weight_loader(
    q_size,
    kv_size,
    shard_dim,
    num_shards,
    is_storage_transposed,
    num_kv_replicas,
    padded_hidden_size,
    unpadded_hidden_size,
):
    """QKV weight loader with interleaved padding for eagle3's concat input.

    The eagle3 QKV input is concat(embeds, hidden_states), each independently
    padded. The weight's input dimension must match this layout:
        [embeds_unpadded, pad, hidden_unpadded, pad]
    NOT:
        [embeds_unpadded, hidden_unpadded, pad_at_end]
    """
    base_loader = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=shard_dim,
        num_shards=num_shards,
        is_storage_transposed=is_storage_transposed,
        num_kv_replicas=num_kv_replicas,
        # No padding in base loader — we handle it here
        padded_hidden_size=None,
        unpadded_hidden_size=None,
    )

    pad = padded_hidden_size - unpadded_hidden_size
    if pad <= 0:
        return base_loader

    def transform(slices, rank):
        # Get unpadded fused QKV: [2*unpadded, qkv_size]
        result = base_loader.transform(slices, rank)
        # Split into embeds half and hidden half
        embeds_w = result[:unpadded_hidden_size, :]
        hidden_w = result[unpadded_hidden_size:, :]
        # Pad each half independently
        embeds_w = torch.nn.functional.pad(embeds_w, (0, 0, 0, pad))
        hidden_w = torch.nn.functional.pad(hidden_w, (0, 0, 0, pad))
        return torch.cat([embeds_w, hidden_w], dim=0)

    return SafetensorsWeightLoader(transform=transform)


class Eagle3LlamaDecoderLayer(LlamaDecoderLayer):
    """Eagle3 decoder layer that extends LlamaDecoderLayer.

    Key differences from standard LlamaDecoderLayer:
    1. Takes both `embeds` (token embeddings) and `hidden_states` (from target model)
    2. Concatenates them before attention: [embeds, hidden_states] -> 2x hidden_size
    3. Additional `hidden_norm` for normalizing target hidden states
    4. Overrides QKV projection to accept 2x hidden_size input
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config

        # >>> PARALLELISM: Use the draft MLP DP column that the base layer
        # selected. A size of 1 keeps the plain TP path. <<<
        self.mlp_dp = self.mlp_dp_group.world_size
        self.tp_size = get_tp_group().world_size

        # Override norms with padding-aware variants when hidden_size is padded
        if config.hidden_size != config.unpadded_hidden_size:
            self.input_layernorm = _make_rmsnorm(config)
            self.post_attention_layernorm = _make_rmsnorm(config)

        # Override QKV projection weight to accept 2x hidden_size input
        attn = self.self_attn
        qkv_size = attn.q_size + 2 * attn.kv_size
        qkv_input_size = 2 * config.hidden_size

        attn.qkv_proj_weight = nn.Parameter(
            torch.empty(qkv_input_size, qkv_size, dtype=config.torch_dtype)
        )

        # Re-setup weight loader for the wider QKV.
        # Use interleaved padding: the input is concat(embeds, hidden_states),
        # each padded independently, so zeros appear at positions
        # [unpadded:hidden] and [hidden+unpadded:2*hidden], not at the end.
        set_weight_loader(
            attn.qkv_proj_weight,
            _eagle3_qkv_weight_loader(
                q_size=attn.q_size,
                kv_size=attn.kv_size,
                shard_dim=1,
                num_shards=attn.world_size,
                is_storage_transposed=True,
                num_kv_replicas=attn.num_kv_replicas,
                padded_hidden_size=config.hidden_size,
                unpadded_hidden_size=config.unpadded_hidden_size,
            ),
        )

        # Override weight loaders for o_proj and MLP when hidden_size is padded
        if config.hidden_size != config.unpadded_hidden_size:
            # o_proj: [num_heads*head_dim // tp, hidden_size] — pad hidden_size dim (dim 1)
            set_weight_loader(
                attn.o_proj_weight,
                sharding_weight_loader_with_padding(
                    shard_dim=0,
                    shard_size=(attn.num_attention_heads * attn.head_dim)
                    // attn.world_size,
                    num_shards=attn.world_size,
                    is_storage_transposed=True,
                    pad_dim=1,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
            )

            mlp = self.mlp
            # The MLP shards its intermediate across the enlarged TP group
            # (TP * mlp_dp_size), so use that group's size and rank. LlamaMLP
            # already allocates the weights at the enlarged shard size, so the
            # padded loaders must match. Override the shard rank with the rank
            # inside the enlarged group. A DP size of 1 makes the enlarged group
            # equal the plain TP group, so this stays the plain TP path.
            intermediate_per_rank = config.intermediate_size // mlp.mlp_tp_size
            # gate_proj/up_proj: [hidden_size, intermediate // tp] — pad hidden_size dim (dim 0)
            gate_up_loader = with_rank_override(
                sharding_weight_loader_with_padding(
                    shard_dim=1,
                    shard_size=intermediate_per_rank,
                    num_shards=mlp.mlp_tp_size,
                    is_storage_transposed=True,
                    pad_dim=0,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
                rank=mlp.mlp_tp_rank,
            )
            set_weight_loader(mlp.gate_proj_weight, gate_up_loader)
            set_weight_loader(mlp.up_proj_weight, gate_up_loader)
            # down_proj: [intermediate // tp, hidden_size] — pad hidden_size dim (dim 1)
            set_weight_loader(
                mlp.down_proj_weight,
                with_rank_override(
                    sharding_weight_loader_with_padding(
                        shard_dim=0,
                        shard_size=intermediate_per_rank,
                        num_shards=mlp.mlp_tp_size,
                        is_storage_transposed=True,
                        pad_dim=1,
                        padded_size=config.hidden_size,
                        unpadded_size=config.unpadded_hidden_size,
                    ),
                    rank=mlp.mlp_tp_rank,
                ),
            )

        # Additional norm for target hidden states
        self.hidden_norm = _make_rmsnorm(config)

        # Decode fuses post-attention residual add + RMSNorm into the MLP kernel.
        # Keep a separately loaded weight so the explicit prefill norm retains
        # the checkpoint's original gamma.
        self.post_attention_fused_norm_weight = nn.Parameter(
            torch.empty(config.hidden_size, dtype=config.torch_dtype)
        )
        self.post_attention_fused_norm_eps = (
            config.rms_norm_eps * config.unpadded_hidden_size / config.hidden_size
        )

    def forward(
        self,
        embeds: torch.Tensor,  # [T, hidden_size]
        hidden_states: torch.Tensor,  # [T, hidden_size]
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Eagle3 concatenation logic.

        Returns:
            hidden_states: Output tensor of shape [T, hidden_size].
        """
        is_decode = self._is_decode(attn_metadata)

        # Normalize token embeddings
        embeds = self.input_layernorm(embeds)

        # Normalize target hidden states and save residual
        if self.config.norm_before_residual:
            hidden_states = self.hidden_norm(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)

        # Eagle3: concatenate embeddings with hidden states
        concat_hidden = torch.cat([embeds, hidden_states], dim=-1)  # [T, 2*hidden_size]

        # Self Attention (QKV handles 2x input size)
        hidden_states = self.self_attn(
            hidden_states=concat_hidden,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )

        # Post-attention residual + norm + MLP
        if is_decode:
            # >>> PARALLELISM: MLP full TP. The draft attention runs on the local
            # batch (its KV cache is local), but the MLP shards its intermediate
            # across the enlarged TP group (TP * mlp_dp_size). Gather the local
            # rows across the MLP DP column so every rank in the enlarged group
            # scores the same rows. Run the MLP, add the residual, then slice
            # back to the local rows. A DP size of 1 keeps the plain TP path. <<<
            if self.mlp_dp > 1:
                hidden_states = self.mlp_dp_group.all_gather(hidden_states, dim=0)
                residual = self.mlp_dp_group.all_gather(residual, dim=0)
            hidden_states, residual = self.mlp(
                hidden_states,
                is_prefill=False,
                ln_w=self.post_attention_fused_norm_weight,
                eps=self.post_attention_fused_norm_eps,
                fused_add_tensor=residual,
            )
            hidden_states = residual + hidden_states
            if self.mlp_dp > 1:
                mlp_dp_rank = (
                    (rank // self.tp_size) % self.mlp_dp if rank is not None else None
                )
                hidden_states = _slice_dp_to_local(
                    hidden_states, self.mlp_dp_group, rank=mlp_dp_rank
                )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=True)
            hidden_states = residual + hidden_states

        return hidden_states


class Eagle3LlamaModel(nn.Module):
    """Eagle3 model backbone without the language model head.

    Contains embedding, decoder layers, hidden state combination, normalization,
    and rotary embeddings. Used as the base model inside Eagle3LlamaForCausalLM.

    Args:
        config: LlamaConfig with model hyperparameters.
        start_layer_idx: Layer index for the first decoder layer.
    """

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        unpadded_hidden_size = config.unpadded_hidden_size

        # >>> PARALLELISM: Embedding TP group (TP * embedding_dp_size) + DP column.
        # The draft reuses the process-wide DP groups that the target sets up.
        # Derive embedding_dp_size from the group so the two share one source of
        # truth; a size of 1 makes the enlarged group equal the plain TP group.
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_dp_group,
            get_neuron_embedding_tp_group,
        )

        self.embedding_tp_group = get_neuron_embedding_tp_group()
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.tp_size = get_tp_group().world_size
        self.embedding_tp_rank = self.embedding_tp_group.rank_in_group
        self.embedding_tp_size = self.embedding_tp_group.world_size
        self.embedding_dp_size = self.embedding_dp_group.world_size

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.embedding_tp_group.device_group,
        )

        assert config.num_hidden_layers == 1
        self.layers = nn.ModuleList(
            [Eagle3LlamaDecoderLayer(config, layer_idx=start_layer_idx)]
        )

        # Combine 3 auxiliary hidden states from target model. Replicated on
        # every rank (tp_size=1). On trn3 the [in, out] storage avoids
        # weight-tile transposes on the PE. CPU mode has no device to query.
        if envs.VLLM_NEURON_CPU_MODE:
            fc_layout = "out_in"
        else:
            fc_layout = fc_weight_layout_for_target(get_platform_target())
        self.fc = neuron_nn.ColumnParallelLinear(
            hidden_size * 3,
            hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            weight_layout=fc_layout,
            tp_size=1,
        )

        # Optional norm applied to the concatenated aux hidden states BEFORE fc
        # (speculators `norm_before_fc`, e.g. RedHatAI gpt-oss eagle3 heads).
        # The aux states arrive in the target's hidden layout, so input_norm and
        # fc below both follow config.target_hidden_pad_groups.
        pad_groups = config.target_hidden_pad_groups
        if config.norm_before_fc:
            self.input_norm = _PaddedRMSNorm(config, n_chunks=3, pad_groups=pad_groups)
        else:
            self.input_norm = None

        self.norm = _make_rmsnorm(config)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # Weight loaders. The embedding shards over the enlarged TP group, so
        # override the shard rank with the rank inside that group.
        set_weight_loader(
            self.embed_tokens.weight,
            with_rank_override(
                embedding_sharding_padding_weight_loader(
                    vocab_size_per_rank=self.embed_tokens.vocab_size_per_rank,
                    num_shards=self.embed_tokens.tp_size,
                    padded_hidden_size=hidden_size,
                ),
                rank=self.embedding_tp_rank,
            ),
        )
        set_weight_loader(
            self.fc.weight,
            fc_interleaved_padding_weight_loader(
                unpadded_hidden_size,
                hidden_size,
                pad_groups=pad_groups,
                num_shards=self.fc.tp_size,
                weight_layout=self.fc.weight_layout,
            ),
        )
        if self.input_norm is not None:
            set_weight_loader(
                self.input_norm.weight,
                concat_norm_interleaved_padding_weight_loader(
                    unpadded_hidden_size, hidden_size, n_chunks=3, pad_groups=pad_groups
                ),
            )
        set_weight_loader(self.norm.weight, last_dim_padding_weight_loader(hidden_size))
        for layer in self.layers:
            set_weight_loader(
                layer.input_layernorm.weight,
                last_dim_padding_weight_loader(hidden_size),
            )
            set_weight_loader(
                layer.hidden_norm.weight, last_dim_padding_weight_loader(hidden_size)
            )
            set_weight_loader(
                layer.post_attention_layernorm.weight,
                last_dim_padding_weight_loader(hidden_size),
            )
            set_weight_loader(
                layer.post_attention_fused_norm_weight,
                fused_rmsnorm_padding_weight_loader(unpadded_hidden_size, hidden_size),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        target_hidden_states: torch.Tensor,  # [T, hidden_size] (already combined)
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Eagle3 model backbone.

        Args:
            input_ids: Input token IDs of shape [T].
            positions: Position indices of shape [T].
            target_hidden_states: Hidden states from target model of shape [T, hidden_size].
            attn_metadata: Optional attention metadata for the attention layer.
            rank: World rank tensor for group-local rank calculations.

        Returns:
            tuple: (hidden_states, hidden_prenorm) where hidden_states is the normalized output
                   and hidden_prenorm is the pre-norm state (used as recurrent state).
        """
        first_layer_name = f"layers.{self.layers[0].layer_idx}.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: Embedding DP (decode). Gather the local token ids
        # across the embedding DP column so every rank in the enlarged TP group
        # embeds the same rows, then slice the result back to the local rows. <<<
        if not is_prefill and self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        # Derive each group rank from the one world rank graph input. This
        # prevents the native backend from receiving independent rank tensors.
        embedding_tp_rank = rank % self.embedding_tp_size if rank is not None else None
        tp_rank = rank % self.tp_size if rank is not None else None
        embedding_dp_rank = (
            (rank // self.tp_size) % self.embedding_dp_size
            if rank is not None
            else None
        )

        if is_prefill and self.embedding_dp_size > 1:
            # >>> PARALLELISM: Embedding DP (prefill). The vocab shards over the
            # enlarged TP group, but the decoder attention and MLP run sequence
            # parallel over the plain TP group. Reduce over the enlarged group to
            # the full sequence, then slice to the plain-TP SP layout. This
            # matches the reduce_scatter output of the embedding_dp_size == 1
            # path. <<<
            embeds = self.embed_tokens(
                input_ids, scatter_tokens=False, rank=embedding_tp_rank
            )
            sp_group = get_tp_group()
            sp_rank = tp_rank if tp_rank is not None else sp_group.rank_in_group_tensor
            embeds = _select_spmd_sequence_shard(embeds, sp_rank, self.tp_size)
            # target_hidden_states arrives full [T, H]. Slice it the same way.
            target_hidden_states = _select_spmd_sequence_shard(
                target_hidden_states, sp_rank, self.tp_size
            )
        else:
            embeds = self.embed_tokens(
                input_ids, scatter_tokens=is_prefill, rank=embedding_tp_rank
            )

            if not is_prefill and self.embedding_dp_size > 1:
                embeds = _slice_dp_to_local(
                    embeds, self.embedding_dp_group, rank=embedding_dp_rank
                )

            # Scatter target_hidden_states to match SP layout. The embedding
            # shards over the enlarged TP group, so use its rank for the split.
            if is_prefill and self.embed_tokens.tp_size > 1:
                target_hidden_states = shard_select(
                    target_hidden_states,
                    0,
                    self.embedding_tp_group,
                    rank=embedding_tp_rank,
                )

        # Position embeddings
        position_embeddings = self.rotary_emb(
            positions, device=embeds.device, dtype=embeds.dtype
        )

        # Forward through Eagle3 decoder layer
        hidden_states = self.layers[0](
            embeds=embeds,
            hidden_states=target_hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            rank=rank,
        )

        # Save pre-norm state for recurrent (matches vLLM's hidden_prenorm)
        hidden_prenorm = hidden_states

        # Final norm
        hidden_states = self.norm(hidden_states)

        # All-gather back from SP layout to the full sequence. The decoder ran
        # sequence parallel over the plain TP group, so gather over that group.
        if is_prefill and self.embedding_dp_size > 1:
            tp_group = get_tp_group()
            hidden_states = tp_group.all_gather(hidden_states, dim=0)
            hidden_prenorm = tp_group.all_gather(hidden_prenorm, dim=0)
        elif is_prefill and self.embed_tokens.tp_size > 1:
            tp_group = self.embedding_tp_group
            hidden_states = tp_group.all_gather(hidden_states, dim=0)
            hidden_prenorm = tp_group.all_gather(hidden_prenorm, dim=0)

        return hidden_states, hidden_prenorm

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine 3 auxiliary hidden states into one via fc layer."""
        if self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        return self.fc(hidden_states)


def _build_eagle3_weight_mappings(
    model: Eagle3LlamaModel, use_midlayer_naming: bool
) -> dict[str, str | list[str]]:
    """Build checkpoint mappings for an EAGLE3 draft model."""
    mappings: dict[str, str | list[str]] = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "model.fc.weight": "fc.weight",
        "lm_head.weight": "lm_head.weight",
    }

    if getattr(model, "input_norm", None) is not None:
        mappings["model.input_norm.weight"] = "input_norm.weight"

    for layer_id in range(len(model.layers)):
        if use_midlayer_naming and layer_id == 0:
            ckpt_prefix = "midlayer"
        else:
            ckpt_prefix = f"layers.{layer_id}"

        mappings[f"model.layers.{layer_id}.self_attn.qkv_proj_weight"] = [
            f"{ckpt_prefix}.self_attn.q_proj.weight",
            f"{ckpt_prefix}.self_attn.k_proj.weight",
            f"{ckpt_prefix}.self_attn.v_proj.weight",
        ]
        mappings[f"model.layers.{layer_id}.self_attn.o_proj_weight"] = (
            f"{ckpt_prefix}.self_attn.o_proj.weight"
        )
        mappings[f"model.layers.{layer_id}.mlp.up_proj_weight"] = (
            f"{ckpt_prefix}.mlp.up_proj.weight"
        )
        mappings[f"model.layers.{layer_id}.mlp.gate_proj_weight"] = (
            f"{ckpt_prefix}.mlp.gate_proj.weight"
        )
        mappings[f"model.layers.{layer_id}.mlp.down_proj_weight"] = (
            f"{ckpt_prefix}.mlp.down_proj.weight"
        )
        mappings[f"model.layers.{layer_id}.input_layernorm.weight"] = (
            f"{ckpt_prefix}.input_layernorm.weight"
        )
        mappings[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = (
            f"{ckpt_prefix}.post_attention_layernorm.weight"
        )
        mappings[f"model.layers.{layer_id}.post_attention_fused_norm_weight"] = (
            f"{ckpt_prefix}.post_attention_layernorm.weight"
        )
        mappings[f"model.layers.{layer_id}.hidden_norm.weight"] = (
            f"{ckpt_prefix}.hidden_norm.weight"
        )

    return mappings


class Eagle3LlamaForCausalLM(nn.Module):
    """Eagle3 draft model with hidden state combination."""

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config

        on_device_sampling_config = None
        if config.neuron_config is not None:
            on_device_sampling_config = config.neuron_config.on_device_sampling_config
        self.on_device_sampling = on_device_sampling_config is not None

        # Return per-pass logits from forward() only when debug_logits is
        # enabled (tests validate per-pass logits against GPU goldens). In
        # production, the logits stack is unused — skipping it avoids a
        # torch.stack op and reduces the NEFF output schema size.
        # Pure kv_producer (set by EagleProposer): forward() stops after the
        # KV-fill pass.
        self.kv_fill_only = False
        self._return_logits = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )

        self.model = Eagle3LlamaModel(
            config=config,
            start_layer_idx=start_layer_idx,
        )

        lm_head_vocab = config.draft_vocab_size or config.vocab_size
        self.has_vocab_mapping = (
            config.draft_vocab_size is not None
            and config.draft_vocab_size != config.vocab_size
        )
        if self.has_vocab_mapping:
            self.register_buffer(
                "_draft_to_target",
                torch.zeros(config.draft_vocab_size, dtype=torch.int32),
                persistent=False,
            )

        # >>> PARALLELISM: LM head TP group (TP * lm_head_dp_size) + DP column,
        # plus the TARGET MLP DP column, which describes the row layout the
        # target hands over. Not the draft's column: the draft MLP uses its own
        # groups (see Eagle3LlamaDecoderLayer). Derive the sizes from the groups
        # so config and groups share one source of truth. <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_dp_group,
            get_neuron_lm_head_tp_group,
            get_neuron_mlp_dp_group,
        )

        self.lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.tp_size = get_tp_group().world_size
        self.lm_head_tp_rank = self.lm_head_tp_group.rank_in_group
        self.lm_head_tp_size = self.lm_head_tp_group.world_size
        self.target_mlp_dp_group = get_neuron_mlp_dp_group()
        self.lm_head_dp_size = self.lm_head_dp_group.world_size
        self.target_mlp_dp_size = self.target_mlp_dp_group.world_size

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            lm_head_vocab,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling,
            tp_group=self.lm_head_tp_group.device_group,
            weight_layout="in_out",
        )

        # The lm_head shards over the enlarged TP group, so override the shard
        # rank with the rank inside that group.
        set_weight_loader(
            self.lm_head.weight,
            with_rank_override(
                sharding_weight_loader_with_padding(
                    shard_dim=1,
                    shard_size=self.lm_head.out_features_per_rank,
                    num_shards=self.lm_head.tp_size,
                    is_storage_transposed=True,
                    pad_dim=0,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
                rank=self.lm_head_tp_rank,
            ),
        )

        if self.on_device_sampling:
            self.sampler = Sampler(
                on_device_sampling_config,
                process_group=self.lm_head_tp_group.device_group,
            )

    def _sample_draft_token(
        self, logits: torch.Tensor, rank: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample a draft token from logits (greedy or on-device)."""
        if self.on_device_sampling:
            lm_head_tp_rank = (
                rank % self.lm_head_tp_size
                if rank is not None
                else self.lm_head_tp_group.rank_in_group_tensor
            )
            return self.sampler(
                logits,
                tp_rank=lm_head_tp_rank,
            ).to(torch.int32)
        return torch.argmax(logits, dim=-1).to(torch.int32)

    def _lm_head_and_sample(
        self, hidden_states: torch.Tensor, rank: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the lm_head and sample one draft token per local row.

        With lm_head DP the head shards its vocab over the enlarged TP group.
        Gather the local sampling rows across the lm_head DP column so every
        rank in that group scores the same rows, run the head, sample, then
        slice the tokens and logits back to the local rows. A DP size of 1 keeps
        the plain TP path.

        Args:
            hidden_states: Local per-request rows of shape [B_local, hidden].

        Returns:
            (draft_token_ids, logits), both sliced to the local rows.
        """
        if self.lm_head_dp_size > 1:
            hidden_states = self.lm_head_dp_group.all_gather(hidden_states, dim=0)
        logits = self.lm_head(hidden_states)
        draft_token_ids = self._sample_draft_token(logits, rank=rank)
        if self.lm_head_dp_size > 1:
            lm_head_dp_rank = (
                (rank // self.tp_size) % self.lm_head_dp_size
                if rank is not None
                else None
            )
            draft_token_ids = _slice_dp_to_local(
                draft_token_ids, self.lm_head_dp_group, rank=lm_head_dp_rank
            )
            logits = _slice_dp_to_local(
                logits, self.lm_head_dp_group, rank=lm_head_dp_rank
            )
        return draft_token_ids, logits

    def _map_output_token_ids_to_target_vocab(
        self,
        stacked_tokens: torch.Tensor,
        drafts_only_stacked: torch.Tensor,
        bonus_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map draft outputs to target vocab while preserving target bonus IDs."""
        if not self.has_vocab_mapping:
            return stacked_tokens, drafts_only_stacked

        mapped_drafts_only = self._draft_to_target[drafts_only_stacked.long()]
        if bonus_token_ids is None:
            return mapped_drafts_only, mapped_drafts_only

        mapped_stacked = torch.cat(
            [bonus_token_ids.unsqueeze(1), mapped_drafts_only], dim=1
        )
        return mapped_stacked, mapped_drafts_only

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full draft propose: KV cache update + recurrent decode loop.

        Processes the initial KV cache update pass, then runs num_speculative_tokens - 1
        recurrent decode steps. The recurrent loop is unrolled by torch.compile into a
        single static graph since num_speculative_tokens is a compile-time constant.

        Args:
            input_ids: Shifted target token IDs [T].
            positions: Position indices [T].
            initial_target_hidden_states: Concatenated 3x auxiliary hidden
                states from the target model [T, hidden_size*3].
            attn_metadata: Attention metadata for the draft model layer.
            sampling_positions: Indices into the token sequence for each request [bs].
            rank: World rank tensor for group-local rank calculations.
            raw_sampled_token_ids: Raw rejection sampler output [bs, max_spec_len+1]
                with -1 for rejected, or [bs, 1] for prefill. Used by
                _extract_accepted_tokens to patch input_ids and adjust
                sampling_positions on-device.

        Returns:
            ``(stacked_tokens, drafts_only_stacked, stacked_logits)``.

            ``stacked_tokens`` is ``[bs, 1+num_spec]`` (column 0 = bonus from
            this step's rejection sampler, columns 1.. = drafts for the next
            step). Flatten with ``view(-1)`` for the next target NEFF's
            ``input_ids``.

            ``drafts_only_stacked`` is ``[bs, num_spec]`` (drafts only,
            contiguous). Flatten with ``view(-1)`` for the next target
            NEFF's ``spec_decode_metadata.draft_token_ids``. Separate output
            because slicing ``stacked_tokens[:, 1:]`` produces a non-contiguous
            view that ``.contiguous()`` cannot resolve on Neuron device.

            ``stacked_logits`` is ``[bs, num_spec, vocab]`` or ``None`` —
            populated only when ``debug_logits_dir`` is set
            (used by ``test_eagle3_fused_propose.py`` for per-pass logit
            validation). In production it's ``None`` to skip the
            ``torch.stack`` and shrink the NEFF output schema.
        """
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        # On-device extract_next_token_ids + compute_token_indices_to_sample
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus_token_ids = (
                self._extract_accepted_tokens(
                    input_ids,
                    sampling_positions,
                    raw_sampled_token_ids,
                )
            )
        else:
            bonus_token_ids = None

        # On decode a target with mlp_dp_size > 1 returns aux states gathered
        # over its MLP DP column, so slice them back to local rows. On prefill
        # they arrive as full rows, so the slice must not run.
        first_layer_name = f"layers.{self.model.layers[0].layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        is_prefill = base_meta["max_query_len"] > base_meta["decode_token_threshold"]
        if not is_prefill and self.target_mlp_dp_size > 1:
            target_mlp_dp_rank = (
                (rank // self.tp_size) % self.target_mlp_dp_size
                if rank is not None
                else None
            )
            initial_target_hidden_states = _slice_dp_to_local(
                initial_target_hidden_states,
                self.target_mlp_dp_group,
                rank=target_mlp_dp_rank,
            )

        # Initial pass: combine aux hidden states and update KV cache.
        target_hidden_states = self.model.combine_hidden_states(
            initial_target_hidden_states
        )

        hidden_states, recurrent_state = self.model(
            input_ids=input_ids,
            positions=positions,
            target_hidden_states=target_hidden_states,
            attn_metadata=attn_metadata,
            rank=rank,
        )

        if self.kv_fill_only:
            # The KV cache is written; skip the lm_head, the sampling and the
            # recurrent passes. One [bs, num_spec] placeholder serves both draft
            # outputs, as the normal prefill path does.
            placeholder = torch.zeros(
                (sampling_positions.shape[0], self.num_speculative_tokens),
                dtype=torch.int32,
                device=input_ids.device,
            )
            return placeholder, placeholder, None

        # Gather per-request states at sampling positions
        hidden_states = hidden_states[sampling_positions]
        recurrent_state = recurrent_state[sampling_positions]
        cur_positions = positions[sampling_positions]

        # Sample first draft token
        draft_token_ids, logits = self._lm_head_and_sample(hidden_states, rank=rank)

        # Collect draft tokens. When bonus_token_ids is provided
        # (steady-state spec), prepend it so the output shape is
        # [bs, 1 + num_spec], directly consumable as the next target NEFF's
        # flattened input_ids (via view(-1)). This keeps the
        # bonus-extraction-and-cat inside this compiled graph — no
        # Python-side ops between draft NEFF and next target NEFF.
        all_draft_tokens = (
            [bonus_token_ids, draft_token_ids]
            if bonus_token_ids is not None
            else [draft_token_ids]
        )
        # Per-pass logits only collected in debug mode (saves a torch.stack
        # and shrinks the NEFF output schema in production).
        all_logits = [logits] if self._return_logits else None

        # Extract block_table and block_size from attn_metadata for slot_mapping
        first_layer_name = f"layers.{self.model.layers[0].layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        block_table_tensor = base_meta["block_table_tensor"]
        block_size = base_meta["block_size"]
        max_blocks_per_seq = base_meta["max_blocks_per_seq"]

        # Recurrent decode loop (torch.compile unrolls this Python for-loop
        # into a single static graph since num_speculative_tokens is constant).
        # When num_speculative_tokens == 1, loop runs 0 times.
        for _ in range(self.num_speculative_tokens - 1):
            cur_input_ids = draft_token_ids
            # Map draft token IDs to target vocab for embedding lookup
            if self.has_vocab_mapping:
                cur_input_ids = self._draft_to_target[cur_input_ids.long()]
            cur_positions = cur_positions + 1

            # On-device slot_mapping computation
            slot_mapping = compute_slot_mapping(
                cur_positions, block_table_tensor, block_size
            )

            # Build step attn_metadata (only slot_mapping changes)
            step_attn_metadata = {}
            for layer in self.model.layers:
                ln = f"layers.{layer.layer_idx}.self_attn"
                step_attn_metadata[ln] = {
                    "block_table_tensor": block_table_tensor,
                    "slot_mapping": slot_mapping,
                    "max_query_len": 1,
                    "block_size": block_size,
                    "max_blocks_per_seq": max_blocks_per_seq,
                    "decode_token_threshold": 1,
                }

            # Recurrent decode step
            hidden_states, recurrent_state = self.model(
                input_ids=cur_input_ids,
                positions=cur_positions,
                target_hidden_states=recurrent_state,
                attn_metadata=step_attn_metadata,
                rank=rank,
            )

            # Sample next draft token
            draft_token_ids, logits = self._lm_head_and_sample(hidden_states, rank=rank)

            all_draft_tokens.append(draft_token_ids)
            if all_logits is not None:
                all_logits.append(logits)

        stacked_tokens = torch.stack(all_draft_tokens, dim=1)
        # Drafts-only stack: stride is (1,) so the runner can flatten with
        # ``.view(-1)`` and feed it as the next step's
        # ``spec_decode_metadata.draft_token_ids`` without an on-device
        # ``[:, 1:].contiguous()`` (which is unsupported on Neuron). When
        # bonus_token_ids is None (prefill / first decode), this is a copy
        # of stacked_tokens; under torch.compile, CSE elides the
        # duplication.
        drafts_only_stacked = torch.stack(
            all_draft_tokens[1:] if bonus_token_ids is not None else all_draft_tokens,
            dim=1,
        )

        stacked_tokens, drafts_only_stacked = (
            self._map_output_token_ids_to_target_vocab(
                stacked_tokens, drafts_only_stacked, bonus_token_ids
            )
        )

        stacked_logits = (
            torch.stack(all_logits, dim=1) if all_logits is not None else None
        )
        return stacked_tokens, drafts_only_stacked, stacked_logits

    def _extract_accepted_tokens(
        self,
        input_ids: torch.Tensor,
        sampling_positions: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """On-device extract_next_token_ids + compute_token_indices_to_sample.

        Thin wrapper around module-level extract_accepted_tokens().
        See that function for full documentation.
        """
        return extract_accepted_tokens(
            input_ids,
            sampling_positions,
            raw_sampled_token_ids,
            self.config.vocab_size,
            self.num_speculative_tokens,
        )

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ):
        llama_config = LlamaConfig.from_configs(
            hf_config=config, neuron_config=neuron_config
        )
        llama_config.use_draft_mlp_groups = True
        return cls(
            llama_config,
            start_layer_idx=start_layer_idx,
        )

    def _init_kv_cache_scales(self, device: torch.device) -> None:
        """Initialize KV cache quantization scales for Eagle3 attention layers.

        The Eagle3 speculator checkpoint does not contain k_scale/v_scale
        tensors. When the KV cache uses a quantized dtype (FP8), these must
        be set to 1.0 so _write_kv_cache can multiply without hitting None.
        """
        from vllm.config import get_current_vllm_config
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        ones = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
        for layer in self.model.layers:
            attn = layer.self_attn
            if attn.k_scale is None:
                attn.k_scale = ones.clone()
            if attn.v_scale is None:
                attn.v_scale = ones.clone()
            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Lightweight weight loading used during CPU compile.

        The Eagle3 speculator checkpoint carries no k_scale/v_scale tensors, so
        (unlike the target model's load_weights_lite) there are no scales to read
        from the checkpoint. We only need to initialize the draft attention
        layers' KV-cache scales to 1.0 so the FP8 KV-cache write path's
        ``* self.k_scale`` does not hit ``None`` during tracing. Without this the
        CPU-compile path skips full weight loading (which is what normally calls
        _init_kv_cache_scales), leaving k_scale None and crashing the trace.
        """
        self._init_kv_cache_scales(device)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load rank-sharded checkpoint to device with pipelined data movement.

        Supports two checkpoint naming conventions:
        - "midlayer.*" (used by some EAGLE3 checkpoints for layer 0)
        - "layers.{layer_id}.*" (standard naming)

        Args:
            checkpoint_path: Directory or HuggingFace model ID containing the
                checkpoint files.
            device: Target device for weight tensors.
            cache_dir: Cache directory for HuggingFace downloads.
        """
        import os

        from safetensors import safe_open

        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        safetensor_files = sorted(
            f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
        )
        checkpoint_keys = set()
        for f in safetensor_files:
            with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                checkpoint_keys.update(sf.keys())

        use_midlayer_naming = any(k.startswith("midlayer.") for k in checkpoint_keys)

        mappings = _build_eagle3_weight_mappings(self.model, use_midlayer_naming)

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict
        self.load_state_dict(rank_sharded_checkpoint, strict=True, assign=True)

        self._init_kv_cache_scales(device)

        # Load d2t and pre-compute direct mapping buffer
        if self.has_vocab_mapping:
            for f in safetensor_files:
                with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                    if "d2t" in sf.keys():
                        d2t = sf.get_tensor("d2t")
                        # Pre-compute direct mapping as int32 buffer (moves with .to(device))
                        draft_ids = torch.arange(len(d2t))
                        setattr(
                            self,
                            "_draft_to_target",
                            (draft_ids + d2t).to(torch.int32),
                        )

    def get_kv_spec(self):
        """Returns KV cache specification for vLLM integration.

        Returns:
            KVSpec: Specification containing layer-wise KV cache requirements.
        """
        layers = []
        for _, layer in enumerate(self.model.layers):
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        """Binds pre-allocated KV cache tensors to the model layers.

        Args:
            kv_caches: Dictionary mapping layer names to [key_cache, value_cache] tensor pairs.

        Raises:
            RuntimeError: If KV cache for any required layer is not initialized.
        """
        for _, layer in enumerate(self.model.layers):
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Packed FP8 K cache is one dim larger than V; auto-detect so the
            # draft's inherited LlamaAttention takes the packed swizzle paths.
            # hasattr guard mirrors LlamaForCausalLM.bind_kv_cache.
            if hasattr(layer.self_attn, "fp8_packed"):
                layer.self_attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1
