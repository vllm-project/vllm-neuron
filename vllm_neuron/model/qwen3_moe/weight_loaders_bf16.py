# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for Qwen3 MoE BF16 model.

Qwen3 MoE differences from GPT-OSS:
- No attention bias (no QKV bias, no O-proj bias)
- No attention sinks
- No hidden dim padding
- Experts stored as individual tensors per expert in HF checkpoint
  (model.layers.X.mlp.experts.Y.gate_proj.weight, etc.)
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    num_kv_replicas: int = 1,
    kv_num_shards: int | None = None,
) -> SafetensorsWeightLoader:
    """Fuses Q, K, V weights with per-tensor sharding. No hidden padding needed."""
    kv_shards = kv_num_shards if kv_num_shards is not None else num_shards

    if kv_shards >= num_kv_heads:
        num_kv_heads_per_rank = 1
    else:
        num_kv_heads_per_rank = num_kv_heads // kv_shards

    kv_size_per_rank = num_kv_heads_per_rank * head_dim

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3, (
            "fused_qkv_weight_loader expects [Q, K, V] slices in order"
        )

        q_slice, k_slice, v_slice = slices

        q_start = rank * q_size
        q_end = q_start + q_size
        q_tensor = q_slice[q_start:q_end, :]

        if kv_shards != num_shards:
            tp_rank_for_kv = rank // (num_shards // kv_shards)
            kv_replicas_for_kv = max(kv_shards // num_kv_heads, 1)
            kv_rank = tp_rank_for_kv // kv_replicas_for_kv
        else:
            kv_rank = rank // num_kv_replicas

        if kv_shards >= num_kv_heads:
            if kv_rank < num_kv_heads:
                kv_start = kv_rank * head_dim
                kv_end = kv_start + head_dim
                k_tensor = k_slice[kv_start:kv_end, :]
                v_tensor = v_slice[kv_start:kv_end, :]
            else:
                k_tensor = torch.zeros(head_dim, hidden_size, dtype=q_tensor.dtype)
                v_tensor = torch.zeros(head_dim, hidden_size, dtype=q_tensor.dtype)
        else:
            kv_start = kv_rank * kv_size_per_rank
            kv_end = kv_start + kv_size_per_rank
            k_tensor = k_slice[kv_start:kv_end, :]
            v_tensor = v_slice[kv_start:kv_end, :]

        result = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)
        return result.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for o_proj that shards input dimension."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        slice_obj = slices[0]

        # Checkpoint is [hidden, out], shard on dim 1
        start = rank * shard_size
        tensor = slice_obj[:, start : start + shard_size]

        return tensor.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
    num_experts: int = 128,
) -> SafetensorsWeightLoader:
    """Weight loader for gate_up_proj from per-expert HF checkpoint tensors.

    Receives 2*num_experts slices: [gate_0, gate_1, ..., gate_E-1, up_0, up_1, ..., up_E-1].
    Each gate/up slice is [I, H] in HF format.
    Shards gate and up independently on I, then concatenates to [E, H, I_shard*2].
    """
    # shard_size is I_per_rank * 2; each of gate/up is sharded by I_per_rank
    i_per_rank = shard_size // 2

    def transform(slices: list, rank: int) -> torch.Tensor:
        E = num_experts
        assert len(slices) == 2 * E, (
            f"Expected {2 * E} slices (gate+up), got {len(slices)}"
        )

        gate_slices = slices[:E]  # each [I, H]
        up_slices = slices[E:]  # each [I, H]

        # Stack and transpose: [E, I, H] -> [E, H, I]
        gates = torch.stack([s[:].to(torch.bfloat16) for s in gate_slices]).transpose(
            1, 2
        )
        ups = torch.stack([s[:].to(torch.bfloat16) for s in up_slices]).transpose(1, 2)

        # Shard gate and up independently on I dimension
        start_idx = (rank % num_shards) * i_per_rank
        gates_shard = gates[:, :, start_idx : start_idx + i_per_rank]
        ups_shard = ups[:, :, start_idx : start_idx + i_per_rank]

        # Concatenate gate||up: [E, H, I_per_rank*2]
        return torch.cat([gates_shard, ups_shard], dim=2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def expert_down_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
    num_experts: int = 128,
) -> SafetensorsWeightLoader:
    """Weight loader for down_proj from per-expert HF checkpoint tensors.

    Receives num_experts slices, each [H, I] in HF format.
    Stacks into [E, I, H], then shards on I dimension.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        E = num_experts
        assert len(slices) == E, f"Expected {E} slices, got {len(slices)}"

        # Stack: each slice is [H, I], transpose to [I, H], stack to [E, I, H]
        downs = torch.stack([s[:].to(torch.bfloat16).T for s in slices])

        # Shard on I dimension (dim=1)
        start_idx = (rank % num_shards) * shard_size
        return downs[:, start_idx : start_idx + shard_size, :].contiguous()

    return SafetensorsWeightLoader(transform=transform)
