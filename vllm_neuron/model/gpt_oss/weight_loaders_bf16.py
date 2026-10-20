# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,  # This should be kv_size PER RANK (num_kv_heads_per_rank * head_dim)
    shard_dim: int,
    num_shards: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    num_kv_replicas: int = 1,
    kv_num_shards: int | None = None,
) -> SafetensorsWeightLoader:
    """
    Create a SafetensorsWeightLoader that fuses Q, K, V weights with per-tensor sharding.

    Args:
        kv_num_shards: Number of shards for KV (defaults to num_shards).
            When attention DP is enabled and kv_needs_a2a is False, KV is sharded
            across TP only (kv_num_shards=tp_size) while Q is sharded across
            TP*attention_dp (num_shards=tp_size*attention_dp_size).
    """
    kv_shards = kv_num_shards if kv_num_shards is not None else num_shards

    # Calculate KV heads per rank based on KV-specific shard count
    if kv_shards >= num_kv_heads:
        num_kv_heads_per_rank = 1
    else:
        num_kv_heads_per_rank = num_kv_heads // kv_shards

    kv_size_per_rank = num_kv_heads_per_rank * head_dim

    def _pad_hidden(tensor: torch.Tensor) -> torch.Tensor:
        """Pad dim 1 (in_features/hidden) to expecthidden_sizeed_hidden_size."""
        pad_amount = hidden_size - tensor.shape[1]
        if pad_amount <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_amount))

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3, (
            "fused_qkv_weight_loader expects [Q, K, V] slices in order"
        )

        q_slice, k_slice, v_slice = slices

        # Q is always sharded normally across all ranks
        q_start = rank * q_size
        q_end = q_start + q_size

        # Determine KV head range for this rank.
        # When kv_num_shards differs from num_shards (attention DP with kv_needs_a2a=False),
        # derive the KV-specific rank from the Q rank.
        if kv_shards != num_shards:
            # rank is effective_q_rank (interleaved). KV rank is TP-based.
            tp_rank_for_kv = rank // (num_shards // kv_shards)
            kv_replicas_for_kv = max(kv_shards // num_kv_heads, 1)
            kv_rank = tp_rank_for_kv // kv_replicas_for_kv
        else:
            kv_rank = rank // num_kv_replicas

        # Checkpoint stores [out_features, in_features]
        q_tensor = _pad_hidden(q_slice[q_start:q_end, :])

        if kv_shards >= num_kv_heads:
            # Each rank gets 1 KV head (possibly padded for extra ranks)
            if kv_rank < num_kv_heads:
                kv_start = kv_rank * head_dim
                kv_end = kv_start + head_dim
                k_tensor = _pad_hidden(k_slice[kv_start:kv_end, :])
                v_tensor = _pad_hidden(v_slice[kv_start:kv_end, :])
            else:
                # Padded KV
                k_tensor = torch.zeros(head_dim, hidden_size, dtype=q_tensor.dtype)
                v_tensor = torch.zeros(head_dim, hidden_size, dtype=q_tensor.dtype)
        else:
            # Each rank gets MULTIPLE KV heads
            kv_start = kv_rank * kv_size_per_rank
            kv_end = kv_start + kv_size_per_rank
            k_tensor = _pad_hidden(k_slice[kv_start:kv_end, :])
            v_tensor = _pad_hidden(v_slice[kv_start:kv_end, :])

        result = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

        # We transpose since the QKV weight in the model is transposed from the original [out_features, in_features]
        return result.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_heads: int,
    head_dim: int,
    num_kv_replicas: int = 1,
    kv_num_shards: int | None = None,
) -> SafetensorsWeightLoader:
    kv_shards = kv_num_shards if kv_num_shards is not None else num_shards

    if kv_shards >= num_kv_heads:
        num_kv_heads_per_rank = 1
    else:
        num_kv_heads_per_rank = num_kv_heads // kv_shards

    kv_size_per_rank = num_kv_heads_per_rank * head_dim

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3

        q_slice, k_slice, v_slice = slices

        q_start = rank * q_size
        q_end = q_start + q_size
        q_tensor = q_slice[q_start:q_end]

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
                k_tensor = k_slice[kv_start:kv_end]
                v_tensor = v_slice[kv_start:kv_end]
            else:
                k_tensor = torch.zeros(head_dim, dtype=q_tensor.dtype)
                v_tensor = torch.zeros(head_dim, dtype=q_tensor.dtype)
        else:
            kv_start = kv_rank * kv_size_per_rank
            kv_end = kv_start + kv_size_per_rank
            k_tensor = k_slice[kv_start:kv_end]
            v_tensor = v_slice[kv_start:kv_end]

        return torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for o_proj that shards and pads hidden dimension."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        slice_obj = slices[0]

        # Checkpoint is [hidden, out], shard on dim 1
        start = rank * shard_size
        tensor = slice_obj[:, start : start + shard_size]

        # Pad hidden dim (dim 0) if needed
        pad = hidden_size - tensor.shape[0]
        if pad > 0:
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad))

        # O proj weight expects [out, hidden] shape tensor
        return tensor.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def sinks_sharding_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader for sink tokens (1D tensor).

    Args:
        shard_size: Size of each shard (num_attention_heads_per_rank).
        num_shards: Total number of shards (TP world size).

    Returns:
        SafetensorsWeightLoader configured for sinks sharding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1

        slice_obj = slices[0]
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        return slice_obj[start_idx:end_idx].to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


def expert_sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader for expert weights (3D tensors).

    Args:
        shard_dim: Dimension to shard along (typically 2 for gate_up, 1 for down).
        shard_size: Size of each shard along shard_dim.
        num_shards: Total number of shards (TP world size).

    Returns:
        SafetensorsWeightLoader configured for expert weight sharding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "expert_sharding_weight_loader only supports a single tensor"
        )

        slice_obj = slices[0]
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        sl = [slice(None)] * len(slice_obj.get_shape())
        sl[shard_dim] = slice(start_idx, end_idx)

        return slice_obj[tuple(sl)]

    return SafetensorsWeightLoader(transform=transform)


def expert_bias_sharding_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader for expert biases (2D tensors).

    Args:
        shard_dim: Dimension to shard along.
        shard_size: Size of each shard.
        num_shards: Total number of shards.

    Returns:
        SafetensorsWeightLoader configured for expert bias sharding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1

        slice_obj = slices[0]
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        sl = [slice(None)] * len(slice_obj.get_shape())
        sl[shard_dim] = slice(start_idx, end_idx)

        return slice_obj[tuple(sl)]

    return SafetensorsWeightLoader(transform=transform)


def _dequantize_mxfp4_to_bf16(blocks: torch.Tensor, scales: torch.Tensor):
    """Dequantize MXFP4 (Microscaling FP4) packed tensor to bfloat16.

    Each byte in blocks contains two FP4 values (low and high nibbles).
    Uses a lookup table to map 4-bit indices to FP4 values, then scales
    by 2^(scale - 127).

    Args:
        blocks: Packed FP4 tensor of shape [..., G, B] where each element
            contains two 4-bit values.
        scales: Per-group scale factors of shape [..., G] as uint8 exponents.

    Returns:
        Dequantized bfloat16 tensor of shape [..., G*B*2].
    """
    dequant_dtype = torch.bfloat16

    FP4_LUT = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=dequant_dtype,
        device=blocks.device,
    )

    exp = (scales.to(torch.int32) - 127).unsqueeze(-1)  # [..., G, 1]

    out = torch.empty(
        *blocks.shape[:-1],
        blocks.shape[-1] * 2,
        dtype=dequant_dtype,
        device=blocks.device,
    )  # [..., G, B*2]
    out[..., 0::2] = FP4_LUT[(blocks & 0x0F).long()]
    out[..., 1::2] = FP4_LUT[(blocks >> 4).long()]
    torch.ldexp(out, exp, out=out)  # broadcasts [..., G, 1] over [..., G, B*2]

    return out.flatten(-2)  # [..., G*B*2]


def expert_gate_up_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for gate_up_proj that correctly shards both gate and up portions.

    Performs dequantization from fp4 to bf16, and pads
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        # Dense bf16 path: checkpoint from save_pretrained() has a single tensor
        # HF layout: [E, H, I*2] with gate/up interleaved on last dim
        if len(slices) == 1:
            tensor = torch.tensor(slices[0][:]).to(torch.bfloat16)
            # Shard on I*2 dimension (dim=2)
            start_idx = (rank % num_shards) * shard_size
            dequant_tensor = tensor[:, :, start_idx : start_idx + shard_size]
            # Pad H (dim=1) and I (dim=2)
            H_padding = hidden_size - dequant_tensor.shape[1]
            I_padding = shard_size - dequant_tensor.shape[2]
            if H_padding > 0 or I_padding > 0:
                dequant_tensor = torch.nn.functional.pad(
                    dequant_tensor, (0, I_padding, 0, H_padding)
                )
            # De-interleave on last dim: [gate0,up0,gate1,up1] -> [gate...,up...]
            gate, up = dequant_tensor[:, :, ::2], dequant_tensor[:, :, 1::2]
            return torch.cat([gate, up], dim=2).contiguous()

        assert len(slices) == 2, "Expected [block, scales]"

        # Note that tensors below are packed (2 FP4 values per value, which is why block dim
        # is 16 instead of the 32 standard in MXFP4)
        # Block Shape: [E, I*2, H / 32, 16]
        # Scale Shape: [E, I*2, H / 32]
        # We shard on I*2 dim for this (essentially just a CPL)
        blocks, scales = slices[0], slices[1]

        # Compute shard for provided rank
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Pull out shard needed by this rank
        blocks: torch.Tensor = blocks[:, start_idx:end_idx]
        scales: torch.Tensor = scales[:, start_idx:end_idx]

        # Dequantize to bf16 (E, I*2, H)
        dequant_tensor: torch.Tensor = _dequantize_mxfp4_to_bf16(blocks, scales)

        # Pad tensor if needed
        I_padding = shard_size - dequant_tensor.shape[1]
        H_padding = hidden_size - dequant_tensor.shape[2]
        if I_padding > 0 or H_padding > 0:
            dequant_tensor = torch.nn.functional.pad(
                dequant_tensor, (0, H_padding, 0, I_padding)
            )

        # Convert gate_up proj from interleaved to chunked:
        # [gate0, up0, gate1, up1]] -> [gate0, ..., gateN, up0, ..., upN]
        gate, up = dequant_tensor[:, ::2], dequant_tensor[:, 1::2]
        result = torch.cat([gate, up], dim=1)

        return result.transpose(1, 2).contiguous()  # Parameter expects transposed shape

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_bias_sharding_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """
    Bias loader for gate_up_proj: shards interleaved data, de-interleaves,
    adds +1 to up bias, then pads.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        bias_slice = slices[0]  # [E, I*2]

        start = (rank % num_shards) * shard_size
        end = start + shard_size
        gate_up_bias = bias_slice[:, start:end]  # Actual elements for this rank

        # De-interleave before padding
        gate = gate_up_bias[:, ::2]  # [E, actual//2]
        up = gate_up_bias[:, 1::2].clone()  # [E, actual//2]

        # Add +1 to up
        up = up + 1

        # Now pad to target size
        target_per_category = shard_size // 2
        gate_pad = target_per_category - gate.shape[1]
        up_pad = target_per_category - up.shape[1]

        if gate_pad > 0:
            gate = torch.nn.functional.pad(gate, (0, gate_pad))
        if up_pad > 0:
            up = torch.nn.functional.pad(up, (0, up_pad))

        return torch.cat([gate, up], dim=1)

    return SafetensorsWeightLoader(transform=transform)


def expert_down_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for down_proj: shards first on I dim, dequantize, then pad"""

    def transform(slices: list, rank: int) -> torch.Tensor:
        # Dense bf16 path: HF layout is [E, I, H]
        if len(slices) == 1:
            tensor = torch.tensor(slices[0][:]).to(torch.bfloat16)
            # Shard on I dimension (dim=1)
            start_idx = (rank % num_shards) * shard_size
            dequant_tensor = tensor[:, start_idx : start_idx + shard_size, :]
            # Pad I (dim=1) and H (dim=2)
            I_padding = shard_size - dequant_tensor.shape[1]
            H_padding = hidden_size - dequant_tensor.shape[2]
            if I_padding > 0 or H_padding > 0:
                dequant_tensor = torch.nn.functional.pad(
                    dequant_tensor, (0, H_padding, 0, I_padding)
                )
            return dequant_tensor.contiguous()

        assert len(slices) == 2, "Expected [blocks, scales]"

        # Note that tensors below are packed (2 FP4 values per value, which is why block dim
        # is 16 instead of the 32 standard in MXFP4)
        # Block Shape: [E, H, I / 32, 16]
        # Scale Shape: [E, H, I / 32]
        # We shard on I dim for this (essentially just a RPL)
        blocks, scales = slices[0], slices[1]

        # Compute the element range we need for this rank
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Compute which groups contain these elements (each group = 32 elements)
        start_group = start_idx // 32
        end_group = (end_idx + 31) // 32  # Round up

        # Slice only the groups needed for this rank
        blocks: torch.Tensor = blocks[:, :, start_group:end_group]
        scales: torch.Tensor = scales[:, :, start_group:end_group]

        # Dequantize to bf16 (E, H, groups*32)
        dequant_tensor: torch.Tensor = _dequantize_mxfp4_to_bf16(blocks, scales)

        # Extract the exact shard from the dequantized groups
        offset = start_idx - (start_group * 32)
        dequant_tensor = dequant_tensor[:, :, offset : offset + shard_size]

        # Pad tensor if needed
        I_padding = shard_size - dequant_tensor.shape[2]
        H_padding = hidden_size - dequant_tensor.shape[1]
        if I_padding > 0 or H_padding > 0:
            dequant_tensor = torch.nn.functional.pad(
                dequant_tensor, (0, I_padding, 0, H_padding)
            )

        return dequant_tensor.transpose(
            1, 2
        ).contiguous()  # Parameter expects transposed shape

    return SafetensorsWeightLoader(transform=transform)
