# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the Qwen3-VL vision encoder.

Custom sharding for the fused QKV weight/bias which stores Q, K, V
concatenated as [Q_all | K_all | V_all]. The default ColumnParallelLinear
loader takes a contiguous row slice, giving rank 0 all of Q and nothing
of K/V. The correct sharding interleaves heads across ranks.
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def vis_qkv_weight_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for the fused QKV weight of vision attention.

    Shards the fused QKV weight ``[3*H, H]`` by interleaving heads:
    ``rank r → rows [r*hd:(r+1)*hd]`` from Q, K, and V each,
    where ``hd = num_heads_per_rank * head_dim``.

    Returns the result transposed to ``[H, 3*hd]`` to match the raw
    parameter layout (input × output_per_rank) for NF.qkv_proj.

    Args:
        num_heads_per_rank: Number of attention heads per TP rank.
        head_dim: Dimension per attention head.
        vis_hidden_size: Total vision hidden size (H = num_heads * head_dim).

    Returns:
        SafetensorsWeightLoader with interleaved QKV sharding.
    """
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        w = slices[0][:]  # [3*H, H]
        q = w[rank * hd : (rank + 1) * hd]  # [hd, H]
        k = w[H + rank * hd : H + (rank + 1) * hd]  # [hd, H]
        v = w[2 * H + rank * hd : 2 * H + (rank + 1) * hd]  # [hd, H]
        fused = torch.cat([q, k, v], dim=0)  # [3*hd, H]
        return fused.T.contiguous()  # [H, 3*hd] — transposed for raw param

    return SafetensorsWeightLoader(transform=transform)


def vis_qkv_bias_loader(
    num_heads_per_rank: int,
    head_dim: int,
    vis_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for the fused QKV bias of vision attention.

    Mirrors ``vis_qkv_weight_loader`` but for the 1-D bias vector ``[3*H]``.

    Args:
        num_heads_per_rank: Number of attention heads per TP rank.
        head_dim: Dimension per attention head.
        vis_hidden_size: Total vision hidden size (H = num_heads * head_dim).

    Returns:
        SafetensorsWeightLoader with interleaved QKV bias sharding.
    """
    hd = num_heads_per_rank * head_dim
    H = vis_hidden_size

    def transform(slices, rank):
        b = slices[0][:]  # [3*H]
        bq = b[rank * hd : (rank + 1) * hd]
        bk = b[H + rank * hd : H + (rank + 1) * hd]
        bv = b[2 * H + rank * hd : 2 * H + (rank + 1) * hd]
        return torch.cat([bq, bk, bv], dim=0)  # [3*hd]

    return SafetensorsWeightLoader(transform=transform)
