# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the Qwen3-ASR audio encoder.

Custom sharding for the fused QKV weight/bias which stores Q, K, V
concatenated as [Q_all | K_all | V_all]. The default ColumnParallelLinear
loader takes a contiguous row slice, giving rank 0 all of Q and nothing
of K/V. The correct sharding interleaves heads across ranks.
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def audio_conv1_weight_loader() -> SafetensorsWeightLoader:
    """Pack the first audio convolution weight for matrix multiplication."""

    def transform(slices, rank):
        del rank
        weight = slices[0][:]
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        assert in_channels == 1
        assert kernel_height == 3
        assert kernel_width == 3
        return (
            weight[:, 0]
            .reshape(out_channels, kernel_height * kernel_width)
            .T.contiguous()
        )

    return SafetensorsWeightLoader(transform=transform)


def audio_qkv_weight_loader(
    num_heads_per_rank: int,
    head_dim: int,
) -> SafetensorsWeightLoader:
    """Weight loader that fuses separate q/k/v projections into the sharded QKV param.

    Args:
        num_heads_per_rank: Number of attention heads per TP rank.
        head_dim: Dimension per attention head.
    """
    hd = num_heads_per_rank * head_dim

    def transform(slices, rank):
        q, k, v = slices[0][:], slices[1][:], slices[2][:]
        q_r = q[rank * hd : (rank + 1) * hd]  # [hd, d_model]
        k_r = k[rank * hd : (rank + 1) * hd]
        v_r = v[rank * hd : (rank + 1) * hd]
        fused = torch.cat([q_r, k_r, v_r], dim=0)  # [3*hd, d_model]
        return fused.T.contiguous()  # [d_model, 3*hd]

    return SafetensorsWeightLoader(transform=transform)


def audio_qkv_bias_loader(
    num_heads_per_rank: int,
    head_dim: int,
) -> SafetensorsWeightLoader:
    """Weight loader for the QKV bias of audio attention.

    Args:
        num_heads_per_rank: Number of attention heads per TP rank.
        head_dim: Dimension per attention head.
    """
    hd = num_heads_per_rank * head_dim

    def transform(slices, rank):
        bq, bk, bv = slices[0][:], slices[1][:], slices[2][:]  # each [d_model]
        return torch.cat(
            [
                bq[rank * hd : (rank + 1) * hd],
                bk[rank * hd : (rank + 1) * hd],
                bv[rank * hd : (rank + 1) * hd],
            ],
            dim=0,
        )  # [3*hd]

    return SafetensorsWeightLoader(transform=transform)
