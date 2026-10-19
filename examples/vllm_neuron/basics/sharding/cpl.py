# SPDX-License-Identifier: Apache-2.0
"""
Column Parallel Linear layer implementation for distributed training.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.out_features_per_rank = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.out_features_per_rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"

        # PyTorch handles missing keys by adding them to missing_keys list
        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            start_idx = self.rank * self.out_features_per_rank
            end_idx = start_idx + self.out_features_per_rank
            state_dict[weight_key] = full_weight[start_idx:end_idx]

        # PyTorch handles missing keys by adding them to missing_keys list
        if bias_key in state_dict and self.bias is not None:
            full_bias = state_dict[bias_key]
            start_idx = self.rank * self.out_features_per_rank
            end_idx = start_idx + self.out_features_per_rank
            state_dict[bias_key] = full_bias[start_idx:end_idx]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        local_output = F.linear(x, self.weight, self.bias)

        if self.world_size == 1 or not self.gather_output:
            return local_output

        gathered_outputs = [
            torch.zeros_like(local_output) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_outputs, local_output)
        return torch.cat(gathered_outputs, dim=-1)
