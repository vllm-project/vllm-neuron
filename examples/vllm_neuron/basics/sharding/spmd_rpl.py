# SPDX-License-Identifier: Apache-2.0
"""
SPMD Row Parallel Linear layer implementation for distributed training.

This implementation uses Single Program Multiple Data (SPMD) approach where
all ranks have the bias parameter and add it during forward pass. When loading
checkpoints, the bias is uniformly divided by the TP degree to ensure correct
mathematical behavior after all-reduce.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class SPMDRowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
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
        self.in_features_per_rank = in_features // world_size
        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_rank))

        if bias:
            # All ranks have bias parameter (SPMD approach)
            self.bias = nn.Parameter(torch.randn(out_features))
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
            start_idx = self.rank * self.in_features_per_rank
            end_idx = start_idx + self.in_features_per_rank
            state_dict[weight_key] = full_weight[:, start_idx:end_idx]

        # All ranks load bias, but divide by TP degree for SPMD correctness
        if bias_key in state_dict and self.bias is not None:
            full_bias = state_dict[bias_key]
            # Divide bias by world_size so that after all-reduce, total bias equals original
            state_dict[bias_key] = full_bias / self.world_size

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
        # x is expected to be sharded along the last dimension (input features)
        # All ranks apply bias (SPMD approach)
        local_output = F.linear(x, self.weight, self.bias)

        if self.world_size == 1:
            return local_output

        # All-reduce to sum partial results from all ranks
        # Since all ranks added bias/world_size, the total bias contribution will be correct
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output
