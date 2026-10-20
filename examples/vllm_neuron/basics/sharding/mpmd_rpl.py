# SPDX-License-Identifier: Apache-2.0
"""
MPMD Row Parallel Linear layer implementation for distributed training.

This implementation uses Multiple Program Multiple Data (MPMD) approach where
only rank 0 has and applies the bias parameter to avoid duplication during
all-reduce operations.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MPMDRowParallelLinear(nn.Module):
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
            # Only rank 0 has bias to avoid duplication during all-reduce
            if rank == 0:
                self.bias = nn.Parameter(torch.randn(out_features))
            else:
                self.register_parameter("bias", None)
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

        # Only rank 0 loads bias
        if bias_key in state_dict:
            if self.rank == 0 and self.bias is not None:
                # Keep the bias as-is for rank 0
                pass
            else:
                # Remove bias from state_dict for other ranks
                state_dict.pop(bias_key, None)

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
        # Apply bias only on rank 0 to avoid duplication
        bias = self.bias if self.rank == 0 else None
        local_output = F.linear(x, self.weight, bias)

        if self.world_size == 1:
            return local_output

        # All-reduce to sum partial results from all ranks
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output
