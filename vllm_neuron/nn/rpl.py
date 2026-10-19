# SPDX-License-Identifier: Apache-2.0
import math
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from torch.nn import init

from vllm_neuron.envs import is_native_backend

from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    SafetensorsWeightLoader,
)


class RowParallelLinear(nn.Module):
    """
    TODO: Add docstring
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        device=None,
        dtype=None,
        tp_group=None,
    ):
        super().__init__()

        # Read the backend at construction. See VocabDimShardedEmbedding.
        self._native = is_native_backend()

        # Use TP group if provided, otherwise fall back to world
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0

        self.in_features = in_features
        self.out_features = out_features
        # TODO: Add flag to enable padding
        assert in_features % self.tp_size == 0, (
            f"RPL layer in_features {out_features} is not evenly divisible by tp_size {self.tp_size}"
        )
        self.in_features_per_rank = in_features // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, self.in_features_per_rank), device=device, dtype=dtype
            )
        )
        nn.init.xavier_uniform_(self.weight)

        if bias:
            if self.tp_size > 1 and dtype != torch.float32:
                # TODO: With Torch XLA lowering, there seems to be an issue in extracting values out of
                # XLAShortType. Only bias of dtype Float32 seems to work. Fix inline_constants_to_hlo pass.
                raise NotImplementedError(
                    "RowParallelLinear only supports float32 bias"
                )

            self.bias = nn.Parameter(
                torch.empty(self.out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.input_is_parallel = input_is_parallel

        if not input_is_parallel:
            warnings.warn(
                "input_is_parallel=False will result in non-SPMD code",
                UserWarning,
                stacklevel=2,
            )

        self.reset_parameters()

        # Set sharding properties for weight (bias is not sharded in RowParallellinear as each rank needs full bias
        # for the computation)
        weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.in_features_per_rank,
            num_shards=self.tp_size,
        )
        set_weight_loader(self.weight, weight_loader)

    def reset_parameters(self) -> None:
        # Match expectation to Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

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

        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            # We only shard if the weight is not pre-sharded
            if not full_weight.shape == self.weight.data.shape:
                start_idx = self.tp_rank * self.in_features_per_rank
                end_idx = start_idx + self.in_features_per_rank
                state_dict[weight_key] = full_weight[:, start_idx:end_idx]

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
        if self.tp_size == 1:
            return F.linear(x, self.weight, self.bias)

        if not self.input_is_parallel:
            # Scatter input across ranks
            input_list = list(torch.chunk(x, self.tp_size, dim=-1))
            x = input_list[self.tp_rank]

        local_output = F.linear(x, self.weight, None)
        if self._native:
            # Use the string reduceOp. Dynamo cannot pickle a ReduceOp object
            # into a precompile package. A guard on it breaks the warm start.
            local_output = all_reduce(local_output, reduceOp="sum", group=self.tp_group)
        else:
            # XLA route: nothing pickles the ReduceOp. Keep the in-place op that
            # the XLA graph already lowers.
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM, group=self.tp_group)

        if self.bias is not None:
            local_output = torch.add(local_output, self.bias)

        return local_output
