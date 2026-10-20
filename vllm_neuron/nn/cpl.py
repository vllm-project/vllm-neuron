# SPDX-License-Identifier: Apache-2.0
import math
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributed._functional_collectives import all_gather_tensor
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)


class ColumnParallelLinear(nn.Module):
    """Linear layer that shards its output dim over a tensor-parallel group.

    Each rank holds ``out_features // tp_size`` output columns. With
    ``gather_output=True`` the forward all-gathers the per-rank outputs along
    the last dim. ``weight_layout`` selects the storage of the weight:
    ``"out_in"`` is the ``nn.Linear`` layout; ``"in_out"`` stores the weight
    transposed, so on Neuron every SBUF weight tile carries the contraction
    dim on the partition axis and the compiler does not transpose weight tiles
    on the PE before each matmul. ``tp_size=1`` makes the layer replicated:
    every rank holds the full weight and the forward adds no collective.

    Args:
        in_features: Input width.
        out_features: Output width before sharding.
        bias: Add a bias.
        gather_output: All-gather the output over the group. Has no effect
            for a replicated layer.
        device: Parameter device.
        dtype: Parameter dtype.
        tp_group: Process group to shard over. Defaults to the world group.
        weight_layout: ``"out_in"`` (``[out, in]``) or ``"in_out"``
            (``[in, out]``).
        tp_size: Expected group size. Pass ``1`` for a replicated layer.
            Any other value must match the size of ``tp_group``.

    Example:
        >>> fc = ColumnParallelLinear(12, 4, bias=False, weight_layout="in_out", tp_size=1)
        >>> fc.weight.shape
        torch.Size([12, 4])
        >>> fc(torch.zeros(2, 12)).shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        device=None,
        dtype=None,
        tp_group=None,
        weight_layout: Literal["out_in", "in_out"] = "out_in",
        tp_size: int | None = None,
    ):
        super().__init__()

        if tp_size == 1:
            # Replicated layer: every rank holds the full weight and the
            # forward adds no collective, so the group does not matter.
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0
        elif dist.is_initialized():
            # Use TP group if provided, otherwise fall back to world
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0
        if tp_size is not None and tp_size != self.tp_size:
            raise ValueError(
                f"tp_size {tp_size} does not match the size {self.tp_size} of "
                f"tp_group. Pass tp_size=1 for a replicated layer, or omit it."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.weight_layout = weight_layout
        # TODO: Add flag to enable padding
        assert out_features % self.tp_size == 0, (
            f"CPL layer out_features {out_features} is not evenly divisible by tp_size {self.tp_size}"
        )
        self.out_features_per_rank = out_features // self.tp_size
        if self.weight_layout == "out_in":
            self.weight = nn.Parameter(
                torch.empty(
                    (self.out_features_per_rank, in_features),
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_features, self.out_features_per_rank),
                    device=device,
                    dtype=dtype,
                )
            )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_rank, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.gather_output = gather_output

        # Set weight loaders for weight and bias parameters
        # weight: "in_out" stores [in, out], which is transposed relative to the
        # [out, in] checkpoint, so the output shard lives on dim 1.
        # bias: [out] in both layouts, so its loader is layout irrelevant.
        weight_shard_dim = 0 if self.weight_layout == "out_in" else 1
        is_transposed = self.weight_layout == "in_out"
        weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
            shard_dim=weight_shard_dim,
            shard_size=self.out_features_per_rank,
            num_shards=self.tp_size,
            is_storage_transposed=is_transposed,
        )

        set_weight_loader(self.weight, weight_loader)
        if bias:
            bias_weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
                shard_dim=0,
                shard_size=self.out_features_per_rank,
                num_shards=self.tp_size,
            )
            set_weight_loader(self.bias, bias_weight_loader)

    def reset_parameters(self) -> None:
        # Initialize the logical [out, in] to match nn.Linear
        linear_weight = self.weight if self.weight_layout == "out_in" else self.weight.T
        init.kaiming_uniform_(linear_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(linear_weight)
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
        bias_key = prefix + "bias"

        # PyTorch handles missing keys by adding them to missing_keys list
        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            # We only shard if the weight is not pre-sharded
            if not full_weight.shape == self.weight.data.shape:
                start_idx = self.tp_rank * self.out_features_per_rank
                end_idx = start_idx + self.out_features_per_rank

                if self.weight_layout == "out_in":
                    state_dict[weight_key] = full_weight[start_idx:end_idx]
                else:
                    state_dict[weight_key] = full_weight[:, start_idx:end_idx]

        # PyTorch handles missing keys by adding them to missing_keys list
        if bias_key in state_dict and self.bias is not None:
            full_bias = state_dict[bias_key]
            # We only shard if the weight is not pre-sharded
            if not full_weight.shape == self.bias.data.shape:
                start_idx = self.tp_rank * self.out_features_per_rank
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
        if self.weight_layout == "out_in":
            local_output = F.linear(x, self.weight, self.bias)
        else:
            local_output = torch.matmul(x, self.weight)
            if self.bias is not None:
                local_output = local_output + self.bias

        if self.tp_size == 1 or not self.gather_output:
            return local_output

        output = all_gather_tensor(
            local_output,
            gather_dim=1,
            group=self.tp_group,
        )

        return output
