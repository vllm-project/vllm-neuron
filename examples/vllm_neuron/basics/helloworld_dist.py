# SPDX-License-Identifier: Apache-2.0
"""
In this example where write a Column Parallel Linear layer from scratch.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.envs import get_compile_backend_name

device = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


class ColumnParallelLinear(nn.Module):
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
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // self.world_size
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

        if self.world_size == 1:
            return local_output

        gathered_output = torch.ops._c10d_functional.all_gather_into_tensor(
            local_output, self.world_size, "0"
        )
        return torch.cat(torch.chunk(gathered_output, self.world_size, dim=0), dim=1)


def run_distributed(rank, world_size, checkpoint_path, inputs_path, backend="gloo"):
    """Initialize the distributed environment"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"
    os.environ["NCCL_DEBUG"] = "ERROR"

    try:
        # Initialize process group with Gloo backend
        # FIXME: Backend mismatch - torch.distributed uses gloo but compiled model uses Neuron collectives
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = ColumnParallelLinear(128, 512)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)

        # compile model
        model = torch.compile(model, backend=get_compile_backend_name())
        model.to(device)

        # Load inputs
        inputs = torch.load(inputs_path)

        # execute
        for i in range(3):
            x = inputs[i].to(device)
            output = model(x).to("cpu")
            print(f"Rank {rank} step {i} output: {output}")

    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


def run_non_distributed(checkpoint_path, inputs_path):
    """Run non-distributed execution with regular Linear layer"""
    checkpoint = torch.load(checkpoint_path)
    inputs = torch.load(inputs_path)

    model = nn.Linear(128, 512)
    model.load_state_dict(checkpoint)

    for i in range(3):
        x = inputs[i]
        output = model(x)
        print(f"Non-distributed step {i} output: {output}")


def create_checkpoint():
    """Create a checkpoint compatible with ColumnParallelLinear"""
    checkpoint = {"weight": torch.randn(512, 128), "bias": torch.randn(512)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        return checkpoint, f.name


def create_inputs():
    """Create shared inputs for both distributed and non-distributed execution"""
    inputs = [torch.randn(2, 128) for _ in range(3)]
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(inputs, f.name)
        return inputs, f.name


def main():
    import argparse

    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(description="Distributed Hello World Example")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of processes for distributed execution",
    )
    args = parser.parse_args()

    # Create temporary files
    checkpoint, checkpoint_path = create_checkpoint()
    inputs, inputs_path = create_inputs()

    try:
        # Run non-distributed version
        print("Running non-distributed execution:")
        run_non_distributed(checkpoint_path, inputs_path)
        print(f"\nRunning distributed execution with world_size={args.world_size}:")

        # Spawn multiple processes
        mp.spawn(
            run_distributed,
            args=(args.world_size, checkpoint_path, inputs_path),
            nprocs=args.world_size,
            join=True,
        )
    finally:
        # Cleanup temp files
        os.remove(checkpoint_path)
        os.remove(inputs_path)


if __name__ == "__main__":
    main()
