# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.nn import ColumnParallelLinear as CPL
from vllm_neuron.nn import RowParallelLinear as RPL


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpl = CPL(128, 128)
        self.rpl = RPL(128, 32)

    def forward(self, x):
        x = self.cpl(x)
        return self.rpl(x)


def run_distributed(rank, world_size, checkpoint_path, inputs_path, backend="gloo"):
    """Initialize the distributed environment"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"
    os.environ["NCCL_DEBUG"] = "ERROR"

    # Initialize process group with Gloo backend
    # FIXME: Backend mismatch - torch.distributed uses gloo but compiled model uses Neuron collectives
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = Model()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model.to("neuron:0")

    # compile model
    model = torch.compile(model, backend=envs.get_compile_backend_name())

    # Load inputs
    inputs = torch.load(inputs_path)

    # execute
    for i in range(3):
        x = inputs[i].to("neuron:0")
        output = model(x).to("cpu")
        print(f"Rank {rank} step {i} output: {output}")

    # Cleanup
    dist.destroy_process_group()


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

    model = Model()

    # Save model checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        checkpoint_path = f.name

    # Save inputs
    inputs = [torch.randn(2, 128) for _ in range(3)]
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(inputs, f.name)
        inputs_path = f.name

    # Run model on CPU
    for i in range(3):
        x = inputs[i]
        output = model(x)
        print(f"Non-distributed step {i} output: {output}")

    try:
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
