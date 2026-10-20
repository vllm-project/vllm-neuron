# SPDX-License-Identifier: Apache-2.0
"""
Sample integration of multi-process weight loading, show casing
how to speed up the weight loading process using pipelining
"""

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import threading

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file

import vllm_neuron  # noqa: F401
from vllm_neuron.nn import ColumnParallelLinear as CPL
from vllm_neuron.nn import RowParallelLinear as RPL
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.sharding import get_shard_config, ShardConfig


# We define an MLP without any fused layers for creating a checkpoint
class MLP(nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.up_proj = CPL(1024, 2048, tp_size=world_size, bias=False)
        self.gate_proj = CPL(1024, 2048, tp_size=world_size, bias=False)
        self.down_proj = RPL(2048, 1024, tp_size=world_size, bias=False)

    def forward(self, x):
        return x

    def load_rank_sharded_checkpoint(
        self, checkpoint_path: str
    ) -> dict[str, torch.Tensor]:
        """
        Loads checkpoint for provided rank

        Args:
            checkpoint_path: Directory containing the checkpoint files
        """

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        param_dict = dict(self.named_parameters())

        def process_tensor(key: str, slice_obj) -> tuple[str, torch.Tensor]:
            shard_config: ShardConfig = get_shard_config(param_dict[key])
            tensor = shard_config.apply_sharding(slice_obj, rank)
            tensor = shard_config.apply_padding(tensor)
            return key, tensor

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            rank, world_size, process_tensor
        ).state_dict

        self.load_state_dict(rank_sharded_checkpoint, strict=False, assign=True)


def initialize_runtime(rank):
    """Initializes runtime"""
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"
    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()


def close_runtime():
    """Closes runtime"""
    runtime = torch.classes.neuron.Runtime()
    runtime.unsafe_close()


def worker_process(rank, world_size, model_dir):
    """Worker process function that runs distributed weight loading from host to device, including any transformations

    Each worker process:
    1. Initialize runtime
    2. Populates OS page cache by reading a portion of the model weights from disk
    3. Loads rank-specific portion of each tensor to user memory + processes as needed (padding, fusing, etc.)
    4. Sends tensors from user memory -> device HBM

    Args:
        rank (int): Process rank (0 to world_size-1)
        world_size (int): Total number of processes
        model (Model): Model that we are loading rank-sharded checkpoint to
        model_dir (string): Path to model directory containing safetensors files
    """
    print(f"[Rank {rank}] Process started", flush=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = MLP(world_size=world_size)

    # 1. Start initializing runtime in the background
    runtime_thread = threading.Thread(target=initialize_runtime, args=(rank,))
    runtime_thread.start()

    # 2. Load portion of each weight tensor that is needed by rank, and do any processing (weight fusing, padding, etc.)
    model.load_rank_sharded_checkpoint(model_dir)

    # 3. Close runtime
    close_runtime()

    print(f"[Rank {rank}] Finished loading weights to device", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Distributed weight loading example")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of processes for distributed execution",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdirname:
        _create_multi_file_safetensors_checkpoint(tmpdirname)

        # Configs to avoid thread contention
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        mp.set_start_method("fork")

        print(
            f"\nRunning distributed execution with world_size={args.world_size}:",
            flush=True,
        )
        processes = []
        for rank in range(args.world_size):
            p = mp.Process(
                target=worker_process, args=(rank, args.world_size, tmpdirname)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def _create_multi_file_safetensors_checkpoint(dirname):
    """Creates a multi-file safetensors checkpoint for model with dummy weights"""

    # Create a model without any TP sharding for creating checkpoint
    model = MLP(world_size=1)

    MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
    current_size_bytes = 0
    file_num = 1
    current_dict = {}
    weight_map = {}

    # Save dummy model checkpoint as multiple safetensors files
    for name, tensor in model.state_dict().items():
        current_dict[name] = tensor
        tensor_size_bytes = tensor.element_size() * tensor.nelement()
        current_size_bytes += tensor_size_bytes

        if current_size_bytes > MAX_FILE_SIZE_BYTES:
            file_name = f"model-{file_num:05d}.safetensors"
            save_file(current_dict, os.path.join(dirname, file_name))

            for key in current_dict.keys():
                weight_map[key] = file_name

            file_num += 1
            current_dict = {}
            current_size_bytes = 0

    # Save remaining tensors
    if current_dict:
        file_name = f"model-{file_num:05d}.safetensors"
        save_file(current_dict, os.path.join(dirname, file_name))
        for key in current_dict.keys():
            weight_map[key] = file_name

    # Create safetensors index file
    with open(os.path.join(dirname, "model.safetensors.index.json"), "w") as f:
        index_map = {"weight_map": weight_map}
        json.dump(index_map, f, indent=2)


if __name__ == "__main__":
    main()
