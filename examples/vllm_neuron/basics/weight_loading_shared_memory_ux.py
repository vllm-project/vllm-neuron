# SPDX-License-Identifier: Apache-2.0
"""
Sample integration of multi-process weight loading, including
padding + fusing and sharding.

Uses shared memory to speed up weight loading.
"""

import argparse
import json
from multiprocessing import shared_memory
import os
import struct
import tempfile
import threading
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import vllm_neuron  # noqa: F401
from vllm_neuron.nn import ColumnParallelLinear as CPL
from vllm_neuron.nn import RowParallelLinear as RPL
from vllm_neuron.utils.sharding import get_shard_config, ShardConfig


class SharedMemoryManager:
    """Shared memory manager for tensors"""

    def __init__(self):
        # Store all tensor names created by this rank
        # so we can cleanup the shared memory after execution ends
        self.shm_objects = []
        self.shm_objects_read = []

    def put_tensor(self, tensor, name):
        """Puts a tensor into shared memory

        Uses the following encoding:
        1. Uint64 indicating num bytes of JSON
        2. JSON containing dtype and shape
        3. Tensor bytes
        """
        try:
            # Prepare tensor metadata JSON
            metadata = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            metadata_json = json.dumps(metadata).encode("utf-8")
            metadata_length = len(metadata_json)

            # Calculate total size: uint64 + JSON + tensor data
            total_size = 8 + metadata_length + tensor.numel() * tensor.element_size()

            # Create shared memory
            shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
            self.shm_objects.append(shm)

            # 1. Write metadata length as uint64 (8 bytes, little-endian)
            struct.pack_into("<Q", shm.buf, 0, metadata_length)

            # 2. Write JSON metadata
            shm.buf[8 : 8 + metadata_length] = metadata_json

            # 3. Write tensor data
            offset = 8 + metadata_length
            shm_tensor = torch.frombuffer(shm.buf[offset:], dtype=tensor.dtype)
            shm_tensor[:] = tensor.flatten()

            return shm
        except Exception as e:
            print(f"Error creating shared memory: {name} {e}")

    def get_tensor(self, name):
        try:
            # This is a way to avoid incorrect shared memory resource leak warnings in
            # SharedMemory in older versions of Python (<=3.12)
            # See https://github.com/python/cpython/issues/82300 for context
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                shm = shared_memory.SharedMemory(name=name, create=False)
            self.shm_objects_read.append(shm)

            # 1. Read metadata length (uint64, 8 bytes, little-endian)
            metadata_length = struct.unpack_from("<Q", shm.buf, 0)[0]

            # 2. Read and parse JSON metadata
            metadata_json = bytes(shm.buf[8 : 8 + metadata_length])
            metadata = json.loads(metadata_json.decode("utf-8"))

            # 3. Extract dtype and shape from metadata
            dtype_str = metadata["dtype"]
            shape = tuple(metadata["shape"])

            # Convert dtype string (e.g., "torch.float32") to torch dtype
            dtype = getattr(torch, dtype_str.replace("torch.", ""))

            # 4. Read tensor data from the correct offset
            offset = 8 + metadata_length
            tensor = torch.frombuffer(shm.buf[offset:], dtype=dtype)
            tensor = tensor.reshape(shape)

            return tensor
        except Exception as e:
            print(f"Error reading tensor from shared memory: {name} {e}")

    def cleanup(self, barrier):
        for shm in self.shm_objects_read:
            shm.close()

        barrier.wait()

        for shm in self.shm_objects:
            shm.unlink()


# We define an MLP without any fused layers for creating a checkpoint
class MLP(nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.up_proj = CPL(1024, 2048, tp_size=world_size, bias=False)
        self.gate_proj = CPL(1024, 2048, tp_size=world_size, bias=False)
        self.down_proj = RPL(2048, 1024, tp_size=world_size, bias=False)

    def forward(self, x):
        return x


# We define an MLP with a fused projection layer to demonstrate loading/sharding process
class FusedMLP(nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.fused_up_gate_proj = CPL(1024, 4096, tp_size=world_size, bias=False)
        self.down_proj = RPL(2048, 1024, tp_size=world_size, bias=False)

    def forward(self, x):
        return x

    def load_rank_sharded_checkpoint(
        self, rank, shared_memory_manager: SharedMemoryManager
    ):
        """
        Depending on the checkpoint and modeling code implementation, we can implement optimizations
        (e.g. pipelined loading from disk to host memory and host memory -> HBM, etc.)
        """
        rank_sharded_checkpoint = {}

        for name, param in self.named_parameters():
            shard_config: ShardConfig = get_shard_config(param)

            # Get appropriate shard corresponding to rank
            if "fused_up_gate" in name:
                # Example of fusing weights
                up_proj = shared_memory_manager.get_tensor("up_proj.weight")
                gate_proj = shared_memory_manager.get_tensor("gate_proj.weight")
                tensor = shard_config._apply_fused_sharding_tensor(
                    [up_proj, gate_proj], rank
                )
            else:
                tensor = shard_config._apply_sharding_tensor(
                    shared_memory_manager.get_tensor(name), rank
                )

            # Apply padding if needed
            tensor = shard_config.apply_padding(tensor)

            print(f"[Rank {rank}] {name} {tensor.shape=}")

            # Clone the shard of the tensor from shared memory to process memory to avoid issues
            # when closing shared memory
            rank_sharded_checkpoint[name] = tensor.clone()

        return rank_sharded_checkpoint


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


def populate_shared_memory(
    rank, world_size, shared_memory_manager: SharedMemoryManager, model_dir
):
    print(f"[Rank {rank}] Populating shared memory...", flush=True)

    # Get mapping from weights -> safetensor files
    with open(
        os.path.join(model_dir, "model.safetensors.index.json"), "r"
    ) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    # Read portion of weight_map (each rank reads a portion)
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    rank_weight_names = sorted(weight_map.keys())[
        rank * weights_per_rank : (rank + 1) * weights_per_rank
    ]
    open_safetensor_files: dict = {}
    for weight_name in rank_weight_names:
        safetensors_file_with_weight = weight_map[weight_name]

        # Note that we are just mmapping the files, not actually reading through them
        if safetensors_file_with_weight not in open_safetensor_files:
            open_safetensor_files[safetensors_file_with_weight] = safe_open(
                os.path.join(model_dir, safetensors_file_with_weight),
                framework="pt",
                device="cpu",
            )

        tensor = open_safetensor_files[safetensors_file_with_weight].get_slice(
            weight_name
        )[:]
        shared_memory_manager.put_tensor(tensor, weight_name)


def worker_process(rank, world_size, barrier, model_dir):
    """Worker process function that runs distributed weight loading from host to device, including any transformations

    Each worker process:
    1. Initialize runtime
    2. Populates OS page cache by reading a portion of the model weights from disk
    3. Loads rank-specific portion of each tensor to user memory + processes as needed (padding, fusing, etc.)
    4. Sends tensors from user memory -> device HBM

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        barrier: Barrier that can be used for synchronization between workers
        model_dir: Path to model directory containing safetensors files
    """
    print(f"[Rank {rank}] Process started", flush=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = FusedMLP(world_size=world_size)
    shared_memory_manager = SharedMemoryManager()

    # 1. Start initializing runtime in the background
    runtime_thread = threading.Thread(target=initialize_runtime, args=(rank,))
    runtime_thread.start()

    # 2. Load model into shared memory
    populate_shared_memory(rank, world_size, shared_memory_manager, model_dir)

    # 3. Wait for shared memory to be filled up (happens in parallel across workers)
    barrier.wait()

    # 4. Load portion of each weight tensor that is needed by rank, and do any processing (weight fusing, padding, etc.)
    sharded_checkpoint: dict[str, torch.Tensor] = model.load_rank_sharded_checkpoint(
        rank, shared_memory_manager
    )

    # 5. Store tensors as part of model
    model.load_state_dict(sharded_checkpoint)

    # 6. Send tensors from host -> device
    model.to("neuron:0")

    # 7. Close runtime
    close_runtime()

    # 8. Cleanup shared memory
    shared_memory_manager.cleanup(barrier)

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

        barrier = mp.Barrier(args.world_size)

        print(
            f"\nRunning distributed execution with world_size={args.world_size}:",
            flush=True,
        )
        processes = []
        for rank in range(args.world_size):
            p = mp.Process(
                target=worker_process, args=(rank, args.world_size, barrier, tmpdirname)
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
