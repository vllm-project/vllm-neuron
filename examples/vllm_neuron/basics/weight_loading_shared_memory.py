# SPDX-License-Identifier: Apache-2.0
"""
Example to demonstrate an optimized multi-process weight loading using shared memory to avoid loading a full
checkpoints into memory.

Essentially, we pipeline the following steps to maximize the overlap of data movement
1. Load weights into shared memory
2. Read shard of each weight tensor from shared memory
3. Write shard to device

Additionally, we start initializing Neuron Runtime in the background (as it is required before writing data to device)

For large models (e.g. Llama3 405B), creating the shard memory ends up taking up the majority of the time.

Before running, download weights to an NVMe-backed location and update MODEL_PATH as needed

Run using the following:
   > sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   > python weight_loading_shared_memory.py
"""

import torch.multiprocessing as mp
from multiprocessing import shared_memory
import time
import os
import json
from safetensors import safe_open
import torch
import struct
import queue
import threading
from unittest.mock import patch


WORLD_SIZE = 64
MODEL_PATH = "/path/to/llama-3.1-405b"
QUEUE_END_SIGNAL = "STOP"


# TODO: Investigate why some shared memory segments are getting cleaned up before explicit cleanup
# only when pipelining shared memory creation and consumption, resulting in errors such as:
# KeyError: '/model.layers.41.mlp.gate_proj.weight'
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


def initialize_runtime(rank):
    """Initializes runtime"""
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"

    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()

    print(f"[Rank {rank}] Finished initializing runtime...", flush=True)


def close_runtime(rank):
    """Close runtime"""
    runtime = torch.classes.neuron.Runtime()
    runtime.unsafe_close()

    print(f"[Rank {rank}] Finished closing runtime...", flush=True)


def populate_shared_memory(
    rank, world_size, shared_memory_manager: SharedMemoryManager, loaded_tensors_list
):
    print(f"[Rank {rank}] Populating shared memory...", flush=True)

    mem_time = time.perf_counter()

    # Get mapping from weights -> safetensor files
    with open(
        os.path.join(MODEL_PATH, "model.safetensors.index.json"), "r"
    ) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    # Read portion of weight_map (each rank reads a portion)
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    rank_weight_names = sorted(weight_map.keys())[
        rank * weights_per_rank : (rank + 1) * weights_per_rank
    ]
    open_safetensor_files: dict = {}
    total_bytes = 0
    for weight_name in rank_weight_names:
        safetensors_file_with_weight = weight_map[weight_name]

        # Note that we are just mmapping the files, not actually reading through them
        if safetensors_file_with_weight not in open_safetensor_files:
            open_safetensor_files[safetensors_file_with_weight] = safe_open(
                os.path.join(MODEL_PATH, safetensors_file_with_weight),
                framework="pt",
                device="cpu",
            )

        tensor = open_safetensor_files[safetensors_file_with_weight].get_slice(
            weight_name
        )[:]
        shared_memory_manager.put_tensor(tensor, weight_name)

        loaded_tensors_list.append(weight_name)
        total_bytes += tensor.numel() * tensor.element_size()

    print(
        f"[Rank {rank}] Populated {total_bytes / (1024**3)} GB shared memory in time: {time.perf_counter() - mem_time}",
        flush=True,
    )


def read_rank_sharded_weights(
    rank,
    world_size,
    shared_memory_manager: SharedMemoryManager,
    loaded_tensors_list,
    device_load_queue,
):
    """Load safetensors files from disk into CPU memory + process as needed"""
    print(f"[Rank {rank}] Reading rank sharded weights...", flush=True)

    read_time = time.perf_counter()

    # Get mapping from weights -> safetensor files
    with open(
        os.path.join(MODEL_PATH, "model.safetensors.index.json"), "r"
    ) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    # Read portion of each tensor needed for model
    processed_index = 0
    total_bytes = 0
    while processed_index < len(weight_map):
        if processed_index < len(loaded_tensors_list):
            weight_name = loaded_tensors_list[processed_index]
            full_tensor = shared_memory_manager.get_tensor(weight_name)

            if len(full_tensor.shape) > 1:
                # Only shard multi-dimensional weights (1d weights are probably normalization)
                sharded_dim = full_tensor.shape[1] // world_size
                start_idx = rank * sharded_dim
                end_idx = (rank + 1) * sharded_dim
                tensor = full_tensor[:, start_idx:end_idx]
            else:
                tensor = full_tensor[:]

            device_load_queue.put(tensor.clone())
            total_bytes += tensor.numel() * tensor.element_size()
            processed_index += 1
        else:
            time.sleep(0.01)

    device_load_queue.put(QUEUE_END_SIGNAL)
    print(
        f"[Rank {rank}] Read {total_bytes / (1024**3)} GB shared memory in time: {time.perf_counter() - read_time}",
        flush=True,
    )


def _load_to_device(rank, device_load_queue):
    print(f"[Rank {rank}] Loading weights to device", flush=True)

    total_bytes = 0
    load_time = time.perf_counter()

    try:
        while True:
            cpu_tensor = device_load_queue.get()

            if cpu_tensor == QUEUE_END_SIGNAL:
                break

            neuron_tensor = cpu_tensor.to("privateuseone:0")

            total_bytes += neuron_tensor.numel() * neuron_tensor.element_size()
    except Exception as e:
        print(f"[Rank {rank}] Exception during load to device: {e}", flush=True)
        return

    print(
        f"[Rank {rank}] Finished loading {total_bytes / (1024**3)} GB to device in time: {time.perf_counter() - load_time}",
        flush=True,
    )


def worker_process(rank, world_size, loaded_tensors_list, barrier):
    print(f"[Rank {rank}] Started up process", flush=True)

    shared_memory_manager = SharedMemoryManager()
    device_load_queue = queue.Queue()

    # 1. Start initializing runtime in the background
    runtime_thread = threading.Thread(target=initialize_runtime, args=(rank,))
    runtime_thread.start()

    # 2. Pipeline the following two steps that need to happen sequentially:
    #   a. Wait for full model checkpoint to be loaded into shared memory
    #   b. Reading rank-specific portion of tensor from OS memory (page cache) -> user memory
    #      and processing as needed (weight fusing, padding, etc.)
    #   c. Send tensor from user memory -> HBM
    shared_mem_thread = threading.Thread(
        target=populate_shared_memory,
        args=(rank, world_size, shared_memory_manager, loaded_tensors_list),
    )
    disk_thread = threading.Thread(
        target=read_rank_sharded_weights,
        args=(
            rank,
            world_size,
            shared_memory_manager,
            loaded_tensors_list,
            device_load_queue,
        ),
    )
    device_thread = threading.Thread(
        target=_load_to_device, args=(rank, device_load_queue)
    )

    shared_mem_thread.start()
    disk_thread.start()
    device_thread.start()

    shared_mem_thread.join()
    disk_thread.join()
    device_thread.join()

    # 3. Close runtime
    close_runtime(rank)

    # 4. Wait for all processes to complete using shared memory
    barrier.wait()

    # 5. Cleanup shared memory
    shared_memory_manager.cleanup(barrier)

    # 6. Wait for all processes to complete
    barrier.wait()


def main():
    # Configs to avoid thread contention
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    mp.set_start_method("fork")

    loaded_tensors_list = mp.Manager().list()
    barrier = mp.Barrier(WORLD_SIZE)

    processes = []
    process_start_time = time.perf_counter()
    for rank in range(WORLD_SIZE):
        p = mp.Process(
            target=worker_process, args=(rank, WORLD_SIZE, loaded_tensors_list, barrier)
        )
        p.start()
        processes.append(p)
    print(
        f"Total process start time: {time.perf_counter() - process_start_time}",
        flush=True,
    )

    start_time = time.perf_counter()
    for p in processes:
        p.join()
    print(f"Total time: {time.perf_counter() - start_time}", flush=True)


if __name__ == "__main__":
    main()
