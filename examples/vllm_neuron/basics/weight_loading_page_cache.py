# SPDX-License-Identifier: Apache-2.0
"""
Example to demonstrate an optimized multi-process weight loading using an OS page cache as a shared memory
between processes.

Essentially, we pipeline the following steps to maximize the overlap of data movement
1. Fill up OS page cache with weights
2. Read shard of each weight tensor
3. Write shard to device

Additionally, we start initializing Neuron Runtime in the background (as it is required before writing data to device)

For large models (e.g. Llama3 405B), filling up the OS page cache ends up taking up the majority of the time.

Before running, download weights to an NVMe-backed location and update MODEL_PATH as needed

Run using the following:
   > sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   > python weight_loading_page_cache.py
"""

import multiprocessing as mp
import time
import glob
import os
from safetensors import safe_open
import torch
import queue
import threading


WORLD_SIZE = 16
MODEL_PATH = "/path/to/llama-3.1-70b"
QUEUE_END_SIGNAL = "STOP"


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


def populate_page_cache(rank, world_size, cached_files_list):
    """Load all files sequentially into page cache

    Given a model checkpoint with 20 safetensor files, and a world size of 4,
    each rank will read 5 files into OS page cache by reading through the files.

    Subsequent accesses to the files will be able to read weights from the OS
    page cache (in OS memory) instead of the disk.

    We do this ahead of time to maximize disk IO bandwidth achieved when loading
    weights, and allow for faster access during sharding (where we have non-contiguous
    access patterns, which is extremely slow from disk). The added overhead of populating
    OS page cache ahead of time is recovered by the faster sharding times.
    """

    # Get all safetensors files that will be loaded
    safetensors_files: list = sorted(
        glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    )

    bytes_cached = 0
    cache_start = time.perf_counter()

    print(f"[Rank {rank}] Populating page cache", flush=True)

    # Each rank reads num_files // world_size files (code below sets up a roughly even distribution)
    # e.g. 10 files, 3 processes, ranks 0, 1, and 2 would read 4, 3, and 3 files
    base = len(safetensors_files) // world_size
    remainder = len(safetensors_files) % world_size
    start_idx = rank * base + min(rank, remainder)
    end_idx = start_idx + base + (1 if rank < remainder else 0)

    files_to_read = safetensors_files[start_idx:end_idx]
    print(f"[Rank {rank}] Reading: {files_to_read}", flush=True)

    # Read through file to load it into OS page cache
    chunk_size = 4 * 1024  # 4 KB
    for file_path in files_to_read:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                bytes_cached += chunk_size
        cached_files_list.append(file_path)

    cache_end = time.perf_counter()
    total_gb = bytes_cached / (1024**3)
    bandwidth = total_gb / (cache_end - cache_start)
    print(
        f"[Rank {rank}] Completed caching files: {total_gb:.1f} GB in {cache_end - cache_start:.1f}s ({bandwidth:.1f} GB/s)",
        flush=True,
    )


def disk_load(rank, world_size, cached_files_list, device_load_queue):
    """Load safetensors files from disk into CPU memory + process as needed"""

    # Get all safetensors files that will be loaded
    safetensors_files: list = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))

    total_bytes_disk_to_memory = 0
    disk_time = time.perf_counter()

    try:
        print(f"[Rank {rank}] Starting disk loading...", flush=True)
        processed_idx = 0

        while processed_idx < len(safetensors_files):
            # Check if next file is available
            if processed_idx < len(cached_files_list):
                file_path = cached_files_list[processed_idx]

                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Get slice does not actually create a tensor object in memory
                        slice_obj = f.get_slice(key)
                        if len(slice_obj.get_shape()) > 1:
                            sharded_dim = slice_obj.get_shape()[1] // world_size
                            start_idx = rank * sharded_dim
                            end_idx = (rank + 1) * sharded_dim
                            tensor = slice_obj[:, start_idx:end_idx]
                        else:
                            tensor = slice_obj[:]

                        # Since we use mmap, tensor's underlying data is only actually copied
                        # from OS page cache to user memory here when we use it (or clone it).
                        # Technically we don't need to do this, but it makes it easier to understand bottlenecks
                        # (one of which happens to be reading non-contiguous shards of tensors from memory)
                        tensor = tensor.clone()
                        # tensor = torch.nn.functional.pad(tensor, (0, 40))
                        # tensor = torch.repeat_interleave(tensor, 2)

                        device_load_queue.put(tensor)
                        total_bytes_disk_to_memory += (
                            tensor.numel() * tensor.element_size()
                        )

                processed_idx += 1
            else:
                # Wait for next file to be cached
                time.sleep(0.001)

        # Signal completion
        device_load_queue.put(QUEUE_END_SIGNAL)
        print(
            f"[Rank {rank}] Finished disk loading: {time.perf_counter() - disk_time}",
            flush=True,
        )

    except Exception as e:
        print(f"[Rank {rank}] Error in disk_load: {e}", flush=True)


def device_load(rank, device_load_queue):
    """Load tensors from CPU memory to device HBM"""
    try:
        print(f"[Rank {rank}] loading to device", flush=True)

        while True:
            cpu_tensor = device_load_queue.get()

            if cpu_tensor == QUEUE_END_SIGNAL:
                print(f"[Rank {rank}] Completed sending tensors to device", flush=True)
                break

            neuron_tensor = cpu_tensor.to("privateuseone:0")
    except Exception as e:
        print(f"[Rank {rank}] device load failed: {e}", flush=True)


def worker_process(rank, world_size, cached_files_list, barrier):
    print(f"[Rank {rank}] Started up process", flush=True)

    # Queue used for pipelining disk -> memory and memory -> HBM data transfer
    device_load_queue = queue.Queue()

    # 1. Start initializing runtime in the background
    runtime_thread = threading.Thread(target=initialize_runtime, args=(rank,))
    runtime_thread.start()

    # 2. Pipeline the execution of the following three steps that need to happen sequentially:
    #   a. Load each file into OS memory (page cache)
    #   b. Read rank-specific portion of tensor from OS memory (page cache) -> user memory and process as needed (weight fusing, padding, etc.)
    #   c. Send tensor from user memory -> HBM
    cache_thread = threading.Thread(
        target=populate_page_cache, args=(rank, world_size, cached_files_list)
    )
    disk_thread = threading.Thread(
        target=disk_load, args=(rank, world_size, cached_files_list, device_load_queue)
    )
    device_thread = threading.Thread(target=device_load, args=(rank, device_load_queue))

    cache_thread.start()
    disk_thread.start()
    device_thread.start()

    cache_thread.join()
    disk_thread.join()
    device_thread.join()

    # 3. Close runtime
    close_runtime(rank)

    # 4. Wait for all processes to complete
    barrier.wait()


def main():
    # Configs to avoid thread contention
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    mp.set_start_method("fork")

    cached_files_list = mp.Manager().list()
    barrier = mp.Barrier(WORLD_SIZE)

    processes = []
    process_start_time = time.perf_counter()
    for rank in range(WORLD_SIZE):
        p = mp.Process(
            target=worker_process, args=(rank, WORLD_SIZE, cached_files_list, barrier)
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
