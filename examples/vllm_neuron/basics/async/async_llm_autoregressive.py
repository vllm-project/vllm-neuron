# SPDX-License-Identifier: Apache-2.0
"""
Multiprocessing Async Execution with Double Buffering vs Sequential Demo

This example demonstrates async execution with double buffering on auto-regressive
workloads across multiple processes.

Usage:
  python async_llm_autoregressive.py --double-buffer [--world-size N]  # Use double buffering
  python async_llm_autoregressive.py --sequential [--world-size N]     # Use sequential execution
  Default world size: 2

Profiling:
  neuron-explorer view -d ./output --output-format perfetto
  view system_profile.pftrace on https://ui.perfetto.dev/

Features:
- Multiprocessing with distributed communication
- Auto-regressive inference
- Configurable execution mode per worker process
"""

import argparse
import os
import socket
import time
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.nn import ColumnParallelLinear as CPL
from vllm_neuron.nn import RowParallelLinear as RPL

WARM_UP_ROUNDS = 10
BENCHMARK_ROUNDS = 100


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=65536)
        self.cpl = CPL(65536, 65536, bias=False, gather_output=False)
        self.rpl = RPL(65536, 65536, bias=False, input_is_parallel=True)
        self.lm_head = torch.nn.Linear(65536, 100, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cpl(x)
        x = self.rpl(x)
        logits = self.lm_head(x)
        # Greedy sampling: return the token ID with highest logit
        return logits.argmax(dim=-1)


class AsyncExecutor:
    def __init__(self, world_size, model_load, use_double_buffer=True) -> None:
        self.world_size = world_size
        self.use_double_buffer = use_double_buffer
        self.processes = []
        self.input_queues = []
        self.output_queues = []

        self.master_port = self._find_free_port()

        for rank in range(world_size):
            input_queue = mp.Queue()  # type: ignore
            output_queue = mp.Queue()  # type: ignore
            p = mp.Process(
                target=worker_process,
                args=(
                    rank,
                    world_size,
                    input_queue,
                    output_queue,
                    model_load,
                    self.master_port,
                    use_double_buffer,
                ),
            )
            p.start()
            self.processes.append(p)
            self.input_queues.append(input_queue)
            self.output_queues.append(output_queue)

    def dispatch(self, input_ids):
        for rank in range(self.world_size):
            self.input_queues[rank].put(input_ids)

    def collect(self):
        output_token_ids = []
        for rank in range(self.world_size):
            output_token_ids.append(self.output_queues[rank].get(timeout=300.0))

        if output_token_ids and output_token_ids[0] == "ERROR":
            raise RuntimeError(f"Error in worker process: {output_token_ids[0][1]}")

        return output_token_ids

    def shutdown(self):
        self.dispatch("STOP")
        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return str(s.getsockname()[1])


def worker_process(
    rank, world_size, input_queue, output_queue, load, master_port, use_double_buffer
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"
    os.environ["NCCL_DEBUG"] = "ERROR"

    # Perf optimization. Avoids synchronizing neff execution.
    # Valid only for SPMD
    os.environ["NEURON_RT_DISABLE_EXECUTION_BARRIER"] = "1"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        model = load()
        i = 1
        done_warm_up = False

        if use_double_buffer:
            # Double buffering mode
            while True:
                try:
                    input = input_queue.get(timeout=1.0)
                    if input == "STOP":
                        break
                    current_input_tensor = torch.tensor([input]).to("neuron:0")
                    current_future = model(current_input_tensor)

                    while True:
                        # print(f"Process ID: {os.getpid()}, {i=}")
                        current_output = current_future
                        current_future = model(current_output)
                        output_val = current_output.to("cpu").item()
                        output_queue.put(output_val)
                        # print(f"Process ID: {os.getpid()}, {i=}, {output_val=}")
                        if not done_warm_up and i == WARM_UP_ROUNDS - 1:
                            done_warm_up = True
                            i = 1
                            break
                        if done_warm_up and i == BENCHMARK_ROUNDS - 1:
                            break
                        i += 1

                    # Handle the final result
                    final_output = current_future
                    output_queue.put(final_output.to("cpu").item())

                except Empty:
                    break
        else:
            # Sequential mode
            while True:
                try:
                    new_input = input_queue.get(timeout=1.0)
                    input_tensor = torch.tensor([new_input]).to("neuron:0")
                    output = model(input_tensor).to("cpu").item()
                    output_queue.put(output)

                    # print(f"Process ID: {os.getpid()}, {i=}, {new_input=}", )
                    if not done_warm_up and i == WARM_UP_ROUNDS:
                        done_warm_up = True
                        i = 1
                        continue
                    if done_warm_up and i == BENCHMARK_ROUNDS:
                        continue
                    if new_input == "STOP":
                        break

                    input_queue.put(output)
                    i += 1
                except Empty:
                    break

    except Exception as e:
        output_queue.put(("ERROR", str(e)))
    finally:
        dist.destroy_process_group()
        # force the neuron profiles to dump the profiles
        torch.classes.neuron.Runtime().unsafe_close()
        print("Ignore the logs, that say unexpected runtime state")


def model_load():
    model = Model()
    model = model.to("neuron:0")
    model = torch.compile(model, backend=envs.get_compile_backend_name())
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Multiprocessing Async vs Sequential Demo"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--double-buffer", action="store_true", help="Use double buffering"
    )
    group.add_argument(
        "--sequential", action="store_true", help="Use sequential execution"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=64,
        help="Number of worker processes (default: 64)",
    )

    args = parser.parse_args()
    use_double_buffer = args.double_buffer
    world_size = args.world_size

    exec = AsyncExecutor(
        world_size=world_size,
        model_load=model_load,
        use_double_buffer=use_double_buffer,
    )

    initial_input = 42

    # Warmup
    exec.dispatch(initial_input)
    for _ in range(WARM_UP_ROUNDS):
        exec.collect()

    # Measure inference time
    start_time = time.time()
    exec.dispatch(initial_input)

    results = []
    for _ in range(BENCHMARK_ROUNDS):
        results.extend(exec.collect())
    end_time = time.time()
    exec.shutdown()

    elapsed_time = end_time - start_time
    total_inferences = len(results)
    throughput = total_inferences / elapsed_time

    mode = "double-buffered" if use_double_buffer else "sequential"
    print(f"Multiprocessing {mode} (world_size={world_size})")
    print(f"Processed {total_inferences} inputs in {elapsed_time:.3f}s")
    print(f"Throughput: {throughput:.2f} inferences/sec")


if __name__ == "__main__":
    main()
