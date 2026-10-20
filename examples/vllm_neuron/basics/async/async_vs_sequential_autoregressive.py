# SPDX-License-Identifier: Apache-2.0
"""
Async Execution with Double Buffering vs Sequential Demo

This example demonstrates async execution with double buffering on auto-regressive
workloads.

Usage:
  python async_vs_sequential_autoregressive.py --double-buffer  # Use double buffering
  python async_vs_sequential_autoregressive.py --sequential     # Use sequential execution

Profiling:
  NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=./output python async_vs_sequential_autoregressive.py --double-buffer
  neuron-explorer view -d ./output --output-format perfetto
  view system_profile.pftrace on https://ui.perfetto.dev/

Expected Results (on Trn2):
  Double-buffered: ~387 inferences/sec (0.258s for 100 inputs)
  Sequential: ~340 inferences/sec (0.295s for 100 inputs)
  Performance gain: ~13.8% improvement with double buffering

"""

import sys
import time

import torch
import torch.nn as nn

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.nn import ColumnParallelLinear as CPL
from vllm_neuron.nn import RowParallelLinear as RPL


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=8192)
        self.cpl = CPL(8192, 8192, bias=False, gather_output=False)
        self.rpl = RPL(8192, 8192, bias=False, input_is_parallel=True)
        self.lm_head = torch.nn.Linear(8192, 100, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cpl(x)
        x = self.rpl(x)
        logits = self.lm_head(x)
        # Greedy sampling: return the token ID with highest logit
        return logits.argmax(dim=-1)


class Inference:
    def __init__(self, model, use_double_buffer=True):
        self.model = model
        self.use_double_buffer = use_double_buffer

    def run(self, initial_input, num_steps):
        """
        Run auto-regressive inference for num_steps.

        Args:
            initial_input: The starting input token
            num_steps: Number of inference steps to run
        """
        if self.use_double_buffer:
            return self._run_double_buffered(initial_input, num_steps)
        else:
            return self._run_sequential(initial_input, num_steps)

    def _run_double_buffered(self, initial_input, num_steps):
        """
        Run inference with double buffering, handling dependencies between steps.

        Key: Use output token directly as next input. Start next computation
        IMMEDIATELY, then do .to("cpu") while it runs to do post-processing.
        """
        results = []

        if num_steps < 2:
            return self._run_sequential(initial_input, num_steps)

        # Start first computation
        current_input_tensor = torch.tensor([initial_input]).to("neuron:0")
        current_future = self.model(current_input_tensor)

        for step in range(num_steps - 1):
            current_output = current_future

            # Start next computation IMMEDIATELY using output token directly
            # This is the key to double buffering - no blocking operations before this!
            current_future = self.model(current_output)

            # NOW do the blocking .to("cpu") operation while next step is running
            output_val = current_output.to("cpu").item()
            results.append(output_val)

        # Handle the final result
        final_output = current_future
        results.append(final_output.to("cpu").item())

        return results

    def _run_sequential(self, initial_input, num_steps):
        """
        Run inference sequentially with dependencies.
        Each step waits for the previous to complete before starting.
        The output token from step N becomes the input token for step N+1.
        """
        results = []
        current_input = initial_input

        for step in range(num_steps):
            # Run inference
            input_tensor = torch.tensor([current_input]).to("neuron:0")
            output = self.model(input_tensor)
            output_val = output.to("cpu").item()
            results.append(output_val)

            # Use the output token directly as the next input
            if step < num_steps - 1:  # Don't process output on final step
                current_input = output_val  # output is already a token ID

        return results


def main():
    use_double_buffer = "--double-buffer" in sys.argv
    if not use_double_buffer and "--sequential" not in sys.argv:
        print(
            "Usage: python async_vs_sequential_autoregressive.py [--double-buffer|--sequential]"
        )
        sys.exit(1)

    model = Model().to("neuron:0")
    model = torch.compile(model, backend=envs.get_compile_backend_name())

    inference = Inference(model, use_double_buffer)

    initial_input = 42  # Starting token
    num_steps = 100  # Number of dependent steps to run

    # Warmup with smaller number of steps
    inference.run(initial_input, 20)

    # Measure inference time
    start_time = time.time()
    results = inference.run(initial_input, num_steps)
    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = num_steps / elapsed_time

    mode = "double-buffered" if use_double_buffer else "sequential"
    print(f"Mode: {mode}")
    print(f"Processed {num_steps} inputs in {elapsed_time:.3f}s")
    print(f"Throughput: {throughput:.2f} inferences/sec")
    print(f"Initial input: {initial_input}")
    print(f"Final output: {results[-1]:.6f}")
    print(f"Chain sample: {results[:5]} -> ... -> {results[-5:]}")


if __name__ == "__main__":
    main()
