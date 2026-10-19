# SPDX-License-Identifier: Apache-2.0
"""
Double Buffering vs Sequential Demo

Usage:
  python async_vs_sequential.py --double-buffer  # Use double buffering
  python async_vs_sequential.py --sequential     # Use sequential execution

Profiling:
  NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=./output python async_vs_sequential.py --double-buffer
  neuron-explorer view -d ./output --output-format perfetto
  view system_profile.pftrace on https://ui.perfetto.dev/

Expected Results:
  Double-buffered: ~204 inferences/sec (0.490s for 100 inputs)
  Sequential: ~187 inferences/sec (0.533s for 100 inputs)
  Performance gain: ~9% improvement with double buffering

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
        self.lm_head = torch.nn.Linear(8192, 1, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cpl(x)
        x = self.rpl(x)
        return self.lm_head(x)


class Inference:
    def __init__(self, model, use_double_buffer=True):
        self.model = model
        self.use_double_buffer = use_double_buffer

    def run(self, inputs):
        if self.use_double_buffer:
            return self._run_double_buffered(inputs)
        else:
            return self._run_sequential(inputs)

    def _run_double_buffered(self, inputs):
        """Run inference with double buffering"""
        results = []

        if len(inputs) < 2:
            return self._run_sequential(inputs)

        input1 = torch.tensor([inputs[0]]).to("neuron:0")
        input2 = torch.tensor([inputs[1]]).to("neuron:0")

        current_output = self.model(input1)
        next_output = self.model(input2)

        for i in range(2, len(inputs)):
            results.append(current_output.to("cpu").item())
            next_input = torch.tensor([inputs[i]]).to("neuron:0")
            current_output = next_output
            next_output = self.model(next_input)

        results.append(current_output.to("cpu").item())
        results.append(next_output.to("cpu").item())
        return results

    def _run_sequential(self, inputs):
        """Run inference sequentially"""
        results = []
        for inp in inputs:
            input_tensor = torch.tensor([inp]).to("neuron:0")
            output = self.model(input_tensor)
            results.append(output.to("cpu").item())
        return results


def main():
    use_double_buffer = "--double-buffer" in sys.argv
    if not use_double_buffer and "--sequential" not in sys.argv:
        print("Usage: python async_vs_sequential.py [--double-buffer|--sequential]")
        sys.exit(1)

    model = Model().to("neuron:0")
    model = torch.compile(model, backend=envs.get_compile_backend_name())

    inference = Inference(model, use_double_buffer)

    inputs = list(range(100))

    # Warmup
    inference.run(inputs[:5])

    # Measure inference time
    start_time = time.time()
    inference.run(inputs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = len(inputs) / elapsed_time

    mode = "double-buffered" if use_double_buffer else "sequential"
    print(f"Mode: {mode}")
    print(f"Processed {len(inputs)} inputs in {elapsed_time:.3f}s")
    print(f"Throughput: {throughput:.2f} inferences/sec")


if __name__ == "__main__":
    main()
