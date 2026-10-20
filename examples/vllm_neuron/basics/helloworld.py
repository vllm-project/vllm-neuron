# SPDX-License-Identifier: Apache-2.0
"""
Basic vLLM Neuron compilation example.

# Run on Neuron hardware:
#   python examples/vllm_neuron/basics/helloworld.py

# Run in CPU mode (no hardware needed):
#   VLLM_NEURON_CPU_MODE=1 python examples/vllm_neuron/basics/helloworld.py
"""

import torch
from vllm_neuron import envs
from vllm_neuron.envs import get_compile_backend_name

import vllm_neuron as vllm_neuron  # noqa: F401

device = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randn(10, 10))

    def forward(self, x):
        return x @ self.weight


def main():
    model = Model()
    input = torch.randn(2, 10)

    # Run on CPU
    expected_output = model(input)

    model = model.to(device)
    input = input.to(device)

    # Compile and execute on neuron
    compiled_model = torch.compile(model, backend=get_compile_backend_name())
    output = compiled_model(input)

    # Validate results
    assert torch.allclose(expected_output, output.to("cpu"))
    print("Accuracy check passed!")


if __name__ == "__main__":
    main()
