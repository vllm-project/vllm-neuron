# SPDX-License-Identifier: Apache-2.0
"""
Manual Tensor Capture Example: Capture tensors from within model code.

Demonstrates using capture_tensor() to manually capture intermediate
tensors at specific points in a model's forward pass.

See also: docs/design/accuracy/tensor_capture_design.md

Usage:
    python run_tensor_capture_manual.py
"""

import torch
import torch.nn as nn

from vllm_neuron.accuracy import capture_tensor
from vllm_neuron.accuracy.tensor_capture import TensorRegistry


class ModelWithCapture(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        h = self.fc1(x)
        capture_tensor("after_fc1", h)

        h = torch.relu(h)
        capture_tensor("after_relu", h)

        return self.fc2(h)


def main():
    torch.manual_seed(42)
    TensorRegistry.reset_instance()
    registry = TensorRegistry.get_instance()
    registry.configure(enabled=True)

    model = ModelWithCapture()
    output = model(torch.randn(1, 64))

    captured = registry.get_all()
    print(f"Captured {len(captured)} tensors:")
    for name, tensor in captured:
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    print(f"\nOutput shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
