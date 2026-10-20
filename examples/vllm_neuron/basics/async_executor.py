# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import torch
import torch.nn as nn

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs
from vllm_neuron.executor import MPExecutor
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


def model_load(checkpoint_path: str):
    # Load and compile model
    model = Model()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to("neuron:0")
    model = torch.compile(model, backend=envs.get_compile_backend_name())
    return model


def main():
    world_size = 2

    # Create model and save checkpoint
    model = Model()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        checkpoint_path = f.name

    # Prepare inputs (same across all processes for SPMD)
    inputs = [torch.randn(2, 128) for _ in range(3)]

    # Compute non-distributed reference outputs
    print("Computing non-distributed reference outputs...")
    reference_outputs = []
    for _i, x in enumerate(inputs):
        output = model(x)
        reference_outputs.append(output)

    try:
        exec = MPExecutor(
            world_size=world_size,
            model_load=model_load,
            checkpoint_path=checkpoint_path,
        )

        for input in inputs:
            exec.dispatch(input)

        outputs = []
        for _request_id in range(len(inputs)):
            outputs.append(exec.collect())

        # Assert outputs are close
        print("\nComparing outputs:")
        for request_id in range(len(inputs)):
            for rank in range(world_size):
                assert torch.allclose(
                    reference_outputs[request_id], outputs[request_id][rank], atol=1e-5
                ), f"Output mismatch for rank {rank} request {request_id}"
                print(f"✓ Request {request_id} Rank {rank} outputs match")

        exec.shutdown()

    finally:
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
