# SPDX-License-Identifier: Apache-2.0
import torch

import vllm_neuron  # noqa: F401
from vllm_neuron import envs


def func(x: torch.Tensor) -> torch.Tensor:
    # Integer addition
    y = x + 10

    # Integer multiplication
    z = y * 2

    # Element-wise operations
    result = z + x

    return result


def main():
    input = torch.randint(0, 100, (2, 5), dtype=torch.int32)

    # CPU
    expected_result = func(input)

    # Compiled
    compiled_func = torch.compile(
        func, backend=envs.get_compile_backend_name(), options={"debug_hlo": True}
    )
    input = input.to("neuron:0")
    result = compiled_func(input).to("cpu")

    assert torch.allclose(result, expected_result), "Results do not match"


if __name__ == "__main__":
    main()
