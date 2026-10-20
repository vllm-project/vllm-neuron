# SPDX-License-Identifier: Apache-2.0
"""
Extensible test runner for distributed model testing.

This runner orchestrates distributed and non-distributed model testing
using the Model Builder pattern. Model implementations are kept in
separate modules for better organization and scalability.
"""

import argparse
import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from base import ModelBuilder
from registry import MODEL_REGISTRY, get_model_builder

import vllm_neuron as vllm_neuron  # noqa: F401
from vllm_neuron import envs

# ============================================================================
# Checkpoint and Input Management
# ============================================================================


def save_checkpoint(builder: ModelBuilder) -> tuple[dict, str]:
    """
    Create and save checkpoint using builder.

    Args:
        builder: ModelBuilder instance to create checkpoint from

    Returns:
        Tuple of (checkpoint dict, temporary file path)
    """
    checkpoint = builder.create_checkpoint()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        return checkpoint, f.name


def save_inputs(
    builder: ModelBuilder, num_steps: int = 3
) -> tuple[list[torch.Tensor], str]:
    """
    Create and save inputs using builder.

    Args:
        builder: ModelBuilder instance to create inputs from
        num_steps: Number of test input samples to generate

    Returns:
        Tuple of (inputs list, temporary file path)
    """
    inputs = builder.create_inputs(num_steps)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(inputs, f.name)
        return inputs, f.name


# ============================================================================
# Distributed Execution
# ============================================================================


def run_distributed(
    rank: int,
    world_size: int,
    builder: ModelBuilder,
    checkpoint_path: str,
    inputs_path: str,
    outputs_path: str,
    backend: str = "gloo",
):
    """
    Initialize the distributed environment and run model.

    Args:
        rank: Process rank in distributed setup
        world_size: Total number of processes
        builder: ModelBuilder instance to create model from
        checkpoint_path: Path to checkpoint file
        inputs_path: Path to inputs file
        outputs_path: Path to save outputs (rank 0 only)
        backend: Distributed backend to use (default: gloo)
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_VISIBLE_CORES"] = f"{rank}"
    os.environ["NCCL_DEBUG"] = "ERROR"

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build model using builder
    model = builder.build_model()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    # Load inputs
    inputs = torch.load(inputs_path)

    # Special handling for SubgroupCollective inputs (modify per-rank)
    if hasattr(builder, "modify_input_for_rank"):
        inputs = [builder.modify_input_for_rank(x, rank) for x in inputs]

    # Shard inputs BEFORE compilation if the builder supports it (e.g., for RPL)
    if hasattr(builder, "shard_input"):
        inputs = [builder.shard_input(x) for x in inputs]

    # Compile model (now sees correctly sharded inputs during tracing)
    model = torch.compile(model, backend=envs.get_compile_backend_name())

    # Execute and collect outputs
    outputs = []
    for i in range(len(inputs)):
        x = inputs[i]
        output = model(x)
        outputs.append(output)

    # Rank 0 saves outputs for comparison
    if rank == 0:
        torch.save(outputs, outputs_path)

    # Cleanup
    dist.destroy_process_group()


# ============================================================================
# Non-Distributed Execution (Reference)
# ============================================================================


def run_non_distributed(
    builder: ModelBuilder,
    checkpoint_path: str,
    inputs_path: str,
) -> list[torch.Tensor]:
    """
    Run non-distributed execution with reference model.

    Args:
        builder: ModelBuilder instance to create reference model from
        checkpoint_path: Path to checkpoint file
        inputs_path: Path to inputs file

    Returns:
        List of output tensors from reference model
    """
    checkpoint = torch.load(checkpoint_path)
    inputs = torch.load(inputs_path)

    # Build reference model using builder
    model = builder.build_reference()

    # Transform checkpoint keys for reference model if builder provides mapping
    if hasattr(builder, "_create_reference_checkpoint_mapping"):
        checkpoint = builder._create_reference_checkpoint_mapping(checkpoint)

    model.load_state_dict(checkpoint)

    # Execute and collect outputs
    outputs = []
    for i in range(len(inputs)):
        x = inputs[i]
        output = model(x)
        outputs.append(output)

    return outputs


# ============================================================================
# Output Comparison
# ============================================================================


def compare_outputs(
    reference_outputs: list[torch.Tensor],
    distributed_outputs: list[torch.Tensor],
    model_name: str,
    world_size: int,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """
    Compare reference and distributed outputs using np.allclose.

    Args:
        reference_outputs: Outputs from reference model
        distributed_outputs: Outputs from distributed model
        model_name: Name of the model being tested
        world_size: Number of distributed processes
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    """
    print("\n" + "=" * 70)
    print("TEST REPORT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"World Size: {world_size}")
    print(f"Test Steps: {len(reference_outputs)}")
    print()
    print("RESULTS:")

    num_steps = len(reference_outputs)
    passed = 0
    failed = 0

    for i in range(num_steps):
        ref_np = reference_outputs[i].detach().cpu().numpy()
        dist_np = distributed_outputs[i].detach().cpu().numpy()

        # Check if outputs match
        is_close = np.allclose(ref_np, dist_np, rtol=rtol, atol=atol)

        # Calculate max difference
        max_diff = np.max(np.abs(ref_np - dist_np))

        if is_close:
            print(f"Step {i}: ✓ PASS (max_diff={max_diff:.2e})")
            passed += 1
        else:
            print(f"Step {i}: ✗ FAIL (max_diff={max_diff:.2e})")
            failed += 1

    print()
    if failed == 0:
        print(f"Overall: ✓ ALL TESTS PASSED ({passed}/{num_steps} steps)")
    else:
        print(
            f"Overall: ✗ SOME TESTS FAILED ({passed}/{num_steps} passed, {failed}/{num_steps} failed)"
        )

    print(f"Tolerance: rtol={rtol}, atol={atol}")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Extensible distributed model test runner with Model Builder pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join(f"  - {name}: {builder.name}" for name, builder in MODEL_REGISTRY.items())}

Example usage:
  python runner.py --model cpl --world_size 2
  python runner.py --model cpl --world_size 4 --num_steps 5

To add a new model:
  1. Create a new model file (e.g., rpl.py)
  2. Create a new builder file (e.g., rpl_builder.py)
  3. Import and register the builder in registry.py
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cpl",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to test (default: cpl)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of processes for distributed execution (default: 2)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="Number of test steps to run (default: 3)",
    )
    args = parser.parse_args()

    # Get model builder
    builder = get_model_builder(args.model)

    # Validate world size if builder supports it
    if hasattr(builder, "validate_world_size"):
        builder.validate_world_size(args.world_size)

    # Create temporary files using builder
    checkpoint, checkpoint_path = save_checkpoint(builder)
    inputs, inputs_path = save_inputs(builder, args.num_steps)

    # Create temporary file for distributed outputs
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        distributed_outputs_path = f.name

    try:
        # Run non-distributed version (silent execution)
        reference_outputs = run_non_distributed(builder, checkpoint_path, inputs_path)

        # Run distributed version (silent execution)
        mp.spawn(
            run_distributed,
            args=(
                args.world_size,
                builder,
                checkpoint_path,
                inputs_path,
                distributed_outputs_path,
            ),
            nprocs=args.world_size,
            join=True,
        )

        # Load distributed outputs from rank 0
        distributed_outputs = torch.load(distributed_outputs_path)

        # Compare outputs with test report
        compare_outputs(
            reference_outputs,
            distributed_outputs,
            model_name=builder.name,
            world_size=args.world_size,
        )

    finally:
        # Cleanup temp files
        for path in [checkpoint_path, inputs_path, distributed_outputs_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
