# SPDX-License-Identifier: Apache-2.0
"""Tensor comparison utilities for accuracy debugging."""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from vllm_neuron.accuracy.tensor_alignment_utils import (
    align_and_truncate_hidden,
    get_seq_dim_size,
    slice_token,
)
from vllm_neuron.accuracy.tensor_io import CapturedForwardPass
from vllm_neuron.accuracy.utils import natural_sort_key

logger = logging.getLogger(__name__)

# Reconstructs a single comparable tensor from per-rank captured tensors.
# Args: rank_tensors (sorted by rank), module_name, phase, positions
# phase is "prefill" or "decode" (extensible for future phases like "speculative").
# The function is responsible for padding stripping and SP gather.
ReconstructionFn = Callable[[List[torch.Tensor], str, str, List[int]], torch.Tensor]

# Aligns three tensors for comparison: fn(baseline, expected, actual) -> (b, e, a, ok)
AlignmentFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool],
]


class _Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    return "color" in term or term in ["xterm", "xterm-256color", "screen", "linux"]


def _colorize(text: str, color: str, colorize: bool = True) -> str:
    """Apply color to text if supported."""
    if not colorize or not _supports_color():
        return text
    code = getattr(_Colors, color.upper(), "")
    return f"{code}{text}{_Colors.RESET}" if code else text


# Default dynamic threshold config (similar to logit_validation)
@dataclass
class DynamicThresholdConfig:
    """Configuration for dynamic threshold pass/fail in three-way comparison.

    A comparison passes if tgt_error < multiplier * base_error for both metrics.
    """

    linf_multiplier: float = 1.5  # tgt_linf < N * base_linf
    l2_multiplier: float = 1.5  # tgt_l2 < N * base_l2


DEFAULT_DYNAMIC_THRESHOLD_CONFIG = DynamicThresholdConfig()


@dataclass
class ComparisonResult:
    """Result of comparing two tensors."""

    name: str
    shape1: Tuple[int, ...]
    shape2: Tuple[int, ...]
    linf_rel: float  # Relative L-infinity error
    l2_rel: float  # Relative L2 error
    max_abs: float  # Maximum absolute difference
    is_match: bool  # Shapes match (after alignment)


@dataclass
class ThreeWayComparisonResult:
    """Result of three-way tensor comparison (baseline vs expected vs actual).

    Three-way comparison isolates target-specific errors from dtype-inherent errors:
    - baseline: FP32 reference (ground truth)
    - expected: same-dtype baseline (e.g., BF16 on CPU)
    - actual: target (e.g., BF16 on Neuron)
    """

    name: str
    base_linf: float  # L-inf error: expected vs baseline
    tgt_linf: float  # L-inf error: actual vs baseline
    base_l2: float  # L2 error: expected vs baseline
    tgt_l2: float  # L2 error: actual vs baseline
    linf_ratio: float  # tgt_linf / base_linf (compare against multiplier)
    l2_ratio: float  # tgt_l2 / base_l2 (compare against multiplier)
    bc: float  # Bhattacharyya Coefficient (error distribution similarity)
    passed: bool  # Whether dynamic thresholds passed
    shape_match: bool
    # Raw errors for aggregate BC computation
    base_errors: Optional[np.ndarray] = None
    tgt_errors: Optional[np.ndarray] = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple prompts."""

    mean_linf_ratio: float
    mean_l2_ratio: float
    max_linf_ratio: float
    max_l2_ratio: float
    aggregate_bc: float  # BC computed from pooled errors across all prompts
    n_prompts: int
    n_modules: int


# Structured return types for compare functions.
# Maps prompt key (e.g. "prompt_0") -> step index -> list of results.
TwoWayResults = Dict[str, Dict[int, List[ComparisonResult]]]
ThreeWayResults = Dict[str, Dict[int, List[ThreeWayComparisonResult]]]


def align_shapes(
    t1: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Align tensor shapes for comparison.

    Handles common mismatches:
    - [1, seq, hidden] vs [seq, hidden] (batch dim)
    - [seq1, hidden] vs [seq2, hidden] (different seq lengths - use min)

    Returns:
        (aligned_t1, aligned_t2, success)

    Example:
        >>> t1 = torch.randn(1, 10, 64)
        >>> t2 = torch.randn(10, 64)
        >>> a1, a2, ok = align_shapes(t1, t2)
        >>> assert a1.shape == a2.shape == (10, 64)
    """
    # TODO: Support custom alignment hooks for edge cases (different sharding,
    # transposed outputs, KV cache layouts). For now, users can preprocess
    # tensors before calling compare functions.
    if t1.shape == t2.shape:
        return t1, t2, True

    # Remove batch dim of 1
    if t1.dim() == 3 and t1.shape[0] == 1 and t2.dim() == 2:
        t1 = t1.squeeze(0)
    elif t2.dim() == 3 and t2.shape[0] == 1 and t1.dim() == 2:
        t2 = t2.squeeze(0)

    # Align 2D tensors: trim seq (dim 0) to min; hidden (dim 1) must match
    if t1.dim() == t2.dim() == 2 and t1.shape[1] == t2.shape[1]:
        seq_len = min(t1.shape[0], t2.shape[0])
        t1 = t1[:seq_len]
        t2 = t2[:seq_len]

    return t1, t2, t1.shape == t2.shape


def _compute_bc(errors1: np.ndarray, errors2: np.ndarray, num_bins: int = 100) -> float:
    """Bhattacharyya Coefficient between two error distributions.

    BC=1.0 means identical distributions, BC=0.0 means no overlap.
    """
    if len(errors1) == 0 or len(errors2) == 0:
        return 0.0
    min_val = min(errors1.min(), errors2.min())
    max_val = max(errors1.max(), errors2.max())
    if min_val == max_val:
        return 1.0
    bins = np.linspace(min_val, max_val, num_bins + 1)
    h1, _ = np.histogram(errors1, bins=bins)
    h2, _ = np.histogram(errors2, bins=bins)
    p1 = h1 / (h1.sum() + 1e-10)
    p2 = h2 / (h2.sum() + 1e-10)
    return float(np.clip(np.sum(np.sqrt(p1 * p2)), 0.0, 1.0))


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    name: str = "tensor",
    align: bool = True,
) -> ComparisonResult:
    """
    Compare two tensors and compute error metrics.

    Args:
        t1: First tensor (reference)
        t2: Second tensor (test)
        name: Name for the comparison result
        align: Whether to attempt shape alignment

    Returns:
        ComparisonResult with error metrics

    Example:
        >>> result = compare_tensors(ref_tensor, test_tensor, name="layer.0")
        >>> print(f"L-inf: {result.linf_rel:.6f}, L2: {result.l2_rel:.6f}")
    """
    shape1, shape2 = t1.shape, t2.shape

    if align:
        t1, t2, is_match = align_shapes(t1.float(), t2.float())
    else:
        t1, t2 = t1.float(), t2.float()
        is_match = t1.shape == t2.shape

    if not is_match:
        return ComparisonResult(
            name=name,
            shape1=shape1,
            shape2=shape2,
            linf_rel=float("inf"),
            l2_rel=float("inf"),
            max_abs=float("inf"),
            is_match=False,
        )

    diff = t1 - t2
    max_abs = diff.abs().max().item()
    t1_max = t1.abs().max().item()
    t1_norm = torch.norm(t1).item()

    linf_rel = max_abs / t1_max if t1_max > 0 else max_abs
    l2_rel = torch.norm(diff).item() / t1_norm if t1_norm > 0 else 0

    return ComparisonResult(
        name=name,
        shape1=shape1,
        shape2=shape2,
        linf_rel=linf_rel,
        l2_rel=l2_rel,
        max_abs=max_abs,
        is_match=True,
    )


def compare_tensors_three_way(
    baseline: torch.Tensor,
    expected: torch.Tensor,
    actual: torch.Tensor,
    name: str = "tensor",
    dynamic_threshold_config: DynamicThresholdConfig = None,
    align: bool = True,
) -> ThreeWayComparisonResult:
    """
    Three-way tensor comparison for isolating target-specific errors.

    Compares expected and actual against a baseline (FP32 reference).
    Uses dynamic thresholds: tgt_error < multiplier * base_error

    Args:
        baseline: FP32 reference tensor (ground truth)
        expected: Same-dtype baseline tensor (e.g., BF16 on CPU)
        actual: Target tensor (e.g., BF16 on Neuron)
        name: Name for the comparison result
        dynamic_threshold_config: Threshold config for pass/fail determination
        align: Whether to attempt shape alignment

    Returns:
        ThreeWayComparisonResult with base/tgt errors and pass/fail

    Example:
        >>> result = compare_tensors_three_way(fp32_ref, bf16_cpu, bf16_neuron, name="layer.0")
        >>> print(f"Passed: {result.passed}, L-inf ratio: {result.linf_ratio:.4f}")
    """
    if dynamic_threshold_config is None:
        dynamic_threshold_config = DEFAULT_DYNAMIC_THRESHOLD_CONFIG

    # Align shapes
    if align:
        baseline, expected, match1 = align_shapes(baseline.float(), expected.float())
        baseline, actual, match2 = align_shapes(baseline, actual.float())
        shape_match = match1 and match2
    else:
        baseline, expected, actual = baseline.float(), expected.float(), actual.float()
        shape_match = baseline.shape == expected.shape == actual.shape

    if not shape_match:
        return ThreeWayComparisonResult(
            name=name,
            base_linf=float("inf"),
            tgt_linf=float("inf"),
            base_l2=float("inf"),
            tgt_l2=float("inf"),
            linf_ratio=float("inf"),
            l2_ratio=float("inf"),
            bc=0.0,
            passed=False,
            shape_match=False,
        )

    # Compute errors vs baseline
    base_diff = expected - baseline
    tgt_diff = actual - baseline

    baseline_max = baseline.abs().max().item()
    baseline_norm = torch.norm(baseline).item()

    base_linf = base_diff.abs().max().item() / baseline_max if baseline_max > 0 else 0
    tgt_linf = tgt_diff.abs().max().item() / baseline_max if baseline_max > 0 else 0
    base_l2 = torch.norm(base_diff).item() / baseline_norm if baseline_norm > 0 else 0
    tgt_l2 = torch.norm(tgt_diff).item() / baseline_norm if baseline_norm > 0 else 0

    # Bhattacharyya Coefficient on error distributions
    base_errors = base_diff.abs().flatten().numpy()
    tgt_errors = tgt_diff.abs().flatten().numpy()
    bc = _compute_bc(base_errors, tgt_errors)

    # Dynamic threshold check
    linf_mult = dynamic_threshold_config.linf_multiplier
    l2_mult = dynamic_threshold_config.l2_multiplier

    # Compute ratios
    linf_ratio = (
        tgt_linf / base_linf
        if base_linf > 0
        else (0.0 if tgt_linf == 0 else float("inf"))
    )
    l2_ratio = (
        tgt_l2 / base_l2 if base_l2 > 0 else (0.0 if tgt_l2 == 0 else float("inf"))
    )

    # Pass if ratio < multiplier
    passed = linf_ratio < linf_mult and l2_ratio < l2_mult

    return ThreeWayComparisonResult(
        name=name,
        base_linf=base_linf,
        tgt_linf=tgt_linf,
        base_l2=base_l2,
        tgt_l2=tgt_l2,
        linf_ratio=linf_ratio,
        l2_ratio=l2_ratio,
        bc=bc,
        passed=passed,
        shape_match=True,
        base_errors=base_errors,
        tgt_errors=tgt_errors,
    )


def _get_prompts(capture_dir: str) -> List[str]:
    """Get all prompt hashes from a capture directory."""
    prompts = []
    for d in Path(capture_dir).iterdir():
        if d.is_dir() and d.name.startswith("prompt_"):
            prompts.append(d.name)
    return sorted(prompts)


def _get_steps(capture_dir: str, prompt: str) -> List[int]:
    """Get all step numbers for a prompt."""
    steps = []
    prompt_dir = Path(capture_dir) / prompt
    for d in prompt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(steps)


def _get_ranks(capture_dir: str, prompt: str, step: int = 0) -> List[int]:
    """Get all rank numbers from a capture directory."""
    step_dir = Path(capture_dir) / prompt / f"step_{step}"
    if not step_dir.exists():
        return [0]
    ranks = set()
    for f in step_dir.glob("*_rank*.pt"):
        match = re.search(r"_rank(\d+)\.pt$", f.name)
        if match:
            ranks.add(int(match.group(1)))
    return sorted(ranks) if ranks else [0]


def compare_capture_dirs(
    dir1: str,
    dir2: str,
    prompt: Optional[str] = None,
    step: Optional[int] = None,
    rank: Optional[int] = 0,
) -> TwoWayResults:
    """
    Compare captured tensors between two directories.

    Args:
        dir1: First capture directory (reference)
        dir2: Second capture directory (test)
        prompt: Prompt hash to compare, or None for all prompts
        step: Step number to compare, or None for all steps
        rank: Rank to compare, or None for all ranks

    Returns:
        Dict mapping prompt -> step -> List of ComparisonResult

    Example:
        >>> results = compare_capture_dirs("/tmp/cpu", "/tmp/neuron")  # all prompts, all steps, rank 0
        >>> results = compare_capture_dirs("/tmp/cpu", "/tmp/neuron", rank=None)  # all ranks
    """
    # TODO: Add module_mapping parameter for cross-implementation comparison
    # (e.g., {"self_attn": "attn"} to map HF module names to vLLM Neuron names)
    prompts = [prompt] if prompt else _get_prompts(dir1)

    all_results = {}
    for p in prompts:
        steps = [step] if step is not None else _get_steps(dir1, p)
        ranks = (
            [rank]
            if rank is not None
            else _get_ranks(dir1, p, steps[0] if steps else 0)
        )

        prompt_results = {}
        for s in steps:
            path1 = Path(dir1) / p / f"step_{s}"
            path2 = Path(dir2) / p / f"step_{s}"

            if not path1.exists() or not path2.exists():
                continue

            results = []
            for r in ranks:
                suffix = f"_rank{r}.pt"
                files1 = {
                    f.stem.replace(suffix.replace(".pt", ""), ""): f
                    for f in path1.glob(f"*{suffix}")
                }
                files2 = {
                    f.stem.replace(suffix.replace(".pt", ""), ""): f
                    for f in path2.glob(f"*{suffix}")
                }

                common = set(files1.keys()) & set(files2.keys())

                for name in sorted(common, key=natural_sort_key):
                    t1 = torch.load(files1[name], weights_only=True)
                    t2 = torch.load(files2[name], weights_only=True)
                    display_name = f"{name}/rank{r}" if len(ranks) > 1 else name
                    result = compare_tensors(t1, t2, name=display_name)
                    results.append(result)

            prompt_results[s] = results

        all_results[p] = prompt_results

    return all_results


def compare_capture_dirs_three_way(
    baseline_dir: str,
    expected_dir: str,
    actual_dir: str,
    prompt: Optional[str] = None,
    step: Optional[int] = None,
    rank: Optional[int] = 0,
    dynamic_threshold_config: DynamicThresholdConfig = None,
) -> ThreeWayResults:
    """
    Three-way comparison of captured tensors.

    Args:
        baseline_dir: FP32 reference captures
        expected_dir: Same-dtype baseline captures (e.g., BF16 CPU)
        actual_dir: Target captures (e.g., BF16 Neuron)
        prompt: Prompt hash to compare, or None for all prompts
        step: Step number to compare, or None for all steps
        rank: Rank to compare, or None for all ranks
        dynamic_threshold_config: Threshold config for pass/fail determination

    Returns:
        Dict mapping prompt -> step -> List of ThreeWayComparisonResult

    Example:
        >>> results = compare_capture_dirs_three_way("/tmp/fp32", "/tmp/bf16_cpu", "/tmp/bf16_neuron")
        >>> results = compare_capture_dirs_three_way("/tmp/fp32", "/tmp/bf16_cpu", "/tmp/bf16_neuron", rank=None)  # all ranks
    """
    # TODO: Add module_mapping parameter for cross-implementation comparison
    # (e.g., {"self_attn": "attn"} to map HF module names to vLLM Neuron names)
    prompts = [prompt] if prompt else _get_prompts(baseline_dir)

    all_results = {}
    for p in prompts:
        steps = [step] if step is not None else _get_steps(baseline_dir, p)
        ranks = (
            [rank]
            if rank is not None
            else _get_ranks(baseline_dir, p, steps[0] if steps else 0)
        )

        prompt_results = {}
        for s in steps:
            path_base = Path(baseline_dir) / p / f"step_{s}"
            path_exp = Path(expected_dir) / p / f"step_{s}"
            path_act = Path(actual_dir) / p / f"step_{s}"

            if not all(p.exists() for p in [path_base, path_exp, path_act]):
                continue

            results = []
            for r in ranks:
                suffix = f"_rank{r}.pt"
                files_base = {
                    f.stem.replace(suffix.replace(".pt", ""), ""): f
                    for f in path_base.glob(f"*{suffix}")
                }
                files_exp = {
                    f.stem.replace(suffix.replace(".pt", ""), ""): f
                    for f in path_exp.glob(f"*{suffix}")
                }
                files_act = {
                    f.stem.replace(suffix.replace(".pt", ""), ""): f
                    for f in path_act.glob(f"*{suffix}")
                }

                common = (
                    set(files_base.keys())
                    & set(files_exp.keys())
                    & set(files_act.keys())
                )

                for name in sorted(common, key=natural_sort_key):
                    baseline = torch.load(files_base[name], weights_only=True)
                    expected = torch.load(files_exp[name], weights_only=True)
                    actual = torch.load(files_act[name], weights_only=True)
                    display_name = f"{name}/rank{r}" if len(ranks) > 1 else name
                    result = compare_tensors_three_way(
                        baseline,
                        expected,
                        actual,
                        name=display_name,
                        dynamic_threshold_config=dynamic_threshold_config,
                    )
                    results.append(result)

            prompt_results[s] = results

        all_results[p] = prompt_results

    return all_results


def _group_by_req(
    forward_passes: List[CapturedForwardPass],
) -> Dict[str, List[CapturedForwardPass]]:
    """Group forward passes by request ID, prefills first then decodes by position."""
    grouped: Dict[str, List[CapturedForwardPass]] = {}
    for fwd in forward_passes:
        for req_id in fwd.metadata.req_ids:
            grouped.setdefault(req_id, []).append(fwd)
    for fwds in grouped.values():
        fwds.sort(
            key=lambda f: (
                0 if f.is_prefill else 1,
                max(f.metadata.positions) if f.metadata.positions else 0,
            )
        )
    return grouped


def compare_captures_two_way(
    ref: List[CapturedForwardPass],
    test: List[CapturedForwardPass],
    prompt_index: Optional[int] = None,
    phase: Optional[str] = None,
    ref_reconstruction_fn: Optional[ReconstructionFn] = None,
    test_reconstruction_fn: Optional[ReconstructionFn] = None,
    alignment_fn: Optional[AlignmentFn] = None,
    module_order: Optional[List[str]] = None,
) -> TwoWayResults:
    """Two-way comparison of captured forward passes.

    Supports APC (Automatic Prefix Caching) and segmented prefill.
    Per-token comparison is used whenever the sequence dimension has
    multiple tokens, regardless of phase.

    .. todo:: Speculative decoding (num_tokens > 1 in decode) is not yet
       validated. The per-token logic should handle it, but capture and
       reconstruction support needs testing.

    Args:
        ref: Reference captures from tensor_io.read().
        test: Test captures from tensor_io.read().
        prompt_index: 0-based prompt index to compare, or None for all.
        phase: "prefill" or "decode" to filter, or None for all
        ref_reconstruction_fn: Reconstructs ref tensor from per-rank
            tensors. Receives (rank_tensors, module_name, phase,
            positions) and returns a single comparable tensor.
            phase is "prefill" or "decode" (extensible for future phases).
        test_reconstruction_fn: Reconstructs test tensor.
        alignment_fn: Aligns tensors before comparison.
            Defaults to align_and_truncate_hidden.
        module_order: Module names in execution order for sorting

    Returns:
        Dict mapping prompt -> step -> List of ComparisonResult

    Raises:
        ValueError: If prompt_index is out of range.

    Example:
        >>> from vllm_neuron.accuracy.tensor_io import read
        >>> results = compare_captures_two_way(read("/tmp/ref"), read("/tmp/test"))
    """
    if alignment_fn is None:
        alignment_fn = align_and_truncate_hidden

    ref_grouped = _group_by_req(ref)
    test_grouped = _group_by_req(test)
    ref_reqs = sorted(ref_grouped.keys())
    test_reqs = sorted(test_grouped.keys())
    num_reqs = min(len(ref_reqs), len(test_reqs))

    if len(ref_reqs) != len(test_reqs):
        logger.warning(
            "Request count mismatch: ref has %d requests, test has %d. "
            "Comparing first %d.",
            len(ref_reqs),
            len(test_reqs),
            num_reqs,
        )

    if prompt_index is not None:
        if prompt_index < 0 or prompt_index >= num_reqs:
            raise ValueError(
                f"prompt_index={prompt_index} out of range, "
                f"only {num_reqs} prompts available"
            )
        indices = [prompt_index]
    else:
        indices = list(range(num_reqs))

    if not indices:
        raise ValueError("No prompts to compare — captures are empty")

    all_results: Dict[str, Dict[int, List[ComparisonResult]]] = {}

    for idx in indices:
        r_req, t_req = ref_reqs[idx], test_reqs[idx]
        prompt_key = f"prompt_{idx}"
        ref_fwds = ref_grouped[r_req]
        test_fwds = test_grouped[t_req]
        num_steps = min(len(ref_fwds), len(test_fwds))

        if len(ref_fwds) != len(test_fwds):
            logger.warning(
                "Step count mismatch for prompt %d: ref has %d steps, "
                "test has %d. Comparing first %d.",
                idx,
                len(ref_fwds),
                len(test_fwds),
                num_steps,
            )

        step_indices = []
        for s in range(num_steps):
            fwd = ref_fwds[s]
            if phase == "prefill" and not fwd.is_prefill:
                continue
            if phase == "decode" and fwd.is_prefill:
                continue
            step_indices.append(s)

        prompt_results: Dict[int, List[ComparisonResult]] = {}
        for s in step_indices:
            rf, tf = ref_fwds[s], test_fwds[s]
            step_phase = "prefill" if rf.is_prefill else "decode"

            def _norm(name):
                return name.replace(".", "_")

            ref_mods = {_norm(k): k for k in rf.tensors}
            test_mods = {_norm(k): k for k in tf.tensors}
            common = set(ref_mods) & set(test_mods)

            if module_order:
                order_map = {_norm(n): i for i, n in enumerate(module_order)}
                sorted_names = sorted(
                    common, key=lambda n: order_map.get(n, float("inf"))
                )
            else:
                sorted_names = sorted(common, key=natural_sort_key)

            results: List[ComparisonResult] = []
            for norm_name in sorted_names:
                r_t = _tensor_from_ranks(
                    rf.tensors[ref_mods[norm_name]],
                    ref_reconstruction_fn,
                    norm_name,
                    step_phase,
                    rf.metadata.positions,
                )
                t_t = _tensor_from_ranks(
                    tf.tensors[test_mods[norm_name]],
                    test_reconstruction_fn,
                    norm_name,
                    step_phase,
                    tf.metadata.positions,
                )
                # AlignmentFn takes (baseline, expected, actual). For two-way
                # comparison there is no "expected" tensor; pass r_t as both
                # baseline and expected so the hidden-dim truncation uses
                # min(ref, test). Custom alignment_fns must tolerate baseline==expected.
                r_a, _, t_a, _ = alignment_fn(r_t, r_t, t_t)

                num_tokens = get_seq_dim_size(r_a)
                if num_tokens > 1:
                    # Per-token comparison (prefill)
                    for i in range(num_tokens):
                        results.append(
                            compare_tensors(
                                slice_token(r_a, i),
                                slice_token(t_a, i),
                                name=f"{norm_name}/token{i}",
                                align=False,
                            )
                        )
                else:
                    results.append(
                        compare_tensors(r_a, t_a, name=norm_name, align=False)
                    )

            prompt_results[s] = results
        all_results[prompt_key] = prompt_results

    return all_results


def compare_captures_three_way(
    baseline: List[CapturedForwardPass],
    expected: List[CapturedForwardPass],
    actual: List[CapturedForwardPass],
    prompt_index: Optional[int] = None,
    phase: Optional[str] = None,
    dynamic_threshold_config: DynamicThresholdConfig = None,
    reference_reconstruction_fn: Optional[ReconstructionFn] = None,
    target_reconstruction_fn: Optional[ReconstructionFn] = None,
    alignment_fn: Optional[AlignmentFn] = None,
    module_order: Optional[List[str]] = None,
) -> ThreeWayResults:
    """Three-way comparison of captured forward passes.

    Supports APC (Automatic Prefix Caching) and segmented prefill.
    Per-token comparison is used whenever the sequence dimension has
    multiple tokens, regardless of phase.

    .. todo:: Speculative decoding (num_tokens > 1 in decode) is not yet
       validated. The per-token logic should handle it, but capture and
       reconstruction support needs testing.

    Args:
        baseline: FP32 reference captures from tensor_io.read().
        expected: Same-dtype baseline captures (e.g., BF16 CPU).
        actual: Target captures (e.g., BF16 Neuron).
        prompt_index: 0-based prompt index to compare, or None for all.
        phase: "prefill" or "decode" to filter, or None for all
        dynamic_threshold_config: Threshold config for pass/fail determination
        reference_reconstruction_fn: Reconstructs baseline/expected tensors.
        target_reconstruction_fn: Reconstructs actual tensors.
        alignment_fn: Aligns three tensors before comparison.
            Defaults to align_and_truncate_hidden.
        module_order: Module names in execution order for sorting

    Returns:
        Dict mapping prompt -> step -> List of ThreeWayComparisonResult

    Raises:
        ValueError: If prompt_index is out of range.

    Example:
        >>> from vllm_neuron.accuracy.tensor_io import read
        >>> results = compare_captures_three_way(
        ...     read("/tmp/fp32"), read("/tmp/bf16"), read("/tmp/neuron"))
    """
    if alignment_fn is None:
        alignment_fn = align_and_truncate_hidden

    base_grouped = _group_by_req(baseline)
    exp_grouped = _group_by_req(expected)
    act_grouped = _group_by_req(actual)
    base_reqs = sorted(base_grouped.keys())
    exp_reqs = sorted(exp_grouped.keys())
    act_reqs = sorted(act_grouped.keys())
    num_reqs = min(len(base_reqs), len(exp_reqs), len(act_reqs))

    req_counts = (len(base_reqs), len(exp_reqs), len(act_reqs))
    if not all(c == req_counts[0] for c in req_counts):
        logger.warning(
            "Request count mismatch: baseline=%d, expected=%d, actual=%d. "
            "Comparing first %d.",
            *req_counts,
            num_reqs,
        )

    if prompt_index is not None:
        if prompt_index < 0 or prompt_index >= num_reqs:
            raise ValueError(
                f"prompt_index={prompt_index} out of range, "
                f"only {num_reqs} prompts available"
            )
        indices = [prompt_index]
    else:
        indices = list(range(num_reqs))

    if not indices:
        raise ValueError("No prompts to compare — captures are empty")

    all_results: Dict[str, Dict[int, List[ThreeWayComparisonResult]]] = {}

    for idx in indices:
        b_req, e_req, a_req = base_reqs[idx], exp_reqs[idx], act_reqs[idx]
        prompt_key = f"prompt_{idx}"
        base_fwds = base_grouped[b_req]
        exp_fwds = exp_grouped[e_req]
        act_fwds = act_grouped[a_req]
        num_steps = min(len(base_fwds), len(exp_fwds), len(act_fwds))

        step_counts = (len(base_fwds), len(exp_fwds), len(act_fwds))
        if not all(c == step_counts[0] for c in step_counts):
            logger.warning(
                "Step count mismatch for prompt %d: baseline=%d, expected=%d, "
                "actual=%d. Comparing first %d.",
                idx,
                *step_counts,
                num_steps,
            )

        step_indices = []
        for s in range(num_steps):
            fwd = act_fwds[s]
            if phase == "prefill" and not fwd.is_prefill:
                continue
            if phase == "decode" and fwd.is_prefill:
                continue
            step_indices.append(s)

        prompt_results: Dict[int, List[ThreeWayComparisonResult]] = {}
        for s in step_indices:
            bf, ef, af = base_fwds[s], exp_fwds[s], act_fwds[s]
            step_phase = "prefill" if af.is_prefill else "decode"

            def _norm(name):
                return name.replace(".", "_")

            base_mods = {_norm(k): k for k in bf.tensors}
            exp_mods = {_norm(k): k for k in ef.tensors}
            act_mods = {_norm(k): k for k in af.tensors}
            common = set(base_mods) & set(exp_mods) & set(act_mods)

            if module_order:
                order_map = {_norm(n): i for i, n in enumerate(module_order)}
                sorted_names = sorted(
                    common, key=lambda n: order_map.get(n, float("inf"))
                )
            else:
                sorted_names = sorted(common, key=natural_sort_key)

            results: List[ThreeWayComparisonResult] = []
            for norm_name in sorted_names:
                b_t = _tensor_from_ranks(
                    bf.tensors[base_mods[norm_name]],
                    reference_reconstruction_fn,
                    norm_name,
                    step_phase,
                    bf.metadata.positions,
                )
                e_t = _tensor_from_ranks(
                    ef.tensors[exp_mods[norm_name]],
                    reference_reconstruction_fn,
                    norm_name,
                    step_phase,
                    ef.metadata.positions,
                )
                a_t = _tensor_from_ranks(
                    af.tensors[act_mods[norm_name]],
                    target_reconstruction_fn,
                    norm_name,
                    step_phase,
                    af.metadata.positions,
                )

                b_a, e_a, a_a, _ = alignment_fn(b_t, e_t, a_t)
                num_tokens = get_seq_dim_size(b_a)
                if num_tokens > 1:
                    # Per-token comparison (prefill)
                    for i in range(num_tokens):
                        results.append(
                            compare_tensors_three_way(
                                slice_token(b_a, i),
                                slice_token(e_a, i),
                                slice_token(a_a, i),
                                name=f"{norm_name}/token{i}",
                                dynamic_threshold_config=dynamic_threshold_config,
                                align=False,
                            )
                        )
                else:
                    results.append(
                        compare_tensors_three_way(
                            b_a,
                            e_a,
                            a_a,
                            name=norm_name,
                            dynamic_threshold_config=dynamic_threshold_config,
                            align=False,
                        )
                    )

            prompt_results[s] = results
        all_results[prompt_key] = prompt_results

    return all_results


def print_comparison_report(
    results: Dict[str, Dict[int, List[ComparisonResult]]],
    threshold: float = 0.01,
    label1: str = "Reference",
    label2: str = "Test",
    colorize: bool = True,
) -> None:
    """
    Print comparison report showing module errors in execution order.

    Args:
        results: Dict mapping prompt -> step -> List of ComparisonResult
        threshold: Threshold for flagging deviations (red if exceeded)
        label1: Label for reference
        label2: Label for test
        colorize: Whether to colorize output

    Example:
        >>> results = compare_capture_dirs("/tmp/cpu", "/tmp/neuron")
        >>> print_comparison_report(results, threshold=0.01)
    """
    print(f"\n{'=' * 80}")
    print(f"TENSOR COMPARISON: {label1} vs {label2}")
    print("=" * 80)
    print(f"Threshold: L-inf > {threshold} highlighted in red")

    for prompt in sorted(results.keys()):
        prompt_results = results[prompt]
        print(f"\n=== {prompt} ===")

        for step in sorted(prompt_results.keys()):
            step_results = prompt_results[step]
            print(f"\n--- Step {step} ---")
            print(f"{'Module':<45} {'L-inf':>12} {'L2':>12} {'Max Abs':>12}")
            print("-" * 80)

            for r in step_results:
                if not r.is_match:
                    status = _colorize("SHAPE MISMATCH", "RED", colorize)
                    print(f"{r.name:<45} {status}")
                    continue

                color = "RED" if r.linf_rel > threshold else "GREEN"
                linf_str = _colorize(f"{r.linf_rel:.2e}", color, colorize)
                l2_str = f"{r.l2_rel:.2e}"
                max_str = f"{r.max_abs:.2e}"

                print(f"{r.name:<45} {linf_str:>12} {l2_str:>12} {max_str:>12}")

    # Print aggregate summary if multiple prompts
    if len(results) > 1:
        _print_two_way_aggregate_summary(results, threshold, colorize)


def _print_two_way_aggregate_summary(
    results: Dict[str, Dict[int, List[ComparisonResult]]],
    threshold: float,
    colorize: bool = True,
) -> None:
    """Print aggregate summary for two-way comparison, grouped by (module, step)."""
    # Group by (step, module_name) across prompts
    grouped: Dict[Tuple[int, str], List[ComparisonResult]] = {}

    for prompt_results in results.values():
        for step, step_results in prompt_results.items():
            for r in step_results:
                if r.is_match:
                    key = (step, r.name)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(r)

    if not grouped:
        return

    n_prompts = len(results)
    print(f"\n{'=' * 80}")
    print("AGGREGATE SUMMARY ACROSS ALL PROMPTS")
    print("=" * 80)
    print(f"Prompts: {n_prompts}")

    # Get module order from first prompt's results (preserves execution order)
    first_prompt = next(iter(results.values()))
    steps_in_order = sorted(first_prompt.keys())
    module_order = (
        [r.name for r in first_prompt[steps_in_order[0]]] if steps_in_order else []
    )

    # Print per (step, module) aggregates in execution order
    for step in steps_in_order:
        for name in module_order:
            key = (step, name)
            if key not in grouped or len(grouped[key]) < 2:
                continue
            results_list = grouped[key]
            linf_vals = [r.linf_rel for r in results_list]
            l2_vals = [r.l2_rel for r in results_list]

            mean_linf = np.mean(linf_vals)
            mean_l2 = np.mean(l2_vals)
            max_linf = max(linf_vals)

            linf_color = "RED" if mean_linf > threshold else "GREEN"
            print(f"\nStep {step} / {name}:")
            print(
                f"  Mean L-inf: {_colorize(f'{mean_linf:.2e}', linf_color, colorize)}, Mean L2: {mean_l2:.2e}, Max L-inf: {max_linf:.2e}"
            )


def print_three_way_report(
    results: Dict[str, Dict[int, List[ThreeWayComparisonResult]]],
    label_baseline: str = "FP32",
    label_expected: str = "BF16 CPU",
    label_actual: str = "BF16 Neuron",
    colorize: bool = True,
) -> None:
    """
    Print three-way comparison report showing module errors in execution order.

    Args:
        results: Dict mapping prompt -> step -> List of ThreeWayComparisonResult
        label_baseline: Label for FP32 baseline
        label_expected: Label for expected (e.g., BF16 CPU)
        label_actual: Label for actual (e.g., BF16 Neuron)
        colorize: Whether to colorize output

    Example:
        >>> results = compare_capture_dirs_three_way("/tmp/fp32", "/tmp/bf16_cpu", "/tmp/bf16_neuron")
        >>> print_three_way_report(results)
    """
    print(f"\n{'=' * 90}")
    print(
        f"THREE-WAY COMPARISON: {label_expected} & {label_actual} vs {label_baseline}"
    )
    print("=" * 90)
    print(
        "Ratio = Tgt/Base error. Red if ratio >= 1.5x (target error significantly exceeds baseline)"
    )

    for prompt in sorted(results.keys()):
        prompt_results = results[prompt]
        print(f"\n=== {prompt} ===")

        for step in sorted(prompt_results.keys()):
            step_results = prompt_results[step]
            print(f"\n--- Step {step} ---")
            print(f"{'Module':<40} {'L-inf Ratio':>12} {'L2 Ratio':>12} {'BC':>8}")
            print("-" * 90)

            for r in step_results:
                if not r.shape_match:
                    status = _colorize("SHAPE MISMATCH", "RED", colorize)
                    print(f"{r.name:<40} {status}")
                    continue

                linf_color = "RED" if r.linf_ratio >= 1.5 else "GREEN"
                l2_color = "RED" if r.l2_ratio >= 1.5 else "GREEN"

                linf_str = _colorize(f"{r.linf_ratio:.4f}", linf_color, colorize)
                l2_str = _colorize(f"{r.l2_ratio:.4f}", l2_color, colorize)
                bc_str = f"{r.bc:.4f}"

                print(f"{r.name:<40} {linf_str:>12} {l2_str:>12} {bc_str:>8}")

    # Print aggregate summary if multiple prompts
    if len(results) > 1:
        agg = compute_aggregate_metrics(results)
        _print_aggregate_summary(agg, results, colorize)


def compute_aggregate_metrics(
    results: Dict[str, Dict[int, List[ThreeWayComparisonResult]]],
) -> Dict[Tuple[int, str], "AggregateMetrics"]:
    """Compute aggregate metrics per (step, module) across all prompts.

    Args:
        results: Dict mapping prompt -> step -> List of ThreeWayComparisonResult

    Returns:
        Dict mapping (step, module_name) -> AggregateMetrics

    Example:
        >>> results = compare_capture_dirs_three_way(fp32_dir, bf16_dir, neuron_dir)
        >>> agg = compute_aggregate_metrics(results)
        >>> for (step, name), metrics in agg.items():
        ...     print(f"{name}: mean_linf_ratio={metrics.mean_linf_ratio:.4f}")
    """
    # Group by (step, module_name) across prompts
    grouped: Dict[Tuple[int, str], List[ThreeWayComparisonResult]] = {}

    for prompt_results in results.values():
        for step, step_results in prompt_results.items():
            for r in step_results:
                if r.shape_match:
                    key = (step, r.name)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(r)

    # Compute aggregate per group
    aggregates = {}
    for key, results_list in grouped.items():
        if len(results_list) < 2:
            continue

        linf_ratios = [r.linf_ratio for r in results_list]
        l2_ratios = [r.l2_ratio for r in results_list]
        base_errors = [r.base_errors for r in results_list if r.base_errors is not None]
        tgt_errors = [r.tgt_errors for r in results_list if r.tgt_errors is not None]

        if base_errors and tgt_errors:
            pooled_base = np.concatenate(base_errors)
            pooled_tgt = np.concatenate(tgt_errors)
            aggregate_bc = _compute_bc(pooled_base, pooled_tgt)
        else:
            aggregate_bc = 0.0

        aggregates[key] = AggregateMetrics(
            mean_linf_ratio=np.mean(linf_ratios),
            mean_l2_ratio=np.mean(l2_ratios),
            max_linf_ratio=max(linf_ratios),
            max_l2_ratio=max(l2_ratios),
            aggregate_bc=aggregate_bc,
            n_prompts=len(results_list),
            n_modules=1,
        )

    return aggregates


def _tensor_from_ranks(
    rank_tensors: Dict[int, torch.Tensor],
    reconstruction_fn: Optional[ReconstructionFn],
    module_name: str,
    phase: str,
    positions: List[int],
) -> torch.Tensor:
    """Apply reconstruction function to rank tensors.

    When no reconstruction_fn is provided, returns rank 0 only. This is the
    safe default because combining ranks requires model-specific knowledge
    (e.g., SP gather along seq dim vs hidden dim concat for column-parallel).
    Users should provide a reconstruction_fn for multi-rank captures.
    """
    if reconstruction_fn:
        tensors = [rank_tensors[r] for r in sorted(rank_tensors.keys())]
        return reconstruction_fn(tensors, module_name, phase, positions)
    return rank_tensors[sorted(rank_tensors.keys())[0]]


def _print_aggregate_summary(
    aggregates: Dict[Tuple[int, str], AggregateMetrics],
    results: Dict[str, Dict[int, List[ThreeWayComparisonResult]]],
    colorize: bool = True,
) -> None:
    """Print aggregate summary per (step, module) across all prompts."""
    if not aggregates:
        return

    # Get module order from first prompt's results (preserves execution order)
    first_prompt = next(iter(results.values()))
    steps_in_order = sorted(first_prompt.keys())
    module_order = (
        [r.name for r in first_prompt[steps_in_order[0]]] if steps_in_order else []
    )

    print(f"\n{'=' * 90}")
    print("AGGREGATE SUMMARY ACROSS ALL PROMPTS")
    print("=" * 90)
    print(f"Prompts: {len(results)}")
    print()
    print(
        f"{'Step':<6} {'Module':<35} {'Mean L-inf':>12} {'Mean L2':>12} {'Agg BC':>10}"
    )
    print("-" * 90)

    for step in steps_in_order:
        for name in module_order:
            key = (step, name)
            if key not in aggregates:
                continue
            agg = aggregates[key]
            linf_color = "RED" if agg.mean_linf_ratio >= 1.5 else "GREEN"
            l2_color = "RED" if agg.mean_l2_ratio >= 1.5 else "GREEN"
            bc_color = (
                "GREEN"
                if agg.aggregate_bc >= 0.95
                else ("YELLOW" if agg.aggregate_bc >= 0.9 else "RED")
            )

            linf_str = _colorize(f"{agg.mean_linf_ratio:.4f}", linf_color, colorize)
            l2_str = _colorize(f"{agg.mean_l2_ratio:.4f}", l2_color, colorize)
            bc_str = _colorize(f"{agg.aggregate_bc:.4f}", bc_color, colorize)

            print(f"{step:<6} {name:<35} {linf_str:>12} {l2_str:>12} {bc_str:>10}")
