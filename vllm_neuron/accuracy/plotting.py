# SPDX-License-Identifier: Apache-2.0
"""Plotting utilities for module-level accuracy debugging.

Provides visual diagnostics when three-way comparison fails. Two main plots:

- **Error distribution histogram**: Overlays CPU (dtype baseline) and Neuron
  (target) error distributions. If they overlap, the error is dtype-inherent.
  If they diverge, it's target-specific.

- **QQ-plot**: Plots Neuron error quantiles vs CPU error quantiles. Points on
  the 45° diagonal mean distributions match. Systematic deviation indicates a
  configuration or implementation issue.

Usage::

    from vllm_neuron.accuracy.plotting import (
        plot_error_distributions,
        plot_error_qqplot,
    )

    # From a ThreeWayComparisonResult or raw error arrays:
    plot_error_distributions(base_errors, tgt_errors, name="attn/layer0")
    plot_error_qqplot(base_errors, tgt_errors, name="attn/layer0")

    # Or from tensors directly:
    plot_three_way(fp32_ref, bf16_cpu, bf16_neuron, name="attn/layer0")
"""

import os
from typing import Optional, Union

import numpy as np
import torch


def _ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    # Handle list/tuple of tensors (e.g., from multi-sample accuracy tests)
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.cat([t.detach().float().cpu() for t in x]).numpy()
    return np.asarray(x, dtype=np.float64)


def _compute_bc(errors1: np.ndarray, errors2: np.ndarray, num_bins: int = 100) -> float:
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


def plot_error_distributions(
    base_errors: Union[np.ndarray, torch.Tensor],
    tgt_errors: Union[np.ndarray, torch.Tensor],
    name: str = "tensor",
    output_path: Optional[str] = None,
    base_label: str = "CPU (dtype baseline)",
    tgt_label: str = "Neuron (target)",
    precision_label: str = "BF16",
    num_bins: int = 50,
    ax=None,
) -> None:
    """Plot overlaid error distribution histograms for CPU vs Neuron.

    If distributions overlap, the error is dtype-inherent (expected).
    If they diverge, the error is target-specific (investigate).

    Args:
        base_errors: |expected_dtype - fp32| error values (flattened).
        tgt_errors: |actual_target - fp32| error values (flattened).
        name: Label for the plot title.
        output_path: File path to save. If None, saves to ``{name}_error_dist.png``.
        base_label: Legend label for baseline errors.
        tgt_label: Legend label for target errors.
        precision_label: Display label for the dtype precision (used in axis label) for axis (e.g., "BF16", "FP16").
        num_bins: Number of histogram bins.
        ax: Optional matplotlib Axes. If provided, draws on it and skips saving.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_errors = _ensure_numpy(base_errors).flatten()
    tgt_errors = _ensure_numpy(tgt_errors).flatten()
    bc = _compute_bc(base_errors, tgt_errors)
    base_sigma = np.sqrt(np.mean(base_errors**2))
    tgt_sigma = np.sqrt(np.mean(tgt_errors**2))
    sigma_ratio = tgt_sigma / base_sigma if base_sigma > 0 else float("inf")

    all_errs = np.concatenate([base_errors, tgt_errors])
    bins = np.linspace(0, np.percentile(all_errs, 99.5), num_bins)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        base_errors, bins=bins, alpha=0.7, label=base_label, density=True, color="blue"
    )
    ax.hist(
        tgt_errors, bins=bins, alpha=0.7, label=tgt_label, density=True, color="red"
    )
    ax.axvline(
        base_sigma,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"CPU σ={base_sigma:.4e}",
    )
    ax.axvline(
        tgt_sigma,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Neuron σ={tgt_sigma:.4e}",
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"|{precision_label} - FP32|")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution  —  BC={bc:.4f}  σ-ratio={sigma_ratio:.3f}")
    ax.legend(fontsize=8)

    if standalone:
        plt.tight_layout()
        if output_path is None:
            safe_name = name.replace("/", "_").replace(" ", "_")
            output_path = f"{safe_name}_error_dist.png"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved error distribution plot to {output_path}")


def plot_error_qqplot(
    base_errors: Union[np.ndarray, torch.Tensor],
    tgt_errors: Union[np.ndarray, torch.Tensor],
    name: str = "tensor",
    output_path: Optional[str] = None,
    num_quantiles: int = 10000,
    ax=None,
) -> None:
    """Plot QQ-plot comparing CPU vs Neuron error quantiles.

    Points on the 45° diagonal mean error distributions match.
    Systematic deviation from the diagonal indicates a bug.

    Args:
        base_errors: |expected_dtype - fp32| error values (flattened).
        tgt_errors: |actual_target - fp32| error values (flattened).
        name: Label for the plot title.
        output_path: File path to save. If None, saves to ``{name}_qqplot.png``.
        num_quantiles: Number of quantile points to plot.
        ax: Optional matplotlib Axes. If provided, draws QQ-plot on it and skips saving.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_errors = _ensure_numpy(base_errors).flatten()
    tgt_errors = _ensure_numpy(tgt_errors).flatten()
    bc = _compute_bc(base_errors, tgt_errors)

    n = min(len(base_errors), len(tgt_errors), num_quantiles)
    p = np.linspace(0, 1, n)
    q_base = np.quantile(np.sort(base_errors), p)
    q_tgt = np.quantile(np.sort(tgt_errors), p)

    standalone = ax is None
    if standalone:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram (only in standalone mode)
        n_bins = min(1000, int(np.sqrt(len(base_errors))))
        ax1.hist(
            tgt_errors,
            bins=n_bins,
            alpha=0.75,
            color="red",
            label="Neuron",
            density=True,
        )
        ax1.hist(
            base_errors,
            bins=n_bins,
            alpha=0.75,
            color="blue",
            label="CPU",
            density=True,
        )
        ax1.set_xlabel("Error")
        ax1.set_ylabel("Density")
        ax1.set_title("Error Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        mean_val = np.mean(tgt_errors)
        std_val = max(np.std(tgt_errors), np.std(base_errors))
        ax1.set_xlim([max(0, mean_val - 5 * std_val), mean_val + 5 * std_val])
    else:
        ax2 = ax

    # QQ-plot
    ax2.scatter(q_base, q_tgt, c="green", s=5, alpha=0.6)
    min_val = min(q_base.min(), q_tgt.min())
    max_val = max(q_base.max(), q_tgt.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="45° line (ideal)")
    ax2.set_xlabel("CPU Error Quantiles")
    ax2.set_ylabel("Neuron Error Quantiles")
    ax2.set_title("QQ-Plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if standalone:
        title = (
            f"{name}  —  BC={bc:.4f}\n"
            f"Neuron: min={np.min(tgt_errors):.4e}, max={np.max(tgt_errors):.4e}, "
            f"mean={np.mean(tgt_errors):.4e}, std={np.std(tgt_errors):.4e}\n"
            f"CPU:    min={np.min(base_errors):.4e}, max={np.max(base_errors):.4e}, "
            f"mean={np.mean(base_errors):.4e}, std={np.std(base_errors):.4e}"
        )
        plt.suptitle(title, fontsize=9)
        plt.tight_layout()

        if output_path is None:
            safe_name = name.replace("/", "_").replace(" ", "_")
            output_path = f"{safe_name}_qqplot.png"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved QQ-plot to {output_path}")
    print(f"Saved QQ-plot to {output_path}")


def plot_scatter(
    actual: Union[np.ndarray, torch.Tensor],
    expected: Union[np.ndarray, torch.Tensor],
    name: str = "actual_vs_expected",
    output_path: Optional[str] = None,
    max_points: int = None,
) -> None:
    """Scatter plot of actual vs expected values for two-way comparison.

    Flattens both tensors and plots each element as a point. Points should
    cluster along the y=x line. Deviation from the line shows where errors
    concentrate.

    Args:
        actual: Target tensor values (e.g., Neuron output).
        expected: Reference tensor values (e.g., HF output).
        name: Label for the plot title.
        output_path: File path to save. If None, saves to ``{name}_scatter.png``.
        max_points: If set, subsample to this many points for readability.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = _ensure_numpy(actual).flatten()
    expected = _ensure_numpy(expected).flatten()

    # Subsample for readability if requested
    if max_points is not None and len(actual) > max_points:
        idx = np.random.default_rng(0).choice(len(actual), max_points, replace=False)
        actual = actual[idx]
        expected = expected[idx]

    abs_diff = np.abs(actual - expected)
    max_abs = np.max(abs_diff)
    exp_max = np.max(np.abs(expected))
    max_rel = max_abs / exp_max if exp_max > 0 else float("inf")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(expected, actual, s=1, alpha=0.3, c="steelblue")
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="y=x (ideal)"
    )
    ax.set_xlabel("Expected (reference)")
    ax.set_ylabel("Actual (target)")
    ax.set_title(
        f"{name}\nmax |diff|={max_abs:.4e}, max rel={max_rel:.4e}, n={len(actual)}"
    )
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    if output_path is None:
        safe_name = name.replace("/", "_").replace(" ", "_")
        output_path = f"{safe_name}_scatter.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def plot_three_way(
    baseline: Union[torch.Tensor, np.ndarray],
    expected: Union[torch.Tensor, np.ndarray],
    actual: Union[torch.Tensor, np.ndarray],
    name: str = "three_way",
    output_dir: Optional[str] = None,
) -> None:
    """Convenience: compute errors from three tensors and generate both plots.

    Args:
        baseline: FP32 reference tensor.
        expected: Same-dtype baseline tensor (e.g., BF16 on CPU).
        actual: Target tensor (e.g., BF16 on Neuron).
        name: Label for the plots.
        output_dir: Directory to save plots. Defaults to current directory.
    """
    baseline = _ensure_numpy(baseline).flatten()
    expected = _ensure_numpy(expected).flatten()
    actual = _ensure_numpy(actual).flatten()

    base_errors = np.abs(expected - baseline)
    tgt_errors = np.abs(actual - baseline)

    output_dir = output_dir or "."
    safe_name = name.replace("/", "_").replace(" ", "_")

    plot_error_distributions(
        base_errors,
        tgt_errors,
        name=name,
        output_path=os.path.join(output_dir, f"{safe_name}_error_dist.png"),
    )
    plot_error_qqplot(
        base_errors,
        tgt_errors,
        name=name,
        output_path=os.path.join(output_dir, f"{safe_name}_qqplot.png"),
    )
