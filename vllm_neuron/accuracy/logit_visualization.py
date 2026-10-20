# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualization functions for vLLM Neuron logit validation results.
"""

import os
from typing import List, Dict
import torch
from .constants import DEFAULT_TOLERANCE_MAP, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE


def _process_logits_for_validation(
    target_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    remove_shift: bool = True,
    dynamic_threshold_config: dict = None,
) -> List[List[dict]]:
    """Process logits tensors into validation results format.

    Args:
        target_logits: Target logits tensor [seq_len, batch_size, vocab_size]
        baseline_logits: Baseline logits tensor [seq_len, batch_size, vocab_size]
        tol_map: Tolerance map for validation
        divergence_difference_tol: Divergence tolerance
        remove_shift: Whether to remove shift in preprocessing
        dynamic_threshold_config: Dynamic threshold configuration

    Returns:
        List of batch results in format expected by visualize_logit_results
    """
    from .logit_validation import _validate_single_token_logits

    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    seq_len, batch_size = target_logits.shape[:2]
    results = [[] for _ in range(batch_size)]

    for token_idx in range(seq_len):
        for batch_idx in range(batch_size):
            passed, result = _validate_single_token_logits(
                expected_logits=baseline_logits[token_idx, batch_idx],
                actual_logits=target_logits[token_idx, batch_idx],
                tol_map=tol_map,
                divergence_difference_tol=divergence_difference_tol,
                remove_shift=remove_shift,
                baseline_logits=None,
                dynamic_threshold_config=dynamic_threshold_config,
            )
            results[batch_idx].append(result)

    return results


def visualize_logits(
    target_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    output_dir: str = "logit_plots",
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    remove_shift: bool = True,
    dynamic_threshold_config: dict = None,
) -> None:
    """Visualize logits comparison using visualize_logit_results.

    Args:
        target_logits: Target logits tensor [seq_len, batch_size, vocab_size]
        baseline_logits: Baseline logits tensor [seq_len, batch_size, vocab_size]
        output_dir: Directory to save visualization files
        tol_map: Tolerance map for validation
        divergence_difference_tol: Divergence tolerance
        remove_shift: Whether to remove shift in preprocessing
        dynamic_threshold_config: Dynamic threshold configuration

    Example:
        ```python
        import torch
        from vllm_neuron.accuracy.logit_visualization import visualize_logits

        # Create sample logits tensors
        seq_len, batch_size, vocab_size = 10, 2, 1000
        target_logits = torch.randn(seq_len, batch_size, vocab_size)
        baseline_logits = torch.randn(seq_len, batch_size, vocab_size)

        # Visualize comparison
        visualize_logits(target_logits, baseline_logits, "output_plots")
        ```
    """
    results = _process_logits_for_validation(
        target_logits,
        baseline_logits,
        tol_map,
        divergence_difference_tol,
        remove_shift,
        dynamic_threshold_config,
    )
    visualize_logit_results(results, output_dir)


def visualize_logit_results(
    results: List[List[Dict]],
    output_dir: str = "logit_validation_plots",
    max_tokens: int = 5,
    save_logits: bool = True,
) -> None:
    """Visualize logit validation results using plotly with comprehensive analysis.

    Args:
        results: List of batch results from logit_validation
        output_dir: Directory to save visualization files
        max_tokens: Maximum tokens to visualize per batch
        save_logits: Whether to save logit tensors

    Example:
        ```python
        passed, results = logit_validation(..., baseline_logits=baseline)
        visualize_logit_results(results, "my_plots")
        ```
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: plotly not available, skipping visualization")
        return

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch_results in enumerate(results):
        if not batch_results:
            continue

        # Extract data for visualization
        token_positions = list(range(len(batch_results)))
        error_map_data = {
            "5": {"token_ids": [], "divergences": []},
            "50": {"token_ids": [], "divergences": []},
            "1000": {"token_ids": [], "divergences": []},
            "all": {"token_ids": [], "divergences": []},
        }
        expected_top1_top2_diffs = []
        actual_top1_top2_diffs = []
        expected_top1_indices = []
        actual_top1_indices = []
        expected_top2_values = []
        actual_top2_values = []
        divergence_positions = []

        for token_idx, token_result in enumerate(batch_results):
            # Error map data
            for k in ["5", "50", "1000", "all"]:
                if k in token_result.get("error_map", {}):
                    error_map_data[k]["token_ids"].append(token_idx)
                    error_map_data[k]["divergences"].append(
                        token_result["error_map"][k]
                    )

            # Top1-Top2 differences
            expected_top1_top2_diffs.append(
                token_result.get("expected_top1_top2_relative_diff", 0)
            )
            actual_top1_top2_diffs.append(
                token_result.get("actual_top1_top2_relative_diff", 0)
            )

            # Top indices
            expected_indices = token_result.get("expected_top2_indices", [0, 0])
            actual_indices = token_result.get("actual_top2_indices", [0, 0])
            expected_top1_indices.append(
                expected_indices[0] if len(expected_indices) > 0 else 0
            )
            actual_top1_indices.append(
                actual_indices[0] if len(actual_indices) > 0 else 0
            )

            # Top2 values
            expected_vals = token_result.get("expected_top2_values", [0, 0])
            actual_vals = token_result.get("actual_top2_values", [0, 0])
            expected_top2_values.append(expected_vals)
            actual_top2_values.append(actual_vals)

            # Divergence positions
            if token_result.get("divergence", False):
                divergence_positions.append(token_idx)

        # Create plotly figure with 4 subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Error Map Analysis",
                "Relative Difference Analysis: Top1 logit - Top2 logit",
                "Top1 Indices Comparison",
                "Top2 Logit Values Analysis",
            ),
            vertical_spacing=0.15,
        )

        # Add error map traces
        colors = {"5": "red", "50": "blue", "1000": "green", "all": "purple"}
        for k, color in colors.items():
            if error_map_data[k]["token_ids"]:
                fig.add_trace(
                    go.Scatter(
                        x=error_map_data[k]["token_ids"],
                        y=error_map_data[k]["divergences"],
                        mode="markers",
                        name=f"Error Map K{k}",
                        marker=dict(size=6, color=color),
                    ),
                    row=1,
                    col=1,
                )

        # Add top1-top2 difference traces
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=expected_top1_top2_diffs,
                mode="lines+markers",
                name="Expected Top1-Top2 Diff",
                line=dict(color="darkblue", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=actual_top1_top2_diffs,
                mode="lines+markers",
                name="Actual Top1-Top2 Diff",
                line=dict(color="darkred", width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )

        # Add top1 indices traces
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=expected_top1_indices,
                mode="markers",
                name="Expected Top1 Indices",
                marker=dict(color="cyan", size=8, symbol="circle"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=actual_top1_indices,
                mode="markers",
                name="Actual Top1 Indices",
                marker=dict(color="magenta", size=8, symbol="x"),
            ),
            row=3,
            col=1,
        )

        # Add top2 values traces
        expected_top1_vals = [v[0] if len(v) > 0 else 0 for v in expected_top2_values]
        expected_top2_vals = [v[1] if len(v) > 1 else 0 for v in expected_top2_values]
        actual_top1_vals = [v[0] if len(v) > 0 else 0 for v in actual_top2_values]
        actual_top2_vals = [v[1] if len(v) > 1 else 0 for v in actual_top2_values]

        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=expected_top1_vals,
                mode="lines+markers",
                name="Expected Top1 Values",
                line=dict(color="blue", width=3),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=expected_top2_vals,
                mode="lines+markers",
                name="Expected Top2 Values",
                line=dict(color="lightblue", width=3, dash="dash"),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=actual_top1_vals,
                mode="lines+markers",
                name="Actual Top1 Values",
                line=dict(color="red", width=3),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=token_positions,
                y=actual_top2_vals,
                mode="lines+markers",
                name="Actual Top2 Values",
                line=dict(color="lightcoral", width=3, dash="dash"),
            ),
            row=4,
            col=1,
        )

        # Add divergence lines
        for div_pos in divergence_positions:
            for row in range(1, 5):
                fig.add_vline(
                    x=div_pos,
                    line_dash="dot",
                    line_color="orange",
                    line_width=2,
                    row=row,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            title=f"Logit Divergence Analysis - Batch {batch_idx}",
            height=1200,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        # Update axes
        for row in range(1, 5):
            fig.update_xaxes(title_text="Token Position", showgrid=True, row=row, col=1)
        fig.update_yaxes(title_text="Divergence Value", row=1, col=1)
        fig.update_yaxes(title_text="Relative Difference", row=2, col=1)
        fig.update_yaxes(title_text="Token Index", row=3, col=1)
        fig.update_yaxes(title_text="Logit Value", row=4, col=1)

        fig.write_html(
            os.path.join(output_dir, f"logit_analysis_b{batch_idx}.html"),
            include_plotlyjs=True,
        )

    print(f"Visualization saved to: {output_dir}")
