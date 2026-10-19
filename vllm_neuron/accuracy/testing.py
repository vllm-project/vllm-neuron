# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for vLLM Neuron module accuracy validation.

Provides assertion functions that help engineers determine whether errors
are inherent to dtype quantization or specific to the Neuron target.

Two modes:

- **Two-way**: ``assert_close(actual, expected)`` — backward compatible with
  ``libtorch_neuronx_lite.testing.assert_close`` semantics (rtol normalized by abs_max).

- **Three-way**: ``assert_close_three_way(baseline, expected, actual)`` — isolates
  target-specific errors from dtype-inherent errors using dynamic thresholds.
  Answers: "Should I relax the threshold or file a bug?"

Example::

    from vllm_neuron.accuracy.testing import assert_close, assert_close_three_way

    # Two-way (drop-in replacement for libtorch_neuronx_lite.testing.assert_close)
    assert_close(neuron_output, hf_output)

    # Three-way (no magic rtol needed)
    assert_close_three_way(fp32_ref, bf16_cpu, bf16_neuron, name="attn.layer0")
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import logging
import numpy as np
import torch

if TYPE_CHECKING:
    # Type-only: the runtime pooling task name (Literal) the engine routes on.
    from vllm.tasks import SupportedTask

logger = logging.getLogger(__name__)

from vllm_neuron.accuracy.tensor_compare import (
    _compute_bc,
    compare_tensors_three_way,
)

# Default tolerances matching neuron_allclose / torch.testing semantics
_DEFAULT_DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}


@dataclass
class AssertCloseResult:
    """Result from assert_close with neuron_allclose-compatible fields."""

    allclose: bool
    num_mismatches: int
    max_rel_error: float
    max_abs_error: float
    max_rel_error_index: tuple
    max_abs_error_index: tuple
    linf_rel: float = 0.0
    l2_rel: float = 0.0


@dataclass
class ThreeWayAssertResult:
    """Result from assert_close_three_way with diagnostic info.

    Fields:
        passed: Whether the assertion passed (BC >= threshold or σ-ratio <= 1.0).
        name: Descriptive label for the comparison.
        base_linf: L-inf of |expected - baseline| (reference error).
        tgt_linf: L-inf of |actual - baseline| (target error).
        base_l2: L2 norm of |expected - baseline|.
        tgt_l2: L2 norm of |actual - baseline|.
        linf_ratio: tgt_linf / base_linf (worst-case across inputs if multi-input).
        l2_ratio: tgt_l2 / base_l2 (worst-case across inputs if multi-input).
        sigma_ratio: RMS(tgt_errors) / RMS(base_errors), aggregated across all inputs.
            < 1.0 means target is more accurate than reference.
        bc: Bhattacharyya Coefficient of error distributions, aggregated.
            1.0 = identical, >= 0.99 = excellent, < 0.95 = divergent.
        n_inputs: Number of input tensors aggregated.
    """

    passed: bool
    name: str
    base_linf: float
    tgt_linf: float
    base_l2: float
    tgt_l2: float
    linf_ratio: float  # worst-case across inputs
    l2_ratio: float  # worst-case across inputs
    sigma_ratio: float  # aggregated RMS(tgt_errors) / RMS(base_errors)
    bc: float  # aggregated BC
    n_inputs: int = 1

    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [f"{status} {self.name}"]
        if self.n_inputs > 1:
            lines.append(f"  inputs: {self.n_inputs}")
            lines.append(
                f"  worst L-inf ratio: {self.linf_ratio:.2f}x"
                f"  worst L2 ratio: {self.l2_ratio:.2f}x"
            )
        else:
            lines.append(
                f"  L-inf: base={self.base_linf:.6f}  tgt={self.tgt_linf:.6f}"
                f"  ratio={self.linf_ratio:.2f}x"
            )
            lines.append(
                f"  L2/σ:  base={self.base_l2:.6f}  tgt={self.tgt_l2:.6f}"
                f"  ratio={self.l2_ratio:.2f}x"
            )
        lines.append(f"  σ-ratio: {self.sigma_ratio:.3f}  BC: {self.bc:.4f}")
        return "\n".join(lines)


@dataclass
class CosineThreeWayAssertResult:
    """Result from assert_pooling_close_three_way(task="embed") with diagnostic info.

    Cosine-space analogue of ThreeWayAssertResult for pooling/embedding outputs,
    where only the DIRECTION of the vector matters, not absolute element magnitudes.
    The pass decision is a per-vector absolute cosine floor;
    the BF16-HF "expected" drift is reported for context but does NOT gate.

    Fields:
        passed: Whether every row's cos(baseline, actual) >= min_cos.
        name: Descriptive label for the comparison.
        n_inputs: Number of embedding rows (vectors) compared.
        min_tgt_cos: Worst-row cos(baseline, actual) — the headline number.
        mean_tgt_cos: Mean cos(baseline, actual) across rows.
        min_base_cos: Worst-row cos(baseline, expected) — the BF16-HF dtype
            floor, reported for context only.
        mean_base_cos: Mean cos(baseline, expected) across rows.
        min_cos: The absolute cosine floor the test gated on.
    """

    passed: bool
    name: str
    n_inputs: int
    min_tgt_cos: float
    mean_tgt_cos: float
    min_base_cos: float
    mean_base_cos: float
    min_cos: float

    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [f"{status} {self.name}"]
        lines.append(f"  vectors: {self.n_inputs}  (floor min_cos={self.min_cos:.5f})")
        lines.append(
            f"  cos(base, actual):   min={self.min_tgt_cos:.6f}"
            f"  mean={self.mean_tgt_cos:.6f}"
        )
        lines.append(
            f"  cos(base, expected): min={self.min_base_cos:.6f}"
            f"  mean={self.mean_base_cos:.6f}  (BF16-HF floor, not gated)"
        )
        return "\n".join(lines)


def _neuron_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = None,
    atol: float = None,
    equal_nan_inf: bool = False,
) -> AssertCloseResult:
    """Core comparison matching neuron_allclose semantics.

    Threshold formula::

        abs_diff = |actual - expected|
        rel_error = (abs_diff - atol) / max(|expected|)
        allclose = all(rel_error <= rtol)

    This differs from torch.allclose which uses per-element denominators.
    Here rtol is normalized by the global abs_max of expected.
    """
    if torch.equal(expected, actual):
        idx = torch.unravel_index(torch.tensor(0), expected.shape)
        return AssertCloseResult(
            allclose=True,
            num_mismatches=0,
            max_rel_error=0.0,
            max_abs_error=0.0,
            max_rel_error_index=idx,
            max_abs_error_index=idx,
        )

    if rtol is None or atol is None:
        default = _DEFAULT_DTYPE_TOLERANCE.get(expected.dtype, (1.6e-2, 1e-5))
        rtol = rtol if rtol is not None else default[0]
        atol = atol if atol is not None else default[1]

    actual_f, expected_f = actual.float(), expected.float()
    abs_diff = torch.abs(actual_f - expected_f)

    # Handle matching inf (same sign) as zero difference
    matching_inf = (
        torch.isinf(actual_f)
        & torch.isinf(expected_f)
        & (torch.sign(actual_f) == torch.sign(expected_f))
    )
    abs_diff = torch.where(matching_inf, torch.zeros_like(abs_diff), abs_diff)

    # Compute expected_abs_max excluding non-finite values
    finite_mask = torch.isfinite(expected_f)
    if finite_mask.any():
        expected_abs_max = torch.max(torch.abs(expected_f[finite_mask]))
    else:
        expected_abs_max = torch.tensor(0.0)

    if torch.is_nonzero(expected_abs_max):
        rel_error = (abs_diff - atol) / expected_abs_max
    else:
        rel_error = torch.full(expected_f.shape, float("inf"))

    close = rel_error <= rtol

    # If equal_nan_inf, treat matching NaN and matching inf as close
    if equal_nan_inf:
        matching_nan = torch.isnan(actual_f) & torch.isnan(expected_f)
        close = close | matching_nan | matching_inf
    allclose = close.all().item()
    num_mismatches = (~close).sum().item()
    max_rel_idx = torch.unravel_index(torch.argmax(rel_error), rel_error.shape)
    max_abs_idx = torch.unravel_index(torch.argmax(abs_diff), abs_diff.shape)

    linf_rel = (
        abs_diff.max().item() / expected_abs_max.item()
        if expected_abs_max > 0
        else float("inf")
    )
    l2_rel = (
        torch.norm(abs_diff).item() / torch.norm(expected_f).item()
        if torch.norm(expected_f) > 0
        else 0
    )

    return AssertCloseResult(
        allclose=allclose,
        num_mismatches=num_mismatches,
        max_rel_error=rel_error[max_rel_idx].item(),
        max_abs_error=abs_diff[max_abs_idx].item(),
        max_rel_error_index=max_rel_idx,
        max_abs_error_index=max_abs_idx,
        linf_rel=linf_rel,
        l2_rel=l2_rel,
    )


def assert_close(
    actual: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    rtol: float = None,
    atol: float = None,
    equal_nan_inf: bool = False,
    name: str = "tensor",
    enable_histograms: bool = False,
) -> AssertCloseResult:
    """Two-way assertion compatible with libtorch_neuronx_lite.testing.assert_close.

    Uses neuron_allclose semantics: rtol normalized by abs_max of expected.
    Supports nested lists/tuples of tensors (compared element-wise).

    Args:
        actual: Target tensor (or nested list/tuple of tensors).
        expected: Reference tensor (or nested list/tuple of tensors).
        rtol: Relative tolerance. Defaults to dtype-specific value.
        atol: Absolute tolerance. Defaults to dtype-specific value.
        equal_nan_inf: If True, matching NaN values and matching infinity
            values (same sign) are treated as equal.
        name: Label for error messages.
        enable_histograms: prints a diagnostic report
            (histograms, stats table, and 2D mismatch map) for each tensor pair
            on both pass and failure. If False (default), no diagnostic report is printed.

    Raises:
        AssertionError with diagnostic info (L-inf, L2, mismatch count).
    """
    pairs = _flatten_tensor_pairs(actual, expected, name)
    failures = []
    last_result = None

    for act, exp, pair_name in pairs:
        result = _neuron_allclose(
            act, exp, rtol=rtol, atol=atol, equal_nan_inf=equal_nan_inf
        )
        last_result = result

        # Resolve effective tolerances for the diagnostic report (matches
        # _neuron_allclose default-resolution logic).
        default = _DEFAULT_DTYPE_TOLERANCE.get(exp.dtype, (1.6e-2, 1e-5))
        eff_rtol = rtol if rtol is not None else default[0]
        eff_atol = atol if atol is not None else default[1]

        if enable_histograms:
            try:
                from vllm_neuron.accuracy.tensor_histogram import TensorHistogram

                TensorHistogram().print_full_comparison_report(
                    actual=act,
                    expected=exp,
                    name=pair_name,
                    atol=eff_atol,
                    rtol=eff_rtol,
                    passed=result.allclose,
                    enable_histograms=True,
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning(
                    "Could not generate tensor histogram for %s: %s",
                    pair_name,
                    exc,
                )

        if not result.allclose:
            failures.append(
                f"  {pair_name}: {result.num_mismatches} mismatches, "
                f"max_rel={result.max_rel_error:.6f}, max_abs={result.max_abs_error:.6f}, "
                f"L-inf_rel={result.linf_rel:.6f}, L2_rel={result.l2_rel:.6f}"
            )

    if failures:
        raise AssertionError(
            f"assert_close failed for {len(failures)}/{len(pairs)} tensor(s):\n"
            + "\n".join(failures)
        )

    return last_result


def _floored_ratio(tgt: float, base: float, floor: float) -> float:
    """Ratio of target error to reference error with a denominator floor.

    With ``floor == 0.0`` this matches plain ratio semantics, including the
    zero-denominator convention (0/0 -> 0.0, x/0 -> inf for x > 0).
    """
    eff_base = max(base, floor)
    if eff_base > 0:
        return tgt / eff_base
    return 0.0 if tgt == 0 else float("inf")


def assert_close_three_way(
    baseline: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    actual: Union[torch.Tensor, list],
    name: str = "tensor",
    bc_threshold: float = 0.99,
    max_linf_ratio: float = 5.0,
    max_l2_ratio: float = 3.0,
    base_linf_floor: float = 0.0,
    base_l2_floor: float = 0.0,
    plot_on_failure: bool = False,
    output_dir: str = None,
) -> ThreeWayAssertResult:
    """Three-way assertion that isolates target errors from dtype errors.

    Compares:
    - baseline (FP32) vs expected (same-dtype on CPU) → dtype error
    - baseline (FP32) vs actual (same-dtype on Neuron) → target error

    Passes if the Bhattacharyya Coefficient (BC) of the two error distributions
    is >= ``bc_threshold``, or if σ-ratio ≤ 1.0 (target more accurate than baseline).

    Additionally, hard guards ensure no extreme element-wise deviations:
    - worst L-inf ratio must be < ``max_linf_ratio`` (default 5.0)
    - worst L2 ratio must be < ``max_l2_ratio`` (default 3.0)

    The guard denominators (per-input reference errors) are sample statistics
    of bf16 rounding and can be heavy-tailed: an input where the reference
    happens to round nearly exactly yields a near-zero denominator, turning an
    ordinary target error into an extreme ratio. ``base_linf_floor`` and
    ``base_l2_floor`` clamp the per-input denominators from below (in the same
    relative units as the errors: fraction of the baseline's abs-max / L2
    norm) so the guards compare against a representative reference-error
    scale instead of a lucky draw. A floor of 0.0 (default) preserves exact
    ratio semantics.

    When ``plot_on_failure=True`` (default), automatically generates error
    distribution and QQ-plot diagnostics on failure.

    Metrics reported:
        - **σ-ratio**: Ratio of target RMS error to baseline RMS error,
          aggregated across all inputs. σ-ratio ≤ 1.0 means the target is
          more accurate than the dtype baseline.
        - **BC** (Bhattacharyya Coefficient): Overlap between target and
          baseline error distributions (0–1). BC ≥ 0.99 means nearly
          identical distributions.
        - **worst L-inf ratio**: Maximum per-input ratio of target max
          absolute error to baseline max absolute error. Catches extreme
          element-wise outliers.
        - **worst L2 ratio**: Maximum per-input ratio of target L2 error
          to baseline L2 error. Catches systematic per-sample degradation.

    Raises:
        AssertionError with diagnostic summary.
    """
    base_list = _to_tensor_list(baseline)
    exp_list = _to_tensor_list(expected)
    act_list = _to_tensor_list(actual)
    assert len(base_list) == len(exp_list) == len(act_list)

    all_base_errors, all_tgt_errors = [], []
    worst_linf_ratio = worst_l2_ratio = 0.0

    for i, (b, e, a) in enumerate(zip(base_list, exp_list, act_list)):
        tag = f"{name}[{i}]" if len(base_list) > 1 else name
        r = compare_tensors_three_way(b, e, a, name=tag)
        worst_linf_ratio = max(
            worst_linf_ratio, _floored_ratio(r.tgt_linf, r.base_linf, base_linf_floor)
        )
        worst_l2_ratio = max(
            worst_l2_ratio, _floored_ratio(r.tgt_l2, r.base_l2, base_l2_floor)
        )
        if r.base_errors is not None:
            all_base_errors.append(r.base_errors)
            all_tgt_errors.append(r.tgt_errors)

    n_inputs = len(base_list)

    if all_base_errors:
        cat_base = np.concatenate(all_base_errors)
        cat_tgt = np.concatenate(all_tgt_errors)
        agg_bc = _compute_bc(cat_base, cat_tgt)
        base_rms = np.sqrt(np.mean(cat_base**2))
        tgt_rms = np.sqrt(np.mean(cat_tgt**2))
        sigma_ratio = tgt_rms / base_rms if base_rms > 0 else float("inf")
    else:
        agg_bc = 0.0
        sigma_ratio = float("inf")

    # Pass if BC >= threshold or target is more accurate than baseline
    # Hard guards: extreme element-wise deviations always fail regardless of BC/σ
    linf_guard_ok = worst_linf_ratio < max_linf_ratio
    l2_guard_ok = worst_l2_ratio < max_l2_ratio
    passed = (
        (agg_bc >= bc_threshold or sigma_ratio <= 1.0) and linf_guard_ok and l2_guard_ok
    )

    result = ThreeWayAssertResult(
        passed=passed,
        name=name,
        base_linf=r.base_linf,
        tgt_linf=r.tgt_linf,
        base_l2=r.base_l2,
        tgt_l2=r.tgt_l2,
        linf_ratio=worst_linf_ratio,
        l2_ratio=worst_l2_ratio,
        sigma_ratio=sigma_ratio,
        bc=agg_bc,
        n_inputs=n_inputs,
    )

    # Always print summary for visibility in test output
    logger.info(f"\n{result.summary()}")

    if not passed:
        if 0.95 <= agg_bc < bc_threshold:
            logger.info(
                f"  → distributions similar but below threshold (BC={agg_bc:.4f}), review plots"
            )
        if not linf_guard_ok:
            logger.info(
                f"  → hard guard: worst L-inf ratio {worst_linf_ratio:.2f}x exceeds limit {max_linf_ratio:.1f}x"
            )
        if not l2_guard_ok:
            logger.info(
                f"  → hard guard: worst L2 ratio {worst_l2_ratio:.2f}x exceeds limit {max_l2_ratio:.1f}x"
            )
        if plot_on_failure and all_base_errors:
            _generate_failure_plots(
                np.concatenate(all_base_errors),
                np.concatenate(all_tgt_errors),
                name,
                output_dir,
            )
        raise AssertionError(f"\n{result.summary()}")
    return result


def _rows_of(x) -> List[torch.Tensor]:
    """Flatten inputs to a list of 1-D embedding vectors (rows)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    rows: List[torch.Tensor] = []
    for t in _to_tensor_list(x):
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        t = t.float()
        if t.ndim <= 1:
            rows.append(t.reshape(-1))
        else:
            rows.extend(t.reshape(t.shape[0], -1))
    return rows


def _assert_embed_close_three_way(
    baseline: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    actual: Union[torch.Tensor, list],
    name: str = "embedding",
    min_cos: float = 0.999,
    eps: float = 1e-8,
) -> CosineThreeWayAssertResult:
    """Per-vector cosine assertion for pooling / embedding model outputs.

    Private per-task handler invoked by :func:`assert_pooling_close_three_way`
    for ``task="embed"``; tests call the dispatcher, not this directly.

    Three inputs are accepted to mirror the element-wise API and to report the
    dtype floor for context:

    - ``baseline``: FP32 reference embedding(s) — ground truth direction.
    - ``expected``: BF16-HF embedding(s) — the dtype-quantization floor.
    - ``actual``:   BF16-on-Neuron embedding(s) — the thing under test.

    A single ``[H]`` vector, a batch ``[N, H]`` (compared as N rows), or a list
    of either are all accepted. Cosine is scale-invariant, so inputs need not be
    L2-normalized (the real pooler output already is).

    Pass rule: ``min over rows of cos(baseline, actual) >= min_cos``. The
    ``expected`` (BF16-HF) cosine is reported for context but does NOT gate —
    in cosine space dtype drift is ~1e-6, so a target/dtype ratio (the element-
    wise ``sigma_ratio`` analogue) is noise-dominated and not a useful
    discriminator; the absolute floor is what matters for embedding quality.

    .. note::
       ``min_cos=0.999`` (≈2.6°) is PROVISIONAL — picked to match the proposal's
       "cosine similarity > 0.99" bar with margin, not yet calibrated against
       measured Qwen3-Embedding-8B BF16-Neuron drift. Tighten once a Neuron
       baseline is established (cf. the MTEB e2e ``MTEB_TOL`` note).

    Raises:
        AssertionError with a diagnostic summary if any row falls below min_cos.
    """

    def _cos(a_rows, b_rows):
        vals = []
        for a, b in zip(a_rows, b_rows):
            na = torch.linalg.vector_norm(a)
            nb = torch.linalg.vector_norm(b)
            if na < eps or nb < eps:
                # A zero-norm vector has no direction — cannot be "aligned".
                vals.append(float("nan"))
            else:
                vals.append((torch.dot(a, b) / (na * nb)).item())
        return vals

    base_rows = _rows_of(baseline)
    exp_rows = _rows_of(expected)
    act_rows = _rows_of(actual)
    assert len(base_rows) == len(exp_rows) == len(act_rows), (
        f"row count mismatch: baseline={len(base_rows)} expected={len(exp_rows)} "
        f"actual={len(act_rows)}"
    )

    tgt_cos = _cos(base_rows, act_rows)
    base_cos = _cos(base_rows, exp_rows)
    n_inputs = len(base_rows)

    # nan (zero-norm row) is treated as a failure: it cannot meet the floor.
    tgt_arr = np.array(tgt_cos, dtype=float)
    base_arr = np.array(base_cos, dtype=float)
    min_tgt_cos = (
        float(np.nanmin(tgt_arr)) if np.isfinite(tgt_arr).any() else float("nan")
    )
    mean_tgt_cos = (
        float(np.nanmean(tgt_arr)) if np.isfinite(tgt_arr).any() else float("nan")
    )
    min_base_cos = (
        float(np.nanmin(base_arr)) if np.isfinite(base_arr).any() else float("nan")
    )
    mean_base_cos = (
        float(np.nanmean(base_arr)) if np.isfinite(base_arr).any() else float("nan")
    )

    # Any non-finite (zero-norm / nan) target row fails; else gate on the floor.
    has_bad_row = not np.isfinite(tgt_arr).all()
    passed = (not has_bad_row) and (min_tgt_cos >= min_cos)

    result = CosineThreeWayAssertResult(
        passed=passed,
        name=name,
        n_inputs=n_inputs,
        min_tgt_cos=min_tgt_cos,
        mean_tgt_cos=mean_tgt_cos,
        min_base_cos=min_base_cos,
        mean_base_cos=mean_base_cos,
        min_cos=min_cos,
    )

    logger.info(f"\n{result.summary()}")

    if not passed:
        if has_bad_row:
            logger.info("  → at least one row has zero/non-finite norm (no direction)")
        else:
            logger.info(
                f"  → worst row cos {min_tgt_cos:.6f} below floor {min_cos:.5f}"
            )
        raise AssertionError(f"\n{result.summary()}")
    return result


def assert_pooling_close_three_way(
    baseline: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    actual: Union[torch.Tensor, list],
    task: "SupportedTask" = "embed",
    name: str = "pooling",
    **task_kwargs,
):
    """Pooling-model accuracy assertion, dispatched by pooling task.

    Single entry point for validating any pooling-model output against the right per-task metric.
    Each task is judged by the metric appropriate to how its output is consumed:

    - ``"embed"``  → per-vector **cosine direction** floor
      (:func:`_assert_embed_close_three_way`). Embeddings are ranked by cosine
      similarity, so only direction matters. Pass ``min_cos`` (and optionally
      ``eps``) via ``task_kwargs``. Returns :class:`CosineThreeWayAssertResult`.
    - ``"classify"`` → per-element **absolute probability difference** (proposal
      bar: Neuron classification probabilities must match HF within ``0.01``).
      **Not implemented yet** — raises :class:`NotImplementedError`. It lands
      with the classifier-head bring-up (the embedding bring-up has no classify
      head, so there are no classify probabilities to assert against today; see
      the pooling scope-boundary note). The atol-based metric and a
      ``ClassifyThreeWayAssertResult`` belong in that PR, where they are testable
      end-to-end.

    Args:
        baseline: FP32 reference output(s) — ground truth.
        expected: BF16-HF output(s) — the dtype-quantization floor.
        actual:   BF16-on-Neuron output(s) — the thing under test.
        task: Pooling task name (``SupportedTask``); ``"embed"`` or ``"classify"``.
        name: Descriptive label for the comparison.
        **task_kwargs: Forwarded to the task's assertion (e.g. ``min_cos`` /
            ``eps`` for embed).

    Raises:
        NotImplementedError: for tasks without an assertion yet (e.g. classify).
        ValueError: for a task this API does not handle.
        AssertionError: if the task's metric gate fails.
    """
    if task == "embed":
        return _assert_embed_close_three_way(
            baseline, expected, actual, name=name, **task_kwargs
        )
    if task == "classify":
        raise NotImplementedError(
            "assert_pooling_close_three_way(task='classify') is not implemented "
            "yet. The classify metric (per-element |Neuron - HF| <= atol over "
            "softmax probabilities) lands with the classifier-head bring-up, "
            "where there are real classify outputs to validate end-to-end. "
        )
    raise ValueError(
        f"assert_pooling_close_three_way: unsupported pooling task {task!r} "
        f"(handled: 'embed'; planned: 'classify')"
    )


def _generate_failure_plots(
    base_errors: np.ndarray,
    tgt_errors: np.ndarray,
    name: str,
    output_dir: str = None,
) -> None:
    """Generate a single combined diagnostic plot on three-way assertion failure."""
    try:
        from vllm_neuron.accuracy.plotting import (
            plot_error_distributions,
            plot_error_qqplot,
        )
        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = output_dir or "accuracy_debug"
        safe_name = name.replace("/", "_").replace(" ", "_")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_error_distributions(base_errors, tgt_errors, name=name, ax=ax1)
        plot_error_qqplot(base_errors, tgt_errors, name=name, ax=ax2)
        plt.suptitle(name, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{safe_name}_debug.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved debug plot to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate debug plots: {e}")


def _to_tensor_list(x) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_to_tensor_list(item))
        return out
    raise TypeError(f"Expected tensor or sequence, got {type(x)}")


def _flatten_tensor_pairs(actual, expected, prefix=""):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return [(actual, expected, prefix)]
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch: {len(actual)} vs {len(expected)}")
        pairs = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            pairs.extend(_flatten_tensor_pairs(a, e, f"{prefix}[{i}]"))
        return pairs
    raise TypeError(f"Type mismatch: {type(actual)} vs {type(expected)}")
