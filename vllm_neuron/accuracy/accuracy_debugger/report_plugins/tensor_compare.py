# SPDX-License-Identifier: Apache-2.0
"""Tensor compare report plugin - three-way hidden state comparison.

This plugin parses the comparison_summary.json produced by the TensorComparePlugin
prompt plugin. The comparison works as follows:

1. Three sets of intermediate tensors are captured for each prompt:
   - FP32: HuggingFace model in FP32 (ground truth)
   - BF16: HuggingFace model in BF16 (expected dtype noise)
   - Neuron: vllm-neuron model (target under test)

2. Per-module metrics are computed for both prefill and decode phases:
   - L-inf ratio: tgt_linf / base_linf (target vs baseline error ratio)
   - L2 ratio: tgt_l2 / base_l2 (target vs baseline L2 error ratio)
   - BC (Bhattacharyya Coefficient): statistical overlap between the
     error distributions of (BF16 vs FP32) and (Neuron vs FP32).

3. Pass/fail is determined by L2 ratio across all modules and phases.
   L2 ratio < max_l2_ratio (default 3.0) everywhere means the hidden state
   errors are within acceptable bounds. A module with anomalously high ratio
   suggests a computation bug in that module.
"""

import html as _html
import json
import os
from typing import Dict, List, Optional, Tuple

from .base import PluginRegistry, ReportPlugin


def _step_sort_key(item: tuple) -> int:
    try:
        return int(item[0])
    except (ValueError, TypeError):
        return 0


@PluginRegistry.register
class TensorComparePlugin(ReportPlugin):
    """Plugin for tensor compare analysis (step 3).

    Parses comparison_summary.json from the tensor_compare prompt plugin,
    which contains per-module L-inf ratio, L2 ratio, and BC for both
    prefill and decode phases.
    """

    name = "tensor_compare"
    display_name = "Tensor Compare"
    step_index = 3
    guide_text = """Three-way hidden state comparison: FP32 (ground truth) vs BF16 (expected) vs Neuron (target).
<ul>
<li><b>L2 ratio ≈ 1.0×</b> → Neuron error matches BF16 baseline (dtype noise, not a bug)</li>
<li><b>L2 ratio >> 1.5×</b> → Neuron has excess error beyond BF16 at that module</li>
<li><b>BC ≥ 0.90</b> → error distributions are similar (good)</li>
<li>Modules are shown in execution order — error propagation is visible top-to-bottom</li>
<li>Prefill and decode are compared separately since they exercise different code paths</li>
</ul>"""

    DEFAULT_L2_RATIO_THRESHOLD = 3.0
    WARN_RATIO = 1.5

    def get_log_path(self) -> Optional[str]:
        """Find comparison_summary.json in the tensor_compare subdirectory."""
        if not self.prompt_dir:
            return None
        path = os.path.join(
            self.prompt_dir, "tensor_compare", "comparison_summary.json"
        )
        if os.path.isfile(path):
            return path
        if self.capture_dir:
            path = os.path.join(
                self.capture_dir, "tensor_compare", "comparison_summary.json"
            )
            if os.path.isfile(path):
                return path
        return None

    def parse_log(self, log_path: str) -> Optional[Dict]:
        """Parse comparison_summary.json into structured data.

        Returns dict with structure:
            {
                "passed": bool,
                "prefill_summary": {prompt_key: {step: [{name, linf_ratio, l2_ratio, bc, passed}]}},
                "decode_summary": {prompt_key: {step: [{name, linf_ratio, l2_ratio, bc, passed}]}}
            }
        """
        if not os.path.isfile(log_path):
            return None
        try:
            with open(log_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
        if not data or "prefill_summary" not in data:
            return None
        return data

    def _collect_metrics(self, data: Dict) -> Tuple[float, float, int]:
        """Collect max L2 ratio, min BC, and module count from data."""
        max_l2 = 0.0
        min_bc = 1.0
        n_modules = 0
        for phase in ("prefill_summary", "decode_summary"):
            for prompt_data in data.get(phase, {}).values():
                for step_data in prompt_data.values():
                    for entry in step_data:
                        n_modules += 1
                        l2 = entry.get("l2_ratio", 0)
                        bc = entry.get("bc", 1.0)
                        if l2 > max_l2:
                            max_l2 = l2
                        if bc < min_bc:
                            min_bc = bc
        return max_l2, min_bc, n_modules

    def _get_threshold(self, data: Dict) -> float:
        """Get L2 ratio threshold from data or use default."""
        return data.get("max_l2_ratio_threshold", self.DEFAULT_L2_RATIO_THRESHOLD)

    def check_status(self, data: Dict) -> tuple[bool, str]:
        """Determine pass/fail based on overall passed flag and max L2 ratio."""
        passed = data.get("passed", False)
        max_l2, _, n_modules = self._collect_metrics(data)
        threshold = self._get_threshold(data)
        if passed:
            return True, f"Max L2 ratio: {max_l2:.4f} ({n_modules} modules)"
        return (
            False,
            f"Max L2 ratio: {max_l2:.4f} (> {threshold}, {n_modules} modules)",
        )

    def _collect_flagged_entries(self, data: Dict) -> List[Tuple[str, str, dict]]:
        """Collect entries with L2 ratio >= WARN_RATIO, sorted by L2 descending."""
        flagged = []
        for phase, label in [
            ("prefill_summary", "Prefill"),
            ("decode_summary", "Decode"),
        ]:
            for prompt_data in data.get(phase, {}).values():
                for step_data in prompt_data.values():
                    for entry in step_data:
                        if entry.get("l2_ratio", 0) >= self.WARN_RATIO:
                            flagged.append((label, entry["name"], entry))
        flagged.sort(key=lambda x: -x[2]["l2_ratio"])
        return flagged

    def build_html(self, data: Dict) -> str:
        """Build HTML fragment for the tensor compare section of the report."""
        parts = []
        threshold = self._get_threshold(data)

        passed = data.get("passed", False)
        max_l2, min_bc, n_modules = self._collect_metrics(data)

        cls = "pass" if passed else "fail"
        parts.append(
            f'<p>Max L2 ratio: <span class="{cls}">{max_l2:.4f}</span>, '
            f"Min BC: {min_bc:.4f} ({n_modules} modules, "
            f"threshold: {threshold}×)</p>"
        )

        # Flagged entries table (L2 ratio >= 1.5x)
        flagged = self._collect_flagged_entries(data)
        if flagged:
            parts.append(
                f'<details open><summary><b style="color:#e67700">'
                f"{len(flagged)} modules with L2 ratio &ge; {self.WARN_RATIO}×"
                f"</b></summary>"
            )
            parts.append('<table class="summary-table">')
            parts.append(
                "<tr><th>Phase</th><th>Module</th><th>L-inf Ratio</th>"
                "<th>L2 Ratio</th><th>BC</th></tr>"
            )
            for phase_label, name, entry in flagged:
                linf = entry.get("linf_ratio", 0)
                l2 = entry.get("l2_ratio", 0)
                bc = entry.get("bc", 0)
                l2_cls = "warn" if l2 < threshold else "fail"
                linf_cls = (
                    "pass"
                    if linf < self.WARN_RATIO
                    else ("warn" if linf < threshold else "fail")
                )
                bc_cls = "pass" if bc >= 0.95 else ("warn" if bc >= 0.9 else "fail")
                parts.append(
                    f"<tr><td>{phase_label}</td><td>{_html.escape(name)}</td>"
                    f'<td class="{linf_cls}">{linf:.4f}×</td>'
                    f'<td class="{l2_cls}">{l2:.4f}×</td>'
                    f'<td class="{bc_cls}">{bc:.4f}</td></tr>'
                )
            parts.append("</table></details>")

        # Full per-phase tables (collapsed)
        for phase, label in [
            ("prefill_summary", "Prefill"),
            ("decode_summary", "Decode"),
        ]:
            phase_data = data.get(phase, {})
            if not phase_data:
                continue

            n_phase = sum(
                len(step_data)
                for prompt_data in phase_data.values()
                for step_data in prompt_data.values()
            )
            parts.append(
                f"<details><summary><b>{label} Phase</b> ({n_phase} modules)</summary>"
            )
            for prompt_key, prompt_data in sorted(phase_data.items()):
                parts.append(f"<h4>{_html.escape(prompt_key)}</h4>")
                parts.append('<table class="summary-table">')
                parts.append(
                    "<tr><th>Step</th><th>Module</th><th>L-inf Ratio</th>"
                    "<th>L2 Ratio</th><th>BC</th><th>Status</th></tr>"
                )
                for step, step_data in sorted(prompt_data.items(), key=_step_sort_key):
                    for entry in step_data:
                        name = entry.get("name", "")
                        linf = entry.get("linf_ratio", 0)
                        l2 = entry.get("l2_ratio", 0)
                        bc = entry.get("bc", 0)
                        entry_passed = entry.get("passed", True)

                        linf_cls = (
                            "pass"
                            if linf < self.WARN_RATIO
                            else ("warn" if linf < threshold else "fail")
                        )
                        l2_cls = (
                            "pass"
                            if l2 < self.WARN_RATIO
                            else ("warn" if l2 < threshold else "fail")
                        )
                        bc_cls = (
                            "pass" if bc >= 0.95 else ("warn" if bc >= 0.9 else "fail")
                        )
                        status_icon = "✓" if entry_passed else "✗"
                        status_cls = "pass" if entry_passed else "fail"

                        parts.append(
                            f"<tr><td>{_html.escape(str(step))}</td>"
                            f"<td>{_html.escape(name)}</td>"
                            f'<td class="{linf_cls}">{linf:.4f}×</td>'
                            f'<td class="{l2_cls}">{l2:.4f}×</td>'
                            f'<td class="{bc_cls}">{bc:.4f}</td>'
                            f'<td class="{status_cls}">{status_icon}</td></tr>'
                        )
                parts.append("</table>")
            parts.append("</details>")

        return "\n".join(parts)

    def build_text_summary(self, data: Dict) -> str:
        """Build plain-text summary showing top offending modules."""
        lines = []
        passed, detail = self.check_status(data)
        lines.append(f"Tensor Compare: {'PASSED' if passed else 'FAILED'} — {detail}")
        flagged = self._collect_flagged_entries(data)
        if flagged:
            lines.append(
                f"  {len(flagged)} modules with L2 ratio >= {self.WARN_RATIO}x:"
            )
            for phase_label, name, entry in flagged[:10]:
                lines.append(
                    f"    [{phase_label}] {name}: "
                    f"L2={entry['l2_ratio']:.4f}, BC={entry['bc']:.4f}"
                )
            if len(flagged) > 10:
                lines.append(f"    ... and {len(flagged) - 10} more")
        return "\n".join(lines)
