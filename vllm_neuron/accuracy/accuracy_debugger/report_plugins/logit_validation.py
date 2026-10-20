# SPDX-License-Identifier: Apache-2.0
"""Logit validation plugin - three-way comparison FP32 vs BF16 vs vllm-neuron."""

import os
from typing import Dict, Optional

from .base import ReportPlugin, PluginRegistry
from .utils import build_logit_html, parse_logit_text


@PluginRegistry.register
class LogitValidationPlugin(ReportPlugin):
    """Plugin for logit validation (step 0)."""

    name = "logit_val"
    display_name = "Logit Validation"
    step_index = 0
    guide_text = """Three-way logit comparison: FP32 (ground truth) vs BF16 (expected) vs Neuron (target).
<ul>
<li><b>Ratio ≈ 1.0x</b> → Neuron error matches BF16 baseline (dtype-inherent, not a bug)</li>
<li><b>Ratio >> 1.0x</b> → Neuron has excess error beyond BF16 (potential vllm-neuron bug)</li>
<li><b>BC ≥ 0.99</b> → error distributions nearly identical (good)</li>
<li><b>Orange dashed lines</b> mark divergent tokens (argmax differs from reference)</li>
<li>Check the threshold table: K5/K50/K1000/All show max relative error per top-k bucket</li>
</ul>"""

    def parse_log(self, log_path: str) -> Optional[Dict]:
        if not os.path.isfile(log_path):
            return None
        with open(log_path) as f:
            text = self.strip_ansi(f.read())
        return parse_logit_text(text)

    def get_log_path(self) -> Optional[str]:
        """Try step-specific log first, then validation_log.txt as fallback."""
        if not self.prompt_dir:
            return None
        path = os.path.join(self.prompt_dir, f"step{self.step_index}_{self.name}.txt")
        if os.path.isfile(path):
            return path
        fallback = os.path.join(self.prompt_dir, "validation_log.txt")
        return fallback if os.path.isfile(fallback) else None

    def load_artifacts(self, artifact_dir: str) -> Optional[Dict]:
        """Load logit_analysis_b0.html from the logit_validation/ subdirectory."""
        for base in [artifact_dir, self.prompt_dir]:
            if not base:
                continue
            path = os.path.join(base, "logit_validation", "logit_analysis_b0.html")
            if os.path.isfile(path):
                return {"html_files": [path]}
        return None

    def check_status(self, data: Dict) -> tuple[bool, str]:
        passed = data.get("passed", False)
        errs = data.get("max_errors", {})
        verdict = data.get("three_way_verdict", {})
        failed = [k for k, v in errs.items() if not v["passed"]]
        if passed:
            if failed and verdict:
                return True, "Passed (three-way: error within BF16 noise)"
            return True, "All thresholds met"
        if failed:
            return False, f"Failed: {', '.join(failed)}"
        return False, "Failed"

    def build_html(self, data: Dict) -> str:
        return build_logit_html(data)

    def build_text_summary(self, data: Dict) -> str:
        lines = []
        passed, detail = self.check_status(data)
        lines.append(f"Logit Validation: {'PASSED' if passed else 'FAILED'} — {detail}")
        for k, v in data.get("max_errors", {}).items():
            status = "✓" if v["passed"] else "✗"
            lines.append(
                f"  {k}: {v['value']:.4f} (threshold: {v['threshold']:.4f}) {status}"
            )
        rows = data.get("three_way", [])
        if rows:
            diverged = [r["token"] for r in rows if r["divergent"]]
            lines.append(f"  Tokens: {len(rows)}, Diverged: {len(diverged)}")
        return "\n".join(lines)
