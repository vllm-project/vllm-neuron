# SPDX-License-Identifier: Apache-2.0
"""Task analysis plugin - visualizes task-level evaluation results."""

import html as _html
import json
import os
from typing import Dict, Optional

from .base import ReportPlugin, PluginRegistry


def _render_sample_row(doc_id, entry: dict, include_worse: bool = False) -> str:
    """Render a single sample as an HTML table row."""
    question = _html.escape(entry.get("doc", {}).get("question", ""))
    expected = _html.escape(entry.get("correct_answer", ""))
    got = _html.escape(str((entry.get("target", {}).get("filtered_resps") or [""])[0]))
    full_expected = _html.escape(entry.get("doc", {}).get("answer", ""))
    resps = entry.get("target", {}).get("resps", [[]])
    full_resp = _html.escape(str(resps[0][0]) if resps and resps[0] else "")
    row = (
        f"<tr><td>{_html.escape(str(doc_id))}</td>"
        f"<td><div class='cell-wrap'>{question}</div></td>"
        f"<td><div class='cell-wrap'>{expected}</div></td>"
        f"<td><div class='cell-wrap'>{got}</div></td>"
        f"<td><div class='cell-wrap'>{full_expected}</div></td>"
        f"<td><div class='cell-wrap'>{full_resp}</div></td>"
    )
    if include_worse:
        worse = _html.escape(", ".join(entry.get("worse_metrics", [])))
        row += f"<td>{worse}</td>"
    row += "</tr>"
    return row


@PluginRegistry.register
class TaskAnalysisPlugin(ReportPlugin):
    """Plugin for task analysis (evaluation results comparison)."""

    name = "task_analysis"
    display_name = "Task Analysis"
    step_index = 10
    guide_text = """Task-level evaluation results comparing reference and target model outputs.
<ul>
<li><b>Deviations</b> = samples where target model accuracy is worse than reference</li>
<li><b>Match%</b> = percentage of samples where target matches reference behavior</li>
<li>Click on task names to expand and see individual deviating samples</li>
</ul>"""

    @property
    def log_filename(self) -> str:
        return "results_summary.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_log_path = None

    def _results_dir(self) -> Optional[str]:
        """Resolve the top-level results directory."""
        if self._last_log_path:
            # task_analysis/results_summary.json -> go up 2 levels
            return os.path.dirname(os.path.dirname(self._last_log_path))
        if self.prompt_dir:
            return os.path.dirname(os.path.dirname(self.prompt_dir.rstrip("/")))
        return None

    def _load_task_status(self) -> Optional[Dict]:
        """Load task_status.json (scores, thresholds, passed) if available."""
        if not self._last_log_path:
            return None
        path = os.path.join(os.path.dirname(self._last_log_path), "task_status.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _load_run_config(self) -> Optional[Dict]:
        """Load run_config.json from the results directory."""
        results_dir = self._results_dir()
        if not results_dir:
            return None
        path = os.path.join(results_dir, "run_config.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def get_log_path(self) -> Optional[str]:
        """Task analysis uses task_analysis/results_summary.json."""
        if not self.prompt_dir:
            return None
        parent = os.path.dirname(self.prompt_dir.rstrip("/"))
        path = os.path.join(parent, "task_analysis", "results_summary.json")
        return path if os.path.isfile(path) else None

    def parse_log(self, log_path: str) -> Optional[Dict]:
        self._last_log_path = log_path
        if not os.path.isfile(log_path):
            return None
        try:
            with open(log_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None

    def check_status(self, data: Dict) -> tuple[bool, str]:
        status = self._load_task_status()
        if status:
            passed = status.get("passed", True)
            thresholds = status.get("thresholds", {})
            scores = status.get("scores", {})
            total_samples = sum(
                td.get("summary_metrics", {}).get("total_comparisons", 0)
                for td in data.values()
            )
            total_devs = sum(
                td.get("summary_metrics", {}).get("deviations_with_worse_accuracy", 0)
                for td in data.values()
            )
            parts = []
            for metric, threshold in thresholds.items():
                score = scores.get(metric)
                if score is not None:
                    icon = "✓" if score >= threshold else "✗"
                    parts.append(
                        f"{icon} {metric}: {score:.3f} (threshold: {threshold:.3f})"
                    )
            detail = f"{total_samples} samples, {total_devs} deviated"
            if parts:
                detail = f"{'; '.join(parts)} — {detail}"
            return passed, detail

        # Fallback: no task_status.json
        total_devs = sum(
            td.get("summary_metrics", {}).get("deviations_with_worse_accuracy", 0)
            for td in data.values()
        )
        total_samples = sum(
            td.get("summary_metrics", {}).get("total_comparisons", 0)
            for td in data.values()
        )
        return True, f"{total_samples} samples, {total_devs} deviated"

    def build_html(self, data: Dict) -> str:
        parts = []

        # Run config table — show only the task_analysis section
        rc = self._load_run_config()
        if rc:
            task_cfg = rc.get("task_analysis", {})
            rows = [
                (k, str(v))
                for k, v in task_cfg.items()
                if v is not None and v != {} and v != []
            ]
            if rows:
                parts.append("<h2>Run Config</h2>")
                parts.append(
                    '<table class="summary-table"><tr><th>Parameter</th><th>Value</th></tr>'
                )
                for k, v in rows:
                    parts.append(
                        f"<tr><td>{_html.escape(k)}</td><td>{_html.escape(v)}</td></tr>"
                    )
                parts.append("</table>")

        # Load thresholds for the metrics table
        status = self._load_task_status()
        thresholds = status.get("thresholds", {}) if status else {}

        # Metrics summary table
        parts.append("<h2>Metrics Summary</h2>")
        parts.append('<table class="summary-table">')
        parts.append(
            "<tr><th>Task</th><th>Metric</th><th>Score</th><th>Threshold</th><th>Ref Score</th>"
            "<th>Samples</th><th>Matching</th><th>Deviating</th><th>Match%</th></tr>"
        )

        for task, td in data.items():
            summary = td.get("summary_metrics", {})
            total = summary.get("total_comparisons", 0)
            matches = summary.get("matches", 0)
            devs = summary.get("deviations_with_worse_accuracy", 0)
            pct = f"{matches / total * 100:.1f}%" if total > 0 else "N/A"

            target_metrics = {
                k: v
                for k, v in td.get("target_metrics", {}).items()
                if "stderr" not in k
            }
            ref_metrics = {
                k: v for k, v in td.get("ref_metrics", {}).items() if "stderr" not in k
            }

            if not target_metrics:
                parts.append(
                    f'<tr><td>{_html.escape(task)}</td><td colspan="8">No metrics available</td></tr>'
                )
                continue

            first = True
            for k, v in target_metrics.items():
                metric_name = k.split("-", 1)[1] if "-" in k else k
                ref_score = ref_metrics.get(k, "")
                ref_str = f"{ref_score:.3f}" if isinstance(ref_score, float) else "—"
                threshold_val = thresholds.get(metric_name)
                threshold_str = (
                    f"{threshold_val:.3f}" if threshold_val is not None else "—"
                )
                meets = v >= threshold_val if threshold_val is not None else True
                score_cls = "pass" if meets else "fail"

                if first:
                    parts.append(
                        f'<tr><td rowspan="{len(target_metrics)}">{task}</td>'
                        f"<td>{metric_name}</td>"
                        f'<td class="{score_cls}">{v:.3f}</td>'
                        f"<td>{threshold_str}</td><td>{ref_str}</td>"
                        f'<td rowspan="{len(target_metrics)}">{total}</td>'
                        f'<td rowspan="{len(target_metrics)}">{matches}</td>'
                        f'<td rowspan="{len(target_metrics)}">{devs}</td>'
                        f'<td rowspan="{len(target_metrics)}">{pct}</td></tr>'
                    )
                    first = False
                else:
                    parts.append(
                        f"<td>{metric_name}</td>"
                        f'<td class="{score_cls}">{v:.3f}</td>'
                        f"<td>{threshold_str}</td><td>{ref_str}</td></tr>"
                    )

        parts.append("</table>")

        # Per-task deviating/matching samples
        for task, td in data.items():
            deviating = td.get("deviating", {})
            matching = td.get("matching", {})
            if not deviating and not matching:
                continue

            parts.append(f"<h2>{_html.escape(task)}</h2>")

            if deviating:
                parts.append(
                    f'<details open><summary><b style="color:#c92a2a">⚠ {len(deviating)} Deviating</b></summary>'
                )
                parts.append('<table class="summary-table">')
                parts.append(
                    "<tr><th>Doc ID</th><th>Question</th><th>Expected</th><th>Actual</th>"
                    "<th>Full Expected</th><th>Full Response</th><th>Worse Metrics</th></tr>"
                )
                for doc_id, entry in deviating.items():
                    parts.append(_render_sample_row(doc_id, entry, include_worse=True))
                parts.append("</table></details>")

            if matching:
                parts.append(
                    f'<details><summary><b style="color:#2b8a3e">✓ {len(matching)} Matching</b></summary>'
                )
                parts.append('<table class="summary-table">')
                parts.append(
                    "<tr><th>Doc ID</th><th>Question</th><th>Expected</th><th>Actual</th>"
                    "<th>Full Expected</th><th>Full Response</th></tr>"
                )
                for doc_id, entry in list(matching.items())[:50]:
                    parts.append(_render_sample_row(doc_id, entry))
                if len(matching) > 50:
                    parts.append(
                        f'<tr><td colspan="6" style="color:#868e96">… {len(matching) - 50} more</td></tr>'
                    )
                parts.append("</table></details>")

        return "\n".join(parts)

    def build_text_summary(self, data: Dict) -> str:
        lines = []
        passed, detail = self.check_status(data)
        lines.append(f"Task Analysis: {'PASSED' if passed else 'FAILED'} — {detail}")
        for task_name, td in data.items():
            sm = td.get("summary_metrics", {})
            total = sm.get("total_comparisons", 0)
            devs = sm.get("deviations_with_worse_accuracy", 0)
            matches = sm.get("matches", 0)
            lines.append(f"  {task_name}: {matches}/{total} correct, {devs} deviations")
        return "\n".join(lines)
