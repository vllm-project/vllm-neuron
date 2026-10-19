# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for report plugins."""

import re
from typing import Dict, Optional


def parse_logit_text(text: str) -> Optional[Dict]:
    """Parse three-way logit validation log text."""
    rows = []
    for m in re.finditer(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*\|\s*([\d.]+)\s*\|?\s*(\*?)",
        text,
        re.MULTILINE,
    ):
        rows.append(
            {
                "token": int(m.group(1)),
                "base_linf": float(m.group(2)),
                "tgt_linf": float(m.group(3)),
                "linf_ratio": float(m.group(4)),
                "base_l2": float(m.group(5)),
                "tgt_l2": float(m.group(6)),
                "l2_ratio": float(m.group(7)),
                "bc": float(m.group(8)),
                "divergent": m.group(9) == "*",
            }
        )

    passed = "Overall Status: PASSED" in text
    max_errors = {}
    for m in re.finditer(
        r"- (K\d+|All|Divergence): ([\d.]+) \(threshold: ([\d.]+)\) ([✓✗])", text
    ):
        max_errors[m.group(1)] = {
            "value": float(m.group(2)),
            "threshold": float(m.group(3)),
            "passed": m.group(4) == "✓",
        }

    # Parse three-way aggregate results
    three_way_verdict = {}
    agg_bc_m = re.search(r"agg_bc:\s*(PASS|FAIL)", text)
    if agg_bc_m:
        three_way_verdict["agg_bc"] = agg_bc_m.group(1) == "PASS"
    agg_l2_m = re.search(r"agg_l2_1\.5x:\s*(PASS|FAIL)", text)
    if agg_l2_m:
        three_way_verdict["agg_l2_1.5x"] = agg_l2_m.group(1) == "PASS"
    agg_linf_m = re.search(r"agg_linf_1\.5x:\s*(PASS|FAIL)", text)
    if agg_linf_m:
        three_way_verdict["agg_linf_1.5x"] = agg_linf_m.group(1) == "PASS"
    sigma_m = re.search(r"agg_sigma_ratio:\s*([\d.]+)", text)
    if sigma_m:
        three_way_verdict["agg_sigma_ratio"] = float(sigma_m.group(1))
    # Check if two-way validation was overridden by three-way
    two_way_m = re.search(r"Two-way validation:\s*(\d+)/(\d+)", text)
    if two_way_m:
        three_way_verdict["two_way_passed"] = int(two_way_m.group(1))
        three_way_verdict["two_way_total"] = int(two_way_m.group(2))

    return (
        {
            "three_way": rows,
            "passed": passed,
            "max_errors": max_errors,
            "three_way_verdict": three_way_verdict,
        }
        if (rows or max_errors)
        else None
    )


def build_logit_html(data: Dict) -> str:
    """Build HTML for logit validation data (shared by sanity and logit plugins)."""
    from .base import ReportPlugin

    parts = []

    # Embed existing plotly HTML files
    html_files = data.get("html_files", [])
    if html_files:
        if len(html_files) == 1:
            with open(html_files[0]) as f:
                parts.append(ReportPlugin.extract_body(f.read()))
        else:
            section_id = f"prompt-sel-{id(data)}"
            options = "".join(
                f'<option value="{i}">Batch {i}</option>'
                for i in range(len(html_files))
            )
            parts.append(
                f'<div style="margin-bottom:12px"><label><b>Batch:</b> '
                f'<select onchange="switchPrompt_{section_id}(this.value)" '
                f'style="padding:4px 8px;font-size:14px">{options}</select></label></div>'
            )
            for i, html_path in enumerate(html_files):
                with open(html_path) as f:
                    body = ReportPlugin.extract_body(f.read())
                display = "block" if i == 0 else "none"
                parts.append(
                    f'<div id="{section_id}-{i}" style="display:{display}">{body}</div>'
                )
            parts.append(f"""<script>
function switchPrompt_{section_id}(idx) {{
  for (var i = 0; i < {len(html_files)}; i++) {{
    var el = document.getElementById('{section_id}-' + i);
    if (el) el.style.display = (i == parseInt(idx)) ? 'block' : 'none';
  }}
  window.dispatchEvent(new Event('resize'));
}}
</script>""")

    rows = data.get("three_way", [])
    if not rows and not data.get("max_errors"):
        return "\n".join(parts) if parts else "<p>No logit validation data.</p>"

    # Summary with both two-way and three-way verdicts
    passed = data.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    status_cls = "pass" if passed else "fail"
    summary_html = (
        f'<h2>Overall Verdict: <span class="{status_cls}">{status}</span></h2>'
    )

    three_way_verdict = data.get("three_way_verdict", {})
    # Compute aggregate values from per-token rows
    if rows and three_way_verdict:
        max_linf_ratio = max(r["linf_ratio"] for r in rows)
        max_l2_ratio = max(r["l2_ratio"] for r in rows)
        min_bc = min(r["bc"] for r in rows)
        mean_bc = sum(r["bc"] for r in rows) / len(rows)
        three_way_verdict["max_linf_ratio"] = max_linf_ratio
        three_way_verdict["max_l2_ratio"] = max_l2_ratio
        three_way_verdict["min_bc"] = min_bc
        three_way_verdict["mean_bc"] = mean_bc
    if three_way_verdict or rows:
        summary_html += '<div class="guide"><b>How pass/fail is determined:</b><br>'
        summary_html += (
            "The overall verdict uses <b>three-way comparison</b> (FP32 → BF16 → Neuron). "
            "If the aggregate σ-ratio ≤ 1.0 or aggregate BC passes, the test passes "
            "even if two-way thresholds are exceeded — because the error is within "
            "dtype-inherent BF16 noise.<br><br>"
            "<b>Two-way thresholds</b> (K5/K50/K1000/All below) are a static fallback. "
            "<b>Three-way metrics</b> (ratio, BC) are the primary validation.</div>"
        )
        if three_way_verdict:
            agg_bc = three_way_verdict.get("agg_bc")
            agg_l2 = three_way_verdict.get("agg_l2_1.5x")
            agg_linf = three_way_verdict.get("agg_linf_1.5x")
            sigma_ratio = three_way_verdict.get("agg_sigma_ratio")
            two_pass = three_way_verdict.get("two_way_passed")
            two_total = three_way_verdict.get("two_way_total")
            summary_html += '<table class="summary-table"><tr><th>Check</th><th>Value</th><th>Threshold</th><th>Result</th></tr>'
            if sigma_ratio is not None:
                cls = "pass" if sigma_ratio <= 1.0 else "fail"
                summary_html += (
                    f"<tr><td>σ-ratio (aggregate)</td>"
                    f"<td>{sigma_ratio:.4f}</td>"
                    f"<td>≤ 1.0</td>"
                    f'<td class="{cls}">{"PASS" if sigma_ratio <= 1.0 else "FAIL"}</td></tr>'
                )
            min_bc = three_way_verdict.get("min_bc")
            max_linf_ratio = three_way_verdict.get("max_linf_ratio")
            max_l2_ratio = three_way_verdict.get("max_l2_ratio")
            if agg_bc is not None:
                cls = "pass" if agg_bc else "fail"
                val = f"{min_bc:.4f}" if min_bc is not None else "—"
                summary_html += (
                    f"<tr><td>Aggregate BC (min)</td>"
                    f"<td>{val}</td>"
                    f"<td>≥ 0.99</td>"
                    f'<td class="{cls}">{"PASS" if agg_bc else "FAIL"}</td></tr>'
                )
            if agg_linf is not None:
                cls = "pass" if agg_linf else "fail"
                val = f"{max_linf_ratio:.2f}×" if max_linf_ratio is not None else "—"
                summary_html += (
                    f"<tr><td>Aggregate L-inf ratio (max)</td>"
                    f"<td>{val}</td>"
                    f"<td>&lt; 1.5×</td>"
                    f'<td class="{cls}">{"PASS" if agg_linf else "FAIL"}</td></tr>'
                )
            if agg_l2 is not None:
                cls = "pass" if agg_l2 else "fail"
                val = f"{max_l2_ratio:.2f}×" if max_l2_ratio is not None else "—"
                summary_html += (
                    f"<tr><td>Aggregate L2 ratio (max)</td>"
                    f"<td>{val}</td>"
                    f"<td>&lt; 1.5×</td>"
                    f'<td class="{cls}">{"PASS" if agg_l2 else "FAIL"}</td></tr>'
                )
            if two_pass is not None:
                cls = "pass" if two_pass == two_total else "fail"
                summary_html += (
                    f"<tr><td>Two-way threshold (static fallback)</td>"
                    f"<td>{two_pass}/{two_total} prompts</td>"
                    f"<td>All prompts pass</td>"
                    f'<td class="{cls}">{"PASS" if two_pass == two_total else "FAIL"}</td></tr>'
                )
            summary_html += "</table>"

    if data.get("max_errors"):
        summary_html += "<h3>Two-Way Thresholds (static)</h3>"
        summary_html += '<table class="summary-table"><tr><th>Metric</th><th>Max Value</th><th>Threshold</th><th>Status</th></tr>'
        for k, v in data["max_errors"].items():
            cls = "pass" if v["passed"] else "fail"
            summary_html += f'<tr><td>{k}</td><td>{v["value"]:.4f}</td><td>{v["threshold"]:.4f}</td><td class="{cls}">{"✓" if v["passed"] else "✗"}</td></tr>'
        summary_html += "</table>"

    if not rows:
        return summary_html + "\n".join(parts)

    # Three-way metrics chart
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tokens = [r["token"] for r in rows]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "L-inf: Base vs Target",
            "L2: Base vs Target",
            "Bhattacharyya Coefficient",
        ),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=[r["base_linf"] for r in rows],
            name="Base L-inf",
            mode="markers",
            marker=dict(color="blue", size=6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=[r["tgt_linf"] for r in rows],
            name="Tgt L-inf",
            mode="markers",
            marker=dict(color="red", size=6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=[r["base_l2"] for r in rows],
            name="Base L2",
            mode="markers",
            marker=dict(color="blue", size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=[r["tgt_l2"] for r in rows],
            name="Tgt L2",
            mode="markers",
            marker=dict(color="red", size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=[r["bc"] for r in rows],
            name="BC",
            mode="markers",
            marker=dict(color="green", size=6),
        ),
        row=3,
        col=1,
    )

    for dt in [r["token"] for r in rows if r["divergent"]]:
        for row in range(1, 4):
            fig.add_vline(
                x=dt, line_dash="dot", line_color="orange", line_width=2, row=row, col=1
            )

    fig.update_layout(height=800, template="plotly_white", hovermode="x unified")
    for row in range(1, 4):
        fig.update_xaxes(title_text="Generation Token", row=row, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    bc_values = [r["bc"] for r in rows]
    bc_min = min(bc_values) if bc_values else 0.98
    bc_range_min = min(bc_min - 0.01, 0.98)
    fig.update_yaxes(title_text="BC", range=[bc_range_min, 1.0], row=3, col=1)

    three_way_html = "<h2>Three-Way Metrics</h2>" + fig.to_html(
        full_html=False, include_plotlyjs=False
    )

    # Per-token table
    three_way_html += (
        "<details><summary><b>Per-token three-way comparison table</b></summary>"
        '<table class="summary-table">'
        "<tr><th>Token</th><th>Base L-inf</th><th>Tgt L-inf</th><th>Ratio</th>"
        "<th>Base L2</th><th>Tgt L2</th><th>Ratio</th><th>BC</th><th>Div</th></tr>"
    )
    for r in rows:
        ratio_cls = (
            "pass"
            if r["linf_ratio"] <= 1.5
            else ("warn" if r["linf_ratio"] <= 2.0 else "fail")
        )
        bc_cls = "pass" if r["bc"] >= 0.99 else ("warn" if r["bc"] >= 0.90 else "fail")
        div_mark = "✗" if r["divergent"] else ""
        three_way_html += (
            f"<tr><td>{r['token']}</td>"
            f"<td>{r['base_linf']:.4f}</td><td>{r['tgt_linf']:.4f}</td>"
            f'<td class="{ratio_cls}">{r["linf_ratio"]:.2f}x</td>'
            f"<td>{r['base_l2']:.4f}</td><td>{r['tgt_l2']:.4f}</td>"
            f'<td class="{ratio_cls}">{r["l2_ratio"]:.2f}x</td>'
            f'<td class="{bc_cls}">{r["bc"]:.4f}</td>'
            f"<td>{div_mark}</td></tr>"
        )
    three_way_html += "</table></details>"

    return summary_html + "\n".join(parts) + three_way_html
