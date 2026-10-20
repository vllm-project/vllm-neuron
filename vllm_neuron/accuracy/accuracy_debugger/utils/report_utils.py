# SPDX-License-Identifier: Apache-2.0
"""Composable HTML report builders for accuracy debugger.

Each builder produces an HTML fragment (string). They can be used
independently or composed via ``generate_report`` / ``render_html``.

Example — full combined report::

    from vllm_neuron.accuracy.accuracy_debugger.utils.report_utils import generate_report
    generate_report("./accuracy_report", "./report.html")

Example — individual sections::

    from vllm_neuron.accuracy.accuracy_debugger.utils.report_utils import (
        build_task_report_html,
        build_prompt_report_html,
        build_overview_html,
        render_html,
    )

    task_html = build_task_report_html("./accuracy_report")
    prompt_html = build_prompt_report_html("./accuracy_report/prompt_analysis/prompt_0")
    overview_html = build_overview_html("./accuracy_report")

    render_html(
        tabs=[("Overview", overview_html), ("Task", task_html), ("Prompt 0", prompt_html)],
        output="./my_report.html",
    )
"""

import glob
import html as _html
import json
import os
from typing import Dict

from vllm_neuron.accuracy.accuracy_debugger.report_plugins import (
    KVAnalysisPlugin,
    LogitValidationPlugin,
    TaskAnalysisPlugin,
    TensorComparePlugin,
)
from vllm_neuron.accuracy.accuracy_debugger.report_plugins.base import PluginResult

PROMPT_PLUGINS = [LogitValidationPlugin, KVAnalysisPlugin, TensorComparePlugin]

REPORT_CSS = """\
body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
.tabs { display: flex; gap: 4px; border-bottom: 2px solid #dee2e6; margin-bottom: 20px; flex-wrap: wrap; }
.tab { padding: 10px 24px; cursor: pointer; border-radius: 6px 6px 0 0; background: #e9ecef; font-size: 14px; }
.tab.active { background: #fff; border: 1px solid #dee2e6; border-bottom: 2px solid #fff; margin-bottom: -2px; }
.tab-content { display: none; background: #fff; padding: 20px; border-radius: 0 0 6px 6px; }
.tab-content.active { display: block; }
.summary-table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
.summary-table th, .summary-table td { border: 1px solid #dee2e6; padding: 6px 10px; }
.summary-table th { background: #f1f3f5; }
.cell-wrap { max-width: 500px; white-space: pre-wrap; word-break: break-word; }
.pass { color: #2b8a3e; font-weight: 600; }
.fail { color: #c92a2a; font-weight: 600; }
.warn { color: #e67700; font-weight: 600; }
.info { color: #1971c2; font-weight: 600; }
.guide { background: #f1f3f5; border-left: 4px solid #868e96; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; border-radius: 0 4px 4px 0; }
.guide summary { cursor: pointer; font-weight: 600; }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.overview-card { border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; background: #fff; }
.overview-card h3 { margin: 0 0 8px 0; font-size: 14px; }
.status-banner { padding: 12px 20px; border-radius: 8px; margin-bottom: 16px; font-size: 16px; font-weight: 600; }
.status-pass { background: #d3f9d8; color: #2b8a3e; border: 1px solid #b2f2bb; }
.status-warn { background: #fff3bf; color: #e67700; border: 1px solid #ffe066; }
.status-fail { background: #ffe3e3; color: #c92a2a; border: 1px solid #ffc9c9; }
.status-info { background: #d0ebff; color: #1971c2; border: 1px solid #a5d8ff; }
.triage { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 16px; font-family: monospace; font-size: 12px; white-space: pre; line-height: 1.5; overflow-x: auto; }
details { margin-bottom: 12px; }
details > summary { cursor: pointer; }
details > summary > h2 { margin: 8px 0; }
pre { white-space: pre-wrap; word-break: break-all; }
.triage pre { white-space: pre; word-break: normal; }
.prompt-browser { display: flex; gap: 16px; min-height: 400px; }
.prompt-list { width: 260px; flex-shrink: 0; border: 1px solid #dee2e6; border-radius: 6px; overflow-y: auto; max-height: 80vh; background: #fff; }
.prompt-list-item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #f1f3f5; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.prompt-list-item:hover { background: #e7f5ff; }
.prompt-list-item.active { background: #d0ebff; font-weight: 600; }
.prompt-list-item .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.prompt-list-item .status-dot.pass { background: #2b8a3e; }
.prompt-list-item .status-dot.fail { background: #c92a2a; }
.prompt-list-item .status-dot.warn { background: #e67700; }
.prompt-detail { flex: 1; border: 1px solid #dee2e6; border-radius: 6px; padding: 20px; background: #fff; overflow-y: auto; max-height: 80vh; }
.prompt-detail-panel { display: none; }
.prompt-detail-panel.active { display: block; }
.section-header { font-size: 15px; font-weight: 600; margin: 16px 0 8px 0; color: #495057; }"""

REPORT_JS = """\
function switchTab(idx) {
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab-content').forEach((c, i) => c.classList.toggle('active', i === idx));
  window.dispatchEvent(new Event('resize'));
}
function switchPrompt(idx) {
  document.querySelectorAll('.prompt-list-item').forEach((t, i) => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.prompt-detail-panel').forEach((c, i) => c.classList.toggle('active', i === idx));
  window.dispatchEvent(new Event('resize'));
}"""

TRIAGE_FLOWCHART = """\
Task Analysis (eval scores)
│
├─ PASSED → model accuracy meets thresholds, done
│
└─ FAILED → deviated prompts found, check Prompt Analysis
    │
    Prompt Analysis: Logit Validation (three-way)
    │
    ├─ σ-ratio ≤ 1.0 or BC ≥ 0.99
    │   └─ Dtype-inherent error (BF16 noise), not a vllm-neuron bug
    │
    ├─ Token 0 ratio >> 1.5×
    │   └─ Prefill bug → check KV Cache and Tensor Compare
    │
    ├─ Tokens 1+ ratio >> 1.5×, token 0 OK
    │   └─ Decode/KV bug → check KV Cache for high L-inf at early tokens
    │
    Prompt Analysis: KV Cache (per-layer BC)
    │
    ├─ All layers BC ≥ 0.9
    │   └─ KV writes are correct, check Tensor Compare for hidden state issues
    │
    └─ Specific layer BC << 0.9
        └─ Bug in that layer's KV write path
    │
    Prompt Analysis: Tensor Compare (per-module hidden states)
    │
    ├─ All modules L2 ratio < 1.5×
    │   └─ Hidden states match BF16 baseline, issue is elsewhere
    │
    ├─ Specific module L2 ratio >> 1.5×
    │   └─ Error introduced at that module — check computation path
    │
    └─ Error grows across layers (increasing L2 ratio)
        └─ Error accumulation — first high-ratio module is root cause"""


# ── Per-prompt report ─────────────────────────────────────────────────────────


def _run_prompt_plugins(prompt_dir: str) -> Dict[str, PluginResult]:
    """Run all prompt plugins on a single prompt directory."""
    captures = os.path.join(prompt_dir, "captures")
    capture_dir = captures if os.path.isdir(captures) else prompt_dir
    results = {}
    for plugin_cls in PROMPT_PLUGINS:
        plugin = plugin_cls(prompt_dir, capture_dir)
        results[plugin.display_name] = plugin.run()
    return results


def build_prompt_report_html(prompt_dir: str) -> str:
    """Build HTML for a single prompt's validation results.

    Args:
        prompt_dir: Path to a prompt directory (e.g. ``prompt_analysis/prompt_0``).

    Returns:
        HTML fragment with logit/KV/tensor sections.
    """
    captures = os.path.join(prompt_dir, "captures")
    capture_dir = captures if os.path.isdir(captures) else prompt_dir
    parts = []

    prompt_file = os.path.join(prompt_dir, "prompt.txt")
    if os.path.isfile(prompt_file):
        with open(prompt_file) as f:
            prompt_text = f.read().strip()
            # Truncate long prompts
            display = prompt_text[:200] + "…" if len(prompt_text) > 200 else prompt_text
            parts.append(
                f"<details><summary><b>Prompt text</b></summary>"
                f'<pre style="white-space:pre-wrap;font-size:12px">{_html.escape(display)}</pre></details>'
            )

    for plugin_cls in PROMPT_PLUGINS:
        plugin = plugin_cls(prompt_dir, capture_dir)
        res = plugin.run()
        if res.data:
            status = "✅" if res.passed else "❌" if res.passed is False else "—"
            open_attr = "" if res.passed else " open"
            parts.append(
                f"<details{open_attr}><summary><h2 style='display:inline'>"
                f"{status} {plugin.display_name}: {res.summary}</h2></summary>"
            )
            parts.append(plugin.wrap_with_guide(res.html))
            parts.append("</details>")

    return "\n".join(parts)


# ── Task report ───────────────────────────────────────────────────────────────


def build_task_report_html(results_dir: str) -> str | None:
    """Build HTML for task analysis results.

    Args:
        results_dir: Root results directory containing ``task_analysis/results_summary.json``.

    Returns:
        HTML fragment, or None if no task analysis data found.
    """
    ta_path = os.path.join(results_dir, "task_analysis", "results_summary.json")
    if not os.path.isfile(ta_path):
        return None
    plugin = TaskAnalysisPlugin()
    result = plugin.run(log_path=ta_path)
    if not result.data:
        return None
    return plugin.wrap_with_guide(result.html)


# ── Overview report ───────────────────────────────────────────────────────────


def build_overview_html(
    results_dir: str,
    prompt_results: list[tuple[str, Dict[str, PluginResult]]] | None = None,
    task_result: PluginResult | None = None,
    run_config: dict | None = None,
) -> str:
    """Build overview HTML with status summary, prompt cards, and triage guide.

    If ``prompt_results`` or ``task_result`` are not provided, they are
    auto-detected from ``results_dir``.

    Args:
        results_dir: Root results directory.
        prompt_results: Pre-computed prompt plugin results (optional).
        task_result: Pre-computed task plugin result (optional).
        run_config: Run configuration dict (optional, loaded from run_config.json).

    Returns:
        HTML fragment for the overview tab.
    """
    # Auto-load run_config
    if run_config is None:
        rc_path = os.path.join(results_dir, "run_config.json")
        if os.path.isfile(rc_path):
            with open(rc_path) as f:
                run_config = json.load(f)

    # Auto-detect prompt results
    if prompt_results is None:
        prompt_results = _collect_prompt_results(results_dir)

    # Auto-detect task result
    if task_result is None:
        ta_path = os.path.join(results_dir, "task_analysis", "results_summary.json")
        if os.path.isfile(ta_path):
            task_result = TaskAnalysisPlugin().run(log_path=ta_path)

    return _build_overview_content(prompt_results, task_result, run_config, results_dir)


def _collect_prompt_results(
    results_dir: str,
) -> list[tuple[str, Dict[str, PluginResult]]]:
    """Scan prompt_analysis/ and run plugins on each prompt directory."""
    prompt_analysis_dir = os.path.join(results_dir, "prompt_analysis")
    if not os.path.isdir(prompt_analysis_dir):
        return []

    prompt_results = []
    for i, pd in enumerate(
        sorted(
            d
            for d in glob.glob(os.path.join(prompt_analysis_dir, "prompt_*"))
            if os.path.isdir(d)
        )
    ):
        prompt_file = os.path.join(pd, "prompt.txt")
        if os.path.isfile(prompt_file):
            with open(prompt_file) as f:
                label = f"Prompt {i}: {_html.escape(f.read().strip()[:60])}…"
        else:
            label = f"Prompt {i}"
        prompt_results.append((label, _run_prompt_plugins(pd)))
    return prompt_results


def _build_overview_content(
    prompt_results: list[tuple[str, Dict[str, PluginResult]]],
    task_result: PluginResult | None,
    run_config: dict | None,
    results_dir: str,
) -> str:
    """Build the overview HTML content."""
    parts = []

    # ── Run config (model only) ─────────────────────────────────────────
    if run_config:
        prompt_cfg = run_config.get("prompt_analysis", {})
        task_cfg = run_config.get("task_analysis", {})
        model = prompt_cfg.get("model_checkpoint") or task_cfg.get("model") or ""
        if model:
            parts.append(
                f'<p style="font-size:13px; color:#495057"><b>Model:</b> <code>{model}</code></p>'
            )

    # ── Overall status (severity-aware) ─────────────────────────────────
    all_pass = True
    any_data = False
    fail_count = 0
    total_checks = 0
    for _, pr in prompt_results:
        for res in pr.values():
            if res.passed is not None:
                any_data = True
                total_checks += 1
                if not res.passed:
                    all_pass = False
                    fail_count += 1
    if task_result and task_result.passed is not None:
        any_data = True
        total_checks += 1
        if not task_result.passed:
            all_pass = False
            fail_count += 1

    if any_data:
        if all_pass:
            parts.append(
                '<div class="status-banner status-pass">✅ All checks passed</div>'
            )
        elif fail_count <= 2 and total_checks > 2:
            parts.append(
                f'<div class="status-banner status-warn">⚠️ Minor deviations detected ({fail_count} of {total_checks} checks)</div>'
            )
        else:
            parts.append(
                f'<div class="status-banner status-fail">❌ Accuracy issues detected ({fail_count} of {total_checks} checks failed)</div>'
            )

    # ── Guide ─────────────────────────────────────────────────────────────
    parts.append(
        '<details class="guide" open><summary>How to read this report</summary>'
        "<p>The accuracy debugger runs a multi-step pipeline:</p>"
        "<ol>"
        "<li><b>Task Evaluation</b> — run lm_eval benchmarks (e.g. GSM8K) against the vLLM server</li>"
        "<li><b>Task Analysis</b> — compare scores against user-defined thresholds to determine pass/fail</li>"
        "<li><b>Extract Deviated Samples</b> — identify prompts where the model's answer differs from the correct answer</li>"
        "<li><b>Prompt Analysis</b> — for each deviated sample, run token-level logit validation and KV cache comparison "
        "to pinpoint where the model diverges from the HF reference</li>"
        "</ol>"
        "<p>Use the <b>Task Analysis</b> tab for score details and the <b>Prompt Analysis</b> tab to drill into individual samples.</p>"
        "</details>"
    )

    # ── Task Analysis section ─────────────────────────────────────────────
    parts.append('<h2 style="margin-top:24px">Task Analysis</h2>')
    if task_result and task_result.passed is not None:
        icon = "✅" if task_result.passed else "❌"
        border = "#2b8a3e" if task_result.passed else "#c92a2a"
        parts.append(
            f'<div class="overview-card" style="border-left:4px solid {border}; margin-bottom:16px">'
            f'<span style="font-size:13px">{icon} {task_result.summary}</span></div>'
        )
    else:
        parts.append('<p style="color:#868e96">No task analysis data available.</p>')

    # ── Prompt Analysis section ───────────────────────────────────────────
    parts.append('<h2 style="margin-top:24px">Prompt Analysis</h2>')
    if prompt_results:
        n_prompts = len(prompt_results)
        n_pass = sum(
            1
            for _, pr in prompt_results
            if all(r.passed is not False for r in pr.values())
        )
        parts.append(
            f'<p style="font-size:13px; color:#495057">'
            f"{n_prompts} deviated samples analyzed — {n_pass} passed, "
            f"{n_prompts - n_pass} with issues</p>"
        )
        cards = []
        for prompt_label, pr in prompt_results:
            step_lines = []
            prompt_pass = True
            for name, res in pr.items():
                if res.passed is None:
                    icon = "—"
                elif res.passed:
                    icon = "✅"
                else:
                    icon = "❌"
                    prompt_pass = False
                step_lines.append(f"{icon} {name}: {res.summary}")

            border = "#2b8a3e" if prompt_pass else "#c92a2a"
            steps_html = "<br>".join(
                f'<span style="font-size:12px">{line}</span>' for line in step_lines
            )
            cards.append(
                f'<div class="overview-card" style="border-left:4px solid {border}">'
                f"<h3>{prompt_label}</h3>{steps_html}</div>"
            )
        parts.append(f'<div class="overview-grid">{"".join(cards)}</div>')
    else:
        parts.append('<p style="color:#868e96">No prompt analysis data available.</p>')

    # ── Triage ────────────────────────────────────────────────────────────
    parts.append(
        '<details><summary><h2 style="display:inline">Triage Flowchart</h2></summary>'
        f'<div class="triage">{TRIAGE_FLOWCHART}</div></details>'
    )

    return "\n".join(parts)


# ── HTML rendering ────────────────────────────────────────────────────────────


def render_html(
    tabs: list[tuple[str, str]],
    output: str,
    title: str = "Accuracy Validation Report",
) -> str:
    """Render a list of (tab_name, html_content) into a complete HTML document.

    Args:
        tabs: List of (name, html_fragment) tuples for each tab.
        output: Path to write the HTML file.
        title: Page title.

    Returns:
        Path to the written HTML file.
    """
    tab_headers = "\n".join(
        f'<div class="tab{" active" if i == 0 else ""}" onclick="switchTab({i})">{name}</div>'
        for i, (name, _) in enumerate(tabs)
    )
    tab_contents = "\n".join(
        f'<div class="tab-content{" active" if i == 0 else ""}" id="tab-{i}">{html}</div>'
        for i, (_, html) in enumerate(tabs)
    )

    try:
        import plotly

        plotly_script = f"<script>{plotly.offline.get_plotlyjs()}</script>"
    except ImportError:
        plotly_script = (
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )

    html = (
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{title}</title>\n'
        f"{plotly_script}\n<style>\n{REPORT_CSS}\n</style></head>\n<body>\n"
        f"<h1>{title}</h1>\n"
        f'<div class="tabs">{tab_headers}</div>\n'
        f"{tab_contents}\n"
        f"<script>\n{REPORT_JS}\n</script>\n"
        "</body></html>"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    print(f"Report saved to: {output}")
    return output


def _build_prompt_browser(
    prompt_dirs: list[str], run_config: dict | None = None
) -> str:
    """Build a sidebar list + detail panel for prompt analysis results."""
    parts = []

    # Guide
    parts.append(
        '<details class="guide" open><summary>How to read this tab</summary>'
        "<p>Deviated samples (prompts where the model answered incorrectly) are analyzed with:</p>"
        "<ul>"
        "<li><b>Logit Validation</b> — three-way token-by-token comparison (HF FP32 vs HF BF16 vs vLLM Neuron). "
        "If the three-way ratio ≈ 1.0x, the deviation is BF16 noise, not a bug.</li>"
        "<li><b>KV Cache</b> — per-layer cache comparison. Low Bhattacharyya Coefficient (BC) indicates a KV write issue.</li>"
        "<li><b>Tensor Compare</b> — per-module hidden state comparison (prefill + decode). "
        "High L2 ratio at a specific module pinpoints where excess error is introduced.</li>"
        "</ul></details>"
    )

    # Reproduce
    parts.append(
        '<details class="guide"><summary>Reproduce</summary>'
        "<p><b>Full pipeline:</b></p>"
        "<pre>pytest test/vllm_neuron/model/llama3/e2e/test_accuracy_debugger.py -s</pre>"
        "<p><b>Prompt analysis only (requires existing task results):</b></p>"
        "<pre>pytest test/vllm_neuron/model/llama3/e2e/test_accuracy_debugger.py -s -k prompt_only</pre>"
        "</details>"
    )

    # Show prompt analysis config
    if run_config:
        prompt_cfg = run_config.get("prompt_analysis", {})
        rows = [
            (k, str(v))
            for k, v in prompt_cfg.items()
            if v is not None and v != {} and v != []
        ]
        if rows:
            table = (
                '<table class="summary-table"><tr><th>Config</th><th>Value</th></tr>'
            )
            for k, v in rows:
                table += (
                    f"<tr><td>{_html.escape(k)}</td><td>{_html.escape(v)}</td></tr>"
                )
            table += "</table>"
            parts.append(table)

    list_items = []
    detail_panels = []

    for i, pd in enumerate(prompt_dirs):
        # Read prompt text
        prompt_file = os.path.join(pd, "prompt.txt")
        prompt_text = ""
        if os.path.isfile(prompt_file):
            with open(prompt_file) as f:
                prompt_text = f.read().strip()

        # Run plugins to determine pass/fail
        plugin_results = _run_prompt_plugins(pd)
        prompt_pass = all(r.passed is not False for r in plugin_results.values())
        status_cls = "pass" if prompt_pass else "fail"

        label = prompt_text[:60] + "…" if len(prompt_text) > 60 else prompt_text
        active = " active" if i == 0 else ""
        list_items.append(
            f'<div class="prompt-list-item{active}" onclick="switchPrompt({i})">'
            f'<span class="status-dot {status_cls}"></span>'
            f"<span>Sample {i}: {_html.escape(label)}</span></div>"
        )

        detail_panels.append(
            f'<div class="prompt-detail-panel{active}">'
            f"{build_prompt_report_html(pd)}</div>"
        )

    parts.append(
        f'<div class="prompt-browser">'
        f'<div class="prompt-list">{"".join(list_items)}</div>'
        f'<div class="prompt-detail">{"".join(detail_panels)}</div>'
        f"</div>"
    )
    return "\n".join(parts)


# ── Combined report (convenience) ────────────────────────────────────────────


def generate_report(results_dir: str, output: str | None = None) -> str:
    """Generate a combined HTML report from a results directory.

    Scans for task_analysis/ and prompt_analysis/ subdirectories and
    assembles overview + per-prompt + task tabs.

    Args:
        results_dir: Directory containing analysis artifacts.
        output: Output HTML path. Defaults to ``results_dir/combined_report.html``.

    Returns:
        Path to the generated HTML file.
    """
    if output is None:
        output = os.path.join(results_dir, "combined_report.html")

    prompt_results = _collect_prompt_results(results_dir)

    # Task analysis
    ta_result = None
    ta_path = os.path.join(results_dir, "task_analysis", "results_summary.json")
    if os.path.isfile(ta_path):
        ta_result = TaskAnalysisPlugin().run(log_path=ta_path)

    # Load run config
    run_config = None
    rc_path = os.path.join(results_dir, "run_config.json")
    if os.path.isfile(rc_path):
        with open(rc_path) as f:
            run_config = json.load(f)

    # Build tabs: Overview → Task Analysis → Prompt Analysis
    tabs = [
        (
            "Overview",
            _build_overview_content(prompt_results, ta_result, run_config, results_dir),
        )
    ]

    # Task analysis tab
    if ta_result and ta_result.data:
        plugin = TaskAnalysisPlugin()
        tabs.append((plugin.display_name, plugin.wrap_with_guide(ta_result.html)))

    # Prompt analysis tab with sidebar browser (scales to many prompts)
    prompt_analysis_dir = os.path.join(results_dir, "prompt_analysis")
    if os.path.isdir(prompt_analysis_dir):
        prompt_dirs = sorted(
            d
            for d in glob.glob(os.path.join(prompt_analysis_dir, "prompt_*"))
            if os.path.isdir(d)
        )
        if prompt_dirs:
            tabs.append(
                (
                    f"Prompt Analysis ({len(prompt_dirs)} samples)",
                    _build_prompt_browser(prompt_dirs, run_config),
                )
            )

    return render_html(tabs, output)
