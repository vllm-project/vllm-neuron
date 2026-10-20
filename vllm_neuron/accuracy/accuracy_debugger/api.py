# SPDX-License-Identifier: Apache-2.0
"""Public API for the Accuracy Debugger.

Key APIs:
  - run_task_analysis: task-level eval analysis
  - run_prompt_analysis: prompt-level logit/KV/tensor analysis
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_neuron.accuracy.accuracy_debugger.utils.api_utils import (
    extract_deviated_prompts,
    extract_server_config,
    resolve_prompts,
    run_prompt_plugins,
    write_prompt_report_txt,
    write_task_report_txt,
)

logger = logging.getLogger("accuracy_debugger")


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class TaskAnalysisResult:
    """Result of task-level accuracy analysis."""

    passed: bool
    scores: dict[str, float]
    thresholds: dict[str, float]
    deviated_prompts: list[str]
    report_path: str


@dataclass
class PromptAnalysisResult:
    """Result of prompt-level accuracy analysis."""

    prompts: list[str]
    plugin_results: dict[str, dict] = field(default_factory=dict)
    report_path: str = ""


# ── Report generation ─────────────────────────────────────────────────────────


# ── Task analysis ─────────────────────────────────────────────────────────────


def run_task_analysis(
    analyzer,
    input_task_results: str | dict,
    output_dir: str = "./accuracy_report",
    thresholds: dict[str, float] | None = None,
) -> TaskAnalysisResult:
    """Analyze pre-computed eval results to identify accuracy deviations.

    The debugger does not run the eval itself: run your eval harness (e.g.
    ``lm_eval``) separately and pass the results in via ``input_task_results``.
    This keeps the debugger decoupled from any particular eval runner — it only
    analyzes the results and judges pass/fail against ``thresholds``.

    Args:
        analyzer: An analyzer instance that knows how to interpret the eval
            results (e.g. ``LmEvalAnalyzer()``). Must provide
            ``resolve_results_dir(input_task_results, output_dir)``,
            ``analyze_all_results(ref_dir=None, target_dir=...)``,
            ``save_eval_results_to_file(task_deviation_data, path)``, and
            ``extract_scores(results_dir)``.
        input_task_results: Pre-computed eval results — either a path to an
            existing results directory (e.g. an lm_eval ``--output_path``) or a
            results dict. Interpretation is delegated to the analyzer.
        output_dir: Output directory for reports and artifacts.
        thresholds: Dict of ``{metric_name: minimum_value}`` for pass/fail
            evaluation. E.g. ``{"exact_match,flexible-extract": 0.435}``. All
            thresholds are lower-bound (">=") checks.

    Returns:
        TaskAnalysisResult with scores, thresholds, pass/fail, and deviated prompts.

    Example::

        from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot
        from vllm_neuron.accuracy.accuracy_debugger import run_task_analysis
        from vllm_neuron.accuracy.accuracy_debugger.task_plugins.lm_eval_analyzer import LmEvalAnalyzer

        # Run the eval yourself against a server you started
        # (e.g. ``vllm serve /path/to/model --tensor-parallel-size 8``):
        _scores, results_dir = run_accuracy_gsm8k_cot(
            base_url="http://localhost:8000", model="/path/to/model",
            # limit caps the sample count for a quick run; omit for the full dataset.
            results_dir="./eval_out", limit=200,
        )
        result = run_task_analysis(
            LmEvalAnalyzer(),
            input_task_results=results_dir,
            thresholds={"exact_match,flexible-extract": 0.435},
        )
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    _thresholds: dict[str, float] = dict(thresholds) if thresholds else {}

    # ── Locate/materialize the pre-computed eval results ──────────────────
    # The analyzer owns interpretation of the results (dir path vs. dict).
    eval_results_dir = analyzer.resolve_results_dir(input_task_results, output_path)

    # ── Task analysis ─────────────────────────────────────────────────────
    task_deviation_data, _ = analyzer.analyze_all_results(
        ref_dir=None, target_dir=eval_results_dir
    )

    task_analysis_dir = output_path / "task_analysis"
    task_analysis_dir.mkdir(exist_ok=True)
    results_summary_path = task_analysis_dir / "results_summary.json"
    analyzer.save_eval_results_to_file(task_deviation_data, results_summary_path)

    # ── Extract scores and deviated prompts ───────────────────────────────
    # Scores come from the caller-supplied eval results (the same raw metric
    # keys thresholds use).
    scores: dict[str, float] = analyzer.extract_scores(eval_results_dir)

    deviated = extract_deviated_prompts(task_deviation_data)
    deviated_prompts = [text for text, _, _ in deviated if text]

    # ── Determine pass/fail ───────────────────────────────────────────────
    # A run passes iff every configured threshold metric is present in scores
    # and meets its minimum. All task thresholds are lower-bound (">=") checks.
    passed = True
    for metric, threshold_val in _thresholds.items():
        actual = scores.get(metric)
        if actual is None or actual < threshold_val:
            passed = False
            logger.info(
                "Task threshold not met: %s = %s (need >= %s)",
                metric,
                actual,
                threshold_val,
            )

    # ── Save artifacts ────────────────────────────────────────────────────
    (task_analysis_dir / "task_status.json").write_text(
        json.dumps(
            {"passed": passed, "scores": scores, "thresholds": _thresholds}, indent=2
        )
    )
    (task_analysis_dir / "deviated_prompts.json").write_text(
        json.dumps(deviated_prompts, indent=2)
    )

    # Record thresholds to run_config.json for the report (merge with existing)
    run_config_path = output_path / "run_config.json"
    run_config = {}
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text())
    task_config: dict[str, Any] = {}
    if _thresholds:
        task_config["thresholds"] = _thresholds
    run_config["task_analysis"] = task_config
    run_config_path.write_text(json.dumps(run_config, indent=2))

    txt_path = str(task_analysis_dir / "task_report.txt")
    write_task_report_txt(txt_path, scores, _thresholds, passed, deviated_prompts)

    return TaskAnalysisResult(
        passed=passed,
        scores=scores,
        thresholds=_thresholds,
        deviated_prompts=deviated_prompts,
        report_path=txt_path,
    )


# ── Prompt analysis ───────────────────────────────────────────────────────────


def run_prompt_analysis(
    server_config: dict,
    prompts: list[str] | str,
    plugin_steps: list,
    output_dir: str = "./accuracy_report",
    output_length: int = 16,
) -> PromptAnalysisResult:
    """Run prompt-level accuracy analysis (logit validation, KV cache).

    Args:
        server_config: Server config dict.
        prompts: List of prompt strings, or path to a JSON file containing prompts.
        plugin_steps: List of instantiated PromptPlugin objects to run
            (e.g. ``[LogitValPlugin(), KvCachePlugin()]``).
        output_dir: Output directory for reports and artifacts.
        output_length: Number of decode tokens to generate per prompt.

    Returns:
        PromptAnalysisResult with per-step results and report path.
    """
    output_path = Path(output_dir).expanduser()
    prompt_analysis_dir = output_path / "prompt_analysis"
    prompt_analysis_dir.mkdir(parents=True, exist_ok=True)

    resolved_prompts = resolve_prompts(prompts)
    server_cfg = extract_server_config(server_config)
    model = server_cfg.get("model", server_cfg.get("model_path", ""))

    # ── Run per-prompt validation ─────────────────────────────────────────
    all_flow_results = {}
    for i, prompt_text in enumerate(resolved_prompts):
        prompt_dir = prompt_analysis_dir / f"prompt_{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt_text)

        logger.info("Prompt %d: %s", i, prompt_text[:80])
        flow_results, captured_output = run_prompt_plugins(
            model=model,
            prompts=[prompt_text],
            server_cfg=server_cfg,
            output_length=output_length,
            output_dir=str(prompt_dir),
            plugin_steps=plugin_steps,
        )
        all_flow_results[f"prompt_{i}"] = flow_results
        (prompt_dir / "validation_log.txt").write_text(captured_output)

    # ── Aggregate results ─────────────────────────────────────────────────
    merged = {}
    for prompt_key, fr in all_flow_results.items():
        for k, v in fr.items():
            merged[f"{prompt_key}/{k}"] = v

    # ── Aggregate results per plugin ───────────────────────────────────────
    plugin_results: dict[str, dict] = {}
    for plugin in plugin_steps:
        step_results = {k: v for k, v in merged.items() if plugin.name in k}
        if step_results:
            plugin_results[plugin.name] = step_results

    # ── Write run_config for report generator (merge with existing) ──────
    run_config_path = output_path / "run_config.json"
    run_config = {}
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text())
    run_config["prompt_analysis"] = {
        "model_checkpoint": model,
        "tp_degree": server_cfg.get("tp_degree", 8),
        "max_model_len": server_cfg.get("max_model_len", 8192),
        "output_length": output_length,
        "plugins": [p.name for p in plugin_steps],
    }
    run_config_path.write_text(json.dumps(run_config, indent=2))

    txt_path = str(output_path / "prompt_analysis" / "prompt_report.txt")
    write_prompt_report_txt(txt_path, resolved_prompts, merged, prompt_analysis_dir)

    return PromptAnalysisResult(
        prompts=resolved_prompts,
        plugin_results=plugin_results,
        report_path=txt_path,
    )
