# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the accuracy-debugger examples.

The accuracy debugger analyzes *pre-computed* eval results — it ships no server
launcher. These helpers let an example own the eval server end-to-end: launch
``vllm serve`` (or point at an operator-managed one), run the eval, then hand
``run_task_analysis`` the results directory. The model-specific configuration
(model, plugins, thresholds, assertions) lives in each example script; only the
model-agnostic mechanics live here.

See ``run_accuracy_debugger_llama.py`` and ``run_accuracy_debugger_gpt_oss.py``
for the two concrete pipelines, and
``docs/model-dev/how-to-use-accuracy-debugger.md`` for the guide.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Server launcher shared across the accuracy examples. Re-exported here so the
# debugger example scripts keep importing get_server/LocalServer from this
# module. Try the package-absolute path first (works when imported as part of
# the examples package), fall back to the sibling import (works when the
# importing example is run as a script, since sys.path[0] is this dir).
try:
    from examples.vllm_neuron.accuracy.server_utils import (  # noqa: F401
        LocalServer,
        free_port,
        get_server,
    )
except ModuleNotFoundError:
    from server_utils import (  # noqa: F401
        LocalServer,
        free_port,
        get_server,
    )


@dataclass
class PipelineResult:
    """Raw results of a debugger pipeline run, for the caller to judge.

    The example populates whichever fields its ``--mode`` produced and returns
    this untouched — it makes no accuracy pass/fail decision. The verdict (which
    thresholds matter, which plugins gate) lives in the caller (the test), so the
    example stays a plain demonstration of the pipeline.
    """

    mode: str
    task_result: Optional[object] = None
    prompt_result: Optional[object] = None


def resolve_output_dir(output_dir: Optional[str]) -> str:
    """Resolve the report/artifact directory.

    Precedence: explicit *output_dir* arg > ``$WORKLOAD_OUTPUT_RW`` >
    ``./accuracy_report``.
    """
    if output_dir:
        return output_dir
    return os.environ.get("WORKLOAD_OUTPUT_RW", "./accuracy_report")


def run_task_stage(
    server: LocalServer,
    output_dir: str,
    *,
    eval_fn: Callable,
    thresholds: dict,
    limit: int,
    gen_kwargs: str,
    batch_size: int,
):
    """Run the eval against *server*, then analyze the results.

    Returns the :class:`TaskAnalysisResult`. The eval runs outside the debugger
    (the debugger only analyzes results), so this stays eval-runner agnostic.
    """
    from vllm_neuron.accuracy.accuracy_debugger import run_task_analysis
    from vllm_neuron.accuracy.accuracy_debugger.task_plugins.lm_eval_analyzer import (
        LmEvalAnalyzer,
    )

    results_dir = str(Path(output_dir) / "eval_results")
    eval_fn(
        base_url=server.base_url,
        model=server.model,
        results_dir=results_dir,
        limit=limit,
        gen_kwargs=gen_kwargs,
        max_concurrent=batch_size,
    )

    task_result = run_task_analysis(
        LmEvalAnalyzer(),
        input_task_results=results_dir,
        thresholds=thresholds,
        output_dir=output_dir,
    )

    print(f"\nTask analysis: {'PASSED' if task_result.passed else 'FAILED'}")
    for metric, score in task_result.scores.items():
        threshold = task_result.thresholds.get(metric)
        status = "✓" if threshold is None or score >= threshold else "✗"
        print(f"  {status} {metric}: {score:.4f} (threshold: {threshold})")
    print(f"Deviated prompts: {len(task_result.deviated_prompts)}")
    return task_result


def run_prompt_stage(
    server_config: dict,
    prompts: list,
    plugin_steps: list,
    output_dir: str,
    *,
    output_length: int = 16,
):
    """Run prompt analysis (logit_val + kv_cache + tensor_compare) and summarize."""
    from vllm_neuron.accuracy.accuracy_debugger import run_prompt_analysis

    print(f"\nRunning prompt analysis on {len(prompts)} prompts...")
    prompt_result = run_prompt_analysis(
        server_config=server_config,
        prompts=prompts,
        plugin_steps=plugin_steps,
        output_dir=output_dir,
        output_length=output_length,
    )

    print("\n" + "=" * 60)
    print("PROMPT ANALYSIS SUMMARY")
    print("=" * 60)
    for plugin_name, per_prompt_results in prompt_result.plugin_results.items():
        if not per_prompt_results:
            print(f"  {plugin_name}: DID NOT COMPLETE")
            continue
        flags = [
            r["passed"]
            for r in per_prompt_results.values()
            if isinstance(r, dict) and "passed" in r
        ]
        if not flags:
            print(f"  {plugin_name}: COMPLETED (no pass/fail threshold)")
        elif all(flags):
            print(f"  {plugin_name}: PASSED ({len(flags)}/{len(flags)} prompts)")
        else:
            n_failed = sum(1 for p in flags if not p)
            print(f"  {plugin_name}: FAILED ({n_failed}/{len(flags)} prompts failed)")
    print("=" * 60)
    return prompt_result


def load_deviated_prompts(output_dir: str) -> list:
    """Load deviated prompts saved by a prior task-analysis run."""
    import json

    deviated_path = Path(output_dir) / "task_analysis" / "deviated_prompts.json"
    if not deviated_path.exists():
        raise FileNotFoundError(
            f"No deviated_prompts.json found at {deviated_path}. Run the pipeline "
            "in 'task_only' mode first, or set WORKLOAD_OUTPUT_RW to existing "
            "results."
        )
    return json.loads(deviated_path.read_text())
