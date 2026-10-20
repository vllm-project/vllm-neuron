# SPDX-License-Identifier: Apache-2.0
"""Accuracy Debugger: task-level and prompt-level accuracy analysis for vllm-neuron.

Entry points:
  - run_task_analysis: task-level eval analysis only
  - run_prompt_analysis: prompt-level logit/KV/tensor analysis only
  - generate_report: generate HTML report from analysis artifacts

Example::

    from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot
    from vllm_neuron.accuracy.accuracy_debugger import (
        run_task_analysis,
        run_prompt_analysis,
    )
    from vllm_neuron.accuracy.accuracy_debugger.task_plugins.lm_eval_analyzer import LmEvalAnalyzer

    # Run the eval yourself against a server you started, e.g. in another shell:
    #   vllm serve /path/to/model --tensor-parallel-size 8
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

from vllm_neuron.accuracy.accuracy_debugger.api import (
    PromptAnalysisResult,
    TaskAnalysisResult,
    run_prompt_analysis,
    run_task_analysis,
)
from vllm_neuron.accuracy.accuracy_debugger.utils.report_utils import generate_report

__all__ = [
    "TaskAnalysisResult",
    "PromptAnalysisResult",
    "run_task_analysis",
    "run_prompt_analysis",
    "generate_report",
]
