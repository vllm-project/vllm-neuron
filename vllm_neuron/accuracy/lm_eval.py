# SPDX-License-Identifier: Apache-2.0
"""Per-dataset lm_eval accuracy runners.

Each function runs a single ``lm_eval`` benchmark against a running vLLM
(OpenAI-compatible) server and returns ``(flat_metrics_dict, results_file_path)``.
The flat dict maps ``"metric_key" -> value`` for every metric tracked, so callers
can assert on any/all of them.

Usage::

    results, path = run_accuracy_gsm8k_cot(base_url, model, results_dir, limit=200)
    assert results["exact_match,flexible-extract"] >= 0.435

These runners invoke the ``lm_eval`` CLI as a subprocess, plus the HuggingFace
``datasets`` cache. ``lm-eval[api,ifeval]`` is not a core wheel dependency;
install it (e.g. via the test requirements) to use these runners.

For access-gated datasets on HuggingFace, the caller pre-syncs the dataset and
passes ``data_dir`` (forwarded to the lm_eval subprocess as
``HF_DATASETS_CACHE``) so it resolves locally; otherwise lm_eval resolves
datasets through its normal HuggingFace path.
"""

import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_latest_results(results_dir: str) -> str:
    """Find the latest results JSON file."""
    files = list(Path(results_dir).rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"No results found in {results_dir}")
    return str(
        max(
            files,
            key=lambda f: datetime.strptime(
                os.path.basename(f).split("results_")[1].split(".json")[0],
                "%Y-%m-%dT%H-%M-%S.%f",
            ),
        )
    )


def resolve_metric(
    results: Dict[str, Any], task: str, metric_keys: List[str]
) -> Dict[str, Any]:
    """Extract *metric_keys* from lm_eval results for *task*.

    Handles both direct tasks (``results[task]``) and group/aggregate
    tasks where the aggregate lives at ``results[task]`` alongside
    per-subtask entries (e.g. ``bbh_cot_fewshot``, ``mmlu_pro``).

    Returns a flat dict ``{metric_key: value}``.  Logs an error and
    returns ``-1`` for any key not found in the results.
    """
    r = results.get(task, {})
    out = {}
    for k in metric_keys:
        if k in r:
            out[k] = r[k]
        else:
            logger.error("Metric '%s' not found in results for task '%s'", k, task)
            out[k] = -1
    return out


def run_lm_eval(
    base_url: str,
    model: str,
    results_dir: str,
    task: str,
    limit: Optional[int] = None,
    use_chat: bool = True,
    gen_kwargs: str = "",
    max_length: Optional[int] = None,
    fewshot_as_multiturn: bool = True,
    num_fewshot: Optional[int] = None,
    max_concurrent: int = 1,
    timeout: int = 7200,
    extra_args: Optional[list] = None,
    data_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run an lm_eval benchmark against a served model and return its metrics.

    The generic runner every per-dataset wrapper funnels through. It invokes the
    ``lm_eval`` CLI as a subprocess against a running vLLM (OpenAI-compatible)
    server, tees the subprocess output to ``<results_dir>/<task>_lm_eval.log``,
    locates the freshest ``results_*.json`` it writes, and returns lm_eval's raw
    per-task results. Use a per-dataset wrapper (e.g. :func:`run_accuracy_gsm8k`)
    for a turnkey benchmark, or call this directly to run an arbitrary lm_eval
    task and parse the results yourself.

    Args:
        base_url: vLLM server URL, e.g. ``"http://localhost:8000"`` (no ``/v1``).
        model: Model name sent in API requests (the server's served-model-name,
            or a path/id it accepts).
        results_dir: Directory lm_eval writes results + the run log into
            (created if missing).
        task: lm_eval task name, e.g. ``"gsm8k"``, ``"mmlu_pro"``,
            ``"bbh_cot_fewshot"``.
        limit: Max samples to evaluate; ``None`` (default) runs the full dataset.
        use_chat: If ``True`` (default) target ``/v1/chat/completions`` with
            ``--apply_chat_template``; if ``False`` use ``/v1/completions``.
        gen_kwargs: lm_eval ``--gen_kwargs`` string (e.g. ``"max_gen_toks=4096"``).
        max_length: Optional max sequence length passed in ``--model_args``.
        fewshot_as_multiturn: Pass ``--fewshot_as_multiturn`` (default ``True``);
            only meaningful with chat + few-shot tasks.
        num_fewshot: Optional ``--num_fewshot`` override.
        max_concurrent: Number of concurrent requests to the server
            (``num_concurrent`` in ``--model_args``).
        timeout: Per-request timeout in seconds passed to the lm_eval client.
        extra_args: Extra CLI flags appended verbatim (e.g.
            ``["--confirm_run_unsafe_code"]`` for MBPP).
        data_dir: Optional path to a directory of pre-synced datasets, exported
            as ``HF_DATASETS_CACHE`` for the subprocess so gated datasets the
            caller synced ahead of time resolve locally. The caller populates it;
            when omitted, lm_eval resolves datasets through its normal
            HuggingFace path.

    Returns:
        Tuple ``(results, results_file)`` where ``results`` is lm_eval's raw
        per-task results dict (``json["results"]`` — task name → metrics) and
        ``results_file`` is the path to the results JSON. Pass ``results`` and
        the task through :func:`resolve_metric` to pull specific metric keys.

    Raises:
        subprocess.CalledProcessError: If the ``lm_eval`` subprocess exits
            non-zero (see the run log under ``results_dir``).
        FileNotFoundError: If no ``results_*.json`` is produced.
    """
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = f"{base_url}/v1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    if data_dir:
        os.environ["HF_DATASETS_CACHE"] = str(data_dir)

    venv_python = sys.executable

    endpoint = "chat/completions" if use_chat else "completions"
    model_type = "local-chat-completions" if use_chat else "local-completions"

    model_args = (
        f"model={model},"
        f"base_url={base_url}/v1/{endpoint},"
        f"tokenized_requests=False,"
        f"tokenizer_backend=None,"
        f"num_concurrent={max_concurrent},"
        f"timeout={timeout}"
    )
    if max_length is not None:
        model_args += f",max_length={max_length}"

    cmd = [
        str(venv_python),
        "-m",
        "lm_eval",
        "--model",
        model_type,
        "--tasks",
        task,
        "--model_args",
        model_args,
        "--log_samples",
        "--output_path",
        results_dir,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if use_chat:
        cmd.append("--apply_chat_template")
    if fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if gen_kwargs:
        cmd.extend(["--gen_kwargs", gen_kwargs])
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running: %s", " ".join(cmd))

    log_file = os.path.join(results_dir, f"{task}_lm_eval.log")
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Logging to: %s", log_file)
    with open(log_file, "w") as lf:
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        logger.error("lm_eval failed (rc=%d), see %s", proc.returncode, log_file)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    results_file = _get_latest_results(results_dir)
    with open(results_file) as f:
        raw = json.load(f)
    return raw["results"], results_file


# ── Per-dataset entrypoints ──────────────────────────────────────────────────


def run_accuracy_gsm8k(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """GSM8K (plain, without chain-of-thought prompting).

    Usage:
        results, path = run_accuracy_gsm8k(base_url, model, results_dir)
        assert results["exact_match,flexible-extract"] >= 0.33

    Args:
        base_url: vLLM server URL (e.g. ``"http://localhost:8000"``).
        model: Model name or path.
        results_dir: Directory to store lm_eval output.
        limit: Max samples to evaluate (``None`` for full dataset).
        max_length: Maximum sequence length.
        gen_kwargs: Extra generation kwargs for lm_eval.
        **kwargs: Forwarded to :func:`run_lm_eval`.

    Returns:
        Tuple of (flat metrics dict, results file path).
    """
    task = "gsm8k"
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task=task,
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    metrics = [
        "exact_match,flexible-extract",
        "exact_match_stderr,flexible-extract",
        "exact_match,strict-match",
        "exact_match_stderr,strict-match",
    ]
    return resolve_metric(results, task, metrics), path


def run_accuracy_gsm8k_cot(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """GSM8K (chain-of-thought)."""
    task = "gsm8k_cot"
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task=task,
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    return resolve_metric(
        results,
        task,
        [
            "exact_match,flexible-extract",
            "exact_match_stderr,flexible-extract",
            "exact_match,strict-match",
            "exact_match_stderr,strict-match",
        ],
    ), path


def run_accuracy_gsm8k_cot_llama(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """GSM8K with Llama-specific prompt template."""
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task="gsm8k_cot_llama",
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    return resolve_metric(
        results,
        "gsm8k_cot_llama",
        [
            "exact_match,flexible-extract",
            "exact_match_stderr,flexible-extract",
            "exact_match,strict-match",
            "exact_match_stderr,strict-match",
        ],
    ), path


def run_accuracy_bbh(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """BIG-Bench Hard (chain-of-thought, fewshot)."""
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task="bbh_cot_fewshot",
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    return resolve_metric(
        results,
        "bbh_cot_fewshot",
        [
            "exact_match,get-answer",
            "exact_match_stderr,get-answer",
        ],
    ), path


def run_accuracy_ifeval(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """IFEval (instruction following)."""
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task="leaderboard_ifeval",
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    return resolve_metric(
        results,
        "leaderboard_ifeval",
        [
            "prompt_level_strict_acc,none",
            "prompt_level_strict_acc_stderr,none",
            "prompt_level_loose_acc,none",
            "prompt_level_loose_acc_stderr,none",
            "inst_level_strict_acc,none",
            "inst_level_loose_acc,none",
        ],
    ), path


def run_accuracy_mmlu_pro(
    base_url: str,
    model: str,
    results_dir: str,
    limit: Optional[int] = None,
    max_length: int = 16384,
    gen_kwargs: str = "",
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    """MMLU-Pro (multi-subject, custom extraction)."""
    results, path = run_lm_eval(
        base_url,
        model,
        results_dir,
        task="mmlu_pro",
        limit=limit,
        max_length=max_length,
        gen_kwargs=gen_kwargs,
        **kwargs,
    )
    return resolve_metric(
        results,
        "mmlu_pro",
        [
            "exact_match,custom-extract",
            "exact_match_stderr,custom-extract",
        ],
    ), path
