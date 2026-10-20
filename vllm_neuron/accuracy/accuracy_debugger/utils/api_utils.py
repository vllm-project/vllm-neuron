# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for the Accuracy Debugger API."""

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("accuracy_debugger")


def extract_server_config(config: dict) -> dict:
    """Extract server section from a combined or standalone config."""
    if "server" in config:
        return config["server"]
    if "model" in config or "name" in config:
        return config
    return config


def resolve_prompts(prompts: list[str] | str) -> list[str]:
    """Resolve prompts from a list, a JSON file path, or a single string."""
    if isinstance(prompts, list):
        return prompts
    prompts_path = Path(prompts)
    if prompts_path.exists():
        content = prompts_path.read_text().strip()
        try:
            loaded = json.loads(content)
            if isinstance(loaded, list):
                return loaded
        except json.JSONDecodeError:
            pass
        return [content]
    return [prompts]


def extract_deviated_prompts(
    task_deviation_data: dict, max_prompts: int = 100
) -> list[tuple[str, Any, str]]:
    """Extract deviated prompts from task deviation data.

    Returns list of (prompt_text, doc_id, task_name) tuples.
    """
    prompts = []
    for task_name, (_, deviations, _) in task_deviation_data.items():
        for dev in deviations:
            if len(prompts) >= max_prompts:
                return prompts
            prompt_text = dev.prompt or ""
            if not prompt_text and dev.target_results:
                doc = dev.target_results.get("doc", {})
                for fld in ("question", "problem", "text", "input"):
                    if fld in doc:
                        prompt_text = doc[fld]
                        break
            prompts.append((prompt_text, dev.doc_id, task_name))
    return prompts


def run_prompt_plugins(
    *,
    model: str,
    prompts: list[str],
    server_cfg: dict,
    output_length: int = 16,
    output_dir: str = "./accuracy_report/prompt_analysis",
    plugin_steps: list = (),
) -> tuple[dict, str]:
    """Run prompt analysis plugins using offline serving.

    Accepts pre-instantiated plugin objects, runs pre_llm hooks,
    creates the vLLM LLM, then runs and saves each plugin.

    Returns:
        Tuple of (results dict, captured_output string).
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.base import PluginContext
    from vllm_neuron.accuracy.logit_validation import create_offline_vllm_generate_fn

    # The caller passes an already-resolved checkpoint path/id (the test harness
    # resolves it via get_model_checkpoint before invoking the example).
    model_checkpoint = model
    model_config = AutoConfig.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = model_config.torch_dtype or torch.bfloat16
    tp = server_cfg.get("tp_degree", 8)
    max_model_len = server_cfg.get("max_model_len", 8192)
    batch_size = server_cfg.get("batch_size", 1)

    ctx = PluginContext(
        model_checkpoint=model_checkpoint,
        prompts=prompts,
        server_cfg=server_cfg,
        output_length=output_length,
        output_dir=output_dir,
        goldens=None,
        tokenizer=tokenizer,
        dtype=dtype,
        batch_size=batch_size,
    )

    # Pre-LLM hooks (e.g. KV cache extraction before LLM eats memory)
    for plugin in plugin_steps:
        plugin.pre_llm(ctx)

    # Build vLLM offline args
    additional_config = server_cfg.get("additional_config", {})
    neuron_config = additional_config.get("neuron_config", {})
    neuron_config.setdefault("on_device_sampling_config", {"all_greedy": True})
    neuron_config.setdefault("num_batched_tokens_buckets", [max_model_len])
    neuron_config.setdefault("num_seqs_buckets", [batch_size])

    vllm_args = {
        "model": model_checkpoint,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tp,
        "max_num_seqs": batch_size,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "additional_config": {"neuron_config": neuron_config},
    }
    hf_overrides = server_cfg.get("hf_overrides")
    if hf_overrides:
        if isinstance(hf_overrides, str):
            hf_overrides = json.loads(hf_overrides)
        vllm_args["hf_overrides"] = hf_overrides

    # Capture stdout
    capture_buf = io.StringIO()

    class TeeWriter:
        def __init__(self, original, buffer):
            self.original, self.buffer = original, buffer

        def write(self, s):
            self.original.write(s)
            self.buffer.write(s)

        def flush(self):
            self.original.flush()

        def fileno(self):
            return self.original.fileno()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, capture_buf)

    shared_plugins = [p for p in plugin_steps if getattr(p, "needs_shared_llm", True)]
    self_managed_plugins = [
        p for p in plugin_steps if not getattr(p, "needs_shared_llm", True)
    ]

    results = {}
    try:
        # Run shared-LLM plugins first
        if shared_plugins:
            ctx.llm = LLM(**vllm_args)
            ctx.generate_fn = create_offline_vllm_generate_fn(ctx.llm, output_length)

            for plugin in shared_plugins:
                plugin_results = plugin.run(ctx)
                results[plugin.name] = plugin_results
                plugin.save(ctx, plugin_results)

            # Free Neuron cores for self-managed plugins
            del ctx.llm
            ctx.llm = None
            ctx.generate_fn = None

        # Run self-managed plugins (each manages its own LLM)
        for plugin in self_managed_plugins:
            plugin_results = plugin.run(ctx)
            results[plugin.name] = plugin_results
            plugin.save(ctx, plugin_results)
    except Exception as e:
        logger.error("Prompt analysis failed: %s", e)
        raise
    finally:
        sys.stdout = original_stdout

    return results, capture_buf.getvalue()


def write_task_report_txt(
    path: str,
    scores: dict,
    thresholds: dict,
    passed: bool,
    deviated_prompts: list[str],
) -> None:
    """Write a plain-text task analysis report."""
    status = "PASSED" if passed else "ISSUES DETECTED"
    lines = [
        "Task Analysis Report",
        "=" * 60,
        f"Overall: {status}",
        f"Deviated prompts: {len(deviated_prompts)}",
        "",
        "Scores:",
    ]
    for metric, score in scores.items():
        threshold_val = thresholds.get(metric)
        if threshold_val is not None:
            status = "✓" if score >= threshold_val else "✗"
            lines.append(
                f"  {status} {metric}: {score:.4f} (threshold: {threshold_val:.4f})"
            )
        else:
            lines.append(f"  {metric}: {score:.4f}")

    if not scores:
        lines.append("  (no scores available)")

    if deviated_prompts:
        lines.append("")
        lines.append("Deviated Prompts:")
        for i, prompt in enumerate(deviated_prompts):
            lines.append(f"  [{i}] {prompt[:120]}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def write_prompt_report_txt(
    path: str, prompts: list[str], results: dict, prompt_analysis_dir: Path
) -> None:
    """Write a plain-text prompt analysis report."""
    from vllm_neuron.accuracy.accuracy_debugger.utils.report_utils import PROMPT_PLUGINS

    lines = [
        "Prompt Analysis Report",
        "=" * 60,
        f"Prompts analyzed: {len(prompts)}",
        "",
        "Plugin Results:",
    ]
    for key, val in results.items():
        completed = val is not None and val != {}
        if not completed:
            status = "DID NOT COMPLETE"
        elif isinstance(val, bool):
            status = "PASSED" if val else "FAILED"
        elif isinstance(val, dict) and "passed" in val:
            status = "PASSED" if val["passed"] else "FAILED"
        else:
            status = "COMPLETED"
        lines.append(f"  {key}: {status}")

    for i, prompt_text in enumerate(prompts):
        prompt_dir = str(prompt_analysis_dir / f"prompt_{i}")
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Prompt {i}: {prompt_text[:100]}")
        lines.append("=" * 60)

        for plugin_cls in PROMPT_PLUGINS:
            plugin = plugin_cls(prompt_dir, prompt_dir)
            res = plugin.run()
            if res.text_summary:
                lines.append(res.text_summary)
                lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
