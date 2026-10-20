# SPDX-License-Identifier: Apache-2.0
"""Accuracy debugger example — Llama-3.2-1B-Instruct.

Runs one dataset eval (gsm8k_cot) then the full debugger pipeline:
task analysis → prompt analysis → combined HTML report.

The debugger analyzes *pre-computed* eval results; it ships no server launcher.
This example owns the eval server: it launches ``vllm serve`` itself (or points
at an operator-managed server via ``--server-url``), runs the eval, then hands
``run_task_analysis`` the results directory.

Usage:
    # Full pipeline (default) — launches its own server:
    python run_accuracy_debugger_llama.py --model /path/to/Llama-3.2-1B-Instruct

    # Task analysis only:
    python run_accuracy_debugger_llama.py --model /path/to/model --mode task_only

    # Prompt analysis only (reuses deviated prompts from a prior task run):
    python run_accuracy_debugger_llama.py --model /path/to/model --mode prompt_only \
        --output-dir ./accuracy_report

    # Validate against an already-running server:
    python run_accuracy_debugger_llama.py --model /path/to/model \
        --server-url http://localhost:8000 --mode task_only

Limitations:
    - KV cache analysis uses a small max_model_len for prompt analysis to avoid
      memory/compilation issues on long context.
    - Logit validation requires ``async_scheduling=False`` in the offline LLM so
      raw logits are returned via ``logprobs_mode="raw_logits"``.

See also: docs/model-dev/how-to-use-accuracy-debugger.md
"""

import argparse
import json
from typing import Optional, Sequence

# Works both when run as a script (sibling import, since sys.path[0] is this
# dir) and when imported as part of the examples package (package-absolute).
try:
    from examples.vllm_neuron.accuracy.accuracy_debugger_pipeline import (
        PipelineResult,
        get_server,
        load_deviated_prompts,
        resolve_output_dir,
        run_prompt_stage,
        run_task_stage,
    )
except ModuleNotFoundError:
    from accuracy_debugger_pipeline import (
        PipelineResult,
        get_server,
        load_deviated_prompts,
        resolve_output_dir,
        run_prompt_stage,
        run_task_stage,
    )
from vllm_neuron.accuracy.accuracy_debugger.utils.report_utils import generate_report
from vllm_neuron.accuracy.lm_eval import run_accuracy_gsm8k_cot

TP_SIZE = 8
MAX_MODEL_LEN = 8192
KV_SEGMENT_SIZE = 2048
BATCH_SIZE = 1
GEN_KWARGS = "max_gen_toks=4096"
EVAL_LIMIT = 10

EVAL_FN = run_accuracy_gsm8k_cot
THRESHOLDS = {
    "exact_match,flexible-extract": 0.395,
    "exact_match,strict-match": 0.395,
}

# Llama tensor compare config: last 3 layers + embed/norm/lm_head
NUM_LAYERS = 16
_TC_START = NUM_LAYERS - 3
TENSOR_COMPARE_MODULES = [
    "model.embed_tokens",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.input_layernorm",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.post_attention_layernorm",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.self_attn",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.mlp",
    "model.norm",
    "lm_head",
]

ADDITIONAL_CONFIG = json.dumps(
    {
        "neuron_config": {
            "on_device_sampling_config": {"all_greedy": True},
            "kv_segment_size_buckets": [KV_SEGMENT_SIZE],
            "num_batched_tokens_buckets": [KV_SEGMENT_SIZE],
            "num_seqs_buckets": [BATCH_SIZE],
        }
    }
)


def _module_order() -> list:
    order = ["model_embed_tokens"]
    for i in range(_TC_START, NUM_LAYERS):
        layer = f"model_layers_{i}"
        order.extend(
            [
                f"{layer}_input_layernorm",
                f"{layer}_self_attn",
                f"{layer}_post_attention_layernorm",
                f"{layer}_mlp",
            ]
        )
    order.extend(["model_norm", "lm_head"])
    return order


def _build_serve_cmd(model: str, port: int) -> str:
    return (
        f"vllm serve {model}"
        f" --tensor-parallel-size {TP_SIZE}"
        f" --max-model-len {MAX_MODEL_LEN}"
        f" --max-num-batched-tokens {KV_SEGMENT_SIZE}"
        f" --max-num-seqs {BATCH_SIZE}"
        f" --no-enable-prefix-caching"
        f" --additional-config '{ADDITIONAL_CONFIG}'"
        f" --port {port}"
    )


def _prompt_server_config(model: str) -> dict:
    return {
        "server": {
            "model": model,
            "tp_degree": TP_SIZE,
            "max_model_len": KV_SEGMENT_SIZE,
            "additional_config": {
                "neuron_config": {
                    "on_device_sampling_config": {"all_greedy": True},
                }
            },
        }
    }


def _run_prompts(model: str, prompts: list, output_dir: str):
    from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.kv_cache import (
        KvCachePlugin,
    )
    from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.logit_val import (
        LogitValPlugin,
    )
    from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.tensor_compare import (
        TensorComparePlugin,
    )

    return run_prompt_stage(
        server_config=_prompt_server_config(model),
        prompts=prompts,
        plugin_steps=[
            LogitValPlugin(),
            KvCachePlugin(
                thresholds={
                    "min_cos_similarity": 0.99,
                    "max_linf_multiplier": 5.0,
                    "min_bc": 0.85,
                }
            ),
            TensorComparePlugin(
                modules=TENSOR_COMPARE_MODULES,
                module_order=_module_order(),
                tp_size=TP_SIZE,
            ),
        ],
        output_dir=output_dir,
        output_length=16,
    )


def main(argv: Optional[Sequence[str]] = None) -> PipelineResult:
    """Run the accuracy-debugger pipeline and return its raw results.

    This is a demonstration of the pipeline: it runs the stages and returns a
    :class:`PipelineResult` for the caller to judge. It does not decide accuracy
    pass/fail (that verdict — which thresholds matter, which plugins gate — lives
    in ``test_accuracy_debugger.py``). It still raises on *operational* failures
    that stop the pipeline from proceeding (for example, a server it cannot stop
    before the offline phase)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the Llama-3.2-1B-Instruct checkpoint."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "task_only", "prompt_only"],
        default="full",
        help="Which stages to run (default: full pipeline).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report/artifact dir (default: $WORKLOAD_OUTPUT_RW or ./accuracy_report).",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=(
            "Base URL of an already-running vLLM server for the eval phase. If "
            "omitted, this script launches 'vllm serve' itself. A server it did "
            "not launch cannot be stopped mid-run, so it is incompatible with "
            "--mode full (use --mode task_only or prompt_only)."
        ),
    )
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT)
    args = parser.parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir)

    # ── Prompt-only: reuse deviated prompts from a prior task run ──────────────
    if args.mode == "prompt_only":
        prompts = load_deviated_prompts(output_dir)[:3]
        prompt_result = (
            _run_prompts(args.model, prompts, output_dir) if prompts else None
        )
        return PipelineResult(mode=args.mode, prompt_result=prompt_result)

    # ── Task analysis (task_only + full) ──────────────────────────────────────
    server = get_server(args.model, _build_serve_cmd, server_url=args.server_url)
    task_result = run_task_stage(
        server,
        output_dir,
        eval_fn=EVAL_FN,
        thresholds=THRESHOLDS,
        limit=args.limit,
        gen_kwargs=GEN_KWARGS,
        batch_size=BATCH_SIZE,
    )

    if args.mode == "task_only":
        return PipelineResult(mode=args.mode, task_result=task_result)

    # ── Full pipeline: free the eval server's cores, then prompt analysis ──────
    if not server.stop():
        raise RuntimeError(
            "Full pipeline needs a stoppable server to free Neuron cores before "
            "the offline phase. Omit --server-url so this script launches (and can "
            "stop) its own server, or use --mode prompt_only on a host with free "
            "cores."
        )

    prompts = task_result.deviated_prompts[:3]
    prompt_result = _run_prompts(args.model, prompts, output_dir) if prompts else None

    # Generate the report so it is always available for inspection.
    report_path = generate_report(output_dir)
    print(f"\nCombined report: {report_path}")

    return PipelineResult(
        mode=args.mode, task_result=task_result, prompt_result=prompt_result
    )


if __name__ == "__main__":
    result = main()
    print(f"\nPipeline finished (mode={result.mode}).")
