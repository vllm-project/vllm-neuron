# SPDX-License-Identifier: Apache-2.0
"""Accuracy debugger example — Qwen3-30B-A3B BF16, with and without YaRN.

Structure follows run_accuracy_debugger_gpt_oss.py; Qwen3-MoE was ported from
GPT-OSS, so the MoE-aware tensor-compare gating carries over unchanged.

Two things are being verified, and they are separate concerns:

1. That the 0.24 port is functionally correct at all. Run ``--rope native``.
   This is the SOW's functional milestone (< 1e-5 atol on logits), which the
   LogitValPlugin tol_map enforces at its tightest setting.

2. That enabling YaRN does not wreck SHORT prompts. Run ``--rope yarn``.
   Qwen is explicit that every open-source framework implements *static* YaRN,
   so the scaling factor applies to a 40-token gsm8k question exactly as it does
   to a 100k-token document. gsm8k under YaRN is therefore a direct measurement
   of that cost. Compare the two exact_match scores; a large drop is the
   expected-but-quantified downside, not a bug in the port.

What this does NOT verify is long-context *quality*. There is no needle-in-a-
haystack or RULER-style eval in this repo, so correctness at 131,072 tokens is
unproven by this script. It checks that YaRN is wired in correctly and that the
short-context regression is measured, nothing more.

Usage:
    python run_accuracy_debugger_qwen3_moe.py --model Qwen/Qwen3-30B-A3B
    python run_accuracy_debugger_qwen3_moe.py --model Qwen/Qwen3-30B-A3B --rope yarn
    python run_accuracy_debugger_qwen3_moe.py --model Qwen/Qwen3-30B-A3B --mode task_only

See also: docs/model-dev/how-to-use-accuracy-debugger.md
"""

import argparse
import json
from typing import Optional, Sequence

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

# TP=4 rather than the 8 used by the GPT-OSS example. Qwen3-30B-A3B has 128
# experts of intermediate width 768; TP=8 shards each to 96 wide, which
# measured at roughly half the per-device throughput of TP=4. TP=4 is also one
# whole device at LNC=2, which keeps this cheap to run.
TP_SIZE = 4
BATCH_SIZE = 1
EVAL_LIMIT = 20

# Native context is 32,768. 10,240 is plenty for gsm8k_cot and keeps the KV
# reservation (and therefore compile time) small -- max_model_len reserves KV
# per sequence whether or not the sequence is that long.
MAX_MODEL_LEN = 10240

# Qwen's published long-context setting: 32,768 x 4 = 131,072 tokens.
YARN_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}

EVAL_FN = run_accuracy_gsm8k_cot
THRESHOLDS = {
    "exact_match,flexible-extract": 0.7,
}

# Same tolerances as GPT-OSS: atol stays at 1e-5 (the SOW functional bar) while
# rtol widens with token count, because MoE routing can send a token to a
# different expert and that shows up as a large relative delta on a small value.
QWEN3_MOE_TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

GEN_KWARGS = json.dumps(
    {
        "max_tokens": MAX_MODEL_LEN,
        "until": ["<|im_end|>", "<|endoftext|>"],
    }
)

ADDITIONAL_CONFIG = json.dumps(
    {
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [128, 256, 2048, 4096, 8192, MAX_MODEL_LEN],
            "num_seqs_buckets": [BATCH_SIZE],
        }
    }
)

# Qwen3-30B-A3B: 48 layers. Last 3 plus embed/norm/lm_head, as GPT-OSS does.
NUM_LAYERS = 48
_TC_START = NUM_LAYERS - 3
TENSOR_COMPARE_MODULES = [
    "model.embed_tokens",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.input_layernorm",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.self_attn",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.post_attention_layernorm",
    f"model.layers.{_TC_START}-{NUM_LAYERS - 1}.mlp",
    "model.norm",
    "lm_head",
]


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


def _hf_overrides(rope: str) -> dict:
    """YaRN is switched on through hf_overrides so the checkpoint is untouched.

    Unlike the GPT-OSS example there is no ``quantization_config`` to blank out;
    Qwen3-30B-A3B is plain bf16.
    """
    return {"rope_scaling": YARN_ROPE_SCALING} if rope == "yarn" else {}


# get_server() calls this with exactly (model, port), so the rope mode travels
# through module state rather than an argument.
_ROPE_MODE = "native"


def _build_serve_cmd(model: str, port: int) -> str:
    cmd = (
        f"vllm serve {model}"
        f" --tensor-parallel-size {TP_SIZE}"
        f" --max-model-len {MAX_MODEL_LEN}"
        f" --max-num-batched-tokens {MAX_MODEL_LEN}"
        f" --max-num-seqs {BATCH_SIZE}"
        f" --no-enable-log-requests"
        f" --no-enable-prefix-caching"
        f" --additional-config '{ADDITIONAL_CONFIG}'"
        f" --port {port}"
    )
    overrides = _hf_overrides(_ROPE_MODE)
    if overrides:
        cmd += f" --hf-overrides '{json.dumps(overrides)}'"
    return cmd


def _prompt_server_config(model: str, rope: str) -> dict:
    cfg = {
        "server": {
            "model": model,
            "tp_degree": TP_SIZE,
            "max_model_len": 8192,
            "additional_config": {
                "neuron_config": {
                    "on_device_sampling_config": {"all_greedy": True},
                }
            },
        }
    }
    overrides = _hf_overrides(rope)
    if overrides:
        cfg["server"]["hf_overrides"] = overrides
    return cfg


def _run_prompts(model: str, prompts: list, output_dir: str, rope: str):
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
        server_config=_prompt_server_config(model, rope),
        prompts=prompts,
        plugin_steps=[
            LogitValPlugin(tol_map=QWEN3_MOE_TOL_MAP),
            KvCachePlugin(),
            TensorComparePlugin(
                modules=TENSOR_COMPARE_MODULES,
                module_order=_module_order(),
                tp_size=TP_SIZE,
                # Relaxed for the same reason as GPT-OSS: MoE routing divergence
                # spikes prefill L2 on individual tokens.
                max_l2_ratio=6.0,
            ),
        ],
        output_dir=output_dir,
        output_length=16,
    )


def main(argv: Optional[Sequence[str]] = None) -> PipelineResult:
    """Run the pipeline and return raw results; the caller judges pass/fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B",
        help="Path or HF id of the Qwen3-30B-A3B checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "task_only", "prompt_only"],
        default="full",
    )
    parser.add_argument(
        "--rope",
        choices=["native", "yarn"],
        default="native",
        help=(
            "native: 32,768-token RoPE, unscaled -- use this to verify the port. "
            "yarn: factor 4.0 for 131,072 tokens -- use this to measure what "
            "static YaRN costs on short prompts."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT)
    args = parser.parse_args(argv)

    global _ROPE_MODE
    _ROPE_MODE = args.rope

    output_dir = resolve_output_dir(args.output_dir)
    print(f"[qwen3-moe] rope={args.rope} tp={TP_SIZE} output_dir={output_dir}")
    if args.rope == "yarn":
        print(f"[qwen3-moe] hf_overrides={json.dumps(_hf_overrides('yarn'))}")

    if args.mode == "prompt_only":
        prompts = load_deviated_prompts(output_dir)[:3]
        prompt_result = (
            _run_prompts(args.model, prompts, output_dir, args.rope)
            if prompts
            else None
        )
        return PipelineResult(mode=args.mode, prompt_result=prompt_result)

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

    if not server.stop():
        raise RuntimeError(
            "Full pipeline needs a stoppable server to free Neuron cores before "
            "the offline phase. Omit --server-url so this script launches (and can "
            "stop) its own server, or use --mode prompt_only on a host with free "
            "cores."
        )

    prompts = task_result.deviated_prompts[:3]
    prompt_result = (
        _run_prompts(args.model, prompts, output_dir, args.rope) if prompts else None
    )

    report_path = generate_report(output_dir)
    print(f"\nCombined report: {report_path}")

    return PipelineResult(
        mode=args.mode, task_result=task_result, prompt_result=prompt_result
    )


if __name__ == "__main__":
    result = main()
    print(f"\nPipeline finished (mode={result.mode}).")
