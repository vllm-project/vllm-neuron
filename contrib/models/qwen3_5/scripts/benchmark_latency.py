#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Measure TTFT / TPOT / E2E for Qwen3.5 on Neuron.

Uses ``AsyncLLM`` and streams tokens, so TTFT is the real time to the *first*
token rather than a whole-request latency divided by something. TPOT is measured
over the remaining tokens only. The first round is discarded as warmup.

Input length is pinned by tokenising filler text and truncating, so a run at
``--input-tokens 1024`` really is a 1024-token prompt. The plugin compiles one
graph per batch bucket, so ``--max-num-seqs`` is the batch measured: run batch 1,
4 and 8 as separate processes.

``max_num_batched_tokens`` is ``--max-model-len``: the model does not support
chunked prefill (see ``src/factory.py``). ``--prefill-buckets`` sets the prefill
token buckets under it. Left unset, the plugin's default applies (powers of two
up to ``--max-model-len``), so a short prompt lands on a small bucket.

Usage (the README's numbers):

    python contrib/models/qwen3_5/scripts/benchmark_latency.py \\
        --model <path>/Qwen3.5-2B --max-model-len 2048 \\
        --max-num-seqs 1 --input-tokens 1024 --output-tokens 128

    # 397B-A17B: the whole trn2.48xlarge
    python contrib/models/qwen3_5/scripts/benchmark_latency.py \\
        --model <path>/Qwen3.5-397B-A17B --tensor-parallel-size 64 \\
        --expert-parallel --ep-degree 8 --gpu-memory-utilization 0.7 \\
        --max-model-len 2048 --max-num-seqs 1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

FILLER = (
    "The history of computing is a long sequence of small steps and occasional "
    "leaps, in which each generation of engineers rediscovers that the hard part "
    "was never the arithmetic but the bookkeeping around it. "
)


def _run_module():
    """``scripts/run.py``, for its registration and head-count helpers."""
    path = Path(__file__).resolve().with_name("run.py")
    spec = importlib.util.spec_from_file_location("qwen3_5_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefill_buckets(spec: str) -> list[int]:
    """Parse ``--prefill-buckets`` into a strictly ascending list of token counts."""
    buckets = [int(x) for x in spec.split(",")]
    if buckets != sorted(set(buckets)):
        raise argparse.ArgumentTypeError(
            f"--prefill-buckets must be strictly ascending, got {buckets}"
        )
    return buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--prefill-buckets",
        type=prefill_buckets,
        default=None,
        metavar="N[,N...]",
        help="num_batched_tokens_buckets; the largest must be --max-model-len",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=None,
        help="expert-parallel degree; requires --expert-parallel",
    )
    parser.add_argument("--sync-scheduling", action="store_true")
    return parser.parse_args()


def build_prompt(model_path: str, num_tokens: int, salt: int) -> str:
    """Filler text truncated to exactly ``num_tokens`` tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # A distinct prefix per request so nothing can be served from a cache.
    text = f"Note {salt}. " + FILLER * (num_tokens // 20 + 4)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:num_tokens]
    return tokenizer.decode(ids)


async def one_round(engine, prompts, sampling_params, request_offset: int):
    """Run ``prompts`` concurrently; return per-request (ttft, tpot, e2e, tokens)."""

    async def stream(index: int, prompt: str):
        start = time.perf_counter()
        first_at = None
        count = 0
        async for out in engine.generate(
            prompt, sampling_params, request_id=f"req-{request_offset}-{index}"
        ):
            produced = len(out.outputs[0].token_ids)
            if first_at is None and produced >= 1:
                first_at = time.perf_counter()
            count = produced
        end = time.perf_counter()
        ttft = (first_at - start) * 1000.0
        # TPOT over the tokens after the first; undefined for a 1-token output.
        tpot = (end - first_at) * 1000.0 / (count - 1) if count > 1 else float("nan")
        return ttft, tpot, (end - start) * 1000.0, count

    return await asyncio.gather(*(stream(i, p) for i, p in enumerate(prompts)))


async def main_async(args) -> None:
    run = _run_module()
    run.register_qwen3_5()
    if args.tensor_parallel_size > 1:
        run.relax_head_divisibility()

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    extra = {}
    if args.gpu_memory_utilization is not None:
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.sync_scheduling:
        extra["async_scheduling"] = False
    if args.expert_parallel:
        extra["enable_expert_parallel"] = True
    if args.ep_degree is not None and not args.expert_parallel:
        raise SystemExit("--ep-degree requires --expert-parallel")
    if args.prefill_buckets and args.prefill_buckets[-1] != args.max_model_len:
        raise SystemExit("the largest --prefill-buckets must equal --max-model-len")

    # No optimization_level: an explicit level disables the model's
    # compiler_args_for, which the dense decode graph needs (see model.py).
    engine_args = AsyncEngineArgs(
        model=args.model,
        **extra,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                **(
                    {"num_batched_tokens_buckets": args.prefill_buckets}
                    if args.prefill_buckets
                    else {}
                ),
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                **({"ep_degree": args.ep_degree} if args.ep_degree else {}),
            },
        },
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.output_tokens, min_tokens=args.output_tokens, temperature=0.0
    )
    num_requests = args.max_num_seqs
    print(
        f"model={args.model}\n"
        f"tp={args.tensor_parallel_size} batch={num_requests} "
        f"input_tokens={args.input_tokens} output_tokens={args.output_tokens} "
        f"iterations={args.iterations}\n"
    )

    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    round_wall: list[float] = []
    total_out = 0
    for iteration in range(args.iterations + 1):
        prompts = [
            build_prompt(args.model, args.input_tokens, salt=iteration * 100 + i)
            for i in range(num_requests)
        ]
        started = time.perf_counter()
        results = await one_round(engine, prompts, sampling_params, iteration)
        wall = time.perf_counter() - started
        if iteration == 0:
            print("(discarding the first round as warmup)\n")
            continue
        for ttft, tpot, e2e, count in results:
            ttfts.append(ttft)
            if count > 1:
                tpots.append(tpot)
            e2es.append(e2e)
            total_out += count
        round_wall.append(wall)

    def stat(name: str, values: list[float], unit: str = "ms") -> None:
        if not values:
            print(f"  {name:22s} n/a")
            return
        print(
            f"  {name:22s} mean {statistics.fmean(values):8.2f} {unit}   "
            f"median {statistics.median(values):8.2f} {unit}   "
            f"min {min(values):8.2f}   max {max(values):8.2f}"
        )

    print(f"results over {len(round_wall)} rounds x {num_requests} requests")
    stat("TTFT", ttfts)
    stat("TPOT", tpots)
    stat("E2E", e2es)
    throughput = total_out / sum(round_wall)
    print(
        f"\n  output throughput     {throughput:8.2f} tok/s "
        f"(all {num_requests} concurrent requests)"
    )
    if tpots:
        print(f"  per-stream decode     {1000.0 / statistics.fmean(tpots):8.2f} tok/s")

    # AsyncLLM.shutdown() is synchronous; awaiting it raises.
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
