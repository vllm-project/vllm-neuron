#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Measure TTFT / TPOT / E2E for Qwen3.5 on Neuron.

Uses ``AsyncLLM`` and streams tokens so TTFT is the real time to the *first*
token rather than a whole-request latency divided by something. TPOT is measured
over the remaining tokens only.

Input length is pinned by tokenising filler text and truncating, so a run at
``--input-tokens 1024`` really is a 1024-token prompt. The plugin compiles one
graph per batch bucket, so ``--max-num-seqs`` must match the concurrency you
want to measure -- measure batch 1 and batch 4 in separate processes rather than
configuring several buckets (supplying several ``num_seqs_buckets`` hangs, see
README.md "Known limitations").

``--prefill-buckets`` sets the prefill token buckets, and
``max_num_batched_tokens`` follows the largest of them. Left unset, the plugin's
own default applies -- powers of two from 128 up to ``--max-model-len`` -- so short
prompts land on a small bucket. Pinning a *single* bucket instead pads every
prefill up to it, which on 27B costs a factor of 2.8 in TTFT. See README.md
"Sizing the prefill bucket".

Usage:

    python examples/vllm_neuron/models/qwen3_5/benchmark_latency.py \
        --model <path-to-checkpoint>/Qwen3.5-2B \
        --max-num-seqs 1 --input-tokens 1024 --output-tokens 128

``--vision-bucket`` switches to a single-image prompt and builds the VL model, so
TTFT then includes the vision tower; it sets the per-block size as well (one
image, one block). Every request then gets a *unique* image (a salted 4x4
corner), because vLLM caches multimodal encoder outputs by image hash and
re-sending one identical image measures that cache instead of the tower.
``--reuse-image`` restores that behaviour deliberately, to quantify the
difference. Pin ``--max-model-len`` across a resolution sweep, or the text
prefill bucket moves under you and swamps the vision difference. Note
``--input-tokens`` must leave room for ``--output-tokens`` inside
``--max-model-len``, or the request is rejected before the engine runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

FILLER = (
    "The history of computing is a long sequence of small steps and occasional "
    "leaps, in which each generation of engineers rediscovers that the hard part "
    "was never the arithmetic but the bookkeeping around it. "
)


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
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--prefill-buckets",
        type=prefill_buckets,
        default=None,
        metavar="N[,N...]",
        help="prefill token buckets, i.e. num_batched_tokens_buckets, with "
        "max_num_batched_tokens taken from the largest. Left unset the plugin's "
        "own default applies (powers of two from 128 up to --max-model-len), "
        "which is usually what you want. Pinning a single value compiles just "
        "one prefill graph -- quickest to compile, but every prompt then pads up "
        'to it. See README.md "Sizing the prefill bucket"',
    )
    # Both needed for 27B on a trn2.3xlarge. HBM is 24 GB per *logical core*,
    # not 96 GB pooled, and at the default gmu the planner fills the leftover
    # budget with recurrent-state blocks until neuronx-cc rejects the graph
    # (NCC_EVRF009). Separately its decode graph fails ISA validation at -O1
    # (NCC_IINAR001) and compiles at -O2/-O3; the platform lowers vLLM's
    # default O2 to O1 and cannot see an explicit offline O2, so pass 3.
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--optlevel", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--sync-scheduling", action="store_true")
    # Concurrency, which no longer has to equal the compiled batch bucket.
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--stagger-ms", type=float, default=0.0)
    parser.add_argument(
        "--vision-bucket",
        type=int,
        default=0,
        help="non-zero switches to a single-image prompt and builds the VL model, "
        "so TTFT then includes the vision tower",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--vision-tp",
        type=int,
        default=0,
        help="vision encoder TP degree. 0 leaves it at the plugin default, which "
        "resolve_tp_dp() turns into tp=1/dp=world_size — an *unsharded* encoder, so "
        "with one image per request three of four ranks idle. Set 4 to shard it",
    )
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        help="send the byte-identical image every round, so vLLM's multimodal "
        "cache can serve it and the tower is skipped. Measures the cache, not the "
        "encoder — use it only to quantify that difference",
    )
    return parser.parse_args()


def build_prompt(model_path: str, num_tokens: int, salt: int) -> str:
    """Filler text truncated to exactly ``num_tokens`` tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # A distinct prefix per request so nothing can be served from a cache.
    text = f"Note {salt}. " + FILLER * (num_tokens // 20 + 4)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:num_tokens]
    return tokenizer.decode(ids)


def build_vl_prompt(model_path: str, image_size: int, salt: int | None = None):
    """A single-image chat prompt. One item, so the vision bucket stays small.

    ``salt`` makes the image unique, the way ``build_prompt`` makes the text
    unique. This matters more than it looks: vLLM caches multimodal *encoder
    outputs* keyed by a hash of the image, so re-sending one identical image every
    round measures the cache rather than the tower, and reports a vision cost far
    below the truth. A 4x4 block of pixels is enough to change the hash while
    leaving the token count — and therefore the compute — identical.
    """
    from transformers import AutoProcessor
    from vllm.assets.image import ImageAsset

    processor = AutoProcessor.from_pretrained(model_path)
    image = ImageAsset("cherry_blossom").pil_image.resize((image_size, image_size))
    if salt is not None:
        image = image.copy()
        for dx in range(4):
            for dy in range(4):
                image.putpixel(
                    (dx, dy), ((salt * 37 + dx * 11 + dy * 7) % 256, 0, 0)
                )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": [image]}}


async def one_round(
    engine, prompts, sampling_params, request_offset: int, stagger_ms: float = 0.0
):
    """Run ``prompts`` concurrently; return per-request (ttft, tpot, e2e, tokens)."""
    async def stream(index: int, prompt: str):
        if stagger_ms:
            await asyncio.sleep(index * stagger_ms / 1000.0)
        start = time.perf_counter()
        first_at = None
        count = 0
        async for out in engine.generate(
            prompt,
            sampling_params,
            request_id=f"req-{request_offset}-{index}",
        ):
            produced = len(out.outputs[0].token_ids)
            if first_at is None and produced >= 1:
                first_at = time.perf_counter()
            count = produced
        end = time.perf_counter()
        ttft = (first_at - start) * 1000.0
        # TPOT over the tokens after the first; undefined for a 1-token output.
        tpot = (
            (end - first_at) * 1000.0 / (count - 1) if count > 1 else float("nan")
        )
        return ttft, tpot, (end - start) * 1000.0, count

    return await asyncio.gather(
        *(stream(i, p) for i, p in enumerate(prompts))
    )


async def main_async(args) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    extra = {}
    if args.gpu_memory_utilization is not None:
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.optlevel is not None:
        from vllm.config.vllm import OptimizationLevel

        extra["optimization_level"] = OptimizationLevel(args.optlevel)
        if not any(a.startswith("--optimization-level") for a in sys.argv):
            sys.argv.append(f"--optimization-level={args.optlevel}")
    if args.sync_scheduling:
        extra["async_scheduling"] = False

    engine_args = AsyncEngineArgs(
        model=args.model,
        **extra,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=(
            args.prefill_buckets[-1] if args.prefill_buckets else args.max_model_len
        ),
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1 if args.vision_bucket else 0, "video": 0},
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
                # See run.py: the runner's default --modular-flow-mac-threshold
                # breaks codegen on this model's decode graph.
                "hlo2tensorizer_options": "",
            },
            # Supplying this at all is what selects the VL implementation, so a
            # zero bucket means "text-only" rather than "vision with no budget".
            **(
                {
                    "vision_neuron_config": {
                        "num_vision_tokens_buckets": [args.vision_bucket],
                        "vision_attention_block_size": args.vision_bucket,
                        # Left unset, resolve_tp_dp() picks tp=1/dp=world_size:
                        # the encoder is replicated per rank and a single-image
                        # request uses one rank. tp_size=4 shards heads and MLP.
                        **(
                            {"tp_size": args.vision_tp, "dp_size": 1}
                            if args.vision_tp
                            else {}
                        ),
                    }
                }
                if args.vision_bucket
                else {}
            ),
        },
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.output_tokens, min_tokens=args.output_tokens, temperature=0.0
    )

    mode = (
        f"vision, one {args.image_size}x{args.image_size} image"
        if args.vision_bucket
        else f"text-only, input_tokens={args.input_tokens}"
    )
    print(
        f"model={args.model}\n"
        f"tp={args.tensor_parallel_size} batch={args.max_num_seqs} "
        f"concurrent={args.num_requests or args.max_num_seqs} "
        f"stagger_ms={args.stagger_ms} {mode} "
        f"output_tokens={args.output_tokens} iterations={args.iterations}\n"
    )

    num_requests = args.num_requests or args.max_num_seqs

    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    round_wall: list[float] = []
    total_out = 0

    for iteration in range(args.iterations + 1):
        if args.vision_bucket:
            # A unique image per request unless asked otherwise. Turning
            # ``--reuse-image`` on measures the multimodal cache instead of the
            # tower, which is occasionally what you want to know.
            prompts = [
                build_vl_prompt(
                    args.model,
                    args.image_size,
                    salt=None if args.reuse_image else iteration * 100 + i,
                )
                for i in range(num_requests)
            ]
        else:
            prompts = [
                build_prompt(args.model, args.input_tokens, salt=iteration * 100 + i)
                for i in range(num_requests)
            ]
        started = time.perf_counter()
        results = await one_round(
            engine, prompts, sampling_params, iteration, args.stagger_ms
        )
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
        median = statistics.median(values)
        print(
            f"  {name:22s} mean {statistics.fmean(values):8.2f} {unit}   "
            f"median {median:8.2f} {unit}   "
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
        print(
            f"  per-stream decode     {1000.0 / statistics.fmean(tpots):8.2f} tok/s"
        )

    # AsyncLLM.shutdown() is synchronous; awaiting it raises.
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
