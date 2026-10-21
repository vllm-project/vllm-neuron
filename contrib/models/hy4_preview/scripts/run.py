# SPDX-License-Identifier: Apache-2.0
"""Offline inference for Tencent Hy4-preview on one trn2.48xlarge.

Needs the Trn2 fp8 checkpoint, written once from the published MXFP8 one::

    python contrib/models/hy4_preview/scripts/convert_mxfp8_to_neuron_fp8.py \\
        --src <dir>/Hy4-preview-FP8 --dst <dir>/Hy4-preview-neuron-fp8

then::

    python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8

    # long context: a pass key hidden at the start of ~30K tokens of filler
    python contrib/models/hy4_preview/scripts/run.py \\
        --model <dir>/Hy4-preview-neuron-fp8 --needle 30000

All 64 logical cores, pure TP: one attention head per rank, every expert
sharded 64 ways on its intermediate dim. See README.md.
"""

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3600")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
]

PASS_KEY = "71432"
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)


def needle_prompt(tokenizer, approx_tokens: int) -> str:
    """The pass-key retrieval prompt: the key first, then filler, then the question."""
    head = (
        "There is an important piece of information hidden inside a lot of "
        "irrelevant text. Find it and memorize it.\n"
        f"The pass key is {PASS_KEY}. Remember it. {PASS_KEY} is the pass key.\n"
    )
    tail = "\nWhat is the pass key? The pass key is"
    per_filler = len(tokenizer.encode(FILLER, add_special_tokens=False))
    budget = approx_tokens - len(
        tokenizer.encode(head + tail, add_special_tokens=False)
    )
    return head + FILLER * max(0, budget // per_filler) + tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32768)
    # The prefill step size. Below max_model_len this is the plugin's segmented
    # prefill: one chunk of one request per step, each reading the prefix the
    # earlier chunks cached. Must be one of 512/1024/2048/4096/8192.
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--num-gpu-blocks", type=int, default=None, help="num_gpu_blocks_override (debugging)"
    )
    parser.add_argument(
        "--host-sampling", action="store_true", help="sample on the host (debugging)"
    )
    parser.add_argument(
        "--hf-overrides", default=None, help='JSON, e.g. {"num_hidden_layers": 4} (debugging)'
    )
    parser.add_argument(
        "--seq-buckets", default=None, help="decode batch buckets, e.g. 4 (default: 1 and --max-num-seqs)"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    # Weights are ~15 GiB of each logical core's 24; the KV planner spends
    # ``24 * gmu - weights`` and does not count activations.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--tensorizer-options",
        default=None,
        help="extra neuronx-cc --tensorizer-options, e.g. --disable-stream-transpose-small "
        "(GLM-5.3's sparse decode graph needs it)",
    )
    parser.add_argument("--optlevel", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--chat",
        action="store_true",
        help="wrap each prompt in the checkpoint's chat template",
    )
    parser.add_argument(
        "--needle",
        type=int,
        default=None,
        metavar="TOKENS",
        help=f"instead of the demo prompts, ask for a pass key ({PASS_KEY}) hidden "
        "at the start of about this many tokens of filler",
    )
    parser.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="run the engine with async_scheduling=False",
    )
    parser.add_argument(
        "--capture-dir",
        default=None,
        help="capture embed_tokens and layer 0 (plus, with HY4_DEBUG_CAPTURE=1, "
        "layer 0's DSA intermediates) to this directory, for comparison against "
        "the CPU port",
    )
    parser.add_argument(
        "--capture-modules",
        nargs="*",
        default=["model.embed_tokens", "model.layers.0"],
        help="module paths for --capture-dir",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="after the prompts, measure TTFT / TPOT at --bench-input-len in / "
        "--bench-output-len out, one request, median of 3 (trn2/hy4/bench_32k.py method)",
    )
    parser.add_argument("--bench-input-len", type=int, default=31744)
    parser.add_argument("--bench-output-len", type=int, default=256)
    args = parser.parse_args()
    if args.tensorizer_options:
        # Read by HYV4ForCausalLM.compiler_args_for in every worker.
        os.environ["HY4_TENSORIZER_OPTIONS"] = args.tensorizer_options

    extra = {}
    if args.optlevel is not None:
        from vllm.config.vllm import OptimizationLevel

        extra["optimization_level"] = OptimizationLevel(args.optlevel)

    if args.no_async_scheduling:
        extra["async_scheduling"] = False
    if args.hf_overrides:
        extra["hf_overrides"] = json.loads(args.hf_overrides)
    if args.num_gpu_blocks:
        extra["num_gpu_blocks_override"] = args.num_gpu_blocks

    chunk = min(args.prefill_chunk, args.max_model_len)
    llm = LLM(
        model=args.model,
        **extra,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=chunk,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        # Rank 0 compiles while the other ranks wait in a gloo barrier whose
        # timeout is this (vLLM's default is 3600 s). The 32K prefill graph
        # takes ~50-70 min at -O1, so the default kills every cold start.
        distributed_timeout_seconds=4 * 3600,
        cpu_distributed_timeout_seconds=4 * 3600,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [chunk],
                # A 32K request is alone in the KV cache, so give decode a
                # batch-1 graph rather than padding to max_num_seqs (2x TPOT).
                "num_seqs_buckets": (
                    [int(b) for b in args.seq_buckets.split(",")]
                    if args.seq_buckets
                    else sorted({1, args.max_num_seqs})
                ),
                "on_device_sampling_config": (
                    None if args.host_sampling else {"all_greedy": True}
                ),
                **(
                    {
                        "tensor_capture": {
                            "modules": args.capture_modules,
                            "capture_dir": args.capture_dir,
                        }
                    }
                    if args.capture_dir
                    else {}
                ),
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompts = PROMPTS
    if args.needle is not None:
        prompts = [needle_prompt(llm.get_tokenizer(), args.needle)]
    start = time.time()
    if args.chat:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(conversations, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    for prompt, output in zip(prompts, outputs):
        shown = prompt if len(prompt) < 200 else f"{prompt[:80]!r} ... {prompt[-60:]!r}"
        print(f"Prompt:    {shown!r} ({len(output.prompt_token_ids)} tokens)")
        print(f"Generated: {output.outputs[0].text!r}\n")
    if args.needle is not None:
        found = PASS_KEY in outputs[0].outputs[0].text
        print(f"NEEDLE {'FOUND' if found else 'MISSED'} in {elapsed:.1f} s")
    if args.bench:
        bench(llm, args.bench_input_len, args.bench_output_len)


def bench(llm, input_len: int, output_len: int, repeats: int = 3) -> None:
    """TTFT = T(1 token); TPOT = (T(n) - T(1)) / (n - 1); random ids, greedy, ignore_eos."""
    import json
    import statistics

    import torch
    from vllm.inputs import TokensPrompt

    vocab = llm.get_tokenizer().vocab_size
    g = torch.Generator().manual_seed(0)

    def timed(n_out: int) -> float:
        ids = torch.randint(1000, min(vocab, 120000) - 1000, (input_len,), generator=g)
        sp = SamplingParams(max_tokens=n_out, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids.tolist())], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert len(out[0].outputs[0].token_ids) == n_out
        return dt

    timed(2)  # warm-up
    ttft = [timed(1) for _ in range(repeats)]
    full = [timed(output_len) for _ in range(repeats)]
    t1, tn = statistics.median(ttft), statistics.median(full)
    tpot = (tn - t1) / (output_len - 1)
    print("BENCH_RESULT", json.dumps({
        "input_len": input_len, "output_len": output_len, "concurrency": 1,
        "ttft_s": round(t1, 3), "ttft_all_s": [round(x, 3) for x in ttft],
        "e2e_all_s": [round(x, 3) for x in full], "tpot_ms": round(tpot * 1e3, 1),
        "input_throughput_tok_s": round(input_len / t1, 1),
        "output_throughput_tok_s": round(1 / tpot, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
