# SPDX-License-Identifier: Apache-2.0
"""Offline inference for GLM-5.3 (``GlmMoeDsaForCausalLM``) on one trn2.48xlarge.

Needs the Trn2 fp8 checkpoint, written once from the published FP8 (128 x 128 block) one::

    python examples/vllm_neuron/models/glm_dsa/convert_fp8_block_to_neuron_fp8.py \\
        --src <dir>/GLM-5.3 --dst <dir>/GLM-5.3-neuron-fp8

then::

    python examples/vllm_neuron/models/glm_dsa/run.py --model <dir>/GLM-5.3-neuron-fp8

    # long context: a pass key hidden at the start of ~30K tokens of filler
    python examples/vllm_neuron/models/glm_dsa/run.py \\
        --model <dir>/GLM-5.3-neuron-fp8 --needle 30000

The layout is the Hy4 port's (vllm_neuron/model/hy_v4): all 64 logical cores, pure
TP, one attention head per rank, every expert sharded 64 ways on its intermediate
dim. See README.md.
"""

import argparse
import os
import time

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3600")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
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
    parser.add_argument("--max-tokens", type=int, default=32)
    # Weights are ~14.4 GiB of each logical core's 24; the KV planner spends
    # ``24 * gmu - weights`` and does not count activations.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
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
    args = parser.parse_args()

    extra = {}
    if args.optlevel is not None:
        from vllm.config.vllm import OptimizationLevel

        extra["optimization_level"] = OptimizationLevel(args.optlevel)

    if args.no_async_scheduling:
        extra["async_scheduling"] = False

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
                "num_seqs_buckets": sorted({1, args.max_num_seqs}),
                "on_device_sampling_config": {"all_greedy": True},
                # No NKI kernel is called by this model yet, which is the case
                # Qwen3.5's dense checkpoints need the mac threshold suppressed
                # for (see that example's mac_threshold_override).
                "hlo2tensorizer_options": "",
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


if __name__ == "__main__":
    main()
