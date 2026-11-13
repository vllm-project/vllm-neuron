"""Validate segmented prefill, prefix-cache hits, decode context buckets and large decode
batches on a reduced Hy3 checkpoint, using the same serving config as the full model.

Usage: python validate_features.py <fp8_ckpt> <ref_long.pt>
"""

import sys

import torch
from vllm import LLM, SamplingParams

ckpt, ref_path = sys.argv[1], sys.argv[2]
DCP = int(sys.argv[3]) if len(sys.argv) > 3 else 1

NEURON_CONFIG = {
    "num_batched_tokens_buckets": [2048],
    "kv_segment_size_buckets": [2048],
    "num_seqs_buckets": [1, 4, 16, 32, 64, 128],
    "decode_context_length_buckets": [2048, 8192],
}


def to_logits(lp_dict, vocab):
    t = torch.full((vocab,), -float("inf"))
    for tid, lp in lp_dict.items():
        if tid < vocab:
            t[tid] = lp.logprob
    return t


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


if __name__ == "__main__":
    ref = [r for p in ref_path.split(",") for r in torch.load(p)]
    vocab = ref[0]["logits"].shape[-1]
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        max_model_len=32768,
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        block_size=32,
        enable_prefix_caching=True,
        async_scheduling=False,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        additional_config={"neuron_config": NEURON_CONFIG},
        decode_context_parallel_size=DCP,
        cp_kv_cache_interleave_size=32,
    )

    # 1+2: long prompts (segmented prefill), then again (prefix-cache hit).
    n = len(ref[0]["gen_ids"])
    sp = SamplingParams(temperature=0.0, max_tokens=n, logprobs=vocab, detokenize=False)
    prompts = [{"prompt_token_ids": r["prompt_ids"]} for r in ref]
    for attempt in ("cold", "prefix-cache hit"):
        outs = llm.generate(prompts, sp)
        for i, o in enumerate(outs):
            got, exp = list(o.outputs[0].token_ids), ref[i]["gen_ids"]
            match = next(
                (k for k, (a, b) in enumerate(zip(got, exp)) if a != b), len(exp)
            )
            cs = [
                cos(to_logits(o.outputs[0].logprobs[s], vocab), ref[i]["logits"][s])
                for s in range(min(match + 1, n))
            ]
            print(
                f"### {attempt} prompt {i} ({len(ref[i]['prompt_ids'])} tok, cached={o.num_cached_tokens}): "
                f"greedy match {match}/{n}, logits cos min={min(cs):.5f} per-step={[round(c, 4) for c in cs]}",
                flush=True,
            )

    # 3: large decode batch (>96 tokens -> shared expert CTE path) vs small batch.
    base = [f"Question {k}: what is {k} plus {3 * k + 1}? Answer:" for k in range(128)]
    sp4 = SamplingParams(
        temperature=0.0, max_tokens=4, logprobs=vocab, detokenize=False
    )
    small = llm.generate(base[:4], sp4)
    big = llm.generate(base, sp4)
    worst = 1.0
    for i in range(4):
        for s in range(4):
            if small[i].outputs[0].token_ids[:s] != big[i].outputs[0].token_ids[:s]:
                break
            c = cos(
                to_logits(small[i].outputs[0].logprobs[s], vocab),
                to_logits(big[i].outputs[0].logprobs[s], vocab),
            )
            worst = min(worst, c)
    same = sum(
        small[i].outputs[0].token_ids == big[i].outputs[0].token_ids for i in range(4)
    )
    print(
        f"### batch-128 vs batch-4: identical greedy {same}/4, min logits cos {worst:.5f}",
        flush=True,
    )
