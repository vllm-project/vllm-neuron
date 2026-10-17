"""Run Hy3 on vLLM Neuron and compare greedy tokens / raw logits to an HF reference.

Usage: python run_vllm.py <ckpt> <ref.pt|-> [tp=64] [max_model_len=1024] [max_num_seqs=4]
"""

import sys
import time

import torch
from vllm import LLM, SamplingParams

ckpt, ref_path = sys.argv[1], sys.argv[2]
tp = int(sys.argv[3]) if len(sys.argv) > 3 else 64
max_len = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
max_seqs = int(sys.argv[5]) if len(sys.argv) > 5 else 4

if __name__ == "__main__":
    ref = torch.load(ref_path) if ref_path != "-" else None
    t0 = time.time()
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        max_model_len=max_len,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_len,
        block_size=32,
        enable_prefix_caching=False,
        async_scheduling=False,
        max_logprobs=-1 if ref is not None else 0,
        logprobs_mode="raw_logits",
    )
    print(f"### engine ready in {time.time() - t0:.0f}s", flush=True)

    if ref is None:
        prompts = [
            "The capital of France is",
            "人工智能的未来发展趋势是",
            "def quicksort(arr):",
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=48)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        print(f"### generate {time.time() - t0:.1f}s", flush=True)
        for o in outs:
            print(f"### {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
        sys.exit(0)

    vocab = ref[0]["logits"].shape[-1]
    n = len(ref[0]["gen_ids"])
    sp = SamplingParams(temperature=0.0, max_tokens=n, logprobs=vocab, detokenize=False)
    prompts = [{"prompt_token_ids": r["prompt_ids"]} for r in ref]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"### generate {time.time() - t0:.1f}s", flush=True)

    worst_cos = 1.0
    for i, o in enumerate(outs):
        got = list(o.outputs[0].token_ids)
        exp = ref[i]["gen_ids"]
        match = 0
        for a, b in zip(got, exp):
            if a != b:
                break
            match += 1
        print(
            f"### [{i}] got {got}\n###     ref {exp}  greedy-prefix-match {match}/{len(exp)}",
            flush=True,
        )
        # Logits are comparable while both sequences share the same prefix.
        for step in range(min(match + 1, len(o.outputs[0].logprobs))):
            ours = torch.full((vocab,), -float("inf"))
            for tid, lp in o.outputs[0].logprobs[step].items():
                if tid < vocab:
                    ours[tid] = lp.logprob
            theirs = ref[i]["logits"][step]
            cos = torch.nn.functional.cosine_similarity(ours, theirs, dim=0).item()
            worst_cos = min(worst_cos, cos)
            lp_o, lp_t = torch.log_softmax(ours, -1), torch.log_softmax(theirs, -1)
            top_t = theirs.topk(10).indices
            top_o = ours.topk(10).indices
            overlap = len(set(top_t.tolist()) & set(top_o.tolist()))
            print(
                f"###     step {step}: logits cos={cos:.5f} max|dlogit|={(ours - theirs).abs().max():.3f} "
                f"top10 overlap={overlap}/10 |dlogprob| on ref top10 max={(lp_o[top_t] - lp_t[top_t]).abs().max():.3f}",
                flush=True,
            )
    print(f"### worst logits cosine {worst_cos:.5f}", flush=True)
