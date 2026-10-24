"""FP8-KV validation run on a (reduced) Hy3 checkpoint; dumps logits for compare_kv_runs.py.

Run once with KV_DTYPE=auto and once with KV_DTYPE=fp8 (same mode, same code),
then compare the two dumps. Phases (greedy, raw logits):

  cold      prompts of odd / unaligned lengths, cold cache
  repeat    the same prompts again (prefix-cache hit; prefix modes only)
  partial   a cached prefix of a non-block-aligned length + a new suffix
  batch     16 concurrent prompts whose decode crosses a context bucket
  reuse     ``cold`` again after enough traffic to recycle every KV block

Modes (``mode`` argument):
  m1  full prefill, no prefix caching, DCP 1, standard (non-packed) FP8 K
  m2  packed FP8 K, prefix caching + segmented prefill, DCP 1
  m3  m2 with DCP 8
  m2big  m2 with the full-model prefill buckets 512..8192 (use NUM_BLOCKS=8139)

Usage: KV_DTYPE=auto|fp8 [HY3_KV_SCALE_PROFILE=...] python validate_fp8_kv.py <ckpt> <out.pt> <mode>
Env: NUM_BLOCKS  --num-gpu-blocks-override (small -> block reuse; default 1024)
"""

import os
import random
import sys
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ckpt, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
KV_DTYPE = os.environ.get("KV_DTYPE", "auto")
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "1024"))
STEPS = 12  # greedy steps with full-vocab raw logits

if mode == "m1":
    MAX_LEN, BATCHED, PREFIX, DCP, PACKED = 4096, 4096, False, 1, False
    NC = {
        "num_batched_tokens_buckets": [4096],
        "num_seqs_buckets": [1, 4, 16],
        "decode_context_length_buckets": [2048],
    }
elif mode == "m2big":
    # Full-model prefill buckets / pool on the reduced model (NEFF size, compile
    # time and TTFT of the FP8 prefill read paths).
    MAX_LEN, BATCHED, PREFIX, PACKED, DCP = 16384, 8192, True, True, 1
    NC = {
        "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
        "kv_segment_size_buckets": [2048],
        "num_seqs_buckets": [1, 4, 16],
        "decode_context_length_buckets": [4096, 8192],
    }
elif mode in ("m2", "m3"):
    MAX_LEN, BATCHED, PREFIX, PACKED = 16384, 2048, True, True
    DCP = 8 if mode == "m3" else 1
    NC = {
        "num_batched_tokens_buckets": [2048],
        "kv_segment_size_buckets": [2048],
        "num_seqs_buckets": [1, 4, 16],
        "decode_context_length_buckets": [2048, 8192],
    }
else:
    raise SystemExit(f"unknown mode {mode}")
if KV_DTYPE != "auto" and PACKED:
    NC["fp8_packed_kv"] = True

WORDS_EN = (
    "the model reads every token of the prompt and keeps its keys and values in a "
    "paged cache so that later steps can attend to them without recomputing".split()
)
WORDS_ZH = list("大型语言模型在推理时会把每个词元的键和值保存在分页缓存中以便后续步骤直接读取")
CODE = "def f(x):\n    return [i * x for i in range(10) if i % 3]\n"


def make_prompt(tok, n_tokens: int, seed: int) -> list[int]:
    """Deterministic natural-ish text (EN / ZH / code) cut to exactly n_tokens."""
    r = random.Random(seed)
    parts = []
    while True:
        k = r.random()
        if k < 0.45:
            parts.append(" ".join(r.choice(WORDS_EN) for _ in range(r.randint(8, 20))) + ".")
        elif k < 0.85:
            parts.append("".join(r.choice(WORDS_ZH) for _ in range(r.randint(8, 24))) + "。")
        else:
            parts.append(CODE)
        ids = tok("\n".join(parts), add_special_tokens=False).input_ids
        if len(ids) >= n_tokens:
            return ids[:n_tokens]


def run(llm, prompts, max_tokens, logprobs):
    sp = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logprobs=logprobs, detokenize=False
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    res = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        rec = {"prompt_len": len(p), "gen_ids": list(c.token_ids), "cached": o.num_cached_tokens}
        if logprobs and c.logprobs:
            rec["logits"] = torch.stack(
                [_dense(step, vocab) for step in c.logprobs[:STEPS]]
            ).to(torch.float32)
        res.append(rec)
    return res


def _dense(lp_dict, v):
    t = torch.full((v,), -float("inf"))
    for tid, lp in lp_dict.items():
        if tid < v:
            t[tid] = lp.logprob
    return t


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(ckpt)
    t0 = time.time()
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        max_model_len=MAX_LEN,
        max_num_seqs=16,
        max_num_batched_tokens=BATCHED,
        block_size=32,
        enable_prefix_caching=PREFIX,
        async_scheduling=False,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        kv_cache_dtype=KV_DTYPE,
        num_gpu_blocks_override=NUM_BLOCKS,
        decode_context_parallel_size=DCP,
        cp_kv_cache_interleave_size=32,
        additional_config={"neuron_config": NC},
    )
    vocab = len(tok)
    print(f"### engine ready in {time.time() - t0:.0f}s (mode {mode}, kv {KV_DTYPE})", flush=True)
    result = {"mode": mode, "kv_dtype": KV_DTYPE, "num_blocks": NUM_BLOCKS, "phases": {}}

    if mode == "m1":
        lens = [37, 511, 1000, 2047, 3001]
    else:
        lens = [37, 1000, 2047, 3100, 5000, 9001]
    cold = [make_prompt(tok, n, 1000 + n) for n in lens]
    result["phases"]["cold"] = run(llm, cold, STEPS, vocab)
    if PREFIX:
        result["phases"]["repeat"] = run(llm, cold, STEPS, vocab)
        # Cached prefix of a non-block-aligned length, new suffix.
        partial = [cold[3][:3100 - 45] + make_prompt(tok, 333, 77), cold[4][:4999] + make_prompt(tok, 101, 78)]
        result["phases"]["partial"] = run(llm, partial, STEPS, vocab)
    # Batch decode crossing the 2048 context bucket (prompt 2000-2040 + 64 tokens).
    batch = [make_prompt(tok, 2000 + 3 * b, 5000 + b) for b in range(16)]
    result["phases"]["batch"] = run(llm, batch, 64, 0)
    # Recycle every block: many distinct prompts through a small pool.
    churn = [make_prompt(tok, 1500 + 7 * b, 9000 + b) for b in range(48)]
    run(llm, churn, 4, 0)
    result["phases"]["reuse"] = run(llm, cold, STEPS, vocab)
    torch.save(result, out_path)
    for ph, recs in result["phases"].items():
        print(f"### {ph}: " + "; ".join(f"len {r['prompt_len']} cached {r['cached']}" for r in recs), flush=True)
    print(f"### saved {out_path} ({time.time() - t0:.0f}s)", flush=True)
