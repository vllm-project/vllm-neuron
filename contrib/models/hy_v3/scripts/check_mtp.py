"""Greedy MTP speculative decoding vs plain decoding on the same prompts.

Greedy speculative decoding is lossless, so both runs must produce the same
tokens. Also reports the draft acceptance. On a reduced checkpoint
(``make_mini.py --keep-mtp``) the acceptance is near zero, because the MTP
layer was trained on the full model's hidden states; the token comparison is
the check there.

Usage: python check_mtp.py <ckpt> <num_speculative_tokens> [max_tokens=64] [mode=both|mtp|plain]
Env: KV_DTYPE=fp8 for an FP8 KV cache (packed K; needs HY3_KV_SCALE_PROFILE
with the MTP layer), PREFIX_CACHING=1 for prefix caching (segmented prefill),
DUMP_IDS=<file.json> to save both runs' token ids for check_mtp_gaps.py, which
tells near-tie flips from real divergence with an HF reference.
Needs a vllm-neuron with MTP support (vllm_neuron/vllm/spec_decode/mtp.py) and
the contrib package (pip install -e contrib/models/hy_v3).
"""

import json
import os
import sys
import time

from vllm import LLM, SamplingParams

ckpt, k = sys.argv[1], int(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
mode = sys.argv[4] if len(sys.argv) > 4 else "both"
fp8_kv = os.environ.get("KV_DTYPE") == "fp8"
prefix = os.environ.get("PREFIX_CACHING") == "1"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "人工智能的未来发展趋势是",
    "Question: what is 17 * 23? Answer:",
]
if os.environ.get("LONG_PROMPT"):  # LONG_PROMPT=<tokens>: one long prompt per entry (past the 512 prefill bucket)
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(ckpt)
    _n = int(os.environ["LONG_PROMPT"])
    _base = ("The history of the river town begins with a mill, a bridge and a market square. " * 2000)
    PROMPTS = [_tok.decode(_tok(f"Story {i}. " + _base)["input_ids"][:_n]) + " The mill" for i in range(4)]


def run(spec: bool):
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        # DCP needs segmented prefill: max_num_batched_tokens < max_model_len, kv segments.
        max_model_len=int(os.environ.get("MAX_LEN", 4096 if int(os.environ.get("DCP", "1")) > 1 else 2048)),
        max_num_seqs=len(PROMPTS),
        max_num_batched_tokens=max(json.loads(os.environ.get("BUCKETS", "[2048]"))),
        block_size=32,
        enable_prefix_caching=prefix,
        # DCP=<n>: decode context parallelism (with MTP needs VLLM_NEURON_DCP_SPEC_DECODE=1)
        **({"decode_context_parallel_size": int(os.environ["DCP"]), "cp_kv_cache_interleave_size": 32}
           if int(os.environ.get("DCP", "1")) > 1 else {}),
        kv_cache_dtype="fp8" if fp8_kv else "auto",
        disable_log_stats=False,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": k} if spec else None
        ),
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": json.loads(os.environ.get("BUCKETS", "[2048]")),
                "num_seqs_buckets": [1, len(PROMPTS)],
                **({"kv_segment_size_buckets": [2048]} if prefix or int(os.environ.get("DCP", "1")) > 1 else {}),
                **({"fp8_packed_kv": True} if fp8_kv else {}),
            }
        },
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.time()
    outs = llm.generate(PROMPTS, sp)
    dt = time.time() - t0
    ids = [list(o.outputs[0].token_ids) for o in outs]
    stats = {
        m.name.removeprefix("vllm:spec_decode_"): getattr(m, "value", None)
        or getattr(m, "values", None)
        for m in llm.get_metrics()
        if "spec_decode" in m.name
    }
    llm.llm_engine.engine_core.shutdown()
    return ids, dt, stats


results = {}
if mode in ("both", "plain"):
    results["plain"] = run(False)
if mode in ("both", "mtp"):
    results["mtp"] = run(True)
for name, (_, dt, stats) in results.items():
    line = f"### {name}: generate {dt:.1f}s"
    if stats:
        drafts = stats.get("num_draft_tokens") or 0
        per_pos = stats.get("num_accepted_tokens_per_pos") or []
        accepted = sum(per_pos)
        rate = accepted / drafts if drafts else 0.0
        line += f"; draft tokens {drafts}, accepted {accepted} ({rate:.1%}), per position {per_pos}"
    print(line)
if os.environ.get("DUMP_IDS"):  # token ids of each run, for an offline reference check
    import json

    with open(os.environ["DUMP_IDS"], "w") as f:
        json.dump({"prompts": PROMPTS, **{k: v[0] for k, v in results.items()}}, f)
if len(results) == 2:
    for i, (x, y) in enumerate(zip(results["plain"][0], results["mtp"][0])):
        same = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
        print(f"### prompt {i}: identical prefix {same}/{len(x)}")
