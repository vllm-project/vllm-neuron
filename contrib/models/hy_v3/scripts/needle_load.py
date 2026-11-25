"""Needle test under decode load, so the needles decode in batch buckets >= 8 (the KVDP path).

  python scripts/needle_load.py bench_data/needle.jsonl.gz bench_data/prefix/in85000_p78336_c8.jsonl.gz [base_url] [n_fill]

n_fill background chat requests (shared-prefix prompts, 1500 output tokens, temperature 0.9)
keep the decode batch full while the needle prompts run one after another (max_tokens 16,
greedy). Prints PASS / FAIL per needle and the total.
"""
import gzip
import json
import sys
import threading
import time
import urllib.request

needle_path, prefix_path = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"
n_fill = int(sys.argv[4]) if len(sys.argv) > 4 else 8


def chat(content, max_tokens, temperature, extra=None):
    body = {"model": "hy3", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": temperature}
    body.update(extra or {})
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


with gzip.open(prefix_path, "rt") as f:
    rows = [json.loads(line) for line in f]
prefix = rows[0]["prefix"]
fills = [prefix + r["suffix"] for r in rows[1: 1 + n_fill]]
done = threading.Event()


def filler(text):
    while not done.is_set():
        chat(text, 1500, 0.9, {"ignore_eos": True})


threads = [threading.Thread(target=filler, args=(t,), daemon=True) for t in fills]
for t in threads:
    t.start()
time.sleep(90)  # all fillers prefilled and decoding

with gzip.open(needle_path, "rt") as f:
    needles = [json.loads(line) for line in f]
ok = 0
for r in needles:
    t0 = time.time()
    out = chat(r["prompt"], 16, 0)
    ans = out["choices"][0]["message"]["content"]
    got = r["key"] in ans
    ok += got
    print(f"prompt_tokens={out['usage']['prompt_tokens']:>6} depth={r['depth']:.1f} key={r['key']} "
          f"answer={ans.strip()!r} {'PASS' if got else 'FAIL'} ({time.time() - t0:.1f}s)", flush=True)
done.set()
print(f"{ok}/{len(needles)} passed under load ({n_fill} fillers)")
