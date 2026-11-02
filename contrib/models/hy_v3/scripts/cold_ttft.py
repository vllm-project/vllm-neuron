"""Cold-prefill TTFT: n requests one at a time, each with its own cache_salt (no prefix hit), output 1 token.

Prompts: a bench_data/prefix file (row 0 the shared prefix, then the suffixes).
Usage: python cold_ttft.py bench_data/prefix/in85000_p78336_c8.jsonl.gz 4   (server on 127.0.0.1:8000)
"""
import gzip, json, sys, time, urllib.request
rows = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
ts = []
for i in range(int(sys.argv[2])):
    body = {"model": "hy3", "prompt": rows[0]["prefix"] + rows[1 + i]["suffix"], "max_tokens": 1,
            "cache_salt": f"cold-{time.time()}-{i}"}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); ts.append(time.time() - t0)
    print(f"cold request {i}: {ts[-1]:.2f} s, prompt tokens {r['usage']['prompt_tokens']}", flush=True)
print(f"cold TTFT median {sorted(ts)[len(ts) // 2]:.2f} s")
