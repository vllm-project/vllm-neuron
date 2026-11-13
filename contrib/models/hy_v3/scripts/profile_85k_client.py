"""Profiling sessions on the 85k shared-prefix prompts, against scripts/profile_85k.sh.

Sessions (each between /start_profile and /stop_profile, so each is its own
profile on every profiled rank):

  c1_cold     one request with a new cache_salt: the whole 85k prompt
              prefills (4096-token chunks), then a few decode steps.
  c1_decode   one request whose 78k prefix is cached, profiled while it decodes.
  cN_decode   N such requests at once (--batch, default 6: the most that fit
              next to the profiler's buffers), profiled once all N decode.

Prompts are bench_data/prefix/in85000_p78336_c8 (85000 tokens, 92% shared
prefix), sent as chat messages like bench_pd_split.sh (temperature 0.9).
Usage: python profile_85k_client.py [--base-url http://127.0.0.1:8000] [--batch 6]
       [--sessions c1_cold,c1_decode,cN_decode]
"""

import argparse
import gzip
import json
import threading
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "bench_data/prefix/in85000_p78336_c8.jsonl.gz"


def post(base, path, timeout=1800):
    req = urllib.request.Request(base + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def prompts():
    with gzip.open(PROMPTS, "rt") as f:
        rows = [json.loads(line) for line in f]
    return [rows[0]["prefix"] + r["suffix"] for r in rows[1:]]


class Request(threading.Thread):
    """One streaming chat request; records when its first token arrives."""

    def __init__(self, client, prompt, salt, max_tokens):
        super().__init__(daemon=True)
        self.client, self.prompt, self.salt, self.max_tokens = client, prompt, salt, max_tokens
        self.first_token = threading.Event()
        self.n_chunks = 0

    def run(self):
        stream = self.client.chat.completions.create(
            model="hy3",
            messages=[{"role": "user", "content": self.prompt}],
            max_tokens=self.max_tokens,
            temperature=0.9,
            stream=True,
            extra_body={"cache_salt": self.salt, "ignore_eos": True},
        )
        for _ in stream:
            self.n_chunks += 1
            self.first_token.set()


def run_requests(client, ps, salt, max_tokens):
    rs = [Request(client, p, salt, max_tokens) for p in ps]
    for r in rs:
        r.start()
    return rs


def profiled(base, name, body):
    t0 = time.time()
    post(base, "/start_profile")
    body()
    post(base, "/stop_profile")
    print(f"[{name}] profiled {time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sessions", default="c1_cold,c1_decode,cN_decode")
    ap.add_argument("--batch", type=int, default=6, help="requests decoding at once in cN_decode")
    ap.add_argument("--decode-window", type=float, default=2.0, help="seconds profiled while decoding")
    a = ap.parse_args()
    base, sessions = a.base_url, a.sessions.split(",")
    client = OpenAI(base_url=base + "/v1", api_key="EMPTY", timeout=3600)
    ps = prompts()
    salt = f"profile-{int(time.time())}"

    if "c1_cold" in sessions:
        def cold():
            [r] = run_requests(client, ps[:1], salt + "-cold", 16)
            r.join()
        profiled(base, "c1_cold", cold)

    # Fill the shared prefix once (not profiled); later requests hit it.
    [w] = run_requests(client, ps[-1:], salt, 1)
    w.join()

    if "c1_decode" in sessions:
        def c1():
            [r] = run_requests(client, ps[:1], salt, 900)
            r.first_token.wait()
            time.sleep(1.0)
            post(base, "/start_profile")
            time.sleep(a.decode_window)
            post(base, "/stop_profile")
            r.join()
        t0 = time.time()
        c1()
        print(f"[c1_decode] done {time.time() - t0:.1f} s", flush=True)

    if "cN_decode" in sessions:
        n = a.batch
        rs = run_requests(client, ps[1 : 1 + n], salt, 900)
        for r in rs:
            r.first_token.wait()
        time.sleep(1.0)
        print(f"[c{n}_decode] all {n} decoding, chunks so far:", [r.n_chunks for r in rs], flush=True)
        profiled(base, f"c{n}_decode", lambda: time.sleep(a.decode_window))
        for r in rs:
            r.join()


if __name__ == "__main__":
    main()
