#!/usr/bin/env python3
"""Profile one DeepSeek-V4.1 tp64 serving config: a prefill window and a decode window.

One server per (context, max_num_seqs); NRT device + system profile on rank 0, toggled
with /start_profile and /stop_profile (max_iterations 0: the windows are explicit).

  prefill: server idle -> start -> <bs> requests with max_tokens=1 -> all done -> stop
  decode : <bs> streaming requests; once every stream has emitted >= 3 chunks (all
           prefills done, steady decode) -> start -> sleep <window> -> stop
"""
import argparse, json, os, random, sys, threading, time, urllib.request
from pathlib import Path

TEST = Path("/home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test")
sys.path.insert(0, str(TEST))
from performance import run_bench as rb  # noqa: E402  (loads the port, sets the host env)

SONNET = TEST / "performance" / "sonnet.txt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompts(n, isl, seed):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/nvme/hf")
    lines = [l for l in SONNET.read_text().splitlines() if l.strip()]
    out = []
    for i in range(n):
        rng = random.Random(seed * 1000 + i)
        ids = []
        while len(ids) < isl:
            ids += tok.encode(rng.choice(lines) + "\n", add_special_tokens=False)
        out.append(ids[:isl])
    return out


def post(port, path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def complete(port, prompt, max_tokens, rec, stream):
    body = {"model": rb.SERVED_NAME, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": stream}
    t0 = time.time()
    rec["t_send"] = t0
    rec["chunks"] = []
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        if not stream:
            j = json.loads(r.read())
            rec["t_done"] = time.time()
            rec["text"] = j["choices"][0]["text"]
            return
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            rec["chunks"].append((time.time(), len(j["choices"][0].get("text", ""))))
    rec["t_done"] = time.time()


def run_batch(port, prompts, max_tokens, stream):
    recs = [dict() for _ in prompts]
    ths = [threading.Thread(target=complete, args=(port, p, max_tokens, r, stream), daemon=True)
           for p, r in zip(prompts, recs)]
    for t in ths:
        t.start()
    return ths, recs


_FATAL = rb._FATAL_PATTERNS + ("notification queue overflow", "multiproc_executor.py:1055] Traceback")


def engine_fatal(proc, server_log):
    if proc.poll() is not None:
        return f"server exited {proc.returncode}"
    try:
        size = server_log.stat().st_size
        with server_log.open("rb") as fh:
            fh.seek(max(0, size - 512 * 1024))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for pat in _FATAL:
        if pat in tail:
            return pat
    return None


def join_all(ths, proc, server_log, timeout=3600):
    deadline = time.time() + timeout
    while any(t.is_alive() for t in ths):
        bad = engine_fatal(proc, server_log)
        if bad:
            raise SystemExit(f"engine failure: {bad}")
        if time.time() > deadline:
            raise SystemExit("requests timed out")
        time.sleep(0.5)


def stamp():
    return {"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--ms", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--decode-window", type=float, default=0.15)
    ap.add_argument("--phases", default="prefill,decode")
    ap.add_argument("--port", type=int, default=8331)
    ap.add_argument("--sys-events", type=int, default=8_000_000)
    ap.add_argument("--warm-rounds", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--delay-iters", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ranks", default="0", help="comma list of worker ranks with a device profile")
    args = ap.parse_args()

    # The "_c2" captures were meant for the last prefill chunk: an 8192-token prompt is 4 chunks of
    # 2048 rows (vLLM's max_num_batched_tokens 2048 is below the port's 4096 segment, which only
    # clamps larger budgets), and vLLM's profiler starts at the delay_iterations-th iteration -> 4.
    if args.tag == "_c2":
        args.delay_iters = 4
    # the port clamps every multi-rank engine to DEEPSEEK_V41_MAX_NUM_SEQS sequences (default 1)
    os.environ["DEEPSEEK_V41_MAX_NUM_SEQS"] = str(args.ms)
    root = Path("/mnt/nvme/ds41_profiling_opt6/profiles") / (args.cell + args.tag)
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "output"
    prof = {"activities": ["device_profile", "system_profile"], "neuron_cores": [int(r) for r in args.ranks.split(",")],
            "output_dir": str(out_dir), "sys_trace_max_events_per_nc": args.sys_events}
    cmd = [sys.executable, "-m", "performance._serve_main", "--model", "/mnt/nvme/hf",
           "--served-model-name", rb.SERVED_NAME, "--tensor-parallel-size", "64",
           "--pipeline-parallel-size", "1", "--max-model-len", str(args.ctx),
           "--max-num-seqs", str(args.ms), "--port", str(args.port), "--no-enable-prefix-caching",
           "--additional-config", json.dumps({"neuron_profiler": prof}),
           "--profiler-config", json.dumps({"profiler": "cuda", "delay_iterations": args.delay_iters, "max_iterations": args.max_iters})]
    env_keys = [k for k in os.environ if k.startswith(("DEEPSEEK_V41", "NEURON_RT", "VLLM", "XLA", "NEURON_FRAMEWORK"))]
    meta = {"cell": vars(args), "cmd": cmd, "env": {k: os.environ[k] for k in sorted(env_keys)}}
    (root / "run_spec.json").write_text(json.dumps(meta, indent=2))
    log("serve: " + " ".join(cmd))

    import ctypes, signal, subprocess
    def child():
        os.setsid()
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    server_log = root / "server.log"
    proc = subprocess.Popen(cmd, stdout=server_log.open("w"), stderr=subprocess.STDOUT,
                            preexec_fn=child, cwd=str(TEST))
    result = {"cell": args.cell, "windows": {}}
    try:
        rb._wait_healthy(proc, args.port, 7200)
        prompts = build_prompts(args.bs, args.isl, seed=7)
        warm = build_prompts(args.bs, args.isl, seed=3)
        for w in range(args.warm_rounds):
            log(f"warm-up {w}: {args.bs} x {args.isl} -> {args.decode_tokens}")
            ths, recs = run_batch(args.port, warm, args.decode_tokens, True)
            join_all(ths, proc, server_log)
            log("warm-up prefill-only pass")
            ths, recs = run_batch(args.port, warm, 1, False)
            join_all(ths, proc, server_log)
        time.sleep(3)

        if "prefill" in args.phases:
            w = {"start": stamp()}
            post(args.port, "/start_profile")
            w["started"] = stamp()
            log("prefill window open")
            ths, recs = run_batch(args.port, prompts, 1, False)
            join_all(ths, proc, server_log)
            w["requests_done"] = stamp()
            post(args.port, "/stop_profile")
            w["stopped"] = stamp()
            w["requests"] = [{"t_send": r["t_send"], "t_done": r["t_done"]} for r in recs]
            log(f"prefill window closed; request latency {[round(r['t_done']-r['t_send'],3) for r in recs]}")
            result["windows"]["prefill"] = w
            time.sleep(5)

        if "decode" in args.phases:
            dp = build_prompts(args.bs, args.isl, seed=11)
            ths, recs = run_batch(args.port, dp, args.decode_tokens, True)
            while not all(len(r.get("chunks", [])) >= 3 for r in recs):
                if not any(t.is_alive() for t in ths):
                    raise SystemExit("decode requests finished before steady state")
                bad = engine_fatal(proc, server_log)
                if bad:
                    raise SystemExit(f"engine failure: {bad}")
                time.sleep(0.005)
            time.sleep(0.2)
            w = {"start": stamp()}
            post(args.port, "/start_profile")
            w["started"] = stamp()
            time.sleep(args.decode_window)
            w["stop"] = stamp()
            post(args.port, "/stop_profile")
            w["stopped"] = stamp()
            log("decode window closed")
            join_all(ths, proc, server_log)
            w["requests"] = [{"t_send": r["t_send"], "t_done": r["t_done"],
                              "chunks": r["chunks"]} for r in recs]
            result["windows"]["decode"] = w
        (root / "windows.json").write_text(json.dumps(result, indent=1))
        log("done")
    finally:
        rb._stop(proc)
        log("server stopped")


if __name__ == "__main__":
    main()
