#!/usr/bin/env python3
"""Where the host time of a decode step goes: one server per cell with DEEPSEEK_V41_HOST_PROF=1
(rank 0's per-forward phases and the waits between the runner's RPCs, logged as medians) and
py-spy sampling the engine core, the API server and three workers during steady decode.

  host_prof.py --cell in1024_bs10 --isl 1024 --bs 10 --ms 10 --ctx 1280 [--rounds 2] [--window 3.0]
"""
import argparse, json, os, re, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import profile_cell as pc  # noqa: E402  (prompts, requests, health, fatal-pattern checks)

rb = pc.rb
PYSPY = "/mnt/nvme/ds41_profiling/venv/bin/py-spy"


def pids(server_log):
    txt = server_log.read_text(errors="replace")
    eng = re.search(r"\(EngineCore[^ ]* pid=(\d+)\)", txt)
    api = re.search(r"\(APIServer pid=(\d+)\)", txt)
    w = {int(r): int(p) for r, p in re.findall(r"\(Worker_TP(\d+)_EP\d+ pid=(\d+)\)", txt)}
    return (int(eng.group(1)) if eng else None), (int(api.group(1)) if api else None), w


def spy(pid, out, seconds, rate):
    # (py-spy's --duration takes whole seconds)
    return subprocess.Popen(["sudo", "-n", PYSPY, "record", "-p", str(pid), "-d", str(int(round(seconds))), "-r", str(rate),
                             "--nonblocking", "--idle", "--threads", "-f", "speedscope", "-o", str(out)],
                            stdout=open(f"{out}.log", "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--ms", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=240)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--rate", type=int, default=200)
    ap.add_argument("--ranks", default="0,9,63")
    ap.add_argument("--port", type=int, default=8361)
    a = ap.parse_args()
    if a.bs > 1 and a.tokens == 240:
        # (bs10 at 1024 + 240 tokens crashed the engine near max_model_len 1280: IndexError in the n-gram
        # hash's block-table lookup, model.py slots(), 2026-10-27; the main captures used 192 tokens)
        a.tokens, a.window = 192, 2.0
    root = Path("/mnt/nvme/ds41_profiling_opt6/hostprof") / a.cell
    root.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPSEEK_V41_MAX_NUM_SEQS"] = str(a.ms)
    os.environ["DEEPSEEK_V41_HOST_PROF"] = "1"
    os.environ["DEEPSEEK_V41_HOST_PROF_EVERY"] = "16"
    cmd = [sys.executable, "-m", "performance._serve_main", "--model", "/mnt/nvme/hf",
           "--served-model-name", rb.SERVED_NAME, "--tensor-parallel-size", "64",
           "--pipeline-parallel-size", "1", "--max-model-len", str(a.ctx),
           "--max-num-seqs", str(a.ms), "--port", str(a.port), "--no-enable-prefix-caching"]
    env_keys = [k for k in os.environ if k.startswith(("DEEPSEEK_V41", "NEURON_RT", "VLLM"))]
    (root / "run_spec.json").write_text(json.dumps({"args": vars(a), "cmd": cmd,
                                                    "env": {k: os.environ[k] for k in sorted(env_keys)}}, indent=1))
    import ctypes, signal

    def child():
        os.setsid()
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    server_log = root / "server.log"
    proc = subprocess.Popen(cmd, stdout=server_log.open("w"), stderr=subprocess.STDOUT, preexec_fn=child,
                            cwd=str(pc.TEST))
    windows = []
    try:
        rb._wait_healthy(proc, a.port, 7200)
        warm = pc.build_prompts(a.bs, a.isl, seed=3)
        pc.log("warm-up")
        ths, recs = pc.run_batch(a.port, warm, 32, True)
        pc.join_all(ths, proc, server_log)
        eng, api, workers = pids(server_log)
        ranks = [int(r) for r in a.ranks.split(",")]
        pc.log(f"pids: engine {eng}, api {api}, workers {[workers.get(r) for r in ranks]}")
        for rnd in range(a.rounds):
            prompts = pc.build_prompts(a.bs, a.isl, seed=100 + rnd)
            ths, recs = pc.run_batch(a.port, prompts, a.tokens, True)
            while not all(len(r.get("chunks", [])) >= 3 for r in recs):
                if pc.engine_fatal(proc, server_log):
                    raise SystemExit("engine failure")
                time.sleep(0.005)
            time.sleep(0.3)
            t0 = time.time()
            sp = [spy(eng, root / f"spy_engine_r{rnd}.json", a.window, a.rate),
                  spy(api, root / f"spy_api_r{rnd}.json", a.window, a.rate)]
            sp += [spy(workers[r], root / f"spy_worker{r}_r{rnd}.json", a.window, a.rate) for r in ranks]
            for p in sp:
                p.wait()
            # the decode steps inside the window: count the stream chunks of request 0
            pc.join_all(ths, proc, server_log)
            ch = [c[0] for c in recs[0]["chunks"]]
            n = sum(1 for c in ch if t0 <= c < t0 + a.window)
            windows.append({"round": rnd, "t0": t0, "window_s": a.window, "chunks_req0_in_window": n})
            pc.log(f"round {rnd}: {n} stream chunks of request 0 in the {a.window} s window")
        (root / "windows.json").write_text(json.dumps(windows, indent=1))
        time.sleep(2)
    finally:
        rb._stop(proc)
        pc.log("server stopped")


if __name__ == "__main__":
    main()
