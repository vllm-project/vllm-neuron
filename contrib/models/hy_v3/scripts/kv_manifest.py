"""Run manifest for an acceptance run: what was served, with what, from where.

Usage: python kv_manifest.py <serve_log> <out.json> [--repo <vllm-neuron checkout>]

Reads the server log (vLLM non-default args, engine version, KV pool size, the
Hy3 KV scale profile and KV pool lines) and, if a `vllm serve` process is
running, its command line and the NEURON_* / VLLM_* / HY3_* / KV_* / *_BUCKETS
environment. Adds the repo commit and a hash of its uncommitted diff, the
sha256 of the KV scale profile file and the installed package versions.
kv_report.py compares the manifests of the BF16 and FP8 runs.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from importlib import metadata

ENV_RE = re.compile(r"^(NEURON|VLLM|HY3|KV_|NUM_BLOCKS|PREFILL_BUCKETS|KV_SEGMENTS|DCP|MODEL|EXTRA_ARGS|FP8)")
PACKAGES = ("vllm", "vllm-neuron", "neuronx-cc", "nki", "libtorch-neuronx-lite", "torch", "transformers")


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def server_process():
    out = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True, text=True).stdout.split()
    for pid in out:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        cmd = [c.decode() for c in cmd if c]
        if "serve" not in cmd:
            continue
        envd = dict(e.decode().split("=", 1) for e in env if b"=" in e)
        return {"pid": int(pid), "cmdline": cmd,
                "env": {k: v for k, v in sorted(envd.items()) if ENV_RE.match(k)}}
    return None


def parse_log(path):
    text = open(path, errors="replace").read()
    m = {}
    a = re.search(r"non-default args: (\{.*\})", text)
    if a:
        try:
            m["non_default_args"] = ast.literal_eval(a.group(1))
        except (ValueError, SyntaxError):
            m["non_default_args_raw"] = a.group(1)
    v = re.search(r"Initializing a V1 LLM engine \((v[^)]+)\)", text)
    m["vllm_engine"] = v.group(1) if v else None
    p = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    m["kv_cache_tokens"] = int(p.group(1).replace(",", "")) if p else None
    pr = sorted(set(re.findall(r"Hy3 FP8 KV scale profile (\S+) \(sha256 ([0-9a-f]+), (\d+) layers\)", text)))
    m["kv_scale_profile"] = [{"path": x[0], "sha256_prefix": x[1], "layers": int(x[2])} for x in pr]
    pools = sorted(set(re.findall(r"Hy3 KV pool: (.*) on this rank", text)))
    m["kv_pool_per_rank"] = pools
    b = [int(x) for x in re.findall(r"Hy3 KV pool: .*?, (\d+) bytes on this rank", text)]
    m["kv_pool_ranks_logged"] = len(b)
    m["kv_pool_bytes_per_rank"] = sorted(set(b))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serve_log")
    ap.add_argument("out")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--no-process", action="store_true",
                    help="post-hoc manifest from a log only (do not read a running server)")
    a = ap.parse_args()
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "serve_log": os.path.abspath(a.serve_log)}
    man.update(parse_log(a.serve_log))
    proc = None if a.no_process else server_process()
    man["post_hoc"] = a.no_process
    man["server_process"] = proc
    prof = (proc or {}).get("env", {}).get("HY3_KV_SCALE_PROFILE")
    if prof and os.path.exists(prof):
        man["kv_scale_profile_file"] = {"path": prof, "sha256": sha256(prof)}
    if a.repo:
        g = lambda *c: subprocess.run(["git", "-c", "safe.directory=*", "-C", a.repo, *c],
                                      capture_output=True, text=True).stdout
        man["repo"] = {"path": a.repo, "head": g("rev-parse", "HEAD").strip(),
                       "diff_sha256": hashlib.sha256(g("diff", "HEAD").encode()).hexdigest(),
                       "diff_files": g("diff", "--name-only", "HEAD").split()}
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    man["packages"] = vers
    json.dump(man, open(a.out, "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("kv_cache_tokens", "kv_pool_bytes_per_rank", "kv_scale_profile")}))


if __name__ == "__main__":
    sys.exit(main())
