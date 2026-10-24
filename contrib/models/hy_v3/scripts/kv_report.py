"""BF16-KV vs FP8-KV acceptance gate over two kv_acceptance.sh output dirs.

Usage: python kv_report.py <bf16_dir> <fp8_dir> [--checks provenance,memory,accuracy,perf,stability]
           [--max-regress 5] [--max-acc-drop 1.0] [--min-stab-seconds 1800] [--json out.json]

Every check reads raw artifacts only and is PASS / FAIL / MISSING (missing
evidence fails the gate). The exit code is 0 only if every selected check passes.

  provenance  manifest.json of both runs (kv_manifest.py): identical vLLM args
              except kv_cache_dtype / num_gpu_blocks_override / fp8_packed_kv,
              the same KV pool token count, BF16 run without and FP8 run with
              one 80-layer KV scale profile (file sha256 matching the logged one)
  memory      persistent KV pool: exact per-rank bytes from the "Hy3 KV pool"
              log line, FP8 / BF16 = 0.5; device "tensors" difference = that pool
              difference (1 %); peak over the run (mem_timeseries.jsonl, every
              runtime category): the effective saving, BF16 peak device total
              minus FP8 peak device total, must be >= --min-peak-saving (default
              0.45) of the BF16 KV pool ("KV memory about -50 %" including every
              FP8-only overhead); each non-KV category that grows is listed (WARN;
              FAIL with --strict-transient); reports the BF16 request-block copy
              bound per prefill step
  accuracy    MATH-500 paired per problem: same 500 problems and sampling config
              (.meta.json if present), every problem re-graded with math-verify
              here, accuracy drop <= --max-acc-drop pp; reports discordant pairs,
              exact McNemar p and a paired-bootstrap 95 % CI of the difference
  perf        load matrix: every cell complete in both runs (no failed request,
              same prompt count and total input tokens), P50 TTFT and TPOT
              regression <= --max-regress %; P99 and throughput reported
  stability   both runs: duration >= --min-stab-seconds, all requests completed,
              0 new server-log error / NaN lines; preemptions and prefix hits as
              the difference of the /metrics counters before and after
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

IGNORED_ARGS = {"kv_cache_dtype", "num_gpu_blocks_override"}
RESULTS = []


def check(name, status, detail):
    RESULTS.append({"check": name, "status": status, "detail": detail})
    print(f"- **{name}**: {status} — {detail}")


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() and p.read_text().strip() else None


def strip_args(args):
    a = {k: v for k, v in (args or {}).items() if k not in IGNORED_ARGS}
    nc = (a.get("additional_config") or {}).get("neuron_config")
    if nc:
        nc = {k: v for k, v in nc.items() if k != "fp8_packed_kv"}
        a["additional_config"] = {**a["additional_config"], "neuron_config": nc}
    return a


# ── provenance ────────────────────────────────────────────────────────────


def provenance(B, F):
    print("\n## Provenance\n")
    mb, mf = load_json(B / "manifest.json"), load_json(F / "manifest.json")
    if not (mb and mf):
        return check("provenance", "MISSING", "manifest.json missing in " + ", ".join(
            str(d) for d, m in ((B, mb), (F, mf)) if not m))
    ab, af = mb.get("non_default_args"), mf.get("non_default_args")
    errs = []
    if ab is None or af is None:
        errs.append("non-default args not parsed from a server log")
    else:
        if af.get("kv_cache_dtype") != "fp8":
            errs.append(f"FP8 run kv_cache_dtype={af.get('kv_cache_dtype')}")
        if ab.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
            errs.append(f"BF16 run kv_cache_dtype={ab.get('kv_cache_dtype')}")
        sb, sf = strip_args(ab), strip_args(af)
        diff = sorted(k for k in set(sb) | set(sf) if sb.get(k) != sf.get(k))
        if diff:
            errs.append("args differ: " + ", ".join(f"{k}={sb.get(k)!r}/{sf.get(k)!r}" for k in diff))
    if mb.get("kv_cache_tokens") != mf.get("kv_cache_tokens") or not mb.get("kv_cache_tokens"):
        errs.append(f"KV pool tokens {mb.get('kv_cache_tokens')} vs {mf.get('kv_cache_tokens')}")
    if mb.get("kv_scale_profile"):
        errs.append("BF16 run loaded a KV scale profile")
    prof = mf.get("kv_scale_profile") or []
    if len(prof) != 1 or prof[0]["layers"] != 80:
        errs.append(f"FP8 run: expected one 80-layer profile, log shows {prof}")
    pf = mf.get("kv_scale_profile_file")
    if pf and prof and not pf["sha256"].startswith(prof[0]["sha256_prefix"]):
        errs.append("profile file sha256 does not match the logged one")
    for lab, m in (("BF16", mb), ("FP8", mf)):
        if m.get("vllm_engine") != mb.get("vllm_engine"):
            errs.append(f"engine versions differ ({lab})")
    detail = (f"pool {mf.get('kv_cache_tokens')} tokens; profile "
              f"{pf['sha256'][:16] if pf else (prof[0]['sha256_prefix'] if prof else None)}; "
              f"repo {(mf.get('repo') or {}).get('head', 'n/a')[:12]}; "
              f"server env captured: {bool(mf.get('server_process'))}")
    check("provenance", "FAIL" if errs else "PASS", "; ".join(errs) if errs else detail)


# ── memory ────────────────────────────────────────────────────────────────


def snapshot(d):
    rep = d / "neuron_monitor.json"
    if not rep.exists() or not rep.read_text().strip():
        return None
    r = json.loads(rep.read_text().splitlines()[0])
    out = {}
    for rt in r.get("neuron_runtime_data", []):
        mu = rt["report"]["memory_used"]["neuron_runtime_used_bytes"]
        for core in mu.get("usage_breakdown", {}).get("neuroncore_memory_usage", {}).values():
            for k, v in core.items():
                out[k] = out.get(k, 0) + v
    return out


def peaks(d):
    p = d / "mem_timeseries.jsonl"
    if not p.exists():
        return None, 0, 0
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("runtimes")]
    if not rows:
        return None, 0, 0
    pk = {}
    for r in rows:
        for k, v in r["device"].items():
            pk[k] = max(pk.get(k, 0), v)
    pk["device_total"] = max(r["device_total"] for r in rows)
    # Boundedness: growth of each category over the last quarter of the run.
    q = rows[len(rows) * 3 // 4:]
    pk["_late_growth"] = {k: max(r["device"].get(k, 0) for r in q) - q[0]["device"].get(k, 0)
                          for k in q[0]["device"]}
    return pk, len(rows), rows[-1]["t"] - rows[0]["t"]


def memory(B, F, min_peak_saving=0.45, strict=False):
    print("\n## Memory\n")
    mb, mf = load_json(B / "manifest.json") or {}, load_json(F / "manifest.json") or {}
    pb, pf = mb.get("kv_pool_bytes_per_rank") or [], mf.get("kv_pool_bytes_per_rank") or []
    G = 2**30
    if len(pb) == 1 and len(pf) == 1:
        ranks = mf.get("kv_pool_ranks_logged") or 0
        ratio = pf[0] / pb[0]
        ok = abs(ratio - 0.5) <= 0.005
        check("memory: persistent KV pool", "PASS" if ok else "FAIL",
              f"per rank {pb[0] / G:.3f} → {pf[0] / G:.3f} GiB (FP8/BF16 = {ratio:.4f}); "
              f"{ranks} ranks → {pb[0] * ranks / G:.2f} → {pf[0] * ranks / G:.2f} GiB")
        pool_delta = (pb[0] - pf[0]) * ranks
    else:
        check("memory: persistent KV pool", "MISSING",
              f"'Hy3 KV pool' log line (exact bytes) not in both runs ({pb} / {pf})")
        pool_delta = None
    sb, sf = snapshot(B), snapshot(F)
    if sb and sf and pool_delta:
        d = sb["tensors"] - sf["tensors"]
        ok = abs(d - pool_delta) <= 0.01 * pool_delta
        check("memory: device tensors snapshot", "PASS" if ok else "FAIL",
              f"tensors {sb['tensors'] / G:.2f} → {sf['tensors'] / G:.2f} GiB, difference "
              f"{d / G:.2f} GiB vs KV pool difference {pool_delta / G:.2f} GiB")
    elif not (sb and sf):
        check("memory: device tensors snapshot", "MISSING", "neuron_monitor.json missing")
    kb, nb, tb = peaks(B)
    kf, nf, tf = peaks(F)
    if not (kb and kf):
        check("memory: peak (persistent + transient)", "MISSING",
              "mem_timeseries.jsonl (peak over the run) missing in "
              + ", ".join(str(x) for x, k in ((B, kb), (F, kf)) if not k))
    else:
        print("| peak over the run (GiB, all ranks) | BF16 KV | FP8 KV | Δ |\n|---|---|---|---|")
        grow = []
        late = {lab: {k: v for k, v in p.pop("_late_growth").items() if v > 0.01 * max(p.get(k, 1), 1)}
                for lab, p in (("BF16", kb), ("FP8", kf))}
        for k in sorted(set(kb) | set(kf)):
            a, b = kb.get(k, 0), kf.get(k, 0)
            print(f"| {k} | {a / G:.2f} | {b / G:.2f} | {(b - a) / G:+.2f} |")
            if k in ("tensors", "device_total"):
                continue
            if b > a * 1.01 + 64 * 2**20:
                grow.append(f"{k} +{(b - a) / G:.2f} GiB")
        if pool_delta and kf["tensors"] + pool_delta > kb["tensors"] * 1.01:
            # Peak tensors minus each run's own KV pool: weights + other persistent tensors.
            grow.append(f"non-KV tensors +{(kf['tensors'] + pool_delta - kb['tensors']) / G:.2f} GiB")
        print(f"\n({nb} / {nf} samples over {tb / 60:.0f} / {tf / 60:.0f} min)\n")
        errs = []
        if pool_delta:
            bf16_pool = 2 * pool_delta  # BF16 pool = 2 x the difference (FP8 / BF16 = 0.5)
            saved = kb["device_total"] - kf["device_total"]
            frac = saved / bf16_pool
            print(f"Effective peak saving {saved / G:.2f} GiB = {100 * frac:.1f} % of the BF16 KV pool "
                  f"({bf16_pool / G:.2f} GiB); FP8-only overhead at peak {(pool_delta - saved) / G:+.2f} GiB.\n")
            if frac < min_peak_saving:
                errs.append(f"effective peak saving {100 * frac:.1f} % < {100 * min_peak_saving:.0f} %")
        for lab, lg in late.items():
            if lg:
                grow.append(f"{lab} still growing in the last quarter: "
                            + ", ".join(f"{k} +{v / G:.2f} GiB" for k, v in lg.items()))
        if grow:
            print("WARN non-KV categories larger with FP8 at peak: " + "; ".join(grow) + "\n")
            if strict:
                errs += grow
        check("memory: peak (persistent + transient)", "FAIL" if errs else "PASS",
              "; ".join(errs) if errs else
              (f"effective peak saving {100 * frac:.1f} % of the BF16 KV pool"
               + (f"; WARN {'; '.join(grow)}" if grow else "; no non-KV category grows")))
    args = mf.get("non_default_args") or {}
    if args:
        W = math.ceil(args.get("max_model_len", 0) / args.get("block_size", 32))
        per_layer = W * args.get("block_size", 32) * 128 * 2 * 2
        print(f"BF16 request-block copy in FP8 segmented prefill (bound, per rank, per layer): "
              f"{W} blocks × {args.get('block_size', 32)} × 128 × 2 B × K,V = {per_layer / 2**20:.1f} MiB; "
              f"it lives inside the prefill NEFF (scratchpad / NEFF tensors), so it is covered by the peak check.\n")


# ── accuracy ──────────────────────────────────────────────────────────────


def regrade():
    """The evaluator's own grader (eval_math500.grade), so both runs are graded identically."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from eval_math500 import grade
    except ImportError:
        return None
    return grade


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def accuracy(B, F, max_drop, name="greedy"):
    print("\n## Accuracy (MATH-500, paired)\n")
    fb, ff = B / f"math500_{name}.jsonl", F / f"math500_{name}.jsonl"
    if not (fb.exists() and ff.exists()):
        return check("accuracy", "MISSING", f"{fb if not fb.exists() else ff} missing")
    rb = [json.loads(x) for x in fb.read_text().splitlines() if x.strip()]
    rf = [json.loads(x) for x in ff.read_text().splitlines() if x.strip()]
    errs = []
    ib, if_ = {r["unique_id"]: r for r in rb}, {r["unique_id"]: r for r in rf}
    if len(rb) != 500 or len(rf) != 500 or len(ib) != 500 or set(ib) != set(if_):
        errs.append(f"problem sets differ or incomplete ({len(rb)}/{len(ib)}, {len(rf)}/{len(if_)})")
    for lab, r in (("BF16", rb), ("FP8", rf)):
        bad = [x["unique_id"] for x in r if x.get("text") is None or x.get("finish_reason") is None]
        if bad:
            errs.append(f"{lab}: {len(bad)} problems without a completion")
    meta = [load_json(str(p) + ".meta.json") for p in (fb, ff)]
    if all(meta):
        keys = ("reasoning_effort", "temperature", "top_p", "max_tokens", "seed", "prompt_template",
                "problems_sha256", "math_verify")
        d = [k for k in keys if meta[0].get(k) != meta[1].get(k)]
        if d:
            errs.append("sampling / grader config differs: " + ", ".join(d))
        cfg = f"temperature {meta[1]['temperature']}, max_tokens {meta[1]['max_tokens']}, math-verify {meta[1]['math_verify']}"
    else:
        cfg = "no .meta.json (config from the acceptance log: temperature 0, max_tokens 4096, no_think)"
    grade = regrade()
    regraded = 0
    if grade:
        for r in rb + rf:
            g = grade(r.get("text") or "", r["answer"])
            regraded += g != r["correct"]
            r["correct"] = g
    common = sorted(set(ib) & set(if_))
    n11 = sum(ib[u]["correct"] and if_[u]["correct"] for u in common)
    n10 = sum(ib[u]["correct"] and not if_[u]["correct"] for u in common)
    n01 = sum(not ib[u]["correct"] and if_[u]["correct"] for u in common)
    n = len(common)
    ab, af = (n11 + n10) / n * 100, (n11 + n01) / n * 100
    delta = af - ab
    rnd = random.Random(0)
    diffs = [(if_[u]["correct"] - ib[u]["correct"]) for u in common]
    boots = sorted(sum(rnd.choice(diffs) for _ in range(n)) / n * 100 for _ in range(4000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    trb = sum(r["finish_reason"] == "length" for r in rb)
    trf = sum(r["finish_reason"] == "length" for r in rf)
    print(f"| | BF16 KV | FP8 KV |\n|---|---|---|\n| accuracy | {ab:.1f}% ({n11 + n10}/{n}) | {af:.1f}% ({n11 + n01}/{n}) |"
          f"\n| truncated (length) | {trb} | {trf} |\n")
    print(f"Paired: both right {n11}, BF16 only {n10}, FP8 only {n01}, both wrong {n - n11 - n10 - n01}; "
          f"exact McNemar p = {mcnemar_p(n10, n01):.3f}; Δ = {delta:+.1f} pp, paired-bootstrap 95 % CI "
          f"[{lo:+.1f}, {hi:+.1f}] pp. Re-graded here: {'yes' if grade else 'NO (math-verify not importable)'}"
          f"{f', {regraded} stored grades changed' if grade else ''}. Config: {cfg}.\n")
    if not grade:
        errs.append("math-verify not importable: run the gate with the client python")
    if delta < -max_drop:
        errs.append(f"accuracy drop {delta:.1f} pp > {max_drop}")
    ni = "shown" if lo >= -max_drop else "not shown (CI lower bound below the margin)"
    print(f"Non-inferiority at the {max_drop} pp margin (paired-bootstrap CI lower bound >= −{max_drop}): {ni}. "
          f"The pass rule is the point estimate (goal: drop <= {max_drop} pp).\n")
    check("accuracy", "FAIL" if errs else "PASS",
          "; ".join(errs) if errs else f"Δ {delta:+.1f} pp (limit −{max_drop}), CI [{lo:+.1f}, {hi:+.1f}], "
          f"non-inferiority {ni.split(' ')[0]}")


# ── performance ───────────────────────────────────────────────────────────


def bench(d):
    out = {}
    for f in sorted((d / "bench").glob("in*_out*_c*.json")):
        m = re.match(r"in(\d+)_out(\d+)_c(\d+)", f.stem)
        out[(int(m.group(1)), int(m.group(3)))] = json.loads(f.read_text())
    return out


def pct(a, b):
    return 100.0 * (b - a) / a if a else float("nan")


def perf(B, F, max_regress):
    print("\n## Load matrix (256 output tokens)\n")
    rb, rf = bench(B), bench(F)
    if not rb or not rf:
        return check("perf", "MISSING", "bench/*.json missing")
    print("| input | conc | TTFT P50 ms BF16 / FP8 (Δ%) | TTFT P99 ms | TPOT P50 ms | TPOT P99 ms | out tok/s | ok |")
    print("|---|---|---|---|---|---|---|---|")
    errs = []
    for key in sorted(set(rb) | set(rf)):
        if key not in rb or key not in rf:
            errs.append(f"cell {key} missing in one run")
            continue
        x, y = rb[key], rf[key]
        cell_errs = []
        for lab, j in (("BF16", x), ("FP8", y)):
            if j.get("failed", 0) or j["completed"] != j["num_prompts"]:
                cell_errs.append(f"{lab} {j['completed']}/{j['num_prompts']} completed, {j.get('failed', 0)} failed")
        if x["num_prompts"] != y["num_prompts"] or x["total_input_tokens"] != y["total_input_tokens"]:
            cell_errs.append("different request sets")
        cells, worst = [], 0.0
        for metric in ("median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"):
            dd = pct(x[metric], y[metric])
            if metric.startswith("median"):
                worst = max(worst, dd)
            cells.append(f"{x[metric]:.1f} / {y[metric]:.1f} ({dd:+.1f}%)")
        t = pct(x["output_throughput"], y["output_throughput"])
        cells.append(f"{x['output_throughput']:.1f} / {y['output_throughput']:.1f} ({t:+.1f}%)")
        if worst > max_regress:
            cell_errs.append(f"P50 regression {worst:+.1f}%")
        errs += [f"{key}: {e}" for e in cell_errs]
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {'PASS' if not cell_errs else 'FAIL'} |")
    print()
    check("perf", "FAIL" if errs else "PASS",
          "; ".join(errs) if errs else f"{len(rb)} cells, P50 TTFT/TPOT within +{max_regress}%")


# ── stability ─────────────────────────────────────────────────────────────


def counters(p):
    out = {}
    if not Path(p).exists():
        return None
    for line in Path(p).read_text().splitlines():
        m = re.match(r'^(vllm:[a-z_]+)\{(.*)\}\s+([0-9.e+]+)', line)
        if m:
            reason = re.search(r'finished_reason="(\w+)"', m.group(2))
            k = m.group(1) + (f":{reason.group(1)}" if reason else "")
            out[k] = out.get(k, 0) + float(m.group(3))
    return out


def stability(B, F, min_s, fp8_only=False):
    print("\n## Stability\n")
    errs, lines = [], []
    for lab, d in ((("FP8", F),) if fp8_only else (("BF16", B), ("FP8", F))):
        j = load_json(d / "stability.json")
        if not j:
            errs.append(f"{lab}: stability.json missing")
            continue
        e = load_json(d / "stability_errlines.json")
        if e is None:
            m = re.search(r"server-log error/nan lines \+(\d+)", (d / "acceptance.log").read_text()
                          if (d / "acceptance.log").exists() else "")
            new_err = int(m.group(1)) if m else None
        else:
            new_err = e["error_lines_after"] - e["error_lines_before"]
        c0, c1 = counters(d / "stability_metrics_before.txt"), counters(d / "stability_metrics.txt")
        if c0 is not None and c1:
            dc = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
            note = (f"preemptions +{dc.get('vllm:num_preemptions_total', 0):.0f}, prefix hit rate "
                    f"{dc.get('vllm:prefix_cache_hits_total', 0) / max(dc.get('vllm:prefix_cache_queries_total', 1), 1) * 100:.1f}%")
        else:
            note = "metrics only cumulative since server start (no before snapshot)"
        lines.append(f"- {lab}: {j['completed']}/{j['num_prompts']} ok, {j.get('failed', 0)} failed, "
                     f"{j['duration']:.0f} s, TTFT P50 {j['median_ttft_ms']:.0f} / P99 {j['p99_ttft_ms']:.0f} ms, "
                     f"TPOT P50 {j['median_tpot_ms']:.1f} / P99 {j['p99_tpot_ms']:.1f} ms, "
                     f"{j['output_throughput']:.1f} out tok/s; new server-log error/NaN lines {new_err}; {note}")
        if j["duration"] < min_s:
            errs.append(f"{lab}: {j['duration']:.0f} s < {min_s} s")
        if j.get("failed", 0) or j["completed"] != j["num_prompts"]:
            errs.append(f"{lab}: {j['completed']}/{j['num_prompts']} completed")
        if new_err is None or new_err:
            errs.append(f"{lab}: new server-log error/NaN lines {new_err}")
    print("\n".join(lines) + "\n")
    check("stability", "FAIL" if errs else "PASS", "; ".join(errs) if errs else f"both runs >= {min_s} s, clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("fp8")
    ap.add_argument("--checks", default="provenance,memory,accuracy,perf,stability")
    ap.add_argument("--max-regress", type=float, default=5.0)
    ap.add_argument("--max-acc-drop", type=float, default=1.0)
    ap.add_argument("--min-stab-seconds", type=float, default=1800)
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-peak-saving", type=float, default=0.45)
    ap.add_argument("--strict-transient", action="store_true")
    ap.add_argument("--stability-fp8-only", action="store_true",
                    help="only the FP8 run needs a stability run (e.g. DCP 8 functional pair)")
    a = ap.parse_args()
    B, F = Path(a.bf16), Path(a.fp8)
    print(f"# KV acceptance gate: {B} (BF16 KV) vs {F} (FP8 KV)\n")
    for c in a.checks.split(","):
        {"provenance": lambda: provenance(B, F), "memory": lambda: memory(B, F, a.min_peak_saving, a.strict_transient),
         "accuracy": lambda: accuracy(B, F, a.max_acc_drop), "perf": lambda: perf(B, F, a.max_regress),
         "stability": lambda: stability(B, F, a.min_stab_seconds, a.stability_fp8_only)}[c]()
    ok = all(r["status"] == "PASS" for r in RESULTS)
    print(f"\n**GATE: {'PASS' if ok else 'FAIL'}** "
          + ("" if ok else "(" + ", ".join(f"{r['check']} {r['status']}" for r in RESULTS if r["status"] != "PASS") + ")"))
    if a.json:
        json.dump({"bf16": str(B), "fp8": str(F), "gate": "PASS" if ok else "FAIL", "checks": RESULTS},
                  open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
