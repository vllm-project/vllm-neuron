#!/usr/bin/env python3
"""What a worker's main thread does while the device is idle (py-spy speedscope samples).

  host_windows.py <hostprof cell dir> [--rank 0] [--top 25]

Each sample is classified by its stack: W = waiting for the next RPC (shm dequeue),
F = blocked reading the step's accepted tokens back (runner `.to("cpu")`), P = blocked reading
the drafter's ids back (`DSparkProposer.propose`), L = submitting a launch (`_lite_execute`), else
host work. The device has nothing to run from the end of a blocking read-back (F/P: the device
just finished) until the next launch has been submitted (L): those samples are the device-idle
windows. Runs of W longer than 50 ms (the request finished, the server idles) are dropped.
Each idle-window sample is charged to the innermost frame of the serving code (vllm_neuron /
deepseek_v41 / vllm) with its caller chain shortened.
"""
import argparse, collections, glob, json, os, re


def label(fr, s):
    names = [fr[i]["name"] for i in s]
    if "dequeue" in names:
        return "W"
    if "propose" in names and any(fr[i]["file"].endswith("drafter.py") and fr[i]["line"] in (171, 173) for i in s):
        return "P"
    if "_lite_execute" in names:
        return "L"
    top = fr[s[-1]]
    if top["name"] == "_execute_model_forward" and top["line"] == 9309:
        return "F"
    return "o"


def where(fr, s, depth=3):
    """innermost serving-code frames"""
    keep = [i for i in s if re.search(r"vllm_neuron|deepseek_v41|/vllm/", fr[i]["file"])]
    if not keep:
        keep = s[-1:]
    out = []
    for i in reversed(keep[-depth:]):
        f = fr[i]
        out.append(f"{f['name']} ({os.path.basename(f['file'])}:{f['line']})")
    return " < ".join(out)


COMPONENTS = [  # (component, any of these function names in the stack), first match wins
    ("finish the step: drafts, sampled tokens, ModelRunnerOutput", ("propose", "_generate_model_runner_output",
                                                                    "_update_batch_state_with_samples", "sample_tokens")),
    ("model: engram n-gram hash on the host (`_engram_hashes`)", ("_engram_hashes",)),
    ("model: spans / slots / step indices (`_sequence_spans`, `_paged_spans`, `make_step`)",
     ("_sequence_spans", "_paged_spans", "make_step", "_mk_spans", "embed_ids")),
    ("runner: sampling params + spec-decode row replication", ("build_sampling_params_tensor", "_maybe_replicate_for_spec_decode")),
    ("runner: states + input / attention / spec-decode metadata", ("_update_states", "_prepare_model_input",
                                                                   "_build_attention_metadata", "_build_spec_decode_metadata")),
    ("drafter input build (`run_draft`) — misattributed step", ("run_draft",)),
]


def component(fr, s):
    names = {fr[i]["name"] for i in s}
    for comp, keys in COMPONENTS:
        if names & set(keys):
            return comp
    return "other host work (model forward / runner glue)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    tot = collections.Counter()
    phase_tot = collections.Counter()
    comp_tot = collections.Counter()
    steps = 0
    seconds = 0.0
    for f in sorted(glob.glob(f"{a.cell}/spy_worker{a.rank}_r*.json")):
        j = json.load(open(f))
        fr = j["shared"]["frames"]
        p = next(q for q in j["profiles"] if "MainThread" in q["name"])
        lab = [label(fr, s) for s in p["samples"]]
        dt = p["weights"][0] if p["weights"] else 0.005
        # drop idle runs (W > 50 ms)
        n = len(lab)
        keep = [True] * n
        i = 0
        while i < n:
            if lab[i] == "W":
                k = i
                while k < n and lab[k] == "W":
                    k += 1
                if (k - i) * dt > 0.05:
                    for x in range(i, k):
                        keep[x] = False
                i = k
            else:
                i += 1
        # device-idle windows: after the last sample of a blocking read-back run until the next L run ends
        state = None
        for i in range(n):
            if not keep[i]:
                state = None
                continue
            c = lab[i]
            seconds += dt
            if c == "P" and (i == 0 or lab[i - 1] != "P"):
                steps += 1              # (one read-back run = one step)
            if c == "P":                 # the drafter (the step's last launch) just finished
                state = "after"
                continue
            if c == "F":
                continue
            if state == "after":
                names = [fr[x]["name"] for x in p["samples"][i]]
                if c == "L" or "run_blocks" in names:
                    # the forward reached its block launches: the first launch's submission follows
                    # (its cost comes from DEEPSEEK_V41_HOST_PROF's per-launch args / entry times)
                    state = None
                    continue
                key = "waiting for the next RPC (engine core)" if c == "W" else "host work"
                phase_tot[key] += 1
                comp_tot["engine core round trip (worker waits for the next RPC)" if c == "W"
                         else component(fr, p["samples"][i])] += 1
                tot[where(fr, p["samples"][i], a.depth) if c != "W" else "(waiting for the next RPC: shm dequeue)"] += 1
    ms = lambda c: 1e3 * c * dt / max(steps, 1)
    print(f"{a.cell} rank {a.rank}: {steps} steps (drafter read-back runs), {seconds:.1f} s of samples, {dt * 1e3:.0f} ms a sample")
    print("device-idle window per step, by state:")
    for k, v in phase_tot.most_common():
        print(f"  {ms(v):6.2f} ms  {k}")
    print(f"  {ms(sum(phase_tot.values())):6.2f} ms  total until the forward starts its block launches")
    print("by component:")
    for k, v in comp_tot.most_common():
        print(f"  {ms(v):6.2f} ms  {k}")
    print("by code location (innermost serving-code frames):")
    for k, v in tot.most_common(a.top):
        print(f"  {ms(v):6.2f} ms  {k}")


if __name__ == "__main__":
    main()
