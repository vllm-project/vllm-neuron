#!/usr/bin/env python3
"""report_parts/configs.json: which run / steps / client time each report section uses."""
import json, os
import pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt10"


def client_ms(cell_dir):
    w = json.load(open(f"{ROOT}/profiles/{cell_dir}/windows.json"))["windows"]["prefill"]["requests"]
    return 1000 * (max(r["t_done"] for r in w) - min(r["t_send"] for r in w)), [r["t_done"] - r["t_send"] for r in w]


def steady(run):
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    s = sorted(ex.step.unique())
    return [int(x) for x in s[1:-1]]


def dev_steps(run):
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    g = ex.groupby("step").apply(lambda d: d.n_inst.notna().all() and not d.partial.any())
    return [int(s) for s, ok in g.items() if ok]


def main():
    cfgs = []
    for isl, ctx in ((1024, 1280), (8192, 8448)):
        for bs in (1, 10):
            cell = f"in{isl}_bs{bs}"
            ms, lat = client_ms(f"{cell}_r0")
            pre = f"{cell}_prefill"
            ex = pd.read_parquet(f"{ROOT}/analysis/{pre}/execs.parquet")
            steps = sorted(int(s) for s in ex.step.unique() if s >= 0)
            roof = [(pre, dev_steps(pre) or [0])]
            if isl == 8192:
                # the 3rd chunk (positions 4096-6143): the last one whose device trace is complete (the device buffer
                # ends inside the 4th; the delay_iterations=4 capture stops before its last blocks: async scheduling
                # returns the iteration to the profiler while they still run)
                roof = [(pre, [2])]
            if isl == 1024 and bs == 1:
                note = f"One prefill iteration of 1024 rows (TTFT {lat[0]:.3f} s measured by the client in this run)."
            elif isl == 1024:
                note = (f"The scheduler prefills one prompt per iteration ({len(steps)} iterations of 1024 rows), so the "
                        f"{bs} requests get their first token {', '.join(f'{x:.2f}' for x in sorted(lat)[:3])}, … "
                        f"{max(lat):.2f} s after arrival (mean TTFT {sum(lat) / len(lat):.2f} s). Totals are for the whole "
                        f"batch; device metrics are from the iterations whose device trace is complete.")
            elif bs == 1:
                note = (f"Chunked prefill: {len(steps)} iterations of 2048 rows (TTFT {lat[0]:.3f} s). Latency columns sum "
                        f"the {len(steps)} chunks; roofline columns are the 3rd chunk (positions 4096–6143).")
            else:
                note = (f"{len(steps)} iterations of one 2048-row chunk each (one prompt's four chunks, then the next), so "
                        f"the requests get their first token {', '.join(f'{x:.2f}' for x in sorted(lat)[:3])}, … "
                        f"{max(lat):.2f} s after arrival (mean TTFT {sum(lat) / len(lat):.2f} s). Roofline columns are the "
                        f"first prompt's 3rd chunk (positions 4096–6143).")
            cfgs.append(dict(id=f"{cell}_prefill", short=f"in{isl} bs{bs} prefill", phase="prefill", lat_run=pre,
                             steps=steps, roof=roof, client_ms=ms,
                             title=f"in{isl}_bs{bs} · prefill — {bs} request{'s' if bs > 1 else ''} × {isl} tokens "
                                   f"(`max_num_seqs={bs}`, context {ctx})", note=note,
                             hot=["r2", "r1_idx"]))
            dec = f"{cell}_decode"
            if not os.path.exists(f"{ROOT}/analysis/{dec}/units.parquet"):
                print("missing", dec)
                continue
            st = steady(dec)
            rows = 6 * bs
            cfgs.append(dict(id=f"{cell}_decode", short=f"in{isl} bs{bs} decode", phase="decode", lat_run=dec,
                             steps=st, roof=[(dec, [s for s in st if s in dev_steps(dec)])],
                             title=(f"in{isl}_bs{bs} · decode — {bs} sequence{'s' if bs > 1 else ''} ({rows} rows a step: "
                                    f"1 + 5 drafted tokens each), context ≈ {isl}–{ctx}"),
                             note=f"Per decode step (mean of {len(st)} steady steps).",
                             hot=["r2", "draft"]))
    os.makedirs(f"{ROOT}/report_parts", exist_ok=True)
    json.dump(cfgs, open(f"{ROOT}/report_parts/configs.json", "w"), indent=1)
    print(json.dumps([(c["id"], c["lat_run"], c["steps"][:3], len(c["steps"]), c["roof"]) for c in cfgs], indent=0))


if __name__ == "__main__":
    main()
