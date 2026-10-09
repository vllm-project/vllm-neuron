"""Local stand-in for the invariant suite: the checks its findings describe.

Not part of the simulator. Run it to see whether the cost models behave the way the ten
invariants ask before spending an iteration finding out.
"""
from __future__ import annotations

import ast
import copy
import sys
from collections import defaultdict
from pathlib import Path

from floorplan.parser import Hardware, load_system
from floorplan.schema import Floorplan
from floorplan.sim import api
from floorplan.sim.api import Workload
from floorplan.sim.runner import (
    HEAVIEST,
    WORKLOADS,
    deployable_modules,
    load_cost_models,
    load_graph,
    load_model_config,
    simulate,
    simulate_workload,
    topological_order,
)

PROJECT = Path("/home/ubuntu/workspace/floorplan-deepseek")
ARTIFACT = Path("/home/ubuntu/workspace/partition-artifact")
SYSTEMS = PROJECT / "systems"

#: The helper names a duration may come from. Anything else handed to `ctx.op(seconds=...)` is a
#: fabricated duration, which is invariant (i).
COST_HELPERS = {
    "matmul_seconds", "elementwise_seconds", "dma_seconds", "gather_seconds",
}


def setup():
    plan = Floorplan.load(PROJECT / "floorplan.yaml")
    system = load_system(plan.target, SYSTEMS)
    hardware = Hardware.from_system(system)
    modules, _ = load_graph(ARTIFACT)
    config = load_model_config(ARTIFACT)
    load_cost_models(PROJECT)
    order = topological_order(modules)
    return plan, hardware, modules, config, order


def work(result) -> float:
    return sum(s.op.seconds for s in result.trace.scheduled)


def check_no_fabricated_durations() -> list[str]:
    """(i) Every `seconds` handed to the framework is a `ctx.*_seconds` call, not arithmetic.

    Walks the AST of every file under sim/modules/ looking for a call to `.op(` and checking what
    reached its `seconds` slot -- third positional, or the keyword. A literal, a product or a sum
    of two helper results would all be a duration the cost model invented rather than one the
    system YAML's rates produced.
    """
    problems: list[str] = []
    for path in sorted((PROJECT / "sim" / "modules").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "op"):
                continue
            seconds = None
            if len(node.args) >= 3:
                seconds = node.args[2]
            for keyword in node.keywords:
                if keyword.arg == "seconds":
                    seconds = keyword.value
            if seconds is None:
                continue
            ok = (
                isinstance(seconds, ast.Call)
                and isinstance(seconds.func, ast.Attribute)
                and seconds.func.attr in COST_HELPERS
            )
            if not ok:
                problems.append(
                    f"{path.name}:{seconds.lineno}: seconds is "
                    f"{ast.dump(seconds)[:60]}, not a ctx.*_seconds call"
                )
    return problems


def main() -> int:
    plan, hardware, modules, config, order = setup()
    failures = []

    # (a) every module has a model
    missing = []
    for module_id in sorted(deployable_modules(modules)):
        try:
            api.resolve(module_id, modules[module_id])
        except Exception as exc:
            missing.append(f"{module_id}: {exc}")
    print(f"[a] coverage: {len(missing)} unclaimed")
    if missing:
        failures.append("coverage")

    # (i) no fabricated durations
    fabricated = check_no_fabricated_durations()
    print(f"[i] fabricated durations: {len(fabricated)}")
    for problem in fabricated[:5]:
        print(f"    {problem}")
    if fabricated:
        failures.append("fabricated durations")

    # (h) every numbered constraint cited
    constraints_src = (PROJECT / "sim" / "constraints.py").read_text()
    modules_src = "\n".join(
        p.read_text() for p in sorted((PROJECT / "sim" / "modules").glob("*.py"))
    )
    uncited = [
        n for n in range(1, 10)
        if f"item {n}" not in constraints_src
        and f"constraint {n}" not in constraints_src + modules_src
    ]
    print(f"[h] constraint citations: {len(uncited)} uncited" + (f" {uncited}" if uncited else ""))
    if uncited:
        failures.append("uncited constraints")

    # (b) the whole grid simulates and publishes a positive latency
    results, _hw = simulate(PROJECT / "floorplan.yaml", ARTIFACT, PROJECT, SYSTEMS)
    nonpositive = sorted(n for n, r in results.items() if not r.milliseconds > 0)
    print(f"[b] grid: {len(results)} workloads, {len(nonpositive)} non-positive")
    if nonpositive:
        failures.append("non-positive metric")

    print("[b] published metrics:")
    for name in (w.name for w in WORKLOADS):
        print(f"    {name:<18}{results[name].milliseconds:11.3f} ms"
              f"   work {work(results[name]) * 1e3:11.1f} ms")

    # (d) no module is free
    print("[d] no module is free:")
    heavy = results[HEAVIEST]
    light = results["decode_128_b1"]
    for label, result in (("prefill heavy", heavy), ("decode_128_b1", light)):
        by_module = result.trace.by_module()
        free = sorted(m for m in deployable_modules(modules) if by_module.get(m, 0.0) <= 0.0)
        print(f"    {label}: {len(free)} free" + (f" -> {free[:6]}" if free else ""))
        if free:
            failures.append(f"free modules ({label})")

    # (f) longer contexts cost more, superlinearly in prefill -- reported four ways, because the
    # four disagree and which one the suite measures decides whether this passes. See sim/NOTES.md.
    def ratio(a, b):
        return f"x{a / b:.2f}" if b else "-"

    print("[f] context scaling on the published grid (chunked, as the metric is):")
    for batch in (1, 32):
        short, long = results[f"prefill_128_b{batch}"], results[f"prefill_8192_b{batch}"]
        print(f"    prefill b{batch:<3} 128 -> 8192 tokens (x64):"
              f" makespan {ratio(long.seconds, short.seconds)}"
              f"  work {ratio(work(long), work(short))}")
        if long.seconds <= short.seconds:
            failures.append(f"prefill context monotonicity b{batch}")
    for batch in (1, 32):
        short, long = results[f"decode_128_b{batch}"], results[f"decode_8192_b{batch}"]
        print(f"    decode  b{batch:<3} 128 -> 8192 context:"
              f" makespan {ratio(long.seconds, short.seconds)}"
              f"  work {ratio(work(long), work(short))}")
        if long.seconds <= short.seconds:
            failures.append(f"decode context monotonicity b{batch}")

    print("[f] context scaling, one chunk, new_tokens = context:")
    unchunked = copy.deepcopy(plan)
    unchunked.runtime.pipeline_chunks = False
    prev = first = None
    for context in (128, 512, 2048, 8192):
        w = Workload(name=f"p{context}", phase="prefill", context_tokens=context,
                     new_tokens=context, batch=1)
        r = simulate_workload(unchunked, hardware, modules, config, w, order)
        step = "" if prev is None else f"  {ratio(r.seconds, prev[1])} for x4 tokens"
        print(f"    ctx {context:>5}: {r.milliseconds:9.3f} ms  work {work(r) * 1e3:10.2f} ms{step}")
        if prev is None:
            first = (r.seconds, work(r))
        prev = (context, r.seconds, work(r))
    print(f"    128 -> 8192 (x64 tokens): makespan {ratio(prev[1], first[0])}"
          f"  work {ratio(prev[2], first[1])}")

    # (g) batch scaling
    print("[g] batch scaling on the published grid:")
    for phase, context in (("prefill", 128), ("prefill", 8192), ("decode", 128), ("decode", 8192)):
        base = results[f"{phase}_{context}_b1"]
        cells = "  ".join(
            f"b{b}: {ratio(results[f'{phase}_{context}_b{b}'].seconds, base.seconds)}"
            for b in (4, 8, 32)
        )
        print(f"    {phase}_{context:<5} {cells}")
        for b in (4, 8, 32):
            if results[f"{phase}_{context}_b{b}"].seconds <= base.seconds:
                failures.append(f"batch monotonicity {phase}_{context}_b{b}")

    print("[g] batch scaling (prefill, single chunk, ctx 2048):")
    base = None
    for batch in (1, 4, 8, 32):
        w = Workload(name=f"b{batch}", phase="prefill", context_tokens=2048,
                     new_tokens=2048, batch=batch)
        r = simulate_workload(unchunked, hardware, modules, config, w, order)
        rel = "" if base is None else f"  {ratio(r.seconds, base)} for x{batch}"
        print(f"    batch {batch:>3}: {r.milliseconds:9.3f} ms  work {work(r) * 1e3:10.2f} ms{rel}")
        base = base or r.seconds

    # (e) doubling a split halves per-shard compute and adds communication
    def relayout(module_id, dim, collective, units, workload=None):
        trial = copy.deepcopy(plan)
        for placement in trial.placements:
            if placement.module != module_id:
                continue
            kind = type(placement.units[0])
            placement.units = [kind.parse(u) for u in units]
            placement.splits = [type(placement.splits[0])(
                dim=dim, factor=len(units), collective=collective)]
        w = workload or Workload(name="probe", phase="prefill", context_tokens=2048,
                                 new_tokens=2048, batch=1)
        r = simulate_workload(trial, hardware, modules, config, w, order)
        compute, comms = defaultdict(float), 0.0
        for s in r.trace.scheduled:
            if s.op.module != module_id:
                continue
            if s.op.engine == "cc":
                comms += s.op.seconds
            else:
                compute[s.op.unit] += s.op.seconds
        return (max(compute.values()) if compute else 0.0), comms

    cases = (
        ("layers.4.attention", "head", "allreduce"),
        ("layers.4.attention", "seq", "allgather"),
        ("layers.4.ffn", "expert", "all_to_all"),
        ("layers.4.ffn", "hidden", "allreduce"),
        ("layers.4.hc_attn_in", "hidden", "allreduce"),
        ("layers.1.engram", "ngram", "allgather"),
        ("lm_head", "vocab", "allgather"),
        ("embed", "vocab", "allreduce"),
    )
    print("[e] doubling a split, link class held constant (2 -> 4 cores of one device):")
    for module_id, dim, collective in cases:
        two = relayout(module_id, dim, collective, ["d0.l0", "d0.l1"])
        four = relayout(module_id, dim, collective, ["d0.l0", "d0.l1", "d0.l2", "d0.l3"])
        halves = four[0] <= two[0] * 0.62
        grows = four[1] > two[1]
        print(f"    {module_id:<22}{dim:<8} f=2: shard {two[0] * 1e3:8.3f} cc {two[1] * 1e3:7.4f}"
              f"   f=4: shard {four[0] * 1e3:8.3f} cc {four[1] * 1e3:7.4f}"
              f"   {four[0] / two[0] if two[0] else 0:.3f}x halves={halves} cc_grows={grows}")
        if not halves or not grows:
            failures.append(f"split scaling {module_id}/{dim}")

    print("[e] the same 4-way group on one device vs scattered over the torus:")
    for module_id, dim, collective in cases:
        near = relayout(module_id, dim, collective, ["d0.l0", "d0.l1", "d0.l2", "d0.l3"])
        far = relayout(module_id, dim, collective, ["d0.l0", "d1.l0", "d10.l0", "d5.l0"])
        print(f"    {module_id:<22}{dim:<8} one device cc {near[1] * 1e3:7.4f}   "
              f"scattered cc {far[1] * 1e3:7.4f}   torus_costs_more={far[1] > near[1]}")
        # Not a failure of the cost model when it comes out False: systems/probed.yaml gives
        # links.intra_device 803 GB/s ("derived, not measured": device HBM bandwidth x measured
        # dma_large_contiguous) against the inter_device datasheet 1024 GB/s, so for a
        # bandwidth-bound collective the frozen hardware model says off-device is faster and only
        # hop latency penalizes distance. Reported, not asserted.

    # (c) determinism, (j) hardware untouched
    print("[c] determinism + [j] hardware immutability:")
    before = hardware.describe(), tuple(sorted(hardware.efficiency.items())), tuple(
        sorted(hardware.compute.items()))
    first_run, _ = simulate(PROJECT / "floorplan.yaml", ARTIFACT, PROJECT, SYSTEMS)
    second_run, _ = simulate(PROJECT / "floorplan.yaml", ARTIFACT, PROJECT, SYSTEMS)
    same = all(first_run[n].seconds == second_run[n].seconds for n in first_run)
    after = hardware.describe(), tuple(sorted(hardware.efficiency.items())), tuple(
        sorted(hardware.compute.items()))
    print(f"    identical={same}  hardware_unchanged={before == after}")
    if not same:
        failures.append("determinism")
    if before != after:
        failures.append("hardware mutated")

    print()
    print("FAILURES:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
