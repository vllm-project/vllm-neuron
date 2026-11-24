#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a command and record, from the kernel, what it read from the checkpoint.

    python -m vllm_neuron.model.deepseek_v41.verifier.io_monitor \\
        --watch /mnt/nvme/hf --trace <path.json> -- <command> [args...]

Exits with the command's own exit code, so it is transparent to wrap around:
whatever it wraps still passes or fails on its own terms, and the trace is a
side effect. `check_hbm_resident.py` is what reads the trace.

**Why this exists.** The claim is that the forward pass reads no weight from
disk or host DRAM -- only from HBM. The port reports that itself: the loader
logs `Weights are RESIDENT: the forward path reads no file.` and the per-forward
counters say `0 admitted, ~0.00 GiB fetched`. Both lines are produced by
`weight_loader.py` and `model.py`, which are inside `scope.editable`, so a loop
can satisfy a gate that reads them by editing the thing that prints them. This
takes the same measurement from outside the process, where the model cannot
reach it:

    /proc/<pid>/fd          open descriptors, resolved to paths
    /proc/<pid>/maps        file-backed mappings (so a closed-but-mmap'd shard shows)
    /proc/<pid>/io          rchar (bytes via read syscalls) and read_bytes (block device)
    /proc/<pid>/stat        minflt / majflt

Sampled across the whole process tree, because a 64-rank run is 64 worker
processes and the parent reads nothing.

**What it proves and what it does not.** An open descriptor on a shard, a
growing `read_bytes`, or a major fault during generation each mean a weight came
from outside HBM, and none of those can be faked from inside the model. The gap
is the opposite case: a shard left mmap'd whose pages are already in the page
cache would be read by *minor* faults, which cost host DRAM but no I/O. `minflt`
is recorded for that reason and is noisy -- every allocation faults -- so
`check_hbm_resident.py` reports it rather than gating it, and the mapping list
is what makes the case visible at all. Say so rather than overclaiming: this
closes the disk path completely and narrows the DRAM path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

#: Seconds between samples. The forward is ~0.5 s and weight load is ~60 s, so
#: 2 s resolves the phases without the sampler itself costing anything
#: measurable -- one pass is a few hundred `/proc` reads.
DEFAULT_INTERVAL = 2.0

_VERIFIER_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE = _VERIFIER_DIR / "io_trace.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", required=True,
                   help="directory whose files count as weights (the checkpoint)")
    p.add_argument("--trace", default=str(DEFAULT_TRACE))
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="the command to run, after `--`")
    return p.parse_args(argv)


def _sample(root_pid: int, watch: str) -> dict:
    """One observation of the whole process tree.

    Every `/proc` read is wrapped: a rank that exits between the tree walk and
    the read is normal, not an error, and a sampler that dies on it would lose
    the rest of the run.
    """
    import psutil

    open_fds: list[str] = []
    mapped: set[str] = set()
    rchar = read_bytes = minflt = majflt = 0
    n_procs = 0

    try:
        root = psutil.Process(root_pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return {"procs": 0, "open_fds": [], "mapped": [], "rchar": 0,
                "read_bytes": 0, "minflt": 0, "majflt": 0}

    for proc in procs:
        pid = proc.pid
        n_procs += 1
        try:
            with open(f"/proc/{pid}/io") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    if key == "rchar":
                        rchar += int(value)
                    elif key == "read_bytes":
                        read_bytes += int(value)
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[-1].split()
            # After the comm field: state is [0]; minflt [7], majflt [9] in the
            # 1-indexed proc(5) numbering (fields 10 and 12).
            minflt += int(fields[7])
            majflt += int(fields[9])
        except (OSError, ValueError, IndexError):
            pass

        fd_dir = f"/proc/{pid}/fd"
        try:
            for name in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{name}")
                except OSError:
                    continue
                if target.startswith(watch):
                    pos = None
                    try:
                        with open(f"/proc/{pid}/fdinfo/{name}") as fh:
                            for line in fh:
                                if line.startswith("pos:"):
                                    pos = int(line.split()[1])
                                    break
                    except (OSError, ValueError):
                        pass
                    open_fds.append(f"{pid}:{Path(target).name}@{pos}")
        except OSError:
            pass

        try:
            with open(f"/proc/{pid}/maps") as fh:
                for line in fh:
                    path = line.rstrip("\n").partition(" ")[2]
                    idx = path.find(watch)
                    if idx != -1:
                        mapped.add(path[idx:])
        except OSError:
            pass

    return {"procs": n_procs, "open_fds": open_fds, "mapped": sorted(mapped),
            "rchar": rchar, "read_bytes": read_bytes,
            "minflt": minflt, "majflt": majflt}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    command = [a for a in args.command if a != "--"]
    if not command:
        print("io_monitor: nothing to run — put the command after `--`", file=sys.stderr)
        return 2

    watch = str(Path(args.watch).resolve())
    trace_path = Path(args.trace)
    print(f"io_monitor: watching {watch} across the process tree, "
          f"sampling every {args.interval:g}s -> {trace_path}", flush=True)

    trace = {
        "watch": watch,
        "command": command,
        "interval_s": args.interval,
        "started_at": time.time(),
        "samples": [],
    }

    def write() -> None:
        # Rewritten on every sample, so a run that is killed -- or one whose
        # command hangs and is timed out by the gate -- still leaves evidence.
        trace_path.write_text(json.dumps(trace) + "\n")

    write()
    # Line-buffered passthrough: the wrapped command's output is the run's log,
    # and interleaving the sampler's own lines into it would corrupt a log the
    # agent constraints read.
    proc = subprocess.Popen(command)
    t0 = time.monotonic()
    try:
        while True:
            try:
                proc.wait(timeout=args.interval)
                break
            except subprocess.TimeoutExpired:
                pass
            sample = _sample(proc.pid, watch)
            sample["t"] = time.monotonic() - t0
            trace["samples"].append(sample)
            write()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    finally:
        trace["duration_s"] = time.monotonic() - t0
        trace["returncode"] = proc.returncode
        trace["ended_at"] = time.time()
        write()

    print(f"io_monitor: {len(trace['samples'])} sample(s) over "
          f"{trace['duration_s']:.0f}s, command exited {proc.returncode} "
          f"(trace: {trace_path})", flush=True)
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
