#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert the forward pass read no weight from disk, using the kernel's counters.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_hbm_resident \\
        [--trace <path.json>] [--load-deadline-pct 40] [--max-read-mib 256]

Reads the trace `io_monitor.py` wrote while the verifier ran, so it needs no
device of its own and costs seconds. The measurement is the monitor's; this is
the rule applied to it. See `io_monitor.py` for why the evidence comes from
`/proc` rather than from the port's own log lines.

**The rule, and the reasoning for each half.**

1. *No descriptor on the checkpoint after the load phase.* The loader opens the
   48 shards, places this rank's slice in HBM, and closes them; load is ~62 s of
   a run that is minutes long. So every sample holding an open checkpoint
   descriptor must fall inside the first `--load-deadline-pct` of the run. A
   forward that streams weights has to open or hold a shard, and it does so
   while generating -- late in the run, where this rule rejects it. The default
   40% is roughly a 4x margin over the real load phase rather than a tight fit,
   because an iteration may legitimately make loading slower.

2. *Almost no block-device reads after the half-way mark.* `read_bytes` is what
   the kernel actually fetched from the device, which catches the case a
   descriptor check misses: a shard left mmap'd and faulted in page by page.
   Streaming this model's weights costs ~13 GiB a rank a forward, so the
   `--max-read-mib 256` default is a ~50x margin -- it is there to tolerate
   logging, the compile cache and Python imports, not to leave room for a
   weight.

**What this does not close**, stated rather than implied: a shard that is mmap'd
*and* already in the page cache would be read by minor faults -- no descriptor,
no block I/O. That costs host DRAM, which the requirement also forbids. Minor
faults are far too noisy to gate on (every allocation faults), so they are
reported, and so is the list of checkpoint files still mapped at the end of the
run, which is what makes such a case visible to a reader. The agent constraint
is what judges intent; this is the part a model cannot talk its way out of.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VERIFIER_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE = _VERIFIER_DIR / "io_trace.json"

#: How late in the run a checkpoint descriptor may still be open, as a
#: percentage of wall clock. See rule 1.
DEFAULT_LOAD_DEADLINE_PCT = 40.0

#: Block-device reads allowed over the second half of the run, in MiB. See rule 2.
DEFAULT_MAX_READ_MIB = 256.0

_MIB = 1024.0 * 1024.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", default=str(DEFAULT_TRACE))
    p.add_argument("--load-deadline-pct", type=float, default=DEFAULT_LOAD_DEADLINE_PCT)
    p.add_argument("--max-read-mib", type=float, default=DEFAULT_MAX_READ_MIB)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.trace)

    if not path.is_file():
        print(f"check_hbm_resident: no trace at {path}.\n"
              "  The run that was supposed to produce it did not, which means this cannot "
              "be established either way — so it fails. The verifier constraint runs under "
              "`io_monitor.py`; if that wrapper was removed from its command, restore it.")
        print("##check_hbm_resident[passed=0]")
        return 1

    trace = json.loads(path.read_text())
    samples = trace.get("samples") or []
    duration = float(trace.get("duration_s") or 0.0)
    watch = trace.get("watch", "(unknown)")
    returncode = trace.get("returncode")

    problems: list[str] = []
    if returncode not in (0, None):
        problems.append(
            f"the monitored command exited {returncode}, so the generation phase this is "
            "supposed to measure may never have happened. Fix that run first."
        )
    if len(samples) < 4:
        problems.append(
            f"only {len(samples)} sample(s) in the trace — too few to tell a load phase "
            "from a generation phase. Either the run was very short or the monitor was "
            "killed early."
        )

    boundary = duration * args.load_deadline_pct / 100.0
    late_opens = [s for s in samples if s.get("open_fds") and s.get("t", 0.0) > boundary]
    if late_opens:
        first = late_opens[0]
        problems.append(
            f"{len(late_opens)} sample(s) hold an open descriptor on {watch} after "
            f"{boundary:.0f}s ({args.load_deadline_pct:g}% of a {duration:.0f}s run). "
            f"First at t={first.get('t', 0.0):.0f}s: {first['open_fds'][:4]}. "
            "A weight was read after loading finished."
        )

    # The second half, so the comparison is not distorted by the startup ramp
    # where the worker processes do not all exist yet.
    midpoint = duration / 2.0
    tail = [s for s in samples if s.get("t", 0.0) >= midpoint]
    read_mib = rchar_mib = None
    majflt = minflt = None
    if len(tail) >= 2:
        # Max rather than last: the sum is over a process set that shrinks as
        # ranks exit at the end of a run, and a drop is not a negative read.
        read_mib = (max(s.get("read_bytes", 0) for s in tail)
                    - tail[0].get("read_bytes", 0)) / _MIB
        rchar_mib = (max(s.get("rchar", 0) for s in tail)
                     - tail[0].get("rchar", 0)) / _MIB
        majflt = max(s.get("majflt", 0) for s in tail) - tail[0].get("majflt", 0)
        minflt = max(s.get("minflt", 0) for s in tail) - tail[0].get("minflt", 0)
        if read_mib > args.max_read_mib:
            problems.append(
                f"{read_mib:.0f} MiB read from the block device over the second half of the "
                f"run, above the {args.max_read_mib:g} MiB bar. Streaming this model's "
                "weights is ~13 GiB a rank a forward, so check whether this is a weight "
                "being faulted in from an mmap'd shard."
            )

    mapped = samples[-1].get("mapped") if samples else []

    print(f"check_hbm_resident: {len(samples)} sample(s) over {duration:.0f}s, "
          f"watching {watch}")
    print(f"  load phase: last checkpoint descriptor closed by "
          f"{max((s.get('t', 0.0) for s in samples if s.get('open_fds')), default=0.0):.0f}s "
          f"(deadline {boundary:.0f}s)")
    if read_mib is not None:
        print(f"  second half: {read_mib:.1f} MiB from the block device "
              f"(bar {args.max_read_mib:g}), {rchar_mib:.1f} MiB via read syscalls, "
              f"{majflt} major / {minflt} minor faults")
        print(f"##autohelix[forward_read_mib={read_mib:.2f}]")
    if mapped:
        print(f"  still mapped at the end: {len(mapped)} checkpoint file(s) — "
              f"{mapped[:3]}{' ...' if len(mapped) > 3 else ''}")
        print("    A mapping is address space, not necessarily a read; it is listed because "
              "it is the one way a weight could reach host DRAM without showing above.")
    else:
        print("  still mapped at the end: none")

    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print("check_hbm_resident: the forward read no weight from disk")
    print(f"##check_hbm_resident[passed={0 if problems else 1}]")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
