#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the DMA-order schedule of the GPT-OSS 120B decode megakernel for one graph.

The compiler binds each constraint of the schedule to a BIR instruction name.
The name ends in the kernel-call scope ``_inst__I-<K>-0``, where ``K`` is the
index of the megakernel call in the outer decode graph. The shipped schedule
carries one scope. This script rewrites it to the ``K`` of the graph that the
server compiles, and writes the copy to the output directory.

``K`` depends on the compile route, because each route builds a different
outer graph. The values below are the ones of the vLLM Neuron release that
ships this example. The compiler skips a constraint whose name it cannot find
and reports no error, so a wrong ``K`` gives a decode graph with no schedule.

Usage::

    python3 prepare_megakernel_schedule.py --out-dir ~/.cache/vllm-neuron/megakernel_schedule
    python3 prepare_megakernel_schedule.py --k 250 --out-dir /some/dir

The script prints the path of the copy on stdout, and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# The kernel-call scope of the disaggregated decode graph, for each compile route.
K_NATIVE = 250  # VLLM_NEURON_BACKEND unset or neuron_native, the default
K_XLA = 261  # VLLM_NEURON_BACKEND=vllm_neuron

# The schedule of the Neuron SDK 2.33 release: 3,299 DMA-order constraints for
# the kernel library that the release ships.
DEFAULT_SOURCE = Path(__file__).with_name("gpt_oss_120b_megakernel_dma_order.json")

_SCOPE_RE = re.compile(r"inst__I-(\d+)-0")


def default_k() -> int:
    """Return the kernel-call scope of the active compile route.

    Example:
        >>> default_k() in (K_NATIVE, K_XLA)
        True
    """
    backend = os.environ.get("VLLM_NEURON_BACKEND", "").strip().lower()
    return K_XLA if backend == "vllm_neuron" else K_NATIVE


def prepare_schedule(k: int, source: Path, out_dir: Path) -> Path:
    """Write a copy of ``source`` whose kernel-call scope is ``k``. Return its path.

    Args:
        k: The kernel-call scope of the decode graph.
        source: The shipped schedule. It carries one scope.
        out_dir: The directory of the copy. The compile-cache key holds the
            absolute path of the copy, so two runs must use the same directory.

    Returns:
        The path of the copy.

    Raises:
        ValueError: If ``source`` carries more or fewer than one scope.

    Example:
        >>> prepare_schedule(250, DEFAULT_SOURCE, Path("/tmp/mk"))  # doctest: +SKIP
        PosixPath('/tmp/mk/gpt_oss_120b_megakernel_dma_order_K250_depsnone.json')
    """
    with open(source, encoding="utf-8") as fh:
        data = json.load(fh)
    ops = data["constraint_ops"]
    scopes = Counter(
        match.group(1) for op in ops for match in _SCOPE_RE.finditer(op["instr_name"])
    )
    if len(scopes) != 1:
        raise ValueError(
            f"{source} holds {len(scopes)} kernel-call scopes ({dict(scopes)}). "
            "The rewrite needs exactly one."
        )
    k_src = next(iter(scopes))
    old, new = f"inst__I-{k_src}-0", f"inst__I-{k}-0"
    for op in ops:
        op["instr_name"] = op["instr_name"].replace(old, new)
        if "dbg" in op:
            op["dbg"] = op["dbg"].replace(old, new)
    deps_mode = data["constraint_deps_mode"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{source.stem}_K{k}_deps{deps_mode}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(
        f"schedule: {len(ops)} ops, scope I-{k_src} -> I-{k}, deps_mode={deps_mode}",
        file=sys.stderr,
    )
    return out


def main() -> int:
    """Parse the arguments, write the schedule, and print its path."""
    parser = argparse.ArgumentParser(
        description="Write the megakernel DMA-order schedule for one decode graph."
    )
    parser.add_argument(
        "--k",
        default="auto",
        help="the kernel-call scope, or 'auto' for the active compile route",
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="the shipped schedule"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="the directory of the copy"
    )
    args = parser.parse_args()
    k = default_k() if args.k == "auto" else int(args.k)
    print(prepare_schedule(k, args.source, args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
