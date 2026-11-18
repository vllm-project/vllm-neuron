#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-node launcher for the Granite (ibm/PowerLM-3b) device module tests.

The tests need 4 TP ranks spread over 2 Trn2 nodes, so every collective crosses
the node boundary. This launcher sets the rendezvous variables that
``test_device_modules.py`` and ``MPExecutor`` read, then runs pytest on this
node. Run it on both nodes.

The in-tree version of this launcher discovered the node topology from the
MPIJob hostnames through a shared CI helper. That helper is outside the model
folder, so this copy takes the topology from the environment instead:

    # on node 0
    NODE_RANK=0 MASTER_ADDR=<node 0 ip> python3 \
        contrib/models/granite/test/module/launch_multinode.py
    # on node 1
    NODE_RANK=1 MASTER_ADDR=<node 0 ip> python3 \
        contrib/models/granite/test/module/launch_multinode.py

Both nodes must run the same stages in the same order, because each stage waits
for the other node. A stage that fails does not stop the next stage. The
launcher exits nonzero if one stage failed.

Pass your own pytest arguments with ``-m pytest``::

    NODE_RANK=0 MASTER_ADDR=<node 0 ip> python3 \
        contrib/models/granite/test/module/launch_multinode.py \
        -m pytest -v contrib/models/granite/test/module/test_device_modules.py
"""

import os
import subprocess
import sys

NNODES = 2
CCOM_PORT = int(os.environ.get("MASTER_PORT", "29500")) + 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# (test file, pytest --timeout cap in seconds). The test markers set the
# timeout of each test. The cap stops a test that has no marker.
DEFAULT_STAGES = (("test_device_modules.py", 3600),)


def _require_topology() -> None:
    """Set the rendezvous variables from NODE_RANK and MASTER_ADDR."""
    missing = [v for v in ("NODE_RANK", "MASTER_ADDR") if not os.environ.get(v)]
    if missing:
        print(
            f"Set {' and '.join(missing)} before you start the launcher. "
            f"NODE_RANK is 0 on the first node and 1 on the second. "
            f"MASTER_ADDR is the address of node 0.",
            file=sys.stderr,
        )
        sys.exit(2)
    node_rank = int(os.environ["NODE_RANK"])
    if not 0 <= node_rank < NNODES:
        print(f"NODE_RANK must be 0 or 1, not {node_rank}.", file=sys.stderr)
        sys.exit(2)
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault(
        "NEURON_RT_ROOT_COMM_ID", f"{os.environ['MASTER_ADDR']}:{CCOM_PORT}"
    )


def _run(cmd: list[str]) -> int:
    print(f"[launcher] node {os.environ['NODE_RANK']}: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    _require_topology()
    print(
        f"[launcher] NODE_RANK={os.environ['NODE_RANK']}, "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
        f"NEURON_RT_ROOT_COMM_ID={os.environ['NEURON_RT_ROOT_COMM_ID']}",
        flush=True,
    )

    args = sys.argv[1:]
    if args:
        if args[0] == "-m" and len(args) > 1:
            sys.exit(_run([sys.executable, "-m", args[1]] + args[2:]))
        print(f"Usage: {sys.argv[0]} [-m pytest <pytest_args...>]", file=sys.stderr)
        sys.exit(2)

    failed = []
    for test_file, timeout_s in DEFAULT_STAGES:
        rc = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"--timeout={timeout_s}",
                "-v",
                os.path.join(TEST_DIR, test_file),
                "--tb=short",
                "--no-cov",
            ]
        )
        if rc != 0:
            failed.append(f"{test_file} (exit code {rc})")
    if failed:
        print(f"[launcher] Failed stages: {', '.join(failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
