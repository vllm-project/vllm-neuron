# SPDX-License-Identifier: Apache-2.0
"""Launch `N` ranks of a unit test on cores 0..N-1, the way the plugin's executor does.

    python kharness_mp.py <world_size> <script.py> [args...]

Each rank is its own process with NEURON_RT_VISIBLE_CORES pinned, the same
NEURON_RT_ROOT_COMM_ID, and a Gloo world group (the plugin's own backend:
collectives are traced into the NEFF, Gloo only bootstraps). The script sees
RANK / WORLD_SIZE in its environment and must call `init_rank()` from this
module before importing `kharness`.
"""

import os
import subprocess
import sys
import time

PORT = int(os.environ.get("KH_MASTER_PORT", "62340"))
COMM = os.environ.get("KH_ROOT_COMM", "127.0.0.1:62341")
#: Per-rank logs go here (`KH_LOG_DIR`), so two launches on different core ranges
#: (`KH_CORE_BASE`, with their own ports) can run side by side.
LOG_DIR = os.environ.get("KH_LOG_DIR", "/tmp/claude-1000")


def init_rank():
    """Called by the per-rank script first thing: pins the core, inits Gloo."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(int(os.environ.get("KH_CORE_BASE", "0")) + rank)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = COMM
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("VLLM_TARGET_DEVICE", "neuron")
    os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "1800")
    os.environ.setdefault("NKI_DISABLE_TRACE_CACHE", "1")
    os.environ.setdefault("NEURON_LIBTORCH_DISABLE_COMPILE_CACHE", "1")
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=world
    )
    return rank, world


def main():
    world = int(sys.argv[1])
    script = sys.argv[2]
    extra = sys.argv[3:]
    procs = []
    logs = []
    for rank in range(world):
        env = dict(os.environ)
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world)
        log = open(f"{LOG_DIR}/kh_rank{rank}.log", "w")
        logs.append(log)
        procs.append(subprocess.Popen(
            [sys.executable, script, *extra], env=env, stdout=log, stderr=subprocess.STDOUT
        ))
    t0 = time.time()
    rc = [p.wait() for p in procs]
    for log in logs:
        log.close()
    print(f"ranks exited {rc} after {time.time() - t0:.0f}s")
    for rank in range(world):
        print(f"===== rank {rank} tail")
        with open(f"{LOG_DIR}/kh_rank{rank}.log") as fh:
            lines = [l for l in fh.read().splitlines()
                     if not any(k in l for k in ("INFO", "WARNING", "UserWarning", "warn_once"))]
        print("\n".join(lines[-25:]))
    sys.exit(max(rc))


if __name__ == "__main__":
    main()
