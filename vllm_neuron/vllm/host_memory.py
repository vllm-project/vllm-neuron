# SPDX-License-Identifier: Apache-2.0
"""Say what a multi-rank run will ask of *host* memory, before it asks.

Every memory guard in this backend is about device HBM. On 2026-10-12 a 4-rank run of a
model that streams its weights through host memory died the other way: committed host
memory went from 1.8 GB to 132.3 GB on a 131 GB box with no swap, in under ten minutes.
The kernel could not reclaim it (`pgsteal/s` 0, `%vmeff` 0) and the OOM killer never
fired, so instead of a failed allocation the *host* wedged -- journald's last line is
11:50:01 -- and the instance needed a reboot. Device HBM was never the problem: the ranks
were at 7.06 GiB each of 24, at layer 39 of 40.

Two things stand between that and a repeat, and this module is the lesser one:

  * `vm.overcommit_memory=2` with `overcommit_ratio` set so `CommitLimit` is *below*
    physical RAM. Then the kernel refuses a commitment it cannot back, the allocating
    rank dies with ENOMEM, and the host lives. That is the real guard, it lives in
    `/etc/sysctl.d/99-neuron-host-memory.conf`, and this module checks it is in force.
  * This preflight, which projects the run's host demand and logs it. A projection is
    only ever an estimate -- read the warning in `_project` before trusting one.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Host commitment a rank of the timeshare port holds, measured two ways:
#:
#:   TP=1, whole run    13-14 GB of `Committed_AS` (four samples, against 1.7 GB idle)
#:   TP=4, per worker   24.5 GB of *resident* anonymous heap (`VmData == VmRSS`), with
#:                      stacks at 140 KB and 97 threads — so it is neither stacks nor
#:                      glibc arenas but NEFF/kbin host images and DMA descriptors,
#:                      which every rank loads in full because the NEFF is not sharded
#:
#: The larger figure is the one to budget with, since it is what a multi-rank run
#: actually holds. Override with `NEURON_HOST_MEM_PER_RANK_GIB` for another model.
DEFAULT_PER_RANK_GIB = 24.5

#: Host memory a run should not eat into, so that ssh, journald and the agent survive a
#: run that misbehaves. A second margin on top of the real one: `CommitLimit` is itself
#: set below physical RAM, so there is already RAM that no commitment can reach.
RESERVE_GIB = 6.0


def _meminfo() -> dict[str, float]:
    """`/proc/meminfo` in GiB, for the handful of fields that matter."""
    want = {"MemTotal", "MemAvailable", "CommitLimit", "Committed_AS", "SwapTotal"}
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in want:
                    out[key] = float(rest.split()[0]) / (1 << 20)
    except OSError:  # pragma: no cover - /proc always exists on Linux
        pass
    return out


def _project(world_size: int, per_rank_gib: float) -> float:
    """Projected host commitment for `world_size` ranks.

    **Linear in the rank count, which the one measurement we have says is optimistic.**
    At TP=1 this port commits ~14 GB; the 4-rank run committed 132 GB, which is 33 GB a
    rank -- 2.4x linear. Each rank is its own process with its own gather buffers, its own
    runtime host allocations and its own compiler, and the mmap'd shards (`MAP_SHARED`,
    read-only) are the *only* part that is shared. So treat a passing projection as "not
    obviously impossible" rather than "fits".
    """
    return world_size * per_rank_gib


def check_host_memory_budget(world_size: int, model_name: str = "") -> None:
    """Log the host-memory budget for a run of `world_size` ranks; raise if it cannot fit.

    Raises `RuntimeError` when even the optimistic projection exceeds what the kernel
    would grant, because the alternative is discovering it ten minutes into a weight load.
    """
    info = _meminfo()
    if not info:
        return

    per_rank = float(os.environ.get("NEURON_HOST_MEM_PER_RANK_GIB", DEFAULT_PER_RANK_GIB))
    projected = _project(world_size, per_rank)
    grantable = info.get("CommitLimit", 0.0) - info.get("Committed_AS", 0.0)
    strict = _overcommit_mode() == 2

    logger.info(
        "Host memory budget%s: %d rank(s) x %.1f GiB = %.1f GiB projected; "
        "%.1f GiB grantable (CommitLimit %.1f - Committed_AS %.1f), "
        "MemAvailable %.1f GiB, swap %.1f GiB, overcommit_memory=%s",
        f" for {model_name}" if model_name else "",
        world_size,
        per_rank,
        projected,
        grantable,
        info.get("CommitLimit", 0.0),
        info.get("Committed_AS", 0.0),
        info.get("MemAvailable", 0.0),
        info.get("SwapTotal", 0.0),
        _overcommit_mode(),
    )

    if world_size > 1 and not strict:
        logger.warning(
            "vm.overcommit_memory=%s, so the kernel will promise host memory it cannot "
            "back and a multi-rank run that asks for too much can wedge the host rather "
            "than fail. Set vm.overcommit_memory=2 with an overcommit_ratio that keeps "
            "CommitLimit below physical RAM -- see "
            "/etc/sysctl.d/99-neuron-host-memory.conf.",
            _overcommit_mode(),
        )

    # `check_and_update_config` runs in every process that builds a platform -- the
    # driver *and* each spawned worker. Only the first of those sees an idle machine, and
    # only it can meaningfully project the whole world's demand: by the time rank 3 asks,
    # ranks 0-2 hold 24 GB each and the projection would always fail. So the later
    # processes report and leave the decision to the one that went first.
    if info.get("Committed_AS", 0.0) > per_rank:
        logger.info(
            "Host memory: %.1f GiB already committed, so a rank is up and this process "
            "is not the one that budgets for the run; not enforcing here.",
            info.get("Committed_AS", 0.0),
        )
        return

    if projected + RESERVE_GIB > grantable:
        raise RuntimeError(
            f"host memory: {world_size} rank(s) project {projected:.1f} GiB of commitment "
            f"plus {RESERVE_GIB:.1f} GiB reserve, and only {grantable:.1f} GiB is "
            f"grantable (CommitLimit {info.get('CommitLimit', 0):.1f} GiB - already "
            f"committed {info.get('Committed_AS', 0):.1f} GiB). This projection is "
            f"linear in the rank count and the only 4-rank measurement came in 2.4x "
            f"higher, so this is the optimistic case failing. Reduce the rank count, "
            f"raise vm.overcommit_ratio if physical RAM genuinely allows it, or set "
            f"NEURON_HOST_MEM_PER_RANK_GIB if {per_rank:.1f} GiB a rank is wrong for "
            f"this model."
        )


def _overcommit_mode() -> int:
    try:
        with open("/proc/sys/vm/overcommit_memory") as f:
            return int(f.read().strip())
    except (OSError, ValueError):  # pragma: no cover
        return -1
