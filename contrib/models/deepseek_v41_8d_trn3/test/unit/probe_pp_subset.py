# SPDX-License-Identifier: Apache-2.0
"""Can a subset of the CC world run its own collectives while the rest runs others?

    python kharness_mp.py 16 probe_pp_subset.py

Real pipeline parallelism needs it: stage `s` (8 ranks) runs its layer kernels, whose
collectives span the stage, while the other stages run *other* kernels, and only the
two stages of a boundary meet in the pair collective. Every collective still declares
the full 16-rank mesh in its replica groups (`{0..7}, {8..15}` or the 8 pairs
`(t, 8 + t)`), as the lockstep kernels do.

Ranks 0-7 run kernel A (an all-reduce over their stage, adding their rank) three
times; ranks 8-15 run kernel B (a different NEFF: the same reduce, then times two)
five times, so neither the kernels, their NEFF loads nor their trip counts line up
across the stages. Then every rank runs the pair kernel (sender = stage 0, receiver =
stage 1). An execution timeout (`NEURON_RT_EXEC_TIMEOUT`) means the runtime wants the
whole world in every collective; matching sums mean stage-local collectives are fine.
"""

import sys
import time

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import nki  # noqa: E402
import nki.collectives as ncc  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from kharness import run_nki  # noqa: E402

PT = 128
N = 512


def _reduce(x, groups, scale):
    pid = nl.program_id(0)
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    src = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    dst = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    out = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm)
    if pid == 0:
        t = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[0:PT, 0:N])
        nisa.dma_copy(dst=src, src=t)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=rg, op=nl.add)
    if pid == 0:
        t = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=dst)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=scale)
        nisa.dma_copy(dst=out, src=t)
    nisa.core_barrier(data=out, cores=(0, 1))
    return out


@nki.jit
def k_stage_a(x, groups):
    return _reduce(x, groups, 1.0)


@nki.jit
def k_stage_b(x, groups):
    return _reduce(x, groups, 2.0)


@nki.jit
def k_pair(x, groups, flags):
    """Sender adds `x`, receiver adds zeros: the receiver's output is the sender's `x`."""
    (role,) = flags
    pid = nl.program_id(0)
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    src = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm, name="pp_src")
    dst = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm, name="pp_dst")
    out = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.shared_hbm)
    if pid == 0:
        t = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.sbuf)
        if role == 1:
            nisa.dma_copy(dst=t, src=x[0:PT, 0:N])
        else:
            nisa.memset(t, 0.0)
        nisa.dma_copy(dst=src, src=t)
    nisa.core_barrier(data=src, cores=(0, 1))
    ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=rg, op=nl.add)
    if pid == 0:
        t = nl.ndarray((PT, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=dst)
        nisa.dma_copy(dst=out, src=t)
    nisa.core_barrier(data=out, cores=(0, 1))
    return out


def main():
    assert world == 16, "two stages of 8 ranks"
    tp = 8
    stage = rank // tp
    stage_groups = tuple(tuple(range(s * tp, (s + 1) * tp)) for s in range(2))
    pairs = tuple((t, tp + t) for t in range(tp))
    x = torch.full((PT, N), float(rank), dtype=torch.float32)
    ok = True
    t0 = time.time()
    if stage == 0:
        for i in range(3):
            out = run_nki(k_stage_a, x, stage_groups)
            want = float(sum(range(0, tp)))
            good = bool((out == want).all())
            ok &= good
            print(f"[rank {rank} stage 0] A#{i}: {out[0, 0].item()} (want {want}) {'OK' if good else 'FAIL'}", flush=True)
    else:
        for i in range(5):
            out = run_nki(k_stage_b, x, stage_groups)
            want = 2.0 * float(sum(range(tp, 2 * tp)))
            good = bool((out == want).all())
            ok &= good
            print(f"[rank {rank} stage 1] B#{i}: {out[0, 0].item()} (want {want}) {'OK' if good else 'FAIL'}", flush=True)
    print(f"[rank {rank}] stage kernels done in {time.time() - t0:.1f}s", flush=True)
    # the boundary: stage 0 sends its rank, stage 1 receives it
    role = 1 if stage == 0 else 2
    out = run_nki(k_pair, x, pairs, (role,))
    want = float(rank - tp) if stage == 1 else float(rank)
    good = bool((out == want).all())
    ok &= good
    print(f"[rank {rank}] pair: {out[0, 0].item()} (want {want}) {'OK' if good else 'FAIL'}", flush=True)
    # and again with the stages out of step: stage 1 runs two more B first
    if stage == 1:
        for i in range(2):
            run_nki(k_stage_b, x, stage_groups)
    out = run_nki(k_pair, x, pairs, (role,))
    good = bool((out == want).all())
    ok &= good
    print(f"[rank {rank}] pair after skew: {out[0, 0].item()} (want {want}) {'OK' if good else 'FAIL'}", flush=True)
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'}", flush=True)
    dist.all_reduce(torch.zeros(1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
