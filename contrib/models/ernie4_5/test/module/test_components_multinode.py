# SPDX-License-Identifier: Apache-2.0
"""ERNIE 4.5 component tests on Neuron devices, on two nodes.

Each MPI rank is one trn2.48xlarge node. Each node runs one MPExecutor at TP 8
(the original scope) or at TP 1 (the reviewed target). At TP 1 one rank holds
both KV heads, so attention decode uses the per-head GQA block table:

- Node 0 runs the prefill path at a sequence length of 1024 (the serving
  prompt length).
- Node 1 runs the decode path. Attention decode starts from a 1024-token
  prefill cache. MLP decode uses a batch of 4 tokens (the serving batch).

Each check compares FP32 HF, BF16 HF and BF16 vLLM Neuron, with the real
weights of layer 0. At the end, the nodes exchange their results with MPI.
Both nodes fail if one check fails on one node.

Each check also asserts kernel dispatch. A probe records the eligibility
result of each NF op during the real trace on each rank
(``ernie4_5_component_utils.DispatchProbe``). The check fails if a required op
rejects its NKI kernel or if a Torch fallback runs. Prefill attention requires
``qkv_proj``, ``flash_attention`` and ``o_proj``. Decode attention requires
``attention_decode``. The MLP requires ``mlp``.

``test/conftest.py`` forces ``VLLM_NEURON_BACKEND=neuron_native`` (the only
route this implementation supports) before pytest imports this module. The
helper modules come from ``test/_local_src.py``, by file location.

Run with (one process for each node):
    mpirun -n 2 pytest \
        contrib/models/ernie4_5/test/module/test_components_multinode.py -v
"""

import importlib.util
import os
import sys
import traceback
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import (
    assert_close,
    assert_close_three_way,
)
from vllm_neuron.utils.executor import MPExecutor
from vllm_neuron.utils.golden_cache import get_or_compute_goldens


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` helper by file location.

    The tests must keep importing after the folder is copied out of the
    repository, so they resolve ``src/`` and the sibling test helpers from
    their own paths instead of through a ``contrib.models.ernie4_5...``
    namespace path that needs the repository root on ``sys.path``. A plain
    ``import _local_src`` would need this directory on ``sys.path`` and would
    resolve to whichever contrib model got imported first in the process. See
    ``test/_local_src.py``.
    """
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    name = "contrib_ernie4_5_local_src"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the local-src helper at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_local_src = _load_local_src()
cu = _local_src.load_helper("component_utils")
_checkpoint = _local_src.load_helper("checkpoint")
get_ernie_checkpoint = _checkpoint.get_ernie_checkpoint
load_hf_config_dict = _checkpoint.load_hf_config_dict

pytestmark = [
    pytest.mark.skipif(envs.VLLM_NEURON_CPU_MODE, reason="Neuron device test"),
]

NUM_NODES = 2
PREFILL_SEQ_LEN = 1024
DECODE_PREFILL_LEN = 1024
NUM_DECODE_STEPS = 8
MLP_DECODE_TOKENS = 4
N_SAMPLES = 3
# Each decode step gives one token for each sample. The BC of the three-way
# check needs more elements than one token gives. Thus collect this number of
# samples, and do one check for each step over all samples (the GPT-OSS
# pattern). The thresholds are the defaults.
N_DECODE_SAMPLES = 10
DEVICE = "neuron:0"


@pytest.fixture(autouse=True)
def dist_cleanup():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_attention_prefill(checkpoint, hf_config_dict, world_size):
    seeds = list(range(N_SAMPLES))
    goldens = get_or_compute_goldens(
        key_config=cu.golden_key(
            "ernie4_5_attention_prefill", seq_len=PREFILL_SEQ_LEN, seeds=seeds
        ),
        compute_fn=lambda: cu.compute_attention_prefill_goldens(
            checkpoint, PREFILL_SEQ_LEN, seeds
        ),
        metadata=cu.golden_metadata(),
    )
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            cu.load_attention_prefill,
            hf_config_dict=hf_config_dict,
            checkpoint=checkpoint,
            seq_len=PREFILL_SEQ_LEN,
            device=DEVICE,
            cpu_mode=False,
        ),
        device=DEVICE,
    )
    positions = torch.arange(PREFILL_SEQ_LEN, dtype=torch.int64)
    cos, sin = cu.rope(positions, hf_config_dict)
    for i, seed in enumerate(seeds):
        h = cu.prefill_hidden(seed, PREFILL_SEQ_LEN, hf_config_dict["hidden_size"])
        executor.dispatch(
            h, positions, (cos, sin), cu.prefill_metadata(PREFILL_SEQ_LEN)
        )
        outputs = executor.collect(timeout=3600)
        assert_close_three_way(
            baseline=goldens["fp32"][i],
            expected=goldens["bf16"][i],
            actual=torch.cat(outputs, dim=0),
            name=f"ernie4_5_attn_prefill/seq{PREFILL_SEQ_LEN}/tp{world_size}/s{seed}",
        )


def _run_attention_decode(checkpoint, hf_config_dict, world_size):
    seeds = list(range(N_DECODE_SAMPLES))
    goldens = get_or_compute_goldens(
        key_config=cu.golden_key(
            "ernie4_5_attention_decode",
            prefill_len=DECODE_PREFILL_LEN,
            num_decode_steps=NUM_DECODE_STEPS,
            prefill_seed=cu.PREFILL_KV_SEED,
            seeds=seeds,
        ),
        compute_fn=lambda: cu.compute_attention_decode_goldens(
            checkpoint, DECODE_PREFILL_LEN, NUM_DECODE_STEPS, seeds
        ),
        metadata=cu.golden_metadata(),
    )
    num_blocks = cu.decode_num_blocks(DECODE_PREFILL_LEN + NUM_DECODE_STEPS)
    k_paged, v_paged = cu.hf_kv_to_paged(
        goldens["prefill_k_bf16"],
        goldens["prefill_v_bf16"],
        cu.DECODE_BLOCK_SIZE,
        num_blocks,
    )
    k_per_rank, v_per_rank = cu.shard_paged_kv(k_paged, v_paged, world_size)
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            cu.load_attention_decode,
            hf_config_dict=hf_config_dict,
            checkpoint=checkpoint,
            k_paged_per_rank=k_per_rank,
            v_paged_per_rank=v_per_rank,
            device=DEVICE,
            cpu_mode=False,
        ),
        device=DEVICE,
    )

    # A decode step writes its own slot before it reads, and the mask hides
    # the later slots. So each sample sees only the shared prefill cache.
    per_step_neuron = [[] for _ in range(NUM_DECODE_STEPS)]
    for seed in seeds:
        h = cu.decode_hidden(seed, NUM_DECODE_STEPS, hf_config_dict["hidden_size"])
        for step in range(NUM_DECODE_STEPS):
            pos = DECODE_PREFILL_LEN + step
            positions = torch.tensor([pos], dtype=torch.int64)
            cos, sin = cu.rope(positions, hf_config_dict)
            executor.dispatch(
                h[step : step + 1],
                positions,
                (cos, sin),
                cu.decode_metadata(pos, num_blocks),
            )
            outputs = executor.collect(timeout=1800)
            for r in range(1, len(outputs)):
                assert_close(
                    outputs[r],
                    outputs[0],
                    rtol=1e-3,
                    atol=1e-3,
                    name=f"rank_{r}_s{seed}_step{step}",
                )
            per_step_neuron[step].append(outputs[0])

    # One check for each step over all samples. The L-inf and L2 guards still
    # apply to each sample.
    failures = []
    for step in range(NUM_DECODE_STEPS):
        try:
            assert_close_three_way(
                baseline=goldens["per_step_fp32"][step],
                expected=goldens["per_step_bf16"][step],
                actual=per_step_neuron[step],
                name=f"ernie4_5_attn_decode/prefill{DECODE_PREFILL_LEN}"
                f"/tp{world_size}/step{step}/n{len(seeds)}",
            )
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, f"{len(failures)} three-way checks failed:\n" + "\n".join(
        failures
    )


def _run_mlp(checkpoint, hf_config_dict, world_size, is_prefill):
    tokens = PREFILL_SEQ_LEN if is_prefill else MLP_DECODE_TOKENS
    seeds = list(range(N_SAMPLES))
    goldens = get_or_compute_goldens(
        key_config=cu.golden_key("ernie4_5_mlp", tokens=tokens, seeds=seeds),
        compute_fn=lambda: cu.compute_mlp_goldens(checkpoint, tokens, seeds),
        metadata=cu.golden_metadata(),
    )
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            cu.load_mlp,
            hf_config_dict=hf_config_dict,
            checkpoint=checkpoint,
            device=DEVICE,
            cpu_mode=False,
        ),
        device=DEVICE,
    )
    phase = "prefill" if is_prefill else "decode"
    for i, seed in enumerate(seeds):
        h = cu.mlp_hidden(seed, tokens, hf_config_dict["hidden_size"])
        if is_prefill:
            chunks = torch.chunk(h, world_size, dim=0)
            executor.dispatch_per_rank([(chunk, True) for chunk in chunks])
            actual = torch.cat(executor.collect(timeout=3600), dim=0)
        else:
            executor.dispatch(h, False)
            outputs = executor.collect(timeout=1800)
            for r in range(1, len(outputs)):
                assert_close(
                    outputs[r], outputs[0], rtol=1e-3, atol=1e-3, name=f"rank_{r}"
                )
            actual = outputs[0]
        assert_close_three_way(
            baseline=goldens["fp32"][i],
            expected=goldens["bf16"][i],
            actual=actual,
            name=f"ernie4_5_mlp_{phase}/t{tokens}/tp{world_size}/s{seed}",
        )


def _raise_on_any_node(comm, node_rank: int, phase: str, error: str | None):
    """Exchange the results of all nodes. Raise on each node if one node failed."""
    results = comm.allgather((node_rank, phase, error))
    failed = [r for r in results if r[2] is not None]
    if failed:
        lines = [f"node {n} ({p}): {e}" for n, p, e in failed]
        raise AssertionError(
            f"{len(failed)} of {len(results)} nodes failed:\n" + "\n".join(lines)
        )


@pytest.mark.parametrize(
    "world_size, component",
    [
        pytest.param(8, "attention", id="8-attention"),
        pytest.param(8, "mlp", id="8-mlp"),
        pytest.param(1, "attention", id="1-attention"),
        pytest.param(1, "mlp", id="1-mlp"),
    ],
)
@pytest.mark.timeout(7200)
def test_ernie4_5_components_multinode(component: str, world_size: int):
    """Prefill on node 0 and decode on node 1, for one component."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == NUM_NODES, (
        f"This test needs {NUM_NODES} MPI ranks, one for each node. "
        f"It got {comm.Get_size()}. Start it with mpirun -n {NUM_NODES}."
    )
    node_rank = comm.Get_rank()
    phase = "prefill" if node_rank == 0 else "decode"
    # A DI test in the same session can leave a core selection behind.
    os.environ.pop("NEURON_VISIBLE_DEVICES", None)
    torch.manual_seed(42)

    error = None
    try:
        checkpoint = get_ernie_checkpoint()
        hf_config_dict = load_hf_config_dict(checkpoint)
        if component == "attention" and phase == "prefill":
            _run_attention_prefill(checkpoint, hf_config_dict, world_size)
        elif component == "attention":
            _run_attention_decode(checkpoint, hf_config_dict, world_size)
        else:
            _run_mlp(checkpoint, hf_config_dict, world_size, phase == "prefill")
    except Exception as e:  # noqa: BLE001 - report to the other node first
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[node {node_rank}] {component} {phase} failed: {error}")
    _raise_on_any_node(comm, node_rank, f"{component}/{phase}", error)
