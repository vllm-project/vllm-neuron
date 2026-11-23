# SPDX-License-Identifier: Apache-2.0
"""HRM-Text device component checks across 2 nodes (TP 2, one rank on each node).

The recorded run started this file through the repository's inter-node
launcher on each node::

    python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \\
        -m pytest -v \\
        contrib/models/hrm_text/test/module/test_hrm_text_modules_multinode.py

That launcher sets ``NODE_RANK``, ``MASTER_ADDR`` and
``NEURON_RT_ROOT_COMM_ID``, and it lives outside this folder. Nothing here
imports it: set those three variables yourself (the same
``NEURON_RT_ROOT_COMM_ID`` on both nodes) and run pytest directly instead.

Each node runs one MPExecutor worker. The worker on node ``r`` is global rank
``r`` of one TP group of 2, so each collective crosses the nodes. Each node
compares the output of its own rank with independent reference math.

Each test also checks the NKI kernel dispatch. The worker compiles the module
with a recording backend. The backend writes the NKI kernels of each compiled
FX graph to a JSON file in the per-test log directory. The test then checks
that a graph calls the expected kernels.
"""

import functools
import os

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` by file location.

    That helper loads this folder's ``src/`` (and the shared
    ``test/module/common.py``) under module names unique to this contrib model.
    Loading the helper itself by location, under its own unique name, keeps
    every name out of ``sys.path``: two contrib models collected in one pytest
    run cannot resolve to each other's code, and nothing needs the repository
    root.
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "contrib_hrm_text_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()
_common = _local_src.load_common()

ATTN_LAYER_PREFIX = _common.ATTN_LAYER_PREFIX
BLOCK_SIZE = _common.BLOCK_SIZE
DECODE_ATTN_KERNELS = _common.DECODE_ATTN_KERNELS
HIDDEN = _common.HIDDEN
MLP_KERNELS = _common.MLP_KERNELS
MLP_LAYER_PREFIX = _common.MLP_LAYER_PREFIX
PREFILL_ATTN_KERNELS = _common.PREFILL_ATTN_KERNELS
assert_nki_dispatch = _common.assert_nki_dispatch
dispatch_log_path = _common.dispatch_log_path
load_attention_harness = _common.load_attention_harness
load_mlp_harness = _common.load_mlp_harness
load_raw_tensors = _common.load_raw_tensors
ref_gated_attention = _common.ref_gated_attention
ref_mlp = _common.ref_mlp
resolve_checkpoint = _common.resolve_checkpoint
three_way_refs = _common.three_way_refs

pytestmark = [
    pytest.mark.trn2,
    pytest.mark.skipif(
        envs.VLLM_NEURON_CPU_MODE,
        reason="HRM-Text multi-node checks need Neuron devices on 2 nodes.",
    ),
]

NNODES = 2
NUM_SAMPLES = 3
COLLECT_TIMEOUT_S = 3600.0
LOAD_TIMEOUT_S = 1800.0
DEVICE = "neuron:0"


def _node_rank() -> int:
    if "NODE_RANK" not in os.environ or "MASTER_ADDR" not in os.environ:
        raise RuntimeError(
            "NODE_RANK or MASTER_ADDR is not set. Set them, and "
            "NEURON_RT_ROOT_COMM_ID, on each node, or run this test through "
            "test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py."
        )
    return int(os.environ["NODE_RANK"])


def _executor(world_size: int, model_load) -> MPExecutor:
    executor = MPExecutor(
        world_size=world_size,
        model_load=model_load,
        device=DEVICE,
        nnodes=NNODES,
        node_rank=_node_rank(),
        load_timeout=LOAD_TIMEOUT_S,
    )
    assert executor.num_local_ranks == 1
    return executor


def _run(executor: MPExecutor, *args) -> torch.Tensor:
    executor.dispatch(*args)
    (output,) = executor.collect(timeout=COLLECT_TIMEOUT_S)
    return output.float()


def _sp_shard(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    per_rank = t.shape[0] // world_size
    return t[rank * per_rank : (rank + 1) * per_rank]


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
def test_attention_prefill_multinode(world_size):
    """Gated attention prefill with SP: all-gather, attention, reduce-scatter."""
    torch.manual_seed(42)
    rank = _node_rank()
    checkpoint = resolve_checkpoint()
    dispatch_log = dispatch_log_path("attn_prefill", rank)
    tokens = 256
    num_blocks = tokens // BLOCK_SIZE
    raw = load_raw_tensors(
        checkpoint,
        [
            f"{ATTN_LAYER_PREFIX}.attn.gqkv_proj.weight",
            f"{ATTN_LAYER_PREFIX}.attn.o_proj.weight",
        ],
    )
    weights = list(raw.values())

    executor = _executor(
        world_size,
        functools.partial(
            load_attention_harness,
            checkpoint_path=checkpoint,
            num_blocks=num_blocks,
            device=DEVICE,
            dispatch_log=dispatch_log,
        ),
    )
    positions = torch.arange(tokens, dtype=torch.int32)
    slot_mapping = torch.arange(tokens, dtype=torch.int64)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    baselines, expecteds, actuals = [], [], []
    for _ in range(NUM_SAMPLES):
        x = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
        actuals.append(_run(executor, x, positions, slot_mapping, block_table))
        baseline, expected = three_way_refs(ref_gated_attention, x, *weights)
        baselines.append(_sp_shard(baseline, rank, world_size))
        expecteds.append(_sp_shard(expected.float(), rank, world_size))

    assert_close_three_way(
        baselines, expecteds, actuals, name=f"hrm_text_attn_prefill_rank{rank}"
    )
    assert_nki_dispatch(dispatch_log, PREFILL_ATTN_KERNELS)


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
def test_attention_decode_multinode(world_size):
    """Gated attention decode on a paged KV cache that prefill wrote.

    Two sequences use separate blocks. Each sample runs a 128-token prefill of
    each sequence, then one decode step of batch 2 at position 128.
    """
    torch.manual_seed(42)
    rank = _node_rank()
    checkpoint = resolve_checkpoint()
    dispatch_log = dispatch_log_path("attn_decode", rank)
    prefill_len = 128
    blocks_per_seq = 8
    num_blocks = 2 * blocks_per_seq
    raw = load_raw_tensors(
        checkpoint,
        [
            f"{ATTN_LAYER_PREFIX}.attn.gqkv_proj.weight",
            f"{ATTN_LAYER_PREFIX}.attn.o_proj.weight",
        ],
    )
    weights = list(raw.values())

    executor = _executor(
        world_size,
        functools.partial(
            load_attention_harness,
            checkpoint_path=checkpoint,
            num_blocks=num_blocks,
            device=DEVICE,
            dispatch_log=dispatch_log,
        ),
    )

    seq_tables = [
        torch.arange(s * blocks_per_seq, (s + 1) * blocks_per_seq, dtype=torch.int32)
        for s in range(2)
    ]
    seq_base_slot = [s * blocks_per_seq * BLOCK_SIZE for s in range(2)]
    prefill_positions = torch.arange(prefill_len, dtype=torch.int32)
    decode_positions = torch.full((2,), prefill_len, dtype=torch.int32)
    decode_slots = torch.tensor(
        [base + prefill_len for base in seq_base_slot], dtype=torch.int64
    )
    decode_table = torch.stack(seq_tables)

    baselines, expecteds, actuals = [], [], []
    for _ in range(NUM_SAMPLES):
        seqs = [
            torch.randn(prefill_len + 1, HIDDEN).to(torch.bfloat16) for _ in range(2)
        ]
        for s, seq in enumerate(seqs):
            slots = seq_base_slot[s] + torch.arange(prefill_len, dtype=torch.int64)
            _run(
                executor,
                seq[:prefill_len],
                prefill_positions,
                slots,
                seq_tables[s].unsqueeze(0),
            )
        x_decode = torch.stack([seq[prefill_len] for seq in seqs])
        actuals.append(
            _run(executor, x_decode, decode_positions, decode_slots, decode_table)
        )

        base_rows, exp_rows = [], []
        for seq in seqs:
            baseline, expected = three_way_refs(ref_gated_attention, seq, *weights)
            base_rows.append(baseline[-1])
            exp_rows.append(expected[-1].float())
        baselines.append(torch.stack(base_rows))
        expecteds.append(torch.stack(exp_rows))

    assert_close_three_way(
        baselines, expecteds, actuals, name=f"hrm_text_attn_decode_rank{rank}"
    )
    # The worker compiles one prefill graph and one decode graph.
    assert_nki_dispatch(dispatch_log, PREFILL_ATTN_KERNELS)
    assert_nki_dispatch(dispatch_log, DECODE_ATTN_KERNELS)


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
def test_attention_decode_partial_table_multinode(world_size):
    """Gated attention decode with the block table layout of the runner.

    The runner keeps block 0 as the null block. It writes -1 for each table
    entry without a block, and for each row of a pad request. Sequence 0 uses
    blocks 1 to 5 of a table of 8 entries. Row 1 is a pad request with slot -1.
    The test compares row 0 only.
    """
    torch.manual_seed(42)
    rank = _node_rank()
    checkpoint = resolve_checkpoint()
    dispatch_log = dispatch_log_path("attn_decode_partial_table", rank)
    prefill_len = 128
    blocks_per_seq = 8
    used_blocks = prefill_len // BLOCK_SIZE + 1
    num_blocks = blocks_per_seq + 1
    raw = load_raw_tensors(
        checkpoint,
        [
            f"{ATTN_LAYER_PREFIX}.attn.gqkv_proj.weight",
            f"{ATTN_LAYER_PREFIX}.attn.o_proj.weight",
        ],
    )
    weights = list(raw.values())

    executor = _executor(
        world_size,
        functools.partial(
            load_attention_harness,
            checkpoint_path=checkpoint,
            num_blocks=num_blocks,
            device=DEVICE,
            dispatch_log=dispatch_log,
        ),
    )

    seq_table = torch.full((blocks_per_seq,), -1, dtype=torch.int32)
    seq_table[:used_blocks] = torch.arange(1, used_blocks + 1, dtype=torch.int32)
    pad_table = torch.full((blocks_per_seq,), -1, dtype=torch.int32)
    base_slot = BLOCK_SIZE
    prefill_positions = torch.arange(prefill_len, dtype=torch.int32)
    prefill_slots = base_slot + torch.arange(prefill_len, dtype=torch.int64)
    decode_positions = torch.tensor([prefill_len, 0], dtype=torch.int32)
    decode_slots = torch.tensor([base_slot + prefill_len, -1], dtype=torch.int64)
    decode_table = torch.stack([seq_table, pad_table])

    baselines, expecteds, actuals = [], [], []
    for _ in range(NUM_SAMPLES):
        seq = torch.randn(prefill_len + 1, HIDDEN).to(torch.bfloat16)
        pad_row = torch.randn(HIDDEN).to(torch.bfloat16)
        _run(
            executor,
            seq[:prefill_len],
            prefill_positions,
            prefill_slots,
            seq_table.unsqueeze(0),
        )
        x_decode = torch.stack([seq[prefill_len], pad_row])
        output = _run(executor, x_decode, decode_positions, decode_slots, decode_table)
        actuals.append(output[:1])

        baseline, expected = three_way_refs(ref_gated_attention, seq, *weights)
        baselines.append(baseline[-1:])
        expecteds.append(expected[-1:].float())

    assert_close_three_way(
        baselines,
        expecteds,
        actuals,
        name=f"hrm_text_attn_decode_partial_table_rank{rank}",
    )
    assert_nki_dispatch(dispatch_log, DECODE_ATTN_KERNELS)


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
@pytest.mark.parametrize(
    "is_prefill,tokens", [(True, 256), (False, 4)], ids=["prefill", "decode"]
)
def test_mlp_multinode(world_size, is_prefill, tokens):
    """SwiGLU MLP: SP reduce-scatter on prefill, all-reduce on decode."""
    torch.manual_seed(42)
    rank = _node_rank()
    checkpoint = resolve_checkpoint()
    phase = "prefill" if is_prefill else "decode"
    dispatch_log = dispatch_log_path(f"mlp_{phase}", rank)
    raw = load_raw_tensors(
        checkpoint,
        [
            f"{MLP_LAYER_PREFIX}.mlp.gate_up_proj.weight",
            f"{MLP_LAYER_PREFIX}.mlp.down_proj.weight",
        ],
    )
    weights = list(raw.values())

    executor = _executor(
        world_size,
        functools.partial(
            load_mlp_harness,
            checkpoint_path=checkpoint,
            is_prefill=is_prefill,
            device=DEVICE,
            dispatch_log=dispatch_log,
        ),
    )

    baselines, expecteds, actuals = [], [], []
    for _ in range(NUM_SAMPLES):
        x = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
        actuals.append(_run(executor, x))
        baseline, expected = three_way_refs(ref_mlp, x, *weights)
        expected = expected.float()
        if is_prefill:
            baseline = _sp_shard(baseline, rank, world_size)
            expected = _sp_shard(expected, rank, world_size)
        baselines.append(baseline)
        expecteds.append(expected)

    assert_close_three_way(
        baselines, expecteds, actuals, name=f"hrm_text_mlp_{phase}_rank{rank}"
    )
    assert_nki_dispatch(dispatch_log, MLP_KERNELS)
