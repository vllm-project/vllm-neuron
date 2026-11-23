# SPDX-License-Identifier: Apache-2.0
"""ERNIE 4.5 attention component tests in CPU mode (torch fallback paths).

The tests use the real weights of layer 0 and compare three results:
FP32 HF, BF16 HF and BF16 vLLM Neuron. TP 2 gives each rank 8 Q heads and
1 KV head.

- Prefill: SP slice, all-gather, ``NF.qkv_proj`` with rotate-half RoPE on the
  reordered weights, flash attention, O projection and reduce-scatter.
- Decode: the paged KV cache gets the HF prefill K/V (K in the port order).
  Then a small number of decode steps run through ``NF.attention_decode``.

These tests check numerical correctness of the torch fallback only. They do
not run an NKI kernel.

``test/conftest.py`` forces ``VLLM_NEURON_BACKEND=neuron_native`` (the only
route this implementation supports) before pytest imports this module. The
helper modules come from ``test/_local_src.py``, by file location.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/ernie4_5/test/module/test_components_cpu.py -v
"""

import importlib.util
import sys
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
    pytest.mark.skipif(not envs.VLLM_NEURON_CPU_MODE, reason="CPU-mode test"),
]

N_SAMPLES = 2
# Each decode step gives one token for each sample. The BC of the three-way
# check needs more elements than one token gives. Thus collect this number of
# samples, and do one check for each step over all samples (the GPT-OSS
# pattern). The thresholds are the defaults.
N_DECODE_SAMPLES = 10


@pytest.fixture(autouse=True)
def dist_cleanup():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("world_size", [2])
def test_ernie4_5_attention_prefill_cpu(seq_len: int, world_size: int):
    """Prefill attention of layer 0 against HF (three-way)."""
    torch.manual_seed(42)
    checkpoint = get_ernie_checkpoint()
    hf_config_dict = load_hf_config_dict(checkpoint)
    seeds = list(range(N_SAMPLES))

    goldens = get_or_compute_goldens(
        key_config=cu.golden_key(
            "ernie4_5_attention_prefill", seq_len=seq_len, seeds=seeds
        ),
        compute_fn=lambda: cu.compute_attention_prefill_goldens(
            checkpoint, seq_len, seeds
        ),
        metadata=cu.golden_metadata(),
    )

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            cu.load_attention_prefill,
            hf_config_dict=hf_config_dict,
            checkpoint=checkpoint,
            seq_len=seq_len,
            device="cpu",
            cpu_mode=True,
        ),
        device="cpu",
    )
    positions = torch.arange(seq_len, dtype=torch.int64)
    cos, sin = cu.rope(positions, hf_config_dict)
    for i, seed in enumerate(seeds):
        h = cu.prefill_hidden(seed, seq_len, hf_config_dict["hidden_size"])
        executor.dispatch(h, positions, (cos, sin), cu.prefill_metadata(seq_len))
        outputs = executor.collect(timeout=900)
        assert_close_three_way(
            baseline=goldens["fp32"][i],
            expected=goldens["bf16"][i],
            actual=torch.cat(outputs, dim=0),
            name=f"ernie4_5_attn_prefill_cpu/seq{seq_len}/tp{world_size}/s{seed}",
        )


@pytest.mark.parametrize("prefill_len", [128])
@pytest.mark.parametrize("num_decode_steps", [4])
@pytest.mark.parametrize("world_size", [2])
def test_ernie4_5_attention_decode_cpu(
    prefill_len: int, num_decode_steps: int, world_size: int
):
    """Decode attention of layer 0 against HF (three-way), seeded from the HF KV."""
    torch.manual_seed(42)
    checkpoint = get_ernie_checkpoint()
    hf_config_dict = load_hf_config_dict(checkpoint)
    seeds = list(range(N_DECODE_SAMPLES))

    goldens = get_or_compute_goldens(
        key_config=cu.golden_key(
            "ernie4_5_attention_decode",
            prefill_len=prefill_len,
            num_decode_steps=num_decode_steps,
            prefill_seed=cu.PREFILL_KV_SEED,
            seeds=seeds,
        ),
        compute_fn=lambda: cu.compute_attention_decode_goldens(
            checkpoint, prefill_len, num_decode_steps, seeds
        ),
        metadata=cu.golden_metadata(),
    )

    num_blocks = cu.decode_num_blocks(prefill_len + num_decode_steps)
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
            device="cpu",
            cpu_mode=True,
        ),
        device="cpu",
    )

    # Each sample starts from the same prefill cache. A decode step writes its
    # own slot before it reads, and the mask hides the later slots, so the
    # tokens of an earlier sample have no effect.
    per_step_neuron = [[] for _ in range(num_decode_steps)]
    for seed in seeds:
        h = cu.decode_hidden(seed, num_decode_steps, hf_config_dict["hidden_size"])
        for step in range(num_decode_steps):
            pos = prefill_len + step
            positions = torch.tensor([pos], dtype=torch.int64)
            cos, sin = cu.rope(positions, hf_config_dict)
            executor.dispatch(
                h[step : step + 1],
                positions,
                (cos, sin),
                cu.decode_metadata(pos, num_blocks),
            )
            outputs = executor.collect(timeout=900)
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
    for step in range(num_decode_steps):
        try:
            assert_close_three_way(
                baseline=goldens["per_step_fp32"][step],
                expected=goldens["per_step_bf16"][step],
                actual=per_step_neuron[step],
                name=f"ernie4_5_attn_decode_cpu/prefill{prefill_len}"
                f"/tp{world_size}/step{step}/n{len(seeds)}",
            )
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, f"{len(failures)} three-way checks failed:\n" + "\n".join(
        failures
    )
