# SPDX-License-Identifier: Apache-2.0
"""Module test: one Nanbeige4.2-3B decoder layer, run once for each loop.

This is a real-weight, per-component accuracy test in the sense of
``docs/design/accuracy/module_test_guidelines.md``: it builds a single
``NanbeigeDecoderLayer`` with real checkpoint weights, runs it in isolation,
and gates on ``assert_close_three_way`` (FP32 baseline, BF16 CPU reference,
TP-reconstructed vLLM-Neuron output) against an upstream HuggingFace
``LlamaDecoderLayer`` reference.

Nanbeige4.2 runs its physical decoder layers ``num_loops`` times with shared
weights, and each ``(loop, layer)`` pair owns its own KV cache. The component
property that needs isolating is exactly that: the same weights, called twice
with different inputs, must not read each other's KV cache. So each case runs
the layer for loop 0 and then for loop 1, and compares both the layer output
and the per-loop K/V cache contents.

Prefill is sequence-parallel, so each rank holds ``1/TP`` of the tokens and the
per-rank outputs are concatenated before the comparison, per the guidelines'
"Sequence Parallelism and Context Parallelism" section.

The layers are parameterized at the two ends of the stack (the first and the
last physical layer). The fused decode megakernel does not run on CPU, so this
test covers the prefill path only.

Run with:
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
        NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
        pytest contrib/models/nanbeige/test/module/ -v -s --timeout=600
"""

import functools
import importlib.util
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor

# Load this model's own code and test helper by file location, so the folder
# keeps working when it is copied out of the repository. See
# ``contrib/models/nanbeige/test/_local_src.py``.
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "contrib_nanbeige_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_local_src)

nb = _local_src.load_test_helper("nanbeige_utils")

BLOCK_SIZE = 32
COLLECT_TIMEOUT_S = 3600


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    return nb.get_checkpoint_path()


@pytest.fixture(autouse=True)
def require_cpu_mode():
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.fail("test_decoder_layer.py needs CPU mode. Set VLLM_NEURON_CPU_MODE=1.")


@pytest.mark.parametrize("layer_idx", [0, nb.NUM_PHYSICAL_LAYERS - 1])
@pytest.mark.parametrize("world_size", [2])
def test_nanbeige_decoder_layer_loop_prefill(checkpoint_path, world_size, layer_idx):
    """Prefill one decoder layer for loop 0, then for loop 1, at TP2.

    The two loops share the weights and get different inputs. The test
    compares each output with the upstream ``LlamaDecoderLayer``. It also
    compares the KV cache of each loop with the reference cache of that loop.
    """
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    seq_len = 256
    num_blocks = seq_len // BLOCK_SIZE
    raw = nb.read_raw_config(checkpoint_path)
    inputs = nb.make_layer_inputs(layer_idx, seq_len, raw["hidden_size"])
    goldens = get_or_compute_goldens(
        key_config={
            "model": nb.MODEL_ID.replace("/", "_"),
            "revision": nb.HF_REVISION,
            "test": "decoder_layer_loops",
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "with_decode": False,
            "reference_impl": nb.REFERENCE_IMPL,
        },
        compute_fn=lambda: nb.compute_layer_goldens(
            checkpoint_path, layer_idx, inputs, with_decode=False
        ),
    )

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            nb.load_decoder_layer_runner,
            checkpoint_path,
            layer_idx,
            num_blocks,
            BLOCK_SIZE,
            "cpu",
            False,
        ),
        device="cpu",
        load_timeout=COLLECT_TIMEOUT_S,
    )

    positions = torch.arange(seq_len, dtype=torch.long)
    cos, sin = nb.rope_tables(checkpoint_path, positions)
    caches = None
    for loop in range(nb.NUM_LOOPS):
        executor.dispatch(
            inputs["prefill"][loop],
            positions,
            cos,
            sin,
            nb.layer_prefill_metadata(layer_idx, loop, seq_len, BLOCK_SIZE, num_blocks),
            loop,
            True,
            True,
        )
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        actual = torch.cat([out[0] for out in outputs], dim=0)
        assert_close_three_way(
            baseline=goldens["fp32"][f"prefill_loop{loop}"],
            expected=goldens["bf16"][f"prefill_loop{loop}"],
            actual=actual,
            name=f"nanbeige/layer{layer_idx}/prefill_loop{loop}/tp{world_size}",
        )
        caches = [out[1:] for out in outputs]

    # After both loops, the cache of each loop holds the K/V of its own input.
    # Each rank holds one block of KV heads in a paged layout.
    for loop in range(nb.NUM_LOOPS):
        for kind, offset in (("k", 0), ("v", 1)):
            paged = torch.cat(
                [rank_caches[2 * loop + offset] for rank_caches in caches], dim=1
            )
            # [blocks, heads, block_size, Dh] -> [heads, tokens, Dh]
            actual = paged.permute(1, 0, 2, 3).reshape(
                paged.shape[1], -1, paged.shape[3]
            )[:, :seq_len]
            assert_close_three_way(
                baseline=goldens["fp32"][f"{kind}_cache_loop{loop}"],
                expected=goldens["bf16"][f"{kind}_cache_loop{loop}"],
                actual=actual,
                name=f"nanbeige/layer{layer_idx}/{kind}_cache_loop{loop}",
            )
