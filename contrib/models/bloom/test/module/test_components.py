# SPDX-License-Identifier: Apache-2.0
"""CPU-mode component tests for Bloom (bigscience/bloomz-560m).

The tests run layer 0 at TP=2 on CPU and compare three ways, as
``docs/design/accuracy/module_test_guidelines.md`` describes: HF FP32
(baseline), HF BF16 (expected) and vLLM Neuron BF16 (actual). The decode step
uses the torch fallback of the decode megakernel.

Run with:
    VLLM_NEURON_CPU_MODE=1 BLOOM_CHECKPOINT_PATH=/path/to/bloomz-560m \
    pytest contrib/models/bloom/test/module/ -v -s --timeout=1800
"""

import pytest
import torch

import bloom_test_utils as U
from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.golden_cache import get_or_compute_goldens

PREFILL_LEN = 256


@pytest.fixture(scope="module")
def checkpoint():
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip(
            "The Bloom component tests run in CPU mode only. Set "
            "VLLM_NEURON_CPU_MODE=1."
        )
    assert envs.is_native_backend(), (
        "The Bloom tests need the native route. Unset VLLM_NEURON_BACKEND or set "
        "it to neuron_native."
    )
    return U.get_checkpoint()


@pytest.fixture(scope="module")
def hf_config_dict(checkpoint):
    return U.load_hf_config_dict(checkpoint)


@pytest.mark.parametrize("world_size", [2])
def test_attention_prefill_decode_cpu(checkpoint, hf_config_dict, world_size):
    """Prefill PREFILL_LEN tokens, then decode the token at PREFILL_LEN."""
    total = PREFILL_LEN + 1
    num_blocks = U.aligned_num_blocks(total)
    goldens = get_or_compute_goldens(
        U.golden_key("attention", seq_len=total),
        lambda: U.compute_attention_goldens(
            checkpoint, hf_config_dict, total, U.N_SAMPLES
        ),
    )
    executor = U.make_attention_executor(
        checkpoint, hf_config_dict, num_blocks, world_size
    )
    hidden_size = U.hf_config_value(hf_config_dict, "hidden_size")

    prefill_actual, decode_actual = [], []
    for s in range(U.N_SAMPLES):
        h = U.make_hidden(total, hidden_size, s)
        executor.dispatch(
            h[:PREFILL_LEN],
            torch.arange(PREFILL_LEN, dtype=torch.int32),
            U.prefill_metadata(0, PREFILL_LEN, num_blocks),
            True,
        )
        prefill_actual.append(
            torch.cat(executor.collect(timeout=U.COLLECT_TIMEOUT_S), dim=0)
        )
        executor.dispatch(
            h[PREFILL_LEN:],
            torch.tensor([PREFILL_LEN], dtype=torch.int32),
            U.decode_metadata(PREFILL_LEN, num_blocks),
            False,
        )
        outs = executor.collect(timeout=U.COLLECT_TIMEOUT_S)
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=0, atol=0)
        decode_actual.append(outs[0])

    assert_close_three_way(
        [g[:PREFILL_LEN] for g in goldens["all_fp32"]],
        [g[:PREFILL_LEN] for g in goldens["all_bf16"]],
        prefill_actual,
        name=f"bloom_attention_prefill_cpu_tp{world_size}",
    )
    assert_close_three_way(
        [g[PREFILL_LEN:] for g in goldens["all_fp32"]],
        [g[PREFILL_LEN:] for g in goldens["all_bf16"]],
        decode_actual,
        name=f"bloom_attention_decode_cpu_tp{world_size}",
    )


@pytest.mark.parametrize("world_size", [2])
def test_mlp_prefill_decode_cpu(checkpoint, hf_config_dict, world_size):
    goldens_prefill = get_or_compute_goldens(
        U.golden_key("mlp", num_tokens=PREFILL_LEN),
        lambda: U.compute_mlp_goldens(
            checkpoint, hf_config_dict, PREFILL_LEN, U.N_SAMPLES
        ),
    )
    goldens_decode = get_or_compute_goldens(
        U.golden_key("mlp", num_tokens=1),
        lambda: U.compute_mlp_goldens(checkpoint, hf_config_dict, 1, U.N_SAMPLES),
    )
    executor = U.make_mlp_executor(checkpoint, hf_config_dict, world_size)
    hidden_size = U.hf_config_value(hf_config_dict, "hidden_size")

    prefill_actual, decode_actual = [], []
    for s in range(U.N_SAMPLES):
        executor.dispatch(U.make_hidden(PREFILL_LEN, hidden_size, s), True)
        prefill_actual.append(
            torch.cat(executor.collect(timeout=U.COLLECT_TIMEOUT_S), dim=0)
        )
        executor.dispatch(U.make_hidden(1, hidden_size, s), False)
        outs = executor.collect(timeout=U.COLLECT_TIMEOUT_S)
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=0, atol=0)
        decode_actual.append(outs[0])

    assert_close_three_way(
        goldens_prefill["all_fp32"],
        goldens_prefill["all_bf16"],
        prefill_actual,
        name=f"bloom_mlp_prefill_cpu_tp{world_size}",
    )
    assert_close_three_way(
        goldens_decode["all_fp32"],
        goldens_decode["all_bf16"],
        decode_actual,
        name=f"bloom_mlp_decode_cpu_tp{world_size}",
    )
