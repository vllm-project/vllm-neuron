# SPDX-License-Identifier: Apache-2.0
"""One OLMo-3 decoder layer at TP 2 in CPU mode against the HF reference.

Three-way check (HF FP32 vs HF BF16 vs this implementation) of one full
decoder layer in prefill, for a sliding-window layer (default RoPE) and a
full-attention layer (YaRN RoPE). The layer runs on two CPU-mode ranks with
sequence parallelism, so the test covers the TP all-reduce of the full-width
QK-norm, the fp32 reduce-scatters and the KV cache write.

The sliding-window layer uses a window that is smaller than the 128-token
prompt, so the window mask is active. The real 4096-token window is longer
than any prompt a CPU test can run.

Two checkpoints:

* ``tiny``: a random-weight checkpoint with the OLMo-3 layout, built in a
  temporary directory. Always runs.
* ``real``: ``allenai/Olmo-3-7B-Instruct`` from ``OLMO3_CHECKPOINT_PATH``
  (only the weights of one layer are read). Skipped when the variable is not
  set. The window is reduced to 48 for this case; the weights are unchanged.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo3/test/unit/ -v
"""

from functools import partial

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way

from _local_src import src_module
from olmo3_test_utils import (
    FULL_LAYER_IDX,
    SLIDING_LAYER_IDX,
    TINY_WINDOW,
    compute_hf_layer_goldens,
    decoder_layer_model_load,
    make_tiny_checkpoint,
    port_config,
    prefill_metadata,
    real_checkpoint,
    sample_hidden,
)

SEQ_LEN = 128
BLOCK_SIZE = 32
WORLD_SIZE = 2
N_SAMPLES = 2
REAL_WINDOW = 48
COLLECT_TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_olmo3")))


def _checkpoint_and_window(kind: str, tiny_checkpoint: str) -> tuple[str, int]:
    if kind == "tiny":
        return tiny_checkpoint, TINY_WINDOW
    path = real_checkpoint()
    if path is None:
        pytest.skip("Set OLMO3_CHECKPOINT_PATH to a local Olmo-3-7B-Instruct snapshot.")
    return path, REAL_WINDOW


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("kind", ["tiny", "real"])
@pytest.mark.parametrize(
    "layer_idx", [SLIDING_LAYER_IDX, FULL_LAYER_IDX], ids=["sliding", "full"]
)
def test_decoder_layer_prefill_tp2(tiny_checkpoint, kind, layer_idx):
    """Three-way: HF FP32 vs HF BF16 vs this implementation, one decoder layer."""
    from transformers import AutoConfig
    from vllm_neuron.utils.executor import MPExecutor

    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("This test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    checkpoint, window = _checkpoint_and_window(kind, tiny_checkpoint)

    hf_config = AutoConfig.from_pretrained(checkpoint)
    hf_config._attn_implementation = "eager"
    hf_config.sliding_window = window
    goldens = compute_hf_layer_goldens(
        checkpoint, hf_config, layer_idx, SEQ_LEN, N_SAMPLES
    )

    cfg = port_config(hf_config, sliding_window=window)
    num_blocks = SEQ_LEN // BLOCK_SIZE
    executor = MPExecutor(
        world_size=WORLD_SIZE,
        model_load=partial(
            decoder_layer_model_load,
            cfg,
            checkpoint,
            layer_idx,
            num_blocks,
            BLOCK_SIZE,
            "cpu",
        ),
        device="cpu",
    )
    rotary = src_module("model").Olmo3RotaryEmbedding(cfg, cfg.layer_types[layer_idx])
    positions = torch.arange(SEQ_LEN, dtype=torch.int32)
    # FP32 cos and sin, the same as Olmo3Model.forward.
    cos, sin = rotary(positions, device=torch.device("cpu"), dtype=torch.float32)
    actual = []
    for s in range(N_SAMPLES):
        h = sample_hidden(s, SEQ_LEN, cfg.hidden_size)
        meta = prefill_metadata(layer_idx, SEQ_LEN, BLOCK_SIZE, num_blocks)
        executor.dispatch(h, positions, (cos, sin), meta)
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        # Each rank returns its SP chunk of rows.
        actual.append(torch.cat(outputs, dim=0).float())

    assert_close_three_way(
        baseline=goldens["all_fp32"],
        expected=goldens["all_bf16"],
        actual=actual,
        name=f"olmo3_{kind}_layer{layer_idx}_prefill/seq{SEQ_LEN}/tp{WORLD_SIZE}",
    )
