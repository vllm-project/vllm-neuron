# SPDX-License-Identifier: Apache-2.0
"""Phi3 (Phi-4-mini-instruct) device component tests on 2 Trn2 nodes.

Each test makes one MPExecutor with TP=4 across 2 nodes (2 ranks on each
node). All 4 ranks run the sharded module and its collectives. A worker
failure on a node raises in the pytest process of that node. The modules use
layer 0 of the full checkpoint with the original BF16 weights.

The tests use the serving geometry (1024-token prefill) and compare with HF
``Phi3Attention`` and ``Phi3MLP`` in FP32 and in BF16 (three-way). The
full-model prefill test loads all 32 layers. It records the statistics of
each stage, and it compares the logits with the HF full model.

The recorded runs used 2 nodes with 2 TP ranks on each node, started by a
multi-node launcher in the plugin test tree that does not ship with a contrib
model. Set ``NODE_RANK`` (and the usual Neuron multi-node environment) on each
node, then run:

    pytest contrib/models/phi3/test/module/test_phi3_device_modules.py \
        -v --timeout=14400 --tb=short

``phi3_utils.DEVICE_NNODES`` holds the node count. Set it to 1 to run all 4
ranks on one node.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs


def _load_local_src():
    """Return ``test/_local_src.py``, loaded by file location.

    A plain ``import _local_src`` would resolve to whichever contrib model
    pytest collected first, so load the helper under a name that is unique to
    this model. Nothing goes on ``sys.path``, so this test file still imports
    the model's own code after the folder is copied out of the repository.
    """
    name = "contrib_phi3_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


U = _load_local_src().load_test_utils()

pytestmark = pytest.mark.skipif(
    envs.VLLM_NEURON_CPU_MODE,
    reason="Phi3 device tests need Neuron hardware.",
)

SERVING_INPUT_LEN = 1024


@pytest.fixture(scope="module")
def checkpoint() -> str:
    return U.resolve_checkpoint()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


@pytest.mark.parametrize("world_size", [U.WORLD_SIZE], ids=["tp4_2node"])
def test_phi3_attention_flash_prefill_decode(checkpoint, world_size):
    """Flash prefill of 1024 tokens, then 3 decode steps on the megakernel."""
    U.check_attention(
        checkpoint,
        chunk_len=SERVING_INPUT_LEN,
        num_chunks=1,
        num_decode=3,
        segmented=False,
        name="phi3/device/attention_flash/tp4",
    )


@pytest.mark.parametrize("world_size", [U.WORLD_SIZE], ids=["tp4_2node"])
def test_phi3_attention_segmented_prefill_decode(checkpoint, world_size):
    """Segmented (APC) prefill of 2 chunks of 1024 tokens, then 3 decode steps.

    The second chunk attends over the 1024 tokens of the first chunk in the
    paged KV cache, as after a prefix-cache hit.
    """
    U.check_attention(
        checkpoint,
        chunk_len=SERVING_INPUT_LEN,
        num_chunks=2,
        num_decode=3,
        segmented=True,
        name="phi3/device/attention_segmented/tp4",
    )


@pytest.mark.parametrize("world_size", [U.WORLD_SIZE], ids=["tp4_2node"])
def test_phi3_mlp_prefill_decode(checkpoint, world_size):
    """MLP prefill of 1024 tokens and MLP decode of 4 tokens (batch 4)."""
    U.check_mlp(
        checkpoint,
        prefill_len=SERVING_INPUT_LEN,
        decode_tokens=4,
        name="phi3/device/mlp/tp4",
    )


# The chat prompt of the serving logit test has 27 tokens.
SERVING_LOGIT_PROMPT_LEN = 27


@pytest.mark.parametrize("world_size", [U.WORLD_SIZE], ids=["tp4_2node"])
def test_phi3_full_model_prefill(checkpoint, world_size):
    """Full-model segmented prefill with the serving input layout.

    The module tests use random hidden states and an identity block table.
    This test adds the embedding, the norms, the residual adds, the 32-layer
    stack and the LM head. It also uses the serving pad layout: pad ID 0, slot
    0, and -1 in each block-table entry that the request does not own.

    - ``short_padded``: a 27-token prompt padded to 1024 tokens, as in the
      serving logit test.
    - ``full_1024``: 1024 real tokens, as in the 1024-input serving test.

    Each stage must be finite on each rank. The logits at the sampled rows
    must be close to HF FP32 and BF16 (three-way).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    short = U.natural_prompt_ids(tokenizer, SERVING_LOGIT_PROMPT_LEN)
    full = U.natural_prompt_ids(tokenizer, SERVING_INPUT_LEN)
    last_short = SERVING_LOGIT_PROMPT_LEN - 1
    last_full = SERVING_INPUT_LEN - 1
    U.check_full_model_prefill(
        checkpoint,
        cases={
            "short_padded": (short, [0, last_short // 2, last_short]),
            "full_1024": (full, [0, last_full // 2, last_full]),
        },
        padded_len=SERVING_INPUT_LEN,
        name="phi3/device/full_model_prefill/tp4",
    )
