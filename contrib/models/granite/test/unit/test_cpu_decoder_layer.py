# SPDX-License-Identifier: Apache-2.0
"""Granite decoder layer on CPU ranks against the HF reference.

The test makes a tiny random Granite checkpoint with the PowerLM-3b scalars
and biases. It runs the Neuron GraniteDecoderLayer on 2 CPU ranks (TP2, SP
prefill), first a prefill and then one decode step from the paged KV cache.
The reference is the HF GraniteDecoderLayer in FP32 and in BF16 on the full
sequence. The decode row is the last row of that sequence.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/granite/test/unit/test_cpu_decoder_layer.py -v
"""

import importlib.util
import pathlib
import sys
from functools import partial

import pytest
import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Load this model folder's ``test/_local_src.py`` by file location.

    Every contrib model ships a file with this name, so a plain
    ``import _local_src`` would bind whichever copy entered ``sys.modules``
    first. The per-model name used here cannot collide.
    """
    name = "contrib_granite_local_src"
    if name not in sys.modules:
        path = pathlib.Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()
_reference = _local_src.helper("granite_reference")

attn_metadata = _reference.attn_metadata
hf_decoder_layer = _reference.hf_decoder_layer
hf_rope = _reference.hf_rope
load_hf_config = _reference.load_hf_config
module_mappings = _reference.module_mappings
run_hf_decoder_layer = _reference.run_hf_decoder_layer

pytestmark = [
    pytest.mark.skipif(
        not envs.VLLM_NEURON_CPU_MODE,
        reason="This test runs the CPU fallback paths (VLLM_NEURON_CPU_MODE=1).",
    ),
]

PREFILL_LEN = 64
BLOCK_SIZE = 32
NUM_BLOCKS = 8
LAYER_IDX = 0
LAYER_NAME = f"layers.{LAYER_IDX}.self_attn"


def _tiny_granite_config():
    from transformers import GraniteConfig as HFGraniteConfig

    return HFGraniteConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=True,
        mlp_bias=True,
        hidden_act="silu",
        tie_word_embeddings=True,
        # The PowerLM-3b scalars.
        embedding_multiplier=12.0,
        attention_multiplier=0.015625,
        residual_multiplier=0.22,
        logits_scaling=9.0,
        architectures=["GraniteForCausalLM"],
    )


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Save a tiny random Granite checkpoint with nonzero biases."""
    from transformers import GraniteForCausalLM as HFGranite

    torch.manual_seed(42)
    model = HFGranite(_tiny_granite_config()).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    path = str(tmp_path_factory.mktemp("tiny_granite"))
    model.save_pretrained(path, safe_serialization=True)
    return path


class _LayerHarness(torch.nn.Module):
    """Run one GraniteDecoderLayer with the SP input slice in prefill."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from vllm_neuron.model.neuron_config import NeuronConfig

        GraniteConfig = _local_src.load("config").GraniteConfig
        GraniteDecoderLayer = _local_src.load("model").GraniteDecoderLayer

        hf_config = load_hf_config(checkpoint_path)
        cfg = GraniteConfig.from_configs(hf_config, NeuronConfig())
        self.layer = GraniteDecoderLayer(cfg, layer_idx=LAYER_IDX)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        attn = self.layer.self_attn
        shape = (
            NUM_BLOCKS,
            attn.num_key_value_heads_per_rank,
            BLOCK_SIZE,
            cfg.head_dim,
        )
        attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16)
        attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16)

    def forward(self, hidden_states, positions, cos, sin, metadata, phase):
        if phase == "prefill":
            per_rank = hidden_states.shape[0] // self.world_size
            hidden_states = hidden_states[
                self.rank * per_rank : (self.rank + 1) * per_rank
            ]
        return self.layer(hidden_states, positions, (cos, sin), metadata)


def _load_layer(checkpoint_path: str):
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    harness = _LayerHarness(checkpoint_path)
    mappings = module_mappings(
        load_hf_config(checkpoint_path), f"model.layers.{LAYER_IDX}.", "layer."
    )
    result = SafetensorsCheckpoint(checkpoint_path).load_sharded(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        model=harness,
        mappings=mappings,
        device="cpu",
    )
    assert not result.missing_keys, result.missing_keys
    # load_sharded keeps the checkpoint dtype. The layer runs in bfloat16.
    state = {k: v.to(torch.bfloat16) for k, v in result.state_dict.items()}
    missing, unexpected = harness.load_state_dict(state, strict=False, assign=True)
    assert not unexpected, unexpected
    assert not missing, missing
    return harness.eval()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("world_size", [2])
def test_decoder_layer_prefill_then_decode(tiny_checkpoint, world_size):
    """Prefill and one decode step of the layer match the HF reference."""
    torch.manual_seed(42)
    hf_config = load_hf_config(tiny_checkpoint)
    total = PREFILL_LEN + 1
    hidden = torch.randn(total, hf_config.hidden_size, dtype=torch.bfloat16)

    # Independent HF references on the full sequence.
    ref_fp32 = run_hf_decoder_layer(
        hf_decoder_layer(tiny_checkpoint, hf_config, LAYER_IDX, torch.float32),
        hidden.float(),
        hf_config,
    )
    ref_bf16 = run_hf_decoder_layer(
        hf_decoder_layer(tiny_checkpoint, hf_config, LAYER_IDX, torch.bfloat16),
        hidden,
        hf_config,
    )

    cos, sin = hf_rope(hf_config, total, torch.bfloat16)
    cos, sin = cos[0], sin[0]

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(_load_layer, tiny_checkpoint),
        device="cpu",
    )

    prefill_meta = attn_metadata(
        [LAYER_NAME], PREFILL_LEN, BLOCK_SIZE, NUM_BLOCKS, is_prefill=True
    )
    executor.dispatch(
        hidden[:PREFILL_LEN],
        torch.arange(PREFILL_LEN, dtype=torch.long),
        cos[:PREFILL_LEN],
        sin[:PREFILL_LEN],
        prefill_meta,
        "prefill",
    )
    prefill_out = executor.collect(timeout=600)
    neuron_prefill = torch.cat([o.float() for o in prefill_out], dim=0)

    decode_meta = attn_metadata(
        [LAYER_NAME], total, BLOCK_SIZE, NUM_BLOCKS, is_prefill=False
    )
    executor.dispatch(
        hidden[PREFILL_LEN:],
        torch.tensor([PREFILL_LEN], dtype=torch.long),
        cos[PREFILL_LEN:],
        sin[PREFILL_LEN:],
        decode_meta,
        "decode",
    )
    decode_out = executor.collect(timeout=600)
    for i in range(1, len(decode_out)):
        assert torch.equal(decode_out[i], decode_out[0]), (
            f"rank {i} decode output is not equal to rank 0 output"
        )

    assert_close_three_way(
        ref_fp32[:PREFILL_LEN],
        ref_bf16[:PREFILL_LEN],
        neuron_prefill,
        name="granite_cpu_layer/prefill",
    )
    assert_close_three_way(
        ref_fp32[PREFILL_LEN:],
        ref_bf16[PREFILL_LEN:],
        decode_out[0],
        name="granite_cpu_layer/decode",
    )
