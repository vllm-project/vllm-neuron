# SPDX-License-Identifier: Apache-2.0
"""GLM-4-0414 CPU tests: registration, config and components.

These tests run in CPU mode (``VLLM_NEURON_CPU_MODE=1``) and need no Neuron
device and no weights.

Coverage:
  - The registration contract: the checkpoint's architecture string, the
    factory that ``src/`` exports, vLLM's model protocol and the plugin entry
    point.
  - The config parse from the checkpoint, and the factory checks.
  - RoPE: the column permutation plus rotate-half RoPE with the GLM-4 tables
    gives the same Q and K as the HF partial interleaved RoPE, with the real
    head geometry (head_dim 128, rotary_dim 64).
  - Decoder layer: a tiny randomly initialized GLM-4 decoder layer at TP 2
    (prefill, then decode steps through the KV cache) agrees with the HF
    ``Glm4DecoderLayer`` (FP32 and BF16 references).

The tests that read the checkpoint metadata download ``config.json``, the
weight index and the tokenizer of the pinned revision (about 20 MB) unless
``GLM4_0414_CHECKPOINT_PATH`` is set.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm4/test/unit/test_glm4_cpu.py -v
"""

import copy
import functools

import pytest
import torch

from _local_src import src_module, src_package
from glm4_test_utils import (
    ARCHITECTURE,
    LAYER_IDX,
    load_tiny_decoder_layer,
    metadata_path,
    reference_rope_permutation,
)


def _hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(metadata_path())


# =============================================================================
# Registration, factory and config
# =============================================================================


def test_registration_contract(monkeypatch):
    """The checkpoint's architecture must resolve to this folder's factory.

    A contrib model is not in the plugin's registry. The user registers the
    factory under the checkpoint's architecture string, either from a script
    or through the ``vllm.general_plugins`` entry point in ``src/plugin.py``.
    """
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    assert _hf_config().architectures == [ARCHITECTURE]

    factory_cls = getattr(src_package(), ARCHITECTURE)
    assert factory_cls is src_module("factory").Glm4ForCausalLM
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)

    # The entry point registers the installed package name for the same
    # architecture string.
    calls = []

    class _FakeRegistry:
        @staticmethod
        def register_model(arch, target):
            calls.append((arch, target))

    import vllm

    monkeypatch.setattr(vllm, "ModelRegistry", _FakeRegistry)
    plugin = src_module("plugin")
    plugin.register()
    assert calls == [(ARCHITECTURE, "glm4_neuron.factory:Glm4ForCausalLM")]
    assert plugin.ARCHITECTURE == ARCHITECTURE


def test_config_from_checkpoint():
    """Glm4Config reads the GLM-4-32B-0414 geometry, and the factory accepts it."""
    hf_config = _hf_config()
    config = src_module("config").Glm4Config.from_configs(hf_config, None)
    assert config.hidden_size == 6144
    assert config.intermediate_size == 23040
    assert config.num_hidden_layers == 61
    assert config.num_attention_heads == 48
    assert config.num_key_value_heads == 2
    assert config.head_dim == 128
    assert config.vocab_size == 151552
    assert config.rope_theta == 10000.0
    assert config.partial_rotary_factor == 0.5
    assert config.rotary_dim == 64
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.tie_word_embeddings is False
    assert config.torch_dtype == torch.bfloat16

    src_module("factory").Glm4ForCausalLM._validate_config(hf_config, None)


_BAD_OVERRIDES = [
    ({"torch_dtype": torch.float16}, "supports only bfloat16"),
    ({"quantization_config": {"quant_method": "fp8"}}, "quantized checkpoint"),
    ({"attention_bias": True}, "attention_bias"),
    ({"partial_rotary_factor": 0.0}, "Rotary dimension"),
]


@pytest.mark.parametrize("override, message", _BAD_OVERRIDES)
def test_validate_config_rejects(override, message):
    """The factory stops at construction for a configuration it cannot serve."""
    factory_cls = src_module("factory").Glm4ForCausalLM
    bad = copy.deepcopy(_hf_config())
    for key, value in override.items():
        setattr(bad, key, value)
        if key == "partial_rotary_factor" and isinstance(
            getattr(bad, "rope_parameters", None), dict
        ):
            bad.rope_parameters["partial_rotary_factor"] = value
    with pytest.raises(ValueError, match=message):
        factory_cls._validate_config(bad, None)


# =============================================================================
# RoPE layout
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def test_rope_permutation_matches_hf():
    """Permuted rotate-half RoPE is equal to HF GLM-4 RoPE on the permuted columns."""
    from transformers.models.glm4.modeling_glm4 import (
        Glm4RotaryEmbedding as HFGlm4RotaryEmbedding,
    )
    from transformers.models.glm4.modeling_glm4 import apply_rotary_pos_emb

    model = src_module("model")
    torch.manual_seed(42)
    hf_config = _hf_config()
    config = src_module("config").Glm4Config.from_configs(hf_config, None)
    hd, rot = config.head_dim, config.rotary_dim
    assert (hd, rot) == (128, 64)

    perm = model.glm4_rope_permutation(hd, rot)
    assert perm.tolist() == reference_rope_permutation(hd, rot)

    # Positions to 9215 cover the 9216-token model length of the integration
    # test and the serve example.
    positions = torch.arange(0, 9216, 16, dtype=torch.int64)
    tokens = positions.shape[0]
    q = torch.randn(1, 6, tokens, hd)
    k = torch.randn(1, 1, tokens, hd)

    hf_cos, hf_sin = HFGlm4RotaryEmbedding(hf_config)(q, positions[None])
    q_ref, k_ref = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    cos, sin = model.Glm4RotaryEmbedding(config)(
        positions, torch.device("cpu"), torch.float32
    )
    assert cos.shape == (tokens, hd)
    q_p, k_p = q[..., perm], k[..., perm]
    q_out = q_p * cos + _rotate_half(q_p) * sin
    k_out = k_p * cos + _rotate_half(k_p) * sin

    torch.testing.assert_close(q_out, q_ref[..., perm], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_out, k_ref[..., perm], atol=1e-4, rtol=1e-4)
    # The permutation is the same for Q and K, so the attention scores do not change.
    torch.testing.assert_close(
        q_out @ k_out.transpose(-1, -2),
        q_ref @ k_ref.transpose(-1, -2),
        atol=1e-3,
        rtol=1e-4,
    )

    # The fused-weight permutation moves only the Q and K head columns.
    num_q, num_kv, hidden = 6, 1, 32
    w = torch.randn(hidden, (num_q + 2 * num_kv) * hd)
    x = torch.randn(5, hidden)
    out = x @ model.glm4_permute_qk_columns(w, num_q, num_kv, hd, perm)
    ref = x @ w
    q_cols, kv_cols = num_q * hd, num_kv * hd
    ref_q = ref[:, :q_cols].reshape(5, num_q, hd)[:, :, perm].reshape(5, -1)
    ref_k = ref[:, q_cols : q_cols + kv_cols].reshape(5, num_kv, hd)[:, :, perm]
    ref_v = ref[:, q_cols + kv_cols :]
    expected = torch.cat([ref_q, ref_k.reshape(5, -1), ref_v], dim=1)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_rope_permutation_ignores_meta_device_context():
    """The permutation is a CPU tensor with index values, also under ``meta``.

    The serving runner builds the model under ``torch.device("meta")``. A meta
    permutation has no index values, and the QKV loader then gives non-finite
    Q/K columns.
    """
    model = src_module("model")
    with torch.device("meta"):
        perm = model.glm4_rope_permutation(128, 64)
    assert perm.device.type == "cpu"
    assert perm.tolist() == reference_rope_permutation(128, 64)


# =============================================================================
# Tiny decoder layer against HF (TP 2, prefill and decode)
# =============================================================================

# Tiny GLM-4 geometry: the same head layout rules as GLM-4-32B-0414
# (partial rotary 0.5, GQA, no attention bias), with small sizes.
_TINY = dict(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rms_norm_eps=1e-5,
    attention_bias=False,
    max_position_embeddings=4096,
)
PREFILL_LEN = 64
# The BC gate needs enough error values. With 4 steps (1024 values), the
# histogram BC of two samples from one distribution is about 0.97, so the
# 0.99 gate cannot pass. 64 steps give 16384 values.
DECODE_STEPS = 64
# Number of decode steps in each input of the three-way gate.
DECODE_INPUT_STEPS = 4
BLOCK_SIZE = 32
# 4 blocks hold the 64 prefill tokens and the 64 decode tokens.
NUM_BLOCKS = 4
assert NUM_BLOCKS * BLOCK_SIZE >= PREFILL_LEN + DECODE_STEPS


def _tiny_hf_config():
    from transformers import Glm4Config as HFGlm4Config

    config = HFGlm4Config(
        **_TINY,
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    config._attn_implementation = "eager"
    return config


def _hf_layer_outputs(hf_config, hf_layer, hidden: torch.Tensor) -> torch.Tensor:
    from transformers.models.glm4.modeling_glm4 import (
        Glm4RotaryEmbedding as HFGlm4RotaryEmbedding,
    )

    tokens = hidden.shape[0]
    positions = torch.arange(tokens, dtype=torch.int64)[None]
    x = hidden[None]
    cos, sin = HFGlm4RotaryEmbedding(hf_config)(x, positions)
    mask = torch.triu(
        torch.full((tokens, tokens), torch.finfo(x.dtype).min, dtype=x.dtype),
        diagonal=1,
    )[None, None]
    with torch.no_grad():
        out = hf_layer(
            x,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=(cos, sin),
        )
    if isinstance(out, tuple):
        out = out[0]
    return out[0]


@pytest.mark.parametrize("world_size", [2])
def test_tiny_decoder_layer_matches_hf(world_size: int):
    """Prefill and decode of a tiny GLM-4 decoder layer agree with HF."""
    from transformers.models.glm4.modeling_glm4 import (
        Glm4DecoderLayer as HFGlm4DecoderLayer,
    )

    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.utils.executor import MPExecutor

    torch.manual_seed(42)
    hf_config = _tiny_hf_config()
    hf_fp32 = HFGlm4DecoderLayer(hf_config, LAYER_IDX).eval()
    with torch.no_grad():
        # Give the norms a weight other than 1, so a norm in the wrong place fails.
        for name, param in hf_fp32.named_parameters():
            if name.endswith("layernorm.weight"):
                param.copy_(1.0 + 0.2 * torch.randn_like(param))
    hf_bf16 = copy.deepcopy(hf_fp32).to(torch.bfloat16)
    hf_state = {k: v.detach().clone() for k, v in hf_bf16.state_dict().items()}

    total = PREFILL_LEN + DECODE_STEPS
    hidden = torch.randn(total, _TINY["hidden_size"], dtype=torch.bfloat16)
    ref_fp32 = _hf_layer_outputs(hf_config, hf_fp32, hidden.float())
    ref_bf16 = _hf_layer_outputs(hf_config, hf_bf16, hidden)

    config = src_module("config").Glm4Config.from_configs(hf_config, None)
    assert config.rotary_dim == 32
    rotary = src_module("model").Glm4RotaryEmbedding(config)
    layer_name = f"layers.{LAYER_IDX}.self_attn"
    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    def metadata(query_len: int, slots: torch.Tensor) -> dict:
        return {
            layer_name: {
                "max_query_len": query_len,
                "decode_token_threshold": 1,
                "slot_mapping": slots,
                "block_size": BLOCK_SIZE,
                "block_table_tensor": block_table,
                "max_blocks_per_seq": NUM_BLOCKS,
                "cached_seq_len": None,
                "kv_segment_size": None,
            }
        }

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_tiny_decoder_layer,
            hf_config.to_dict(),
            hf_state,
            NUM_BLOCKS,
            BLOCK_SIZE,
            "cpu",
        ),
        device="cpu",
    )

    positions = torch.arange(PREFILL_LEN, dtype=torch.int64)
    cos, sin = rotary(positions, torch.device("cpu"), torch.bfloat16)
    executor.dispatch(
        hidden[:PREFILL_LEN],
        positions,
        (cos, sin),
        metadata(PREFILL_LEN, torch.arange(PREFILL_LEN, dtype=torch.int64)),
    )
    prefill_out = torch.cat(executor.collect(timeout=600), dim=0)
    assert_close_three_way(
        baseline=ref_fp32[:PREFILL_LEN],
        expected=ref_bf16[:PREFILL_LEN],
        actual=prefill_out.reshape(PREFILL_LEN, -1),
        name=f"tiny_decoder_layer/prefill{PREFILL_LEN}/tp{world_size}",
    )

    decode_rows = []
    for step in range(DECODE_STEPS):
        pos = PREFILL_LEN + step
        positions = torch.tensor([pos], dtype=torch.int64)
        cos, sin = rotary(positions, torch.device("cpu"), torch.bfloat16)
        executor.dispatch(
            hidden[pos : pos + 1], positions, (cos, sin), metadata(1, positions.clone())
        )
        outputs = executor.collect(timeout=600)
        # Decode ends with an all-reduce, so each rank holds the full row.
        for rank_out in outputs[1:]:
            torch.testing.assert_close(rank_out, outputs[0])
        decode_rows.append(outputs[0].reshape(1, -1))
    # Give each group of 4 steps to the gate as one input. The L-inf and L2
    # guards then apply to each group.
    # The BC gate uses the errors of all the groups.
    chunk = DECODE_INPUT_STEPS
    actual = torch.cat(decode_rows, dim=0)
    assert_close_three_way(
        baseline=list(ref_fp32[PREFILL_LEN:].split(chunk)),
        expected=list(ref_bf16[PREFILL_LEN:].split(chunk)),
        actual=list(actual.split(chunk)),
        name=f"tiny_decoder_layer/decode{DECODE_STEPS}/tp{world_size}",
    )
