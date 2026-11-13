# SPDX-License-Identifier: Apache-2.0
"""CPU tests: registration and config parsing (no Neuron hardware, no weights)."""

import pytest
import torch

from gemma4_neuron import ARCHS
from gemma4_neuron.config import Gemma4Config

# Abbreviated Gemma 4 31B HF config (nested text_config, as shipped).
HF_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "torch_dtype": "bfloat16",
    "vision_config": {"hidden_size": 1152},
    "text_config": {
        "vocab_size": 262144,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_hidden_layers": 6,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "final_logit_softcapping": 30.0,
        "attention_k_eq_v": True,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    },
}


def test_vllm_registry_override():
    from vllm import ModelRegistry

    for arch in ARCHS:
        assert arch in ModelRegistry.get_supported_archs()


def test_neuron_model_list():
    from vllm_neuron.model import registry

    names = {name for name, _ in registry.get_models()}
    assert set(ARCHS) <= names
    # Idempotent: a second register() does not duplicate entries.
    import gemma4_neuron

    gemma4_neuron.register()
    entries = [name for name, _ in registry.get_models() if name in ARCHS]
    assert sorted(entries) == sorted(ARCHS)


def test_config_flattens_text_config():
    cfg = Gemma4Config.from_configs(HF_CONFIG)
    assert cfg.hidden_size == 5376
    assert cfg.num_hidden_layers == 6
    assert cfg.torch_dtype is torch.bfloat16
    assert cfg.attention_k_eq_v is True


@pytest.mark.parametrize(
    "layer, head_dim, kv_heads, theta, rotary",
    [(0, 256, 16, 10000.0, 1.0), (5, 512, 4, 1000000.0, 0.25)],
)
def test_per_layer_geometry(layer, head_dim, kv_heads, theta, rotary):
    cfg = Gemma4Config.from_configs(HF_CONFIG)
    assert cfg.get_layer_head_dim(layer) == head_dim
    assert cfg.get_layer_num_kv_heads(layer) == kv_heads
    assert cfg.get_layer_rope_theta(layer) == theta
    assert cfg.get_layer_partial_rotary_factor(layer) == rotary
    assert cfg.is_global_layer(layer) == (layer == 5)


def test_rejects_non_bf16_quantization():
    from gemma4_neuron.factory import Gemma4ForConditionalGeneration
    from vllm_neuron.model.neuron_config import NeuronConfig

    nc = NeuronConfig()
    nc.quantization = "mxfp8"
    with pytest.raises(ValueError, match="BF16 only"):
        Gemma4ForConditionalGeneration._validate_config(None, nc)


@pytest.mark.parametrize("per_rank_i, chunks", [(5376, 2), (2688, 1), (10752, 3), (4096, 1)])
def test_prefill_mlp_chunk_count(per_rank_i, chunks):
    from gemma4_neuron.model import _num_intermediate_chunks

    assert _num_intermediate_chunks(per_rank_i) == chunks


def test_chunked_geglu_matches_unchunked():
    """Summing per-chunk MLPs over I equals one MLP over the full I (TP=4 shape)."""
    import vllm_neuron.functional as NF
    from nkilib.core.utils.common_types import ActFnType, NormType

    torch.manual_seed(0)
    H, I, T = 256, 5376, 8
    x = torch.randn(T, H, dtype=torch.float32)
    gate, up = torch.randn(H, I) * 0.05, torch.randn(H, I) * 0.05
    down = torch.randn(I, H) * 0.05
    ln_w = torch.rand(1, H) + 0.5
    kw = dict(act_fn=ActFnType.GELU_Tanh_Approx, eps=1e-6, ln_w=ln_w, norm_type=NormType.RMS_NORM)
    full = NF.mlp(x, gate, up, down, **kw)
    s = I // 2
    chunked = NF.mlp(x, gate[:, :s], up[:, :s], down[:s], **kw) + NF.mlp(
        x, gate[:, s:], up[:, s:], down[s:], **kw
    )
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


def test_recovers_global_fields_dropped_by_transformers(tmp_path):
    """transformers 5.17 drops global_head_dim / num_global_key_value_heads when parsing."""
    import json

    from transformers import PretrainedConfig

    (tmp_path / "config.json").write_text(json.dumps(HF_CONFIG))
    parsed = PretrainedConfig(**{k: v for k, v in HF_CONFIG["text_config"].items()
                                 if k not in ("global_head_dim", "num_global_key_value_heads")})
    parsed._name_or_path = str(tmp_path)
    cfg = Gemma4Config.from_configs(parsed)
    assert cfg.get_layer_num_kv_heads(5) == 4
    assert cfg.get_layer_head_dim(5) == 512


@pytest.mark.parametrize("dh, nkh, groups, window, offset", [(256, 4, 2, 64, 32), (512, 1, 8, None, 0)])
def test_grouped_decode_attention_matches_expanded(dh, nkh, groups, window, offset):
    """Grouped (no GQA expansion) decode attention == expanded fp32 reference."""
    from types import SimpleNamespace

    from gemma4_neuron.model import Gemma4Attention

    torch.manual_seed(0)
    B, S_dec, S_ctx = 2, 1, 256
    nh = nkh * groups
    fake = SimpleNamespace(num_key_value_heads_per_rank=nkh, num_key_value_groups=groups,
                           head_dim=dh, scaling=1.0, sliding_window=window)
    q = torch.randn(nh, B * S_dec, dh).bfloat16()
    k = torch.randn(B, nkh, S_ctx, dh).bfloat16()
    v = torch.randn(B, nkh, S_ctx, dh).bfloat16()
    positions = torch.tensor([200 + offset, 150 + offset])
    off = torch.full((B,), offset, dtype=torch.int32) if offset else None
    out = Gemma4Attention._grouped_decode_attention(fake, q, k, v, positions, off, B, S_dec, S_ctx)

    # Reference: expand GQA, fp32, explicit mask (the original decode path).
    ke = k.unsqueeze(2).expand(B, nkh, groups, S_ctx, dh).reshape(B, nh, S_ctx, dh).float()
    ve = v.unsqueeze(2).expand(B, nkh, groups, S_ctx, dh).reshape(B, nh, S_ctx, dh).float()
    qr = q.view(nh, B, S_dec, dh).permute(1, 0, 2, 3).float()
    ctx = torch.arange(S_ctx).view(1, S_ctx) + (off.view(B, 1) if off is not None else 0)
    pos = positions.view(B, 1)
    keep = ctx <= pos
    if window is not None:
        keep &= ctx >= pos - window + 1
    s = (qr @ ke.transpose(-2, -1)).masked_fill(~keep.view(B, 1, 1, S_ctx), float("-inf"))
    ref = (torch.softmax(s, -1) @ ve).permute(0, 2, 1, 3).reshape(B * S_dec, nh * dh)
    torch.testing.assert_close(out.float(), ref.bfloat16().float(), rtol=1e-2, atol=1e-2)


def test_flat_kv_is_one_object_for_swa_and_global_views():
    """SWA and global layers view the same bytes with different shapes; both must get ONE flat tensor."""
    from gemma4_neuron.model import Gemma4ForCausalLM

    raw = torch.zeros(2 * 8 * 4 * 32 * 256, dtype=torch.bfloat16)
    sw = raw.view(2, 8, 4, 32, 256)          # [2, N, heads, bs, D]
    gl = raw.view(2, 8, 1, 128, 256 * 4 // 4)  # same bytes, other shape
    cache = {}
    a = Gemma4ForCausalLM._flat_kv(sw[0], sw[1], cache)
    b = Gemma4ForCausalLM._flat_kv(gl[0], gl[1], cache)
    assert a is b and a.numel() == raw.numel() and a._base is not None
