# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MuseGlimmer components (no checkpoint, no Neuron device).

    pip install -e contrib/models/muse_glimmer
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/muse_glimmer/test/unit/test_components.py -v
"""

import copy

import pytest
import torch

pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")

from muse_glimmer_neuron import ARCHITECTURE, register  # noqa: E402
from muse_glimmer_neuron.config import MuseGlimmerConfig  # noqa: E402
from muse_glimmer_neuron.model_bf16 import pad_swa_decode_table  # noqa: E402
from muse_glimmer_neuron.utils.decode_kv import build_per_head_block_table  # noqa: E402


def _released_dict() -> dict:
    """The released 52-layer config, built from transformers' defaults."""
    cfg = transformers.MuseGlimmerConfig()
    cfg.architectures = [ARCHITECTURE]
    d = cfg.to_dict()
    t = d["text_config"]
    # The released checkpoint uses RoPE theta 5e5 on sliding layers (NoPE on full layers).
    t["rope_parameters"] = {"rope_theta": 500000.0, "rope_type": "default"}
    t["layer_rope_theta"] = [500000.0 if lt == "sliding_attention" else 0 for lt in t["layer_types"]]
    return d


# ── registration ─────────────────────────────────────────────────────────


def test_register_overrides_the_vllm_gpu_class():
    from vllm import ModelRegistry

    register()
    register()  # idempotent
    cls = ModelRegistry.models[ARCHITECTURE].load_model_cls()
    assert cls.__module__ == "muse_glimmer_neuron.factory"
    assert cls.supports_multimodal


def test_out_of_tree_capabilities_reach_the_plugin_core():
    """The encoder-cache sizing must see this class (it is not in get_models())."""
    from vllm_neuron.model.registry import get_model_cls
    from vllm_neuron.utils import mm_utils

    register()
    hf = transformers.MuseGlimmerConfig()
    hf.architectures = [ARCHITECTURE]
    assert get_model_cls(ARCHITECTURE).__module__ == "muse_glimmer_neuron.factory"
    assert mm_utils.get_vision_token_merge_factor(hf) == 4


# ── config guards ────────────────────────────────────────────────────────


def test_released_config_validates():
    c = MuseGlimmerConfig.from_configs(_released_dict())
    t = c.text_config
    assert t.num_hidden_layers == 52 and t.sliding_window == 2048
    assert [i for i in range(52) if not t.is_sliding(i)] == list(range(3, 52, 4))
    assert all(t.uses_rope(i) == t.is_sliding(i) for i in range(52))
    assert abs(t.attention_scale - 3.87 / 128 ** 0.5) < 1e-12


@pytest.mark.parametrize("key", ["sliding_window", "layer_rope_theta", "post_norm_eps", "qk_scale_factor",
                                 "output_multiplier", "final_logit_softcapping", "layer_types"])
def test_missing_text_key_raises(key):
    d = copy.deepcopy(_released_dict())
    d["text_config"].pop(key)
    with pytest.raises(Exception, match="missing required"):
        MuseGlimmerConfig.from_configs(d)


@pytest.mark.parametrize("key, val", [("post_norm_eps", 1e-5), ("sliding_window", 4096), ("qk_scale_factor", 1.0)])
def test_released_geometry_rejects_changed_values(key, val):
    d = copy.deepcopy(_released_dict())
    d["text_config"][key] = val
    with pytest.raises(Exception, match=key):
        MuseGlimmerConfig.from_configs(d)


# ── sliding-window decode table ──────────────────────────────────────────


@pytest.mark.parametrize("nb, block, lnc, want", [
    (68, 32, 2, 128),   # window 2048 / block 32: 2176 tokens -> 4096
    (68, 32, 1, 128),   # LNC=1: same power-of-two target
    (128, 32, 2, 128),  # already a power of two
    (136, 16, 2, 256),  # block 16: 2176 -> 4096
    (3, 64, 2, 4),      # 192 -> 256
    (1, 32, 2, 8),      # 32 -> 256 (at least LNC * 128)
])
def test_pad_swa_decode_table_width(nb, block, lnc, want):
    table = torch.arange(2 * nb, dtype=torch.int32).view(2, nb)
    out = pad_swa_decode_table(table, block, lnc)
    assert out.shape == (2, want)
    assert torch.equal(out[:, :nb], table)
    assert (out[:, nb:] == -1).all()
    if want == nb:
        assert out is table


def test_pad_swa_decode_table_keeps_3d_tables():
    table = torch.zeros(2, 1, 68, dtype=torch.int32)
    assert pad_swa_decode_table(table, 32, 2).shape == (2, 1, 128)


def test_padded_table_gives_the_same_attention():
    """Torch decode attention over the -1 padded table equals the same window over real blocks."""
    import vllm_neuron.functional as NF

    torch.manual_seed(0)
    H, d, nq, blk, W, B = 256, 128, 8, 32, 2048, 2
    kc, vc = torch.randn(B * 300 + 4, blk, d), torch.randn(B * 300 + 4, blk, d)
    perm = torch.randperm(kc.shape[0] - 1)[: B * 300].view(B, 300) + 1
    x, wqkv = torch.randn(B, 1, H) * 0.5, torch.randn(H, d * (nq + 2)) * 0.03
    ctx = [5000, 5037]

    def tensors(nb):
        table = torch.zeros(B, nb, dtype=torch.int32)
        pos, start = torch.zeros(B, 1), torch.zeros(B, 1)
        for b, c in enumerate(ctx):
            sb = max(c // blk - nb + 1, 0)
            table[b] = perm[b, sb:sb + nb].to(torch.int32)
            pos[b] = c - sb * blk
            start[b] = max(c - sb * blk - (W - 1), 0)
        return table, pos, start

    inv = 1.0 / (5e5 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    f = torch.tensor(ctx, dtype=torch.float32)[:, None] * inv[None]
    cos = f.cos().view(B, 1, d // 2).permute(2, 0, 1).contiguous()
    sin = f.sin().view(B, 1, d // 2).permute(2, 0, 1).contiguous()

    def attend(table, pos, start):
        return NF.attention_decode(
            X=x, W_qkv=wqkv, rmsnorm_X_enabled=False, rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=1e-5, rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None,
            cos=cos, sin=sin, rope_contiguous_layout=True, active_blocks_table=table, K_cache=kc,
            V_cache=vc, pos_ids=pos, swa_start_pos_ids=start, softmax_scale=0.34, update_cache=False,
            W_out=None, K_cache_transposed=False, transposed_out=False, out_in_sb=False)[0]

    t68, p68, s68 = tensors(68)
    ref = attend(*tensors(128))
    out = attend(pad_swa_decode_table(t68, blk, 2), p68, s68)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ── GQA per-head decode table ────────────────────────────────────────────


def test_build_per_head_block_table():
    t = torch.tensor([[0, 1, -1]])
    out = build_per_head_block_table(t, kv_heads=2)
    assert out.tolist() == [[[0, 2, -1], [1, 3, -1]]]
