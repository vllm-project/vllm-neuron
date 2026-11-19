# SPDX-License-Identifier: Apache-2.0
"""Host-only tests: the config parse, the state cache geometry and the record addressing (torch fallbacks).

    pytest contrib/models/qwen38/test/unit -v
"""

import torch

from qwen38_neuron.config import Qwen38Config
from qwen38_neuron.state import StateBank, StateLayout, plane_views

QWEN38_27B = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 64, "full_attention_interval": 4,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 17408, "vocab_size": 248320,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
    },
}


class _NC:
    tp_size = 8


def test_config_tp8():
    c = Qwen38Config.from_configs(QWEN38_27B, _NC())
    assert len(c.gdn_layers) == 48 and len(c.attention_layers) == 16
    assert c.attention_layers[:2] == (3, 7)
    assert (c.gdn_v_heads, c.gdn_k_heads, c.conv_channels) == (6, 2, 1280)
    assert (c.q_heads, c.kernel_q_heads, c.kv_heads) == (3, 4, 1)
    assert c.rotary_dim == 64 and c.intermediate_per_rank == 2176
    assert c.problems() == []


def test_config_refuses_tp16():
    nc = _NC()
    nc.tp_size = 16
    assert any("query heads" in p for p in Qwen38Config.from_configs(QWEN38_27B, nc).problems())


def test_state_layout_27b_tp8():
    for block in (512, 1024):
        L = StateLayout(gdn_layers=48, v_heads=6, conv_channels=1280, block_size=block)
        assert L.rec_elems == 128 * 6 * 128 and L.conv_elems == 3840
        # each plane holds half of the layers' records per block, exactly
        assert block * L.head_size("rec") * 2 == 24 * L.rec_elems * 4
        assert block * L.head_size("conv") * 2 == 24 * L.conv_elems * 2
        assert L.block_stride("rec") == 24 * L.record_rows("rec")
        assert L.block_stride("conv") == 24 * L.record_rows("conv")


def _planes(L, n_blocks):
    kv = {}
    for which in ("rec", "conv"):
        buf = torch.zeros(2, n_blocks, 1, L.block_size, L.head_size(which), dtype=torch.bfloat16)
        kv[which] = plane_views([buf[0], buf[1]], which)
    return kv


def test_bank_roundtrip_and_isolation():
    """Rows read their own block's records, padding rows (slot -1) read zeros and write nothing, and a write
    touches only the written blocks."""
    L = StateLayout(gdn_layers=4, v_heads=2, conv_channels=640, block_size=128)
    planes = _planes(L, 6)
    torch.manual_seed(0)
    for which in planes:
        for p in planes[which]:
            if which == "rec":
                p.view(torch.float32).copy_(torch.randn(p.numel() // 2))
            else:
                p.copy_(torch.randn(p.shape).to(p.dtype))
        planes[which][0].view(6, -1)[0].zero_()
        planes[which][1].view(6, -1)[0].zero_()
    before = {w: [p.clone() for p in planes[w]] for w in planes}
    slots = torch.tensor([3, -1, 1])
    bank = StateBank(L, planes, slots)
    rec_view = lambda p: p.view(torch.float32).view(6, -1)[:, :2 * L.rec_elems].view(6, 2, 128, 2, 128)  # noqa: E731
    for g in range(4):
        conv, rec = bank.get(g)
        p, j = divmod(g, 2)
        assert torch.equal(rec[0], rec_view(before["rec"][p])[3, j])
        assert torch.equal(rec[2], rec_view(before["rec"][p])[1, j])
        assert not rec[1].any()
        bank.put(g, conv + 1, rec + 1)
    bank.publish()
    for p in range(2):
        after, old = rec_view(planes["rec"][p]), rec_view(before["rec"][p])
        assert torch.equal(after[3], old[3] + 1) and torch.equal(after[1], old[1] + 1)
        for untouched in (0, 2, 4, 5):
            assert torch.equal(after[untouched], old[untouched])


def test_bank_reset_zeroes_first_chunk():
    L = StateLayout(gdn_layers=2, v_heads=2, conv_channels=640, block_size=128)
    planes = _planes(L, 3)
    for w in planes:
        for p in planes[w]:
            if w == "rec":
                p.view(torch.float32).fill_(2)
            else:
                p.fill_(2)
    conv, rec = StateBank(L, planes, torch.tensor([2]), reset=torch.tensor(False)).get(0)
    assert not conv.any() and not rec.any()
    conv, rec = StateBank(L, planes, torch.tensor([2]), reset=torch.tensor(True)).get(1)
    assert bool((rec == 2).all())
