# SPDX-License-Identifier: Apache-2.0
"""CPU tests: a pre-generated GQA decode mask follows the kernel's folded layout.

``attention_block_tkg`` folds the KV heads into the batch (``B * kv_heads`` batches of
``q_heads // kv_heads`` heads). Two layout decisions depend on that folded shape:

1. the block_len resize of the block-KV mask rows (``_fold_gqa_block_kv_mask``), and
2. whether the QK-swap path runs, which wants the mask transposed to
   ``[B, q_heads, S_tkg, S_ctx]`` (``maybe_transpose_mask_for_qk_swap``).

Both must be decided for the folded shape, the way the kernel decides them. These tests
pin the second one: the plugin must pass ``kv_heads`` to nkilib's ``is_qk_swapped``.
Without it, Qwen3 at TP=1 (8 KV heads per rank) with a 16-token verify step
(speculative decoding, ``S_ctx = 1024``) would get the default layout while the kernel takes
the swap path. ``vllm_neuron.envs`` turns the swap path off by default
(``NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1``), so this is latent until the swap is turned on;
the tests turn it on.

    VLLM_NEURON_CPU_MODE=1 pytest \
    test/vllm_neuron/functional/attention/test_gqa_decode_mask_layout.py -q
"""

from __future__ import annotations

import pytest
import torch

import vllm_neuron.functional.attention.attention_decode_mask as adm
from vllm_neuron.functional.attention import attention_decode as ad

try:
    import nkilib.core.attention.attention_tkg_utils as _tkg_utils
except ImportError:  # pragma: no cover
    _tkg_utils = None

pytestmark = [pytest.mark.fast]


@pytest.fixture(autouse=True)
def _swap_on(monkeypatch):
    """The plugin turns the QK-swap path off at import; these tests need it on."""
    if _tkg_utils is not None and hasattr(_tkg_utils, "_ATTN_TKG_NO_SWAP"):
        monkeypatch.setattr(_tkg_utils, "_ATTN_TKG_NO_SWAP", False)
    monkeypatch.setattr(adm, "get_logical_nc_config", lambda: 2)


_swap = pytest.mark.skipif(
    adm._nki_is_qk_swapped is None or "kv_heads" not in adm._QK_SWAP_PARAMS,
    reason="nkilib without the GQA-aware is_qk_swapped predicate",
)


def _kernel_decision(bs, q_heads, kv_heads, s_active, s_ctx, d_head=128):
    return adm._nki_is_qk_swapped(
        bs=bs,
        q_head=q_heads,
        d_head=d_head,
        s_active=s_active,
        curr_sprior=s_ctx,
        lnc=2,
        p_max=128,
        is_block_kv=True,
        is_2byte_kv=True,
        fp8_packed=False,
        fuse_rope=False,
        kv_heads=kv_heads,
    )


@_swap
@pytest.mark.parametrize(
    "bs,q_heads,kv_heads,s_active,s_ctx",
    [
        (1, 32, 8, 16, 1024),  # Qwen3-8B TP1, DFlash verify (K=15)
        (1, 16, 8, 16, 1024),  # Qwen3-ASR-1.7B TP1, DFlash verify
        (4, 16, 8, 16, 2048),
        (8, 32, 8, 1, 1024),  # plain decode, batch 8
        (1, 8, 2, 16, 1024),  # Qwen3-8B TP4 verify
        (1, 4, 1, 16, 1024),  # one KV head per rank (no fold)
        (16, 4, 2, 1, 2048),
    ],
)
def test_mask_layout_matches_kernel_decision(
    monkeypatch, bs, q_heads, kv_heads, s_active, s_ctx
):
    monkeypatch.setattr(adm, "get_logical_nc_config", lambda: 2)
    mask = torch.rand(s_ctx, bs, q_heads, s_active)
    out = adm.maybe_transpose_mask_for_qk_swap(mask, d_head=128, kv_heads=kv_heads)
    if _kernel_decision(bs, q_heads, kv_heads, s_active, s_ctx):
        assert out.shape == (bs, q_heads, s_active, s_ctx)
        torch.testing.assert_close(out, mask.permute(1, 2, 3, 0))
    else:
        assert out is mask


@_swap
def test_tp1_verify_shape_needs_kv_heads(monkeypatch):
    """The case the old call got wrong: unfolded says no swap, the kernel swaps."""
    monkeypatch.setattr(adm, "get_logical_nc_config", lambda: 2)
    if not _kernel_decision(1, 32, 8, 16, 1024):
        pytest.skip(
            "this nkilib does not take the QK-swap path at the TP1 verify shape"
        )
    assert _kernel_decision(1, 32, 1, 16, 1024) is False
    mask = torch.rand(1024, 1, 32, 16)
    assert adm.maybe_transpose_mask_for_qk_swap(mask, d_head=128).shape == mask.shape
    assert adm.maybe_transpose_mask_for_qk_swap(mask, d_head=128, kv_heads=8).shape == (
        1,
        32,
        16,
        1024,
    )


@_swap
def test_attention_decode_wrapper_passes_cache_kv_heads(monkeypatch):
    seen = {}

    def fake(mask, d_head, is_fp8_kv=False, fp8_packed=False, kv_heads=1):
        seen.update(d_head=d_head, kv_heads=kv_heads)
        return mask

    monkeypatch.setattr(ad, "maybe_transpose_mask_for_qk_swap", fake)
    v_cache = torch.zeros(4, 8, 32, 128)
    ad._maybe_transpose_mask_for_qk_swap(torch.zeros(1), v_cache, None, None, False)
    assert seen == {"d_head": 128, "kv_heads": 8}
    ad._maybe_transpose_mask_for_qk_swap(
        torch.zeros(1), torch.zeros(4, 32, 128), None, None, False
    )
    assert seen["kv_heads"] == 1
