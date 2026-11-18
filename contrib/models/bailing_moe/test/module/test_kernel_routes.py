# SPDX-License-Identifier: Apache-2.0
"""Kernel route probe for BailingMoeV2 (inclusionAI/Ling-mini-2.0).

This test calls each NKI eligibility predicate that the model reaches, with
meta tensors in the shapes of one TP 8 / EP 8 rank. It checks the shapes of
the logit test (prefill 256 tokens, batch 1) and of the serving test
(prefill 1024 tokens, decode batch 4).

The result shows kernel SELECTION only. A meta tensor does not execute a
kernel. The device component tests record the NKI nodes of each compiled
graph on each rank. They are the dispatch evidence for the component shapes.
This probe is the only route check for the serving geometry.

Run this test on a Neuron host only. In CPU mode ``can_run_kernel`` is False
for each predicate, so the probe has no meaning there.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    NormType,
    OProjAttentionLayout,
    QKVOutputLayout,
    QuantizationType,
)

from vllm_neuron import envs
from vllm_neuron.functional.attention.attention_cte import (
    _can_use_flash_attention_kernel,
)
from vllm_neuron.functional.attention.attention_decode import (
    _can_use_attention_block_kernel,
)
from vllm_neuron.functional.attention.attention_decode_mask import (
    _can_use_kernel as _can_use_decode_mask_kernel,
)
from vllm_neuron.functional.attention.o_proj import _can_use_kernel as _can_use_o_proj
from vllm_neuron.functional.attention.qkv import _can_use_qkv_kernel
from vllm_neuron.functional.mlp import _can_use_kernel as _can_use_mlp_kernel
from vllm_neuron.functional.moe.moe_cte import _can_use_moe_cte_kernel
from vllm_neuron.functional.moe.moe_tkg import _can_use_kernel as _can_use_moe_tkg

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

utils = _local_src.load_test_module("utils")
_can_use_nki_moe_tkg = _local_src.load("model")._can_use_nki_moe_tkg

HIDDEN = 2048
HEAD_DIM = 128
Q_HEADS_PER_RANK = 2
KV_HEADS_PER_RANK = 1
QKV_WIDTH = (Q_HEADS_PER_RANK + 2 * KV_HEADS_PER_RANK) * HEAD_DIM
DENSE_INTERMEDIATE_PER_RANK = 5120 // utils.WORLD_SIZE
MOE_INTERMEDIATE = 512
LOCAL_EXPERTS = 256 // utils.EP_DEGREE

# (name, prefill tokens, decode batch)
GEOMETRIES = [("logits", 256, 1), ("serving", 1024, 4)]


@pytest.fixture(autouse=True)
def require_device_mode():
    if envs.VLLM_NEURON_CPU_MODE:
        pytest.fail(
            "test_kernel_routes.py is a device test. Unset VLLM_NEURON_CPU_MODE, "
            "and run it on a Neuron host."
        )
    if envs.VLLM_NEURON_DISABLE_NKI_KERNELS:
        pytest.fail(
            "VLLM_NEURON_DISABLE_NKI_KERNELS is set. Unset it to probe the kernel routes."
        )
    torch.manual_seed(42)


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _routes(prefill_tokens: int, decode_batch: int) -> dict[str, bool]:
    """Evaluate each predicate at one geometry of one TP 8 / EP 8 rank."""
    sp_tokens = prefill_tokens // utils.WORLD_SIZE
    routes = {}

    # Prefill attention: QKV with fused RoPE and QK norm, flash attention, o_proj.
    routes["qkv_proj prefill"] = _can_use_qkv_kernel(
        hidden=_meta(1, prefill_tokens, HIDDEN),
        qkv_weights=_meta(HIDDEN, QKV_WIDTH),
        mlp_prev=None,
        attn_prev=None,
        d_head=HEAD_DIM,
        output_layout=QKVOutputLayout.BSD,
        norm_type=NormType.NO_NORM,
        norm_b=None,
        quantization_type=QuantizationType.NONE,
        fused_rope=True,
    )
    routes["flash_attention prefill"] = _can_use_flash_attention_kernel(
        v=_meta(Q_HEADS_PER_RANK, prefill_tokens, HEAD_DIM)
    )
    # NF.o_proj reshapes the [1, T, N*D] input to [1, N, D, T] (BNdS).
    routes["o_proj prefill"] = _can_use_o_proj(
        _meta(1, Q_HEADS_PER_RANK, HEAD_DIM, prefill_tokens),
        _meta(Q_HEADS_PER_RANK * HEAD_DIM, HIDDEN),
        QuantizationType.NONE,
        OProjAttentionLayout.BNdS,
    )

    # Decode attention: mask kernel and fused attention block kernel. With
    # TP 8 each rank holds 1 KV head, so the model passes a 3D cache and a 2D
    # block table.
    routes["attention_decode_mask"] = _can_use_decode_mask_kernel(
        _meta(1, decode_batch, dtype=torch.float32),
        decode_batch,
        1,
        utils.NUM_BLOCKS * utils.BLOCK_SIZE,
    )
    routes["attention_decode"] = _can_use_attention_block_kernel(
        X=_meta(decode_batch, 1, HIDDEN),
        V_cache=_meta(utils.NUM_BLOCKS, utils.BLOCK_SIZE, HEAD_DIM),
        active_blocks_table=_meta(decode_batch, utils.NUM_BLOCKS, dtype=torch.int32),
        attention_dp=1,
    )

    # Dense MLP of layer 0: TP shard of 640. Prefill runs on all tokens.
    routes["dense mlp prefill"] = _can_use_mlp_kernel(
        _meta(prefill_tokens, HIDDEN), _meta(HIDDEN, DENSE_INTERMEDIATE_PER_RANK)
    )
    routes["dense mlp decode"] = _can_use_mlp_kernel(
        _meta(decode_batch, HIDDEN), _meta(HIDDEN, DENSE_INTERMEDIATE_PER_RANK)
    )
    # Shared expert: replicated, full intermediate 512. Prefill runs on the SP shard.
    routes["shared expert prefill"] = _can_use_mlp_kernel(
        _meta(sp_tokens, HIDDEN), _meta(HIDDEN, MOE_INTERMEDIATE)
    )
    routes["shared expert decode"] = _can_use_mlp_kernel(
        _meta(decode_batch, HIDDEN), _meta(HIDDEN, MOE_INTERMEDIATE)
    )

    # Routed experts: 32 local experts with the full intermediate size 512.
    routes["moe_cte prefill"] = _can_use_moe_cte_kernel(
        MoECTEImplementation.shard_on_block,
        _meta(prefill_tokens, HIDDEN),
        _meta(LOCAL_EXPERTS, HIDDEN, 2, MOE_INTERMEDIATE),
        None,
        None,
    )
    decode_hidden = _meta(decode_batch, HIDDEN)
    routes["model moe_tkg gate"] = _can_use_nki_moe_tkg(decode_hidden)
    routes["moe_tkg decode"] = _can_use_moe_tkg(
        hidden_input=decode_hidden,
        expert_down_weights=_meta(LOCAL_EXPERTS, MOE_INTERMEDIATE, HIDDEN),
    )

    return routes


def test_kernel_routes_selected():
    """Assert that each attention, FFN and expert predicate selects its kernel.

    The test checks the logit geometry and the serving geometry.
    """
    fallbacks = []
    for name, prefill_tokens, decode_batch in GEOMETRIES:
        for route, selected in _routes(prefill_tokens, decode_batch).items():
            print(
                f"[bailing kernel route] {name}: {route} -> "
                f"{'NKI' if selected else 'torch'}"
            )
            if not selected:
                fallbacks.append(f"{name}: {route}")
    assert not fallbacks, (
        f"The predicates select the torch fallback for {fallbacks}. "
        "Fix the predicate inputs, or implement the missing kernel path."
    )
