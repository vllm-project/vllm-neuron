# SPDX-License-Identifier: Apache-2.0
"""StableLM CPU component tests.

The references come from the HF transformers StableLM modules. The code under
``src/`` does not make a reference.

- Partial RoPE: the port rotates the full head in a permuted layout. The test
  compares the attention scores Q.K with the HF partial RoPE.
- LayerNorm: the port LayerNorm (weight and bias) against ``nn.LayerNorm``.
- Attention prefill, attention prefill to decode, and the MLP in prefill and
  decode: three-way comparison (FP32 HF, BF16 HF, BF16 port) at TP 2 in CPU
  mode, on a tiny random-weight checkpoint with head_dim 80 and rotary dim
  20 (``stablelm_test_utils.TINY_OVERRIDES``). These cases run the torch
  fallback of each NKI op. The same cases run on Neuron with the real
  checkpoint in ``test/module``.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/stablelm/test/unit/test_components.py -v
"""

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_module
from stablelm_test_utils import (
    HEAD_DIM,
    TINY_TP,
    hf_rotary,
    load_hf_config,
    make_tiny_checkpoint,
    run_attention_prefill,
    run_attention_prefill_to_decode,
    run_mlp,
    write_pinned_config,
)

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE, reason="Set VLLM_NEURON_CPU_MODE=1."
)

SEED = 42


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_stablelm")))


def test_permuted_rope_matches_hf_partial_rope(tmp_path):
    """Q.K scores with the permuted full-head RoPE equal the HF partial-RoPE scores."""
    from transformers.models.stablelm.modeling_stablelm import (
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )

    model_module = src_module("model")
    checkpoint = write_pinned_config(str(tmp_path))
    cfg = src_module("config").StableLmConfig.from_configs(
        load_hf_config(checkpoint), NeuronConfig()
    )
    rotary_dim = cfg.rotary_dim

    torch.manual_seed(SEED)
    num_heads, seq_len = 4, 64
    # Positions far from 0 make a wrong frequency show up.
    positions = torch.arange(seq_len, dtype=torch.int64) * 61
    q = torch.randn(1, num_heads, seq_len, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(1, num_heads, seq_len, HEAD_DIM, dtype=torch.float32)

    # HF reference: rotate the first rotary_dim dims, pass the rest through.
    cos_hf, sin_hf = hf_rotary(checkpoint)(q, positions.unsqueeze(0))
    q_rot, k_rot = hf_apply_rotary_pos_emb(
        q[..., :rotary_dim], k[..., :rotary_dim], cos_hf, sin_hf
    )
    q_hf = torch.cat((q_rot, q[..., rotary_dim:]), dim=-1)
    k_hf = torch.cat((k_rot, k[..., rotary_dim:]), dim=-1)
    scores_hf = q_hf @ k_hf.transpose(-1, -2)

    # Port: permute the head dims, then apply the full-head RoPE.
    perm = model_module.rotary_head_dim_permutation(HEAD_DIM, rotary_dim)
    cos, sin = model_module.StableLmRotaryEmbedding(cfg)(
        positions, device=torch.device("cpu"), dtype=torch.float32
    )
    q_port, k_port = model_module.apply_rotary_pos_emb(
        q[0][..., perm], k[0][..., perm], cos, sin
    )
    scores_port = q_port @ k_port.transpose(-1, -2)

    torch.testing.assert_close(scores_port, scores_hf[0], rtol=1e-4, atol=1e-4)


def test_layer_norm_matches_torch():
    """The port LayerNorm uses its weight, its bias and the eps."""
    norm_cls = src_module("model").StableLmLayerNorm

    torch.manual_seed(SEED)
    hidden = 640
    ref = torch.nn.LayerNorm(hidden, eps=1e-5)
    with torch.no_grad():
        ref.weight.copy_(1.0 + 0.1 * torch.randn(hidden))
        ref.bias.copy_(0.1 * torch.randn(hidden))
    port = norm_cls(hidden, 1e-5, torch.float32)
    port.load_state_dict(ref.state_dict())
    x = 3.0 + 2.0 * torch.randn(16, hidden)
    torch.testing.assert_close(port(x), ref(x), rtol=1e-6, atol=1e-6)


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("seq_len", [256])
def test_attention_prefill(tiny_checkpoint, seq_len: int):
    """Three-way: attention prefill at TP 2 against HF."""
    baseline, expected, actual = run_attention_prefill(
        tiny_checkpoint, TINY_TP, seq_len, SEED
    )
    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"stablelm/attn_prefill_cpu/seq{seq_len}/tp{TINY_TP}",
    )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("prefill_len", [256])
def test_attention_prefill_to_decode(tiny_checkpoint, prefill_len: int):
    """Three-way: one decode step after a prefill, against the HF DynamicCache."""
    baseline, expected, actual = run_attention_prefill_to_decode(
        tiny_checkpoint, TINY_TP, prefill_len, SEED
    )
    assert_close_three_way(
        baseline=[baseline],
        expected=[expected],
        actual=[actual],
        name=f"stablelm/attn_prefill_to_decode_cpu/prefill{prefill_len}/tp{TINY_TP}",
    )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize(
    "phase, num_tokens",
    [
        pytest.param("prefill", 256, id="prefill-256"),
        pytest.param("decode", 4, id="decode-bs4"),
    ],
)
def test_mlp(tiny_checkpoint, phase: str, num_tokens: int):
    """Three-way: the SwiGLU MLP at TP 2 for the prefill and the decode shape."""
    baseline, expected, actual = run_mlp(
        tiny_checkpoint, TINY_TP, phase, num_tokens, SEED
    )
    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"stablelm/mlp_cpu/{phase}{num_tokens}/tp{TINY_TP}",
    )
