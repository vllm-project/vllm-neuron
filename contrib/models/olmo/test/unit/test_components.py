# SPDX-License-Identifier: Apache-2.0
"""OLMo CPU component tests against independent reference math.

- ``OlmoLayerNorm`` against HF ``OlmoLayerNorm`` and a direct mean/variance
  formula. OLMo's norm is a LayerNorm with no weight and no bias, not an
  RMSNorm, so a non-zero input mean separates the two.
- ``OlmoRotaryEmbedding`` and ``apply_rotary_pos_emb`` against the HF RoPE of
  the real config.

The decode kernel computes RoPE on device from the same cos/sin tables; the
integration test covers that path.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo/test/unit/test_components.py -v
"""

import pytest
import torch

from _local_src import src_module
from olmo_test_utils import config_dir

NUM_TOKENS = 1024


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(config_dir())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layer_norm_matches_hf_and_formula(hf_config, dtype):
    """The non-parametric LayerNorm matches HF and the direct formula."""
    from transformers.models.olmo.modeling_olmo import OlmoLayerNorm as HFLayerNorm

    OLMO_LAYER_NORM_EPS = src_module("config").OLMO_LAYER_NORM_EPS
    OlmoLayerNorm = src_module("model").OlmoLayerNorm

    torch.manual_seed(42)
    H = hf_config.hidden_size
    # A non-zero mean and a large scale make an RMSNorm give a wrong result.
    x = (torch.randn(NUM_TOKENS, H) * 3.0 + 0.5).to(dtype)

    norm = OlmoLayerNorm(H, OLMO_LAYER_NORM_EPS)
    port = norm(x)
    hf = HFLayerNorm(H)(x)

    x32 = x.float()
    mean = x32.mean(dim=-1, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
    formula = ((x32 - mean) / torch.sqrt(var + 1e-5)).to(dtype)

    assert port.dtype == dtype
    assert list(norm.parameters()) == []
    torch.testing.assert_close(port, hf, rtol=0, atol=0)
    tol = 1e-5 if dtype == torch.float32 else 1.6e-2
    torch.testing.assert_close(port, formula, rtol=tol, atol=tol)


def test_rotary_matches_hf(hf_config):
    """The RoPE tables and the RoPE rotation match the HF RoPE."""
    from transformers.models.olmo.modeling_olmo import (
        OlmoRotaryEmbedding as HFRotaryEmbedding,
    )
    from transformers.models.olmo.modeling_olmo import (
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )

    OlmoConfig = src_module("config").OlmoConfig
    model_mod = src_module("model")

    torch.manual_seed(42)
    config = OlmoConfig.from_configs(hf_config, None)
    positions = torch.arange(NUM_TOKENS, dtype=torch.int64)

    cos, sin = model_mod.OlmoRotaryEmbedding(config)(
        positions, torch.device("cpu"), torch.float32
    )
    dummy = torch.zeros(1, NUM_TOKENS, hf_config.hidden_size)
    hf_cos, hf_sin = HFRotaryEmbedding(config=hf_config)(dummy, positions[None])
    hf_cos, hf_sin = hf_cos[0], hf_sin[0]
    torch.testing.assert_close(cos, hf_cos, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin, hf_sin, rtol=1e-5, atol=1e-5)

    q = torch.randn(config.num_attention_heads, NUM_TOKENS, config.head_dim)
    k = torch.randn(config.num_key_value_heads, NUM_TOKENS, config.head_dim)
    q_port, k_port = model_mod.apply_rotary_pos_emb(q, k, cos, sin)
    # HF takes [B, heads, T, D] and cos/sin [B, T, D].
    q_hf, k_hf = hf_apply_rotary_pos_emb(
        q.unsqueeze(0), k.unsqueeze(0), hf_cos.unsqueeze(0), hf_sin.unsqueeze(0)
    )
    torch.testing.assert_close(q_port, q_hf[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_port, k_hf[0], rtol=1e-5, atol=1e-5)
