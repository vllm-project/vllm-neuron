# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Mistral components against Hugging Face transformers.

Covers the parts of the port that differ from Llama or that are pure torch:
RoPE (default type, rope_theta 1e6), RMSNorm, and the checkpoint guard that
rejects ``consolidated.safetensors``. No checkpoint and no Neuron device:

    pip install -e contrib/models/mistral
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/mistral/test/unit/test_components.py -v
"""

import pytest
import torch

pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")
from transformers.models.mistral import modeling_mistral as hf_mistral  # noqa: E402

from mistral_neuron.config import MistralConfig  # noqa: E402
from mistral_neuron.model import (  # noqa: E402
    MistralRMSNorm,
    MistralRotaryEmbedding,
    _open_checkpoint,
)


def _hf_config() -> "transformers.MistralConfig":
    return transformers.MistralConfig(
        vocab_size=32768,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
    )


# ── RoPE ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("positions", [[0, 1, 2, 3], list(range(0, 32768, 997)), [32767]])
def test_rope_matches_hf(positions):
    """cos/sin for default RoPE with rope_theta 1e6 match transformers exactly."""
    hf_cfg = _hf_config()
    ours = MistralRotaryEmbedding(MistralConfig.from_configs(hf_cfg))
    hf = hf_mistral.MistralRotaryEmbedding(config=hf_cfg)

    pos = torch.tensor(positions, dtype=torch.int64)
    cos, sin = ours(pos, device=torch.device("cpu"), dtype=torch.float32)

    x = torch.zeros(1, len(positions), 128, dtype=torch.float32)
    hf_cos, hf_sin = hf(x, pos[None, :])

    assert cos.shape == (len(positions), 128)
    torch.testing.assert_close(cos, hf_cos[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin, hf_sin[0], rtol=1e-5, atol=1e-5)


def test_rope_uses_theta_1e6_not_llama_5e5():
    """Guards against the Llama default (5e5) leaking into the Mistral config."""
    ours = MistralRotaryEmbedding(MistralConfig.from_configs(_hf_config()))
    inv_freq = ours._compute_inv_freq(torch.device("cpu"))
    expected = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    torch.testing.assert_close(inv_freq, expected, rtol=0, atol=0)


def test_rope_rejects_llama3_scaling():
    cfg = MistralConfig(
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )
    with pytest.raises(NotImplementedError, match="llama3"):
        MistralRotaryEmbedding(cfg)


# ── RMSNorm ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype):
    torch.manual_seed(0)
    weight = torch.randn(4096) * 0.1 + 1.0
    x = torch.randn(7, 4096, dtype=dtype) * 3.0

    ours = MistralRMSNorm(4096, 1e-5, dtype)
    hf = hf_mistral.MistralRMSNorm(4096, eps=1e-5).to(dtype)
    with torch.no_grad():
        ours.weight.copy_(weight.to(dtype))
        hf.weight.copy_(weight.to(dtype))
        out, ref = ours(x), hf(x)

    assert out.dtype == dtype
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    else:
        # HF casts to bf16 before the weight multiply; this port multiplies in
        # fp32 and casts once. Both are valid; allow one bf16 rounding step.
        torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)


# ── checkpoint guard ─────────────────────────────────────────────────────


def _write_safetensors(path, name):
    from safetensors.torch import save_file

    save_file({name: torch.zeros(2, 2)}, str(path))


def test_open_checkpoint_accepts_hf_layout(tmp_path):
    _write_safetensors(tmp_path / "model-00001-of-00001.safetensors", "model.norm.weight")
    checkpoint = _open_checkpoint(str(tmp_path), None)
    assert checkpoint.get_tensor_names() == {"model.norm.weight"}


def test_open_checkpoint_rejects_consolidated(tmp_path):
    _write_safetensors(tmp_path / "model-00001-of-00001.safetensors", "model.norm.weight")
    _write_safetensors(tmp_path / "consolidated.safetensors", "norm.weight")
    with pytest.raises(ValueError, match="consolidated.safetensors"):
        _open_checkpoint(str(tmp_path), None)
