# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MTP proposer plumbing and the Neuron model registry hook.

VLLM_NEURON_CPU_MODE=1 pytest test/unit/vllm/spec_decode/test_mtp_proposer.py -v
"""

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_neuron.model import registry
from vllm_neuron.vllm.spec_decode import eagle
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer
from vllm_neuron.vllm.spec_decode.mtp import MTPProposer

TARGET_LAYERS = 80
HIDDEN = 4096


def _vllm_config(method: str, num_spec: int = 1, **draft_cfg):
    draft_hf = SimpleNamespace(num_hidden_layers=TARGET_LAYERS, hidden_size=HIDDEN)
    for k, v in draft_cfg.items():
        setattr(draft_hf, k, v)
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        draft_model_config=SimpleNamespace(hf_config=draft_hf),
        model="ckpt",
    )
    return SimpleNamespace(
        speculative_config=spec,
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=TARGET_LAYERS)
        ),
    )


@pytest.fixture(autouse=True)
def _no_world_group(monkeypatch):
    monkeypatch.setattr(eagle, "get_world_group", lambda: None)


def test_eagle_proposer_still_rejects_other_methods():
    with pytest.raises(AssertionError, match="eagle3"):
        EagleProposer(_vllm_config("mtp"), device="cpu")


def test_mtp_proposer_shapes_and_layers():
    p = MTPProposer(_vllm_config("mtp", n_predict=1), device="cpu")
    assert p.method == "mtp"
    # The draft config keeps the target's 80 layers; MTP has n_predict layers.
    assert p._num_draft_layers() == 1
    assert p._target_hidden_states_dim(HIDDEN) == HIDDEN
    assert p.get_target_aux_hidden_state_layers() == (TARGET_LAYERS,)
    with pytest.raises(AssertionError, match="mtp"):
        MTPProposer(_vllm_config("eagle3"), device="cpu")


def test_mtp_proposer_layer_count_fallback():
    p = MTPProposer(_vllm_config("mtp", num_nextn_predict_layers=2), device="cpu")
    assert p._num_draft_layers() == 2
    assert MTPProposer(_vllm_config("mtp"), device="cpu")._num_draft_layers() == 1


def test_eagle_defaults_unchanged():
    p = EagleProposer(_vllm_config("eagle3"), device="cpu")
    assert p._num_draft_layers() == TARGET_LAYERS
    assert p._target_hidden_states_dim(HIDDEN) == 3 * HIDDEN


def test_alias_target_weights_is_idempotent():
    calls = []

    class Draft(nn.Module):
        def alias_target_weights(self, target):
            calls.append(target)

    p = MTPProposer(_vllm_config("mtp"), device="cpu")
    draft = Draft()
    p.model = SimpleNamespace(_orig_mod=draft)  # torch.compile wrapper shape
    target = nn.Linear(2, 2)
    p.alias_target_weights(target)
    p.alias_target_weights(target)
    assert calls == [target]


def test_register_neuron_model(monkeypatch):
    monkeypatch.setattr(registry, "_EXTERNAL_MODELS", {})

    class FakeDraft:
        pass

    registry.register_neuron_model("FakeMTPModel", FakeDraft)
    models = dict(registry.get_models())
    assert models["FakeMTPModel"] is FakeDraft
    assert "LlamaForCausalLM" in models
