# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the speculative decoding inputs of the compile-cache fingerprint.

On Neuron, num_speculative_tokens is a compile-time constant of the target
verify graph and of the EAGLE3 draft graph. vLLM's
``SpeculativeConfig.compute_hash()`` covers neither the token count nor the
method name, so the vLLM config hash alone gives every token count the same
fingerprint and the same cache directory.
"""

from types import SimpleNamespace

import pytest
from vllm.config import SpeculativeConfig

from vllm_neuron.vllm.inductor_cache import (
    build_fingerprint_inputs,
    compute_fingerprint,
)

# The inputs a run without speculative decoding had before this change.
_BASE_KEYS = {
    "config_hash",
    "buckets",
    "platform_target",
    "compiler_options",
    "versions",
    "neff_cache_dir",
}

_NEURON_CONFIG = SimpleNamespace(num_seqs_buckets=[1], num_batched_tokens_buckets=[256])


@pytest.fixture(autouse=True)
def _cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")


def _speculative_config(method: str, num_speculative_tokens: int) -> SimpleNamespace:
    # The attributes that SpeculativeConfig.compute_hash() reads, plus the
    # token count. No draft model config keeps the test free of a checkpoint.
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config=None,
    )


def _vllm_config(speculative_config: SimpleNamespace | None) -> SimpleNamespace:
    # Hash the speculative config with vLLM's own compute_hash(), as
    # VllmConfig.compute_hash() does, so the test sees what vLLM hashes.
    def compute_hash() -> str:
        if speculative_config is None:
            return "None"
        return str(SpeculativeConfig.compute_hash(speculative_config))  # type: ignore[arg-type]

    return SimpleNamespace(
        speculative_config=speculative_config,
        optimization_level=None,
        compute_hash=compute_hash,
    )


def _fingerprint(speculative_config: SimpleNamespace | None) -> str:
    inputs = build_fingerprint_inputs(_vllm_config(speculative_config), _NEURON_CONFIG)
    return compute_fingerprint(inputs)


def test_vllm_speculative_hash_ignores_num_speculative_tokens() -> None:
    # The premise: vLLM's hash is the same for every token count.
    hashes = {
        SpeculativeConfig.compute_hash(_speculative_config("eagle3", k))  # type: ignore[arg-type]
        for k in (1, 2, 3)
    }
    assert len(hashes) == 1


def test_fingerprint_differs_by_num_speculative_tokens() -> None:
    fingerprints = [_fingerprint(_speculative_config("eagle3", k)) for k in (1, 2, 3)]
    assert len(set(fingerprints)) == 3


def test_fingerprint_differs_by_speculative_method() -> None:
    # eagle3 and dflash both return auxiliary hidden states, so vLLM hashes
    # them the same when the draft model config is the same.
    assert _fingerprint(_speculative_config("eagle3", 3)) != _fingerprint(
        _speculative_config("dflash", 3)
    )


def test_same_speculative_config_keeps_fingerprint() -> None:
    assert _fingerprint(_speculative_config("eagle3", 2)) == _fingerprint(
        _speculative_config("eagle3", 2)
    )


def test_no_speculative_decoding_keeps_inputs() -> None:
    inputs = build_fingerprint_inputs(_vllm_config(None), _NEURON_CONFIG)
    assert set(inputs) == _BASE_KEYS
    assert _fingerprint(None) == compute_fingerprint(inputs)
