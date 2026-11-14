# SPDX-License-Identifier: Apache-2.0
"""NeuronConfig.decode_steps_between_prefill_chunks parsing and validation (CPU)."""

import pytest

neuron_config = pytest.importorskip("vllm_neuron.model.neuron_config")


@pytest.mark.parametrize("value", [0, 1, 4])
def test_valid_values(value):
    assert neuron_config.validate_decode_steps_between_prefill_chunks(value) == value


@pytest.mark.parametrize("value", [-1, 1.5, "2", True, None])
def test_invalid_values(value):
    with pytest.raises(ValueError, match="decode_steps_between_prefill_chunks"):
        neuron_config.validate_decode_steps_between_prefill_chunks(value)


def test_default_is_off():
    assert neuron_config.NeuronConfig().decode_steps_between_prefill_chunks == 0
