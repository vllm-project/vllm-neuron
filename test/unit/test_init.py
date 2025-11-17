# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

# Import the functions directly
from vllm_neuron.__init__ import _is_neuron_dev, register


def test_is_neuron_dev_true():
    """Test _is_neuron_dev when devices are present."""
    with patch('vllm_neuron.__init__.glob.glob') as mock_glob:
        mock_glob.return_value = ['/dev/neuron0', '/dev/neuron1']
        assert _is_neuron_dev() is True


def test_is_neuron_dev_false():
    """Test _is_neuron_dev when no devices are present."""
    with patch('vllm_neuron.__init__.glob.glob') as mock_glob:
        mock_glob.return_value = []
        assert _is_neuron_dev() is False


def test_register_with_devices():
    """Test register when Neuron devices are present."""
    with patch('vllm_neuron.__init__._is_neuron_dev') as mock_is_neuron:
        mock_is_neuron.return_value = True
        result = register()
        assert result == "vllm_neuron.platform.NeuronPlatform"


def test_register_without_devices():
    """Test register when no Neuron devices are present."""
    with patch('vllm_neuron.__init__._is_neuron_dev') as mock_is_neuron:
        mock_is_neuron.return_value = False
        with pytest.warns(UserWarning) as warning_info:
            result = register()
            assert result is None
        # Verify warning message
        assert len(warning_info) == 1
        assert "No Neuron devices found" in str(warning_info[0].message)
