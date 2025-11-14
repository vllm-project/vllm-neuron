# SPDX-License-Identifier: Apache-2.0
"""VllmNeuronPlugin module."""

import glob
import warnings


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob('/dev/neuron*')
    return len(neuron_devices) > 0


def register():
    """Register the Neuron platform if Neuron devices are present, else return None."""
    if not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. "
            "Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None
    return "vllm_neuron.platform.NeuronPlatform"
