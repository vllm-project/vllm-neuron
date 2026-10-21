# SPDX-License-Identifier: Apache-2.0
"""The class vLLM's ``ModelRegistry`` builds for ``HYV4ForCausalLM`` checkpoints.

``HYV4ForCausalLM.from_configs`` validates the converted checkpoint (see
``scripts/convert_mxfp8_to_neuron_fp8.py``) and builds the model.
"""

from .model import HYV4ForCausalLM

__all__ = ["HYV4ForCausalLM"]
