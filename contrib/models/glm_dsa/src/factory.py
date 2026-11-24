# SPDX-License-Identifier: Apache-2.0
"""The class vLLM's ``ModelRegistry`` builds for ``GlmMoeDsaForCausalLM`` checkpoints.

``GLMDsaForCausalLM.from_configs`` validates the converted checkpoint (see
``scripts/convert_fp8_block_to_neuron_fp8.py``) and builds the model.
"""

from .model import GLMDsaForCausalLM

__all__ = ["GLMDsaForCausalLM"]
