# SPDX-License-Identifier: Apache-2.0
# Import hf_config first. The import registers the "ouro" model type with
# AutoConfig, so that transformers and vLLM parse the checkpoint config without
# remote code.
from . import hf_config  # noqa: F401
from .config import OuroConfig
from .factory import OuroForCausalLM

__all__ = ["OuroForCausalLM", "OuroConfig"]
