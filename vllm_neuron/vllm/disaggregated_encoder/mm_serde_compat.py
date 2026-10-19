# SPDX-License-Identifier: Apache-2.0
"""Version-portable access to vLLM's multimodal token serde module.

vLLM relocated the module that owns ``encode_mm_kwargs_item`` /
``decode_mm_kwargs_item``: newer releases ship it as
``vllm.entrypoints.scale_out.token_in_token_out.mm_serde``, older ones as
``vllm.entrypoints.serve.disagg.mm_serde``. Both expose the same function
pair, so callers resolve the module here instead of hard-coding one path.
"""

import importlib

MM_SERDE_MODULE_NAMES = (
    "vllm.entrypoints.scale_out.token_in_token_out.mm_serde",
    "vllm.entrypoints.serve.disagg.mm_serde",
)

SERVING_MODULE_NAMES = (
    "vllm.entrypoints.scale_out.token_in_token_out.serving",
    "vllm.entrypoints.serve.disagg.serving",
)


def import_mm_serde():
    """Return the live mm_serde module, trying each known location in order."""
    last_exc = None
    for name in MM_SERDE_MODULE_NAMES:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_exc = exc
    raise last_exc
