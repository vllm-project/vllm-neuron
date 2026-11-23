# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the GLM (zai-org/glm-4-9b-chat-hf) contrib tests.

This file travels with the model. It is the port's own test helper, cut down
to what the tests in this folder use, and it sits at the ``test/`` root because
``test/unit`` and ``test/integration`` both import it (``test/conftest.py`` puts
this directory on ``sys.path``).

Checkpoint resolution:

* ``GLM4_CHECKPOINT_PATH`` points at a local copy of the pinned revision. The
  tests that need the weights (about 19 GB) run only when it is set, or, for
  the integration test, download the pinned revision.
* Without it, the CPU tests that read only the checkpoint metadata
  (``config.json``, ``model.safetensors.index.json`` and the tokenizer files,
  about 20 MB) download those files of the pinned revision from the
  HuggingFace hub. They skip when the hub cannot be reached.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

MODEL_ID = "zai-org/glm-4-9b-chat-hf"
# The revision that every recorded run of this model used.
HF_REVISION = "8599336fc6c125203efb2360bfaf4c80eef1d1bf"
# Namespaces the cached HF goldens by checkpoint revision.
GOLDEN_VARIANT = f"rev_{HF_REVISION[:12]}"
ARCHITECTURE = "GlmForCausalLM"

CHECKPOINT_ENV = "GLM4_CHECKPOINT_PATH"

_METADATA_PATTERNS = ["*.json", "tokenizer*"]


def _checked(path: str) -> str:
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            f"{CHECKPOINT_ENV}={path!r} has no config.json. Set {CHECKPOINT_ENV} "
            f"to a local copy of {MODEL_ID} at revision {HF_REVISION}."
        )
    return path


def metadata_path() -> str:
    """Return a directory with the checkpoint metadata of the pinned revision.

    The directory holds ``config.json``, ``model.safetensors.index.json`` and
    the tokenizer files, and possibly the weights. Skip the calling test when
    ``GLM4_CHECKPOINT_PATH`` is not set and the hub cannot be reached.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return _checked(path)

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            MODEL_ID, revision=HF_REVISION, allow_patterns=_METADATA_PATTERNS
        )
    except Exception as e:  # noqa: BLE001 - any hub failure skips the test
        pytest.skip(
            f"Cannot download the {MODEL_ID} metadata ({type(e).__name__}: {e}). "
            f"Set {CHECKPOINT_ENV} to a local copy of the checkpoint."
        )


def weights_path() -> str:
    """Return a local checkpoint directory with the weights.

    The weights are about 19 GB, so the unit tests do not download them. Skip
    the calling test when ``GLM4_CHECKPOINT_PATH`` is not set.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        pytest.skip(
            f"This test reads the full {MODEL_ID} weights. Set {CHECKPOINT_ENV} "
            f"to a local copy of revision {HF_REVISION}."
        )
    return _checked(path)


def resolve_checkpoint() -> str:
    """Return a local checkpoint directory with the weights, downloading if needed.

    The integration test uses this: ``GLM4_CHECKPOINT_PATH`` if set, otherwise
    the pinned revision from the HuggingFace hub.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        return _checked(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )


def load_config_dict(checkpoint_path: str) -> dict:
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        return json.load(f)


def load_weight_index(checkpoint_path: str) -> dict[str, str]:
    """Return the checkpoint ``weight_map`` (tensor name -> shard file)."""
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    from safetensors import safe_open

    single = os.path.join(checkpoint_path, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def hf_glm_config(checkpoint_path: str):
    """Return the HF GLM config with the eager attention implementation."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path)
    config._attn_implementation = "eager"
    return config


class MockSlice:
    """Mock checkpoint slice with ``__getitem__`` and ``get_shape``."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return list(self.tensor.shape)
