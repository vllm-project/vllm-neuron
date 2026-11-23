# SPDX-License-Identifier: Apache-2.0
"""Shared constants and helpers for the StarCoder2 (bigcode/starcoder2-3b) tests.

This file travels with the model: it is the port's own test helper, vendored
into the contrib folder so that nothing outside ``contrib/models/starcoder2/``
is needed. It sits at the ``test/`` root because ``test/unit``, ``test/module``
and ``test/integration`` all import it (``test/conftest.py`` puts this
directory on ``sys.path``).

Checkpoint access, from cheapest to most expensive:

- ``hf_config()`` reads ``config.json`` only.
- ``checkpoint_tensor_info()`` reads the safetensors headers only (names,
  shapes and dtypes). Without a local checkpoint it asks the HuggingFace Hub
  for the header of the pinned revision, which is a few hundred kilobytes.
- ``require_checkpoint()`` returns a local checkpoint directory with the
  weights. The FP32 checkpoint is about 12 GB, so the tests do not download
  it: set ``STARCODER2_CHECKPOINT_PATH`` to a local copy, or the test skips.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

MODEL_ID = "bigcode/starcoder2-3b"
# Revision of the checkpoint that the tests validate. The golden cache key of
# the integration test includes it, so another revision cannot reuse a stale
# golden.
HF_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
ARCHITECTURE = "Starcoder2ForCausalLM"

# 24 query heads and 2 KV heads: TP 4 gives 6 query heads and 1 KV head per
# rank, with 2 ranks for each KV head. TP 4 is the topology of the recorded
# device run.
TP_SIZE = 4

CHECKPOINT_ENV = "STARCODER2_CHECKPOINT_PATH"

_LOCAL_SRC_NAME = "contrib_starcoder2_local_src"


def local_src() -> ModuleType:
    """Return ``test/_local_src.py``, loaded by file location under a unique name.

    A plain ``import _local_src`` would resolve to whichever contrib model
    pytest collected first, so the helper is loaded under a name that is unique
    to this model.
    """
    if _LOCAL_SRC_NAME not in sys.modules:
        path = Path(__file__).resolve().parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(_LOCAL_SRC_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_LOCAL_SRC_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_LOCAL_SRC_NAME]
            raise
    return sys.modules[_LOCAL_SRC_NAME]


def src_package() -> ModuleType:
    """Return this model's ``src`` package."""
    return local_src().src_package()


def src_module(name: str) -> ModuleType:
    """Return a module of this model's ``src`` package, for example ``"model"``."""
    return local_src().src_module(name)


def local_checkpoint() -> str | None:
    """Return the directory in ``STARCODER2_CHECKPOINT_PATH``, or None if unset."""
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        return None
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            f"{CHECKPOINT_ENV}={path} has no config.json. Set {CHECKPOINT_ENV} "
            "to a StarCoder2-3B checkpoint directory."
        )
    return path


def require_checkpoint() -> str:
    """Return a local checkpoint with weights, or skip the test."""
    path = local_checkpoint()
    if path is None:
        pytest.skip(
            f"Needs the StarCoder2-3B weights (about 12 GB). Set {CHECKPOINT_ENV} "
            f"to a local copy of {MODEL_ID} at revision {HF_REVISION}."
        )
    return path


def hf_config():
    """Return the HF config of the checkpoint (``config.json`` only)."""
    from transformers import AutoConfig

    path = local_checkpoint()
    if path is not None:
        return AutoConfig.from_pretrained(path)
    try:
        return AutoConfig.from_pretrained(MODEL_ID, revision=HF_REVISION)
    except OSError as e:
        pytest.skip(f"Cannot read the config of {MODEL_ID}: {e}")


def checkpoint_tensor_info() -> dict[str, tuple[list[int], str]]:
    """Return ``{key: (shape, dtype)}`` for each checkpoint tensor.

    Only the safetensors headers are read, not the tensor data.
    """
    path = local_checkpoint()
    if path is not None:
        from safetensors import safe_open

        info = {}
        for file in safetensor_files(path):
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_slice = f.get_slice(key)
                    info[key] = (
                        list(tensor_slice.get_shape()),
                        tensor_slice.get_dtype(),
                    )
        return info

    from huggingface_hub import get_safetensors_metadata

    try:
        metadata = get_safetensors_metadata(MODEL_ID, revision=HF_REVISION)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Cannot read the safetensors header of {MODEL_ID}: {e}")
    info = {}
    for file_metadata in metadata.files_metadata.values():
        for key, tensor in file_metadata.tensors.items():
            info[key] = (list(tensor.shape), tensor.dtype)
    return info


def safetensor_files(checkpoint: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} has no safetensors file. Use a "
            "checkpoint in the safetensors format."
        )
    return files


def read_checkpoint_tensors(
    checkpoint: str, keys: list[str]
) -> dict[str, torch.Tensor]:
    """Read the given tensors from the checkpoint in their stored dtype."""
    from safetensors import safe_open

    wanted = set(keys)
    tensors = {}
    for path in safetensor_files(checkpoint):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in wanted.intersection(f.keys()):
                tensors[key] = f.get_tensor(key)
    missing = sorted(wanted - set(tensors))
    if missing:
        raise KeyError(f"Checkpoint {checkpoint} has no tensors {missing}.")
    return tensors


class FakeTPGroup:
    """Give the TP size and the rank without a process group."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.device_group = None


def patch_tp_group(monkeypatch, world_size: int, rank: int) -> None:
    """Make ``get_tp_group()`` in ``src/model.py`` return a fake TP group."""
    monkeypatch.setattr(
        src_module("model"), "get_tp_group", lambda: FakeTPGroup(world_size, rank)
    )
