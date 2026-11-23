# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the HyperCLOVAX-SEED-Think-14B tests.

This file travels with the model, so nothing outside
``contrib/models/hyperclovax/`` is needed. It sits at the ``test/`` root
because ``test/unit`` and ``test/integration`` both import it
(``test/conftest.py`` puts this directory on ``sys.path``).

The checkpoint comes from one of two sources:

1. ``HYPERCLOVAX_CHECKPOINT_PATH``, a local directory that holds the checkpoint
   at the pinned revision. The directory itself or its ``files/``
   subdirectory is accepted.
2. Without it, the HuggingFace hub at the pinned revision.

The checkpoint stores float32 weights, about 59 GB. The unit tests never
download it: the tests that read tensor values call
:func:`local_checkpoint_or_skip`, and the tests that read the config, the
weight index or the safetensors headers download those few files only
(:func:`metadata_dir`, :func:`safetensors_tensor_info`). Only the integration
test, which needs Neuron hardware anyway, calls :func:`resolve_checkpoint`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch

MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
# Every test pins this HuggingFace revision of MODEL_ID.
HF_REVISION = "9b74e35d4c7e4ffec489f4171273caca8948a2b9"
ARCHITECTURE = "HyperCLOVAXForCausalLM"
CHECKPOINT_ENV = "HYPERCLOVAX_CHECKPOINT_PATH"
NUM_HIDDEN_LAYERS = 38
HIDDEN_SIZE = 6144

# The topology of the recorded runs: TP 8. 48 Q heads and 8 KV heads give
# 6 Q heads and 1 KV head on each rank.
WORLD_SIZE = 8

# The checkpoint keys of one decoder layer, and the top-level keys.
LAYER_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "post_norm1.weight",
    "post_norm2.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
TOP_LEVEL_KEYS = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")

_METADATA_FILES = ("config.json", "model.safetensors.index.json")
_REQUIRED_FILES = (
    *_METADATA_FILES,
    "tokenizer.json",
    "tokenizer_config.json",
)


def expected_checkpoint_keys(num_layers: int) -> set[str]:
    """Return the checkpoint keys of a HyperCLOVAX model with ``num_layers`` layers."""
    keys = set(TOP_LEVEL_KEYS)
    for layer in range(num_layers):
        keys.update(f"model.layers.{layer}.{k}" for k in LAYER_KEYS)
    return keys


def _is_hyperclovax_config(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    with open(config_path) as f:
        config = json.load(f)
    return (
        ARCHITECTURE in (config.get("architectures") or [])
        and config.get("num_hidden_layers") == NUM_HIDDEN_LAYERS
        and config.get("hidden_size") == HIDDEN_SIZE
    )


def missing_files(path: Path) -> list[str]:
    """Return the names of the checkpoint files that are not in ``path``."""
    missing = [name for name in _REQUIRED_FILES if not (path / name).is_file()]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        missing += [s for s in shards if not (path / s).is_file()]
    return missing


def _complete_dir(root: Path) -> tuple[Path | None, list[str]]:
    """Return the complete checkpoint directory under ``root``, and the notes."""
    notes = []
    for candidate in (root, root / "files"):
        if not candidate.is_dir():
            notes.append(f"{candidate}: directory does not exist")
            continue
        if not _is_hyperclovax_config(candidate):
            notes.append(f"{candidate}: no HyperCLOVAX-SEED-Think-14B config.json")
            continue
        missing = missing_files(candidate)
        if not missing:
            return candidate, notes
        notes.append(f"{candidate}: missing {missing[:6]}")
    return None, notes


def local_checkpoint() -> str | None:
    """Return the checkpoint in ``HYPERCLOVAX_CHECKPOINT_PATH``, or None if unset.

    Raises:
        RuntimeError: If the variable is set but holds no complete checkpoint.
    """
    value = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not value:
        return None
    found, notes = _complete_dir(Path(value))
    if found is None:
        raise RuntimeError(
            f"{CHECKPOINT_ENV}={value!r} holds no complete {MODEL_ID} checkpoint: "
            f"{notes}. Set it to a snapshot of revision {HF_REVISION}."
        )
    return str(found)


def local_checkpoint_or_skip() -> str:
    """Return the local checkpoint, or skip the test if none is configured."""
    import pytest

    path = local_checkpoint()
    if path is None:
        pytest.skip(
            f"Set {CHECKPOINT_ENV} to a local snapshot of {MODEL_ID} at revision "
            f"{HF_REVISION} (about 59 GB) to run this test."
        )
    return path


def resolve_checkpoint() -> str:
    """Return a local directory that holds the pinned checkpoint.

    Use ``HYPERCLOVAX_CHECKPOINT_PATH`` if it is set. Otherwise download the
    pinned revision from the HuggingFace hub (about 59 GB).
    """
    path = local_checkpoint()
    if path is not None:
        return path

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID,
        revision=HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    found, notes = _complete_dir(Path(path))
    if found is None:
        raise RuntimeError(
            f"The download of {MODEL_ID} at revision {HF_REVISION} is not "
            f"complete: {notes}"
        )
    return str(found)


def metadata_dir() -> str:
    """Return a directory with ``config.json`` and the safetensors weight index.

    Use ``HYPERCLOVAX_CHECKPOINT_PATH`` if it is set. Otherwise download only
    those two files of the pinned revision (about 35 KB).
    """
    path = local_checkpoint()
    if path is not None:
        return path

    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(MODEL_ID, name, revision=HF_REVISION)
        for name in _METADATA_FILES
    ]
    return os.path.dirname(paths[0])


def read_weight_map(path: str) -> dict[str, str]:
    """Return the ``weight_map`` of ``model.safetensors.index.json`` in ``path``."""
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def safetensors_tensor_info() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return ``{key: (dtype, shape)}`` for each tensor of the pinned checkpoint.

    Read the safetensors headers of the local checkpoint if
    ``HYPERCLOVAX_CHECKPOINT_PATH`` is set. Otherwise read the headers from the
    HuggingFace hub with ranged requests, without a download of the tensor
    data. The dtype is the safetensors name, for example ``"F32"``.
    """
    path = local_checkpoint()
    info: dict[str, tuple[str, tuple[int, ...]]] = {}
    if path is not None:
        from safetensors import safe_open

        by_file: dict[str, list[str]] = {}
        for key, filename in read_weight_map(path).items():
            by_file.setdefault(filename, []).append(key)
        for filename, keys in sorted(by_file.items()):
            with safe_open(os.path.join(path, filename), framework="pt") as f:
                for key in keys:
                    sl = f.get_slice(key)
                    info[key] = (sl.get_dtype(), tuple(sl.get_shape()))
        return info

    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(MODEL_ID, revision=HF_REVISION)
    for file_metadata in metadata.files_metadata.values():
        for key, tensor in file_metadata.tensors.items():
            info[key] = (tensor.dtype, tuple(tensor.shape))
    return info


def load_checkpoint_tensors(
    checkpoint_path: str, keys: Iterable[str]
) -> dict[str, torch.Tensor]:
    """Read the given tensors from the safetensors files of a checkpoint."""
    from safetensors import safe_open

    keys = list(keys)
    weight_map = read_weight_map(checkpoint_path)
    missing = [k for k in keys if k not in weight_map]
    if missing:
        raise KeyError(f"The checkpoint does not have these keys: {missing}.")

    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for filename, file_keys in by_file.items():
        with safe_open(os.path.join(checkpoint_path, filename), framework="pt") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def use_native_hf_classes(config):
    """Make the transformers Auto classes select the built-in HyperCLOVAX classes.

    The checkpoint ``config.json`` has an ``auto_map`` that points to
    ``modeling_hyperclovax.py``, and the checkpoint does not have that file.
    The plugin's reference loader calls ``from_pretrained`` with
    ``trust_remote_code=True``, which then tries to load the missing file and
    raises ``OSError``. A config that transformers builds itself can have
    ``auto_map=None``, and ``from_config`` then raises ``TypeError``. An empty
    ``auto_map`` prevents both errors. transformers 5.17 has the built-in
    ``HyperCLOVAXForCausalLM``.

    Example:
        >>> config = use_native_hf_classes(AutoConfig.from_pretrained(path))
    """
    config.auto_map = {}
    return config
