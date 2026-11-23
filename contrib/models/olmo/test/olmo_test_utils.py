# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OLMo contrib tests.

The CPU tests never need the 4.7 GB FP32 weights of ``allenai/OLMo-1B-hf``:

* ``config_dir`` gives a directory with the checkpoint's ``config.json`` and
  tokenizer files. It downloads only the JSON files of the pinned revision.
* ``checkpoint_tensor_shapes`` gives the name and shape of every checkpoint
  tensor. It reads only the safetensors header (over HTTP range requests when
  the checkpoint is not on disk).

Set ``OLMO_CHECKPOINT_PATH`` to a local copy of the pinned revision to run
every test without network access.
"""

import json
import os
from pathlib import Path

MODEL_ID = "allenai/OLMo-1B-hf"
# The revision the port was validated with.
HF_REVISION = "aee7752d9c08ee4775e9b0091426d8410e8f6a89"
CHECKPOINT_ENV = "OLMO_CHECKPOINT_PATH"

# The topology of the recorded runs.
WORLD_SIZE = 8


def _checkpoint_override() -> str | None:
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        return None
    if not (Path(path) / "config.json").exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_ENV}={path!r} has no config.json. Set it to a local "
            f"copy of {MODEL_ID} at revision {HF_REVISION}."
        )
    return path


def resolve_checkpoint() -> str:
    """Return a local directory with the full checkpoint (weights included).

    Uses ``OLMO_CHECKPOINT_PATH`` when set. Otherwise downloads the pinned
    revision from the Hugging Face Hub (about 4.7 GB).
    """
    override = _checkpoint_override()
    if override:
        return override

    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=HF_REVISION)


def config_dir() -> str:
    """Return a local directory with the config and tokenizer files, no weights.

    Example:
        >>> os.path.exists(os.path.join(config_dir(), "config.json"))
        True
    """
    override = _checkpoint_override()
    if override:
        return override

    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=HF_REVISION, allow_patterns=["*.json"])


def read_config_json(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def checkpoint_tensor_shapes() -> dict[str, tuple[int, ...]]:
    """Return ``{tensor name: stored shape}`` for every checkpoint tensor.

    Reads the safetensors headers only, never the tensor data.
    """
    override = _checkpoint_override()
    if override and any(Path(override).glob("*.safetensors")):
        from safetensors import safe_open

        shapes = {}
        for f in sorted(Path(override).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shapes[key] = tuple(handle.get_slice(key).get_shape())
        return shapes

    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(MODEL_ID, revision=HF_REVISION)
    return {
        name: tuple(info.shape)
        for file_metadata in metadata.files_metadata.values()
        for name, info in file_metadata.tensors.items()
    }
