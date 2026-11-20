# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers for the ERNIE 4.5 dense tests.

This copy is local to the model folder, so the tests stay self-contained: the
guide asks a contrib model to keep its own copy of a shared test helper rather
than reach outside its folder. It sits at the root of the model's ``test/``
folder, so ``test/unit``, ``test/module`` and ``test/integration`` share this
one copy; they reach it with ``_local_src.load_helper("checkpoint")``, by file
location, so nothing depends on ``sys.path``.
"""

import json
import os

MODEL_ID = "baidu/ERNIE-4.5-0.3B-PT"
# The tests pin this HF revision. The goldens include it in the cache key.
REVISION = "b565cf6caebdb7a1eadf00100857b1ed5e044f12"

# Facts about the pinned checkpoint. The tests use them to find a wrong
# checkpoint before they compare numbers.
CHECKPOINT_FILE = "model.safetensors"
CHECKPOINT_FILE_BYTES = 721_514_672
CHECKPOINT_NUM_TENSORS = 164
ARCHITECTURE = "Ernie4_5ForCausalLM"

# Set this variable to a local snapshot of the pinned revision to skip the
# download.
CHECKPOINT_DIR_ENV = "ERNIE4_5_CHECKPOINT_DIR"


def _validate_checkpoint_dir(path: str) -> str:
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"ERNIE 4.5 checkpoint {path} has no config.json. Give a snapshot "
            f"of {MODEL_ID} at revision {REVISION}."
        )
    with open(config_path) as f:
        config = json.load(f)
    if ARCHITECTURE not in config.get("architectures", []):
        raise ValueError(
            f"Checkpoint {path} has architectures {config.get('architectures')}. "
            f"Give a snapshot of {MODEL_ID} ({ARCHITECTURE})."
        )
    weights_path = os.path.join(path, CHECKPOINT_FILE)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Checkpoint {path} has no {CHECKPOINT_FILE}. Give a complete "
            f"snapshot of {MODEL_ID} at revision {REVISION}."
        )
    size = os.path.getsize(weights_path)
    if size != CHECKPOINT_FILE_BYTES:
        raise ValueError(
            f"{weights_path} has {size} bytes, but revision {REVISION} has "
            f"{CHECKPOINT_FILE_BYTES} bytes. Download the pinned revision again."
        )
    return path


def get_ernie_checkpoint() -> str:
    """Return a local directory that holds the pinned ERNIE 4.5 checkpoint.

    The function tries these sources in this sequence:

    1. The directory in ``ERNIE4_5_CHECKPOINT_DIR``.
    2. ``huggingface_hub.snapshot_download`` at the pinned revision.

    Each source must pass the same checks (architecture and file size).

    Example::

        checkpoint = get_ernie_checkpoint()
    """
    override = os.environ.get(CHECKPOINT_DIR_ENV)
    if override:
        return _validate_checkpoint_dir(override)

    errors = []
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL_ID,
            revision=REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
        return _validate_checkpoint_dir(path)
    except Exception as e:  # noqa: BLE001 - try the next source
        errors.append(f"snapshot_download: {e!r}")

    raise RuntimeError(
        f"Cannot get the {MODEL_ID} checkpoint at revision {REVISION}. "
        f"Set {CHECKPOINT_DIR_ENV} to a local snapshot. Errors: {errors}"
    )


def load_hf_config_dict(checkpoint: str) -> dict:
    """Return the raw ``config.json`` of the checkpoint as a dict."""
    with open(os.path.join(checkpoint, "config.json")) as f:
        return json.load(f)


def kv_heads_for_rank(rank: int, world_size: int, num_kv_heads: int) -> slice:
    """Return the KV heads that one TP rank holds, as a slice.

    This repeats the head sharding rule of ``Ernie4_5Attention``. When the TP
    size is not smaller than the KV head count, each rank holds one KV head,
    and ``world_size // num_kv_heads`` ranks hold the same head.
    """
    if world_size >= num_kv_heads:
        replicas = world_size // num_kv_heads
        start = rank // replicas
        return slice(start, start + 1)
    per_rank = num_kv_heads // world_size
    return slice(rank * per_rank, (rank + 1) * per_rank)
