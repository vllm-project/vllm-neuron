# SPDX-License-Identifier: Apache-2.0
"""Helpers this model needs that vLLM Neuron 0.29.0.1.2.0 does not ship.

They are copied unchanged from a later vLLM Neuron release, so the model
runs on the 0.29 plugin as installed.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def hoist_rope_parameters(
    config_dict: dict, supported_rope_types: tuple[str, ...] = ("default",)
) -> dict:
    """Return ``config_dict`` with ``rope_parameters`` unpacked to the pre-v5 layout.

    transformers >= 5 keeps all RoPE settings under ``rope_parameters``.
    Configs that filter the HF dict to their dataclass fields (``rope_theta``,
    ``rope_scaling``) would otherwise drop them and fall back to the field
    defaults. ``rope_theta`` becomes a top-level key and the remaining
    settings (``rope_type``, ``factor``, ...) become ``rope_scaling`` when the
    type is not ``default``. Existing top-level keys win. Logs a warning if the
    RoPE type is not in ``supported_rope_types``.

    Example::

        >>> d = hoist_rope_parameters({"rope_parameters": {"rope_theta": 1e8}})
        >>> d["rope_theta"]
        100000000.0
        >>> p = {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0}
        >>> hoist_rope_parameters({"rope_parameters": p}, ("yarn",))["rope_scaling"]
        {'rope_type': 'yarn', 'factor': 4.0}
    """
    rope_parameters = config_dict.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        return config_dict
    out = dict(config_dict)
    if "rope_theta" in rope_parameters:
        out.setdefault("rope_theta", rope_parameters["rope_theta"])
    scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
    rope_type = scaling.get("rope_type", "default")
    if rope_type != "default" and out.get("rope_scaling") is None:
        out["rope_scaling"] = scaling
    if rope_type not in supported_rope_types:
        logger.warning(
            "rope_type=%r is not implemented for this model; using default RoPE",
            rope_type,
        )
    return out


def build_per_head_block_table(
    block_table_2d: torch.Tensor, kv_heads: int
) -> torch.Tensor:
    """Build a per-head 3D block table from a 2D base table.

    The kernel folds the 4D KV cache into a ``[num_blocks * kv_heads]`` pool.
    Each ``(batch, head)`` entry is the global pool index ``base * kv_heads + h``.
    Preserve ``-1`` (inactive) entries unchanged.

    Expects a 2D ``[B, num_blocks]`` table with ``-1`` as the inactive sentinel.
    Returns a ``[B, kv_heads, num_blocks]`` table.

    Example:
        >>> import torch
        >>> t = torch.tensor([[0, 1, -1]])
        >>> build_per_head_block_table(t, kv_heads=2)
        tensor([[[ 0,  2, -1],
                 [ 1,  3, -1]]])
    """
    assert block_table_2d.dim() == 2, (
        f"build_per_head_block_table expects a 2D [B, num_blocks] table, got "
        f"{block_table_2d.dim()}D {tuple(block_table_2d.shape)}"
    )
    base_3d = block_table_2d.unsqueeze(1)  # [B, 1, num_blocks]
    kv_h = torch.arange(
        kv_heads, dtype=block_table_2d.dtype, device=block_table_2d.device
    ).view(1, kv_heads, 1)
    shifted = base_3d * kv_heads + kv_h  # [B, kv_heads, num_blocks]
    return torch.where(base_3d == -1, base_3d, shifted).contiguous()
