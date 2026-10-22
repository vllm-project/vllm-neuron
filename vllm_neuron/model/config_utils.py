# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the per-model config dataclasses."""

import logging

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
