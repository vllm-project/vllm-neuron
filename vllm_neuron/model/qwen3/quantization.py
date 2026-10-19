# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MXFP8 checkpoint quantization parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Canonical MX group size (E8M0 shared exponent per 32 elements). Imported by
# weight_loaders_mxfp8 as MX_GROUP_SIZE so the contract is single-sourced.
MXFP8_GROUP_SIZE = 32


class QuantScheme(str, Enum):
    NONE = "none"
    MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE = "mxfp8_dynamic_per_token_channelwise"


@dataclass(frozen=True)
class QuantizationSpec:
    """Producer-agnostic view of the checkpoint quantization contract."""

    linear_scheme: QuantScheme

    def get_scheme(self, layer_index: int | None, prefix: str) -> QuantScheme:
        del layer_index, prefix
        return self.linear_scheme

    @classmethod
    def from_hf_quantization_config(
        cls, quantization_config: dict[str, Any] | None
    ) -> "QuantizationSpec | None":
        if not quantization_config:
            return None
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "Expected quantization_config to be a dict, got "
                f"{type(quantization_config).__name__}."
            )

        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if not quant_method:
            return None
        if quant_method == "compressed-tensors":
            return _parse_compressed_tensors_mxfp8(quantization_config)

        raise ValueError(
            f"Unsupported quantization_config.quant_method={quant_method!r}. "
            "Qwen3-32B dense currently supports compressed-tensors "
            "static-weight / dynamic-activation MXFP8 checkpoints."
        )


def _parse_compressed_tensors_mxfp8(
    quantization_config: dict[str, Any],
) -> QuantizationSpec:
    fmt = str(quantization_config.get("format", "")).lower()
    # "mxfp8-quantized" is the canonical MX label; "float-quantized" is also
    # accepted because some llm-compressor MX exporters stamp the generic FP8
    # label even for group-32 MX weights. The group_size==32 + strategy==group
    # checks below still reject any genuinely non-MX float-quantized checkpoint.
    supported_formats = {"float-quantized", "mxfp8-quantized"}
    if fmt not in supported_formats:
        raise ValueError(
            "Qwen3-32B dense MXFP8 support requires "
            "quantization_config.format in "
            f"{sorted(supported_formats)!r}, got "
            f"{fmt!r}."
        )

    config_groups = quantization_config.get("config_groups")
    if not isinstance(config_groups, dict) or not config_groups:
        raise ValueError(
            "Qwen3-32B dense MXFP8 support requires a non-empty config_groups mapping."
        )

    for group_name, group in config_groups.items():
        if not isinstance(group, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires each config_groups "
                f"entry to be a dict, got {type(group).__name__} for "
                f"{group_name!r}."
            )

        weights = group.get("weights")
        if not isinstance(weights, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight quantization "
                f"config in {group_name!r}."
            )

        input_activations = group.get("input_activations")
        if not isinstance(input_activations, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires input activation "
                f"quantization config in {group_name!r}."
            )

        if group.get("output_activations") is not None:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support does not accept output_activations "
                f"in {group_name!r}."
            )

        if (
            str(weights.get("type", "")).lower() != "float"
            or int(weights.get("num_bits", 0)) != 8
        ):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires float8 weights in "
                f"{group_name!r}."
            )
        if bool(weights.get("dynamic")):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires static weight "
                f"quantization in {group_name!r}."
            )
        if str(weights.get("strategy", "")).lower() != "group":
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires groupwise weights in "
                f"{group_name!r}."
            )
        if int(weights.get("group_size", 0)) != MXFP8_GROUP_SIZE:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight group_size=32 in "
                f"{group_name!r}."
            )
        scale_dtype = weights.get("scale_dtype")
        if scale_dtype is not None and str(scale_dtype).lower() not in {
            "uint8",
            "u8",
            "torch.uint8",
        }:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight scale_dtype=torch.uint8 "
                f"(E8M0 storage) when declared, got {scale_dtype!r} in {group_name!r}."
            )

        if (
            str(input_activations.get("type", "")).lower() != "float"
            or int(input_activations.get("num_bits", 0)) != 8
        ):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires float8 input "
                f"activations in {group_name!r}."
            )
        if not bool(input_activations.get("dynamic")):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires dynamic input "
                f"activations in {group_name!r}."
            )
        if str(input_activations.get("strategy", "")).lower() != "group":
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires groupwise input "
                f"activations in {group_name!r}."
            )
        if int(input_activations.get("group_size", 0)) != MXFP8_GROUP_SIZE:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires input group_size=32 in "
                f"{group_name!r}."
            )

        targets = group.get("targets") or []
        if not any(str(target) == "Linear" for target in targets):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support currently requires Linear "
                f"targets in {group_name!r}."
            )

    return QuantizationSpec(
        linear_scheme=QuantScheme.MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE
    )
