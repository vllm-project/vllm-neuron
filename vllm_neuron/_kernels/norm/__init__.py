# SPDX-License-Identifier: Apache-2.0
"""In-tree normalization NKI kernel stage helpers.

The sequence-parallel composers import the RMSNorm + swizzle stages from
``qwen3_32b_mxfp8_rmsnorm_quant_tkg`` directly; there is no standalone entry point.
"""
