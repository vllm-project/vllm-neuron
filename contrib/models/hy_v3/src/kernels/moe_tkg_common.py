# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers of the moe_tkg_hits kernels.

Local copies of two private nkilib 2.33 helpers (moe_tkg._extract_quantization_type
without its MX branches, selective_expert_impl._select_quant_scales), so the
kernels only import public nkilib names.
"""

from typing import Optional

import nki.language as nl
from nkilib.core.moe.moe_tkg.mlp_parameters import MLPQuantizationParameters
from nkilib.core.moe.moe_tkg.moe_tkg_utils import reshape_scale_for_mlp
from nkilib.core.utils.common_types import GateUpDim, QuantizationType

_MX_DTYPES = (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4)


def quantization_type(
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    expert_gate_up_input_scale: Optional[nl.NkiTensor],
    expert_down_input_scale: Optional[nl.NkiTensor],
) -> QuantizationType:
    """STATIC with input scales, ROW with weight scales only, else NONE (MX unsupported)."""
    assert expert_gate_up_weights.dtype not in _MX_DTYPES, "MX weights are not supported"
    if expert_gate_up_input_scale is not None and expert_down_input_scale is not None:
        return QuantizationType.STATIC
    if expert_gate_up_weights_scale is not None and expert_down_weights_scale is not None:
        return QuantizationType.ROW
    return QuantizationType.NONE


def select_quant_scales(
    quant_params: MLPQuantizationParameters, expert_id_offset: nl.NkiTensor
) -> MLPQuantizationParameters:
    """The scales of one expert (``expert_id_offset``), reshaped for the MLP projections."""

    def pick(scale, gate_up_dim=None):
        if scale is None:
            return None
        scale = scale.select(dim=0, index=expert_id_offset)
        if gate_up_dim is not None:
            scale = scale.select(dim=0, index=gate_up_dim.value)
        return reshape_scale_for_mlp(scale)

    return MLPQuantizationParameters(
        quantization_type=quant_params.quantization_type,
        gate_w_scale=pick(quant_params.gate_w_scale, GateUpDim.GATE),
        up_w_scale=pick(quant_params.up_w_scale, GateUpDim.UP),
        down_w_scale=pick(quant_params.down_w_scale),
        gate_up_in_scale=pick(quant_params.gate_up_in_scale),
        down_in_scale=pick(quant_params.down_in_scale),
        clipping_bound=quant_params.clipping_bound,
    )
