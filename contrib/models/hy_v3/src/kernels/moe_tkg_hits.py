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
#
# Derived from nkilib.core.moe.moe_tkg.selective_expert_impl (nkilib 2.33).

"""Single-token MoE decode that runs only the experts routed to this rank.

nkilib's selective ``moe_tkg`` unrolls a loop over all ``K`` top-k slots, so a
rank computes ``K`` experts per token even when only one or two of them are
local: with expert parallelism the other slots point at a dummy expert with
affinity 0, whose weights are still read and multiplied. Here the caller packs
the slots of local experts first and passes their count; the kernel loops over
that many slots at runtime (``nl.fori_loop`` on a register), so a rank with no
routed expert does no expert work at all.

Each iteration processes slot 0 and then shifts the slot list left by one, so
every access pattern inside the loop is static.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.moe.moe_tkg.mlp_parameters import (
    MLPBiasParameters,
    MLPExpertParameters,
    MLPParameters,
    MLPQuantizationParameters,
)
from nkilib.core.moe.moe_tkg.mlp_tkg_constants import MLPTKGConstants
from nkilib.core.moe.moe_tkg.mlp_tkg_down_projection import process_down_projection
from nkilib.core.moe.moe_tkg.mlp_tkg_gate_up_projection import process_gate_up_projection
from nkilib.core.moe.moe_tkg.moe_tkg import _extract_quantization_type
from nkilib.core.moe.moe_tkg.moe_tkg_utils import (
    broadcast_token_affinity,
    gather_expert_affinities,
)
from nkilib.core.moe.moe_tkg.projection_utils import input_norm_load, transpose_store
from nkilib.core.moe.moe_tkg.selective_expert_impl import _select_quant_scales
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    GateUpDim,
    NormType,
    QuantizationType,
)
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


@nki.jit
def moe_tkg_hits(
    hidden_input: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    num_hits: nl.NkiTensor,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    expert_gate_up_input_scale: Optional[nl.NkiTensor] = None,
    expert_down_input_scale: Optional[nl.NkiTensor] = None,
    activation_fn: ActFnType = ActFnType.SiLU,
):
    """MoE MLP for one token over its ``num_hits`` local experts.

    Args:
        hidden_input: [1, H] in HBM.
        expert_gate_up_weights: [E_L, H, 2, I].
        expert_down_weights: [E_L, I, H].
        expert_affinities: [1, E_L] fp32, affinities of the local experts.
        expert_index: [1, K] int32 local expert ids, routed slots first.
        num_hits: [1, 1] int32, number of leading slots of ``expert_index`` to run.
        expert_*_scale: FP8 row or static scales, as for ``moe_tkg``.

    Returns:
        [1, H] in HBM: sum over the hit slots of affinity * expert(hidden).
    """
    quant_type, is_mx = _extract_quantization_type(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
    )
    assert not is_mx, "moe_tkg_hits: MX weights are not supported"
    assert hidden_input.shape[0] == 1, "moe_tkg_hits: one token only"
    assert num_hits.dtype == nl.int32, "moe_tkg_hits: num_hits must be int32"
    assert expert_index.dtype == nl.int32, "moe_tkg_hits: expert_index must be int32"

    params = MLPParameters(
        hidden_tensor=hidden_input,
        gate_proj_weights_tensor=expert_gate_up_weights,
        up_proj_weights_tensor=expert_gate_up_weights,
        down_proj_weights_tensor=expert_down_weights,
        activation_fn=activation_fn,
        normalization_type=NormType.NO_NORM,
        gate_w_scale=expert_gate_up_weights_scale,
        up_w_scale=expert_gate_up_weights_scale,
        down_w_scale=expert_down_weights_scale,
        gate_up_in_scale=expert_gate_up_input_scale,
        down_in_scale=expert_down_input_scale,
        output_dtype=hidden_input.dtype,
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
        shard_on_h_disabled=False,
        expert_params=MLPExpertParameters(
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            expert_affinities_eager=None,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        ),
        quantization_type=quant_type,
    )
    output = nl.ndarray(hidden_input.shape, dtype=hidden_input.dtype, buffer=nl.shared_hbm)
    _hit_expert_moe_tkg(params, num_hits, output)
    return output


def _hit_expert_moe_tkg(params: MLPParameters, num_hits: nl.NkiTensor, output: nl.NkiTensor):
    H = params.hidden_tensor.shape[-1]
    sbm = SbufManager(0, 200 * 1024, get_logger("hit_expert_moe_tkg"), use_auto_alloc=H >= 16 * 1024)
    sbm.open_scope()

    io_dtype = params.hidden_tensor.dtype
    expert_index_input = params.expert_params.expert_index
    expert_affinities = params.expert_params.expert_affinities
    dims = MLPTKGConstants.calculate_constants(params)
    K = dims.K

    input_sb = sbm.alloc_stack([dims.H0, 1, dims.H1_shard], dtype=io_dtype, buffer=nl.sbuf, name="input_sb")
    input_norm_load(params.hidden_tensor, input_sb, params, dims, sbm=sbm)

    # Accumulator, zeroed so a rank with no hit returns 0.
    output_temp = sbm.alloc_stack((dims.H0, dims.H1_shard, 1), dtype=io_dtype, buffer=nl.sbuf, name="out_acc")
    nisa.memset(output_temp, value=0.0)
    gate_up_output = sbm.alloc_stack(
        (dims.I0, div_ceil(dims.I, dims.I0), 1), dtype=io_dtype, buffer=nl.sbuf, name="gate_up_sb"
    )
    down_sb = sbm.alloc_stack((dims.H0, dims.H1_shard), dtype=io_dtype, buffer=nl.sbuf, name="down_sb")

    E, H_w, i_2, I = params.gate_proj_weights_tensor.shape

    # Slot lists, shifted left by one slot per iteration.
    expert_idx = sbm.alloc_stack((1, K), dtype=expert_index_input.dtype, buffer=nl.sbuf, name="slot_idx")
    nisa.dma_copy(dst=expert_idx, src=expert_index_input[0:1, 0:K])
    idx_shift = sbm.alloc_stack((1, K), dtype=expert_index_input.dtype, buffer=nl.sbuf, name="slot_idx_shift")

    expert_affinities_sb = sbm.alloc_stack(
        (dims._pmax, dims.E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="aff_sb"
    )
    nisa.memset(expert_affinities_sb, value=0.0)
    nisa.dma_copy(dst=expert_affinities_sb[0:1, 0 : dims.E], src=expert_affinities[0:1, 0 : dims.E])
    gathered_affinities_sb = gather_expert_affinities(expert_affinities_sb, expert_idx, dims, sbm)
    slot_aff = sbm.alloc_stack((dims._pmax, K), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="slot_aff")
    broadcast_token_affinity(slot_aff, gathered_affinities_sb, 0, dims, sbm)
    aff_shift = sbm.alloc_stack((dims._pmax, K), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="slot_aff_shift")

    n_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="num_hits_sb")
    nisa.dma_copy(dst=n_sb, src=num_hits[0:1, 0:1])
    n_reg = nisa.register_alloc()
    nisa.register_load(src=n_sb, dst=n_reg)

    initial_gate = params.gate_proj_weights_tensor
    initial_up = params.up_proj_weights_tensor
    initial_down = params.down_proj_weights_tensor
    initial_quant = params.quant_params
    params.bias_params = MLPBiasParameters(None, None, None)

    dims.T = 1

    def _body(_):
        sbm.open_scope()
        slot0 = expert_idx.ap(pattern=[[K, 1], [1, 1]], offset=0)
        params.gate_proj_weights_tensor = initial_gate.select(dim=0, index=slot0).select(
            dim=1, index=GateUpDim.GATE.value
        )
        params.up_proj_weights_tensor = initial_up.select(dim=0, index=slot0).select(dim=1, index=GateUpDim.UP.value)
        params.down_proj_weights_tensor = initial_down.select(dim=0, index=slot0)
        params.quant_params = _select_quant_scales(initial_quant, slot0)

        if initial_quant.quantization_type == QuantizationType.STATIC:
            # Per-expert scalar scales: load to partition 0, broadcast on chip (as selective_expert_impl).
            gate_w_dq = nl.ndarray((dims._pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            up_w_dq = nl.ndarray((dims._pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            down_w_dq = nl.ndarray((dims._pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gate_w_dq[0:1, 0:1], src=params.quant_params.gate_w_scale.slice(dim=0, start=0, end=1))
            stream_shuffle_broadcast(gate_w_dq, gate_w_dq)
            nisa.dma_copy(dst=up_w_dq[0:1, 0:1], src=params.quant_params.up_w_scale.slice(dim=0, start=0, end=1))
            stream_shuffle_broadcast(up_w_dq, up_w_dq)
            nisa.dma_copy(dst=down_w_dq[0:1, 0:1], src=params.quant_params.down_w_scale.slice(dim=0, start=0, end=1))
            stream_shuffle_broadcast(down_w_dq, down_w_dq)
            params.quant_params = MLPQuantizationParameters(
                quantization_type=QuantizationType.STATIC,
                gate_w_scale=gate_w_dq,
                up_w_scale=up_w_dq,
                down_w_scale=down_w_dq,
                gate_up_in_scale=None,
                down_in_scale=None,
                clipping_bound=params.quant_params.clipping_bound,
            )

        gate_tile_info = process_gate_up_projection(
            hidden=input_sb,
            output=gate_up_output,
            params=params,
            dims=dims,
            sbm=sbm,
            share_memory_scope=True,
        )
        process_down_projection(
            hidden=gate_up_output,
            output=down_sb,
            params=params,
            dims=dims,
            gate_tile_info=gate_tile_info,
            sbm=sbm,
        )
        nisa.tensor_scalar(dst=down_sb, data=down_sb, op0=nl.multiply, operand0=slot_aff[:, 0])
        nisa.tensor_tensor(
            dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, 0],
            data1=output_temp[0 : dims.H0, 0 : dims.H1_shard, 0],
            data2=down_sb,
            op=nl.add,
        )

        # Next slot moves to position 0.
        nisa.tensor_copy(dst=idx_shift[0:1, 0 : K - 1], src=expert_idx[0:1, 1:K])
        nisa.tensor_copy(dst=expert_idx[0:1, 0 : K - 1], src=idx_shift[0:1, 0 : K - 1])
        nisa.tensor_copy(dst=aff_shift[:, 0 : K - 1], src=slot_aff[:, 1:K])
        nisa.tensor_copy(dst=slot_aff[:, 0 : K - 1], src=aff_shift[:, 0 : K - 1])
        sbm.close_scope()

    nl.fori_loop(0, n_reg, _body)

    dims.T = 1
    transpose_store(output_temp, output, dims, params.output_dtype, sbm, 0)
    sbm.close_scope()
    return output
