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
# Derived from nkilib.core.moe.moe_tkg.all_expert_impl (nkilib 2.33).

"""Small-batch MoE decode over the local experts that any token in the batch routed to.

``moe_tkg``'s all-expert mode computes every local expert for every token. At
a few tokens per step most local experts get no token at all. Here the caller
packs the local experts that at least one token routed to first, reorders the
affinity columns the same way, and passes the count; the kernel loops over
that many experts at runtime (``nl.fori_loop`` on a register). Each expert's
weights are read once for the whole batch and its output is scaled per token
by that token's affinity (0 for tokens that did not route to it).

Each iteration processes expert slot 0 and then shifts the expert list and
the affinity columns left by one, so every access pattern inside the loop is
static.
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
from nkilib.core.moe.moe_tkg.moe_tkg_utils import broadcast_all_expert_affinity
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
def moe_tkg_hits_batch(
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
    """MoE MLP for T tokens over the ``num_hits`` local experts they routed to.

    Args:
        hidden_input: [T, H] in HBM, T <= 128.
        expert_gate_up_weights: [E_L, H, 2, I].
        expert_down_weights: [E_L, I, H].
        expert_affinities: [T, E_L] fp32; column j is the affinity of every
            token for the expert in ``expert_index[0, j]`` (0 if not routed).
        expert_index: [1, E_L] int32 local expert ids, routed experts first.
        num_hits: [1, 1] int32, number of leading experts to run.
        expert_*_scale: FP8 row or static scales, as for ``moe_tkg``.

    Returns:
        [T, H] in HBM: for each token, the sum over the hit experts of
        affinity * expert(hidden).
    """
    quant_type, is_mx = _extract_quantization_type(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
    )
    assert not is_mx, "moe_tkg_hits_batch: MX weights are not supported"
    assert hidden_input.shape[0] <= 128, "moe_tkg_hits_batch: at most 128 tokens"
    assert num_hits.dtype == nl.int32, "moe_tkg_hits_batch: num_hits must be int32"
    assert expert_index.dtype == nl.int32, "moe_tkg_hits_batch: expert_index must be int32"
    assert expert_affinities.dtype == nl.float32, "moe_tkg_hits_batch: affinities must be fp32"

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
    _hit_expert_batch(params, num_hits, output)
    return output


def _hit_expert_batch(params: MLPParameters, num_hits: nl.NkiTensor, output: nl.NkiTensor):
    T, H = params.hidden_tensor.shape
    sbm = SbufManager(0, 200 * 1024, get_logger("hit_expert_batch"), use_auto_alloc=H >= 16 * 1024)
    sbm.open_scope()
    allocator = sbm.alloc_stack

    io_dtype = params.hidden_tensor.dtype
    expert_affinities = params.expert_params.expert_affinities
    dims = MLPTKGConstants.calculate_constants(params)
    E = dims.E
    num_I_tiles = div_ceil(dims.I, dims.I0)

    input_sb = allocator([dims.H0, T, dims.H1_shard], dtype=io_dtype, buffer=nl.sbuf, name="input_sb")
    input_norm_load(params.hidden_tensor, input_sb, params, dims, sbm=sbm)

    # Accumulator, zeroed so a batch with no local hit returns 0.
    output_temp = allocator((dims.H0, dims.H1_shard, T), dtype=io_dtype, buffer=nl.sbuf, name="out_acc")
    nisa.memset(output_temp, value=0.0)
    gate_up_output = allocator((dims.I0, num_I_tiles, T), dtype=io_dtype, buffer=nl.sbuf, name="gate_up_sb")
    down_sb = allocator((dims.H0, dims.H1_shard, T), dtype=io_dtype, buffer=nl.sbuf, name="down_sb")
    # Gate/up fp32 accumulators live outside the loop (NCC_IGCA108, see moe_tkg_hits).
    gate_up_fp32 = tuple(nl.ndarray((dims.I0, num_I_tiles, T), dtype=nl.float32, buffer=nl.sbuf) for _ in range(2))
    identity_sb = nl.shared_identity_matrix(T, dtype=io_dtype)

    # Expert list and affinity columns, shifted left by one per iteration.
    expert_idx = allocator((1, E), dtype=nl.int32, buffer=nl.sbuf, name="hit_idx")
    nisa.dma_copy(dst=expert_idx, src=params.expert_params.expert_index[0:1, 0:E])
    idx_shift = allocator((1, E), dtype=nl.int32, buffer=nl.sbuf, name="hit_idx_shift")
    aff_sb = allocator((T, E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="aff_sb")
    nisa.dma_copy(dst=aff_sb[:T, :E], src=expert_affinities[:T, :E])
    aff_shift = allocator((T, E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="aff_shift")

    n_sb = allocator((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="num_hits_sb")
    nisa.dma_copy(dst=n_sb, src=num_hits[0:1, 0:1])
    n_reg = nisa.register_alloc()
    nisa.register_load(src=n_sb, dst=n_reg)

    initial_gate = params.gate_proj_weights_tensor
    initial_up = params.up_proj_weights_tensor
    initial_down = params.down_proj_weights_tensor
    initial_quant = params.quant_params
    params.bias_params = MLPBiasParameters(None, None, None)
    dims.T = T

    def _body(_):
        sbm.open_scope()
        slot0 = expert_idx.ap(pattern=[[E, 1], [1, 1]], offset=0)
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
            use_dge=True,
            prealloc_gate_up_buffers=gate_up_fp32,
        )
        process_down_projection(
            hidden=gate_up_output,
            output=down_sb,
            params=params,
            dims=dims,
            gate_tile_info=gate_tile_info,
            sbm=sbm,
            use_dge=True,
        )

        # Column 0 of the affinities, broadcast over partitions: [pmax, T].
        aff_b = broadcast_all_expert_affinity(
            aff_sb, 1, 0, 0, T, io_dtype, identity_sb, dims, allocator, sbm.use_auto_alloc
        )
        for shard in range(dims.H1_shard):
            nisa.tensor_tensor(
                data1=down_sb[: dims.H0, shard, :T],
                data2=aff_b,
                op=nl.multiply,
                dst=down_sb[: dims.H0, shard, :T],
            )
        nisa.tensor_tensor(
            data1=down_sb[: dims.H0, : dims.H1_shard, :T],
            data2=output_temp[: dims.H0, : dims.H1_shard, :T],
            op=nl.add,
            dst=output_temp[: dims.H0, : dims.H1_shard, :T],
        )

        # Next expert moves to slot 0.
        nisa.tensor_copy(dst=idx_shift[0:1, 0 : E - 1], src=expert_idx[0:1, 1:E])
        nisa.tensor_copy(dst=expert_idx[0:1, 0 : E - 1], src=idx_shift[0:1, 0 : E - 1])
        nisa.tensor_copy(dst=aff_shift[:T, 0 : E - 1], src=aff_sb[:T, 1:E])
        nisa.tensor_copy(dst=aff_sb[:T, 0 : E - 1], src=aff_shift[:T, 0 : E - 1])
        sbm.close_scope()

    nl.fori_loop(0, n_reg, _body)

    dims.T = T
    transpose_store(output_temp, output, dims, params.output_dtype, sbm, 0)
    sbm.close_scope()
    return output
