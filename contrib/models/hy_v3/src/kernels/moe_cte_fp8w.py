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
# Entry point for bwmm_shard_on_I_fp8w.py (derived from nkilib 2.33).

"""Prefill routed experts with the FP8 weights fed to the matmul (no BF16 weight copy).

nkilib's ``shard_on_i_hybrid`` takes per-channel FP8 weight scales and, for
them, copies every FP8 weight tile to BF16 on the vector engine before a
BF16 x BF16 matmul; in a Hy3 4096-token prefill chunk that copy is ~60 of the
~80 ms the routed experts take. ``bwmm_shard_on_I_fp8w`` is that kernel with
the copy dropped: the matmuls take the FP8 tiles directly (BF16 activations x
FP8 weights) and the per-channel scales are applied after them as before. The
FP8 -> BF16 copy is exact and the matmul accumulates in fp32, so the output is
the same (bit-identical on one core in a test).
"""


import nki
import nki.language as nl
from nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from .bwmm_shard_on_I_fp8w import blockwise_mm_baseline_shard_intermediate_hybrid


@nki.jit
def moe_cte_fp8w(
    conditions,
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    gate_up_proj_scale,
    down_proj_scale,
    block_size: int,
    num_static_block: int | None = None,
    skip_token: bool = True,
):
    """shard_on_i_hybrid with per-channel FP8 weight scales and no BF16 weight copy.

    Same arguments as nkilib's hybrid kernel (BF16 hidden states, scales
    [E, 1, 2I] / [E, 1, H]); runs the copy in bwmm_shard_on_I_fp8w, whose
    matmuls take the FP8 weight tiles directly.
    """
    return blockwise_mm_baseline_shard_intermediate_hybrid(
        conditions=conditions,
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        num_static_block=num_static_block,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(skip_token, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )
