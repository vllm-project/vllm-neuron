# SPDX-License-Identifier: Apache-2.0
from .argsort_unstable import argsort_unstable
from .spec_decode_correction import (
    correct_spec_decode_positions_and_slot_mapping as correct_spec_decode_positions_and_slot_mapping,
)  # noqa: F401
from .attention.attention_decode_mask import gen_attention_decode_mask
from .attention.attention_decode import attention_decode
from .attention.qkv import qkv_proj
from .attention.o_proj import o_proj
from .attention.attention_cte import flash_attention
from .attention.attention_segmented_cte import (
    segmented_attention,
    segmented_attention_cp,
)
from .attention.swa_fused import swa_fused_attention
from .collectives.all_gather_v import all_gather_v
from .collectives.all_to_all import all_to_all
from .collectives.all_to_all_v import all_to_all_v
from .collectives.reduce_scatter_v import reduce_scatter_v
from .embedding import embedding
from .expert_parallel import (
    calculate_local_expert_indices,
    validate_expert_parallelism_config,
    get_local_expert_affinities,
)
from .gpt_oss_mega_decode import (
    can_use_gpt_oss_mega_decode,
    gpt_oss_mega_decode,
)
from .mlp import mlp
from .moe.build_all2all_combine_metadata import build_all2all_combine_metadata
from .moe.build_all2all_dispatch_metadata import build_all2all_dispatch_metadata
from .moe.build_all_gatherv_metadata import build_all_gatherv_metadata
from .moe.hierarchical_all2all_combine_reduce import hierarchical_all2all_combine_reduce
from .moe.hierarchical_all2all_dispatch_permute import (
    hierarchical_all2all_dispatch_permute,
)
from .moe.moe_blockwise import build_blockwise_mapping
from .moe.moe_block_tkg import moe_block_tkg
from .moe.moe_tkg import moe_tkg
from .moe.moe_cte import moe_cte
from .moe.pack_tokens import pack_tokens
from .moe.permute_routed_tokens import permute_routed_tokens
from .moe.rmsnorm_mx_prefill import (
    rmsnorm_mx_prefill,
    slice_affinities_from_packed,
)
from .moe.rmsnorm_router_topk_tkg import rmsnorm_router_topk_tkg
from .rmsnorm_residual_prefill import rmsnorm_residual_prefill
from .moe.router import router
from .moe.topk_reduce import topk_reduce
from .rmsnorm_quant import rmsnorm_quant
from .sampling import sample
from .cumsum import cumsum
from .prompt_embeds import merge_prompt_embeds
from .process_groups import create_row_col_groups, get_group_slice_indices
from .merge_multimodal_embeddings import merge_multimodal_embeddings


# Alphabetical
__all__ = [
    "all_gather_v",
    "all_to_all",
    "all_to_all_v",
    "argsort_unstable",
    "attention_decode",
    "build_all2all_combine_metadata",
    "build_all2all_dispatch_metadata",
    "build_all_gatherv_metadata",
    "build_blockwise_mapping",
    "calculate_local_expert_indices",
    "can_use_gpt_oss_mega_decode",
    "create_row_col_groups",
    "cumsum",
    "embedding",
    "flash_attention",
    "gpt_oss_mega_decode",
    "gen_attention_decode_mask",
    "get_group_slice_indices",
    "get_local_expert_affinities",
    "hierarchical_all2all_combine_reduce",
    "hierarchical_all2all_dispatch_permute",
    "merge_multimodal_embeddings",
    "merge_prompt_embeds",
    "mlp",
    "moe_block_tkg",
    "moe_cte",
    "moe_tkg",
    "o_proj",
    "pack_tokens",
    "permute_routed_tokens",
    "qkv_proj",
    "reduce_scatter_v",
    "rmsnorm_mx_prefill",
    "rmsnorm_quant",
    "rmsnorm_residual_prefill",
    "rmsnorm_router_topk_tkg",
    "router",
    "sample",
    "segmented_attention",
    "segmented_attention_cp",
    "slice_affinities_from_packed",
    "swa_fused_attention",
    "topk_reduce",
    "validate_expert_parallelism_config",
]
