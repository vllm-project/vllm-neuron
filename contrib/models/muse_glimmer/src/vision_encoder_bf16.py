# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer vision path for vLLM Neuron: the validated fullgraph vision graph.

Same code that compiled and ran on trn3 with fullgraph=True in the bare bring-up
(repo muse_compile.py, rungs R5/V5, and the one-core vision profile):
  - HF transformers vision modules (MuseGlimmerVisionModel, adapter, projection,
    perception norm), unchanged weights and math;
  - grid bookkeeping precomputed on the host for one fixed grid (popped from kwargs by
    the HF tower);
  - per-segment attention replaced by one SDPA call under a block-diagonal mask.
The only addition: the output is written into vLLM's on-device encoder cache at
`write_block_ids` (the runner's contract), instead of being returned.

Fixed sizes (one compiled vision graph each) are in shapes.py. The graph is picked by
patch count; embeddings fill the item's cache blocks in order.
"""

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.muse_glimmer import modeling_muse_glimmer as mg
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_window_index,
)

from .shapes import GRIDS

META_KEYS = (
    "cu_seqlens",
    "window_index",
    "cu_window_seqlens",
    "position_ids",
    "interp_indices",
    "interp_weights",
    "pixel_shuffle_index",
)


def precompute_vision_kwargs(tower, grid_thw):
    """Grid-dependent tensors for one grid, computed on the host (muse_compile.py)."""
    pe = tower.patch_embedder
    window_index, cu_window_seqlens = get_vision_window_index(
        grid_thw,
        spatial_merge_size=tower.spatial_merge_size,
        window_size=tower.window_size,
        patch_size=tower.patch_size,
    )
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=pe.num_grid_per_side,
        mode=pe.interpolation_mode,
        align_corners=pe.interpolation_align_corners,
        spatial_merge_size=1,
        padding=pe.interpolation_padding,
    )
    return dict(
        cu_seqlens=get_vision_cu_seqlens(grid_thw),
        window_index=window_index,
        cu_window_seqlens=torch.as_tensor(cu_window_seqlens, dtype=torch.int32),
        position_ids=get_vision_position_ids(
            grid_thw, spatial_merge_size=tower.spatial_merge_size
        ),
        interp_indices=interp_indices,
        interp_weights=interp_weights,
        pixel_shuffle_index=mg.get_vision_pixel_shuffle_index(
            grid_thw, tower.merge_size
        ),
    )


def _block_diag_mask(cu_seqlens, seq_len, device):
    pos = torch.arange(seq_len, device=device)
    seg = (pos[:, None] >= cu_seqlens[None, 1:].to(device)).sum(-1)
    return seg[:, None] == seg[None, :]


def _vision_attention_forward(
    self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs
):
    seq_length = hidden_states.shape[0]
    q = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    k = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    v = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    cos, sin = position_embeddings
    q, k = mg.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(2, 1) for t in (q, k, v))
    mask = _block_diag_mask(cu_seqlens, seq_length, hidden_states.device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
    out = out.transpose(1, 2).reshape(seq_length, -1).contiguous()
    return self.proj(out)


class MuseGlimmerVisionEncoder(nn.Module):
    """HF vision tower + adapter + projection + perception norm; writes into the encoder cache."""

    def __init__(self, hf_config):
        super().__init__()
        vcfg = hf_config.vision_config
        vcfg._attn_implementation = "sdpa"
        # The bring-up (transformers 5.20.2) loaded the vision RoPE as {"rope_type": "axial",
        # "rope_theta": 10000}. The vLLM image's loader (transformers 5.17.0 / vLLM) keeps the
        # checkpoint's "default", which MuseGlimmerVisionRotaryEmbedding rejects. Same theta.
        vcfg.rope_parameters = {
            **dict(vcfg.rope_parameters or {}),
            "rope_type": "axial",
        }
        # One SDPA call per attention layer, the validated fullgraph form. Patched on the
        # class (a per-instance method is not saved by the torch.compile cache), here so
        # that only the process that builds the vision encoder (a worker) changes it.
        mg.MuseGlimmerVisionAttention.forward = _vision_attention_forward
        self.tower = mg.MuseGlimmerVisionModel(vcfg)
        self.adapter = mg.MuseGlimmerVisionAdapter(hf_config)
        self.projection = nn.Linear(
            hf_config.projector_hidden_size,
            hf_config.text_config.hidden_size,
            bias=False,
        )
        self.perception_emb_norm = mg.MuseGlimmerRMSNorm(
            eps=hf_config.text_config.rms_norm_eps, with_scale=False
        )
        self.merge_factor = vcfg.merge_size**2

    def forward(
        self,
        pixel_values,
        cu_seqlens,
        window_index,
        cu_window_seqlens,
        position_ids,
        interp_indices,
        interp_weights,
        pixel_shuffle_index,
        encoder_cache_buffer,
        write_block_ids,
    ):
        meta = dict(
            cu_seqlens=cu_seqlens,
            window_index=window_index,
            cu_window_seqlens=cu_window_seqlens,
            position_ids=position_ids,
            interp_indices=interp_indices,
            interp_weights=interp_weights,
            pixel_shuffle_index=pixel_shuffle_index,
        )
        grid = torch.tensor(
            [GRIDS[pixel_values.shape[0]]], device=pixel_values.device
        )  # static per graph
        feats = self.tower(
            pixel_values=pixel_values, grid_thw=grid, **meta
        ).last_hidden_state
        emb = self.perception_emb_norm(
            self.projection(self.adapter(feats))
        )  # [n, hidden]
        # embeddings fill the item's blocks in order (dense: full blocks, then the remainder);
        # trailing write_block_ids are the scratch pad
        n, d = emb.shape
        block = encoder_cache_buffer.shape[1]
        t = torch.arange(n, device=emb.device)
        rows = write_block_ids[t // block] * block + t % block
        encoder_cache_buffer.view(-1, d).index_copy_(
            0, rows, emb.to(encoder_cache_buffer.dtype)
        )
        return encoder_cache_buffer
