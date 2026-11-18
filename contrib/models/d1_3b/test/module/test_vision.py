# SPDX-License-Identifier: Apache-2.0
"""CPU module test: d1-3B vision tower + projector vs. HF SigLIP2 + LFM2-VL projector.

Random weights, a small config, fp32, so the comparison is exact up to float
rounding. Checks the parts with data-dependent shapes that the host prepares
(position-embedding resize, key mask, pixel-unshuffle gather) and the cache write.
Needs only torch and transformers (no Neuron, no vLLM):

    pytest contrib/models/d1_3b/test/module/test_vision.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Lfm2VlConfig, Siglip2VisionConfig
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlMultiModalProjector,
)
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

# Load src/vision.py directly: src/__init__ imports vLLM, this test must not.
_spec = importlib.util.spec_from_file_location(
    "d1_vision", Path(__file__).resolve().parents[2] / "src" / "vision.py"
)
vision = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vision  # dataclasses resolve annotations through it
_spec.loader.exec_module(vision)

# (h, w) patch grids: a full 512 px tile, a non-square thumbnail, a small tile.
TILE_SHAPES = [(32, 32), (20, 28), (8, 12)]


def _configs():
    vcfg = Siglip2VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        patch_size=16,
        num_patches=256,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attn_implementation="eager",
    )
    cfg = Lfm2VlConfig(
        vision_config=vcfg.to_dict(),
        text_config={"model_type": "lfm2", "hidden_size": 48},
        projector_hidden_size=32,
        projector_bias=True,
        projector_hidden_act="gelu",
        projector_use_layernorm=False,
        downsample_factor=2,
    )
    ours = vision.D1VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=16,
        num_patches=256,
        layer_norm_eps=1e-6,
        downsample_factor=2,
        projector_hidden_size=32,
        projector_bias=True,
        text_hidden_size=48,
        dtype=torch.float32,
    )
    return vcfg, cfg, ours


@torch.no_grad()
def test_vision_tower_matches_hf():
    torch.manual_seed(0)
    vcfg, cfg, ours = _configs()
    hf_tower = Siglip2VisionModel(vcfg).eval()
    hf_proj = Lfm2VlMultiModalProjector(cfg).eval()
    for p in list(hf_tower.parameters()) + list(hf_proj.parameters()):
        p.normal_(0, 0.05)

    # Checkpoint-style names, as in d1-3B's model.safetensors (which keeps the
    # "vision_model." level that the transformers 5 module tree drops).
    ckpt = {
        f"model.vision_tower.vision_model.{k.removeprefix('vision_model.')}": v
        for k, v in hf_tower.state_dict().items()
    }
    ckpt |= {
        f"model.multi_modal_projector.{k}": v for k, v in hf_proj.state_dict().items()
    }

    tower = vision.D1VisionTower(ours)
    mapping = vision.vision_mappings("visual") | vision.vision_layer_mappings(
        "visual", ours.num_hidden_layers
    )
    params = dict(tower.named_parameters())
    assert {f"visual.{n}" for n in params} == set(mapping), (
        "mapping must cover every parameter"
    )
    for name, key in mapping.items():
        params[name.removeprefix("visual.")].copy_(ckpt[key])

    # Inputs as the processor gives them: patches zero-padded to 1024 per tile.
    T, n_tiles = vision.PATCHES_PER_BLOCK, len(TILE_SHAPES)
    pixel_values = torch.zeros(n_tiles, T, ours.patch_dim)
    mask = torch.zeros(n_tiles, T, dtype=torch.long)
    for i, (h, w) in enumerate(TILE_SHAPES):
        pixel_values[i, : h * w] = torch.randn(h * w, ours.patch_dim)
        mask[i, : h * w] = 1
    spatial_shapes = torch.tensor(TILE_SHAPES)

    # HF reference: Lfm2VlModel.get_image_features.
    hidden = hf_tower(
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        pixel_attention_mask=mask,
    ).last_hidden_state
    want = []
    for i, (h, w) in enumerate(TILE_SHAPES):
        feat = hidden[i, : h * w].reshape(1, h, w, -1)
        want.append(hf_proj(feat).reshape(-1, ours.text_hidden_size))

    # Ours: host packing (one tile per block, one padding block) + device graph.
    num_blocks = n_tiles + 1
    host = vision.pack_tiles(
        ours, pixel_values, spatial_shapes, tower.position_embedding_weight, num_blocks
    )
    cache = torch.full(
        (num_blocks + 2, ours.tokens_per_block, ours.text_hidden_size), 7.0
    )
    write_ids = torch.tensor(
        [2, 0, 4, 1]
    )  # arbitrary cache blocks; last = padding block
    tower(
        host["pixel_values"],
        host["pos_embeds"],
        host["key_bias"],
        host["unshuffle_idx"],
        cache,
        write_ids,
    )

    for i, (h, w) in enumerate(TILE_SHAPES):
        n = vision.merged_tokens(ours, h, w)
        got = cache[write_ids[i], :n]
        torch.testing.assert_close(got, want[i], rtol=1e-4, atol=1e-5)
    assert torch.all(cache[3] == 7.0), "an unused cache block was written"


def test_unshuffle_index_order():
    # 4x4 grid -> 2x2 tokens; token 0 takes patches 0, 1, 4, 5 (row-major 2x2).
    idx = vision.unshuffle_index(4, 4, 2, tokens_per_block=8)
    assert idx[:4].tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
    assert idx[4:].abs().sum() == 0
