# SPDX-License-Identifier: Apache-2.0
"""SigLIP2 NaFlex vision tower + LFM2-VL projector for d1-3B on Neuron.

Pipeline for one image (HF ``Lfm2VlModel.get_image_features``):

  - The processor splits the image into 512 px tiles plus a thumbnail (or keeps it
    as one tile when small). Each tile is up to 1024 patches of 16x16x3 = 768
    values, with its own (h, w) patch grid (``spatial_shapes``).
  - SigLIP2: linear patch embed + position embeddings (the 16x16 learned grid
    resized to (h, w) with antialiased bilinear interpolation), 27 pre-LN encoder
    layers (bidirectional attention over the tile's valid patches), post LayerNorm.
  - Projector: 2x2 pixel-unshuffle (4 * 1152 = 4608 channels) -> linear -> GELU ->
    linear -> 2048 (the text hidden size). A tile gives (h/2) * (w/2) <= 256 tokens.

Neuron layout: one tile per encoder block (``encoder_attention_block_size=1024``
raw patches), and the merge factor 4 makes one encoder-cache block 256 tokens, so
each tile writes exactly one cache block. The host does the per-tile work that has
data-dependent shapes (position-embedding resize, key mask, unshuffle gather
index); the device graph has fixed shapes per bucket (number of blocks).

The tower is replicated on every rank (no collectives): every rank encodes every
tile and writes the same values into its own copy of the encoder cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PATCHES_PER_BLOCK = 1024  # max_num_patches of the d1-3B processor
MASK_VALUE = -1e9


# GELU written out: libtorch_neuronx_lite replaces F.gelu with a C extension that
# torch.compile cannot trace (the in-tree Qwen3-VL encoder does the same).
def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


@dataclass
class D1VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    num_patches: int = 256  # learned position grid: 16 x 16
    layer_norm_eps: float = 1e-6
    downsample_factor: int = 2
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    text_hidden_size: int = 2048
    dtype: torch.dtype = torch.bfloat16

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def merge_factor(self) -> int:
        return self.downsample_factor**2

    @property
    def tokens_per_block(self) -> int:
        return PATCHES_PER_BLOCK // self.merge_factor

    @classmethod
    def from_hf(cls, hf_config) -> D1VisionConfig:
        d = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
        v = d["vision_config"]
        if v.get("model_type") != "siglip2_vision_model":
            raise NotImplementedError(f"vision model_type {v.get('model_type')}")
        if v.get("vision_use_head"):
            raise NotImplementedError("vision_use_head=True is not implemented")
        if v.get("hidden_act") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"vision hidden_act {v.get('hidden_act')}")
        if d.get("projector_use_layernorm"):
            raise NotImplementedError("projector_use_layernorm=True is not implemented")
        if d.get("projector_hidden_act", "gelu") != "gelu":
            raise NotImplementedError(
                f"projector_hidden_act {d['projector_hidden_act']}"
            )
        if d.get("max_num_patches", PATCHES_PER_BLOCK) > PATCHES_PER_BLOCK:
            raise NotImplementedError(f"max_num_patches {d['max_num_patches']} > 1024")
        return cls(
            hidden_size=v["hidden_size"],
            intermediate_size=v["intermediate_size"],
            num_hidden_layers=v["num_hidden_layers"],
            num_attention_heads=v["num_attention_heads"],
            num_channels=v.get("num_channels", 3),
            patch_size=v["patch_size"],
            num_patches=v["num_patches"],
            layer_norm_eps=v.get("layer_norm_eps", 1e-6),
            downsample_factor=d.get("downsample_factor", 2),
            projector_hidden_size=d["projector_hidden_size"],
            projector_bias=d.get("projector_bias", True),
            text_hidden_size=d["text_config"]["hidden_size"],
        )


def _linear_params(module: nn.Module, name: str, out_f: int, in_f: int, bias: bool, dt):
    # Checkpoint layout ([out, in] weight, [out] bias), loaded as-is (identity loader).
    setattr(module, f"{name}_weight", nn.Parameter(torch.empty(out_f, in_f, dtype=dt)))
    if bias:
        setattr(module, f"{name}_bias", nn.Parameter(torch.empty(out_f, dtype=dt)))


class _LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dt):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dt))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dt))
        self.eps = eps

    def forward(self, x):
        y = F.layer_norm(
            x.float(), (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps
        )
        return y.to(x.dtype)


class D1VisionLayer(nn.Module):
    """SigLIP2 encoder layer: x += attn(ln1(x)); x += mlp(ln2(x))."""

    def __init__(self, cfg: D1VisionConfig):
        super().__init__()
        H, I, dt = cfg.hidden_size, cfg.intermediate_size, cfg.dtype
        self.num_heads = cfg.num_attention_heads
        self.head_dim = H // self.num_heads
        self.scale = self.head_dim**-0.5
        self.layer_norm1 = _LayerNorm(H, cfg.layer_norm_eps, dt)
        self.layer_norm2 = _LayerNorm(H, cfg.layer_norm_eps, dt)
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _linear_params(self, n, H, H, True, dt)
        _linear_params(self, "fc1", I, H, True, dt)
        _linear_params(self, "fc2", H, I, True, dt)

    def _attn(self, x, key_bias):
        B, T, H = x.shape
        nh, hd = self.num_heads, self.head_dim

        def heads(t):
            return t.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]

        q = heads(F.linear(x, self.q_proj_weight, self.q_proj_bias))
        k = heads(F.linear(x, self.k_proj_weight, self.k_proj_bias))
        v = heads(F.linear(x, self.v_proj_weight, self.v_proj_bias))
        # Explicit matmuls with fp32 scores and softmax (HF eager path numerics).
        s = torch.matmul(q, k.transpose(-1, -2)).float() * self.scale + key_bias
        p = torch.softmax(s, dim=-1).to(v.dtype)
        o = torch.matmul(p, v).transpose(1, 2).reshape(B, T, H)
        return F.linear(o, self.out_proj_weight, self.out_proj_bias)

    def forward(self, x, key_bias):
        x = x + self._attn(self.layer_norm1(x), key_bias)
        h = F.linear(self.layer_norm2(x), self.fc1_weight, self.fc1_bias)
        h = gelu_tanh(h)
        return x + F.linear(h, self.fc2_weight, self.fc2_bias)


class D1VisionTower(nn.Module):
    """Encoder graph: packed tile blocks in, projected tokens written to the cache.

    ``forward`` inputs (``B`` = blocks in the bucket, one tile per block):
        pixel_values: [B, 1024, patch_dim] bf16, tile patches zero-padded.
        pos_embeds: [B, 1024, hidden] bf16, resized position embeddings (host).
        key_bias: [B, 1, 1, 1024] fp32, 0 for valid patches, -1e9 for padding.
        unshuffle_idx: [B, 256 * 4] int32, flat patch index per (token, 2x2 slot),
            already offset by ``block * 1024`` (padding tokens point at row 0).
        encoder_cache_buffer: [num_cache_blocks, 256, 2048] bf16 (aliased output).
        write_block_ids: [B] int64, cache block for each encoder block.
    """

    def __init__(self, cfg: D1VisionConfig):
        super().__init__()
        self.cfg = cfg
        H, dt = cfg.hidden_size, cfg.dtype
        _linear_params(self, "patch_embedding", H, cfg.patch_dim, True, dt)
        # Learned grid; read on the host to build per-tile position embeddings.
        self.position_embedding_weight = nn.Parameter(
            torch.empty(cfg.num_patches, H, dtype=dt)
        )
        self.layers = nn.ModuleList(
            [D1VisionLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.post_layernorm = _LayerNorm(H, cfg.layer_norm_eps, dt)
        in_ch = H * cfg.merge_factor
        _linear_params(
            self, "linear_1", cfg.projector_hidden_size, in_ch, cfg.projector_bias, dt
        )
        _linear_params(
            self,
            "linear_2",
            cfg.text_hidden_size,
            cfg.projector_hidden_size,
            cfg.projector_bias,
            dt,
        )

    def forward(
        self,
        pixel_values,
        pos_embeds,
        key_bias,
        unshuffle_idx,
        encoder_cache_buffer,
        write_block_ids,
    ):
        cfg = self.cfg
        B, T, _ = pixel_values.shape
        x = F.linear(
            pixel_values.to(cfg.dtype),
            self.patch_embedding_weight,
            self.patch_embedding_bias,
        )
        x = x + pos_embeds.to(cfg.dtype)
        for layer in self.layers:
            x = layer(x, key_bias)
        x = self.post_layernorm(x)  # [B, T, H]

        # 2x2 pixel-unshuffle as one gather: [B*256, 4*H].
        flat = x.reshape(B * T, cfg.hidden_size)
        g = torch.index_select(flat, 0, unshuffle_idx.reshape(-1).to(torch.int64))
        g = g.reshape(B * cfg.tokens_per_block, cfg.merge_factor * cfg.hidden_size)
        h = F.linear(g, self.linear_1_weight, getattr(self, "linear_1_bias", None))
        h = gelu_erf(h)  # projector_hidden_act "gelu" (erf)
        h = F.linear(h, self.linear_2_weight, getattr(self, "linear_2_bias", None))
        out = h.reshape(B, cfg.tokens_per_block, cfg.text_hidden_size).to(
            encoder_cache_buffer.dtype
        )
        encoder_cache_buffer.index_put_((write_block_ids,), out)
        return encoder_cache_buffer


# =============================================================================
# Host-side preparation
# =============================================================================
def resize_position_embeddings(table: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """HF ``Siglip2VisionEmbeddings.resize_positional_embeddings`` for one tile.

    Args:
        table: [num_patches, hidden] learned grid (CPU).
        h, w: The tile's patch grid.

    Returns:
        [h * w, hidden] fp32.
    """
    side = math.isqrt(table.shape[0])
    grid = table.float().reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(
        grid, size=(h, w), mode="bilinear", align_corners=False, antialias=True
    )
    return out.reshape(table.shape[1], h * w).transpose(0, 1)


def unshuffle_index(h: int, w: int, factor: int, tokens_per_block: int) -> torch.Tensor:
    """Patch index for each (merged token, 2x2 slot) of one tile, row-major.

    Matches HF ``Lfm2VlMultiModalProjector.pixel_unshuffle`` (slot order
    (0,0), (0,1), (1,0), (1,1)). Padding tokens point at patch 0.
    """
    if h % factor or w % factor:
        raise ValueError(f"tile grid ({h}, {w}) not divisible by {factor}")
    ho, wo = h // factor, w // factor
    rr, cc = torch.meshgrid(torch.arange(ho), torch.arange(wo), indexing="ij")
    dh, dw = torch.meshgrid(torch.arange(factor), torch.arange(factor), indexing="ij")
    idx = (rr.reshape(-1, 1) * factor + dh.reshape(1, -1)) * w + (
        cc.reshape(-1, 1) * factor + dw.reshape(1, -1)
    )  # [ho*wo, factor^2]
    full = torch.zeros(tokens_per_block, factor * factor, dtype=torch.int64)
    full[: ho * wo] = idx
    return full


def pack_tiles(
    cfg: D1VisionConfig,
    pixel_values: torch.Tensor,
    spatial_shapes: torch.Tensor,
    pos_table: torch.Tensor,
    num_blocks: int,
) -> dict[str, torch.Tensor]:
    """Host tensors for ``D1VisionTower.forward``: one tile per block, padded."""
    shapes = [tuple(int(v) for v in s) for s in spatial_shapes.tolist()]
    if len(shapes) > num_blocks:
        raise ValueError(f"{len(shapes)} tiles do not fit {num_blocks} blocks")
    T, H = PATCHES_PER_BLOCK, cfg.hidden_size
    pix = torch.zeros(num_blocks, T, cfg.patch_dim, dtype=cfg.dtype)
    pos = torch.zeros(num_blocks, T, H, dtype=cfg.dtype)
    bias = torch.full((num_blocks, 1, 1, T), MASK_VALUE, dtype=torch.float32)
    bias[:, :, :, 0] = 0.0  # padding blocks: keep one key so softmax stays finite
    idx = torch.zeros(
        num_blocks, cfg.tokens_per_block, cfg.merge_factor, dtype=torch.int64
    )
    for b, (h, w) in enumerate(shapes):
        n = h * w
        if n > T:
            raise ValueError(f"tile with {n} patches exceeds the {T}-patch block")
        pix[b, :n] = pixel_values[b, :n].to(cfg.dtype)
        pos[b, :n] = resize_position_embeddings(pos_table, h, w).to(cfg.dtype)
        bias[b, :, :, :n] = 0.0
        bias[b, :, :, n:] = MASK_VALUE
        idx[b] = unshuffle_index(h, w, cfg.downsample_factor, cfg.tokens_per_block)
    idx = idx + (torch.arange(num_blocks) * T).view(-1, 1, 1)
    return {
        "pixel_values": pix,
        "pos_embeds": pos,
        "key_bias": bias,
        "unshuffle_idx": idx.reshape(num_blocks, -1).to(torch.int32),
    }


def merged_tokens(cfg: D1VisionConfig, h: int, w: int) -> int:
    return (h // cfg.downsample_factor) * (w // cfg.downsample_factor)


def vision_mappings(prefix: str = "visual") -> dict[str, str]:
    """Parameter name -> checkpoint key for the tower and projector."""
    vt = "model.vision_tower.vision_model"
    m = {
        f"{prefix}.patch_embedding_weight": f"{vt}.embeddings.patch_embedding.weight",
        f"{prefix}.patch_embedding_bias": f"{vt}.embeddings.patch_embedding.bias",
        f"{prefix}.position_embedding_weight": f"{vt}.embeddings.position_embedding.weight",
        f"{prefix}.post_layernorm.weight": f"{vt}.post_layernorm.weight",
        f"{prefix}.post_layernorm.bias": f"{vt}.post_layernorm.bias",
    }
    for n in ("linear_1", "linear_2"):
        for t in ("weight", "bias"):
            m[f"{prefix}.{n}_{t}"] = f"model.multi_modal_projector.{n}.{t}"
    return m


def vision_layer_mappings(prefix: str, num_layers: int) -> dict[str, str]:
    vt = "model.vision_tower.vision_model.encoder.layers"
    m = {}
    for i in range(num_layers):
        p, c = f"{prefix}.layers.{i}", f"{vt}.{i}"
        for ln in ("layer_norm1", "layer_norm2"):
            for t in ("weight", "bias"):
                m[f"{p}.{ln}.{t}"] = f"{c}.{ln}.{t}"
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for t in ("weight", "bias"):
                m[f"{p}.{n}_{t}"] = f"{c}.self_attn.{n}.{t}"
        for n in ("fc1", "fc2"):
            for t in ("weight", "bias"):
                m[f"{p}.{n}_{t}"] = f"{c}.mlp.{n}.{t}"
    return m
