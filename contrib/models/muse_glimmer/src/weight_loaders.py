# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the MuseGlimmer perception encoder.

The checkpoint stores q / k / v as **three separate** ``[H, H]`` Linear
weights (``attn.{q,k,v}_proj``), unlike Qwen3-VL's fused ``qkv``. The plugin
keeps a fused per-rank ``[H, 3 * hd]`` parameter for ``NF.qkv_proj``, so the
loader takes the rank's head slice from each of q, k and v and concatenates.
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def vis_qkv3_weight_loader(num_heads_per_rank: int, head_dim: int) -> SafetensorsWeightLoader:
    """Fuse separate q/k/v ``[H, H]`` weights into the per-rank ``[H, 3*hd]`` layout.

    Rank r takes output rows ``[r*hd, (r+1)*hd)`` of each of q, k, v (heads
    ``r*n .. r*n+n-1``), then transposes for ``x @ W``.
    """
    hd = num_heads_per_rank * head_dim

    def transform(slices, rank):
        assert len(slices) == 3, "expected [q_proj, k_proj, v_proj] weights"
        parts = [s[rank * hd : (rank + 1) * hd, :] for s in slices]
        return torch.cat(parts, dim=0).T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def vis_qkv3_bias_loader(num_heads_per_rank: int, head_dim: int) -> SafetensorsWeightLoader:
    """Bias counterpart of :func:`vis_qkv3_weight_loader` (``[3*hd]``)."""
    hd = num_heads_per_rank * head_dim

    def transform(slices, rank):
        assert len(slices) == 3, "expected [q_proj, k_proj, v_proj] biases"
        return torch.cat([s[rank * hd : (rank + 1) * hd] for s in slices], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def transposed_loader() -> SafetensorsWeightLoader:
    """Replicated Linear weight stored ``[out, in]`` -> parameter ``[in, out]``."""

    def transform(slices, rank):
        assert len(slices) == 1
        return slices[0][:].T.contiguous()

    return SafetensorsWeightLoader(transform=transform)
