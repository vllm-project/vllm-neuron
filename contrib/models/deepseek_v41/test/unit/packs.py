# SPDX-License-Identifier: Apache-2.0
"""Random weights in the checkpoint's block-quantized formats, for the kernel tests."""

import torch


def fp8_pack(w: torch.Tensor, block=32):
    """bf16 weight -> (uint8 e4m3fn, uint8 e8m0 scales per 32x32 block)."""
    n, k = w.shape
    wb = w.float().reshape(n // block, block, k // block, block)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 448.0))
    scale = torch.exp2(e)
    q = (wb / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        q.reshape(n, k).view(torch.uint8).contiguous(),
        (e.reshape(n // block, k // block) + 127).to(torch.uint8).contiguous(),
    )


def fp4_pack(w: torch.Tensor, block=32):
    """bf16 weight -> (uint8 packed e2m1 pairs, uint8 e8m0 scale per row per 32)."""
    n, k = w.shape
    wb = w.float().reshape(n, k // block, block)
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(e)
    x = (wb / scale).clamp(-6, 6)
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = (mags[1:] + mags[:-1]) / 2
    code = torch.bucketize(x.abs(), mids)
    code = torch.where(x < 0, code + 8, code).reshape(n, k).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, (e.reshape(n, k // block) + 127).to(torch.uint8).contiguous()
