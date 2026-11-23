# SPDX-License-Identifier: Apache-2.0
"""`kernels.hc_in.sublayer_in` against the port's own shipped path.

Reference A: the model's existing device path (`HyperConnections.mixes` ->
`HyperConnections.pre` -> `RMSNorm`), which the integration test validated
against the checkpoint's reference. Reference B: the same math in fp32 torch
on the CPU. The fused kernel has to agree with A to bf16 noise.
"""

import sys

import torch

from kharness import DEVICE, compare, compile_fn, run_nki  # noqa: E402
from deepseek_v41 import model as M  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import hc_in  # noqa: E402


def cpu_reference(x, pre_mix, fn, base, scale, norm_w, cfg):
    hc, dim = cfg.hc_mult, cfg.hidden_size
    flat = x.float().reshape(x.shape[0], -1)
    rsq = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mixes = (flat @ fn.float().t()) * rsq
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + cfg.hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = torch.softmax(comb, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    y = (pre_mix.unsqueeze(-1) * x.float()).sum(1).to(torch.bfloat16)
    yf = y.float()
    normed = (yf * torch.rsqrt(yf.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
              * norm_w.float()).to(torch.bfloat16)
    return normed, pre, post, comb.reshape(-1, hc * hc)


def main():
    cfg = DeepseekV41Config()
    hc, dim = cfg.hc_mult, cfg.hidden_size
    torch.manual_seed(0)
    hcm = M.HyperConnections(cfg)
    width = M._mix_width(hc)
    hcm.fn.copy_(torch.randn(width, hc * dim) * 0.02)
    hcm.base.copy_(torch.randn(width) * 0.5)
    hcm.scale.copy_(torch.tensor([1.0, 0.8, 1.2]))
    norm = M.RMSNorm(dim, cfg.rms_norm_eps)
    norm.weight.copy_(1.0 + 0.1 * torch.randn(dim))
    fn_cpu, base_cpu, scale_cpu = hcm.fn.clone(), hcm.base.clone(), hcm.scale.clone()
    w_cpu = norm.weight.clone()
    hcm_d, norm_d = hcm.to(DEVICE), norm.to(DEVICE)

    def ship(x, pre_mix):
        pre, post, comb = hcm_d.mixes(x)
        y = M.HyperConnections.pre(x, pre_mix)
        return norm_d(y), pre, post, comb

    ship_c = compile_fn(ship)

    ok = True
    for T in (6, 128, 300):
        x = (torch.randn(T, hc, dim) * 1.5).to(torch.bfloat16)
        pm = torch.softmax(torch.randn(T, hc), -1)
        got = run_nki(
            hc_in.sublayer_in, x, pm, fn_cpu.t().contiguous(),
            base_cpu.reshape(1, -1), scale_cpu.reshape(1, 3), w_cpu.reshape(1, -1),
        )
        with torch.no_grad():
            a = ship_c(x.unsqueeze(0).to(DEVICE), pm.unsqueeze(0).to(DEVICE))
        a = [t.cpu().reshape(T, -1) for t in a]
        b = cpu_reference(x, pm, fn_cpu, base_cpu, scale_cpu, w_cpu, cfg)
        for name, g, ra, rb, tol in (
            ("normed", got[0], a[0], b[0], 2e-2),
            ("pre", got[1], a[1], b[1], 2e-3),
            ("post", got[2], a[2], b[2], 2e-3),
            ("comb", got[3], a[3], b[3], 2e-3),
        ):
            print(f"T={T} {name}: vs shipped-device path / vs cpu fp32")
            ok &= compare(f"T={T} {name} vs device", g, ra, tol_rel=tol)
            ok &= compare(f"T={T} {name} vs cpu", g, rb.reshape(T, -1), tol_rel=tol)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
