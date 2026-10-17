"""Decode (TKG) static-FP8 MLP across small batch sizes: fused-norm vs rmsnorm_quant input."""

import sys
import torch
from nkilib.core.utils.common_types import NormType, QuantizationType
import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, bcast, compile_and_run, dequant, downscale, report

ck = Ckpt()
H = 4096
torch.manual_seed(0)
T, I, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
p = "model.layers.1.mlp.shared_mlp"
g = ck.get(f"{p}.gate_proj.weight")[:I]
u = ck.get(f"{p}.up_proj.weight")[:I]
d = ck.get(f"{p}.down_proj.weight")[:, :I]
sg, su, sd = (
    ck.scalar(f"{p}.{n}.weight_scale") for n in ("gate_proj", "up_proj", "down_proj")
)
si, sdi = (
    ck.scalar(f"{p}.gate_proj.input_scale"),
    ck.scalar(f"{p}.down_proj.input_scale"),
)
ln_w = ck.get("model.layers.1.post_attention_layernorm.weight")
x = torch.randn(T, H).to(torch.bfloat16)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g", downscale(g).T.contiguous())
        self.register_buffer("u", downscale(u).T.contiguous())
        self.register_buffer("d", downscale(d).T.contiguous())
        for n, v in (("sg", sg), ("su", su), ("sd", sd), ("si", si), ("sdi", sdi)):
            self.register_buffer(n, bcast(v * 2))
        self.register_buffer("ln_w", ln_w)

    def forward(self, x):
        if mode == "fused":
            h, nt, lw = x, NormType.RMS_NORM, self.ln_w.view(1, -1)
        else:
            h = NF.rmsnorm_quant(
                x,
                ln_w=self.ln_w,
                input_dequant_scale=self.si,
                eps=1e-5,
                quantization_type=QuantizationType.STATIC,
            )
            nt, lw = NormType.NO_NORM, None
            if mode == "bf16norm":
                xf = x.float()
                h = (
                    xf
                    * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
                    * self.ln_w.float()
                ).to(torch.bfloat16)
        return NF.mlp(
            h,
            self.g,
            self.u,
            self.d,
            eps=1e-5,
            ln_w=lw,
            norm_type=nt,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.sg,
            up_w_scale=self.su,
            down_w_scale=self.sd,
            gate_up_in_scale=self.si,
            down_in_scale=self.sdi,
            output_dtype="bfloat16",
        )


out = compile_and_run(M(), x)
xf = x.float()
xn = (
    (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * ln_w.float())
    .to(torch.bfloat16)
    .float()
)
G, U, D = dequant(g, sg).T, dequant(u, su).T, dequant(d, sd).T
report(
    f"decode mlp T={T} I={I} {mode}",
    out,
    (torch.nn.functional.silu(xn @ G) * (xn @ U)) @ D,
)
