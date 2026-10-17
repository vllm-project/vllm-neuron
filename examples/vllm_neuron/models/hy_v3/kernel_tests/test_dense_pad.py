"""Check CTE MLP with I=208 zero-padded to 256 (and unpadded 128/192/208 for comparison)."""

import sys
import torch
from nkilib.core.utils.common_types import NormType, QuantizationType
import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, bcast, compile_and_run, dequant, downscale, fake_quant, report

ck = Ckpt()
H = 4096
torch.manual_seed(0)
p = "model.layers.0.mlp"
I_real, I_pad = int(sys.argv[1]), int(sys.argv[2])
g = ck.get(f"{p}.gate_proj.weight")[0:I_real]
u = ck.get(f"{p}.up_proj.weight")[0:I_real]
d = ck.get(f"{p}.down_proj.weight")[:, 0:I_real]
sg, su, sd = (
    ck.scalar(f"{p}.{n}.weight_scale") for n in ("gate_proj", "up_proj", "down_proj")
)
s_in, s_d_in = (
    ck.scalar(f"{p}.gate_proj.input_scale"),
    ck.scalar(f"{p}.down_proj.input_scale"),
)
ln_w = ck.get("model.layers.0.post_attention_layernorm.weight")
x = torch.randn(256, H).to(torch.bfloat16)


def pad_cols(w, n):  # w [H, I] -> [H, n]
    return torch.cat([w, torch.zeros(w.shape[0], n - w.shape[1], dtype=w.dtype)], 1)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g", pad_cols(downscale(g).T.contiguous(), I_pad))
        self.register_buffer("u", pad_cols(downscale(u).T.contiguous(), I_pad))
        self.register_buffer("d", pad_cols(downscale(d), I_pad).T.contiguous())
        for n, v in (("sg", sg), ("su", su), ("sd", sd), ("si", s_in), ("sdi", s_d_in)):
            self.register_buffer(n, bcast(v * 2))
        self.register_buffer("ln_w", ln_w)

    def forward(self, x):
        h = NF.rmsnorm_quant(
            x,
            ln_w=self.ln_w,
            input_dequant_scale=self.si,
            eps=1e-5,
            quantization_type=QuantizationType.STATIC,
        )
        return NF.mlp(
            h,
            self.g,
            self.u,
            self.d,
            eps=1e-5,
            norm_type=NormType.NO_NORM,
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
xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * ln_w.float()
xq = fake_quant(xn, s_in * 2)
G, U, D = dequant(g, sg).T, dequant(u, su).T, dequant(d, sd).T
inter = torch.nn.functional.silu(xq @ G) * (xq @ U)
report(
    f"cte mlp I={I_real} pad {I_pad} vs fake-quant",
    out,
    fake_quant(inter, s_d_in * 2) @ D,
)
