"""Micro-test: static-FP8 QKV / O-proj / MLP kernels with Hy3 TP=64 rank-0 shapes."""

import sys

import torch
from nkilib.core.utils.common_types import NormType, QuantizationType

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, bcast, compile_and_run, dequant, downscale, fake_quant, report

torch.manual_seed(0)
ck = Ckpt()
H, D, TP = 4096, 128, 64
which = sys.argv[1] if len(sys.argv) > 1 else "all"


def rmsnorm(x, w, eps=1e-5):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


# ---------------------------------------------------------------- QKV (layer 1)
if which in ("all", "qkv"):
    p = "model.layers.1.self_attn"
    ws, raw = [], []
    for n, rows in (("q", slice(0, D)), ("k", slice(0, D)), ("v", slice(0, D))):
        w = ck.get(f"{p}.{n}_proj.weight")[rows]  # [D, H] fp8
        raw.append(w)
        ws.append(ck.scalar(f"{p}.{n}_proj.weight_scale"))
    s_in = ck.scalar(f"{p}.q_proj.input_scale")
    W = torch.cat([downscale(w) for w in raw], dim=0).T.contiguous()  # [H, 3D]
    w_scale = bcast(torch.stack(ws) * 2.0, 3)
    in_scale = bcast(s_in * 2.0)
    W_ref = torch.cat([dequant(w, s) for w, s in zip(raw, ws)], 0).T  # [H, 3D]

    T = 256
    x = rmsnorm(torch.randn(T, H), ck.get("model.layers.1.input_layernorm.weight")).to(
        torch.bfloat16
    )
    print("qkv x absmax / (448 s_in) =", (x.float().abs().max() / (448 * s_in)).item())

    class QKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", W)
            self.register_buffer("ws", w_scale)
            self.register_buffer("ins", in_scale)

        def forward(self, x):
            return NF.qkv_proj(
                hidden=x.unsqueeze(0),
                qkv_weights=self.w,
                bias=None,
                quantization_type=QuantizationType.STATIC,
                qkv_w_scale=self.ws,
                qkv_in_scale=self.ins,
                d_head=D,
                num_q_heads=1,
                num_kv_heads=1,
            ).squeeze(0)

    out = compile_and_run(QKV(), x)
    report("qkv STATIC vs fake-quant ref", out, fake_quant(x, s_in * 2) @ W_ref)
    report("qkv STATIC vs bf16 ref", out, x.float() @ W_ref)

# ---------------------------------------------------------------- O proj
if which in ("all", "o"):
    p = "model.layers.1.self_attn"
    w = ck.get(f"{p}.o_proj.weight")[:, 0:D]  # [H, D]
    s_w = ck.scalar(f"{p}.o_proj.weight_scale")
    s_in = ck.scalar(f"{p}.o_proj.input_scale")
    W = downscale(w).T.contiguous()  # [D, H]
    W_ref = dequant(w, s_w).T
    T = 256
    a = (torch.randn(1, 1, D, T) * 448 * s_in / 5).to(torch.bfloat16)  # [B, Nh, Dh, T]

    class O(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", W)
            self.register_buffer("ws", bcast(s_w * 2.0))
            self.register_buffer("ins", bcast(s_in * 2.0))

        def forward(self, a):
            return NF.o_proj(
                a,
                self.w,
                None,
                quantization_type=QuantizationType.STATIC,
                weight_scales=self.ws,
                input_scales=self.ins,
            ).squeeze(0)

    out = compile_and_run(O(), a)
    a_tok = a[0, 0].T  # [T, D]
    report("o_proj STATIC vs fake-quant ref", out, fake_quant(a_tok, s_in * 2) @ W_ref)
    report("o_proj STATIC vs bf16 ref", out, a_tok.float() @ W_ref)


# ---------------------------------------------------------------- MLP
def mlp_case(label, prefix, I_rank, T, prefill, ln_key):
    g = ck.get(f"{prefix}.gate_proj.weight")[0:I_rank]  # [I, H]
    u = ck.get(f"{prefix}.up_proj.weight")[0:I_rank]
    d = ck.get(f"{prefix}.down_proj.weight")[:, 0:I_rank]  # [H, I]
    sg, su, sd = (
        ck.scalar(f"{prefix}.{n}.weight_scale")
        for n in ("gate_proj", "up_proj", "down_proj")
    )
    s_gu_in = ck.scalar(f"{prefix}.gate_proj.input_scale")
    s_d_in = ck.scalar(f"{prefix}.down_proj.input_scale")
    ln_w = ck.get(ln_key)
    x = (torch.randn(T, H) * 2.0).to(torch.bfloat16)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("g", downscale(g).T.contiguous())
            self.register_buffer("u", downscale(u).T.contiguous())
            self.register_buffer("d", downscale(d).T.contiguous())
            self.register_buffer("sg", bcast(sg * 2))
            self.register_buffer("su", bcast(su * 2))
            self.register_buffer("sd", bcast(sd * 2))
            self.register_buffer("sgu_in", bcast(s_gu_in * 2))
            self.register_buffer("sd_in", bcast(s_d_in * 2))
            self.register_buffer("ln_w", ln_w)

        def forward(self, x):
            if prefill:
                h = NF.rmsnorm_quant(
                    x,
                    ln_w=self.ln_w,
                    input_dequant_scale=self.sgu_in,
                    eps=1e-5,
                    quantization_type=QuantizationType.STATIC,
                )
                norm_type, lw = NormType.NO_NORM, None
            else:
                h, norm_type, lw = x, NormType.RMS_NORM, self.ln_w.view(1, -1)
            return NF.mlp(
                h,
                self.g,
                self.u,
                self.d,
                eps=1e-5,
                ln_w=lw,
                norm_type=norm_type,
                quantization_type=QuantizationType.STATIC,
                gate_w_scale=self.sg,
                up_w_scale=self.su,
                down_w_scale=self.sd,
                gate_up_in_scale=self.sgu_in,
                down_in_scale=self.sd_in,
                output_dtype="bfloat16",
            )

    out = compile_and_run(M(), x)
    xn = rmsnorm(x, ln_w).to(torch.bfloat16).float()
    G, U, Dn = dequant(g, sg).T, dequant(u, su).T, dequant(d, sd).T
    xq = fake_quant(xn, s_gu_in * 2)
    inter = torch.nn.functional.silu(xq @ G) * (xq @ U)
    ref_q = fake_quant(inter, s_d_in * 2) @ Dn
    ref = (torch.nn.functional.silu(xn @ G) * (xn @ U)) @ Dn
    print(
        label,
        "xn absmax/(448 s_in)=",
        (xn.abs().max() / (448 * s_gu_in)).item(),
        "inter absmax/(448 s_d_in)=",
        (inter.abs().max() / (448 * s_d_in)).item(),
    )
    report(f"mlp {label} vs fake-quant ref", out, ref_q)
    report(f"mlp {label} vs bf16 ref", out, ref)


if which in ("all", "mlp"):
    mlp_case(
        "dense I=208 prefill T=256",
        "model.layers.0.mlp",
        13312 // TP,
        256,
        True,
        "model.layers.0.post_attention_layernorm.weight",
    )
    mlp_case(
        "dense I=208 decode T=8",
        "model.layers.0.mlp",
        13312 // TP,
        8,
        False,
        "model.layers.0.post_attention_layernorm.weight",
    )
    mlp_case(
        "shared I=1536 prefill T=256",
        "model.layers.1.mlp.shared_mlp",
        1536,
        256,
        True,
        "model.layers.1.post_attention_layernorm.weight",
    )
    mlp_case(
        "shared I=1536 decode T=8",
        "model.layers.1.mlp.shared_mlp",
        1536,
        8,
        False,
        "model.layers.1.post_attention_layernorm.weight",
    )
