"""Micro-test: attention_decode megakernel with STATIC FP8 QKV/O + pre-RoPE QK-norm.

TP=64 rank-0 shapes: 1 Q head, 1 KV head, head_dim 128. Layer 1 weights.
"""

import sys

import torch
from nkilib.core.utils.common_types import QuantizationType

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, bcast, compile_and_run, dequant, downscale, report

torch.manual_seed(0)
ck = Ckpt()
H, D = 4096, 128
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
block_size, blocks_per_seq = 32, 8
S_ctx = block_size * blocks_per_seq
num_blocks = B * blocks_per_seq + 1
theta = 11158840.0
eps = 1e-5
p = "model.layers.1.self_attn"

raw = [ck.get(f"{p}.{n}_proj.weight")[0:D] for n in "qkv"]
ws = torch.stack([ck.scalar(f"{p}.{n}_proj.weight_scale") for n in "qkv"])
s_in = ck.scalar(f"{p}.q_proj.input_scale")
wo = ck.get(f"{p}.o_proj.weight")[:, 0:D]
s_wo, s_o_in = (
    ck.scalar(f"{p}.o_proj.weight_scale"),
    ck.scalar(f"{p}.o_proj.input_scale"),
)
q_norm = ck.get(f"{p}.q_norm.weight")
k_norm = ck.get(f"{p}.k_norm.weight")
ln = ck.get("model.layers.1.input_layernorm.weight").float()

positions = torch.tensor([37, 100, 5, 200, 64, 150, 1, 255][:B])
block_table = (torch.arange(B * blocks_per_seq).view(B, blocks_per_seq) + 1).to(
    torch.int32
)
slot = (
    block_table[torch.arange(B), positions // block_size].long() * block_size
    + positions % block_size
)

k_cache = (torch.randn(num_blocks, 1, block_size, D)).to(torch.bfloat16)
v_cache = (torch.randn(num_blocks, 1, block_size, D) * 0.05).to(torch.bfloat16)
x = (torch.randn(B, H) * ln).to(torch.bfloat16)

inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
freqs = positions.float()[:, None] * inv_freq[None]
emb = torch.cat([freqs, freqs], -1)
cos, sin = emb.cos(), emb.sin()


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "w_qkv", torch.cat([downscale(w) for w in raw], 0).T.contiguous()
        )
        self.register_buffer("w_qkv_s", bcast(ws * 2, 3))
        self.register_buffer("in_s", bcast(s_in * 2))
        self.register_buffer("w_o", downscale(wo).T.contiguous())
        self.register_buffer("w_o_s", bcast(s_wo * 2))
        self.register_buffer("o_in_s", bcast(s_o_in * 2))
        self.register_buffer("qn", q_norm.view(1, -1))
        self.register_buffer("kn", k_norm.view(1, -1))

    def forward(self, x, cos, sin, positions, block_table, slot, k_cache, v_cache):
        half = D // 2
        cos_k = (
            cos[:, :half]
            .view(B, 1, half)
            .permute(2, 0, 1)
            .contiguous()
            .to(torch.bfloat16)
        )
        sin_k = (
            sin[:, :half]
            .view(B, 1, half)
            .permute(2, 0, 1)
            .contiguous()
            .to(torch.bfloat16)
        )
        mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B).to(torch.float32),
            bs=B,
            q_head=1,
            s_active=1,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )
        out = NF.attention_decode(
            X=x.view(B, 1, H),
            X_hidden_dim_actual=H,
            rmsnorm_X_enabled=False,
            W_qkv=self.w_qkv,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.w_qkv_s,
            input_dequant_scale_qkv=self.in_s,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=eps,
            rmsnorm_QK_pre_rope_W_Q=self.qn,
            rmsnorm_QK_pre_rope_W_K=self.kn,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=mask,
            softmax_scale=D**-0.5,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot.view(B, 1).to(torch.uint32),
            W_out=self.w_o,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=self.w_o_s,
            input_dequant_scale_out=self.o_in_s,
            transposed_out=False,
            out_in_sb=False,
            k_scale=None,
            v_scale=None,
        )
        return out, k_cache, v_cache


kc, vc = k_cache.clone(), v_cache.clone()
out, kc_out, vc_out = compile_and_run(
    Attn(), x, cos, sin, positions, block_table, slot, kc, vc
)


# ---------------------------------------------------------------- reference
def rms(t, w):
    t = t.float()
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def rot(t):
    return torch.cat([-t[..., half:], t[..., :half]], -1)


half = D // 2
Wq, Wk, Wv = (dequant(w, s).T for w, s in zip(raw, ws))
xf = x.float()
q = rms(xf @ Wq, q_norm)
k = rms(xf @ Wk, k_norm)
v = xf @ Wv
q = q * cos + rot(q) * sin
k = k * cos + rot(k) * sin
ref_k, ref_v = k_cache.float().clone(), v_cache.float().clone()
bi, pi = slot // block_size, slot % block_size
ref_k[bi, 0, pi] = k
ref_v[bi, 0, pi] = v
outs = []
for b in range(B):
    ks = ref_k[block_table[b].long(), 0].reshape(S_ctx, D)[: positions[b] + 1]
    vs = ref_v[block_table[b].long(), 0].reshape(S_ctx, D)[: positions[b] + 1]
    a = torch.softmax((q[b] @ ks.T) * D**-0.5, -1) @ vs
    outs.append(a)
attn = torch.stack(outs)
ref = attn @ dequant(wo, s_wo).T

report(f"attention_decode STATIC+QKnorm B={B} out", out.view(B, H), ref)
report("k_cache written slots", kc_out[bi, 0, pi], k)
report("v_cache written slots", vc_out[bi, 0, pi], v)
