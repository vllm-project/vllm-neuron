"""Timing of per-layer Hy3 decode components at bs=1 on one NeuronCore (rank-0 shapes, TP=64/EP=64).

Usage: python bench_decode.py <part> where part in:
  router      torch fp32 router (matmul, sigmoid, topk, renorm, scatter)
  moe_all     moe_tkg STATIC, all-expert, 3 local experts
  shared      NF.mlp STATIC TKG, full shared expert I=1536 (fused RMSNorm)
  shared192   same, I=192 (shared expert sharded over 8 ranks)
  attn        attention_decode STATIC + QK-norm, S_ctx=2048
  attn16k     same, S_ctx=16384
  layer       attn + router + moe_all + shared (one MoE decode layer, no collectives)
"""

import sys
import time

import torch
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    QuantizationType,
)

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from vllm_neuron.envs import get_compile_backend_name

part = sys.argv[1]
torch.manual_seed(0)
H, I, E, K, L, D = 4096, 1536, 192, 8, 3, 128
FP8 = torch.float8_e4m3fn
B = 1


def fp8(*shape):
    return (torch.randn(*shape) * 4).clamp(-200, 200).to(FP8)


def scale(cols=1):
    return torch.full((128, cols), 1e-3)


class Parts(torch.nn.Module):
    def __init__(self, s_ctx):
        super().__init__()
        self.s_ctx = s_ctx
        self.register_buffer("rw", torch.randn(E, H) * 0.02)
        self.register_buffer("rb", torch.zeros(E))
        self.register_buffer("gu", fp8(L, H, 2, I))
        self.register_buffer("dw", fp8(L, I, H))
        self.register_buffer("gus", torch.full((L, 2, 1), 1e-3))
        self.register_buffer("ds", torch.full((L, 1), 1e-3))
        for n, i_sz in (("s", 1536), ("s192", 192)):
            self.register_buffer(f"{n}_g", fp8(H, i_sz))
            self.register_buffer(f"{n}_u", fp8(H, i_sz))
            self.register_buffer(f"{n}_d", fp8(i_sz, H))
        self.register_buffer("sc", scale())
        self.register_buffer("ln", torch.ones(H, dtype=torch.bfloat16))
        self.register_buffer("wqkv", fp8(H, 3 * D))
        self.register_buffer("sqkv", scale(3))
        self.register_buffer("wo", fp8(D, H))
        self.register_buffer("qn", torch.ones(1, D, dtype=torch.bfloat16))
        bs = 32
        nb = s_ctx // bs
        self.register_buffer("kc", torch.zeros(nb + 1, 1, bs, D, dtype=torch.bfloat16))
        self.register_buffer("vc", torch.zeros(nb + 1, 1, bs, D, dtype=torch.bfloat16))
        self.register_buffer("bt", (torch.arange(nb) + 1).view(1, nb).to(torch.int32))

    def router(self, x):
        scores = torch.sigmoid(x.float() @ self.rw.T)
        _, idx = torch.topk(scores + self.rb, K, dim=-1)
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * 2.826
        return torch.zeros_like(scores).scatter(1, idx, w), idx

    def moe(self, x, aff, idx):
        ones = torch.ones((L, 1), dtype=torch.float32, device=x.device)
        return NF.moe_tkg(
            hidden_input=x,
            expert_gate_up_weights=self.gu,
            expert_down_weights=self.dw,
            expert_affinities=aff,
            expert_index=idx.to(torch.int32),
            is_all_expert=True,
            rank_id=torch.zeros((1, 1), dtype=torch.int32, device=x.device),
            expert_gate_up_weights_scale=self.gus,
            expert_down_weights_scale=self.ds,
            expert_gate_up_input_scale=ones,
            expert_down_input_scale=ones,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )

    def shared(self, x, n="s"):
        return NF.mlp(
            x,
            getattr(self, f"{n}_g"),
            getattr(self, f"{n}_u"),
            getattr(self, f"{n}_d"),
            eps=1e-5,
            ln_w=self.ln.view(1, -1),
            norm_type=NormType.RMS_NORM,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.sc,
            up_w_scale=self.sc,
            down_w_scale=self.sc,
            gate_up_in_scale=self.sc,
            down_in_scale=self.sc,
            output_dtype="bfloat16",
        )

    def attn(self, x, pos):
        cos = torch.ones(D // 2, B, 1, dtype=torch.bfloat16, device=x.device)
        mask = NF.gen_attention_decode_mask(
            pos_ids=pos.view(1, B).float(),
            bs=B,
            q_head=1,
            s_active=1,
            s_prior=self.s_ctx,
            start_pos=None,
            block_len=32,
        )
        return NF.attention_decode(
            X=x.view(B, 1, H),
            X_hidden_dim_actual=H,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=1e-5,
            rmsnorm_X_gamma=self.ln.view(1, -1),
            W_qkv=self.wqkv,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.sqkv,
            input_dequant_scale_qkv=self.sc,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=1e-5,
            rmsnorm_QK_pre_rope_W_Q=self.qn,
            rmsnorm_QK_pre_rope_W_K=self.qn,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos,
            sin=cos,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=self.bt,
            K_cache=self.kc,
            V_cache=self.vc,
            attention_mask=mask,
            softmax_scale=D**-0.5,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=pos.view(B, 1).to(torch.uint32) + 32,
            W_out=self.wo,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=self.sc,
            input_dequant_scale_out=self.sc,
            transposed_out=False,
            out_in_sb=False,
            k_scale=None,
            v_scale=None,
        ).view(B, H)

    def forward(self, x, pos):
        if part == "router":
            return self.router(x)[0]
        if part == "moe_all":
            aff, idx = self.router(x)
            return self.moe(x, aff, idx)
        if part == "shared":
            return self.shared(x)
        if part == "shared192":
            return self.shared(x, "s192")
        if part.startswith("attn"):
            return self.attn(x, pos)
        if part == "layer":
            h = x + self.attn(x, pos)
            aff, idx = self.router(h)
            return h + self.moe(h, aff, idx) + self.shared(h)
        raise ValueError(part)


s_ctx = 16384 if part == "attn16k" else 2048
dev = "neuron:0"
m = Parts(s_ctx).to(dev)
x = torch.randn(B, H, dtype=torch.bfloat16).to(dev)
pos = torch.tensor([s_ctx // 2], dtype=torch.int32).to(dev)
f = torch.compile(m, backend=get_compile_backend_name(), fullgraph=True)
for _ in range(3):
    f(x, pos).cpu()
n = 30
t0 = time.perf_counter()
for _ in range(n):
    f(x, pos).cpu()
t_run = (time.perf_counter() - t0) / n
o = f(x, pos)
t0 = time.perf_counter()
for _ in range(n):
    o.cpu()
t_copy = (time.perf_counter() - t0) / n
print(
    f"### {part}: {(t_run - t_copy) * 1000:.3f} ms (run+copy {t_run * 1000:.3f}, copy {t_copy * 1000:.3f})",
    flush=True,
)
