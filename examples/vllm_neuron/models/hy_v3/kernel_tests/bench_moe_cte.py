"""Timing: moe_cte shard_on_i vs shard_on_i_hybrid (dynamic blocks) for Hy3 prefill shapes.

EP=64 rank 0: 3 local experts, I=1536, H=4096, block 256, real layer-79 routing.
Usage: python bench_moe_cte.py <impl: shard_on_i|shard_on_i_hybrid> [T=2048] [num_static_block]
"""

import sys
import time

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, dequant, downscale, report
from vllm_neuron.envs import get_compile_backend_name

impl_name = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
n_static = int(sys.argv[3]) if len(sys.argv) > 3 else None
REPEAT = int(sys.argv[4]) if len(sys.argv) > 4 else 1
impl = getattr(MoECTEImplementation, impl_name)
torch.manual_seed(0)
ck = Ckpt()
H, I, E, K, L, layer = 4096, 1536, 192, 8, 3, 79
p = f"model.layers.{layer}.mlp"
local = list(range(L))
gate = [ck.get(f"{p}.experts.{e}.gate_proj.weight") for e in local]
up = [ck.get(f"{p}.experts.{e}.up_proj.weight") for e in local]
down = [ck.get(f"{p}.experts.{e}.down_proj.weight") for e in local]
sg = torch.stack([ck.scalar(f"{p}.experts.{e}.gate_proj.weight_scale") for e in local])
su = torch.stack([ck.scalar(f"{p}.experts.{e}.up_proj.weight_scale") for e in local])
sd = torch.stack([ck.scalar(f"{p}.experts.{e}.down_proj.weight_scale") for e in local])
gate_up_w = torch.stack(
    [torch.stack([downscale(g).T, downscale(u).T], dim=1) for g, u in zip(gate, up)]
).contiguous()
down_w = torch.stack([downscale(d).T for d in down]).contiguous()
gu_scale = (
    torch.cat([sg.view(L, 1, 1).expand(L, 1, I), su.view(L, 1, 1).expand(L, 1, I)], -1)
    * 2
).contiguous()
d_scale = (sd.view(L, 1, 1).expand(L, 1, H) * 2).contiguous()

W = ck.get(f"{p}.router.gate.weight").float()
bias = ck.get(f"{p}.expert_bias").float()
x = (
    torch.randn(T, H)
    * ck.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
).to(torch.bfloat16)
scores = torch.sigmoid(x.float() @ W.T)
idx = torch.topk(scores + bias, K, -1).indices
w = scores.gather(1, idx)
w = w / w.sum(-1, keepdim=True) * 2.826
aff = torch.zeros_like(scores).scatter(1, idx, w)
local_aff = aff[:, :L].contiguous()
print("tokens per local expert:", (local_aff != 0).sum(0).tolist(), flush=True)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gu", gate_up_w)
        self.register_buffer("dw", down_w)
        self.register_buffer("gus", gu_scale.float())
        self.register_buffer("ds", d_scale.float())

    def forward(self, x, aff):
        out = self.one(x, aff)
        for _ in range(REPEAT - 1):
            out = out + self.one(x, aff)
        return out

    def one(self, x, aff):
        masked, tok2id, blk2exp, cond = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=L,
            num_experts_per_token=K,
            block_size=256,
            moe_group=None,
            tp_degree=1,
        )
        return NF.moe_cte(
            implementation=impl,
            conditions=cond,
            hidden_states=x,
            expert_affinities_masked=masked,
            gate_up_proj_weight=self.gu,
            down_proj_weight=self.dw,
            gate_up_proj_scale=self.gus,
            down_proj_scale=self.ds,
            activation_function=ActFnType.SiLU,
            block_size=256,
            token_position_to_id=tok2id,
            block_to_expert=blk2exp,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            num_static_block=n_static,
        )


dev = "neuron:0"
m = M().to(dev)
xd, ad = x.to(dev), local_aff.to(dev)
f = torch.compile(m, backend=get_compile_backend_name(), fullgraph=True)
out = f(xd, ad).cpu()
ref = torch.zeros(T, H)
for j in range(L):
    G, U, D = (
        dequant(gate[j], sg[j]).T,
        dequant(up[j], su[j]).T,
        dequant(down[j], sd[j]).T,
    )
    ref += ((torch.nn.functional.silu(x.float() @ G) * (x.float() @ U)) @ D) * aff[
        :, j : j + 1
    ]
report(f"{impl_name} T={T} x{REPEAT}", out[:T], ref * REPEAT)
for _ in range(3):
    f(xd, ad).cpu()
n = 10
t0 = time.perf_counter()
for _ in range(n):
    f(xd, ad).cpu()
t_run = (time.perf_counter() - t0) / n
o = f(xd, ad)
t0 = time.perf_counter()
for _ in range(n):
    o.cpu()
t_copy = (time.perf_counter() - t0) / n
print(
    f"### {impl_name} T={T} static={n_static}: {(t_run - t_copy) * 1000:.2f} ms/iter "
    f"(run+copy {t_run * 1000:.2f}, copy {t_copy * 1000:.2f}; incl. blockwise mapping)",
    flush=True,
)
