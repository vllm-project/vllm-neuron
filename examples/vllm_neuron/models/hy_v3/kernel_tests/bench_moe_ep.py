"""Timing: moe_cte shard_on_i_hybrid for the most loaded rank under EP x TP splits.

Routing: real layer-79 router on synthetic inputs (T tokens), counted over all 192 experts.
EP=e -> 192/e local experts per rank at I=1536*e/64. Times the rank whose group has
the largest sum(tokens) * I_tp.
Usage: python bench_moe_ep.py <ep_degree> [T=16384]

EP=16 (I=384) and EP=8 (I=192) fail to compile: the shard_on_I gate/up scale step
indexes a full 128-row tile, so I_TP / 2 must be a multiple of 128 (NCC_IBIR243).
"""

import sys
import time

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, downscale
from vllm_neuron.envs import get_compile_backend_name

EP = int(sys.argv[1])
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
torch.manual_seed(0)
ck = Ckpt()
H, I_full, E, K, layer = 4096, 1536, 192, 8, 79
L, I = E // EP, I_full * EP // 64
p = f"model.layers.{layer}.mlp"

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
counts = (aff != 0).sum(0)
group_load = counts.view(EP, L).sum(1)
g = int(group_load.argmax())
print(
    f"EP={EP} L={L} I={I}: group tokens max {group_load.max().item()} mean {group_load.float().mean():.0f} "
    f"(hot group {g}: {counts.view(EP, L)[g].tolist()})",
    flush=True,
)
local = list(range(g * L, (g + 1) * L))


def sl(t):  # rank 0 of the EP-TP group: first I slice
    return t[:I]


gate = [sl(ck.get(f"{p}.experts.{e}.gate_proj.weight")) for e in local]
up = [sl(ck.get(f"{p}.experts.{e}.up_proj.weight")) for e in local]
down = [ck.get(f"{p}.experts.{e}.down_proj.weight")[:, :I] for e in local]
gate_up_w = torch.stack(
    [torch.stack([downscale(a).T, downscale(b).T], dim=1) for a, b in zip(gate, up)]
).contiguous()
down_w = torch.stack([downscale(d).T for d in down]).contiguous()
gu_scale = torch.full((L, 1, 2 * I), 1e-3)
d_scale = torch.full((L, 1, H), 1e-3)
local_aff = aff[:, g * L : (g + 1) * L].contiguous()


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gu", gate_up_w)
        self.register_buffer("dw", down_w)
        self.register_buffer("gus", gu_scale)
        self.register_buffer("ds", d_scale)

    def forward(self, x, aff):
        masked, tok2id, blk2exp, cond = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=L,
            num_experts_per_token=K,
            block_size=256,
            moe_group=None,
            tp_degree=1,
        )
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i_hybrid,
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
            num_static_block=L,
        )


dev = "neuron:0"
m = M().to(dev)
xd, ad = x.to(dev), local_aff.to(dev)
f = torch.compile(m, backend=get_compile_backend_name(), fullgraph=True)
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
    f"### EP={EP} T={T} hot rank: {(t_run - t_copy) * 1000:.2f} ms/layer (incl. blockwise mapping)",
    flush=True,
)
