"""Micro-test: FP8 MoE kernels (moe_cte shard_on_i prefill, moe_tkg STATIC decode).

EP=64 rank-0 shapes: 3 local experts (0..2), full I=1536.
"""

import sys

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from common import Ckpt, compile_and_run, dequant, downscale, report

torch.manual_seed(0)
ck = Ckpt()
H, I, E, K, L = 4096, 1536, 192, 8, 3
which = sys.argv[1]
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
p = f"model.layers.{layer}.mlp"
local = list(range(L))

gate = [ck.get(f"{p}.experts.{e}.gate_proj.weight") for e in local]  # [I, H]
up = [ck.get(f"{p}.experts.{e}.up_proj.weight") for e in local]
down = [ck.get(f"{p}.experts.{e}.down_proj.weight") for e in local]  # [H, I]
sg = torch.stack([ck.scalar(f"{p}.experts.{e}.gate_proj.weight_scale") for e in local])
su = torch.stack([ck.scalar(f"{p}.experts.{e}.up_proj.weight_scale") for e in local])
sd = torch.stack([ck.scalar(f"{p}.experts.{e}.down_proj.weight_scale") for e in local])

# Kernel layouts: gate_up [E, H, 2, I], down [E, I, H]; weights halved, scales doubled.
gate_up_w = torch.stack(
    [torch.stack([downscale(g).T, downscale(u).T], dim=1) for g, u in zip(gate, up)]
).contiguous()
down_w = torch.stack([downscale(d).T for d in down]).contiguous()


def route(x):
    """Hy3 router: sigmoid scores, bias only for selection, renorm, x2.826."""
    W = ck.get(f"{p}.router.gate.weight").float()
    bias = ck.get(f"{p}.expert_bias").float()
    scores = torch.sigmoid(x.float() @ W.T)
    idx = torch.topk(scores + bias, K, dim=-1).indices
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * 2.826
    return torch.zeros_like(scores).scatter(1, idx, w), idx


def reference(x, aff):
    out = torch.zeros(x.shape[0], H)
    for j, e in enumerate(local):
        G, U, D = (
            dequant(gate[j], sg[j]).T,
            dequant(up[j], su[j]).T,
            dequant(down[j], sd[j]).T,
        )
        y = (torch.nn.functional.silu(x.float() @ G) * (x.float() @ U)) @ D
        out += y * aff[:, e : e + 1]
    return out


if which == "cte":
    T = 1024
    x = (
        torch.randn(T, H)
        * ck.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
    ).to(torch.bfloat16)
    aff, _ = route(x)
    local_aff = aff[:, :L].contiguous()
    print("tokens routed to local experts:", (local_aff != 0).sum(0).tolist())
    gu_scale = (
        torch.cat(
            [sg.view(L, 1, 1).expand(L, 1, I), su.view(L, 1, 1).expand(L, 1, I)], -1
        )
        * 2
    )
    d_scale = sd.view(L, 1, 1).expand(L, 1, H) * 2

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gu", gate_up_w)
            self.register_buffer("dw", down_w)
            self.register_buffer("gus", gu_scale.contiguous().float())
            self.register_buffer("ds", d_scale.contiguous().float())

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
                implementation=MoECTEImplementation.shard_on_i,
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
            )

    out = compile_and_run(M(), x, local_aff)
    report(f"moe_cte shard_on_i fp8 T={T} layer{layer}", out[:T], reference(x, aff))

if which == "tkg":
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    x = (
        torch.randn(T, H)
        * ck.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
    ).to(torch.bfloat16)
    aff, idx = route(x)
    # Make sure local experts get hit in this tiny batch.
    aff[:, 0] = 0.5
    aff[:, 2] = 0.25

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gu", gate_up_w)
            self.register_buffer("dw", down_w)
            self.register_buffer(
                "gus", torch.stack([sg, su], 1).view(L, 2, 1).float() * 2
            )
            self.register_buffer("ds", sd.view(L, 1).float() * 2)
            self.register_buffer("ins", torch.ones(L, 1))

        def forward(self, x, aff, idx):
            rank_id = torch.zeros((1, 1), dtype=torch.int32, device=x.device)
            return NF.moe_tkg(
                hidden_input=x,
                expert_gate_up_weights=self.gu,
                expert_down_weights=self.dw,
                expert_affinities=aff,
                expert_index=idx,
                is_all_expert=True,
                rank_id=rank_id,
                expert_gate_up_weights_scale=self.gus,
                expert_down_weights_scale=self.ds,
                expert_gate_up_input_scale=self.ins,
                expert_down_input_scale=self.ins,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                activation_fn=ActFnType.SiLU,
            )

    out = compile_and_run(M(), x, aff, idx.to(torch.int32))
    report(f"moe_tkg STATIC all-expert T={T} layer{layer}", out, reference(x, aff))
