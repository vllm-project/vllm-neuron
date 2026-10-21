"""Minimal repro: neuronx-cc fails when segmented attention is followed by another
kernel with a register-driven loop in the same graph.

Single NeuronCore, BF16, small shapes, no collectives. Kernels:
  attn  NF.segmented_attention (fused_segmented_attention; register_load + fori_loop)
  moe   NF.moe_cte shard_on_i_hybrid (register_load + fori_loop over non-empty blocks)
  pack  NF.pack_tokens (register_load + fori_loop)
Usage: python repro_icfg_segmented_hybrid.py <k1+k2+...>   e.g. attn+moe

Observed (neuronx-cc 2.27.5334, nki 0.6.0):
  attn, moe, pack, pack+pack, pack+moe, moe+attn, pack+attn, attn+attn   compile and run
  attn+moe, attn+pack   [NCC_IBIR735] use of register reg_N with no definition in function sg0000
In the full Hy3 model (attention then MoE in every layer) the same combination
fails earlier with [NCC_ICFG001] Control flow graph of function sg0001 has
multiple exit nodes.
"""

import sys

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

import vllm_neuron  # noqa: F401
import vllm_neuron.functional as NF
from vllm_neuron.envs import get_compile_backend_name

mode = sys.argv[1]
torch.manual_seed(0)
S, D, BS, SEG = 512, 128, 32, 512  # query len, head dim, KV block size, kv segment size
T, H, E, I, K, B = (
    512,
    512,
    2,
    256,
    1,
    256,
)  # MoE tokens, hidden, experts, intermediate, top-k, block
NB = 64
bf = torch.bfloat16


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kc", torch.randn(NB, 1, BS, D, dtype=bf))
        self.register_buffer("vc", torch.randn(NB, 1, BS, D, dtype=bf))
        self.register_buffer("bt", torch.arange(NB, dtype=torch.int32).view(1, NB))
        self.register_buffer("gu", torch.randn(E, H, 2, I, dtype=bf) * 0.02)
        self.register_buffer("dw", torch.randn(E, I, H, dtype=bf) * 0.02)

    def attn(self, q, prior):
        return NF.segmented_attention(
            q,
            k_cache=self.kc,
            v_cache=self.vc,
            block_tables=self.bt,
            prior_tokens=prior,
            block_size=BS,
            kv_segment_size=SEG,
            scale=1.0,
            tp_q=True,
            tp_out=True,
        )

    def moe(self, x, aff):
        masked, tok2id, blk2exp, cond = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=E,
            num_experts_per_token=K,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i_hybrid,
            num_static_block=E,
            conditions=cond,
            hidden_states=x,
            expert_affinities_masked=masked,
            gate_up_proj_weight=self.gu,
            down_proj_weight=self.dw,
            activation_function=ActFnType.SiLU,
            block_size=B,
            token_position_to_id=tok2id,
            block_to_expert=blk2exp,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )[:T]

    def pack(self, p):
        return NF.pack_tokens(p)[:, :H]

    def forward(self, q, prior, x, aff, p):
        outs = []
        for k in mode.split("+"):
            if k == "attn":
                outs.append(self.attn(q, prior).float().sum())
            elif k == "moe":
                outs.append(self.moe(x, aff).float().sum())
            elif k == "pack":
                outs.append(self.pack(p).float().sum())
        return torch.stack(outs)


q = torch.randn(1, S, D, dtype=bf)
prior = torch.full((1, 1), 256, dtype=torch.int32)
x = torch.randn(T, H, dtype=bf)
aff = torch.zeros(T, E)
aff[torch.arange(T), torch.randint(0, E, (T,))] = 1.0
# pack_tokens input: [T, H+2], last 2 cols = int32 token index bitcast to bf16 (0 = padding)
idx = torch.where(torch.rand(T) < 0.5, torch.arange(1, T + 1), 0).to(torch.int32)
p = torch.cat(
    [torch.randn(T, H, dtype=bf), idx.view(T, 1).view(bf)], 1
)  # int32 -> 2 x bf16
dev = "neuron:0"
m = M().to(dev)
f = torch.compile(m, backend=get_compile_backend_name(), fullgraph=True)
print(
    f"### {mode}: {f(q.to(dev), prior.to(dev), x.to(dev), aff.to(dev), p.to(dev)).cpu()}",
    flush=True,
)
