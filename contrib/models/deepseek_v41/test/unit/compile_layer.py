"""Compile the layer kernel directly (no device, no dynamo) to surface NKI errors.
    python compile_layer.py --T 6 --stage 2
"""
import os, sys, argparse
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "8")
import torch
from kharness import compile_only, run_nki
from deepseek_v41.config import DeepseekV41Config
from deepseek_v41.kernels import layer_kernel
from deepseek_v41 import mk_host
import test_layer as TL

ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=6); ap.add_argument("--spans", type=int, default=1)
ap.add_argument("--stage", type=int, default=9); ap.add_argument("--world", type=int, default=8)
ap.add_argument("--run", action="store_true", help="compile through lite and execute on one core (timing; collectives over a 1-rank group)"); ap.add_argument("--ratio", type=int, default=0); ap.add_argument("--src", action="store_true"); ap.add_argument("--engram", action="store_true"); ap.add_argument("--tap", action="store_true")
args = ap.parse_args()
cfg = DeepseekV41Config(); world = args.world
heads = cfg.num_attention_heads // world if world <= 8 else 8
ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
p = TL.make_params(cfg, 0, world, heads, ngroups, 6, 4096)
T, B = args.T, args.spans; L = T // B; win = cfg.sliding_window
spans, tables, slots = [], [], []
for b in range(B):
    start = 200 + 37 * b
    spans.append((b * L, L, start)); table = torch.arange(b * 16, (b + 1) * 16); tables.append(table)
    slots.append(mk_host.slots_of(table, torch.arange(start, start + L), 32))
ratio = args.ratio
read_len = 0
if ratio:
    read_len = max(128, -(-((200 + 37 * (B - 1) + L) // ratio) // 128) * 128)
idx = mk_host.build_attn_indices(spans, tables, torch.cat(slots), 32, win, read_len, ratio)
topk_k = torch.zeros(T, cfg.index_topk if ratio else 1, dtype=torch.int32)
ccache = torch.zeros(64 * 32, cfg.head_dim, dtype=torch.bfloat16)
one = torch.zeros(1, 1, dtype=torch.int32)
hash_ids = torch.zeros(T, 24, dtype=torch.int32); etable = torch.zeros(4096, 256, dtype=torch.bfloat16); erow_start = one
p["e_wkv"], p["e_wkv_s"] = TL.fp8_pack(torch.zeros(5 * 5120, 24 * 256, dtype=torch.bfloat16)); p["e_qk"] = torch.ones(4 * 5120)
hc, dim, hd = cfg.hc_mult, cfg.hidden_size, cfg.head_dim
x = torch.zeros(T, hc, dim, dtype=torch.bfloat16); pre_mix = torch.zeros(T, hc)
cache = torch.zeros(64 * 32, hd, dtype=torch.bfloat16)
groups = ((0,),) if args.run else mk_host.replica_groups(world, world)
args_k = [
    x, pre_mix,
    p["hca_fn"].t().contiguous(), p["hca_base"].reshape(1, -1), p["hca_scale"].reshape(1, 3), p["attn_norm_w"].reshape(1, -1),
    p["hcf_fn"].t().contiguous(), p["hcf_base"].reshape(1, -1), p["hcf_scale"].reshape(1, 3), p["ffn_norm_w"].reshape(1, -1),
    *TL.mx8(p["wq_a"], p["wq_a_s"]), p["q_norm_w"].reshape(1, -1), *TL.mx8(p["wq_b"], p["wq_b_s"]), *TL.mx8(p["wkv"], p["wkv_s"]), p["kv_norm_w"].reshape(1, -1),
    *TL.mx8(p["wo_a"], p["wo_a_s"]), *TL.mx8(p["wo_b"], p["wo_b_s"]), p["attn_sink"].reshape(-1, 1), p["freqs"],
    idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_k,
    cache, ccache,
    torch.zeros(64 * 32, hd), torch.zeros(64 * 32, hd), torch.zeros(64 * 32, 128, dtype=torch.bfloat16),
    p["cm_wkv"], p["cm_wkv"].t().float().contiguous(), p["cm_wgate"].t().float().contiguous(), p["cm_norm_w"].reshape(1, -1),
    idx.grp_pos if args.src else one, idx.grp_mslots if args.src else one, idx.grp_wslots if args.src else one,
    *TL.mx8(p["ix_wq_b"], p["ix_wq_b_s"]), p["ix_wproj"].t().float().contiguous(), p["ix_wk"], p["ix_knorm_w"].reshape(1, -1),
    idx.clen if ratio else one,
    hash_ids, etable, erow_start, *TL.mx8(p["e_wkv"], p["e_wkv_s"]), p["e_qk"].reshape(1, -1),
    p["gate_w"].t().float().contiguous(), p["gate_b"].reshape(1, -1), torch.tensor([[0.0]]),
    *TL.mx4_stack(p["w1"], p["s1"]), *TL.mx4_stack(p["w3"], p["s3"]), *TL.mx4_stack(p["w2"], p["s2"]),
    *TL.mx8(p["sh_w1"], p["sh_s1"]), *TL.mx8(p["sh_w3"], p["sh_s3"]), *TL.mx8(p["sh_w2"], p["sh_s2"]),
    groups, groups, (ratio, B, win, args.stage, int(args.src), int(args.src), int(args.engram), int(args.tap), 0),
]
import time; t0 = time.time()
if args.run:
    out = run_nki(layer_kernel.layer_forward, *args_k)
    print(f"RAN stage {args.stage} T={T} in {time.time()-t0:.0f}s (compile + one execution)", flush=True)
else:
    r = compile_only(layer_kernel.layer_forward, *args_k)
    print(f"compiled stage {args.stage} T={T} in {time.time()-t0:.0f}s; outputs={r.return_types}; aliases={r.operand_output_aliases}")
