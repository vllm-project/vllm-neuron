# SPDX-License-Identifier: Apache-2.0
"""Argument builders for tracing the megakernels without a device (random weights of
the real shapes at an 8-rank world): `layer_args(...)`, `draft_args(...)`."""
import os
from types import SimpleNamespace

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "8")
import torch  # noqa: E402

from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41 import mk_host  # noqa: E402
import test_layer as TL  # noqa: E402
from test_layer import mx8, mx4_stack  # noqa: E402,F401

WORLD = 8
_P = {}


def params(world=WORLD, n_local=6):
    key = (world, n_local)
    if key not in _P:
        cfg = DeepseekV41Config()
        heads = cfg.num_attention_heads // world if world <= 8 else 8
        ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
        p = TL.make_params(cfg, 0, world, heads, ngroups, n_local, 4096)
        p["e_wkv"], p["e_wkv_s"] = TL.fp8_pack(torch.zeros(5 * 5120, 24 * 256, dtype=torch.bfloat16))
        p["e_qk"] = torch.ones(4 * 5120)
        _P[key] = p
    return _P[key]


def layer_args(T=6, spans=1, ratio=0, src=False, engram=False, tap=False, stage=9, world=WORLD,
               n_local=6, groups=None):
    cfg = DeepseekV41Config()
    p = params(world, n_local)
    B = spans
    L = T // B
    win = cfg.sliding_window
    spans_l, tables, slots = [], [], []
    for b in range(B):
        start = 200 + 37 * b
        spans_l.append((b * L, L, start))
        table = torch.arange(b * 16, (b + 1) * 16)
        tables.append(table)
        slots.append(mk_host.slots_of(table, torch.arange(start, start + L), 32))
    read_len = 0
    if ratio:
        read_len = max(128, -(-((200 + 37 * (B - 1) + L) // ratio) // 128) * 128)
    idx = mk_host.build_attn_indices(spans_l, tables, torch.cat(slots), 32, win, read_len, ratio)
    topk_k = torch.zeros(T, cfg.index_topk if ratio else 1, dtype=torch.int32)
    ccache = torch.zeros(64 * 32, cfg.head_dim, dtype=torch.bfloat16)
    one = torch.zeros(1, 1, dtype=torch.int32)
    hash_ids = torch.zeros(T, 24, dtype=torch.int32)
    etable = torch.zeros(4096, 256, dtype=torch.bfloat16)
    hc, dim, hd = cfg.hc_mult, cfg.hidden_size, cfg.head_dim
    x = torch.zeros(T, hc, dim, dtype=torch.bfloat16)
    pre_mix = torch.zeros(T, hc)
    cache = torch.zeros(64 * 32, hd, dtype=torch.bfloat16)
    if groups is None:
        groups = mk_host.replica_groups(world, world)
    d_u8 = torch.zeros(1, 1, dtype=torch.uint8)
    d_i32 = one
    if engram:
        e_args = [hash_ids, etable, one, *mx8(p["e_wkv"], p["e_wkv_s"]), p["e_qk"].reshape(1, -1)]
    else:
        e_args = [d_i32, torch.zeros(1, 1, dtype=torch.bfloat16), one, d_i32, d_u8, torch.zeros(1, 1)]
    return [
        x, pre_mix,
        p["hca_fn"].t().contiguous(), p["hca_base"].reshape(1, -1), p["hca_scale"].reshape(1, 3), p["attn_norm_w"].reshape(1, -1),
        p["hcf_fn"].t().contiguous(), p["hcf_base"].reshape(1, -1), p["hcf_scale"].reshape(1, 3), p["ffn_norm_w"].reshape(1, -1),
        *mx8(p["wq_a"], p["wq_a_s"]), p["q_norm_w"].reshape(1, -1), *mx8(p["wq_b"], p["wq_b_s"]), *mx8(p["wkv"], p["wkv_s"]), p["kv_norm_w"].reshape(1, -1),
        *mx8(p["wo_a"], p["wo_a_s"]), *mx8(p["wo_b"], p["wo_b_s"]), p["attn_sink"].reshape(-1, 1), p["freqs"],
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_k,
        cache, ccache,
        torch.zeros(64 * 32, hd), torch.zeros(64 * 32, hd), torch.zeros(64 * 32, 128, dtype=torch.bfloat16),
        p["cm_wkv"], p["cm_wkv"].t().float().contiguous(), p["cm_wgate"].t().float().contiguous(), p["cm_norm_w"].reshape(1, -1),
        idx.grp_pos if src else one, idx.grp_mslots if src else one, idx.grp_wslots if src else one,
        *mx8(p["ix_wq_b"], p["ix_wq_b_s"]), p["ix_wproj"].t().float().contiguous(), p["ix_wk"], p["ix_knorm_w"].reshape(1, -1),
        idx.clen if ratio else one,
        *e_args,
        p["gate_w"].t().float().contiguous(), p["gate_b"].reshape(1, -1), torch.tensor([[0.0]]),
        *mx4_stack(p["w1"], p["s1"]), *mx4_stack(p["w3"], p["s3"]), *mx4_stack(p["w2"], p["s2"]),
        *mx8(p["sh_w1"], p["sh_s1"]), *mx8(p["sh_w3"], p["sh_s3"]), *mx8(p["sh_w2"], p["sh_s2"]),
        groups, groups, (ratio, B, win, stage, int(src), int(src), int(engram), int(tap), 0),
    ]


def draft_args(at_zero=False, L=6, world=WORLD, stage=9, scheme="tp8pp8"):
    """The drafter's arguments at an 8-rank world (random weights, zero inputs): the
    batched kernel's layout (`meta [B, 3]`, `rows_c`) for `tp8pp8`, the per-sequence
    kernel's (`meta [1, 3]`) for `tp64`."""
    from test_layer import make_params
    cfg = DeepseekV41Config()
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    heads, ngroups = 8, 1
    n_local = cfg.dspark_n_routed_experts // world
    win = cfg.sliding_window
    K = cfg.dspark_block_size
    VOCAB, NOISE = 16084, 15999
    from deepseek_v41 import model as M
    part = M.vocab_shard_width(VOCAB, world)
    mrank = cfg.dspark_markov_rank
    ps = []
    for s in range(3):
        p = make_params(cfg, 100 * s, world, heads, ngroups, n_local, 4096)
        g = torch.Generator().manual_seed(7000 + 100 * s)
        p["gate_w"] = (torch.randn(cfg.dspark_n_routed_experts, dim, generator=g) * 0.02).to(torch.bfloat16)
        p["gate_b"] = torch.randn(cfg.dspark_n_routed_experts, generator=g) * 0.1
        ps.append(p)
    g = torch.Generator().manual_seed(9000)
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    main_proj, main_proj_s = TL.fp8_pack(rn(dim, 3 * dim, sd=0.02).to(torch.bfloat16))
    stack = lambda key: torch.stack([p[key] for p in ps]).contiguous()  # noqa: E731

    def stack8(wkey, skey):
        pairs = [mx8(p[wkey], p[skey]) for p in ps]
        return torch.stack([w for w, _ in pairs]).contiguous(), torch.stack([s_ for _, s_ in pairs]).contiguous()

    def stack4(wkey, skey):
        pairs = [mx4_stack(p[wkey], p[skey]) for p in ps]
        return torch.stack([w for w, _ in pairs]).contiguous(), torch.stack([s_ for _, s_ in pairs]).contiguous()

    i32 = lambda x: x.to(torch.int32).reshape(-1, 1).contiguous()  # noqa: E731
    start = 0 if at_zero else 200
    row_pos = torch.arange(start, start + L)
    table = torch.arange(16)
    write_slots = mk_host.slots_of(table, row_pos, 32)
    groups = mk_host.replica_groups(world, world)
    flags = (int(at_zero), K, win, stage, 0, NOISE, VOCAB, cfg.dspark_num_experts_per_tok, 32)
    from deepseek_v41.mk_model import draft_consts
    from deepseek_v41.kernels import mk_kernels
    PT, pad_to = mk_kernels.pkg("tp8pp8").mk_common.PT, mk_kernels.pkg("tp8pp8").mk_common.pad_to
    meta = torch.tensor([[1234, L - 1, start]], dtype=torch.int32)
    head = [meta] if scheme == "tp64" else [meta, draft_consts(1, K, pad_to(win + K, PT))]
    return head + [
        torch.zeros(L, 3 * dim), i32(row_pos), i32(write_slots), i32(table),
        *mx8(main_proj, main_proj_s), torch.ones(1, dim, dtype=torch.bfloat16), torch.zeros(part, dim, dtype=torch.bfloat16),
        torch.tensor([[0]], dtype=torch.int32),
        ps[0]["hca_fn"].t().contiguous(), ps[1]["hca_fn"].t().contiguous(), ps[2]["hca_fn"].t().contiguous(),
        stack("hca_base"), stack("hca_scale"), stack("attn_norm_w"),
        ps[0]["hcf_fn"].t().contiguous(), ps[1]["hcf_fn"].t().contiguous(), ps[2]["hcf_fn"].t().contiguous(),
        stack("hcf_base"), stack("hcf_scale"), stack("ffn_norm_w"),
        *stack8("wq_a", "wq_a_s"), stack("q_norm_w"), *stack8("wq_b", "wq_b_s"),
        *stack8("wkv", "wkv_s"), stack("kv_norm_w"), *stack8("wo_a", "wo_a_s"), *stack8("wo_b", "wo_b_s"),
        stack("attn_sink").reshape(3, heads, 1), ps[0]["freqs"],
        ps[0]["gate_w"].t().float().contiguous(), ps[1]["gate_w"].t().float().contiguous(),
        ps[2]["gate_w"].t().float().contiguous(), stack("gate_b"), torch.tensor([[0.0]]),
        *stack4("w1", "s1"), *stack4("w3", "s3"), *stack4("w2", "s2"),
        *stack8("sh_w1", "sh_s1"), *stack8("sh_w3", "sh_s3"), *stack8("sh_w2", "sh_s2"),
        torch.zeros(64 * 32, hd, dtype=torch.bfloat16), torch.zeros(64 * 32, hd, dtype=torch.bfloat16),
        torch.zeros(64 * 32, hd, dtype=torch.bfloat16),
        torch.ones(1, dim, dtype=torch.bfloat16), torch.zeros(part, dim, dtype=torch.bfloat16),
        torch.zeros(part, mrank, dtype=torch.bfloat16), torch.zeros(part, mrank, dtype=torch.bfloat16),
        torch.zeros(1, dim + mrank, dtype=torch.bfloat16), groups, groups, flags,
    ]
