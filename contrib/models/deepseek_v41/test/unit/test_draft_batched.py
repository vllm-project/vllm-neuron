# SPDX-License-Identifier: Apache-2.0
"""The batched drafter (`draft_batched.draft_decode_batched`) against B launches of the
single-sequence `draft_kernel.draft_decode`, on N ranks: bit-identical or it fails.

    python kharness_mp.py 8 test_draft_batched.py --B 4

B decode sequences (6 main rows each, each its own start position, tap, seed, main rows
and block table, disjoint pages) over the same random weights as `test_draft.py`. The
single kernel drafts them one after the other on one set of caches, the batched kernel all
at once on a copy of the caches as they were before: every rank's drafted ids, confidences
and the three window caches must be bitwise identical.
"""

import argparse
import inspect
import os
import sys

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()
else:
    rank, world = 0, int(os.environ.get("KH_WORLD", "8"))

import torch  # noqa: E402

from kharness import run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import draft_batched, draft_kernel  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from packs import fp8_pack  # noqa: E402
from test_draft import K, NOISE, VOCAB  # noqa: E402
from test_layer import kb16, kmajor4, make_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--L", type=int, default=6)
    args = ap.parse_args()
    B, L = args.B, args.L

    cfg = DeepseekV41Config()
    dim, hd = cfg.hidden_size, cfg.head_dim
    heads, ngroups = 8, 1
    n_local = cfg.dspark_n_routed_experts // world
    win = cfg.sliding_window
    block_size = 32
    nblk = 16
    max_pos = 4096
    part = M.vocab_shard_width(VOCAB, world)
    vstart = rank * part
    mrank = cfg.dspark_markov_rank

    ps = []
    for s in range(3):
        p = make_params(cfg, rank + 100 * s, world, heads, ngroups, n_local, max_pos)
        g = torch.Generator().manual_seed(7000 + rank + 100 * s)
        p["gate_w"] = (torch.randn(cfg.dspark_n_routed_experts, dim, generator=g) * 0.02).to(torch.bfloat16)
        p["gate_b"] = torch.randn(cfg.dspark_n_routed_experts, generator=g) * 0.1
        ps.append(p)
    g = torch.Generator().manual_seed(9000 + rank)
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    main_proj, main_proj_s = fp8_pack(rn(dim, 3 * dim, sd=0.02).to(torch.bfloat16))
    main_norm_w = (1 + 0.1 * rn(dim)).to(torch.bfloat16)
    embed_w = rn(part, dim, sd=0.5).to(torch.bfloat16)
    head_w = rn(part, dim, sd=0.02).to(torch.bfloat16)
    mk_embed_w = rn(part, mrank, sd=0.5).to(torch.bfloat16)
    mk_head_w = rn(part, mrank, sd=0.05).to(torch.bfloat16)
    conf_w = rn(1, dim + mrank, sd=0.02).to(torch.bfloat16)
    fnorm_w = (1 + 0.1 * rn(dim)).to(torch.bfloat16)

    # ---- B sequences, shared by every rank (seed 0) ------------------------------------
    g0 = torch.Generator().manual_seed(0)
    num_slots = B * nblk * block_size
    caches = []
    for s in range(3):
        c = (torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16)
        caches.append(M.act_quant_dequant(c, 32))
    seqs = []
    for b in range(B):
        start = 150 + 37 * b
        tap = (b * 3) % L
        seed = 100 + 997 * b
        table = torch.arange(nblk) + b * nblk
        row_pos = torch.arange(start, start + L)
        seqs.append(dict(
            start=start, tap=tap, seed=seed, table=table, row_pos=row_pos,
            write_slots=mk_host.slots_of(table, row_pos, block_size),
            main_hidden=torch.randn(L, 3 * dim, generator=g0) * 1.5,
        ))
    i32 = lambda x: x.to(torch.int32).reshape(-1, 1).contiguous()  # noqa: E731

    groups = mk_host.replica_groups(world, world)
    stack = lambda key: torch.stack([p[key] for p in ps]).contiguous()  # noqa: E731
    stack8 = lambda w, s_: torch.stack([kb16(p[w], p[s_]) for p in ps]).contiguous()  # noqa: E731
    stack4 = lambda key: torch.stack([kmajor4(p[key]) for p in ps]).contiguous()  # noqa: E731
    weights = dict(
        main_proj=kb16(main_proj, main_proj_s), main_norm_w=main_norm_w.reshape(1, -1),
        embed_w=embed_w, vocab_start=torch.tensor([[vstart]], dtype=torch.int32),
        hca_base=stack("hca_base"), hca_scale=stack("hca_scale"), attn_norm_w=stack("attn_norm_w"),
        hcf_base=stack("hcf_base"), hcf_scale=stack("hcf_scale"), ffn_norm_w=stack("ffn_norm_w"),
        wq_a=stack8("wq_a", "wq_a_s"), q_norm_w=stack("q_norm_w"), wq_b=stack8("wq_b", "wq_b_s"),
        wkv=stack8("wkv", "wkv_s"), kv_norm_w=stack("kv_norm_w"), wo_a=stack8("wo_a", "wo_a_s"),
        wo_b=stack8("wo_b", "wo_b_s"), attn_sink=stack("attn_sink").reshape(3, heads, 1), freqs=ps[0]["freqs"],
        gate_b=stack("gate_b"), expert_lo=torch.tensor([[float(rank * n_local)]]),
        w1=stack4("w1"), s1=stack4("s1"), w3=stack4("w3"), s3=stack4("s3"), w2=stack4("w2"), s2=stack4("s2"),
        sh_w1=stack8("sh_w1", "sh_s1"), sh_w3=stack8("sh_w3", "sh_s3"), sh_w2=stack8("sh_w2", "sh_s2"),
        fnorm_w=fnorm_w.reshape(1, -1), head_w=head_w.t().contiguous(), mk_embed_w=mk_embed_w,
        mk_head_w=mk_head_w, conf_w=conf_w, groups_attn=groups, groups_world=groups,
        flags=(K, win, 9, 0, NOISE, VOCAB, cfg.dspark_num_experts_per_tok, block_size),
    )
    for i in range(3):
        weights[f"hca_fn_t{i}"] = ps[i]["hca_fn"].t().contiguous()
        weights[f"hcf_fn_t{i}"] = ps[i]["hcf_fn"].t().contiguous()
        weights[f"gate_w_t{i}"] = ps[i]["gate_w"].t().float().contiguous()

    def call(kernel, named):
        func = kernel.func if hasattr(kernel, "func") else kernel
        return run_nki(kernel, *[named[n] for n in inspect.signature(func).parameters], grid=2)

    # ---- B single-sequence launches, one after the other, on one set of caches ----------
    dev_a = [c.to("neuron:0") for c in caches]
    ids_a, conf_a = [], []
    for sq in seqs:
        named = dict(weights, meta=torch.tensor([[sq["seed"], sq["tap"], sq["start"]]], dtype=torch.int32),
                     main_hidden=sq["main_hidden"], row_pos=i32(sq["row_pos"]), write_slots=i32(sq["write_slots"]),
                     block_table=i32(sq["table"]), live=torch.tensor([[float(K)]]),
                     cache0=dev_a[0], cache1=dev_a[1], cache2=dev_a[2])
        out_ids, conf, *_ = call(draft_kernel.draft_decode, named)
        ids_a.append(out_ids.reshape(-1))
        conf_a.append(conf.reshape(-1))
    caches_a = [c.cpu() for c in dev_a]

    # ---- one batched launch on the caches as they were before ----------------------------
    dev_b = [c.to("neuron:0") for c in caches]
    ntab_p = -(-(win + K) // 128) * 128
    rows_c = torch.tensor([[b, i, b * ntab_p, 0] for b in range(B) for i in range(K)], dtype=torch.int32)
    named = dict(weights,
                 meta=torch.tensor([[sq["seed"], sq["tap"], sq["start"]] for sq in seqs], dtype=torch.int32),
                 rows_c=rows_c, main_hidden=torch.cat([sq["main_hidden"] for sq in seqs]),
                 row_pos=i32(torch.cat([sq["row_pos"] for sq in seqs])),
                 write_slots=i32(torch.cat([sq["write_slots"] for sq in seqs])),
                 block_table=i32(torch.cat([sq["table"] for sq in seqs])), live=torch.tensor([[float(B * K)]]),
                 cache0=dev_b[0], cache1=dev_b[1], cache2=dev_b[2])
    out_ids_b, conf_b, *_ = call(draft_batched.draft_decode_batched, named)
    caches_b = [c.cpu() for c in dev_b]

    tag = f"[rank {rank} B={B}]"
    ok = True
    for b in range(B):
        same = torch.equal(out_ids_b[b].reshape(-1), ids_a[b])
        ok &= same
        print(f"{tag} seq {b} ids single={ids_a[b].tolist()} batched={out_ids_b[b].reshape(-1).tolist()} "
              f"{'==' if same else 'DIFFER'}", flush=True)
    cb = conf_b.reshape(-1)
    ca = torch.cat(conf_a)
    same_conf = torch.equal(cb.view(torch.int32), ca.view(torch.int32))
    ok &= same_conf
    print(f"{tag} confidence {'bit-identical' if same_conf else f'max diff {float((cb - ca).abs().max()):.3g}'}",
          flush=True)
    for s in range(3):
        same_c = torch.equal(caches_a[s].view(torch.int16), caches_b[s].view(torch.int16))
        ok &= same_c
        print(f"{tag} cache{s} {'bit-identical' if same_c else 'DIFFER in %d rows' % int((caches_a[s] != caches_b[s]).any(1).sum())}",
              flush=True)
    print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
