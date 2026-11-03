# SPDX-License-Identifier: Apache-2.0
"""The weight-layout hook: how each rank's drafter weights are laid out for the kernel.

`prepare(rank, world, kernel, weights, load)` runs ONCE per rank on the host before the
kernel is launched (the model does the same once, when it loads the checkpoint: it is not
part of the measured latency). It returns the tensors the kernel receives for the WEIGHT
parameters (`WEIGHTS` below): `{name: tensor}`; a name it leaves out keeps its recorded tensor.

  rank, world   this rank and the world size (64)
  kernel        the entry being prepared ("draft_decode_tp64" or "draft_decode_batched_tp64")
  weights       {name: tensor}: this rank's recorded weight tensors (the deployment's layout:
                attention heads split over the 8-rank attention group, the 128 routed experts
                spread over the 64 ranks, n_local = 2 per rank, `expert_lo` = the first one's
                global id; vocabulary-parallel embedding / head / markov tables)
  load          load(name, r) -> rank r's recorded tensor of weight `name` (any rank), e.g. to
                re-shard the routed experts (TP8EP8: each 8-rank group holds 16 experts, each
                rank a 1/8 slice of their intermediate dimension) or to replicate some

Rules (README.md): only weight parameters; the result may change their shapes, dtypes,
contents and layout (re-shard, transpose, re-tile, re-quantize only if exact), and may change
`expert_lo` into any expert map the kernel reads; it must be a pure function of the weights
(never of activations, caches, routing or the recorded outputs), and a rank's prepared
weights may not exceed 1.25x its recorded weight bytes (run.py checks).
"""
import torch

#: The weight parameters (the same names in both entries).
WEIGHTS = ['main_proj', 'main_proj_s', 'main_norm_w', 'embed_w', 'vocab_start', 'hca_fn_t0', 'hca_fn_t1', 'hca_fn_t2', 'hca_base', 'hca_scale', 'attn_norm_w', 'hcf_fn_t0', 'hcf_fn_t1', 'hcf_fn_t2', 'hcf_base', 'hcf_scale', 'ffn_norm_w', 'wq_a', 'wq_a_s', 'q_norm_w', 'wq_b', 'wq_b_s', 'wkv', 'wkv_s', 'kv_norm_w', 'wo_a', 'wo_a_s', 'wo_b', 'wo_b_s', 'attn_sink', 'freqs', 'gate_w_t0', 'gate_w_t1', 'gate_w_t2', 'gate_b', 'expert_lo', 'w1', 's1', 'w3', 's3', 'w2', 's2', 'sh_w1', 'sh_s1', 'sh_w3', 'sh_s3', 'sh_w2', 'sh_s2', 'fnorm_w', 'head_w', 'mk_embed_w', 'mk_head_w', 'conf_w']


#: main_proj (the taps' projection, replicated in the recording: ~79 MB a rank) column-sharded:
#: rank r keeps output columns [r * dim / world, (r + 1) * dim / world) -- the kernel projects
#: its slice and all_gathers main_x over the world group (whose position p holds rank p's
#: slice for the recorded groups; the kernel maps any other order)
MAIN_PROJ_TP = True

#: the shared expert's gate / up projections (sh_w1 / sh_s1, sh_w3 / sh_s3: [S, K/4 | K/32, inter],
#: replicated in the recording) column-sharded: rank r keeps hidden columns
#: [r * inter / world, (r + 1) * inter / world); the kernel computes its slice of the SwiGLU
#: hidden rows and all_gathers them over the world group for the replicated down projection
SH_TP = True


#: the routers' fp32 weights (gate_w_t0..2: bf16 values widened) as their exact bf16 copies: the
#: router GEMM (`gemm_fp32`) then runs one bf16 pass instead of fp32 passes (the same products)
ROUTER_BF16 = True


#: the routed experts (EP64 in the recording: 2 a rank) and the shared expert in the target model's
#: decode TP x EP layout (`../moe_tp_layout.py`, read by `moe_tp.py`): each 8-rank group holds the 16
#: experts of its group, each rank 288 hidden columns of every one, stacked per stage; the shared
#: expert in 72 blocks of 32 hidden columns. Needs `load` (the other ranks' experts) or `ckpt` (the
#: model's checkpoint slices: `mk_model.draft_weights_decode(tp=True)`, the single-sequence entry's);
#: the batched entry keeps the recorded layout.
MOE_TP = True
MOE_NAMES = ("gate_w_t0", "gate_w_t1", "gate_w_t2", "expert_lo", "w1", "s1", "w3", "s3", "w2", "s2",
             "sh_w1", "sh_s1", "sh_w3", "sh_s3", "sh_w2", "sh_s2")


def _layout_mod():
    try:
        from .. import moe_tp_layout as ML  # the model's package (kernels/tp64/moe_tp_layout.py)
    except (ImportError, ValueError):
        from . import moe_tp_layout as ML   # a workspace's flat copy
    return ML


def moe_tp(rank, world, weights, load, ckpt=None):
    """The stages' MoE weights in the TP x EP layout: `{name: tensor}` (stacked on the stage axis).
    `ckpt(s, name, e, h0, h1)`, when given, makes the routed slices instead of `load`."""
    ML = _layout_mod()
    nst, n_local = weights["w1"].shape[0], weights["w1"].shape[1]
    n_experts = weights["gate_w_t0"].shape[1]
    e0, ng, _ = ML.group_of(rank, n_experts, world)
    cache = {}

    def recorded(name, r):
        if (name, r) not in cache:
            cache[(name, r)] = weights[name] if r == rank else load(name, r)
        return cache[(name, r)]

    st = {k: [] for k in ("w1", "s1", "w2", "sh_w1", "sh_s1", "sh_w2", "sh_s2")}
    for s in range(nst):
        def get_slice(name, e, h0, h1, s=s):
            # expert e's MX tensor `name` (rank e // n_local's slot e % n_local) at hidden columns [h0, h1)
            if ckpt is not None:
                return ckpt(s, name, e, h0, h1)
            t = recorded(name, e // n_local)[s, e % n_local]
            if name in ("w1", "w3", "s1", "s3"):
                return t[:, h0:h1].contiguous()
            return t[h0 // (4 if name == "w2" else 32) : h1 // (4 if name == "w2" else 32)].contiguous()

        r = ML.routed(rank, world, n_experts, get_slice)
        sh = ML.shared(rank, world, *(weights[n][s] for n in ("sh_w1", "sh_s1", "sh_w3", "sh_s3", "sh_w2", "sh_s2")))
        for k in ("w1", "s1", "w2"):
            st[k].append(r[k])
        for k in ("sh_w1", "sh_s1", "sh_w2", "sh_s2"):
            st[k].append(sh[k])
    out = {k: torch.stack(v).contiguous() for k, v in st.items()}
    for k, like in (("w3", "w1"), ("s3", "s1"), ("s2", "s1"), ("sh_w3", "sh_w1"), ("sh_s3", "sh_s1")):
        out[k] = torch.zeros((nst, 1, 1), dtype=weights[k if k in weights else like].dtype)
    out["expert_lo"] = torch.tensor([[float(e0)]], dtype=torch.float32)
    for n in ("gate_w_t0", "gate_w_t1", "gate_w_t2"):
        out[n] = ML.router(weights[n])
    return out


def prepare(rank, world, kernel, weights, load, ckpt=None):
    out = {}
    if ROUTER_BF16:
        for name in ("gate_w_t0", "gate_w_t1", "gate_w_t2"):
            t = weights.get(name)
            if t is not None and t.dtype == torch.float32:
                b = t.to(torch.bfloat16)
                if torch.equal(b.float(), t):
                    out[name] = b.contiguous()
    if MAIN_PROJ_TP:
        w, sc = weights["main_proj"], weights["main_proj_s"]
        dim = w.shape[1]
        if dim % world == 0:
            c = dim // world
            out["main_proj"] = w[:, rank * c : (rank + 1) * c].contiguous()
            out["main_proj_s"] = sc[:, rank * c : (rank + 1) * c].contiguous()
    if MOE_TP and (load is not None or ckpt is not None) and kernel == "draft_decode_tp64":
        out.update(moe_tp(rank, world, weights, load, ckpt))
        return out
    if SH_TP:
        n = weights["sh_w1"].shape[-1]
        if n % world == 0 and (n // world) % 4 == 0:
            c = n // world
            for name in ("sh_w1", "sh_s1", "sh_w3", "sh_s3"):
                out[name] = weights[name][..., rank * c : (rank + 1) * c].contiguous()
    return out
