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


def prepare(rank, world, kernel, weights, load):
    out = {}
    if MAIN_PROJ_TP:
        w, sc = weights["main_proj"], weights["main_proj_s"]
        dim = w.shape[1]
        if dim % world == 0:
            c = dim // world
            out["main_proj"] = w[:, rank * c : (rank + 1) * c].contiguous()
            out["main_proj_s"] = sc[:, rank * c : (rank + 1) * c].contiguous()
    if SH_TP:
        n = weights["sh_w1"].shape[-1]
        if n % world == 0 and (n // world) % 4 == 0:
            c = n // world
            for name in ("sh_w1", "sh_s1", "sh_w3", "sh_s3"):
                out[name] = weights[name][..., rank * c : (rank + 1) * c].contiguous()
    return out
