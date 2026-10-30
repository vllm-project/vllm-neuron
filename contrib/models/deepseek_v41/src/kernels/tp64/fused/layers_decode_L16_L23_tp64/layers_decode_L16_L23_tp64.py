# SPDX-License-Identifier: Apache-2.0
"""`layers_decode_L16_L23_tp64`: one launch of a DeepSeek-V4.1-Flash decode step (tp64, LNC=2) running
blocks 16-23, in order:

    L16 r2, L17 r2, L18 r2, L19 r2, L20 r1_src_idx, L21 r1, L22 r1, L23 r1

Each member is this launch's own copy of the kernel of its block archetype (`<kind>/<kind>.py`,
from the port's `kernels/tp64/<kind>/`, with its own helper copies), called inside `member`
(see fused_member.py). Member m's arguments are this launch's parameters, or outputs of an
earlier member (`o<m>[0]` stream, `o<m>[1]` pre-mix, `o<m>[2]` an index source's top-k,
`o<m>[3]` a tap). One parameter is one tensor of the model: several members read the same
one where the model passes them the same tensor (a kv source's caches, the step's index
columns, the (1, 1) placeholders); a write to it in one member is ordered before the reads
in the next. Parameters are named by their first use (`L24_wq_a`); plan.json is the model's plan they come
from (each member's kernel and arguments). Returns: the last block's stream and pre-mix, the taps,
the last index source's top-k, the greedy head's tokens (what the launch has of these).

Written out from the model's fused plan (`mk_kernels.fused`) and composed from the optimized
launches' members (not run through a loop of its own): r2 (blocks 16-19) from
`layers_decode_embed_L00_L07_tp64`, with its table sharing, SBUF hand-over of the output tiles
and next-block projection prefetch; r1 (21-23) from `layers_decode_L24_L31_tp64`, sharing the
slot / index tables; r1_src_idx (20) the port's `r1_src_idx_decode`. Validated bit for bit on
the recorded cases of workspace ~/ds41_fused_L16_L23 (all 64 ranks, 4 configurations).
`mk_kernels.optimized_fused` runs it in place of the generated launch when the model's plan is
plan.json.
"""
import nki

from .fused_member import member
from .r2_decode import r2_decode
from .r1_src_idx_decode import r1_src_idx_decode
from .r1_decode import r1_decode


@nki.jit
def layers_decode_L16_L23_tp64(
    L16_stream, L16_pre_mix, L16_hca_fn_t, L16_hca_base, L16_hca_scale, L16_attn_norm_w,
    L16_hcf_fn_t, L16_hcf_base, L16_hcf_scale, L16_ffn_norm_w, L16_wq_a, L16_wq_a_s, L16_q_norm_w,
    L16_wq_b, L16_wq_b_s, L16_wkv, L16_wkv_s, L16_kv_norm_w, L16_wo_a, L16_wo_a_s, L16_wo_b,
    L16_wo_b_s, L16_attn_sink, L16_freqs, L16_row_pos, L16_write_slots, L16_win_slots,
    L16_cmp_slots, L16_twin, L16_wv, L16_cbase, L16_topk_in, L16_window_kv_cache,
    L16_compress_kv_cache, placeholder_float32, L16_index_k_cache, L16_cm_wkv, L16_cm_wkv_t,
    L16_cm_wgate_t, L16_cm_norm_w, placeholder_int32, L16_ix_wq_b, L16_ix_wq_b_s, L16_ix_wproj_t,
    L16_ix_wk, L16_ix_knorm_w, L16_clen, L16_etable, placeholder_uint8, L16_gate_w_t, L16_gate_b,
    L16_expert_lo, L16_w1, L16_s1, L16_w3, L16_s3, L16_w2, L16_s2, L16_sh_w1, L16_sh_s1, L16_sh_w3,
    L16_sh_s3, L16_sh_w2, L16_sh_s2, L16_groups_attn, L16_groups_world, L16_flags, L17_hca_fn_t,
    L17_hca_base, L17_hca_scale, L17_attn_norm_w, L17_hcf_fn_t, L17_hcf_base, L17_hcf_scale,
    L17_ffn_norm_w, L17_wq_a, L17_wq_a_s, L17_q_norm_w, L17_wq_b, L17_wq_b_s, L17_wkv, L17_wkv_s,
    L17_kv_norm_w, L17_wo_a, L17_wo_a_s, L17_wo_b, L17_wo_b_s, L17_attn_sink, L17_window_kv_cache,
    L17_cm_wkv_t, L17_cm_wgate_t, L17_cm_norm_w, L17_ix_wproj_t, L17_ix_knorm_w, L17_gate_w_t,
    L17_gate_b, L17_expert_lo, L17_w1, L17_s1, L17_w3, L17_s3, L17_w2, L17_s2, L17_sh_w1,
    L17_sh_s1, L17_sh_w3, L17_sh_s3, L17_sh_w2, L17_sh_s2, L18_hca_fn_t, L18_hca_base,
    L18_hca_scale, L18_attn_norm_w, L18_hcf_fn_t, L18_hcf_base, L18_hcf_scale, L18_ffn_norm_w,
    L18_wq_a, L18_wq_a_s, L18_q_norm_w, L18_wq_b, L18_wq_b_s, L18_wkv, L18_wkv_s, L18_kv_norm_w,
    L18_wo_a, L18_wo_a_s, L18_wo_b, L18_wo_b_s, L18_attn_sink, L18_window_kv_cache, L18_cm_wkv_t,
    L18_cm_wgate_t, L18_cm_norm_w, L18_ix_wproj_t, L18_ix_knorm_w, L18_gate_w_t, L18_gate_b,
    L18_expert_lo, L18_w1, L18_s1, L18_w3, L18_s3, L18_w2, L18_s2, L18_sh_w1, L18_sh_s1, L18_sh_w3,
    L18_sh_s3, L18_sh_w2, L18_sh_s2, L19_hca_fn_t, L19_hca_base, L19_hca_scale, L19_attn_norm_w,
    L19_hcf_fn_t, L19_hcf_base, L19_hcf_scale, L19_ffn_norm_w, L19_wq_a, L19_wq_a_s, L19_q_norm_w,
    L19_wq_b, L19_wq_b_s, L19_wkv, L19_wkv_s, L19_kv_norm_w, L19_wo_a, L19_wo_a_s, L19_wo_b,
    L19_wo_b_s, L19_attn_sink, L19_window_kv_cache, L19_cm_wkv_t, L19_cm_wgate_t, L19_cm_norm_w,
    L19_ix_wproj_t, L19_ix_knorm_w, L19_gate_w_t, L19_gate_b, L19_expert_lo, L19_w1, L19_s1,
    L19_w3, L19_s3, L19_w2, L19_s2, L19_sh_w1, L19_sh_s1, L19_sh_w3, L19_sh_s3, L19_sh_w2,
    L19_sh_s2, L20_hca_fn_t, L20_hca_base, L20_hca_scale, L20_attn_norm_w, L20_hcf_fn_t,
    L20_hcf_base, L20_hcf_scale, L20_ffn_norm_w, L20_wq_a, L20_wq_a_s, L20_q_norm_w, L20_wq_b,
    L20_wq_b_s, L20_wkv, L20_wkv_s, L20_kv_norm_w, L20_wo_a, L20_wo_a_s, L20_wo_b, L20_wo_b_s,
    L20_attn_sink, L20_row_pos, L20_write_slots, L20_win_slots, L20_cmp_slots, L20_twin, L20_wv,
    L20_cbase, L20_window_kv_cache, L20_compress_kv_cache, L20_index_k_cache, L20_cm_wkv,
    L20_cm_wkv_t, L20_cm_norm_w, L20_grp_pos, L20_grp_mslots, L20_grp_wslots, L20_ix_wq_b,
    L20_ix_wq_b_s, L20_ix_wproj_t, L20_ix_wk, L20_ix_knorm_w, L20_clen, L20_gate_w_t, L20_gate_b,
    L20_expert_lo, L20_w1, L20_s1, L20_w3, L20_s3, L20_w2, L20_s2, L20_sh_w1, L20_sh_s1, L20_sh_w3,
    L20_sh_s3, L20_sh_w2, L20_sh_s2, L20_flags, L21_hca_fn_t, L21_hca_base, L21_hca_scale,
    L21_attn_norm_w, L21_hcf_fn_t, L21_hcf_base, L21_hcf_scale, L21_ffn_norm_w, L21_wq_a,
    L21_wq_a_s, L21_q_norm_w, L21_wq_b, L21_wq_b_s, L21_wkv, L21_wkv_s, L21_kv_norm_w, L21_wo_a,
    L21_wo_a_s, L21_wo_b, L21_wo_b_s, L21_attn_sink, L21_window_kv_cache, L21_cm_wkv_t,
    L21_cm_norm_w, L21_ix_wproj_t, L21_ix_knorm_w, L21_gate_w_t, L21_gate_b, L21_expert_lo, L21_w1,
    L21_s1, L21_w3, L21_s3, L21_w2, L21_s2, L21_sh_w1, L21_sh_s1, L21_sh_w3, L21_sh_s3, L21_sh_w2,
    L21_sh_s2, L21_flags, L22_hca_fn_t, L22_hca_base, L22_hca_scale, L22_attn_norm_w, L22_hcf_fn_t,
    L22_hcf_base, L22_hcf_scale, L22_ffn_norm_w, L22_wq_a, L22_wq_a_s, L22_q_norm_w, L22_wq_b,
    L22_wq_b_s, L22_wkv, L22_wkv_s, L22_kv_norm_w, L22_wo_a, L22_wo_a_s, L22_wo_b, L22_wo_b_s,
    L22_attn_sink, L22_window_kv_cache, L22_cm_wkv_t, L22_cm_norm_w, L22_ix_wproj_t,
    L22_ix_knorm_w, L22_gate_w_t, L22_gate_b, L22_expert_lo, L22_w1, L22_s1, L22_w3, L22_s3,
    L22_w2, L22_s2, L22_sh_w1, L22_sh_s1, L22_sh_w3, L22_sh_s3, L22_sh_w2, L22_sh_s2, L23_hca_fn_t,
    L23_hca_base, L23_hca_scale, L23_attn_norm_w, L23_hcf_fn_t, L23_hcf_base, L23_hcf_scale,
    L23_ffn_norm_w, L23_wq_a, L23_wq_a_s, L23_q_norm_w, L23_wq_b, L23_wq_b_s, L23_wkv, L23_wkv_s,
    L23_kv_norm_w, L23_wo_a, L23_wo_a_s, L23_wo_b, L23_wo_b_s, L23_attn_sink, L23_window_kv_cache,
    L23_cm_wkv_t, L23_cm_norm_w, L23_ix_wproj_t, L23_ix_knorm_w, L23_gate_w_t, L23_gate_b,
    L23_expert_lo, L23_w1, L23_s1, L23_w3, L23_s3, L23_w2, L23_s2, L23_sh_w1, L23_sh_s1, L23_sh_w3,
    L23_sh_s3, L23_sh_w2, L23_sh_s2
):
    # the slot / index tables the members share (built by the first member that needs them)
    share = {}
    # r2 members (L00_L07's): the slot / index tables (`r2_decode.SHARE_TAB`), the previous
    # block's output tiles in SBUF (`XS_SHARE`), and each r2 block's q / kv projection
    # weights loaded by the r2 block before it, after that block's world all-reduce trigger
    # (`r2_decode.prefetch_proj`)
    r2_decode.SHARE_TAB = {}
    r2_decode.XS_SHARE = {}

    def prefetch(wq_a, wq_a_s, wkv, wkv_s, wq_b=None, wq_b_s=None):
        if not r2_decode.PREFETCH_NEXT_PROJ or L16_stream.shape[0] > r2_decode.PREFETCH_MAX_T:
            return None

        def f(pid, after):
            r2_decode.PRE_PROJ = (wq_a, r2_decode.prefetch_proj(pid, wq_a, wq_a_s, wkv, wkv_s, after=after,
                                                                wq_b=wq_b, wq_b_s=wq_b_s))
        return f

    r2_decode.AFTER_WORLD_AR = prefetch(L17_wq_a, L17_wq_a_s, L17_wkv, L17_wkv_s, L17_wq_b, L17_wq_b_s)
    # L16 (r2)
    with member(0, r2_decode):
        o0 = r2_decode.layer_decode_r2_tp64.func(
            L16_stream, L16_pre_mix, L16_hca_fn_t, L16_hca_base, L16_hca_scale, L16_attn_norm_w,
            L16_hcf_fn_t, L16_hcf_base, L16_hcf_scale, L16_ffn_norm_w, L16_wq_a, L16_wq_a_s,
            L16_q_norm_w, L16_wq_b, L16_wq_b_s, L16_wkv, L16_wkv_s, L16_kv_norm_w, L16_wo_a,
            L16_wo_a_s, L16_wo_b, L16_wo_b_s, L16_attn_sink, L16_freqs, L16_row_pos,
            L16_write_slots, L16_win_slots, L16_cmp_slots, L16_twin, L16_wv, L16_cbase,
            L16_topk_in, L16_window_kv_cache, L16_compress_kv_cache, placeholder_float32,
            placeholder_float32, L16_index_k_cache, L16_cm_wkv, L16_cm_wkv_t, L16_cm_wgate_t,
            L16_cm_norm_w, placeholder_int32, placeholder_int32, placeholder_int32, L16_ix_wq_b,
            L16_ix_wq_b_s, L16_ix_wproj_t, L16_ix_wk, L16_ix_knorm_w, L16_clen, placeholder_int32,
            L16_etable, placeholder_int32, placeholder_uint8, placeholder_uint8,
            placeholder_float32, L16_gate_w_t, L16_gate_b, L16_expert_lo, L16_w1, L16_s1, L16_w3,
            L16_s3, L16_w2, L16_s2, L16_sh_w1, L16_sh_s1, L16_sh_w3, L16_sh_s3, L16_sh_w2,
            L16_sh_s2, L16_groups_attn, L16_groups_world, L16_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L18_wq_a, L18_wq_a_s, L18_wkv, L18_wkv_s, L18_wq_b, L18_wq_b_s)
    # L17 (r2)
    with member(1, r2_decode):
        o1 = r2_decode.layer_decode_r2_tp64.func(
            o0[0], o0[1], L17_hca_fn_t, L17_hca_base, L17_hca_scale, L17_attn_norm_w, L17_hcf_fn_t,
            L17_hcf_base, L17_hcf_scale, L17_ffn_norm_w, L17_wq_a, L17_wq_a_s, L17_q_norm_w,
            L17_wq_b, L17_wq_b_s, L17_wkv, L17_wkv_s, L17_kv_norm_w, L17_wo_a, L17_wo_a_s,
            L17_wo_b, L17_wo_b_s, L17_attn_sink, L16_freqs, L16_row_pos, L16_write_slots,
            L16_win_slots, L16_cmp_slots, L16_twin, L16_wv, L16_cbase, L16_topk_in,
            L17_window_kv_cache, L16_compress_kv_cache, placeholder_float32, placeholder_float32,
            L16_index_k_cache, L16_cm_wkv, L17_cm_wkv_t, L17_cm_wgate_t, L17_cm_norm_w,
            placeholder_int32, placeholder_int32, placeholder_int32, L16_ix_wq_b, L16_ix_wq_b_s,
            L17_ix_wproj_t, L16_ix_wk, L17_ix_knorm_w, L16_clen, placeholder_int32, L16_etable,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L17_gate_w_t, L17_gate_b, L17_expert_lo, L17_w1, L17_s1, L17_w3, L17_s3, L17_w2,
            L17_s2, L17_sh_w1, L17_sh_s1, L17_sh_w3, L17_sh_s3, L17_sh_w2, L17_sh_s2,
            L16_groups_attn, L16_groups_world, L16_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L19_wq_a, L19_wq_a_s, L19_wkv, L19_wkv_s, L19_wq_b, L19_wq_b_s)
    # L18 (r2)
    with member(2, r2_decode):
        o2 = r2_decode.layer_decode_r2_tp64.func(
            o1[0], o1[1], L18_hca_fn_t, L18_hca_base, L18_hca_scale, L18_attn_norm_w, L18_hcf_fn_t,
            L18_hcf_base, L18_hcf_scale, L18_ffn_norm_w, L18_wq_a, L18_wq_a_s, L18_q_norm_w,
            L18_wq_b, L18_wq_b_s, L18_wkv, L18_wkv_s, L18_kv_norm_w, L18_wo_a, L18_wo_a_s,
            L18_wo_b, L18_wo_b_s, L18_attn_sink, L16_freqs, L16_row_pos, L16_write_slots,
            L16_win_slots, L16_cmp_slots, L16_twin, L16_wv, L16_cbase, L16_topk_in,
            L18_window_kv_cache, L16_compress_kv_cache, placeholder_float32, placeholder_float32,
            L16_index_k_cache, L16_cm_wkv, L18_cm_wkv_t, L18_cm_wgate_t, L18_cm_norm_w,
            placeholder_int32, placeholder_int32, placeholder_int32, L16_ix_wq_b, L16_ix_wq_b_s,
            L18_ix_wproj_t, L16_ix_wk, L18_ix_knorm_w, L16_clen, placeholder_int32, L16_etable,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L18_gate_w_t, L18_gate_b, L18_expert_lo, L18_w1, L18_s1, L18_w3, L18_s3, L18_w2,
            L18_s2, L18_sh_w1, L18_sh_s1, L18_sh_w3, L18_sh_s3, L18_sh_w2, L18_sh_s2,
            L16_groups_attn, L16_groups_world, L16_flags
        )
    r2_decode.AFTER_WORLD_AR = None
    # L19 (r2)
    with member(3, r2_decode):
        o3 = r2_decode.layer_decode_r2_tp64.func(
            o2[0], o2[1], L19_hca_fn_t, L19_hca_base, L19_hca_scale, L19_attn_norm_w, L19_hcf_fn_t,
            L19_hcf_base, L19_hcf_scale, L19_ffn_norm_w, L19_wq_a, L19_wq_a_s, L19_q_norm_w,
            L19_wq_b, L19_wq_b_s, L19_wkv, L19_wkv_s, L19_kv_norm_w, L19_wo_a, L19_wo_a_s,
            L19_wo_b, L19_wo_b_s, L19_attn_sink, L16_freqs, L16_row_pos, L16_write_slots,
            L16_win_slots, L16_cmp_slots, L16_twin, L16_wv, L16_cbase, L16_topk_in,
            L19_window_kv_cache, L16_compress_kv_cache, placeholder_float32, placeholder_float32,
            L16_index_k_cache, L16_cm_wkv, L19_cm_wkv_t, L19_cm_wgate_t, L19_cm_norm_w,
            placeholder_int32, placeholder_int32, placeholder_int32, L16_ix_wq_b, L16_ix_wq_b_s,
            L19_ix_wproj_t, L16_ix_wk, L19_ix_knorm_w, L16_clen, placeholder_int32, L16_etable,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L19_gate_w_t, L19_gate_b, L19_expert_lo, L19_w1, L19_s1, L19_w3, L19_s3, L19_w2,
            L19_s2, L19_sh_w1, L19_sh_s1, L19_sh_w3, L19_sh_s3, L19_sh_w2, L19_sh_s2,
            L16_groups_attn, L16_groups_world, L16_flags
        )
    # L20 (r1_src_idx)
    with member(4, r1_src_idx_decode):
        o4 = r1_src_idx_decode.layer_decode_r1_src_idx_tp64.func(
            o3[0], o3[1], L20_hca_fn_t, L20_hca_base, L20_hca_scale, L20_attn_norm_w, L20_hcf_fn_t,
            L20_hcf_base, L20_hcf_scale, L20_ffn_norm_w, L20_wq_a, L20_wq_a_s, L20_q_norm_w,
            L20_wq_b, L20_wq_b_s, L20_wkv, L20_wkv_s, L20_kv_norm_w, L20_wo_a, L20_wo_a_s,
            L20_wo_b, L20_wo_b_s, L20_attn_sink, L16_freqs, L20_row_pos, L20_write_slots,
            L20_win_slots, L20_cmp_slots, L20_twin, L20_wv, L20_cbase, L16_topk_in,
            L20_window_kv_cache, L20_compress_kv_cache, placeholder_float32, placeholder_float32,
            L20_index_k_cache, L20_cm_wkv, L20_cm_wkv_t, placeholder_float32, L20_cm_norm_w,
            L20_grp_pos, L20_grp_mslots, L20_grp_wslots, L20_ix_wq_b, L20_ix_wq_b_s,
            L20_ix_wproj_t, L20_ix_wk, L20_ix_knorm_w, L20_clen, placeholder_int32, L16_etable,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L20_gate_w_t, L20_gate_b, L20_expert_lo, L20_w1, L20_s1, L20_w3, L20_s3, L20_w2,
            L20_s2, L20_sh_w1, L20_sh_s1, L20_sh_w3, L20_sh_s3, L20_sh_w2, L20_sh_s2,
            L16_groups_attn, L16_groups_world, L20_flags
        )
    # L21 (r1)
    with member(5, r1_decode):
        o5 = r1_decode.layer_decode_r1_tp64.func(
            o4[0], o4[1], L21_hca_fn_t, L21_hca_base, L21_hca_scale, L21_attn_norm_w, L21_hcf_fn_t,
            L21_hcf_base, L21_hcf_scale, L21_ffn_norm_w, L21_wq_a, L21_wq_a_s, L21_q_norm_w,
            L21_wq_b, L21_wq_b_s, L21_wkv, L21_wkv_s, L21_kv_norm_w, L21_wo_a, L21_wo_a_s,
            L21_wo_b, L21_wo_b_s, L21_attn_sink, L16_freqs, L20_row_pos, L20_write_slots,
            L20_win_slots, L20_cmp_slots, L20_twin, L20_wv, L20_cbase, o4[2], L21_window_kv_cache,
            L20_compress_kv_cache, placeholder_float32, placeholder_float32, L20_index_k_cache,
            L20_cm_wkv, L21_cm_wkv_t, placeholder_float32, L21_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L20_ix_wq_b, L20_ix_wq_b_s, L21_ix_wproj_t,
            L20_ix_wk, L21_ix_knorm_w, L20_clen, placeholder_int32, L16_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L21_gate_w_t, L21_gate_b,
            L21_expert_lo, L21_w1, L21_s1, L21_w3, L21_s3, L21_w2, L21_s2, L21_sh_w1, L21_sh_s1,
            L21_sh_w3, L21_sh_s3, L21_sh_w2, L21_sh_s2, L16_groups_attn, L16_groups_world,
            L21_flags,
            share=share,
        )
    # L22 (r1)
    with member(6, r1_decode):
        o6 = r1_decode.layer_decode_r1_tp64.func(
            o5[0], o5[1], L22_hca_fn_t, L22_hca_base, L22_hca_scale, L22_attn_norm_w, L22_hcf_fn_t,
            L22_hcf_base, L22_hcf_scale, L22_ffn_norm_w, L22_wq_a, L22_wq_a_s, L22_q_norm_w,
            L22_wq_b, L22_wq_b_s, L22_wkv, L22_wkv_s, L22_kv_norm_w, L22_wo_a, L22_wo_a_s,
            L22_wo_b, L22_wo_b_s, L22_attn_sink, L16_freqs, L20_row_pos, L20_write_slots,
            L20_win_slots, L20_cmp_slots, L20_twin, L20_wv, L20_cbase, o4[2], L22_window_kv_cache,
            L20_compress_kv_cache, placeholder_float32, placeholder_float32, L20_index_k_cache,
            L20_cm_wkv, L22_cm_wkv_t, placeholder_float32, L22_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L20_ix_wq_b, L20_ix_wq_b_s, L22_ix_wproj_t,
            L20_ix_wk, L22_ix_knorm_w, L20_clen, placeholder_int32, L16_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L22_gate_w_t, L22_gate_b,
            L22_expert_lo, L22_w1, L22_s1, L22_w3, L22_s3, L22_w2, L22_s2, L22_sh_w1, L22_sh_s1,
            L22_sh_w3, L22_sh_s3, L22_sh_w2, L22_sh_s2, L16_groups_attn, L16_groups_world,
            L21_flags,
            share=share,
        )
    # L23 (r1)
    with member(7, r1_decode):
        o7 = r1_decode.layer_decode_r1_tp64.func(
            o6[0], o6[1], L23_hca_fn_t, L23_hca_base, L23_hca_scale, L23_attn_norm_w, L23_hcf_fn_t,
            L23_hcf_base, L23_hcf_scale, L23_ffn_norm_w, L23_wq_a, L23_wq_a_s, L23_q_norm_w,
            L23_wq_b, L23_wq_b_s, L23_wkv, L23_wkv_s, L23_kv_norm_w, L23_wo_a, L23_wo_a_s,
            L23_wo_b, L23_wo_b_s, L23_attn_sink, L16_freqs, L20_row_pos, L20_write_slots,
            L20_win_slots, L20_cmp_slots, L20_twin, L20_wv, L20_cbase, o4[2], L23_window_kv_cache,
            L20_compress_kv_cache, placeholder_float32, placeholder_float32, L20_index_k_cache,
            L20_cm_wkv, L23_cm_wkv_t, placeholder_float32, L23_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L20_ix_wq_b, L20_ix_wq_b_s, L23_ix_wproj_t,
            L20_ix_wk, L23_ix_knorm_w, L20_clen, placeholder_int32, L16_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L23_gate_w_t, L23_gate_b,
            L23_expert_lo, L23_w1, L23_s1, L23_w3, L23_s3, L23_w2, L23_s2, L23_sh_w1, L23_sh_s1,
            L23_sh_w3, L23_sh_s3, L23_sh_w2, L23_sh_s2, L16_groups_attn, L16_groups_world,
            L21_flags,
            share=share,
        )
    r2_decode.SHARE_TAB = None
    r2_decode.XS_SHARE = None
    r2_decode.AFTER_WORLD_AR = None
    r2_decode.PRE_PROJ = None
    return o7[0], o7[1], o4[2]
