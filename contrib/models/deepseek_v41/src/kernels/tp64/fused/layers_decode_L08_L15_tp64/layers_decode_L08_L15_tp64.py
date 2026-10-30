# SPDX-License-Identifier: Apache-2.0
"""`layers_decode_L08_L15_tp64`: one launch of a DeepSeek-V4.1-Flash decode step (tp64, LNC=2) running
blocks 8-15, in order:

    L08 r2_src_idx, L09 r2, L10 r2, L11 r2, L12 r2, L13 r2, L14 r2_src_idx_e, L15 r2

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

Written out from the model's fused plan (`mk_kernels.fused`) and optimized as a whole in an
AutoHelix loop (workspace ~/ds41_fused_L08_L15: its `git log` lists the changes, README.md the setup).
`mk_kernels.optimized_fused` runs it in place of the generated launch when the model's plan
is plan.json.
"""
import nki

from .fused_member import member
from .r2_src_idx_decode import r2_src_idx_decode
from .r2_decode import r2_decode
from .r2_src_idx_e_decode import r2_src_idx_e_decode


@nki.jit
def layers_decode_L08_L15_tp64(
    L08_stream, L08_pre_mix, L08_hca_fn_t, L08_hca_base, L08_hca_scale, L08_attn_norm_w,
    L08_hcf_fn_t, L08_hcf_base, L08_hcf_scale, L08_ffn_norm_w, L08_wq_a, L08_wq_a_s, L08_q_norm_w,
    L08_wq_b, L08_wq_b_s, L08_wkv, L08_wkv_s, L08_kv_norm_w, L08_wo_a, L08_wo_a_s, L08_wo_b,
    L08_wo_b_s, L08_attn_sink, L08_freqs, L08_row_pos, L08_write_slots, L08_win_slots,
    L08_cmp_slots, L08_twin, L08_wv, L08_cbase, L08_topk_in, L08_window_kv_cache,
    L08_compress_kv_cache, L08_pool_kv, L08_pool_score, L08_index_k_cache, L08_cm_wkv,
    L08_cm_wkv_t, L08_cm_wgate_t, L08_cm_norm_w, L08_grp_pos, L08_grp_mslots, L08_grp_wslots,
    L08_ix_wq_b, L08_ix_wq_b_s, L08_ix_wproj_t, L08_ix_wk, L08_ix_knorm_w, L08_clen,
    placeholder_int32, L08_etable, placeholder_uint8, placeholder_float32, L08_gate_w_t,
    L08_gate_b, L08_expert_lo, L08_w1, L08_s1, L08_w3, L08_s3, L08_w2, L08_s2, L08_sh_w1,
    L08_sh_s1, L08_sh_w3, L08_sh_s3, L08_sh_w2, L08_sh_s2, L08_groups_attn, L08_groups_world,
    L08_flags, L09_hca_fn_t, L09_hca_base, L09_hca_scale, L09_attn_norm_w, L09_hcf_fn_t,
    L09_hcf_base, L09_hcf_scale, L09_ffn_norm_w, L09_wq_a, L09_wq_a_s, L09_q_norm_w, L09_wq_b,
    L09_wq_b_s, L09_wkv, L09_wkv_s, L09_kv_norm_w, L09_wo_a, L09_wo_a_s, L09_wo_b, L09_wo_b_s,
    L09_attn_sink, L09_window_kv_cache, L09_cm_wkv_t, L09_cm_wgate_t, L09_cm_norm_w,
    L09_ix_wproj_t, L09_ix_knorm_w, L09_gate_w_t, L09_gate_b, L09_expert_lo, L09_w1, L09_s1,
    L09_w3, L09_s3, L09_w2, L09_s2, L09_sh_w1, L09_sh_s1, L09_sh_w3, L09_sh_s3, L09_sh_w2,
    L09_sh_s2, L09_flags, L10_hca_fn_t, L10_hca_base, L10_hca_scale, L10_attn_norm_w, L10_hcf_fn_t,
    L10_hcf_base, L10_hcf_scale, L10_ffn_norm_w, L10_wq_a, L10_wq_a_s, L10_q_norm_w, L10_wq_b,
    L10_wq_b_s, L10_wkv, L10_wkv_s, L10_kv_norm_w, L10_wo_a, L10_wo_a_s, L10_wo_b, L10_wo_b_s,
    L10_attn_sink, L10_window_kv_cache, L10_cm_wkv_t, L10_cm_wgate_t, L10_cm_norm_w,
    L10_ix_wproj_t, L10_ix_knorm_w, L10_gate_w_t, L10_gate_b, L10_expert_lo, L10_w1, L10_s1,
    L10_w3, L10_s3, L10_w2, L10_s2, L10_sh_w1, L10_sh_s1, L10_sh_w3, L10_sh_s3, L10_sh_w2,
    L10_sh_s2, L11_hca_fn_t, L11_hca_base, L11_hca_scale, L11_attn_norm_w, L11_hcf_fn_t,
    L11_hcf_base, L11_hcf_scale, L11_ffn_norm_w, L11_wq_a, L11_wq_a_s, L11_q_norm_w, L11_wq_b,
    L11_wq_b_s, L11_wkv, L11_wkv_s, L11_kv_norm_w, L11_wo_a, L11_wo_a_s, L11_wo_b, L11_wo_b_s,
    L11_attn_sink, L11_window_kv_cache, L11_cm_wkv_t, L11_cm_wgate_t, L11_cm_norm_w,
    L11_ix_wproj_t, L11_ix_knorm_w, L11_gate_w_t, L11_gate_b, L11_expert_lo, L11_w1, L11_s1,
    L11_w3, L11_s3, L11_w2, L11_s2, L11_sh_w1, L11_sh_s1, L11_sh_w3, L11_sh_s3, L11_sh_w2,
    L11_sh_s2, L12_hca_fn_t, L12_hca_base, L12_hca_scale, L12_attn_norm_w, L12_hcf_fn_t,
    L12_hcf_base, L12_hcf_scale, L12_ffn_norm_w, L12_wq_a, L12_wq_a_s, L12_q_norm_w, L12_wq_b,
    L12_wq_b_s, L12_wkv, L12_wkv_s, L12_kv_norm_w, L12_wo_a, L12_wo_a_s, L12_wo_b, L12_wo_b_s,
    L12_attn_sink, L12_window_kv_cache, L12_cm_wkv_t, L12_cm_wgate_t, L12_cm_norm_w,
    L12_ix_wproj_t, L12_ix_knorm_w, L12_gate_w_t, L12_gate_b, L12_expert_lo, L12_w1, L12_s1,
    L12_w3, L12_s3, L12_w2, L12_s2, L12_sh_w1, L12_sh_s1, L12_sh_w3, L12_sh_s3, L12_sh_w2,
    L12_sh_s2, L13_hca_fn_t, L13_hca_base, L13_hca_scale, L13_attn_norm_w, L13_hcf_fn_t,
    L13_hcf_base, L13_hcf_scale, L13_ffn_norm_w, L13_wq_a, L13_wq_a_s, L13_q_norm_w, L13_wq_b,
    L13_wq_b_s, L13_wkv, L13_wkv_s, L13_kv_norm_w, L13_wo_a, L13_wo_a_s, L13_wo_b, L13_wo_b_s,
    L13_attn_sink, L13_window_kv_cache, L13_cm_wkv_t, L13_cm_wgate_t, L13_cm_norm_w,
    L13_ix_wproj_t, L13_ix_knorm_w, L13_gate_w_t, L13_gate_b, L13_expert_lo, L13_w1, L13_s1,
    L13_w3, L13_s3, L13_w2, L13_s2, L13_sh_w1, L13_sh_s1, L13_sh_w3, L13_sh_s3, L13_sh_w2,
    L13_sh_s2, L14_hca_fn_t, L14_hca_base, L14_hca_scale, L14_attn_norm_w, L14_hcf_fn_t,
    L14_hcf_base, L14_hcf_scale, L14_ffn_norm_w, L14_wq_a, L14_wq_a_s, L14_q_norm_w, L14_wq_b,
    L14_wq_b_s, L14_wkv, L14_wkv_s, L14_kv_norm_w, L14_wo_a, L14_wo_a_s, L14_wo_b, L14_wo_b_s,
    L14_attn_sink, L14_window_kv_cache, L14_compress_kv_cache, L14_pool_kv, L14_pool_score,
    L14_index_k_cache, L14_cm_wkv, L14_cm_wkv_t, L14_cm_wgate_t, L14_cm_norm_w, L14_ix_wq_b,
    L14_ix_wq_b_s, L14_ix_wproj_t, L14_ix_wk, L14_ix_knorm_w, L14_hash_ids, L14_etable,
    L14_erow_start, L14_e_wkv, L14_e_wkv_s, L14_e_qk, L14_gate_w_t, L14_gate_b, L14_expert_lo,
    L14_w1, L14_s1, L14_w3, L14_s3, L14_w2, L14_s2, L14_sh_w1, L14_sh_s1, L14_sh_w3, L14_sh_s3,
    L14_sh_w2, L14_sh_s2, L14_flags, L15_hca_fn_t, L15_hca_base, L15_hca_scale, L15_attn_norm_w,
    L15_hcf_fn_t, L15_hcf_base, L15_hcf_scale, L15_ffn_norm_w, L15_wq_a, L15_wq_a_s, L15_q_norm_w,
    L15_wq_b, L15_wq_b_s, L15_wkv, L15_wkv_s, L15_kv_norm_w, L15_wo_a, L15_wo_a_s, L15_wo_b,
    L15_wo_b_s, L15_attn_sink, L15_window_kv_cache, L15_cm_wkv_t, L15_cm_wgate_t, L15_cm_norm_w,
    L15_ix_wproj_t, L15_ix_knorm_w, L15_gate_w_t, L15_gate_b, L15_expert_lo, L15_w1, L15_s1,
    L15_w3, L15_s3, L15_w2, L15_s2, L15_sh_w1, L15_sh_s1, L15_sh_w3, L15_sh_s3, L15_sh_w2,
    L15_sh_s2
):
    # the slot / index tables an index source shares with the r2 blocks after it
    share = {}
    # L08 (r2_src_idx)
    with member(0, r2_src_idx_decode):
        o0 = r2_src_idx_decode.layer_decode_r2_src_idx_tp64.func(
            L08_stream, L08_pre_mix, L08_hca_fn_t, L08_hca_base, L08_hca_scale, L08_attn_norm_w,
            L08_hcf_fn_t, L08_hcf_base, L08_hcf_scale, L08_ffn_norm_w, L08_wq_a, L08_wq_a_s,
            L08_q_norm_w, L08_wq_b, L08_wq_b_s, L08_wkv, L08_wkv_s, L08_kv_norm_w, L08_wo_a,
            L08_wo_a_s, L08_wo_b, L08_wo_b_s, L08_attn_sink, L08_freqs, L08_row_pos,
            L08_write_slots, L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase,
            L08_topk_in, L08_window_kv_cache, L08_compress_kv_cache, L08_pool_kv, L08_pool_score,
            L08_index_k_cache, L08_cm_wkv, L08_cm_wkv_t, L08_cm_wgate_t, L08_cm_norm_w,
            L08_grp_pos, L08_grp_mslots, L08_grp_wslots, L08_ix_wq_b, L08_ix_wq_b_s,
            L08_ix_wproj_t, L08_ix_wk, L08_ix_knorm_w, L08_clen, placeholder_int32, L08_etable,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L08_gate_w_t, L08_gate_b, L08_expert_lo, L08_w1, L08_s1, L08_w3, L08_s3, L08_w2,
            L08_s2, L08_sh_w1, L08_sh_s1, L08_sh_w3, L08_sh_s3, L08_sh_w2, L08_sh_s2,
            L08_groups_attn, L08_groups_world, L08_flags, share=share
        )
    # L09 (r2)
    with member(1, r2_decode):
        o1 = r2_decode.layer_decode_r2_tp64.func(
            o0[0], o0[1], L09_hca_fn_t, L09_hca_base, L09_hca_scale, L09_attn_norm_w, L09_hcf_fn_t,
            L09_hcf_base, L09_hcf_scale, L09_ffn_norm_w, L09_wq_a, L09_wq_a_s, L09_q_norm_w,
            L09_wq_b, L09_wq_b_s, L09_wkv, L09_wkv_s, L09_kv_norm_w, L09_wo_a, L09_wo_a_s,
            L09_wo_b, L09_wo_b_s, L09_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L09_window_kv_cache,
            L08_compress_kv_cache, placeholder_float32, placeholder_float32, L08_index_k_cache,
            L08_cm_wkv, L09_cm_wkv_t, L09_cm_wgate_t, L09_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L08_ix_wq_b, L08_ix_wq_b_s, L09_ix_wproj_t,
            L08_ix_wk, L09_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L09_gate_w_t, L09_gate_b,
            L09_expert_lo, L09_w1, L09_s1, L09_w3, L09_s3, L09_w2, L09_s2, L09_sh_w1, L09_sh_s1,
            L09_sh_w3, L09_sh_s3, L09_sh_w2, L09_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    # L10 (r2)
    with member(2, r2_decode):
        o2 = r2_decode.layer_decode_r2_tp64.func(
            o1[0], o1[1], L10_hca_fn_t, L10_hca_base, L10_hca_scale, L10_attn_norm_w, L10_hcf_fn_t,
            L10_hcf_base, L10_hcf_scale, L10_ffn_norm_w, L10_wq_a, L10_wq_a_s, L10_q_norm_w,
            L10_wq_b, L10_wq_b_s, L10_wkv, L10_wkv_s, L10_kv_norm_w, L10_wo_a, L10_wo_a_s,
            L10_wo_b, L10_wo_b_s, L10_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L10_window_kv_cache,
            L08_compress_kv_cache, placeholder_float32, placeholder_float32, L08_index_k_cache,
            L08_cm_wkv, L10_cm_wkv_t, L10_cm_wgate_t, L10_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L08_ix_wq_b, L08_ix_wq_b_s, L10_ix_wproj_t,
            L08_ix_wk, L10_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L10_gate_w_t, L10_gate_b,
            L10_expert_lo, L10_w1, L10_s1, L10_w3, L10_s3, L10_w2, L10_s2, L10_sh_w1, L10_sh_s1,
            L10_sh_w3, L10_sh_s3, L10_sh_w2, L10_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    # L11 (r2)
    with member(3, r2_decode):
        o3 = r2_decode.layer_decode_r2_tp64.func(
            o2[0], o2[1], L11_hca_fn_t, L11_hca_base, L11_hca_scale, L11_attn_norm_w, L11_hcf_fn_t,
            L11_hcf_base, L11_hcf_scale, L11_ffn_norm_w, L11_wq_a, L11_wq_a_s, L11_q_norm_w,
            L11_wq_b, L11_wq_b_s, L11_wkv, L11_wkv_s, L11_kv_norm_w, L11_wo_a, L11_wo_a_s,
            L11_wo_b, L11_wo_b_s, L11_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L11_window_kv_cache,
            L08_compress_kv_cache, placeholder_float32, placeholder_float32, L08_index_k_cache,
            L08_cm_wkv, L11_cm_wkv_t, L11_cm_wgate_t, L11_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L08_ix_wq_b, L08_ix_wq_b_s, L11_ix_wproj_t,
            L08_ix_wk, L11_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L11_gate_w_t, L11_gate_b,
            L11_expert_lo, L11_w1, L11_s1, L11_w3, L11_s3, L11_w2, L11_s2, L11_sh_w1, L11_sh_s1,
            L11_sh_w3, L11_sh_s3, L11_sh_w2, L11_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    # L12 (r2)
    with member(4, r2_decode):
        o4 = r2_decode.layer_decode_r2_tp64.func(
            o3[0], o3[1], L12_hca_fn_t, L12_hca_base, L12_hca_scale, L12_attn_norm_w, L12_hcf_fn_t,
            L12_hcf_base, L12_hcf_scale, L12_ffn_norm_w, L12_wq_a, L12_wq_a_s, L12_q_norm_w,
            L12_wq_b, L12_wq_b_s, L12_wkv, L12_wkv_s, L12_kv_norm_w, L12_wo_a, L12_wo_a_s,
            L12_wo_b, L12_wo_b_s, L12_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L12_window_kv_cache,
            L08_compress_kv_cache, placeholder_float32, placeholder_float32, L08_index_k_cache,
            L08_cm_wkv, L12_cm_wkv_t, L12_cm_wgate_t, L12_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L08_ix_wq_b, L08_ix_wq_b_s, L12_ix_wproj_t,
            L08_ix_wk, L12_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L12_gate_w_t, L12_gate_b,
            L12_expert_lo, L12_w1, L12_s1, L12_w3, L12_s3, L12_w2, L12_s2, L12_sh_w1, L12_sh_s1,
            L12_sh_w3, L12_sh_s3, L12_sh_w2, L12_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    # L13 (r2)
    with member(5, r2_decode):
        o5 = r2_decode.layer_decode_r2_tp64.func(
            o4[0], o4[1], L13_hca_fn_t, L13_hca_base, L13_hca_scale, L13_attn_norm_w, L13_hcf_fn_t,
            L13_hcf_base, L13_hcf_scale, L13_ffn_norm_w, L13_wq_a, L13_wq_a_s, L13_q_norm_w,
            L13_wq_b, L13_wq_b_s, L13_wkv, L13_wkv_s, L13_kv_norm_w, L13_wo_a, L13_wo_a_s,
            L13_wo_b, L13_wo_b_s, L13_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L13_window_kv_cache,
            L08_compress_kv_cache, placeholder_float32, placeholder_float32, L08_index_k_cache,
            L08_cm_wkv, L13_cm_wkv_t, L13_cm_wgate_t, L13_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L08_ix_wq_b, L08_ix_wq_b_s, L13_ix_wproj_t,
            L08_ix_wk, L13_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L13_gate_w_t, L13_gate_b,
            L13_expert_lo, L13_w1, L13_s1, L13_w3, L13_s3, L13_w2, L13_s2, L13_sh_w1, L13_sh_s1,
            L13_sh_w3, L13_sh_s3, L13_sh_w2, L13_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    # L14 (r2_src_idx_e)
    with member(6, r2_src_idx_e_decode):
        o6 = r2_src_idx_e_decode.layer_decode_r2_src_idx_e_tp64.func(
            o5[0], o5[1], L14_hca_fn_t, L14_hca_base, L14_hca_scale, L14_attn_norm_w, L14_hcf_fn_t,
            L14_hcf_base, L14_hcf_scale, L14_ffn_norm_w, L14_wq_a, L14_wq_a_s, L14_q_norm_w,
            L14_wq_b, L14_wq_b_s, L14_wkv, L14_wkv_s, L14_kv_norm_w, L14_wo_a, L14_wo_a_s,
            L14_wo_b, L14_wo_b_s, L14_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o0[2], L14_window_kv_cache,
            L14_compress_kv_cache, L14_pool_kv, L14_pool_score, L14_index_k_cache, L14_cm_wkv,
            L14_cm_wkv_t, L14_cm_wgate_t, L14_cm_norm_w, L08_grp_pos, L08_grp_mslots,
            L08_grp_wslots, L14_ix_wq_b, L14_ix_wq_b_s, L14_ix_wproj_t, L14_ix_wk, L14_ix_knorm_w,
            L08_clen, L14_hash_ids, L14_etable, L14_erow_start, L14_e_wkv, L14_e_wkv_s, L14_e_qk,
            L14_gate_w_t, L14_gate_b, L14_expert_lo, L14_w1, L14_s1, L14_w3, L14_s3, L14_w2,
            L14_s2, L14_sh_w1, L14_sh_s1, L14_sh_w3, L14_sh_s3, L14_sh_w2, L14_sh_s2,
            L08_groups_attn, L08_groups_world, L14_flags, share=share
        )
    # L15 (r2)
    with member(7, r2_decode):
        o7 = r2_decode.layer_decode_r2_tp64.func(
            o6[0], o6[1], L15_hca_fn_t, L15_hca_base, L15_hca_scale, L15_attn_norm_w, L15_hcf_fn_t,
            L15_hcf_base, L15_hcf_scale, L15_ffn_norm_w, L15_wq_a, L15_wq_a_s, L15_q_norm_w,
            L15_wq_b, L15_wq_b_s, L15_wkv, L15_wkv_s, L15_kv_norm_w, L15_wo_a, L15_wo_a_s,
            L15_wo_b, L15_wo_b_s, L15_attn_sink, L08_freqs, L08_row_pos, L08_write_slots,
            L08_win_slots, L08_cmp_slots, L08_twin, L08_wv, L08_cbase, o6[2], L15_window_kv_cache,
            L14_compress_kv_cache, placeholder_float32, placeholder_float32, L14_index_k_cache,
            L14_cm_wkv, L15_cm_wkv_t, L15_cm_wgate_t, L15_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L14_ix_wq_b, L14_ix_wq_b_s, L15_ix_wproj_t,
            L14_ix_wk, L15_ix_knorm_w, L08_clen, placeholder_int32, L08_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L15_gate_w_t, L15_gate_b,
            L15_expert_lo, L15_w1, L15_s1, L15_w3, L15_s3, L15_w2, L15_s2, L15_sh_w1, L15_sh_s1,
            L15_sh_w3, L15_sh_s3, L15_sh_w2, L15_sh_s2, L08_groups_attn, L08_groups_world,
            L09_flags, share=share
        )
    return o7[0], o7[1], o6[2]
