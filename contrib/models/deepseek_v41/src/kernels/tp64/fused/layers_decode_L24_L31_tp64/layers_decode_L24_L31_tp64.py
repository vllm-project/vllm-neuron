# SPDX-License-Identifier: Apache-2.0
"""`layers_decode_L24_L31_tp64`: one launch of a DeepSeek-V4.1-Flash decode step (tp64, LNC=2) running
blocks 24-31, in order:

    L24 r1_idx, L25 r1, L26 r1, L27 r1, L28 r1_idx, L29 r1, L30 r1, L31 r1

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
AutoHelix loop (workspace ~/ds41_fused_L24_L31: its `git log` lists the changes, README.md the setup).
`mk_kernels.optimized_fused` runs it in place of the generated launch when the model's plan
is plan.json.
"""
import nki

from .fused_member import member
from .r1_idx_decode import r1_idx_decode
from .r1_decode import r1_decode


@nki.jit
def layers_decode_L24_L31_tp64(
    L24_stream, L24_pre_mix, L24_hca_fn_t, L24_hca_base, L24_hca_scale, L24_attn_norm_w,
    L24_hcf_fn_t, L24_hcf_base, L24_hcf_scale, L24_ffn_norm_w, L24_wq_a, L24_wq_a_s, L24_q_norm_w,
    L24_wq_b, L24_wq_b_s, L24_wkv, L24_wkv_s, L24_kv_norm_w, L24_wo_a, L24_wo_a_s, L24_wo_b,
    L24_wo_b_s, L24_attn_sink, L24_freqs, L24_row_pos, L24_write_slots, L24_win_slots,
    L24_cmp_slots, L24_twin, L24_wv, L24_cbase, L24_topk_in, L24_window_kv_cache,
    L24_compress_kv_cache, placeholder_float32, L24_index_k_cache, L24_cm_wkv, L24_cm_wkv_t,
    L24_cm_norm_w, placeholder_int32, L24_ix_wq_b, L24_ix_wq_b_s, L24_ix_wproj_t, L24_ix_wk,
    L24_ix_knorm_w, L24_clen, L24_etable, placeholder_uint8, L24_gate_w_t, L24_gate_b,
    L24_expert_lo, L24_w1, L24_s1, L24_w3, L24_s3, L24_w2, L24_s2, L24_sh_w1, L24_sh_s1, L24_sh_w3,
    L24_sh_s3, L24_sh_w2, L24_sh_s2, L24_groups_attn, L24_groups_world, L24_flags, L25_hca_fn_t,
    L25_hca_base, L25_hca_scale, L25_attn_norm_w, L25_hcf_fn_t, L25_hcf_base, L25_hcf_scale,
    L25_ffn_norm_w, L25_wq_a, L25_wq_a_s, L25_q_norm_w, L25_wq_b, L25_wq_b_s, L25_wkv, L25_wkv_s,
    L25_kv_norm_w, L25_wo_a, L25_wo_a_s, L25_wo_b, L25_wo_b_s, L25_attn_sink, L25_window_kv_cache,
    L25_cm_wkv_t, L25_cm_norm_w, L25_ix_wproj_t, L25_ix_knorm_w, L25_gate_w_t, L25_gate_b,
    L25_expert_lo, L25_w1, L25_s1, L25_w3, L25_s3, L25_w2, L25_s2, L25_sh_w1, L25_sh_s1, L25_sh_w3,
    L25_sh_s3, L25_sh_w2, L25_sh_s2, L25_flags, L26_hca_fn_t, L26_hca_base, L26_hca_scale,
    L26_attn_norm_w, L26_hcf_fn_t, L26_hcf_base, L26_hcf_scale, L26_ffn_norm_w, L26_wq_a,
    L26_wq_a_s, L26_q_norm_w, L26_wq_b, L26_wq_b_s, L26_wkv, L26_wkv_s, L26_kv_norm_w, L26_wo_a,
    L26_wo_a_s, L26_wo_b, L26_wo_b_s, L26_attn_sink, L26_window_kv_cache, L26_cm_wkv_t,
    L26_cm_norm_w, L26_ix_wproj_t, L26_ix_knorm_w, L26_gate_w_t, L26_gate_b, L26_expert_lo, L26_w1,
    L26_s1, L26_w3, L26_s3, L26_w2, L26_s2, L26_sh_w1, L26_sh_s1, L26_sh_w3, L26_sh_s3, L26_sh_w2,
    L26_sh_s2, L27_hca_fn_t, L27_hca_base, L27_hca_scale, L27_attn_norm_w, L27_hcf_fn_t,
    L27_hcf_base, L27_hcf_scale, L27_ffn_norm_w, L27_wq_a, L27_wq_a_s, L27_q_norm_w, L27_wq_b,
    L27_wq_b_s, L27_wkv, L27_wkv_s, L27_kv_norm_w, L27_wo_a, L27_wo_a_s, L27_wo_b, L27_wo_b_s,
    L27_attn_sink, L27_window_kv_cache, L27_cm_wkv_t, L27_cm_norm_w, L27_ix_wproj_t,
    L27_ix_knorm_w, L27_gate_w_t, L27_gate_b, L27_expert_lo, L27_w1, L27_s1, L27_w3, L27_s3,
    L27_w2, L27_s2, L27_sh_w1, L27_sh_s1, L27_sh_w3, L27_sh_s3, L27_sh_w2, L27_sh_s2, L28_hca_fn_t,
    L28_hca_base, L28_hca_scale, L28_attn_norm_w, L28_hcf_fn_t, L28_hcf_base, L28_hcf_scale,
    L28_ffn_norm_w, L28_wq_a, L28_wq_a_s, L28_q_norm_w, L28_wq_b, L28_wq_b_s, L28_wkv, L28_wkv_s,
    L28_kv_norm_w, L28_wo_a, L28_wo_a_s, L28_wo_b, L28_wo_b_s, L28_attn_sink, L28_window_kv_cache,
    L28_cm_wkv_t, L28_cm_norm_w, L28_ix_wq_b, L28_ix_wq_b_s, L28_ix_wproj_t, L28_ix_knorm_w,
    L28_gate_w_t, L28_gate_b, L28_expert_lo, L28_w1, L28_s1, L28_w3, L28_s3, L28_w2, L28_s2,
    L28_sh_w1, L28_sh_s1, L28_sh_w3, L28_sh_s3, L28_sh_w2, L28_sh_s2, L29_hca_fn_t, L29_hca_base,
    L29_hca_scale, L29_attn_norm_w, L29_hcf_fn_t, L29_hcf_base, L29_hcf_scale, L29_ffn_norm_w,
    L29_wq_a, L29_wq_a_s, L29_q_norm_w, L29_wq_b, L29_wq_b_s, L29_wkv, L29_wkv_s, L29_kv_norm_w,
    L29_wo_a, L29_wo_a_s, L29_wo_b, L29_wo_b_s, L29_attn_sink, L29_window_kv_cache, L29_cm_wkv_t,
    L29_cm_norm_w, L29_ix_wproj_t, L29_ix_knorm_w, L29_gate_w_t, L29_gate_b, L29_expert_lo, L29_w1,
    L29_s1, L29_w3, L29_s3, L29_w2, L29_s2, L29_sh_w1, L29_sh_s1, L29_sh_w3, L29_sh_s3, L29_sh_w2,
    L29_sh_s2, L30_hca_fn_t, L30_hca_base, L30_hca_scale, L30_attn_norm_w, L30_hcf_fn_t,
    L30_hcf_base, L30_hcf_scale, L30_ffn_norm_w, L30_wq_a, L30_wq_a_s, L30_q_norm_w, L30_wq_b,
    L30_wq_b_s, L30_wkv, L30_wkv_s, L30_kv_norm_w, L30_wo_a, L30_wo_a_s, L30_wo_b, L30_wo_b_s,
    L30_attn_sink, L30_window_kv_cache, L30_cm_wkv_t, L30_cm_norm_w, L30_ix_wproj_t,
    L30_ix_knorm_w, L30_gate_w_t, L30_gate_b, L30_expert_lo, L30_w1, L30_s1, L30_w3, L30_s3,
    L30_w2, L30_s2, L30_sh_w1, L30_sh_s1, L30_sh_w3, L30_sh_s3, L30_sh_w2, L30_sh_s2, L31_hca_fn_t,
    L31_hca_base, L31_hca_scale, L31_attn_norm_w, L31_hcf_fn_t, L31_hcf_base, L31_hcf_scale,
    L31_ffn_norm_w, L31_wq_a, L31_wq_a_s, L31_q_norm_w, L31_wq_b, L31_wq_b_s, L31_wkv, L31_wkv_s,
    L31_kv_norm_w, L31_wo_a, L31_wo_a_s, L31_wo_b, L31_wo_b_s, L31_attn_sink, L31_window_kv_cache,
    L31_cm_wkv_t, L31_cm_norm_w, L31_ix_wproj_t, L31_ix_knorm_w, L31_gate_w_t, L31_gate_b,
    L31_expert_lo, L31_w1, L31_s1, L31_w3, L31_s3, L31_w2, L31_s2, L31_sh_w1, L31_sh_s1, L31_sh_w3,
    L31_sh_s3, L31_sh_w2, L31_sh_s2
):
    # tables / tensors members share (built by the first member that needs them, see
    # SHARE_TABLES in the member kernels)
    share = {}
    # L24 (r1_idx)
    with member(0, r1_idx_decode):
        o0 = r1_idx_decode.layer_decode_r1_idx_tp64.func(
            L24_stream, L24_pre_mix, L24_hca_fn_t, L24_hca_base, L24_hca_scale, L24_attn_norm_w,
            L24_hcf_fn_t, L24_hcf_base, L24_hcf_scale, L24_ffn_norm_w, L24_wq_a, L24_wq_a_s,
            L24_q_norm_w, L24_wq_b, L24_wq_b_s, L24_wkv, L24_wkv_s, L24_kv_norm_w, L24_wo_a,
            L24_wo_a_s, L24_wo_b, L24_wo_b_s, L24_attn_sink, L24_freqs, L24_row_pos,
            L24_write_slots, L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase,
            L24_topk_in, L24_window_kv_cache, L24_compress_kv_cache, placeholder_float32,
            placeholder_float32, L24_index_k_cache, L24_cm_wkv, L24_cm_wkv_t, placeholder_float32,
            L24_cm_norm_w, placeholder_int32, placeholder_int32, placeholder_int32, L24_ix_wq_b,
            L24_ix_wq_b_s, L24_ix_wproj_t, L24_ix_wk, L24_ix_knorm_w, L24_clen, placeholder_int32,
            L24_etable, placeholder_int32, placeholder_uint8, placeholder_uint8,
            placeholder_float32, L24_gate_w_t, L24_gate_b, L24_expert_lo, L24_w1, L24_s1, L24_w3,
            L24_s3, L24_w2, L24_s2, L24_sh_w1, L24_sh_s1, L24_sh_w3, L24_sh_s3, L24_sh_w2,
            L24_sh_s2, L24_groups_attn, L24_groups_world, L24_flags, share=share
        )
    # L25 (r1)
    with member(1, r1_decode):
        o1 = r1_decode.layer_decode_r1_tp64.func(
            o0[0], o0[1], L25_hca_fn_t, L25_hca_base, L25_hca_scale, L25_attn_norm_w, L25_hcf_fn_t,
            L25_hcf_base, L25_hcf_scale, L25_ffn_norm_w, L25_wq_a, L25_wq_a_s, L25_q_norm_w,
            L25_wq_b, L25_wq_b_s, L25_wkv, L25_wkv_s, L25_kv_norm_w, L25_wo_a, L25_wo_a_s,
            L25_wo_b, L25_wo_b_s, L25_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o0[2], L25_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L25_cm_wkv_t, placeholder_float32, L25_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L24_ix_wq_b, L24_ix_wq_b_s, L25_ix_wproj_t,
            L24_ix_wk, L25_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L25_gate_w_t, L25_gate_b,
            L25_expert_lo, L25_w1, L25_s1, L25_w3, L25_s3, L25_w2, L25_s2, L25_sh_w1, L25_sh_s1,
            L25_sh_w3, L25_sh_s3, L25_sh_w2, L25_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    # L26 (r1)
    with member(2, r1_decode):
        o2 = r1_decode.layer_decode_r1_tp64.func(
            o1[0], o1[1], L26_hca_fn_t, L26_hca_base, L26_hca_scale, L26_attn_norm_w, L26_hcf_fn_t,
            L26_hcf_base, L26_hcf_scale, L26_ffn_norm_w, L26_wq_a, L26_wq_a_s, L26_q_norm_w,
            L26_wq_b, L26_wq_b_s, L26_wkv, L26_wkv_s, L26_kv_norm_w, L26_wo_a, L26_wo_a_s,
            L26_wo_b, L26_wo_b_s, L26_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o0[2], L26_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L26_cm_wkv_t, placeholder_float32, L26_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L24_ix_wq_b, L24_ix_wq_b_s, L26_ix_wproj_t,
            L24_ix_wk, L26_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L26_gate_w_t, L26_gate_b,
            L26_expert_lo, L26_w1, L26_s1, L26_w3, L26_s3, L26_w2, L26_s2, L26_sh_w1, L26_sh_s1,
            L26_sh_w3, L26_sh_s3, L26_sh_w2, L26_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    # L27 (r1)
    with member(3, r1_decode):
        o3 = r1_decode.layer_decode_r1_tp64.func(
            o2[0], o2[1], L27_hca_fn_t, L27_hca_base, L27_hca_scale, L27_attn_norm_w, L27_hcf_fn_t,
            L27_hcf_base, L27_hcf_scale, L27_ffn_norm_w, L27_wq_a, L27_wq_a_s, L27_q_norm_w,
            L27_wq_b, L27_wq_b_s, L27_wkv, L27_wkv_s, L27_kv_norm_w, L27_wo_a, L27_wo_a_s,
            L27_wo_b, L27_wo_b_s, L27_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o0[2], L27_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L27_cm_wkv_t, placeholder_float32, L27_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L24_ix_wq_b, L24_ix_wq_b_s, L27_ix_wproj_t,
            L24_ix_wk, L27_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L27_gate_w_t, L27_gate_b,
            L27_expert_lo, L27_w1, L27_s1, L27_w3, L27_s3, L27_w2, L27_s2, L27_sh_w1, L27_sh_s1,
            L27_sh_w3, L27_sh_s3, L27_sh_w2, L27_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    # L28 (r1_idx)
    with member(4, r1_idx_decode):
        o4 = r1_idx_decode.layer_decode_r1_idx_tp64.func(
            o3[0], o3[1], L28_hca_fn_t, L28_hca_base, L28_hca_scale, L28_attn_norm_w, L28_hcf_fn_t,
            L28_hcf_base, L28_hcf_scale, L28_ffn_norm_w, L28_wq_a, L28_wq_a_s, L28_q_norm_w,
            L28_wq_b, L28_wq_b_s, L28_wkv, L28_wkv_s, L28_kv_norm_w, L28_wo_a, L28_wo_a_s,
            L28_wo_b, L28_wo_b_s, L28_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o0[2], L28_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L28_cm_wkv_t, placeholder_float32, L28_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L28_ix_wq_b, L28_ix_wq_b_s, L28_ix_wproj_t,
            L24_ix_wk, L28_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L28_gate_w_t, L28_gate_b,
            L28_expert_lo, L28_w1, L28_s1, L28_w3, L28_s3, L28_w2, L28_s2, L28_sh_w1, L28_sh_s1,
            L28_sh_w3, L28_sh_s3, L28_sh_w2, L28_sh_s2, L24_groups_attn, L24_groups_world,
            L24_flags, share=share
        )
    # L29 (r1)
    with member(5, r1_decode):
        o5 = r1_decode.layer_decode_r1_tp64.func(
            o4[0], o4[1], L29_hca_fn_t, L29_hca_base, L29_hca_scale, L29_attn_norm_w, L29_hcf_fn_t,
            L29_hcf_base, L29_hcf_scale, L29_ffn_norm_w, L29_wq_a, L29_wq_a_s, L29_q_norm_w,
            L29_wq_b, L29_wq_b_s, L29_wkv, L29_wkv_s, L29_kv_norm_w, L29_wo_a, L29_wo_a_s,
            L29_wo_b, L29_wo_b_s, L29_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o4[2], L29_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L29_cm_wkv_t, placeholder_float32, L29_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L28_ix_wq_b, L28_ix_wq_b_s, L29_ix_wproj_t,
            L24_ix_wk, L29_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L29_gate_w_t, L29_gate_b,
            L29_expert_lo, L29_w1, L29_s1, L29_w3, L29_s3, L29_w2, L29_s2, L29_sh_w1, L29_sh_s1,
            L29_sh_w3, L29_sh_s3, L29_sh_w2, L29_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    # L30 (r1)
    with member(6, r1_decode):
        o6 = r1_decode.layer_decode_r1_tp64.func(
            o5[0], o5[1], L30_hca_fn_t, L30_hca_base, L30_hca_scale, L30_attn_norm_w, L30_hcf_fn_t,
            L30_hcf_base, L30_hcf_scale, L30_ffn_norm_w, L30_wq_a, L30_wq_a_s, L30_q_norm_w,
            L30_wq_b, L30_wq_b_s, L30_wkv, L30_wkv_s, L30_kv_norm_w, L30_wo_a, L30_wo_a_s,
            L30_wo_b, L30_wo_b_s, L30_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o4[2], L30_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L30_cm_wkv_t, placeholder_float32, L30_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L28_ix_wq_b, L28_ix_wq_b_s, L30_ix_wproj_t,
            L24_ix_wk, L30_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L30_gate_w_t, L30_gate_b,
            L30_expert_lo, L30_w1, L30_s1, L30_w3, L30_s3, L30_w2, L30_s2, L30_sh_w1, L30_sh_s1,
            L30_sh_w3, L30_sh_s3, L30_sh_w2, L30_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    # L31 (r1)
    with member(7, r1_decode):
        o7 = r1_decode.layer_decode_r1_tp64.func(
            o6[0], o6[1], L31_hca_fn_t, L31_hca_base, L31_hca_scale, L31_attn_norm_w, L31_hcf_fn_t,
            L31_hcf_base, L31_hcf_scale, L31_ffn_norm_w, L31_wq_a, L31_wq_a_s, L31_q_norm_w,
            L31_wq_b, L31_wq_b_s, L31_wkv, L31_wkv_s, L31_kv_norm_w, L31_wo_a, L31_wo_a_s,
            L31_wo_b, L31_wo_b_s, L31_attn_sink, L24_freqs, L24_row_pos, L24_write_slots,
            L24_win_slots, L24_cmp_slots, L24_twin, L24_wv, L24_cbase, o4[2], L31_window_kv_cache,
            L24_compress_kv_cache, placeholder_float32, placeholder_float32, L24_index_k_cache,
            L24_cm_wkv, L31_cm_wkv_t, placeholder_float32, L31_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L28_ix_wq_b, L28_ix_wq_b_s, L31_ix_wproj_t,
            L24_ix_wk, L31_ix_knorm_w, L24_clen, placeholder_int32, L24_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L31_gate_w_t, L31_gate_b,
            L31_expert_lo, L31_w1, L31_s1, L31_w3, L31_s3, L31_w2, L31_s2, L31_sh_w1, L31_sh_s1,
            L31_sh_w3, L31_sh_s3, L31_sh_w2, L31_sh_s2, L24_groups_attn, L24_groups_world,
            L25_flags, share=share
        )
    return o7[0], o7[1], o4[2]
