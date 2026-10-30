# SPDX-License-Identifier: Apache-2.0
"""`layers_decode_L32_L39_head_tp64`: one launch of a DeepSeek-V4.1-Flash decode step (tp64, LNC=2) running
blocks 32-39 and then the greedy head, in order:

    L32 r1_idx, L33 r1, L34 r1, L35 r1, L36 r1_idx, L37 r1_tap, L38 r1_tap, L39 r1_tap, head head

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
AutoHelix loop (workspace ~/ds41_fused_L32_L39: its `git log` lists the changes, README.md the setup).
`mk_kernels.optimized_fused` runs it in place of the generated launch when the model's plan
is plan.json.
"""
import nki

from .fused_member import member
from .r1_idx_decode import r1_idx_decode
from .r1_decode import r1_decode
from .r1_tap_decode import r1_tap_decode
from .head import head_kernel


@nki.jit
def layers_decode_L32_L39_head_tp64(
    L32_stream, L32_pre_mix, L32_hca_fn_t, L32_hca_base, L32_hca_scale, L32_attn_norm_w,
    L32_hcf_fn_t, L32_hcf_base, L32_hcf_scale, L32_ffn_norm_w, L32_wq_a, L32_wq_a_s, L32_q_norm_w,
    L32_wq_b, L32_wq_b_s, L32_wkv, L32_wkv_s, L32_kv_norm_w, L32_wo_a, L32_wo_a_s, L32_wo_b,
    L32_wo_b_s, L32_attn_sink, L32_freqs, L32_row_pos, L32_write_slots, L32_win_slots,
    L32_cmp_slots, L32_twin, L32_wv, L32_cbase, L32_topk_in, L32_window_kv_cache,
    L32_compress_kv_cache, placeholder_float32, L32_index_k_cache, L32_cm_wkv, L32_cm_wkv_t,
    L32_cm_norm_w, placeholder_int32, L32_ix_wq_b, L32_ix_wq_b_s, L32_ix_wproj_t, L32_ix_wk,
    L32_ix_knorm_w, L32_clen, L32_etable, placeholder_uint8, L32_gate_w_t, L32_gate_b,
    L32_expert_lo, L32_w1, L32_s1, L32_w3, L32_s3, L32_w2, L32_s2, L32_sh_w1, L32_sh_s1, L32_sh_w3,
    L32_sh_s3, L32_sh_w2, L32_sh_s2, L32_groups_attn, L32_groups_world, L32_flags, L33_hca_fn_t,
    L33_hca_base, L33_hca_scale, L33_attn_norm_w, L33_hcf_fn_t, L33_hcf_base, L33_hcf_scale,
    L33_ffn_norm_w, L33_wq_a, L33_wq_a_s, L33_q_norm_w, L33_wq_b, L33_wq_b_s, L33_wkv, L33_wkv_s,
    L33_kv_norm_w, L33_wo_a, L33_wo_a_s, L33_wo_b, L33_wo_b_s, L33_attn_sink, L33_window_kv_cache,
    L33_cm_wkv_t, L33_cm_norm_w, L33_ix_wproj_t, L33_ix_knorm_w, L33_gate_w_t, L33_gate_b,
    L33_expert_lo, L33_w1, L33_s1, L33_w3, L33_s3, L33_w2, L33_s2, L33_sh_w1, L33_sh_s1, L33_sh_w3,
    L33_sh_s3, L33_sh_w2, L33_sh_s2, L33_flags, L34_hca_fn_t, L34_hca_base, L34_hca_scale,
    L34_attn_norm_w, L34_hcf_fn_t, L34_hcf_base, L34_hcf_scale, L34_ffn_norm_w, L34_wq_a,
    L34_wq_a_s, L34_q_norm_w, L34_wq_b, L34_wq_b_s, L34_wkv, L34_wkv_s, L34_kv_norm_w, L34_wo_a,
    L34_wo_a_s, L34_wo_b, L34_wo_b_s, L34_attn_sink, L34_window_kv_cache, L34_cm_wkv_t,
    L34_cm_norm_w, L34_ix_wproj_t, L34_ix_knorm_w, L34_gate_w_t, L34_gate_b, L34_expert_lo, L34_w1,
    L34_s1, L34_w3, L34_s3, L34_w2, L34_s2, L34_sh_w1, L34_sh_s1, L34_sh_w3, L34_sh_s3, L34_sh_w2,
    L34_sh_s2, L35_hca_fn_t, L35_hca_base, L35_hca_scale, L35_attn_norm_w, L35_hcf_fn_t,
    L35_hcf_base, L35_hcf_scale, L35_ffn_norm_w, L35_wq_a, L35_wq_a_s, L35_q_norm_w, L35_wq_b,
    L35_wq_b_s, L35_wkv, L35_wkv_s, L35_kv_norm_w, L35_wo_a, L35_wo_a_s, L35_wo_b, L35_wo_b_s,
    L35_attn_sink, L35_window_kv_cache, L35_cm_wkv_t, L35_cm_norm_w, L35_ix_wproj_t,
    L35_ix_knorm_w, L35_gate_w_t, L35_gate_b, L35_expert_lo, L35_w1, L35_s1, L35_w3, L35_s3,
    L35_w2, L35_s2, L35_sh_w1, L35_sh_s1, L35_sh_w3, L35_sh_s3, L35_sh_w2, L35_sh_s2, L36_hca_fn_t,
    L36_hca_base, L36_hca_scale, L36_attn_norm_w, L36_hcf_fn_t, L36_hcf_base, L36_hcf_scale,
    L36_ffn_norm_w, L36_wq_a, L36_wq_a_s, L36_q_norm_w, L36_wq_b, L36_wq_b_s, L36_wkv, L36_wkv_s,
    L36_kv_norm_w, L36_wo_a, L36_wo_a_s, L36_wo_b, L36_wo_b_s, L36_attn_sink, L36_window_kv_cache,
    L36_cm_wkv_t, L36_cm_norm_w, L36_ix_wq_b, L36_ix_wq_b_s, L36_ix_wproj_t, L36_ix_knorm_w,
    L36_gate_w_t, L36_gate_b, L36_expert_lo, L36_w1, L36_s1, L36_w3, L36_s3, L36_w2, L36_s2,
    L36_sh_w1, L36_sh_s1, L36_sh_w3, L36_sh_s3, L36_sh_w2, L36_sh_s2, L37_hca_fn_t, L37_hca_base,
    L37_hca_scale, L37_attn_norm_w, L37_hcf_fn_t, L37_hcf_base, L37_hcf_scale, L37_ffn_norm_w,
    L37_wq_a, L37_wq_a_s, L37_q_norm_w, L37_wq_b, L37_wq_b_s, L37_wkv, L37_wkv_s, L37_kv_norm_w,
    L37_wo_a, L37_wo_a_s, L37_wo_b, L37_wo_b_s, L37_attn_sink, L37_window_kv_cache, L37_cm_wkv_t,
    L37_cm_norm_w, L37_ix_wproj_t, L37_ix_knorm_w, L37_gate_w_t, L37_gate_b, L37_expert_lo, L37_w1,
    L37_s1, L37_w3, L37_s3, L37_w2, L37_s2, L37_sh_w1, L37_sh_s1, L37_sh_w3, L37_sh_s3, L37_sh_w2,
    L37_sh_s2, L37_flags, L38_hca_fn_t, L38_hca_base, L38_hca_scale, L38_attn_norm_w, L38_hcf_fn_t,
    L38_hcf_base, L38_hcf_scale, L38_ffn_norm_w, L38_wq_a, L38_wq_a_s, L38_q_norm_w, L38_wq_b,
    L38_wq_b_s, L38_wkv, L38_wkv_s, L38_kv_norm_w, L38_wo_a, L38_wo_a_s, L38_wo_b, L38_wo_b_s,
    L38_attn_sink, L38_window_kv_cache, L38_cm_wkv_t, L38_cm_norm_w, L38_ix_wproj_t,
    L38_ix_knorm_w, L38_gate_w_t, L38_gate_b, L38_expert_lo, L38_w1, L38_s1, L38_w3, L38_s3,
    L38_w2, L38_s2, L38_sh_w1, L38_sh_s1, L38_sh_w3, L38_sh_s3, L38_sh_w2, L38_sh_s2, L39_hca_fn_t,
    L39_hca_base, L39_hca_scale, L39_attn_norm_w, L39_hcf_fn_t, L39_hcf_base, L39_hcf_scale,
    L39_ffn_norm_w, L39_wq_a, L39_wq_a_s, L39_q_norm_w, L39_wq_b, L39_wq_b_s, L39_wkv, L39_wkv_s,
    L39_kv_norm_w, L39_wo_a, L39_wo_a_s, L39_wo_b, L39_wo_b_s, L39_attn_sink, L39_window_kv_cache,
    L39_cm_wkv_t, L39_cm_norm_w, L39_ix_wproj_t, L39_ix_knorm_w, L39_gate_w_t, L39_gate_b,
    L39_expert_lo, L39_w1, L39_s1, L39_w3, L39_s3, L39_w2, L39_s2, L39_sh_w1, L39_sh_s1, L39_sh_w3,
    L39_sh_s3, L39_sh_w2, L39_sh_s2, head_norm_w, head_head_w, head_vocab_start, head_flags
):
    # what the members share across blocks (the slot table / attention index tables of
    # blocks reading the same top-k, see the members' SHARE_SLOT_TABLE)
    share = {}
    # L32 (r1_idx)
    with member(0, r1_idx_decode):
        o0 = r1_idx_decode.layer_decode_r1_idx_tp64.func(
            L32_stream, L32_pre_mix, L32_hca_fn_t, L32_hca_base, L32_hca_scale, L32_attn_norm_w,
            L32_hcf_fn_t, L32_hcf_base, L32_hcf_scale, L32_ffn_norm_w, L32_wq_a, L32_wq_a_s,
            L32_q_norm_w, L32_wq_b, L32_wq_b_s, L32_wkv, L32_wkv_s, L32_kv_norm_w, L32_wo_a,
            L32_wo_a_s, L32_wo_b, L32_wo_b_s, L32_attn_sink, L32_freqs, L32_row_pos,
            L32_write_slots, L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase,
            L32_topk_in, L32_window_kv_cache, L32_compress_kv_cache, placeholder_float32,
            placeholder_float32, L32_index_k_cache, L32_cm_wkv, L32_cm_wkv_t, placeholder_float32,
            L32_cm_norm_w, placeholder_int32, placeholder_int32, placeholder_int32, L32_ix_wq_b,
            L32_ix_wq_b_s, L32_ix_wproj_t, L32_ix_wk, L32_ix_knorm_w, L32_clen, placeholder_int32,
            L32_etable, placeholder_int32, placeholder_uint8, placeholder_uint8,
            placeholder_float32, L32_gate_w_t, L32_gate_b, L32_expert_lo, L32_w1, L32_s1, L32_w3,
            L32_s3, L32_w2, L32_s2, L32_sh_w1, L32_sh_s1, L32_sh_w3, L32_sh_s3, L32_sh_w2,
            L32_sh_s2, L32_groups_attn, L32_groups_world, L32_flags,
            share=share,
        )
    # L33 (r1)
    with member(1, r1_decode):
        o1 = r1_decode.layer_decode_r1_tp64.func(
            o0[0], o0[1], L33_hca_fn_t, L33_hca_base, L33_hca_scale, L33_attn_norm_w, L33_hcf_fn_t,
            L33_hcf_base, L33_hcf_scale, L33_ffn_norm_w, L33_wq_a, L33_wq_a_s, L33_q_norm_w,
            L33_wq_b, L33_wq_b_s, L33_wkv, L33_wkv_s, L33_kv_norm_w, L33_wo_a, L33_wo_a_s,
            L33_wo_b, L33_wo_b_s, L33_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o0[2], L33_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L33_cm_wkv_t, placeholder_float32, L33_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L32_ix_wq_b, L32_ix_wq_b_s, L33_ix_wproj_t,
            L32_ix_wk, L33_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L33_gate_w_t, L33_gate_b,
            L33_expert_lo, L33_w1, L33_s1, L33_w3, L33_s3, L33_w2, L33_s2, L33_sh_w1, L33_sh_s1,
            L33_sh_w3, L33_sh_s3, L33_sh_w2, L33_sh_s2, L32_groups_attn, L32_groups_world,
            L33_flags,
            share=share,
        )
    # L34 (r1)
    with member(2, r1_decode):
        o2 = r1_decode.layer_decode_r1_tp64.func(
            o1[0], o1[1], L34_hca_fn_t, L34_hca_base, L34_hca_scale, L34_attn_norm_w, L34_hcf_fn_t,
            L34_hcf_base, L34_hcf_scale, L34_ffn_norm_w, L34_wq_a, L34_wq_a_s, L34_q_norm_w,
            L34_wq_b, L34_wq_b_s, L34_wkv, L34_wkv_s, L34_kv_norm_w, L34_wo_a, L34_wo_a_s,
            L34_wo_b, L34_wo_b_s, L34_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o0[2], L34_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L34_cm_wkv_t, placeholder_float32, L34_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L32_ix_wq_b, L32_ix_wq_b_s, L34_ix_wproj_t,
            L32_ix_wk, L34_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L34_gate_w_t, L34_gate_b,
            L34_expert_lo, L34_w1, L34_s1, L34_w3, L34_s3, L34_w2, L34_s2, L34_sh_w1, L34_sh_s1,
            L34_sh_w3, L34_sh_s3, L34_sh_w2, L34_sh_s2, L32_groups_attn, L32_groups_world,
            L33_flags,
            share=share,
        )
    # L35 (r1)
    with member(3, r1_decode):
        o3 = r1_decode.layer_decode_r1_tp64.func(
            o2[0], o2[1], L35_hca_fn_t, L35_hca_base, L35_hca_scale, L35_attn_norm_w, L35_hcf_fn_t,
            L35_hcf_base, L35_hcf_scale, L35_ffn_norm_w, L35_wq_a, L35_wq_a_s, L35_q_norm_w,
            L35_wq_b, L35_wq_b_s, L35_wkv, L35_wkv_s, L35_kv_norm_w, L35_wo_a, L35_wo_a_s,
            L35_wo_b, L35_wo_b_s, L35_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o0[2], L35_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L35_cm_wkv_t, placeholder_float32, L35_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L32_ix_wq_b, L32_ix_wq_b_s, L35_ix_wproj_t,
            L32_ix_wk, L35_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L35_gate_w_t, L35_gate_b,
            L35_expert_lo, L35_w1, L35_s1, L35_w3, L35_s3, L35_w2, L35_s2, L35_sh_w1, L35_sh_s1,
            L35_sh_w3, L35_sh_s3, L35_sh_w2, L35_sh_s2, L32_groups_attn, L32_groups_world,
            L33_flags,
            share=share,
        )
    # L36 (r1_idx)
    with member(4, r1_idx_decode):
        o4 = r1_idx_decode.layer_decode_r1_idx_tp64.func(
            o3[0], o3[1], L36_hca_fn_t, L36_hca_base, L36_hca_scale, L36_attn_norm_w, L36_hcf_fn_t,
            L36_hcf_base, L36_hcf_scale, L36_ffn_norm_w, L36_wq_a, L36_wq_a_s, L36_q_norm_w,
            L36_wq_b, L36_wq_b_s, L36_wkv, L36_wkv_s, L36_kv_norm_w, L36_wo_a, L36_wo_a_s,
            L36_wo_b, L36_wo_b_s, L36_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o0[2], L36_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L36_cm_wkv_t, placeholder_float32, L36_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L36_ix_wq_b, L36_ix_wq_b_s, L36_ix_wproj_t,
            L32_ix_wk, L36_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L36_gate_w_t, L36_gate_b,
            L36_expert_lo, L36_w1, L36_s1, L36_w3, L36_s3, L36_w2, L36_s2, L36_sh_w1, L36_sh_s1,
            L36_sh_w3, L36_sh_s3, L36_sh_w2, L36_sh_s2, L32_groups_attn, L32_groups_world,
            L32_flags,
            share=share,
        )
    # L37 (r1_tap)
    with member(5, r1_tap_decode):
        o5 = r1_tap_decode.layer_decode_r1_tap_tp64.func(
            o4[0], o4[1], L37_hca_fn_t, L37_hca_base, L37_hca_scale, L37_attn_norm_w, L37_hcf_fn_t,
            L37_hcf_base, L37_hcf_scale, L37_ffn_norm_w, L37_wq_a, L37_wq_a_s, L37_q_norm_w,
            L37_wq_b, L37_wq_b_s, L37_wkv, L37_wkv_s, L37_kv_norm_w, L37_wo_a, L37_wo_a_s,
            L37_wo_b, L37_wo_b_s, L37_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o4[2], L37_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L37_cm_wkv_t, placeholder_float32, L37_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L36_ix_wq_b, L36_ix_wq_b_s, L37_ix_wproj_t,
            L32_ix_wk, L37_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L37_gate_w_t, L37_gate_b,
            L37_expert_lo, L37_w1, L37_s1, L37_w3, L37_s3, L37_w2, L37_s2, L37_sh_w1, L37_sh_s1,
            L37_sh_w3, L37_sh_s3, L37_sh_w2, L37_sh_s2, L32_groups_attn, L32_groups_world,
            L37_flags,
            share=share,
        )
    # L38 (r1_tap)
    with member(6, r1_tap_decode):
        o6 = r1_tap_decode.layer_decode_r1_tap_tp64.func(
            o5[0], o5[1], L38_hca_fn_t, L38_hca_base, L38_hca_scale, L38_attn_norm_w, L38_hcf_fn_t,
            L38_hcf_base, L38_hcf_scale, L38_ffn_norm_w, L38_wq_a, L38_wq_a_s, L38_q_norm_w,
            L38_wq_b, L38_wq_b_s, L38_wkv, L38_wkv_s, L38_kv_norm_w, L38_wo_a, L38_wo_a_s,
            L38_wo_b, L38_wo_b_s, L38_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o4[2], L38_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L38_cm_wkv_t, placeholder_float32, L38_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L36_ix_wq_b, L36_ix_wq_b_s, L38_ix_wproj_t,
            L32_ix_wk, L38_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L38_gate_w_t, L38_gate_b,
            L38_expert_lo, L38_w1, L38_s1, L38_w3, L38_s3, L38_w2, L38_s2, L38_sh_w1, L38_sh_s1,
            L38_sh_w3, L38_sh_s3, L38_sh_w2, L38_sh_s2, L32_groups_attn, L32_groups_world,
            L37_flags,
            share=share,
        )
    # L39 (r1_tap)
    with member(7, r1_tap_decode):
        o7 = r1_tap_decode.layer_decode_r1_tap_tp64.func(
            o6[0], o6[1], L39_hca_fn_t, L39_hca_base, L39_hca_scale, L39_attn_norm_w, L39_hcf_fn_t,
            L39_hcf_base, L39_hcf_scale, L39_ffn_norm_w, L39_wq_a, L39_wq_a_s, L39_q_norm_w,
            L39_wq_b, L39_wq_b_s, L39_wkv, L39_wkv_s, L39_kv_norm_w, L39_wo_a, L39_wo_a_s,
            L39_wo_b, L39_wo_b_s, L39_attn_sink, L32_freqs, L32_row_pos, L32_write_slots,
            L32_win_slots, L32_cmp_slots, L32_twin, L32_wv, L32_cbase, o4[2], L39_window_kv_cache,
            L32_compress_kv_cache, placeholder_float32, placeholder_float32, L32_index_k_cache,
            L32_cm_wkv, L39_cm_wkv_t, placeholder_float32, L39_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L36_ix_wq_b, L36_ix_wq_b_s, L39_ix_wproj_t,
            L32_ix_wk, L39_ix_knorm_w, L32_clen, placeholder_int32, L32_etable, placeholder_int32,
            placeholder_uint8, placeholder_uint8, placeholder_float32, L39_gate_w_t, L39_gate_b,
            L39_expert_lo, L39_w1, L39_s1, L39_w3, L39_s3, L39_w2, L39_s2, L39_sh_w1, L39_sh_s1,
            L39_sh_w3, L39_sh_s3, L39_sh_w2, L39_sh_s2, L32_groups_attn, L32_groups_world,
            L37_flags,
            share=share,
        )
    # head
    with member(8, head_kernel):
        o8 = head_kernel.head_greedy_body(
            o7[0], o7[1], head_norm_w, head_head_w, head_vocab_start, L32_groups_world, head_flags
        )
    return o7[0], o7[1], o5[3], o6[3], o7[3], o4[2], o8[0]
