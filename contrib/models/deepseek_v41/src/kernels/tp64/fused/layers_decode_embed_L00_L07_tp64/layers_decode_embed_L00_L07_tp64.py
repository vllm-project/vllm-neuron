# SPDX-License-Identifier: Apache-2.0
"""`layers_decode_embed_L00_L07_tp64`: one launch of a DeepSeek-V4.1-Flash decode step (tp64, LNC=2) running
blocks 0-7 after the token embedding, in order:

    embed head, L00 r0, L01 r0_e, L02 r2_src_idx, L03 r2, L04 r2, L05 r2, L06 r2, L07 r2

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
AutoHelix loop (workspace ~/ds41_fused_L00_L07: its `git log` lists the changes, README.md the setup).
`mk_kernels.optimized_fused` runs it in place of the generated launch when the model's plan
is plan.json.
"""
import nki

from .fused_member import member
from .head import head_kernel
from .r0_decode import r0_decode
from .r0_e_decode import r0_e_decode
from .r2_src_idx_decode import r2_src_idx_decode
from .r2_decode import r2_decode


@nki.jit
def layers_decode_embed_L00_L07_tp64(
    embed_ids, embed_embed_w, embed_vocab_start, embed_groups_world, embed_flags, L00_hca_fn_t,
    L00_hca_base, L00_hca_scale, L00_attn_norm_w, L00_hcf_fn_t, L00_hcf_base, L00_hcf_scale,
    L00_ffn_norm_w, L00_wq_a, L00_wq_a_s, L00_q_norm_w, L00_wq_b, L00_wq_b_s, L00_wkv, L00_wkv_s,
    L00_kv_norm_w, L00_wo_a, L00_wo_a_s, L00_wo_b, L00_wo_b_s, L00_attn_sink, L00_freqs,
    L00_row_pos, L00_write_slots, L00_win_slots, L00_cmp_slots, L00_twin, L00_wv, L00_cbase,
    L00_topk_in, L00_window_kv_cache, placeholder_bfloat16, placeholder_float32, L00_cm_wkv,
    L00_cm_wkv_t, L00_cm_wgate_t, L00_cm_norm_w, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s,
    L00_ix_wproj_t, L00_ix_wk, L00_ix_knorm_w, placeholder_uint8, L00_gate_w_t, L00_gate_b,
    L00_expert_lo, L00_w1, L00_s1, L00_w3, L00_s3, L00_w2, L00_s2, L00_sh_w1, L00_sh_s1, L00_sh_w3,
    L00_sh_s3, L00_sh_w2, L00_sh_s2, L00_groups_attn, L00_flags, L01_hca_fn_t, L01_hca_base,
    L01_hca_scale, L01_attn_norm_w, L01_hcf_fn_t, L01_hcf_base, L01_hcf_scale, L01_ffn_norm_w,
    L01_wq_a, L01_wq_a_s, L01_q_norm_w, L01_wq_b, L01_wq_b_s, L01_wkv, L01_wkv_s, L01_kv_norm_w,
    L01_wo_a, L01_wo_a_s, L01_wo_b, L01_wo_b_s, L01_attn_sink, L01_window_kv_cache, L01_cm_wkv_t,
    L01_cm_wgate_t, L01_cm_norm_w, L01_ix_wproj_t, L01_ix_knorm_w, L01_hash_ids, L01_etable,
    L01_erow_start, L01_e_wkv, L01_e_wkv_s, L01_e_qk, L01_gate_w_t, L01_gate_b, L01_expert_lo,
    L01_w1, L01_s1, L01_w3, L01_s3, L01_w2, L01_s2, L01_sh_w1, L01_sh_s1, L01_sh_w3, L01_sh_s3,
    L01_sh_w2, L01_sh_s2, L01_flags, L02_hca_fn_t, L02_hca_base, L02_hca_scale, L02_attn_norm_w,
    L02_hcf_fn_t, L02_hcf_base, L02_hcf_scale, L02_ffn_norm_w, L02_wq_a, L02_wq_a_s, L02_q_norm_w,
    L02_wq_b, L02_wq_b_s, L02_wkv, L02_wkv_s, L02_kv_norm_w, L02_wo_a, L02_wo_a_s, L02_wo_b,
    L02_wo_b_s, L02_attn_sink, L02_freqs, L02_row_pos, L02_write_slots, L02_win_slots,
    L02_cmp_slots, L02_twin, L02_wv, L02_cbase, L02_topk_in, L02_window_kv_cache,
    L02_compress_kv_cache, L02_pool_kv, L02_pool_score, L02_index_k_cache, L02_cm_wkv_t,
    L02_cm_wgate_t, L02_cm_norm_w, L02_grp_pos, L02_grp_mslots, L02_grp_wslots, L02_ix_wproj_t,
    L02_ix_knorm_w, L02_clen, L02_gate_w_t, L02_gate_b, L02_expert_lo, L02_w1, L02_s1, L02_w3,
    L02_s3, L02_w2, L02_s2, L02_sh_w1, L02_sh_s1, L02_sh_w3, L02_sh_s3, L02_sh_w2, L02_sh_s2,
    L02_flags, L03_hca_fn_t, L03_hca_base, L03_hca_scale, L03_attn_norm_w, L03_hcf_fn_t,
    L03_hcf_base, L03_hcf_scale, L03_ffn_norm_w, L03_wq_a, L03_wq_a_s, L03_q_norm_w, L03_wq_b,
    L03_wq_b_s, L03_wkv, L03_wkv_s, L03_kv_norm_w, L03_wo_a, L03_wo_a_s, L03_wo_b, L03_wo_b_s,
    L03_attn_sink, L03_window_kv_cache, L03_cm_wkv_t, L03_cm_wgate_t, L03_cm_norm_w,
    L03_ix_wproj_t, L03_ix_knorm_w, L03_gate_w_t, L03_gate_b, L03_expert_lo, L03_w1, L03_s1,
    L03_w3, L03_s3, L03_w2, L03_s2, L03_sh_w1, L03_sh_s1, L03_sh_w3, L03_sh_s3, L03_sh_w2,
    L03_sh_s2, L03_flags, L04_hca_fn_t, L04_hca_base, L04_hca_scale, L04_attn_norm_w, L04_hcf_fn_t,
    L04_hcf_base, L04_hcf_scale, L04_ffn_norm_w, L04_wq_a, L04_wq_a_s, L04_q_norm_w, L04_wq_b,
    L04_wq_b_s, L04_wkv, L04_wkv_s, L04_kv_norm_w, L04_wo_a, L04_wo_a_s, L04_wo_b, L04_wo_b_s,
    L04_attn_sink, L04_window_kv_cache, L04_cm_wkv_t, L04_cm_wgate_t, L04_cm_norm_w,
    L04_ix_wproj_t, L04_ix_knorm_w, L04_gate_w_t, L04_gate_b, L04_expert_lo, L04_w1, L04_s1,
    L04_w3, L04_s3, L04_w2, L04_s2, L04_sh_w1, L04_sh_s1, L04_sh_w3, L04_sh_s3, L04_sh_w2,
    L04_sh_s2, L05_hca_fn_t, L05_hca_base, L05_hca_scale, L05_attn_norm_w, L05_hcf_fn_t,
    L05_hcf_base, L05_hcf_scale, L05_ffn_norm_w, L05_wq_a, L05_wq_a_s, L05_q_norm_w, L05_wq_b,
    L05_wq_b_s, L05_wkv, L05_wkv_s, L05_kv_norm_w, L05_wo_a, L05_wo_a_s, L05_wo_b, L05_wo_b_s,
    L05_attn_sink, L05_window_kv_cache, L05_cm_wkv_t, L05_cm_wgate_t, L05_cm_norm_w,
    L05_ix_wproj_t, L05_ix_knorm_w, L05_gate_w_t, L05_gate_b, L05_expert_lo, L05_w1, L05_s1,
    L05_w3, L05_s3, L05_w2, L05_s2, L05_sh_w1, L05_sh_s1, L05_sh_w3, L05_sh_s3, L05_sh_w2,
    L05_sh_s2, L06_hca_fn_t, L06_hca_base, L06_hca_scale, L06_attn_norm_w, L06_hcf_fn_t,
    L06_hcf_base, L06_hcf_scale, L06_ffn_norm_w, L06_wq_a, L06_wq_a_s, L06_q_norm_w, L06_wq_b,
    L06_wq_b_s, L06_wkv, L06_wkv_s, L06_kv_norm_w, L06_wo_a, L06_wo_a_s, L06_wo_b, L06_wo_b_s,
    L06_attn_sink, L06_window_kv_cache, L06_cm_wkv_t, L06_cm_wgate_t, L06_cm_norm_w,
    L06_ix_wproj_t, L06_ix_knorm_w, L06_gate_w_t, L06_gate_b, L06_expert_lo, L06_w1, L06_s1,
    L06_w3, L06_s3, L06_w2, L06_s2, L06_sh_w1, L06_sh_s1, L06_sh_w3, L06_sh_s3, L06_sh_w2,
    L06_sh_s2, L07_hca_fn_t, L07_hca_base, L07_hca_scale, L07_attn_norm_w, L07_hcf_fn_t,
    L07_hcf_base, L07_hcf_scale, L07_ffn_norm_w, L07_wq_a, L07_wq_a_s, L07_q_norm_w, L07_wq_b,
    L07_wq_b_s, L07_wkv, L07_wkv_s, L07_kv_norm_w, L07_wo_a, L07_wo_a_s, L07_wo_b, L07_wo_b_s,
    L07_attn_sink, L07_window_kv_cache, L07_cm_wkv_t, L07_cm_wgate_t, L07_cm_norm_w,
    L07_ix_wproj_t, L07_ix_knorm_w, L07_gate_w_t, L07_gate_b, L07_expert_lo, L07_w1, L07_s1,
    L07_w3, L07_s3, L07_w2, L07_s2, L07_sh_w1, L07_sh_s1, L07_sh_w3, L07_sh_s3, L07_sh_w2,
    L07_sh_s2
):
    # Each block's hc_in(attn) pre-mix reads the previous block's output rows from SBUF
    # (`XS_SHARE`: the final hc_out's spread tiles)
    xs_share = {}
    for m in (r0_decode, r0_e_decode, r2_src_idx_decode, r2_decode):
        m.XS_SHARE = xs_share
    # embed
    with member(0, head_kernel):
        o0 = head_kernel.embed_expand_body(
            embed_ids, embed_embed_w, embed_vocab_start, embed_groups_world, embed_flags
        )
    # L00 (r0)
    with member(1, r0_decode):
        o1 = r0_decode.layer_decode_r0_tp64.func(
            o0[0], o0[1], L00_hca_fn_t, L00_hca_base, L00_hca_scale, L00_attn_norm_w, L00_hcf_fn_t,
            L00_hcf_base, L00_hcf_scale, L00_ffn_norm_w, L00_wq_a, L00_wq_a_s, L00_q_norm_w,
            L00_wq_b, L00_wq_b_s, L00_wkv, L00_wkv_s, L00_kv_norm_w, L00_wo_a, L00_wo_a_s,
            L00_wo_b, L00_wo_b_s, L00_attn_sink, L00_freqs, L00_row_pos, L00_write_slots,
            L00_win_slots, L00_cmp_slots, L00_twin, L00_wv, L00_cbase, L00_topk_in,
            L00_window_kv_cache, placeholder_bfloat16, placeholder_float32, placeholder_float32,
            placeholder_bfloat16, L00_cm_wkv, L00_cm_wkv_t, L00_cm_wgate_t, L00_cm_norm_w,
            placeholder_int32, placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s,
            L00_ix_wproj_t, L00_ix_wk, L00_ix_knorm_w, placeholder_int32, placeholder_int32,
            placeholder_bfloat16, placeholder_int32, placeholder_uint8, placeholder_uint8,
            placeholder_float32, L00_gate_w_t, L00_gate_b, L00_expert_lo, L00_w1, L00_s1, L00_w3,
            L00_s3, L00_w2, L00_s2, L00_sh_w1, L00_sh_s1, L00_sh_w3, L00_sh_s3, L00_sh_w2,
            L00_sh_s2, L00_groups_attn, embed_groups_world, L00_flags
        )
    # L01 (r0_e)
    with member(2, r0_e_decode):
        o2 = r0_e_decode.layer_decode_r0_e_tp64.func(
            o1[0], o1[1], L01_hca_fn_t, L01_hca_base, L01_hca_scale, L01_attn_norm_w, L01_hcf_fn_t,
            L01_hcf_base, L01_hcf_scale, L01_ffn_norm_w, L01_wq_a, L01_wq_a_s, L01_q_norm_w,
            L01_wq_b, L01_wq_b_s, L01_wkv, L01_wkv_s, L01_kv_norm_w, L01_wo_a, L01_wo_a_s,
            L01_wo_b, L01_wo_b_s, L01_attn_sink, L00_freqs, L00_row_pos, L00_write_slots,
            L00_win_slots, L00_cmp_slots, L00_twin, L00_wv, L00_cbase, L00_topk_in,
            L01_window_kv_cache, placeholder_bfloat16, placeholder_float32, placeholder_float32,
            placeholder_bfloat16, L00_cm_wkv, L01_cm_wkv_t, L01_cm_wgate_t, L01_cm_norm_w,
            placeholder_int32, placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s,
            L01_ix_wproj_t, L00_ix_wk, L01_ix_knorm_w, placeholder_int32, L01_hash_ids, L01_etable,
            L01_erow_start, L01_e_wkv, L01_e_wkv_s, L01_e_qk, L01_gate_w_t, L01_gate_b,
            L01_expert_lo, L01_w1, L01_s1, L01_w3, L01_s3, L01_w2, L01_s2, L01_sh_w1, L01_sh_s1,
            L01_sh_w3, L01_sh_s3, L01_sh_w2, L01_sh_s2, L00_groups_attn, embed_groups_world,
            L01_flags
        )
    # L02 .. L07 read the same compressed cache (after L02's compressor), slot lists and
    # top-k (L02's): L02's slot table and index tables are shared, L03 .. L07 gather only
    # their window rows into it (`r2_decode.SHARE_TAB`)
    share_tab = {}
    r2_src_idx_decode.SHARE_TAB = share_tab
    r2_decode.SHARE_TAB = share_tab
    # Each r2 block's q / kv projection weights are loaded by the block before it, right
    # after that block's world all-reduce trigger (`r2_decode.prefetch_proj`)

    def prefetch(wq_a, wq_a_s, wkv, wkv_s, wq_b=None, wq_b_s=None):
        if not r2_decode.PREFETCH_NEXT_PROJ or o0[0].shape[0] > r2_decode.PREFETCH_MAX_T:
            return None

        def f(pid, after):
            r2_decode.PRE_PROJ = (wq_a, r2_decode.prefetch_proj(pid, wq_a, wq_a_s, wkv, wkv_s, after=after,
                                                                wq_b=wq_b, wq_b_s=wq_b_s))
        return f

    r2_src_idx_decode.AFTER_WORLD_AR = prefetch(L03_wq_a, L03_wq_a_s, L03_wkv, L03_wkv_s, L03_wq_b, L03_wq_b_s)
    # L02 (r2_src_idx)
    with member(3, r2_src_idx_decode):
        o3 = r2_src_idx_decode.layer_decode_r2_src_idx_tp64.func(
            o2[0], o2[1], L02_hca_fn_t, L02_hca_base, L02_hca_scale, L02_attn_norm_w, L02_hcf_fn_t,
            L02_hcf_base, L02_hcf_scale, L02_ffn_norm_w, L02_wq_a, L02_wq_a_s, L02_q_norm_w,
            L02_wq_b, L02_wq_b_s, L02_wkv, L02_wkv_s, L02_kv_norm_w, L02_wo_a, L02_wo_a_s,
            L02_wo_b, L02_wo_b_s, L02_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, L02_topk_in,
            L02_window_kv_cache, L02_compress_kv_cache, L02_pool_kv, L02_pool_score,
            L02_index_k_cache, L00_cm_wkv, L02_cm_wkv_t, L02_cm_wgate_t, L02_cm_norm_w,
            L02_grp_pos, L02_grp_mslots, L02_grp_wslots, L00_ix_wq_b, L00_ix_wq_b_s,
            L02_ix_wproj_t, L00_ix_wk, L02_ix_knorm_w, L02_clen, placeholder_int32,
            placeholder_bfloat16, placeholder_int32, placeholder_uint8, placeholder_uint8,
            placeholder_float32, L02_gate_w_t, L02_gate_b, L02_expert_lo, L02_w1, L02_s1, L02_w3,
            L02_s3, L02_w2, L02_s2, L02_sh_w1, L02_sh_s1, L02_sh_w3, L02_sh_s3, L02_sh_w2,
            L02_sh_s2, L00_groups_attn, embed_groups_world, L02_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L04_wq_a, L04_wq_a_s, L04_wkv, L04_wkv_s, L04_wq_b, L04_wq_b_s)
    # L03 (r2)
    with member(4, r2_decode):
        o4 = r2_decode.layer_decode_r2_tp64.func(
            o3[0], o3[1], L03_hca_fn_t, L03_hca_base, L03_hca_scale, L03_attn_norm_w, L03_hcf_fn_t,
            L03_hcf_base, L03_hcf_scale, L03_ffn_norm_w, L03_wq_a, L03_wq_a_s, L03_q_norm_w,
            L03_wq_b, L03_wq_b_s, L03_wkv, L03_wkv_s, L03_kv_norm_w, L03_wo_a, L03_wo_a_s,
            L03_wo_b, L03_wo_b_s, L03_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, o3[2], L03_window_kv_cache,
            L02_compress_kv_cache, placeholder_float32, placeholder_float32, L02_index_k_cache,
            L00_cm_wkv, L03_cm_wkv_t, L03_cm_wgate_t, L03_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s, L03_ix_wproj_t,
            L00_ix_wk, L03_ix_knorm_w, L02_clen, placeholder_int32, placeholder_bfloat16,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L03_gate_w_t, L03_gate_b, L03_expert_lo, L03_w1, L03_s1, L03_w3, L03_s3, L03_w2,
            L03_s2, L03_sh_w1, L03_sh_s1, L03_sh_w3, L03_sh_s3, L03_sh_w2, L03_sh_s2,
            L00_groups_attn, embed_groups_world, L03_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L05_wq_a, L05_wq_a_s, L05_wkv, L05_wkv_s, L05_wq_b, L05_wq_b_s)
    # L04 (r2)
    with member(5, r2_decode):
        o5 = r2_decode.layer_decode_r2_tp64.func(
            o4[0], o4[1], L04_hca_fn_t, L04_hca_base, L04_hca_scale, L04_attn_norm_w, L04_hcf_fn_t,
            L04_hcf_base, L04_hcf_scale, L04_ffn_norm_w, L04_wq_a, L04_wq_a_s, L04_q_norm_w,
            L04_wq_b, L04_wq_b_s, L04_wkv, L04_wkv_s, L04_kv_norm_w, L04_wo_a, L04_wo_a_s,
            L04_wo_b, L04_wo_b_s, L04_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, o3[2], L04_window_kv_cache,
            L02_compress_kv_cache, placeholder_float32, placeholder_float32, L02_index_k_cache,
            L00_cm_wkv, L04_cm_wkv_t, L04_cm_wgate_t, L04_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s, L04_ix_wproj_t,
            L00_ix_wk, L04_ix_knorm_w, L02_clen, placeholder_int32, placeholder_bfloat16,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L04_gate_w_t, L04_gate_b, L04_expert_lo, L04_w1, L04_s1, L04_w3, L04_s3, L04_w2,
            L04_s2, L04_sh_w1, L04_sh_s1, L04_sh_w3, L04_sh_s3, L04_sh_w2, L04_sh_s2,
            L00_groups_attn, embed_groups_world, L03_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L06_wq_a, L06_wq_a_s, L06_wkv, L06_wkv_s, L06_wq_b, L06_wq_b_s)
    # L05 (r2)
    with member(6, r2_decode):
        o6 = r2_decode.layer_decode_r2_tp64.func(
            o5[0], o5[1], L05_hca_fn_t, L05_hca_base, L05_hca_scale, L05_attn_norm_w, L05_hcf_fn_t,
            L05_hcf_base, L05_hcf_scale, L05_ffn_norm_w, L05_wq_a, L05_wq_a_s, L05_q_norm_w,
            L05_wq_b, L05_wq_b_s, L05_wkv, L05_wkv_s, L05_kv_norm_w, L05_wo_a, L05_wo_a_s,
            L05_wo_b, L05_wo_b_s, L05_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, o3[2], L05_window_kv_cache,
            L02_compress_kv_cache, placeholder_float32, placeholder_float32, L02_index_k_cache,
            L00_cm_wkv, L05_cm_wkv_t, L05_cm_wgate_t, L05_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s, L05_ix_wproj_t,
            L00_ix_wk, L05_ix_knorm_w, L02_clen, placeholder_int32, placeholder_bfloat16,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L05_gate_w_t, L05_gate_b, L05_expert_lo, L05_w1, L05_s1, L05_w3, L05_s3, L05_w2,
            L05_s2, L05_sh_w1, L05_sh_s1, L05_sh_w3, L05_sh_s3, L05_sh_w2, L05_sh_s2,
            L00_groups_attn, embed_groups_world, L03_flags
        )
    r2_decode.AFTER_WORLD_AR = prefetch(L07_wq_a, L07_wq_a_s, L07_wkv, L07_wkv_s, L07_wq_b, L07_wq_b_s)
    # L06 (r2)
    with member(7, r2_decode):
        o7 = r2_decode.layer_decode_r2_tp64.func(
            o6[0], o6[1], L06_hca_fn_t, L06_hca_base, L06_hca_scale, L06_attn_norm_w, L06_hcf_fn_t,
            L06_hcf_base, L06_hcf_scale, L06_ffn_norm_w, L06_wq_a, L06_wq_a_s, L06_q_norm_w,
            L06_wq_b, L06_wq_b_s, L06_wkv, L06_wkv_s, L06_kv_norm_w, L06_wo_a, L06_wo_a_s,
            L06_wo_b, L06_wo_b_s, L06_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, o3[2], L06_window_kv_cache,
            L02_compress_kv_cache, placeholder_float32, placeholder_float32, L02_index_k_cache,
            L00_cm_wkv, L06_cm_wkv_t, L06_cm_wgate_t, L06_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s, L06_ix_wproj_t,
            L00_ix_wk, L06_ix_knorm_w, L02_clen, placeholder_int32, placeholder_bfloat16,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L06_gate_w_t, L06_gate_b, L06_expert_lo, L06_w1, L06_s1, L06_w3, L06_s3, L06_w2,
            L06_s2, L06_sh_w1, L06_sh_s1, L06_sh_w3, L06_sh_s3, L06_sh_w2, L06_sh_s2,
            L00_groups_attn, embed_groups_world, L03_flags
        )
    r2_decode.AFTER_WORLD_AR = None
    # L07 (r2)
    with member(8, r2_decode):
        o8 = r2_decode.layer_decode_r2_tp64.func(
            o7[0], o7[1], L07_hca_fn_t, L07_hca_base, L07_hca_scale, L07_attn_norm_w, L07_hcf_fn_t,
            L07_hcf_base, L07_hcf_scale, L07_ffn_norm_w, L07_wq_a, L07_wq_a_s, L07_q_norm_w,
            L07_wq_b, L07_wq_b_s, L07_wkv, L07_wkv_s, L07_kv_norm_w, L07_wo_a, L07_wo_a_s,
            L07_wo_b, L07_wo_b_s, L07_attn_sink, L02_freqs, L02_row_pos, L02_write_slots,
            L02_win_slots, L02_cmp_slots, L02_twin, L02_wv, L02_cbase, o3[2], L07_window_kv_cache,
            L02_compress_kv_cache, placeholder_float32, placeholder_float32, L02_index_k_cache,
            L00_cm_wkv, L07_cm_wkv_t, L07_cm_wgate_t, L07_cm_norm_w, placeholder_int32,
            placeholder_int32, placeholder_int32, L00_ix_wq_b, L00_ix_wq_b_s, L07_ix_wproj_t,
            L00_ix_wk, L07_ix_knorm_w, L02_clen, placeholder_int32, placeholder_bfloat16,
            placeholder_int32, placeholder_uint8, placeholder_uint8, placeholder_float32,
            L07_gate_w_t, L07_gate_b, L07_expert_lo, L07_w1, L07_s1, L07_w3, L07_s3, L07_w2,
            L07_s2, L07_sh_w1, L07_sh_s1, L07_sh_w3, L07_sh_s3, L07_sh_w2, L07_sh_s2,
            L00_groups_attn, embed_groups_world, L03_flags
        )
    r2_src_idx_decode.SHARE_TAB = None
    r2_decode.SHARE_TAB = None
    for m in (r0_decode, r0_e_decode, r2_src_idx_decode, r2_decode):
        m.XS_SHARE = None
    r2_src_idx_decode.AFTER_WORLD_AR = None
    r2_decode.PRE_PROJ = None
    return o8[0], o8[1], o3[2]
