# SPDX-License-Identifier: Apache-2.0
"""KV data parallelism (KVDP) for Hy3 decode attention (``HY3_DECODE_KVDP=<n>``, experimental).

At TP=64 with 8 KV heads, the 8 ranks that hold one KV head each read that head's full history
for every request. With KVDP=8, nkilib's ``attention_block_tkg`` gives each of the 8 ranks B/8 of
the requests: an in-kernel all-to-all gathers the 8 Q heads of a rank's requests onto it before
attention and returns every rank its own head for all B requests after it, so the O-projection and
everything else stay TP=64. Each rank reads 1/8 of the KV bytes.

The kernel can only write the KV of its own B/8 requests. vLLM may move a request to another batch
row (another KVDP rank) between steps, so every rank must hold every request's KV. The kernel runs
with ``update_cache=False``, returns its requests' new (FP8) K / V, one TP all-gather collects the
group's rows, and every rank writes all B rows into its cache with ``kvdp_kv_write`` (the
kernel's own vectorized scatter; torch ``index_put_`` on the caches cost ~0.8 ms per layer).
"""
import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nkilib.experimental.transformer.attention_block_tkg import (
    _update_block_cache,
    attention_block_tkg as _nki_attention_block_tkg,
)
from nkilib.experimental.transformer.attention_block_tkg_sharding import KVDPCollectiveMode


@nki.jit
def kvdp_kv_write(K_cache, V_cache, K_groups, V_new, kv_idx, fp8_packed=True):
    """Write a decode step's FP8 K / V of all B requests into the block caches.

    K_cache [blocks, 1, block // 2, D, 2] (packed) or [blocks, 1, block, D]; V_cache
    [blocks, 1, block, D]; K_groups [KVDP, D, B / KVDP * S] (each KVDP rank's K as the attention
    kernel returns it, in group order); V_new [B * S, D]; kv_idx [B, S] uint32 slots (uint32 max:
    skip). Returns the caches (aliased: updated in place).
    """
    kvdp, D, n = K_groups.shape
    nb, bl = V_cache.shape[0], V_cache.shape[2]
    B, S = kv_idx.shape
    K_sb = nl.ndarray((D, kvdp * n), dtype=K_groups.dtype, buffer=nl.sbuf)
    for r in range(kvdp):
        nisa.dma_copy(dst=K_sb[:, nl.ds(r * n, n)], src=K_groups[r])
    if fp8_packed:
        Kc = K_cache.reshape((nb, bl // 2, D, 2))
    else:
        Kc = K_cache.reshape((nb, bl, D))
    Vc = V_cache.reshape((nb, bl, D))
    _update_block_cache(Kc, Vc, K_sb, V_new.reshape((B, 1, S, D)), kv_idx, S, B, fp8_packed=fp8_packed)
    return K_cache, V_cache


@nki.jit
def kvdp_attention_block_tkg(
    X,
    X_hidden_dim_actual=None,
    rmsnorm_X_enabled=False,
    rmsnorm_X_eps=None,
    rmsnorm_X_gamma=None,
    W_qkv=None,
    quantization_type_qkv=None,
    weight_dequant_scale_qkv=None,
    input_dequant_scale_qkv=None,
    rmsnorm_QK_pre_rope_eps=1e-6,
    rmsnorm_QK_pre_rope_W_Q=None,
    rmsnorm_QK_pre_rope_W_K=None,
    cos=None,
    sin=None,
    active_blocks_table=None,
    K_cache=None,
    V_cache=None,
    attention_mask=None,
    softmax_scale=None,
    k_scale=None,
    v_scale=None,
    fp8_packed=False,
    W_out=None,
    quantization_type_out=None,
    weight_dequant_scale_out=None,
    input_dequant_scale_out=None,
    KVDP=1,
    KVDP_groups=None,
    KVDP_rank=None,
):
    """attention_block_tkg with KVDP: the Hy3 decode configuration, no cache update.

    ``KVDP_groups`` is a tuple of rank tuples (every group, so the graph is identical on every
    rank); the ReplicaGroup cannot cross the HOP boundary and is rebuilt here.
    """
    return _nki_attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=1e-6,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        update_cache=False,
        kv_cache_update_idx=None,
        W_out=W_out,
        bias_out=None,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        fp8_packed=fp8_packed,
        k_scale=k_scale,
        v_scale=v_scale,
        KVDP=KVDP,
        KVDP_replica_group=ncc.ReplicaGroup([list(g) for g in KVDP_groups]),
        KVDP_collective_mode=KVDPCollectiveMode.ALL_TO_ALL,
        KVDP_rank=KVDP_rank,
    )
