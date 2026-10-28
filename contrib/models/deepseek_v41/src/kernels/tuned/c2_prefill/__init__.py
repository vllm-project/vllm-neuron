"""`c2_prefill` as tuned by its autohelix loop (`~/ds41_mk_c2_prefill` @ 1836f03).

A self-contained copy of the layer-kernel modules, routed only for ('c2', 'prefill') by
`..layer_kernel.KERNELS`; every other archetype keeps the shared modules. Shared from the
parent package unchanged: blockquant, moe_expert.

Commits on the folder's main branch (newest first; the autohelix-accepted iterations plus
any fix made after the loop):

    1836f03 Pipeline the attn all_reduce in row chunks (hc_out(attn) of chunk k overlaps the later c
    8500f52 Fuse hc_out(attn) into hc_in(ffn) per row tile (rows stay in SBUF, no stream2 reload), s
    208a854 Attention: top-k gathers as one batched 2D-vector-offset DMA per token, A/B pipeline car
    fba6ef0 Attention: window tile as an HWDGE slab (register start) instead of a SWDGE gather, DVE-
    43c263b Attention: rolling cross-trip prefetch into persistent per-token buffers (8 blocks/trip)
    991489b Attention loop: 16 tokens per fori_loop trip with loads-first ordering + A/B software pi
"""
