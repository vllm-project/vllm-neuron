"""Qwen3.5 gated full-attention decode step for one TP rank (NKI, LNC=1).

    out_partial = o_proj( attention(q, K, V) * sigmoid(x W_gate) )      (this rank's heads)

Self-contained (no nkilib attention block), so it can be inlined once per attention layer of
a mega kernel and tuned for this model's shape (head_dim 256, 1 KV head per rank, few tokens):
  * QKV / gate projections keep the tokens stationary and stream the weights;
  * zero-centred Q/K RMSNorm and the partial RoPE (first 64 of 256 dims, HF rotate-half over
    those 64) run token-major on the natural channel order;
  * KV cache is slot-contiguous: a slot owns ``k_cache_t[slot] [d, S]`` (K stored transposed, so
    scores need no transposes) and ``v_cache[slot] [S, d]``. Without ``slot`` row ``b`` uses slot
    ``b`` (static DMA bases, unchanged); with ``slot`` [B, 1] row ``b`` uses pool slot ``slot[b]`` of
    a pool of NSLOT >= B slots: history DMAs take the slot as a runtime ``scalar_offset`` on the slot
    dim, so no address carries ``slot * d * S`` as a kernel-computed integer;
  * scores for all B*nq (token, head) rows accumulate in one PSUM tile (zero-padded query
    stationaries), so the softmax runs once over a [B*nq, S] tile. The new token's own K/V
    enter as an extra column / row, so the history read never depends on this step's write;
  * P.V keeps the probabilities stationary per token and streams V in its natural layout.

Semantics (HF ``Qwen3_5MoeAttention``, one new token per sequence): sequence ``b`` attends to
history positions ``0 .. pos[b]-1`` of its slot plus the new token; the new K/V are written at
``pos[b]`` of its slot.

New-token K/V writes with a runtime slot (``kv_write``). NKI 0.7 allows one runtime offset per
access pattern (``scalar_offset`` xor ``vector_offset``, one dynamic select), but (slot, pos) are two:
  * ``"split"`` (simulation only: its zero / negative register strides FAIL on trn3 hardware): the
    slot is the ``scalar_offset`` on the slot dim (scaled by the slot stride d*S in the DGE) and the
    position enters as a runtime register *stride* of a 2-element pattern dim based at position S-1:
    element 0 -> position S-1, element 1 -> S-1 + (pos - (S-1)) = pos, both carrying the new K (V)
    values. Only per-slot quantities are integers in the kernel: |pos - (S-1)| < S (K) and
    |pos - (S-1)| * d < d*S (V, an exact shift). Side effect: position S-1 of each written slot also
    receives the new token's K/V. That position is never valid history while pos < S-1 (masked;
    rewritten when the sequence reaches it) and is the target itself when pos == S-1, so decode /
    verify / prefill (which read positions < length) are unaffected.
  * ``"flat"`` (default): one combined int32 element offset per row, ``slot * d*S + pos`` (K, scalar
    offset on the flat cache) and ``slot * S + pos`` (V rows, vector offset). No extra write, but
    NSLOT*d*S <= 2^31. The engines' integer add / multiply may run through fp32 (exact only below
    2^24), so the offsets are built from exact pieces only. Power-of-two S: ``(slot << log2(d*S)) | pos`` and
    ``(slot << log2 S) | pos`` (shifts / ors). Any other S (e.g. 9216 = 8K prompt + 1K decode) uses the
    radix split S = m * 2^k (k = trailing zeros of S >= 7, since S % 128 == 0):
        hi  = slot * (d*m) + (pos >> k)              (< NSLOT*d*S / 2^k <= 2^24: exact even in fp32)
        off = (hi << k) | (pos & (2^k - 1))          (shift / or only, like the power-of-two path)
    and the same with m instead of d*m for V. ``QWEN35_ATTN_FLAT_OFFSET=radix`` forces the radix path
    for power-of-two S too (m = 1; for hardware validation), default ``auto``.
Limits with ``slot`` ("split"): d*S < 2^31 elements per slot, d a power of two; NSLOT is not bounded
by kernel arithmetic (slot is a runtime int32 index scaled by the DGE). Rows need distinct slots.
Without ``slot`` the original path is kept bit-for-bit (incl. its d*S < 2^24 element-offset assert).
``QWEN35_ATTN_SLOT_DGE=hw`` forces hardware DGE for the slot-addressed history reads (and the "flat"
K write); the register-stride "split" writes are SWDGE-only.

Paged KV (prototype, ``block_table`` given). The caches are page pools
``k_cache_t [NPAGE, d, PG]`` / ``v_cache [NPAGE, PG, d]`` (PG = page size, a power of two >= 128) and
``block_table [B, MP]`` int32 maps row b's logical page j (positions j*PG .. j*PG+PG-1) to a pool page.
The attended span is S = MP * PG (any multiple of PG, no power-of-two rounding); entries past a row's
length must still name a page with finite data (serving reserves a zero page). History reads are one
DMA per (row, page segment) with the page id as the DMA's single runtime ``scalar_offset`` (an SBUF
scalar of the block table: no register per page); scores / softmax / P.V are the slot path's. The new
token goes to page ``bt[b, pos >> log2 PG]`` (looked up on device: one-hot over the row's MP entries,
exact in fp32 for page ids < 2^24) at offset ``pos & (PG-1)``: K at the flat element offset
``(page << log2(d*PG)) | off`` (scalar offset, needs NPAGE*d*PG <= 2^31), V at row ``(page << log2 PG)
| off`` (vector offset), i.e. the "flat" write with the page in place of the slot. Rows may share pages
(prefix sharing) for reading; the pages written (``pos`` of each row) must be private and distinct.
``k_page_layout``: "dp" (default) = each page is K^T [d, PG] (rows j*128+p): runs of 2*PG bytes per
partition; "pd" = the same page with its d rows stored in (p, j) order (row p*DH + j holds dim j*128+p):
partition p reads one contiguous run of 2*DH*PG bytes (512 B at PG=128 instead of 256 B) and the score
matmuls take MCH/PG pages at once through a 3D moving pattern (``QWEN35_ATTN_PD_GROUP=0``: one per page;
needs PG <= 2048). ``QWEN35_ATTN_PAGE_Q=spread`` round-robins the page DMAs over SP / ACT HWDGE and
GpSimd SWDGE (default "auto": the compiler's queue, or ``QWEN35_ATTN_SLOT_DGE``).
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

# DGE mode of the runtime-slot DMAs: "auto" (compiler's choice) or "hw" (force hardware DGE)
_SLOT_DGE = os.environ.get("QWEN35_ATTN_SLOT_DGE", "auto")
# kv_write="flat" offset arithmetic: "auto" (shift/or for power-of-two S, radix split otherwise) or
# "radix" (radix split for every S)
_FLAT_OFFSET = os.environ.get("QWEN35_ATTN_FLAT_OFFSET", "auto")
# queue of the paged history DMAs (block_table path only): "auto" = the compiler's choice / QWEN35_ATTN_SLOT_DGE
# (all on one issuing engine), "spread" = round-robin SP HWDGE / ACT HWDGE / GpSimd SWDGE per page DMA
_PAGE_Q = os.environ.get("QWEN35_ATTN_PAGE_Q", "auto")
_PAGE_QI = [0]
# k_page_layout="pd" with PG < 512: one score matmul over MCH/PG pages (3D moving access pattern) instead of one per page
_PD_GROUP = os.environ.get("QWEN35_ATTN_PD_GROUP", "1") == "1"


def _page_dma(dst, src, dge):
    """One paged history DMA on the queue chosen by QWEN35_ATTN_PAGE_Q."""
    if _PAGE_Q == "spread":
        i = _PAGE_QI[0] % 3
        _PAGE_QI[0] += 1
        if i < 2:
            eng = nisa.engine.sync if i == 0 else nisa.engine.scalar
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)
        return
    nisa.dma_copy(dst=dst, src=src, **dge)

P = 128
ROT = 64  # rotary dims (partial_rotary_factor 0.25 of head_dim 256)
SCH = 2048  # score columns per PSUM pass (fp32: 4 banks)
# split-K (patch 0005): spans longer than _SPLIT_MAX positions run in blocks of _SPLIT_BLOCK (a multiple of SCH); one
# pass over a row's whole span keeps its [B * nq, S] fp32 scores in SBUF, which fails allocation past ~16-24K
_SPLIT_MAX = int(os.environ.get("QWEN35_ATTN_SPLIT_MAX", "16384"))
_SPLIT_BLOCK = int(os.environ.get("QWEN35_ATTN_SPLIT_BLOCK", "4096"))
# split-K passes over the K/V blocks (patch 0006): 1 = online softmax (each block's K and V read once, the row's
# P.V rescaled as its running max moves) | 2 = the two-pass form of patch 0005 (K read twice)
_SPLIT_PASSES = os.environ.get("QWEN35_ATTN_SPLIT_PASSES", "1")
# split-K blocks in a run-time loop (patch 0007): the paged online split-K (one pass, "dp" pages) iterates its
# _SPLIT_BLOCK blocks on device (nl.fori_loop: one copy of the block body in the program, the pass's block-table
# columns read by a dynamic-offset DMA) instead of tracing one body per block, so the program stops growing with
# the span. "0" keeps the trace-unrolled form (short spans, where the compile is cheap anyway).
_BLOCK_LOOP = os.environ.get("QWEN35_ATTN_BLOCK_LOOP", "1") == "1"
NEG = 30000.0


def block_loop_engages(paged, S, PG, k_page_layout):
    """True when attn_decode_tkg takes the run-time block loop (_split_attn_online_loop). multistep mirrors it:
    the loop's per-pass block-table DMA must read a kernel INPUT (device loops inside _multistep_jit trip the
    allocator on dynamic-offset DMAs from HBM intermediates — NCC_IBIR853), so multistep hands the loop the raw
    shared table plus a per-layer pool offset instead of a staged per-layer copy."""
    return (_SPLIT_PASSES == "1" and paged and _BLOCK_LOOP and k_page_layout == "dp" and S > _SPLIT_MAX
            and S % _SPLIT_BLOCK == 0 and _SPLIT_BLOCK % PG == 0)


def cp_block_loop_engages(S_loc):
    """True when attn_cp_core iterates its span in a run-time block loop — the DCP twin of
    block_loop_engages. Each unrolled SCH block repeats the G x B chunk/row fan-out (~200k instructions
    at B=16), so past one block the compile grows with the page count even though a rank's token share
    stays small (256K at kv_cp 16: 9 blocks, 808k MATMUL instructions). Whole SCH blocks loop; a
    S_loc % SCH tail runs once unrolled, so any page count works. multistep mirrors this for the raw
    table + pool_off hand-off (a device loop's dynamic-offset DMAs must read kernel inputs —
    NCC_IBIR853)."""
    return _BLOCK_LOOP and S_loc > SCH


@nki.jit
def attn_decode_tkg(
    x,  # [B, H] bf16 (post input-RMSNorm)
    w_qkv,  # [H, (nq + 2) * d] bf16: q heads | k | v (natural channel order)
    q_norm_w,  # [1, d] fp32 raw zero-centred weight (kernel uses 1 + w)
    k_norm_w,  # [1, d] fp32
    w_gate_p,  # [128, H/128, nq * d] bf16 gate projection, rows pre-permuted
    cos,  # [B, ROT/2] fp32 cos(pos * inv_freq)
    sin,  # [B, ROT/2] fp32
    k_cache_t,  # [B, d, S] bf16, K transposed per sequence slot ([NSLOT, d, S] with slot)
    v_cache,  # [B, S, d] bf16 ([NSLOT, S, d] with slot)
    pos,  # [B, 1] int32 position of the new token (= history length)
    w_o,  # [nq * d, H] bf16 (or [nq * d, H + extra] with folded extra output columns)
    slot=None,  # optional [B, 1] int32 pool slot per row; caches are then [NSLOT, d, S] / [NSLOT, S, d]
    kv_write="flat",  # new-token K/V write when slot is given: "flat" (default: passes on trn3 hardware) |
    # "split" (no pool-size bound, but FAILS on trn3 hardware: zero/negative register strides; simulation only)
    block_table=None,  # optional [B, MP] int32 page table (paged KV prototype): caches [NPAGE, d, PG] / [NPAGE, PG, d]
    k_page_layout="dp",  # paged K page layout: "dp" (K^T [d, PG]) | "pd" (K^T rows in (partition, d-tile) order)
    pool_off=0,  # block loop only: block_table is the raw shared table; this layer's pages start at pool_off
):
    B, H = x.shape
    d = v_cache.shape[2]
    S = v_cache.shape[1]
    NSLOT = v_cache.shape[0]
    paged = block_table is not None
    if paged:  # page pool: v_cache [NPAGE, PG, d]; the attended span is MP pages
        PG = S
        MP = block_table.shape[1]
        S = MP * PG
    nq = w_gate_p.shape[2] // d
    nd = nq * d
    H1 = H // P
    DH = d // P
    R = B * nq
    NQKV = (nq + 2) * d
    n_ot = nd // P
    half = ROT // 2
    kernel_assert(d % P == 0 and R <= P and S % P == 0, f"need d%128==0, B*nq<=128, S%128==0 (d={d}, R={R}, S={S})")
    kernel_assert(ROT <= d, "rotary dims exceed head_dim")
    pooled = slot is not None
    s_row = None
    s_reg = None
    dge = {"dge_mode": nisa.dge_mode.hwdge} if (pooled or paged) and _SLOT_DGE == "hw" else {}
    bt_row = None
    if paged:
        kernel_assert(not pooled, "pass either slot (slot pool) or block_table (page pool), not both")
        kernel_assert(PG % P == 0 and PG & (PG - 1) == 0, f"page size must be a power of two >= 128, got {PG}")
        kernel_assert(d & (d - 1) == 0, f"head_dim must be a power of two with block_table, got {d}")
        kernel_assert(tuple(k_cache_t.shape) == (NSLOT, d, PG), f"k_cache_t must be [NPAGE, d, PG] = {(NSLOT, d, PG)}")
        kernel_assert(tuple(block_table.shape) == (B, MP), f"block_table must be [B, MP] int32, got {tuple(block_table.shape)}")
        kernel_assert(NSLOT * d * PG <= (1 << 31), f"paged K write needs NPAGE*d*PG <= 2^31, got {NSLOT * d * PG}")
        kernel_assert(NSLOT < (1 << 24), f"page ids must stay below 2^24 (exact fp32 lookup), NPAGE = {NSLOT}")
        kernel_assert(k_page_layout in ("dp", "pd"), f"k_page_layout must be 'dp' or 'pd', got {k_page_layout!r}")
        if k_page_layout == "pd":
            kernel_assert(SCH % PG == 0, f"k_page_layout='pd' needs PG <= {SCH}, got {PG}")
        # the whole block table on one partition: bt_row[0, b*MP + j] is the scalar_offset (page id) of row
        # b's logical page j (an SBUF scalar per DMA: the backend loads it into a DMA register itself)
        bt_row = nl.ndarray((1, B * MP), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bt_row, src=block_table.reshape((1, B * MP)))
        _PAGE_QI[0] = 0
    if pooled:
        kernel_assert(tuple(k_cache_t.shape) == (NSLOT, d, S), f"k_cache_t must be [NSLOT, d, S] = {(NSLOT, d, S)}")
        kernel_assert(tuple(slot.shape) == (B, 1), f"slot must be [B, 1] int32, got {tuple(slot.shape)}")
        kernel_assert(kv_write in ("split", "flat"), f"kv_write must be 'split' or 'flat', got {kv_write!r}")
        kernel_assert(d & (d - 1) == 0, f"head_dim must be a power of two with slot, got {d}")
        kernel_assert(d * S < (1 << 31), f"per-slot cache d*S = {d * S} must stay below 2^31 elements")
        if kv_write == "flat":
            kernel_assert(NSLOT * d * S <= (1 << 31), f"kv_write='flat' needs NSLOT*d*S <= 2^31, got {NSLOT * d * S}")
            kernel_assert(
                _FLAT_OFFSET in ("auto", "radix"), f"QWEN35_ATTN_FLAT_OFFSET must be auto|radix, got {_FLAT_OFFSET!r}"
            )
        # slot ids on one partition, then one register per row: s_reg[b] is row b's runtime scalar_offset
        # on the slot dim, loaded once and reused by all of the row's history reads and writes
        s_row = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s_row, src=slot.reshape((1, B)))
        s_reg = []
        for b in range(B):
            r = nisa.register_alloc()
            nisa.register_load(r, s_row[0:1, b : b + 1])
            s_reg.append(r)

    # ---------------- x^T [128, H1, B] ----------------
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=x_sb, src=x)
    xT = nl.ndarray((P, H1, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)

    # ---------------- QKV projection (tokens stationary) -> token-major bf16 [B, NQKV] ----------------
    MCH = 512
    qkv = nl.ndarray((P, NQKV), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, NQKV, MCH):
        cn = min(MCH, NQKV - c0)
        wq = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq[:, :, 0:cn], src=w_qkv.ap(pattern=[[NQKV, P], [P * NQKV, H1], [1, cn]], offset=c0))
        ps = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for h1 in range(H1):
            nisa.nc_matmul(ps[:, 0:cn], xT[:, h1, :], wq[:, h1, 0:cn])
        nisa.tensor_copy(dst=qkv[:, c0 : c0 + cn], src=ps[:, 0:cn])

    # ---------------- Q/K zero-centred RMSNorm per head (fp32 math, bf16 out) ----------------
    NH = nq + 1  # q heads + the k head, contiguous in qkv
    wn = nl.ndarray((P, NH, d), dtype=nl.float32, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        src = q_norm_w if hh < nq else k_norm_w
        nisa.dma_copy(dst=wn[:, hh, :], src=src.ap(pattern=[[0, B], [1, d]], offset=0))
    nisa.tensor_scalar(dst=wn, data=wn, op0=nl.add, operand0=1.0)
    ss = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)[:B]
        nisa.activation_reduce(
            dst=sq, op=nl.square, data=qkv[:, hh * d : (hh + 1) * d], reduce_op=nl.add, reduce_res=ss[:, hh : hh + 1]
        )
    rs = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / d, bias=1e-6)
    qk = nl.ndarray((P, NH, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        nisa.scalar_tensor_tensor(
            dst=qk[:, hh, :], data=qkv[:, hh * d : (hh + 1) * d], op0=nl.multiply, operand0=rs[:, hh : hh + 1],
            op1=nl.multiply, operand1=wn[:, hh, :],
        )

    # ---------------- partial RoPE on dims [0, ROT): rotate-half over those ROT dims ----------------
    cs = nl.ndarray((P, 2, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=cs[:, 0, :, :], src=cos.ap(pattern=[[half, B], [0, NH], [1, half]], offset=0))
    nisa.dma_copy(dst=cs[:, 1, :, :], src=sin.ap(pattern=[[half, B], [0, NH], [1, half]], offset=0))
    x1 = qk[:, :, 0:half]
    x2 = qk[:, :, half:ROT]
    t1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    t2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    r1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    r2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=t1, data1=x1, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x2, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r1, data1=t1, data2=t2, op=nl.subtract)  # x1 cos - x2 sin
    nisa.tensor_tensor(dst=t1, data1=x2, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x1, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r2, data1=t1, data2=t2, op=nl.add)  # x2 cos + x1 sin
    nisa.tensor_copy(dst=qk[:, :, 0:half], src=r1)
    nisa.tensor_copy(dst=qk[:, :, half:ROT], src=r2)

    # ---------------- query / new-key transposes ----------------
    # Qz[:, b, j, :] = q^T of token b (half j of d), scaled by d^-0.5, zero outside columns b*nq..b*nq+nq-1:
    # accumulating Qz_b . K_b^T over b gives every (token, head) row its own token's scores.
    scale = float(d) ** -0.5
    qT = nl.ndarray((P, DH, B, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(nq):
        for j in range(DH):
            tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=qk[:, h, j * P : (j + 1) * P])
            nisa.tensor_scalar(dst=qT[:, j, :, h], data=tp, op0=nl.multiply, operand0=scale)
    Qz = nl.ndarray((P, B, DH, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(Qz, 0.0)
    for b in range(B):
        nisa.tensor_copy(dst=Qz[:, b, :, b * nq : (b + 1) * nq], src=qT[:, :, b, :])
    kT_new = nl.ndarray((P, DH, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(DH):
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=qk[:, nq, j * P : (j + 1) * P])
        nisa.tensor_copy(dst=kT_new[:, j, :], src=tp)
    v_new = qkv[:, nd + d : nd + 2 * d]  # [B, d] bf16 (V has no norm / RoPE)

    # ---------------- attention over the history + the new token -> attn_hbm [B, nq, d] ----------------
    # A span past _SPLIT_MAX runs split-K (_split_attn, patch 0005): blocks of _SPLIT_BLOCK positions, so SBUF
    # holds one block's scores instead of the whole row's. Shorter spans keep the single pass below unchanged.
    ctx = dict(paged=paged, pooled=pooled, k_cache_t=k_cache_t, v_cache=v_cache, bt_row=bt_row,
               block_table=block_table, s_reg=s_reg, MP=MP if paged else 0, PG=PG if paged else 0, DH=DH, MCH=MCH,
               d=d, S=S, B=B, k_page_layout=k_page_layout, dge=dge, pool_off=pool_off)
    if S > _SPLIT_MAX:
        kernel_assert(_SPLIT_PASSES in ("1", "2"), f"QWEN35_ATTN_SPLIT_PASSES must be 1 or 2, got {_SPLIT_PASSES!r}")
        if block_loop_engages(paged, S, PG if paged else 0, k_page_layout):
            attn_hbm, v_row = _split_attn_online_loop(ctx, Qz, kT_new, v_new, pos, B, nq, R, d, DH, S)
        else:
            split = _split_attn_online if _SPLIT_PASSES == "1" else _split_attn
            attn_hbm, v_row = split(ctx, Qz, kT_new, v_new, pos, B, nq, R, d, DH, S)
    else:
        # ---------------- history mask: column s valid iff s < pos[b] ----------------
        pos_r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
        for b in range(B):
            nisa.dma_copy(dst=pos_r[b * nq : (b + 1) * nq, :], src=pos.ap(pattern=[[0, nq], [1, 1]], offset=b))
        pos_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_copy(dst=pos_rf, src=pos_r)
        iota_s = nl.ndarray((P, S), dtype=nl.int32, buffer=nl.sbuf)[:R]
        nisa.iota(iota_s, pattern=[[1, S]], offset=0, channel_multiplier=0)
        bias = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)[:R]  # 0 (valid) or -NEG
        nisa.tensor_scalar(dst=bias, data=iota_s, op0=nl.less, operand0=pos_rf)
        nisa.tensor_scalar(dst=bias, data=bias, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)

        # ---------------- scores [R, S + 1]: history from the cache, last column = the new token ----------------
        sc = nl.ndarray((P, S + 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        for s0 in range(0, S, SCH):
            sw = min(SCH, S - s0)
            ps_s = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.psum)[:R]
            for b in range(B):
                if paged:
                    _paged_scores(ps_s, Qz, k_cache_t, bt_row, b, s0, sw, MP, PG, DH, MCH, k_page_layout, dge)
                    continue
                kT = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
                if pooled:
                    k_src = k_cache_t.ap(
                        pattern=[[S, P], [P * S, DH], [1, sw]], offset=s0, scalar_offset=s_reg[b], indirect_dim=0
                    )
                else:
                    k_src = k_cache_t.ap(pattern=[[S, P], [P * S, DH], [1, sw]], offset=b * d * S + s0)
                nisa.dma_copy(dst=kT[:, :, 0:sw], src=k_src, **dge)
                for j in range(DH):
                    for c0 in range(0, sw, MCH):
                        cn = min(MCH, sw - c0)
                        nisa.nc_matmul(ps_s[:, c0 : c0 + cn], Qz[:, b, j, :], kT[:, j, c0 : c0 + cn])
            nisa.tensor_tensor(dst=sc[:, s0 : s0 + sw], data1=ps_s[:, 0:sw], data2=bias[:, s0 : s0 + sw], op=nl.add)
        ps_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:R]
        for b in range(B):
            for j in range(DH):
                nisa.nc_matmul(ps_n, Qz[:, b, j, :], kT_new[:, j, b : b + 1])
        nisa.tensor_copy(dst=sc[:, S : S + 1], src=ps_n)

        # ---------------- softmax over the row (fp32), normalized P in bf16 ----------------
        mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_reduce(dst=mx, op=nl.max, data=sc, axis=1)
        nisa.tensor_scalar(dst=mx, data=mx, op0=nl.multiply, operand0=-1.0)
        ex = nl.ndarray((P, S + 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        ssum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, reduce_op=nl.add, reduce_res=ssum, bias=mx)
        rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.reciprocal(dst=rsum, data=ssum)
        pb = nl.ndarray((P, S + 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rsum)

        # P^T chunks [128 positions, R] (+ the new-token column as a 1-row tile)
        NC = S // P
        PT = nl.ndarray((P, NC, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(NC):
            tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=pb[:, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=PT[:, c, :], src=tp)
        PTn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpn, data=pb[:, S : S + 1])
        nisa.tensor_copy(dst=PTn, src=tpn)
        v_hbm = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
        nisa.dma_copy(dst=v_hbm, src=v_new)
        v_row = nl.ndarray((1, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)  # new V rows on one partition
        nisa.dma_copy(dst=v_row.reshape((1, B * d)), src=v_hbm.reshape((1, B * d)))

        # ---------------- P.V per token -> attn [B, nq*d] (via a small HBM hop to token-major) ----------------
        attn_hbm = nl.ndarray((B, nq, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
        VCH = SCH // P
        for b in range(B):
            ps_o = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
            for c0 in range(0, NC, VCH):
                cn = min(VCH, NC - c0)
                vb = nl.ndarray((P, VCH, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                if paged:  # one DMA per page segment of positions c0*128 .. (c0+cn)*128
                    for jp, o, w, dc in _page_segs(c0 * P, cn * P, PG):
                        _page_dma(
                            vb[:, dc // P : (dc + w) // P, :],
                            v_cache.ap(
                                pattern=[[d, P], [P * d, w // P], [1, d]], offset=o * d,
                                scalar_offset=bt_row[0:1, b * MP + jp : b * MP + jp + 1], indirect_dim=0,
                            ),
                            dge,
                        )
                else:
                    if pooled:
                        v_src = v_cache.ap(
                            pattern=[[d, P], [P * d, cn], [1, d]], offset=c0 * P * d, scalar_offset=s_reg[b],
                            indirect_dim=0,
                        )
                    else:
                        v_src = v_cache.ap(pattern=[[d, P], [P * d, cn], [1, d]], offset=(b * S + c0 * P) * d)
                    nisa.dma_copy(dst=vb[:, 0:cn, :], src=v_src, **dge)
                for c in range(cn):
                    nisa.nc_matmul(ps_o, PT[:, c0 + c, b * nq : (b + 1) * nq], vb[:, c, :])
            nisa.nc_matmul(ps_o, PTn[:, b * nq : (b + 1) * nq], v_row[:, b, :])
            o_b = nl.ndarray((nq, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_b, src=ps_o)
            nisa.dma_copy(dst=attn_hbm[b], src=o_b)

    # ---------------- write the new token's K (transposed) / V at (slot, pos[b]) ----------------
    if paged:
        _write_kv_paged(k_cache_t, v_cache, kT_new, v_new, pos, block_table, B, d, PG, MP, k_page_layout, dge)
    elif pooled and kv_write == "split":
        _write_kv_split(k_cache_t, v_cache, kT_new, v_row, pos, s_reg, B, d, S)
    elif pooled:
        _write_kv_flat(k_cache_t, v_cache, kT_new, v_new, pos, slot, s_row, B, d, S, dge)
    else:
        _write_kv_fixed(k_cache_t, v_cache, kT_new, v_new, pos, B, d, S)

    # ---------------- gate = sigmoid(x W_gate): x^T stationary, gate weights moving ----------------
    a_sb = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=a_sb, src=attn_hbm.reshape((B, nd)))
    og = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, nd, MCH):
        cn = min(MCH, nd - c0)
        wg = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)  # streamed per column chunk
        nisa.dma_copy(dst=wg[:, :, 0:cn], src=w_gate_p[:, :, c0 : c0 + cn])
        ps_g = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for h1 in range(H1):
            nisa.nc_matmul(ps_g[:, 0:cn], xT[:, h1, :], wg[:, h1, 0:cn])
        g_bf = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        nisa.tensor_copy(dst=g_bf[:, 0:cn], src=ps_g[:, 0:cn])  # gate leaves q_proj in bf16
        sg = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)[:B]
        nisa.activation(dst=sg[:, 0:cn], op=nl.sigmoid, data=g_bf[:, 0:cn])
        nisa.tensor_tensor(dst=og[:, c0 : c0 + cn], data1=a_sb[:, c0 : c0 + cn], data2=sg[:, 0:cn], op=nl.multiply)

    # ---------------- o_proj: [B, nd] @ [nd, H] -> token-major partial ----------------
    oT = nl.ndarray((P, n_ot, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(n_ot):
        tpo = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpo, data=og[:, t * P : (t + 1) * P])
        nisa.tensor_copy(dst=oT[:, t, :], src=tpo)
    N_out = w_o.shape[1]  # H, or H + extra folded columns (low-concurrency router fold)
    out = nl.ndarray((B, N_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    o_sb = nl.ndarray((P, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, N_out, MCH):
        cn = min(MCH, N_out - c0)
        wo = nl.ndarray((P, n_ot, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)  # streamed per column chunk
        nisa.dma_copy(dst=wo[:, :, 0:cn], src=w_o.ap(pattern=[[N_out, P], [P * N_out, n_ot], [1, cn]], offset=c0))
        ps_o = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for t in range(n_ot):
            nisa.nc_matmul(ps_o[:, 0:cn], oT[:, t, :], wo[:, t, 0:cn])
        nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
    nisa.dma_copy(dst=out, src=o_sb)
    return out, k_cache_t, v_cache


def _log2(n):
    return n.bit_length() - 1


def _write_kv_fixed(k_cache_t, v_cache, kT_new, v_new, pos, B, d, S):
    """Row b -> slot b (no ``slot`` input): per-element offsets within slot b, static slot base b*d*S."""
    DH = d // P
    NSLOT = v_cache.shape[0]
    pos_bc = nl.ndarray((P, DH * B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_bc.reshape((P, DH, B)), src=pos.ap(pattern=[[0, P], [0, DH], [1, B]], offset=0))
    # k_off = S * (j*128 + p) + pos[b] within slot b (the slot base b*d*S is the DMA's static offset);
    # iota steps must fit int16, so scale by S afterwards (values < 2^24: exact even in fp32 math)
    kernel_assert(d * S < (1 << 24), "K cache element offsets must stay below 2^24 (pass slot= for larger slots)")
    k_off = nl.ndarray((P, DH * B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(k_off.reshape((P, DH, B)), pattern=[[P, DH], [0, B]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=k_off, data=k_off, op0=nl.multiply, operand0=S)
    nisa.tensor_tensor(dst=k_off, data1=k_off, data2=pos_bc, op=nl.add)
    k_flat = k_cache_t.reshape((NSLOT * d * S, 1))
    for j in range(DH):
        for b in range(B):
            nisa.dma_copy(
                dst=k_flat.ap(
                    pattern=[[1, P], [1, 1]], offset=b * d * S, vector_offset=k_off[:, j * B + b : j * B + b + 1],
                    indirect_dim=0,
                ),
                src=kT_new[:, j, b : b + 1],
                dge_mode=nisa.dge_mode.swdge,
            )
    pos_t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=pos_t, src=pos)
    v_off = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]  # b*S + pos[b]
    nisa.iota(v_off, pattern=[[0, 1]], offset=0, channel_multiplier=S)
    nisa.tensor_tensor(dst=v_off, data1=v_off, data2=pos_t, op=nl.add)
    v_flat = v_cache.reshape((NSLOT * S, d))
    nisa.dma_copy(
        dst=v_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=v_off, indirect_dim=0),
        src=v_new,
        dge_mode=nisa.dge_mode.swdge,
    )


def _write_kv_split(k_cache_t, v_cache, kT_new, v_row, pos, s_reg, B, d, S):
    """kv_write="flat": slot = scalar_offset on the slot dim, position = runtime register stride.

    Row b: rel = pos[b] - (S - 1) <= 0. K: dst [[S, 128], [rel, 2], [1, 1]] at column S-1 of row block j
    -> columns S-1 and pos of each K^T row. V: [[d, 1], [rel * d, 2], [1, d]] at row S-1 -> rows S-1 and
    pos. The source repeats the new values (stride-0 dim), so the extra element writes the same data
    into the never-valid position S-1 (or onto pos itself when pos == S-1)."""
    DH = d // P
    p_row = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_row, src=pos.reshape((1, B)))
    rel_k = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)  # pos - (S-1): |.| < S, exact
    nisa.tensor_scalar(dst=rel_k, data=p_row, op0=nl.subtract, operand0=S - 1)
    rel_v = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)  # (pos - (S-1)) * d: |.| < d*S, exact shift
    nisa.tensor_scalar(dst=rel_v, data=rel_k, op0=nl.left_shift, operand0=_log2(d))
    for b in range(B):
        s_b = s_reg[b]
        rk = nisa.register_alloc()
        nisa.register_load(rk, rel_k[0:1, b : b + 1])
        nisa.dma_copy(  # all d rows of K^T: partition p, row block j -> row j*128 + p
            dst=k_cache_t.ap(
                pattern=[[S, P], [P * S, DH], [rk, 2], [1, 1]], offset=S - 1, scalar_offset=s_b, indirect_dim=0
            ),
            src=kT_new.ap(pattern=[[DH * B, P], [B, DH], [0, 2], [1, 1]], offset=b),
            dge_mode=nisa.dge_mode.swdge,  # dynamic (register) pattern: SWDGE only
        )
        rv = nisa.register_alloc()
        nisa.register_load(rv, rel_v[0:1, b : b + 1])
        nisa.dma_copy(
            dst=v_cache.ap(pattern=[[d, 1], [rv, 2], [1, d]], offset=(S - 1) * d, scalar_offset=s_b, indirect_dim=0),
            src=v_row.ap(pattern=[[B * d, 1], [0, 2], [1, d]], offset=b * d),
            dge_mode=nisa.dge_mode.swdge,
        )


def _trailing_zeros(n):
    return (n & -n).bit_length() - 1


def _flat_off(dst, sl, ps, mult, S):
    """dst = sl * mult * S + ps (int32, same shape) from exact pieces; mult is d (K) or 1 (V).

    Power-of-two S (default): (sl << log2(mult*S)) | ps. Otherwise (or QWEN35_ATTN_FLAT_OFFSET=radix):
    S = m * 2^k, hi = sl * (mult*m) + (ps >> k), dst = (hi << k) | (ps & (2^k - 1)). Every add / multiply
    result is < NSLOT*mult*S / 2^k <= 2^31 / 2^7 = 2^24 (exact even through an fp32 datapath); the
    values >= 2^24 only ever come out of shifts / ors."""
    if S & (S - 1) == 0 and _FLAT_OFFSET == "auto":
        nisa.tensor_scalar(dst=dst, data=sl, op0=nl.left_shift, operand0=_log2(mult * S))
        nisa.tensor_tensor(dst=dst, data1=dst, data2=ps, op=nl.bitwise_or)
        return
    k = _trailing_zeros(S)
    m = S >> k
    shp = tuple(dst.shape)
    lo = nl.ndarray(shp, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo, data=ps, op0=nl.right_shift, operand0=k)  # pos >> k  (< m)
    nisa.tensor_scalar(dst=dst, data=sl, op0=nl.multiply, operand0=mult * m)  # slot * mult*m  (< 2^24)
    nisa.tensor_tensor(dst=dst, data1=dst, data2=lo, op=nl.add)  # hi  (< 2^24)
    nisa.tensor_scalar(dst=dst, data=dst, op0=nl.left_shift, operand0=k)
    nisa.tensor_scalar(dst=lo, data=ps, op0=nl.bitwise_and, operand0=(1 << k) - 1)
    nisa.tensor_tensor(dst=dst, data1=dst, data2=lo, op=nl.bitwise_or)


def _write_kv_flat(k_cache_t, v_cache, kT_new, v_new, pos, slot, s_row, B, d, S, dge):
    """kv_write="flat": one combined int32 element offset per row (exact pieces, see ``_flat_off``).

    K: scalar offset slot*d*S + pos on the flat cache (rows j*128 + p static in the pattern).
    V: vector offset slot*S + pos over the B rows of the [NSLOT*S, d] view."""
    NSLOT = v_cache.shape[0]
    DH = d // P
    kernel_assert(
        NSLOT * d * S <= (1 << (24 + _trailing_zeros(S))), "flat offset pieces must stay below 2^24 (exact in fp32)"
    )
    p_row = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_row, src=pos.reshape((1, B)))
    xk = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    _flat_off(xk, s_row, p_row, d, S)
    k_flat = k_cache_t.reshape((NSLOT * d * S, 1))
    for b in range(B):
        xr = nisa.register_alloc()
        nisa.register_load(xr, xk[0:1, b : b + 1])
        nisa.dma_copy(
            dst=k_flat.ap(pattern=[[S, P], [P * S, DH], [1, 1]], offset=0, scalar_offset=xr, indirect_dim=0),
            src=kT_new.ap(pattern=[[DH * B, P], [B, DH], [1, 1]], offset=b),
            **dge,
        )
    s_t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=s_t, src=slot)
    pos_t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=pos_t, src=pos)
    v_off = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]  # slot[b] * S + pos[b]
    _flat_off(v_off, s_t, pos_t, 1, S)
    v_flat = v_cache.reshape((NSLOT * S, d))
    nisa.dma_copy(
        dst=v_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=v_off, indirect_dim=0),
        src=v_new,
        dge_mode=nisa.dge_mode.swdge,
    )


def _block_scores(ctx, Qz, pos_rf, R, s0, kw):
    """[R, kw] fp32 scores of history positions s0 .. s0 + kw - 1 with the history mask added (0, or -NEG past a
    row's position), from the slot / page / static cache as the single pass reads it."""
    B, DH, MCH, d, S = ctx["B"], ctx["DH"], ctx["MCH"], ctx["d"], ctx["S"]
    iota_s = nl.ndarray((P, kw), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(iota_s, pattern=[[1, kw]], offset=s0, channel_multiplier=0)
    bias = nl.ndarray((P, kw), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=bias, data=iota_s, op0=nl.less, operand0=pos_rf)
    nisa.tensor_scalar(dst=bias, data=bias, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)
    sc = nl.ndarray((P, kw), dtype=nl.float32, buffer=nl.sbuf)[:R]
    for c0 in range(s0, s0 + kw, SCH):
        sw = min(SCH, s0 + kw - c0)
        ps_s = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.psum)[:R]
        for b in range(B):
            if ctx["paged"]:
                _paged_scores(ps_s, Qz, ctx["k_cache_t"], ctx["bt_row"], b, c0, sw, ctx["MP"], ctx["PG"], DH, MCH,
                              ctx["k_page_layout"], ctx["dge"])
                continue
            kT = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
            if ctx["pooled"]:
                k_src = ctx["k_cache_t"].ap(pattern=[[S, P], [P * S, DH], [1, sw]], offset=c0,
                                            scalar_offset=ctx["s_reg"][b], indirect_dim=0)
            else:
                k_src = ctx["k_cache_t"].ap(pattern=[[S, P], [P * S, DH], [1, sw]], offset=b * d * S + c0)
            nisa.dma_copy(dst=kT[:, :, 0:sw], src=k_src, **ctx["dge"])
            for j in range(DH):
                for m0 in range(0, sw, MCH):
                    cn = min(MCH, sw - m0)
                    nisa.nc_matmul(ps_s[:, m0 : m0 + cn], Qz[:, b, j, :], kT[:, j, m0 : m0 + cn])
        nisa.tensor_tensor(dst=sc[:, c0 - s0 : c0 - s0 + sw], data1=ps_s[:, 0:sw], data2=bias[:, c0 - s0 : c0 - s0 + sw],
                           op=nl.add)
    return sc


def _block_values(ctx, vb, b, c0, cn):
    """vb[:, 0:cn, :] = row b's V rows of positions c0*128 .. (c0+cn)*128 - 1 (as the single pass reads them)."""
    d, S, dge = ctx["d"], ctx["S"], ctx["dge"]
    if ctx["paged"]:
        MP, PG = ctx["MP"], ctx["PG"]
        for jp, o, w, dc in _page_segs(c0 * P, cn * P, PG):
            _page_dma(
                vb[:, dc // P : (dc + w) // P, :],
                ctx["v_cache"].ap(pattern=[[d, P], [P * d, w // P], [1, d]], offset=o * d,
                                  scalar_offset=ctx["bt_row"][0:1, b * MP + jp : b * MP + jp + 1], indirect_dim=0),
                dge,
            )
        return
    if ctx["pooled"]:
        v_src = ctx["v_cache"].ap(pattern=[[d, P], [P * d, cn], [1, d]], offset=c0 * P * d,
                                  scalar_offset=ctx["s_reg"][b], indirect_dim=0)
    else:
        v_src = ctx["v_cache"].ap(pattern=[[d, P], [P * d, cn], [1, d]], offset=(b * S + c0 * P) * d)
    nisa.dma_copy(dst=vb[:, 0:cn, :], src=v_src, **dge)


def _split_attn(ctx, Qz, kT_new, v_new, pos, B, nq, R, d, DH, S):
    """Split-K attention over a long span (patch 0005). Two passes over blocks of _SPLIT_BLOCK history positions:

    1. scores of each block, folded into each row's running max M and running sum L = sum exp(s - M) (L rescaled by
       exp(M_old - M_new) when the max moves), with the new token's own column first;
    2. scores again, P = bf16(exp(s - M) / L) of each block (the single pass's normalized probabilities, up to the
       order L's fp32 sum was taken in), P.V of each row accumulated in fp32 over the blocks, then the new token.

    Only one block's scores, probabilities and P^T live in SBUF. Returns (attn_hbm [B, nq, d] bf16, v_row)."""
    KB = _SPLIT_BLOCK
    kernel_assert(KB % SCH == 0 and S % P == 0, f"QWEN35_ATTN_SPLIT_BLOCK {KB} must be a multiple of {SCH}")
    pos_r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    for b in range(B):
        nisa.dma_copy(dst=pos_r[b * nq : (b + 1) * nq, :], src=pos.ap(pattern=[[0, nq], [1, 1]], offset=b))
    pos_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pos_rf, src=pos_r)
    # the new token's score: the first (M, L) = (s_new, 1)
    ps_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:R]
    for b in range(B):
        for j in range(DH):
            nisa.nc_matmul(ps_n, Qz[:, b, j, :], kT_new[:, j, b : b + 1])
    s_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=s_new, src=ps_n)
    M = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=M, src=s_new)
    Lsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.memset(Lsum, 1.0)
    blocks = [(s0, min(KB, S - s0)) for s0 in range(0, S, KB)]
    # ---- pass 1: running max and sum ----
    for s0, kw in blocks:
        sc = _block_scores(ctx, Qz, pos_rf, R, s0, kw)
        mb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_reduce(dst=mb, op=nl.max, data=sc, axis=1)
        Mn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=Mn, data1=M, data2=mb, op=nl.maximum)
        nMn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nMn, data=Mn, op0=nl.multiply, operand0=-1.0)
        corr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation(dst=corr, op=nl.exp, data=M, bias=nMn)  # exp(M_old - M_new)
        ex = nl.ndarray((P, kw), dtype=nl.float32, buffer=nl.sbuf)[:R]
        sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, reduce_op=nl.add, reduce_res=sb, bias=nMn)
        nisa.scalar_tensor_tensor(dst=Lsum, data=Lsum, op0=nl.multiply, operand0=corr, op1=nl.add, operand1=sb)
        nisa.tensor_copy(dst=M, src=Mn)
    nM = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=nM, data=M, op0=nl.multiply, operand0=-1.0)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.reciprocal(dst=rsum, data=Lsum)
    # ---- pass 2: normalized P of each block, P.V accumulated per row in fp32 ----
    acc = nl.ndarray((nq, B, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(acc, 0.0)
    VCH = SCH // P
    for s0, kw in blocks:
        sc = _block_scores(ctx, Qz, pos_rf, R, s0, kw)
        ex = nl.ndarray((P, kw), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation(dst=ex, op=nl.exp, data=sc, bias=nM)
        pb = nl.ndarray((P, kw), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rsum)
        NCb = kw // P
        PT = nl.ndarray((P, NCb, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(NCb):
            tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=pb[:, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=PT[:, c, :], src=tp)
        c_base = s0 // P
        for b in range(B):
            ps_o = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
            for c0 in range(0, NCb, VCH):
                cn = min(VCH, NCb - c0)
                vb = nl.ndarray((P, VCH, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                _block_values(ctx, vb, b, c_base + c0, cn)
                for c in range(cn):
                    nisa.nc_matmul(ps_o, PT[:, c0 + c, b * nq : (b + 1) * nq], vb[:, c, :])
            nisa.tensor_tensor(dst=acc[:, b, :], data1=acc[:, b, :], data2=ps_o, op=nl.add)
    # ---- the new token ----
    en = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.activation(dst=en, op=nl.exp, data=s_new, bias=nM)
    pn = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=pn, data=en, op0=nl.multiply, operand0=rsum)
    PTn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    tpn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tpn, data=pn)
    nisa.tensor_copy(dst=PTn, src=tpn)
    v_hbm = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=v_hbm, src=v_new)
    v_row = nl.ndarray((1, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_row.reshape((1, B * d)), src=v_hbm.reshape((1, B * d)))
    attn_hbm = nl.ndarray((B, nq, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for b in range(B):
        ps_v = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_v, PTn[:, b * nq : (b + 1) * nq], v_row[:, b, :])
        o32 = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o32, data1=acc[:, b, :], data2=ps_v, op=nl.add)
        o_b = nl.ndarray((nq, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_b, src=o32)
        nisa.dma_copy(dst=attn_hbm[b], src=o_b)
    return attn_hbm, v_row


def _row_to_heads(sel, mask, vals, R, nq, B):
    """[nq, B] fp32 of a per-row column vals [R, 1] (row r = b * nq + h -> [h, b]): one matmul with the 0/1 selector
    sel [R, nq] stationary (exact for an fp32 moving operand) over mask [R, B] * vals."""
    mv = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=mv, data=mask, op0=nl.multiply, operand0=vals)
    ps = nl.ndarray((nq, B), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(ps, sel, mv, is_stationary_onezero=True)
    out = nl.ndarray((nq, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


def _split_attn_online(ctx, Qz, kT_new, v_new, pos, B, nq, R, d, DH, S):
    """Split-K attention in one pass over blocks of _SPLIT_BLOCK history positions (patch 0006): online softmax.

    Each block's scores fold into each row's running max M and sum L (both rescaled by corr = exp(M_old - M_new) when
    the max moves), and its unnormalized P = bf16(exp(s - M_new)) times V into the row's fp32 accumulator, which is
    rescaled by the same corr first. The new token's column seeds (M, L) = (s_new, 1); the output is
    (acc + p_new v_new) / L. Each block's K and V are read once (patch 0005's two-pass form reads K twice).

    The rows' (M, L) live on R = B * nq partitions, the accumulator on nq partitions per row ([nq, B, d], as P.V
    lands): corr and 1/L move there through _row_to_heads. Returns (attn_hbm [B, nq, d] bf16, v_row)."""
    KB = _SPLIT_BLOCK
    kernel_assert(KB % SCH == 0 and S % P == 0, f"QWEN35_ATTN_SPLIT_BLOCK {KB} must be a multiple of {SCH}")
    pos_r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    for b in range(B):
        nisa.dma_copy(dst=pos_r[b * nq : (b + 1) * nq, :], src=pos.ap(pattern=[[0, nq], [1, 1]], offset=b))
    pos_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pos_rf, src=pos_r)
    # row r = b * nq + h: mask[r, b'] = (0 <= r - b' * nq < nq), sel[r, h'] = sum_b' (r - b' * nq == h')
    rb_i = nl.ndarray((P, B), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(rb_i, pattern=[[-nq, B]], offset=0, channel_multiplier=1)
    rb = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=rb, src=rb_i)
    ge = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=ge, data=rb, op0=nl.greater_equal, operand0=0.0)
    mask = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=mask, data=rb, op0=nl.less, operand0=float(nq))
    nisa.tensor_tensor(dst=mask, data1=mask, data2=ge, op=nl.multiply)
    sel = nl.ndarray((P, nq), dtype=nl.float32, buffer=nl.sbuf)[:R]
    for h in range(nq):
        eq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=eq, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.greater_equal, operand1=0.0)
        lt = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=lt, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.less, operand1=1.0)
        nisa.tensor_tensor(dst=eq, data1=eq, data2=lt, op=nl.multiply)
        nisa.tensor_reduce(dst=sel[:, h : h + 1], op=nl.add, data=eq, axis=1)
    # the new token's score: the first (M, L) = (s_new, 1)
    ps_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:R]
    for b in range(B):
        for j in range(DH):
            nisa.nc_matmul(ps_n, Qz[:, b, j, :], kT_new[:, j, b : b + 1])
    s_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=s_new, src=ps_n)
    M = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=M, src=s_new)
    Lsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.memset(Lsum, 1.0)
    acc = nl.ndarray((nq, B, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(acc, 0.0)
    VCH = SCH // P
    for s0 in range(0, S, KB):
        kw = min(KB, S - s0)
        sc = _block_scores(ctx, Qz, pos_rf, R, s0, kw)
        mb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_reduce(dst=mb, op=nl.max, data=sc, axis=1)
        Mn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=Mn, data1=M, data2=mb, op=nl.maximum)
        nMn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nMn, data=Mn, op0=nl.multiply, operand0=-1.0)
        corr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation(dst=corr, op=nl.exp, data=M, bias=nMn)  # exp(M_old - M_new)
        ex = nl.ndarray((P, kw), dtype=nl.float32, buffer=nl.sbuf)[:R]
        sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, reduce_op=nl.add, reduce_res=sb, bias=nMn)
        nisa.scalar_tensor_tensor(dst=Lsum, data=Lsum, op0=nl.multiply, operand0=corr, op1=nl.add, operand1=sb)
        nisa.tensor_copy(dst=M, src=Mn)
        pb = nl.ndarray((P, kw), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
        nisa.tensor_copy(dst=pb, src=ex)
        NCb = kw // P
        PT = nl.ndarray((P, NCb, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(NCb):
            tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=pb[:, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=PT[:, c, :], src=tp)
        corr_h = _row_to_heads(sel, mask, corr, R, nq, B)
        c_base = s0 // P
        for b in range(B):
            ps_o = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
            for c0 in range(0, NCb, VCH):
                cn = min(VCH, NCb - c0)
                vb = nl.ndarray((P, VCH, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                _block_values(ctx, vb, b, c_base + c0, cn)
                for c in range(cn):
                    nisa.nc_matmul(ps_o, PT[:, c0 + c, b * nq : (b + 1) * nq], vb[:, c, :])
            nisa.scalar_tensor_tensor(dst=acc[:, b, :], data=acc[:, b, :], op0=nl.multiply,
                                      operand0=corr_h[:, b : b + 1], op1=nl.add, operand1=ps_o)
    # ---- the new token, then / L ----
    nM = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=nM, data=M, op0=nl.multiply, operand0=-1.0)
    en = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.activation(dst=en, op=nl.exp, data=s_new, bias=nM)
    pn = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pn, src=en)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.reciprocal(dst=rsum, data=Lsum)
    rs_h = _row_to_heads(sel, mask, rsum, R, nq, B)
    PTn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    tpn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tpn, data=pn)
    nisa.tensor_copy(dst=PTn, src=tpn)
    v_hbm = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=v_hbm, src=v_new)
    v_row = nl.ndarray((1, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_row.reshape((1, B * d)), src=v_hbm.reshape((1, B * d)))
    attn_hbm = nl.ndarray((B, nq, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for b in range(B):
        ps_v = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_v, PTn[:, b * nq : (b + 1) * nq], v_row[:, b, :])
        o32 = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o32, data1=acc[:, b, :], data2=ps_v, op=nl.add)
        o_b = nl.ndarray((nq, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_b, data=o32, op0=nl.multiply, operand0=rs_h[:, b : b + 1])
        nisa.dma_copy(dst=attn_hbm[b], src=o_b)
    return attn_hbm, v_row


def _split_attn_online_loop(ctx, Qz, kT_new, v_new, pos, B, nq, R, d, DH, S):
    """_split_attn_online with the blocks iterated in a run-time loop (patch 0007, the GLM-5.2 decode kernel's pattern): one copy
    of the block body in the program instead of one per block, so the instruction count stops growing with the
    span (a loop pass costs little on trn3 against the body at the 16-row bucket). Paged "dp" only, whole
    blocks (S a multiple of KB = _SPLIT_BLOCK, KB a multiple of PG); the caller keeps the unrolled form otherwise.

    The body is the unrolled one with the block base s0 gone from the trace: the pass's NPB = KB / PG block-table
    columns come from one dynamic-offset DMA (scalar_offset the loop-carried first page, on the flat table) into
    bt_p, whose static slices are the K / V DMAs' page ids exactly as bt_row's are un-looped, and the history mask
    compares the block-local iota with pos - s0 (both exact fp32 integers below 2^24) instead of the absolute iota
    with pos — the same booleans, so the same scores bit for bit. M, L and the fp32 accumulator are loop-carried
    in SBUF (rescaled in place as un-looped); s0f / pg0 carry the block's first position / page. No collectives in
    the body (trn3 bans them in device loops). Returns (attn_hbm [B, nq, d] bf16, v_row)."""
    KB = _SPLIT_BLOCK
    kernel_assert(KB % SCH == 0 and S % P == 0, f"QWEN35_ATTN_SPLIT_BLOCK {KB} must be a multiple of {SCH}")
    PG, MP, MCH = ctx["PG"], ctx["MP"], ctx["MCH"]
    NPB = KB // PG  # block-table columns (pages) per block
    NB = S // KB
    pos_r = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    for b in range(B):
        nisa.dma_copy(dst=pos_r[b * nq : (b + 1) * nq, :], src=pos.ap(pattern=[[0, nq], [1, 1]], offset=b))
    pos_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pos_rf, src=pos_r)
    # row r = b * nq + h: mask[r, b'] = (0 <= r - b' * nq < nq), sel[r, h'] = sum_b' (r - b' * nq == h')
    rb_i = nl.ndarray((P, B), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(rb_i, pattern=[[-nq, B]], offset=0, channel_multiplier=1)
    rb = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=rb, src=rb_i)
    ge = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=ge, data=rb, op0=nl.greater_equal, operand0=0.0)
    mask = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=mask, data=rb, op0=nl.less, operand0=float(nq))
    nisa.tensor_tensor(dst=mask, data1=mask, data2=ge, op=nl.multiply)
    sel = nl.ndarray((P, nq), dtype=nl.float32, buffer=nl.sbuf)[:R]
    for h in range(nq):
        eq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=eq, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.greater_equal, operand1=0.0)
        lt = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=lt, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.less, operand1=1.0)
        nisa.tensor_tensor(dst=eq, data1=eq, data2=lt, op=nl.multiply)
        nisa.tensor_reduce(dst=sel[:, h : h + 1], op=nl.add, data=eq, axis=1)
    # the new token's score: the first (M, L) = (s_new, 1)
    ps_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:R]
    for b in range(B):
        for j in range(DH):
            nisa.nc_matmul(ps_n, Qz[:, b, j, :], kT_new[:, j, b : b + 1])
    s_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=s_new, src=ps_n)
    M = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=M, src=s_new)
    Lsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.memset(Lsum, 1.0)
    acc = nl.ndarray((nq, B, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(acc, 0.0)
    VCH = SCH // P
    # the loop: s0f = the pass's first position (fp32, exact below 2^24), pg0 = its first block-table column
    s0f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(s0f, 0.0)
    pg0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(pg0, 0)
    passc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)  # passes done (while-loop condition)
    nisa.memset(passc, 0.0)
    bt_flat = ctx["block_table"].reshape((B * MP, 1))

    def cond_reg():
        more = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=more, data=passc, op0=nl.less, operand0=float(NB))
        more_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=more_i, src=more)
        r = nisa.register_alloc()
        nisa.register_load(r, more_i)
        return r

    def block_pass(_):
        # bt_p[c, b] = block_table[b, pg0 + c]: one dynamic-offset DMA of the pass's NPB columns
        bt_p = nl.ndarray((NPB, B), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bt_p,
            src=bt_flat.ap(pattern=[[1, NPB], [MP, B]], offset=0, scalar_offset=pg0, indirect_dim=0),
        )
        if ctx["pool_off"]:  # raw shared table: this layer's pool pages start at pool_off (int32 exact)
            nisa.tensor_scalar(dst=bt_p, data=bt_p, op0=nl.add, operand0=ctx["pool_off"])
        pos_rel = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]  # pos - s0
        nisa.tensor_tensor(dst=pos_rel, data1=pos_rf, data2=s0f[0:R], op=nl.subtract)
        # ---- scores [R, KB] with the history mask (block-local iota vs pos - s0) ----
        sc = nl.ndarray((P, KB), dtype=nl.float32, buffer=nl.sbuf)[:R]
        for c0 in range(0, KB, SCH):
            iota_s = nl.ndarray((P, SCH), dtype=nl.int32, buffer=nl.sbuf)[:R]
            nisa.iota(iota_s, pattern=[[1, SCH]], offset=c0, channel_multiplier=0)
            bias = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.sbuf)[:R]  # 0 (valid) or -NEG
            nisa.tensor_scalar(dst=bias, data=iota_s, op0=nl.less, operand0=pos_rel)
            nisa.tensor_scalar(dst=bias, data=bias, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)
            ps_s = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.psum)[:R]
            for b in range(B):
                kT = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
                for jp, o, w, dc in _page_segs(c0, SCH, PG):
                    _page_dma(
                        kT[:, :, dc : dc + w],
                        ctx["k_cache_t"].ap(pattern=[[PG, P], [P * PG, DH], [1, w]], offset=o,
                                            scalar_offset=bt_p[jp : jp + 1, b : b + 1], indirect_dim=0),
                        ctx["dge"],
                    )
                for j in range(DH):
                    for m0 in range(0, SCH, MCH):
                        cn = min(MCH, SCH - m0)
                        nisa.nc_matmul(ps_s[:, m0 : m0 + cn], Qz[:, b, j, :], kT[:, j, m0 : m0 + cn])
            nisa.tensor_tensor(dst=sc[:, c0 : c0 + SCH], data1=ps_s, data2=bias, op=nl.add)
        # ---- the online update of the unrolled body, unchanged ----
        mb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_reduce(dst=mb, op=nl.max, data=sc, axis=1)
        Mn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=Mn, data1=M, data2=mb, op=nl.maximum)
        nMn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nMn, data=Mn, op0=nl.multiply, operand0=-1.0)
        corr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation(dst=corr, op=nl.exp, data=M, bias=nMn)  # exp(M_old - M_new)
        ex = nl.ndarray((P, KB), dtype=nl.float32, buffer=nl.sbuf)[:R]
        sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, reduce_op=nl.add, reduce_res=sb, bias=nMn)
        nisa.scalar_tensor_tensor(dst=Lsum, data=Lsum, op0=nl.multiply, operand0=corr, op1=nl.add, operand1=sb)
        nisa.tensor_copy(dst=M, src=Mn)
        pb = nl.ndarray((P, KB), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
        nisa.tensor_copy(dst=pb, src=ex)
        NCb = KB // P
        PT = nl.ndarray((P, NCb, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(NCb):
            tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=pb[:, c * P : (c + 1) * P])
            nisa.tensor_copy(dst=PT[:, c, :], src=tp)
        corr_h = _row_to_heads(sel, mask, corr, R, nq, B)
        for b in range(B):
            ps_o = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
            for c0 in range(0, NCb, VCH):
                cn = min(VCH, NCb - c0)
                vb = nl.ndarray((P, VCH, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                for jp, o, w, dc in _page_segs(c0 * P, cn * P, PG):
                    _page_dma(
                        vb[:, dc // P : (dc + w) // P, :],
                        ctx["v_cache"].ap(pattern=[[d, P], [P * d, w // P], [1, d]], offset=o * d,
                                          scalar_offset=bt_p[jp : jp + 1, b : b + 1], indirect_dim=0),
                        ctx["dge"],
                    )
                for c in range(cn):
                    nisa.nc_matmul(ps_o, PT[:, c0 + c, b * nq : (b + 1) * nq], vb[:, c, :])
            nisa.scalar_tensor_tensor(dst=acc[:, b, :], data=acc[:, b, :], op0=nl.multiply,
                                      operand0=corr_h[:, b : b + 1], op1=nl.add, operand1=ps_o)
        nisa.tensor_scalar(dst=s0f, data=s0f, op0=nl.add, operand0=float(KB))
        nisa.tensor_scalar(dst=pg0, data=pg0, op0=nl.add, operand0=NPB)
        nisa.tensor_scalar(dst=passc, data=passc, op0=nl.add, operand0=1.0)
        return cond_reg()

    nl.while_loop(cond_reg(), block_pass)
    # ---- the new token, then / L (the unrolled epilogue, unchanged) ----
    nM = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=nM, data=M, op0=nl.multiply, operand0=-1.0)
    en = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.activation(dst=en, op=nl.exp, data=s_new, bias=nM)
    pn = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=pn, src=en)
    rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.reciprocal(dst=rsum, data=Lsum)
    rs_h = _row_to_heads(sel, mask, rsum, R, nq, B)
    PTn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    tpn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tpn, data=pn)
    nisa.tensor_copy(dst=PTn, src=tpn)
    v_hbm = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=v_hbm, src=v_new)
    v_row = nl.ndarray((1, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_row.reshape((1, B * d)), src=v_hbm.reshape((1, B * d)))
    attn_hbm = nl.ndarray((B, nq, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for b in range(B):
        ps_v = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_v, PTn[:, b * nq : (b + 1) * nq], v_row[:, b, :])
        o32 = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o32, data1=acc[:, b, :], data2=ps_v, op=nl.add)
        o_b = nl.ndarray((nq, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_b, data=o32, op0=nl.multiply, operand0=rs_h[:, b : b + 1])
        nisa.dma_copy(dst=attn_hbm[b], src=o_b)
    return attn_hbm, v_row


def _page_segs(s0, sw, PG):
    """Split logical positions [s0, s0 + sw) at page boundaries: [(page j, offset in page, width, offset in chunk)]."""
    segs = []
    s = s0
    while s < s0 + sw:
        jp = s // PG
        o = s - jp * PG
        w = min(PG - o, s0 + sw - s)
        segs.append((jp, o, w, s - s0))
        s += w
    return segs


def _paged_scores(ps_s, Qz, k_cache_t, bt_row, b, s0, sw, MP, PG, DH, MCH, layout, dge):
    """Row b's K^T for score columns [s0, s0 + sw) from its pages, accumulated into ps_s (paged KV).

    "dp": one DMA per page segment into the slot path's [128, DH, SCH] tile (runs of 2*w bytes), same
    matmuls as the slot path. "pd": one DMA per page into [128, pages, DH*PG] (one 2*DH*PG-byte run per
    partition), score matmuls per (page, d-tile)."""
    bt = lambda jp: bt_row[0:1, b * MP + jp : b * MP + jp + 1]
    if layout == "dp":
        kT = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        for jp, o, w, dc in _page_segs(s0, sw, PG):
            _page_dma(
                kT[:, :, dc : dc + w],
                k_cache_t.ap(pattern=[[PG, P], [P * PG, DH], [1, w]], offset=o, scalar_offset=bt(jp), indirect_dim=0),
                dge,
            )
        for j in range(DH):
            for c0 in range(0, sw, MCH):
                cn = min(MCH, sw - c0)
                nisa.nc_matmul(ps_s[:, c0 : c0 + cn], Qz[:, b, j, :], kT[:, j, c0 : c0 + cn])
        return
    npc = sw // PG  # s0, sw are multiples of PG here (PG divides SCH)
    kTp = nl.ndarray((P, SCH // PG, DH * PG), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(npc):
        _page_dma(
            kTp[:, c, :],
            k_cache_t.ap(pattern=[[DH * PG, P], [1, DH * PG]], offset=0, scalar_offset=bt(s0 // PG + c), indirect_dim=0),
            dge,
        )
    g = max(1, MCH // PG) if _PD_GROUP else 1  # pages per matmul (a [128, g, PG] strided moving operand)
    for j in range(DH):  # d-tile outer: consecutive matmuls share the stationary Qz[:, b, j]
        for c in range(0, npc, g):
            gn = min(g, npc - c)
            if gn > 1:
                nisa.nc_matmul(ps_s[:, c * PG : (c + gn) * PG], Qz[:, b, j, :], kTp[:, c : c + gn, j * PG : (j + 1) * PG])
                continue
            for c0 in range(0, PG, MCH):
                cn = min(MCH, PG - c0)
                nisa.nc_matmul(
                    ps_s[:, c * PG + c0 : c * PG + c0 + cn], Qz[:, b, j, :], kTp[:, c, j * PG + c0 : j * PG + c0 + cn]
                )


def _page_loc(block_table, pos, np_, nb, MP, PG, shift):
    """Write location of each row's new token, (page << shift) | (pos & (PG-1)) with page = bt[b, pos >> log2 PG].

    Tile layout [np_, nb] with np_ * nb == B rows: (1, B) = all rows on one partition (per-row DMA scalar
    offsets), (B, 1) = one row per partition (vector offsets). The page is selected by a one-hot over the
    row's MP block-table entries (0 <= pos - j*PG < PG) and a sum: exact in fp32 for ids < 2^24."""
    bt_i = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)
    pos_rep = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)
    pos_i = nl.ndarray((np_, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bt_i, src=block_table.ap(pattern=[[nb * MP, np_], [MP, nb], [1, MP]], offset=0))
    nisa.dma_copy(dst=pos_rep, src=pos.ap(pattern=[[nb, np_], [1, nb], [0, MP]], offset=0))
    nisa.dma_copy(dst=pos_i, src=pos.ap(pattern=[[nb, np_], [1, nb]], offset=0))
    lo = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)  # j * PG
    nisa.iota(lo, pattern=[[0, nb], [PG, MP]], offset=0, channel_multiplier=0)
    dif = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)  # pos - j*PG
    nisa.tensor_tensor(dst=dif, data1=pos_rep, data2=lo, op=nl.subtract)
    ge = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ge, data=dif, op0=nl.greater_equal, operand0=0.0)
    oh = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh, data=dif, op0=nl.less, operand0=float(PG))
    nisa.tensor_tensor(dst=oh, data1=oh, data2=ge, op=nl.multiply)
    btf = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=btf, src=bt_i)
    nisa.tensor_tensor(dst=oh, data1=oh, data2=btf, op=nl.multiply)
    pgf = nl.ndarray((np_, nb), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(nb):
        nisa.tensor_reduce(dst=pgf[:, i : i + 1], op=nl.add, data=oh[:, i, :], axis=1)
    loc = nl.ndarray((np_, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=loc, src=pgf)
    nisa.tensor_scalar(dst=loc, data=loc, op0=nl.left_shift, operand0=shift)
    off = nl.ndarray((np_, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=off, data=pos_i, op0=nl.bitwise_and, operand0=PG - 1)
    nisa.tensor_tensor(dst=loc, data1=loc, data2=off, op=nl.bitwise_or)
    return loc


def _write_kv_paged(k_cache_t, v_cache, kT_new, v_new, pos, block_table, B, d, PG, MP, layout, dge):
    """Paged KV: the new token's K^T column / V row at (bt[b, pos >> log2 PG], pos & (PG-1)).

    K: one combined element offset (page << log2(d*PG)) | off per row as the scalar offset on the flat
    pool (rows static in the pattern; "pd" stores dim j*128+p at page row p*DH + j). V: row offsets
    (page << log2 PG) | off over the [NPAGE*PG, d] view, one vector-offset DMA for all rows."""
    NPAGE = v_cache.shape[0]
    DH = d // P
    xk = _page_loc(block_table, pos, 1, B, MP, PG, _log2(d * PG))
    k_flat = k_cache_t.reshape((NPAGE * d * PG, 1))
    kpat = [[PG, P], [P * PG, DH], [1, 1]] if layout == "dp" else [[DH * PG, P], [PG, DH], [1, 1]]
    for b in range(B):
        xr = nisa.register_alloc()
        nisa.register_load(xr, xk[0:1, b : b + 1])
        nisa.dma_copy(
            dst=k_flat.ap(pattern=kpat, offset=0, scalar_offset=xr, indirect_dim=0),
            src=kT_new.ap(pattern=[[DH * B, P], [B, DH], [1, 1]], offset=b),
            **dge,
        )
    v_off = _page_loc(block_table, pos, B, 1, MP, PG, _log2(PG))
    v_flat = v_cache.reshape((NPAGE * PG, d))
    nisa.dma_copy(
        dst=v_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=v_off, indirect_dim=0),
        src=v_new,
        dge_mode=nisa.dge_mode.swdge,
    )


# ---------------------------------------------------------------------------
# Decode context parallelism (DCP, kv_cp = G > 1; the GLM-5.2 in-kernel pattern, patch 0009).
#
# Every CP rank holds a 128-token block (CPB = 2048 / G) of every 2048-token native page (owner of token t:
# rank (t % 2048) // CPB), projects Q/K/V for its 2 query + both KV heads, attends online-split-K over its
# LOCAL share of the paged history (the block table is the same native-page table on every rank; this rank's
# local block lb is table column lb), and emits unnormalized (M, L, acc) per chunk of 2 query heads. The
# layer collectives merge: AllGather q (rank-as-data, one NEFF), per-rank partials, AllGather (M, L), the
# combine scale, a ReduceScatter of the scaled partials over the head rows, then the gate + o_proj tail.
# Owner-only new-token writes: the owner's K/V lands at its block's column, every other rank's write drops
# into the sink pool row (cp_sink, never read). Rank-as-data: no rank expression reaches an index the
# compiler must constant-fold; cp_rank enters only through data (one-hot selects, broadcast compares).
#
# Numerics: the empty-share sink seeds (M, L) = (-NEG, 1) and the CP bias masks with -1e30 (not -NEG: an
# empty rank's running max would otherwise track masked scores at ~-NEG and let exp(score - M) of masked
# columns overflow); masked columns are then exactly exp(-1e30 - M) = 0, so an empty rank contributes
# exactly s * e^(m - M*) = 1 * e^(-30000 - M*) = 0 (fp32 underflow) in the merge.
KV_CP = int(os.environ.get("QWEN35_KV_CP", "1"))
_CP_NEG = 1e30  # masked-column bias under DCP (see above)


def _cp_own(pos_t, rk_t, NPB, CPB):
    """0/1 fp32 tile: this rank owns the row's new token ((pos % NPB) >> log2 CPB == rank). int32 exact."""
    t = nl.ndarray(pos_t.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=pos_t, op0=nl.bitwise_and, operand0=NPB - 1)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.right_shift, operand0=_log2(CPB))
    dif = nl.ndarray(pos_t.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dif, data1=t, data2=rk_t, op=nl.subtract)
    ge = nl.ndarray(pos_t.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ge, data=dif, op0=nl.greater_equal, operand0=0.0)
    own = nl.ndarray(pos_t.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=own, data=dif, op0=nl.less, operand0=1.0)
    nisa.tensor_tensor(dst=own, data1=own, data2=ge, op=nl.multiply)
    return own


def _cp_page_sel(block_table, pos, np_, nb, MP, NPB):
    """Native-page id (fp32, exact below 2^24) of each row's new token: bt[b, pos >> log2 NPB], [np_, nb]."""
    bt_i = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)
    pos_rep = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bt_i, src=block_table.ap(pattern=[[nb * MP, np_], [MP, nb], [1, MP]], offset=0))
    nisa.dma_copy(dst=pos_rep, src=pos.ap(pattern=[[nb, np_], [1, nb], [0, MP]], offset=0))
    lo = nl.ndarray((np_, nb, MP), dtype=nl.int32, buffer=nl.sbuf)  # j * NPB
    nisa.iota(lo, pattern=[[0, nb], [NPB, MP]], offset=0, channel_multiplier=0)
    dif = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dif, data1=pos_rep, data2=lo, op=nl.subtract)
    ge = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ge, data=dif, op0=nl.greater_equal, operand0=0.0)
    oh = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh, data=dif, op0=nl.less, operand0=float(NPB))
    nisa.tensor_tensor(dst=oh, data1=oh, data2=ge, op=nl.multiply)
    btf = nl.ndarray((np_, nb, MP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=btf, src=bt_i)
    nisa.tensor_tensor(dst=oh, data1=oh, data2=btf, op=nl.multiply)
    pgf = nl.ndarray((np_, nb), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(nb):
        nisa.tensor_reduce(dst=pgf[:, i : i + 1], op=nl.add, data=oh[:, i, :], axis=1)
    return pgf


@nki.jit
def attn_cp_front(
    x,  # [B, H] bf16 (post input-RMSNorm), the same gathered rows on every CP rank
    w_qkv,  # [H, (nq + 2*nkv) * d] bf16: this rank's q heads | k heads | v heads (natural channel order)
    q_norm_w,  # [1, d] fp32 raw zero-centred weight (kernel uses 1 + w)
    k_norm_w,  # [1, d] fp32
    cos,  # [B, ROT/2] fp32
    sin,  # [B, ROT/2] fp32
    nq: int = 2,
    nkv: int = 2,
):
    """DCP front: QKV projection + zero-centred Q/K RMSNorm + partial RoPE, attn_decode_tkg's front with
    NH = nq + nkv normed heads. Returns (q_loc [B, nq*d] bf16, k_new [B, nkv, d], v_new [B, nkv, d]); the
    d^-0.5 score scale is the core's (it applies it where it builds the stationary, the single-rank order)."""
    B, H = x.shape
    d = q_norm_w.shape[1]
    H1 = H // P
    NQKV = (nq + 2 * nkv) * d
    MCH = 512
    half = ROT // 2
    kernel_assert(d % P == 0 and ROT <= d, f"need d%128==0 and ROT<=d (d={d})")
    kernel_assert(tuple(w_qkv.shape) == (H, NQKV), f"w_qkv must be [H, {NQKV}], got {tuple(w_qkv.shape)}")

    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=x_sb, src=x)
    xT = nl.ndarray((P, H1, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)

    qkv = nl.ndarray((P, NQKV), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, NQKV, MCH):
        cn = min(MCH, NQKV - c0)
        wq = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq[:, :, 0:cn], src=w_qkv.ap(pattern=[[NQKV, P], [P * NQKV, H1], [1, cn]], offset=c0))
        ps = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for h1 in range(H1):
            nisa.nc_matmul(ps[:, 0:cn], xT[:, h1, :], wq[:, h1, 0:cn])
        nisa.tensor_copy(dst=qkv[:, c0 : c0 + cn], src=ps[:, 0:cn])

    NH = nq + nkv  # q heads + the k heads, contiguous in qkv
    wn = nl.ndarray((P, NH, d), dtype=nl.float32, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        src = q_norm_w if hh < nq else k_norm_w
        nisa.dma_copy(dst=wn[:, hh, :], src=src.ap(pattern=[[0, B], [1, d]], offset=0))
    nisa.tensor_scalar(dst=wn, data=wn, op0=nl.add, operand0=1.0)
    ss = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        sq = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.sbuf)[:B]
        nisa.activation_reduce(
            dst=sq, op=nl.square, data=qkv[:, hh * d : (hh + 1) * d], reduce_op=nl.add, reduce_res=ss[:, hh : hh + 1]
        )
    rs = nl.ndarray((P, NH), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / d, bias=1e-6)
    qk = nl.ndarray((P, NH, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for hh in range(NH):
        nisa.scalar_tensor_tensor(
            dst=qk[:, hh, :], data=qkv[:, hh * d : (hh + 1) * d], op0=nl.multiply, operand0=rs[:, hh : hh + 1],
            op1=nl.multiply, operand1=wn[:, hh, :],
        )

    cs = nl.ndarray((P, 2, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=cs[:, 0, :, :], src=cos.ap(pattern=[[half, B], [0, NH], [1, half]], offset=0))
    nisa.dma_copy(dst=cs[:, 1, :, :], src=sin.ap(pattern=[[half, B], [0, NH], [1, half]], offset=0))
    x1 = qk[:, :, 0:half]
    x2 = qk[:, :, half:ROT]
    t1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    t2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    r1 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    r2 = nl.ndarray((P, NH, half), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=t1, data1=x1, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x2, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r1, data1=t1, data2=t2, op=nl.subtract)  # x1 cos - x2 sin
    nisa.tensor_tensor(dst=t1, data1=x2, data2=cs[:, 0, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=x1, data2=cs[:, 1, :, :], op=nl.multiply)
    nisa.tensor_tensor(dst=r2, data1=t1, data2=t2, op=nl.add)  # x2 cos + x1 sin
    nisa.tensor_copy(dst=qk[:, :, 0:half], src=r1)
    nisa.tensor_copy(dst=qk[:, :, half:ROT], src=r2)

    q_loc = nl.ndarray((B, nq * d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    k_new = nl.ndarray((B, nkv, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    v_new = nl.ndarray((B, nkv, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for h in range(nq):
        nisa.dma_copy(dst=q_loc[:, h * d : (h + 1) * d], src=qk[:, h, :])
    for h in range(nkv):
        nisa.dma_copy(dst=k_new[:, h, :], src=qk[:, nq + h, :])
        nisa.dma_copy(dst=v_new[:, h, :], src=qkv[:, (nq + nkv) * d + h * d : (nq + nkv + 1 + h) * d])
    return q_loc, k_new, v_new


def _cp_probe_store(probe, passc, tail, seg_dmas, blocks=None):
    """Probe harness output: probe [1, 4] fp32 <- [span blocks run, tail blocks, K gather
    DMAs per full block, V gather DMAs per full block]. ``passc`` is the block loop's [1, 1] fp32 trip
    counter (None on the unrolled path, whose block count is the trace-time ``blocks``)."""
    pv = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(pv, 0.0)
    if passc is not None:
        nisa.tensor_copy(dst=pv[0:1, 0:1], src=passc)
    else:
        nisa.tensor_scalar(dst=pv[0:1, 0:1], data=pv[0:1, 0:1], op0=nl.add, operand0=blocks)
    nisa.tensor_scalar(dst=pv[0:1, 1:2], data=pv[0:1, 1:2], op0=nl.add, operand0=tail)
    nisa.tensor_scalar(dst=pv[0:1, 2:3], data=pv[0:1, 2:3], op0=nl.add, operand0=seg_dmas)
    nisa.tensor_scalar(dst=pv[0:1, 3:4], data=pv[0:1, 3:4], op0=nl.add, operand0=seg_dmas)
    nisa.dma_copy(dst=probe, src=pv)


@nki.jit
def attn_cp_core(
    q_all,  # [B, G*nq*d] bf16: every CP rank's queries, token-major (rank g's nq heads at g*nq*d ..)
    k_new,  # [B, nkv, d] bf16 this rank's new-token K (normed + RoPE; the same values on every rank)
    v_new,  # [B, nkv, d] bf16
    k_cache_t,  # [NP, nkv, CPB, d] bf16: this rank's CPB-token block of every native page, token-major like
    # v_cache (the PD scatter's K block of a page is one contiguous run per layer on both roles)
    v_cache,  # [NP, nkv, CPB, d] bf16
    pos,  # [B, 1] int32 new-token position (global, the same on every rank)
    cp_rank,  # [1, 1] int32 this rank's position in the CP group
    cp_sink,  # [1, 1] int32 pool row the non-owners' new-token writes drop into (never read)
    block_table,  # [B, MP] int32 native-page ids (the same on every rank; local block lb -> column lb)
    pool_off: int = 0,  # this layer's first pool row (0 when block_table is pre-offset, e.g. multistep's)
    cp_size: int = KV_CP,
    nq: int = 2,
    nkv: int = 2,
    probe=None,  # probe harnesses only: [1, 4] fp32 HBM out — [span blocks run (the loop's
    # trip count), tail blocks, K gather DMAs per full block, V gather DMAs per full block]. Serving never
    # passes it; None adds nothing to the trace.
):
    """DCP partial attention: the online split-K over this rank's share of the history, per chunk of nq
    query heads (chunk c = rank c's heads on KV head c*nkv//G), plus the owner-only new-token write.
    Returns (o_hat [G*nq*B, d] bf16 unnormalized acc/L rows gh*B + b, ms [2, G*nq*B] fp32 (M row, L row),
    k_cache_t, v_cache). The collectives and the merge live in the layer (rank-as-data keeps one NEFF)."""
    G = cp_size
    B = q_all.shape[0]
    NP, nkv_p, CPB, d = k_cache_t.shape
    MP = block_table.shape[1]
    S_loc = MP * CPB
    NPB = G * CPB  # tokens per native page (the physical block size at kv_cp=1)
    DH = d // P
    R = B * nq
    Rtot = G * R
    MCH = 512
    kernel_assert(nkv_p == nkv and G % nkv == 0, f"pool heads {nkv_p} != {nkv} or G % nkv != 0 (G={G})")
    kernel_assert(tuple(q_all.shape) == (B, G * nq * d), f"q_all must be [B, {G * nq * d}], got {tuple(q_all.shape)}")
    kernel_assert(tuple(v_cache.shape) == (NP, nkv, CPB, d), f"v_cache must be [NP, {nkv}, CPB, d]")
    kernel_assert(tuple(k_cache_t.shape[1:]) == (nkv, CPB, d),
                  f"k_cache_t must be [NP, {nkv}, CPB, d] (token-major), got {tuple(k_cache_t.shape)}")
    kernel_assert(CPB % P == 0 and CPB & (CPB - 1) == 0, f"CPB must be a power of two >= 128, got {CPB}")
    kernel_assert(R <= P and S_loc % P == 0, f"need B*nq<=128 and S_loc%128==0 (R={R}, S_loc={S_loc})")
    kernel_assert(NP < (1 << 24), f"pool rows must stay below 2^24 (exact fp32 lookup), NP = {NP}")
    kernel_assert(NP * nkv * d * CPB <= (1 << 31), f"CP K write needs NP*nkv*d*CPB <= 2^31")

    pos_rr = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    for b in range(B):
        nisa.dma_copy(dst=pos_rr[b * nq : (b + 1) * nq, :], src=pos.ap(pattern=[[0, nq], [1, 1]], offset=b))
    rkR = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.dma_copy(dst=rkR, src=cp_rank.ap(pattern=[[0, R], [1, 1]], offset=0))
    # u = pos - rank*CPB: local column g of local block jp holds the global position jp*NPB + rank*CPB +
    # (g % CPB), valid iff g + jp*(NPB - CPB) < u (exact fp32 / int32 below 2^24)
    u_rf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=u_rf, src=pos_rr)
    rkcf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=rkcf, src=rkR)
    nisa.tensor_scalar(dst=rkcf, data=rkcf, op0=nl.multiply, operand0=float(CPB))
    nisa.tensor_tensor(dst=u_rf, data1=u_rf, data2=rkcf, op=nl.subtract)
    ownR = _cp_own(pos_rr, rkR, NPB, CPB)  # [R, 1] fp32 0/1: this rank writes row b's new token
    # row r = b * nq + h: mask[r, b'] = (0 <= r - b' * nq < nq), sel[r, h'] = sum_b' (r - b' * nq == h')
    rb_i = nl.ndarray((P, B), dtype=nl.int32, buffer=nl.sbuf)[:R]
    nisa.iota(rb_i, pattern=[[-nq, B]], offset=0, channel_multiplier=1)
    rb = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_copy(dst=rb, src=rb_i)
    ge = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=ge, data=rb, op0=nl.greater_equal, operand0=0.0)
    mask = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
    nisa.tensor_scalar(dst=mask, data=rb, op0=nl.less, operand0=float(nq))
    nisa.tensor_tensor(dst=mask, data1=mask, data2=ge, op=nl.multiply)
    sel = nl.ndarray((P, nq), dtype=nl.float32, buffer=nl.sbuf)[:R]
    for h in range(nq):
        eq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=eq, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.greater_equal, operand1=0.0)
        lt = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=lt, data=rb, op0=nl.subtract, operand0=float(h), op1=nl.less, operand1=1.0)
        nisa.tensor_tensor(dst=eq, data1=eq, data2=lt, op=nl.multiply)
        nisa.tensor_reduce(dst=sel[:, h : h + 1], op=nl.add, data=eq, axis=1)

    # per-chunk stationary Qz (d^-0.5 folded in, the single-rank rounding order) + new-token K^T per KV head
    scale = float(d) ** -0.5
    Qz = []
    for c in range(G):
        q_sb = nl.ndarray((P, nq * d), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        nisa.dma_copy(dst=q_sb, src=q_all[:, c * nq * d : (c + 1) * nq * d])
        qT = nl.ndarray((P, DH, B, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(nq):
            for j in range(DH):
                tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=q_sb[:, h * d + j * P : h * d + (j + 1) * P])
                nisa.tensor_scalar(dst=qT[:, j, :, h], data=tp, op0=nl.multiply, operand0=scale)
        Qz_c = nl.ndarray((P, B, DH, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(Qz_c, 0.0)
        for b in range(B):
            nisa.tensor_copy(dst=Qz_c[:, b, :, b * nq : (b + 1) * nq], src=qT[:, :, b, :])
        Qz.append(Qz_c)
    kT_new = []
    v_row = []
    v_hbm = []
    v_sb = []
    for h in range(nkv):
        kn_sb = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        nisa.dma_copy(dst=kn_sb, src=k_new[:, h, :])
        kT_h = nl.ndarray((P, DH, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(DH):
            tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=kn_sb[:, j * P : (j + 1) * P])
            nisa.tensor_copy(dst=kT_h[:, j, :], src=tp)
        kT_new.append(kT_h)
        # SBUF copy for the owner-only write (a Vector DGE's non-indirect side must be SB)
        vv_sb = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        nisa.dma_copy(dst=vv_sb, src=v_new[:, h, :])
        v_sb.append(vv_sb)
        v_h = nl.ndarray((B, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
        nisa.dma_copy(dst=v_h, src=v_new[:, h, :])
        v_hbm.append(v_h)
        vr = nl.ndarray((1, B, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=vr.reshape((1, B * d)), src=v_h.reshape((1, B * d)))
        v_row.append(vr)

    # the new token's score seeds (M, L): the owner seeds (s_new, 1), a non-owner (-NEG, 1) — its share of
    # every page is empty of this token and the merge's s * e^(m - M*) must be exactly 0
    M = []
    Lsum = []
    acc = []
    for c in range(G):
        kvh = c * nkv // G
        ps_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)[:R]
        for b in range(B):
            for j in range(DH):
                nisa.nc_matmul(ps_n, Qz[c][:, b, j, :], kT_new[kvh][:, j, b : b + 1])
        s_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_copy(dst=s_new, src=ps_n)
        M_c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=M_c, data1=ownR, data2=s_new, op=nl.multiply)
        nseed = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nseed, data=ownR, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)
        nisa.tensor_tensor(dst=M_c, data1=M_c, data2=nseed, op=nl.add)
        L_c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.memset(L_c, 1.0)
        acc_c = nl.ndarray((nq, B, d), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc_c, 0.0)
        M.append((M_c, s_new))
        Lsum.append(L_c)
        acc.append(acc_c)

    # ---- the local span: SCH-column blocks, K/V of each block read once and shared by all G chunks ----
    VCH = SCH // P
    NPBb = SCH // CPB  # local blocks (table columns) per span block

    def _span_block(sw, urel, pgtile):
        """One span block, local column base 0: the mask compares the block-local iota with
        urel = u - s0 * G — local column s0 + g sits exactly s0 * G native positions ahead of g (s0 a
        multiple of CPB), so the booleans, and every score downstream, are the absolute form's bit for
        bit. pgtile(jp, b) is the [1, 1] int32 SBUF tile with this block's jp-th local block of row b."""
        # CP mask: local column g valid iff g + (g >> log2 CPB) * (NPB - CPB) < urel (see above)
        ig = nl.ndarray((P, SCH), dtype=nl.int32, buffer=nl.sbuf)[:R]
        nisa.iota(ig, pattern=[[1, SCH]], offset=0, channel_multiplier=0)  # columns >= sw unused
        jpm = nl.ndarray((P, SCH), dtype=nl.int32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=jpm[:, 0:sw], data=ig[:, 0:sw], op0=nl.right_shift, operand0=_log2(CPB))
        nisa.tensor_scalar(dst=jpm[:, 0:sw], data=jpm[:, 0:sw], op0=nl.multiply, operand0=NPB - CPB)
        nisa.tensor_tensor(dst=jpm[:, 0:sw], data1=jpm[:, 0:sw], data2=ig[:, 0:sw], op=nl.add)
        bias = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=bias[:, 0:sw], data=jpm[:, 0:sw], op0=nl.less, operand0=urel)
        nisa.tensor_scalar(dst=bias[:, 0:sw], data=bias[:, 0:sw], op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=_CP_NEG)
        # scores: this block's K tiles once per (row, KV head), every chunk's matmuls against them. The
        # psum holds one chunk's [R, SCH] pass (PSUM fits one); each chunk's scores accumulate across rows
        # in its SBUF tile (the single-rank kernel accumulates across b in the psum itself — G live
        # [R, SCH] psums do not fit).
        psc = [nl.ndarray((R, SCH), dtype=nl.float32, buffer=nl.sbuf) for _ in range(G)]  # G * R*SCH fp32 (compact)
        for c in range(G):
            nisa.memset(psc[c], 0.0)
        tmp_s = nl.ndarray((R, SCH), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(B):
            kT = []
            # stage the span's whole [sw, d] run for BOTH heads up front: a page's nkv heads are one
            # contiguous [nkv, CPB, d] run in the token-major pool, so a single per-page DMA covers both
            # heads (halving the gather's per-DMA fixed cost; V's contiguous burst form — a strided
            # [[1, P], [P, DH], [d, w]] read steps 2 bytes at a time and the DMA engine cannot burst:
            # measured much slower on silicon). Issuing every seg DMA before the transposes lets the DMA
            # queue overlap the transpose engine instead of serializing per seg.
            kW2 = nl.ndarray((P, (sw // CPB) * nkv, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            for jp, o, w, dc in _page_segs(0, sw, CPB):
                nisa.dma_copy(
                    dst=kW2[:, (dc // CPB) * nkv : (dc // CPB) * nkv + nkv, :],
                    src=k_cache_t.ap(pattern=[[d, P], [P * d, nkv], [1, d]], offset=o * d,
                                     scalar_offset=pgtile(jp, b),
                                     indirect_dim=0),
                    dge_mode=nisa.dge_mode.hwdge,
                )
            # kT_h[dim j, dim chunk jj, token t] = kW2[token t, chunk cc * nkv + h, dim jj*P + j]; the
            # psum tiles alternate so a copy never stalls the next chunk's transpose
            tq = (nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum),
                  nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum))
            for h in range(nkv):
                kT_h = nl.ndarray((P, DH, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)
                for cc in range(sw // P):
                    for jj in range(DH):
                        nisa.nc_transpose(dst=tq[cc % 2], data=kW2[:, cc * nkv + h, jj * P : (jj + 1) * P])
                        nisa.tensor_copy(dst=kT_h[:, jj, cc * P : (cc + 1) * P], src=tq[cc % 2])
                kT.append(kT_h)
            for c in range(G):
                kvh = c * nkv // G
                ps_s = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.psum)[:R]
                for j in range(DH):
                    for m0 in range(0, sw, MCH):
                        cn = min(MCH, sw - m0)
                        nisa.nc_matmul(ps_s[:, m0 : m0 + cn], Qz[c][:, b, j, :], kT[kvh][:, j, m0 : m0 + cn])
                nisa.tensor_copy(dst=tmp_s[:, 0:sw], src=ps_s[:, 0:sw])  # rows outside b's are exactly 0
                nisa.tensor_tensor(dst=psc[c][:, 0:sw], data1=psc[c][:, 0:sw], data2=tmp_s[:, 0:sw], op=nl.add)
        # online update + P^T per chunk (P^T must be complete before this block's V pass)
        PT = []
        corr_h = []
        for c in range(G):
            sc = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_tensor(dst=sc[:, 0:sw], data1=psc[c][:, 0:sw], data2=bias[:, 0:sw], op=nl.add)
            mb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_reduce(dst=mb, op=nl.max, data=sc[:, 0:sw], axis=1)
            Mn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_tensor(dst=Mn, data1=M[c][0], data2=mb, op=nl.maximum)
            nMn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_scalar(dst=nMn, data=Mn, op0=nl.multiply, operand0=-1.0)
            corr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.activation(dst=corr, op=nl.exp, data=M[c][0], bias=nMn)  # exp(M_old - M_new)
            ex = nl.ndarray((P, SCH), dtype=nl.float32, buffer=nl.sbuf)[:R]
            sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.activation_reduce(dst=ex[:, 0:sw], op=nl.exp, data=sc[:, 0:sw], reduce_op=nl.add,
                                   reduce_res=sb, bias=nMn)
            nisa.scalar_tensor_tensor(dst=Lsum[c], data=Lsum[c], op0=nl.multiply, operand0=corr,
                                      op1=nl.add, operand1=sb)
            nisa.tensor_copy(dst=M[c][0], src=Mn)
            pb = nl.ndarray((P, SCH), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
            nisa.tensor_copy(dst=pb[:, 0:sw], src=ex[:, 0:sw])
            NCb = sw // P
            PT_c = nl.ndarray((P, NCb, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            for cc in range(NCb):
                tp = nl.ndarray((P, R), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=pb[:, cc * P : (cc + 1) * P])
                nisa.tensor_copy(dst=PT_c[:, cc, :], src=tp)
            PT.append(PT_c)
            corr_h.append(_row_to_heads(sel, mask, corr, R, nq, B))
        # P.V: this block's V tiles once per (row, KV head), every chunk's matmuls against them. Each
        # chunk's [nq, d] psum accumulates across the whole block (the single-rank form's per-row psum),
        # then acc rescales by corr ONCE per (row, block).
        for b in range(B):
            ps_o = [nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum) for _ in range(G)]
            for v0 in range(0, sw // P, VCH):
                cn = min(VCH, sw // P - v0)
                # both heads staged by one DMA per page (same head-combined form as the K gather)
                vb2 = nl.ndarray((P, cn * nkv, d), dtype=nl.bfloat16, buffer=nl.sbuf)
                for jp, o, w, dc in _page_segs(v0 * P, cn * P, CPB):
                    nisa.dma_copy(
                        dst=vb2[:, (dc // CPB) * nkv : (dc // CPB) * nkv + nkv, :],
                        src=v_cache.ap(pattern=[[d, P], [P * d, nkv], [1, d]], offset=o * d,
                                       scalar_offset=pgtile(jp, b),
                                       indirect_dim=0),
                        dge_mode=nisa.dge_mode.hwdge,
                    )
                for c in range(G):
                    kvh = c * nkv // G
                    for cc in range(cn):
                        nisa.nc_matmul(ps_o[c], PT[c][:, v0 + cc, b * nq : (b + 1) * nq],
                                       vb2[:, cc * nkv + kvh, :])
            for c in range(G):
                nisa.scalar_tensor_tensor(dst=acc[c][:, b, :], data=acc[c][:, b, :], op0=nl.multiply,
                                          operand0=corr_h[c][:, b : b + 1], op1=nl.add, operand1=ps_o[c])

    if _BLOCK_LOOP and S_loc > SCH:
        # run-time block loop (patch 0007's form): each unrolled block repeats the G x B chunk/row
        # fan-out (~200k instructions at B=16), so the compile would grow with the page count even
        # though a rank's token share stays small. Whole SCH blocks loop (one body in the program); a
        # S_loc % SCH tail runs once on the carried base. The pass's table columns come from one
        # dynamic-offset DMA of the RAW table (a kernel input — dynamic-offset DMAs from an HBM
        # intermediate trip the allocator inside a device loop, NCC_IBIR853), + pool_off in-kernel.
        NB = S_loc // SCH
        sw_t = S_loc - NB * SCH
        bt_flat = block_table.reshape((B * MP, 1))
        u_rel = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]  # u - s0 * G, carried
        nisa.tensor_copy(dst=u_rel, src=u_rf)
        pg0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # the pass's first table column
        nisa.memset(pg0, 0)
        passc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)  # passes done (loop condition)
        nisa.memset(passc, 0.0)
        # pos-derived runtime trip bound: pass k's columns are valid for row b only while
        # k * SCH * G < u_b (the CP mask's urel = u - s0 * G, s0 = k * SCH), and an all-masked pass is an
        # exact no-op (every score biased to -NEG -> exp 0, corr 1; M, L, acc unchanged), so running only
        # ceil(max_b u_b / (SCH * G)) passes is bit-identical to running NB. Rows at pos 0 (padding) have
        # u <= 0 and never extend the bound: the trip count follows the cohort's longest live row, not
        # the block-table width. u_rf's rows duplicate per head; the max over rows is the max over B.
        umax_p = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=umax_p[:1, :], data=u_rf)
        nb_rt = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:1]
        nisa.tensor_reduce(dst=nb_rt, op=nl.max, data=umax_p[:1, :], axis=1)
        nisa.tensor_scalar(dst=nb_rt, data=nb_rt, op0=nl.add, operand0=float(SCH * G - 1))
        nisa.tensor_scalar(dst=nb_rt, data=nb_rt, op0=nl.multiply, operand0=1.0 / float(SCH * G))
        # fp32 -> int32 tensor_copy ROUNDS to nearest (not trunc): bias down so round == floor. 0.49999
        # clears every residue class: y integer (u == 1 mod SCH*G) -> k - 0.49999 rounds to k; max
        # fraction 32767/32768 -> k + 0.49998 rounds to k; both >= 2e-5 from the round-half ties while
        # the fp32 subtract error is ~5e-7. (Measured on trn3: un-biased, u=220000 -> 7.714 rounds to 8,
        # one wasted pass.)
        nisa.tensor_scalar(dst=nb_rt, data=nb_rt, op0=nl.subtract, operand0=0.49999)
        nb_ri = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nb_ri, src=nb_rt)  # round-to-nearest == floor after the bias
        nisa.tensor_copy(dst=nb_rt, src=nb_ri)
        nisa.tensor_scalar(dst=nb_rt, data=nb_rt, op0=nl.maximum, operand0=1.0)
        nisa.tensor_scalar(dst=nb_rt, data=nb_rt, op0=nl.minimum, operand0=float(NB))

        def cond_reg():
            more = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=more, data1=passc, data2=nb_rt, op=nl.less)
            more_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=more_i, src=more)
            r = nisa.register_alloc()
            nisa.register_load(r, more_i)
            return r

        def _bt_pass(cols):
            # bt_p[c, b] = block_table[b, pg0 + c] (+ pool_off): this pass's local-block pool rows
            bt_p = nl.ndarray((cols, B), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=bt_p,
                src=bt_flat.ap(pattern=[[1, cols], [MP, B]], offset=0, scalar_offset=pg0, indirect_dim=0),
            )
            if pool_off:  # raw shared table: this layer's pool pages start at pool_off (int32 exact)
                nisa.tensor_scalar(dst=bt_p, data=bt_p, op0=nl.add, operand0=pool_off)
            return bt_p

        def block_pass(_):
            bt_p = _bt_pass(NPBb)
            _span_block(SCH, u_rel, lambda jp, b: bt_p[jp : jp + 1, b : b + 1])
            nisa.tensor_scalar(dst=u_rel, data=u_rel, op0=nl.subtract, operand0=float(SCH * G))
            nisa.tensor_scalar(dst=pg0, data=pg0, op0=nl.add, operand0=NPBb)
            nisa.tensor_scalar(dst=passc, data=passc, op0=nl.add, operand0=1.0)
            return cond_reg()

        nl.while_loop(cond_reg(), block_pass)
        if sw_t:  # the partial tail block, at most once, on the carried base (reads exactly its own
            # columns): needed only when the pos-derived bound reaches the last whole block (a shorter
            # cohort leaves the tail's columns all-masked, and an all-masked block is an exact no-op)
            go = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=go, data=nb_rt, op0=nl.greater_equal, operand0=float(NB))
            go_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=go_i, src=go)
            done = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(done, 0)

            def tail_pass(_):
                bt_p = _bt_pass(sw_t // CPB)
                _span_block(sw_t, u_rel, lambda jp, b: bt_p[jp : jp + 1, b : b + 1])
                r = nisa.register_alloc()
                nisa.register_load(r, done)
                return r

            tr = nisa.register_alloc()
            nisa.register_load(tr, go_i)
            nl.while_loop(tr, tail_pass)
        if probe is not None:
            _cp_probe_store(probe, passc, float(sw_t != 0), float(B * (SCH // CPB)))
    else:
        # short spans: trace-unrolled blocks over the staged whole-row table (absolute block base)
        bt_row = nl.ndarray((1, B * MP), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bt_row, src=block_table.reshape((1, B * MP)))
        for s0 in range(0, S_loc, SCH):
            sw = min(SCH, S_loc - s0)
            urel = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
            nisa.tensor_scalar(dst=urel, data=u_rf, op0=nl.subtract, operand0=float(s0 * G))
            _span_block(sw, urel,
                        lambda jp, b: bt_row[0:1, b * MP + s0 // CPB + jp : b * MP + s0 // CPB + jp + 1])
        if probe is not None:
            _cp_probe_store(probe, None, float(S_loc % SCH != 0), float(B * (SCH // CPB)),
                            blocks=float(-(-S_loc // SCH)))

    # ---- epilogue: + the owner's new-token term, / L (unnormalized (M, L, o_hat) out for the merge) ----
    o_hat = nl.ndarray((Rtot, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ms = nl.ndarray((2, Rtot), dtype=nl.float32, buffer=nl.private_hbm)
    for c in range(G):
        kvh = c * nkv // G
        M_c, s_new = M[c]
        nM = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nM, data=M_c, op0=nl.multiply, operand0=-1.0)
        # mask s_new BEFORE the exp: an idle row's M is -NEG, so exp(s_new - M) would overflow (inf * 0 =
        # NaN); with the input masked the exp underflows to exactly 0 instead
        snm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=snm, data1=s_new, data2=ownR, op=nl.multiply)
        nm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_scalar(dst=nm, data=ownR, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=_CP_NEG)
        nisa.tensor_tensor(dst=snm, data1=snm, data2=nm, op=nl.add)
        en = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.activation(dst=en, op=nl.exp, data=snm, bias=nM)
        pnf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.tensor_tensor(dst=pnf, data1=en, data2=ownR, op=nl.multiply)  # non-owners add nothing
        pn = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)[:R]
        nisa.tensor_copy(dst=pn, src=pnf)
        rsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:R]
        nisa.reciprocal(dst=rsum, data=Lsum[c])
        rs_h = _row_to_heads(sel, mask, rsum, R, nq, B)
        PTn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpn = nl.ndarray((1, R), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpn, data=pn)
        nisa.tensor_copy(dst=PTn, src=tpn)
        for b in range(B):
            ps_v = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps_v, PTn[:, b * nq : (b + 1) * nq], v_row[kvh][:, b, :])
            o32 = nl.ndarray((nq, d), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o32, data1=acc[c][:, b, :], data2=ps_v, op=nl.add)
            o_b = nl.ndarray((nq, d), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_b, data=o32, op0=nl.multiply, operand0=rs_h[:, b : b + 1])
            # one pattern DMA per (chunk, row): a per-head slice o_b[h:h+1] is an illegal 1-partition
            # access off {0, 32, 64, 96}. The SBUF side walks partition-fastest (ordinal h + nq*t), so the
            # pattern dims are (h, t): o_b's heads land B rows apart in o_hat at (c*nq + h)*B + b
            nisa.dma_copy(dst=o_hat.ap(pattern=[[B * d, nq], [1, d]], offset=(c * nq * B + b) * d), src=o_b)
        # (M, L) rows r = b*nq + h land at columns gh*B + b of ms's two rows: stage the column in HBM, then
        # one strided gather per head (a contiguous-span pattern gets coalesced into a linear copy, and the
        # per-head 1-partition SBUF copies are illegal off a multiple of 32 partitions)
        mlh = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.private_hbm)
        for i, vals in enumerate((M_c, Lsum[c])):
            nisa.dma_copy(dst=mlh, src=vals)
            for h in range(nq):
                nisa.dma_copy(dst=ms[i : i + 1, (c * nq + h) * B : (c * nq + h + 1) * B],
                              src=mlh.ap(pattern=[[nq, B]], offset=h))

    # ---- the owner-only new-token write (non-owners drop it into the sink row) ----
    posW = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=posW, src=pos.reshape((1, B)))
    rkW = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rkW, src=cp_rank.ap(pattern=[[0, B], [1, 1]], offset=0))
    ownW = _cp_own(posW, rkW, NPB, CPB)  # [1, B] fp32
    pgW = _cp_page_sel(block_table, pos, 1, B, MP, NPB)
    if pool_off:
        nisa.tensor_scalar(dst=pgW, data=pgW, op0=nl.add, operand0=float(pool_off))
    skW_i = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=skW_i, src=cp_sink.ap(pattern=[[0, B], [1, 1]], offset=0))
    skW = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=skW, src=skW_i)
    t1 = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=ownW, data2=pgW, op=nl.multiply)
    t2 = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t2, data1=ownW, data2=skW, op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=skW, data2=t2, op=nl.subtract)
    nisa.tensor_tensor(dst=t1, data1=t1, data2=t2, op=nl.add)  # pgf = own*page + (1-own)*sink
    pgW_i = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pgW_i, src=t1)
    nisa.tensor_scalar(dst=pgW_i, data=pgW_i, op0=nl.left_shift, operand0=_log2(nkv * d * CPB))
    offW = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=offW, data=posW, op0=nl.bitwise_and, operand0=CPB - 1)
    # token-major: this rank's K block of the page is [CPB, d], so a new token's dim row lands at
    # (page base) | h*CPB*d | offW*d, and the [P, DH] dim tile's strides are (1, P) like V's write
    nisa.tensor_scalar(dst=offW, data=offW, op0=nl.left_shift, operand0=_log2(d))
    k_flat = k_cache_t.reshape((NP * nkv * CPB * d, 1))
    for h in range(nkv):
        xk = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xk, src=pgW_i)
        if h:
            nisa.tensor_scalar(dst=xk, data=xk, op0=nl.bitwise_or, operand0=h * CPB * d)
        nisa.tensor_tensor(dst=xk, data1=xk, data2=offW, op=nl.bitwise_or)
        for b in range(B):
            xr = nisa.register_alloc()
            nisa.register_load(xr, xk[0:1, b : b + 1])
            nisa.dma_copy(
                dst=k_flat.ap(pattern=[[1, P], [P, DH], [1, 1]], offset=0, scalar_offset=xr,
                              indirect_dim=0),
                src=kT_new[h].ap(pattern=[[DH * B, P], [B, DH], [1, 1]], offset=b),
            )
    posB = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=posB, src=pos)
    rkB = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=rkB, src=cp_rank.ap(pattern=[[0, B], [1, 1]], offset=0))
    ownV = _cp_own(posB, rkB, NPB, CPB)  # [B, 1] fp32
    pgV = _cp_page_sel(block_table, pos, B, 1, MP, NPB)
    if pool_off:
        nisa.tensor_scalar(dst=pgV, data=pgV, op0=nl.add, operand0=float(pool_off))
    skV_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=skV_i, src=cp_sink.ap(pattern=[[0, B], [1, 1]], offset=0))
    skV = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_copy(dst=skV, src=skV_i)
    u1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=u1, data1=ownV, data2=pgV, op=nl.multiply)
    u2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:B]
    nisa.tensor_tensor(dst=u2, data1=ownV, data2=skV, op=nl.multiply)
    nisa.tensor_tensor(dst=u2, data1=skV, data2=u2, op=nl.subtract)
    nisa.tensor_tensor(dst=u1, data1=u1, data2=u2, op=nl.add)
    v_off = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.tensor_copy(dst=v_off, src=u1)
    nisa.tensor_scalar(dst=v_off, data=v_off, op0=nl.left_shift, operand0=_log2(nkv * CPB))
    offB = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.tensor_scalar(dst=offB, data=posB, op0=nl.bitwise_and, operand0=CPB - 1)
    v_flat = v_cache.reshape((NP * nkv * CPB, d))
    for h in range(nkv):
        vo = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B]
        nisa.tensor_copy(dst=vo, src=v_off)
        if h:
            nisa.tensor_scalar(dst=vo, data=vo, op0=nl.bitwise_or, operand0=h * CPB)
        nisa.tensor_tensor(dst=vo, data1=vo, data2=offB, op=nl.bitwise_or)
        nisa.dma_copy(
            dst=v_flat.ap(pattern=[[d, B], [1, d]], offset=0, vector_offset=vo, indirect_dim=0),
            src=v_sb[h],
            dge_mode=nisa.dge_mode.swdge,
        )
    return o_hat, ms, k_cache_t, v_cache


@nki.jit
def attn_cp_combine(
    o_hat,  # [G*nq*B, d] bf16: this rank's locally normalized attention, rows gh*B + b (global head gh)
    ms_all,  # [2*G, G*nq*B] fp32: every CP rank's (M, L), rank g's at rows 2g (m) and 2g+1 (s)
    cp_rank,  # [1, 1] int32
    cp_size: int = KV_CP,
):
    """The CP merge scale (the GLM-5.2 decode's combine, in (M, L) form): per row M* = max_g m_g, w = s_r e^(m_r - M*) / sum_g s_g e^(m_g - M*);
    scaled = o_hat * w (this rank's share of the merge sum; the layer ReduceScatters the rows)."""
    G = cp_size
    R, d = o_hat.shape
    nt = (R + P - 1) // P
    # rank broadcast to every [P, G] lane (tensor_tensor has no [P, G] x [P, 1] free-dim broadcast)
    rk_i = nl.ndarray((P, G), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rk_i, src=cp_rank.ap(pattern=[[0, P], [0, G]], offset=0))
    rk = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rk, src=rk_i)
    iog = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(iog, pattern=[[1, G]], offset=0, channel_multiplier=0)
    oh = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)  # oh[p, g] = (g == rank)
    nisa.tensor_tensor(dst=oh, data1=iog, data2=rk, op=nl.subtract)
    ge = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ge, data=oh, op0=nl.greater_equal, operand0=0.0)
    nisa.tensor_scalar(dst=oh, data=oh, op0=nl.less, operand0=1.0)
    nisa.tensor_tensor(dst=oh, data1=oh, data2=ge, op=nl.multiply)
    scaled = nl.ndarray((R, d), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for c in range(nt):
        rn = min(P, R - c * P)
        m_sb = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.dma_copy(dst=m_sb, src=ms_all.ap(pattern=[[1, rn], [2 * R, G]], offset=c * P))
        s_sb = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.dma_copy(dst=s_sb, src=ms_all.ap(pattern=[[1, rn], [2 * R, G]], offset=R + c * P))
        negM = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.tensor_reduce(dst=negM, op=nl.maximum, data=m_sb, axis=1, negate=True)
        e = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.activation(dst=e, op=nl.exp, data=m_sb, bias=negM)
        num = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.tensor_tensor(dst=num, data1=e, data2=s_sb, op=nl.multiply)
        den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.tensor_reduce(dst=den, op=nl.add, data=num, axis=1)
        mine = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.tensor_tensor(dst=mine, data1=num, data2=oh[:rn, :], op=nl.multiply)
        w = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.tensor_reduce(dst=w, op=nl.add, data=mine, axis=1)
        rden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:rn]
        nisa.reciprocal(dst=rden, data=den)
        nisa.tensor_tensor(dst=w, data1=w, data2=rden, op=nl.multiply)
        o_sb = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:rn]
        nisa.dma_copy(dst=o_sb, src=o_hat[c * P : c * P + rn, :])
        o_sc = nl.ndarray((P, d), dtype=nl.bfloat16, buffer=nl.sbuf)[:rn]
        nisa.tensor_scalar(dst=o_sc, data=o_sb, op0=nl.multiply, operand0=w)
        nisa.dma_copy(dst=scaled[c * P : c * P + rn, :], src=o_sc)
    return scaled


@nki.jit
def attn_cp_tail(
    o_loc,  # [nq*B, d] bf16: the merged attention of this rank's own nq heads, rows h*B + b (head-major)
    x,  # [B, H] bf16 (post input-RMSNorm; the gate's input)
    w_gate_p,  # [128, H/128, nq*d] bf16 gate projection, rows pre-permuted
    w_o,  # [nq*d, N_out] bf16 this rank's o_proj shard
    nq: int = 2,
):
    """The gate + o_proj tail of attn_decode_tkg on the merged heads (duplicated, not shared, so the
    single-rank path stays byte-identical). Returns out [B, N_out] bf16 shared_hbm."""
    R = o_loc.shape[0]
    B = R // nq
    d = o_loc.shape[1]
    H = x.shape[1]
    N_out = w_o.shape[1]
    nd = nq * d
    H1 = H // P
    n_ot = nd // P
    MCH = 512
    # head-major rows -> token-major [B, nd]: one DMA over the cross pattern
    a_sb = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=a_sb, src=o_loc.ap(pattern=[[d, B], [B * d, nq], [1, d]], offset=0))
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=x_sb, src=x)
    xT = nl.ndarray((P, H1, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
        nisa.tensor_copy(dst=xT[:, h1, :], src=tp)
    og = nl.ndarray((P, nd), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, nd, MCH):
        cn = min(MCH, nd - c0)
        wg = nl.ndarray((P, H1, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wg[:, :, 0:cn], src=w_gate_p[:, :, c0 : c0 + cn])
        ps_g = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for h1 in range(H1):
            nisa.nc_matmul(ps_g[:, 0:cn], xT[:, h1, :], wg[:, h1, 0:cn])
        g_bf = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
        nisa.tensor_copy(dst=g_bf[:, 0:cn], src=ps_g[:, 0:cn])
        sg = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)[:B]
        nisa.activation(dst=sg[:, 0:cn], op=nl.sigmoid, data=g_bf[:, 0:cn])
        nisa.tensor_tensor(dst=og[:, c0 : c0 + cn], data1=a_sb[:, c0 : c0 + cn], data2=sg[:, 0:cn], op=nl.multiply)
    oT = nl.ndarray((P, n_ot, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(n_ot):
        tpo = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpo, data=og[:, t * P : (t + 1) * P])
        nisa.tensor_copy(dst=oT[:, t, :], src=tpo)
    out = nl.ndarray((B, N_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    o_sb = nl.ndarray((P, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)[:B]
    for c0 in range(0, N_out, MCH):
        cn = min(MCH, N_out - c0)
        wo = nl.ndarray((P, n_ot, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wo[:, :, 0:cn], src=w_o.ap(pattern=[[N_out, P], [P * N_out, n_ot], [1, cn]], offset=c0))
        ps_o = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)[:B]
        for t in range(n_ot):
            nisa.nc_matmul(ps_o[:, 0:cn], oT[:, t, :], wo[:, t, 0:cn])
        nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
    nisa.dma_copy(dst=out, src=o_sb)
    return out
