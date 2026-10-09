"""All of `layers.1.engram` on 4 ranks. Two cuts, two collectives, nothing replicated twice.

    _widen_scales          e8m0 block scales -> bfloat16 powers of two      (replicated)
    _build_table           embed.weight | embed.scale -> tbl, 264 B a row   (replicated)
    embed_body             all 24 hash columns -> embT, (6144, 8192)        (replicated)
    value_weight_body      this rank's quarter of wkv's value half -> vT    (sharded: ngram)
    value_out_body         embT[1536r : 1536r+1536] @ vT  ->  cc_src        (sharded: a PARTIAL)
    key_gemm_body          wkv[5120r : 5120r+5120] @ embT, hc copy r's      (sharded: hc copy)
                           three reductions, for every token
    gate                   sigmoid(copysign(sqrt(|dot|), dot)) -> cc_gs     (hc copy r only)
    ---- core_barrier(cc_src) ; core_barrier(cc_gs) -------------------------
    ncc.all_reduce (cc_src -> cc_dst, op=add)  the value partials,  bfloat16, 80 MiB
    ncc.all_gather (cc_gs  -> cc_gd, dim=0)    the four hc gates,   float32, 128 KB
    combine_body           out[:, c, :] = bfloat16(h[:, c, :] + gate[:, c] * value)

**The ngram cut, on the value half.** `ngram` is the contraction axis of `wkv`:
`(engram_max_ngram_size - 1) * engram_n_heads = 24` hash columns x `engram_head_dim = 256`, i.e.
6144. Rank `r` owns contraction columns `[1536r, 1536r + 1536)` of the *value* half, which are
exactly hash columns `6r .. 6r + 5`. A split of a contraction axis rejoins by summation, so that
collective is `all_reduce(op=add)`. This is the floorplan's `ngramx4(allgather)` projected onto
one device, and the cut `submodule/submodule.json` declares.

**The hc cut, on the key half -- the gate.** The gate is a sigmoid of a normalized dot product
against the *summed* key, and `sum_r(key_r)**2 != sum_r(key_r**2)`, so no rank can form it from a
quarter of the n-grams: its *contraction* is indivisible, which is why `embT`, the lookup table,
the hash ids and the q/k weights are still replicated. Its *output rows* divide exactly four
ways, because `KROWS = HC * DIM = 4 * 5120` and hc copy `c` is key rows `[5120c, 5120c + 5120)`.
So rank `r` computes hc copy `r`'s three reductions for every token and nothing else, and the
four ranks rejoin by **concatenation rather than summation**: one `all_gather` of an (8192, 1)
float32 buffer, whose dim-0 concatenation order puts hc copy `c` at rows `[8192c, 8192c + 8192)`,
which is the layout `combine_body` wants with no shuffling. 128 KB out and 512 KB back, 0.16% of
the value reduction's payload, for a cut that divides the key matmul, its weight dequantization,
its `embT` re-reads **and** its read of the 320 MiB residual stream. Measured: **21.2665 ->
10.5639 ms**, and `key_gemm_body` was 69.3% of that first number.

The five 1024-row blocks of a copy are summed in increasing block order, which is the order the
replicated version summed them in, so **every float32 add has the same two operands it had
before** and the gate is bit-identical rather than merely close. That is not a nicety here: the
compare's `max_abs` is a hard elementwise ceiling at 0.0171875 and this module already sits at
0.015625, one bfloat16 ulp below it. A bfloat16 gate -- a narrower seam, or riding the three
reductions on the value `all_reduce` as 12 more columns -- does not fit in that budget at any
speed, so the gate's seam is float32 and separate.

Within a rank the two physical cores split the key loop by *token*, not by key row, so each core
owns in the key GEMM exactly the token tiles its gate and its combine already owned. That removed
the `(20480, 192)` float32 HBM scratch the replicated version reduced through, its
`core_barrier`, and the 15.4 KB/lane of SBUF its readback held.

What crosses the wire is `value`, not `gate * value`. The value half enters the output
linearly, so `sum_r (gate * value_r) == gate * sum_r value_r`: reducing the (8192, 5120)
`value` instead of the (8192, 4, 5120) product is a quarter of the payload. The residual `h` is
added once, after the ranks rejoin -- folding it into the partials would count it four times.

The seam is **bfloat16**, which is a measured choice and not a convenient one. The reference's
own `value` is bfloat16 (`wkv`'s output dtype) and reaches the output only through
`gate * value`, which is bounded by 1.0371 on this sample while the output itself reaches
4.8779 -- so the output's own bfloat16 ulp is 8x the reduction's and sets the error either way.
Checked on the four recorded per-rank goldens before the kernel was built: a float32 reduction,
bfloat16 partials summed in float32, and a bfloat16 ring (what a bfloat16 `all_reduce` actually
does, rounding at every hop) all give **max_abs_err 0.015625** and pass_fraction 1.0 against
`tensors/reference.bin`; cosine is 0.9999990, 0.9999990 and 0.9999976 against a 0.99998 bar.
On device the bfloat16 seam measures max_abs_err 0.015625 and cosine 0.9999983 -- identical
worst element, 8x the cosine headroom the bar needs -- and it is worth 0.96 ms of the wall,
because it halves the collective and the 160 MiB the seam reads and writes around it.

The ladder, all on the frozen validator. 24.1458 ms for the faithful assembly (one reduce last,
depth-2 combine, float32 seam); 22.2596 ms with the reduce ahead of the key GEMM and the combine
4 transfers deep; 21.2985 ms with the seam in bfloat16; **10.8531 ms** with the key half cut by
hc copy; **10.5639 ms** with the key loop's `embT` prefetch at 2 buffers instead of 3;
**9.5861 ms** with the n-gram weight and its block scales interleaved into one gathered row;
**9.2872 ms** with that interleave done by a uint8 `tensor_copy` in SBUF rather than by two
strided HBM writes. The
submodule's bound is 1.1x19.7239 = 21.6963 ms and the bootstrapped module's is 51.115 ms, so this
is 0.53x the ceiling and 4.84x the floor.

This host runs LNC=2, so each rank's logical NeuronCore is two physical cores and the kernel is
launched as `kernel[2]`. Nothing in the toolchain divides independent work across the two, so
every loop bound below is split by hand on `nl.program_id(0)`: core `p` takes half the tokens of
the embed pass, half the output rows of the value-weight pass, **half the token slabs of the key
GEMM**, and the token tiles `[32p, 32p + 32)` of the gate, the value GEMM and the combine. Both
cores reach both collectives, which are logical-NeuronCore operations and not per-core ones.

A rank's identity reaches this kernel as *data*, never as a trace-time constant: one NEFF serves
all four ranks, so every tile width and loop bound here is identical on every rank. Its values --
its quarter of `wkv.value_weight` and `wkv.value_scale` -- are cut by `inference.py` as byte
ranges of the recorded files. Its *index* arrives as `shard_offset`, a (1, 1) int32 holding
`1536 * rank`, from which this kernel derives two addresses: `voff` itself, the first of this
rank's 1536 rows of the replicated `embT`, and `koff = 5120 * rank`, the first of its 5120 key
rows. `5120 / 1536 = 10/3` is not an integer and so cannot be a DMA pattern stride on a stride-1
axis; `koff` is three instructions on a (1, 1) tile instead, int32 -> float32 -> int32, where the
`+ 0.5` makes the narrowing exact whether the datapath rounds or truncates (the product is within
0.0012 of its integer).

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with an
internal compiler error that names something else:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 and a LNC=1 NEFF leaves the
   collective's buffer unallocated on the logical core's second physical core (`NCC_ILLC059`);
2. `cc_src` / `cc_dst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`);
3. they are scratch, because a collective may not read or write an IO tensor (`NCC_INLA001`);
4. both are `nl.shared_hbm`, because a collective's operands may not mix buffer kinds;
5. the `ReplicaGroup` is built by `inference.py` and passed in, because the tracer rejects
   `range` in a traced body;
6. the inputs are built on CPU and `.to(device)`'d, because `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902` -- and that error names the broadcast, not the
   collective.

A seventh, which cost this iteration a compile: **no DMA that follows a collective in program
order may carry a runtime `scalar_offset`, whatever the offset's provenance.** The key loop reads
rows `[koff, koff + 5120)` of the replicated key weight, so with the value reduce still ahead of
it -- the placement iter-0 shipped, and the only one that could overlap the ring with compute --
`neuronx-cc` aborts with `[INTERNAL_ERROR] [NCC_ISCH719] topological order violations`. That is
`8df4ee2cf87b` reproducing and correcting itself: the row describes the offset as
`ncc.rank_id()`-derived and `ASSEMBLY.md` hypothesised that an input-carried `(1, 1)` int32 would
avoid it. It does not -- `koff` never touches `ncc.rank_id()`. Hence both collectives at the end.
Giving up the early placement cost nothing measurable, because the reduce was **100% exposed
there too**: 0.645 ms with every compute engine idle underneath it even with ~12 ms of
independent key-half work behind it in program order, which is what `08ed0fdc5e2e` and
`44a3a1661946` predict and what iter-0's own profile measured.

Everything between `_e8m0_to_bf16` and the end of `value_out_body` is still the optimized
single-rank kernel at its best accepted commit (`4844073`), unchanged but for what the assembly
needs. The key half is no longer that kernel's: it computes a quarter of the rows, split across
the cores by token rather than by key row, and reduces in SBUF instead of through HBM.

Where the 10.56 ms this started from went, and what the profile said about it -- the shape still
holds at 9.29 ms, with `embed_body` roughly 2 ms lighter. **No engine is
saturated** -- Tensor 48.7%, Vector 41.8%, Sync 37.1%, GpSimd 34.5%, Scalar 12.8%, HBM at
181 GB/s and `mbu_estimated_percent` 25.1% -- while `tensor_engine_instruction_time` is 20.09 ms
against 5.31 ms active, so Tensor instructions spend three quarters of their issued time stalled.
This kernel is latency- and descriptor-bound, not engine-bound. Two things follow:

1. `embed_body` is the largest replicated term left and GpSimd's 3.765 ms is essentially all of
   it -- GpSimd issues every software-DGE descriptor here and does nothing else. It cannot be
   batched: `validate_vector_offset` states the hardware contract as
   `addr = vec_off[p] * coef + f`, one offset **per partition** with `indirect_dim` pinned to 0,
   so 8192 tokens x 24 hash columns is already at its 1,536-DMA floor. The unexplored door is
   `nisa.dma_transpose`, whose `validate_dma_gather_transpose_swdge` *does* take a 2-D index -- and
   this pass transposes every gathered tile anyway.

   **That floor is per *row gathered*, and the row count was the thing to cut.** One offset a
   partition does cap a gather at 128 rows, but `embed_body` was gathering every n-gram row
   *twice* -- 256 B of fp8 from `embed.weight` and 8 B of e8m0 from `embed.scale`, two HBM
   tensors, one id -- so it was issuing 393,216 indirect descriptors a rank where the lookup only
   needs 196,608. `_build_table` concatenates the two rows into one 264-byte row in `private_hbm`
   once, using strided DMAs only, and `embed_body` issues a single gather a column. `vector_offset`
   needs nothing new: `coef` is the pattern's dim-0 stride, so moving the pitch 256 -> 264 is the
   whole change and the raw n-gram ids keep working. SBUF is *exactly* unchanged -- one
   `(PT, NCOL, 264)` int16 tile is 12,672 B/lane where `q` + `s` was 12,288 + 384.
   **10.5021 -> 9.5861 ms**, then **9.2872 ms** once the interleave moved from two strided HBM
   writes (1024 descriptors a group) to a uint8 `nisa.tensor_copy` in SBUF followed by one fully
   contiguous write-back (128 a group). The arithmetic is untouched -- same bytes, same order,
   through a different access pattern -- so `max_abs_err` 0.015625 and `cosine` 0.9999983 come
   back identical to the digit.

   Three things priced here and **not** taken, each one run:

   * padding the row to 288 B so every gather starts 32-byte aligned, reading the same 264:
     **10.3417 ms, +1.05 ms**, more than the whole interleave bought. Whatever this path rewards
     it is not start alignment. Make the gathered length equal the row pitch.
   * merging the key loop's 48 per-block `embT` reads into one 3-D-pattern DMA a slab -- same
     bytes, same 512-byte descriptors, 48x fewer instructions and semaphores: **9.4508 ms,
     +1.7%**. Those 48 instructions are buying DMA-engine *concurrency* (the profile's 9.14 of 32),
     not wasting overhead.
   * and the inverse, splitting the embed store from 1 instruction into 12: **9.3864 ms, +1.1%**.
     Both groupings are already at a local optimum; do not spend another run on either.
2. `KSLAB` is closed, not open. `ASSEMBLY.md` called widening it "the largest single dial"; the
   key loop's live set is `wsb` 98,304 + `esb`x2 49,152 + ~18,000 B/lane, so two 512-wide `esb`
   buffers would be 196 KB before anything else, and halving `NBT` to pay for it doubles the
   `embT` re-read bytes for the same descriptor count. Taking `esb` from 3 buffers to 2 -- i.e.
   spending *less* SBUF, not more -- was worth 0.29 ms, which is the sign of the gradient here.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

L = 8192
HC = 4
DIM = 5120
NCOL = 24
HEAD_DIM = 256
INF = NCOL * HEAD_DIM
XF = HC * DIM
KROWS = XF
VROWS = DIM

# The floorplan's width and dimension: ngram x 4, one rank per logical NeuronCore of this one
# device. `inference.py` checks both against the width it launches at and builds the
# ReplicaGroup from WORLD_SIZE.
WORLD_SIZE = 4
N_CORES = 2

VK = INF // WORLD_SIZE
BS = 32
NBLK = INF // BS
VNBLK = VK // BS
KT = 128
NK = INF // KT
VNK = VK // KT
PT = 128
MB = 512
NBT = 1024
NMB = NBT // MB
SLAB = 512
NRH = 4
TPG = 8
TPGV = 6
HDIM = DIM // NRH
KSLAB = 256
MSUB = 8
STW = MSUB * KT
MSUBV = 10
STWV = MSUBV * KT
NBG = 12
NTT = L // PT
NB_TOT = KROWS // NBT
PWP = NTT // N_CORES
PWA = 6
PW = PWA * PWP
CDEP = 4
# The key half, cut four ways by hc copy: rank `r` owns key rows [KSH*r, KSH*r + KSH), which
# are exactly hc copy `r`. Its five 1024-row blocks are the same five the gate's reduction for
# that copy already summed, in the same order, so the gate it produces is bit-identical to the
# replicated one. Within a rank the two physical cores split *tokens* instead of key rows, so
# each core owns NSLC of the 32 token slabs -- the same token tiles its gate and combine own.
KSH = KROWS // WORLD_SIZE
NKB = KSH // NBT
NSLC = (L // KSLAB) // N_CORES

# The n-gram table, re-laid-out once so a token's lookup is **one** indirect DMA instead of two.
# `embed.weight` is (46596, 256) fp8 bytes and `embed.scale` is (46596, 8) e8m0 bytes, in two
# separate HBM tensors, so `embed_body` gathered each row twice: 256 B for the quantized row and
# 8 B for its block scales, both keyed on the same n-gram id. Indirect DMA here costs per *row*
# far more than per byte -- the two gathers together are 393,216 software-DGE descriptors a rank
# and 34.9% of the wall, at 51 MB -- so concatenating the two rows into one 264-byte row removes
# half the descriptors for 3% more bytes. The table is rebuilt into `private_hbm` by both
# physical cores in full, like `ksc`/`vsc`/`wp_hbm`: a buffer the two cores fill *half each* must
# be `shared_hbm`, and an *indirect* read of `shared_hbm` costs 1.28 ms on this part.
TBLR = 46596
NBS = HEAD_DIM // BS
ROWB = HEAD_DIM + NBS
RPP = 4
TGRP = KT * RPP
NTG = TBLR // TGRP
TTAIL = TBLR - NTG * TGRP
EPS = 1e-20
CLAMP = 1e-6
INV_DIM = 1.0 / DIM
DIM_RSQRT = 0.0139754248593737


def _e8m0_to_bf16(dst_i16, byte_i16, part, nblk):
    nisa.tensor_scalar(dst=dst_i16, data=byte_i16, op0=nl.multiply, operand0=128,
                       op1=nl.add, operand1=1024)


def _dequant(dst, byte_i16, scale_bf, part, nblk, scratch_a, scratch_b, flat=None):
    nf = nblk * BS
    if flat is None:
        flat = byte_i16.ap(pattern=[[nf, part], [1, nf]])
    nisa.tensor_scalar(dst=scratch_a, data=flat, op0=nl.bitwise_and, operand0=0x80,
                       op1=nl.left_shift, operand1=8)
    nisa.tensor_scalar(dst=scratch_b, data=flat, op0=nl.bitwise_and, operand0=0x7F,
                       op1=nl.left_shift, operand1=7)
    nisa.tensor_tensor(dst=scratch_b, data1=scratch_b, data2=scratch_a,
                       op=nl.bitwise_or)
    nisa.tensor_tensor(
        dst=dst,
        data1=scratch_b.view(nl.float16),
        data2=scale_bf.ap(pattern=[[nblk, part], [1, nblk], [0, BS]]),
        op=nl.multiply,
    )


def _transpose_into(tsb, src_sb, m, ntile, tpb, tpg):
    for g in nl.static_range(ntile // tpg):
        for j in nl.static_range(tpg):
            nisa.nc_transpose(dst=tpb[:, j, :],
                              data=src_sb[:, (g * tpg + j) * KT:(g * tpg + j + 1) * KT])
        nisa.tensor_copy(dst=tsb[:, g * tpg:(g + 1) * tpg, m * KT:(m + 1) * KT], src=tpb)


def _store_transposed(dst_hbm, ncol, col0, tsb, ntile, stw, nbg):
    for g in nl.static_range(ntile // nbg):
        nisa.dma_copy(
            dst=dst_hbm.ap(pattern=[[ncol, KT], [KT * ncol, nbg], [1, stw]],
                           offset=g * nbg * KT * ncol,
                           scalar_offset=col0, indirect_dim=1),
            src=tsb[:, g * nbg:(g + 1) * nbg, :],
        )


def _build_table(tbl, wsrc, ssrc):
    """`embed.weight` and `embed.scale` interleaved into one 264-byte row per n-gram.

    Every DMA here is *strided*, not indirect, and all three are large: 1024 B a descriptor in,
    1056 B out. That is the whole point -- a strided descriptor on this part costs a few
    nanoseconds where an indirect one costs ~15, so ~35 K descriptors paid once delete 98 K
    indirect gathers a core from a loop that runs 8192 times.

    The interleave itself happens on a **compute** engine, not in the DMA: `nisa.tensor_copy` on
    `nl.uint8` tiles writes the two pieces into one staged row, so the write back to HBM is a
    single contiguous `RPP * ROWB` run a partition. Writing the two pieces straight into the
    264-byte HBM pitch instead is 1024 descriptors a group rather than 128, and measures
    +0.30 ms.
    """
    for g in nl.static_range(NTG):
        w = nl.ndarray((KT, RPP, HEAD_DIM), dtype=nl.uint8, buffer=nl.sbuf)
        s = nl.ndarray((KT, RPP, NBS), dtype=nl.uint8, buffer=nl.sbuf)
        stg = nl.ndarray((KT, RPP, ROWB), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wsrc.ap(
            pattern=[[RPP * HEAD_DIM, KT], [HEAD_DIM, RPP], [1, HEAD_DIM]],
            offset=g * TGRP * HEAD_DIM))
        nisa.dma_copy(dst=s, src=ssrc.ap(
            pattern=[[RPP * NBS, KT], [NBS, RPP], [1, NBS]],
            offset=g * TGRP * NBS))
        # The interleave happens in SBUF, on a compute engine, so the write back to HBM is one
        # fully contiguous `RPP * ROWB` run a partition. Done with two strided DMAs into the
        # 264-byte pitch instead, the same bytes cost 1024 descriptors a group rather than 128.
        nisa.tensor_copy(dst=stg[:, :, 0:HEAD_DIM], src=w)
        nisa.tensor_copy(dst=stg[:, :, HEAD_DIM:ROWB], src=s)
        nisa.dma_copy(dst=tbl.ap(
            pattern=[[RPP * ROWB, KT], [1, RPP * ROWB]],
            offset=g * TGRP * ROWB), src=stg)

    # 46596 = 91 * 512 + 4. The tail is four rows on four partitions.
    wt = nl.ndarray((TTAIL, HEAD_DIM), dtype=nl.uint8, buffer=nl.sbuf)
    st = nl.ndarray((TTAIL, NBS), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=wt, src=wsrc.ap(pattern=[[HEAD_DIM, TTAIL], [1, HEAD_DIM]],
                                      offset=NTG * TGRP * HEAD_DIM))
    nisa.dma_copy(dst=st, src=ssrc.ap(pattern=[[NBS, TTAIL], [1, NBS]],
                                      offset=NTG * TGRP * NBS))
    nisa.dma_copy(dst=tbl.ap(pattern=[[ROWB, TTAIL], [1, HEAD_DIM]],
                             offset=NTG * TGRP * ROWB), src=wt)
    nisa.dma_copy(dst=tbl.ap(pattern=[[ROWB, TTAIL], [1, NBS]],
                             offset=NTG * TGRP * ROWB + HEAD_DIM), src=st)


def _widen_scales(dst_hbm, scale_hbm, nrows, nblk, GB):
    for g0 in nl.static_range(nrows // (KT * GB)):
        rows = nl.ndarray((KT, GB, nblk), dtype=nl.int16, buffer=nl.sbuf)
        for g in nl.static_range(KT // BS):
            nisa.dma_copy(
                dst=rows[g * BS:(g + 1) * BS, :, :],
                src=scale_hbm.ap(
                    pattern=[[0, BS], [(KT // BS) * nblk, GB], [1, nblk]],
                    offset=(g0 * GB * (KT // BS) + g) * nblk),
            )
        shifted = nl.ndarray((KT, GB, nblk), dtype=nl.int16, buffer=nl.sbuf)
        _e8m0_to_bf16(shifted, rows, KT, nblk)
        nisa.dma_copy(
            dst=dst_hbm.ap(pattern=[[nblk, KT], [KT * nblk, GB], [1, nblk]],
                           offset=g0 * GB * KT * nblk),
            src=shifted.view(nl.bfloat16),
        )


@nki.jit
def kernel(input_0, input_1, shard_offset,
           layers_1_engram_k_weight, layers_1_engram_q_weight,
           layers_1_engram_wkv_key_scale, layers_1_engram_wkv_key_weight,
           layers_1_engram_wkv_value_scale, layers_1_engram_wkv_value_weight,
           layers_1_engram_embed_scale, layers_1_engram_embed_weight,
           replica_group):
    # The module's output, and the only IO tensor written here. bfloat16, because the reference
    # rounds the sum back to the input's dtype (`.to(x.dtype)`).
    out = nl.ndarray((1, L, HC, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out2 = out.reshape((L, XF))

    # The seam's two buffers: named, both nl.shared_hbm, neither an IO tensor. bfloat16, which
    # is the dtype the reference's own `value` has, and which the module docstring prices
    # against a float32 reduction on the four recorded per-rank goldens: same worst element.
    cc_src = nl.ndarray((L, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((L, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cc_dst")

    # The gate's seam. Rank `r` now computes hc copy `r`'s gate for every token and nothing
    # else, so rejoining it is a concatenation, not a sum: one `all_gather` of (8192, 1)
    # float32 -> (32768, 1), 128 KB, which is 0.16% of the value reduction's payload. float32
    # because the gate multiplies a bfloat16 `value` whose product already sets the output's
    # own ulp; rounding the gate itself would spend error budget this module does not have.
    cc_gs = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_gs")
    cc_gd = nl.ndarray((WORLD_SIZE * L, 1), dtype=nl.float32, buffer=nl.shared_hbm,
                       name="cc_gd")

    h2 = input_0.reshape((L, XF))
    ids2 = input_1.reshape((L, NCOL))
    qw2 = layers_1_engram_q_weight.reshape((1, XF))
    kw2 = layers_1_engram_k_weight.reshape((1, XF))

    # Which of this rank's two physical cores this instruction stream is. A trace-time constant,
    # so every split below is static and both cores' schedules are fixed.
    pid = nl.program_id(0)
    nprog = nl.num_programs()
    tok_lo = pid * (L // nprog)
    tok_hi = tok_lo + (L // nprog)
    slab_lo = pid * (L // SLAB // nprog)
    slab_hi = slab_lo + (L // SLAB // nprog)

    # This rank's first row of `embT`: `VK * rank`, as data. A runtime scalar offset, so the loop
    # that reads it keeps the trace-time bounds every rank shares.
    voff = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=voff, src=shard_offset)

    # This rank's first *key* row, `KSH * rank`, from the same datum. `KSH / VK = 10 / 3` is
    # not an integer, so it cannot be a DMA pattern stride; it is three instructions on a
    # (1, 1) tile instead. The `+ 0.5` makes the float32 -> int32 narrowing exact whether the
    # datapath rounds or truncates: the product is within 0.0012 of its integer.
    koff_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    koff = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=koff_f, src=voff)
    nisa.tensor_scalar(dst=koff_f, data=koff_f, op0=nl.multiply,
                       operand0=float(KSH) / float(VK), op1=nl.add, operand1=0.5)
    nisa.tensor_copy(dst=koff, src=koff_f)

    ksc = nl.ndarray((KROWS, NBLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    vsc = nl.ndarray((VROWS, VNBLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    embT = nl.ndarray((INF, L), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    vT = nl.ndarray((VK, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wp_hbm = nl.ndarray((1, XF), dtype=nl.float32, buffer=nl.private_hbm)
    tbl = nl.ndarray((TBLR, ROWB), dtype=nl.uint8, buffer=nl.private_hbm)

    _widen_scales(ksc, layers_1_engram_wkv_key_scale, KROWS, NBLK, 16)
    _widen_scales(vsc, layers_1_engram_wkv_value_scale, VROWS, VNBLK, 8)
    _build_table(tbl, layers_1_engram_embed_weight, layers_1_engram_embed_scale)

    qsb = nl.ndarray((1, XF), dtype=nl.bfloat16, buffer=nl.sbuf)
    ksb = nl.ndarray((1, XF), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qsb, src=qw2)
    nisa.dma_copy(dst=ksb, src=kw2)
    wpsb = nl.ndarray((1, XF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wpsb, data1=qsb, data2=ksb, op=nl.multiply)
    nisa.dma_copy(dst=wp_hbm, src=wpsb)

    def embed_body(t0):
        tsb = nl.ndarray((KT, NK, STW), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpb = nl.ndarray((KT, TPG, PT), dtype=nl.bfloat16, buffer=nl.psum)
        idx = nl.ndarray((PT, MSUB, NCOL), dtype=nl.uint32, buffer=nl.sbuf)
        # One tile for both halves of the interleaved row: bytes [0, 256) are the quantized
        # n-gram and [256, 264) its eight e8m0 block scales. Same 12,672 B/lane the separate
        # `q` and `s` tiles cost, and one gather instead of two.
        qs = nl.ndarray((PT, NCOL, ROWB), dtype=nl.int16, buffer=nl.sbuf)
        esc = nl.ndarray((PT, NCOL, NBS), dtype=nl.int16, buffer=nl.sbuf)
        sa = nl.ndarray((PT, INF), dtype=nl.int16, buffer=nl.sbuf)
        sb = nl.ndarray((PT, INF), dtype=nl.int16, buffer=nl.sbuf)
        emb = nl.ndarray((PT, INF), dtype=nl.bfloat16, buffer=nl.sbuf)

        nisa.dma_copy(dst=idx, src=ids2.ap(
            pattern=[[NCOL, PT], [PT * NCOL, MSUB], [1, NCOL]],
            scalar_offset=t0, indirect_dim=0))

        for m in nl.static_range(MSUB):
            for j in nl.static_range(NCOL):
                nisa.dma_copy(
                    dst=qs[:, j, :],
                    src=tbl.ap(
                        pattern=[[ROWB, PT], [1, ROWB]],
                        vector_offset=idx[:, m, j:j + 1], indirect_dim=0),
                )

            _e8m0_to_bf16(esc, qs[:, :, HEAD_DIM:ROWB], PT, NBLK)
            _dequant(emb, None, esc.view(nl.bfloat16), PT, NBLK, sa, sb,
                     flat=qs.ap(pattern=[[NCOL * ROWB, PT], [ROWB, NCOL],
                                         [1, HEAD_DIM]]))
            _transpose_into(tsb, emb, m, NK, tpb, TPG)

        _store_transposed(embT, L, t0, tsb, NK, STW, NK)

    nl.fori_loop(tok_lo, tok_hi, embed_body, step=STW)
    if nprog > 1:
        nisa.core_barrier(embT, cores=(0, 1))

    def value_weight_body(r0):
        tsb = nl.ndarray((KT, VNK, STWV), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpb = nl.ndarray((KT, TPGV, KT), dtype=nl.bfloat16, buffer=nl.psum)
        wq = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        wsc = nl.ndarray((KT, VNBLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        sa = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        sb = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        wbf = nl.ndarray((KT, VK), dtype=nl.bfloat16, buffer=nl.sbuf)

        for m in nl.static_range(MSUBV):
            nisa.dma_copy(dst=wq, src=layers_1_engram_wkv_value_weight.ap(
                pattern=[[VK, KT], [1, VK]], offset=m * KT * VK,
                scalar_offset=r0, indirect_dim=0))
            nisa.dma_copy(dst=wsc, src=vsc.ap(
                pattern=[[VNBLK, KT], [1, VNBLK]], offset=m * KT * VNBLK,
                scalar_offset=r0, indirect_dim=0))
            _dequant(wbf, wq, wsc, KT, VNBLK, sa, sb)
            _transpose_into(tsb, wbf, m, VNK, tpb, TPGV)

        _store_transposed(vT, VROWS, r0, tsb, VNK, STWV, VNK)

    v_lo = pid * (VROWS // nprog)
    nl.fori_loop(v_lo, v_lo + (VROWS // nprog), value_weight_body, step=STWV)
    if nprog > 1:
        nisa.core_barrier(vT, cores=(0, 1))

    # ------------------------------------------------------------------ the value half
    # This rank's 1536 contraction columns -- its quarter of `ngram` -- over all 5120 output
    # columns. A partial sum of the module's `value`, which is what crosses the wire.
    #
    # The value half runs *before* the key half, which is 7x its size and has no data
    # dependence on it, so the ring crosses under the key GEMM instead of in front of an idle
    # machine. Issued the other way round the collective was 1.364 ms with every compute
    # engine at 0-4%, 5.6% of the wall. (`d1ba829dac9c` / `8df4ee2cf87b` report that this
    # placement fails to compile as `NCC_ISCH719` when the following loop's DMAs carry a
    # `ncc.rank_id()`-derived `scalar_offset`. It compiles here: the rank-derived offset is in
    # `value_out_body`, which is now *before* the collective, and the key loop's own
    # `scalar_offset` is its `nl.fori_loop` index, which every rank shares.)
    def value_out_body(n0):
        wsb = nl.ndarray((KT, VNK, VROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in nl.static_range(VNK):
            nisa.dma_copy(dst=wsb[:, kb, :],
                          src=vT.ap(pattern=[[VROWS, KT], [1, VROWS]],
                                    offset=kb * KT * VROWS,
                                    scalar_offset=n0, indirect_dim=1))
        esb = nl.ndarray((KT, VNK, SLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
        val0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        val1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vals = (val0, val1)

        base = slab_lo * (SLAB // PT)
        k = 0
        for sl in range(slab_lo, slab_hi):
            for kb in nl.static_range(VNK):
                # The one rank-dependent address in the kernel: rows
                # `[VK * rank + kb * 128, ...)` of the replicated `embT`.
                nisa.dma_copy(dst=esb[:, kb, :],
                              src=embT.ap(pattern=[[L, KT], [1, SLAB]],
                                          offset=kb * KT * L + sl * SLAB,
                                          scalar_offset=voff, indirect_dim=0))
            for m in range(SLAB // PT):
                value = vals[k % 2]
                for nb in nl.static_range(VROWS // MB):
                    ps = nl.ndarray((PT, MB), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(VNK):
                        nisa.nc_matmul(
                            dst=ps,
                            stationary=esb[:, kb, m * PT:(m + 1) * PT],
                            moving=wsb[:, kb, nb * MB:(nb + 1) * MB],
                            accumulate=(kb > 0),
                        )
                    nisa.tensor_copy(dst=value[:, nb * MB:(nb + 1) * MB], src=ps,
                                     engine=nisa.engine.vector)

                if k > 0:
                    pv = vals[(k - 1) % 2]
                    pt0 = (base + k - 1) * PT
                    nisa.dma_copy(dst=cc_src[pt0:pt0 + PT, :], src=pv)
                k += 1

        lv = vals[(k - 1) % 2]
        lt0 = (base + k - 1) * PT
        nisa.dma_copy(dst=cc_src[lt0:lt0 + PT, :], src=lv)

    nl.fori_loop(0, VROWS, value_out_body, step=VROWS)

    # ------------------------------------------------------------------- the key half
    # Only this rank's hc copy. `KSH * rank` rides on `koff` exactly as `VK * rank` rides on
    # `voff` in `value_out_body`; the five blocks are a `static_range` rather than a
    # `fori_loop` because a runtime `scalar_offset` cannot be added to a loop index, and the
    # per-block buffers are hoisted out so the five copies share one allocation instead of
    # five (the `fori_loop` got that for free).
    #
    # The two physical cores split the 32 token slabs rather than the key rows, so each core's
    # key GEMM, gate and combine all own the same token tiles -- which removes the `pb` HBM
    # round trip and the `core_barrier` on it that the replicated version needed.
    pa3 = nl.ndarray((PT, NKB * 3, PWP), dtype=nl.float32, buffer=nl.sbuf)
    wsb = nl.ndarray((KT, NK, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
    wq = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wsc = nl.ndarray((KT, NBLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    wsa = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wsx = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wbf = nl.ndarray((KT, INF), dtype=nl.bfloat16, buffer=nl.sbuf)
    tpb = nl.ndarray((KT, TPG, KT), dtype=nl.bfloat16, buffer=nl.psum)
    wpk = nl.ndarray((KT, NBT), dtype=nl.float32, buffer=nl.sbuf)
    esb0 = nl.ndarray((KT, NK, KSLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
    esb1 = nl.ndarray((KT, NK, KSLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
    esbs = (esb0, esb1)
    hs = nl.ndarray((PT, KSLAB // PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
    sl0 = pid * NSLC

    for kblk in nl.static_range(NKB):
        kbase = kblk * NBT
        for rb in nl.static_range(NBT // KT):
            nisa.dma_copy(dst=wq, src=layers_1_engram_wkv_key_weight.ap(
                pattern=[[INF, KT], [1, INF]], offset=(kbase + rb * KT) * INF,
                scalar_offset=koff, indirect_dim=0))
            nisa.dma_copy(dst=wsc, src=ksc.ap(
                pattern=[[NBLK, KT], [1, NBLK]], offset=(kbase + rb * KT) * NBLK,
                scalar_offset=koff, indirect_dim=0))
            _dequant(wbf, wq, wsc, KT, NBLK, wsa, wsx)
            for g in nl.static_range(NK // TPG):
                for j in nl.static_range(TPG):
                    kb = g * TPG + j
                    nisa.nc_transpose(dst=tpb[:, j, :],
                                      data=wbf[:, kb * KT:(kb + 1) * KT])
                nisa.tensor_copy(dst=wsb[:, g * TPG:(g + 1) * TPG,
                                        rb * KT:(rb + 1) * KT], src=tpb)

        nisa.dma_copy(dst=wpk, src=wp_hbm.ap(pattern=[[0, KT], [1, NBT]],
                                             offset=kbase,
                                             scalar_offset=koff, indirect_dim=1))

        for kb in nl.static_range(NK):
            nisa.dma_copy(dst=esbs[0][:, kb, :],
                          src=embT[kb * KT:(kb + 1) * KT,
                                   sl0 * KSLAB:(sl0 + 1) * KSLAB])

        for sl in nl.static_range(0, NSLC):
            esb = esbs[sl % 2]
            if sl + 1 < NSLC:
                nxt = sl + 1
                eb = esbs[nxt % 2]
                for kb in nl.static_range(NK):
                    nisa.dma_copy(dst=eb[:, kb, :],
                                  src=embT[kb * KT:(kb + 1) * KT,
                                           (sl0 + nxt) * KSLAB:(sl0 + nxt + 1) * KSLAB])
            nisa.dma_copy(dst=hs, src=h2.ap(
                pattern=[[XF, PT], [PT * XF, KSLAB // PT], [1, NBT]],
                offset=(sl0 + sl) * KSLAB * XF + kbase,
                scalar_offset=koff, indirect_dim=1))

            for m in nl.static_range(KSLAB // PT):
                tile = sl * (KSLAB // PT) + m
                osb = nl.ndarray((PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
                ps = nl.ndarray((PT, NBT), dtype=nl.float32, buffer=nl.psum)
                for nb in nl.static_range(NMB):
                    for kb in nl.static_range(NK):
                        nisa.nc_matmul(
                            dst=ps[:, nb * MB:(nb + 1) * MB],
                            stationary=esb[:, kb, m * PT:(m + 1) * PT],
                            moving=wsb[:, kb, nb * MB:(nb + 1) * MB],
                            accumulate=(kb > 0),
                        )
                nisa.tensor_copy(dst=osb, src=ps, engine=nisa.engine.scalar)

                hsb = hs[:, m, :]
                # One dead destination for both squares: the reduce result is the only thing
                # read out of either, and both run on the same engine anyway.
                dk = nl.ndarray((PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
                a0 = kblk * 3
                nisa.activation(dst=dk, op=nl.square, data=osb, reduce_op=nl.add,
                                reduce_res=pa3[:, a0, tile:tile + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.activation(dst=dk, op=nl.square, data=hsb, reduce_op=nl.add,
                                reduce_res=pa3[:, a0 + 1, tile:tile + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                prod = nl.ndarray((PT, NBT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=hsb, data2=wpk, op=nl.multiply)
                nisa.tensor_tensor(dst=prod, data1=prod, data2=osb, op=nl.multiply)
                nisa.tensor_reduce(dst=pa3[:, a0 + 2, tile:tile + 1],
                                   op=nl.add, data=prod, axis=(1,))

    # The gate for hc copy `rank`, over this core's token tiles. The five blocks are summed in
    # increasing order, which is the order the replicated version summed them in, so every
    # float32 add here has the same two operands it had before.
    sk = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    sh = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    sd = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sk, data1=pa3[:, 0, :], data2=pa3[:, 3, :], op=nl.add)
    nisa.tensor_tensor(dst=sh, data1=pa3[:, 1, :], data2=pa3[:, 4, :], op=nl.add)
    nisa.tensor_tensor(dst=sd, data1=pa3[:, 2, :], data2=pa3[:, 5, :], op=nl.add)
    for b in nl.static_range(2, NKB):
        nisa.tensor_tensor(dst=sk, data1=sk, data2=pa3[:, 3 * b, :], op=nl.add)
        nisa.tensor_tensor(dst=sh, data1=sh, data2=pa3[:, 3 * b + 1, :], op=nl.add)
        nisa.tensor_tensor(dst=sd, data1=sd, data2=pa3[:, 3 * b + 2, :], op=nl.add)

    tmp = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    rh = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    rk = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=tmp, op=nl.sqrt, data=sh, scale=INV_DIM, bias=EPS)
    nisa.reciprocal(dst=rh, data=tmp)
    nisa.activation(dst=tmp, op=nl.sqrt, data=sk, scale=INV_DIM, bias=EPS)
    nisa.reciprocal(dst=rk, data=tmp)

    red = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=red, data1=sd, data2=rh, op=nl.multiply)
    nisa.tensor_tensor(dst=red, data1=red, data2=rk, op=nl.multiply)
    nisa.tensor_scalar(dst=red, data=red, op0=nl.multiply, operand0=DIM_RSQRT)

    mag = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=mag, op=nl.abs, data=red)
    nisa.tensor_scalar(dst=mag, data=mag, op0=nl.maximum, operand0=CLAMP)
    nisa.activation(dst=mag, op=nl.sqrt, data=mag)
    sgn = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sgn, op=nl.sign, data=red)
    nisa.tensor_tensor(dst=mag, data1=mag, data2=sgn, op=nl.multiply)
    gl = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gl, op=nl.sigmoid, data=mag)

    # ------------------------------------------------------------------------- the seam
    # Both collectives, together, at the one point where every rank holds an incomplete
    # answer: `cc_src` is a partial sum over `ngram`, `cc_gs` is one of the four hc copies of
    # the gate. The two physical cores fill disjoint token runs of each; the barriers are what
    # make the other core's rows visible to the reads the collectives are about to do.
    #
    # The value reduce used to sit *ahead* of the key half, because the key half is seven
    # times its size and independent of it, and that is the only placement that could overlap
    # the ring with compute. It cannot sit there any more: the key loop now carries a
    # rank-derived `scalar_offset` and `neuronx-cc` rejects that ordering as
    # `[NCC_ISCH719] topological order violations` -- the failure `8df4ee2cf87b` reports,
    # provoked here from an *input-carried* offset rather than `ncc.rank_id()`, so the
    # mechanism is the runtime offset itself and not where it came from. The cost of giving
    # that placement up is small and measured: the profile showed the reduce 100% exposed in
    # the early position too, 0.645 ms with every compute engine idle underneath it, so
    # nothing was hiding behind it to lose.
    base = pid * PWP
    nisa.dma_copy(dst=cc_gs.ap(pattern=[[1, PT], [PT, PWP]], offset=base * PT), src=gl)
    if nprog > 1:
        nisa.core_barrier(cc_src, cores=(0, 1))
        nisa.core_barrier(cc_gs, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    ncc.all_gather(srcs=[cc_gs], dsts=[cc_gd], replica_group=replica_group,
                   collective_dim=0)

    # Column `c * PWP + i` is hc copy `c` of token tile `pid * PWP + i`. Flat rather than
    # (PT, HC, PWP) so that one column is a 2-D (PT, 1) per-partition scalar for
    # `scalar_tensor_tensor` below.
    gt = nl.ndarray((PT, HC * PWP), dtype=nl.float32, buffer=nl.sbuf)
    for c in nl.static_range(HC):
        nisa.dma_copy(dst=gt[:, c * PWP:(c + 1) * PWP],
                      src=cc_gd.ap(pattern=[[1, PT], [PT, PWP]],
                                   offset=c * L + base * PT))

    # ------------------------------------------------------------------------ the output
    # After the reduce every rank holds the module's whole `value`, rank 0 included.
    # `out[:, c, :] = bfloat16(h[:, c, :] + gate[:, c] * value)`, one `scalar_tensor_tensor`
    # per (token tile, hc copy), over the token tiles this physical core's gate covers.
    #
    # This pass is the assembly's real cost, and it is a DMA pipeline rather than an arithmetic
    # one: 377 MB a physical core (42 MB of `value`, 168 MB of `h`, 168 MB of output) against
    # 128 instructions of 5120 columns. The residual stream has to be read a second time -- the
    # key half read it once already, for `h.square().mean(-1)` and the dot -- and the module's
    # whole output written, and neither has anything to overlap with, because the gate is only
    # complete when the last key row is. `CDEP` is the depth of the (load h, combine, store out)
    # pipeline. Going from 2 to 4 was measured together with moving the collective, so the
    # 24.1458 -> 22.2596 ms those two bought jointly is not attributable to either alone; what
    # is attributable is 6 against 4 on an otherwise identical kernel, 21.3007 against
    # 21.2985 ms. So 4 is already where the outstanding descriptors stop being the limit.
    def combine_body(n0):
        vv0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vv1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh2 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh3 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo2 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo3 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vvs = (vv0, vv1)
        hhs = (hh0, hh1, hh2, hh3)
        oos = (oo0, oo1, oo2, oo3)

        base = pid * PWP
        NU = PWP * HC
        nisa.dma_copy(dst=vvs[0], src=cc_dst[base * PT:(base + 1) * PT, :])
        for u in range(CDEP - 1):
            nisa.dma_copy(dst=hhs[u], src=h2.ap(
                pattern=[[XF, PT], [1, DIM]],
                offset=(base + u // HC) * PT * XF + (u % HC) * DIM))

        for u in range(NU):
            i = u // HC
            c = u % HC
            if c == 0 and i + 1 < PWP:
                nisa.dma_copy(dst=vvs[(i + 1) % 2],
                              src=cc_dst[(base + i + 1) * PT:(base + i + 2) * PT, :])
            v = u + CDEP - 1
            if v < NU:
                nisa.dma_copy(dst=hhs[v % CDEP],
                              src=h2.ap(pattern=[[XF, PT], [1, DIM]],
                                        offset=(base + v // HC) * PT * XF
                                        + (v % HC) * DIM))
            nisa.scalar_tensor_tensor(dst=oos[u % CDEP], data=vvs[i % 2],
                                      op0=nl.multiply,
                                      operand0=gt[:, c * PWP + i:c * PWP + i + 1],
                                      op1=nl.add, operand1=hhs[u % CDEP])
            nisa.dma_copy(dst=out2.ap(pattern=[[XF, PT], [1, DIM]],
                                      offset=(base + i) * PT * XF + c * DIM),
                          src=oos[u % CDEP])

    nl.fori_loop(0, PT, combine_body, step=PT)

    return out
