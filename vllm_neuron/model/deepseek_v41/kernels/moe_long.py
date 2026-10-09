"""All of `layers.1.ffn` at 8192 tokens on 4 ranks, rejoining through `nki.collectives`.

Every rank runs the whole `ffn_norm -> router` prologue (replicated), the SwiGLU experts for its
own 96-expert shard, and one quarter of the shared expert. It adds all of that into a float32
accumulator, and `ncc.all_reduce` sums the four accumulators. The sum is the module's output for
every rank, rounded to the module's bfloat16 once, at the end.

The routed-expert path -- the prologue, the token/expert compaction, and the fp4 expert loop -- is
`submodule/source.py` at its best accepted commit, unchanged apart from where its results land.
What is added here is the shared expert, the split across the two physical cores, and the seam.

**The shared expert is sharded, not replicated.** `MoE.forward` adds it after `dist.all_reduce`,
which is why `submodule/SUBMODULE.md` left it out of a rank's partial and made it a `then_add`
term of the reassembly. At module scale that placement is a choice, not a constraint: the reduce
is a sum, so the shared expert only has to appear across the four partials exactly once in total
-- and splitting it beats replicating it, because SwiGLU is elementwise along the intermediate
dimension and `w2` contracts over it. Rank *r* owns intermediate columns `r*640 .. r*640+639` of
`w1`/`w3` and the matching `w2` rows; the four `w2` partials sum to the whole shared expert. The
intermediate is padded 2304 -> `INTER_PAD` = 2560 so that all four ranks get the same shapes (one
NEFF serves all four) and every slice is a whole number of 128-row GEMM tiles; the padding is zero
weights, which contribute zero. Checked on the host, straight from `tensors/*.bin`, in
`scratch/sh_ref.py`: the four shards sum to the unsharded shared expert to 0.0039, which is the
bfloat16 rounding of four partials rather than one, against a 0.34375 ceiling.

**The fp8 shared-expert weights are decoded by hand, once, into bfloat16 in HBM.** Two separate
points.

*By hand*, because the hardware's fp8 converter is not an option: `nl.float8_e4m3` and
`nl.float8_e4m3fn` both read exponent 15 as inf/NaN on this toolchain, and 0.52% of `w1`'s bytes,
0.61% of `w2`'s and 0.67% of `w3`'s live there -- 61399 / 71502 / 78556 of 11796480, counted from
`tensors/` here -- so converting through it produces NaN rather than a small error. `_fp8_dequant`
builds the bfloat16 bit pattern directly, `(b & 0x80) << 8 | (b & 0x7f) << 4`, which reads as the
value times `2**-120` uniformly across normals and subnormals, and `_scale_rows` folds the
`2**120` back into the e8m0 scale's exponent so the correction is free: four elementwise
instructions per weight tile, two of them on a uint32 view carrying two weights per lane.

*Once*, because at 8192 tokens there are 64 token tiles where the 128-token predecessor had one.
That predecessor decoded every weight tile inside the GEMM, which cost nothing when each tile was
used once; here it would be paid 64 times over. `bits * scale` is exact in bfloat16 -- an fp8 e4m3
value has 4 significant bits and an e8m0 scale is a power of two -- so the pre-decoded weight is
bit-identical to what the in-GEMM decode produces. The whole slice is 9.83 M weights, 19.7 MB of
bfloat16, and the decode is paid once per invocation.

**A rank's work is split across the two physical cores of its logical core.** This host is LNC=2,
so the kernel is launched as `kernel[2]` and a rank's logical core is two physical cores. Core *c*
takes every other expert group of the plan the compaction lays out -- groups `c, c+2, c+4, ...` --
and the shared expert for token tiles `c*32 .. c*32+31`; it writes its own half of `part` and
zeroes the token tiles of that half it did not compute, so the two halves sum to one whole rank
partial. `nisa.core_barrier` makes the other half readable and both cores then sum the two halves
into the collective's input.

Two things make that split cheap. `nl.program_id(0)` is a **trace-time constant** -- the launch
grid is traced once per physical core and the NEFF carries both programs -- so *which* groups and
which token tiles a core takes is ordinary Python, with no predication. And every group costs the
same whether or not a token routes to its expert, so halving the count halves the work with no load
imbalance to absorb.

**The routed loop runs only the groups the routing filled, and it is the same count on both
cores.** The plan is a prefix sum, so the groups that hold a token are a contiguous prefix of it
and the empty ones a suffix; `cum[CH - 1]` is how long that prefix is, computed on device from this
invocation's own routing. The loop's bound is that count divided by `CORES`, rounded up to keep the
two cores' trip counts identical -- divergent trip counts across one logical core's two physical
cores are the shape of several of this toolchain's hangs -- and clamped to the provisioned `NPGC`.
That clamp is load-bearing rather than cosmetic: capacity is still the static worst case `NPGT`
(`CH` part-full groups plus the spill of `TOKENS * TOPK` pairs), `pk3`, `eidh` and `part` are still
sized for it, and a routing that overruns the plan runs the full static loop. The interleave is
what makes the dynamic bound worth anything: with the cores taking contiguous halves, one core held
the whole live prefix and shortening the loop moved nothing.

A skipped group is **bit-identical**, not within tolerance: `pk3` gives an empty slot routing
weight 0.0, so the group's contribution is an exact `+0.0` into a float32 accumulator, scattered
into a trash tile nothing reads.

**Every indirect access is in bounds.** The submodule leaves a group no token routes to pointing
at token index `BIG` and relies on `oob_mode.skip` to drop the scatter. On this toolchain that is
not safe: the same scatter, at the same indices, either skips silently or reports "scatter/gather
(indirect memory copy via scalar DGE) out-of-bound access" and wedges the DMA engine with
`NRT_EXEC_HW_ERR_DMA_ABORT`, depending on what else is in the kernel. `part` therefore carries one
trash token tile per core and the scatter index is clamped into it, and `eidh` carries one trash
row and the group-id scatter is clamped into that. Nothing here is ever out of range; `oob_mode`
is left on only as a backstop. `part` is also an **IO tensor** -- returned and ignored -- because
the same scatter into `private_hbm`, or into named `shared_hbm` scratch, fails the same way.
`ASSEMBLY.md` records all four measurements.

The kernel is rank-agnostic on purpose -- one NEFF runs on all four ranks. A rank's identity lives
entirely in the values `inference.py` hands it: its own 96 experts' weights, its slice of the
shared expert, and a `gate_wt`/`gate_b` whose 384 columns are permuted so that this rank's experts
are columns 0..95. Top-6 selection by value, the renormalising sum and the bias add are all
invariant under a column permutation, so the router still scores every one of the 384 experts; only
the column at which a local expert's weight lands moves.

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with an
internal compiler error that names something else:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 and a LNC=1 NEFF leaves the
   collective's buffer unallocated on the logical core's second physical core (`NCC_ILLC059`);
2. `src`/`dst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`);
3. they are scratch, because a collective may not read or write an IO tensor (`NCC_INLA001`) --
   hence the `dma_copy` in and the tiled `nl.store` out;
4. both are `nl.shared_hbm`, because buffer kinds may not be mixed;
5. the `ReplicaGroup` is built by `inference.py` and passed in, because the tracer rejects `range`
   in a traced body;
6. the inputs are built on CPU and `.to(device)`\'d, because `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902` -- and that error is the broadcast, not the collective.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
N_EXPERTS = 384
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
SUM_EPS = 1e-20
# Above this logit `softplus(x)` is `x` to the last float32 bit, so the router evaluates it on a
# capped argument and floors the result at the uncapped one. Nothing about this input picks the
# number: `log1p(exp(-20)) = 2.06e-9` is below half an ulp of 20.0 in float32, which is the only
# property it has to have, and it holds for every value above it too.
SP_CAP = 20.0
TOK = 128

PAR = 128
NMAX = 512
ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4

CH = 96
BPB = ABLK // 2
KBLK1 = DIM // ABLK
KBLK2 = INTER // ABLK
KB1 = DIM // 2
KB2 = INTER // 2
NW1 = 768
NW2 = 1024
NBG1 = 1
NBG2 = 1
GTOK = 4
KSP1 = 4
KSP2 = 3
MIDENG = nisa.engine.vector

SCENG1 = (nisa.engine.scalar,)
SCENG2 = (nisa.engine.scalar,)
DRNENG = (nisa.engine.scalar,)
TRENG = (nisa.engine.scalar,)
TRB = 8
# The nibble premultiply (`_gemm_fp4` stage 3) stays wholly on Scalar. It is `nl.multiply`, an
# arithmetic op, so unlike the nibble masks -- bitvec, which the ISA pins to Vector -- it *may* run
# on GpSimd, and Scalar is the least loaded compute engine here (49% of the wall against vector's
# 58.5% and gpsimd's 59%). Alternating the two planes across Scalar and GpSimd measures
# **125.168 ms against 40.1474**, 3.1x slower, bit-identical: GpSimd is an order of magnitude
# slower per element on a wide `(128, nw)` elementwise op than either Vector or Scalar, so its
# share of this profile's wall is many cheap instructions, not work worth taking from it. Do not
# move any wide elementwise instruction onto GpSimd.
PRENG1 = (nisa.engine.scalar,)
PRENG2 = (nisa.engine.scalar,)

EXP_BIAS = 32
BF_BIAS = 127
F32_MANT = 23
F32_MANT_MAX = (1 << F32_MANT) - 1
RATIO_MAX = float(1 << (EXP_BIAS - 1))
RATIO_MIN_BITS = ((BF_BIAS - EXP_BIAS - 1) << F32_MANT) + 1
BF_STEP = 128.0
BF_HALF_BIAS = float((2 * BF_BIAS - 1) << 7)

NIB_MUL = 4160.0
PAT_MASK2 = -2118090304
PAIR_NIB_MASK = 0x0F0F
HI_NIB_SHIFT = 4
# `KSP1 = 4` / `PDEP1 = 6` is a measured local optimum in both directions, and the two are not
# independent: `nib`/`pre`/`dbufs` all cost `PDEP1 * KSP1` units, so trading jobs per pipeline stage
# against rotation depth at roughly constant SBUF is the one schedule axis round 1 left open (it
# closed `LEAD*`/`DLEAD` and stage reordering as no-ops). Both directions lose, and by more than any
# structural change in this iteration won or lost:
#
#   `KSP1 = 5, PDEP1 = 5` (5 jobs per stage, 4 contraction steps, 50 units)  41.7333  +1.586
#   `KSP1 = 4, PDEP1 = 8` (deeper rotation, 64 units -- it *does* fit SBUF)  41.9042  +1.757
#
# So depth 6 is not a buffer-budget compromise, it is the optimum, and more prefetch run-ahead is
# actively worse. Do not spend another run anywhere on this axis.
PDEP1 = 6
PDEP2 = 3
LEAD1 = 2
LEAD2 = 2
CHSP1 = True
CHSP2 = False
SC_MUL = 128.0
SC_ADD = 16128.0

# ---------------------------------------------------------------- the shared expert's shard
# `inference.py` reads INTER_PAD from here rather than keeping its own copy: the host packing
# and the kernel have to agree on it exactly, and a second copy of the number is a way for them
# to disagree quietly. At least INTER, and a multiple of SHARDS * PAR so that every rank's shard
# is a whole number of 128-row GEMM tiles.
INTER_PAD = 2560
SHARDS = 4
INTER_L = INTER_PAD // SHARDS            # 640 intermediate columns per rank
NT1 = DIM // PAR                         # 40 contraction tiles of the w1/w3 GEMMs
NT2 = INTER_L // PAR                     # 5 contraction tiles of the w2 GEMM
assert INTER_PAD >= INTER and INTER_PAD % (SHARDS * PAR) == 0

GSH = 4                                  # token tiles per shared-expert loop body
WDEP = 3                                 # rotating weight-tile buffers in _gemm_bf16
WLEAD = 2                                # k-tiles the weight DMA runs ahead of its matmuls
assert WDEP > WLEAD >= 0
PREDQ_W = 2560                           # column chunk of the one-time fp8 weight decode

# fp8 e4m3fn weight decode, see _fp8_dequant / _scale_rows
BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)       # 0x80008000 written as a signed int32
SC_MUL8 = 128.0
SC_ADD8 = float(120 << 7)                # folds the decode's 2**-120 back in


def _act_quant(x_bf16, m, k, chunks, dst=None, koff=0, natout=None):
    nb = k // ABLK
    xf = x_bf16.reshape((m, nb, ABLK))
    mx = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=xf, axis=2)
    nmn = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nmn, op=nl.min, data=xf, axis=2, negate=True)
    amax = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=amax, data=mx, op0=nl.maximum, operand0=AMAX_FLOOR,
                              op1=nl.maximum, operand1=nmn)

    ratio = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ratio, data=amax, op0=nl.multiply, operand0=1.0 / FP8_MAX,
                       op1=nl.minimum, operand1=RATIO_MAX, engine=nisa.engine.vector)
    eb = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eb, data=ratio.view(nl.int32), op0=nl.maximum,
                       operand0=RATIO_MIN_BITS, op1=nl.add, operand1=F32_MANT_MAX,
                       engine=nisa.engine.vector)
    ef = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ef, data=eb, op0=nl.right_shift, operand0=F32_MANT,
                       engine=nisa.engine.vector)

    sb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sb, data=ef, op0=nl.multiply, operand0=BF_STEP,
                       engine=nisa.engine.vector)
    s = sb.view(nl.bfloat16)
    ivb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ivb, data=ef, op0=nl.multiply, operand0=-BF_STEP,
                       op1=nl.add, operand1=BF_HALF_BIAS, engine=nisa.engine.vector)
    inv = ivb.view(nl.bfloat16)

    ib = inv.reshape((m, nb, 1)).broadcast(2, ABLK)

    q = nl.ndarray((m, nb, ABLK), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q, data1=xf, data2=ib, op=nl.multiply,
                       engine=nisa.engine.vector)
    if natout is not None:
        # The same quantized activation in the plain (m, k) layout, for the shared expert's
        # fp8 GEMM, which wants it transposed rather than flattened over (k-block, byte
        # offset). One extra instruction on a tile that is already built: `q` and `s` are the
        # whole of the quantization, and the fp4 layout below is only a permutation of them.
        nisa.tensor_tensor(dst=natout, data1=q,
                           data2=nl.multiply(s, 2.0).reshape((m, nb, 1)).broadcast(2, ABLK),
                           op=nl.multiply, engine=nisa.engine.vector)
    q4 = q.reshape((m, nb, BPB, 2))
    fls = []
    for c in range(len(chunks)):
        kb0 = chunks[c][0]
        nk = chunks[c][1]
        s2 = nl.multiply(s[:, kb0:kb0 + nk], 2.0).reshape((m, 1, 1, nk))
        if dst is None:
            fl = nl.ndarray((m, 2, BPB, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
            tgt = fl
        else:
            fl = dst[c]
            tgt = fl[:, :, :, koff:koff + nk]
        nisa.tensor_tensor(dst=tgt,
                           data1=q4[:, kb0:kb0 + nk, :, :].permute((0, 3, 2, 1)),
                           data2=s2.broadcast(1, 2).broadcast(2, BPB),
                           op=nl.multiply, engine=nisa.engine.vector)
        fls.append(fl)
    return fls


def _transpose_stack(t, k):
    nt = k // PAR
    out = nl.ndarray((PAR, nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nt):
        out[:, i, :] = nl.transpose(t[:, i * PAR:(i + 1) * PAR])
    return out


def _swiglu(g_bf16, u_bf16, wcol, dst):
    m, k = g_bf16.shape
    g = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       engine=nisa.engine.vector)
    u = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=MIDENG)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.scalar_tensor_tensor(dst=dst, data=sg, op0=nl.multiply, operand0=wcol,
                              op1=nl.multiply, operand1=u)


def _swiglu_sh(g_bf16, u_bf16, dst):
    """`Expert.forward`'s SwiGLU with no routing weight: the up branch clamped on both sides,
    the gate branch only from above, the product in float32, the result rounded to bfloat16."""
    m, k = g_bf16.shape
    g = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       engine=nisa.engine.vector)
    u = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=MIDENG)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.tensor_tensor(dst=dst, data1=sg, data2=u, op=nl.multiply,
                       engine=nisa.engine.vector)


def _flat_chunks(kblk, split):
    if not split:
        return [(0, kblk, 0, kblk * BPB // PAR)]
    ch = []
    kb0 = 0
    while kb0 < kblk:
        nk = PAR
        while nk > kblk - kb0:
            nk //= 2
        assert nk * BPB % PAR == 0
        ch.append((kb0, nk, PAR // nk, nk * BPB // PAR))
        kb0 += nk
    return ch


def _flat_plan(kblk, split):
    tiles = []
    if not split:
        nt = kblk * BPB // PAR
        for _ in range(nt):
            tiles.append([])
        for off in range(BPB):
            kb = 0
            while kb < kblk:
                f = off * kblk + kb
                t = f // PAR
                p = f - t * PAR
                ln = min(kblk - kb, PAR - p)
                tiles[t].append((p, off, kb, ln))
                kb += ln
        return tiles
    chs = _flat_chunks(kblk, True)
    for c in range(len(chs)):
        kb0 = chs[c][0]
        nk = chs[c][1]
        nof = chs[c][2]
        for a in range(chs[c][3]):
            segs = []
            for i in range(nof):
                segs.append((i * nk, a * nof + i, kb0, nk))
            tiles.append(segs)
    return tiles


def _assert_covered(segs, name):
    for t in range(len(segs)):
        cov = []
        for _ in range(PAR):
            cov.append(0)
        for si in range(len(segs[t])):
            p = segs[t][si][0]
            ln = segs[t][si][3]
            for i in range(p, p + ln):
                cov[i] += 1
        for i in range(PAR):
            assert cov[i] == 1, \
                "%s tile %d partition %d covered %d times, not once" % (name, t, i, cov[i])


def _scale_plan(segs):
    keys = []
    idx = []
    for t in range(len(segs)):
        key = []
        for si in range(len(segs[t])):
            key.append((segs[t][si][0], segs[t][si][2], segs[t][si][3]))
        key = tuple(key)
        if key not in keys:
            keys.append(key)
        idx.append(keys.index(key))
    return idx


CHK1 = _flat_chunks(KBLK1, CHSP1)
CHK2 = _flat_chunks(KBLK2, CHSP2)
SEG1 = _flat_plan(KBLK1, CHSP1)
SEG2 = _flat_plan(KBLK2, CHSP2)
_assert_covered(SEG1, "SEG1")
_assert_covered(SEG2, "SEG2")
SSC1 = _scale_plan(SEG1)
SSC2 = _scale_plan(SEG2)


def _flat_tr(fls, chunks):
    nch = len(chunks)
    nt = 0
    for c in range(nch):
        nt += chunks[c][3]
    out = nl.ndarray((PAR, 2 * nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    srcs = []
    for c in range(nch):
        flat = fls[c].reshape((TOK, 2, BPB * chunks[c][1]))
        for a in range(chunks[c][3]):
            for high in range(2):
                srcs.append(flat[:, high, PAR * a:PAR * a + PAR])
    ci = 0
    i = 0
    n = len(srcs)
    while i < n:
        b = min(TRB, n - i)
        tp = nl.ndarray((PAR, b, TOK), dtype=nl.bfloat16, buffer=nl.psum)
        for k in range(b):
            nisa.nc_transpose(dst=tp[:, k, :], data=srcs[i + k])
        nisa.tensor_copy(dst=out[:, i:i + b, :], src=tp, engine=TRENG[ci % len(TRENG)])
        ci += 1
        i += b
    return out


def _quant_layout(x, m, k, chunks):
    return _flat_tr(_act_quant(x, m, k, chunks), chunks)


def _fp4_dequant(q, sc):
    nisa.tensor_scalar(dst=q.view(nl.uint32), data=q.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=PAT_MASK2, engine=nisa.engine.vector)
    dq = q.view(nl.bfloat16)
    nisa.tensor_tensor(dst=dq, data1=dq, data2=sc, op=nl.multiply,
                       engine=nisa.engine.vector)
    return dq


def _dma_buf(nw, pdep):
    bf = []
    for _ in range(pdep):
        bf.append([nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf),
                   nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)])
    return bf


def _alloc_db(nbg, nw, nj, pdep):
    if nbg > 1:
        bf = []
        for _ in range(nj // nbg):
            bf.append(_dma_buf(nbg * nw, pdep))
        return bf
    bf = []
    for _ in range(nj):
        bf.append(_dma_buf(nw, pdep))
    return bf


def _gemm_fp4(ws, ss, j, kbytes, kblk, n, nw, nbg, segs, pid, alays, dbufs, outs, yacc,
              sceng, preng, ksp, pdep, d, lead=0, mid=None):
    ng = len(ws)
    widths = []
    for q in range(0, nw, NMAX):
        widths.append(min(NMAX, nw - q))
    nt = len(segs)
    nbt = n // nw
    assert nt % ksp == 0
    assert nbt % nbg == 0
    # The token-tile count is the caller's, not the static `GTOK`: the routed loop runs several
    # passes over disjoint buckets of the group plan, each with its own static tile count, so this
    # body is traced once per bucket. Every per-tile allocation, matmul and PSUM assertion below
    # is sized from `gtk` instead.
    gtk = len(alays)
    assert 1 <= gtk <= GTOK
    assert pdep > lead
    assert 4 * ng * nbg * gtk * nw <= 16384
    ntc = nt // ksp
    ntc = nt // ksp
    njob = ng * nbg * ksp
    nib = []
    pre = []
    for _ in range(pdep):
        gn = []
        gp = []
        for _ in range(njob):
            gn.append([nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf),
                       nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)])
            gp.append([nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf),
                       nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf)])
        nib.append(gn)
        pre.append(gp)
    plans = []
    for nb0 in range(0, nbt, nbg):
        jg = []
        jn = []
        jt = []
        ja = []
        for g in range(ng):
            for i in range(nbg):
                for kc in range(ksp):
                    jg.append(g)
                    jn.append((nb0 + i) * nw)
                    jt.append(kc * ntc)
                    ja.append(g * nbg + i)
        njob = len(jg)
        acc = []
        for _ in range(ng * nbg):
            gt = []
            for _ in range(gtk):
                ga = []
                for wi in range(len(widths)):
                    ga.append(nl.ndarray((TOK, widths[wi]), dtype=nl.float32, buffer=nl.psum))
                gt.append(ga)
            acc.append(gt)
        keys = []
        for pos in range(ntc):
            for q in range(njob):
                k = (jg[q], jn[q], pid[jt[q] + pos])
                if k not in keys:
                    keys.append(k)
        scb = []
        for _ in range(len(keys)):
            scb.append(nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf))
        sfl = []
        sbi = []
        filled = []
        for q in range(njob):
            sfl.append([])
            sbi.append([])
        for pos in range(ntc):
            for q in range(njob):
                k = (jg[q], jn[q], pid[jt[q] + pos])
                sbi[q].append(keys.index(k))
                sfl[q].append(k not in filled)
                if k not in filled:
                    filled.append(k)
        merge = (ng == 1 and nbg > 1)
        dsc = []
        dwt = []
        for q in range(njob):
            gs = []
            gw = []
            kc = q % ksp
            i = (q // ksp) % nbg
            for c in range(pdep):
                if merge:
                    w16 = dbufs[kc][c][1].view(nl.uint16)
                    gs.append(dbufs[kc][c][0][:, i * nw:(i + 1) * nw])
                    gw.append(w16[:, i * (nw // 2):(i + 1) * (nw // 2)])
                else:
                    gs.append(dbufs[q][c][0])
                    gw.append(dbufs[q][c][1].view(nl.uint16))
            dsc.append(gs)
            dwt.append(gw)
        plans.append((nb0, jg, jn, jt, ja, acc, scb, sfl, sbi, dsc, dwt, merge))
    ntot = len(plans) * ntc
    for ti in range(5 * (ntot + lead)):
        step = ti // 5
        st = ti % 5
        gpos = step - lead if st == 4 else step
        if gpos < 0 or gpos >= ntot:
            continue
        nbi = gpos // ntc
        pos = gpos % ntc
        nb0, jg, jn, jt, ja, acc, scb, sfl, sbi, dsc, dwt, merge = plans[nbi]
        t = pos
        c = gpos % pdep
        if st == 0:
            for u in range(ksp if merge else njob):
                nwd = nbg * nw if merge else nw
                jn0 = nb0 * nw if merge else jn[u]
                sg = segs[jt[u] + t]
                for si in range(len(sg)):
                    p = sg[si][0]
                    off = sg[si][1]
                    kb0 = sg[si][2]
                    ln = sg[si][3]
                    if sfl[u][pos]:
                        nisa.dma_copy(
                            dst=dbufs[u][c][0][p:p + ln, :],
                            src=ss[jg[u]].ap(pattern=[[kblk * n, 1], [n, ln], [1, nwd]],
                                             offset=kb0 * n + jn0 + d * kblk * n,
                                             scalar_offset=j,
                                             indirect_dim=0).squeeze_dim(0))
                    nisa.dma_copy(
                        dst=dbufs[u][c][1][p:p + ln, :],
                        src=ws[jg[u]].ap(pattern=[[kbytes * n, 1], [BPB * n, ln],
                                                  [1, nwd]],
                                         offset=(BPB * kb0 + off) * n + jn0
                                         + d * kbytes * n,
                                         scalar_offset=j,
                                         indirect_dim=0).squeeze_dim(0))
        elif st == 1:
            for q in range(njob):
                b16 = dwt[q][c]
                nisa.tensor_scalar(dst=nib[c][q][0].view(nl.uint16), data=b16,
                                   op0=nl.bitwise_and, operand0=PAIR_NIB_MASK,
                                   engine=nisa.engine.vector)
                nisa.tensor_scalar(dst=nib[c][q][1].view(nl.uint16), data=b16,
                                   op0=nl.right_shift, operand0=HI_NIB_SHIFT,
                                   op1=nl.bitwise_and, operand1=PAIR_NIB_MASK,
                                   engine=nisa.engine.vector)
        elif st == 2:
            for q in range(njob):
                if sfl[q][pos]:
                    nisa.tensor_scalar(dst=scb[sbi[q][pos]], data=dsc[q][c],
                                       op0=nl.multiply, operand0=SC_MUL,
                                       op1=nl.add, operand1=SC_ADD,
                                       engine=sceng[(t * njob + q) % len(sceng)])
        elif st == 3:
            for q in range(njob):
                for e in range(2):
                    nisa.tensor_scalar(dst=pre[c][q][e], data=nib[c][q][e],
                                       op0=nl.multiply, operand0=NIB_MUL,
                                       engine=preng[(2 * q + e) % len(preng)])
        else:
            wdq = []
            for q in range(njob):
                sc = scb[sbi[q][pos]].view(nl.bfloat16)
                wdq.append([_fp4_dequant(pre[c][q][0], sc),
                            _fp4_dequant(pre[c][q][1], sc)])
            for q in range(njob):
                for high in range(2):
                    for tt in range(gtk):
                        for wi in range(len(widths)):
                            qw = wi * NMAX
                            nisa.nc_matmul(
                                dst=acc[ja[q]][tt][wi],
                                stationary=alays[tt][:, 2 * (jt[q] + t) + high, :],
                                moving=wdq[q][high][:, qw:qw + widths[wi]],
                                accumulate=(t > 0 or high > 0 or jt[q] > 0))
        if st == 4 and pos == ntc - 1:
            di = 0
            for a in range(ng * nbg):
                for tt in range(gtk):
                    for wi in range(len(widths)):
                        qw = wi * NMAX
                        jn0 = (nb0 + a % nbg) * nw
                        sl = slice(jn0 + qw, jn0 + qw + widths[wi])
                        deng = DRNENG[di % len(DRNENG)]
                        di += 1
                        if yacc is None:
                            nisa.tensor_copy(dst=outs[a // nbg][tt][:, sl], src=acc[a][tt][wi],
                                             engine=deng)
                        else:
                            nisa.tensor_copy(dst=yacc[tt][:, sl], src=acc[a][tt][wi], engine=deng)
            if mid is not None:
                gs, us, wc, f2 = mid
                sl = slice(nb0 * nw, (nb0 + nbg) * nw)
                kw = nbg * nw
                tsl = []
                for tt in range(gtk):
                    tb = nl.ndarray((TOK, kw), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _swiglu(gs[tt][:, sl], us[tt][:, sl], wc[tt], tb)
                    tsl.append(tb)
                for tt in range(gtk):
                    _act_quant(tsl[tt], TOK, kw, [(0, kw // ABLK, 0, kw * BPB // (ABLK * PAR))],
                               dst=f2[tt], koff=nb0 * nw // ABLK)


def _route_col(j, base, weff):
    wcol = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wcol,
                  src=weff.ap(pattern=[[N_EXPERTS, TOK], [1, 1]], offset=base,
                              scalar_offset=j, indirect_dim=1))
    return wcol


def _expert_step(j, d, wcols, w1, s1, w3, s3, w2, s2, seg1, alay1, seg2, db, ys):
    gtk = len(alay1)
    g = []
    u = []
    for _ in range(gtk):
        g.append(nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf))
        u.append(nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf))
    fl2 = []
    for tt in range(gtk):
        rw = []
        for c in range(len(CHK2)):
            rw.append(nl.ndarray((TOK, 2, BPB, CHK2[c][1]), dtype=nl.bfloat16, buffer=nl.sbuf))
        fl2.append(rw)
    # **w1 and w3 stay two `_gemm_fp4` calls, measured.** Running them as one job pool (`ng = 2`,
    # `njob = 2 * KSP1`, one pipeline fill and drain per group instead of two) is PSUM-legal in the
    # narrow buckets -- the accumulator is `4 * ng * nbg * gtk * nw` bytes per partition, so `ng = 2`
    # at `nw = NW1` needs `gtk <= 2` -- and it compiles and is bit-identical, but it measures
    # **41.6724 ms against 40.1474**, i.e. -3.8%. Doubling the jobs per stage also doubles the live
    # PSUM accumulator in those buckets and halves the DMA rotation depth that the same SBUF can
    # carry (`PDEPF = 3` against `PDEP1 = 6`), and one of those costs more than the fill and drain
    # it saves. Do not re-try it without first finding SBUF for depth 6 at eight jobs.
    # **Pre-issuing the w2 pipeline's fill is also a no-op, measured.** `w3` -> `mid` -> `_flat_tr`
    # -> `w2` is the longest serialization inside a group and the one stretch with no weight DMA in
    # flight, and w2's weight and scale reads depend only on the expert index, so its whole pipeline
    # fill (`LEAD2 + 1` contraction steps of stage 0, bounded by the `PDEP2` rotating buffers) can
    # legally be issued here, before the w1 call. Implemented as a `warm`/`dmaonly` pair of
    # `_gemm_fp4` calls -- the DMA-only call emits exactly those transfers, the real call skips
    # re-emitting them -- it compiles, is bit-identical, and measures **40.1661 against 40.1783 /
    # 40.1535 for these same bytes without it**: the shipped mean to three decimal places.
    # So round 1's finding that the compiler already hoists stage-0 DMAs as far as the rotating
    # buffers allow extends *across* a `_gemm_fp4` call boundary as well, not just within one call.
    # The ~10 us `EVENT_SEMAPHORE` waits the profile counts on Vector and Scalar are therefore not
    # cold weight-DMA latency at a pipeline head. Whatever they are, program order does not move
    # them, and this is the third independent attempt to say so.
    _gemm_fp4([w1], [s1], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              db[0], [g], None, SCENG1, PRENG1, KSP1, PDEP1, d, LEAD1)
    _gemm_fp4([w3], [s3], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              db[1], [u], None, SCENG1, PRENG1, KSP1, PDEP1, d, LEAD1,
              mid=(g, u, wcols, fl2))

    lay2 = []
    for tt in range(gtk):
        lay2.append(_flat_tr(fl2[tt], CHK2))

    _gemm_fp4([w2], [s2], j, KB2, KBLK2, DIM, NW2, NBG2, seg2, SSC2, lay2,
              db[2], None, ys, SCENG2, PRENG2, KSP2, PDEP2, d, LEAD2)


# ------------------------------------------------------- the shared expert's fp8 machinery

def _scale_rows(s, nt, nbtot):
    """One (128, nbtot) bfloat16 tile per k-tile: `2 ** (e8m0_byte - 7)`, i.e. the block scale
    with `_fp8_dequant`'s leftover `2**120` already folded into its exponent.

    The replication across a k-block's 32 partitions is the DMA's, with stride 0 on the
    partition axis: one transfer reads scale row `kb` into 32 partitions at once. That is what
    `nl.broadcast_to` would do, except it can only source partition 0, which forces a (1, nbtot)
    landing tile and therefore a separate load, bit-build and copy per row -- 12 instructions per
    tile, each touching 8 to 20 elements, all of it per-instruction overhead.
    """
    row = s.shape[1]
    tiles = []
    for i in range(nt):
        raw = nl.ndarray((PAR, nbtot), dtype=nl.uint8, buffer=nl.sbuf)
        for r in range(PAR // ABLK):
            nisa.dma_copy(dst=raw[r * ABLK:(r + 1) * ABLK, :],
                          src=s.ap(pattern=[[0, ABLK], [1, nbtot]],
                                   offset=(i * (PAR // ABLK) + r) * row))
        out = nl.ndarray((PAR, nbtot), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out.view(nl.uint16), data=raw, op0=nl.multiply,
                           operand0=SC_MUL8, op1=nl.add, operand1=SC_ADD8,
                           engine=nisa.engine.vector)
        tiles.append(out)
    return tiles


def _fp8_dequant(byte, sc, m, nw, nb):
    """fp8 e4m3fn weight bytes -> scaled bfloat16, in four elementwise instructions.

    The bfloat16 bit pattern of code b is `(b & 0x80) << 8 | (b & 0x7f) << 4`, and that pattern
    reads as the value times `2**-120` -- uniformly, normals and subnormals alike, because
    bfloat16 subnormals reach far enough down to hold b's. `_scale_rows` carries the `2**120`
    back. The two mask/shift steps run on a uint32 view, so each lane carries two weights; the
    masks are doubled accordingly and `E4M3_SGN2` is written as a signed int32.
    """
    t = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=byte, op0=nl.multiply, operand0=BYTE_HI,
                       engine=nisa.engine.scalar)
    hi = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi.view(nl.uint32), data=t.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4, engine=nisa.engine.vector)
    bits = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=bits.view(nl.uint32), data=t.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=hi.view(nl.uint32))
    wdq = nl.ndarray((m, nw), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wdq.reshape((m, nb, ABLK)),
                       data1=bits.view(nl.bfloat16).reshape((m, nb, ABLK)),
                       data2=sc.reshape((m, nb, 1)).broadcast(2, ABLK),
                       op=nl.multiply, engine=nisa.engine.vector)
    return wdq


def _predq(w, s, k, n, dst):
    """Decode one whole fp8 weight matrix into bfloat16 in HBM, once per invocation."""
    nt = k // PAR
    sct = _scale_rows(s, nt, n // ABLK)
    for i in range(nt):
        for n0 in range(0, n, PREDQ_W):
            nw = min(PREDQ_W, n - n0)
            byte = nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=byte, src=w[i * PAR:(i + 1) * PAR, n0:n0 + nw])
            wdq = _fp8_dequant(byte, sct[i][:, n0 // ABLK:(n0 + nw) // ABLK],
                               PAR, nw, nw // ABLK)
            nisa.dma_copy(dst=dst[i * PAR:(i + 1) * PAR, n0:n0 + nw], src=wdq)


def _gemm_bf16(ats, w, k, n):
    """(TOK, k) x (k, n) for len(ats) token tiles against bfloat16 weights read from HBM.

    One weight tile serves every token tile in `ats`, so the weight traffic is 1/len(ats) of the
    per-token-tile form -- 315 MB over the whole shared expert at GSH = 4 rather than 1.26 GB.
    At most len(ats) PSUM accumulators of NMAX float32 are live, which is what keeps this inside
    the 8 banks whatever `n` is.

    The weight tiles rotate over WDEP buffers with the DMA running WLEAD k-tiles ahead of the
    matmuls that read them.
    """
    nt = k // PAR
    nj = len(ats)
    outs = []
    for _ in range(nj):
        outs.append(nl.ndarray((TOK, n), dtype=nl.bfloat16, buffer=nl.sbuf))
    for n0 in range(0, n, NMAX):
        nw = min(NMAX, n - n0)
        accs = []
        for _ in range(nj):
            accs.append(nl.ndarray((TOK, nw), dtype=nl.float32, buffer=nl.psum))
        wb = []
        for _ in range(WDEP):
            wb.append(nl.ndarray((PAR, nw), dtype=nl.bfloat16, buffer=nl.sbuf))
        for i in range(nt + WLEAD):
            if i < nt:
                nisa.dma_copy(dst=wb[i % WDEP], src=w[i * PAR:(i + 1) * PAR, n0:n0 + nw])
            m = i - WLEAD
            if 0 <= m < nt:
                for jj in range(nj):
                    nisa.nc_matmul(dst=accs[jj], stationary=ats[jj][:, m, :],
                                   moving=wb[m % WDEP], accumulate=(m > 0))
        for jj in range(nj):
            nisa.tensor_copy(dst=outs[jj][:, n0:n0 + nw], src=accs[jj],
                             engine=nisa.engine.scalar)
    return outs


TOKENS = 8192
NTILE = TOKENS // TOK

P1U = 2
assert NTILE % P1U == 0

GRP = GTOK * TOK
# The static worst case, and the capacity everything downstream is sized from: a (token, expert)
# pair count of TOKENS * TOPK spread over CH experts costs at most one part-full group each, so the
# plan can need `CH + (TOKENS * TOPK - CH) // GRP` groups, and that bound is attainable rather than
# theoretical. NPG rounds it up to a whole number of `CORES` so the plan divides evenly between the
# two physical cores.
#
# How many of those groups *run* is a device value (see `nbreg`), but the provisioning is static:
# any group the plan can reach has a row in `pk3`/`eidh` and a landing place in `part`. A
# provisioned group the routing did not fill reads expert 0 out of `eidh` and token index BIG out
# of `pk3`, which the body clamps into a trash tile nothing downstream reads -- so running one is
# harmless and skipping one is bit-identical.
NPGT = CH + (TOKENS * TOPK - CH) // GRP
CORES = 2                                  # LNC=2: the rank's logical core is two physical cores
NPGC = (NPGT + CORES - 1) // CORES         # expert groups per physical core, provisioned
NPG = CORES * NPGC
assert NPG >= NPGT

# **The token-tile buckets.** A group is one expert's GRP slots and the compaction fills them in
# order, so group `k` of expert `e` holds `ceil(min(GRP, tot[e] - k * GRP) / TOK)` of its GTOK token
# tiles and the rest hold nothing but the padding index. Running those tiles is exact -- their
# routing weight is 0.0 -- and it is also most of the token slots the loop executes.
#
# So the routed loop runs one pass per bucket below, each with its own *static* tile count, over a
# disjoint window of a group ordering built on device: pass p takes every group whose tile count is
# in `(GPASS[p + 1], GPASS[p]]`, and a group no token landed in is in no bucket and never runs.
# Within a pass every group has the same static tile count, both physical cores have the same trip
# count, and nothing nests -- the passes are sequential register-bounded loops, which is the shape
# this toolchain's dynamic-trip-count loops are documented in.
#
# The schedule itself is a geometric ladder off GTOK, not a fit to any routing: a group's tile
# count is in [1, GTOK] for every input, and halving is what makes every bucket's worst-case waste
# a bounded fraction of the group it runs. A host replay of the recorded router prices `(GTOK, 2,
# 1)` against `(GTOK, 3, 2)`, `(GTOK, 3, 1)` and a four-bucket `(GTOK, 3, 2, 1)` at within 0.5% of
# each other, so the ladder is not a tuned number -- the schedule is flat and any of them would do.
GPASS = (GTOK, GTOK // 2, 1)
NPASS = len(GPASS)
assert GPASS[0] == GTOK and GPASS[-1] >= 1
# One nested `def` per pass lives in the kernel, so the count is wired into its body.
assert NPASS == 3
assert all(GPASS[i] > GPASS[i + 1] for i in range(NPASS - 1))
# `(GPASS[1], GPASS[2], ..., 0)`: pass p takes a group whose tile count is strictly above this.
GTHR = tuple(list(GPASS[1:]) + [0])

# **Groups per back edge, per pass.** The three passes are three different regimes, and only one of
# them is short of work to overlap: the widest bucket runs `GTOK` token tiles against one weight
# decode and keeps the tensor engine near its throttled ceiling, while the 1-tile bucket runs a
# quarter of the matmuls against the same whole-expert decode and leaves tensor idle at ~36% with no
# engine above ~60% and the union of all four at ~90% -- i.e. it is a dependency chain with bubbles
# in it, not a saturated engine. Two *independent* groups in one loop body is what fills those: the
# experts differ, so the two decode chains share nothing until the final scatter, and the scheduler
# can issue one group's vector work under the other's scalar work.
#
# It is affordable only in the narrow buckets. The per-group working set is dominated by its token
# tiles -- `alay1`, `ys`, `g`/`u` are all `gtk` tiles wide -- so unrolling the 1-tile pass costs what
# a single 2-tile group costs, while unrolling the widest pass would put `2 * GTOK` tiles in flight,
# which is what an earlier whole-loop `UNROLL = 2` (before the buckets existed, when every group ran
# at `GTOK` tiles) measured negative on.
UNROLLQ = (1, 1, 1)
assert len(UNROLLQ) == NPASS and all(NPGC % u == 0 for u in UNROLLQ)
# The stride between one core's consecutive groups is `CORES`; a back edge consumes `CORES * unroll`
# positions of the ordering across both cores.
GSTEPQ = tuple(CORES * u for u in UNROLLQ)
# The group ordering's length. Every pass's window is padded up to a whole number of its own
# `GSTEPQ` and is at least that wide, so a clamped-to-one trip count can only read padding and never
# the next pass's groups -- which is what keeps the skip bit-identical rather than double-counting a
# group; and a pass whose window start had to be clamped still cannot run past the end, since a
# pass reads at most `GSTEPQ[q] * (NPGC // UNROLLQ[q]) == CORES * NPGC` positions from its start.
GOFFMAX = NPG + sum(GSTEPQ)
GORDN = GOFFMAX + CORES * NPGC
SHT = NTILE // CORES                       # token tiles of the shared expert per physical core
assert SHT % GSH == 0
BIG = 1.0e7

# `pk3` holds one GRP-slot landing table per group plus one trash row that out-of-range slots
# land in. Its row count is rounded up so that the memset below is a whole number of
# (TOK x PKW)-shaped transfers.
PKW = 64
PKSTEP = TOK * PKW // GRP
PKR = ((NPG + 1 + PKSTEP - 1) // PKSTEP) * PKSTEP
assert TOK * PKW % GRP == 0 and PKR >= NPG + 1


def _tile3(t3d, t, n1, n2, tt=0):
    # `scalar_offset` must be an `NkiTensor` or a `VirtualRegister` -- a Python int is rejected
    # outright -- so a trace-time tile index folds into the access pattern's own `offset` instead.
    # `scalar_offset` is scaled by the stride of the axis `indirect_dim` names, which is `n1 * n2`
    # here, so the two forms address the same tile.
    if isinstance(t, int):
        return t3d.ap(pattern=[[n2, n1], [1, n2]], offset=(tt + t) * n1 * n2)
    return t3d.ap(pattern=[[n1 * n2, 1], [n2, n1], [1, n2]], offset=tt * n1 * n2,
                  scalar_offset=t, indirect_dim=0).squeeze_dim(0)


def _pack_cols(pk3, g, col, ncol=GTOK):
    # `g` is the group's row of `pk3`, and it is a (1, 1) int32 SBUF tensor rather than a Python
    # int: `scalar_offset` scales whatever it is handed by the stride of the axis `indirect_dim`
    # names, which is `GRP * 2` here, so the same expression serves a traced group index and a
    # device-computed one.
    dst = nl.ndarray((TOK, ncol), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dst,
                  src=pk3.ap(pattern=[[GRP * 2, 1], [2, TOK], [TOK * 2, ncol]],
                             offset=col,
                             scalar_offset=g, indirect_dim=0).squeeze_dim(0))
    return dst


# ------------------------------------------------------- the routed expert group's body
#
# At module level, not nested in the kernel: an inner function may only be used as a
# `fori_loop`/`while_loop` body argument, so the three passes below are three one-line nested defs
# that each call this with their own static token-tile count.
def _grp_body(gtk, gptr, ctx, unroll):
    (pk3, eidh, gordh, aqf, w1, s1, w3, s3, w2, s2, db, partf, strash, tbase) = ctx
    # **The advance stays at the end of each group's body, measured.** Hoisting it -- and all
    # `unroll` groups' `gordh`/`eidh` lookups -- above the work is what an unrolled body needs to
    # overlap at all, since these engines issue in order and with the advance written last the next
    # group's first lookup cannot issue until this group's whole instruction stream has issued on the
    # engine that owns the advance. But at `unroll = 1` there is nothing to overlap *within* the
    # body, and the hoist across the `fori_loop`'s back edge measures 40.1854 against 40.1474 -- a
    # no-op inside this host's 23 us run-to-run spread, because the back edge serializes anyway.
    # And at `unroll = 2` the hoist recovers only 0.147 ms of the unroll's own 0.828 ms cost
    # (40.9757 -> 40.8284), so the serialization was not what made two groups per body lose.
    for u in range(unroll):
        gv = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gv, src=gordh.ap(pattern=[[1, 1], [1, 1]], offset=0,
                                           scalar_offset=gptr, indirect_dim=0))
        e1 = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e1, src=eidh.ap(pattern=[[1, 1], [1, 1]], offset=0,
                                          scalar_offset=gv, indirect_dim=0))
        ys = []
        wcols = []
        alay1 = []
        sidx = []
        pkr = _pack_cols(pk3, gv, 0, gtk)
        pkw = _pack_cols(pk3, gv, 1, gtk)
        for tt in range(gtk):
            row = pkr[:, tt:tt + 1]
            wcols.append(pkw[:, tt:tt + 1])
            # This core's rows of `part`, with an empty slot's BIG token index landing on this
            # core's trash tile instead of off the end of the buffer.
            sidx.append(nl.copy(nl.minimum(nl.add(row, float(tbase * TOK)), strash),
                                dtype=nl.uint32))
            gi = nl.copy(nl.minimum(row, float(TOKENS - 1)), dtype=nl.uint32)
            fls = []
            for c in range(len(CHK1)):
                lc = 2 * BPB * CHK1[c][1]
                fl = nl.ndarray((TOK, 2, BPB, CHK1[c][1]), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=fl, src=aqf[c].ap(pattern=[[lc, TOK], [1, lc]],
                                                    vector_offset=gi, indirect_dim=0))
                fls.append(fl)
            ys.append(nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf))
            alay1.append(_flat_tr(fls, CHK1))

        _expert_step(e1, 0, wcols, w1, s1, w3, s3, w2, s2, SEG1, alay1, SEG2, db, ys)

        for tt in range(gtk):
            nisa.dma_compute(
                dst=partf.ap(pattern=[[DIM, TOK], [1, DIM]],
                             vector_offset=sidx[tt], indirect_dim=0),
                srcs=[partf.ap(pattern=[[DIM, TOK], [1, DIM]],
                               vector_offset=sidx[tt], indirect_dim=0), ys[tt]],
                reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)

        nisa.tensor_scalar(dst=gptr, data=gptr, op0=nl.add, operand0=CORES)
        # **No `core_barrier` on the back edge, deliberately.** It is the form the reference
        # register-bounded loop ships, and it protects nothing here: `gptr` is core-private; the
        # two cores' trip counts are identical for any routing by construction, both deriving them
        # from the same device values, so there is no divergence to guard; `pk3`/`eidh`/`nth`/
        # `gordh` are read-only in this loop and already barriered before it; and the two cores
        # scatter into disjoint halves of `part` (`tbase = pid * PT`), which the barrier before the
        # seam is what makes mutually readable. Measured both ways: with the barrier the kernel runs
        # 40.281 / 40.289 / 40.324 ms, without it 40.169 / 40.145, numerics bit-identical either
        # way and no hang. Putting it back costs ~0.14 ms and buys nothing.


@nki.jit
def kernel(hidden, norm_w, gate_wt, gate_b, w1, s1, w3, s3, w2, s2,
           sw1, ss1, sw2, ss2, sw3, ss3, replica_group):
    # The module's output: one bfloat16 round of the summed float32 partials, which is the
    # rounding the reference does and the only one the four-way reduce adds.
    out = nl.ndarray((TOKENS, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # The two physical cores' float32 partials, one half each, and one **trash token tile** per
    # core on the end of its half.
    #
    # The trash tile is what makes every indirect access in this kernel in-bounds. The routed
    # experts land in `part` through `nisa.dma_compute`'s indirect read-modify-write scatter, and
    # a group no token routes to carries token index BIG, which the submodule left to
    # `oob_mode.skip`. On this toolchain that is not safe: the same scatter, at the same indices,
    # either skips silently or reports "scatter/gather (indirect memory copy via scalar DGE)
    # out-of-bound access" and wedges the DMA engine with `NRT_EXEC_HW_ERR_DMA_ABORT` depending on
    # what else is in the kernel -- measured four ways, see ASSEMBLY.md. Clamping the index to a
    # tile nobody reads removes the question: nothing here is ever out of range, and `oob_mode`
    # is left on only as a backstop.
    #
    # It is also an **IO tensor**, returned and ignored, because a scatter of this shape into
    # `private_hbm` or into named `shared_hbm` scratch fails the same way.
    PT = NTILE + 1
    part = nl.ndarray((CORES * PT, TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    partf = part.reshape((CORES * PT * TOK, DIM))
    # name= on the collective's src/dst, or DRAM allocation fails with NCC_IBIR440; both
    # shared_hbm, because a collective may not mix buffer kinds; both scratch, because a
    # collective may not read or write an IO tensor (NCC_INLA001).
    cc_src = nl.ndarray((TOKENS, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOKENS, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")

    # nl.program_id is a trace-time constant here -- the launch grid is traced once per physical
    # core and the NEFF carries both programs -- so *which* work each core takes is chosen with
    # ordinary Python and no predication. Only the routed loop's trip count is dynamic, and it is
    # the same value on both cores (see `nbreg` below). This core's first expert group is `pid`;
    # it then takes every CORES-th group from there.
    pid = nl.program_id(0)
    tbase = pid * PT                        # this core's first row of `part`, in token tiles
    shlo = pid * SHT                        # this core's first shared-expert token tile

    # The quantized activation and the routing weights are **shared** HBM, not private, because
    # the prologue that fills them is split between the two physical cores: core `pid` runs
    # `t1` over its own 32 token tiles only, and both cores then read all 64 tiles' worth. The
    # prologue was the one piece of this kernel still run twice for one copy of the answer --
    # `ASSEMBLY.md` prices it at 5.71 ms standalone -- and `nisa.core_barrier` is the documented
    # way to make one core's `dma_copy` into disjoint rows of a shared-HBM tensor readable on the
    # other. `hqt` stays private: the shared expert reads exactly the token tiles its own core
    # wrote, because `shlo == pid * SHT` is the same split.
    aql = []
    aqf = []
    for c in range(len(CHK1)):
        aql.append(nl.ndarray((NTILE, TOK, 2 * BPB * CHK1[c][1]), dtype=nl.bfloat16,
                              buffer=nl.shared_hbm, name=f"aql{c}"))
        aqf.append(aql[c].reshape((TOKENS, 2 * BPB * CHK1[c][1])))
    # The same quantized activation the fp4 experts consume, transposed for the shared expert's
    # fp8 GEMMs. The prologue already builds both the quantization and a transpose, so this
    # costs one elementwise instruction and one store per token tile; the shared expert then
    # needs no activation work of its own.
    hqt = nl.ndarray((NTILE, PAR, NT1 * TOK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    weffL = nl.ndarray((NTILE, TOK, CH), dtype=nl.float32, buffer=nl.shared_hbm, name="weffL")
    # One row per core of the per-expert (token, expert) pair counts, summed after the barrier:
    # the compaction's group plan has to be the same on both cores, so it is built from the sum of
    # the two cores' halves rather than from either core's own count.
    totx = nl.ndarray((CORES, CH), dtype=nl.float32, buffer=nl.shared_hbm, name="totx")
    pk3 = nl.ndarray((PKR, GRP, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="pk3")
    eidh = nl.ndarray((NPG + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    # How many token tiles each provisioned group actually holds, and the order the passes visit
    # the groups in. Both are built on device from the compaction's own `tot`; both are sized from
    # the static capacity, like `eidh`. A group the routing did not fill keeps `nth = 0`, which puts
    # it in no bucket.
    nth = nl.ndarray((NPG + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    gordh = nl.ndarray((GORDN, 1), dtype=nl.float32, buffer=nl.private_hbm)
    d1 = nl.ndarray((DIM, INTER_L), dtype=nl.bfloat16, buffer=nl.private_hbm)
    d3 = nl.ndarray((DIM, INTER_L), dtype=nl.bfloat16, buffer=nl.private_hbm)
    d2 = nl.ndarray((INTER_L, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)

    dba = _alloc_db(NBG1, NW1, NBG1 * KSP1, PDEP1)
    dbb = _alloc_db(NBG1, NW1, NBG1 * KSP1, PDEP1)
    db2 = _alloc_db(NBG2, NW2, NBG2 * KSP2, PDEP2)
    db = (dba, dbb, db2)

    _predq(sw1, ss1, DIM, INTER_L, d1)
    _predq(sw3, ss3, DIM, INTER_L, d3)
    _predq(sw2, ss2, INTER_L, DIM, d2)

    zt = nl.ndarray((TOK, PKW, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(zt[:, :, 0], BIG)
    nisa.memset(zt[:, :, 1], 0.0)
    pkflat = pk3.reshape((PKR * GRP, 2))
    PKC = (PKR * GRP) // (TOK * PKW)
    assert PKC * TOK * PKW == PKR * GRP
    for i in range(PKC):
        nisa.dma_copy(dst=pkflat.ap(pattern=[[PKW * 2, TOK], [2, PKW], [1, 2]],
                                    offset=i * TOK * PKW * 2),
                      src=zt)
    ez = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ez, 0.0)
    for i in range(0, NPG + 1, CH):
        nz = min(CH, NPG + 1 - i)
        nisa.dma_copy(dst=eidh[i:i + nz, :], src=ez[0:nz, :])
        nisa.dma_copy(dst=nth[i:i + nz, :], src=ez[0:nz, :])
    # Every position of the ordering that no group is scattered into names the plan's trash row:
    # `pk3` row NPG carries a routing weight of exactly 0.0 on every slot, so a pass that reads one
    # scatters an exact +0.0 and the answer does not move.
    gz = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(gz, float(NPG))
    for i in range(0, GORDN, CH):
        nz = min(CH, GORDN - i)
        nisa.dma_copy(dst=gordh[i:i + nz, :], src=gz[0:nz, :])

    tot = nl.zeros((1, CH), dtype=nl.float32, buffer=nl.sbuf)

    # Two constant rows the router's exact top-`TOPK` needs, built once rather than per token tile.
    #
    # `rkey[j] = N_EXPERTS - j`, so it is strictly positive and strictly *decreasing* in the column
    # index: taking the largest `rkey` among a set of columns takes the lowest-numbered column, and
    # zero (what a column outside the set carries) can never win. `ohk[i] = TOPK - 1 - i` turns a
    # count into a one-hot selector over the eight lanes `nisa.max8` returns.
    rit = nl.ndarray((1, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(rit, pattern=[[1, N_EXPERTS]], channel_multiplier=0)
    rkey = nl.ndarray((1, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    rkey[...] = nl.multiply(nl.add(rit, -float(N_EXPERTS)), -1.0)
    # Partition-broadcast once, not per token tile: `nl.broadcast_to` over the partition axis is a
    # real materialisation, so a constant row broadcast inside a 32-iteration unrolled loop is 31
    # wasted Vector passes. Same for the router bias, which is also loop-invariant.
    rkb = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    rkb[...] = nl.broadcast_to(rkey, (TOK, N_EXPERTS))
    gbb = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    gbb[...] = nl.broadcast_to(nl.load(gate_b, dtype=nl.float32), (TOK, N_EXPERTS))
    oit = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(oit, pattern=[[1, 8]], channel_multiplier=0)
    ohk = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    ohk[...] = nl.multiply(nl.add(oit, -float(TOPK - 1)), -1.0)

    # The norm's gain, loaded and partition-broadcast **once** rather than per token tile. It does
    # not depend on `t`, and the broadcast is a real `(TOK, DIM)` materialisation -- one full
    # Vector pass over 5120 elements per partition -- so hoisting it out of a 32-iteration
    # unrolled loop removes 31 of them. Held in bfloat16, which is the dtype the recorded tensor
    # already has, so widening it inside the multiply below is exact.
    gwb = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    gwb[...] = nl.broadcast_to(nl.load(norm_w, dtype=nl.bfloat16), (TOK, DIM))

    # **A plain Python loop, not `nl.fori_loop`** -- same reason as the shared-expert loop below:
    # constant bounds, but a `fori_loop` back edge still costs an 8-step ordered cross-engine
    # barrier per iteration, and this loop's are 0.561 ms of Vector+Scalar waiting.
    assert SHT % P1U == 0
    for t in range(shlo, shlo + SHT, P1U):
        for s in range(P1U):
            xb = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=_tile3(hidden, t, TOK, DIM, s))

            # RMSNorm, at three Vector passes over `(TOK, DIM)` instead of seven. The prologue is
            # the most Vector-saturated region in the kernel (74.5% busy on the profile), and four
            # of the seven passes it used to spend here were bookkeeping rather than arithmetic:
            #
            #  * the `float32` widening of `xb` is not needed. `nl.multiply(xb, xb, float32)` is
            #    the same exact product -- a bfloat16 square has 16 significant bits and fits
            #    float32 without rounding -- so squaring the loaded tile directly is bit-identical
            #    to squaring a widened copy of it, one pass cheaper;
            #  * `xb * rsqrt(var)` is `nisa.activation`'s own `data * scale` with a per-partition
            #    scale vector, free of charge on the **Scalar** engine, which this region has
            #    spare. The Scalar engine's math is float32, so this too is bit-identical;
            #  * the gain's partition broadcast is hoisted out of the loop (see `gwb`);
            #  * and the bfloat16 round is the gain multiply's own output dtype rather than a
            #    second pass. This one is *not* bit-identical: it rounds float32 -> bfloat16 once
            #    where the old form rounded to float32 and then to bfloat16, so an element landing
            #    exactly on a bfloat16 tie can differ by one ulp. Single rounding is the more
            #    accurate of the two.
            # The squares are written into `hf` and then overwritten by the scaled copy: the
            # Scalar engine's free-axis reduction is what the first activation is actually for,
            # and its output tile is a by-product nothing reads, so it does not need a buffer of
            # its own.
            hf = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=hf, op=nl.square, data=xb, reduce_op=nl.add, reduce_res=ss,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            var = nl.multiply(ss, 1.0 / DIM)
            nisa.activation(dst=hf, op=nl.copy, data=xb,
                            scale=nl.rsqrt(nl.add(var, NORM_EPS)))
            h = nl.multiply(hf, gwb, dtype=nl.bfloat16)
            hq = nl.ndarray((TOK, DIM // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
            aq = _act_quant(h, TOK, DIM, CHK1, natout=hq)
            for c in range(len(CHK1)):
                nisa.dma_copy(dst=_tile3(aql[c], t, TOK, 2 * BPB * CHK1[c][1], s),
                              src=aq[c].reshape((TOK, 2 * BPB * CHK1[c][1])))
            nisa.dma_copy(dst=_tile3(hqt, t, PAR, NT1 * TOK, s),
                          src=_transpose_stack(hq.reshape((TOK, DIM)), DIM))

            h_t = _transpose_stack(h, DIM)
            logits = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.psum)
            for i in range(DIM // PAR):
                nisa.nc_matmul(dst=logits, stationary=h_t[:, i, :],
                               moving=nl.load(gate_wt[i * PAR:(i + 1) * PAR, :]),
                               accumulate=(i > 0))
            # `softplus` is evaluated on a capped argument and then floored at the argument itself.
            # For `lg <= SP_CAP` the cap is the identity, and `softplus(x) > x` holds for every real
            # x -- and rounding a value strictly above the float32 `x` can never land below it --
            # so the floor is the identity too and this is bit-for-bit the plain `softplus(lg)` it
            # replaces. For `lg > SP_CAP` the cap is what stops `exp` overflowing to `inf`: at that
            # point `log1p(exp(-x)) < 2.1e-9` is below half an ulp of x, so `softplus(x) == x`
            # exactly in float32 and the floor returns the right answer rather than a saturated one.
            lg = nl.copy(logits, dtype=nl.float32)
            scores = nl.sqrt(nl.maximum(nl.softplus(nl.minimum(lg, SP_CAP)), lg))

            sel = nl.add(scores, gbb)
            # Exactly `TOPK` experts per token, for any input. `nl.greater_equal` against the
            # TOPK'th largest value is *not* that: any column tying it is admitted too, so a tie
            # selects more than TOPK experts, adds SwiGLU output that should not be there, and
            # inflates `total` so the renormaliser scales the legitimate TOPK down with it. The
            # tie is reachable, not theoretical -- `softplus` underflows to exactly 0.0 for
            # strongly negative logits, so every such expert has `sel == gate_b` and any repeated
            # bias value ties -- and on an all-zero input every one of the 384 ties.
            #
            # Ties break toward the lower *column*, which is the lower global expert id only
            # because the router's columns are a permutation of the global order that the kernel
            # cannot see (one NEFF serves all four ranks and no input names the rank), so an
            # ascending-global-id tie-break is not expressible here. It costs nothing on the tie
            # that is actually reachable: a column tied at `scores == 0.0` carries routing weight
            # exactly 0.0, so which of them is kept moves neither the output nor `tot`.
            #
            # `nisa.max8` gives the eight largest *values* of the multiset, so `kth` is the TOPK'th
            # largest and `#{sel > kth} <= TOPK - 1 <= #{sel >= kth}`. Split the selection at it:
            # every column strictly above `kth` is in, and of the `>= 1` columns that equal it
            # exactly, the lowest-numbered `TOPK - ngt` are in.
            #
            # `cand` is `rkey` on the tied columns and 0 elsewhere, so a second `nisa.max8` returns
            # the tied columns' keys in descending key order -- i.e. the tied column *indices* in
            # ascending order, at most eight of them, which covers the at most `TOPK` we may need.
            # `thr` picks lane `TOPK - 1 - ngt` out of those eight by a one-hot inner product
            # (a per-partition dynamic lane index is not addressable, a masked reduce is), and
            # `cand >= thr` is then `column <= that column`, which admits exactly `TOPK - ngt` of
            # the tied columns. A non-tied column carries `cand = 0 < 1 <= thr` and is never
            # admitted twice, so the two terms are disjoint and `keep` sums to exactly `TOPK`.
            #
            # `nisa.tensor_scalar(data=sel, op0=nl.greater, operand0=kc, engine=scalar)` would take
            # the threshold as a per-partition vector operand and run the comparison on the engine
            # this region has spare, instead of broadcasting `kth` across 384 columns first. The
            # compiler rejects it: **`NCC_IBIR444` "Invalid InstTensorScalarPtr ... on Activation
            # Engine, op0: is_gt"** -- comparison ALU ops are Vector-only in this instruction, for
            # all that `tensor_scalar`'s own documentation offers Vector, Scalar or GpSimd. So the
            # comparisons stay as they are and only the loop-invariant broadcasts are hoisted.
            kth = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
            nisa.max8(dst=kth, src=sel)
            kb = nl.broadcast_to(nl.copy(kth[:, TOPK - 1:TOPK]), (TOK, N_EXPERTS))
            gt = nl.greater(sel, kb, dtype=nl.float32)
            cand = nl.multiply(nl.equal(sel, kb, dtype=nl.float32), rkb)
            tk = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
            nisa.max8(dst=tk, src=cand)
            ngt = nl.sum(gt, axis=1, keepdims=True)
            onh = nl.equal(nl.broadcast_to(ngt, (TOK, 8)),
                           nl.broadcast_to(ohk, (TOK, 8)), dtype=nl.float32)
            thr = nl.sum(nl.multiply(tk, onh), axis=1, keepdims=True)
            keep = nl.add(gt, nl.greater_equal(cand, nl.broadcast_to(thr, (TOK, N_EXPERTS)),
                                               dtype=nl.float32))
            picked = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
            picked[...] = nl.multiply(scores, keep)
            total = nl.add(nl.sum(picked, axis=1, keepdims=True), SUM_EPS)
            inv = nl.multiply(nl.reciprocal(total), ROUTE_SCALE)
            wl = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
            wl[...] = nl.multiply(picked[:, 0:CH], nl.broadcast_to(inv, (TOK, CH)))
            nisa.dma_copy(dst=_tile3(weffL, t, TOK, CH, s), src=wl)

            kf = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
            kf[...] = nl.greater(wl, 0.0)
            c = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(dst=c, op=nl.add, data=kf)
            tot[...] = nl.add(tot, c)


    # Half of this rank's quarter of the shared expert: this core takes token tiles
    # `shlo .. shlo + SHT - 1`, GSH of them at a time. It *writes* its half of `part` rather than
    # adding into it, which is what makes the zero-fill those tiles would otherwise need free:
    # the routed experts scatter-add on top of it afterwards. The token tiles this core does not
    # own are zeroed below, so the two halves of `part` sum to one whole shared expert.
    #
    # **A plain Python loop, not `nl.fori_loop`.** Its bounds are compile-time constants (`shlo` is
    # `pid * SHT` and `pid` is a trace-time constant), and the profile says a `fori_loop` back edge
    # is not free even then: the compiler emits an 8-step *ordered* cross-engine token-ring barrier
    # on every iteration of every `fori_loop` region, and this loop's 8 of them cost 1.605 ms of
    # Vector+Scalar waiting on semaphore `$S[2]` -- 12% of all such waiting in the kernel -- plus
    # most of its own entry/resume block. A Python loop has no back edge and so pays none of it.
    # The price is eight traced copies of this body instead of one.
    yz = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(yz, 0.0)
    for t in range(SHT):
        nisa.dma_copy(dst=part[tbase + (1 - pid) * SHT + t], src=yz)
    nisa.dma_copy(dst=part[tbase + NTILE], src=yz)

    assert SHT % GSH == 0
    for tb in range(shlo, shlo + SHT, GSH):
        ats = []
        for j in range(GSH):
            a = nl.ndarray((PAR, NT1, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=_tile3(hqt, tb, PAR, NT1 * TOK, j))
            ats.append(a)
        gs = _gemm_bf16(ats, d1, DIM, INTER_L)
        us = _gemm_bf16(ats, d3, DIM, INTER_L)
        tts = []
        for j in range(GSH):
            tb16 = nl.ndarray((TOK, INTER_L), dtype=nl.bfloat16, buffer=nl.sbuf)
            _swiglu_sh(gs[j], us[j], tb16)
            tq = nl.ndarray((TOK, INTER_L // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
            # No chunks: the shared expert's w2 GEMM wants the plain layout, not the fp4 one.
            _act_quant(tb16, TOK, INTER_L, [], natout=tq)
            tts.append(_transpose_stack(tq.reshape((TOK, INTER_L)), INTER_L))
        ys = _gemm_bf16(tts, d2, INTER_L, DIM)
        for j in range(GSH):
            yf = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yf, src=ys[j], engine=nisa.engine.scalar)
            nisa.dma_copy(dst=_tile3(part, tb, TOK, DIM, j + tbase), src=yf)

    # The prologue's rejoin. Placed after the shared expert rather than straight after `t1` so
    # that whichever core gets there first spends the wait on its own shared-expert work.
    #
    # Hoisting it above the shared-expert loop was measured: **39.7914 against 39.8041, a no-op.**
    # The idea was that the barrier is the hard ordering point keeping the compaction -- 2.228 ms
    # of which the profile says 62% is all-engines-idle, a latency-bound chain over a running
    # prefix that cannot fill its own stalls -- from being scheduled against anything, so putting
    # it and the shared expert in one dependency-free scheduling region would let the list
    # scheduler issue the shared expert's GEMMs into the compaction's waits. It does not: whatever
    # the compaction is waiting on, the scheduler will not cover it with later independent work.
    nisa.dma_copy(dst=totx[pid:pid + 1, :], src=tot)
    for c in range(len(CHK1)):
        nisa.core_barrier(data=aql[c], cores=(0, 1))
    nisa.core_barrier(data=weffL, cores=(0, 1))
    nisa.core_barrier(data=totx, cores=(0, 1))
    # Both cores memset the whole of `pk3` with the same bytes, which is idempotent; this barrier
    # is what puts every one of those writes before either core's first scatter into it.
    nisa.core_barrier(data=pk3, cores=(0, 1))
    tot[...] = nl.add(nl.load(totx[0:1, :]), nl.load(totx[1:2, :]))

    ngr = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    ngr[...] = nl.floor(nl.multiply(nl.add(tot, GRP - 1.0), 1.0 / GRP))
    cum = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=cum, src=ngr, op0=nl.multiply, imm0=1.0, op1=nl.add)
    gst = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    gst[...] = nl.subtract(cum, ngr)
    base = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    base[...] = nl.multiply(gst, float(GRP))

    tps = nl.ndarray((CH, 3), dtype=nl.float32, buffer=nl.psum)
    tps[:, 0:1] = nl.transpose(gst)
    tps[:, 1:2] = nl.transpose(ngr)
    tps[:, 2:3] = nl.transpose(tot)
    gstT = nl.copy(tps[:, 0:1])
    ngrT = nl.copy(tps[:, 1:2])
    totT = nl.copy(tps[:, 2:3])
    jid = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(jid, pattern=[[0, 1]], channel_multiplier=1)
    for k in range(TOKENS // GRP):
        sl = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
        # Clamped to `eidh`'s trash row rather than left to run off the end: an expert with
        # fewer than k+1 groups has no slot k, and a write nobody reads is safer here than an
        # out-of-range one.
        sl[...] = nl.minimum(nl.add(nl.add(gstT, float(k)),
                                    nl.multiply(nl.less_equal(ngrT, float(k)), BIG)),
                             float(NPG))
        slu = nl.copy(sl, dtype=nl.uint32)
        nisa.dma_copy(dst=eidh.ap(pattern=[[1, CH], [1, 1]],
                                  vector_offset=slu,
                                  indirect_dim=0),
                      src=jid, oob_mode=nisa.oob_mode.skip)
        # ...and the same scatter carries the group's token-tile count, which is what decides
        # which pass runs it: `ceil(min(GRP, tot[e] - k * GRP) / TOK)`, clamped to [0, GTOK].
        ntv = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
        ntv[...] = nl.minimum(
            nl.floor(nl.multiply(nl.add(nl.maximum(nl.add(totT, -float(k * GRP)), 0.0),
                                        float(TOK - 1)), 1.0 / TOK)),
            float(GTOK))
        nisa.dma_copy(dst=nth.ap(pattern=[[1, CH], [1, 1]],
                                 vector_offset=slu,
                                 indirect_dim=0),
                      src=ntv, oob_mode=nisa.oob_mode.skip)

    # ---- the group ordering: a counting sort of the plan's rows into the GPASS buckets.
    #
    # Everything here is NPG-wide vector work on one partition -- three masks, three cumulative
    # sums and one scatter -- and it is all derived from `nth`, i.e. from `tot`. No bucket
    # boundary, group count or expert identity is a constant.
    nts = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nts, src=nth.reshape((1, NPG + 1))[0:1, 0:NPG])
    gmsk = []
    for q in range(NPASS):
        m = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        m[...] = nl.greater(nts, float(GTHR[q]))
        gmsk.append(m)
    gsel = []
    gcum = []
    for q in range(NPASS):
        a = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        a[...] = nl.copy(gmsk[q]) if q == 0 else nl.subtract(gmsk[q], gmsk[q - 1])
        gsel.append(a)
        c = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar_cumulative(dst=c, src=a, op0=nl.multiply, imm0=1.0, op1=nl.add)
        gcum.append(c)

    # Each pass's window start, and its width rounded up to a whole number of CORES and floored at
    # CORES. Both cores compute these from the same `tot`, so both agree on them.
    goff = []
    gom1 = []
    gcnt = []
    run0 = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    for q in range(NPASS):
        o = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        o[...] = nl.minimum(nl.maximum(run0, 0.0), float(GOFFMAX))
        goff.append(o)
        d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        d[...] = nl.add(o, -1.0)
        gom1.append(d)
        nq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nq[...] = nl.copy(gcum[q][0:1, NPG - 1:NPG])
        gcnt.append(nq)
        nxt = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nxt[...] = nl.add(o, nl.maximum(
            nl.multiply(nl.floor(nl.multiply(nl.add(nq, float(GSTEPQ[q] - 1)),
                                             1.0 / GSTEPQ[q])),
                        float(GSTEPQ[q])),
            float(GSTEPQ[q])))
        run0 = nxt

    # A group's position in the ordering; one no pass claims is parked on the last slot, which is
    # padding and names the trash row.
    gpos = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(gpos, float(GORDN - 1))
    for q in range(NPASS):
        tq = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tq, data=gcum[q], op0=nl.add, operand0=gom1[q])
        gpos[...] = nl.add(gpos, nl.multiply(gsel[q],
                                             nl.subtract(tq, float(GORDN - 1))))
    for h in range(0, NPG, PAR):
        gw = min(PAR, NPG - h)
        pt = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.psum)
        pt[...] = nl.transpose(gpos[0:1, h:h + gw])
        pu = nl.copy(nl.minimum(nl.maximum(pt, 0.0), float(GORDN - 1)), dtype=nl.uint32)
        gcl = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(gcl, pattern=[[0, 1]], channel_multiplier=1)
        gcv = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.sbuf)
        gcv[...] = nl.add(gcl, float(h))
        nisa.dma_copy(dst=gordh.ap(pattern=[[1, gw], [1, 1]],
                                   vector_offset=pu, indirect_dim=0),
                      src=gcv, oob_mode=nisa.oob_mode.skip)

    ipat = nl.ndarray((PAR, PAR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(ipat, pattern=[[1, PAR]], channel_multiplier=-1)
    ut = nl.ndarray((PAR, PAR), dtype=nl.bfloat16, buffer=nl.sbuf)
    ut[...] = nl.greater(ipat, 0.0)
    tokid = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(tokid, pattern=[[0, 1]], channel_multiplier=1)
    run = nl.zeros((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    eord = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    eord[...] = nl.multiply(nl.add(ipat[0:1, 0:CH], -float(CH)), -1.0)
    trash = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    trash[...] = nl.add(tokid, float(NPG * GRP))

    # The compaction is split between the cores as well, but not the way `t1` was: the slot a
    # (token, expert) pair lands in is a running prefix over token tiles, so tile `t`'s offsets
    # depend on every tile before it. What makes it splittable anyway is that the prefix is cheap
    # and the body is not. Core 1 replays core 0's 32 tiles **counting only** -- one load, one
    # compare, one partition reduce, no `max8` and no scatter -- to arrive at exactly the `run`
    # core 0 would hand it, and then does its own 32 tiles for real. `kf` is 0.0/1.0 either way, so
    # the replayed counts are bit-identical to the ones the real body produces.
    for t in range(shlo):
        wlc = nl.load(weffL[t], dtype=nl.float32)
        kfc = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        kfc[...] = nl.greater(wlc, 0.0)
        cc = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=cc, op=nl.add, data=kfc)
        run[...] = nl.add(run, cc)

    for t in range(shlo, shlo + SHT):
        wl = nl.load(weffL[t], dtype=nl.float32)
        kb = nl.ndarray((TOK, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
        kb[...] = nl.greater(wl, 0.0)
        kf = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        kf[...] = nl.copy(kb, dtype=nl.float32)
        ps = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=ut, moving=kb)
        off = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        off[...] = nl.add(base, run)

        npos = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        npos[...] = nl.subtract(nl.multiply(nl.less_equal(wl, 0.0), -BIG),
                                nl.add(ps, nl.broadcast_to(off, (TOK, CH))))
        bp = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=bp, src=npos)
        slq = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        slq[...] = nl.minimum(nl.multiply(bp, -1.0), nl.broadcast_to(trash, (TOK, 8)))
        su = nl.copy(slq, dtype=nl.uint32)

        key = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        key[...] = nl.add(nl.multiply(kf, nl.broadcast_to(eord, (TOK, CH))),
                          nl.multiply(wl, 0.5))
        bk = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=bk, src=key)
        wr = nl.multiply(nl.subtract(bk, nl.floor(bk)), 2.0)

        pay = nl.ndarray((TOK, 8, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pay[:, :, 0], float(t * TOK))
        nisa.tensor_scalar(dst=pay[:, :, 0], data=pay[:, :, 0], op0=nl.add, operand0=tokid)
        pay[:, :, 1] = nl.copy(wr)
        for r in range(8):
            nisa.dma_copy(dst=pk3.reshape((PKR * GRP, 2)).ap(
                              pattern=[[2, TOK], [1, 2]],
                              vector_offset=su[:, r:r + 1], indirect_dim=0),
                          src=pay[:, r, :], oob_mode=nisa.oob_mode.skip)

        c = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=c, op=nl.add, data=kf)
        run[...] = nl.add(run, c)

    # This core's share of the expert groups, which of them it runs, and with how many token
    # tiles each.
    #
    # **Which groups, and in what order.** The plan's rows are visited through `gordh`, the
    # device-built ordering above: pass `q` walks the window `[goff[q], goff[q] + gcnt[q])` of it and
    # the two cores **ping-pong** that window, core `pid` taking positions
    # `goff[q] + pid, goff[q] + pid + CORES, ...`. Ping-ponging is what makes a shortened loop worth
    # anything: without it one core would hold a whole window and the other none, and the wall
    # would not move.
    #
    # **How many.** `gcnt[q]` is how many groups the routing put in bucket `q`, summed on device
    # from `nth` -- which is itself `ceil(min(GRP, tot[e] - k * GRP) / TOK)`, i.e. from the
    # compaction's own `tot`. The trip count is that divided by this pass's `GSTEPQ` and rounded **up**, so the
    # two cores keep an identical trip count: a loop whose count differs between the two physical
    # cores of one logical core is divergent control flow, the shape of several of this toolchain's
    # hangs. The round-up reads one padding position, which names the trash row.
    #
    # **With how many tiles.** `GPASS[q]`, a static Python int, so the pass is a fully pipelined
    # static body -- `_gemm_fp4`'s `PDEP1`/`LEAD1` software pipeline needs that. The cost is one
    # traced copy of the expert body per bucket.
    #
    # Clamped to [1, NPGC // UNROLLQ[q]]. The *provisioned* capacity does not move: `NPG` groups are
    # still laid out and still reachable, `pk3`/`eidh`/`nth`/`part` are still sized for them, and
    # the clamp is what keeps a routing that overruns the plan (every expert live for every token,
    # as an all-zero gate produces) inside `gordh` rather than off the end of it.
    #
    # Both the skipped groups and the skipped tiles are **bit-identical**, not approximate, and for
    # two independent reasons each. Every (token, expert) pair lands in slot `gst[e] * GRP + k` for
    # some `k < tot[e]`, so it is inside its group's first `nth` token tiles: every slot a pass does
    # not reach carries token index BIG, which `sidx` clamps into this core's trash tile, so its
    # output goes somewhere nothing downstream reads -- and its routing weight is 0.0, so what it
    # would have scattered is an exact `+0.0` anyway. The same holds for a padding position of the
    # ordering, which names `pk3` row NPG: the compaction only ever parks weight-0.0 slots there.
    #
    # This core's trash row for a token index an empty slot never fills: `tokid + TOKENS`
    # inside this core's half, so the 128 rows of one scatter stay distinct and none is
    # out of range.
    strash = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    strash[...] = nl.add(tokid, float(tbase * TOK + TOKENS))

    # One trip-count register and one window pointer per pass, all of them device values.
    gptrs = []
    nbregs = []
    for q in range(NPASS):
        nbf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nbf[...] = nl.minimum(
            nl.maximum(nl.floor(nl.multiply(nl.add(gcnt[q], float(GSTEPQ[q] - 1)),
                                            1.0 / GSTEPQ[q])), 1.0),
            float(NPGC // UNROLLQ[q]))
        nbi = nl.copy(nbf, dtype=nl.int32)
        reg = nisa.register_alloc()
        nisa.register_load(reg, nbi)
        nbregs.append(reg)
        pf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        pf[...] = nl.add(goff[q], float(pid))
        gp = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        gp[...] = nl.copy(pf, dtype=nl.int32)
        gptrs.append(gp)
    gctx = (pk3, eidh, gordh, aqf, w1, s1, w3, s3, w2, s2, db, partf, strash, tbase)

    # Every group this core runs can hold tokens the *other* core compacted, so `pk3` has to be
    # whole before the first `_pack_cols` reads it.
    nisa.core_barrier(data=pk3, cores=(0, 1))

    # One nested def per pass, because that is the only form an inner function may take. Adding a
    # bucket to GPASS means adding one here, which the assertion beside GPASS spells out.
    def pass0(i):
        _grp_body(GPASS[0], gptrs[0], gctx, UNROLLQ[0])

    def pass1(i):
        _grp_body(GPASS[1], gptrs[1], gctx, UNROLLQ[1])

    def pass2(i):
        _grp_body(GPASS[2], gptrs[2], gctx, UNROLLQ[2])

    nl.fori_loop(0, nbregs[0], pass0)
    nl.fori_loop(0, nbregs[1], pass1)
    nl.fori_loop(0, nbregs[2], pass2)

    # The seam, in three steps, and every one of them is split between the two physical cores
    # rather than run twice.
    #
    # First the cores rejoin inside the logical core. Each has written its own half of `part`, and
    # `core_barrier` makes the other half readable. The sum of the two halves is 64 token tiles, and
    # **each core sums 32 of them**: the bytes landing in `cc_src` are the same either way, and a
    # tile summed on one core does not need summing on the other. That is the whole change here.
    # The duplicated version read 2 x 167.8 MB and wrote 167.8 MB *on each core* -- 1.0 GB of HBM
    # traffic for 503 MB of work; this halves it, and `ASSEMBLY.md` prices the duplicated pass at
    # ~1.5 ms per core against the 656 GB/s this host sustains HBM->HBM.
    #
    # The price of splitting it is one more barrier. `cc_src` is now written by both cores into
    # disjoint halves and then read whole by the collective, which is exactly the case
    # `nisa.core_barrier` documents: both cores `dma_copy` into disjoint portions of a shared-HBM
    # tensor, barrier, then both consume it. After the barrier each core's `all_reduce` sees the
    # same complete buffer it saw when both cores wrote all of it, so the collective's semantics
    # are untouched -- only who wrote which half of its input.
    #
    # Then the four ranks rejoin through the collective, and the sum is rounded to the module's
    # bfloat16 once. That round is split the same way: `cc_dst` is whole on both cores after their
    # own `all_reduce`, `out` is an IO tensor either core may write, and each core writing only its
    # own 32 tiles takes another 167.8 MB of reads and 83.9 MB of writes off each core. No barrier
    # is needed for this one -- a core reads only rows its own collective has filled, and the two
    # cores' halves of `out` are disjoint.
    CCT = NTILE // CORES
    clo = pid * CCT
    nisa.core_barrier(data=part, cores=(0, 1))
    for t in range(clo, clo + CCT):
        tsum = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tsum, data1=nl.load(part[t]), data2=nl.load(part[PT + t]),
                           op=nl.add, engine=nisa.engine.vector)
        nisa.dma_copy(dst=cc_src[t * TOK:(t + 1) * TOK, :], src=tsum)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    for t in range(clo, clo + CCT):
        sl = slice(t * TOK, (t + 1) * TOK)
        nl.store(out[sl, :], value=nl.copy(nl.load(cc_dst[sl, :]), dtype=nl.bfloat16))
    return out, part
