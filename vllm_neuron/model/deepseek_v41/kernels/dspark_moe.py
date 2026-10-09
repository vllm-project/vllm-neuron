"""All of `mtp.0.ffn` on 4 ranks, rejoining through `nki.collectives`.

The module is the chain `mtp.0.ffn_norm -> mtp.0.ffn`: an RMSNorm, then a 128-expert top-3 MoE
with fp4 routed experts and one fp8 shared expert. `mtp.0` is a multi-token-prediction layer, so
`ModelArgs.get_moe_config` returns the `dspark_*` pair -- 128 routed experts at top-3, not the
backbone's 384 at top-6.

Every rank runs the whole `ffn_norm -> router` prologue (replicated), the routed SwiGLU experts
for its own 32-expert shard, and one quarter of the shared expert, adds all of it into one
float32 accumulator, and `ncc.all_reduce` sums the four accumulators. The sum is the module's
output on every rank, rank 0 included, rounded to the module's bfloat16.

The routed-expert body is the optimized single-rank kernel at its best accepted commit. Four
things here are not in it.

**The shared expert is sharded, not replicated.** `MoE.forward` adds it *after*
`dist.all_reduce`, so one rank's partial leaves it out entirely. The reduce is a sum, so at
module scale the shared expert only has to appear across the four partials exactly once in
total -- and splitting it beats replicating it, because SwiGLU is elementwise along the
intermediate dimension and `w2` contracts over it. Rank r owns intermediate columns
`r*INTER_L .. r*INTER_L+INTER_L-1` of `w1`/`w3`, the SwiGLU on them, and the `w2` k-blocks they
contract with; the four `w2` partials sum to the whole shared expert. The intermediate is
zero-padded 2304 -> `INTER_PAD` first, so that all four ranks get identical shapes (one NEFF
serves all four) and every slice is a whole number of 128-row tiles and of 32-element scale
blocks. Zero weights contribute zero, so the padding changes no value.

**The fp8 weights are decoded by hand, in three elementwise instructions.** The bf16 bit
pattern of an e4m3fn code `b` is `(b & 0x80) << 8 | (b & 0x7f) << 4`, and that pattern reads as
the value times `2**-120` uniformly -- normals and subnormals alike, because bf16's subnormals
reach down far enough to hold e4m3fn's. `F8_SHIFT` carries the `2**120` back in the e8m0 block
scale's exponent, where it is free. Two of the three instructions run on a uint32 view that
carries two weights per lane. The hardware's own fp8 converter is not an option here: it reads
exponent 15 as inf/NaN, and these weights reach 448 (code 0x7e), so converting through it would
produce NaN rather than a small error.

**Every GEMM output is rounded to bfloat16, as the reference's own GEMMs are.**
`vendor/kernel.py`'s `fp4_gemm` and `fp8_gemm` both return `out_dtype=BF16`, so in the reference
`w1`, `w3` and `w2` each hand a bfloat16 tensor to the next step: the SwiGLU reads rounded
inputs and the float32 accumulator takes a rounded addend. Dropping those three roundings is
worth 4.5x of the module's error -- a host model of this module's arithmetic reproduces the
recorded output to `max_abs_err` 0.031250 with them and 0.140625 without, the second being
exactly the bootstrapped kernel's recorded figure. They cost six instructions per expert on
tiles 18 to 40 elements wide.

**The rank's work is split across the two physical cores of its logical core.** This host is
LNC=2, so the kernel is launched as `kernel[2]` and a rank's logical core is two physical
cores; without a split they run the identical program twice for one copy of the answer.
`nl.program_id(0)` is a trace-time constant -- the launch grid is traced once per physical core
and the NEFF carries both programs -- so each core's share is chosen with ordinary Python, with
no dynamic trip count and no predication. The 15 routed slots split **8 on core 0 against 7 on
core 1**, which also owns the shared-expert shard: core 0 takes tokens 0-1 whole plus choices
0-1 of token 2, core 1 takes token 2's remaining choice plus tokens 3-4 whole. Along whole
tokens the split can only be 9/6, and the measured 9/6 wall was one whole slot longer. The two
float32 partials are rejoined inside the logical core by `nisa.core_barrier` before the
collective.

The kernel is rank-agnostic on purpose, so that one NEFF runs on all four ranks and
`neuron-explorer capture` can replay it on four workers. A rank's identity lives entirely in
the values `inference.py` hands it: its own 32 experts' weights, its quarter of the shared
expert, and a `gate_w`/`gate_b` whose 128 rows are permuted so that this rank's experts are
rows 0..31. Top-3 selection, the renormalising sum and the bias add are all invariant under a
permutation of the experts, so the router still scores every one of the 128 -- only the row at
which a local expert's weight lands moves.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
NEXP = 128
NTOK = 5
TOPK = 3
BLK = 32
NBD = DIM // BLK            # 160 k-blocks along the hidden dimension
NBI = INTER // BLK          # 72 k-blocks along the intermediate dimension
NRD = INTER // 128          # 18 output-row tiles of w1/w3
NRO = DIM // 128            # 40 output-row tiles of w2
# How many 32-element k-blocks one decode chunk covers. This is a *pipelining* knob as much as a
# width one: each chunk is an independent extract(Vector) -> widen(Activation) -> mask(Vector)
# chain, and Vector can only hide its wait on the Activation prescale if another chunk's work is
# ready to issue. Measured below.
CH = 80
# Where `_project`'s halving tree stops. At 1 the tree runs 32 -> 1 and `nisa.tensor_reduce`
# drops out of the per-block reduction entirely: measured, that is worth 2.2% of the wall,
# because `tensor_reduce` is the one stage in `_project` that does not reach the Vector engine's
# 2x element rate. It costs accuracy, since `tensor_reduce` accumulates in float32 internally
# while the tree rounds to `dtype` at every level -- `max_abs_err` 0.0625 -> 0.0693 against a
# 0.1547 ceiling, cosine 0.999966 -> 0.999954 against 0.999842.
TAIL = 1

# How many independent 128-row weight tiles one `_project` call carries on its free axis. The
# partition axis is full (128 output rows), so this is the only dimension left to widen the
# narrow stages with: after the halving tree bottoms out, every instruction is `batch * nb`
# elements, and at `batch = 1` that is 160 elements for `w1`/`w3` and 72 for `w2` against a fixed
# ~150 ns of issue cost per instruction. `w1` and `w3` share a tile's output rows and its
# activation, so they pair for free at `W13B = 2`; `w2`'s 40 output-row tiles are independent of
# each other, so `W2B` is a straight unroll of its loop, bounded by SBUF -- the product tile is
# `W2B * NBI * BLK` float16 per partition and three more tiles of that order live beside it.
W13B = 2
W2B = 4

# A slot is one (token, rank-of-the-token's-choice) pair. An expert can be chosen at most once
# per token, so a token lights at most TOPK of this rank's experts and NTOK * TOPK slots cover
# every (token, local expert) pair that can exist. 15 against the dense 32.
NSLOT = NTOK * TOPK

EST1 = INTER * NBD * 16
EST1S = INTER * NBD
EST2 = DIM * NBI * 16
EST2S = DIM * NBI

WORLD_SIZE = 4
NLOCAL = NEXP // WORLD_SIZE          # 32 routed experts per rank

# LNC=2: the rank's logical core is two physical cores and the kernel is launched as kernel[2].
CORES = 2
# Which of the 15 routed slots each physical core runs. A slot is one `(token, choice)` pair and
# costs one expert's three GEMMs whatever its routing weight, so 15 slots split 9/6 along whole
# tokens left core 1 -- which also owns the shared-expert shard -- with real slack and put core 0
# on the wall. The split here is 8/7: core `pid` runs tokens `FULL_LO[pid] .. FULL_HI[pid]-1`
# whole (3 slots each) plus choices `PART_LO[pid] .. PART_HI[pid]-1` of the one *split* token
# `PART_TOK`. The whole tokens keep their `nl.sequential_range(TOPK)` at the full static trip
# count -- a *reduced* `sequential_range` is the known failure where this kernel's compacted
# indirect weight gathers start reading out of bounds -- and the split token's choices are
# unrolled at trace time instead, so there is no trip count to shrink.
FULL_LO = (0, 3)
FULL_HI = (2, NTOK)
PART_TOK = 2
PART_LO = (0, 2)
PART_HI = (2, TOPK)
SHARED_CORE = 1

# The shared expert's intermediate, padded so that each of the four ranks owns a whole number of
# 128-row tiles (5) and of 32-element scale blocks (20). `inference.py` reads INTER_PAD from
# here: the host packing and the kernel have to agree on it exactly, and a second copy of the
# number is a way for them to disagree quietly.
INTER_PAD = 2560
INTER_L = INTER_PAD // WORLD_SIZE    # 640 intermediate columns per rank
NRS = INTER_L // 128                 # 5 output-row tiles of the shared w1/w3
NBS = INTER_L // BLK                 # 20 k-blocks of the shared w2's contraction

NORM_EPS = 1e-20
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
FP8_MAX_INV = 1.0 / 448.0
AMAX_FLOOR = 1e-4
RTNE_MAGIC = 12582912.0

# fp4 e2m1 decode: one multiply places both bfloat16 fields, the mask clears the spare bits.
NIB_MASK = 0x0F0F
NIB_MUL = 4160.0
FIELD_MASK = 0x81C081C0 - (1 << 32)

# fp8 e4m3fn decode: `b << 8` widens the byte into a uint16 lane, then the magnitude shifts down
# four and the sign stays where it is.
BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)

# The activation leaves `_act_quant` scaled by 2**ACT_EXP and the fp4 decode leaves a weight
# scaled by 2**-126, so the products land near 1 where float16 has its mantissa. The block scale
# carries the difference back: 126 - ACT_EXP for an fp4 weight, 120 - ACT_EXP for an fp8 one.
ACT_EXP = 118
W_SHIFT = 126 - ACT_EXP
F8_SHIFT = 120 - ACT_EXP

# The shared expert's product and within-block reduction type. float32 was chosen in the
# single-rank kernel because an fp8 code reaches 448 and a 32-element block sum of raw codes
# against raw activation codes overflows float16's 65504 -- so the routed path's float16 is not
# available here. bfloat16 has the exponent range and the Vector engine's 2x mode, and it costs
# nothing measurable in accuracy: the reference's own `fp8_gemm` returns bfloat16, so each of
# these three GEMM outputs is rounded to bfloat16 one instruction later anyway (`_bf16`, and
# `_act_quant`'s bfloat16 result). Only the 32 products inside one scale block are summed in
# bfloat16; the cross-block accumulation stays float32, where the block scale multiplies in.
SHARED_DTYPE = nl.bfloat16


def _chunk(nb):
    n = (nb + CH - 1) // CH
    while nb % n:
        n += 1
    return nb // n


def _fp4_values(packed, nb, nt=1):
    """Packed fp4 e2m1 bytes -> bfloat16 values times 2**-126, a block deinterleaved.

    A block's 32 weights come out in the order `w0 w2 .. w30 w1 w3 .. w31` -- all 16 low nibbles,
    then all 16 high ones -- which is the order `_act_quant` hands the activation in.

    `packed` is `(m, nt, nb, 16)` and the result `(m, nt, nb, BLK)`. `nt` is a *weight* batch:
    several independent 128-row weight tiles stacked on the free axis so that every instruction
    downstream of here is `nt` times wider. The decode itself still chunks along `nb` only,
    because a chunk has to be one affine slice of the (nt, nb) pair.
    """
    m = packed.shape[0]
    out = nl.ndarray((m, nt, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    o32 = out.view(nl.uint32)
    p16 = packed.view(nl.uint16)
    cs = _chunk(nb)
    # The two nibble extracts run **once for the whole `(nt, nb)` weight tile**, not once per
    # decode chunk. They are the kernel's two most issue-cost-bound instructions: at `cs = 80`
    # one of them is 640 uint16 elements, which the Vector engine's 4x mode retires in ~160
    # cycles against ~150 ns of fixed per-instruction cost -- 51% efficiency, ~130 us/core each
    # of pure issue overhead. `p16` is one contiguous affine slice over all of `(nt, nb, 8)`, and
    # the destination's "8 contiguous, stride 16" pattern is exactly the one the chunked version
    # already wrote, so this is 4x the width at the same mode and the same bytes touched.
    # Only the Activation prescale and its mask stay chunked: they are what the chunking knob is
    # actually for (an extract -> widen -> mask chain per chunk, so Vector can hide its wait on
    # the Activation engine behind another chunk's mask), and the extracts do not participate in
    # that overlap -- they all have to finish before the first prescale can start anyway.
    nib = nl.ndarray((m, nt, nb, 16), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nib[:, :, :, 0:8], data=p16, op0=nl.bitwise_and, operand0=NIB_MASK)
    nisa.tensor_scalar(dst=nib[:, :, :, 8:16], data=p16, op0=nl.right_shift, operand0=4,
                       op1=nl.bitwise_and, operand1=NIB_MASK)
    nib8 = nib.view(nl.uint8)
    for u in range(nt):
        for c in nl.affine_range(nb // cs):
            # Both nibble fields land in *one* tile, low nibbles then high nibbles, so that the
            # widening multiply reads 32 contiguous bytes and writes 32 contiguous uint16 in one
            # instruction. It used to write `pre[:, :, half:BLK:2]` -- a stride-2 uint16
            # destination, twice -- and that one Activation-engine instruction was 94.7% of all
            # Scalar work on the core, running at 0.749 elements/lane/cycle: it touched 64 bytes
            # of address range to deliver 32 bytes of data. Contiguous it measures 0.9415, which
            # is 94.1% of the Activation engine's ceiling -- and that ceiling is **1.0**
            # element/lane/cycle, not the 2.0 the Vector engine reaches on 16-bit operands. A
            # per-(engine, mode, opcode) fit of duration against free size across this kernel's
            # whole instruction trace finds no 2x or 4x mode on the Activation engine anywhere:
            # TENSOR_SCALAR 0.999, ACTIVATE 1.006. So ~115 us/core is all that is left here.
            # The price is that a block's 32 weights come out *deinterleaved* --
            # `w0 w2 .. w30 w1 w3 .. w31` -- which is free, because `_project` sums all 32 of a
            # block against the activation and `_act_quant` hands the activation the same
            # permutation for nothing. The block scale is per block, so it does not care at all.
            # The field mask cannot ride along as this instruction's `op1`: on the Activation
            # path the compiler rejects it outright ("unsupported operator 'bitwise_and'"), so
            # the mask stays a Vector instruction of its own on the uint32 view.
            pre = nl.ndarray((m, cs, BLK), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pre, data=nib8[:, u, c * cs:(c + 1) * cs, :],
                               op0=nl.multiply, operand0=NIB_MUL,
                               engine=nisa.engine.scalar)
            nisa.tensor_scalar(dst=o32[:, u, c * cs:(c + 1) * cs, :], data=pre.view(nl.uint32),
                               op0=nl.bitwise_and, operand0=FIELD_MASK)
    return out.view(nl.bfloat16)


def _fp8_values(raw, nb):
    """fp8 e4m3fn bytes -> bfloat16 values times 2**-120, in four instructions.

    The widening multiply runs on the Activation engine, where the fp4 decode's prescale runs;
    the two bit steps act on a uint32 view, so each lane carries two weights.

    The widening is split in two so that this decode emits a block's 32 weights in the same
    deinterleaved order `_fp4_values` now does (`w0 w2 .. w30 w1 w3 .. w31`): the two halves read
    a stride-2 *source* and write a contiguous destination. One activation permutation then
    serves both decoders. The two bit steps are per-16-bit-field symmetric, so they do not care
    which weight sits in which field.
    """
    m = raw.shape[0]
    out = nl.ndarray((m, 1, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    o32 = out.view(nl.uint32)[:, 0]
    wide = nl.ndarray((m, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    for half in range(2):
        nisa.tensor_scalar(dst=wide[:, :, half * 16:(half + 1) * 16],
                           data=raw[:, :, half:BLK:2],
                           op0=nl.multiply, operand0=BYTE_HI,
                           engine=nisa.engine.scalar)
    mag = nl.ndarray((m, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mag.view(nl.uint32), data=wide.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4)
    nisa.scalar_tensor_tensor(dst=o32, data=wide.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=mag.view(nl.uint32))
    return out.view(nl.bfloat16)


def _pow2_from_e8m0(code, shift, shape=None):
    """An e8m0 block scale read as a **bfloat16** bit pattern, with the decode's inflation folded in.

    `(code + shift) << 7` is the bfloat16 bit pattern of `2**(code + shift - 127)` -- bfloat16 is
    float32 with sixteen mantissa bits chopped off, so it shares float32's 8-bit exponent and a
    power of two is exactly representable in it -- and it is one *arithmetic* expression,
    `code * 2**7 + shift * 2**7`, so it rides a single `tensor_scalar` on the Activation engine,
    which the API documents as supporting exactly the `op0=multiply, op1=add` pair. The product is
    exact (`(code + shift) * 128 <= 33664` against uint16's 65535) and `shift` is positive for
    both decoders, so the old `maximum(.., 0)` clamp was dead code.

    **Why bfloat16 and not the float32 this used to return.** The scale's only consumer is
    `_project`'s tail multiply, and the Vector engine's element rate is set by how many *bytes of
    operand* an instruction reads per lane per cycle, not by its opcode: two 2-byte operands get
    the 2x mode, anything with a 4-byte operand gets 1x. A float32 scale against the float16
    block sums was the one `tensor_tensor` in the hot path still running at 1x. In bfloat16 it is
    2x, the scale tiles halve (`s13` 23 KB/partition -> 11.5 KB, `s2` 11.5 -> 5.75), and the
    values are bit-for-bit what float32 delivered. Measured 4.72634 -> 4.56076 ms (-3.5%), output
    bit-identical -- 4x the ~0.8% the instruction's own share predicted, because on this kernel
    footprint buys schedule: the win only appears *together* with the hoisted nibble extract in
    `_fp4_values` (bf16 scale alone, chunked extract, measures 4.8043 ms -- worse than doing
    neither). Treat the two as one change.

    This used to be four instructions -- a uint8->int32 widening copy, an add, the dead maximum
    and the shift -- over the *largest* small tiles in the kernel: `s13` is
    `W13B * NRD * NBD = 5760` scales per partition and `s2` is `NRO * NBI = 2880`, so a slot paid
    four passes over 8640 int32 elements and kept up to four live 34.5 KB/partition int32 tiles
    alive while it did (the profile's `qSyncSpillReload0` SBUF spill cost 92 us of Vector idle).
    One pass over one tile, on an engine at 42% duty, is strictly less of everything.
    """
    r = nl.ndarray(code.shape, dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=code,
                       op0=nl.multiply, operand0=float(1 << 7),
                       op1=nl.add, operand1=float(shift << 7),
                       engine=nisa.engine.scalar)
    return r.view(nl.bfloat16).reshape(
        shape if shape is not None else (code.shape[0], 1, code.shape[1]))


def _act_quant(x, nb):
    """`act_quant(x, 32, "ue8m0")` and its dequant, as values, times 2**ACT_EXP.

    The 1e-4 amax floor, the `2**ceil(log2(amax/448))` power-of-two scale and the
    round-to-nearest-even onto the e4m3 grid, all by bit surgery: the exponent fields are built
    by shifting into place and the rounding is an add and subtract of a magic constant. No clamp
    is needed because the scale is chosen so that `|x| / s <= 448` already.
    """
    xf = nl.copy(x, dtype=nl.float32)
    amax = nl.ndarray((x.shape[0], nb, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=amax, op=nl.maximum, data=nl.abs(xf), axis=(2,), keepdims=True)
    v = nl.multiply(nl.maximum(amax, AMAX_FLOOR), FP8_MAX_INV)
    vi = v.view(nl.int32)
    ce = nl.add(nl.subtract(nl.bitwise_and(nl.right_shift(vi, 23), 255), 127),
                nl.minimum(nl.bitwise_and(vi, 0x7FFFFF), 1))
    s = nl.left_shift(nl.add(ce, 127 + ACT_EXP), 23).view(nl.float32)
    sinv = nl.left_shift(nl.subtract(127, ce), 23).view(nl.float32)
    q = nl.multiply(xf, sinv.broadcast(2, BLK))
    qe = nl.bitwise_and(nl.right_shift(q.view(nl.int32), 23), 255)
    se = nl.maximum(nl.subtract(qe, 130), -9)
    step = nl.left_shift(nl.add(se, 127), 23).view(nl.float32)
    stepinv = nl.left_shift(nl.subtract(127, se), 23).view(nl.float32)
    r = nl.subtract(nl.add(nl.multiply(q, stepinv), RTNE_MAGIC), RTNE_MAGIC)
    val = nl.multiply(nl.multiply(r, step), s.broadcast(2, BLK))
    # Both weight decoders emit a block's 32 weights deinterleaved -- even positions first --
    # because that is what lets their widening multiply write a contiguous destination. The
    # activation has to match, and this is where it costs nothing: two strided reads over a tile
    # with at most NTOK partitions and a few thousand elements, against the 892M the decode
    # feeds. A within-block permutation is invisible to everything else, because `_project` sums
    # the whole block and the e8m0 scale is per block.
    out = nl.ndarray((x.shape[0], nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[:, :, 0:BLK // 2], src=val[:, :, 0:BLK:2])
    nisa.tensor_copy(dst=out[:, :, BLK // 2:BLK], src=val[:, :, 1:BLK:2])
    return out


def _broadcast_tokens(src, nb, scratch, ntok):
    """The quantized activation on all 128 partitions, replicated by one stride-0 DMA."""
    nisa.dma_copy(dst=scratch[0, 0:ntok, 0:nb, 0:BLK], src=src)
    return _bcast(scratch, 0, ntok, nb)


def _bcast(scratch, t0, ntok, nb):
    """`ntok` tokens of an already-stored quantized activation, on all 128 partitions.

    Split out of `_broadcast_tokens` so the routed path can replicate **one token at a time**.
    The full five-token `x` activation is `128 * NTOK * NBD * BLK` bfloat16 = **51.2 KB per
    partition**, it was live across the whole routed loop, and a slot reads exactly one of its
    tokens. Against a 192 KB/partition SBUF that one tile was a quarter of the budget -- and
    footprint is the measured binding constraint on this kernel (`W2B = 8` costs +5.4% even
    after iter-4's scale-decode change freed four int32 tiles, and iter-3's `qSyncSpillReload0`
    charged 92 us of Vector idle to spill). One token at a time is 10.24 KB live, and it is also
    *less* DMA, because a core only ever needs three of the five tokens: 3 x 1.3 MB against one
    6.5 MB replication.
    """
    dst = nl.ndarray((128, ntok, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=dst, src=scratch[0:1, t0:t0 + ntok, 0:nb, 0:BLK].broadcast(0, 128))
    return dst


def _project(wvals, sf, actb, nb, dst, ntok, dtype=nl.float16, wbatch=False):
    """One GEMM pass: 128 output rows x `ntok` lanes, contracting `nb` k-blocks.

    The block scale is applied once per k-block, after the 32 products inside it have been
    summed -- so the scale costs one element per k-block rather than one per weight. `dtype` is
    the product and reduction type: float16 for the routed fp4 path, where the operands are
    scaled so the products land near 1 and its extra mantissa bit is worth having, float32 for
    the shared fp8 path, where an fp8 code reaches 448 and the block sums are correspondingly
    wider.

    The `ntok` lanes on the free axis are either **tokens** against one weight tile
    (`wbatch=False`: the weight and its scales are a stride-0 view across them, as the shared
    expert's `w2` uses) or **weight tiles** against one token (`wbatch=True`: the activation is
    the stride-0 view instead). The two cost the same elements; what the second buys is
    instruction *width*. Every stage below is `ntok * nb * width` elements wide, and the last
    halvings and the tail reduce are only `ntok * nb` -- at `nb = 72` and `ntok = 1` that is
    72 elements against ~150 ns of fixed issue cost per instruction, i.e. a stage running at
    well under a fifth of the Vector engine's element rate. Stacking independent weight tiles
    on this axis is the one way to widen them without changing what is computed, because the
    partition axis is already full with 128 output rows.
    """
    m = wvals.shape[0]
    if wbatch:
        w, f = wvals, sf
        a = actb.broadcast(1, ntok) if ntok > 1 else actb
    else:
        w = wvals.broadcast(1, ntok) if ntok > 1 else wvals
        f = sf.broadcast(1, ntok) if ntok > 1 else sf
        a = actb
    cur = nl.ndarray((m, ntok, nb, BLK), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cur, data1=w, data2=a, op=nl.multiply)
    width = BLK
    while width > TAIL:
        v = cur.reshape((m, ntok, nb, 2, width // 2))
        nxt = nl.ndarray((m, ntok, nb, width // 2), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nxt, data1=v[:, :, :, 0, :], data2=v[:, :, :, 1, :],
                           op=nl.add)
        cur, width = nxt, width // 2
    if width == 1:
        # The tree ran all the way down, so there is nothing left for `tensor_reduce` to do.
        # `nisa.tensor_reduce` is the one stage here that does not reach the Vector engine's
        # 2x element rate, so every element it would have read is cheaper as another halving.
        part = cur.reshape((m, ntok, nb))
    else:
        part = nl.ndarray((m, ntok, nb), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=part, op=nl.add, axis=(3,), data=cur)
    # The block scale is bfloat16 (`_pow2_from_e8m0`), so this multiply reads 2 + 2 bytes of
    # operand per lane and gets the Vector engine's 2x mode; against the float32 scale it used to
    # read it was the last 1x `tensor_tensor` in the hot path. The destination stays float32 even
    # though both operands are 16-bit: a `dtype`-wide destination here measures 4.6060 against
    # this 4.5608 ms, so the following `tensor_reduce` -- which has no performance mode at all --
    # would rather be handed float32 than save the write bandwidth.
    scaled = nl.ndarray((m, ntok, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=part, data2=f, op=nl.multiply)
    nisa.tensor_reduce(dst=dst, op=nl.add, axis=(2,), data=scaled)


def _compact(rwl):
    """The slot list: which local expert each (token, choice) slot runs, and where its weight is.

    Built from this kernel's own routing weights with four small matmuls against a strict lower
    triangle. `live[e, t]` is 1 where local expert e has a nonzero weight for token t, `pfx` is
    how many live local experts precede e for that token, and slot `TOPK * t + r` selects the
    expert with `pfx == r`. A slot with no expert -- a token that lit fewer than TOPK of this
    rank's experts -- gets expert 0 and a routing-weight row of zeros, so it contributes exactly
    zero rather than needing to be skipped. Both indices are clamped in float32 before the int32
    cast: an index outside the array is an indirect-DMA out-of-bounds at runtime, which is a
    dead device rather than a wrong number.
    """
    livef = nl.ndarray((NLOCAL, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    livef[...] = nl.copy(nl.greater(rwl, 0.0), dtype=nl.float32)
    live = nl.ndarray((NLOCAL, NTOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    live[...] = nl.copy(livef, dtype=nl.bfloat16)

    tri = nl.zeros((NLOCAL, NLOCAL), dtype=nl.bfloat16, buffer=nl.sbuf)
    for f in range(1, NLOCAL):
        nisa.memset(dst=tri[0:f, f:f + 1], value=1.0)
    onec = nl.ndarray((NLOCAL, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onec, value=1.0)
    pfxp = nl.ndarray((NLOCAL, NTOK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pfxp, stationary=tri, moving=live)
    iotap = nl.ndarray((NLOCAL, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=iotap, stationary=tri, moving=onec)

    pfx = nl.copy(pfxp, dtype=nl.float32)
    sel = nl.ndarray((NLOCAL, NSLOT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(NTOK):
        for r in range(TOPK):
            g = TOPK * t + r
            nisa.tensor_copy(
                dst=sel[:, g:g + 1],
                src=nl.multiply(livef[:, t:t + 1],
                                nl.copy(nl.equal(pfx[:, t:t + 1], float(r)),
                                        dtype=nl.float32)))

    ecol = nl.ndarray((NLOCAL, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    ecol[...] = nl.copy(iotap, dtype=nl.bfloat16)
    idxp = nl.ndarray((1, NSLOT), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=idxp, stationary=ecol, moving=sel)
    vldp = nl.ndarray((1, NSLOT), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vldp, stationary=onec, moving=sel)

    idxf = nl.copy(idxp, dtype=nl.float32)
    vldf = nl.copy(vldp, dtype=nl.float32)
    eidx = nl.copy(nl.minimum(nl.maximum(idxf, 0.0), float(NLOCAL - 1)), dtype=nl.int32)
    ridx = nl.copy(nl.minimum(nl.maximum(nl.add(idxf,
                                                nl.multiply(nl.subtract(1.0, vldf),
                                                            float(NLOCAL))), 0.0),
                              float(NLOCAL)), dtype=nl.int32)
    return eidx, ridx


def _to_tokens(src, ntile, dst, ntok):
    """(128 rows, tile, token) -> (token, tile, 128 rows), one PE transpose per tile."""
    for i in nl.affine_range(ntile):
        ps = nl.ndarray((ntok, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ps, data=src[:, i, :])
        nisa.tensor_copy(dst=dst[:, i, :], src=ps)


def _bf16(x):
    """One GEMM output, rounded the way the reference's own bf16-returning GEMMs round it."""
    return nl.copy(nl.copy(x, dtype=nl.bfloat16), dtype=nl.float32)


def _swiglu(gate, up):
    """SwiGLU over two rounded GEMM outputs: the up branch clamped both sides, the gate only
    from above, exactly as `Expert.forward` clamps them at `swiglu_limit = 10.0`."""
    u = nl.minimum(nl.maximum(_bf16(up), -SWIGLU_LIMIT), SWIGLU_LIMIT)
    return nl.multiply(nl.silu(nl.minimum(_bf16(gate), SWIGLU_LIMIT)), u)


def _routed_slot(t, r, lst, xa, rwscr, hscr, yT, w1p, w1s, w3p, w3s, w2p, w2s):
    """One routed slot -- token `t`'s choice `r` -- as one expert's three fp4 GEMMs.

    A module-level function, not an inlined loop body, so that the two physical cores can run
    *different numbers* of slots: the whole tokens come from a full-trip-count
    `nl.sequential_range(TOPK)` and the split token's choices from a trace-time unrolled
    `range`, and both call this. `r` is a traced index in the first case and a Python int in the
    second; everything it indexes is affine in it either way.
    """
    ed0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rd0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ed0, src=lst[0:1, TOPK * t + r:TOPK * t + r + 1])
    nisa.dma_copy(dst=rd0, src=lst[1:2, TOPK * t + r:TOPK * t + r + 1])
    e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rd = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=e, src=ed0)
    nisa.tensor_copy(dst=rd, src=rd0)

    # `w1` and `w3` share this tile's 128 output rows and its activation, so they are one
    # width-2 weight batch rather than two passes: every instruction from the decode's field
    # mask through the tail reduce is twice as wide, and the two of them no longer alternate
    # into one `wb` tile, which was a write-after-read the schedule had to serialise on.
    s13 = nl.ndarray((128, W13B, NRD, NBD), dtype=nl.uint8, buffer=nl.sbuf)
    for i in nl.affine_range(NRD):
        nisa.dma_copy(dst=s13[:, 0, i, :], src=w1s[0:1, i * 128:(i + 1) * 128, 0:NBD].ap(
            pattern=[[EST1S, 1], [NBD, 128], [1, NBD]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        nisa.dma_copy(dst=s13[:, 1, i, :], src=w3s[0:1, i * 128:(i + 1) * 128, 0:NBD].ap(
            pattern=[[EST1S, 1], [NBD, 128], [1, NBD]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
    f13 = _pow2_from_e8m0(s13.reshape((128, W13B * NRD * NBD)), W_SHIFT,
                          (128, W13B, NRD, NBD))

    guT = nl.ndarray((128, NRD, W13B), dtype=nl.float32, buffer=nl.sbuf)
    for i in nl.affine_range(NRD):
        wb = nl.ndarray((128, W13B, NBD, 16), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb[:, 0], src=w1p[0:1, i * 128:(i + 1) * 128, 0:NBD, 0:16].ap(
            pattern=[[EST1, 1], [NBD * 16, 128], [16, NBD], [1, 16]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        nisa.dma_copy(dst=wb[:, 1], src=w3p[0:1, i * 128:(i + 1) * 128, 0:NBD, 0:16].ap(
            pattern=[[EST1, 1], [NBD * 16, 128], [16, NBD], [1, 16]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        _project(_fp4_values(wb, NBD, W13B), f13[:, :, i, :], xa, NBD,
                 guT[:, i, :], W13B, wbatch=True)
    gateT = guT[:, :, 0:1]
    upT = guT[:, :, 1:2]

    rw = nl.ndarray((128, 1, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[:, 0, :], src=rwscr[0:1, 0:1, 0:NTOK].ap(
        pattern=[[NTOK, 1], [0, 128], [1, NTOK]],
        scalar_offset=rd, indirect_dim=0).squeeze_dim(0))
    hid = nl.multiply(_swiglu(gateT, upT),
                      rw[:, 0:1, t:t + 1].broadcast(1, NRD))
    hsb = nl.ndarray((1, NRD, 128), dtype=nl.float32, buffer=nl.sbuf)
    _to_tokens(hid, NRD, hsb, 1)
    hdqb = _broadcast_tokens(
        _act_quant(nl.copy(hsb.reshape((1, NBI, BLK)), dtype=nl.bfloat16), NBI),
        NBI, hscr, 1)

    s2 = nl.ndarray((128, NRO, NBI), dtype=nl.uint8, buffer=nl.sbuf)
    for j in nl.affine_range(NRO):
        nisa.dma_copy(dst=s2[:, j, :], src=w2s[0:1, j * 128:(j + 1) * 128, 0:NBI].ap(
            pattern=[[EST2S, 1], [NBI, 128], [1, NBI]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
    f2 = _pow2_from_e8m0(s2.reshape((128, NRO * NBI)), W_SHIFT, (128, NRO, NBI))

    # One accumulator tile for all NRO output-row tiles, added into `yT` once at the end. The
    # read-modify-write of `yT[:, j, t]` used to sit inside this loop, and NKI tracks
    # dependencies per *tile*: instructions writing disjoint columns of the one `yT` tile
    # serialise against each other's consumers, which pins the whole `affine_range` into a
    # chain. Writing disjoint columns of a private `accs` instead leaves the iterations
    # independent, and the accumulate becomes one 40-element instruction instead of 80
    # one-element ones. The loop itself runs `W2B` output-row tiles per iteration as one weight
    # batch, so the tail stages of `_project` are `W2B * NBI` elements wide rather than NBI.
    accs = nl.ndarray((128, NRO, 1), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(NRO // W2B):
        wb2 = nl.ndarray((128, W2B, NBI, 16), dtype=nl.uint8, buffer=nl.sbuf)
        for u in range(W2B):
            o = (j * W2B + u) * 128
            nisa.dma_copy(dst=wb2[:, u], src=w2p[0:1, o:o + 128, 0:NBI, 0:16].ap(
                pattern=[[EST2, 1], [NBI * 16, 128], [16, NBI], [1, 16]],
                scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        _project(_fp4_values(wb2, NBI, W2B), f2[:, j * W2B:(j + 1) * W2B, :], hdqb, NBI,
                 accs[:, j * W2B:(j + 1) * W2B, 0], W2B, wbatch=True)
    nisa.tensor_copy(dst=yT[:, 0:NRO, t:t + 1],
                     src=nl.add(yT[:, 0:NRO, t:t + 1], _bf16(accs)))


@nki.jit
def kernel(x_in, norm_w, gate_w, gate_b, w1p, w1s, w3p, w3s, w2p, w2s,
           sw1p, sw1s, sw3p, sw3s, sw2p, sw2s, replica_group):
    out = nl.ndarray((1, NTOK, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # `name=` on the collective's src/dst, or DRAM allocation fails with NCC_IBIR440; both
    # shared_hbm, because a collective may not mix buffer kinds; both scratch, because a
    # collective may not read or write an IO tensor (NCC_INLA001) -- hence the copies either
    # side of it. `part` is where the two physical cores publish their partials to each other.
    cc_src = nl.ndarray((NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    part = nl.ndarray((CORES * NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="part")

    xscr = nl.ndarray((1, NTOK, NBD, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    hscr = nl.ndarray((1, 1, NBI, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    sscr = nl.ndarray((1, NTOK, NBS, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    rwscr = nl.ndarray((NLOCAL + 1, 1, NTOK), dtype=nl.float32, buffer=nl.private_hbm)
    lst = nl.ndarray((2, NSLOT), dtype=nl.int32, buffer=nl.private_hbm)

    pid = nl.program_id(0)

    # ---------------------------------------------------------------- ffn_norm, replicated
    xt = nl.ndarray((NTOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_in[0, 0:NTOK, 0:DIM])
    nwt = nl.ndarray((NTOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=nwt, src=norm_w[0:1, 0:DIM].broadcast(0, NTOK))
    xf = nl.copy(xt, dtype=nl.float32)
    ss = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=nl.multiply(xf, xf), axis=(1,),
                       keepdims=True)
    rs = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / DIM), NORM_EPS))
    hx = nl.copy(nl.multiply(nl.multiply(xf, rs.broadcast(1, DIM)),
                             nl.copy(nwt, dtype=nl.float32)), dtype=nl.bfloat16)
    hscr2 = nl.ndarray((1, NTOK, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hscr2[0, 0:NTOK, 0:DIM], src=hx)

    # ---------------------------------------------------------------- the router, replicated
    # Every rank scores all 128 experts, because that is what decides which of *its own* 32 each
    # token picked. The correction bias steers the selection only; the weights come from the
    # unbiased scores, renormalised against the selected ones' sum and scaled by route_scale.
    gw = nl.ndarray((NEXP, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gw, src=gate_w[0:NEXP, 0:DIM])
    logitsT = nl.ndarray((NEXP, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    hb = nl.ndarray((NEXP, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(NTOK):
        nisa.dma_copy(dst=hb, src=hscr2[0:1, t, 0:DIM].broadcast(0, NEXP))
        nisa.tensor_reduce(dst=logitsT[:, t:t + 1], op=nl.add,
                           data=nl.multiply(gw, hb, dtype=nl.float32),
                           axis=(1,), keepdims=True)
    lps = nl.ndarray((NTOK, NEXP), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=lps, data=logitsT)
    scores = nl.sqrt(nl.softplus(nl.copy(lps, dtype=nl.float32)))

    gb = nl.ndarray((NTOK, NEXP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gb, src=gate_b[0:1, 0:NEXP].broadcast(0, NTOK))
    biased = nl.add(scores, gb)
    cur = nl.copy(biased, dtype=nl.float32)
    thr = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(TOPK):
        nisa.tensor_reduce(dst=thr, op=nl.maximum, data=cur, axis=(1,), keepdims=True)
        if r < TOPK - 1:
            hit = nl.copy(nl.equal(cur, thr.broadcast(1, NEXP)), dtype=nl.float32)
            cur = nl.subtract(cur, nl.multiply(hit, 1.0e30))
    keep = nl.copy(nl.greater_equal(biased, thr.broadcast(1, NEXP)), dtype=nl.float32)
    wsel = nl.multiply(scores, keep)
    wsum = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=wsum, op=nl.add, data=wsel, axis=(1,), keepdims=True)
    winv = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=winv, data=nl.add(wsum, 1.0e-20))
    rweight = nl.multiply(nl.multiply(wsel, winv.broadcast(1, NEXP)), ROUTE_SCALE)
    rwps = nl.ndarray((NEXP, NTOK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=rwps, data=rweight)
    # Rows 0..NLOCAL-1 are this rank's experts: `inference.py` permuted them to the front.
    rwl = nl.copy(rwps[0:NLOCAL, 0:NTOK], dtype=nl.float32)
    nisa.dma_copy(dst=rwscr[0:NLOCAL, 0, 0:NTOK], src=rwl)
    nisa.dma_copy(dst=rwscr[NLOCAL:NLOCAL + 1, 0, 0:NTOK],
                  src=nl.zeros((1, NTOK), dtype=nl.float32, buffer=nl.sbuf))
    eidx, ridx = _compact(rwl)
    nisa.dma_copy(dst=lst[0:1, 0:NSLOT], src=eidx)
    nisa.dma_copy(dst=lst[1:2, 0:NSLOT], src=ridx)

    nisa.dma_copy(dst=xscr[0, 0:NTOK, 0:NBD, 0:BLK],
                  src=_act_quant(hx.reshape((NTOK, NBD, BLK)), NBD))

    yT = nl.zeros((128, NRO, NTOK), dtype=nl.float32, buffer=nl.sbuf)

    # ------------------------------------------------- this core's routed experts, sharded
    # 8 slots on core 0 against 7 plus the shared shard on core 1. This core's whole tokens keep
    # their full-trip-count `nl.sequential_range(TOPK)`; its share of the split token's choices
    # is unrolled at trace time, so no traced trip count is ever below its static maximum.
    for t in range(FULL_LO[pid], FULL_HI[pid]):
        xat = _bcast(xscr, t, 1, NBD)
        for r in nl.sequential_range(TOPK):
            _routed_slot(t, r, lst, xat, rwscr, hscr, yT, w1p, w1s, w3p, w3s, w2p, w2s)
    if PART_HI[pid] > PART_LO[pid]:
        xap = _bcast(xscr, PART_TOK, 1, NBD)
        for r in range(PART_LO[pid], PART_HI[pid]):
            _routed_slot(PART_TOK, r, lst, xap, rwscr, hscr, yT,
                         w1p, w1s, w3p, w3s, w2p, w2s)

    # ------------------------------------------- this rank's quarter of the shared expert
    if pid == SHARED_CORE:
        # The shared expert needs all five tokens, but it runs *after* the routed loop, so its
        # 51.2 KB replication no longer overlaps the routed loop's weight and product tiles.
        xdqb = _bcast(xscr, 0, NTOK, NBD)
        sgateT = nl.ndarray((128, NRS, NTOK), dtype=nl.float32, buffer=nl.sbuf)
        supT = nl.ndarray((128, NRS, NTOK), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.affine_range(NRS):
            sb = nl.ndarray((128, NBD, BLK), dtype=nl.uint8, buffer=nl.sbuf)
            sq = nl.ndarray((128, NBD), dtype=nl.uint8, buffer=nl.sbuf)
            # One e8m0 scale covers 32 contiguous output rows here -- `fp8_gemm` indexes
            # `scales_b[n // 32, k]` -- so four stride-0 DMAs fill a 128-row tile's scales.
            nisa.dma_copy(dst=sb, src=sw1p[i * 128:(i + 1) * 128, 0:NBD, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq[g * BLK:(g + 1) * BLK, 0:NBD],
                              src=sw1s[i * 4 + g:i * 4 + g + 1, 0:NBD].broadcast(0, BLK))
            sf1 = _pow2_from_e8m0(sq, F8_SHIFT)
            wv1 = _fp8_values(sb, NBD)
            for t in range(NTOK):
                _project(wv1, sf1, xdqb[:, t:t + 1], NBD, sgateT[:, i, t:t + 1], 1,
                         SHARED_DTYPE)
            nisa.dma_copy(dst=sb, src=sw3p[i * 128:(i + 1) * 128, 0:NBD, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq[g * BLK:(g + 1) * BLK, 0:NBD],
                              src=sw3s[i * 4 + g:i * 4 + g + 1, 0:NBD].broadcast(0, BLK))
            sf3 = _pow2_from_e8m0(sq, F8_SHIFT)
            wv3 = _fp8_values(sb, NBD)
            for t in range(NTOK):
                _project(wv3, sf3, xdqb[:, t:t + 1], NBD, supT[:, i, t:t + 1], 1,
                         SHARED_DTYPE)

        shsb = nl.ndarray((NTOK, NRS, 128), dtype=nl.float32, buffer=nl.sbuf)
        _to_tokens(_swiglu(sgateT, supT), NRS, shsb, NTOK)
        sdqb = _broadcast_tokens(
            _act_quant(nl.copy(shsb.reshape((NTOK, NBS, BLK)), dtype=nl.bfloat16), NBS),
            NBS, sscr, NTOK)

        for j in nl.affine_range(NRO):
            sb2 = nl.ndarray((128, NBS, BLK), dtype=nl.uint8, buffer=nl.sbuf)
            sq2 = nl.ndarray((128, NBS), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb2, src=sw2p[j * 128:(j + 1) * 128, 0:NBS, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq2[g * BLK:(g + 1) * BLK, 0:NBS],
                              src=sw2s[j * 4 + g:j * 4 + g + 1, 0:NBS].broadcast(0, BLK))
            sf2 = _pow2_from_e8m0(sq2, F8_SHIFT)
            wv2 = _fp8_values(sb2, NBS)
            # All five tokens in one pass: the weight tile is a stride-0 view across them, so
            # five one-token calls cost the same elements in five times the instructions.
            sacc = nl.ndarray((128, NTOK), dtype=nl.float32, buffer=nl.sbuf)
            _project(wv2, sf2, sdqb[:, 0:NTOK], NBS, sacc, NTOK, SHARED_DTYPE)
            nisa.tensor_copy(dst=yT[:, j, :], src=nl.add(yT[:, j, :], _bf16(sacc)))

    # ------------------------------------------------------------------------- the seam
    ysb = nl.ndarray((NTOK, NRO, 128), dtype=nl.float32, buffer=nl.sbuf)
    _to_tokens(yT, NRO, ysb, NTOK)
    y = ysb.reshape((NTOK, DIM))

    # Both cores publish their float32 partial, barrier, and then add the other core's into
    # their own -- so both hold this rank's whole partial and write `cc_src` with the same
    # bytes. The collective then depends only on its own core's writes, and the one cross-core
    # ordering this kernel rests on is `core_barrier` between a `dma_copy` into shared HBM and a
    # later read of it.
    oth = (1 - pid) * NTOK
    nisa.dma_copy(dst=part[pid * NTOK:pid * NTOK + NTOK, 0:DIM], src=y)
    nisa.core_barrier(data=part, cores=(0, 1))
    nisa.dma_copy(dst=cc_src[0:NTOK, 0:DIM],
                  src=nl.add(y, nl.load(part[oth:oth + NTOK, 0:DIM])))

    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)

    nisa.dma_copy(dst=out[0, 0:NTOK, 0:DIM],
                  src=nl.copy(nl.load(cc_dst[0:NTOK, 0:DIM]), dtype=nl.bfloat16))
    return out
