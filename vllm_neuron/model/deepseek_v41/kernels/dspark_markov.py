"""All of `mtp.2.markov_head` on 4 ranks, rejoined by two pipelined `nki.collectives.all_reduce`.

The arithmetic is the optimized single-rank kernel (`submodule/source.py`) at its best
accepted commit, unchanged: not a tile width, not a loop bound, not an engine split. What is
new is the seam -- four named collective buffers, two pipelined `ncc.all_reduce` calls, two
DMAs out, and the dropping of the per-block HBM stores the single-rank kernel ended each
block with.

`FLOORPLAN.md` is binding on the split: **`hiddenx4(allreduce)`**. So rank `r` holds hidden
columns `[64r, 64r+64)` of both `(129280, 256)` tables and computes a *partial sum* over its
own quarter of the contraction, for every one of the 129280 vocabulary rows:

    logits_partial[v] = sum(k in this rank's 64 columns) embed_row[k] * head_weight[v, k]

Summing the four ranks' partials is the module's output. `head` is the module's row-parallel
(contraction-sharded) projection and carries no bias, no norm and no activation, so nothing
sits outside the contraction: after the reduce every rank -- rank 0 included -- holds the
whole module's logits and nothing is applied afterwards. The collective therefore sits at the
one point in the chain where a rank holds a tensor of the module's output shape carrying an
incomplete sum: after the sweep, before anything is written out.

    acc_0       the first 248 of this core's 505 partial logits a partition, float32, SBUF
    ncc.all_reduce(acc_0 -> cc_dst_0)      issued the moment column 248 retires, so it runs
                                           while the last two blocks are still being swept
    acc_1       the remaining 257, reduced by a second call after the sweep
    cc_dst_0, cc_dst_1 -> logits           two SBUF->HBM DMAs per physical core

`ASSEMBLY.md` ruled this split out on paper -- "K chunks pay the 21.4 us fixed cost K-1 extra
times to recover at most the 25.8 us the dependency costs" -- and measured it is wrong, because
the first call's cost is not additive: it overlaps the tail of the sweep. Across four draws the
two-call form reads 0.127278-0.128414 ms against the single call's 0.130119-0.131711, and
`cc_op_count` goes to 2 with the two calls' `cc_op_time` summing to *more* than the single
call's while the kernel finishes sooner -- which is only possible if the first one is
concurrent with compute. The two column ranges are independent sums, so the output is
bit-identical to the single-call form: `max_abs_err` 3.814697265625e-06 at every cut tried.

Where the cut goes barely matters and K does. `CUTS` is swept: a second cut costs a full extra
fixed collective cost and buys no new cover (K=3 at 168/337 is **0.149162 ms**, +22 us, with
`total_minus_cc` up to 0.108 ms on *every* rank), while sliding the single cut inside the broad
flat region does nothing (248 is 0.127278 and 206 is 0.128488, one draw spread apart, against
iter-1's 120 at 0.129411 and 408 at 0.135001). So the ring is a serial resource whose per-call
cost is the thing to minimise, the count is pinned at two, and the position only has to put the
cut where the remaining sweep roughly covers the first call -- anywhere in 206-248 does.

The sweep's weight loads do not wait for the ring any more. The collective's bring-up -- event
waits, MMIO doorbell writes, `SET_ORDERING_MODE` -- is queued on the GpSimd engine, and GpSimd
is also what SWDGE uses to generate DMA descriptors, which is the mode the compiler picks for
`nl.load` by default. On physical core 0 the bring-up therefore sits *ahead of every weight
load*, so none of the 8.3 MB a core can start moving until the ring is up. Asking for
`dge_mode=nisa.dge_mode.hwdge` with `engine=nisa.engine.sync` hands descriptor generation to the
hardware DGE block off an otherwise-idle sequencer, and the loads then run *during* the bring-up
instead of after it: **0.128043 -> 0.126079 ms** on the gated metric, output bit-identical, and
the NEFF rehashed, so this is code generation and not a redraw. The token-id fetch and the two
mandated copy-outs move with them; the indirect gather of the embedding row cannot, because
`nisa.dma_copy`'s indirect form asserts `dge_mode in (unknown, swdge)`, and that one DMA is what
still pins the start of the arithmetic behind the bring-up.

With that done the reported rank's wall is additive to within 0.2 us in three terms, which is
the decomposition to aim the next iteration with:

    0.1261 = 0.0240 (the collective's bring-up, ahead of all arithmetic)
           + 0.0744 (`nocc`: the sweep at Vector's arithmetic floor)
           + 0.0277 (`cc_op_time`, flat to +-0.6 us across every variant ever tried here)

`cc_op_time` is a constant of this part, not a tunable: it reads 0.0275-0.0281 ms at cuts 206,
248 and 320, at `RAMP_MAX` 128, 160 and 256, and at both Scalar shares. Everything any knob has
ever moved is the middle term, and the first term is why `latency_ms` is 52 us above `nocc`
while `cc_op_time` is only 28 of it.

What is left exposed is not reachable from here. On the reported rank the wall is 0.0993 ms
outside the collective plus 0.0280 ms inside it, and `nocc` -- the same sweep with the
collective deleted -- is 0.0734 ms, so 26 us of the non-collective time is the runtime's own
SBUF->HBM staging ("Falling back to intermediate buffer in HBM"), not this kernel's work. It is
not the mandated copy-out: issuing each chunk's `dma_copy` immediately after its own call
instead of holding both to the end leaves `total_minus_cc` unmoved (0.1003 ms) and latency
inside the draw spread. And it cannot be bought back by handing the collective
`nl.private_hbm` operands that each broadcast pass tops up as it retires: that does delete the
staging (`cc_op_time` 0.0280 -> 0.0236 ms, and the fallback warning stops being logged), but
the per-pass staging DMAs then contend with the sweep's 8.3 MB of weight loads and
`total_minus_cc` rises further than `cc_op_time` falls. `nl.private_hbm` is also private to the
*rank* and not to the physical core, so one buffer is shared by both `nl.program_id(0)`
instances and half the vocabulary comes out as the other core's -- `pass_fraction` 0.5015.

The module's second recorded output is the embedding row itself, and here it stays sharded:
rank `r` returns its own 64 columns of it, which is the hidden-parallel layout the floorplan
puts every other tensor in. Concatenating the four ranks' slices reproduces the module's
second recorded golden under `tensors/` bit for bit (`secondary_reassembly` in the submodule
declaration), and no collective is needed for that -- each rank already holds its own slice.
`ASSEMBLY.md` has the measurement of that choice and of every variant of the seam.

The reduction is float32, and here that is forced rather than chosen. The bar is the
bootstrapped module's own achieved worst element plus 10% -- `MAX_ABS_ERR = 6.29e-06` on
logits that reach 20.353355 -- about 1.7 ulp of float32 at that magnitude. One bfloat16
rounding step there is 0.0625 and one float16 step is 0.0156: four and three orders past the
ceiling. No narrower reduction is admissible at any price, and float32 is also what the
reference reduces in.

Six things about `nki.collectives` on this toolchain, each of which otherwise fails with an
internal compiler error naming something else. **1, 2, 3 and 4 were re-tested for this
repo**; two of the four came out differently, and `ASSEMBLY.md` has every log:

1. the kernel is launched as `kernel[2]`, because this host is LNC=2 -- **confirmed,
   verbatim**: a correct `NCORE = 1` kernel launched as `kernel[1]` dies at
   `[INTERNAL_ERROR] [NCC_ILLC059] Could not find MemoryLocation named inst__I-6-0:cc_src on
   core 1`. The **frozen** `inference.py` does the launch, so no loop iteration can drop it,
   and it refuses to run unless `NCORE` below agrees with that launch width -- a 2-wide
   launch whose kernel does not split on `nl.program_id(0)` silently does all the work twice;
2. `accall` / `accdst` carry `name=`, or DRAM allocation fails (`NCC_IBIR440`) --
   **refuted on this toolchain version.** Dropping `name=` entirely compiles, runs and
   matches the bar in both the SBUF and the `nl.shared_hbm` form, at a latency inside the
   draw spread either way. Kept here regardless: it costs nothing and it is what labels these
   two buffers in the profile;
3. they are scratch, not the returned tensor, because a collective may not read or write an
   IO tensor -- **confirmed**: `dsts=[out3]` dies in `birverifier::checkCollective` with
   `[INTERNAL_ERROR] [NCC_INLA001] ... Collective instruction cannot write IO tensors`. That
   is what makes the `dma_copy` out mandated rather than chosen;
4. every `src` and `dst` must be `nl.shared_hbm` -- **half refuted, half confirmed.** SBUF
   operands work, and at this payload they are the faster ones: the accumulators the sweep
   already fills *are* the collective's source, so there is no staging copy in, no HBM read
   back out, and no `nisa.core_barrier`, since each physical core reduces the half of the
   vocabulary it computed. Measured on this repo, both passing the bar with a `max_abs_err`
   identical to the last bit: **0.130119 ms SBUF against 0.141191 ms HBM**, 11.1 us (7.8%)
   cheaper. The runtime does log `ENC:rebuild_dataarrays_for_allreduce ... Falling back to
   intermediate buffer in HBM; sbuf partition size: 126, input partition elem: 505, chunk
   size: 16160` for the SBUF form, so what SBUF buys is the staging copy, not an
   SBUF-resident ring. What does hold is the half `ncc.all_reduce`'s own docstring states --
   the kinds may not be *mixed*: one SBUF operand and one HBM operand is rejected at
   `nki/collectives/_ops.py:58`, "All src & dst tensors must have the same buffer type";
5. the `ReplicaGroup` is built by `inference.py` as `ReplicaGroup([[0, 1, 2, 3]])` and passed
   in, because the tracer rejects `range` in a traced body. Honoured as given, not re-tested;
6. `inference.py` builds every tensor on CPU and `.to(device)`s it, because
   `torch.*(..., device=xla)` emits a broadcast HLO that hits `NCC_ISMP902` -- and that error
   names the broadcast, not the collective. Honoured as given, not re-tested.

Two more, inherited from the feedback archive and honoured without a test because nothing
here needed one: a collective's operand has to be a whole buffer, which is why `accall` is
one buffer whose column windows the blocks write rather than a slice of something larger; and
an operand of free size 1 compiles and then dies in the runtime at
`instr_collectives.c:538`, which 505 is comfortably clear of.

A rank's identity lives entirely in the values `inference.py` hands it -- its own 64 columns
of both tables -- and never in this code. One NEFF runs on all four ranks, which is what lets
`neuron-explorer capture` replay it on four workers and keeps the four ranks' schedules
identical.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

PMAX = 128

# The four logical NeuronCores the module is split over. `inference.py` checks this against
# its own WORLD_SIZE and builds the ReplicaGroup from it.
WORLD_SIZE = 4

# The two physical cores behind one logical NeuronCore at LNC=2, and the width the kernel is
# launched at. The vocabulary is split between them; each one reduces and writes out its own
# half.
NCORE = 2

# --------------------------------------------------------------------- the single-rank sweep
# Everything from here to the seam is `submodule/source.py` at commit 97ffc0ea0157, verbatim.

# The share of each vocabulary row's 64-term reduction that goes to the Scalar engine's free
# accumulator rather than to `tensor_reduce` on Vector.
SCALAR_NUM = 20
SCALAR_DEN = 101

# Free-axis width of one broadcast-multiply pass, and of the first (smallest) block's.
XWIDTH = 48
XWIDTH0 = 4

# Rows of the broadcast matmul issued per `nc_matmul`.
MMROWS = 8

# The vocabulary is swept in geometrically growing blocks, so the first weight tile is in
# flight almost immediately and the last ones are wide enough to amortize their loads.
RAMP0 = 8
RAMP = 2.0
RAMP_MAX = 160

# Columns of the 505 at which the sweep's accumulator is cut into separately-reduced chunks.
# Chunk `j` is reduced by an `all_reduce` issued the moment its last column retires, so every
# chunk but the last runs while the sweep is still going. A cut need *not* fall on a block
# boundary: the inner broadcast-multiply pass is clamped so no `tensor_reduce` straddles a cut,
# which costs at most one extra narrow instruction per cut. `()` is the single-call form.
CUTS = (248,)

# Issue each finished chunk's mandated copy-out immediately after its `all_reduce` instead of
# holding every DMA to the end, so the earlier chunks' SBUF->HBM traffic overlaps the later
# chunks' collectives.
EARLY_DMA = False

# Hand the collective `nl.private_hbm` operands instead of the SBUF accumulators, and push each
# broadcast pass's columns out to them the moment that pass retires. The runtime stages an SBUF
# operand through HBM anyway ("Falling back to intermediate buffer in HBM"), and that staging
# copy sits on the critical path *after* the sweep; done a pass at a time it overlaps the sweep
# instead. `private_hbm` rather than `shared_hbm` so each physical core owns its own buffer and
# no `nisa.core_barrier` is needed.
#
# **Closed, and not for the reason iter-2 recorded.** Re-tested here with the buffer shaped
# `(NCORE, PMAX, n)` and indexed by `nl.program_id(0)` -- the shape iter-2's notes prescribed as
# the fix for its half-wrong output -- and with the per-pass staging DMAs on HWDGE so they no
# longer share a queue with the weight loads. The mechanism still works (`cc_op_time`
# 0.0275 -> 0.0238 ms, the `rebuild_dataarrays_for_allreduce` fallback stops being logged) and
# `total_minus_cc` still rises further than that (0.0982 -> 0.1035 ms), so it is a net loss even
# with the DMA contention removed. And it is still **wrong**: `pass_fraction` is 0.5014851 to the
# digit, exactly as with the unshaped buffer. A collective resolves its HBM operand's address
# statically, so `buf[nl.program_id(0)]` does *not* give the two physical cores different
# slices -- both reduce slice 0. There is no per-physical-core HBM collective operand on this
# toolchain at LNC=2, so the SBUF form below is the only correct one.
CC_HBM = False

# Generate the sweep's weight-load descriptors on the hardware DGE block, driven by the Sync
# engine's sequencer, instead of letting the compiler pick -- which picks SWDGE, i.e. GpSimd.
# The collective's bring-up is queued on GpSimd too, and on physical core 0 it sits *ahead* of
# every weight load, so with SWDGE the sweep's 8.3 MB/core of weight traffic cannot start moving
# until the ring is up. HWDGE takes it off that queue, so the loads run *during* the bring-up
# instead of after it: 0.1280 -> 0.1261 ms, with the NEFF rehashed, so this is codegen and not a
# redraw.
HWDGE_W = True

# The same for the two mandated SBUF->HBM copy-outs of the collective's destinations, and for
# the token-id fetch at the head of the kernel. The indirect gather of the embedding row that
# follows the latter *cannot* move -- `nisa.dma_copy`'s indirect form asserts
# `dge_mode in (unknown, swdge)` -- so these two are the last GpSimd items that can be taken out
# of its way. Worth 0 on `latency_ms` (0.1264/0.1228 against 0.1261 for `HWDGE_W` alone, one
# draw spread) and worth 5.8 us on `slowest_rank_ms` and 6.1 us on `rank_spread_ms`.
HWDGE_OUT = True
HWDGE_IDX = True

# Which sequencer invokes the HWDGE block. Only `nisa.engine.sync` and `nisa.engine.scalar` are
# legal; Scalar is busy with its 20/101 share of the contraction, Sync does nothing else here.
HWDGE_ENGINE = "sync"


_hwdge_engine = nisa.engine.scalar if HWDGE_ENGINE == "scalar" else nisa.engine.sync


def _out_dma(dst, src):
    if HWDGE_OUT:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
    else:
        nisa.dma_copy(dst=dst, src=src)


def _blocks(rows_per_part):
    out = []
    left = rows_per_part
    size = RAMP0
    while left > 0:
        take = min(int(size), RAMP_MAX, left)
        if left - take < RAMP0:
            take = left
        out.append(take)
        left -= take
        size = size * RAMP
    return tuple(out)


def _chunk_widths(blocks):
    out = []
    i = 0
    for bw in blocks:
        cap = XWIDTH0 if i == 0 else XWIDTH
        n = (bw + cap - 1) // cap
        out.append((bw + n - 1) // n)
        i += 1
    return tuple(out)


def _needed_xb(widths):
    out = []
    high = 0
    for cw in widths:
        high = max(high, cw)
        out.append(high)
    return tuple(out)


def _impl(input_ids, embed_weight_shard, head_weight_shard, replica_group):
    embed_weight = embed_weight_shard
    head_weight = head_weight_shard

    vocab, dim = head_weight.shape
    n_tokens = 1
    for size in input_ids.shape:
        n_tokens *= size

    assert embed_weight.shape == head_weight.shape, (
        f"embed {embed_weight.shape} and head {head_weight.shape} must be the same shard"
    )
    assert vocab % (PMAX * NCORE) == 0, (
        f"PMAX * NCORE ({PMAX * NCORE}) must divide vocab ({vocab})"
    )
    assert dim % NCORE == 0, f"NCORE ({NCORE}) must divide the shard width ({dim})"
    assert 0 <= SCALAR_NUM <= SCALAR_DEN, "the Scalar share must be a fraction in [0, 1]"

    # The module's two recorded outputs. `logits` is the whole module's after the reduce;
    # `embed` is this rank's own 64 hidden columns of the embedding row, the shard the
    # floorplan's hidden cut puts it in.
    logits = nl.ndarray((n_tokens, vocab), dtype=nl.float32, buffer=nl.shared_hbm)
    embed = nl.ndarray((n_tokens, dim), dtype=embed_weight.dtype, buffer=nl.shared_hbm)

    core = nl.program_id(0)

    if HWDGE_IDX:
        idx = nl.ndarray((n_tokens, 1), dtype=input_ids.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=input_ids.reshape((n_tokens, 1)),
                      dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
    else:
        idx = nl.load(input_ids.reshape((n_tokens, 1)))

    rows_per_core = vocab // NCORE
    rows_per_part = rows_per_core // PMAX

    blocks = _blocks(rows_per_part)
    widths = _chunk_widths(blocks)
    need = _needed_xb(widths)

    # One indirect DMA fetches this rank's 64 columns of the token's embedding row, as many
    # copies of it as the widest broadcast pass needs.
    WFULL = need[-1]
    rowr = nl.ndarray((n_tokens, WFULL, dim), dtype=embed_weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=rowr,
        src=embed_weight.ap(
            pattern=[[dim, n_tokens], [0, WFULL], [1, dim]],
            offset=0,
            vector_offset=idx,
            indirect_dim=0,
        ),
    )

    ones = nl.ndarray((1, PMAX), dtype=embed_weight.dtype, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)

    bc0 = nl.ndarray((PMAX, need[0], dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bc0, stationary=ones, moving=rowr[:, 0:need[0], :],
                   is_stationary_onezero=True)

    bc = nl.ndarray((PMAX, WFULL, dim), dtype=nl.float32, buffer=nl.psum)
    k = 0
    while k < WFULL:
        m = min(k + MMROWS, WFULL)
        nisa.nc_matmul(dst=bc[:, k:m, :], stationary=ones, moving=rowr[:, k:m, :],
                       is_stationary_onezero=True)
        k = m

    w4 = head_weight.reshape((NCORE, PMAX, rows_per_part, dim))
    out3 = logits.reshape((NCORE, PMAX, rows_per_part))

    # The collective's source and destination. `accall` is not a staging buffer: every
    # block's accumulator is a column window of it, so the sweep's reductions land directly
    # in what the collective reads and the single-rank kernel's per-block HBM stores are
    # gone. 505 float32 a partition, 2020 B of each partition's SBUF, 252.5 KiB a core.
    bounds = (0,) + tuple(CUTS) + (rows_per_part,)
    for _i in range(len(bounds) - 1):
        assert bounds[_i] < bounds[_i + 1], f"CUTS must be strictly increasing within the sweep"
    nchunk = len(bounds) - 1

    # `acc` is what the sweep reduces into and must be SBUF. When the collective's operands are
    # HBM they are separate buffers that each pass's tail DMA fills; when they are SBUF the
    # accumulator *is* the operand and there is no copy in at all.
    acc_sb = []
    cc_src = []
    cc_dst = []
    for _j in range(nchunk):
        _n = bounds[_j + 1] - bounds[_j]
        acc_sb.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.sbuf,
                                 name=f"acc_{_j}"))
        if CC_HBM:
            cc_src.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.private_hbm))
            cc_dst.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.private_hbm))
        else:
            cc_src.append(acc_sb[_j])
            cc_dst.append(nl.ndarray((PMAX, _n), dtype=nl.float32, buffer=nl.sbuf,
                                     name=f"cc_dst_{_j}"))

    bi = 0
    r0 = 0
    while bi < len(blocks):
        bw = blocks[bi]
        cw = widths[bi]
        r1 = r0 + bw
        if HWDGE_W:
            w = nl.ndarray((PMAX, bw, dim), dtype=head_weight.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=w4[core, :, r0:r1, :],
                          dge_mode=nisa.dge_mode.hwdge, engine=_hwdge_engine)
        else:
            w = nl.load(w4[core, :, r0:r1, :])

        nscalar = (bw * SCALAR_NUM) // SCALAR_DEN
        split = bw - nscalar

        sink = nl.ndarray((PMAX, dim), dtype=nl.float8_e4m3, buffer=nl.sbuf)

        c0 = 0
        while c0 < bw:
            # Which chunk this pass's first column belongs to, and how far it may run before
            # it would cross into the next one.
            j = 0
            while bounds[j + 1] <= r0 + c0:
                j += 1
            base = bounds[j]
            acc = acc_sb[j]
            c1 = min(c0 + cw, bw, bounds[j + 1] - r0)

            prod = nl.ndarray((PMAX, c1 - c0, dim), dtype=nl.float32, buffer=nl.sbuf)
            if bi == 0:
                nisa.tensor_tensor(dst=prod, data1=w[:, c0:c1, :],
                                   data2=bc0[:, 0:c1 - c0, :], op=nl.multiply)
            else:
                nisa.tensor_tensor(dst=prod, data1=w[:, c0:c1, :],
                                   data2=bc[:, 0:c1 - c0, :], op=nl.multiply)

            v1 = min(c1, split)
            if v1 > c0:
                nisa.tensor_reduce(dst=acc[:, r0 + c0 - base:r0 + v1 - base], op=nl.add,
                                   data=prod[:, 0:v1 - c0, :], axis=(2,))

            b = max(c0, split)
            while b < c1:
                nisa.activation(dst=sink, op=nl.copy, data=prod[:, b - c0, :],
                                reduce_op=nl.add, reduce_res=acc[:, r0 + b - base:r0 + b - base + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                b += 1

            # This pass's columns are final now, so the HBM operand's copy of them goes out
            # here, a pass at a time, under the next pass's arithmetic.
            if CC_HBM:
                nisa.dma_copy(dst=cc_src[j][:, r0 + c0 - base:r0 + c1 - base],
                              src=acc[:, r0 + c0 - base:r0 + c1 - base])

            c0 = c1

            # ------------------------------------------------------------------- the seam
            # Chunk `j` is complete the instant this pass retires its last column, so the
            # collective goes out here, inside the sweep, and the blocks still to come are
            # its cover. The final chunk's call is the one with no cover left; it is issued
            # below. Each physical core's operand is its own SBUF, so core `c` of the four
            # ranks reduces the vocabulary half core `c` computed, and the two halves never
            # have to be made visible to each other -- which is why there is no
            # `nisa.core_barrier` here and the `nl.shared_hbm` form of this seam needs one.
            if j + 1 < nchunk and r0 + c1 == bounds[j + 1]:
                ncc.all_reduce(srcs=[cc_src[j]], dsts=[cc_dst[j]],
                               replica_group=replica_group, op=nl.add)
                if EARLY_DMA:
                    _out_dma(dst=out3[core, :, base:bounds[j + 1]], src=cc_dst[j])

        r0 = r1
        bi += 1

    ncc.all_reduce(srcs=[cc_src[nchunk - 1]], dsts=[cc_dst[nchunk - 1]],
                   replica_group=replica_group, op=nl.add)

    # A collective may not write an IO tensor, so these DMAs are mandated rather than chosen.
    for _j in range(nchunk):
        if _j + 1 < nchunk and EARLY_DMA:
            continue
        _out_dma(dst=out3[core, :, bounds[_j]:bounds[_j + 1]], src=cc_dst[_j])

    # The embedding row needs no collective: this rank's 64 columns are already in `rowr`.
    # The two physical cores split the store so neither writes the other's bytes.
    half = dim // NCORE
    nl.store(embed[0:n_tokens, core * half:core * half + half],
             value=rowr[:, 0, core * half:core * half + half])

    return logits, embed


# Launched by `inference.py` as `kernel[NCORE]`: the grid is bound at the call site, not
# here, so the frozen validator is what pins the LNC=2 launch the collective needs.
kernel = nki.jit(_impl)
