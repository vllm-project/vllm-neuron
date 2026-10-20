# SPDX-License-Identifier: Apache-2.0
"""Vendored GPT-OSS chunked decode tail.

This file contains code from these nkilib modules, taken at nkilib mainline commit
``325d425c6580b6116126e903786477235bbc4403``:

* ``experimental.gpt_oss_tail.tail_chunked_megakernel``
* ``experimental.topk.chunked_topk``
* ``experimental.topk.gpsimd_topk``

The dependency image lacks the first two modules and ``snake_padded_n`` in its
``gpsimd_topk`` module.

The standalone top-k path still uses nkilib's ``gpsimd_topk``. The tail uses this
vendored copy. Do not mix a config constructor from one copy with a kernel from the
other copy. This module does not export the vendored constructor or kernel.

The vLLM integration uses batch groups of one. At this setting,
``owned_row_block`` remains the row-ownership contract. A multi-hop gather must
restore batch order. A single-hop gather is already in batch order.

The simulator cannot run the collective in this kernel. Accuracy tests need
hardware or a reference implementation.

Delete this file and import nkilib when the dependency provides both missing
modules and ``snake_padded_n``.
"""

from typing import Optional, Tuple
import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_verified_program_sharding_info,
)
import inspect
from dataclasses import dataclass
import numpy as np


# ======================================================================================
# gpsimd_topk -- vendored from nkilib experimental/topk
# ======================================================================================


"""GpSIMD top-k kernel using the single-instruction nisa.topk primitive.

Computes torch.topk(inp, k, dim=-1, largest=True) over inp[BxS, vocab] by mapping each
vocab row onto the 16-partition "snake" layout nisa.topk consumes (one top-k per group
of 16 partitions, 8 rows per 128-partition tile) with a blocked contiguous load, then
de-snaking the K (value, snake-position) pairs, optionally sorting them descending, and
remapping snake positions back to vocab indices on-chip.
"""


# SENTINELS. The kernel is MIXED PRECISION, and which sentinel is legal depends on the
# dtype of the buffer being written:
#     phase 1   src_snake ....... bfloat16   padding -inf; NaN folded to BFLOAT16_MIN
#     phase 2   sort buffers .... float32    padding FLOAT32_MIN; real -inf to NEG_INF_FLOOR
#     output    topk_values ..... bfloat16   (== inp.dtype)
# So NEG_INF_FLOOR is a FLOAT32-ONLY value despite the kernel's input and output being
# bfloat16: _clamp_sort_keys writes it from a float32 floor_tile into the float32 phase-2
# sort buffers, and it is never stored to a bfloat16 buffer. Phase 2 reloads through
# float32 (a bfloat16 -inf widens to float32 -inf exactly), clamps, then narrows once on
# the output store.

# Smallest finite bfloat16 value, used to pad snake slots that hold no real data.
BFLOAT16_MIN = -3.3895314e38  # most-negative bf16; padded slots never enter the top-k.
# Also the phase-1 NaN fold target. The asymmetry with NEG_INF_FLOOR below is forced by
# the dtype: bfloat16 has NO unrepresentable band to hide a sentinel in, so the most
# negative value phase 1 can write is this FINITE one, and it is therefore caller-visible
# (a NaN input slot comes back as BFLOAT16_MIN). That is accepted -- NaN has no ordering,
# so no correct value was available to return -- and the pairing check exempts exactly
# this substitution. Phase 2 needs no such compromise because float32 has the band.
# Most-negative float32, used to pad the sort buffer so padding never wins.
FLOAT32_MIN = float(np.finfo(np.float32).min)
# Sort-key floor for REAL -inf data, one float32 ULP above the FLOAT32_MIN padding
# sentinel. See _clamp_sort_keys for why the phase-2 sort needs a three-level ordering
#     FLOAT32_MIN (padding)  <  NEG_INF_FLOOR (real -inf)  <  every finite bf16
# and why raising real -inf to this particular value is NOT caller-visible: it lands in the
# band bfloat16 cannot represent finitely (bf16 saturates to -inf below about -3.3962e38,
# and this is -3.4028233e38, ~6.6e35 past it), so the float32 -> bfloat16 cast on the
# output store maps it back to EXACTLY -inf. A floor just above BFLOAT16_MIN instead stays
# finite in bfloat16 and leaks into the returned values.
NEG_INF_FLOOR = float(np.nextafter(np.float32(FLOAT32_MIN), np.float32(0)))

# nisa.topk runs one independent top-k per group of 16 partitions.
PARTS_PER_GROUP = 16
# Hardware max partition dimension (8 groups of 16).
PMAX = 128
GROUPS_PER_TILE = PMAX // PARTS_PER_GROUP  # 8


@dataclass(frozen=True, eq=True)
class GpsimdTopkConfig(nl.NKIObject):
    """Configuration for the GpSIMD nisa.topk kernel.

    Attributes:
        BxS: Combined batch*sequence dimension (number of rows).
        vocab_size: Length of the dimension reduced over (n for nisa.topk).
        k: Number of largest elements to return.
        sorted: Whether output must be in descending order.
        inp_dtype: Input/value data type (must be bfloat16).
        index_dtype: Output index data type (uint32).
        n_prgs: Number of logical cores (SPMD programs) over BxS.
        prg_id: Program id (set at config-build time for BxS == 1 path).
        per_lnc_BxS: Rows handled per logical core.
        out_shape: Logical output shape (mirrors input leading dims + k).
    """

    BxS: int
    vocab_size: int
    k: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    n_prgs: int
    prg_id: int
    per_lnc_BxS: int
    out_shape: tuple

    @property
    def topk_config(self):
        """Self-reference so the shared torch_ref (which reads .topk_config.k /
        .topk_config.sorted) works against this config without a separate object."""
        return self


def create_gpsimd_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    sorted: bool = True,
    num_programs: int = 2,
) -> GpsimdTopkConfig:
    """Build a GpsimdTopkConfig from an input shape (2D or 3D) and parameters."""
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    out_shape = tuple(list(inp_shape[:-1]) + [k])

    n_prgs = num_programs
    prg_id = 0
    if BxS == 1:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    return GpsimdTopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        n_prgs=n_prgs,
        prg_id=prg_id,
        per_lnc_BxS=per_lnc_BxS,
        out_shape=out_shape,
    )


def _invalid_desnake_col_runs(
    k: int, k_cols: int, k_pad: int
) -> Tuple[Tuple[int, int], ...]:
    """Contiguous runs of de-snaked cv columns that hold UNWRITTEN topk output slots.

    nisa.topk writes only snake positions [0, k); positions [k, k_pad) are unwritten
    and carry garbage. The de-snake places snake position pos at cv column
    j = (pos % 16) * k_cols + (pos // 16), i.e. cv column j holds snake position
    pos = (j // k_cols) + 16 * (j % k_cols). This returns the contiguous (start, length)
    runs of cv columns j whose pos >= k, so the kernel can memset each run to
    FLOAT32_MIN (a legal full-partition free-dim write) before the descending sort.

    All arguments are compile-time constants, so this resolves fully at trace time.
    Computed in a module-level helper (NOT inline in the @nki.jit body, where the
    parser frontend rejects the list-building expression as an "unsupported expression").
    """
    runs = []
    in_run = False
    run_start = 0
    run_len = 0
    for col_idx in range(k_pad):
        col_quotient, col_remainder = divmod(col_idx, k_cols)
        pos = col_quotient + PARTS_PER_GROUP * col_remainder
        if pos >= k:
            if in_run:
                run_len += 1
            else:
                in_run = True
                run_start = col_idx
                run_len = 1
        elif in_run:
            runs.append((run_start, run_len))
            in_run = False
            run_len = 0
    if in_run:
        runs.append((run_start, run_len))
    return tuple(runs)


def _valid_desnake_col_runs(
    k: int, k_cols: int, k_pad: int
) -> Tuple[Tuple[int, int, int], ...]:
    """Contiguous runs mapping VALID de-snaked cv columns to compacted output columns.

    The complement of _invalid_desnake_col_runs: cv column j holds snake-output
    position pos = (j // k_cols) + 16 * (j % k_cols), which is a real topk output iff
    pos < k. When k_pad > k (k % 16 != 0) the valid columns are NOT a contiguous prefix
    -- they are scattered among the k_pad columns, interleaved with the unwritten
    garbage columns. The unsorted fast path must COMPACT the valid columns to the
    contiguous output prefix [0, k). Each returned run is (src_col_start,
    dst_col_start, length): copy cv[:, src:src+length] -> out[:, dst:dst+length].

    For k % 16 == 0 (k_pad == k) the de-snake maps positions [0, k) onto cv columns
    [0, k), so the valid columns are already a contiguous prefix and this collapses to
    a SINGLE run (0, 0, k) (a full-width copy). Likewise for k_cols == 1 (k <= 16),
    where cv column j holds position j, so positions [0, k) occupy cv columns [0, k).

    All arguments are compile-time constants, so this resolves fully at trace time.
    Computed in a module-level helper (NOT inline in the @nki.jit body, where the
    parser frontend rejects the list-building expression as an "unsupported expression").
    """
    runs = []
    dst_col = 0
    in_run = False
    run_src_start = 0
    run_dst_start = 0
    run_len = 0
    for col_idx in range(k_pad):
        col_quotient, col_remainder = divmod(col_idx, k_cols)
        pos = col_quotient + PARTS_PER_GROUP * col_remainder
        if pos < k:
            if in_run:
                run_len += 1
            else:
                in_run = True
                run_src_start = col_idx
                run_dst_start = dst_col
                run_len = 1
            dst_col += 1
        elif in_run:
            runs.append((run_src_start, run_dst_start, run_len))
            in_run = False
            run_len = 0
    if in_run:
        runs.append((run_src_start, run_dst_start, run_len))
    return tuple(runs)


def _clamp_sort_keys(buf, n_parts: int, width: int) -> None:
    """Raise real -inf sort keys to NEG_INF_FLOOR so they cannot alias the sort marker.

    THE DEFECT. The phase-2 descending sort marks a slot it has already consumed by
    OVERWRITING it: ``nc_match_replace8(..., imm=float("-inf"))`` replaces the matched
    value with -inf. A real -inf in the DATA is then bit-identical to a consumed slot.
    match_replace8 reports the FIRST occurrence of each value max8 selected, so once any
    pass has stamped a marker at a lower buffer position, a later pass that legitimately
    selects a real -inf matches the STALE MARKER instead of the real element and reports
    its position. That position gathers the paired snake index, so the returned index
    points at a different element than the returned value: values[i, j] is -inf while
    input[i, indices[i, j]] is some unrelated element. The index stays IN RANGE, so no
    range check and no value-set comparison can see it -- only an elementwise
    value<->index pairing check. -inf is a legitimate logit (constrained and speculative
    decoding mask disallowed tokens to -inf), so this is not garbage-in-garbage-out: the
    input is orderable and the correct answer is well defined. A sampler consuming
    (probability, token_id) at slot j emits the WRONG TOKEN, silently.

    THE FIX. Move the real data OFF the marker value, into a sort key that is ordered
    correctly against everything else it must be compared with. The phase-2 sort needs a
    strict three-level ordering, and exactly one float32 value satisfies it:

        FLOAT32_MIN  <  NEG_INF_FLOOR  <  every finite bfloat16 value
        (padding)       (real -inf)        (real data)

      - ABOVE FLOAT32_MIN, the sentinel the de-snake padding columns are memset to, so
        an unwritten padding slot still loses to real -inf. This is why the clamp must
        run BEFORE the padding memset at each call site: clamping afterwards would raise
        the padding too and collapse the two levels.
      - BELOW BFLOAT16_MIN (-3.3895e38), the most negative FINITE bfloat16, so real -inf
        still loses to every real finite value and the k-largest SET is unchanged.
      - ABOVE -inf, so it is distinguishable from the match_replace8 marker. This is the
        whole point: the marker value no longer occurs in the data, so a stale marker can
        never be mistaken for a real element.

    WHY A FLOAT32 VALUE IN A BFLOAT16 KERNEL. ``buf`` is always one of the phase-2 FLOAT32
    sort buffers (sv / masked_v / cv), never the bfloat16 phase-1 snake, so NEG_INF_FLOOR is
    representable exactly where it is written. The bfloat16 legs are the phase-1 snake and
    the output store; the reload into phase 2 widens bfloat16 -> float32 (a bfloat16 -inf
    widens to float32 -inf exactly), and the narrowing happens once, on the store. That
    single narrowing cast is not a hazard to work around -- it is the mechanism the fix
    relies on, per the next paragraph.

    WHY THIS IS NOT CALLER-VISIBLE. Raising the sort key would normally corrupt the
    returned VALUE -- trading a wrong index for a wrong value, which is no fix at all.
    It does not here, because of where NEG_INF_FLOOR sits relative to bfloat16's dynamic
    range. The kernel's returned values are bfloat16, and bfloat16 saturates to -inf for
    anything below about -3.3962e38. NEG_INF_FLOOR is -3.4028233e38, comfortably past
    that threshold, so the float32 -> bfloat16 cast on the output store maps it back to
    EXACTLY -inf -- bit-identical to the true value. The raise is visible only inside the
    float32 sort buffers and is undone by the output cast.

    This is the specific reason a floor chosen just above BFLOAT16_MIN does NOT work:
    that lands in bfloat16's FINITE range, so it survives the output cast as
    -3.3895e38 and leaks into the returned values as a value the input never contained.
    The usable window is the part of (FLOAT32_MIN, BFLOAT16_MIN) that overflows bfloat16,
    and NEG_INF_FLOOR takes the most-negative end of it -- one ULP above the padding
    sentinel -- which maximises the margin to the bfloat16 saturation threshold.

    WHY THIS IS PREDICATED AND NOT A max(). The obvious spelling, a single
    ``tensor_scalar(op0=nl.maximum, operand0=NEG_INF_FLOOR)``, is WRONG: the hardware
    maximum returns the non-NaN operand when one side is NaN, so it silently rewrites
    every NaN in the buffer to NEG_INF_FLOOR. That destroys NaN input -- measured on
    trn3, it turned 32/512 and 256/4096 NaN slots into -inf values paired with NaN
    elements, i.e. it converted a clean case into a desync. Instead build an
    ``== -inf`` predicate and write the floor only there: NaN compares equal to nothing,
    so NaN slots are left exactly as they are, and finite slots are untouched.

    The predicate also makes the clamp ORDER-INDEPENDENT with respect to the padding
    memsets, since FLOAT32_MIN is finite and never matches ``== -inf``. The call sites
    still clamp before their memset, which keeps the ordering argument above local and
    obvious rather than relying on this.

    Cost: TWO Vector ops over the [n_parts, width] sort buffer per sort tile, hoisted OUT
    of the n_pass = ceil(k/8) max8 -> match_replace8 -> gather chain (32 passes at k=256,
    each scanning the full width). The sort loop is the measured bottleneck; this adds no
    per-pass work, so it is a fixed ~2/(3*32) of the sort's op count at k=256.
    """
    is_ninf = nl.ndarray((PMAX, width), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=is_ninf[nl.ds(0, n_parts), :],
        data=buf[nl.ds(0, n_parts), nl.ds(0, width)],
        op0=nl.equal,
        operand0=float("-inf"),
    )
    # ALSO fold NaN into the same floor. NaN is not merely an aliasing problem like
    # -inf: it is UNMATCHABLE. nc_match_replace8 finds a consumed slot by value
    # equality, and NaN never compares equal to itself, so a NaN that max8 selects can
    # never be matched and its reported dst_idx is UNDEFINED BY ISA CONTRACT -- for any
    # choice of imm. That undefined position is then gathered, which pairs the returned
    # value with an unrelated index (a value<->index desync) and, downstream, hands a
    # consumer an in-range-but-arbitrary index to dereference.
    #
    # Detected as (x != x), which is true only for NaN, and folded to NEG_INF_FLOOR so
    # every sort key is orderable and every max8 pick is matchable. NaN carries no
    # ordering information, so mapping it to the bottom of the range loses nothing that
    # was ever well defined -- there is no "k largest" when the input is not orderable.
    # It buys the property that DOES matter: the kernel returns an index that faithfully
    # pairs with the value it reports, for every input.
    is_nan = nl.ndarray((PMAX, width), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=is_nan[nl.ds(0, n_parts), :],
        data1=buf[nl.ds(0, n_parts), nl.ds(0, width)],
        data2=buf[nl.ds(0, n_parts), nl.ds(0, width)],
        op=nl.not_equal,
    )
    nisa.tensor_tensor(
        dst=is_ninf[nl.ds(0, n_parts), :],
        data1=is_ninf[nl.ds(0, n_parts), :],
        data2=is_nan[nl.ds(0, n_parts), :],
        op=nl.logical_or,
    )
    # tensor_copy_predicated's scalar-src form is documented but rejected by this
    # backend's validator ('float' object has no attribute 'shape'), so materialise the
    # floor as a full-width tile and copy from that.
    floor_tile = nl.ndarray((PMAX, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=floor_tile[nl.ds(0, n_parts), :], value=NEG_INF_FLOOR)
    nisa.tensor_copy_predicated(
        dst=buf[nl.ds(0, n_parts), nl.ds(0, width)],
        src=floor_tile[nl.ds(0, n_parts), :],
        predicate=is_ninf[nl.ds(0, n_parts), :],
    )


def snake_padded_n(n: int) -> int:
    """The ``n`` actually passed to ``nisa.topk`` for a ``n``-wide row: the 16-column snake
    rounds the row up to a whole column per partition, ``16 * ceil(n / 16)``.

    THE bound every validator must apply (here and in chunked_topk): the instruction's
    ``n < 65536`` holds for the PADDED size, so raw widths 65,521..65,535 -- legal-looking
    -- reach exactly 65,536 and die inside the ISA rather than at a named assert.
    """
    return PARTS_PER_GROUP * div_ceil(n, PARTS_PER_GROUP)


def _validate_gpsimd_topk(config: GpsimdTopkConfig) -> None:
    """Validate that the requested shape satisfies every nisa.topk constraint."""
    n = config.vocab_size
    k = config.k
    kernel_assert(
        config.inp_dtype == nl.bfloat16, "gpsimd_topk requires bfloat16 input"
    )
    kernel_assert(8 <= n, f"gpsimd_topk requires vocab ({n}) >= 8")
    kernel_assert(
        snake_padded_n(n) < 65536,
        f"gpsimd_topk requires the 16-padded snake width ({snake_padded_n(n)}) of vocab ({n}) to be < 65536",
    )
    kernel_assert(1 <= k < 32768, f"gpsimd_topk requires 1 <= k ({k}) < 32768")
    kernel_assert(k <= n, f"gpsimd_topk requires k ({k}) <= vocab ({n})")


@nki.jit
def gpsimd_topk(
    inp: nl.NkiTensor, config: GpsimdTopkConfig
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Top-k over the last dimension using the GpSIMD nisa.topk instruction.

    Dimensions:
        BxS: number of rows (flattened batch*sequence)
        V:   vocab size (reduction dimension), 8 <= V and 16*ceil(V/16) < 65536
        k:   number of largest elements, 1 <= k <= V

    Args:
        inp: [BxS, V] bfloat16 input tensor in HBM.
        config: GpsimdTopkConfig describing the problem and sharding.

    Returns:
        (topk_values [BxS, k] bf16, topk_indices [BxS, k] uint32), both in HBM.
        Descending along the last dim when config.sorted is True (default);
        arbitrary order within each row when config.sorted is False.

    Notes:
        - Each vocab row is loaded into its 16-partition nisa.topk snake with a
          BLOCKED / contiguous DMA: partition p reads the contiguous HBM run
          inp[row, p*n_cols:(p+1)*n_cols] (free-stride 1), so snake position
          s = p + 16*c holds vocab index p*n_cols + c. The 16-partition snake
          LAYOUT is mandated by nisa.topk, but the ORDER of the placement is a
          free bijection (any order yields the same top-k value set). The blocked
          order is chosen so the load is contiguous; the alternative "snake
          position i == vocab index i" fill gives an identity remap but forces a
          transpose-on-load (.ap [[1,16],[16,n]], free-stride 16 -> non-contiguous
          per partition), which is avoided here. The returned snake-position
          indices are remapped back to vocab space on-chip before the index store.
        - 8 rows (groups of 16 partitions) are processed per nisa.topk call.
        - The hardware nisa.topk output order is not relied upon: the K values +
          paired snake-position indices are de-snaked and then sorted descending
          on-chip; the snake->vocab index remap is applied to the final k indices.
        - Two-phase structure: phase 1 runs nisa.topk per 8-row tile and
          de-snakes the K (value, index) pairs into per-row HBM buffers; phase 2
          runs the descending sort ONCE over up to 128 rows (one row per
          partition) instead of once per 8-row tile. max8 / nc_match_replace8 /
          nc_n_gather are per-partition free-dim ops, so widening the sort from
          8 to up to 128 partitions is free Vector-engine parallelism and removes
          the redundant per-tile sort passes (the measured HW bottleneck).
        - config.sorted gates only the phase-2 descending sort. When False the
          sort is skipped and the K results are compacted to [0, k) in arbitrary
          order; the value set and value<->index pairing are unchanged.

    Pseudocode:
        n_cols = ceil(V / 16)            # blocked snake free width
        for tile in range(ceil(BxS / 8)):          # 8 rows per 128-partition tile
            snake[p, c] = inp[row, p*n_cols + c]   # blocked contiguous load
            val_snake, idx_snake = nisa.topk(snake, n=16*n_cols)  # snake positions
            asc_val[row], asc_idx[row] = de_snake(val_snake, idx_snake)  # to HBM
        for row_tile in range(ceil(BxS / 128)):    # up to 128 rows in parallel
            v, s = reload(asc_val, asc_idx)
            if config.sorted:
                v, s = sort_descending(v, s)         # max8 / match_replace8 / gather
            else:
                v, s = compact_valid_columns(v, s)   # drop padding cols, keep [0, k)
            idx = (s % 16) * n_cols + (s // 16)      # snake position -> vocab index
            topk_values[row_tile], topk_indices[row_tile] = v[:, :k], idx[:, :k]
        return topk_values, topk_indices
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k
    index_dtype = config.index_dtype

    # Resolve runtime sharding (LNC). BxS == 1 collapses to a single program.
    shard_info = get_verified_program_sharding_info("gpsimd_topk", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray((BxS, k), dtype=index_dtype, buffer=nl.shared_hbm)

    # Snake free-dim sizes (BLOCKED layout: snake[p, c] = inp[row, p*n_cols + c]).
    n_cols = div_ceil(vocab, PARTS_PER_GROUP)  # src free dim; >= ceil(n/16)
    full_n = (
        PARTS_PER_GROUP * n_cols
    )  # FULL snake size; this is the n passed to nisa.topk
    # Blocked-layout valid/padding boundary: partitions 0..q_full-1 are fully valid
    # (n_cols real vocab elements each), partition q_full holds the first r_partial
    # valid elements, and the rest are padding. r_partial == 0 (=> q_full == 16) when
    # 16 | vocab. Padding only exists when vocab % 16 != 0 (has_pad).
    q_full = vocab // n_cols  # number of fully-valid partitions
    r_partial = vocab % n_cols  # valid elements in the partial partition q_full
    has_pad = full_n != vocab  # True iff some snake slots are padding (vocab % 16 != 0)

    inp_flat = inp.reshape((BxS * vocab,))

    # The hardware nisa.topk output snake layout differs from a simple ascending
    # snake (and from the simulator), so the topk output ORDER is treated as
    # unspecified: only the SET of K values + their paired snake-position indices is
    # trusted. Per tile, the K outputs are de-snaked into a contiguous [rows, k_pad]
    # tile (via a private-HBM round trip with one proven 2D strided .ap per group),
    # sorted into descending order on-chip with the proven max8 / nc_match_replace8 /
    # nc_n_gather sequence, and finally the snake-position -> vocab-index remap (see
    # the index-remap block below) is applied to the k indices. The remap is NOT an
    # identity here because the blocked load places vocab index p*n_cols+c at snake
    # position p+16*c (not at position == vocab index).
    #
    # De-snake all k_cols snake columns (a full [16, k_cols] block) with one proven
    # 2D strided .ap per group; the buffers are padded to k_pad = k_cols*16 (>= k)
    # so the partial last column needs no special-case 1D .ap (a 1D .ap store was
    # observed to drop to zero on hardware). Both HBM buffers are float32: a strided
    # .ap SBUF->HBM store was observed to zero out for the narrow dtypes (bf16
    # values, uint32 indices), so the DMA auto-casts up to float32 (indices < 65536
    # are exact and cast back to uint32 on the final store).
    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP  # >= k; padded snake-column width

    # --- Fast-vs-safe de-snake DMA selection (correctness gate) ----------------
    # CORRECTNESS BUG (low/odd batch): the phase-1 de-snake stores to private_hbm and
    # the phase-2 reload of the SAME region were issued with dge_mode.none (descriptors
    # pre-generated out-of-band, store taken off the GpSIMD engine) AND asc_val_hbm was
    # bf16. On a PARTIAL phase-1 tile -- one with par_dim < 128 (n_rows < 8), e.g. a
    # single-row par_dim==16 tile produced by an ODD per_lnc_BxS shard -- that combo let
    # the reload RACE the store and/or the narrow-dtype strided store silently zero on
    # hardware, corrupting both values and indices. It only manifests in the fully
    # overlapped device run; the simulator and the serialized device-dump/debugger build
    # never reproduce it (every per-op SBUF dump is correct yet the device output is
    # wrong). All shards have ONLY full 128-partition tiles iff per_lnc_BxS is a multiple
    # of GROUPS_PER_TILE AND no shard is clamped (BxS == n_prgs * per_lnc_BxS); that is
    # the case for every batched/GPT-OSS shape. fast_dma_safe is that compile-time
    # predicate: when True keep the original fast path (dge_mode.none + bf16 value
    # buffer, halving the value-bounce bytes); when False (any partial/single-row tile)
    # use the proven-safe path -- default DMA mode (SWDGE keeps the store ordered with
    # its top-k producer and the reload) and a float32 value buffer (4-byte elements
    # round-trip correctly under the strided store; the reload casts bf16->f32 anyway).
    # Many-tile end of the SAME store/reload hazard the small/partial-tile case above
    # describes: under dge_mode.none the de-snake stores are unordered against their
    # consumer, and the number in flight grows as n_tiles = ceil(per_lnc_BxS / 8). An
    # end-to-end gpt-oss-120b decode run (bs4: 0/640 -> 640/640 requests) was fixed by
    # capping the unordered path at n_tiles < 8, but that cap is NOT the fix it was
    # believed to be, for two reasons found by the ordering audit:
    #
    #   1. Its stated mechanism is impossible as written. The claim was that the reload
    #      overtakes the store, so `pos` holds garbage and "the gather faults". `pos` is
    #      CLAMPED to [0, HALF-1] / [0, k_pad-1] at both nc_n_gather sites before it is
    #      ever used as a gather index, so a garbage `pos` cannot produce an
    #      out-of-bounds gather. Whatever the cap fixed, it was not that.
    #   2. It moves FIVE things at once. Because the single `fast_dma_safe` predicate was
    #      overloaded, lowering it flipped the de-snake DGE mode AND the value-buffer
    #      dtype AND three unrelated cross-engine SBUF sync gates (the valley sync, the
    #      out_i sync, and the A-half copy method). Any one of those could have been the
    #      actual repair, so the tile count is not established as the causal variable.
    #
    # The predicate is therefore SPLIT into the two independent concerns it conflated, and the
    # DGE mode is no longer one of them -- see THE STORE/RELOAD RACE IS REAL below.
    #
    # `fast_dma_safe` keeps its ORIGINAL meaning and its original definition: this
    # shard's phase-1 tiles are all full 128-partition tiles, which is what makes the
    # bf16 de-snake round trip safe against the narrow-dtype strided store. It no longer
    # selects the DGE mode. The empirical n_tiles < 8 cap stays REMOVED: it was a threshold
    # standing in for an ordering guarantee, and the ordering is now unconditional.
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (
        BxS == n_prgs * per_lnc_BxS
    )
    # --- THE STORE/RELOAD RACE IS REAL, AND NOT CONFINED TO PARTIAL TILES ---------------
    # dge_mode.unknown UNCONDITIONALLY (it used to be dge_mode.none whenever fast_dma_safe).
    # dge_mode.none pre-generates the de-snake descriptors out-of-band and takes the store off
    # the GpSIMD engine, which is exactly why nothing then orders that store against the
    # phase-2 reload of the SAME private_hbm region. The compiler-selected mode does preserve
    # that ordering. HWDGE cannot generate the strided de-snake descriptor pattern
    # ([NCC_IBIR098]) so it is never chosen here, and an explicitly-forced SWDGE was observed
    # to STILL drop values on the par_dim==16 case -- so let the compiler pick rather than
    # pinning a mode.
    #
    # MEASURED, at 128 ranks through the GPT-OSS tail (gbs=1024, two hops, 8 rows/rank):
    # 2 of 128 row blocks returned REPEATED vocab ids with the top-k VALUES exactly right;
    # with this one line changed, all four gates pass on all 128 ranks at gbs 128, 512 AND
    # 1024. The asymmetry is the tell: values and indices ride in SEPARATE private_hbm buffers
    # (asc_val_hbm, asc_idx_hbm) of different widths, so a race can take the index buffer while
    # the value buffer lands in time -- correct values carrying wrong, typically duplicated,
    # token ids, which is the failure mode a sampler cannot detect.
    #
    # Why it hid for so long, and why a tile-count threshold could never be the fix: exposure
    # tracks how busy the DMA subsystem is when phase 1 runs, not the tile count. Standalone the
    # same reduce geometry and tie density is clean over 544 independent blocks; at 32 ranks the
    # full tail dataflow is clean over 128 blocks (its de-interleave moves 64 KB against
    # 6.4 MB at 128 ranks); and at 128 ranks gbs 128 and 512 were clean while only gbs=1024
    # failed. Same code path in every one of those -- red_rows is 8 and fast_dma_safe is True at
    # BxS 1, 4 and 8 alike -- so nothing about the shape distinguishes them. Only the amount of
    # collective traffic landing just before the top-k does.
    #
    # COST: +2.7% on chunked_topk at the GPT-OSS vocab (A/B in one session, TpbSgCyclesSum:
    # +2.9% at BxS=1, +1.7% at BxS=4, +3.5% at BxS=8). That is the price of the ordering.
    #
    # If someone wants that back, the tempting refinement is to keep dge_mode.none on the STORE
    # and take the compiler's mode only for the RELOAD. Do NOT land that on the strength of a
    # green 128-rank run: this defect showed on 2 of 128 row blocks, so a variant that merely
    # narrows the window passes that test by luck. dge_mode.unknown is worth its 2.7% because it
    # is a categorical ordering guarantee rather than a statistical one, and the failure it
    # prevents is a silently wrong token id.
    desnake_dge = nisa.dge_mode.unknown
    # KNOWN-WEAKER INVARIANT: the bf16 fast-path de-snake store was originally validated
    # under the pinned dge_mode.none; with the mode now compiler-chosen, the bf16/mode
    # pairing's evidence is the 128-rank four-gate pass above plus the two-program
    # DMA-pressure perms in test_gpsimd_topk (the lnc=2 specdecode/-inf/padded rows exist
    # as the canary for exactly this). If a compiler update starts dropping the narrow
    # strided store again, flip this to unconditional float32 first (costs the halved
    # value-bounce bytes, changes no results) and bisect after.
    asc_val_dtype = nl.bfloat16 if fast_dma_safe else nl.float32
    asc_val_hbm = nl.ndarray((BxS, k_pad), dtype=asc_val_dtype, buffer=nl.private_hbm)
    asc_idx_hbm = nl.ndarray((BxS, k_pad), dtype=nl.float32, buffer=nl.private_hbm)
    asc_val_flat = asc_val_hbm.reshape((BxS * k_pad,))
    asc_idx_flat = asc_idx_hbm.reshape((BxS * k_pad,))

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    # This core's actual row range (clamped to BxS), used by both phases.
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # --- Coalesced + double-buffered phase-1 load (the load-cadence fix) ------
    # The per-tile blocked snake load was the phase-1 pacing item: 8 separate
    # HWDGE transfers, each ~1128-cyc issue interval, serialized one-per-tile on
    # the DMA queue while the ~100-cyc nisa.topk could not hide that latency.
    # Fix: COALESCE several tiles per load (fewer, larger HWDGE transfers so
    # load throughput, not per-tile latency, paces phase 1) while keeping
    # load<->compute OVERLAP via DOUBLE BUFFERING (a single whole-core DMA was
    # tried and regressed: it forced every topk to wait for the full load, the
    # opposite of overlap). When 16 | vocab and this core's n_tiles tiles are
    # ALL full (no per-tile clamping), each full tile is exactly 128*n_cols ==
    # 8*vocab contiguous HBM elements, and consecutive tiles are back-to-back, so
    # a chunk of CHUNK_TILES tiles is ONE contiguous HBM block loadable in a
    # single big HWDGE transfer into a [128, CHUNK_TILES, n_cols] SBUF buffer
    # (tile t-in-chunk at free offset t*n_cols). The blocked-layout offset for
    # (chunk-tile t, partition pp, col c) is
    #   base + t*8*vocab + (pp//16)*16*n_cols + (pp%16)*n_cols + c
    #     = base + t*(128*n_cols) + pp*n_cols + c      [vocab == 16*n_cols],
    # i.e. partition stride n_cols, tile stride 128*n_cols, free stride 1.
    # Two chunk buffers are alternated: while the GpSIMD top-k drains chunk c the
    # DMA prefetches chunk c+1, so loads run concurrently with compute and the
    # ~8 serialized per-tile completions become ceil(n_tiles/CHUNK_TILES) larger
    # overlapped transfers. Each tile's topk reads its slice
    # chunk_buf[:, t, :] (free-contiguous, identical layout to the old per-tile
    # src_snake), so the snake->vocab remap is unchanged.
    fast_coalesced = (
        (not has_pad)
        and (per_lnc_BxS % GROUPS_PER_TILE == 0)
        and (lnc_row_start + per_lnc_BxS <= BxS)
    )
    # Tiles coalesced into one DMA. 2 tiles == 128*2*n_cols bf16 (>=2KB/partition,
    # bandwidth-efficient) and ceil(n_tiles/2) chunks keeps the pipeline deep so
    # compute overlaps the next prefetch.
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    # Engines to round-robin the per-chunk HWDGE loads across, so consecutive
    # loads land on DIFFERENT DGE sequencer queues and run concurrently instead
    # of serializing one-after-another on a single sync queue (this is the
    # "split across more DMA queues" lever; lets the FIRST topk start sooner).
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    # N_PREFETCH_BUFS physically distinct buffers => up to N_PREFETCH_BUFS-1
    # chunks in flight (no anti-dependency forcing a later load to wait on an
    # earlier consumer's last read). 3 buffers keeps two loads in flight while
    # the GpSIMD top-k drains the third.
    N_PREFETCH_BUFS = min(3, n_chunks)
    chunk_bufs = []
    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(
                nl.ndarray(
                    (PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf
                )
            )
        # Prime the pipeline: prefetch the first N_PREFETCH_BUFS-1 chunks before the
        # tile loop starts (priming the multi-buffer pipeline), each a coalesced
        # HWDGE transfer (partition stride n_cols, tile stride 128*n_cols, free
        # stride 1) over its contiguous HBM block, round-robined across engines.
        # The last chunk may hold fewer than CHUNK_TILES tiles; clamp the tile count
        # so the DMA never over-reads past this core's region.
        n_prime = min(N_PREFETCH_BUFS - 1, n_chunks)
        for pc in range(n_prime):
            prime_first_tile = pc * CHUNK_TILES
            prime_row_start = lnc_row_start + prime_first_tile * GROUPS_PER_TILE
            prime_tiles = min(CHUNK_TILES, n_tiles - prime_first_tile)
            nisa.dma_copy(
                dst=chunk_bufs[pc % N_PREFETCH_BUFS][
                    nl.ds(0, PMAX), nl.ds(0, prime_tiles), nl.ds(0, n_cols)
                ],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, prime_tiles], [1, n_cols]],
                    offset=prime_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[pc % 2],
            )

    # =====================================================================
    # PHASE 1 -- per-8-row tile: blocked load -> nisa.topk -> de-snake the
    # K (value, snake-index) pairs into the contiguous [BxS, k_pad] HBM
    # buffers asc_val_hbm / asc_idx_hbm. The expensive descending sort is
    # NOT done here: it runs ONCE in phase 2 over all this core's rows
    # (one row per partition, up to 128) instead of n_tiles serial sorts
    # that each used only n_rows (<= 8) of the 128 partitions. The sort is a
    # per-partition free-dim operation, so widening it from 8 to up to 128
    # partitions is free Vector-engine parallelism and removes the (n_tiles-1)
    # redundant serial sort passes -- the measured HW bottleneck.
    # =====================================================================
    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        # Rows handled by this tile, clamped to this core's shard and to BxS.
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue

        par_dim = n_rows * PARTS_PER_GROUP  # multiple of 16

        # --- Blocked / contiguous load of each row into its 16-partition snake ---
        # snake[base + p, c] = inp[row, p*n_cols + c]: partition p reads the
        # CONTIGUOUS HBM run inp[row, p*n_cols:(p+1)*n_cols] (.ap partition-stride
        # n_cols, free-stride 1). Snake position s = p + 16*c therefore holds vocab
        # index p*n_cols + c; the returned indices are remapped back to vocab space
        # after the sort (see the index-remap block below). The snake LAYOUT is
        # required by nisa.topk; this blocked placement ORDER is a free choice (any
        # bijection gives the same top-k value set) picked for a contiguous load.
        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            # Keep the pipeline (N_PREFETCH_BUFS-1 chunks deep) full: at the FIRST tile
            # of the current chunk, prefetch the chunk that is (N_PREFETCH_BUFS-1) ahead
            # into its rolling buffer slot, so its (coalesced) HWDGE load overlaps the
            # GpSIMD top-k of the chunks already resident. The destination slot differs
            # from every slot still being consumed, so there is no anti-dependency that
            # forces the load to wait. Loads round-robin across two DGE queues so two
            # are in flight concurrently.
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                # Clamp to the tiles actually present (last chunk may be partial) so the
                # prefetch never over-reads past this core's region.
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                nisa.dma_copy(
                    dst=chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS][
                        nl.ds(0, PMAX), nl.ds(0, pf_tiles), nl.ds(0, n_cols)
                    ],
                    src=inp_flat.ap(
                        pattern=[
                            [n_cols, PMAX],
                            [PMAX * n_cols, pf_tiles],
                            [1, n_cols],
                        ],
                        offset=pf_row_start * vocab,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[prefetch_chunk % 2],
                )
            # This tile views its [128, n_cols] slice of the prefetched chunk buffer
            # (free-contiguous, identical to the old per-tile src_snake). par_dim == PMAX.
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][
                nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)
            ]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # FAST PATH FALLBACK (16 | vocab but the coalesced precondition does not
            # hold, e.g. a partial last tile or a clamped shard). The whole-tile
            # blocked load is ONE contiguous HBM run: for partition pp = g*16 + p the
            # source offset is row*vocab + p*n_cols
            #   = (tile_row_start + pp//16)*vocab + (pp%16)*n_cols
            #   = tile_row_start*vocab + (pp//16)*16*n_cols + (pp%16)*n_cols   [vocab=16*n_cols]
            #   = tile_row_start*vocab + n_cols*(16*(pp//16) + pp%16)
            #   = tile_row_start*vocab + n_cols*pp,
            # i.e. partition stride n_cols, free stride 1 over the contiguous block
            # inp_flat[tile_row_start*vocab : tile_row_start*vocab + par_dim*n_cols].
            # Emitted as a SINGLE par_dim-partition DMA per tile. HWDGE on the Sync
            # engine keeps descriptor generation OFF GpSIMD so the load overlaps the
            # previous tile's top-k. No memset (every snake slot is a real vocab elt).
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, par_dim], [1, n_cols]],
                    offset=tile_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # vocab % 16 != 0: snake positions whose vocab index >= vocab are padding
            # (partition q_full's tail, plus any fully-padded higher partitions).
            # memset the whole tile so every padded slot loses the top-k; the valid
            # contiguous prefix is then DMA'd over the top.
            #
            # The sentinel is -inf, NOT the most-negative FINITE bf16 (BFLOAT16_MIN).
            # BFLOAT16_MIN is a SECOND value<->index defect, independent of the phase-2
            # marker collision: being finite, it is strictly GREATER than a real -inf
            # logit, so on a row where masking pushed real tokens to -inf every padding
            # slot outranks the real data and nisa.topk pulls padding into the top-k. The
            # kernel then returns BFLOAT16_MIN -- a value the input never contained -- and
            # an index that the final [0, vocab) clamp pins to vocab-1, so the pair is
            # (value not in input, index of an unrelated element). Measured on trn3 at
            # vocab 3142 and 1022: values=[-3.3895e38] vs input[indices]=[-inf].
            #
            # -inf as the sentinel restores the invariant that padding can never outrank
            # real data: it is <= every real value including -inf itself. When it ties
            # real -inf and a padding slot is still selected, the returned value is -inf
            # and the clamped index addresses a real element that is ALSO -inf, so the
            # value is right and the pairing holds. For all finite input nothing changes,
            # since both sentinels already lost to every finite value.
            nisa.memset(dst=src_snake, value=float("-inf"))

            for g in nl.affine_range(n_rows):
                row = tile_row_start + g
                base = g * PARTS_PER_GROUP
                row_off = row * vocab
                # Bulk: the q_full fully-valid partitions in one contiguous DMA. Reads
                # inp_flat[row_off : row_off + q_full*n_cols]; q_full*n_cols <= vocab so
                # there is no over-read past this row.
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(
                            pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off
                        ),
                    )
                # Tail: partition q_full holds the first r_partial valid elements (only
                # present when vocab % 16 != 0). One tiny contiguous DMA into a single
                # partition; reads inp_flat[row_off + q_full*n_cols : row_off + vocab].
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(
                            pattern=[[1, r_partial]], offset=row_off + q_full * n_cols
                        ),
                    )

        # --- Run the GpSIMD top-k over the FULL snake (n = full_n) ---
        # nisa.topk derives K from val_dst free dim, so val/idx must be [par_dim, k];
        # allocate the padded width k_pad so the partial last column is fully present
        # and de-snakeable with the 2D .ap below.
        # n = full_n (NOT vocab): with the blocked layout the valid vocab elements are
        # not a contiguous prefix of snake positions, so all full_n snake slots are
        # active. Padded slots hold BFLOAT16_MIN and never win the top-k.
        # Fold NaN out of the snake BEFORE nisa.topk selects from it. This is the
        # earliest point NaN can do damage and the phase-2 sort-key clamps are far too
        # late: nisa.topk already picked its k winners and emitted their paired snake
        # positions, so a NaN that confused the selection has ALREADY produced a
        # (value, position) pair that does not correspond. Folding here means every
        # value nisa.topk ranks is orderable, so the pairing it emits is meaningful and
        # every downstream match_replace8 lookup is matchable.
        #
        # (x != x) is true only for NaN. Fold to BFLOAT16_MIN -- the same value the
        # padding memset uses -- so a NaN slot simply loses the top-k like padding does.
        #
        # NOT NEG_INF_FLOOR: src_snake is BFLOAT16, and the floor's whole property is that
        # bfloat16 cannot hold it finitely, so storing it here would saturate straight back
        # to -inf. BFLOAT16_MIN is the most negative value this dtype can express. That is
        # sufficient for phase 1, where the requirement is only that every key nisa.topk
        # ranks be ORDERABLE -- there is no match_replace8 marker yet to collide with, which
        # is the separate problem NEG_INF_FLOOR exists to solve in phase 2.
        # NaN carries no ordering information, so there is nothing well defined to lose:
        # with NaN present there is no "k largest". What this buys is the property that
        # matters to every consumer: the returned index faithfully pairs with the
        # returned value, so an index handed to a downstream table gather is the one the
        # kernel actually selected rather than an arbitrary in-range value.
        _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=_nan_mask[nl.ds(0, par_dim), :],
            data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
            data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
            op=nl.not_equal,
        )
        _nan_floor = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=_nan_floor[nl.ds(0, par_dim), :], value=BFLOAT16_MIN)
        nisa.tensor_copy_predicated(
            dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
            src=_nan_floor[nl.ds(0, par_dim), :],
            predicate=_nan_mask[nl.ds(0, par_dim), :],
        )

        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(
            val_dst=val_snake[:, nl.ds(0, k)],
            idx_dst=idx_snake[:, nl.ds(0, k)],
            src=src_snake,
            n=full_n,
        )
        # When k_pad > k the topk leaves the unwritten snake positions [k, k_pad) with
        # garbage. Those slots live at val_snake[pos % 16, pos // 16] (column pos // 16 <
        # k_cols, partition pos % 16) -- a PARTITION-SUBRANGE that a phase-1 full-partition
        # memset cannot target (and a partition-subrange memset is rejected by the backend).
        # So the padding is NOT masked here; each phase-2 sort path masks it after the
        # de-snake reload, where the [n_srows, k_pad] layout makes it a legal full-partition
        # free-dim write: the full-width path memsets the invalid cv columns, the split path
        # masks + re-splits a full-width reload, and the unsorted path compacts only the valid
        # columns. (k % 16 == 0 -> k_pad == k -> no padding, so all gptoss configs skip it.)

        # --- De-snake the topk output set into a CONTIGUOUS [rows, k_pad] HBM tile ---
        # asc[row, p*k_cols + c] = val_snake[base + p, c]: partition p's k_cols outputs
        # land at HBM offset row*k_pad + p*k_cols, free-stride 1. Because k_pad ==
        # PARTS_PER_GROUP*k_cols, consecutive row-groups (16 partitions each) are exactly
        # k_pad apart, so the WHOLE par_dim-partition tile maps to ONE contiguous HBM run
        # [tile_row_start*k_pad : tile_row_start*k_pad + par_dim*k_cols] with partition
        # stride k_cols, free stride 1. Emitting it as a SINGLE par_dim-partition DMA per
        # tile (instead of n_rows per-row [16,k_cols] DMAs) cuts the de-snake DMA
        # instruction/descriptor count by ~8x -- the de-snake stores (lines 334/338) were
        # the top serial contributors in the HW profile (each Count=64, ~58.9k+39.5k
        # non-overlapped ns on the critical path). The flattened DMA packs all par_dim*k_cols
        # elements behind one descriptor stream. Row ORDER within a row is unspecified (the
        # phase-2 sort re-derives descending order; the per-slot value<->snake-position
        # pairing is preserved, so the snake->vocab remap is still correct). Unwritten
        # padding slots (k_pad > k) carry garbage here and are masked per-path after the
        # phase-2 reload (see the top-k call comment above).
        # The de-snake stores are STRIDED SBUF->HBM writes (.ap partition-stride k_cols).
        # HWDGE cannot generate descriptors for that pattern ([NCC_IBIR098]). The mode is
        # desnake_dge = dge_mode.unknown, UNCONDITIONALLY -- see the desnake_dge comment
        # above for the measured silent-index-corruption story and why no faster mode may
        # be restored here (dge_mode.none's out-of-band descriptors drop the ordering
        # against the phase-2 reload of this same private_hbm region).
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=asc_val_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=asc_idx_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    # =====================================================================
    # PHASE 2 -- batch-parallel descending sort. Each row's K (value, snake
    # index) pairs occupy one partition's free dim (k_pad wide). max8 /
    # nc_match_replace8 / nc_n_gather are per-partition free-dim operations,
    # so one invocation over up to 128 partitions sorts up to 128 rows at the
    # same wall-clock cost as one row. Replacing the n_tiles serial 8-row
    # sorts with ceil(per_lnc_BxS / 128) sorts removes the redundant passes
    # that dominated the measured HW profile. The per-partition math is
    # IDENTICAL to the old per-tile sort (independent rows, no cross-row
    # interaction), so correctness is unchanged.
    # =====================================================================
    n_pass = div_ceil(k, 8)
    log2_ppg = 4  # == log2(PARTS_PER_GROUP), PARTS_PER_GROUP == 16
    n_sort_tiles = div_ceil(per_lnc_BxS, PMAX)

    # --- Half-width split-sort + bitonic-merge gate ----------------------------
    # The descending sort is the measured HW bottleneck: a serial chain of
    # n_pass = ceil(k_pad/8) (=32 for k_pad=256) max8->match_replace8 passes, each
    # scanning the FULL k_pad-wide buffer. max8 / match_replace8 latency is
    # proportional to the per-partition scan width (Formula A, Vector engine), so
    # cutting both the pass count AND the width is a ~4x win on the sort. Phase 2
    # for this shape uses only 64 of 128 partitions (n_srows = per_lnc_BxS = 64),
    # so the idle 64 partitions are free parallelism: place each row's LOW half on
    # partition r and its HIGH half on partition r+n_srows, sort each half
    # descending in n_pass_h = ceil(HALF/8) (=16) passes over width HALF (=128) in
    # PARALLEL across all 2*n_srows partitions, then MERGE the two sorted halves
    # into one descending k_pad-row with a bitonic merge (log2(k_pad)=8 stages of
    # cheap select compare-exchanges that carry the paired indices). The merge
    # input is the "valley" [A_desc | reverse(B_desc)]; reverse(B) is one gather.
    # Requires k_pad a power of two, k_pad even, and 2*n_srows <= 128 so both
    # halves fit on distinct partitions.
    HALF = k_pad // 2
    use_split = (
        (k_pad >= 16)
        and (k_pad % 2 == 0)
        and ((k_pad & (k_pad - 1)) == 0)  # power of two -> valid bitonic length
        and (2 * per_lnc_BxS <= PMAX)
        # The merge writes a full k_pad-wide result into out_v/out_i (width
        # n_pass*8 == ceil(k/8)*8). For small k, k_pad = ceil(k/16)*16 can exceed
        # n_pass*8 (e.g. k<=8 -> k_pad=16 > n_pass*8=8), overrunning those buffers.
        # Only take the split path when the result fits; tiny k falls through to the
        # full-width sort below (which writes out_v only up to n_pass*8).
        and (k_pad <= n_pass * 8)
    )

    for sort_idx in nl.sequential_range(n_sort_tiles):
        sort_row_start = lnc_row_start + sort_idx * PMAX
        sort_row_end = min(sort_row_start + PMAX, lnc_row_end)
        n_srows = sort_row_end - sort_row_start
        if n_srows <= 0:
            continue

        out_v = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        out_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)

        # config.sorted gates ONLY the descending sort. Both paths share the reload
        # (mandatory), the snake->vocab index remap (below), and the store. When
        # sorted is False the K results are emitted in arbitrary order: reload, COMPACT
        # the valid de-snaked columns to the output prefix [0, k), and skip the sort.
        if config.sorted and use_split and (2 * n_srows <= PMAX):
            # ===== Half-width split-sort + bitonic merge ======================
            n_pass_h = div_ceil(HALF, 8)
            two = 2 * n_srows  # partitions used: A on [0:n_srows), B on [n_srows:2n)
            # Split-load: partition r holds row r's LOW half [0:HALF]; partition
            # r+n_srows holds row r's HIGH half [HALF:k_pad]. This reload reads the SAME
            # private_hbm region the phase-1 de-snake just wrote, so it uses desnake_dge
            # (= dge_mode.unknown, unconditional -- see its definition comment): any mode
            # with out-of-band descriptors drops the store->reload ordering and lets this
            # reload race the de-snake store, reading stale/zeroed HBM on device only.
            sv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            si = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=sv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), nl.ds(0, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=sv[nl.ds(n_srows, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), nl.ds(HALF, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=si[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), nl.ds(0, HALF)],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=si[nl.ds(n_srows, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), nl.ds(HALF, HALF)],
                dge_mode=desnake_dge,
            )
            # Lift real -inf sort keys off the match_replace8 marker value (see
            # _clamp_sort_keys). Covers the k_pad == k case, where sv is the buffer the
            # half-sort actually reads; when k_pad > k the re-split below overwrites sv
            # from masked_v, which is clamped separately at its own reload.
            _clamp_sort_keys(sv, two, HALF)
            # Mask the de-snaked padding (unwritten snake positions [k, k_pad)) before the
            # half-sort. The phase-1 val_snake[:, k:k_pad] memset does NOT cover it: that
            # writes snake COLUMNS [k, k_pad), but the de-snake reads columns [0, k_cols) and
            # the unwritten slots live at column pos//16 (< k_cols), partition pos%16 -- a
            # partition-subrange the phase-1 full-partition memset cannot target. Left
            # unmasked, those slots carry garbage (often large positive) that wins max8 in the
            # half-sort and corrupts the merged top-k VALUES (observed for padded split k such
            # as 10/12/60 on device). The invalid columns cannot be masked in the split sv/si
            # halves directly: they land on the HIGH-half partitions only, and a
            # partition-subrange memset is rejected by the backend ([NCC_INLA001] "Invalid
            # access of N partitions starting at partition P"). Instead mask them the same way
            # as the full-width path: memset the invalid columns of the FULL-width reload
            # (a legal full-partition free-dim write on masked_v[0:n_srows, :]) and split from
            # that masked buffer via SBUF->SBUF copies. k % 16 == 0 (no padding) skips this;
            # all fast_dma_safe/GPT-OSS shapes have k_pad == k so this is a no-op there.
            if k_pad > k:
                masked_v = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=masked_v[nl.ds(0, n_srows), :],
                    src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                    dge_mode=desnake_dge,
                )
                # Clamp BEFORE the padding memset: the clamp must not raise the padding
                # sentinel, or FLOAT32_MIN (padding) and NEG_INF_FLOOR (real -inf) would
                # collapse to one level and a padding slot could tie real -inf again.
                _clamp_sort_keys(masked_v, n_srows, k_pad)
                _pad_runs = _invalid_desnake_col_runs(k, k_cols, k_pad)
                for _run_idx in range(len(_pad_runs)):
                    _col_start = _pad_runs[_run_idx][0]
                    _col_len = _pad_runs[_run_idx][1]
                    nisa.memset(
                        dst=masked_v[:, nl.ds(_col_start, _col_len)], value=FLOAT32_MIN
                    )
                # Re-split the masked full-width values into the two half-partition layout
                # (low half [0:HALF] -> partitions [0:n_srows); high half [HALF:k_pad] ->
                # [n_srows:2n)). The low half stays on the same partitions (tensor_copy); the
                # high half shifts partitions [0:n_srows) -> [n_srows:2n), which is a
                # partition move only DMA can express (tensor_copy is per-lane). si is left as
                # loaded: invalid columns carry garbage indices but their paired values are now
                # FLOAT32_MIN, so they lose the sort and are never gathered into the output.
                nisa.tensor_copy(
                    dst=sv[nl.ds(0, n_srows), :],
                    src=masked_v[nl.ds(0, n_srows), nl.ds(0, HALF)],
                )
                nisa.dma_copy(
                    dst=sv[nl.ds(n_srows, n_srows), :],
                    src=masked_v[nl.ds(0, n_srows), nl.ds(HALF, HALF)],
                )

            # Sort each partition's HALF values descending (16 passes, width HALF,
            # 2*n_srows partitions in parallel). hv = sorted values, hsnake =
            # gathered snake-positions (still float32) in the same descending order.
            hv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            hsnake = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            for ps in nl.sequential_range(n_pass_h):
                cur = nl.ds(ps * 8, 8)
                pos = nl.ndarray((PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.max8(dst=hv[nl.ds(0, two), cur], src=sv[nl.ds(0, two), :])
                nisa.nc_match_replace8(
                    dst=sv[nl.ds(0, two), :],
                    dst_idx=pos[nl.ds(0, two), :],
                    data=sv[nl.ds(0, two), :],
                    vals=hv[nl.ds(0, two), cur],
                    imm=float("-inf"),
                )
                # Bound the gather index. nc_match_replace8 reports the position of the
                # FIRST OCCURRENCE of each value from max8; when a value cannot be
                # matched -- NaN never compares equal to itself, and duplicate extremes
                # let one pass consume a slot a later pass still searches for -- the
                # corresponding dst_idx is undefined. nc_n_gather requires
                # indices < data.size / data.shape[0] and does not check, so an
                # undefined position becomes an out-of-bounds indirect DGE access that
                # faults the device. Garbage logits (NaN/Inf mixed with huge finite
                # values) reach here in practice, so clamp rather than assume.
                nisa.tensor_scalar(
                    dst=pos[nl.ds(0, two), :],
                    data=pos[nl.ds(0, two), :],
                    op0=nl.minimum,
                    operand0=HALF - 1,
                )
                nisa.nc_n_gather(
                    dst=hsnake[nl.ds(0, two), cur],
                    data=si[nl.ds(0, two), :],
                    indices=pos[nl.ds(0, two), :],
                )

            # --- Co-locate the two sorted halves on one partition + build valley ---
            # mv[r, 0:HALF]   = A_desc      = hv[r, :]
            # mv[r, HALF:k]   = reverse(B_desc) = hv[r+n_srows, :] reversed
            # The reversed HIGH half makes [A_desc | B_asc] a valley (bitonic).
            # Bring partitions [n_srows:2n) down to [0:n_srows) via SBUF->SBUF DMA,
            # then reverse on the free dim with one nc_n_gather (reversed indices).
            mv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            mi = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # A half (already on partitions [0:n_srows)). Both halves go through a
            # synchronizing SBUF->SBUF DMA, UNCONDITIONALLY -- formerly this was gated on
            # fast_dma_safe, with a tensor_copy for the value half on full-tile shards.
            #
            # hsnake is written column-by-column by the per-pass nc_n_gather (GpSIMD); a
            # tensor_copy reading the full [0:HALF] width races the LAST gather pass and
            # reads its tail columns as stale zeros -> snake position 0 -> vocab index 0 at
            # the row tail, values left correct. First attributed to n_srows == 1 only, then
            # measured on trn3 at n_srows == 8 (BxS=16, lnc=2, v=3142, k=256: 63 mis-paired
            # slots, and only when run alongside other shapes -- the signature of a timing
            # race, not a data-dependent bug). So the shape gate was already known-wrong for
            # the index half.
            #
            # The VALUE half is now routed the same way, because its old justification does
            # not hold either. It read: "hv is written by max8 (Vector), the same engine that
            # reads it here, so there is no cross-engine hazard." The premise is false --
            # nisa.tensor_copy takes engine=unknown by default, i.e. the COMPILER selects
            # among Vector/Scalar/GpSimd based on engine workload (see nki/isa/_copy.py).
            # Nothing pins this copy to Vector, so "the same engine" is an assumption the
            # code never enforces, and if the compiler places it on Scalar or GpSimd the
            # value half has exactly the cross-engine hazard the index half has.
            #
            # Using DMA rather than pinning engine=vector is deliberate: max8 /
            # nc_match_replace8 are Vector and the sort is the measured bottleneck, so
            # forcing these copies onto Vector would add work to the critical engine. The
            # DMA keeps them off all three compute engines. This is also what the
            # not-fast_dma_safe path always did, so it is the already-proven spelling.
            nisa.dma_copy(
                dst=mv[nl.ds(0, n_srows), nl.ds(0, HALF)], src=hv[nl.ds(0, n_srows), :]
            )
            nisa.dma_copy(
                dst=mi[nl.ds(0, n_srows), nl.ds(0, HALF)],
                src=hsnake[nl.ds(0, n_srows), :],
            )
            # B half: shift partitions [n_srows:2n) -> a contiguous [0:n_srows) scratch.
            bv = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            bi = nl.ndarray((PMAX, HALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=bv[nl.ds(0, n_srows), :], src=hv[nl.ds(n_srows, n_srows), :]
            )
            nisa.dma_copy(
                dst=bi[nl.ds(0, n_srows), :], src=hsnake[nl.ds(n_srows, n_srows), :]
            )
            # Reversed-index constant [n_srows, HALF] = HALF-1, HALF-2, ..., 0.
            rev_idx = nl.ndarray((PMAX, HALF), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.iota(
                dst=rev_idx[nl.ds(0, n_srows), :], pattern=[[-1, HALF]], offset=HALF - 1
            )
            nisa.nc_n_gather(
                dst=mv[nl.ds(0, n_srows), nl.ds(HALF, HALF)],
                data=bv[nl.ds(0, n_srows), :],
                indices=rev_idx[nl.ds(0, n_srows), :],
            )
            nisa.nc_n_gather(
                dst=mi[nl.ds(0, n_srows), nl.ds(HALF, HALF)],
                data=bi[nl.ds(0, n_srows), :],
                indices=rev_idx[nl.ds(0, n_srows), :],
            )

            # --- Bitonic merge (descending) of the valley sequence -------------
            # For stride s in [HALF, HALF/2, ..., 1]: view free dim as
            # [k_pad/(2s)] blocks of [2, s]; compare lo=block[:,0,:] with
            # hi=block[:,1,:], keep the larger in lo (descending), carry indices
            # with the SAME predicate (select). PING-PONG between two physical
            # buffers each stage (read src strided blocks, write dst strided blocks
            # DIRECTLY) so no per-stage write-back copies are needed -- the value
            # max/min and the index selects write straight into the destination
            # buffer's strided views. log2(HALF)+1 = 8 stages for k_pad=256 (even),
            # so the final result lands back in mv/mi. Per stage: 1 pred + 2 value
            # (max,min) + 4 index (copy-on_false + predicated-copy-on_true x2) = 7
            # ops, down from 11 (the 4 write-backs are eliminated).
            mv2 = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            mi2 = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # Order the valley writes before the merge reads them. mv[:, HALF:] /
            # mi[:, HALF:] are written by the reverse-gather (nc_n_gather, GpSIMD engine)
            # just above; the merge's first stage reads the full valley via a reshape view
            # on a DIFFERENT engine (tensor_tensor, Vector). That cross-engine read was
            # observed on hardware to RACE the reverse-gather and read the high half as
            # stale zeros, dropping the whole B (high) half -- including the global max --
            # in the first max/min (device-only; the simulator serializes and never
            # reproduced it). Route the valley through a synchronizing SBUF->SBUF DMA into
            # the merge's source buffers so stage 0 reads the settled valley.
            #
            # UNCONDITIONAL, formerly `if not fast_dma_safe`. The gate was unsound. It
            # encoded a belief that the race needs a small/partial tile (n_srows < 8), but
            # the producer/consumer pair here does not depend on n_srows at all: the
            # gather writes mv/mi[:, HALF:] on GpSIMD and the merge reads them on Vector at
            # EVERY n_srows, and widening n_srows makes the gather take LONGER, which
            # widens the window rather than closing it. The identical belief about the
            # sibling `hsnake` pair -- documented as racing "only at n_srows == 1" -- was
            # measured racing at n_srows == 8, which is direct evidence that
            # "n_srows >= 8 never showed the race" means only that no test had looked.
            # "Never observed" on shapes whose sync was never removed for a controlled A/B
            # is not evidence of safety. Cost is two SBUF->SBUF DMAs per sort tile.
            mv_sync = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            mi_sync = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=mv_sync[nl.ds(0, n_srows), :], src=mv[nl.ds(0, n_srows), :]
            )
            nisa.dma_copy(
                dst=mi_sync[nl.ds(0, n_srows), :], src=mi[nl.ds(0, n_srows), :]
            )
            mv, mi = mv_sync, mi_sync
            src_v, src_i, dst_v, dst_i = mv, mi, mv2, mi2
            s = HALF
            while s >= 1:
                ng = k_pad // (2 * s)
                sv3 = src_v.reshape((PMAX, ng, 2, s))
                si3 = src_i.reshape((PMAX, ng, 2, s))
                dv3 = dst_v.reshape((PMAX, ng, 2, s))
                di3 = dst_i.reshape((PMAX, ng, 2, s))
                lo_v = sv3[nl.ds(0, n_srows), :, 0, :]
                hi_v = sv3[nl.ds(0, n_srows), :, 1, :]
                lo_i = si3[nl.ds(0, n_srows), :, 0, :]
                hi_i = si3[nl.ds(0, n_srows), :, 1, :]
                d_lo_v = dv3[nl.ds(0, n_srows), :, 0, :]
                d_hi_v = dv3[nl.ds(0, n_srows), :, 1, :]
                d_lo_i = di3[nl.ds(0, n_srows), :, 0, :]
                d_hi_i = di3[nl.ds(0, n_srows), :, 1, :]
                # predicate m = (lo_v >= hi_v): keep lo when lo already larger.
                pred = nl.ndarray((PMAX, ng, s), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=pred[nl.ds(0, n_srows), :, :],
                    data1=lo_v,
                    data2=hi_v,
                    op=nl.greater_equal,
                )
                # Values: lo' = max, hi' = min (write straight into dst strided blocks).
                nisa.tensor_tensor(dst=d_lo_v, data1=lo_v, data2=hi_v, op=nl.maximum)
                nisa.tensor_tensor(dst=d_hi_v, data1=lo_v, data2=hi_v, op=nl.minimum)
                # Indices follow the value winner: d_lo_i = pred?lo_i:hi_i,
                # d_hi_i = pred?hi_i:lo_i. tensor_copy_predicated only overwrites
                # where pred, so seed with the on_false operand first.
                nisa.tensor_copy(dst=d_lo_i, src=hi_i)
                nisa.tensor_copy_predicated(
                    dst=d_lo_i, src=lo_i, predicate=pred[nl.ds(0, n_srows), :, :]
                )
                nisa.tensor_copy(dst=d_hi_i, src=lo_i)
                nisa.tensor_copy_predicated(
                    dst=d_hi_i, src=hi_i, predicate=pred[nl.ds(0, n_srows), :, :]
                )
                src_v, dst_v = dst_v, src_v
                src_i, dst_i = dst_i, src_i
                s //= 2

            # After log2(HALF)+1 stages the result is in src_v / src_i (post-swap).
            nisa.tensor_copy(
                dst=out_v[nl.ds(0, n_srows), nl.ds(0, k_pad)],
                src=src_v[nl.ds(0, n_srows), :],
            )
            nisa.tensor_copy(
                dst=out_i[nl.ds(0, n_srows), nl.ds(0, k_pad)],
                src=src_i[nl.ds(0, n_srows), :],
            )
        elif config.sorted:
            # ===== Original full-width sort (small k / padding / generic) ======
            # --- Reload [n_srows, k_pad] and sort descending (max8 / match_replace8 / gather) ---
            # k_pad = k_cols*16 >= 8 already satisfies max8's >= 8 elements requirement.
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            # Reload DMA mode is desnake_dge (= dge_mode.unknown, unconditional -- see its
            # definition comment): this reload reads the SAME private_hbm region the
            # phase-1 de-snake store wrote, and any mode with out-of-band descriptors
            # drops that store->reload ordering, reading stale/zeroed HBM and corrupting
            # values+indices on device only.
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            # Padding handling: nisa.topk only writes the K outputs into snake positions
            # [0, k); positions [k, k_pad) are NOT written and the de-snake carries their
            # (garbage) values into cv. Those garbage values can be large positives and
            # would spuriously win the descending sort, so they MUST be set to FLOAT32_MIN.
            # The de-snake places snake-output position pos at cv column j with
            #   j = (pos % 16) * k_cols + (pos // 16),   i.e. cv[row, j] holds position
            #   pos = (j // k_cols) + 16 * (j % k_cols).
            # The unwritten positions [k, k_pad) therefore map to a set of cv columns that
            # is the CONTIGUOUS tail [k, k_pad) only when k_cols == 1; for k_cols > 1 (e.g.
            # k=99 -> k_cols=7, k_pad=112) they are SCATTERED single columns (one per
            # partition >= k%16 in the partial last snake column). A single free-dim tail
            # memset (the old k_cols==1 path) misses those scattered columns and let
            # garbage (e.g. 4.59) outrank real values, corrupting k-non-multiple-of-16
            # shapes that are not a power-of-two k_pad (so they miss the split path too).
            # Enumerate the invalid cv columns at COMPILE TIME (k, k_cols, k_pad are
            # constants) and memset each contiguous run. Each memset is a legal
            # full-partition free-dim sub-range write (partition-sub-range memsets are
            # rejected by the backend, so the masking is done in cv's free dim, not by
            # partition). For k_cols == 1 this collapses to the single tail run [k, k_pad).
            # _pad_runs is a compile-time tuple of (col_start, col_len); the parser
            # frontend rejects tuple-unpacking in a for-target ("expecting simple
            # variable"), so iterate by index and read the two fields explicitly.
            # Lift real -inf sort keys off the match_replace8 marker value (see
            # _clamp_sort_keys). Runs BEFORE the padding memset below so the padding
            # sentinel stays strictly below the real--inf floor.
            _clamp_sort_keys(cv, n_srows, k_pad)
            _pad_runs = _invalid_desnake_col_runs(k, k_cols, k_pad)
            if k_pad > k:
                for _run_idx in range(len(_pad_runs)):
                    _col_start = _pad_runs[_run_idx][0]
                    _col_len = _pad_runs[_run_idx][1]
                    nisa.memset(
                        dst=cv[:, nl.ds(_col_start, _col_len)], value=FLOAT32_MIN
                    )
            for ps in nl.sequential_range(n_pass):
                cur = nl.ds(ps * 8, 8)
                pos = nl.ndarray((PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
                # max8 returns the 8 largest of the remaining values (descending),
                # match_replace8 removes them (-inf) and reports their buffer positions.
                nisa.max8(
                    dst=out_v[nl.ds(0, n_srows), cur], src=cv[nl.ds(0, n_srows), :]
                )
                nisa.nc_match_replace8(
                    dst=cv[nl.ds(0, n_srows), :],
                    dst_idx=pos[nl.ds(0, n_srows), :],
                    data=cv[nl.ds(0, n_srows), :],
                    vals=out_v[nl.ds(0, n_srows), cur],
                    imm=float("-inf"),
                )
                # Bound the gather index for the same reason as the split path above: an
                # unmatchable value (NaN, or a duplicate extreme already consumed by an
                # earlier pass) leaves dst_idx undefined, and nc_n_gather does not range
                # check its indices.
                nisa.tensor_scalar(
                    dst=pos[nl.ds(0, n_srows), :],
                    data=pos[nl.ds(0, n_srows), :],
                    op0=nl.minimum,
                    operand0=k_pad - 1,
                )
                # Gather the paired snake-position indices at those buffer positions
                # (remapped to vocab indices in the block below).
                nisa.nc_n_gather(
                    dst=out_i[nl.ds(0, n_srows), cur],
                    data=ci[nl.ds(0, n_srows), :],
                    indices=pos[nl.ds(0, n_srows), :],
                )
            # Order the gather writes before the shared index-remap reads them. out_i is
            # written ONLY by nc_n_gather (GpSIMD engine) above, one 8-wide slice per pass;
            # the remap's first op reads the whole out_i on a DIFFERENT engine (a
            # tensor_copy feeding the snake->vocab bit-ops, which force Vector). That
            # cross-engine read RACES the last gather pass and reads the tail columns as
            # stale zeros -> snake position 0 -> vocab index 0 at the row tail (values stay
            # correct; only indices corrupt). Route out_i through a synchronizing
            # SBUF->SBUF DMA, matching the split-path valley fix above.
            #
            # UNCONDITIONAL, formerly `if not fast_dma_safe`. Same unsound gate as the
            # valley sync: the GpSIMD-write / Vector-read pair exists at every n_srows, and
            # a larger n_srows lengthens the gather chain rather than shortening it. This
            # is the sync with the largest blast radius of the three, because out_i feeds
            # the snake->vocab remap that produces the RETURNED INDICES: a stale read here
            # corrupts an index that a caller will use to address memory. The final
            # [0, vocab) clamp keeps such an index in range, so this cannot itself cause an
            # out-of-bounds access -- but it silently mis-pairs values with token ids,
            # which is the failure mode a sampler cannot detect. The split and unsorted
            # paths write out_i with tensor_copy (Vector, same engine as the remap read) so
            # they are program-ordered and correctly need no sync.
            out_i_sync = nl.ndarray(
                (PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=out_i_sync[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), :]
            )
            out_i = out_i_sync
        else:
            # ===== Unsorted fast path (config.sorted == False) ================
            # Skip the descending sort entirely: the K results may be returned in any
            # order. Reload the de-snaked buffers (same reload + desnake_dge DMA mode as
            # the full-width sort above -- that mode keeps the reload ordered after the
            # phase-1 de-snake store, the low/odd-batch correctness fix), then COMPACT
            # the valid de-snaked columns into the output prefix [0, k).
            #
            # nisa.topk writes only snake positions [0, k); the de-snake scatters those K
            # valid columns among the k_pad cv columns (interleaved with unwritten garbage
            # columns when k % 16 != 0). _valid_desnake_col_runs enumerates the contiguous
            # (src, dst, length) runs that gather the valid columns to [0, k); for
            # k % 16 == 0 (or k_cols == 1) this is the single full-width run (0, 0, k).
            # Each tensor_copy is a full-partition free-dim sub-range write (SBUF->SBUF,
            # float32 buffers, so no narrow-dtype strided HBM store). Garbage columns are
            # never copied; the remap + store below read only out_v/out_i[:, 0:k].
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            _valid_runs = _valid_desnake_col_runs(k, k_cols, k_pad)
            # _valid_runs is a compile-time tuple of (src_col, dst_col, length); the parser
            # frontend rejects tuple-unpacking in a for-target, so iterate by index and
            # read the three fields explicitly.
            for _run_idx in range(len(_valid_runs)):
                _src_col = _valid_runs[_run_idx][0]
                _dst_col = _valid_runs[_run_idx][1]
                _run_len = _valid_runs[_run_idx][2]
                nisa.tensor_copy(
                    dst=out_v[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=cv[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )
                nisa.tensor_copy(
                    dst=out_i[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=ci[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )

        # --- Blocked-layout snake-position -> vocab-index remap ---
        # out_i holds snake positions s as float32 (exact integers; s < full_n <= 2^24).
        # The blocked load placed vocab index p*n_cols + c at snake position s = p + 16*c,
        # so the inverse is v = (s % 16) * n_cols + (s // 16). PARTS_PER_GROUP == 16 is a
        # power of two, so the decomposition is EXACT via integer bit-ops (no float
        # floor / rounding):
        #     c = s // 16 == s >> 4   (right_shift by log2(16) = 4)
        #     p = s %  16 == s & 15   (bitwise_and with 16 - 1)
        # then v = p*n_cols + c. Bit-ops require integer dtype, so they run on uint32;
        # the recombine is done in float32 (exact since v <= vocab < 2^24) and cast back
        # to the uint32 index dtype on the store. Applied to only the k results per row.
        # log2(PARTS_PER_GROUP) as a plain literal: an int method (e.g. .bit_length())
        # inside @nki.jit is rejected by the parser frontend.
        s_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        c_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        p_u32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        remapped_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        # s (float32 exact integer) -> uint32 (exact for s < 2^24).
        nisa.tensor_copy(
            dst=s_u32[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), :]
        )
        # c = s >> 4  (== s // 16).
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.right_shift,
            operand0=log2_ppg,
        )
        # p = s & 15  (== s % 16).
        nisa.tensor_scalar(
            dst=p_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        # Recombine in float32: v = p*n_cols + c.
        nisa.tensor_copy(
            dst=c_f32[nl.ds(0, n_srows), :], src=c_u32[nl.ds(0, n_srows), :]
        )
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=p_u32[nl.ds(0, n_srows), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data1=remapped_i[nl.ds(0, n_srows), :],
            data2=c_f32[nl.ds(0, n_srows), :],
            op=nl.add,
        )
        # Final guarantee on the public contract: every returned index is in [0, vocab).
        # Two independent ways it could otherwise escape that range:
        #   - 16 does not divide vocab, so the snake carries full_n - vocab padding
        #     slots. A padded slot that ties the real data (a row of -inf logits) wins
        #     the top-k and remaps to as much as full_n - 1.
        #   - a gathered snake position was itself garbage, which happens when the
        #     values are not orderable (NaN, duplicate extremes) -- see the pos clamps
        #     at the two nc_n_gather sites above.
        # An index outside [0, vocab) breaks the contract callers rely on, and one that
        # indexes a per-shard table (a gather of the form table[v // shard_size]) turns
        # into an out-of-bounds indirect DGE access that faults the device. This clamp
        # is unconditional: it is one Vector op on the k-wide output, and restricting it
        # to has_pad would leave the garbage-position case unguarded when 16 | vocab.
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=remapped_i[nl.ds(0, n_srows), :],
            op0=nl.minimum,
            operand0=float(vocab - 1),
        )

        nisa.dma_copy(
            dst=topk_values[nl.ds(sort_row_start, n_srows), :],
            src=out_v[nl.ds(0, n_srows), nl.ds(0, k)],
        )
        nisa.dma_copy(
            dst=topk_indices[nl.ds(sort_row_start, n_srows), :],
            src=remapped_i[nl.ds(0, n_srows), nl.ds(0, k)],
        )

    return topk_values, topk_indices


# ======================================================================================
# chunked_topk -- vendored from nkilib experimental/topk
# ======================================================================================


"""CHUNKED top-k: a top-k over a vocab WIDER than ``nisa.topk``'s 65,535 ceiling.

WHY THIS EXISTS
---------------
``nisa.topk`` reduces at most ``n < 65536`` elements per call, so a single top-k over the
GPT-OSS vocab (``V = 201,088``) is not expressible. The existing way around that is to
shard the vocab across RANKS and gather the per-rank top-k with a collective. This kernel
does it with no collective at all: it splits the vocab into ``P`` chunks ON ONE CORE, takes
a top-k of each, and reduces the ``P*k`` candidates with a second top-k. Both stages are
local, so there is nothing to reshard between them.

THE IDEA: LET THE INSTRUCTION'S OWN GROUP STRUCTURE BE THE VOCAB SPLIT
---------------------------------------------------------------------
``nisa.topk`` runs one independent top-k per group of ``PARTS_PER_GROUP == 16`` partitions,
and a 128-partition tile holds ``GROUPS_PER_TILE == 8`` of them. So a 128-partition tile
already computes 8 independent top-k per issue. If the tile is loaded as ONE FLAT
CONTIGUOUS SWEEP of the vocab -- partition ``q`` gets vocab ``[q*m, (q+1)*m)`` with
``m = V/128`` -- then group ``g`` (partitions ``16g .. 16g+15``) automatically covers the
contiguous vocab range ``[16g*m, (16g+16)*m)``. The 8-way vocab split is a SIDE EFFECT of
how the instruction is wired; nothing has to build it.

For GPT-OSS the numbers land on it exactly. ``V = 201,088 = 2^7 * 1571`` and
``m = V/128 = 1571``, so each group covers ``16 * 1571 = 25,136 == V/8``:

    25,136 < 65,536              -> legal for nisa.topk
    25,136 % 16 == 0             -> has_pad False, so the coalesced prefetch path is
                                    taken (see fast_coalesced), and stage 1 can skip its
                                    whole phase 2 (see _stage1_desnake_only)
    n_cols = 1,571               -> 3,142 B contiguous DMA run per partition

WHAT IS AND IS NOT AVOIDED
--------------------------
* **Chunk concatenation: avoided entirely.** Stage 1 returns ``[BxS*P, k]``, and
  ``reshape(BxS, P*k)`` puts chunk ``c`` at columns ``[c*k, (c+1)*k)`` of token ``j``. Also
  a pure view. This is why the row order is token-major (``j*P + c``) and not chunk-major:
  token-major makes BOTH reshapes free.
* **Index normalisation: free, or one add.** A stage-1 index is chunk-local, in ``[0, V/P)``,
  and the global id is ``c*(V/P) + local`` with ``c`` fixed per column block. On the fast path
  it costs NOTHING extra: the offset is a whole number of snake partitions, so it folds into
  the snake -> vocab remap's existing multiply (see ``_stage1_desnake_only``). Otherwise it is
  ``P-1`` ``tensor_scalar`` adds on free-dim slices, the cheapest form of vector op there is.
  Compare the sharded alternative, which needs a boundary-counting divide plus a
  select-and-sum over a runtime ``group_map`` tensor.
* **Snaking on the LOAD: avoided entirely -- there is no rearrangement at all.** This is
  the crux, so it is worth being exact. ``inp.reshape(BxS*P, V/P)`` is a pure view of a
  C-contiguous ``[BxS, V]`` tensor (row ``j*P + c`` IS chunk ``c`` of token ``j``), and
  the blocked load is
  ``ap(pattern=[[n_cols, 128], [128*n_cols, tiles], [1, n_cols]], offset=row_start*vocab)``.
  When ``vocab == 16*n_cols`` exactly -- which a chunk width of ``16*1571 = 25,136``
  satisfies -- partition ``q`` reads flat offset ``q*n_cols``, i.e. ONE contiguous
  128-partition sweep of HBM. The "snake" is then purely nominal: it is how the 16
  partitions are interpreted, not a permutation anything performs. Data is loaded directly
  and the instruction's group boundaries ARE the vocab division.
* **De-snaking on the OUTPUT: still required.** After ``nisa.topk`` each group's k results
  are spread across its 16 partitions (snake positions ``[0,k)`` -> partitions 0..15,
  columns ``0..k/16-1``), and both consumers need them on ONE partition per row: the phase-2
  ops are per-partition free-dim ops, and the ``[BxS, k]`` output is row-major. So
  each stage de-snakes once per tile -- two stages, two de-snakes. It cannot be dropped:
  ``nisa.topk`` returns ORIGINAL snake positions rather than output positions, so the
  value<->index pairing is not recoverable from the output layout, and cross-partition
  movement needs a DMA or a transpose either way.

  **Stage 1's phase 2, however, is gone.** It used to reload those de-snake buffers, compact
  the valid columns, remap the snake positions and store ``[BxS*P, k]`` -- which this kernel
  then viewed as ``[BxS, P*k]`` and handed to the reduce. When ``k % 16 == 0`` that entire
  pass is a COPY: ``asc_val_hbm`` is already ``[BxS*P, k]``, so viewed as ``[BxS, P*k]`` it IS
  the candidate array, in an order the index buffer beside it matches slot for slot -- and the
  concat convention only ever required the two to agree, never a particular order. So the
  de-snake buffers are used directly and the pass is deleted, along with the separate padded
  reduce-input staging. The snake -> vocab remap moves to where it is 16x narrower: onto
  ``idx_snake``, ``k_cols`` columns wide instead of ``k``, with the per-chunk offset joining it
  as one add per 16-partition group. See ``_stage1_desnake_only``.

  WHAT IS STILL LEFT, and the trap in it. The inter-stage HBM bounce itself: stage 1 writes
  the candidates to HBM and the reduce loads them straight back. Collapsing that means keeping
  the inter-stage snake in SBUF, and the value side is genuinely cheap -- stage-2 snake
  partition ``p2`` takes stage-1 tile partitions ``{p2, 16+p2, ..., 16(P-1)+p2}``, so with
  chunk ``g`` placed at stage-2 columns ``[g*k_cols, (g+1)*k_cols)`` it is P SBUF->SBUF block
  copies of ``[16, k_cols]``, not a general permutation. The index mapping stays closed-form:

      stage-2 snake position t  ->  g = (t >> 4) // k_cols,   p2 = t & 15
      stage-1 slot              ->  s = idx_snake1[16g + p2, (t >> 4) % k_cols]
      global vocab id            =  (16g + (s & 15)) * n_cols1 + (s >> 4)

  (``16g + (s & 15)`` is just the tile partition that held the value, so the id reduces to
  ``partition * n_cols1 + column``. Verified against a global top-k over 5 geometries x 4
  adversarial distributions x 25 seeds with the modelled hardware output order randomised, and
  with four injected mutations -- dropped chunk term, swapped shift/mask, wrong shuffle
  partition, wrong lookup slot -- all caught.)

  The trap is the INDEX side, and it is why this is not done. That layout makes the reduce's
  candidate position ``v = q*(P*k_cols) + g*k_cols + c1``, which is NOT affine in the source
  partition ``pp = 16g + q`` -- the strides over ``g`` and over ``q`` differ. So the one
  contiguous ``par_dim``-partition de-snake store has to become P stores of 16 partitions
  each, and merging those very stores 8:1 is what an earlier round measured as a large win
  (they were the top serial contributors in the profile). The value side saves DMAs and the
  index side spends them back, with the balance depending on P and the tile count -- so
  measure both ends before committing, and do not assume the SBUF version is faster because
  it moves fewer bytes.

THE HARDWARE SORT IS THE ONLY SORT (THE SOFTWARE SORT IS GONE)
--------------------------------------------------------------
``nisa.topk(sorted=True)`` makes the instruction emit rank ``j`` at snake position
``j``. Because the de-snake mapping is also a compile-time fact, recovering descending order is
then a FIXED PERMUTATION of columns. The software sort that used to serve ``sorted=True`` -- 16
``max8``, 16 ``nc_match_replace8``, 16 gathers, two reverse-gathers, 8 bitonic merge stages,
three synchronising DMAs and both sort-key clamps per sort tile -- WAS DELETED, not kept as a
fallback: a sorted request on a toolchain without the capability (patched nki + neuronx-cc >=
2.0.280935) or with ``k > 8192`` is refused at trace time. Beyond the speed, the deletion
removes the value-addressed index recovery and its tie hazards outright, and it removes a
family of GpSIMD-producer handoffs (max8 / nc_match_replace8 / nc_n_gather feeding cross-engine
readers) whose sync-DMA fences the store-visibility work below showed to be DISTANCE rather
than ordering. The ``sorted=False`` compaction path is not a sort and stays -- stage 1 runs on
it. The ``p == 1`` small-vocab case runs through this kernel's own inlined top-k for the same
reason, rather than delegating to ``gpsimd_topk`` and silently re-entering a software sort.

**Deleting that work moved the wall clock by NOTHING** -- the first landing (permutation folded
into the reload's access pattern) measured 170.3 -> 172.1 us at 128 ranks, gbs=128 -- and the
work/wait split of the profile says why (summed occupancy over all engines):

                    instructions   occupancy
    software sort:  work 644        214.3 us
                    wait 101        167.1 us
    hardware sort:  work 464        173.9 us      -180 instructions, -40 us
                    wait  61        167.5 us      -40 instructions,  +0.4 us

The waiting was invariant: 40 us of work vanished and the wait absorbed it exactly. The machine
was fully idle for only ~13 us of the ~171 us body -- a timeline full of semaphore waits, with
the critical path the serial chain norm -> GEMM -> hop 0 -> hop 1 -> stage 1 -> reduce ->
sampler. **The lever was therefore never the work; it was the synchronisation on that chain**,
and the strided reload was itself a synchronisation point: 1,968 serialised 4-byte DMA transfers
(see the next section). Replacing it with a contiguous reload + on-chip permutation, and the
fused tail's gather with the Tensor-Engine one-hot, is what finally moved the body:

    gbs=128:  170.0 -> 146.2 us   (-14.0% vs the shipped software path; -18.8% vs baseline)
    gbs=512:  348.5 -> 334.4 us   (-4.0%)
    gbs=1024: 580.1 -> 577.9 us   (-0.4% -- the body there is transfer/compute bound)
    semaphore stall: -90 / -194 / -341 us respectively; 4-byte packets 3,070 -> 1,054 at gbs=128

Two lessons worth keeping: a standalone measurement of this kernel is NOT predictive of the tail
(the same sort is worth 1.32x alone and nothing in situ until the synchronisation went too); and
before optimising anything here, split the profile into work and wait -- if wait occupancy is
~equal to the body, instruction count is the wrong lever, and a store/reload or a serialised DMA
burst on the critical chain is the right one.

THE 4-BYTE DMA PROBLEM (solved -- the permutation moved on chip, which exposed a deeper bug)
--------------------------------------------------------------------------------------------
Folding the rank permutation into the reload's access pattern was NOT free, and the wording that
said so was wrong. **The last pattern level is the FASTEST-VARYING**, so a pattern ending in
``[k_cols, PARTS_PER_GROUP]`` gives the innermost dimension a stride of ``k_cols`` rather than 1,
and every single element becomes its own transfer. Measured at gbs=128 against the software sort:

    4-byte DMA packets     1,102 -> 3,070      (+1,968)
    their summed time        7.3 -> 57.3 us    (+50 us)

Rank order and de-snaked order are a TRANSPOSE of one another (rank ``j = p + 16c``, de-snaked
column ``m = p*k_cols + c``), so exactly one side has to be strided, and the strided side belongs
ON CHIP. That is what the hw_sort branch now does: the reload is one contiguous run per row, and
the permutation is 16 strided ``tensor_copy`` ops per tensor, k_cols wide each.

**Every earlier attempt at exactly this broke correctness at BxS == 1, and the fault was never
the permutation.** Moving ~30 Vector ops into phase 2 shifted the engine schedule enough to
expose a LATENT RACE in the fused tail's ``nc_n_gather`` -- present since the first version of
this kernel, hit by no schedule before it. The debug taps (L4) plus the Instruction/Flow tables
of four failing NEFFs established, against escalating fences, that the gather's completion
signal does not mean its stores are visible: a Vector consumer with a correct FLOW_DEPENDENCE
edge read the gather's tail as stale zeros 500 ns after it retired; a synchronising SBUF->SBUF
DMA read the same mid-flight state; a same-dtype GpSIMD fence copy was copy-propagated away;
and an unfoldable GpSIMD CAST fence was itself entered by its consumer 119 ns into its 900 ns
execution, with two GpSIMD instructions visibly overlapping on the engine. Meanwhile
``nisa.topk``'s consumers at SHORTER distances are always correct -- the defect is the gather
ucode's, not GpSIMD stores in general (the multi-group corner of the same defect).

So the gather is not fenced, it is GONE: the position -> id lookup is an exact one-hot
contraction on the Tensor Engine (see the fused tail), whose PSUM -> Vector handoff is the
best-validated producer edge on the chip. Everything is fp32, ids are < 2^24, every one-hot
column selects exactly one partition, so every sum is one exact product plus zeros. The software
sort's own gathers (max8 / nc_match_replace8 / the bitonic reverse) carried the same hazard
class behind sync-DMA fences that this evidence showed to be DISTANCE rather than ordering --
one more reason that code is now deleted rather than kept as a fallback.

The synchronisation-removal lesson (see THE HARDWARE SORT above, where it is now measured at
-23.8 us) also reframes the remaining inter-stage HBM bounce: its value is not the 2.6 us of DMA
bytes, it is that a store followed by a reload of the same region is a serialisation point on
the critical chain. Same for the hops -- hop 1 costs 2.8x hop 0 for identical bytes purely
because it crosses devices.

STAGE 1 IS NEVER SORTED
-----------------------
Stage 1 always runs with ``sorted=False``, and this is deliberately NOT a caller option.
Its ordering is unobservable -- stage-1 values and indices never leave this kernel, and the
reduce re-sorts -- so the choice cannot change the output, only the cost. Measured on trn3
at LNC=1, ``V = 201,088``, ``k = 256``:

    BxS=1 (P=8):  97.32 us -> 76.33 us   (-21.6%)
    BxS=8 (P=4): 176.53 us -> 151.88 us  (-14.0%)

The profile explains the size of it: the VECTOR engine dominates this kernel, not GpSIMD
(54.24 of 97.32 us at BxS=1), because the phase-2 descending sort is sized by
``k`` rather than by the vocab width. Dropping stage 1's sort removed 132 Vector
instructions (317 -> 185) and 21.2 us.

The wider lesson for anyone extending this: cost here tracks the NUMBER OF TOP-K
INVOCATIONS times ``k``, not the number of elements scanned. A three-stage tree reduce
would add a third phase-2 sort and lose more than the narrower reduce could win.

CHOOSING P
----------
Stage-1 cost has a closed form. With ``rows = P*R`` and ``width = V/P``, GpSIMD cost
``ceil(rows/8) * ceil(width/16)`` becomes ``R*V/128`` -- the ``P`` cancels. So P CANNOT make
stage 1 faster; it can only make it worse by failing to fill a tile, shortening the DMA
runs, or losing the ``% 16`` coalesced path. Meanwhile the reduce grows as ``k*P``. The
optimum is therefore the SMALLEST P that fills a whole tile and is legal:

    P = min { P : (BxS*P) % 8 == 0,  V/P < 65536,  (V/P) % 16 == 0,  P*k < 65536 }

``create_chunked_topk_config`` picks it. At ``V = 201,088, k = 256`` that is ``P = 8`` for
``BxS = 1`` (8 rows == exactly one tile) and ``P = 4`` for ``BxS >= 4``.

LNC=1 BY DESIGN
---------------
Both stages pass ``num_programs=1``. NOTE that this does not pin anything: the inlined top-k
re-reads the program count from the live grid (``get_verified_program_sharding_info``), so at
LNC=2 its ``per_lnc_BxS`` halves and ``fast_dma_safe`` can silently flip off. This kernel is
written for and tested at LNC=1 only, and asserts it at trace time.

DEBUG TAPS
----------
Every correctness defect this kernel has had was invisible except on hardware, under full
engine/DMA overlap: the simulator serialises, so does a device dump, and both return the right
answer while the device returns the wrong one. Diagnosing that by hypothesis costs a 128-rank
sweep per guess. ``config.debug_taps`` instead makes the index chain OBSERVABLE in situ -- it
appends seven extra outputs (``TAP_NAMES``) to the return tuple, each a copy of one link:

    s1_pos    [BxS*P, k]       stage-1 snake positions phase 2 worked from, pre-remap
    s1_gid    [BxS*P, k]       stage-1 chunk-local vocab ids, post-remap
    s1_late   [BxS*P, k_pad]   a SECOND read of stage 1's de-snake buffer
    cand_i    [BxS, P*k]       global ids after the chunk offsets -- the gather's data
    r_pos     [BxS, k]         the reduce's positions -- the gather's indices
    gathered  [BxS, k]         the gather's result, before the uint32 narrowing
    s1_val    [BxS*P, k]       the VALUE side of stage 1's candidates (appended last)

A bad returned id is then localised in ONE run rather than one guess per run: walk the chain
backwards and the first tap that disagrees with an offline oracle names the culprit. Values have
never been the failing side, which is itself diagnostic -- values and indices ride in separate
buffers, so a defect that takes only the indices is a per-buffer fault (a race, a dropped
strided store) and not a bad selection.

TWO PROPERTIES MAKE THE TAPS TRUSTWORTHY, and both are easy to lose:

* **A tap must not create the ordering it is measuring.** Adding a reader of a buffer adds a
  dependency edge on it, which can hide the very race being hunted. So every tap point above is
  a buffer that ALREADY has a full-width consumer immediately after it (``out_i`` is read whole
  by the remap, ``i_f`` and ``pf`` by the gather, ``got`` by the output cast). The tap rides
  along an edge that exists; it does not add one.
* **``s1_late`` works without being guaranteed late.** It is traced last so it TENDS to be
  scheduled late, but nothing pins it there. It does not need to be: the signature of a
  store/reload race is TWO READS OF THE SAME ADDRESS DISAGREEING, which cannot happen unless a
  write was in flight between them. ``s1_pos`` wrong + ``s1_late`` right means the de-snake
  store landed after phase 2 read it; both wrong means ``nisa.topk`` reported those positions
  and nothing about DMA ordering is involved.

Taps change the kernel's arity, so they are opt-in and off by default. The harness maps
validators to outputs POSITIONALLY, so a compile+infer test run must not enable them; the
128-rank pod path is unaffected because the on-device profiler's capture flow saves every
output as ``output_<n>.2.npy`` regardless of how many there are.

Complementary, and the right first question for any new DMA pattern: ``--enable-perf-analysis``
emits ``analysis_nc00.log`` with a ``critical_dep_type`` per instruction, where ``Flow`` means a
semaphore edge exists and ``Engine`` means the only ordering is same-engine issue. If a
consumer's dependency on a producer is not ``Flow``, it is unordered by construction -- no
amount of measurement is needed to know it is a bug waiting for the right traffic.
"""


# nisa.topk reduces n < _TOPK_N_MAX elements per call. This is the instruction's own
# bound (the returned index is a snake POSITION), not a kernel choice -- which is why
# P >= ceil(V / (_TOPK_N_MAX - 1)) is a hardware floor rather than a tuning knob.
_TOPK_N_MAX = 65536
# nisa.topk's k bound, mirrored so an out-of-envelope k fails here with a clear message.
_TOPK_K_MAX = 32768

# --- Hardware sort capability ------------------------------------------------------------
# nisa.topk gains a `sorted=` flag in newer NKI toolchains, which makes the
# GpSIMD instruction emit rank j at snake position j. That turns the software descending sort
# into a COMPILE-TIME COLUMN PERMUTATION -- see the hw_sort branch in _inline_topk, where it
# folds into the reload's access pattern -- which is NOT free, see THE 4-BYTE DMA PROBLEM.
#
# Probed at TRACE TIME rather than version-gated, because passing `sorted=` to an nki that does
# not have it is a TypeError while tracing, not a graceful fallback. A version check would also
# be wrong twice over: the flag needs BOTH a patched nki AND the BIR field from
# neuronx-cc >= 2.0.280935, and the two move independently.
_HW_SORT_AVAILABLE = "sorted" in inspect.signature(nisa.topk).parameters
# The ISA caps the sorted mode at k <= 8192. Sorted requests above it are REFUSED: the
# software sort that used to serve them was deleted.
_HW_SORT_MAX_K = 8192


@dataclass(frozen=True, eq=True)
class ChunkedTopkConfig(nl.NKIObject):
    """Configuration for the chunked (wide-vocab) top-k kernel.

    Attributes:
        BxS: Combined batch*sequence dimension (number of logical rows / tokens).
        vocab_size: Full vocab length reduced over. May exceed nisa.topk's 65,535.
        k: Number of largest elements to return.
        chunks: P, the number of disjoint vocab chunks stage 1 reduces independently.
            ``chunks == 1`` short-circuits to a plain gpsimd_topk.
        sorted: Whether the FINAL output must be descending.
        inp_dtype: Input/value dtype (must be bfloat16 -- gpsimd_topk's own requirement).
        index_dtype: Output index dtype (uint32).
        out_shape: Logical output shape (input leading dims + k).
        debug_taps: Append the seven diagnostic taps to the return tuple. See DEBUG TAPS in the
            module docstring. Off by default and free when off; it changes the kernel's arity,
            so a caller that turns it on has to consume the extra outputs.
        skip_nan_fold: Skip stage 1's NaN -> BFLOAT16_MIN fold, under a CALLER CONTRACT that
            the input contains no NaN. The fold exists because nisa.topk is undefined on NaN
            (see the fold site); on a NaN-free input it can never fire, yet it still costs a
            compare plus a predicated copy over the FULL input width per tile -- the widest
            non-topk work in stage 1. +/-inf need no fold and stay legal either way (they are
            orderable; the bitpattern gate covers them). Default False: only a caller that can
            PROVE its input finite (e.g. the GPT-OSS tail, whose logits are its own GEMM
            output) should set it, because with NaN present the fold is what keeps the output
            defined. The reduce stage always skips independently of this flag -- its input is
            stage 1's own already-folded output.
        interleaved_peer_width: 0 (default) means ``inp`` is the ordinary C-contiguous
            ``[BxS, vocab]``. A value ``w > 0`` declares the ALL-TO-ALL RECEIVE LAYOUT
            instead: ``inp`` is ``[BxS * (vocab/w), w]`` where row ``p*BxS + j`` holds token
            ``j``'s vocab columns ``[p*w, (p+1)*w)``. Stage 1 then folds the row -> column
            de-interleave into its own load patterns (three levels, unit innermost stride),
            which DELETES the caller's [rows, vocab] materialisation -- a full HBM round trip
            of the received logits. Results are BITWISE identical to de-interleaving first:
            only the DMA descriptors change, not one value or its snake position. Fast-path
            only: requires the desnake-only stage-1 shape and ``w | (vocab/P)/16`` so every
            snake partition covers whole peer blocks; anything else is refused at trace time.
        interleaved_batch_groups: 1 (default) means one receive buffer. ``G > 1`` declares a
            BATCH-GROUP SLABBED receive: the producer ran G separate exchanges, each delivering
            tokens ``[g*BxS/G, (g+1)*BxS/G)`` as its own peer-major slab (what lets the
            exchange start under the producer's GEMM), and ``inp`` is a LIST of those G slab
            tensors (jit-in-jit callers only). Each slab is read in place with its own data
            dependency, so group g's stage-1 sorts start when ITS exchange lands. Requires
            ``interleaved_peer_width > 0`` and ``G | BxS``. A stage-1 sort tile MAY straddle
            two slabs (each 16-partition group's load selects its slab per token); it then
            waits for the later exchange, so groups aligned to whole 8-row tiles
            (``(BxS/G) * P % 8 == 0``) overlap best.
    """

    BxS: int
    vocab_size: int
    k: int
    chunks: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    out_shape: tuple
    debug_taps: bool = False
    skip_nan_fold: bool = False
    interleaved_peer_width: int = 0
    # > 1 when the receive buffer is BATCH-GROUP slabbed: the producer ran G separate
    # exchanges, each delivering tokens [g*T/G, (g+1)*T/G) as its own peer-major slab (what
    # lets the exchange start under the producer's GEMM). Token j's slab base is
    # (j // (T/G)) * (T/G) * vocab elements; within a slab the layout is exactly the
    # interleaved_peer_width one with T/G tokens. Requires interleaved_peer_width > 0.
    interleaved_batch_groups: int = 1

    @property
    def topk_config(self):
        """Self-reference so a shared torch_ref reading ``.topk_config.k`` /
        ``.topk_config.sorted`` works against this config without a separate object."""
        return self


def _chunk_legal(width: int, p: int, k: int) -> bool:
    """THE legality predicate for one (chunk width, P, k) triple, shared by ``pick_chunks``
    and ``_validate_chunked_topk`` so the two can never disagree (an auto-picked P the
    validator then refuses, while another P is legal, was exactly that disagreement).

    In order: P is a power of two (the stage-1 bypass derives chunk offsets with a bitwise
    mask that is a valid modulus only for pow2); the instruction bound applies to the
    16-PADDED snake width (see ``snake_padded_n``); k fits one chunk; and for P > 1 the
    fused reduce's one-hot position lookup holds P*k positions across at most 128 chunks of
    128 partitions -- tighter than the instruction's own ``n < 65536``.
    """
    if p < 1 or p & (p - 1):
        return False
    if width < 8 or snake_padded_n(width) >= _TOPK_N_MAX or k > width:
        return False
    if p > 1 and not (8 <= p * k <= PMAX * PMAX):
        return False
    return True


def _slab_sizes_match(shapes, expected_elems: int) -> bool:
    """Every receive slab holds exactly ``expected_elems`` (= tokens-per-slab * vocab).

    The single-tensor path gets this check for free from its ``reshape``; a slab LIST is
    flattened per tensor, which cannot fail, so a mis-sized slab would otherwise be read
    through the hand-built flat pattern at wrong token/peer bases with no trace-time error.
    """
    return all(int(np.prod(shape)) == expected_elems for shape in shapes)


def pick_chunks(BxS: int, vocab: int, k: int) -> int:
    """Smallest legal P that fills a whole 128-partition tile. See CHOOSING P above.

    Ranking, in order: fills a tile (``BxS*P % 8 == 0``), then hits the coalesced load
    path (``(vocab/P) % 16 == 0``), then smallest P. Ties on the first two keys are what
    make this pick ``P = 8`` at ``BxS = 1`` (where ``P = 4`` would leave half a tile
    padding) and ``P = 4`` at ``BxS >= 4`` (where both fill, so the smaller reduce wins).

    Returns 1 when the vocab already fits one nisa.topk call (its PADDED snake width under
    the bound), i.e. no chunking is needed. Legality per P is ``_chunk_legal`` -- the same
    predicate the validator applies -- so whatever this returns traces.
    """
    if snake_padded_n(vocab) < _TOPK_N_MAX:
        return 1
    best_key, best_p = None, None
    p = 1
    while p <= PMAX:
        if vocab % p == 0:
            width = vocab // p
            if _chunk_legal(width, p, k):
                key = (
                    0 if (BxS * p) % GROUPS_PER_TILE == 0 else 1,
                    0 if width % PARTS_PER_GROUP == 0 else 1,
                    p,
                )
                if best_key is None or key < best_key:
                    best_key, best_p = key, p
        p *= 2
    if best_p is None:
        raise ValueError(
            f"no legal chunk count for vocab={vocab}, k={k}: need a power-of-two P dividing "
            f"vocab with 8 <= vocab/P, 16*ceil((vocab/P)/16) < {_TOPK_N_MAX}, k <= vocab/P and "
            f"8 <= P*k <= {PMAX * PMAX}"
        )
    return best_p


def interleaved_fold_supported(bxs: int, vocab: int, k: int, peer_width: int) -> bool:
    """True iff this kernel accepts ``interleaved_peer_width=peer_width`` at this geometry.

    THE eligibility predicate for the receive-layout fold, exported so callers decide
    "fold or explicit de-interleave" with the callee's OWN rules instead of re-deriving a
    subset at the call site (an incomplete re-derivation turned legal configs into
    trace-time refusals -- e.g. a vocab small enough that ``pick_chunks`` returns 1
    delegates to the single-call path, which does not implement the fold at all).
    Mirrors, in order: the p == 1 delegate refusal, the desnake-only stage-1 path
    requirements (k, row and width alignment -- which also imply the coalesced whole-tile
    load path), and the peer-width divisibility asserts.
    """
    if peer_width <= 0:
        return False
    p = pick_chunks(bxs, vocab, k)
    if p == 1:
        return False
    width = vocab // p
    s1_rows = bxs * p
    if not (
        k % PARTS_PER_GROUP == 0
        and s1_rows % GROUPS_PER_TILE == 0
        and width % PARTS_PER_GROUP == 0
    ):
        return False
    n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
    return (
        vocab % peer_width == 0
        and n_cols_s1 % peer_width == 0
        and width % peer_width == 0
    )


def create_chunked_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    chunks: int = 0,
    sorted: bool = True,
    debug_taps: bool = False,
    skip_nan_fold: bool = False,
    interleaved_peer_width: int = 0,
    interleaved_batch_groups: int = 1,
) -> ChunkedTopkConfig:
    """Build a ChunkedTopkConfig. ``chunks=0`` (default) auto-picks via ``pick_chunks``.

    ``inp_shape`` is always the LOGICAL ``[..., vocab]`` shape, including when
    ``interleaved_peer_width`` declares the receive layout (the physical tensor then has the
    same element count in ``[BxS*(vocab/w), w]`` order; every internal access is a reshape or
    an explicit pattern over the same flat buffer).
    """
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    return ChunkedTopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        chunks=chunks if chunks else pick_chunks(BxS, vocab_size, k),
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        out_shape=tuple(list(inp_shape[:-1]) + [k]),
        debug_taps=debug_taps,
        skip_nan_fold=skip_nan_fold,
        interleaved_peer_width=interleaved_peer_width,
        interleaved_batch_groups=interleaved_batch_groups,
    )


#: Names of the diagnostic taps, in the order ``chunked_topk`` appends them to its return tuple
#: when ``config.debug_taps``. Exposed so a caller that forwards them (the GPT-OSS tail does) can
#: label them without hard-coding the count. See DEBUG TAPS in the module docstring.
TAP_NAMES = ("s1_pos", "s1_gid", "s1_late", "cand_i", "r_pos", "gathered", "s1_val")


def _validate_chunked_topk(config: ChunkedTopkConfig) -> None:
    """Validate the requested shape against every nisa.topk constraint, per stage."""
    v, k, p = config.vocab_size, config.k, config.chunks
    kernel_assert(
        config.inp_dtype == nl.bfloat16, "chunked_topk requires bfloat16 input"
    )
    kernel_assert(
        1 <= k < _TOPK_K_MAX, f"chunked_topk requires 1 <= k ({k}) < {_TOPK_K_MAX}"
    )
    # Sorted output is hardware-sort ONLY. The software sort was deleted (see THE HARDWARE SORT
    # IS THE ONLY SORT in the module docstring), so a sorted request on a toolchain without the
    # capability is REFUSED at trace time rather than silently served by a slower, race-prone
    # fallback. The capability needs BOTH a patched nki and neuronx-cc >= 2.0.280935.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE,
            "chunked_topk sorted=True requires nisa.topk(sorted=True); this toolchain does not "
            "have it (patched nki + neuronx-cc >= 2.0.280935 required) and the software sort "
            "was removed. Pass sorted=False or upgrade the toolchain.",
        )
        kernel_assert(
            k <= _HW_SORT_MAX_K,
            f"sorted top-k is capped at k <= {_HW_SORT_MAX_K} by the ISA, got k={k}",
        )
    kernel_assert(p >= 1, f"chunks ({p}) must be >= 1")
    # Power-of-two P is a CORRECTNESS requirement, not a picker preference: the stage-1 bypass
    # derives each row's chunk offset with a bitwise-AND mask (16*P - 1), which is a valid
    # modulus only for power-of-two P. pick_chunks only ever returns powers of two; this guard
    # is for an explicit caller-set chunks, where e.g. chunks=3 passed every divisibility check
    # and returned in-range, exactly-right VALUES with silently wrong global ids.
    # The per-bound asserts below spell out _chunk_legal (the predicate pick_chunks selects
    # with) one named message at a time; test_pick_chunks_agrees_with_validator pins that
    # the two stay the same set.
    kernel_assert(p & (p - 1) == 0, f"chunks ({p}) must be a power of two")
    kernel_assert(v % p == 0, f"vocab ({v}) must be divisible by chunks ({p})")
    width = v // p
    # Stage 1 sees [BxS*P, V/P], so the CHUNK width is what faces the instruction bound -- and
    # what faces it is the 16-PADDED snake size 16*ceil(width/16), which reaches 65536 (past
    # the bound) for widths 65521..65535. Bound the padded width, not the raw one.
    kernel_assert(8 <= width, f"chunk width V/P ({width}) must be >= 8")
    kernel_assert(
        PARTS_PER_GROUP * div_ceil(width, PARTS_PER_GROUP) < _TOPK_N_MAX,
        f"chunk width V/P ({width}) padded to a whole 16-column snake "
        f"({PARTS_PER_GROUP * div_ceil(width, PARTS_PER_GROUP)}) must be < {_TOPK_N_MAX}",
    )
    kernel_assert(k <= width, f"k ({k}) must be <= chunk width ({width})")
    if p > 1:
        # The reduce sees [BxS, P*k]; P*k is a SECOND, independent bound -- and a TIGHTER one
        # than the instruction's: the fused reduce translates sort positions through the
        # candidate-id array with a Tensor-Engine one-hot lookup that lays the P*k positions
        # across ceil(P*k / 128) partition chunks, so P*k must fit in 128 chunks of 128.
        # Batch sharding keeps it small (2,048 at P=8) but a large P or k inflates it.
        red_w = p * k
        kernel_assert(
            8 <= red_w <= PMAX * PMAX,
            f"reduce width P*k ({red_w}) must be in [8, {PMAX * PMAX}]: the fused reduce's "
            f"one-hot position lookup holds all P*k positions across at most {PMAX} "
            f"{PMAX}-wide partition chunks",
        )
    if config.interleaved_peer_width:
        w = config.interleaved_peer_width
        n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
        kernel_assert(
            v % w == 0 and n_cols_s1 % w == 0,
            f"interleaved_peer_width ({w}) must divide the vocab ({v}) and the stage-1 snake "
            f"row width ({n_cols_s1}); the receive-layout fold needs whole peer blocks per "
            f"snake partition",
        )
    g = config.interleaved_batch_groups
    kernel_assert(g >= 1, f"interleaved_batch_groups ({g}) must be >= 1")
    if g > 1:
        kernel_assert(
            config.interleaved_peer_width > 0,
            "interleaved_batch_groups > 1 describes a slabbed RECEIVE layout and needs interleaved_peer_width",
        )
        bxs = config.BxS
        kernel_assert(
            bxs % g == 0, f"batch ({bxs}) must divide by interleaved_batch_groups ({g})"
        )
        # No sort-tile alignment requirement: every 16-partition group's load selects its
        # slab per TOKEN (inp_flats[j // tok_span]), so a stage-1 tile may straddle two slabs
        # and simply waits for the later of its two exchanges. Aligned groups
        # (((bxs/g)*p) % 8 == 0) overlap best -- a performance preference, not legality.


# =======================================================================================
# THE INLINED TOP-K
# =======================================================================================
# ``_inline_topk`` is a VERBATIM inline of ``gpsimd_topk``'s ``@nki.jit`` body: the same
# instructions, in the same order, over the same buffers, with the same DMA modes. Read that
# way it buys nothing -- and measured that way it is worth about -1% -- but it is what puts
# BOTH ``nisa.topk`` calls inside one traced body, which is the precondition for every
# inter-stage optimisation this kernel wants: keeping the candidate array in SBUF instead of
# bouncing it through HBM, and handing stage 1's snake layout straight to stage 2 instead of
# de-snaking and re-snaking around a [BxS, k] HBM interface. None of that is expressible while
# the stages are separate kernels whose only interface is that HBM array.
#
# It is a COPY rather than a shared helper on purpose. ``gpsimd_topk`` is a public kernel with
# its own callers and its own contract; the specialisations wanted here -- a stage that never
# de-snakes, a stage whose "vocab" is a candidate array rather than logits -- are not modes
# worth bolting onto it. This is a peak-performance path, and the duplication is the cheaper
# side of that trade.
#
# THE CORRECTNESS RATIONALE IS DELIBERATELY NOT DUPLICATED. Nearly every non-obvious line
# below is there because of a specific measured defect, documented at length beside the SAME
# line in ``gpsimd_topk``: the three unconditional synchronising SBUF->SBUF DMAs, the
# ``dge_mode.unknown`` on the de-snake round trip, the two NaN folds, the ``NEG_INF_FLOOR``
# sort-key clamp, the two gather-index clamps, and the compile-time padding-column
# enumeration. Each comment here names its counterpart. Do not "simplify" any of them without
# reading the original first -- several look redundant and are not.
#: Slots of ``_inline_topk``'s ``fused`` argument. A plain tuple rather than a dataclass on
#: purpose: nki hashes config objects for its trace cache, and a dataclass holding live tensor
#: handles is a bad thing to put in a hash key.
_FUSE_CAND_I = 0  # [rows, P*k] uint32 HBM: stage-1 chunk-local ids, the gather's data
_FUSE_CHUNKS = 1  # P
_FUSE_WIDTH = 2  # vocab // P, the per-chunk id offset
_FUSE_OUT_VALS = 3  # [rows, k] bf16 shared_hbm destination
_FUSE_OUT_IDX = 4  # [rows, k] uint32 shared_hbm destination
_FUSE_N_ROWS = 5  # rows to emit; the reduce's own row count may be padded above it
_FUSE_TAP_CAND = 6
_FUSE_TAP_POS = 7
_FUSE_TAP_GOT = 8
_FUSE_IDS_GLOBAL = (
    9  # cand_i already holds float32 GLOBAL ids: skip the cast and the offsets
)


def _stage1_desnake_only(
    inp: nl.NkiTensor,
    config,
    chunks: int,
    chunk_width: int,
    pad_rows: int,
    tap_pos=None,
    skip_nan_fold=False,
    peer_width=0,
    batch_groups=1,
):
    """Stage 1 with NO PHASE 2 AT ALL: the de-snake buffers ARE the candidate arrays.

    THE OBSERVATION. ``gpsimd_topk``'s phase 1 de-snakes each tile into ``[BxS, k_pad]`` HBM
    buffers, and phase 2 then reloads them, compacts the valid columns to ``[0, k)``, remaps the
    snake positions, and stores ``[BxS, k]`` results -- which ``chunked_topk`` immediately views
    as ``[BxS/P, P*k]`` and hands to the reduce. When ``k_pad == k`` that whole pass is a COPY.
    ``asc_val_hbm`` is ``[BxS*P, k]``, so viewed as ``[BxS, P*k]`` it already IS the candidate
    value array, in a candidate order the index buffer beside it matches slot for slot -- and the
    concat convention only requires the two agree, never a particular order. So phase 2 is not
    optimised here, it is DELETED: two reloads, two compacting copies, the remap over a k-wide
    tile, two stores, and (in the caller) the P-1 chunk-offset adds and the uint32 -> float32
    cast of a P*k-wide tile all go away, and so does the separate padded reduce-input staging.

    What has to happen instead, and why it is cheaper: the snake -> vocab remap still has to be
    applied, but it can be applied to ``idx_snake`` BEFORE the de-snake store, where the tile is
    ``k_cols`` wide instead of ``k``. That is 16x narrower for the same op count, and Vector cost
    tracks free-dim width, not partition count -- the same work over 16 columns of 128 partitions
    instead of 256 columns of 8. The per-chunk global-id offset joins it there, as one add per
    16-partition group on a compile-time constant, replacing P-1 adds over k columns.

    Args:
        inp: ``[BxS*P, V/P]`` bfloat16 view of the caller's vocab, one chunk per row.
        config: a ``GpsimdTopkConfig`` for that view. ``k % 16 == 0`` is REQUIRED (it is what
            makes ``k_pad == k`` and therefore makes phase 2 a copy), as is a shard of whole
            128-partition tiles (which is what keeps the value buffer bfloat16 -- see
            ``fast_dma_safe`` -- and the reduce needs bfloat16 input).
        chunks: P. Row ``r`` of ``inp`` is chunk ``r % P``, which is what makes the offset a
            per-group compile-time constant.
        chunk_width: ``V/P``, the per-chunk id offset unit.
        pad_rows: total rows to allocate, ``red_rows * P`` -- so the ``[red_rows, P*k]`` view the
            reduce wants exists, with its tail rows carrying the sentinel. Those rows are whole
            reduce ROWS whose results are never read, so their garbage indices cannot leak into
            a real row's answer.

    Returns:
        ``(cand_v [pad_rows, k] bf16, cand_i [pad_rows, k] f32)``, whose ``[pad_rows/P, P*k]``
        views are the reduce's input and the gather's data. Indices are GLOBAL vocab ids.
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k

    shard_info = get_verified_program_sharding_info("chunked_topk_stage1", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0
    per_lnc_BxS = div_ceil(BxS, n_prgs)

    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = PARTS_PER_GROUP * n_cols
    has_pad = full_n != vocab
    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (
        BxS == n_prgs * per_lnc_BxS
    )
    # The two preconditions of the whole scheme, asserted rather than assumed: the caller picks
    # this path on exactly these conditions, and if either were false the buffers below would not
    # be the candidate arrays (k_pad > k leaves unwritten columns interleaved with the valid ones)
    # or would not be bfloat16 (which nisa.topk requires of the reduce's source).
    kernel_assert(k_pad == k, f"_stage1_desnake_only needs k % 16 == 0, got k={k}")
    kernel_assert(fast_dma_safe, "_stage1_desnake_only needs whole 128-partition tiles")
    kernel_assert(
        pad_rows % chunks == 0,
        f"pad_rows ({pad_rows}) must be a multiple of chunks ({chunks})",
    )
    # chunk_width == 16*n_cols is what makes the chunk offset a whole number of snake
    # partitions, which is what lets it fold into the remap's multiply (see chunk_base).
    kernel_assert(
        not has_pad and chunk_width == PARTS_PER_GROUP * n_cols,
        f"_stage1_desnake_only needs 16 | chunk width, got {chunk_width}",
    )

    # inp may be a LIST of batch-group slab tensors (interleaved_batch_groups > 1,
    # jit-in-jit only): slab g holds tokens [g*tok_span, (g+1)*tok_span)'s receive layout
    # as its own tensor, so a stage-1 tile's loads data-depend only on ITS slab's producing
    # exchange. One tensor is the one-slab degenerate case.
    inp_slabs = list(inp) if isinstance(inp, (list, tuple)) else [inp]
    inp_flats = [t.reshape((int(np.prod(t.shape)),)) for t in inp_slabs]
    inp_flat = inp_flats[0]
    # --- The interleaved (all-to-all receive) source layout --------------------------------
    # peer_width > 0 says inp is the RECEIVE layout: token j's vocab column v = p*w + q lives
    # at flat address (p*BxS_tok + j)*w + q. The blocked snake load still wants vocab-in-chunk
    # index u = part*n_cols + col at snake (part, col) -- and because w | n_cols (asserted),
    # each snake partition covers WHOLE peer blocks, so the load stays a three-level pattern
    # with a UNIT innermost stride (see THE 4-BYTE DMA PROBLEM for why that matters):
    #     [[ppp*BxS_tok*w, 16], [BxS_tok*w, ppp], [1, w]]   ppp = n_cols // w
    #     offset(j, c) = c*(chunk_width//w)*BxS_tok*w + j*w
    # The snake CONTENT is identical to loading a de-interleaved [BxS_tok, vocab] -- only the
    # descriptors change -- so everything downstream (topk, de-snake, remap, ids) is bitwise
    # the same. What it deletes is the caller's de-interleave: a full HBM round trip of the
    # received logits. Cost: one DMA per 16-partition GROUP instead of one per 128-partition
    # tile (the partition stride is uniform only within a group).
    BxS_tok = BxS // chunks
    ppp = ppc = 0
    # tok_span: how many TOKENS share one peer-major interleave. One exchange (batch_groups
    # == 1) interleaves all BxS_tok tokens; a batch-group-slabbed receive (see
    # interleaved_batch_groups in ChunkedTopkConfig) interleaves tok_span = BxS_tok/G tokens
    # per slab, and token j's slab starts at (j // tok_span) * tok_span * vocab elements.
    tok_span = BxS_tok
    if peer_width:
        kernel_assert(
            not has_pad and n_cols % peer_width == 0 and chunk_width % peer_width == 0,
            f"interleaved_peer_width ({peer_width}) must divide the snake row width "
            f"({n_cols}) and the chunk width ({chunk_width})",
        )
        ppp = n_cols // peer_width
        ppc = chunk_width // peer_width
        kernel_assert(
            BxS_tok % batch_groups == 0,
            f"tokens ({BxS_tok}) must divide by groups ({batch_groups})",
        )
        tok_span = BxS_tok // batch_groups
    kernel_assert(
        len(inp_slabs) == max(1, batch_groups if peer_width else 1),
        f"got {len(inp_slabs)} input slabs for batch_groups={batch_groups}",
    )
    kernel_assert(
        _slab_sizes_match(
            [tuple(t.shape) for t in inp_slabs], tok_span * chunks * chunk_width
        ),
        f"each input slab must hold tokens-per-slab * vocab = {tok_span * chunks * chunk_width} elements, "
        f"got {[int(np.prod(t.shape)) for t in inp_slabs]}",
    )
    # See the desnake_dge comment in _inline_topk: unconditional, and the reason is a measured
    # silent index corruption, not tidiness.
    desnake_dge = nisa.dge_mode.unknown
    cand_v = nl.ndarray((pad_rows, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    cand_i = nl.ndarray((pad_rows, k), dtype=nl.float32, buffer=nl.private_hbm)
    cand_v_flat = cand_v.reshape((pad_rows * k,))
    cand_i_flat = cand_i.reshape((pad_rows * k,))

    # Sentinel the pad rows once. They are whole reduce rows, so BFLOAT16_MIN here cannot
    # outrank a real -inf in a REAL row -- it only has to lose within its own row, and that row's
    # result is never read. cand_i's pad rows are left as they are for the same reason.
    if pad_rows > BxS:
        n_pad = pad_rows - BxS
        for t in range(div_ceil(n_pad, PMAX)):
            rows = min(PMAX, n_pad - t * PMAX)
            pad_tile = nl.ndarray((PMAX, k), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=pad_tile[0:rows, :], value=BFLOAT16_MIN)
            nisa.dma_copy(
                dst=cand_v[nl.ds(BxS + t * PMAX, rows), :], src=pad_tile[0:rows, :]
            )

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # Coalesced + double-buffered load, exactly as in _inline_topk's phase 1.
    fast_coalesced = (
        (not has_pad)
        and (per_lnc_BxS % GROUPS_PER_TILE == 0)
        and (lnc_row_start + per_lnc_BxS <= BxS)
    )
    kernel_assert(
        peer_width == 0 or fast_coalesced,
        "interleaved_peer_width requires the coalesced whole-tile load path",
    )
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    # Interleaved path: one buffer PER TILE (four at the production shapes, ~3.1 MB of SBUF
    # that is entirely free post-collective). With buffer REUSE, a reused buffer's load
    # must wait for the sort still reading it, and everything queued behind that descriptor
    # stalls too -- measured as a ~7us gap between the first two sorts at gbs=1024. With no
    # reuse every load free-runs from the collective's completion. The contiguous path keeps
    # its historical 3 (it serves every other suite; don't perturb those schedules).
    N_PREFETCH_BUFS = min(4 if peer_width else 3, n_chunks)
    chunk_bufs = []

    def _load_chunk(dst3, first_s1_row, tiles, engine_phase):
        """One chunk-buffer fill from either source layout.

        Contiguous [BxS_tok, vocab]: the whole 128-partition tile is ONE run-per-partition
        DMA. Interleaved receive layout: the partition stride is uniform only WITHIN a
        16-partition group (each group is a different (token, chunk) with its own base), so
        it is one DMA per group -- 8 descriptors instead of 1, all unit-innermost-stride,
        ALTERNATING engines per group: descriptor generation (~0.9us per dma_copy on one
        engine) is what paces these loads, so splitting a tile's eight across both load
        engines halves its ready time.
        """
        if not peer_width:
            nisa.dma_copy(
                dst=dst3[nl.ds(0, PMAX), nl.ds(0, tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, tiles], [1, n_cols]],
                    offset=first_s1_row * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[engine_phase % 2],
            )
            return
        for tt in range(tiles):
            for grp in range(GROUPS_PER_TILE):
                r = first_s1_row + tt * GROUPS_PER_TILE + grp
                j = r // chunks
                c = r % chunks
                nisa.dma_copy(
                    dst=dst3[
                        nl.ds(grp * PARTS_PER_GROUP, PARTS_PER_GROUP),
                        nl.ds(tt, 1),
                        nl.ds(0, n_cols),
                    ],
                    # Token j's SLAB (index j // tok_span; slab 0 for every j at batch_groups
                    # == 1, where tok_span == BxS_tok), then the chunk and token offsets
                    # WITHIN that slab's peer-major interleave. At one slab this is the
                    # original pattern verbatim -- bitwise-identical descriptors.
                    src=inp_flats[j // tok_span].ap(
                        pattern=[
                            [ppp * tok_span * peer_width, PARTS_PER_GROUP],
                            [tok_span * peer_width, ppp],
                            [1, peer_width],
                        ],
                        offset=c * ppc * tok_span * peer_width
                        + (j % tok_span) * peer_width,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[(engine_phase + grp) % 2],
                )

    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(
                nl.ndarray(
                    (PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf
                )
            )
        for pc in range(min(N_PREFETCH_BUFS - 1, n_chunks)):
            prime_row_start = lnc_row_start + pc * CHUNK_TILES * GROUPS_PER_TILE
            prime_tiles = min(CHUNK_TILES, n_tiles - pc * CHUNK_TILES)
            _load_chunk(
                chunk_bufs[pc % N_PREFETCH_BUFS], prime_row_start, prime_tiles, pc
            )

    # The constant NaN floor, hoisted OUT of the tile loop: it is the same tile every time, and
    # at the GPT-OSS chunk width the memset covers 128 x 3,142 bf16, which is not a rounding
    # error when there are four tiles. Not built at all under the caller's finite-input
    # contract (config.skip_nan_fold) -- see the fold site in the tile loop.
    _nan_floor = None
    if not skip_nan_fold:
        _nan_floor = nl.ndarray((PMAX, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=_nan_floor, value=BFLOAT16_MIN)

    # --- The per-chunk global-id offset, as a PER-PARTITION CONSTANT ---------------------
    # The offset a row needs is (chunk index) * chunk_width, and within a tile the chunk index
    # is g % P where g = pp // 16 is the partition's group. The obvious spelling -- one
    # tensor_scalar add per group, on partitions [16g, 16g+16) -- IS NOT EXPRESSIBLE: the
    # hardware requires a 16-partition access to start at partition 0, 32, 64 or 96, and the
    # compiler rejects anything else ("Expected 'dst' partition start in {0, 32, 64, 96} for a
    # 16-partition access"). So build it as a tile instead, with iota's channel_multiplier,
    # which is the one thing that varies a generated value BY PARTITION.
    #
    # It is then folded into the remap rather than added afterwards. Because chunk_width is
    # exactly 16*n_cols (asserted below), the offset is a whole number of snake partitions:
    #     global = (r % P)*chunk_width + (s % 16)*n_cols + (s // 16)
    #            = (16*(r % P) + (s % 16)) * n_cols + (s // 16)
    # where r is the s1 ROW index (row j*P + c is chunk c of token j, so chunk == r % P).
    # so it costs one uint32 add on a k_cols-wide tile instead of P-1 adds on a P*k-wide one.
    #
    # The row's chunk is derived from the PARTITION index: iota seeds 16*(first_row % P) + pp
    # and a bitwise_and with (16P - 1) & ~15 (both powers of two minus one; pick_chunks only
    # returns power-of-two P) reduces it to 16*((first_row + pp//16) % P) exactly. When P
    # divides GROUPS_PER_TILE the tile's first row is always 0 mod P, so ONE hoisted tile
    # serves every iteration; for P > GROUPS_PER_TILE the residue changes per tile and the
    # base must be rebuilt with that tile's first row (a hoisted P<=8 tile reused at P=16
    # gave every row past tile 0 an id short by GROUPS_PER_TILE*chunk_width -- in-range,
    # silently wrong).
    _base_mask = (PARTS_PER_GROUP * chunks - 1) & ~(PARTS_PER_GROUP - 1)

    def _build_chunk_base(first_row):
        cb = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(
            dst=cb,
            pattern=[[0, k_cols]],
            offset=PARTS_PER_GROUP * (first_row % chunks),
            channel_multiplier=1,
        )
        nisa.tensor_scalar(dst=cb, data=cb, op0=nl.bitwise_and, operand0=_base_mask)
        return cb

    chunk_base = _build_chunk_base(0) if GROUPS_PER_TILE % chunks == 0 else None

    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue
        par_dim = n_rows * PARTS_PER_GROUP

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                _load_chunk(
                    chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS],
                    pf_row_start,
                    pf_tiles,
                    prefetch_chunk,
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][
                nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)
            ]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, par_dim], [1, n_cols]],
                    offset=tile_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # -inf, NOT the most negative finite bf16: a finite sentinel outranks a real -inf
            # logit and pulls padding into the top-k. See _inline_topk's memset comment.
            nisa.memset(dst=src_snake, value=float("-inf"))
            for g in nl.affine_range(n_rows):
                row_off = (tile_row_start + g) * vocab
                base = g * PARTS_PER_GROUP
                q_full = vocab // n_cols
                r_partial = vocab % n_cols
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(
                            pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off
                        ),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(
                            pattern=[[1, r_partial]], offset=row_off + q_full * n_cols
                        ),
                    )

        # Fold NaN out of the snake BEFORE nisa.topk selects from it: afterwards the instruction
        # has already emitted a (value, position) pair that does not correspond, and no later
        # clamp can undo that. (x != x) is true only for NaN. Skipped under the caller's
        # finite-input contract (config.skip_nan_fold): with no NaN the predicate can never
        # fire, yet the compare + predicated copy span the full chunk width -- the widest
        # non-topk work in this loop. +/-inf are orderable and never needed the fold.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(
            val_dst=val_snake[:, nl.ds(0, k)],
            idx_dst=idx_snake[:, nl.ds(0, k)],
            src=src_snake,
            n=full_n,
        )

        # Stage the positions through a dge_mode.unknown DMA before ANY compute engine reads
        # them. The remap below used to read idx_snake directly on Vector -- a direct
        # GpSIMD-write -> Vector-read edge that neither gpsimd_topk's phase 2 (which reads its
        # positions only after the de-snake store/reload round trip) nor any other path in this
        # file has. That edge is timing-unsafe: with the debug taps shifting the schedule, the
        # bxs=128 seed sweep FAILED THE PAIRING GATE on hardware (58 slots on one row, ids
        # plausible and in-range, values exactly right) and the L2 tap named this remap -- the
        # Vector ops had read idx_snake before nisa.topk's stores were visible, exactly the
        # store-visibility defect measured on nc_n_gather in the fused tail. The staging DMA is
        # the same handoff the de-snake stores themselves use (descriptor generation on the
        # GpSIMD queue, completion on a real DMA semaphore), which is the highest-validated
        # GpSIMD-producer idiom in this kernel.
        idx_stage = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_stage[nl.ds(0, par_dim), :],
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

        # --- The remap, ON THE SNAKE TILE: k_cols wide, not k ---
        # The blocked load put vocab index p*n_cols + c at snake position s = p + 16*c, so the
        # chunk-local id is (s % 16)*n_cols + (s // 16), and the global id folds the chunk
        # offset into the same multiply (see chunk_base):
        #     global = (16*(g % P) + (s % 16)) * n_cols + (s // 16)
        # 16 is a power of two, so the split is EXACT with bit-ops on the uint32 the instruction
        # already reports -- and unlike the phase-2 version there is no float32 -> uint32 cast
        # first, because idx_stage IS uint32.
        c_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        q_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        gid = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.right_shift,
            operand0=4,  # log2(PARTS_PER_GROUP); an int method call is rejected by the parser
        )
        # Read from idx_stage, NOT from c_u32: two ops reading one source and writing two
        # distinct destinations have no hazard between them, whereas chaining them through a
        # shared buffer creates exactly the write-after-read pair that has bitten this kernel.
        nisa.tensor_scalar(
            dst=q_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        # + 16*(row % P), the chunk offset expressed in snake partitions. Both operands are
        # uint32 and bounded by 15 and 16*(P-1) <= 2032, so the sum cannot overflow anything
        # (and the fp32 gid below stays exact: ids are < 2^24 by kernel contract).
        _cb = (
            chunk_base if chunk_base is not None else _build_chunk_base(tile_row_start)
        )
        nisa.tensor_tensor(
            dst=q_u32[nl.ds(0, par_dim), :],
            data1=q_u32[nl.ds(0, par_dim), :],
            data2=_cb[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        nisa.tensor_copy(
            dst=c_f32[nl.ds(0, par_dim), :], src=c_u32[nl.ds(0, par_dim), :]
        )
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=q_u32[nl.ds(0, par_dim), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=gid[nl.ds(0, par_dim), :],
            data1=gid[nl.ds(0, par_dim), :],
            data2=c_f32[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        # Every id must land in [0, full vocab). It can only escape if the snake position was
        # itself garbage, which happens when the values are not orderable (nisa.topk's reported
        # position is then undefined). Such an id is already unpaired from its value, so the only
        # thing left to guarantee is the range -- an out-of-range id handed to a table gather
        # faults the device. Clamping to the FULL vocab rather than to this chunk is a deliberate
        # weakening: the offset is now folded in before the clamp, so a garbage id can land in a
        # neighbouring chunk's range. Both are in-contract, and neither is paired.
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=gid[nl.ds(0, par_dim), :],
            op0=nl.minimum,
            operand0=float(chunks * vocab - 1),
        )

        # TAP: the RAW snake positions, de-snaked through the same pattern as everything else so
        # the tap means exactly what it means on the general path. It reads idx_stage -- the
        # staged buffer the remap consumed -- so an L2 break really does mean the remap's own
        # arithmetic disagreed with its own input, not that the two taps read one racy buffer
        # at different times.
        if tap_pos is not None:
            nisa.dma_copy(
                dst=tap_pos.reshape((BxS * k,)).ap(
                    pattern=[[k_cols, par_dim], [1, k_cols]],
                    offset=tile_row_start * k_pad,
                ),
                src=idx_stage[nl.ds(0, par_dim), :],
                dge_mode=desnake_dge,
            )

        # De-snake straight into the candidate arrays. One contiguous par_dim-partition DMA per
        # tensor per tile, as in _inline_topk -- the only difference is where it lands.
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=cand_v_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=cand_i_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=gid[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    return cand_v, cand_i


def _inline_topk(
    inp: nl.NkiTensor, config, tap_pos=None, fused=None, skip_nan_fold=False
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, V]`` with ``nisa.topk`` -- inlined copy of ``gpsimd_topk``.

    Args:
        inp: ``[BxS, V]`` bfloat16 tensor in HBM, ``8 <= V < 65536``.
        config: a ``GpsimdTopkConfig`` (build it with ``create_gpsimd_topk_config``).
        tap_pos: optional ``[BxS, k]`` float32 HBM tap. Receives the RAW snake positions phase 2
            worked from, before the snake->vocab remap. See DEBUG TAPS in the module docstring.
        fused: optional tuple (slots ``_FUSE_*``) that replaces the plain output store with the
            REDUCE'S TAIL: translate this call's positions through the caller's candidate-id
            array and write the final values and global ids directly. Only meaningful for the
            second stage, whose "vocab index" IS a candidate position. See WHY THE REDUCE FUSES
            ITS TAIL below.

    Returns:
        ``(values, indices, desnaked_idx [BxS, k_pad] f32)``. Without ``fused`` the first two are
        this call's own ``[BxS, k]`` private_hbm buffers, holding indices into ``inp``'s last
        dimension, descending when ``config.sorted``; with ``fused`` they are the caller's own
        destinations, already carrying global ids. The third element is the phase-1 de-snake
        index buffer, handed back so a caller can re-read it (the tap that distinguishes a
        late-landing store from a bad top-k) -- returning a handle to a buffer that already
        exists costs nothing.
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k
    index_dtype = config.index_dtype

    # Resolve runtime sharding (LNC). BxS == 1 collapses to a single program. chunked_topk has
    # already asserted n_prgs == 1, so this only ever resolves to (1, 0) here; it is kept
    # verbatim so the inline stays a faithful copy.
    shard_info = get_verified_program_sharding_info("gpsimd_topk", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    # Without fusion this stage publishes its own [BxS, k] results; with it the caller's
    # destinations are written in place and these are never allocated.
    if fused is None:
        topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.private_hbm)
        topk_indices = nl.ndarray((BxS, k), dtype=index_dtype, buffer=nl.private_hbm)
    else:
        topk_values = fused[_FUSE_OUT_VALS]
        topk_indices = fused[_FUSE_OUT_IDX]

    # Snake free-dim sizes (BLOCKED layout: snake[p, c] = inp[row, p*n_cols + c], so snake
    # position s = p + 16*c holds vocab index p*n_cols + c -- inverted by the remap below).
    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = (
        PARTS_PER_GROUP * n_cols
    )  # the n passed to nisa.topk; padded slots hold -inf
    q_full = vocab // n_cols  # fully-valid partitions
    r_partial = vocab % n_cols  # valid elements in the partial partition q_full
    has_pad = full_n != vocab  # some snake slots are padding (vocab % 16 != 0)

    inp_flat = inp.reshape((BxS * vocab,))

    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP  # >= k; padded snake-column width

    # fast_dma_safe: this shard's phase-1 tiles are ALL full 128-partition tiles, which is what
    # makes a bf16 de-snake round trip safe against the narrow-dtype strided store. It does NOT
    # select the DGE mode -- see the desnake_dge comment.
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (
        BxS == n_prgs * per_lnc_BxS
    )
    # Sorted output is served EXCLUSIVELY by the hardware sort -- the software sort was
    # deleted (see THE SOFTWARE SORT IS GONE in the module docstring). _validate_chunked_topk
    # has already refused a sorted request without the capability; this assert is the
    # belt-and-suspenders for any direct caller of this helper.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE and k <= _HW_SORT_MAX_K,
            "sorted top-k requires nisa.topk(sorted=True) (patched nki + neuronx-cc >= 2.0.280935) "
            f"and k <= {_HW_SORT_MAX_K}; the software sort was removed",
        )
    hw_sort = bool(config.sorted)
    # dge_mode.unknown UNCONDITIONALLY. dge_mode.none pre-generates the de-snake descriptors
    # out-of-band and takes the store off GpSIMD, so nothing orders it against the phase-2
    # reload of the same private_hbm region; measured at 128 ranks, 2 of 128 row blocks came
    # back with REPEATED vocab ids and exactly-right values. Costs +2.7%. See THE STORE/RELOAD
    # RACE IS REAL in gpsimd_topk -- and its warning that a variant which merely narrows the
    # window passes a 128-rank run by luck.
    desnake_dge = nisa.dge_mode.unknown
    asc_val_dtype = nl.bfloat16 if fast_dma_safe else nl.float32
    asc_val_hbm = nl.ndarray((BxS, k_pad), dtype=asc_val_dtype, buffer=nl.private_hbm)
    asc_idx_hbm = nl.ndarray((BxS, k_pad), dtype=nl.float32, buffer=nl.private_hbm)
    asc_val_flat = asc_val_hbm.reshape((BxS * k_pad,))
    asc_idx_flat = asc_idx_hbm.reshape((BxS * k_pad,))

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # --- Coalesced + double-buffered phase-1 load ------------------------------------------
    # When 16 | vocab and every tile is full, CHUNK_TILES tiles are ONE contiguous HBM block
    # (partition stride n_cols, tile stride 128*n_cols, free stride 1), so the load is a few
    # large HWDGE transfers into rolling buffers instead of one latency-bound transfer per
    # tile. N_PREFETCH_BUFS distinct buffers keep loads in flight while GpSIMD drains another.
    fast_coalesced = (
        (not has_pad)
        and (per_lnc_BxS % GROUPS_PER_TILE == 0)
        and (lnc_row_start + per_lnc_BxS <= BxS)
    )
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    # Round-robin the loads across two DGE sequencer queues so consecutive loads run
    # concurrently instead of serializing on one sync queue.
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    N_PREFETCH_BUFS = min(3, n_chunks)
    chunk_bufs = []
    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(
                nl.ndarray(
                    (PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf
                )
            )
        n_prime = min(N_PREFETCH_BUFS - 1, n_chunks)
        for pc in range(n_prime):
            prime_first_tile = pc * CHUNK_TILES
            prime_row_start = lnc_row_start + prime_first_tile * GROUPS_PER_TILE
            # The last chunk may hold fewer tiles; clamp so the DMA never over-reads.
            prime_tiles = min(CHUNK_TILES, n_tiles - prime_first_tile)
            nisa.dma_copy(
                dst=chunk_bufs[pc % N_PREFETCH_BUFS][
                    nl.ds(0, PMAX), nl.ds(0, prime_tiles), nl.ds(0, n_cols)
                ],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, prime_tiles], [1, n_cols]],
                    offset=prime_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[pc % 2],
            )

    # =====================================================================
    # PHASE 1 -- per-8-row tile: blocked load -> nisa.topk -> de-snake the K
    # (value, snake-index) pairs into the contiguous [BxS, k_pad] HBM buffers.
    # The descending sort is NOT done here: it runs ONCE in phase 2 over all
    # this core's rows (one row per partition, up to 128), because max8 /
    # nc_match_replace8 / nc_n_gather are per-partition free-dim ops, so
    # widening the sort from 8 to 128 partitions is free.
    # =====================================================================
    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue

        par_dim = n_rows * PARTS_PER_GROUP  # multiple of 16

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            # Keep the pipeline full: at the first tile of a chunk, prefetch the chunk
            # (N_PREFETCH_BUFS-1) ahead into a slot nothing is still consuming, so no
            # anti-dependency forces the load to wait.
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                nisa.dma_copy(
                    dst=chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS][
                        nl.ds(0, PMAX), nl.ds(0, pf_tiles), nl.ds(0, n_cols)
                    ],
                    src=inp_flat.ap(
                        pattern=[
                            [n_cols, PMAX],
                            [PMAX * n_cols, pf_tiles],
                            [1, n_cols],
                        ],
                        offset=pf_row_start * vocab,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[prefetch_chunk % 2],
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][
                nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)
            ]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # 16 | vocab but the coalesced precondition does not hold. The whole-tile blocked
            # load is still ONE contiguous HBM run (partition stride n_cols, free stride 1),
            # emitted as a single par_dim-partition DMA. HWDGE on Sync keeps descriptor
            # generation off GpSIMD so the load overlaps the previous tile's top-k.
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, par_dim], [1, n_cols]],
                    offset=tile_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # vocab % 16 != 0: some snake slots are padding. The sentinel is -inf, NOT the
            # most-negative FINITE bf16: a finite sentinel outranks a real -inf logit, so on a
            # masked row padding wins the top-k and the kernel returns a value the input never
            # contained paired with a clamped index. See gpsimd_topk's memset comment.
            nisa.memset(dst=src_snake, value=float("-inf"))

            for g in nl.affine_range(n_rows):
                row = tile_row_start + g
                base = g * PARTS_PER_GROUP
                row_off = row * vocab
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(
                            pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off
                        ),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(
                            pattern=[[1, r_partial]], offset=row_off + q_full * n_cols
                        ),
                    )

        # Fold NaN out of the snake BEFORE nisa.topk selects from it -- the earliest point NaN
        # can do damage, and the phase-2 clamps are far too late (the topk has already emitted
        # a (value, position) pair that does not correspond). (x != x) is true only for NaN.
        # Fold to BFLOAT16_MIN, not NEG_INF_FLOOR: src_snake is BF16, which cannot hold the
        # floor finitely. See the phase-1 NaN-fold comment in gpsimd_topk. The REDUCE stage
        # skips this (skip_nan_fold): its input is stage 1's own output plus BFLOAT16_MIN pad
        # rows, and stage 1 already folded -- so its fold can never fire, and it sat as two
        # full-width Vector ops on the reduce's critical path anyway.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            _nan_floor = nl.ndarray(
                (par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.memset(dst=_nan_floor[nl.ds(0, par_dim), :], value=BFLOAT16_MIN)
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        # n = full_n (NOT vocab): with the blocked layout the valid vocab elements are not a
        # contiguous prefix of snake positions, so all full_n slots are active.
        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        if hw_sort:
            # sorted=True puts rank j at snake position j, which is what lets phase 2 replace
            # the entire descending sort with a fixed column permutation. Two call forms because
            # passing `sorted=` to an nki without it is a trace-time TypeError.
            nisa.topk(
                val_dst=val_snake[:, nl.ds(0, k)],
                idx_dst=idx_snake[:, nl.ds(0, k)],
                src=src_snake,
                n=full_n,
                sorted=True,
            )
        else:
            nisa.topk(
                val_dst=val_snake[:, nl.ds(0, k)],
                idx_dst=idx_snake[:, nl.ds(0, k)],
                src=src_snake,
                n=full_n,
            )
        # k_pad > k leaves snake positions [k, k_pad) unwritten. They live at a PARTITION
        # SUBRANGE here, which a full-partition memset cannot target, so they are masked after
        # the de-snake reload in phase 2 where the layout makes it a legal free-dim write.
        # (The hw_sort path needs no masking at all: it reads BY POSITION, and positions
        # [k, k_pad) are never among the ranks it asks for.)

        # --- De-snake the topk output into a CONTIGUOUS [rows, k_pad] HBM tile ---
        # asc[row, p*k_cols + c] = val_snake[base + p, c]. Because k_pad == 16*k_cols,
        # consecutive row-groups are exactly k_pad apart, so the WHOLE par_dim-partition tile
        # is ONE contiguous HBM run: a single DMA per tile rather than n_rows of them (the
        # de-snake stores were the top serial contributors in the HW profile).
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=asc_val_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=asc_idx_flat.ap(
                pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off
            ),
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    # =====================================================================
    # PHASE 2 -- batch-parallel descending sort, ONE invocation over up to 128
    # rows (one row per partition) rather than one per 8-row tile.
    # =====================================================================
    n_pass = div_ceil(k, 8)
    log2_ppg = 4  # == log2(PARTS_PER_GROUP)
    n_sort_tiles = div_ceil(per_lnc_BxS, PMAX)

    for sort_idx in nl.sequential_range(n_sort_tiles):
        sort_row_start = lnc_row_start + sort_idx * PMAX
        sort_row_end = min(sort_row_start + PMAX, lnc_row_end)
        n_srows = sort_row_end - sort_row_start
        if n_srows <= 0:
            continue

        out_v = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        out_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)

        if hw_sort:
            # ===== THERE IS NO SORT ============================================
            # nisa.topk(sorted=True) emitted rank j at snake position j, and the de-snake places
            # snake position pos at column (pos % 16)*k_cols + (pos // 16). Both are compile-time
            # facts, so recovering descending order is a FIXED PERMUTATION of columns. Writing
            # j = a + 16b with a = j % 16 and b = j // 16, rank j lives at row offset
            # a*k_cols + b -- so a [k_cols, 16] destination whose flat column index is b*16 + a
            # holds the ranks in order once source column a*k_cols + b lands at (b, a).
            #
            # The reload itself -- which the software path performs anyway -- stays ONE
            # CONTIGUOUS RUN PER ROW, and the permutation runs ON CHIP: 16 tensor_copy ops per
            # tensor, each moving the contiguous k_cols-wide block for one value of `a` into the
            # strided [:, :, a] slice. Folding the permutation into the reload's access pattern
            # instead is NOT free -- the last pattern level is the fastest-varying, so it makes
            # every element its own 4-byte transfer (+1,968 packets, +50 us at gbs=128; see THE
            # 4-BYTE DMA PROBLEM in the module docstring, including why landing this exposed a
            # latent race elsewhere).
            #
            # What the branch deletes, per sort tile: 16 max8, 16 nc_match_replace8, 16 gathers,
            # the two reverse-gathers, 8 bitonic merge stages of 7 ops each, three synchronising
            # SBUF->SBUF DMAs, and the two sort-key clamps. About 110 instructions become 34.
            #
            # It also removes two whole classes of hazard rather than mitigating them:
            #   * the value-addressed index recovery. nc_match_replace8 found a value's position
            #     by equality, so duplicate extremes left a dst_idx UNDEFINED and a real -inf
            #     could alias the marker it stamps -- the defect _clamp_sort_keys and
            #     NEG_INF_FLOOR exist to work around. Here the instruction reports the index of
            #     the value it selected, so there is nothing to look up and nothing to alias.
            #     Both clamps are therefore absent, not merely unnecessary.
            #   * the k_pad > k padding. The software path must mask the unwritten snake
            #     positions [k, k_pad) because garbage there wins max8. This path never reads
            #     them: it asks for ranks, and no rank maps to an unwritten position.
            #
            # The NaN fold in phase 1 STAYS. The ISA is explicit that "the behavior for NaN
            # inputs is undefined; a NaN may appear in the output spuriously" -- sorting does not
            # make an unorderable key orderable, so it must still be folded before selection.
            pv = nl.ndarray(
                (PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf
            )
            pi = nl.ndarray(
                (PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf
            )
            lv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            li = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=li[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            for _a in range(PARTS_PER_GROUP):
                nisa.tensor_copy(
                    dst=pv[nl.ds(0, n_srows), :, _a],
                    src=lv[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)],
                )
                nisa.tensor_copy(
                    dst=pi[nl.ds(0, n_srows), :, _a],
                    src=li[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)],
                )
            # The 3D tile's free dims are contiguous, so this view is the [n_srows, k_pad]
            # descending array the shared tail below expects. k <= k_pad, so the tail's
            # [:, 0:k] slice takes ranks 0..k-1.
            out_v = pv.reshape((PMAX, k_pad))
            out_i = pi.reshape((PMAX, k_pad))
        else:
            # ===== Unsorted fast path (config.sorted == False) ================
            # Skip the sort; COMPACT the valid de-snaked columns into the output prefix
            # [0, k). nisa.topk writes only snake positions [0, k), and the de-snake scatters
            # those among the k_pad columns when k % 16 != 0; for k % 16 == 0 this is the
            # single full-width run (0, 0, k).
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            _valid_runs = _valid_desnake_col_runs(k, k_cols, k_pad)
            # The parser frontend rejects tuple-unpacking in a for-target, so index explicitly.
            for _run_idx in range(len(_valid_runs)):
                _src_col = _valid_runs[_run_idx][0]
                _dst_col = _valid_runs[_run_idx][1]
                _run_len = _valid_runs[_run_idx][2]
                nisa.tensor_copy(
                    dst=out_v[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=cv[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )
                nisa.tensor_copy(
                    dst=out_i[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=ci[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )

        # TAP: the raw snake positions phase 2 ended up with, before the remap. Reads out_i
        # full-width, which the remap immediately below does too, so it adds no ordering edge
        # that is not already there -- the property that keeps a tap from hiding the race it is
        # meant to catch (see DEBUG TAPS).
        if tap_pos is not None:
            nisa.dma_copy(
                dst=tap_pos[nl.ds(sort_row_start, n_srows), :],
                src=out_i[nl.ds(0, n_srows), nl.ds(0, k)],
            )

        # --- Blocked-layout snake-position -> vocab-index remap ---
        # The blocked load put vocab index p*n_cols + c at snake position s = p + 16*c, so
        # v = (s % 16) * n_cols + (s // 16). 16 is a power of two, so the split is EXACT via
        # bit-ops on uint32 (c = s >> 4, p = s & 15); the recombine is float32 (exact for
        # v < 2^24) and casts back to uint32 on the store.
        # Exactly k columns wide, not the sort buffers' width: only [0, k) is ever stored, and
        # the branches above leave out_i at different widths (n_pass*8 for the unsorted compaction,
        # k_pad for the hardware one), so k is both the cheapest and the only common one.
        s_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        p_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        remapped_i = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=s_u32[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), nl.ds(0, k)]
        )
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.right_shift,
            operand0=log2_ppg,
        )
        nisa.tensor_scalar(
            dst=p_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        nisa.tensor_copy(
            dst=c_f32[nl.ds(0, n_srows), :], src=c_u32[nl.ds(0, n_srows), :]
        )
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=p_u32[nl.ds(0, n_srows), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data1=remapped_i[nl.ds(0, n_srows), :],
            data2=c_f32[nl.ds(0, n_srows), :],
            op=nl.add,
        )
        # Every returned index must be in [0, vocab). Two ways it could escape: a padding slot
        # that tied real -inf data and remapped up to full_n - 1, or a gathered snake position
        # that was itself garbage. An out-of-range index breaks the contract and, if it feeds a
        # table gather, faults the device. Unconditional -- one Vector op on k columns.
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=remapped_i[nl.ds(0, n_srows), :],
            op0=nl.minimum,
            operand0=float(vocab - 1),
        )

        if fused is None:
            nisa.dma_copy(
                dst=topk_values[nl.ds(sort_row_start, n_srows), :],
                src=out_v[nl.ds(0, n_srows), nl.ds(0, k)],
            )
            nisa.dma_copy(
                dst=topk_indices[nl.ds(sort_row_start, n_srows), :],
                src=remapped_i[nl.ds(0, n_srows), nl.ds(0, k)],
            )
            continue

        # ===== WHY THE REDUCE FUSES ITS TAIL ==============================================
        # For the second stage `remapped_i` is not a vocab index at all -- this call's "vocab"
        # IS the candidate array -- so it is a POSITION that still has to be translated through
        # the caller's chunk-local ids. Unfused, that translation costs a full round trip: store
        # the positions to HBM as uint32, reload them, widen back to float32, and separately
        # store the values only for the caller to reload and re-store them. Every one of those
        # is removed here. The positions are already in SBUF, already float32, and already
        # clamped into [0, red_w) by the range clamp above -- which is exactly what
        # nc_n_gather requires of its indices -- so they can be used as they stand.
        n_out = fused[_FUSE_N_ROWS]
        rows = min(n_srows, n_out - sort_row_start)
        if rows <= 0:
            continue
        p_chunks = fused[_FUSE_CHUNKS]
        chunk_width = fused[_FUSE_WIDTH]
        red_w = p_chunks * k

        # Values: ONE fp32 SBUF -> bf16 HBM DMA, the cast riding on the transfer.
        nisa.dma_copy(
            dst=fused[_FUSE_OUT_VALS][nl.ds(sort_row_start, rows), :],
            src=out_v[nl.ds(0, rows), nl.ds(0, k)],
        )

        # Index normalisation. fp32 throughout: every global id is < vocab <= 2^24, so the
        # uint32 -> fp32 -> uint32 round trip is exact.
        if fused[_FUSE_IDS_GLOBAL]:
            # _stage1_desnake_only already remapped and offset the ids, on a k_cols-wide tile,
            # and wrote them as float32 -- and the Tensor-Engine lookup below reads them
            # STRAIGHT FROM HBM (the d_all loads), so no SBUF copy of the full candidate row
            # exists on this path at all. Only the cand_i tap, which captures exactly what the
            # lookup consumes, materialises one.
            i_f = None
            if fused[_FUSE_TAP_CAND] is not None:
                i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=i_f[0:rows, :],
                    src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :],
                )
        else:
            i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
            iu = nl.ndarray((PMAX, red_w), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=iu[0:rows, :],
                src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :],
            )
            nisa.tensor_copy(dst=i_f[0:rows, :], src=iu[0:rows, :])
            # global = c*width + chunk_local. c is constant per column block, so this is a
            # free-dim slice add per chunk. Chunk 0 needs no offset.
            for c in range(1, p_chunks):
                nisa.tensor_scalar(
                    dst=i_f[0:rows, nl.ds(c * k, k)],
                    data=i_f[0:rows, nl.ds(c * k, k)],
                    op0=nl.add,
                    operand0=float(c * chunk_width),
                )
        if fused[_FUSE_TAP_CAND] is not None:
            nisa.dma_copy(
                dst=fused[_FUSE_TAP_CAND][nl.ds(sort_row_start, rows), :],
                src=i_f[0:rows, :],
            )
        if fused[_FUSE_TAP_POS] is not None:
            nisa.dma_copy(
                dst=fused[_FUSE_TAP_POS][nl.ds(sort_row_start, rows), :],
                src=remapped_i[0:rows, nl.ds(0, k)],
            )

        # ===== THE POSITION -> ID LOOKUP RUNS ON THE TENSOR ENGINE, NOT ON A GATHER =========
        # got[r, j] = i_f[r, pos] with pos = remapped_i[r, j] is the one data-dependent lookup
        # in the kernel, and nc_n_gather CANNOT BE MADE SAFE HERE. That is a measured statement,
        # not a cautious one -- four escalating fence constructions all failed, each localised by
        # debug_taps (the L4 tap) plus the Instruction/Flow tables of its failing NEFF:
        #   * a Vector consumer with a correct FLOW_DEPENDENCE edge, dispatched 500 ns after the
        #     gather retired, read the row TAIL of its output as stale zeros;
        #   * a default-mode SBUF->SBUF DMA fence read the same mid-flight state (its descriptors
        #     are generated out-of-band, so nothing places it after the gather);
        #   * a same-dtype GpSIMD COPY fence was copy-propagated away by the compiler, re-exposing
        #     exactly the window whose fence was elided;
        #   * with an unfoldable GpSIMD CAST fence, the profile shows the Vector consumer STARTING
        #     119 ns into the cast's own 900 ns execution -- its semaphore wait already satisfied
        #     -- and two GpSIMD instructions visibly overlapping on the engine. A GpSIMD
        #     instruction's completion signal does not mean its stores are visible, and same-queue
        #     issue order does not serialise execution either.
        # nisa.topk's own consumers at SHORTER distances are always correct, so this is the gather
        # ucode's early completion signal, not a property of GpSIMD stores in general (the
        # multi-group corner of the same defect). The sort paths' gathers keep their historical
        # sync-DMA fences and their far-downstream consumers; THIS site had neither distance nor a
        # working fence, so the gather is not fenced here -- it is REMOVED.
        #
        # The replacement is an exact one-hot contraction on the Tensor Engine, whose PSUM handoff
        # to Vector is the most-validated producer/consumer edge on the chip. Split each position
        # as pos = q*c_w + c (c_w = min(128, red_w)):
        #     D[(g, q), c] = i_f[r0+g, q*c_w + c]                 the group's candidates, fp32
        #     B[(g, q), (g', j)] = (g*n_q + q == key[g', j])      key = g'*n_q + q_{g',j}
        #     M[c, (g', j)] = sum_p D[p, c] * B[p, (g', j)]       = i_f[r0+g', q_j*c_w + c]
        #     got[g', j] = sum_c M[c, (g', j)] * (c == c_{g',j})  the c-selection + ones-reduce
        # EXACTNESS: everything is fp32. Ids are < vocab <= 2^24, so fp32 holds them exactly; keys
        # and iotas are small integers; each one-hot column matches exactly one partition, so every
        # PSUM sum is one exact product plus zeros -- no accumulation error exists to reason about.
        # A padded tail chunk (c_w does not divide red_w) is memset to 0.0 so the matmul multiplies
        # B's zeros against 0.0 rather than uninitialised bits (0 * NaN is NaN).
        c_w = min(PMAX, red_w)
        n_q = div_ceil(red_w, c_w)
        # Rows per matmul group: the block-diagonal key trick batches g rows into one contraction
        # so long as their K fits the partition dim and their flattened outputs fit one PSUM-bank
        # moving tile. k > _PSUM_MOVING falls back to one row per group, tiled along k.
        _PSUM_MOVING = 512
        g_max = max(1, min(PMAX // n_q, _PSUM_MOVING // k if k <= _PSUM_MOVING else 1))
        # Shared small constants, sliced per use: one integer ramp down the partitions (serves
        # both the K-side and the c-side one-hot compares) and a ones vector for the broadcasts
        # and the partition reduction.
        iota_pf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        _iota_pu = nl.ndarray((PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(dst=_iota_pu, pattern=[[0, 1]], channel_multiplier=1)
        nisa.tensor_copy(dst=iota_pf, src=_iota_pu)
        ones_row = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)
        ones_col = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        # bf16 twin of the ones vector, for the two BROADCAST matmuls (key_b, c_b) below.
        # Those carry only small integers (keys < 128, c < c_w <= 128 -- all exact in bf16's
        # 8 significand bits) and every PSUM element is ONE exact product, so running them at
        # the PE's bf16 rate instead of the quarter-rate fp32 mode is BITWISE free. The id
        # contraction (m_ps) and the ones-reduce (sel) carry vocab ids up to 2^24 and STAY
        # fp32 -- see the exactness paragraph above.
        ones_row_bf = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_row_bf, value=1.0)

        for r0 in range(0, rows, g_max):
            g = min(g_max, rows - r0)
            n_flat = g * k
            k_g = g * n_q
            # The group's positions, flattened onto one partition, and split into (q, c). The
            # per-row DMAs are partition moves; everything after them is exact fp32 arithmetic.
            pos_flat = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for gi in range(g):
                nisa.dma_copy(
                    dst=pos_flat[0:1, nl.ds(gi * k, k)],
                    src=remapped_i[nl.ds(r0 + gi, 1), nl.ds(0, k)],
                )
            key_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            c_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            key_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            c_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            if n_q == 1:
                # red_w <= 128: q is identically zero and c is the position itself.
                nisa.tensor_copy(dst=c_f[0:1, :], src=pos_flat[0:1, :])
                nisa.memset(dst=key_f, value=0.0)
            else:
                # c_w == 128 here (red_w > 128), so the split is exact bit arithmetic.
                pos_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                q_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                c_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pos_u[0:1, :], src=pos_flat[0:1, :])
                nisa.tensor_scalar(
                    dst=q_u[0:1, :], data=pos_u[0:1, :], op0=nl.right_shift, operand0=7
                )
                nisa.tensor_scalar(
                    dst=c_u[0:1, :],
                    data=pos_u[0:1, :],
                    op0=nl.bitwise_and,
                    operand0=PMAX - 1,
                )
                nisa.tensor_copy(dst=key_f[0:1, :], src=q_u[0:1, :])
                nisa.tensor_copy(dst=c_f[0:1, :], src=c_u[0:1, :])
            if g > 1:
                # key = g'*n_q + q: fold the row-within-group block into the key so ONE compare
                # builds the block-diagonal one-hot.
                for gi in range(g):
                    nisa.tensor_scalar(
                        dst=key_f[0:1, nl.ds(gi * k, k)],
                        data=key_f[0:1, nl.ds(gi * k, k)],
                        op0=nl.add,
                        operand0=float(gi * n_q),
                    )

            # bf16 twins of the finished key/c vectors, for the broadcast matmuls: keys and
            # c are integers below 128/c_w, exact in bf16, so this is a pure rate change.
            nisa.tensor_copy(dst=key_bf[0:1, :], src=key_f[0:1, :])
            nisa.tensor_copy(dst=c_bf[0:1, :], src=c_f[0:1, :])

            # The group's candidate ids, reshaped one row -> n_q partition chunks of c_w. A DMA
            # cannot fan one SBUF partition out to n_q partitions, so the reshape happens where
            # each path can afford it: the bypass path's ids are already global float32 in HBM,
            # where an access pattern reshapes for free (contiguous c_w-runs); the general path
            # computed i_f in SBUF, so it moves one single-partition chunk at a time.
            d_all = nl.ndarray((PMAX, c_w), dtype=nl.float32, buffer=nl.sbuf)
            if n_q * c_w != red_w:
                nisa.memset(dst=d_all[0:k_g, :], value=0.0)
            q_full = red_w // c_w
            rem = red_w % c_w
            if fused[_FUSE_IDS_GLOBAL]:
                cand_flat = fused[_FUSE_CAND_I].reshape(
                    (fused[_FUSE_CAND_I].shape[0] * red_w,)
                )
                for gi in range(g):
                    row_off = (sort_row_start + r0 + gi) * red_w
                    if q_full > 0:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q, q_full), nl.ds(0, c_w)],
                            src=cand_flat.ap(
                                pattern=[[c_w, q_full], [1, c_w]], offset=row_off
                            ),
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=cand_flat.ap(
                                pattern=[[1, rem]], offset=row_off + q_full * c_w
                            ),
                        )
            else:
                for gi in range(g):
                    for q in range(q_full):
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q, 1), nl.ds(0, c_w)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q * c_w, c_w)],
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q_full * c_w, rem)],
                        )

            got_flat = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
            got_tap = None
            if fused[_FUSE_TAP_GOT] is not None:
                got_tap = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for t0 in range(0, n_flat, _PSUM_MOVING):
                n_t = min(_PSUM_MOVING, n_flat - t0)
                tile = nl.ds(t0, n_t)
                # Broadcast the tile's keys down K_g partitions and compare against the ramp:
                # the block-diagonal one-hot, one column of which selects one (row, q) chunk.
                key_b = nl.ndarray(
                    (PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(
                    dst=key_b[0:k_g, 0:n_t],
                    stationary=ones_row_bf[0:1, 0:k_g],
                    moving=key_bf[0:1, tile],
                )
                b_hot = nl.ndarray(
                    (PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=b_hot[0:k_g, 0:n_t],
                    data=key_b[0:k_g, 0:n_t],
                    op0=nl.equal,
                    operand0=iota_pf[0:k_g, 0:1],
                )
                # M[c, col] = i_f[row_col, q_col*c_w + c] for every c at once.
                m_ps = nl.ndarray(
                    (PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(
                    dst=m_ps[0:c_w, 0:n_t],
                    stationary=d_all[0:k_g, 0:c_w],
                    moving=b_hot[0:k_g, 0:n_t],
                )
                # The c-selection: broadcast c down c_w partitions, one-hot it against the same
                # ramp, zero everything but row c_col, and sum the partitions with a ones matmul.
                c_b = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=c_b[0:c_w, 0:n_t],
                    stationary=ones_row_bf[0:1, 0:c_w],
                    moving=c_bf[0:1, tile],
                )
                m_sel = nl.ndarray(
                    (PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=m_sel[0:c_w, 0:n_t],
                    data=c_b[0:c_w, 0:n_t],
                    op0=nl.equal,
                    operand0=iota_pf[0:c_w, 0:1],
                )
                nisa.tensor_tensor(
                    dst=m_sel[0:c_w, 0:n_t],
                    data1=m_ps[0:c_w, 0:n_t],
                    data2=m_sel[0:c_w, 0:n_t],
                    op=nl.multiply,
                )
                sel = nl.ndarray((1, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=sel[0:1, 0:n_t],
                    stationary=ones_col[0:c_w, 0:1],
                    moving=m_sel[0:c_w, 0:n_t],
                )
                nisa.tensor_copy(dst=got_flat[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t])
                if got_tap is not None:
                    nisa.tensor_copy(
                        dst=got_tap[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t]
                    )

            # The destination rows are contiguous in HBM, so the whole group's ids leave in
            # ONE flat DMA rather than one per row.
            out_idx_flat = fused[_FUSE_OUT_IDX].reshape(
                (fused[_FUSE_OUT_IDX].shape[0] * k,)
            )
            nisa.dma_copy(
                dst=out_idx_flat.ap(
                    pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k
                ),
                src=got_flat[0:1, :],
            )
            if got_tap is not None:
                tap_flat = fused[_FUSE_TAP_GOT].reshape(
                    (fused[_FUSE_TAP_GOT].shape[0] * k,)
                )
                nisa.dma_copy(
                    dst=tap_flat.ap(
                        pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k
                    ),
                    src=got_tap[0:1, :],
                )

    return topk_values, topk_indices, asc_idx_hbm


@nki.jit
def chunked_topk(
    inp: nl.NkiTensor, config: ChunkedTopkConfig
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's ceiling.

    Dimensions:
        BxS:   number of logical rows (tokens)
        vocab: full reduction length; may exceed 65,535
        k:     number of largest elements, 1 <= k <= vocab/P

    Args:
        inp: ``[BxS, vocab]`` bfloat16 tensor in HBM -- or, when
            ``config.interleaved_batch_groups > 1`` (jit-in-jit callers only), a LIST of
            batch-group slab tensors in the slabbed receive layout, one per producing
            exchange, so stage-1 tiles data-depend per slab.
        config: ChunkedTopkConfig. ``config.chunks == 1`` delegates to gpsimd_topk.

    Returns:
        ``(topk_values [BxS, k] bf16, topk_indices [BxS, k] uint32)``, both in HBM.
        Indices are GLOBAL vocab indices in ``[0, vocab)``. Descending along the last dim
        when ``config.sorted``. With ``config.debug_taps`` the taps of ``TAP_NAMES`` are
        APPENDED, so these two keep their positions.

    Pseudocode:
        W = vocab // P
        s1_v, s1_i = topk(inp.reshape(BxS*P, W), k)              # free reshape: chunking
        cand_v = s1_v.reshape(BxS, P*k)                          # free reshape: concat
        cand_i = s1_i.reshape(BxS, P*k) + c*W  per column block  # P-1 scalar adds
        r_v, r_pos = topk(cand_v, k)                             # local reduce, no collective
        out_i = nc_n_gather(cand_i, r_pos)                       # positions -> global ids
        # both topk calls are INLINED here, and the reduce's r_pos never leaves SBUF
    """
    _validate_chunked_topk(config)

    bxs = config.BxS
    vocab = config.vocab_size
    k = config.k
    p = config.chunks

    # LNC=1 is a correctness requirement, so assert it rather than document it -- and assert it
    # BEFORE the p == 1 short-circuit, so no path can reach the kernel body at LNC=2.
    #
    # Passing num_programs=1 to the inner calls pins nothing: gpsimd_topk re-reads the count from the
    # live grid, so at LNC=2 stage 1 would shard its rows across programs while the reduce below is
    # NOT sharded. Every program would then read the full candidate array, including the rows the
    # other program produced, with no core_barrier between them, and both would write the same output
    # rows. fast_dma_safe would also silently flip off as per_lnc_BxS halves. A caller that wired up
    # two programs has a bug either way, and failing loudly at trace time beats returning
    # half-written output rows.
    _, n_prgs, _ = get_verified_program_sharding_info("chunked_topk", (0, 1), 2)
    kernel_assert(n_prgs == 1, f"chunked_topk is LNC=1 only, got n_prgs={n_prgs}")

    # A vocab that already fits one call needs none of the chunking machinery -- but it runs
    # through THIS kernel's inlined top-k, not gpsimd_topk: chunked_topk is hardware-sort only,
    # and delegating a sorted request to gpsimd_topk would silently re-enter the software sort
    # this kernel deleted. The unfused _inline_topk publishes private buffers, so its results
    # are staged into the kernel's own shared outputs.
    if p == 1:
        # Every tap point is a link of the CHUNKING chain, none of which exists here, so asking
        # for taps on this path is a caller mistake rather than something to silently return
        # fewer outputs for.
        kernel_assert(
            not config.debug_taps, "debug_taps has nothing to observe when chunks == 1"
        )
        kernel_assert(
            config.interleaved_peer_width == 0,
            "interleaved_peer_width is not implemented on the chunks == 1 delegate path",
        )
        p1_vals, p1_idx, _ = _inline_topk(
            inp,
            create_gpsimd_topk_config(
                inp_shape=(bxs, vocab),
                inp_dtype=config.inp_dtype,
                k=k,
                sorted=config.sorted,
                num_programs=1,
            ),
            skip_nan_fold=config.skip_nan_fold,
        )
        p1_out_v = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=nl.shared_hbm)
        p1_out_i = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=nl.shared_hbm)
        for t in range(div_ceil(bxs, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, bxs - r0)
            sv = nl.ndarray((PMAX, k), dtype=config.inp_dtype, buffer=nl.sbuf)
            si = nl.ndarray((PMAX, k), dtype=config.index_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sv[0:rows, :], src=p1_vals[r0 : r0 + rows, :])
            nisa.dma_copy(dst=si[0:rows, :], src=p1_idx[r0 : r0 + rows, :])
            nisa.dma_copy(dst=p1_out_v[nl.ds(r0, rows), :], src=sv[0:rows, :])
            nisa.dma_copy(dst=p1_out_i[nl.ds(r0, rows), :], src=si[0:rows, :])
        return p1_out_v, p1_out_i

    width = vocab // p
    s1_rows = bxs * p
    red_w = p * k
    # Stage 1's de-snake width, needed here only to shape the s1_late tap.
    s1_k_pad = div_ceil(k, PARTS_PER_GROUP) * PARTS_PER_GROUP

    # ---- Diagnostic taps (config.debug_taps) ------------------------------------------
    # Seven extra outputs that make the index chain observable ON HARDWARE, in a fully
    # overlapped run. See DEBUG TAPS in the module docstring for the method and for why each
    # tap point was chosen to perturb the schedule as little as possible.
    tap_s1_pos = None
    tap_s1_gid = None
    tap_s1_late = None
    tap_cand_i = None
    tap_r_pos = None
    tap_gathered = None
    tap_s1_val = None
    if config.debug_taps:
        tap_s1_pos = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # float32 on both paths so the reader never has to know which one produced it: the
        # general path's uint32 ids cast exactly on the way out (every id is < 2^24).
        tap_s1_gid = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_s1_late = nl.ndarray(
            (s1_rows, s1_k_pad), dtype=nl.float32, buffer=nl.shared_hbm
        )
        tap_cand_i = nl.ndarray((bxs, red_w), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_r_pos = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_gathered = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # The VALUE side of stage 1, appended LAST so every earlier tap keeps its position.
        # It makes the intermediate candidate VALUES checkable on hardware -- per-chunk pairing
        # against the caller's own input row, and per-chunk top-k multiset equality -- closing
        # the one link the index chain does not cover.
        tap_s1_val = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Which stage-1 shape can skip its phase 2 -------------------------------------
    # See _stage1_desnake_only: when k % 16 == 0 its de-snake buffers ARE the candidate arrays,
    # so stage 1's entire phase 2 (reload, compact, remap, store) is a copy and is deleted. The
    # second condition is that every stage-1 tile is a whole 128-partition tile, which is what
    # keeps the value buffer bfloat16 -- the reduce's nisa.topk requires bfloat16 input, and a
    # shard with a partial tile falls back to a float32 de-snake buffer. Both hold for every
    # auto-picked P at the production shape; anything else takes the general path below, which
    # is why both remain covered by the suite rather than one becoming dead code.
    red_rows = max(GROUPS_PER_TILE, div_ceil(bxs, GROUPS_PER_TILE) * GROUPS_PER_TILE)
    bypass_s1_phase2 = (
        (k % PARTS_PER_GROUP == 0)
        and (s1_rows % GROUPS_PER_TILE == 0)
        and (width % PARTS_PER_GROUP == 0)
    )

    # ---- Stage 1: one top-k per vocab chunk -------------------------------------------
    # The reshape is a pure view: a C-contiguous [BxS, vocab] tensor viewed as
    # [BxS*P, vocab/P] has row j*P + c == chunk c of token j. So the blocked contiguous load IS
    # the flat vocab sweep, and the instruction's 16-partition groups ARE the chunks.
    #
    # sorted=False is UNCONDITIONAL, not a knob. Stage 1's ordering is unobservable: its
    # values and indices never leave this kernel, and the reduce below re-sorts. So the
    # choice cannot affect the output, only the cost -- and sorting here is pure waste.
    # See the STAGE 1 IS NEVER SORTED section of the module docstring for the measurement.
    s1_cfg = create_gpsimd_topk_config(
        inp_shape=(s1_rows, width),
        inp_dtype=config.inp_dtype,
        k=k,
        sorted=False,
        num_programs=1,
    )
    kernel_assert(
        config.interleaved_peer_width == 0 or bypass_s1_phase2,
        "interleaved_peer_width is only implemented on the desnake-only stage-1 path",
    )
    _slabbed = isinstance(inp, (list, tuple))
    kernel_assert(
        not _slabbed or (config.interleaved_batch_groups > 1 and bypass_s1_phase2),
        "a slab-list input requires interleaved_batch_groups > 1 (and the desnake-only path)",
    )
    if bypass_s1_phase2:
        cand_v, cand_i = _stage1_desnake_only(
            inp if _slabbed else inp.reshape((s1_rows, width)),
            s1_cfg,
            chunks=p,
            chunk_width=width,
            pad_rows=red_rows * p,
            tap_pos=tap_s1_pos,
            skip_nan_fold=config.skip_nan_fold,
            peer_width=config.interleaved_peer_width,
            batch_groups=config.interleaved_batch_groups,
        )
        # The de-snake buffers, viewed as the reduce wants them. Both views are free, and they
        # agree slot for slot because both were written by the same de-snake pattern.
        red_in = cand_v.reshape((red_rows, red_w))
        cand_i_view = cand_i.reshape((red_rows, red_w))
        # On this path stage 1 publishes ONE index buffer, already global, so it is what both
        # the "what stage 1 said" tap and the second-read tap look at -- which makes the s1_late
        # comparison a three-read agreement test on the de-snake store rather than two.
        s1_published = cand_i
        s1_desnaked = cand_i
        # First s1_rows rows of the [pad_rows, k] buffer are the real candidates; the rest pad.
        s1_vals_pub = cand_v
    else:
        s1_vals, s1_idx, s1_desnaked = _inline_topk(
            inp.reshape((s1_rows, width)),
            s1_cfg,
            tap_pos=tap_s1_pos,
            skip_nan_fold=config.skip_nan_fold,
        )
        s1_published = s1_idx
        s1_vals_pub = s1_vals
        # ---- Reduce input: another free reshape, row-padded to a whole tile ---------------
        # [BxS*P, k] viewed as [BxS, P*k] places chunk c at columns [c*k, (c+1)*k) of row j.
        cand_v = s1_vals.reshape((bxs, red_w))
        cand_i_view = s1_idx.reshape((bxs, red_w))
        # Pad the row count up to a multiple of GROUPS_PER_TILE so the reduce stays on its
        # fast_dma_safe path (bf16 rather than fp32 de-snake buffers). This is a PERFORMANCE
        # pad, not a correctness one: a partial tile is handled correctly. Pad rows carry the
        # most-negative FINITE bf16, so they can never win a slot within their own row, and
        # their row's results are simply never read.
        if red_rows > bxs:
            red_in = nl.ndarray(
                (red_rows, red_w), dtype=config.inp_dtype, buffer=nl.private_hbm
            )
            n_pad = red_rows - bxs
            pad_tile = nl.ndarray((PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf)
            # Only the n_pad rows that are actually stored need the sentinel. Memsetting the
            # whole [PMAX, red_w] allocation wrote 128 partitions x 2,048 bf16 to fill 7 rows.
            nisa.memset(dst=pad_tile[0:n_pad, :], value=BFLOAT16_MIN)
            nisa.dma_copy(dst=red_in[nl.ds(bxs, n_pad), :], src=pad_tile[0:n_pad, :])
            for t in range(div_ceil(bxs, PMAX)):
                r0 = t * PMAX
                rows = min(PMAX, bxs - r0)
                stage = nl.ndarray(
                    (PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(dst=stage[0:rows, :], src=cand_v[r0 : r0 + rows, :])
                nisa.dma_copy(dst=red_in[r0 : r0 + rows, :], src=stage[0:rows, :])
        else:
            red_in = cand_v

    # ---- Reduce: top-k over the P*k candidates, entirely local ------------------------
    # The reduce's tail is FUSED into it (see WHY THE REDUCE FUSES ITS TAIL in _inline_topk):
    # its positions never reach HBM, they are translated through the candidate ids and written
    # as global ids from the same SBUF tile the sort produced them in, and its values go
    # straight to out_vals in one cast-on-DMA. Unfused this cost two stores, three loads and
    # two casts, all of them pure interface between two things that are now one body.
    out_vals = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=nl.shared_hbm)
    out_idx = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=nl.shared_hbm)
    _inline_topk(
        red_in,
        create_gpsimd_topk_config(
            inp_shape=(red_rows, red_w),
            inp_dtype=config.inp_dtype,
            k=k,
            sorted=config.sorted,
            num_programs=1,
        ),
        # The reduce input is stage 1's own (already NaN-folded) candidates plus finite
        # sentinel pad rows: the fold can never fire here. See the fold site for the argument.
        skip_nan_fold=True,
        fused=(
            cand_i_view,
            p,
            width,
            out_vals,
            out_idx,
            bxs,
            tap_cand_i,
            tap_r_pos,
            tap_gathered,
            bypass_s1_phase2,
        ),
    )

    if config.debug_taps:
        # A SECOND, independent read of stage 1's de-snake buffer, and a copy of stage 1's
        # remapped ids. Traced last so it tends to be scheduled late, but the diagnostic does
        # NOT depend on that: the signature of a store/reload race is TWO READS OF THE SAME
        # ADDRESS DISAGREEING, which is only possible while a write is in flight. So if s1_pos
        # (what phase 2 worked from) is wrong while s1_late is right, the de-snake store landed
        # after phase 2 read it; if BOTH are wrong the top-k itself reported those positions and
        # the fault is upstream of any DMA ordering.
        for t in range(div_ceil(s1_rows, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, s1_rows - r0)
            lt = nl.ndarray((PMAX, s1_k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lt[0:rows, :], src=s1_desnaked[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_late[r0 : r0 + rows, :], src=lt[0:rows, :])
            gt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt[0:rows, :], src=s1_published[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_gid[r0 : r0 + rows, :], src=gt[0:rows, :])
            vt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt[0:rows, :], src=s1_vals_pub[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_val[r0 : r0 + rows, :], src=vt[0:rows, :])
        return (
            out_vals,
            out_idx,
            tap_s1_pos,
            tap_s1_gid,
            tap_s1_late,
            tap_cand_i,
            tap_r_pos,
            tap_gathered,
            tap_s1_val,
        )

    return out_vals, out_idx


# ======================================================================================
# tail_chunked_megakernel -- vendored from nkilib experimental/gpt_oss_tail
# ======================================================================================


"""GPT-OSS decode sampling tail: RMSNorm -> lm_head -> top-k -> sampler, ONE collective.

    final RMSNorm -> lm_head GEMM -> all_to_all(all ranks) -> chunked_topk -> sampler

WHY ONE COLLECTIVE
------------------
The vocab is column-parallel across ``VTP`` ranks, so no rank can take a global top-k
alone. The established answer is to top-k each rank's shard and reduce across ranks with a
second collective, which costs two collectives, a group-aware index correction, a packed
value/index collective, and a batch-row permutation the caller has to undo.

This kernel instead spends its ONE collective bringing the FULL vocab onto each rank -- an
all-to-all that trades vocab-parallelism for batch-parallelism -- and then takes the whole
top-k locally with ``chunked_topk``, which exists precisely because a 201,088-wide vocab
exceeds ``nisa.topk``'s per-call ceiling. Everything the two-collective form needs in
between simply has no counterpart here:

* **No index correction.** After the all-to-all over the replica group ``[0 .. VTP-1]`` in
  RANK ORDER, peer ``p``'s shard lands at columns ``[p*V_s, (p+1)*V_s)``, and peer index IS
  vocab-shard index. So a column index in the assembled row already IS the global vocab id,
  and ``chunked_topk`` returns global ids with no fix-up of any kind.
* **No packed collective.** Nothing has to travel with its index, so there is no
  ``[rows, 2k]``-vs-``[2*rows, k]`` packing orientation to get wrong and no fp32 inflation
  of bf16 values.
* **Row ownership is a closed form, not a lookup.** With a single hop it is the identity --
  rank ``r`` owns ``[r*R, (r+1)*R)``, ``R = gbs/VTP`` -- so the caller's final ``all_gather``
  over the same rank-ordered group concatenates straight into batch order. Multi-hop (which
  trn3 forces, see below) makes it a mixed-radix sum instead, still closed-form and provided
  by this module's ``owned_row_block``, but the caller must then de-permute. The
  batch-grouped exchange (``a2a_batch_groups > 1``) adds one more mixed-radix digit and
  makes ownership STRIDED -- ``owned_rows`` is the map in force in every case.

WHICH COLLECTIVE ACTUALLY RUNS (why ``hop_meshes`` exists)
---------------------------------------------------------
"One collective" is the DATAFLOW claim -- one vocab-parallel-to-batch-parallel exchange rather
than a top-k-then-reduce pair. It is not a claim that the exchange is one hardware mesh
operation, because on trn3 it cannot be. Measured on a reserved 16-device trn3 host (16
devices x 8 cores = the 128 LNC=1 ranks of production):

    a2a group                        | device layout        | result
    ---------------------------------+----------------------+---------------------------------
    8 contiguous   {0..7}            | 1 device, 8 cores    | PASS
    4 stride 8     {0,8,16,24}       | 4 devices, 1 core ea | PASS
    16 stride 8    {0,8,...,120}     | 16 devices, 1 core ea| PASS
    16 contiguous  {0..15}           | 2 devices, 8 cores ea| FAIL: op_type 7 unsupported
    128 all ranks                    | 16 devices, 8 cores  | FAIL: op_type 7 unsupported
    4 stride 4     {0,4,8,12}        | 2 devices, 2 cores ea| FAIL: encd_mesh_memcopy

The constraint is the group's DEVICE LAYOUT, not its size. A group must be entirely within one
device, or hold exactly one core per device; a MIXED group -- several devices with several cores
each -- has no mesh subtype that implements all-to-all and is rejected with
``__select_mesh_subtype: not a supported operation for mesh (op_type 7)``. Payload size is not
the trigger: the 16-contiguous rejection carries ``total_sz 0x800`` and the 128-rank one
``0x62300``.

Reading it as a size limit is the easy mistake, and it costs a hop: it makes 8 look like the
ceiling, which forces three hops, when in fact ``(8, 16)`` is legal and sufficient -- hop 0
device-local, hop 1 one core per device across all 16. That is exactly the decomposition the
existing sharded tail uses.

``all_gather`` at 128 also runs, but it is not the right tool here: it hands every rank EVERY
batch row, which at gbs=1024 is 412 MB per rank against 6.4 MB for the two-hop all-to-all, and
extracting just this rank's rows needs a rank-dependent slice one shared NEFF cannot express.

Note these limits differ by host. On the shared fleet's trn3_a0 a single mesh passes at 16 and
32 but aborts at >= 64 in ``encd_mesh_memcopy``, while MORE THAN ONE mesh per NEFF fails to
load at any rank count -- the exact opposite of what reserved trn3 allows. So a decomposition
must be validated on the host it will run on; ``collective_probe.py`` holds both matrices.

STATUS
------
Correct and passing every gate at VTP = 8, 16 and 32 (values, indices, tokens, logits, with
per-row-varying sampling params and degenerate/extreme inputs), single-hop and multi-hop.

CALLER CONTRACTS
----------------
Three, all of them load-bearing:

1. ``lm_head_weight_t`` is ``[H, V_s]`` **bf16**, i.e. ALREADY TRANSPOSED, and C-contiguous.
   bf16 is the GPT-OSS checkpoint's own lm_head precision, so passing fp32 would not be more
   accurate than the model -- it would just double the dominant HBM read. The kernel DMAs it
   straight to bf16 SBUF and matmuls in bf16 with an fp32 PSUM accumulator.
   ``nc_matmul`` contracts the partition dim, so the weight must present H on partitions.
   Reading an untransposed ``[V_s, H]`` shard forces ``V_s`` separate short runs per
   H-chunk instead of one contiguous run; a transposed VIEW (no copy) keeps the original
   strides and the DMA descriptor is built from strides, so it provides none of the benefit.
   Transposing is a one-time host cost on a tensor that never changes.
2. ``replica_groups`` is a single group listing ALL ``VTP`` ranks IN RANK ORDER. The
   no-index-correction property above is a direct consequence of that ordering; a permuted
   group silently returns wrong vocab ids.
3. ``sampling_params`` is ``[R, 3]`` or ``[R, 4]`` -- THIS RANK'S rows only, not the whole
   batch. Column 3 is the per-row draw when present; without it the sampler uses its own
   compiled constant. The
   caller slices by ``owned_rows`` (``[r*R, (r+1)*R)`` for a single ungrouped hop; strided
   with ``a2a_batch_groups > 1``; a mixed-radix block multi-hop). Passing the full ``[gbs, 3]``
   would make every rank apply rows ``0..R-1``'s temperature and top-p to its own rows, a
   wrong-token bug invisible to any test whose sampling params happen to be uniform.

THE mxfp4 VARIANCE CONVENTION (do not "simplify")
-------------------------------------------------
GPT-OSS pads hidden 2880 -> 3072, and under the mxfp4 checkpoint the hidden dim is SHUFFLED,
so the padding lanes are INTERLEAVED rather than a contiguous tail. Two consequences:
variance is ``sum(x^2)/H_u`` (equal to ``mean(x^2 over all H) * H/H_u`` only because the
padding is exactly zero), and zeroing the padding (applied to GAMMA at load time -- see
``_load_gamma_broadcast`` for why that is equivalent to re-zeroing the normed output) goes
through a ``[4, H/4]`` view killing columns ``>= H_u/4`` of each group. A contiguous
``[..., H_u:] = 0`` zeroes the WRONG lanes; that is the bf16 checkpoint's convention and it
changes the answer here.

LNC=1 BY DESIGN
---------------
``VTP = 128`` LNC=1 ranks is what makes ``V_s = 201088/128 = 1571``. ``n_prgs == 1`` is
asserted rather than assumed: the stages after the collective are not program-sharded and
carry no ``core_barrier``, so at LNC=2 both programs would race on the same ``shared_hbm``
collective buffers. Note that passing ``num_programs=1`` to an inner kernel does NOT pin
this -- gpsimd_topk re-reads the count from the live grid -- which is why the assert is here.
"""


_PMAX = nl.tile_size.pmax  # 128


def compose_hops(vtp: int, hops: Tuple[int, ...]):
    """Replica-group meshes for a sequence of all-to-all hops whose sizes multiply to ``vtp``.

    Hop ``h`` groups ranks that agree on every LATER hop's coordinate, so with ``hops=(8, 16)``
    and ``r = q*8 + s``: hop 0 groups the 8 CONTIGUOUS ranks sharing ``q``, and hop 1 groups
    the 16 ranks sharing ``s``, strided by 8. Listing each group in ascending rank order is
    what keeps a column index equal to a global vocab id.

    Returns a tuple of meshes, one per hop, each a tuple of rank-tuples.
    """
    prod = 1
    for h in hops:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hops} multiply to {prod}, not vtp={vtp}")
    stride = 1
    meshes = []
    for h in hops:
        groups = []
        for base in range(vtp):
            # One group per distinct (everything except this hop's coordinate).
            if (base // stride) % h != 0:
                continue
            groups.append(tuple(base + i * stride for i in range(h)))
        meshes.append(tuple(sorted(groups)))
        stride *= h
    return tuple(meshes)


def owned_row_block(rank: int, vtp: int, gbs: int, hops=None):
    """Batch rows that ``rank`` owns after the all-to-all. Returns ``(row_start, n_rows)``.

    THE caller-facing half of the kernel's row-ownership contract, living next to
    ``compose_hops`` because the two must agree: a production caller de-permutes this
    kernel's per-rank outputs with exactly this map. With a single hop it is the IDENTITY
    -- rank ``r`` owns ``[r*gbs/vtp, (r+1)*gbs/vtp)`` -- which is the point of routing the
    collective over a rank-ordered group. A multi-hop decomposition keeps, per hop, only
    the sub-block selected by this rank's coordinate within that hop's group, so the
    offsets compose as a mixed-radix sum. Kept as one named function so the tests can
    assert the map tiles the batch exactly once and a change to hop structure has exactly
    one place to break.

    ``hops=None`` means a single hop over all ``vtp`` ranks, which the general formula
    reproduces exactly (``start = rank * gbs/vtp``); it is not a separate code path.
    """
    if gbs % vtp:
        raise ValueError(f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows = gbs // vtp
    hop_sizes = (vtp,) if hops is None else tuple(hops)
    prod = 1
    for h in hop_sizes:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hop_sizes} multiply to {prod}, not vtp={vtp}")
    span, stride, start = gbs, 1, 0
    for h in hop_sizes:
        span //= h
        start += ((rank // stride) % h) * span
        stride *= h
    return start, rows


def owned_rows(rank: int, vtp: int, gbs: int, hops=None, batch_groups: int = 1):
    """The GLOBAL batch rows ``rank`` owns, IN RECEIVE ORDER, as a list.

    Generalizes ``owned_row_block`` to the batch-group-chunked all-to-all
    (``a2a_batch_groups > 1``), where ownership is no longer one contiguous block: chunk
    ``g``'s collective carries mirror rows ``[g*S, (g+1)*S)`` (``S = gbs/G``, one or more
    whole lm_head tiles -- which is what lets the exchange start under the lm_head), and a
    full-mesh all-to-all of that slab hands rank ``p`` its ``S/vtp`` rows of EVERY group:

        row(g, j) = g*S + p*(S/vtp) + j,   g in [0, G),  j in [0, S/vtp)

    i.e. the same mixed-radix ownership map with the group index as one more (outermost)
    digit. ``batch_groups = 1`` reproduces ``owned_row_block`` exactly. ``batch_groups > 1``
    is defined for the single-hop exchange only, matching the kernel.
    """
    # Derived FROM owned_row_block rather than re-implemented, so its validation (hop sizes
    # multiply to vtp, gbs divisible) covers the grouped form too, and "G = 1 reproduces
    # owned_row_block" holds by construction. With G groups each group is gbs/G rows and
    # this rank's contiguous block splits into G equal sub-blocks, one per group, at the
    # same relative offset (start/G) inside each group.
    start, rows = owned_row_block(rank, vtp, gbs, hops)
    if batch_groups == 1:
        return list(range(start, start + rows))
    if hops is not None and len(tuple(hops)) != 1:
        raise ValueError("batch_groups > 1 is defined for the single-hop exchange only")
    if rows % batch_groups or start % batch_groups:
        raise ValueError(
            f"rows_per_rank ({rows}) must be divisible by batch_groups ({batch_groups})"
        )
    group_rows = gbs // batch_groups
    return [
        g * group_rows + start // batch_groups + j
        for g in range(batch_groups)
        for j in range(rows // batch_groups)
    ]


# The mxfp4 hidden shuffle is defined over 4 interleaved groups, so the padding boundary is
# expressed in a [4, H/4] view. See THE mxfp4 VARIANCE CONVENTION above.
_MXFP4_SHUFFLE_GROUPS = 4

# Sampler constants. The below-threshold sentinel is FINITE by design: -inf would make a
# fully-masked row's manual softmax produce NaN.
_TOPK_MASK_SENTINEL = -3000.0
# temperature < eps => treat the row as greedy.
_SAMPLING_EPS = 1e-5
# The inverse-transform draw when ``sampling_params`` carries no 4th column: COMPILED IN, so
# the 3-column NEFF stays deterministic by construction. A 4-column ``sampling_params``
# supplies the draw per row at RUNTIME instead (see _sampler_stage), which is what lets a
# test sweep u as data without rebuilding the NEFF.
_DETERMINISTIC_DRAW = 0.5


def _const_tile(shape, value: float):
    """Materialize a constant SBUF tile.

    ``nisa.tensor_copy_predicated`` validates its ``src`` as a partitioned tensor and
    rejects a Python scalar despite the type hint.
    """
    tile = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tile, value=value)
    return tile


def _make_scan_triangle():
    """``[128, 128]`` fp32 upper-triangular ones, ``T[p, n] = 1 iff p <= n``.

    The stationary-side constant of the Tensor Engine prefix sum in ``_inclusive_scan_pe``:
    contracting data against it turns "sum over partitions" into "sum over partitions up to
    n", which IS the inclusive prefix. Built from two iotas and one compare, with no data
    dependencies, so the scheduler hoists it off every critical path.
    """
    part = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(part, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    cols = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(cols, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0)
    tri = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tri, data=cols, op0=nl.greater_equal, operand0=part)
    return tri


def _inclusive_scan_pe(dst, src, n_rows: int, width: int, tri) -> None:
    """Inclusive prefix sum along the free dim on the TENSOR Engine: ``[rows, w] -> [rows, w]``.

    Per 128-wide chunk: transpose the chunk so its slice of the row sits on partitions, then
    contract against the upper-triangular ones -- ``out[r, n] = sum_{p <= n} chunk_t[p, r]`` --
    and read the PSUM back with the previous chunks' running total fused into the copy as a
    per-partition scalar add (the last column of the previous chunk's output, which already
    carries every earlier chunk).

    This replaces a Hillis-Steele scan whose ceil(log2(w)) shifted adds were the sampler's
    single largest block of Vector work, and whose every step was serial ON VECTOR. Here
    Vector pays two 128-wide copies per chunk and the contraction runs on the PE, which is
    otherwise idle through the whole sampler phase. Each output element is ONE fp32 PE
    contraction of its prefix rather than a tree of partial adds -- a different (not worse)
    fp32 summation order than either the tree or a sequential cumsum, the same class of
    last-ulp wiggle the gates' tie-guards already absorb.
    """
    for c in range(div_ceil(width, _PMAX)):
        c0 = c * _PMAX
        c_len = min(_PMAX, width - c0)
        chunk_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=chunk_ps[0:c_len, 0:n_rows],
            data=src[0:n_rows, nl.ds(c0, c_len)],
            engine=nisa.tensor_engine,
        )
        chunk_t = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=chunk_t[0:c_len, 0:n_rows], src=chunk_ps[0:c_len, 0:n_rows]
        )
        pre_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=pre_ps[0:n_rows, 0:c_len],
            stationary=chunk_t[0:c_len, 0:n_rows],
            moving=tri[0:c_len, 0:c_len],
        )
        if c == 0:
            nisa.tensor_copy(
                dst=dst[0:n_rows, nl.ds(c0, c_len)], src=pre_ps[0:n_rows, 0:c_len]
            )
        else:
            # The carry rides the PSUM read: previous chunk's last output column is the total
            # of everything before this chunk, and it lives on the same partition as its row.
            nisa.tensor_scalar(
                dst=dst[0:n_rows, nl.ds(c0, c_len)],
                data=pre_ps[0:n_rows, 0:c_len],
                op0=nl.add,
                operand0=dst[0:n_rows, nl.ds(c0 - 1, 1)],
            )


def _skew_fence(hop_meshes, hidden):
    """All-reduce a ZERO on every hop mesh, and return it for the norm to consume.

    PROFILING INSTRUMENT, not computation. Without it the FIRST collective absorbs all the
    rank-arrival skew and every LATER collective absorbs whatever skew re-accumulates while the
    ranks run compute -- so a per-hop duration reads as transfer cost when it is mostly dispatch
    jitter. Fencing each mesh separately moves that charge onto the fence.

    It cannot change the answer: the reduced value is 0.0 and the caller adds it into the
    hidden states, and ``x + 0.0`` is bit-identical for every finite fp32 x. The result MUST
    stay consumed by the norm -- that data dependency is the only thing stopping the scheduler
    from sinking the fence past the real work and making it useless.
    """
    acc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    zero_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    nisa.dma_copy(dst=acc, src=zero_sb)
    for hop in range(len(hop_meshes)):
        src = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=zero_sb)
        dst = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        ncc.all_reduce(
            srcs=[src],
            dsts=[dst],
            replica_group=ncc.ReplicaGroup(hop_meshes[hop]),
            op=nl.add,
        )
        got = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=got, src=dst)
        prev = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=prev, src=acc)
        nisa.tensor_tensor(dst=prev, data1=prev, data2=got, op=nl.add)
        nisa.dma_copy(dst=acc, src=prev)
    return acc


def _load_gamma_broadcast(gamma_hbm, hidden, hidden_actual, zero_interleaved_pad):
    """gamma ``[1, H]`` -> ``[_PMAX, H]`` SBUF broadcast, with the padding lanes zeroed.

    Loads ONLY the kept columns and zeroes the pad lanes with memsets on DISJOINT columns, so the
    memsets carry no WAW on the DMA and never wait for it.

    gamma is ``[1, H]`` and every row needs it, but a PARTITION-dim broadcast of an on-chip
    tensor is illegal -- do it at the DMA boundary with an HBM-side ``.broadcast`` view. Loaded
    ONCE per kernel, not once per 128-row tile: gamma never changes between tiles, and the
    broadcast moves ``_PMAX * H`` fp32 (1.5 MB at the production shape) per load.

    ZEROING GAMMA'S PADDING REPLACES RE-ZEROING THE NORMED OUTPUT. The input's padding lanes
    are exactly zero by contract -- the variance formula divides by ``H_u`` on exactly that
    assumption (see THE mxfp4 VARIANCE CONVENTION) -- so the normed padding is
    ``(0 * rms) * gamma_pad``: zero for any finite ``gamma_pad``, and NaN for a non-finite one
    only when the variance (and therefore every normed lane) is already poisoned anyway. Doing
    it on gamma moves the memsets off the per-tile critical path, where the old form re-ran
    them per tile AFTER the full-width gamma multiply, and lets the whole normed row feed the
    GEMM the moment it is scaled. The mxfp4 padding is INTERLEAVED, so kill columns
    ``>= H_u/4`` of each of 4 groups.
    """
    gamma_sb = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    if hidden_actual < hidden:
        _loads = []
        if zero_interleaved_pad:
            gw = hidden // _MXFP4_SHUFFLE_GROUPS
            keep = hidden_actual // _MXFP4_SHUFFLE_GROUPS
            for g in range(_MXFP4_SHUFFLE_GROUPS):
                _loads.append(
                    nisa.dma_copy(
                        dst=gamma_sb[:, nl.ds(g * gw, keep)],
                        src=gamma_hbm[0:1, nl.ds(g * gw, keep)].broadcast(0, _PMAX),
                    )
                )
            for g in range(_MXFP4_SHUFFLE_GROUPS):
                if gw - keep > 0:
                    _ms = nisa.memset(
                        dst=gamma_sb[:, nl.ds(g * gw + keep, gw - keep)], value=0
                    )
                    if hasattr(_ms, "depends_on"):
                        _ms.depends_on(*_loads, kind=nisa.dependency.schedule)
        else:
            _loads.append(
                nisa.dma_copy(
                    dst=gamma_sb[:, 0:hidden_actual],
                    src=gamma_hbm[0:1, 0:hidden_actual].broadcast(0, _PMAX),
                )
            )
            _ms = nisa.memset(
                dst=gamma_sb[:, nl.ds(hidden_actual, hidden - hidden_actual)], value=0
            )
            if hasattr(_ms, "depends_on"):
                _ms.depends_on(*_loads, kind=nisa.dependency.schedule)
        return gamma_sb
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm[0:1, :].broadcast(0, _PMAX))
    return gamma_sb


# DMA granule for the row tile's hidden dim in _norm_lm_head_stage. One monolithic [rows, H]
# load serialises the whole norm behind the full transfer (12 us of engine idle at the
# production shape); block loads let the sum of squares run behind each block while the next
# is still in flight, so "rms ready" trails the LAST block's arrival by one 0.7 us activation
# instead of trailing the whole row by three full-width ops.
_NORM_DMA_BLOCK = 6 * _PMAX


def _load_weight_resident(wt_hbm, hidden, vocab_shard, after=None):
    """Load the whole ``[H, V_s]`` bf16 lm_head shard into SBUF ONCE, as per-chunk tiles.

    ``after``: a COMPUTE instruction handle every chunk load takes a RUNTIME
    dependency on (the first consumer of the first row tile's input block 0), so the 9.65 MB stream
    starts once the hidden rows have landed instead of wherever SBUF frees up -- in the fused decode
    graph that was under the last decode layer's EP ReduceScatter. The chunks are needed by the GEMM
    one per ~1 us from its start; at the measured ~0.9 us per 402 KB chunk they keep ahead of it.

    The GEMM was re-reading the shard from HBM once per pair of 128-row tiles (measured:
    2x/4x the 9.65 MB shard at gbs 512/1024). In isolation that costs no wall clock -- the
    head is PE-bound and the stream keeps pace -- but in INTEGRATION the repeated stream is
    contended HBM bandwidth that cannot be prefetched behind preceding work, whereas one
    resident copy (24 x 393 KB = 9.4 MB of SBUF, ~40%, live only through the head) loads
    once, entirely prefetchable before the first matmul needs chunk 0. Static descriptors
    for the same reason as the input blocks above.
    """
    wt_res = []
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        wt_c = nl.ndarray((_PMAX, vocab_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        _ld = nisa.dma_copy(
            dst=wt_c[0:h_len, :],
            src=wt_hbm[nl.ds(h0, h_len), :],
            dge_mode=nisa.dge_mode.none,
        )
        if after is not None:
            _ld.depends_on(after, kind=nisa.dependency.runtime)
        wt_res.append(wt_c)
    return wt_res


def _norm_lm_head_stage(
    x_hbm,
    gamma_sb,
    wt_res,
    eps,
    n_rows,
    row_start,
    hidden,
    hidden_actual,
    vocab_shard,
    fence,
    deferred_rms,
    wt_hbm=None,
):
    """RMSNorm + lm_head GEMM for one 128-row tile, pipelined at block/chunk granularity.

    Returns ``(acc_ps, rms, wt_res)``. ``wt_res`` is the resident weight passed in, or -- when
    ``wt_res`` arrives as None -- the one this stage loads from ``wt_hbm``
    with a runtime dependency on the first compute consumer of its input block 0, for the caller
    to reuse on later tiles. With ``deferred_rms`` (the default), ``acc_ps`` is the
    ``[_PMAX, V_s]`` fp32 PSUM accumulator holding this tile's UNNORMALIZED logits
    ``(x * gamma) @ W`` and the CALLER multiplies the ``[_PMAX, 1]`` fp32 per-row ``rms`` in
    on its PSUM-read copies. DEFERRED RMS: since ``rms`` is a per-row scalar and the GEMM is
    linear, ``((x*rms)*gamma) @ W == rms * ((x*gamma) @ W)`` exactly -- and deferring it
    removes the one true data dependency that stalled the PE: every chunk used to wait on
    ``rms``, which needs ``sum(x^2)`` over the WHOLE row (~9 us of PE idle at the head of
    every tile; measured -3.8 us on the fenced head window). With ``deferred_rms=False`` the
    legacy in-chain form runs instead -- ``(x * rms) * gamma`` per chunk, ``acc_ps`` already
    normalized, the returned ``rms`` unused -- as a safety fallback with the exact
    pre-deferred numerics realization.

    THE POINT IS THE SCHEDULE'S SHAPE, NOT THE MATH. What the original rewrite fixed was three
    strictly serial phases -- a 12 us monolithic input DMA that every engine waited out, then
    ~15 us of full-width [rows, H] Vector/Scalar ops each gated on the previous, then 24
    transposes and 24 matmuls -- 54 us for a tile whose PE work is 16 us. This form breaks
    every one of those serialisations:

    * The input row loads in ``_NORM_DMA_BLOCK`` column blocks, and each block's
      ``sum(x^2)`` contribution runs on the Scalar Engine (``nisa.activation`` with
      ``op=nl.square`` accumulating into the engine's ``reduce_regs`` across blocks -- the
      idiom of ``mlp_tkg_rmsnorm``) while later blocks are still in flight. NOTE the
      ``reduce_regs`` chain is engine state: it tolerates interleaved ``reduce_cmd.idle``
      activations (every other activation in this kernel), but a hoisted reducing activation
      from other code WOULD corrupt it -- the WAW on ``sq_scratch`` is what pins the chain's
      own order.
    * ``rsqrt(ssq/H_u + eps)`` is ONE activation: the Scalar Engine applies
      ``op(data*scale + bias)`` natively, so the two full-width-free tensor_scalars the old
      form spent on ``/H_u`` and ``+eps`` cost nothing here.
    * Gamma applies in ONE ``scalar_tensor_tensor`` per 128-column GEMM chunk --
      ``(x + fence0) * gamma`` -- feeding that chunk's transpose+matmul immediately. RMS is
      NOT in this chain (see DEFERRED RMS above): the rsqrt runs concurrently on the Scalar
      Engine and lands on the caller's PSUM reads, so no matmul waits for the full row.

    The bf16 boundary moves by exactly one algebraic step and not in magnitude: the bf16
    rounding lands on ``x * gamma`` instead of ``x * rms * gamma``. bf16's relative error is
    scale-invariant (same exponent range as fp32), so the per-element error bound is
    IDENTICAL -- one bf16 rounding either way -- and the trailing fp32 multiply by rms adds
    2^-24, invisible at bf16 output granularity. Verified by simulation at the production
    shape against fp64 ground truth: both forms sit at the same distance from exact (mean
    1.66e-3 relative), greedy argmax flips ZERO rows, and the sampling distributions differ
    by <= 0.7% total variation. What DOES change is the last-ulp REALIZATION -- logits move
    within the bf16 noise floor relative to the previous build, so fixed-seed sampled tokens
    can land on an equally-valid neighbour at cdf crossings. Same class of wiggle as any
    resummation change, arbitrated the same way: tie-guarded gates, regenerated hashes.

    On the GEMM half nothing changed: ``nc_matmul`` contracts the PARTITION dim of both
    operands, so H must be on partitions for both. The activations get an on-chip per-chunk
    ``nc_transpose``; the WEIGHT is required to arrive already transposed as ``[H, V_s]`` --
    see CALLER CONTRACTS -- which makes the per-chunk read ``h_len`` ADJACENT rows of ``V_s``
    contiguous elements, i.e. ONE contiguous run rather than ``V_s`` short strided ones.
    BF16 THROUGHOUT, INCLUDING THE WEIGHT DMA: the GPT-OSS lm_head ships bf16, so fp32 here
    buys no accuracy over the checkpoint -- it just doubles the dominant HBM read (9.65 MB per
    rank per pass instead of 19.30 MB, re-read once per row tile).
    """
    # ---- load + sum of squares, block by block ----
    # The block DMAs are issued FIRST, before the fence broadcast: DMA descriptors on a queue
    # execute in issue order, so a fence-dependent load issued ahead of them would stall the
    # whole input behind the skew-absorbing all-reduce (measured: +14 us of engine idle in a
    # fenced profile). The fence rides TWO operand slots, both computing +0.0: the squares'
    # BIAS (gating ssq -> rms) and each GEMM chunk's scalar_tensor_tensor ADD (gating every
    # matmul and, through the mirror store, the collective). The second one exists because
    # DEFERRED RMS removed the rms edge that used to carry the fence into the matmuls -- an
    # unfenced GEMM would front-run the skew window and the fence-corrected head would
    # under-count it. Both adds are exact (+0.0) and cost no extra instructions.
    # A 16-bit input loads half the bytes of this kernel's largest input DMA, then widens to
    # the fp32 xt through an EXPLICIT per-block tensor_copy. The copy is what makes 16-bit
    # input bitwise-equivalent to a pre-widened fp32 copy of the same data: a dtype-conversion
    # COPY widens exactly (fp16/bf16 are subsets of fp32), whereas feeding 16-bit data
    # straight into the square/gamma chains does NOT -- the engines compute at the INPUT's
    # precision, measured as a real output diff on hardware (2026-08-31). Everything below
    # this loop reads only the fp32 xt, so the numerics are dtype-independent.
    widen = x_hbm.dtype != nl.float32
    x_raw = (
        nl.ndarray((_PMAX, hidden), dtype=x_hbm.dtype, buffer=nl.sbuf)
        if widen
        else None
    )
    xt = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    ssq = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq_scratch = nl.ndarray((_PMAX, _NORM_DMA_BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    n_blk = div_ceil(hidden, _NORM_DMA_BLOCK)
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        # Static descriptors (dge_mode.none): generated at compile time, so the load neither
        # waits on a DGE engine nor -- in fenced profiling NEFFs -- queues behind the skew
        # fence's trigger, which lives on the GpSimd DGE queue.
        nisa.dma_copy(
            dst=(x_raw if widen else xt)[0:n_rows, nl.ds(b0, b_len)],
            src=x_hbm[row_start : row_start + n_rows, nl.ds(b0, b_len)],
            dge_mode=nisa.dge_mode.none,
        )
    _anchor = None
    if widen:
        for b in range(n_blk):
            b0 = b * _NORM_DMA_BLOCK
            b_len = min(_NORM_DMA_BLOCK, hidden - b0)
            _cp = nisa.tensor_copy(
                dst=xt[0:n_rows, nl.ds(b0, b_len)],
                src=x_raw[0:n_rows, nl.ds(b0, b_len)],
            )
            if b == 0:
                _anchor = _cp
    f_sb = None
    if fence is not None:
        # Broadcast on the partition dim at the DMA boundary, as with gamma.
        f_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=f_sb, src=fence[0:1, 0:1].broadcast(0, _PMAX))
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        _act = nisa.activation(
            dst=sq_scratch[0:n_rows, 0:b_len],
            op=nl.square,
            data=xt[0:n_rows, nl.ds(b0, b_len)],
            bias=f_sb[0:n_rows, :] if f_sb is not None else None,
            reduce_op=nl.add,
            reduce_res=ssq[0:n_rows, :] if b == n_blk - 1 else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce
            if b == 0
            else nisa.reduce_cmd.reduce,
        )
        if b == 0 and _anchor is None:
            _anchor = _act
    if wt_res is None:
        kernel_assert(
            wt_hbm is not None and _anchor is not None,
            "wt_res is None and no wt_hbm / block-0 anchor to load it after",
        )
        wt_res = _load_weight_resident(wt_hbm, hidden, vocab_shard, after=_anchor)
    # Divide by H_u, NOT H: the mxfp4 convention (padding contributes exactly zero to ssq).
    # DEFERRED: nothing below consumes rms -- it runs concurrently with the GEMM on the
    # Scalar Engine and the CALLER multiplies it onto the PSUM-read copies.
    rms = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rms[0:n_rows, :],
        op=nl.rsqrt,
        data=ssq[0:n_rows, :],
        scale=1.0 / float(hidden_actual),
        bias=float(eps),
    )

    # ---- per-128-column chunk: scale -> transpose -> bf16 -> matmul ----
    # Deferred: (x + fence0) * gamma -- chunk c depends only on its OWN input block (plus the
    # fence), so the first transpose can start one block-DMA after the fence instead of after
    # the whole row's ssq. Legacy (deferred_rms=False): (x * rms) * gamma, the rms edge
    # carrying the fence into the matmuls exactly as before.
    acc_ps = nl.ndarray((_PMAX, vocab_shard), dtype=nl.float32, buffer=nl.psum)
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        scaled_c = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        if deferred_rms:
            nisa.scalar_tensor_tensor(
                dst=scaled_c[0:n_rows, 0:h_len],
                data=xt[0:n_rows, nl.ds(h0, h_len)],
                op0=nl.add,
                operand0=f_sb[0:n_rows, :] if f_sb is not None else 0.0,
                op1=nl.multiply,
                operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
            )
        else:
            nisa.scalar_tensor_tensor(
                dst=scaled_c[0:n_rows, 0:h_len],
                data=xt[0:n_rows, nl.ds(h0, h_len)],
                op0=nl.multiply,
                operand0=rms[0:n_rows, :],
                op1=nl.multiply,
                operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
            )
        xt_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=xt_ps[0:h_len, 0:n_rows],
            data=scaled_c[0:n_rows, 0:h_len],
            engine=nisa.tensor_engine,
        )
        xt_sb = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xt_sb[0:h_len, 0:n_rows], src=xt_ps[0:h_len, 0:n_rows])
        # The weight chunk is RESIDENT (loaded once for all row tiles by
        # _load_weight_resident), not re-streamed per tile.
        nisa.nc_matmul(
            dst=acc_ps[0:n_rows, :],
            stationary=xt_sb[0:h_len, 0:n_rows],
            moving=wt_res[c][0:h_len, :],
        )
    return acc_ps, rms, wt_res


def _select_by_position(
    data, ramp, position, n_rows: int, k: int, inf_safe: bool = False
):
    """Return ``data[row, position[row]]`` as a ``[P, 1]`` fp32 tile, without nc_n_gather.

    Selection is a masked sum: build ``ramp == position`` and reduce ``mask * data``. That is
    four vector ops on a ``[rows, k]`` tile, against one gather instruction -- but the gather
    is avoided deliberately. ``nc_n_gather`` returned ZERO here for a narrow ``data`` tile
    (k=8) even though the position and the data were both verified correct on device, and the
    same primitive is documented as corrupting results in its multi-group form above
    ``gather_group_size`` (see rotational_topk_utils). A masked sum has no such
    width-dependent behaviour, so this is correct for every k the kernel accepts.

    ``position`` is fp32 holding a small exact integer, and ``ramp`` holds exact integers, so
    the equality is exact rather than approximate. ``(ramp == position) * data`` is ONE fused
    ``scalar_tensor_tensor`` on the Vector Engine -- the compare's 1.0/0.0 feeds the multiply
    inside the instruction -- so the whole selection is two instructions, and every one of the
    sampler's instructions counts: at ``gbs = VTP`` the sampler owns a single batch row, every
    op runs on ONE partition in ~0.3 us of mostly dispatch, and this function sits twice on
    that serial chain.

    ``inf_safe``: the fused form NaN-poisons on ``+/-inf`` data -- ``0.0 * inf = NaN`` in
    every NON-selected slot, and the add-reduce then returns NaN instead of the selected
    value. The values this kernel samples may legally be ``+/-inf`` (GEMM overflow saturates,
    declared in-contract by ``skip_nan_fold``), so the top-k THRESHOLD selection must use the
    inf-safe form: ONE ``nisa.range_select`` (index-range mask + max-reduce, no multiply).
    It requires ``data`` to be DESCENDING along the free dim (the sorted top-k values are)
    and FLOORS ``position``: it returns the max over slots >= floor(position), which on a
    descending row is the slot floor(position) itself. The fused form needs an exact
    integer position, which its one caller (the sampled position, a count) always is.
    """
    out = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((_PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
    if inf_safe:
        # ONE Vector instruction: range_select compares each slot's index + range_start against
        # two per-partition bounds and max-reduces the selected elements. With range_start=1
        # the mask is (s + 1 > position) AND (s + 1 >= position), i.e. every slot s >= FLOOR
        # (position) -- so position may be a non-integer (top_k is fp32 runtime data and the
        # model floors it) and the reduce over a DESCENDING row returns exactly the slot
        # floor(position). No multiply anywhere, so +/-inf data is safe. Unselected slots read
        # FP32_MIN; the one observable edge is a row whose selected slot IS -inf: the
        # threshold comes back FP32_MIN rather than -inf, so -inf values below it get the
        # finite sentinel instead of staying -inf -- both exp to exactly 0 in the softmax.
        # Replaces six ops (iota, two compares, AND, memset, predicated copy, add-reduce).
        sel = nl.ndarray((_PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.range_select(
            dst=sel[0:n_rows, :],
            on_true_tile=data[0:n_rows, :],
            comp_op0=nl.greater,
            comp_op1=nl.greater_equal,
            bound0=position[0:n_rows, :],
            bound1=position[0:n_rows, :],
            reduce_op=nl.maximum,
            reduce_res=out[0:n_rows, :],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            range_start=1,
        )
        return out
    else:
        nisa.scalar_tensor_tensor(
            dst=prod[0:n_rows, :],
            data=ramp[0:n_rows, :],
            op0=nl.equal,
            operand0=position[0:n_rows, :],
            op1=nl.multiply,
            operand1=data[0:n_rows, :],
        )
    nisa.tensor_reduce(dst=out[0:n_rows, :], data=prod[0:n_rows, :], op=nl.add, axis=1)
    return out


def _sampler_stage(vals_hbm, idx_hbm, params_hbm, n_rows, k, token_out):
    """Sampler on [rows, k]: top-k mask, temperature, softmax, top-p, position, gather.

    Each operand is loaded from HBM INSIDE this function, immediately before its use, and the
    index tile in particular is loaded last. That is not stylistic: an earlier version took
    the index tile as an SBUF argument from the caller, and because the gather is the very
    last thing this function does, that tile was reused by the sampler's own allocations
    before it got there. The gather then read zeros and every rank emitted token id 0 --
    while ``pos`` was provably correct and the top-k outputs validated clean, so only an
    end-to-end token check catches it.

    Two behaviours that look like bugs but are the model's semantics: top-p uses a STRICT
    ``>`` on the inclusive cumsum so the crossing token is DROPPED, and the mask sentinel is
    a finite ``-3000.0`` (which rounds to -3008.0 in bf16), not -inf.
    """
    P = _PMAX
    vals = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals[0:n_rows, :], src=vals_hbm[0:n_rows, :])
    # Column ramp 0..k-1, identical on every partition (channel_multiplier=0). Used to pick a
    # slot by EQUALITY instead of by nc_n_gather -- see _select_by_position.
    ramp = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(ramp, pattern=[[1, k]], offset=0, channel_multiplier=0)
    # 3 columns [top_k, top_p, temperature], or 4 with a per-row runtime draw appended -- the
    # width is a trace-time fact of the caller's tensor, so each form specialises cleanly.
    p_cols = params_hbm.shape[1]
    params = nl.ndarray((P, p_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=params[0:n_rows, :], src=params_hbm[0:n_rows, :])
    top_k_col, top_p_col, temp_col = (
        params[0:n_rows, 0:1],
        params[0:n_rows, 1:2],
        params[0:n_rows, 2:3],
    )

    # --- top-k threshold: mask everything below the (top_k-1)-th value ---
    kth = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kth[0:n_rows, :], data=top_k_col, op0=nl.minimum, operand0=float(k)
    )
    nisa.tensor_scalar(
        dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.subtract, operand0=1.0
    )
    nisa.tensor_scalar(
        dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.maximum, operand0=0.0
    )
    # kth may be NON-INTEGER (top_k is fp32 runtime data; the model floors it): the inf-safe
    # select below floors it by construction (range select), so no separate floor op runs.
    thresh = _select_by_position(vals, ramp, kth, n_rows, k, inf_safe=True)

    below = nl.ndarray((P, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=below[0:n_rows, :],
        data=vals[0:n_rows, :],
        op0=nl.less,
        operand0=thresh[0:n_rows, :],
    )
    masked = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=masked[0:n_rows, :], src=vals[0:n_rows, :])
    sent = _const_tile((P, k), _TOPK_MASK_SENTINEL)
    nisa.tensor_copy_predicated(
        dst=masked[0:n_rows, :], src=sent[0:n_rows, :], predicate=below[0:n_rows, :]
    )

    # --- temperature (greedy rows divide by 1.0 instead) ---
    greedy = nl.ndarray((P, 1), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=greedy[0:n_rows, :], data=temp_col, op0=nl.less, operand0=_SAMPLING_EPS
    )
    temp_safe = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=temp_safe[0:n_rows, :], src=temp_col)
    ones = _const_tile((P, 1), 1.0)
    nisa.tensor_copy_predicated(
        dst=temp_safe[0:n_rows, :], src=ones[0:n_rows, :], predicate=greedy[0:n_rows, :]
    )
    inv_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=inv_t[0:n_rows, :], data=temp_safe[0:n_rows, :], op=nl.reciprocal
    )

    # --- softmax (max-shifted), NO normalisation anywhere downstream ---
    # Two ops, and zero divisions on the serial chain. The temperature ratio rides the exp's
    # SCALE operand and the max shift rides its BIAS (the Scalar Engine computes
    # op(data*scale + bias)), so ``exp((masked - rmax)/temp)`` is the max-reduce plus one
    # activation, which also emits the denominator from its reduce registers. The denominator
    # is never divided by: every consumer below compares against a threshold instead, because
    # ``cum/den > p  <=>  cum > p*den`` and ``fcum/last < draw  <=>  fcum < draw*last`` for the
    # positive den/last this softmax produces. That removes two reciprocal activations, two
    # full-width normalising multiplies, and the activation-table reload the reciprocals forced
    # between the exp and the scans (1.28 us on the serial chain). The division and the
    # multiplication round differently in the last ulp -- the same class of wiggle as the PE
    # scan's summation order, arbitrated the same way: by the tie-guarded gates on hardware.
    rmax_neg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=rmax_neg[0:n_rows, :],
        data=masked[0:n_rows, :],
        op=nl.max,
        axis=1,
        negate=True,
    )
    exp_bias = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=exp_bias[0:n_rows, :],
        data1=rmax_neg[0:n_rows, :],
        data2=inv_t[0:n_rows, :],
        op=nl.multiply,
    )
    expv = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=expv[0:n_rows, :],
        data=masked[0:n_rows, :],
        op=nl.exp,
        scale=inv_t[0:n_rows, :],
        bias=exp_bias[0:n_rows, :],
        reduce_op=nl.add,
        reduce_res=den[0:n_rows, :],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    # --- top-p: strict > on the INCLUSIVE cumsum, so the crossing token is dropped ---
    # The compare is against top_p * den, so the scan runs on the raw exponentials.
    tri = _make_scan_triangle()
    cum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(cum, expv, n_rows, k, tri)
    pden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=pden[0:n_rows, :], data1=top_p_col, data2=den[0:n_rows, :], op=nl.multiply
    )
    # The compare writes 1.0/0.0 straight into fp32 so the mask multiplies without a cast.
    over_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=over_f[0:n_rows, :],
        data=cum[0:n_rows, :],
        op0=nl.greater,
        operand0=pden[0:n_rows, :],
    )
    nisa.memset(dst=over_f[0:n_rows, 0:1], value=0)  # never drop the argmax
    # filt = (1 - over) * expv in ONE fused op.
    filt = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=filt[0:n_rows, :],
        data=over_f[0:n_rows, :],
        op0=nl.subtract,
        operand0=1.0,
        reverse0=True,
        op1=nl.multiply,
        operand1=expv[0:n_rows, :],
    )
    # --- position: count cdf entries below the draw; greedy rows take slot 0 ---
    # The cdf is never materialised: the reference's ``cumsum(filt)/cumsum(filt)[k-1] < draw``
    # is evaluated as ``fcum < draw * last`` (same last-ulp caveat as the top-p compare above).
    fcum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(fcum, filt, n_rows, k, tri)
    # With a 4-column params tensor the draw u is a RUNTIME per-row operand, so sweeping u is
    # a data change rather than a recompile; with 3 columns the compiled-in constant keeps the
    # legacy contract (and its bit-exact determinism) unchanged.
    draw_thresh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=draw_thresh[0:n_rows, :],
        data=fcum[0:n_rows, k - 1 : k],
        op0=nl.multiply,
        operand0=params[0:n_rows, 3:4] if p_cols == 4 else _DETERMINISTIC_DRAW,
    )
    lt_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=lt_f[0:n_rows, :],
        data=fcum[0:n_rows, :],
        op0=nl.less,
        operand0=draw_thresh[0:n_rows, :],
    )
    pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pos[0:n_rows, :], data=lt_f[0:n_rows, :], op=nl.add, axis=1)

    # Greedy rows take slot 0 BY CONTRACT, so it costs one memset and no scan. The contract
    # (mirrored verbatim in the reference sampler): greedy returns the highest KEPT candidate,
    # and slot 0 is always kept -- the input is DESCENDING and the top-k threshold compare is
    # strict-less against a value <= vals[0]. This is deliberately NOT "argmax of the masked
    # array": in the sub-sentinel regime (every real value below the finite -3000 sentinel,
    # unreachable for real logits but legal bf16) the masked slots numerically outrank the
    # kept ones, and an argmax would return a token the caller's top_k cut explicitly
    # excluded. A masked slot is never an eligible greedy answer. On every other input the
    # two formulations agree (mask replaces a suffix with a smaller sentinel; temperature,
    # exp and top-p all preserve "slot 0 is a maximum"; ties resolve to the first maximum),
    # including fully degenerate all-equal rows.
    #
    # This replaces a max-reduce, a compare, two casts, a full ceil(log2(k))-step prefix scan and
    # a reduce (~30 Vector instructions at k=256) with a single [rows, 1] memset. If a future
    # caller feeds an UNSORTED value tile, this is the line that breaks -- the descending-order
    # gate in the tests is what protects it.
    zero_pos = _const_tile((P, 1), 0.0)
    nisa.tensor_copy_predicated(
        dst=pos[0:n_rows, :], src=zero_pos[0:n_rows, :], predicate=greedy[0:n_rows, :]
    )

    # --- gather the token id at the chosen position ---
    idx_u32 = nl.ndarray((P, k), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_u32[0:n_rows, :], src=idx_hbm[0:n_rows, :])
    idx_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_f[0:n_rows, :], src=idx_u32[0:n_rows, :])
    tok_f = _select_by_position(idx_f, ramp, pos, n_rows, k)
    tok_u32 = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_u32[0:n_rows, :], src=tok_f[0:n_rows, :])
    nisa.dma_copy(dst=token_out[0:n_rows, :], src=tok_u32[0:n_rows, :])


def _exchange_hops(cur, hops, rows_per_rank, vocab, k):
    """Stages 3-4, plain form: raw logits through every hop, de-interleaving between.

    Returns ``(full_vocab, recv_peer_width)``: the ``[rows_per_rank, vocab]`` assembled rows
    (or, when the last hop's de-interleave folds into chunked_topk, that hop's RECEIVE
    buffer with its peer width so chunked_topk reads it in place -- see
    ``interleaved_peer_width``).
    """
    cur_rows, cur_w = cur.shape
    # The LAST hop's de-interleave can be FOLDED INTO chunked_topk's own load patterns when
    # every stage-1 snake partition covers whole peer blocks (see interleaved_peer_width in
    # ChunkedTopkConfig): stage 5 then reads the receive buffer directly and the [rows, vocab]
    # materialisation -- a full HBM round trip of the received logits, the head of the
    # post-collective serial chain -- is deleted. Bitwise identical: only DMA descriptors
    # change. Holds for the single-hop production shape (peer width == V_s divides the snake
    # row width); the two-hop form's last-hop peer blocks are wider than a snake partition,
    # so it keeps the explicit de-interleave.
    recv_peer_width = 0
    for hop in range(len(hops)):
        groups = hops[hop]
        g = len(groups[0])
        kernel_assert(
            cur_rows % g == 0,
            f"hop {hop}: rows ({cur_rows}) must divide by group size ({g})",
        )
        out_rows = cur_rows // g

        recv = nl.ndarray((cur_rows, cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        ncc.all_to_all(
            dsts=[recv],
            srcs=[cur],
            replica_group=ncc.ReplicaGroup(groups),
            collective_dim=0,
        )

        # Eligibility is the CALLEE's predicate, not a re-derivation: interleaved_fold_supported
        # carries every chunked_topk precondition (chunks > 1, the desnake-path alignments,
        # the peer-width divisibilities). An earlier geometry-only condition here turned legal
        # small-vocab configs into trace-time refusals on the chunks == 1 delegate path.
        fold_deint = (
            hop == len(hops) - 1
            and out_rows > 1
            and interleaved_fold_supported(rows_per_rank, vocab, k, cur_w)
        )
        # De-interleave so peer p's block becomes columns [p*cur_w, (p+1)*cur_w). At
        # out_rows==1 the received rows are ALREADY in peer order, so laying them end to
        # end is exactly the target layout and the de-interleave is a pure reshape.
        if fold_deint:
            cur = recv
            recv_peer_width = cur_w
        elif out_rows == 1:
            cur = recv.reshape((1, g * cur_w))
        else:
            # shared_hbm, not private_hbm: this buffer is the SOURCE of the next hop's
            # collective, and every collective src/dst must be a shared buffer. One
            # descriptor per peer block: splitting each block into row halves was tried and
            # measured as no gain (the de-interleave is an HBM round trip at ~90 GB/s,
            # bound by the round trip rather than by descriptor count).
            wide = nl.ndarray(
                (out_rows, g * cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm
            )
            for p in range(g):
                nisa.dma_copy(
                    dst=wide[:, nl.ds(p * cur_w, cur_w)],
                    src=recv[nl.ds(p * out_rows, out_rows), :],
                )
            cur = wide
        cur_rows, cur_w = out_rows, g * cur_w

    kernel_assert(
        cur_rows == rows_per_rank and cur_w == vocab,
        f"hops left [{cur_rows}, {cur_w}], expected [{rows_per_rank}, {vocab}]",
    )
    return cur, recv_peer_width


@nki.jit
def gpt_oss_tail_chunked_megakernel(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    k: int = 256,
    sampling_params: Optional[nl.NkiTensor] = None,
    replica_groups: Optional[tuple] = None,
    hop_meshes: Optional[tuple] = None,
    dummy_collectives: bool = False,
    emit_logits: bool = True,
    debug_taps: bool = False,
    deferred_rms: bool = True,
    skip_nan_fold: bool = True,
    a2a_batch_groups: int = 1,
) -> Tuple[nl.NkiTensor, ...]:
    """GPT-OSS decode sampling tail with a single collective. See the module docstring.

    Args:
        hidden_states: ``[gbs, H]`` post-MoE hidden states in HBM; fp32, fp16, or bf16.
            A 16-bit input is widened to fp32 through an explicit on-chip conversion
            copy immediately after its (half-sized) load, which is EXACT (both half
            formats are subsets of fp32) -- so it produces bitwise the same outputs as
            passing a pre-widened fp32 copy, while halving this kernel's largest input
            DMA and letting a 16-bit producer skip an HBM widen round trip. The copy is
            load-bearing: engines fed 16-bit data directly compute at input precision.
        gamma: ``[1, H]`` final RMSNorm weight, fp32.
        lm_head_weight_t: ``[H, V_s]`` **bfloat16** lm_head shard for this rank, ALREADY
            TRANSPOSED and C-contiguous. ``V_s = V/VTP``. bf16 because that is the checkpoint's
            own precision -- see CALLER CONTRACTS.
        eps: RMSNorm epsilon (1e-5 for GPT-OSS).
        hidden_actual: ``H_u``, the unpadded hidden (2880). Defaults to H.
        zero_interleaved_pad: True for the mxfp4 checkpoint (interleaved padding lanes).
        k: candidates kept (256 in production).
        emit_logits: whether to return the fp32 ``[gbs, V_s]`` pre-collective logits. They are a
            DEBUG/GATING output -- nothing downstream of this kernel reads them; the collective
            carries the bf16 mirror instead. Keeping them costs a full extra HBM write of the
            shard (805 KB at the production shape, roughly half this kernel's hbm_write) plus its
            DMA and the fp32 SBUF tile. The tests set it True so the norm and the lm_head GEMM
            stay independently gated; a deployment that only wants tokens should set it False.
        sampling_params: ``[gbs/VTP, 3]`` or ``[gbs/VTP, 4]`` fp32. Columns are
            ``[top_k, top_p, temperature]``; the optional 4th column is the per-row
            inverse-transform draw ``u`` in ``[0, 1)`` -- the RUNTIME form of the compiled-in
            ``_DETERMINISTIC_DRAW`` (0.5), so sweeping the draw is a data change rather than a
            NEFF rebuild. Rows are THIS RANK'S only -- see ``owned_row_block`` for which rows
            those are when more than one hop is used. When None the sampler is skipped and no
            token is returned.
        replica_groups: one group listing all ``VTP`` ranks IN RANK ORDER. Shorthand for a
            single-hop ``hop_meshes=(replica_groups,)``; supply exactly one of the two.
        hop_meshes: one replica-group mesh per all-to-all hop, from
            ``collective_probe.compose_hops``. Hop sizes must multiply to ``VTP``. Needed
            because an all-to-all replica group must be device-local or one-core-per-device --
            see WHICH COLLECTIVE ACTUALLY RUNS in the module docstring.
        dummy_collectives: prepend a zero all-reduce fence per hop mesh so the fence, not the
            real hops, absorbs rank-arrival skew. A PROFILING instrument: it cannot change the
            answer but it adds a collective per mesh to every call, so it defaults off and
            profiling runs opt in. Without it a per-hop duration is mostly dispatch jitter.
        debug_taps: forward ``chunked_topk``'s seven diagnostic taps as extra outputs, APPENDED
            after everything else so the existing output positions do not move. A DIAGNOSTIC
            instrument for hardware-only index defects -- see DEBUG TAPS in ``chunked_topk``.
            It changes this kernel's arity, so leave it off unless the consumer expects them.
        deferred_rms: apply the per-row RMSNorm scale on the GEMM's PSUM-read copies instead
            of on the GEMM's input (exact algebra: ``rms`` is a per-row scalar and the GEMM
            is linear), which stops every matmul from waiting on the full row's sum of
            squares -- measured -3.8 us on the fenced gbs=128 head. Numerically the SAME
            one-bf16-rounding-per-element bound; only the last-ulp realization moves, so
            fixed-seed sampled tokens can shift at cdf crossings (same class as any
            resummation change). Default True; a SAFETY FALLBACK to the legacy in-chain
            form -- ``(x*rms)*gamma`` feeding the GEMM, the pre-deferred realization
            bit-for-bit -- is one trace-time flag away if a consumer needs it.
        skip_nan_fold: skip ``chunked_topk``'s stage-1 NaN fold. Sound here because the top-k
            input is this kernel's OWN GEMM output: finite hidden/gamma/weight inputs give
            finite bf16 products summed in fp32, and even an overflow saturates to +/-inf --
            which ``nisa.topk`` orders fine (the fold only exists for NaN, which would take
            inf arithmetic INSIDE the GEMM, i.e. already-garbage inputs). The fold costs a
            full-width compare + predicated copy per stage-1 tile on the post-collective
            serial chain. Default True; the fold is one trace-time flag away for a caller
            that wants the belt-and-suspenders behavior back. Outputs are BITWISE identical
            either way on any NaN-free input -- unlike ``deferred_rms`` this cannot move even
            an ulp, because it deletes ops whose effect on finite data is the identity.

        a2a_batch_groups: split the exchange into this many batch-group collectives, one
            per gbs/G whole lm_head tiles (single-hop only; G must divide rows_per_rank and
            the geometry must support the receive-layout fold -- asserted; groups aligned to
            whole 8-row sort tiles overlap best but are not required). Group g's
            exchange fires as soon as its own lm_head tiles' mirror stores land -- under
            the rest of the GEMM -- pulling the exchange forward by up to (G-1)/G of its
            wire time minus (G-1) launch serializations. OWNERSHIP CONTRACT CHANGES with
            G > 1: rank r owns the STRIDED row set given by ``owned_rows`` (one extra
            mixed-radix digit), not ``owned_row_block``'s contiguous block --
            ``sampling_params`` rows and every consumer of the per-rank outputs must use
            that map. Gate-validated, not bitwise: regrouping a collective changes the
            runtime's internal transfer realization.

    Returns:
        ``(logits, topk_values, topk_indices)`` and, when sampling, ``tokens``.
        ``logits`` is this rank's ``[gbs, V_s]`` fp32 shard. ``topk_values`` and
        ``topk_indices`` are this rank's ``[gbs/VTP, k]`` rows, with indices being GLOBAL
        vocab ids. ``tokens`` is ``[gbs/VTP, 1]``. Which batch rows this rank owns is
        ``owned_rows(rank, VTP, gbs, hops, a2a_batch_groups)``: the IDENTITY for a single
        ungrouped hop, so a rank-ordered ``all_gather`` concatenates straight into batch
        order, but a mixed-radix shuffle for multi-hop or a STRIDED set for the batch-grouped
        exchange, either of which the caller must undo.
    """
    # hidden_states may also be a LIST of row-slab tensors [rows_i, H] (concatenated-by-rows
    # == one [gbs, H] tensor) -- the pipelined-seam form, reachable only jit-in-jit (a
    # top-level kernel invocation rejects list args). Separate slab tensors are the point:
    # each lm_head row tile then data-depends only on ITS slab's producing collective, so a
    # caller that splits its final all_reduce into K row-slab collectives gets the head
    # overlapped under the later slabs (measured on the fused GPT-OSS decode: the tail
    # window from seam-collective start to tail end shrinks ~6-8% at gbs 512/1024 with
    # K = 2; gate-validated, not bitwise -- splitting a collective changes the runtime's
    # internal reduction realization).
    hidden_slabs = (
        list(hidden_states)
        if isinstance(hidden_states, (list, tuple))
        else [hidden_states]
    )
    for hs in hidden_slabs:
        kernel_assert(len(hs.shape) == 2, "hidden_states (slab) must be [rows, H]")
        kernel_assert(
            hs.dtype in (nl.float32, nl.float16, nl.bfloat16),
            f"hidden_states must be fp32/fp16/bf16 (16-bit widens exactly), got {hs.dtype}",
        )
        kernel_assert(
            hs.shape[1] == hidden_slabs[0].shape[1], "hidden_states slabs must share H"
        )
        if len(hidden_slabs) > 1:
            kernel_assert(
                hs.shape[0] % _PMAX == 0,
                f"each hidden_states slab must hold whole {_PMAX}-row tiles, got {hs.shape[0]} rows",
            )
    kernel_assert(len(lm_head_weight_t.shape) == 2, "lm_head_weight_t must be [H, V_s]")
    # Enforced, not just documented: any other dtype would still trace and compute correctly
    # (the resident load casts on DMA), but silently doubles the kernel's dominant HBM read --
    # the exact failure mode only a profile would ever catch.
    kernel_assert(
        lm_head_weight_t.dtype == nl.bfloat16,
        f"lm_head_weight_t must be bfloat16 (caller contract 1; any wider dtype would silently "
        f"inflate the dominant HBM read via cast-on-DMA), got {lm_head_weight_t.dtype}",
    )
    kernel_assert(
        (replica_groups is None) != (hop_meshes is None),
        "supply exactly one of replica_groups (single hop, all VTP ranks in rank order) or hop_meshes",
    )
    hops = (replica_groups,) if hop_meshes is None else tuple(hop_meshes)
    kernel_assert(len(hops) >= 1, "at least one hop is required")

    gbs = sum(hs.shape[0] for hs in hidden_slabs)
    hidden = hidden_slabs[0].shape[1]
    w_hidden, v_s = lm_head_weight_t.shape
    kernel_assert(
        w_hidden == hidden,
        f"weight hidden {w_hidden} must match hidden_states hidden {hidden}",
    )

    vtp = 1
    for mesh in hops:
        vtp *= len(mesh[0])
    kernel_assert(gbs % vtp == 0, f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows_per_rank = gbs // vtp
    vocab = vtp * v_s

    h_actual = hidden if hidden_actual is None else hidden_actual
    kernel_assert(
        0 < h_actual <= hidden, f"hidden_actual ({h_actual}) must be in (0, {hidden}]"
    )
    if h_actual < hidden and zero_interleaved_pad:
        kernel_assert(
            hidden % _MXFP4_SHUFFLE_GROUPS == 0
            and h_actual % _MXFP4_SHUFFLE_GROUPS == 0,
            f"mxfp4 padding needs H ({hidden}) and H_u ({h_actual}) divisible by {_MXFP4_SHUFFLE_GROUPS}",
        )
    if sampling_params is not None:
        kernel_assert(
            len(sampling_params.shape) == 2,
            "sampling_params must be 2-D: [gbs/VTP, 3] or [gbs/VTP, 4]",
        )
        kernel_assert(
            sampling_params.shape[0] == rows_per_rank,
            f"sampling_params must hold THIS RANK'S {rows_per_rank} rows, got {sampling_params.shape[0]}; "
            f"passing the full [gbs, 3] batch makes every rank use rows 0..{rows_per_rank - 1}",
        )
        kernel_assert(
            sampling_params.shape[1] in (3, 4),
            f"sampling_params needs columns [top_k, top_p, temperature] plus an optional 4th "
            f"per-row draw in [0, 1), got {sampling_params.shape[1]} columns",
        )

    # Every stage after the collective is unsharded and carries no core_barrier, so two
    # programs would race on the same shared_hbm buffers. Assert rather than assume: passing
    # num_programs=1 to an inner kernel does not pin the live grid.
    _, n_prgs, _ = get_verified_program_sharding_info(
        "gpt_oss_tail_chunked_megakernel", (0, 1), 2
    )
    kernel_assert(n_prgs == 1, f"this kernel is LNC=1 only, got n_prgs={n_prgs}")

    fence = _skew_fence(hops, hidden) if dummy_collectives else None

    # DEBUG-ONLY output, gated: see emit_logits. When off, neither the buffer nor its store
    # exists, so it costs nothing at all rather than costing a store nobody reads.
    logits = (
        nl.ndarray((gbs, v_s), dtype=nl.float32, buffer=nl.shared_hbm)
        if emit_logits
        else None
    )
    # bf16 mirror for the collective: chunked_topk is bf16-only and the collective then moves
    # half the bytes. shared_hbm because every collective src/dst must share a buffer type.
    # With a2a_batch_groups > 1 the mirror is G SEPARATE tensors, one per batch group of
    # gbs/G rows: separate tensors are what give each group's exchange a data dependency on
    # ONLY its own lm_head tiles, so group g's all-to-all fires as soon as tile-group g's
    # stores land -- under the rest of the GEMM -- instead of after the whole head.
    _bg = a2a_batch_groups
    kernel_assert(
        _bg >= 1 and gbs % _bg == 0, f"a2a_batch_groups ({_bg}) must divide gbs ({gbs})"
    )
    _bg_rows = gbs // _bg
    if _bg > 1:
        kernel_assert(
            len(hops) == 1,
            "a2a_batch_groups > 1 is defined for the single-hop exchange only",
        )
        kernel_assert(
            _bg_rows % _PMAX == 0,
            f"each batch group ({_bg_rows} rows) must be whole lm_head tiles",
        )
        kernel_assert(
            rows_per_rank % _bg == 0,
            f"a2a_batch_groups ({_bg}) must divide rows_per_rank ({rows_per_rank})",
        )
        kernel_assert(
            interleaved_fold_supported(rows_per_rank, vocab, k, v_s),
            "a2a_batch_groups > 1 requires the receive-layout fold (single-hop production-class geometry)",
        )
        # OWNERSHIP CONTRACT CHANGE: with G groups rank r owns rows {g*gbs/G + r*rpc + j}
        # (see owned_rows) -- strided, not the contiguous owned_row_block. sampling_params
        # and every consumer of this kernel's per-rank outputs must use that map.
    mirrors = [
        nl.ndarray(
            (_bg_rows, v_s),
            dtype=nl.bfloat16,
            buffer=nl.shared_hbm,
            name=f"logits_bf16_g{i}",
        )
        for i in range(_bg)
    ]

    # ---- Stages 1-2: norm + lm_head, fused through SBUF, per 128-row tile ----
    gamma_sb = _load_gamma_broadcast(gamma, hidden, h_actual, zero_interleaved_pad)
    wt_res = None
    # Slab-start offsets, so each 128-row tile reads ITS slab at a local offset (a tile
    # never straddles slabs: slab rows are whole multiples of _PMAX, asserted above).
    slab_row0 = []
    _acc_rows = 0
    for hs in hidden_slabs:
        slab_row0.append(_acc_rows)
        _acc_rows += hs.shape[0]
    for t in range(div_ceil(gbs, _PMAX)):
        row0 = t * _PMAX
        rows = min(_PMAX, gbs - row0)
        s = max(i for i in range(len(hidden_slabs)) if slab_row0[i] <= row0)
        acc_ps, rms, wt_res = _norm_lm_head_stage(
            hidden_slabs[s],
            gamma_sb,
            wt_res,
            eps,
            rows,
            row0 - slab_row0[s],
            hidden,
            h_actual,
            v_s,
            fence,
            deferred_rms,
            wt_hbm=lm_head_weight_t,
        )
        # Both readers pull straight from PSUM. With deferred_rms the per-row rms rides each
        # read as the multiply that replaces what was a plain copy -- the bf16 cast and the
        # fp32 debug copy both had to happen anyway, so normalization costs zero extra ops
        # here; without it acc_ps is already normalized and the reads are plain copies.
        # The two reads stay independent rather than the cast chaining behind the fp32 copy.
        # The mirror store is split by row halves (two contiguous descriptors on separate
        # queues): it is the last transfer standing between the GEMM and the first hop's
        # trigger, so its latency is bare critical path.
        # (An engine-parallel column-half split of this read -- Scalar || Vector, to free
        # acc_ps sooner for the next tile's PSUM -- measured ZERO at 512 and 1024 on
        # 2026-08-28: the tile-boundary PE gap does not gate on this drain. Reverted.)
        bf = nl.ndarray((_PMAX, v_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        if deferred_rms:
            nisa.tensor_scalar(
                dst=bf[0:rows, :],
                data=acc_ps[0:rows, :],
                op0=nl.multiply,
                operand0=rms[0:rows, :],
            )
        else:
            nisa.tensor_copy(dst=bf[0:rows, :], src=acc_ps[0:rows, :])
        st_half = rows // 2
        _mg, _mr0 = row0 // _bg_rows, row0 % _bg_rows
        if st_half > 0:
            nisa.dma_copy(
                dst=mirrors[_mg][_mr0 : _mr0 + st_half, :], src=bf[0:st_half, :]
            )
            nisa.dma_copy(
                dst=mirrors[_mg][_mr0 + st_half : _mr0 + rows, :],
                src=bf[st_half:rows, :],
            )
        else:
            nisa.dma_copy(dst=mirrors[_mg][_mr0 : _mr0 + rows, :], src=bf[0:rows, :])
        if emit_logits:
            logits_sb = nl.ndarray((_PMAX, v_s), dtype=nl.float32, buffer=nl.sbuf)
            if deferred_rms:
                nisa.tensor_scalar(
                    dst=logits_sb[0:rows, :],
                    data=acc_ps[0:rows, :],
                    op0=nl.multiply,
                    operand0=rms[0:rows, :],
                )
            else:
                nisa.tensor_copy(dst=logits_sb[0:rows, :], src=acc_ps[0:rows, :])
            nisa.dma_copy(dst=logits[row0 : row0 + rows, :], src=logits_sb[0:rows, :])

    # ---- Stages 3-4, plain: raw logits through every hop, de-interleaving between ----
    # Each hop is an all_to_all followed by a rows->columns de-interleave, and hops
    # compose: a hop over g peers turns [rows, w] into [rows/g, g*w]. HBM forces
    # collective_dim=0, so the all_to_all splits AND concatenates on rows -- rank r ships
    # row block p to peer p and gets every peer's block back. Because every hop's group is
    # listed in ASCENDING RANK ORDER and the first hop's is 8 contiguous ranks, peer p's
    # block is peer p's vocab shard at every hop, so after the last hop a column index
    # already IS a global vocab id and no index correction of any kind appears. What
    # multi-hop costs instead is row ownership: see owned_row_block.
    if _bg > 1:
        # G exchanges, one per batch group: exchange g's src is mirror g (whole lm_head
        # tiles), so it fires as soon as those tiles' stores land -- under the rest of the
        # GEMM -- and the last exchange trails the head by only ~1/G of the wire time plus
        # one launch. Same-mesh collectives serialize end-to-end (measured), which is why G
        # stays small and why the win is overlap, not wire speed. Each exchange delivers
        # every rank its rpc = rows_per_rank/G rows of that group as a complete peer-major
        # slab; the slabs go to chunked_topk as a LIST (per-slab data dependencies), whose
        # receive-layout fold reads each in place -- no de-interleave, no concat.
        _recv_slabs = []
        for _g in range(_bg):
            _rv = nl.ndarray(
                (_bg_rows, v_s),
                dtype=nl.bfloat16,
                buffer=nl.shared_hbm,
                name=f"a2a_recv_g{_g}",
            )
            ncc.all_to_all(
                dsts=[_rv],
                srcs=[mirrors[_g]],
                replica_group=ncc.ReplicaGroup(hops[0]),
                collective_dim=0,
            )
            _recv_slabs.append(_rv)
        full_vocab = _recv_slabs
        recv_peer_width = v_s
    else:
        full_vocab, recv_peer_width = _exchange_hops(
            mirrors[0], hops, rows_per_rank, vocab, k
        )

    # ---- Stage 5: the whole top-k, locally. Indices come back as GLOBAL vocab ids ----
    topk_out = chunked_topk(
        full_vocab,
        create_chunked_topk_config(
            inp_shape=(rows_per_rank, vocab),
            inp_dtype=nl.bfloat16,
            k=k,
            sorted=True,
            debug_taps=debug_taps,
            # The logits are this kernel's own GEMM output, provably NaN-free -- see the
            # skip_nan_fold docstring for the argument and the flag to restore the fold.
            skip_nan_fold=skip_nan_fold,
            # Non-zero when the last hop's de-interleave was folded into stage 1's loads:
            # full_vocab is then the RECEIVE buffer and stage 1 reads it in place.
            interleaved_peer_width=recv_peer_width,
            interleaved_batch_groups=_bg,
        ),
    )
    tk_vals = topk_out[0]
    tk_idx = topk_out[1]
    # The taps, if any, ride through unchanged and are appended to this kernel's outputs
    # LAST, so no existing output position moves and a consumer that ignores them works.
    tk_taps = topk_out[2:] if debug_taps else ()

    # ---- Stage 6: publish the top-k results as this kernel's own outputs ----
    # chunked_topk's buffers are private_hbm (invisible to the caller), so a republish is
    # required -- but it is a straight HBM->HBM DMA per tensor, not a bounce through SBUF.
    topk_values = nl.ndarray(
        (rows_per_rank, k), dtype=nl.bfloat16, buffer=nl.shared_hbm
    )
    topk_indices = nl.ndarray((rows_per_rank, k), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=topk_values, src=tk_vals)
    nisa.dma_copy(dst=topk_indices, src=tk_idx)

    if sampling_params is None:
        base = (
            (logits, topk_values, topk_indices)
            if emit_logits
            else (topk_values, topk_indices)
        )
        return base + tk_taps

    # ---- Stage 7: sampler, in its OWN loop over this rank's rows ----
    # Separate from the loop above so each operand is read from HBM inside the sampler, close
    # to its use. See _sampler_stage for why that ordering is load-bearing. The sampler reads
    # chunked_topk's own tk_* buffers, NOT the published shared copies: same bytes, but this
    # way the two republish DMAs above run concurrently with the sampler instead of gating it.
    tokens = nl.ndarray((rows_per_rank, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    for t in range(div_ceil(rows_per_rank, _PMAX)):
        r0 = t * _PMAX
        rows = min(_PMAX, rows_per_rank - r0)
        _sampler_stage(
            tk_vals[r0 : r0 + rows, :],
            tk_idx[r0 : r0 + rows, :],
            sampling_params[r0 : r0 + rows, :],
            rows,
            k,
            tokens[r0 : r0 + rows, :],
        )
    base = (
        (logits, topk_values, topk_indices, tokens)
        if emit_logits
        else (topk_values, topk_indices, tokens)
    )
    return base + tk_taps


# This copy's OWN export surface, not any upstream module's. It carries what this repo's integration
# imports plus the row-ownership maps. ``create_gpsimd_topk_config`` and ``gpsimd_topk`` are
# deliberately ABSENT: they now name a SECOND copy of each, and re-exporting them is the wildcard
# route by which a config could cross into nkilib's copy (see TWO COPIES above). ``chunked_topk`` is
# not re-exported upstream either -- it stays module-internal and the megakernel calls it directly.
__all__ = [
    "compose_hops",
    "gpt_oss_tail_chunked_megakernel",
    "owned_row_block",
    "owned_rows",
]
