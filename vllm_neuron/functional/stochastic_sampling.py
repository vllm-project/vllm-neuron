# SPDX-License-Identifier: Apache-2.0
"""Packed full-vocabulary stochastic sampling for Neuron devices (Trainium).

The logits come from the model's forward pass; they are not random inputs.
Separately, the host supplies one uniform random number in [0, 1) per
stochastic request (batch row). The kernel uses that number to choose a token
from the probabilities computed from the logits. Seeded requests use their
own host RNG state; greedy requests consume no random number.
It implements full-vocabulary temperature sampling and returns the selected
token's raw (before temperature) or processed (after temperature) log probability.
The packed layout supports at most 128 vocabulary tiles per request.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def stochastic_sample_kernel(
    logits,
    uniforms,
    temperatures,
    tile_size=4096,
    greedy_rows=None,
    return_ranks=False,
    raw_logprobs=False,
):
    """Return token IDs and selected logprobs for caller-supplied uniforms.

    Callers supply FP32 uniforms in [0, 1) and positive FP32 temperatures
    whose reciprocals are finite normal values. Logits must contain no NaN or
    positive infinity, and each row must have at least one finite value.
    The default tile width supports vocabularies up to 524,288 tokens.
    Sampling uses parallel on-chip ISA scans and O(vocab) on-chip storage
    per request. There is no top-k/top-p filtering or filtering scratch buffer.

    For serving, ``greedy_rows`` is an optional int32 [batch, 1] mask. The
    caller must set those rows to temperature=1.
    ``return_ranks`` adds the selected-token ranks required by vLLM.
    ``raw_logprobs`` is a compile-time choice of scoring distribution only;
    token selection always uses temperature-scaled probabilities.
    """
    # Inputs: logits [B, V], uniforms [B, 1], temperatures [B, 1].
    # RNG and input-value requirements are the caller's responsibility.
    # Outputs are one int32 token ID and one FP32 selected logprob per row.

    # 1. Define the layout. Each vocabulary tile occupies one SBUF partition.
    # For V=248,320 and tile_size=4,096, a request uses 61 parallel partitions.
    batch, vocab = logits.shape
    assert 1 <= batch <= 128, "batch must be in [1, 128]"
    assert uniforms.shape == temperatures.shape == (batch, 1)
    if greedy_rows is not None:
        assert greedy_rows.shape == (batch, 1)
    assert 1 <= tile_size <= 8192, "tile_size must be in [1, 8192]"
    tiles = (vocab + tile_size - 1) // tile_size
    # Use this same packed path for every supported shape; no sampling fallback.
    assert 1 <= tiles <= 128, "packed sampling requires 1..128 vocabulary tiles"

    # 2. Allocate compact outputs in HBM (device memory). Process requests
    # sequentially; the vocabulary tiles within each request run in parallel.
    ids = nl.ndarray((batch, 1), nl.int32, buffer=nl.shared_hbm)
    scores = nl.ndarray((batch, 1), nl.float32, buffer=nl.shared_hbm)
    if return_ranks:
        ranks = nl.ndarray((batch, 1), nl.int32, buffer=nl.shared_hbm)
    full_tiles = vocab // tile_size
    tail = vocab % tile_size
    for row in range(batch):
        # 3. Load this request into on-chip SBUF as [tiles, tile_size].
        # Initialize padding to -inf so it receives zero probability.
        # DMA loads the partial tile directly into its destination partition.
        values = nl.full((tiles, tile_size), float("-inf"), nl.float32)
        if full_tiles:
            nisa.dma_copy(
                values[:full_tiles, :],
                logits[row : row + 1, : full_tiles * tile_size].reshape(
                    (full_tiles, tile_size)
                ),
            )
        if tail:
            nisa.dma_copy(
                values[full_tiles : full_tiles + 1, :tail],
                logits[row : row + 1, full_tiles * tile_size :],
            )

        # 4. Compute stable, unnormalized weights exp((logit - max) / T).
        # First reduce within tiles, then transpose the tile maxima to reduce
        # across tiles. Broadcast the resulting global maximum to every tile.
        block_max = nl.max(values, axis=1, keepdims=True)
        # Tensor-Engine transpose can turn 0 * -inf into NaN. Empty blocks
        # contribute the lowest finite FP32 value to the maximum reduction.
        block_max = nl.maximum(block_max, -3.4028234663852886e38)
        row_maximum = nl.max(nl.transpose(block_max), axis=1, keepdims=True)
        maximum = nl.broadcast_to(row_maximum, (tiles, 1))
        # Load this row's temperature directly: slicing a precomputed SBUF
        # vector here can require a partition-offset copy unsupported on Inf2.
        row_inverse = nl.ndarray((1, 1), nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(row_inverse, nl.load(temperatures[row : row + 1, :]))
        inv_t = nl.broadcast_to(row_inverse, (tiles, 1))
        weights = nl.exp(nl.multiply(nl.subtract(values, maximum), inv_t))

        # 5. Sum tile masses and compute the global normalizer Z.
        # tensor_tensor_scan with multiplier 1 and initial value 0 is cumsum.
        # Subtract each tile's own mass to get its exclusive prefix ("carry").
        mass = nl.sum(weights, axis=1, keepdims=True)
        mass_row = nl.transpose(mass)
        total = nl.sum(mass_row, axis=1, keepdims=True)
        prefix = nl.ndarray((1, tiles), nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            prefix,
            nl.full((1, tiles), 1.0, nl.float32),
            mass_row,
            0.0,
            nl.multiply,
            nl.add,
        )
        carry = nl.transpose(nl.subtract(prefix, mass_row))

        # 6. Build the unnormalized token CDF. Scan ALL tiles in parallel,
        # then add the preceding tiles' mass to each local prefix.
        # Keep weights and token CDF in SBUF, with no explicit HBM intermediate.
        local_cdf = nl.ndarray((tiles, tile_size), nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            local_cdf,
            nl.full((tiles, tile_size), 1.0, nl.float32),
            weights,
            0.0,
            nl.multiply,
            nl.add,
        )
        cdf = nl.add(local_cdf, carry)

        # 7. Inverse-CDF sampling: find the first positive-weight token with
        # CDF > uniform * Z. Using unnormalized values avoids dividing by Z.
        threshold = nl.multiply(nl.load(uniforms[row : row + 1, :]), total)
        threshold = nl.broadcast_to(threshold, (tiles, 1))
        # Convert each tile's local positions into original vocabulary IDs.
        indices = nl.ndarray((tiles, tile_size), nl.float32, buffer=nl.sbuf)
        nisa.iota(indices, [[1, tile_size]], offset=0)
        offsets = nl.ndarray((1, tiles), nl.float32, buffer=nl.sbuf)
        nisa.iota(offsets, [[tile_size, tiles]], offset=0)
        indices = nl.add(indices, nl.transpose(offsets))
        positive = nl.greater(weights, 0.0)
        crossing = nl.multiply(nl.greater(cdf, threshold), positive)
        # vocab is an out-of-range sentinel for tiles without a crossing.
        # Reduce once within tiles, then across tiles to get the earliest ID.
        candidates = nl.min(
            nl.where(
                crossing, indices, nl.full(indices.shape, float(vocab), nl.float32)
            ),
            axis=1,
            keepdims=True,
        )
        best = nl.min(nl.transpose(candidates), axis=1, keepdims=True)

        # 8. Handle endpoint rounding: the final CDF can be slightly below Z.
        # If no crossing exists, use the last positive-weight token, never a
        # masked or padded token. Then retrieve its original logit.
        last = nl.max(
            nl.where(positive, indices, nl.full(indices.shape, -1.0, nl.float32)),
            axis=1,
            keepdims=True,
        )
        last = nl.max(nl.transpose(last), axis=1, keepdims=True)
        best = nl.where(nl.less(best, float(vocab)), best, last)
        if greedy_rows is not None:
            # The caller sets T=1 for greedy rows. Their score
            # is model confidence, matching vLLM's processed-logprob convention.
            maxima = nl.min(
                nl.where(
                    nl.equal(values, maximum),
                    indices,
                    nl.full(indices.shape, float(vocab), nl.float32),
                ),
                axis=1,
                keepdims=True,
            )
            greedy_best = nl.min(nl.transpose(maxima), axis=1, keepdims=True)
            best = nl.where(nl.load(greedy_rows[row : row + 1, :]), greedy_best, best)
        chosen = nl.sum(
            # Clamp masked/padded -inf before multiplying by the selection
            # mask: this SDK also implements nl.where using multiplication.
            nl.multiply(
                nl.maximum(values, -3.4028234663852886e38),
                nl.equal(indices, nl.broadcast_to(best, (tiles, 1))),
            ),
            axis=1,
            keepdims=True,
        )
        chosen = nl.sum(nl.transpose(chosen), axis=1, keepdims=True)
        if return_ranks:
            # Positive temperature preserves logit order. Count ties using
            # original logits, avoiding ties introduced by centering or exp.
            greater = nl.sum(
                nl.greater_equal(values, nl.broadcast_to(chosen, (tiles, 1))),
                axis=1,
                keepdims=True,
            )
            rank = nl.sum(nl.transpose(greater), axis=1, keepdims=True)
            nisa.dma_copy(ranks[row : row + 1, :], rank)

        # 9. Score the same token under the requested distribution.
        # Raw scores normalize original logits; processed scores reuse the
        # sampling normalizer. Work in log space so a tiny raw probability
        # can still have a finite score when high temperature selects it.
        chosen = nl.subtract(chosen, row_maximum)
        if raw_logprobs:
            raw_mass = nl.sum(
                nl.exp(nl.subtract(values, maximum)), axis=1, keepdims=True
            )
            total = nl.sum(nl.transpose(raw_mass), axis=1, keepdims=True)
        else:
            chosen = nl.multiply(chosen, row_inverse)
        # Correct the approximate log(Z) by its first-order residual:
        # if y ~= log(Z), then log(Z) ~= y + (Z * exp(-y) - 1).
        log_total = nl.log(total)
        inverse_exp = nl.ndarray((1, 1), nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(inverse_exp, nl.exp(log_total))
        log_total = nl.add(log_total, nl.subtract(nl.multiply(total, inverse_exp), 1.0))
        score = nl.subtract(chosen, log_total)
        # Only the token ID and scalar logprob are written to output HBM.
        nisa.dma_copy(ids[row : row + 1, :], best)
        nisa.dma_copy(scores[row : row + 1, :], score)
    if return_ranks:
        return ids, scores, ranks
    return ids, scores
