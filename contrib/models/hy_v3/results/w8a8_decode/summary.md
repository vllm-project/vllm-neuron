# Decode routed experts: does W8A8 make the matmuls faster?

`scripts/tensore_mm_bench.py`, one NeuronCore (trn2, LNC=2), the matmul shapes
of the decode MoE kernel (`moe_tkg` all-expert, T = 32 tokens: an 8-request
MTP k=3 verify step). Cost per 128-row contraction, from the time difference of
8192 and 1024 matmuls in one NKI kernel (`NKI_DISABLE_TRACE_CACHE=1`):

| gate / up: stationary hidden [128, T], moving FP8 weight [128, 512] | us |
|---|---|
| BF16 hidden, column tiling (what `moe_tkg` does: 128 / T matmuls side by side) | 0.078 |
| BF16 hidden, no tiling | 0.237 |
| FP8 hidden (W8A8), no tiling | 0.243 |
| FP8 hidden, `perf_mode=double_row` (2 x 128 rows per instruction; cannot be combined with column tiling) | 0.149 |

| down: stationary FP8 weight [128, 128], moving hidden [128, T] | us |
|---|---|
| BF16 hidden | 0.032-0.043 (two runs) |
| FP8 hidden (W8A8) | 0.033-0.034 (two runs) |

- An FP8 x FP8 matmul without `double_row` runs at the BF16 x FP8 rate.
- `double_row` doubles the FP8 rate, but at decode sizes the column-tiled
  BF16 x FP8 matmul is ~2x faster than it, and the two cannot be combined.
  At T = 4 `double_row` does not compile (BIR verification failed).
- The down projection is bound by loading the weight tile (stationary); the
  moving operand's dtype makes no measurable difference (at most ~0.8 ms per
  step over its ~91k matmuls).

So W8A8 cannot speed up the decode MoE matmuls on TRN2 at these shapes; it
would only add the activation quantization. Not implemented.
