# Prefill routed experts: nkilib shard_on_i_hybrid vs kernels/moe_cte_fp8w

Device profile of the 85k cold prefill (`scripts/profile_85k.sh`, session
c1_cold, rank 0, 4096-token chunks), before (2026-10, nkilib kernel) and after
(fp8w kernel, the FP8 weight tiles go to the matmul without a BF16 copy).

| | before | after |
|---|---|---|
| chunk 0 / 2 (target NEFF) | 450.6 / 457.6 ms | 426.1 / 445.1 ms |
| routed experts kernel (union) | 78.8 / 85.7 ms | 53.1 / 57.6 ms |
| CAST fp8 -> bf16 (sum of instructions) | 62.9 ms | 0 |
| Vector busy | 55.5 ms | 30.5 ms |
| GpSimd busy (weight DMA, HBM read) | 36.9 ms (6.2 GB) | 28.6 ms (6.75 GB) |
| Tensor busy | 45.1 ms | 33.9 ms |
| chunks 2-7 | 470, 495, 513, 539, 564, 604 ms | 460, 483, 502, 529, 554, 592 ms |

One core, the kernel alone (`T` = 4096, 3 local experts, 3 blocks): 1.14 ->
0.54 ms, bit-identical output. Inside the prefill graph the kernel is a third
faster: without the vector-engine copy it is bound by the weight DMA and the
matmuls.
