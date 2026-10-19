# Block Packing Vision Attention

<!-- meta: description: FFD block packing for multi-image attention efficiency -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-07-10 -->

## Overview

Block Packing Vision Attention eliminates wasted computation in multi-image attention by packing variable-sized image token sequences into fixed-size blocks using First-Fit-Decreasing (FFD) bin packing. Each block is processed as an independent batch element, removing all cross-block attention and delivering 2x-2.5x efficiency gains over standard approaches.

## Problem Statement

Vision understanding models apply all-to-all attention independently within each image — every visual token in one image attends to every other token in that same image. When a request contains multiple images at varying resolutions, the token sequences differ in length and must be organized for batched computation. The two standard approaches both incur significant waste:

### Standard approach 1: Batch-dimension packing

Each image occupies a separate batch element, padded to the length of the largest image:

```text
Image A: 1024 tokens ████████████████
Image B:   64 tokens ██░░░░░░░░░░░░░░  (93% padding)
Image C:  384 tokens ██████░░░░░░░░░░  (62% padding)
                     ↑ padded to 1024
```

Short sequences carry massive padding overhead and require two-dimensional bucketing (number of images × max image size).

### Standard approach 2: Sequence-dimension concatenation

All image tokens are concatenated into a single sequence with minimal trailing padding:

```text
[Image A: 1024 tokens][Image B: 64 tokens][Image C: 384 tokens][pad: 64]
 ← total: 1536 tokens (next bucket) →
```

This minimizes padding but computes a full `seq_len × seq_len` attention matrix. A non-causal mask prevents cross-image attention, but the kernel still executes the masked-out regions. For the example of two images (1024 + 2048 tokens):

- Useful compute: `1024² + 2048²` = 5.2M
- Total compute: `3072²` = 9.4M
- **Wasted: 44%**

The waste grows quadratically with more images. For 24 small images of 128 tokens each:

- Useful compute: `24 × 128²` = 393K
- Total compute: `3072²` = 9.4M
- **Wasted: 95.8%**

## Solution: Block Packing with FFD

Block packing restructures the computation by grouping images into fixed-size blocks rather than one flat sequence or one-per-batch-element:

1. **Pack** image token sequences into blocks of a compile-time-fixed size using First-Fit-Decreasing (FFD).
2. **Process** each block as an independent batch element — the attention kernel computes only the `block_size × block_size` matrix per block with no cross-block computation.

### First-Fit-Decreasing (FFD) Algorithm

FFD is a bin-packing heuristic that operates in two steps:

1. **Sort** images by token count in descending order.
2. **Place** each image into the first block that has sufficient remaining capacity.

FFD guarantees at most `⌈11/9 · OPT⌉ + 1` blocks, where OPT is the theoretical minimum — worst-case within 22% of optimal. In practice, production image-size distributions (heterogeneous resolutions with natural size gaps) achieve optimal or OPT+1 packing.

### Worked Example

Consider 5 images: one 1024-token, one 512-token, one 256-token, and two 128-token images, with `block_size = 1024`:

```text
Block 1: [Image 1K]                    → 1024/1024 used
Block 2: [Image 512][Image 256][2×128] → 1024/1024 used
```

Compute comparison:

| Approach | Total compute | Useful compute | Efficiency |
| --- | --- | --- | --- |
| Batch packing (pad to 1024) | 5 × 1024² | 1024² + 512² + 256² + 2×128² | ~28% |
| Sequence packing (flat) | 2048² | same | ~35% |
| **Block packing (1024)** | 2 × 1024² | same | **~70%** |

Block packing achieves **2.5x** better efficiency than batch packing and **2x** better than sequence packing in this scenario.

### Relationship to Standard Approaches

Block packing generalizes both standard approaches:

- **When `block_size` = total sequence length**: a single block spans the entire attention matrix, reducing to standard sequence-dimension concatenation.
- **When all images have equal size**: setting `block_size` to that size (rounded to the next multiple of 128) gives each image its own block, reducing to batch-dimension parallelism.
- **When images vary in size** (the common case): block packing is strictly more efficient than both alternatives.

## How It Works

### Step 1: Preprocessing (Framework Level)

The framework performs FFD packing before kernel dispatch:

1. Sort images by token count (descending).
2. Assign each to the first block with remaining capacity ≥ image token count.
3. Shuffle vision token position IDs to reflect the image-to-block mapping — the result is a `[num_blocks × block_size]` tensor with padded positions at each block's tail.
4. Compute `bound_min` and `bound_max` tensors per block, encoding the local start/end indices of each image within that block.

### Step 2: Execution (Kernel)

The packed tensor is dispatched to the `attention_cte` prefill kernel:

- `num_blocks` maps to the kernel's batch dimension.
- `block_size` maps to the kernel's sequence length (`seqlen_q`).
- The kernel processes blocks sequentially, using `bound_min`/`bound_max` to restrict attention to valid image-local regions within each block.

Bucketing occurs on two dimensions:

1. **Block size** — determined at compile time based on maximum image resolution.
2. **Number of blocks** — computed as `max_seq_len // block_size`.

## Choosing the Right Block Size

Block size is the key tuning parameter. The right value depends on your workload characteristics:

### Guidelines

In general, we want to balance intra-block padding with the wasted compute of fully padded blocks. Efficient packing will produce fewer blocks with less intra-padding within each block, which reduce both.

| Workload | Recommended block size | Rationale |
| --- | --- | --- |
| Single large image (e.g., 4K×4K) | Match the image token count (rounded to 128) | Single block, no packing overhead |
| Many small images (e.g., thumbnails) | 1024 | Packs many images per block, minimizes block count — fewer blocks means less wasted compute from fully padded trailing blocks and better NeuronCore utilization along the sequence dimension |
| Mixed resolutions | 1024–2048 | Balances packing density against per-block padding |
| Video workload | Multiple of `num_tokens_per_frame` (rounded to 128) | Aligns block boundaries to frame boundaries, avoiding mid-frame splits (e.g., 16 tokens/frame × 64 frames = 1024) |
| Max resolution known at deploy time | `max_num_tokens_per_image` (rounded to 128) | Guarantees any single image fits in one block |

### Key Considerations

1. **Block size must be ≥ the largest single image's token count.** If an image exceeds the block size, it cannot be packed and the kernel will fail. Set block size to at least `max_num_tokens_per_image` for your model's configured resolution.

2. **Larger blocks = fewer blocks but more intra-block padding.** A 4096-token block can absorb any combination of smaller images, but each block carries more unused slots when images don't fill it completely.

3. **Smaller blocks = more blocks but better packing density.** A 1024-token block forces large images into dedicated blocks while letting small images share blocks efficiently. However, very small blocks increase the batch dimension and may underutilize compute for workloads dominated by large images.

4. **Alignment constraint: block size must be a multiple of 128** to satisfy NKI kernel tile alignment requirements.

### Decision Flowchart

```text
What is your max single-image token count?
│
├─ ≤ 1024  → block_size = 1024
│             (good packing density)
│
├─ ≤ 2048  → block_size = 2048
│             (accommodates large images, still packable)
│
└─ > 2048  → block_size = ceil(max_tokens / 128) × 128
              (ensures single-image fit, packing for smaller images)
```

### Example: Estimating Efficiency

For a request with `N` images producing token counts `[t₁, t₂, ..., tₙ]` and a chosen `block_size B`:

```text
Useful compute  = Σ tᵢ²
Block compute   = num_blocks × B²
Efficiency      = Useful compute / Block compute
```

Where `num_blocks` is determined by FFD packing. Higher efficiency means less wasted compute.

## Performance Impact

The efficiency gain depends on workload heterogeneity:

- **Homogeneous images** (all same resolution): Block packing matches batch-dimension parallelism — no gain, no regression.
- **Heterogeneous images** (mixed resolutions): 2x–2.5x efficiency improvement over the best standard approach.
- **Many small images**: Up to 10x+ improvement over flat sequence packing (e.g., 24 thumbnails: ~4% → ~50%+ efficiency).

## Key Files

| File | Purpose |
|------|---------|
| `vllm_neuron/model/qwen3_vl/utils/vision_block_packing.py` | FFD packing, block scatter, bounds computation |
| `vllm_neuron/model/qwen3_vl/utils/vision_preprocessing.py` | CPU preprocessing (RoPE, attention bounds from `grid_thw`) |
| `vllm_neuron/model/qwen3_vl/vision_encoder_bf16.py` | Vision encoder (block-level forward with `bound_min`/`bound_max`) |
| `vllm_neuron/model/qwen3_vl/model_bf16.py` | Model integration (packing invocation, position shuffling) |
| `vllm_neuron/functional/attention/attention_cte.py` | NKI attention kernel (`attention_cte` with sequence bounds) |
