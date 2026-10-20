# Tensor Parallelism (TP)

<!-- meta: description: Tensor parallelism overview -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Tensor Parallelism shards model weights across ranks along specific dimensions (head dimension for attention, intermediate dimension for MoE/MLP). Each rank computes a partial result, then collectives combine them. TP is the foundational parallelism — all other parallelism strategies (EP, DP) build on top of the TP group.

**Case study**: GPT-OSS-20B with TP=8 on 8 Neuron cores.

## Problem Statement

Large language models have weight matrices that exceed the memory of a single Neuron core. For example, GPT-OSS-20B has:

- **Embedding**: `[201088, 3072]` — 600M parameters
- **Attention QKV per layer**: `[3072, 3072+2*384]` — 11.5M parameters
- **MoE experts per layer**: 32 experts × (`[3072, 6144]` + `[3072, 3072]`) — 900M parameters

With 24 layers, the total exceeds what fits on a single core. TP distributes these weights so each rank holds 1/TP_SIZE of each weight matrix, reducing per-rank memory by TP_SIZE×.

## Process Groups

When vLLM starts with `--tensor-parallel-size 8`:

```python
initialize_model_parallel(tensor_model_parallel_size=8)
```

Creates:

- **`_TP`**: All 8 ranks in one group → `[0, 1, 2, 3, 4, 5, 6, 7]`

The model accesses this via `get_tp_group()`. All TP collectives (all-gather, reduce-scatter, all-reduce) operate on this group.

## Component Sharding

### Embedding (`VocabDimShardedEmbedding`)

**Sharding**: Vocabulary dimension split across ranks. Rank k holds rows `[k*V/8, (k+1)*V/8)`.

```text
Checkpoint: [V=201088, H=3072]

Rank 0: [V/8=25136, H=3072]  ← vocab rows 0..25135
Rank 1: [V/8=25136, H=3072]  ← vocab rows 25136..50271
...
Rank 7: [V/8=25136, H=3072]  ← vocab rows 175952..201087
```

**Prefill**: Each rank looks up token embeddings from its vocab shard. Most tokens map to a single rank's shard (the others produce zeros). `reduce_scatter(dim=0)` combines partial embeddings and enters Sequence Parallel (SP) layout:

```text
All ranks: full lookup → [T, H] (mostly zeros except for local vocab range)
reduce_scatter(dim=0) → [T/8, H] per rank (SP layout)
```

**Decode**: `all_reduce` — sums partial embeddings, every rank gets full `[T, H]`.

### Attention (`GptOssAttention`)

**Sharding**: Attention heads split across ranks.

```text
GPT-OSS-20B: 64 Q heads, 8 KV heads, head_dim=45

TP=8:
  Q heads per rank: 64/8 = 8
  KV heads per rank: 8/8 = 1

TP=16 (with KV replication):
  Q heads per rank: 64/16 = 4
  KV heads per rank: 1 (replicated: 16/8 = 2 ranks share each KV head)
```

**QKV Projection** (Column-parallel):

```text
W_qkv per rank: [H=3072, q_size + 2*kv_size]

TP=8: [3072, 8*45 + 2*1*45] = [3072, 450]
  q_size = 8 heads × 45 dim = 360
  kv_size = 1 head × 45 dim = 45
```

Each rank computes its local Q/K/V heads. No collective needed — the sharding is on the output dimension.

**O Projection** (Row-parallel):

```text
W_out per rank: [q_heads_per_rank * head_dim, H]

TP=8: [360, 3072]
```

Each rank computes a partial output. The partial results are summed via collective.

**Prefill collectives**:

```text
Input: [T/8, H] (SP layout)

1. QKV: [T/8, H] @ [H, 450] → [T/8, 450] (local heads, no collective)
2. RoPE + Flash/Segmented Attention (local computation)
3. O-proj: [T/8, 360] @ [360, H] → [T/8, H] (partial sum)
4. reduce_scatter(dim=0): combines O-proj partial sums AND returns to SP
   Result: [T/8, H]
```

**Decode collectives**:

```text
Input: [T, H] (all ranks have all tokens)

1. Fused megakernel: QKV + RoPE + Attention + O-proj (all fused)
2. all_reduce: combines O-proj partial sums
   Result: [T, H] on all ranks
```

### MoE Experts (`GptOssExperts`) — TP without EP

**Sharding**: Intermediate dimension split across ranks. All experts exist on every rank.

```text
GPT-OSS-20B: 32 experts, intermediate_size=3072, hidden_size=3072

TP=8:
  Experts per rank: 32 (all experts)
  Intermediate per rank: 3072/8 = 384

  gate_up_proj per rank: [E=32, H=3072, 2*I/8=768]
  down_proj per rank:    [E=32, I/8=384, H=3072]
```

```text
Checkpoint: gate_up [32, 3072, 6144], down [32, 3072, 3072]

Rank 0: gate_up [32, 3072, 768] down [32, 384, 3072]  ← intermediate shard 0
Rank 1: gate_up [32, 3072, 768] down [32, 384, 3072]  ← intermediate shard 1
...
Rank 7: gate_up [32, 3072, 768] down [32, 384, 3072]  ← intermediate shard 7
```

**Prefill collectives** (CTE kernel):

```text
Input: [T/8, H] (SP layout)

1. Router: [T/8, H] @ [H, 32] → [T/8, 32] affinities
2. all_gather(dim=0) on tp_group:
     hidden_states: [T/8, H] → [T, H]
     affinities: [T/8, 32] → [T, 32]
   (MoE routing needs full sequence to build blockwise token dispatch)
3. Blockwise mapping: assign tokens to experts
4. CTE kernel: per-expert gate_up (column-parallel) + activation + down (row-parallel)
   Each rank computes partial intermediate → partial output
5. reduce_scatter(dim=0) on tp_group:
     [T, H] → [T/8, H]
   (Combines TP partial sums AND returns to SP layout)
```

**Decode collectives** (TKG kernel):

```text
Input: [T, H] (all tokens on all ranks)

1. TKG kernel: fused RMSNorm + Router + Expert MLP
   Each rank computes partial intermediate → partial output
2. all_reduce on tp_group:
     Combines TP partial sums
     Result: [T, H] on all ranks
```

### RMSNorm

**Sharding**: Weight `[H]` is NOT sharded — replicated on all ranks (padded to hardware alignment). RMSNorm is a per-element operation that doesn't require collectives.

### LM Head (`ColumnParallelLinear`)

**Sharding**: Vocabulary dimension split (same as embedding). Each rank computes `[T, V/8]` logits.

```text
lm_head weight per rank: [V/8=25136, H=3072]

Rank 0: computes logits for vocab 0..25135
Rank 1: computes logits for vocab 25136..50271
...
```

**Without on-device sampling**: `all_gather(dim=1)` to reconstruct full `[T, V]` logits.

**With on-device sampling**: Sampler handles TP-sharded logits internally — finds the global argmax across shards without gathering.

## Sequence Parallelism (SP)

SP is tightly coupled with TP. During prefill, the token dimension is distributed across ranks between collective operations:

```text
Full sequence [T, H]
  → embedding reduce_scatter → [T/8, H] per rank         (enter SP)
  → decoder layer:
      → attention all-gather internally on [T/8, H]
      → attention reduce-scatter back to [T/8, H]         (stay in SP)
      → MoE all-gather to [T, H] for routing
      → MoE reduce-scatter back to [T/8, H]               (stay in SP)
  → ... repeat for all layers ...
  → final norm on [T/8, H]
  → all-gather → [T, H]                                   (exit SP)
  → logits computation
```

During decode, there is no SP — all ranks process all `[T, H]` tokens and use all-reduce for collectives. This is because decode typically has small T (batch_size tokens, one per request) where splitting provides no benefit.

**SP validation**: The model asserts during prefill that `T > world_size` and `T % world_size == 0`, since SP requires evenly divisible token counts.

## Weight Loading

Each rank loads only its shard of each weight matrix via `SafetensorsCheckpoint.load_sharded_pipelined`:

```python
checkpoint = SafetensorsCheckpoint(checkpoint_path)
rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
    rank=tp_group.rank_in_group,
    world_size=tp_group.world_size,
    model=model,
    mappings=weight_mappings,
    device=device,
)
```

Weight loaders attached to each parameter define how to shard. Common patterns:

- **Column-parallel** (QKV, gate_up, lm_head): shard on output dimension
- **Row-parallel** (O-proj, down): shard on input (intermediate) dimension
- **Replicated** (RMSNorm, router, biases): no sharding, optional padding

## Runner/Worker Implications

- **Rank tensor**: `rank_tensor = tp_group.rank_in_group` — the model receives its TP-local rank (0..7), not the world rank. This is important for weight indexing and EP rank calculations.
- **Warmup**: Compiles separate graphs for each prefill bucket and decode bucket. Total compiled shapes = `len(prefill_buckets) + len(decode_buckets)`.
- **SP constraint**: Prefill bucket sizes must be divisible by `tp_group.world_size`. The scheduler pads token counts to the nearest bucket.
