# Expert Parallelism (EP)

<!-- meta: description: Expert parallelism for MoE -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Expert Parallelism distributes MoE experts across ranks. Unlike tensor parallelism which shards each expert's weight matrices, EP assigns different subsets of experts to different ranks. Each rank owns its experts' full weights and processes all tokens against only its local experts.

EP is a logical remapping of the TP group for MoE layers — attention and other dense layers continue to use standard TP on the same ranks. Enabled via `--enable-expert-parallel`.

A specific EP degree can be set via `ep_degree` in neuron config. By default, EP spans all ranks (`ep_degree = TP × DP`). When `ep_degree < TP`, MoE layers use EP for expert partitioning AND TP for intermediate sharding simultaneously. With DP, `ep_degree` must equal `TP × DP` — MoE layers exchange tokens across DP replicas via all-gather/reduce-scatter on the DP group.

**Case study**: GPT-OSS-20B with `--tensor-parallel-size 8 --enable-expert-parallel` on 8 Neuron cores.

## Problem Statement

MoE models present unique scaling challenges that tensor parallelism alone cannot address:

1. **Small Intermediate Dimensions**: High TP degrees result in very small per-rank intermediate sizes. With `intermediate_size=3072` and TP=64, each rank computes only 48 features — too small for optimal hardware utilization.

2. **Memory vs Compute Tradeoff**: TP shards every expert's weights, but during inference only top-k experts (e.g., 4 out of 32) are active per token. All ranks store all 32 experts but only use a fraction, wasting memory on inactive expert weights.

3. **Load Balancing**: Expert selection is data-dependent. Some experts may be selected more frequently than others, creating load imbalance across devices.

EP solves this by distributing experts instead of sharding them:

```text
TP=8 only (no EP):    32 experts × I/8=384 per rank  → small intermediate
EP=8 (with TP=1):     4 experts × I=3072 per rank    → full intermediate, 8x fewer experts
```

## Process Groups

### Full EP (default)

With `--enable-expert-parallel` and no `ep_degree` specified:

```python
initialize_model_parallel(tensor_model_parallel_size=8)
```

Creates:

- **`_TP`**: `[0, 1, 2, 3, 4, 5, 6, 7]` — used by attention (TP=8) and by MoE (reinterpreted as EP=8)
- **`_NEURON_EP_TP`**: Single-rank groups `[0], [1], ..., [7]` — TP sub-groups within each EP partition (world_size=1, no-op collectives)
- **`_NEURON_EP`**: `[0, 1, 2, 3, 4, 5, 6, 7]` — EP group (all ranks)

The model reads EP configuration from `neuron_parallel_state`:

```python
if self.ep_enabled:
    self.ep_degree = get_neuron_ep_degree()      # 8
    self.ep_rank = get_neuron_ep_rank()           # 0..7
    self.tp_degree = tp_group.world_size // ep_degree  # 1
    self.ep_tp_group = get_neuron_ep_tp_group()   # single-rank GroupCoordinator
```

### Variable EP (EP+TP for MoE)

With `--enable-expert-parallel` and `ep_degree=2` on 8 ranks:

- **`_TP`**: `[0, 1, 2, 3, 4, 5, 6, 7]` — used by attention (TP=8)
- **`_NEURON_EP_TP`**: `[0, 1, 2, 3]`, `[4, 5, 6, 7]` — TP sub-groups within each EP partition (world_size=4)
- **`_NEURON_EP`**: `[0, 4]`, `[1, 5]`, `[2, 6]`, `[3, 7]` — EP groups (ranks at same position across partitions)

```python
if self.ep_enabled:
    self.ep_degree = get_neuron_ep_degree()      # 2
    self.ep_rank = get_neuron_ep_rank()           # 0 for ranks 0-3, 1 for ranks 4-7
    self.tp_degree = tp_group.world_size // ep_degree  # 4
    self.ep_tp_group = get_neuron_ep_tp_group()   # [0,1,2,3] or [4,5,6,7]
```

**Key insight**: Attention always uses the full `_TP` group. MoE uses `ep_tp_group` for TP coordination within the EP partition (blockwise mapping) and the full `_TP` group for outer collectives (all-gather, reduce-scatter, all-reduce). The single full-group collective correctly combines both TP partial sums and EP expert contributions since addition is associative.

### Cross-DP EP (DP + EP)

With `--enable-expert-parallel` and `--data-parallel-size 2` on 8 ranks (TP=4, DP=2):

```text
Ranks: [0, 1, 2, 3, 4, 5, 6, 7]

_TP group 0 (DP replica 0): [0, 1, 2, 3]   ← attention TP=4
_TP group 1 (DP replica 1): [4, 5, 6, 7]   ← attention TP=4

_DP groups: [0,4], [1,5], [2,6], [3,7]      ← cross-DP peers

ep_degree = TP × DP = 8, each rank owns 32/8 = 4 experts
```

Process groups created:

- **`_TP`**: `[0,1,2,3]`, `[4,5,6,7]` — attention TP=4 per DP replica
- **`_DP`**: `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` — vLLM DP groups, used for cross-DP token exchange
- **`_NEURON_EP_TP`**: `[0],[1],...,[7]` — single-rank groups (tp_degree_moe=1)
- **`_NEURON_EP`**: `[0,1,2,3,4,5,6,7]` — all ranks in EP domain

The model detects cross-DP EP when `dp_size > 1 and ep_enabled`:

```python
self.cross_dp_ep = self.dp_size > 1 and self.ep_enabled
if self.cross_dp_ep:
    self.dp_group = get_dp_group()  # [0,4], [1,5], etc.
```

## Expert Weight Distribution

### How EP Differs from TP for MoE

| Aspect | TP=8 (no EP) | EP=8 (full EP) | EP=2 + TP=4 (variable) |
|--------|-------------|------|------------------------|
| Experts per rank | 32 (all) | 32/8 = 4 | 32/2 = 16 |
| Intermediate per rank | 3072/8 = 384 | 3072 (full) | 3072/4 = 768 |
| gate_up weight shape | `[32, 3072, 768]` | `[4, 3072, 6144]` | `[16, 3072, 1536]` |
| down weight shape | `[32, 384, 3072]` | `[4, 3072, 3072]` | `[16, 768, 3072]` |
| What's distributed | Intermediate dimension | Expert assignment | Both |
| Collective semantics | Sum partial intermediates | Sum expert contributions | Sum both (single collective) |

### Weight Layout Across Ranks

```text
Checkpoint: 32 experts, each with gate_up [H, 2*I] and down [I, H]

EP=8 (4 experts per rank, full intermediate):

Rank 0: Experts [0,1,2,3]     gate_up [4, 3072, 6144]  down [4, 3072, 3072]
Rank 1: Experts [4,5,6,7]     gate_up [4, 3072, 6144]  down [4, 3072, 3072]
Rank 2: Experts [8,9,10,11]   gate_up [4, 3072, 6144]  down [4, 3072, 3072]
...
Rank 7: Experts [28,29,30,31] gate_up [4, 3072, 6144]  down [4, 3072, 3072]

Memory per rank: 4 experts × full weights (vs 32 experts × 1/8 weights with TP)
```

### Router Weights

The router weight `[E_total=32, H=3072]` is **replicated** on all ranks. Every rank computes affinities for all 32 experts globally, then slices to its local experts. This ensures consistent routing decisions across ranks.

## Collective Patterns

### Prefill (CTE kernel)

```text
Input: [T/8, H] (SP layout from attention reduce-scatter)

Step 1: Router
  [T/8, H] @ [H, 32] → [T/8, 32] affinities (local tokens, all experts)

Step 2: All-gather on tp_group (all 8 ranks)
  hidden_states: [T/8, H] → [T, H]
  affinities:    [T/8, 32] → [T, 32]
  (Every rank now sees the full sequence — needed for global token-to-expert dispatch)

Step 3: Slice to local experts
  [T, 32] → [T, 4] via get_local_expert_affinities(affinities, local_expert_indices)

Step 4: Build blockwise mapping
  Assigns all T tokens to the 4 local experts based on routing affinities

Step 5: CTE kernel
  Process T tokens × 4 local experts (full intermediate, no TP sharding)
  Output: [T, H] (this rank's contribution from its 4 experts)

Step 6: Reduce-scatter on tp_group
  [T, H] → [T/8, H]
  Combines expert contributions from all 8 ranks AND returns to SP layout

  Why this works: Each rank contributed output from different experts.
  rank 0: experts 0-3 output, rank 1: experts 4-7 output, ...
  reduce_scatter sums them → complete MoE output (all 32 experts combined)
```

### Decode (TKG kernel)

```text
Input: [T, H] (all ranks have all tokens, no SP during decode)

Step 1: TKG kernel (fused RMSNorm + Router + Expert MLP)
  Routes tokens to local 4 experts, computes expert outputs
  Output: [T, H] (this rank's contribution)

Step 2: All-reduce on tp_group
  Sums contributions from all 8 ranks
  Result: [T, H] with all 32 experts' contributions combined
```

### Why a Single All-Reduce/Reduce-Scatter Works

With full EP=8, each rank contributes output from 4 different experts. Tokens not routed to a rank's experts produce zero output (affinity = 0). Summing across all ranks gives the correct weighted combination of all 32 experts:

```text
Token t is routed to experts [2, 7, 15, 28] with affinities [0.3, 0.25, 0.25, 0.2]

Rank 0 (experts 0-3):   contributes expert 2's output × 0.3
Rank 1 (experts 4-7):   contributes expert 7's output × 0.25
Rank 3 (experts 12-15): contributes expert 15's output × 0.25
Rank 7 (experts 28-31): contributes expert 28's output × 0.2
Other ranks:            contribute zeros for this token

all_reduce(sum) → correct weighted sum of all 4 selected experts
```

With variable EP+TP (e.g., EP=2, TP=4), each rank's output is a **partial sum** from both:

- **EP**: only contributions from local experts (16 out of 32)
- **TP**: partial intermediate products (1/4 of the intermediate dimension)

A single all-reduce/reduce-scatter across the full TP group correctly combines both because addition is associative — summing across all 8 ranks simultaneously reduces TP partial products within each EP partition AND combines expert contributions across EP partitions.

### Cross-DP EP: Decode

```text
DP0 ranks [0-3] have tokens T_a, DP1 ranks [4-7] have tokens T_b

Step 1: All-gather across DP group [0,4]
  [T, H] → [T*2, H]  (each rank sees both replicas' tokens)

Step 2: TKG kernel (routes all tokens to local experts)
  [T*2, H] → [T*2, H]  (partial: only local experts' contribution)

Step 3: All-reduce across TP group [0,1,2,3]
  Combines expert contributions from TP peers within DP replica

Step 4: Reduce-scatter across DP group [0,4]
  [T*2, H] → [T, H]  (sums cross-DP contributions and redistributes)
```

### Cross-DP EP: Prefill

```text
Step 1: All-gather within TP group (undo SP)
  [T/4, H] → [T, H]

Step 2: All-gather across DP group
  [T, H] → [T*2, H]  (+ positions gathered too)

Step 3: Router + local expert slicing + blockwise mapping + CTE kernel
  [T*2, H] → [T*2, H]

Step 4: Reduce-scatter across DP group
  [T*2, H] → [T, H]

Step 5: Reduce-scatter within TP group (return to SP)
  [T, H] → [T/4, H]
```

### Variable EP: Blockwise Mapping with ep_tp_group

In the prefill path, `build_blockwise_mapping` coordinates TP sharding of the expert work. With variable EP, it receives `ep_tp_group` (the TP sub-group within the EP partition) instead of the full TP group:

```text
Full EP (EP=8):     ep_tp_group = single-rank group (world_size=1), tp_degree=1
                    → non-sharded path, each rank processes all local experts independently

Variable (EP=2):    ep_tp_group = [0,1,2,3] or [4,5,6,7] (world_size=4), tp_degree=4
                    → sharded path, ranks within partition coordinate on expert work

No EP (TP=8):       ep_tp_group = full tp_group (world_size=8), tp_degree=8
                    → sharded path, all ranks coordinate (same as before EP support)
```

## Attention with EP

Attention layers are **unaffected** by EP. They continue to use TP=8:

- Q heads sharded across 8 ranks (8 Q heads per rank for 64 total)
- KV heads replicated as needed (1 per rank for 8 total)
- Same collectives as described in the TP doc (reduce-scatter for prefill, all-reduce for decode)

The EP flag only changes MoE layer behavior. Attention uses `tp_group` for TP collectives, MoE uses the same `tp_group` for EP collectives — same group, different semantics per layer type.

## Weight Loading

Weight loaders must account for EP — loading only the local experts' weights:

```python
# In GptOssExperts.load_weights:
if self.ep_degree > 1:
    # Wrap base loader to filter to this rank's expert indices
    loader = expert_parallel_weight_loader(
        local_expert_indices=[ep_rank * E_L, ..., (ep_rank+1) * E_L - 1],
        original_loader=base_sharding_loader,
        expert_dim=0,
    )
```

The `expert_parallel_weight_loader` wraps an existing loader and extracts only the rows corresponding to this rank's expert indices from the checkpoint tensor. It composes with any base loader (e.g., TP sharding, padding, dequantization).

**Router weights** use the standard (non-EP) loader since they're replicated across all ranks.

### Example: Loading gate_up_proj with EP=8 (full EP)

```python
# Checkpoint tensor: gate_up_proj [E=32, H=3072, 2*I=6144]
# EP rank 1 needs experts [4,5,6,7]

# 1. Base loader: no TP sharding (tp_degree=1), just load full weights
base_loader = identity_loader()

# 2. EP wrapper: extract experts 4-7
ep_loader = expert_parallel_weight_loader(
    local_expert_indices=[4, 5, 6, 7],
    original_loader=base_loader,
    expert_dim=0,
)

# Result: [4, 3072, 6144] — only local experts, full intermediate
```

### Example: Loading gate_up_proj with EP=2, TP=4 (variable EP)

```python
# Checkpoint tensor: gate_up_proj [E=32, H=3072, 2*I=6144]
# EP rank 0, TP rank 2 (global rank 2) needs experts [0..15], intermediate shard 2/4

# 1. Base loader: TP sharding on intermediate dim (shard_size=I/4, num_shards=4)
base_loader = expert_gate_up_weight_sharding_loader(
    shard_size=768*2, num_shards=4, hidden_size=3072,
)

# 2. EP wrapper: extract experts 0-15 from the already-TP-sharded result
ep_loader = expert_parallel_weight_loader(
    local_expert_indices=list(range(16)),
    original_loader=base_loader,
    expert_dim=0,
)

# Result: [16, 3072, 1536] — local experts with TP-sharded intermediate
# rank % num_shards = 2 % 4 = 2 → correct TP shard within EP partition
```

## Configuration

EP is enabled via vLLM's `--enable-expert-parallel` flag. By default, the EP degree equals `tp_group.world_size` (full EP). To use a variable EP degree, set `ep_degree` in `neuron_config`:

```bash
# Full EP: 8 cores, EP=8 for MoE (tp_degree=1), TP=8 for attention
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --enable-expert-parallel

# Variable EP: 8 cores, EP=2 + TP=4 for MoE, TP=8 for attention
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --additional-config '{"neuron_config": {"ep_degree": 2}}'
```

**Validation**: The worker enforces that `ep_degree` requires `--enable-expert-parallel`. Setting `ep_degree` without the flag raises a `ValueError`.

**Defaults**: When `--enable-expert-parallel` is set without `ep_degree`, the worker defaults `ep_degree` to `world_size` (`TP × DP`), ensuring experts span all ranks. The Neuron EP process groups (`GroupCoordinator`) are always created when EP is active.

**DP constraint**: With `data_parallel_size > 1`, `ep_degree` must equal `TP × DP`. Setting a different `ep_degree` with DP raises a `ValueError`.

## Limitations

1. **Divisibility Constraints**: `world_size % ep_degree == 0` and `num_experts % ep_degree == 0` must hold.

2. **Cross-DP EP requires `ep_degree = TP × DP`**: When DP is active, EP must span all ranks. Partial cross-DP (EP spanning some but not all DP replicas) is not supported.

3. **Static Expert Assignment**: Experts are assigned in contiguous blocks at initialization (rank k gets experts `[k*E_L, (k+1)*E_L)`). No dynamic load balancing or expert replication.

4. **Contiguous Assignment Only**: Round-robin or custom expert placement patterns are not supported.

## Runner/Worker Implications

- **EP process groups**: When `--enable-expert-parallel` is set, the worker creates Neuron EP process groups (`_NEURON_EP`, `_NEURON_EP_TP`) via `init_neuron_distributed_environment`. These are `GroupCoordinator` instances accessible via `get_neuron_ep_group()`, `get_neuron_ep_tp_group()`, `get_neuron_ep_degree()`, and `get_neuron_ep_rank()`.
- **Worker default**: When `--enable-expert-parallel` is set without `ep_degree` in neuron config, the worker defaults `ep_degree` to `tensor_parallel_size` (full EP, inline with vLLM default).
- **Validation**: Setting `ep_degree > 1` without `--enable-expert-parallel` raises a `ValueError` in the worker.
- **`enable_expert_parallel` flag**: Read from `vllm_config.parallel_config.enable_expert_parallel`. Toggles MoE between TP sharding (intermediate dim) and EP sharding (expert assignment), or a combination of both with variable EP degree.
- **DP padding**: When `enable_expert_parallel=True` with DP, the model runner synchronizes batch sizes across DP ranks inside `_prepare_model_input` (before attention metadata is built). When one DP rank is completely idle, `execute_dummy_batch` runs a shape-matched dummy forward pass. See the "DP with EP" section in the DP doc for the three cases and implementation details.
- **Warmup**: Same as TP — no special EP warmup needed. The compiled graphs handle both attention (TP) and MoE (EP) paths.
