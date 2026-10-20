# Data Parallelism (DP)

<!-- meta: description: Data parallelism overview -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Data Parallelism replicates the entire model across multiple independent groups of ranks. Each DP rank (a group of TP ranks) serves requests independently with no cross-DP communication during normal inference. DP increases throughput by handling more concurrent requests across replicas.

**Case study**: GPT-OSS-20B with `--tensor-parallel-size 8 --data-parallel-size 4` on 32 Neuron cores.

## Problem Statement

A single model replica (TP=8) can only process one request at a time during prefill and a limited batch during decode. To serve high-throughput workloads:

1. **Prefill bottleneck**: Prefill is compute-intensive and blocks the entire TP group. While one request prefills, no other requests can be processed.

2. **Hardware utilization**: A trn3 instance has 64 Neuron cores but a single GPT-OSS-20B replica only needs 8 cores (TP=8), leaving 56 cores idle.

3. **Throughput scaling**: More replicas = more concurrent requests = higher tokens/second.

DP solves this by running 4 independent replicas of the TP=8 model across all 32 cores.

## Process Groups

```bash
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --data-parallel-size 4
```

Creates (32 ranks total):

- **`_TP`**: 4 groups of 8 ranks each
  - TP group 0: `[0, 1, 2, 3, 4, 5, 6, 7]`
  - TP group 1: `[8, 9, 10, 11, 12, 13, 14, 15]`
  - TP group 2: `[16, 17, 18, 19, 20, 21, 22, 23]`
  - TP group 3: `[24, 25, 26, 27, 28, 29, 30, 31]`
- **`_DP`**: 8 groups of 4 ranks (cross-TP peers)
  - DP group for TP rank 0: `[0, 8, 16, 24]`
  - DP group for TP rank 1: `[1, 9, 17, 25]`
  - ...
  - DP group for TP rank 7: `[7, 15, 23, 31]`

Each DP rank is a complete, independent model replica. The model weights are identical across DP ranks — each replica is a full copy.

## How DP Affects the Model

**It doesn't.** Each DP rank runs the model independently. The model code only sees its local TP group (8 ranks) and has no awareness of other DP ranks:

```python
tp_group = get_tp_group()
# DP rank 0: tp_group.world_size = 8, ranks [0..7]
# DP rank 1: tp_group.world_size = 8, ranks [8..15]
# DP rank 2: tp_group.world_size = 8, ranks [16..23]
# DP rank 3: tp_group.world_size = 8, ranks [24..31]
```

All weight sharding, TP collectives, SP, attention, and MoE work identically to the single-replica case described in the TP doc. The model code is completely DP-unaware.

## Request Distribution and Scheduling

The vLLM engine runs a **separate scheduler per DP rank**:

```text
                       ┌─── Scheduler 0 ─── TP group 0 (ranks 0-7)
                       │
Client requests ───────┤─── Scheduler 1 ─── TP group 1 (ranks 8-15)
  (load balanced)      │
                       ├─── Scheduler 2 ─── TP group 2 (ranks 16-23)
                       │
                       └─── Scheduler 3 ─── TP group 3 (ranks 24-31)
```

Each DP rank independently:

1. Receives its own set of requests from the load balancer
2. Schedules prefill and decode independently
3. Processes requests through its local TP group
4. Returns results independently

**DP ranks are fully asynchronous**: one rank can be prefilling while another is decoding. They never wait for each other.

## DP Independence: No Cross-DP Synchronization

For models without cross-DP MoE collectives (including GPT-OSS with standard DP), DP ranks are fully independent:

### No Batch Size Synchronization

```python
# In NeuronModelRunner._get_dp_padding:
ep_enabled = self.vllm_config.parallel_config.enable_expert_parallel
if not ep_enabled:
    return 0, None  # No padding — ranks operate independently
```

Without EP, there are no cross-DP collectives that require matching batch sizes. Each rank processes whatever the scheduler gives it.

### No Dummy Batches

When one DP rank has work and another is idle:

```python
# In NeuronModelRunner.execute_dummy_batch:
ep_enabled = self.vllm_config.parallel_config.enable_expert_parallel
if not ep_enabled:
    return  # Skip — no cross-DP collectives to synchronize
```

Dummy batches exist to prevent collective deadlocks when DP ranks have different amounts of work. Without cross-DP collectives, deadlocks can't happen.

### Why This Matters

DP padding and dummy batches must only be applied when `enable_expert_parallel` is set. Without EP, DP ranks are fully independent and must NOT synchronize — they may be on different phases (one prefilling, one decoding) with incompatible shapes:

```text
DP rank 0: prefill (2048 tokens)  ──┐
                                     ├── all_reduce to synchronize → BROKEN
DP rank 1: decode (1 token)      ──┘   (shapes don't match!)
```

## TP-Local Rank

The model runner uses TP-local rank (not world rank) for the model's rank tensor:

```python
tp_group = get_tp_group()
tp_rank = tp_group.rank_in_group  # 0..7 within each DP replica
self.rank_tensor = torch.tensor(tp_rank, dtype=torch.int32, device=self.device)
```

This ensures DP doesn't affect model computation — rank 0 in DP replica 0 and rank 0 in DP replica 3 both see `tp_rank=0` and load the same weight shards.

## Decode Batch Bucketing

Each DP rank independently pads its decode batch to the nearest compiled bucket:

```python
padded_num_reqs = get_decode_padded_batch_size(
    num_reqs, max_query_len, num_seqs_buckets
)
```

DP ranks may have different actual batch sizes but each pads to its own bucket independently. The compiled graphs (from warmup) are shared via the compile cache.

## Compile Cache

All DP ranks compile identical model graphs (same shapes, same buckets). However, the compile cache key includes the TP group's replica group ranks:

```text
DP rank 0: cache key includes ranks [0,1,2,3,4,5,6,7]
DP rank 1: cache key includes ranks [8,9,10,11,12,13,14,15]
```

Different cache keys mean DP ranks can't share vLLM-level compile cache entries. But the underlying neuronx-cc compiler cache (on disk) is shared — so the first DP rank to compile populates it and subsequent ranks get cache hits at the compiler level.

## DP with EP

When `--enable-expert-parallel` is combined with DP, MoE layers introduce cross-DP dependencies. EP collectives (all-gather, reduce-scatter, all-to-all) span across DP replicas and require all ranks to participate with **identical tensor shapes**. If DP ranks have different batch sizes or one rank is idle, these collectives deadlock.

The runner handles three cases at the model runner level:

### Case 1: Both DP ranks have equal work

Both ranks bucket to the same decode batch size. `_get_dp_padding` runs an `all_reduce` across the DP group, finds the max equals the local count, and returns `num_pad=0`. No extra padding needed — normal execution proceeds.

```text
DP0: 2 requests → bucket [2] → _get_dp_padding(2) → max=2, pad=0
DP1: 2 requests → bucket [2] → _get_dp_padding(2) → max=2, pad=0
→ Both run NEFF compiled for batch=2
```

### Case 2: DP ranks have unequal work (partial imbalance)

Ranks bucket to different sizes. `_get_dp_padding` is called inside `_prepare_model_input`, **before** attention metadata is built. The smaller rank increases `padded_num_reqs` to match the larger rank. All downstream tensors (block_table, slot_mapping, input_ids, logits_indices, logit_mask) are built for the coordinated size.

```text
DP0: 3 requests → bucket [4] → _get_dp_padding(4) → max=4, pad=0
DP1: 1 request  → bucket [2] → _get_dp_padding(2) → max=4, pad=2

DP1: padded_num_reqs increases from 2 to 4
     block_table: [4, max_blocks_per_seq]  (built for 4 rows)
     slot_mapping: [4]  (extra slots = PAD_SLOT_ID)
     input_ids: [4]     (extra tokens = 0)
→ Both run NEFF compiled for batch=4
```

This must happen before `_build_attention_metadata` because the model derives shapes from block_table dimensions. Post-hoc padding of attention metadata causes shape mismatches (e.g., `view(max_blocks_per_seq, S_decode, H)` where `S_decode = tokens // block_table_rows`).

### Case 3: One DP rank is completely idle (dummy batch)

One rank has zero requests while the other has work. The idle rank's engine core calls `execute_dummy_batch` instead of `execute_model`. The dummy batch must produce tensor shapes **exactly matching** what a real decode batch of the same size would produce — otherwise `torch.compile` traces a different graph, producing a different NEFF, and the EP collectives deadlock.

```text
DP0: 2 requests → bucket [2] → normal execute_model
DP1: 0 requests → execute_dummy_batch
     starts with smallest bucket (e.g., 2)
     _get_dp_padding(2) → max=2, pad=0
     builds: input_ids[2], positions[2], logits_indices[2],
             logit_mask[2, vocab_size],
             block_table[2, max_blocks_per_seq], slot_mapping[2]
→ Both run NEFF compiled for batch=2
```

Key constraints for `execute_dummy_batch`:

- **Smallest bucket start**: Uses `num_seqs_buckets[0]` as initial token count. `_get_dp_padding` pads up to match the real rank's bucket. This avoids forcing the real rank to pad beyond its natural size.
- **Decode path**: `max_query_len=1` to avoid the prefill SP constraint (`prompt_length > world_size`).
- **Direct attn_metadata**: Builds block_table and slot_mapping directly (not via `_build_attention_metadata`) because `self.input_batch` has no requests and would produce stale shapes.
- **`max_blocks_per_seq = ceil(max_model_len / block_size)`**: Must match the real block table column count, otherwise a different NEFF is compiled.
- **All-True logit_mask**: `[num_tokens, vocab_size]` matching the real path's shape.
- **PAD_SLOT_ID slot_mapping**: Ensures dummy tokens don't write to KV cache.

### How `_get_dp_padding` coordinates ranks

Each rank creates a tensor of size `[dp_size]` initialized to zeros, writes its token count at its own index, then calls `all_reduce(SUM)`. Since each rank writes to a unique slot, the sum acts as an all-gather:

```text
DP=3, token counts: [4, 1, 0]

Rank 0 creates: [4, 0, 0]
Rank 1 creates: [0, 1, 0]
Rank 2 creates: [0, 0, 0]

After all_reduce(SUM): [4, 1, 0] on all ranks
torch.max → 4

Rank 0: pad = 4 - 4 = 0
Rank 1: pad = 4 - 1 = 3
Rank 2: pad = 4 - 0 = 4
```

### Engine-level orchestration

The DP coordination is driven by the vLLM engine core's `run_busy_loop` (in `DPEngineCoreProc`):

```python
while True:
    executed = self._process_engine_step()
    local_unfinished = self.scheduler.has_unfinished_requests()
    if not executed:
        if not local_unfinished and not self.engines_running:
            continue  # All idle, skip
        self.execute_dummy_batch()  # Idle but others have work
    self.engines_running = self._has_global_unfinished_reqs(local_unfinished)
```

When a rank has no scheduled work (`executed=False`) but other ranks are still running (`engines_running=True`), it calls `execute_dummy_batch` to participate in EP collectives. The `_has_global_unfinished_reqs` check uses a separate DP all-reduce to determine when all ranks are truly idle.

## Configuration

```bash
# 4 independent replicas of TP=8
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --data-parallel-size 4 \
    --max-num-seqs 4            # max concurrent requests per DP rank
```

No model code changes needed. DP is entirely handled by the vLLM engine (scheduling, process group creation) and the Neuron worker (core allocation).
