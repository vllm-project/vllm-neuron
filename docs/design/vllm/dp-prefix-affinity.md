# DP prefix-affinity routing

## Problem

With `--data-parallel-size` greater than 1, one server runs several engines,
and each engine has its own prefix cache: the KV cache blocks of the full
attention layers, and for Qwen3.5 the DeltaNet state snapshots (see
`vllm_neuron/vllm/core/deltanet_snapshots.py`). The server sends each request
to the least loaded engine. It does not look at the prompt.

An agent rollout sends one request per turn, and each turn starts with all
the tokens of the turn before. With N engines, a turn lands on the engine of
the turn before only about 1 time in N. On any other engine it prefills its
whole history again, and that prefill also slows the decode of the other
requests on that engine. The rollouts of one prompt (a GRPO group) spread the
same way, so each engine prefills the shared prompt once.

No session ID reaches the server. The Tinker `sample()` call that ART's rollout
gateway makes has no session field. SkyRL forwards each sample with an
`X-Session-ID` header that combines the sampling client and a per-request
sequence number, so two turns of one session carry different values. vLLM does
not read that header.

## Design

`VLLM_NEURON_DP_PREFIX_AFFINITY=1` turns on an index in the DP load balancer
of the API server (`vllm_neuron/vllm/patches/dp_prefix_affinity.py`, called
from `dp_load_balancer_patch.py`):

- **Chunks.** The index hashes each prompt in 128-token chunks. A chunk hash
  covers the chunk and every chunk before it, the chain that vLLM uses for its
  block hashes, so one hash identifies a whole prefix. A LoRA adapter or a
  cache salt starts another chain. 128 tokens is the alignment of the DeltaNet
  snapshots, so a matched chunk is a prefix that a Qwen3.5 engine can reuse.
- **What it records.** For each engine: the chunks of each prompt routed to it,
  and the chunks of the tokens it generated for that prompt, as they arrive in
  the engine outputs. The generated tokens matter for the rollouts of one
  prompt: they share the prompt and differ only in what they generated.
- **Choice.** For a load-balanced request, the index finds the engines with the
  longest matching prefix. If only some engines hold it, the request goes to
  the least loaded of them, unless the load guard applies:
  - that engine carries more than `VLLM_NEURON_DP_PREFIX_AFFINITY_SLACK`
    requests (default `max_num_seqs // 2`, at least 1) above the least loaded
    engine, or
  - that engine has no free sequence slot (`max_num_seqs` requests) while the
    least loaded engine has one. A request that waits for a slot waits for a
    decode to finish, which costs more than prefilling its prefix again.

  Then, and when no engine or every engine holds the prefix, the request goes
  to the least loaded engine as before.
- **Unchanged.** An `X-data-parallel-rank` header still selects the engine,
  and the index records the prompt for that engine. Multimodal, prompt-embeds
  and pooling requests keep the load-balanced choice. MoE models keep vLLM's
  own routing (this patch handles non-MoE models only).
- **Capacity.** Each engine keeps about its KV cache capacity in chunks
  (the KV blocks from the engine handshake). The index evicts the least
  recently routed chunks first, and records the chunks of a prompt from the
  last to the first, so the first chunks of a prefix go last, as in the vLLM
  block pool.
- **Cost.** In the API server, 0.9 ms to hash a 64K-token prompt, 0.03 ms
  to match it over 8 engines, and 0.3 us per generated token (measured on a
  trn2.48xlarge host). No graph changes: turning it on
  or off does not change the compiled graphs or their cache.

Each API server keeps its own index. With more than one API server, a session
whose turns reach different API servers can lose its engine, so serve with
`--api-server-count 1` (ART's sampling server requires it anyway).

The balancer logs its counts at most every 60 s, when it routes a request:

```text
DP prefix affinity: 194 load-balanced requests, 181 with a prefix on some
engines, 178 sent to that engine (9319 matched tokens each), 3 sent to the
least loaded engine by the load guard
```

## Measured

Qwen3.8-27B, TP8 x DP2 on one trn2.48xlarge (the rollout profile of
[Full-vocabulary NKI sampling](../../guides/reference-configuration.md#full-vocabulary-nki-sampling)
with `--api-server-count 1`), 16 sessions at once: 4 tasks x 4 rollouts, 6
turns. A session starts with a 1,024-token system prompt shared by all
sessions and a 6,144-token task prompt shared by the 4 rollouts of its task.
Each turn generates 64 to 256 tokens and adds a 1,024-token tool output after a
tool delay of 0 to 4 s, both random and the same in every run of a seed. The
last turn reads about 13K tokens. Two seeds per row.

| Routing | Prefix cache hit rate | Time to first token, later turns, median / mean | Worst first-turn TTFT | Wall time |
| --- | --- | --- | --- | --- |
| Least loaded (off) | 77.7%, 77.6% | 0.43 / 0.54 s, 0.42 / 0.52 s | 3.8 s, 3.8 s | 60.8 s, 58.0 s |
| Prefix affinity, slack only | 85.4%, 84.6% | 0.37 / 0.47 s, 0.37 / 0.48 s | 5.5 s, 4.8 s | 54.6 s, 54.7 s |
| Prefix affinity, slack and free slot (the default rule) | 84.8%, 84.9% | 0.37 / 0.39 s, 0.37 / 0.40 s | 3.1 s, 3.1 s | 55.3 s, 53.2 s |

The hit rate counts the prompt tokens served from the prefix cache (vLLM's
`prefix_cache_hits` over `prefix_cache_queries`, both engines). A Qwen3.5 hit
is cut to the last DeltaNet snapshot, about 64 to 191 tokens before the end of
the prompt of the turn before, which bounds this workload below about 89%.

The slack-only rule let the 4 rollouts of a task share one engine at their
first turn and exceed its 8 slots, which raised the worst first-turn time. The
free-slot rule keeps the hit rate, and cuts the mean time to first token of
the later turns by about a quarter and the worst first-turn time by a fifth
against least loaded routing.

With 2 engines, a request that lands on the other engine still finds the
shared system and task prompts there, so the gain grows with the engine count
(about 1 turn in 8 finds its engine at DP8 without affinity). A replay without
tool delays and with equal lengths sends the turns in the same order every
time, and least loaded routing then keeps each session on one engine by
itself: 80.4% without affinity and 82.5% with it, a later-turn median time to
first token of 1.12 s and 1.41 s, a mean of 1.65 s and 1.55 s, and a wall time
of 35.1 s and 34.0 s (one run each).

## Potential optimizations (TODO)

1. **Routing across nodes.** The index sees the engines of one server. For
   rollouts over several nodes, either:
   - run one server per node behind an HTTP router that applies the same
     prefix rule across nodes (and pins the engine inside the node with
     `X-data-parallel-rank`), or
   - run one vLLM data-parallel deployment over the nodes
     (`--data-parallel-address`, `--headless` on the other nodes), so this
     index sees every engine. vLLM runs the DP engines of a non-MoE model
     independently, so no device collective crosses nodes. To validate: ART's
     weight refresh through `collective_rpc` on remote engines, its merged
     checkpoint path (`ART_WEIGHT_REFRESH_ROOT`) on storage that every node
     reads, the load on one API server, and a shared compile cache
     (`NEURON_LIBTORCH_REMOTE_CACHE`).
2. **DeltaNet snapshots during decode.** A snapshot is saved only in prefill,
   so the next turn of a session restores the state near the end of the
   prompt of the turn before, and prefills the generated tokens of that turn
   again (up to `max_tokens`). Saving a snapshot when decode crosses a
   128-token boundary, or when a request finishes, would let the next turn
   restore after the generated tokens. Cost: one state copy per 128 generated
   tokens per sequence, and snapshot rows.
3. **Follow the engine's cache instead of estimating it.** The index estimates
   each engine's cache from the requests it routed. vLLM KV cache events
   (`--kv-events-config`: block stored and removed) would give the exact
   blocks of each engine. At least, clear the index when the prefix cache is
   reset: ART's weight refresh resets every engine's cache
   (`pause_generation(clear_cache=True)`), and the index then still routes by
   prefixes that no engine holds (harmless for the load, no reuse).
4. **One index for all API servers.** Share the index in shared memory, as the
   balancer already shares the in-flight counts, so `--api-server-count`
   greater than 1 keeps full affinity.
5. **A session key from the caller.** When a caller knows the session (ART's
   rollout gateway knows the AgentCore session), it can pin the engine with
   `X-data-parallel-rank` from a hash of the session ID. That needs a change
   in the caller; the prefix rule needs none.
6. **MoE models.** The patch leaves MoE models on vLLM's routing. Their DP
   engines run in lockstep but still keep separate prefix caches, so the same
   index applies.
7. **Measure at scale, then choose the default.** Replay real agent traces
   (for example OfficeBench) on TP8 x DP8 with and without affinity: time to
   first token per turn, prefix cache hit rate, DeltaNet snapshot hits, and
   the balance across engines. If affinity is never worse, turn it on by
   default with prefix caching and DP.
8. **Restart without a recompile.** In the measurements above, each restart of
   the TP8 x DP2 rollout profile compiled every graph again (18 to 22 minutes),
   also on the same tree with the same flags. A rollout server restarts after a
   crash or a node change, so find which input changes the cache key and keep
   it stable.
