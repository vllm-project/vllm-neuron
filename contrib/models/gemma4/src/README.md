# Gemma 4 on vLLM Neuron: implementation notes and a debugging story

How the model is built, and why decode time grew with the size of the KV cache before it was fixed. The second half is a checklist for
porting other models that mix layer shapes. All numbers are from a `trn2.3xlarge` (TP=4, LNC=2, BF16) with vLLM Neuron 0.29.0.1.2.0.

## Files

| File | What it holds |
|---|---|
| `__init__.py` | `register()`: registers the factory with vLLM's `ModelRegistry` and adds it to vLLM Neuron's model list |
| `factory.py` | Registry entry for `Gemma4ForConditionalGeneration` and `Gemma4ForCausalLM`; rejects non-BF16 quantization |
| `config.py` | `Gemma4Config`, parsed from the HF `text_config` (and the global-layer fields that transformers 5.17 drops) |
| `model.py` | Text decoder, KV-cache spec and binding, weight loading |
| `kv_write.py` | NKI kernels that write new K/V rows into, and gather blocks from, the paged cache |

## Architecture notes

- Layers repeat five sliding-window layers (head_dim 256, 16 KV heads, 1024-token window) and one global layer (head_dim 512, 4 KV
  heads, V = K, partial RoPE). Both head dims are above the 128 accepted by the plugin's NKI attention kernels (`MAX_HEAD_DIM` in
  `functional/attention/attention_cte.py`), so attention is PyTorch over the gathered blocks, with float32 scores.
- At TP=4 the MLP intermediate size per rank is 5376. The prefill MLP kernel requires `ceil(I / 512) <= 8`, so prefill splits I into two
  chunks and sums the partial down projections (the sum is exact because GeGLU is elementwise over I). Decode calls the kernel on the full
  width.
- transformers 5.17 drops `global_head_dim` and `num_global_key_value_heads` while parsing the Gemma 4 config. `config.py` reads them back
  from `config.json`.
- `model.py` also contains a per-layer input embedding path (PLE) that the 31B checkpoint does not use (`hidden_size_per_layer_input` is 0).
  It is not tested.

## The KV cache: one flat tensor per shared buffer

vLLM's hybrid KV-cache manager put the sliding and global layers on one buffer in our runs. The runner binds each layer's K and V as views
of that buffer, so the two layer types see the same bytes with different shapes (`[N, 4, 32, 256]` and `[N, 1, 64, 512]`, for blocks of 32
and 64 tokens). The model therefore:

1. builds **one** flat tensor per shared buffer, keyed on the bytes of the buffer and not on the layer's own shape, so every layer on the
   buffer gets the same object (`_flat_kv`);
2. keeps **no** other reference to the per-layer views (`bind_kv_cache` does not store `k_cache` or `v_cache`), so the compiler is never
   given overlapping inputs;
3. writes new K/V rows with an NKI kernel (one indirect DMA per tile of 128 rows; an out-of-range row index writes nothing) on the flat
   tensor, and reads blocks with a gather kernel; prefill reads through views derived inside the graph;
4. maps padded slots (`-1`) to the null block with `paged_slot_indices`. A plain `slot // block_size` sends slot -1 to the last block of
   the cache.

In one experiment, replacing the gather kernel with a static slice of the cache in torch (in a graph that also writes the cache through
the kernel) took a 6-layer build from 12.8 to 105.7 ms per token. The decode path therefore reads the cache only through kernels.

## Debugging story: decode time grew with the KV cache size

**Symptom.** The full model gave correct answers but needed about 840 ms per decode token. A host profile put the time inside the compiled
graph. On a 6-layer build the time followed the number of KV blocks: 99.2 ms at 1300 blocks and 27.2 ms at 450. Removing the cache write
brought it to 12.7 ms.

**Hypotheses that were tested and did not explain it:**

| Hypothesis | Test | Result |
|---|---|---|
| Attention math (GQA expansion, float32) | grouped attention without GQA expansion | 100.7 to 98.6 ms |
| The block gather | static slice instead of the gather | worse: 251 ms |
| `index_put_` lowers badly | `index_copy_`, or `index_put_` with one index | 99.6 ms and 160 ms |
| A custom NKI write kernel fixes it | indirect-DMA row write, bit-exact against `index_put_` | 93.5 ms |
| Reads must be ordered after the write | read the original cache, ordered with a token | 86.2 ms |
| Passing the whole `[2, ...]` tensor, or a flat one | whole or flat operand | write-back merges went from 12 to 6 per step, not to 0 |

**What it was.** The device profile showed HBM traffic dominated by twelve ops of the form `select(mask, pad(x), base)`, about 2 GB each
(6 layers times K and V). With `TORCH_LOGS=aot_graphs`, each cache write was followed by `pad`, `arange(<size of the whole buffer>)`, a
comparison and a `where`. AOTAutograd merged the two overlapping views into one synthetic base and regenerated each view from it (see the
CPU reproduction below). Every write through a view became an `as_strided_scatter` into the base, which lowered to that
`pad + arange + where` over the entire buffer.

A plain-PyTorch reproduction on CPU (`aot_function` with `keep_inference_input_mutations=True`):

| Inputs | Graph arguments | Write-back ops |
|---|---|---|
| two non-overlapping views, same shape | 2 | `copy_` |
| two overlapping views, different shapes | 1 (merged) | `as_strided_scatter`, `scatter` |
| one input (view or not) | 1 | `copy_` |

A reference model on the same stack (Llama 3.2 1B, which updates its cache inside the attention kernel) had no buffer-sized
`arange` or `where` in any of its graphs, and its graphs ended with direct `copy_` into separate K and V arguments.

**Result** (6-layer build, 3 repetitions each, greedy-token hash unchanged):

| KV blocks | before | after |
|---|---|---|
| 1300 | 99.2 ms/token | 12.8 ms/token |
| 450 | 27.2 ms/token | 12.7 ms/token |

Between 450 and 1300 blocks the time after the change was the same.

## Checklist for the next port with layers of different shapes

1. Time a short decode at two `--num-gpu-blocks-override` values. If the time follows the block count, the cause is not the attention math.
2. Compile with `TORCH_LOGS=aot_graphs` and count `arange(<buffer size>)` and `where` ops in the decode and prefill graphs. Classify graphs
   by what they contain (write and gather kernels present), not by order: the order changed between runs. Also count the KV graph
   arguments; here the graph had one flat KV argument of 85,196,800 elements.
3. In a device profile, group HBM bytes by HLO op. In our case a few buffer-sized `select` and `pad` ops dominated.
4. Make sure every layer that shares a buffer receives the same tensor object, and that no other view of that storage is reachable from
   the module.
5. Gate every optimization on a greedy-token hash of a fixed prompt.
