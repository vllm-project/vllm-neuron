# Qwen3.8 on vLLM Neuron: implementation notes

How the model is put together on the plugin's runner, and the problems that had to be solved on the chip. All
numbers are from a `trn2.3xlarge` (TP=8, LNC=1, BF16) with vLLM Neuron 0.29.0.1.2.0 and neuronx-cc 2.28.7631.

## Files

| File | What it holds |
|---|---|
| `__init__.py` | `register()`: registers the factory with vLLM's `ModelRegistry` and adds it to vLLM Neuron's model list |
| `factory.py` | Registry entry for `Qwen3_5ForConditionalGeneration`; refuses non-BF16 weights, the prefix cache and an optimization level other than 1 |
| `config.py` | `Qwen38Config`, parsed from the HF `text_config`, with what each rank holds and every geometry the kernels do not serve |
| `model.py` | The decoder over the kernels, the KV-cache spec and binding, the weight loading |
| `weights.py` | The checkpoint tensors in the kernels' layouts |
| `state.py` | Where the GDN state lives in the KV cache and how a forward reads and writes it |
| `rows.py` | NKI kernels that gather rows from, and write rows into, a cache in place (indirect DMA) |
| `kernels/` | The Qwen3.5 per-layer kernels, copied unchanged (`kernels/README.md`) |

## One decoder layer

Each layer is the input norm, one fused kernel for the mixer, the post-attention norm and the MLP. The GDN layers
run `gdn_decode_tkg` (one token per row) or `gdn_prefill_tkg` (one request's chunk): in_proj, the causal
convolution and SiLU, l2norm and gates, the gated delta rule, the gated RMSNorm and this rank's slice of out_proj.
The attention layers run `attn_decode_tkg` or `attn_prefill_tkg`: QKV and gate projections, zero-centred QK norms,
partial RoPE, the K/V write into the paged cache, attention, the sigmoid gate and this rank's slice of o_proj. Every
kernel returns this rank's partial sum of the layer output, followed by an all-reduce, as the MLP is.

At TP=8 a rank holds 6 GDN value heads, 2 GDN key heads, 3 attention query heads (the kernels take an even count,
so a zero fourth head is added whose o_proj rows are zero) and the KV head its query heads use.

## The attention KV cache

The kernels read pages `k_cache_t [N, d, block]` (K transposed per page) and `v_cache [N, block, d]`. The runner
gives each attention layer two planes `[N, 1, block, d]`; the model reinterprets them once, outside the graph, in the
kernels' layout, keeps no other view of them, and passes the block table as the page table (null entries, `-1`, read
page 0, which is finite and masked).

## The GDN state lives in the KV cache

The runner has no recurrent-state cache (it routes Mamba-style specs away), and a model-owned state buffer would be
invisible to vLLM's memory budget. So the model declares two more KV layers, `state.rec` (the fp32 temporal state,
`[128 k, vh, 128 v]` per layer and request) and `state.conv` (the bf16 convolution window, `[Cc, 3]`), with head
sizes chosen so that each of their two planes holds half of the 48 GDN layers' records for one block. vLLM puts
layers of one spec type with different head sizes into a single cache group (`UniformTypeKVCacheSpecs`), so the
state layers share the attention layers' block table: every block a request owns also owns a state region, and the
region of the request's **first** block holds its state. With the prefix cache off, no two live requests share a
block. vLLM budgets the memory and frees it with the request. Block 0 is vLLM's null block and never a request's
first block: padded decode rows read their state from it, and their writes are skipped, so it stays zero.

A forward gathers the batch's records of all layers once (four indirect-DMA gathers, one per plane), the layers
read their rows of that bank, and the advanced records are written back once at the end (four indirect-DMA
writes). A prefill whose request has no computed tokens yet starts from zero state.

## Memory per rank

One NeuronCore at LNC 1 has 12 GiB. With `--gpu-memory-utilization 0.90` and
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.22` the runner gives the KV cache 2.38 GiB after 6.42 GiB of weights (vLLM
log). What one block costs on one rank, from the layer geometry (16 attention layers, one KV head of 256; 48 GDN
layers, 6 value heads of 128 and 1280 conv channels per rank):

| Block size | Attention K/V | GDN temporal state | GDN conv window | Total per block | Per token | Blocks in 2.38 GiB | Tokens (vLLM log) |
|---|---|---|---|---|---|---|---|
| 1024 | 16 MiB | 18 MiB | 0.35 MiB | 34.4 MiB | 34.4 KiB | 70 | 71,680 |
| 2048 | 32 MiB | 18 MiB | 0.35 MiB | 50.4 MiB | 25.2 KiB | 48 | 98,304 |

Every block carries a state region, though only a request's first block uses it: 53 percent of the cache at block
size 1024 and 36 percent at 2048. A layout with one state region per running request instead of per block would leave
the cache to K/V; that is computed here, not built or measured.

## Problems found on the chip

| Symptom | Cause | Fix |
|---|---|---|
| `Matmul moving free dimension 2048 exceeds max 512` in the GDN prefill | the kernel's widest matmul defaults to a Trainium3 size | `QWEN35_GDNP_WIDE=512`, as sglang-neuron sets it (`kernels/__init__.py`) |
| `aot_autograd() does not yet handle input mutations on views with different dtypes` | the fp32 state plane was a dtype view of the runner's bf16 buffer | the planes stay bf16; the row kernels reinterpret them as fp32 inside NKI (`rows.py`) |
| `NCC_IIOT902 InsertOffloadedTransposes` in neuronx-cc | a `bitcast-convert` of the state rows inside the graph (the module that failed held one) | same fix: no dtype reinterpretation in torch |
| decode after a short prompt disagreed with Hugging Face | the runner pads a prefill chunk with slot 0, not -1, so the GDN kernel took the padding as real tokens | the chunk's real rows are the slots greater than 0 |
| prompts past 512 tokens disagreed from the first generated token | the prefill graphs at the runner's default `-O2` (the 2048-token bucket; the 512-token bucket agreed) | `--optimization-level 1`; the cause in the compiler was not investigated |
| four clients ran as two | at block size 512 the cache held 2.88 requests of 16k, because each block carries a state region | block size 1024 (4.38 requests) |

The model also declares `supports_async_reorder` and consumes the runner's in-graph reorder arguments, so
asynchronous scheduling assembles the decode input ids on the device.
