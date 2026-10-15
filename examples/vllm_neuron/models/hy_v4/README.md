# Tencent Hy4-preview on Neuron

[`tencent/Hy4-preview`](https://huggingface.co/tencent/Hy4-preview)
(`HYV4ForCausalLM`, 770B total / 49B active) on one trn2.48xlarge, text only.

| | |
|---|---|
| layers | 78: layer 0 dense FFN (18432), 77 MoE |
| MoE | 256 routed experts of 2048, top-8, sigmoid + correction-bias routing, x2.827, SwiGLU clamp 10, 1 shared expert |
| attention | gated MLA, 64 heads, q_lora 2048, kv_lora 512, qk 192+64, v 256, learnable sink per head |
| sparse attention | DSA, top-2048 keys, indexer shared across groups of 4 layers |
| residual | iHC, 4 streams |
| vocab / hidden | 120832 / 6144 |

## Quick start

```bash
# once: MXFP8 -> Trn2 fp8 (~7 min with 24 workers on a trn2.48xlarge)
python examples/vllm_neuron/models/hy_v4/convert_mxfp8_to_neuron_fp8.py \
    --src <dir>/Hy4-preview-FP8 --dst <dir>/Hy4-preview-neuron-fp8 --workers 24

python examples/vllm_neuron/models/hy_v4/run.py --model <dir>/Hy4-preview-neuron-fp8
```

## Environment

The plugin's `pip install -e .` plus, separately, the compiler packages, with
**`islpy` pinned**:

```bash
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "neuronx-cc==2.27.*" "nki==0.6.0" "islpy==2026.1"
```

`neuronx-cc` 2.27 asks for `islpy~=2026.1`, which pip resolves to 2026.2.x; with
that every compile dies in the Tensorizer with `[NCC_ISMP902] Simplifier error:
is_subset(): incompatible function arguments`. The import-time warning about
`libtorch_neuronx_lite/_compiler/_C.so: undefined symbol
PyObject_ClearManagedDict` (its vendored compiler is built for cp313) is
harmless on Python 3.12.

The compile cache key does not cover this model's source, so after editing
`vllm_neuron/model/hy_v4/` clear `$VLLM_CACHE_ROOT` (or point it elsewhere) --
otherwise the previous trace is compiled again.

## Measured

trn2.48xlarge, world 64, `max_model_len` 2048, 4 sequences, greedy, default
(interleaved) RoPE:

* weights load in ~55 s per rank; KV cache 53,824 tokens at
  `--gpu-memory-utilization 0.8`;
* engine init 335 s, of which 329 s is compiling the two graphs (cold cache);
* `run.py`'s demo prompts, 32 tokens:

```
'The capital of France is'  -> ' __________.\nA. Sydney\nB. New York\nC. Paris\nD. Ottawa 【解析】句意：法国的首都是巴黎。Sydney'
'I am gonna keep counting forever, 1 2 3 4 5' -> ' 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21'
'def fibonacci(n):'         -> '\n    """Return the nth value in the Fibonacci series."""\n    if n == 0:\n        return 0\n    elif n == 1:\n'
'Once upon a time, there was a' -> ' king who was very rich. He had a lot of gold and silver, but he was not happy. He wanted to be the happiest man in the world'
```

No latency or throughput numbers yet; this is functional validation.

## Why the fp8 checkpoint, and why converted

BF16 is 1.56 TB; a trn2.48xlarge has 64 x 24 GB = 1.5 TB of HBM, partitioned per
logical core. The published `Hy4-preview-FP8` is ModelOpt **MXFP8** (one e8m0
exponent per 32 elements), which has no Trn2 kernel, and its values use the OCP
e4m3fn range (448) where Trn2's fp8 tops out at 240.
`convert_mxfp8_to_neuron_fp8.py` therefore:

* requantises the **routed experts** to fp8 with one scale per output row, inside
  the 240 range (`down_proj` also stored transposed as `down_proj_t [E, I, H]`
  so a TP shard is a contiguous read);
* dequantises **everything else** to bf16, which is lossless (e4m3 x 2^k fits
  bf16 exactly);
* drops `quantization_config` from `config.json` and adds
  `neuron_quantization`, which the model checks for.

Requantisation error, MX -> per-row, measured by the converter over 156 expert
tensors: **median 2.56%, max 2.70%** relative (Frobenius). That is on top of the
MXFP8 quantisation itself.

Per rank at world 64: ~11.6 GiB of fp8 experts + ~3.2 GiB bf16 (q_a/kv_a are
replicated) ~= 15 GiB.

## Layout

**Pure TP over 64 ranks.** One attention head per rank. Every rank holds all
256 experts and 32 of each one's 2048 intermediate columns. The MoE runs in
torch, dense over the 256 local experts with zero affinity on the unselected
ones -- exact, one prefill and one decode graph, and independent of whether
Trn2's MoE kernels accept fp8 weights (`NF.moe_tkg` rejects `float8` today).
The kernel path (`moe_cte` shard_on_block FP8 / `moe_tkg`) needs
`2048 / tp_degree` to be a multiple of 128, so `ep_degree >= 4`; not done yet.

**MLA in latent space.** The cache stores the normed 512-wide latent plus the
64-wide rope key per token, as the K and V halves of a 288-wide single head
(576 = 2 x 288, so a page is exactly what the planner reserved): 1152 B per token
per layer, replicated on every rank. `kv_b_proj` is absorbed into the query and
the output.

## Known limitations

* **Context <= 2048 tokens.** Past `index_topk` = 2048, Hy4 attends to the keys
  the DSA indexer picks. Up to it the selection is every key, so dense attention
  is exact; the model raises on a longer `max_model_len` rather than silently
  diverging. The indexer (and its cache) is the next piece.
* **RoPE pairing: interleaved, and transformers disagrees.** vLLM's and SGLang's
  Hy4 ports rotate interleaved pairs; transformers' `hy_v4` rotates halves.
  Measured on device with the same weights, `HY4_ROPE_STYLE=neox` degrades
  within a few tokens -- `'... 1 2 3 4 5'` continues `' 6 7 8 9 10 13 20 5 6 8
  ...'`, `def fibonacci(n):` comes out broken -- while interleaved (the default)
  gives the outputs above. So the checkpoint wants interleaved. The CPU check
  below still runs `neox`, because it compares against transformers.
* No MTP (the `mtp_layers.0` tensors are ignored), no prefix caching, no
  chunked/mixed prefill (one sequence per prefill, as for Qwen3.5).
  `run.py` sets `max_num_batched_tokens = max_model_len`, which is what turns
  chunked prefill off: the prefill attends only within the tokens it is given.
* vLLM 0.24 has no class for `HYV4ForCausalLM`, so the plugin registers it from
  a `vllm.general_plugins` entry point (`vllm_neuron:register_models`); reinstall
  with `pip install -e .` so that entry point exists.

## Correctness check (CPU, one layer vs transformers)

```bash
HY4_ROPE_STYLE=neox VLLM_NEURON_CPU_MODE=1 python examples/vllm_neuron/models/hy_v4/check_layer_vs_hf.py \
    --model <dir>/Hy4-preview-neuron-fp8 --layer 1
```

Loads one layer through this port's weight loaders, runs a 40-token prefill and
one paged decode step, and compares the layer's update against
`transformers.models.hy_v4.HYV4DecoderLayer` in fp32 on the same (dequantised)
weights:

| layer | kind | prefill, per-token median | decode |
|---|---|---|---|
| 0 | dense, full indexer | 0.43% | 0.41% |
| 1 | MoE, full indexer | 0.42% (tokens 26, 32 flip an expert) | 0.40% |
| 2 | MoE, shared indexer | 0.42% | 0.42% |
| 5 | MoE, full indexer | 0.49% (token 23 flips) | 0.50% |
| 77 | MoE, full indexer | 0.66% (tokens 1, 29 flip) | 0.84% |

The attention sub-block alone agrees to 0.5%. The few outlier tokens appear
only in MoE layers, only in prefill, and at 20-45% rather than drifting, which
is what an 8th/9th-expert near-tie resolving differently in bf16 and fp32 looks
like; that explanation is inferred, not yet checked expert by expert.
