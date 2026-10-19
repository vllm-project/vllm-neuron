# GLM-5.3 on Neuron

[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (`GlmMoeDsaForCausalLM`)
on one trn2.48xlarge, text only, at up to 32K context.

| | |
|---|---|
| layers | 78: layers 0-2 dense FFN (12288), 75 MoE (+1 MTP layer, not loaded) |
| MoE | 256 routed experts of 2048, top-8, sigmoid + correction-bias routing, x2.5, 1 shared expert |
| attention | MLA, 64 heads, q_lora 2048, kv_lora 512, qk 192+64, v 256 |
| sparse attention | DSA, top-2048 keys; 21 "full" layers (0, 1, 2, 6, 10, ..., 74) run the indexer, 57 "shared" ones reuse it |
| vocab / hidden | 154880 / 6144 |

GLM-5.3 has the same attention, DSA and MoE as Tencent Hy4-preview, so it runs on
that port's code (`vllm_neuron/model/hy_v4`; see
[its README](../hy_v4/README.md) for the layout, the DSA implementation and the
compiler workarounds). What is GLM's own, in `vllm_neuron/model/glm_dsa`, is a plain
pre-norm residual (no iHC), no output gate and no attention sinks, the indexer's
RoPE on its *first* 64 dims, three dense layers, and the checkpoint names.

## Quick start

```bash
# once: FP8 (128 x 128 blocks) -> Trn2 fp8 (~8 min on a trn2.48xlarge)
python examples/vllm_neuron/models/glm_dsa/convert_fp8_block_to_neuron_fp8.py \
    --src <dir>/GLM-5.3 --dst <dir>/GLM-5.3-neuron-fp8 --workers 24

python examples/vllm_neuron/models/glm_dsa/run.py --model <dir>/GLM-5.3-neuron-fp8

# long context (max_model_len 32768, prefill in 2048-token chunks): a pass key
# hidden at the start of ~30K tokens of filler
python examples/vllm_neuron/models/glm_dsa/run.py --model <dir>/GLM-5.3-neuron-fp8 \
    --needle 30000
```

The environment, the compile times and the timeouts `run.py` raises are the Hy4
port's; see [its README](../hy_v4/README.md#environment).

## Why the checkpoint is converted

The published checkpoint stores every linear weight as `float8_e4m3fn` with one
fp32 scale per 128 x 128 tile. Trn2's fp8 tops out at 240, not e4m3fn's 448, so
`convert_fp8_block_to_neuron_fp8.py` rewrites it once, one shard per layer:

* the **routed experts** are requantised to one scale per output row inside the 240
  range and fused per layer into the layout the Hy4 port loads
  (`mlp.experts.gate_up_proj [E, 2I, H]`, `down_proj_t [E, I, H]`, per-row scales);
* **everything else** that is fp8 is dequantised to bf16;
* the MTP layer is dropped, and `config.json` gets
  `neuron_quantization.format = fp8_row_experts`, which the model checks for.

Requantisation error, block -> per-row, over 600 sampled expert tensors: median
2.64%, max 2.65% relative (Frobenius). 710 GiB on disk; per rank at world 64, ~14.4 GiB of each core's 24.

## Measured

trn2.48xlarge, world 64, `max_model_len` 32768, 2048-token prefill chunks, decode
buckets `[1, 4]` (`run.py`'s defaults), greedy. Cold start compiles the prefill graph
and both decode graphs in ~28 min.

`run.py`'s demo prompts, 48 output tokens:

```
'The capital of France is'  -> ' Paris. Distance from Paris to Paris is 0 km, while distance from Toulouse to Paris is approximately 590 km.\nCheapest flights to France...'
'I am gonna keep counting forever, 1 2 3 4 5' -> ' 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29'
'def fibonacci(n):'         -> ' \n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef main():...'
'Q: Natalia sold clips to 48 of her friends in April, ...\nA:' -> ' Natalia sold 48/2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May. The answer is 72.'
```

`--needle 30000` (28,782-token prompt, 15 prefill chunks): `' 71432. Remember it. 71432 is
the pass key. ...'`, found in 94.2 s end to end.

Throughput at a 31,744-token random prompt, 256 output tokens, greedy, median of 3
(each run within 0.02%). As for Hy4, only one 32K request fits the KV cache, so this
single stream is also the maximum at this length:

| TTFT | input tok/s | TPOT | output tok/s |
|---|---|---|---|
| 84.9 s | 374 | 230.2 ms | 4.34 |

Weights are 14.43 GiB of each core's 24; KV cache ~46K tokens at
`--gpu-memory-utilization 0.8`.

## Known limitations

* **Context is 32K.** As for Hy4, the latent cache is replicated on every rank, so
  the KV capacity (~46K tokens at `--gpu-memory-utilization 0.8`) is per rank.
* **The DSA attention is dense attention masked to the selection**: exact, but its
  cost grows with the cache span, not with the 2048 keys it keeps (Hy4 README,
  "Known limitations").
* No MTP, no prefix caching, no mixed prefill/decode batches.

## Correctness check (CPU, one layer vs transformers)

```bash
VLLM_NEURON_CPU_MODE=1 python examples/vllm_neuron/models/glm_dsa/check_layer_vs_hf.py \
    --model <dir>/GLM-5.3-neuron-fp8 --layer 6
```

Loads one layer through this port's weight loaders, runs a 2600-token prefill in
2048-token chunks (the second reads its prefix from the paged cache, and its rows
past 2048 attend to a DSA selection) and one paged decode step, and compares the
layer's update with `transformers`' `GlmMoeDsaDecoderLayer` in fp32 on the same
(dequantised) weights. transformers' GLM attention rotates interleaved pairs, like
the port, so unlike Hy4 no RoPE switch is needed.

Layers 0 (dense, full indexer) and 6 (MoE, full indexer) on the current code, chunk 1 /
later chunks / decode, per-token median of the layer's update: 0.53% / 0.63% / 0.95%
(layer 0) and 0.41% / 0.59% / 0.59% (layer 6). The few MoE-layer tokens above 5% are the
expert near-ties described in the Hy4 README.
