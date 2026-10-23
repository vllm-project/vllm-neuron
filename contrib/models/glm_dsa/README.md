# GLM-5.3 on Trn2

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
that port's code: `src/hy_v4/`, a copy of `contrib/models/hy4_preview/src/` (see
[its README](../hy4_preview/README.md) for the layout, the DSA implementation and
the compiler workarounds). What is GLM's own, in `src/model.py` and
`src/config.py`, is a plain pre-norm residual (no iHC), no output gate and no
attention sinks, the indexer's RoPE on its *first* 64 dims, three dense layers,
and the checkpoint names.

> **Note:** a community contribution. See the compatibility matrix for exactly
> what was tested.

## Usage

Use the Neuron SDK 2.33 vLLM environment (vllm-neuron 0.29.0.1.2.0, vLLM 0.29,
neuronx-cc 2.28). Then:

```bash
# 1. Register the architecture: installs src/ as the ``glm_dsa`` package with a
#    ``vllm.general_plugins`` entry point, so every process vLLM starts has it.
#    (vLLM's own GPU class for GlmMoeDsaForCausalLM is replaced.)
pip install -e contrib/models/glm_dsa

# 2. Once: FP8 (128 x 128 blocks) -> Trn2 fp8 (~8 min on a trn2.48xlarge)
python contrib/models/glm_dsa/scripts/convert_fp8_block_to_neuron_fp8.py \
    --src <dir>/GLM-5.3 --dst <dir>/GLM-5.3-neuron-fp8 --workers 24

# 3. The native PyTorch route, with fewer parallel compiles per rank (see
#    "Known limitations")
export VLLM_NEURON_BACKEND=neuron_native VLLM_NEURON_PARALLEL_COMPILE_WORKERS=2
```

### Serve with the OpenAI-compatible server

```bash
vllm serve <dir>/GLM-5.3-neuron-fp8 \
    --tensor-parallel-size 64 --max-model-len 32768 \
    --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.8 --no-enable-prefix-caching \
    --distributed-timeout-seconds 14400 \
    --additional-config '{"neuron_config": {"quantization": "bf16",
        "num_batched_tokens_buckets": [2048], "num_seqs_buckets": [1, 4],
        "on_device_sampling_config": {"all_greedy": true}}}'
```

### Offline inference

```bash
python contrib/models/glm_dsa/scripts/run.py --model <dir>/GLM-5.3-neuron-fp8
# a pass key hidden at the start of ~30K tokens of filler
python contrib/models/glm_dsa/scripts/run.py --model <dir>/GLM-5.3-neuron-fp8 --needle 30000
# TTFT / TPOT at 31,744 in / 256 out, one request, median of 3
python contrib/models/glm_dsa/scripts/run.py --model <dir>/GLM-5.3-neuron-fp8 --bench
```

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29.0.1.2.0) | 2.27 (vllm-neuron 0.24) | earlier |
| ---------------------- | ------------------------------- | ----------------------- | ---------- |
| Trn3                   | Not tested                      | Not tested              | Not tested |
| Trn2 (trn2.48xlarge)   | Working                         | Working (previous port) | Not tested |

## Example Checkpoints

- <https://huggingface.co/zai-org/GLM-5.3> (FP8, 128 x 128 blocks; convert with
  `scripts/convert_fp8_block_to_neuron_fp8.py`)

## Testing

```bash
pip install -e contrib/models/glm_dsa
GLM_MODEL=<dir>/GLM-5.3-neuron-fp8 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/glm_dsa/test/integration/test_model.py -v -s --timeout=7200
```

The integration test checks greedy generations against the recorded outputs
below and the ~30K-token pass-key retrieval; the template's logit validation
would need the whole model in fp32 on the host.

### Optional: one-layer check against transformers (CPU)

```bash
VLLM_NEURON_CPU_MODE=1 python contrib/models/glm_dsa/scripts/check_layer_vs_hf.py \
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

## Measured

trn2.48xlarge, world 64, `max_model_len` 32768, 2048-token prefill chunks, decode
buckets `[1, 4]` (`run.py`'s defaults), greedy, XLA compile route. A cold start
compiles the prefill graph and both decode graphs in ~30 min.

Throughput at a 31,744-token random prompt, 256 output tokens, one request, median
of 3 (`run.py --bench`; each run within 0.01%). As for Hy4, only one 32K request fits
the KV cache, so this single stream is also the maximum at this length:

| stack | TTFT | input tok/s | TPOT | output tok/s |
|---|---|---|---|---|
| SDK 2.33, vllm-neuron 0.29 (this folder) | 75.3 s | 422 | 182.5 ms | 5.48 |
| SDK 2.27, vllm-neuron 0.24 (PR #61) | 84.9 s | 374 | 230.2 ms | 4.34 |

`--needle 30000` (28,782-token prompt, 15 prefill chunks): found, 76.3 s end to end
on SDK 2.33 (94.2 s on 2.27).

`run.py`'s demo prompts on SDK 2.33. Three match 2.27 token for token; the France
continuation parts after `' Distance from'` (2.27: `' Paris to Paris is 0 km, while
distance from Toulouse to Paris is approximately 590 km.'`):

```
'The capital of France is'  -> ' Paris. Distance from London to Paris is 343 km, while direct flight time is 1 h 5 min. According to Yandex.Za rule'
'I am gonna keep counting forever, 1 2 3 4 5' -> ' 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21'
'def fibonacci(n):'         -> ' \n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef main():\n   '
'Q: Natalia sold clips to 48 of her friends in April, ...\nA:' -> ' Natalia sold 48/2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May.'
```

Weights are 14.43 GiB of each core's 24; KV cache ~46K tokens at
`--gpu-memory-utilization 0.8`.

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

## Known limitations

* **Context is 32K.** As for Hy4, the latent cache is replicated on every rank, so
  the KV capacity (~46K tokens at `--gpu-memory-utilization 0.8`) is per rank.
* **The DSA attention is dense attention masked to the selection**: exact, but its
  cost grows with the cache span, not with the 2048 keys it keeps (`contrib/models/hy4_preview/README.md`,
  "Known limitations").
* No MTP, no prefix caching, no mixed prefill/decode batches.
* **Native PyTorch route (`VLLM_NEURON_BACKEND=neuron_native`).** On it the decode
  graphs compile once per rank, and vllm-neuron's default of 8 parallel
  neuronx-cc jobs per rank ran the host out of memory compiling the 32K bench
  graphs (`[F137] neuronx-cc was forcibly killed`); cap it with
  `VLLM_NEURON_PARALLEL_COMPILE_WORKERS=2`. Re-validated on the native route
  (SDK 2.33): the 30K needle passes (29.8 s, against 76.3 s on the XLA
  route) and the integration test passes. The "Measured" throughput above is
  from the XLA route (`VLLM_NEURON_BACKEND=vllm_neuron`, deprecated); the native
  32K bench has not been rerun with the cap yet.
* `src/hy_v4/` is a copy of `contrib/models/hy4_preview/src/` (the contrib rules
  forbid importing another model's folder); a fix to one belongs in both.
* No disaggregated (1P1D) scripts yet: Hy4's `Hy4NixlConnector` (a key-only
  cache, one NIXL region per layer) applies unchanged, but has not been run here.
