# Contrib Model: Inkling 2 Mini (text)

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (architecture
`Inkling2LoopForConditionalGeneration`) is a looped mixture-of-experts transformer: 24 decoder layers, of which the
middle 12 (layers 6 to 17) run three times with shared weights, so a token visits 48 layers. Layers 5, 11, 17 and 23
attend the whole context; the other 20 attend a 1024-token sliding window. There is no RoPE: position comes from a
learned, input-dependent relative-position bias over distances 0 to 1023, short causal convolutions on the keys,
values and sublayer outputs, and a log scaling of the queries in the full-attention layers. Each layer also has
per-token attention sinks, a sigmoid output gate, attention residuals across depth (learned queries over block
states), and an MoE of 128 experts (top 8) with a softcapped SwiGLU. From the second pass on, each core layer's
inputs are gated with messages read from the previous pass.

This folder serves the text decoder with vLLM on AWS Trainium2 through the `vllm-neuron` plugin, on one
`trn2.3xlarge` (TP=8, one rank per physical NeuronCore). The checkpoint stores the experts in NVFP4 (layers 0 to 19)
and MXFP8 (20 to 23); this model dequantizes them to bf16 when it loads them, as Hugging Face transformers does, and
runs bf16 activations. The vision tower and the speculative-decoding draft are not loaded.

> **Note:** Contrib models are community contributions and may have limited testing or performance optimization
> compared to the officially-supported models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Use a Neuron environment with `vllm-neuron` 0.29.0.1.2.0, for example the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`, then
install this model. The install adds a `vllm.general_plugins` entry point that registers the architecture (and the
`inkling2_loop` config class, which the image's transformers 5.17 predates) in every vLLM process:

```bash
pip install -e contrib/models/inkling2
```

The checkpoint declares ModelOpt NVFP4 quantization, which vLLM would try to apply. Serve a view of it without the
quantization metadata (symlinks; the original directory is not modified):

```bash
contrib/models/inkling2/examples/prepare_checkpoint.sh /models/Inkling-2-Mini /models/Inkling-2-Mini-bf16
```

Serve at logical NeuronCore config 1, one rank per physical core:

```bash
export NEURON_LOGICAL_NC_CONFIG=1 NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.10
vllm serve /models/Inkling-2-Mini-bf16 --served-model-name inkling \
  --tensor-parallel-size 8 --max-model-len 16384 --max-num-seqs 2 --max-num-batched-tokens 256 \
  --block-size 256 --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256], "num_seqs_buckets": [2],
                        "on_device_sampling_config": {"all_greedy": true}}}'
```

- `--no-enable-prefix-caching` is required (`factory.py` refuses the launch otherwise): the convolution windows live
  in the request's first KV block, which two requests sharing a cached prefix would share.
- Do not pass `--optimization-level`: the model then compiles its decode graph at `-O3`
  (`Inkling2ForCausalLM.compiler_args_for`), and the prefill graph at the runner's base level. An explicit level
  applies to every graph (measured: 42.97 ms a token with `--optimization-level 1`, 31.27 ms with the decode graph at
  `-O3`).
- `num_seqs_buckets: [2]` runs a single request padded to two rows, which decodes faster than the one-row graph
  (31.3 against 61.6 ms a token, see Results).
- The KV cap of 0.10 (1.08 GiB a core, 38,400 tokens) is what fits beside the weights (6.17 GiB a core) and the
  compiled graphs' scratchpad (4.0 GB); see Limitations.
- The first launch compiles the prefill graph and the decode graphs (about 25 minutes on a trn2.3xlarge); a relaunch
  with the same flags reuses the compile cache.

## How it is built

- `src/ops.py`: the arithmetic of one TP rank in plain torch (layers, the 3-pass loop schedule, convolutions with their
  windows, relative bias, sinks, depth attentions, message gates, the MoE), with the caches behind a small interface.
  The unit tests run it against Hugging Face's implementation.
- `src/model.py`: the vLLM side. Every layer visit has its own cache layer (24 + 2 x 12 = 48), one KV head of 128 a
  rank. The 20 sliding-window layers read only the blocks of their window. The short-convolution windows (3 tokens
  for each of 4 convolutions per cache layer) are an extra KV layer, `state.conv`, addressed by the request's first
  block and read and written one block row a request.
- `src/weights.py`: this rank's share of each checkpoint tensor. Attention splits by head (3 query heads and 1 KV
  head a rank); the MoE keeps all 128 experts on every rank and splits their intermediate dim (144 of 1152 a rank),
  so a decode step reads only each token's 8 experts. NVFP4 / MXFP8 dequantization and the `w13` gate/up
  de-interleave follow Hugging Face's load ops.
- The relative-position bias is built with the Transformer-XL relative shift (pad and reshape) rather than an
  element gather, which on Neuron is an indirect copy per element.

## Results

One trn2.3xlarge, TP=8, bf16, the configuration above, greedy.

**Accuracy.**
- Teacher-forced against Hugging Face on CPU (transformers 5.24, experts dequantized to bf16), 24 steps on each of
  four prompts of 5, 759, 2021 and 4863 tokens (the last two past the sliding window): 94 of 96 steps with the same
  argmax. At each of the 2 others the server's token is Hugging Face's second choice, with a top-1 to top-2 gap of
  0.000 and 0.105.
- GSM8K, first 100 questions, five-shot completions at temperature 0: 0.880, none unparsed.
- A code hidden at 10, 50 and 90 percent of filler text is answered exactly at 2147 and at 12122 tokens of prompt
  (6 of 6).

**Speed**, `vllm bench serve` with random prompts of 1024 tokens and 1024 output tokens (`--ignore-eos`):

| Clients | TTFT p50 | TPOT p50 | Output tokens/s |
| --- | --- | --- | --- |
| 1 | 2.09 s | 31.3 ms | 30.0 |
| 2 | 3.13 s | 32.1 ms | 57.0 |

With a one-row decode graph (`num_seqs_buckets: [1, 2]`) a single client decodes at 61.6 ms a token (15.7 tokens/s).
With `--optimization-level 1` (prefill and decode at `-O1`): TTFT 1.61 s, TPOT 43.0 ms at one client.

**Steadiness.** 64 requests at 2 clients with random lengths (410 to about 7800 prompt tokens, 51 to 972 output
tokens): 64 completed, none failed; TTFT p50 7.85 s, TPOT p50 40.3 ms, p99 115.6 ms (chunks of other requests' prefills run between
decode steps).

## Limitations

- Two requests at a 16k context. A 512-token prefill chunk, more rows or a larger KV pool do not leave room to load
  the graphs: the largest graph's scratchpad is 4.0 GB a core. Not investigated further.
- The cache layers of the sliding-window layers keep the whole context although they read only the window; declaring
  them as sliding-window layers would cut the KV per token about 6 times (longer contexts).
- The decoder is torch. `src/kernels/moe_decode.py` runs the decode MoE on nkilib's `moe_tkg` with Inkling 2's
  activation (opt-in, `INKLING2_NKI_MOE=1`); inside the 48-visit decode graph it measured slower (35.9 against 31.3
  ms a token), so it is off by default.
- The one-row decode graph is slower than the two-row one (61.6 against 31.3 ms a token); not explained. The
  configuration above avoids it.
- The prefill graph compiles at the runner's base level (`-O2` without `--optimization-level`); at `-O1` the TTFT is
  1.61 s instead of 2.09 s, but the runner applies a level to the prefill graph only through the flag, which also
  sets the decode graph.
- Not served: images, the DSpark speculative decoding draft, prefix caching, sampling other than greedy was not
  tested.

## Compatibility Matrix

| Instance / SDK version | 2.33       | 2.32 and earlier |
| ---------------------- | ---------- | ---------------- |
| Trn3                   | Not tested | Not tested       |
| Trn2                   | Working    | Not tested       |

## Example Checkpoints

- <https://huggingface.co/thinkingmachines/Inkling-2-Mini>

## Testing

The integration test serves the checkpoint view and checks a factual answer, greedy determinism, batch invariance,
and two continuations checked against Hugging Face, inside and past the sliding window:

```bash
INKLING2_MODEL_PATH=/models/Inkling-2-Mini-bf16 NEURON_LOGICAL_NC_CONFIG=1 NEURON_SKIP_EFA_AFFINITY=1 \
    pytest contrib/models/inkling2/test/integration/test_model.py -v -s
```

The unit tests run on CPU and need a transformers with `inkling2_loop` (5.24 or later) for the reference:

```bash
pytest contrib/models/inkling2/test/unit/
```

- `test_ops_vs_hf.py`: `ops.py` against `Inkling2LoopForCausalLM` on a small random configuration with the real
  architecture (window and bias extent shorter than the prompt, log scaling active): a whole-prompt prefill, a
  prefill followed by one-token decode steps through the caches, and a chunked prefill plus decode with the sliding
  layers reading only their window (max |diff| 1.2e-4 in fp32, argmax 100 percent); the relative shift and the window
  block selection against their brute-force forms.
- `test_weights_vs_hf.py`: the experts' dequantization and sharding against Hugging Face's load ops on the real
  checkpoint (`INKLING2_CKPT`), one NVFP4 and one MXFP8 layer: bit-identical.
