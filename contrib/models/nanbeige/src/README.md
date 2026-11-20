# Nanbeige4.2 on vLLM Neuron

This directory holds the native-route (`neuron_native`) implementation of
`NanbeigeForCausalLM`. The target checkpoint is `Nanbeige/Nanbeige4.2-3B`,
revision `b82e54bd609793562a75cbf9337970a93369eab5`. The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Physical layers | 22 |
| Loops | 2 (44 logical layers, 44 KV cache slots) |
| Hidden size | 3072 |
| Intermediate size | 10752 (SwiGLU, `silu`) |
| Attention | GQA, 48 Q heads, 8 KV heads, head_dim 128 |
| RoPE | Default (no scaling), theta 7.0e7 |
| Vocabulary | 166144, untied LM head |
| Dtype | BF16 weights |

Nanbeige4.2 is a Llama-style dense decoder with one addition: the stack of
physical decoder layers runs `num_loops` times with the same weights. Each
`(loop, physical layer)` pair gets its own KV cache slot, at index
`loop_idx * num_hidden_layers + layer_idx`, which is the layout the
checkpoint's own model code uses. The final RMSNorm runs after each loop
unless `skip_loop_final_norm` is set.

The checkpoint uses Llama tensor names, so `get_weight_mappings` maps the
parameters directly; the fused QKV loader splits `q_proj`, `k_proj` and
`v_proj` per rank.

Because two loops double the effective depth (88 residual additions), the
residual stream stays in FP32 across loops and the LM head runs in FP32. The
kernels take BF16 inputs.

## Parallelism

- Prefill uses sequence parallelism: each rank holds 1/TP of the tokens, so a
  prefill length must be greater than the TP size and a multiple of it.
  Attention and the MLP all-gather the tokens and reduce-scatter the result.
- Decode uses the fused attention megakernel, which updates the KV cache in
  place, followed by a TP all-reduce.
- KV heads are replicated when there are fewer KV heads than TP ranks.
- The embedding and the LM head are sharded on the vocabulary dimension.

TP8 is the degree the device runs used: one trn2.48xlarge node with vLLM
Neuron 0.29.0.1.2.0. The CPU tests build the model at TP2. No other degree has
a recorded result.

## Limits

The factory refuses a configuration it cannot run, rather than loading and
producing wrong output. It rejects:

- `qk_layernorm`, `attention_bias`, `mlp_bias`, `enable_hyper_connection`,
  `enable_mhc`, `enable_depth_attention`, `enable_double_loop_split` and
  `loop_share_kv` when set to `True`;
- `emb_neighbor_num` (n-gram embeddings) and a non-empty `loop_loss_weights`;
- `hidden_act` other than `silu`;
- `rope_scaling` other than `default`;
- a dtype other than BF16.

Other limits:

- `inputs_embeds` is not supported. Pass `input_ids`.
- There is no quantized path and no sliding-window attention.
- The forward pass has a speculative-decoding branch (it calls
  `vllm_neuron.nn.rejection_sampler` when `spec_decode_metadata` is present),
  but no recorded run exercised it, so its status is unknown.
- The decode megakernel does not run on CPU, so CPU-mode tests cover the
  prefill path only.

## Feature status

| Feature | Status |
|---|---|
| BF16 dense inference, text only | Supported; validated (see the folder README's validation report) |
| Looped layer stack with per-loop KV cache | Supported; CPU tests compare both loops against an independent reference |
| TP with prefill sequence parallelism | Supported; TP8 on device, TP2 on CPU |
| Fused decode megakernel | Supported on device; not available in CPU mode |
| On-device greedy sampling | Supported; used by the device logit run |
| Quantization (FP8, MXFP4, ...) | Not supported (no quantized path in the code) |
| Sliding-window / chunked attention | Not applicable (`sliding_window_size=None`, `chunk_size=None` in the KV spec) |
| Multimodal input | Not applicable (text-only checkpoint) |
| `inputs_embeds` | Not supported (explicit error) |
| Speculative decoding | Unknown (code path present, no evidence) |
| Data parallelism / disaggregated serving | Unknown (no evidence for this model) |

## Tests

- `../test/unit/test_nanbeige_cpu.py`: CPU-mode registration, config parsing,
  factory refusals, a TP2 full-model load and prefill-logit comparison, and a
  cross-check of the independent reference against the checkpoint's own model
  code.
- `../test/module/test_decoder_layer.py`: CPU-mode per-component test of one
  physical decoder layer at TP2, run for each loop, against an upstream
  `LlamaDecoderLayer` reference, including the KV cache that each loop owns.
- `../test/integration/test_model.py`: single-node TP8 offline logit
  validation through vLLM.

The tests load this directory by file location through
`../test/_local_src.py` (module name `contrib_nanbeige_src`), not through a
repository-rooted import path, so the folder keeps working when it is copied
into an application. `../test/nanbeige_utils.py` holds the helpers that the
unit and module tests share.
