# Ouro

Neuron implementation of the ByteDance Ouro looped language model
(`OuroForCausalLM`, for example `ByteDance/Ouro-1.4B`). The model runs on the
`neuron_native` route in bf16.

## Files

| File | Purpose |
|---|---|
| `hf_config.py` | HuggingFace configuration class for `model_type: "ouro"`. Importing it registers the class with `AutoConfig`, so the checkpoint config parses without `trust_remote_code`. |
| `config.py` | `OuroConfig` dataclass, built from the HF config with `from_configs`. |
| `factory.py` | `OuroForCausalLM` factory that vLLM's `ModelRegistry` maps the architecture to. It validates the config and builds `model.py`. |
| `model.py` | The implementation (attention, MLP, decoder layer, UT loop, early exit, LM head, KV spec, weight loading). |
| `plugin.py` | `register()`, the `vllm.general_plugins` entry point: imports `hf_config` and registers the factory. |
| `compat.py` | Two helpers that vLLM Neuron 0.29.0.1.2.0 does not ship (`hoist_rope_parameters`, `build_per_head_block_table`). |

## Architecture

- One stack of `num_hidden_layers` decoder layers runs `total_ut_steps` times
  with shared weights (Universal Transformer loop). The final norm applies
  after each step. The normed output is the input of the next step.
- Each decoder layer has sandwich norms. `input_layernorm_2` applies to the
  attention output, and `post_attention_layernorm_2` applies to the MLP
  output, before each residual add.
- An early-exit gate (`Linear(hidden_size, 1)` with bias) after each step
  gives an exit probability. The language-model head uses the hidden state
  of the first step where the cumulative exit probability is at or above
  `early_exit_threshold`. If no step reaches the threshold, the head uses the
  last step.
- Attention is MHA with RoPE (default type) and no bias. The MLP is SwiGLU.

## KV cache

Each UT step writes its own KV cache. The KV spec has
`num_hidden_layers * total_ut_steps` layers with the names
`layers.{ut_step * num_hidden_layers + layer_idx}.self_attn`. This is the
cache index of the HF model. Each attention module holds one `OuroKVSlot` for
each UT step.

## Kernels

| Phase | Operation | Function |
|---|---|---|
| Prefill | QKV projection and RoPE | `NF.qkv_proj` |
| Prefill | Attention, segmented prefill off | `NF.flash_attention` |
| Prefill | Attention, segmented prefill on | `NF.segmented_attention` |
| Prefill | Output projection | `NF.o_proj` |
| Decode | QKV, RoPE, attention, output projection | `NF.attention_decode` |
| Both | MLP | `NF.mlp` |
| Both | Early-exit gate | `NF.router` (torch path, see below) |

The model takes the segmented path whenever the attention metadata has a KV
segment size. With `kv_segment_size_buckets` set (the configuration of the
README and of the integration test), the runner gives one for every prefill
chunk, so every chunk, the first one included, runs `NF.segmented_attention`,
and each later chunk attends to the KV cache of the earlier ones.
`NF.flash_attention` runs when segmented prefill is off, for example when
`max_num_batched_tokens` is at least `max_model_len`. The integration test
does not run it; the CPU component tests and the module test
`test/module/test_components_device.py` cover it.

`NF.router` has no kernel path at this time, so the early-exit gate runs as a
torch matmul. The gate runs only at the sampling positions.

## Limits

- Implemented: bf16, TP with sequence parallelism (SP) between the layers in
  prefill, DP. Text only.
- Exercised by the device integration test: TP 8, batch 1 and 4, maximum
  model length 9216 with prompts of 8000-8192 tokens, 1024-token prefill
  chunks with 1024-token KV segments. DP, other TP degrees, other chunk
  sizes and longer contexts are not validated.
- Not supported: `early_exit_step`, RoPE scaling, sliding window attention.
  The factory stops with an error for these configurations.
- The KV memory is `total_ut_steps` times the memory of a model with
  `num_hidden_layers` layers.
