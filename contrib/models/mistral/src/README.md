# Mistral-7B-v0.3

BF16 text decoder for Mistral-7B-v0.3 (`MistralForCausalLM`) on the Neuron
model runner. The code is derived from the plugin's dense reference
implementation (`vllm_neuron/model/llama3/`), with the classes renamed and the
Mistral-specific behaviour applied. It uses the plugin's shared operators
(`vllm_neuron.functional`: fused QKV projection, flash and segmented attention,
decode attention, fused MLP) and so the plugin's NKI kernels.

Deployment, compatibility, test results and known limitations: see
[../README.md](../README.md).

## Files

| File | What it holds |
|---|---|
| `__init__.py` | Package marker, kept import-free so vLLM's plugin load does not pull in torch |
| `plugin.py` | `register()`: registers `MistralForCausalLM` with vLLM's `ModelRegistry`, replacing vLLM's GPU class. Run in every process through the `vllm.general_plugins` entry point in `../pyproject.toml` |
| `config.py` | `MistralConfig`, parsed from the Hugging Face `config.json` (`from_configs`) |
| `factory.py` | Registry entry; validates the config (rejects sliding window and quantized checkpoints) and builds the model with `from_configs`. Carries the `VllmModel` protocol stubs vLLM's front end inspects |
| `model.py` | The model: `MistralRMSNorm`, `MistralRotaryEmbedding`, `MistralAttention`, `MistralMLP`, `MistralDecoderLayer`, `MistralModel`, `MistralForCausalLM` (`forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`) |

## Architecture

From `config.py` (Mistral-7B-v0.3): 32 layers, hidden 4096, 32 query heads,
8 KV heads, head_dim 128, SwiGLU intermediate 14336, vocab 32768, RMSNorm
eps 1e-5, max position 32768, untied embeddings.

| Item | Mistral-7B-v0.3 |
|---|---|
| Attention | GQA, 32 Q / 8 KV heads, no bias, causal |
| RoPE | default, theta 1e6; `rope_scaling` of type `llama3` is rejected |
| `head_dim` | not in `config.json`; derived as `hidden_size // num_attention_heads` |
| Sliding window | none in v0.3; checkpoints that set it, such as v0.1, are rejected |
| MLP | SwiGLU, 4096 to 14336 to 4096 |
| Norms | RMSNorm before attention and MLP, final norm |
| Vocabulary | 32768 |
| Weights | BF16 only; `quantization_config` is rejected |

Not carried over from the reference implementation: the Eagle3 and P-EAGLE
draft-model factories, the draft-model config fields, and the FP8 and MX FP8
model variants and weight loaders.

## Execution paths

Prefill (`MistralAttention.forward_prefill`): sequence-parallel layout between
layers; all-gather before attention, fused QKV projection, RoPE, flash
attention over the full prompt (or segmented attention when prefix caching
supplies cached blocks), output projection, reduce-scatter back to the SP
layout. The MLP follows the same gather / reduce-scatter pattern.

Decode (`MistralAttention.forward_decode`): one token per sequence, decode
attention reading K/V through the block table, all-reduce after the output
projection and after the MLP. Decode context parallelism (`dcp_size > 1`)
splits the cached sequence across ranks.

Parallelism: tensor parallel over attention heads and the MLP intermediate
dimension; vocab-sharded embedding and column-parallel LM head. With TP above
the KV-head count (8), KV heads are replicated.

## Checkpoint

Hugging Face layout (`model-0000N-of-00003.safetensors`) with the standard
`model.layers.N.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj,
input_layernorm,post_attention_layernorm}` names; `load_weights` fuses Q/K/V
and gate/up at load time. A directory that also contains Mistral's
`consolidated.safetensors` is rejected before loading (the plugin's loader
would otherwise read both copies).

## Validated configuration

Trn2 (trn2.3xlarge, LNC 2), TP 4 and TP 2, BF16, `max_model_len` 2048 to
16384, `max_num_seqs` up to 32, prefix caching on and off. Details and
numbers in [../README.md](../README.md).

## Not supported

TP 1 on Trn2 (the shared fused decode MLP kernel does not fit 14336
intermediate per rank in SBUF), sliding-window checkpoints, quantized
checkpoints. Supported in the code but not run for this model: data
parallelism inside one server, decode context parallelism, FP8 KV cache,
Eagle3 target support.
