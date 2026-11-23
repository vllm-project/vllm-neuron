# OLMo-3 (Olmo-3-7B-Instruct) on vLLM Neuron

This directory holds the `Olmo3ForCausalLM` implementation for the native
route (`neuron_native`, the default route of the plugin). The target
checkpoint is `allenai/Olmo-3-7B-Instruct`. The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 32 |
| Hidden size | 4096 |
| Intermediate size | 11008 (SiLU gate, up and down projections, no bias) |
| Attention | MHA, 32 Q heads, 32 KV heads, head_dim 128 |
| Layer types | Three sliding-window layers (window 4096), then one full-attention layer, repeated (24 sliding, 8 full) |
| RoPE | Default RoPE (theta 500000) on the sliding-window layers. YaRN (factor 8, original length 8192, attention factor 1.2079) on the full-attention layers |
| Normalization | RMSNorm, eps 1e-6 |
| Vocabulary | 100278, untied `lm_head` |
| Data type | BF16 checkpoint and BF16 weights; FP32 activations (see below) |

OLMo-3 differs from Llama in these places, and all of them are in this folder:

- The decoder layer is post-norm: `h = h + norm(attn(h))`, then
  `h = h + norm(mlp(h))`. There is no input norm.
- `q_norm` and `k_norm` are one RMSNorm each over the full projected width
  (4096), before RoPE. Under tensor parallelism each rank holds a slice of
  that width, so the sum of squares of q and k goes through one FP32 TP
  all-reduce.
- The RoPE is different for the two layer types. `config.py` reads the
  transformers 5 per-layer-type `rope_parameters` (and the older flat
  `rope_scaling` / `rope_theta` layout) and rejects a non-default RoPE on the
  sliding-window layers.
- The vocabulary does not divide by the TP degree at TP 16. The embedding and
  the LM head pad the vocabulary with zero rows, and the forward pass masks
  the pad logits before on-device sampling and drops them from the returned
  logits.

## Files

| File | Content |
|------|---------|
| `__init__.py` | Exports the factory `Olmo3ForCausalLM` and `Olmo3Config` |
| `plugin.py` | `register()` for the `vllm.general_plugins` entry point in `../pyproject.toml` |
| `factory.py` | Validates the HF config (BF16, no attention bias, known layer types) and builds `model.py` |
| `config.py` | `Olmo3Config` and the RoPE-parameter parse; `OLMO3_OPTIMIZATION_LEVEL` |
| `model.py` | RMSNorm, the two RoPE variants, attention (prefill and decode), MLP, decoder layer, backbone, LM head, KV cache spec, weight loading |
| `compat.py` | `hoist_rope_parameters`, which vLLM Neuron 0.29.0.1.2.0 does not ship |

## Parallelism

Tensor parallelism with sequence parallelism in prefill. Legal TP degrees are
2, 4, 8 and 16: the TP degree must divide the 32 heads and leave an even
number of Q heads on each rank, so TP 32 is rejected at construction. The
recorded runs use TP 16.

## Numerics and kernel paths

The implementation keeps every activation in FP32. Only the weights and the
KV cache are BF16; each projection casts its BF16 weight to FP32.

| Operation | Prefill | Decode |
|-----------|---------|--------|
| QKV projection | Torch matmul, FP32 | Torch matmul, FP32 |
| QK-norm and RoPE | Torch, FP32, full width, one all-reduce | Same |
| Attention | Torch attention in FP32 with a causal (and window) mask; `NF.segmented_attention` when the runner sets a KV segment size (segmented prefill) | Torch attention over the gathered paged KV cache in FP32 |
| O projection | Torch matmul, FP32, then an FP32 reduce-scatter | Torch matmul, FP32, then an FP32 all-reduce |
| MLP | Torch matmuls, FP32, then an FP32 reduce-scatter | Torch matmuls, FP32, then an FP32 all-reduce |
| Residual stream | FP32 between all 32 layers | Same |
| Final norm | FP32, rounded to BF16 one time before the LM head | Same |

The port did not use the NKI QKV, flash-attention, O-projection, MLP and
decode-attention kernels, because each of them rounds an intermediate (the
input, the softmax probabilities, or the per-rank partial sums before the
collective) to BF16. The post-norm layer feeds the raw residual into each
sublayer, and the error added up over the 32 layers: with BF16 activations
the full model was clearly less accurate than the HF BF16 reference, although
each layer alone passed. With FP32 activations the per-layer error is below
the HF BF16 error. The cost is speed: this is a correctness-first port, and
no performance figure is claimed.

The RoPE module computes its cos and sin tables one time on the CPU in FP32,
for every position up to `max_position_embeddings` (65536), with the HF
math. The tables are non-persistent buffers, so the meta default device of
the model runner and the weight loading do not touch them; the runner moves
them to the device with the model.

### Compiler optimization level

Compile at neuronx-cc `-O3` (`OLMO3_OPTIMIZATION_LEVEL` in `config.py`), by
passing `--optimization-level 3` to `vllm serve` or `optimization_level=3`
to `LLM(...)`. At the default `-O2`, the compiled full model showed a large
error that starts at layer 20 (and grows again at layers 21 and 24), while
each of those layers was correct when compiled and run alone, and the same
graph with extra outputs was correct too. `-O3` disables the neuronx-cc
modular flow, the probable cause, and the recorded logit-validation run used
it. vLLM Neuron maps the vLLM optimization level 1:1 onto the neuronx-cc
level, and an explicit level applies to every graph.

## Sliding window

Every layer declares a full-attention KV cache (`get_kv_spec` sets no
`sliding_window_size`), so the scheduler does not trim any block table and
the cache layout is the same for every layer. The sliding-window layers
apply the window with the attention mask: a key `j` is visible to a query at
position `i` when `j <= i` and `j > i - 4096`, the HF semantics.

- Prefill (torch attention): the mask is built in the graph. A layer passes
  no window when the window is not smaller than the longest context of the
  bucket, because such a window masks no key.
- Segmented prefill: the window goes to `NF.segmented_attention`.
- Decode: the mask over the gathered cache keeps the keys in
  `(pos - 4096, pos]`.

With `max_model_len` up to 4096 the window masks nothing, and the
sliding-window layers compute the same result as full attention. The recorded
device run used 2048. The window mask itself is checked on CPU
(`../test/unit`) with a reduced window: 16 tokens on the tiny checkpoint in
prefill (one decoder layer) and in decode (the full vLLM pipeline), and 48
tokens on the real weights in prefill when `OLMO3_CHECKPOINT_PATH` is set.
Removing the window from the prefill path or from the decode path makes
those tests fail (the end-to-end logit error rises to about 80x the HF BF16
error). The torch prefill and decode paths were not run on Neuron beyond
4096 tokens, where the window becomes active, and the segmented-prefill path
was not run at all.
