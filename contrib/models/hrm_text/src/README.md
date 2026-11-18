# HRM-Text

This directory holds the native (`neuron_native`) implementation of HRM-Text
(Hierarchical Reasoning Model, text). The reference checkpoint is
`sapientinc/HRM-Text-1B`.

HRM-Text is a text-only model. It uses standard text generation. It is not an
Omni model.

## Architecture

| Item | Value |
|---|---|
| Registry name | `HrmTextForCausalLM` |
| Stacks | 2 (H and L), 16 decoder layers each |
| Recurrence | `H_cycles = 2`, `L_cycles = 3` |
| KV cache slots | 128 (`16 x H_cycles x (L_cycles + 1)`) |
| Hidden size | 1536 |
| Intermediate size | 4096 (SwiGLU) |
| Attention | MHA, 12 heads, head dimension 128 |
| Position embedding | RoPE, `rope_theta = 10000` |
| Vocabulary | 65536, untied LM head |
| Dtype | BF16 |

The forward pass runs this recurrence over the same token embeddings:

```text
z_H = embed(input_ids) * embedding_scale
z_L = z_L_init
for h in range(H_cycles):
    for l in range(L_cycles):
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)
logits = lm_head(z_H)
```

## Differences from Llama

- The RMSNorm has no weight. Each stack also has a final RMSNorm without a weight.
- The attention has a sigmoid output gate: `o_proj(sigmoid(gate_proj(x)) * attn)`.
  The fused decode kernel cannot apply the gate. Thus the kernel runs with
  `W_out=None`, and the model applies the gate and `o_proj` after the kernel.
- The checkpoint fuses the gate, Q, K and V weights in `gqkv_proj`, and the
  MLP gate and up weights in `gate_up_proj`.
- The embedding output is multiplied by `embedding_scale`.
- Each pass of a layer writes a different KV cache slot. Thus one layer owns
  more than one KV cache.

## Feature Status

| Feature | Status |
|---|---|
| Tensor parallelism | TP 1 and TP 2. A larger TP gives an odd number of heads on each rank, and the decode kernel needs an even number. |
| Sequence parallelism (prefill) | Supported |
| Multi-node TP | TP 2 with one rank on each of 2 nodes |
| On-device sampling | Supported |
| PrefixLM attention (`token_type_ids`) | Not supported. The model uses causal attention for all tokens. |
| Quantization | Not supported. The checkpoint is BF16. |

## Kernel Paths

On a Neuron device, each main path uses an NKI kernel through an NF API. Each
NF API uses its torch fallback when its eligibility check fails. In CPU mode,
`can_run_kernel` is false, so each API uses its torch fallback.

| Phase | Operation | NF API | NKI kernel |
|---|---|---|---|
| Prefill | QKV projection | `NF.qkv_proj` | `qkv` |
| Prefill | Causal attention | `NF.flash_attention` | `attention_cte` |
| Prefill | Chunked prefill attention | `NF.segmented_attention` | `attention_segmented_cte` |
| Prefill and decode | Gate projection | `NF.qkv_proj` | `qkv` |
| Prefill and decode | Output projection | `NF.o_proj` (BNdS layout) | `output_projection_cte` |
| Prefill and decode | SwiGLU MLP | `NF.mlp` | `mlp` |
| Decode | QKV, RoPE, KV cache write, attention | `NF.attention_decode` (`W_out=None`) | `attention_block_tkg` |
| Decode | Attention mask | `NF.gen_attention_decode_mask` | `gen_mask_tkg_hbm` |

The decode kernel folds the KV heads into the batch. Each rank has more than
one KV head, so the model gives the kernel a per-head block table
`[B, Nkh, blocks]` and per-head write slots `[B, Nkh, S]`. The torch fallback
of `NF.attention_decode` does not accept a per-head table. Thus the model gives
the fallback the 2D block table and the `[B, S]` slots.

The first decode layer of a step builds the decode mask, the per-head block
table and the per-head slots. The other layer passes of the step use them again.

These operations stay in torch:

- The sigmoid gate and the elementwise multiply.
- The prefill RoPE and the prefill KV cache write (`index_put_`).
- The RMSNorm without a weight, the embedding scale and the residual adds.

## Running This Model

Registering the architecture with vLLM, serving the checkpoint and running the
tests are all in the model folder's [`README.md`](../README.md). The tests that
ship with this code are `test/integration/test_model.py` (logit validation
through vLLM), `test/unit/test_hrm_text_cpu.py` (CPU-mode loading and
components) and `test/module/test_hrm_text_modules_multinode.py` (device
components at TP 2 over 2 nodes).
