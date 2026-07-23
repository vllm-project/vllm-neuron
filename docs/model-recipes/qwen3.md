# Qwen3 / Qwen3MoE (Text) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3 and Qwen3MoE text models
with vLLM on Neuron, including supported checkpoints, feature support, accuracy
results, and hardware validation for dense and MoE variants on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3, Qwen3MoE, Qwen3ForCausalLM,
Qwen3MoeForCausalLM, dense, MoE, BF16, model recipe, model card, LLM serving,
Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-23 -->
<!-- Content type: model-card -->

## Introduction

[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
is a family of dense and Mixture-of-Experts (MoE) language models developed by
the Qwen team. Qwen3 models support multilingual text generation, reasoning, and
tool use. The MoE variants (e.g. Tongyi-DeepResearch-30B-A3B) use top-k expert
routing for efficient inference at scale.

Qwen3 and Qwen3MoE are supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Type | Hardware |
|-------|-------------|------|----------|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Dense | Trn2 |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Dense | Trn2 |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Dense | Trn2 |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Dense | Trn2 |
| Tongyi-DeepResearch-30B-A3B | [Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | MoE (128 experts, top-8) | Trn2 |

> Both `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` architectures are
> registered natively — no `--hf-overrides` required. All models run in BF16.

## Features

Per-model feature availability for Qwen3/Qwen3MoE. See the
[features guide](../guides/features-guide.md) for configuration details and the
cross-model feature compatibility matrix.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | BF16 weights | ✅ |
| | FP8 KV cache | ❌ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ✅ (MoE) |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| | Speculative decoding (EAGLE3) | ❌ |
| | Disaggregated inference | ❌ |
| **Serving** | OpenAI-compatible logprobs | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3/Qwen3MoE
- ❌ Not supported: may be considered for future releases

For MoE models on four NeuronCores, use:

```bash
--tensor-parallel-size 4 --enable-expert-parallel
```

This yields EP=4 (e.g. 32 of Tongyi's 128 experts per rank).

## Accuracy Validation

Accuracy measured on Trn2.3xlarge hardware with BF16 weights.

**Dense (Qwen3-0.6B, TP=2, segment size 512):**

| Test | Result |
|------|--------|
| 1509-token prompt (3 segments), greedy decode | 16/16 tokens match HF BF16 |
| Repeated request (APC active, 46.7% hit) | 16/16 tokens match |
| Boundary prompts (508, 521, 807 tokens) | 16/16 tokens match |
| 989-token adversarial prompt, segmented | 15/16 (first token swaps to HF rank-2 at 0.25-logit margin) |
| 989-token prompt, full prefill | 16/16 tokens match |

**MoE (tiny Qwen3MoE, TP=2, 32 experts, top-8, segment 512):**

| Test | Result |
|------|--------|
| 807-token segmented prefill, teacher forcing | 8/8 tokens match HF BF16 |
| Repeated-prefix run (APC active, 90.9% hit) | Match |

**MoE (Tongyi-DeepResearch-30B-A3B, TP=4, EP=4):**

| Test | Result |
|------|--------|
| Full 48-layer checkpoint load + compile | ✅ |
| Coherent completion and chat output | ✅ |

## Limitations

- Text-only. Qwen2/Qwen2.5 and multimodal Qwen are separate implementations.
- BF16 and `head_dim=128` only.
- Segmented prefill inherits batch-size-1 restriction (no mixed prefill/decode).
- Sliding-window attention, attention sinks, and bias-enabled variants are not
  supported and are rejected during model construction.
- Some MoE checkpoints (e.g. local Tongyi copies) may lack tokenizer files —
  pass a compatible Qwen3 tokenizer with `--tokenizer`.
