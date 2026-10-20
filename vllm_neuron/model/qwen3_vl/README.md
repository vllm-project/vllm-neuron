# Qwen3-VL Vision Encoder

BF16 vision encoder (ViT) for the Qwen3-VL multimodal model. Produces main
embeddings and deepstack features at specified intermediate layers for
injection into the text decoder.

## Architecture

| Parameter              | Qwen3-VL-8B       | Qwen3-VL-32B      |
|------------------------|--------------------|--------------------|
| hidden_size            | 1152               | 1152               |
| num_heads              | 16                 | 16                 |
| head_dim               | 72                 | 72                 |
| depth                  | 27                 | 27                 |
| intermediate_size      | 4304               | 4304               |
| out_hidden_size        | 4096               | 5120               |
| patch_size             | 16                 | 16                 |
| temporal_patch_size    | 2                  | 2                  |
| in_channels            | 3                  | 3                  |
| spatial_merge_size     | 2                  | 2                  |
| deepstack_visual_indexes | [8, 16, 24]      | [8, 16, 24]        |
| Activation             | GELU               | GELU               |
| Normalization          | LayerNorm (eps=1e-6) | LayerNorm (eps=1e-6) |
| Position Embedding     | 2D RoPE (rotate_half) + bilinear interpolation | 2D RoPE (rotate_half) + bilinear interpolation |
| Attention              | Full (non-GQA), bidirectional with bounds masking | Full (non-GQA), bidirectional with bounds masking |

## Key Differences from Reference (GPT-OSS BF16)

- **Vision encoder, not text decoder** — single forward pass, no KV cache,
  no decode path, no `is_decode` dispatch.
- **No Sequence Parallelism (SP)** — vision encoder runs the full sequence
  on all TP ranks. No all-gather/reduce-scatter collectives.
- **No GQA** — all heads are Q=K=V (full multi-head attention).
- **Bidirectional attention** — no causal mask. Uses bounds masking to
  isolate attention within image/frame groups.
- **LayerNorm instead of RMSNorm** — with learnable bias, computed in FP32.
- **GELU activation** — not SwiGLU. Uses `NF.mlp` with `skip_gate=True`.
- **2D RoPE** — interleaved rotate_half style, pre-computed on CPU from
  grid_thw spatial positions.
- **Custom QKV weight loader** — `vis_qkv_weight_loader` uses interleaved
  head sharding (not contiguous row slicing) because the checkpoint stores
  fused `[Q_all | K_all | V_all]`.
- **PatchMerger** — spatial merge via two-layer MLP with asymmetric dims
  (merged_hidden → merged_hidden → out_hidden). Uses manual matmul instead
  of `NF.mlp` due to asymmetric dimensions.
- **Deepstack mergers** — intermediate layer features are merged and injected
  into the text decoder at specified layer indices.
- **Conv3d patch embedding** — handles temporal + spatial patch extraction.

## Feature Status

Reference model: [`model/gpt_oss/model_bf16.py`](../gpt_oss/model_bf16.py)

| Feature                    | Status | Notes                                              |
|----------------------------|--------|----------------------------------------------------|
| TP (head sharding)         | ✅     | Attention heads + MLP intermediate sharded across vision TP group |
| SP (sequence parallel)     | N/A    | Vision encoder runs full sequence on all ranks     |
| DP (data parallel)         | ✅     | Each DP rank processes independent images          |
| Dependent DP               | N/A    | Not applicable to vision encoder                   |
| EP (expert parallel)       | N/A    | No MoE in vision encoder                           |
| Cross-DP EP                | N/A    | No MoE in vision encoder                           |
| Eagle3 spec decode         | N/A    | Vision encoder only (no autoregressive decoding)   |
| FP8 KV cache               | N/A    | No KV cache in vision encoder                      |
| Segmented prefill          | N/A    | Single-pass encoder                                |
| On-device sampling         | N/A    | Vision encoder only                                |
| torch.compile              | ✅     | Full-graph compilation with the active backend selector |
| CPU mode                   | ✅     | All modules run on CPU with `VLLM_NEURON_CPU_MODE=1` |
| Deepstack features         | ✅     | Intermediate layer features extracted and merged   |
| Bounds masking             | ✅     | Per-image/frame attention isolation via bound_min/bound_max |
| Bilinear pos_embed interp  | ✅     | CPU preprocessing matches HF `fast_pos_embed_interpolate` |
| 2D RoPE                    | ✅     | CPU preprocessing matches HF `rot_pos_emb`         |
| Weight loading             | ✅     | `SafetensorsCheckpoint.load_sharded` with custom QKV loader |
| Three-way accuracy tests   | ✅     | All modules validated against FP32 HF + BF16 HF   |

## Module Structure

```text
vllm_neuron/model/qwen3_vl/
├── __init__.py
├── README.md                          # This file
├── weight_loaders.py                  # vis_qkv_weight_loader, vis_qkv_bias_loader
├── modules/
│   ├── __init__.py
│   ├── vision_attention.py            # Multi-head attention + 2D RoPE + bounds mask
│   ├── vision_block.py                # Pre-norm transformer block (norm→attn→norm→MLP)
│   ├── vision_layernorm.py            # LayerNorm computed in FP32
│   ├── vision_mlp.py                  # GELU MLP via NF.mlp(skip_gate=True)
│   ├── vision_model.py                # Full ViT: PatchEmbed → Blocks → Mergers
│   ├── vision_patch_embed.py          # Conv3d patch projection
│   └── vision_patch_merger.py         # Spatial merge + two-layer MLP projection
└── utils/
    ├── __init__.py
    └── vision_preprocessing.py        # CPU: RoPE, pos_embed interpolation, attention bounds
```

## Test Structure

```text
test/vllm_neuron/model/qwen3_vl/bf16/
├── modules/
│   ├── test_vision_attention.py       # Three-way attention comparison
│   ├── test_vision_block.py           # Three-way block comparison
│   ├── test_vision_layernorm.py       # Three-way LayerNorm comparison
│   ├── test_vision_mlp.py             # Three-way MLP comparison
│   ├── test_vision_model.py           # Three-way full ViT comparison
│   ├── test_vision_patch_embed.py     # Three-way Conv3d comparison
│   ├── test_vision_patch_merger.py    # Three-way merger comparison
│   └── test_weight_loaders.py         # Unit tests with mock data (fast, no checkpoint)
└── utils/
    └── test_vision_preprocessing.py   # CPU preprocessing vs HF reference
```
