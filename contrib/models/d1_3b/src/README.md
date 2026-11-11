# d1-3B modeling code

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports the factory and `register()`. |
| `config.py` | `D1TextConfig.from_configs`: parses `text_config` of the `lfm2_vl` checkpoint. |
| `factory.py` | `D1ForConditionalGeneration`, registered as `Lfm2VlForConditionalGeneration` (multimodal interface with the upstream LFM2-VL processor, plus the plugin's encoder-sizing protocols); `register()`. |
| `model.py` | Attention, gated short conv, dense SwiGLU MLP, backbone, `D1NeuronForCausalLM` (`forward` with the encoder-cache merge, `embed_multimodal`, `get_encoder`, `build_encoder_synthetic_inputs`, `get_kv_spec`, `bind_kv_cache`, `load_weights`). |
| `vision.py` | SigLIP2 NaFlex tower + LFM2-VL projector (`D1VisionTower`), host-side tile packing, checkpoint mappings. |
| `decode_kv.py` | Copy of the plugin's decode-time KV helpers (self-containment rule). |

## Feature status

| Feature | Status |
| --- | --- |
| Text states, TP prefill (sequence-parallel) | Implemented |
| Conv state in the paged cache (chunked prefill, prefix caching) | Implemented |
| Decode path | Implemented (not used by d1-3B: `max_tokens=1`) |
| Vision tower + projector (one tile per encoder block, replicated per rank) | Implemented; matches HF on CPU and on Trn2 (4/4 image decisions) |
| NKI block kernels / fp8 MLP | Not implemented |

## KV spec

| Layer type | Cache name | kv heads per rank | head size |
| --- | --- | --- | --- |
| `full_attention` (8) | `layers.{i}.self_attn` | `8 / TP` (min 1) | 64 (128 with `D1_ATTN_PAD128=1`) |
| `conv` (22) | `layers.{i}.conv` | 1 | `(2048 / TP) / 2` (K: first half of `B*x`, V: second half) |

## Vision encoder

| Item | Value |
| --- | --- |
| Tower | SigLIP2 NaFlex: hidden 1152, 16 heads (head_dim 72), 27 layers, MLP 4304 (GELU tanh), LayerNorm eps 1e-6, post LayerNorm |
| Patches | 16x16x3 = 768 values; up to 1024 per tile (512 px tile = 32x32 patches) |
| Position embedding | learned 16x16 grid, resized per tile on the host (bilinear, antialias) |
| Projector | 2x2 pixel-unshuffle (4608) -> linear 2048 -> GELU -> linear 2048 |
| Encoder block | 1024 raw patches = one tile (`encoder_attention_block_size=1024`) |
| Encoder-cache block | 256 merged tokens = one tile |
