# SPDX-License-Identifier: Apache-2.0
"""Build a tiny random-weight MuseGlimmer checkpoint and HF fp32 greedy goldens (CPU).

Text: 4 layers (3 sliding + 1 full), the released head geometry (32 q / 2 kv heads,
head_dim 128), hidden 1024, the full 202048-token vocabulary. Vision: 4 layers
(3 window + 1 full) at the released width 1536 / 16 heads. Norm weights, biases and
the LM head are randomized beyond the HF init so every load path matters and the
greedy margins sit well above BF16 noise.

The image goldens need the MuseGlimmer tokenizer and processor config. Pass a local
directory or a Hugging Face model id (``meta-models/Muse-Glimmer-30B``, gated) as
``processor_src``; only the processor files are downloaded.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

IMAGE_TOKEN_ID = 200092
VIDEO_TOKEN_ID = 200091
BOS = "<|begin_of_text|>"
FIXED_DATE = "2026-01-01"


def synth_image(w: int, h: int, seed: int):
    """Deterministic structured test image (gradients + discs)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)),
                    ((xx + yy) * 127 // max(w + h, 1))], -1).astype(np.uint8)
    for _ in range(6):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(h // 12, h // 5)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(0, 255, 3)
    return Image.fromarray(img)


TEXT_PROMPT_LENS = {"short13": 13, "mid40": 40, "long300": 300}
IMAGE_PROMPTS = {
    "image_448": ((448, 448, 0), "Describe this image in detail."),
    "image_896x672": ((896, 672, 1), "How many circles are in this image?"),
}


def build(out: str, processor_src: str | None, sliding_window: int = 64, new_tokens: int = 6,
          seed: int = 0) -> dict:
    """Write the checkpoint to ``out`` and return the goldens (also saved as ``goldens.pt``)."""
    from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration

    torch.manual_seed(seed)
    text = dict(
        vocab_size=202048, hidden_size=1024, intermediate_size=2048, num_hidden_layers=4,
        num_attention_heads=32, num_key_value_heads=2, head_dim=128, hidden_activation="silu",
        max_position_embeddings=131072, rms_norm_eps=1e-5, sliding_window=sliding_window,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
        final_logit_softcapping=20.0, qk_scale_factor=3.87, output_multiplier=0.19611613513818404,
        post_norm_eps=1e-8, rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        tie_word_embeddings=False, bos_token_id=200000, eos_token_id=200001,
    )
    vision = dict(
        hidden_size=1536, intermediate_size=8960, num_hidden_layers=4, num_attention_heads=16,
        layer_types=["window_attention"] * 3 + ["full_attention"], patch_size=14, patch_temporal=2,
        merge_size=2, pos_emb_height=32, pos_emb_width=32,
    )
    cfg = MuseGlimmerConfig(text_config=text, vision_config=vision, image_token_id=IMAGE_TOKEN_ID,
                            video_token_id=VIDEO_TOKEN_ID, out_hidden_size=6144,
                            projector_hidden_size=4096)
    cfg.architectures = ["MuseGlimmerForConditionalGeneration"]
    model = MuseGlimmerForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        lm = model.model.language_model
        for layer in lm.layers:
            for n in ("input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                getattr(layer, n).weight.normal_(0.0, 0.2)
        lm.norm.weight.copy_(1.0 + 0.2 * torch.randn_like(lm.norm.weight))
        model.lm_head.weight.normal_(0.0, 20.0 / text["hidden_size"] ** 0.5)
        vt = model.model.vision_tower
        vt.patch_embedder.position_embedding_table.weight.normal_(0.0, 0.5)
        for name, p in vt.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.05)
            elif ("norm" in name or "ln_" in name) and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
        for lin in (model.model.vision_adapter.fc1, model.model.vision_adapter.fc2,
                    model.model.vision_projection):
            lin.weight.normal_(0.0, 1.0 / lin.weight.shape[1] ** 0.5)
    os.makedirs(out, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    del model

    proc = None
    if processor_src:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(processor_src)
        proc.save_pretrained(out)

    # fp32 reference from the saved bf16 weights (the same weights vLLM loads).
    ref = MuseGlimmerForConditionalGeneration.from_pretrained(
        out, dtype=torch.float32, attn_implementation="eager").eval()
    g = torch.Generator().manual_seed(seed + 1)
    goldens = {"new_tokens": new_tokens, "text": {}, "image": {}}

    def greedy(ids, **mm):
        seq, margins = list(ids), []
        with torch.no_grad():
            for _ in range(new_tokens):
                last = ref(input_ids=torch.tensor([seq]), use_cache=False, **mm).logits[0, -1]
                top2 = torch.topk(last, 2).values
                margins.append(float(top2[0] - top2[1]))
                seq.append(int(last.argmax()))
        return seq[len(ids):], margins

    for name, n in TEXT_PROMPT_LENS.items():
        ids = torch.randint(1000, 199000, (n,), generator=g).tolist()
        toks, margins = greedy(ids)
        goldens["text"][name] = dict(prompt_token_ids=ids, tokens=toks, margins=margins)

    if proc is not None:
        for name, ((w, h, s), q) in IMAGE_PROMPTS.items():
            msgs = [{"role": "user", "content": [{"type": "image", "image": synth_image(w, h, s)},
                                                 {"type": "text", "text": q}]}]
            kw = dict(add_generation_prompt=True, current_date=FIXED_DATE)
            inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", **kw)
            text_prompt = proc.apply_chat_template(msgs, tokenize=False, **kw)
            toks, margins = greedy(inp["input_ids"][0].tolist(), pixel_values=inp["pixel_values"].float(),
                                   image_grid_thw=inp["image_grid_thw"])
            assert text_prompt.startswith(BOS)
            goldens["image"][name] = dict(
                prompt=text_prompt[len(BOS):], image=(w, h, s),
                prompt_token_ids=inp["input_ids"][0].tolist(), tokens=toks, margins=margins)
    torch.save(goldens, os.path.join(out, "goldens.pt"))
    with open(os.path.join(out, "goldens.json"), "w") as f:
        json.dump({k: list(v) for k, v in goldens.items() if isinstance(v, dict)}, f)
    return goldens
