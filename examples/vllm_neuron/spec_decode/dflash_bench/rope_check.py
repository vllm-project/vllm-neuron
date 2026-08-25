"""Compare the draft's RoPE against the one its weights were trained with.

vllm-neuron's DFlash draft uses GptOssRotaryEmbedding (the target's YaRN).
The z-lab reference drafter uses transformers' Qwen3RotaryEmbedding with
rope_type=yarn. If those disagree, every draft position is encoded slightly
wrong, which would lower acceptance smoothly with position.
"""

import json

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from vllm_neuron.model.dflash.config import DFlashConfig
from vllm_neuron.model.gpt_oss.model_bf16 import GptOssRotaryEmbedding

cfg_raw = json.load(open("/models/gpt-oss-20b-DFlash/config.json"))
cfg = DFlashConfig.from_configs(cfg_raw)
print("rope_parameters:", cfg.rope_parameters)

gpt = GptOssRotaryEmbedding(cfg)
pos = torch.tensor([0, 1, 7, 64, 512, 4095, 4096, 8191])
cos_g, sin_g = gpt(pos, device=torch.device("cpu"), dtype=torch.float32)
print("gptoss cos shape:", tuple(cos_g.shape))

# transformers' YaRN for the same config

hf = AutoConfig.for_model(
    "qwen3",
    **{
        k: v
        for k, v in cfg_raw.items()
        if k not in ("architectures", "auto_map", "model_type")
    },
)
fn = ROPE_INIT_FUNCTIONS["yarn"]
inv_freq, attn_scale = fn(hf, device=torch.device("cpu"))
print("transformers yarn attention_factor:", attn_scale)
freqs = pos.float().unsqueeze(1) * inv_freq.unsqueeze(0)
cos_t = freqs.cos() * attn_scale
sin_t = freqs.sin() * attn_scale

half = cos_g.shape[-1]
print(f"{'pos':>6} {'max|cos_gptoss-cos_hf|':>24} {'max|sin diff|':>15}")
for i, p in enumerate(pos.tolist()):
    dc = (cos_g[i, :half].float() - cos_t[i, :half]).abs().max().item()
    ds = (sin_g[i, :half].float() - sin_t[i, :half]).abs().max().item()
    print(f"{p:>6} {dc:>24.6f} {ds:>15.6f}")
