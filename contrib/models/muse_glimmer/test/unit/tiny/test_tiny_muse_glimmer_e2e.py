# SPDX-License-Identifier: Apache-2.0
"""Tiny E2E test for Muse-Glimmer: vLLM Neuron (CPU mode) vs HF transformers, greedy.

Random weights, real architecture: [local, local, local, global] layers, gated
attention, weight-less Q/K norm with qk_scale_factor, NoPE global layers,
centered norms, logit softcap. ``sliding_window=16`` with a 24-token prompt and
16 new tokens, so the window is crossed in both prefill and decode. The
checkpoint uses the real weight names (``model.language_model.*``,
``model.vision_tower.*``), so the weight mapping is covered too.

With random weights the top logits are nearly tied (gaps of ~0.01), so BF16
rounding alone can change the greedy token. The check is therefore near-greedy:
HF scores vLLM's tokens teacher-forced, and every token vLLM picked must be
within ``MARGIN`` of HF's best logit at that step. A random token is typically
0.3-0.6 below the best, so a wrong model or a wrong weight mapping fails.

Text only: a multimodal request needs a tokenizer and the transformers
processor, which the tiny checkpoint does not have. The vision tower is still
built and loaded.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/muse_glimmer/test/unit/tiny/test_tiny_muse_glimmer_e2e.py -v
"""

import importlib.metadata

import pytest
import torch
from transformers import AutoModelForImageTextToText
from transformers.models.muse_glimmer import MuseGlimmerConfig
from vllm import LLM, SamplingParams

# TP=2 workers are separate processes: they see the model only through the
# installed vllm.general_plugins entry point.
INSTALLED = any(
    ep.value.startswith("muse_glimmer_neuron")
    for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
)

PROMPT = list(range(5, 29))  # 24 tokens > sliding_window
NEW_TOKENS = 16
MARGIN = 0.05  # logit units; see the module docstring


@pytest.fixture(scope="module")
def tiny_muse_glimmer_dir(tmp_path_factory):
    cfg = MuseGlimmerConfig(
        text_config=dict(
            num_hidden_layers=4,
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=4096,
            sliding_window=16,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
        ),
        vision_config=dict(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
        ),
        out_hidden_size=128 * 4,
        projector_hidden_size=256,
        image_token_id=4000,
        video_token_id=4001,
    )
    assert cfg.text_config.layer_types == ["sliding_attention"] * 3 + ["full_attention"]
    torch.manual_seed(0)
    hf = AutoModelForImageTextToText.from_config(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():  # norms near 0 / 1 so that every norm does something
        for name, prm in hf.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                base = 0.0 if "layernorm" in name else 1.0
                prm.copy_(torch.randn_like(prm) * 0.1 + base)
    model_dir = tmp_path_factory.mktemp("tiny_muse_glimmer")
    hf.save_pretrained(str(model_dir))
    return str(model_dir), hf


def _hf_margins(hf, tokens):
    """HF's best logit minus HF's logit of each generated token (teacher-forced)."""
    with torch.no_grad():
        logits = hf(input_ids=torch.tensor([PROMPT + tokens])).logits[0].float()
    steps = logits[len(PROMPT) - 1 : len(PROMPT) - 1 + len(tokens)]
    picked = steps[torch.arange(len(tokens)), torch.tensor(tokens)]
    return (steps.max(dim=-1).values - picked).tolist()


@pytest.mark.parametrize(
    "tp_size,world_size",
    [
        pytest.param(1, 1, id="tp1"),
        pytest.param(
            2,
            2,
            id="tp2",
            marks=pytest.mark.skipif(
                not INSTALLED,
                reason="TP=2 workers need the installed entry point: "
                "pip install -e contrib/models/muse_glimmer",
            ),
        ),
    ],
)
def test_greedy_matches_hf(tiny_muse_glimmer_dir, tp_size, world_size):
    """vLLM's greedy tokens are HF's best (within MARGIN) at every step.

    TP=2 covers sharded loading and collectives. ``world_size`` is the
    core-allocator parameter of the test suite.
    """
    model_dir, hf = tiny_muse_glimmer_dir
    llm = LLM(
        model=model_dir,
        max_num_seqs=1,
        max_model_len=128,
        block_size=16,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        # vLLM makes 4 KV-cache groups (sliding-window and full layers):
        # 4 x (128 / 16) blocks for one sequence.
        num_gpu_blocks_override=64,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [32, 128]}},
    )
    out = llm.generate(
        [{"prompt_token_ids": PROMPT}],
        SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True),
    )
    got = list(out[0].outputs[0].token_ids)
    assert len(got) == NEW_TOKENS
    margins = _hf_margins(hf, got)
    print(f"TP={tp_size}: tokens {got}, HF margins {[round(m, 4) for m in margins]}")
    assert max(margins) <= MARGIN, (
        f"TP={tp_size}: a vLLM token is {max(margins):.3f} below HF's best logit "
        f"(margins {[round(m, 3) for m in margins]})"
    )
