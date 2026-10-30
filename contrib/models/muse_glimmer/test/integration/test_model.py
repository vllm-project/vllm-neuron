# SPDX-License-Identifier: Apache-2.0
"""Integration accuracy test for MuseGlimmer, served through vLLM on Neuron.

The full model (30B text decoder + 1.8B perception encoder) is too large for an FP32
HuggingFace reference on the host, so the test builds a reduced checkpoint with the
real weights of the first ``NUM_LAYERS`` text and vision layers
(``scripts/make_mini.py``) and checks it against HF transformers on the same weights:

* ``test_text_logits``: three-way logit validation (``vllm_neuron.accuracy``) on text
  prompts. One prompt is longer than the 2048-token sliding window, so the trimmed
  sliding-window decode table is exercised.
* ``test_image_greedy``: image prompts through the plugin's multimodal processor and the
  Neuron perception encoder. The prompt token ids must equal HF's, and the greedy tokens
  must equal HF FP32's until the first step where HF's own top-1 / top-2 margin is a
  near-tie.

Registration: install the contrib package first (``pip install -e
contrib/models/muse_glimmer``). vLLM ships its own GPU ``MuseGlimmerForConditionalGeneration``,
and the engine core and the Neuron workers are spawned processes, so the package registers
the Neuron class in every process through a ``vllm.general_plugins`` entry point.

Environment:
  MUSE_GLIMMER_CKPT       Full meta-models/Muse-Glimmer-30B checkpoint directory (required
                          unless MUSE_GLIMMER_MINI_CKPT already holds a reduced checkpoint).
  MUSE_GLIMMER_MINI_CKPT  Where the reduced checkpoint is built / cached
                          (default: /tmp/muse_glimmer_mini<NUM_LAYERS>).

Run on a trn2.3xlarge (TP=4):

    MUSE_GLIMMER_CKPT=/path/to/Muse-Glimmer-30B \\
        pytest contrib/models/muse_glimmer/test/integration/test_model.py -v -s --timeout=7200

The same test runs in CPU mode (``VLLM_NEURON_CPU_MODE=1``) as a functional smoke run.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

NUM_LAYERS = 4
TP = int(os.environ.get("MUSE_GLIMMER_TP", "4"))
MINI_CKPT = Path(os.environ.get("MUSE_GLIMMER_MINI_CKPT", f"/tmp/muse_glimmer_mini{NUM_LAYERS}"))
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
UNIT = Path(__file__).resolve().parents[1] / "unit"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In 1969, Neil Armstrong became the first person to",
]
# Longer than the 2048-token sliding window.
LONG_PROMPT = " ".join(f"Item {i}: the value is {i * 7 % 13}." for i in range(250)) + " Summary:"  # ~2500 tokens

OUTPUT_LENGTH = 16
SEQ_LEN = int(os.environ.get("MUSE_GLIMMER_SEQ_LEN", "4096"))
FIXED_DATE = "2026-01-01"

# BF16 against the BF16 / FP32 HF goldens: the template's dense-golden convention.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Argmax divergence tolerance, in logits. A 4-layer slice has a flat next-token
# distribution, so BF16 rounding can flip the argmax at a near-tie. The CPU-mode smoke run
# of this test measured one flip with a gap of 0.094. 0.15 still fails a real defect,
# which shows up as gaps of several logits.
DIVERGENCE_TOL = 0.15

torch.manual_seed(42)


def _register():
    try:
        from muse_glimmer_neuron import register
    except ImportError:
        pytest.fail("install the contrib package first: pip install -e contrib/models/muse_glimmer")
    register()


@pytest.fixture(scope="module")
def mini_ckpt() -> Path:
    """Build (once) the reduced real-weight checkpoint."""
    if not (MINI_CKPT / "config.json").exists():
        full = os.environ.get("MUSE_GLIMMER_CKPT")
        if not full:
            pytest.skip("set MUSE_GLIMMER_CKPT to the meta-models/Muse-Glimmer-30B checkpoint directory")
        subprocess.run([sys.executable, str(SCRIPTS / "make_mini.py"), full, str(MINI_CKPT), str(NUM_LAYERS)],
                       check=True)
    return MINI_CKPT


def _llm(ckpt: Path, *, image: bool):
    from vllm import LLM

    neuron_config = {
        "on_device_sampling_config": {"all_greedy": True},
        "num_batched_tokens_buckets": sorted({1024, SEQ_LEN}),
        "num_seqs_buckets": [1],
    }
    additional = {"neuron_config": neuron_config}
    kw = {}
    if image:
        additional["vision_neuron_config"] = {
            "vision_attention_block_size": 4096,
            "num_vision_tokens_buckets": [4096],
            "encoder_cache_num_blocks": 8,
            "tp_size": TP,
        }
        kw = dict(limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_image_tokens": 1024})
    # max_logprobs / logprobs_mode / async_scheduling / all_greedy are required by the
    # logit validation (see contrib/models/template).
    return LLM(
        model=str(ckpt),
        tensor_parallel_size=TP,
        max_model_len=SEQ_LEN,
        max_num_batched_tokens=SEQ_LEN,
        max_num_seqs=1,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config=additional,
        **kw,
    )


@pytest.mark.integration
def test_text_logits(mini_ckpt: Path) -> None:
    """Three-way logit validation of text prompts (one longer than the sliding window)."""
    _register()
    from transformers import AutoConfig, AutoTokenizer

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    model_config = AutoConfig.from_pretrained(mini_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(mini_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    goldens = get_cached_reference_goldens(
        model_id=f"meta-models/Muse-Glimmer-30B-mini{NUM_LAYERS}",
        model_checkpoint=str(mini_ckpt),
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS + [LONG_PROMPT],
        output_length=OUTPUT_LENGTH,
    )
    llm = _llm(mini_ckpt, image=False)
    try:
        result = multi_prompt_logit_validation(
            prompts_input_ids=[[list(ids) for ids in p] for p in goldens["input_ids"]],
            generate_fn=create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH),
            prompts_expected_logits=goldens["dtype_logits"],
            prompts_baseline_logits=goldens["fp32_logits"],
            colorize=True,
            tol_map=TOL_MAP,
            divergence_n_ulps=None,
            divergence_difference_tol=DIVERGENCE_TOL,
        )
        assert result.passed, "MuseGlimmer text logit validation failed"
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.integration
def test_image_greedy(mini_ckpt: Path) -> None:
    """Image prompts: prompt ids equal HF's; greedy tokens equal HF FP32's up to a near-tie."""
    _register()
    from transformers import AutoProcessor, MuseGlimmerForConditionalGeneration
    from vllm import SamplingParams

    sys.path.insert(0, str(UNIT))
    from tiny_checkpoint import BOS, IMAGE_PROMPTS, synth_image

    proc = AutoProcessor.from_pretrained(mini_ckpt)
    ref = MuseGlimmerForConditionalGeneration.from_pretrained(
        mini_ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    with torch.no_grad():
        for (w, h, seed), question in IMAGE_PROMPTS.values():
            img = synth_image(w, h, seed)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": question}]}]
            kw = dict(add_generation_prompt=True, current_date=FIXED_DATE)
            inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", **kw)
            text = proc.apply_chat_template(msgs, tokenize=False, **kw)
            seq, margins = inp["input_ids"][0].tolist(), []
            P = len(seq)
            for _ in range(OUTPUT_LENGTH):
                last = ref(input_ids=torch.tensor([seq]), pixel_values=inp["pixel_values"].float(),
                           image_grid_thw=inp["image_grid_thw"], use_cache=False).logits[0, -1]
                top2 = torch.topk(last, 2).values
                margins.append(float(top2[0] - top2[1]))
                seq.append(int(last.argmax()))
            cases.append((text[len(BOS):], img, seq[:P], seq[P:], margins))
    del ref

    llm = _llm(mini_ckpt, image=True)
    try:
        sp = SamplingParams(max_tokens=OUTPUT_LENGTH, temperature=0.0, ignore_eos=True)
        for prompt, img, ids, toks, margins in cases:
            out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)[0]
            assert list(out.prompt_token_ids) == ids, "prompt token ids differ from HF"
            got = list(out.outputs[0].token_ids)
            n = next((i for i, (a, b) in enumerate(zip(got, toks)) if a != b), OUTPUT_LENGTH)
            print(f"image prompt (P={len(ids)}): {n}/{OUTPUT_LENGTH} greedy tokens equal HF FP32")
            if n < OUTPUT_LENGTH:
                # BF16 against an FP32 golden: a divergence is allowed only at an HF near-tie.
                assert n >= 1 and margins[n] < 0.5, f"divergence at step {n}, HF margin {margins[n]:.3f}"
    finally:
        llm.llm_engine.engine_core.shutdown()
