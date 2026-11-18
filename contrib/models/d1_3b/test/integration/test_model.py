# SPDX-License-Identifier: Apache-2.0
"""Accuracy test for LiquidAI d1-3B served through vLLM on Neuron.

d1-3B is a decision model: each (state, question) is one forward pass, and the
answer is a softmax over the question's option tokens at the answer slot. This test
renders the prompt and reads the answer out with the checkpoint's own code
(``runner.SystemOne.render`` / ``_readout``, loaded with ``trust_remote_code``),
so only the forward pass differs between the reference and vLLM:

  * Reference: the HF model on CPU in fp32 (``D1Model.engine.probabilities``).
  * Neuron: vLLM with ``max_tokens=1`` and raw logits; the full-vocabulary logits
    at the last position go through the same log-softmax and readout.

Gate (the d1-3B project's accuracy gate): every answer (top option) matches the
fp32 reference, and every option probability is within 0.0197 of it.

The fp32 reference runs first, in a separate process (``_reference.py``): a CPU
reference next to a Neuron load or compile can exhaust host memory on a
trn2.3xlarge, and torch CPU work in this process before vLLM forks its engine can
deadlock the engine's image preprocessing.

Running (Neuron hardware; see the model README):

    pytest contrib/models/d1_3b/test/integration/test_model.py -v -s --timeout=3600
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[2]  # contrib/models/d1_3b

# A local snapshot or a Hub id. Pin the revision the README was validated with.
MODEL_PATH = os.environ.get("D1_MODEL_PATH", "LiquidAI/d1-3B")
MODEL_REVISION = os.environ.get(
    "D1_MODEL_REVISION", "051bcc464b01b9f92942b364d9586b0ef5912432"
)
TP = int(os.environ.get("D1_TP", "4"))
# D1_PREFIX_CACHING=1: prefix caching + 512-token chunked prefill (conv state read
# back from the paged cache). Default: one-shot prefill, no prefix caching.
PREFIX_CACHING = os.environ.get("D1_PREFIX_CACHING", "0") == "1"
# D1_IMAGES=1: also build the vision tower and test image states.
IMAGES = os.environ.get("D1_IMAGES", "0") == "1"
# Vision buckets in raw patches: one 512 px tile per 1024-patch block, so 4096 holds
# 4 tiles and 12288 holds 12 (the processor's maximum is 10 tiles + a thumbnail).
ENCODER_NEURON_CONFIG = {
    "num_encoder_input_buckets": [4096, 12288],
    "encoder_attention_block_size": 1024,
    # Explicit: the derived size follows the scheduler's encoder budget (at least
    # max_num_batched_tokens), which is too small for a 12-tile image when the step
    # budget is 512 (prefix caching). 25 = two largest images + 1 scratch block.
    "encoder_cache_num_blocks": 25,
}

IMAGE_QUESTIONS = {
    "circle": {"type": "noul", "instructions": "Is there a red circle in the picture?"},
    "square": {
        "type": "choice",
        "instructions": "What color is the square?",
        "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
    },
}


# A small picture (one tile) and a large one (tiles + thumbnail); drawn by
# _reference.test_image.
IMAGE_SIZES = ((384, 384), (1024, 768))


PROB_GATE = 0.0197

STATE = (
    "The order arrived damaged yesterday. The customer has a receipt and asks for a "
    "replacement today."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"],
    },
}
# A longer state, so the prompt spans several KV blocks and prefill buckets.
LONG_STATE = json.dumps(
    {
        "ticket": STATE,
        "history": [
            {
                "date": f"2026-10-{d:02d}",
                "note": f"Customer called about order {1000 + d}.",
            }
            for d in range(1, 25)
        ],
    }
)


def _snapshot() -> str:
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_PATH, revision=MODEL_REVISION)


def _remote(path: str, name: str):
    """A module of the checkpoint's remote code (pure-Python prompt helpers)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Loads <name>.py in the checkpoint's dynamic-module package; any attribute
    # of the module works as the anchor.
    anchor = {"prompt": "as_question", "runner": "cap_pixels"}[name]
    fn = get_class_from_dynamic_module(f"{name}.{anchor}", path)
    return sys.modules[fn.__module__]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """fp32 CPU reference (in a subprocess, see _reference.py) and readout helpers."""
    from transformers import AutoConfig, AutoTokenizer

    path = _snapshot()
    req = {
        "model_path": path,
        "text": [
            {"name": n, "state": s, "spec": spec}
            for s in (STATE, LONG_STATE)
            for n, spec in QUESTIONS.items()
        ],
        "images": [
            {"name": f"{n}@{w}x{h}", "size": [w, h], "spec": spec}
            for w, h in (IMAGE_SIZES if IMAGES else ())
            for n, spec in IMAGE_QUESTIONS.items()
        ],
    }
    d = tmp_path_factory.mktemp("d1_reference")
    (d / "req.json").write_text(json.dumps(req))
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_reference.py")),
            str(d / "req.json"),
            str(d / "out.json"),
        ],
        check=True,
        # All CPU threads for the fp32 reference, even when the serving environment sets
        # OMP_NUM_THREADS=1 (fork-safety note in the README); it is a separate process.
        env={**os.environ, "OMP_NUM_THREADS": str(os.cpu_count() or 1)},
    )
    out = json.loads((d / "out.json").read_text())

    # Host-side helpers only (no torch CPU compute before vLLM forks).
    prompt = _remote(path, "prompt")
    runner = _remote(path, "runner")
    tokenizer = AutoTokenizer.from_pretrained(path)
    specs = {c["name"]: c["spec"] for c in req["text"]}
    cases = [
        {
            "name": c["name"],
            "q": prompt.as_question(specs[c["name"]]),
            "ids": c["ids"],
            "probs": c["probs"],
        }
        for c in out["text"]
    ]
    image_specs = {c["name"]: c for c in req["images"]}
    sys.path.insert(0, str(Path(__file__).parent))
    from _reference import test_image

    image_cases = []
    for c in out["images"]:
        r = image_specs[c["name"]]
        image_cases.append(
            {
                "name": c["name"],
                "q": prompt.as_question(r["spec"]),
                "text": c["text"],
                "images": [runner.cap_pixels(test_image(tuple(r["size"])))],
                "probs": c["probs"],
            }
        )

    def readout(q, logz):
        # D1Model.engine's readout: no calibration by default.
        return prompt.readout(tokenizer, q, logz, None)

    def option_ids(q):
        return sorted({i for g in prompt.readout_ids(tokenizer, q) for i in g})

    vocab = AutoConfig.from_pretrained(path).text_config.vocab_size
    return {
        "path": path,
        "cases": cases,
        "image_cases": image_cases,
        "readout": readout,
        "option_ids": option_ids,
        "vocab": vocab,
    }


@pytest.fixture(scope="module")
def llm(reference):
    # vLLM forks its engine. An OpenMP pool does not survive fork: the forked engine
    # deadlocks in libgomp the first time it runs a parallel CPU op (the HF image
    # processor while vLLM sizes the multimodal budget). With one CPU thread torch
    # never enters an OpenMP parallel region. Device work is unaffected.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    # Workers are separate processes: make the contrib package importable there too.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(MODEL_DIR), os.environ.get("PYTHONPATH", "")]
    )
    sys.path.insert(0, str(MODEL_DIR))
    from src import register

    register()
    from vllm import LLM

    longest = max(len(c["ids"]) for c in reference["cases"])
    max_model_len = 4096 if (PREFIX_CACHING or IMAGES) else 2048
    neuron_config = {"on_device_sampling_config": {"all_greedy": True}}
    # The encoder config is needed even for text-only runs: the class is multimodal,
    # so the runner always builds the encoder (with unusable defaults otherwise).
    additional_config = {
        "neuron_config": neuron_config,
        "encoder_neuron_config": ENCODER_NEURON_CONFIG,
    }
    assert longest < max_model_len, longest
    engine = LLM(
        model=reference["path"],
        tensor_parallel_size=TP,
        max_model_len=max_model_len,
        # Prefix caching needs segmented prefill on Neuron, so the step budget must
        # be a supported segment size below max_model_len; that also chunks the
        # long state, which exercises the conv state carried between chunks.
        max_num_batched_tokens=512 if PREFIX_CACHING else max_model_len,
        max_num_seqs=1,
        # Raw logits (logprobs_mode) of the option tokens (logprob_token_ids).
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        async_scheduling=False,
        # Conv state lives in the paged cache, so prefix reuse is correct; the
        # questions about one state then share it as a prefix.
        enable_prefix_caching=PREFIX_CACHING,
        additional_config=additional_config,
        **({"limit_mm_per_prompt": {"image": 1}} if IMAGES else {}),
    )
    yield engine
    engine.llm_engine.engine_core.shutdown()


def _neuron_probs(llm, reference, case) -> tuple[list[float], int]:
    """Option probabilities and the number of prompt tokens served from the prefix cache."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    if "images" in case:
        # Text + pictures: vLLM runs the upstream LFM2-VL processor, which expands
        # each <image> into its tile markers and image tokens.
        prompt = {"prompt": case["text"], "multi_modal_data": {"image": case["images"]}}
    else:
        prompt = TokensPrompt(prompt_token_ids=case["ids"])
    # Raw logits of the question's option tokens only (logprob_token_ids): the readout
    # is a softmax over those, so the vocabulary normalisation cancels. Returning the
    # full vocabulary instead (logprobs=-1) costs ~0.9 s per request on the host.
    ids = reference["option_ids"](case["q"])
    out = llm.generate(
        [prompt],
        SamplingParams(
            max_tokens=1, temperature=0.0, logprob_token_ids=ids, logprobs=len(ids)
        ),
        use_tqdm=False,
    )[0]
    step = out.outputs[0].logprobs[0]  # {token_id: Logprob}, raw logits
    missing = set(ids) - set(step)
    assert not missing, f"option tokens missing from the logprobs: {missing}"
    logits = {t: lp.logprob for t, lp in step.items()}
    return list(reference["readout"](case["q"], logits)), out.num_cached_tokens or 0


@pytest.mark.integration
def test_image_decisions_match_fp32_reference(llm, reference):
    if not IMAGES:
        pytest.skip("set D1_IMAGES=1")
    failures = []
    for case in reference["image_cases"]:
        got, _ = _neuron_probs(llm, reference, case)
        want = case["probs"]
        diff = max(abs(a - b) for a, b in zip(got, want))
        top_ok = max(range(len(got)), key=got.__getitem__) == max(
            range(len(want)), key=want.__getitem__
        )
        print(
            f"{case['name']:18s} max|dp|={diff:.6f} top={'ok' if top_ok else 'MISMATCH'} "
            f"neuron={[round(p, 4) for p in got]} fp32={[round(p, 4) for p in want]}"
        )
        if not top_ok or diff > PROB_GATE:
            failures.append((case["name"], diff, top_ok))
    assert not failures, (
        f"Outside the gate (top-1 match, |dp| <= {PROB_GATE}): {failures}"
    )


@pytest.mark.integration
def test_decisions_match_fp32_reference(llm, reference):
    failures = []
    total_cached = 0
    for case in reference["cases"]:
        got, cached = _neuron_probs(llm, reference, case)
        total_cached += cached
        want = case["probs"]
        diff = max(abs(a - b) for a, b in zip(got, want))
        top_ok = max(range(len(got)), key=got.__getitem__) == max(
            range(len(want)), key=want.__getitem__
        )
        print(
            f"{case['name']:8s} len={len(case['ids']):4d} cached={cached:4d} max|dp|={diff:.6f} "
            f"top={'ok' if top_ok else 'MISMATCH'} neuron={[round(p, 4) for p in got]} "
            f"fp32={[round(p, 4) for p in want]}"
        )
        if not top_ok or diff > PROB_GATE:
            failures.append((case["name"], len(case["ids"]), diff, top_ok))
    assert not failures, (
        f"Outside the gate (top-1 match, |dp| <= {PROB_GATE}): {failures}"
    )
    if PREFIX_CACHING:
        # Questions about one state share it as a prefix; later ones must hit the cache,
        # so the conv state for those tokens really is read back from the paged cache.
        assert total_cached > 0, (
            "prefix caching enabled but no prompt tokens were cached"
        )
