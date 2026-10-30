# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU check: a tiny random MuseGlimmer checkpoint through vLLM vs HF transformers.

Runs vLLM with the Neuron plugin in CPU mode (``VLLM_NEURON_CPU_MODE=1``), in fp32
(``MUSE_GLIMMER_COMPUTE_DTYPE=float32``), and requires the greedy tokens to equal the HF
fp32 greedy tokens exactly. Text prompts cover the sliding window (300 tokens over a
64-token window); image prompts go through the plugin's multimodal processor and the
perception encoder. TP=1 and TP=4 (KV heads replicated 2x).

The image cases need the MuseGlimmer processor files: set ``MUSE_GLIMMER_PROCESSOR`` to
a local checkpoint directory or to ``meta-models/Muse-Glimmer-30B`` (gated). Without it
the image cases are skipped.

    pip install -e contrib/models/muse_glimmer
    VLLM_NEURON_CPU_MODE=1 MUSE_GLIMMER_PROCESSOR=meta-models/Muse-Glimmer-30B \\
        pytest contrib/models/muse_glimmer/test/unit/test_tiny_e2e.py -v -s

Each case takes a few minutes on CPU and about 10 GB of host memory at TP=4.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("vllm")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tiny_checkpoint import build  # noqa: E402

PROCESSOR = os.environ.get("MUSE_GLIMMER_PROCESSOR")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    path = Path(os.environ.get("MUSE_GLIMMER_TINY_CKPT", tmp_path_factory.mktemp("tiny_muse_glimmer")))
    if not (path / "goldens.pt").exists():
        build(str(path), PROCESSOR)
    return path


def _run(ckpt: Path, tp: int, mode: str) -> dict:
    env = dict(os.environ, VLLM_NEURON_CPU_MODE="1")
    out = subprocess.run([sys.executable, str(HERE / "run_tiny.py"), str(ckpt), "--tp", str(tp), "--mode", mode],
                         env=env, capture_output=True, text=True, timeout=3600)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
    assert out.returncode == 0 and line, f"run_tiny failed:\n{out.stdout[-4000:]}\n{out.stderr[-4000:]}"
    return json.loads(line[len("RESULT_JSON "):])


@pytest.mark.parametrize("tp", [1, 4])
def test_text_greedy_equals_hf(ckpt, tp):
    gold = torch.load(ckpt / "goldens.pt", weights_only=False)["text"]
    res = _run(ckpt, tp, "text")
    for name, g in gold.items():
        assert res[name]["tokens"] == g["tokens"], (name, res[name]["tokens"], g["tokens"])


@pytest.mark.parametrize("tp", [1, 4])
def test_image_greedy_equals_hf(ckpt, tp):
    gold = torch.load(ckpt / "goldens.pt", weights_only=False)["image"]
    if not gold:
        pytest.skip("set MUSE_GLIMMER_PROCESSOR to run the image cases")
    res = _run(ckpt, tp, "image")
    for name, g in gold.items():
        assert res[name]["prompt_token_ids"] == g["prompt_token_ids"], name
        assert res[name]["tokens"] == g["tokens"], (name, res[name]["tokens"], g["tokens"])
