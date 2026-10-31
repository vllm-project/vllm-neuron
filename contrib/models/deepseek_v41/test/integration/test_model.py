# SPDX-License-Identifier: Apache-2.0
"""Accuracy test for the contributed DeepSeek-V4.1-Flash model, served through
vLLM on Neuron.

**Why not the template's logit validation.**
``contrib/models/template/test/integration/test_model.py`` compares per-token
logits against a HuggingFace reference from
``vllm_neuron.accuracy.get_cached_reference_goldens``. That reference cannot be
built for this checkpoint: its default loader wants an upstream ``transformers``
implementation, and this architecture (hyper-connections, engram memory, a
sparse indexer, the DSpark MTP stages) has none; and the three-way golden needs
an fp32 baseline, which for a 475 GiB fp8/fp4 checkpoint is roughly 1.9 TB of
parameters. So this takes the "or equivalent" route ``contrib/README.md``
allows: serve the model and check the generated text.

**What is asserted.** The model package ships a verifier (``src/verifier/``)
that builds a real engine per phase and generates greedily at temperature 0.
This test drives it and asserts that every phase ran, that every selected case
produced tokens, and that every case declaring an ``expected_substring``
contains it. ``long-needle-2048-0`` is the case that carries the weight: it
plants an access code 2008 tokens deep, which broken long-context attention, a
mis-sharded weight or a wrong RoPE cannot recall by luck. Every completion is
printed either way.

**Running.** Needs a trn2.48xlarge with all 64 logical cores free (the port
serves at ``config.TENSOR_PARALLEL_SIZE = 64``) and the checkpoint on local
disk::

    DEEPSEEK_V41_CHECKPOINT=/path/to/DeepSeek-V4.1-Flash \\
        pytest contrib/models/deepseek_v41/test/integration/test_model.py \\
        -v -s --timeout=14400

``-s`` matters: 64 workers log progress and a cold compile takes tens of
minutes, so captured output makes a live run look hung.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_TEST_ROOT = Path(__file__).resolve().parent.parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import model_package  # noqa: E402

#: No default path is baked in: 475 GiB does not live anywhere predictable, and
#: falling back to the Hub id would turn a misconfigured run into an hours-long
#: download.
_CHECKPOINT_ENV = "DEEPSEEK_V41_CHECKPOINT"


def _checkpoint() -> str:
    path = os.environ.get(_CHECKPOINT_ENV)
    if not path:
        pytest.skip(
            f"{_CHECKPOINT_ENV} is not set. Point it at a local "
            f"DeepSeek-V4.1-Flash checkpoint directory; see this model's "
            f"README.md for the Hub id and the download command."
        )
    if not (Path(path) / "config.json").is_file():
        pytest.fail(f"{_CHECKPOINT_ENV}={path!r} has no config.json")
    return path


@pytest.mark.integration
def test_model_accuracy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the contributed model through vLLM and check the generated text."""
    checkpoint = _checkpoint()
    model_package.load()

    from deepseek_v41.verifier import prompts
    from deepseek_v41.verifier.run_verifier import main as run_verifier

    # The NKI toolchain writes one directory per compiled kernel into the
    # current working directory, and pytest's rootdir is the repository.
    monkeypatch.chdir(tmp_path)

    report_path = tmp_path / "verifier_report.json"
    returncode = run_verifier(["--model", checkpoint, "--report", str(report_path)])

    assert report_path.is_file(), (
        f"the verifier wrote no report to {report_path}; it exited {returncode} "
        f"before reaching the generation phase."
    )
    report = json.loads(report_path.read_text())

    # `.get`, not `in`: the key is pre-initialised to None in the report, so a
    # membership test is always true and would never catch a failed phase.
    execution_error = report.get("execution_error")
    assert not execution_error, f"a verifier phase raised:\n{execution_error}"

    cases = {c["id"]: c for c in report.get("cases", [])}
    missing = sorted(set(prompts.SELECTED_IDS) - set(cases))
    assert not missing, (
        f"no completion for {missing}; the engine dropped the case(s) rather "
        f"than failing. Report notes: {report.get('notes')}"
    )
    assert cases, "the verifier reported no cases at all"
    for case_id, case in sorted(cases.items()):
        assert case.get("num_tokens"), f"{case_id} generated no tokens"

    checks = report.get("checks") or []
    assert checks, (
        "the verifier checked no expected substrings, so the run proved nothing "
        "about accuracy."
    )
    failed = [c for c in checks if not c["matched"]]
    assert not failed, "\n".join(
        f"{c['id']}: expected {c['expected']!r} in {cases[c['id']]['text']!r}"
        for c in failed
    )

    assert returncode == 0, f"verifier exited {returncode}"
    print(f"\n{len(checks)} expected substring(s) matched across {len(cases)} case(s).")
