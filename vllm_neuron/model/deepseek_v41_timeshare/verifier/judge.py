# SPDX-License-Identifier: Apache-2.0
"""The second pass/fail condition: is the generated text reasonable, or gibberish?

A 32-token greedy completion cannot be checked by string equality — there is no
single right answer to "continue this prose" — but gibberish is obvious to a
reader. So the verdict is delegated to a small agent: Sonnet 5 at medium effort,
via the `claude` CLI already on this host.

Deliberately strict about what counts as a pass:

  - The judge returns JSON, and a reply that will not parse is a FAIL, not a
    pass-by-default. A judge that cannot answer has not cleared the model.
  - Every case needs its own verdict. A missing one is a FAIL.
  - The judge never sees whether a previous run passed, so it cannot be
    anchored by it.

Runs entirely on the host. Nothing here touches a Trainium device.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

#: Sonnet 5 at medium effort, per the verifier's spec.
JUDGE_MODEL = "claude-sonnet-5"
JUDGE_EFFORT = "medium"
JUDGE_TIMEOUT_S = 600

_RUBRIC = """\
Grade a newly ported LLM's output. Per case you get the prompt (long ones elided \
in the middle) and the completion, capped at 32 tokens. Verdict per case: \
REASONABLE or GIBBERISH.

REASONABLE: text that is about the prompt. Cut off mid-sentence is fine, so is \
incomplete or restating the question. This is a reasoning model in thinking \
mode, so a completion is often its chain of thought — terse and note-like, \
quoting the prompt back, sometimes closed by a literal </think> — and that is \
REASONABLE whether or not an answer follows. Empty is not.

GIBBERISH: looping tokens or phrases; random unrelated words; mojibake or \
control characters; unexpected script; punctuation only; or no relation to the \
prompt.

Judge fluency and relevance only — factual errors, refusals and style are not \
gibberish. Where a case lists expected_substring, containing it is strong \
evidence of REASONABLE but is never required; never mark GIBBERISH for omitting \
it.

Reply with ONLY a JSON object, no prose and no code fence:

{"verdicts": [{"id": "<case id>", "verdict": "REASONABLE"|"GIBBERISH", \
"reason": "<one short sentence>"}]}

Include exactly one entry per case, using the case ids verbatim.

=== CASES ===
"""


@dataclass
class Verdict:
    id: str
    verdict: str
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict.upper() == "REASONABLE"


@dataclass
class JudgeResult:
    verdicts: list[Verdict]
    passed: bool
    error: str | None = None
    raw: str = ""

    @property
    def gibberish(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.ok]


def _render(results: list[dict]) -> str:
    blocks = []
    for r in results:
        lines = [f"--- case id: {r['id']} (source: {r.get('source', '?')})"]
        if r.get("expected_substring"):
            lines.append(f"expected_substring: {r['expected_substring']}")
        lines.append(f"prompt:\n{r['prompt_excerpt']}")
        lines.append(f"completion ({r.get('num_tokens', '?')} tokens):\n{r['text']!r}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def judge_outputs(results: list[dict]) -> JudgeResult:
    """Ask the judge whether every completion is reasonable.

    Args:
        results: One dict per case with keys `id`, `prompt_excerpt`, `text`,
            and optionally `source`, `expected_substring`, `num_tokens`.

    Returns:
        A `JudgeResult`. `passed` is True only when every case came back
        REASONABLE and the reply parsed cleanly.
    """
    if not results:
        return JudgeResult([], False, error="no generations to judge")

    cli = shutil.which("claude")
    if cli is None:
        return JudgeResult(
            [], False, error="the 'claude' CLI is not on PATH; cannot run the judge"
        )

    prompt = _RUBRIC + _render(results)
    try:
        proc = subprocess.run(
            [cli, "--print", "--model", JUDGE_MODEL, "--effort", JUDGE_EFFORT],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=JUDGE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return JudgeResult(
            [], False, error=f"judge timed out after {JUDGE_TIMEOUT_S}s"
        )

    raw = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return JudgeResult(
            [],
            False,
            error=f"judge exited {proc.returncode}: {(proc.stderr or '').strip()[:400]}",
            raw=raw,
        )

    # Tolerate a code fence or surrounding prose, but not a missing object.
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return JudgeResult(
            [], False, error="judge reply contained no JSON object", raw=raw
        )
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        return JudgeResult([], False, error=f"judge reply not valid JSON: {exc}", raw=raw)

    verdicts = [
        Verdict(
            id=str(v.get("id", "")),
            verdict=str(v.get("verdict", "")),
            reason=str(v.get("reason", "")),
        )
        for v in payload.get("verdicts", [])
    ]

    seen = {v.id for v in verdicts}
    missing = [r["id"] for r in results if r["id"] not in seen]
    if missing:
        return JudgeResult(
            verdicts,
            False,
            error=f"judge returned no verdict for: {', '.join(missing)}",
            raw=raw,
        )

    return JudgeResult(verdicts, all(v.ok for v in verdicts), raw=raw)
