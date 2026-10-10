# SPDX-License-Identifier: Apache-2.0
"""The verifier's input set.

Three sources, 4 cases:

  - **TranSonic's sample inputs** (2): `short-prose` and the 8192-token
    needle-in-a-haystack `long-needle-8192-0`. These are the prompts the
    partition loop traces and verifies against, so a reasonable answer here is
    comparable to that loop's recorded goldens.
  - **The onboarding guide's smoke test** (1): `"Hello, my name is"`, verbatim.
  - **A trivia question** (1): `"What is the capital of France?"` — the cheapest
    possible check that the model knows anything at all.

One short and one long, not all nine. Every prompt costs a full pass over 475
GiB of timeshared weights, so the input set is where runtime is decided; two
TranSonic cases keep both a chat-templated short prompt and a long-context
retrieval while the other seven were re-measuring the same code path. Dropping
the 16k case also halves `max_model_len`, which is what the KV cache is sized
from. The unselected rows stay in `inputs/` — switching one line below widens
the set again without re-copying anything.

The TranSonic sets are **copied into `inputs/` beside this file**, byte-for-byte
from their source, and read from there by a path anchored to this package.
The verifier holds no
path to anything outside vllm-neuron, so it does not depend on TranSonic being
checked out alongside it. `inputs/long.provenance.json` is copied too, recording
how `long.jsonl` was generated (seed 20260920), so the copies stay traceable to
their origin without naming its location.

`role` is TranSonic's: `user` means apply the tokenizer's chat template, `raw`
means a plain continuation. `expected_substring`, where the set provides it, is
the needle a correct answer should contain — passed through to the judge as a
hint, never used as a hard assertion, since a 32-token completion may
legitimately not restate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: The copied input sets, anchored to this file's own location.
#:
#: Not a hardcoded absolute path (which would break the moment the repo is
#: checked out anywhere else) and not a path relative to the working directory
#: (which would break depending on where the verifier is launched from).
#: Anchoring to `__file__` resolves to the same files either way, and keeps
#: every input inside this package.
INPUTS_DIR = Path(__file__).resolve().parent / "inputs"

SHORT_JSONL = INPUTS_DIR / "short.jsonl"
LONG_JSONL = INPUTS_DIR / "long.jsonl"

#: Which rows of the copied sets to use, by `id`. One short and one long:
#: `short-prose` is chat-templated (`role: user`), `long-needle-8192-0` is a raw
#: 8192-token continuation carrying a needle, so between them the two exercise
#: both prompt paths. Add ids here to widen the set — every file in `inputs/` is
#: still present, so nothing needs re-copying.
SELECTED_IDS = ("short-prose", "long-needle-8192-0")

#: Verbatim from the onboarding guide's smoke test.
SMOKE_TEST_PROMPT = "Hello, my name is"

#: The trivia check.
TRIVIA_PROMPT = "What is the capital of France?"


@dataclass
class Case:
    """One prompt to generate from, and what we know about a good answer."""

    id: str
    prompt: str
    role: str = "raw"
    source: str = "builtin"
    target_tokens: int | None = None
    expected_substring: str | None = None

    def excerpt(self, head: int = 600, tail: int = 400) -> str:
        """A judge-sized view of the prompt.

        The 16k case is 105 KB; sending it whole would swamp the verdict with
        filler. Head and tail are what matter — the instruction is at the top
        and the actual question is at the bottom.
        """
        if len(self.prompt) <= head + tail:
            return self.prompt
        cut = len(self.prompt) - head - tail
        return f"{self.prompt[:head]}\n\n[... {cut} chars elided ...]\n\n{self.prompt[-tail:]}"


@dataclass
class CaseSet:
    cases: list[Case] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The verifier's input sets are copied into "
            f"{INPUTS_DIR} and are tracked as part of this repository; restore "
            "them from version control if they have gone missing."
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_cases(max_prompt_tokens: int | None = None) -> CaseSet:
    """Assemble the full verifier input set.

    Reads the copied sets at `INPUTS_DIR`; there is no path argument, so
    every run uses the same 4 prompts.

    Args:
        max_prompt_tokens: If set, drop long cases whose `target_tokens`
            exceeds it, so a verifier run with a small `max_model_len` skips
            what it cannot fit instead of failing mid-generation. Recorded in
            `notes` either way, so a short run can never silently look like a
            full one.

    Returns:
        A `CaseSet` of 4 cases (fewer if long prompts are filtered out).

    Raises:
        FileNotFoundError: if an input set is missing.
        KeyError: if `SELECTED_IDS` names a row the sets do not contain — a
            typo there would otherwise silently shrink the verifier.
    """
    out = CaseSet()
    seen_ids: set[str] = set()

    for path in (SHORT_JSONL, LONG_JSONL):
        for row in _load_jsonl(path):
            if row["id"] not in SELECTED_IDS:
                continue
            seen_ids.add(row["id"])
            target = row.get("target_tokens")
            if (
                max_prompt_tokens is not None
                and target is not None
                and target > max_prompt_tokens
            ):
                out.notes.append(
                    f"skipped {row['id']}: target_tokens={target} exceeds "
                    f"max_prompt_tokens={max_prompt_tokens}"
                )
                continue
            out.cases.append(
                Case(
                    id=row["id"],
                    prompt=row["prompt"],
                    role=row.get("role", "raw"),
                    source=f"transonic:{path.name}",
                    target_tokens=target,
                    expected_substring=row.get("expected_substring"),
                )
            )

    unknown = sorted(set(SELECTED_IDS) - seen_ids)
    if unknown:
        raise KeyError(
            f"SELECTED_IDS names {unknown}, which is not in {SHORT_JSONL.name} or "
            f"{LONG_JSONL.name}. Fix the id rather than letting the verifier run a "
            "smaller set than it reports."
        )

    out.cases.append(
        Case(
            id="smoke-test",
            prompt=SMOKE_TEST_PROMPT,
            role="raw",
            source="onboarding-guide",
        )
    )
    out.cases.append(
        Case(
            id="trivia-capital-france",
            prompt=TRIVIA_PROMPT,
            role="user",
            source="builtin:trivia",
            expected_substring="Paris",
        )
    )
    return out
