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

One short and one long, not all nine: every prompt costs a full pass over 475
GiB of timeshared weights. Dropping the 16k case also halves `max_model_len`.
The unselected rows stay in `inputs/`, so widening the set is a one-line change.

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

#: Which rows of the copied sets to use. `short-prose` is chat-templated,
#: `long-needle-2048-0` is a raw continuation, so the two cover both prompt
#: paths. Add ids to widen the set — `long.jsonl` also carries an 8192 row,
#: which encodes to 9328 tokens.
SELECTED_IDS = ("short-prose", "long-needle-2048-0")

#: The onboarding guide's smoke test is the raw fragment `"Hello, my name is"`. This asks the
#: question instead, through the chat template, because the raw fragment made a
#: reasoning-tuned model degenerate: greedy continuation of five untemplated tokens came back
#: `" not\nHello, my name is not\nHello, my name is not..."`, which a judge scores as
#: gibberish without that being a fault in the port. A templated question exercises the same
#: path and has a right answer.
SMOKE_TEST_PROMPT = "Hello, what is your name?"

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


def build_cases() -> CaseSet:
    """Assemble the full verifier input set.

    Reads the copied sets at `INPUTS_DIR`; there is no path argument, so
    every run uses the same 4 prompts.

    Takes no budget and drops nothing. Deciding what fits needs a token count, and only the
    tokenizer has one — `render` measures, and `run_verifier` drops against the measurement.
    An earlier version filtered here on `target_tokens`, which is the length a sample was
    generated to *target* rather than its length, so it kept a 9328-token prompt under an
    8192-token budget and the engine rejected it.

    Returns:
        A `CaseSet` of all 4 selected cases.

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
            role="user",
            source="onboarding-guide (adapted: templated question)",
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


def render(
    case_set: CaseSet, model_path: str, chat_template_path: Path
) -> tuple[list[tuple[str, int]], list[str]]:
    """Render each case to the exact string the engine will see, and measure it.

    Returns `[(text, n_tokens), ...]` parallel to `case_set.cases`, plus notes.

    This exists because `max_model_len` is an engine-construction argument, so the context has
    to be sized *before* there is an engine to ask — and the only thing that knows how many
    tokens a prompt is, is the tokenizer that will encode it. Measuring here rather than
    trusting `target_tokens` is what stops the verifier rejecting its own prompts: that field
    is the length a sample was *generated to target*, and `long-needle-8192-0` encodes to 9328.

    The chat template is applied here too, not after the engine is built, because it changes
    the token count — measuring the untemplated prompt would reintroduce the same error in
    miniature.
    """
    from transformers import AutoTokenizer

    notes: list[str] = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template = chat_template_path.read_text() if chat_template_path.is_file() else None
    if template is None:
        notes.append(f"no chat template at {chat_template_path}; `role: user` cases sent raw")

    out: list[tuple[str, int]] = []
    for case in case_set.cases:
        text = case.prompt
        if case.role == "user" and template is not None:
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": case.prompt}],
                    chat_template=template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                notes.append(f"{case.id}: chat template failed ({exc}); sent raw")
        out.append((text, len(tokenizer(text)["input_ids"])))
    return out, notes
