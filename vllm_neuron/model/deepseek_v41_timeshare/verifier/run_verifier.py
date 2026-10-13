#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verifier for the DeepSeek-V4.1-Flash (timeshare) port.

The onboarding guide's smoke test with more inputs and a judge. Two pass
conditions, both required:

  1. **Nothing raised.** Engine construction and all 4 generations complete
     without an exception.
  2. **Nothing gibberish.** A Sonnet 5 judge at medium effort reads every
     completion and finds each one reasonable rather than gibberish.

Exits 0 only when both hold. Prints `##verifier[passed=1]` on success and
`##verifier[passed=0]` otherwise, so a queue can grep for it.

It also **measures**, which does not bear on either condition. Each phase
captures two NRT device profiles and reports on-device TTFT and time-per-output-token
as `##autohelix[...]` metrics — see `_measure_phase` and `device_time.py`. The
measurement runs after the generation it is measuring beside, is wrapped so
nothing it does can fail the run, and is skipped entirely under
`--no-device-time`.

Inputs: 4 prompts — TranSonic's `short-prose` and its 8192-token
`long-needle-8192-0`, `"Hello, what is your name?"` and `"What is the capital
of France?"`. 32 tokens each, greedy. The TranSonic sets are copied
into `inputs/` beside this file and read via a path anchored to this package,
so the verifier needs nothing outside its own directory; `prompts.SELECTED_IDS`
picks which rows of them to use.

    python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.run_verifier \\
        --model <checkpoint-dir>

`--model` is required and has no default: the checkpoint lives outside this
repository, so naming a location here would bake an outside path into the
verifier. Every path the verifier *owns* resolves inside vllm-neuron.

NOTE: this runs the model, so it needs a Trainium device. It is the one piece
of this port that must not be run while another job holds the device.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

#: Per the spec: 32 tokens, greedy so a rerun is comparable.
MAX_TOKENS = 32

#: Slack above `longest prompt + MAX_TOKENS`, so a context sized to the exact sum does not
#: leave the scheduler with zero room. Small, because context costs KV cache, and on this
#: port KV cache competes with the graph set for the same HBM.
CONTEXT_HEADROOM = 64
TEMPERATURE = 0.0

#: Anchored to this file's own location: no hardcoded absolute path to break if
#: the repo moves, no cwd-relative path to break depending on where the
#: verifier is launched from, and nothing resolving outside this package.
_VERIFIER_DIR = Path(__file__).resolve().parent
_PORT_DIR = _VERIFIER_DIR.parent

CHAT_TEMPLATE = _PORT_DIR / "chat_template.jinja"
DEFAULT_REPORT = _VERIFIER_DIR / "verifier_report.json"

#: Tokens generated in the decode measurement window. Bounded from above, not
#: below: a forward of this model is ~1400 NEFF executions and the runtime's
#: notification buffer drops events past ~6700, so a window has room for about
#: four forwards. At 1-3 tokens a step, 4 tokens is two or three decode steps
#: inside that budget — enough that the per-step figure is not one step's
#: noise, few enough that the trace survives.
MEASURE_TOKENS = 3

#: The throwaway prompt in the prefill window, whose single forward is the one
#: `delay_iterations` skips. Short enough to be one forward and to cost little,
#: and never judged — it is not one of the verifier's cases.
WARMUP_PROMPT = "Hello."

#: Where the NRT profiles land. The one path here that is deliberately *not*
#: inside the repository: an NTFF is hundreds of MB of one host's trace, and the
#: loop runs this verifier inside a git worktree it then diffs. Per-pid so two
#: runs cannot read each other's artifacts.
PROFILE_DIR = Path(tempfile.gettempdir()) / "deepseek_v41_timeshare-profiles"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="checkpoint directory (required; no default, so no path outside "
        "this repository is baked in). Use the local copy rather than "
        "re-downloading — see the port's references/INDEX.md.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="engine tensor-parallel degree. Defaults to the port's own "
        "`config.TENSOR_PARALLEL_SIZE`, so the parallelism is the "
        "implementation's choice rather than the verifier's; pass this only to "
        "override it for one run.",
    )
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="engine context length. Default: measured — the longest selected prompt as "
        f"the real tokenizer encodes it, plus {MAX_TOKENS} generated tokens plus "
        f"{CONTEXT_HEADROOM} headroom. Pass this only to override, and note that a value "
        "too small drops cases rather than failing, which is recorded in the report's notes.",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="default: the largest supported KV segment size below the resolved max_model_len, "
        "which is what auto-enables segmented prefill. A value >= max_model_len is single-shot "
        "prefill, which this build only accepts with prefix caching off — the verifier turns it "
        "off for you in that case.",
    )
    p.add_argument(
        "--chat-template",
        default=CHAT_TEMPLATE,
        help="applied to cases with role=user (default: the port's bundled template)",
    )
    p.add_argument(
        "--report",
        default=None,
        help=f"write the full result JSON here (default: {DEFAULT_REPORT})",
    )
    p.add_argument(
        "--no-device-time",
        dest="device_time",
        action="store_false",
        default=True,
        help="skip the on-device time measurement. The two extra profiled "
        "windows per phase cost a prefill plus a few decode steps each, and "
        "arm the runtime's inspect path on rank 0; pass this to run the "
        "correctness check alone.",
    )
    p.add_argument(
        "--profile-dir",
        default=None,
        help=f"where NRT profiles and their ingests go (default: a per-pid "
        f"directory under {PROFILE_DIR})",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="run generations and report them, but do not call the judge. The "
        "run then cannot pass — use it to inspect raw output.",
    )
    return p.parse_args(argv)


def _phases(cases, kept, args) -> list[dict]:
    """Split the cases into one-engine-each groups: the longest alone, the rest together.

    The longest case is the one whose context and graph set dominate, and the one whose
    failures have historically taken the others down with it. Everything else shares a phase so
    the multi-sequence paths stay covered -- iteration 4's three state bugs (`SharedAttentionState`
    per batch, the n-gram hash cache shared between sequences, the indexer's mask under chunked
    prefill) were all invisible at `max_num_seqs=1`.

    An explicit `--max-model-len` is an override of the whole run, so it collapses to a single
    phase: the caller asked for a specific configuration and splitting it would silently not
    deliver it. `--max-num-seqs` is read the same way when it is distinguishable from the
    default.
    """
    pairs = list(zip(cases, kept))
    overridden = args.max_model_len is not None or getattr(
        args, "max_num_seqs_explicit", False
    )
    if overridden:
        return [{
            "name": "all",
            "metric": "all",
            "cases": pairs,
            "max_model_len": None,          # resolved by the caller's own arguments
            "max_num_seqs": args.max_num_seqs,
        }]
    if len(pairs) < 2:
        return [{"name": "all", "metric": "all", "cases": pairs,
                 "max_model_len": None, "max_num_seqs": args.max_num_seqs}]

    longest = max(pairs, key=lambda p: p[1][1])
    rest = [p for p in pairs if p is not longest]

    def context_for(group) -> int:
        """The phase's context: what its own prompts need, floored so a segment fits under it.

        Without the floor a phase of 37-to-158-token prompts asks for 254, no
        `SUPPORTED_KV_SEGMENT_SIZES` entry is below that, and it silently becomes a single-shot
        prefill with prefix caching off -- a path the gate's own default never takes. The floor
        keeps every phase on segmented prefill, which is the one the port is built for and the
        one its graphs are cached for.
        """
        from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

        need = max(n for _, (_, n) in group) + MAX_TOKENS + CONTEXT_HEADROOM
        floor = min(SUPPORTED_KV_SEGMENT_SIZES) + MAX_TOKENS + CONTEXT_HEADROOM
        return max(need, floor)

    # `metric` is the suffix the phase's `##autohelix[...]` names carry: the two
    # phases are exactly the long input and the short ones, which is the split
    # the measurement wants to report anyway.
    return [
        {"name": longest[0].id, "metric": "long", "cases": [longest],
         "max_model_len": context_for([longest]), "max_num_seqs": 1},
        {"name": "short", "metric": "short", "cases": rest,
         "max_model_len": context_for(rest), "max_num_seqs": args.max_num_seqs},
    ]


def _prefill_forwards(cases, max_num_batched_tokens: int | None) -> int:
    """How many forwards the prefills of these prompts take.

    `ceil(prompt / max_num_batched_tokens)` per prompt, which is what the
    scheduler has done on every run observed here — each prompt's prefill
    arrives in its own step or chunk. It matters because it is what says
    whether a recorded forward was a prefill or a decode step.
    """
    mnbt = max_num_batched_tokens or 0
    total = 0
    for _, (_, n_prompt) in cases:
        total += max(1, -(-n_prompt // mnbt)) if mnbt else 1
    return total


def _new_measurement(phase: dict) -> dict:
    return {
        "phase": phase["name"],
        "metric": phase["metric"],
        "measure_tokens": MEASURE_TOKENS,
        "captures": {},
        "sessions": {},
        "windows": {},
        "notes": [],
    }


def _session(llm, out: dict, profile_dir: Path, label: str, kind: str | None,
             fn) -> None:
    """Record one window, and note what it is expected to hold.

    `kind` is `"prefill"` or `"decode"` where the window's *construction*
    guarantees it — the forward the window records is positioned by
    `delay_iterations`, so a generate whose first forward is the throwaway one
    records a prefill and a generate whose first forward is a cached re-prefill
    records decode steps. `None` means the window wraps something whose
    schedule is not ours to assume (the run's own 32-token generate), and
    `_resolve_measurement` identifies it from the trace instead.

    Nothing here swallows a failure of `fn`: the measurement's windows are
    extra generates, and one of those failing is worth seeing.
    """
    from . import device_time

    before = device_time.engine_counters(llm)
    try:
        result, capture = device_time.measure(llm, profile_dir, label, fn)
    except Exception:
        out["notes"].append(f"window {label} raised:\n{traceback.format_exc()}")
        raise
    after = device_time.engine_counters(llm)
    forwards = (
        int(after["steps"] - before["steps"])
        if before is not None and after is not None
        else None
    )
    tokens = (
        sum(len(o.outputs[0].token_ids) for o in result)
        if isinstance(result, list)
        else None
    )
    recorded = device_time.recorded_count(forwards)
    out["captures"][label] = capture
    out["sessions"][label] = {
        "label": label,
        "kind": kind,
        "forwards": forwards,
        "recorded": recorded,
        "tokens": tokens,
    }
    print(f"verifier: window {label} — {forwards} forward(s), recorded "
          f"{recorded} ({kind or 'kind from the trace'}), {tokens} token(s), "
          f"{len(capture.ntffs)} profile(s)")
    for note in capture.notes:
        print(f"  note: {note}")


def _measure_phase(llm, phase: dict, out: dict, profile_dir: Path) -> dict:
    """Capture the two extra windows the phase's figures need.

    The phase's own 32-token generate is already one recorded window — see
    `_run_phase`, which wraps it — and this adds the two whose *kind* is known
    by construction, both in the same engine with the same configuration:

      `cold-prefill`  a throwaway prompt (one forward, the one the window
                      skips), the prefix cache reset, then the phase's longest
                      prompt for one token. What gets recorded is that cold
                      prefill, alone.
      `cold-decode`   the cache reset again, then the same prompt for
                      `MEASURE_TOKENS`. Its first forward is that prompt's
                      prefill, which is the one skipped, so what gets recorded
                      is decode steps, alone.

    The decode window resets the cache rather than reusing what the window
    before it left warm, even though a warm re-prefill would be cheaper. A warm
    one is also a path the gate never takes — the run's own prefills are all
    cold — and taking it on the 2008-token needle killed the engine with
    `RuntimeError: nrta status=1006`, an out-of-bounds execution inside the
    model's forward. Measuring must not go anywhere the thing being measured
    does not.

    These two also serve as the fingerprints that say what the 32-token
    generate's own window caught, which is not something to assume: this port
    forces `max_num_seqs` to 1 above one rank, so a phase's prompts run one
    after another rather than together, and "the first N forwards are the
    prefills" is false for it.

    The ingest is *not* done here — it peaks around 17 GiB of host memory, which
    is not something to ask for while four ranks hold an engine. This collects
    the captures; `_resolve_measurement` reads them after the teardown.
    """
    from vllm import SamplingParams

    cases = phase["cases"]
    pick = max(range(len(cases)), key=lambda i: cases[i][1][1])
    case, (text, n_prompt) = cases[pick]
    out["measured_case"] = case.id
    out["measured_prompt_tokens"] = n_prompt
    print(f"verifier: measuring on {case.id} ({n_prompt} prompt tokens)")

    one_token = SamplingParams(max_tokens=1, temperature=TEMPERATURE)

    def cold_prefill():
        # Forward 1 is the throwaway the window's delay is there to skip —
        # short, so it costs little and is one forward. Then the prompt twice,
        # with the cache reset before each so both are as cold as the 32-token
        # run's prefill was: two forwards recorded, because one prefill of this
        # port is a noisy sample (two windows of identical work measured 23.8 s
        # and 29.5 s).
        llm.generate([WARMUP_PROMPT], one_token)
        llm.reset_prefix_cache()
        first = llm.generate([text], one_token)
        llm.reset_prefix_cache()
        return first + llm.generate([text], one_token)

    try:
        _session(llm, out, profile_dir, f"{phase['metric']}-cold-prefill",
                 "prefill", cold_prefill)
    except Exception:  # noqa: BLE001 - a measurement is not a pass condition
        return out

    def cold_decode():
        llm.reset_prefix_cache()
        return llm.generate(
            [text],
            SamplingParams(max_tokens=MEASURE_TOKENS, temperature=TEMPERATURE),
        )

    try:
        _session(llm, out, profile_dir, f"{phase['metric']}-cold-decode",
                 "decode", cold_decode)
    except Exception:  # noqa: BLE001
        return out
    return out


def _dominant_neff(window) -> str | None:
    """The graph a window spent most of its device time in.

    A prefill forward and a decode forward run different graphs for the same
    layer — different token-dimension buckets — and on this port one graph per
    layer dominates either way, so this is a usable fingerprint of which kind
    of forward a trace holds.
    """
    if not getattr(window, "per_neff", None):
        return None
    return max(window.per_neff.items(), key=lambda kv: kv[1]["busy_s"])[0]


def _resolve_measurement(raw: dict, correctness: dict) -> dict:
    """Read the captured windows and turn them into TTFT, step time and TPOT.

    Runs after the engine is gone, because the ingest is memory-hungry.

    Every recorded forward is pooled by kind — each window knows whether it
    caught prefills or decode steps, from the engine's forward count and the
    phase's prefill-chunk count — so TTFT is the mean device time of a prefill
    forward and the step figure is the mean of a decode forward. A window that
    caught one of each is reported but not pooled: this port's execution spans
    include waiting for host-streamed weights, so a mixed window's total cannot
    be split by subtracting a figure measured somewhere else.

    `correctness` carries the counters differenced around the phase's real
    32-token generate, which is where tokens-per-step comes from: with
    speculation a step emits `1 + accepted` tokens, so one step's device time is
    not one token's. Measuring that ratio on the generate that actually ran,
    rather than on a 3-token window whose last step the sampler truncates,
    keeps it honest.
    """
    from . import device_time

    out = {k: v for k, v in raw.items() if k != "captures"}
    out.update(ttft_ms=None, decode_step_ms=None, tpot_ms=None, total_ms=None,
               tokens_per_step=None, correctness=correctness)
    pooled: dict[str, dict[str, float]] = {
        "prefill": {"busy_s": 0.0, "forwards": 0},
        "decode": {"busy_s": 0.0, "forwards": 0},
    }

    windows: dict[str, object] = {}
    for label, capture in raw.get("captures", {}).items():
        window = device_time.summarize(capture)
        windows[label] = window
        session = raw["sessions"].get(label, {})
        entry = window.as_dict()
        entry.update(
            forwards=session.get("forwards"),
            recorded=session.get("recorded"),
            tokens=session.get("tokens"),
            kind=session.get("kind"),
            dominant_neff=_dominant_neff(window),
        )
        out["windows"][label] = entry
        busy = entry["device_busy_ms"]
        print(f"verifier: [{raw['metric']}] {label}: "
              + ("not measured" if busy is None else f"{busy:.1f} ms on device")
              + f" over {entry['executions']} execution(s) in "
                f"{entry['recorded']} forward(s) of "
                f"{session.get('kind') or 'unknown kind'}"
              + (" (DROPPED)" if entry["dropped"] else ""))
        for note in entry["notes"]:
            print(f"  note: {note}")

    # The two purpose-built windows hold a known kind of forward, so the graph
    # each one spent most of its device time in is a fingerprint of that kind —
    # a prefill and a decode forward run different graphs for the same layer,
    # being different token-dimension buckets. That is what identifies the
    # third window, the one around the 32-token generate, whose schedule is not
    # ours to assume. A window matching neither fingerprint is reported and not
    # pooled rather than guessed at.
    reference = {
        session["kind"]: out["windows"][label]["dominant_neff"]
        for label, session in raw.get("sessions", {}).items()
        if session.get("kind") and out["windows"].get(label, {}).get("dominant_neff")
    }

    for label in raw.get("captures", {}):
        window = windows[label]
        entry = out["windows"][label]
        session = raw["sessions"].get(label) or {}
        recorded = session.get("recorded") or 0
        entry["pooled"] = False
        if not window.ok or not recorded:
            continue
        if window.dropped:
            out["notes"].append(
                f"{label} lost notifications, so its device time is a lower "
                "bound and it is reported but not used"
            )
            continue

        kind = session.get("kind")
        if kind is None:
            dominant = entry["dominant_neff"]
            matches = [k for k, ref in reference.items() if ref == dominant]
            if len(matches) != 1:
                out["notes"].append(
                    f"{label}'s most expensive graph ({dominant}) matches "
                    f"{matches or 'neither'} of the reference windows, so what "
                    "it recorded cannot be told apart; reported, not used"
                )
                continue
            kind = matches[0]
            entry["kind"] = kind
            entry["kind_from"] = "trace fingerprint"

        entry["pooled"] = True
        bucket = pooled[kind]
        bucket["busy_s"] += window.device_busy_s
        bucket["forwards"] += recorded

    out["pooled"] = {
        kind: {"device_busy_ms": v["busy_s"] * 1e3, "forwards": v["forwards"]}
        for kind, v in pooled.items()
    }
    if pooled["prefill"]["forwards"]:
        out["ttft_ms"] = (
            pooled["prefill"]["busy_s"] * 1e3 / pooled["prefill"]["forwards"]
        )
    if pooled["decode"]["forwards"]:
        out["decode_steps"] = pooled["decode"]["forwards"]
        out["decode_step_ms"] = (
            pooled["decode"]["busy_s"] * 1e3 / pooled["decode"]["forwards"]
        )

    tps = correctness.get("tokens_per_step")
    if tps and out["decode_step_ms"] is not None:
        out["tokens_per_step"] = tps
        out["tpot_ms"] = out["decode_step_ms"] / tps
    elif out["decode_step_ms"] is not None:
        # No ratio to divide by: one token a step is the floor, so this becomes
        # an upper bound on TPOT rather than a measurement. Said out loud.
        out["tokens_per_step"] = 1.0
        out["tpot_ms"] = out["decode_step_ms"]
        out["notes"].append(
            "tokens per step could not be measured, so TPOT is reported at one "
            "token per step — an upper bound, since speculation emits more"
        )

    if out["ttft_ms"] is not None and out["tpot_ms"] is not None:
        out["total_ms"] = out["ttft_ms"] + (MAX_TOKENS - 1) * out["tpot_ms"]
    return out


def _correctness_counters(before, after, phase: dict,
                          max_num_seqs: int | None,
                          completion_tokens: int | None = None) -> dict:
    """Steps and tokens-per-step for the phase's own 32-token generate.

    This is the one figure the profiles cannot supply: with speculation a decode
    step emits `1 + accepted` tokens, so a step's device time is not a token's.
    Measured on the generate that actually ran — `vllm:iteration_tokens_total`'s
    count is forwards, `vllm:generation_tokens` is tokens — rather than on a
    3-token window whose last step the sampler truncates.

    Two things have to be right for the division. Prefill forwards are taken as
    one per chunk per prompt (`ceil(prompt / max_num_batched_tokens)`), which is
    what the scheduler has done on every run observed here. And a decode step
    advances `min(sequences, max_num_seqs)` of them at once — `max_num_seqs` as
    the *engine* resolved it, since this port forces it to 1 above one rank, in
    which case the phase's prompts run one after another and each decode step
    belongs to a single sequence.
    """
    out: dict = {"steps": None, "generation_tokens": None, "prefill_forwards": None,
                 "decode_steps": None, "tokens_per_step": None,
                 "concurrency": None, "notes": []}
    if before is None or after is None:
        out["notes"].append(
            "the engine's step counter was unavailable, so tokens per step "
            "could not be measured"
        )
        return out

    n_seqs = len(phase["cases"])
    chunks = _prefill_forwards(phase["cases"], phase.get("max_num_batched_tokens"))
    concurrency = min(n_seqs, max_num_seqs) if max_num_seqs else n_seqs

    steps = int(after["steps"] - before["steps"])
    counter_gen = int(after["generation_tokens"] - before["generation_tokens"])
    # The completions themselves, not `vllm:generation_tokens`. Under
    # speculation that counter follows the sampler's output width — `1 + k` a
    # verify step whatever the sampler kept — so on the needle it read 32 for a
    # 19-token completion, which would have made tokens-per-step 6.2 against a
    # ceiling of 1 + k = 6. The completion lengths cannot drift that way, and
    # they agree with the drafter's own acceptance counters.
    gen = completion_tokens if completion_tokens is not None else counter_gen
    decode_steps = steps - chunks
    out.update(steps=steps, generation_tokens=gen,
               counter_generation_tokens=counter_gen, prefill_forwards=chunks,
               decode_steps=decode_steps, sequences=n_seqs,
               concurrency=concurrency, engine_max_num_seqs=max_num_seqs)
    if decode_steps > 0 and gen > n_seqs:
        # The first token of each sequence comes out of its prefill, so the
        # decode steps are responsible for the rest.
        out["tokens_per_step"] = (gen - n_seqs) / decode_steps / concurrency
    else:
        out["notes"].append(
            f"{steps} forward(s) and {gen} generated token(s) over {n_seqs} "
            f"sequence(s) with {chunks} prefill chunk(s) leaves no decode step "
            "to measure"
        )
    return out


def _run_phase(phase: dict, args, tensor_parallel_size: int,
               profile_dir: Path | None) -> list[dict]:
    """One engine, one `generate`, then tear it down so its HBM comes back."""
    import gc

    from vllm import LLM, SamplingParams
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    max_model_len = phase["max_model_len"] or args.max_model_len
    segments = sorted(s for s in SUPPORTED_KV_SEGMENT_SIZES if s < max_model_len)
    if args.max_num_batched_tokens is not None:
        mnbt = args.max_num_batched_tokens
    else:
        mnbt = segments[-1] if segments else max_model_len
    apc = mnbt < max_model_len

    print(f"\nverifier: phase {phase['name']} — max_model_len={max_model_len}, "
          f"max_num_batched_tokens={mnbt}, max_num_seqs={phase['max_num_seqs']}, "
          f"prefix caching {'on' if apc else 'off'}")
    phase.update(max_model_len=max_model_len, max_num_batched_tokens=mnbt,
                 enable_prefix_caching=apc)

    # Arming the profiler is an engine-construction argument, so it goes in here
    # even though nothing is captured until `_measure_phase` starts a session.
    # `NeuronProfiler` is created lazily on the first `start_profile`.
    profiler_kwargs: dict = {}
    if profile_dir is not None:
        from . import device_time

        profiler_kwargs = device_time.profiler_engine_kwargs(profile_dir)

    llm = LLM(
        model=args.model,
        max_num_seqs=phase["max_num_seqs"],
        max_model_len=max_model_len,
        max_num_batched_tokens=mnbt,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=apc,
        **profiler_kwargs,
    )
    try:
        texts = [text for _, (text, _) in phase["cases"]]
        from . import device_time

        # The run's own generate: every case, `MAX_TOKENS` tokens, greedy —
        # unchanged by any of the measurement below. What the measurement adds
        # is a profile session *around* it and the engine's counters either
        # side, neither of which changes what is scheduled or sampled: the
        # session records two forwards and stops itself, and the completions
        # the judge reads are the same completions.
        measurement = None
        before = after = None
        generate = lambda: llm.generate(  # noqa: E731 - handed to `measure`
            texts, SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        )
        if profile_dir is None:
            outputs = generate()
        else:
            measurement = _new_measurement(phase)
            label = f"{phase['metric']}-generate"
            before = device_time.engine_counters(llm)
            outputs, capture = device_time.measure(
                llm, profile_dir, label, generate
            )
            after = device_time.engine_counters(llm)
            forwards = (
                int(after["steps"] - before["steps"])
                if before is not None and after is not None
                else None
            )
            measurement["captures"][label] = capture
            measurement["sessions"][label] = {
                "label": label, "kind": None, "forwards": forwards,
                "recorded": device_time.recorded_count(forwards),
                "tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
            }
            print(f"\nverifier: the {MAX_TOKENS}-token generate ran {forwards} "
                  f"forward(s); the profile recorded "
                  f"{measurement['sessions'][label]['recorded']} of them "
                  f"(kind read from the trace)")
            for note in capture.notes:
                print(f"  note: {note}")
        out = []
        for (case, (_, n_prompt)), o in zip(phase["cases"], outputs):
            completion = o.outputs[0]
            out.append({
                "id": case.id,
                "phase": phase["name"],
                "source": case.source,
                "role": case.role,
                "expected_substring": case.expected_substring,
                "prompt_excerpt": case.excerpt(),
                "prompt_chars": len(case.prompt),
                "prompt_tokens": n_prompt,
                "text": completion.text,
                "num_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            })

        # After the generation that decides the run, never before: the
        # measurement opens a profile session on rank 0 and runs the prompts
        # twice more, and none of that may stand between a working port and a
        # recorded completion.
        if profile_dir is not None and measurement is not None:
            print(f"\nverifier: measuring on-device time for phase {phase['name']}")
            try:
                phase["_measurement"] = _measure_phase(
                    llm, phase, measurement, profile_dir
                )
                phase["_measurement"]["correctness"] = _correctness_counters(
                    before, after, phase, device_time.engine_max_num_seqs(llm),
                    completion_tokens=sum(
                        len(o.outputs[0].token_ids) for o in outputs
                    ),
                )
            except Exception:  # noqa: BLE001 - belt and braces: see the docstring
                phase["device_time"] = {
                    "phase": phase["name"], "metric": phase["metric"],
                    "ttft_ms": None, "decode_step_ms": None, "tpot_ms": None,
                    "total_ms": None, "windows": {},
                    "notes": [f"measurement raised:\n{traceback.format_exc()}"],
                }
                print("verifier: measurement failed; the run is unaffected",
                      file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        return out
    finally:
        # `del` then collect: the engine holds the device until its process exits, and vLLM v1
        # runs EngineCore out of process, so dropping the last reference is what releases HBM.
        # Without this the next phase builds its engine alongside this one's resident graphs,
        # which is the accumulation the split exists to avoid.
        del llm
        gc.collect()
        print(f"verifier: phase {phase['name']} torn down")


def _device_time_report(phases: list[dict]) -> dict | None:
    """Collect each phase's measurement, keyed by its metric suffix."""
    measured = {
        p["metric"]: p["device_time"] for p in phases if p.get("device_time")
    }
    return measured or None


def _print_device_time(measured: dict | None) -> None:
    """Print the measurement, and the `##autohelix[...]` metrics for a loop.

    Four metrics per phase. The first two are measured, the last two derived
    from them, and all four are *device* time — no host transfer, no host
    runtime, no setup:

      `device_ttft_ms_<phase>`        the prefill forward: time to first token.
      `device_step_ms_<phase>`        one decode forward.
      `device_tpot_ms_<phase>`        per output token, i.e. the step divided by
                                      the tokens a step emits (speculation
                                      makes that more than one).
      `device_total_ms_<phase>`       what the verifier's own 32-token generate
                                      would cost on device at those rates.

    Deliberately not wall clock. The target is a 16-device instance that keeps
    the weights resident and builds its graphs once; what survives that move is
    the device's own execution time, so that is what is recorded as the
    reference point. On this host the difference is not a detail — a prefill
    window measured 62 s on device inside a 372 s wall-clock span.
    """
    if not measured:
        return
    print("\n" + "=" * 72)
    print("ON-DEVICE TIME — NRT device profile, rank 0, host transfer/runtime "
          "excluded")
    for suffix, m in measured.items():
        print(f"\n  [{suffix}] phase {m['phase']}, measured on "
              f"{m.get('measured_case')} "
              f"({m.get('measured_prompt_tokens')} prompt tokens)")
        pooled = m.get("pooled") or {}
        if m.get("ttft_ms") is not None:
            n = (pooled.get("prefill") or {}).get("forwards")
            print(f"    TTFT        {m['ttft_ms']:10.1f} ms  "
                  f"mean of {n} prefill forward(s)")
        if m.get("decode_step_ms") is not None:
            print(f"    decode step {m['decode_step_ms']:10.1f} ms  "
                  f"mean of {m.get('decode_steps')} decode forward(s)")
        if m.get("tokens_per_step"):
            print(f"    tokens/step {m['tokens_per_step']:10.2f}     "
                  "measured on the 32-token generate")
        if m.get("tpot_ms") is not None:
            print(f"    TPOT        {m['tpot_ms']:10.1f} ms  per output token")
        if m.get("total_ms") is not None:
            print(f"    {MAX_TOKENS} tokens  {m['total_ms'] / 1000:10.1f} s   "
                  "of device time at those rates")

        for label, w in (m.get("windows") or {}).items():
            busy = w.get("device_busy_ms")
            print(f"      {label:<22} "
                  + ("not measured" if busy is None else f"{busy:9.1f} ms")
                  + f"  {w.get('recorded')} of {w.get('forwards')} forward(s), "
                    f"{w.get('kind') or 'unknown'}, "
                    f"{w.get('executions')} execution(s)"
                  + ("" if w.get("pooled", False) else "  [not pooled]")
                  + (" DROPPED" if w.get("dropped") else ""))
        prefill_window = next(
            (w for label, w in (m.get("windows") or {}).items()
             if w.get("kind") == "prefill" and w.get("per_neff")),
            None,
        )
        top = list((prefill_window or {}).get("per_neff", {}).items())[:5]
        if top:
            print("      the prefill forward's most expensive graphs:")
            for name, v in top:
                print(f"        {v['device_busy_ms']:9.1f} ms  x{v['count']:<5} {name}")
        c = m.get("correctness") or {}
        if c.get("steps"):
            print(f"      the 32-token generate: {c['steps']} forward(s), "
                  f"{c.get('prefill_forwards')} prefill chunk(s), "
                  f"{c.get('decode_steps')} decode step(s), "
                  f"{c.get('generation_tokens')} token(s) over "
                  f"{c.get('sequences')} sequence(s)")
        for note in (m.get("notes") or []) + (c.get("notes") or []):
            print(f"    note: {note}")
    print("=" * 72)

    for suffix, m in measured.items():
        for key, value in (
            (f"device_ttft_ms_{suffix}", m.get("ttft_ms")),
            (f"device_step_ms_{suffix}", m.get("decode_step_ms")),
            (f"device_tpot_ms_{suffix}", m.get("tpot_ms")),
            (f"device_total_ms_{suffix}", m.get("total_ms")),
        ):
            if value is not None:
                print(f"##autohelix[{key}={value:.6g}]")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # The port owns its parallelism (see `config.TENSOR_PARALLEL_SIZE`). Read at
    # run time, not import time, so it tracks whatever the implementation
    # currently says. A nonsensical value is rejected here rather than becoming
    # an obscure engine failure several minutes in.
    tensor_parallel_size = args.tensor_parallel_size
    source = "--tensor-parallel-size"
    if tensor_parallel_size is None:
        from ..config import TENSOR_PARALLEL_SIZE

        tensor_parallel_size = TENSOR_PARALLEL_SIZE
        source = "config.TENSOR_PARALLEL_SIZE"
    if not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 1:
        print(
            f"verifier: {source} = {tensor_parallel_size!r}, which is not a positive "
            "integer",
            file=sys.stderr,
        )
        print("##verifier[passed=0]")
        return 1
    print(f"verifier: tensor_parallel_size={tensor_parallel_size} (from {source})")

    from .prompts import build_cases, render

    case_set = build_cases()

    # Render and measure with the real tokenizer *before* the engine exists, because
    # `max_model_len` is an engine-construction argument and the only thing that knows how long
    # a prompt is, is the tokenizer that will encode it.
    #
    # The earlier version sized the context from `target_tokens` and had vLLM reject its own
    # prompt: `long-needle-8192-0` is named for the 8192 tokens it was *generated to target* and
    # encodes to 9328 here, so a `8192 + MAX_TOKENS` context was 1136 short, the gate failed
    # with `VLLMValidationError` before a single token, and the `verifier` constraint was
    # unsatisfiable no matter what the port did. Nominal lengths are not lengths.
    prompts, render_notes = render(case_set, args.model, Path(args.chat_template))
    case_set.notes.extend(render_notes)
    lengths = [n for _, n in prompts]

    if args.max_model_len is not None:
        max_model_len = args.max_model_len
    else:
        max_model_len = (max(lengths) if lengths else 0) + MAX_TOKENS + CONTEXT_HEADROOM
        print(
            f"verifier: max_model_len={max_model_len} "
            f"(longest prompt {max(lengths) if lengths else 0} tok + {MAX_TOKENS} generated "
            f"+ {CONTEXT_HEADROOM} headroom)"
        )

    # Only now can a case be dropped honestly, against the budget it actually has to fit.
    budget = max_model_len - MAX_TOKENS
    kept: list[tuple[str, int]] = []
    kept_cases = []
    for case, (text, n) in zip(list(case_set.cases), prompts):
        if n > budget:
            case_set.notes.append(
                f"skipped {case.id}: {n} prompt tokens exceeds "
                f"max_model_len {max_model_len} - {MAX_TOKENS} generated = {budget}"
            )
            continue
        kept.append((text, n))
        kept_cases.append(case)
    case_set.cases = kept_cases

    # Chunked prefill, at the largest segment this context can use.
    #
    # The default used to be `max_model_len` — single-shot prefill — which this build rejects
    # before any model code runs, so the constraint was unsatisfiable regardless of the port for
    # a second reason after the nominal-length one above. Two rules collide:
    # `resolve_segmented_prefill_config` reads `max_num_batched_tokens >= max_model_len` as
    # single-shot and returns no `kv_segment_size_buckets`, and `neuron_model_runner` requires
    # those buckets whenever prefix caching is on — which vLLM 0.24 defaults to. The raise is in
    # `NeuronModelRunner.__init__`, upstream of `load_model`, so no model code could have
    # intervened. Every invocation in the repo's own `docs/` passes one or the other.
    #
    # `SUPPORTED_KV_SEGMENT_SIZES` rather than a literal 8192: the segment has to be a size the
    # attention kernel has buckets for, and strictly below `max_model_len` or it is single-shot
    # again — which is why this is not simply `or 8192`. A `--max-model-len 544` run would then
    # ask for a segment larger than its own context and fail the same way.
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    segments = sorted(s for s in SUPPORTED_KV_SEGMENT_SIZES if s < max_model_len)
    if args.max_num_batched_tokens is not None:
        max_num_batched_tokens = args.max_num_batched_tokens
    else:
        max_num_batched_tokens = segments[-1] if segments else max_model_len

    # Prefix caching needs segmented prefill. When the configuration is single-shot anyway —
    # the caller asked for it, or the context is below the smallest segment — turn APC off
    # instead of letting engine construction raise. This is the remedy the error names first.
    enable_prefix_caching = max_num_batched_tokens < max_model_len
    print(
        f"verifier: max_num_batched_tokens={max_num_batched_tokens} "
        f"({'segmented' if enable_prefix_caching else 'single-shot'} prefill, "
        f"prefix caching {'on' if enable_prefix_caching else 'off'})"
    )

    print(f"verifier: {len(case_set)} case(s)")
    for note in case_set.notes:
        print(f"  note: {note}")

    report: dict = {
        "model": args.model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        # In the report because `runs-on-trainium`'s judge reads it, and because which prefill
        # mode ran is the difference between a 9k-token segment and a short one.
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": enable_prefix_caching,
        "prompt_tokens": [n for _, n in kept],
        "notes": list(case_set.notes),
        "cases": [],
        "execution_error": None,
        "judge": None,
        "passed": False,
    }
    report_path = Path(args.report) if args.report else DEFAULT_REPORT

    def _write() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    # ── Condition 1: nothing raised ───────────────────────────────────────
    #
    # One engine per phase, torn down between them, because HBM does not come
    # back any other way. A phase's cost is not its own prompts' alone: every
    # prefill arrives with a different `read_len` as the cache fills, so each
    # keys its own `attn_half` graphs, and a graph stays resident with its code,
    # DMA rings and scratchpad. Measured across four prefills in one engine:
    # 13.3 -> 19.0 GiB of a 24 GiB bank, with the long prompt itself accounting
    # for 13.3 of it. The runtime scrubs HBM when a process attaches (the
    # driver logs `ncdev_hbm_scrub_start` on every `npid_attach`), so a second
    # engine starts from a clean bank rather than inheriting the first one's.
    #
    # It also sizes each phase for its own prompts. `max_model_len` is set by
    # the longest case in the phase and the KV ceiling is one of those per
    # sequence times the batch, so grouping a 2008-token needle with three
    # 37-to-158-token prompts made the short ones reserve the needle's context
    # four times over.
    #
    # And it isolates failure. Previously all four went through one `generate`,
    # so an engine that died on the long prompt returned `cases: []` and the
    # three that work produced no evidence at all.
    phases = _phases(case_set.cases, kept, args)
    report["phases"] = [
        {k: v for k, v in p.items() if k != "cases"} for p in phases
    ]
    print(f"verifier: {len(phases)} phase(s), one engine each")
    for p in phases:
        print(f"  {p['name']}: {len(p['cases'])} case(s), "
              f"max_model_len={p['max_model_len']}, max_num_seqs={p['max_num_seqs']}")

    profile_dir = None
    if args.device_time:
        profile_dir = Path(args.profile_dir or (PROFILE_DIR / str(os.getpid())))
        print(f"verifier: measuring on-device time, profiles under {profile_dir}")

    results: list[dict] = []
    errors: list[str] = []
    for phase in phases:
        try:
            results.extend(_run_phase(phase, args, tensor_parallel_size, profile_dir))
        except Exception:  # noqa: BLE001 - a phase failing is a verifier failure
            errors.append(f"phase {phase['name']}:\n{traceback.format_exc()}")
            print(f"\nverifier: phase {phase['name']} FAILED", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

        # Read the phase's profiles now that its engine is gone: the ingest
        # peaks around 17 GiB of host memory and the next phase is about to
        # build an engine.
        raw = phase.pop("_measurement", None)
        if raw is not None:
            try:
                phase["device_time"] = _resolve_measurement(
                    raw, raw.pop("correctness", {}) or {}
                )
            except Exception:  # noqa: BLE001 - not a pass condition
                phase["device_time"] = {
                    "phase": phase["name"], "metric": phase["metric"],
                    "ttft_ms": None, "decode_step_ms": None, "tpot_ms": None,
                    "total_ms": None, "windows": {},
                    "notes": [f"reading the profiles raised:\n"
                              f"{traceback.format_exc()}"],
                }
                print("verifier: reading the profiles failed; the run is "
                      "unaffected", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    report["cases"] = results
    # Rebuilt after the runs, not only before them: `_run_phase` fills in the
    # resolved engine arguments and the measurement as it goes.
    report["phases"] = [
        {k: v for k, v in p.items() if k not in ("cases", "_measurement")}
        for p in phases
    ]
    report["device_time"] = _device_time_report(phases)
    _print_device_time(report["device_time"])
    if errors:
        report["execution_error"] = "\n\n".join(errors)
        _write()
        print(f"\nverifier: execution FAILED in {len(errors)} of {len(phases)} "
              f"phase(s) — report at {report_path}")
        print(f"  {len(results)} completion(s) from the phases that did run")
        print("##verifier[passed=0]")
        return 1

    print("\n" + "=" * 72)
    for r in results:
        print(f"[{r['id']}]  ({r['num_tokens']} tok, {r['finish_reason']})")
        print(f"  {r['text']!r}")
    print("=" * 72)

    if args.skip_judge:
        report["judge"] = {"skipped": True}
        _write()
        print(f"\nverifier: generations OK, judge skipped — report at {report_path}")
        print("##verifier[passed=0]")
        return 1

    # ── Condition 2: nothing gibberish ────────────────────────────────────
    from .judge import JUDGE_EFFORT, JUDGE_MODEL, judge_outputs

    print(f"\njudging {len(results)} completion(s) with {JUDGE_MODEL} "
          f"(effort {JUDGE_EFFORT})...")
    verdict = judge_outputs(results)
    report["judge"] = {
        "model": JUDGE_MODEL,
        "effort": JUDGE_EFFORT,
        "passed": verdict.passed,
        "error": verdict.error,
        "verdicts": [
            {"id": v.id, "verdict": v.verdict, "reason": v.reason}
            for v in verdict.verdicts
        ],
    }

    for v in verdict.verdicts:
        print(f"  {'ok  ' if v.ok else 'GIB '} {v.id}: {v.reason}")
    if verdict.error:
        print(f"  judge error: {verdict.error}", file=sys.stderr)

    report["passed"] = verdict.passed
    _write()

    print(f"\nverifier: {'PASSED' if verdict.passed else 'FAILED'} — "
          f"report at {report_path}")
    print(f"##verifier[passed={1 if verdict.passed else 0}]")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
