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


def _segment_rows() -> int:
    """The token width of one prefill pass: the smallest KV segment this build has.

    The port prefills in segments, and the segment — not `max_num_batched_tokens`
    — is what one pass through the layers covers. Measured on the device: a
    2008-token prompt with `max_num_batched_tokens=2048` still runs as 512-row
    passes, which the NEFF declares as `hidden_states [1 512 5120]`.
    """
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    return min(SUPPORTED_KV_SEGMENT_SIZES)


def _decode_delay(prompt_tokens: int) -> int:
    """`delay_iterations` for a window that must start on a *decode* pass.

    A prompt of `prompt_tokens` prefills as `ceil(prompt / segment)` forwards, so
    a window recording from forward `segments + 1` is the first one that cannot
    contain a prefill segment. Getting this wrong is not a rounding error: at
    `delay=2` on the 2008-token needle the "decode" window recorded prefill
    segments 2 and 3, and every long-phase decode figure the previous cut
    reported was a prefill segment.
    """
    return max(1, -(-prompt_tokens // _segment_rows())) + 1


def _measure_phase(llm, phase: dict, out: dict, profile_dir: Path) -> dict:
    """Capture the prefill window, on the engine that just ran the phase.

    `delay_iterations=1, max_iterations=3` records the first three forwards of
    whatever this window wraps, so:

      `cold-prefill`  the prefix cache reset, then the phase's longest prompt for
                      one token, three times. What gets recorded is three cold
                      prefill *segments* — the first three of however many the
                      prompt needs — which is what TTFT is built from. Three
                      because the first is a warm-up `_resolve_measurement`
                      drops (measured 1097.9 ms against 291.2 ms for the same
                      512 rows, and the gap is engine time, not DMA), leaving
                      two steady samples rather than one.

    The decode pass cannot come from this engine: it would have to be recorded
    from forward `segments + 1`, and `delay_iterations` is fixed when the engine
    is built. `_measure_decode` builds a second engine for it. The alternative —
    reaching a decode forward early by leaving the prefix cache warm — is the
    path that killed the engine with `RuntimeError: nrta status=1006` on the
    2008-token needle, and it is a path the run itself never takes.

    The ingest is *not* done here: it needs ~20 GiB and the instruction trace,
    and it OOM'd at 22 GiB while four ranks held 85 GiB of the 115 GiB limit.
    This collects the captures; `_resolve_measurement` reads them after teardown.
    """
    from vllm import SamplingParams

    from . import device_time

    cases = phase["cases"]
    pick = max(range(len(cases)), key=lambda i: cases[i][1][1])
    case, (text, n_prompt) = cases[pick]
    out["measured_case"] = case.id
    out["measured_prompt_tokens"] = n_prompt
    out["segment_rows"] = _segment_rows()
    out["decode_delay"] = _decode_delay(n_prompt)
    print(f"verifier: measuring on {case.id} ({n_prompt} prompt tokens, "
          f"{max(1, -(-n_prompt // _segment_rows()))} prefill segment(s))")

    one_token = SamplingParams(max_tokens=1, temperature=TEMPERATURE)

    # One generate per *window-full* of forwards, not one per forward. A
    # 2008-token prompt prefills as four 512-row forwards, so its first generate
    # already closes a three-forward window and every later one is a full cold
    # prefill that no session records — measured at ~4.7 min of device time each
    # on the long phase, for nothing. A short prompt is one forward, so it still
    # needs one generate per recorded pass.
    segments = max(1, -(-n_prompt // _segment_rows()))
    repeats = max(1, -(-device_time.MAX_ITERATIONS // segments))

    def cold_prefill():
        out_passes = []
        for _ in range(repeats):
            llm.reset_prefix_cache()
            out_passes += llm.generate([text], one_token)
        return out_passes

    try:
        _session(llm, out, profile_dir, f"{phase['metric']}-cold-prefill",
                 "prefill", cold_prefill)
    except Exception:  # noqa: BLE001 - a measurement is not a pass condition
        pass
    return out


def _measure_decode(phase: dict, out: dict, args, tensor_parallel_size: int,
                    profile_dir: Path) -> dict:
    """Capture a decode pass, on an engine built to record past the prefill.

    A second engine for one window, because the window has to start at forward
    `segments + 1` and that offset is `delay_iterations`, fixed at engine
    construction. It costs one more engine build and one more prefill per phase,
    which is the price of the long phase having a decode figure at all.
    """
    import gc

    from vllm import LLM, SamplingParams

    from . import device_time

    cases = phase["cases"]
    pick = max(range(len(cases)), key=lambda i: cases[i][1][1])
    _case, (text, n_prompt) = cases[pick]
    delay = _decode_delay(n_prompt)

    kwargs = device_time.profiler_engine_kwargs(
        profile_dir, delay_iterations=delay, max_iterations=2,
    )
    print(f"\nverifier: decode engine for phase {phase['name']} "
          f"(records from forward {delay})")
    llm = LLM(
        model=args.model,
        max_num_seqs=phase["max_num_seqs"],
        max_model_len=phase["max_model_len"],
        max_num_batched_tokens=phase["max_num_batched_tokens"],
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=phase["enable_prefix_caching"],
        **kwargs,
    )
    try:
        _session(
            llm, out, profile_dir, f"{phase['metric']}-cold-decode", "decode",
            lambda: llm.generate(
                [text],
                SamplingParams(max_tokens=MEASURE_TOKENS, temperature=TEMPERATURE),
            ),
        )
    except Exception:  # noqa: BLE001 - a measurement is not a pass condition
        out["notes"].append(f"decode window raised:\n{traceback.format_exc()}")
    finally:
        del llm
        gc.collect()
        print(f"verifier: decode engine for {phase['name']} torn down")
    return out


def _resolve_measurement(raw: dict, correctness: dict, model: str | None = None) -> dict:
    """Read the captured windows and turn them into TTFT, step time and TPOT.

    Runs after the engine is gone, because the ingest needs ~20 GiB and the
    instruction trace it now keeps is what makes the measurement mean anything.

    Every window is split into *passes* — one sweep of the model's layers — and
    each pass is priced as **engine-busy time plus what its collectives actually
    transferred**. Both exclusions matter and both were measured on this port:

      * a per-layer all-reduce's execution span is ~20 s of which 30 ms is
        transfer, the rest being the engine armed and idle while three peer
        ranks catch up;
      * an execution's span also contains the DMA wait for that layer's weights,
        which arrive from NVMe at ~155 MB/s a rank rather than from HBM at
        ~725 GB/s.

    Neither is device work and neither exists on the 16-device target, where the
    weights are resident. Priced this way a 512-row prefill pass is 215-840 ms
    where its span was 12-17 s.

    The pass's *kind* is read off the NEFF — the `hidden_states` width it
    declares, 512 for a KV segment and `1 + k` for a verify step — so "this is
    prefill" is a fact from the compiled graph rather than an inference from
    where the forward sat in the run. That inference is what the previous cut
    got wrong: a 2008-token prompt prefills as four 512-row passes, so its
    second and third forwards are still prefill, and every long-phase figure it
    reported as decode was a prefill segment.
    """
    from . import device_time

    # How many per-layer executions make one pass, from the checkpoint's own
    # `config.json` rather than from a constant: a depth baked in here would be
    # one more thing that breaks on a model this verifier has not seen.
    num_layers = device_time.layer_count(model)
    out = {k: v for k, v in raw.items() if k != "captures"}
    out.update(ttft_ms=None, decode_step_ms=None, tpot_ms=None, total_ms=None,
               tokens_per_step=None, correctness=correctness, passes={})
    pooled: dict[str, list[float]] = {"prefill": [], "decode": []}
    widths: dict[str, int] = {}
    #: The first prefill pass of each window, which is not a steady-state sample
    #: and is excluded from the pool. See the loop below.
    warmup: list[float] = []

    for label, capture in raw.get("captures", {}).items():
        window = device_time.summarize(capture, num_layers=num_layers)
        session = raw["sessions"].get(label, {})
        entry = window.as_dict()
        entry.update(forwards=session.get("forwards"), tokens=session.get("tokens"),
                     expected=session.get("kind"))
        passes = window.passes
        entry["passes"] = [p.as_dict() for p in passes]
        by_kind: dict[str, list[float]] = {}
        out["windows"][label] = entry
        print(f"verifier: [{raw['metric']}] {label}: {len(passes)} pass(es)")
        for p in passes:
            print(f"    {p.kind:<8} rows={p.rows} {p.device_ns / 1e6:9.1f} ms device "
                  f"(engine {p.engine_ns / 1e6:.1f} + dma {p.dma_ns / 1e6:.1f} + cc "
                  f"{p.collective_ns / 1e6:.1f}, union)  util {p.utilisation * 100:.0f}%"
                  f"  traced "
                  f"{'n/a' if p.traced_fraction is None else f'{p.traced_fraction * 100:.0f}%'}"
                  f"  [span-sum {p.busy_ns / 1e6:.0f} ms]")
            if p.kind in pooled:
                by_kind.setdefault(p.kind, []).append(p.device_ns / 1e6)
                if p.rows:
                    widths[p.kind] = p.rows
        # The first prefill pass of a window is not a sample of steady-state
        # work: it is the first forward of a fresh process, and it carries
        # first-touch costs the 16-device target pays at startup rather than per
        # request. Measured in one window: 1097.9 ms then 291.2 ms for the same
        # 512 rows — 3.8x, and *engine* time (809.6 vs 189.6 ms), not DMA, so
        # averaging the two reports a TTFT 2.4x the steady one. Decode passes
        # get no such treatment: measured first-then-second they were 407/414
        # and 483/392 ms, i.e. noise with no warm-up direction, and dropping one
        # of two samples there would leave a gated metric on a single pass.
        for kind, values in by_kind.items():
            if kind == "prefill" and len(values) > 1:
                warmup.append(values[0])
                values = values[1:]
            pooled[kind].extend(values)
        for note in entry["notes"]:
            print(f"  note: {note}")
        if session.get("kind") and passes:
            seen = {p.kind for p in passes}
            if seen != {session["kind"]}:
                out["notes"].append(
                    f"{label} was built to hold {session['kind']} passes but the NEFFs say "
                    f"{sorted(seen)}; the passes are pooled by what the NEFFs say"
                )

    out["passes"] = {k: {"count": len(v), "device_ms": v} for k, v in pooled.items()}
    out["prefill_warmup_ms"] = warmup
    if warmup:
        out["notes"].append(
            "the first prefill pass of each window is a warm-up and is excluded: "
            f"{', '.join(f'{v:.1f}' for v in warmup)} ms against "
            f"{', '.join(f'{v:.1f}' for v in pooled['prefill']) or 'nothing'} ms kept"
        )

    # TTFT: one prefill pass covers `rows` tokens, so the prompt needs
    # ceil(prompt / rows) of them. Measured per pass and multiplied, because a
    # window cannot hold every segment of a long prompt without the runtime
    # dropping notifications partway through.
    prefill = pooled["prefill"] or warmup[:1]
    if prefill:
        per_pass = sum(prefill) / len(prefill)
        rows = widths.get("prefill")
        prompt = raw.get("measured_prompt_tokens") or 0
        segments = max(1, -(-prompt // rows)) if rows else 1
        out.update(prefill_pass_ms=per_pass, prefill_pass_rows=rows,
                   prefill_segments=segments, ttft_ms=per_pass * segments)

    decode = pooled["decode"]
    if decode:
        out["decode_steps"] = len(decode)
        out["decode_step_ms"] = sum(decode) / len(decode)
        out["decode_rows"] = widths.get("decode")

    tps = correctness.get("tokens_per_step")
    if tps and out["decode_step_ms"] is not None:
        out["tokens_per_step"] = tps
        out["tpot_ms"] = out["decode_step_ms"] / tps
    elif out["decode_step_ms"] is not None:
        out["tokens_per_step"] = 1.0
        out["tpot_ms"] = out["decode_step_ms"]
        out["notes"].append(
            "tokens per step could not be measured, so TPOT is reported at one token a "
            "step — an upper bound, since speculation emits more"
        )
    if out["ttft_ms"] is not None and out["tpot_ms"] is not None:
        out["total_ms"] = out["ttft_ms"] + (MAX_TOKENS - 1) * out["tpot_ms"]
    if out["ttft_ms"] is None:
        out["notes"].append("no prefill pass was captured, so there is no TTFT")
    if out["decode_step_ms"] is None:
        out["notes"].append(
            "no decode pass was captured, so there is no step time and no TPOT. The "
            "decode window has to start after every prefill segment — see "
            "`_decode_delay`."
        )
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

    Per phase, measured first and derived after:

      `device_ttft_ms_<phase>`   one prefill pass x the segments the prompt needs.
      `device_step_ms_<phase>`   one decode pass.
      `device_tpot_ms_<phase>`   the step divided by the tokens a step emits,
                                 which speculation makes more than one.
      `device_total_ms_<phase>`  what a 32-token generate would cost at those
                                 rates.

    **Device time means engine-busy plus what the collectives transferred.** Not
    the execution span: on this port a span is mostly host waiting — a per-layer
    all-reduce sitting armed for its peers (20 s of span, 30 ms of transfer) and
    the DMA wait for weights arriving from NVMe at ~155 MB/s a rank instead of
    HBM at ~725 GB/s. Neither is device work and neither exists on the 16-device
    target. `span_sum_ms` is printed beside each window so the gap stays visible;
    it ran 15-60x the device time on every window measured here.
    """
    if not measured:
        return
    print("\n" + "=" * 72)
    print("ON-DEVICE TIME — engine-busy + real collective transfer; host waits "
          "excluded")
    for suffix, m in measured.items():
        print(f"\n  [{suffix}] phase {m['phase']}, measured on "
              f"{m.get('measured_case')} ({m.get('measured_prompt_tokens')} prompt tokens)")
        if m.get("ttft_ms") is not None:
            print(f"    TTFT        {m['ttft_ms']:10.1f} ms  "
                  f"{m.get('prefill_segments')} x {m.get('prefill_pass_ms', 0):.1f} ms "
                  f"prefill pass of {m.get('prefill_pass_rows')} rows")
        if m.get("decode_step_ms") is not None:
            print(f"    decode step {m['decode_step_ms']:10.1f} ms  "
                  f"mean of {m.get('decode_steps')} pass(es) of "
                  f"{m.get('decode_rows')} rows")
        if m.get("tokens_per_step"):
            print(f"    tokens/step {m['tokens_per_step']:10.2f}     "
                  "measured on the 32-token generate")
        if m.get("tpot_ms") is not None:
            print(f"    TPOT        {m['tpot_ms']:10.1f} ms  per output token "
                  f"({1000 / m['tpot_ms']:.0f} tok/s)")
        if m.get("total_ms") is not None:
            print(f"    {MAX_TOKENS} tokens  {m['total_ms'] / 1000:10.2f} s   "
                  "of device time at those rates")

        for label, w in (m.get("windows") or {}).items():
            print(f"      {label:<22} {w.get('device_ms', 0):9.1f} ms device   "
                  f"[span-sum {w.get('span_sum_ms') or 0:9.0f} ms]  "
                  f"{len(w.get('passes') or [])} pass(es)"
                  + (" DROPPED" if w.get("dropped") else ""))
        c = m.get("correctness") or {}
        if c.get("steps"):
            print(f"      the {MAX_TOKENS}-token generate: {c['steps']} forward(s), "
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
            # The decode window needs its own engine, built now that the phase's
            # is gone: it has to record from past the last prefill segment, and
            # that offset is `delay_iterations`, fixed when an engine is built.
            try:
                _measure_decode(phase, raw, args, tensor_parallel_size, profile_dir)
            except Exception:  # noqa: BLE001 - not a pass condition
                raw.setdefault("notes", []).append(
                    f"the decode engine raised:\n{traceback.format_exc()}"
                )
                print("verifier: the decode measurement failed; the run is "
                      "unaffected", file=sys.stderr)
            try:
                phase["device_time"] = _resolve_measurement(
                    raw, raw.pop("correctness", {}) or {}, model=args.model
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
