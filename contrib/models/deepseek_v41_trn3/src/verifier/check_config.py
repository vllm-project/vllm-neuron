#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert the model being served is really DeepSeek-V4.1-Flash.

A port made to fit by shrinking the model — 8 layers instead of 40, 64 experts
instead of 384 — would still load, run, and emit fluent text, so the verifier
cannot catch it. This checks the shape, in both places it can be weakened: the
port's own `DeepseekV41Config` defaults, and the `config.json` of the checkpoint
under `--checkpoint` (as raw JSON and after `from_configs` flattens it).

Ground truth is the official config, revision
`dba1be0a40aa45a94ad051997016db3960a90277`, transcribed into `EXPECTED`.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_config \\
        --checkpoint <checkpoint-dir>

Exit 0 only when every field matches. Needs no device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Architectural fields as named on `DeepseekV41Config` after flattening, with
#: the official values. Anything that changes what the model *is* belongs here;
#: tuning knobs (dtype, neuron_config) do not.
EXPECTED: dict[str, object] = {
    # core transformer shape
    "vocab_size": 129280,
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "rms_norm_eps": 1e-20,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "sliding_window": 128,
    # attention: low-rank Q/O projections
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "qk_rope_head_dim": 64,
    # rope
    "max_position_embeddings": 1_048_576,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    # MoE FFN
    "moe_intermediate_size": 2304,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "routed_scaling_factor": 1.5,
    # hyper-connections
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    # Engram n-gram memory
    "engram_layer_ids": [1, 14],
    "engram_vocab_size": 16_000_000,
    "engram_max_ngram_size": 4,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    # sparse index / KV compression
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    # DSpark speculative decoding
    "num_nextn_predict_layers": 3,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "dspark_markov_rank": 256,
    "dspark_block_size": 5,
    # special tokens
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
}

#: Compared key by key: an extra key here is not a changed model.
EXPECTED_ROPE_SCALING: dict[str, object] = {
    "rope_type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536,
}

#: Also checked against the raw JSON, so a `from_configs` that ignored the file
#: and returned defaults cannot hide a shrunken checkpoint.
RAW_CRITICAL = (
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
    "num_attention_heads",
    "head_dim",
    "n_routed_experts",
)


def _mismatches(config: object, where: str) -> list[str]:
    """Every architectural field of `config` that is not the official value."""
    out: list[str] = []
    for field, want in EXPECTED.items():
        if not hasattr(config, field):
            out.append(f"{where}: field '{field}' is gone")
            continue
        got = getattr(config, field)
        if isinstance(want, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - want) <= abs(want) * 1e-9
        else:
            ok = got == want
        if not ok:
            out.append(f"{where}: {field} = {got!r}, official is {want!r}")

    scaling = getattr(config, "rope_scaling", None) or {}
    if not isinstance(scaling, dict):
        out.append(f"{where}: rope_scaling = {scaling!r}, expected a mapping")
    else:
        for key, want in EXPECTED_ROPE_SCALING.items():
            got = scaling.get(key)
            if got != want:
                out.append(f"{where}: rope_scaling[{key!r}] = {got!r}, official is {want!r}")
    return out


def _raw_mismatches(raw: dict, where: str) -> list[str]:
    """The critical fields as the checkpoint file itself spells them."""
    out: list[str] = []
    text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    for field in RAW_CRITICAL:
        got = text.get(field, raw.get(field, "<absent>"))
        want = EXPECTED[field]
        if got != want:
            out.append(f"{where}: {field} = {got!r}, official is {want!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory; its config.json is checked alongside the "
        "port's own defaults",
    )
    args = p.parse_args(argv)

    from ..config import DeepseekV41Config

    problems: list[str] = []

    # ── 1. the port's own defaults ────────────────────────────────────────
    try:
        problems += _mismatches(DeepseekV41Config(), "config.py defaults")
    except Exception as exc:  # noqa: BLE001 - an unconstructable config is a failure
        problems.append(f"config.py defaults: DeepseekV41Config() raised {exc!r}")

    # ── 2. the checkpoint being served ────────────────────────────────────
    ckpt_json = Path(args.checkpoint) / "config.json"
    if not ckpt_json.is_file():
        problems.append(
            f"checkpoint: no config.json at {ckpt_json}. vLLM cannot load a "
            "checkpoint directory without one, so there is nothing to serve yet."
        )
    else:
        try:
            raw = json.loads(ckpt_json.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint: {ckpt_json} is not readable JSON ({exc})")
        else:
            problems += _raw_mismatches(raw, "checkpoint config.json")
            try:
                problems += _mismatches(
                    DeepseekV41Config.from_configs(raw), "checkpoint via from_configs"
                )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"checkpoint via from_configs: raised {exc!r}")

    if problems:
        print(f"check_config: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  FAIL {problem}")
        print("##check_config[passed=0]")
        return 1

    print(
        f"check_config: {len(EXPECTED)} field(s) + rope_scaling match the official "
        "config, in config.py and in the checkpoint"
    )
    print("##check_config[passed=1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
