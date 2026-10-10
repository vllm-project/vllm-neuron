#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert the checkpoint directory is a complete, full-size DeepSeek-V4.1-Flash.

vLLM loads a checkpoint *directory*, which needs three things the raw shard dump
in `partition-artifact/hf/` does not have: a `config.json`, a
`model.safetensors.index.json` mapping every weight to its shard, and a
tokenizer. This checks all three are there and that the index actually describes
the whole model.

The last part is the one that matters. A checkpoint can be complete and still be
the wrong model: an index naming 8 layers loads, serves, and emits fluent text.
So every layer `0..num_hidden_layers-1` must appear in the weight map, and every
shard the map names must exist on disk.

What this does NOT check is the architectural shape of `config.json` — that is
`check_config.py`'s job, against the official values.

Lives under `verifier/`, outside the integration loop's editable scope.

    python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.check_checkpoint \\
        --checkpoint <checkpoint-dir>

Exit 0 only when the checkpoint is loadable and whole. Needs no device, and
reads no tensor data — only the index and the config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: A tokenizer is present if any of these groups is complete. vLLM accepts a
#: fast tokenizer alone; a sentencepiece checkpoint needs the model file beside
#: its config.
TOKENIZER_ALTERNATIVES = (
    ("tokenizer.json",),
    ("tokenizer_config.json", "tokenizer.model"),
)

#: This checkpoint keys its decoder weights `layers.<i>.…`, not the
#: `model.layers.<i>.…` that most HF text models use — the tensor names carry no
#: `model.` prefix, and `vision.blocks.<i>` is the vision tower rather than a
#: decoder layer. The DSpark/MTP layers are indexed separately under `mtp.<i>`.
LAYER_KEY_PREFIX = "layers."
MTP_KEY_PREFIX = "mtp."


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="checkpoint directory")
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint)
    problems: list[str] = []

    if not ckpt.is_dir():
        print(f"check_checkpoint: FAIL {ckpt} is not a directory")
        print("##check_checkpoint[passed=0]")
        return 1

    # ── config.json ───────────────────────────────────────────────────────
    config_path = ckpt / "config.json"
    n_layers: int | None = None
    n_mtp: int | None = None
    if not config_path.is_file():
        problems.append(f"no config.json at {config_path}")
    else:
        try:
            raw = json.loads(config_path.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"config.json is not readable JSON ({exc})")
        else:
            text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
            n_layers = text.get("num_hidden_layers", raw.get("num_hidden_layers"))
            if not isinstance(n_layers, int):
                problems.append("config.json declares no num_hidden_layers")
                n_layers = None
            n_mtp = text.get("num_nextn_predict_layers", raw.get("num_nextn_predict_layers"))
            if not isinstance(n_mtp, int):
                n_mtp = None
            arches = raw.get("architectures") or []
            if "DeepseekV41ForCausalLM" not in arches:
                problems.append(
                    f"config.json architectures = {arches!r}; vLLM dispatches on this, "
                    "so it must contain 'DeepseekV41ForCausalLM'"
                )

    # ── tokenizer ─────────────────────────────────────────────────────────
    if not any(all((ckpt / f).is_file() for f in group) for group in TOKENIZER_ALTERNATIVES):
        wanted = " or ".join(" + ".join(g) for g in TOKENIZER_ALTERNATIVES)
        problems.append(f"no tokenizer in {ckpt} (need {wanted})")

    # ── weight index, and the shards it names ─────────────────────────────
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.is_file():
        problems.append(
            f"no model.safetensors.index.json at {index_path}. Without it vLLM cannot "
            "find which shard holds which weight."
        )
    else:
        try:
            index = json.loads(index_path.read_text())
            weight_map: dict = index["weight_map"]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"index is unusable ({exc})")
        else:
            missing_shards = sorted(
                {s for s in set(weight_map.values()) if not (ckpt / s).exists()}
            )
            if missing_shards:
                problems.append(
                    f"{len(missing_shards)} shard(s) named by the index are absent, "
                    f"starting with {missing_shards[:3]}"
                )

            def indices(prefix: str) -> set[int]:
                """The layer indices the weight map carries under `prefix`."""
                out: set[int] = set()
                for key in weight_map:
                    if not key.startswith(prefix):
                        continue
                    part = key[len(prefix):].split(".", 1)[0]
                    if part.isdigit():
                        out.add(int(part))
                return out

            for what, count, prefix in (
                ("decoder layers", n_layers, LAYER_KEY_PREFIX),
                ("MTP/DSpark layers", n_mtp, MTP_KEY_PREFIX),
            ):
                if count is None:
                    continue
                absent = sorted(set(range(count)) - indices(prefix))
                if absent:
                    problems.append(
                        f"the index carries no weights for {len(absent)} of {count} "
                        f"{what} under '{prefix}' "
                        f"({absent[:6]}{'...' if len(absent) > 6 else ''}). "
                        "A checkpoint missing layers is not this model."
                    )
                else:
                    print(f"  ok  index covers all {count} {what}")

            if not problems:
                print(
                    f"  ok  {len(weight_map)} weight(s) across "
                    f"{len(set(weight_map.values()))} shard(s), all present"
                )

    if problems:
        print(f"check_checkpoint: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  FAIL {problem}")
        print("##check_checkpoint[passed=0]")
        return 1

    print(f"check_checkpoint: {ckpt} is loadable and whole")
    print("##check_checkpoint[passed=1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
