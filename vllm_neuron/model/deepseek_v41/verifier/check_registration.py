#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert the port is importable and wired into vLLM's model registry.

The cheapest constraint and the one everything else depends on: until
`DeepseekV41ForCausalLM` resolves through `vllm_neuron.model.registry`, vLLM
cannot instantiate the model and every later check fails for an unrelated
reason. Checks that the registry imports at all (a missing dependency anywhere
in its chain breaks this, not just the port's code), that the arch string maps
to a class, that the class is the port's, and that the port's own four modules
import.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_registration

Needs no device and no checkpoint.
"""

from __future__ import annotations

import sys
import traceback

#: The checkpoint config's `architectures[0]`, which is what vLLM looks up.
ARCH = "DeepseekV41ForCausalLM"

#: Every module the port must expose; the registry alone does not reach them.
REQUIRED_MODULES = (
    "vllm_neuron.model.deepseek_v41.config",
    "vllm_neuron.model.deepseek_v41.model",
    "vllm_neuron.model.deepseek_v41.factory",
    "vllm_neuron.model.deepseek_v41.weight_loader",
)

PACKAGE = "vllm_neuron.model.deepseek_v41"


def main() -> int:
    problems: list[str] = []

    # ── 1/2/3. the registry ───────────────────────────────────────────────
    try:
        from vllm_neuron.model.registry import get_models

        registered = dict(get_models())
        found = registered.get(ARCH)
        if found is None:
            problems.append(
                f"'{ARCH}' is not in get_models(); registered arches are "
                f"{sorted(registered)}"
            )
        elif not getattr(found, "__module__", "").startswith(PACKAGE):
            problems.append(
                f"'{ARCH}' resolves to {found.__module__}.{found.__name__}, which is "
                f"not in {PACKAGE}"
            )
        else:
            print(f"  ok  {ARCH} -> {found.__module__}.{found.__name__}")
    except Exception:  # noqa: BLE001 - an unimportable registry is a failure
        problems.append(
            "importing vllm_neuron.model.registry raised:\n" + traceback.format_exc()
        )

    # ── 4. the port's own modules ─────────────────────────────────────────
    import importlib

    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            print(f"  ok  {name}")
        except Exception:  # noqa: BLE001
            problems.append(f"importing {name} raised:\n" + traceback.format_exc())

    if problems:
        print(f"\ncheck_registration: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        print("##check_registration[passed=0]")
        return 1

    print("\ncheck_registration: the port imports and is registered")
    print("##check_registration[passed=1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
