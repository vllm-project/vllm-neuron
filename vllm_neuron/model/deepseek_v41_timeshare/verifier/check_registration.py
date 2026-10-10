#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert the port is importable and wired into vLLM's model registry.

The cheapest constraint, and the one everything else depends on: until
`DeepseekV41ForCausalLM` imports and resolves through `vllm_neuron.model.registry`,
vLLM cannot instantiate the model and every later check fails for a reason that
has nothing to do with the port.

Checks, in order:

  1. `vllm_neuron.model.registry` imports at all. A missing dependency anywhere
     in the registry's import chain breaks this, not just the port's own code.
  2. `get_models()` maps the arch string `DeepseekV41ForCausalLM` to a class.
  3. That class is the port's, not a stand-in from another model directory.
  4. The factory and weight loader import, since the registry alone does not
     reach them.

Lives under `verifier/`, outside the integration loop's editable scope, so the
check cannot be relaxed by the thing being measured.

    python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.check_registration

Exit 0 only when all four hold. Needs no device and no checkpoint.
"""

from __future__ import annotations

import sys
import traceback

#: The arch string vLLM looks up, which is the checkpoint config's
#: `architectures[0]`. Renaming it in the registry would deregister the model.
ARCH = "DeepseekV41ForCausalLM"

#: Every module the port must expose. `model` is the forward path, `factory`
#: builds it, `weight_loader` feeds it; a port missing any of them is not wired.
REQUIRED_MODULES = (
    "vllm_neuron.model.deepseek_v41_timeshare.config",
    "vllm_neuron.model.deepseek_v41_timeshare.model",
    "vllm_neuron.model.deepseek_v41_timeshare.factory",
    "vllm_neuron.model.deepseek_v41_timeshare.weight_loader",
)

PACKAGE = "vllm_neuron.model.deepseek_v41_timeshare"


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
