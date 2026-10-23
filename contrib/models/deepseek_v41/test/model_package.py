# SPDX-License-Identifier: Apache-2.0
"""Make the contributed modeling code importable, by name, as ``deepseek_v41``.

The name has to resolve from ``sys.path`` in a *fresh* interpreter, not just in
this process: the package registers itself as
``ModelRegistry.register_model(ARCH, f"{__name__}.factory:{ARCH}")``, and vLLM
re-imports that string in its architecture-inspection subprocess and in all 64
spawned workers. So ``load()`` also exports the path into ``PYTHONPATH``.

``test/deepseek_v41`` is a symlink to ``../src`` because ``contrib/README.md``
fixes the folder name as ``src/`` while ``register_model`` needs the package
named ``deepseek_v41``.

Importing the package *is* the registration step — contrib Option 1 — which is
why nothing here touches ``vllm_neuron/model/registry.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IMPORT_ROOT = Path(__file__).resolve().parent
PACKAGE = "deepseek_v41"


def load():
    """Import the contributed model package and return it. Idempotent."""
    root = str(IMPORT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in (root, existing) if p
        )

    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]

    init = IMPORT_ROOT / PACKAGE / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(
            f"no modeling code at {init}. The `{PACKAGE}` symlink in "
            f"{IMPORT_ROOT} should point at the sibling `src/`; a checkout that "
            f"did not preserve symlinks is the usual cause."
        )

    import importlib

    return importlib.import_module(PACKAGE)
