# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package by file location.

A contrib model is not part of the ``vllm_neuron`` package, so there is no
``vllm_neuron.model.olmo3`` to import. Two obvious alternatives both fail:

* a plain ``import src`` would collide with every other contrib model, and
* an absolute ``contrib.models.olmo3.src`` path resolves only while the
  repository root is on ``sys.path``, which stops holding the moment the folder
  is copied into an application — and "simple to copy into an application" is
  the whole rationale of the self-containment rule in ``contrib/README.md``.

So load ``src/`` once, by file location, under a name that is unique to this
model. Relative imports inside the package (``from .config import ...``)
resolve, because the spec carries ``submodule_search_locations`` and the
package is in ``sys.modules`` before ``exec_module`` runs. Nothing here reads
the repository root, and the package stays in ``sys.modules``, so a forked
child process (an ``MPExecutor`` worker, or the vLLM engine core) resolves the
same classes.

Every test in this folder reaches the implementation through this module.
``test/conftest.py`` puts this directory on ``sys.path`` so that
``test/unit`` and ``test/integration`` can both import it.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType

SRC_PACKAGE_NAME = "contrib_olmo3_src"
_SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"


def src_package() -> ModuleType:
    """Return the ``src`` package of this contrib model, loading it once."""
    if SRC_PACKAGE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            SRC_PACKAGE_NAME,
            _SRC_DIR / "__init__.py",
            submodule_search_locations=[str(_SRC_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module, so that the relative imports inside the
        # package resolve.
        sys.modules[SRC_PACKAGE_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[SRC_PACKAGE_NAME]


def src_module(name: str) -> ModuleType:
    """Return a module of the ``src`` package, for example ``"model"``."""
    src_package()
    return importlib.import_module(f"{SRC_PACKAGE_NAME}.{name}")
