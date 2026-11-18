# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package by file location.

A contrib model must stay self-contained and "simple to copy into an
application" (contrib/README.md, "Modeling code must be self-contained"), so
its tests must not reach the modeling code through a repository-rooted import
path such as ``contrib.models.exaone4.src``. That path resolves only while the
repository root happens to be on ``sys.path``, which is exactly what breaks
once the folder is copied out of the repository.

This helper instead loads ``<model folder>/src/__init__.py`` through
``importlib`` and registers it in ``sys.modules`` under :data:`SRC_PACKAGE`,
with ``src/`` as the package search path. Submodule imports
(``load("model")``) and the relative imports inside ``src/`` both work, and
nothing is added to ``sys.path``.

:data:`SRC_PACKAGE` is the name the loaded modules report as their
``__module__`` / ``__name__``, so it is also the name the tests use for
``logging.getLogger(...)`` and for ``__module__`` assertions. Derive those from
:data:`SRC_PACKAGE` rather than hardcoding the string.

Each test module loads this file by file location too, under its own per-model
name, so that two contrib models collected in one pytest run cannot resolve
each other's helper.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# The package name the loaded modeling code lives under. It is unique to this
# model, so it cannot collide with another contrib model in the same process.
SRC_PACKAGE = "contrib_exaone4_src"

# <model folder>/test/_local_src.py -> <model folder>/src
SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def load(submodule: str | None = None) -> ModuleType:
    """Return this model's ``src`` package, or one of its submodules.

    Args:
        submodule: A submodule name such as ``"model"``, ``"config"`` or
            ``"factory"``. ``None`` returns the ``src`` package itself.

    Returns:
        The requested module. The ``src`` package is loaded on the first call
        and cached in ``sys.modules`` under :data:`SRC_PACKAGE`.
    """
    if SRC_PACKAGE not in sys.modules:
        _load_src_package()
    if submodule is None:
        return sys.modules[SRC_PACKAGE]
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def _load_src_package() -> ModuleType:
    init_py = SRC_DIR / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"No modeling package found at {SRC_DIR}")

    spec = importlib.util.spec_from_file_location(
        SRC_PACKAGE, init_py, submodule_search_locations=[str(SRC_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[SRC_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a half-initialized package behind for the next call.
        del sys.modules[SRC_PACKAGE]
        raise
    return module
