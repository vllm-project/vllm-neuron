# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package, and its test helpers, by file location.

contrib/README.md requires a contrib model to be "simple to copy into an
application", so nothing in this test tree may depend on the repository root
being on ``sys.path``. Every test module here reaches the modeling code through
``load()`` instead of importing ``contrib.models.nanbeige.src``, and reaches
the shared test helper through ``load_test_helper()``.

Nothing is ever put on ``sys.path``. Each module is registered in
``sys.modules`` under a name derived from this model's folder name, so two
contrib models loaded in the same process cannot collide, and a
``multiprocessing`` fork child (``MPExecutor``) reuses the parent's modules.
Each test module loads *this* file by file location too, under its own
per-model name, for the same reason: a plain ``import _local_src`` would be
cached process-wide under one generic name.

Derive string module paths from ``SRC_PACKAGE`` rather than hardcoding them: a
logger name or a monkeypatch target inside ``src/model.py`` resolves to
``{SRC_PACKAGE}.model``.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# ``contrib/models/nanbeige/``, the folder that holds ``src/`` and ``test/``.
MODEL_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = MODEL_ROOT.name
SRC_DIR = MODEL_ROOT / "src"
SRC_PACKAGE = f"contrib_{MODEL_NAME}_src"


def _load(
    module_name: str, location: Path, search_locations: list[str] | None = None
) -> ModuleType:
    """Import ``location`` as ``module_name``, once per process."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        module_name, location, submodule_search_locations=search_locations
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot build an import spec for {location}.")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so that the module's own relative imports resolve.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def load(submodule: str | None = None) -> ModuleType:
    """Return this model's ``src`` package, or one of its submodules.

    Args:
        submodule: A module name inside ``src/`` (for example ``"model"``).
            ``None`` returns the package itself.
    """
    package = _load(SRC_PACKAGE, SRC_DIR / "__init__.py", [str(SRC_DIR)])
    if submodule is None:
        return package
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_test_helper(name: str) -> ModuleType:
    """Return this model's ``test/<name>.py`` helper module."""
    return _load(f"contrib_{MODEL_NAME}_{name}", MODEL_ROOT / "test" / f"{name}.py")
