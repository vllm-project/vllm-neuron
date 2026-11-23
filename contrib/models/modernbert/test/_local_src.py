# SPDX-License-Identifier: Apache-2.0
"""Load this model folder's own ``src/`` and test helpers by file location.

A contrib model has to be "simple to copy into an application"
(contrib/README.md, "Modeling code must be self-contained"). A test that
imports the modeling code as ``contrib.models.modernbert.src...`` only works
while the folder sits in this repository and the repository root is on
``sys.path``. This helper loads the code by file location instead, relative to
this file, so every test in this folder works from any working directory and
keeps working after the folder is copied somewhere else. Nothing is added to
``sys.path``.

Every module this helper publishes is named after the contrib model folder, so
two contrib models loaded in one pytest session cannot collide in
``sys.modules`` (every contrib model folder has a ``src/config.py``, a
``src/model.py`` and a ``test/module/common.py``).

Each test module loads this file the same way, by location, under the unique
name ``contrib_modernbert_local_src`` — see the loader stanza at the top of
each test module — and then calls::

    src = _local_src.load()                  # the src package
    config = _local_src.load("config")       # src/config.py
    common = _local_src.load_test_module("module/common.py")

Derive any module-path string (a ``mock.patch`` target, a ``__module__``
assertion, a logger name) from :data:`SRC_PACKAGE` rather than hardcoding it:
``logging.getLogger(__name__)`` inside ``src/model.py`` resolves to
``f"{SRC_PACKAGE}.model"``.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Prefix of every module name this helper publishes. It names the contrib model
# folder, so the names are unique across the contrib tree.
MODULE_PREFIX = "contrib_modernbert"

# contrib/models/modernbert/ and contrib/models/modernbert/test/.
MODEL_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent

# The modeling code and the top-level module name it is published under.
SRC_DIR = MODEL_ROOT / "src"
SRC_PACKAGE = f"{MODULE_PREFIX}_src"


def _load_file(name: str, path: Path, package_dir: Path | None = None) -> ModuleType:
    """Import ``path`` as the module ``name``, once per interpreter.

    Args:
        name: Module name to publish in ``sys.modules``.
        path: File to execute.
        package_dir: Directory of the package's submodules, when ``path`` is a
            package ``__init__.py``. Setting it gives the module a ``__path__``,
            so ``importlib.import_module(f"{name}.config")`` resolves.

    Returns:
        ModuleType: The loaded module.

    Raises:
        ImportError: If the file has no loadable spec.
    """
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    search_locations = None if package_dir is None else [str(package_dir)]
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=search_locations
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module, so a relative import inside the package
    # ("from .config import ..." in src/model.py) finds its parent package.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load(submodule: str | None = None) -> ModuleType:
    """Return this folder's ``src`` package, or one module of it.

    Args:
        submodule: Module name inside ``src/``, for example ``"config"``.
            ``None`` returns the package itself, which exports
            ``ModernBertModel``.

    Returns:
        ModuleType: The ``src`` package or the named submodule.
    """
    package = _load_file(SRC_PACKAGE, SRC_DIR / "__init__.py", package_dir=SRC_DIR)
    if submodule is None:
        return package
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_test_module(relative_path: str) -> ModuleType:
    """Return one module of this folder's test tree.

    The unit, module and integration tests share ``test/module/common.py``, and
    a shared test helper needs the same file-location treatment as ``src/``.

    Args:
        relative_path: Path of the file under ``test/``, for example
            ``"module/common.py"``.

    Returns:
        ModuleType: The loaded module.
    """
    name = f"{MODULE_PREFIX}_test_" + relative_path.removesuffix(".py").replace(
        "/", "_"
    )
    return _load_file(name, TEST_ROOT.joinpath(*relative_path.split("/")))
