# SPDX-License-Identifier: Apache-2.0
"""Import this model folder's own code by file location.

A contrib model has to stay "simple to copy into an application"
(``contrib/README.md``, "Modeling code must be self-contained"). A test that
says ``from contrib.models.bailing_moe.src... import ...`` only imports while
the repository root is on ``sys.path``, so it breaks the moment the folder is
copied out of the repository.

This module loads ``src/`` from its own location on disk, under the module name
``contrib_bailing_moe_src``, and the shared test helpers of ``test/module/``
under ``contrib_bailing_moe_test``. Both names are unique to this model, so two
contrib folders copied into one application never collide. Nothing here depends
on where the folder sits, only on the location of this file.

Load this helper by file location too, never with a plain ``import``: a plain
``import _local_src`` caches the first ``_local_src`` of the process, so a
pytest run that collects two contrib models would resolve the second model's
tests against the first model's ``src/``. Each test module of this folder does
this, after its own imports (the block is code, not an ``import`` statement, so
no import follows code out of order)::

    _LOADER = importlib.util.spec_from_file_location(
        "contrib_bailing_moe_local_src",
        Path(__file__).resolve().parents[1] / "_local_src.py",
    )
    _local_src = importlib.util.module_from_spec(_LOADER)
    _LOADER.loader.exec_module(_local_src)

    utils = _local_src.load_test_module("utils")
    BailingMoeV2Config = _local_src.load("config").BailingMoeV2Config

Executing this file more than once is harmless. The loaded packages stay cached
in ``sys.modules``, so every test module of this folder shares one copy of the
code under test. That one copy is also what lets ``MPExecutor`` ship a
``functools.partial`` of a helper to its workers: pickle stores the helper by
module name, and ``vllm_neuron/utils/executor.py`` pins ``fork``, so a worker
already holds these module names when it unpickles. Derive any module-path string from ``SRC_PACKAGE`` rather than
writing one out: the modeling code's own logger, for example, comes from
``logging.getLogger(__name__)`` in ``src/model.py``, which is
``contrib_bailing_moe_src.model``.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The model folder (``contrib/models/bailing_moe``), found from this file.
MODEL_DIR = Path(__file__).resolve().parents[1]

#: The modeling code of this model.
SRC_DIR = MODEL_DIR / "src"

#: The module name of ``src/``. Unique to this model.
SRC_PACKAGE = "contrib_bailing_moe_src"

#: The module name of the shared test helpers in ``test/module/``.
TEST_PACKAGE = "contrib_bailing_moe_test"


def _load_src_package() -> ModuleType:
    """Load ``src/__init__.py`` as the package ``SRC_PACKAGE``."""
    package = sys.modules.get(SRC_PACKAGE)
    if package is not None:
        return package
    spec = importlib.util.spec_from_file_location(
        SRC_PACKAGE,
        SRC_DIR / "__init__.py",
        submodule_search_locations=[str(SRC_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    # Register before executing: ``src/__init__.py`` imports its own submodules.
    sys.modules[SRC_PACKAGE] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        del sys.modules[SRC_PACKAGE]
        raise
    return package


def _test_package() -> ModuleType:
    """Register ``TEST_PACKAGE`` for the helper modules in ``test/module/``.

    ``test/module/`` holds no ``__init__.py``, so the package is synthesized
    from a spec whose search path is that directory. Do not add an
    ``__init__.py`` anywhere under this ``test/`` tree to make the helpers
    importable the ordinary way: the repository ships its own top-level
    ``test`` package (``test/__init__.py``) and puts the repository root on
    ``sys.path`` (``pythonpath = .`` in ``pytest.ini``), so a packaged test
    tree here would collide with it under pytest's default import mode.
    """
    package = sys.modules.get(TEST_PACKAGE)
    if package is not None:
        return package
    spec = importlib.machinery.ModuleSpec(TEST_PACKAGE, None, is_package=True)
    spec.submodule_search_locations = [str(MODEL_DIR / "test" / "module")]
    package = importlib.util.module_from_spec(spec)
    sys.modules[TEST_PACKAGE] = package
    return package


def load(submodule: str | None = None) -> ModuleType:
    """Import this folder's ``src/``, or one module of it.

    Args:
        submodule: A module of ``src/``, for example ``"model"``. ``None``
            returns the ``src`` package itself, which exports the factory and
            the config class.

    Returns:
        ModuleType: The imported module, cached in ``sys.modules`` under
            ``SRC_PACKAGE``.

    Example:
        >>> load("config").BailingMoeV2Config  # doctest: +SKIP
        <class 'contrib_bailing_moe_src.config.BailingMoeV2Config'>
    """
    package = _load_src_package()
    if submodule is None:
        return package
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_test_module(name: str) -> ModuleType:
    """Import one shared test helper of ``test/module/``.

    Args:
        name: The helper module, ``"utils"`` or ``"reference_model"``.

    Returns:
        ModuleType: The imported module, cached in ``sys.modules`` under
            ``TEST_PACKAGE``.
    """
    _test_package()
    return importlib.import_module(f"{TEST_PACKAGE}.{name}")
