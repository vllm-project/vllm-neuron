# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package, and the sibling test helpers,
by file location.

The point of a contrib model is that the folder is "simple to copy into an
application" (``contrib/README.md``, "Modeling code must be self-contained").
A test that writes ``from contrib.models.ernie4_5.src import ...`` only imports
while the repository root sits on ``sys.path``, so it stops working the moment
the folder is copied out of the repository. This helper imports the sibling
``src/`` directory from its own file location instead, under the unique
top-level module name :data:`SRC_PACKAGE`, so the tests in this folder run from
any working directory and from any parent project.

Nothing here touches ``sys.path``. Load the helper itself by file location too
(it is not importable by name unless its directory happens to be on
``sys.path``, and a plain ``import _local_src`` would resolve to whichever
contrib model got imported first in the process)::

    def _load_local_src():
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        name = "contrib_ernie4_5_local_src"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module

Then::

    _local_src = _load_local_src()
    Ernie4_5ForCausalLM = _local_src.load().Ernie4_5ForCausalLM  # src/__init__.py
    Ernie4_5Attention = _local_src.load("model").Ernie4_5Attention  # src/model.py
    checkpoint = _local_src.load_helper("checkpoint")  # test/_checkpoint.py

Every loaded module lands in ``sys.modules``, so the ``MPExecutor`` workers
(forks of the test process, see ``vllm_neuron/utils/executor.py``) resolve the
same objects when they unpickle a ``model_load`` callable, which pickle
references by module name.

``src/`` still imports the plugin's shared building blocks
(``vllm_neuron.functional``, ``vllm_neuron.nn``,
``vllm_neuron.model.neuron_config``, the weight loaders) and vLLM's public
APIs, which ``contrib/README.md`` explicitly allows. Only the resolution of
this model's *own* modules is local.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: Top-level module name that ``src/`` is registered under. It is unique to
#: this contrib model, so several contrib models can be loaded in one process.
#: The tests also build their string module paths from it -- the target of
#: ``mock.patch``, the ``caplog`` logger name and any ``__module__`` assertion
#: all have to name whatever this loader produces.
SRC_PACKAGE = "contrib_ernie4_5_src"

#: Prefix that :func:`load_helper` registers the sibling test helpers under.
HELPER_PREFIX = "contrib_ernie4_5_test_"

#: The directory this file sits in: the model's ``test/`` folder.
TEST_DIR = Path(__file__).resolve().parent

#: The ``src/`` directory of this contrib model, resolved from this file.
SRC_DIR = TEST_DIR.parent / "src"


def _load_by_location(name: str, path: Path, package_dir: Path | None = None):
    """Load one module from ``path`` under the module name ``name``, once."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=[str(package_dir)] if package_dir else None,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the contrib module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Register the module before executing it, so that the relative imports
    # inside a package ("from .config import Ernie4_5Config") resolve against
    # this parent.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load(submodule: str | None = None) -> ModuleType:
    """Import this model's ``src`` package, or one module inside it.

    The package is imported once per process and cached in ``sys.modules``
    under :data:`SRC_PACKAGE`, so repeated calls return the same objects and
    ``isinstance`` / ``is`` checks across test modules stay meaningful.

    Args:
        submodule: A module name inside ``src/``, for example ``"model"`` or
            ``"factory"``. When ``None``, the package itself (the exports of
            ``src/__init__.py``) is returned.

    Returns:
        The imported module.

    Raises:
        ImportError: If ``src/__init__.py`` cannot be found next to this file.

    Example::

        Ernie4_5Attention = load("model").Ernie4_5Attention
    """
    _load_by_location(SRC_PACKAGE, SRC_DIR / "__init__.py", package_dir=SRC_DIR)
    if submodule is None:
        return sys.modules[SRC_PACKAGE]
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_helper(name: str) -> ModuleType:
    """Import a sibling test helper of this file, by file location.

    The helpers live at the root of the model's ``test/`` folder so that
    ``test/unit``, ``test/module`` and ``test/integration`` share one copy of
    each. ``name`` is the helper's file stem without its leading underscore, so
    ``load_helper("checkpoint")`` loads ``test/_checkpoint.py`` under the module
    name ``contrib_ernie4_5_test_checkpoint``.

    Args:
        name: The helper name, for example ``"checkpoint"`` or
            ``"component_utils"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the helper cannot be found next to this file.

    Example::

        get_ernie_checkpoint = load_helper("checkpoint").get_ernie_checkpoint
    """
    return _load_by_location(f"{HELPER_PREFIX}{name}", TEST_DIR / f"_{name}.py")
