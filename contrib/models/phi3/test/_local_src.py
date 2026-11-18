# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package by file location.

A contrib model is not part of the ``vllm_neuron`` package, so there is no
``vllm_neuron.model.phi3`` to import, and a plain ``import src`` would collide
with every other contrib model. An absolute namespace path
(``contrib.models.phi3.src...``) resolves only while the repository root is on
``sys.path``, which stops being true as soon as this folder is copied into an
application. So load the sibling ``src/`` folder once, by file location, under
a module name that is unique to this model.

The package stays in ``sys.modules``, so an ``MPExecutor`` worker (a fork of
the test process) resolves the same classes when it unpickles a ``model_load``
callable.

Every string module path must be derived from ``SRC_PACKAGE`` rather than
written out, so that it keeps matching what ``__name__`` is inside ``src/``.

A test module loads this helper itself, by file location under its own unique
name, so that collecting two contrib models in one pytest run cannot resolve
one model's tests against the other model's ``src/``::

    import importlib.util
    from pathlib import Path

    def _load_local_src():
        name = "contrib_phi3_local_src"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
"""

import importlib
import importlib.util
import sys
from pathlib import Path

#: Top-level module name that ``src/`` is loaded under. ``src/model.py`` sees
#: ``__name__ == f"{SRC_PACKAGE}.model"``, so patch targets and logger names
#: must be built from this value.
SRC_PACKAGE = "contrib_phi3_src"

#: The ``src/`` directory of this contrib model.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

#: Module name that ``test/phi3_utils.py`` is loaded under, unique to this
#: model for the same reason ``SRC_PACKAGE`` is.
TEST_UTILS_MODULE = "contrib_phi3_test_utils"

_TEST_DIR = Path(__file__).resolve().parent


def load(submodule: str | None = None):
    """Return the ``src`` package, or one of its submodules.

    Args:
        submodule: A module of the package, for example ``"model"``. ``None``
            returns the package itself.

    Returns:
        The loaded module. The package is loaded once for the process.

    Example::

        >>> factory = load().Phi3ForCausalLM            # doctest: +SKIP
        >>> Phi3Config = load("config").Phi3Config      # doctest: +SKIP
    """
    if SRC_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            SRC_PACKAGE,
            SRC_DIR / "__init__.py",
            submodule_search_locations=[str(SRC_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module, so that the relative imports inside the
        # package resolve.
        sys.modules[SRC_PACKAGE] = module
        spec.loader.exec_module(module)
    package = sys.modules[SRC_PACKAGE]
    if submodule is None:
        return package
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_test_utils():
    """Return ``test/phi3_utils.py``, the helper the unit and module tests share.

    Loaded by file location for the same reason as ``src/``: no ``sys.path``
    entry, and a name that cannot collide with another contrib model.

    Example::

        >>> U = load_test_utils()                       # doctest: +SKIP
        >>> U.WORLD_SIZE                                # doctest: +SKIP
        4
    """
    if TEST_UTILS_MODULE not in sys.modules:
        path = _TEST_DIR / "phi3_utils.py"
        spec = importlib.util.spec_from_file_location(TEST_UTILS_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module: the helper's own module-level functions
        # are pickled by name for the forked MPExecutor workers.
        sys.modules[TEST_UTILS_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[TEST_UTILS_MODULE]
