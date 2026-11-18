# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's own code and test helpers by file location.

A contrib model is not part of the ``vllm_neuron`` package, so the tests cannot
import ``vllm_neuron.model.granite``. The implementation sits in
``contrib/models/granite/src/``, which is not importable by name either: every
contrib model has a folder called ``src``, and this one is on no import path.

This module loads it from its path, once, under a name that no other contrib
model can take. The same applies to the shared helpers next to this file: they
are loaded by location, under names prefixed per model, so collecting two
contrib models in one pytest session cannot resolve one model's tests against
the other model's code.

This file lives once, at the ``test/`` root. Each test module loads *it* by file
location too (a small ``_load_local_src()``), rather than importing the bare
name ``_local_src``, for the same reason: a plain import would bind the first
``_local_src`` that any contrib model put in ``sys.modules``.

The loaded modules land in ``sys.modules``, so the ``MPExecutor`` workers (which
are forks of the test process) resolve the same classes when they unpickle a
``model_load`` callable.

Example::

    GraniteForCausalLM = _local_src.load().GraniteForCausalLM
    GraniteAttention = _local_src.load("model").GraniteAttention
    reference = _local_src.helper("granite_reference")
"""

import importlib
import importlib.util
import pathlib
import sys

_TEST_DIR = pathlib.Path(__file__).resolve().parent

#: The ``src`` package of this contrib model, and the name it is loaded under.
#: Derive module paths from these names (for example ``f"{SRC_PACKAGE}.model"``)
#: instead of writing them out, so one change of name reaches everything.
SRC_DIR = _TEST_DIR.parent / "src"
SRC_PACKAGE = "contrib_granite_src"

#: Prefix for the shared test helpers that sit next to this file.
HELPER_PREFIX = "contrib_granite_"


def _load_by_location(module_name: str, path: pathlib.Path, is_package: bool = False):
    """Load ``path`` as ``module_name``, once, and return it."""
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=[str(path.parent)] if is_package else None,
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module, so the relative imports inside the
        # package resolve.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def load(submodule: str | None = None):
    """Return this contrib model's ``src`` package, or one of its submodules.

    ``load()`` gives the package (so ``load().GraniteForCausalLM`` is the
    factory); ``load("model")`` gives ``src/model.py``.
    """
    _load_by_location(SRC_PACKAGE, SRC_DIR / "__init__.py", is_package=True)
    if submodule is None:
        return sys.modules[SRC_PACKAGE]
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def helper(name: str):
    """Return a shared test helper from this ``test/`` directory.

    ``helper("granite_reference")`` loads ``test/granite_reference.py``.
    """
    return _load_by_location(HELPER_PREFIX + name, _TEST_DIR / f"{name}.py")
