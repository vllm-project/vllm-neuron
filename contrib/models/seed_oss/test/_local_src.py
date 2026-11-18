# SPDX-License-Identifier: Apache-2.0
"""Load this contrib model's ``src/`` package by file location.

The point of a contrib model is that the folder is "simple to copy into an
application" (``contrib/README.md``, "Modeling code must be self-contained").
A test that writes ``from contrib.models.seed_oss.src import ...`` only
imports while the repository root sits on ``sys.path``, so it stops working
the moment the folder is copied out of the repository. This helper imports the
sibling ``src/`` directory from its own file location instead, under the
unique top-level module name :data:`SRC_PACKAGE`, so the tests in this folder
run from any working directory and from any parent project.

Load the helper itself by file location too (it is not importable by name
unless its directory happens to be on ``sys.path``)::

    def _load_local_src():
        path = Path(__file__).resolve().parents[1] / "_local_src.py"
        name = "contrib_seed_oss_local_src"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module

Then::

    _local_src = _load_local_src()
    SeedOssForCausalLM = _local_src.load().SeedOssForCausalLM  # src/__init__.py
    Factory = _local_src.load("factory").SeedOssForCausalLM   # src/factory.py

``src/`` still imports the plugin's shared building blocks (``vllm_neuron.nn``,
``vllm_neuron.functional``, ``vllm_neuron.model.neuron_config``, the weight
loaders) and vLLM's public APIs, which ``contrib/README.md`` explicitly
allows. Only the resolution of the model's *own* modules is local.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: Top-level module name that ``src/`` is registered under. It is unique to
#: this contrib model, so several contrib models can be loaded in one process.
#: The tests also build their string module paths from it -- the target of
#: ``mock.patch``, the ``caplog`` logger name and the ``__module__`` assertion
#: all have to name whatever this loader produces.
SRC_PACKAGE = "contrib_seed_oss_src"

#: The ``src/`` directory of this contrib model, resolved from this file.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"


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
    """
    if SRC_PACKAGE not in sys.modules:
        init_path = SRC_DIR / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            SRC_PACKAGE,
            init_path,
            submodule_search_locations=[str(SRC_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the contrib src package at {init_path}")
        module = importlib.util.module_from_spec(spec)
        # Register the package before executing it, so that the relative
        # imports inside ``src/`` ("from .config import SeedOssConfig")
        # resolve against this parent.
        sys.modules[SRC_PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[SRC_PACKAGE]
            raise

    if submodule is None:
        return sys.modules[SRC_PACKAGE]
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")
