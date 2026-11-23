# SPDX-License-Identifier: Apache-2.0
"""Load the HRM-Text code that ships in this contrib folder, by file location.

A contrib model is not part of the ``vllm_neuron`` package, so the tests cannot
import ``vllm_neuron.model.hrm_text``. The implementation sits in
``contrib/models/hrm_text/src/``, which is not importable by name either (every
contrib model has a folder called ``src``). This helper loads that folder from
its path, once, under a module name no other contrib model can take, and hands
out its modules. Nothing here needs the repository root, so the folder keeps
working when it is copied into an application.

``load_common()`` loads this folder's shared test helper
(``test/module/common.py``) the same way, so the unit, module and integration
tests all reach it without a package path.

Each test module loads *this* file by location too, under its own unique name
(see ``_load_local_src()`` in any of them), so nothing goes on ``sys.path``:
two contrib models collected in one pytest run cannot resolve to each other's
code.

The loaded modules land in ``sys.modules``, so the ``MPExecutor`` workers
(forks of the test process) resolve the same objects when they unpickle a
``model_load`` callable.

Example::

    SRC = _local_src.SRC_PACKAGE                     # "contrib_hrm_text_src"
    HrmTextForCausalLM = _local_src.load().HrmTextForCausalLM
    HrmTextAttention = _local_src.load("model").HrmTextAttention
    checkpoint = _local_src.load_common().resolve_checkpoint()
"""

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType

#: Module name of this folder's ``src`` package. Every string module path the
#: tests need derives from it (``f"{SRC_PACKAGE}.model"`` and so on), which is
#: also what ``logging.getLogger(__name__)`` inside ``src/model.py`` produces.
SRC_PACKAGE = "contrib_hrm_text_src"

#: Module name of this folder's shared test helper.
COMMON_MODULE = "contrib_hrm_text_test_common"

_TEST_DIR = pathlib.Path(__file__).resolve().parent

#: The ``src`` directory this helper loads.
SRC_DIR = _TEST_DIR.parent / "src"

#: The shared test helper this helper loads.
COMMON_PATH = _TEST_DIR / "module" / "common.py"


def _load_file(
    name: str, path: pathlib.Path, search_locations: list[str] | None = None
) -> ModuleType:
    """Load ``path`` under the module name ``name``, once."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=search_locations
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module, so the relative imports inside the
        # package resolve.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load(submodule: str | None = None) -> ModuleType:
    """Return this folder's ``src`` package, or one module inside it.

    Args:
        submodule: ``None`` for the package itself, or the name of a module in
            it, for example ``"model"`` or ``"config"``.

    Returns:
        ModuleType: The requested module.
    """
    package = _load_file(SRC_PACKAGE, SRC_DIR / "__init__.py", [str(SRC_DIR)])
    if submodule is None:
        return package
    return importlib.import_module(f"{SRC_PACKAGE}.{submodule}")


def load_common() -> ModuleType:
    """Return this folder's shared test helper, ``test/module/common.py``."""
    return _load_file(COMMON_MODULE, COMMON_PATH)
