# SPDX-License-Identifier: Apache-2.0
"""Make ``gemma4_neuron`` importable and registered for the tests.

Installed (``pip install -e contrib/models/gemma4``), vLLM calls ``register()``
through the ``vllm.general_plugins`` entry point in every process. Not
installed, this loads ``src/`` as the package and registers it in the test
process only.
"""

import importlib.metadata
import importlib.util
import pathlib
import sys


def _load_from_src():
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    spec = importlib.util.spec_from_file_location(
        "gemma4_neuron",
        src / "__init__.py",
        submodule_search_locations=[str(src)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemma4_neuron"] = module
    spec.loader.exec_module(module)
    return module


try:
    import gemma4_neuron
except ImportError:
    gemma4_neuron = _load_from_src()

gemma4_neuron.register()

INSTALLED = any(
    ep.value.startswith("gemma4_neuron")
    for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
)
