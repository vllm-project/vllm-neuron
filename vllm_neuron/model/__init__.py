# SPDX-License-Identifier: Apache-2.0
import importlib as _importlib


def __getattr__(name):
    if name in ("llama3", "gpt_oss"):
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
