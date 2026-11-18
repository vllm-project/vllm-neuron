# SPDX-License-Identifier: Apache-2.0
"""The fused launch's member context: the model's `mk_kernels._fused_member`.

Inside `member(index, module)` the member's named scratch tensors are prefixed `L<index>_`
(a block kernel's scratch names are unique only within one block), its unnamed shared-HBM
ones (its outputs: stream, pre-mix, top-k, tap) are named `L<index>_out<n>` (left to the
compiler, two members' same-shape ones were not kept apart), and the module's cache-output
copies (`CACHE_COPY_EARLY` / `CACHE_COPY_AT_AR`) are off: inside a fused launch the caches
are updated in place and a member's returned caches are not outputs.
"""
import contextlib

import nki.language as _nl

_CACHE_COPY_FLAGS = ("CACHE_COPY_EARLY", "CACHE_COPY_AT_AR")


@contextlib.contextmanager
def member(index: int, module):
    saved = {f: getattr(module, f) for f in _CACHE_COPY_FLAGS if hasattr(module, f)}
    for f in saved:
        setattr(module, f, False)
    orig = _nl.ndarray
    anon = [0]

    def ndarray(*args, **kwargs):
        if kwargs.get("name"):
            kwargs = dict(kwargs, name=f"L{index}_{kwargs['name']}")
        elif kwargs.get("buffer") is _nl.shared_hbm:
            anon[0] += 1
            kwargs = dict(kwargs, name=f"L{index}_out{anon[0]}")
        return orig(*args, **kwargs)

    _nl.ndarray = ndarray
    try:
        yield
    finally:
        _nl.ndarray = orig
        for f, v in saved.items():
            setattr(module, f, v)
