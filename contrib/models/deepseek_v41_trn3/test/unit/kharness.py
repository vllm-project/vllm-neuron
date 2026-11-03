# SPDX-License-Identifier: Apache-2.0
"""Run one `@nki.jit` kernel on a single NeuronCore, the way the model runs it.

    out = run_nki(kernel_fn, *tensors, grid=0)

compiles `kernel_fn` through `libtorch_neuronx_lite`'s `wrap_nki` inside a
`torch.compile` graph with the plugin's compile backend -- exactly what
`model._nki_entry` does -- so a kernel that passes here runs unchanged in the
model. CPU tensors are moved to `neuron:0`; results come back on the CPU.

`compare(name, got, ref)` prints the component-level metrics the equivalence
skill asks for: max abs error, relative L2 error, cosine similarity and the
R-ratio (error norm over the reference's own bf16 rounding noise).
"""

from __future__ import annotations

import functools
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", os.environ.get("KH_CORE", "0"))
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
# Both kernel caches key on the kernel *function's* source, not its helpers': a unit
# test must always recompile what it tests.
os.environ.setdefault("NKI_DISABLE_TRACE_CACHE", "1")
os.environ.setdefault("NEURON_LIBTORCH_DISABLE_COMPILE_CACHE", "1")
# This port's own cache, keyed on the *contents of every kernel file*: lite's NKI
# cache keys a kernel on its own function source, so an edit to a helper module
# would otherwise be served a stale NEFF.
import glob as _glob
import hashlib as _hashlib

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deepseek_v41", "kernels")
_h = _hashlib.sha256()
for _f in sorted(_glob.glob(os.path.join(_KDIR, "*.py"))):
    with open(_f, "rb") as _fh:
        _h.update(_fh.read())
os.environ["NEURON_COMPILE_CACHE_URL"] = "/mnt/nvme/neuron_cache_unit/" + _h.hexdigest()[:16]

_TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import torch  # noqa: E402

import deepseek_v41  # noqa: E402,F401  - registers the architecture, imports the venv plugin
from deepseek_v41 import model as M  # noqa: E402

DEVICE = torch.device("neuron:0")


@functools.lru_cache(maxsize=None)
def _entry(kernel_fn, grid: int):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    M._raise_dynamo_limits()
    wrapped = wrap_nki(kernel_fn)
    if grid:
        wrapped = wrapped[grid]

    def entry(*args):
        return wrapped(*args)

    entry.__name__ = f"unit_{kernel_fn.__name__}"
    return torch.compile(
        entry, backend=M._compile_backend(), fullgraph=True, options=M._COMPILE_OPTIONS
    )


def to_dev(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().to(DEVICE)


def run_nki(kernel_fn, *args, grid: int = 0):
    """Run `kernel_fn(*args)` on the device; returns CPU tensors."""
    dev_args = [to_dev(a) if isinstance(a, torch.Tensor) else a for a in args]
    out = _entry(kernel_fn, grid)(*dev_args)
    if isinstance(out, (tuple, list)):
        return tuple(o.cpu() for o in out)
    return out.cpu()


def compile_fn(fn):
    """A torch function as a device graph (for torch-side references on device)."""
    return M._compile_fn(fn)


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, tol_rel: float = 1e-2,
            tol_r: float | None = None) -> bool:
    """Print the metrics and return whether `got` is close to `ref`."""
    g = got.detach().float().reshape(-1)
    r = ref.detach().float().reshape(-1)
    if g.shape != r.shape:
        print(f"[{name}] SHAPE MISMATCH got {tuple(got.shape)} ref {tuple(ref.shape)}")
        return False
    err = g - r
    max_abs = float(err.abs().max()) if g.numel() else 0.0
    rnorm = float(r.norm())
    rel = float(err.norm()) / max(rnorm, 1e-30)
    cos = float(torch.nn.functional.cosine_similarity(g, r, dim=0)) if rnorm > 0 else 1.0
    # bf16 rounding noise of the reference itself: one bf16 ulp is 2^-8 relative.
    noise = float((r.to(torch.bfloat16).float() - r).norm())
    r_ratio = float(err.norm()) / max(noise, 1e-30)
    ok = rel <= tol_rel and (tol_r is None or r_ratio <= tol_r)
    nan = bool(torch.isnan(g).any())
    print(
        f"[{name}] max_abs={max_abs:.4g} rel_l2={rel:.3e} cos={cos:.6f} "
        f"R={r_ratio:.2f} nan={nan} -> {'OK' if ok and not nan else 'FAIL'}",
        flush=True,
    )
    return ok and not nan


def compile_only(kernel_fn, *args, grid: int = 2):
    """Compile `kernel_fn(*args)` through lite's `compile_nki` directly (no dynamo), so
    an NKI tracer/compiler error surfaces with its own traceback. Returns the result."""
    import inspect

    from libtorch_neuronx_lite.nki.nki_compile import compile_nki

    func = kernel_fn.func if hasattr(kernel_fn, "func") else kernel_fn
    names = list(inspect.signature(func).parameters)
    meta = {}
    for name, a in zip(names, args):
        meta[name] = torch.empty(a.shape, dtype=a.dtype, device="meta") if isinstance(a, torch.Tensor) else a
    return compile_nki(func, meta, (grid,))
