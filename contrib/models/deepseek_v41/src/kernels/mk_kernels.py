# SPDX-License-Identifier: Apache-2.0
"""One `@nki.jit` megakernel per phase, block archetype and parallelization scheme.

The bodies (`layer_kernel.layer_body`, `draft_kernel.draft_body`, `pp_kernels.*`)
take their configuration as keyword constants. This module makes a distinct
`@nki.jit` entry point for every combination the deployment uses, each a plain
function with the body's positional signature and a name that says what it is:

    layer_prefill_r2_src_tp64, layer_decode_r1_idx_tap_tp64, layer_decode_r0_e_tp8pp8,
    draft_prefill_tp64, draft_decode_tp8pp8, pp_boundary_even, pp_boundary_odd, ...

so that a kernel is one function with its constants baked rather than one
function branching on flags, and the compiled artifacts carry the name.
`entry(kernel)` wraps one through `wrap_nki(...)[2]` + `torch.compile` the way
`mk_model` runs every megakernel.

Archetypes (`arch` = `(ratio, is_kv_src, is_idx_src, has_engram, tap)`) are taken
from the config at first use; the names spell them `r<ratio>`, `src` (kv source),
`idx` (index source), `e` (engram), `tap`.
"""

from __future__ import annotations

import functools
import inspect
import linecache

import nki

from . import tp64 as _tp64
from . import tp8pp8 as _tp8pp8

#: The scheme's kernel package: its own block kernel, drafter and helpers.
_PKG = {"tp64": _tp64, "tp8pp8": _tp8pp8}


def pkg(scheme: str):
    """The kernel package of `scheme` (`kernels/tp64`, `kernels/tp8pp8`)."""
    return _PKG[scheme]

SCHEMES = ("tp64", "tp8pp8")
PHASES = ("prefill", "decode")


def arch_name(arch) -> str:
    ratio, kv, idx, eng, tap = arch
    parts = [f"r{ratio}"]
    if kv:
        parts.append("src")
    if idx:
        parts.append("idx")
    if eng:
        parts.append("e")
    if tap:
        parts.append("tap")
    return "_".join(parts)


def _positional_names(body) -> list[str]:
    return [p.name for p in inspect.signature(body).parameters.values()
            if p.default is inspect.Parameter.empty]


@functools.lru_cache(maxsize=None)
def specialize(body, name: str, consts: tuple) -> object:
    """A `@nki.jit` entry point `name(<body's positional args>)` that calls `body`
    with `consts` (a tuple of `(keyword, value)`) baked in. The generated source is
    registered with `linecache`, so tracebacks and source lookups work."""
    names = _positional_names(body)
    call = ", ".join(names + [f"{k}={v!r}" for k, v in consts])
    src = f"def {name}({', '.join(names)}):\n    return _body({call})\n"
    filename = f"<deepseek_v41.kernels.mk_kernels/{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns = {"_body": body}
    exec(compile(src, filename, "exec"), ns)
    fn = ns[name]
    fn.__module__ = __name__
    fn.__qualname__ = name
    return nki.jit(fn)


def layer(phase: str, arch, scheme: str):
    """The block kernel for `phase` x `arch` x `scheme`."""
    assert phase in PHASES and scheme in SCHEMES
    arch = tuple(int(v) for v in arch)
    return specialize(pkg(scheme).layer_kernel.layer_body, f"layer_{phase}_{arch_name(arch)}_{scheme}",
                      (("phase", phase), ("arch", arch), ("scheme", scheme)))


def draft(phase: str, scheme: str):
    """The drafter entry of a scheme: the flat scheme's kernel drafts one sequence
    per launch, the pipeline scheme's every sequence of the step in one launch
    (`pkg(scheme).draft_kernel`)."""
    assert phase in PHASES and scheme in SCHEMES
    return specialize(pkg(scheme).draft_kernel.draft_body, f"draft_{phase}_{scheme}",
                      (("phase", phase), ("scheme", scheme)))


def boundary(parity: int):
    """The stage-boundary kernel of the even / odd sending-stage pair layout. One
    program for both sides of a boundary: the role is a runtime input."""
    return specialize(_tp8pp8.pp_kernels.boundary_body,
                      f"pp_boundary_{'even' if parity == 0 else 'odd'}", ())
