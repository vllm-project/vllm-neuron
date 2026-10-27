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

`layer` returns an optimized kernel instead where one exists: every (phase, archetype) of
the flat scheme has one, each a package of its own under `tp64/` named `<archetype>_<phase>`
with its own helper copies. The plain r2 / r1 blocks' (`r2_prefill`, `r2_decode`,
`r1_prefill`, `r1_decode`) came out of their workspaces' AutoHelix loops; every other
archetype's kernel is the nearest of those -- r1_prefill (which carries r2_prefill's
optimizations) for prefill, r2_decode for the r0 / r2 archetypes' decode, r1_decode for the r1
ones' -- with the generic body's sections that archetype has (engram, tap, compressor, index
scores / top-k, ratio 0). `DEEPSEEK_V41_OPT_KERNELS` selects which of them run (see
`_enabled`); the others fall back to the generic body.

Archetypes (`arch` = `(ratio, is_kv_src, is_idx_src, has_engram, tap)`) are taken
from the config at first use; the names spell them `r<ratio>`, `src` (kv source),
`idx` (index source), `e` (engram), `tap`.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import linecache
import os

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


#: The flat scheme's optimized block kernels: (phase, arch, scheme) -> the package (its
#: `kernels/tp64/<kind>/` directory, which is also its `mk_model.kernel_version` kind: the hash
#: covers that package's own files only). Each package's `<kind>.py` defines the kernel
#: `layer_<phase>_<archetype>_tp64`: the generic body's positional signature and outputs.
_KIND = {
    ("prefill", (0, 0, 0, 0, 0), "tp64"): "r0_prefill",
    ("prefill", (0, 0, 0, 1, 0), "tp64"): "r0_e_prefill",
    ("prefill", (2, 1, 1, 0, 0), "tp64"): "r2_src_idx_prefill",
    ("prefill", (2, 0, 0, 0, 0), "tp64"): "r2_prefill",
    ("prefill", (2, 1, 1, 1, 0), "tp64"): "r2_src_idx_e_prefill",
    ("prefill", (1, 1, 1, 0, 0), "tp64"): "r1_src_idx_prefill",
    ("prefill", (1, 0, 0, 0, 0), "tp64"): "r1_prefill",
    ("prefill", (1, 0, 1, 0, 0), "tp64"): "r1_idx_prefill",
    ("prefill", (1, 0, 0, 0, 1), "tp64"): "r1_tap_prefill",
    ("decode", (0, 0, 0, 0, 0), "tp64"): "r0_decode",
    ("decode", (0, 0, 0, 1, 0), "tp64"): "r0_e_decode",
    ("decode", (2, 1, 1, 0, 0), "tp64"): "r2_src_idx_decode",
    ("decode", (2, 0, 0, 0, 0), "tp64"): "r2_decode",
    ("decode", (2, 1, 1, 1, 0), "tp64"): "r2_src_idx_e_decode",
    ("decode", (1, 1, 1, 0, 0), "tp64"): "r1_src_idx_decode",
    ("decode", (1, 0, 0, 0, 0), "tp64"): "r1_decode",
    ("decode", (1, 0, 1, 0, 0), "tp64"): "r1_idx_decode",
    ("decode", (1, 0, 0, 0, 1), "tp64"): "r1_tap_decode",
}
OPTIMIZED_KINDS = tuple(sorted(set(_KIND.values())))
#: (phase, arch, scheme) -> the optimized kernel (`@nki.jit`)
_OPTIMIZED = {
    key: getattr(importlib.import_module(f".{key[2]}.{kind}.{kind}", __package__),
                 f"layer_{key[0]}_{arch_name(key[1])}_{key[2]}")
    for key, kind in _KIND.items()
}


def _enabled() -> frozenset:
    """The optimized kernels in use. `DEEPSEEK_V41_OPT_KERNELS`: unset or "all" -> every
    one, "none" / "0" -> none (the generic body everywhere: A/B, debugging), else a comma
    list of kinds (`r2_prefill,r1_decode`). `DEEPSEEK_V41_R2_PREFILL_OPT=0` (the original
    switch) additionally drops `r2_prefill`."""
    spec = os.environ.get("DEEPSEEK_V41_OPT_KERNELS", "all").strip().lower()
    if spec in ("", "all", "1"):
        kinds = set(OPTIMIZED_KINDS)
    elif spec in ("none", "0"):
        kinds = set()
    else:
        kinds = {k.strip() for k in spec.split(",") if k.strip()}
        unknown = kinds - set(OPTIMIZED_KINDS)
        if unknown:
            raise ValueError(f"DEEPSEEK_V41_OPT_KERNELS: unknown kernels {sorted(unknown)}; "
                             f"known: {', '.join(OPTIMIZED_KINDS)}")
    if os.environ.get("DEEPSEEK_V41_R2_PREFILL_OPT", "1") in ("", "0"):
        kinds.discard("r2_prefill")
    return frozenset(kinds)


_ENABLED = _enabled()


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
    """The block kernel for `phase` x `arch` x `scheme`: an optimized kernel where
    one exists (`_OPTIMIZED`), else the generic body specialized."""
    key = (phase, tuple(int(v) for v in arch), scheme)
    if _KIND.get(key) in _ENABLED:
        return _OPTIMIZED[key]
    return layer_generic(*key)


def layer_generic(phase: str, arch, scheme: str):
    """The generic body specialized to `phase` x `arch` x `scheme`."""
    assert phase in PHASES and scheme in SCHEMES
    arch = tuple(int(v) for v in arch)
    return specialize(pkg(scheme).layer_kernel.layer_body, f"layer_{phase}_{arch_name(arch)}_{scheme}",
                      (("phase", phase), ("arch", arch), ("scheme", scheme)))


def version_kind(phase: str, arch, scheme: str) -> str:
    """The `mk_model.kernel_version` kind of the block kernel `layer` returns."""
    kind = _KIND.get((phase, tuple(int(v) for v in arch), scheme))
    return kind if kind in _ENABLED else "block"


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
