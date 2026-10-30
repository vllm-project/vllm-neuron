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

A fused decode launch (`fused`: several block kernels in sequence) can have an optimized
replacement, `tp64/fused/<launch>/`, optimized as a whole; `optimized_fused` returns it for
the plan it was written from (its `plan.json`).
"""

from __future__ import annotations

import contextlib
import functools
import glob
import hashlib
import importlib
import inspect
import json
import linecache
import os
import sys

import nki
import nki.language as _nl

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


def draft_batched(phase: str):
    """The flat scheme's drafter for every sequence of a step in one launch: the batched
    body (`tp8pp8/draft_kernel.draft_body`, whose groups are the ranks the head and sampler
    run on -- the world here) specialized for tp64. `mk_model.run_draft` launches it
    unless `DEEPSEEK_V41_DRAFT_BATCHED=0` (one launch per sequence: `draft(phase, "tp64")`).
    Its `mk_model.kernel_version` is `kernel_version("draft", "tp8pp8")`, the package it is
    built from."""
    assert phase in PHASES
    return specialize(_tp8pp8.draft_kernel.draft_body, f"draft_{phase}_batched_tp64",
                      (("phase", phase), ("scheme", "tp64")))


@functools.lru_cache(maxsize=None)
def head_greedy():
    """The flat scheme's greedy head (`tp64/head_kernel.head_greedy_body`: collapse, final
    norm, vocab-parallel head, the global argmax) as a member of a decode step's last
    fused launch (`mk_model.run_fused`)."""
    from .tp64 import head_kernel

    return specialize(head_kernel.head_greedy_body, "head_greedy_decode_tp64", ())


@functools.lru_cache(maxsize=None)
def embed_expand():
    """The flat scheme's token embedding and hyper-connection expand
    (`tp64/head_kernel.embed_expand_body`) as the first member of a decode step's first
    fused launch (`mk_model.run_fused`)."""
    from .tp64 import head_kernel

    return specialize(head_kernel.embed_expand_body, "embed_expand_decode_tp64", ())


def boundary(parity: int):
    """The stage-boundary kernel of the even / odd sending-stage pair layout. One
    program for both sides of a boundary: the role is a runtime input."""
    return specialize(_tp8pp8.pp_kernels.boundary_body,
                      f"pp_boundary_{'even' if parity == 0 else 'odd'}", ())


#: The module flags that make a decode kernel return copies of the caches it was given
#: (the r2_decode-based kernels' / the r1_decode-based ones'): off inside a fused launch,
#: where a member's returned caches are not outputs (the caches are updated in place).
_CACHE_COPY_FLAGS = ("CACHE_COPY_EARLY", "CACHE_COPY_AT_AR")


@contextlib.contextmanager
def _fused_member(index: int, kernel):
    """Trace member `index` of a fused launch: its named scratch tensors prefixed
    `L<index>_` (a kernel's scratch names are unique only within one block), its unnamed
    shared-HBM ones named `L<index>_out<n>`, and its cache copies off."""
    mod = sys.modules[kernel.func.__module__]
    saved = {f: getattr(mod, f) for f in _CACHE_COPY_FLAGS if hasattr(mod, f)}
    for f in saved:
        setattr(mod, f, False)
    orig = _nl.ndarray
    anon = [0]

    def ndarray(*args, **kwargs):
        if kwargs.get("name"):
            kwargs = dict(kwargs, name=f"L{index}_{kwargs['name']}")
        elif kwargs.get("buffer") is _nl.shared_hbm:
            # an unnamed shared-HBM tensor (a member's outputs: its stream, top-k, tap) is
            # named too: left to the compiler, two members' same-shape ones (two index
            # sources' top-k) were not kept apart (wrong top-k, measured at T=60, 8k)
            anon[0] += 1
            kwargs = dict(kwargs, name=f"L{index}_out{anon[0]}")
        return orig(*args, **kwargs)

    _nl.ndarray = ndarray
    try:
        yield
    finally:
        _nl.ndarray = orig
        for f, v in saved.items():
            setattr(mod, f, v)


@functools.lru_cache(maxsize=None)
def fused(name: str, kernels: tuple, plan: tuple, nparams: int, returns: tuple):
    """One `@nki.jit` launch of several block kernels in sequence (`name(p0 .. p<n-1>)`).

    `kernels[m]` is member m's kernel (an optimized block kernel); `plan[m]` its
    positional arguments, each `("p", j)` (the launch's parameter j) or `("o", i, k)`
    (output k of an earlier member i: the stream / pre_mix a block hands the next, an
    index source's top-k). A tensor several members read (a kv source's caches, the
    dummies) is one parameter, so a write to it in one member is ordered before the
    reads in the next. `returns` are the launch's outputs, `("o", i, k)` each. The
    members' own outputs stay internal: their cache copies are off (`_fused_member`)."""
    params = [f"p{j}" for j in range(nparams)]

    def ref(src):
        return f"p{src[1]}" if src[0] == "p" else f"o{src[1]}[{src[2]}]"

    # (the members' kernels named in the source: the compile caches key a kernel on it)
    members = ", ".join(f"{k.func.__module__}.{k.func.__name__}" for k in kernels)
    lines = [f"def {name}({', '.join(params)}):", f'    """fused: {members}"""']
    for m, srcs in enumerate(plan):
        lines.append(f"    with _member({m}, _k{m}):")
        lines.append(f"        o{m} = _k{m}.func({', '.join(ref(s) for s in srcs)})")
    lines.append(f"    return {', '.join(ref(r) for r in returns)}")
    src = "\n".join(lines) + "\n"
    filename = f"<deepseek_v41.kernels.mk_kernels/{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns = {"_member": _fused_member, **{f"_k{m}": k for m, k in enumerate(kernels)}}
    exec(compile(src, filename, "exec"), ns)
    fn = ns[name]
    fn.__module__ = __name__
    fn.__qualname__ = name
    return nki.jit(fn)



#: The optimized fused decode launches: `tp64/fused/<launch>/`, one of a decode step's launches
#: (`mk_model.run_fused`) written out and optimized as a whole in an AutoHelix loop. Its
#: `<launch>.py` calls its own copies of the members' kernels (packages next to it, which may
#: share tables and on-chip tiles across blocks and prefetch the next block's weights) and
#: takes the parameters of the model's launch it replaces; `plan.json` is that launch's plan
#: (each member's kernel and arguments, the parameter count, the returns). `optimized_fused`
#: returns it for a plan that matches exactly. `DEEPSEEK_V41_FUSED_OPT`: unset or "all" ->
#: every one, "none" / "0" -> none (the generated launch everywhere: A/B, debugging), else a
#: comma list of launch names (`layers_decode_L24_L31_tp64,...`).
_FUSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp64", "fused")
_MK_HOST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mk_host.py")
FUSED_LAUNCHES = tuple(sorted(d for d in os.listdir(_FUSED_DIR)
                              if os.path.isfile(os.path.join(_FUSED_DIR, d, "plan.json"))))


def _fused_enabled() -> frozenset:
    spec = os.environ.get("DEEPSEEK_V41_FUSED_OPT", "all").strip()
    if spec.lower() in ("", "all", "1"):
        return frozenset(FUSED_LAUNCHES)
    if spec.lower() in ("none", "0"):
        return frozenset()
    names = {n.strip() for n in spec.split(",") if n.strip()}
    unknown = names - set(FUSED_LAUNCHES)
    if unknown:
        raise ValueError(f"DEEPSEEK_V41_FUSED_OPT: unknown launches {sorted(unknown)}; "
                         f"known: {', '.join(FUSED_LAUNCHES)}")
    return frozenset(names)


_FUSED_ENABLED = _fused_enabled()


def plan_signature(kernels: tuple, plan: tuple, nparams: int, returns: tuple) -> dict:
    """A fused plan (`fused`'s arguments) as `plan.json` spells it: each member's kernel
    (`<package>.<module>:<name>`; the generic body's and the head's `kernels.mk_kernels:<name>`)
    and arguments (`p<j>` / `o<m>.<k>`)."""
    def ref(s):
        return f"p{s[1]}" if s[0] == "p" else f"o{s[1]}.{s[2]}"

    return {"nparams": int(nparams),
            "members": [{"kernel": ".".join(k.func.__module__.rsplit(".", 2)[-2:]) + ":" + k.func.__name__,
                         "args": " ".join(ref(s) for s in srcs)} for k, srcs in zip(kernels, plan)],
            "returns": " ".join(ref(r) for r in returns)}


def _tree_hash(root: str) -> int:
    """A hash of an optimized launch's sources (every file of its package, and `mk_host.py`,
    which builds the index columns it reads), put in its entry's source: the NKI compile
    caches key a kernel on its own function source (see `mk_model.kernel_version`)."""
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    for path in files + [os.path.join(root, "plan.json"), _MK_HOST]:
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def optimized_fused(name: str, kernels: tuple, plan: tuple, nparams: int, returns: tuple):
    """`(kernel, None)`: the optimized launch `tp64/fused/<name>/` (`@nki.jit`) for this plan
    (`fused`'s arguments), else `(None, why not)` (no such launch, not enabled, or its
    `plan.json` differs from the plan)."""
    if name not in FUSED_LAUNCHES:
        return None, "no optimized launch"
    if name not in _FUSED_ENABLED:
        return None, "not in DEEPSEEK_V41_FUSED_OPT"
    root = os.path.join(_FUSED_DIR, name)
    with open(os.path.join(root, "plan.json")) as fh:
        want = json.load(fh)
    got = plan_signature(kernels, plan, nparams, returns)
    for key in ("nparams", "returns"):
        if got[key] != want[key]:
            return None, f"plan.json {key} {want[key]!r} != the model's {got[key]!r}"
    if len(got["members"]) != len(want["members"]):
        return None, f"plan.json has {len(want['members'])} members, the model's plan {len(got['members'])}"
    for m, (g, w) in enumerate(zip(got["members"], want["members"])):
        if g != w:
            return None, f"member {m}: plan.json {w['kernel']} {w['args']!r} != the model's {g['kernel']} {g['args']!r}"
    body = getattr(importlib.import_module(f".tp64.fused.{name}.{name}", __package__), name).func
    names = _positional_names(body)
    src = (f"def {name}({', '.join(names)}):\n"
           f'    """tp64/fused/{name} ({_tree_hash(root):08x})"""\n'
           f"    return _body({', '.join(names)})\n")
    filename = f"<deepseek_v41.kernels.mk_kernels/{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns = {"_body": body}
    exec(compile(src, filename, "exec"), ns)
    fn = ns[name]
    fn.__module__ = __name__
    fn.__qualname__ = name
    return nki.jit(fn), None
