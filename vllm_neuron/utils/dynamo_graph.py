# SPDX-License-Identifier: Apache-2.0
"""Helpers for functions that the plugin deliberately makes visible to Dynamo."""

import hashlib
import marshal

import torch


def _body_fingerprint(fn) -> str:
    """Return a digest of ``fn``'s compiled body.

    ``allow_in_graph`` emits the function as one node keyed on its name, so the
    graph hash never sees the body. Without this, editing the formula and keeping
    the torch version reuses the old artifact silently. Builtins have no
    ``__code__``, so they fall back to the name.
    """
    try:
        return hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()[:16]
    except (AttributeError, ValueError):
        name = "%s.%s" % (
            getattr(fn, "__module__", "?"),
            getattr(fn, "__qualname__", fn),
        )
        return hashlib.sha256(name.encode()).hexdigest()[:16]


def allow_in_graph_cacheable(fn):
    """Put ``fn`` in the FX graph and keep the enclosing graph cacheable.

    ``allow_in_graph`` emits ``fn`` as one ``call_function`` node, and
    AOTAutograd's ``check_node_safe`` rejects that node because a plain Python
    function is neither an ``OpOverload`` nor a builtin. One rejection discards
    the whole graph, silently: nothing is cached and every warm start retraces.

    So the two belong together. The value registered is a cache key, not a flag:
    the torch version plus a digest of the body, so an upgrade or an edit both
    invalidate it.

    Use this in place of ``torch.compiler.allow_in_graph`` on the Neuron routes.

    Args:
        fn: The function to make graph-visible.

    Returns:
        ``fn``, registered with Dynamo and marked cacheable.

    Example:
        >>> @allow_in_graph_cacheable
        ... def gelu(x):
        ...     return x
    """
    # Assign rather than setdefault: an edited body must replace the stale key,
    # and setdefault would keep the first one registered in the process.
    torch._inductor.config.unsafe_marked_cacheable_functions[
        f"{fn.__module__}.{fn.__name__}"
    ] = f"{torch.__version__}+{_body_fingerprint(fn)}"
    return torch.compiler.allow_in_graph(fn)
