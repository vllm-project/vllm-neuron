# SPDX-License-Identifier: Apache-2.0
"""Compiler-argument helper for the Logical NeuronCore (LNC) configuration.

Read the degree itself with
:func:`vllm_neuron.utils.hardware_config.get_logical_nc_config`.
"""

from __future__ import annotations

from vllm_neuron.utils.hardware_config import (
    _DEFAULT_LNC_CONFIG,
    get_logical_nc_config,
)


def get_explicit_compiler_lnc_arg(
    configured_lnc: int | str | None = None,
) -> str | None:
    """Return the ``--logical-nc-config=...`` arg when the compiler needs it.

    ``neuronx-cc`` compiles for LNC-2 by default, while NRT reads
    ``NEURON_LOGICAL_NC_CONFIG``. A caller that runs at LNC-1 must pass this flag,
    else the NEFF holds the wrong core grouping and fails at load, not at compile.
    The flag also partitions the NEFF cache by degree, because the cache key covers
    the canonicalized compiler args. A caller that passes it needs no separate
    cache root.

    Returns ``None`` for the degree the compiler already defaults to. The arg list
    of an LNC-2 deployment then stays the same as before this helper existed, which
    keeps every cached NEFF valid.

    Args:
        configured_lnc: The degree to compile for. Reads
            ``NEURON_LOGICAL_NC_CONFIG`` when omitted.

    Returns:
        str | None: The compiler arg, or ``None`` when the compiler default
        already matches the configured degree.

    Raises:
        ValueError: The environment holds a degree other than 1 or 2. An
            explicitly passed degree comes from a caller that resolved it
            already, so this function trusts it.

    Example:
        >>> import os
        >>> os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
        >>> get_explicit_compiler_lnc_arg()
        '--logical-nc-config=1'
        >>> get_explicit_compiler_lnc_arg(2) is None
        True
    """
    # Resolve through the one degree reader, so the 1-or-2 check is shared.
    lnc = get_logical_nc_config() if configured_lnc is None else int(configured_lnc)
    # _DEFAULT_LNC_CONFIG is also the degree neuronx-cc compiles for when the flag
    # is absent, so one constant covers the runtime default and the compiler
    # default. A flag that repeats it changes no compiled output, but the compile
    # cache key covers the compiler args, so it does change the key.
    if lnc == _DEFAULT_LNC_CONFIG:
        return None
    return f"--logical-nc-config={lnc}"
