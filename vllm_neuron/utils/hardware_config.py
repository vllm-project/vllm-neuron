# SPDX-License-Identifier: Apache-2.0
"""
Hardware Configuration for Trainium Instances

This module provides hardware-specific configurations for different Trainium instance types,
including EFA (Elastic Fabric Adapter) BDF mappings for optimal NUMA placement.

BDF (Bus:Device:Function) addresses are stable PCI identifiers that don't change,
while EFA interface names (e.g., rdmapXXXs0, efa_X) can vary between boots/driver versions.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceConfig:
    """Configuration for a specific instance type."""

    instance_family: str
    max_neuron_devices: int
    efa_bdf_mapping: tuple[str, ...]


# EFA BDF mappings per instance family
# Each tuple maps Neuron device index to its corresponding EFA BDF address
_INSTANCE_CONFIGS: dict[str, InstanceConfig] = {
    "trn2": InstanceConfig(
        instance_family="trn2",
        max_neuron_devices=16,
        efa_bdf_mapping=(
            "0000:c9:00.0",  # ND 0
            "0000:b4:00.0",  # ND 1
            "0000:b3:00.0",  # ND 2
            "0000:ca:00.0",  # ND 3
            "0000:6c:00.0",  # ND 4
            "0000:57:00.0",  # ND 5
            "0000:56:00.0",  # ND 6
            "0000:6d:00.0",  # ND 7
            "0000:98:00.0",  # ND 8
            "0000:83:00.0",  # ND 9
            "0000:82:00.0",  # ND 10
            "0000:99:00.0",  # ND 11
            "0000:f5:00.0",  # ND 12
            "0000:e0:00.0",  # ND 13
            "0000:df:00.0",  # ND 14
            "0000:f6:00.0",  # ND 15
        ),
    ),
    "trn3pd": InstanceConfig(
        instance_family="trn3pd",
        max_neuron_devices=16,
        efa_bdf_mapping=(
            "0000:be:00.0",  # ND 0
            "0000:a9:00.0",  # ND 1
            "0000:a8:00.0",  # ND 2
            "0000:bf:00.0",  # ND 3
            "0000:61:00.0",  # ND 4
            "0000:4c:00.0",  # ND 5
            "0000:4b:00.0",  # ND 6
            "0000:62:00.0",  # ND 7
            "0000:8d:00.0",  # ND 8
            "0000:78:00.0",  # ND 9
            "0000:77:00.0",  # ND 10
            "0000:8e:00.0",  # ND 11
            "0000:ea:00.0",  # ND 12
            "0000:d5:00.0",  # ND 13
            "0000:d4:00.0",  # ND 14
            "0000:eb:00.0",  # ND 15
        ),
    ),
    "trn3pds": InstanceConfig(
        instance_family="trn3pds",
        max_neuron_devices=16,
        efa_bdf_mapping=(
            "0000:a9:00.0",  # ND 0
            "0000:a8:00.0",  # ND 1
            "0000:bf:00.0",  # ND 2
            "0000:be:00.0",  # ND 3
            "0000:d5:00.0",  # ND 4
            "0000:d4:00.0",  # ND 5
            "0000:eb:00.0",  # ND 6
            "0000:ea:00.0",  # ND 7
            "0000:4c:00.0",  # ND 8
            "0000:4b:00.0",  # ND 9
            "0000:62:00.0",  # ND 10
            "0000:61:00.0",  # ND 11
            "0000:78:00.0",  # ND 12
            "0000:77:00.0",  # ND 13
            "0000:8e:00.0",  # ND 14
            "0000:8d:00.0",  # ND 15
        ),
    ),
}


def get_instance_family() -> str:
    """
    Get the instance family.

    Resolution order:
      1. If ``VLLM_NEURON_EFA_INSTANCE_FAMILY`` is set, its value is returned.
      2. Otherwise the family is read from
         ``/sys/devices/virtual/dmi/id/product_name`` and mapped by prefix:
         any ``trn3*`` resolves to ``"trn3pds"`` and any ``trn2*`` to ``"trn2"``.

    Returns:
        str: Instance family (e.g., "trn2", "trn3pd", "trn3pds")

    Raises:
        RuntimeError: If the instance family cannot be determined
    """
    # Env var access goes through envs (see vllm_neuron/envs.py); imported
    # lazily so hardware_config stays importable without the full package.
    from vllm_neuron import envs

    override = envs.VLLM_NEURON_EFA_INSTANCE_FAMILY
    if override:
        return override

    sysfs_path = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(sysfs_path) as f:
            product_name = f.read().strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot read instance type from {sysfs_path}") from e

    # Extract family from "trn2.48xlarge" -> "trn2", then default by prefix.
    family = product_name.split(".")[0]
    if family.startswith("trn3"):
        return "trn3pds"
    if family.startswith("trn2"):
        return "trn2"
    return family


def get_efa_bdf_mapping() -> tuple[str, ...]:
    """
    Get the EFA BDF mapping for the current instance.

    Returns:
        tuple[str, ...]: Tuple of BDF addresses indexed by Neuron device number

    Raises:
        RuntimeError: If the instance family cannot be determined
        ValueError: If the instance family is not supported
    """
    family = get_instance_family()

    if family not in _INSTANCE_CONFIGS:
        supported = ", ".join(sorted(_INSTANCE_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported instance family: {family}. Supported: {supported}"
        )

    return _INSTANCE_CONFIGS[family].efa_bdf_mapping


def get_efa_interface_from_bdf(bdf: str) -> str:
    """
    Look up the EFA interface name from a BDF address via sysfs.

    Args:
        bdf: PCI Bus:Device:Function address (e.g., "0000:c9:00.0")

    Returns:
        str: EFA interface name (e.g., "rdmap201s0")

    Raises:
        RuntimeError: If no EFA interface is found for the given BDF
    """
    infiniband_path = f"/sys/bus/pci/devices/{bdf}/infiniband"
    try:
        interfaces = os.listdir(infiniband_path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"No EFA device found at {infiniband_path} for BDF {bdf}. "
            f"This is expected on instances without EFA (e.g. trn3 3xlarge). "
            f"EFA affinity is a CPU performance optimization, not a "
            f"correctness requirement; set NEURON_SKIP_EFA_AFFINITY=1 to skip "
            f"it on instances without EFA. If you are using an instance that "
            f"supports EFA, we recommend installing EFA to improve performance."
        ) from e

    if not interfaces:
        raise RuntimeError(
            f"No infiniband interfaces found in {infiniband_path} for BDF {bdf}. "
            f"The EFA device may not be properly initialized."
        )

    return interfaces[0]


# Default Logical NC config. Controls how many physical NCs are grouped
# into one logical core. Hardcoded to 2 for current Trainium chips.
_DEFAULT_LNC_CONFIG = 2
_SUPPORTED_LNC_CONFIGS = (1, 2)

# Number of physical NeuronCores that share one HBM stack. This is a hardware
# property of the chip, and it is not the LNC degree. Both values are 2 on the
# current Trainium chips. A future chip can change one value and keep the other.
_PHYSICAL_NCS_PER_HBM_STACK = 2


def get_logical_nc_config() -> int:
    """Return the logical NeuronCore config the runtime (NRT) honours.

    Reads the ``NEURON_LOGICAL_NC_CONFIG`` environment variable — the same value
    NRT reads to decide how many physical NeuronCores are fused into one logical
    core — defaulting to ``_DEFAULT_LNC_CONFIG`` (2) when unset. The compiler must
    be told the SAME value via ``--logical-nc-config`` so the compiled NEFF's core
    grouping matches what NRT runs with; callers that need lnc=1 (e.g. the GPT-OSS
    giga-kernel decode path) set the env var to 1.

    The variable is process-local, so a heterogeneous prefill/decode server can
    run one degree in each process. NKI launch grids read this value too, which
    is why an unsupported degree fails here rather than inside the compiler.

    Returns:
        int: The logical NC config (1 or 2).

    Raises:
        ValueError: The environment variable holds a degree other than 1 or 2.

    Examples:
        >>> import os
        >>> os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
        >>> get_logical_nc_config()
        1
    """
    lnc_config = int(
        os.environ.get("NEURON_LOGICAL_NC_CONFIG", str(_DEFAULT_LNC_CONFIG))
    )
    if lnc_config not in _SUPPORTED_LNC_CONFIGS:
        raise ValueError(
            "Only LNC-1 and LNC-2 are supported; got "
            f"NEURON_LOGICAL_NC_CONFIG={lnc_config!r}. Set the variable to 1 or 2, "
            "or leave it unset for the default of 2."
        )
    return lnc_config


def get_logical_cores_per_hbm_stack() -> int:
    """Return the number of logical NeuronCores that share one HBM stack.

    One HBM stack serves ``_PHYSICAL_NCS_PER_HBM_STACK`` (2) physical NeuronCores
    on the current Trainium chips. A logical core holds ``lnc`` physical cores, so
    an LNC-2 logical core owns a full stack, and 2 LNC-1 logical cores share one.
    The runtime reports the stack to each logical core that holds a part of it. A
    caller that sizes a per-core memory budget must divide by this factor. Else 2
    LNC-1 cores each plan for memory that the other also claims.

    Returns:
        int: 1 at LNC 2, and 2 at LNC 1.

    Raises:
        ValueError: ``NEURON_LOGICAL_NC_CONFIG`` holds a degree other than 1 or 2.

    Examples:
        >>> import os
        >>> os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
        >>> get_logical_cores_per_hbm_stack()
        2
        >>> os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
        >>> get_logical_cores_per_hbm_stack()
        1
    """
    return _PHYSICAL_NCS_PER_HBM_STACK // get_logical_nc_config()


def get_efa_interface(local_rank: int, visible_devices: list[int]) -> str:
    """
    Get the EFA interface name for the given local rank.

    Encapsulates the full logical-core → Neuron Device → host device → BDF
    → EFA interface resolution. The caller only needs to provide the local
    rank and the visible device list; all hardware-specific computation is
    handled here.

    Args:
        local_rank: The worker's local rank index.
        visible_devices: List of visible Neuron device indices (logical cores),
            as returned by the worker's _get_visible_devices().

    Returns:
        str: The EFA interface name (e.g., "rdmap201s0")

    Raises:
        RuntimeError: If the Neuron device cannot be accessed or EFA
            interface not found
        IndexError: If the computed device index is out of bounds
    """
    try:
        lnc = visible_devices[local_rank]
    except IndexError:
        lnc = local_rank

    lnc_config = get_logical_nc_config()
    nd = lnc * lnc_config // 8

    try:
        device_stat = os.stat(f"/dev/neuron{nd}")
    except OSError as e:
        raise RuntimeError(
            f"Cannot access /dev/neuron{nd}: {e}. "
            f"Verify device exists and permissions are correct. "
            f"local_rank={local_rank}, visible_devices={visible_devices}"
        ) from e

    host_nd = os.minor(device_stat.st_rdev)
    efa_bdf_mapping = get_efa_bdf_mapping()

    if host_nd >= len(efa_bdf_mapping):
        raise IndexError(
            f"Computed Neuron device index {host_nd} exceeds available EFA BDF "
            f"mappings (max index: {len(efa_bdf_mapping) - 1}). "
            f"local_rank={local_rank}, visible_devices={visible_devices}, "
            f"lnc_config={lnc_config}"
        )

    bdf = efa_bdf_mapping[host_nd]
    return get_efa_interface_from_bdf(bdf)


def parse_range_list(range_list_str: str) -> list[int]:
    """
    Parse a range list string into a list of integers.

    Supports both single numbers and ranges. Used for parsing
    NEURON_VISIBLE_DEVICES and CPU lists from sysfs.

    Args:
        range_list_str: String like "0,1,2" or "0-3,5,7-9"

    Returns:
        list[int]: Expanded list of integers

    Examples:
        >>> parse_range_list("0-3")
        [0, 1, 2, 3]
        >>> parse_range_list("0,1,5")
        [0, 1, 5]
        >>> parse_range_list("0-2,5,7-9")
        [0, 1, 2, 5, 7, 8, 9]
    """
    result = []
    for start_end in range_list_str.split(","):
        try:
            start, end = start_end.split("-")
        except ValueError:
            start = start_end
            end = start_end
        result.extend(range(int(start), int(end) + 1))
    return result
