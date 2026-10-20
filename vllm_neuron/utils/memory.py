# SPDX-License-Identifier: Apache-2.0
"""Host and cgroup memory helpers for CPU-mode KV cache sizing.

Why cgroup first:
- In containers, psutil reports host memory, which can significantly exceed the
  process/container memory quota.
- CPU-mode KV sizing should respect the process' effective limit, so we prefer
  cgroup limit/usage when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import psutil

# cgroup v2 paths
_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_USAGE = Path("/sys/fs/cgroup/memory.current")

# cgroup v1 paths
_CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")

# Sentinel value the kernel uses for "no limit" in cgroup v1.
_NO_LIMIT_V1 = 9223372036854771712


def _read_cgroup_memory_limit() -> Optional[int]:
    """Return the cgroup memory limit in bytes, or None if unconstrained."""
    if _CGROUP_V2_MAX.exists():
        # cgroup v2: "max" means no explicit memory limit.
        raw = _CGROUP_V2_MAX.read_text().strip()
        if raw != "max":
            return int(raw)
        return None

    if _CGROUP_V1_LIMIT.exists():
        # cgroup v1 encodes "no limit" as a very large sentinel value.
        value = int(_CGROUP_V1_LIMIT.read_text().strip())
        if value < _NO_LIMIT_V1:
            return value
        return None

    return None


def _read_cgroup_memory_usage() -> Optional[int]:
    """Return current cgroup memory usage in bytes, or None if unavailable."""
    for path in (_CGROUP_V2_USAGE, _CGROUP_V1_USAGE):
        if path.exists():
            return int(path.read_text().strip())
    return None


def get_available_memory_bytes() -> int:
    """Return usable memory in bytes, respecting container limits.

    Priority:
      1. cgroup limit - cgroup usage   (container-aware)
      2. psutil MemAvailable            (bare metal / VM fallback)
    """
    limit = _read_cgroup_memory_limit()
    usage = _read_cgroup_memory_usage()

    if limit is not None and usage is not None:
        # Guard against transient sampling races where usage > limit.
        return max(limit - usage, 0)

    # Missing or unconstrained cgroup info: fall back to host-level available memory.
    return psutil.virtual_memory().available
