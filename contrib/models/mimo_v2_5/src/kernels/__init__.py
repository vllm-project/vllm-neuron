# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.5-specific NKI kernels (kept in the contrib folder by design)."""

from .kv_cache_write import write_paged_kv_cache

__all__ = ["write_paged_kv_cache"]
