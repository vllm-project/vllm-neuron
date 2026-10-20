# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for accuracy validation modules."""

import re


def natural_sort_key(name: str):
    """Sort key for natural ordering (layers.2 before layers.10)."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]
