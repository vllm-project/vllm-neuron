# SPDX-License-Identifier: Apache-2.0
"""Routing for the EPD Router: media-to-encoder affinity + pools.

The Router consistent-hashes each media item's mm_hash to a VE via hrw_pick
(rendezvous / highest-random-weight hashing). VEs and PDs are held in static
registries for the prototype.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass


def hrw_pick(key: str, candidates: Sequence[str]) -> str:
    """Select a candidate for key via rendezvous (highest-random-weight) hashing.

    For each candidate, a score is computed by hashing key together with the
    candidate id; the highest-scoring candidate wins. This is deterministic for a
    fixed (key, candidates), distributes keys evenly with no tuning, and remaps
    only ~1/N of keys when a candidate is added or removed (unlike hash(key) % N,
    which reshuffles nearly everything).

    Args:
        key: The content key to route on (the media item's mm_hash).
        candidates: The non-empty set of candidate ids to choose among.

    Returns:
        The id of the chosen candidate.

    Raises:
        ValueError: If candidates is empty.

    Example:
        >>> hrw_pick("img-hash-1", ["ve-0", "ve-1", "ve-2"])  # doctest: +SKIP
        've-2'
    """
    if not candidates:
        raise ValueError("hrw_pick requires at least one candidate")

    def score(candidate: str) -> tuple[bytes, str]:
        # Separate key and candidate with a NUL byte so e.g. ("a", "bc") and
        # ("ab", "c") cannot collide into the same hash input. Append the candidate id
        # as a secondary key so a digest tie resolves by id rather
        # than by candidate ordering — keeping the result order-independent.
        digest = hashlib.blake2b(
            key.encode("utf-8") + b"\x00" + candidate.encode("utf-8"),
            digest_size=8,
        )
        return digest.digest(), candidate

    return max(candidates, key=score)


@dataclass(frozen=True)
class Endpoint:
    """A pool member the Router talks to over HTTP.

    Args:
        id: Stable id used as the HRW candidate key (e.g. "ve-0"); must not
            change at runtime or warm-cache affinity breaks.
        host: Hostname or IP of the pool member's API server.
        port: Port of the pool member's API server.

    Example:
        >>> Endpoint(id="ve-0", host="127.0.0.1", port=8300).base_url
        'http://127.0.0.1:8300'
    """

    id: str
    host: str
    port: int

    @property
    def base_url(self) -> str:
        """The http://host:port base URL for this endpoint."""
        return f"http://{self.host}:{self.port}"


class VeRegistry:
    """Static set of Vision Encoders the Router routes media items to.

    ids() is the single source of truth for both the HRW candidate set and the
    per-id endpoint lookup, so the routing decision and the HTTP target can never
    drift apart. Media items are HRW-routed per mm_hash (see hrw_pick); items
    landing on the same VE are batched into one /encode POST.

    Example:
        >>> reg = VeRegistry([Endpoint("ve-0", "h", 8300), Endpoint("ve-1", "h", 8301)])
        >>> reg.ids()
        ['ve-0', 've-1']
        >>> reg.get("ve-1").port
        8301
    """

    def __init__(self, endpoints: Sequence[Endpoint]) -> None:
        if not endpoints:
            raise ValueError("VeRegistry requires at least one endpoint")
        self._by_id: dict[str, Endpoint] = {}
        for ep in endpoints:
            if ep.id in self._by_id:
                raise ValueError(f"duplicate VE id: {ep.id}")
            self._by_id[ep.id] = ep
        # Preserve insertion order for a stable, deterministic candidate list.
        self._ids: list[str] = list(self._by_id)

    def ids(self) -> list[str]:
        """Return the VE ids, in registration order (the HRW candidate set)."""
        return list(self._ids)

    def get(self, ve_id: str) -> Endpoint:
        """Return the endpoint for ve_id.

        Raises:
            KeyError: If ve_id is not registered.
        """
        return self._by_id[ve_id]


class PdRegistry:
    """Static set of Prefill/Decode servers, dispensed round-robin.

    PD needs no content affinity (a PD pulls each embedding by mm_hash regardless
    of which PD lands the request), so a whole request goes to one PD chosen
    round-robin. The Router does not drive PD yet; a load-aware policy can replace
    next() later.

    Example:
        >>> reg = PdRegistry([Endpoint("pd-0", "h", 8200), Endpoint("pd-1", "h", 8201)])
        >>> [reg.next().id for _ in range(3)]
        ['pd-0', 'pd-1', 'pd-0']
    """

    def __init__(self, endpoints: Sequence[Endpoint]) -> None:
        if not endpoints:
            raise ValueError("PdRegistry requires at least one endpoint")
        self._endpoints: list[Endpoint] = list(endpoints)
        self._cursor = itertools.cycle(self._endpoints)

    def next(self) -> Endpoint:
        """Return the next PD endpoint in round-robin order."""
        return next(self._cursor)

    def ids(self) -> list[str]:
        """Return the PD ids, in registration order."""
        return [ep.id for ep in self._endpoints]
