# SPDX-License-Identifier: Apache-2.0
"""Send a request to the data-parallel engine that holds its prefix.

Each data-parallel (DP) engine keeps its own prefix cache. vLLM sends each
request to the least loaded engine, so with N engines the next turn of an agent
session lands on the engine that served the turn before only about 1 time in N.
On any other engine the request prefills its whole history again. The rollouts
of one prompt spread the same way and prefill the shared prompt once per engine.

The index in this module remembers which engine received which tokens: the
prompt of each request it routes, and the tokens that the engine generated for
it. A new request goes to the engine that holds the longest prefix of its
prompt, unless that engine carries more than ``slack`` requests above the least
loaded engine, or has no free sequence slot while the least loaded engine has
one. Then the request goes to the least loaded engine, as before: waiting for a
slot costs a decode, more than prefilling the prefix again.

The index needs no session ID: the next turn of a session starts with the
tokens of the turn before, so its prefix finds the engine. It works on 128-token
chunks. A chunk hash covers the chunk and every chunk before it, the same chain
that vLLM uses for its block hashes, so one hash identifies a whole prefix.

Each engine keeps at most about its KV cache capacity in chunks, and the index
evicts the least recently routed chunks first. It records the chunks of a prompt
from the last to the first, so the first chunks of a prefix are evicted last,
as in the vLLM block pool.

This module imports nothing from vLLM, so the tests can use it directly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

#: The chunk size of the index in tokens. It is the alignment of the Qwen3.5
#: DeltaNet snapshots, so a matched chunk is a prefix that the engine can reuse.
CHUNK_TOKENS = 128

#: The capacity of an engine in chunks when the KV cache size is not known.
DEFAULT_MAX_CHUNKS_PER_ENGINE = 1 << 16


@dataclass
class PrefixAffinityStats:
    """Counts of the routing decisions since the start of the server."""

    requests: int = 0
    matched: int = 0
    followed: int = 0
    overridden: int = 0
    matched_tokens: int = 0


@dataclass
class _Tail:
    """The state of the hash chain after the last full chunk of a request."""

    engine: int
    last_hash: int
    pending: list[int] = field(default_factory=list)


@dataclass
class PromptPrefix:
    """The chunk hashes of a prompt, and the rest of the chain after them."""

    hashes: list[int]
    last_hash: int
    pending: list[int]


class PrefixAffinityIndex:
    """Remember which engine holds which prefix, and pick an engine for a prompt.

    Args:
        num_engines: The number of DP engines.
        slack: The largest load difference, in requests, that a request accepts
            to go to the engine that holds its prefix instead of the least loaded
            engine.
        capacity: The sequence slots of an engine (max_num_seqs), or None. A
            request does not go to a full engine while the least loaded engine
            has a free slot.
        max_chunks_per_engine: The capacity of each engine in chunks.
        chunk_tokens: The chunk size in tokens.

    Example:
        >>> index = PrefixAffinityIndex(num_engines=2, slack=4)
        >>> prefix = index.prefix(list(range(300)))
        >>> index.admit("a", 1, prefix)
        >>> index.choose(prefix, scores=[(0.0,), (1.0,)], least=0, order=[0, 1])
        1
    """

    def __init__(
        self,
        num_engines: int,
        slack: float,
        capacity: float | None = None,
        max_chunks_per_engine: int = DEFAULT_MAX_CHUNKS_PER_ENGINE,
        chunk_tokens: int = CHUNK_TOKENS,
    ) -> None:
        if num_engines < 1:
            raise ValueError(f"num_engines must be 1 or more, got {num_engines}")
        if chunk_tokens < 1 or max_chunks_per_engine < 1:
            raise ValueError(
                "chunk_tokens and max_chunks_per_engine must be 1 or more, got "
                f"{chunk_tokens} and {max_chunks_per_engine}"
            )
        self.slack = slack
        self.capacity = capacity
        self.chunk_tokens = chunk_tokens
        self.max_chunks_per_engine = max_chunks_per_engine
        self.stats = PrefixAffinityStats()
        self._engines: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]
        self._tails: dict[str, _Tail] = {}

    @property
    def num_engines(self) -> int:
        return len(self._engines)

    def resize(self, num_engines: int) -> None:
        """Forget everything when the engine count changes (elastic scaling)."""
        if num_engines != self.num_engines:
            self._engines = [OrderedDict() for _ in range(num_engines)]
            self._tails.clear()

    def prefix(
        self,
        token_ids: Sequence[int],
        lora_name: str | None = None,
        cache_salt: str | None = None,
    ) -> PromptPrefix:
        """Hash the full chunks of a prompt.

        A LoRA adapter and a cache salt start a different chain, as they do
        for the vLLM block hashes.
        """
        last: int = hash(("vllm-neuron-prefix-affinity", lora_name, cache_salt))
        size = self.chunk_tokens
        num_full = len(token_ids) // size * size
        hashes = []
        for start in range(0, num_full, size):
            last = hash((last, tuple(token_ids[start : start + size])))
            hashes.append(last)
        return PromptPrefix(hashes, last, list(token_ids[num_full:]))

    def match(self, hashes: Sequence[Hashable]) -> tuple[list[int], int]:
        """Find the engines that hold the longest prefix of a prompt.

        Returns:
            The engines with the longest match, and its length in chunks. With
            no match, every engine and 0.
        """
        candidates = list(range(self.num_engines))
        matched = 0
        for chunk_hash in hashes:
            holders = [e for e in candidates if chunk_hash in self._engines[e]]
            if not holders:
                break
            candidates = holders
            matched += 1
        return candidates, matched

    def choose(
        self,
        prefix: PromptPrefix,
        scores: Sequence[Sequence[float]],
        least: int,
        order: Sequence[int],
    ) -> int:
        """Pick the engine for a load-balanced request.

        Args:
            prefix: The prompt prefix of the request.
            scores: The load score of each engine; the first item is the load in
                requests. A lower score is a lighter load.
            least: The engine that load balancing picked.
            order: The engines in the order that breaks ties.

        Returns:
            The engine that holds the longest prefix, if its load is within
            ``slack`` of the load of ``least`` and it has a free slot (or
            ``least`` has none either), and otherwise ``least``.
        """
        self.stats.requests += 1
        candidates, matched = self.match(prefix.hashes)
        if matched == 0 or len(candidates) == self.num_engines:
            return least
        self.stats.matched += 1
        holders = set(candidates)
        best = min((e for e in order if e in holders), key=lambda e: scores[e])
        load, least_load = scores[best][0], scores[least][0]
        full = self.capacity is not None and load >= self.capacity
        least_full = self.capacity is not None and least_load >= self.capacity
        if load - least_load <= self.slack and (not full or least_full):
            self.stats.followed += 1
            self.stats.matched_tokens += matched * self.chunk_tokens
            return best
        self.stats.overridden += 1
        return least

    def admit(self, request_id: str, engine: int, prefix: PromptPrefix) -> None:
        """Record that an engine received a prompt, and follow its outputs."""
        self._record(engine, prefix.hashes)
        self._tails[request_id] = _Tail(engine, prefix.last_hash, prefix.pending)

    def extend(self, request_id: str, new_token_ids: Sequence[int]) -> None:
        """Record the tokens that an engine generated for a request."""
        tail = self._tails.get(request_id)
        if tail is None:
            return
        tail.pending.extend(new_token_ids)
        size = self.chunk_tokens
        if len(tail.pending) < size:
            return
        hashes = []
        while len(tail.pending) >= size:
            tail.last_hash = hash((tail.last_hash, tuple(tail.pending[:size])))
            hashes.append(tail.last_hash)
            del tail.pending[:size]
        self._record(tail.engine, hashes)

    def finish(self, request_id: str) -> None:
        """Stop following a request."""
        self._tails.pop(request_id, None)

    def _record(self, engine: int, hashes: Sequence[int]) -> None:
        chunks = self._engines[engine]
        # Last chunk first: the first chunks of the prefix become the most
        # recent ones, so the eviction removes the end of a prefix first.
        for chunk_hash in reversed(hashes):
            chunks[chunk_hash] = None
            chunks.move_to_end(chunk_hash)
        while len(chunks) > self.max_chunks_per_engine:
            chunks.popitem(last=False)
