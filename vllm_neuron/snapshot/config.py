# SPDX-License-Identifier: Apache-2.0
"""Snapshot capture configuration and selection, resolved from the environment.

Selection is proactive: token/request rules are evaluated live against a
forward's positions and request ids, so only the targeted run is dumped rather
than capturing broadly and searching after. A malformed selection fails fast; a
malformed rank list degrades to the default instead of aborting startup.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vllm_neuron import envs
from vllm_neuron.snapshot.context import SnapshotMatch

logger = logging.getLogger(__name__)

# On-disk artifact formats, mirroring what the write_tensors op accepts. "pt" is
# a pickled tensor (any dtype); "npy" is raw value-preserving bytes. Keep in sync
# with the op's own format check on the C++ side. The default itself lives in
# envs.py; an override outside this set is rejected at startup.
_SUPPORTED_FORMATS = ("pt", "npy")


class CaptureSelector:
    """Decide which forwards of a NEFF to capture.

    Three rules OR'd together: call index (Nth post-warmup call), generated
    token (decode), and request id. With no rule configured, defaults to the
    first call. Token/request rules need per-forward positions/ids and are
    evaluated by :meth:`evaluate_forward`; the call-index rule is local to each
    executable (:meth:`call_index_match`).
    """

    def __init__(self) -> None:
        self._call_indices: set[int] = set()
        self._capture_all_calls: bool = False
        self._call_rule_configured: bool = False
        self._tokens: set[int] = set()
        self._requests: set[str] = set()

    @classmethod
    def from_env(cls) -> "CaptureSelector":
        """Build the selector from the selection env vars.

        Parses eagerly so a malformed selection fails at construction, not on
        the hot path.
        """
        selector = cls()
        call_all, call_indices = _parse_index_set(
            envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL,
            "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL",
        )
        if call_all or call_indices is not None:
            selector._call_rule_configured = True
            selector._capture_all_calls = call_all
            selector._call_indices = call_indices or set()

        _, tokens = _parse_index_set(
            envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN,
            "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN",
            allow_all=False,
        )
        selector._tokens = tokens or set()
        selector._requests = _parse_str_set(
            envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST
        )
        return selector

    def has_any_rule(self) -> bool:
        """Whether any selection rule is configured at all."""
        return bool(self._call_rule_configured or self._tokens or self._requests)

    def has_value_rule(self) -> bool:
        """Whether a rule needs per-forward positions/request ids.

        Gates the per-forward host sync of positions: call-index-only capture
        pays nothing.
        """
        return bool(self._tokens or self._requests)

    def call_index_match(self, call_index: int) -> bool:
        """Whether this NEFF's 0-based post-warmup call index is selected.

        With no rule anywhere, defaults to the first call; with non-call rules
        only, never fires (those forwards are selected by token/request).
        """
        if not self.has_any_rule():
            return call_index == 0
        if not self._call_rule_configured:
            return False
        return self._capture_all_calls or call_index in self._call_indices

    def evaluate_forward(
        self,
        req_ids: List[str],
        positions: List[int],
        is_decode: bool,
        max_generated: int,
    ) -> Tuple[bool, List[SnapshotMatch]]:
        """Resolve the token/request verdict for one forward, returning
        ``(capture, matches)``.

        A request rule fires for any targeted row. A token rule fires on decode
        when a row at position ``p`` will generate a targeted token ``p +
        offset`` (offset ``1..max_generated``, >1 under speculative decode);
        skipped on prefill, where that relationship does not hold.
        """
        matches: List[SnapshotMatch] = []

        if self._requests:
            for batch_index, req_id in enumerate(req_ids):
                if self._request_match(req_id):
                    pos = (
                        positions[batch_index]
                        if is_decode and batch_index < len(positions)
                        else -1
                    )
                    matches.append(SnapshotMatch(batch_index, req_id, pos, "request"))

        if self._tokens and is_decode:
            # Iterate real requests, not raw positions: a decode batch may pad
            # positions beyond the request count, and a padding row (often at
            # position 0) must not be matched as a real token capture.
            for batch_index, req_id in enumerate(req_ids):
                if batch_index >= len(positions):
                    break
                pos = positions[batch_index]
                for offset in range(1, max_generated + 1):
                    generated = pos + offset
                    if generated in self._tokens:
                        matches.append(
                            SnapshotMatch(batch_index, req_id, generated, "token")
                        )
                        break

        return (bool(matches), matches)

    def _request_match(self, req_id: str) -> bool:
        """Whether a forward's request id matches a targeted request.

        vLLM appends a unique ``-<suffix>`` to the caller's request id (e.g.
        caller ``abc`` becomes ``abc-9a8546d5``), so the full id is not knowable
        in advance. Match the caller-supplied base id as a prefix, and still
        accept an exact full id.
        """
        return any(
            req_id == target or req_id.startswith(target + "-")
            for target in self._requests
        )


@dataclass(frozen=True)
class SnapshotConfig:
    """Resolved snapshot settings for the current worker process."""

    enabled: bool
    output_dir: str
    selector: CaptureSelector
    max_captures: int
    fmt: str
    # None means all ranks capture; an explicit list narrows to those tp-ranks.
    ranks: Optional[List[int]] = None

    @classmethod
    def from_env(cls) -> "SnapshotConfig":
        """Build a config from the snapshot env vars."""
        return cls(
            enabled=envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE,
            output_dir=envs.get_neuron_snapshot_dir(),
            selector=CaptureSelector.from_env(),
            ranks=_parse_ranks(envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS),
            max_captures=_parse_max_captures(
                envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES
            ),
            fmt=_parse_format(envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT),
        )

    def is_active(self) -> bool:
        """Whether capture should run at all (rank scoping applied later)."""
        return self.enabled


# Resolved once per process. The config is derived purely from environment
# variables that are fixed for the process lifetime, so the startup validation,
# the model runner's selector, and every executable's capture spec share one
# resolution rather than each re-parsing the environment.
_RESOLVED_CONFIG: Optional[SnapshotConfig] = None


def get_snapshot_config() -> SnapshotConfig:
    """Return the process-wide snapshot config, resolving it once on first use.

    Parsing is fail-fast and deterministic for a fixed environment, so the
    result is cached and reused everywhere it is needed.
    """
    global _RESOLVED_CONFIG
    if _RESOLVED_CONFIG is None:
        _RESOLVED_CONFIG = SnapshotConfig.from_env()
    return _RESOLVED_CONFIG


def reset_snapshot_config() -> None:
    """Clear the cached config so the next call re-resolves from the environment.

    Intended for tests that vary the snapshot env vars between cases.
    """
    global _RESOLVED_CONFIG
    _RESOLVED_CONFIG = None


def _parse_index_set(
    raw: Optional[str], env_name: str, allow_all: bool = True
) -> Tuple[bool, Optional[set]]:
    """Parse a comma-separated 0-based index set with an optional ``-1`` "all".

    Returns ``(capture_all, indices)`` where an unset or empty value yields
    ``(False, None)`` to signal "no rule configured". A misconfigured value
    raises ``ValueError`` so a typo surfaces at startup instead of silently
    disabling a debugging run. ``allow_all=False`` rejects the ``-1`` sentinel
    (token selection has no "all tokens" meaning).
    """
    if raw is None:
        return (False, None)
    trimmed = raw.strip()
    if not trimmed:
        return (False, None)

    indices: set = set()
    capture_all = False
    for token in trimmed.split(","):
        stripped = token.strip()
        try:
            value = int(stripped)
        except ValueError:
            raise ValueError(
                f"{env_name}={raw!r} contains non-integer token {token!r}"
            ) from None
        if value == -1 and allow_all:
            capture_all = True
        elif value < 0:
            raise ValueError(
                f"{env_name}={raw!r} contains illegal negative index {value}"
            )
        else:
            indices.add(value)

    if capture_all and indices:
        raise ValueError(
            f"{env_name}={raw!r} combines the -1 (all) sentinel with explicit indices"
        )
    return (capture_all, indices)


def _parse_format(raw: Optional[str]) -> str:
    """Parse the artifact format.

    Raises ``ValueError`` on an unsupported or empty value so a typo surfaces at
    startup rather than mid-run on the first capture (where the op would reject
    it). The default is supplied by envs.py, so a real run never sees an empty
    value here.
    """
    fmt = (raw or "").strip().lower()
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT={raw!r} is unsupported; "
            f"expected one of {_SUPPORTED_FORMATS}"
        )
    return fmt


def _parse_str_set(raw: Optional[str]) -> set:
    """Parse a comma-separated set of exact-match strings (e.g. request ids)."""
    if raw is None:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _parse_max_captures(raw: Optional[str]) -> int:
    """Parse the capture budget.

    Raises ``ValueError`` on an empty, non-integer, or non-positive value so a
    typo surfaces at startup rather than silently applying a different cap. The
    default is supplied by envs.py, so a real run never sees an empty value here.
    """
    trimmed = (raw or "").strip()
    try:
        value = int(trimmed)
    except ValueError:
        raise ValueError(
            f"VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES={raw!r} is not an integer"
        ) from None
    if value <= 0:
        raise ValueError(
            f"VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES={raw!r} must be positive"
        )
    return value


def _parse_ranks(raw: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated non-negative rank set.

    An unset or empty value (``None``) captures all ranks. A malformed token
    degrades to that same default with a warning rather than failing startup,
    so a bad rank list never blocks an otherwise-valid capture run.
    """
    if raw is None:
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None

    ranks: List[int] = []
    for token in trimmed.split(","):
        stripped = token.strip()
        try:
            rank = int(stripped)
        except ValueError:
            logger.warning(
                "Ignoring malformed VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS=%r (bad token %r); "
                "capturing all ranks (the default)",
                raw,
                token,
            )
            return None
        if rank < 0:
            logger.warning(
                "Ignoring malformed VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS=%r (negative rank %d); "
                "capturing all ranks (the default)",
                raw,
                rank,
            )
            return None
        ranks.append(rank)
    return ranks
