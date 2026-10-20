# SPDX-License-Identifier: Apache-2.0
"""Per-request phase records for the EPD Router.

The Router appends one JSON object per client request to a JSONL file, so an
investigation can name the step that controls TTFT instead of seeing only the
total. Pass the destination with the Router's ``--metrics-file``; without one the
recorder measures but writes nothing.

Serial terms; they sum to router_total_ms minus residual_ms::

    preproc_wait_ms    Pool submit until a worker starts. Holds the arg IPC too.
    preproc_render_ms  The worker's HF render.
    preproc_stage_ms   The worker's shm staging + per-item kwargs encode.
    route_ms           Dedup by mm_hash, HRW pick, grouping per VE.
    ve_gather_ms       Wall-clock of the parallel VE POSTs: a MAX, not a sum.
    build_pd_ms        PD body build + PD endpoint pick.
    pd_first_token_ms  PD dispatch until the first content frame.
    router_total_ms    Router entry until the first content frame is yielded.
    residual_ms        router_total_ms minus the serial terms. Small, positive.

Subsets; inside a serial term, so never add them to it::

    hrw_pick_ms        The hrw_pick calls only, summed over the distinct items.
                       A subset of route_ms, which also holds the dedup and the
                       grouping per VE.

Both TTFT terms stop at the first frame carrying content, matching where the
bench measures TTFT; a role-only frame does not stop them. A request that emits
no content reports both as None. The non-streaming path has no first-token event
(one POST spans the generation), so it reports pd_first_token_ms and residual_ms
as None. A text-only request reports route_ms and ve_gather_ms as a measured 0.0.
An unmeasured phase is None, never 0.0.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

SERIAL_PHASE_KEYS: tuple[str, ...] = (
    "preproc_wait_ms",
    "preproc_render_ms",
    "preproc_stage_ms",
    "route_ms",
    "ve_gather_ms",
    "build_pd_ms",
    "pd_first_token_ms",
)

# Inside a serial term, so excluded from the sum that residual_ms checks.
SUBSET_KEYS: tuple[str, ...] = ("hrw_pick_ms",)

# The base of the subtraction; an ordinary phase key.
TOTAL_KEY = "router_total_ms"

RECORD_KEYS: tuple[str, ...] = (
    ("request_id", "stream")
    + SERIAL_PHASE_KEYS
    + (TOTAL_KEY, "residual_ms")
    + SUBSET_KEYS
)


class RouterPhaseRecorder:
    """Collect one request's phase spans, then emit one record.

    Starts the total at construction; stop(TOTAL_KEY) at the first content
    frame; emit() once from a finally block so a failed request still records.

    Example:
        >>> rec = RouterPhaseRecorder("req-1", stream=True)
        >>> with rec.span("route_ms"):
        ...     pass
        >>> rec.stop(TOTAL_KEY)
        >>> rec.emit()
    """

    __slots__ = ("_request_id", "_stream", "_values", "_starts", "_writer")

    def __init__(
        self, request_id: str, *, stream: bool, writer: "RecordWriter | None" = None
    ) -> None:
        self._request_id = request_id
        self._stream = stream
        self._writer = writer
        self._values: dict[str, Any] = {}
        self._starts: dict[str, float] = {}
        self.start(TOTAL_KEY)

    @contextlib.contextmanager
    def span(self, key: str) -> "Iterator[None]":
        """Measure the enclosed block and store the result in milliseconds."""
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def start(self, key: str) -> None:
        """Start a phase that a later event ends."""
        self._starts[key] = time.perf_counter()

    def stop(self, key: str) -> None:
        """End a phase :meth:`start` began. First call wins; no-op otherwise."""
        start = self._starts.get(key)
        if start is None or key in self._values:
            return
        self._values[key] = (time.perf_counter() - start) * 1000.0

    def set(self, key: str, value: Any) -> None:
        """Store a value the recorder does not measure. None if unmeasured."""
        self._values[key] = value

    def record(self) -> dict[str, Any]:
        """Build the record: every key of :data:`RECORD_KEYS`.

        residual_ms is None unless the total and every serial term are
        measured, because a remainder over partial data has no meaning.
        """
        out: dict[str, Any] = {
            "request_id": self._request_id,
            "stream": self._stream,
        }
        serial_total = 0.0
        serial_measured = True
        for key in SERIAL_PHASE_KEYS:
            value = self._values.get(key)
            out[key] = value
            if value is None:
                serial_measured = False
            else:
                serial_total += value
        total = self._values.get(TOTAL_KEY)
        out[TOTAL_KEY] = total
        out["residual_ms"] = (
            total - serial_total if serial_measured and total is not None else None
        )
        # Subsets sit inside a serial term, so they stay out of that sum.
        for key in SUBSET_KEYS:
            out[key] = self._values.get(key)
        return out

    def emit(self) -> None:
        """Append the record to the writer, if there is one. Never raises."""
        if self._writer is not None:
            self._writer.append(self.record())


class RecordWriter:
    """Append records to a JSONL file, one object per line.

    Opens the file once and flushes each record, so a killed Router keeps the
    records it already produced.

    Example:
        >>> writer = RecordWriter("/tmp/epd/router_metrics.jsonl")
        >>> writer.append({"request_id": "req-1"})
    """

    __slots__ = ("_path", "_handle")

    def __init__(self, path: str) -> None:
        self._path = path
        self._handle: Any = None

    def append(self, record: dict[str, Any]) -> None:
        """Write one record. Never raises: reporting must not fail a request."""
        try:
            if self._handle is None:
                os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
                self._handle = open(self._path, "a", buffering=1)
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 - never fail a request over reporting
            logger.warning(
                "EPD Router metrics: cannot append to %s. The split is not "
                "available for this run.",
                self._path,
                exc_info=True,
            )
            self.close()

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                logger.warning("EPD Router metrics: close failed", exc_info=True)
