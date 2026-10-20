# SPDX-License-Identifier: Apache-2.0
"""Data transfer objects for the EPD Router.

Inherits field shapes of upstream vLLM's disaggregated DTOs
(vllm.entrypoints.scale_out.token_in_token_out.protocol): per-modality items
are positionally aligned and tensor payloads cross the wire as base64 strings
via mm_serde.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.multimodal.inputs import MultiModalKwargsItem


@dataclass(frozen=True)
class PreprocessedItem:
    """One media item after HF preprocessing.

    Args:
        mm_hash: Content hash of the item. The routing key and the embedding cache key
            downstream.
        mm_kwargs_item: The preprocessed tensors the VE encodes (pixel_values +
            {image,video}_grid_thw).
        offset: Placeholder start offset in the token sequence (from PlaceholderRange).
        length: Placeholder token count (== merged-token count for this item).
        modality: The renderer's modality key for this item ("image" / "video").
        pixel_descriptor: Wire payload for this item's pixel_values from the preproc
            pool -- normally an EPDSHM1 shm handle. If the worker's shm segment is
            full, the item falls back to a base64-encoded blob passed inline over
            HTTP instead.
    """

    mm_hash: str
    mm_kwargs_item: MultiModalKwargsItem
    offset: int
    length: int
    modality: str
    pixel_descriptor: str | None = None


@dataclass(frozen=True)
class PreprocessedRequest:
    """A complete request after HF preprocessing: token ids plus per-item media records.

    Args:
        prompt_token_ids: The prompt token ids with vision placeholders expanded.
        items: Per-media-item records, each independently routable to a VE.
        preproc_timing: The worker's own timing of this render, in milliseconds:
            {"render_ms", "stage_ms"}. The preprocessing pool cannot report the
            queue wait from the caller side, so the worker reports the work
            instead. The Router subtracts the work from the span it observes, and
            the remainder is the wait. The field is None when no worker reported
            a timing, for example on the in-process render path.
    """

    prompt_token_ids: list[int]
    items: list[PreprocessedItem]
    preproc_timing: dict[str, float] | None = None


@dataclass(frozen=True)
class VeReady:
    """A VE's acknowledgement that an embedding for mm_hash is ready to pull.

    TODO: update locator type to match schema decided with on-device encoder cache

    Args:
        mm_hash: The item the VE encoded (content key; also the embedding cache key).
        locator: Where/how PD pulls the embedding
    """

    mm_hash: str
    locator: dict[str, Any]


@dataclass(frozen=True)
class EncodeResult:
    """Outcome of encoding a request: the preprocessed request + per-item VE acks.

    Args:
        pre: The preprocessed request (token ids + per-item records). The prefill/decode
            drive consumes this (token ids + per-item placeholder positions).
        ready: Map of mm_hash → VeReady — the VE ack per distinct mm_hash, carrying the
            locator PD pulls by.

    Note:
        ready is keyed by content hash, so duplicate media within one request (two items
        sharing an mm_hash) yields one entry — by design, since both pull the same
        embedding. Do NOT assume len(ready) == len(pre.items); look up per item via
        ready[item.mm_hash].
    """

    pre: PreprocessedRequest
    ready: dict[str, VeReady]
