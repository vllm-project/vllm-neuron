# SPDX-License-Identifier: Apache-2.0
"""Same-host shared-memory transport for the EPD preproc-pool->VE pixel payload.

A preproc-pool worker writes each ``MultiModalKwargsItem`` into a shared-memory object
store and returns a JSON descriptor (placed in the HTTP body); the VE reads the bytes
back from shm. Built on vLLM's ``SingleWriterShmObjectStorage`` -- the same primitive
vLLM uses for its API-server->worker mm-tensor cache.

One store per pool worker (sole writer of its segment); a VE attaches whichever segment a
descriptor names. Pool workers and VE must be co-located. ``put_item`` degrades to a
base64 blob for any item shm can't take, which the VE decodes normally.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalKwargsItem


class _CpuMsgpackSerde(MsgpackSerde):
    """MsgpackSerde whose decoders produce plain CPU-owned tensors (no pin: the VE API
    server has no device to pin to, and we want tensors that outlive the shm slot)."""

    def __init__(self):
        import torch
        from vllm.multimodal.inputs import MultiModalKwargsItem
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.encoder = MsgpackEncoder()
        self.tensor_decoder = MsgpackDecoder(torch.Tensor, share_mem=False)
        self.mm_decoder = MsgpackDecoder(MultiModalKwargsItem, share_mem=False)
        self.tensor_decoder.pin_tensors = False
        self.mm_decoder.pin_tensors = False
        self._mm_kwargs_item_cls = MultiModalKwargsItem


# A kwargs_data["image"] string with this prefix is a shm handle, not a base64 blob.
_TAG = "EPDSHM1:"
# Cap a single object below the ring size to leave room for metadata/msgpack framing.
_MAX_OBJECT_FRACTION = 0.5


def make_writer_store(name: str, buffer_bytes: int) -> SingleWriterShmObjectStorage:
    """Create a shm object store as the WRITER (pool-worker side); ``name`` must be unique
    per worker, ``buffer_bytes`` must hold the worker's max in-flight pixel payload."""
    ring = SingleWriterShmRingBuffer(
        data_buffer_size=buffer_bytes, name=name, create=True
    )
    return SingleWriterShmObjectStorage(
        max_object_size=int(buffer_bytes * _MAX_OBJECT_FRACTION),
        n_readers=1,
        ring_buffer=ring,
        serde_class=_CpuMsgpackSerde,
        reader_lock=None,
    )


def make_reader_store(
    name: str, buffer_bytes: int, max_object_size: int
) -> SingleWriterShmObjectStorage:
    """Attach an existing shm object store as a READER (VE side) by ``name`` alone; the
    three sizing args arrive in the descriptor the writer emitted."""
    ring = SingleWriterShmRingBuffer(
        data_buffer_size=buffer_bytes, name=name, create=False
    )
    return SingleWriterShmObjectStorage(
        max_object_size=max_object_size,
        n_readers=1,
        ring_buffer=ring,
        serde_class=_CpuMsgpackSerde,
        reader_lock=threading.Lock(),
    )


def put_item(
    store: SingleWriterShmObjectStorage,
    mm_hash: str,
    item: "MultiModalKwargsItem",
) -> str:
    """Write one media item to the store; return its wire descriptor (tagged JSON).

    Dedups on ``mm_hash`` (identical media shares one slot). Falls back to a base64 blob
    when the store can't take the item -- ring full (``MemoryError`` after the store's
    free+retry) or item > ``max_object_size`` (``ValueError``); the VE decode path handles
    the untagged string via base64.
    """
    try:
        if store.is_cached(mm_hash):
            address, monotonic_id = store.get_cached(mm_hash)
        else:
            address, monotonic_id = store.put(mm_hash, item)
    except (MemoryError, ValueError) as exc:
        from vllm_neuron.vllm.disaggregated_encoder.mm_serde_compat import (
            import_mm_serde,
        )

        mm_serde = import_mm_serde()

        logger.warning(
            "EPD shm degraded to base64 for mm_hash=%s (%s: %s) -- raise "
            "VLLM_NEURON_EPD_SHM_BUFFER_BYTES to maintain shared memory performance",
            mm_hash,
            type(exc).__name__,
            exc,
        )
        return mm_serde.encode_mm_kwargs_item(item)
    desc = {
        "name": store.ring_buffer.shared_memory.name,
        "buffer_bytes": store.ring_buffer.data_buffer_size,
        "max_object_size": store.max_object_size,
        "address": address,
        "monotonic_id": monotonic_id,
    }
    return _TAG + json.dumps(desc, separators=(",", ":"))


def is_shm_descriptor(s: str) -> bool:
    """True if a kwargs_data string is a shm descriptor (vs. a base64 blob)."""
    return isinstance(s, str) and s.startswith(_TAG)


def parse_descriptor(s: str) -> dict[str, Any]:
    """Parse a tagged descriptor string into its fields."""
    return json.loads(s[len(_TAG) :])
