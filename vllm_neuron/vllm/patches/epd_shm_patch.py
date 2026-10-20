# SPDX-License-Identifier: Apache-2.0
"""VE-side shared-memory receive for the EPD Router->VE pixel payload.

Writer side is vllm_neuron/vllm/disaggregated_encoder/shm_transport.py. The VE is a stock
``vllm serve --mm-encoder-only`` process, so the decoder of the wire string is upstream code with
no plugin frame in its path. We wrap it: an ``EPDSHM1:`` descriptor attaches the named
reader store (once, cached) and returns ``store.get(...)``; any other string falls through
to the original base64 decode. The wrapper is inert for non-shm payloads, so it is always
safe to install.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_applied = False
_original_decode = None
_reader_stores: dict = {}  # segment name -> attached reader store (attach once per name)
_stores_lock = threading.Lock()

from vllm_neuron.vllm.disaggregated_encoder.mm_serde_compat import (
    MM_SERDE_MODULE_NAMES,
    SERVING_MODULE_NAMES,
)


def _get_reader_store(name: str, buffer_bytes: int, max_object_size: int):
    """Attach (once) and cache the reader-side store for a shm segment name."""
    store = _reader_stores.get(name)
    if store is None:
        with _stores_lock:
            store = _reader_stores.get(name)
            if store is None:
                from vllm_neuron.vllm.disaggregated_encoder import shm_transport

                store = shm_transport.make_reader_store(
                    name, buffer_bytes, max_object_size
                )
                _reader_stores[name] = store
                logger.info("EPD shm: VE attached reader store %r", name)
    return store


def _make_shm_decode(original):
    """Wrap decode_mm_kwargs_item: EPDSHM1 -> shm read, else defer to base64 ``original``."""
    from vllm_neuron.vllm.disaggregated_encoder import shm_transport

    def _shm_decode(data):
        if shm_transport.is_shm_descriptor(data):
            desc = shm_transport.parse_descriptor(data)
            store = _get_reader_store(
                desc["name"], desc["buffer_bytes"], desc["max_object_size"]
            )
            return store.get(desc["address"], desc["monotonic_id"])
        return original(data)

    return _shm_decode


def _wrap_now() -> bool:
    """Wrap decode_mm_kwargs_item on the loaded disagg modules. Returns False (defer) if
    the mm_serde module is not importable yet."""
    global _applied, _original_decode
    if _applied:
        return True
    import sys

    mm_serde = next(
        (m for n in MM_SERDE_MODULE_NAMES if (m := sys.modules.get(n)) is not None),
        None,
    )
    if mm_serde is None:
        try:
            from vllm_neuron.vllm.disaggregated_encoder.mm_serde_compat import (  # noqa: PLC0415
                import_mm_serde,
            )

            mm_serde = import_mm_serde()
        except ImportError:
            return False

    _applied = True
    _original_decode = mm_serde.decode_mm_kwargs_item
    shm_decode = _make_shm_decode(_original_decode)

    # Rebind at the origin module AND in serving, which imported the name by value
    # (``from ...mm_serde import decode_mm_kwargs_item`` -- patching only the origin would
    # miss the VE's actual call site).
    mm_serde.decode_mm_kwargs_item = shm_decode
    for serving_name in SERVING_MODULE_NAMES:
        serving = sys.modules.get(serving_name)
        if (
            serving is not None
            and getattr(serving, "decode_mm_kwargs_item", None) is _original_decode
        ):
            serving.decode_mm_kwargs_item = shm_decode

    logger.info("EPD shm patch applied (VE decode_mm_kwargs_item -> shm reader)")
    return True


def apply_epd_shm_patch() -> None:
    """Install the VE-side shm receive (idempotent; no-op without the disagg module).

    Must be called from a per-process hook that runs in the VE request process AFTER the
    platform is materialized (see ``NeuronPlatform.check_and_update_config``). Calling it
    only from ``register()`` misses on vLLM 0.24: platform resolution is memoized and can
    run before the disagg module is importable, so the wrap would be lost.
    """
    if not _wrap_now():
        logger.debug("EPD shm patch: disagg mm_serde not importable; skipped")
