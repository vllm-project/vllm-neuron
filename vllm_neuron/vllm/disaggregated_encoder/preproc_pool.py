# SPDX-License-Identifier: Apache-2.0
"""Separate-process HF preprocessing for the EPD Router.

Runs render_request in a spawn process pool: each worker builds its own renderer and a
single-writer shm store once, then per request renders and writes to shm,  returning token ids, metadata amd the shm descriptor.
VLLM_NEURON_EPD_PREPROC_WORKERS (default 8).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_RENDERER: Any = None
_STORE: Any = None

# Per-worker render timeout: a stuck/dead worker surfaces as a request error, not a hang.
_RENDER_TIMEOUT_S = 120.0


def _warmup() -> float:
    """Render one synthetic image to warm the worker's HF processor (moves the cold-start
    cost off the first real request). Returns the warm-render ms, or -1.0 on failure."""
    import base64
    import io
    import time

    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (127, 127, 127)).save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "warmup"},
                ],
            }
        ]
        t0 = time.perf_counter()
        _render_and_stage(messages, None)
        return (time.perf_counter() - t0) * 1000.0
    except Exception:
        logger.warning("EPD preproc worker warmup failed (non-fatal)", exc_info=True)
        return -1.0


def _worker_init(
    model: str,
    shm_buffer_bytes: int,
    quantization: str | None,
    media_io_kwargs: "Mapping[str, Any] | None" = None,
    mm_processor_kwargs: "Mapping[str, Any] | None" = None,
) -> None:
    """Build this worker's renderer + single-writer shm store once (spawn pool init)."""
    global _RENDERER, _STORE

    from vllm_neuron import envs

    # Cap this worker's CPU thread pools BEFORE importing torch/renderer. With
    # PREPROC_WORKERS spawn workers each rendering in parallel, an uncapped per-worker
    # BLAS/OMP/torch pool sizes to all vCPUs and multiplies to hundreds of threads,
    # oversubscribing the host against the co-located TP engine driver processes and
    # starving their scheduler/prefill-admit/NIXL-feed loops (measured ~3x throughput
    # loss / ~9x TTFT at 2VE14PD c32). Keep each worker to WORKER_THREADS threads.
    threads = max(1, envs.VLLM_NEURON_EPD_PREPROC_WORKER_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    from vllm_neuron.vllm.disaggregated_encoder import shm_transport
    from vllm_neuron.vllm.disaggregated_encoder.codec import build_renderer

    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        logger.warning("preproc worker could not cap torch threads", exc_info=True)

    _RENDERER = build_renderer(
        model,
        renderer_num_workers=threads,
        quantization=quantization,
        media_io_kwargs=media_io_kwargs,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    # PID-named segment (this worker is its sole writer)
    _STORE = shm_transport.make_writer_store(
        f"epd_preproc_{os.getpid()}", shm_buffer_bytes
    )


def _render_and_stage(
    messages: "Sequence[Mapping[str, Any]] | Mapping[str, Any]",
    chat_template_kwargs: "Mapping[str, Any] | None",
) -> dict[str, Any]:
    """Render one request and stage its pixels to this worker's shm store.

    Returns a picklable dict: {prompt_token_ids, items: [{mm_hash, offset, length,
    pixel_descriptor, lm_kwargs_b64}], timing: {render_ms, stage_ms}}

    The worker reports its own timing, because the caller cannot see when a pool
    worker starts the work. The Router subtracts this work from the span it
    observes to get the pool wait.
    """
    import time

    from vllm_neuron.vllm.disaggregated_encoder.mm_serde_compat import (
        import_mm_serde,
    )

    mm_serde = import_mm_serde()

    from vllm_neuron.utils.mm_utils import get_epd_kwargs
    from vllm_neuron.vllm.disaggregated_encoder import shm_transport
    from vllm_neuron.vllm.disaggregated_encoder.codec import render_request

    hf_config = _RENDERER.model_config.hf_config
    t_render = time.perf_counter()
    pre = asyncio.run(
        render_request(_RENDERER, messages, chat_template_kwargs=chat_template_kwargs)
    )
    t_stage = time.perf_counter()
    items_out: list[dict[str, Any]] = []
    for it in pre.items:
        desc = shm_transport.put_item(_STORE, it.mm_hash, it.mm_kwargs_item)
        lm_kwargs_b64 = mm_serde.encode_mm_kwargs_item(
            get_epd_kwargs(hf_config, it.mm_kwargs_item)
        )
        items_out.append(
            {
                "mm_hash": it.mm_hash,
                "offset": it.offset,
                "length": it.length,
                "modality": it.modality,
                "pixel_descriptor": desc,
                "lm_kwargs_b64": lm_kwargs_b64,
            }
        )
    t_end = time.perf_counter()
    return {
        "prompt_token_ids": list(pre.prompt_token_ids),
        "items": items_out,
        "timing": {
            "render_ms": (t_stage - t_render) * 1000.0,
            "stage_ms": (t_end - t_stage) * 1000.0,
        },
    }


class PreprocPool:
    """A spawn process pool that runs HF preprocessing off the router event loop."""

    def __init__(
        self,
        model: str,
        workers: int,
        shm_buffer_bytes: int,
        quantization: str | None = None,
        media_io_kwargs: "Mapping[str, Any] | None" = None,
        mm_processor_kwargs: "Mapping[str, Any] | None" = None,
    ):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                model,
                shm_buffer_bytes,
                quantization,
                dict(media_io_kwargs) if media_io_kwargs else None,
                dict(mm_processor_kwargs) if mm_processor_kwargs else None,
            ),
        )
        try:
            warm = [self._pool.apply_async(_warmup) for _ in range(workers)]
            ms = [w.get(timeout=_RENDER_TIMEOUT_S) for w in warm]
            ok = [m for m in ms if m >= 0]
            logger.info(
                "EPD PreprocPool started: %d spawn worker(s), %.1f GB shm each; "
                "warmup %d/%d ok, first-render median %.0fms",
                workers,
                shm_buffer_bytes / 2**30,
                len(ok),
                len(ms),
                (sorted(ok)[len(ok) // 2] if ok else -1.0),
            )
        except Exception as exc:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                logger.warning("PreprocPool terminate after failed init", exc_info=True)
            raise RuntimeError(
                f"EPD PreprocPool failed to initialize {workers} worker(s) for "
                f"model {model!r} (renderer build or shm allocation failed; each "
                f"worker needs {shm_buffer_bytes / 2**30:.1f} GB of /dev/shm). "
                f"See worker logs above."
            ) from exc

    async def render(
        self,
        messages: "Sequence[Mapping[str, Any]]",
        *,
        chat_template_kwargs: "Mapping[str, Any] | None" = None,
    ):
        """Render one request in a pool worker and rebuild a PreprocessedRequest locally.

        The heavy messages are pickled to the worker; only the shm descriptor + token ids
        come back (the pixels stay in the worker's shm).
        """
        from vllm_neuron.vllm.disaggregated_encoder.mm_serde_compat import (
            import_mm_serde,
        )

        mm_serde = import_mm_serde()

        from vllm_neuron.vllm.disaggregated_encoder.protocol import (
            PreprocessedItem,
            PreprocessedRequest,
        )

        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def _done(result):
            loop.call_soon_threadsafe(fut.set_result, result)

        def _err(exc):
            loop.call_soon_threadsafe(fut.set_exception, exc)

        task_args = (list(messages), dict(chat_template_kwargs or {}))
        self._pool.apply_async(
            _render_and_stage, task_args, callback=_done, error_callback=_err
        )
        try:
            raw = await asyncio.wait_for(fut, timeout=_RENDER_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"EPD preproc pool render exceeded {_RENDER_TIMEOUT_S:.0f}s -- a pool "
                f"worker likely died mid-render (OOM/segfault) so its result never "
                f"returned. The request fails rather than hanging."
            ) from exc

        items: list[PreprocessedItem] = []
        for d in raw["items"]:
            items.append(
                PreprocessedItem(
                    mm_hash=d["mm_hash"],
                    mm_kwargs_item=mm_serde.decode_mm_kwargs_item(d["lm_kwargs_b64"]),
                    offset=d["offset"],
                    length=d["length"],
                    modality=d["modality"],
                    pixel_descriptor=d["pixel_descriptor"],
                )
            )
        return PreprocessedRequest(
            prompt_token_ids=list(raw["prompt_token_ids"]),
            items=items,
            preproc_timing=raw.get("timing"),
        )

    def close(self) -> None:
        try:
            self._pool.terminate()
            self._pool.join()
        except Exception:
            logger.warning("PreprocPool close failed", exc_info=True)
