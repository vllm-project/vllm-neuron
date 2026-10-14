# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) weight loading
==============================================

The checkpoint is 475.2 GiB across 48 safetensors shards. One Trainium 2 device
has 96 GiB of HBM, 24 GiB per logical NeuronCore. So the loader cannot be a key
rename: it has to decide, per tensor, whether that tensor lives on the device for
the whole run or is streamed in when the computation that needs it runs.

Where the bytes are
-------------------
Measured on this checkpoint (see `weight_census()`, which reports it from the
index rather than from these notes):

    routed experts   275.67 GiB  fp4, 384 per layer x 40 layers + 128 x 3 (DSpark)
    engram tables    188.83 GiB  two fp8 tables of 384,006,168 x 256 rows
    resident           9.83 GiB  embedding, LM head, attention, norms, hc, gates
    unmapped           0.90 GiB  the vision tower, off the tokens -> logits path
                     ----------
                     475.24 GiB

98% of the checkpoint is in two places, and both are *sparsely read*:

  - **Routed experts.** 6 of 384 fire per token. A decode step with 4 sequences
    touches at most 24 experts of 384 in a layer — 6% of that layer's expert
    bytes. Loading all 384 would be loading 94% of nothing.
  - **Engram tables.** They are embedding tables. A token needs
    `(max_ngram_size - 1) * n_heads` = 24 rows of 256 fp8 values, so 6 KiB per
    token per layer. The 94.6 GiB is the *domain* of the lookup, not its cost.

So the split is: the 9.83 GiB of dense weights are loaded once, sharded 4 ways,
and stay resident (about 2.5 GiB per rank); the experts and the engram rows are
fetched per forward by `TimeshareWeightStore` into fixed device-resident staging
buffers that `model.py` reuses. "Timesharing the HBM" is that: one set of
buffers, refilled, rather than 475 GiB of parameters.

Why not `SafetensorsCheckpoint.load_sharded_pipelined`
------------------------------------------------------
That is the right tool for Qwen3/Llama3 and the wrong one here: it walks every
tensor named in the mapping and materializes it. Driving it with a mapping that
covers the whole checkpoint would try to realize 475 GiB. `SafetensorsIndex`
below reads shard *headers* only, and reads tensor bytes through `mmap`, so a
single expert's 2.6 MiB costs one page-fault burst instead of a shard.

Where the bytes are read from
-----------------------------
Throughput decides whether the model can run at all, and the two disks on this
instance differ by 9x:

    /            (EBS)             131 MB/s   measured, does not scale with
                                              parallel readers
    /dev/nvme1n1 (instance store)  1.1 GB/s   measured with O_DIRECT

At 131 MB/s, streaming the 286 GiB of non-engram weights once takes 37 minutes,
and the verifier has to finish 4 prompts in its budget. At 1.1 GB/s the same
pass is 4.3 minutes. So each shard is read from `TIMESHARE_CACHE_ROOT` when a
whole copy of it is staged there and from the original path otherwise, decided
per shard because the fast volume (430 GiB) is smaller than the checkpoint
(475.24 GiB). The cache holds the same bytes — a faster copy of the same shards,
admitted only on an exact size match against the original, not a different
checkpoint.

Intermediate state
------------------
`spill`/`unspill` put activations on the same fast volume. The residual stream
is `[tokens, hc_mult, hidden]`, which at 8192 tokens is 320 MiB in bf16 — small
next to the weights, but it has to survive while a layer's expert buffers own
the HBM it would otherwise sit in.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Container, Iterable, Iterator

import torch

if TYPE_CHECKING:
    from .model import DeepseekV41ForCausalLM

logger = logging.getLogger(__name__)

#: Where a faster copy of the checkpoint is looked for. Overridable so a host
#: that mounts its instance store somewhere else needs no code change.
TIMESHARE_CACHE_ROOT = Path(
    os.environ.get("DEEPSEEK_V41_TIMESHARE_CACHE", "/mnt/nvme/hf")
)

#: Threads `SafetensorsIndex.rows` issues its `pread`s from.
#:
#: The engram gather is 12,288 scattered rows per layer per 512-row forward
#: (`(engram_max_ngram_size - 1) * engram_n_heads = 24` columns a token), twice
#: — once for the fp8 values and once for their scales. Issued one at a time
#: they are 24,576 serialized device round trips, and the engram tables are the
#: two shards that do *not* fit on the instance store, so they come off the
#: root EBS volume. Measured on the verifier's own needle prefill: the two
#: engram layers cost 11-15 s each of a 140-227 s forward, ~10-18% of it.
#:
#: `os.pread` releases the GIL, so threads here really do raise the queue
#: depth; faulting the `mmap` from threads would not, because the faulting
#: thread holds the GIL while the page comes in.
_ROW_READ_THREADS = max(1, int(os.environ.get("DEEPSEEK_V41_ROW_THREADS", 16)))

#: Below this many rows the pool costs more than it saves and the gather runs
#: inline. One decode token asks for 24 rows.
_ROW_READ_MIN = 64

#: Lazily created, shared by every `rows()` call in the process.
_ROW_POOL: ThreadPoolExecutor | None = None


def _row_pool() -> ThreadPoolExecutor:
    global _ROW_POOL
    if _ROW_POOL is None:
        _ROW_POOL = ThreadPoolExecutor(
            max_workers=_ROW_READ_THREADS, thread_name_prefix="ds41-rows"
        )
    return _ROW_POOL


def _forget_row_pool() -> None:
    """Drop the pool in a forked child, where its threads do not exist.

    vLLM may start its workers with either `spawn` or `fork`, and a pool
    inherited across `fork()` has a work queue and no workers — every `map`
    would block forever. Rebuilding it lazily costs one thread creation.
    """
    global _ROW_POOL
    _ROW_POOL = None


os.register_at_fork(after_in_child=_forget_row_pool)

#: Where `spill`/`unspill` put intermediate activations. Falls back to the
#: cache volume's parent, then to /tmp.
TIMESHARE_SPILL_ROOT = Path(
    os.environ.get(
        "DEEPSEEK_V41_TIMESHARE_SPILL",
        str(TIMESHARE_CACHE_ROOT.parent / "timeshare-spill"),
    )
)

#: safetensors dtype strings -> torch dtypes. `I8` is how this checkpoint
#: stores fp4: two `e2m1` values packed per byte, so a logical `[out, in]`
#: weight is stored `[out, in // 2]` (see `vendor/model.py`'s `Linear`).
_ST_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.uint8,  # read as raw exponent bytes; see `dequant_block_fp8`
}


# =============================================================================
# Section 1: the checkpoint, read by header and by mmap
# =============================================================================


@dataclass(frozen=True)
class TensorRef:
    """Where one checkpoint tensor's bytes are, without having read them."""

    key: str
    shard: Path
    dtype: torch.dtype
    shape: tuple[int, ...]
    begin: int  # byte offset into the shard's data section
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class SafetensorsIndex:
    """Header-only index over a directory of safetensors shards.

    Reads `model.safetensors.index.json` for the key -> shard map and each
    shard's own JSON header for dtype, shape and byte range. That is ~100 MiB
    of header across the 48 shards and nothing else, so building this is
    seconds, not minutes, and never touches tensor data.

    `tensor()` and `rows()` read through one `mmap` per shard, kept open for the
    process's life. `rows()` is what makes the engram tables affordable: it
    faults in only the pages holding the requested rows.
    """

    def __init__(
        self, root: str | os.PathLike, cache_root: str | os.PathLike | None = None
    ) -> None:
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._refs: dict[str, TensorRef] = {}
        self._maps: dict[Path, mmap.mmap] = {}
        self._files: dict[Path, object] = {}
        #: `shard -> O_DIRECT descriptor`, or -1 where the kernel refused.
        self._direct: dict[Path, int] = {}
        self._build()

    def _resolve_shard(self, name: str) -> Path:
        """Prefer the staged copy of this shard, if there is a whole one.

        Resolved **per shard**, not per checkpoint, because the fast volume is
        smaller than the checkpoint: 430 GiB of instance store against 475.24
        GiB. Shards 1-46 (286 GiB of dense weights and routed experts) fit with
        room for one of the two 94.6 GiB engram tables; the other stays on EBS.
        An all-or-nothing rule would throw away the 46 shards' worth of speedup
        because of the one that does not fit.

        Size equality against the original is the admission test. A truncated
        shard — an interrupted stage, a volume that filled — would otherwise be
        read as if it were whole and yield silently wrong tensors.
        """
        source = self.root / name
        if self.cache_root is None:
            return source
        staged = self.cache_root / name
        try:
            if staged.stat().st_size == source.stat().st_size:
                return staged
        except OSError:
            pass
        return source

    # ── Index construction ────────────────────────────────────────────────

    def _shard_names(self) -> list[str]:
        index_file = self.root / "model.safetensors.index.json"
        if index_file.is_file():
            weight_map = json.loads(index_file.read_text())["weight_map"]
            # dict preserves insertion order; sorted() makes the read order
            # deterministic regardless of how the index was written.
            return sorted(set(weight_map.values()))
        shards = sorted(p.name for p in self.root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no safetensors shards and no index under {self.root}"
            )
        return shards

    def _build(self) -> None:
        staged = 0
        for name in self._shard_names():
            shard = self._resolve_shard(name)
            if self.cache_root is not None and shard.parent == self.cache_root:
                staged += 1
            if not shard.is_file():
                raise FileNotFoundError(f"shard named in the index is missing: {shard}")
            with open(shard, "rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                dtype = _ST_DTYPES.get(meta["dtype"])
                if dtype is None:
                    raise ValueError(f"{key}: unhandled dtype {meta['dtype']!r}")
                begin, end = meta["data_offsets"]
                self._refs[key] = TensorRef(
                    key=key,
                    shard=shard,
                    dtype=dtype,
                    shape=tuple(meta["shape"]),
                    begin=data_start + begin,
                    end=data_start + end,
                )
        shards = len({ref.shard for ref in self._refs.values()})
        logger.info(
            "indexed %d tensors across %d shards under %s; %d of them read from "
            "the staged copy at %s (~1.1 GB/s), %d from the original (~131 MB/s)",
            len(self._refs),
            shards,
            self.root,
            staged,
            self.cache_root,
            shards - staged,
        )

    # ── Lookup ────────────────────────────────────────────────────────────

    def __contains__(self, key: str) -> bool:
        return key in self._refs

    def keys(self) -> Iterable[str]:
        return self._refs.keys()

    def ref(self, key: str) -> TensorRef:
        try:
            return self._refs[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not in the checkpoint under {self.root}. This is a "
                "mapping bug, not a missing file: every key this loader asks "
                "for comes from build_weight_mappings()."
            ) from None

    def _map(self, shard: Path) -> mmap.mmap:
        existing = self._maps.get(shard)
        if existing is not None:
            return existing
        handle = open(shard, "rb")
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        # Keep the file object alive alongside the mapping.
        self._files[shard] = handle
        self._maps[shard] = mapping
        return mapping

    # ── Reads ─────────────────────────────────────────────────────────────

    def tensor(self, key: str) -> torch.Tensor:
        """A whole tensor, as a read-only view over the mmap.

        No copy, so the caller must `.clone()` (or `.to(device)`, which copies)
        before mutating. Views are cheap to make and cheap to drop, which is
        what lets a group be loaded and released per layer.
        """
        ref = self.ref(key)
        mapping = self._map(ref.shard)
        buffer = memoryview(mapping)[ref.begin : ref.end]
        flat = torch.frombuffer(buffer, dtype=ref.dtype)
        return flat.view(ref.shape) if ref.shape else flat

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Gather rows `indices` out of a 2-D tensor, reading only their pages.

        The engram path. `indices` may repeat and need not be sorted; the result
        is in `indices` order and is a fresh tensor (not a view), since the rows
        it came from are scattered.

        Read with `pread` from a thread pool rather than by faulting the `mmap`,
        and de-duplicated first. Both are about the same measurement: this
        gather is 12,288 scattered rows a layer on a table that lives on the
        slow volume, so its cost is round trips, not bytes.

          * **de-duplication.** A row is read once however many of the 24
            columns land on it. The result is still assembled in `indices`
            order, so a repeat is a dict lookup instead of a second read.
          * **`pread` instead of a page fault.** A fault on a shared
            file-backed mapping pulls in the kernel's fault-around window — 16
            pages on this kernel — so a 256-byte row costs 64 KiB off the
            volume. `pread` of exactly the row asks for one page. Same bytes
            out, an eighth of the traffic in.
          * **threads.** `pread` releases the GIL, so sixteen of them are
            sixteen requests in flight on a volume whose single-request
            latency, not its bandwidth, is the limit here.

        `_read_rows` falls back to the inline loop for small gathers, which is
        what a decode token's 24 rows takes.
        """
        ref = self.ref(key)
        if len(ref.shape) != 2:
            raise ValueError(f"rows() needs a 2-D tensor, {key} is {ref.shape}")
        n_rows, row_width = ref.shape
        itemsize = torch.empty((), dtype=ref.dtype).element_size()
        row_bytes = row_width * itemsize

        flat_indices = indices.reshape(-1).to(torch.int64)
        if flat_indices.numel() and (
            int(flat_indices.min()) < 0 or int(flat_indices.max()) >= n_rows
        ):
            raise IndexError(
                f"{key}: row index out of range for {n_rows} rows "
                f"(min={int(flat_indices.min())}, max={int(flat_indices.max())})"
            )

        wanted = flat_indices.tolist()
        blobs = self._read_rows(ref, sorted(set(wanted)), row_bytes)
        out = torch.empty((len(wanted), row_width), dtype=ref.dtype, device="cpu")
        for position, row in enumerate(wanted):
            out[position] = torch.frombuffer(blobs[row], dtype=ref.dtype)
        return out.view(*indices.shape, row_width)

    def _read_rows(
        self, ref: TensorRef, rows: list[int], row_bytes: int
    ) -> dict[int, bytes]:
        """`{row: bytes}` for the distinct rows of one gather. See `rows`."""
        # The mapping is still what `release_pages` madvises, and opening it
        # here is what keeps `self._files` populated for the descriptor below.
        self._map(ref.shard)
        handle = self._files[ref.shard]
        fileno = handle.fileno()  # type: ignore[attr-defined]

        def read(part: list[int]) -> list[tuple[int, bytes]]:
            return [
                (row, os.pread(fileno, row_bytes, ref.begin + row * row_bytes))
                for row in part
            ]

        if len(rows) < _ROW_READ_MIN or _ROW_READ_THREADS == 1:
            return dict(read(rows))

        workers = min(_ROW_READ_THREADS, len(rows))
        # Strided rather than contiguous slices: every worker then walks the
        # whole range, so the volume sees requests spread over the table
        # instead of sixteen hot spots.
        parts = [rows[i::workers] for i in range(workers)]
        out: dict[int, bytes] = {}
        for got in _row_pool().map(read, parts):
            out.update(got)
        return out

    def release_pages(self, key: str) -> None:
        """Drop one tensor's pages out of the page cache.

        Needed, not an optimization. Streaming 276 GiB of experts through `mmap`
        fills the page cache — measured 118 GiB of 124 — and the Neuron runtime
        then cannot get the physically contiguous host memory it wants for its
        DMA rings:

            TDRV:dmem_alloc_internal  Failed to allocate HOST memory
            (4194304 bytes): memory allocation failed (ret=-12)
            RuntimeError: The PyTorch Neuron Runtime could not be initialized

        `echo 3 > /proc/sys/vm/drop_caches` clears it, but a run that fills the
        cache *while* the device is live cannot be rescued that way. Since a
        streamed tensor is copied to the device immediately and never read
        twice, telling the kernel so keeps the cache flat.

        **`posix_fadvise` on the file, not only `madvise` on the mapping.**
        `MADV_DONTNEED` was doing nothing here and the docstring used to claim
        it did. For a *shared file-backed* mapping it tears down this process's
        page-table entries and leaves the page cache exactly as it was — it is
        `MAP_PRIVATE` anonymous memory that MADV_DONTNEED frees. Measured during
        the verifier run of 2026-10-12: `Cached:` climbed back to 44-59 GiB
        within a minute of every manual `drop_caches`, i.e. ~10-15 GiB a minute
        of expert bytes, with this method being called after every single read.
        `POSIX_FADV_DONTNEED` on the descriptor is the call that evicts clean
        page-cache pages for a byte range, and these pages are always clean
        (the mapping is `ACCESS_READ`).

        Both are issued: the madvise still releases the PTEs, which is cheap and
        keeps this process's resident size down, and the fadvise is what the
        page cache answers to. Neither is load-bearing for correctness — a
        dropped page is re-read, and the only cost of a refusal is a fuller
        cache — so both are best-effort.

        The range is widened to whole pages (down to the page below, up to the
        page above). A neighbouring tensor sharing those pages is re-faulted if
        it is read again, which costs a page fault and not correctness.
        """
        ref = self.ref(key)
        mapping = self._maps.get(ref.shard)
        if mapping is None:
            return
        page = mmap.PAGESIZE
        begin = (ref.begin // page) * page
        end = -(-ref.end // page) * page
        try:
            mapping.madvise(mmap.MADV_DONTNEED, begin, end - begin)
        except (OSError, ValueError):
            # Not fatal: the kernel is free to refuse, and the only cost is a
            # fuller page cache.
            pass
        handle = self._files.get(ref.shard)
        fileno = getattr(handle, "fileno", None)
        if fileno is None:
            return
        try:
            os.posix_fadvise(
                fileno(), begin, end - begin, os.POSIX_FADV_DONTNEED
            )
        except (OSError, ValueError, AttributeError):
            pass

    # ── Bulk reads, coalesced ─────────────────────────────────────────────

    def read_plan(
        self, keys: list[str], align: int = 1
    ) -> tuple[int, list[tuple], dict[str, tuple]]:
        """Plan one packed read of `keys`: `(nbytes, spans, layout)`.

        `spans` is `(shard, file_offset, nbytes, buffer_offset, needed)` per
        *contiguous run* of the requested byte ranges, and `layout` says where
        each key lands in the packed buffer. The point is the coalescing: one
        routed expert is six tensors, and in this checkpoint `w1/w2/w3.weight`
        are written back to back and so are the three `.scale`s, so six reads
        become **two**. Nothing assumes that — adjacency is tested per pair, and
        a layout that interleaved them would simply yield six spans.

        The buffer order follows the file order, which is what makes a
        coalesced span land contiguously in the destination.

        `align` widens every span to a multiple of it, at both ends, and aligns
        each span's buffer offset the same way. `O_DIRECT` needs that — offset,
        length and buffer address all have to be block-aligned — and the keys'
        own offsets inside the buffer absorb the padding, so a caller reads the
        same bytes either way. `needed` is how much of a widened span has to
        arrive for the data to be whole, which is less than `nbytes` only
        because of the padding past the end of the shard. `align=1` is the
        buffered layout, where `needed == nbytes`.
        """
        refs = {key: self.ref(key) for key in keys}
        order = sorted(keys, key=lambda k: (str(refs[k].shard), refs[k].begin))
        # Contiguous runs first, then their placement, because a span's buffer
        # offset depends on how long the run turned out to be.
        runs: list[list] = []
        for key in order:
            ref = refs[key]
            count = ref.end - ref.begin
            if runs and runs[-1][0] == ref.shard and runs[-1][1] + runs[-1][2] == ref.begin:
                run = runs[-1]
                run[3].append((key, run[2]))
                run[2] += count
            else:
                runs.append([ref.shard, ref.begin, count, [(key, 0)]])

        spans: list[tuple] = []
        layout: dict[str, tuple] = {}
        cursor = 0
        for shard, begin, count, members in runs:
            low = (begin // align) * align
            high = -(-(begin + count) // align) * align
            cursor = -(-cursor // align) * align
            spans.append((shard, low, high - low, cursor, begin + count - low))
            data = cursor + (begin - low)
            for key, relative in members:
                ref = refs[key]
                layout[key] = (
                    data + relative, ref.end - ref.begin, ref.shape, ref.dtype
                )
            cursor += high - low
        return cursor, spans, layout

    def _direct_fd(self, shard: Path) -> int | None:
        """A cached `O_DIRECT` descriptor on `shard`, or None if it is refused.

        Separate from the buffered descriptor `_map` keeps, because `O_DIRECT`
        is a property of the open file, and `release_pages`/`forget_spans` still
        want the buffered one.
        """
        fd = self._direct.get(shard)
        if fd is None:
            try:
                fd = os.open(shard, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            except OSError as exc:
                logger.warning(
                    "deepseek_v41_timeshare: O_DIRECT refused on %s (%s); "
                    "expert reads will go through the page cache",
                    shard, exc,
                )
                self._direct[shard] = -1
                return None
            self._direct[shard] = fd
        return None if fd == -1 else fd

    def read_spans(
        self, spans: list[tuple], buffer: torch.Tensor, direct: bool = False
    ) -> None:
        """Fill `buffer` (1-D `uint8`, host) from `spans`, with `preadv`.

        `pread` rather than faulting the `mmap` for the same reason `rows()`
        does it: it releases the GIL, so this can run on a background thread
        while the main thread stages the expert before it. A fault holds the
        GIL for the duration and cannot overlap with anything.

        **`direct` is what makes an expert read fast, and the reason is queue
        depth.** `max_hw_sectors_kb` on this instance store is 128, so every
        read is split into 128 KiB device requests, and the rate is
        `depth / latency`. A *buffered* `pread` of 16.87 MiB is served page by
        page with readahead capped at `read_ahead_kb` = 128, so it keeps one or
        two requests in flight: measured during a live prefill, 4 ranks reading
        this way gave 4585 requests/s at 0.90 ms and **599-835 MB/s aggregate**.
        An `O_DIRECT` `preadv` of the same 16.87 MiB submits the whole extent as
        one bio chain — ~135 requests at once — and DMAs into the caller's pages
        with no page-cache allocation and nothing for `forget_spans` to undo.
        """
        view = memoryview(buffer.numpy().data).cast("B")
        for shard, begin, count, at, needed in spans:
            fd = self._direct_fd(shard) if direct else None
            if fd is None:
                self._map(shard)
                fd = self._files[shard].fileno()  # type: ignore[attr-defined]
            got = os.preadv(fd, [view[at : at + count]], begin)
            if got < needed:
                # A span `align` widened past the end of the shard comes back
                # short, which is fine as long as every requested byte
                # arrived. Anything shorter than that is not.
                raise OSError(
                    f"short read of {shard.name}: needed {needed} bytes at "
                    f"{begin}, got {got}"
                )

    def forget_spans(self, spans: list[tuple]) -> None:
        """`release_pages` for a planned read: one `fadvise` per span.

        Same contract and same reason as `release_pages` — these bytes are read
        once per forward and leaving them in the page cache starves the Neuron
        runtime of contiguous host memory — but two syscalls for an expert
        instead of twelve, and issued off the critical path by whichever thread
        did the read.
        """
        page = mmap.PAGESIZE
        for shard, begin, count, _at, _needed in spans:
            handle = self._files.get(shard)
            if handle is None:
                continue
            low = (begin // page) * page
            high = -(-(begin + count) // page) * page
            try:
                os.posix_fadvise(
                    handle.fileno(), low, high - low, os.POSIX_FADV_DONTNEED  # type: ignore[attr-defined]
                )
            except (OSError, ValueError, AttributeError):
                pass

    def close(self) -> None:
        for fd in self._direct.values():
            if fd != -1:
                os.close(fd)
        self._direct.clear()
        for mapping in self._maps.values():
            mapping.close()
        for handle in self._files.values():
            handle.close()  # type: ignore[attr-defined]
        self._maps.clear()
        self._files.clear()


def staged_cache_root(checkpoint_path: str | os.PathLike) -> Path | None:
    """The staged copy to prefer, or None if there is nothing usable there.

    Only checks that the directory exists and holds at least one shard the
    original names — which shards are usable is decided per shard by
    `SafetensorsIndex._resolve_shard`, because the fast volume is smaller than
    the checkpoint and a partial stage is the expected steady state, not an
    error.

    Throughput, measured on this instance, is why any of this exists:

        /            (EBS)            131-139 MB/s, and it does **not** scale
                                      with parallel readers — 8 concurrent
                                      O_DIRECT readers over 8 shards gave
                                      129 MB/s aggregate
        /mnt/nvme    (instance store) 2.3 GB/s written with O_DIRECT; ~730 MB/s
                                      effective through this module's mmap

    A layer's 384 experts are 7.3 GiB, so the per-layer cost of one prefill
    forward is ~55 s unstaged against ~10 s staged, and a 40-layer prefill is
    **64 minutes against 7.5** (both measured, 2026-10-12). The verifier's
    budget for all four prompts is 5000 s, so the staged copy is not a
    performance nicety here: unstaged, the gate cannot pass.

    **The volume is an instance store and does not survive a stop/start.** It
    was found empty, with no filesystem at all, at the start of iteration 6 —
    after five iterations had attributed the resulting slowness to the model.
    Hence the error-level log below rather than a warning: an iteration that
    misses this loses an hour per run.
    """
    cache = TIMESHARE_CACHE_ROOT
    index_file = Path(checkpoint_path) / "model.safetensors.index.json"
    if not cache.is_dir() or not index_file.is_file():
        if not cache.is_dir():
            logger.error(
                "NO STAGED CHECKPOINT at %s — every weight read will come off "
                "EBS at ~131 MB/s, which is 8x slower and makes a 40-layer "
                "prefill ~64 min instead of ~7.5. Check `df -h %s`; if the "
                "instance store is unmounted (it does not survive a "
                "stop/start) then mkfs+mount it and copy shards 1-46 of %s "
                "there, which takes ~38 min of pure wall clock and can run in "
                "the background. See .autohelix/notes/iter-6.md section 4a.",
                cache,
                cache,
                checkpoint_path,
            )
        return None
    wanted = set(json.loads(index_file.read_text())["weight_map"].values())
    if not any((cache / name).is_file() for name in wanted):
        logger.error(
            "the staged checkpoint directory %s exists but holds none of the "
            "48 shards this checkpoint names; reading everything from %s at "
            "~131 MB/s. Same remedy as above.",
            cache,
            checkpoint_path,
        )
        return None
    return cache


def stage_checkpoint(
    checkpoint_path: str | os.PathLike, max_shards: int = 46
) -> int:
    """Copy the checkpoint's shards onto `TIMESHARE_CACHE_ROOT`. Returns how
    many it staged.

    An operational entry point, called by hand and by nothing in the serving
    path — `staged_cache_root` only *reads* what this put there. It exists in
    this module rather than as a shell script because the admission rule it has
    to satisfy lives here: `_resolve_shard` accepts a staged shard only on an
    exact size match against the original, and a half-written file that happened
    to match would yield silently wrong tensors. So the copy goes to `.part` and
    is renamed only after its size is checked, which makes the whole thing
    interruptible and resumable — re-running it skips what is already whole.

        python -c "import vllm_neuron.model.deepseek_v41_timeshare.weight_loader as w; \\
                   w.stage_checkpoint('/home/ubuntu/workspace/partition-artifact/hf')"

    The mount is the caller's job, since it needs root:

        sudo mkfs.ext4 -F -m 0 -E lazy_itable_init=1,lazy_journal_init=1 /dev/nvme1n1
        sudo mount -o noatime /dev/nvme1n1 /mnt/nvme
        sudo mkdir -p /mnt/nvme/hf && sudo chown -R "$USER" /mnt/nvme

    `max_shards=46` is the default because this checkpoint's last two shards are
    the two 94.6 GiB engram tables: 286 GiB of dense weights and experts fit on
    the 430 GiB volume and 476 GiB does not, and the engram path reads 256-byte
    rows at random, which is an IOPS workload that a faster volume helps far
    less. Raise it if the volume is bigger.

    Reads are `O_DIRECT` so that staging ~286 GiB does not leave ~286 GiB in the
    page cache — the same concern as `release_pages`, and the reason this does
    not simply call `shutil.copyfile`.
    """
    source = Path(checkpoint_path)
    target = TIMESHARE_CACHE_ROOT
    index_file = source / "model.safetensors.index.json"
    if not target.is_dir():
        raise RuntimeError(
            f"{target} is not a directory; mount the fast volume there first "
            "(see this function's docstring)"
        )
    names = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
    block = 8 << 20
    staged = 0
    for name in names[:max_shards]:
        src, dst = source / name, target / name
        want = src.stat().st_size
        if dst.is_file() and dst.stat().st_size == want:
            continue
        part = dst.with_suffix(dst.suffix + ".part")
        # O_DIRECT needs an aligned buffer, which `os.preadv` into a memoryview
        # of a page-aligned mmap gives us without a third-party dependency.
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        fd = os.open(src, flags)
        try:
            with open(part, "wb") as out:
                buffer = mmap.mmap(-1, block)
                try:
                    offset = 0
                    while offset < want:
                        got = os.preadv(fd, [buffer], offset)
                        if got <= 0:
                            break
                        out.write(memoryview(buffer)[: min(got, want - offset)])
                        offset += got
                finally:
                    buffer.close()
                out.flush()
                os.fsync(out.fileno())
        finally:
            os.close(fd)
        written = part.stat().st_size
        if written != want:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"staging {name} produced {written} bytes, expected {want}; "
                "removed the partial copy rather than leaving one that "
                "_resolve_shard might accept"
            )
        part.rename(dst)
        staged += 1
        logger.info("staged %s (%d MiB)", name, want >> 20)
    logger.info("staged %d shard(s) under %s", staged, target)
    return staged


# =============================================================================
# Section 2: dequantization
# =============================================================================
# The checkpoint stores three formats, all with `e8m0` (power-of-two) block
# scales, and `vendor/kernel.py` is the reference for each:
#
#   fp8  e4m3, one scale per 32x32 weight block     attention, shared experts
#   fp4  e2m1, two per byte, one scale per 32 along K    routed experts
#   fp8  e4m3, one scale per 32 along the row       engram table rows
#
# These helpers exist for the host staging path and for checking a kernel's
# output against a reference. On the device the fp8/fp4 GEMMs under `kernels/`
# consume the quantized form directly — dequantizing to bf16 first would
# quadruple both the HBM a staged expert occupies and the bytes crossing PCIe.


#: The dtypes the checkpoint uses for *packed* weights, which this loader hands
#: over as raw `uint8` instead. The NKI kernels in `kernels/blockquant.py` want
#: the bytes, and a `Tensor.view(torch.uint8)` performed inside a traced graph
#: is rejected at execution time by the HOP bridge —
#:
#:     FATAL: Expected all tensors in the given list to be XLA tensors.
#:     Element at index 1 is not an XLA tensor. Got: XLAByteType
#:
#: — so the bitcast has to happen on the host, before the buffer reaches the
#: device. `float8_e4m3fn` and `int8` are both one byte per element, so this is
#: a reinterpretation and never a copy or a conversion.
_PACKED_DTYPES = (torch.float8_e4m3fn, torch.int8)


def as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a packed fp8/fp4 weight as `uint8`; pass anything else on."""
    if tensor.dtype in _PACKED_DTYPES:
        return tensor.view(torch.uint8)
    return tensor


def e8m0_to_scale(raw: torch.Tensor, dtype: torch.dtype = torch.float32):
    """Decode `e8m0` exponent bytes into multipliers: `2 ** (byte - 127)`.

    `device=raw.device` is load-bearing: this runs inside the traced graph, and a
    CPU `ones` against a device exponent is a fake-tensor device mismatch that
    dynamo reports from the middle of warmup —

        Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two
        different devices cpu, neuron:0

    which names `ldexp` and not the constant that caused it.
    """
    return torch.ldexp(
        torch.ones(raw.shape, dtype=dtype, device=raw.device),
        raw.to(torch.int32) - 127,
    )


def dequant_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """fp8 `e4m3` weights with one `e8m0` scale per `block` x `block` tile.

    `weight` is `[out, in]`; `scale` is `[ceil(out/block), ceil(in/block)]`.
    """
    out_features, in_features = weight.shape
    if weight.dtype is torch.uint8:
        weight = weight.view(torch.float8_e4m3fn)
    values = weight.to(torch.float32)
    factors = e8m0_to_scale(scale)
    factors = factors.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (values * factors[:out_features, :in_features]).to(dtype)


def dequant_block_fp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    block: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """fp4 `e2m1` weights, two per byte, one `e8m0` scale per `block` along K.

    `packed` is `[out, in // 2]` as stored; the result is `[out, in]`. The low
    nibble of a byte is the lower K index, matching `vendor/kernel.py`'s
    `fp4_gemm`.
    """
    out_features, packed_width = packed.shape
    in_features = packed_width * 2
    raw = packed.to(torch.uint8)
    nibbles = torch.stack([raw & 0x0F, raw >> 4], dim=-1).reshape(
        out_features, in_features
    )
    values = _E2M1_TABLE.to(nibbles.device)[nibbles.to(torch.int64)]
    factors = e8m0_to_scale(scale).repeat_interleave(block, 1)
    return (values * factors[:, :in_features]).to(dtype)


#: `e2m1`: sign in bit 3, 2 exponent bits, 1 mantissa bit. The 16 values, in
#: code order, as `vendor/kernel.py` decodes them.
_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_engram_rows(
    rows: torch.Tensor, scales: torch.Tensor, block: int = 32
) -> torch.Tensor:
    """Engram table rows: fp8 `e4m3` with one `e8m0` scale per 32 columns.

    Mirrors `ParallelEngramEmbedding.forward` in `vendor/model.py`: unflatten
    the row into blocks, scale, flatten back, cast to bf16.
    """
    *lead, width = rows.shape
    blocks = width // block
    values = rows.to(torch.float32).reshape(*lead, blocks, block)
    factors = e8m0_to_scale(scales).reshape(*lead, blocks, 1)
    return (values * factors).reshape(*lead, width).to(torch.bfloat16)


# =============================================================================
# Section 3: what lives where — the parameter/checkpoint mapping
# =============================================================================
# Checkpoint keys are the reference implementation's own module paths
# (`vendor/model.py`), with no `model.` prefix: `layers.<i>.attn.wq_b.weight`,
# `layers.<i>.ffn.experts.<e>.w1.weight`, `mtp.<s>....`. `model.py` keeps those
# names, so the mapping is 1:1 and auditable against the file.


def _attention_keys(prefix: str, config, layer_id: int) -> dict[str, str]:
    """One attention block's resident keys.

    `prefix` is the checkpoint prefix (`layers.3.attn` or `mtp.0.attn`).
    Which extras exist is decided by the config, not guessed: a layer carries a
    `compressor` only if it is in `kv_source_layer_ids`, and an `indexer` only
    if it is in `index_source_layer_ids`. An indexer owns `wk`/`k_norm` only
    when it is also a KV source — the others read the index keys their source
    published (`Indexer.owns_k` in `vendor/model.py`).
    """
    keys = {
        f"{prefix}.attn_sink": f"{prefix}.attn_sink",
        f"{prefix}.q_norm.weight": f"{prefix}.q_norm.weight",
        f"{prefix}.kv_norm.weight": f"{prefix}.kv_norm.weight",
    }
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        keys[f"{prefix}.{name}.weight"] = f"{prefix}.{name}.weight"
        keys[f"{prefix}.{name}.scale"] = f"{prefix}.{name}.scale"

    is_kv_source = layer_id in config.kv_source_layer_ids
    is_index_source = layer_id in config.index_source_layer_ids

    if is_kv_source:
        for name in ("norm.weight", "wkv.weight"):
            keys[f"{prefix}.compressor.{name}"] = f"{prefix}.compressor.{name}"
        # `wgate` only exists above compress_ratio 1: ratio 1 is a plain
        # projection with no softmax pooling to gate (`Compressor.__init__`).
        if config.compress_ratios[layer_id] > 1:
            keys[f"{prefix}.compressor.wgate.weight"] = (
                f"{prefix}.compressor.wgate.weight"
            )

    if is_index_source:
        keys[f"{prefix}.indexer.weights_proj.weight"] = (
            f"{prefix}.indexer.weights_proj.weight"
        )
        keys[f"{prefix}.indexer.wq_b.weight"] = f"{prefix}.indexer.wq_b.weight"
        keys[f"{prefix}.indexer.wq_b.scale"] = f"{prefix}.indexer.wq_b.scale"
        if is_kv_source:
            keys[f"{prefix}.indexer.wk.weight"] = f"{prefix}.indexer.wk.weight"
            keys[f"{prefix}.indexer.k_norm.weight"] = f"{prefix}.indexer.k_norm.weight"

    return keys


def _block_keys(prefix: str, config, layer_id: int) -> dict[str, str]:
    """One decoder block's resident keys: norms, hyper-connections, attention,
    the MoE gate and shared expert. The 384 routed experts are deliberately
    absent — they are streamed, see `TimeshareWeightStore.expert`.
    """
    keys = {
        f"{prefix}.attn_norm.weight": f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight": f"{prefix}.ffn_norm.weight",
        f"{prefix}.ffn.gate.weight": f"{prefix}.ffn.gate.weight",
        f"{prefix}.ffn.gate.bias": f"{prefix}.ffn.gate.bias",
        # The VL routing bias. Present in the checkpoint for every layer and
        # loaded, so the routing path can be read against the reference, but
        # `Gate.forward` only selects it for tokens inside an image span and
        # this port serves text, where `image_mask` is None.
        f"{prefix}.ffn.gate.bias_vl": f"{prefix}.ffn.gate.bias_vl",
    }
    for name in ("w1", "w2", "w3"):
        keys[f"{prefix}.ffn.shared_experts.{name}.weight"] = (
            f"{prefix}.ffn.shared_experts.{name}.weight"
        )
        keys[f"{prefix}.ffn.shared_experts.{name}.scale"] = (
            f"{prefix}.ffn.shared_experts.{name}.scale"
        )
    for name in (
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    ):
        keys[f"{prefix}.{name}"] = f"{prefix}.{name}"

    keys.update(_attention_keys(f"{prefix}.attn", config, layer_id))

    if layer_id in config.engram_layer_ids:
        # The table itself (`engram.embed.weight`/`.scale`, 94.6 GiB) is streamed
        # by row; only the projection that turns looked-up rows into a key/value
        # and the two gate vectors are resident.
        for name in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            keys[f"{prefix}.engram.{name}"] = f"{prefix}.engram.{name}"

    return keys


def build_weight_mappings(model: DeepseekV41ForCausalLM) -> dict[str, str | list[str]]:
    """Map this model's resident parameter names to checkpoint key(s).

    Resident only. Every key here is loaded once, sharded across the 4 ranks and
    left on the device; the two streamed families are not named here at all:

      - `layers.<i>.ffn.experts.<e>.{w1,w2,w3}.{weight,scale}` — 384 per layer
        (128 for the DSpark stages), fetched per forward by expert id.
      - `layers.{1,14}.engram.embed.{weight,scale}` — fetched per forward by
        row.

    Parameter names equal checkpoint keys here, because `model.py` mirrors the
    reference's module paths. The identity is the point: it makes "is this the
    weight the reference uses" a string comparison against the index rather
    than an argument about a renaming scheme.
    """
    config = model.config
    mappings: dict[str, str | list[str]] = {
        "embed.weight": "embed.weight",
        "norm.weight": "norm.weight",
        "head.weight": "head.weight",
    }

    for layer_id in range(config.num_hidden_layers):
        mappings.update(_block_keys(f"layers.{layer_id}", config, layer_id))

    # The DSpark stages live under `mtp.<stage>` in the checkpoint and are
    # indexed as layers `num_hidden_layers + stage` by the config's per-layer
    # tables (`compress_ratios` carries 43 entries: 40 + 3).
    for stage in range(config.num_nextn_predict_layers):
        layer_id = config.num_hidden_layers + stage
        mappings.update(_block_keys(f"mtp.{stage}", config, layer_id))

    # Stage 0 projects the concatenated hidden states of the target layers
    # (37, 38, 39) down to one hidden; the last stage carries the heads.
    mappings["mtp.0.main_proj.weight"] = "mtp.0.main_proj.weight"
    mappings["mtp.0.main_proj.scale"] = "mtp.0.main_proj.scale"
    mappings["mtp.0.main_norm.weight"] = "mtp.0.main_norm.weight"

    last = config.num_nextn_predict_layers - 1
    mappings[f"mtp.{last}.norm.weight"] = f"mtp.{last}.norm.weight"
    mappings[f"mtp.{last}.markov_head.embed.weight"] = (
        f"mtp.{last}.markov_head.embed.weight"
    )
    mappings[f"mtp.{last}.markov_head.head.weight"] = (
        f"mtp.{last}.markov_head.head.weight"
    )
    mappings[f"mtp.{last}.confidence_head.proj.weight"] = (
        f"mtp.{last}.confidence_head.proj.weight"
    )

    return mappings


def streamed_key_patterns(config) -> dict[str, list[str]]:
    """The two families `build_weight_mappings` leaves out, spelled as keys.

    Returned as concrete key lists rather than globs so a caller can check them
    against the index — the point of this function is to make "what is streamed"
    enumerable and therefore auditable, not to describe it in prose.
    """
    experts: list[str] = []
    for layer_id in range(config.num_hidden_layers):
        for expert in range(config.n_routed_experts):
            for name in ("w1", "w2", "w3"):
                experts.append(
                    f"layers.{layer_id}.ffn.experts.{expert}.{name}.weight"
                )
                experts.append(f"layers.{layer_id}.ffn.experts.{expert}.{name}.scale")
    for stage in range(config.num_nextn_predict_layers):
        for expert in range(config.dspark_n_routed_experts):
            for name in ("w1", "w2", "w3"):
                experts.append(f"mtp.{stage}.ffn.experts.{expert}.{name}.weight")
                experts.append(f"mtp.{stage}.ffn.experts.{expert}.{name}.scale")

    engram = [
        f"layers.{layer_id}.engram.embed.{suffix}"
        for layer_id in config.engram_layer_ids
        for suffix in ("weight", "scale")
    ]
    return {"experts": experts, "engram": engram}


# =============================================================================
# Section 4: the store that streams
# =============================================================================


#: How many routed experts the reader thread may run ahead of the expert loop.
#: **0, because it was measured on the device twice and it is not worth its
#: cost** — see `_ExpertStream`, which is kept for the reason the comment on
#: `_EXPERT_DIRECT_IO` is kept.
#:
#: The idea was overlap, not parallelism. Iteration 1's profile shows 68.9 ms of
#: device idle between consecutive expert GEMMs; run those back to back and the
#: read and the staging add, run the read of expert `i+1` against the staging of
#: expert `i` and the pair costs the larger of the two. It works mechanically —
#: a verifier run reports `prefetch 2094 served (2085 stalled, 0 inline)`, so
#: every read was prefetched and nothing fell back — and it buys nothing,
#: because the consumer is never what is late: the 68.9 ms is ~100% the read,
#: and the read is at this volume's rate for four concurrent readers (599-835
#: MB/s aggregate, from `/proc/diskstats` during a live prefill).
#:
#: What it costs, from the two verifier runs of 2026-10-14 (both at four ranks,
#: the gate's own command): a prefill forward's wall clock goes 211.2 s -> 220.7
#: s, and `device_ttft_ms_short` goes **+5.6% and +16.5%** against iteration 1's
#: three samples of 6460-6812 ms — no overlap, so a real effect and not the
#: instrument. `device_ttft_ms_long` moved -9.1% and then -0.6% on the same
#: tree, which is what this instrument's noise looks like. It is exact — run 2's
#: four completions are character-identical to iteration 1's and DSpark
#: acceptance is unchanged at 3.6 tokens a step — so the only thing wrong with
#: it is that it is slower.
#:
#: Set `DEEPSEEK_V41_EXPERT_PREFETCH=4` to turn it back on. Worth re-testing
#: only if the storage changes: on a volume that is latency-bound rather than
#: bandwidth-bound, the overlap would pay.
#:
#: If it is ever turned on, keep it at **one reader thread per rank** — the ring
#: depth is not a thread count. Reading 17.93 MiB chunks off `/mnt/nvme`:
#: 1 reader 2.00 GB/s, 2 readers 2.11, 4 readers 1.97, **8 readers 0.84** (and
#: those absolute numbers are inflated by a warm page cache; the shape held).
_EXPERT_PREFETCH_DEPTH = max(
    0, int(os.environ.get("DEEPSEEK_V41_EXPERT_PREFETCH", 0))
)

#: Read a routed expert with two coalesced `preadv`s into a reusable buffer
#: (`TimeshareWeightStore._read_expert`) rather than six page-faulted
#: `.clone()`s out of the `mmap` plus twelve `madvise`/`fadvise` calls.
#:
#: **Off, and this is the one thing in this file worth resolving next.** It is
#: where the model's time demonstrably is: measured on the device at four ranks,
#: a 512-row prefill forward spends **121-123 s of its ~180 s inside
#: `TimeshareWeightStore.expert()`** — 70.4 ms for each of ~1,695 routed experts
#: — against 2.6 ms for `_stage`'s host->device copies and 1.0 ms for the expert
#: GEMM launch. A clean four-process benchmark reads the same 18.8 MiB in
#: 24-25 ms, so ~45 ms an expert (~76 s a forward) is host memcpy and syscall
#: time that this path exists to remove. See notes iter-2 §12.
#:
#: Why it is nevertheless off: the two measurements of it disagree and there was
#: no verifier run left to break the tie. Single reader, device idle, warm
#: cache: **9.3-9.6 ms an expert against the `mmap` path's 13.1-13.6**
#: (`/tmp/it2/streamtest.py`). Single reader with four other ranks hammering the
#: volume: **73.6 ms against 58.2** (`/tmp/it2/synctest.py`), i.e. the other
#: way round. Shipping a read-path change on a 50/50 is how iteration 2 spent
#: its first run — see `_EXPERT_PREFETCH_DEPTH`. The A/B is one env var and
#: the bytes are identical either way (`/tmp/it2/synctest.py` checks every byte
#: of twelve experts across layers 0, 5, 14, 39 and the three MTP stages).
#:
#: `DEEPSEEK_V41_EXPERT_READ=preadv` turns it on.
_EXPERT_READ_COALESCED = (
    os.environ.get("DEEPSEEK_V41_EXPERT_READ", "mmap") != "mmap"
)

#: Read routed experts with `O_DIRECT`. **Off, because it was measured and it
#: bought nothing** — kept because the reasoning for trying it is sound and the
#: next iteration should not pay to rediscover that it does not apply here.
#:
#: The theory: `max_hw_sectors_kb` on this volume is 128, so the read rate is
#: `queue_depth / latency`, and a *buffered* `pread` of a 16.87 MiB extent is
#: served page by page with readahead capped at `read_ahead_kb` = 128, keeping
#: one or two requests in flight — measured live at 4585 requests/s, 0.90 ms
#: each, depth 4.1 over four ranks, 599-835 MB/s. An `O_DIRECT` `preadv` submits
#: the whole extent as one bio chain, ~135 requests at once, and skips the page
#: cache entirely (so `forget_spans` has nothing to undo either).
#:
#: What happened: **no change at all.** The verifier run of 2026-10-14 07:17
#: served its whole short phase through this path and its per-layer times match
#: iteration 1's to within 3% (182.2 s against 176.6 s for the cold prefill
#: forward, layer for layer). So the ~600-800 MB/s is this volume's real rate
#: for this access pattern, not an artifact of how the read is issued.
#:
#: Set `DEEPSEEK_V41_EXPERT_DIRECT=1` to turn it back on; the bytes are
#: identical either way (`/tmp/it2/directtest.py` checks them against the
#: `mmap` path, including the alignment widening).
_EXPERT_DIRECT_IO = os.environ.get("DEEPSEEK_V41_EXPERT_DIRECT", "0") != "0"

#: `O_DIRECT` wants offset, length and buffer address aligned to the logical
#: block size (512 here). A page is a safe multiple of every plausible value and
#: costs at most 8 KiB of extra read per expert.
_DIRECT_ALIGN = mmap.PAGESIZE


#: `[served, waited, unplanned]` since the last forward boundary: experts the
#: reader thread had ready, of those the ones the consumer still had to wait
#: for, and reads that were never announced and so ran inline. Reported and
#: reset by `model._hbm_mark`. Instrumentation only.
EXPERT_STREAM_STATS = [0, 0, 0]


class _ExpertStream:
    """One background thread reading routed experts into reusable host buffers.

    **Off by default and not on the live path — `_EXPERT_PREFETCH_DEPTH` is 0.**
    Read that comment before reaching for this: it is correct (a verifier run
    through it produced completions character-identical to the run without it)
    and it is measurably *slower*, because the thing it overlaps is not the
    thing that is late. It is kept so the measurement is reproducible from one
    env var rather than from a re-implementation.

    The consumer (`MoE.forward`'s expert loop) announces the experts it is about
    to ask for, in order, with `submit`; `take` then either hands back a buffer
    the thread has already filled or waits for the one in flight. A `take` for
    an expert that was never announced returns `None`, and the caller reads it
    the old way — so this is an accelerator for the one call site that can plan
    ahead, not a new requirement on the others.

    Buffer discipline, which is what keeps this safe: the tensors handed out are
    *views into a reusable buffer*, valid until the next `take` or `submit`. The
    only consumer is `MoE._expert`, which copies them into device buffers
    immediately (`_stage`) and keeps nothing. The ring is `depth` buffers of one
    expert each — 17.93 MiB, so 4 of them is 72 MiB of host memory per rank.

    Ordering: the thread reads in announced order and the consumer consumes in
    the same order, so whatever is ready when the consumer is waiting for a
    still-queued expert can only be an expert the consumer has already skipped
    (its cache turned a planned miss into a hit). Those are dropped, which is
    also what stops the ring deadlocking on buffers nobody will claim.
    """

    def __init__(self, store: "TimeshareWeightStore", depth: int) -> None:
        self._store = store
        self._depth = max(1, depth)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[tuple[int, int]] = []
        self._pending: set[tuple[int, int]] = set()
        self._ready: dict[tuple[int, int], tuple[int, dict]] = {}
        self._failed: dict[tuple[int, int], BaseException] = {}
        self._buffers: list[torch.Tensor | None] = [None] * self._depth
        self._free: list[int] = list(range(self._depth))
        self._held: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = False

    # ── Consumer side ─────────────────────────────────────────────────────

    def submit(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        """Announce the experts the caller is about to fetch, in order."""
        wanted = [(layer_id, int(e)) for e in expert_ids]
        if not wanted:
            return
        with self._lock:
            self._recycle_locked()
            self._drop_ready_locked()
            self._queue = list(wanted)
            self._pending = set(wanted)
            self._failed.clear()
            self._cv.notify_all()
        self._ensure_thread()

    def take(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor] | None:
        """The expert's bytes if this stream owns it, else `None`."""
        key = (layer_id, int(expert_id))
        with self._lock:
            self._recycle_locked()
            waited = False
            while True:
                got = self._ready.pop(key, None)
                if got is not None:
                    self._held = got[0]
                    EXPERT_STREAM_STATS[0] += 1
                    if waited:
                        EXPERT_STREAM_STATS[1] += 1
                    return got[1]
                if key in self._failed:
                    # Hand the caller back to the inline `mmap` path rather
                    # than re-raising: that path is what ran before this class
                    # existed, it reads the same bytes, and if the cause is
                    # real (a bad key, a truncated shard) it raises there too,
                    # from the stack frame that asked for the expert.
                    logger.warning(
                        "deepseek_v41_timeshare: prefetch of layer %d expert "
                        "%d failed, reading it inline instead: %r",
                        key[0], key[1], self._failed.pop(key),
                    )
                    self._pending.discard(key)
                    EXPERT_STREAM_STATS[2] += 1
                    return None
                if key not in self._pending:
                    EXPERT_STREAM_STATS[2] += 1
                    return None
                if self._ready:
                    # Waiting on something still queued, so anything ready is
                    # an expert the consumer has already passed. Free it.
                    self._drop_ready_locked()
                    continue
                waited = True
                self._cv.wait()

    def finish(self) -> None:
        """Release what the last `take` handed out. Called between layers."""
        with self._lock:
            self._recycle_locked()
            self._drop_ready_locked()
            self._queue.clear()
            self._pending.clear()

    # ── Producer side ─────────────────────────────────────────────────────

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="ds41-experts", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._stop and not (self._queue and self._free):
                    self._cv.wait()
                if self._stop:
                    return
                key = self._queue.pop(0)
                index = self._free.pop()
            try:
                blob = self._fill(index, key)
            except BaseException as exc:  # noqa: BLE001 - handed to the consumer
                with self._lock:
                    self._failed[key] = exc
                    self._free.append(index)
                    self._cv.notify_all()
                continue
            with self._lock:
                if key in self._pending:
                    self._ready[key] = (index, blob)
                else:
                    # The layer moved on while this was in flight.
                    self._free.append(index)
                self._cv.notify_all()

    def _fill(self, index: int, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        layer_id, expert_id = key
        names = self._store.expert_keys(layer_id, expert_id)
        align = _DIRECT_ALIGN if _EXPERT_DIRECT_IO else 1
        nbytes, spans, layout = self._store.index.read_plan(
            list(names.values()), align
        )
        buffer = self._aligned_buffer(index, nbytes, align)
        self._store.index.read_spans(spans, buffer, direct=_EXPERT_DIRECT_IO)
        if not _EXPERT_DIRECT_IO:
            # Nothing to drop when the read never entered the page cache.
            self._store.index.forget_spans(spans)
        blob = {}
        for short, full in names.items():
            at, count, shape, _dtype = layout[full]
            flat = buffer[at : at + count]
            blob[short] = flat.view(shape) if shape else flat
        return blob

    def _aligned_buffer(self, index: int, nbytes: int, align: int) -> torch.Tensor:
        """Ring slot `index`, grown if it has to be, aligned if `O_DIRECT` asks.

        `align` bytes of slack are allocated so an aligned window always exists
        inside the slot; the window is a view, so the blob views below still
        reference the slot's one allocation and the ring stays `depth` buffers.
        """
        raw = self._buffers[index]
        want = nbytes + align
        if raw is None or raw.numel() < want:
            raw = torch.empty(want, dtype=torch.uint8)
            self._buffers[index] = raw
        if align <= 1:
            return raw
        pad = (-raw.data_ptr()) % align
        return raw[pad : pad + nbytes]

    # ── Ring bookkeeping (caller holds the lock) ──────────────────────────

    def _recycle_locked(self) -> None:
        if self._held is not None:
            self._free.append(self._held)
            self._held = None
            self._cv.notify_all()

    def _drop_ready_locked(self) -> None:
        for index, _blob in self._ready.values():
            self._free.append(index)
        if self._ready:
            self._ready.clear()
            self._cv.notify_all()


class TimeshareWeightStore:
    """Serves the streamed weights, one module group at a time.

    Holds no weight bytes of its own: every read goes through the index's mmap,
    so what the host keeps is page cache, which the kernel can evict under
    pressure. That is deliberate — 124 GiB of RAM cannot hold 289 GiB of experts,
    and pretending otherwise just moves the failure from "slow" to "OOM".

    The three access patterns, each matching how `model.py` consumes them:

      `expert(layer_id, expert_id)`  one expert's three fp4 matrices + scales
      `experts(layer_id, ids)`       a decode step's routed set, in one call
      `engram(layer_id, hash_ids)`   the rows for a batch of n-gram hashes
    """

    def __init__(self, index: SafetensorsIndex, config) -> None:
        self.index = index
        self.config = config
        self._spill_root: Path | None = None
        self._stream_owner: tuple[int, _ExpertStream] | None = None
        #: One expert's worth of host memory, refilled by `_read_expert`.
        self._blob: torch.Tensor | None = None

    # ── Routed experts ────────────────────────────────────────────────────

    @property
    def _stream(self) -> _ExpertStream | None:
        """The reader thread of *this* process, or None.

        Keyed on the pid because vLLM may start its workers with `fork`, and a
        thread does not survive into the child: an inherited `_ExpertStream`
        would have a ring whose producer does not exist, so every `take` of an
        announced expert would block forever. A child gets a fresh one on its
        first `prefetch_experts`, and until then `expert()` reads inline.
        """
        owner = self._stream_owner
        if owner is None:
            return None
        if owner[0] != os.getpid():
            self._stream_owner = None
            return None
        return owner[1]

    def expert_stream(self) -> _ExpertStream | None:
        """`_stream`, created on first use. None when prefetching is off."""
        if _EXPERT_PREFETCH_DEPTH < 1:
            return None
        stream = self._stream
        if stream is None:
            stream = _ExpertStream(self, _EXPERT_PREFETCH_DEPTH)
            self._stream_owner = (os.getpid(), stream)
        return stream

    def prefetch_experts(
        self, layer_id: int, expert_ids: Iterable[int], resident: Container = ()
    ) -> None:
        """Announce the routed experts a layer is about to fetch, in order.

        Called from `MoE._route_plan` as soon as the router's output is on the
        host — which is the earliest moment the set is knowable at all — so the
        read of the first expert overlaps the ~50 ms of row-index copies and
        gate arithmetic that still stand between here and the expert loop.

        `resident` is the experts the caller already has on the device and will
        not read; filtering happens here rather than at the call site so that
        with the reader thread off (the default — see
        `_EXPERT_PREFETCH_DEPTH`) this costs one call and one `if`, and not a
        list comprehension over ~48 experts on every MoE layer.
        """
        stream = self.expert_stream()
        if stream is None:
            return
        stream.submit(
            layer_id, [e for e in expert_ids if e not in resident]
        )

    def end_expert_loop(self) -> None:
        """Release the last expert's host buffer. See `_ExpertStream.finish`."""
        stream = self._stream
        if stream is not None:
            stream.finish()

    def expert_keys(self, layer_id: int, expert_id: int) -> dict[str, str]:
        """`{"w1.weight": <checkpoint key>, ...}` for one expert's six tensors."""
        prefix = self._expert_prefix(layer_id, expert_id)
        return {
            f"{name}.{suffix}": f"{prefix}.{name}.{suffix}"
            for name in ("w1", "w2", "w3")
            for suffix in ("weight", "scale")
        }

    def _expert_prefix(self, layer_id: int, expert_id: int) -> str:
        if layer_id < self.config.num_hidden_layers:
            return f"layers.{layer_id}.ffn.experts.{expert_id}"
        stage = layer_id - self.config.num_hidden_layers
        return f"mtp.{stage}.ffn.experts.{expert_id}"

    def n_routed_experts(self, layer_id: int) -> int:
        """384 on the backbone, 128 on a DSpark stage (`get_moe_config`)."""
        if layer_id < self.config.num_hidden_layers:
            return self.config.n_routed_experts
        return self.config.dspark_n_routed_experts

    def num_experts_per_tok(self, layer_id: int) -> int:
        if layer_id < self.config.num_hidden_layers:
            return self.config.num_experts_per_tok
        return self.config.dspark_num_experts_per_tok

    def expert(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
        """One expert's weights, as stored: fp4 nibbles plus `e8m0` scales.

        Returned quantized on purpose. An expert is 3 x 2304 x 5120 values;
        in fp4 that is 17.7 MiB plus 1.1 MiB of scales, and in bf16 it would be
        70.8 MiB. The fp4 GEMM in `kernels/moe_short.py` takes the stored form,
        so dequantizing here would cost 4x the PCIe traffic and 4x the HBM for
        the same arithmetic.

        Served from the reader thread when the caller announced this expert
        through `prefetch_experts`: the bytes are then already in host memory
        and the tensors are views into its ring, valid until the next
        `expert()` call — which happens after `MoE._stage` has copied them to
        the device. The loop below is the unannounced path: the same bytes, read
        now, through the `mmap`.
        """
        stream = self._stream
        if stream is not None:
            served = stream.take(layer_id, expert_id)
            if served is not None:
                return served
        if _EXPERT_READ_COALESCED:
            return self._read_expert(layer_id, expert_id)

        prefix = self._expert_prefix(layer_id, expert_id)
        out: dict[str, torch.Tensor] = {}
        for name in ("w1", "w2", "w3"):
            for suffix in ("weight", "scale"):
                key = f"{prefix}.{name}.{suffix}"
                # Cloned, then the pages are dropped: an expert is read once per
                # forward and never twice, and leaving 276 GiB of them in the
                # page cache starves the Neuron runtime of host DMA memory.
                # See `SafetensorsIndex.release_pages`.
                out[f"{name}.{suffix}"] = as_bytes(self.index.tensor(key).clone())
                self.index.release_pages(key)
        return out

    def _read_expert(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
        """One expert, read with two coalesced `preadv`s into a reusable buffer.

        This is what the loop above does, with the two measured differences that
        make it ~3x cheaper (notes iter-2 §12):

          * **two reads, not six, and no `.clone()`.** `read_plan` coalesces the
            six tensors into the two contiguous spans the checkpoint actually
            stores them in, and `preadv` lands them straight in the destination
            buffer. The `mmap` path page-faults each tensor in and then memcpys
            18.8 MiB out of the mapping, per expert, on the critical path;
          * **two `fadvise`s, not twelve syscalls.** `release_pages` issues a
            `madvise` and a `posix_fadvise` per *tensor*, each over ~4,800 pages
            of a shared mapping; `forget_spans` does the same eviction in two.

        Measured on the device, four ranks, a 512-row prefill forward: the
        `mmap` path is **70.4 ms an expert** of which a clean four-process
        benchmark says only ~25 ms is the volume. 1,695 experts a forward.

        The returned tensors are **views into a buffer reused by the next
        call**, which is safe for the one consumer there is: `MoE._expert`
        copies them to the device through `_stage` before asking for another
        expert, and `_stage`'s host->device step is synchronous. That is the
        same discipline `_ExpertStream` uses, and run 2 of 2026-10-14 validated
        it end to end — the reader ring reused its buffers and produced
        completions character-identical to iteration 1's.
        """
        names = self.expert_keys(layer_id, expert_id)
        nbytes, spans, layout = self.index.read_plan(list(names.values()))
        buffer = self._blob
        if buffer is None or buffer.numel() < nbytes:
            buffer = torch.empty(nbytes, dtype=torch.uint8)
            self._blob = buffer
        self.index.read_spans(spans, buffer)
        self.index.forget_spans(spans)
        out: dict[str, torch.Tensor] = {}
        for short, full in names.items():
            at, count, shape, _dtype = layout[full]
            flat = buffer[at : at + count]
            out[short] = flat.view(shape) if shape else flat
        return out

    def experts(
        self, layer_id: int, expert_ids: Iterable[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        """The experts a step actually routed to, de-duplicated and in order.

        Sorted so the reads walk the shard forward: expert `e+1` follows expert
        `e` in the file, and sequential faulting is what the measured 1.1 GB/s
        assumes.
        """
        unique = sorted({int(expert_id) for expert_id in expert_ids})
        return {expert_id: self.expert(layer_id, expert_id) for expert_id in unique}

    def expert_bytes(self, layer_id: int) -> int:
        """Bytes one expert of this layer occupies, for buffer sizing."""
        prefix = self._expert_prefix(layer_id, 0)
        return sum(
            self.index.ref(f"{prefix}.{name}.{suffix}").nbytes
            for name in ("w1", "w2", "w3")
            for suffix in ("weight", "scale")
        )

    # ── Engram tables ─────────────────────────────────────────────────────

    def engram(self, layer_id: int, hash_ids: torch.Tensor) -> torch.Tensor:
        """Look up n-gram rows and dequantize them to bf16.

        `hash_ids` is `[..., n_hash_cols]` of row indices into this layer's
        table. The result is `[..., n_hash_cols, engram_head_dim]` bf16, which is
        what `Engram.forward` flattens and feeds to `wkv`.

        Dequantized here, unlike the experts: a row is 256 values, the gather is
        already random-access, and the bf16 result is 6 KiB per token per layer.
        """
        if layer_id not in list(self.config.engram_layer_ids):
            raise ValueError(
                f"layer {layer_id} has no engram table; "
                f"engram_layer_ids={list(self.config.engram_layer_ids)}"
            )
        prefix = f"layers.{layer_id}.engram.embed"
        rows = self.index.rows(f"{prefix}.weight", hash_ids)
        scales = self.index.rows(f"{prefix}.scale", hash_ids)
        # Same reason as `expert()`: 94.6 GiB of scattered rows would otherwise
        # accumulate in the page cache across a generation.
        self.index.release_pages(f"{prefix}.weight")
        self.index.release_pages(f"{prefix}.scale")
        return dequant_engram_rows(rows, scales)

    # ── Intermediate state on disk ────────────────────────────────────────

    @property
    def spill_root(self) -> Path:
        """Lazily created, so a run that never spills creates no directory."""
        if self._spill_root is None:
            for candidate in (TIMESHARE_SPILL_ROOT, Path("/tmp/timeshare-spill")):
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                except OSError:
                    continue
                self._spill_root = candidate
                break
            else:
                raise RuntimeError(
                    f"no writable spill directory: tried {TIMESHARE_SPILL_ROOT} "
                    "and /tmp/timeshare-spill"
                )
            logger.info("spilling intermediates to %s", self._spill_root)
        return self._spill_root

    def spill(self, name: str, tensor: torch.Tensor) -> Path:
        """Write an activation out, freeing the HBM it held.

        The tensor is brought to host first: a device tensor cannot be written
        directly, and the copy is the point — after this returns, the device
        memory is the caller's to reuse for the next group's weights.
        """
        path = self.spill_root / f"{name}.pt"
        torch.save(tensor.detach().to("cpu"), path)
        return path

    def unspill(
        self, name: str, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Read an activation back, onto `device` if one is given."""
        path = self.spill_root / f"{name}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no spilled tensor named {name!r} at {path}")
        tensor = torch.load(path, map_location="cpu")
        return tensor if device is None else tensor.to(device)

    def drop_spill(self, name: str) -> None:
        path = self.spill_root / f"{name}.pt"
        path.unlink(missing_ok=True)


# =============================================================================
# Section 5: census — the numbers in this module's docstring, from the index
# =============================================================================


def weight_census(index: SafetensorsIndex, config) -> dict[str, int]:
    """Bytes per family, so the split can be checked instead of believed.

    Deliberately computed from the index rather than from the shapes in
    `config.py`: if the checkpoint ever stops matching what this loader assumes,
    this is where it shows up as a number.
    """
    totals = {"experts": 0, "engram_tables": 0, "resident": 0, "unmapped": 0}
    resident = set(build_weight_mappings_from_config(config))
    streamed = streamed_key_patterns(config)
    expert_keys = set(streamed["experts"])
    engram_keys = set(streamed["engram"])

    for key in index.keys():
        nbytes = index.ref(key).nbytes
        if key in expert_keys:
            totals["experts"] += nbytes
        elif key in engram_keys:
            totals["engram_tables"] += nbytes
        elif key in resident:
            totals["resident"] += nbytes
        else:
            # The vision tower and the image-span embeddings land here: present
            # in the checkpoint, off the tokens -> logits path, never loaded.
            totals["unmapped"] += nbytes
    return totals


def build_weight_mappings_from_config(config) -> dict[str, str | list[str]]:
    """`build_weight_mappings` without a model, for the census and for tests.

    The mapping depends only on the config; taking a model is a convenience for
    the loader's call site, not a requirement.
    """

    class _ConfigOnly:
        pass

    holder = _ConfigOnly()
    holder.config = config  # type: ignore[attr-defined]
    return build_weight_mappings(holder)  # type: ignore[arg-type]


# =============================================================================
# Section 6: the entry point
# =============================================================================


def iter_resident_tensors(
    index: SafetensorsIndex,
    mappings: dict[str, str | list[str]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Walk the resident mapping in shard order, yielding `(name, tensor)`.

    Shard order, not mapping order: the resident weights are spread over all 46
    non-engram shards, and reading them in key order would seek across the
    volume once per layer.
    """
    by_shard: dict[Path, list[tuple[str, str]]] = {}
    for name, keys in mappings.items():
        key = keys if isinstance(keys, str) else keys[0]
        by_shard.setdefault(index.ref(key).shard, []).append((name, key))

    for shard in sorted(by_shard):
        entries = sorted(by_shard[shard], key=lambda item: index.ref(item[1]).begin)
        for name, key in entries:
            yield name, as_bytes(index.tensor(key))


def load_weights(
    model: DeepseekV41ForCausalLM,
    checkpoint_path: str,
    device: torch.device,
    cache_dir: str | None = None,
) -> None:
    """Load the checkpoint into `model`, in place.

    Two halves, matching the split this module's docstring describes:

      1. The resident weights are read once and assigned, sharded per rank by
         the per-parameter transforms `model.py` registered.
      2. A `TimeshareWeightStore` is attached as `model.weight_store`, which is
         how `forward` reaches the routed experts and the engram rows.

    `cache_dir` is accepted for signature compatibility with the other models in
    this repo and is unused: this loader never downloads, per
    `references/INDEX.md` ("do not redownload from huggingface"), and its own
    fast-copy location is `TIMESHARE_CACHE_ROOT`.
    """
    del cache_dir  # see docstring

    root = Path(checkpoint_path)
    index = SafetensorsIndex(root, staged_cache_root(root))
    mappings = build_weight_mappings(model)

    missing = [
        key
        for keys in mappings.values()
        for key in ([keys] if isinstance(keys, str) else keys)
        if key not in index
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} resident key(s) are not in the checkpoint at "
            f"{root}; first few: {missing[:5]}. The mapping and the checkpoint "
            "have diverged — fix build_weight_mappings, do not skip the keys."
        )

    model.weight_store = TimeshareWeightStore(index, model.config)

    problems = model.verify_shard_plan(index)
    if problems:
        raise ValueError(
            f"the sharding plan and the checkpoint disagree in {len(problems)} "
            f"place(s); first few: {problems[:5]}"
        )

    # One pass in shard order. `shard_parameter` narrows to this rank's slice
    # and `.to(device)` copies it off the mmap, so the view dies with the loop
    # iteration and the host never holds more than one tensor at a time.
    state: dict[str, torch.Tensor] = {}
    loaded_bytes = 0
    from .model import _buffer_name

    for key, tensor in iter_resident_tensors(index, mappings):
        shard = model.shard_parameter(key, tensor, device)
        state[_buffer_name(key)] = shard
        loaded_bytes += shard.numel() * shard.element_size()

    incompatible = model.load_state_dict(state, strict=False, assign=True)
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise KeyError(
            f"{len(unexpected)} loaded tensor(s) have no buffer to land in; "
            f"first few: {unexpected[:5]}. _buffer_name and model.py have "
            "diverged."
        )

    # Anything still on `meta` after this would silently become garbage on the
    # first forward, and `model.to(device)` does not fix a meta tensor — it
    # keeps it meta. Catch it here, where the name is still available.
    # Persistent buffers only — the non-persistent ones are runtime state (the
    # KV ring, the compressor's partial group, the RoPE tables) with no
    # checkpoint key, allocated by `bind_kv_cache` after this runs.
    persistent = set(model.state_dict())
    still_meta = [
        name
        for name, buffer in model.named_buffers()
        if buffer is not None and buffer.is_meta and name in persistent
    ]
    if still_meta:
        raise RuntimeError(
            f"{len(still_meta)} buffer(s) were never assigned and are still on "
            f"the meta device; first few: {still_meta[:5]}"
        )

    # The expert staging slots, before the runtime's `model.to(device)`: they
    # have no checkpoint key, so nothing above assigned them, and `Module.to()`
    # refuses to move a meta tensor.
    model.allocate_staging_buffers(device)

    logger.info(
        "loaded %.2f GiB of resident weights for rank %d/%d from %s; %d routed "
        "experts per layer and the two engram tables stay on disk and are "
        "streamed per forward",
        loaded_bytes / 2**30,
        model.rank,
        model.world_size,
        root,
        model.config.n_routed_experts,
    )
