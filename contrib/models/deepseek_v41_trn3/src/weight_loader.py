# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash weight loading — 16 devices, resident
=========================================================

The checkpoint is 475.2 GiB across 48 safetensors shards. A trn2.48xlarge has
1,536 GiB of HBM in 64 banks of 24, so the whole checkpoint fits — but no single
bank holds a 94.6 GiB engram table or a layer's 384 experts, so "it fits" is a
statement about the *partition*, not about any one rank. This loader builds that
partition: every tensor lands in HBM, and which rank's HBM is decided per family
by the axis that divides.

    dense weights   sharded by head/group (8 ways) or by vocab (64), as
                    `model._build_shard_plan` says per key
    routed experts  partitioned by expert id, `n_routed / world_size` a rank,
                    into the slots `MoE.allocate_cache_slots` made
    engram tables   partitioned by row, `ceil(N / world_size)` a rank,
                    dequantized to bf16 on the way in

    routed experts   275.67 GiB  fp4, 384 per layer x 40 layers + 128 x 3 (DSpark)
    engram tables    188.83 GiB  two fp8 tables of 384,006,168 x 256 rows
    resident           9.83 GiB  embedding, LM head, attention, norms, hc, gates
    unmapped           0.90 GiB  the vision tower, off the tokens -> logits path
                     ----------
                     475.24 GiB

  - **Routed experts.** Partitioned by id, and the id axis is 384 (128 in a
    DSpark stage), so 64 ranks own 6 each per layer: **4.3 GiB a rank** against
    275.7 across the model. The partition is disjoint and covers the pool, which
    is what makes the per-rank partial sums add up to the routed set exactly
    once — `MoE.forward`'s all-reduce depends on that and on the gate being
    replicated so every rank selects the same experts.
  - **Engram tables.** Embedding tables, 384,006,168 rows of 256 fp8 values.
    Partitioned by row, so 64 ranks own ~6.0M rows each: **2.86 GiB a rank** per
    table in bf16, 5.72 for both. A lookup is reassembled by masking and
    all-reducing, exactly as the vocab-parallel embedding is.

    dense (a rank)      ~3.9 GiB   embedding, LM head, attention, norms, hc,
                                   gates, shared experts
    routed experts       4.3 GiB   246 of 15,744, by id
    engram rows          5.7 GiB   two shards, bf16
                        ---------
                        ~13.9 GiB  of a 24 GiB bank

    /            (EBS)             67.6 MB/s read, direct — ~2.5 h for that
    /mnt/nvme    (instance store)   2.6 GB/s read, direct — ~4 min
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
    """Drop the pool in a forked child, where its threads do not exist."""
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
    "F8_E8M0": torch.uint8,
}


@dataclass(frozen=True)
class TensorRef:
    """Where one checkpoint tensor's bytes are, without having read them."""

    key: str
    shard: Path
    dtype: torch.dtype
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class SafetensorsIndex:
    """Header-only index over a directory of safetensors shards."""

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
        """Prefer the staged copy of this shard, if there is a whole one."""
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


    def _shard_names(self) -> list[str]:
        index_file = self.root / "model.safetensors.index.json"
        if index_file.is_file():
            weight_map = json.loads(index_file.read_text())["weight_map"]
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
        self._files[shard] = handle
        self._maps[shard] = mapping
        return mapping


    def tensor(self, key: str) -> torch.Tensor:
        """A whole tensor, as a read-only view over the mmap."""
        ref = self.ref(key)
        mapping = self._map(ref.shard)
        buffer = memoryview(mapping)[ref.begin : ref.end]
        flat = torch.frombuffer(buffer, dtype=ref.dtype)
        return flat.view(ref.shape) if ref.shape else flat

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Gather rows `indices` out of a 2-D tensor, reading only their pages.

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
        self._map(ref.shard)
        handle = self._files[ref.shard]
        fileno = handle.fileno()

        def read(part: list[int]) -> list[tuple[int, bytes]]:
            return [
                (row, os.pread(fileno, row_bytes, ref.begin + row * row_bytes))
                for row in part
            ]

        if len(rows) < _ROW_READ_MIN or _ROW_READ_THREADS == 1:
            return dict(read(rows))

        workers = min(_ROW_READ_THREADS, len(rows))
        parts = [rows[i::workers] for i in range(workers)]
        out: dict[int, bytes] = {}
        for got in _row_pool().map(read, parts):
            out.update(got)
        return out

    def release_pages(self, key: str) -> None:
        """Drop one tensor's pages out of the page cache.

            TDRV:dmem_alloc_internal  Failed to allocate HOST memory
            (4194304 bytes): memory allocation failed (ret=-12)
            RuntimeError: The PyTorch Neuron Runtime could not be initialized
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


    def read_plan(
        self, keys: list[str], align: int = 1
    ) -> tuple[int, list[tuple], dict[str, tuple]]:
        """Plan one packed read of `keys`: `(nbytes, spans, layout)`."""
        refs = {key: self.ref(key) for key in keys}
        order = sorted(keys, key=lambda k: (str(refs[k].shard), refs[k].begin))
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
        """A cached `O_DIRECT` descriptor on `shard`, or None if it is refused."""
        fd = self._direct.get(shard)
        if fd is None:
            try:
                fd = os.open(shard, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            except OSError as exc:
                logger.warning(
                    "deepseek_v41: O_DIRECT refused on %s (%s); "
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
        """Fill `buffer` (1-D `uint8`, host) from `spans`, with `preadv`."""
        view = memoryview(buffer.numpy().data).cast("B")
        for shard, begin, count, at, needed in spans:
            fd = self._direct_fd(shard) if direct else None
            if fd is None:
                self._map(shard)
                fd = self._files[shard].fileno()
            got = os.preadv(fd, [view[at : at + count]], begin)
            if got < needed:
                raise OSError(
                    f"short read of {shard.name}: needed {needed} bytes at "
                    f"{begin}, got {got}"
                )

    def forget_spans(self, spans: list[tuple]) -> None:
        """`release_pages` for a planned read: one `fadvise` per span."""
        page = mmap.PAGESIZE
        for shard, begin, count, _at, _needed in spans:
            handle = self._files.get(shard)
            if handle is None:
                continue
            low = (begin // page) * page
            high = -(-(begin + count) // page) * page
            try:
                os.posix_fadvise(
                    handle.fileno(), low, high - low, os.POSIX_FADV_DONTNEED
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
            handle.close()
        self._maps.clear()
        self._files.clear()


def staged_cache_root(checkpoint_path: str | os.PathLike) -> Path | None:
    """The staged copy to prefer, or None if there is nothing usable there.

        /            (EBS)            131-139 MB/s, and it does **not** scale
                                      with parallel readers — 8 concurrent
                                      O_DIRECT readers over 8 shards gave
                                      129 MB/s aggregate
        /mnt/nvme    (instance store) 2.3 GB/s written with O_DIRECT; ~730 MB/s
                                      effective through this module's mmap
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

        python -c "import vllm_neuron.model.deepseek_v41.weight_loader as w; \\
                   w.stage_checkpoint('/mnt/nvme/hf')"

        sudo mkfs.ext4 -F -m 0 -E lazy_itable_init=1,lazy_journal_init=1 /dev/nvme1n1
        sudo mount -o noatime /dev/nvme1n1 /mnt/nvme
        sudo mkdir -p /mnt/nvme/hf && sudo chown -R "$USER" /mnt/nvme
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


#: The dtypes the checkpoint uses for *packed* weights, which this loader hands
#: over as raw `uint8` instead. The NKI kernels in `kernels/blockquant.py` want
#: the bytes, and a `Tensor.view(torch.uint8)` performed inside a traced graph
_PACKED_DTYPES = (torch.float8_e4m3fn, torch.int8)


def as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a packed fp8/fp4 weight as `uint8`; pass anything else on."""
    if tensor.dtype in _PACKED_DTYPES:
        return tensor.view(torch.uint8)
    return tensor


def e8m0_to_scale(raw: torch.Tensor, dtype: torch.dtype = torch.float32):
    """Decode `e8m0` exponent bytes into multipliers: `2 ** (byte - 127)`.

        Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two
        different devices cpu, neuron:0
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
    """fp8 `e4m3` weights with one `e8m0` scale per `block` x `block` tile."""
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
    """fp4 `e2m1` weights, two per byte, one `e8m0` scale per `block` along K."""
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
    """Engram table rows: fp8 `e4m3` with one `e8m0` scale per 32 columns."""
    *lead, width = rows.shape
    blocks = width // block
    values = rows.to(torch.float32).reshape(*lead, blocks, block)
    factors = e8m0_to_scale(scales).reshape(*lead, blocks, 1)
    return (values * factors).reshape(*lead, width).to(torch.bfloat16)


def _attention_keys(prefix: str, config, layer_id: int) -> dict[str, str]:
    """One attention block's resident keys."""
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
        for name in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            keys[f"{prefix}.engram.{name}"] = f"{prefix}.engram.{name}"

    return keys


def build_weight_mappings(model: DeepseekV41ForCausalLM) -> dict[str, str | list[str]]:
    """Map this model's resident parameter names to checkpoint key(s).

      - `layers.<i>.ffn.experts.<e>.{w1,w2,w3}.{weight,scale}` — 384 per layer
        (128 for the DSpark stages), fetched per forward by expert id.
      - `layers.{1,14}.engram.embed.{weight,scale}` — fetched per forward by
        row.
    """
    config = model.config
    plan = model.mk_plan
    mappings: dict[str, str | list[str]] = {}
    # Under the pipeline scheme a rank holds its stage's blocks only: the embedding
    # on the first stage (and the last, whose DSpark stages embed the drafted ids),
    # the norm, the head and the DSpark stages on the last.
    if plan.is_first or plan.is_last:
        mappings["embed.weight"] = "embed.weight"
    if plan.is_last:
        mappings["norm.weight"] = "norm.weight"
        mappings["head.weight"] = "head.weight"

    for layer_id in model.block_ids:
        mappings.update(_block_keys(f"layers.{layer_id}", config, layer_id))

    # The stage's replica of the kv source it receives rows from (`model.src_attn`):
    # the source layer's attention weights, under the replica's module path.
    for sid in model.src_attn:
        src_prefix = f"layers.{sid}.attn"
        for name, key in _attention_keys(src_prefix, config, int(sid)).items():
            mappings[name.replace(src_prefix, f"src_attn.{sid}", 1)] = key

    if not plan.is_last:
        return mappings

    for stage in range(config.num_nextn_predict_layers):
        layer_id = config.num_hidden_layers + stage
        mappings.update(_block_keys(f"mtp.{stage}", config, layer_id))

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
    """The two families `build_weight_mappings` leaves out, spelled as keys."""
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


#: How many routed experts the reader thread may run ahead of the expert loop.
_EXPERT_PREFETCH_DEPTH = max(
    0, int(os.environ.get("DEEPSEEK_V41_EXPERT_PREFETCH", 0))
)

#: Read a routed expert with two coalesced `preadv`s into a reusable buffer
#: (`TimeshareWeightStore._read_expert`) rather than six page-faulted
#: `.clone()`s out of the `mmap` plus twelve `madvise`/`fadvise` calls.
_EXPERT_READ_COALESCED = (
    os.environ.get("DEEPSEEK_V41_EXPERT_READ", "mmap") != "mmap"
)

#: Read routed experts with `O_DIRECT`. **Off, because it was measured and it
#: bought nothing** — kept because the reasoning for trying it is sound and the
#: next iteration should not pay to rediscover that it does not apply here.
_EXPERT_DIRECT_IO = os.environ.get("DEEPSEEK_V41_EXPERT_DIRECT", "0") != "0"

#: `O_DIRECT` wants offset, length and buffer address aligned to the logical
#: block size (512 here). A page is a safe multiple of every plausible value and
#: costs at most 8 KiB of extra read per expert.
_DIRECT_ALIGN = mmap.PAGESIZE


#: `[served, waited, unplanned]` since the last forward boundary: experts the
#: reader thread had ready, of those the ones the consumer still had to wait
#: for, and reads that were never announced and so ran inline. Reported and
EXPERT_STREAM_STATS = [0, 0, 0]


class _ExpertStream:
    """One background thread reading routed experts into reusable host buffers."""

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
                    logger.warning(
                        "deepseek_v41: prefetch of layer %d expert "
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
            self._store.index.forget_spans(spans)
        blob = {}
        for short, full in names.items():
            at, count, shape, _dtype = layout[full]
            flat = buffer[at : at + count]
            blob[short] = flat.view(shape) if shape else flat
        return blob

    def _aligned_buffer(self, index: int, nbytes: int, align: int) -> torch.Tensor:
        """Ring slot `index`, grown if it has to be, aligned if `O_DIRECT` asks."""
        raw = self._buffers[index]
        want = nbytes + align
        if raw is None or raw.numel() < want:
            raw = torch.empty(want, dtype=torch.uint8)
            self._buffers[index] = raw
        if align <= 1:
            return raw
        pad = (-raw.data_ptr()) % align
        return raw[pad : pad + nbytes]


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
        #: Set by `seal()` once every weight is resident. Then `expert()` and
        #: `engram()` raise rather than read: see `seal`.
        self._sealed = False

    def seal(self) -> None:
        """Forbid any further weight read. Called when the model came up resident."""
        self._sealed = True

    def _refuse_if_sealed(self, what: str) -> None:
        if not self._sealed:
            return
        raise RuntimeError(
            f"weight store: refusing to read {what} — every weight this rank "
            "needs is resident in HBM and the store was sealed after loading, "
            "so a read here means the forward path fell back to streaming. "
            "That is a performance regression of ~3 orders of magnitude, not a "
            "correctness one, which is why it is raised rather than logged. "
            "Check MoE.resident and Engram.resident; see "
            "weight_loader.TimeshareWeightStore.seal."
        )


    @property
    def _stream(self) -> _ExpertStream | None:
        """The reader thread of *this* process, or None."""
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
        """Announce the routed experts a layer is about to fetch, in order."""
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
        """One expert's weights, as stored: fp4 nibbles plus `e8m0` scales."""
        self._refuse_if_sealed(f"expert {expert_id} of layer {layer_id}")
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
                out[f"{name}.{suffix}"] = as_bytes(self.index.tensor(key).clone())
                self.index.release_pages(key)
        return out

    def _read_expert(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
        """One expert, read with two coalesced `preadv`s into a reusable buffer.

          * **two reads, not six, and no `.clone()`.** `read_plan` coalesces the
            six tensors into the two contiguous spans the checkpoint actually
            stores them in, and `preadv` lands them straight in the destination
            buffer. The `mmap` path page-faults each tensor in and then memcpys
            18.8 MiB out of the mapping, per expert, on the critical path;
          * **two `fadvise`s, not twelve syscalls.** `release_pages` issues a
            `madvise` and a `posix_fadvise` per *tensor*, each over ~4,800 pages
            of a shared mapping; `forget_spans` does the same eviction in two.
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
        """The experts a step actually routed to, de-duplicated and in order."""
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


    def engram(self, layer_id: int, hash_ids: torch.Tensor) -> torch.Tensor:
        """Look up n-gram rows and dequantize them to bf16."""
        self._refuse_if_sealed(f"engram rows for layer {layer_id}")
        if layer_id not in list(self.config.engram_layer_ids):
            raise ValueError(
                f"layer {layer_id} has no engram table; "
                f"engram_layer_ids={list(self.config.engram_layer_ids)}"
            )
        prefix = f"layers.{layer_id}.engram.embed"
        rows = self.index.rows(f"{prefix}.weight", hash_ids)
        scales = self.index.rows(f"{prefix}.scale", hash_ids)
        self.index.release_pages(f"{prefix}.weight")
        self.index.release_pages(f"{prefix}.scale")
        return dequant_engram_rows(rows, scales)


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
        """Write an activation out, freeing the HBM it held."""
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


def weight_census(index: SafetensorsIndex, config) -> dict[str, int]:
    """Bytes per family, so the split can be checked instead of believed."""
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
            totals["unmapped"] += nbytes
    return totals


def build_weight_mappings_from_config(config) -> dict[str, str | list[str]]:
    """`build_weight_mappings` without a model, for the census and for tests."""

    class _ConfigOnly:
        pass

    holder = _ConfigOnly()
    holder.config = config
    return build_weight_mappings(holder)


def iter_resident_tensors(
    index: SafetensorsIndex,
    mappings: dict[str, str | list[str]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Walk the resident mapping in shard order, yielding `(name, tensor)`."""
    by_shard: dict[Path, list[tuple[str, str]]] = {}
    for name, keys in mappings.items():
        key = keys if isinstance(keys, str) else keys[0]
        by_shard.setdefault(index.ref(key).shard, []).append((name, key))

    for shard in sorted(by_shard):
        entries = sorted(by_shard[shard], key=lambda item: index.ref(item[1]).begin)
        for name, key in entries:
            yield name, as_bytes(index.tensor(key))


#: Rows of an engram table dequantized per step by `load_engram_tables`.
_ENGRAM_LOAD_ROWS = 1 << 16


def load_engram_tables(
    model: DeepseekV41ForCausalLM,
    index: SafetensorsIndex,
    device: torch.device,
) -> int:
    """Put this rank's rows of each n-gram table in HBM. Returns the bytes."""
    loaded = 0
    for layer_id in list(model.config.engram_layer_ids):
        block = model.blocks_by_id.get(layer_id)
        engram = block.engram if block is not None else None
        if engram is None:
            continue
        weight_key = f"layers.{layer_id}.engram.embed.weight"
        scale_key = f"layers.{layer_id}.engram.embed.scale"
        if weight_key not in index or scale_key not in index:
            engram.table = torch.empty(
                0, engram.head_dim, dtype=torch.bfloat16, device=device
            )
            logger.warning(
                "deepseek_v41: layer %d has no engram table in the checkpoint "
                "(%s); falling back to the streamed lookup",
                layer_id, weight_key,
            )
            continue

        total_rows = int(index.ref(weight_key).shape[0])
        width = -(-total_rows // model.world_size)
        start = model.rank * width
        have = max(0, min(width, total_rows - start))

        from .model import _ENGRAM_TABLE_BYTES

        shard_bytes = width * engram.head_dim * 2
        owned_engram = [
            lid for lid in model.config.engram_layer_ids if lid in model.blocks_by_id
        ]
        budget = _ENGRAM_TABLE_BYTES // max(1, len(owned_engram))
        if shard_bytes > budget:
            engram.table = torch.empty(
                0, engram.head_dim, dtype=torch.bfloat16, device=device
            )
            logger.warning(
                "deepseek_v41: engram layer %d shard is %.2f GiB (%d rows of "
                "%d over %d ranks) against a %.2f GiB budget, so this layer "
                "streams its rows off disk instead. Correct, and disk-bound; "
                "widen the world or raise "
                "DEEPSEEK_V41_ENGRAM_TABLE_BYTES.",
                layer_id, shard_bytes / 2**30, width, total_rows,
                model.world_size, budget / 2**30,
            )
            continue

        table = torch.zeros(
            width, engram.head_dim, dtype=torch.bfloat16, device=device
        )
        for lo in range(0, have, _ENGRAM_LOAD_ROWS):
            hi = min(lo + _ENGRAM_LOAD_ROWS, have)
            rows = index.tensor(weight_key)[start + lo : start + hi]
            scales = index.tensor(scale_key)[start + lo : start + hi]
            table[lo:hi].copy_(dequant_engram_rows(rows, scales))
        index.release_pages(weight_key)
        index.release_pages(scale_key)

        engram.bind_table(table, start, width)
        loaded += table.numel() * table.element_size()
        logger.info(
            "deepseek_v41: engram layer %d resident — rows [%d, %d) of %d "
            "(%.2f GiB bf16, padded to %d) for rank %d/%d",
            layer_id, start, start + have, total_rows,
            table.numel() * table.element_size() / 2**30, width,
            model.rank, model.world_size,
        )
    return loaded


def load_resident_experts(
    model: DeepseekV41ForCausalLM, weight_store: "TimeshareWeightStore"
) -> tuple[int, int]:
    """Fill every expert slot this rank owns. Returns `(experts, bytes)`."""
    blocks = list(model.layers.values()) + list(model.mtp)
    per_expert = sum(
        buffer.numel() * buffer.element_size()
        for buffer in blocks[0].ffn.expert_slot.buffers()
    )
    experts = sum(block.ffn.load_resident_experts(weight_store) for block in blocks)
    return experts, experts * per_expert


def load_weights(
    model: DeepseekV41ForCausalLM,
    checkpoint_path: str,
    device: torch.device,
    cache_dir: str | None = None,
) -> None:
    """Load the checkpoint into `model`, in place.

      1. The resident weights are read once and assigned, sharded per rank by
         the per-parameter transforms `model.py` registered.
      2. A `TimeshareWeightStore` is attached as `model.weight_store`, which is
         how `forward` reaches the routed experts and the engram rows.
    """
    del cache_dir

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

    model.allocate_staging_buffers(device)

    experts, expert_bytes = load_resident_experts(model, model.weight_store)
    engram_bytes = load_engram_tables(model, index, device)

    wanted_experts = sum(
        block.ffn.n_local_experts for block in list(model.layers.values()) + list(model.mtp)
    )
    engram_resident = all(
        model.blocks_by_id[layer_id].engram is None
        or model.blocks_by_id[layer_id].engram.resident
        for layer_id in list(model.config.engram_layer_ids)
        if layer_id in model.blocks_by_id
    )
    resident = experts == wanted_experts and engram_resident
    if resident:
        model.weight_store.seal()
        from .model import _RESIDENT_WEIGHTS

        _RESIDENT_WEIGHTS[0] = True
    total = loaded_bytes + expert_bytes + engram_bytes
    logger.info(
        "deepseek_v41: rank %d/%d loaded %.2f GiB from %s — %.2f GiB dense, "
        "%.2f GiB of %d routed experts (%d wanted), %.2f GiB of engram rows. "
        "%s",
        model.rank,
        model.world_size,
        total / 2**30,
        root,
        loaded_bytes / 2**30,
        expert_bytes / 2**30,
        experts,
        wanted_experts,
        engram_bytes / 2**30,
        "Weights are RESIDENT: the forward path reads no file."
        if resident
        else "NOT fully resident — the forward will stream what is missing, "
        "which is correct but disk-bound; see the slot and shard logs above.",
    )
