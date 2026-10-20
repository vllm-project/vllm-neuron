# SPDX-License-Identifier: Apache-2.0
"""Build a per-engine compile-artifact manifest (``manifest.json``).

Each engine emits one manifest describing every graph it compiled (HLO / NEFF)
and the config that produced them, so a reader can consume one file instead of
crawling the raw compile-cache directories.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from vllm_neuron.manifest.schema import (
    GraphEntry,
    Manifest,
    RunConfig,
    Topology,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


def _read_text(path: Path) -> Optional[str]:
    """Return file text, or None if missing/unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("manifest: cannot read %s: %s", path, e)
        return None


def parse_output_shapes(hash_dir: Path) -> list[str]:
    """Ordered output shape strings for the compiled graph.

    ``graph.hlo`` is a serialized HLO module; the graph's outputs are its entry
    result shape. A tuple result gives one string per element, a single result
    gives one string, and a scalar gives ``""`` (matching ``input_shapes``).
    Best-effort: returns [] if the HLO is missing or can't be parsed.

    Example:
        >>> parse_output_shapes(Path("cache/06dc.../"))  # doctest: +SKIP
        ['1, 128, 576', '1, 128']
    """
    try:
        from libtorch_neuronx_lite.compile.cache import (
            NEURON_COMPILE_GRAPH_HLO_FILE,
        )
        from libtorch_neuronx_lite.compile.hlo import load_hlo_module

        module = load_hlo_module(str(hash_dir / NEURON_COMPILE_GRAPH_HLO_FILE))
    except Exception as e:
        logger.warning("manifest: no output shapes for graph %s: %s", hash_dir.name, e)
        return []
    result = module.host_program_shape.result
    shapes = list(result.tuple_shapes) or [result]
    return [", ".join(str(d) for d in s.dimensions) for s in shapes]


class _NeffInfo(NamedTuple):
    """A graph's compiled NEFF, from one stat: archive-relative ``path``, ``size``
    in bytes, and ``created_timestamp`` (its mtime, ISO 8601). All ``None`` when no
    NEFF exists."""

    path: Optional[str]
    size: Optional[int]
    created_timestamp: Optional[str]


def _neff_stat(hash_dir: Path, graph_hash: str) -> _NeffInfo:
    """Stat the compiled NEFF once for its path, size, and compile time.

    Returns an :class:`_NeffInfo` — all-``None`` when no NEFF exists (an HLO-only
    directory). ``created_timestamp`` is the NEFF's mtime (when it was compiled).
    """
    try:
        from libtorch_neuronx_lite.compile.cache import get_neff_filename

        neff_name = get_neff_filename(str(hash_dir))
        st = os.stat(hash_dir / neff_name)
    except (OSError, ImportError) as e:
        logger.warning("manifest: no NEFF for graph %s: %s", graph_hash, e)
        return _NeffInfo(None, None, None)
    created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return _NeffInfo(f"{graph_hash}/{neff_name}", st.st_size, created)


def _jsonable_default(obj: Any) -> Any:
    """``json.dumps`` fallback: dict-ify a config object before stringifying.

    The resolved ``neuron_config`` can hold nested config objects (e.g. a
    sampling config). Turn a dataclass or a plain object into its dict so it
    serializes as structured JSON rather than an opaque ``<... object at 0x...>``
    repr; fall back to ``str`` only for values with no usable dict.

    Dataclasses are read one level deep (``fields`` + ``getattr``) rather than via
    ``dataclasses.asdict``: ``asdict`` deep-copies every leaf, which raises on a
    non-copyable value (a lock, socket, or device handle) and would drop the whole
    object. A shallow read never copies -- ``json.dumps`` recurses into the
    referenced values itself, and this fallback ``str``-ifies any leaf it can't
    encode, so an un-copyable field can't break serialization.
    """
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return d
    return str(obj)


def _json_safe(obj: Any) -> Any:
    """Coerce a value to JSON-serializable form.

    Round-trips through ``json`` so the manifest stays serializable and diffable;
    nested config objects become dicts via :func:`_jsonable_default` rather than
    opaque repr strings. Returns None if it still can't be serialized.
    """
    try:
        return json.loads(json.dumps(obj, default=_jsonable_default))
    except Exception as e:
        logger.warning("manifest: dropping value that is not JSON-serializable: %s", e)
        return None


def config_from_vllm_config(vllm_config: "VllmConfig") -> RunConfig:
    """Derive a :class:`RunConfig` from a live ``VllmConfig``.

    Reads parallel sizes, batch/seq shape, dtype, compiler target, the full
    resolved ``neuron_config``, and the resolved ``VllmConfig`` (every
    sub-config, JSON-coerced). Best-effort: a sub-config that is missing or
    unreadable degrades to ``None``.

    Example:
        >>> cfg = config_from_vllm_config(vllm_config)  # doctest: +SKIP
        >>> cfg.topology.tensor_parallel_size, cfg.target  # doctest: +SKIP
        (8, 'trn2')
    """
    pc = getattr(vllm_config, "parallel_config", None)
    mc = getattr(vllm_config, "model_config", None)
    sc = getattr(vllm_config, "scheduler_config", None)
    add = getattr(vllm_config, "additional_config", None)
    nc = add.get("neuron_config", {}) or {} if isinstance(add, dict) else {}

    from libtorch_neuronx_lite.compile.platform import get_platform_target

    target = get_platform_target()

    # Full resolved NeuronConfig; fall back to the raw dict if resolution fails.
    neuron_config: Optional[dict] = dict(nc) if nc else None
    try:
        from dataclasses import asdict

        from vllm_neuron.model.neuron_config import NeuronConfig

        neuron_config = asdict(NeuronConfig.from_dict(nc))
    except Exception as e:
        logger.warning("neuron_config resolution skipped, using raw dict: %s", e)

    tp = getattr(pc, "tensor_parallel_size", None)
    pp = getattr(pc, "pipeline_parallel_size", None)
    dp = getattr(pc, "data_parallel_size", None)
    dcp = getattr(pc, "decode_context_parallel_size", None)
    adp = nc.get("attention_dp_size", 1) if isinstance(nc, dict) else None
    ep = nc.get("ep_degree", 1) if isinstance(nc, dict) else None
    world = getattr(pc, "world_size_across_dp", None)  # TP*PP*DP = total ranks
    # Under EP, a default ep_degree of 1 means experts span the full TP*DP world.
    if (
        getattr(pc, "enable_expert_parallel", False)
        and ep == 1
        and isinstance(world, int)
    ):
        ep = world
    topology = Topology(
        world_size=world if isinstance(world, int) else None,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        attention_dp_size=adp,
        ep_degree=ep,
        decode_context_parallel_size=dcp,
        has_collectives=(world > 1) if isinstance(world, int) else None,
    )
    dtype = getattr(mc, "dtype", None)
    model = getattr(mc, "model", None)
    return RunConfig(
        model=model if isinstance(model, str) else None,
        target=target,
        dtype=str(dtype) if dtype is not None else None,
        max_num_seqs=getattr(sc, "max_num_seqs", None),
        max_model_len=getattr(mc, "max_model_len", None),
        topology=topology,
        neuron_config=_json_safe(neuron_config),
        vllm_config=_json_safe(vllm_config),
    )


def build_graph_entry(
    graph_hash: str,
    cache_root: Path,
    *,
    ranks: list[int],
    input_shapes: Optional[list[str]] = None,
    compiler_flags: Optional[str] = None,
) -> GraphEntry:
    """Assemble one ``graphs[]`` record from a cache-hash directory.

    ``ranks`` are the global ranks that traced this graph. ``input_shapes`` and
    ``compiler_flags`` are recorded inline by the compile layer (carried on the
    ledger record). The NEFF path/size/compile-time come from a single stat of the
    NEFF; ``output_shapes`` is parsed from ``graph.hlo``.

    Example:
        >>> build_graph_entry(  # doctest: +SKIP
        ...     "06dc...", cache_root, ranks=[0]
        ... )
        GraphEntry(graph_hash='06dc...', hlo_path='06dc.../graph.hlo', ...)
    """
    from libtorch_neuronx_lite.compile.cache import NEURON_COMPILE_GRAPH_HLO_FILE

    hash_dir = cache_root / graph_hash
    # One stat on the NEFF gives its path, size, and mtime (compile time).
    neff = _neff_stat(hash_dir, graph_hash)

    return GraphEntry(
        graph_hash=graph_hash,
        hlo_path=f"{graph_hash}/{NEURON_COMPILE_GRAPH_HLO_FILE}",
        neff_path=neff.path,
        compiler_flags=compiler_flags,
        input_shapes=input_shapes or [],
        output_shapes=parse_output_shapes(hash_dir),
        created_timestamp=neff.created_timestamp,
        neff_size=neff.size,
        ranks=ranks,
    )


def _native_neff_stat(cache_key: str) -> _NeffInfo:
    """Stat native's NEFF for ``cache_key``, or all-``None`` if unresolvable.

    Native stores a bare ``<pk>.neff`` in a digest-partitioned tree, so the key
    resolves through Lite's own two-hop lookup rather than a directory name.
    """
    try:
        from libtorch_neuronx_lite.compile.native_backend import (
            _lookup_persistent_key,
            _neff_path,
        )

        # NATIVE_BACKEND_NAME is Lite's own name for the route and is NOT the
        # ledger's "native": the cache API rejects anything else.
        from libtorch_neuronx_lite.envs import (
            NATIVE_BACKEND_NAME,
            get_neff_cache_dir,
        )

        persistent_key = _lookup_persistent_key(cache_key)
        if not persistent_key:
            raise KeyError(f"no key-index entry for {cache_key!r}")
        cache_dir = get_neff_cache_dir(NATIVE_BACKEND_NAME)
        path = _neff_path(cache_dir, persistent_key)
        st = os.stat(path)
    # TypeError is in the tuple on purpose. Both lookups are Lite private
    # symbols, so a changed signature raises it here, and so does a return value
    # that os.stat cannot take. Without it the error reaches build_manifest,
    # which drops the record and writes graphs: [] -- the exact failure this
    # function exists to prevent.
    except (OSError, ImportError, KeyError, AttributeError, TypeError, ValueError) as e:
        logger.warning("manifest: no native NEFF for graph %s: %s", cache_key, e)
        return _NeffInfo(None, None, None)
    created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    # Relative to native's cache root, as the XLA entry is to the compile cache.
    return _NeffInfo(os.path.relpath(path, cache_dir), st.st_size, created)


def build_native_graph_entry(
    cache_key: str,
    *,
    ranks: list[int],
    input_shapes: Optional[list[str]] = None,
    compiler_flags: Optional[str] = None,
) -> GraphEntry:
    """Assemble one ``graphs[]`` record for a graph the NATIVE route compiled.

    ``hlo_path``/``output_shapes`` stay empty: native writes no ``graph.hlo``.
    """
    neff = _native_neff_stat(cache_key)

    return GraphEntry(
        graph_hash=cache_key,
        hlo_path=None,
        neff_path=neff.path,
        compiler_flags=compiler_flags,
        input_shapes=input_shapes or [],
        output_shapes=[],
        created_timestamp=neff.created_timestamp,
        neff_size=neff.size,
        ranks=ranks,
    )


def _read_ledger(ledger_dir: Path) -> list[dict]:
    """Parse every ``*.jsonl`` line in this run's ledger dir into a record list.

    Each compile writer appends one JSON object per graph to its own
    ``<pid>.jsonl``. A single torn or partial line (e.g. from a fork child killed
    mid-write) is skipped so it never discards the rest of the file. Reading this
    run's own dedicated directory is a keyed read, not a compile-cache scan.
    """
    records: list[dict] = []
    for path in sorted(ledger_dir.glob("*.jsonl")):
        text = _read_text(path)
        if text is None:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "manifest: skipping unparsable ledger line in %s: %s", path.name, e
                )
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _group_records(records: list[dict]) -> list[dict]:
    """Group records to one entry per graph ``key``.

    ``ranks`` is the union of every contributor's ledger ``rank`` (the writer's
    TP-local ``rank_in_group``), sorted; :func:`build_manifest` maps these to
    global ranks. ``input_shapes``/``compiler_flags`` carry through (first
    non-empty). Records missing ``key`` are dropped.
    """
    out: dict[str, dict] = {}
    for rec in records:
        key = rec.get("key")
        if not key:
            continue
        g = out.setdefault(
            key,
            {
                "key": key,
                "ranks": set(),
                "input_shapes": rec.get("input_shapes") or [],
                "compiler_flags": rec.get("compiler_flags"),
                "backend": rec.get("backend"),  # build_manifest routes on it
            },
        )
        if isinstance(rec.get("rank"), int):
            g["ranks"].add(rec["rank"])
        if not g["input_shapes"] and rec.get("input_shapes"):
            g["input_shapes"] = rec["input_shapes"]
        if g["compiler_flags"] is None and rec.get("compiler_flags") is not None:
            g["compiler_flags"] = rec["compiler_flags"]
        if g["backend"] is None and rec.get("backend") is not None:
            g["backend"] = rec["backend"]
    # Sort the accumulated ranks for deterministic output.
    for g in out.values():
        g["ranks"] = sorted(g["ranks"])
    return list(out.values())


def build_manifest(
    *,
    records: list[dict],
    compile_cache_root: Optional[Path],
    environment: dict[str, str],
    config: Optional[RunConfig] = None,
    rank_map: Optional[dict[int, int]] = None,
) -> Manifest:
    """Assemble the validated manifest from grouped compile-ledger records.

    Each graph is enriched from its ``<key>/`` folder in the compile cache
    (``graph.hlo`` + the NEFF). A record whose cache dir is missing is skipped,
    and any record that fails to build is skipped with a warning, so one bad
    record never sinks the whole manifest.

    ``rank_map`` maps each ledger rank (the writer's TP-local ``rank_in_group``)
    to its global rank; when omitted the ledger ranks are used as-is.

    Example:
        >>> build_manifest(  # doctest: +SKIP
        ...     records=[{"key": "06dc...", "ranks": [0]}],
        ...     compile_cache_root=cache_root,
        ...     environment={"python": "3.13.7"},
        ...     config=config_from_vllm_config(vllm_config),
        ...     rank_map={0: 0, 1: 1},
        ... )
        ManifestV1(manifest_version='1', ...)
    """
    graphs: list[GraphEntry] = []
    for rec in records:  # already grouped
        key = rec.get("key")
        try:
            if not key:
                raise ValueError("record missing key")
            # Map each writer's TP-local rank to its global rank (identity when
            # no map was provided, e.g. distributed state unavailable).
            ranks = (
                sorted({rank_map.get(r, r) for r in rec["ranks"]})
                if rank_map
                else rec["ranks"]
            )
            # Native's key names no directory, so the probe below would reject
            # every native record and leave `graphs: []` on a normal run.
            if rec.get("backend") == "native":
                graphs.append(
                    build_native_graph_entry(
                        key,
                        ranks=ranks,
                        input_shapes=rec.get("input_shapes"),
                        compiler_flags=rec.get("compiler_flags"),
                    )
                )
                continue
            # Skip a record whose cache dir is missing (stale or never written).
            if (
                compile_cache_root is None
                or not (Path(compile_cache_root) / key).is_dir()
            ):
                raise ValueError(f"cache dir missing for key {key!r}")
            entry = build_graph_entry(
                key,
                compile_cache_root,
                ranks=ranks,
                input_shapes=rec.get("input_shapes"),
                compiler_flags=rec.get("compiler_flags"),
            )
            graphs.append(entry)
        except Exception as e:
            logger.warning("skipping bad graph record (key=%s): %s", key, e)
            continue

    return Manifest(
        graphs=graphs,
        config=config,
        environment=environment,
    )


def collect_environment() -> dict[str, str]:
    """Snapshot installed package versions (like ``pip freeze``), plus Python.

    Records every installed distribution so newly added dependencies are
    captured automatically. Consumers pick out the versions they care about.

    Example:
        >>> collect_environment()  # doctest: +SKIP
        {'python': '3.13.7', 'torch': '2.11.0', 'vllm': '0.24.0', ...}
    """
    env: dict[str, str] = {"python": sys.version.split()[0]}
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            name = dist.name
            if name and name not in env:
                env[name] = dist.version
    except Exception as e:
        logger.warning("environment snapshot incomplete: %s", e)
    return env


def emit_server_manifest(
    vllm_config: "VllmConfig",
    *,
    ledger_dir: Path,
    compile_cache_root: Path,
    manifest_dir: Path,
    rank_map: Optional[dict[int, int]] = None,
) -> Optional[Path]:
    """Read this run's compile ledger and write the engine's ``manifest.json``.

    Reads the per-writer JSONL records the compile wrote into ``ledger_dir``,
    groups them by graph ``key`` (accumulating ``ranks``), maps each writer's
    TP-local rank to its global rank via ``rank_map``,
    enriches each graph from its ``<hash>/`` folder under ``compile_cache_root``,
    joins the live ``vllm_config``, and always writes ``manifest_dir/manifest.json``
    (an empty or absent ledger yields a manifest with ``graphs: []``). The file is
    written atomically (temp file then ``os.replace``). Returns the written path,
    or ``None`` on any failure. Best-effort: never raises, so a caller off the
    serving path is unaffected.

    Example:
        >>> emit_server_manifest(  # doctest: +SKIP
        ...     vllm_config,
        ...     ledger_dir=Path("/artifacts/ledger/dev8_15"),
        ...     compile_cache_root=Path("/root/.cache/neuron_libtorch/neuron/compile_cache"),
        ...     manifest_dir=Path("/artifacts/dev8_15"),
        ...     rank_map={0: 8, 1: 9},
        ... )
        PosixPath('/artifacts/dev8_15/manifest.json')
    """
    try:
        records = _group_records(_read_ledger(ledger_dir))
        manifest = build_manifest(
            records=records,
            compile_cache_root=compile_cache_root,
            environment=collect_environment(),
            config=config_from_vllm_config(vllm_config),
            rank_map=rank_map,
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        out_path = manifest_dir / "manifest.json"
        payload = (
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        )
        # Write to a temp file in the same dir, then atomically replace, so a
        # reader never sees a half-written manifest.
        tmp_path = manifest_dir / f".manifest.json.{os.getpid()}.tmp"
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, out_path)
        return out_path
    except Exception as e:
        logger.warning("Failed to emit server manifest: %s", e)
        return None
