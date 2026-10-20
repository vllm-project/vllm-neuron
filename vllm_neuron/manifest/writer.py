# SPDX-License-Identifier: Apache-2.0
"""Per-engine compile-artifact manifest orchestration.

:class:`ManifestWriter` owns everything the manifest needs at engine startup:
where this engine's ledger and manifest live (derived from
``VLLM_NEURON_MANIFEST_DIR`` and the device window), arming the compile layer's
ledger writer before graph capture, and — after warmup — emitting the manifest
and clearing the ledger. Every rank arms the ledger; ``local_rank == 0`` emits.

A caller that set ``NEURON_LIBTORCH_COMPILE_LEDGER_DIR`` before the writer was
built keeps that ledger dir: the writer reads it, and leaves it in place. The
writer tells a caller's value from a value that an earlier writer armed, so two
engines in one process keep separate ledgers.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vllm_neuron.manifest.builder import emit_server_manifest

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Set by this writer, read by libtorch_neuronx_lite.compile.manifest_ledger.
_LEDGER_DIR_ENV = "NEURON_LIBTORCH_COMPILE_LEDGER_DIR"

# Ledger dirs that :meth:`ManifestWriter.arm_ledger` set in this process. A
# writer built after another one armed must not read that value as a dir its
# caller chose.
_armed_ledger_dirs: set[str] = set()


class ManifestWriter:
    """Resolves this engine's manifest paths and writes its ``manifest.json``.

    Built via :meth:`from_env`, which returns ``None`` when the feature is off.
    The engine id (device window) is resolved lazily, since it is only per-engine
    after ``init_device`` narrows ``NEURON_VISIBLE_DEVICES``.
    """

    def __init__(self, manifest_root: Path):
        if manifest_root is None:
            raise ValueError("manifest_root is required; use from_env().")
        # Root under which ``ledger/`` and per-engine manifest dirs are written.
        self._root = manifest_root
        # A ledger dir the caller chose before this writer existed, if any. A
        # value that an earlier writer armed is not a caller's choice.
        configured = os.environ.get(_LEDGER_DIR_ENV)
        caller_set = configured and configured not in _armed_ledger_dirs
        self._caller_ledger_dir = Path(configured) if caller_set else None

    @classmethod
    def from_env(cls) -> Optional["ManifestWriter"]:
        """Build from ``VLLM_NEURON_MANIFEST_DIR``, or ``None`` when unset/empty.

        A path enables the feature and is the root for ledgers and manifests;
        unset or empty means the feature is off (there is no default location).
        """
        configured = os.environ.get("VLLM_NEURON_MANIFEST_DIR")
        return cls(Path(configured)) if configured else None

    def _engine_id(self) -> str:
        """Name identifying this engine: its device window from
        ``NEURON_VISIBLE_DEVICES`` (e.g. ``dev8_15``), or ``nodev`` when unset.

        Every rank of an engine resolves the same value, and co-located engines
        resolve different ones, since each holds a disjoint device window.
        """
        from libtorch_neuronx_lite.compile.platform import get_server_prefix

        return get_server_prefix() or "nodev"

    def _ledger_dir(self) -> Path:
        """This engine's compile-ledger directory (``<root>/ledger/<engine-id>``).

        Every rank of the engine writes its per-graph records here; after warmup
        the emitter reads them all to build the manifest. A caller that set
        ``NEURON_LIBTORCH_COMPILE_LEDGER_DIR`` first keeps its own directory, so
        a test can read the same records the manifest was built from.
        """
        if self._caller_ledger_dir is not None:
            return self._caller_ledger_dir
        return self._root / "ledger" / self._engine_id()

    def _out_dir(self) -> Path:
        """Where this engine's ``manifest.json`` is written (``<root>/<engine-id>``)."""
        return self._root / self._engine_id()

    def _compile_cache_root(self) -> Path:
        """The libtorch compile-cache root the manifest's graph hashes live under."""
        import libtorch_neuronx_lite.envs as lite_envs

        return Path(lite_envs.get_neuron_compile_cache_dir())

    def arm_ledger(self) -> None:
        """Point the compile layer's ledger writer at this engine's ledger dir.

        The compile layer appends one record per graph to the directory named by
        ``NEURON_LIBTORCH_COMPILE_LEDGER_DIR``, which this sets from the manifest
        root. Call on **every rank** before graph extraction: each rank's worker,
        and the trace fork children that inherit its environment, then write into
        this engine's directory, which :meth:`emit_and_cleanup` reads afterward.
        A caller that set the variable first keeps its own value.
        Best-effort: a failure here is logged and swallowed, so it can never break
        warmup.
        """
        try:
            ledger_dir = str(self._ledger_dir())
            if self._caller_ledger_dir is None:
                _armed_ledger_dirs.add(ledger_dir)
            else:
                logger.info(
                    "Compile ledger dir %s comes from the environment. This "
                    "writer did not derive it, so per-engine isolation is off "
                    "for it, and this writer does not remove it.",
                    ledger_dir,
                )
            os.environ[_LEDGER_DIR_ENV] = ledger_dir
        except Exception as e:
            logger.warning("manifest ledger arm failed: %s", e)

    def emit_and_cleanup(self, *, vllm_config: "VllmConfig") -> None:
        """Write this engine's manifest, then clear its ledger (both best-effort).

        Call only on the emitter (``local_rank == 0``). Reads the ledger this
        engine wrote under ``ledger/<engine-id>/`` and writes
        ``<engine-id>/manifest.json`` next to it, then removes the ledger subdir.
        Never raises, so it cannot break startup.
        """
        ledger_dir = self._ledger_dir()
        self._emit(vllm_config, ledger_dir=ledger_dir, out_dir=self._out_dir())
        self._cleanup(ledger_dir)

    def _rank_map(self) -> dict[int, int]:
        """Map each TP-local rank (``rank_in_group`` -- what the ledger records) to
        its global rank in the job.

        ``get_tp_group().ranks`` is this engine's ordered list of global ranks,
        indexed by ``rank_in_group``, so ``{i: ranks[i]}`` is the local->global map.
        Best-effort: returns ``{}`` when the group is unavailable (e.g. the
        distributed state isn't initialized), and the builder then leaves the
        ledger's ranks as-is.
        """
        try:
            from vllm.distributed.parallel_state import get_tp_group

            return dict(enumerate(get_tp_group().ranks))
        except Exception as e:
            logger.warning("rank map unavailable: %s", e)
            return {}

    def _emit(
        self, vllm_config: "VllmConfig", *, ledger_dir: Path, out_dir: Path
    ) -> None:
        try:
            emit_server_manifest(
                vllm_config,
                ledger_dir=ledger_dir,
                compile_cache_root=self._compile_cache_root(),
                manifest_dir=out_dir,
                rank_map=self._rank_map(),
            )
        except Exception as e:
            logger.warning("server manifest emission failed: %s", e)

    def _cleanup(self, ledger_dir: Path) -> None:
        """Remove this engine's ``ledger/<engine-id>`` subdir after emitting.

        Only this engine's own subdir is removed -- the shared ``ledger/`` root is
        left in place for any engine still writing to it. A ledger dir outside
        this writer's root belongs to the caller, so it is kept. Never raises.
        """
        if self._root not in ledger_dir.parents:
            return
        try:
            shutil.rmtree(ledger_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("manifest ledger cleanup failed: %s", e)
