# SPDX-License-Identifier: Apache-2.0
"""File-lock-based contiguous NeuronCore allocator for parallel tests.

Cross-process allocator for NeuronCores on a shared host. Uses ``fcntl.flock``
on a JSON state file so multiple pytest-xdist workers can coordinate without
an external service.

Usage:

    >>> from vllm_neuron.utils.core_allocator import CoreAllocator
    >>> alloc = CoreAllocator(num_cores=64)
    >>> cores = alloc.acquire(8)        # blocks until 8 contiguous cores free
    >>> try:
    ...     ...  # set NEURON_RT_VISIBLE_CORES, run work
    ... finally:
    ...     alloc.release(cores)
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("/tmp/vllm_neuron/core_allocator")
_STATE_FILENAME = "cores.json"
_LOCK_FILENAME = "cores.lock"


class CoreAllocator:
    """Contiguous NeuronCore allocator backed by a file lock.

    Args:
        num_cores: Total cores on the host (e.g. 64 on trn2).
        lock_dir: Directory for lock and state files.
    """

    def __init__(
        self,
        num_cores: int,
        lock_dir: Path | None = None,
    ) -> None:
        lock_dir = Path(lock_dir) if lock_dir is not None else _DEFAULT_DIR
        self._num_cores = num_cores
        self._state_path = lock_dir / _STATE_FILENAME
        self._lock_path = lock_dir / _LOCK_FILENAME
        # Temporarily clear umask so mkdir/open honour the 0o777/0o666 modes.
        old_umask = os.umask(0)
        try:
            lock_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        finally:
            os.umask(old_umask)

    def acquire(
        self, n: int, timeout: float | None = None, poll: float = 0.1
    ) -> list[int]:
        """Acquire ``n`` contiguous, power-of-2-aligned free cores.

        Blocks until available. Stale entries from dead processes are
        automatically cleaned up on each attempt.

        Args:
            n: Number of contiguous cores (must be power of 2).
            timeout: Max seconds to wait, or ``None`` to wait indefinitely.
            poll: Initial seconds between retries (exponential backoff
                with jitter, capped at 5 s).

        Returns:
            Sorted list of core IDs.

        Raises:
            ValueError: If ``n`` exceeds ``num_cores`` or is not a
                power of 2.
            TimeoutError: If ``timeout`` is set and not satisfied in time.

        Example:
            Alignment means the starting core ID is a multiple of ``n``::

                acquire(2)  → blocks starting at 0, 2, 4, 6, 8, ...
                acquire(4)  → blocks starting at 0, 4, 8, 12, ...
                acquire(8)  → blocks starting at 0, 8, 16, 24, ...

            >>> alloc = CoreAllocator(num_cores=64)
            >>> cores = alloc.acquire(4)
            >>> alloc.release(cores)
        """
        if n > self._num_cores:
            raise ValueError(f"Requested {n} cores but host has {self._num_cores}")
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"n must be a power of 2, got {n}")
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        backoff = poll
        while True:
            with self._locked():
                state = self._read()
                self._reap_stale(state)
                block = self._find_contiguous(state, n)
                if block is not None:
                    pid_key = str(os.getpid())
                    existing = state["held"].get(pid_key, [])
                    state["held"][pid_key] = sorted(set(existing) | set(block))
                    self._write(state)
                    logger.debug("Acquired cores %s (pid=%d)", block, os.getpid())
                    return block
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire {n} contiguous cores within {timeout}s"
                )
            sleep = min(backoff, 5.0) * (0.5 + random.random())
            time.sleep(sleep)
            backoff = min(backoff * 2, 5.0)

    def release(self, cores: list[int]) -> None:
        """Release previously acquired cores.

        Args:
            cores: Core IDs returned by ``acquire``.

        Example:
            >>> alloc = CoreAllocator(num_cores=64)
            >>> cores = alloc.acquire(2)
            >>> alloc.release(cores)
        """
        if not cores:
            return
        with self._locked():
            state = self._read()
            pid_key = str(os.getpid())
            if pid_key in state["held"]:
                held_cores = set(state["held"][pid_key])
                unknown = set(cores) - held_cores
                if unknown:
                    logger.warning(
                        "Releasing cores not owned by pid %d: %s",
                        os.getpid(),
                        sorted(unknown),
                    )
                held_cores -= set(cores)
                if held_cores:
                    state["held"][pid_key] = sorted(held_cores)
                else:
                    del state["held"][pid_key]
            else:
                logger.warning(
                    "Releasing cores %s but pid %d has no allocation",
                    cores,
                    os.getpid(),
                )
            self._write(state)
        logger.debug("Released cores %s (pid=%d)", cores, os.getpid())

    def _find_contiguous(self, state: dict, n: int) -> list[int] | None:
        all_held: set[int] = set()
        for pid_cores in state["held"].values():
            all_held.update(pid_cores)
        for start in range(0, self._num_cores - n + 1, n):
            block = set(range(start, start + n))
            if not block & all_held:
                return sorted(block)
        return None

    def _reap_stale(self, state: dict) -> None:
        dead = [pid for pid in state["held"] if not self._pid_alive(int(pid))]
        for pid in dead:
            logger.warning(
                "Reaping stale cores %s from dead pid %s", state["held"][pid], pid
            )
            del state["held"][pid]
        if dead:
            self._write(state)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # alive but owned by another user

    def _read(self) -> dict:
        try:
            data = json.loads(self._state_path.read_text())
            if "held" not in data or not isinstance(data["held"], dict):
                return {"held": {}}
            return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"held": {}}

    def _write(self, state: dict) -> None:
        self._state_path.write_text(json.dumps(state))
        try:
            os.chmod(self._state_path, 0o666)
        except OSError:
            pass

    def _locked(self) -> _FileLock:
        return _FileLock(self._lock_path)


class _FileLock:
    """Blocking fcntl-based exclusive file lock."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self):
        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            os.chmod(self._path, 0o666)
        except OSError:
            pass
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        assert self._fd is not None
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None
