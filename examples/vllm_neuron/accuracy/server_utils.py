# SPDX-License-Identifier: Apache-2.0
"""Shared vLLM server launcher for the accuracy examples.

The wheel ships no server launcher (the accuracy tools point at an
already-running server). These helpers let an example own a ``vllm serve``
subprocess end-to-end — launch it, wait for ``/health``, and stop it — without
importing anything from the test tree (``test.*``), so the examples remain
runnable straight from the wheel.

Used by ``accuracy_debugger_pipeline.py`` (eval server) and
``run_logit_validation_online.py`` (validation server).
"""

import os
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

_HEALTH_TIMEOUT_S = 1800
_HEALTH_POLL_S = 5


@dataclass
class LocalServer:
    """A vLLM server the example talks to.

    ``stop()`` frees the server's Neuron cores (e.g. before an in-process
    offline LLM runs). It is idempotent and returns ``False`` for an
    operator-managed server the example did not launch (and therefore cannot
    stop).
    """

    base_url: str
    model: str
    stop_hook: Optional[Callable[[], None]] = None

    def stop(self) -> bool:
        if self.stop_hook is None:
            return False
        self.stop_hook()
        self.stop_hook = None
        return True


def free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} before /health"
            )
        try:
            if requests.get(f"{base_url}/health", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(_HEALTH_POLL_S)
    raise TimeoutError(f"Server at {base_url} not healthy after {timeout_s}s")


def get_server(
    model: str,
    build_serve_cmd: Callable[[str, int], str],
    *,
    server_url: Optional[str] = None,
    health_timeout_s: int = _HEALTH_TIMEOUT_S,
) -> LocalServer:
    """Return a :class:`LocalServer`.

    If *server_url* is set, point at that already-running (operator-managed)
    server; its ``stop()`` is a no-op. Otherwise launch ``vllm serve`` in its
    own process group via *build_serve_cmd(model, port)* and wait for
    ``/health`` before returning a handle whose ``stop()`` kills the group.
    """
    os.environ.setdefault("VLLM_RPC_TIMEOUT", "100000")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "neuron")

    if server_url:
        return LocalServer(base_url=server_url, model=model)

    port = free_port()
    base_url = f"http://localhost:{port}"
    cmd = build_serve_cmd(model, port)
    print(f"Launching server:\n    {cmd}\n")
    proc = subprocess.Popen(shlex.split(cmd), preexec_fn=os.setsid)

    def _stop() -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)

    try:
        _wait_for_health(base_url, proc, health_timeout_s)
    except BaseException:
        _stop()
        raise
    return LocalServer(base_url=base_url, model=model, stop_hook=_stop)
