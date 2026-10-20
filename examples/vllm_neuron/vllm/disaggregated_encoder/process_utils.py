# SPDX-License-Identifier: Apache-2.0
"""Process and port teardown helpers for the EPD example.

Provides two utilities for cleaning up between engine runs: terminating a
process and its children, and waiting for a server's ports to be released.
Requires ``psutil`` and ``requests``.
"""

import logging
import signal
import time

import psutil
import requests

logger = logging.getLogger(__name__)


def terminate_process_tree(pid: int, name: str = "", timeout: int = 30) -> None:
    """Terminate a process and all its children.

    Sends ``SIGTERM`` to the whole tree first, then ``SIGKILL`` to any that
    survive ``timeout`` seconds.

    Args:
        pid: Process ID of the tree root to terminate.
        name: Human-readable label used in log lines.
        timeout: Seconds to wait for graceful termination before killing.

    Example:
        >>> terminate_process_tree(proc.pid, name="PD-0", timeout=45)
    """
    label = f"{name} (PID: {pid})" if name else f"PID: {pid}"
    try:
        logger.info("Terminating %s", label)
        parent = psutil.Process(pid)
        procs = [parent] + parent.children(recursive=True)

        for p in procs:
            p.send_signal(signal.SIGTERM)

        _, alive = psutil.wait_procs(procs, timeout=timeout)

        for p in alive:
            logger.warning("PID %s did not terminate, sending SIGKILL", p.pid)
            p.send_signal(signal.SIGKILL)

    except psutil.NoSuchProcess:
        logger.debug("Process %s already terminated", pid)
    except Exception as e:
        logger.error("Failed to terminate %s: %s", label, e)


def wait_for_ports_released(
    ports: list[int],
    host: str = "localhost",
    timeout: int = 90,
    poll_interval: int = 3,
) -> None:
    """Block until each port stops answering ``/health``.

    Polls every port's ``/health`` endpoint until it stops responding (server
    fully down) or ``timeout`` elapses. Used between engine teardowns so the
    Neuron cores a server held are freed before the next engine claims them.

    Args:
        ports: Port numbers to wait on.
        host: Host to poll (use the engine's actual IP for remote nodes).
        timeout: Maximum seconds to wait per port.
        poll_interval: Seconds between polls.

    Example:
        >>> wait_for_ports_released([8100], timeout=120)
    """
    for port in ports:
        logger.info("Waiting for %s:%s to be released...", host, port)
        start = time.time()
        deadline = start + timeout
        while time.time() < deadline:
            try:
                requests.get(f"http://{host}:{port}/health", timeout=2)
                time.sleep(poll_interval)
            except (requests.ConnectionError, requests.Timeout, OSError):
                elapsed = int(time.time() - start)
                logger.info("%s:%s released after %ss", host, port, elapsed)
                break
        else:
            logger.warning("%s:%s still responding after %ss", host, port, timeout)
