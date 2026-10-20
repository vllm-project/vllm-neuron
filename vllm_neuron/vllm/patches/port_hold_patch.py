# SPDX-License-Identifier: Apache-2.0
"""
Eliminate TOCTOU port-theft race in distributed initialization.

Between get_open_port() releasing a port and init_process_group() binding it,
NRT from a sibling DI server can steal the port (EADDRINUSE).

Fix: monkey-patch init_process_group for loopback TCP init_methods so that
rank 0 binds immediately (original port or fresh ephemeral), detaches the fd
to TCPStore(master_listen_fd=), and writes the actual port to a rendezvous
file. Ranks 1-N poll that file instead of probing the original port (which
NRT may hold — it accepts TCP but doesn't speak TCPStore, causing hangs).

The file name also holds a per-server key: two servers can request the same
port, and one shared file makes them overwrite each other's entry.

Only activates for tcp://127.0.0.1:<port>. Multi-node, env://, and store=
calls pass through unmodified.
"""

import errno
import os
import socket
import time
from datetime import timedelta
from urllib.parse import urlparse

from vllm.logger import init_logger

logger = init_logger(__name__)

_applied = False
_original_init_process_group = None

_RENDEZVOUS_DIR = "/tmp"
_RENDEZVOUS_PREFIX = "vllm_dist_port_"
_POLL_INTERVAL_S = 0.01
_CLIENT_CONNECT_TIMEOUT_S = 5


def _group_key() -> int:
    """Return an id that is equal for every rank of one server.

    One group can span the engine cores of a server, for example
    tensor_parallel_size=2 with data_parallel_size=2. The parent pid then
    differs inside one group, but the process group id does not: every worker
    inherits it from the server, and each server launch gets its own.

    Example:
        >>> _group_key()  # doctest: +SKIP
        1853
    """
    return os.getpgid(0)


def _rendezvous_path(port: int) -> str:
    """Return the rendezvous file of this server."""
    return os.path.join(_RENDEZVOUS_DIR, f"{_RENDEZVOUS_PREFIX}{port}_{_group_key()}")


def _init_process_group_patched(
    backend=None, init_method=None, store=None, rank=-1, world_size=-1, **kwargs
):
    """Intercept loopback TCP init to handle EADDRINUSE via file rendezvous."""
    if store is not None or not init_method or not init_method.startswith("tcp://"):
        return _original_init_process_group(
            backend=backend,
            init_method=init_method,
            store=store,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )

    parsed = urlparse(init_method)
    host = parsed.hostname
    port = parsed.port

    # Only intercept single-node loopback; multi-node uses real IPs.
    if host != "127.0.0.1" or port is None:
        return _original_init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )

    timeout = kwargs.get("timeout")
    if timeout is None:
        # Match unpatched init_process_group: None means torch's default (30 min
        # for gloo). A short fallback loses to worker spawn stagger at large
        # world sizes.
        from torch.distributed.constants import default_pg_timeout

        timeout = default_pg_timeout

    if rank == 0:
        tcp_store = _rank0_create_store(host, port, world_size, timeout)
    else:
        tcp_store = _rankN_connect_store(host, port, world_size, timeout)

    result = _original_init_process_group(
        backend=backend,
        store=tcp_store,
        rank=rank,
        world_size=world_size,
        **kwargs,
    )

    # init_process_group is a barrier — all ranks have connected by this point.
    # The rendezvous file is no longer needed.
    if rank == 0:
        try:
            os.unlink(_rendezvous_path(port))
        except FileNotFoundError:
            pass

    return result


def _rank0_create_store(host: str, port: int, world_size: int, timeout: timedelta):
    """Bind port (original or fresh), write rendezvous file, return TCPStore."""
    from torch.distributed import TCPStore

    file_path = _rendezvous_path(port)

    # Remove stale file from a previous server that used the same port number
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.listen(128)
        actual_port = port
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            sock.close()
            raise
        sock.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        sock.listen(128)
        actual_port = sock.getsockname()[1]
        logger.warning(
            "EADDRINUSE on port %d — retrying with port %d", port, actual_port
        )

    fd = sock.detach()

    # Write file BEFORE TCPStore — the socket is already in LISTEN state (kernel
    # accepts connections into backlog). TCPStore will process them on construction.
    tmp_path = f"{file_path}.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(str(actual_port))
    os.rename(tmp_path, file_path)

    store = TCPStore(
        host_name=host,
        port=actual_port,
        world_size=world_size,
        is_master=True,
        master_listen_fd=fd,
        use_libuv=False,
    )

    logger.info(
        "Rank 0 TCPStore on port %d (fd=%d, file=%s)", actual_port, fd, file_path
    )
    return store


def _rankN_connect_store(host: str, port: int, world_size: int, timeout: timedelta):
    """Poll rendezvous file for rank 0's actual port, then connect."""
    from torch.distributed import TCPStore

    file_path = _rendezvous_path(port)
    deadline = time.monotonic() + timeout.total_seconds()

    while True:
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
            if content:
                actual_port = int(content)
                if 1 <= actual_port <= 65535:
                    store = TCPStore(
                        host_name=host,
                        port=actual_port,
                        world_size=world_size,
                        is_master=False,
                        timeout=timedelta(seconds=_CLIENT_CONNECT_TIMEOUT_S),
                        use_libuv=False,
                    )
                    return store
        except (FileNotFoundError, ValueError):
            pass
        except Exception as e:
            # TCPStore connect failed (stale file pointing to dead port).
            # Rank 0 will overwrite the file shortly — keep polling.
            logger.debug("Rank N store connect failed, retrying: %s", e)

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Rank N timed out waiting for rendezvous file: {file_path}"
            )
        time.sleep(_POLL_INTERVAL_S)


def apply_port_hold_patch():
    """Patch init_process_group with EADDRINUSE-resilient file rendezvous."""
    global _applied, _original_init_process_group

    if _applied:
        return

    import torch.distributed as dist

    _original_init_process_group = dist.init_process_group
    dist.init_process_group = _init_process_group_patched

    _applied = True
    logger.info("Port-hold patch applied (EADDRINUSE file-rendezvous)")
