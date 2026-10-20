# SPDX-License-Identifier: Apache-2.0
"""Reusable launcher for a single-node EPD stack.

Brings up a pool of Vision Encoders (VE), a pool of Prefill+Decode engines
(PD), and one Router on disjoint Neuron device slices of a single node, and
yields the Router's OpenAI-compatible base URL.

Example:

    topo = EPDTopology.build(
        num_ve=2, ve_dp=4, num_pd=7, pd_tp=8, model=ckpt,
    )
    with run_epd_single_node(topo, artifacts_dir=out) as stack:
        run_bench(f"vllm bench serve --base-url {stack.base_url} ...")
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

from .process_utils import terminate_process_tree, wait_for_ports_released


_SUBPROC_ENV_OVERRIDES = {
    "FI_EFA_ENABLE_SHM_TRANSFER": "0",
    "FI_EFA_USE_DEVICE_RDMA": "1",
    "TORCH_CACHING_PRECOMPILE": "0",
}

_DEFAULT_COMPILATION_TIMEOUT = "6000"

# The Router entrypoint (server.py) is a sibling of this module.
_ROUTER_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")


@dataclass(frozen=True)
class EngineSpec:
    """One VE or PD engine in the stack.

    Attributes:
        role: "ve" or "pd".
        index: 0-based index within its pool (VE-0, PD-3, …).
        tp: tensor-parallel size (== number of Neuron cores the engine binds).
        dp: encoder data-parallel size (VE only; PD leaves this 1).
        port: HTTP port the engine's vllm serve binds.
        device_slice: NeuronCore ids this engine owns.
        engine_id: NIXL EC engine_id — MUST be distinct per VE.
        side_channel_port: NIXL side-channel base port (VE producer only).
    """

    role: str
    index: int
    tp: int
    dp: int
    port: int
    device_slice: range
    engine_id: str
    side_channel_port: int | None = None

    @property
    def label(self) -> str:
        return f"{self.role}-{self.index}"

    @property
    def is_ve(self) -> bool:
        return self.role == "ve"


@dataclass(frozen=True)
class EPDTopology:
    """A single-node EPD topology: VE pool + PD pool + Router.

    Construct via build(), which derives disjoint device slices, ports,
    and distinct engine ids from the counts. Callers that need an exotic layout
    can build the ve / pd lists directly.
    """

    model: str
    ve: list[EngineSpec]
    pd: list[EngineSpec]
    router_port: int
    max_model_len: int
    max_num_seqs: int
    encoder_seq_len: int
    num_images: int
    quantization: str = "bf16"
    # MXFP8 only: modules to leave in bf16 (e.g. ["mlp"] at TP64). Ignored for bf16.
    modules_to_not_convert: list[str] = field(default_factory=list)
    label: str | None = None
    logit_validation: bool = False
    limit_mm_per_prompt: dict[str, int] = field(default_factory=dict)
    neuron_config: dict[str, Any] | None = None
    encoder_neuron_config: dict[str, Any] | None = None
    # VE-only overlay. build_serve_cmd applies it after encoder_neuron_config,
    # so it goes only into the command of a VE engine. Put the encoder-cache
    # sizing here: only a VE keeps an encoded item until a remote PD pulls it,
    # so only a VE needs a cache that is larger than one prefill batch.
    ve_encoder_neuron_config: dict[str, Any] | None = None
    # Extra Router-renderer kwargs keyed by CLI-flag name, e.g.
    # {"media_io_kwargs": {"video": {"num_frames": 60}}, "mm_processor_kwargs": {...}}.
    router_kwargs: dict[str, Any] | None = None

    @property
    def num_ve(self) -> int:
        return len(self.ve)

    @property
    def num_pd(self) -> int:
        return len(self.pd)

    @property
    def total_cores(self) -> int:
        return sum(e.tp for e in self.ve) + sum(e.tp for e in self.pd)

    @property
    def topology_label(self) -> str:
        # e.g. "2ve4_7pd8" — VE(dp) + PD(tp), matching the perf-test label style.
        ve0 = self.ve[0] if self.ve else None
        pd0 = self.pd[0] if self.pd else None
        ve_part = f"{self.num_ve}ve{ve0.dp}" if ve0 else "0ve"
        pd_part = f"{self.num_pd}pd{pd0.tp}" if pd0 else "0pd"
        return f"{ve_part}_{pd_part}"

    @classmethod
    def build(
        cls,
        *,
        model: str,
        num_ve: int,
        ve_dp: int,
        num_pd: int,
        pd_tp: int,
        max_model_len: int,
        max_num_seqs: int,
        encoder_seq_len: int,
        num_images: int,
        ve_tp: int | None = None,
        quantization: str = "bf16",
        modules_to_not_convert: list[str] | None = None,
        label: str | None = None,
        logit_validation: bool = False,
        limit_mm_per_prompt: dict[str, int] | None = None,
        neuron_config: dict[str, Any] | None = None,
        encoder_neuron_config: dict[str, Any] | None = None,
        ve_encoder_neuron_config: dict[str, Any] | None = None,
        router_kwargs: dict[str, Any] | None = None,
        base_device: int = 0,
        ve_port_base: int = 18300,
        pd_port_base: int = 18100,
        router_port: int = 18800,
        ve_sidechannel_base: int = 14600,
        engine_id_prefix: str = "epd",
    ) -> "EPDTopology":
        """Derive a topology from pool counts.

        Device layout: the VE pool takes the first num_ve * ve_tp cores
        starting at base_device; the PD pool takes the next
        num_pd * pd_tp cores. Caller keeps
        num_ve*ve_tp + num_pd*pd_tp <= node core count

        ve_tp is the VE's vllm-level tensor-parallel size (== cores per VE;
        the worker asserts TP == len(NEURON_VISIBLE_DEVICES)). ve_dp is
        the encoder data-parallel degree inside encoder_neuron_config.
        """
        ve_tp = ve_tp if ve_tp is not None else ve_dp
        ve: list[EngineSpec] = []
        for i in range(num_ve):
            start = base_device + i * ve_tp
            ve.append(
                EngineSpec(
                    role="ve",
                    index=i,
                    tp=ve_tp,
                    dp=ve_dp,
                    port=ve_port_base + i,
                    device_slice=range(start, start + ve_tp),
                    engine_id=f"{engine_id_prefix}-ve-{i}",
                    side_channel_port=ve_sidechannel_base + i * ve_tp,
                )
            )

        pd_offset = base_device + num_ve * ve_tp
        pd: list[EngineSpec] = []
        for j in range(num_pd):
            start = pd_offset + j * pd_tp
            pd.append(
                EngineSpec(
                    role="pd",
                    index=j,
                    tp=pd_tp,
                    dp=1,
                    port=pd_port_base + j,
                    device_slice=range(start, start + pd_tp),
                    engine_id=f"{engine_id_prefix}-pd-{j}",
                )
            )

        return cls(
            model=model,
            ve=ve,
            pd=pd,
            router_port=router_port,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            encoder_seq_len=encoder_seq_len,
            num_images=num_images,
            quantization=quantization,
            modules_to_not_convert=modules_to_not_convert or [],
            label=label,
            logit_validation=logit_validation,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": num_images},
            neuron_config=neuron_config,
            encoder_neuron_config=encoder_neuron_config,
            ve_encoder_neuron_config=ve_encoder_neuron_config,
            router_kwargs=router_kwargs,
        )


def build_serve_cmd(topo: EPDTopology, engine: EngineSpec) -> str:
    """Build the vllm serve command for one VE or PD engine."""
    neuron_cfg: dict = {
        "num_batched_tokens_buckets": [topo.max_model_len],
        "on_device_sampling_config": {"all_greedy": True},
    }
    encoder_cfg: dict = {
        "num_encoder_input_buckets": [topo.encoder_seq_len],
        "encoder_attention_block_size": 1024,
        "encoder_cache_num_blocks": 128,
    }
    neuron_cfg.update(topo.neuron_config or {})
    encoder_cfg.update(topo.encoder_neuron_config or {})
    neuron_cfg["quantization"] = topo.quantization
    encoder_cfg["dp_size"] = engine.dp
    # MXFP8: which modules stay bf16 (mirrors the monolith mxfp8 perf test's
    # modules_to_not_convert). Empty list = full-transformer MXFP8.
    if topo.quantization == "mxfp8":
        neuron_cfg["modules_to_not_convert"] = list(topo.modules_to_not_convert)
    if topo.logit_validation:
        neuron_cfg["max_logprobs"] = -1
    ac: dict = {"neuron_config": neuron_cfg, "encoder_neuron_config": encoder_cfg}

    if engine.is_ve:
        neuron_cfg["num_seqs_buckets"] = [1]
        # ac holds a reference to encoder_cfg, so this update reaches the command.
        encoder_cfg.update(topo.ve_encoder_neuron_config or {})
        max_num_seqs = 1
    else:
        ac["mm_language_model_only"] = True
        max_num_seqs = topo.max_num_seqs

    parts = [
        "vllm",
        "serve",
        topo.model,
        "--tensor-parallel-size",
        str(engine.tp),
        "--host",
        "127.0.0.1",
        "--port",
        str(engine.port),
        "--max-model-len",
        str(topo.max_model_len),
        "--max-num-batched-tokens",
        str(topo.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--dtype",
        "bfloat16",
        "--limit-mm-per-prompt",
        json.dumps(topo.limit_mm_per_prompt),
        # Prefix caching off on PD: Prefix cache hits are not guaranteed as PD
        # nodes are scheduled in round robin fashion.
        "--no-enable-prefix-caching",
    ]
    if topo.logit_validation:
        # Full-vocab per-token logits for teacher-forced validation.
        parts += ["--max-logprobs", "-1", "--logprobs-mode", "raw_logits"]
    if engine.is_ve:
        parts += ["--mm-encoder-only"]
        if not topo.logit_validation:
            # Enable async scheduling on the VE. The VE then defers the encoder
            # cache drain to get_output(). The next request CPU work (block
            # packing and host-to-device copy) overlaps the current encoder
            # NEFF execution.
            parts += ["--async-scheduling"]
    else:
        parts += ["--mm-processor-cache-gb", "0"]
    if topo.logit_validation:
        parts += ["--no-async-scheduling"]
    parts += ["--additional-config", json.dumps(ac)]

    ec_extra: dict = {"backends": ["LIBFABRIC"]}
    if engine.side_channel_port is not None:
        ec_extra["side_channel_host"] = "127.0.0.1"
        ec_extra["side_channel_port"] = engine.side_channel_port
    parts += [
        "--ec-transfer-config",
        json.dumps(
            {
                "ec_connector": "NeuronNixlECConnector",
                "ec_role": "ec_producer" if engine.is_ve else "ec_consumer",
                "engine_id": engine.engine_id,
                "ec_connector_extra_config": ec_extra,
            }
        ),
    ]
    return " ".join(shlex.quote(p) if any(c in p for c in " \"'") else p for p in parts)


@dataclass
class EPDStack:
    """A live EPD stack. base_url is the Router's OpenAI-compatible URL."""

    base_url: str
    topology: EPDTopology
    artifacts_dir: str
    router_server_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _engine_env(engine: EngineSpec) -> dict:
    env = os.environ.copy()
    env.update(_SUBPROC_ENV_OVERRIDES)
    env.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", _DEFAULT_COMPILATION_TIMEOUT)
    env["NEURON_VISIBLE_DEVICES"] = ",".join(str(d) for d in engine.device_slice)
    # Give this engine its own PROMETHEUS_MULTIPROC_DIR. vllm_neuron/__init__.py
    # sets the variable when the launcher imports the package, so every engine
    # would otherwise inherit ONE directory. prometheus_client then merges the
    # mmap files of every engine, and each /metrics endpoint answers with the
    # sum over the whole stack. A per-engine directory keeps each endpoint to its
    # own engine, so a reader can attribute a metric to the VE or to the PD.
    # test/utils/simple_server.py applies the same override for the same reason.
    env["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix=f"epd_{engine.label}_prom_"
    )
    # Cap per-engine host threads. OpenMP defaults its pool to the physical core
    # count while the cgroup grants only a few CPUs, so co-locating many engines
    # (2 VE + N PD) on one node oversubscribes the CPU and thrashes host-side stages.
    # Pinning OMP/MKL to 4 threads per engine matches the quota and is a large TTFT
    # win on the workload (e.g. trn3 C32 image 10.7s -> 3.4s).
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    return env


def _read_log_tail(log_path: str, max_lines: int = 100) -> str:
    try:
        from collections import deque

        with open(log_path) as f:
            tail = deque(f, maxlen=max_lines)
        if not tail:
            return ""
        return f"--- {log_path} (last {len(tail)} lines) ---\n{''.join(tail)}"
    except Exception:
        return f"(could not read log: {log_path})"


def _wait_ready(
    url: str,
    timeout: float,
    label: str,
    *,
    proc: subprocess.Popen | None = None,
    log_path: str | None = None,
) -> None:
    """Poll url until it returns 200, failing fast if the process dies."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _read_log_tail(log_path) if log_path else ""
            raise RuntimeError(
                f"{label} exited with code {proc.returncode} before becoming "
                f"ready at {url}." + (f"\n{tail}" if tail else "")
            )
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    tail = _read_log_tail(log_path) if log_path else ""
    raise RuntimeError(
        f"{label} did not become ready at {url} after {timeout:.0f}s."
        + (f"\n{tail}" if tail else "")
    )


@contextmanager
def _engine_proc(cmd: str, *, env: dict, log_path: str, label: str, port: int):
    """Launch one engine subprocess; on exit tear down its whole process tree
    and block until its port is released, so the Neuron cores it held are freed
    before anything else claims them.

    The launch is non-blocking (Popen returns immediately); callers health-check
    the proc separately so a pool of engines can compile concurrently. On
    context exit the whole tree is killed and the port drained.
    """
    with open(log_path, "w") as log_f:
        print(f"    [{label}] launch → {log_path}", flush=True)
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            yield proc
        finally:
            terminate_process_tree(proc.pid, name=label, timeout=45)
            wait_for_ports_released([port], timeout=120)


def _wait_all_ready(
    engines: "list[tuple[str, subprocess.Popen, str, str]]",
    timeout: float,
) -> None:
    """Health-check a pool of already-launched engines concurrently."""
    import concurrent.futures

    def _one(item):
        url, proc, label, log_path = item
        _wait_ready(url, timeout, label, proc=proc, log_path=log_path)
        print(f"    [{label}] healthy", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as ex:
        # list() forces all futures to complete; the first exception propagates.
        list(ex.map(_one, engines))


def _router_module_path() -> str:
    assert os.path.exists(_ROUTER_MODULE), (
        f"Router entrypoint not found: {_ROUTER_MODULE}"
    )
    return _ROUTER_MODULE


def _router_cmd(topo: EPDTopology, metrics_file: str | None = None) -> str:
    module = _router_module_path()
    parts = [
        sys.executable,
        "-u",
        module,
        "--model",
        topo.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(topo.router_port),
        "--ve-timeout-s",
        "600",
        "--pd-timeout-s",
        "600",
    ]
    if topo.quantization and topo.quantization != "bf16":
        parts += ["--quantization", topo.quantization]
    # Router-renderer kwargs by CLI-flag name (media_io_kwargs, mm_processor_kwargs).
    router_kwargs = topo.router_kwargs or {}
    for key, flag in (
        ("media_io_kwargs", "--media-io-kwargs"),
        ("mm_processor_kwargs", "--mm-processor-kwargs"),
    ):
        if router_kwargs.get(key):
            parts += [flag, json.dumps(router_kwargs[key])]
    if metrics_file:
        parts += ["--metrics-file", metrics_file]
    for e in topo.ve:
        parts += ["--ve-endpoint", f"127.0.0.1:{e.port}"]
    for e in topo.pd:
        parts += ["--pd-endpoint", f"127.0.0.1:{e.port}"]
    return " ".join(shlex.quote(p) if any(c in p for c in " \"'") else p for p in parts)


def write_router_server_config(stack: EPDStack) -> None:
    """Drop a server_config.json for the Router into artifacts_dir."""
    topo = stack.topology
    dest = Path(stack.artifacts_dir)
    dest.mkdir(parents=True, exist_ok=True)
    config = {
        "server_id": stack.router_server_id,
        "label": topo.label or topo.topology_label,
        "model": topo.model,
        "concurrency": topo.max_num_seqs,
        "max_model_len": topo.max_model_len,
        "tensor_parallel_size": topo.pd[0].tp if topo.pd else None,
        "epd_num_ve": topo.num_ve,
        "epd_num_pd": topo.num_pd,
        "epd_ve_dp": topo.ve[0].dp if topo.ve else None,
        "epd_total_cores": topo.total_cores,
    }
    (dest / "server_config.json").write_text(json.dumps(config, indent=2) + "\n")


@contextmanager
def run_epd_single_node(
    topo: EPDTopology,
    *,
    artifacts_dir: str,
    engine_ready_timeout_s: int = 1800,
    router_ready_timeout_s: int = 300,
    router_env: dict | None = None,
    startup_delay_s: float = 10.0,
) -> Iterator[EPDStack]:
    """Bring up the whole VE + PD + Router stack; yield a live EPDStack.

    All VE + PD engines are launched UP FRONT (staggered by startup_delay_s
    to avoid a device-slice binding race), then health-checked concurrently.
    On exit every process tree is torn down and its cores released.

    Args:
        topo: the topology to launch.
        artifacts_dir: root dir for per-engine logs, the Router's
            server_config.json, and (written by the caller) the bench
            performance/ output.
        engine_ready_timeout_s: per-engine health-check timeout (cold compile
            of a 32B engine can take >10 min).
        router_ready_timeout_s: Router /healthcheck timeout.
        router_env: optional extra env for the Router process (merged over a
            copy of os.environ).
        startup_delay_s: stagger between engine Popens. A brief gap avoids a
            thundering-herd race on NEURON_VISIBLE_DEVICES binding / the
            shared compile cache.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    for e in topo.ve + topo.pd:
        os.makedirs(os.path.join(artifacts_dir, e.label), exist_ok=True)

    with ExitStack() as stack:
        pending: list[tuple[str, subprocess.Popen, str, str]] = []
        engines = topo.ve + topo.pd
        for i, engine in enumerate(engines):
            cmd = build_serve_cmd(topo, engine)
            log_path = os.path.join(artifacts_dir, engine.label, "server.log")
            proc = stack.enter_context(
                _engine_proc(
                    cmd,
                    env=_engine_env(engine),
                    log_path=log_path,
                    label=engine.label,
                    port=engine.port,
                )
            )
            pending.append(
                (
                    f"http://127.0.0.1:{engine.port}/health",
                    proc,
                    engine.label,
                    log_path,
                )
            )
            if i < len(engines) - 1:
                time.sleep(startup_delay_s)

        _wait_all_ready(pending, engine_ready_timeout_s)

        r_env = os.environ.copy()
        # Keep the Router out of the engines' metric directories too. See
        # _engine_env for the reason.
        r_env["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="epd_router_prom_")
        if topo.logit_validation:
            r_env["VLLM_NEURON_DEBUG_MODE"] = "1"
        if router_env:
            r_env.update(router_env)
        router_log = os.path.join(artifacts_dir, "router.log")
        # Per-request phase records land beside router.log, so the artifact
        # upload carries them with no extra plumbing.
        router_metrics = os.path.join(artifacts_dir, "router_metrics.jsonl")
        router_proc = stack.enter_context(
            _engine_proc(
                _router_cmd(topo, router_metrics),
                env=r_env,
                log_path=router_log,
                label="router",
                port=topo.router_port,
            )
        )
        base_url = f"http://127.0.0.1:{topo.router_port}"
        _wait_ready(
            f"{base_url}/healthcheck",
            router_ready_timeout_s,
            "router",
            proc=router_proc,
            log_path=router_log,
        )

        epd_stack = EPDStack(
            base_url=base_url, topology=topo, artifacts_dir=artifacts_dir
        )
        write_router_server_config(epd_stack)
        yield epd_stack
