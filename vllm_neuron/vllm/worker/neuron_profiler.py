# SPDX-License-Identifier: Apache-2.0
"""
NeuronProfiler - Neuron runtime profiling via vLLM's profiler interface.

Hooks into vLLM's /start_profile and /stop_profile endpoints to toggle
Neuron runtime (NRT) profiling on and off at runtime.

Usage:
    Start the server with:
        vllm serve <model> \
          --profiler-config '{"profiler": "cuda"}' \
          --additional-config '{"neuron_profiler": {"output_dir": "/tmp/nrt_profile"}}'

    Then toggle at runtime:
        curl -X POST http://localhost:8000/start_profile
        curl -X POST http://localhost:8000/stop_profile

    Traces (NTFF files) will be saved to the configured output directory.
"""

import os
from typing import Any

import torch
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

from vllm_neuron import envs
import libtorch_neuronx_lite.envs as libtorch_envs

import logging

logger = logging.getLogger(__name__)

_DEFAULT_ACTIVITIES = ["device_profile", "system_profile"]


class NeuronProfilerConfig:
    """Parsed neuron_profiler configuration from --additional-config."""

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        config_dict = config_dict or {}
        self.activities: list[str] = config_dict.get("activities", _DEFAULT_ACTIVITIES)
        self.neuron_cores: list[int] | None = config_dict.get("neuron_cores", None)
        self.output_dir: str = config_dict.get(
            "output_dir", os.path.join(".", "neuron_profiles")
        )
        self.sys_trace_max_events_per_nc: int | None = config_dict.get(
            "sys_trace_max_events_per_nc", None
        )


class NeuronProfiler(WorkerProfiler):
    """Wraps NRT inspect APIs via torch.classes.neuron.Runtime bindings.

    On the Neuron platform, the ``"cuda"`` profiler kind is reinterpreted
    to use Neuron profiling instead of CUDA profiler APIs.

    Args:
        profiler_config: vLLM profiler configuration.
        neuron_profiler_config: Parsed Neuron-specific profiling configuration.
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        neuron_profiler_config: NeuronProfilerConfig,
    ) -> None:
        if envs.VLLM_NEURON_CPU_MODE:
            raise RuntimeError("Neuron profiling is not available in CPU mode.")
        super().__init__(profiler_config)
        self._neuron_profiler_config = neuron_profiler_config
        self._profiling = False

    def _start(self) -> None:
        # NRT inspect APIs copy NEFFs from the cache to the profiler output
        # to make the NEFFs available to Neuron explorer.
        neff_cache_dir = libtorch_envs.get_neff_cache_dir(
            backend=envs.get_compile_backend_name()
        )

        runtime = torch.classes.neuron.Runtime()
        # neuron_cores is not passed here — per-rank gating in NeuronWorker
        # ensures only the configured ranks invoke profiling. Each worker
        # has a single visible NeuronCore, so we profile all visible cores
        # (i.e., the one core this worker owns).
        runtime.start_profiling(
            self._neuron_profiler_config.output_dir,
            self._neuron_profiler_config.activities,
            None,
            self._neuron_profiler_config.sys_trace_max_events_per_nc,
            neff_cache_dir,
        )
        logger.info(
            "Neuron profiling started. Output dir: %s",
            self._neuron_profiler_config.output_dir,
        )
        self._profiling = True

    def _stop(self) -> None:
        if not self._profiling:
            return
        os.makedirs(self._neuron_profiler_config.output_dir, exist_ok=True)

        runtime = torch.classes.neuron.Runtime()
        runtime.stop_profiling()
        logger.info(
            "Neuron profiling stopped. Traces saved to: %s",
            self._neuron_profiler_config.output_dir,
        )
        self._profiling = False
