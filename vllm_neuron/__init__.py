# SPDX-License-Identifier: Apache-2.0
"""vLLM Neuron plugin module."""

import glob
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm_neuron import envs as envs

# The kernel configuration the shipped GPT-OSS giga-kernel DMA order was derived on. nkilib derived
# it with these flags set, and several of them rename or split the DMAs the order names: the batched
# quant-row transpose produces ``qmx_quant_store`` and ``qmx_scale_store``, and the split gate
# prefetch produces ``gu_w_prefetch_e0_0_h0..h5``. An order applied to a kernel without them names
# ops the graph never emits, and the compiler skips each one with no error.
#
# The flag list comes from the derive invocation in nkilib
# steering_private/gptoss_dma_json_rev15_nki07/RUNBOOK_gk_hostmask.md. Update both together.
#
# Seed these HERE, before the first nkilib import. ``VLLM_NEURON_MEGA_GU_PREFETCH_HSPLIT`` is read
# at module level in nkilib (gate_up_projection_mx.py), so it freezes on import. A write in
# ``init_device`` is already too late: the order then loses the 216 ``gu_w_prefetch`` constraints
# and the bind count drops with no error.
#
# The kernel library owns and reads the VLLM_NEURON_MEGA_* flags at trace time, so they are not
# declared in vllm_neuron/envs.py. ``setdefault`` seeds them, so an explicit value still wins.
_MEGA_DMA_ORDER_KERNEL_FLAGS = {
    "VLLM_NEURON_MEGA_ROUTERW_DMA_ENGINE": "sync",
    "VLLM_NEURON_MEGA_H0_RESID1": "1",
    "VLLM_NEURON_MEGA_QMX_BATCH_XPOSE": "1",
    "VLLM_NEURON_MEGA_GU_RS_FLOWGATE": "1",
    "VLLM_NEURON_MEGA_MOE_OUT_FP8": "1",
    "VLLM_NEURON_MEGA_GU_IOUTER": "2",
    "VLLM_NEURON_MEGA_GU_PREFETCH_HSPLIT": "1",
}
if (envs.VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE or "").strip():
    for _name, _value in _MEGA_DMA_ORDER_KERNEL_FLAGS.items():
        os.environ.setdefault(_name, _value)

# Initialize logging early so VLLM_NEURON_LOG_LEVEL takes effect
from vllm_neuron.logging_config import setup_logging as _setup_logging

_setup_logging()
import logging

logger = logging.getLogger(__name__)
# Enable prometheus multiprocess mode so that metrics observed in vLLM Neuron
# EngineCore/Worker processes (e.g. scheduler padding metrics) are written to
# shared mmap files and visible at the API server's /metrics endpoint.
# Note: Prometheus requires PROMETHEUS_MULTIPROC_DIR to be set before
# prometheus_client is imported anywhere. Typically a user would set the env
# var manually, but we set it during package init to set this up for the user.
# In case prometheus_client is imported before vLLM Neuron, users can
# set PROMETHEUS_MULTIPROC_DIR manually in their env to enable metrics.
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
    import tempfile

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix="vllm_neuron_prometheus_"
    )

# Suppress PyTorch kernel override warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Overriding a previously registered kernel.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*other operators may also be overridden.*"
)


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob("/dev/neuron*")
    return len(neuron_devices) > 0


def _is_cpu_mode() -> bool:
    """Check if VLLM_NEURON_CPU_MODE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


def _is_cpu_compile() -> bool:
    """Check if VLLM_NEURON_CPU_COMPILE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_COMPILE", "0") == "1"


def _init_backend():
    """Initialize the native route's Lite ownership, or the XLA backend."""
    if envs.is_native_backend() and not envs.VLLM_NEURON_CPU_MODE:
        # Import Lite so it names the PrivateUse1 "neuron" device before vLLM
        # builds DeviceConfig.
        import libtorch_neuronx_lite  # noqa: F401
        import torch._dynamo.backends.registry as registry

        # Lite skips its own backend registration during vllm_neuron import,
        # so register the FX-to-HLO capture backend here.
        from libtorch_neuronx_lite.compile.capture_backend import capture

        if "neuron_libtorch_graph_capture" not in registry.list_backends():
            registry.register_backend(
                compiler_fn=capture, name="neuron_libtorch_graph_capture"
            )

        return

    # In CPU mode, the NKI CPU simulator is OFF by default.
    # Users opt-in by setting NKI_SIMULATOR=1 for kernel development,
    # integration and accuracy validation on small shapes and tiny models.
    if envs.VLLM_NEURON_CPU_MODE:
        if os.environ.get("NKI_SIMULATOR") == "1":
            os.environ.setdefault("NKI_PRECISE_FP", "1")

    if envs.VLLM_NEURON_CPU_MODE and _is_cpu_compile():
        raise RuntimeError(
            "VLLM_NEURON_CPU_MODE and VLLM_NEURON_CPU_COMPILE are not compatible with each other"
        )

    os.environ["PJRT_DEVICE"] = "CPU"

    import torch
    import torch._dynamo.backends.registry as registry

    _has_neuron_hw = _is_neuron_dev()

    import libtorch_neuronx_lite

    from libtorch_neuronx_lite.compile.backend import compile
    from libtorch_neuronx_lite.compile.capture_backend import capture

    if "neuron_libtorch" not in registry.list_backends():
        registry.register_backend(compiler_fn=compile, name="neuron_libtorch")

    if "neuron_libtorch_graph_capture" not in registry.list_backends():
        registry.register_backend(
            compiler_fn=capture, name="neuron_libtorch_graph_capture"
        )

    if not envs.VLLM_NEURON_CPU_MODE or _has_neuron_hw:
        try:
            from libtorch_neuronx_lite.overrides import neuron_collectives  # noqa: F401
            from libtorch_neuronx_lite.overrides import xla_collectives  # noqa: F401
        except RuntimeError:
            pass

        sys.modules["torch.neuron"] = libtorch_neuronx_lite
        torch.neuron = libtorch_neuronx_lite
        torch.neuron.is_available = lambda: True
        torch.neuron.current_device = lambda: 0

    else:
        # CPU-only: register minimal privateuse1 backend so dynamo's
        # SymbolicStreamState doesn't fall through to CUDA
        import types

        torch.utils.rename_privateuse1_backend("neuron")
        neuron_module = types.ModuleType("torch.neuron")
        neuron_module.is_available = lambda: False
        neuron_module.current_device = lambda: 0
        neuron_module.current_stream = lambda device=None: type(
            "_S", (), {"device": torch.device("cpu")}
        )()
        neuron_module.set_stream = lambda stream: None
        sys.modules["torch.neuron"] = neuron_module
        torch.neuron = neuron_module

        try:
            torch.utils.generate_methods_for_privateuse1_backend(
                for_tensor=True,
                for_module=True,
                for_storage=True,
            )
        except RuntimeError:
            pass

    _original_current_accelerator = torch.accelerator.current_accelerator

    def _current_accelerator_wrapper(check_available: bool = False):
        device = _original_current_accelerator(check_available)
        if device is not None:
            if envs.VLLM_NEURON_CPU_MODE:
                return torch.device("cpu")
            elif envs.VLLM_NEURON_CPU_COMPILE:
                return torch.device("neuron", torch.neuron.current_device())
            elif device.type == "neuron" and device.index is None:
                return torch.device("neuron", torch.neuron.current_device())
            else:
                return device
        return device

    torch.accelerator.current_accelerator = _current_accelerator_wrapper

    # Patch accelerator stream/device APIs for CPU mode to prevent CUDA fallthrough
    if envs.VLLM_NEURON_CPU_MODE:
        torch.accelerator.current_stream = lambda device=None: type(
            "_S", (), {"device": torch.device("cpu")}
        )()
        torch.accelerator.current_device_index = lambda: 0


def register():
    """Register the Neuron platform if Neuron devices are present, else return None.

    The backend is selected based on the VLLM_NEURON_BACKEND environment variable:
    - "vllm_neuron": Use the XLA route. The route is opt-in and deprecated.
    - "neuron_native": Use the neuron native route.

    If VLLM_NEURON_BACKEND is not set, defaults to neuron_native.
    """
    if not _is_cpu_mode() and not _is_cpu_compile() and not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None

    from vllm_neuron.backend import get_platform_class
    from vllm_neuron.vllm.model_runner_v2_policy import (
        patch_vllm_config_for_neuron,
    )
    from vllm_neuron.vllm.platform import (
        _patch_dcp_config_validation,
        _patch_llm_init_optimization_level,
    )

    _patch_dcp_config_validation()
    _patch_llm_init_optimization_level()
    patch_vllm_config_for_neuron()

    return get_platform_class()


def __getattr__(name):
    import importlib

    if name == "nn":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import-time backend init + patches run LAST, after `register` and
# `__getattr__` are bound. vLLM's plugin discovery
# (vllm.plugins.load_general_plugins) can re-enter this module while these
# imports execute; deferring them until `register` exists avoids a circular
# import crash ("partially initialized module 'vllm_neuron' has no attribute
# 'register'") on multi-worker device runs.
# Patches are applied at import time so they survive spawn-mode re-imports
# (the EngineCore subprocess never calls check_and_update_config).
try:
    _init_backend()
except (ImportError, KeyError):
    pass

from vllm_neuron.vllm.patches.port_hold_patch import apply_port_hold_patch

apply_port_hold_patch()

# vLLM's pooling path pins host memory via a cached PIN_MEMORY constant; force
# it off to match NeuronPlatform.is_pin_memory_available() (privateuse1 has no
# pinned-memory hooks in CPU mode). See the patch module docstring.
from vllm_neuron.vllm.patches.pin_memory_patch import apply_pin_memory_patch

apply_pin_memory_patch()

# vLLM's cleanup_dist_env_and_memory calls torch.accelerator.empty_cache() on
# engine shutdown; torch asserts the accelerator has a caching DeviceAllocator,
# which Neuron's backend does not register. The crash leaks Neuron cores and
# fails every later engine in the pod. See the patch module docstring.
from vllm_neuron.vllm.patches.accelerator_empty_cache_patch import (
    apply_accelerator_empty_cache_patch,
)

apply_accelerator_empty_cache_patch()

# dynamo's recompilation-reason report evaluates the old entry's tensor-aliasing
# sources against the new frame's locals with no error handling; a frame
# re-entered with a module lacking a submodule the earlier entry had raises
# KeyError out of the diagnostic and kills the forward. See the patch module.
from vllm_neuron.vllm.patches.dynamo_recompile_reason_patch import (
    apply_dynamo_recompile_reason_patch,
)

apply_dynamo_recompile_reason_patch()

# Replica-mesh patches are installed by NeuronWorker.init_device() right before
# it calls _init_neuron_distributed_environment_and_runtime(). That location
# runs *after* vLLM has finished importing (avoiding the vllm.utils.torch_utils
# partially-initialized-module circular import at plugin-load time) and *before*
# any process group is created (preserving the "meshes registered before PGs
# exist" invariant the patches rely on). Spawn-mode workers re-enter
# init_device(), so this survives re-imports without needing to run here.
