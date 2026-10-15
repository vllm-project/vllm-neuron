# SPDX-License-Identifier: Apache-2.0
"""vLLM Neuron plugin module."""

import glob
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm_neuron import envs as envs

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
    if envs.is_native_backend():
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

    # Keep base vLLM on its V1 code paths. When `use_v2_model_runner` is on,
    # vLLM flips its scheduler, async scheduler, input processor, and
    # max_concurrent_batches onto V2 paths -- e.g. the scheduler stops
    # populating `CachedRequestData.all_token_ids`. Neuron always runs its own
    # v1-style NeuronModelRunner (never vLLM's V2 runner), so those V2 paths
    # break us: `_update_states` reads `all_token_ids[req_id]` on resume and
    # hits a KeyError that kills the engine. The flag also auto-enables for a
    # fixed set of architectures (Llama/Qwen3/Mistral/Qwen2Moe/DeepseekV2/
    # GraniteMoe), so we force it off here -- respecting an explicit user
    # override, which platform._reject_v2_model_runner then rejects loudly.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

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
    - "neuron_libtorch": Use the vLLM Neuron backend (default)
    - "neuron_native": Use the neuron native backend

    If VLLM_NEURON_BACKEND is not set, defaults to vllm_neuron.
    """
    if not _is_cpu_mode() and not _is_cpu_compile() and not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None

    from vllm_neuron.backend import get_platform_class
    from vllm_neuron.vllm.platform import _patch_dcp_config_validation

    _patch_dcp_config_validation()

    return get_platform_class()


# Architectures this plugin implements that the pinned vLLM has no class for.
# The workers register the plugin's models themselves (``NeuronWorker``), but
# the engine process resolves ``ModelConfig.architecture`` *before* any worker
# exists, and an architecture vLLM does not know falls through to its
# Transformers backend there -- whose class the worker then cannot build
# ("TransformersMoEForCausalLM has no attribute from_configs"). Registered as a
# lazy "module:Class" string so nothing model-side is imported here.
#
# Run as a ``vllm.general_plugins`` entry point, not from ``register()``:
# importing ``vllm.ModelRegistry`` while the platform is still being resolved
# pins ``current_platform`` to the unspecified one ("Device string must not be
# empty").
_ARCHITECTURES_UNKNOWN_TO_VLLM = {
    "HYV4ForCausalLM": "vllm_neuron.model.hy_v4:HYV4ForCausalLM",
}


def register_models() -> None:
    from vllm import ModelRegistry

    known = set(ModelRegistry.get_supported_archs())
    for arch, path in _ARCHITECTURES_UNKNOWN_TO_VLLM.items():
        if arch not in known:
            ModelRegistry.register_model(arch, path)


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
