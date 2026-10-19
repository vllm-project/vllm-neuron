# SPDX-License-Identifier: Apache-2.0
"""Define the Neuron model-runner selection and validation policy."""

from __future__ import annotations

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
import sys
from typing import TYPE_CHECKING, NamedTuple

import vllm.envs as vllm_envs

from vllm_neuron import envs as neuron_envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Largest in-flight batch count the runner is known to survive. See
# _apply_max_concurrent_batches. Depth 6 corrupts ``input_ids``, so a request above this is
# refused rather than served. Do not raise it without re-testing generated output.
NEURON_MAX_BATCH_QUEUE_DEPTH = 4

# Smallest in-flight batch count that still gives a batch queue at all. ``EngineCore`` builds
# the queue only above 1, so depth 1 leaves async scheduling on while every step runs serially,
# which is worse than the upstream default. Refuse it rather than serve it.
NEURON_MIN_BATCH_QUEUE_DEPTH = 2


# Model architectures the V2 runner cannot serve, so they fall back to V1. This is a
# denylist: any "generate" model runs on V2 by default unless its architecture name is
# listed here. Names match the registry keys in vllm_neuron/model/registry.py.
#
# The synthetic test models run their forward eagerly on device (no compiled graph),
# and the V2 forward adapter's device-state update is compiled-only, so they crash on
# V2 at the first step. They do not set enforce_eager, so the eager check below
# cannot catch them.
UNSUPPORTED_V2_NEURON_RUNNER_ARCHITECTURES = frozenset(
    {"SyntheticNeuronModel", "SyntheticStackedNeuronModel"}
)

# Architectures whose V2 debug-logits emission path has been validated
# (adapter returns gathered logits and the writer's row selection is correct).
_DEBUG_LOGITS_V2_ARCHITECTURES = frozenset({"LlamaForCausalLM", "GptOssForCausalLM"})

# Neuron component data-parallel fields. These shard attention/embedding/MLP/LM-head work across cores
# and are NOT covered by vLLM's data_parallel_size, so a DI decode server can set them with
# data_parallel_size == 1. The V2 first cell is pure TP, so any of these > 1 routes to V1.
_COMPONENT_DP_FIELDS = (
    "attention_dp_size",
    "embedding_dp_size",
    "mlp_dp_size",
    "lm_head_dp_size",
)

V2_COMPATIBLE_SCHEDULER_CLASSES = frozenset(
    {
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_neuron.vllm.core.scheduler.NeuronScheduler",
        "vllm_neuron.vllm.core.scheduler.NeuronAsyncScheduler",
    }
)


def _model_architectures(model_config: object) -> list[str]:
    """Return the model architecture names.

    Read ``model_config.architectures`` and fall back to
    ``model_config.hf_config.architectures``. Accept only a nonempty list or
    tuple so partially populated configurations do not raise an error.

    Args:
        model_config: Model configuration to inspect.

    Returns:
        Architecture names, or an empty list when none are available.
    """
    for source in (model_config, getattr(model_config, "hf_config", None)):
        archs = getattr(source, "architectures", None)
        if isinstance(archs, (list, tuple)) and archs:
            return [str(arch) for arch in archs]
    return []


class NeuronModelRunnerPolicy(NamedTuple):
    """Describe whether the runner is enabled and why it may be unsupported.

    Attributes:
        default_enabled: Whether the configuration is eligible by default.
        unsupported_features: Configuration features the runner cannot serve.
    """

    default_enabled: bool
    unsupported_features: tuple[str, ...] = ()


def _parallel_degree(config: object, name: str) -> int:
    """Read a parallelism degree from a configuration object.

    Treat an absent or noninteger field as degree one so partially populated
    configurations represent the feature as disabled.

    Args:
        config: Configuration object to inspect.
        name: Attribute that stores the parallelism degree.

    Returns:
        The integer degree, or ``1`` when the field is unavailable or invalid.
    """
    value = getattr(config, name, 1)
    return value if isinstance(value, int) else 1


def _has_component_dp(neuron_cfg: dict) -> bool:
    """Return whether any Neuron component data-parallel degree exceeds one.

    Component DP shards attention/embedding/MLP/LM-head work across cores and is
    not represented by vLLM's ``data_parallel_size``, so it must be checked from
    the Neuron configuration directly.

    Args:
        neuron_cfg: The ``additional_config["neuron_config"]`` dictionary.

    Returns:
        ``True`` when a component-DP degree is a valid integer greater than one.
    """
    for name in _COMPONENT_DP_FIELDS:
        value = neuron_cfg.get(name, 1)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            return True
    return False


def _scheduler_class_path(scheduler_cls: object) -> str | None:
    """Return the import path for a scheduler class or configured path."""
    if scheduler_cls is None:
        return None
    if isinstance(scheduler_cls, str):
        return scheduler_cls
    module = getattr(scheduler_cls, "__module__", None)
    qualname = getattr(scheduler_cls, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return None


def get_neuron_model_runner_v2_policy(
    vllm_config: "VllmConfig",
) -> NeuronModelRunnerPolicy:
    """Build the Neuron model-runner policy for a configuration.

    Args:
        vllm_config: Complete vLLM configuration to evaluate.

    Returns:
        The default eligibility decision and unsupported feature names.
    """
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    unsupported: list[str] = []

    dense_generate = model_config is not None
    if model_config is None:
        unsupported.append("missing model configuration")
    else:
        if getattr(model_config, "runner_type", None) != "generate":
            unsupported.append("non-generate runner")
            dense_generate = False
        if getattr(model_config, "is_moe", False):
            # MoE defaults to V1 while the V2 MoE bring-up is in progress, but
            # is not recorded unsupported, so an explicit
            # VLLM_USE_V2_MODEL_RUNNER=1 request is honored. MoE with data
            # parallelism stays hard-blocked unless the experimental
            # VLLM_NEURON_ALLOW_V2_WIDE_EP gate is set: the V2 cross-DP shape
            # sync assumes phase-homogeneous traffic, and mixed prefill/decode
            # phases across ranks raise at runtime (no cross-rank phase
            # coordination; the disaggregated-inference deployment that keeps
            # each fleet single-phase is not supported on V2 yet).
            dense_generate = False
            if (
                _parallel_degree(parallel_config, "data_parallel_size") > 1
                and not neuron_envs.VLLM_NEURON_ALLOW_V2_WIDE_EP
            ):
                unsupported.append("Mixture-of-Experts with data parallelism")
        if getattr(model_config, "is_hybrid", False):
            unsupported.append("hybrid model")
            dense_generate = False
        if getattr(model_config, "is_attention_free", False):
            unsupported.append("attention-free model")
            dense_generate = False
        if getattr(model_config, "is_encoder_decoder", False):
            unsupported.append("encoder-decoder model")
            dense_generate = False
        if getattr(model_config, "multimodal_config", None) is not None:
            unsupported.append("multimodal model")
            dense_generate = False
        if getattr(model_config, "enable_prompt_embeds", False):
            unsupported.append("prompt embeddings")
        if getattr(model_config, "logits_processors", None):
            unsupported.append("custom logits processors")
        # On-device sampling gathers raw logits, so only the raw logprobs modes
        # are supported; route processed/unknown modes to V1.
        logprobs_mode = getattr(model_config, "logprobs_mode", None)
        if isinstance(logprobs_mode, str) and logprobs_mode not in (
            "raw_logprobs",
            "raw_logits",
        ):
            unsupported.append(f"logprobs mode '{logprobs_mode}'")
        unsupported_arch = next(
            (
                arch
                for arch in _model_architectures(model_config)
                if arch in UNSUPPORTED_V2_NEURON_RUNNER_ARCHITECTURES
            ),
            None,
        )
        if unsupported_arch is not None:
            unsupported.append(f"model architecture {unsupported_arch}")
            dense_generate = False

    if getattr(vllm_config, "diffusion_config", None) is not None:
        unsupported.append("diffusion model")
        dense_generate = False
    if getattr(vllm_config, "speculative_config", None) is not None:
        unsupported.append("speculative decoding")
    # MRv2 supports KV transfer for dense, non-speculative models.
    # Configurations that use unsupported model types, parallelism modes, or
    # serving features continue to use MRv1.
    if getattr(vllm_config, "ec_transfer_config", None) is not None:
        unsupported.append("EC transfer")
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        unsupported.append("weight transfer")
    if getattr(vllm_config, "lora_config", None) is not None:
        unsupported.append("LoRA")
    if getattr(vllm_config, "reasoning_config", None) is not None:
        unsupported.append("reasoning configuration")

    if cache_config is not None:
        if getattr(cache_config, "kv_offloading_size", None) is not None:
            unsupported.append("KV offloading")
        if getattr(cache_config, "kv_sharing_fast_prefill", False):
            unsupported.append("KV sharing fast prefill")

    if parallel_config is not None:
        # Replica data parallelism (data_parallel_size > 1) IS supported. For a dense
        # model upstream treats DP ranks as fully independent replicas: it sets
        # data_parallel_size to 1 inside each engine-core process and uses the plain
        # EngineCoreProc rather than DPEngineCoreProc, which is MoE-only
        # (vllm/v1/engine/core.py). No cross-DP collective fires either, so there is no
        # lockstep or dummy-batch requirement.
        #
        # Component DP (attention/embedding/mlp/lm_head DP, set in neuron_config) is a
        # different feature and is NOT admitted by removing this gate. The explicit
        # _has_component_dp check below rejects it as "component data parallelism",
        # independent of this gate and of disaggregated inference. See
        # test_component_dp_still_routes_to_v1.
        if getattr(parallel_config, "enable_expert_parallel", False):
            # Expert parallelism defaults to V1 while the V2 MoE bring-up is in
            # progress, but is not recorded unsupported: with data_parallel_size
            # == 1 the expert all-to-all runs entirely inside the compiled
            # graph and an explicit VLLM_USE_V2_MODEL_RUNNER=1 request is
            # honored. EP with data parallelism stays hard-blocked unless the
            # experimental VLLM_NEURON_ALLOW_V2_WIDE_EP gate is set (same
            # single-phase limitation as the Mixture-of-Experts guard above).
            dense_generate = False
            if (
                _parallel_degree(parallel_config, "data_parallel_size") > 1
                and not neuron_envs.VLLM_NEURON_ALLOW_V2_WIDE_EP
            ):
                unsupported.append("expert parallelism with data parallelism")
        if _parallel_degree(parallel_config, "decode_context_parallel_size") > 1:
            unsupported.append("decode context parallelism")
        if _parallel_degree(parallel_config, "prefill_context_parallel_size") > 1:
            unsupported.append("prefill context parallelism")
        if _parallel_degree(parallel_config, "pipeline_parallel_size") > 1:
            unsupported.append("pipeline parallelism")
        if getattr(parallel_config, "enable_dbo", False):
            unsupported.append("dual batch overlap")
        if getattr(parallel_config, "enable_elastic_ep", False):
            unsupported.append("elastic expert parallelism")

    if scheduler_config is not None:
        scheduler_cls = getattr(scheduler_config, "scheduler_cls", None)
        scheduler_path = _scheduler_class_path(scheduler_cls)
        if scheduler_cls is not None and (
            scheduler_path not in V2_COMPATIBLE_SCHEDULER_CLASSES
        ):
            unsupported.append("custom scheduler")

    # These features would be served incorrectly by V2, so route them to V1
    # until the corresponding support lands.
    # The neuron_config lives in additional_config["neuron_config"] (a dict); read it defensively so a
    # partially-populated or absent config never raises here.
    additional = getattr(vllm_config, "additional_config", None)
    if (
        isinstance(additional, dict)
        and "neuron_config" in additional
        and not isinstance(additional["neuron_config"], dict)
    ):
        unsupported.append("malformed neuron_config")
    neuron_cfg = _neuron_config_dict(vllm_config)
    if neuron_cfg.get("decode_context_length_buckets") is not None:
        unsupported.append("decode context-length bucketing")

    # MRv2 supports dense, non-speculative disaggregated inference with tensor
    # parallelism. Connector compatibility is validated during KV transfer
    # setup. Component data parallelism remains unsupported.
    #
    # A request with a complete transferred prompt prefix can start decoding
    # directly. An incomplete prefix still requires segmented prefill.
    if _has_component_dp(neuron_cfg):
        unsupported.append("component data parallelism")
    # On-device sampling is required (the compiled adapter returns sampled TOKENS, not logits). ABSENT =
    # enabled by default (do not reject); only an EXPLICIT None disables it.
    if (
        "on_device_sampling_config" in neuron_cfg
        and neuron_cfg["on_device_sampling_config"] is None
    ):
        unsupported.append("CPU sampling (on-device sampling disabled)")
    if neuron_cfg.get("tensor_capture") is not None:
        unsupported.append("tensor capture")
    if neuron_cfg.get("tensor_replacement") is not None:
        unsupported.append("tensor replacement")
    if neuron_cfg.get("debug_logits_dir") is not None:
        component_dp_fields = (
            "attention_dp_size",
            "embedding_dp_size",
            "mlp_dp_size",
            "draft_mlp_dp_size",
            "lm_head_dp_size",
        )
        has_component_dp = any(
            isinstance(neuron_cfg.get(name, 1), int)
            and not isinstance(neuron_cfg.get(name, 1), bool)
            and neuron_cfg.get(name, 1) > 1
            for name in component_dp_fields
        )
        if (
            model_config is None
            or not _DEBUG_LOGITS_V2_ARCHITECTURES.intersection(
                _model_architectures(model_config)
            )
            or has_component_dp
        ):
            unsupported.append("debug logits for unsupported V2 model/configuration")
    if neuron_cfg.get("fp8_packed_kv"):
        unsupported.append("packed FP8 KV")
    # The GPT-OSS decode giga-kernel consumes the contiguous stacked KV buffer
    # (use_stacked_kv_cache); the V2 runner only allocates per-layer KV.
    if neuron_cfg.get("gk_decode"):
        unsupported.append("GK decode (stacked KV cache)")
    if neuron_cfg.get("enable_structured_outputs", False):
        unsupported.append("structured outputs")
    if model_config is not None:
        # Device eager is unsupported (the adapter's in-graph gathers are compiled-only). CPU-sim eager
        # is a test-only mode that never selects V2 through this policy.
        if getattr(model_config, "enforce_eager", False):
            unsupported.append("eager execution")
        # Sliding-window models default to V1 while V2 SWA support is being
        # brought up, but are not recorded unsupported: the V2 runner can now
        # express windowed layers (SlidingWindowSpec groups + host-trimmed block
        # rows), so an explicit VLLM_USE_V2_MODEL_RUNNER=1 request is honored.
        get_sw = getattr(model_config, "get_sliding_window", None)
        if callable(get_sw) and get_sw() is not None:
            dense_generate = False

    return NeuronModelRunnerPolicy(dense_generate, tuple(unsupported))


def _neuron_config_dict(vllm_config: object) -> dict:
    """Return the nested Neuron configuration dictionary.

    Treat absent, malformed, or partially populated configuration data as no
    Neuron overrides.

    Args:
        vllm_config: Configuration object to inspect.

    Returns:
        The ``additional_config["neuron_config"]`` dictionary, or an empty
        dictionary when unavailable.
    """
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return {}
    neuron_cfg = additional.get("neuron_config")
    return neuron_cfg if isinstance(neuron_cfg, dict) else {}


def resolve_neuron_model_runner(vllm_config: "VllmConfig") -> bool:
    """Resolve the model-runner override against the Neuron policy.

    Args:
        vllm_config: Complete vLLM configuration to evaluate.

    Returns:
        ``True`` when the standalone runner should be used.
    """
    override = vllm_envs.VLLM_USE_V2_MODEL_RUNNER
    if override is not None:
        if override:
            unsupported = get_neuron_model_runner_v2_policy(
                vllm_config
            ).unsupported_features
            if unsupported:
                logger.warning_once(
                    "VLLM_USE_V2_MODEL_RUNNER=1 forces Neuron Model Runner V2 "
                    "with unsupported features (%s); execution may fail.",
                    ", ".join(unsupported),
                )
        return override

    policy = get_neuron_model_runner_v2_policy(vllm_config)
    if not policy.default_enabled:
        return False
    if policy.unsupported_features:
        logger.warning_once(
            "Neuron Model Runner V2 does not yet support %s; using V1.",
            ", ".join(policy.unsupported_features),
        )
        return False
    return True


def cache_neuron_model_runner_decision(
    vllm_config: "VllmConfig",
) -> bool:
    """Cache the model-runner decision on the vLLM configuration.

    Args:
        vllm_config: Configuration that owns the cached server decision.

    Returns:
        The cached model-runner decision.

    Raises:
        TypeError: If an existing cached value is not a Boolean.
    """
    state = vars(vllm_config)
    # Keep the decision under the public property name. deepcopy/pickle retain
    # the instance dictionary, and the patched getter below reads this value
    # instead of recomputing policy in each process.
    key = "use_v2_model_runner"
    if key not in state:
        state[key] = resolve_neuron_model_runner(vllm_config)

    decision = state[key]
    if not isinstance(decision, bool):
        raise TypeError("VllmConfig.use_v2_model_runner must be a bool")
    return decision


def validate_neuron_model_runner(vllm_config: "VllmConfig") -> None:
    """Validate an explicit standalone-runner request.

    Args:
        vllm_config: Complete vLLM configuration to validate.

    Raises:
        ValueError: If the configuration contains unsupported features.
    """
    unsupported = get_neuron_model_runner_v2_policy(vllm_config).unsupported_features
    if unsupported:
        raise ValueError(
            "Neuron Model Runner V2 does not yet support: " + ", ".join(unsupported)
        )


def _run_engine_core_with_cached_neuron_decision(*args, **kwargs):
    """Run EngineCore after installing and caching the Neuron policy.

    Cache the decision before EngineCore normalizes data-parallel state.

    Args:
        *args (Any): Positional arguments for the EngineCore entry point.
        **kwargs (Any): Keyword arguments for the EngineCore entry point,
            including ``vllm_config``.

    Returns:
        Any: The original EngineCore entry point result.

    Raises:
        RuntimeError: If the patch no longer retains the original entry point.
    """
    # The CLI uses spawn, so runtime class patches from the server process are
    # not present in a newly started EngineCore process. The cached bool does
    # survive pickling, making this cache call a no-op for normal server starts.
    patch_vllm_config_for_neuron()
    cache_neuron_model_runner_decision(kwargs["vllm_config"])

    # The engine is a MessageQueue reader too: it reads one response queue for each worker.
    # Those readers are built while EngineCore builds its executor, which the call below
    # starts, so the reader default must change first. The worker processes are covered
    # separately by NeuronWorker.init_device, and the patch is idempotent.
    from vllm_neuron.vllm.patches.shm_reader_spin_patch import (
        apply_shm_reader_spin_patch,
    )

    apply_shm_reader_spin_patch()

    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(
        EngineCoreProc,
        "_neuron_original_run_engine_core",
        EngineCoreProc.run_engine_core,
    )
    if original is _run_engine_core_with_cached_neuron_decision:
        raise RuntimeError("Neuron EngineCore entrypoint patch lost its original")
    return original(*args, **kwargs)


_run_engine_core_with_cached_neuron_decision._neuron_decision_patch = (  # type: ignore[attr-defined]
    True
)


def _patch_engine_core_entrypoint_for_neuron(engine_core_proc_cls: type) -> None:
    """Patch the EngineCore entry point to cache the Neuron decision.

    Args:
        engine_core_proc_cls: EngineCore process class to patch.
    """
    current = engine_core_proc_cls.run_engine_core
    if getattr(current, "_neuron_decision_patch", False):
        return

    # Save the unwrapped entrypoint once so repeated platform initialization
    # cannot build a wrapper chain or recurse.
    engine_core_proc_cls._neuron_original_run_engine_core = staticmethod(current)
    engine_core_proc_cls.run_engine_core = staticmethod(
        _run_engine_core_with_cached_neuron_decision
    )


_ENGINE_CORE_MODULE = "vllm.v1.engine.core"


class _EngineCorePatchLoader(Loader):
    """Delegate module loading and patch EngineCore before import returns."""

    def __init__(self, loader: Loader, finder: "_EngineCorePatchFinder"):
        self._loader = loader
        self._finder = finder

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _patch_engine_core_entrypoint_for_neuron(module.EngineCoreProc)
        self._finder.uninstall()


class _EngineCorePatchFinder(MetaPathFinder):
    """Find EngineCore once and install a post-import patch loader."""

    _neuron_engine_core_finder = True

    def find_spec(self, fullname, path, target=None):
        if fullname != _ENGINE_CORE_MODULE:
            return None

        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _EngineCorePatchLoader(spec.loader, self)
        return spec

    def uninstall(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)


def patch_engine_core_for_neuron() -> None:
    """Patch EngineCore now or after its natural import.

    Keep configuration validation lazy by installing a one-shot finder when
    EngineCore has not yet been imported. Replace the spawn target before the
    importing caller reads ``EngineCoreProc.run_engine_core``.
    """
    module = sys.modules.get(_ENGINE_CORE_MODULE)
    if module is not None and hasattr(module, "EngineCoreProc"):
        _patch_engine_core_entrypoint_for_neuron(module.EngineCoreProc)
        return

    if any(
        getattr(finder, "_neuron_engine_core_finder", False) for finder in sys.meta_path
    ):
        return
    sys.meta_path.insert(0, _EngineCorePatchFinder())


_neuron_policy_patch_installed = False
_neuron_policy_import_hook_installed = False


def _apply_max_concurrent_batches(vllm_config_cls: type) -> None:
    """Let a performance run deepen the async batch queue, so the host gets more slack.

    ``max_concurrent_batches`` is a read-only property returning 2 for async scheduling at
    ``pipeline_parallel_size <= 1``, which leaves the host exactly one device step of slack. A
    host critical path longer than that makes the step period host-bound by construction.

    Opt-in, because deeper is not generally correct: the runner keeps one step of sampled tokens
    and reads it as the immediately preceding step, which only holds while one step is
    outstanding. Deeper, that slot goes stale and tokens come out wrong at scattered offsets
    while the rest of the sequence looks healthy. Whoever sets the variable owns proving their
    own output.

    Pipeline parallel keeps the upstream value, where the number must equal the pipeline depth.

    Args:
        vllm_config_cls: The ``VllmConfig`` class to patch.

    Example:
        >>> _apply_max_concurrent_batches(VllmConfig)  # doctest: +SKIP
    """
    requested_depth = neuron_envs.VLLM_NEURON_BATCH_QUEUE_DEPTH
    if requested_depth <= 0:
        # The common case, and every service and test. Upstream owns the depth.
        return
    if not (
        NEURON_MIN_BATCH_QUEUE_DEPTH <= requested_depth <= NEURON_MAX_BATCH_QUEUE_DEPTH
    ):
        # Above the maximum corrupts input_ids. Below the minimum removes the batch queue and
        # serialises every step. Both are worse than upstream, so serve upstream and say so.
        logger.error(
            "Neuron: VLLM_NEURON_BATCH_QUEUE_DEPTH=%d is outside the range the model runner "
            "serves (%d to %d). Below the minimum there is no batch queue at all, and above "
            "the maximum input_ids come out corrupted. Keeping the upstream depth.",
            requested_depth,
            NEURON_MIN_BATCH_QUEUE_DEPTH,
            NEURON_MAX_BATCH_QUEUE_DEPTH,
        )
        return

    existing = vllm_config_cls.__dict__.get("max_concurrent_batches")
    if not isinstance(existing, property):
        # Absent, or no longer a property. Either way there is nothing to wrap, and reading
        # .fget would raise. Guessing another name would be worse than leaving it to upstream.
        logger.debug(
            "Neuron: VllmConfig.max_concurrent_batches is %s, not a property. The "
            "batch-queue depth stays with upstream.",
            type(existing).__name__,
        )
        return
    if getattr(existing.fget, "_neuron_batch_queue_patch", False):
        return

    original_getter = existing.fget

    def max_concurrent_batches(vllm_config: "VllmConfig") -> int:
        # Test our own case first. The upstream getter reads use_v2_model_runner, which
        # resolves and caches the Neuron policy, so do not call it when the answer is ours.
        if (
            vllm_config.scheduler_config.async_scheduling
            and vllm_config.parallel_config.pipeline_parallel_size <= 1
        ):
            return requested_depth
        return original_getter(vllm_config)

    max_concurrent_batches._neuron_batch_queue_patch = True  # type: ignore[attr-defined]
    vllm_config_cls.max_concurrent_batches = property(
        max_concurrent_batches,
        doc=existing.__doc__,
    )


def _apply_neuron_policy_patch(vllm_config_cls: type) -> None:
    _apply_max_concurrent_batches(vllm_config_cls)
    use_model_runner_property = vllm_config_cls.__dict__["use_v2_model_runner"]
    if getattr(use_model_runner_property.fget, "_neuron_policy_patch", False):
        return

    # The getter is intentionally thin: policy is resolved once per server
    # config and every copied config observes that same cached decision.
    def use_model_runner(vllm_config: "VllmConfig") -> bool:
        return cache_neuron_model_runner_decision(vllm_config)

    use_model_runner._neuron_policy_patch = True  # type: ignore[attr-defined]
    vllm_config_cls.use_v2_model_runner = property(
        use_model_runner,
        doc=use_model_runner_property.__doc__,
    )
    # Upstream validation is GPU-specific; use the Neuron feature policy while
    # preserving the same VllmConfig validation entrypoint.
    vllm_config_cls._validate_v2_model_runner = validate_neuron_model_runner


def patch_vllm_config_for_neuron() -> None:
    """Install the Neuron model-runner policy on ``VllmConfig``.

    Defer the patch with an import audit hook when configuration modules are
    still participating in an import cycle.
    """
    global _neuron_policy_import_hook_installed, _neuron_policy_patch_installed

    if _neuron_policy_patch_installed:
        return

    try:
        from vllm.config.vllm import VllmConfig
    except (ImportError, AttributeError):
        if _neuron_policy_import_hook_installed:
            return
        _neuron_policy_import_hook_installed = True

        # Platform pre-registration can run while vllm.config.vllm is still
        # importing. Defer the patch until that module has defined VllmConfig.
        def _audit_hook(event, args):
            global _neuron_policy_patch_installed
            if _neuron_policy_patch_installed or event != "import":
                return
            module_name = str(args[0])
            if "vllm.config" not in module_name:
                return
            module = sys.modules.get("vllm.config.vllm")
            if module is not None and hasattr(module, "VllmConfig"):
                _apply_neuron_policy_patch(module.VllmConfig)
                _neuron_policy_patch_installed = True

        sys.addaudithook(_audit_hook)
        return

    _apply_neuron_policy_patch(VllmConfig)
    _neuron_policy_patch_installed = True
