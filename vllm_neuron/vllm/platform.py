# SPDX-License-Identifier: Apache-2.0
"""
Neuron Platform implementation for vLLM

This module provides the NeuronPlatform class that integrates vLLM Neuron with vLLM's
platform plug-in system.
"""

import logging
import os
import sys
import time as _time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING

import torch
from vllm_neuron import envs

from vllm.platforms import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.inputs import EngineInput
    from vllm.sampling_params import SamplingParams
    from vllm.pooling_params import PoolingParams
else:
    VllmConfig = None

logger = logging.getLogger(__name__)

# vLLM's upstream default for ParallelConfig.cp_kv_cache_interleave_size
# (token-level interleaving). Verified against vLLM 0.24.0
# (vllm/config/parallel.py). Used as the "unset" sentinel: DCP replaces it with
# block_size, and an explicit non-default value is honored/validated as-is.
# If a future vLLM bump changes this default, revisit the DCP interleave
# defaulting in update_block_size_for_backend and _validate_dcp_config.
_UPSTREAM_DCP_INTERLEAVE_DEFAULT = 1

SO_DISABLED_MESSAGE = (
    "Structured outputs were requested, but this Neuron server was started "
    "with additional_config.neuron_config.enable_structured_outputs=false. "
    "Restart the server with enable_structured_outputs=true to serve "
    "structured-output requests. For this SO-off-only server, retry the "
    "request without structured_outputs."
)


# Relax DCP config validation for prefill-side DCP where TP <= num_kv_heads.
# Upstream asserts TP > num_kv_heads when DCP is enabled, but prefill DCP
# shards attention across the DCP sub-group, not full TP.
#
# During plugin registration, vllm.config.model may not be importable due to
# circular imports. We attempt eagerly; if that fails we use sys.addaudithook
# to patch as soon as the module loads. The hook self-disables after one
# successful patch, so ongoing overhead is a single boolean check per import.
_dcp_config_patched = False


def _patch_dcp_config_validation():
    """Patch ModelConfig.verify_with_parallel_config to skip DCP checks on prefill.

    Called during plugin registration. If the import fails (circular import),
    installs an audit hook that patches on first successful import of
    vllm.config.model. The hook is lightweight (one boolean check) after
    firing and cannot be removed per CPython API, but the cost is negligible
    compared to import overhead itself.
    """
    global _dcp_config_patched
    if _dcp_config_patched:
        return
    _dcp_config_patched = True

    try:
        from vllm.config.model import ModelConfig

        _apply_dcp_patch(ModelConfig)
    except (ImportError, AttributeError):
        # Circular import during plugin registration. Use an audit hook to
        # apply the patch as soon as vllm.config.model finishes loading.
        # The hook fires on every "import" event but self-disables (via
        # _applied flag) after the first successful patch, reducing ongoing
        # cost to a single boolean check per subsequent import.
        import sys

        _applied = [False]

        def _audit_hook(event, args):
            if _applied[0]:
                return
            if event == "import" and "vllm.config" in str(args[0]):
                mod = sys.modules.get("vllm.config.model")
                if mod is not None and hasattr(mod, "ModelConfig"):
                    _applied[0] = True
                    _apply_dcp_patch(mod.ModelConfig)

        sys.addaudithook(_audit_hook)


# Detects an explicit `optimization_level` kwarg on the offline `LLM()`
# constructor. vLLM's own EngineArgs/VllmConfig defaulting can't distinguish
# "caller passed O2" from "O2 is just the dataclass default" once the value
# lands on VllmConfig, so we capture the signal one layer up -- from the raw
# kwargs dict `LLM.__init__` forwards into `EngineArgs(**kwargs)` -- while it
# still carries that information. Mirrors the CLI-side `sys.argv` check in
# `apply_config_platform_defaults` below for the offline API.
_llm_init_explicit_optimization_level = False
_llm_init_patched = False


def _patch_llm_init_optimization_level():
    """Patch LLM.__init__ to record whether `optimization_level` was passed.

    Called during plugin registration, where `vllm.entrypoints.llm` usually
    can't be imported yet (circular import: it pulls in `vllm.config`,
    which isn't finished initializing while the platform plugin is still
    being resolved). Unlike the DCP patch's `sys.addaudithook` fallback --
    whose "import" event fires *before* the target module is registered in
    `sys.modules`, so a later re-check inside the same hook invocation sees
    nothing and the patch silently never lands -- this wraps `__import__`
    itself and checks `sys.modules` right after each real import call
    returns, when the module is guaranteed fully loaded. Self-removes after
    the first successful patch.
    """
    global _llm_init_patched
    if _llm_init_patched:
        return
    _llm_init_patched = True

    try:
        from vllm.entrypoints.llm import LLM

        _apply_llm_init_patch(LLM)
    except (ImportError, AttributeError):
        import builtins

        _orig_import = builtins.__import__
        _applied = [False]

        def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
            result = _orig_import(name, globals, locals, fromlist, level)
            if not _applied[0]:
                mod = sys.modules.get("vllm.entrypoints.llm")
                if mod is not None and hasattr(mod, "LLM"):
                    _applied[0] = True
                    builtins.__import__ = _orig_import
                    _apply_llm_init_patch(mod.LLM)
            return result

        builtins.__import__ = _tracking_import


def _apply_llm_init_patch(LLM):
    """Monkey-patch LLM.__init__ to set a module-level flag around the call.

    `EngineArgs`/`VllmConfig` construction happens synchronously inside the
    original `__init__` (no fork yet), so `apply_config_platform_defaults`
    can read the flag while it's still set.
    """
    if getattr(LLM.__dict__.get("__init__"), "_neuron_optlevel_patched", False):
        return

    _orig_init = LLM.__init__

    def _patched_init(self, *args, **kwargs):
        global _llm_init_explicit_optimization_level
        prev = _llm_init_explicit_optimization_level
        _llm_init_explicit_optimization_level = "optimization_level" in kwargs
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            _llm_init_explicit_optimization_level = prev

    _patched_init._neuron_optlevel_patched = True
    LLM.__init__ = _patched_init


def _apply_dcp_patch(ModelConfig):
    """Monkey-patch verify_with_parallel_config to bypass DCP assertion for prefill."""
    if getattr(
        ModelConfig.__dict__.get("verify_with_parallel_config"),
        "_neuron_dcp_patched",
        False,
    ):
        return

    _orig_verify = ModelConfig.verify_with_parallel_config

    def _patched_verify(self, parallel_config):
        dcp = parallel_config.decode_context_parallel_size
        if dcp > 1 and not self.use_mla:
            saved_dcp = parallel_config.decode_context_parallel_size
            parallel_config.decode_context_parallel_size = 1
            _orig_verify(self, parallel_config)
            parallel_config.decode_context_parallel_size = saved_dcp
            return
        _orig_verify(self, parallel_config)

    _patched_verify._neuron_dcp_patched = True
    ModelConfig.verify_with_parallel_config = _patched_verify


class NeuronPlatform(Platform):
    # OOT (Out-Of-Tree) refers to hardware platforms that are integrated with vLLM
    # through plugin projects.
    _enum = PlatformEnum.OOT
    # Note we need to make sure libtorch_neuronx_lite has been imported before
    # getting here. This is to guarantee the privateuseone binding and
    # neuron renaming has been registered with torch.
    device_name: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    device_type: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = [
        "neuron_quant",
        "compressed-tensors",
        "modelopt",
    ]
    # Neuron quantization paths that CPU-dequant compressed-tensors weights to
    # BF16 on the loader thread. On these paths the device only ever sees BF16
    # (the graph is identical to bf16), so an on-disk compressed-tensors weight/
    # activation quant config describes the checkpoint, not what the device runs.
    # The KV-cache-only guard in _validate_quantization_config is waived for
    # them -- gated on operator intent (neuron_config.quantization), NOT the
    # checkpoint format, so pointing at an MXFP8 checkpoint WITHOUT selecting
    # this path still fails loud.
    _cpu_dequant_quantizations: frozenset[str] = frozenset({"mxfp8"})
    device_control_env_var: str = "NEURON_VISIBLE_DEVICES"
    _device_count: int = -1
    _termination_timeout_patched: bool = False
    # Max embeds one multimodal item can produce. Set once in
    # check_and_update_config. None means "no encoder validation". In embed
    # (post-merge) space, so validate_request can compare directly against
    # mm_pos.get_num_embeds().
    _max_embeds_per_image: int | None = None
    # Structured outputs are request-level, while the SO-off perf mode is a
    # server-level compile choice. Validate requests before they reach the
    # engine so a bad request returns a normal request error.
    _enable_structured_outputs: bool = False

    @classmethod
    def import_kernels(cls) -> None:
        """Import platform C kernels without leaking the caller's stack.

        The base implementation passes the live ImportError to
        ``logger.warning_once``, whose ``lru_cache`` retains the exception --
        and through ``__traceback__`` every frame on the import path,
        including the ``LLM`` instance under construction. ``vllm._C`` never
        exists on Neuron, so that would pin each process's first ``LLM``
        (and its EngineCore, and its NeuronCores) until interpreter exit.
        Suppress the imports without capturing the exception instead.
        """
        import contextlib

        with contextlib.suppress(ImportError):
            import vllm._C  # noqa: F401
        with contextlib.suppress(ImportError):
            import vllm._moe_C_stable_libtorch  # noqa: F401

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # TODO: Return based on hardware generation like GPUs and TPUs
        # It is only used for logging and debugging - hence not critical
        return f"neuron:{device_id}"

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """Register Neuron model architectures before ModelConfig validation."""
        import os

        from vllm_neuron.vllm.model_runner_v2_policy import (
            patch_vllm_config_for_neuron,
        )

        # VllmConfig.__post_init__ reads and validates this property, so install
        # the Neuron policy before ModelConfig validation constructs the config.
        patch_vllm_config_for_neuron()

        if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
            from vllm.model_executor.models.registry import ModelRegistry

            ModelRegistry.register_model(
                "SyntheticNeuronModel",
                "vllm_neuron.model.synthetic:SyntheticNeuronModel",
            )
            # SyntheticStackedNeuronModel (GK stacked-KV DI adaptor test) is a
            # SyntheticNeuronModel subclass; register it under the same synthetic
            # gate so it resolves in every worker subprocess.
            ModelRegistry.register_model(
                "SyntheticStackedNeuronModel",
                "vllm_neuron.model.synthetic:SyntheticStackedNeuronModel",
            )

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        """Default block_size to 32 for Neuron when the user didn't override.

        Also default the DCP KV-cache interleave size to the (now-final)
        block_size. This runs in the worker after upstream has settled
        block_size, so block_size is authoritative here — unlike
        check_and_update_config, which runs on earlier passes where block_size
        may still be the upstream default. Keying on the upstream interleave
        default (1) both leaves a user's explicit value alone (CLI or offline
        override_neuron_config) and avoids re-clobbering across passes.
        """
        cache_config = vllm_config.cache_config
        if not cache_config.user_specified_block_size:
            cache_config.block_size = 32

        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.decode_context_parallel_size > 1
            and parallel_config.cp_kv_cache_interleave_size
            == _UPSTREAM_DCP_INTERLEAVE_DEFAULT
        ):
            parallel_config.cp_kv_cache_interleave_size = cache_config.block_size

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        """Default the optimization level to O2 on the native route.

        vLLM defaults optimization_level to O2. The native (backend=neuron)
        route keeps O2. -O1 has nondeterministic instruction scheduling that
        makes native compiles flaky, so the native route compiles at O2.
        The XLA route keeps its O1 default.

        An explicit level is always honored: `--optimization-level` (CLI) or
        the offline `LLM(optimization_level=...)` API (detected via
        `_llm_init_explicit_optimization_level`).

        TODO: address the -O1 issue so the native route does not need a
        different default.
        """
        from vllm.config.vllm import OptimizationLevel

        explicit = (
            any(a.startswith("--optimization-level") for a in sys.argv)
            or _llm_init_explicit_optimization_level
        )
        # Record the signal for the model runner. An explicit level from the
        # caller must win over a model's per-graph default (see
        # compiler_args_for). This hook is the only point where the signal is
        # valid: the CLI argv and the offline-API flag are both still readable
        # here, before EngineArgs defaulting hides them.
        vllm_config._neuron_optimization_level_explicit = explicit
        if explicit:
            logger.info(
                "Explicit --optimization-level: keeping O%s",
                vllm_config.optimization_level.value,
            )
            return
        # The native route keeps vLLM's O2 default. Do not downgrade it.
        if envs.is_native_backend():
            logger.info(
                "Using default optimization level O%s on the Neuron native route",
                vllm_config.optimization_level.value,
            )
            return
        # XLA route: keep the historical O1 default. Non-O2 values come only
        # from the offline API, so lower the ambiguous O2 default alone.
        if vllm_config.optimization_level == OptimizationLevel.O2:
            vllm_config.optimization_level = OptimizationLevel.O1
            logger.info("Defaulting optimization level to O1 on the XLA route")

    @classmethod
    def _resolve_ec_producer_encoder_cache_size(
        cls, vllm_config: "VllmConfig", model_config, scheduler_config
    ) -> None:
        """Derive the encoder-cache budget from the block count on an EC producer.

        Upstream derives ``scheduler_config.encoder_cache_size`` from
        ``max_num_batched_tokens``, a language-model batching budget. That fits one
        engine that encodes an item and consumes it itself, because the item leaves
        the cache after the prefill that reads it.

        A disaggregated vision engine instead holds each encoded item until a remote
        prefill engine pulls it. Its cache must cover every item that is encoded but
        not read yet, which the batching budget cannot express.

        This runs before the engine builds the scheduler and the model runner, so
        both the scheduler's EncoderCacheManager and the worker's device buffer
        follow the new size.

        A consumer and a monolithic engine keep the upstream value. Upstream also
        raises the budget to hold one largest single item, so a block count below
        that item has no effect.

        Args:
            vllm_config: The engine configuration.
            model_config: The model configuration, for the vision merge factor.
            scheduler_config: The scheduler configuration to update.
        """
        ec_config = getattr(vllm_config, "ec_transfer_config", None)
        if ec_config is None or not ec_config.is_ec_producer:
            return

        encoder_config_dict = vllm_config.additional_config.get("encoder_neuron_config")
        if not encoder_config_dict:
            return
        num_blocks = encoder_config_dict.get("encoder_cache_num_blocks")
        if not num_blocks:
            return

        from vllm_neuron.model.neuron_config import EncoderNeuronConfig
        from vllm_neuron.utils.mm_utils import get_encoder_cache_block_size

        encoder_config = EncoderNeuronConfig.from_dict(encoder_config_dict)
        # One block is scratch: it absorbs encoder padding writes and never holds
        # an item, so it cannot count toward the budget.
        cache_block_size = get_encoder_cache_block_size(
            model_config.hf_config, encoder_config.encoder_attention_block_size
        )
        budget = (num_blocks - 1) * cache_block_size
        if budget <= scheduler_config.encoder_cache_size:
            return

        logger.info(
            "Encoder-cache budget for this encoder-cache producer: %d -> %d embeds "
            "(%d blocks of %d). The default comes from max_num_batched_tokens, "
            "which does not cover the items a remote consumer has not read yet.",
            scheduler_config.encoder_cache_size,
            budget,
            num_blocks - 1,
            cache_block_size,
        )
        scheduler_config.encoder_cache_size = budget

    @classmethod
    def _multimodal_inputs_enabled(cls, model_config) -> bool:
        """False when every modality's per-prompt limit is 0.

        A vision-capable checkpoint can be served text-only (Qwen3.5's text
        decoder is one), and ``limit_mm_per_prompt={"image": 0, "video": 0}`` is
        how that is expressed. Vision bucket resolution is then skipped rather
        than validated against a text-sized ``max_model_len``: below
        ``encoder_attention_block_size`` the auto-generated bucket is smaller
        than the block, and the second ``check_and_update_config`` (in the
        engine core) rejects it as an explicit bucket.
        """
        mm_config = getattr(model_config, "multimodal_config", None)
        if mm_config is None:
            return False
        limits = getattr(mm_config, "limit_per_prompt", None) or {}
        # vLLM normalises the raw ints into per-modality options that carry a
        # ``count``; read it when present.
        counts = [int(getattr(limit, "count", limit)) for limit in limits.values()]
        return not (counts and all(count == 0 for count in counts))

    @classmethod
    def _resolve_vision_auto_config(
        cls, vllm_config: "VllmConfig", model_config
    ) -> None:
        """Auto-generate vision token buckets when not explicitly configured.

        Vision only: each value counts pre-merge vision tokens, and an audio
        encoder counts blocks in the same field. The caller gates on
        ``hasattr(hf_config, "vision_config")``.
        """
        from vllm_neuron.model.neuron_config import EncoderNeuronConfig
        from vllm_neuron.utils.bucket_utils import (
            get_default_num_encoder_input_buckets,
            next_power_of_2,
        )

        if "encoder_neuron_config" not in vllm_config.additional_config:
            vllm_config.additional_config["encoder_neuron_config"] = {}
        encoder_config_dict = vllm_config.additional_config["encoder_neuron_config"]
        block_size = encoder_config_dict.get(
            "encoder_attention_block_size",
            EncoderNeuronConfig.encoder_attention_block_size,
        )

        mm_config = getattr(model_config, "multimodal_config", None)
        user_max_pixels = None
        if mm_config and mm_config.mm_processor_kwargs:
            user_max_pixels = mm_config.mm_processor_kwargs.get("max_pixels")
        if user_max_pixels:
            from vllm_neuron.utils.mm_utils import get_max_pixels_token_count

            per_image_tokens = get_max_pixels_token_count(
                model_config.hf_config, user_max_pixels
            )
            if per_image_tokens is not None and per_image_tokens > block_size:
                block_size = next_power_of_2(per_image_tokens)
                encoder_config_dict["encoder_attention_block_size"] = block_size
                logger.warning(
                    "Block size increased to %d to fit max_pixels=%d. Performance may be degraded for lower-resolution inputs.",
                    block_size,
                    user_max_pixels,
                )

        if encoder_config_dict.get("num_encoder_input_buckets"):
            if encoder_config_dict.get("max_encoder_seq_len"):
                raise ValueError(
                    "Cannot set both num_encoder_input_buckets and max_encoder_seq_len in encoder_neuron_config. Use num_encoder_input_buckets for explicit control, or max_encoder_seq_len to cap auto-generated buckets."
                )
            max_explicit = max(encoder_config_dict["num_encoder_input_buckets"])
            if max_explicit < block_size:
                raise ValueError(
                    f"Largest bucket ({max_explicit}) is smaller than encoder_attention_block_size ({block_size}). Increase num_encoder_input_buckets or reduce max_pixels."
                )
            return

        from vllm_neuron.utils.mm_utils import get_vision_token_merge_factor

        merge_factor = get_vision_token_merge_factor(model_config.hf_config)
        max_model_len = vllm_config.model_config.max_model_len
        max_bucket = next_power_of_2(max_model_len * merge_factor)

        max_encoder_seq_len = (
            encoder_config_dict.get("max_encoder_seq_len") or block_size
        )
        if max_encoder_seq_len < block_size:
            raise ValueError(
                f"max_encoder_seq_len ({max_encoder_seq_len}) must be >= "
                f"encoder_attention_block_size ({block_size})."
            )
        max_bucket = min(max_bucket, max_encoder_seq_len)

        buckets = get_default_num_encoder_input_buckets(max_bucket, block_size)
        encoder_config_dict["num_encoder_input_buckets"] = buckets

        logger.info(
            "Auto-generated encoder buckets %s (max_model_len=%d, max_encoder_seq_len=%d).",
            buckets,
            max_model_len,
            max_encoder_seq_len,
        )
        if max_encoder_seq_len < next_power_of_2(max_model_len * merge_factor):
            logger.info(
                "Encoder buckets are capped by max_encoder_seq_len=%d. "
                "Set max_encoder_seq_len in encoder_neuron_config to increase.",
                max_encoder_seq_len,
            )

    # Deprecated encoder option names, mapped to current name.
    _DEPRECATED_ENCODER_OPTIONS = {
        "num_vision_tokens_buckets": "num_encoder_input_buckets",
        "vision_attention_block_size": "encoder_attention_block_size",
        "max_vision_seq_len": "max_encoder_seq_len",
    }

    @classmethod
    def _migrate_deprecated_encoder_config(cls, vllm_config: "VllmConfig") -> None:
        """Accept the old ``vision_neuron_config`` option names."""

        additional_config = vllm_config.additional_config
        if "vision_neuron_config" in additional_config:
            if additional_config.get("encoder_neuron_config"):
                raise ValueError(
                    "'vision_neuron_config' is deprecated and conflicts with "
                    "'encoder_neuron_config'. Remove 'vision_neuron_config'."
                )
            logger.warning(
                "'vision_neuron_config' is deprecated. Its value applies as "
                "'encoder_neuron_config'. Rename the option to "
                "'encoder_neuron_config'."
            )
            additional_config["encoder_neuron_config"] = additional_config.pop(
                "vision_neuron_config"
            )

        encoder_config = additional_config.get("encoder_neuron_config")
        if not isinstance(encoder_config, dict):
            return

        for old_name, new_name in cls._DEPRECATED_ENCODER_OPTIONS.items():
            if old_name not in encoder_config:
                continue
            if new_name in encoder_config:
                raise ValueError(
                    f"'{old_name}' is deprecated and conflicts with "
                    f"'{new_name}'. Remove '{old_name}'."
                )
            logger.warning(
                "'%s' is deprecated. Its value applies as '%s'. Rename the "
                "option to '%s'.",
                old_name,
                new_name,
                new_name,
            )
            encoder_config[new_name] = encoder_config.pop(old_name)

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Configure vLLM for vLLM Neuron platform."""
        from vllm_neuron.vllm.model_runner_v2_policy import (
            cache_neuron_model_runner_decision,
            patch_engine_core_for_neuron,
        )

        # Resolve once on the server while the config still contains its full
        # parallelism state. Config copies then carry this bool to child processes.
        cache_neuron_model_runner_decision(vllm_config)
        # EngineCore uses spawn and does not inherit class patches; its wrapper
        # reinstalls the getter before upstream can normalize DP state.
        patch_engine_core_for_neuron()

        # EPD patch for VE SHM receive, no-op for non-EPD setups
        from vllm_neuron.vllm.patches.epd_shm_patch import apply_epd_shm_patch

        apply_epd_shm_patch()

        # Patches for bidirectional KV transfer and streaming KV (DI features).

        cls._patch_termination_timeouts()

        cache_config = vllm_config.cache_config
        if (
            cache_config is not None
            and envs.get_kv_headroom_env() is not None
            and cache_config.gpu_memory_utilization != 1.0
        ):
            logger.info(
                "Neuron uses a KV cache headroom budget. "
                "gpu_memory_utilization is set to 1.0."
            )
            cache_config.gpu_memory_utilization = 1.0

        model_config = vllm_config.model_config
        if model_config is None:
            return

        # Frontend multimodal patchify runs under vLLM's
        # set_default_torch_num_threads(), which pins torch to OMP_NUM_THREADS or
        # 1 when unset -- so patchify runs single-threaded by default. Default OMP
        # to a small count so it parallelizes. Workers inherit this and reset to 1
        # in NeuronWorker.init_device; setdefault lets an operator value win.
        if model_config.multimodal_config is not None:
            os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))

        # Rename each deprecated encoder option before any code reads it.
        cls._migrate_deprecated_encoder_config(vllm_config)

        neuron_config = vllm_config.additional_config.get("neuron_config", {})

        # vLLM-Neuron owns the GPT-OSS decode giga-kernel selection; the customer
        # does not set gk_decode. Derive it here, where kv_transfer_config is
        # available, and write it to the shared neuron_config dict so the model
        # factory and the NIXL connector observe the same decision.
        architectures = getattr(model_config.hf_config, "architectures", None) or []
        if "GptOssForCausalLM" in architectures:
            from vllm_neuron.model.gpt_oss.factory import should_use_gk_decode
            from libtorch_neuronx_lite.compile.platform import get_platform_target

            kv_cfg = vllm_config.kv_transfer_config
            kv_role = kv_cfg.kv_role if kv_cfg is not None else None
            neuron_config = vllm_config.additional_config.setdefault(
                "neuron_config", {}
            )
            neuron_config["gk_decode"] = should_use_gk_decode(
                neuron_config.get("quantization"), get_platform_target(), kv_role
            )
            if neuron_config["gk_decode"]:
                # envs.py forces NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1 on
                # every process as the Qwen3-MoE packed-KV workaround.
                # The GPT-OSS decode giga-kernel is correct only on nkilib's
                # default QK-swap path. On the non-swap path, its single-pass
                # attention over a long prior context returns wrong values.
                # The model then loses the prompt. Restore the default here.
                # This runs in the launcher process, before the engine and
                # worker processes inherit the environment and import
                # nkilib. nkilib reads the variable once at import time.
                # The write is unconditional on purpose: an explicit "1" in
                # the environment is indistinguishable from the envs.py
                # default, and the giga-kernel decode must not run on the
                # non-swap path. Log it, so the trace shows the kernel path.
                logger.info(
                    "[gk_decode] Set NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=0: "
                    "the giga-kernel decode uses the kernel default QK-swap "
                    "attention path."
                )
                os.environ["NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP"] = "0"

        # Qwen3-ASR runs the Whisper mel/fbank extractor (torch.stft, OpenMP)
        # during multimodal encoder-budget profiling in scheduler init. Under
        # the default fork start method, the threaded STFT deadlocks in the
        # forked EngineCore. Force spawn so this profiling runs in a clean
        # process. Scope this to Qwen3-ASR; other models keep fork.
        if "Qwen3ASRForConditionalGeneration" in architectures:
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # vLLM-Neuron owns the decode-only weight decision; the customer does not
        # set decode_only_weights. A model reads it to skip its prefill-only
        # weight copies. The gate matches the prefill-warmup gate in
        # NeuronWorker.compile_or_warm_up_model: a kv_consumer pool traces no
        # prefill graph, so its prefill weights are unreachable HBM. A kv_both
        # pool serves prefill too, so it keeps them.
        kv_cfg = vllm_config.kv_transfer_config
        # kv_role == "kv_consumer" is the same test as
        # `is_kv_consumer and not is_kv_producer`, because kv_both sets both.
        decode_only = bool(kv_cfg is not None and kv_cfg.kv_role == "kv_consumer")
        neuron_config = vllm_config.additional_config.setdefault("neuron_config", {})
        neuron_config["decode_only_weights"] = decode_only

        enable_structured_outputs = neuron_config.get(
            "enable_structured_outputs", False
        )
        if not isinstance(enable_structured_outputs, bool):
            raise ValueError(
                "enable_structured_outputs must be a boolean, "
                f"got {enable_structured_outputs}"
            )
        cls._enable_structured_outputs = enable_structured_outputs

        # JF decoding was initially built for xgrammar only. vLLM-Neuron also
        # supports llguidance, but it is not yet compatible with JF decoding.
        if neuron_config.get("max_jump_forward_tokens", 0) > 0:
            # guard for none for SO
            so_config = vllm_config.structured_outputs_config
            backend = so_config.backend if so_config is not None else None
            if backend != "xgrammar":
                raise ValueError(
                    "max_jump_forward_tokens > 0 requires the xgrammar structured "
                    "outputs backend, but the configured backend is "
                    f"'{backend}'. Set structured_outputs_config.backend to "
                    "'xgrammar' or disable jump-forward decoding."
                )

        # Jump-forward and speculative decoding are mutually exclusive. Resolve the
        # conflict here, on the shared neuron_config dict, before the engine-core and
        # worker processes each build their own NeuronConfig from it -- so both the
        # scheduler and the model runner observe max_jump_forward_tokens=0 and the
        # worker skips warming the (now unused) JF NEFF.
        if (
            neuron_config.get("max_jump_forward_tokens", 0) > 0
            and vllm_config.speculative_config is not None
        ):
            logger.warning(
                "max_jump_forward_tokens > 0 and speculative decoding are both "
                "configured, despite being mutually exclusive. "
                "Defaulting to spec-decoding only."
            )
            neuron_config["max_jump_forward_tokens"] = 0

        # Component DP on dense models needs the MoE/DP engine path to
        # preserve data_parallel_size across engine core subprocesses.
        if cls._has_neuron_component_dp(vllm_config):
            cls._validate_component_dp_config(vllm_config)
        cls._maybe_enable_moe_for_component_dp(vllm_config)

        cls._validate_dcp_config(vllm_config)
        cls._auto_set_neuron_connector_module_path(vllm_config)
        cls._auto_set_neuron_ec_connector_module_path(vllm_config)
        cls._validate_quantization_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        # TODO: Implement a CPU fallback based on the value from DeviceConfig
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm_neuron.vllm.worker.neuron_worker.NeuronWorker"
            )
            # block_size is set in `update_block_size_for_backend`.
            # TODO: Move all Neuron specific config overrides into a separate function
            # TODO: Can vLLM to vLLM Neuron config mapping be done here after the overrides are applied?

        allowed_backends = ("mp", "uni")
        if parallel_config.distributed_executor_backend not in allowed_backends:
            # multi-process executor is now the default executor in vLLM. But want
            # to explicitly set it here for clarity
            parallel_config.distributed_executor_backend = "mp"

        logger.warning(
            f"Neuron Platform only supports the following distributed executor backends: {allowed_backends}, using {parallel_config.distributed_executor_backend}"
        )

        # Set custom scheduler for Neuron platform
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is enabled on Neuron platform. Currently Neuron only supports"
                " chunking prefills with batch size of 1. Mixing prefill and decode in the same"
                " batch is not supported"
            )

        # Note: we do NOT set disable_chunked_mm_input=True here because
        # upstream vLLM enforces max_tokens_per_mm_item <= max_num_batched_tokens
        # when that flag is set — which fails for models like Qwen3-VL where a
        # single image can theoretically produce more tokens than our
        # max_num_batched_tokens. The encoder budget cap in NeuronScheduler
        # already prevents overflow without needing this flag.

        if hasattr(model_config.hf_config, "vision_config"):
            if cls._multimodal_inputs_enabled(model_config):
                cls._resolve_vision_auto_config(vllm_config, model_config)
            else:
                logger.info(
                    "All multimodal per-prompt limits are 0: skipping vision "
                    "bucket resolution and serving %s text-only",
                    model_config.architecture,
                )

        # Compute the per-item embed limit for request validation.
        # get_max_encoder_embed_budget converts the buckets to embed-token space.
        # Vision buckets are raw pre-merge tokens. Audio buckets are block
        # counts. The model applies the correct conversion for each modality,
        # so it replaces the vision-only max_bucket / merge_factor formula.
        encoder_config_dict = vllm_config.additional_config.get("encoder_neuron_config")
        if encoder_config_dict:
            buckets = encoder_config_dict.get("num_encoder_input_buckets")
            if buckets:
                from vllm_neuron.model.neuron_config import EncoderNeuronConfig
                from vllm_neuron.utils.mm_utils import get_max_encoder_embed_budget

                encoder_cfg = EncoderNeuronConfig.from_dict(encoder_config_dict)
                cls._max_embeds_per_image = get_max_encoder_embed_budget(
                    model_config.hf_config, encoder_cfg
                )

        cls._resolve_ec_producer_encoder_cache_size(
            vllm_config, model_config, scheduler_config
        )

        if envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
            # Capture copies a forward's inputs to host via a standalone op run
            # just before a plain execute. Under async scheduling the input
            # buffers may still hold, or be overwritten by, another in-flight
            # forward, so the op could serialize the wrong tensors — require
            # synchronous scheduling.
            if scheduler_config.async_scheduling:
                raise ValueError(
                    "Snapshot capture (VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE) is "
                    "incompatible with async scheduling. Disable async "
                    "scheduling (async_scheduling=False or "
                    "--no-async-scheduling) to capture snapshots."
                )
            # Resolve (and cache) the snapshot config now so a malformed
            # selection or FORMAT fails at startup, not on the first forward.
            from vllm_neuron.snapshot.config import get_snapshot_config

            get_snapshot_config()

        # Override the default vLLM scheduler with our custom Neuron scheduler
        # We need to override even if it's already set to the default scheduler
        if scheduler_config.scheduler_cls is None or scheduler_config.scheduler_cls in (
            "vllm.v1.core.sched.scheduler.Scheduler",
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        ):
            if scheduler_config.async_scheduling:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronAsyncScheduler"
                )
            else:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronScheduler"
                )
        else:
            logger.warning(
                "scheduler_cls already set to non-default: %s, "
                "NOT overriding with custom Neuron scheduler",
                scheduler_config.scheduler_cls,
            )

    @classmethod
    def validate_request(
        cls,
        processed_inputs: "EngineInput",
        params: "SamplingParams | PoolingParams",
    ) -> None:
        """Reject requests unsupported by the current Neuron server config.

        Called per-request before scheduling. Raises ValueError before model
        execution so SO-off servers do not enter the SO mask path.
        """
        if (
            not cls._enable_structured_outputs
            and getattr(params, "structured_outputs", None) is not None
        ):
            raise ValueError(SO_DISABLED_MESSAGE)

        if cls._max_embeds_per_image is None:
            return

        mm_placeholders = processed_inputs.get("mm_placeholders")
        if not mm_placeholders:
            return

        for modality, positions in mm_placeholders.items():
            for mm_pos in positions:
                # Use get_num_embeds(), not length: a video placeholder span
                # interleaves timestamp TEXT tokens with the vision embeds
                # (is_embed mask), so length overcounts what the vision
                # encoder actually produces. For images they are equal.
                num_embeds = mm_pos.get_num_embeds()
                if num_embeds > cls._max_embeds_per_image:
                    raise ValueError(
                        f"One {modality} item in this request produces "
                        f"{num_embeds} embedding tokens. This value is more "
                        f"than the maximum that the compiled encoder "
                        f"supports ({cls._max_embeds_per_image}). Send a "
                        f"smaller {modality} item, or increase "
                        f"num_encoder_input_buckets in "
                        f"encoder_neuron_config. For an image, you can also "
                        f"set max_pixels in mm_processor_kwargs."
                    )

    @classmethod
    def _validate_quantization_config(cls, vllm_config) -> None:
        """Validate quantization config. Only KV cache quantization
        (q_scale/k_scale/v_scale) is supported for compressed-tensors, except on
        Neuron CPU-dequant paths (e.g. quantization="mxfp8") where the device
        never sees on-device weight/activation quant."""
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        if neuron_config.get("quantization") in cls._cpu_dequant_quantizations:
            return
        model_config = vllm_config.model_config
        quant_cfg = getattr(model_config.hf_config, "quantization_config", None)
        if not quant_cfg or quant_cfg.get("quant_method") != "compressed-tensors":
            return
        for group_name, group in quant_cfg.get("config_groups", {}).items():
            if group.get("weights"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Weight quantization "
                    f"in config_groups['{group_name}'] is not supported."
                )
            if group.get("output_activations"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Output activation "
                    f"quantization in config_groups['{group_name}'] is not "
                    f"supported."
                )
            targets = group.get("targets", [])
            if group.get("input_activations") and not any(
                "Attention" in t for t in targets
            ):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Input activation "
                    f"quantization on non-attention targets in "
                    f"config_groups['{group_name}'] is not supported."
                )

    @staticmethod
    def _has_neuron_component_dp(vllm_config) -> bool:
        """Check if any Neuron component DP size > 1 in additional_config."""
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        return (
            neuron_config.get("attention_dp_size", 1) > 1
            or neuron_config.get("embedding_dp_size", 1) > 1
            or neuron_config.get("mlp_dp_size", 1) > 1
            or neuron_config.get("draft_mlp_dp_size", 1) > 1
            or neuron_config.get("lm_head_dp_size", 1) > 1
        )

    @classmethod
    def _maybe_enable_moe_for_component_dp(cls, vllm_config) -> None:
        """Mark model as MoE when component DP needs cross-DP process groups.

        vLLM's run_engine_core only keeps data_parallel_size > 1 for MoE models.
        Component DP creates cross-DP process groups that require all DP workers
        in the same torch distributed world. Setting num_experts=1 causes
        run_engine_core to take the DPEngineCoreProc path.
        """
        if (
            vllm_config.parallel_config.data_parallel_size > 1
            and not vllm_config.model_config.is_moe
            and cls._has_neuron_component_dp(vllm_config)
        ):
            vllm_config.model_config.model_arch_config.num_experts = 1
            logger.info(
                "Component DP enabled — setting num_experts=1 to preserve "
                "data_parallel_size for cross-DP process groups"
            )

    @classmethod
    def _validate_component_dp_config(cls, vllm_config) -> None:
        """Validate component DP (attention/embedding/mlp/lm_head DP) constraints.

        Component DP is decode-only and requires disaggregated inference (DI)
        with data_parallel_size > 1. Single-node non-DI serving is not supported.
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        dp_configs = {
            "attention_dp_size": neuron_config.get("attention_dp_size", 1),
            "embedding_dp_size": neuron_config.get("embedding_dp_size", 1),
            "mlp_dp_size": neuron_config.get("mlp_dp_size", 1),
            "draft_mlp_dp_size": neuron_config.get("draft_mlp_dp_size", 1),
            "lm_head_dp_size": neuron_config.get("lm_head_dp_size", 1),
        }
        active = {k: v for k, v in dp_configs.items() if v > 1}
        if not active:
            return

        # On a MoE model the experts shard with EP, so mlp_dp_size only
        # duplicates the rows of the static cross-DP-EP dispatch. The all2all-v
        # dispatch is exempt: it keeps the rows sequence parallel and never
        # gathers over the DP group.
        expert_parallel = (
            vllm_config.parallel_config.enable_expert_parallel
            or neuron_config.get("ep_degree", 1) > 1
        )
        all2all = neuron_config.get("all2all_backend") == "neuron"
        if dp_configs["mlp_dp_size"] > 1 and expert_parallel and not all2all:
            raise ValueError(
                f"mlp_dp_size ({dp_configs['mlp_dp_size']}) must be 1 with "
                "expert parallelism and the static cross-DP-EP dispatch. A MoE "
                "model shards its experts with EP, so mlp_dp_size only "
                "duplicates the dispatch rows. Set draft_mlp_dp_size to shard "
                'the Eagle3 draft MLP instead, or set all2all_backend="neuron".'
            )

        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            raise ValueError(
                f"Component DP ({active}) requires disaggregated inference "
                f"(--kv-transfer-config). Single-node serving without DI is "
                f"not supported with component DP."
            )

        kv_role = kv_config.kv_role
        if kv_role not in ("kv_consumer", "kv_both"):
            raise ValueError(
                f"Component DP ({active}) is only supported on the decode "
                f"server (kv_role='kv_consumer' or 'kv_both'), got "
                f"kv_role='{kv_role}'. The prefill server should not set "
                f"component DP sizes > 1."
            )

        if vllm_config.parallel_config.data_parallel_size < 2:
            raise ValueError(
                f"Component DP ({active}) requires --data-parallel-size >= 2. "
                f"Component DP shards modules across DP replicas and needs "
                f"multiple DP ranks to function."
            )

    @classmethod
    def _validate_dcp_config(cls, vllm_config) -> None:
        """Validate all constraints for decode_context_parallel_size > 1."""
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            return

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        num_q_heads = vllm_config.model_config.hf_config.num_attention_heads

        block_size = vllm_config.cache_config.block_size
        long_prefill_token_threshold = (
            vllm_config.scheduler_config.long_prefill_token_threshold
        )
        ownership_stride = dcp_size * block_size
        if (
            long_prefill_token_threshold > 0
            and long_prefill_token_threshold % ownership_stride != 0
        ):
            raise ValueError(
                "DCP requires long_prefill_token_threshold to be divisible by "
                "decode_context_parallel_size * block_size "
                f"({dcp_size} * {block_size} = {ownership_stride}); got "
                f"{long_prefill_token_threshold}."
            )

        # DI is optional: prefill and decode share one layout/semantics, so DCP
        # runs single-node. If a connector IS set (P/D split) it must be Nixl.
        kv_config = vllm_config.kv_transfer_config
        if kv_config is not None and kv_config.kv_connector != "NeuronNixlConnector":
            raise ValueError(
                "DCP requires kv_connector='NeuronNixlConnector'. "
                'Set "kv_connector": "NeuronNixlConnector" in --kv-transfer-config'
            )

        # Each KV head must be replicated across a complete DCP group, and its
        # GQA query-head group must split evenly across those ranks.
        if tp_size < dcp_size * num_kv_heads:
            raise ValueError(
                "DCP requires tensor_parallel_size >= "
                "decode_context_parallel_size * num_kv_heads "
                f"(tp={tp_size}, dcp={dcp_size}, num_kv_heads={num_kv_heads}; "
                f"need tp >= {dcp_size * num_kv_heads}). KV must be replicated "
                "across the DCP group so each group member shares a KV head."
            )

        gqa_ratio = num_q_heads // num_kv_heads
        if gqa_ratio % dcp_size != 0:
            raise ValueError(
                "DCP requires (num_q_heads // num_kv_heads) % dcp == 0 "
                f"(gqa_ratio={gqa_ratio}, dcp={dcp_size})."
            )

        # DCP groups cannot cover every TP rank unless the sizes divide evenly.
        if tp_size % dcp_size != 0:
            raise ValueError(
                f"DCP requires tp % decode_context_parallel_size == 0 so the replica "
                f"mesh covers every rank (tp={tp_size}, dcp={dcp_size})"
            )

        # Neuron DCP requires the KV-cache interleave granularity to equal the
        # physical block size (the segmented-CP kernel and the slot-mapping
        # ownership math both assume block-granular interleaving). The default is
        # applied in update_block_size_for_backend (which runs in the worker,
        # where block_size is final); here we only reject an explicit, conflicting
        # value. The default sentinel (1) is left alone so the not-yet-defaulted
        # early passes of check_and_update_config don't false-positive — the
        # worker's update_block_size_for_backend replaces it with block_size.
        block_size = vllm_config.cache_config.block_size
        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        if (
            interleave_size != _UPSTREAM_DCP_INTERLEAVE_DEFAULT
            and interleave_size != block_size
        ):
            raise ValueError(
                "DCP requires cp_kv_cache_interleave_size to equal the physical "
                f"KV cache block size (interleave={interleave_size}, "
                f"block_size={block_size}). Omit --cp-kv-cache-interleave-size to "
                "default it to block_size, or set it to match --block-size."
            )

        # TEMPORARY: DCP ownership is computed from one active position per request.
        # Lifting this needs that math generalized to a token span, not just this
        # gate deleted (with test_dcp_with_speculative_decoding_rejected).
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DCP does not support speculative decoding. Disable speculative "
                "decoding or set decode_context_parallel_size=1."
            )

        # NOTE: whether a DCP prefill can consume an FP8 KV cache is model
        # specific (it depends on the attention kernel the model routes to and
        # whether that model wrote a packed layout), so it is NOT validated here
        # in the shared DCP config check. Each model enforces its own rule in its
        # modeling code — e.g. llama3 calls
        # ``_validate_dcp_prefill_kv_cache`` (rejects FP8 unless packed).

        if tp_size > num_q_heads:
            logger.warning(
                "tensor_parallel_size (%d) > num_attention_heads (%d). "
                "Multiple ranks will share the same Q heads, wasting attention "
                "computation FLOPs. This may still be useful to shard the KV "
                "sequence across more DCP ranks for long-context workloads.",
                tp_size,
                num_q_heads,
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        import torch

        torch.manual_seed(seed)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_neuron.parallel.neuron_communicator.NeuronDeviceCommunicator"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        logger.info("Using NEURON custom attention backend.")
        return "vllm_neuron.vllm.attention.attn.NeuronAttentionBackend"

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        # "cuda" here means device. Neuron DI uses device-to-device
        # direct transfer.
        return {NeuronPlatform.device_type: {"cuda"}}

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """
        Whether the platform supports hybrid KV cache with multiple cache groups.

        When True, models with mixed attention types (e.g., full attention and
        sliding window attention) will use separate KV cache groups, enabling
        memory optimization by freeing blocks outside the sliding window.

        Returns:
            True to enable hybrid KV cache support on Neuron.
        """
        return True

    @classmethod
    def device_count(cls) -> int:
        """
        Gets the visible device count. This returns the configured device count
        if set, otherwise falls back to the number of visible devices returned
        by the torch runtime.
        """
        if cls._device_count != -1:
            return cls._device_count

        runtime = torch.classes.neuron.Runtime()
        nc_count = runtime.get_nc_count()
        if nc_count == -1:
            raise RuntimeError(
                "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"
            )
        return nc_count

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def set_device_count(cls, device_count: int) -> None:
        """
        Sets the visible device count.

        In the new vLLM Neuron backend, each worker thread has a single visible device,
        so the torch runtime can only see one device. This is a problem because
        vLLM expects that the device count returns the number of devices visible
        to a node, such as when initializing MultiprocExecutor for a multi-node
        server. To fix that issue, each worker thread sets the device count to
        the number of visible devices on the local node.
        """
        cls._device_count = device_count

    @classmethod
    def _auto_set_neuron_connector_module_path(cls, vllm_config) -> None:
        """Auto-inject kv_connector_module_path for NeuronNixlConnector.

        The factory needs module_path to locate the class in worker subprocesses.
        We set it here so users only need to specify kv_connector name.
        """
        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            return
        if kv_config.kv_connector == "NixlConnector":
            # The base connector cannot register Neuron's K/V-slab cache
            # layout (it assumes one packed region per layer); the Neuron
            # subclass registers two regions per layer. Redirect so every
            # nixl deployment gets the layout-aware connector.
            kv_config.kv_connector = "NeuronNixlConnector"
            logger.info(
                "Neuron platform: redirecting kv_connector NixlConnector "
                "-> NeuronNixlConnector (Neuron K/V-slab cache layout)"
            )
        if (
            kv_config.kv_connector == "NeuronNixlConnector"
            and not kv_config.kv_connector_module_path
        ):
            kv_config.kv_connector_module_path = (
                "vllm_neuron.vllm.kv_connector.neuron_nixl_connector"
            )
        if (
            kv_config.kv_connector == "NeuronDecodeBenchConnector"
            and not kv_config.kv_connector_module_path
        ):
            kv_config.kv_connector_module_path = (
                "vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector"
            )

    @classmethod
    def _auto_set_neuron_ec_connector_module_path(cls, vllm_config) -> None:
        """Auto-inject ec_connector_module_path for NeuronNixlECConnector.

        The EC connector factory imports the class by module path in worker
        subprocesses (it is not in the built-in registry), so we set it here when
        the user selects the connector by name. No-op when no EC transfer is
        configured (the default monolith path).
        """
        ec_config = getattr(vllm_config, "ec_transfer_config", None)
        if ec_config is None:
            return
        if (
            ec_config.ec_connector == "NeuronNixlECConnector"
            and not ec_config.ec_connector_module_path
        ):
            ec_config.ec_connector_module_path = (
                "vllm_neuron.vllm.ec_connector.neuron_nixl_ec_connector"
            )

    @classmethod
    def _patch_termination_timeouts(cls) -> None:
        """Patch vLLM process termination to use a configurable timeout.

        Upstream vLLM has two hardcoded SIGTERM-to-SIGKILL timeout paths:

        1. ``v1/utils.py:shutdown()`` (5 s) — Main process kills the
           EngineCore process tree via ``kill_process_tree()``.
        2. ``MultiprocExecutor._ensure_worker_termination()`` (4 s) —
           EngineCore kills individual worker processes.

        When Neuron profiling is enabled (``NEURON_RT_INSPECT_ENABLE=1``),
        the runtime needs extra time after SIGTERM to flush profiling data
        to disk.  This method monkey-patches both paths so the timeout is
        controlled by ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT`` (default
        5 s, matching upstream).

        For path (1), ``v1/engine/utils.py`` does
        ``from vllm.v1.utils import shutdown`` which creates a module-level
        binding later captured by ``weakref.finalize``.  We must replace
        the name in **both** modules so the finalizer gets our version.
        """
        if cls._termination_timeout_patched:
            return
        cls._termination_timeout_patched = True

        timeout = envs.VLLM_NEURON_WORKER_TERMINATION_TIMEOUT

        if timeout == 5 and os.getenv("NEURON_RT_INSPECT_ENABLE") == "1":
            logger.warning(
                "Neuron profiling is enabled but VLLM_NEURON_WORKER_TERMINATION_TIMEOUT "
                "is at default (5s). Profile data may be incomplete. "
                "Consider setting VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60."
            )
            return

        if timeout == 5:
            return

        logger.info(
            "Patching vLLM termination timeouts to %ss (pid=%s)", timeout, os.getpid()
        )
        cls._patch_shutdown(timeout)
        cls._patch_ensure_worker_termination(timeout)

    @classmethod
    def _patch_shutdown(cls, neuron_timeout: int) -> None:
        """Patch ``v1.utils.shutdown`` (5 s → *timeout*)."""
        import vllm.v1.engine.utils as engine_utils
        import vllm.v1.utils as v1_utils
        from vllm.utils.system_utils import kill_process_tree

        def _shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
            logger.info(
                "patched shutdown: pid=%s, timeout=%ss, %s proc(s)",
                os.getpid(),
                timeout,
                len(procs),
            )
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            timeout = max(neuron_timeout, timeout) if timeout else neuron_timeout
            deadline = _time.monotonic() + timeout
            for proc in procs:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if proc.is_alive():
                    proc.join(remaining)

            for proc in procs:
                if proc.is_alive() and (pid := proc.pid) is not None:
                    logger.warning(
                        "%s (pid=%s) still alive after %ss — sending SIGKILL",
                        proc.name,
                        pid,
                        timeout,
                    )
                    kill_process_tree(pid)

        v1_utils.shutdown = _shutdown
        engine_utils.shutdown = _shutdown

    @classmethod
    def _patch_ensure_worker_termination(cls, timeout: int) -> None:
        """Patch ``MultiprocExecutor._ensure_worker_termination`` (4 s → *timeout*)."""
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        @staticmethod  # type: ignore[misc]
        def _ensure_worker_termination(worker_procs: list[BaseProcess]) -> None:
            active = [p for p in worker_procs if p.is_alive()]
            logger.info(
                "patched _ensure_worker_termination: pid=%s, timeout=%ss, %s worker(s)",
                os.getpid(),
                timeout,
                len(active),
            )
            for p in active:
                p.terminate()

            deadline = _time.monotonic() + timeout
            for p in active:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if p.is_alive():
                    p.join(remaining)

            still_alive = [p for p in active if p.is_alive()]
            if still_alive:
                for p in still_alive:
                    logger.warning(
                        "Worker %s (pid=%s) still alive after %ss — sending SIGKILL",
                        p.name,
                        p.pid,
                        timeout,
                    )
                    p.kill()

        MultiprocExecutor._ensure_worker_termination = _ensure_worker_termination

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        """Map logical device ID to physical Neuron core ID."""
        if cls.device_control_env_var in os.environ:
            # Parse NEURON_VISIBLE_DEVICES range (e.g., "0-7")
            core_range = os.environ[cls.device_control_env_var]
            if "-" in core_range:
                start, end = map(int, core_range.split("-"))
                available_cores = list(range(start, end + 1))
            else:
                # Single core or comma-separated list
                available_cores = [int(c.strip()) for c in core_range.split(",")]
            logger.debug(
                "device id: %s, physical device id: %s",
                device_id,
                available_cores[device_id],
            )
            return available_cores[device_id]
        return device_id
