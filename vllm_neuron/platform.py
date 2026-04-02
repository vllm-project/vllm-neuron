# SPDX-License-Identifier: Apache-2.0
import enum
import importlib.metadata
import logging
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    VllmConfig = None
    FlexibleArgumentParser = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronFramework(enum.Enum):
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"


class _Backend(enum.Enum):
    # Attention backend supported by Neuron
    NEURON_ATTN = enum.auto()


class NeuronPlatform(Platform):
    # OOT (Out-Of-Tree) refers to hardware platforms that are integrated with vLLM
    # through plugin projects, rather than being part of the vLLM core.
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = ["neuron_quant", "fbgemm_fp8"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    # Guards to ensure overrides are only applied once per process
    _config_overrides_applied = False
    _openai_overrides_applied = False
    _engine_client_override_applied = False

    @classmethod
    def _apply_world_size_override(cls, ParallelConfig) -> None:
        """
        Override world_size_across_dp property by replacing it with a method
        that checks for Neuron platform and returns the correct value.
        This approach works at the class level and is timing-independent.
        """
        # Store the original property getter
        original_property = getattr(ParallelConfig, "world_size_across_dp", None)
        if original_property is None:
            logger.warning("world_size_across_dp property not found on ParallelConfig")
            return

        # Get the original getter function
        original_getter = (
            original_property.fget if hasattr(original_property, "fget") else None
        )
        if original_getter is None:
            logger.warning(
                "Could not get original getter for world_size_across_dp property"
            )
            return

        def neuron_world_size_across_dp(self) -> int:
            """Neuron-aware world_size_across_dp implementation."""
            from vllm.platforms import current_platform

            # Check if we're on Neuron platform
            if (
                hasattr(current_platform, "__class__")
                and current_platform.__class__.__name__ == "NeuronPlatform"
            ):
                logger.info(
                    "Using Neuron override: world_size_across_dp returning data_parallel_size"
                )
                return self.data_parallel_size

            # Fall back to original implementation for other platforms
            return original_getter(self)

        # Replace the property with our Neuron-aware version
        ParallelConfig.world_size_across_dp = property(neuron_world_size_across_dp)

        logger.debug(
            "Successfully applied world_size_across_dp override via class-level property replacement"
        )

    @classmethod
    def _apply_config_overrides(cls) -> None:
        """
        Apply Neuron config overrides to ModelConfig.
        Must be called before VllmConfig is instantiated.
        Safe to call multiple times - will only apply once per process.
        """
        if cls._config_overrides_applied:
            return

        try:
            from vllm.config import ModelConfig, ParallelConfig

            from vllm_neuron import platform_overrides

            logger.info("Applying Neuron config overrides")

            ModelConfig.verify_with_parallel_config = (
                platform_overrides.skip_verify_with_parallel_config
            )
            ModelConfig._verify_quantization = (
                platform_overrides.skip_verify_quantization
            )
            ModelConfig._verify_cuda_graph = platform_overrides.skip_verify_cuda_graph
            ModelConfig.get_and_verify_max_len = (
                platform_overrides.changed_get_and_verify_max_len
            )
            # Apply the world_size_across_dp override using a different approach
            cls._apply_world_size_override(ParallelConfig)

            cls._config_overrides_applied = True
            logger.info("Neuron config overrides applied successfully")

        except ImportError as e:
            logger.warning("Could not import vLLM config module for overrides: %s", e)
        except Exception as e:
            logger.error("Error applying Neuron config overrides: %s", e)
            raise

    @classmethod
    def _get_vllm_version(cls) -> str:
        """Get the current vLLM version."""
        try:
            return importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not determine vLLM version")
            return "unknown"

    @classmethod
    def _apply_openai_overrides(cls) -> None:
        """
        Apply OpenAI serving overrides.
        Must be called after vLLM is fully initialized to avoid circular imports.
        """
        if cls._openai_overrides_applied:
            return

        try:
            from vllm_neuron import platform_overrides

            (
                OpenAIServing,
                changed_validate_input,
                CompletionRenderer,
                changed_create_tokens_prompt,
            ) = platform_overrides.get_openai_overrides()

            # for v1/chat/completions
            OpenAIServing._validate_input = changed_validate_input
            # for v1/completions
            CompletionRenderer._create_tokens_prompt = changed_create_tokens_prompt

            cls._openai_overrides_applied = True
            logger.info("Neuron OpenAI serving overrides applied successfully")

        except ImportError as e:
            logger.warning(
                "Could not import OpenAI serving modules for overrides: %s", e
            )
        except Exception as e:
            logger.error("Error applying OpenAI serving overrides: %s", e)

    @classmethod
    def _apply_engine_client_override(cls) -> None:
        """
        Patch EngineCoreClient.make_async_mp_client to use AsyncMPClient
        for external DP load balancing.

        Workaround for vLLM v0.16.0 bug: DPAsyncMPClient is selected for
        external LB but requires a DPCoordinator, which is not started for
        external LB with non-MoE models.
        """
        if cls._engine_client_override_applied:
            return

        try:
            from vllm.v1.engine.core_client import (
                AsyncMPClient,
                EngineCoreClient,
            )

            _original = EngineCoreClient.make_async_mp_client

            @staticmethod
            def _patched_make_async_mp_client(
                vllm_config,
                executor_class,
                log_stats,
                client_addresses=None,
                client_count=1,
                client_index=0,
            ):
                pc = vllm_config.parallel_config
                if pc.data_parallel_size > 1 and pc.data_parallel_external_lb:
                    logger.info(
                        "Neuron override: Using AsyncMPClient for external "
                        "DP load balancing (dp_size=%d, dp_rank=%d)",
                        pc.data_parallel_size,
                        pc.data_parallel_rank,
                    )
                    return AsyncMPClient(
                        vllm_config,
                        executor_class,
                        log_stats,
                        client_addresses,
                        client_count,
                        client_index,
                    )
                return _original(
                    vllm_config,
                    executor_class,
                    log_stats,
                    client_addresses,
                    client_count,
                    client_index,
                )

            EngineCoreClient.make_async_mp_client = _patched_make_async_mp_client

            cls._engine_client_override_applied = True
            logger.info("Neuron engine client override applied successfully")

        except ImportError as e:
            logger.warning("Could not import engine core client for override: %s", e)
        except Exception as e:
            logger.error("Error applying engine client override: %s", e)
            raise

    @classmethod
    def _get_vllm_version(cls) -> str:
        """Get the current vLLM version."""
        try:
            return importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not determine vLLM version")
            return "unknown"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink=False,
        use_sparse=False,
        **kwargs,
    ) -> str:
        if selected_backend != _Backend.NEURON_ATTN:
            logger.warning(
                "Cannot use %s backend on Neuron. Switch to NEURON_ATTN instead",
                selected_backend,
            )

        logger.info("Using NEURON_ATTN backend.")
        return "vllm_neuron.attention.neuron_attn.NeuronAttentionBackend"

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        # Apply config overrides here - this is called before VllmConfig is created
        cls._apply_config_overrides()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Ensure config overrides are applied (in case pre_register_and_update wasn't called)
        cls._apply_config_overrides()
        # Apply OpenAI overrides now that vLLM is fully loaded
        cls._apply_openai_overrides()
        # Apply engine client override for external DP
        cls._apply_engine_client_override()

        # As of 0.10.2 check_and_update_config is being called every time
        # a VllmConfig object is created, even a default one, to validate params.
        # Our checks are not compatible with an empty VllmConfig. Currently one
        # of the most obvious signs that a VllmConfig is empty is that the model_config
        # is None, however this isn't guaranteed to be true in future vLLM versions.
        # TODO figure out a better way to verify empty VllmConfigs instead of just
        # checking if the VllmConfig.model_config is None.
        model_config = vllm_config.model_config
        if model_config is None:
            return

        disable_scheduler_override = bool(
            int(os.getenv("DISABLE_NEURON_CUSTOM_SCHEDULER", "0"))
        )

        model_config.max_prompt_length = vllm_config.additional_config.get(
            "max_prompt_length", None
        )

        # Add 1 to num_gpu_blocks_override to account for lazy null block allocation
        cache_config = vllm_config.cache_config
        if (
            cache_config
            and cache_config.num_gpu_blocks_override is not None
            and "_neuron_null_block_adjusted" not in cache_config.__dict__
        ):
            logger.info(
                "Adding 1 to num_gpu_blocks_override (%d -> %d) "
                "to account for null block allocation",
                cache_config.num_gpu_blocks_override,
                cache_config.num_gpu_blocks_override + 1,
            )
            cache_config.num_gpu_blocks_override += 1
            cache_config._neuron_null_block_adjusted = True

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_neuron.worker.neuron_worker.NeuronWorker"

        # Configure executor backend for Neuron devices
        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if disable_scheduler_override:
            logger.warning(
                "The vLLM V1 native scheduler will be used with chunked prefill enabled. "
                "This may lead to suboptimal performance on Neuron devices."
            )
            assert vllm_config.cache_config.block_size is not None, (
                "When vLLM V1 native scheduler is enabled, block_size must be set."
            )
        else:
            logger.info(
                "The custom Neuron scheduler will disable chunked prefill and schedule requests using "
                "the continuous batching mechanism, prioritizing prefill over decode."
            )
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler"
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False

            sched_cfg = vllm_config.scheduler_config

            # Set default token budget for Neuron to 128k
            sched_cfg.max_num_batched_tokens = 131072
            logger.info(
                "Neuron custom scheduler default: max_num_batched_tokens set to %d. "
                "Override with --max-num-batched-tokens if needed.",
                sched_cfg.max_num_batched_tokens,
            )

            # Set default batch size for Neuron to 32
            if not sched_cfg.max_num_seqs:
                sched_cfg.max_num_seqs = 32
                logger.info(
                    "Neuron custom scheduler default: max_num_seqs set to %d.",
                    sched_cfg.max_num_seqs,
                )

            if vllm_config.cache_config is not None:
                if not vllm_config.cache_config.enable_prefix_caching:
                    # Neuron requires block_size = max_model_len when blockwise KV cache is disabled
                    vllm_config.cache_config.block_size = (
                        vllm_config.model_config.max_model_len  # type: ignore
                    )
                else:
                    assert vllm_config.cache_config.block_size is not None, (
                        "When prefix caching is enabled, block_size must be set."
                    )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    @lru_cache
    def is_neuronx_distributed_inference(cls) -> bool:
        try:
            import neuronx_distributed_inference
        except ImportError:
            neuronx_distributed_inference = None
        return neuronx_distributed_inference is not None

    def get_neuron_framework_to_use(self):
        """Return the specified framework if corresponding installations are
        available.

        If no framework is specified, use neuronx-distributed-inference by
        default.
        If that's unavailable, check and switch to transformers-neuronx.
        """
        if not self.is_neuron():
            raise AssertionError(f"Neuron Framework unavailable for platform: {self}")

        nxd_installed = self.is_neuronx_distributed_inference()
        if not nxd_installed:
            raise AssertionError(
                "Unable to import neuronx_distributed_inference. Please verify it is properly installed. "
            )

        return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

    def use_neuronx_distributed(self):
        """
        Return True if the framework determined in get_neuron_framework_to_use()
        is NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, False otherwise. This
        is used to select the Neuron model framework and framework-specific
        configuration to apply during model compilation.
        """
        nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
        return self.get_neuron_framework_to_use() == nxd_framework

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """Neuron supports NIXL KV transfer via CPU host buffer only.

        Trainium/Inferentia device memory cannot be directly registered with
        NIXL (no CUDA driver, no Level Zero). KV cache blocks are staged
        through pinned CPU DRAM and transferred via NIXL LIBFABRIC over
        EFA RDMA -- the same pattern used by TPU and XPU.
        """
        return {NeuronPlatform.device_type: ("cpu",)}

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        """NIXL memory type for Neuron: always DRAM (host-buffered)."""
        return "DRAM"

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        """Map logical device ID to physical Neuron core ID."""
        if cls.device_control_env_var in os.environ:
            # Parse NEURON_RT_VISIBLE_CORES range (e.g., "0-7")
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
