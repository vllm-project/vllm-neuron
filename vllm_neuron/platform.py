# SPDX-License-Identifier: Apache-2.0
import enum
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ParallelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    VllmConfig = None
    FlexibleArgumentParser = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# TEST PR 


class NeuronFramework(enum.Enum):
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"


class NeuronPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = ["neuron_quant", "fbgemm_fp8"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    # Guard to ensure config overrides are only applied once
    _config_overrides_applied = False

    def __init__(self):
        """Initialize NeuronPlatform and ensure config overrides are applied."""
        super().__init__()
        self._ensure_config_overrides_applied()

    @classmethod
    def _ensure_config_overrides_applied(cls) -> None:
        """
        Ensure Neuron config overrides are applied in every process.
        This method can be called from multiple places to guarantee
        overrides are available in both main and spawned processes.
        """
        if cls._config_overrides_applied:
            logger.debug("Neuron config overrides already applied, skipping")
            return

        try:
            from vllm.config import ModelConfig

            logger.info("Applying Neuron config overrides")

            # Skip attention head divisibility check
            def skip_verify_with_parallel_config(
                self,
                parallel_config: "ParallelConfig",
            ) -> None:
                logger.info(
                    "Skipping attention head divisibility check for Neuron platform"
                )
                if parallel_config.distributed_executor_backend == "external_launcher":
                    assert self.seed is not None, (
                        "Seed must be set when using external launcher backend to "
                        "make sure sampling results are the same across workers."
                    )

                if parallel_config.enable_expert_parallel:
                    self._verify_with_expert_parallelism()

                pipeline_parallel_size = parallel_config.pipeline_parallel_size
                if pipeline_parallel_size > 1:
                    if not self.registry.is_pp_supported_model(
                            self.architectures):
                        raise NotImplementedError(
                            "Pipeline parallelism is not supported for this model. "
                            "Supported models implement the `SupportsPP` interface."
                        )

                    if self.use_async_output_proc:
                        self.use_async_output_proc = False

            def skip_verify_quantization(self):
                pass

            def skip_verify_cuda_graph(self):
                pass

            def changed_get_and_verify_max_len(self, max_model_len: int):
                # NOTE: Don't use HF config values like sliding_window
                # to impact max_model_len validation when on Neuron.
                if self.spec_target_max_model_len is not None:
                    return self.spec_target_max_model_len
                return max_model_len

            # Apply the overrides
            ModelConfig.verify_with_parallel_config = skip_verify_with_parallel_config
            ModelConfig._verify_quantization = skip_verify_quantization
            ModelConfig._verify_cuda_graph = skip_verify_cuda_graph
            ModelConfig.get_and_verify_max_len = changed_get_and_verify_max_len

            cls._config_overrides_applied = True
            logger.info("Neuron config overrides applied successfully")

        except ImportError as e:
            logger.warning(
                f"Could not import vLLM config module for overrides: {e}")
        except Exception as e:
            logger.error(f"Error applying Neuron config overrides: {e}")
            raise

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def pre_register_and_update(cls,
                                parser: "FlexibleArgumentParser | None" = None
                                ) -> None:
        # Ensure config overrides are applied
        cls._ensure_config_overrides_applied()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Ensure config overrides are applied in every process
        cls._ensure_config_overrides_applied()

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
            int(os.getenv("DISABLE_NEURON_CUSTOM_SCHEDULER", "0")))

        # Add 1 to num_gpu_blocks_override to account for lazy null block allocation
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.num_gpu_blocks_override is not None \
            and '_neuron_null_block_adjusted' not in cache_config.__dict__:
            logger.info(
                "Adding 1 to num_gpu_blocks_override (%d -> %d) "
                "to account for null block allocation",
                cache_config.num_gpu_blocks_override,
                cache_config.num_gpu_blocks_override + 1)
            cache_config.num_gpu_blocks_override += 1
            cache_config._neuron_null_block_adjusted = True

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm_neuron.worker.neuron_worker.NeuronWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if disable_scheduler_override:
            logger.warning(
                "The vLLM V1 native scheduler will be used with chunked prefill enabled. "
                "This may lead to suboptimal performance on Neuron devices.")
            assert vllm_config.cache_config.block_size is not None, (
                "When vLLM V1 native scheduler is enabled, block_size must be set."
            )
        else:
            logger.info(
                "The custom Neuron scheduler will disable chunked prefill and schedule requests using "
                "the continuous batching mechanism, prioritizing prefill over decode."
            )
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

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

            if not vllm_config.cache_config.enable_prefix_caching:
                # Neuron requires block_size = max_model_len when blockwise KV cache is disabled
                vllm_config.cache_config.block_size = (
                    vllm_config.model_config.max_model_len  # type: ignore
                )
            else:
                assert vllm_config.cache_config.block_size is not None, (
                    "When prefix caching is enabled, block_size must be set.")

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
            raise AssertionError(
                f"Neuron Framework unavailable for platform: {self}")

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
