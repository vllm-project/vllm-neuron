# SPDX-License-Identifier: Apache-2.0
"""A Neuron worker class."""

from typing import Set

import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.torch_utils import set_random_seed
from vllm.tasks import GenerationTask, SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput, GrammarOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)


class NeuronWorker(WorkerBase):
    """A worker class that executes the model on a group of neuron cores."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        # In vLLM 0.16.0+, verify_with_parallel_config runs before check_and_update_config from platform.py so applying patches here ensures they're active
        from vllm_neuron.platform import NeuronPlatform

        NeuronPlatform._apply_config_overrides()

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from transformers.dynamic_module_utils import init_hf_modules

            init_hf_modules()
        self.device = self.device_config.device
        self.model_runner = self.get_neuronx_distributed_model_runner(
            vllm_config, self.device
        )
        self.vllm_config = vllm_config

    def init_device(self) -> None:
        self.init_distributed_environment(self.vllm_config)

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def _query_runtime_memory(self) -> int:
        """Query Neuron runtime for available memory."""
        import torch

        try:
            rt = torch.classes.neuron.Runtime()
            bytes_used, bytes_free = rt.get_vnc_memory_stats()
            if bytes_used == -1 and bytes_free == -1:
                raise RuntimeError("Failed to initialize neuron runtime")
            logger.debug(
                "Device memory stats: %s bytes used, %s bytes free (%.2f GB free)",
                bytes_used,
                bytes_free,
                bytes_free / (1024**3) if bytes_free > 0 else 0,
            )
            return bytes_free
        except Exception as e:
            logger.debug("Failed to get memory stats: %s", e)
            return 20 * 1024**3  # 20GB fallback

    def _calculate_contiguous_kv_memory(self) -> int:
        """
        Calculate memory size for contiguous KV cache blocks.

        Returns memory corresponding to max_num_seqs + 1 blocks,
        where +1 accounts for vLLM's null_block reservation.
        """
        from vllm.utils.torch_utils import get_dtype_size
        from vllm_neuron.worker.utils import get_num_layers_from_hf_config

        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        num_blocks = max_num_seqs + 1  # +1 for null_block

        num_layers = get_num_layers_from_hf_config(self.model_config.hf_config)
        block_size = self.cache_config.block_size
        head_size = self.model_config.get_head_size()
        dtype_size = get_dtype_size(self.model_config.dtype)
        num_kv_heads = self.parallel_config.tensor_parallel_size

        page_size_per_layer = 2 * block_size * num_kv_heads * head_size * dtype_size
        available_memory = num_blocks * num_layers * page_size_per_layer

        logger.debug(
            "Contiguous KV cache: %d bytes (%.2f GB) for %d blocks",
            available_memory,
            available_memory / (1024**3),
            num_blocks,
        )
        return available_memory

    def determine_available_memory(self):
        """
        For block KV layout, we query Neuron runtime for available memory.
        When running models using a contiguous KV cache layout on Neuron, we need
        exactly max_num_seqs number of blocks, where each block size is equal to
        max_model_len. Thus we manually calculate the memory size that this many
        blocks would correspond to.

        """
        if self.model_runner.is_block_kv_layout:
            return self._query_runtime_memory()

        return self._calculate_contiguous_kv_memory()

    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | None:
        output = self.model_runner.execute_model(scheduler_output)

        return output if self.is_driver_worker or has_kv_transfer_group() else None

    def profile(self, is_start: bool = True):
        raise NotImplementedError

    def get_neuronx_distributed_model_runner(self, vllm_config, device):
        from vllm_neuron.worker.neuronx_distributed_model_runner import (
            NeuronxDistributedModelRunner,
        )

        return NeuronxDistributedModelRunner(vllm_config=vllm_config, device=device)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def load_model(self):
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # Skip for NeuronX Distributed Inference
        return None

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def init_distributed_environment(self, vllm_config: VllmConfig):
        """
        vLLM still needs the environment initialized when TP/PP > 1
        """

        from vllm.config import ParallelConfig

        from vllm_neuron.platform import NeuronPlatform

        NeuronPlatform._apply_world_size_override(ParallelConfig)

        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        ensure_model_parallel_initialized(
            1,
            1,
        )

        ensure_kv_transfer_initialized(vllm_config)

    def get_kv_connector_handshake_metadata(self):
        """Return NIXL handshake metadata for KV connector discovery.

        The V1 engine calls this during startup to exchange NIXL agent
        metadata between prefill and decode workers. Without this, the
        workers cannot discover each other for KV cache transfer.
        """
        from vllm.distributed.kv_transfer import (
            has_kv_transfer_group,
            get_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_tp_group
        if not has_kv_transfer_group():
            return None
        connector = get_kv_transfer_group()
        metadata = connector.get_handshake_metadata()
        if metadata is None:
            return None
        return {get_tp_group().rank_in_group: metadata}

    def shutdown(self) -> None:
        self.model_runner.ensure_kv_transfer_shutdown()

    # vLLM uses add_adapter() to add LoRA, rather than add_lora()
    def add_adapter(self, lora_request: LoRARequest) -> bool:
        # Todo: support dynamic add/remove lora
        return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        supported_tasks = list[GenerationTask]()
        supported_tasks.append("generate")
        return supported_tasks

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput:
        """
        Sample tokens with grammar constraints.
        Called by executor after execute_model() returns None.
        """
        return self.model_runner.sample_tokens(grammar_output)

    def execute_dummy_batch(self) -> None:
        """Execute a dummy batch for engine initialization and warmup."""
        self.model_runner._dummy_run(1, uniform_decode=True)
