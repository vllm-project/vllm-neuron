# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import torch
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from neuronx_distributed_inference.modules.lora_serving import (
    LoraServingConfig,
)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.config.vllm import set_current_vllm_config
from vllm.multimodal import BatchedTensorInputs
from vllm.multimodal.inputs import MultiModalFeatureSpec, NestedTensors
from vllm.utils.jsontree import json_map_leaves
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
    GrammarOutput,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    DraftTokenIds,
    ModelRunnerOutput,
    SamplerOutput,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm_neuron.worker.constants import NEURON_MULTI_MODAL_MODELS
from vllm_neuron.worker.neuronx_distributed_model_loader import get_neuron_model
from vllm_neuron.worker.utils import get_num_layers_from_hf_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _mm_kwargs_to_device(
    mm_kwargs: dict | None,
    device: torch.types.Device,
) -> dict:
    """
    Convert multimodal kwargs dict to device by moving all tensors.
    """
    if mm_kwargs is None:
        return {}
    if device is None:
        return dict(mm_kwargs)
    return json_map_leaves(
        lambda x: x.to(device=device, non_blocking=True)
        if isinstance(x, torch.Tensor)
        else x,
        mm_kwargs,
    )


@dataclass(frozen=True)
class ModelInputForNeuron:
    """
    Model input for NeuronX Distributed Inference model runner.
    """

    request_ids: list[str] | None = None
    input_tokens: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    input_block_ids: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    full_context_lens: torch.Tensor | None = None
    computed_context_lens: torch.Tensor | None = None
    sampling_params: torch.Tensor | None = None
    multi_modal_kwargs: BatchedTensorInputs = None
    adapter_ids: str | None = None
    # Boolean tensor to indicate if the request is ready
    # for sampling. Needed by chunked prefill.
    prefill_completion_state: torch.Tensor | None = None


# This class is used for constructing ModelInputForNeuron and
# is not frozen.
@dataclass
class IntermediateInputData:
    request_ids: list[str] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    input_block_ids: list[int] = field(default_factory=list)
    full_context_lens: list[int] = field(default_factory=list)
    computed_context_lens: list[int] = field(default_factory=list)
    slot_mapping: list[int] = field(default_factory=list)
    block_tables: list[int] = field(default_factory=list)
    prefill_completion_state: list[bool] = field(default_factory=list)
    adapter_ids: list[int] = field(default_factory=list)
    multi_modal_kwargs: BatchedTensorInputs = None


class NeuronxDistributedModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    # NEURON has an upper limit on the top_k
    _MAX_NEURON_SAMPLING_TOP_K = 256

    # NOTE: Padding table id for slot mapping, note that this will be
    # used as the block index to update KV cache, so we need to make
    # sure no real tokens are mapped to this block_id, we current
    # assume that block 0 will never be used.
    _SLOT_MAPPING_PAD = -1
    _BLOCK_TABLE_PAD = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device

        self.pin_memory = False
        self.block_size = cache_config.block_size
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.block_size],
        )

        self.requests: dict[str, CachedRequestState] = {}
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}
        self.model = None

        # Initialize KV cache mode flags based on config
        self.is_prefix_caching = cache_config.enable_prefix_caching
        self.is_chunked_prefill = scheduler_config.enable_chunked_prefill
        self.is_block_kv_layout = self.is_prefix_caching or self.is_chunked_prefill

        # The following fields are used to support custom sequence id mapping.
        # The goal is to retain the batch line information for contiguous kv cache.
        # A mapping of vLLM request Id to neuron sequence id.
        self.use_custom_seq_id_mapping = not self.is_chunked_prefill
        self.vllm_req_to_neuron_seq_id_mapping: Dict[str, int] = {}
        # Set of neuron sequence id that are free for use.
        self.free_seq_ids = set(range(self.max_num_reqs))
        self._draft_token_ids = None

        self.num_accepted_tokens = 0  # Total draft tokens accepted
        self.num_draft_tokens = 0  # Total draft tokens proposed

        # Initialize CPU sampler for when on-device sampling is not available
        self.cpu_sampler = Sampler()

        # Vocabulary size for bitmask operations
        self._vocab_size = self.model_config.get_vocab_size()

        # For deferred sampling (structured outputs)
        self._cached_logits: Optional[torch.Tensor] = None
        self._cached_model_input: Optional[ModelInputForNeuron] = None

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput:
        """
        Sample tokens from cached model output.

        Handles both on-device (hardware) sampling and CPU sampling.
        Note: Structured outputs (grammar bitmask) is only supported with CPU sampling.

        Args:
            grammar_output: Contains grammar_bitmask and request IDs for
                        structured output requests. None if no constraints.
        Returns:
            ModelRunnerOutput with sampled tokens
        """
        sample_start = time.perf_counter()

        if self._cached_logits is None:
            raise RuntimeError(
                "sample_tokens() called without prior execute_model(). "
                "Logits must be cached first."
            )

        hidden_states = self._cached_logits
        model_input = self._cached_model_input

        # Clear cache
        self._cached_logits = None
        self._cached_model_input = None

        # Handle both tensor and list cases (multimodal models may return lists)
        if isinstance(hidden_states, torch.Tensor):
            batch_size = hidden_states.shape[0] if hidden_states.dim() > 0 else 0
        else:
            batch_size = len(hidden_states) if hidden_states else 0

        # Check if using on-device sampling
        if self.model.neuron_config.on_device_sampling_config is not None:
            # On-device (hardware) sampling path
            # Note: Structured outputs not supported with on-device sampling
            if (
                grammar_output is not None
                and grammar_output.grammar_bitmask is not None
            ):
                logger.warning(
                    "Grammar bitmask provided but on-device sampling is enabled. "
                    "Structured outputs are not supported with on-device sampling. "
                    "The grammar bitmask will be ignored."
                )

            sampling_start = time.perf_counter()
            sampler_output = self._sample_on_device(hidden_states, model_input)
            sampling_elapsed = (time.perf_counter() - sampling_start) * 1000
            logger.debug("[PERF] on_device_sampling: %.2fms", sampling_elapsed)
        else:
            # CPU sampling path - supports structured outputs
            # Prepare logits (handle prefill completion state and reorder)
            logits = self._prepare_logits_for_sampling(hidden_states, model_input)

            # Apply grammar bitmask if present
            if (
                grammar_output is not None
                and grammar_output.grammar_bitmask is not None
            ):
                bitmask_start = time.perf_counter()
                logits = self._apply_grammar_bitmask_from_output(logits, grammar_output)
                bitmask_elapsed = (time.perf_counter() - bitmask_start) * 1000
                logger.debug("[PERF] apply_grammar_bitmask: %.2fms", bitmask_elapsed)

            # Sample from logits
            sampling_start = time.perf_counter()
            sampling_metadata = self.input_batch.sampling_metadata
            if sampling_metadata is None:
                raise RuntimeError("Sampling metadata not available")

            sampler_output = self.cpu_sampler(logits, sampling_metadata)
            sampling_elapsed = (time.perf_counter() - sampling_start) * 1000
            logger.debug("[PERF] cpu_sampling: %.2fms", sampling_elapsed)

        # Generate model runner output
        output_gen_start = time.perf_counter()
        result = self._generate_model_runner_output(sampler_output)
        output_gen_elapsed = (time.perf_counter() - output_gen_start) * 1000
        logger.debug("[PERF] _generate_model_runner_output: %.2fms", output_gen_elapsed)

        total_elapsed = (time.perf_counter() - sample_start) * 1000
        logger.debug(
            "[PERF] sample_tokens total: %.2fms [batch=%d]", total_elapsed, batch_size
        )

        return result

    def _sample_on_device(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelInputForNeuron,
    ) -> SamplerOutput:
        """
        Perform on-device (hardware) sampling.

        Args:
            hidden_states: Model output tensor
            model_input: Model input containing request information

        Returns:
            SamplerOutput with sampled token IDs
        """
        logger.debug("output from model forward: hidden_states=%s", hidden_states)

        # Mask incomplete prefills with -1 (invalid token ID)
        if model_input.prefill_completion_state is not None:
            for i, state in enumerate(model_input.prefill_completion_state):
                if not state.item():
                    hidden_states[i] = -1

        logger.debug(
            "output after excluding partial prefill results: hidden_states=%s",
            hidden_states,
        )

        # Reorder outputs to match input batch order
        request_id_order = {
            request_id: idx for idx, request_id in enumerate(model_input.request_ids)
        }

        reorder_indices = torch.tensor(
            [request_id_order[request_id] for request_id in self.input_batch.req_ids]
        )

        hidden_states = hidden_states[reorder_indices]

        # Perform hardware sampling
        try:
            logger.debug(
                "Using hardware sampling: on_device_sampling_config is configured"
            )
            result = self.model.sample(logits=hidden_states)
            return result

        except Exception as e:
            logger.error(
                "Hardware sampling failed for requests %s: %s",
                model_input.request_ids,
                str(e),
            )
            raise RuntimeError("Hardware sampling operation failed: %s" % str(e)) from e

    def _apply_grammar_bitmask_from_output(
        self,
        logits: torch.Tensor,
        grammar_output: "GrammarOutput",
    ) -> torch.Tensor:
        """
        Apply grammar bitmask from GrammarOutput to logits.
        Args:
            logits: [batch_size, vocab_size]
            grammar_output: Contains grammar_bitmask and structured_output_request_ids
        Returns:
            Masked logits (new tensor, not in-place modification)
        """
        grammar_bitmask = grammar_output.grammar_bitmask
        grammar_req_ids = grammar_output.structured_output_request_ids

        if grammar_bitmask is None or len(grammar_req_ids) == 0:
            return logits

        # Build mapping from request_id to bitmask index
        grammar_req_id_to_idx = {req_id: i for i, req_id in enumerate(grammar_req_ids)}

        batch_size, vocab_size = logits.shape
        device = logits.device
        logits = logits.clone()

        # For each request in the batch, find its bitmask (if any)
        for batch_idx, req_id in enumerate(self.input_batch.req_ids[:batch_size]):
            bitmask_idx = grammar_req_id_to_idx.get(req_id, -1)

            if bitmask_idx == -1:
                # No grammar constraint for this request
                continue

            # Get the bitmask for this request
            # grammar_bitmask is numpy array: [num_grammar_reqs, bitmask_size]
            req_bitmask = torch.from_numpy(grammar_bitmask[bitmask_idx]).to(device)

            # Unpack and apply bitmask
            logits[batch_idx] = self._apply_packed_bitmask_to_row(
                logits[batch_idx], req_bitmask
            )

        return logits

    def _apply_packed_bitmask_to_row(
        self,
        logits_row: torch.Tensor,  # [vocab_size]
        packed_bitmask: torch.Tensor,  # [bitmask_size] int32
    ) -> torch.Tensor:
        """Apply packed bitmask to a single row of logits."""
        vocab_size = logits_row.shape[0]
        device = logits_row.device

        # Unpack bitmask
        token_indices = torch.arange(vocab_size, device=device)
        word_indices = token_indices // 32
        bit_indices = token_indices % 32

        # Clamp to bitmask size
        bitmask_size = packed_bitmask.shape[0]
        word_indices = word_indices.clamp(max=bitmask_size - 1)

        # Get words and extract bits
        words = packed_bitmask[word_indices]
        bits = (words >> bit_indices) & 1
        valid_mask = bits.bool()

        # Apply mask
        return logits_row.masked_fill(~valid_mask, float("-inf"))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig):
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # For Neuron, we don't need to initialize kv cache memory space here as it is done
        # under the hood in NxD-I library. But we do need to register the kv cache space
        # for kv connector when has_kv_transfer_group() returns True.

        if has_kv_transfer_group():
            nxdi_kv_cache_state = (
                self.model.model.context_encoding_model.model.nxd_model.state
            )
            kv_caches = {}
            logger.info("Registering host memory space")
            for rank, kv_on_tp_rank in enumerate(nxdi_kv_cache_state):
                # The NEURON_COMBINE_KV_ALLOCATIONS is exposed by the NxD library.
                # If enabled, it provides a merged view of a k tensor and its v tensor.
                # KV cache shape becomes (2, ...).
                if os.environ.get("NEURON_COMBINE_KV_ALLOCATIONS", "0") == "1":
                    logger.debug("Using combined kv allocations")
                    for key in kv_on_tp_rank:
                        # Filter only the `combined` KV tensors. Note the `combined` tensors
                        # are not tensor copy, but a view.
                        if key.startswith("kv_mgr.past_key_values.combined"):
                            layer = int(key.rsplit(".", 1)[-1])
                            kv_caches[f"tp{rank}_layer{layer}_kv"] = kv_on_tp_rank[key]
                else:
                    logger.debug("Using separate kv allocations")
                    # The vanilla NxDI kv state is a dict with the keys formulated as the following
                    # ['kv_mgr.past_key_values.0', 'kv_mgr.past_key_values.1', ... ].
                    # where K tensor and V tensor are alternating.

                    # Sort keys by numeric suffix so 0,1,2,3,... then loop through pairs of (0,1), (2,3)...
                    # so that we know if it is a K tensor or V tensor.
                    keys = sorted(
                        kv_on_tp_rank.keys(),
                        key=lambda name: int(name.rsplit(".", 1)[-1]),
                    )
                    if len(keys) % 2 != 0:
                        raise ValueError(
                            f"Uneven KV keys on tp rank {rank}: {len(keys)} keys"
                        )
                    for layer, (k_key, v_key) in enumerate(zip(keys[0::2], keys[1::2])):
                        kv_caches[f"tp{rank}_layer{layer}_k"] = kv_on_tp_rank[k_key]
                        kv_caches[f"tp{rank}_layer{layer}_v"] = kv_on_tp_rank[v_key]

            # P2/P9/P10: Move KV tensors to CPU and reshape NxDI BHSD layout
            # into CUDA paged block format for NIXL compatibility.
            #
            # NxDI produces K,V tensors of shape (batch, kv_heads, seq_len, head_dim).
            # NIXL/NixlConnector expects CUDA FlashAttention paged format:
            #   (2, num_blocks, block_size, kv_heads, head_dim) per layer.
            #
            # Also aligns the number of blocks to kv_cache_config.num_blocks
            # (the scheduler's block pool) to prevent:
            #   makeXferReq: remote index out of range
            import torch as _torch
            _block_size = self.cache_config.block_size
            _num_gpu_blocks = kv_cache_config.num_blocks
            _aligned_kv = {}
            _layer_k = {}
            _layer_v = {}

            logger.info(
                "NIXL KV reshape: block_size=%d, num_gpu_blocks=%d (scheduler pool)",
                _block_size, _num_gpu_blocks
            )

            for key, tensor in kv_caches.items():
                t = tensor.cpu().contiguous() if str(tensor.device) != 'cpu' else tensor.contiguous()
                if t.dim() == 4:
                    batch, kv_heads, seq_len, head_dim = t.shape
                    assert batch == 1, f"Expected batch=1, got {batch}"
                    t = t.squeeze(0)
                    _nxdi_blocks = seq_len // _block_size
                    assert seq_len == _nxdi_blocks * _block_size, \
                        f"seq_len {seq_len} not divisible by block_size {_block_size}"
                    t = t.reshape(kv_heads, _nxdi_blocks, _block_size, head_dim)
                    t = t.permute(1, 2, 0, 3).contiguous()
                    if _nxdi_blocks < _num_gpu_blocks:
                        padded = _torch.zeros(
                            (_num_gpu_blocks, _block_size, kv_heads, head_dim),
                            dtype=t.dtype, device='cpu'
                        )
                        padded[:_nxdi_blocks].copy_(t)
                        t = padded
                    elif _nxdi_blocks > _num_gpu_blocks:
                        t = t[:_num_gpu_blocks].contiguous()

                parts = key.split('_')
                layer_idx = None
                is_k = None
                for part in parts:
                    if part.startswith('layer'):
                        layer_idx = int(part[5:])
                    if part == 'k':
                        is_k = True
                    elif part == 'v':
                        is_k = False
                    elif part == 'kv':
                        is_k = None

                if layer_idx is not None:
                    if is_k is True:
                        _layer_k[layer_idx] = t
                    elif is_k is False:
                        _layer_v[layer_idx] = t
                    else:
                        _aligned_kv[f"layer_{layer_idx}"] = t
                else:
                    _aligned_kv[key] = t

            for layer_idx in sorted(set(list(_layer_k.keys()) + list(_layer_v.keys()))):
                k_t = _layer_k.get(layer_idx)
                v_t = _layer_v.get(layer_idx)
                if k_t is not None and v_t is not None:
                    _aligned_kv[f"layer_{layer_idx}"] = _torch.stack([k_t, v_t], dim=0)
                elif k_t is not None:
                    _aligned_kv[f"layer_{layer_idx}_k"] = k_t
                elif v_t is not None:
                    _aligned_kv[f"layer_{layer_idx}_v"] = v_t

            if _aligned_kv:
                _sample = next(iter(_aligned_kv.values()))
                logger.info(
                    "NIXL KV reshape: %d layers, sample shape=%s, dtype=%s",
                    len(_aligned_kv), _sample.shape, _sample.dtype
                )
            get_kv_transfer_group().register_kv_caches(_aligned_kv)

    def _get_nxdi_lora_config(self):
        """
        Neuron Multi-LoRA serving requires a json file to specify the available LoRA adapters
        on both device and CPU memory. The file is expected to pass via additional_config.override_neuron_config
        and the content is expected to be in the following format:
        {
            "lora-ckpt-dir": "/path/to/lora_adapter_dir",
            "lora-ckpt-paths": {
                "lora_id_1": "path/to/lora_adapter_1",
                "lora_id_2": "path/to/lora_adapter_2"
            },
            "lora-ckpt-paths-cpu": {
                "lora_id_1": "path/to/lora_adapter_1",
                "lora_id_2": "path/to/lora_adapter_2",
                "lora_id_3": "path/to/lora_adapter_3",
                "lora_id_4": "path/to/lora_adapter_4"
            }
        }
        """
        override_neuron_config = self.vllm_config.additional_config.get(
            "override_neuron_config", None
        )
        target_modules = override_neuron_config.pop("target_modules", None)
        lora_ckpt_json = override_neuron_config.pop("lora_ckpt_json", None)
        max_cpu_loras = self.lora_config.max_cpu_loras
        dynamic_multi_lora = (
            os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0") == "1"
            or max_cpu_loras > 0
        )

        return LoraServingConfig(
            max_loras=self.lora_config.max_loras,
            max_lora_rank=self.lora_config.max_lora_rank,
            target_modules=target_modules,
            lora_ckpt_json=lora_ckpt_json,
            max_cpu_loras=max_cpu_loras,
            lora_memory_transpose=False,
            dynamic_multi_lora=dynamic_multi_lora,
            batch_size=self.scheduler_config.max_num_seqs,
            base_model_quantized=override_neuron_config.get("quantized", False),
        )

    def _get_last_token_position(self, state: CachedRequestState) -> int:
        """
        This is used to determine the position ID for the next decode step,
        where we process the last generated token.

        Notes:
            - We calculate position id based on prompt len + total generated
            tokens (by draft and target model).
            - We do not use the request_data.num_computed_tokens from the
            scheduler output because that excludes speculated tokens.
            - This step is necessary to support Neuron's fused speculation feature.

        Args:
            state: The cached request state containing token information.

        Returns:
            int: The 0-indexed position of the last processed token.
        """

        return len(state.prompt_token_ids) + len(state.output_token_ids) - 1

    def load_model(self) -> None:
        # Update LoRA config
        lora_serving_config = None
        if self.lora_config is not None:
            lora_serving_config = self._get_nxdi_lora_config()
        with set_current_vllm_config(self.vllm_config):
            self.model = get_neuron_model(
                self.model_config,
                cache_config=self.cache_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                lora_serving_config=lora_serving_config,
                speculative_config=self.speculative_config,
                additional_config=self.vllm_config.additional_config,
            )
        if self.lora_config is not None:
            self.lora_manager = self.model.model.lora_model_manager
        self.model.is_reorder_needed = not (
            self.is_prefix_caching or self.is_chunked_prefill
        )

        # Validate and log sampling configuration
        self._validate_sampling_configuration()
        self._validate_max_prompt_length()

    def _validate_sampling_configuration(self) -> None:
        """
        Validate the sampling configuration and log the sampling strategy.

        Raises:
            RuntimeError: If sampling configuration is invalid
        """
        try:
            has_on_device_sampling = (
                hasattr(self.model, "neuron_config")
                and hasattr(self.model.neuron_config, "on_device_sampling_config")
                and self.model.neuron_config.on_device_sampling_config is not None
            )

            if has_on_device_sampling:
                logger.info(
                    "Hardware sampling enabled: config=%s",
                    self.model.neuron_config.on_device_sampling_config,
                )
                # Validate hardware sampling configuration
                config = self.model.neuron_config.on_device_sampling_config
                if hasattr(config, "global_topk") and (
                    config.global_topk <= 0
                    or config.global_topk > self._MAX_NEURON_SAMPLING_TOP_K
                ):
                    logger.warning(
                        "Hardware sampling global_topk=%s is outside the accepted range of [1-%s]. "
                        "Actual topk will be set to %s, the max for neuron.",
                        config.global_topk,
                        self._MAX_NEURON_SAMPLING_TOP_K,
                        self._MAX_NEURON_SAMPLING_TOP_K,
                    )
            else:
                logger.info(
                    "CPU sampling enabled: on_device_sampling_config is None. "
                    "All sampling will be performed on CPU using vLLM's standard sampler."
                )

                # Ensure CPU sampler is available
                if not hasattr(self, "cpu_sampler") or self.cpu_sampler is None:
                    raise RuntimeError(
                        "CPU sampling is required but cpu_sampler is not initialized"
                    )

            # Validate model has required sampling interface
            if not hasattr(self.model, "sample"):
                raise RuntimeError(
                    "Model does not have required 'sample' method for hardware sampling"
                )

        except Exception as e:
            logger.error("Sampling configuration validation failed: %s", str(e))
            raise RuntimeError("Invalid sampling configuration: %s" % str(e)) from e

    def _validate_max_prompt_length(self) -> None:
        """
        Validate that the maximum prompt length configuration is consistent.

        **Terminology clarification:**

        There is a terminology mismatch between NxDI (Neuron) and vLLM:

        - **NxDI/Neuron**: Uses `neuron_config.max_context_length` to specify the
        Maximum Prompt Length (MPL) - the longest prompt sequence the model accepts.
        This can differ from the model's total capacity.

        - **vLLM**: Uses `max_model_len` or 'max_context_len` (MCL) to specify the model's total capacity.
        vLLM has no concept of distinguishing prompt tokens from output tokens. When validating input lengths,
        it treats all tokens the same regardless of whether they are part of the initial
        prompt or generated output tokens from previous iterations. Vllm also considers
        max_model_len and max_context_len (MCL) to be the same thing

        To support Neuron models with MPL ≠ MCL, we expose `max_prompt_length` in
        vLLM's additional_config, which maps to Neuron's `neuron_config.max_context_length`.

        **What this function does:**

        Validates that if the user specifies `max_prompt_length` in
        `additional_config`, it matches the value configured in the Neuron model's
        `neuron_config.max_context_length`. This ensures the API server and
        model loader have consistent prompt length limits.

        **Validation logic:**

        - **If user provides `max_prompt_length`**: Validates it matches the
        neuron model's configuration. Raises error if mismatch.

        - **If user does NOT provide `max_prompt_length`**: Issues a warning if the
        neuron model's max prompt length differs from vLLM's `max_model_len`,
        as this may cause server crashes.

        Raises:
            RuntimeError: If user-provided `max_prompt_length` doesn't match the
                value in `neuron_config.max_context_length`.

        Warns:
            If no `max_prompt_length` is provided and the neuron model's configured
            value differs from `max_model_len`.
        """
        neuron_config = self.model.neuron_config

        def _get_max_prompt_len_from_neuron_config(neuron_config):
            if hasattr(neuron_config, "max_context_length"):
                return neuron_config.max_context_length
            # Bucketing case - find max bucket size
            if getattr(neuron_config, "enable_bucketing", False) and hasattr(
                neuron_config, "context_encoding_buckets"
            ):
                buckets = neuron_config.context_encoding_buckets
                return max(buckets) if buckets else None

            return None

        # mpl_value set in platform.py
        mpl_value = self.vllm_config.model_config.max_prompt_length
        mpl_nc_value = _get_max_prompt_len_from_neuron_config(neuron_config)

        if mpl_value is None:
            if mpl_nc_value != self.max_model_len:
                logger.warning(
                    "Your Neuron model was compiled with max prompt length %s, "
                    "but max_model_len is set to %s. "
                    "To prevent the vLLM engine from crashing when prompts exceed %s tokens, "
                    'add "max_prompt_length": %s to --additional-config when using the '
                    "OpenAI API server. This will return a 400 error for oversized prompts instead of "
                    "terminating the engine. Alternatively, if you need to handle longer prompts, "
                    "you can recompile your Neuron model with a larger max_prompt_length by setting "
                    '"max_context_length": %s in override_neuron_config when compiling.',
                    mpl_nc_value,
                    self.max_model_len,
                    mpl_nc_value,
                    mpl_nc_value,
                    self.max_model_len,
                )
        else:
            if mpl_value != mpl_nc_value:
                raise RuntimeError(
                    f"Configuration mismatch: max_prompt_length in --additional-config ({mpl_value}) "
                    f"does not match the Neuron model's compiled max prompt length ({mpl_nc_value}). "
                    f'Please update --additional-config to set "max_prompt_length": {mpl_nc_value}, '
                    f"or recompile your Neuron model with the desired max prompt length by setting "
                    f'"max_context_length": {mpl_value} in override_neuron_config when compiling.'
                )

        self.max_prompt_length = mpl_nc_value

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | None:
        """
        Execute model forward pass.

        Always returns None to defer sampling to sample_tokens().
        """
        execute_start = time.perf_counter()
        batch_size = len(scheduler_output.scheduled_new_reqs) + len(
            scheduler_output.scheduled_cached_reqs.req_ids
        )

        logger.debug("scheduler_output: %s", scheduler_output)

        # Free slots of finished requests
        # We intentionally do this before updating the cached states as
        # the _update_states method is common across all hardware platforms.
        if self.use_custom_seq_id_mapping:
            for req_id in scheduler_output.finished_req_ids:
                if req_id in self.vllm_req_to_neuron_seq_id_mapping:
                    freed_slot = self.vllm_req_to_neuron_seq_id_mapping.pop(req_id)
                    self.free_seq_ids.add(freed_slot)

        # Update cached state
        update_start = time.perf_counter()
        self._update_states(scheduler_output)
        update_elapsed = (time.perf_counter() - update_start) * 1000
        logger.debug("[PERF] _update_states: %.2fms", update_elapsed)

        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # _prepare_model_input converts the scheduler output to ModelInputForNeuron
        prepare_start = time.perf_counter()
        model_input = self._prepare_model_input(scheduler_output)
        prepare_elapsed = (time.perf_counter() - prepare_start) * 1000
        logger.debug("[PERF] _prepare_model_input: %.2fms", prepare_elapsed)

        logger.debug("model_input: %s", model_input)

        # NOTE: setup current batch's metadata for kv connector.
        # Currently, only verified with NixlConnector.
        # P6: Skip the KV connector context manager when there are no pending
        # transfers, to avoid blocking on start_load_kv() every iteration.
        _kv_md = getattr(scheduler_output, 'kv_connector_metadata', None)
        _has_kv_work = bool(
            _kv_md and (
                getattr(_kv_md, 'reqs_to_recv', None)
                or getattr(_kv_md, 'reqs_to_send', None)
            )
        )
        if _has_kv_work:
            _ctx_mgrs = (
                set_forward_context(None, self.vllm_config),
                self.maybe_get_kv_connector_output(scheduler_output),
            )
        else:
            _ctx_mgrs = (
                set_forward_context(None, self.vllm_config),
            )

        import contextlib
        kv_connector_output = None
        with contextlib.ExitStack() as _stack:
            for _ctx in _ctx_mgrs:
                _result = _stack.enter_context(_ctx)
            if _has_kv_work:
                kv_connector_output = _result

            # Execute model forward pass (no sampling - deferred to sample_tokens())
            model_exec_start = time.perf_counter()
            if self.model.architecture in NEURON_MULTI_MODAL_MODELS:
                hidden_states = self._execute_model_for_multimodal_models(
                    model_input,
                    intermediate_tensors,
                )
            else:
                hidden_states = self._execute_model_for_text(
                    model_input,
                    intermediate_tensors,
                )
            model_exec_elapsed = (time.perf_counter() - model_exec_start) * 1000
            logger.debug("[PERF] model_execution: %.2fms", model_exec_elapsed)

        # Store KV connector output for sample_tokens() to use
        self._kv_connector_output = kv_connector_output

        # Cache for sample_tokens()
        self._cached_logits = hidden_states
        self._cached_model_input = model_input

        total_elapsed = (time.perf_counter() - execute_start) * 1000
        logger.debug(
            "[PERF] execute_model total: %.2fms [batch=%d]", total_elapsed, batch_size
        )

        # Return None to defer sampling to sample_tokens()
        return None

    def _prepare_logits_for_sampling(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelInputForNeuron,
    ) -> torch.Tensor:
        """
        Prepare logits for sampling.
        NOTE: For NxDI models without on-device sampling, the model forward
        pass should return logits directly (not hidden states).
        Verify your model configuration returns logits when
        on_device_sampling_config is None.
        """
        # Clone to avoid modifying model output tensor
        hidden_states = hidden_states.clone()

        # Handle prefill completion state (mask incomplete prefills)
        if model_input.prefill_completion_state is not None:
            for i, state in enumerate(model_input.prefill_completion_state):
                if not state.item():
                    # This request's prefill isn't complete, mask its logits
                    hidden_states[i] = float("-inf")

        # Reorder to match input batch order
        request_id_order = {
            request_id: idx for idx, request_id in enumerate(model_input.request_ids)
        }

        reorder_indices = torch.tensor(
            [request_id_order[request_id] for request_id in self.input_batch.req_ids],
            dtype=torch.long,
            device=hidden_states.device,
        )

        return hidden_states[reorder_indices]

    def _generate_model_runner_output(
        self,
        sampler_outputs: SamplerOutput | None,
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> ModelRunnerOutput:
        if sampler_outputs is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = sampler_outputs.sampled_token_ids
        spec_token_ids = None

        if self.speculative_config is None:
            # No spec decode tokens.
            # Convert to list of Python int lists for vLLM scheduler compatibility
            # (vLLM's scheduler expects native Python ints, not numpy.int64)
            valid_sampled_token_ids = [
                [int(x) for x in row if x != -1] for row in sampled_token_ids.tolist()
            ]

        else:
            # Modify NxDI output to conform to vLLM ModelRunnerOutput
            # sampled_token_ids: list[list[int]] (native Python ints)
            # spec_token_ids: Optional[list[list[int]]]
            # If NxDI returns [B, T, 1], squeeze the trailing dim.
            squeezed_tensor = (
                sampled_token_ids.squeeze(-1)
                if sampled_token_ids.dim() == 3 and sampled_token_ids.size(-1) == 1
                else sampled_token_ids
            )

            # Work directly on tensor; only drop -1 pads (0 is a valid token).
            valid_sampled_token_ids = []
            spec_token_ids = []
            for row in squeezed_tensor.cpu():
                kept = [int(x) for x in row if x != -1]  # keep 0s; drop only -1 pads
                valid_sampled_token_ids.append(kept)
                spec_token_ids.append(kept[:-1] if kept else [])

            self.spec_token_ids = spec_token_ids

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if len(sampled_ids) == 0:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # sampled_ids is already a list of native Python ints
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logger.debug("final valid_sampled_token_ids: %s", valid_sampled_token_ids)

        logprobs = None
        if sampler_outputs.logprobs_tensors is not None:
            logprobs = sampler_outputs.logprobs_tensors.tolists()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            # CPU sampling supports logprobs.
            logprobs=logprobs,
            # TODO: support the following fields.
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec for Neuron models.
        Since Neuron uses compiled models without individual Attention layers,
        we manually create specs for each layer based on model config.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # Get number of layers from model config
        num_layers = get_num_layers_from_hf_config(self.model_config.hf_config)

        kv_cache_spec: dict[str, KVCacheSpec] = {}

        # Create a spec for each layer
        for layer_idx in range(num_layers):
            layer_name = f"layers.{layer_idx}.self_attn"  # standard naming convention
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.parallel_config.tensor_parallel_size,
                head_size=self.model.head_dim,
                dtype=self.model_config.dtype,
                sliding_window=self.model_config.get_sliding_window(),
            )

        return kv_cache_spec

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            if self.lora_config is not None:
                self.lora_manager.remove_req_id(req_id)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features or [],
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            req_state.num_computed_tokens = self._get_last_token_position(req_state)

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ()
            )
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index
                ] = spec_token_ids

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        # self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _execute_model_for_text(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor:
        """Execute model forward pass for text models.

        Returns:
            hidden_states: Model output tensor
        """
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            position_ids=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            slot_mapping=model_input.slot_mapping,
            block_tables=model_input.block_tables,
            full_context_lens=model_input.full_context_lens,
            computed_context_lens=model_input.computed_context_lens,
            sampling_params=model_input.sampling_params,
            adapter_ids=model_input.adapter_ids,
            prefill_completion_state=model_input.prefill_completion_state,
            **_mm_kwargs_to_device(model_input.multi_modal_kwargs, self.device),
        )
        return hidden_states

    def _execute_model_for_multimodal_models(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor:
        """Execute model forward pass for multimodal models.

        Returns:
            hidden_states: Model output tensor
        """
        hidden_states = self.model.execute_model(model_input)
        return hidden_states

    def _process_multi_modal_data_neuron(
        self, mm_data: list[MultiModalFeatureSpec]
    ) -> BatchedTensorInputs:
        if len(mm_data) == 1:
            mm_item = mm_data[0].data  # Type: Optional[MultiModalKwargsItem]
            # Get tensor data from MultiModalKwargsItem
            if mm_item is None:
                mm_kwargs: dict[str, NestedTensors] = {}
            else:
                mm_kwargs = mm_item.get_data()
        elif len(mm_data) > 1:
            mm_kwargs = {}
            for mm_item in mm_data:  # Type: List[MultiModalFeatureSpec]
                for k, v in mm_item.data.items():
                    if k not in mm_kwargs:
                        mm_kwargs[k] = []
                    mm_kwargs[k].append(v.data)

        logger.debug("mm_data in _process_multi_modal_data_neuron: %s", mm_kwargs)

        if self.model.model.config.model_type == "llava":
            mm_kwargs = self._process_multi_modal_data_neuron_llava(mm_kwargs)
        elif self.model.model.config.model_type == "qwen2_vl":
            mm_kwargs = self._process_multi_modal_data_neuron_qwen2_vl(mm_kwargs)
        elif self.model.model.config.model_type == "qwen3_vl":
            # Qwen3-VL uses the same processing as Qwen2-VL
            mm_kwargs = self._process_multi_modal_data_neuron_qwen2_vl(mm_kwargs)
        elif self.model.model.config.model_type == "llama4":
            pass  # llama4 doesn't require special processing
        else:
            raise NotImplementedError(
                f"processing mm data for model type {self.model.model.config.model_type} not supported on Neuron yet!"
            )

        return mm_kwargs

    def _process_multi_modal_data_neuron_llava(self, mm_data):
        # We reconstruct image_sizes here to match HF's implementation
        # since vLLM implementation slices pixel_values for each image separately
        # (see vllm/model_executor/models/llava.py)
        if isinstance(mm_data["pixel_values"], torch.Tensor):
            logger.debug("pixel_values tensor shape: %s", mm_data["pixel_values"].shape)
            pixel_values = mm_data["pixel_values"]
            # Ensure pixel_values is 4D [B, C, H, W]
            # vLLM may pass 3D [C, H, W] for single images
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
                mm_data["pixel_values"] = pixel_values
                logger.debug("Added batch dimension, new shape: %s", pixel_values.shape)
            # Now pixel_values is 4D [B, C, H, W]
            img_height = pixel_values.shape[2]
            img_width = pixel_values.shape[3]
            # image_sizes must be 2D [[height, width]] for NxDI patchify
            mm_data["image_sizes"] = torch.tensor(
                [[img_height, img_width]], dtype=torch.int32
            )
        elif isinstance(mm_data["pixel_values"], list):
            image_sizes_list = []
            # The below logic pads multiple images within one request to
            # max height and width across all images
            # This mimics the same logic as HF processor
            max_height = 0
            max_width = 0
            for pixel_values in mm_data["pixel_values"]:
                logger.debug("pixel_values.shape: %s", pixel_values.shape)
                # vLLM may pass 3D [C, H, W] for each image
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                    logger.debug(
                        "Added batch dimension, new shape: %s", pixel_values.shape
                    )
                img_height = pixel_values.shape[2]
                img_width = pixel_values.shape[3]
                max_height = max(max_height, img_height)
                max_width = max(max_width, img_width)
                image_sizes_list.append(
                    torch.tensor([img_height, img_width], dtype=torch.int32)
                )
            mm_data["image_sizes"] = torch.stack(image_sizes_list, dim=0)
            padded_pixel_values = []
            for pixel_values in mm_data["pixel_values"]:
                # vLLM may pass 3D [C, H, W] for each image
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                    logger.debug(
                        "Added batch dimension, new shape: %s", pixel_values.shape
                    )
                img_height = pixel_values.shape[2]
                img_width = pixel_values.shape[3]
                padded_pixel_value = torch.nn.functional.pad(
                    pixel_values,
                    (0, max_width - img_width, 0, max_height - img_height),
                    value=0,
                )
                logger.debug("padded_pixel_value shape: %s", padded_pixel_value.shape)
                padded_pixel_values.append(padded_pixel_value.unsqueeze(0))
            mm_data["pixel_values"] = torch.cat(padded_pixel_values, dim=0)
        logger.debug("mm_data in _process_multi_modal_data_neuron_llava: %s", mm_data)
        return mm_data

    def _process_multi_modal_data_neuron_qwen2_vl(self, mm_data):
        """
        Normalize Qwen2-VL multimodal kwargs into tensors expected by batching.
        """
        pixel_values = mm_data.get("pixel_values")
        if isinstance(pixel_values, list):
            mm_data["pixel_values"] = torch.cat(pixel_values, dim=0)

        image_grid_thw = mm_data.get("image_grid_thw")

        if isinstance(image_grid_thw, list):
            mm_data["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)
        elif isinstance(image_grid_thw, torch.Tensor):
            # vLLM may pass 1-D image_grid_thw for single image
            mm_data["image_grid_thw"] = image_grid_thw.unsqueeze(0)

        logger.debug(
            "mm_data in _process_multi_modal_data_neuron_qwen2_vl: %s", mm_data
        )
        return mm_data

    def _prepare_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForNeuron:
        if self.is_chunked_prefill:
            chunked_prefill_model_input = self._prepare_chunked_prefill_inputs(
                scheduler_output
            )

            multi_modal_kwargs = None
            lora_adapter_ids = None

            return self._finalize_chunked_prefill_inputs(
                chunked_prefill_model_input,
                multi_modal_kwargs,
                lora_adapter_ids,
            )
        else:
            continuous_batching_model_input, is_prefill = (
                self._prepare_continuous_batching_inputs(scheduler_output)
            )
            return self._finalize_continuous_batching_inputs(
                continuous_batching_model_input,
                is_prefill,
            )

    def _prepare_chunked_prefill_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> IntermediateInputData:
        """
        This function is used to prepare the inputs for chunked prefill.
        It needs to treat prefill and decoding requests differently.
          *  For NewRequestData, it is guaranteed to be a prefill request.
          *  For CachedRequestData, it can be a prefill request or a decoding request.
          The way to tell if it is a prefill request is to check if the number of
          computed tokens is less than the number of context tokens.
        """
        data = IntermediateInputData()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        logger.debug("num_scheduled_tokens: %s", num_scheduled_tokens)

        for request_data in scheduler_output.scheduled_new_reqs:
            self._process_new_request_for_chunked_prefill(
                request_data, num_scheduled_tokens[request_data.req_id], data
            )

        cached_request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_request_data.req_ids):
            self._process_cached_request_for_chunked_prefill(
                cached_request_data, i, num_scheduled_tokens[req_id], data
            )

        return data

    def _prepare_continuous_batching_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Tuple[IntermediateInputData, bool]:
        """
        This function is used to prepare the inputs for continuous batching.
            For NewRequestData,
                1. On decode nodes in the DI setting, it should always do decoding.
                2. For all other cases (i.e., prefill nodes in DI, non-DI case), do
                   prefills.
            For CachedRequestData, it is guaranteed to be a decoding request.
        """
        data = IntermediateInputData()
        is_prefill = False
        for request_data in scheduler_output.scheduled_new_reqs:
            if (
                scheduler_output.kv_connector_metadata is not None
                and not scheduler_output.kv_connector_metadata.reqs_to_save
            ):
                # This branch is activated only if the current node is a decode node in the DI setting.
                # 1. we cannot use `request_data.num_computed_tokens > 0` to indicate the node is decode because
                # prefill with APC can also lead to request_data.num_computed_tokens > 0
                # 2. we cannot use `scheduler_output.kv_connector_metadata.reqs_to_recv is not None` because on decode node,
                # the scheduler always schedules transfer before scheduling decode for a request. So the scheduler output
                # for the new decode request has empty reqs_to_recv.

                logger.debug("Prepare model input for decoding for the new request")
                self._process_new_request_for_disaggregated_inference(
                    request_data, data
                )
                is_prefill = False
            else:
                self._process_new_request_for_continuous_batching(request_data, data)
                is_prefill = True

        cached_request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_request_data.req_ids):
            self._process_cached_request_for_continuous_batching(
                cached_request_data, i, data
            )

        return data, is_prefill

    def _process_new_request_for_continuous_batching(
        self, request_data: NewRequestData, data: IntermediateInputData
    ) -> None:
        # Assign a free sequence id to the new request.
        assert request_data.req_id not in self.vllm_req_to_neuron_seq_id_mapping, (
            "Encountered an existing request ID while prefilling a new request"
        )
        assert self.free_seq_ids, "No free sequence ID available!"
        assigned_slot = self.free_seq_ids.pop()
        self.vllm_req_to_neuron_seq_id_mapping[request_data.req_id] = assigned_slot

        data.request_ids.append(request_data.req_id)

        data.input_tokens.append(request_data.prompt_token_ids)
        if len(request_data.prompt_token_ids) > self.max_prompt_length:
            raise ValueError(
                f"Prompt length ({len(request_data.prompt_token_ids)} tokens) exceeds the maximum "
                f"prompt length ({self.max_prompt_length} tokens) for this Neuron model. "
                f'To handle this gracefully during online serving, add "max_prompt_length": '
                f"{self.max_prompt_length} to --additional-config. This will return a 400 error "
                f"for oversized prompts instead of terminating the engine (supported on OpenAI "
                f"/v1/completions and /v1/chat/completions endpoints). Alternatively, provide "
                f"a shorter prompt or recompile the Neuron model with a larger max_prompt_length "
                f'by setting "max_context_length": <desired_length> in override_neuron_config when compiling.'
            )
        data.position_ids.append(list(range(len(request_data.prompt_token_ids))))
        data.input_block_ids.append(assigned_slot)

        data.full_context_lens.append(len(request_data.prompt_token_ids))
        data.prefill_completion_state.append(None)
        data.adapter_ids.append(self._prepare_adapter_id_in_new_request(request_data))

        if self.is_prefix_caching:
            self._process_new_request_for_continuous_batching_with_prefix_caching(
                request_data, data
            )

        if request_data.mm_features:
            data.multi_modal_kwargs = self._process_multi_modal_data_neuron(
                request_data.mm_features
            )

    def _process_new_request_for_continuous_batching_with_prefix_caching(
        self, request_data: NewRequestData, data: IntermediateInputData
    ) -> None:
        assert len(request_data.block_ids) == 1
        block_table = copy.deepcopy(request_data.block_ids)[0]

        # pad the block_table to have the length of num_gpu_blocks
        block_size = self.cache_config.block_size
        max_len = self.model_config.max_model_len
        max_blocks_per_seq = max_len // block_size
        padded_block_table = [self._BLOCK_TABLE_PAD] * max_blocks_per_seq
        padded_block_table[: len(block_table)] = block_table[:]
        data.block_tables.append(padded_block_table)

        data.computed_context_lens.append(request_data.num_computed_tokens)

        prompt_len = len(request_data.prompt_token_ids)
        slot_mapping_for_cur_seq = [
            (block_table[i // block_size] * block_size + i % block_size)
            if i < prompt_len
            else self._SLOT_MAPPING_PAD
            for i in range(max_len)
        ]
        data.slot_mapping.append(
            slot_mapping_for_cur_seq[request_data.num_computed_tokens :]
        )

    def _process_new_request_for_disaggregated_inference(
        self, request_data: NewRequestData, data: IntermediateInputData
    ) -> None:
        """
        This method prepares model input for decoding based on NewRequestData.
        This method should only be invoked by decode nodes in the DI setting.
        """
        # Assign a free sequence id to the new request.
        assert request_data.req_id not in self.vllm_req_to_neuron_seq_id_mapping, (
            "Encountered an existing request ID while prefilling a new request"
        )
        assert self.free_seq_ids, "No free sequence ID available!"
        assigned_slot = self.free_seq_ids.pop()
        self.vllm_req_to_neuron_seq_id_mapping[request_data.req_id] = assigned_slot

        req_id = request_data.req_id
        data.request_ids.append(req_id)

        data.input_tokens.append([request_data.prompt_token_ids[-1]])
        data.position_ids.append([request_data.num_computed_tokens])
        data.input_block_ids.append(assigned_slot)

        data.full_context_lens.append(request_data.num_computed_tokens + 1)
        data.prefill_completion_state.append(None)

        assert len(request_data.block_ids) == 1
        block_table = copy.deepcopy(request_data.block_ids)[0]

        # pad the block_table to have the length of num_gpu_blocks
        block_size = self.cache_config.block_size
        max_len = self.model_config.max_model_len
        max_blocks_per_seq = max_len // block_size
        padded_block_table = [self._BLOCK_TABLE_PAD] * max_blocks_per_seq
        padded_block_table[: len(block_table)] = block_table[:]
        data.block_tables.append(padded_block_table)

        data.computed_context_lens.append(request_data.num_computed_tokens)

        position = request_data.num_computed_tokens
        block_number = block_table[position // self.cache_config.block_size]
        block_offset = position % self.cache_config.block_size
        slot = block_number * self.cache_config.block_size + block_offset

        data.slot_mapping.append([slot])

        # TODO: support speculative decoding in disaggregated inference.
        # TODO: support multi-lora in disaggregated inference.
        # TODO: support multi-modal in disaggregated inference.

    def _process_cached_request_for_continuous_batching(
        self, request_data: CachedRequestData, index: int, data: IntermediateInputData
    ) -> None:
        req_id = request_data.req_ids[index]
        assert req_id in self.vllm_req_to_neuron_seq_id_mapping, (
            "The request ID for the current decode request "
            " is not found in request to sequence ID "
            "mapping"
        )
        data.request_ids.append(req_id)
        state = self.requests[req_id]

        data.input_tokens.append([state.output_token_ids[-1]])

        position = self._get_last_token_position(state)

        data.position_ids.append([position])
        data.input_block_ids.append(self.vllm_req_to_neuron_seq_id_mapping[req_id])

        data.full_context_lens.append(position + 1)
        data.computed_context_lens.append(position)
        data.prefill_completion_state.append(None)
        data.adapter_ids.append(self._prepare_adapter_id_in_cached_request(req_id))

        if self.is_prefix_caching:
            self._process_cached_request_for_continuous_batching_with_prefix_caching(
                request_data, index, data
            )

    def _process_cached_request_for_continuous_batching_with_prefix_caching(
        self, request_data: CachedRequestData, index: int, data: IntermediateInputData
    ) -> None:
        req_id = request_data.req_ids[index]
        state = self.requests[req_id]
        block_table = copy.deepcopy(state.block_ids)[0]

        attn_tkg_nki_kernel_enabled = (
            self.model.neuron_config.attn_tkg_nki_kernel_enabled
            or self.model.neuron_config.attn_block_tkg_nki_kernel_enabled
        )
        # Pad -1 to allow DMA skipping that is supported
        # by attention TKG kernel.
        block_table_padding = (
            -1 if attn_tkg_nki_kernel_enabled else self._BLOCK_TABLE_PAD
        )
        block_size = self.cache_config.block_size
        max_len = self.model_config.max_model_len
        max_blocks_per_seq = max_len // block_size
        padded_block_table = [block_table_padding] * max_blocks_per_seq
        padded_block_table[: len(block_table)] = block_table[:]
        data.block_tables.append(padded_block_table)

        position = self._get_last_token_position(state)

        block_number = block_table[position // self.cache_config.block_size]
        block_offset = position % self.cache_config.block_size
        slot = block_number * self.cache_config.block_size + block_offset

        # When speculative decoding is enabled, append consecutive slots
        # for the speculative tokens (draft + final alignment on device).
        slots = [slot]
        if self.speculative_config is not None:
            for i in range(1, self.speculative_config.num_speculative_tokens):
                slots.append(slots[0] + i)

        data.slot_mapping.append(slots)

    def _prepare_adapter_id_in_new_request(self, request_data: NewRequestData):
        if self.lora_config is None:
            return None
        req_id = request_data.req_id
        lora_name = request_data.lora_request.lora_name
        adapter_id = self.lora_manager.convert_adapter_id_to_index(lora_name)
        self.lora_manager.add_req_id_to_adapter_id_mapping(req_id, adapter_id)
        return adapter_id

    def _prepare_adapter_id_in_cached_request(self, req_id):
        if self.lora_config is None:
            return None
        return self.lora_manager.get_adapter_id_with_req_id(req_id)

    def _finalize_continuous_batching_inputs(
        self,
        data: IntermediateInputData,
        is_prefill: bool,
    ) -> ModelInputForNeuron:
        if is_prefill:
            max_seq_len = max(data.full_context_lens)
            assert max_seq_len > 0
            input_tokens = make_tensor_with_pad(
                data.input_tokens,
                pad=0,
                max_len=max_seq_len,
                dtype=torch.long,
                device=self.device,
            )
            position_ids = make_tensor_with_pad(
                data.position_ids,
                pad=0,
                max_len=max_seq_len,
                dtype=torch.long,
                device=self.device,
            )
            input_block_ids = torch.tensor(
                data.input_block_ids, dtype=torch.long, device=self.device
            )
            slot_mapping = make_tensor_with_pad(
                data.slot_mapping,
                pad=self._SLOT_MAPPING_PAD,
                max_len=self.model_config.max_model_len,
                dtype=torch.long,
                device=self.device,
            )
            block_tables = torch.tensor(
                data.block_tables, dtype=torch.long, device=self.device
            )
            full_context_lens = torch.tensor(
                data.full_context_lens, dtype=torch.long, device=self.device
            ).reshape(-1, 1)
            computed_context_lens = torch.tensor(
                data.computed_context_lens, dtype=torch.long, device=self.device
            ).reshape(-1, 1)

        else:
            input_tokens = make_tensor_with_pad(
                data.input_tokens,
                pad=0,
                max_len=1,
                dtype=torch.long,
                device=self.device,
            )
            position_ids = make_tensor_with_pad(
                data.position_ids,
                pad=0,
                max_len=1,
                dtype=torch.long,
                device=self.device,
            )
            input_block_ids = torch.tensor(
                data.input_block_ids, dtype=torch.long, device=self.device
            )
            slot_mapping = torch.tensor(
                data.slot_mapping, dtype=torch.long, device=self.device
            )
            block_tables = torch.tensor(
                data.block_tables, dtype=torch.long, device=self.device
            )

            full_context_lens = torch.tensor(
                data.full_context_lens, dtype=torch.long, device=self.device
            ).reshape(-1, 1)

            # Convert computed_context_lens to tensor
            computed_context_lens = torch.tensor(
                data.computed_context_lens, dtype=torch.long, device=self.device
            ).reshape(-1, 1)
        lora_adapter_ids = None
        if self.lora_config is not None:
            lora_adapter_ids = torch.tensor(
                data.adapter_ids, dtype=torch.long, device=self.device
            )
        return ModelInputForNeuron(
            request_ids=data.request_ids,
            input_tokens=input_tokens,
            position_ids=position_ids,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            prefill_completion_state=None,
            sampling_params=self.get_nxd_sampling_params(input_tokens),
            multi_modal_kwargs=data.multi_modal_kwargs,
            adapter_ids=lora_adapter_ids,
        )

    def _process_new_request_for_chunked_prefill(
        self,
        request_data: NewRequestData,
        num_scheduled_tokens: int,
        data: IntermediateInputData,
    ) -> None:
        data.request_ids.append(request_data.req_id)
        assert len(request_data.block_ids) == 1
        block_table = copy.deepcopy(request_data.block_ids)[0]

        start = request_data.num_computed_tokens
        end = start + num_scheduled_tokens

        data.input_tokens.extend(request_data.prompt_token_ids[start:end])
        data.position_ids.extend(range(start, end))
        data.input_block_ids.append(0)

        for i in range(start, end):
            block_number = block_table[i // self.cache_config.block_size]
            offset = i % self.cache_config.block_size
            data.slot_mapping.append(
                block_number * self.cache_config.block_size + offset
            )

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(end >= len(request_data.prompt_token_ids))

    def _process_cached_request_for_chunked_prefill(
        self,
        request_data: CachedRequestData,
        index: int,
        num_scheduled_tokens: int,
        data: IntermediateInputData,
    ) -> None:
        req_id = request_data.req_ids[index]
        data.request_ids.append(req_id)
        state = self.requests[req_id]
        logger.debug("for req_id %s, state: %s", req_id, state)
        block_table = copy.deepcopy(state.block_ids)[0]

        start = request_data.num_computed_tokens[index]
        end = start + num_scheduled_tokens

        if num_scheduled_tokens > 1:
            logger.debug("start: %s, end: %s", start, end)
            resumed_prompt_tokens = state.prompt_token_ids[start:end]
            data.input_tokens.extend(resumed_prompt_tokens)
            logger.debug("resumed prompt tokens: %s", resumed_prompt_tokens)

        if len(state.output_token_ids) > 0:
            data.input_tokens.append(state.output_token_ids[-1])
            logger.debug("appended output token %s", state.output_token_ids[-1])
        data.position_ids.extend(range(start, end))
        data.input_block_ids.append(0)

        for i in range(start, end):
            block_number = block_table[i // self.cache_config.block_size]
            offset = i % self.cache_config.block_size
            data.slot_mapping.append(
                block_number * self.cache_config.block_size + offset
            )

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(end >= len(state.prompt_token_ids))

    def _finalize_chunked_prefill_inputs(
        self,
        data: IntermediateInputData,
        multi_modal_kwargs: BatchedTensorInputs,
        lora_adapter_ids: str | None,
    ) -> ModelInputForNeuron:
        device = self.device

        input_tokens = torch.tensor(
            data.input_tokens, dtype=torch.long, device=device
        ).reshape(1, -1)
        position_ids = torch.tensor(
            data.position_ids, dtype=torch.long, device=device
        ).reshape(1, -1)
        input_block_ids = torch.tensor(
            data.input_block_ids[:1], dtype=torch.long, device=device
        )
        slot_mapping = torch.tensor(data.slot_mapping, dtype=torch.long, device=device)

        max_blocks = max(len(b) for b in data.block_tables)
        for b in data.block_tables:
            b.extend([self._BLOCK_TABLE_PAD] * (max_blocks - len(b)))

        block_tables = torch.tensor(data.block_tables, dtype=torch.long, device=device)
        full_context_lens = torch.tensor(
            data.full_context_lens, dtype=torch.long, device=device
        )
        computed_context_lens = torch.tensor(
            data.computed_context_lens, dtype=torch.long, device=device
        )
        prefill_completion_state = torch.tensor(
            data.prefill_completion_state, dtype=torch.bool, device=device
        )

        return ModelInputForNeuron(
            request_ids=data.request_ids,
            input_tokens=input_tokens,
            position_ids=position_ids,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
            prefill_completion_state=prefill_completion_state,
            sampling_params=self.get_nxd_sampling_params(input_tokens),
            multi_modal_kwargs=multi_modal_kwargs,
            adapter_ids=lora_adapter_ids,
        )

    def get_nxd_sampling_params(self, input_ids: torch.Tensor):
        if self.model.neuron_config.on_device_sampling_config:
            max_topk = self.model.neuron_config.on_device_sampling_config.global_topk
        else:
            max_topk = self.model_config.get_vocab_size()

        max_topk = min(max_topk, self._MAX_NEURON_SAMPLING_TOP_K)

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, request in enumerate(self.requests.values()):
            top_k[index] = (
                request.sampling_params.top_k
                if request.sampling_params.top_k > 0
                and request.sampling_params.top_k < max_topk
                else max_topk
            )
            top_p[index] = request.sampling_params.top_p
            temperature[index] = request.sampling_params.temperature
            if request.sampling_params.temperature == 0.0:
                top_k[index] = 1
                temperature[index] = 1.0

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        if not self.is_chunked_prefill:
            if input_ids.shape[0] != sampling_params.shape[0]:
                sampling_params = sampling_params[: input_ids.shape[0]]

        return sampling_params

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _dummy_run(self, num_tokens: int, uniform_decode: bool = False) -> None:
        """Execute a dummy forward pass for engine initialization and warmup."""
        if self.model is None:
            logger.warning("Model is not loaded, skipping dummy run")
            return

        try:
            # Create minimal dummy input
            dummy_input = ModelInputForNeuron(
                request_ids=["dummy_request"],
                input_tokens=torch.ones(
                    (1, num_tokens), dtype=torch.long, device=self.device
                ),
                position_ids=torch.arange(
                    num_tokens, dtype=torch.long, device=self.device
                ).unsqueeze(0),
                input_block_ids=torch.zeros(1, dtype=torch.long, device=self.device),
                slot_mapping=torch.arange(
                    num_tokens, dtype=torch.long, device=self.device
                ),
                block_tables=torch.arange(
                    (num_tokens + self.block_size - 1) // self.block_size,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                full_context_lens=torch.tensor(
                    [num_tokens], dtype=torch.long, device=self.device
                ).reshape(-1, 1),
                computed_context_lens=torch.tensor(
                    [num_tokens - 1], dtype=torch.long, device=self.device
                ).reshape(-1, 1),
                sampling_params=self.get_nxd_sampling_params(
                    torch.ones((1, num_tokens), dtype=torch.long, device=self.device)
                ),
                multi_modal_kwargs=None,
                adapter_ids=None,
                prefill_completion_state=None,
            )

            # Execute dummy forward pass
            if self.model.architecture in NEURON_MULTI_MODAL_MODELS:
                self._execute_model_for_multimodal_models(dummy_input)
            else:
                self._execute_model_for_text(dummy_input)

        except Exception as e:
            logger.warning(
                "Dummy run failed: %s. This may be expected during initialization.", e
            )
