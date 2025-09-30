# SPDX-License-Identifier: Apache-2.0
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from neuronx_distributed_inference.modules.generation.sampling import \
    prepare_sampling_params
from neuronx_distributed_inference.modules.lora_serving import (
    LoraModelManager, LoraServingConfig)
from neuronx_distributed_inference.modules.padding import pad_tensor
from vllm.config import VllmConfig
from vllm.multimodal import BatchedTensorInputs, MultiModalKwargs
from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem
from vllm.sequence import IntermediateTensors
from vllm.utils import make_tensor_with_pad
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             ModelRunnerOutput, SamplerOutput)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from vllm_neuron.worker.constants import NEURON_MULTI_MODAL_MODELS
from vllm_neuron.worker.neuronx_distributed_model_loader import \
    get_neuron_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ModelInputForNeuron:
    """
    Model input for NeuronX Distributed Inference model runner.
    """
    request_ids: Optional[list[str]] = None
    input_tokens: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    full_context_lens: Optional[torch.Tensor] = None
    computed_context_lens: Optional[torch.Tensor] = None
    sampling_params: Optional[torch.Tensor] = None
    multi_modal_kwargs: BatchedTensorInputs = None
    adapter_ids: Optional[str] = None
    # Boolean tensor to indicate if the request is ready
    # for sampling. Needed by chunked prefill.
    prefill_completion_state: Optional[torch.Tensor] = None


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


class NeuronxDistributedModelRunner(LoRAModelRunnerMixin):
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
        )

        self.requests: dict[str, CachedRequestState] = {}
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # vLLM uses lora_manager to manage LoRA modules
        self.lora_manager = None
        self.model = None
        self.lora_serving_config = None

        self.is_block_kv_layout = False
        self.is_prefix_caching = False
        self.is_chunked_prefill = False

        # The following fields are used to support custom sequence id mapping.
        # The goal is to retain the batch line information for contiguous kv cache.
        # A mapping of vLLM request Id to neuron sequence id.
        self.use_custom_seq_id_mapping = not self.is_chunked_prefill
        self.vllm_req_to_neuron_seq_id_mapping: Dict[str, int] = {}
        # Set of neuron sequence id that are free for use.
        self.free_seq_ids = set(range(self.max_num_reqs))
        self._draft_token_ids = None

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig):
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # Not required for NeuronX Distributed Inference. To satisfy the interface.
        return

    def _get_nxdi_lora_config(self):
        override_neuron_config = self.vllm_config.additional_config.get(
            "override_neuron_config", None)
        lora_ckpt_paths = override_neuron_config.pop("lora_modules", None)
        target_modules = override_neuron_config.pop("target_modules", None)
        lora_ckpt_json = override_neuron_config.pop("lora_ckpt_json", None)

        return LoraServingConfig(
            max_loras=self.lora_config.max_loras,
            max_lora_rank=self.lora_config.max_lora_rank,
            target_modules=target_modules,
            lora_ckpt_paths=lora_ckpt_paths,
            lora_ckpt_json=lora_ckpt_json,
            batch_size=self.scheduler_config.max_num_seqs,
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
        if self.lora_config is not None:
            self.lora_serving_config = self._get_nxdi_lora_config()
            self.lora_manager = LoraModelManager(self.lora_serving_config)
        self.model = get_neuron_model(
            self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            lora_serving_config=self.lora_serving_config,
            speculative_config=self.speculative_config,
            additional_config=self.vllm_config.additional_config)
        self.is_block_kv_layout = self.model.neuron_config.is_block_kv_layout
        self.is_prefix_caching = self.model.neuron_config.is_prefix_caching
        self.is_chunked_prefill = \
            self.model.neuron_config.chunked_prefill_config is not None
        self.model.is_reorder_needed = not (self.is_prefix_caching
                                            or self.is_chunked_prefill)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        logger.debug(f"scheduler_output: {scheduler_output}")

        # Free slots of finished requests
        # We intentionally do this before updating the cached states as
        # the _update_states method is common across all hardware platforms.
        if self.use_custom_seq_id_mapping:
            for req_id in scheduler_output.finished_req_ids:
                if req_id in self.vllm_req_to_neuron_seq_id_mapping:
                    freed_slot = self.vllm_req_to_neuron_seq_id_mapping.pop(
                        req_id)
                    self.free_seq_ids.add(freed_slot)

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # _prepare_model_input converts the scheduler output to ModelInputForNeuron
        model_input = self._prepare_model_input(scheduler_output)
        logger.debug(f"model_input: {model_input}")

        if self.model.architecture in NEURON_MULTI_MODAL_MODELS:
            sampler_outputs = self._execute_model_for_multimodal_models(
                model_input,
                intermediate_tensors,
            )
        else:
            sampler_outputs = self._execute_model_for_text(
                model_input,
                intermediate_tensors,
            )

        return self._generate_model_runner_output(sampler_outputs)

    def _generate_model_runner_output(
            self,
            sampler_outputs: Optional[SamplerOutput]) -> ModelRunnerOutput:
        if sampler_outputs is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = sampler_outputs.sampled_token_ids
        spec_token_ids = None

        if self.speculative_config is None:
            # No spec decode tokens.
            valid_sampled_token_ids = [[x for x in row if x != -1]
                                       for row in sampled_token_ids.tolist()]

        else:
            # Modify NxDI output to conform to vLLM ModelRunnerOutput
            # sampled_token_ids: list[list[int]]
            # spec_token_ids: Optional[list[list[int]]]
            # If NxDI returns [B, T, 1], squeeze the trailing dim.
            squeezed_tensor = (
                sampled_token_ids.squeeze(-1) if sampled_token_ids.dim() == 3
                and sampled_token_ids.size(-1) == 1 else sampled_token_ids)

            # Work directly on tensor; only drop -1 pads (0 is a valid token).
            valid_sampled_token_ids = []
            spec_token_ids = []
            for row in squeezed_tensor.cpu():
                kept = row[row != -1].tolist()  # keep 0s; drop only -1 pads
                valid_sampled_token_ids.append(kept)
                spec_token_ids.append(kept[:-1] if kept else [])

            self.spec_token_ids = spec_token_ids

        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logger.debug(
            f"final valid_sampled_token_ids: {valid_sampled_token_ids}")
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            # TODO: support the following fields. currently they are hardcoded to None
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[])

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        return {
            "layer":
            FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.model.num_key_value_heads,
                head_size=self.model.head_dim,
                # TODO: take the following from the model config
                dtype=torch.bfloat16,
                use_mla=False,
                sliding_window=None,
            )
        }

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
                mm_kwargs=new_req_data.mm_kwargs,
                mm_positions=new_req_data.mm_positions,
                mm_hashes=new_req_data.mm_hashes,
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
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = self._get_last_token_position(
                req_state)

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
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
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        #self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _execute_model_for_text(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[SamplerOutput]:
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
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )

        sampled_output = self._sample(hidden_states, model_input)
        return sampled_output

    def _execute_model_for_multimodal_models(
        self,
        model_input: ModelInputForNeuron,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[SamplerOutput]:
        hidden_states = self.model.execute_model(model_input)
        sampled_output = self._sample(hidden_states, model_input)
        return sampled_output

    def _prepare_model_input(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelInputForNeuron:
        if self.is_chunked_prefill:
            chunked_prefill_model_input = self._prepare_chunked_prefill_inputs(
                scheduler_output)

            multi_modal_kwargs = None
            lora_adapter_ids = None

            return self._finalize_chunked_prefill_inputs(
                chunked_prefill_model_input,
                multi_modal_kwargs,
                lora_adapter_ids,
            )
        else:
            continuous_batching_model_input, is_prefill = self._prepare_continuous_batching_inputs(
                scheduler_output)
            return self._finalize_continuous_batching_inputs(
                continuous_batching_model_input,
                is_prefill,
            )

    def _process_multi_modal_data_neuron(
            self, mm_data: list[MultiModalKwargsItem]) -> None:
        assert len(
            mm_data
        ) <= 1, "Processing multiple MultiModalKwargsItem within one request is not yet supported"

        mm_data = self._make_mm_data_dict(mm_data[0])
        mm_data_neuron = None
        if self.model.model.config.model_type == 'llava':
            mm_data_neuron = self._process_multi_modal_data_neuron_llava(
                mm_data)
        elif self.model.model.config.model_type == 'llama4':
            mm_data_neuron = self._process_multi_modal_data_neuron_llama4(
                mm_data)
        else:
            raise NotImplementedError(
                f"processing mm data for model type {self.model.model.config.model_type} not supported on Neuron yet!"
            )
        return MultiModalKwargs.batch([mm_data_neuron])

    # NOTE: borrowed from PR #158
    # TODO: this helper seems like an anti-pattern (persisting deprecated interfaces). We should revisit and
    # see if we can conform to the new interfaces.
    def _make_mm_data_dict(self, mm_data):
        """
        Extract data from MultiModalFieldElem to adapt to the data format in _try_stack() of vllm/multimodal/inputs.py 
        """
        for k, v in mm_data.items():
            if isinstance(v, MultiModalFieldElem):
                assert k == v.key, f"the key of MultiModalKwargsItem is not the same as the key in its MultiModalFieldElem. {k} != {v.key}"
                mm_data[k] = v.data
        logger.debug(f"mm_data in _make_mm_data_dict: {mm_data}")
        return mm_data

    def _process_multi_modal_data_neuron_llava(self, mm_data):
        # We reconstruct image_sizes here to match HF's implementation
        # since vLLM implementation slices pixel_values for each image separately
        # (see vllm/model_executor/models/llava.py)
        if isinstance(mm_data["pixel_values"], torch.Tensor):
            logger.debug("pixel_values tensor shape: %s",
                         mm_data["pixel_values"].shape)
            img_height = mm_data["pixel_values"].shape[1]
            img_width = mm_data["pixel_values"].shape[2]
            mm_data["image_sizes"] = torch.tensor([img_height, img_width],
                                                  dtype=torch.int32)
        elif isinstance(mm_data["pixel_values"], list):
            image_sizes_list = []
            # The below logic pads multiple images within one request to
            # max height and width across all images
            # This mimics the same logic as HF processor
            max_height = 0
            max_width = 0
            for pixel_values in mm_data["pixel_values"]:
                logger.debug("pixel_values.shape: %s", pixel_values.shape)
                img_height = pixel_values.shape[1]
                img_width = pixel_values.shape[2]
                max_height = max(max_height, img_height)
                max_width = max(max_width, img_width)
                image_sizes_list.append(
                    torch.tensor([img_height, img_width], dtype=torch.int32))
            mm_data["image_sizes"] = torch.cat(image_sizes_list, dim=0)
            padded_pixel_values = []
            for pixel_values in mm_data["pixel_values"]:
                padded_pixel_value, _ = pad_tensor(
                    pixel_values,
                    [pixel_values.shape[0], max_height, max_width], 0)
                logger.debug("padded_pixel_value shape: %s",
                             padded_pixel_value.shape)
                padded_pixel_values.append(padded_pixel_value.unsqueeze(0))
            mm_data["pixel_values"] = torch.cat(padded_pixel_values, dim=0)
        logger.debug(
            f"mm_data in _process_multi_modal_data_neuron_llava: {mm_data}")
        return mm_data

    def _process_multi_modal_data_neuron_llama4(self, mm_data):
        """
        Extract data from MultiModalFieldElem to adapt to the data format in _try_stack() of vllm/multimodal/inputs.py 
        """
        for k, v in mm_data.items():
            if isinstance(v, MultiModalFieldElem):
                assert k == v.key, f"the key of mm_inputs is not the same as the key in it's MultiModalFieldElem. {k} != {v.key}"
                mm_data[k] = v.data
        logger.debug(
            f"mm_data in _process_multi_modal_data_neuron_llama4: {mm_data}")
        return mm_data

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
        logger.debug(f"num_scheduled_tokens: {num_scheduled_tokens}")

        for request_data in scheduler_output.scheduled_new_reqs:
            self._process_new_request_for_chunked_prefill(
                request_data, num_scheduled_tokens[request_data.req_id], data)

        cached_request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_request_data.req_ids):
            self._process_cached_request_for_chunked_prefill(
                cached_request_data, i, num_scheduled_tokens[req_id], data)

        return data

    def _prepare_continuous_batching_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Tuple[IntermediateInputData, bool]:
        """
        This function is used to prepare the inputs for continuous batching.
          *  For NewRequestData, it is guaranteed to be a prefill request.
          *  For CachedRequestData, it is guaranteed to be a decoding request.
        """
        data = IntermediateInputData()
        is_prefill = False
        for request_data in scheduler_output.scheduled_new_reqs:
            self._process_new_request_for_continuous_batching(
                request_data, data)
            is_prefill = True

        cached_request_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_request_data.req_ids):
            self._process_cached_request_for_continuous_batching(
                cached_request_data, i, data)

        return data, is_prefill

    def _process_new_request_for_continuous_batching(
            self, request_data: NewRequestData,
            data: IntermediateInputData) -> None:
        # Assign a free sequence id to the new request.
        assert request_data.req_id not in \
            self.vllm_req_to_neuron_seq_id_mapping, \
            (
                "Encountered an existing request ID "
                "while prefilling a new request"
            )
        assert self.free_seq_ids, "No free sequence ID available!"
        assigned_slot = self.free_seq_ids.pop()
        self.vllm_req_to_neuron_seq_id_mapping[
            request_data.req_id] = assigned_slot

        data.request_ids.append(request_data.req_id)

        data.input_tokens.append(request_data.prompt_token_ids)
        data.position_ids.append(
            list(range(len(request_data.prompt_token_ids))))
        data.input_block_ids.append(assigned_slot)

        data.full_context_lens.append(len(request_data.prompt_token_ids))
        data.prefill_completion_state.append(None)
        data.adapter_ids.append(
            self._prepare_adapter_id_in_new_request(request_data))

        if self.is_prefix_caching:
            self._process_new_request_for_continuous_batching_with_prefix_caching(
                request_data, data)

        if request_data.mm_kwargs:
            data.multi_modal_kwargs = self._process_multi_modal_data_neuron(
                request_data.mm_kwargs)

    def _process_new_request_for_continuous_batching_with_prefix_caching(
            self, request_data: NewRequestData,
            data: IntermediateInputData) -> None:

        assert len(request_data.block_ids) == 1
        block_table = copy.deepcopy(request_data.block_ids)[0]

        # pad the block_table to have the length of num_gpu_blocks
        block_size = self.cache_config.block_size
        max_len = self.scheduler_config.max_model_len
        max_blocks_per_seq = max_len // block_size
        padded_block_table = [self._BLOCK_TABLE_PAD] * max_blocks_per_seq
        padded_block_table[:len(block_table)] = block_table[:]
        data.block_tables.append(padded_block_table)

        data.computed_context_lens.append(request_data.num_computed_tokens)

        prompt_len = len(request_data.prompt_token_ids)
        slot_mapping_for_cur_seq = [
            (block_table[i // block_size] * block_size +
             i % block_size) if i < prompt_len else self._SLOT_MAPPING_PAD
            for i in range(max_len)
        ]
        data.slot_mapping.append(
            slot_mapping_for_cur_seq[request_data.num_computed_tokens:])

    def _process_cached_request_for_continuous_batching(
            self, request_data: CachedRequestData, index: int,
            data: IntermediateInputData) -> None:

        req_id = request_data.req_ids[index]
        assert req_id in \
            self.vllm_req_to_neuron_seq_id_mapping, \
            (
                "The request ID for the current decode request "
                " is not found in request to sequence ID "
                "mapping"
            )
        data.request_ids.append(req_id)
        state = self.requests[req_id]

        data.input_tokens.append([state.output_token_ids[-1]])

        position = self._get_last_token_position(state)

        data.position_ids.append([position])
        data.input_block_ids.append(
            self.vllm_req_to_neuron_seq_id_mapping[req_id])

        data.full_context_lens.append(position + 1)
        data.computed_context_lens.append(position)
        data.prefill_completion_state.append(None)
        data.adapter_ids.append(
            self._prepare_adapter_id_in_cached_request(req_id))

        if self.is_prefix_caching:
            self._process_cached_request_for_continuous_batching_with_prefix_caching(
                request_data, index, data)

    def _process_cached_request_for_continuous_batching_with_prefix_caching(
            self, request_data: CachedRequestData, index: int,
            data: IntermediateInputData) -> None:
        req_id = request_data.req_ids[index]
        state = self.requests[req_id]
        block_table = copy.deepcopy(state.block_ids)[0]

        attn_tkg_nki_kernel_enabled = (
            self.model.neuron_config.attn_tkg_nki_kernel_enabled
            or self.model.neuron_config.attn_block_tkg_nki_kernel_enabled)
        # Pad -1 to allow DMA skipping that is supported
        # by attention TKG kernel.
        block_table_padding = -1 if attn_tkg_nki_kernel_enabled \
                                    else self._BLOCK_TABLE_PAD
        block_size = self.cache_config.block_size
        max_len = self.scheduler_config.max_model_len
        max_blocks_per_seq = max_len // block_size
        padded_block_table = [block_table_padding] * max_blocks_per_seq
        padded_block_table[:len(block_table)] = block_table[:]
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
            input_tokens = make_tensor_with_pad(data.input_tokens,
                                                pad=0,
                                                max_len=max_seq_len,
                                                dtype=torch.long,
                                                device=self.device)
            position_ids = make_tensor_with_pad(data.position_ids,
                                                pad=0,
                                                max_len=max_seq_len,
                                                dtype=torch.long,
                                                device=self.device)
            input_block_ids = torch.tensor(data.input_block_ids,
                                           dtype=torch.long,
                                           device=self.device)
            slot_mapping = make_tensor_with_pad(
                data.slot_mapping,
                pad=self._SLOT_MAPPING_PAD,
                max_len=self.scheduler_config.max_model_len,
                dtype=torch.long,
                device=self.device)
            block_tables = torch.tensor(data.block_tables,
                                        dtype=torch.long,
                                        device=self.device)
            full_context_lens = torch.tensor(data.full_context_lens,
                                             dtype=torch.long,
                                             device=self.device).reshape(
                                                 -1, 1)
            computed_context_lens = torch.tensor(data.computed_context_lens,
                                                 dtype=torch.long,
                                                 device=self.device).reshape(
                                                     -1, 1)

        else:
            input_tokens = make_tensor_with_pad(data.input_tokens,
                                                pad=0,
                                                max_len=1,
                                                dtype=torch.long,
                                                device=self.device)
            position_ids = make_tensor_with_pad(data.position_ids,
                                                pad=0,
                                                max_len=1,
                                                dtype=torch.long,
                                                device=self.device)
            input_block_ids = torch.tensor(data.input_block_ids,
                                           dtype=torch.long,
                                           device=self.device)
            slot_mapping = torch.tensor(data.slot_mapping,
                                        dtype=torch.long,
                                        device=self.device)
            block_tables = torch.tensor(data.block_tables,
                                        dtype=torch.long,
                                        device=self.device)

            full_context_lens = torch.tensor(data.full_context_lens,
                                             dtype=torch.long,
                                             device=self.device).reshape(
                                                 -1, 1)

            # Convert computed_context_lens to tensor
            computed_context_lens = torch.tensor(data.computed_context_lens,
                                                 dtype=torch.long,
                                                 device=self.device).reshape(
                                                     -1, 1)
        lora_adapter_ids = None
        if self.lora_config is not None:
            lora_adapter_ids = torch.tensor(data.adapter_ids,
                                            dtype=torch.long,
                                            device=self.device)
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
            self, request_data: NewRequestData, num_scheduled_tokens: int,
            data: IntermediateInputData) -> None:
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
            data.slot_mapping.append(block_number *
                                     self.cache_config.block_size + offset)

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(
            end >= len(request_data.prompt_token_ids))

    def _process_cached_request_for_chunked_prefill(
            self, request_data: CachedRequestData, index: int,
            num_scheduled_tokens: int, data: IntermediateInputData) -> None:
        req_id = request_data.req_ids[index]
        data.request_ids.append(req_id)
        state = self.requests[req_id]
        logger.debug(f"for req_id {req_id}, state: {state}")
        block_table = copy.deepcopy(state.block_ids)[0]

        start = request_data.num_computed_tokens[index]
        end = start + num_scheduled_tokens

        if num_scheduled_tokens > 1:
            logger.debug(f"start: {start}, end: {end}")
            resumed_prompt_tokens = state.prompt_token_ids[start:end]
            data.input_tokens.extend(resumed_prompt_tokens)
            logger.debug(f"resumed prompt tokens: {resumed_prompt_tokens}")

        if len(state.output_token_ids) > 0:
            data.input_tokens.append(state.output_token_ids[-1])
            logger.debug(f"appended output token {state.output_token_ids[-1]}")
        data.position_ids.extend(range(start, end))
        data.input_block_ids.append(0)

        for i in range(start, end):
            block_number = block_table[i // self.cache_config.block_size]
            offset = i % self.cache_config.block_size
            data.slot_mapping.append(block_number *
                                     self.cache_config.block_size + offset)

        data.block_tables.append(block_table)
        data.full_context_lens.append(end)
        data.computed_context_lens.append(start)
        data.prefill_completion_state.append(
            end >= len(state.prompt_token_ids))

    def _finalize_chunked_prefill_inputs(
        self,
        data: IntermediateInputData,
        multi_modal_kwargs: BatchedTensorInputs,
        lora_adapter_ids: Optional[str],
    ) -> ModelInputForNeuron:
        device = self.device

        input_tokens = torch.tensor(data.input_tokens,
                                    dtype=torch.long,
                                    device=device).reshape(1, -1)
        position_ids = torch.tensor(data.position_ids,
                                    dtype=torch.long,
                                    device=device).reshape(1, -1)
        input_block_ids = torch.tensor(data.input_block_ids[:1],
                                       dtype=torch.long,
                                       device=device)
        slot_mapping = torch.tensor(data.slot_mapping,
                                    dtype=torch.long,
                                    device=device)

        max_blocks = max(len(b) for b in data.block_tables)
        for b in data.block_tables:
            b.extend([self._BLOCK_TABLE_PAD] * (max_blocks - len(b)))

        block_tables = torch.tensor(data.block_tables,
                                    dtype=torch.long,
                                    device=device)
        full_context_lens = torch.tensor(data.full_context_lens,
                                         dtype=torch.long,
                                         device=device)
        computed_context_lens = torch.tensor(data.computed_context_lens,
                                             dtype=torch.long,
                                             device=device)
        prefill_completion_state = torch.tensor(data.prefill_completion_state,
                                                dtype=torch.bool,
                                                device=device)

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

    def _sample(
        self,
        hidden_states: torch.Tensor,
        model_input: ModelInputForNeuron,
    ):

        logger.debug(f"output from model forward: {hidden_states=}")
        if model_input.prefill_completion_state is not None:
            for i, state in enumerate(model_input.prefill_completion_state):
                if not state.item():
                    hidden_states[i] = -1

        logger.debug(
            f"output after excluding partial prefill results: {hidden_states=}"
        )

        # The following logic reorders the model output to match the incoming request order
        # First obtain the order of requests processed by Neuron hardware
        request_id_order = {
            request_id: idx
            for idx, request_id in enumerate(model_input.request_ids)
        }

        # Identify the correct indices for each request in the original input batch based on request ids
        reorder_indices = torch.tensor([
            request_id_order[request_id]
            for request_id in self.input_batch.req_ids
        ])

        # Reorder along the batch dimension to restore outputs into the original request order
        hidden_states = hidden_states[reorder_indices]

        # Sample the next token.
        output = self.model.sample(logits=hidden_states, )
        return output

    def get_nxd_sampling_params(self, input_ids: torch.Tensor):
        if self.model.neuron_config.on_device_sampling_config:
            # TODO: fix bug when passing in sampling params via override_neuron_config
            max_topk = (
                self.model.neuron_config.on_device_sampling_config.global_topk)
        else:
            max_topk = self.model.neuron_config.vocab_size

        max_topk = min(max_topk, self._MAX_NEURON_SAMPLING_TOP_K)

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, request in enumerate(self.requests.values()):
            top_k[index] = (request.sampling_params.top_k
                            if request.sampling_params.top_k > 0
                            and request.sampling_params.top_k < max_topk else
                            max_topk)
            top_p[index] = request.sampling_params.top_p
            temperature[index] = request.sampling_params.temperature
            if request.sampling_params.temperature == 0.0:
                top_k[index] = 1
                temperature[index] = 1.0

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)

        if not self.is_chunked_prefill:
            if input_ids.shape[0] != sampling_params.shape[0]:
                sampling_params = sampling_params[:input_ids.shape[0]]

        return sampling_params

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)
