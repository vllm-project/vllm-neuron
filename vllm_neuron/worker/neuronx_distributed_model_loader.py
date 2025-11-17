# SPDX-License-Identifier: Apache-2.0
"""
A model loader implementation for NeuronX Distributed Inference (NxDI).

This class serves as the primary interface for loading and managing 
machine learning models optimized for AWS Neuron hardware. It provides 
functionality for:
    - Loading pre-trained models and their configurations
    - Managing model compilation
    - Handling distributed inference across multiple Neuron cores
    - Supporting various model architectures and configurations
    - Managing key-value caches for optimized inference
    - Implementing sampling strategies for model outputs

The loader supports various model architectures and can be extended to handle
different model types and configurations. It integrates with the broader 
vLLM framework while providing specific optimizations for AWS Neuron hardware.
"""

import collections
import copy
import hashlib
import logging
import os
import shutil
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Any, Union

import regex as re
import torch
import torch.nn as nn
from neuronx_distributed_inference.models.config import (  # yapf: disable
    ChunkedPrefillConfig, FusedSpecNeuronConfig, NeuronConfig,
    OnDeviceSamplingConfig)
from neuronx_distributed_inference.modules.lora_serving import \
    LoraServingConfig
from neuronx_distributed_inference.utils.constants import MODEL_TYPES
from neuronx_distributed_inference.utils.hf_adapter import \
    load_pretrained_config
from transformers import AutoModelForCausalLM, PretrainedConfig
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample import sampler as Sampler

from vllm_neuron.worker.constants import (NEURON_MULTI_MODAL_MODELS,
                                          TORCH_DTYPE_TO_NEURON_AMP)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronModelBase(nn.Module):
    """
    Base class for all Neuron models.
    It is used to load the model, run the model, and sample the model.
    It is also used to get the KV caches.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.logits_processor = LogitsProcessor(
            config.get_text_config().vocab_size, logits_as_input=True)
        self.on_device_sampling_disabled = bool(
            int(os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0")))
        if self.on_device_sampling_disabled:
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module
        self.kv_caches: list[Any] | None = None
        self.neuron_config: NeuronConfig
        self.is_reorder_needed: bool
        self.architecture: str
        self.num_key_value_heads: int
        self.head_size: int
        self.dtype: torch.dtype

    def forward(self, input_ids, positions, input_block_ids, sampling_params,
                **kwargs):
        raise NotImplementedError

    def sample(self, logits: torch.Tensor) -> SamplerOutput | None:
        raise NotImplementedError

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        raise NotImplementedError

    def get_kv_caches(self):
        if self.kv_caches is None:
            kv_caches = []
            tp_tensors_map = collections.defaultdict(list)
            state = self.model.context_encoding_model.model.nxd_model.state

            for tp_idx, per_tp_state in enumerate(state):
                for key, val in per_tp_state.items():
                    tp_tensors_map[tp_idx].append(val)

            for i in range(len(tp_tensors_map[0])):
                for tp, tensors in tp_tensors_map.items():
                    kv_caches.append(tensors[i])
            self.kv_caches = kv_caches

        return self.kv_caches

    @contextmanager
    def _reordered(self, input_block_ids: torch.Tensor, **inputs):
        """
        Context manager that yields reordered input_block_ids, inputs, and a restore function.
        Automatically restores output to original order if needed.
        
        [NOTE] This is MANDATORY for contiguous kv cache as it will impact the output accuracy.
        
        TODO: This sequence id reordering is better to live in NxD-Inference.
        """
        logger.debug(f"is_reorder_needed: {self.is_reorder_needed}")
        if self.is_reorder_needed:
            sorted_ids, sorted_indices = torch.sort(input_block_ids)
            reordered_inputs = self._sort_inputs(inputs, sorted_indices)

            def restore(output: torch.Tensor) -> torch.Tensor:
                if sorted_ids.shape[0] != 1:
                    return torch.index_select(output, 0,
                                              torch.argsort(sorted_indices))
                return output

            yield sorted_ids, reordered_inputs, restore
        else:
            yield input_block_ids, inputs, lambda x: x

    @staticmethod
    def _sort_inputs(inputs: dict[str, Any],
                     sorted_indices: torch.Tensor) -> dict[str, Any]:
        """Apply sorting to a dict of tensor/list inputs along batch dimension."""
        sorted_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] > 0:  # avoid empty tensors
                    if v.shape[0] != sorted_indices.shape[
                            0]:  # mm inputs are only sorted during prefill
                        logger.debug(
                            f"Skipping reorder for key {k} which has batch size {v.shape[0]} "
                            f"but sorted_indices has len {sorted_indices.shape[0]}"
                        )
                        sorted_inputs[k] = v
                        continue
                    sorted_inputs[k] = torch.index_select(v, 0, sorted_indices)
                else:
                    sorted_inputs[k] = v
            elif isinstance(v, list):
                sorted_inputs[k] = [v[i.item()] for i in sorted_indices]
            else:
                sorted_inputs[k] = v
        return sorted_inputs

    def _load_weights_common(self, model_name_or_path: str, neuronx_model_cls,
                             **kwargs):
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **kwargs['neuron_config'])

        config = kwargs.get('config') or neuronx_model_cls.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(model_name_or_path))

        # If fused speculation is enabled, attach the draft model config.
        if getattr(neuron_config, "enable_fused_speculation", False):
            assert kwargs.get("speculative_config") is not None, (
                "Must pass speculative_config to load weights if using Neuron Speculation."
            )
            self._init_fused_spec_config(
                config,
                neuronx_model_cls,
                kwargs["speculative_config"],
            )

        hashed_config = hashlib.md5(
            config.to_json_string().encode('utf-8')).hexdigest()
        compiled_model_path = self._get_compiled_model_path(
            model_name_or_path, hashed_config)

        try:
            self._load_compiled_model(compiled_model_path, neuronx_model_cls,
                                      kwargs)
            return True, compiled_model_path, config
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Exception: {e}")
            logger.warning(
                f"Unable to find precompiled artifacts from {compiled_model_path}. Recompiling..."
            )
            return False, compiled_model_path, config

    def _get_compiled_model_path(self, model_name_or_path: str,
                                 hashed_config: str):
        if os.getenv("NEURON_COMPILED_ARTIFACTS"):
            return os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            path = Path(model_name_or_path
                        ) / "neuron-compiled-artifacts" / hashed_config
            path.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(path, ignore_errors=True)
            return path
        else:
            path = Path(
                "local-models"
            ) / model_name_or_path / "neuron-compiled-artifacts" / hashed_config
            path.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(path, ignore_errors=True)
            return path

    def _load_compiled_model(self, compiled_model_path: str, neuronx_model_cls,
                             kwargs):
        self.model = neuronx_model_cls(compiled_model_path)

        self.model.load(compiled_model_path)
        logger.info("Successfully loaded pre-compiled model artifacts from %s",
                    compiled_model_path)
        # When loading a pre-compiled model don't do any more overrides.
        override_neuron_config = kwargs.get("override_neuron_config")
        if override_neuron_config:
            logger.warning(
                "Using pre-compiled artifacts, override_neuron_config will be ignored"
            )

    def _save_pretrained_model(self, model_name: str):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path

    def _compile_and_load_model(self, model_path: str, neuronx_model_cls,
                                config, compiled_path: str):
        self.model = neuronx_model_cls(model_path, config)
        # Quantize model.
        if config.neuron_config.quantized:
            neuronx_model_cls.save_quantized_state_dict(model_path, config)
        self.model.compile(compiled_path)
        self.model.load(compiled_path)

    def _init_fused_spec_config(self, config, neuronx_model_cls,
                                speculative_config):
        """
        Initialize and attach a `fused_spec_config` to the target model's
        NeuronXDistributed config when fused speculation is enabled.

        Behavior:
        • Clone the target model's `neuron_config` to build a draft config.
        • Force-disable `enable_fused_speculation` in the draft to prevent
            recursion (only the target should run fused speculation).
        • Zero out `speculation_length` for the draft unless EAGLE is used,
            since the target controls speculation length.
        • Remove deprecated fields (`trace_tokengen_model` is now inferred
            automatically by NxDI and should not be set explicitly).
        • Apply any optional overrides such as
            `draft_model_modules_to_not_convert`.
        • For EAGLE drafts, mark `is_eagle_draft=True`
        • Load the draft HF config and wrap everything in a
            `FusedSpecNeuronConfig`, which is then attached to the target's
            config.

        Args:
            config: The target model's NxDI config (will be modified in place
                    to attach `fused_spec_config`).
            neuronx_model_cls: The NxDI model class providing config loaders.
            speculative_config: vLLM's `SpeculativeConfig` describing the draft
                    model path and speculation parameters.

        Notes:
            Only the **target** model should have
            `neuron_config.enable_fused_speculation=True`. The draft must not,
            otherwise NxDI would attempt to compile nested fused speculation.
        """
        draft_neuron_config = copy.deepcopy(config.neuron_config)

        if not getattr(config.neuron_config, "enable_eagle_speculation",
                       False):
            draft_neuron_config.speculation_length = 0

        draft_neuron_config.enable_fused_speculation = False

        if getattr(config.neuron_config, "draft_model_modules_to_not_convert",
                   None):
            draft_neuron_config.modules_to_not_convert = (
                draft_neuron_config.draft_model_modules_to_not_convert)

        if getattr(config.neuron_config, "enable_eagle_speculation", False):
            draft_neuron_config.is_eagle_draft = True

        draft_config = neuronx_model_cls.get_config_cls()(
            draft_neuron_config,
            load_config=load_pretrained_config(
                speculative_config.draft_model_config.model),
        )

        fused_spec_config = FusedSpecNeuronConfig(
            neuronx_model_cls._model_cls,
            draft_config=draft_config,
            draft_model_path=speculative_config.draft_model_config.model,
        )
        config.fused_spec_config = fused_spec_config


class NeuronCausalLM(NeuronModelBase):

    def _remask_fused_spec_output(self, fused, inputs):
        """
        Handle NxDI fused speculation output.

        NxDI fused spec returns:
        fused[0] = accepted_tokens_with_padding : [B, T], 0-padded
        fused[-1] = next_pos_ids

        We convert the 0-padding to -1 past the number of tokens actually
        generated in this step, so the runner can strip pads.
        """
        accepted_tokens_with_padding = fused[0]
        next_pos_ids = fused[-1].squeeze(-1)  # [B]
        positions_vec = inputs["position_ids"][:, -1].to(next_pos_ids.device)

        # Number of tokens generated this step
        generated_token_counts = (next_pos_ids - positions_vec).to(torch.long)

        # Mask tail with -1 so runner can strip pads
        B, T = accepted_tokens_with_padding.shape
        generated_token_counts = generated_token_counts.clamp_(0, T)

        masked = accepted_tokens_with_padding.clone()
        for b in range(B):
            masked[b, generated_token_counts[b]:] = -1

        return masked

    def forward(self, input_ids, input_block_ids, **kwargs):
        with self._reordered(input_block_ids, input_ids=input_ids,
                             **kwargs) as (sorted_ids, inputs, restore):
            output = self.model(
                inputs['input_ids'],
                attention_mask=None,
                seq_ids=sorted_ids,
                block_table=inputs['block_tables'],
                **{
                    k: v
                    for k, v in inputs.items() if k not in
                    ['input_ids', 'block_tables', 'prefill_completion_state']
                })

            if self.model.config.neuron_config.on_device_sampling_config:
                output = output.hidden_states
                if getattr(self.model.config.neuron_config,
                           "enable_fused_speculation", False):
                    fused = output
                    output = self._remask_fused_spec_output(fused, inputs)
            else:
                if self.neuron_config.is_chunked_prefill:
                    assert kwargs.get('prefill_completion_state') is not None
                    idx_for_sampling = kwargs[
                        'prefill_completion_state'].nonzero().flatten()
                    output = output.logits[0, idx_for_sampling, :]
                else:
                    output = output.logits[:, -1, :]

            return restore(output)

    def sample(self, logits: torch.Tensor) -> SamplerOutput | None:
        if self.model.config.neuron_config.on_device_sampling_config:
            return SamplerOutput(
                # The sampled tokens are expanded to 2D tensor with shape
                # [num_requests, 1], where each row represents one generated
                # token per request.
                sampled_token_ids=logits.unsqueeze(-1),
                logprobs_tensors=None,
            )
        else:
            # CPU sampling is now handled by the model runner
            # This should not be called when on_device_sampling_config is None
            # as the model runner will use its own CPU sampler
            raise RuntimeError(
                "CPU sampling should be handled by the model runner, not the model. "
                "This indicates a bug in the sampling path routing.")

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        neuronx_model_cls = _get_neuron_model_cls(architecture)
        success, compiled_model_path, config = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, **kwargs)

        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(
                    model_name_or_path)
            self._compile_and_load_model(model_name_or_path, neuronx_model_cls,
                                         config, compiled_model_path)
        return success, compiled_model_path


class NeuronMultiModalCausalLM(NeuronCausalLM):

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        neuronx_model_cls = _get_neuron_model_cls(architecture)

        # Neuron ImageToText model configs have nested text and vision config
        # each has their own neuron_config. The structure looks like:
        # ImageToTextInferenceConfig
        # ├── text_config
        # |   ├── text_neuron_config
        # |   |   └── ... ...
        # |   ├── text_config_arg0
        # |   └── ... ...
        # ├── vision_config
        # |   ├── vision_neuron_config
        # |   |   └── ... ...
        # |   ├── vision_config_arg0
        # |   └── ... ...
        # └── neuron_config (default to same as text_neuron_config)
        # so we override text and vision neuron_config individually

        default_neuron_config = kwargs["neuron_config"]
        override_neuron_config = _validate_image_to_text_override_neuron_config(
            kwargs["override_neuron_config"])

        vision_neuron_config = copy.deepcopy(default_neuron_config)
        vision_neuron_config.update(
            override_neuron_config.get("vision_neuron_config", {}))
        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **vision_neuron_config)

        text_neuron_config = copy.deepcopy(default_neuron_config)
        text_neuron_config.update(
            override_neuron_config.get("text_neuron_config", {}))
        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **text_neuron_config)

        config = neuronx_model_cls.get_config_cls()(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            load_config=load_pretrained_config(model_name_or_path))

        success, compiled_model_path, _ = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, config=config, **kwargs)

        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(
                    model_name_or_path)

            self._compile_and_load_model(model_name_or_path, neuronx_model_cls,
                                         config, compiled_model_path)
        return success, compiled_model_path

    def execute_model(self, model_input, **kwargs):
        """Helper to run model with multimodal inputs."""

        pixel_values = None
        if (model_input.multi_modal_kwargs is not None
                and model_input.multi_modal_kwargs.get("pixel_values")
                is not None):
            pixel_values = model_input.multi_modal_kwargs["pixel_values"]

        hidden_states = self.forward(
            input_ids=model_input.input_tokens,
            positions=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            sampling_params=model_input.sampling_params,
            pixel_values=pixel_values,
            **kwargs)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with multimodal support for multi-modal model."""
        with self._reordered(
                input_block_ids,
                input_ids=input_ids,
                positions=positions,
                sampling_params=sampling_params,
                pixel_values=pixel_values,
                vision_mask=vision_mask,
                **kwargs,
        ) as (sorted_ids, inputs, restore):

            output = self.model(
                inputs["input_ids"].to(torch.int32),
                attention_mask=None,
                position_ids=inputs["positions"].to(torch.int32),
                seq_ids=sorted_ids.flatten().to(torch.int32),
                pixel_values=inputs.get("pixel_values"),
                vision_mask=inputs.get("vision_mask"),
                sampling_params=inputs["sampling_params"],
            )

            if self.model.config.neuron_config.on_device_sampling_config:
                output = output.hidden_states
            else:
                output = output.logits[:, -1, :]

            return restore(output)


class NeuronPixtralForCausalLM(NeuronMultiModalCausalLM):

    def execute_model(self, model_input):
        """Helper to run model with defaults for missing multimodal inputs."""
        vision_mask = (model_input.input_tokens ==
                       self.model.config.image_token_index).unsqueeze(-1)

        if model_input.multi_modal_kwargs is not None and model_input.multi_modal_kwargs.get(
                "pixel_values") is not None:
            image_sizes = model_input.multi_modal_kwargs.get("image_sizes")
        else:
            image_sizes = torch.tensor([[512, 512]], dtype=torch.int32)

        return super().execute_model(model_input,
                                     vision_mask=vision_mask,
                                     image_sizes=image_sizes)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
        pixel_values: Union[torch.Tensor, list] | None = None,
        image_sizes: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with multimodal support."""

        # Cast vision tensors to the configured dtype
        if pixel_values is not None:
            dtype = self.model.config.vision_config.neuron_config.torch_dtype
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(dtype)
            elif isinstance(pixel_values, list):
                pixel_values = [p.to(dtype) for p in pixel_values]

        return super().forward(input_ids,
                               positions,
                               input_block_ids=input_block_ids,
                               sampling_params=sampling_params,
                               pixel_values=pixel_values,
                               vision_mask=vision_mask,
                               image_sizes=image_sizes,
                               **kwargs)


class NeuronLlama4ForCausalLM(NeuronMultiModalCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.vision_token_id = None

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        success, compiled_model_path = super().load_weights(
            model_name_or_path, architecture, **kwargs)

        # Load tokenizer to get vision token ID
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vision_token_id = tokenizer("<|image|>",
                                         add_special_tokens=False).input_ids[0]
        return success, compiled_model_path

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with multimodal support for Llama4."""

        if pixel_values is not None:
            logger.debug(f"pixel_values.shape = {pixel_values.shape}")
            if len(pixel_values.shape) == 5:
                bsz, n_chunks, n_channels, h, w = pixel_values.shape  # (1, 5, 3, 336, 336)
                pixel_values = pixel_values.reshape(bsz * n_chunks, n_channels,
                                                    h, w)  # (5, 3, 336, 336)
                pixel_values = pixel_values.to(torch.bfloat16)
        if pixel_values is not None and vision_mask is None:
            vision_mask = (
                input_ids == self.model.config.image_token_index).unsqueeze(-1)

        if vision_mask is not None:
            vision_mask = vision_mask.to(torch.bool)

        # Ensure sampling params match input batch size
        if input_ids.shape[0] != sampling_params.shape[0]:
            sampling_params = sampling_params[:input_ids.shape[0]]

        return super().forward(input_ids, positions, input_block_ids,
                               sampling_params, pixel_values, vision_mask,
                               **kwargs)


def _get_model_configs(config: PretrainedConfig) -> str:
    logger.debug(f"PretrainedConfig: {config}")

    archs = getattr(config, "architectures", [])
    if not archs:
        raise ValueError(
            "No architectures specified in the pretrained config.")
    architecture = archs[0]
    if architecture in NEURON_MULTI_MODAL_MODELS:
        config = getattr(config, "text_config", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        num_attention_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        if num_attention_heads and hidden_size:
            head_dim = hidden_size // num_attention_heads
    if not num_key_value_heads or not head_dim:
        raise ValueError("Missing required fields in the pretrained config.")
    return architecture, int(num_key_value_heads), int(head_dim)


def _camel_to_kebab(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()


def _get_neuron_model_cls(architecture: str):
    try:
        if "For" in architecture:
            model, task = architecture.split("For", 1)
            if task == "ConditionalGeneration":
                task = "CausalLM"  # to match NxDI class names for Mllama and Pixtral
            model, task = model.lower(), _camel_to_kebab(task)

            if model == "qwen3moe":
                model = "qwen3_moe"

            if architecture == "LlavaForConditionalGeneration":
                model = "pixtral"

            return MODEL_TYPES[model][task]
        else:
            raise KeyError
    except KeyError:
        raise ValueError(
            f"Model {architecture} is not supported on Neuron for now. Supported models: {list(MODEL_TYPES.keys())}"
        )


def get_neuron_model(model_config: ModelConfig,
                     cache_config: CacheConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig,
                     lora_serving_config: LoraServingConfig,
                     speculative_config: SpeculativeConfig | None = None,
                     additional_config: Any | None = None) -> nn.Module:
    architecture, num_key_value_heads, head_dim = _get_model_configs(
        model_config.hf_config)

    if architecture == "LlavaForConditionalGeneration":
        raise NotImplementedError(
            "Pixtral is not yet supported on the Neuron plugin")
        model = NeuronPixtralForCausalLM(model_config.hf_config)
    elif architecture == "Llama4ForConditionalGeneration":
        model = NeuronLlama4ForCausalLM(model_config.hf_config)
    else:
        model = NeuronCausalLM(model_config.hf_config)

    if lora_serving_config:
        raise NotImplementedError(
            "Multi-lora is not yet supported on the Neuron plugin")

    default_neuron_config_args = _get_default_neuron_config(
        model_config, cache_config, parallel_config, scheduler_config,
        lora_serving_config, speculative_config)

    override_neuron_config = additional_config.get("override_neuron_config",
                                                   None)
    if override_neuron_config is not None:
        logger.info(
            f"Retrieved override_neuron_config from additional_config: {override_neuron_config}"
        )
    else:
        logger.info(
            f"No neuron overrides are passed via additional_config: {additional_config}. Proceeding with defaults."
        )

    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, override_neuron_config)

    # Handle pa_num_blocks increment logic before validation
    if neuron_config.get("is_block_kv_layout"):
        neuron_config = _handle_pa_num_blocks(cache_config, neuron_config,
                                              override_neuron_config)

    neuron_config = _validate_neuron_config(cache_config, scheduler_config,
                                            neuron_config)

    model.load_weights(model_name_or_path=model_config.model,
                       architecture=architecture,
                       neuron_config=neuron_config,
                       override_neuron_config=override_neuron_config,
                       speculative_config=speculative_config)
    model.neuron_config = model.model.config.neuron_config
    model.architecture = architecture
    model.num_key_value_heads = num_key_value_heads
    model.head_dim = head_dim

    return model.eval()


# Helper functions for getting default configs
def _get_default_neuron_config(model_config: ModelConfig,
                               cache_config: CacheConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig,
                               lora_serving_config: LoraServingConfig,
                               speculative_config: SpeculativeConfig | None):
    on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True,
                                                       deterministic=False)

    if scheduler_config.chunked_prefill_enabled:
        batch_size = 1
        max_context_length = scheduler_config.max_num_batched_tokens
    else:
        batch_size = scheduler_config.max_num_seqs
        max_context_length = scheduler_config.max_model_len

    default_num_blocks = ceil(
        scheduler_config.max_model_len //
        cache_config.block_size) * scheduler_config.max_num_seqs
    if cache_config.num_gpu_blocks_override is not None:
        default_num_blocks = cache_config.num_gpu_blocks_override

    logger.debug(
        f"Setting num_blocks to {default_num_blocks} in the default neuron config."
    )

    neuron_config = {
        "tp_degree":
        parallel_config.tensor_parallel_size,
        "ctx_batch_size":
        1,
        "batch_size":
        batch_size,
        "max_context_length":
        max_context_length,
        "seq_len":
        scheduler_config.max_model_len,
        "enable_bucketing":
        True,
        "is_continuous_batching": (batch_size > 1),
        "quantized":
        False,
        "torch_dtype":
        TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        "padding_side":
        "right",
        "on_device_sampling_config":
        on_device_sampling_config,
        "lora_config":
        lora_serving_config,
        "pa_num_blocks":
        default_num_blocks,
        "pa_block_size":
        cache_config.block_size,
        "is_block_kv_layout": (scheduler_config.chunked_prefill_enabled
                               or cache_config.enable_prefix_caching),
        "is_prefix_caching":
        cache_config.enable_prefix_caching,
    }

    # Enable fused speculation flags when requested
    if speculative_config is not None:
        neuron_config["enable_fused_speculation"] = True
        neuron_config["speculation_length"] = getattr(
            speculative_config, "num_speculative_tokens", 0)
        if getattr(speculative_config, "method", None) == "eagle":
            neuron_config["enable_eagle_speculation"] = True

    return neuron_config


def _handle_pa_num_blocks(cache_config: CacheConfig, neuron_config: dict,
                          override_neuron_config: dict) -> dict:
    """Handle the pa_num_blocks increment logic to ensure vLLM and NxDI have consistent block counts."""
    if cache_config.num_gpu_blocks_override is not None:
        pa_num_blocks = neuron_config.get("pa_num_blocks")
        original_user_override = cache_config.num_gpu_blocks_override - 1  # Remove the +1 increment to get original user value

        # Check if pa_num_blocks was explicitly set in override_neuron_config
        pa_num_blocks_explicitly_set = override_neuron_config and "pa_num_blocks" in override_neuron_config

        if pa_num_blocks_explicitly_set:
            # User explicitly set pa_num_blocks, it must match their original intent
            if pa_num_blocks == original_user_override:
                # User provided original intended value, increment pa_num_blocks to match the incremented num_gpu_blocks_override
                neuron_config[
                    "pa_num_blocks"] = cache_config.num_gpu_blocks_override
                logger.info(
                    f"User provided pa_num_blocks ({pa_num_blocks}) matching original --num-gpu-blocks-override intent. "
                    f"Incrementing pa_num_blocks to {cache_config.num_gpu_blocks_override} to match the increment for a null block in vllm."
                )
            else:
                # pa_num_blocks doesn't match the original user intent, this creates a mismatch
                raise ValueError(
                    f"pa_num_blocks ({pa_num_blocks}) must match your --num-gpu-blocks-override intent({original_user_override}) to ensure vLLM and NxDI have consistent block counts. "
                )

    else:
        # User didn't set num_gpu_blocks_override, check if they explicitly set pa_num_blocks
        pa_num_blocks_explicitly_set = override_neuron_config and "pa_num_blocks" in override_neuron_config

        if pa_num_blocks_explicitly_set:
            # User set pa_num_blocks without num_gpu_blocks_override
            raise ValueError(f"When setting pa_num_blocks ({neuron_config.get('pa_num_blocks')}) in override_neuron_config, " \
            f"you must also set --num-gpu-blocks-override to the same value to ensure vLLM and NxDI have consistent block counts.")

    return neuron_config


def _validate_neuron_config(cache_config: CacheConfig,
                            scheduler_config: SchedulerConfig,
                            neuron_config: dict):
    if cache_config.enable_prefix_caching:
        assert neuron_config.get("is_prefix_caching", False)
        assert neuron_config.get("is_block_kv_layout", False)

    if scheduler_config.chunked_prefill_enabled:
        assert neuron_config.get("chunked_prefill_config")
        assert neuron_config.get("is_block_kv_layout", False)

    if neuron_config.get("is_block_kv_layout"):
        min_blocks_required = ceil(
            scheduler_config.max_model_len /
            cache_config.block_size) * scheduler_config.max_num_seqs

        # Calculate effective blocks based on whether num_gpu_blocks_override was set
        if cache_config.num_gpu_blocks_override is not None:
            # User set num_gpu_blocks_override, so the effective blocks = original user intent
            effective_blocks = cache_config.num_gpu_blocks_override - 1
        else:
            # No override set, pa_num_blocks contains the raw calculated value (no increment applied)
            effective_blocks = neuron_config.get("pa_num_blocks")

        assert effective_blocks >= min_blocks_required, \
        f"At least {min_blocks_required} blocks are required for max_model_len {scheduler_config.max_model_len}, but only {effective_blocks} blocks are available (user-intended blocks, excluding the +1 for null block)"

    assert "text_neuron_config" not in neuron_config, \
        "text_neuron_config should not be in the default neuron_config. It should be initialized in specific ImageToText models."
    assert "vision_neuron_config" not in neuron_config, \
        "vision_neuron_config should not be in the default neuron_config. It should be initialized in specific ImageToText models."

    logger.debug("Neuron Config: %s", neuron_config)
    return neuron_config


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    overridden_neuron_config = overridden_neuron_config or {}
    cfg = overridden_neuron_config.pop("chunked_prefill_config", None)
    if cfg:
        overridden_neuron_config[
            "chunked_prefill_config"] = ChunkedPrefillConfig(**cfg)
    default_neuron_config.update(overridden_neuron_config)

    # Let specific ImageToText models handle the text and vision neuron config overrides
    if "text_neuron_config" in default_neuron_config:
        default_neuron_config.pop("text_neuron_config")
    if "vision_neuron_config" in default_neuron_config:
        default_neuron_config.pop("vision_neuron_config")

    # Get quantization config if specified
    if "quantized" in overridden_neuron_config:
        quantization_cfg = {
            "quantized":
            overridden_neuron_config.pop("quantized", False),
            "quantized_checkpoints_path":
            overridden_neuron_config.pop("quantized_checkpoints_path", None),
            "quantization_type":
            overridden_neuron_config.pop("quantization_type",
                                         "per_tensor_symmetric"),
            "quantization_dtype":
            overridden_neuron_config.pop("quantization_dtype", "int8"),
        }
        default_neuron_config.update(quantization_cfg)
    logger.debug("Neuron Config after override: %s", default_neuron_config)
    return default_neuron_config


def _validate_image_to_text_override_neuron_config(
        override_neuron_config: dict):
    allowed_keys = {"text_neuron_config", "vision_neuron_config"}
    assert len(override_neuron_config) == 0 or (override_neuron_config.keys() <= allowed_keys), \
        f"override_neuron_config for ImageToText models can only contain keys {allowed_keys}, got {override_neuron_config.keys()}"

    logger.debug("Override Neuron Config: %s", override_neuron_config)
    return override_neuron_config
