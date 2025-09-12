# SPDX-License-Identifier: Apache-2.0
# test/unit/worker/test_model_loader.py
import sys
from unittest.mock import MagicMock, Mock

import pytest
import torch
from transformers import PretrainedConfig

# Create a base mock module
mock_base = MagicMock()
mock_base.utils = MagicMock()
mock_base.utils.constants = MagicMock()
mock_base.utils.constants.MODEL_TYPES = {
    'llama': 'llama',
    'llava': 'llava',
    'mixtral': 'mixtral'
}
mock_base.utils.hf_adapter = MagicMock()
mock_base.models = MagicMock()
mock_base.models.config = MagicMock()
mock_base.modules = MagicMock()
mock_base.modules.lora_serving = MagicMock()
mock_base.modules.generation = MagicMock()
mock_base.modules.generation.sampling = MagicMock()
mock_base.modules.padding = MagicMock()

# Install the mock module
sys.modules['neuronx_distributed_inference'] = mock_base
sys.modules['neuronx_distributed_inference.utils'] = mock_base.utils
sys.modules[
    'neuronx_distributed_inference.utils.constants'] = mock_base.utils.constants
sys.modules[
    'neuronx_distributed_inference.utils.hf_adapter'] = mock_base.utils.hf_adapter
sys.modules['neuronx_distributed_inference.models'] = mock_base.models
sys.modules[
    'neuronx_distributed_inference.models.config'] = mock_base.models.config
sys.modules['neuronx_distributed_inference.modules'] = mock_base.modules
sys.modules[
    'neuronx_distributed_inference.modules.lora_serving'] = mock_base.modules.lora_serving
sys.modules[
    'neuronx_distributed_inference.modules.generation'] = mock_base.modules.generation
sys.modules[
    'neuronx_distributed_inference.modules.generation.sampling'] = mock_base.modules.generation.sampling
sys.modules[
    'neuronx_distributed_inference.modules.padding'] = mock_base.modules.padding

from neuronx_vllm_plugin.worker.neuronx_distributed_model_loader import \
    get_neuron_model

@pytest.fixture
def base_configs():
    scheduler_config = Mock()
    scheduler_config.max_model_len = 2048
    scheduler_config.max_num_seqs = 32
    scheduler_config.chunked_prefill_enabled = False
    scheduler_config.max_num_batched_tokens = 4096

    cache_config = Mock()
    cache_config.block_size = 8
    cache_config.num_gpu_blocks_override = None
    cache_config.enable_prefix_caching = False

    parallel_config = Mock()
    parallel_config.tensor_parallel_size = 1

    return scheduler_config, cache_config, parallel_config


def test_get_neuron_model(mocker, base_configs):
    scheduler_config, cache_config, parallel_config = base_configs

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"],
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="llama")
    model_config.model = "meta-llama/Llama-2-7b-hf"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = None

    # Mock the model loading
    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config, cache_config, parallel_config,
                             scheduler_config, Mock())

    assert model is not None


@pytest.mark.parametrize("model_type,architecture", [
    ("llama", "LlamaForCausalLM"),
    ("llava", "LlavaForConditionalGeneration"),
    ("mixtral", "MixtralForCausalLM"),
])
def test_get_neuron_model_different_architectures(mocker, base_configs,
                                                  model_type, architecture):
    scheduler_config, cache_config, parallel_config = base_configs

    # Create text config for LLaVA
    text_config = PretrainedConfig(
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="llama"  # LLaVA uses LLaMA as base
    )

    # Create the main config
    model_config = Mock()
    if model_type == "llava":
        model_config.hf_config = PretrainedConfig(
            architectures=[architecture],
            text_config=text_config,  # Add text_config for LLaVA
            model_type=model_type)
    else:
        model_config.hf_config = PretrainedConfig(architectures=[architecture],
                                                  num_key_value_heads=32,
                                                  head_dim=64,
                                                  vocab_size=32000,
                                                  model_type=model_type)

    model_config.model = f"test/{model_type}-model"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = None

    # Mock the model loading
    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model

    if architecture == "LlavaForConditionalGeneration":
        mocker.patch(
            'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronPixtralForCausalLM',
            return_value=mock_causal_lm)
    else:
        mocker.patch(
            'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
            return_value=mock_causal_lm)

    model = get_neuron_model(model_config, cache_config, parallel_config,
                             scheduler_config, Mock())

    assert model is not None
    if model_type == "llava":
        # Add specific assertions for LLaVA model
        assert hasattr(model_config.hf_config, 'text_config')
        assert model_config.hf_config.text_config.num_key_value_heads == 32
        assert model_config.hf_config.text_config.head_dim == 64


def test_get_neuron_model_with_prefix_caching(mocker, base_configs):
    scheduler_config, cache_config, parallel_config = base_configs
    cache_config.enable_prefix_caching = True

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"],
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="llama")
    model_config.model = "meta-llama/Llama-2-7b-hf"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = {
        "is_prefix_caching": True,
        "is_block_kv_layout": True
    }

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config, cache_config, parallel_config,
                             scheduler_config, Mock())

    assert model is not None
    assert model.model.config.neuron_config.is_prefix_caching


def test_get_neuron_model_with_chunked_prefill(mocker, base_configs):
    scheduler_config, cache_config, parallel_config = base_configs
    scheduler_config.chunked_prefill_enabled = True

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"],
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="llama")
    model_config.model = "meta-llama/Llama-2-7b-hf"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = {
        "chunked_prefill_config": {
            "enabled": True
        },
        "is_block_kv_layout": True
    }

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config, cache_config, parallel_config,
                             scheduler_config, Mock())

    assert model is not None
    assert hasattr(model.model.config.neuron_config, 'chunked_prefill_config')


def test_get_neuron_model_error_handling(mocker, base_configs):
    scheduler_config, cache_config, parallel_config = base_configs

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["UnsupportedModel"],
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="unsupported")
    model_config.model = "unsupported/model"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = None

    with pytest.raises(ValueError, match="Model .* is not supported"):
        get_neuron_model(model_config, cache_config, parallel_config,
                         scheduler_config, Mock())
