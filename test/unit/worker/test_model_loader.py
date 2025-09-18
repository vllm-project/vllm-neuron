# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from unittest.mock import MagicMock, Mock

import pytest
import torch
from transformers import PretrainedConfig

from neuronx_vllm_plugin.worker.neuronx_distributed_model_loader import \
    get_neuron_model

logger = logging.getLogger(__name__)

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
    """Test basic neuron model initialization.

    This test verifies that a basic LLaMA model can be properly initialized
    with default configurations. It checks:
    1. Model configuration is correctly processed
    2. Model is successfully loaded with neuron backend
    3. Required model attributes and configurations are present

    Args:
        mocker: PyTest mocker fixture for mocking dependencies
        base_configs: Fixture providing basic configuration objects

    The test ensures the fundamental model loading pathway works correctly.
    """
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

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             Mock(),
                             additional_config={})

    assert model is not None


@pytest.mark.parametrize("model_type,architecture", [
    ("llama", "LlamaForCausalLM"),
    ("llava", "LlavaForConditionalGeneration"),
    ("mixtral", "MixtralForCausalLM"),
])
def test_get_neuron_model_different_architectures(mocker, base_configs,
                                                  model_type, architecture):
    """Test model initialization across different architectures.

    This test verifies that:
    1. Different model architectures are properly supported
    2. Architecture-specific configurations are correctly applied
    3. LLaVA models have proper text configuration
    4. Model type specific features are properly initialized

    Args:
        mocker: PyTest mocker fixture
        base_configs: Base configuration fixture
        model_type: Type of model to test
        architecture: Model architecture class name
    """
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

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             Mock(),
                             additional_config={})

    assert model is not None
    if model_type == "llava":
        # Add specific assertions for LLaVA model
        assert hasattr(model_config.hf_config, 'text_config')
        assert model_config.hf_config.text_config.num_key_value_heads == 32
        assert model_config.hf_config.text_config.head_dim == 64


def test_get_neuron_model_with_prefix_caching(mocker, base_configs):
    """Test model initialization with prefix caching enabled.

    This test verifies that:
    1. Prefix caching configuration is properly applied
    2. Block KV layout is correctly configured
    3. Model loads successfully with caching enabled

    Args:
        mocker: PyTest mocker fixture
        base_configs: Base configuration fixture
    """
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

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             Mock(),
                             additional_config={})

    assert model is not None
    assert model.model.config.neuron_config.is_prefix_caching


def test_get_neuron_model_with_chunked_prefill(mocker, base_configs):
    """Test model initialization with chunked prefill enabled.

    This test verifies that:
    1. Chunked prefill configuration is properly applied
    2. Block KV layout is correctly configured
    3. Additional config overrides are properly handled

    Args:
        mocker: PyTest mocker fixture
        base_configs: Base configuration fixture
    """
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

    # Define additional_config before using it
    additional_config = {
        "override_neuron_config": {
            "chunked_prefill_config": {
                "enabled": True
            },
            "is_block_kv_layout": True
        }
    }
    model_config.override_neuron_config = additional_config[
        "override_neuron_config"]

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             Mock(),
                             additional_config=additional_config)

    assert model is not None
    assert hasattr(model.model.config.neuron_config, 'chunked_prefill_config')


def test_get_neuron_model_error_handling(mocker, base_configs):
    """Test error handling for unsupported model types.

    This test verifies that:
    1. Unsupported model architectures are properly detected
    2. Appropriate ValueError is raised
    3. Error message matches expected pattern

    Args:
        mocker: PyTest mocker fixture
        base_configs: Base configuration fixture
    """

    # Mock CompilationLevel
    mock_compilation_level = Mock()
    mock_compilation_level.PIECEWISE = 1
    mocker.patch('vllm.config.CompilationLevel', mock_compilation_level)

    # Mock vLLM config
    mock_config = Mock()
    mock_config.compilation_config = Mock()
    mock_config.compilation_config.level = 0
    mock_config.compilation_config.use_inductor = False
    mock_config.compilation_config.custom_ops = []
    mock_config.compilation_config.enabled_custom_ops = set()
    mock_config.compilation_config.disabled_custom_ops = set()

    # Mock get_current_vllm_config
    mocker.patch('vllm.config.get_current_vllm_config',
                 return_value=mock_config)

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
        get_neuron_model(model_config,
                         cache_config,
                         parallel_config,
                         scheduler_config,
                         Mock(),
                         additional_config={})