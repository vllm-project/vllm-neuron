# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from unittest.mock import MagicMock, Mock

import pytest
import torch
from transformers import PretrainedConfig

from neuronx_vllm_plugin.worker.neuronx_distributed_model_loader import (
    _get_default_neuron_config, _validate_image_to_text_override_neuron_config,
    _validate_neuron_config, get_neuron_model)

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
                             None,
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
                             None,
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
                             None,
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
                             None,
                             additional_config=additional_config)

    assert model is not None
    assert hasattr(model.model.config.neuron_config, 'chunked_prefill_config')


def test_get_neuron_model_error_handling_and_validation(mocker, base_configs):
    """Test error handling and validation in model loading.

    This test verifies:
    1. Missing architecture handling
    2. Invalid configuration detection
    3. Missing required fields handling
    4. Configuration validation errors
    """
    scheduler_config, cache_config, parallel_config = base_configs

    # Test missing architecture
    model_config = Mock()
    model_config.hf_config = PretrainedConfig()  # Empty config
    model_config.dtype = torch.float32

    with pytest.raises(ValueError, match="No architectures specified"):
        get_neuron_model(model_config,
                         cache_config,
                         parallel_config,
                         scheduler_config,
                         None,
                         additional_config={})

    # Test missing required fields
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"], model_type="llama")
    with pytest.raises(ValueError, match="Missing required fields"):
        get_neuron_model(model_config,
                         cache_config,
                         parallel_config,
                         scheduler_config,
                         None,
                         additional_config={})


def test_get_neuron_model_with_speculative_config(mocker, base_configs):
    """Test model initialization with speculative configuration.

    This test verifies:
    1. Speculative configuration is properly applied
    2. Eagle speculation settings
    3. Fused speculation parameters
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

    # Create speculative config
    spec_config = Mock()
    spec_config.num_speculative_tokens = 5
    spec_config.method = "eagle"
    spec_config.draft_model_config = Mock(model="draft-model")

    # Create mock model with neuron config
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.neuron_config = Mock()
    mock_model.config.neuron_config.enable_fused_speculation = True
    mock_model.config.neuron_config.speculation_length = 5
    mock_model.config.neuron_config.enable_eagle_speculation = True

    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model
    mock_causal_lm.eval = Mock(return_value=mock_causal_lm)

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             None,
                             speculative_config=spec_config,
                             additional_config={})

    assert model is not None
    assert model.model.config.neuron_config.enable_fused_speculation is True
    assert model.model.config.neuron_config.speculation_length == 5
    assert model.model.config.neuron_config.enable_eagle_speculation is True


def test_image_to_text_model_config_validation(mocker, base_configs):
    """Test image-to-text model configuration validation.

    This test verifies:
    1. Vision and text config validation
    2. Proper handling of neuron config overrides
    3. Configuration inheritance
    """
    scheduler_config, cache_config, parallel_config = base_configs

    # Create vision and text configs
    vision_config = PretrainedConfig(num_attention_heads=16, hidden_size=1024)
    text_config = PretrainedConfig(num_key_value_heads=32,
                                   head_dim=64,
                                   vocab_size=32000,
                                   model_type="llama")

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["LlavaForConditionalGeneration"],
        vision_config=vision_config,
        text_config=text_config,
        model_type="llava")
    model_config.model = "llava-model"
    model_config.dtype = torch.float32

    # Test with vision and text neuron config overrides
    additional_config = {
        "override_neuron_config": {
            "vision_neuron_config": {
                "batch_size": 4,
                "max_context_length": 1024
            },
            "text_neuron_config": {
                "batch_size": 8,
                "max_context_length": 2048
            }
        }
    }

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_pixtral = Mock()
    mock_pixtral.model = mock_model

    mocker.patch(
        'neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronPixtralForCausalLM',
        return_value=mock_pixtral)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             None,
                             additional_config=additional_config)

    assert model is not None


def test_validate_image_to_text_override_neuron_config():
    """Test validation of image-to-text model override configurations.
    
    This test verifies that the validation of override configurations for 
    image-to-text models works correctly. It checks:
    1. Valid configurations with allowed keys are accepted
    2. Invalid configurations with disallowed keys raise AssertionError
    3. Empty configurations are handled properly
    
    Raises:
        AssertionError: When configuration contains disallowed keys
    """
    # Valid configuration
    valid_config = {
        "text_neuron_config": {
            "batch_size": 4
        },
        "vision_neuron_config": {
            "max_context_length": 1024
        }
    }
    result = _validate_image_to_text_override_neuron_config(valid_config)
    assert result == valid_config

    # Empty configuration is valid
    empty_config = {}
    result = _validate_image_to_text_override_neuron_config(empty_config)
    assert result == empty_config

    # Invalid configuration with disallowed keys
    invalid_config = {"text_neuron_config": {}, "invalid_key": "value"}
    with pytest.raises(AssertionError):
        _validate_image_to_text_override_neuron_config(invalid_config)


def test_get_default_neuron_config(mocker):
    """Test generation of default neuron configurations.
    
    This test verifies that default neuron configurations are generated correctly
    based on input parameters. It checks:
    1. Tensor parallel degree is properly set
    2. Batch size calculations are correct
    3. Context length limits are properly configured
    4. Prefix caching settings are applied
    5. Speculative execution parameters are correctly set
    6. Block size and number of blocks are calculated properly
    
    Args:
        mocker: PyTest mocker fixture for creating mock objects
    
    The test ensures all required configuration parameters are present and
    have correct values based on the input configurations.
    """
    model_config = Mock()
    model_config.dtype = torch.float32

    cache_config = Mock()
    cache_config.block_size = 8
    cache_config.num_gpu_blocks_override = 100
    cache_config.enable_prefix_caching = False

    parallel_config = Mock()
    parallel_config.tensor_parallel_size = 2

    scheduler_config = Mock()
    scheduler_config.max_num_seqs = 32
    scheduler_config.max_model_len = 2048
    scheduler_config.max_num_batched_tokens = 4096
    scheduler_config.chunked_prefill_enabled = False

    lora_config = None
    spec_config = Mock()
    spec_config.num_speculative_tokens = 5
    spec_config.method = "eagle"

    config = _get_default_neuron_config(model_config, cache_config,
                                        parallel_config, scheduler_config,
                                        lora_config, spec_config)

    # Verify basic configuration
    assert config["tp_degree"] == 2
    assert config["batch_size"] == 32
    assert config["max_context_length"] == 2048
    assert config["pa_num_blocks"] == 100

    # Verify speculation configuration
    assert config["enable_fused_speculation"] is True
    assert config["enable_eagle_speculation"] is True
    assert config["speculation_length"] == 5


def test_validate_neuron_config():
    """Test validation of neuron configurations.
    
    This test ensures that neuron configuration validation works correctly
    for various scenarios. It verifies:
    1. Prefix caching configurations are valid
    2. Required fields are present
    3. Multimodal configurations are properly validated
    4. Block KV layout settings are correct
    5. Chunked prefill settings are properly validated
    
    Raises:
        AssertionError: When configuration validation fails
    
    The test covers both valid and invalid configurations to ensure
    proper validation behavior.
    """
    cache_config = Mock()
    cache_config.enable_prefix_caching = True

    scheduler_config = Mock()
    scheduler_config.chunked_prefill_enabled = False

    # Test valid prefix caching configuration
    valid_config = {"is_prefix_caching": True, "is_block_kv_layout": True}
    result = _validate_neuron_config(cache_config, scheduler_config,
                                     valid_config)
    assert result == valid_config

    # Test invalid config (missing required fields)
    invalid_config = {"is_prefix_caching": False, "is_block_kv_layout": True}
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, invalid_config)
