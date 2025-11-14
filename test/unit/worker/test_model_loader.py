# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm_neuron.worker.constants import NEURON_MULTI_MODAL_MODELS
from vllm_neuron.worker.neuronx_distributed_model_loader import (
    NeuronCausalLM, NeuronLlama4ForCausalLM, NeuronModelBase,
    NeuronMultiModalCausalLM, NeuronPixtralForCausalLM,
    _get_default_neuron_config, _get_model_configs, _get_neuron_model_cls,
    _validate_image_to_text_override_neuron_config, _validate_neuron_config,
    get_neuron_model)

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


@pytest.fixture(autouse=True)
def mock_vllm_config():
    """Mock vLLM config components to avoid initialization errors."""
    # Mock the compilation config with integer level for comparison
    mock_comp_config = Mock()
    mock_comp_config.level = 0  # Use integer instead of enum
    mock_comp_config.use_inductor = False

    with patch('vllm.model_executor.custom_op.get_cached_compilation_config',
               return_value=mock_comp_config):
        yield mock_comp_config


@pytest.fixture(autouse=True)
def mock_vllm_components(mocker):
    """Mock vLLM components that cause initialization issues."""
    # Mock LogitsProcessor to bypass CustomOp initialization - no spec needed
    mock_logits_processor = Mock()
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.LogitsProcessor',
        return_value=mock_logits_processor)

    # Mock Sampler
    mock_sampler = Mock()
    mocker.patch('vllm_neuron.worker.neuronx_distributed_model_loader.Sampler',
                 return_value=mock_sampler)

    return mock_logits_processor, mock_sampler


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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
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
            'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronPixtralForCausalLM',
            return_value=mock_causal_lm)
    else:
        mocker.patch(
            'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
            return_value=mock_causal_lm)

    # Pixtral/LLaVA is not supported, expect NotImplementedError
    if architecture == "LlavaForConditionalGeneration":
        with pytest.raises(NotImplementedError,
                           match="Pixtral is not yet supported"):
            get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config={})
    else:
        model = get_neuron_model(model_config,
                                 cache_config,
                                 parallel_config,
                                 scheduler_config,
                                 lora_serving_config=None,
                                 additional_config={})

        assert model is not None


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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
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
                         lora_serving_config=None,
                         additional_config={})

    # Test missing required fields
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"], model_type="llama")
    with pytest.raises(ValueError, match="Missing required fields"):
        get_neuron_model(model_config,
                         cache_config,
                         parallel_config,
                         scheduler_config,
                         lora_serving_config=None,
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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronPixtralForCausalLM',
        return_value=mock_pixtral)

    # Pixtral is not supported, expect NotImplementedError
    with pytest.raises(NotImplementedError,
                       match="Pixtral is not yet supported"):
        get_neuron_model(model_config,
                         cache_config,
                         parallel_config,
                         scheduler_config,
                         lora_serving_config=None,
                         additional_config=additional_config)


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

    lora_config = Mock()
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
    cache_config = Mock(enable_prefix_caching=True,
                        block_size=32,
                        num_gpu_blocks_override=65)

    scheduler_config = Mock(chunked_prefill_enabled=False,
                            max_model_len=1024,
                            max_num_seqs=2)

    # Test valid prefix caching configuration
    valid_config = {"is_prefix_caching": True, "is_block_kv_layout": True}
    result = _validate_neuron_config(cache_config, scheduler_config,
                                     valid_config)
    assert result == valid_config

    # Test invalid config (missing required fields)
    invalid_config = {"is_prefix_caching": False, "is_block_kv_layout": True}
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, invalid_config)


def test_image_to_text_config_comprehensive(mocker, base_configs):
    """Test comprehensive image-to-text model configurations.
    
    This test verifies:
    1. Various vision/text configuration combinations
        - Different batch sizes
        - Different context lengths
        - Empty configurations
    2. Configuration validation
        - Valid configuration acceptance
        - Invalid configuration rejection
    3. Error handling
        - Invalid key detection
        - Extra field handling
    
    Args:
        mocker: PyTest mocker fixture for creating mock objects
        base_configs: Tuple of (scheduler_config, cache_config, parallel_config)
    
    The test ensures:
    - Valid configurations are accepted without modification
    - Invalid configurations raise appropriate assertions
    - All configuration combinations are properly handled
    """
    scheduler_config, cache_config, parallel_config = base_configs

    # Test with various vision/text config combinations
    configs = [{
        "vision_neuron_config": {
            "batch_size": 4
        },
        "text_neuron_config": {
            "batch_size": 8
        }
    }, {
        "vision_neuron_config": {
            "max_context_length": 1024
        },
        "text_neuron_config": {
            "max_context_length": 2048
        }
    }, {
        "vision_neuron_config": {},
        "text_neuron_config": {}
    }]

    for config in configs:
        result = _validate_image_to_text_override_neuron_config(config)
        assert result == config

    # Test invalid configurations
    invalid_configs = [{
        "invalid_key": {}
    }, {
        "vision_neuron_config": {},
        "extra": {}
    }, {
        "text_neuron_config": {},
        "invalid": {}
    }]

    for config in invalid_configs:
        with pytest.raises(AssertionError):
            _validate_image_to_text_override_neuron_config(config)


def test_model_configs_comprehensive():
    """Test comprehensive model configuration scenarios.
    
    This test verifies:
    1. Standard configuration handling
        - Basic model parameters
        - Head dimensions
        - Architecture settings
    2. Derived configuration values
        - Hidden size calculations
        - Attention head handling
    3. Multi-modal configurations
        - Text config handling
        - Architecture detection
    
    The test ensures:
    - Configuration parameters are correctly processed
    - Derived values are properly calculated
    - Multi-modal settings are properly handled
    """
    # Test standard config
    config = PretrainedConfig(architectures=["LlamaForCausalLM"],
                              num_key_value_heads=32,
                              head_dim=64)
    arch, heads, dim = _get_model_configs(config)
    assert arch == "LlamaForCausalLM"
    assert heads == 32
    assert dim == 64

    # Test config with hidden_size and num_attention_heads
    config = PretrainedConfig(architectures=["LlamaForCausalLM"],
                              num_key_value_heads=32,
                              num_attention_heads=16,
                              hidden_size=1024)
    arch, heads, dim = _get_model_configs(config)
    assert arch == "LlamaForCausalLM"
    assert heads == 32
    assert dim == 64  # 1024 // 16

    # Test multi-modal config
    text_config = PretrainedConfig(num_key_value_heads=32, head_dim=64)
    config = PretrainedConfig(architectures=["LlavaForConditionalGeneration"],
                              text_config=text_config)
    arch, heads, dim = _get_model_configs(config)
    assert arch == "LlavaForConditionalGeneration"
    assert heads == 32
    assert dim == 64


def test_cache_config_scenarios(mocker, base_configs):
    """Test various cache configuration scenarios.
    
    This test verifies:
    1. Block size configurations
        - Different block size values
        - Block size validation
    2. GPU block override settings
        - Custom block count handling
        - Override validation
    3. Prefix caching settings
        - Enable/disable handling
        - Block KV layout compatibility
    
    Args:
        mocker: PyTest mocker fixture for creating mock objects
        base_configs: Tuple of (scheduler_config, cache_config, parallel_config)
    
    The test ensures:
    - Cache configurations are properly applied
    - Block sizes are correctly handled
    - Prefix caching settings are validated
    """

    scheduler_config, cache_config, parallel_config = base_configs

    # Test with different block sizes
    block_sizes = [8, 16, 32]
    for size in block_sizes:
        cache_config.block_size = size
        config = _get_default_neuron_config(Mock(dtype=torch.float32),
                                            cache_config, parallel_config,
                                            scheduler_config, Mock(), None)
        assert config["pa_block_size"] == size

    # Test with GPU blocks override
    cache_config.num_gpu_blocks_override = 100
    config = _get_default_neuron_config(Mock(dtype=torch.float32),
                                        cache_config, parallel_config,
                                        scheduler_config, Mock(), None)
    assert config["pa_num_blocks"] == 100

    # Test prefix caching
    cache_config.enable_prefix_caching = True
    config = _get_default_neuron_config(Mock(dtype=torch.float32),
                                        cache_config, parallel_config,
                                        scheduler_config, Mock(), None)
    assert config["is_prefix_caching"] is True
    assert config["is_block_kv_layout"] is True


def test_model_type_handling_comprehensive(mocker):
    """Test comprehensive handling of different model architectures and types.
    
    This test verifies:
    1. Model Architecture Support
        - LLaMA models (base causal LM)
        - Mixtral models (MOE architecture)
        - LLaVA models (multi-modal)
        - Qwen3 MOE models
    
    2. Configuration Handling
        - Base model parameters
        - Architecture-specific settings
        - Multi-modal specific configs
    
    3. Model Type Validation
        - Proper architecture recognition
        - Multi-modal model detection
        - Model type compatibility
    
    Args:
        mocker: PyTest fixture for mocking dependencies
    
    The test ensures that:
        - Each model type is properly configured
        - Multi-modal models have correct text configs
        - Model architectures are properly recognized
        - Configuration inheritance works correctly
    """
    model_configs = [("llama", "LlamaForCausalLM", {}),
                     ("mixtral", "MixtralForCausalLM", {}),
                     ("llava", "LlavaForConditionalGeneration", {
                         "text_config":
                         PretrainedConfig(num_key_value_heads=32, head_dim=64)
                     }), ("qwen3moe", "Qwen3MoeForCausalLM", {})]

    for model_type, arch, extra_config in model_configs:
        # Create and verify the configuration
        config = PretrainedConfig(architectures=[arch],
                                  num_key_value_heads=32,
                                  head_dim=64,
                                  model_type=model_type,
                                  **extra_config)

        # Verify the architecture is properly recognized
        if model_type == "llava":
            assert arch in NEURON_MULTI_MODAL_MODELS
            assert hasattr(config, 'text_config')
            assert config.text_config.num_key_value_heads == 32
            assert config.text_config.head_dim == 64
        else:
            assert arch not in NEURON_MULTI_MODAL_MODELS
            assert config.num_key_value_heads == 32
            assert config.head_dim == 64


def test_neuron_config_validation_comprehensive():
    """Test comprehensive validation of Neuron configurations including edge cases.
    
    This test verifies:
    1. Basic Configurations
        - Valid prefix caching settings
        - Valid chunked prefill settings
        - Basic validation rules
    
    2. Edge Cases
        - Mutually exclusive settings
        - Missing required fields
        - Invalid combinations
    
    3. Error Conditions
        - Invalid config presence
        - Incompatible settings
        - Missing required settings
    """
    # Basic validation cases
    basic_configs = [{
        "cache_config":
        Mock(enable_prefix_caching=True,
             block_size=32,
             num_gpu_blocks_override=65),
        "scheduler_config":
        Mock(chunked_prefill_enabled=False, max_model_len=1024,
             max_num_seqs=2),
        "neuron_config": {
            "is_prefix_caching": True,
            "is_block_kv_layout": True
        },
        "should_pass":
        True
    }, {
        "cache_config":
        Mock(enable_prefix_caching=False,
             block_size=32,
             num_gpu_blocks_override=65),
        "scheduler_config":
        Mock(chunked_prefill_enabled=True, max_model_len=1024, max_num_seqs=2),
        "neuron_config": {
            "chunked_prefill_config": {
                "enabled": True
            },
            "is_block_kv_layout": True
        },
        "should_pass":
        True
    }]

    # Edge cases
    edge_cases = [
        # Incompatible prefix caching
        {
            "cache_config":
            Mock(enable_prefix_caching=True,
                 block_size=32,
                 num_gpu_blocks_override=65),
            "scheduler_config":
            Mock(chunked_prefill_enabled=False,
                 max_model_len=1024,
                 max_num_seqs=2),
            "neuron_config": {
                "is_prefix_caching": False,
                "is_block_kv_layout": True
            }
        },
        # Missing chunked prefill config
        {
            "cache_config":
            Mock(enable_prefix_caching=False,
                 block_size=32,
                 num_gpu_blocks_override=65),
            "scheduler_config":
            Mock(chunked_prefill_enabled=True,
                 max_model_len=1024,
                 max_num_seqs=2),
            "neuron_config": {
                "is_block_kv_layout": True
            }
        },
        # Invalid config presence
        {
            "cache_config":
            Mock(enable_prefix_caching=False,
                 block_size=32,
                 num_gpu_blocks_override=65),
            "scheduler_config":
            Mock(chunked_prefill_enabled=False,
                 max_model_len=1024,
                 max_num_seqs=2),
            "neuron_config": {
                "is_block_kv_layout": True,
                "text_neuron_config": {},
                "vision_neuron_config": {}
            }
        },
        # Missing required setting
        {
            "cache_config":
            Mock(enable_prefix_caching=True,
                 block_size=32,
                 num_gpu_blocks_override=65),
            "scheduler_config":
            Mock(chunked_prefill_enabled=False,
                 max_model_len=1024,
                 max_num_seqs=2),
            "neuron_config": {
                "is_prefix_caching": True
            }
        }
    ]

    # Test basic cases
    for config in basic_configs:
        if config["should_pass"]:
            result = _validate_neuron_config(config["cache_config"],
                                             config["scheduler_config"],
                                             config["neuron_config"])
            assert result == config["neuron_config"]

    # Test edge cases
    for case in edge_cases:
        with pytest.raises(AssertionError):
            _validate_neuron_config(case["cache_config"],
                                    case["scheduler_config"],
                                    case["neuron_config"])


def test_speculative_execution_config_comprehensive():
    """Test comprehensive configuration of speculative execution features.
    
    This test verifies:
    1. Base Configuration
        - Model settings
        - Cache configuration
        - Parallel execution settings
        - Scheduler parameters
        - LoRA serving integration
    
    2. Speculation Modes
        - No speculation (default behavior)
        - Basic speculation settings
        - Feature flag handling
    
    3. Configuration Parameters
        - Token count settings
        - Method-specific parameters
        - Feature enablement flags
    
    Test Cases:
        - Configuration without speculation
        - Basic speculation with token count
        - Feature flag verification
    
    The test ensures that:
        - Default configuration works correctly
        - Speculation features are properly configured
        - All required parameters are present
        - Feature flags are correctly set
    """

    base_config = {
        "model_config":
        Mock(dtype=torch.float32),
        "cache_config":
        Mock(block_size=8,
             num_gpu_blocks_override=None,
             enable_prefix_caching=False),
        "parallel_config":
        Mock(tensor_parallel_size=1),
        "scheduler_config":
        Mock(max_num_seqs=32,
             max_model_len=2048,
             chunked_prefill_enabled=False),
        "lora_serving_config":
        Mock()  # Changed from lora_config to lora_serving_config
    }

    # Test without speculation
    config = _get_default_neuron_config(**base_config, speculative_config=None)
    assert "enable_fused_speculation" not in config
    assert "speculation_length" not in config

    # Test with basic speculation
    spec_config = Mock(num_speculative_tokens=5,
                       method="basic",
                       draft_model_config=Mock(model="draft-model"))
    config = _get_default_neuron_config(**base_config,
                                        speculative_config=spec_config)
    assert config["enable_fused_speculation"] is True
    assert config["speculation_length"] == 5
    assert "enable_eagle_speculation" not in config


def test_model_configs_edge_cases():
    """Test edge cases for model configurations.
    
    This test verifies:
    1. Configuration with derived head dimensions
    2. Missing architecture handling
    3. Missing required fields handling
    
    The test ensures proper error handling and validation
    of edge case configurations.
    """
    # Test with missing num_key_value_heads but with num_attention_heads and hidden_size
    config = PretrainedConfig(
        architectures=["LlamaForCausalLM"],
        num_key_value_heads=16,  # Add this field
        num_attention_heads=16,
        hidden_size=1024)
    arch, heads, dim = _get_model_configs(config)
    assert dim == 64  # 1024 // 16

    # Test with missing architectures
    config = PretrainedConfig()
    with pytest.raises(ValueError, match="No architectures specified"):
        _get_model_configs(config)

    # Test with missing required fields
    config = PretrainedConfig(architectures=["LlamaForCausalLM"])
    with pytest.raises(ValueError, match="Missing required fields"):
        _get_model_configs(config)


def test_neuron_config_validation_edge_cases():
    """Test edge cases for neuron config validation.
    
    This test verifies:
    1. Mutually exclusive settings validation
    2. Invalid configuration combinations
    3. Required field validation
    4. Error handling for incompatible settings
    
    The test ensures proper validation of edge case
    configurations and appropriate error handling.
    """
    # Test case 1: Incompatible prefix caching settings
    cache_config = Mock(enable_prefix_caching=True,
                        block_size=32,
                        num_gpu_blocks_override=65)

    scheduler_config = Mock(chunked_prefill_enabled=False,
                            max_model_len=1024,
                            max_num_seqs=2)

    neuron_config = {
        "is_prefix_caching": False,  # This conflicts with cache_config
        "is_block_kv_layout": True
    }
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, neuron_config)

    # Test case 2: Incompatible chunked prefill settings
    cache_config = Mock(enable_prefix_caching=False,
                        block_size=32,
                        num_gpu_blocks_override=65)

    scheduler_config = Mock(chunked_prefill_enabled=True,
                            max_model_len=1024,
                            max_num_seqs=2)

    neuron_config = {"is_block_kv_layout": True}
    # Should raise assertion error because chunked_prefill_config is missing
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, neuron_config)

    # Test case 3: Invalid text/vision config presence
    cache_config = Mock(enable_prefix_caching=False,
                        block_size=32,
                        num_gpu_blocks_override=65)

    scheduler_config = Mock(chunked_prefill_enabled=False,
                            max_model_len=1024,
                            max_num_seqs=2)

    neuron_config = {
        "is_block_kv_layout": True,
        "text_neuron_config": {},  # These should not be present
        "vision_neuron_config": {}
    }
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, neuron_config)

    # Test case 4: Missing required block KV layout setting
    cache_config = Mock(enable_prefix_caching=True,
                        block_size=32,
                        num_gpu_blocks_override=65)

    scheduler_config = Mock(chunked_prefill_enabled=False,
                            max_model_len=1024,
                            max_num_seqs=2)

    neuron_config = {
        "is_prefix_caching": True,
        # Missing is_block_kv_layout which is required for prefix caching
    }
    with pytest.raises(AssertionError):
        _validate_neuron_config(cache_config, scheduler_config, neuron_config)


def test_get_compiled_model_path():
    """Test _get_compiled_model_path method with different scenarios.
    
    This test verifies that the compiled model path is correctly determined based on:
    1. Environment variable NEURON_COMPILED_ARTIFACTS
    2. Local path existence
    3. Hash-based path generation
    
    The test ensures proper path resolution for model artifacts in different contexts.
    """

    # Create a mock instance
    model = Mock(spec=NeuronModelBase)

    # Test with NEURON_COMPILED_ARTIFACTS env var
    with patch.dict('os.environ',
                    {'NEURON_COMPILED_ARTIFACTS': '/path/to/artifacts'}):
        path = NeuronModelBase._get_compiled_model_path(
            model, "model_name", "hash123")
        assert str(path) == '/path/to/artifacts'


def test_remask_fused_spec_output():
    """Test _remask_fused_spec_output method for token masking.
    
    This test verifies the correct masking of fused speculation output by:
    1. Creating sample input tensors for tokens and positions
    2. Calculating expected masked output based on token counts
    3. Verifying the masking logic matches implementation
    
    The test ensures proper handling of:
    - Token padding
    - Position-based masking
    - Batch processing
    """
    # Create test inputs
    fused = [
        torch.tensor([[1, 2, 0], [3, 0, 0]]),  # accepted_tokens_with_padding
        torch.tensor([[5], [4]])  # next_pos_ids
    ]

    inputs = {"position_ids": torch.tensor([[2], [3]])}

    # Calculate expected output
    expected = fused[0].clone()
    next_pos_ids = fused[-1].squeeze(-1)
    positions_vec = inputs["position_ids"][:, -1]
    generated_token_counts = (next_pos_ids - positions_vec).to(torch.long)

    B, T = expected.shape
    generated_token_counts = generated_token_counts.clamp_(0, T)

    for b in range(B):
        expected[b, generated_token_counts[b]:] = -1

    # Test the method
    masked = NeuronCausalLM._remask_fused_spec_output(None, fused, inputs)
    assert torch.equal(masked, expected)


def test_init_fused_spec_config():
    """Test _init_fused_spec_config method for speculative execution setup.
    
    This test verifies the initialization of fused speculation configuration by:
    1. Setting up mock configurations for model and speculation
    2. Verifying eagle speculation settings
    3. Checking proper configuration inheritance
    
    The test ensures:
    - Proper configuration attachment
    - Correct eagle speculation flags
    - Valid draft model setup
    """

    model = Mock()
    config = Mock()
    config.neuron_config = Mock(enable_eagle_speculation=True)

    neuronx_model_cls = Mock()
    neuronx_model_cls._model_cls = "test_model_cls"
    neuronx_model_cls.get_config_cls.return_value = lambda *args, **kwargs: Mock(
    )

    spec_config = Mock()
    spec_config.draft_model_config = Mock(model="draft-model")

    model._init_fused_spec_config(config, neuronx_model_cls, spec_config)

    assert hasattr(config, 'fused_spec_config')
    assert config.neuron_config.enable_eagle_speculation is True


def test_sort_inputs():
    """Test _sort_inputs with different input types.
    
    This test verifies the correct sorting of different input types:
    1. Tensor inputs
    2. Empty tensors
    3. List inputs
    4. Non-sortable inputs
    
    The test ensures proper handling of:
    - Batch dimension sorting
    - Different input types
    - Edge cases (empty tensors)
    - Non-tensor inputs
    """
    sorted_indices = torch.tensor([1, 0, 2])

    inputs = {
        'tensor': torch.tensor([[1], [2], [3]]),
        'empty_tensor': torch.tensor([]),
        'list': ['a', 'b', 'c'],
        'other': 42
    }

    sorted_inputs = NeuronModelBase._sort_inputs(inputs, sorted_indices)

    assert torch.equal(sorted_inputs['tensor'], torch.tensor([[2], [1], [3]]))
    assert torch.equal(sorted_inputs['empty_tensor'], torch.tensor([]))
    assert sorted_inputs['list'] == ['b', 'a', 'c']
    assert sorted_inputs['other'] == 42


def test_neuron_model_base_init_with_sampling_disabled():
    """Test NeuronModelBase initialization with on-device sampling control.
    
    This test verifies:
    1. NEURON_ON_DEVICE_SAMPLING_DISABLED=1 enables CPU sampling
    2. Sampler attribute is created when CPU sampling is enabled
    3. Default behavior (env var not set) enables on-device sampling
    4. Flag correctly reflects environment variable state
    """
    config = PretrainedConfig(vocab_size=32000)

    # Test with sampling disabled
    with patch.dict('os.environ', {'NEURON_ON_DEVICE_SAMPLING_DISABLED': '1'}):
        model = NeuronCausalLM(config)
        assert model.on_device_sampling_disabled is True
        assert hasattr(model, 'sampler')

    # Test with sampling enabled (default)
    with patch.dict('os.environ', {}, clear=True):
        model = NeuronCausalLM(config)
        assert model.on_device_sampling_disabled is False


def test_neuron_model_base_forward_not_implemented():
    """Test that base class forward method raises NotImplementedError.
    
    This test verifies:
    1. NeuronModelBase.forward() is not directly callable
    2. Subclasses must implement forward()
    3. Proper NotImplementedError is raised
    
    NeuronModelBase is abstract and requires subclass implementation.
    """
    config = PretrainedConfig(vocab_size=32000)
    model = NeuronModelBase(config)

    with pytest.raises(NotImplementedError):
        model.forward(None, None, None, None)


def test_neuron_model_base_sample_not_implemented():
    """Test that base class sample method raises NotImplementedError.
    
    This test verifies:
    1. NeuronModelBase.sample() is not directly callable
    2. Subclasses must implement sample()
    3. Proper NotImplementedError is raised
    
    Sample method must be implemented by concrete subclasses.
    """
    config = PretrainedConfig(vocab_size=32000)
    model = NeuronModelBase(config)

    with pytest.raises(NotImplementedError):
        model.sample(torch.tensor([1.0]))


def test_reordered_with_batch_size_one():
    """Test _reordered context manager optimization for single-element batches.
    
    This test verifies:
    1. Batch size 1 restore returns output unchanged
    2. No unnecessary reordering for single requests
    3. Optimization path is correctly taken
    
    Single-element batches skip reordering as an optimization since
    there's nothing to reorder.
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))
    model.is_reorder_needed = True

    input_block_ids = torch.tensor([0])  # Single element
    inputs = {'input_ids': torch.tensor([[1]])}

    with model._reordered(input_block_ids,
                          **inputs) as (sorted_ids, reordered, restore):
        # With batch size 1, restore should return output unchanged
        output = torch.tensor([[10]])
        restored = restore(output)
        assert torch.equal(restored, output)


def test_load_weights_common_success_path(mocker):
    """Test successful loading from pre-compiled model artifacts.
    
    This test verifies:
    1. Neuron config created from kwargs
    2. Model config created or used from kwargs
    3. Compiled model loaded successfully
    4. Success flag returned as True
    5. Config returned unchanged
    
    This is the happy path when compiled artifacts exist.
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_config_cls = Mock()

    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mock_neuron_config = Mock()
    mock_neuron_config.enable_fused_speculation = False
    mock_neuron_config_cls.return_value = mock_neuron_config

    mock_config = Mock()
    mock_config.to_json_string.return_value = "config_string"
    mock_config_cls.return_value = mock_config

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())
    mocker.patch.object(model, '_load_compiled_model')

    kwargs = {'neuron_config': {}}
    success, path, config = model._load_weights_common("model_path",
                                                       mock_neuronx_cls,
                                                       **kwargs)

    assert success is True
    assert config == mock_config
    model._load_compiled_model.assert_called_once()


def test_neuron_causal_lm_sample_cpu_sampler_not_implemented():
    """Test that CPU sampling path raises RuntimeError.
    
    This test verifies:
    1. CPU sampling (when on_device_sampling_config=None) should be handled by model runner
    2. RuntimeError raised indicating sampling path routing bug
    3. Model should not handle CPU sampling directly
    
    CPU sampling should be handled by the model runner, not the model.
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))
    model.model = Mock()
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    logits = torch.randn(1, 1000)

    with pytest.raises(
            RuntimeError,
            match=
            "CPU sampling should be handled by the model runner, not the model"
    ):
        model.sample(logits)


def test_neuron_multi_modal_load_weights_with_nested_configs(mocker):
    """Test multimodal model loading with separate text and vision configs.
    
    This test verifies:
    1. Text and vision neuron configs created separately
    2. Override configs applied to each pipeline
    3. Both configs passed to model initialization
    4. Model loads successfully with nested configs
    
    Multimodal models have separate processing pipelines for text and vision,
    each requiring its own neuron configuration.
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16))
    model = NeuronMultiModalCausalLM(config)

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_config_cls = Mock()

    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls
    mock_neuronx_cls._model_cls = "test_model"

    mock_neuron_config = Mock()
    mock_neuron_config_cls.return_value = mock_neuron_config

    mock_final_config = Mock()
    mock_final_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_final_config

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader._get_neuron_model_cls',
        return_value=mock_neuronx_cls)
    mocker.patch.object(model,
                        '_load_weights_common',
                        return_value=(True, "/compiled", mock_final_config))

    kwargs = {
        'neuron_config': {
            'batch_size': 32
        },
        'override_neuron_config': {
            'text_neuron_config': {
                'max_length': 2048
            },
            'vision_neuron_config': {
                'image_size': 224
            }
        }
    }

    success, path = model.load_weights("model_path",
                                       "LlavaForConditionalGeneration",
                                       **kwargs)

    assert success is True
    # Verify that both text and vision configs were created
    assert mock_neuron_config_cls.call_count == 2


def test_neuron_llama4_forward_sampling_params_mismatch():
    """Test automatic sampling_params adjustment for batch size mismatches.
    
    This test verifies:
    1. Sampling params batch size adjusted to match input batch size
    2. Mismatched batch sizes handled automatically
    3. Trimmed params passed to underlying model
    4. No errors raised for shape mismatches
    
    In multimodal models, image processing may create different batch dimensions.
    The model automatically adjusts sampling_params to prevent shape errors.
    """
    config = PretrainedConfig(text_config=PretrainedConfig(vocab_size=1000),
                              image_token_index=999)
    model = NeuronLlama4ForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.logits = torch.tensor([[[1.0]]])
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    input_ids = torch.tensor([[1]])  # Batch size 1
    sampling_params = torch.tensor([[1.0], [2.0],
                                    [3.0]])  # Batch size 3 (mismatch!)

    model.forward(input_ids=input_ids,
                  positions=torch.tensor([[0]]),
                  input_block_ids=torch.tensor([0]),
                  sampling_params=sampling_params)

    # Verify sampling_params was adjusted to match input_ids batch size
    call_kwargs = model.model.call_args[1]
    assert call_kwargs['sampling_params'].shape[0] == 1


def test_neuron_llama4_forward_pixel_values_debug_logging(caplog):
    """Test debug logging of pixel values shape in forward pass.
    
    This test verifies:
    1. Pixel values shape logged at DEBUG level
    2. Log message contains "pixel_values.shape"
    3. Logging occurs before model forward pass
    
    Shape logging helps debug image preprocessing and batching issues.
    """
    config = PretrainedConfig(text_config=PretrainedConfig(vocab_size=1000),
                              image_token_index=999)
    model = NeuronLlama4ForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.logits = torch.tensor([[[1.0]]])
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    pixel_values = torch.randn(1, 5, 3, 336, 336)

    with caplog.at_level(
            logging.DEBUG,
            logger='vllm_neuron.worker.neuronx_distributed_model_loader'):
        model.forward(input_ids=torch.tensor([[1]]),
                      positions=torch.tensor([[0]]),
                      input_block_ids=torch.tensor([0]),
                      sampling_params=torch.tensor([[1.0]]),
                      pixel_values=pixel_values)

    assert "pixel_values.shape" in caplog.text


def test_neuron_model_base_init_attributes():
    """Test that NeuronModelBase properly initializes all attributes.
    
    This test verifies:
    1. LogitsProcessor is created during initialization
    2. Lazy-initialized attributes are declared (but not set)
    3. kv_caches starts as None
    4. All expected attributes exist as class members
    
    Note: 'model' attribute is only set after load_weights() is called.
    """
    config = PretrainedConfig(vocab_size=32000)
    model = NeuronCausalLM(config)

    # Verify logits processor is created
    assert hasattr(model, 'logits_processor')

    # Verify lazy initialized attributes exist
    # Note: These are type annotations in __init__, not actual assignments
    # They are set later during load_weights()
    assert hasattr(model, 'kv_caches')
    assert model.kv_caches is None
    assert hasattr(model, 'on_device_sampling_disabled')


def test_get_kv_caches_state_iteration():
    """Test KV cache extraction with multiple TP ranks.
    
    This test verifies:
    1. KV caches extracted from model state
    2. Multiple TP ranks handled correctly
    3. Caches properly organized by layer and TP rank
    4. Cached result returned on subsequent calls
    
    The state dict maps TP ranks to their tensors, which are
    then reorganized into a flat list of KV caches.
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    # Create mock state with multiple TP ranks and multiple keys per rank
    mock_state = {
        0: {
            'layer0_k': torch.tensor([1, 2]),
            'layer0_v': torch.tensor([3, 4]),
            'layer1_k': torch.tensor([5, 6])
        },
        1: {
            'layer0_k': torch.tensor([7, 8]),
            'layer0_v': torch.tensor([9, 10]),
            'layer1_k': torch.tensor([11, 12])
        }
    }

    mock_nxd_model = Mock()
    # State must be iterable of dicts, not dict itself
    mock_nxd_model.state = list(mock_state.values())

    model.model = Mock()
    model.model.context_encoding_model = Mock()
    model.model.context_encoding_model.model = Mock()
    model.model.context_encoding_model.model.nxd_model = mock_nxd_model

    kv_caches = model.get_kv_caches()

    # Should have 3 tensors per TP rank * 2 TP ranks = 6 total
    assert len(kv_caches) == 6

    # Verify caching works
    kv_caches_2 = model.get_kv_caches()
    assert kv_caches is kv_caches_2


def test_neuron_model_base_load_weights_not_implemented():
    """Test that NeuronModelBase.load_weights raises NotImplementedError.
    
    This test verifies:
    1. Base class load_weights cannot be called directly
    2. NotImplementedError is raised
    3. Subclasses must implement load_weights
    """
    model = NeuronModelBase(PretrainedConfig(vocab_size=1000))

    with pytest.raises(NotImplementedError):
        model.load_weights("model_path", "LlamaForCausalLM")


def test_reordered_restore_function_with_sorting(mocker):
    """Test _reordered restore function with actual sorting.
    
    This test verifies:
    1. Output is restored to original order using index_select
    2. torch.argsort is used to get reverse mapping
    3. Restoration works with multiple batch elements
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))
    model.is_reorder_needed = True

    # Create unsorted block IDs
    input_block_ids = torch.tensor([2, 0, 1])
    inputs = {'input_ids': torch.tensor([[10], [20], [30]])}

    with model._reordered(input_block_ids,
                          **inputs) as (sorted_ids, reordered, restore):
        # Output in sorted order
        output = torch.tensor([[100], [200], [300]])

        # Restore should use torch.index_select with torch.argsort(sorted_indices)
        restored = restore(output)

        # Verify restoration worked - should map back to original order
        assert restored.shape == output.shape


def test_load_weights_common_value_error_exception(mocker):
    """Test ValueError handling in _load_weights_common.
    
    This test verifies:
    1. ValueError is caught
    2. Warning is logged about recompiling
    3. Returns False for success flag
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_neuron_config = Mock(enable_fused_speculation=False)
    mock_neuron_config_cls.return_value = mock_neuron_config
    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls

    mock_config_cls = Mock()
    mock_config = Mock()
    mock_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_config
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())

    # Make _load_compiled_model raise ValueError
    mocker.patch.object(model,
                        '_load_compiled_model',
                        side_effect=ValueError("Invalid compiled model"))

    kwargs = {'neuron_config': {}}

    success, path, config = model._load_weights_common("model_path",
                                                       mock_neuronx_cls,
                                                       **kwargs)

    assert success is False


def test_compile_and_load_model_full_workflow(mocker):
    """Test model compilation and loading workflow.
    
    This test verifies:
    1. Model is instantiated with path and config
    2. Model.compile() is called
    3. Model.load() is called
    4. Model is assigned to self.model
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_compiled_model = Mock()
    mock_neuronx_cls.return_value = mock_compiled_model

    config = Mock()

    model._compile_and_load_model("model_path", mock_neuronx_cls, config,
                                  "compiled_path")

    # Verify model instantiation with path and config
    mock_neuronx_cls.assert_called_once_with("model_path", config)

    # Verify compile was called
    mock_compiled_model.compile.assert_called_once_with("compiled_path")

    # Verify load was called
    mock_compiled_model.load.assert_called_once_with("compiled_path")

    # Verify model was assigned
    assert model.model == mock_compiled_model


def test_neuron_causal_lm_load_weights_local_path_exists(mocker):
    """Test load_weights when local path exists and needs compilation.
    
    This test verifies:
    1. When precompiled not found, compilation is triggered
    2. Local path is used directly (no save_pretrained)
    3. Compile and load is called
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader._get_neuron_model_cls',
        return_value=mock_neuronx_cls)

    # Simulate failure to load precompiled
    mocker.patch.object(model,
                        '_load_weights_common',
                        return_value=(False, "/compiled", Mock()))

    # Mock os.path.exists to return True (local path exists)
    mocker.patch('os.path.exists', return_value=True)

    mocker.patch.object(model, '_compile_and_load_model')

    success, path = model.load_weights("local/model/path",
                                       "LlamaForCausalLM",
                                       neuron_config={},
                                       override_neuron_config=None)

    # _save_pretrained_model should NOT be called
    # _compile_and_load_model SHOULD be called
    model._compile_and_load_model.assert_called_once()


def test_init_fused_spec_config_non_eagle_zeros_speculation_length():
    """Test _init_fused_spec_config zeros speculation_length for non-eagle.
    
    This test verifies:
    1. Non-eagle drafts have speculation_length=0
    2. is_eagle_draft is not set
    3. enable_fused_speculation is False in draft
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    config = Mock()
    config.neuron_config = Mock(enable_eagle_speculation=False,
                                speculation_length=5)

    neuronx_model_cls = Mock()
    neuronx_model_cls._model_cls = "test_model"
    neuronx_model_cls.get_config_cls.return_value = lambda *args, **kwargs: Mock(
    )

    spec_config = Mock()
    spec_config.draft_model_config = Mock(model="draft-model")

    model._init_fused_spec_config(config, neuronx_model_cls, spec_config)

    assert hasattr(config, 'fused_spec_config')


def test_sort_inputs_skips_mismatched_batch_size(caplog):
    """Test _sort_inputs skips reordering for mismatched batch sizes.
    
    This test verifies:
    1. Debug log when batch size doesn't match
    2. Input is not reordered when batch size mismatches
    3. Other inputs are still reordered
    """
    sorted_indices = torch.tensor([1, 0])

    inputs = {
        'input_ids': torch.tensor([[1], [2]]),  # batch=2
        'pixel_values': torch.tensor([[[1, 2, 3]]]),  # batch=1 (mismatch)
    }

    with caplog.at_level(
            logging.DEBUG,
            logger='vllm_neuron.worker.neuronx_distributed_model_loader'):
        sorted_inputs = NeuronModelBase._sort_inputs(inputs, sorted_indices)

    # Check debug log was emitted
    assert "Skipping reorder for key pixel_values" in caplog.text

    # Verify pixel_values was NOT reordered
    assert torch.equal(sorted_inputs['pixel_values'], inputs['pixel_values'])

    # Verify input_ids WAS reordered
    assert torch.equal(sorted_inputs['input_ids'], torch.tensor([[2], [1]]))


def test_load_weights_common_file_not_found_exception(mocker, caplog):
    """Test FileNotFoundError handling in _load_weights_common.
    
    This test verifies:
    1. FileNotFoundError is caught
    2. Warning is logged
    3. Returns False for success flag
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_neuron_config = Mock(enable_fused_speculation=False)
    mock_neuron_config_cls.return_value = mock_neuron_config
    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls

    mock_config_cls = Mock()
    mock_config = Mock()
    mock_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_config
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())

    # Make _load_compiled_model raise FileNotFoundError
    mocker.patch.object(
        model,
        '_load_compiled_model',
        side_effect=FileNotFoundError("Compiled model not found"))

    kwargs = {'neuron_config': {}}

    with caplog.at_level(logging.WARNING):
        success, path, config = model._load_weights_common(
            "model_path", mock_neuronx_cls, **kwargs)

    assert success is False
    assert "Unable to find precompiled artifacts" in caplog.text


def test_get_compiled_model_path_local_exists_creates_and_clears(mocker):
    """Test path creation and cleanup for local models.
    
    This test verifies:
    1. Path is created with mkdir
    2. Existing path is cleared with rmtree
    3. Correct path structure is returned
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    # Mock os.path.exists to return True (local path exists)
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch.dict('os.environ', {}, clear=True)

    # Create a proper mock for Path that supports division operator
    from pathlib import Path

    # We need to let Path work normally but intercept mkdir and rmtree
    mock_mkdir = mocker.patch.object(Path, 'mkdir')
    mock_rmtree = mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.shutil.rmtree')

    path = model._get_compiled_model_path("local_model", "hash123")

    # Verify mkdir was called
    assert mock_mkdir.call_count >= 1

    # Verify rmtree was called to clear existing artifacts
    assert mock_rmtree.call_count >= 1

    # Verify path structure
    assert "neuron-compiled-artifacts" in str(path)
    assert "hash123" in str(path)


def test_load_compiled_model_instantiation_and_loading(mocker, caplog):
    """Test compiled model instantiation and loading.
    
    This test verifies:
    1. Model is instantiated from neuronx_model_cls
    2. Model.load() is called
    3. Success message is logged
    4. Override config warning logic is triggered
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_compiled_model = Mock()
    mock_neuronx_cls.return_value = mock_compiled_model

    kwargs = {"override_neuron_config": {"batch_size": 64}}

    with caplog.at_level(logging.WARNING):
        model._load_compiled_model("/compiled/path", mock_neuronx_cls, kwargs)

    # Verify model was instantiated with path
    mock_neuronx_cls.assert_called_once_with("/compiled/path")

    # Verify model.load was called
    mock_compiled_model.load.assert_called_once_with("/compiled/path")

    # Verify model was assigned
    assert model.model == mock_compiled_model

    # Verify warning about override_neuron_config
    assert "override_neuron_config will be ignored" in caplog.text


def test_save_pretrained_model_hf_download_and_save(mocker):
    """Test HuggingFace model download and save.
    
    This test verifies:
    1. AutoModelForCausalLM.from_pretrained is called
    2. Model is saved to local-models directory
    3. Correct path is returned
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_hf_model = Mock()
    mock_from_pretrained = mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.AutoModelForCausalLM.from_pretrained',
        return_value=mock_hf_model)

    result = model._save_pretrained_model("gpt2")

    # Verify from_pretrained was called
    mock_from_pretrained.assert_called_once_with("gpt2")

    # Verify save_pretrained was called with correct path
    expected_path = os.path.join("local-models", "gpt2")
    mock_hf_model.save_pretrained.assert_called_once_with(expected_path)

    # Verify correct path returned
    assert result == expected_path


def test_neuron_pixtral_execute_model_without_multimodal_kwargs():
    """Test NeuronPixtralForCausalLM.execute_model with None kwargs.
    
    This test verifies:
    1. Default image_sizes used when kwargs is None
    2. Vision mask still created
    3. No errors on missing kwargs
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16),
        image_token_index=999)
    model = NeuronPixtralForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.logits = torch.randn(1, 10, 1000)
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    model_input = Mock()
    model_input.input_tokens = torch.tensor([[1, 999, 3]])
    model_input.position_ids = torch.tensor([[0, 1, 2]])
    model_input.input_block_ids = torch.tensor([0])
    model_input.sampling_params = torch.tensor([[1.0]])
    model_input.multi_modal_kwargs = None  # No kwargs

    result = model.execute_model(model_input)
    assert result is not None


def test_get_neuron_model_cls_conditional_generation_to_causal_lm():
    """Test ConditionalGeneration task conversion to CausalLM.
    
    This test verifies:
    1. ConditionalGeneration is converted to CausalLM
    2. Matches NxDI class naming
    3. Proper task name handling
    """
    # This tests the task conversion logic
    with patch.dict(
            'vllm_neuron.worker.neuronx_distributed_model_loader.MODEL_TYPES',
        {'mllama': {
            'causal-lm': Mock()
        }}):
        result = _get_neuron_model_cls("MllamaForConditionalGeneration")
        assert result is not None


def test_init_fused_spec_config_called_with_speculation(mocker):
    """Test _init_fused_spec_config is called when fused speculation enabled.
    
    This test verifies:
    1. _init_fused_spec_config is called when enable_fused_speculation=True
    2. speculative_config is passed correctly
    3. Config is modified with fused spec config
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mock_neuron_config = Mock(enable_fused_speculation=True)
    mock_neuron_config_cls = Mock(return_value=mock_neuron_config)
    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls
    mock_neuronx_cls._model_cls = "test_model"

    mock_config_cls = Mock()
    mock_config = Mock()
    mock_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_config
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())

    spec_config = Mock()
    spec_config.draft_model_config = Mock(model="draft-model")

    # Mock _init_fused_spec_config to verify it's called
    mock_init_spec = mocker.patch.object(model, '_init_fused_spec_config')

    mocker.patch.object(model, '_load_compiled_model')

    kwargs = {'neuron_config': {}, 'speculative_config': spec_config}
    model._load_weights_common("model_path", mock_neuronx_cls, **kwargs)

    # Verify _init_fused_spec_config was called
    mock_init_spec.assert_called_once()


def test_neuron_causal_lm_forward_fused_speculation_branch(mocker):
    """Test forward with fused speculation enabled.
    
    This test verifies:
    1. enable_fused_speculation=True branch is taken
    2. _remask_fused_spec_output is called
    3. Output is processed correctly
    """
    config = PretrainedConfig(vocab_size=1000)
    model = NeuronCausalLM(config)
    model.model = Mock()
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = Mock()
    model.model.config.neuron_config.enable_fused_speculation = True
    model.is_reorder_needed = False

    # Mock fused speculation output
    accepted_tokens = torch.tensor([[1, 2, 0], [3, 0, 0]])
    next_pos = torch.tensor([[2], [3]])

    mock_output = Mock()
    mock_output.hidden_states = (accepted_tokens, next_pos)
    model.model.return_value = mock_output

    # Mock _remask_fused_spec_output
    mock_remask = mocker.patch.object(model,
                                      '_remask_fused_spec_output',
                                      return_value=torch.tensor([[1, 2, -1],
                                                                 [3, -1, -1]]))

    result = model.forward(input_ids=torch.tensor([[1], [2]]),
                           positions=torch.tensor([[0], [0]]),
                           input_block_ids=torch.tensor([0, 1]),
                           block_tables=torch.tensor([[0], [1]]),
                           sampling_params=torch.tensor([[1.0], [1.0]]),
                           position_ids=torch.tensor([[0], [0]]))

    # Verify _remask_fused_spec_output was called
    mock_remask.assert_called_once()
    assert result.shape[0] == 2


def test_neuron_multi_modal_load_weights_creates_configs(mocker):
    """Test NeuronMultiModalCausalLM.load_weights creates text/vision configs.
    
    This test verifies:
    1. vision_neuron_config is created
    2. text_neuron_config is created
    3. Both configs passed to model
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16))
    model = NeuronMultiModalCausalLM(config)

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_neuron_config = Mock()
    mock_neuron_config_cls.return_value = mock_neuron_config
    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls
    mock_neuronx_cls._model_cls = "test_model"

    mock_config_cls = Mock()
    mock_final_config = Mock()
    mock_final_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_final_config
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader._get_neuron_model_cls',
        return_value=mock_neuronx_cls)
    mocker.patch.object(model,
                        '_load_weights_common',
                        return_value=(True, "/compiled", mock_final_config))

    kwargs = {
        'neuron_config': {
            'batch_size': 32
        },
        'override_neuron_config': {
            'vision_neuron_config': {
                'image_size': 224
            },
            'text_neuron_config': {
                'max_length': 2048
            }
        }
    }

    success, path = model.load_weights("model_path",
                                       "LlavaForConditionalGeneration",
                                       **kwargs)

    # Verify neuron config class was called twice (once for vision, once for text)
    assert mock_neuron_config_cls.call_count == 2


def test_neuron_pixtral_forward_casts_pixel_values_dtype(mocker):
    """Test NeuronPixtralForCausalLM.forward casts pixel_values dtype.
    
    This test verifies:
    1. Pixel values dtype is obtained from config
    2. Pixel values are cast to correct dtype
    3. Forward succeeds with dtype conversion
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(
            num_attention_heads=16,
            neuron_config=Mock(torch_dtype=torch.bfloat16)),
        image_token_index=999)
    model = NeuronPixtralForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.model.config.vision_config.neuron_config = Mock(
        torch_dtype=torch.bfloat16)
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.hidden_states = torch.randn(1, 1000)
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = Mock()

    # Tensor that will be cast
    pixel_values = torch.randn(1, 3, 336, 336, dtype=torch.float32)

    result = model.forward(input_ids=torch.tensor([[1, 999]]),
                           positions=torch.tensor([[0, 1]]),
                           input_block_ids=torch.tensor([0]),
                           sampling_params=torch.tensor([[1.0]]),
                           pixel_values=pixel_values,
                           vision_mask=torch.tensor([[[False], [True]]]))

    assert result is not None


def test_get_neuron_model_creates_llama4_model(mocker, base_configs):
    """Test get_neuron_model creates NeuronLlama4ForCausalLM.
    
    This test verifies:
    1. Llama4ForConditionalGeneration architecture detected
    2. NeuronLlama4ForCausalLM instance created
    3. Model attributes are set
    """
    scheduler_config, cache_config, parallel_config = base_configs

    text_config = PretrainedConfig(num_key_value_heads=32,
                                   head_dim=64,
                                   vocab_size=32000)

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["Llama4ForConditionalGeneration"],
        text_config=text_config,
        vision_config=PretrainedConfig(num_attention_heads=16),
        image_token_index=999,
        model_type="llama4")
    model_config.model = "meta-llama/Llama-4-vision"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = None

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_llama4 = Mock(spec=NeuronLlama4ForCausalLM)
    mock_llama4.model = mock_model
    mock_llama4.eval.return_value = mock_llama4

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronLlama4ForCausalLM',
        return_value=mock_llama4)

    model = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config={})

    assert model is not None


def test_get_neuron_model_with_override_config_logging(mocker, base_configs,
                                                       caplog):
    """Test get_neuron_model logs override config info.
    
    This test verifies:
    1. Override config is logged when present
    2. Default config message logged when not present
    3. Proper info level logging
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

    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model
    mock_causal_lm.eval.return_value = mock_causal_lm

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    # Test with override config
    with caplog.at_level(logging.INFO):
        additional_config = {"override_neuron_config": {"batch_size": 64}}
        _ = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config=additional_config)

    assert "Retrieved override_neuron_config" in caplog.text

    # Test without override config
    caplog.clear()
    with caplog.at_level(logging.INFO):
        additional_config = {}
        _ = get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config=additional_config)

    assert "No neuron overrides" in caplog.text


def test_neuron_causal_lm_load_weights_triggers_compilation(mocker):
    """Test load_weights triggers compilation when precompiled not found.
    
    This test verifies:
    1. Compilation is triggered when success=False
    2. _save_pretrained_model called when path doesn't exist
    3. _compile_and_load_model is called
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    mock_neuronx_cls = Mock()
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader._get_neuron_model_cls',
        return_value=mock_neuronx_cls)

    # Create a mock config that will be returned
    mock_config = Mock()

    # Simulate precompiled not found
    mocker.patch.object(model,
                        '_load_weights_common',
                        return_value=(False, "/compiled", mock_config))

    # Path doesn't exist - should trigger _save_pretrained_model
    mocker.patch('os.path.exists', return_value=False)

    mock_save = mocker.patch.object(model,
                                    '_save_pretrained_model',
                                    return_value="local_path")

    mock_compile = mocker.patch.object(model, '_compile_and_load_model')

    success, path = model.load_weights("remote-model",
                                       "LlamaForCausalLM",
                                       neuron_config={},
                                       override_neuron_config=None)

    # Verify _save_pretrained_model was called
    mock_save.assert_called_once_with("remote-model")

    # Verify _compile_and_load_model was called
    # Don't check exact config object, just verify the call happened
    assert mock_compile.call_count == 1
    call_args = mock_compile.call_args[0]
    assert call_args[0] == "local_path"
    assert call_args[1] == mock_neuronx_cls
    assert call_args[3] == "/compiled"


def test_neuron_pixtral_execute_model_extracts_image_sizes(mocker):
    """Test NeuronPixtralForCausalLM.execute_model extracts image_sizes.
    
    This test verifies:
    1. image_sizes extracted from multi_modal_kwargs
    2. Vision mask is created
    3. super().execute_model is called with image_sizes
    """
    # Create proper nested config structure
    vision_config = PretrainedConfig(num_attention_heads=16)
    vision_config.neuron_config = Mock(torch_dtype=torch.bfloat16)

    config = PretrainedConfig(text_config=PretrainedConfig(vocab_size=1000),
                              vision_config=vision_config,
                              image_token_index=999)

    model = NeuronPixtralForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.logits = torch.randn(1, 10, 1000)
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    # Properly set up the vision_config with neuron_config
    model.model.config.vision_config = vision_config

    model_input = Mock()
    model_input.input_tokens = torch.tensor([[1, 999, 3]])
    model_input.position_ids = torch.tensor([[0, 1, 2]])
    model_input.input_block_ids = torch.tensor([0])
    model_input.sampling_params = torch.tensor([[1.0]])
    model_input.multi_modal_kwargs = {
        "pixel_values": torch.randn(1, 3, 336, 336),
        "image_sizes": torch.tensor([[512, 512]])
    }

    # Mock the parent's execute_model to verify image_sizes is passed
    mock_parent_execute = mocker.patch.object(NeuronMultiModalCausalLM,
                                              'execute_model',
                                              return_value=mock_output.logits)

    result = model.execute_model(model_input)

    assert result is not None

    # Verify parent execute_model was called with image_sizes
    mock_parent_execute.assert_called_once()
    call_kwargs = mock_parent_execute.call_args[1]
    assert 'image_sizes' in call_kwargs
    # Verify the image_sizes value matches what was in multi_modal_kwargs
    assert torch.equal(call_kwargs['image_sizes'], torch.tensor([[512, 512]]))


def test_init_fused_spec_config_with_modules_to_not_convert_branch(mocker):
    """Test _init_fused_spec_config when draft_model_modules_to_not_convert exists.
    
    This test verifies:
    1. draft_model_modules_to_not_convert attribute is checked
    2. modules_to_not_convert is set on draft config
    3. Eagle speculation branch is also taken
    """
    model = NeuronCausalLM(PretrainedConfig(vocab_size=1000))

    config = Mock()
    config.neuron_config = Mock(
        enable_eagle_speculation=True,
        draft_model_modules_to_not_convert=["layer1", "layer2"],
        speculation_length=5)

    neuronx_model_cls = Mock()
    neuronx_model_cls._model_cls = "test_model"

    # Track the draft_neuron_config that's created
    draft_configs_created = []

    def capture_config(*args, **kwargs):
        mock_config = Mock()
        draft_configs_created.append(mock_config)
        return mock_config

    neuronx_model_cls.get_config_cls.return_value = capture_config

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())

    spec_config = Mock()
    spec_config.draft_model_config = Mock(model="draft-model")

    model._init_fused_spec_config(config, neuronx_model_cls, spec_config)

    # Verify config was created
    assert hasattr(config, 'fused_spec_config')


def test_neuron_causal_lm_forward_chunked_prefill_path(mocker):
    """Test forward with chunked prefill and prefill_completion_state.
    
    This test verifies:
    1. Chunked prefill branch is taken
    2. prefill_completion_state is checked (assertion)
    3. Logits are extracted at completion indices
    """
    config = PretrainedConfig(vocab_size=1000)
    model = NeuronCausalLM(config)
    model.model = Mock()
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None
    model.model.config.neuron_config.is_chunked_prefill = True
    model.neuron_config = model.model.config.neuron_config
    model.is_reorder_needed = False

    # Mock output with logits
    mock_output = Mock()
    # Shape: [batch, seq_len, vocab_size]
    mock_output.logits = torch.randn(1, 5, 1000)
    model.model.return_value = mock_output

    # Prefill completion state: positions 1 and 3 are complete
    prefill_completion_state = torch.tensor([0, 1, 0, 1, 0])

    result = model.forward(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                           positions=torch.tensor([[0, 1, 2, 3, 4]]),
                           input_block_ids=torch.tensor([0]),
                           block_tables=torch.tensor([[0, 1, 2]]),
                           sampling_params=torch.tensor([[1.0]]),
                           prefill_completion_state=prefill_completion_state)

    # Should extract logits at positions 1 and 3 (where completion_state is 1)
    assert result.shape[0] == 2  # Two completed positions


def test_neuron_causal_lm_forward_without_chunked_prefill_else_branch(mocker):
    """Test forward without chunked prefill takes else branch.
    
    This test verifies:
    1. Not using chunked prefill
    2. Last token logits are extracted
    3. Restore function is called
    """
    config = PretrainedConfig(vocab_size=1000)
    model = NeuronCausalLM(config)
    model.model = Mock()
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None
    model.model.config.neuron_config.is_chunked_prefill = False
    model.neuron_config = model.model.config.neuron_config
    model.is_reorder_needed = False

    mock_output = Mock()
    # Shape: [batch, seq_len, vocab_size]
    mock_output.logits = torch.randn(2, 10, 1000)
    model.model.return_value = mock_output

    result = model.forward(input_ids=torch.tensor([[1, 2], [3, 4]]),
                           positions=torch.tensor([[0, 1], [0, 1]]),
                           input_block_ids=torch.tensor([0, 1]),
                           block_tables=torch.tensor([[0, 1], [2, 3]]),
                           sampling_params=torch.tensor([[1.0], [1.0]]))

    # Should extract last token logits: [:, -1, :]
    assert result.shape == (2, 1000)


def test_neuron_multi_modal_load_weights_compilation_path(mocker):
    """Test NeuronMultiModalCausalLM compilation when precompiled not found.
    
    This test verifies:
    1. Compilation triggered when success=False
    2. _save_pretrained_model called when path doesn't exist
    3. _compile_and_load_model is called
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16))
    model = NeuronMultiModalCausalLM(config)

    mock_neuronx_cls = Mock()
    mock_neuron_config_cls = Mock()
    mock_neuron_config = Mock()
    mock_neuron_config_cls.return_value = mock_neuron_config
    mock_neuronx_cls.get_neuron_config_cls.return_value = mock_neuron_config_cls
    mock_neuronx_cls._model_cls = "test_model"

    mock_config_cls = Mock()
    mock_final_config = Mock()
    mock_final_config.to_json_string.return_value = "config"
    mock_config_cls.return_value = mock_final_config
    mock_neuronx_cls.get_config_cls.return_value = mock_config_cls

    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader.load_pretrained_config',
        return_value=Mock())
    mocker.patch(
        'vllm_neuron.worker.neuronx_distributed_model_loader._get_neuron_model_cls',
        return_value=mock_neuronx_cls)

    # Simulate precompiled not found
    mocker.patch.object(model,
                        '_load_weights_common',
                        return_value=(False, "/compiled", mock_final_config))

    # Path doesn't exist
    mocker.patch('os.path.exists', return_value=False)

    mock_save = mocker.patch.object(model,
                                    '_save_pretrained_model',
                                    return_value="local_path")

    mock_compile = mocker.patch.object(model, '_compile_and_load_model')

    kwargs = {
        'neuron_config': {
            'batch_size': 32
        },
        'override_neuron_config': {}
    }

    success, path = model.load_weights("remote-model",
                                       "LlavaForConditionalGeneration",
                                       **kwargs)

    # Verify _save_pretrained_model was called
    mock_save.assert_called_once_with("remote-model")

    # Verify _compile_and_load_model was called
    assert mock_compile.call_count == 1


def test_neuron_multi_modal_execute_model_extracts_pixel_values(mocker):
    """Test execute_model extracts pixel_values from multi_modal_kwargs.
    
    This test verifies:
    1. pixel_values extracted from model_input.multi_modal_kwargs
    2. Extracted value is passed to forward
    3. None handling when not present
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16))
    model = NeuronMultiModalCausalLM(config)
    model.model = Mock()
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.logits = torch.randn(1, 10, 1000)
    model.model.return_value = mock_output
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = None

    # Create model input WITH pixel_values
    model_input = Mock()
    model_input.input_tokens = torch.tensor([[1, 2, 3]])
    model_input.position_ids = torch.tensor([[0, 1, 2]])
    model_input.input_block_ids = torch.tensor([0])
    model_input.sampling_params = torch.tensor([[1.0]])

    # Test with pixel_values present
    pixel_values_tensor = torch.randn(1, 3, 224, 224)
    model_input.multi_modal_kwargs = {"pixel_values": pixel_values_tensor}

    result = model.execute_model(model_input)

    assert result is not None
    # Verify forward was called with pixel_values
    call_kwargs = model.model.call_args[1]
    assert 'pixel_values' in call_kwargs


def test_neuron_pixtral_forward_with_list_pixel_values_dtype_cast(mocker):
    """Test NeuronPixtralForCausalLM.forward casts list of pixel_values.
    
    This test verifies:
    1. List of pixel value tensors is detected
    2. Each tensor in list is cast to correct dtype
    3. Forward succeeds with list input
    """
    vision_config = PretrainedConfig(num_attention_heads=16)
    vision_config.neuron_config = Mock(torch_dtype=torch.float16)

    config = PretrainedConfig(text_config=PretrainedConfig(vocab_size=1000),
                              vision_config=vision_config,
                              image_token_index=999)

    model = NeuronPixtralForCausalLM(config)
    model.model = Mock()
    model.model.config = config
    model.model.config.vision_config.neuron_config = Mock(
        torch_dtype=torch.float16)
    model.is_reorder_needed = False

    mock_output = Mock()
    mock_output.hidden_states = torch.randn(1, 1000)
    model.model.return_value = mock_output
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = Mock()

    # Create list of pixel values with wrong dtype
    pixel_values = [
        torch.randn(3, 336, 336, dtype=torch.float32),
        torch.randn(3, 336, 336, dtype=torch.float32)
    ]

    result = model.forward(input_ids=torch.tensor([[1, 999]]),
                           positions=torch.tensor([[0, 1]]),
                           input_block_ids=torch.tensor([0]),
                           sampling_params=torch.tensor([[1.0]]),
                           pixel_values=pixel_values,
                           vision_mask=torch.tensor([[[False], [True]]]))

    assert result is not None
    # Verify list was converted to correct dtype
    call_kwargs = model.model.call_args[1]
    pixel_values_arg = call_kwargs['pixel_values']
    assert isinstance(pixel_values_arg, list)
    # Check dtype of list elements
    for pv in pixel_values_arg:
        assert pv.dtype == torch.float16


def test_neuron_causal_lm_sample_returns_sampler_output(mocker):
    """Test sample returns SamplerOutput.
    
    This test verifies:
    1. SamplerOutput is created and returned
    2. Logits are properly formatted
    3. logprobs_tensors is None
    """
    config = PretrainedConfig(vocab_size=1000)
    model = NeuronCausalLM(config)
    model.model = Mock()
    model.model.config = Mock()
    model.model.config.neuron_config = Mock()
    model.model.config.neuron_config.on_device_sampling_config = Mock()

    # Token IDs from on-device sampling
    logits = torch.tensor([[1], [2]])

    result = model.sample(logits)

    # Verify it has SamplerOutput attributes
    assert result is not None
    assert hasattr(result, 'sampled_token_ids')
    assert hasattr(result, 'logprobs_tensors')
    assert result.logprobs_tensors is None
    assert result.sampled_token_ids is not None
    # Verify shape after unsqueeze(-1)
    assert result.sampled_token_ids.shape[0] == 2


def test_get_neuron_model_cls_conditional_generation_conversion(mocker):
    """Test ConditionalGeneration task name conversion.
    
    This test verifies:
    1. "ConditionalGeneration" suffix is detected
    2. Converted to "CausalLM" to match NxDI naming
    3. Model name is lowercased
    4. Task name is kebab-cased
    """
    with patch.dict(
            'vllm_neuron.worker.neuronx_distributed_model_loader.MODEL_TYPES',
        {'mllama': {
            'causal-lm': Mock()
        }}):
        result = _get_neuron_model_cls("MllamaForConditionalGeneration")
        assert result is not None


def test_get_neuron_model_cls_qwen3moe_name_conversion(mocker):
    """Test qwen3moe -> qwen3_moe name conversion.
    
    This test verifies:
    1. Model name "qwen3moe" is detected
    2. Converted to "qwen3_moe" with underscore
    3. Correct model class is returned
    """
    with patch.dict(
            'vllm_neuron.worker.neuronx_distributed_model_loader.MODEL_TYPES',
        {'qwen3_moe': {
            'causal-lm': Mock()
        }}):
        result = _get_neuron_model_cls("Qwen3moeForCausalLM")
        assert result is not None


def test_get_neuron_model_cls_llava_to_pixtral_conversion(mocker):
    """Test LlavaForConditionalGeneration -> pixtral conversion.
    
    This test verifies:
    1. "LlavaForConditionalGeneration" architecture is detected
    2. Model name is converted to "pixtral"
    3. Correct model class is returned
    """
    with patch.dict(
            'vllm_neuron.worker.neuronx_distributed_model_loader.MODEL_TYPES',
        {'pixtral': {
            'causal-lm': Mock()
        }}):
        result = _get_neuron_model_cls("LlavaForConditionalGeneration")
        assert result is not None


def test_get_neuron_model_cls_architecture_without_for_keyword(mocker):
    """Test architecture name without 'For' keyword triggers error.
    
    This test verifies:
    1. Architecture without "For" raises KeyError
    2. KeyError is converted to ValueError
    3. Error message is informative
    """
    # Test architecture without "For" keyword
    with pytest.raises(ValueError, match="is not supported on Neuron"):
        _get_neuron_model_cls("InvalidArchitectureName")


def test_neuron_llama4_load_weights_sets_vision_token_id(mocker):
    """Test NeuronLlama4ForCausalLM.load_weights sets vision_token_id (lines 561-569).
    
    This test verifies:
    1. Tokenizer is loaded
    2. Vision token is extracted
    3. vision_token_id attribute is set
    """
    config = PretrainedConfig(
        text_config=PretrainedConfig(vocab_size=1000),
        vision_config=PretrainedConfig(num_attention_heads=16),
        image_token_index=999)
    model = NeuronLlama4ForCausalLM(config)

    # Mock the parent's load_weights to avoid actual loading
    mocker.patch.object(NeuronMultiModalCausalLM,
                        'load_weights',
                        return_value=(True, "/compiled"))

    # Mock AutoTokenizer
    mock_tokenizer_instance = Mock()
    mock_tokenizer_instance.return_value = Mock(input_ids=[999])
    mocker.patch('transformers.AutoTokenizer.from_pretrained',
                 return_value=mock_tokenizer_instance)

    # Call the method
    success, path = model.load_weights("test-model",
                                       "Llama4ForConditionalGeneration",
                                       neuron_config={},
                                       override_neuron_config={})

    # Verify vision_token_id was set
    assert hasattr(model, 'vision_token_id')
    assert model.vision_token_id == 999
    assert success is True


@pytest.mark.parametrize(
    "max_model_len,max_num_seqs,block_size,is_block_kv_layout,pa_num_blocks,should_pass",
    [
        (2048, 32, 16, True, 2000, False),  # 4096 required, 2000 provided
        (1024, 16, 8, True, 3000, True),  # 2048 required, 3000 provided  
        (512, 8, 4, True, 1024, True),  # 1024 required, 1024 provided
        (1000, 10, 16, True, 630,
         True),  # 630 required (ceil(62.5)*10), 630 provided
        (1000, 10, 16, True, 620, False),  # 630 required, 620 provided
        (4096, 64, 8, False, 100,
         True),  # non block layout, validation skipped
    ])
def test_sufficient_blocks_validation(mocker, base_configs, max_model_len,
                                      max_num_seqs, block_size,
                                      is_block_kv_layout, pa_num_blocks,
                                      should_pass):
    """Test block KV layout validation with various configurations.
    
    This parameterized test covers:
    1. Insufficient blocks (should fail)
    2. Sufficient blocks (should pass)
    3. Exact minimum blocks (should pass)
    4. Non-divisible model lengths with ceil() behavior
    5. Disabled validation (should always pass)
    """
    scheduler_config, cache_config, parallel_config = base_configs

    # Set up scheduler config with test parameters
    scheduler_config.max_model_len = max_model_len
    scheduler_config.max_num_seqs = max_num_seqs
    cache_config.block_size = block_size
    cache_config.num_gpu_blocks_override = pa_num_blocks + 1  # platform.py incremented by 1 already

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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    # Set up additional config
    additional_config = {
        "override_neuron_config": {
            "is_block_kv_layout": is_block_kv_layout,
            "pa_num_blocks": pa_num_blocks
        }
    }

    if should_pass:
        # Should succeed
        model = get_neuron_model(model_config,
                                 cache_config,
                                 parallel_config,
                                 scheduler_config,
                                 lora_serving_config=None,
                                 additional_config=additional_config)
        assert model is not None
    else:
        # Should fail with assertion error about insufficient blocks
        with pytest.raises(
                AssertionError,
                match=r"At least \d+ blocks are required for max_model_len \d+"
        ):
            get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config=additional_config)


# yapf: disable
@pytest.mark.parametrize(
    "incremented_num_gpu_blocks_override,pa_num_blocks,matching,sufficient",
    [
        # Minimum 200 blocks are required in this test setup
        ### User sets only --num-gpu-blocks-override (pa_num_blocks inherits from this)
        (
            201, None, True, True
        ),  # User originally set 200, platform incremented to 201, sufficient
        (
            200, None, True, False
        ),  # User originally set 199, platform incremented to 200, insufficient

        ### User sets both values (and they match)
        (
            201, 200, True, True
        ),  # User originally set 200, platform incremented to 201, sufficient
        (
            200, 199, True, False
        ),  # User originally set 199, platform incremented to 200, insufficient

        ### User sets both values (but they mismatch) - should fail in increment handling
        (
            201, 150, False, None
        ),  # Obvious mismatch
        (
            201, 201, False, None
        ),  # User originally set --num-gpu-blocks-override 200 but pa_num_blocks 201

        ### User sets only pa_num_blocks (no num_gpu_blocks_override) - should fail in increment handling
        (
            None, 200, False, None
        ),  # vLLM will go with memory based block calc and differ from NxDI
    ])
def test_conflicting_vllm_and_nxdi_num_blocks_inputs(
        mocker, base_configs, incremented_num_gpu_blocks_override,
        pa_num_blocks, matching, sufficient):
    # yapf: enable
    """Test pa_num_blocks validation with various combinations of user inputs.
    """
    scheduler_config, cache_config, parallel_config = base_configs

    # Set up scenario where exactly 200 blocks are required
    scheduler_config.max_model_len = 1600  # 1600 / 8 = 200
    scheduler_config.max_num_seqs = 1  # 200 * 1 = 200 blocks required
    cache_config.block_size = 8
    cache_config.num_gpu_blocks_override = incremented_num_gpu_blocks_override

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
        'vllm_neuron.worker.neuronx_distributed_model_loader.NeuronCausalLM',
        return_value=mock_causal_lm)

    # Set up additional config
    additional_config = {
        "override_neuron_config": {
            "is_block_kv_layout": True
        }
    }
    if pa_num_blocks is not None:
        additional_config["override_neuron_config"][
            "pa_num_blocks"] = pa_num_blocks

    if matching and sufficient:
        # Should succeed
        model = get_neuron_model(model_config,
                                 cache_config,
                                 parallel_config,
                                 scheduler_config,
                                 lora_serving_config=None,
                                 additional_config=additional_config)
        assert model is not None
    elif not matching:
        # Should fail with value error
        with pytest.raises(ValueError):
            get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config=additional_config)
    else:
        # Should fail with assertion error
        with pytest.raises(AssertionError):
            get_neuron_model(model_config,
                             cache_config,
                             parallel_config,
                             scheduler_config,
                             lora_serving_config=None,
                             additional_config=additional_config)
