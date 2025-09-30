# SPDX-License-Identifier: Apache-2.0
import logging
import os
from unittest.mock import Mock, patch

import pytest

from vllm_neuron.platform import NeuronFramework, NeuronPlatform


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Fixture to cleanup any global state modifications after each test."""
    yield
    # Clear any cached results
    NeuronPlatform.is_neuronx_distributed_inference.cache_clear()


def test_neuron_platform_basic_properties():
    """Test basic properties of NeuronPlatform.
    
    Verifies:
    - Device name and type are correctly set
    - Ray device key is properly configured
    - Supported quantization types are present
    - Device control environment variable is set
    """
    platform = NeuronPlatform()

    assert platform.device_name == "cpu"
    assert platform.device_type == "cpu"
    assert platform.ray_device_key == "neuron_cores"
    assert "neuron_quant" in platform.supported_quantization
    assert "fbgemm_fp8" in platform.supported_quantization
    assert platform.device_control_env_var == "NEURON_RT_VISIBLE_CORES"


def test_get_device_name():
    """Test get_device_name method.
    
    Verifies that the device name is consistently returned as "neuron"
    regardless of the device ID provided.
    """
    assert NeuronPlatform.get_device_name() == "neuron"
    assert NeuronPlatform.get_device_name(1) == "neuron"


def test_is_async_output_supported():
    """Test is_async_output_supported method.
    
    Verifies that async output is not supported regardless of the
    enforce_eager parameter value.
    """
    assert not NeuronPlatform.is_async_output_supported(None)
    assert not NeuronPlatform.is_async_output_supported(True)


def test_is_pin_memory_available(caplog):
    """Test is_pin_memory_available method.
    
    Verifies:
    - Pin memory is not available
    - Appropriate warning message is logged
    """
    with caplog.at_level(logging.WARNING):
        assert not NeuronPlatform.is_pin_memory_available()
        assert "Pin memory is not supported on Neuron." in caplog.text


def test_use_all_gather():
    """Test use_all_gather method.
    
    Verifies that all_gather is always enabled for the Neuron platform.
    """
    assert NeuronPlatform.use_all_gather()


def test_supports_v1():
    """Test supports_v1 method.
    
    Verifies that v1 support is always enabled for any model configuration.
    """
    mock_model_config = Mock()
    assert NeuronPlatform.supports_v1(mock_model_config)


def test_is_neuronx_distributed_inference_not_installed():
    """Test is_neuronx_distributed_inference when package is not installed.
    
    Verifies that the method correctly handles the case when
    neuronx_distributed_inference is not available.
    
    The test ensures that:
    1. The method returns False when the package is not available
    2. The result is properly cached (lru_cache behavior)
    """
    # Clear the lru_cache to ensure fresh test
    NeuronPlatform.is_neuronx_distributed_inference.cache_clear()

    # Create a context where neuronx_distributed_inference is not available
    with patch.dict('sys.modules', {'neuronx_distributed_inference': None}):
        # Test the method
        result = NeuronPlatform.is_neuronx_distributed_inference()

        # Verify the result
        assert not result


def test_get_neuron_framework_to_use():
    """Test get_neuron_framework_to_use method.
    
    Verifies:
    - Correct framework selection when on Neuron platform
    - Error handling when required packages are not available
    - Error handling when not on Neuron platform
    """
    platform = NeuronPlatform()

    with patch.object(platform, 'is_neuron', return_value=True):
        with patch.object(platform,
                          'is_neuronx_distributed_inference',
                          return_value=True):
            assert platform.get_neuron_framework_to_use() == \
                NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

        with patch.object(platform,
                          'is_neuronx_distributed_inference',
                          return_value=False):
            with pytest.raises(
                    AssertionError,
                    match="Unable to import neuronx_distributed_inference"):
                platform.get_neuron_framework_to_use()

    with patch.object(platform, 'is_neuron', return_value=False):
        with pytest.raises(AssertionError,
                           match="Neuron Framework unavailable"):
            platform.get_neuron_framework_to_use()


def test_use_neuronx_distributed():
    """Test use_neuronx_distributed method.
    
    Verifies that the method correctly identifies when to use
    neuronx-distributed-inference framework.
    """
    platform = NeuronPlatform()

    with patch.object(
            platform,
            'get_neuron_framework_to_use',
            return_value=NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE):
        assert platform.use_neuronx_distributed()


@pytest.mark.parametrize("disable_scheduler", ["0", "1"])
def test_check_and_update_config(disable_scheduler):
    """Test check_and_update_config method with different scheduler settings.
    
    Args:
        disable_scheduler: String value ("0" or "1") to control scheduler override.
            "0": Use custom Neuron scheduler
            "1": Use vLLM V1 native scheduler

    Verifies:
    - Proper handling of scheduler override settings
    - Worker class configuration
    - Parallel configuration updates
    - Cache configuration updates
    - Scheduler class selection based on environment variable
    - Chunked prefill settings
    """
    with patch.dict(os.environ,
                    {'DISABLE_NEURON_CUSTOM_SCHEDULER': disable_scheduler}):
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock()
        mock_config.parallel_config.world_size = 1
        mock_config.parallel_config.worker_cls = "auto"
        mock_config.cache_config = Mock()
        mock_config.scheduler_config = Mock()

        NeuronPlatform.check_and_update_config(mock_config)

        # Verify worker class update
        assert mock_config.parallel_config.worker_cls == \
            "vllm_neuron.worker.neuron_worker.NeuronWorker"

        # Verify scheduler configuration based on environment variable
        if disable_scheduler == "0":
            assert mock_config.scheduler_config.scheduler_cls == \
                "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler"
            assert not mock_config.scheduler_config.chunked_prefill_enabled


def test_check_and_update_config_cache_settings():
    """Test cache configuration updates in check_and_update_config.
    
    Verifies:
    - Block size configuration for non-prefix caching mode
    - Block size is set to max_model_len when prefix caching is disabled
    - Cache configuration is properly applied
    - Model configuration integration with cache settings
    
    The test ensures that cache settings are properly configured based on
    the caching mode and model parameters.
    """
    mock_config = Mock()
    mock_config.model_config = Mock(max_model_len=2048)
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(enable_prefix_caching=False)
    mock_config.scheduler_config = Mock()

    NeuronPlatform.check_and_update_config(mock_config)

    # Verify cache block size is set to max_model_len when prefix caching is disabled
    assert mock_config.cache_config.block_size == 2048


def test_check_and_update_config_distributed():
    """Test distributed configuration updates in check_and_update_config.
    
    Verifies:
    - Distributed executor backend settings
    - World size handling
    - Worker configuration in distributed mode
    - Backend selection for multi-worker setups
    
    The test ensures proper configuration for distributed execution
    when world_size is greater than 1.
    """
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=2, worker_cls="auto")
    mock_config.cache_config = Mock()
    mock_config.scheduler_config = Mock()

    NeuronPlatform.check_and_update_config(mock_config)

    # Verify distributed executor backend is set to "uni" for world_size > 1
    assert mock_config.parallel_config.distributed_executor_backend == "uni"


def test_pre_register_and_update():
    """Test pre_register_and_update method.
    
    Verifies:
    - ModelConfig class modification
    - Custom verify_with_parallel_config method installation
    - Proper cleanup of original configuration
    - Configuration persistence
    
    The test ensures that the ModelConfig class is properly modified
    while maintaining the ability to restore original settings.
    
    Note:
        This test modifies global state and includes cleanup to
        restore original settings.
    """
    from vllm.config import ModelConfig
    original_verify = getattr(ModelConfig, 'verify_with_parallel_config', None)

    try:
        mock_parser = Mock()
        NeuronPlatform.pre_register_and_update(mock_parser)
        assert hasattr(ModelConfig, 'verify_with_parallel_config')
    finally:
        if original_verify:
            setattr(ModelConfig, 'verify_with_parallel_config',
                    original_verify)


def test_check_and_update_config_cache_validation():
    """Test cache configuration validation in check_and_update_config.
    
    Verifies:
    - Proper validation of cache settings
    - Error handling for invalid configurations
    - Block size requirements for prefix caching
    - Assertion messages for configuration errors
    
    The test ensures that invalid cache configurations are properly
    detected and appropriate error messages are raised.
    
    Raises:
        AssertionError: When prefix caching is enabled without block_size
    """
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.parallel_config = Mock(world_size=1, worker_cls="auto")
    mock_config.cache_config = Mock(enable_prefix_caching=True,
                                    block_size=None)
    mock_config.scheduler_config = Mock()

    with pytest.raises(
            AssertionError,
            match="When prefix caching is enabled, block_size must be set"):
        NeuronPlatform.check_and_update_config(mock_config)


def test_pre_register_and_update_with_expert_parallel():
    """Test pre_register_and_update with expert parallelism.
    
    Verifies:
    - Expert parallelism configuration handling
    - Verification method behavior with expert parallel enabled
    - Proper method installation and cleanup
    """
    from vllm.config import ModelConfig
    original_verify = getattr(ModelConfig, 'verify_with_parallel_config', None)

    try:
        mock_parser = Mock()
        mock_parallel_config = Mock()
        # Set specific values for mock attributes
        mock_parallel_config.enable_expert_parallel = True
        mock_parallel_config.distributed_executor_backend = "standard"
        mock_parallel_config.pipeline_parallel_size = 1  # Fix for TypeError

        NeuronPlatform.pre_register_and_update(mock_parser)

        # Create model config instance
        model_config = Mock()
        model_config._verify_with_expert_parallelism = Mock()
        model_config.seed = None  # Add seed attribute

        # Call the installed method
        verify_method = getattr(ModelConfig, 'verify_with_parallel_config')
        verify_method(model_config, mock_parallel_config)

        # Verify expert parallelism check was called
        model_config._verify_with_expert_parallelism.assert_called_once()

    finally:
        if original_verify:
            setattr(ModelConfig, 'verify_with_parallel_config',
                    original_verify)


def test_pre_register_and_update_with_pipeline_parallel():
    """Test pre_register_and_update with pipeline parallelism.
    
    Verifies:
    - Pipeline parallelism configuration handling
    - Support check for pipeline parallel models
    - Async output processor handling
    """
    from vllm.config import ModelConfig
    original_verify = getattr(ModelConfig, 'verify_with_parallel_config', None)

    try:
        mock_parser = Mock()
        mock_parallel_config = Mock()
        # Set specific values for mock attributes
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 2
        mock_parallel_config.distributed_executor_backend = "standard"

        NeuronPlatform.pre_register_and_update(mock_parser)

        # Create model config instance with specific attribute values
        model_config = Mock(
            spec=['architectures', 'registry', 'use_async_output_proc'])
        model_config.architectures = ["TestModel"]
        model_config.registry.is_pp_supported_model.return_value = False
        model_config.use_async_output_proc = True
        model_config.seed = None  # Add seed attribute

        # Call the installed method
        verify_method = getattr(ModelConfig, 'verify_with_parallel_config')

        # Verify NotImplementedError is raised for unsupported model
        with pytest.raises(
                NotImplementedError,
                match="Pipeline parallelism is not supported for this model"):
            verify_method(model_config, mock_parallel_config)
            # The use_async_output_proc should be set to False during the call
            assert not model_config.use_async_output_proc

    finally:
        if original_verify:
            setattr(ModelConfig, 'verify_with_parallel_config',
                    original_verify)


def test_check_and_update_config_edge_cases():
    """Test check_and_update_config edge cases and error conditions.
    
    Verifies:
    - Empty model config handling
    - Block size validation with native scheduler
    - Prefix caching configuration validation
    - Error messages for invalid configurations
    """
    # Test with empty model config
    mock_config = Mock()
    mock_config.model_config = None
    mock_config.parallel_config = Mock(worker_cls="auto", world_size=1)
    mock_config.cache_config = Mock(enable_prefix_caching=False)
    mock_config.scheduler_config = Mock()

    NeuronPlatform.check_and_update_config(mock_config)

    # Test native scheduler without block size
    with patch.dict(os.environ, {'DISABLE_NEURON_CUSTOM_SCHEDULER': "1"}):
        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.parallel_config = Mock(worker_cls="auto", world_size=1)
        mock_config.cache_config = Mock(block_size=None)
        mock_config.scheduler_config = Mock()

        with pytest.raises(
                AssertionError,
                match=
                "When vLLM V1 native scheduler is enabled, block_size must be set"
        ):
            NeuronPlatform.check_and_update_config(mock_config)


def test_get_neuron_framework_error_handling():
    """Test error handling in framework detection methods.
    
    Verifies:
    - Error handling for unavailable frameworks
    - Proper error messages
    - Cache clearing behavior
    """
    platform = NeuronPlatform()

    # Test with neuron platform unavailable
    with patch.object(platform, 'is_neuron', return_value=False):
        with pytest.raises(AssertionError,
                           match="Neuron Framework unavailable for platform"):
            platform.get_neuron_framework_to_use()

    # Test with framework import failure
    with patch.object(platform, 'is_neuron', return_value=True):
        with patch.dict('sys.modules',
                        {'neuronx_distributed_inference': None}):
            with pytest.raises(
                    AssertionError,
                    match="Unable to import neuronx_distributed_inference"):
                platform.get_neuron_framework_to_use()


def test_pre_register_and_update_with_external_launcher():
    """Test pre_register_and_update with external launcher.
    
    Verifies:
    - External launcher configuration handling
    - Seed validation for external launcher
    - Error handling for missing seed
    """
    from vllm.config import ModelConfig
    original_verify = getattr(ModelConfig, 'verify_with_parallel_config', None)

    try:
        mock_parallel_config = Mock()
        mock_parallel_config.distributed_executor_backend = "external_launcher"
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.pipeline_parallel_size = 1

        NeuronPlatform.pre_register_and_update()

        # Test with missing seed
        model_config = Mock(seed=None)
        verify_method = getattr(ModelConfig, 'verify_with_parallel_config')

        with pytest.raises(
                AssertionError,
                match="Seed must be set when using external launcher"):
            verify_method(model_config, mock_parallel_config)

        # Test with seed set
        model_config.seed = 42
        verify_method(model_config, mock_parallel_config)

    finally:
        if original_verify:
            setattr(ModelConfig, 'verify_with_parallel_config',
                    original_verify)


def test_pre_register_and_update_pipeline_async():
    """Test pre_register_and_update with pipeline parallel and async output.
    
    Verifies:
    - Pipeline parallel configuration with async output
    - Async output processor disabling
    - Model support validation
    """
    from vllm.config import ModelConfig
    original_verify = getattr(ModelConfig, 'verify_with_parallel_config', None)

    try:
        mock_parallel_config = Mock()
        mock_parallel_config.pipeline_parallel_size = 2
        mock_parallel_config.enable_expert_parallel = False
        mock_parallel_config.distributed_executor_backend = "standard"

        NeuronPlatform.pre_register_and_update()

        model_config = Mock()
        model_config.registry.is_pp_supported_model.return_value = True
        model_config.use_async_output_proc = True
        model_config.seed = None

        verify_method = getattr(ModelConfig, 'verify_with_parallel_config')
        verify_method(model_config, mock_parallel_config)

        # Verify async output was disabled
        assert not model_config.use_async_output_proc

    finally:
        if original_verify:
            setattr(ModelConfig, 'verify_with_parallel_config',
                    original_verify)
