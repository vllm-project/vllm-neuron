# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CPU sampling functionality in NeuronxDistributedModelRunner.
"""
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_neuron.worker.neuronx_distributed_model_runner import (
    ModelInputForNeuron, NeuronxDistributedModelRunner)


class TestCPUSampling:
    """Test CPU sampling functionality."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig for testing."""
        config = Mock(spec=VllmConfig)

        # Mock model config
        model_config = Mock(spec=ModelConfig)
        model_config.get_vocab_size.return_value = 32000
        model_config.max_model_len = 2048

        # Mock cache config
        cache_config = Mock(spec=CacheConfig)
        cache_config.block_size = 16

        # Mock scheduler config
        scheduler_config = Mock(spec=SchedulerConfig)
        scheduler_config.max_num_seqs = 8
        scheduler_config.max_num_batched_tokens = 1024
        scheduler_config.max_model_len = 2048

        config.model_config = model_config
        config.cache_config = cache_config
        config.scheduler_config = scheduler_config
        config.lora_config = None
        config.load_config = Mock()
        config.parallel_config = Mock()
        config.speculative_config = None
        config.observability_config = Mock()
        config.device_config = Mock()
        config.additional_config = {}

        return config

    @pytest.fixture
    def mock_model_runner(self, mock_vllm_config):
        """Create a mock model runner for testing."""
        device = torch.device('cpu')

        with patch('vllm_neuron.worker.neuronx_distributed_model_runner.InputBatch'), \
             patch('vllm_neuron.worker.neuronx_distributed_model_runner.Sampler'):

            runner = NeuronxDistributedModelRunner(mock_vllm_config, device)

            # Mock the model and its config
            runner.model = Mock()
            runner.model.neuron_config = Mock()
            runner.model.neuron_config.on_device_sampling_config = None  # CPU sampling

            # Mock input batch with sampling metadata
            runner.input_batch = Mock()
            runner.input_batch.sampling_metadata = Mock(spec=SamplingMetadata)
            runner.input_batch.req_ids = ['req1', 'req2']

            # Mock CPU sampler
            runner.cpu_sampler = Mock()

            return runner

    def test_cpu_sample_success(self, mock_model_runner):
        """Test successful CPU sampling."""
        # Prepare test data
        batch_size, vocab_size = 2, 32000
        logits = torch.randn(batch_size, vocab_size)

        model_input = ModelInputForNeuron(request_ids=['req1', 'req2'])

        # Mock sampler output
        expected_output = Mock(spec=SamplerOutput)
        expected_output.sampled_token_ids = [[1, 2], [3, 4]]
        mock_model_runner.cpu_sampler.return_value = expected_output

        # Test CPU sampling
        result = mock_model_runner._cpu_sample(logits, model_input)

        # Verify
        assert result == expected_output
        mock_model_runner.cpu_sampler.assert_called_once_with(
            logits, mock_model_runner.input_batch.sampling_metadata)

    def test_cpu_sample_invalid_logits_dimensions(self, mock_model_runner):
        """Test CPU sampling with invalid logits dimensions."""
        # 3D logits should fail
        logits = torch.randn(2, 32000, 1)
        model_input = ModelInputForNeuron(request_ids=['req1'])

        with pytest.raises(RuntimeError,
                           match="Expected logits to be 2D tensor"):
            mock_model_runner._cpu_sample(logits, model_input)

    def test_cpu_sample_invalid_vocab_size(self, mock_model_runner):
        """Test CPU sampling with mismatched vocab size."""
        # Wrong vocab size
        logits = torch.randn(2, 16000)  # Should be 32000
        model_input = ModelInputForNeuron(request_ids=['req1'])

        with pytest.raises(
                RuntimeError,
                match="Logits vocab size .* does not match model vocab size"):
            mock_model_runner._cpu_sample(logits, model_input)

    def test_cpu_sample_no_sampling_metadata(self, mock_model_runner):
        """Test CPU sampling when sampling metadata is None."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1'])

        # Set sampling metadata to None
        mock_model_runner.input_batch.sampling_metadata = None

        with pytest.raises(RuntimeError,
                           match="CPU sampling requires sampling metadata"):
            mock_model_runner._cpu_sample(logits, model_input)

    def test_cpu_sample_sampler_returns_none(self, mock_model_runner):
        """Test CPU sampling when sampler returns None."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1'])

        # Mock sampler to return None
        mock_model_runner.cpu_sampler.return_value = None

        with pytest.raises(RuntimeError,
                           match="CPU sampler returned None output"):
            mock_model_runner._cpu_sample(logits, model_input)

    def test_cpu_sample_sampler_returns_none_token_ids(self,
                                                       mock_model_runner):
        """Test CPU sampling when sampler returns output with None token IDs."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1'])

        # Mock sampler to return output with None token IDs
        mock_output = Mock(spec=SamplerOutput)
        mock_output.sampled_token_ids = None
        mock_model_runner.cpu_sampler.return_value = mock_output

        with pytest.raises(
                RuntimeError,
                match="CPU sampler returned None sampled_token_ids"):
            mock_model_runner._cpu_sample(logits, model_input)

    def test_sampling_method_selection_cpu(self, mock_model_runner):
        """Test that CPU sampling is selected when on_device_sampling_config is None."""
        hidden_states = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1', 'req2'])

        # Mock _cpu_sample method
        expected_output = Mock(spec=SamplerOutput)
        mock_model_runner._cpu_sample = Mock(return_value=expected_output)

        # Ensure on_device_sampling_config is None
        mock_model_runner.model.neuron_config.on_device_sampling_config = None

        result = mock_model_runner._sample(hidden_states, model_input)

        # Verify CPU sampling was called
        mock_model_runner._cpu_sample.assert_called_once()
        assert result == expected_output

    def test_sampling_method_selection_hardware(self, mock_model_runner):
        """Test that hardware sampling is selected when on_device_sampling_config is set."""
        hidden_states = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1', 'req2'])

        # Mock hardware sampling
        expected_output = Mock(spec=SamplerOutput)
        mock_model_runner.model.sample = Mock(return_value=expected_output)

        # Set on_device_sampling_config to non-None
        mock_model_runner.model.neuron_config.on_device_sampling_config = Mock(
        )

        result = mock_model_runner._sample(hidden_states, model_input)

        # Verify hardware sampling was called - use manual verification to avoid tensor comparison issues
        mock_model_runner.model.sample.assert_called_once()
        call_args = mock_model_runner.model.sample.call_args

        # Verify the call was made with logits keyword argument
        assert 'logits' in call_args.kwargs
        actual_logits = call_args.kwargs['logits']

        # Verify the logits tensor has the expected shape and values
        assert isinstance(actual_logits, torch.Tensor)
        assert torch.allclose(actual_logits, hidden_states, rtol=1e-5)
        assert result == expected_output

    def test_sampling_configuration_validation_cpu_mode(
            self, mock_model_runner):
        """Test sampling configuration validation for CPU mode."""
        # Set up CPU sampling mode
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        mock_model_runner.model.sample = Mock()  # Ensure sample method exists

        # Should not raise any exceptions
        mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_hardware_mode(
            self, mock_model_runner):
        """Test sampling configuration validation for hardware mode."""
        # Set up hardware sampling mode
        mock_config = Mock()
        mock_config.global_topk = 50
        mock_model_runner.model.neuron_config.on_device_sampling_config = mock_config
        mock_model_runner.model.sample = Mock()  # Ensure sample method exists

        # Should not raise any exceptions
        mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_missing_cpu_sampler(
            self, mock_model_runner):
        """Test validation fails when CPU sampler is missing."""
        # Set up CPU sampling mode but remove CPU sampler
        mock_model_runner.model.neuron_config.on_device_sampling_config = None
        mock_model_runner.cpu_sampler = None

        with pytest.raises(
                RuntimeError,
                match=
                "CPU sampling is required but cpu_sampler is not initialized"):
            mock_model_runner._validate_sampling_configuration()

    def test_sampling_configuration_validation_missing_sample_method(
            self, mock_model_runner):
        """Test validation fails when model sample method is missing."""
        # Remove sample method from model
        if hasattr(mock_model_runner.model, 'sample'):
            delattr(mock_model_runner.model, 'sample')

        with pytest.raises(
                RuntimeError,
                match="Model does not have required 'sample' method"):
            mock_model_runner._validate_sampling_configuration()

    def test_sampling_error_handling(self, mock_model_runner):
        """Test error handling in sampling method selection."""
        hidden_states = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1', 'req2'])

        # Mock _cpu_sample to raise an exception
        mock_model_runner._cpu_sample = Mock(
            side_effect=RuntimeError("CPU sampling failed"))
        mock_model_runner.model.neuron_config.on_device_sampling_config = None

        with pytest.raises(RuntimeError, match="Sampling operation failed"):
            mock_model_runner._sample(hidden_states, model_input)

    def test_cpu_sample_exception_handling(self, mock_model_runner):
        """Test exception handling in CPU sampling method."""
        logits = torch.randn(2, 32000)
        model_input = ModelInputForNeuron(request_ids=['req1'])

        # Mock sampler to raise an exception
        mock_model_runner.cpu_sampler.side_effect = RuntimeError(
            "Sampler error")

        with pytest.raises(RuntimeError, match="CPU sampling failed"):
            mock_model_runner._cpu_sample(logits, model_input)


class TestCPUSamplingIntegration:
    """Integration tests for CPU sampling with realistic scenarios."""

    @pytest.fixture
    def integration_model_runner(self):
        """Create a more realistic model runner for integration testing."""
        # Create real config objects (but still mock some dependencies)
        model_config = Mock()
        model_config.get_vocab_size.return_value = 32000
        model_config.max_model_len = 2048

        cache_config = Mock()
        cache_config.block_size = 16

        scheduler_config = Mock()
        scheduler_config.max_num_seqs = 8
        scheduler_config.max_num_batched_tokens = 1024
        scheduler_config.max_model_len = 2048

        vllm_config = Mock()
        vllm_config.model_config = model_config
        vllm_config.cache_config = cache_config
        vllm_config.scheduler_config = scheduler_config
        vllm_config.lora_config = None
        vllm_config.load_config = Mock()
        vllm_config.parallel_config = Mock()
        vllm_config.speculative_config = None
        vllm_config.observability_config = Mock()
        vllm_config.device_config = Mock()
        vllm_config.additional_config = {}

        device = torch.device('cpu')

        with patch('vllm_neuron.worker.neuronx_distributed_model_runner.InputBatch'), \
             patch('vllm_neuron.worker.neuronx_distributed_model_runner.Sampler') as mock_sampler_class:

            # Create a real sampler instance (mocked)
            mock_sampler_instance = Mock()
            mock_sampler_class.return_value = mock_sampler_instance

            runner = NeuronxDistributedModelRunner(vllm_config, device)

            # Set up model with CPU sampling configuration
            runner.model = Mock()
            runner.model.neuron_config = Mock()
            runner.model.neuron_config.on_device_sampling_config = None
            runner.model.sample = Mock()

            # Set up input batch
            runner.input_batch.sampling_metadata = Mock(spec=SamplingMetadata)
            runner.input_batch.req_ids = ['req1', 'req2']

            return runner

    def test_end_to_end_cpu_sampling_flow(self, integration_model_runner):
        """Test the complete CPU sampling flow from logits to output."""
        # Prepare realistic test data
        batch_size, vocab_size = 2, 32000
        logits = torch.randn(batch_size, vocab_size)

        model_input = ModelInputForNeuron(request_ids=['req1', 'req2'])

        # Mock realistic sampler output
        sampler_output = Mock(spec=SamplerOutput)
        sampler_output.sampled_token_ids = torch.tensor([[123], [456]])
        integration_model_runner.cpu_sampler.return_value = sampler_output

        # Test the sampling flow
        result = integration_model_runner._cpu_sample(logits, model_input)

        # Verify the result
        assert result == sampler_output
        integration_model_runner.cpu_sampler.assert_called_once_with(
            logits, integration_model_runner.input_batch.sampling_metadata)

    def test_cpu_sampling_with_different_batch_sizes(self,
                                                     integration_model_runner):
        """Test CPU sampling with various batch sizes."""
        test_cases = [1, 2, 4, 8]

        for batch_size in test_cases:
            logits = torch.randn(batch_size, 32000)
            model_input = ModelInputForNeuron(
                request_ids=[f'req{i}' for i in range(batch_size)])

            # Mock sampler output for this batch size
            sampler_output = Mock(spec=SamplerOutput)
            sampler_output.sampled_token_ids = torch.tensor(
                [[i] for i in range(batch_size)])
            integration_model_runner.cpu_sampler.return_value = sampler_output

            # Test sampling
            result = integration_model_runner._cpu_sample(logits, model_input)

            # Verify
            assert result == sampler_output
            integration_model_runner.cpu_sampler.assert_called_with(
                logits, integration_model_runner.input_batch.sampling_metadata)

            # Reset mock for next iteration
            integration_model_runner.cpu_sampler.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__])
