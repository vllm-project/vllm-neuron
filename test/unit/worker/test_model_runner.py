# SPDX-License-Identifier: Apache-2.0
# test/unit/worker/test_model_runner.py
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from neuronx_vllm_plugin.worker.neuronx_distributed_model_runner import (
    ModelInputForNeuron, NeuronxDistributedModelRunner)


# Create mock sampling params that return tensors
class MockSamplingModule(MagicMock):

    def prepare_sampling_params(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)

    def __getitem__(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        return torch.tensor([1.0], dtype=torch.float32)


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
# Use the custom sampling mock
sampling_mock = MockSamplingModule()
mock_base.modules.generation.sampling = sampling_mock
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


@dataclass
class MockVllmConfig:
    model_config: Mock
    cache_config: Mock
    lora_config: Mock = None
    load_config: Mock = None
    parallel_config: Mock = None
    scheduler_config: Mock = None
    speculative_config: Mock = None
    observability_config: Mock = None
    device_config: Mock = None


class TestModelRunner:

    @pytest.fixture
    def vllm_config(self):
        return MockVllmConfig(model_config=Mock(max_model_len=2048),
                              cache_config=Mock(block_size=8),
                              lora_config=Mock(),
                              load_config=Mock(),
                              parallel_config=Mock(tensor_parallel_size=1),
                              scheduler_config=Mock(
                                  max_model_len=2048,
                                  max_num_seqs=32,
                                  max_num_batched_tokens=4096,
                                  chunked_prefill_enabled=False),
                              speculative_config=Mock(),
                              observability_config=Mock(),
                              device_config=Mock(device="cpu"))

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.neuron_config = Mock(on_device_sampling_config=None,
                                   vocab_size=32000,
                                   is_block_kv_layout=False,
                                   is_prefix_caching=False,
                                   chunked_prefill_config=None)
        model.architecture = "LlamaForCausalLM"
        model.num_key_value_heads = 32
        model.head_dim = 64
        return model

    @pytest.fixture
    def model_runner(self, vllm_config, mock_model):
        runner = NeuronxDistributedModelRunner(vllm_config=vllm_config,
                                               device="cpu")
        runner.model = mock_model
        runner.input_batch = Mock()
        runner.input_batch.req_ids = ["req1"]
        return runner

    @pytest.fixture
    def mock_scheduler_output(self):
        cached_reqs = Mock(req_ids=["req1"],
                           num_computed_tokens=[0],
                           new_block_ids=[[0]],
                           resumed_from_preemption=[False])
        return SchedulerOutput(scheduled_new_reqs=[],
                               scheduled_cached_reqs=cached_reqs,
                               num_scheduled_tokens={"req1": 1},
                               finished_req_ids=[],
                               free_encoder_input_ids=[],
                               total_num_scheduled_tokens=1,
                               scheduled_spec_decode_tokens={},
                               scheduled_encoder_inputs=[],
                               num_common_prefix_blocks=0,
                               structured_output_request_ids=[],
                               grammar_bitmask=None,
                               kv_connector_metadata=None)

    @pytest.fixture
    def mock_sampling_module(self):
        return MockSamplingModule()

    @pytest.fixture
    def model_runner(self, vllm_config, mock_model, mock_sampling_module):
        runner = NeuronxDistributedModelRunner(vllm_config=vllm_config,
                                               device="cpu")
        runner.model = mock_model
        runner.input_batch = Mock()
        runner.input_batch.req_ids = ["req1"]
        runner.sampling_module = mock_sampling_module
        return runner

    def test_prepare_model_input(self, model_runner, mock_scheduler_output):
        # Disable LoRA for this test
        model_runner.lora_config = None

        # Setup required state
        req_id = "req1"
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Create a mock sampling params with proper attributes
        mock_sampling_params = Mock()
        mock_sampling_params.top_k = 10  # Set a specific value
        mock_sampling_params.top_p = 0.9
        mock_sampling_params.temperature = 1.0

        model_runner.requests[req_id] = Mock(
            output_token_ids=[1],
            prompt_token_ids=[1],
            block_ids=[[0]],
            sampling_params=mock_sampling_params  # Add sampling params
        )

        # Setup input batch
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock(return_value=None)
        model_runner.input_batch.req_ids = [req_id]
        # Mock the get_nxd_sampling_params method to return a proper tensor
        model_runner.get_nxd_sampling_params = Mock(
            return_value=torch.ones((1, 3), dtype=torch.float32))

        # Setup scheduler output
        mock_scheduler_output.scheduled_cached_reqs.req_ids = [req_id]
        mock_scheduler_output.scheduled_cached_reqs.num_computed_tokens = [0]
        mock_scheduler_output.scheduled_cached_reqs.new_block_ids = [[0]]
        mock_scheduler_output.scheduled_cached_reqs.resumed_from_preemption = [
            False
        ]
        mock_scheduler_output.num_scheduled_tokens = {req_id: 1}

        # Test continuous batching input preparation
        model_input = model_runner._prepare_model_input(mock_scheduler_output)

        # Debug prints
        print("\nDEBUG test_prepare_model_input:")
        print(f"model_input type: {type(model_input)}")
        print(f"model_input.request_ids: {model_input.request_ids}")
        print(
            f"model_input.input_tokens type: {type(model_input.input_tokens)}")
        print(f"model_input.input_tokens: {model_input.input_tokens}")
        print(
            f"model_input.position_ids type: {type(model_input.position_ids)}")
        print(f"model_input.position_ids: {model_input.position_ids}")
        print(
            f"model_input.input_block_ids type: {type(model_input.input_block_ids)}"
        )
        print(f"model_input.input_block_ids: {model_input.input_block_ids}")
        print(
            f"model_input.sampling_params type: {type(model_input.sampling_params)}"
        )
        print(f"model_input.sampling_params: {model_input.sampling_params}")

        # Verify the output
        assert isinstance(model_input, ModelInputForNeuron)
        assert model_input.request_ids is not None
        assert isinstance(model_input.input_tokens, torch.Tensor)
        assert isinstance(model_input.position_ids, torch.Tensor)
        assert isinstance(model_input.input_block_ids, torch.Tensor)
        assert isinstance(model_input.sampling_params, torch.Tensor)

    def test_update_states(self, model_runner, mock_scheduler_output):
        # Setup mock input batch
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock(return_value=None)

        # Create a mutable list for block_ids that can be extended
        class MutableList(list):
            pass

        # Create the block_ids structure: a list containing a mutable list
        inner_list = MutableList([0, 1, 2])
        mock_block_ids = [inner_list]

        # Create a custom Mock for the request state
        class CustomMockState:

            def __init__(self):
                self.block_ids = mock_block_ids
                self.num_computed_tokens = 0

        mock_req_state = CustomMockState()
        model_runner.requests = {"req1": mock_req_state}

        # Setup mock cached request data with proper structure
        class CustomCachedReqs:

            def __init__(self):
                self.req_ids = ["req1"]
                self.num_computed_tokens = [3]
                # Create nested structure: list of lists of lists
                inner_list = MutableList([3, 4, 5])
                self.new_block_ids = [[inner_list]]  # Triple nesting
                self.resumed_from_preemption = [False]

            def __getitem__(self, idx):
                # Return the correct level of nesting
                return self.new_block_ids[0]

        mock_cached_reqs = CustomCachedReqs()
        mock_scheduler_output.scheduled_cached_reqs = mock_cached_reqs

        # Mock other necessary attributes
        mock_scheduler_output.finished_req_ids = []
        mock_scheduler_output.free_encoder_input_ids = []
        mock_scheduler_output.num_scheduled_tokens = {"req1": 1}
        mock_scheduler_output.scheduled_new_reqs = []

        # Initialize encoder cache
        model_runner.encoder_cache = {}

        # Add more debug prints
        print(f"mock_block_ids: {mock_block_ids}")
        print(
            f"mock_cached_reqs.new_block_ids: {mock_cached_reqs.new_block_ids}"
        )
        print(f"mock_cached_reqs[0]: {mock_cached_reqs[0]}")

        # Debug prints and assertions
        test_block_ids = mock_req_state.block_ids[0]
        test_new_ids = mock_scheduler_output.scheduled_cached_reqs.new_block_ids[
            0]
        print(
            f"test_block_ids: {test_block_ids}, type: {type(test_block_ids)}")
        print(f"test_new_ids: {test_new_ids}, type: {type(test_new_ids)}")
        assert isinstance(test_block_ids, list)
        assert isinstance(test_new_ids, list)

        # Execute the update
        result = model_runner._update_states(mock_scheduler_output)

        # Verify the results
        assert isinstance(result, bool)
        assert mock_req_state.num_computed_tokens == 3
        # Verify that the block_ids were extended correctly
        assert len(inner_list) == 6  # Original 3 + 3 new ones
        assert list(inner_list) == [0, 1, 2, 3, 4, 5]  # Should contain all IDs

    def test_chunked_prefill(self, model_runner, mock_scheduler_output):
        # Enable chunked prefill
        model_runner.is_chunked_prefill = True

        # Setup required state
        req_id = "req1"
        model_runner.requests[req_id] = Mock(
            output_token_ids=[1],
            prompt_token_ids=[1, 2, 3],
            block_ids=[[0, 1, 2]]  # Add proper block_ids structure
        )

        # Setup mock cached request data
        mock_scheduler_output.scheduled_cached_reqs.new_block_ids = [[0, 1, 2]]

        # Test chunked prefill input preparation
        data = model_runner._prepare_chunked_prefill_inputs(
            mock_scheduler_output)
        assert hasattr(data, 'request_ids')
        assert hasattr(data, 'input_tokens')
        assert hasattr(data, 'position_ids')

    def test_process_cached_request(self, model_runner):
        req_id = "cached_req"
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0
        model_runner.requests[req_id] = Mock(output_token_ids=[1, 2, 3],
                                             prompt_token_ids=[1, 2, 3])

        # Initialize lora_manager if lora_config exists
        if model_runner.lora_config is not None:
            model_runner.lora_manager = Mock()
            model_runner.lora_manager.get_adapter_id_with_req_id.return_value = 0

        request_data = Mock()
        request_data.req_ids = [req_id]
        request_data.num_computed_tokens = [3]

        data = Mock()
        data.request_ids = []
        data.input_tokens = []
        data.position_ids = []
        data.input_block_ids = []
        data.full_context_lens = []
        data.computed_context_lens = []
        data.prefill_completion_state = []
        data.adapter_ids = []

        model_runner._process_cached_request_for_continuous_batching(
            request_data, 0, data)

        assert len(data.request_ids) == 1
        assert len(data.input_tokens) == 1

    def test_error_handling(self, model_runner):
        # Test invalid request handling
        with pytest.raises(AssertionError):
            model_runner._process_cached_request_for_continuous_batching(
                Mock(req_ids=["invalid_req"]), 0, Mock())

    @pytest.mark.parametrize("model_type", ["llava", "llama4"])
    def test_multi_modal_processing(self, model_runner, model_type):
        model_runner.model.model = Mock()
        model_runner.model.model.config.model_type = model_type
        mm_data = [{"pixel_values": torch.randn(1, 3, 224, 224)}]

        result = model_runner._process_multi_modal_data_neuron(mm_data)
        assert result is not None
        if model_type == "llava":
            # For llava, result should be a dictionary with image_sizes
            assert isinstance(result, dict)
            assert "image_sizes" in result
            assert isinstance(result["image_sizes"], torch.Tensor)

    def test_lora_support(self, model_runner):
        # Test LoRA adapter handling
        model_runner.lora_config = Mock()
        model_runner.lora_manager = Mock()

        request_data = Mock()
        request_data.req_id = "req1"
        request_data.lora_request = Mock(lora_name="adapter1")

        adapter_id = model_runner._prepare_adapter_id_in_new_request(
            request_data)
        assert adapter_id is not None

    def test_finalize_inputs(self, model_runner):
        data = Mock()
        data.input_tokens = [[1, 2, 3]]
        data.position_ids = [[0, 1, 2]]
        data.input_block_ids = [0]
        data.slot_mapping = [[0]]
        data.block_tables = [[0]]
        data.full_context_lens = [3]
        data.computed_context_lens = [0]
        data.adapter_ids = []  # Changed from [None] to []
        data.request_ids = ["req1"]

        # Disable lora config for this test
        model_runner.lora_config = None

        result = model_runner._finalize_continuous_batching_inputs(data, True)
        assert isinstance(result, ModelInputForNeuron)
        assert isinstance(result.input_tokens, torch.Tensor)
        assert isinstance(result.position_ids, torch.Tensor)

    def test_runner_initialization(self, model_runner):
        assert model_runner is not None
        assert hasattr(model_runner, 'scheduler_config')
        assert hasattr(model_runner, 'speculative_config')
        assert hasattr(model_runner, 'observability_config')
        assert hasattr(model_runner, 'device_config')
        assert model_runner.device_config.device == "cpu"

    def test_model_execution(self, model_runner, mocker):
        # Create mock input
        mock_input = ModelInputForNeuron(
            request_ids=["req1"],
            input_tokens=torch.tensor([[1, 2, 3]]),
            position_ids=torch.tensor([[0, 1, 2]]),
            input_block_ids=torch.tensor([0]),
            slot_mapping=torch.tensor([0]),
            block_tables=torch.tensor([[0]]),
            full_context_lens=torch.tensor([[3]]),
            computed_context_lens=torch.tensor([[0]]),
            sampling_params=torch.tensor([1.0]),
            multi_modal_kwargs=None,
            adapter_ids=None,
            prefill_completion_state=None)

        # Mock input batch
        model_runner.input_batch.req_ids = ["req1"]  # Match the request ID
        model_runner.input_batch.req_id_to_index = {"req1": 0}  # Add this line

        # Create actual tensor for hidden states
        mock_hidden_states = torch.randn(1, 3,
                                         32000)  # [batch, seq_len, vocab_size]

        class MockModel:

            def __init__(self):
                self.neuron_config = Mock(on_device_sampling_config=None,
                                          vocab_size=32000,
                                          is_block_kv_layout=False,
                                          is_prefix_caching=False,
                                          chunked_prefill_config=None)
                self.architecture = "LlamaForCausalLM"
                self.num_key_value_heads = 32
                self.head_dim = 64

            def __call__(self, *args, **kwargs):
                return mock_hidden_states

            def forward(self, *args, **kwargs):
                return mock_hidden_states

            def sample(self, logits):
                # Create a proper SamplerOutput-like object with correct method signature
                class SamplerOutput:

                    def __init__(self):
                        self.sampled_token_ids = torch.tensor([[4]])

                    def __len__(self):
                        return 1

                return SamplerOutput()

        # Replace the model
        model_runner.model = MockModel()

        # Execute model
        output = model_runner._execute_model_for_text(mock_input, None)

        # Verify execution
        assert output is not None
        assert len(output) == 1
        assert torch.equal(output.sampled_token_ids, torch.tensor([[4]]))

    def test_get_kv_cache_spec(self, model_runner):
        spec = model_runner.get_kv_cache_spec()
        assert "layer" in spec
        assert spec["layer"].block_size == model_runner.block_size
        assert spec[
            "layer"].num_kv_heads == model_runner.model.num_key_value_heads
        assert spec["layer"].head_size == model_runner.model.head_dim

    def test_sampling_params(self, model_runner):
        input_ids = torch.tensor([[1, 2, 3]])

        # Setup the mock model's neuron config
        model_runner.model.neuron_config.on_device_sampling_config = None
        model_runner.model.neuron_config.vocab_size = 32000

        # Setup requests with sampling parameters
        model_runner.requests = {
            "req1":
            Mock(sampling_params=Mock(top_k=10, top_p=0.9, temperature=1.0))
        }

        sampling_params = model_runner.get_nxd_sampling_params(input_ids)
        # Debug prints
        print("\nDEBUG test_sampling_params:")
        print(f"sampling_params type: {type(sampling_params)}")
        print(f"sampling_params: {sampling_params}")
        if hasattr(sampling_params, 'shape'):
            print(f"sampling_params shape: {sampling_params.shape}")
        print(f"Is tensor?: {isinstance(sampling_params, torch.Tensor)}")
        if isinstance(sampling_params, MagicMock):
            print(f"MagicMock details: {sampling_params._mock_return_value}")
            print(f"MagicMock methods: {dir(sampling_params)}")

        assert sampling_params is not None
        assert isinstance(sampling_params, torch.Tensor)
        assert sampling_params.shape == torch.Size([1])
        assert torch.allclose(sampling_params,
                              torch.tensor([1.0], dtype=torch.float32))
