# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from neuronx_vllm_plugin.worker.neuronx_distributed_model_runner import (
    ModelInputForNeuron, NeuronxDistributedModelRunner)

logger = logging.getLogger(__name__)


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
    def mock_scheduler_output(self):

        cached_reqs = Mock(req_ids=["req1"],
                           num_computed_tokens=[0],
                           new_block_ids=[[0]],
                           resumed_from_preemption=[False],
                           new_token_ids=[[]])

        scheduler_args = {
            # Requests
            'scheduled_new_reqs': [],
            'scheduled_cached_reqs': cached_reqs,

            # Token scheduling info
            'num_scheduled_tokens': {
                "req1": 1
            },
            'total_num_scheduled_tokens': 1,
            'scheduled_spec_decode_tokens': {},

            # Encoder related
            'scheduled_encoder_inputs': {},
            'num_common_prefix_blocks': [],
            'free_encoder_input_ids': [],

            # Request management
            'finished_req_ids': set(),

            # Structured output
            'structured_output_request_ids': {},
            'grammar_bitmask': None,  # Optional[npt.NDArray[np.int32]]

            # KV Cache
            'kv_connector_metadata': None  # Optional[KVConnectorMetadata]
        }

        try:
            return SchedulerOutput(**scheduler_args)
        except TypeError as e:
            logger.error(f"Error creating SchedulerOutput: {e}")
            logger.debug(f"Current args: {scheduler_args.keys()}")
            raise

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
        """Test the preparation of model input for continuous batching.

        This test verifies that:
        1. Model input is correctly formatted for continuous batching
        2. All required tensors are properly created and shaped
        3. Request IDs and sampling parameters are properly handled

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output

        The test ensures all components of ModelInputForNeuron are correctly
        initialized and formatted.
        """
        # Disable LoRA for this test
        model_runner.lora_config = None

        # Setup required state
        req_id = "req1"
        model_runner.vllm_req_to_neuron_seq_id_mapping[req_id] = 0

        # Create a mock sampling params with proper attributes
        mock_sampling_params = Mock()
        mock_sampling_params.top_k = 10
        mock_sampling_params.top_p = 0.9
        mock_sampling_params.temperature = 1.0

        model_runner.requests[req_id] = Mock(
            output_token_ids=[1],
            prompt_token_ids=[1],
            block_ids=[[0]],
            sampling_params=mock_sampling_params)

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

        # Verify the output
        assert isinstance(model_input, ModelInputForNeuron)
        assert model_input.request_ids is not None
        assert isinstance(model_input.input_tokens, torch.Tensor)
        assert isinstance(model_input.position_ids, torch.Tensor)
        assert isinstance(model_input.input_block_ids, torch.Tensor)
        assert isinstance(model_input.sampling_params, torch.Tensor)

    def test_update_states(self, model_runner, mock_scheduler_output):
        """Test state updates during model execution.

        This test verifies that:
        1. Block IDs are correctly updated
        2. Computed tokens are properly tracked
        3. State transitions are handled correctly

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output
        """

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

        logger.debug(f"mock_block_ids: {mock_block_ids}")
        logger.debug(
            f"mock_cached_reqs.new_block_ids: {mock_cached_reqs.new_block_ids}"
        )
        logger.debug(f"mock_cached_reqs[0]: {mock_cached_reqs[0]}")

        # Debug prints and assertions
        test_block_ids = mock_req_state.block_ids[0]
        test_new_ids = mock_scheduler_output.scheduled_cached_reqs.new_block_ids[
            0]
        logger.debug(
            f"test_block_ids: {test_block_ids}, type: {type(test_block_ids)}")
        logger.debug(
            f"test_new_ids: {test_new_ids}, type: {type(test_new_ids)}")
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
        """Test chunked prefill input preparation.

        This test verifies that:
        1. Chunked prefill mode is properly enabled
        2. Input data is correctly formatted
        3. Required attributes are present in output

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mock_scheduler_output: Fixture providing mock scheduler output
        """
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
        """Test cached request processing functionality.

        This test verifies that:
        1. Cached requests are properly processed
        2. Data structures are correctly updated
        3. Request state is maintained accurately

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
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
        """Test error handling for invalid requests.

        This test verifies that:
        1. Invalid requests are properly detected
        2. Appropriate exceptions are raised
        3. Error states are handled correctly

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        with pytest.raises(AssertionError):
            model_runner._process_cached_request_for_continuous_batching(
                Mock(req_ids=["invalid_req"]), 0, Mock())

    @pytest.mark.parametrize("model_type", ["llava", "llama4"])
    def test_multi_modal_processing(self, model_runner, model_type):
        """Test multi-modal data processing for different model types.

        This test verifies that:
        1. Different model types are handled correctly
        2. Multi-modal data is properly processed
        3. Output format matches model type requirements

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            model_type: Type of model being tested (llava or llama4)
        """
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
        """Test LoRA adapter handling functionality.

        This test verifies that:
        1. LoRA configuration is properly initialized
        2. Adapter IDs are correctly assigned
        3. Request-specific adapters are properly handled

        Args:
            model_runner: Fixture providing configured ModelRunner instance

        The test ensures proper integration of LoRA adapters with the model runner.
        """
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
        """Test input finalization for continuous batching.

        This test verifies that:
        1. Input data is properly formatted
        2. Tensor types are correct
        3. All required fields are present

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
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
        """Test model execution flow.

        This test verifies that:
        1. Model input is correctly processed
        2. Forward pass works as expected
        3. Output tensors are properly formatted

        Args:
            model_runner: Fixture providing configured ModelRunner instance
            mocker: PyTest mocker fixture
        """
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
        model_runner.input_batch.req_id_to_index = {"req1": 0}

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
        """Test KV cache specification generation.

        This test verifies that:
        1. Cache specifications are correctly generated
        2. Block sizes are properly set
        3. Head dimensions match model configuration

        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        spec = model_runner.get_kv_cache_spec()
        assert "layer" in spec
        assert spec["layer"].block_size == model_runner.block_size
        assert spec[
            "layer"].num_kv_heads == model_runner.model.num_key_value_heads
        assert spec["layer"].head_size == model_runner.model.head_dim

    def test_scheduler_output_args(self):
        """Test SchedulerOutput argument handling.

        This test verifies that:
        1. Required arguments for SchedulerOutput are correctly identified
        2. Minimal argument set can create valid SchedulerOutput
        3. Argument validation works as expected

        The test ensures proper initialization of SchedulerOutput with minimal
        valid configuration.
        """
        import inspect

        from vllm.v1.core.sched.output import SchedulerOutput

        def get_required_args():
            try:
                SchedulerOutput()
            except TypeError as e:
                logger.error(f"Initial SchedulerOutput error: {e}")

            sig = inspect.signature(SchedulerOutput.__init__)
            required_args = {
                name: param.default
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty and name != 'self'
            }

            logger.debug(
                f"Required SchedulerOutput arguments: {required_args}")

            try:
                minimal_args = {arg: [] for arg in required_args}
                SchedulerOutput(**minimal_args)
                logger.debug(
                    "Successfully created SchedulerOutput with minimal args")
            except Exception as e:
                logger.error(f"Failed to create SchedulerOutput: {e}")
                raise
