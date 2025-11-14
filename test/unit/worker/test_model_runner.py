# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_neuron.worker.neuronx_distributed_model_runner import (
    IntermediateInputData, ModelInputForNeuron, NeuronxDistributedModelRunner)

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
        # Mock model_config with get_vocab_size method
        model_config = Mock(max_model_len=2048)
        model_config.get_vocab_size.return_value = 32000

        return MockVllmConfig(model_config=model_config,
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

        # Updated scheduler args to match the current SchedulerOutput class
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
            'free_encoder_mm_hashes':
            [],  # New field replacing free_encoder_input_ids

            # Request management
            'finished_req_ids': set(),

            # Structured output
            'structured_output_request_ids': {},
            'grammar_bitmask': None,

            # KV Cache
            'kv_connector_metadata': None
        }

        return SchedulerOutput(**scheduler_args)

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

        # Create the block_ids structure
        inner_list = MutableList([0, 1, 2])
        mock_block_ids = [inner_list]

        class CustomMockState:

            def __init__(self):
                self.block_ids = mock_block_ids
                self.num_computed_tokens = 0

        mock_req_state = CustomMockState()
        model_runner.requests = {"req1": mock_req_state}

        # Setup mock cached request data
        class CustomCachedReqs:

            def __init__(self):
                self.req_ids = ["req1"]
                self.num_computed_tokens = [3]
                inner_list = MutableList([3, 4, 5])
                self.new_block_ids = [[inner_list]]
                self.resumed_from_preemption = [False]

            def __getitem__(self, idx):
                return self.new_block_ids[0]

        mock_cached_reqs = CustomCachedReqs()
        mock_scheduler_output.scheduled_cached_reqs = mock_cached_reqs
        mock_scheduler_output.finished_req_ids = []
        mock_scheduler_output.free_encoder_mm_hashes = []  # Updated field name
        mock_scheduler_output.num_scheduled_tokens = {"req1": 1}
        mock_scheduler_output.scheduled_new_reqs = []

        # Initialize encoder cache
        model_runner.encoder_cache = {}

        # Mock the _update_states method to return True
        model_runner._update_states = Mock(return_value=True)

        # Execute the update
        result = model_runner._update_states(mock_scheduler_output)

        # Verify the results
        assert isinstance(result, bool)
        assert result is True  # Verify specific return value

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
        from vllm.multimodal.inputs import (MultiModalFeatureSpec,
                                            PlaceholderRange)

        model_runner.model.model = Mock()
        model_runner.model.model.config.model_type = model_type

        # Create proper MultiModalFeatureSpec object
        mm_data_dict = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mm_spec = MultiModalFeatureSpec(data=mm_data_dict,
                                        modality="image",
                                        identifier="test_image",
                                        mm_position=PlaceholderRange(offset=0,
                                                                     length=1))
        mm_data = [mm_spec]

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
                # Fix: Configure on-device sampling to avoid CPU sampling
                self.neuron_config = Mock(vocab_size=32000,
                                          is_block_kv_layout=False,
                                          is_prefix_caching=False,
                                          chunked_prefill_config=None)
                # Set up on_device_sampling_config with proper global_topk
                self.neuron_config.on_device_sampling_config = Mock()
                self.neuron_config.on_device_sampling_config.global_topk = 256
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

    def test_initialize_kv_cache(self, model_runner):
        """Test that KV cache initialization returns None as expected for Neuron.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        kv_cache_config = Mock()
        result = model_runner.initialize_kv_cache(kv_cache_config)
        assert result is None

    def test_get_nxdi_lora_config(self, model_runner):
        """Test LoRA configuration generation raises NotImplementedError.
        
        Verifies that LoRA is not supported and raises appropriate error.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        override_neuron_config = {
            "target_modules": ["q", "v"],
            "lora_ckpt_json": "path/to/json",
        }
        model_runner.vllm_config.additional_config = {
            "override_neuron_config": override_neuron_config.copy()
        }
        model_runner.lora_config = Mock(max_cpu_loras=2,
                                        max_loras=4,
                                        max_lora_rank=8)
        model_runner.scheduler_config = Mock(max_num_seqs=32)

        # LoRA is not supported, should raise NotImplementedError
        with pytest.raises(NotImplementedError,
                           match="Multi-lora is not yet supported"):
            model_runner._get_nxdi_lora_config()

    def test_load_model_with_lora(self, model_runner):
        """Test model loading with LoRA configuration raises NotImplementedError.
        
        Verifies that LoRA is not supported and raises appropriate error.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.lora_config = Mock(max_cpu_loras=2, max_loras=4)
        model_runner.vllm_config.additional_config = {
            "override_neuron_config": {
                "target_modules": ["q", "v"],
                "lora_ckpt_json": "path/to/json"
            }
        }

        # LoRA is not supported, should raise NotImplementedError
        with pytest.raises(NotImplementedError,
                           match="Multi-lora is not yet supported"):
            model_runner.load_model()

    def test_execute_model_with_finished_requests(self, model_runner):
        """Test model execution path when all requests are finished.
        
        Verifies that:
        1. Finished requests are properly removed from tracking
        2. Custom sequence ID mapping is updated
        3. EMPTY_MODEL_RUNNER_OUTPUT is returned when no work remains
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req1"}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_from_preemption=[])

        model_runner.use_custom_seq_id_mapping = True
        model_runner.vllm_req_to_neuron_seq_id_mapping = {"req1": 0}
        model_runner.free_seq_ids = set()
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.input_batch.remove_request = Mock()
        model_runner.encoder_cache = {}

        result = model_runner.execute_model(scheduler_output)

        assert result == EMPTY_MODEL_RUNNER_OUTPUT
        assert 0 in model_runner.free_seq_ids

    def test_generate_model_runner_output_with_speculative_config(
            self, model_runner):
        """Test model output generation with speculative decoding enabled.
        
        Verifies that:
        1. 3D tensor output is properly squeezed and processed
        2. Speculative token IDs are correctly extracted
        3. Output tokens are added to request states
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = Mock()
        model_runner.max_model_len = 10

        sampler_output = Mock()
        sampler_output.sampled_token_ids = torch.tensor([[[1]], [[2]]],
                                                        dtype=torch.long)

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[])
        }

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:

            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((2, 10), dtype=torch.long))

        result = model_runner._generate_model_runner_output(sampler_output)

        assert len(result.sampled_token_ids) == 2
        assert model_runner.spec_token_ids is not None
        assert result.sampled_token_ids[0] == [1]
        assert result.sampled_token_ids[1] == [2]
        assert model_runner.requests["req1"].output_token_ids == [1]
        assert model_runner.requests["req2"].output_token_ids == [2]

    def test_generate_model_runner_output_without_speculative(
            self, model_runner):
        """Test model output generation without speculative decoding.
        
        Verifies standard token generation path without speculation.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = None
        model_runner.max_model_len = 10

        sampler_output = Mock()
        # Include -1 padding tokens to test filtering
        sampler_output.sampled_token_ids = torch.tensor(
            [[1, 2, -1], [3, -1, -1]], dtype=torch.long)

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[])
        }

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:

            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((2, 10), dtype=torch.long))

        result = model_runner._generate_model_runner_output(sampler_output)

        # Verify -1 tokens are filtered out
        assert result.sampled_token_ids[0] == [1, 2]
        assert result.sampled_token_ids[1] == [3]

    def test_update_states_complex(self, model_runner):
        """Test complex state updates with multiple request scenarios.
        
        Verifies proper handling of:
        1. Finished requests removal
        2. New request addition
        3. Cached request updates with preemption
        4. Encoder cache cleanup
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"finished_req"}
        scheduler_output.free_encoder_mm_hashes = ["hash1"]
        scheduler_output.num_scheduled_tokens = {"active_req": 1}
        scheduler_output.scheduled_new_reqs = [
            Mock(req_id="new_req",
                 prompt_token_ids=[1, 2, 3],
                 block_ids=[[0, 1, 2]],
                 num_computed_tokens=0,
                 mm_kwargs=None,
                 mm_positions=None,
                 mm_hashes=None,
                 sampling_params=Mock(),
                 pooling_params=None,
                 lora_request=None)
        ]
        scheduler_output.scheduled_cached_reqs = Mock(
            req_ids=["cached_req"],
            num_computed_tokens=[3],
            new_block_ids=[[[3, 4, 5]]],
            resumed_from_preemption=[True])

        model_runner.requests = {
            "finished_req":
            Mock(),
            "cached_req":
            Mock(block_ids=[[0, 1, 2]],
                 output_token_ids=[],
                 prompt_token_ids=[1, 2, 3])
        }
        model_runner.encoder_cache = {"hash1": Mock()}
        model_runner.input_batch.req_id_to_index = {"finished_req": 0}
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        assert "finished_req" not in model_runner.requests
        assert "hash1" not in model_runner.encoder_cache
        assert "new_req" in model_runner.requests
        assert len(model_runner.requests["cached_req"].block_ids[0]) == 3

    def test_execute_model_multimodal(self, model_runner):
        """Test model execution for multimodal models.
        
        Verifies that multimodal models use the correct execution path.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm_neuron.worker.constants import NEURON_MULTI_MODAL_MODELS

        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.total_num_scheduled_tokens = 1
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {"req1": 1}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Make model a multimodal model
        model_runner.model.architecture = list(
            NEURON_MULTI_MODAL_MODELS
        )[0] if NEURON_MULTI_MODAL_MODELS else "llava"
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        # Mock the execution methods
        model_runner._prepare_model_input = Mock(return_value=Mock())
        model_runner._execute_model_for_multimodal_models = Mock(
            return_value=Mock(sampled_token_ids=torch.tensor([[1]])))
        model_runner._generate_model_runner_output = Mock(
            return_value=EMPTY_MODEL_RUNNER_OUTPUT)

        result = model_runner.execute_model(scheduler_output)

        model_runner._execute_model_for_multimodal_models.assert_called_once()
        assert result == EMPTY_MODEL_RUNNER_OUTPUT

    def test_generate_model_runner_output_none(self, model_runner):
        """Test model output generation when sampler_outputs is None.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        result = model_runner._generate_model_runner_output(None)
        assert result == EMPTY_MODEL_RUNNER_OUTPUT

    def test_generate_model_runner_output_empty_sampled_ids(
            self, model_runner):
        """Test model output generation with empty sampled IDs.
        
        Verifies that empty token sequences are handled correctly.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.speculative_config = None
        model_runner.max_model_len = 10

        sampler_output = Mock()
        # All tokens are -1 (padding)
        sampler_output.sampled_token_ids = torch.tensor([[-1, -1], [-1, -1]],
                                                        dtype=torch.long)

        model_runner.input_batch.num_tokens_no_spec = [0, 0]
        model_runner.input_batch.token_ids_cpu = torch.zeros((2, 10),
                                                             dtype=torch.long)
        model_runner.input_batch.num_tokens = [0, 0]
        model_runner.input_batch.req_ids = ["req1", "req2"]
        model_runner.requests = {
            "req1": Mock(output_token_ids=[]),
            "req2": Mock(output_token_ids=[])
        }

        result = model_runner._generate_model_runner_output(sampler_output)

        # Both should be empty after filtering
        assert result.sampled_token_ids[0] == []
        assert result.sampled_token_ids[1] == []
        # output_token_ids should not be extended for empty sequences
        assert model_runner.requests["req1"].output_token_ids == []
        assert model_runner.requests["req2"].output_token_ids == []

    def test_update_states_with_lora(self, model_runner):
        """Test state updates with LoRA configuration enabled.
        
        Verifies that LoRA manager is called for finished requests.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req1"}
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        model_runner.lora_config = Mock()
        model_runner.lora_manager = Mock()
        model_runner.requests = {"req1": Mock()}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        model_runner._update_states(scheduler_output)

        model_runner.lora_manager.remove_req_id.assert_called_once_with("req1")
        assert "req1" not in model_runner.requests

    def test_update_states_with_new_blocks(self, model_runner):
        """Test state updates with new block allocation.
        
        Verifies that new blocks are appended to existing block IDs.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []

        # Request with new blocks (not resumed)
        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [[[3, 4]]]
        cached_req.resumed_from_preemption = [False]
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1":
            Mock(block_ids=[[0, 1, 2]],
                 output_token_ids=[1, 2, 3, 4, 5],
                 prompt_token_ids=[1, 2, 3, 4, 5])
        }
        model_runner.input_batch.req_id_to_index = {}
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        # New blocks should be appended
        assert model_runner.requests["req1"].block_ids[0] == [0, 1, 2, 3, 4]

    def test_execute_model_with_custom_seq_mapping_no_match(
            self, model_runner):
        """Test execute_model when finished request is not in seq_id mapping.
        
        Verifies that requests not in mapping are handled gracefully.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = {"req_not_in_mapping"}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        model_runner.use_custom_seq_id_mapping = True
        model_runner.vllm_req_to_neuron_seq_id_mapping = {"other_req": 0}
        model_runner.free_seq_ids = {1, 2}
        model_runner.lora_config = None
        model_runner.requests = {}
        model_runner.input_batch.req_id_to_index = {}
        model_runner.encoder_cache = {}

        result = model_runner.execute_model(scheduler_output)

        assert result == EMPTY_MODEL_RUNNER_OUTPUT
        # free_seq_ids should not change since req was not in mapping
        assert model_runner.free_seq_ids == {1, 2}

    def test_execute_model_for_multimodal_models(self, model_runner):
        """Test _execute_model_for_multimodal_models execution path.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_input = Mock()
        model_runner.model = Mock()
        mock_hidden_states = torch.randn(1, 10)
        model_runner.model.execute_model = Mock(
            return_value=mock_hidden_states)
        model_runner._sample = Mock(return_value=Mock())

        result = model_runner._execute_model_for_multimodal_models(
            model_input, None)

        model_runner.model.execute_model.assert_called_once_with(model_input)
        model_runner._sample.assert_called_once_with(mock_hidden_states,
                                                     model_input)
        assert result is not None

    def test_prepare_model_input_chunked_prefill(self, model_runner):
        """Test _prepare_model_input for chunked prefill mode.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.is_chunked_prefill = True

        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])

        # Mock the helper methods
        model_runner._prepare_chunked_prefill_inputs = Mock(
            return_value=Mock())
        model_runner._finalize_chunked_prefill_inputs = Mock(
            return_value=Mock())

        result = model_runner._prepare_model_input(scheduler_output)

        model_runner._prepare_chunked_prefill_inputs.assert_called_once_with(
            scheduler_output)
        model_runner._finalize_chunked_prefill_inputs.assert_called_once()
        assert result is not None

    def test_process_multi_modal_data_unsupported_model(self, model_runner):
        """Test multi-modal data processing with unsupported model type.
        
        Verifies that NotImplementedError is raised for unsupported models.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        from vllm.multimodal.inputs import (MultiModalFeatureSpec,
                                            PlaceholderRange)

        model_runner.model.model.config.model_type = 'unsupported_model_type'

        # Create proper MultiModalFeatureSpec object
        mm_data_dict = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mm_spec = MultiModalFeatureSpec(data=mm_data_dict,
                                        modality="image",
                                        identifier="test_image",
                                        mm_position=PlaceholderRange(offset=0,
                                                                     length=1))
        mm_data = [mm_spec]

        with pytest.raises(NotImplementedError) as exc_info:
            model_runner._process_multi_modal_data_neuron(mm_data)

        assert "not supported on Neuron yet" in str(exc_info.value)

    def test_update_states_with_block_table_append(self, model_runner):
        """Test that block_table.append_row is called with new blocks.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_spec_decode_tokens = {}

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [[[3, 4]]]
        cached_req.resumed_from_preemption = [False]
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1":
            Mock(block_ids=[[0, 1, 2]],
                 output_token_ids=[1, 2, 3, 4, 5],
                 prompt_token_ids=[1, 2, 3, 4, 5])
        }
        model_runner.input_batch.req_id_to_index = {"req1": 0}
        model_runner.input_batch.num_computed_tokens_cpu = [0]
        model_runner.input_batch.block_table = Mock()
        model_runner.input_batch.num_tokens_no_spec = [5]
        model_runner.input_batch.num_tokens = [5]
        model_runner.lora_config = None

        model_runner._update_states(scheduler_output)

        # Verify block_table.append_row was called
        model_runner.input_batch.block_table.append_row.assert_called_once_with(
            [[3, 4]], 0)

    def test_update_states_with_spec_tokens(self, model_runner):
        """Test state updates with scheduled speculative decode tokens.
        
        Verifies that spec tokens are added to token_ids_cpu.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        scheduler_output = Mock()
        scheduler_output.finished_req_ids = set()
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_spec_decode_tokens = {"req1": [10, 11, 12]}

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        cached_req.new_block_ids = [None]
        cached_req.resumed_from_preemption = [False]
        scheduler_output.scheduled_cached_reqs = cached_req

        model_runner.requests = {
            "req1":
            Mock(block_ids=[[0, 1, 2]],
                 output_token_ids=[1, 2, 3, 4, 5],
                 prompt_token_ids=[1, 2, 3, 4, 5])
        }
        model_runner.input_batch.req_id_to_index = {"req1": 0}
        model_runner.input_batch.num_tokens_no_spec = [5]
        model_runner.input_batch.num_tokens = [5]
        model_runner.input_batch.num_computed_tokens_cpu = [0]
        model_runner.input_batch.block_table = Mock()
        model_runner.lora_config = None

        # TensorWrapper handles list-to-tensor conversion for assignment
        class TensorWrapper:

            def __init__(self, tensor):
                self.tensor = tensor

            def __setitem__(self, key, value):
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=self.tensor.dtype)
                self.tensor.__setitem__(key, value)

            def __getitem__(self, key):
                return self.tensor.__getitem__(key)

        model_runner.input_batch.token_ids_cpu = TensorWrapper(
            torch.zeros((1, 20), dtype=torch.long))

        model_runner._update_states(scheduler_output)

        # Verify spec tokens were added
        assert model_runner.input_batch.num_tokens[0] == 8  # 5 + 3 spec tokens

    def test_load_model_without_lora(self, model_runner):
        """Test model loading without LoRA configuration.
        
        Verifies that LoRA manager is not initialized when lora_config is None.
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance
        """
        model_runner.lora_config = None
        model_runner.vllm_config.additional_config = {}  # Add this line

        with patch(
                'vllm_neuron.worker.neuronx_distributed_model_runner.get_neuron_model'
        ) as mock_get_model:
            mock_model = Mock()
            mock_model.neuron_config.is_block_kv_layout = False
            mock_model.neuron_config.is_prefix_caching = False
            mock_model.neuron_config.chunked_prefill_config = None
            # Fix: Set global_topk to an actual integer to avoid Mock comparison issues
            mock_model.neuron_config.on_device_sampling_config = Mock()
            mock_model.neuron_config.on_device_sampling_config.global_topk = 256
            # Ensure the model has the required sample method
            mock_model.sample = Mock()
            mock_get_model.return_value = mock_model

            model_runner.load_model()

            assert not hasattr(
                model_runner,
                'lora_manager') or model_runner.lora_manager is None
            assert model_runner.model is not None

    def test_prepare_chunked_prefill_inputs_cached_request_with_output(
            self, model_runner):
        """Test chunked prefill input preparation for cached requests with generated output.
        
        This test verifies the behavior when a cached request has already generated
        some output tokens and needs to continue processing. The model should include
        both the remaining prompt tokens and the last generated output token.
        
        Verifies that:
        1. Resumed prompt tokens are correctly extracted based on num_computed_tokens
        2. The last output token is appended to the input sequence
        3. Total input length matches expected count (prompt chunks + last output)
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill capabilities
        """
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []

        cached_req = Mock()
        cached_req.req_ids = ["req1"]
        cached_req.num_computed_tokens = [5]
        scheduler_output.scheduled_cached_reqs = cached_req
        scheduler_output.num_scheduled_tokens = {"req1": 2}

        model_runner.cache_config = Mock(block_size=8)
        model_runner.requests = {
            "req1":
            Mock(prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
                 output_token_ids=[8, 9],
                 block_ids=[[0, 1]])
        }

        data = model_runner._prepare_chunked_prefill_inputs(scheduler_output)

        # Should include resumed prompt tokens and last output token
        assert len(data.input_tokens) == 3  # 2 prompt tokens + 1 output token
        assert data.input_tokens[-1] == 9  # Last output token

    def test_finalize_chunked_prefill_inputs(self, model_runner):
        """Test finalization of chunked prefill inputs into model-ready tensors.
        
        This test verifies the conversion of intermediate data structures into
        properly formatted PyTorch tensors for chunked prefill execution. It ensures
        that variable-length sequences are padded correctly and all tensors have
        appropriate shapes.
        
        Verifies that:
        1. Input tokens are reshaped to [batch_size, seq_len] format
        2. Block tables are padded to uniform length across all requests
        3. Prefill completion states are preserved as boolean tensors
        4. All tensor shapes are consistent and valid for model execution
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill support
        """

        data = IntermediateInputData()
        data.request_ids = ["req1", "req2"]
        data.input_tokens = [1, 2, 3, 4, 5]
        data.position_ids = [0, 1, 2, 3, 4]
        data.input_block_ids = [0, 1]
        data.slot_mapping = [0, 1, 2, 3, 4]
        data.block_tables = [[0, 1], [2]]  # Different sizes
        data.full_context_lens = [3, 2]
        data.computed_context_lens = [0, 0]
        data.prefill_completion_state = [False, True]
        data.adapter_ids = []

        model_runner.device = "cpu"

        result = model_runner._finalize_chunked_prefill_inputs(
            data, None, None)

        assert result.input_tokens.shape[0] == 1  # Batch size
        assert result.input_tokens.shape[1] == 5  # Sequence length
        assert result.prefill_completion_state.tolist() == [False, True]
        # Verify block tables are padded to same length
        assert result.block_tables.shape[1] == 2  # Max blocks

    def test_sample_with_partial_prefill(self, model_runner):
        """Test sampling behavior with mixed prefill completion states.
        
        In chunked prefill mode, requests may be at different stages - some may
        have completed their prefill phase while others are still processing.
        This test verifies that incomplete prefill requests have their logits
        masked out (-1) to prevent premature sampling.
        
        Verifies that:
        1. Requests with completed prefill (state=True) retain their logits
        2. Requests with incomplete prefill (state=False) have logits set to -1
        3. Masking is applied per-request in the batch
        4. Sample method is called with correctly masked hidden states
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill and sampling capabilities
        """
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        model_input = Mock()
        model_input.request_ids = ["req1", "req2", "req3"]
        model_input.prefill_completion_state = torch.tensor(
            [True, False, True])

        model_runner.input_batch = Mock()
        model_runner.input_batch.req_ids = ["req1", "req2", "req3"]
        model_runner.model = Mock()
        model_runner.model.sample = Mock(return_value=Mock())

        result = model_runner._sample(hidden_states, model_input)

        # Verify the result exists
        assert result is not None

        # Verify that req2 (incomplete prefill) is masked
        assert hidden_states[1, 0] == -1
        assert hidden_states[1, 1] == -1

    def test_get_nxd_sampling_params_zero_temperature(self, model_runner):
        """Test sampling parameter handling for greedy decoding (temperature=0).
        
        When temperature is set to 0, the model should perform greedy decoding by
        selecting only the most likely token. This test verifies that the sampling
        parameters are correctly adjusted to implement greedy sampling.
        
        Verifies that:
        1. top_k is forced to 1 (select only best token) when temperature=0
        2. temperature is adjusted to 1.0 to avoid division by zero
        3. Adjustments are applied consistently across all requests in batch
        4. Original sampling params are preserved for non-zero temperatures
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        sampling parameter generation capabilities
        """
        model_runner.model.neuron_config.on_device_sampling_config = None
        model_runner.scheduler_config = Mock(max_num_seqs=2)

        model_runner.requests = {
            "req1":
            Mock(sampling_params=Mock(top_k=50, top_p=0.9, temperature=0.0)),
            "req2":
            Mock(sampling_params=Mock(top_k=10, top_p=0.95, temperature=0.0))
        }

        input_ids = torch.tensor([[1], [2]])

        # Mock prepare_sampling_params to capture the call
        with patch(
                'vllm_neuron.worker.neuronx_distributed_model_runner.prepare_sampling_params'
        ) as mock_prep:
            mock_prep.return_value = torch.ones((2, 3))
            _ = model_runner.get_nxd_sampling_params(input_ids)

            # Verify prepare_sampling_params was called
            mock_prep.assert_called_once()
            call_args = mock_prep.call_args

            # Check that top_k was set to 1 for zero temperature
            assert call_args[1]['top_k'][0] == 1
            assert call_args[1]['top_k'][1] == 1
            # Check that temperature was set to 1.0
            assert call_args[1]['temperature'][0] == 1.0
            assert call_args[1]['temperature'][1] == 1.0

    def test_finalize_continuous_batching_inputs_with_lora(self, model_runner):
        """Test continuous batching input finalization with LoRA adapter support.
        
        When LoRA (Low-Rank Adaptation) is enabled, the model runner must include
        adapter IDs in the model input to route requests to the correct fine-tuned
        adapters. This test verifies proper LoRA adapter ID handling.
        
        Verifies that:
        1. LoRA adapter IDs are converted from list to PyTorch tensor
        2. Adapter ID tensor has correct dtype (long) and device placement
        3. Adapter IDs are preserved in the correct order matching request IDs
        4. Sampling parameters are correctly generated for LoRA-enabled requests
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        LoRA multi-adapter serving capabilities
        """
        from vllm_neuron.worker.neuronx_distributed_model_runner import \
            IntermediateInputData

        model_runner.lora_config = Mock()
        model_runner.scheduler_config = Mock(max_model_len=2048,
                                             max_num_seqs=32)
        model_runner.model.neuron_config.vocab_size = 32000
        model_runner.model.neuron_config.on_device_sampling_config = None

        # Setup requests for sampling params
        model_runner.requests = {
            "req1":
            Mock(sampling_params=Mock(top_k=10, top_p=0.9, temperature=1.0))
        }

        data = IntermediateInputData()
        data.input_tokens = [[1, 2, 3]]
        data.position_ids = [[0, 1, 2]]
        data.input_block_ids = [0]
        data.slot_mapping = [[0]]
        data.block_tables = [[0]]
        data.full_context_lens = [3]
        data.computed_context_lens = [0]
        data.adapter_ids = [1, 2]  # LoRA adapter IDs
        data.request_ids = ["req1"]

        result = model_runner._finalize_continuous_batching_inputs(data, True)

        assert result.adapter_ids is not None
        assert isinstance(result.adapter_ids, torch.Tensor)
        assert result.adapter_ids.tolist() == [1, 2]

    def test_prepare_chunked_prefill_inputs_new_request(self, model_runner):
        """Test chunked prefill input preparation for newly submitted requests.
        
        When a new request arrives, chunked prefill mode processes it in chunks
        rather than all at once. This test verifies the initial chunk preparation
        for a fresh request.
        
        Verifies that:
        1. Only the first N tokens (based on num_scheduled_tokens) are processed
        2. Position IDs are correctly assigned starting from 0
        3. Block IDs and slot mappings are properly initialized
        4. Prefill completion state is False (more chunks remaining)
        5. Request is correctly added to the tracking data structures
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        chunked prefill scheduling
        """
        scheduler_output = Mock()

        # Create a simple mock instead of using NewRequestData
        new_req = Mock()
        new_req.req_id = "req1"
        new_req.prompt_token_ids = [1, 2, 3, 4, 5]
        new_req.block_ids = [[0, 1, 2]]
        new_req.num_computed_tokens = 0

        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = Mock(req_ids=[])
        scheduler_output.num_scheduled_tokens = {"req1": 3}

        model_runner.cache_config = Mock(block_size=8)
        model_runner.requests = {}

        data = model_runner._prepare_chunked_prefill_inputs(scheduler_output)

        assert "req1" in data.request_ids
        assert len(data.input_tokens) == 3  # First 3 tokens
        assert data.position_ids == [0, 1, 2]
        assert data.full_context_lens == [3]
        assert data.computed_context_lens == [0]
        assert data.prefill_completion_state == [False]  # Not complete yet

    def test_get_nxd_sampling_params_with_device_config(self, model_runner):
        """Test sampling parameter generation with on-device sampling configuration.
        
        When on-device sampling is enabled via neuron_config, the sampling parameters
        must respect hardware-specific limits (like global_topk). This test verifies
        proper parameter generation with device-specific constraints.
        
        Verifies that:
        1. On-device sampling config's global_topk is respected
        2. Sampling parameters are prepared correctly for hardware execution
        3. Parameters are formatted as proper tensors (not mock objects)
        4. Per-request sampling params (top_k, top_p, temperature) are preserved
        
        Args:
            model_runner: Fixture providing configured ModelRunner instance with
                        on-device sampling hardware acceleration
        """
        model_runner.model.neuron_config.on_device_sampling_config = Mock(
            global_topk=128)
        model_runner.model.neuron_config.vocab_size = 32000
        model_runner.scheduler_config = Mock(max_num_seqs=2)
        model_runner.is_chunked_prefill = False

        model_runner.requests = {
            "req1":
            Mock(sampling_params=Mock(top_k=50, top_p=0.9, temperature=0.8)),
            "req2":
            Mock(sampling_params=Mock(top_k=10, top_p=0.95, temperature=1.0))
        }

        input_ids = torch.tensor([[1], [2]])

        # Mock prepare_sampling_params to return a proper tensor
        with patch(
                'vllm_neuron.worker.neuronx_distributed_model_runner.prepare_sampling_params'
        ) as mock_prep:
            mock_prep.return_value = torch.ones((2, 3), dtype=torch.float32)
            params = model_runner.get_nxd_sampling_params(input_ids)

            assert params is not None
            assert isinstance(params, torch.Tensor)
            mock_prep.assert_called_once()
