# SPDX-License-Identifier: Apache-2.0
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import pytest

from neuronx_vllm_plugin.worker.neuron_worker import NeuronWorker


# Mock CUDA-related modules
class MockWorkerBase:

    def __init__(self,
                 vllm_config,
                 local_rank,
                 rank,
                 distributed_init_method,
                 is_driver_worker=False):
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.scheduler_config = vllm_config.scheduler_config
        self.lora_config = vllm_config.lora_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker


# Create vllm mock structure
vllm_mock = MagicMock()


# Mock utils functions
def mock_init_cached_hf_modules():
    pass


# Create vllm structure
vllm_mock.config = MagicMock()
vllm_mock.config.VllmConfig = MagicMock()

vllm_mock.distributed = MagicMock()
vllm_mock.distributed.init_distributed_environment = MagicMock()
vllm_mock.distributed.ensure_model_parallel_initialized = MagicMock()

vllm_mock.logger = MagicMock()
vllm_mock.logger.init_logger = MagicMock(return_value=MagicMock())

vllm_mock.utils = MagicMock()
vllm_mock.utils.init_cached_hf_modules = mock_init_cached_hf_modules

worker_module = MagicMock()
worker_module.WorkerBase = MockWorkerBase
vllm_mock.worker = MagicMock()
vllm_mock.worker.worker = worker_module

vllm_mock.v1 = MagicMock()
vllm_mock.v1.worker = MagicMock()
vllm_mock.v1.worker.worker_base = MagicMock()
vllm_mock.v1.worker.worker_base.WorkerBase = MockWorkerBase


class MockLoRARequest:

    def __init__(self, lora_name=None):
        self.lora_name = lora_name


class MockSchedulerOutput:

    def __init__(self):
        pass


lora_module = MagicMock()
lora_module.request = MagicMock()
lora_module.request.LoRARequest = MockLoRARequest
vllm_mock.lora = lora_module
model_executor_module = MagicMock()
model_executor_module.set_random_seed = MagicMock()
vllm_mock.model_executor = model_executor_module
core_module = MagicMock()
sched_module = MagicMock()
output_module = MagicMock()
output_module.SchedulerOutput = MockSchedulerOutput
vllm_mock.v1.core = core_module
vllm_mock.v1.core.sched = sched_module
vllm_mock.v1.core.sched.output = output_module


# Mock v1.kv_cache_interface module
class MockKVCacheConfig:

    def __init__(self):
        pass


class MockKVCacheSpec:

    def __init__(self):
        pass


kv_cache_interface_module = MagicMock()
kv_cache_interface_module.KVCacheConfig = MockKVCacheConfig
kv_cache_interface_module.KVCacheSpec = MockKVCacheSpec

vllm_mock.v1.kv_cache_interface = kv_cache_interface_module


# Mock v1.outputs module
class MockModelRunnerOutput:

    def __init__(self):
        pass


outputs_module = MagicMock()
outputs_module.ModelRunnerOutput = MockModelRunnerOutput

vllm_mock.v1.outputs = outputs_module

# Install vllm mock modules
sys.modules['vllm'] = vllm_mock
sys.modules['vllm.config'] = vllm_mock.config
sys.modules['vllm.distributed'] = vllm_mock.distributed
sys.modules['vllm.logger'] = vllm_mock.logger
sys.modules['vllm.utils'] = vllm_mock.utils
sys.modules['vllm.worker'] = vllm_mock.worker
sys.modules['vllm.worker.worker'] = worker_module
sys.modules['vllm.v1'] = vllm_mock.v1
sys.modules['vllm.v1.worker'] = vllm_mock.v1.worker
sys.modules['vllm.v1.worker.worker_base'] = vllm_mock.v1.worker.worker_base
sys.modules['vllm.lora'] = lora_module
sys.modules['vllm.lora.request'] = lora_module.request
sys.modules['vllm.model_executor'] = model_executor_module
sys.modules['vllm.v1.core'] = core_module
sys.modules['vllm.v1.core.sched'] = sched_module
sys.modules['vllm.v1.core.sched.output'] = output_module
sys.modules['vllm.v1.kv_cache_interface'] = kv_cache_interface_module
sys.modules['vllm.v1.outputs'] = outputs_module

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


@dataclass
class MockVllmConfig:
    model_config: Mock
    cache_config: Mock
    device_config: Mock
    parallel_config: Mock
    scheduler_config: Mock = None
    load_config: Mock = None
    lora_config: Mock = None
    speculative_config: Mock = None
    observability_config: Mock = None
    kv_transfer_config: Mock = None
    compilation_config: Mock = None


class TestNeuronWorker:

    @pytest.fixture
    def vllm_config(self):
        parallel_config = Mock()
        parallel_config.rank = 0
        parallel_config.world_size = 1
        parallel_config.worker_cls = "auto"
        parallel_config.data_parallel_size = 1

        # Create lora_config mock
        lora_config = Mock()
        lora_config.lora_module_names = []
        lora_config.lora_model_dir = None
        lora_config.max_lora_rank = None
        lora_config.max_cpu_loras = None
        lora_config.max_gpu_loras = None
        lora_config.max_num_seqs = 32
        lora_config.max_num_batched_tokens = 4096

        # Create speculative_config mock
        speculative_config = Mock()
        speculative_config.enabled = False
        speculative_config.model_config = None
        speculative_config.cache_config = None
        speculative_config.device_config = None

        # Create observability_config mock
        observability_config = Mock()
        observability_config.enable_metrics = False
        observability_config.metrics_port = None

        # Create kv_transfer_config mock
        kv_transfer_config = Mock()
        kv_transfer_config.enabled = False
        kv_transfer_config.mode = None

        # Create compilation_config mock
        compilation_config = Mock()
        compilation_config.enable_compilation = False
        compilation_config.cache_dir = None
        compilation_config.force_recompile = False

        return MockVllmConfig(model_config=Mock(trust_remote_code=True,
                                                seed=42,
                                                max_model_len=2048),
                              cache_config=Mock(block_size=8,
                                                enable_prefix_caching=False),
                              device_config=Mock(device="cpu"),
                              parallel_config=parallel_config,
                              scheduler_config=Mock(
                                  scheduler_cls="auto",
                                  chunked_prefill_enabled=True),
                              load_config=Mock(),
                              lora_config=lora_config,
                              speculative_config=speculative_config,
                              observability_config=observability_config,
                              kv_transfer_config=kv_transfer_config,
                              compilation_config=compilation_config)

    @pytest.fixture
    def worker(self, vllm_config, mocker):
        mocker.patch('vllm.distributed.init_distributed_environment')
        mocker.patch('vllm.distributed.ensure_model_parallel_initialized')

        # Mock model runner
        mock_model_runner = Mock()
        mocker.patch(
            'neuronx_vllm_plugin.worker.neuron_worker.NeuronWorker.get_neuronx_distributed_model_runner',
            return_value=mock_model_runner)

        worker = NeuronWorker(vllm_config=vllm_config,
                              local_rank=0,
                              rank=0,
                              distributed_init_method="tcp://localhost:1234",
                              is_driver_worker=True)
        return worker

    def test_worker_initialization(self, worker):
        """Test basic worker initialization and configuration.

        This test verifies that:
        1. Worker is properly instantiated
        2. Required attributes are present
        3. Device configuration is correct

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        assert worker is not None
        assert hasattr(worker, 'model_runner')
        assert hasattr(worker, 'model_config')
        assert worker.device == "cpu"

    def test_worker_methods(self, worker):
        """Test presence and accessibility of required worker methods.

        This test verifies that:
        1. All required methods are present
        2. Methods are properly inherited
        3. Core functionality methods exist

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        assert hasattr(worker, 'load_model')
        assert hasattr(worker, 'execute_model')
        assert hasattr(worker, 'init_device')
        assert hasattr(worker, 'initialize_cache')

    def test_determine_available_memory(self, worker):
        """Test worker's memory determination functionality.

        This test verifies that:
        1. Memory calculation is performed correctly
        2. Expected default memory size is returned
        3. Memory size matches device configuration

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        memory = worker.determine_available_memory()
        assert memory == 1024 * 1024 * 1024  # 1GB

    def test_execute_model(self, worker):
        """Test model execution functionality.

        This test verifies that:
        1. Driver worker correctly executes model and returns output
        2. Non-driver worker returns None
        3. Model runner is called with correct parameters

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        mock_output = Mock()
        worker.model_runner.execute_model.return_value = mock_output

        # Test driver worker
        output = worker.execute_model(Mock())
        assert output == mock_output

        # Test non-driver worker
        worker.is_driver_worker = False
        output = worker.execute_model(Mock())
        assert output is None

    def test_initialize_cache(self, worker):
        """Test cache initialization configuration.

        This test verifies that:
        1. Cache blocks are correctly allocated
        2. GPU and CPU block counts are properly set
        3. Cache configuration is updated

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        worker.initialize_cache(num_gpu_blocks=10, num_cpu_blocks=20)
        assert worker.cache_config.num_gpu_blocks == 10
        assert worker.cache_config.num_cpu_blocks == 20

    def test_load_model(self, worker):
        """Test model loading process.

        This test verifies that:
        1. Model loading is triggered
        2. Model runner's load_model method is called
        3. Loading process completes successfully

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        worker.load_model()
        worker.model_runner.load_model.assert_called_once()

    def test_get_kv_cache_spec(self, worker):
        """Test KV cache specification retrieval.

        This test verifies that:
        1. KV cache specification is correctly returned
        2. Model runner's get_kv_cache_spec is called
        3. Specification format is correct

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        mock_spec = {'key': 'value'}
        worker.model_runner.get_kv_cache_spec.return_value = mock_spec
        result = worker.get_kv_cache_spec()
        assert result == mock_spec

    def test_initialize_from_config(self, worker):
        """Test worker initialization from configuration.

        This test verifies that:
        1. Configuration is properly applied
        2. KV cache is initialized with config
        3. Model runner receives correct configuration

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        mock_config = Mock()
        worker.initialize_from_config(mock_config)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            mock_config)

    def test_check_health(self, worker):
        """Test worker health check functionality.

        This test verifies that:
        1. Health check returns expected result
        2. No errors are raised during check
        3. Worker state is valid

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        assert worker.check_health() is None

    def test_init_distributed_environment(self, worker, mocker):
        """Test distributed environment initialization.

        This test verifies that:
        1. Distributed environment is properly initialized
        2. Correct parameters are passed to initialization
        3. Model parallel settings are properly configured
        4. Backend selection is correct

        Args:
            worker: Fixture providing configured NeuronWorker instance
            mocker: PyTest mocker fixture
        """
        # Patch at the correct import path
        mock_init = mocker.patch(
            'neuronx_vllm_plugin.worker.neuron_worker.init_distributed_environment'
        )
        mock_ensure = mocker.patch(
            'neuronx_vllm_plugin.worker.neuron_worker.ensure_model_parallel_initialized'
        )

        # Mock get_current_vllm_config to return a config with proper parallel settings
        mock_config = Mock()
        mock_config.parallel_config = Mock()
        mock_config.parallel_config.data_parallel_size = 1
        mock_config.parallel_config.tensor_parallel_size = 1
        mock_config.parallel_config.pipeline_parallel_size = 1
        mocker.patch('vllm.config.get_current_vllm_config',
                     return_value=mock_config)

        # Call the method
        worker.init_distributed_environment()

        # Verify the calls
        mock_init.assert_called_once_with(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://localhost:1234",
            backend="gloo")
        mock_ensure.assert_called_once_with(1, 1)

        # Verify that the distributed environment was initialized correctly
        assert worker.parallel_config.data_parallel_size == 1

    def test_lora_operations(self, worker):
        """Test LoRA adapter operations.

        This test verifies that:
        1. LoRA adapters can be added and removed
        2. Adapter pinning works correctly
        3. Adapter listing returns correct set
        4. All LoRA operations call appropriate model runner methods

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        # Test add_lora
        mock_request = Mock()
        worker.add_lora(mock_request)
        worker.model_runner.add_lora.assert_called_once_with(mock_request)

        # Test remove_lora
        worker.remove_lora(1)
        worker.model_runner.remove_lora.assert_called_once_with(1)

        # Test pin_lora
        worker.pin_lora(1)
        worker.model_runner.pin_lora.assert_called_once_with(1)

        # Test list_loras
        mock_loras = {1, 2, 3}
        worker.model_runner.list_loras.return_value = mock_loras
        result = worker.list_loras()
        assert result == mock_loras

    def test_unsupported_operations(self, worker):
        """Test handling of unsupported operations.

        This test verifies that:
        1. Unsupported methods raise NotImplementedError
        2. Profile method is properly blocked
        3. Get model method is properly blocked

        Args:
            worker: Fixture providing configured NeuronWorker instance
        """
        with pytest.raises(NotImplementedError):
            worker.profile()

        with pytest.raises(NotImplementedError):
            worker.get_model()
