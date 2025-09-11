# test/unit/worker/test_neuron_worker.py
import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
import torch
from neuronx_vllm_plugin.worker.neuron_worker import NeuronWorker


@dataclass
class MockVllmConfig:
    model_config: Mock
    cache_config: Mock
    device_config: Mock
    parallel_config: Mock
    scheduler_config: Mock = None
    load_config: Mock = None


class TestNeuronWorker:

    @pytest.fixture
    def vllm_config(self):
        parallel_config = Mock()
        parallel_config.rank = 0
        parallel_config.world_size = 1
        parallel_config.worker_cls = "auto"
        # Set data_parallel_size as an integer
        parallel_config.data_parallel_size = 1

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
                              load_config=Mock())

    @pytest.fixture
    def worker(self, vllm_config, mocker):
        # Mock WorkerBase initialization
        def mock_init(self, *args, **kwargs):
            self.parallel_config = vllm_config.parallel_config
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.device_config = vllm_config.device_config
            self.scheduler_config = vllm_config.scheduler_config

        # Mock vLLM config
        mock_vllm_config = Mock()
        mock_vllm_config.model_config = vllm_config.model_config
        mock_vllm_config.cache_config = vllm_config.cache_config
        mock_vllm_config.parallel_config = vllm_config.parallel_config
        mock_vllm_config.scheduler_config = vllm_config.scheduler_config

        # Set up the get_current_vllm_config mock
        mocker.patch('vllm.config.get_current_vllm_config',
                     return_value=mock_vllm_config)
        mocker.patch('vllm.worker.worker.WorkerBase.__init__', mock_init)
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
        """Test basic worker initialization"""
        assert worker is not None
        assert hasattr(worker, 'model_runner')
        assert hasattr(worker, 'model_config')
        assert worker.device == "cpu"

    def test_worker_methods(self, worker):
        """Test presence of required methods"""
        assert hasattr(worker, 'load_model')
        assert hasattr(worker, 'execute_model')
        assert hasattr(worker, 'init_device')
        assert hasattr(worker, 'initialize_cache')

    def test_init_device(self, worker, mocker):
        """Test device initialization"""
        mock_set_seed = mocker.patch('vllm.model_executor.set_random_seed')
        mock_init_env = mocker.patch.object(worker,
                                            'init_distributed_environment')

        worker.init_device()

    def test_determine_available_memory(self, worker):
        """Test memory determination"""
        memory = worker.determine_available_memory()
        assert memory == 1024 * 1024 * 1024  # 1GB

    def test_execute_model(self, worker):
        """Test model execution"""
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
        """Test cache initialization"""
        worker.initialize_cache(num_gpu_blocks=10, num_cpu_blocks=20)
        assert worker.cache_config.num_gpu_blocks == 10
        assert worker.cache_config.num_cpu_blocks == 20

    def test_load_model(self, worker):
        """Test model loading"""
        worker.load_model()
        worker.model_runner.load_model.assert_called_once()

    def test_get_kv_cache_spec(self, worker):
        """Test getting KV cache spec"""
        mock_spec = {'key': 'value'}
        worker.model_runner.get_kv_cache_spec.return_value = mock_spec
        result = worker.get_kv_cache_spec()
        assert result == mock_spec

    def test_initialize_from_config(self, worker):
        """Test initialization from config"""
        mock_config = Mock()
        worker.initialize_from_config(mock_config)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            mock_config)

    def test_check_health(self, worker):
        """Test health check"""
        assert worker.check_health() is None

    def test_init_distributed_environment(self, worker, mocker):
        """Test distributed environment initialization"""
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
        """Test LoRA-related operations"""
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
        """Test operations that raise NotImplementedError"""
        with pytest.raises(NotImplementedError):
            worker.profile()

        with pytest.raises(NotImplementedError):
            worker.get_model()
