# test/unit/worker/test_neuron_worker.py
import pytest
from unittest.mock import Mock
from dataclasses import dataclass
import torch
from neuronx_vllm_plugin.worker.neuron_worker import NeuronWorker

@dataclass
class MockVllmConfig:
    model_config: Mock
    cache_config: Mock
    device_config: Mock
    parallel_config: Mock
    load_config: Mock = None

class TestNeuronWorker:
    @pytest.fixture
    def vllm_config(self):
        return MockVllmConfig(
            model_config=Mock(
                trust_remote_code=True,
                seed=42,
                max_model_len=2048
            ),
            cache_config=Mock(),
            device_config=Mock(device="cpu"),
            parallel_config=Mock(rank=0),
            load_config=Mock()
        )

    @pytest.fixture
    def worker(self, vllm_config, mocker):
        # Mock WorkerBase initialization
        def mock_init(self, *args, **kwargs):
            self.parallel_config = vllm_config.parallel_config
            self.model_config = vllm_config.model_config  # Set model_config
            self.cache_config = vllm_config.cache_config
            self.device_config = vllm_config.device_config
        
        mocker.patch('vllm.worker.worker.WorkerBase.__init__', mock_init)
        mocker.patch('vllm.distributed.init_distributed_environment')
        mocker.patch('vllm.distributed.ensure_model_parallel_initialized')
        
        # Mock other required methods
        mocker.patch('neuronx_vllm_plugin.worker.neuron_worker.NeuronWorker.get_neuronx_distributed_model_runner',
                    return_value=Mock())
        
        worker = NeuronWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:1234",
            is_driver_worker=True
        )
        return worker

    def test_worker_initialization(self, worker):
        assert worker is not None
        assert hasattr(worker, 'model_runner')
        assert hasattr(worker, 'model_config')

    def test_worker_methods(self, worker):
        assert hasattr(worker, 'load_model')