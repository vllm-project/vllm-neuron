# test/unit/conftest.py
import sys
import pytest
from unittest.mock import Mock
from collections import deque

@pytest.fixture
def mock_config():
    return Mock(
        model_config=Mock(
            trust_remote_code=True,
            seed=42,
            max_model_len=2048
        ),
        cache_config=Mock(block_size=8),
        parallel_config=Mock(rank=0),
        device_config=Mock(device="cpu"),
        load_config=Mock()
    )

@pytest.fixture
def mock_model():
    return Mock()

# Add these new fixtures for scheduler tests
class MockBaseScheduler:
    def __init__(self, *args, **kwargs):
        self.waiting = deque()
        self.running = []

    def schedule(self):
        return Mock(name="SchedulerOutput")

# Mock vllm scheduler
mock_vllm_scheduler = Mock()
mock_vllm_scheduler.Scheduler = MockBaseScheduler

# Add to sys.modules before any imports
sys.modules['vllm.v1.core.sched.scheduler'] = mock_vllm_scheduler

@pytest.fixture
def scheduler_config():
    return Mock(
        max_model_len=2048,
        max_num_seqs=32,
        max_num_batched_tokens=4096
    )

@pytest.fixture
def cache_config():
    return Mock(block_size=8)

@pytest.fixture
def scheduler(scheduler_config, cache_config):
    """Create a properly initialized scheduler instance"""
    from neuronx_vllm_plugin.core.scheduler import ContinuousBatchingNeuronScheduler
    scheduler = ContinuousBatchingNeuronScheduler(
        scheduler_config=scheduler_config,
        cache_config=cache_config
    )
    # Set required attributes
    scheduler.scheduler_config = scheduler_config
    scheduler.max_num_running_reqs = scheduler_config.max_num_seqs
    scheduler.waiting = deque()
    scheduler.running = []
    scheduler.holdback_queue = deque()
    return scheduler