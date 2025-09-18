# SPDX-License-Identifier: Apache-2.0
# test/unit/conftest.py
import logging
import sys
from collections import deque
from unittest.mock import MagicMock, Mock

import pytest

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Setting up mock modules...")

# Create base mock module for neuronx_distributed_inference
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

# Install neuronx mock modules
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


# Mock vLLM modules and classes
class MockRequest:

    def __init__(self, *args, **kwargs):
        self.prompt_len = kwargs.get('prompt_len', 0)
        self.max_output_len = kwargs.get('max_output_len', 0)
        self.arrival_time = kwargs.get('arrival_time', 0)


# Create mock modules
mock_request = MagicMock()
mock_request.Request = MockRequest


class MockBaseScheduler:

    def __init__(self, *args, **kwargs):
        self.waiting = deque()
        self.running = []

    def schedule(self):
        return Mock(name="SchedulerOutput")


# Mock vllm scheduler
mock_vllm_scheduler = Mock()
mock_vllm_scheduler.Scheduler = MockBaseScheduler

# Add vLLM mocks to sys.modules
sys.modules['vllm.v1.core.sched.scheduler'] = mock_vllm_scheduler
sys.modules['vllm.v1.request'] = mock_request

logger.debug("Mock modules setup complete")


@pytest.fixture
def mock_neuronx_model():
    """Fixture for mocked neuronx model"""
    model = MagicMock()
    model.config = MagicMock()
    model.config.model_type = "llama"
    return model


@pytest.fixture
def mock_sampling_params():
    """Fixture for mocked sampling parameters"""
    return MagicMock()


@pytest.fixture
def mock_config():
    return Mock(model_config=Mock(trust_remote_code=True,
                                  seed=42,
                                  max_model_len=2048),
                cache_config=Mock(block_size=8),
                parallel_config=Mock(rank=0),
                device_config=Mock(device="cpu"),
                load_config=Mock())


@pytest.fixture
def mock_model():
    return Mock()


@pytest.fixture
def scheduler_config():
    return Mock(max_model_len=2048,
                max_num_seqs=32,
                max_num_batched_tokens=4096)


@pytest.fixture
def cache_config():
    return Mock(block_size=8)


@pytest.fixture
def scheduler(scheduler_config, cache_config):
    """Create a properly initialized scheduler instance"""
    from neuronx_vllm_plugin.core.scheduler import \
        ContinuousBatchingNeuronScheduler
    scheduler = ContinuousBatchingNeuronScheduler(
        scheduler_config=scheduler_config, cache_config=cache_config)
    # Set required attributes
    scheduler.scheduler_config = scheduler_config
    scheduler.max_num_running_reqs = scheduler_config.max_num_seqs
    scheduler.waiting = deque()
    scheduler.running = []
    scheduler.holdback_queue = deque()
    return scheduler
