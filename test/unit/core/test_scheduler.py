# test/unit/core/test_scheduler.py
import pytest
from neuronx_vllm_plugin.core.scheduler import ContinuousBatchingNeuronScheduler
from unittest.mock import Mock

class TestNeuronScheduler:
    @pytest.fixture
    def scheduler_config(self):
        return Mock(
            max_model_len=2048,
            max_num_seqs=32,
            max_num_batched_tokens=4096
        )

    @pytest.fixture
    def cache_config(self):
        return Mock(block_size=8)

    @pytest.fixture
    def scheduler(self, scheduler_config, cache_config):
        # Initialize with config objects
        scheduler = ContinuousBatchingNeuronScheduler(
            scheduler_config=scheduler_config,
            cache_config=cache_config
        )
        return scheduler