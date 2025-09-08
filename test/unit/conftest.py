# test/unit/conftest.py
import pytest
from unittest.mock import Mock

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