# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for NeuronAsyncScheduler unit tests."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.v1.request import RequestStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Request Factory
# =============================================================================


def create_mock_request(
    request_id: str = "req_0",
    num_prompt_tokens: int = 100,
    num_computed_tokens: int = 0,
    num_tokens: int | None = None,
    is_prefill: bool | None = None,
    max_tokens: int = 1,
) -> Mock:
    """Create a Mock request object with essential attributes.

    Uses Mock so any undefined attributes are auto-handled. ``max_tokens`` is a
    real int (a bare Mock would raise on the ``request.max_tokens > 1`` comparison
    the send-lane producer guard makes); defaults to 1, the disagg-producer value.
    """
    if num_tokens is None:
        num_tokens = num_prompt_tokens

    mock = Mock()
    mock.request_id = request_id
    mock.num_prompt_tokens = num_prompt_tokens
    mock.num_computed_tokens = num_computed_tokens
    mock.num_tokens = num_tokens
    mock.max_tokens = max_tokens

    # Computed property
    if is_prefill is None:
        mock.is_prefill = num_computed_tokens < num_prompt_tokens
    else:
        mock.is_prefill = is_prefill

    # Essential defaults
    mock.num_output_placeholders = 0
    mock.spec_token_ids = []
    mock.arrival_time = 0.0
    mock.priority = 0
    # vLLM 0.27: base Scheduler reads these in the schedule loop; real
    # Request initializes them to 0/False/0 (vllm/v1/request.py). Without
    # explicit values Mock auto-attrs are truthy and break comparisons.
    mock.num_stale_output_tokens = 0
    mock.drop_stale_output = False
    mock.num_in_flight_tokens = 0
    mock.num_preemptions = 0
    # Real Request holds a PrefillStats(); a plain Mock is a faithful truthy
    # stand-in (the scheduler only mutates its fields here).
    mock.prefill_stats = Mock()
    mock.status = RequestStatus.WAITING
    mock.has_encoder_inputs = False
    # vLLM 0.29: NewRequestData.from_request pipes mm_features through
    # strip_covered_mm_data, which iterates it. Real Request holds an empty
    # list for text-only requests; a bare Mock auto-attr is not iterable.
    mock.mm_features = []
    mock.num_cached_tokens = 0
    mock.prompt_token_ids = []
    mock.use_structured_output = False
    mock.num_tokens_with_spec = num_tokens
    mock.next_decode_eligible_step = 0

    # Methods
    mock.is_finished.return_value = False
    mock.record_event.return_value = None

    return mock


# Alias for backward compatibility
MockRequest = create_mock_request


# =============================================================================
# Mock SchedulerOutput Class
# =============================================================================


@dataclass
class MockSchedulerOutput:
    """Mock SchedulerOutput for testing."""

    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    num_scheduled_tokens_padded: dict[str, int] | None = None
    total_num_scheduled_tokens: int = 0
    has_structured_output_requests: bool = False
    scheduled_new_reqs: list[Any] = field(default_factory=list)
    scheduled_cached_reqs: Any = None
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    kv_connector_metadata: Any = None


# =============================================================================
# Mock AsyncScheduler Base Class
# =============================================================================


class MockAsyncScheduler:
    """Mock AsyncScheduler base class for testing NeuronAsyncScheduler."""

    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size: int,
        mm_registry=None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.structured_output_manager = structured_output_manager
        self.block_size = block_size
        self.mm_registry = mm_registry
        self.include_finished_set = include_finished_set
        self.log_stats = log_stats

        # Scheduler state
        self.waiting: deque[MockRequest] = deque()
        self.running: list[MockRequest] = []
        self.requests: dict[str, MockRequest] = {}

        # Configuration from vllm_config
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

    def schedule(self) -> MockSchedulerOutput:
        """Base schedule method - returns empty output."""
        output = MockSchedulerOutput()

        # Process waiting requests (simplified)
        while self.waiting:
            req = self.waiting.popleft()
            self.running.append(req)
            tokens = req.num_prompt_tokens - req.num_computed_tokens
            output.num_scheduled_tokens[req.request_id] = tokens
            output.total_num_scheduled_tokens += tokens

        # Process running requests (decode)
        for req in self.running:
            if req.request_id not in output.num_scheduled_tokens:
                output.num_scheduled_tokens[req.request_id] = 1
                output.total_num_scheduled_tokens += 1

        return output

    def _update_after_schedule(self, scheduler_output: MockSchedulerOutput) -> None:
        """Update state after scheduling (from AsyncScheduler)."""
        pass


# =============================================================================
# Mock Configuration Classes
# =============================================================================


@dataclass
class MockSchedulerConfig:
    """Mock scheduler configuration."""

    max_num_seqs: int = 8
    max_num_batched_tokens: int = 2048
    max_num_scheduled_tokens: int = 2048
    max_model_len: int = 2048
    policy: str = "fcfs"
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    # Read by Scheduler.__init__. Defaults match upstream SchedulerConfig.
    long_prefill_token_threshold_adaptive: bool = False
    max_num_active_seqs: int | None = None
    scheduler_reserve_full_isl: bool = False
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    max_num_encoder_input_tokens: int = 0
    disable_chunked_mm_input: bool = False
    disable_hybrid_kv_cache_manager: bool = False
    async_scheduling: bool = False
    encoder_cache_size: int = 0
    stream_interval: int = 0
    watermark: float = 0.0


def _make_mock_observability_config(**overrides):
    """Create a MagicMock observability_config with upstream defaults.

    Attributes the upstream Scheduler.__init__ reads are set to the values
    ObservabilityConfig declares, so a MagicMock's truthiness cannot flip a
    metrics path on.
    """
    from unittest.mock import MagicMock

    cfg = MagicMock()
    defaults = {
        "kv_cache_metrics": False,
        "kv_cache_metrics_sample": 0.01,
        "enable_mfu_metrics": False,
        "per_request_spec_decode_metrics": "none",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_model_config(**overrides):
    """Create a MagicMock model_config that auto-handles new upstream attributes.

    Uses MagicMock so any attribute added by future vLLM versions will
    return a MagicMock instead of raising AttributeError. Attributes that
    are validated or compared by the upstream Scheduler.__init__ are set
    to sensible defaults.
    """
    from unittest.mock import MagicMock

    cfg = MagicMock()
    defaults = {
        "max_model_len": 2048,
        "is_encoder_decoder": False,
        "is_multimodal_model": False,
        # Scheduler.__init__ reads this ModelConfig property; a MagicMock
        # attribute is truthy.
        "supports_multimodal_inputs": False,
        "uses_mrope": False,
        # vLLM 0.29 Scheduler.__init__ reads these flags; a MagicMock is truthy.
        "enable_return_routed_experts": False,
        "return_sampling_mask": False,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_cache_config(**overrides):
    """Create a MagicMock cache_config.

    Attributes that are validated/compared in upstream Scheduler.__init__
    are set to sensible defaults.
    """
    import torch
    from unittest.mock import MagicMock

    cfg = MagicMock()
    defaults = {
        "block_size": 32,
        "num_gpu_blocks": 1000,
        "enable_prefix_caching": False,
        # Use torch.dtype directly to avoid string validation cascades
        # (cache_dtype="auto" requires model_config.dtype to also be valid)
        "cache_dtype": torch.bfloat16,
        "mamba_cache_mode": "none",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_vllm_config(**overrides):
    """Create a mock VllmConfig with sensible defaults.

    Returns a simple namespace-like object with all known attributes set.
    Unknown attributes return None (not Mock) to avoid arithmetic errors
    like 'Mock * Mock'.
    """

    class _MockVllmConfig:
        """Mock VllmConfig that returns None for unknown attributes."""

        def __getattr__(self, name):
            # Return None for any attribute not explicitly set.
            # This avoids MagicMock arithmetic errors (Mock * Mock)
            # and AttributeError for new upstream attributes.
            return None

    cfg = _MockVllmConfig()
    defaults = {
        "scheduler_config": MockSchedulerConfig(),
        "model_config": _make_mock_model_config(),
        "cache_config": _make_mock_cache_config(),
        "additional_config": {},
        "lora_config": None,
        "kv_events_config": None,
        "parallel_config": None,
        "speculative_config": None,
        "num_speculative_tokens": 0,
        # Scheduler.__init__ reads vllm_config.num_lookahead_tokens (0 when
        # speculative_config is None) and feeds it into allocate_slots
        # arithmetic, so None from __getattr__ breaks.
        "num_lookahead_tokens": 0,
        # Scheduler.__init__ reads num_prefill_lookahead_tokens and
        # aux_output_config.enabled; None breaks both.
        "num_prefill_lookahead_tokens": 0,
        "aux_output_config": Mock(enabled=False),
        "kv_transfer_config": None,
        "ec_transfer_config": None,
        "observability_config": None,
        # vLLM 0.27: Scheduler.__init__ calls
        # vllm_config.ec_manager_config.get_encoder_cache_manager_obj();
        # returning None selects the default EncoderCacheManager.
        "ec_manager_config": Mock(
            get_encoder_cache_manager_obj=Mock(return_value=None)
        ),
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        object.__setattr__(cfg, k, v)
    return cfg


# Alias for backward compatibility — test files import this name directly
MockVllmConfig = _make_mock_vllm_config


@dataclass
class SchedulerMockKVCacheConfig:
    """Mock KV cache configuration for scheduler tests.

    This is a simplified mock used directly in scheduler tests,
    separate from the vllm.v1.kv_cache_interface.KVCacheConfig mock.
    """

    num_blocks: int = 1000
    block_size: int = 32
    kv_cache_groups: list = field(default_factory=list)
    has_mamba_layers: bool = False
    needs_kv_cache_zeroing: bool = False
    # Added in 0.29 and read unconditionally: KVCacheCoordinator.__init__ reads
    # prefix_cache_retention_interval, and the engine core publishes the resolved
    # kv_cache_layout to every worker. Defaults match upstream KVCacheConfig.
    prefix_cache_retention_interval: int | None = None
    kv_cache_layout: str | None = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_request_factory():
    """Factory fixture for creating mock requests."""

    def _create_request(
        request_id: str = "req_0",
        num_prompt_tokens: int = 100,
        num_computed_tokens: int = 0,
    ) -> MockRequest:
        return MockRequest(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            num_tokens=num_prompt_tokens,
        )

    return _create_request


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig with default settings."""
    config = _make_mock_vllm_config()
    config.parallel_config = Mock(
        data_parallel_rank=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    config.observability_config = _make_mock_observability_config()
    return config


@pytest.fixture
def mock_vllm_config_with_buckets():
    """Create a mock VllmConfig with custom bucket configuration."""
    config = _make_mock_vllm_config()
    config.scheduler_config.max_num_batched_tokens = 1024
    config.scheduler_config.max_num_scheduled_tokens = 1024
    config.scheduler_config.max_num_seqs = 8
    config.additional_config = {
        "neuron_config": {
            "num_batched_tokens_buckets": [128, 512, 1024],
        }
    }
    config.parallel_config = Mock(
        data_parallel_rank=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    config.observability_config = _make_mock_observability_config()
    return config


@pytest.fixture
def scheduler_with_max_model_len(mock_kv_cache_config):
    """Factory fixture to create scheduler with custom max_model_len."""

    def _create_scheduler(max_model_len: int):
        # Create config with specific max_model_len
        from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

        mock_config = Mock()
        mock_config.model_config = Mock()
        mock_config.model_config.max_model_len = max_model_len
        mock_config.model_config.is_encoder_decoder = False
        mock_config.model_config.is_multimodal_model = (
            False  # Skip multimodal processing
        )
        mock_config.scheduler_config = Mock()
        mock_config.scheduler_config.max_num_seqs = 256
        mock_config.scheduler_config.max_num_batched_tokens = max_model_len
        mock_config.scheduler_config.max_num_scheduled_tokens = max_model_len
        mock_config.scheduler_config.enable_chunked_prefill = False
        mock_config.scheduler_config.policy = (
            "fcfs"  # First Come First Serve scheduling policy
        )
        # Scheduler.__init__ reads these; a bare Mock is truthy.
        mock_config.scheduler_config.max_num_active_seqs = None
        mock_config.scheduler_config.long_prefill_token_threshold_adaptive = False
        mock_config.model_config.supports_multimodal_inputs = False
        mock_config.num_prefill_lookahead_tokens = 0
        mock_config.aux_output_config = Mock(enabled=False)
        mock_config.cache_config = Mock()
        mock_config.cache_config.num_gpu_blocks = 100  # Parent scheduler checks this
        mock_config.lora_config = None
        mock_config.kv_events_config = None
        mock_config.kv_transfer_config = None
        mock_config.ec_transfer_config = None
        mock_config.additional_config = {}  # For neuron_config
        mock_config.parallel_config = Mock()
        mock_config.parallel_config.decode_context_parallel_size = 1
        mock_config.parallel_config.prefill_context_parallel_size = 1
        mock_config.parallel_config.pipeline_parallel_size = 1
        mock_config.observability_config = _make_mock_observability_config()
        mock_config.observability_config.kv_cache_metrics = False
        mock_config.observability_config.kv_cache_metrics_sample = 0.01

        scheduler = NeuronAsyncScheduler(
            mock_config,
            mock_kv_cache_config,
            Mock(),  # structured_output_manager
            block_size=32,
        )

        return scheduler

    return _create_scheduler


@pytest.fixture
def mock_kv_cache_config():
    """Create a mock KV cache configuration."""
    return SchedulerMockKVCacheConfig()


@pytest.fixture
def mock_structured_output_manager():
    """Create a mock structured output manager."""
    return Mock()


@pytest.fixture
def neuron_scheduler(
    mock_vllm_config, mock_kv_cache_config, mock_structured_output_manager
):
    """Create a NeuronAsyncScheduler instance with mocked dependencies.

    This fixture patches the AsyncScheduler base class to use our mock,
    allowing unit testing without the full vLLM stack.
    """
    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    scheduler = NeuronAsyncScheduler(
        vllm_config=mock_vllm_config,
        kv_cache_config=mock_kv_cache_config,
        structured_output_manager=mock_structured_output_manager,
        block_size=32,
        log_stats=False,
    )
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.running = []
    return scheduler


@pytest.fixture
def neuron_scheduler_with_buckets(
    mock_vllm_config_with_buckets, mock_kv_cache_config, mock_structured_output_manager
):
    """Create a NeuronAsyncScheduler with custom bucket configuration."""
    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    scheduler = NeuronAsyncScheduler(
        vllm_config=mock_vllm_config_with_buckets,
        kv_cache_config=mock_kv_cache_config,
        structured_output_manager=mock_structured_output_manager,
        block_size=32,
        log_stats=True,
    )
    return scheduler


@pytest.fixture
def prefill_request(mock_request_factory):
    """Create a prefill request (num_computed_tokens < num_prompt_tokens)."""
    return mock_request_factory(
        request_id="prefill_req",
        num_prompt_tokens=200,
        num_computed_tokens=0,
    )


@pytest.fixture
def decode_request(mock_request_factory):
    """Create a decode request (num_computed_tokens >= num_prompt_tokens)."""
    req = mock_request_factory(
        request_id="decode_req",
        num_prompt_tokens=100,
        num_computed_tokens=100,
    )
    return req
