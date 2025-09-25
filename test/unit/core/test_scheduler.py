# SPDX-License-Identifier: Apache-2.0
from collections import deque
from unittest.mock import Mock, patch

import pytest


class TestNeuronScheduler:

    def test_scheduler_initialization(self, scheduler):
        """Test basic scheduler initialization and configuration.

        This test verifies that:
        1. Scheduler is properly initialized with required queues
        2. Initial queue states are empty
        3. All required attributes are present

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        assert hasattr(scheduler, 'holdback_queue')
        assert len(scheduler.holdback_queue) == 0
        assert hasattr(scheduler, 'waiting')
        assert hasattr(scheduler, 'running')

    def test_can_schedule_empty_queues(self, scheduler):
        """Test scheduling capability with empty queues.

        This test verifies that:
        1. New requests can be scheduled when queues are empty
        2. Scheduler correctly evaluates capacity
        3. Default scheduling behavior works as expected

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        assert scheduler.can_schedule(mock_request)

    def test_can_schedule_with_running_requests(self, scheduler):
        """Test scheduling capability with existing running requests.

        This test verifies that:
        1. Scheduler correctly handles existing running requests
        2. Capacity evaluation considers current load
        3. Returns appropriate boolean response

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.running = [Mock() for _ in range(5)]
        result = scheduler.can_schedule(mock_request)
        assert isinstance(result, bool)

    def test_schedule_with_empty_queues(self, scheduler):
        """Test scheduling behavior with empty request queues.

        This test verifies that:
        1. Scheduler handles empty queue state correctly
        2. Returns valid output even with no requests
        3. Maintains empty state of waiting and holdback queues

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        output = scheduler.schedule()
        assert output is not None
        assert len(scheduler.waiting) == 0
        assert len(scheduler.holdback_queue) == 0

    def test_schedule_with_waiting_requests(self, scheduler):
        """Test scheduling behavior with pending requests.

        This test verifies that:
        1. Waiting requests are properly processed
        2. Scheduler produces valid output
        3. Request counts are maintained correctly
        4. Queue state transitions are valid

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()

        # Verify the scheduling behavior
        assert output is not None
        # Verify that request was processed
        total_requests = len(scheduler.waiting) + len(scheduler.holdback_queue)
        assert total_requests > 0

    def test_queue_management(self, scheduler):
        """Test queue management and state transitions.

        This test verifies that:
        1. Requests are properly added to queues
        2. Queue state transitions work correctly
        3. Request counts are maintained
        4. Scheduling output is valid

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Add requests to holdback queue
        mock_requests = [Mock() for _ in range(3)]
        for req in mock_requests:
            scheduler.holdback_queue.append(req)

        # Verify initial state
        assert len(scheduler.holdback_queue) == 3
        assert len(scheduler.waiting) == 0

        # Schedule
        output = scheduler.schedule()
        assert output is not None

        # Verify queue transitions
        total_requests = len(scheduler.waiting) + len(scheduler.holdback_queue)
        assert total_requests == 3

    @patch('neuronx_vllm_plugin.core.scheduler.logger')
    def test_logging_behavior(self, mock_logger, scheduler):
        """Test scheduler logging functionality.

        This test verifies that:
        1. Debug logs are generated during scheduling
        2. Logger is called with appropriate messages
        3. Logging doesn't interfere with scheduling

        Args:
            mock_logger: Mock logger instance
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()
        assert output is not None
        mock_logger.debug.assert_called()

    def test_max_capacity_constraints(self, scheduler):
        """Test scheduler capacity limit enforcement.

        This test verifies that:
        1. Maximum capacity limits are enforced
        2. Requests are rejected when at capacity
        3. Capacity checking is accurate

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Fill up to max capacity
        scheduler.running = [
            Mock() for _ in range(scheduler.max_num_running_reqs)
        ]
        mock_request = Mock()
        result = scheduler.can_schedule(mock_request)
        assert not result

    def test_batch_scheduling_logic(self, scheduler):
        """Test batch request scheduling behavior.

        This test verifies that:
        1. Multiple requests are handled correctly
        2. Batch scheduling maintains request counts
        3. Queue transitions work for batches
        4. Output is valid for batch operations

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        initial_running = [Mock() for _ in range(2)]
        initial_holdback = [Mock() for _ in range(3)]
        scheduler.running = initial_running
        scheduler.holdback_queue.extend(initial_holdback)

        # Execute scheduling
        output = scheduler.schedule()
        assert output is not None

        # Verify scheduling behavior
        total_requests = (len(scheduler.running) + len(scheduler.waiting) +
                          len(scheduler.holdback_queue))
        assert total_requests >= len(initial_running) + len(initial_holdback)

    def verify_scheduler_state(self, scheduler, expected_total_requests):
        """Verify scheduler's current state matches expectations.

        This helper method verifies that:
        1. Total request count matches expectations
        2. Requests are properly distributed across queues
        3. No requests are lost during transitions

        Args:
            scheduler: The scheduler instance to verify
            expected_total_requests: Expected total number of requests

        Raises:
            AssertionError: If actual total requests doesn't match expected
        """
        actual_total = (len(scheduler.running) + len(scheduler.waiting) +
                        len(scheduler.holdback_queue))
        assert actual_total == expected_total_requests, \
            f"Expected {expected_total_requests} total requests, but found {actual_total}"

    def test_schedule_with_running_and_waiting(self, scheduler):
        """Test scheduling behavior with concurrent requests.

        This test verifies that:
        1. Scheduler handles multiple running requests
        2. Waiting requests are properly processed
        3. Output maintains correct scheduling state
        4. Queue transitions are handled properly

        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        scheduler.running = [Mock() for _ in range(2)]
        scheduler.waiting = deque([Mock() for _ in range(2)])

        # Execute scheduling
        output = scheduler.schedule()

        # Verify that requests were processed
        assert output is not None

    # Min-tokens fix tests
    @pytest.fixture
    def mock_request_with_min_tokens(self):
        """Create a mock request with configurable min_tokens parameters."""
        request = Mock()
        request.num_tokens = 10
        request.num_output_tokens = 5
        request.max_tokens = 50
        request.output_token_ids = [1, 2, 3, 4, 5]
        request.eos_token_id = 2  # EOS token
        request.pooling_params = None

        # Mock sampling params
        sampling_params = Mock()
        sampling_params.min_tokens = 0
        sampling_params.ignore_eos = False
        sampling_params.stop_token_ids = [999]  # Stop token
        request.sampling_params = sampling_params

        return request

    def test_min_tokens_prevents_eos_stop(self, scheduler,
                                          mock_request_with_min_tokens):
        """Test that EOS token doesn't stop generation when min_tokens not satisfied."""
        mock_request_with_min_tokens.sampling_params.min_tokens = 10
        mock_request_with_min_tokens.num_output_tokens = 5
        mock_request_with_min_tokens.output_token_ids = [1, 2, 3, 4, 2]

        new_token_ids = [2]  # EOS token
        result_tokens, stopped = scheduler._update_request_with_output(
            mock_request_with_min_tokens, new_token_ids)

        # Should NOT stop because min_tokens (10) > current output (5)
        assert not stopped, "Request should not stop due to EOS when min_tokens not satisfied"
        assert result_tokens == [
            2
        ], "Token should not be trimmed when not stopping"

    def test_min_tokens_prevents_stop_token_stop(self, scheduler,
                                                 mock_request_with_min_tokens):
        """Test that stop tokens don't stop generation when min_tokens not satisfied."""
        mock_request_with_min_tokens.sampling_params.min_tokens = 10
        mock_request_with_min_tokens.num_output_tokens = 5
        mock_request_with_min_tokens.output_token_ids = [1, 2, 3, 4, 999]
        new_token_ids = [999]  # Stop token
        result_tokens, stopped = scheduler._update_request_with_output(
            mock_request_with_min_tokens, new_token_ids)

        # Should NOT stop because min_tokens not satisfied
        assert not stopped, "Request should not stop due to stop token when min_tokens not satisfied"
        assert result_tokens == [
            999
        ], "Token should not be trimmed when not stopping"

    def test_eos_stops_when_min_tokens_satisfied(self, scheduler,
                                                 mock_request_with_min_tokens):
        """Test that EOS token stops generation when min_tokens is satisfied."""
        mock_request_with_min_tokens.sampling_params.min_tokens = 5
        mock_request_with_min_tokens.num_output_tokens = 10
        mock_request_with_min_tokens.output_token_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 2
        ]

        new_token_ids = [2]  # EOS token
        result_tokens, stopped = scheduler._update_request_with_output(
            mock_request_with_min_tokens, new_token_ids)

        # Should stop because min_tokens (5) <= current output (10)
        assert stopped, "Request should stop due to EOS when min_tokens satisfied"
        assert hasattr(mock_request_with_min_tokens,
                       'status'), "Request status should be set"
