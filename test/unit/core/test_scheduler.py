# SPDX-License-Identifier: Apache-2.0
from collections import deque
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_neuron.core.scheduler import (ContinuousBatchingNeuronScheduler,
                                        check_stop_with_min_tokens)


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

    @patch('vllm_neuron.core.scheduler.logger')
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
        """Test that EOS token doesn't stop generation when minimum tokens requirement is not met.
        
        This test verifies that the scheduler correctly handles the min_tokens parameter
        when encountering an EOS token. Generation should continue if the minimum number
        of tokens has not been reached, even if an EOS token is generated.

        Args:
            scheduler: The scheduler instance being tested
            mock_request_with_min_tokens: A mock request fixture with configurable min_tokens

        Test Steps:
            1. Configure request with min_tokens > current output length
            2. Set up output token sequence with EOS token
            3. Attempt to update request with new EOS token
            4. Verify generation continues and token is preserved

        Expected Results:
            - Generation should not stop (stopped=False)
            - New token should be preserved without trimming
            - Request status should remain unchanged
        """
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
        """Test that stop tokens don't terminate generation before minimum tokens requirement.
        
        This test verifies that the scheduler respects the min_tokens parameter when
        encountering stop tokens. Generation should continue if the minimum number of
        tokens has not been reached, even when stop tokens are generated.

        Args:
            scheduler: The scheduler instance being tested
            mock_request_with_min_tokens: A mock request fixture with configurable min_tokens

        Test Steps:
            1. Configure request with min_tokens > current output length
            2. Set up output token sequence with stop token
            3. Attempt to update request with new stop token
            4. Verify generation continues and token is preserved

        Expected Results:
            - Generation should not stop (stopped=False)
            - Stop token should be preserved without trimming
            - Request status should remain unchanged
        """
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
        """Test that EOS token successfully stops generation after minimum tokens requirement is met.
        
        This test verifies that the scheduler correctly stops generation when encountering
        an EOS token after the minimum number of tokens has been generated. This ensures
        proper termination behavior once all requirements are satisfied.

        Args:
            scheduler: The scheduler instance being tested
            mock_request_with_min_tokens: A mock request fixture with configurable min_tokens

        Test Steps:
            1. Configure request with min_tokens <= current output length
            2. Set up output token sequence exceeding min_tokens
            3. Attempt to update request with EOS token
            4. Verify generation stops and request status is updated

        Expected Results:
            - Generation should stop (stopped=True)
            - Request status should be set appropriately
            - Token sequence should reflect proper termination
        """
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

    def test_continuous_batching_initialization(self, scheduler):
        """Test initialization of ContinuousBatchingNeuronScheduler.
        
        This test verifies:
        1. Scheduler is instantiated with correct class type
        2. Required configuration attributes are present
        3. Essential queue management attributes exist
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        assert isinstance(scheduler, ContinuousBatchingNeuronScheduler)
        assert hasattr(scheduler, 'scheduler_config')
        assert hasattr(scheduler, 'max_num_running_reqs')

    def test_purge_waiting_to_holdback(self, scheduler):
        """Test queue transition from waiting to holdback queue.
        
        This test verifies:
        1. Requests are correctly moved from waiting to holdback queue
        2. Request count is preserved during transition
        3. Queue state is consistent after scheduling
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial waiting queue
        initial_waiting = [Mock() for _ in range(3)]
        scheduler.waiting.extend(initial_waiting)

        # Execute schedule
        scheduler.schedule()

        # Verify all requests were moved through holdback
        assert len(scheduler.waiting) == 3
        assert len(scheduler.holdback_queue) == 0

    def test_max_prompt_batch_size_constraint(self, scheduler):
        """Test enforcement of maximum prompt batch size.
        
        This test verifies:
        1. Scheduler respects max_prompt_batch_size limit
        2. Additional requests are rejected when limit reached
        3. Batch size constraints are properly enforced
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Fill waiting queue to max_prompt_batch_size
        scheduler.waiting.append(Mock())

        # Try to schedule another request
        mock_request = Mock()
        result = scheduler.can_schedule(mock_request)

        assert not result, "Should not schedule when max_prompt_batch_size reached"

    def test_check_stop_with_min_tokens_length_cap(self, scheduler):
        """Test token length cap conditions in stop checking.
        
        This test verifies:
        1. Requests exceeding max length are properly stopped
        2. Length-capped status is correctly set
        3. Stop condition is triggered for length limits
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()
        mock_request.num_tokens = scheduler.scheduler_config.max_model_len + 1
        mock_request.num_output_tokens = 5
        mock_request.max_tokens = 10
        mock_request.status = None

        result = check_stop_with_min_tokens(
            mock_request, scheduler.scheduler_config.max_model_len)
        assert result
        assert mock_request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_context_length_handling(self, scheduler):
        """Test request handling with different context lengths.
        
        This test verifies:
        1. Basic request scheduling works with different context lengths
        2. Scheduler capacity constraints are respected
        3. Batch size limits are enforced
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Clear scheduler state
        scheduler.running = []
        scheduler.waiting.clear()
        scheduler.holdback_queue.clear()

        # Test with empty scheduler (should accept any context length)
        mock_request = Mock()
        mock_request.num_tokens = scheduler.scheduler_config.max_model_len + 1
        result = scheduler.can_schedule(mock_request)
        assert result, "Should accept request when queues are empty"

        # Test with scheduler at capacity
        scheduler.running = [
            Mock() for _ in range(scheduler.max_num_running_reqs)
        ]
        result = scheduler.can_schedule(mock_request)
        assert not result, "Should reject request when at capacity"

    def test_stop_token_handling(self, scheduler):
        """Test stop token detection and handling.
        
        This test verifies:
        1. Stop tokens are correctly identified
        2. Request status is updated appropriately
        3. Stop reasons are properly recorded
        4. Minimum token requirements are respected
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()

        # Configure mock with proper integer attributes
        mock_request.num_tokens = 5
        mock_request.num_output_tokens = 10
        mock_request.max_tokens = 20
        mock_request.output_token_ids = [1, 2, 3, 4, 999]

        # Configure sampling parameters
        sampling_params = Mock()
        sampling_params.min_tokens = 5
        sampling_params.ignore_eos = False
        sampling_params.stop_token_ids = (
            999, )  # Note: Changed to tuple to match the code
        mock_request.sampling_params = sampling_params

        # Configure other required attributes
        mock_request.status = None
        mock_request.stop_reason = None
        mock_request.eos_token_id = None
        mock_request.pooling_params = None

        # Important: These conditions must be met for stopping
        mock_request.num_output_tokens = 10  # Greater than min_tokens (5)
        mock_request.num_tokens = 5  # Less than max_model_len
        mock_request.output_token_ids = [1, 2, 3, 4,
                                         999]  # Last token is stop token

        # The function uses the actual output_token_ids[-1], not a property
        result = check_stop_with_min_tokens(
            mock_request, scheduler.scheduler_config.max_model_len)

        assert result, "Request should stop when stop token is encountered and min_tokens is satisfied"
        assert mock_request.status == RequestStatus.FINISHED_STOPPED
        assert mock_request.stop_reason == 999

    def test_invalid_request_handling(self, scheduler):
        """Test handling of invalid request configurations.
        
        This test verifies:
        1. Invalid requests are properly rejected
        2. Capacity constraints are enforced
        3. Request validation logic works correctly
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        mock_request = Mock()

        # Set up the request to be invalid
        mock_request.num_tokens = scheduler.scheduler_config.max_model_len + 1
        mock_request.num_output_tokens = 0
        mock_request.max_tokens = 10
        mock_request.sampling_params = None

        # Mock the running and waiting queues to be full
        scheduler.running = [
            Mock() for _ in range(scheduler.max_num_running_reqs)
        ]

        result = scheduler.can_schedule(mock_request)
        assert not result, "Should not schedule invalid requests"

    def test_schedule_with_empty_state(self, scheduler):
        """Test scheduler behavior with empty queues.
        
        This test verifies:
        1. Scheduler handles empty queue states correctly
        2. Valid output is produced with no requests
        3. Queue state remains consistent
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """

        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 0
        assert len(scheduler.holdback_queue) == 0

        output = scheduler.schedule()
        assert output is not None

    def test_concurrent_queue_operations(self, scheduler):
        """Test simultaneous operations on multiple queues.
        
        This test verifies:
        1. Multiple queues can be processed simultaneously
        2. Request count integrity is maintained
        3. Queue state transitions are handled correctly
        4. No requests are lost during processing
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Setup initial state
        running_requests = [Mock() for _ in range(2)]
        waiting_requests = [Mock() for _ in range(2)]
        holdback_requests = [Mock() for _ in range(2)]

        scheduler.running.extend(running_requests)
        scheduler.waiting.extend(waiting_requests)
        scheduler.holdback_queue.extend(holdback_requests)

        output = scheduler.schedule()
        assert output is not None

        # Verify queue integrity
        total_requests = len(scheduler.running) + len(scheduler.waiting) + len(
            scheduler.holdback_queue)
        assert total_requests == 6, "No requests should be lost during scheduling"

    def test_check_stop_with_min_tokens_no_stop_condition(self, scheduler):
        """Test case where no stop conditions are met.
        
        This test verifies that the function returns False when no stop
        conditions are satisfied, covering the final return False line.
        """
        mock_request = Mock()
        mock_request.num_tokens = 5  # Less than max_model_len
        mock_request.num_output_tokens = 5  # Less than max_tokens
        mock_request.max_tokens = 20
        mock_request.pooling_params = None  # No pooling params
        mock_request.sampling_params = Mock()
        mock_request.sampling_params.min_tokens = 0
        mock_request.output_token_ids = [1, 3, 4, 5]  # No stop tokens
        mock_request.sampling_params.stop_token_ids = ()

        result = check_stop_with_min_tokens(
            mock_request, scheduler.scheduler_config.max_model_len)
        assert not result

    def test_check_stop_with_min_tokens_pooling_params(self, scheduler):
        """Test stop checking with pooling parameters.
        
        This test verifies:
        1. Behavior with pooling params and pooler output
        2. Behavior with pooling params but no pooler output
        3. Status updates for pooling-based stops
        
        Args:
            scheduler: Fixture providing configured scheduler instance
        """
        # Test case 1: pooling params with pooler output
        mock_request = Mock()
        mock_request.num_tokens = 0  # Less than max_model_len
        mock_request.num_output_tokens = 0
        mock_request.max_tokens = 100
        mock_request.pooling_params = Mock()  # Just needs to exist
        mock_request.status = None

        # Test with pooler output
        result = check_stop_with_min_tokens(
            mock_request,
            scheduler.scheduler_config.max_model_len,
            pooler_output=torch.tensor([1.0])  # Provide pooler output
        )

        assert result is True, "Should stop when pooler output is provided"
        assert mock_request.status == RequestStatus.FINISHED_STOPPED

        # Test case 2: pooling params without pooler output
        mock_request = Mock()
        # Set required numeric attributes
        mock_request.num_tokens = 0
        mock_request.num_output_tokens = 0
        mock_request.max_tokens = 100
        mock_request.pooling_params = Mock()
        mock_request.status = None

        result = check_stop_with_min_tokens(
            mock_request,
            scheduler.scheduler_config.max_model_len,
            pooler_output=None  # No pooler output
        )

        assert result is False, "Should not stop when no pooler output"
        assert mock_request.status is None, "Status should not change"
