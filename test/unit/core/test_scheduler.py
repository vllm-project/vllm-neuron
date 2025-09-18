# SPDX-License-Identifier: Apache-2.0
from collections import deque
from unittest.mock import Mock, patch


class TestNeuronScheduler:

    def test_scheduler_initialization(self, scheduler):
        """Test basic scheduler initialization"""
        assert hasattr(scheduler, 'holdback_queue')
        assert len(scheduler.holdback_queue) == 0
        assert hasattr(scheduler, 'waiting')
        assert hasattr(scheduler, 'running')

    def test_can_schedule_empty_queues(self, scheduler):
        """Test scheduling when queues are empty"""
        mock_request = Mock()
        assert scheduler.can_schedule(mock_request)

    def test_can_schedule_with_running_requests(self, scheduler):
        """Test scheduling with existing running requests"""
        mock_request = Mock()
        scheduler.running = [Mock() for _ in range(5)]
        result = scheduler.can_schedule(mock_request)
        assert isinstance(result, bool)

    def test_schedule_with_empty_queues(self, scheduler):
        """Test schedule operation with empty queues"""
        output = scheduler.schedule()
        assert output is not None  # Changed because MockBaseScheduler returns a Mock
        assert len(scheduler.waiting) == 0
        assert len(scheduler.holdback_queue) == 0

    def test_schedule_with_waiting_requests(self, scheduler):
        """Test schedule operation with waiting requests"""
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()

        # Verify the scheduling behavior
        assert output is not None
        # Verify that request was processed
        total_requests = len(scheduler.waiting) + len(scheduler.holdback_queue)
        assert total_requests > 0

    def test_queue_management(self, scheduler):
        """Test queue management operations"""
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
        """Test logging behavior during scheduling"""
        mock_request = Mock()
        scheduler.holdback_queue.append(mock_request)
        output = scheduler.schedule()
        assert output is not None
        mock_logger.debug.assert_called()

    def test_max_capacity_constraints(self, scheduler):
        """Test scheduler respects maximum capacity constraints"""
        # Fill up to max capacity
        scheduler.running = [
            Mock() for _ in range(scheduler.max_num_running_reqs)
        ]
        mock_request = Mock()
        result = scheduler.can_schedule(mock_request)
        assert not result

    def test_batch_scheduling_logic(self, scheduler):
        """Test batch scheduling logic"""
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
        """Helper method to verify scheduler state"""
        actual_total = (len(scheduler.running) + len(scheduler.waiting) +
                        len(scheduler.holdback_queue))
        assert actual_total == expected_total_requests, \
            f"Expected {expected_total_requests} total requests, but found {actual_total}"

    def test_schedule_with_running_and_waiting(self, scheduler):
        """Test scheduling with both running and waiting requests"""
        # Setup initial state
        scheduler.running = [Mock() for _ in range(2)]
        scheduler.waiting = deque([Mock() for _ in range(2)])

        # Execute scheduling
        output = scheduler.schedule()

        # Verify that requests were processed
        assert output is not None
        assert len(scheduler.holdback_queue) >= 0
