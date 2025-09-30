# SPDX-License-Identifier: Apache-2.0
import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    SchedulerOutput = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support continuous 
    batching respecting Neuron constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)

        # Requests are temporarily moved to this queue so that the base
        # scheduler does not see them. This lets us ensure that the set of
        # requests scheduled have at least one common warmup shape.
        self.holdback_queue: deque[Request] = deque()

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """
        Override to use our enhanced stop checker that respects min_tokens.
        
        This fixes the bug where min_tokens is ignored in the scheduler's
        stop checking logic.
        """
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state using our enhanced checker.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop_with_min_tokens(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped


class ContinuousBatchingNeuronScheduler(NeuronScheduler):
    """ Support of continuous batching """

    # inherited from V1 base scheduler but mypy needs to know the type
    running: list[Request]

    def __init__(self, *args, **kwargs) -> None:
        # Initialize NeuronScheduler
        super().__init__(*args, **kwargs)

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after.
        """

        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled.
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                # Add request to the waiting queue
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        # Schedule Prefill and Decode separately
        if len(self.waiting) > 0:
            # For prefill, hide current decodes from the scheduler
            running_holdback = self.running
            self.running = []
            logger.debug(
                f"Scheduling a prefill step of {len(self.waiting)} requests, holding back {len(self.holdback_queue)} "
                "requests")
        else:
            running_holdback = []
            logger.debug(
                f"Scheduling a decode step of {len(self.running)} requests")

        # delegate to super of NeuronScheduler: base V1 Scheduler
        outputs = super(NeuronScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        return outputs

    def can_schedule(self, request) -> bool:
        max_prompt_batch_size = 1
        _max_context_len = self.scheduler_config.max_model_len

        # running and waiting queues are both empty -> start new batch
        start_new_batch = len(self.running) + len(self.waiting) == 0
        # check that there is space in the current decode batch
        cond1 = len(self.running) + len(
            self.waiting) < self.max_num_running_reqs
        # check that there is space in the prefill batch
        cond2 = len(self.waiting) < max_prompt_batch_size
        # TODO: add context length checks
        return start_new_batch or (cond1 and cond2)


def check_stop_with_min_tokens(
        request: Request,
        max_model_len: int,
        pooler_output: Optional[torch.Tensor] = None) -> bool:
    """
    Temporary fix to address missing min_token check. The V1 scheduler's stop checker 
    does not properly respect min_tokens.
    """
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params
    assert sampling_params is not None

    min_tokens = sampling_params.min_tokens
    if min_tokens > 0 and request.num_output_tokens < min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
