# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallelism: no draft-token RPC after each scheduled step.

``EngineCore.post_step`` collects speculative drafts with a blocking
``take_draft_token_ids`` collective RPC after every scheduled batch. The reply
comes from the output rank -- the last pipeline stage -- whose worker answers
only once it has run the ``execute_model`` queued ahead of the RPC, i.e. after
the batch has drained the whole pipeline. With a batch queue that means the
engine never schedules batch ``i+1`` before batch ``i`` has finished: no two
batches overlap and pipeline parallelism degenerates into a serial run of the
stages. The drafts also reach the scheduler several outputs ahead of the step
they belong to.

Under PP with synchronous scheduling the Neuron model runner ships each step's
drafts inside that step's ``ModelRunnerOutput`` (``neuron_pp_drafts``; the
result queue pickles the object, so the extra attribute survives) and
``NeuronScheduler.update_from_output`` attaches them together with the sampled
tokens of the same step. This patch makes ``post_step`` a no-op in that mode so
the RPC is never issued. Async scheduling keeps vLLM's own behaviour
(``post_step`` already skips the RPC there).

Applied from ``apply_patches`` (vLLM's general-plugin hook), which
``EngineCore.__init__`` runs before it builds the executor and before its first
step. The same module sizes the executor's RPC ring for a full pipeline of
steps in flight (``apply_pp_mq_depth_patch``, below).
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_applied = False
_original_post_step = None


def pp_sync_scheduling(vllm_config) -> bool:
    """True when drafts travel in the step outputs: PP > 1, sync scheduling -- or
    one RPC a step (`VLLM_NEURON_ONE_RPC_STEP=1`, `NeuronModelRunner._one_rpc_step`)."""
    if vllm_config.scheduler_config.async_scheduling:
        return False
    return bool(
        vllm_config.parallel_config.pipeline_parallel_size > 1
        or os.environ.get("VLLM_NEURON_ONE_RPC_STEP") == "1"
    )


def apply_pp_draft_patch() -> None:
    """Make ``EngineCore.post_step`` skip ``take_draft_token_ids`` under PP."""
    global _applied, _original_post_step
    if _applied:
        return

    from vllm.v1.engine.core import EngineCore

    _original_post_step = EngineCore.post_step

    def post_step(self, model_executed: bool) -> None:
        if pp_sync_scheduling(self.vllm_config):
            return None
        return _original_post_step(self, model_executed)

    post_step.__doc__ = _original_post_step.__doc__
    EngineCore.post_step = post_step
    _applied = True
    logger.info(
        "Applied pipeline-parallel draft patch: no take_draft_token_ids RPC "
        "under PP, drafts ride in each step's ModelRunnerOutput"
    )


# --- RPC ring depth -----------------------------------------------------------
#
# `MultiprocExecutor` broadcasts every RPC through one shared-memory ring
# (`MessageQueue`, `max_chunks=10`). A chunk is reused only once *every* worker
# has read it, and a pipeline stage reads its next RPC only when it is done with
# the previous one, so the last stage -- `pipeline_parallel_size - 1` steps
# behind the first -- gates the ring. Each scheduled step costs two messages
# (`execute_model`, `sample_tokens`); with 10 chunks the engine blocks in
# `enqueue` after about five steps in flight, short of the
# `pipeline_parallel_size` batches the batch queue is sized for. Under PP the
# ring is sized to `4 * pipeline_parallel_size` chunks (16 MiB each by default,
# `VLLM_MQ_MAX_CHUNK_BYTES_MB`), a few hundred MiB of /dev/shm.

_applied_mq = False
_original_init_executor = None
_original_mq_init = None
_min_rpc_chunks = 0


def apply_pp_mq_depth_patch() -> None:
    """Size the executor's RPC ring for ``pipeline_parallel_size`` steps in flight."""
    global _applied_mq, _original_init_executor, _original_mq_init
    if _applied_mq:
        return

    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    _original_init_executor = MultiprocExecutor._init_executor
    _original_mq_init = MessageQueue.__init__

    def _init_executor(self, *args, **kwargs):
        global _min_rpc_chunks
        pp = self.vllm_config.parallel_config.pipeline_parallel_size
        _min_rpc_chunks = 4 * pp if pp > 1 else 0
        try:
            return _original_init_executor(self, *args, **kwargs)
        finally:
            _min_rpc_chunks = 0

    def mq_init(self, *args, **kwargs):
        # MessageQueue(n_reader, n_local_reader, local_reader_ranks=None,
        #              max_chunk_bytes=..., max_chunks=10, connect_ip=None)
        n_reader = args[0] if args else kwargs.get("n_reader", 1)
        if _min_rpc_chunks and n_reader > 1:
            if len(args) >= 5:
                if args[4] < _min_rpc_chunks:
                    args = (*args[:4], _min_rpc_chunks, *args[5:])
            elif kwargs.get("max_chunks", 10) < _min_rpc_chunks:
                kwargs["max_chunks"] = _min_rpc_chunks
            logger.info(
                "Pipeline parallelism: RPC ring sized to %d chunks for %d readers",
                _min_rpc_chunks, n_reader,
            )
        return _original_mq_init(self, *args, **kwargs)

    mq_init.__doc__ = _original_mq_init.__doc__
    MultiprocExecutor._init_executor = _init_executor
    MessageQueue.__init__ = mq_init
    _applied_mq = True
