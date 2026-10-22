# SPDX-License-Identifier: Apache-2.0
"""
NeuronModelRunner - Model runner implementation for vLLM integration with Neuron

This provides the execution engine that processes SchedulerOutput and returns
ModelRunnerOutput with proper request tracking, output reordering, and persistent
state management.
"""

import functools
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Collection
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from vllm import ModelRegistry
from vllm.v1.attention.backend import AttentionMetadata
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.interfaces import (
    supports_eagle3,
    supports_transcription,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.lnc import get_explicit_compiler_lnc_arg
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)

from contextlib import AbstractContextManager, contextmanager, nullcontext

from vllm_neuron import envs
from vllm_neuron.snapshot.runner_hook import SnapshotRunnerHook
from vllm_neuron.utils.neuron_utils import model_forward_context
from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args
from vllm_neuron.vllm.worker.neuron_ec_connector_model_runner_mixin import (
    ECLoadFailure,
    EncoderCacheReadError,
    NeuronECConnectorModelRunnerMixin,
)
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    DEVICE_ENCODER_CACHE_ACTIVE_BLOCKS,
    DEVICE_ENCODER_CACHE_ACTIVE_REAL_TOKENS,
    DEVICE_ENCODER_CACHE_EVICTIONS,
    DEVICE_ENCODER_CACHE_FREE_BLOCKS,
    DEVICE_ENCODER_CACHE_HELD_BLOCKS,
    DEVICE_ENCODER_CACHE_MISSES,
    DEVICE_ENCODER_CACHE_QUERIES,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.interfaces import SupportsMRoPE
from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
)
from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from vllm_neuron.vllm.sample.rejection_sampler import RejectionSampler
from vllm_neuron.vllm.spec_decode.dflash import DFlashProposer
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer
from vllm_neuron.vllm.spec_decode.mtp import MTPProposer
from vllm_neuron.vllm.platform import SO_DISABLED_MESSAGE
from vllm_neuron.utils.bucket_utils import (
    SUPPORTED_KV_SEGMENT_SIZES,
    get_max_num_batched_tokens,
    get_decode_padded_batch_size,
    pick_kv_segment_size,
    segmented_prefill_supports_q_len,
)
from vllm_neuron.utils.spec_decode_utils import (
    extract_next_token_ids,
    replicate_per_seq_rows,
)
from vllm_neuron.accuracy.tensor_replacement import (
    TensorReplacer,
    set_active_context,
)
from vllm_neuron.parallel.neuron_parallel_state import stamp_spmd_rank_tensors

logger = logging.getLogger(__name__)


NULL_BLOCK_ID = 0
# Read-side sentinel for block tables; write-side padding uses NULL_BLOCK_ID.
PAD_SLOT_ID = -1


def _segmented_prefill_required_msg(
    feature: str,
    disable_hint: str,
    *,
    non_packed_fp8_kv: bool,
    cache_dtype: str,
    max_num_batched_tokens: int,
    max_model_len: int,
) -> str:
    """Build the error message for a feature that requires segmented prefill.

    Both APC and DCP need segmented prefill (kv_segment_size_buckets set); they
    differ only in which feature is asking and how to turn it off. Shared here so
    the two call sites stay consistent.

    Segmented prefill can be unavailable for three unrelated reasons, so the
    message names the actual one: blaming max_model_len for an FP8 config, or
    for a single-shot prefill where a segment size does fit, sends the user off
    changing a limit that was never the problem.

    Args:
        feature: Human-readable name of the feature requiring segmented prefill,
            e.g. ``"Automatic Prefix Caching (APC)"``.
        disable_hint: How to disable that feature, e.g.
            ``"disable APC with '--no-enable-prefix-caching'"``.
        non_packed_fp8_kv: Whether the KV cache is FP8 in the standard (unpacked)
            layout, which the segmented attention kernel cannot read.
        cache_dtype: Configured KV cache dtype, quoted into the FP8 message.
        max_num_batched_tokens: Q chunk size, i.e. the value whose relation to
            ``max_model_len`` and to the supported segment sizes decided that
            segmented prefill stayed off.
        max_model_len: Maximum model length, quoted into the segment-size message.

    Returns:
        The formatted error string.

    Example:
        >>> msg = _segmented_prefill_required_msg(
        ...     "APC",
        ...     "disable APC with X",
        ...     non_packed_fp8_kv=False,
        ...     cache_dtype="auto",
        ...     max_num_batched_tokens=256,
        ...     max_model_len=256,
        ... )
        >>> msg.startswith("APC requires segmented prefill to be enabled, but")
        True
        >>> "single-shot" in msg
        True
    """
    supported = sorted(SUPPORTED_KV_SEGMENT_SIZES)
    if non_packed_fp8_kv:
        reason = (
            "the segmented attention kernel cannot read a non-packed FP8 KV "
            f"cache (cache_dtype={cache_dtype!r}). Either {disable_hint}, drop "
            "'--kv-cache-dtype fp8' so the cache follows the model dtype, or "
            "enable the packed layout with neuron_config fp8_packed_kv=true."
        )
    elif max_num_batched_tokens >= max_model_len:
        reason = (
            "prefill is single-shot, so there is no prior KV to segment over "
            f"(max_num_batched_tokens={max_num_batched_tokens} >= "
            f"max_model_len={max_model_len}). Either {disable_hint}, or chunk "
            "prefill by lowering 'max_num_batched_tokens' to one of the "
            f"supported segment sizes {supported}, or set "
            "'kv_segment_size_buckets' explicitly in neuron_config."
        )
    else:
        reason = (
            f"max_num_batched_tokens={max_num_batched_tokens} is not one of the "
            f"supported KV segment sizes {supported}, so no diagonal segment "
            f"target was defaulted on. Either {disable_hint}, set "
            "'max_num_batched_tokens' to one of those sizes (which also needs "
            f"max_model_len > {min(supported)}; it is {max_model_len}), or set "
            "'kv_segment_size_buckets' explicitly in neuron_config."
        )
    return f"{feature} requires segmented prefill to be enabled, but {reason}"


def _remap_null_block_to_sentinel(block_table: torch.Tensor) -> torch.Tensor:
    """Remap vLLM's null-block (0) in an int32 block table to -1 so the
    Neuron attention kernel can DMA-skip inactive slots via oob_mode.skip.

    Out-of-place so prior-step async work holding the previous tensor is
    unaffected. torch.compile-friendly (no python branches on tensor values).
    """
    return torch.where(
        block_table == NULL_BLOCK_ID,
        torch.full_like(block_table, PAD_SLOT_ID),
        block_table,
    )


# Depth of the host staging ring for the block-table remap and the SWA column gather.
#
# Each staged array reaches the device through ``.to()``, and reusing one array is only correct
# if that copy is synchronous. A ring removes the dependency on copy semantics this code does not
# control. Keep it above the engine's in-flight window, which NEURON_MAX_BATCH_QUEUE_DEPTH caps
# at 4, so 8 leaves two windows of margin.
_BLOCK_TABLE_STAGE_DEPTH = 8

# threading.Lock is a factory and not a class, so capture the instance type one time for the
# seam check in _batch_mutation_lock. A test stand-in holds a Mock there, and a Mock satisfies
# `is not None` while failing the context-manager protocol.
_LOCK_TYPE = type(threading.Lock())


@dataclass(slots=True)
class _BlockTableGroupState:
    """Host staging ring and the last device upload for the block table of one KV group.

    The runner holds one instance per KV-cache group, so it cannot leak into a second runner.

    Attributes:
        stage: Ring of host arrays that hold the remapped table.
        null_mask: Ring of boolean arrays that hold the null-block mask.
        next_slot: Index of the ring slot for the next remap.
        uploaded_host: Copy of the raw host table of the last upload, or None.
        uploaded_device: Device tensor of the last upload, or None.
    """

    stage: list[np.ndarray] = field(default_factory=list)
    null_mask: list[np.ndarray] = field(default_factory=list)
    next_slot: int = 0
    uploaded_host: np.ndarray | None = None
    uploaded_device: torch.Tensor | None = None

    def fits(self, shape: tuple[int, ...], dtype: np.dtype) -> bool:
        """Report whether the ring fits a table of this shape and this dtype."""
        return (
            bool(self.stage)
            and self.stage[0].shape == shape
            and self.stage[0].dtype == dtype
        )

    def reset(self, raw: np.ndarray) -> None:
        """Allocate a new ring for the shape and the dtype of a raw table.

        A shape change or a dtype change invalidates the previous upload, so drop it here.
        """
        self.stage = [np.empty_like(raw) for _ in range(_BLOCK_TABLE_STAGE_DEPTH)]
        self.null_mask = [
            np.empty(raw.shape, dtype=bool) for _ in range(_BLOCK_TABLE_STAGE_DEPTH)
        ]
        self.next_slot = 0
        self.uploaded_host = None
        self.uploaded_device = None


@dataclass(slots=True)
class _SwaGroupState:
    """Host staging ring and the last result for the SWA decode tensors of one KV group.

    One instance per KV-cache group, owned by the runner.

    Attributes:
        stage: Ring of host arrays that hold the gathered columns.
        null_mask: Ring of boolean arrays that hold the null-block mask.
        next_slot: Index of the ring slot for the next gather.
        gather_index: Copy of the gather index of the last result, or None.
        rows: Copy of the block-table rows of the last result, or None.
        trimmed_table: Trimmed device block table of the last result, or None.
        position_offset: Device position offset of the last result, or None.
    """

    stage: list[np.ndarray] = field(default_factory=list)
    null_mask: list[np.ndarray] = field(default_factory=list)
    next_slot: int = 0
    gather_index: np.ndarray | None = None
    rows: np.ndarray | None = None
    trimmed_table: torch.Tensor | None = None
    position_offset: torch.Tensor | None = None

    def fits(self, shape: tuple[int, ...], dtype: np.dtype) -> bool:
        """Report whether the ring fits a gather of this shape and this dtype."""
        return (
            bool(self.stage)
            and self.stage[0].shape == shape
            and self.stage[0].dtype == dtype
        )

    def reset(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        """Allocate a new ring for this shape and this dtype, and drop the last result."""
        self.stage = [
            np.empty(shape, dtype=dtype) for _ in range(_BLOCK_TABLE_STAGE_DEPTH)
        ]
        self.null_mask = [
            np.empty(shape, dtype=bool) for _ in range(_BLOCK_TABLE_STAGE_DEPTH)
        ]
        self.next_slot = 0
        self.gather_index = None
        self.rows = None
        self.trimmed_table = None
        self.position_offset = None

    def hits(self, gather_index: np.ndarray, rows: np.ndarray) -> bool:
        """Report whether the last result came from these exact inputs.

        Both outputs are a pure function of the gather index and the block-table rows, so equal
        inputs give equal outputs. Compare the values, because the scheduler changes the host
        block table in place between steps.
        """
        # ``np.array_equal`` reports False for a shape mismatch, so it needs no shape test.
        return (
            self.trimmed_table is not None
            and np.array_equal(self.gather_index, gather_index)
            and np.array_equal(self.rows, rows)
        )


def _supports_async_reorder(model: Any) -> bool:
    """Return True when the forward of the model consumes the in-graph reorder arguments.

    A model opts in with the class attribute ``supports_async_reorder``, and only one whose
    forward assembles the input_ids from those arguments may set it. Without it the model keeps
    the host swap path. ``torch.compile`` wraps the module, so read the attribute from the
    original.

    Args:
        model: The loaded model, wrapped or unwrapped.

    Returns:
        True when the model consumes the reorder arguments.
    """
    model = getattr(model, "_orig_mod", model)
    return getattr(model, "supports_async_reorder", False) is True


def _supports_aux_hidden_states(model: Any) -> bool:
    # vLLM 0.21.0 only exposes SupportsEagle3, but the capability DFlash needs
    # is generic auxiliary hidden-state capture.
    return supports_eagle3(model)


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    positions: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    aux_hidden_states: list[torch.Tensor] | None = None
    input_ids: torch.Tensor | None = None
    attn_metadata: dict | None = None


class AsyncNeuronModelRunnerOutput(AsyncModelRunnerOutput):
    """Wraps a ModelRunnerOutput whose sampled_token_ids is still a device
    tensor future.  Exactly one instance is created per step in
    sample_tokens() and shared between the async output thread (via
    enqueue_output) and the worker main thread (via the pre-_update_states
    guard).  The per-instance lock in get_output() prevents both threads
    from materializing the same future concurrently."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        model_runner: "NeuronModelRunner",
        partial_prefill_req_ids: set[str] | None = None,
    ):
        self.model_runner_output = model_runner_output
        self._model_runner = model_runner
        self._partial_prefill_req_ids = partial_prefill_req_ids or set()
        # Per-object lock: prevents double-materialization when both the async
        # output thread and the main thread call get_output() on the same object.
        self._lock = threading.Lock()
        # When True, get_output() returns empty sampled_token_ids (one empty
        # list per req). Set by the spec→non-spec transition handler when
        # the bonus token will be re-emitted by the transition step itself,
        # to avoid duplicate emission to the client.
        self._skip_emit: bool = False

    def _discard_partial_prefill_samples(
        self, sampled_token_ids: list[list[int]]
    ) -> None:
        """Drop sampled tokens produced by intermediate chunked-prefill steps."""
        if not self._partial_prefill_req_ids:
            return
        for req_idx, req_id in enumerate(self.model_runner_output.req_ids):
            if req_id not in self._partial_prefill_req_ids:
                continue
            if req_idx < len(sampled_token_ids):
                sampled_token_ids[req_idx] = []

    def is_all_partial_prefill(self) -> bool:
        """True when every request in this output is an intermediate
        (non-final) segmented-prefill chunk, whose sampled tokens are
        discarded. Used to decide both that the readback can be done on the
        async-output thread (here) and that the worker submit thread should
        NOT materialize this output (see
        ``NeuronModelRunner._materialize_pending_async_output``)."""
        return bool(self._partial_prefill_req_ids) and len(
            self._partial_prefill_req_ids
        ) >= len(self.model_runner_output.req_ids)

    def get_output(self) -> ModelRunnerOutput:
        """
        Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        involves copying device tensors to the host.

        Thread-safety: Both the async output thread (via enqueue_output) and
        the worker main thread (via the pre-_update_states guard) may call
        this method concurrently on the same object. A lock ensures only one
        thread performs the materialization; the other sees the already-
        converted list and skips _update_batch_state_with_samples.
        """
        with self._lock:
            sampled_token_ids = self.model_runner_output.sampled_token_ids
            if torch.is_tensor(sampled_token_ids):
                if self.is_all_partial_prefill():
                    # Intermediate segmented-prefill chunk: the sampled tokens
                    # are discarded, but we MUST read the future back to drain
                    # the segment NEFF from the NRT execution queue. Without a
                    # drain the queue grows by one per segment and a prompt
                    # longer than kv_segment_size * NRT_queue_cap overflows it
                    # ("Execution queue full").
                    sampled_token_ids.cpu()
                    sampled_token_ids = [
                        [] for _ in range(len(self.model_runner_output.req_ids))
                    ]
                    self.model_runner_output.sampled_token_ids = sampled_token_ids
                    return self.model_runner_output

                # First call: materialize device future to CPU.
                if sampled_token_ids.ndim == 2 and sampled_token_ids.shape[1] > 1:
                    # Spec decode: rejection sampler output [bs, num_spec+1]
                    # with -1 padding for rejected positions. Strip -1s to
                    # produce variable-length list[list[int]].
                    sampled_token_ids = (
                        self._model_runner._parse_rejection_sampling_output(
                            sampled_token_ids,
                            self._model_runner.input_batch.vocab_size,
                        )
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        drafter = self._model_runner.drafter
                        num_spec = (
                            drafter.num_speculative_tokens if drafter is not None else 0
                        )
                        for req_i, tokens in enumerate(sampled_token_ids):
                            num_accepted = max(0, len(tokens) - 1)
                            logger.debug(
                                "async-spec step: req=%d accepted=%d/%d",
                                req_i,
                                num_accepted,
                                num_spec,
                            )
                else:
                    # Non-spec: [bs] of sampled tokens.
                    sampled_token_ids = [[x] for x in sampled_token_ids.cpu().tolist()]
                self._discard_partial_prefill_samples(sampled_token_ids)
                # If this output was marked for skip-emit (e.g. the last
                # spec step's bonus will be re-emitted by the following
                # transition step), replace with empty lists so the client
                # doesn't see a duplicate. Do this AFTER materialization
                # so _update_batch_state_with_samples still writes the
                # real sampled tokens into token_ids_cpu (needed for
                # input_ids building in later steps if they fall back).
                if self._skip_emit:
                    empty_tokens = [[] for _ in sampled_token_ids]
                else:
                    empty_tokens = None
                # Use the snapshotted req_ids from the time of sampling so that
                # tokens are written to the correct batch slot even if condense()
                # has reordered the batch since then.
                self._model_runner._update_batch_state_with_samples(
                    sampled_token_ids,
                    snapshot_req_ids=self.model_runner_output.req_ids,
                )
                if empty_tokens is not None:
                    sampled_token_ids = empty_tokens
                self.model_runner_output.sampled_token_ids = sampled_token_ids
                logger.debug(
                    "sampled_token_ids on cpu after moving: sampled_token_ids=%s",
                    sampled_token_ids,
                )
            else:
                logger.debug(
                    "sampled_token_ids has already been actualized. "
                    "Skip updating batch state."
                )

        # Logprobs are computed in _sample() when on-device logits are available
        return self.model_runner_output


class VeEncoderProducerAsyncOutput(AsyncModelRunnerOutput):
    """Deferred-drain wrapper for a vision-encoder (VE) EC-producer step.

    The VE Neff scatter-writes embeddings into the on-device encoder cache. The
    NRT submit does not block the host, so the submit thread is free right after
    it. The EC producer must guarantee the embedding and the hash tag are in HBM
    before the router gets the ack, or the consumer reads stale data. The
    synchronous path forces that guarantee with a device->host read on the submit
    thread, which idles the Neff between requests.

    This wrapper moves only that drain off the submit thread. The EC locator is
    already attached on the submit thread, so it reflects this step's cache
    state and cannot race the next step's cache changes. vLLM's async-output
    handler calls get_output() one step later, so the submit thread starts the
    next request's CPU block-packing while this request's Neff runs. get_output()
    drains the NRT queue (this guarantees the HBM write), then returns the
    token-less output. The drain keeps the exact HBM-before-ack order of the
    synchronous path.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        model_runner: "NeuronModelRunner",
    ) -> None:
        self.model_runner_output = model_runner_output
        self._model_runner = model_runner
        # get_output() may run on the async-output handler thread. A lock plus a
        # one-shot flag makes the drain idempotent if it is called more than once.
        self._lock = threading.Lock()
        self._drained = False

    def get_output(self) -> ModelRunnerOutput:
        """Drain the NRT queue, then return the token-less VE output.

        The single-element reads are queue-ordered after the encoder Neff, so
        they block until the full embedding and tag writes land in HBM. This
        runs before vLLM emits the response, so the router acks only after the
        write is visible to the consumer.
        """
        with self._lock:
            if not self._drained:
                ec = self._model_runner.encoder_cache
                _ = ec.buffer[0, 0, 0].detach().to("cpu")
                _ = ec.mm_hash_tags[0].detach().to("cpu")
                self._drained = True
            return self.model_runner_output


def _reinterpret_uint32_as_int32(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Reinterpret uint32 NEFF output bits as int32 without a device cast.

    NKI argmax kernels return UInt32 token IDs. Token IDs fit in int32, so
    reinterpreting the bits via ``.view(int32)`` is safe and avoids a device
    cast that would either materialize or fail on Neuron XLA. No-op when the
    tensor dtype already matches.
    """
    if tensor.dtype == torch.uint32 and target_dtype == torch.int32:
        return tensor.view(torch.int32)
    return tensor


def _compute_slot_mapping_cpu(
    block_table_cpu: np.ndarray,
    slot_mapping_np: np.ndarray,
    positions: np.ndarray,
    req_indices: np.ndarray,
    block_size: int,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    """Compute slot mapping on CPU using vectorized numpy.

    This was originally how vllm computed slot mappings (vectorized numpy on
    CPU). In vllm 0.19.0 (commit fafe76b), this was replaced with a Triton
    GPU kernel. Since Triton is not available on Neuron, we keep the original
    numpy approach here.

    Supports context parallelism (CP): when cp_world_size > 1, only local
    tokens get valid slot IDs; remote-token writes target the null block.
    """
    max_blocks_per_req = block_table_cpu.shape[1]
    virtual_block_size = block_size * cp_world_size

    block_indices = positions // virtual_block_size
    flat_indices = req_indices * max_blocks_per_req + block_indices

    # Guard the gather: a (request, block) coordinate outside the allocated
    # block table would raise IndexError and kill the worker. Redirect such
    # writes to the reserved null block (the discard target used for padding
    # tokens) and warn. Bounds are checked per axis; block_table_cpu may be a
    # numpy array or a torch tensor, so only shape is used.
    num_table_rows = block_table_cpu.shape[0]
    oob_mask = (block_indices >= max_blocks_per_req) | (req_indices >= num_table_rows)
    has_oob = bool(oob_mask.any())
    if has_oob:
        oob_sample = np.flatnonzero(oob_mask)[:8]
        logger.warning(
            "slot mapping OOB: %d/%d positions map outside the block table "
            "(max_blocks_per_req=%d, num_table_rows=%d, block_size=%d); "
            "first offenders as (req_index, position): %s; "
            "redirecting their KV writes to the null block.",
            int(oob_mask.sum()),
            len(positions),
            max_blocks_per_req,
            num_table_rows,
            block_size,
            [(int(req_indices[i]), int(positions[i])) for i in oob_sample],
        )
        flat_indices = np.where(oob_mask, 0, flat_indices)
    block_numbers = block_table_cpu.ravel()[flat_indices]

    if cp_world_size > 1:
        virtual_block_offsets = positions % virtual_block_size
        is_local = (
            virtual_block_offsets // cp_kv_cache_interleave_size % cp_world_size
            == cp_rank
        )
        block_offsets = (
            virtual_block_offsets
            // (cp_world_size * cp_kv_cache_interleave_size)
            * cp_kv_cache_interleave_size
            + virtual_block_offsets % cp_kv_cache_interleave_size
        )
        slot_mapping = block_numbers * block_size + block_offsets
        if has_oob:
            slot_mapping = np.where(oob_mask, NULL_BLOCK_ID, slot_mapping)
        slot_mapping_np[: len(positions)] = np.where(
            is_local, slot_mapping, NULL_BLOCK_ID
        )
    else:
        block_offsets = positions % block_size
        np.add(
            block_numbers * block_size,
            block_offsets,
            out=slot_mapping_np[: len(positions)],
        )
        if has_oob:
            slot_mapping_np[: len(positions)][oob_mask] = NULL_BLOCK_ID


# Cache size for the default sampling tensors.
#
# One entry holds three small device tensors for one batch size. The runner uses a small set of
# batch sizes, so this size covers every bucket and it bounds the device memory of the cache.
_SAMPLING_DEFAULTS_CACHE_SIZE = 16


@functools.lru_cache(maxsize=_SAMPLING_DEFAULTS_CACHE_SIZE)
def _default_sampling_params(
    num_reqs: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the default top_k, top_p and temperature tensors for one batch size.

    Constants for a batch size and device. vLLM omits a sampling tensor when every request shares
    the same setting, so the defaults are the common case and cost three allocations per step.
    The caller only reads them, so sharing across steps is safe.

    Args:
        num_reqs: Number of active requests in the batch.
        device: Device that holds the tensors.

    Returns:
        The top_k, top_p and temperature defaults, in that order.

    Example:
        >>> top_k, top_p, temperature = _default_sampling_params(8, torch.device("cpu"))
        >>> int(top_k[0]), float(top_p[0]), float(temperature[0])
        (-1, 1.0, 0.0)
    """
    return (
        torch.full((num_reqs,), -1, dtype=torch.float32, device=device),
        torch.ones(num_reqs, dtype=torch.float32, device=device),
        torch.zeros(num_reqs, dtype=torch.float32, device=device),
    )


def build_sampling_params_tensor(
    sampling_metadata, num_reqs: int, device: torch.device
) -> torch.Tensor:
    """Build a (num_reqs, 3) tensor of [top_k, top_p, temperature] from vLLM sampling metadata.

    vLLM's InputBatch (gpu_input_batch.py) optimizes away sampling tensors when
    all requests share the same type: all_greedy → temperature=None, no_top_k →
    top_k=None, no_top_p → top_p=None. The provided tensors live on the InputBatch
    device (e.g. neuron:0), while absent ones need defaults.

    All tensors are placed on the caller's device to avoid mixed-device errors
    in torch.stack. Provided tensors are defensively moved to the target device
    in case they originate from a different device.

    Dtype handling: top_k is int32 in InputBatch but we need float32 for
    stacking. We use .float() instead of .to(torch.float32) because the latter
    fails on XLA devices with a dtype mismatch error.

    Args:
        sampling_metadata: vLLM SamplingMetadata with per-request params.
        num_reqs: Number of active requests in the batch.
        device: Device to create tensors on (must match InputBatch device).

    Returns:
        Tensor of shape (num_reqs, 3) with columns [top_k, top_p, temperature].
    """
    # Defaults: top_k=-1 (disabled), top_p=1.0 (disabled), temperature=0.0 (greedy).
    top_k, top_p, temperature = _default_sampling_params(num_reqs, device)

    # Override with per-request values when vLLM provides them.
    # Defensive .to(device) in case tensors originate from a different device.
    if sampling_metadata.top_k is not None:
        # [CHRS-711] TODO: Find a way to do avoid round-trip.
        top_k = sampling_metadata.top_k.cpu().float().to(device)
    if sampling_metadata.top_p is not None:
        top_p = sampling_metadata.top_p.to(device)
    if sampling_metadata.temperature is not None:
        temperature = sampling_metadata.temperature.to(device)

    result = torch.stack([top_k, top_p, temperature], dim=1)
    # `.tolist()` is an argument, so lazy %-formatting does not skip it. The call copies
    # the tensor from the device on every step while debug logging is off. Test the
    # level first.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "On-device sampling params (top_k, top_p, temp): %s", result.tolist()
        )
    return result


# TODO: Inherit from LoRAModelRunnerMixin to support LoRA
def _kv_tensor_layer_names(kv_cache_tensor) -> list[str]:
    """Return the layer names a ``KVCacheTensor`` covers, from ``layers`` or ``shared_by``.
    vLLM renamed the field, and reading the missing one raises inside KV allocation.
    """
    names = getattr(kv_cache_tensor, "layers", None)
    if names is None:
        names = getattr(kv_cache_tensor, "shared_by", None)
    assert names is not None, (
        "This KVCacheTensor has neither 'layers' nor 'shared_by'. vLLM renamed the "
        f"field again; add the new name here. Fields: "
        f"{sorted(getattr(kv_cache_tensor, '__dataclass_fields__', {}))}"
    )
    return list(names)


def _kv_buffer_identity(kv_cache_tensor, tensor_index: int, layer_index: int) -> tuple:
    """Return a hashable key for the BUFFER a layer's KV lives in.
    An offset key aliases layers. Without ``layer_stride`` every tensor reports ``offset == 0``.
    """
    layer_stride = getattr(kv_cache_tensor, "layer_stride", None)
    if layer_stride is not None:
        return ("byte", kv_cache_tensor.offset + layer_index * layer_stride)
    if getattr(kv_cache_tensor, "block_stride", 0) > 0:
        return ("packed", kv_cache_tensor.offset)
    return ("tensor", tensor_index)


class NeuronModelRunner(KVConnectorModelRunnerMixin, NeuronECConnectorModelRunnerMixin):
    """
    Model runner that executes the NeuronModel with proper state management.

    This class processes SchedulerOutput from vLLM and returns ModelRunnerOutput
    with correct request ordering and persistent state tracking.
    """

    # Per-runner decode state, declared on the class because a caller can build a runner
    # without ``__init__`` and ``Mock(spec=...)`` allows class attributes only. ``__init__``
    # assigns the real value. Each dictionary starts as None and its owner method creates it on
    # first use, since a mutable class attribute would be shared between runners.
    _async_reorder_kwargs: dict | None = None
    _model_supports_async_reorder: bool | None = None
    _block_table_groups: "dict[int, _BlockTableGroupState] | None" = None
    _constant_tensors: "dict[tuple, torch.Tensor] | None" = None
    _swa_groups: "dict[int, _SwaGroupState] | None" = None
    # A proven result can be None, so a second flag records whether the proof ran.
    _dp_identity_resolved: bool = False
    _dp_identity_constant: int | None = None
    _previous_dummy_output: Any = None
    # Serialises InputBatch mutation between the worker main thread and the async-output
    # thread. See _batch_mutation_lock for the race and for why it cannot deadlock. A
    # mutable class attribute would be shared between runners, so __init__ makes the lock.
    _input_batch_lock: "threading.Lock | None" = None

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------
    def __init__(
        self,
        vllm_config: Any,
        device: torch.device | None = None,
    ):
        """
        Initialize the model runner.

        Args:
            vllm_config: vLLM configuration object
            device: Device to use for execution (neuron:0 by default)
        """
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config

        # Pure kv_producer engine (prefill-only; decode is remote): every output
        # is a do_remote_decode prefill that finishes and hands off, so no later
        # step re-reads its sampled token. That lets _materialize_pending_async_output
        # skip the submit-thread materialize for it. Excludes kv_both, which
        # decodes locally and needs the token fed back.
        _kv_cfg = vllm_config.kv_transfer_config
        self._pure_producer = (
            _kv_cfg is not None
            and _kv_cfg.is_kv_producer
            and not _kv_cfg.is_kv_consumer
        )
        # A pure KV consumer receives every prompt over the network, so it never runs a
        # prefill-shape step. The DP reduce identity proof needs that, because the proof must
        # hold on every rank of the group at the same step. ``is_kv_consumer`` is also true
        # for kv_both, which prefills locally, so the producer test is load-bearing.
        self._pure_consumer = (
            _kv_cfg is not None
            and _kv_cfg.is_kv_consumer
            and not _kv_cfg.is_kv_producer
        )

        # TODO: Initialize LORA configuration and manager
        # Why required: Required before LORA adapters can be used
        # Current: No LORA config or manager initialization
        # Target: self.lora_config = vllm_config.lora_config if vllm_config else None; self.lora_manager = None

        self.model: Any | None = None
        self._is_synthetic_model: bool = False
        self._target_tensor_capture = None
        self._draft_tensor_capture = None
        self._encoder_tensor_capture = None
        model_config = vllm_config.model_config
        # Pooling/embedding models run forward -> hidden states -> pooler ->
        # embedding. Derive the mode from the resolved runner_type.
        self.is_pooling_model = model_config.runner_type == "pooling"
        self._pooling_num_scheduled_tokens_per_seq: np.ndarray | None = None
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim

        # Extract config from vllm_config.
        # A multimodal model nests vocab_size in a text_config (Qwen3-VL), or
        # in thinker_config.text_config (Qwen3-ASR thinker format).
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            thinker_config = getattr(hf_config, "thinker_config", None)
            text_config = getattr(thinker_config, "text_config", None)
        if text_config is not None and hasattr(text_config, "vocab_size"):
            self.vocab_size = text_config.vocab_size
        else:
            self.vocab_size = hf_config.vocab_size
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # Initialize persistent batch and request tracking
        # Use provided device or default to neuron:0
        # Device should be provided by worker after setting env vars
        self.device = device if device is not None else torch.device("neuron:0")
        self.pin_memory = False  # Default to False
        # Prompt-embeds config and per-step staging state.
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()

        # Initialize NeuronConfig from additional_config
        neuron_config_dict = vllm_config.additional_config.get("neuron_config", {})
        # TODO: Remove max_logprobs from NeuronConfig; use vllm_config directly in the model instead
        neuron_config_dict.setdefault(
            "max_logprobs", getattr(vllm_config.model_config, "max_logprobs", 0)
        )
        self.neuron_config = NeuronConfig.from_dict(neuron_config_dict)

        # Disable JF decoding for M-RoPE models
        if self.neuron_config.max_jump_forward_tokens > 0 and self.uses_mrope:
            raise ValueError(
                "Jump Forward decoding is not supported for M-RoPE models."
            )

        # Encoder neuron config for multimodal models (None for text-only)
        encoder_config_dict = vllm_config.additional_config.get("encoder_neuron_config")
        if encoder_config_dict is not None:
            from vllm_neuron.model.neuron_config import EncoderNeuronConfig

            self.encoder_neuron_config = EncoderNeuronConfig.from_dict(
                encoder_config_dict
            )
            self.encoder_neuron_config.resolve_tp_dp(
                vllm_config.parallel_config.world_size
            )
        else:
            self.encoder_neuron_config = None

        # Multimodal detection via registry (consistent with GPU model runner)
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )
        if self.supports_mm_inputs:
            logger.info(
                "Multimodal on-device encoder cache path enabled. "
                "Encoder embeddings injected during prefill via "
                "encoder_embedding_buffer + encoder_block_ids."
            )
        elif self.enable_prompt_embeds:
            logger.info("Prompt embeds path enabled for text model.")
        # Staged prompt_embeds for the current step (None => token-id-only step).
        self._current_inputs_embeds: torch.Tensor | None = None
        self._current_is_token_ids: torch.Tensor | None = None
        self.max_num_batched_tokens = get_max_num_batched_tokens(
            vllm_config.scheduler_config.max_num_batched_tokens, self.max_model_len
        )

        self.on_device_sampling: bool = (
            self.neuron_config.on_device_sampling_config is not None
        )
        self._on_device_logits: torch.Tensor | None = None
        self._debug_logits_dir: str | None = self.neuron_config.debug_logits_dir
        self._debug_logits_step_counter: int = 0
        self._tensor_replacer = None

        # Context parallelism: DCP rank for KV cache sharding.
        parallel_config = vllm_config.parallel_config
        self._dcp_size = parallel_config.decode_context_parallel_size
        neuron_cfg = vllm_config.additional_config.get("neuron_config", {})
        self.cp_world_size = self._dcp_size
        self.cp_kv_cache_interleave_size = getattr(
            parallel_config, "cp_kv_cache_interleave_size", 1
        )
        if self._dcp_size > 1 and vllm_config.kv_transfer_config is not None:
            block_size = vllm_config.cache_config.block_size
            assert self.cp_kv_cache_interleave_size == block_size, (
                f"DI with DCP requires --cp-kv-cache-interleave-size={block_size} "
                f"(must equal block_size), got {self.cp_kv_cache_interleave_size}"
            )
        # cp_rank computed lazily on first access (process groups not yet initialized)
        self._cp_rank: int | None = None

        # Parse num_batched_tokens_buckets from config or compute defaults
        # This must happen before InputBatch creation to ensure buffers are sized correctly
        from vllm_neuron.utils.bucket_utils import (
            get_default_kv_segment_size_buckets,
            get_default_num_seqs_buckets,
            resolve_num_batched_tokens_buckets,
            validate_decode_context_length_buckets,
            validate_kv_segment_size_buckets,
            validate_num_seqs_buckets,
        )

        # The two axes are independent, but only when the user says so. Resolve
        # the KV segment default first, then collapse the Q buckets onto it when
        # both are defaulted, so a defaulted config compiles exactly one
        # prefill target on the diagonal q_active == kv_segment_size. Taking the
        # power-of-two Q defaults *and* a multi-value segment default would hand
        # _prefill_compile_targets their cartesian product, which is what
        # timed out warmup on the shared-compile lock. DCP alignment still
        # applies to whichever bucket list is selected.
        user_set_num_batched_tokens_buckets = (
            "num_batched_tokens_buckets" in neuron_config_dict
        )
        user_set_kv_segment_size_buckets = (
            "kv_segment_size_buckets" in neuron_config_dict
        )
        default_kv_segment_size_buckets = get_default_kv_segment_size_buckets(
            self.max_num_batched_tokens, self.max_model_len
        )
        auto_num_batched_tokens_buckets = (
            default_kv_segment_size_buckets
            if not user_set_kv_segment_size_buckets
            else None
        )

        dcp_stride = (
            self._dcp_size * vllm_config.cache_config.block_size
            if self._dcp_size > 1
            else 1
        )
        self.neuron_config.num_batched_tokens_buckets = (
            resolve_num_batched_tokens_buckets(
                self.max_num_batched_tokens,
                configured_buckets=neuron_config_dict.get("num_batched_tokens_buckets"),
                use_configured_buckets=user_set_num_batched_tokens_buckets,
                auto_buckets=auto_num_batched_tokens_buckets,
                dcp_stride=dcp_stride,
            )
        )

        if user_set_num_batched_tokens_buckets:
            logger.info(
                "Using num_batched_tokens_buckets from config: %s",
                self.neuron_config.num_batched_tokens_buckets,
            )
        else:
            logger.info(
                "Using default num_batched_tokens_buckets: %s",
                self.neuron_config.num_batched_tokens_buckets,
            )

        # Parse num_seqs_buckets from config or compute defaults
        if "num_seqs_buckets" in neuron_config_dict:
            buckets_from_config = neuron_config_dict.get("num_seqs_buckets")
            self.neuron_config.num_seqs_buckets = validate_num_seqs_buckets(
                buckets_from_config, self.max_num_reqs
            )
            logger.info(
                "Using num_seqs_buckets from config: %s",
                self.neuron_config.num_seqs_buckets,
            )
        else:
            self.neuron_config.num_seqs_buckets = get_default_num_seqs_buckets(
                self.max_num_reqs
            )
            logger.info(
                "Using default num_seqs_buckets: %s",
                self.neuron_config.num_seqs_buckets,
            )

        # Parse decode_context_length_buckets — opt-in feature that compiles a
        # 2D bucket grid (batch, seq). Default None preserves today's
        # behavior of compiling decode at max_model_len.
        if "decode_context_length_buckets" in neuron_config_dict:
            self.neuron_config.decode_context_length_buckets = (
                validate_decode_context_length_buckets(
                    neuron_config_dict["decode_context_length_buckets"],
                    self.max_model_len,
                )
            )
            logger.info(
                "Using decode_context_length_buckets from config: %s",
                self.neuron_config.decode_context_length_buckets,
            )

        # kv_segment_size_buckets: user-provided or default-generated.
        #
        # attention_segmented_cte cannot read a non-packed FP8 K cache (see
        # validate_fp8_segmented_supported), so the two paths are treated
        # differently on purpose:
        #
        #   - explicit buckets -> raise. Asking for segmented prefill with a
        #     KV layout that cannot be read is unsatisfiable, and the fix
        #     (fp8_packed_kv=true) is one line. Silently serving full prefill
        #     instead would mean a config named "segmented prefill" quietly
        #     measures the non-segmented path -- accuracy and perf numbers
        #     attributed to the wrong kernel.
        #   - defaulted buckets -> fall back to full prefill. Nobody asked for
        #     segmented prefill here, so auto-enabling it into a hard warmup
        #     failure is our bug, not the user's (this was the #1904
        #     regression).
        if user_set_kv_segment_size_buckets:
            self.neuron_config.kv_segment_size_buckets = (
                validate_kv_segment_size_buckets(
                    neuron_config_dict.get("kv_segment_size_buckets")
                )
            )
            # validate_kv_segment_size_buckets either raises or returns a
            # non-empty list, so reaching here means segmented prefill was
            # explicitly requested.
            if self._is_non_packed_fp8_kv():
                raise ValueError(
                    "kv_segment_size_buckets="
                    f"{self.neuron_config.kv_segment_size_buckets} requests "
                    "segmented prefill, but the segmented attention kernel "
                    "cannot read a non-packed FP8 K cache (cache_dtype="
                    f"{self.vllm_config.cache_config.cache_dtype!r}, "
                    "neuron_config.fp8_packed_kv is unset). Set "
                    "'fp8_packed_kv': true in neuron_config to use segmented "
                    "prefill with an FP8 KV cache, or drop "
                    "kv_segment_size_buckets to run full prefill, which reads "
                    "dense BF16 K/V."
                )
            logger.info(
                "Using kv_segment_size_buckets from config: %s",
                self.neuron_config.kv_segment_size_buckets,
            )
        elif self._is_non_packed_fp8_kv():
            self.neuron_config.kv_segment_size_buckets = None
            logger.warning(
                "Not defaulting segmented prefill on: KV cache dtype %s is FP8 "
                "without the packed layout (neuron_config.fp8_packed_kv is "
                "unset), which the segmented attention kernel cannot read. "
                "Using full prefill. Set 'fp8_packed_kv': true to enable "
                "segmented prefill for an FP8 KV cache.",
                self.vllm_config.cache_config.cache_dtype,
            )
        else:
            self.neuron_config.kv_segment_size_buckets = default_kv_segment_size_buckets
            if default_kv_segment_size_buckets is None:
                logger.info(
                    "Segmented prefill left off by default: max_num_batched_tokens"
                    " (%s) is either >= max_model_len (%s), making prefill "
                    "single-shot, or not one of the supported segment sizes %s. "
                    "Set kv_segment_size_buckets explicitly to enable it.",
                    self.max_num_batched_tokens,
                    self.max_model_len,
                    sorted(SUPPORTED_KV_SEGMENT_SIZES),
                )
            else:
                logger.info(
                    "Using default kv_segment_size_buckets: %s (mirrors "
                    "max_num_batched_tokens; set kv_segment_size_buckets "
                    "explicitly for off-diagonal q_active/segment pairs)",
                    self.neuron_config.kv_segment_size_buckets,
                )

        # APC and DCP both require segmented prefill. APC needs it for
        # correctness: a prefix-cache hit means prefill has to attend prior KV
        # read back out of the cache, and the full-prefill path flash-attends
        # only the K/V it just projected, so the hit would silently be treated
        # as absent context. DCP requires it as a supported-configuration
        # policy: DCP is only exercised and validated on segmented prefill, so
        # refuse to start rather than serve an unvalidated pairing.
        #
        # Same precondition (kv_segment_size_buckets is None), different reason,
        # so share the check via a helper that names the requesting feature and
        # how to fix it. Why segmented prefill is off is a property of the
        # config, not of the requesting feature, so both call sites pass the
        # same evidence.
        segmented_off = self.neuron_config.kv_segment_size_buckets is None
        segmented_off_evidence = {
            "non_packed_fp8_kv": segmented_off and self._is_non_packed_fp8_kv(),
            "cache_dtype": self.vllm_config.cache_config.cache_dtype,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
        }
        if segmented_off and getattr(
            vllm_config.cache_config, "enable_prefix_caching", False
        ):
            raise ValueError(
                _segmented_prefill_required_msg(
                    feature="Automatic Prefix Caching (APC)",
                    disable_hint="disable APC with '--no-enable-prefix-caching'",
                    **segmented_off_evidence,
                )
            )
        if segmented_off and self._dcp_size > 1:
            raise ValueError(
                _segmented_prefill_required_msg(
                    feature=(
                        "Decode-context parallelism "
                        f"(--decode-context-parallel-size={self._dcp_size})"
                    ),
                    disable_hint=(
                        "disable DCP with '--decode-context-parallel-size 1'"
                    ),
                    **segmented_off_evidence,
                )
            )

        # Ensure max_num_batched_tokens is large enough for the largest prefill bucket
        # This is required because slot_mapping buffer is sized by max_num_batched_tokens
        max_bucket_size = max(self.neuron_config.num_batched_tokens_buckets)
        if self.max_num_batched_tokens < max_bucket_size:
            logger.warning(
                "max_num_batched_tokens (%s) is smaller than largest prefill bucket "
                "(%s). Adjusting to %s to ensure buffers are large enough for warmup "
                "compilation.",
                self.max_num_batched_tokens,
                max_bucket_size,
                max_bucket_size,
            )
            self.max_num_batched_tokens = max_bucket_size

        logger.debug("Neuron config: %s", self.neuron_config)

        # In-graph async-reorder inputs (future, permutation, mask) for the current
        # step. A value of None selects the plain decode graph. The execute path sets
        # it on each step when the reorder is active.
        self._async_reorder_kwargs = None
        # Cached answer to "does the loaded model consume the reorder arguments".
        # The runner resolves it on first use, because the model loads later.
        self._model_supports_async_reorder = None
        self.use_async_scheduling: bool = vllm_config.scheduler_config.async_scheduling
        if self.use_async_scheduling:
            # The on-device-sampling precondition is checked below, after the
            # mm_encoder_only construction role resolves. A vision-encoder-only
            # pool has no sampler and is exempt: its async output is the encoder
            # embedding write, not a sampled token.
            # The buffer to facilitate async execution.
            self.async_execution_buffer = dict()
            # Guards _apply_async_fallback_token_fixup's precondition: the async
            # output must have been materialized this step, or the host-side
            # token list it reads back is not there and the fixup would bail
            # silently, handing the caller stale tokens with no signal.
            self._async_output_materialized_this_step = False
            # Holds the most recent dummy-batch output so its NrtaFuture stays
            # referenced (preventing GC from freeing in-flight NRT input slices)
            # without a device->host copy. Superseded on each dummy batch. See
            # execute_dummy_batch.
            self._dummy_output_keepalive = None
            # Counters for observability: track how many forward steps used
            # the async path vs sync fallback. Logged periodically and useful
            # for tests to verify async scheduling is actually engaged.
            self._async_steps: int = 0
            self._sync_fallback_steps: int = 0
            self._batch_composition_changed: bool = False
            # Last accepted token tensor from prev spec step's rejection
            # sampler ([bs] int32 with default stride). Set at spec→non-spec
            # transition by reading
            # ``async_execution_buffer["futures_last_accepted_token"]``.
            # Consumed by ``_maybe_swap_async_input_ids`` as the non-spec
            # step's input_ids without slicing or stride manipulation.
            self._transition_bonus_tensor: torch.Tensor | None = None

        # Set up speculative decoding.
        self.drafter: EagleProposer | DFlashProposer | None = None
        self.is_eagle3_spec = False
        self.is_mtp_spec = False
        self.is_dflash_spec = False
        self._draft_token_ids = None
        # Async speculative decoding drafts rows by req_id. Only used at
        # batch-composition changes (e.g. prefills merging into bs-wide decode).
        # Steady-state decode goes through the cross-step future path in
        # async_execution_buffer.
        self._async_spec_transition_full_draft_cache: dict[str, torch.Tensor] = {}

        if self.speculative_config:
            if self.speculative_config.method == "eagle3":
                self.is_eagle3_spec = True
                self.drafter = EagleProposer(
                    self.vllm_config,
                    self.device,
                    self.on_device_sampling,
                )
            elif self.speculative_config.method == "mtp":
                # Same fused-draft flow as EAGLE3 (MTPProposer subclasses
                # EagleProposer); the target hands over its final hidden state.
                self.is_mtp_spec = True
                self.drafter = MTPProposer(
                    self.vllm_config,
                    self.device,
                    self.on_device_sampling,
                )
            elif self.speculative_config.method == "dflash":
                self.is_dflash_spec = True
                self.drafter = DFlashProposer(
                    self.vllm_config, self.device, self.on_device_sampling
                )
            else:
                raise ValueError(
                    f"Unsupported speculative decoding method: {self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler()

        # InputBatch is initialized in initialize_kv_cache once KV cache groups are known
        self.input_batch = None
        self._kv_snapshot_enabled: bool = False
        self._block_table_snapshot: list[torch.Tensor] | None = None
        # Host staging ring and last device upload for each KV-cache group, keyed by the
        # group index. The index is stable for the life of the runner, so the dict stays
        # small and it needs no eviction. See _block_table_device_tensor.
        self._block_table_groups = {}
        # Device tensors whose contents depend only on their key. See
        # _cached_constant_tensor for the rule that makes reuse safe.
        self._constant_tensors = {}
        # Host staging ring and last SWA result for each KV-cache group, keyed by the
        # group index. See _compute_swa_decode_tensors.
        self._swa_groups = {}
        # Cached result of the DP reduce identity proof. The flag separates "not resolved
        # yet" from a proven None. See _dp_reduce_identity_constant.
        self._dp_identity_resolved = False
        self._dp_identity_constant = None
        # Output of the previous dummy batch, held so the drain can read it back. See
        # _drain_previous_dummy_output.
        self._previous_dummy_output = None
        # Serialises InputBatch mutation between this thread and the async-output thread.
        # See _batch_mutation_lock.
        self._input_batch_lock = threading.Lock()
        self._encoder_cache_snapshot_enabled: bool = False
        self._encoder_cache_snapshot: dict[str, Any] | None = None
        # Input-snapshot capture: the shared runner hook (one capturer, built
        # lazily) installed only around this model's forward when enabled.
        self._snapshot_hook = SnapshotRunnerHook()
        # Caches the current step's _get_dp_padding all-reduce as
        # (reduce_inputs, dp_pad, max_decode_ctx_len), where reduce_inputs is
        # (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode). An
        # async-scheduling re-prep of the same batch reuses the result instead
        # of issuing a second cross-DP reduce (which would desync the
        # host-reduce/EP-forward collective streams across DP ranks and
        # deadlock); reduce_inputs lets the re-prep assert its inputs are
        # unchanged. Overwritten on every primary (non-reuse) prep.
        # Cache tuple: (reduce_inputs, dp_pad, max_decode_ctx_len,
        # group_force_nonspec, any_rank_jf). reduce_inputs =
        # (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode). The
        # async re-prep reuses dp_pad/max_decode_ctx_len/group_force_nonspec/
        # any_rank_jf/any_rank_spec_decode.
        self._dp_padding_cache: (
            tuple[tuple[int, int, bool], int, int, bool, bool, bool] | None
        ) = None
        # Per-step draft-lockstep state, set during input prep and read in
        # sample_tokens: whether the DP group speculates this step, the
        # DP-padded request count, and the DP-synced decode ctx length. Used to
        # dispatch a dummy draft on skip-propose ranks (see _dispatch_dummy_draft).
        self._group_spec_enabled: bool = False
        self._step_dummy_draft_batch: int = 0
        self._step_max_decode_ctx_len: int = 0
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # EPD: Vision encoder only construction gate and PD language-only construction gate.
        # Resolved by set_construction_role below; default False = monolith.
        self.mm_encoder_only: bool = False
        self.mm_language_model_only: bool = False

        # EPD construction-role flags ride on encoder_neuron_config (the VE and
        # PD pools always have one; text-only models have none and default to
        # the monolith both-False path). Must resolve before _init_encoder_cache
        # so min_hold_time_ms derives correctly (0 for monolithic, 100ms for EPD).
        if self.encoder_neuron_config is not None:
            self.encoder_neuron_config.set_construction_role(
                vllm_config.model_config.multimodal_config,
                vllm_config.additional_config,
            )
            self.mm_encoder_only = self.encoder_neuron_config.mm_encoder_only
            self.mm_language_model_only = (
                self.encoder_neuron_config.mm_language_model_only
            )
        if self.mm_encoder_only:
            logger.info("EPD: mm_encoder_only=True - building encoder-only model")
        if self.mm_language_model_only:
            logger.info(
                "EPD: mm_language_model_only=True - building language-only model"
            )

        # Async execution needs an on-device output that no host read forces.
        # A sampling pool meets this with on-device sampling. A vision-encoder-only
        # pool meets it with the encoder embedding write to the on-device cache, so
        # it is exempt. Reject async on a sampling pool that has no on-device
        # sampling. Enable on-device sampling, or set async_scheduling=False.
        if (
            self.use_async_scheduling
            and not self.on_device_sampling
            and not self.mm_encoder_only
        ):
            raise RuntimeError(
                "Async execution needs on-device sampling. Enable on-device "
                "sampling, or set async_scheduling=False."
            )

        # On-device encoder cache for multimodal models.
        # The cache is modality-neutral: it stores encoder embeddings of
        # fat_dim width. The size comes from the encoder config, which the
        # encoder_neuron_config carries. Build the cache only when that config
        # is present.
        self.encoder_cache = None
        if self.supports_mm_inputs and self.encoder_neuron_config is not None:
            self._init_encoder_cache(vllm_config)

        # EPD EC (encoder-cache) transfer connector, worker-role. Built from
        # ec_transfer_config; None on the default monolith path (no ec_role).
        # A consumer (PD) uses it to pull vision embeddings over NIXL; a producer
        # (VE) registers its cache region + serves the handshake.
        self.ec_connector = self._init_ec_connector(vllm_config)

        # Ephemeral state for execute_model() -> sample_tokens() flow
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        # Full (2, ...) KV cache tensors for DI connector registration
        self._kv_cache_full_tensors: dict[str, torch.Tensor] = {}

        # Initialize vLLM Sampler for proper token sampling
        logprobs_mode = getattr(
            vllm_config.model_config, "logprobs_mode", "raw_logprobs"
        )
        self.sampler = Sampler(logprobs_mode=logprobs_mode)

        # Cache numpy arange for efficient tensor operations
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_batched_tokens),
            dtype=np.int64,
        )

        # We pass rank as a tensor input to avoid it becoming a constant
        # TODO: Update dist.get_rank() to avoid becoming a constant during lowering, and remove this
        # Use TP-local rank (not world rank) so DP doesn't affect model computation
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        self.rank_tensor = torch.tensor(tp_rank, dtype=torch.int32, device=self.device)

        # SPMD: stamp each parallel group's rank as an on-device tensor so the model
        # traces one rank-invariant graph. Held to keep the tensors alive.
        self._spmd_rank_tensors = self._build_spmd_ranks()

        self.lm_head_dp_size = self.neuron_config.lm_head_dp_size
        self.lm_head_tp_rank = tp_rank
        self.lm_head_tp_size = tp_group.world_size if tp_group else 1
        self.lm_head_dp_group = None
        # The dedicated lm_head TP/DP GroupCoordinators are only created by
        # initialize_neuron_parallel_state() when lm_head_dp_size > 1 (their
        # getters assert on that). For the common lm_head_dp_size == 1 case the
        # lm_head TP group is just the regular TP group, so the tp_group-derived
        # defaults above are already correct — fetching the dedicated groups
        # would raise. Only reach for them when lm_head DP is actually enabled.
        if self.on_device_sampling and self.lm_head_dp_size > 1:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_lm_head_dp_group,
                get_neuron_lm_head_tp_group,
            )

            lm_head_tp_group = get_neuron_lm_head_tp_group()
            self.lm_head_tp_rank = lm_head_tp_group.rank_in_group
            self.lm_head_tp_size = lm_head_tp_group.world_size
            self.lm_head_dp_group = get_neuron_lm_head_dp_group()

        # block_size is logged in initialize_kv_cache (post-override). At this
        # point in __init__ it may still hold vLLM's pre-override default.
        logger.info(
            "Initialized NeuronModelRunner with config: max_num_reqs=%s, "
            "vocab_size=%s, max_model_len=%s, device=%s",
            self.max_num_reqs,
            self.vocab_size,
            self.max_model_len,
            self.device,
        )

    def _init_encoder_cache(self, vllm_config) -> None:
        """Initialize the on-device encoder cache for multimodal models.

        Derives cache sizing from the scheduler's encoder budget to stay in
        sync with EncoderCacheManager. Sets self.encoder_cache and
        self.max_encoder_blocks_per_request.
        """
        from vllm.multimodal.encoder_budget import MultiModalBudget
        from vllm_neuron.utils.mm_utils import (
            get_encoder_cache_block_size,
            get_max_encoder_blocks_per_request,
        )
        from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

        encoder_neuron_config = self.encoder_neuron_config
        model_config = vllm_config.model_config

        # fat_dim = encoder_dim * (1 + num_deepstack_levels)
        # This is the embedding width stored per token in the cache buffer.
        hf_config = model_config.hf_config
        vis_config = getattr(hf_config, "vision_config", None)
        if vis_config is not None:
            encoder_dim = getattr(
                vis_config, "out_hidden_size", model_config.get_hidden_size()
            )
            deepstack_indexes = (
                getattr(vis_config, "deepstack_visual_indexes", None) or []
            )
            num_deepstack = len(deepstack_indexes)
        else:
            encoder_dim = model_config.get_hidden_size()
            num_deepstack = 0
        fat_dim = encoder_dim * (1 + num_deepstack)
        cache_block_size = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )

        # Max encoder-cache blocks one request can use, from the largest bucket.
        # The model converts the bucket to a block count per modality. Audio
        # buckets are block counts. Vision buckets are raw pre-merge tokens. This
        # shares one unit conversion with get_max_encoder_embed_budget.
        max_bucket = max(encoder_neuron_config.num_encoder_input_buckets)
        self.max_encoder_blocks_per_request = get_max_encoder_blocks_per_request(
            hf_config, encoder_neuron_config
        )

        # Derive min_blocks_needed from scheduler's encoder_cache_size:
        #   encoder_cache_size = max(max_num_batched_tokens, max_single_mm_item_tokens)
        # This is the max merged encoder tokens the scheduler can have cached
        # simultaneously (set in vllm.v1.core.encoder_cache_manager).
        # We auto-derive from this value to stay in sync with the scheduler's
        # EncoderCacheManager — both use the same budget so the worker never
        # runs out of blocks for items the scheduler considers cacheable.
        # We divide by cache_block_size to get blocks, plus one scratch block
        # that absorbs encoder padding writes and is never allocated to real items.
        #
        # TODO: The scheduler's EncoderCacheManager sizes and evicts based on
        # token counts, unaware of block-level padding waste. This mismatch
        # means the scheduler may cache more items than the block allocator
        # can hold (each item uses a full block regardless of token count).
        # A proper fix requires integrating block layout awareness into the
        # scheduler's EncoderCacheManager. Until then, set encoder_neuron_config.encoder_cache_num_blocks
        # explicitly for workloads with many small images.
        num_blocks = encoder_neuron_config.encoder_cache_num_blocks
        mm_budget = MultiModalBudget(vllm_config, self.mm_registry)
        min_blocks_needed = (
            math.ceil(mm_budget.encoder_cache_size / cache_block_size) + 1
        )
        # Use min_blocks_needed to init encoder cache blocks if user did not specify
        # or specify smaller value
        if num_blocks is None or num_blocks < min_blocks_needed:
            logger.info(
                "encoder_cache_num_blocks: %s → %d "
                "(derived from scheduler encoder_cache_size=%d "
                "merged tokens = %d blocks + 1 scratch)",
                num_blocks,
                min_blocks_needed,
                mm_budget.encoder_cache_size,
                min_blocks_needed - 1,
            )
            num_blocks = min_blocks_needed

        if self.max_encoder_blocks_per_request > num_blocks - 1:
            raise ValueError(
                f"encoder_cache_num_blocks={num_blocks} is too small: "
                f"a single request needs up to {self.max_encoder_blocks_per_request} "
                f"blocks (largest encoder bucket={max_bucket}, "
                f"cache_block_size={cache_block_size}), but only "
                f"{num_blocks - 1} allocatable blocks available. "
                f"Increase encoder_cache_num_blocks or reduce "
                f"num_encoder_input_buckets."
            )

        min_hold_time_ms = encoder_neuron_config.encoder_cache_min_hold_time_ms or 0.0

        self.encoder_cache = EncoderCacheBlocks(
            num_blocks=num_blocks,
            block_size=cache_block_size,
            fat_dim=fat_dim,
            dtype=torch.bfloat16,
            device=self.device,
            min_hold_time_ms=min_hold_time_ms,
            model_id=model_config.model,
        )
        logger.info(
            "On-device encoder cache: num_blocks=%d, block_size=%d, "
            "fat_dim=%d, max_encoder_blocks_per_request=%d, buffer_size_mb=%.1f",
            num_blocks,
            cache_block_size,
            fat_dim,
            self.max_encoder_blocks_per_request,
            num_blocks * cache_block_size * fat_dim * 2 / (1024 * 1024),
        )

    @property
    def is_spec_decode(self) -> bool:
        return self.is_eagle3_spec or self.is_mtp_spec or self.is_dflash_spec

    @property
    def cp_rank(self) -> int:
        if self._cp_rank is None:
            if self.cp_world_size <= 1:
                self._cp_rank = 0
            else:
                from vllm.distributed.parallel_state import get_tp_group

                tp_rank = get_tp_group().rank_in_group
                # Prefill and decode use the SAME DCP scheme now: cp_rank is
                # the position within the DCP group (which interleaved token
                # slice this rank owns).
                self._cp_rank = tp_rank % self._dcp_size
        return self._cp_rank

    def _build_spmd_ranks(self) -> list[torch.Tensor]:
        """# Stamp each parallel group's rank_in_group onto it as an on-device tensor, for SPMD compatibility."""

        return stamp_spmd_rank_tensors(self.device)

    def _setup_capture(
        self, compiled_model, modules, subdirectory="", capture_filter=None
    ):
        """Set up tensor capture for a compiled model.

        Args:
            compiled_model: The torch.compile'd model (hooks go on _orig_mod,
                clear hook goes on the compiled wrapper).
            modules: Module name patterns to capture (can be empty for inline-only).
            subdirectory: Subdirectory under capture_dir (e.g. "draft").
            capture_filter: Optional set of tensor names to write.

        Returns:
            ModelCapture instance, or None if no capture config.
        """
        from vllm_neuron.accuracy.tensor_capture import ModelCapture

        capture_config = self.neuron_config.tensor_capture
        if not capture_config:
            return None

        try:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        except Exception:
            rank = 0

        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        tp_rank = rank % tp_size

        raw_model = getattr(compiled_model, "_orig_mod", compiled_model)
        capture = ModelCapture(
            model=raw_model,
            modules=modules,
            capture_dir=capture_config.capture_dir,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            capture_filter=capture_filter,
            subdirectory=subdirectory,
        )
        capture.register_clear_hook(compiled_model)
        return capture

    def _reset_state(self) -> None:
        """
        Reset internal state to clean state.

        This is called after warmup to clear dummy requests and prepare
        the model runner for real inference.
        """
        # Remove all requests from input_batch
        for req_id in list(self.input_batch.req_id_to_index.keys()):
            self.input_batch.remove_request(req_id)
        self.input_batch.condense()

        # Clear request cache
        self.requests.clear()

        logger.debug("Model runner state reset successfully")

    def _get_grammar_bitmask(
        self,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor | None:
        """
        Get grammar bitmask from scheduler_output for structured outputs.

        Builds a full logit-row-aligned bitmask by reordering the compact bitmask
        (which only contains rows for structured output requests) to match the
        order of requests in input_batch. Non-SO rows are filled with -1 (all tokens
        allowed). Handles speculative decode offsets.

        This mirrors vLLM's apply_grammar_bitmask() reordering logic but adapted
        for on-device sampling where the mask is passed into the model forward.

        Args:
            scheduler_output: Output from scheduler containing:
                - _grammar_bitmask: Compact numpy array [num_so_reqs, packed_vocab_size]
                - _structured_output_request_ids: List of request IDs with SO

        Returns:
            CPU boolean tensor [num_logit_rows, vocab_size] or None if no bitmask
        """
        packed_bitmask_np = getattr(scheduler_output, "_grammar_bitmask", None)
        struct_req_ids = getattr(
            scheduler_output, "_structured_output_request_ids", None
        )

        if packed_bitmask_np is None:
            logger.debug("[SO] model_runner: no grammar bitmask in scheduler_output")
            return None

        # Guard: bitmask present but request IDs missing indicates scheduler bug
        if not struct_req_ids:
            logger.warning(
                "[SO] Grammar bitmask present but structured_output_request_ids missing. "
                "This indicates a scheduler invariant violation."
            )
            return None

        # Convert numpy to torch on CPU once
        packed_bitmask = torch.from_numpy(packed_bitmask_np).to(torch.int32)
        # Rows = logit rows (SO reqs + spec), NOT prompt tokens / prefill bucket
        num_compact_rows = packed_bitmask.shape[0]
        packed_vocab_size = packed_bitmask.shape[1]

        # Build mapping: req_id -> logit_index (accounting for spec tokens)
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        struct_out_req_ids_set = set(struct_req_ids) if struct_req_ids else set()

        for batch_index, req_id in enumerate(self.input_batch.req_ids):
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(spec_tokens.get(req_id, ()))
            if req_id in struct_out_req_ids_set:
                struct_out_req_batch_indices[req_id] = logit_index

        # Calculate total logit rows (requests + spec tokens)
        num_logit_rows = len(self.input_batch.req_ids) + sum(
            len(spec_tokens.get(r, ())) for r in self.input_batch.req_ids
        )

        # Full bitmask: -1 (all allowed) for non-SO rows, aligned to logit rows
        sorted_bitmask = torch.full(
            (num_logit_rows, packed_vocab_size),
            fill_value=-1,
            dtype=torch.int32,
        )

        # Reorder: copy SO request masks to correct positions
        cumulative_index = 0
        for req_id in struct_req_ids:
            num_spec = len(spec_tokens.get(req_id, ()))
            if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
                for i in range(1 + num_spec):
                    bitmask_index = logit_idx + i
                    sorted_bitmask[bitmask_index] = packed_bitmask[cumulative_index + i]
            cumulative_index += 1 + num_spec

        # Sanity check: ensure we consumed all rows in packed bitmask
        # Use explicit check instead of assert (asserts can be optimized away with -O)
        if cumulative_index != num_compact_rows:
            raise RuntimeError(
                f"Bitmask alignment error: consumed {cumulative_index} rows but "
                f"packed_bitmask has {num_compact_rows} rows. "
                "This indicates a mismatch between scheduler and model_runner."
            )

        # Unpack the bitmask: int32 -> 32 boolean bits per value
        packed_uint = sorted_bitmask.view(num_logit_rows, packed_vocab_size, 1)

        bit_positions = torch.arange(32, dtype=torch.int32)

        unpacked = ((packed_uint >> bit_positions) & 1).view(num_logit_rows, -1)
        unpacked = unpacked[:, : self.vocab_size]
        bitmask = unpacked.bool()

        # Keep the full mask on CPU here. CPU sampling applies it to full CPU
        # logits later; ODS prepares the lm-head-local device mask after decode
        # padding so the compiled model does not need the full-vocab mask.

        # Log bitmask statistics for structured output debugging
        num_allowed = bitmask.sum().item()
        total_tokens = bitmask.numel()
        pct_allowed = 100 * num_allowed / total_tokens if total_tokens > 0 else 0

        logger.debug(
            "[SO] model_runner: bitmask shape=%s, allowed=%d/%d (%.1f%%), reqs=%s",
            bitmask.shape,
            num_allowed,
            total_tokens,
            pct_allowed,
            list(struct_out_req_ids_set),
        )

        return bitmask

    def _apply_grammar_bitmask_cpu(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply grammar bitmask to logits on CPU for structured output constraints.

        Used in the CPU sampling path where logits return from the device to CPU
        before sampling. This is the CPU-side equivalent of the on-device
        masked_fill performed by the vLLM Neuron sampler.

        Args:
            logits: [num_logit_rows, vocab_size] logits tensor on CPU.
            bitmask: [num_logit_rows, vocab_size] boolean mask on CPU.
                     True = token allowed, False = token disallowed (set to -inf).

        Returns:
            Logits tensor with disallowed tokens set to -inf.
        """
        num_allowed = bitmask.sum().item()
        total_tokens = bitmask.numel()
        pct_allowed = 100.0 * num_allowed / total_tokens if total_tokens > 0 else 0.0
        logger.debug(
            "[SO] CPU bitmask applied: logits=%s, allowed=%d/%d (%.1f%%)",
            logits.shape,
            num_allowed,
            total_tokens,
            pct_allowed,
        )
        return logits.masked_fill(~bitmask, float("-inf"))

    def _compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap the model with torch.compile, with the arguments of load_model."""
        from vllm_neuron.envs import (
            get_compile_backend_name,
            get_precompile_guard_filter_fn,
        )

        # torch.compile pops guard_filter_fn from options, so build a fresh dict.
        guard_filter_fn = get_precompile_guard_filter_fn()
        guard_opt = {"guard_filter_fn": guard_filter_fn} if guard_filter_fn else {}
        return torch.compile(
            model,
            backend=get_compile_backend_name(),
            fullgraph=self._fullgraph_enabled,
            options={**self.compile_options, **guard_opt},
        )

    def redecorate_compiled_model(self) -> None:
        """Wrap the model, its encoder, and the drafter again, so fresh wrappers load the package.

        The first wrappers load their precompile packages at model load, before the
        trace lanes write them. A fresh wrapper dispatches each warmup bucket, in
        place of a re-trace.

        Tensor capture clears its registry from a pre-hook on the wrapper, so move
        that hook to each fresh wrapper. Without it the registry grows on every
        forward, and the compiled entries, which guard on its length, stop matching.

        Examples:
            >>> runner.redecorate_compiled_model()  # doctest: +SKIP
        """
        inner = getattr(self.model, "_orig_mod", self.model)
        self.model = self._compile_model(inner)
        target_capture = getattr(self, "_target_tensor_capture", None)
        if target_capture is not None:
            target_capture.register_clear_hook(self.model)
        attr = getattr(self, "_compiled_encoder_attr", None)
        if attr is not None:
            encoder = getattr(inner, attr)
            fresh_encoder = self._compile_model(getattr(encoder, "_orig_mod", encoder))
            setattr(inner, attr, fresh_encoder)
            encoder_capture = getattr(self, "_encoder_tensor_capture", None)
            if encoder_capture is not None:
                encoder_capture.register_clear_hook(fresh_encoder)
        drafter = getattr(self, "drafter", None)
        if drafter is not None:
            drafter.model = torch.compile(
                getattr(drafter.model, "_orig_mod", drafter.model),
                **drafter.compile_kwargs,
            )
            draft_capture = getattr(self, "_draft_tensor_capture", None)
            if draft_capture is not None:
                draft_capture.register_clear_hook(drafter.model)

    def load_model(self) -> None:
        """Load the NeuronModel."""
        logger.info("Starting NeuronModel loading process")

        model_cls, _ = ModelRegistry.resolve_model_cls(
            self.vllm_config.model_config.architecture,
            model_config=self.vllm_config.model_config,
        )

        # TODO: validate and ensure all vLLM configs are complied to

        # TODO: remove assert once eager mode on Neuron is supported
        cpu_mode = envs.VLLM_NEURON_CPU_MODE
        cpu_compile = envs.VLLM_NEURON_CPU_COMPILE
        eager_mode = self.vllm_config.model_config.enforce_eager
        from libtorch_neuronx_lite import envs as libtorch_envs

        skip_graph_capture_backend = (
            libtorch_envs.NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND
        )
        assert not (eager_mode and not cpu_mode), (
            "Eager mode on Neuron is not yet supported."
        )
        if eager_mode:
            torch.compiler.set_stance("force_eager")

        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        self._is_synthetic_model = issubclass(model_cls, SyntheticNeuronModel)
        if self._is_synthetic_model:
            self.model = model_cls.from_configs(
                self.vllm_config.model_config.hf_config,
            )
            logger.info(
                "SyntheticNeuronModel — skipping weight loading and compilation"
            )
            self._target_tensor_capture = None
            return

        with torch.device("meta"):
            if self.encoder_neuron_config is not None:
                # Init multimodal model with encoder_neuron_config if provided
                self.model = model_cls.from_configs(
                    hf_config=self.vllm_config.model_config.hf_config,
                    text_neuron_config=self.neuron_config,
                    encoder_neuron_config=self.encoder_neuron_config,
                )
            else:
                # Use only (text) neuron_config to init model. Works for both text-only and multimodal model
                self.model = model_cls.from_configs(
                    self.vllm_config.model_config.hf_config,
                    self.neuron_config,
                )

        # Generic --convert seam: retype any *ForCausalLM into a pooling model
        # here. See pooling_adapter.as_neuron_pooling_model.
        if (
            self.vllm_config.model_config.runner_type == "pooling"
            and not self._is_synthetic_model
        ):
            from vllm_neuron.model.pooling_adapter import as_neuron_pooling_model

            self.model = as_neuron_pooling_model(
                self.model,
                convert_type=self.vllm_config.model_config.convert_type,
            )

        # vLLM sets uses_mrope from the HF rope_scaling. Use M-RoPE only when
        # the model also implements the SupportsMRoPE protocol. A model that
        # declares mrope in its config but implements a standard 1-D rope (for
        # example the Qwen3-ASR text decoder) must use standard positions.
        unwrapped_model = getattr(self.model, "_orig_mod", self.model)
        if self.uses_mrope and not isinstance(unwrapped_model, SupportsMRoPE):
            logger.info(
                "Model %s declares M-RoPE in its config but does not implement "
                "SupportsMRoPE. Use standard positions.",
                type(unwrapped_model).__name__,
            )
            self.uses_mrope = False

        # TODO: Load LORA model wrapper if LORA config exists
        # Why required: Required to actually load LORA adapters onto the model
        # Current: Base model loaded without LORA support
        # Target: if self.lora_config: self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        # Load weights - let the model handle checkpoint loading
        if not cpu_compile:
            logger.info("Loading weights from %s", self.vllm_config.model_config.model)
            weight_load_start_time = time.perf_counter()
            self.model.load_weights(
                self.vllm_config.model_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )
            logger.info(
                "Weight loading completed in %.2f seconds",
                time.perf_counter() - weight_load_start_time,
            )

            # Move model to Neuron device
            logger.info("Moving model to device: %s", self.device)
            self.model = self.model.to(self.device)

            model_name = self.vllm_config.model_config.model
            MODEL_LOAD_TIME.labels(model_name=model_name).set(
                time.perf_counter() - weight_load_start_time,
            )
            MODEL_LOAD_SIZE.labels(model_name=model_name).set(
                sum(p.nbytes for p in self.model.parameters()),
            )

            # EPD per-model footprint: log the per-tower bytes on every load
            # (confirms the gate worked), then fail loud if the loaded towers
            # don't match the construction role — a wrong flag (e.g. a VE pool
            # that still loaded the language model) silently wastes HBM.
            inner = getattr(self.model, "_model", self.model)
            if hasattr(inner, "mm_encoder_only"):
                from vllm_neuron.model.qwen3_vl.model_bf16 import (
                    per_tower_param_bytes,
                )

                stats = per_tower_param_bytes(inner)
                visual_bytes = stats["visual_param_bytes"]
                lm_bytes = stats["language_model_param_bytes"]
                logger.info(
                    "EPD footprint: mm_encoder_only=%s mm_language_model_only=%s "
                    "visual_param_bytes=%d language_model_param_bytes=%d",
                    inner.mm_encoder_only,
                    inner.mm_language_model_only,
                    visual_bytes,
                    lm_bytes,
                )
                if inner.mm_encoder_only and lm_bytes > 0:
                    raise ValueError(
                        "mm_encoder_only (VE) pool loaded language-model "
                        f"weights ({lm_bytes} bytes); the construction gate "
                        f"failed to skip the language model "
                        f"(visual={visual_bytes} bytes)."
                    )
                if inner.mm_language_model_only and visual_bytes > 0:
                    raise ValueError(
                        "mm_language_model_only (PD) pool loaded vision-tower "
                        f"weights ({visual_bytes} bytes); the construction gate "
                        f"failed to skip the vision tower "
                        f"(language_model={lm_bytes} bytes)."
                    )
        else:
            if hasattr(self.model, "load_weights_lite"):
                self.model.load_weights_lite(
                    self.vllm_config.model_config.model,
                    torch.device("cpu"),  # cpu needed getting compile time constants
                    self.vllm_config.load_config.download_dir,
                )
                logger.info("Light weight loading complete for CPU compilation.")
            logger.info("CPU Compilation is enabled. Skipping full weight loading.")
            # Explicit call to force all buffers to use meta.
            self.model = self.model.to("meta")

        if self.is_spec_decode:
            if _supports_aux_hidden_states(self.model):
                if self.is_dflash_spec or self.is_mtp_spec:
                    assert isinstance(self.drafter, (DFlashProposer, MTPProposer))
                    aux_layers = self.drafter.get_target_aux_hidden_state_layers()
                    logger.info(
                        "Using %s auxiliary layers: %s",
                        self.speculative_config.method,
                        aux_layers,
                    )
                else:
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = self.model.get_eagle3_aux_hidden_state_layers()
                        logger.info(
                            "Using model's default auxiliary layers: %s", aux_layers
                        )
                self.model.set_aux_hidden_state_layers(aux_layers)
            else:
                raise RuntimeError(
                    "Model does not support auxiliary hidden-state capture "
                    "but speculative decoding was requested"
                )

        self._target_tensor_capture = None
        self._draft_tensor_capture = None
        self._encoder_tensor_capture = None

        # Compile model with vllm_neuron backend for Neuron execution
        # This is where torch.compile uses the registered "vllm_neuron" backend
        logger.info("Compiling model with vllm_neuron backend")

        # Disable dynamo cache limit — recompilations are caught at runtime
        # by FailOnRecompileLimitHit, so a static limit adds no value.
        torch._dynamo.config.cache_size_limit = 2**62

        # Build compile options
        self.compile_options = {"alias_meta_to_neuron": True}

        # FIXME: -O1 and mac-threshold are temporary until NKI adds MAC count estimates for kernels.
        hlo2tensorizer_opts = "--modular-flow-mac-threshold=10"
        # The unsafe fp8 cast flag is only needed on Trn2 where kernels use
        # legacy nl.float8_e4m3 (max=240). Trn3 supports OCP e4m3fn natively.
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        has_fp8 = self.vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
        if not has_fp8 and self.vllm_config.model_config.quantization in ("modelopt",):
            has_fp8 = True
        if not has_fp8:
            has_fp8 = getattr(self.neuron_config, "quantization", None) == "fp8"
        if has_fp8 and get_platform_target() not in ("trn3", "trn3pre"):
            hlo2tensorizer_opts += " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
        # The native GK decode graph takes XLA's compiler inputs (NF-351). Stop hlo2penguin
        # tagging the sampling-params all-gather as FSDP; it must join this string, since a
        # second --internal-hlo2tensorizer-options replaces the first. And override the
        # --model-type transformer torch-neuronx adds; neuronx-cc keeps the last one.
        native_gk_decode = (
            envs.is_native_backend()
            and not cpu_mode
            and getattr(self.model, "use_stacked_kv_cache", False)
        )
        if native_gk_decode:
            hlo2tensorizer_opts += " --enable-annotate-fsdp-cc=false"
        # vLLM optimization levels map 1:1 onto neuronx-cc optlevels (CHRS-721).
        # Opt-level precedence, highest first:
        #   1. Explicit --optimization-level from the caller (CLI or offline API)
        #   2. A model's compiler_args_for(graph_type, example_inputs, base)
        #   3. The neuronx-cc default (O1 on Neuron)
        # The base args use the caller level. Prefill, jump-forward, and the
        # spec-verify target use these. A model's compiler_args_for can
        # override the args per graph, for example a higher -O for the plain
        # decode graph and the vision graph. -O3 disables neuronx-cc modular
        # flow, which improves the latency of these graphs.
        default_opt_level = self.vllm_config.optimization_level.value
        base_model_for_opt = getattr(self.model, "_orig_mod", self.model)
        # platform.apply_config_platform_defaults records this signal; it is the
        # only point where an explicit level is still distinguishable from the
        # default. Absent (older config) -> treat as not explicit.
        opt_level_explicit = getattr(
            self.vllm_config, "_neuron_optimization_level_explicit", False
        )

        self.compile_options["compiler_args"] = [
            "--auto-cast=none",
            "--verbose=35",
            f"-O{default_opt_level}",
            f"--internal-hlo2tensorizer-options={hlo2tensorizer_opts}",
            # --enable-nested-dynamic-loop: NKI kernels may contain nested
            # dynamic loops (scf.for inside scf.for). After inline_nki_kernel
            # inlines kernel BIR into the parent graph, birverifier rejects the
            # resulting multiple back-edges without this flag.
            "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
        ]
        if native_gk_decode:
            self.compile_options["compiler_args"].append("--model-type=generic")
        # An explicit caller level wins over the model default. Enable the model
        # policy only when the caller did not set the level. The runner then
        # resolves compiler_args_for eagerly, one time per graph, before the
        # cache hash, and carries the arguments to the compiler (on the trace
        # job for the parallel path, through a libtorch scope in-process). Stamp
        # the base arguments on the model so its own in-process sites (the
        # vision execute in embed_multimodal) can resolve the same policy.
        policy = getattr(base_model_for_opt, "compiler_args_for", None)
        per_graph = not opt_level_explicit and policy is not None
        self._per_graph_compiler_args = per_graph
        self._compiler_args_policy = policy if per_graph else None
        if per_graph:
            base_model_for_opt._neuron_base_compiler_args = list(
                self.compile_options["compiler_args"]
            )
        # Custom mega-decode DMA-order schedule. ``VLLM_NEURON_MEGA_DMA_ORDER_DERIVE=1`` emits the
        # compiler's own order; ``..._CONSTRAINTS_FILE=<json>`` applies one derived on THIS graph.
        if envs.VLLM_NEURON_MEGA_DMA_ORDER_DERIVE:
            self.compile_options["compiler_args"] += [
                '--internal-dma-order-constraints={"timing_mode":2}',
                "--internal-bir-enable-sb-allocation",
            ]
            logger.info("mega decode DMA-order DERIVE mode (timing_mode:2)")
        else:
            _dma_order_file = (
                envs.VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE or ""
            ).strip()
            if _dma_order_file:
                _dma_order_file = os.path.abspath(_dma_order_file)
                for _i, _arg in enumerate(self.compile_options["compiler_args"]):
                    if _arg.startswith("--internal-backend-options="):
                        self.compile_options["compiler_args"][_i] = (
                            _arg + f" --dma-order-constraint-file={_dma_order_file}"
                        )
                        break
                # nkilib pairs the constraint file with the BIR SBUF allocator, so the pass reads
                # the file's own ``sbuf_alloc_limit``. This is a top-level neuronx-cc argument, so
                # it needs its own entry and does NOT go inside --internal-backend-options.
                _sb_alloc = envs.VLLM_NEURON_MEGA_SBUF_ALLOC
                if _sb_alloc:
                    self.compile_options["compiler_args"] += [
                        "--internal-bir-enable-sb-allocation"
                    ]
                # The constraint pass reports each bind at INFO. neuronx-cc clamps the backend
                # logfile to ERROR unless this argument is explicit, which hides every bind.
                self.compile_options["compiler_args"] += ["--logfile-verbose=20"]
                logger.info(
                    "mega decode DMA-order constraints file: %s (sb-allocation=%s)",
                    _dma_order_file,
                    "on" if _sb_alloc else "off",
                )
        logger.info(
            "neuronx-cc base optlevel -O%s; explicit=%s, per-graph policy=%s",
            default_opt_level,
            opt_level_explicit,
            "on" if per_graph else "off",
        )

        # NOTE: compile_options is shared by the prefill and decode NEFFs. The
        # retained per-layer GK prefill kernels are grid[2] (lnc=2). A whole compile
        # at lnc=1 therefore assumes a GK deployment runs decode only. Such a
        # deployment traces prefill separately. Per-NEFF lnc is a follow-up.
        lnc = get_logical_nc_config()
        logger.info("Compiling at logical NC config %s", lnc)
        if getattr(self.model, "use_stacked_kv_cache", False) and lnc != 1:
            raise ValueError(
                f"GPT-OSS giga-kernel decode requires lnc=1, but "
                f"NEURON_LOGICAL_NC_CONFIG is {lnc}. Set it to 1."
            )
        # neuronx-cc defaults the flag to 2 while NRT reads NEURON_LOGICAL_NC_CONFIG,
        # so an implicit flag makes a non-default degree fail at load, not at compile.
        # The helper returns None at the default degree: the compile cache key covers
        # the compiler args, so an unconditional flag would change the key of every
        # LNC-2 graph and invalidate each cached NEFF. V2 calls the same helper.
        lnc_arg = get_explicit_compiler_lnc_arg(lnc)
        if lnc_arg is not None:
            self.compile_options["compiler_args"].append(lnc_arg)

        # Check for debug mode environment variable
        # Set VLLM_NEURON_DEBUG_MODE=1 to disable fullgraph for debugging (allows print statements)
        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        fullgraph_enabled = not debug_mode

        if debug_mode:
            logger.info(
                "VLLM_NEURON_DEBUG_MODE enabled: fullgraph=False (allows print statements and graph breaks)"
            )

        # We should fail explicitly for graph cuts (unless debug mode is enabled).
        # Note: When eager mode is enabled, torch compile becomes a no op.

        # A single torch.compile per backend. The plain single-token decode
        # graph compiles with the "decode" graph type (see _plain_decode_forward
        # / _plain_decode_capture). The model compiler_args_for maps that type
        # to its arguments. The encoder graph uses the "encoder" type. Prefill
        # and JF keep the base args on self.model.
        if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
            logger.debug("Graph capture and parallel compilation is disabled.")
            self.capture_backend_model = None
        else:
            # No guard filter here: the graph-capture backend holds a lazy
            # torch.compile that the native precompile route never calls.
            self.capture_backend_model = torch.compile(
                self.model,
                backend="neuron_libtorch_graph_capture",
                fullgraph=fullgraph_enabled,
                options=self.compile_options,
            )

        self._fullgraph_enabled = fullgraph_enabled
        self.model = self._compile_model(self.model)

        # Compile the multimodal encoder separately. It runs from
        # embed_multimodal, not from the main text model forward graph, so the
        # forward compile above does not cover it. The encoder attribute is
        # modality-specific: "visual" for VL, "audio_tower" for ASR. Compile and
        # reassign the SAME attribute the model calls, so embed_multimodal's
        # self.<attr>(...) reaches the compiled graph instead of running eager.
        inner_model = getattr(self.model, "_orig_mod", self.model)
        # VL names its encoder "visual"; ASR names it "audio_tower". TODO: read
        # the name through get_encoder(), which the worker already uses, so one
        # resolver serves each site. The reassignment below needs the attribute
        # name, so get_encoder() alone does not cover this site yet.
        encoder_attr = (
            "visual"
            if getattr(inner_model, "visual", None) is not None
            else "audio_tower"
            if getattr(inner_model, "audio_tower", None) is not None
            else None
        )
        if self.encoder_neuron_config is not None and encoder_attr is not None:
            encoder = getattr(inner_model, encoder_attr)

            if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
                self.encoder_capture_backend = None
            else:
                self.encoder_capture_backend = torch.compile(
                    encoder,
                    backend="neuron_libtorch_graph_capture",
                    fullgraph=fullgraph_enabled,
                    options=self.compile_options,
                )

            setattr(inner_model, encoder_attr, self._compile_model(encoder))
            self._compiled_encoder_attr = encoder_attr
            logger.info("Multimodal encoder (%s) compiled separately", encoder_attr)
        else:
            self.encoder_capture_backend = None
            self._compiled_encoder_attr = None

        logger.info("NeuronModel loading complete (moved to device and compiled)")

        if self.drafter is not None:
            logger.info("Spec decode enabled. Loading draft model ...")
            # TODO: model loading logic could be extracted
            # Pass target model's padded hidden_size to draft model
            target_hidden_size = self.model.config.hidden_size
            # Layout of that padded hidden dim. A target that deinterleaves the
            # padded axis reports its group count; the default 1 means the
            # padding sits at the tail.
            get_pad_groups = getattr(self.model, "get_eagle3_hidden_pad_groups", None)
            target_hidden_pad_groups = get_pad_groups() if get_pad_groups else 1
            self.drafter.load_model(
                target_hidden_size=target_hidden_size,
                target_hidden_pad_groups=target_hidden_pad_groups,
                compile_options=self.compile_options,
            )
            logger.info("Draft model loading complete (moved to device and compiled)")

    def resolve_graph_compiler_args(self, graph_type):
        """Return the concrete compiler arguments for ``graph_type``.

        Read the base arguments from the compile options and delegate to the
        model's ``compiler_args_for`` policy. Return ``None`` when per-graph
        tuning is off (an explicit optimization level, or a model with no
        policy) or the graph type is ``None``, so the caller leaves the base
        arguments untouched.
        """
        if not getattr(self, "_per_graph_compiler_args", False) or graph_type is None:
            return None
        base = list((self.compile_options or {}).get("compiler_args") or [])
        return self._compiler_args_policy(graph_type, None, base)

    @contextmanager
    def _plain_decode_forward(self):
        """Yield the compiled model for the plain single-token decode forward.

        Scope the "decode" graph's resolved arguments, then yield the shared
        ``self.model``. Route ONLY the plain single-token decode graph here.
        Prefill, JF, and spec targets use ``self.model`` directly. The encoder
        graph scopes "encoder" at its own execute and capture sites.
        """
        with scoped_compiler_args(self.resolve_graph_compiler_args("decode")):
            yield self.model

    @contextmanager
    def _plain_decode_capture(self):
        """Graph-capture backend counterpart to _plain_decode_forward."""
        with scoped_compiler_args(self.resolve_graph_compiler_args("decode")):
            yield self.capture_backend_model

    def init_tensor_capture(self) -> None:
        """Initialize tensor capture for target, vision, and draft models.

        Must be called after load_model() (models are compiled) and before
        warmup (so dynamo traces hooks and inline capture_tensor() calls).
        """
        from vllm_neuron.accuracy.tensor_capture import TensorRegistry, expand_patterns

        capture_config = self.neuron_config.tensor_capture
        if not capture_config:
            return

        # Split modules into text and vision (visual.* prefix). The prefix is
        # vision-only, so an audio module never routes here and the audio_tower
        # branch below stays unreachable until the prefix is modality-neutral.
        all_modules = (
            expand_patterns(capture_config.modules) if capture_config.modules else []
        )
        text_modules = [m for m in all_modules if not m.startswith("visual.")]
        encoder_modules = [
            m[len("visual.") :] for m in all_modules if m.startswith("visual.")
        ]

        # Target (text) model capture
        self._target_tensor_capture = self._setup_capture(
            self.model,
            text_modules,
            capture_filter=(
                set(capture_config.capture_filter)
                if capture_config.capture_filter is not None
                else None
            ),
        )
        # Set global TensorRegistry instance so inline capture_tensor()
        # calls inside model code are traced as active during warmup.
        if self._target_tensor_capture is not None:
            TensorRegistry._instance = self._target_tensor_capture._registry

        # Multimodal encoder capture ("visual" for VL, "audio_tower" for ASR).
        self._encoder_tensor_capture = None
        if encoder_modules and self.encoder_neuron_config is not None:
            inner_model = getattr(self.model, "_orig_mod", self.model)
            encoder_attr = (
                "visual"
                if getattr(inner_model, "visual", None) is not None
                else "audio_tower"
                if getattr(inner_model, "audio_tower", None) is not None
                else None
            )
            if encoder_attr is not None:
                self._encoder_tensor_capture = self._setup_capture(
                    getattr(inner_model, encoder_attr),
                    encoder_modules,
                    subdirectory="vision",
                )

        # Draft model capture
        if self.drafter is not None:
            draft_modules = getattr(capture_config, "draft_modules", None)
            self._draft_tensor_capture = self._setup_capture(
                self.drafter.model, draft_modules or [], subdirectory="draft"
            )
            if self._draft_tensor_capture is not None:
                self.drafter._draft_tensor_capture = self._draft_tensor_capture

        logger.info(
            "Tensor capture initialized, capture_dir=%s", capture_config.capture_dir
        )

    def init_tensor_replacement(self) -> None:
        """Initialize tensor replacement from neuron_config."""
        replacement_config = self.neuron_config.tensor_replacement
        if not replacement_config:
            return

        tensors, prompt_token_ids = replacement_config.resolve()
        self._tensor_replacer = TensorReplacer(
            tensors, prompt_token_ids=prompt_token_ids
        )
        logger.info("Tensor replacement initialized")

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the target model should
        be used as auxiliary inputs for the Eagle3 draft model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    # ------------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------------
    def get_model(self) -> Any:
        """Get the underlying model.

        Simple accessor, provided for API compatibility with
        GPUModelRunner._update_states. For pooling models the returned model
        exposes a ``.pooler`` attribute.
        """
        if self.model is None:
            raise ValueError("Cannot get model before model has been initialized")
        return self.model

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder batch to match scheduler output order if needed.

        This is a no-op for Neuron since we don't support reordering yet.
        """
        pass

    def _get_valid_sampled_token_count(self) -> list[int]:
        """Get the number of valid sampled tokens for each request.

        Used in async scheduling + spec decode: ``_update_states`` subtracts
        rejected draft counts from scheduler-provided ``num_computed_tokens``
        (which assumes all drafts accepted).

        Returns optimistic full-acceptance so the CPU-side correction loop in
        ``_update_states`` is a no-op. The target NEFF's on-device correction
        handles the drift inside the compile boundary (see
        ``correct_spec_decode_positions_and_slot_mapping``).

        Returns:
            ``[1 + num_speculative_tokens] * prev_num_reqs`` — the full-
            acceptance count per request, sized to the **previous** step's
            batch (since ``_update_states`` indexes via
            ``prev_req_id_to_index``). ``[0]`` when speculative decoding is
            disabled.
        """
        if self.speculative_config is None:
            return [0]
        num_spec = self.speculative_config.num_speculative_tokens
        # Size by the previous step's batch (or the current step's, whichever
        # is larger) so any ``prev_req_index`` lookup in
        # ``_update_states`` is in-range. Using ``max_num_reqs`` is also
        # safe and simpler.
        num_reqs = self.max_num_reqs
        return [1 + num_spec] * max(num_reqs, 1)

    def _get_prev_spec_tensors_for_on_device_correction(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
        num_reqs_padded: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build inputs for on-device async-spec num_computed_tokens correction.

        Returns the previous step's rejection sampler output tensor and the
        per-request draft count, both shaped to match the current decode
        NEFF's batch dim. The target NEFF reads these inside its compile
        boundary to correct positions for rejected drafts without a CPU
        sync on the previous step's output.

        For the first decode step after prefill (no prior spec step), or
        when the previous step was not spec decode, returns a dummy
        ``prev_sampled_token_ids`` filled with zeros and
        ``prev_num_draft_tokens`` filled with zeros so the device-side
        correction is a no-op (``1 + num_draft - valid_count == 0``).

        Args:
            spec_decode_metadata: Current step's spec decode metadata.
                Used to derive the expected output shape (num_reqs_padded,
                num_spec+1).

        Returns:
            ``(prev_sampled_token_ids, prev_num_draft_tokens)`` — device
            tensors. Shapes ``[num_reqs_padded, num_spec+1]`` int32 and
            ``[num_reqs_padded]`` int32, respectively.
        """
        if device is None:
            device = self.device
        num_spec = self.speculative_config.num_speculative_tokens
        # Derive num_reqs_padded from spec_decode_metadata — the
        # bonus_logits_indices length matches padded decode batch size.
        if spec_decode_metadata is not None:
            num_reqs_padded = spec_decode_metadata.bonus_logits_indices.shape[0]
        else:
            assert num_reqs_padded is not None, (
                "num_reqs_padded must be provided when spec_decode_metadata is None"
            )

        prev_future = self.async_execution_buffer.get("futures_sampled_token_ids")
        use_real_prev = (
            prev_future is not None
            and isinstance(prev_future, torch.Tensor)
            and prev_future.ndim == 2
            and prev_future.shape[1] == num_spec + 1
            and prev_future.shape[0] > 0
        )

        if use_real_prev:
            prev_sampled = self._remap_prev_sampled_by_req_id(
                prev_future, num_reqs_padded
            )
            # When the previous future has shape [bs, num_spec+1], the previous
            # step was a spec step with a full num_spec_tokens bucket. Use
            # num_spec directly: it's always the scheduled draft count for
            # that step (we never mid-trim draft counts — NeuronAsyncScheduler
            # clears spec_token_ids entirely near max_model_len instead).
            prev_num_draft_tokens_tensor = torch.full(
                (num_reqs_padded,), num_spec, dtype=torch.int32, device=device
            )
        else:
            # First decode step after prefill, or shape mismatch. Pass an
            # all-zero (= all-valid) dummy so valid_count == num_spec + 1.
            # Pair with num_draft == num_spec so:
            #   num_rejected = 1 + num_spec - (num_spec + 1) = 0
            # → correction is a no-op.
            prev_sampled = torch.zeros(
                num_reqs_padded,
                num_spec + 1,
                dtype=torch.int32,
                device=device,
            )
            prev_num_draft_tokens_tensor = torch.full(
                (num_reqs_padded,), num_spec, dtype=torch.int32, device=device
            )

        return prev_sampled, prev_num_draft_tokens_tensor

    def _build_async_spec_kwargs(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
        num_total_tokens: int,
        num_reqs_padded: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the three ``prev_*`` kwargs the decorator's prologue
        consumes.

        Returns a dict that is *always* populated when the model is
        decorated with ``@async_speculative_decoding``. Real previous-step
        data is used when available; otherwise zero/identity dummies are
        returned that make the on-device correction a no-op:

            valid_count == 1 + num_spec → num_rejected == 0

        Always-on injection is required so ``torch.compile`` traces a
        single graph per shape combination (no Python branching on
        kwarg presence).

        Args:
            spec_decode_metadata: Current step's metadata (``None`` for
                non-spec or prefill).
            num_total_tokens: Length of the ``input_ids`` tensor for this
                step. Used to size ``req_indices_per_token``.
            num_reqs_padded: Padded decode batch size. Required when
                ``spec_decode_metadata`` is ``None`` (prefill / non-spec
                decode warmup); ignored otherwise.
        """
        if device is None:
            device = self.device
        prev_sampled, prev_num_draft = (
            self._get_prev_spec_tensors_for_on_device_correction(
                spec_decode_metadata,
                num_reqs_padded=num_reqs_padded,
                device=device,
            )
        )
        bs = prev_num_draft.shape[0]
        tokens_per_req = max(num_total_tokens // bs, 1)
        req_indices_per_token = (
            torch.arange(bs, dtype=torch.int64)
            .repeat_interleave(tokens_per_req)
            .to(device)
        )
        return {
            "prev_sampled_token_ids": prev_sampled,
            "prev_num_draft_tokens": prev_num_draft,
            "req_indices_per_token": req_indices_per_token,
        }

    def _model_is_async_spec_decoded(self) -> bool:
        """Whether the model uses the ``@async_speculative_decoding``
        correction prologue. Returns ``True`` only when:

        * The model class carries the ``_async_spec_decoded`` marker
          (also looking through ``torch.compile``'s ``_orig_mod`` wrap),
        * a ``speculative_config`` exists (so ``num_spec`` is known), and
        * ``use_async_scheduling`` is enabled (the correction is only
          needed when the scheduler advances ``num_computed_tokens``
          optimistically before knowing the rejection sampler's output).
        """
        if self.speculative_config is None or not self.use_async_scheduling:
            return False
        return getattr(self.model, "_async_spec_decoded", False) or getattr(
            getattr(self.model, "_orig_mod", None), "_async_spec_decoded", False
        )

    def _reinstate_mm_is_embed(self, req_state: CachedRequestState) -> None:
        """Re-derive the placeholder is_embed mask the EPD disagg wire dropped.

        Upstream's disagg wire carries only (offset, length), so
        mm_position.is_embed arrives None on the LM pool. Sparse
        modalities (Qwen3-VL video) need the mask so _gather_mm_embeddings
        scatters encoded rows past the interleaved timestamp/marker text tokens.
        Temporary workaround till upstream support is merged.
        """
        if req_state.prompt_token_ids is None:
            return
        from dataclasses import replace

        from vllm_neuron.utils.mm_utils import get_placeholder_is_embed

        hf_config = self.vllm_config.model_config.hf_config
        prompt = req_state.prompt_token_ids
        for feature in req_state.mm_features:
            pos = feature.mm_position
            if pos.is_embed is not None:
                continue
            span = prompt[pos.offset : pos.offset + pos.length]
            mask = get_placeholder_is_embed(hf_config, feature.modality, span)
            if mask is not None:
                feature.mm_position = replace(
                    pos, is_embed=torch.tensor(mask, dtype=torch.bool)
                )

    def _init_mrope_positions(self, req_state: CachedRequestState) -> None:
        """Initialize M-RoPE positions for multimodal models (e.g. Qwen3-VL).

        Computes 3D position IDs [3, seq_len] encoding temporal, height, and
        width axes. Text tokens get identical values across all axes; vision
        tokens get spatial grid coordinates.

        Called once per request at first scheduling, before encoder output
        caching clears mm_features[i].data.
        """
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_features = [
            f for f in req_state.mm_features if f.modality != "prompt_embeds"
        ]
        model = self.get_model()
        # Unwrap torch.compile (OptimizedModule)
        unwrapped = getattr(model, "_orig_mod", model)
        if not isinstance(unwrapped, SupportsMRoPE):
            raise TypeError(
                f"Model {type(unwrapped).__name__} sets uses_mrope=True but does "
                f"not implement the SupportsMRoPE protocol."
            )
        req_state.mrope_positions, req_state.mrope_position_delta = (
            model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                mrope_features,
            )
        )

    def _calc_mrope_positions(
        self,
        req_ids: list[str],
        actual_num_tokens: np.ndarray,
    ) -> torch.Tensor:
        """Assemble [3, total_tokens] MRoPE positions from per-request data.

        For prompt (prefill) tokens: copy from pre-computed mrope_positions.
        For completion (decode) tokens: all 3 axes get same sequential value
        = mrope_position_delta + context_len + offset.

        Args:
            req_ids: Batch-ordered request IDs (from input_batch.req_ids).
            actual_num_tokens: Per-request actual token counts [num_reqs].
        """
        total_tokens = int(actual_num_tokens.sum())
        mrope_pos = np.zeros((3, total_tokens), dtype=np.int64)
        ptr = 0

        for i, req_id in enumerate(req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None, (
                f"mrope_positions not initialized for request {req_id}"
            )

            num_scheduled = int(actual_num_tokens[i])
            num_computed = int(self.input_batch.num_computed_tokens_cpu[i])
            num_prompt = req.num_prompt_tokens

            prompt_part = max(0, min(num_scheduled, num_prompt - num_computed))
            completion_part = num_scheduled - prompt_part

            if prompt_part > 0:
                src_start = num_computed
                src_end = num_computed + prompt_part
                mrope_pos[:, ptr : ptr + prompt_part] = req.mrope_positions[
                    :, src_start:src_end
                ].numpy()
                ptr += prompt_part

            if completion_part > 0:
                ctx = num_computed + prompt_part
                delta = req.mrope_position_delta
                vals = np.arange(
                    delta + ctx, delta + ctx + completion_part, dtype=np.int64
                )
                # Broadcast: all 3 RoPE axes get the same sequential decode position.
                mrope_pos[:, ptr : ptr + completion_part] = vals
                ptr += completion_part

        return torch.from_numpy(mrope_pos).to(dtype=torch.long)

    def _init_xdrope_positions(self, req_state: CachedRequestState) -> None:
        """Initialize XD-RoPE positions for models like HunYuan-VL.

        This is a placeholder for future XD-RoPE support.
        """
        pass

    def _update_streaming_request(
        self, req_id: str, new_req_data: Any
    ) -> CachedRequestState:
        """Update a streaming request with new data.

        For streaming case only.
        """
        # Update with new data as needed
        req_state = self.requests[req_id]
        return req_state

    def _register_requests_for_replacement(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Register new request prompts for tensor replacement matching."""
        if self._tensor_replacer is None:
            return
        for new_req_data in scheduler_output.scheduled_new_reqs:
            self._tensor_replacer.register_request(
                new_req_data.req_id, new_req_data.prompt_token_ids
            )

    def _batch_mutation_lock(self) -> AbstractContextManager:
        """Return the lock that serialises InputBatch mutation.

        Two threads mutate the same state: the main thread in ``_update_states``, and the
        async-output thread in ``_update_batch_state_with_samples``, which also appends to
        ``CachedRequestState.output_token_ids``. The symptom was a torn read, because
        ``InputBatch.add_request`` reads that list twice and an append can land in between.
        Upstream has no second thread, so the extra writer is ours.

        This deliberately stops short of input construction, which
        ``_apply_async_fallback_token_fixup`` already corrects. No deadlock is possible: the
        async thread takes ``get_output``'s lock then this one, the main thread takes each alone
        and never nested.

        The check is an isinstance, not ``is not None``, because ``Mock(spec=NeuronModelRunner)``
        turns the class attribute into a Mock that passes a None check and then fails the
        context-manager protocol.

        Returns:
            The real lock for a runner, else a null context.
        """
        lock = self._input_batch_lock
        return lock if isinstance(lock, _LOCK_TYPE) else nullcontext()

    # !! NEURON MAINTAINER NOTE !!
    # The _update_states method below is copied VERBATIM from upstream
    # vllm/v1/worker/gpu_model_runner.py (GPUModelRunner._update_states).
    # DO NOT modify it directly. To update, copy the latest upstream
    # implementation and replace entirely. Neuron-specific behavior goes
    # in the helper stubs above (get_model, _may_reorder_batch,
    # _get_valid_sampled_token_count, _init_mrope_positions,
    # _init_xdrope_positions, _update_streaming_request).
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        # NOTE: upstream GPUModelRunner stores encoder cache as a dict[str, torch.Tensor]
        # so freeing a cached output means `self.encoder_cache.pop(mm_hash, None)`.
        # Our EncoderCacheBlocks uses `.free()` which returns blocks to the
        # free queue (with optional hold-time guard).
        # Check vllm_neuron/vllm/worker/encoder_cache_blocks.py for details.
        evicted_blocks = 0
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            evicted_blocks += self.encoder_cache.free(mm_hash)
        if evicted_blocks:
            DEVICE_ENCODER_CACHE_EVICTIONS.inc(evicted_blocks)
            self._update_encoder_cache_occupancy_gauges()

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        if self._keep_batch_on_empty_step(
            scheduled_req_ids, scheduler_output.finished_req_ids
        ):
            unscheduled_req_ids = frozenset()
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # EPD LM pool only: the disagg wire drops the renderer's is_embed
            # mask, so re-derive it for sparse modalities (Qwen3-VL video) from
            # the placeholder token ids before _gather_mm_embeddings consumes it.
            if self.mm_language_model_only:
                self._reinstate_mm_is_embed(req_state)

            # Only relevant for models using M-RoPE (e.g, Qwen3-VL)
            # Skip when running vision tower in isolation as it does not consume M-Rope positions
            if self.uses_mrope and not self.mm_encoder_only:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    new_token_ids: list[int] = []
                else:
                    new_token_ids = req_data.new_token_ids[i]
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids

            if req_index is None:
                if self.use_async_scheduling and num_output_tokens > 0:
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            if not is_last_rank:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        logger.debug(
            "Completed state update. Current batch size: %s",
            len(self.input_batch.req_ids),
        )

    def _compute_cached_seq_len(self) -> int:
        """
        Compute the maximum cached sequence length across all requests in the batch.
        Note that Neuron currently only supports a prefill batch size of 1.

        Returns:
            int: Maximum number of already-computed tokens across all requests.
                 Returns 0 if no tokens are cached.
        """
        cached_seq_len = 0
        for req_index in range(self.input_batch.num_reqs):
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            if num_computed > cached_seq_len:
                cached_seq_len = num_computed

        if cached_seq_len > 0:
            logger.debug("Computed cached_seq_len=%s", cached_seq_len)

        return cached_seq_len

    # ------------------------------------------------------------------------
    # INPUT PREPARATION
    # TODO: Methods here and elsewhere are shared with GPUs and TPUs.
    # There is opportunity to move this common code to vLLM
    # ------------------------------------------------------------------------
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.

        E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        Equivalent to but faster than:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _remap_prev_sampled_by_req_id(
        self,
        prev_future: torch.Tensor,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Reorder ``prev_sampled_token_ids`` from previous batch order into
        current compacted order, keyed by request id.

        After ``condense()`` permutes surviving requests to fill gaps, the
        previous step's per-slot output tensor is still in the previous
        ordering. The async spec correction kernel reads it via the current
        ``req_indices_per_token`` index, so any moved request would read
        another request's row. This method gathers along dim 0 by a
        permutation built from ``prev_req_ids_ordered`` vs the current
        ``input_batch.req_ids``.

        For current rows whose request didn't exist in the previous step
        (newly-added), the row is replaced with an all-valid dummy so
        num_rejected = 0 (no-op). Existing padded rows are preserved because
        overwriting their previous sampled-token rows wedges the async
        worker-side sample_tokens path. If the current padded batch is wider
        than the previous future, missing padded rows are added as no-op
        dummies to produce the required shape.

        Args:
            prev_future: Device tensor ``[previous_rows, num_spec+1]`` in the
                previous step's batch ordering.
            num_reqs_padded: Padded decode batch size for the current step.

        Returns:
            Device tensor ``[num_reqs_padded, num_spec+1]`` reordered to the
            current batch's compacted slot order.
        """
        prev_req_ids_ordered = self.async_execution_buffer.get("prev_req_ids_ordered")
        assert prev_req_ids_ordered is not None, (
            "prev_req_ids_ordered must accompany futures_sampled_token_ids"
        )

        prev_future_num_rows = prev_future.shape[0]
        assert len(prev_req_ids_ordered) <= prev_future_num_rows, (
            "prev_req_ids_ordered cannot contain more requests than "
            "futures_sampled_token_ids rows"
        )

        curr_req_ids = list(self.input_batch.req_ids[: self.input_batch.num_reqs])
        prev_id_to_idx = {req_id: i for i, req_id in enumerate(prev_req_ids_ordered)}

        perm: list[int] = []
        missing_indices: list[int] = []
        for cur_idx, req_id in enumerate(curr_req_ids):
            old_idx = prev_id_to_idx.get(req_id)
            if old_idx is None:
                # Request wasn't in the previous step (new).
                # Use row 0 as a placeholder; we'll overwrite with a dummy row below.
                perm.append(0)
                missing_indices.append(cur_idx)
            else:
                perm.append(old_idx)
        # Padded slots: identity when available, placeholder otherwise.
        for pad_idx in range(self.input_batch.num_reqs, num_reqs_padded):
            if pad_idx < prev_future_num_rows:
                perm.append(pad_idx)
            else:
                perm.append(0)
                missing_indices.append(pad_idx)

        needs_remap = prev_future_num_rows != num_reqs_padded or any(
            old_idx != cur_idx for cur_idx, old_idx in enumerate(perm)
        )
        needs_dummy_rows = bool(missing_indices)
        if not needs_remap and not needs_dummy_rows:
            return prev_future

        if needs_remap:
            perm_tensor = torch.tensor(
                perm, dtype=torch.long, device=prev_future.device
            )
            # ``index_select`` may return a non-contiguous tensor on Neuron, and
            # the downstream NEFF requires contiguous input.
            prev_sampled = torch.index_select(prev_future, 0, perm_tensor).contiguous()
        else:
            prev_sampled = prev_future

        if needs_dummy_rows:
            dummy_mask_values = [False] * num_reqs_padded
            for cur_idx in missing_indices:
                dummy_mask_values[cur_idx] = True
            dummy_mask = torch.tensor(
                dummy_mask_values, dtype=torch.bool, device=prev_future.device
            ).view(num_reqs_padded, 1)
            # Avoid in-place row writes here. On Neuron those writes can yield
            # intermittent non-contiguous inputs for the downstream NEFF.
            prev_sampled = torch.where(
                dummy_mask, torch.zeros_like(prev_sampled), prev_sampled
            ).contiguous()

        return prev_sampled

    @staticmethod
    def _keep_batch_on_empty_step(
        scheduled_req_ids: "Collection[str]",
        finished_req_ids: "Collection[str] | None",
    ) -> bool:
        """Report whether to keep the persistent batch on a step that does no work.

        To start an asynchronous KV load the scheduler emits a step that schedules and finishes
        nothing, hiding every cached request and scheduling them again next step. Evicting there
        buys only a full rebuild of the batch, far more expensive than the incremental update.
        The skip is safe because nothing reads the batch on a zero-token step, and the caller
        recomputes the unscheduled set every step, so it can delay a removal but never lose one.

        Args:
            scheduled_req_ids: Request ids that this step schedules.
            finished_req_ids: Request ids that finished on this step, or None.

        Returns:
            True when the caller must keep every cached request.

        Example:
            >>> NeuronModelRunner._keep_batch_on_empty_step(set(), set())
            True
            >>> NeuronModelRunner._keep_batch_on_empty_step({"a"}, set())
            False
        """
        return not scheduled_req_ids and not finished_req_ids

    def _cached_constant_tensor(
        self, key: tuple, factory: Callable[[], torch.Tensor]
    ) -> torch.Tensor:
        """Return a device tensor whose contents depend only on ``key``.

        Built once per key. Reuse is safe while an earlier step's NEFF still reads the buffer,
        because a hit would rebuild the same bits and a new value gives a new key.

        ``key`` must include every input the contents depend on, plus the KV-cache group id when
        the tensor is an input of the compiled graph: ``torch.compile`` guards reject a duplicate
        tensor identity, so one shared object changes what the graph sees. Only for values that
        repeat, since each key holds a tensor for the life of the runner.

        Args:
            key: Every input that determines the contents, plus the group id when the graph
                takes the tensor as an input.
            factory: Builds the tensor on a miss.

        Returns:
            The device tensor for this key.
        """
        if self._constant_tensors is None:
            self._constant_tensors = {}
        tensor = self._constant_tensors.get(key)
        if tensor is None:
            tensor = factory()
            self._constant_tensors[key] = tensor
        return tensor

    @staticmethod
    def _raw_block_table(buffer: Any) -> np.ndarray:
        """Return a zero-copy host view of the raw block table of a buffer.

        ``CpuGpuBuffer`` defines ``np`` only when built with ``with_numpy=True``, so fall back to
        a view of the same storage, and test the type because a stand-in can sit there.

        Args:
            buffer: Buffer whose ``cpu`` mirror holds the raw table.

        Returns:
            A host view of the raw table.
        """
        raw = getattr(buffer, "np", None)
        return raw if isinstance(raw, np.ndarray) else buffer.cpu.numpy()

    def _block_table_device_tensor(self, group_id: int, buffer: Any) -> torch.Tensor:
        """Return the device block table of one KV-cache group.

        Remaps the null block to the sentinel on the host and uploads the result, skipping both
        when the raw host table has not changed since the last upload. The remap runs in numpy,
        because the earlier order of uploading the raw table and remapping on device with
        ``torch.where`` added a host round trip for a value the host already held.

        Args:
            group_id: Index of the KV-cache group that owns the block table.
            buffer: Buffer whose ``cpu`` mirror holds the raw table.

        Returns:
            A device tensor that holds the remapped block table.
        """
        if self._block_table_groups is None:
            self._block_table_groups = {}
        state = self._block_table_groups.setdefault(group_id, _BlockTableGroupState())
        raw = self._raw_block_table(buffer)
        if not state.fits(raw.shape, raw.dtype):
            state.reset(raw)
        elif state.uploaded_device is not None and np.array_equal(
            state.uploaded_host, raw
        ):
            # The scheduler did not change the table, so the last upload is still
            # correct.
            return state.uploaded_device

        slot = state.next_slot
        state.next_slot = (slot + 1) % _BLOCK_TABLE_STAGE_DEPTH
        stage, null_mask = state.stage[slot], state.null_mask[slot]
        np.copyto(stage, raw)
        # Reuse the mask and pass ``out`` to keep the hot path free of allocations.
        np.equal(stage, NULL_BLOCK_ID, out=null_mask)
        np.copyto(stage, PAD_SLOT_ID, where=null_mask)

        # Build a new device tensor for each upload. An in-place copy is unsafe, because
        # the NEFF of the previous step can still reference the old device tensor.
        device_table = torch.from_numpy(stage).to(self.device)
        # Copy the raw table. The scheduler changes the host buffer in place between
        # steps, so a view would compare the next table against itself and report a
        # false cache hit.
        state.uploaded_host = raw.copy()
        state.uploaded_device = device_table
        return device_table

    def _save_block_table_to_device(self) -> None:
        """
        To avoid data dependencies that block async execution, we should store the
        full block table tensor on device instead of updating it in place with commit_block_table
        """
        for group_id, block_table in enumerate(
            self.input_batch.block_table.block_tables
        ):
            buffer = block_table.block_table
            buffer.gpu = self._block_table_device_tensor(group_id, buffer)
        # Snapshot block tables for KV cache analysis (survives request freeing).
        # Apply remap on CPU tensor directly to avoid device→CPU transfer.
        if self._kv_snapshot_enabled:
            self._block_table_snapshot = [
                _remap_null_block_to_sentinel(bt.block_table.cpu)
                for bt in self.input_batch.block_table.block_tables
            ]

    def _save_slot_mapping_to_device(self, num_tokens: int) -> None:
        """Upload the slot mapping of each KV-cache group to the device.

        Out of place, so async work from the previous step keeps the tensor it already holds.
        The buffer holds ``max_num_batched_tokens`` while the consumer reads only the first
        ``total_num_scheduled_tokens``, which on a decode step differ by orders of magnitude.

        Args:
            num_tokens: Number of leading entries to upload.

        Example:
            >>> runner._save_slot_mapping_to_device(8)  # doctest: +SKIP
        """
        for block_table in self.input_batch.block_table.block_tables:
            slot_mapping = block_table.slot_mapping
            # Clamp a negative count. A negative value would slice a suffix. A count above
            # the length needs no clamp, because a slice stops at the end.
            source = slot_mapping.cpu[: max(num_tokens, 0)]
            slot_mapping.gpu = source.to(self.device)

    # Must match the grad mode encoder warmup compiles in.
    @torch.no_grad()
    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput") -> None:
        """Encode uncached multimodal inputs into the on-device encoder cache.

        Allocates cache blocks for each new mm_item, then calls
        embed_multimodal() which has the encoder NEFF scatter-write directly
        into the cache buffer. No device→host round-trip.

        Bucket overflow is handled by the scheduler's encoder_compute_budget
        — it won't schedule more encoder inputs than the budget allows.
        """
        if not scheduler_output.scheduled_encoder_inputs:
            return

        from vllm.multimodal.utils import group_and_batch_mm_kwargs

        mm_hashes: list[str] = []
        mm_kwargs: list[tuple] = []

        for (
            req_id,
            encoder_input_ids,
        ) in scheduler_output.scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))

        if not mm_kwargs:
            return

        # prompt_embeds modality: pre-computed embeddings that bypass the encoder.
        # Stored directly in cache via CPU slice ops (no encoder run needed).
        # The prefill read path (_gather_mm_embeddings) retrieves them the same
        # way as encoder embeddings — block-based lookup is modality-agnostic.
        # NOTE: this path is functional but not E2E validated with on-device cache.
        pe_indices = [
            i
            for i, (modality, _) in enumerate(mm_kwargs)
            if modality == "prompt_embeds"
        ]
        for i in pe_indices:
            self.encoder_cache.put(mm_hashes[i], mm_kwargs[i][1]["embedding"].data)
        if pe_indices:
            mm_hashes = [h for i, h in enumerate(mm_hashes) if i not in pe_indices]
            mm_kwargs = [k for i, k in enumerate(mm_kwargs) if i not in pe_indices]

        if not mm_kwargs:
            return

        # Each scheduled mm item is a query against the on-device encoder cache;
        # items not present are misses that must run the encoder. hits =
        # queries - misses (derived, ~always 0 here — see below). Count queries
        # before the early return so a fully-cached batch still records them.
        DEVICE_ENCODER_CACHE_QUERIES.inc(len(mm_hashes))

        # Filter already-cached items
        uncached = [
            (mm_hash, modality, data)
            for mm_hash, (modality, data) in zip(mm_hashes, mm_kwargs)
            if not self.encoder_cache.contains(mm_hash)
        ]
        if not uncached:
            return
        DEVICE_ENCODER_CACHE_MISSES.inc(len(uncached))

        # Group by modality and batch, then delegate to model.
        # The model handles allocation internally (model-specific token counting).
        uncached_kwargs = [(modality, data) for _, modality, data in uncached]
        uncached_hashes = [mm_hash for mm_hash, _, _ in uncached]

        t0 = time.perf_counter()
        item_cursor = 0
        if self._encoder_tensor_capture is not None:
            self._encoder_tensor_capture.setup_tensor_capture()
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            uncached_kwargs
        ):
            batch_hashes = uncached_hashes[item_cursor : item_cursor + num_items]
            item_cursor += num_items

            self.model.embed_multimodal(
                encoder_cache=self.encoder_cache,
                mm_hashes=batch_hashes,
                **mm_kwargs_batch,
            )
            for mm_hash in batch_hashes:
                self.encoder_cache.mark_written(mm_hash)
        t_encode = time.perf_counter()

        if self._encoder_tensor_capture is not None:
            self._encoder_tensor_capture.save_tensor_captures(
                positions=torch.tensor([]),
                is_prefill=True,
                req_ids=list(scheduler_output.scheduled_encoder_inputs.keys()),
            )

        logger.debug(
            "[mm_encoder] %d items encoded in %.1fms",
            len(uncached),
            (t_encode - t0) * 1000,
        )

        # Allocation changed the cache occupancy — refresh the gauges.
        self._update_encoder_cache_occupancy_gauges()

    def _update_encoder_cache_occupancy_gauges(self) -> None:
        """Publish point-in-time on-device encoder cache occupancy gauges.

        Called after allocation (``_execute_mm_encoder``) and eviction (the
        ``free_encoder_mm_hashes`` loop) — the two points that mutate cache
        state. No-op when the model has no encoder cache (text-only models).
        """
        ec = getattr(self, "encoder_cache", None)
        if ec is None:
            return
        DEVICE_ENCODER_CACHE_ACTIVE_BLOCKS.set(ec.num_active_blocks)
        DEVICE_ENCODER_CACHE_FREE_BLOCKS.set(ec.num_free_blocks)
        DEVICE_ENCODER_CACHE_HELD_BLOCKS.set(ec.num_held_blocks)
        DEVICE_ENCODER_CACHE_ACTIVE_REAL_TOKENS.set(ec.num_active_real_tokens)

    def _gather_mm_embeddings(
        self,
        num_tokens: int,
        req_ids: list[str],
        scheduler_output: "SchedulerOutput",
        padding_map: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather multimodal embeddings for the current prefill step.

        Analogous to upstream vLLM's GPUModelRunner._gather_mm_embeddings:
        determines which cached multimodal items overlap the current scheduling
        window and returns them ready for the prefill graph.

        Unlike upstream (which returns variable-length tensor slices for
        CPU-side scatter into inputs_embeds), this returns the on-device cache
        buffer + the block indices to gather + a position map. The prefill
        graph uses index_select + index_put_ to merge them into hidden_states.
        Passing the buffer + an index tensor lets the graph do a single
        coalesced gather instead of one strided DMA copy per block.

        Args:
            num_tokens: Total padded token count for this step.
            req_ids: Ordered request IDs in the batch.
            scheduler_output: Scheduler output with scheduling info.
            padding_map: Per-request padding counts.

        Returns:
            (encoder_embedding_buffer, encoder_block_ids, encoder_positions) where:
            - encoder_embedding_buffer: the on-device cache buffer,
              [num_blocks, block_size, fat_dim] (the whole buffer is passed).
            - encoder_block_ids: [max_num_encoder_blocks] int32 block indices to
              gather from the buffer (padded with the scratch block id).
            - encoder_positions: [max_num_encoder_blocks, block_size] int32 —
              encoder_positions[i, j] = batch position for token j in block i.
              Sentinel value = num_tokens for don't-care positions.
        """
        assert len(req_ids) == 1, (
            f"_gather_mm_embeddings expects single-request prefill but got "
            f"{len(req_ids)} requests. The prefill NEFF input is sized for "
            f"max_encoder_blocks_per_request={self.max_encoder_blocks_per_request} "
            f"which assumes one request per step."
        )
        # EPD EC consumer: the NIXL READs for this step's vision items were issued
        # asynchronously (in _run_ec_consumer_lifecycle), so block until they land
        # before reading any cache block.
        # has_ec_transfer() is only true on a configured EC pool, so the monolith
        # and producer paths skip this.
        # TODO(epd-transport): maybe_wait_for_ec_load + ECLoadFailure come from a
        # plugin mixin copied off vllm #40695; inherit upstream's once it lands.
        from vllm.distributed.ec_transfer import has_ec_transfer

        if has_ec_transfer():
            ec_failed_mm_hashes = self.maybe_wait_for_ec_load()
            if ec_failed_mm_hashes:
                raise ECLoadFailure(ec_failed_mm_hashes)

        max_num_encoder_blocks = self.max_encoder_blocks_per_request

        # int32 (not int64): both the NKI kernel's indirect-DMA address pattern
        # and the torch reference's index_select/index_put_ accept int32, so
        # building it narrow here avoids an on-device dtype cast in the graph.
        encoder_positions = torch.full(
            (max_num_encoder_blocks, self.encoder_cache.block_size),
            num_tokens,
            dtype=torch.int32,
            device="cpu",
        )
        block_ids_list: list[int] = []
        block_cursor = 0

        # Fail-loud bookkeeping: any scheduled in-window multimodal item that is
        # not resident or fails validation is recorded here (not silently
        # skipped), then raised after the loop. expected_embed_positions counts
        # the embedding positions we *should* fill this step, checked against the
        # positions actually written by the post-gather backstop.
        missed: list[tuple[str, str]] = []
        expected_embed_positions = 0

        req_batch_offset = 0
        for req_id in req_ids:
            req_state = self.requests[req_id]
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            padding = padding_map.get(req_id, 0)
            num_computed = req_state.num_computed_tokens

            if req_state.mm_features:
                for mm_feature in req_state.mm_features:
                    mm_start = mm_feature.mm_position.offset
                    mm_length = mm_feature.mm_position.length
                    mm_end = mm_start + mm_length

                    if (
                        mm_end <= num_computed
                        or mm_start >= num_computed + num_scheduled
                    ):
                        continue

                    mm_hash = mm_feature.identifier
                    block_ids = self.encoder_cache.get_block_ids(mm_hash)
                    tokens_per_block = self.encoder_cache.get_tokens_per_block(mm_hash)
                    if block_ids is None or tokens_per_block is None:
                        # Fail loud: this item is scheduled and overlaps the current
                        # window, so a missing slot is a real fault (evicted before
                        # read / empty transfer / stale hint), not a legitimate skip.
                        # Never leave its placeholder tokens unfilled and feed the
                        # model as if no image were present.
                        missed.append(
                            (
                                mm_hash,
                                f"not-resident (mm_start={mm_start}, "
                                f"mm_length={mm_length}, num_computed={num_computed}, "
                                f"num_scheduled={num_scheduled})",
                            )
                        )
                        continue

                    # Validate the stored token count against what the sequence
                    # placeholder expects (independent code paths; a mismatch is a
                    # merge/bucket bug in monolith or a wrong-sized transfer in EPD).
                    is_embed_full = mm_feature.mm_position.is_embed
                    expected_item_tokens = (
                        int(is_embed_full.sum())
                        if is_embed_full is not None
                        else mm_length
                    )
                    num_tokens_err = self.encoder_cache.check_num_tokens(
                        mm_hash, expected_item_tokens
                    )
                    if num_tokens_err is not None:
                        missed.append((mm_hash, num_tokens_err))
                        continue

                    overlap_start = max(mm_start, num_computed)
                    overlap_end = min(mm_end, num_computed + num_scheduled)
                    tensor_start = overlap_start - mm_start
                    tensor_end = overlap_end - mm_start
                    batch_start = req_batch_offset + (overlap_start - num_computed)

                    # Video (is_embed set): the placeholder span is non-contiguous
                    # -- timestamp / vision-marker tokens interleave the embedding
                    # runs, so the contiguous cache rows map to scattered sequence
                    # positions. seq_pos[k] = batch position of the k-th cached
                    # embedding row. Image (is_embed None) keeps the contiguous map.
                    is_embed = mm_feature.mm_position.is_embed
                    seq_pos = None
                    if is_embed is not None:
                        embeds_start, embeds_end = (
                            mm_feature.mm_position.get_embeds_indices_in_range(
                                tensor_start, tensor_end
                            )
                        )
                        if embeds_start == embeds_end:
                            continue  # only marker tokens scheduled this step
                        window_mask = is_embed[tensor_start:tensor_end]
                        seq_pos = torch.arange(
                            batch_start, batch_start + (overlap_end - overlap_start)
                        )[window_mask]

                    # Embedding positions this item contributes this step: for a
                    # video the in-window is_embed rows, for an image the full
                    # overlap span. Summed across items, compared to the positions
                    # actually written by the post-gather backstop below.
                    if seq_pos is not None:
                        expected_embed_positions += int(seq_pos.shape[0])
                    else:
                        expected_embed_positions += overlap_end - overlap_start

                    item_token_offset = 0
                    for block_idx, block_id in enumerate(block_ids):
                        if block_cursor >= max_num_encoder_blocks:
                            raise RuntimeError(
                                f"Encoder block truncation: request has more "
                                f"cached blocks than "
                                f"max_encoder_blocks_per_request="
                                f"{max_num_encoder_blocks}. This indicates a "
                                f"configuration mismatch between encoder bucket "
                                f"size and encoder cache sizing."
                            )
                        block_ids_list.append(block_id)

                        # Real merged tokens for this block: for images the dense
                        # expansion [block_size, ..., remainder]; for videos, the
                        # per-block list whose non-tiling blocks carry a pad tail
                        # that this count lets the reader skip.
                        chunk = tokens_per_block[block_idx]
                        if seq_pos is not None:
                            # Video: row j -> its is_embed sequence position; rows
                            # outside this step's window keep the sentinel from
                            # the initial fill.
                            rel = item_token_offset - embeds_start
                            lo = max(0, -rel)
                            hi = min(chunk, seq_pos.shape[0] - rel)
                            if hi > lo:
                                encoder_positions[block_cursor, lo:hi] = seq_pos[
                                    rel + lo : rel + hi
                                ]
                        else:
                            # Image: contiguous span, unchanged.
                            item_toks = torch.arange(
                                item_token_offset,
                                item_token_offset + self.encoder_cache.block_size,
                            )
                            in_range = (
                                (item_toks >= tensor_start)
                                & (item_toks < tensor_end)
                                & (item_toks < item_token_offset + chunk)
                            )
                            batch_positions = batch_start + (item_toks - tensor_start)
                            encoder_positions[block_cursor] = torch.where(
                                in_range, batch_positions, num_tokens
                            )

                        block_cursor += 1
                        item_token_offset += chunk

            req_batch_offset += num_scheduled + padding

        # Mandatory post-gather correctness guard (defense-in-depth): every
        # scheduled in-window embedding position must have been written with a
        # real batch position. A shortfall means some placeholder would reach the
        # compiled model unfilled -> record it as a fault. This-step scoped, so
        # chunked-prefill partial windows and marker-only video steps contribute 0.
        filled = int((encoder_positions != num_tokens).sum())
        if filled != expected_embed_positions:
            missed.append(
                (
                    "<post-gather>",
                    f"unfilled scheduled multimodal tokens: wrote {filled} != "
                    f"expected {expected_embed_positions}",
                )
            )

        # Fail loud on any recorded miss/validation failure before the embeddings
        # reach the model. Per-item faults carry a real mm_hash and on an EPD
        # consumer raise ECLoadFailure.
        # The post-gather backstop (synthetic "<post-gather>", no real hash) fires
        # only on an internal gather inconsistency retry cannot fix, so it falls
        # through to the hard EncoderCacheReadError even on EPD (monolith always does).
        if missed:
            real_hashes = {h for h, _ in missed if not h.startswith("<")}
            if has_ec_transfer() and real_hashes:
                raise ECLoadFailure(real_hashes)
            context = (
                f"req_ids={req_ids}, num_tokens={num_tokens}; cache state: "
                f"active_blocks={self.encoder_cache.num_active_blocks}, "
                f"free_blocks={self.encoder_cache.num_free_blocks}, "
                f"held_blocks={self.encoder_cache.num_held_blocks}"
            )
            raise EncoderCacheReadError(missed, context)

        # Pad with scratch block for unused slots.
        while len(block_ids_list) < max_num_encoder_blocks:
            block_ids_list.append(self.encoder_cache.scratch_block_id)

        # Pass the whole buffer + block indices; the graph gathers with a single
        # index_select instead of a per-block stack (one strided DMA per block).
        # int32 (see encoder_positions above): avoids an on-device cast in the
        # NKI kernel's indirect-DMA address pattern.
        encoder_block_ids = torch.tensor(
            block_ids_list, dtype=torch.int32, device=self.device
        )
        return (
            self.encoder_cache.buffer,
            encoder_block_ids,
            encoder_positions.to(self.device),
        )

    def _prepare_model_input(
        self, scheduler_output: Any, reuse_dp_padding: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        AttentionMetadata | None,
        SpecDecodeMetadata | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """
        Prepare model inputs from scheduler output with padding for prefill requests.

        Args:
            scheduler_output: Output from vLLM scheduler
            reuse_dp_padding: When True, reuse the cross-DP padding decision
                (``dp_pad`` / ``max_decode_ctx_len``) cached by the immediately
                preceding call instead of issuing a fresh ``_get_dp_padding``
                all-reduce. Used by the async-scheduling re-prep path, which
                rebuilds inputs for the *same* batch composition after
                materializing stale async output. Issuing a second host reduce
                there would break the per-step 1:1 host-reduce/EP-forward ratio
                and desync DP ranks' collective streams.

        Returns:
            Tuple of (input_ids, positions, logit_indices, attention_metadata,
            spec_decode_metadata, rotary_position_ids, mm_kwargs)
        """
        import traceback

        try:
            return self._prepare_model_input_impl(
                scheduler_output, reuse_dp_padding=reuse_dp_padding
            )
        except Exception as e:
            logger.error("Error in _prepare_model_input: %s", e)
            logger.error("Full traceback:\n%s", traceback.format_exc())
            raise

    def _prepare_jf_model_input(
        self,
        scheduler_output: Any,
        jf_tokens: dict[str, list[int]],
        padded_num_reqs: int,
        max_decode_ctx_len: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        "AttentionMetadata | None",
        "SpecDecodeMetadata | None",
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """Prepare model inputs for a jump-forward decode step.

        Shape: batch_size × (1 + jf_max) tokens per request.
        Layout per request (1 + jf_max tokens):
        - JF (N tokens): [decode_token, jf_0, ..., jf_N-1, PAD, ..., PAD]
        - Non-JF (N=0):  [decode_token, PAD, ..., PAD]

        Sample at position n_jf (last real token) per request.
        decode_token is the previously sampled token that needs KV computed.
        PAD positions use PAD_SLOT_ID (no KV persisted to cache).

        Args:
            padded_num_reqs: The DP-synchronized decode batch bucket, using the max.
            max_decode_ctx_len: max decode context len among DP group
        """
        num_reqs = self.input_batch.num_reqs
        jf_max = self.neuron_config.max_jump_forward_tokens
        tokens_per_req = 1 + jf_max

        total_tokens = padded_num_reqs * tokens_per_req

        self._save_block_table_to_device()

        req_ids = self.input_batch.req_ids

        # Layout per request: [decode_token, jf_0..jf_{n-1}, PAD..PAD].
        # PAD positions repeat base_pos (kept in-block), skip KV writes for PAD.
        # triu active mask ensures the sampled query never attends to PAD KV.
        base_pos = self.input_batch.num_computed_tokens_cpu[:num_reqs].astype(np.int64)
        # decode_token for each request lives at token_ids_cpu[req, base_pos].
        decode_tokens = self.input_batch.token_ids_cpu[
            np.arange(num_reqs), base_pos
        ].astype(np.int64)

        n_jf_per_req = np.array(
            [len(jf_tokens.get(r, ())) for r in req_ids[:num_reqs]],
            dtype=np.int64,
        )

        input_grid = np.zeros((padded_num_reqs, tokens_per_req), dtype=np.int64)
        pos_grid = np.zeros((padded_num_reqs, tokens_per_req), dtype=np.int64)
        real_grid = np.zeros((padded_num_reqs, tokens_per_req), dtype=bool)

        # Column offsets [0, 1, ..., jf_max] broadcast against base positions.
        col = np.arange(tokens_per_req, dtype=np.int64)
        pos_grid[:num_reqs] = base_pos[:, None] + col[None, :]
        # Real columns: 0..n_jf inclusive (decode_token + n_jf JF tokens).
        real_grid[:num_reqs] = col[None, :] <= n_jf_per_req[:, None]
        # PAD columns collapse to base_pos so slot mapping stays in-block.
        pos_grid[:num_reqs] = np.where(
            real_grid[:num_reqs], pos_grid[:num_reqs], base_pos[:, None]
        )
        # decode_token at column 0.
        input_grid[:num_reqs, 0] = decode_tokens
        # JF tokens at columns 1..n_jf.
        for req_idx, r in enumerate(req_ids[:num_reqs]):
            req_jf = jf_tokens.get(r, ())
            if req_jf:
                input_grid[req_idx, 1 : 1 + len(req_jf)] = req_jf

        all_input_ids = input_grid.reshape(-1)
        all_positions = pos_grid.reshape(-1)
        real_mask_np = real_grid.reshape(-1)

        input_ids = torch.from_numpy(all_input_ids.astype(np.int32))
        cache_positions = torch.from_numpy(all_positions)

        # Sample at the last REAL token per request: column n_jf (0 for non-JF).
        logits_indices_np = np.arange(padded_num_reqs, dtype=np.int64) * tokens_per_req
        logits_indices_np[:num_reqs] += n_jf_per_req
        logits_indices = torch.from_numpy(logits_indices_np)

        # Use _build_attention_metadata to read real block table
        cached_seq_len = self._compute_cached_seq_len()
        attn_metadata = self._build_attention_metadata(
            padded_num_reqs,
            total_tokens,
            max_query_len=tokens_per_req,
            max_num_draft_tokens=tokens_per_req - 1,
            cached_seq_len=cached_seq_len,
            max_decode_ctx_len=max_decode_ctx_len,
        )

        # Override slot_mapping: compute from real positions, then mask PADs.
        positions_np = all_positions
        req_indices_np = np.repeat(
            np.arange(padded_num_reqs, dtype=np.int64), tokens_per_req
        )
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            slot_mapping_np = np.zeros(total_tokens, dtype=np.int64)
            _compute_slot_mapping_cpu(
                # Pass the numpy view. The gather below indexes with a numpy index
                # array, and a torch source sends it through torch dispatch instead.
                self._raw_block_table(blk_table.block_table),
                slot_mapping_np,
                positions_np,
                req_indices_np,
                block_size,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
                cp_kv_cache_interleave_size=block_size,
            )
            slot_mapping_np[~real_mask_np] = PAD_SLOT_ID
            real_slot_mapping = torch.from_numpy(slot_mapping_np).to(self.device)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name]["slot_mapping"] = real_slot_mapping

        spec_decode_metadata = None
        self._jf_active_tokens = jf_tokens

        # Clear leftover prompt embeddings
        self._current_inputs_embeds = None
        self._current_is_token_ids = None

        return (
            input_ids,
            cache_positions,
            logits_indices,
            attn_metadata,
            spec_decode_metadata,
            None,
            {},
        )

    def _prepare_model_input_impl(
        self, scheduler_output: Any, reuse_dp_padding: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        AttentionMetadata | None,
        SpecDecodeMetadata | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """Actual implementation of _prepare_model_input."""
        # We must dispatch the JF NEFF among the entire DP group.
        jf_max = self.neuron_config.max_jump_forward_tokens
        jf_tokens = getattr(scheduler_output, "_jf_decode_tokens", None) or {}
        jf_possible = jf_max > 0 and self._is_decode()

        assert (
            scheduler_output.total_num_scheduled_tokens > 0
        )  # this is actual token count, no padding
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # DI mixed-batch + cross-DP spec/JF coordination. With DP+EP the
        # spec-vs-non-spec and JF-vs-plain-decode choices each select which
        # compiled NEFF runs, and that NEFF holds the cross-DP collectives —
        # every rank must dispatch the SAME graph or the collectives mismatch
        # (``status=1100 Collective Operation Pending``). Under DI the ranks
        # fall out of phase as each request's KV transfer completes
        # independently, so on a step some ranks are on their first (non-spec)
        # decode while others speculate.
        #
        # We MAX-reduce each rank's "must run non-spec" vote and "have JF
        # tokens" vote across DP (folded into the _get_dp_padding reduce below
        # as its 4th and 5th elements). If any rank can't speculate we strip the
        # WHOLE group to the non-spec NEFF; if any rank has JF tokens the whole
        # group runs the JF NEFF. The strip must happen BEFORE the token tensors
        # are built (so they're sized 1-token-per-req), and therefore before the
        # _get_dp_padding reduce. That ordering is sound because the reduce's
        # other inputs are invariant to the strip: padded_num_reqs buckets
        # purely on num_reqs (get_bucket_for_count) and the strip drops draft
        # *tokens*, not requests; max_query_len only gates decode-vs-prefill
        # (both 1 and 1+num_spec are decode); local_max_decode_ctx_len reads
        # num_computed_tokens. So we run the reduce up front, then strip.
        # padded_num_reqs is strip-invariant, and on a JF step it buckets to the
        # same value the JF path needs: num_scheduled_tokens_padded is 1+n_jf
        # per decode req, so pre_strip_max_query_len = 1+max(n_jf) <= 1+jf_max =
        # _decode_token_threshold(), classified as decode → get_bucket_for_count(
        # num_reqs). Identical on the primary prep and the async re-prep —
        # recompute it on both paths.
        pre_strip_max_query_len = max(
            scheduler_output.num_scheduled_tokens_padded[req_id]
            for req_id in self.input_batch.req_ids
        )
        padded_num_reqs = get_decode_padded_batch_size(
            num_reqs,
            pre_strip_max_query_len,
            self.neuron_config.num_seqs_buckets,
            decode_token_threshold=self._decode_token_threshold(),
        )

        if reuse_dp_padding:
            # Async re-prep of the SAME step: the primary prep already issued
            # the reduce AND applied the strip (which mutated scheduler_output
            # in place). Recomputing the vote/spec inputs here would read the
            # post-strip state and diverge from the cached pre-strip values, so
            # we reuse the cache wholesale and DO NOT strip again. Issuing a
            # second reduce here would desync the DP ranks' collective streams.
            assert self._dp_padding_cache is not None, (
                "reuse_dp_padding=True but _dp_padding_cache is None; a second "
                "_get_dp_padding all-reduce here would desync DP collectives."
            )
            (
                cached_inputs,
                dp_pad,
                max_decode_ctx_len,
                group_force_nonspec,
                any_rank_jf,
                any_rank_spec_decode,
            ) = self._dp_padding_cache
            # The strip-invariant size must match what the primary prep saw.
            assert cached_inputs[0] == padded_num_reqs, (
                "DP-padding reuse invariant violated: primary-prep padded_num_reqs "
                f"{cached_inputs[0]} != re-prep {padded_num_reqs}. The async "
                "re-prep must not change this rank's cross-DP batch size."
            )
            logger.debug("Reusing cached DP padding for async re-prep")
        else:
            # Primary prep: compute this rank's spec/non-spec vote, the JF vote,
            # and the decode-ctx input, run the single folded reduce, cache the
            # result, then apply the synced strip — all BEFORE the token tensors
            # below are built, so they're sized 1-token-per-req when we fall
            # back to the non-spec NEFF. See _get_dp_padding /
            # _maybe_strip_spec_for_nonspec_step for the deadlock rationale.
            local_force_nonspec = (
                self.vllm_config.kv_transfer_config is not None
                and self.vllm_config.kv_transfer_config.is_kv_consumer
                and self._local_step_forces_nonspec(scheduler_output)
            )
            local_max_decode_ctx_len = (
                int(self.input_batch.num_computed_tokens_cpu[:padded_num_reqs].max())
                if padded_num_reqs > 0
                and self.neuron_config.decode_context_length_buckets is not None
                else 0
            )
            is_spec_decode_step = bool(scheduler_output.scheduled_spec_decode_tokens)
            local_jf = jf_possible and bool(jf_tokens)
            (
                dp_pad,
                max_decode_ctx_len,
                any_rank_spec_decode,
                group_force_nonspec,
                any_rank_jf,
                _any_rank_real,
            ) = self._get_dp_padding(
                padded_num_reqs,
                local_max_decode_ctx_len,
                is_spec_decode=is_spec_decode_step,
                local_force_nonspec=local_force_nonspec,
                local_jf=local_jf,
                local_is_real=True,
                # A prefill-shape step does not bucket the batch size, so the identity
                # fast path cannot hold. Give the shape, and let the method decide.
                local_is_decode_shaped=(
                    pre_strip_max_query_len <= self._decode_token_threshold()
                ),
            )
            # Cache the (strip-invariant) batch size + reduced outputs so the
            # same-step re-prep reuses them without a second reduce.
            self._dp_padding_cache = (
                (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode_step),
                dp_pad,
                max_decode_ctx_len,
                group_force_nonspec,
                any_rank_jf,
                any_rank_spec_decode,
            )
            # Spec strip is a no-op on JF steps (JF ⊥ spec), but skip it
            # explicitly when the group runs JF to keep the intent clear.
            if not any_rank_jf:
                self._maybe_strip_spec_for_nonspec_step(
                    scheduler_output, local_force_nonspec, group_force_nonspec
                )

        if dp_pad > 0:
            padded_num_reqs = padded_num_reqs + dp_pad
            logger.debug(
                "DP_PADDING: Increased padded_num_reqs by %s to %s",
                dp_pad,
                padded_num_reqs,
            )

        # Record draft-lockstep state for sample_tokens. group_force_nonspec
        # wins over any_rank_spec_decode (same rule as execute_dummy_batch), so
        # a skip-propose rank only runs a dummy draft when the group actually
        # speculates this step.
        self._group_spec_enabled = any_rank_spec_decode and not group_force_nonspec
        self._step_dummy_draft_batch = padded_num_reqs
        self._step_max_decode_ctx_len = max_decode_ctx_len

        # JF dispatch on entire DP group.
        if any_rank_jf and jf_possible:
            return self._prepare_jf_model_input(
                scheduler_output, jf_tokens, padded_num_reqs, max_decode_ctx_len
            )

        # Non-JF path: clear previous JF state
        self._jf_active_tokens = None

        # OPTIMIZATION: Start copying the block table to neuron first.
        # This way, we can overlap the copy with the following CPU operations.
        self._save_block_table_to_device()

        logger.debug(
            "Starting vectorized model input preparation for %s requests, %s tokens",
            num_reqs,
            scheduler_output.total_num_scheduled_tokens,
        )

        # Get number of padded scheduled tokens for each request using vectorized approach
        req_ids = self.input_batch.req_ids
        tokens = [
            scheduler_output.num_scheduled_tokens_padded[req_id] for req_id in req_ids
        ]
        num_scheduled_tokens_padded = np.array(tokens, dtype=np.int64)
        total_num_scheduled_tokens_padded = int(num_scheduled_tokens_padded.sum())
        max_num_scheduled_tokens_padded = max(tokens)
        draft_tokens = (
            [0]
            if not scheduler_output.scheduled_spec_decode_tokens
            else [
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
                for req_id in req_ids
            ]
        )
        max_num_draft_tokens = max(draft_tokens)

        logger.debug("draft_tokens=%s", draft_tokens)
        logger.debug("max_num_draft_tokens=%s", max_num_draft_tokens)

        # Compute ACTUAL tokens to extract for each request (not padded count)
        # For prefill: actual = num_prompt_tokens - num_computed_tokens
        # For decode: actual = num_scheduled_tokens (scheduler knows the real count)
        # This is needed because:
        # - Scheduler pads prefill to bucket size (e.g., 17 -> 256)
        # - But token_ids_cpu only contains actual prompt tokens (e.g., 17)
        # - With prefix cache, num_computed_tokens > 0, so actual may be even smaller
        actual_tokens_list = []
        actual_tokens_list = [
            scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
        ]

        actual_num_tokens = np.array(actual_tokens_list, dtype=np.int64)
        # Persist per-request scheduled token counts for the pooling path:
        # _pool() (called from sample_tokens) needs them to build the pooling
        # cursor's gather indices over the flattened [T, H] buffer.
        self._pooling_num_scheduled_tokens_per_seq = actual_num_tokens

        # The two `.tolist()` calls are arguments, so lazy %-formatting does not skip
        # them. Both lists get built on every step while debug logging is off. Test the
        # level first.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Token counts: scheduled=%s, actual=%s",
                num_scheduled_tokens_padded.tolist(),
                actual_num_tokens.tolist(),
            )

        # Get request indices using ACTUAL token counts (not padded)
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], actual_num_tokens)

        # Get cumsum and arange efficiently using ACTUAL counts
        cu_num_tokens, arange = self._get_cumsum_and_arange(actual_num_tokens)

        # Calculate positions using vectorized operations
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        # Get token indices efficiently using vectorized indexing
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # Use torch.index_select for fast token extraction
        input_ids = torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
        ).to(dtype=torch.int32)

        # Validate token IDs are within vocabulary range before device transfer
        self._validate_token_ids(input_ids)

        # Sequential positions for attention mask + KV cache (always 1D)
        cache_positions = torch.from_numpy(positions_np).to(dtype=torch.long)
        # M-RoPE positions for RoPE computation (3D, only for mRoPE models)
        # Skip when running vision tower in isolation as it does not consume M-Rope positions
        if self.uses_mrope and not self.mm_encoder_only:
            rotary_position_ids = self._calc_mrope_positions(req_ids, actual_num_tokens)
        else:
            rotary_position_ids = None

        # Apply padding for prefill requests
        input_ids, cache_positions, rotary_position_ids, padding_map = (
            self._create_padded_inputs(
                input_ids,
                cache_positions,
                scheduler_output,
                req_ids,
                num_scheduled_tokens_padded,
                rotary_position_ids=rotary_position_ids,
            )
        )

        # Store padding map for later use in output extraction
        self._current_padding_map = padding_map

        # Compute slot mapping positions.
        # After padding, cache_positions contains bounded values for padding
        # tokens (last_pos repeated) — safe for block table lookup. Padding
        # writes are redirected to the reserved null block below.
        if padding_map:
            padded_req_indices = np.repeat(
                self.arange_np[:num_reqs], num_scheduled_tokens_padded
            )
            slot_positions = cache_positions.numpy()
        else:
            padded_req_indices = req_indices
            slot_positions = positions_np

        for block_table in self.input_batch.block_table.block_tables:
            # Hybrid KV manager scales block_size per KV group. We need
            # to always use an interleave size equal to block_size
            _compute_slot_mapping_cpu(
                # Pass the numpy view. The gather below indexes with a numpy index
                # array, and a torch source sends it through torch dispatch instead.
                self._raw_block_table(block_table.block_table),
                block_table.slot_mapping.np,
                slot_positions,
                padded_req_indices,
                block_table.block_size,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
                cp_kv_cache_interleave_size=block_table.block_size,
            )
        # Redirect padding writes to the reserved null block.
        if padding_map:
            # Calculate padding token indices
            # Token layout: [req0_real, req0_pad, req1_real, req1_pad, ...]
            current_idx = 0
            for i, req_id in enumerate(req_ids):
                scheduled = int(num_scheduled_tokens_padded[i])
                padding_count = padding_map.get(req_id, 0)
                actual = scheduled - padding_count

                if padding_count > 0:
                    # Padding tokens start at current_idx + actual
                    # and end at current_idx + scheduled
                    pad_start = current_idx + actual
                    pad_end = current_idx + scheduled

                    for block_table in self.input_batch.block_table.block_tables:
                        block_table.slot_mapping.np[pad_start:pad_end] = NULL_BLOCK_ID

                    logger.debug(
                        "SLOT_MAPPING: Request %s: set slots [%s:%s] "
                        "(%s padding tokens) to NULL_BLOCK_ID (0)",
                        req_id,
                        pad_start,
                        pad_end,
                        padding_count,
                    )

                current_idx += scheduled

        # padded_num_reqs / max_decode_ctx_len were already synced across DP
        # (and the spec strip applied) at the top of this method, before the
        # token tensors above were built — see the DP-coordination block there.
        # They are carried down as locals; nothing to recompute here.
        if padded_num_reqs != num_reqs:
            logger.debug(
                "DECODE_BATCH_PADDING: Need to pad %s requests to %s requests "
                "(num_seqs_buckets: %s)",
                num_reqs,
                padded_num_reqs,
                self.neuron_config.num_seqs_buckets,
            )

        # Apply decode batch padding to slot_mapping if needed
        total_num_scheduled_tokens_padded = self._apply_decode_slot_mapping_padding(
            num_reqs,
            padded_num_reqs,
            total_num_scheduled_tokens_padded,
            max_num_scheduled_tokens_padded,
        )

        self._save_slot_mapping_to_device(total_num_scheduled_tokens_padded)

        # Compute cached_seq_len for segmented prefill
        cached_seq_len = self._compute_cached_seq_len()

        # Build attention metadata
        attn_metadata = self._build_attention_metadata(
            padded_num_reqs,
            total_num_scheduled_tokens_padded,  # Includes decode padding
            max_num_scheduled_tokens_padded,
            max_num_draft_tokens,
            cached_seq_len,
            max_decode_ctx_len=max_decode_ctx_len,
        )

        # Spec decoding
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # Only sample logits for the last token of each request
            # Since padding was applied, we need to compute indices based on padded cumsum
            if padding_map:
                # Compute padded cumsum for correct indexing into output tensor
                padded_cu_num_tokens = np.cumsum(num_scheduled_tokens_padded)
                # Last real token is at: (end of request's padded segment) - 1 - padding_count
                logits_indices_list = []
                for i, req_id in enumerate(req_ids):
                    # Index of last position in this request's padded segment
                    end_idx = int(padded_cu_num_tokens[i]) - 1
                    # Subtract padding to get last real token
                    req_padding = padding_map.get(req_id, 0)
                    actual_last_token_idx = end_idx - req_padding
                    logits_indices_list.append(actual_last_token_idx)

                logits_indices = torch.tensor(logits_indices_list, dtype=torch.long)
                logger.debug(
                    "Computed logits_indices with padding: "
                    "padded_cu=%s, padding_map=%s, logits_indices=%s",
                    padded_cu_num_tokens.tolist(),
                    padding_map,
                    logits_indices.tolist(),
                )
            else:
                # No padding, so use the cumsum directly. `from_numpy` wraps the existing
                # buffer, where `torch.tensor(x.tolist())` would box and unbox one Python int
                # per request, and `cu_num_tokens - 1` gives a new array that does not alias
                # the cumsum. Keep the int64 cast: the graph takes this as an input and the
                # earlier code used `dtype=torch.long`. It is normally free, and it states
                # the contract.
                logits_indices = torch.from_numpy(
                    (cu_num_tokens - 1).astype(np.int64, copy=False)
                )

            num_draft_tokens = None
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)

            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            # Build spec decode metadata, padded to the decode bucket so
            # tensor shapes match the warmed-up decode-with-spec NEFF.
            spec_decode_metadata = self._build_spec_decode_metadata(
                input_ids,
                num_draft_tokens,
                cu_num_tokens,
                padded_num_reqs=padded_num_reqs,
            )
            logits_indices = spec_decode_metadata.logits_indices

            logger.debug("spec_decode_metadata=%s", spec_decode_metadata)

        logger.debug(
            "Completed vectorized model input preparation: "
            "input_ids.shape=%s, positions.shape=%s",
            input_ids.shape,
            cache_positions.shape,
        )

        # Apply decode batch padding if needed
        # This pads all tensors to match compiled NEFF shapes in one pass
        (
            input_ids,
            cache_positions,
            logits_indices,
            spec_decode_metadata,
            rotary_position_ids,
        ) = self._pad_to_compiled_shapes(
            input_ids,
            cache_positions,
            logits_indices,
            spec_decode_metadata,
            padded_num_reqs,
            rotary_position_ids=rotary_position_ids,
        )

        # Multimodal path: gather encoder embeddings from on-device encoder cache.
        # Only during prefill — decode NEFF has no encoder inputs
        # (encoder contributions are already in KV cache from prefill).
        mm_kwargs: dict[str, Any] = {}
        is_prefill_step = not self._is_decode()
        # Gather encoder embeddings only when the encoder cache exists. A
        # text-only run of a multimodal model (for example a Qwen3-ASR text
        # decode) has no cache, so it takes the plain text path.
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and is_prefill_step
        ):
            num_tokens = input_ids.shape[0]
            encoder_embedding_buffer, encoder_block_ids, encoder_positions = (
                self._gather_mm_embeddings(
                    num_tokens, req_ids, scheduler_output, padding_map
                )
            )
            mm_kwargs["encoder_embedding_buffer"] = encoder_embedding_buffer
            mm_kwargs["encoder_block_ids"] = encoder_block_ids
            mm_kwargs["encoder_positions"] = encoder_positions

        # Standard prompt_embeds path (for non-multimodal text embedding injection).
        # Uses state vars for backward compatibility with existing prompt_embeds
        # infrastructure (warmup, dummy batch save/restore, etc.).
        self._current_inputs_embeds = None
        self._current_is_token_ids = None
        prompt_embeds_path_active = False
        if not mm_kwargs and self.enable_prompt_embeds:
            req_prompt_embeds = getattr(self.input_batch, "req_prompt_embeds", None)
            if req_prompt_embeds:
                for req_id in req_ids:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    if req_idx is None or req_idx not in req_prompt_embeds:
                        continue
                    num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                    if num_sched <= 0:
                        continue
                    req_embeds = req_prompt_embeds[req_idx]
                    start_pos = int(self.input_batch.num_computed_tokens_cpu[req_idx])
                    if start_pos < req_embeds.shape[0]:
                        prompt_embeds_path_active = True
                        break
            if prompt_embeds_path_active:
                self._current_inputs_embeds, self._current_is_token_ids = (
                    self._build_prompt_embeds_tensors(
                        input_ids,
                        token_indices_tensor,
                        req_ids,
                        scheduler_output,
                        padding_map,
                    )
                )

        return (
            input_ids,
            cache_positions,
            logits_indices,
            attn_metadata,
            spec_decode_metadata,
            rotary_position_ids,
            mm_kwargs,
        )

    def _maybe_replicate_for_spec_decode(
        self,
        per_seq_params: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> torch.Tensor:
        """Replicate per-sequence sampling params for the spec-decode verify step.

        When speculative decoding is active, the sampler operates on
        ``num_seqs * (num_speculative_tokens + 1)`` logits rows (one per
        draft + bonus position), so the ``[num_seqs, ...]`` sampling params
        need to be replicated row-wise to match. Returns ``per_seq_params``
        unchanged when speculative decoding is not active.
        """
        if spec_decode_metadata is None or self.speculative_config is None:
            return per_seq_params
        return replicate_per_seq_rows(
            per_seq_params, self.speculative_config.num_speculative_tokens + 1
        )

    def _build_spec_decode_metadata(
        self,
        input_ids: torch.Tensor,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int | None = None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        #
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106, 206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]
        #
        # If `padded_num_reqs` > len(num_draft_tokens), fake full-spec reqs
        # are appended so the returned tensors have the shapes expected by
        # `_create_warmup_spec_decode_metadata` (bucket-sized decode batch).
        # The fake rows point at the last real token slot — rejection
        # sampling on them is a no-op and their outputs are trimmed
        # downstream.

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Guard: In DI mixed batches, a request may have fewer scheduled
        # tokens than num_draft_tokens + 1 (e.g., newly arrived request
        # with partial tokens). Clamp num_draft_tokens to avoid negative
        # logits_indices from cu_num_scheduled_tokens - num_sampled_tokens.
        # Skip the check when total scheduled tokens matches the uniform
        # expectation (common non-DI case).
        total_scheduled = int(cu_num_scheduled_tokens[-1])
        total_expected = int(num_sampled_tokens.sum())
        if total_scheduled < total_expected:
            tokens_per_req = np.diff(cu_num_scheduled_tokens, prepend=0)
            max_drafts = np.maximum(tokens_per_req - 1, 0)
            num_draft_tokens = np.minimum(num_draft_tokens, max_drafts)
            num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int64
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int64
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens)
        logits_indices = torch.from_numpy(logits_indices)
        target_logits_indices = torch.from_numpy(target_logits_indices)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices)

        # Populate draft_token_ids:
        # - Sync mode: CPU gather from input_ids (the scheduler placed real
        #   draft tokens in these positions). Matches upstream vLLM pattern.
        # - Async mode: input_ids may still be CPU with `-1` placeholders (the
        #   real drafts only exist on device as the previous step's draft
        #   NEFF output future). Fill a zero placeholder sized to match the
        #   warmup shape (total drafts across the padded batch) so
        #   torch.compile's input-guard on draft_token_ids.shape[0] holds;
        #   execute_model overwrites with the draft future post-async-swap
        #   (zero-compute view). When the previous step skipped propose
        #   (e.g. near max_model_len), no draft future exists — the zero
        #   placeholder flows through and the rejection sampler rejects
        #   all draft slots (since target's sampled tokens differ from 0),
        #   so the target verifies only the bonus token for this step.
        #   Progress is maintained without a recompile.
        if self.use_async_scheduling:
            num_reqs_real = len(num_draft_tokens)
            num_spec_tokens_config = (
                self.speculative_config.num_speculative_tokens
                if self.speculative_config is not None
                else 0
            )
            total_draft_tokens = int(num_draft_tokens.sum())
            if padded_num_reqs is not None and padded_num_reqs > num_reqs_real:
                total_draft_tokens += (
                    padded_num_reqs - num_reqs_real
                ) * num_spec_tokens_config
            draft_token_ids = torch.zeros(total_draft_tokens, dtype=torch.int32)
        else:
            draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]

        num_reqs = len(num_draft_tokens)
        if padded_num_reqs is None or padded_num_reqs <= num_reqs:
            return SpecDecodeMetadata(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens.tolist(),
                cu_num_draft_tokens=cu_num_draft_tokens,
                cu_num_sampled_tokens=cu_num_sampled_tokens,
                target_logits_indices=target_logits_indices,
                bonus_logits_indices=bonus_logits_indices,
                logits_indices=logits_indices,
            )

        # Append fake full-spec reqs so shapes match the warmed-up NEFF.
        assert self.speculative_config is not None, (
            "padded_num_reqs requires speculative_config"
        )
        num_spec_tokens = self.speculative_config.num_speculative_tokens
        num_pad_reqs = padded_num_reqs - num_reqs
        pad_draft = num_pad_reqs * num_spec_tokens

        padded_num_draft = num_draft_tokens.tolist() + [num_spec_tokens] * num_pad_reqs

        last_draft = int(cu_num_draft_tokens[-1].item())
        last_sampled = int(cu_num_sampled_tokens[-1].item())
        cu_draft_pad = torch.tensor(
            [last_draft + (i + 1) * num_spec_tokens for i in range(num_pad_reqs)],
            dtype=cu_num_draft_tokens.dtype,
        )
        cu_sampled_pad = torch.tensor(
            [
                last_sampled + (i + 1) * (num_spec_tokens + 1)
                for i in range(num_pad_reqs)
            ],
            dtype=cu_num_sampled_tokens.dtype,
        )
        cu_num_draft_tokens = torch.cat([cu_num_draft_tokens, cu_draft_pad])
        cu_num_sampled_tokens = torch.cat([cu_num_sampled_tokens, cu_sampled_pad])

        # Fake reqs reuse the last real draft index so gather is safe.
        safe_target_idx = (
            int(target_logits_indices[-1].item())
            if target_logits_indices.numel() > 0
            else 0
        )
        target_logits_indices = torch.cat(
            [
                target_logits_indices,
                torch.full(
                    (pad_draft,), safe_target_idx, dtype=target_logits_indices.dtype
                ),
            ]
        )

        # bonus_logits_indices: fake reqs point into the padded logits slots.
        # Use last_sampled (actual end of real tokens) as base, not
        # num_reqs * (num_spec_tokens + 1) which assumes all real requests
        # have full spec tokens (incorrect in DI mixed batches).
        bonus_logits_indices = torch.cat(
            [
                bonus_logits_indices,
                torch.tensor(
                    [
                        last_sampled + i * (num_spec_tokens + 1) + num_spec_tokens
                        for i in range(num_pad_reqs)
                    ],
                    dtype=bonus_logits_indices.dtype,
                ),
            ]
        )

        logits_indices = torch.cat(
            [
                logits_indices,
                torch.arange(
                    last_sampled,
                    last_sampled + num_pad_reqs * (num_spec_tokens + 1),
                    dtype=logits_indices.dtype,
                ),
            ]
        )

        # Sync mode: extend draft_token_ids for padded fake reqs so the
        # flattened tensor matches padded shapes in the rejection sampler.
        # Async mode: draft_token_ids is still the empty placeholder; will
        # be overwritten by execute_model from the draft future.
        if not self.use_async_scheduling:
            pad_draft_ids = torch.zeros(
                pad_draft, dtype=draft_token_ids.dtype, device=draft_token_ids.device
            )
            draft_token_ids = torch.cat([draft_token_ids, pad_draft_ids])

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=padded_num_draft,
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _decode_token_threshold(self) -> int:
        """Largest `max_query_len` still classified as decode.

        With spec decode the target verifies `1 + num_speculative_tokens`
        tokens per request per step. With jump-forward decoding, the JF NEFF
        processes the extra `max_jump_forward_tokens` per request.
        Otherwise decode is 1 token per request.
        """
        if self.speculative_config is not None:
            return 1 + self.speculative_config.num_speculative_tokens
        if self.neuron_config.max_jump_forward_tokens > 0:
            return 1 + self.neuron_config.max_jump_forward_tokens
        return 1

    def _create_padded_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        scheduler_output: Any,
        req_ids: list[str],
        num_scheduled_tokens_padded: np.ndarray,
        rotary_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, int]]:
        """
        Create padded input tensors for prefill requests.

        Args:
            input_ids: Original input token IDs [T]
            positions: Sequential position IDs [T]
            scheduler_output: Scheduler output with padding info
            req_ids: List of request IDs
            num_scheduled_tokens_padded: Array of scheduled token counts per request
            rotary_position_ids: M-RoPE positions [3, T] or None

        Returns:
            Tuple of (padded_input_ids, padded_positions, padded_rotary_position_ids, padding_map)
            padding_map: dict mapping req_id to number of padding tokens
        """
        padding_map = {}

        # Check if self.requests exists
        if not hasattr(self, "requests") or not self.requests:
            logger.warning(
                "INPUT_PADDING: self.requests not available, skipping padding"
            )
            return input_ids, positions, rotary_position_ids, padding_map

        # Check if any requests need padding (prefill requests)
        # NOTE: A request is in prefill if num_computed_tokens < num_prompt_tokens.
        # We cannot use "num_computed_tokens == 0" because prefix caching may have
        # already set num_computed_tokens to a non-zero value for cached tokens.
        has_prefill = any(
            req_id in self.requests
            and self.requests[req_id].num_computed_tokens
            < self.requests[req_id].num_prompt_tokens
            for req_id in req_ids
        )

        if not has_prefill:
            logger.debug("INPUT_PADDING: No prefill requests, skipping padding")
            return input_ids, positions, rotary_position_ids, padding_map

        logger.debug(
            "=== INPUT_PADDING: Creating padded inputs for %s requests ===",
            len(req_ids),
        )

        # Calculate padding for each request
        padded_segments = []
        position_segments = []
        rotary_segments = [] if rotary_position_ids is not None else None
        current_pos = 0
        total_padding_added = 0

        for i, req_id in enumerate(req_ids):
            # Check if request exists
            if req_id not in self.requests:
                logger.warning(
                    "INPUT_PADDING: Request %s not found in self.requests, "
                    "skipping padding",
                    req_id,
                )
                # Use original tokens without padding
                actual_tokens = num_scheduled_tokens_padded[i]
                actual_input_ids = input_ids[current_pos : current_pos + actual_tokens]
                actual_positions = positions[current_pos : current_pos + actual_tokens]
                padded_segments.append(actual_input_ids)
                position_segments.append(actual_positions)
                if rotary_segments is not None:
                    rotary_segments.append(
                        rotary_position_ids[
                            :, current_pos : current_pos + actual_tokens
                        ]
                    )
                current_pos += actual_tokens
                continue

            req_state = self.requests[req_id]
            # NOTE: Use same prefill detection as has_prefill above (handles prefix caching)
            num_prompt_tokens = req_state.num_prompt_tokens
            is_prefill = req_state.num_computed_tokens < num_prompt_tokens

            scheduled_tokens_padded = scheduler_output.num_scheduled_tokens_padded[
                req_id
            ]
            actual_tokens = scheduler_output.num_scheduled_tokens[req_id]
            padding_needed = scheduled_tokens_padded - actual_tokens
            padding_map[req_id] = padding_needed

            # Extract actual tokens
            actual_input_ids = input_ids[current_pos : current_pos + actual_tokens]
            actual_positions = positions[current_pos : current_pos + actual_tokens]
            actual_rotary = (
                rotary_position_ids[:, current_pos : current_pos + actual_tokens]
                if rotary_position_ids is not None
                else None
            )

            if padding_needed > 0:
                # Create padding tokens (use token_id=0)
                pad_ids = torch.zeros(padding_needed, dtype=torch.int32)
                # INVARIANT: Padding positions must be <= last real position.
                # _compute_slot_mapping_cpu processes these before the padding
                # overwrite; unbounded values cause OOB in block_table lookup.
                last_pos = (
                    actual_positions[-1].item() if len(actual_positions) > 0 else 0
                )
                pad_positions = torch.full(
                    (padding_needed,), last_pos, dtype=torch.long
                )

                # Concatenate actual + padding
                padded_input_ids = torch.cat([actual_input_ids, pad_ids])
                padded_positions = torch.cat([actual_positions, pad_positions])

                if actual_rotary is not None:
                    pad_rotary = torch.zeros(3, padding_needed, dtype=torch.long)
                    padded_rotary = torch.cat([actual_rotary, pad_rotary], dim=1)
                else:
                    padded_rotary = None

                total_padding_added += padding_needed

                # Enhanced logging
                logger.debug(
                    "INPUT_PADDING: Request %s (%s): actual=%s, scheduled=%s, "
                    "padding=%s, input_ids: %s -> %s, positions: [%s..%s] + "
                    "%s padding tokens at position %s",
                    req_id,
                    "PREFILL" if is_prefill else "DECODE",
                    actual_tokens,
                    scheduled_tokens_padded,
                    padding_needed,
                    actual_input_ids.shape,
                    padded_input_ids.shape,
                    actual_positions[0].item(),
                    actual_positions[-1].item(),
                    padding_needed,
                    last_pos,
                )
            else:
                padded_input_ids = actual_input_ids
                padded_positions = actual_positions
                padded_rotary = actual_rotary
                logger.debug(
                    "INPUT_PADDING: Request %s (%s): actual=%s, no padding needed",
                    req_id,
                    "PREFILL" if is_prefill else "DECODE",
                    actual_tokens,
                )

            padded_segments.append(padded_input_ids)
            position_segments.append(padded_positions)
            if rotary_segments is not None:
                rotary_segments.append(padded_rotary)
            current_pos += actual_tokens

        # Concatenate all segments
        final_input_ids = torch.cat(padded_segments)
        final_positions = torch.cat(position_segments)
        final_rotary = (
            torch.cat(rotary_segments, dim=1) if rotary_segments is not None else None
        )

        original_size = input_ids.shape[0]
        final_size = final_input_ids.shape[0]
        overhead_pct = (
            ((final_size - original_size) / original_size * 100)
            if original_size > 0
            else 0
        )

        logger.debug(
            "INPUT_PADDING: Summary - Original: %s tokens, Final: %s tokens, "
            "Padding added: %s (%.1f%% overhead)",
            original_size,
            final_size,
            total_padding_added,
            overhead_pct,
        )

        return final_input_ids, final_positions, final_rotary, padding_map

    def _apply_decode_slot_mapping_padding(
        self, num_reqs: int, padded_num_reqs: int, total_tokens: int, max_query_len: int
    ) -> int:
        """
        Apply decode batch padding to slot_mapping in block tables.

        For decode, pads slot_mapping with null-block writes to match the batch bucket.

        Args:
            num_reqs: Number of actual requests
            padded_num_reqs: Padded number of requests (from get_decode_padded_batch_size)
            total_tokens: Total tokens (should equal num_reqs for decode)
            max_query_len: Max query length (should be 1 for decode)

        Returns:
            Total token count after padding for decode batch
        """

        # Only apply to decode phase. tokens_per_req = max_query_len:
        #   1 for non-spec decode, 1+num_spec_tokens for spec decode.
        # Prefill (large max_query_len) is skipped.
        tokens_per_req = max_query_len
        decode_token_threshold = self._decode_token_threshold()
        if max_query_len < 1 or max_query_len > decode_token_threshold:
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: Skipping - not decode phase "
                "(max_query_len=%s, total_tokens=%s, num_reqs=%s)",
                max_query_len,
                total_tokens,
                num_reqs,
            )
            return total_tokens

        # Handle mixed batches: in DI, new requests (1 token each) can be
        # scheduled alongside existing requests with spec decode tokens.
        # In this case total_tokens won't equal num_reqs * tokens_per_req.
        # We still need to pad slot_mapping to padded_num_reqs * tokens_per_req.
        is_mixed_spec_batch = (
            tokens_per_req > 1
            and total_tokens != num_reqs * tokens_per_req
            and total_tokens > num_reqs
            and total_tokens <= num_reqs * tokens_per_req
        )

        if not is_mixed_spec_batch and total_tokens != num_reqs * tokens_per_req:
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: Skipping - not decode phase "
                "(max_query_len=%s, total_tokens=%s, num_reqs=%s)",
                max_query_len,
                total_tokens,
                num_reqs,
            )
            return total_tokens

        if padded_num_reqs == num_reqs and not is_mixed_spec_batch:
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: No padding needed (num_reqs=%s)",
                num_reqs,
            )
            return total_tokens

        num_pad_reqs = padded_num_reqs - num_reqs

        logger.debug(
            "DECODE_SLOT_MAPPING_PADDING: Padding slot_mapping from %s to %s requests "
            "(num_seqs_buckets: %s, adding %s padding slots, tokens_per_req=%s, "
            "mixed_batch=%s)",
            num_reqs,
            padded_num_reqs,
            self.neuron_config.num_seqs_buckets,
            num_pad_reqs,
            tokens_per_req,
            is_mixed_spec_batch,
        )

        # For mixed batches, pad after actual tokens; otherwise pad after requests.
        pad_start = total_tokens if is_mixed_spec_batch else num_reqs * tokens_per_req
        pad_end = padded_num_reqs * tokens_per_req

        for block_table in self.input_batch.block_table.block_tables:
            block_table.slot_mapping.np[pad_start:pad_end] = NULL_BLOCK_ID
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: Set slot_mapping[%s:%s] = %s",
                pad_start,
                pad_end,
                NULL_BLOCK_ID,
            )

        padded_total_tokens = padded_num_reqs * tokens_per_req

        logger.debug(
            "DECODE_SLOT_MAPPING_PADDING: Padded total_tokens: %s -> %s",
            total_tokens,
            padded_total_tokens,
        )

        return padded_total_tokens

    def _pad_logit_mask_for_decode(
        self, logit_mask: torch.Tensor, padded_batch_size: int
    ) -> torch.Tensor:
        """Pad logit_mask to match decode batch padding size.

        During decode, input_ids may be padded to a bucket size larger than
        the number of active requests. The logit_mask must match so that
        masked_fill inside the compiled model sees matching batch dimensions.

        Padding rows are all-True (no constraints on dummy sequences).

        Args:
            logit_mask: [num_reqs, vocab_size] boolean mask.
            padded_batch_size: Target batch size after decode padding.

        Returns:
            Padded logit_mask if decode padding was applied, otherwise unchanged.
        """
        if (
            padded_batch_size <= self.max_num_reqs
            and logit_mask.shape[0] < padded_batch_size
        ):
            pad_rows = padded_batch_size - logit_mask.shape[0]
            return torch.nn.functional.pad(logit_mask, (0, 0, 0, pad_rows), value=True)
        return logit_mask

    def _use_lm_head_sharded_logit_mask(self) -> bool:
        """Whether ODS logits are local to an lm-head vocab shard."""
        return (
            getattr(self, "on_device_sampling", False) is True
            and not getattr(self, "use_async_scheduling", False)
            and isinstance(getattr(self, "lm_head_tp_size", 1), int)
            and self.lm_head_tp_size > 1
        )

    def _lm_head_vocab_shard_size(self) -> int:
        lm_head_tp_size = getattr(self, "lm_head_tp_size", 1)
        if self.vocab_size % lm_head_tp_size != 0:
            raise ValueError(
                "vocab_size must be divisible by lm_head_tp_size for "
                "logit-mask sharding. "
                f"Got vocab_size={self.vocab_size}, "
                f"lm_head_tp_size={lm_head_tp_size}."
            )
        return self.vocab_size // lm_head_tp_size

    def _slice_logit_mask_for_lm_head_tp(
        self, logit_mask: torch.Tensor
    ) -> torch.Tensor:
        """Slice a full-vocab mask to this rank's lm-head vocab shard."""
        shard_size = self._lm_head_vocab_shard_size()
        start = self.lm_head_tp_rank * shard_size
        end = start + shard_size
        # Allocate fresh storage so single-row slices do not keep the parent
        # full-vocab row stride. torch.compile guards on exact stride.
        shard = torch.empty(
            (logit_mask.shape[0], shard_size),
            dtype=logit_mask.dtype,
            device=logit_mask.device,
        )
        shard.copy_(logit_mask[:, start:end])
        return shard

    def _build_lm_head_dp_mask_send_buffer(
        self, logit_mask: torch.Tensor
    ) -> torch.Tensor:
        """Stack per-destination vocab slices for lm-head-DP all-to-all.

        Ranks in an lm_head_dp column own different lm-head TP vocab shards.
        Each source rank starts with local batch rows and a full-vocab grammar
        mask, then sends each peer the peer's vocab slice. After all-to-all,
        every rank has all batch rows for its own vocab shard.
        """
        lm_head_dp_size = getattr(self, "lm_head_dp_size", 1)
        if self.lm_head_tp_size % lm_head_dp_size != 0:
            raise ValueError(
                "lm_head_tp_size must be divisible by lm_head_dp_size. "
                f"Got lm_head_tp_size={self.lm_head_tp_size}, "
                f"lm_head_dp_size={lm_head_dp_size}."
            )

        base_tp_size = self.lm_head_tp_size // lm_head_dp_size
        tp_position = self.lm_head_tp_rank % base_tp_size
        shard_size = self._lm_head_vocab_shard_size()

        if logit_mask.shape[-1] == shard_size:
            return torch.cat([logit_mask] * lm_head_dp_size, dim=0).contiguous()
        if logit_mask.shape[-1] != self.vocab_size:
            raise ValueError(
                "logit_mask width must be either full vocab or lm-head shard "
                "for lm_head_dp exchange. "
                f"Got width={logit_mask.shape[-1]}, "
                f"vocab_size={self.vocab_size}, shard_size={shard_size}."
            )

        chunks = []
        for dp_rank in range(lm_head_dp_size):
            dst_lm_head_tp_rank = dp_rank * base_tp_size + tp_position
            start = dst_lm_head_tp_rank * shard_size
            end = start + shard_size
            chunks.append(logit_mask[:, start:end])
        return torch.cat(chunks, dim=0).contiguous()

    def _exchange_lm_head_dp_logit_mask(self, logit_mask: torch.Tensor) -> torch.Tensor:
        """Exchange a logit mask into [B_lm_head_dp_gathered, vocab_shard]."""
        if self.lm_head_dp_group is None:
            raise RuntimeError(
                "lm_head_dp_group is required for lm_head_dp logit-mask exchange"
            )

        send = self._build_lm_head_dp_mask_send_buffer(logit_mask)
        # Split device move and dtype cast; XLA cannot fuse cross-device casts.
        send_u8 = send.to(device="cpu").to(dtype=torch.uint8).contiguous()
        recv_u8 = torch.empty_like(send_u8)
        group = getattr(self.lm_head_dp_group, "cpu_group", None)
        if group is None:
            raise RuntimeError(
                "lm_head_dp_group.cpu_group is required for CPU logit-mask exchange"
            )
        torch.distributed.all_to_all_single(recv_u8, send_u8, group=group)
        return recv_u8.to(dtype=torch.bool).to(device=self.device).contiguous()

    def _prepare_ods_logit_mask(
        self,
        logit_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Prepare logit_mask so it matches ODS lm-head logits."""
        if logit_mask is None:
            return None
        if not self._use_lm_head_sharded_logit_mask():
            return logit_mask.to(device=self.device).contiguous()

        shard_size = self._lm_head_vocab_shard_size()
        if getattr(self, "lm_head_dp_size", 1) > 1:
            return self._exchange_lm_head_dp_logit_mask(logit_mask)
        if logit_mask.shape[-1] == shard_size:
            return logit_mask.to(device=self.device).contiguous()
        if logit_mask.shape[-1] != self.vocab_size:
            raise ValueError(
                "logit_mask width must be either full vocab or lm-head shard. "
                f"Got width={logit_mask.shape[-1]}, "
                f"vocab_size={self.vocab_size}, shard_size={shard_size}."
            )
        return (
            self._slice_logit_mask_for_lm_head_tp(logit_mask)
            .to(device=self.device)
            .contiguous()
        )

    def _build_noop_logit_mask(
        self,
        num_logit_rows: int,
        device: torch.device,
        compiled_graph_input: bool = False,
    ) -> torch.Tensor | None:
        """Build the all-True mask used for mixed SO/SO-off graph stability.

        When enable_structured_outputs is false, the server is explicitly
        configured for SO-off-only traffic and can skip this no-op mask path.

        ``compiled_graph_input`` builds the shape seen at the model.forward
        boundary during warmup/graph capture. With runner-side lm_head_dp mask
        exchange, runtime masks have already been row-gathered before model
        entry, so warmup must use the same gathered-row shape.
        """
        if not self.neuron_config.enable_structured_outputs:
            return None
        sharded = self._use_lm_head_sharded_logit_mask()
        vocab_width = self._lm_head_vocab_shard_size() if sharded else self.vocab_size
        num_rows = num_logit_rows
        if compiled_graph_input and sharded and getattr(self, "lm_head_dp_size", 1) > 1:
            num_rows = num_logit_rows * self.lm_head_dp_size
        return torch.ones(num_rows, vocab_width, dtype=torch.bool, device=device)

    def _raise_if_structured_outputs_disabled(
        self, grammar_bitmask: torch.Tensor | None
    ) -> None:
        if grammar_bitmask is None or self.neuron_config.enable_structured_outputs:
            return
        raise ValueError(SO_DISABLED_MESSAGE)

    def _pad_to_compiled_shapes(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        spec_decode_metadata: "SpecDecodeMetadata | None",
        padded_num_reqs: int,
        rotary_position_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        "SpecDecodeMetadata | None",
        torch.Tensor | None,
    ]:
        """
        Single-pass padding to ensure all tensors match compiled NEFF shapes.

        Neuron requires exact shape matches against pre-compiled graphs.
        This method centralizes all decode-phase tensor padding:
          - input_ids, positions → padded_num_reqs * tokens_per_req
          - logits_indices (sampling_positions) → padded_num_reqs * tokens_per_req
          - spec_decode_metadata.draft_token_ids → padded_num_reqs * num_spec_tokens

        Slot_mapping padding is handled separately in _apply_decode_slot_mapping_padding
        (must run before _build_attention_metadata).

        Args:
            input_ids: Input token IDs [num_tokens]
            positions: Sequential token positions [num_tokens]
            logits_indices: Indices for logits extraction
            spec_decode_metadata: Speculative decode metadata (may be None)
            padded_num_reqs: Target number of requests (from bucket sizing)
            rotary_position_ids: M-RoPE positions [3, num_tokens] or None

        Returns:
            Tuple of (input_ids, positions, logits_indices, spec_decode_metadata, rotary_position_ids)
            with all tensors padded to compiled shapes.
        """
        num_reqs = len(self.input_batch.req_ids)
        num_tokens = input_ids.shape[0]
        tokens_per_req = self._decode_token_threshold()

        # Detect effective tokens_per_req for this step.
        # DI first decode after KV transfer: 1 token per request even with
        # spec decode enabled (no draft tokens yet).
        if num_tokens == num_reqs and tokens_per_req > 1:
            tokens_per_req = 1

        # Determine if this is a decode batch that needs padding.
        # A "mixed spec batch" occurs in DI when some requests have draft
        # tokens and others don't (newly arrived from prefill).
        is_mixed_spec_batch = (
            tokens_per_req > 1
            and num_tokens != num_reqs * tokens_per_req
            and num_tokens != num_reqs
            and num_tokens > num_reqs
            and num_tokens <= num_reqs * tokens_per_req
        )

        # Skip if not a decode phase
        if not is_mixed_spec_batch and num_tokens != num_reqs * tokens_per_req:
            return (
                input_ids,
                positions,
                logits_indices,
                spec_decode_metadata,
                rotary_position_ids,
            )

        # Skip if no buckets configured
        if not self.neuron_config.num_seqs_buckets:
            return (
                input_ids,
                positions,
                logits_indices,
                spec_decode_metadata,
                rotary_position_ids,
            )

        # Skip if already at target size and not a mixed batch
        if padded_num_reqs == num_reqs and not is_mixed_spec_batch:
            return (
                input_ids,
                positions,
                logits_indices,
                spec_decode_metadata,
                rotary_position_ids,
            )

        # --- Compute target shape ---
        target_num_tokens = padded_num_reqs * tokens_per_req

        logger.debug(
            "PAD_TO_COMPILED_SHAPES: num_reqs=%s, padded_num_reqs=%s, "
            "num_tokens=%s, target_num_tokens=%s, tokens_per_req=%s, "
            "mixed_batch=%s",
            num_reqs,
            padded_num_reqs,
            num_tokens,
            target_num_tokens,
            tokens_per_req,
            is_mixed_spec_batch,
        )

        # --- Pad input_ids and positions ---
        if num_tokens < target_num_tokens:
            pad_size = target_num_tokens - num_tokens
            input_ids = torch.cat(
                [input_ids, torch.zeros(pad_size, dtype=input_ids.dtype)]
            )
            positions = torch.cat(
                [positions, torch.zeros(pad_size, dtype=positions.dtype)]
            )
            if rotary_position_ids is not None:
                pad_rotary = torch.zeros(3, pad_size, dtype=rotary_position_ids.dtype)
                rotary_position_ids = torch.cat(
                    [rotary_position_ids, pad_rotary], dim=1
                )

        # --- Pad logits_indices (becomes sampling_positions in model kwargs) ---
        # Padding indices must point within the valid hidden_states range.
        # Using the last real logits index is safe: it points to a position
        # that was computed during the forward pass, and the sampled result
        # at padding positions is discarded downstream (fake requests).
        if logits_indices.shape[0] < target_num_tokens:
            pad_size = target_num_tokens - logits_indices.shape[0]
            last_real_idx = logits_indices[-1] if logits_indices.numel() > 0 else 0
            pad_indices = torch.full(
                (pad_size,), last_real_idx, dtype=logits_indices.dtype
            )
            logits_indices = torch.cat([logits_indices, pad_indices])

        # --- Pad spec_decode_metadata.draft_token_ids ---
        if spec_decode_metadata is not None and self.speculative_config is not None:
            expected_draft_size = (
                padded_num_reqs * self.speculative_config.num_speculative_tokens
            )
            actual_draft_size = spec_decode_metadata.draft_token_ids.shape[0]
            if actual_draft_size < expected_draft_size:
                pad = torch.zeros(
                    expected_draft_size - actual_draft_size,
                    dtype=spec_decode_metadata.draft_token_ids.dtype,
                )
                spec_decode_metadata.draft_token_ids = torch.cat(
                    [spec_decode_metadata.draft_token_ids, pad]
                )

        logger.debug(
            "PAD_TO_COMPILED_SHAPES: Applied - input_ids: %s, "
            "logits_indices: %s, draft_token_ids: %s",
            input_ids.shape,
            logits_indices.shape,
            spec_decode_metadata.draft_token_ids.shape
            if spec_decode_metadata is not None
            else None,
        )

        return (
            input_ids,
            positions,
            logits_indices,
            spec_decode_metadata,
            rotary_position_ids,
        )

    def _validate_token_ids(self, input_ids: torch.Tensor) -> None:
        """
        Validate that all token IDs are within the vocabulary range.

        Args:
            input_ids: Input token IDs tensor

        Raises:
            ValueError: If any token ID is outside the valid range [0, vocab_size).
                Exception: -1 placeholders are allowed in async spec decode mode,
                where the scheduler pre-populates spec positions with -1 and the
                worker substitutes them device-side from draft token futures.
        """
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        # Allow -1 placeholders for async spec decode.
        allow_minus_one = (
            self.use_async_scheduling and self.speculative_config is not None
        )
        min_valid = -1 if allow_minus_one else 0
        if max_id >= self.vocab_size or min_id < min_valid:
            invalid_mask = (input_ids >= self.vocab_size) | (input_ids < min_valid)
            invalid_ids = input_ids[invalid_mask].tolist()
            raise ValueError(
                f"Token IDs out of range [{min_valid}, {self.vocab_size}). "
                f"Found min={min_id}, max={max_id}. Invalid IDs: {invalid_ids}"
            )

    def _compute_swa_num_blocks(
        self, sliding_window: int, block_size: int, bs: int
    ) -> int:
        """Number of KV blocks for a SWA decode group's block_table.

        Single source of truth for the SWA block_table width: warmup
        (compile), runtime, and the idle-DP dummy batch all call this, and
        they must agree or torch.compile's fail_on_recompile guard trips and,
        once silenced, the SWA kernel reads OOB.

        Sizing: the sliding window needs ``sliding_window // block_size + 1``
        blocks, padded up to the attention kernel's P_MAX tile divisibility
        (trimmed SWA block counts must produce S_ctx divisible by P_MAX). That
        window count is then clipped to the full-context block count
        ``ceil(max_model_len / (block_size * dcp))`` because the InputBatch
        block_table is only allocated that wide. The clip matters when
        ``sliding_window >= max_model_len`` (small / smoke configs, e.g.
        sliding_window == max_model_len == 128 with block_size=16: the window
        rounds 9 up to P_MAX-multiple 16, but the whole sequence is only 8
        blocks). In that regime trimming "to the window" would over-allocate
        past what the scheduler ever fills, so the full-context width is both
        correct and the only storage-backed choice. When the window is the
        tighter bound (the common case) the clip is a no-op, and the runtime
        gather (``_compute_swa_decode_tensors``) only runs in that
        no-op-clip regime, so its P_MAX-aligned geometry is preserved.
        """
        # Trimmed SWA kv block counts must produce S_ctx divisible by P_MAX.
        P_MAX = 128
        min_blocks = sliding_window // block_size + 1
        blocks_per_pmax = P_MAX // block_size
        num_swa_blocks = (
            (min_blocks + blocks_per_pmax - 1) // blocks_per_pmax * blocks_per_pmax
        )
        # The packed FP8 layout pairs two tokens per slot, so it needs the
        # kernel's resized block_len to stay >= 2. The resize divides by
        # sprior_n_prgs * P_MAX, so the safe condition is that s_prior is a
        # multiple of 2 * P_MAX. Raise the window count to that floor.
        #
        # Gate on bs, because only a bucket that shards s_prior needs the floor.
        # A batch-shardable bucket (bs % lnc == 0) halves the requirement, and at
        # the GPT-OSS window (s_prior 256) it already keeps block_len at 2, so
        # widening it there only costs context and pushes the bucket off
        # batch-sharding. The gate is a proxy, not a proof: a batch-sharded
        # bucket whose s_prior / P_MAX is odd still resizes to 1. That needs a
        # sliding_window outside [128, 255], which no current model has, and it
        # degrades to the un-swizzle fallback rather than to a wrong result. The
        # clip below still bounds the floor by the full-context width.
        lnc = get_logical_nc_config()
        if bs % lnc != 0:
            min_viable_blocks = (2 * lnc * P_MAX) // block_size
            num_swa_blocks = max(num_swa_blocks, min_viable_blocks)
        dcp_block_size = block_size * max(self._dcp_size, 1)
        full_ctx_blocks = (self.max_model_len + dcp_block_size - 1) // dcp_block_size
        return min(num_swa_blocks, full_ctx_blocks)

    def _compute_swa_decode_tensors(
        self,
        blk_table,
        padded_num_reqs: int,
        sliding_window: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For a SWA decode step, compute:
        - the block_table_tensor trimmed to the window-relevant blocks, so
          the attention kernel reads only ~window/block_size blocks from HBM.
        - the per-seq `swa_kv_pos_offset` tensor (= start_block * block_size),
          which the model subtracts from pos_ids so the causal mask is
          computed in the trimmed frame.
        """
        num_swa_blocks = self._compute_swa_num_blocks(
            sliding_window, block_size, padded_num_reqs
        )
        # Build the index in numpy. The gather below is a numpy operation, and the cache
        # compares numpy arrays, so a torch build would add nine host dispatches for each
        # step and then convert the result back.
        ctx_lens = self.input_batch.num_computed_tokens_cpu[:padded_num_reqs]
        start_blocks = np.clip(ctx_lens // block_size - num_swa_blocks + 1, 0, None)
        gather_index = start_blocks[:, None] + np.arange(
            num_swa_blocks, dtype=start_blocks.dtype
        )

        rows = self._raw_block_table(blk_table.block_table)[:padded_num_reqs]
        if self._swa_groups is None:
            self._swa_groups = {}
        state = self._swa_groups.setdefault(kv_cache_group_id, _SwaGroupState())
        if state.hits(gather_index, rows):
            return state.trimmed_table, state.position_offset

        trimmed_bt = self._gather_swa_columns(state, rows, gather_index).to(self.device)
        swa_kv_pos_offset = torch.from_numpy(start_blocks * block_size).to(
            dtype=torch.int32, device=self.device
        )
        # Copy both inputs. The scheduler changes the host block table in place, so a
        # view would compare the next step against itself and report a false cache hit.
        state.gather_index = gather_index.copy()
        state.rows = rows.copy()
        state.trimmed_table = trimmed_bt
        state.position_offset = swa_kv_pos_offset
        return trimmed_bt, swa_kv_pos_offset

    @staticmethod
    def _gather_swa_columns(
        state: _SwaGroupState, rows: np.ndarray, gather_index: np.ndarray
    ) -> torch.Tensor:
        """Gather the window columns of a block table on the host, then remap the null block.

        Gathering then remapping gives the same result as the reverse, because the remap works
        element by element, so the whole operation can stay on the host. A device gather instead
        pulls the full block table to the host and pushes the result back, because the single
        device operand forces the fallback path.

        Args:
            state: Per-group state that owns the staging ring.
            rows: Host block-table rows for the active requests.
            gather_index: Column index for each row.

        Returns:
            A host tensor that views one ring slot. The caller must copy it to the device.
        """
        shape = gather_index.shape
        if not state.fits(shape, rows.dtype):
            state.reset(shape, rows.dtype)
        slot = state.next_slot
        state.next_slot = (slot + 1) % _BLOCK_TABLE_STAGE_DEPTH
        stage, null_mask = state.stage[slot], state.null_mask[slot]
        # take_along_axis has no `out` parameter, so it allocates. Copy into the ring
        # slot, so the tensor that goes to the device is a stable buffer.
        np.copyto(stage, np.take_along_axis(rows, gather_index, axis=1))
        # Reuse the mask and pass `out` to keep the hot path free of allocations.
        np.equal(stage, NULL_BLOCK_ID, out=null_mask)
        np.copyto(stage, PAD_SLOT_ID, where=null_mask)
        return torch.from_numpy(stage)

    def _decode_ctx_bucket_from_max_decode_ctx_len(
        self,
        max_decode_ctx_len: int,
        max_num_draft_tokens: int,
    ) -> int | None:
        """Pick the ctx-length bucket *boundary* for a non-SWA decode group.

        Single source of truth for the raw-ctx-length → bucket-boundary
        mapping, shared by the runtime decode path
        (`_decode_ctx_blocks_from_max_decode_ctx_len`) and the idle-DP dummy
        path (`execute_dummy_batch`) so all three of runtime / warmup / dummy
        dispatch to the same NEFF. Returns the boundary (a configured bucket
        or `max_model_len`), or `None` when bucketing is unset so the caller
        falls back to `max_model_len`.

        Pure function of (max_decode_ctx_len, max_num_draft_tokens, model
        config). Under DP, callers must pass the *DP-synced* value so every
        rank picks the same bucket; `_get_dp_padding` produces it in the same
        MAX-reduce that syncs `padded_num_reqs`. SWA groups bypass this — their
        block count comes from `_compute_swa_num_blocks`, which is already
        config-deterministic across ranks.
        """
        if self.neuron_config.decode_context_length_buckets is None:
            return None

        from vllm_neuron.utils.bucket_utils import get_bucket_for_count

        # Eagle uses 1 + k target positions followed by k - 1 recurrent
        # draft positions. DFlash follows the target with a k + 1 query block.
        speculative_headroom = 2 * max_num_draft_tokens
        if self.is_dflash_spec and max_num_draft_tokens > 0:
            speculative_headroom += 2
        needed_seq = max_decode_ctx_len + max(1, speculative_headroom)
        return get_bucket_for_count(
            needed_seq,
            self.neuron_config.decode_context_length_buckets + [self.max_model_len],
        )

    def _decode_ctx_blocks_from_max_decode_ctx_len(
        self,
        max_decode_ctx_len: int,
        block_size: int,
        max_num_draft_tokens: int,
    ) -> int:
        """Pick max_blocks_per_seq for a non-SWA decode group from a max ctx length.

        Composes the boundary picker
        (`_decode_ctx_bucket_from_max_decode_ctx_len`) with the
        boundary → block-count conversion. The divisor matches the one used
        by `_build_warmup_attention_metadata` so the runtime block_table shape
        matches the compiled NEFF shape. See that helper for the DP-sync
        contract.
        """
        ctx_bucket = self._decode_ctx_bucket_from_max_decode_ctx_len(
            max_decode_ctx_len, max_num_draft_tokens
        )
        # With DCP, each rank stores only ctx_for_blocks / dcp_size tokens.
        dcp_block_size = block_size * max(self._dcp_size, 1)
        ctx_for_blocks = ctx_bucket if ctx_bucket is not None else self.max_model_len
        return (ctx_for_blocks + dcp_block_size - 1) // dcp_block_size

    def _build_attention_metadata(
        self,
        padded_num_reqs: int,
        total_num_scheduled_tokens: int,
        max_query_len: int,
        max_num_draft_tokens: int,
        cached_seq_len: int = 0,
        max_decode_ctx_len: int = 0,
    ) -> AttentionMetadata | None:
        """
        Build attention metadata for KV cache and attention computation.

        Args:
            padded_num_reqs: Padded number of requests
            total_num_scheduled_tokens: Total number of tokens in the batch
            max_query_len: Max size of each batch line
            max_num_draft_tokens: Max number of draft tokens for spec decode
            cached_seq_len: Actual number of valid cached tokens
            max_decode_ctx_len: DP-synchronized max ctx across active
                slots (already MAX-reduced via `_get_dp_padding`). Drives the
                `decode_context_length_buckets` pick for non-SWA decode
                groups; ignored when bucketing is unset or the group is SWA.

        Returns:
            AttentionMetadata object or None for simplified cases
        """
        attn_metadata = {}
        decode_token_threshold = 1 + max_num_draft_tokens
        is_decode = max_query_len <= decode_token_threshold

        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            spec = kv_cache_group_spec.kv_cache_spec
            block_size = spec.block_size
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor(padded_num_reqs)
            # Clone so ``full_blk_table_tensor`` is a distinct tensor object
            # from ``blk_table_tensor``. Required because torch.compile's
            # input guards reject duplicate tensor identities, and warmup
            # always traces with two distinct tensors. For SWA models, the
            # clone is also necessary because ``blk_table_tensor`` gets
            # reassigned to a trimmed view below.
            full_blk_table_tensor = blk_table_tensor.clone()
            # for dcp, slot mapping would set the slot for non-owned tokens to NULL_BLOCK_ID
            # A prefix upload sizes the device tensor for the token count of its own call
            # site. Another caller can ask for more tokens, because the jump-forward branch
            # and the speculative branch use a different total. Refill the whole buffer here
            # instead of a silent truncation. A steady state does not reach this.
            #
            # Refill only when both sides are real tensors. A refill from anything else would
            # replace a usable tensor with the result of an unknown conversion.
            slot_mapping_gpu = blk_table.slot_mapping.gpu
            slot_mapping_cpu = blk_table.slot_mapping.cpu
            if (
                isinstance(slot_mapping_gpu, torch.Tensor)
                and isinstance(slot_mapping_cpu, torch.Tensor)
                and slot_mapping_gpu.shape[0] < total_num_scheduled_tokens
            ):
                blk_table.slot_mapping.gpu = slot_mapping_cpu.to(self.device)
            slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]
            if not is_decode and self.cp_world_size > 1:
                I = block_size
                W = self.cp_world_size
                ownership_stride = W * I
                if cached_seq_len % ownership_stride != 0:
                    raise ValueError(
                        "DCP prefill requires cached_seq_len to be divisible by "
                        "decode_context_parallel_size * block_size "
                        f"({W} * {I} = {ownership_stride}); got {cached_seq_len}."
                    )
            # Do not cache this tensor. ``cached_seq_len`` is the largest computed-token
            # count in the batch, and it increases on each decode step. A cache therefore
            # misses every time, and it holds one tensor for each value that the run sees.
            cached_seq_len_tensor = torch.tensor(
                [[cached_seq_len]], dtype=torch.int32, device=self.device
            )

            # Note there are more data available in self.input_batch you can
            # use in attention_metadata. Now we only use the ones below.
            #
            # Leaving kv_segment_size at 0 routes this step to full prefill:
            # every model's dispatch is `if kv_segment_size:`, so a 0 falls
            # through on its own. A Q chunk that is not a multiple of 128
            # cannot use the segmented kernel (see
            # segmented_prefill_supports_q_len), and this must agree with
            # _prefill_compile_targets, which skips the same pairs at compile
            # time -- otherwise we would dispatch a shape we never compiled.
            kv_segment_size = 0
            if (
                self.neuron_config.kv_segment_size_buckets is not None
                and segmented_prefill_supports_q_len(total_num_scheduled_tokens)
            ):
                kv_segment_size = pick_kv_segment_size(
                    cached_seq_len=cached_seq_len,
                    kv_segment_size_buckets=(
                        self.neuron_config.kv_segment_size_buckets
                    ),
                )

            swa_kv_pos_offset = None

            # Only include the raw (pre-transform) block table when running
            # the synthetic test model — it needs the scheduler's original
            # allocation (0-padded, no sentinel remap, no SWA trim).
            raw_blk_table_tensor = (
                blk_table.get_cpu_tensor()[:padded_num_reqs].to(self.device)
                if self._is_synthetic_model
                else None
            )

            # For SWA decode: slice the block table to only the window-relevant
            # blocks so the attention kernel reads the minimum necessary blocks
            # from HBM. Prefill uses the full table (writes via slot_mapping
            # span every scheduled slot, and the prefill NEFF is compiled for
            # the full shape).
            if is_decode and isinstance(spec, SlidingWindowSpec):
                num_swa_blocks = self._compute_swa_num_blocks(
                    spec.sliding_window, block_size, padded_num_reqs
                )
                if num_swa_blocks < blk_table_tensor.shape[1]:
                    blk_table_tensor, swa_kv_pos_offset = (
                        self._compute_swa_decode_tensors(
                            blk_table,
                            padded_num_reqs,
                            spec.sliding_window,
                            block_size,
                            kv_cache_group_id,
                        )
                    )
                else:
                    # Short sequence: no trimming needed, but always pass
                    # swa_kv_pos_offset to keep the FX graph consistent
                    # with warmup (avoids torch._dynamo recompilation).
                    swa_kv_pos_offset = self._cached_constant_tensor(
                        ("swa_kv_pos_offset", kv_cache_group_id, padded_num_reqs),
                        lambda: torch.zeros(
                            padded_num_reqs, dtype=torch.int32, device=self.device
                        ),
                    )

            elif (
                is_decode
                and self.neuron_config.decode_context_length_buckets is not None
            ):
                bucket_blocks = self._decode_ctx_blocks_from_max_decode_ctx_len(
                    max_decode_ctx_len=max_decode_ctx_len,
                    block_size=block_size,
                    max_num_draft_tokens=max_num_draft_tokens,
                )
                if bucket_blocks < blk_table_tensor.shape[1]:
                    # index_select rather than a slice: dynamo guards check
                    # the literal stride of the input, and a slice
                    # (`blk_table_tensor[:, :bucket_blocks]`) only changes
                    # shape — the stride still reflects the parent's full
                    # `max_blocks_per_seq`. A `.contiguous()` or `.clone()`
                    # would normally rebuild the stride, but at B=1
                    # PyTorch treats any tensor with a size-1 dim as
                    # contiguous and skips the copy, leaving the wrong
                    # stride in place. index_select always allocates a
                    # fresh row-major tensor whose stride matches what
                    # warmup compiled, so guards hit instead of recompile.
                    # The extra device-side copy is asynchronous and
                    # overlaps with the prior decode forward, so it costs
                    # no extra wall time.
                    # The index vector depends only on its length, so build it once.
                    # index_select does not write to it.
                    bucket_idx = self._cached_constant_tensor(
                        ("bucket_idx", bucket_blocks),
                        lambda: torch.arange(bucket_blocks, device=self.device),
                    )
                    blk_table_tensor = torch.index_select(
                        blk_table_tensor, 1, bucket_idx
                    )
                swa_kv_pos_offset = self._cached_constant_tensor(
                    ("swa_kv_pos_offset", kv_cache_group_id, padded_num_reqs),
                    lambda: torch.zeros(
                        padded_num_reqs, dtype=torch.int32, device=self.device
                    ),
                )

            attn_metadata_i = {
                "block_table_tensor": blk_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": max_query_len,
                "block_size": block_size,
                "max_blocks_per_seq": blk_table_tensor.shape[1],
                "decode_token_threshold": decode_token_threshold,
                "cached_seq_len": cached_seq_len_tensor,
                "kv_segment_size": kv_segment_size,
                # Full (untrimmed) block_table_tensor for use cases that
                # need to compute slot indices into the *full* KV cache,
                # not the SWA-windowed view. Currently used by
                # ``correct_spec_decode_positions_and_slot_mapping``.
                "full_block_table_tensor": full_blk_table_tensor,
            }
            if raw_blk_table_tensor is not None:
                attn_metadata_i["raw_block_table_tensor"] = raw_blk_table_tensor
            if swa_kv_pos_offset is not None:
                attn_metadata_i["swa_kv_pos_offset"] = swa_kv_pos_offset

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def _build_warmup_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        cached_seq_len: int = 0,
        decode_token_threshold: int | None = None,
        ctx_bucket: int | None = None,
        device: torch.device | None = None,
        null_kv: bool = False,
        kv_segment_size: int = 0,
    ) -> dict:
        """
        Build attention metadata for warmup without using InputBatch.

        Creates synthetic block_table_tensor and slot_mapping tensors
        that match the shapes used by InputBatch during actual inference.

        Args:
            num_tokens: Total number of tokens in the batch
            num_reqs: Number of requests in the batch
            cached_seq_len: Actual number of valid cached tokens
            decode_token_threshold: Threshold to distinguish prefill from decode.
                Defaults to 1 + num_speculative_tokens when drafter is present,
                otherwise 1.
            ctx_bucket: Decode sequence bucket size. When None, sized at
                ``max_model_len`` (today's behavior). Otherwise sized at
                ``ceil(ctx_bucket / block_size)`` blocks. Only meaningful
                for non-SWA groups; SWA groups always trim to the window.
            device: Device to allocate synthetic tensors on. Defaults to
                ``self.device``.
            kv_segment_size: Segment size to bake into the warmup
                attn_metadata. Ignored for decode and draft warmups (which
                don't run segmented attention). The prefill warmup caller
                always passes a value from kv_segment_size_buckets.

        Returns:
            dict mapping layer names to attention metadata dicts
        """
        if device is None:
            device = self.device
        attn_metadata = {}

        if decode_token_threshold is None:
            num_spec_tokens = 0
            if self.drafter is not None:
                num_spec_tokens = self.drafter.num_speculative_tokens
            decode_token_threshold = 1 + num_spec_tokens

        is_prefill = (num_tokens // num_reqs) > decode_token_threshold

        for _, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = kv_cache_group_spec.kv_cache_spec
            block_size = spec.block_size
            # Block-table seq dim. Defaults to max_model_len; overridden by
            # ctx_bucket for decode_context_length_buckets warmup. SWA path below
            # may further reduce this for SWA groups.
            # With DCP, each rank stores only ctx_for_blocks / dcp_size tokens.
            # vLLM's BlockTableManager uses total_cp_world_size = dcp_size (no PCP).
            ctx_for_blocks = (
                ctx_bucket if ctx_bucket is not None else self.max_model_len
            )
            dcp_block_size = block_size * max(self._dcp_size, 1)
            max_num_blocks_per_req = (
                ctx_for_blocks + dcp_block_size - 1
            ) // dcp_block_size
            # Match upstream's TRTLLM alignment (vLLM PR #39324): InputBatch
            # block_table width is rounded up to a multiple of 128/block_size.
            # Prefill doesn't trim at runtime so warmup must use the aligned
            # width. Decode with ctx-length buckets trims back below.
            alignment = 128 // block_size if block_size <= 128 else 1
            max_num_blocks_per_req = (
                (max_num_blocks_per_req + alignment - 1) // alignment * alignment
            )
            # Untrimmed block-table seq dim — used by the on-device
            # ``correct_spec_decode_positions_and_slot_mapping`` correction
            # which needs to compute slot indices into the *full* KV cache,
            # not the SWA-windowed view. Always sized at max_model_len.
            full_max_blocks_per_req = (
                self.max_model_len + block_size - 1
            ) // block_size

            swa_kv_pos_offset = None
            # For SWA decode, warmup must match the trimmed shape that
            # `_build_attention_metadata` produces at runtime so the compiled
            # NEFF receives the correct block_table dim. `_compute_swa_num_blocks`
            # already clips to the full-context width; here we additionally take
            # the min with the per-bucket `max_num_blocks_per_req` (which may be
            # sized from a decode_context_length_buckets `ctx_bucket` rather than
            # max_model_len).
            if not is_prefill and isinstance(spec, SlidingWindowSpec):
                num_swa_blocks = self._compute_swa_num_blocks(
                    spec.sliding_window, block_size, num_reqs
                )
                if num_swa_blocks < max_num_blocks_per_req:
                    max_num_blocks_per_req = num_swa_blocks
                swa_kv_pos_offset = torch.zeros(
                    num_reqs, dtype=torch.int32, device=device
                )
            elif (
                not is_prefill
                and self.neuron_config.decode_context_length_buckets is not None
            ):
                # Trim to the DCP-aware bucket block count, matching the
                # index_select trim that _build_attention_metadata applies at
                # runtime. Without this, the aligned width would be wider than
                # what the runtime produces after trimming.
                dcp_blocks = (ctx_for_blocks + dcp_block_size - 1) // dcp_block_size
                if dcp_blocks < max_num_blocks_per_req:
                    max_num_blocks_per_req = dcp_blocks
                # Non-SWA decode with ctx-length bucketing also passes a zero
                # swa_kv_pos_offset at runtime; warmup must include it or
                # torch.compile will refuse to recompile when the dict gains
                # a key on the first spec-decode call.
                swa_kv_pos_offset = torch.zeros(
                    num_reqs, dtype=torch.int32, device=device
                )

            # Dummy block_table_tensor: [num_reqs, max_num_blocks_per_req]
            # Use sequential block IDs starting from 0
            # When null_kv, fill the block table with the reserved null block.
            # The fused draft recomputes slot_mapping from positions and the
            # block table on each recurrent pass, so nulling only slot_mapping
            # would still let later passes write to real blocks. A null block
            # table keeps every computed slot in the discard block.
            def _block_ids(width: int) -> torch.Tensor:
                base = (
                    torch.full((width,), NULL_BLOCK_ID, dtype=torch.int32)
                    if null_kv
                    else torch.arange(width, dtype=torch.int32)
                )
                return base.unsqueeze(0).expand(num_reqs, -1).contiguous().to(device)

            block_table_tensor = _block_ids(max_num_blocks_per_req)
            # Always provide an untrimmed block_table with the full
            # ``max_model_len`` dim so the on-device spec-decode correction
            # (which needs to compute slot indices into the full KV cache)
            # traces a shape consistent with runtime.
            full_block_table_tensor = _block_ids(full_max_blocks_per_req)

            # Dummy slot_mapping: full-length [num_tokens] for all prefill paths.
            # DCP prefill now writes the FULL sequence to the per-rank cache
            # (owned tokens → real slots, non-owned → null block), matching the
            # runtime path (_build_attention_metadata), so warmup must compile the
            # NEFF for a full-length slot_mapping — no S/DCP compaction.
            # When null_kv, point every write at the reserved null block so the
            # dummy pass does not touch a real request's KV. Used by the draft
            # dummy dispatch on ranks that skip the real proposal but must keep
            # the draft cross-DP collectives in lockstep.
            if null_kv:
                slot_mapping = torch.full(
                    (num_tokens,), NULL_BLOCK_ID, dtype=torch.int64
                ).to(device)
            else:
                slot_mapping = torch.arange(num_tokens, dtype=torch.int64).to(device)

            # max_query_len: for prefill = bucket_size, for decode = 1
            max_query_len = num_tokens // num_reqs

            attn_metadata_i = {
                "block_table_tensor": block_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": max_query_len,
                "block_size": block_size,
                "max_blocks_per_seq": max_num_blocks_per_req,
                "decode_token_threshold": decode_token_threshold,
                "cached_seq_len": torch.tensor(
                    [[cached_seq_len]], dtype=torch.int32, device=device
                ),
                "kv_segment_size": kv_segment_size,
                "full_block_table_tensor": full_block_table_tensor,
            }
            if swa_kv_pos_offset is not None:
                attn_metadata_i["swa_kv_pos_offset"] = swa_kv_pos_offset

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def _build_prefill_synthetic_inputs(
        self,
        bucket_size: int,
        kv_segment_size: int,
        device: torch.device | None = None,
    ) -> dict:
        """Build synthetic inputs for prefill warmup.

        Args:
            bucket_size: Number of active tokens (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket
            device: Override device for synthetic tensors. Defaults to
                ``self.device``.

        Returns:
            dict: Keyword arguments for self.model() forward call
        """
        if device is None:
            device = self.device
        cached_seq_len = 0
        num_reqs = 1
        num_tokens = bucket_size

        # Create synthetic inputs: 1 request with bucket_size tokens
        input_ids = torch.arange(1, num_tokens + 1, dtype=torch.int32).to(device)
        # Positions start from cached_seq_len (simulating continuation after cached prefix)
        positions = torch.arange(
            cached_seq_len, cached_seq_len + num_tokens, dtype=torch.long
        )
        if self.uses_mrope:
            rotary_position_ids = (
                positions.unsqueeze(0).expand(3, -1).contiguous().to(device)
            )
        else:
            rotary_position_ids = None
        positions = positions.to(device)
        # Sample from last token only
        logits_indices = torch.tensor([num_tokens - 1], dtype=torch.long).to(device)

        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            cached_seq_len=cached_seq_len,
            # Prefill has no draft tokens, so decode_token_threshold=1
            # (matches _build_attention_metadata where max_num_draft_tokens=0).
            decode_token_threshold=1,
            device=device,
            kv_segment_size=kv_segment_size,
        )

        # Create dummy sampling params for warmup
        dummy_sampling_params = torch.tensor(
            [[0, 1.0, 1.0]],  # [top_k, top_p, temperature]
            dtype=torch.float32,
            device=device,
        )

        dummy_logit_mask = self._build_noop_logit_mask(
            num_reqs, device, compiled_graph_input=True
        )

        # Mirror rank_tensor onto the requested device. Same-device .to()
        # is a no-op so this is safe to call unconditionally.
        rank_tensor = self.rank_tensor.to(device)

        warmup_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=dummy_sampling_params,
            spec_decode_metadata=None,
            rank=rank_tensor,
            logit_mask=dummy_logit_mask,
        )
        if rotary_position_ids is not None:
            warmup_kwargs["rotary_position_ids"] = rotary_position_ids
        if self.supports_mm_inputs and self.encoder_cache is not None:
            # The encoder cache exists only when an encoder config is
            # present (see _init_encoder_cache). A text-only run of a
            # multimodal model (for example a Qwen3-ASR text decode) has no
            # cache, so it skips the encoder warmup inputs.
            max_num_encoder_blocks = self.max_encoder_blocks_per_request
            cache_buffer = self.encoder_cache.buffer
            # parallel_trace forks a pool that traces on ``device=meta`` and
            # validates that EVERY job kwarg is on meta. Aliasing the real
            # on-device cache buffer (or pinning the index tensors to its
            # device) trips that check. On the meta trace path build a
            # shape/dtype-matched synthetic buffer and keep the index tensors on
            # ``device`` too -- the trace is symbolic (no real gather; block_ids
            # are all scratch and positions all sentinel), so co-location on the
            # real device is neither possible nor needed. On the native
            # (non-meta) path reuse the real buffer and pin the index tensors to
            # its device so the gather stays co-located.
            if torch.device(device).type == "meta":
                encoder_buffer = torch.empty(
                    cache_buffer.shape, dtype=cache_buffer.dtype, device=device
                )
                index_device = device
            else:
                encoder_buffer = cache_buffer
                index_device = cache_buffer.device
            warmup_kwargs["encoder_embedding_buffer"] = encoder_buffer
            warmup_kwargs["encoder_block_ids"] = torch.full(
                (max_num_encoder_blocks,),
                self.encoder_cache.scratch_block_id,
                dtype=torch.int32,
                device=index_device,
            )
            warmup_kwargs["encoder_positions"] = torch.full(
                (max_num_encoder_blocks, self.encoder_cache.block_size),
                num_tokens,
                dtype=torch.int32,
                device=index_device,
            )
        elif self.enable_prompt_embeds:
            warmup_kwargs["inputs_embeds"] = torch.zeros(
                num_tokens,
                self.inputs_embeds_size,
                dtype=torch.bfloat16,
                device=device,
            )
            warmup_kwargs["is_token_ids"] = torch.ones(
                num_tokens,
                dtype=torch.bool,
                device=device,
            )
        return warmup_kwargs

    def extract_prefill_graphs(
        self,
        bucket_size: int,
        kv_segment_size: int,
        device: torch.device | None = None,
    ) -> None:
        """
        Extract the prefill HLO graphs with a specific bucket size and KV segment size.
        This uses the neuron graph capture backend to get all the HLO Graphs and throws
        CaptureComplete exception to dynamo which is suppressed here as it marks
        successful graph capture.

        Args:
            bucket_size: The number of active tokens to warm up (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket for warmup
            device: Override device for synthetic inputs. Defaults to
                ``self.device``.
        """
        if self.capture_backend_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        logger.info(
            "Capturing prefill graphs for: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )
        kwargs = self._build_prefill_synthetic_inputs(
            bucket_size, kv_segment_size, device=device
        )
        try:
            _ = self.capture_backend_model(**kwargs)
        except CaptureComplete:
            logger.debug(
                "Graph capture for prefill completed: bucket_size=%s, kv_segment_size=%s",
                bucket_size,
                kv_segment_size,
            )

        # === Draft model graph capture ===
        if self.drafter is not None:
            logger.info(
                "Capturing %s prefill graphs for bucket size: %d",
                self.speculative_config.method.upper(),
                bucket_size,
            )
            self.drafter.graph_extract(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=kwargs["attn_metadata"],
                device=device,
            )

    def _drain_previous_dummy_output(self, model_output: Any) -> None:
        """Read the previous dummy output back, then hold the current one.

        Call this AFTER submitting the current dummy. The read frees the previous NEFF's queue
        slot and blocks until it completes, so an idle rank's queue stays at one dummy and cannot
        reach "Execution Queue Full". Order matters: the current dummy goes in first, so a busy
        rank always has a collective partner. This replaces the pacing the per-step
        ``_get_dp_padding`` reduce gave, and it keeps the reference the in-flight NRT input
        slices need.

        Args:
            model_output: Output of the dummy batch of this step.
        """
        previous = self._previous_dummy_output
        # Keep the reference first. The readback below can raise, and the in-flight NRT
        # input slices need this reference until the next step supersedes it.
        self._previous_dummy_output = model_output
        if previous is not None:
            self._materialize_warmup_output(previous)

    def _materialize_warmup_output(self, model_output: Any) -> None:
        """Bring a warmup forward's output to CPU to consume the async future.

        With async scheduling, ``self.model(...)`` returns device tensor futures
        (NrtaFuture) rather than materialized values. The warmup path discards
        the output, so without an explicit readback the future is never consumed
        and the async request lingers in the NRT queue. Reading the output
        tensor(s) back to CPU forces the future to resolve, which dequeues the
        request. The output may be a bare tensor or a nested tuple (EAGLE3), so
        recurse to materialize every tensor it contains.
        """

        def _to_cpu(value: Any) -> None:
            if torch.is_tensor(value):
                value.cpu()
            elif isinstance(value, (tuple, list)):
                for item in value:
                    _to_cpu(item)

        _to_cpu(model_output)

    def warmup_prefill(self, bucket_size: int, kv_segment_size: int) -> None:
        """
        Warm up the model for prefill with a specific bucket size and KV segment size.

        Creates synthetic inputs and calls self.model directly to trigger
        compilation without modifying any internal state. Also warms up the
        draft model's initial pass if speculative decoding is enabled.

        Args:
            bucket_size: The number of active tokens to warm up (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket for warmup
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.kv_cache_config is None:
            raise RuntimeError(
                "KV cache not initialized. Call initialize_kv_cache() first."
            )
        logger.info(
            "Warming up prefill: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )
        warmup_kwargs = self._build_prefill_synthetic_inputs(
            bucket_size, kv_segment_size
        )
        compile_start = time.perf_counter()
        if self._tensor_replacer is not None:
            set_active_context(
                self._tensor_replacer.warmup_context(bucket_size, self.device)
            )
        model_output = self.model(**warmup_kwargs)
        if self.use_async_scheduling:
            self._materialize_warmup_output(model_output)
        if self._tensor_replacer is not None:
            set_active_context(None)
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"prefill_s{bucket_size}"
        if kv_segment_size > 0:
            bucket_name += f"_kv{kv_segment_size}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)
        logger.debug(
            "Prefill warmup completed: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )

        # === Draft model warmup ===
        if self.drafter is not None:
            logger.info(
                "Warming up %s for bucket size: %d",
                self.speculative_config.method.upper(),
                bucket_size,
            )
            self.drafter.warmup(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=warmup_kwargs["attn_metadata"],
            )

    def _create_warmup_spec_decode_metadata(
        self,
        batch_size: int,
        num_spec_tokens: int,
    ) -> SpecDecodeMetadata:
        """
        Create synthetic spec_decode_metadata for warmup.

        Creates metadata with consistent tensor shapes matching inference,
        so torch.dynamo caches the correct compiled graph.

        Args:
            batch_size: Number of requests
            num_spec_tokens: Number of speculative tokens per request

        Returns:
            SpecDecodeMetadata with synthetic data
        """
        # Each request has num_spec_tokens draft tokens
        num_draft_tokens = [num_spec_tokens] * batch_size
        total_draft_tokens = batch_size * num_spec_tokens

        # Dummy draft token ids (real shape) so the compiled graph's
        # rejection_sampler gather sees the correct shape during warmup.
        draft_token_ids = torch.ones(total_draft_tokens, dtype=torch.int32)

        # Cumulative sums
        cu_num_draft = np.cumsum(num_draft_tokens, dtype=np.int64)
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft.copy())

        num_sampled = [num_spec_tokens + 1] * batch_size
        cu_num_sampled = np.cumsum(num_sampled, dtype=np.int64)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled.copy())

        # Build indices matching _build_spec_decode_metadata logic
        # target_logits_indices: indices for draft token verification
        target_indices = []
        for i in range(batch_size):
            base = i * (num_spec_tokens + 1)
            for j in range(num_spec_tokens):
                target_indices.append(base + j)
        target_logits_indices = torch.tensor(target_indices, dtype=torch.int64)

        # bonus_logits_indices: last position of each request
        bonus_logits_indices = torch.tensor(
            [cu_num_sampled[i] - 1 for i in range(batch_size)], dtype=torch.int64
        )

        # logits_indices: all sampled positions (total = batch_size * (num_spec_tokens + 1))
        logits_indices = torch.arange(
            batch_size * (num_spec_tokens + 1), dtype=torch.int64
        )

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _build_decode_synthetic_inputs(
        self,
        batch_size: int,
        context_len: int = 256,
        spec_decode_enabled=False,
        decode_token_threshold: int | None = None,
        ctx_bucket: int | None = None,
        device: torch.device | None = None,
        compiled_graph_input: bool = False,
    ) -> dict:
        """Build synthetic inputs for decode warmup.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length instead of max_model_len (enables per-seq-bucket NEFFs).
            device: Override device for synthetic tensors. Defaults to
                ``self.device``.

        Returns:
            dict: Keyword arguments for self.model() forward call
        """
        if device is None:
            device = self.device
        # With spec decode: target model verifies batch_size * (1 + num_spec_tokens) tokens
        # Without spec decode: target model decodes batch_size tokens (1 per request)
        if spec_decode_enabled:
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            num_tokens = batch_size * (1 + num_spec_tokens)
            num_reqs = batch_size

            # Create synthetic spec_decode_metadata to match inference kwargs
            spec_decode_metadata = self._create_warmup_spec_decode_metadata(
                batch_size=batch_size,
                num_spec_tokens=num_spec_tokens,
            )
            # Move spec_decode_metadata tensors to device (matching execute_model path)
            spec_decode_metadata.cu_num_draft_tokens = (
                spec_decode_metadata.cu_num_draft_tokens.to(device)
            )
            spec_decode_metadata.cu_num_sampled_tokens = (
                spec_decode_metadata.cu_num_sampled_tokens.to(device)
            )
            spec_decode_metadata.logits_indices = (
                spec_decode_metadata.logits_indices.to(device)
            )
            spec_decode_metadata.target_logits_indices = (
                spec_decode_metadata.target_logits_indices.to(device)
            )
            spec_decode_metadata.bonus_logits_indices = (
                spec_decode_metadata.bonus_logits_indices.to(device)
            )
            spec_decode_metadata.draft_token_ids = (
                spec_decode_metadata.draft_token_ids.to(device)
            )
        else:
            num_reqs = batch_size
            num_tokens = batch_size  # 1 token per request
            spec_decode_metadata = None

        # Create synthetic inputs
        input_ids = torch.ones(num_tokens, dtype=torch.int32).to(device)
        positions = torch.full((num_tokens,), context_len - 1, dtype=torch.long)
        if self.uses_mrope:
            rotary_position_ids = (
                positions.unsqueeze(0).expand(3, -1).contiguous().to(device)
            )
        else:
            rotary_position_ids = None
        positions = positions.to(device)
        logits_indices = torch.arange(num_tokens, dtype=torch.long).to(device)

        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            decode_token_threshold=decode_token_threshold,
            ctx_bucket=ctx_bucket,
            device=device,
        )

        # Create dummy sampling params for warmup
        dummy_sampling_params = torch.tensor(
            [[0, 1.0, 1.0]] * batch_size,  # [top_k, top_p, temperature] per request
            dtype=torch.float32,
            device=device,
        )

        # Replicate dummy sampling params for spec-decode verify step
        # (no-op when speculative decoding is not active).
        dummy_sampling_params = self._maybe_replicate_for_spec_decode(
            dummy_sampling_params, spec_decode_metadata
        )

        mask_device = device
        if getattr(self, "on_device_sampling", False) and not compiled_graph_input:
            mask_device = torch.device("cpu")
        dummy_logit_mask = self._build_noop_logit_mask(
            num_tokens, mask_device, compiled_graph_input=compiled_graph_input
        )

        rank_tensor = self.rank_tensor.to(device)

        decode_warmup_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=dummy_sampling_params,
            spec_decode_metadata=spec_decode_metadata,
            rank=rank_tensor,
            logit_mask=dummy_logit_mask,
        )
        # Always inject prev-step kwargs when the model is decorated. The
        # decorator's prologue calls
        # ``correct_spec_decode_positions_and_slot_mapping`` unconditionally
        # so torch.compile traces a single graph per shape; zero dummies
        # make the correction a no-op (valid_count == 1 + num_spec).
        if self._model_is_async_spec_decoded():
            decode_warmup_kwargs.update(
                self._build_async_spec_kwargs(
                    spec_decode_metadata=spec_decode_metadata,
                    num_total_tokens=num_tokens,
                    num_reqs_padded=batch_size,
                    device=device,
                )
            )
        if rotary_position_ids is not None:
            decode_warmup_kwargs["rotary_position_ids"] = rotary_position_ids
        # For multimodal models, decode NEFF does not include the merge path —
        # encoder embeddings are already in KV cache from prefill. Only pass
        # inputs_embeds during decode for non-multimodal prompt_embeds use cases.
        if self.enable_prompt_embeds and self.encoder_neuron_config is None:
            decode_warmup_kwargs["inputs_embeds"] = torch.zeros(
                num_tokens,
                self.inputs_embeds_size,
                dtype=torch.bfloat16,
                device=device,
            )
            decode_warmup_kwargs["is_token_ids"] = torch.ones(
                num_tokens,
                dtype=torch.bool,
                device=device,
            )

        if self._use_async_reorder() and self._is_decode_shaped(attn_metadata):
            # Warm the reorder NEFF here. A real decode step, an idle rank, and a dummy
            # batch then dispatch the same graph. The no-op reorder traces
            # index_select and where without a change of the values.
            decode_warmup_kwargs.update(
                self._noop_async_reorder_kwargs(
                    input_ids.shape[0], input_ids.dtype, device
                )
            )

        return decode_warmup_kwargs

    def _build_jf_dummy_inputs(
        self,
        batch_size: int,
        ctx_bucket: int | None = None,
        context_len: int = 256,
    ) -> dict:
        """Build JF-NEFF synthetic inputs for an idle DP rank's dummy batch.
        Mirrors JF warmup. Used by ``execute_dummy_batch`` when the group runs the
        JF NEFF this step, so an idle rank dispatches the JF NEFF
        — running the plain decode NEFF instead would mismatch the
        cross-DP collectives.
        """
        device = self.device
        jf_max = self.neuron_config.max_jump_forward_tokens
        tokens_per_req = 1 + jf_max
        jf_num_tokens = batch_size * tokens_per_req

        input_ids = torch.ones(jf_num_tokens, dtype=torch.int32).to(device)
        positions = torch.full((jf_num_tokens,), context_len - 1, dtype=torch.long).to(
            device
        )
        # n_jf=0 for every request: sample at column 0 of each request's span.
        logits_indices = torch.tensor(
            [i * tokens_per_req for i in range(batch_size)],
            dtype=torch.long,
        ).to(device)
        sampling_params = torch.tensor(
            [[0, 1.0, 1.0]] * batch_size,
            dtype=torch.float32,
            device=device,
        )
        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=jf_num_tokens,
            num_reqs=batch_size,
            decode_token_threshold=tokens_per_req,
            ctx_bucket=ctx_bucket,
            device=device,
        )
        # Match the JF warmup mask shape (one row per sampling position).
        logit_mask = self._build_noop_logit_mask(
            batch_size, device, compiled_graph_input=True
        )
        rank_tensor = self.rank_tensor.to(device)

        jf_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=sampling_params,
            spec_decode_metadata=None,
            rank=rank_tensor,
            logit_mask=logit_mask,
        )
        if self._use_async_reorder() and self._is_decode_shaped(attn_metadata):
            # The JF NEFF also takes the reorder inputs. A no-op reorder over the JF
            # input_ids passes them through.
            jf_kwargs.update(
                self._noop_async_reorder_kwargs(
                    input_ids.shape[0], input_ids.dtype, device
                )
            )
        if self.enable_prompt_embeds and not self.supports_mm_inputs:
            jf_kwargs["inputs_embeds"] = torch.zeros(
                jf_num_tokens,
                self.inputs_embeds_size,
                dtype=torch.bfloat16,
                device=device,
            )
            jf_kwargs["is_token_ids"] = torch.ones(
                jf_num_tokens,
                dtype=torch.bool,
                device=device,
            )
        return jf_kwargs

    def extract_decode_graphs(
        self,
        batch_size: int,
        context_len: int = 256,
        ctx_bucket: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Extract the decode HLO graphs with a specific bucket size and KV segment size.
        This uses the neuron graph capture backend to get all the HLO Graphs and throws
        CaptureComplete exception to dynamo which is suppressed here as it marks
        successful graph capture.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing (default 256)
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length for per-seq-bucket NEFF compilation.
            device: Override device for synthetic inputs. Defaults to
                ``self.device``.
        """
        if self.capture_backend_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        logger.info("Capturing decode graphs for batch size: %d", batch_size)
        spec_decode_enabled = (
            self.drafter is not None and self.speculative_config is not None
        )
        kwargs = self._build_decode_synthetic_inputs(
            batch_size,
            context_len,
            spec_decode_enabled=spec_decode_enabled,
            ctx_bucket=ctx_bucket,
            device=device,
            compiled_graph_input=True,
        )
        # When spec is disabled this first capture is the plain single-token
        # decode graph, so route it through the decode compile variant (the
        # per-phase optlevel override). This site never captures the
        # jump-forward graph. So no JF guard is needed here. When spec is
        # enabled it is the draft-length target graph and must use the shared
        # capture backend. The plain decode graph is captured separately below.
        try:
            if not spec_decode_enabled:
                with self._plain_decode_capture() as plain_decode_capture:
                    _ = plain_decode_capture(**kwargs)
            else:
                # TODO(compiler-args): this is the spec-verify (draft-length
                # target) graph. To tune its arguments, wrap this call with
                # scoped_compiler_args(self.resolve_graph_compiler_args(
                # "spec_verify")) and add a branch in the model
                # compiler_args_for.
                _ = self.capture_backend_model(**kwargs)
        except CaptureComplete:
            logger.debug(
                "Graph capture for decode completed: batch size=%s", batch_size
            )

        if spec_decode_enabled:
            # === Target model decode WITHOUT spec decode ===
            logger.debug(
                "Target model decode graph capture WITH spec decode completed: batch size=%s",
                batch_size,
            )
            kwargs = self._build_decode_synthetic_inputs(
                batch_size,
                context_len,
                spec_decode_enabled=False,
                decode_token_threshold=1,
                ctx_bucket=ctx_bucket,
                device=device,
                compiled_graph_input=True,
            )
            # Plain single-token target decode graph (spec off) -> decode variant.
            try:
                with self._plain_decode_capture() as plain_decode_capture:
                    _ = plain_decode_capture(**kwargs)
            except CaptureComplete:
                logger.debug(
                    "Graph capture for target model decode completed: batch size=%s",
                    batch_size,
                )
            logger.debug(
                "Target model decode graph capture WITHOUT spec decode completed: batch size=%s",
                batch_size,
            )

            # === Draft model decode graph capture ===
            logger.info(
                "Capturing %s decode graphs for batch size: %d",
                self.speculative_config.method.upper(),
                batch_size,
            )
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_num_tokens = batch_size * (1 + num_spec_tokens)

            draft_attn_metadata = self._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
                device=device,
            )

            self.drafter.graph_extract(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
                device=device,
            )

    def warmup_decode(
        self,
        batch_size: int,
        context_len: int = 256,
        ctx_bucket: int | None = None,
    ) -> None:
        """
        Warm up the model for decode with a specific batch size.

        Creates synthetic inputs for batch_size concurrent decode requests,
        each generating 1 token, and calls self.model directly to trigger
        compilation without modifying any internal state. Also warms up the
        draft model's recurrent pass if speculative decoding is enabled.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing (default 256)
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length for per-seq-bucket NEFF compilation.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.kv_cache_config is None:
            raise RuntimeError(
                "KV cache not initialized. Call initialize_kv_cache() first."
            )
        logger.info("Warming up decode for batch size: %d", batch_size)
        spec_decode_enabled = (
            self.drafter is not None and self.speculative_config is not None
        )
        kwargs = self._build_decode_synthetic_inputs(
            batch_size,
            context_len,
            spec_decode_enabled=spec_decode_enabled,
            ctx_bucket=ctx_bucket,
            compiled_graph_input=True,
        )
        compile_start = time.perf_counter()
        if self._tensor_replacer is not None:
            set_active_context(
                self._tensor_replacer.warmup_context(
                    kwargs["input_ids"].shape[0], self.device
                )
            )
        # This site builds a single-token graph when spec is off. It never
        # builds the jump-forward graph. The JF NEFF has a separate warmup block
        # below that always uses self.model. So no JF guard is needed here. This
        # matches the execute-time route (max_query_len == 1). With spec on this
        # is the draft-length graph, so use the shared self.model.
        if not spec_decode_enabled:
            with self._plain_decode_forward() as plain_decode_forward:
                model_output = plain_decode_forward(**kwargs)
        else:
            # TODO(compiler-args): this is the spec-verify (draft-length target)
            # graph. To tune its arguments, wrap this call with
            # scoped_compiler_args(self.resolve_graph_compiler_args(
            # "spec_verify")) and add a branch in the model compiler_args_for.
            model_output = self.model(**kwargs)
        if self.use_async_scheduling:
            self._materialize_warmup_output(model_output)
        if self._tensor_replacer is not None:
            set_active_context(None)
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"decode_b{batch_size}"
        if kwargs["spec_decode_metadata"] is not None:
            bucket_name += f"_s{1 + self.speculative_config.num_speculative_tokens}"
        # Include ctx_bucket in the label so per-pair compile time
        # is observable in the COMPILATION_TIME metric.
        if ctx_bucket is not None:
            bucket_name += f"_ctx{ctx_bucket}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)
        logger.debug(
            "Target model decode warmup completed: %d tokens",
            kwargs["input_ids"].shape[0],
        )

        if spec_decode_enabled:
            # === Target model decode warmup WITHOUT spec decode ===
            # When spec decode is enabled, also compile the non-spec path.
            # This NEFF is used when:
            # 1. Propose is skipped near max_model_len (max_position > limit)
            # 2. DI (disaggregated inference): prefill aux_hidden_states aren't
            #    transferred to decode, so the first decode re-runs without spec.
            kwargs = self._build_decode_synthetic_inputs(
                batch_size,
                context_len,
                spec_decode_enabled=False,
                decode_token_threshold=1,
                ctx_bucket=ctx_bucket,
                compiled_graph_input=True,
            )
            compile_start = time.perf_counter()
            if self._tensor_replacer is not None:
                set_active_context(
                    self._tensor_replacer.warmup_context(
                        kwargs["input_ids"].shape[0], self.device
                    )
                )
            # Plain single-token target decode graph (spec off) -> decode variant.
            with self._plain_decode_forward() as plain_decode_forward:
                model_output = plain_decode_forward(**kwargs)
            if self.use_async_scheduling:
                self._materialize_warmup_output(model_output)
            if self._tensor_replacer is not None:
                set_active_context(None)
            compile_elapsed = time.perf_counter() - compile_start
            no_spec_bucket_name = f"decode_b{batch_size}_s1"
            if ctx_bucket is not None:
                no_spec_bucket_name += f"_ctx{ctx_bucket}"
            COMPILATION_TIME.labels(
                model_name=self.vllm_config.model_config.model,
                bucket_name=no_spec_bucket_name,
            ).set(compile_elapsed)
            logger.debug(
                "Target model decode warmup WITHOUT spec decode completed: %d tokens",
                kwargs["input_ids"].shape[0],
            )

            # === Draft model warmup ===
            logger.info(
                "Warming up %s for batch size: %d",
                self.speculative_config.method.upper(),
                batch_size,
            )
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_num_tokens = batch_size * (1 + num_spec_tokens)

            draft_attn_metadata = self._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
            )

            self.drafter.warmup(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
            )

        # === Jump-forward decode NEFF warmup ===
        # Compile the JF NEFF: (1 + jf_max) tokens per request. Includes
        # decode_token + JF tokens. Handles both JF and non-JF requests.
        jf_max = self.neuron_config.max_jump_forward_tokens
        if jf_max > 0:
            tokens_per_req = 1 + jf_max
            jf_num_tokens = batch_size * tokens_per_req
            device = self.device
            logger.info(
                "Warming up JF decode NEFF: batch=%d, tokens_per_req=%d, "
                "total_tokens=%d",
                batch_size,
                tokens_per_req,
                jf_num_tokens,
            )
            # Build kwargs that match what _prepare_jf_model_input + execute_model
            # Matrix/tensor: Sizes:
            # - input_ids: [batch_size * jf_max]
            # - positions: [batch_size * jf_max]
            # - sampling_positions: [batch_size] (last pos per request)
            # - sampling_params: [batch_size, 3]
            # - logit_mask: [batch_size, vocab_shard] (one row per sample)
            # - spec_decode_metadata: None
            # - attn_metadata: decode-style with decode_token_threshold=tokens_per_req
            input_ids = torch.ones(jf_num_tokens, dtype=torch.int32).to(device)
            positions = torch.full(
                (jf_num_tokens,), context_len - 1, dtype=torch.long
            ).to(device)
            logits_indices = torch.tensor(
                [i * tokens_per_req for i in range(batch_size)],
                dtype=torch.long,
            ).to(device)
            sampling_params = torch.tensor(
                [[0, 1.0, 1.0]] * batch_size,
                dtype=torch.float32,
                device=device,
            )
            attn_metadata = self._build_warmup_attention_metadata(
                num_tokens=jf_num_tokens,
                num_reqs=batch_size,
                decode_token_threshold=tokens_per_req,
                ctx_bucket=ctx_bucket,
                device=device,
            )
            logit_mask = self._build_noop_logit_mask(
                batch_size, device, compiled_graph_input=True
            )
            rank_tensor = self.rank_tensor.to(device)

            jf_kwargs = dict(
                input_ids=input_ids,
                positions=positions,
                attn_metadata=attn_metadata,
                sampling_positions=logits_indices,
                sampling_params=sampling_params,
                spec_decode_metadata=None,
                rank=rank_tensor,
                logit_mask=logit_mask,
            )

            # Include inputs_embeds/is_token_ids when prompt_embeds is enabled,
            # matching what _execute_model_forward injects at runtime.
            if self.enable_prompt_embeds and not self.supports_mm_inputs:
                jf_kwargs["inputs_embeds"] = torch.zeros(
                    jf_num_tokens,
                    self.inputs_embeds_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
                jf_kwargs["is_token_ids"] = torch.ones(
                    jf_num_tokens,
                    dtype=torch.bool,
                    device=device,
                )

            if self._use_async_reorder() and self._is_decode_shaped(attn_metadata):
                # Warm the JF graph with the reorder inputs. The warmed signature then
                # matches the signature that the runtime JF step dispatches.
                jf_kwargs.update(
                    self._noop_async_reorder_kwargs(
                        input_ids.shape[0], input_ids.dtype, device
                    )
                )

            compile_start = time.perf_counter()
            if self._tensor_replacer is not None:
                set_active_context(
                    self._tensor_replacer.warmup_context(jf_num_tokens, self.device)
                )
            # TODO(compiler-args): this is the jump-forward graph. To tune its
            # arguments, wrap this call and its execute-time counterpart with
            # scoped_compiler_args(self.resolve_graph_compiler_args("jf")) and
            # add a branch in the model compiler_args_for.
            model_output = self.model(**jf_kwargs)
            if self.use_async_scheduling:
                self._materialize_warmup_output(model_output)
            if self._tensor_replacer is not None:
                set_active_context(None)
            compile_elapsed = time.perf_counter() - compile_start

            jf_bucket_name = f"jf_decode_b{batch_size}_s{tokens_per_req}"
            if ctx_bucket is not None:
                jf_bucket_name += f"_ctx{ctx_bucket}"
            COMPILATION_TIME.labels(
                model_name=self.vllm_config.model_config.model,
                bucket_name=jf_bucket_name,
            ).set(compile_elapsed)
            logger.info(
                "JF decode NEFF warmup completed: %s (%.1fs)",
                jf_bucket_name,
                compile_elapsed,
            )

    def parallel_compile(self):
        """Deterministic parallel compile — requires a barrier before and after.

        Assumes all ranks in the TP group have finished graph extraction and
        been synchronized via a TP-scoped barrier, so the cache directory
        contains the complete set of HLOs for this model replica. Only
        TP-rank 0 performs compilation.
        """
        logger.info("Initiating parallel compile for captured graphs")
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        rank = tp_group.rank_in_group if tp_group.world_size >= 1 else None
        world_size = tp_group.world_size if tp_group.world_size >= 1 else None

        from libtorch_neuronx_lite.compile.parallel_compile import (
            parallel_compile as neuron_parallel_compile,
        )

        from libtorch_neuronx_lite import envs as libtorch_envs

        neuron_parallel_compile(
            options=self.compile_options,
            rank=rank,
            world_size=world_size,
            remote_cache_dir=libtorch_envs.NEURON_LIBTORCH_REMOTE_CACHE,
        )

    # ------------------------------------------------------------------------
    # MODEL EXECUTION
    # ------------------------------------------------------------------------
    def _materialize_pending_async_output(self, reason: str) -> bool:
        """Materialize the previous async step if it has not been consumed.

        Used by the async-fallback path: when this step can't reuse the
        previous step's device future, we need the previous step's outputs
        written back to CPU request state before rebuilding inputs, so the
        rebuild picks up fresh token values.

        Args:
            reason: Debug context explaining why async flow is being broken.

        Returns:
            True if a pending async output was materialized, False otherwise.
        """
        if not self.use_async_scheduling:
            return False

        async_output = self.async_execution_buffer.get("async_output")
        if async_output is None:
            return False

        # Don't materialize an intermediate segmented-prefill chunk on the
        # worker submit thread. Its readback would sit between consecutive
        # segment submits and stall the device (the ~3ms inter-segment gap).
        # It is unnecessary: the next segment's inputs are the next slice of
        # the prompt (built from token_ids_cpu / scheduler-advanced
        # num_computed_tokens), independent of this chunk's discarded sampled
        # tokens. The segment NEFF is instead drained off the submit path by
        # the async-output thread's get_output() (see is_all_partial_prefill).
        if async_output.is_all_partial_prefill():
            logger.debug(
                "Skipping worker-thread materialize for all-partial prefill "
                "segment: %s",
                reason,
            )
            return False

        # Pure kv_producer: the sampled token is never re-read (the request
        # finishes and hands off after this prefill), so the submit-thread
        # materialize is pure latency -- defer it to the async-output thread,
        # which still delivers the token.
        if self._pure_producer:
            logger.debug(
                "Deferring pure-producer materialize to the async-output thread "
                "(token still delivered): %s",
                reason,
            )
            return False

        logger.debug("Materializing pending async output: %s", reason)
        async_output.get_output()
        self._async_output_materialized_this_step = True
        return True

    def _apply_async_fallback_token_fixup(
        self,
        input_ids: torch.Tensor,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor:
        """Take each decode row's token from the previous step's sampler output.

        A CPU-built decode ``input_ids`` gathers ``token_ids_cpu`` at
        ``num_computed_tokens_cpu`` (see ``_prepare_model_input_impl``). Those
        two cursors advance independently: the read cursor comes from the
        scheduler and advances every step, while the write cursor
        (``num_tokens_no_spec``) only advances where a sampled token is
        materialized back to host state. Under async scheduling upstream
        materializes on ``output_rank`` alone
        (``multiproc_executor.py``: ``if output_rank is None or self.rank ==
        output_rank``), so every other rank's write cursor lags by the number
        of steps since its last fallback and the gather reads a slot that rank
        never wrote — silently feeding a stale token into decode.

        The tokens themselves are not missing: this step consumes exactly one
        token per decode row, and it is the one the caller just materialized.
        So overwrite those rows from the materialized output, keyed by request
        id, and leave every other row on the host gather (a row absent from the
        previous step has its history written densely by ``add_request`` on all
        ranks).

        Must be called only right after ``_materialize_pending_async_output``
        returned True, so ``sampled_token_ids`` is a host list rather than a
        device future.

        Args:
            input_ids: CPU-built input IDs for this step, one token per row.
            scheduler_output: This step's scheduler output, for per-request
                scheduled token counts.

        Returns:
            ``input_ids``, with decode rows overwritten in place where a
            previous-step token was available.

        Example:
            >>> # rank whose write cursor lags: the gather returned a stale 5,
            >>> # while the previous step actually sampled 42.
            >>> ids = runner._apply_async_fallback_token_fixup(
            ...     torch.tensor([5], dtype=torch.int32), scheduler_output
            ... )  # doctest: +SKIP
            >>> ids.tolist()  # doctest: +SKIP
            [42]
        """
        # Enforce the docstring's contract rather than trusting it. Without this,
        # a caller that has not materialized this step gets a silent no-op below
        # and proceeds on stale tokens.
        assert self._async_output_materialized_this_step, (
            "_apply_async_fallback_token_fixup requires "
            "_materialize_pending_async_output to have succeeded this step"
        )

        async_output = self.async_execution_buffer.get("async_output")
        if async_output is None:
            return input_ids

        model_runner_output = getattr(async_output, "model_runner_output", None)
        prev_sampled = getattr(model_runner_output, "sampled_token_ids", None)
        # A device future means the materialize declined (e.g. partial
        # prefill); there is nothing to read back without forcing it.
        if not isinstance(prev_sampled, list):
            return input_ids

        prev_req_ids = getattr(model_runner_output, "req_ids", None) or []
        prev_token_by_req = {
            req_id: prev_sampled[i][-1]
            for i, req_id in enumerate(prev_req_ids)
            if i < len(prev_sampled) and prev_sampled[i]
        }
        if not prev_token_by_req:
            return input_ids

        # Row r of input_ids is request input_batch.req_ids[r] only while every
        # request occupies exactly one row. That layout is built from
        # ``num_scheduled_tokens_padded`` (see _create_padded_inputs), not the
        # unpadded count -- a 1-token prefill pads to a bucket while the unpadded
        # count stays 1 -- so validate the PADDED layout. Any padded>1 row
        # (prefill bucket, or spec decode's 1+num_spec) breaks the 1:1 mapping,
        # so bail on the whole batch rather than misattribute a token.
        req_ids = self.input_batch.req_ids
        num_reqs = self.input_batch.num_reqs
        if input_ids.shape[0] < num_reqs or any(
            scheduler_output.num_scheduled_tokens_padded.get(req_id, 0) != 1
            for req_id in req_ids[:num_reqs]
        ):
            logger.debug(
                "Async fallback token fixup skipped: not a 1-token-per-row step."
            )
            return input_ids

        fixed = 0
        decode_rows = 0
        for row, req_id in enumerate(req_ids[:num_reqs]):
            # A one-token row can be a non-decode last-token recompute
            # co-scheduled with real decodes (the bench connector admits a KV
            # consumer at prompt_len-1 as a decode-shaped row). Gate per row so
            # its prompt token is preserved, rather than let _is_decode() veto --
            # and strand on the stale gather -- the genuine decode rows with it.
            if not self._is_decode_row(row):
                continue
            decode_rows += 1
            token = prev_token_by_req.get(req_id)
            if token is None:
                continue
            if int(input_ids[row]) != int(token):
                fixed += 1
            input_ids[row] = token

        if fixed:
            logger.debug(
                "Async fallback token fixup corrected %d/%d decode row(s) whose "
                "token_ids_cpu slot was stale.",
                fixed,
                decode_rows,
            )
            # These values bypass the _prepare_model_input_impl validation, and
            # the non-spec materialize does not range-check what the sampler
            # returned, so re-assert the invariant rather than let an
            # out-of-vocab id reach the embedding as a silent wrong row.
            self._validate_token_ids(input_ids)
        return input_ids

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        kv_caches: list[torch.Tensor] | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """
        Execute model forward pass and store state for sampling.

        For async execution, this returns None and stores state in execute_model_state.
        For sync execution, this calls sample_tokens() internally and returns output.

        Args:
            scheduler_output: Output from vLLM scheduler

        Returns:
            - None for async execution (call sample_tokens() separately)
            - ModelRunnerOutput for sync execution
            - AsyncModelRunnerOutput if use_async_scheduling is on
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.use_async_scheduling:
            self._async_output_materialized_this_step = False

        logger.debug("=== Starting NeuronModelRunner.execute_model ===")
        logger.debug("scheduler_output=%s", scheduler_output)
        logger.debug(
            "Processing scheduler output: %s new, %s cached requests",
            len(getattr(scheduler_output, "scheduled_new_reqs", [])),
            len(getattr(scheduler_output.scheduled_cached_reqs, "req_ids", [])),
        )

        # Materialize pending async output BEFORE _update_states when the
        # batch composition is about to change.
        #
        # With async scheduling, the previous step's sampled tokens may still
        # be on-device futures (not yet written to token_ids_cpu or
        # req_state.output_token_ids). When _update_states calls
        # input_batch.add_request() for resumed/new requests, it copies from
        # req_state.output_token_ids into token_ids_cpu. If we haven't
        # materialized, output_token_ids is missing the latest sampled token
        # and the copied data would be stale.
        #
        # We detect batch composition changes by comparing the current step's
        # scheduled request IDs against the previous step's. This is simpler
        # and more robust than checking individual signals (finished_req_ids,
        # scheduled_new_reqs, etc.) because it catches ALL cases including
        # the unscheduled gap (max_tokens pipelining) in a single comparison.
        if self.use_async_scheduling:
            # Skip on first iteration — no pending async output to materialize.
            if not self.async_execution_buffer:
                self._batch_composition_changed = False
            else:
                curr_req_ids = frozenset(scheduler_output.num_scheduled_tokens.keys())
                prev_req_ids_ordered = self.async_execution_buffer.get(
                    "prev_req_ids_ordered"
                )
                prev_req_ids = (
                    frozenset(prev_req_ids_ordered)
                    if prev_req_ids_ordered is not None
                    else None
                )
                composition_changed = (
                    prev_req_ids is None or curr_req_ids != prev_req_ids
                )
                # Detect spec→non-spec transition: previous step had spec
                # decode (prev future is [bs, num_spec+1]), current step is
                # non-spec (no scheduled_spec_decode_tokens). In async mode
                # the transition creates a CPU position mismatch because
                # scheduler's num_computed_tokens_cpu reflects the previous
                # spec step's optimistic advance but not its rejections —
                # reading token_ids_cpu at that optimistic position gives
                # garbage. Force a sync at this boundary by materializing
                # and reconciling num_computed_tokens_cpu.
                prev_future = self.async_execution_buffer.get(
                    "futures_sampled_token_ids"
                )
                prev_was_spec = (
                    isinstance(prev_future, torch.Tensor)
                    and prev_future.ndim == 2
                    and prev_future.shape[1] > 1
                )
                curr_is_nonspec = not bool(
                    scheduler_output.scheduled_spec_decode_tokens
                )
                spec_to_nonspec_transition = prev_was_spec and curr_is_nonspec

                if composition_changed:
                    logger.debug(
                        "Breaking async flow: prev_req_ids=%s, curr_req_ids=%s.",
                        sorted(prev_req_ids) if prev_req_ids else None,
                        sorted(curr_req_ids),
                    )
                    # A resumed request keeps the host materialize. ``_update_states``
                    # re-seeds its slot through ``add_request``, and skipping the
                    # materialize would let the async-output thread then apply the
                    # previous step's token on top of the re-seeded slot. Resumption is
                    # rare, so the cost is one step on the slower path.
                    has_resumed_request = bool(
                        scheduler_output.scheduled_cached_reqs.resumed_req_ids
                    )
                    if self._use_async_reorder() and not has_resumed_request:
                        # Skip the device-to-host materialize: the in-graph reorder
                        # assembles the input_ids on the device. Keep the flag False so the
                        # execute path builds the reorder inputs. The swap remaps the future
                        # by request id when the reorder is not usable, so a composition
                        # change stays correct. A spec-to-non-spec change cannot reach here,
                        # because the reorder needs a non-spec step.
                        self._batch_composition_changed = False
                        self._transition_bonus_tensor = None
                    else:
                        self._materialize_pending_async_output(
                            "batch composition changed"
                        )
                        self._batch_composition_changed = True
                        self._transition_bonus_tensor = None
                elif spec_to_nonspec_transition:
                    # At spec→non-spec transition, we must provide the correct
                    # input_ids for the non-spec step. The scheduler's
                    # num_computed_tokens is optimistic (includes the full
                    # num_spec draft length), so reading from token_ids_cpu
                    # at that offset returns garbage. Instead use the
                    # ``futures_last_accepted_token`` tensor produced by
                    # the previous step's rejection sampler — it's
                    # already shape [bs] with default stride and contains
                    # the LAST non-rejected token per request, which is
                    # exactly the input_ids the next decode step needs.
                    logger.debug("Breaking async flow: spec→non-spec transition.")
                    self._transition_bonus_tensor = self.async_execution_buffer.get(
                        "futures_last_accepted_token"
                    )
                    self._batch_composition_changed = True
                    # Materialize prev spec step's output so output_token_ids
                    # contains the real bonus token before _update_states
                    # trims based on the scheduler's authoritative count.
                    self._materialize_pending_async_output("spec→non-spec transition")
                else:
                    self._batch_composition_changed = False
                    self._transition_bonus_tensor = None

        self._drop_transition_spec_decode_if_cache_incomplete(scheduler_output)

        # Update cached states (includes sequence ID management).
        # _update_states is a verbatim copy of upstream, so the lock goes here. It keeps the
        # async-output thread out of the InputBatch for the whole update. See
        # _batch_mutation_lock.
        with record_function_or_nullcontext("neuron_model_runner: update_states"):
            with self._batch_mutation_lock():
                self._update_states(scheduler_output)

        # Register new requests for tensor replacement prompt matching.
        self._register_requests_for_replacement(scheduler_output)

        # Execute multimodal encoder for scheduled inputs (prefill only).
        # Gather happens later in _prepare_model_input_impl (also prefill-guarded)
        # because it depends on padding_map computed there.
        if self.encoder_neuron_config is not None and not self._is_decode():
            with record_function_or_nullcontext(
                "neuron_model_runner: execute_mm_encoder"
            ):
                self._execute_mm_encoder(scheduler_output)
            if self._encoder_cache_snapshot_enabled:
                # Snapshot all encoder entries used by the current request
                # (includes both fresh encodings and cache hits).
                self._snapshot_encoder_entries(scheduler_output)
            # EPD EC consumer (PD): the vision tower is absent, so _execute_mm_encoder
            # cannot produce remote items locally -- pull them from the VE over NIXL
            # into the local encoder cache HERE, before _prepare_model_input gathers
            # the vision blocks. No-op on the producer / default path.
            if self._is_ec_consumer():
                self._run_ec_consumer_lifecycle(scheduler_output)

        # Vision-encoder-only pool (mm_encoder_only): finish the request right
        # after the encoder. This pool builds no language model and no sampler,
        # so once the encoder has run (and cached the embedding above) there is
        # nothing left to do. We hand back an output with one placeholder token
        # per request; because the client requested max_tokens=1, vLLM's normal
        # "hit the token limit -> done" path finishes the request. We add no
        # custom completion logic. (Mirrors upstream gpu_model_runner.py.)
        #
        # Check the construction flag first so the default LM path is never
        # diverted; _is_ve_encode_step then applies the prefill + DP-safety gate.
        if self.mm_encoder_only and self._is_ve_encode_step():
            logger.debug(
                "EPD VE encode-finish: token-less output for %d req(s)",
                len(getattr(scheduler_output, "num_scheduled_tokens", {})),
            )
            out = make_empty_encoder_model_runner_output(scheduler_output)
            # EPD EC: when this VE is an EC producer, attach the per-item
            # embedding locator so the scheduler can emit it on the response
            # (consumed by the PD to pull the embedding over NIXL).
            if self._is_ec_producer():
                # Attach the EC locator now, on the submit thread, so it reflects
                # this step's cache state and cannot race the next step's cache
                # changes. Safe before the drain: _attach_ec_locator reads only
                # host-side block-allocation metadata, not device memory.
                self._attach_ec_locator(out, scheduler_output)
                if self.use_async_scheduling:
                    # Defer only the HBM drain off the submit thread. The submit
                    # thread starts the next request's CPU block-packing while
                    # this request's Neff runs. get_output() drains before the
                    # router acks, so the HBM-before-ack order holds.
                    return VeEncoderProducerAsyncOutput(out, self)
                # Synchronous path: drain the NRT execution queue before the
                # ready-ack goes to the router. Read the embedding buffer and the
                # tag tensor. This forces both writes to HBM before the consumer
                # can pull them. A stale tag would else cause a false
                # receipt-validation miss on the consumer.
                _ = self.encoder_cache.buffer[0, 0, 0].detach().to("cpu")
                _ = self.encoder_cache.mm_hash_tags[0].detach().to("cpu")
            return out

        # Check if there's work to do
        total_scheduled_tokens = getattr(
            scheduler_output, "total_num_scheduled_tokens", 0
        )
        logger.debug("Total scheduled tokens: %s", total_scheduled_tokens)

        if total_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                logger.debug("No scheduled tokens, returning empty output")
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # Prepare model inputs using vectorized operations
        with record_function_or_nullcontext("neuron_model_runner: prepare_input"):
            (
                input_ids,
                positions,
                logits_indices,
                attn_metadata,
                spec_decode_metadata,
                rotary_position_ids,
                mm_kwargs,
            ) = self._prepare_model_input(scheduler_output)

        # Async scheduling: replace CPU-built input_ids with a device tensor
        # assembled from the previous step's NEFF output futures, eliminating
        # the device→host→device roundtrip in steady state.
        if self.use_async_scheduling:
            # Build the in-graph reorder inputs for a decode-shape step, so the reorder
            # assembles the input_ids on the device.
            #
            # A False usable flag means no device future backs it, which happens on the
            # first step and after a decode-bucket or dtype change. The no-op inputs still
            # go to the forward so the graph stays uniform across DP ranks, but the swap,
            # the materialize and the fixup below must then all run.
            self._async_reorder_kwargs = None
            reorder_usable = False
            if self._use_async_reorder() and self._is_decode_shaped(attn_metadata):
                (
                    self._async_reorder_kwargs,
                    reorder_usable,
                ) = self._build_async_reorder_kwargs(input_ids)
            original_input_ids = input_ids
            if reorder_usable:
                # The reorder read the device future with no host copy, so this step is
                # asynchronous. Count it, to keep the counters equal to the swap path.
                self._async_steps += 1
            else:
                input_ids = self._maybe_swap_async_input_ids(input_ids)
            swap_declined = (not reorder_usable) and input_ids is original_input_ids
            if swap_declined and not (
                self.supports_mm_inputs and not self._is_decode()
            ):
                # The swap rejected the buffered future (shape/dtype mismatch
                # or no spec-decode match), so we need to use CPU-built
                # input_ids. But those may be stale: the previous async step's
                # sampled tokens may still be on-device futures, not yet
                # written back into token_ids_cpu / req_state.output_token_ids.
                # Materialize the pending async output, reconcile
                # num_computed_tokens_cpu with real accepted-token count
                # (the scheduler-provided value is optimistic), then rebuild
                # inputs from the refreshed request state.
                #
                # Skip on multimodal prefill: input_ids were just built fresh
                # from the scheduler output — no stale async state to reconcile.
                if self._materialize_pending_async_output(
                    "async future shape/dtype mismatch"
                ):
                    # Re-prep after materializing stale async output. The batch
                    # composition is unchanged from the first _prepare_model_input
                    # above (only token *values* were refreshed), so the cross-DP
                    # padding decision is identical — reuse it instead of issuing
                    # a SECOND _get_dp_padding all-reduce. A second host reduce
                    # here breaks the per-step 1:1 host-reduce/EP-forward ratio and
                    # desyncs the DP ranks' two collective streams, deadlocking
                    # cross-DP EP under DI (one rank stuck in the host reduce while
                    # peers are in the device EP collective).
                    (
                        input_ids,
                        positions,
                        logits_indices,
                        attn_metadata,
                        spec_decode_metadata,
                        rotary_position_ids,
                        mm_kwargs,
                    ) = self._prepare_model_input(
                        scheduler_output, reuse_dp_padding=True
                    )

                    # The re-prep above refreshes token *values* only where
                    # this rank's write cursor reached; off-rank cursors lag.
                    input_ids = self._apply_async_fallback_token_fixup(
                        input_ids, scheduler_output
                    )

            # Populate spec_decode_metadata.draft_token_ids from the previous
            # step's draft NEFF output. The draft NEFF emits a separate
            # contiguous ``drafts_only`` tensor of shape ``[bs, num_spec]``
            # so the runner can feed it as the rejection sampler's
            # ``draft_token_ids`` via ``.view(-1)`` -- without a non-contiguous
            # ``[:, 1:]`` slice that ``.contiguous()`` can't resolve on
            # Neuron device.
            #
            # Skip swap when batch composition changed: the future from the
            # previous step was sized for a different batch, so its flattened
            # shape doesn't match the current step's
            # ``spec_decode_metadata.draft_token_ids`` placeholder.
            input_ids = self._fill_transition_input_and_spec_metadata_from_cache(
                input_ids,
                spec_decode_metadata,
            )

            if spec_decode_metadata is not None and not self._batch_composition_changed:
                drafts_only_future = self.async_execution_buffer.get(
                    "futures_drafts_only"
                )
                if (
                    drafts_only_future is not None
                    and drafts_only_future.ndim == 2
                    and drafts_only_future.numel()
                    == spec_decode_metadata.draft_token_ids.shape[0]
                ):
                    draft_token_ids = _reinterpret_uint32_as_int32(
                        drafts_only_future, torch.int32
                    ).view(-1)
                    spec_decode_metadata.draft_token_ids = draft_token_ids

        # num_valid_tokens tracks the pre-DP-padding count so we can strip
        # padding from logits later.  DP padding is applied inside
        # _prepare_model_input (before attn_metadata is built) so that
        # block_table, slot_mapping, and input_ids all share the same shape.
        num_valid_tokens = len(self.input_batch.req_ids)

        logger.debug(
            "Executing model with %s real requests, %s tokens (after DP+bucket padding)",
            num_valid_tokens,
            input_ids.shape[0],
        )

        # Get grammar bitmask for structured outputs.
        # The application point differs based on sampling mode:
        #   - On-device sampling: mask passed INTO model.forward() for device-side masked_fill
        #   - CPU sampling: mask applied to logits AFTER they return to CPU, before _sample()
        # For JF steps: bitmask applies only to the bonus position (sampling_positions).
        # The grammar is already advanced past JF tokens, so the mask constrains the next valid token.
        is_jf_step = getattr(self, "_jf_active_tokens", None) is not None
        grammar_bitmask = self._get_grammar_bitmask(scheduler_output)
        cpu_grammar_bitmask = None
        self._raise_if_structured_outputs_disabled(grammar_bitmask)

        # logit_mask must match lm_head output shape.
        # After DP padding, logits_indices may have grown, so use its length.
        num_logit_rows = (
            input_ids.shape[0]
            if spec_decode_metadata is not None
            else len(logits_indices)
        )
        noop_mask_device = (
            torch.device("cpu") if self.on_device_sampling else self.device
        )
        all_true_mask = self._build_noop_logit_mask(num_logit_rows, noop_mask_device)

        # For JF decode, input_ids has batch_size * (1+jf_max_tokens) but logit_mask
        # should only have batch_size rows (one per sampling position). Use
        # num_logit_rows as the pad target so the mask matches the warmup shape.
        logit_mask_pad_target = num_logit_rows if is_jf_step else input_ids.shape[0]

        if self.on_device_sampling:
            # ON-DEVICE PATH: In mixed-mode servers, pass a real grammar mask
            # for SO requests or an all-True no-op mask for SO-off requests so
            # torch.compile traces a single tensor path. In SO-off-only perf
            # mode, all_true_mask is None and SO requests are rejected above.
            logit_mask = (
                grammar_bitmask if grammar_bitmask is not None else all_true_mask
            )

            # Pad logit_mask to match decode batch padding.
            if logit_mask is not None:
                logit_mask = self._pad_logit_mask_for_decode(
                    logit_mask, logit_mask_pad_target
                )

            # TODO: Structured output + async scheduling is supported in vLLM 0.16
            # on GPU. Integrate this for Neuron so the guard can be removed.
            # Guard: Structured outputs with async scheduling is not supported
            if grammar_bitmask is not None and self.use_async_scheduling:
                raise AssertionError(
                    "Structured outputs with async scheduling is not supported. "
                    "Disable async scheduling with: --no-async-scheduling"
                )
            if logit_mask is not None:
                logit_mask = self._prepare_ods_logit_mask(logit_mask)
        else:
            # CPU SAMPLING PATH: Grammar mask will be applied to logits on CPU
            # after the forward pass, before sampling.
            cpu_grammar_bitmask = grammar_bitmask
            logit_mask = all_true_mask

            # Pad logit_mask for CPU path too (same graph-stability reason as
            # the on-device path). In SO-off-only perf mode this remains None.
            if logit_mask is not None:
                logit_mask = self._pad_logit_mask_for_decode(
                    logit_mask, logit_mask_pad_target
                )

            # Guard: SO with async scheduling requires synchronous bitmask application
            if cpu_grammar_bitmask is not None and self.use_async_scheduling:
                raise AssertionError(
                    "Structured outputs with CPU sampling requires synchronous scheduling. "
                    "Grammar bitmask must be applied to logits before sampling. "
                    "Disable async scheduling with: --no-async-scheduling"
                )

        # Wrap with set_forward_context for DI support
        with (
            set_forward_context(
                None,
                self.vllm_config,
            ),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
            self._kv_cache_dump_context(),
        ):
            # Execute model with padded inputs.
            (
                model_output_tensor,
                aux_hidden_states,
                last_accepted_token,
            ) = self._execute_model_forward(
                input_ids,
                positions,
                logits_indices,
                attn_metadata,
                spec_decode_metadata,
                logit_mask=logit_mask,
                rotary_position_ids=rotary_position_ids,
                mm_kwargs=mm_kwargs,
            )

        # Store the last-accepted token from the rejection sampler (spec
        # decode only) so the next step's transition handler can use it
        # directly as input_ids without slicing or stride manipulation on
        # the [bs, num_spec+1] rejection output tensor.
        if self.use_async_scheduling and last_accepted_token is not None:
            self.async_execution_buffer["futures_last_accepted_token"] = (
                last_accepted_token
            )

        # CPU SAMPLING PATH: Apply grammar bitmask to logits on CPU.
        # At this point, model_output_tensor contains logits on CPU (padding removed).
        # masked_fill runs in eager PyTorch — no torch.compile graph sensitivity.
        if cpu_grammar_bitmask is not None:
            model_output_tensor = self._apply_grammar_bitmask_cpu(
                model_output_tensor, cpu_grammar_bitmask
            )

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        )

        self.kv_connector_output = kv_connector_output
        return None

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """
        Sample tokens from model output and generate final output.

        This is the second phase of the execute_model() -> sample_tokens() flow.
        It retrieves the state stored by execute_model() and performs sampling.

        Args:
            grammar_output: Grammar output for structured generation (unused for now)

        Returns:
            ModelRunnerOutput with sampled tokens
        """
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Minimal bookkeeping logic to retain request id information for
        # vLLM async scheduling.
        # Note _bookkeeping_sync is not introducing new data structures to InputBatch, but
        # rather keeping a snapshot of the states of some fields in InputBatch before they
        # are overwritten by vllm scheduler to schedule next step asynchronously. We use
        # the snapshots for preparing model input for the current step.
        # This is similar to `_bookkeeping_sync` in gpu_model_runner.
        (
            req_ids_output_copy,
            req_id_to_index_output_copy,
        ) = self._bookkeeping_sync()

        # Sample
        # run sampler with torch compile calls disabled, and ran eagerly
        # ex. https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/logprobs.py
        with (
            record_function_or_nullcontext("neuron_model_runner: sample"),
            torch.compiler.set_stance("force_eager"),
        ):
            if self.is_pooling_model:
                # For pooling models, model_output_tensor carries the flattened
                # [T, H] hidden states: run the pooler head and
                # return the embeddings, bypassing the sampler entirely.
                return self._pool(
                    model_output_tensor,
                    self._pooling_num_scheduled_tokens_per_seq,
                )
            sampler_output = self._sample(model_output_tensor, spec_decode_metadata)

        # Capture raw sampler output for the draft model's on-device token
        # extraction. The draft NEFF's ``extract_accepted_tokens`` consumes
        # this tensor to find the last-valid (bonus) token per request.
        # Shape: [bs, num_spec+1] when spec decode ran, else [bs] (prefill or
        # non-spec decode). None here triggers a CPU fallback in
        # ``_propose_draft_token_ids`` for the non-on-device-sampling path.
        raw_sampled_token_ids: torch.Tensor | None = None
        if self.on_device_sampling:
            sampled = sampler_output.sampled_token_ids
            if isinstance(sampled, torch.Tensor):
                raw_sampled_token_ids = sampled
            elif spec_decode_metadata is not None:
                # Sync + on-device sampling + spec decode: sampler returned
                # parsed list[list[int]], so use the pre-parse rejection
                # tensor directly.
                raw_sampled_token_ids = model_output_tensor

        # Debug: write raw logits to shared memory for test validation.
        # Written AFTER sampling so we can filter to accepted tokens for spec decode.
        if self._debug_logits_dir:
            logits_to_write = (
                self._on_device_logits
                if self._on_device_logits is not None
                else model_output_tensor
            )
            # Debug-only: per-request jump-forward width for cur step. JF
            # NEFF processed [decode_token, jf_0..jf_{n-1}] and the
            # single written logit corresponds to how we actually sampled the last
            # token. Normal decode steps have width 0. The JF-aware test
            # collector uses this to locate each bonus token in the
            # output stream (see create_online_jf_generate_fn).
            # Only computed when debug_logits_dir is set.
            jf_decode_tokens = getattr(scheduler_output, "_jf_decode_tokens", None)
            jf_width_by_req = (
                {rid: len(toks) for rid, toks in jf_decode_tokens.items()}
                if jf_decode_tokens
                else {}
            )
            # Skip rows for requests still in chunked/segmented prefill: their
            # logits aren't a real model output yet (the sampled token is
            # discarded by ``_discard_partial_prefill_samples``), and the
            # per-request counter inside ``_debug_write_logits`` would
            # otherwise tag each chunk-end logit with a sequential
            # ``num_prompt_tokens - 1 + emitted`` position, misaligning all
            # subsequent real prefill+decode entries against the golden cache.
            partial_prefill_req_ids = self._get_partial_prefill_req_ids(
                scheduler_output, self.input_batch.req_ids
            )
            if spec_decode_metadata is not None:
                # In async mode the sampler returns the raw rejection tensor
                # (not a parsed list), so materialize it here for the debug
                # writer. This path is test-only and doesn't affect the async
                # hot path.
                accepted_ids = sampler_output.sampled_token_ids
                if isinstance(accepted_ids, torch.Tensor):
                    accepted_ids = self._parse_rejection_sampling_output(
                        accepted_ids, self.input_batch.vocab_size
                    )
                # Write logits for all accepted token positions (including
                # the correction position). The correction position's logit
                # was computed with the rejected draft token as input, but
                # the correct logit will be recomputed in the next verify
                # step. The collector uses the last entry per position_id,
                # so the correct logit overwrites the wrong one.
                #
                # Note: requests with num_draft_tokens=0 (newly joined in DI)
                # are skipped here. Their logit from this step may be
                # incorrect due to the mixed spec-decode batch shape. Their
                # first valid logit will be written in a subsequent step
                # when they have proper draft tokens.
                accepted_indices = []
                accepted_req_ids = []
                offset = 0
                # accepted_ids may be padded to padded_num_reqs for spec decode;
                # only write logits for real requests.
                real_num_reqs = self.input_batch.num_reqs
                for req_i, req_tokens in enumerate(accepted_ids):
                    if req_i >= real_num_reqs:
                        break
                    n_sampled = 1 + spec_decode_metadata.num_draft_tokens[req_i]
                    req_id = req_ids_output_copy[req_i]
                    if req_id in partial_prefill_req_ids:
                        offset += n_sampled
                        continue
                    n_accepted = len(req_tokens)
                    for j in range(n_accepted):
                        idx = offset + j
                        if idx < logits_to_write.shape[0]:
                            accepted_indices.append(idx)
                            accepted_req_ids.append(req_id)
                    offset += n_sampled
                if accepted_indices:
                    indices_t = torch.tensor(accepted_indices, dtype=torch.long)
                    self._debug_write_logits(
                        logits_to_write[indices_t],
                        accepted_req_ids,
                    )
            else:
                # Only write logits for real requests (not padding rows).
                # logits_to_write may be padded to max_num_seqs but only
                # the first len(req_ids_output_copy) rows are valid.
                num_real = len(req_ids_output_copy)
                kept_indices = [
                    i
                    for i, rid in enumerate(req_ids_output_copy[:num_real])
                    if rid not in partial_prefill_req_ids
                ]
                if kept_indices:
                    indices_t = torch.tensor(kept_indices, dtype=torch.long)
                    kept_req_ids = [req_ids_output_copy[i] for i in kept_indices]
                    self._debug_write_logits(
                        logits_to_write[indices_t],
                        kept_req_ids,
                        jf_width_by_req=jf_width_by_req,
                    )

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )

        # If spec decode enabled, log acceptance stats and propose draft tokens.
        # aux_hidden_states is None when eagle3 is not active.
        if self.is_spec_decode and aux_hidden_states is not None:
            max_position = positions.max().item() if positions.numel() > 0 else 0
            num_spec_tokens = self.drafter.num_speculative_tokens
            # Stop proposing early enough that the scheduler never trims
            # draft tokens. After this step's verification with full acceptance,
            # num_computed_tokens becomes max_position + 1. The scheduler trims
            # when num_computed_tokens >= max_model_len - 1 - num_spec_tokens.
            # So we skip when max_position + 1 >= max_model_len - 1 - num_spec_tokens,
            # i.e., max_position >= max_model_len - num_spec_tokens - 2.
            #
            # Async scheduling adds one step of pipelining (batch_queue=2), so
            # by the time we decide to skip_propose at step K, step K+1 has
            # already been scheduled by the engine. To ensure K+1 doesn't
            # schedule drafts that the scheduler would then trim, we need to
            # skip one step earlier: subtract one full spec-decode step
            # (1 + num_spec_tokens) from the limit.
            spec_decode_limit = self.max_model_len - num_spec_tokens - 2
            if self.use_async_scheduling:
                spec_decode_limit -= 1 + num_spec_tokens

            # Skip draft proposal in two cases to avoid recompilation:
            # 1. Near max_model_len: proposed tokens would exceed the
            #    sequence length limit.
            # 2. Previous step had no spec decode (spec_decode_metadata is
            #    None in a decode step): the target ran with 1 token per
            #    request, so calling propose() would hit an unwarmed NEFF.
            #
            # For case 2 on DI decode servers: instead of leaving
            # _draft_token_ids=None (which permanently disables spec
            # decode), we inject placeholder tokens. The scheduler will
            # then schedule spec decode on the next step, bootstrapping
            # the verify→propose loop with the correctly-warmed NEFFs.
            num_reqs = self.input_batch.num_reqs
            skip_propose = max_position >= spec_decode_limit or (
                spec_decode_metadata is None
                and input_ids.shape[0]
                <= get_decode_padded_batch_size(
                    num_reqs,
                    1,  # max_query_len=1 (no spec decode)
                    self.neuron_config.num_seqs_buckets,
                    decode_token_threshold=1,
                )
            )

            if skip_propose:
                logger.debug(
                    "Skipping draft proposal: max_position=%d, limit=%d, "
                    "spec_decode_metadata=%s, num_tokens=%d, num_reqs=%d",
                    max_position,
                    spec_decode_limit,
                    "present" if spec_decode_metadata is not None else "None",
                    input_ids.shape[0],
                    self.input_batch.num_reqs,
                )
                # This rank skips the real proposal, but if the DP group is
                # speculating this step the draft NEFF runs cross-DP collectives
                # on the busy ranks. Dispatch a KV-safe dummy draft so those
                # collectives stay in lockstep. No-op unless the draft shards
                # embedding/lm_head over the enlarged TP group.
                if self._group_spec_enabled:
                    num_spec = self.drafter.num_speculative_tokens
                    ctx_bucket = None
                    if self._step_max_decode_ctx_len > 0:
                        ctx_bucket = self._decode_ctx_bucket_from_max_decode_ctx_len(
                            self._step_max_decode_ctx_len, num_spec
                        )
                    self._dispatch_dummy_draft(self._step_dummy_draft_batch, ctx_bucket)
                is_decode_server = (
                    self.vllm_config.kv_transfer_config is not None
                    and self.vllm_config.kv_transfer_config.is_kv_consumer
                )
                if (
                    is_decode_server
                    and spec_decode_metadata is None
                    and max_position < spec_decode_limit
                ):
                    # Bootstrap spec decode: inject placeholder drafts so
                    # the scheduler schedules spec decode on the next step.
                    # These will be rejected by the rejection sampler
                    # since they won't match the target model's output.
                    # On the next step the target model verifies with
                    # spec decode NEFFs, after which the real EAGLE3
                    # proposer takes over.
                    num_spec_tokens = self.drafter.num_speculative_tokens
                    pad_token_id = getattr(
                        self.vllm_config.model_config.hf_config,
                        "pad_token_id",
                        None,
                    )
                    if pad_token_id is None:
                        logger.warning(
                            "Model does not define pad_token_id, falling "
                            "back to token 0 for DI spec decode placeholder "
                            "drafts."
                        )
                        pad_token_id = 0
                    self._draft_token_ids = [
                        [pad_token_id] * num_spec_tokens for _ in range(num_reqs)
                    ]
                else:
                    self._draft_token_ids = None
            else:
                logger.debug(
                    "Proposing drafts: max_position=%d, "
                    "spec_decode_metadata=%s, num_tokens=%d, num_reqs=%d",
                    max_position,
                    "present" if spec_decode_metadata is not None else "None",
                    input_ids.shape[0],
                    self.input_batch.num_reqs,
                )
                self._draft_token_ids = self._propose_draft_token_ids(
                    scheduler_output=scheduler_output,
                    sampled_token_ids=sampler_output.sampled_token_ids,
                    aux_hidden_states=aux_hidden_states,
                    spec_decode_metadata=spec_decode_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    attn_metadata=attn_metadata,
                    logits_indices=logits_indices,
                    raw_sampled_token_ids=raw_sampled_token_ids,
                )

        # Trim sampler output back to real num_reqs before output generation.
        # With spec decode batch padding, the sampler runs at padded_num_reqs;
        # output/bookkeeping expect only real requests.
        #
        # Skip when async scheduling is active: the device tensor future must
        # retain its padded shape so the next step's swap logic
        # (future.shape[0] == input_ids.shape[0]) matches the
        # decode-batch-padded input_ids. Trimming to [real_num_reqs] while
        # input_ids stays at [padded_num_reqs] would permanently disable the
        # async device-to-device path. The async materialization path in
        # get_output() already indexes by snapshot_req_ids, so only real
        # requests get their tokens written back.
        if not self.use_async_scheduling:
            real_num_reqs = self.input_batch.num_reqs
            sids = sampler_output.sampled_token_ids
            if isinstance(sids, list) and len(sids) > real_num_reqs:
                sampler_output.sampled_token_ids = sids[:real_num_reqs]
            elif isinstance(sids, torch.Tensor) and sids.shape[0] > real_num_reqs:
                sampler_output.sampled_token_ids = sids[:real_num_reqs]

        # Generate output
        with record_function_or_nullcontext("neuron_model_runner: postprocess"):
            output = self._generate_model_runner_output(
                sampler_output,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                kv_connector_output,
                scheduler_output,
            )

        logger.debug("=== Completed NeuronModelRunner.execute_model ===")

        if not self.use_async_scheduling:
            logger.debug(f"Model runner output: {output}")
            return output

        async_output = AsyncNeuronModelRunnerOutput(
            model_runner_output=output,
            model_runner=self,
            partial_prefill_req_ids=getattr(
                self, "_async_partial_prefill_req_ids", set()
            ),
        )
        self.async_execution_buffer["async_output"] = async_output
        return async_output

    @staticmethod
    def _is_decode_shaped(attn_metadata: dict | None) -> bool:
        """Return True when the attention metadata describes a decode-shape step.

        Read the shape rather than calling ``_is_decode()``, which compares computed tokens
        against the prompt length and is False on a decode-shape admission step. Every reorder
        site uses this predicate, so warmup and runtime agree on the forward inputs.

        Example:
            >>> meta = {"layers.0.self_attn": {"max_query_len": 1,
            ...                                "decode_token_threshold": 1}}
            >>> NeuronModelRunner._is_decode_shaped(meta)
            True
        """
        if not attn_metadata:
            return False
        layer_metadata = next(iter(attn_metadata.values()))
        return (
            layer_metadata["max_query_len"] <= layer_metadata["decode_token_threshold"]
        )

    def _use_async_reorder(self) -> bool:
        """Return True when the in-graph reorder is the active decode path.

        The reorder assembles the decode input_ids on the device from the previous step's future
        through a permutation and a mask, so a change of batch composition needs no host copy.

        Four cases keep the earlier swap path: a model that does not consume the arguments, whose
        forward would raise or embed stale tokens; speculative decode, which has its own
        draft-NEFF handoff; multimodal, whose encoder path needs the materialized future; and
        synchronous scheduling, where no cross-step future exists.

        Example:
            >>> runner._use_async_reorder()  # doctest: +SKIP
            True
        """
        # Resolve until the answer is a real bool. A stand-in object is not an opt-in, so
        # the gate fails closed on it and the decode keeps the host swap path.
        if not isinstance(self._model_supports_async_reorder, bool):
            # A runner without a loaded model cannot consume the reorder, so fail closed.
            model = getattr(self, "model", None)
            self._model_supports_async_reorder = _supports_async_reorder(model)
            logger.info(
                "In-graph async reorder: the model %s the reorder arguments (%s).",
                "consumes" if self._model_supports_async_reorder else "ignores",
                type(getattr(model, "_orig_mod", model)).__module__,
            )
        return (
            self._model_supports_async_reorder is True
            and self.use_async_scheduling
            and not self.is_spec_decode
            and not self.supports_mm_inputs
        )

    @staticmethod
    def _noop_async_reorder_kwargs(
        num_slots: int, dtype: torch.dtype, device: torch.device
    ) -> dict:
        """Build reorder inputs that pass the host input_ids through unchanged.

        The identity permutation with an all-True mask makes
        ``where(mask, input_ids, index_select(future, 0, perm))`` return ``input_ids``.

        The future has to be a separate tensor: ``torch.compile`` removes a duplicate
        placeholder, the graph then takes fewer inputs than the real reorder NEFF, and the
        cross-DP collective fails. Warmup, an idle rank and the first step use this, so every
        rank dispatches the same NEFF and only the data changes.

        Args:
            num_slots: Number of input_ids slots in the step.
            dtype: Data type of the input_ids.
            device: Device that holds the forward inputs.

        Returns:
            The three reorder inputs of the forward.
        """
        return {
            "async_reorder_future": torch.zeros(num_slots, dtype=dtype, device=device),
            "async_reorder_perm": torch.arange(
                num_slots, dtype=torch.long, device=device
            ),
            "async_reorder_mask": torch.ones(
                num_slots, dtype=torch.bool, device=device
            ),
        }

    def _build_async_reorder_kwargs(
        self, cpu_input_ids: torch.Tensor
    ) -> tuple[dict, bool]:
        """Build the reorder inputs for one plain decode step, plus a usable flag.

        The flag is True when a reusable ``[num_slots]`` future exists. The compiled forward then
        assembles the input_ids on the device, and the host never reads the future because the
        permutation and the mask come from the request id strings alone: a continuing request
        reads the reordered future, while a new request, a resumed request and a pad slot read
        the host input_ids. The caller skips the swap.

        The flag is False on the first step and after a change of decode bucket or dtype, where
        no reusable future exists. The result is then a no-op reorder, which the caller still
        needs so every plain decode step dispatches the same NEFF, and it must also run the swap
        and the materialize because the no-op returns the host input_ids unchanged.

        Args:
            cpu_input_ids: Host input_ids of this step.

        Returns:
            A tuple of the reorder inputs and the usable flag.

        Example:
            >>> kwargs, usable = runner._build_async_reorder_kwargs(ids)  # doctest: +SKIP
            >>> usable  # doctest: +SKIP
            True
        """
        num_slots = cpu_input_ids.shape[0]
        future = self.async_execution_buffer.get("futures_sampled_token_ids")
        previous_req_ids = self.async_execution_buffer.get("prev_req_ids_ordered")
        usable = (
            isinstance(future, torch.Tensor)
            and future.ndim == 1
            and future.shape == cpu_input_ids.shape
            and previous_req_ids is not None
        )
        if usable:
            future = _reinterpret_uint32_as_int32(future, cpu_input_ids.dtype)
            usable = future.dtype == cpu_input_ids.dtype
        if not usable:
            return (
                self._noop_async_reorder_kwargs(
                    num_slots, cpu_input_ids.dtype, self.device
                ),
                False,
            )

        previous_index = {req_id: i for i, req_id in enumerate(previous_req_ids)}
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        # The mask defaults to True, meaning read the host input_ids, which is right for a
        # new request, a resumed request, a row that still prefills, and a pad slot.
        #
        # Membership in the previous batch is not enough: a row whose chunked prefill has one
        # token left is a member, but its next input is its last prompt token rather than what
        # the sampler produced. So test the decode phase too.
        permutation = np.zeros(num_slots, dtype=np.int64)
        mask = np.ones(num_slots, dtype=bool)
        for slot in range(num_slots):
            req_id = req_ids[slot] if slot < num_reqs else None
            source = previous_index.get(req_id) if req_id is not None else None
            if source is not None and self._is_decode_row(slot):
                permutation[slot] = source
                mask[slot] = False
        # Build a new tensor on each step. The device copy can be asynchronous, so a
        # reused host buffer could change while an earlier copy still reads it.
        return (
            {
                "async_reorder_future": future,
                "async_reorder_perm": torch.from_numpy(permutation).to(self.device),
                "async_reorder_mask": torch.from_numpy(mask).to(self.device),
            },
            True,
        )

    def _maybe_swap_async_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace CPU-built ``input_ids`` with a device tensor built from
        the previous step's NEFF output futures, when async scheduling is
        in steady state.

        Two shapes of previous-step output are handled:

        * **Spec decode** (steady state): previous step's draft NEFF
          output is ``[bs, 1 + num_spec]`` with the bonus token already
          prepended (column 0 = bonus, columns 1.. = drafts). Flattening
          via ``view(-1)`` yields the next target NEFF's input_ids
          directly — no Python-side cat or gather between NEFFs.
        * **Non-spec** (or prefill): previous step's sampler output is
          ``[bs]``. Matches the next target NEFF's ``[bs]`` input_ids
          1-to-1, so we reuse it directly.

        In either case the returned tensor stays on device — the Neuron
        runtime handles the data dependency on the producing NEFF. When
        the flow is broken (first iteration, batch composition change,
        shape/dtype mismatch), falls back to the CPU-built ``input_ids``.

        Args:
            input_ids: CPU-built input_ids from ``_prepare_model_input``.

        Returns:
            Either the original ``input_ids`` (sync fallback) or a device
            tensor from the previous step's NEFF output future.
        """
        # spec→non-spec transition: use the device-side bonus tensor
        # extracted from prev_future[:, 0] as this step's input_ids.
        # Scheduler's num_computed_tokens_cpu is one-step stale (doesn't
        # reflect prev spec step's rejections yet), so reading CPU
        # token_ids_cpu at the scheduler offset would return garbage.
        if (
            self._batch_composition_changed
            and getattr(self, "_transition_bonus_tensor", None) is not None
        ):
            bonus_tensor = self._transition_bonus_tensor
            # Pad to match input_ids shape (bonus_tensor is [num_reqs_real],
            # input_ids is [padded_num_reqs]).
            if bonus_tensor.shape[0] < input_ids.shape[0]:
                pad_size = input_ids.shape[0] - bonus_tensor.shape[0]
                pad = torch.zeros(
                    pad_size, dtype=bonus_tensor.dtype, device=bonus_tensor.device
                )
                bonus_tensor = torch.cat([bonus_tensor, pad])
            self._sync_fallback_steps += 1
            return bonus_tensor

        if not (
            self.async_execution_buffer
            and self._is_decode()
            and not self._batch_composition_changed
        ):
            self._sync_fallback_steps += 1
            return input_ids

        future = self.async_execution_buffer["futures_sampled_token_ids"]
        draft_future = self.async_execution_buffer.get("futures_draft_token_ids")

        assembled = self._try_assemble_spec_input_ids(input_ids, draft_future)
        if assembled is not None:
            self._async_steps += 1
            return assembled

        assembled, forced = self._try_reuse_nonspec_future(input_ids, future)
        if assembled is not None:
            # Remap via _maybe_remap_async_future forces the future on the
            # worker main thread (clone + gather), so it's logically a sync
            # fallback even though we return a device tensor.
            if forced:
                self._sync_fallback_steps += 1
            else:
                self._async_steps += 1
            return assembled

        logger.debug(
            "Async future mismatch: future=(shape=%s, dtype=%s), "
            "input_ids=(shape=%s, dtype=%s). Using CPU-built input_ids.",
            tuple(future.shape),
            future.dtype,
            tuple(input_ids.shape),
            input_ids.dtype,
        )
        self._sync_fallback_steps += 1
        return input_ids

    def _try_assemble_spec_input_ids(
        self,
        input_ids: torch.Tensor,
        draft_future: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Try to assemble spec-decode input_ids from the draft NEFF output.

        The draft NEFF outputs ``[bs, 1 + num_spec]`` device tensor with
        column 0 = bonus token from target NEFF's rejection output, columns
        1.. = drafts. Flattening gives the next target NEFF's input_ids
        directly — no Python-side ops between the two NEFFs.

        Args:
            input_ids: CPU-built input_ids shape ``[bs * (1 + num_spec)]``
                (scheduler placeholders in spec positions).
            draft_future: Draft NEFF output ``[bs, 1 + num_spec]``.

        Returns:
            The flattened ``[bs * (1 + num_spec)]`` device future, or
            ``None`` if ``draft_future`` is absent or shape mismatches.
        """
        if draft_future is None or draft_future.ndim != 2:
            return None
        if draft_future.numel() != input_ids.shape[0]:
            return None
        draft_future = _reinterpret_uint32_as_int32(draft_future, input_ids.dtype)
        if draft_future.dtype != input_ids.dtype:
            return None
        return draft_future.view(-1)

    def _try_reuse_nonspec_future(
        self,
        input_ids: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        """Try to reuse a ``[bs]`` sampled-token future directly as input_ids.

        Applies to non-spec decode, where input_ids has shape ``[bs]``.
        Returns ``(future, forced)`` — ``future`` may be the original
        device future (after uint32→int32 reinterpretation) or a remapped
        clone when ``condense()`` has reordered the batch since the future
        was produced. ``forced`` is True when remapping cloned/forced the
        future on the worker main thread (accounted as sync fallback by
        the caller). Returns ``(None, False)`` if shape or dtype mismatch.
        """
        if future.shape != input_ids.shape:
            return None, False
        future = _reinterpret_uint32_as_int32(future, input_ids.dtype)
        if future.dtype != input_ids.dtype:
            return None, False
        remapped, forced = self._maybe_remap_async_future(future, input_ids)
        return remapped, forced

    def _is_ve_encode_step(self) -> bool:
        """True when this step must finish right after the encoder.

        A vision-encoder-only pool (mm_encoder_only=True) has no language model
        or sampler, so once _execute_mm_encoder has run we return a token-less
        output instead of falling through to the LM forward.

        That return happens before the language model's cross-DP collectives
        (_coordinate_dp_spec_decode / _get_dp_padding). Those collectives must
        run on every rank or not at all -- if some ranks skip them and others
        don't, the group deadlocks. A vision-only pool runs no language model,
        so they are inactive and skipping them is safe; the assert below
        enforces that and fails loudly rather than risk a silent hang. The
        encoder's own collectives are unaffected -- they already ran
        inside _execute_mm_encoder, before this return.
        """
        if not self.mm_encoder_only:
            return False
        if self._is_decode():
            # An encoder never decodes; never finish a decode step here.
            return False
        assert not self._dp_collectives_active(), (
            "VE encode-finish returns before the language-model cross-DP "
            "collectives; those collectives are active on this pool, so skipping "
            "them would hang the group. A vision-encoder-only pool must not "
            "enable language-model DP (expert/attention/embedding/mlp/lm_head) "
            "-- this is the LM-DP axis, not encoder_dp_size."
        )
        return True

    def _is_ec_producer(self) -> bool:
        """True when this pool is configured as an EC (encoder-cache) producer.

        Reads the upstream ec_transfer_config role (set at launch). Falls back
        to False when no EC transfer is configured, so the default monolith path
        and the standalone VE pool (no ec_role) are unaffected.
        """
        vllm_config = getattr(self, "vllm_config", None)
        ec_config = getattr(vllm_config, "ec_transfer_config", None)
        return bool(ec_config is not None and ec_config.is_ec_producer)

    def _attach_ec_locator(self, output, scheduler_output) -> None:
        """Build per-item EC locators {engine_id, remote_block_ids, num_tokens} and
        attach them to the VE output for the PD to pull by.

        Iterates each request's full mm_features (not scheduled_encoder_inputs) so
        cache-hit items, whose blocks are resident but carry no scheduled encoder
        input this step, still get a locator. Skips prompt_embeds (share the cache
        but bypass the VE transfer) and items with no resident blocks.
        """
        from vllm_neuron.vllm.ec_connector.neuron_nixl_ec_connector import (
            NeuronECWorkerMetadata,
        )

        scheduled_reqs = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not scheduled_reqs:
            return
        engine_id = self.vllm_config.ec_transfer_config.engine_id
        # The consumer needs the VE's handshake address to open the one-time
        # metadata exchange; carry it on every item's locator (mirrors #40695's
        # remote_host/remote_port). Resolved from the connector's started server.
        host, port = None, None
        connector = getattr(self, "ec_connector", None)
        if connector is not None and hasattr(connector, "side_channel_address"):
            host, port = connector.side_channel_address
        embed_locator: dict[str, dict] = {}
        for req_id in scheduled_reqs:
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            for mm_feature in req_state.mm_features or []:
                if mm_feature.modality == "prompt_embeds":
                    # A prompt_embeds item is a client-supplied precomputed
                    # embedding put directly into this same encoder cache (see
                    # _execute_mm_encoder), not a vision-tower output, so it is
                    # deliberately not advertised over NIXL.
                    continue
                mm_hash = mm_feature.identifier
                block_ids = self.encoder_cache.get_block_ids(mm_hash)
                if block_ids is None:
                    # Not resident (e.g. not yet encoded this step); nothing to
                    # advertise for it here.
                    continue
                # scratch_block_id is never a real allocation (excluded from the
                # free queue), so get_block_ids cannot return it -- but assert as
                # cheap defense against advertising a sentinel block over NIXL.
                assert self.encoder_cache.scratch_block_id not in block_ids, (
                    f"scratch block leaked into locator for {mm_hash}"
                )
                embed_locator[mm_hash] = {
                    "engine_id": engine_id,
                    "host": host,
                    "port": port,
                    "remote_block_ids": list(block_ids),
                    "num_tokens": self.encoder_cache.get_num_tokens(mm_hash),
                }
        if embed_locator:
            from vllm.v1.outputs import ECConnectorOutput

            # Worker metadata survives cross-rank EC output aggregation.
            output.ec_connector_output = ECConnectorOutput(
                ec_connector_worker_meta=NeuronECWorkerMetadata(
                    embed_locator=embed_locator
                ),
            )

    def _init_ec_connector(self, vllm_config):
        """Build the worker-role EC connector from ec_transfer_config, or None.

        Delegates to ensure_ec_transfer_initialized, which builds the WORKER-role
        connector only when ec_transfer_config.is_ec_transfer_instance (mirrors the
        upstream worker-init path); we hold a reference for the producer/consumer
        hooks. Returns None when no ec_transfer_config is set (the default path),
        so no EC code runs and the non-EC behavior is unchanged.
        """
        if getattr(vllm_config, "ec_transfer_config", None) is None:
            return None
        from vllm.distributed.ec_transfer import (
            ensure_ec_transfer_initialized,
            get_ec_transfer,
            has_ec_transfer,
        )

        ensure_ec_transfer_initialized(vllm_config)
        if not has_ec_transfer():
            return None
        connector = get_ec_transfer()
        ec_config = vllm_config.ec_transfer_config
        assert connector.is_consumer == bool(ec_config.is_ec_consumer) and (
            connector.is_producer == bool(ec_config.is_ec_producer)
        ), (
            "EC connector role mismatch: process global is "
            f"(consumer={connector.is_consumer}, producer={connector.is_producer}) "
            f"but this pool is (consumer={ec_config.is_ec_consumer}, "
            f"producer={ec_config.is_ec_producer}). One EC role per process."
        )
        return connector

    def _is_ec_consumer(self) -> bool:
        """True when this pool is configured as an EC (encoder-cache) consumer.

        Symmetric with _is_ec_producer; False when no EC transfer is configured
        so the default monolith path is unaffected.
        """
        vllm_config = getattr(self, "vllm_config", None)
        ec_config = getattr(vllm_config, "ec_transfer_config", None)
        return bool(ec_config is not None and ec_config.is_ec_consumer)

    def _run_ec_consumer_lifecycle(self, scheduler_output) -> None:
        """Start the consumer-side EC load (issue the NIXL READs) for this step.

        On an ec_consumer (PD) pool, the vision embeddings live on a remote VE;
        this reserves local landing slots and issues the asynchronous NIXL READs
        into this pool's encoder cache. The completion barrier is separate:
        maybe_wait_for_ec_load() blocks at the top of _gather_mm_embeddings before
        any block is read. No-op on the producer / default path, and when no
        remote items are scheduled.
        """
        connector = getattr(self, "ec_connector", None)
        if connector is None or not self._is_ec_consumer():
            return
        ec_meta = getattr(scheduler_output, "ec_connector_metadata", None)
        load = getattr(ec_meta, "load", None)
        if not load:
            return

        # Reserve landing slots in this pool's own encoder cache (NIXL reads
        # straight into EncoderCacheBlocks.buffer via the existing allocate). Record
        # them as local_block_ids so start_load_caches maps remote_block_ids (the
        # VE's blocks) -> local_block_ids (our blocks) for the READ.
        for mm_hash, locator in load.items():
            local_ids = self.encoder_cache.allocate(
                mm_hash,
                self.encoder_cache.dense_tokens_per_block(
                    locator["num_tokens"], self.encoder_cache.block_size
                ),
            )
            locator["local_block_ids"] = list(local_ids)

        # Issue the READs. Our worker reads the load list from the metadata=
        # kwarg, so bind/clear_connector_metadata is a no-op here today; it is a
        # forward-compat shim mirroring the upstream EC path (bound metadata),
        # kept so the switch to the stock mixin when vLLM #40695 lands is a
        # drop-in.
        connector.bind_connector_metadata(ec_meta)
        try:
            connector.start_load_caches(self.encoder_cache, metadata=ec_meta)
        finally:
            connector.clear_connector_metadata()

    def maybe_register_ec_cache(self) -> None:
        """Register the encoder-cache buffer with the EC connector (post-warmup).

        No-op on the default path (no ec_connector) and when there is no encoder
        cache. The producer (VE) also starts its handshake server here. Called by
        the worker after warmup, when self.encoder_cache.buffer is a materialized
        HBM tensor with a stable data_ptr (registration needs the real address).
        """
        connector = getattr(self, "ec_connector", None)
        cache = getattr(self, "encoder_cache", None)
        if connector is None or cache is None:
            return
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        connector.register_and_serve(cache.buffer, cache.mm_hash_tags, tp_rank=tp_rank)

    def _is_decode_row(self, i: int) -> bool:
        """Check if batch row ``i`` is a request in decode phase.

        A request is in decode phase when all its prompt tokens have been
        computed, i.e. ``num_computed_tokens >= num_prompt_tokens``.
        Prefix-caching may set ``num_computed_tokens`` to a non-zero value
        before the first forward pass, so we cannot simply check for zero.

        Returns:
            True if row ``i`` is in decode phase; False for a prefill row or an
            unknown request (conservative).
        """
        req_id = self.input_batch.req_ids[i]
        req_state = self.requests.get(req_id)
        if req_state is None:
            # Unknown request — be conservative, treat as prefill
            return False
        num_computed = self.input_batch.num_computed_tokens_cpu[i]
        return num_computed >= req_state.num_prompt_tokens

    def _is_decode(self) -> bool:
        """Check if current batch is decode-only (no prefill requests).

        This is used by the async scheduling path in ``execute_model()`` to
        decide whether the previous step's sampled-token future can be reused
        as ``input_ids``. Feeding a stale 1-token future into a new prefill
        request causes a shape mismatch in the NKI attention kernel.

        Returns:
            True if all requests are in decode phase, False otherwise
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs == 0:
            return False

        return all(self._is_decode_row(i) for i in range(num_reqs))

    def _maybe_remap_async_future(
        self,
        future: torch.Tensor,
        cpu_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Remap the device future if condense() reordered the batch.

        When async scheduling reuses the previous step's sampled-token tensor
        as ``input_ids``, the tokens are in the slot ordering from when the
        future was produced. If ``condense()`` has since shifted requests to
        fill gaps left by finished requests, the slot ordering no longer
        matches and each request would receive another request's token.

        This method compares the current batch ordering against the stored
        ``prev_req_ids_ordered`` and, if they differ, builds a permutation
        index to gather the future's tokens into the correct slots on-device.

        Args:
            future: The device tensor of sampled token IDs from the previous
                step, shaped ``[padded_batch_size]``.
            cpu_input_ids: CPU-built input IDs for the current step. Used as
                fallback if the previous future cannot be remapped safely.

        Returns:
            A tuple of ``(input_ids, broke_async)``. ``broke_async`` is False
            only when the original future can be reused directly. If remapping
            is needed, the clone/gather materializes the future on the worker
            main thread, so ``broke_async`` is True. If a request in the
            current batch wasn't present in the previous step (shouldn't happen
            when composition_changed is False, but handled defensively), the
            CPU-built input IDs are returned and ``broke_async`` is True.
        """
        prev_req_ids_ordered = self.async_execution_buffer.get("prev_req_ids_ordered")
        num_reqs = self.input_batch.num_reqs
        if prev_req_ids_ordered is None or num_reqs == 0:
            return future, False

        # Build a mapping from req_id -> old slot index.
        prev_id_to_idx = {
            req_id: idx for idx, req_id in enumerate(prev_req_ids_ordered)
        }

        needs_remap = False
        perm: list[int] = []
        for cur_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[cur_idx]
            old_idx = prev_id_to_idx.get(req_id)
            if old_idx is None:
                # Request wasn't in the previous step. This shouldn't happen
                # when composition_changed is False, but if it does, signal
                # the caller to fall back to the CPU-built input_ids.
                logger.debug("Async future remap failed, using CPU-built input_ids.")
                return cpu_input_ids, True
            perm.append(old_idx)
            if old_idx != cur_idx:
                needs_remap = True

        if not needs_remap:
            return future, False

        # Gather tokens from old positions into new positions on-device.
        # Only remap the real request slots; padding slots stay as-is.
        # NOTE: clone() forces the future and therefore breaks the steady-state
        # async path for accounting purposes, even though the remapped tensor
        # remains the correct input for this step.
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=future.device)
        remapped = future.clone()
        remapped[:num_reqs] = future[perm_tensor]
        logger.debug("Remapped device future: perm=%s", perm)
        return remapped, True

    def _uses_async_spec_transition_cache(self) -> bool:
        """True when async spec decode hands draft tokens NEFF-to-NEFF as a
        cross-step device future, so the transition cache must replay those
        futures across batch-composition changes.
        """
        return self.use_async_scheduling and self.is_spec_decode

    def _cache_async_spec_transition_drafts(
        self,
        req_ids_output_copy: list[str],
    ) -> None:
        """Stash spec-decode draft rows by req_id so we can replay them later.

        Steady state hands draft futures straight from one NEFF to the next —
        previous and current batches match in shape and order, so nothing else
        is needed. At a prefill->decode batch-composition change that breaks:
        the previous future only covers the request that just prefilled, but
        the next decode is bs-wide. The cache fills that gap. It lives in the
        plugin; we don't rely on upstream scheduler draft propagation.
        """
        if not self._uses_async_spec_transition_cache():
            return

        full_draft_tokens = self._draft_token_ids
        if not (
            isinstance(full_draft_tokens, torch.Tensor) and full_draft_tokens.ndim == 2
        ):
            return

        full_draft_tokens = _reinterpret_uint32_as_int32(full_draft_tokens, torch.int32)

        n_rows = min(len(req_ids_output_copy), full_draft_tokens.shape[0])

        for i in range(n_rows):
            req_id = req_ids_output_copy[i]
            self._async_spec_transition_full_draft_cache[req_id] = full_draft_tokens[
                i
            ].detach()

        self._prune_async_spec_transition_cache(req_ids_output_copy)

        logger.debug(
            "Cached async spec transition drafts for req_ids=%s full_shape=%s",
            req_ids_output_copy[:n_rows],
            tuple(full_draft_tokens.shape),
        )

    def _prune_async_spec_transition_cache(
        self, extra_live_req_ids: list[str] | None = None
    ) -> None:
        """Drop cached rows for requests that are no longer live."""
        extra_live_req_ids = extra_live_req_ids or []
        live_req_ids = set(self.requests.keys()) | set(extra_live_req_ids)
        for req_id in list(self._async_spec_transition_full_draft_cache.keys()):
            if req_id not in live_req_ids:
                del self._async_spec_transition_full_draft_cache[req_id]

    def _missing_async_spec_transition_cache_req_ids(
        self,
        req_ids: list[str],
        num_draft_tokens: list[int],
    ) -> list[str]:
        """Req ids whose cached rows can't satisfy this step's draft demand."""
        if len(req_ids) != len(num_draft_tokens):
            return req_ids

        missing: list[str] = []
        for req_id, n_draft in zip(req_ids, num_draft_tokens):
            full_row = self._async_spec_transition_full_draft_cache.get(req_id)
            if full_row is None or full_row.numel() < 1 + n_draft:
                missing.append(req_id)
        return missing

    def _get_cached_transition_draft_pieces(
        self,
        req_ids: list[str],
        num_draft_tokens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._missing_async_spec_transition_cache_req_ids(req_ids, num_draft_tokens):
            return None

        full_pieces: list[torch.Tensor] = []
        drafts_pieces: list[torch.Tensor] = []

        for req_id, n_draft in zip(req_ids, num_draft_tokens):
            full_row = self._async_spec_transition_full_draft_cache[req_id]
            full_pieces.append(full_row[: 1 + n_draft])
            if n_draft:
                drafts_pieces.append(full_row[1 : 1 + n_draft])

        if not full_pieces:
            return None

        full_flat = torch.cat(full_pieces).to(dtype=torch.int32)
        if drafts_pieces:
            drafts_flat = torch.cat(drafts_pieces).to(dtype=torch.int32)
        else:
            drafts_flat = torch.empty(
                0, dtype=torch.int32, device=full_pieces[0].device
            )
        return full_flat, drafts_flat

    def _drop_transition_spec_decode_if_cache_incomplete(
        self,
        scheduler_output: Any,
    ) -> None:
        """Strip spec decode for this step if any cached row is missing.

        A normal batch-change step keeps the scheduler's placeholder
        ``scheduled_spec_decode_tokens`` and overwrites target inputs and
        rejection metadata from cached draft rows later. If even one row is
        missing we'd end up verifying ``-1`` placeholders, so we drop spec
        decode for this one step and let the next proposer output refill
        the cache.
        """
        if not (
            self._uses_async_spec_transition_cache()
            and getattr(self, "_batch_composition_changed", False)
            and getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
        ):
            return

        req_ids = list(scheduler_output.scheduled_spec_decode_tokens.keys())
        num_draft_tokens = [
            len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for req_id in req_ids
        ]
        missing_req_ids = self._missing_async_spec_transition_cache_req_ids(
            req_ids, num_draft_tokens
        )
        if not missing_req_ids:
            return

        logger.warning(
            "Missing async spec transition drafts for req_ids=%s; "
            "running this batch-composition-change step without spec decode.",
            missing_req_ids,
        )
        for req_id, n_draft in list(
            (req_id, len(draft_ids))
            for req_id, draft_ids in scheduler_output.scheduled_spec_decode_tokens.items()
        ):
            if req_id in scheduler_output.num_scheduled_tokens:
                scheduler_output.num_scheduled_tokens[req_id] -= n_draft
            if req_id in scheduler_output.num_scheduled_tokens_padded:
                scheduler_output.num_scheduled_tokens_padded[req_id] -= n_draft
            scheduler_output.total_num_scheduled_tokens -= n_draft
        scheduler_output.scheduled_spec_decode_tokens.clear()

    def _consume_async_spec_transition_cache(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            self._async_spec_transition_full_draft_cache.pop(req_id, None)

    def _fill_transition_input_and_spec_metadata_from_cache(
        self,
        input_ids: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> torch.Tensor:
        """Fill target inputs and rejection metadata from the cached rows."""
        if not (
            self._uses_async_spec_transition_cache()
            and getattr(self, "_batch_composition_changed", False)
            and spec_decode_metadata is not None
        ):
            return input_ids

        num_reqs = self.input_batch.num_reqs
        req_ids = list(self.input_batch.req_ids)[:num_reqs]
        num_draft_tokens = spec_decode_metadata.num_draft_tokens[:num_reqs]
        pieces = self._get_cached_transition_draft_pieces(req_ids, num_draft_tokens)
        if pieces is None:
            # _drop_transition_spec_decode_if_cache_incomplete should already
            # have stripped spec decode before spec metadata was built. If we
            # got here, something upstream broke that invariant — fail loudly
            # rather than silently verifying placeholder drafts.
            raise RuntimeError(
                "Missing async spec transition drafts after spec metadata "
                f"was built for req_ids={req_ids}"
            )

        full_flat, drafts_flat = pieces
        if full_flat.numel() > input_ids.numel():
            raise RuntimeError(
                "Cached async spec transition input_ids exceed placeholder "
                f"size: cached={full_flat.numel()} placeholder={input_ids.numel()}"
            )
        if drafts_flat.numel() > spec_decode_metadata.draft_token_ids.numel():
            raise RuntimeError(
                "Cached async spec transition draft metadata exceeds "
                "placeholder size: cached="
                f"{drafts_flat.numel()} placeholder="
                f"{spec_decode_metadata.draft_token_ids.numel()}"
            )

        # Keep padded placeholder slots in input_ids and write the real
        # transition draft rows over the live ones. Cloning after .to()
        # avoids mutating the CPU-built fallback when input_ids already
        # shares full_flat's device.
        filled_input_ids = input_ids.to(
            device=full_flat.device,
            dtype=input_ids.dtype,
        ).clone()
        filled_input_ids[: full_flat.numel()] = full_flat.to(
            device=filled_input_ids.device,
            dtype=filled_input_ids.dtype,
        )
        filled_draft_token_ids = spec_decode_metadata.draft_token_ids.to(
            device=drafts_flat.device,
            dtype=spec_decode_metadata.draft_token_ids.dtype,
        ).clone()
        filled_draft_token_ids[: drafts_flat.numel()] = drafts_flat.to(
            device=filled_draft_token_ids.device,
            dtype=filled_draft_token_ids.dtype,
        )
        spec_decode_metadata.draft_token_ids = filled_draft_token_ids
        self._consume_async_spec_transition_cache(req_ids)
        logger.debug(
            "Filled async spec batch-change inputs from transition cache: "
            "req_ids=%s input_tokens=%d draft_tokens=%d",
            req_ids,
            full_flat.numel(),
            drafts_flat.numel(),
        )
        return filled_input_ids

    def _build_prompt_embeds_tensors(
        self,
        input_ids: torch.Tensor,
        token_indices_tensor: torch.Tensor,
        req_ids: list[str],
        scheduler_output: Any,
        padding_map: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build batch-aligned `inputs_embeds` and `is_token_ids` tensors.

        vLLM flattens scheduled requests into one per-step token layout.
        `is_token_ids` is a per-position mask used by `torch.where`:
        True keeps `embed_tokens(input_ids)`, False uses `inputs_embeds`.

        Example for one request with 5 prompt tokens padded to 8:
          - token ids: [10, 20, 30, 40, 50, 0, 0, 0]
          - user embeddings are provided for positions 1..3

          inputs_embeds:
            [[0,...], [e1], [e2], [e3], [0,...], [0,...], [0,...], [0,...]]
          is_token_ids:
            [True, False, False, False, True, True, True, True]

        In the model backbone:
          torch.where(is_token_ids, embed_tokens_output, inputs_embeds)
        keeps `embed_tokens_output` where True, and swaps in user embeddings
        where False.
        """
        num_tokens = input_ids.shape[0]
        dtype = torch.bfloat16

        inputs_embeds = torch.zeros(num_tokens, self.inputs_embeds_size, dtype=dtype)
        is_token_ids = torch.ones(num_tokens, dtype=torch.bool)

        if not self.enable_prompt_embeds:
            return inputs_embeds, is_token_ids

        if (
            not hasattr(self.input_batch, "req_prompt_embeds")
            or not self.input_batch.req_prompt_embeds
        ):
            return inputs_embeds, is_token_ids

        if hasattr(self.input_batch, "is_token_ids_tensor"):
            is_token_ids_flat = self.input_batch.is_token_ids_tensor.flatten()
            # src_idx walks scheduled (unpadded) tokens; dst_idx walks padded output slots.
            src_idx = 0
            dst_idx = 0
            flat_max_idx = is_token_ids_flat.shape[0] - 1

            for req_id in req_ids:
                num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                padding = padding_map.get(req_id, 0)
                total_for_req = num_sched + padding

                if num_sched > 0:
                    req_token_indices = token_indices_tensor[
                        src_idx : src_idx + num_sched
                    ]
                    req_token_indices = req_token_indices.clamp(0, flat_max_idx)
                    req_is_token_ids = torch.index_select(
                        is_token_ids_flat, 0, req_token_indices
                    )
                    is_token_ids[dst_idx : dst_idx + num_sched] = req_is_token_ids

                src_idx += num_sched
                dst_idx += total_for_req

        output_idx = 0
        for i, req_id in enumerate(req_ids):
            req_idx = self.input_batch.req_id_to_index.get(req_id)
            if req_idx is None:
                num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                padding = padding_map.get(req_id, 0)
                output_idx += num_sched + padding
                continue

            num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            padding = padding_map.get(req_id, 0)
            total_for_req = num_sched + padding

            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += total_for_req
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]
            actual_num_sched = min(num_sched, req_embeds.shape[0] - start_pos)
            if actual_num_sched > 0:
                embed_slice = req_embeds[start_pos : start_pos + actual_num_sched]
                inputs_embeds[output_idx : output_idx + actual_num_sched] = (
                    embed_slice.to(dtype)
                )

            output_idx += total_for_req

        logger.debug(
            "Built prompt_embeds tensors: inputs_embeds=%s, is_token_ids=%s, "
            "num_embed_tokens=%d",
            inputs_embeds.shape,
            is_token_ids.shape,
            int((~is_token_ids).sum().item()),
        )

        return inputs_embeds, is_token_ids

    @contextmanager
    def _snapshot_capture_context(self, positions, spec_decode_metadata):
        """Scope input-snapshot capture to this model's forward.

        Thin adapter over the shared :class:`SnapshotRunnerHook`: resolves the
        runner-specific inputs (request ids, decode phase, speculative window)
        the hook cannot know generically. The disabled path returns before
        touching any of them so capture stays zero-cost as opt-in debug.
        """
        if not envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
            yield
            return

        spec_window = 1
        spec_cfg = getattr(self.vllm_config, "speculative_config", None)
        if spec_decode_metadata is not None and spec_cfg is not None:
            spec_window = (getattr(spec_cfg, "num_speculative_tokens", 0) or 0) + 1
        with self._snapshot_hook.capture_context(
            req_ids=self.input_batch.req_ids,
            positions=positions,
            is_decode=self._is_decode(),
            spec_window=spec_window,
        ):
            yield

    def _execute_model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
        logit_mask: torch.Tensor | None = None,
        rotary_position_ids: torch.Tensor | None = None,
        mm_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Execute the model forward pass.

        Args:
            input_ids: Input token IDs
            positions: Position IDs
            attn_metadata: Attention metadata
            spec_decode_metadata: Spec decode metadata for rejection sampling
            kv_caches: Key-value caches

        Returns:
            - model_output_tensor: torch.Tensor
                Can be logits (for CPU sampling)
                or sampled token ids (already sampled on device)
        """
        # TODO: avoid f-string in production logging as f-strings are evaluated even if
        # the log is not printed
        if not self.use_async_scheduling:
            # we need to avoid actualizing the input_ids tensor futures here.
            logger.debug(
                "Starting model forward pass with input_ids=%s, positions=%s",
                input_ids,
                positions,
            )

        # Move inputs to Neuron device
        input_ids = input_ids.to(self.device)
        positions = positions.to(self.device)
        logits_indices = logits_indices.to(self.device)
        if spec_decode_metadata is not None:
            spec_decode_metadata.cu_num_draft_tokens = (
                spec_decode_metadata.cu_num_draft_tokens.to(self.device)
            )
            # logits_indices / target_logits_indices / bonus_logits_indices
            # are consumed on-device by the target NEFF (rejection sampler,
            # bonus extraction, draft_token_ids gather).
            spec_decode_metadata.logits_indices = (
                spec_decode_metadata.logits_indices.to(self.device)
            )
            spec_decode_metadata.target_logits_indices = (
                spec_decode_metadata.target_logits_indices.to(self.device)
            )
            spec_decode_metadata.bonus_logits_indices = (
                spec_decode_metadata.bonus_logits_indices.to(self.device)
            )
            # draft_token_ids may be a CPU placeholder (async mode) that was
            # overwritten earlier by a device future from the draft NEFF, or
            # the sync-mode CPU gather result; move to device either way.
            spec_decode_metadata.draft_token_ids = (
                spec_decode_metadata.draft_token_ids.to(self.device)
            )
        # Prepare sampling params tensor for on-device sampling
        sampling_params_tensor = None
        if self.on_device_sampling:
            sampling_metadata = self.input_batch.sampling_metadata
            num_reqs = self.input_batch.num_reqs

            sampling_params_tensor = build_sampling_params_tensor(
                sampling_metadata, num_reqs, self.device
            )

            # Pad sampling_params to match the decode batch bucket size.
            # block_table rows == padded_num_reqs (1 token per request),
            # but num_reqs may be smaller before padding.
            if attn_metadata:
                first_meta = next(iter(attn_metadata.values()))
                bt_rows = first_meta["block_table_tensor"].shape[0]
                if sampling_params_tensor.shape[0] < bt_rows:
                    pad_rows = bt_rows - sampling_params_tensor.shape[0]
                    sampling_params_tensor = torch.nn.functional.pad(
                        sampling_params_tensor, (0, 0, 0, pad_rows), value=0
                    )

            # Replicate sampling params for spec-decode verify step
            # (no-op when speculative decoding is not active).
            sampling_params_tensor = self._maybe_replicate_for_spec_decode(
                sampling_params_tensor, spec_decode_metadata
            )

            sampling_params_tensor = sampling_params_tensor.to(self.device)

        # Execute model on Neuron device (compiled with vllm_neuron backend)
        model_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "attn_metadata": attn_metadata,
            "sampling_positions": logits_indices,
            "sampling_params": sampling_params_tensor,
            "spec_decode_metadata": spec_decode_metadata,
            "rank": self.rank_tensor,
        }
        # Test the type. The reorder inputs are a dict, or the reorder is off.
        if isinstance(self._async_reorder_kwargs, dict):
            model_kwargs.update(self._async_reorder_kwargs)
        model_kwargs["logit_mask"] = logit_mask
        if rotary_position_ids is not None:
            model_kwargs["rotary_position_ids"] = rotary_position_ids.to(self.device)

        # On-device num_computed_tokens correction (async spec decode only).
        # Provide the previous step's rejection sampler output and the per-
        # scheduled-token request index so the target NEFF can subtract
        # rejections from the optimistic positions without a CPU sync.
        #
        # Only inject for decode steps. Prefill takes the no-correction
        # branch in the decorator. Warmup precompiles both graph variants.
        is_decode_step = (
            attn_metadata
            and next(iter(attn_metadata.values()))["max_query_len"]
            <= next(iter(attn_metadata.values()))["decode_token_threshold"]
        )
        if self._model_is_async_spec_decoded() and is_decode_step:
            num_reqs_padded = (
                spec_decode_metadata.bonus_logits_indices.shape[0]
                if spec_decode_metadata is not None
                else input_ids.shape[0]
            )
            async_spec_kwargs = self._build_async_spec_kwargs(
                spec_decode_metadata,
                num_total_tokens=input_ids.shape[0],
                num_reqs_padded=num_reqs_padded,
            )
            model_kwargs.update(async_spec_kwargs)

        # Multimodal kwargs (encoder_embedding_buffer + encoder_block_ids + encoder_positions)
        if mm_kwargs:
            model_kwargs.update(mm_kwargs)

        # Text prompt_embeds kwargs (inputs_embeds + is_token_ids).
        # Mutually exclusive with mm_kwargs (matches upstream's if/elif pattern).
        if self.enable_prompt_embeds and not self.supports_mm_inputs:
            if self._current_inputs_embeds is not None:
                embeds = self._current_inputs_embeds
                is_tids = self._current_is_token_ids
                if embeds.device != self.device:
                    embeds = embeds.to(self.device)
                    is_tids = is_tids.to(self.device)
                model_kwargs["inputs_embeds"] = embeds
                model_kwargs["is_token_ids"] = is_tids
            else:
                num_tokens = input_ids.shape[0]
                model_kwargs["inputs_embeds"] = torch.zeros(
                    num_tokens,
                    self.inputs_embeds_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                model_kwargs["is_token_ids"] = torch.ones(
                    num_tokens,
                    dtype=torch.bool,
                    device=self.device,
                )

        # Execute model on Neuron device (compiled with vllm_neuron backend)
        if self._tensor_replacer is not None:
            first_attn_meta = (
                next(iter(attn_metadata.values())) if attn_metadata else None
            )
            if first_attn_meta is not None:
                is_prefill = (
                    first_attn_meta["max_query_len"]
                    > first_attn_meta["decode_token_threshold"]
                )
            else:
                is_prefill = len(positions) > len(self.input_batch.req_ids)
            set_active_context(
                self._tensor_replacer.build_context(
                    req_ids=list(self.input_batch.req_ids),
                    positions=positions.cpu().tolist(),
                    is_prefill=is_prefill,
                    device=self.device,
                    tokens_per_req=first_attn_meta["decode_token_threshold"]
                    if first_attn_meta and not is_prefill
                    else 1,
                )
            )
        if self._target_tensor_capture is not None:
            self._target_tensor_capture.setup_tensor_capture()

        # Tag the plain single-token decode graph with the "decode" kind. It is
        # uniquely identified by per-request query length 1 with no spec-decode
        # metadata, which excludes prefill (query len = bucket), JF (query len =
        # 1 + jf_max), and the spec verify graph. Other graphs use self.model
        # directly (the model returns the base args for them).
        _first_am = next(iter(attn_metadata.values())) if attn_metadata else None
        is_plain_decode = (
            _first_am is not None
            and _first_am["max_query_len"] == 1
            and spec_decode_metadata is None
        )
        with self._snapshot_capture_context(positions, spec_decode_metadata):
            with model_forward_context(self.vllm_config):
                if is_plain_decode:
                    with self._plain_decode_forward() as forward_model:
                        model_output = forward_model(**model_kwargs)
                else:
                    model_output = self.model(**model_kwargs)
        if self._tensor_replacer is not None:
            set_active_context(None)

        if self._target_tensor_capture is not None:
            # Determine is_prefill early for capture write
            _first_am = next(iter(attn_metadata.values())) if attn_metadata else None
            _is_pf = (
                _first_am is not None
                and _first_am["max_query_len"] > _first_am["decode_token_threshold"]
            )
            self._target_tensor_capture.save_tensor_captures(
                positions=positions,
                is_prefill=_is_pf,
                req_ids=list(self.input_batch.req_ids),
            )

        # Determine bucket name for NEFF execution count metric.
        first_attn_metadata = (
            next(iter(attn_metadata.values())) if attn_metadata else None
        )
        is_prefill = (
            first_attn_metadata is not None
            and first_attn_metadata["max_query_len"]
            > first_attn_metadata["decode_token_threshold"]
        )
        if is_prefill:
            neff_bucket_name = f"prefill_s{input_ids.shape[0]}"
            kv_segment_size = first_attn_metadata.get("kv_segment_size", 0)
            if kv_segment_size > 0:
                neff_bucket_name += f"_kv{kv_segment_size}"
        else:
            num_reqs = self.input_batch.num_reqs or input_ids.shape[0]
            tokens_per_req = input_ids.shape[0] // num_reqs
            neff_bucket_name = f"decode_b{num_reqs}"
            if self.drafter is not None:
                neff_bucket_name += f"_s{tokens_per_req}"

        NEFF_EXECUTION_COUNT.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=neff_bucket_name,
        ).inc()

        # Parse model output based on on-device sampling and spec decode configuration
        self._on_device_logits = None
        last_accepted_token: torch.Tensor | None = None

        if self.on_device_sampling:
            if self.is_spec_decode:
                if spec_decode_metadata is not None:
                    # Eagle3 + spec decode: model returns 4-tuple
                    # (sampled_tokens, aux_hidden_states, gathered_logits,
                    #  last_accepted_token). The fourth element is emitted
                    # by the rejection sampler and is shape [bs] with
                    # default stride; consumers (spec→non-spec transition)
                    # use it as input_ids without slicing or stride
                    # manipulation on a NEFF output.
                    (
                        model_output_tensor,
                        aux_hidden_states,
                        self._on_device_logits,
                        last_accepted_token,
                    ) = model_output
                else:
                    # Eagle3 non-spec: model returns 3-tuple
                    # (sampled_tokens, aux_hidden_states, gathered_logits)
                    (
                        model_output_tensor,
                        aux_hidden_states,
                        self._on_device_logits,
                    ) = model_output
            else:
                if isinstance(model_output, tuple):
                    # TODO: Require all ODS models to return (sampled_tokens, gathered_logits)
                    # and remove this isinstance check once gpt-oss and other models are updated.
                    model_output_tensor, self._on_device_logits = model_output
                else:
                    model_output_tensor = model_output
                aux_hidden_states = None
        elif self.is_spec_decode:
            # Eagle3 without ODS: model returns (logits, aux_hidden_states)
            model_output_tensor, aux_hidden_states = model_output
        else:
            model_output_tensor = model_output
            aux_hidden_states = None
            # Store logits for debug logits writing in non-ODS mode
            if self._debug_logits_dir:
                self._on_device_logits = model_output_tensor

        # Move model_output_tensor (logits or sampled token ids) back to CPU
        if not self.use_async_scheduling:
            model_output_tensor = model_output_tensor.to("cpu")

        return model_output_tensor, aux_hidden_states, last_accepted_token

    def _has_component_dp(self) -> bool:
        """Return True if any component DP size > 1 (attention, embedding, mlp, lm_head)."""
        nc = self.neuron_config
        return (
            nc.attention_dp_size > 1
            or nc.embedding_dp_size > 1
            or nc.mlp_dp_size > 1
            or nc.draft_mlp_dp_size > 1
            or nc.lm_head_dp_size > 1
        )

    def _dp_collectives_active(self) -> bool:
        """True iff this step actually issues cross-DP collectives.

        Cross-DP collectives only fire when ``data_parallel_size > 1`` AND
        either expert parallelism is enabled (MoE all-to-all/all-gather) or a
        component DP is enabled (attention_dp / embedding_dp / mlp_dp /
        lm_head_dp introduces all-gather/all-reduce on TP supergroups that
        span DP). When this returns False, any DP-coordination reduce is a
        no-op and should be skipped — both for correctness (no participants
        on the other side) and to avoid unnecessary host-side traffic on
        single-process / TP-only setups.
        """
        if self.vllm_config.parallel_config.data_parallel_size == 1:
            return False
        ep_enabled = self.vllm_config.parallel_config.enable_expert_parallel
        return ep_enabled or self._has_component_dp()

    def _local_step_forces_nonspec(self, scheduler_output: "SchedulerOutput") -> bool:
        """True iff this rank must run the *non-spec* decode NEFF this step.

        Mirror of the local DI mixed-batch condition: with speculation
        configured, a rank dispatches the non-spec decode graph whenever it
        has no scheduled draft tokens (e.g. the first decode step right after
        a request's KV cache is transferred in — EAGLE3 has no
        aux_hidden_states to propose from yet) or when only *some* requests in
        the batch carry drafts (a mixed batch, which the runner strips down to
        non-spec). When speculation is not configured every step is non-spec,
        so there is nothing to reconcile and we report ``False``.

        Caller contract: only invoked after ``input_batch`` is populated. The
        sole caller is ``_prepare_model_input_impl`` (to compute this rank's
        ``local_force_nonspec`` vote), which runs inside ``execute_model``
        after ``_update_states``, so ``input_batch`` is always initialized.
        """
        if self.speculative_config is None:
            return False
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            # No drafts scheduled at all → this rank runs the 1-token,
            # non-spec decode NEFF (typical first decode after KV load).
            return True
        # Mixed batch: at least one scheduled request has zero drafts.
        assert self.input_batch is not None, (
            "_local_step_forces_nonspec requires an initialized input_batch; "
            "call only after _update_states/load_model"
        )
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids[:num_reqs]
        return any(not spec_tokens.get(req_id) for req_id in req_ids)

    def _maybe_strip_spec_for_nonspec_step(
        self,
        scheduler_output: "SchedulerOutput",
        local_force_nonspec: bool,
        group_force_nonspec: bool,
    ) -> None:
        """Strip this step down to the non-spec decode NEFF when required.

        DI mixed-batch guard + DP spec coordination, applied in-place on
        ``scheduler_output``. With DP + EP the spec-decode vs non-spec decode
        decision selects which compiled NEFF runs, and that NEFF contains the
        cross-DP collectives. Every rank must dispatch the *same* graph or the
        collectives mismatch (``status=1100 Collective Operation Pending``).

        We strip when EITHER:
          - ``local_force_nonspec`` — this rank itself can't speculate this
            step (no scheduled drafts after a fresh KV transfer, or a mixed
            batch where only some requests carry drafts). The spec-decode NEFF
            would produce incorrect logits for the 0-draft requests.
          - ``group_force_nonspec`` — a peer DP rank voted non-spec, so even
            though this rank's local batch looks like a clean full-spec step it
            must drop its drafts to keep the cross-DP NEFF uniform.

        Both flags come from the single ``_get_dp_padding`` MAX-reduce
        (``local_force_nonspec`` is this rank's input, ``group_force_nonspec``
        the reduced result), so they are consistent across the group. The
        placeholder-draft injection re-bootstraps speculation on the next step.
        """
        if not scheduler_output.scheduled_spec_decode_tokens:
            return
        if not (local_force_nonspec or group_force_nonspec):
            return

        req_ids = self.input_batch.req_ids[: self.input_batch.num_reqs]
        logger.debug(
            "DI mixed-batch: stripping spec decode tokens "
            "(local_force_nonspec=%s, group_force_nonspec=%s). req_ids=%s, "
            "spec_decode_keys=%s",
            local_force_nonspec,
            group_force_nonspec,
            req_ids,
            list(scheduler_output.scheduled_spec_decode_tokens.keys()),
        )
        # Remove draft tokens from num_scheduled_tokens for requests that had
        # them, so input preparation builds 1-token-per-req. This drops draft
        # *tokens* only — num_reqs (and thus padded_num_reqs, already reduced
        # across DP) is unchanged.
        for req_id, draft_ids in list(
            scheduler_output.scheduled_spec_decode_tokens.items()
        ):
            n_draft = len(draft_ids)
            if req_id in scheduler_output.num_scheduled_tokens:
                scheduler_output.num_scheduled_tokens[req_id] -= n_draft
            if req_id in scheduler_output.num_scheduled_tokens_padded:
                scheduler_output.num_scheduled_tokens_padded[req_id] -= n_draft
            scheduler_output.total_num_scheduled_tokens -= n_draft
        scheduler_output.scheduled_spec_decode_tokens.clear()

    def _dp_reduce_identity_constant(self) -> int | None:
        """Return the batch size every DP rank contributes, or None when it is not provable.

        Returns a value only when the configuration pins the first five inputs of
        :meth:`_get_dp_padding`, which makes their MAX an identity, so each rank can compute the
        group result from its own inputs. All six conditions are static configuration, fixed
        before the first step, so one check at start is sound:

        1. Asynchronous scheduling. Removing the reduce removes the only thing pacing an idle
           rank, and the replacement readback lives on the asynchronous path.
        2. A pure KV consumer, so every prompt arrives over the network and no prefill-shape step
           runs, which would not bucket the batch size.
        3. One ``num_seqs_buckets`` entry, so every rank rounds to the same constant.
        4. ``decode_context_length_buckets`` unset.
        5. No ``speculative_config``.
        6. ``max_jump_forward_tokens`` is 0.

        The sixth input, ``local_is_real``, is NOT pinned; :meth:`_get_dp_padding` handles it.

        Returns:
            The pinned batch size, or None when the reduce must run.
        """
        if self._dp_identity_resolved:
            return self._dp_identity_constant

        config = self.neuron_config
        buckets = getattr(config, "num_seqs_buckets", None) or ()
        pinned = (
            # Asynchronous scheduling only. Removing the reduce also removes the pacing an idle
            # rank relied on, and the replacement, _drain_previous_dummy_output, is installed on
            # the asynchronous path alone. Keep the reduce under synchronous scheduling so that
            # mode keeps both the pacing and the all-dummy skip it has today.
            self.use_async_scheduling
            and self._pure_consumer
            and len(buckets) == 1
            and not getattr(config, "decode_context_length_buckets", None)
            and self.speculative_config is None
            and getattr(config, "max_jump_forward_tokens", 0) == 0
        )
        self._dp_identity_constant = int(buckets[0]) if pinned else None
        self._dp_identity_resolved = True
        logger.info(
            "DP reduce identity fast path: %s. async_scheduling=%s, pure_kv_consumer=%s, "
            "num_seqs_buckets=%s, decode_context_length_buckets=%s, speculative=%s, "
            "max_jump_forward_tokens=%s",
            "enabled" if pinned else "not provable, so the reduce runs",
            self.use_async_scheduling,
            self._pure_consumer,
            list(buckets),
            getattr(config, "decode_context_length_buckets", None),
            self.speculative_config is not None,
            getattr(config, "max_jump_forward_tokens", 0),
        )
        return self._dp_identity_constant

    def _get_dp_padding(
        self,
        num_tokens: int,
        local_max_decode_ctx_len: int = 0,
        is_spec_decode: bool = False,
        local_force_nonspec: bool = False,
        local_jf: bool = False,
        local_is_real: bool = True,
        local_is_decode_shaped: bool = False,
    ) -> tuple[int, int, bool, bool, bool, bool]:
        """Synchronize per-step shape inputs across DP ranks in one all-reduce.

        **Critical: keeps cross-DP collectives from deadlocking or
        corrupting hidden state.**

        In DP deployments with MoE/expert layers or component DP
        (attention_dp / embedding_dp / mlp_dp / lm_head_dp), collectives
        (all-to-all, all-gather, all-reduce) operate across DP ranks.
        Those collectives require identical tensor shapes from every
        participant, and they live inside compiled NEFFs whose shapes are
        determined by per-step inputs. If ranks dispatch to different NEFFs:
          - Operations expect uniform shapes but receive mismatched shapes
          - Best case: hang indefinitely waiting for data that never arrives
          - Worst case: collectives complete but cross-feed data from
            structurally divergent graphs, corrupting hidden state silently
            (observed on GPT-OSS-120B `tp8 dp8 ep64` DI: garbage `\\n`
            repeated from the first decode token).

        Five per-step inputs vary per rank and feed shape decisions; all
        must be reconciled to a single value across the DP group (inputs 3-4,
        the spec flag and the spec/non-spec vote, are what fix the DI
        dp4tp8ep32 Eagle3 crash — ranks fall out of phase as KV transfers
        complete independently, so some are on their first non-spec decode
        while others speculate, and a divergent NEFF choice mismatches the
        cross-DP collectives):

        1. **`num_tokens`** — local padded batch size. Each rank rounds
           to the smallest configured `num_seqs_buckets`; if rank A has
           2 requests and rank B has 5, A buckets to 2 and B to 8. The
           MoE all-to-all expects uniform `(num_tokens, hidden)` from
           every rank, so we MAX-reduce and pad up.
        2. **`local_max_decode_ctx_len`** — `max(num_computed_tokens)`
           across this rank's active slots. Drives the
           `decode_context_length_buckets` pick, which sizes the non-SWA
           `block_table_tensor`'s second dim. A rank at ctx=100 picks
           bucket 2048 (64 blocks); a rank at ctx=4000 picks 8192 (256
           blocks). Different second dim ⇒ different compiled NEFF.
        3. **`is_spec_decode`** — whether this rank scheduled draft tokens
           this step. MAX-reduced to `any_rank_spec_decode` so idle dummy
           ranks dispatch the same spec-vs-non-spec NEFF as the busy ranks.
        4. **`local_force_nonspec`** — whether this rank *must* run the
           non-spec NEFF (first decode after KV transfer / mixed batch).
           MAX-reduced to `group_force_nonspec` so the whole group falls
           back to non-spec when any rank can't speculate.
        5. **`local_jf`** — whether this rank has jump-forward tokens to
           emit this step. The entire DP group dispatches the JF NEFF on any
           step where at least one rank has JF work. On steps where
           no rank has JF tokens the group we use the 1-token NEFF.

        We send all five values (`num_tokens`, `local_max_decode_ctx_len`,
        `is_spec_decode`, `local_force_nonspec`, `local_jf`) since all are
        needed at the same dispatch decision point. Idle dummy ranks contribute
        `0` for the context length, the spec vote, and the JF vote.

        We reduce the **input** to bucket selection (max decode ctx len)
        rather than the chosen bucket-blocks because
        `get_bucket_for_count` is monotonic — each rank picks the same
        bucket from the synced value independently, and the helper that
        does the pick stays a pure function with no implicit DP
        coordination.

        Why CPU-side reduce: all four values are host-side dispatch
        decisions (they pick which compiled NEFF to call) and must
        materialize as Python ints before any device op is issued. A
        device-side reduce would force a device→host sync that defeats
        the purpose.

        Example preventing deadlock:
            4 DP ranks with token counts [100, 150, 80, 120] and ctx
            lengths [4000, 100, 100, 100]:
              Without sync → rank 0 dispatches to a different NEFF than
                             ranks 1-3 → cross-DP collectives corrupt
                             rank 0's hidden state.
              With sync → all ranks pad tokens to 150 and pick the
                          bucket for ctx 4000 → same NEFF on every rank
                          → safe.

        No-op when DP collectives are inactive (`dp_size == 1` or no
        EP / no component DP); returns
        `(0, local_max_decode_ctx_len, False, local_force_nonspec, local_jf)`
        so callers always get a usable value.

        Args:
            num_tokens: Pre-padding batch size on this rank.
            local_max_decode_ctx_len: `max(num_computed_tokens_cpu)`
                over active slots on this rank, or `0` when idle (dummy
                batch) or when `decode_context_length_buckets` is unset.
            is_spec_decode: True if this rank is executing a speculative
                decode step (with draft tokens). Propagated via MAX-reduce
                so idle ranks dispatch the same NEFF (spec vs non-spec).
            local_force_nonspec: True if this rank must run the non-spec
                decode NEFF this step (from ``_local_step_forces_nonspec``).
                MAX-reduced so the whole group falls back to non-spec when any
                rank can't speculate. Idle dummy ranks pass ``False``.
            local_jf: True if this rank has jump-forward tokens to emit this
                step. We take max ``any_rank_jf`` so the whole group
                dispatches the JF NEFF when any rank has JF work. Idle dummy
                ranks pass ``False``.
            local_is_real: True if this rank runs a real batch this step. The
                real input-prep path passes ``True``; ``execute_dummy_batch``
                passes ``False``. MAX-reduced to ``any_rank_real`` so an idle
                rank can detect an all-dummy step and skip the dummy NEFF.
            local_is_decode_shaped: True if this step is decode-shaped, which means
                ``max_query_len <= decode_token_threshold``. The identity fast path
                needs this, because the batch size is bucketed on a decode-shape step
                only. The default is False, so a new call site keeps the reduce.

        Returns:
            `(num_pad_tokens, max_decode_ctx_len, any_rank_spec_decode,
            group_force_nonspec, any_rank_jf, any_rank_real)` — additional rows
            this rank must pad to match the busiest rank, the DP-MAX of all
            ranks' local context lengths for downstream bucket selection,
            whether any rank is speculating, whether the group must fall back to
            the non-spec decode NEFF, whether the group must dispatch the JF
            decode NEFF, and whether any rank runs a real batch this step.
        """
        # TODO: Leverage vLLM's forward_context DPMetadata for DP coordination
        # Why: vLLM has built-in DP coordination via forward_context.py that could replace/enhance this logic
        # See vllm/forward_context.py:
        # - DPMetadata.num_tokens_across_dp() for token count synchronization via all_reduce
        # - DPMetadata.should_ubatch_across_dp() for coordinated microbatching decisions for PP cases
        # - Automatic integration via set_forward_context() when parallel_config.data_parallel_size > 1
        if not self._dp_collectives_active():
            return (
                0,
                local_max_decode_ctx_len,
                False,
                local_force_nonspec,
                local_jf,
                local_is_real,
            )

        # Consult the proof on every step. The proof reads static configuration only, so
        # every rank of the group reaches the same answer and no rank enters the collective
        # alone. Test the type, so a stand-in object keeps the collective.
        pinned_num_tokens = self._dp_reduce_identity_constant()
        if isinstance(pinned_num_tokens, int):
            # The configuration pins the first five inputs, so their MAX cannot change any
            # rank's answer and each rank computes the group result locally, with no
            # communication. A violated precondition RAISES rather than falling back into the
            # reduce: a collective is all or nothing, so one rank re-entering it would block
            # until the timeout while its peers moved on.
            if (
                not local_is_decode_shaped
                or num_tokens != pinned_num_tokens
                or local_max_decode_ctx_len
                or is_spec_decode
                or local_force_nonspec
                or local_jf
            ):
                raise RuntimeError(
                    "The DP reduce identity precondition failed. Expected a decode-shape step, "
                    f"num_tokens={pinned_num_tokens}, and every other input zero or False, but "
                    f"got local_is_decode_shaped={local_is_decode_shaped}, "
                    f"num_tokens={num_tokens}, "
                    f"local_max_decode_ctx_len={local_max_decode_ctx_len}, "
                    f"is_spec_decode={is_spec_decode}, "
                    f"local_force_nonspec={local_force_nonspec}, local_jf={local_jf}. "
                    "Change the configuration so the fast path turns off, or report this."
                )
            # The sixth output is an OR over the ranks. A rank running a real batch makes it true
            # by itself; a rank running a dummy cannot know, so it assumes True. That is safe
            # because it is UNIFORM, so every rank dispatches the same NEFF and the cross-DP
            # collectives stay matched. The only cost is one dummy batch on an all-dummy step,
            # which _drain_previous_dummy_output bounds.
            return (0, 0, False, False, False, True)

        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group().cpu_group
        t = torch.tensor(
            [
                num_tokens,
                local_max_decode_ctx_len,
                int(is_spec_decode),
                int(local_force_nonspec),
                int(local_jf),
                int(local_is_real),
            ],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
        (
            max_tokens_across_dp,
            max_decode_ctx_len,
            any_rank_spec,
            group_force_nonspec,
            any_rank_jf,
            any_rank_real,
        ) = (
            int(t[0]),
            int(t[1]),
            bool(t[2]),
            bool(t[3]),
            bool(t[4]),
            bool(t[5]),
        )
        num_pad = max_tokens_across_dp - num_tokens

        logger.debug(
            "DP coord: rank=%s local_tokens=%s max_tokens_across_dp=%s num_pad=%s "
            "local_max_decode_ctx_len=%s max_decode_ctx_len=%s "
            "any_rank_spec_decode=%s group_force_nonspec=%s any_rank_jf=%s "
            "any_rank_real=%s",
            self.vllm_config.parallel_config.data_parallel_rank,
            num_tokens,
            max_tokens_across_dp,
            num_pad,
            local_max_decode_ctx_len,
            max_decode_ctx_len,
            any_rank_spec,
            group_force_nonspec,
            any_rank_jf,
            any_rank_real,
        )
        return (
            num_pad,
            max_decode_ctx_len,
            any_rank_spec,
            group_force_nonspec,
            any_rank_jf,
            any_rank_real,
        )

    def _add_padding_to_inputs(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_pad: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add padding tokens to input tensors for DP synchronization.

        Args:
            input_ids: Original input token IDs
            positions: Original position IDs
            num_pad: Number of padding tokens to add

        Returns:
            tuple of (padded_input_ids, padded_positions)
        """
        if num_pad == 0:
            return input_ids, positions

        # Create padding tokens (use token_id=0, position=0)
        pad_ids = torch.zeros(num_pad, dtype=torch.int32)
        pad_pos = torch.zeros(num_pad, dtype=torch.long)

        # Concatenate original + padding
        # Tensors to torch.compile functions must be contiguous
        # (Neuron runtime requires contiguous input tensors for compiled graphs)
        # NOTE: This works for async execution as inputs are dummy inputs and there is
        # no data dependency from previous forward output.
        padded_input_ids = torch.cat([input_ids, pad_ids], dim=0).contiguous()
        padded_positions = torch.cat([positions, pad_pos], dim=0).contiguous()

        logger.debug(
            "Added %s padding tokens: input_ids %s → %s",
            num_pad,
            input_ids.shape,
            padded_input_ids.shape,
        )

        return padded_input_ids, padded_positions

    def _remove_padding_from_logits(
        self, logits: torch.Tensor, num_valid_tokens: int
    ) -> torch.Tensor:
        """
        Remove padding from logits tensor to get only valid outputs.

        Args:
            logits: Logits tensor that may contain padding
            num_valid_tokens: Number of valid (non-padding) tokens

        Returns:
            Logits tensor with padding removed
        """
        if logits.shape[0] == num_valid_tokens:
            # No padding to remove
            return logits

        # Slice to remove padding positions
        valid_logits = logits[:num_valid_tokens]

        logger.debug(
            "Removed padding from logits: %s → %s", logits.shape, valid_logits.shape
        )

        return valid_logits

    def _debug_write_logits(
        self,
        logits: torch.Tensor,
        req_ids: list[str],
        jf_width_by_req: dict[str, int] | None = None,
    ) -> None:
        """Write raw logits to shared memory for test infrastructure validation.

        Enabled by setting debug_logits_dir in additional_config.neuron_config.
        Only TP rank 0 writes to avoid duplicates within DP groups.
        Each DP rank writes its own file with _dp{rank} suffix.

        Position IDs are derived from a per-request cumulative counter
        ``self._debug_position_counter`` that tracks how many logits have
        already been emitted for each request. This is independent of any
        CPU ``positions`` tensor, which in async spec decode can carry
        optimistic (uncorrected-for-rejection) values. The counter-derived
        position matches the natural sequential ordering expected by
        golden-comparison tests.

        Filters out positions beyond each request's ``max_tokens`` boundary.
        This matters for async spec decode: the target NEFF may execute one
        step past ``max_tokens`` because the engine is one step ahead in the
        pipeline. The scheduler truncates client-visible output at
        ``max_tokens``, but the NEFF's logits still reach the debug writer.
        Without this filter, the extra logits pollute the debug dir and the
        test collector reads them on the NEXT request (due to stale step
        counter), producing wrong validation results.

        Binary format:
            Header (16 bytes): num_items (int64), vocab_size (int64)
            Per item: batch_idx (int64), position_id (int64), logits (float32[vocab_size])

        File naming: step_{N}_dp{rank}.bin

        The position_id allows the reader to consolidate logits across prefill/decode
        steps regardless of batch size variations. Each logit corresponds to exactly
        one sequence position where a token is generated.

        Jump-forward" when ``jf_width_by_req`` is supplied,
        we output a ``step_{N}_dp{rank}.jf.json`` file,
        mapping ``str(batch_idx) -> jf_width`` for the rows in this step.
        A width of ``n > 0`` means the wide JF NEFF processed n grammar-forced tokens
        before this step's (single) written logit. The JF-aware test collector
        uses these widths to locate each bonus token in the output
        stream.
        """
        # Only TP rank 0 writes (avoids duplicates within DP group)
        tp_group = get_tp_group()
        if tp_group and tp_group.rank_in_group != 0:
            return

        # data_parallel_index, NOT data_parallel_rank: vLLM resets
        # data_parallel_rank to 0 for dense (non-MoE) DP engines, so under
        # replica DP every engine would write the same step_*_dp0.bin and
        # clobber its peers. data_parallel_index reflects the true engine
        # index for both dense and MoE engines. Only a real int is trusted;
        # anything else falls back to data_parallel_rank.
        parallel_config = self.vllm_config.parallel_config
        dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if not isinstance(dp_rank, int) or isinstance(dp_rank, bool):
            dp_rank = parallel_config.data_parallel_rank
        logits_dir = self._debug_logits_dir
        os.makedirs(logits_dir, exist_ok=True)

        step = self._debug_logits_step_counter
        tmp_path = os.path.join(logits_dir, f"step_{step}_dp{dp_rank}.tmp")
        final_path = os.path.join(logits_dir, f"step_{step}_dp{dp_rank}.bin")

        # Write logits for all sampling positions (handles spec decode
        # verification where one step produces logits for multiple tokens).
        logits_cpu = logits.cpu().to(torch.float32).numpy()
        num_logit_rows = logits_cpu.shape[0]
        vocab_size = logits_cpu.shape[-1]

        # Per-request emitted counter — lazy initialized.
        if not hasattr(self, "_debug_position_counter"):
            self._debug_position_counter = {}

        items = []
        filtered_positions = []
        jf_width_items: dict[str, int] = {}  # str(batch_idx) -> jf_width
        final_req_ids = set()  # reqs whose final logit is in this step
        for row in range(num_logit_rows):
            if row >= len(req_ids):
                continue
            req_id = req_ids[row]
            try:
                batch_idx = int(req_id.split("-")[-2])
            except (ValueError, IndexError):
                batch_idx = row

            # Derive position from cumulative emitted logits for this req.
            # Logit at position p predicts output token at position p + 1.
            # First logit for a req (from prefill) is at position prompt_len - 1,
            # predicting output 0 at prompt_len. Per-req counter starts at 0.
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            emitted = self._debug_position_counter.get(req_id, 0)
            position_id = req_state.num_prompt_tokens - 1 + emitted

            # Skip positions beyond the request's max_tokens budget.
            # Valid range: [prompt_len - 1, prompt_len + max_tokens - 2] (inclusive),
            # i.e., max_tokens logits predicting output tokens 0..max_tokens-1.
            if req_state.sampling_params is not None:
                max_tokens = req_state.sampling_params.max_tokens
                if max_tokens is not None:
                    last_valid_pos = req_state.num_prompt_tokens + max_tokens - 2
                    if position_id > last_valid_pos:
                        filtered_positions.append(position_id)
                        continue
                    if position_id == last_valid_pos:
                        # Final logit for this req; emit a sentinel after it.
                        final_req_ids.add(req_id)

            items.append((batch_idx, position_id, logits_cpu[row]))
            self._debug_position_counter[req_id] = emitted + 1
            if jf_width_by_req:
                jf_width_items[str(batch_idx)] = int(jf_width_by_req.get(req_id, 0))

        if not items and not filtered_positions:
            # Nothing to do (all rows filtered out via exceptions etc.).
            return

        if items:
            with open(tmp_path, "wb") as f:
                np.array([len(items), vocab_size], dtype=np.int64).tofile(f)
                for batch_idx, position_id, logit_row in items:
                    np.array([batch_idx], dtype=np.int64).tofile(f)
                    np.array([position_id], dtype=np.int64).tofile(f)
                    logit_row.tofile(f)

            # Atomic rename signals completion to reader
            os.rename(tmp_path, final_path)

            # Emit per-batch JF widths for this step so the
            # JF-aware collector can find each sampled token.
            # Non-JF steps write no additional info
            if jf_width_items:
                jf_final_path = os.path.join(
                    logits_dir, f"step_{step}_dp{dp_rank}.jf.json"
                )
                jf_tmp_path = os.path.join(
                    logits_dir, f"step_{step}_dp{dp_rank}.jf.tmp"
                )
                with open(jf_tmp_path, "w") as f:
                    json.dump(jf_width_items, f)
                os.rename(jf_tmp_path, jf_final_path)

            self._debug_logits_step_counter = step + 1

            # Sentinel renamed after the .bin in this same path: seeing it
            # proves the request's step files are on disk (test-harness barrier).
            for done_req_id in final_req_ids:
                self._write_debug_logits_sentinel(
                    logits_dir, done_req_id, dp_rank, step
                )

        position_ids = [p for (_, p, _) in items]
        logger.debug(
            "LOGIT_DEBUG: step=%d wrote %d positions: %s (filtered beyond-max-tokens: %s)",
            step,
            len(position_ids),
            position_ids,
            filtered_positions,
        )

    def _write_debug_logits_sentinel(
        self,
        logits_dir: str,
        req_id: str,
        dp_rank: int,
        step: int,
    ) -> None:
        """Write ``done_<req_id>_dp{rank}`` (body: final step) after the last
        step file, so a reader seeing it knows all step files are on disk.

        Args:
            logits_dir: Directory the step files and sentinels are written to.
            req_id: Full request id as seen by the worker.
            dp_rank: Data parallel rank writing this sentinel.
            step: Step index of the request's final step file.
        """
        safe_req_id = req_id.replace("/", "_")
        tmp = os.path.join(logits_dir, f"done_{safe_req_id}_dp{dp_rank}.tmp")
        final = os.path.join(logits_dir, f"done_{safe_req_id}_dp{dp_rank}")
        with open(tmp, "w") as f:
            f.write(str(step))
        os.rename(tmp, final)

    def _dispatch_dummy_draft(self, batch_size: int, ctx_bucket: int | None) -> None:
        """Run a KV-safe dummy draft NEFF so a rank that does not propose keeps
        the draft's cross-DP collectives in lockstep with the busy ranks.

        No-op unless Eagle3 spec decode is active and the draft shards its
        embedding vocab, its lm_head vocab, or its MLP intermediate over an
        enlarged TP group. Only draft degrees count here. A target that gathers
        its rows changes the shape the draft slices, but a slice carries no
        collective, and draft_mlp_dp_size already follows mlp_dp_size unless it
        is set. These are the cases where the draft NEFF carries
        cross-DP collectives. The dummy writes to the reserved null block, so it
        does not touch a real request's KV. Safe on an idle rank or a busy rank
        that skipped the real proposal. The caller decides whether the group
        speculates this step.

        Args:
            batch_size: DP-padded request count the busy ranks propose with.
            ctx_bucket: Decode context bucket for the dummy attention metadata.

        Example:
            >>> self._dispatch_dummy_draft(batch_size=4, ctx_bucket=2048)
        """
        if not (
            (self.is_eagle3_spec or self.is_mtp_spec)
            and self.drafter is not None
            and (
                self.neuron_config.embedding_dp_size > 1
                or self.neuron_config.lm_head_dp_size > 1
                or self.neuron_config.draft_mlp_dp_size > 1
            )
        ):
            return
        assert self.speculative_config is not None
        num_spec_tokens = self.speculative_config.num_speculative_tokens
        draft_num_tokens = batch_size * (1 + num_spec_tokens)
        # null_kv=True: the dummy writes to the reserved null block, so it does
        # not clobber a real request's KV on a busy skip-propose rank. Use the
        # lean dispatch (no blocking readback) since this is on the step path.
        draft_attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=draft_num_tokens,
            num_reqs=batch_size,
            ctx_bucket=ctx_bucket,
            null_kv=True,
        )
        # dummy_draft_dispatch calls propose with is_warmup=False, which enters
        # model_forward_context itself, the same as the real proposal. Do not
        # wrap it here.
        self._dummy_draft_keepalive = self.drafter.dummy_draft_dispatch(
            num_tokens=draft_num_tokens,
            num_reqs=batch_size,
            attn_metadata=draft_attn_metadata,
        )

    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        When using Data Parallel (DP), if this rank has no real requests but
        other DP ranks do, this method runs a minimal dummy forward pass to
        keep all ranks synchronized during collective operations.

        This is called automatically by the vLLM engine when DP workload
        imbalance is detected (see vllm/v1/engine/llm_engine.py). vLLM does
        not wait for previous steps before calling this function, and its
        _has_global_unfinished_reqs function only all-reduces DP state every
        32 steps. So at a request boundary — while a request waits on its KV
        transfer, or just after one finishes — the whole DP group can be
        marked busy while no rank runs a real batch. In that all-dummy case
        there is no busy rank to partner, so this function skips the
        submission (see the any_rank_real check below). Without the submission
        the NRT execution queue cannot build up and overflow ("Execution
        Queue Full").

        All tensor shapes must exactly match what execute_model produces for
        a real decode batch of the same size, otherwise torch.compile traces
        a different graph and the collectives across DP ranks deadlock.
        """
        logger.debug("Executing dummy batch for DP synchronization")

        if not self._dp_collectives_active():
            return

        # Use the smallest decode bucket as the dummy batch size (num_reqs).
        # Use the smallest decode bucket as the dummy token count.  This rank
        # has zero real requests; the only purpose is to participate in the
        # cross-DP EP collectives so that busy ranks don't deadlock.
        # _get_dp_padding will pad us up to match whatever the busy rank
        # actually has, so starting at the smallest bucket avoids forcing the
        # busy rank to pad beyond its natural bucket size.
        batch_size = (
            self.neuron_config.num_seqs_buckets[0]
            if self.neuron_config.num_seqs_buckets
            else self.max_num_reqs
        )

        # Coordinate with other DP ranks via padding logic. Idle dummy rank
        # has no real ctx (local_max_decode_ctx_len=0), can't itself force
        # non-spec (local_force_nonspec=False), and has no JF tokens
        # (local_jf=False); the MAX-reduce promotes us to whatever the busy
        # ranks picked — batch size, decode ctx bucket, whether any rank
        # speculates, whether the group fell back to non-spec, and whether the
        # group runs the JF NEFF — so we dispatch the same NEFF.
        (
            num_pad,
            max_decode_ctx_len,
            any_rank_spec,
            group_force_nonspec,
            any_rank_jf,
            any_rank_real,
        ) = self._get_dp_padding(
            batch_size,
            local_max_decode_ctx_len=0,
            is_spec_decode=False,
            local_force_nonspec=False,
            local_jf=False,
            local_is_real=False,
            # A dummy batch is always decode-shaped. It runs the plain decode NEFF or
            # the jump-forward NEFF, and both classify as decode.
            local_is_decode_shaped=True,
        )

        # No busy rank this step: every DP rank runs a dummy batch. This happens
        # at a request boundary — during the KV transfer at request start, or
        # just after a request finishes while the coarse finish-sync still
        # reports the group busy. A dummy batch exists only to partner a busy
        # rank's cross-DP collective. With no busy rank, the submission is waste.
        # Additionally, an all-dummy group has no real rank to pace submission
        # against device completion, so the async execution queue free-runs and
        # can overflow (status=7 Execution Queue Full).
        if not any_rank_real:
            logger.debug("Dummy batch skipped: no busy DP rank this step")
            return

        if num_pad > 0:
            batch_size = batch_size + num_pad

        # Match the busy ranks' NEFF choice exactly. A busy rank votes its
        # PRE-strip spec flag into any_rank_spec but its strip decision into
        # group_force_nonspec, so on a step where a speculating rank is pulled
        # down by a non-spec peer BOTH are True — group_force_nonspec must win.
        # The ``and not group_force_nonspec`` is therefore load-bearing.
        spec_decode_enabled = any_rank_spec and not group_force_nonspec

        # Convert the DP-synced *raw* max_decode_ctx_len into a configured
        # ctx-length bucket *boundary* before handing it to
        # `_build_warmup_attention_metadata`. That helper treats `ctx_bucket`
        # as a boundary and sizes blocks as `ceil(ctx_bucket / dcp_block_size)`
        # — it does NOT round. The runtime decode path rounds first via the
        # shared `_decode_ctx_bucket_from_max_decode_ctx_len` picker, so the
        # dummy must round through the same picker or it dispatches a
        # wrong-width block_table and trips the torch.compile recompile guard
        # ("block_table_tensor size mismatch at index 1"). Passing the raw
        # length straight through collapses the width to 1 under DCP (where
        # dcp_block_size is large), which is the GPT-OSS 1P1D HMA TRN3PDS
        # startup failure.
        ctx_bucket = None
        if max_decode_ctx_len > 0:
            max_num_draft_tokens = (
                self.speculative_config.num_speculative_tokens
                if spec_decode_enabled
                else 0
            )
            ctx_bucket = self._decode_ctx_bucket_from_max_decode_ctx_len(
                max_decode_ctx_len, max_num_draft_tokens
            )

        # Make idle rank run JF NEFF if we're running JF NEFF on the DP group.
        if any_rank_jf:
            decode_kwargs = self._build_jf_dummy_inputs(batch_size, ctx_bucket)
        else:
            decode_kwargs = self._build_decode_synthetic_inputs(
                batch_size,
                spec_decode_enabled=spec_decode_enabled,
                decode_token_threshold=1 if not spec_decode_enabled else None,
                ctx_bucket=ctx_bucket,
            )
        if self.on_device_sampling and decode_kwargs.get("logit_mask") is not None:
            # Pair the busy ranks' runner-side lm_head_dp mask exchange. Idle
            # ranks send an all-True shard so every lm_head_dp member enters
            # the same host collective before dispatching the compiled graph.
            decode_kwargs["logit_mask"] = self._prepare_ods_logit_mask(
                decode_kwargs["logit_mask"]
            )

        # Execute model forward pass directly (same as warmup_decode) to
        # match the exact traced graph signature including all kwargs. Route the
        # plain single-token decode dummy (no JF, no spec) to the decode compile
        # variant so it hits the same NEFF the busy ranks run.
        with model_forward_context(self.vllm_config):
            if not any_rank_jf and not spec_decode_enabled:
                with self._plain_decode_forward() as plain_decode_forward:
                    model_output = plain_decode_forward(**decode_kwargs)
            else:
                model_output = self.model(**decode_kwargs)
        if self.use_async_scheduling:
            # The dummy batch produces no token anyone consumes; it exists so this idle DP rank
            # joins the cross-DP EP collective inside the NEFF. The backend slices the
            # input-output-aliased KV cache out of the returned tuple, so `model_output` is the
            # only live reference to the NEFF's NrtaFuture, and dropping it would let the
            # collector free the in-flight NRT input slices. So hold it either way.
            #
            # Which path to take depends on the DP reduce, a per-step host rendezvous that paces
            # this rank against the group. With the reduce a reference is enough, and a readback
            # would only add latency between two decode NEFFs. Without it nothing paces this
            # rank, so an idle queue can free-run and overflow; the drain then gives both the
            # reference and the pacing.
            if self._dp_reduce_identity_constant() is None:
                self._dummy_output_keepalive = model_output
            else:
                self._drain_previous_dummy_output(model_output)

        # >>> PARALLELISM: Draft DP lockstep. When a peer is speculating and the
        # Eagle3 draft shards its embedding/lm_head vocab weights over the
        # enlarged TP group, the draft NEFF runs cross-DP collectives. An idle
        # rank runs the target dummy above, but it must also run a dummy draft
        # proposal, or the busy ranks' draft collectives never complete and time
        # out (NRT_TIMEOUT). Mirror the startup draft warmup so the dummy draft
        # NEFF shape matches the busy ranks (batch_size is already DP-padded). <<<
        if spec_decode_enabled:
            self._dispatch_dummy_draft(batch_size, ctx_bucket)

        logger.debug("Dummy batch execution completed")

    def _generate_model_runner_output(
        self,
        sampler_output: SamplerOutput,
        req_ids_output_copy: list[str],
        req_id_to_index_output_copy: dict[str, int],
        kv_connector_output: KVConnectorOutput | None = None,
        scheduler_output: "SchedulerOutput | None" = None,
    ) -> ModelRunnerOutput:
        """
        Generate ModelRunnerOutput from model logits using vLLM Sampler.

        Args:
            sampler_output: sampler output
            req_ids_output_copy: snapshot of req_ids for async scheduling
            req_id_to_index_output_copy: snapshot of req_id_to_index
            kv_connector_output: kv connector output
            scheduler_output: scheduler output, used to detect partial prefills

        Returns:
            ModelRunnerOutput with correctly ordered results
        """
        if self.use_async_scheduling:
            self._async_partial_prefill_req_ids = self._get_partial_prefill_req_ids(
                scheduler_output,
                req_ids_output_copy,
            )
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs=(
                    sampler_output.logprobs_tensors.tolists()
                    if sampler_output.logprobs_tensors
                    else None
                ),
                prompt_logprobs_dict={},
                pooler_output=[None] * len(self.input_batch.req_ids),
                kv_connector_output=kv_connector_output,
            )
            self.async_execution_buffer["futures_sampled_token_ids"] = (
                sampler_output.sampled_token_ids
            )
            self._cache_async_spec_transition_drafts(req_ids_output_copy)
            # Store draft token future for cross-step device swap in spec mode.
            # None when no spec decode (prefill step, skip_propose fired, etc.)
            self.async_execution_buffer["futures_draft_token_ids"] = (
                self._draft_token_ids
                if isinstance(self._draft_token_ids, torch.Tensor)
                else None
            )
            # Drafts-only tensor (no bonus column) for the next step's
            # rejection sampler input. See ``_propose_draft_token_ids``.
            self.async_execution_buffer["futures_drafts_only"] = getattr(
                self, "_futures_drafts_only", None
            )
            self._futures_drafts_only = None
            # Record the ordered request IDs at the time the future was
            # produced. The next step uses this both for composition change
            # detection (set comparison) and for remapping the device future
            # when condense() reorders the batch (index lookup).
            self.async_execution_buffer["prev_req_ids_ordered"] = list(
                req_ids_output_copy
            )
            return model_runner_output

        # Not async scheduling, tensor to list conversion
        sampled_token_ids_tensor = sampler_output.sampled_token_ids

        # Spec decode: rejection sampler returns [B, max_spec_len+1] padded with -1.
        # Strip placeholders before updating state / returning output.
        if isinstance(sampled_token_ids_tensor, torch.Tensor):
            if sampled_token_ids_tensor.ndim == 1:
                sampled_token_ids_tensor = sampled_token_ids_tensor.unsqueeze(1)
            if (
                sampled_token_ids_tensor.ndim == 2
                and sampled_token_ids_tensor.shape[1] > 1
            ):
                sampled_token_ids = self._parse_rejection_sampling_output(
                    sampled_token_ids_tensor,
                    self.input_batch.vocab_size,
                )
            else:
                sampled_token_ids = sampled_token_ids_tensor.cpu().tolist()
        else:
            # Already a list for on-device sampling
            sampled_token_ids = sampled_token_ids_tensor

        # Discard sampled tokens from partial (chunked) prefill requests.
        # During chunked prefill, the model produces logits for the last
        # position of each chunk, but those are NOT valid output tokens
        # until the entire prompt has been processed.
        partial_prefill_req_ids = self._get_partial_prefill_req_ids(
            scheduler_output,
            self.input_batch.req_ids,
        )
        for req_idx, req_id in enumerate(self.input_batch.req_ids):
            if req_id not in partial_prefill_req_ids:
                continue
            logger.debug(
                "Discarding sampled token for partial prefill request %s",
                req_id,
            )
            if req_idx < len(sampled_token_ids):
                sampled_token_ids[req_idx] = []

        # Convert logprobs efficiently if present
        logprobs_lists = (
            sampler_output.logprobs_tensors.tolists()
            if sampler_output.logprobs_tensors
            else None
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(self.input_batch.req_ids),
            kv_connector_output=kv_connector_output,
        )

        # Update batch state with sampled tokens
        self._update_batch_state_with_samples(model_runner_output.sampled_token_ids)

        logger.debug(
            "Generated output for %s requests", len(model_runner_output.req_ids)
        )

        return model_runner_output

    def _get_partial_prefill_req_ids(
        self,
        scheduler_output: "SchedulerOutput | None",
        req_ids: list[str],
    ) -> set[str]:
        """Return request IDs whose current chunked-prefill step is not final."""
        if scheduler_output is None:
            return set()

        partial_prefill_req_ids = set()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id in req_ids:
            if req_id not in self.requests:
                continue
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            req_state = self.requests[req_id]
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            num_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_computed + num_scheduled < req_state.num_prompt_tokens:
                partial_prefill_req_ids.add(req_id)

        return partial_prefill_req_ids

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        """Run the pooler head on backbone hidden states and emit pooler_output.

        Args:
            hidden_states: Flattened ``[T, H]`` backbone hidden states.
            num_scheduled_tokens_np: Per-request scheduled token counts
                ``[num_reqs]``, used to build the pooling cursor's gather
                indices over the flattened buffer.

        Returns:
            ModelRunnerOutput whose ``pooler_output`` is a list of per-request
            CPU tensors (or None for unfinished requests).
        """
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling"
        )
        # Pooling has no decode loop, so async scheduling does not apply.
        assert not self.use_async_scheduling, (
            "Async scheduling is not supported for pooling models."
        )

        # Running seq length after this step = already-computed (prefix-cache
        # hits under APC) + scheduled-this-step. APC off / single-shot
        # => num_computed==0, so this equals num_prompt_tokens; under chunked
        # prefill it is < prompt_len on non-final chunks. Used only for
        # finished_mask below — the pooling cursor's gather index comes purely
        # from num_scheduled_tokens_np.
        seq_lens_cpu = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens_np
        )

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        raw_pooler_output = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            # Pooling models never do KV transfer, so this is always None.
            kv_connector_output=None,
        )

        if raw_pooler_output is None or not any(finished_mask):
            output.pooler_output = [None] * num_reqs
            return output

        output.pooler_output = self._copy_pooler_output_to_cpu(
            raw_pooler_output, finished_mask
        )
        return output

    @staticmethod
    def _copy_pooler_output_to_cpu(
        raw_pooler_output: "torch.Tensor | list[torch.Tensor | None]",
        finished_mask: list[bool],
    ) -> list[torch.Tensor | None]:
        if isinstance(raw_pooler_output, torch.Tensor):
            cpu_rows: list[torch.Tensor | None] = list(raw_pooler_output.to("cpu"))
        else:
            cpu_rows = [
                None if row is None else row.to("cpu") for row in raw_pooler_output
            ]
        return [
            row if include else None for row, include in zip(cpu_rows, finished_mask)
        ]

    def _sample(
        self,
        model_output_tensor: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> SamplerOutput:
        # Use InputBatch's automatically maintained sampling metadata
        # The metadata is created and updated by InputBatch.refresh_metadata() in _update_cached_states()
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        if self.use_async_scheduling:
            if spec_decode_metadata is not None:
                # Async spec decode: keep the raw rejection-sampler output
                # (device tensor [bs, num_spec+1]) as-is. Materialization is
                # deferred to the async output thread in
                # AsyncNeuronModelRunnerOutput.get_output().
                return SamplerOutput(model_output_tensor, None)
            return SamplerOutput(model_output_tensor, None)

        if spec_decode_metadata is None:
            # Jump-forward: emit [jf_0..jf_{n-1}, bonus] per request for this
            # step's sampled tokens. The wide NEFF computed KV for the prev decode
            # token + the n_jf JF tokens in one pass and sampled the bonus at
            # column n_jf.
            jf_tokens_active = getattr(self, "_jf_active_tokens", None)
            if jf_tokens_active and self.on_device_sampling:
                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                all_sampled = model_output_tensor.tolist()
                output_token_ids = []
                for i in range(num_reqs):
                    bonus = int(all_sampled[i]) if i < len(all_sampled) else 0
                    jf = list(jf_tokens_active.get(req_ids[i], ()))
                    output_token_ids.append(jf + [bonus])
                logger.debug("JF output: [jf..,bonus]=%s", output_token_ids)
                self._jf_active_tokens = None
                sampler_output = SamplerOutput(output_token_ids, None)
                return sampler_output

            if not self.on_device_sampling:
                logger.info("Using vLLM Sampler")
                sampler_output = self.sampler(
                    logits=model_output_tensor,
                    sampling_metadata=sampling_metadata,
                )
                if jf_tokens_active:
                    # If we sample on CPU, we must prepend the JF tokens to get
                    # the expected output size.
                    req_ids = self.input_batch.req_ids
                    bonus_ids = sampler_output.sampled_token_ids
                    output_token_ids = []
                    for i, bonus_row in enumerate(bonus_ids):
                        jf = list(jf_tokens_active.get(req_ids[i], ()))
                        output_token_ids.append(jf + list(bonus_row))
                    logger.debug(
                        "JF output (CPU sampling): [jf..,bonus]=%s",
                        output_token_ids,
                    )
                    # clear active tokens, set output
                    self._jf_active_tokens = None
                    sampler_output.sampled_token_ids = output_token_ids
            else:
                output_token_ids = [[x] for x in model_output_tensor.tolist()]
                # Use gathered logits to compute logprobs via vLLM Sampler
                logprobs = None
                if (
                    self._on_device_logits is not None
                    and (sampling_metadata.max_num_logprobs or 0) > 0
                ):
                    logits_cpu = self._on_device_logits.to("cpu")
                    logprobs_output = self.sampler(
                        logits=logits_cpu,
                        sampling_metadata=sampling_metadata,
                    )
                    logprobs = logprobs_output.logprobs_tensors
                sampler_output = SamplerOutput(output_token_ids, logprobs)
            return sampler_output

        # Spec decode
        else:
            # On-device rejection sampling
            if self.on_device_sampling:
                # Model already performed rejection sampling on-device
                # model_output_tensor: [batch_size, max_spec_len+1] with -1 padding
                # Parse to list[list[int]] format expected by vLLM
                output_token_ids = self._parse_rejection_sampling_output(
                    model_output_tensor,
                    self.input_batch.vocab_size,
                )
                sampler_output = SamplerOutput(output_token_ids, None)
                return sampler_output

            # CPU rejection sampling
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            logger.info("Using vLLM Sampler")
            bonus_logits = model_output_tensor[
                spec_decode_metadata.bonus_logits_indices.cpu()
            ]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = model_output_tensor[
                spec_decode_metadata.target_logits_indices.cpu()
            ]

            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )

            sampler_output.sampled_token_ids = output_token_ids

        return sampler_output

    def _parse_rejection_sampling_output(
        self,
        output_token_ids: torch.Tensor,
        vocab_size: int,
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID`(=-1) by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.

        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != -1) & (output_token_ids_np < vocab_size)
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs

    def _update_states_after_model_execute(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
    ) -> None:
        """Update the cached states after model execution.

        On GPU this handles MTP/EAGLE for hybrid models (linear attention
        state shifting). Neuron does not support hybrid models yet, so this
        is a no-op.
        """
        pass

    def _update_batch_state_with_samples(
        self,
        sampled_token_ids: list[list[int]],
        snapshot_req_ids: list[str] | None = None,
    ) -> None:
        """
        Update persistent batch state with newly sampled tokens.

        This is critical for maintaining token history across iterations.

        Args:
            sampled_token_ids: List of sampled token lists for each request,
                indexed by the batch order at the time of sampling.
            snapshot_req_ids: Optional snapshot of req_ids from the time of
                sampling. When provided, tokens are mapped to the current
                batch index via req_id lookup, which is safe across
                condense() reordering. When None, positional indexing is
                used (only safe when the batch hasn't been reordered).
        """
        # The async-output thread reaches this method through get_output, so it writes the
        # same InputBatch arrays and the same output_token_ids lists that the worker main
        # thread writes in _update_states. Hold the lock for the whole update. See
        # _batch_mutation_lock.
        with self._batch_mutation_lock():
            self._apply_sampled_tokens(sampled_token_ids, snapshot_req_ids)

    def _apply_sampled_tokens(
        self,
        sampled_token_ids: list[list[int]],
        snapshot_req_ids: list[str] | None = None,
    ) -> None:
        """Write the sampled tokens into the batch state. Call under the mutation lock.

        Split out of ``_update_batch_state_with_samples`` so the lock is one line at the entry
        point and this code keeps its original indentation.

        Args:
            sampled_token_ids: Sampled token lists, indexed by the batch order at sampling.
            snapshot_req_ids: Request ids at sampling time, or None for positional indexing.

        Example:
            >>> with runner._batch_mutation_lock():  # doctest: +SKIP
            ...     runner._apply_sampled_tokens([[5]], ["req-0"])
        """
        for sample_idx, tokens in enumerate(sampled_token_ids):
            if not tokens:
                continue

            # Resolve the current batch index for this request.
            if snapshot_req_ids is not None:
                if sample_idx >= len(snapshot_req_ids):
                    continue
                req_id = snapshot_req_ids[sample_idx]
                req_idx = self.input_batch.req_id_to_index.get(req_id)
                if req_idx is None:
                    # Request was removed from the batch (finished).
                    continue
            else:
                req_idx = sample_idx
                if req_idx >= len(self.input_batch.req_ids):
                    continue
                # Resolve req_id eagerly so debug logging and the post-loop
                # state update both see it. Sync mode relied on a late
                # binding previously.
                req_id = self.input_batch.req_ids[req_idx]

            # Update token counts using vLLM's numpy arrays
            if req_idx < len(self.input_batch.num_tokens_no_spec):
                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(tokens)

                # Update token_ids_cpu using vLLM's token_ids_cpu tensor
                if (
                    req_idx < self.input_batch.token_ids_cpu_tensor.shape[0]
                    and end_idx <= self.input_batch.token_ids_cpu_tensor.shape[1]
                ):
                    # Convert tokens to tensor and update the CPU tensor
                    token_tensor = torch.tensor(tokens, dtype=torch.int32)
                    self.input_batch.token_ids_cpu_tensor[
                        req_idx, start_idx:end_idx
                    ] = token_tensor

                # Update counters in numpy arrays
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # Update request state
            if req_id in self.requests:
                self.requests[req_id].output_token_ids.extend(tokens)

        logger.debug(
            "Updated batch state with %s total tokens",
            len([t for tokens in sampled_token_ids for t in tokens]),
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._draft_token_ids is None:
            if self.use_async_scheduling and self.speculative_config is not None:
                # Async mode: we must proactively signal "no drafts next step"
                # to the scheduler so ``_update_after_schedule``'s optimistic
                # placeholder is overwritten to ``[]``. Otherwise the next
                # scheduling step reserves ``1 + num_spec`` tokens per
                # request, and if we near max_model_len that causes the
                # scheduler to trim to a non-full shape which the target
                # NEFF wasn't warmed for.
                return DraftTokenIds(
                    self.input_batch.req_ids,
                    [[] for _ in self.input_batch.req_ids],
                )
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            if self.use_async_scheduling:
                # Async mode keeps the real draft tensors in the plugin
                # worker for cross-step device reuse. We hand the scheduler
                # placeholders just so it reserves the right shape; the
                # runner overwrites target input_ids and rejection metadata
                # from device futures (or the per-request transition cache)
                # before anything reads them.
                num_spec = self.drafter.num_speculative_tokens
                draft_token_ids = [[-1] * num_spec for _ in req_ids]
            else:
                draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _propose_draft_token_ids(
        self,
        scheduler_output: Any,
        sampled_token_ids: torch.Tensor | list[list[int]],
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        logits_indices: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        assert isinstance(self.drafter, (EagleProposer, DFlashProposer))

        num_reqs = self.input_batch.num_reqs

        if num_reqs == 0:
            logger.info("No requests to process, returning empty tensor")
            return torch.empty(
                (0, self.drafter.num_speculative_tokens), dtype=torch.int32
            )

        # Compute last_token_indices: one index per request pointing to the
        # last real token in the padded input tensor. Keep on-device when
        # possible to avoid device→host sync between target and draft NEFFs.
        if spec_decode_metadata is not None:
            last_token_indices = spec_decode_metadata.bonus_logits_indices.to(
                torch.long
            )
        elif logits_indices is not None:
            last_token_indices = logits_indices.to(torch.long)
        else:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_per_req = [
                scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
            ]
            cu_num_tokens = list(
                torch.tensor(num_scheduled_tokens_per_req).cumsum(0).tolist()
            )
            last_token_indices = torch.tensor(
                [c - 1 for c in cu_num_tokens], dtype=torch.long
            )

        logger.debug(
            "Draft model indices: last_token_indices=%s",
            last_token_indices,
        )

        # Build raw_sampled_token_ids tensor for on-device extraction.
        # For on-device sampling + spec decode, raw_sampled_token_ids is
        # the [bs, max_spec_len+1] tensor passed from sample_tokens().
        # For prefill (no spec decode), build a [bs, 1] tensor from the list.
        if raw_sampled_token_ids is None:
            # Prefill case or CPU sampling: build from list
            next_token_ids_cpu, _ = extract_next_token_ids(
                sampled_token_ids, self.vocab_size
            )
            raw_sampled_token_ids = next_token_ids_cpu.unsqueeze(1)
        elif raw_sampled_token_ids.ndim == 1:
            # Async prefill / non-spec decode: 1D device tensor [bs] of
            # sampled tokens. Unsqueeze to [bs, 1] to match the extract_
            # accepted_tokens contract inside the draft NEFF.
            raw_sampled_token_ids = raw_sampled_token_ids.unsqueeze(1)

        # EAGLE3: aux_hidden_states from the target NEFF is already a
        # single concatenated [T, 3*hidden_size] device tensor (the cat is
        # done on the target side, inside its compile boundary). Passing
        # it through as-is avoids any Python-side op between the target
        # NEFF and the draft NEFF.
        assert aux_hidden_states is not None

        async_spec_kwargs = {}
        if self._model_is_async_spec_decoded() and spec_decode_metadata is not None:
            async_spec_kwargs = self._build_async_spec_kwargs(
                spec_decode_metadata,
                num_total_tokens=input_ids.shape[0],
            )

        # Per-request count of drafts proposed in the current step, as a
        # [num_reqs] device tensor. DFlash needs this to derive current rejected
        # drafts for its post-verify sequence lengths. In async mode the
        # prev_num_draft_tokens kwarg belongs to the previous step's correction,
        # so keep the current count separate.
        if spec_decode_metadata is not None:
            cur_num_draft_tokens = torch.tensor(
                spec_decode_metadata.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.is_dflash_spec:
                async_spec_kwargs["cur_num_draft_tokens"] = cur_num_draft_tokens
            elif "prev_num_draft_tokens" not in async_spec_kwargs:
                async_spec_kwargs["prev_num_draft_tokens"] = cur_num_draft_tokens

        if self._draft_tensor_capture is not None:
            self._draft_tensor_capture.setup_tensor_capture()

        draft_token_ids, drafts_only = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=aux_hidden_states,
            last_token_indices=last_token_indices,
            attn_metadata=attn_metadata,
            raw_sampled_token_ids=raw_sampled_token_ids,
            **async_spec_kwargs,
        )
        if self._draft_tensor_capture is not None:
            is_prefill = input_ids.shape[0] > num_reqs * self._decode_token_threshold()
            self._draft_tensor_capture.save_tensor_captures(
                positions=positions,
                is_prefill=is_prefill,
                req_ids=list(self.input_batch.req_ids),
            )

        # ``draft_token_ids`` is ``[bs, 1+num_spec]`` (bonus + drafts), used
        # by the async path to feed the next target NEFF's input_ids in one
        # contiguous flatten. ``drafts_only`` is ``[bs, num_spec]`` (drafts
        # only, contiguous), used by the rejection sampler. In sync mode,
        # only ``drafts_only`` is needed by the scheduler.
        if not self.use_async_scheduling:
            return drafts_only.cpu()

        # Stash the drafts-only tensor for the next step's runner to feed
        # into ``spec_decode_metadata.draft_token_ids`` without slicing the
        # bonus column off (which produces a non-contiguous view that
        # ``.contiguous()`` cannot resolve on Neuron device).
        self._futures_drafts_only = drafts_only
        return draft_token_ids

    # ------------------------------------------------------------------------
    # KV CACHE INTERFACE
    # ------------------------------------------------------------------------
    def _initialize_stacked_kv_cache(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """Allocate the GK per-buffer KV cache, one base-0 buffer for each HMA cache group.
        The buffer count is the number of distinct buffer identities, not ``len(tensors)``."""
        tensors = kv_cache_config.kv_cache_tensors
        assert len(tensors) > 0, "The GK KV path needs at least one KV cache tensor"

        # One representative spec (all groups share dtype/geometry under HMA).
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        page = spec.page_size_bytes  # ALREADY includes the K+V factor of 2
        num_blocks = kv_cache_config.num_blocks  # blocks in ONE buffer

        sizes = {t.size for t in tensors}
        assert len(sizes) == 1, f"The GK KV path needs a uniform size; got {sizes}"

        # Reject the block-outermost layout: it strides a layer's bytes inside each block, which a
        # contiguous per-buffer allocation cannot address.
        for t in tensors:
            layer_stride = getattr(t, "layer_stride", None)
            block_stride = getattr(t, "block_stride", 0)
            layer_outermost = (
                block_stride == 0
                if layer_stride is None
                else layer_stride >= block_stride
            )
            assert layer_outermost, (
                f"The GK KV path needs a layer-outermost KV layout; got layer_stride "
                f"{layer_stride} and block_stride {block_stride}"
            )

        # One buffer for each distinct identity, so a paired even-SWA and odd-full layer share one.
        buffer_of: dict[tuple, int] = {}
        buffer_of_layer: dict[str, int] = {}
        for tensor_index, t in enumerate(tensors):
            names = _kv_tensor_layer_names(t)
            for layer_index, name in enumerate(names):
                key = _kv_buffer_identity(t, tensor_index, layer_index)
                buffer_of_layer[name] = buffer_of.setdefault(key, len(buffer_of))
            # With a layer stride a tensor spans one buffer per named layer, else it IS the buffer.
            spanned = len(names) if getattr(t, "layer_stride", None) is not None else 1
            assert t.size == num_blocks * page * spanned, (
                f"KV tensor size {t.size} != num_blocks {num_blocks} x page {page} x "
                f"{spanned} buffer(s); the scheduler block count is authoritative"
            )
        # Aliased cache groups each report the whole pool, so take the largest size and not the sum.
        aliased = getattr(tensors[0], "layer_stride", None) is not None
        budget = max(sizes) if aliased else sum(t.size for t in tensors)

        kv = spec.num_kv_heads
        block = spec.block_size
        d = spec.head_size
        group_size = len(buffer_of)

        packed = self._kv_cache_is_fp8_packed(spec.dtype)
        logger.info(
            "[GK] KV cache: packed=%s group_size=%d num_blocks=%d kv=%d block=%d d=%d",
            packed,
            group_size,
            num_blocks,
            kv,
            block,
            d,
        )

        # One [2, nb, kv, block, d] tensor per HMA group, each its own base-0
        # allocation. Per-layer shape lets DI/NIXL register each group's storage.
        group_bufs = [
            torch.zeros(
                2, num_blocks, kv, block, d, dtype=spec.dtype, device=self.device
            )
            for _ in range(group_size)
        ]
        total_bytes = sum(gb.numel() * gb.element_size() for gb in group_bufs)
        assert total_bytes == budget, (
            f"GK per-buffer KV bytes {total_bytes} != planner budget {budget}"
        )
        kv_caches: dict[str, torch.Tensor] = {}
        K_list, V_list = [], []
        for gb in group_bufs:
            # Packed FP8: view the K half [nb, kv, block, d] as
            # [nb, kv, block//2, d, 2] (same numel); V stays 4-D.
            k = gb[0].view(num_blocks, kv, block // 2, d, 2) if packed else gb[0]
            K_list.append(k)
            V_list.append(gb[1])
        for layer_name, i in buffer_of_layer.items():
            kv_caches[layer_name] = [K_list[i], V_list[i]]
            self._kv_cache_full_tensors[layer_name] = group_bufs[i]
        self.model.bind_kv_cache(kv_caches)
        self.model.set_kv_cache_buffers(K_list, V_list)

        if has_kv_transfer_group():
            kv_caches_view = self.get_kv_cache_view_for_connector_registration(
                kv_caches
            )
            get_kv_transfer_group().register_kv_caches(kv_caches_view)

        return kv_caches

    def _kv_cache_is_fp8_packed(self, cache_dtype: torch.dtype) -> bool:
        """Whether the K cache should use the swizzled packed FP8 layout.

        Two conditions must both hold:

        1. The packed layout is explicitly opted in via
           ``neuron_config.fp8_packed_kv``. It is off by default: only models
           wired for the packed layout (packed reads in their decode /
           segmented-prefill attention paths, packed writes out-of-kernel in
           forward_prefill, and packed-aware bind_kv_cache) can consume the
           5-D K cache, so other models keep the standard 4-D FP8 cache.
        2. The cache dtype is float8_e4m3fn. The packed layout reinterprets two
           consecutive FP8 tokens as one BF16 element so the attention kernels
           can DMA-transpose the K load instead of using the slower PE-transpose
           path. The BF16-reinterpret relies on the e4m3fn bit layout, so
           float8_e5m2 is excluded.
        """
        if not self.neuron_config.fp8_packed_kv:
            return False
        return cache_dtype == torch.float8_e4m3fn

    def _is_non_packed_fp8_kv(self) -> bool:
        """Whether the KV cache is FP8 in a layout segmented prefill cannot read.

        ``attention_segmented_cte`` can only read an FP8 K cache in the packed
        (5-D swizzled) layout; a standard 4-D FP8 cache makes its
        ``load_kv_cache`` assert (see
        :func:`~vllm_neuron.utils.dtype_utils.validate_fp8_segmented_supported`).
        Used to decide whether segmented prefill may be defaulted on, so such a
        config keeps full prefill instead of failing at warmup.

        Resolved from server config only -- ``cache_config.cache_dtype`` and
        ``neuron_config.fp8_packed_kv`` -- because this runs in ``__init__``,
        before ``bind_kv_cache`` sets each layer's ``fp8_packed``.

        Only the dtype *string* is inspected, deliberately: this runs well
        before KV-cache allocation validates ``cache_dtype`` for real, so an
        unrecognized value must not raise here. Resolving it through
        ``kv_cache_dtype_str_to_dtype`` would turn a clear downstream error
        into a confusing one during bucket setup. ``"auto"`` is not FP8 (it
        follows the model dtype), so it correctly falls through to False.

        Returns:
            True when the KV cache is FP8 and the packed layout is not enabled.

        Example:
            >>> # --kv-cache-dtype fp8_e4m3 without fp8_packed_kv
            >>> runner._is_non_packed_fp8_kv()
            True
        """
        cache_dtype = self.vllm_config.cache_config.cache_dtype
        kv_is_fp8 = isinstance(cache_dtype, str) and cache_dtype.startswith("fp8")
        return kv_is_fp8 and not self.neuron_config.fp8_packed_kv

    @staticmethod
    def _k_cache_alloc_shape(
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_size: int,
        packed: bool,
    ) -> tuple[int, ...]:
        """K cache allocation shape. When ``packed``, K is swizzled to
        ``[num_blocks, num_kv_heads, block_size // 2, head_size, 2]`` (two tokens
        per BF16-width slot); otherwise the standard
        ``[num_blocks, num_kv_heads, block_size, head_size]``. The packed shape
        has the same element count as the unpacked one
        (``block_size // 2 * 2 == block_size``), so page_size / num_blocks
        accounting is unchanged. V always uses the unpacked shape.
        """
        if packed:
            return (num_blocks, num_kv_heads, block_size // 2, head_size, 2)
        return (num_blocks, num_kv_heads, block_size, head_size)

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initialize KV cache with the given configuration.

        Args:
            kv_cache_config: KV cache configuration object
        """
        logger.debug("Initializing KV cache in NeuronModelRunner")

        logger.debug("kv_cache_config = %s", kv_cache_config)

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        # TODO do we need to initialize attn_backend?

        # Metadata Builder Initialization: It sets up attention metadata builders
        # for each attention group. These builders are responsible for creating
        # the attention metadata that kernels need during forward passes.

        # Extract block sizes for each KV cache group (one per group for InputBatch)
        block_sizes = [
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        ]
        max_num_blocks_per_req = [
            group.kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config, self.max_model_len
            )
            for group in kv_cache_config.kv_cache_groups
        ]
        logger.info(
            "KV cache block_size resolved: cache_config=%s, per_group=%s",
            self.vllm_config.cache_config.block_size,
            block_sizes,
        )

        # Initialize InputBatch with block_sizes list matching the number of KV cache groups
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            device=self.device,
            vocab_size=self.vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
            max_num_blocks_per_req=max_num_blocks_per_req,
            logitsprocs=None,
            logitsprocs_need_output_token_ids=False,
            is_pooling_model=self.is_pooling_model,
        )
        # A new InputBatch brings new block-table buffers, so drop the staging rings and the
        # previous batch's cached results. Both caches compare values, so a stale entry would
        # still give correct numbers, but a new batch ends the provenance of every device
        # tensor the old buffers produced. Keep ``_constant_tensors``: each entry is a pure
        # function of its key and none reads the InputBatch.
        if self._block_table_groups:
            self._block_table_groups.clear()
        if self._swa_groups:
            self._swa_groups.clear()

        # >>> GK: contiguous group_size-deep stacked KV cache <<<
        # The GPT-OSS decode giga-kernel indexes its cache axis by buffer
        # (group_size), not per layer, and wants K_cache/V_cache each contiguous.
        # Gate on the model capability flag so every non-GK model keeps the stock
        # per-tensor allocation below.
        if getattr(self.model, "use_stacked_kv_cache", False):
            return self._initialize_stacked_kv_cache(kv_cache_config)

        # Initialize the KV Cache tensors
        tensors = kv_cache_config.kv_cache_tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        _layer_slices: dict[tuple, torch.Tensor] = {}
        # A no-KV engine (e.g. an EPD vision encoder) gets an empty tensor list:
        # skip allocation and fall through so bind_kv_cache runs with empty caches.
        if tensors:
            num_blocks = kv_cache_config.num_blocks

            for tensor_index, tensor in enumerate(tensors):
                # With ``layer_stride`` one KVCacheTensor spans several layers back to back, so
                # each has its own offset. Without it, every layer in it shares one buffer.
                _has_layer_stride = getattr(tensor, "layer_stride", None) is not None
                if _has_layer_stride:
                    span = num_blocks * tensor.block_stride
                    assert tensor.layer_stride == span, (
                        "Neuron KV init needs each layer contiguous (layer-outermost); "
                        f"got layer_stride={tensor.layer_stride}, "
                        f"num_blocks*block_stride={span}"
                    )
                    _layer_names = list(tensor.layers)
                else:
                    span = tensor.size
                    _layer_names = _kv_tensor_layer_names(tensor)
                for layer_index, layer_name in enumerate(_layer_names):
                    # Key on the BUFFER IDENTITY, never on a raw byte offset. Without
                    # ``layer_stride`` every tensor reports ``offset == 0``, which aliases layers.
                    key = _kv_buffer_identity(tensor, tensor_index, layer_index)
                    raw_tensor = _layer_slices.get(key)
                    if raw_tensor is None:
                        # Allocate per layer-start instead of slicing one giant
                        # backing: views into a multi-GiB shared storage carry
                        # element offsets past 2**32, which the lowering path
                        # can truncate (silent cross-layer KV corruption).
                        raw_tensor = torch.zeros(
                            span, dtype=torch.int8, device=self.device
                        )
                        _layer_slices[key] = raw_tensor
                    kv_cache_raw_tensors[layer_name] = raw_tensor

        # Cache raw.view(dtype) per (raw_tensor, dtype) so layers pooled onto one
        # buffer reuse ONE reinterpret -> their K/V share a single ._base.
        _dtype_views: dict[tuple[int, torch.dtype], torch.Tensor] = {}

        def _shared_dtype_view(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            key = (id(raw), dtype)
            view = _dtype_views.get(key)
            if view is None:
                view = raw.view(dtype)
                _dtype_views[key] = view
            return view

        # Reuse exact slice objects for pooled layers. Re-slicing shared storage
        # per layer gives Dynamo overlapping inputs that AOTAutograd merges into
        # a whole-buffer synthetic base, widening slot updates to full-buffer writes.
        _kv_slices: dict[tuple[int, torch.dtype, tuple], list[torch.Tensor]] = {}

        def _shared_kv_slices(
            raw: torch.Tensor, dtype: torch.dtype, kv_shape: tuple
        ) -> list[torch.Tensor]:
            """[k, v, typed] for a pooled buffer, shared across its layers.

            ``v`` is None for a key-only (1, ...) allocation, which has no V half.
            """
            key = (id(raw), dtype, tuple(kv_shape))
            slices = _kv_slices.get(key)
            if slices is None:
                typed = _shared_dtype_view(raw, dtype).view(kv_shape)
                v = typed[1] if kv_shape[0] > 1 else None
                slices = [typed[0], v, typed]
                _kv_slices[key] = slices
            return slices

        _packed_k_views: dict[tuple[int, tuple], torch.Tensor] = {}

        def _shared_packed_k(k_cache: torch.Tensor, shape: tuple) -> torch.Tensor:
            """Packed-FP8 K reshape, memoized for the same reason as the slices."""
            key = (id(k_cache), tuple(shape))
            view = _packed_k_views.get(key)
            if view is None:
                view = k_cache.view(shape)
                _packed_k_views[key] = view
            return view

        # Build KV caches per group
        kv_caches = {}
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec

            # This is the case that all layers have the same kv_hidden_size.
            if isinstance(kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)):
                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0

                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

                    block_size = kv_cache_spec.block_size
                    num_kv_heads = kv_cache_spec.num_kv_heads
                    head_size = kv_cache_spec.head_size

                    # Packed FP8 K cache: store K swizzled as
                    # [num_blocks, num_kv_heads, block_size // 2, head_size, 2]
                    # so the decode kernel can bf16-reinterpret + DMA-transpose.
                    # V is never packed. Same byte footprint as the unpacked
                    # cache (block_size // 2 * 2 == block_size), so page_size /
                    # num_blocks accounting is unchanged. Only the K view shape
                    # differs.
                    k_is_packed = self._kv_cache_is_fp8_packed(kv_cache_spec.dtype)
                    k_cache_shape = self._k_cache_alloc_shape(
                        num_blocks, num_kv_heads, block_size, head_size, k_is_packed
                    )

                    # MLAAttentionSpec is a FullAttentionSpec subclass, so it
                    # reaches this branch; its page_size_bytes budgets ONE vector
                    # per token, so it must be allocated as (1, ...).
                    num_caches = 1 if isinstance(kv_cache_spec, MLAAttentionSpec) else 2
                    kv_shape = (
                        num_caches,
                        num_blocks,
                        num_kv_heads,
                        block_size,
                        head_size,
                    )
                    # K and V are backed by the same (2, ...) storage; only the K
                    # slice is reshaped to the packed layout when FP8 packing is
                    # enabled (same numel, different view).
                    k_cache, v_cache, typed_tensor = _shared_kv_slices(
                        raw_tensor, kv_cache_spec.dtype, kv_shape
                    )
                    if k_is_packed:
                        k_cache = _shared_packed_k(k_cache, k_cache_shape)
                    # Key-only caches bind a 1-element list: there is no V half.
                    kv_caches[layer_name] = (
                        [k_cache] if num_caches == 1 else [k_cache, v_cache]
                    )
                    # The full (num_caches, ...) tensor -- k_cache/v_cache are
                    # views into it -- is what DI connector registration needs.
                    self._kv_cache_full_tensors[layer_name] = typed_tensor

            # Spec decoding specifically use this because hidden_size of different layers
            # (draft and target model) are different.
            # https://github.com/vllm-project/vllm/pull/25101
            elif isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    kv_cache_layer_spec = kv_cache_spec.kv_cache_specs[layer_name]
                    assert raw_tensor.numel() % kv_cache_layer_spec.page_size_bytes == 0

                    num_blocks = (
                        raw_tensor.numel() // kv_cache_layer_spec.page_size_bytes
                    )

                    block_size = kv_cache_layer_spec.block_size
                    num_kv_heads = kv_cache_layer_spec.num_kv_heads
                    head_size = kv_cache_layer_spec.head_size

                    if isinstance(
                        kv_cache_layer_spec, (FullAttentionSpec, SlidingWindowSpec)
                    ):
                        # Per-layer spec: this group holds caches of DIFFERENT
                        # widths (that is why vLLM kept them out of a merged
                        # single-spec group), so shape and cache count must both
                        # come from the layer's own spec, never the group's.
                        num_caches = (
                            1
                            if isinstance(kv_cache_layer_spec, MLAAttentionSpec)
                            else 2
                        )
                        kv_shape = (
                            num_caches,
                            num_blocks,
                            num_kv_heads,
                            block_size,
                            head_size,
                        )
                    else:
                        raise NotImplementedError(
                            f"Unsupported Attention spec type: {type(kv_cache_layer_spec)}"
                        )

                    k_cache, v_cache, typed_tensor = _shared_kv_slices(
                        raw_tensor, kv_cache_layer_spec.dtype, kv_shape
                    )
                    # Key-only caches bind a 1-element list: there is no V half.
                    kv_caches[layer_name] = (
                        [k_cache] if num_caches == 1 else [k_cache, v_cache]
                    )
                    self._kv_cache_full_tensors[layer_name] = typed_tensor

            else:
                raise NotImplementedError(
                    f"Unsupported Attention spec type: {type(kv_cache_spec)}"
                )

        # This binds the cache tensors to the model
        self.model.bind_kv_cache(kv_caches)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, (EagleProposer, DFlashProposer))
            assert self.drafter.model is not None
            self.drafter.model.bind_kv_cache(kv_caches)
            self.drafter.validate_same_kv_cache_group(kv_cache_config)
            if isinstance(self.drafter, (DFlashProposer, MTPProposer)):
                self.drafter.alias_target_weights(target_model=self.model)

        if has_kv_transfer_group():
            kv_caches_view = self.get_kv_cache_view_for_connector_registration(
                kv_caches
            )
            get_kv_transfer_group().register_kv_caches(kv_caches_view)

        return kv_caches

    def get_kv_cache_view_for_connector_registration(self, kv_caches):
        """
        Get KV cache view for connector registration.

        Returns the full (2, num_blocks, num_kv_heads, block_size, head_size)
        tensors that share memory with the [k_cache, v_cache] slices used by
        the model. `initialize_kv_cache` populates `_kv_cache_full_tensors`
        for every backend, so the fallback to `kv_caches` is only exercised
        when this is called before initialization.
        """
        if self._kv_cache_full_tensors:
            return self._kv_cache_full_tensors
        return kv_caches

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Get KV cache specifications for all model layers.

        Returns:
            Dictionary mapping layer names to their KV cache specifications
        """
        all_kv_cache_specs = {}
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config
        )

        target_kv_spec = self.model.get_kv_spec()
        for layer in target_kv_spec.layers:
            layer_name = layer.name
            # Use SlidingWindowSpec for SWA layers so HMA can create separate
            # KV cache groups. When --no-disable-hybrid-kv-cache-manager is set,
            # this enables block clipping in the NiXL connector.
            if layer.key_only:
                # Budgets ONE vector per token instead of FullAttentionSpec's 2x
                # for a K/V pair, and keeps caches of different widths (e.g. MLA
                # 576 vs indexer 128) in separate groups. See KeyOnlyAttentionSpec.
                spec = KeyOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                    attention_chunk_size=layer.chunk_size,
                )
            elif layer.sliding_window_size is None:
                spec = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                    attention_chunk_size=layer.chunk_size,
                )
            else:
                spec = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                )
            all_kv_cache_specs[layer_name] = spec

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, (EagleProposer, DFlashProposer))
            assert self.drafter.model is not None

            drafter_kv_spec = self.drafter.model.get_kv_spec()
            for layer in drafter_kv_spec.layers:
                layer_name = layer.name
                all_kv_cache_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                    attention_chunk_size=layer.chunk_size,
                )

        return all_kv_cache_specs

    def _bookkeeping_sync(
        self,
    ) -> tuple[
        list[str],
        dict[str, int],
    ]:
        """
        Bookkeeping a snapshot of the states of some fields in InputBatch before they
        are overwritten by vllm scheduler to schedule future step asynchronously. We use
        the snapshots for preparing model input for the current step.

        This is similar to `_bookkeeping_sync` in gpu_model_runner.
        """
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        # For async scheduling + spec decode, save prev_req_id_to_index so
        # the next step's _update_states can reconcile scheduler-provided
        # num_computed_tokens (which assumes all drafts accepted) against
        # actual accepted counts via _get_valid_sampled_token_count.
        if self.use_async_scheduling and self.speculative_config is not None:
            self.input_batch.prev_req_id_to_index = dict(req_id_to_index_output_copy)

        return req_ids_output_copy, req_id_to_index_output_copy

    # ------------------------------------------------------------------------
    # KV CACHE DEBUG UTILITIES
    # ------------------------------------------------------------------------
    def get_kv_caches(self) -> dict[str, bytes]:
        """Extract KV cache tensors from model layers, serialized as bytes.

        Example:
            >>> kv = model_runner.get_kv_caches()
        """
        import io

        if self.model is None:
            return {}

        kv_caches = {}
        # Unwrap torch.compile to reach actual model
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        layers = getattr(getattr(model, "model", model), "layers", [])

        for i, layer in enumerate(layers):
            layer_name = f"layers.{i}.self_attn"
            attn = getattr(layer, "self_attn", None)
            if attn and getattr(attn, "k_cache", None) is not None:
                # Serialize tensors to bytes to avoid pickle issues
                k_cpu = attn.k_cache.detach().cpu()
                v_cpu = attn.v_cache.detach().cpu()
                buf = io.BytesIO()
                torch.save({"k": k_cpu, "v": v_cpu}, buf)
                kv_caches[layer_name] = buf.getvalue()

        return kv_caches

    def get_block_table_info(self) -> dict:
        """Extract block table info for KV cache reconstruction.

        Returns per-group block tables and layer-to-group mapping.
        Non-hybrid models have a single group containing all layers.

        Example:
            >>> info = model_runner.get_block_table_info()
        """
        if self.input_batch is None:
            return {}

        groups = []
        for bt in self.input_batch.block_table.block_tables:
            groups.append(
                {
                    "block_table": bt.block_table.cpu.numpy().tolist(),
                    "num_blocks_per_row": bt.num_blocks_per_row.tolist(),
                    "block_size": bt.block_size,
                }
            )

        layer_to_group = {}
        if self.kv_cache_config:
            for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    layer_to_group[layer_name] = group_idx

        return {
            "groups": groups,
            "layer_to_group": layer_to_group,
            "seq_lens": self.input_batch.num_tokens_no_spec.tolist(),
        }

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig info as serializable dict for KV reconstruction.

        Returns:
            Dict with:
            - groups: List of per-group dicts, each containing the group's
              layer_names and all fields from its KVCacheSpec (block_size,
              num_kv_heads, head_size, dtype, sliding_window, etc.)
              plus 'spec_type' indicating the spec class name.
            - tp_size: Tensor parallel size
            - total_num_kv_heads: Total KV heads across all TP ranks
            - dcp_size: Decode context parallel size (must be 1 for now)
            - pcp_size: Prefill context parallel size (must be 1 for now)

        Example:
            >>> cfg = model_runner.get_kv_cache_config()
        """
        from dataclasses import fields as dc_fields

        if self.kv_cache_config is None:
            return {}

        groups = []
        for g in self.kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            group_info = {
                "layer_names": g.layer_names,
                "spec_type": type(spec).__name__,
            }
            for f in dc_fields(spec):
                val = getattr(spec, f.name)
                if isinstance(val, torch.dtype):
                    val = str(val)
                group_info[f.name] = val
            groups.append(group_info)

        parallel_config = self.vllm_config.parallel_config
        model_config = self.vllm_config.model_config

        return {
            "num_blocks": self.kv_cache_config.num_blocks,
            "groups": groups,
            "tp_size": parallel_config.tensor_parallel_size,
            "total_num_kv_heads": model_config.get_total_num_kv_heads(),
            "dcp_size": parallel_config.decode_context_parallel_size,
            "pcp_size": parallel_config.prefill_context_parallel_size,
        }

    def enable_kv_snapshot(self) -> None:
        """Enable block-table snapshotting on subsequent forward passes.

        Call before the ``generate()`` whose block tables you want to read:
        each forward pass then keeps a host-side copy of the tables that
        survives request freeing. Zero overhead until enabled.

        Example:
            >>> model_runner.enable_kv_snapshot()
        """
        self._kv_snapshot_enabled = True

    def get_block_tables(self) -> list[bytes]:
        """Return block tables for all KV cache groups as serialized bytes.

        Returns the current block table if requests are active, otherwise
        returns the last snapshot taken during the most recent forward pass.

        On first call, enables block table snapshotting for subsequent
        forward passes (zero overhead until first call).

        Example:
            >>> tables = model_runner.get_block_tables()
        """
        import io

        # Enable snapshotting for future forward passes
        self._kv_snapshot_enabled = True

        if self.input_batch is None:
            if self._block_table_snapshot is None:
                return []
            result = []
            for bt_cpu in self._block_table_snapshot:
                buf = io.BytesIO()
                torch.save(bt_cpu, buf)
                result.append(buf.getvalue())
            return result

        # Check if any requests are still active
        has_active = False
        for bt in self.input_batch.block_table.block_tables:
            if hasattr(bt, "num_blocks_per_row") and any(
                n > 0 for n in bt.num_blocks_per_row
            ):
                has_active = True
                break

        if not has_active:
            # Blocks freed — use snapshot from last forward pass
            if self._block_table_snapshot is not None:
                result = []
                for bt_cpu in self._block_table_snapshot:
                    buf = io.BytesIO()
                    torch.save(bt_cpu, buf)
                    result.append(buf.getvalue())
                return result

        result = []
        for bt in self.input_batch.block_table.block_tables:
            buf = io.BytesIO()
            torch.save(_remap_null_block_to_sentinel(bt.block_table.cpu), buf)
            result.append(buf.getvalue())
        return result

    def clear_kv_snapshot(self) -> None:
        """Release block table snapshot memory and disable snapshotting.

        Call after KV cache analysis is complete to free memory and
        restore zero-overhead production behavior.

        Example:
            >>> model_runner.clear_kv_snapshot()
        """
        self._kv_snapshot_enabled = False
        self._block_table_snapshot = None

    def _snapshot_encoder_entries(self, scheduler_output) -> None:
        """Capture encoder cache entries used by the current step.

        Accumulates entries across multiple prefill steps (multi-step
        encoding when the items are larger than the max encoder bucket). Each
        step adds newly-encoded entries without discarding previous ones,
        so get_encoder_cache() returns all items for the full request.
        """
        if self._encoder_cache_snapshot is None:
            self._encoder_cache_snapshot = {}
        # New requests scheduled this step
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_state = self.requests.get(new_req_data.req_id)
            if req_state is None or not req_state.mm_features:
                continue
            for mm_feature in req_state.mm_features:
                mm_hash = mm_feature.identifier
                if self.encoder_cache.contains(mm_hash):
                    self._encoder_cache_snapshot[mm_hash] = self.encoder_cache.get(
                        mm_hash
                    )
        # Requests with scheduled encoder inputs (covers multi-step
        # encoding where the remaining items are encoded in later steps).
        for req_id in scheduler_output.scheduled_encoder_inputs:
            req_state = self.requests.get(req_id)
            if req_state is None or not req_state.mm_features:
                continue
            for mm_feature in req_state.mm_features:
                mm_hash = mm_feature.identifier
                if self.encoder_cache.contains(mm_hash):
                    self._encoder_cache_snapshot[mm_hash] = self.encoder_cache.get(
                        mm_hash
                    )

    def get_encoder_cache(self) -> dict[str, bytes]:
        """Return encoder cache contents (multimodal embeddings) as serialized bytes.

        Returns the snapshot if snapshotting is enabled (contains all
        entries used by the last forward pass, including cache hits),
        otherwise falls back to the live encoder_cache.

        Call enable_encoder_cache_snapshot() before generate() to activate
        per-request snapshotting.

        Returns:
            Dict[mm_hash, bytes] — serialized torch tensors.

        Example:
            >>> model_runner.enable_encoder_cache_snapshot()
            >>> # ... generate ...
            >>> enc = model_runner.get_encoder_cache()
        """
        import io

        if self._encoder_cache_snapshot is not None:
            source = self._encoder_cache_snapshot
        else:
            source = self.encoder_cache

        result = {}
        for mm_hash, tensor in source.items():
            if tensor is None:
                continue
            buf = io.BytesIO()
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor.detach().cpu(), buf)
            else:
                torch.save(tensor, buf)
            result[mm_hash] = buf.getvalue()
        return result

    def enable_encoder_cache_snapshot(self) -> None:
        """Enable encoder cache snapshotting for subsequent forward passes.

        Once enabled, each prefill step captures all encoder cache entries
        used by that step's requests (both fresh encodings and cache hits).

        Example:
            >>> model_runner.enable_encoder_cache_snapshot()
        """
        self._encoder_cache_snapshot_enabled = True
        self._encoder_cache_snapshot = None

    def clear_encoder_cache_snapshot(self) -> None:
        """Release encoder cache snapshot memory and disable snapshotting.

        Example:
            >>> model_runner.clear_encoder_cache_snapshot()
        """
        self._encoder_cache_snapshot_enabled = False
        self._encoder_cache_snapshot = None

    @contextmanager
    def _kv_cache_dump_context(self):
        """Context manager that dumps KV cache before and after a block.

        Enabled by setting the environment variable VLLM_NEURON_DUMP_KV_CACHE=1.
        When disabled this is a zero-cost no-op.
        """
        enabled = os.environ.get("VLLM_NEURON_DUMP_KV_CACHE", "0") == "1"
        if not enabled:
            yield
            return

        self._dump_kv_cache_impl("before_forward")
        try:
            yield
        finally:
            self._dump_kv_cache_impl("after_forward")

    def _dump_kv_cache_impl(self, tag: str) -> None:
        """Save KV cache tensors to disk with timestamp for debugging.

        Args:
            tag: A label for the save point (e.g. 'begin', 'end').
        """
        import datetime
        import io
        import pathlib

        rank = self.rank_tensor.item()
        if rank != 0:
            return

        save_dir = pathlib.Path("/tmp/kv_cache_dumps")
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        kv_caches = self.get_kv_caches()
        if not kv_caches:
            logger.debug("dump_kv_cache_impl(%s): no KV caches to save", tag)
            return

        save_path = save_dir / f"kv_cache_rank{rank}_{ts}_{tag}.pt"

        tensors = {}
        for layer_name, raw_bytes in kv_caches.items():
            buf = io.BytesIO(raw_bytes)
            tensors[layer_name] = torch.load(buf, weights_only=True)

        torch.save(tensors, str(save_path))

    # ------------------------------------------------------------------------
    # TENSOR CAPTURE
    # ------------------------------------------------------------------------
    def enable_capture(self) -> None:
        """Enable tensor capture writing to disk."""
        if self._target_tensor_capture is not None:
            self._target_tensor_capture.enable()
        if self._encoder_tensor_capture is not None:
            self._encoder_tensor_capture.enable()
        if self._draft_tensor_capture is not None:
            self._draft_tensor_capture.enable()

    def disable_capture(self) -> None:
        """Disable tensor capture writing to disk."""
        if self._target_tensor_capture is not None:
            self._target_tensor_capture.disable()
        if self._encoder_tensor_capture is not None:
            self._encoder_tensor_capture.disable()
        if self._draft_tensor_capture is not None:
            self._draft_tensor_capture.disable()

    # ------------------------------------------------------------------------
    # OTHER VLLM INTERFACE METHODS
    # ------------------------------------------------------------------------
    def profile_run(self) -> None:
        """Profile the model (no-op for now)."""
        logger.debug("Profile run called - no-op implementation")

    def capture_model(self) -> None:
        """Capture the model for optimization (no-op for now)."""
        logger.debug("Capture model called - no-op implementation")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get the tasks supported by this model runner.

        For pooling models, reports the pooler's tasks (e.g. ("embed",)) so the
        engine/API layer mounts the matching endpoints (e.g. /v1/embeddings).
        For a model that supports transcription, adds the "transcription" task,
        so the API layer mounts /v1/audio/transcriptions. Otherwise reports
        generation.

        This mirrors the upstream GPUModelRunner task logic.

        Returns:
            Tuple of supported tasks
        """
        if self.vllm_config.model_config.runner_type == "pooling":
            return tuple(self.get_model().pooler.get_supported_tasks())

        tasks: list[SupportedTask] = ["generate"]

        model = self.get_model()
        if supports_transcription(model):
            # A transcription-only model reports the transcription task only.
            if getattr(model, "supports_transcription_only", False):
                return ("transcription",)
            tasks.append("transcription")

        return tuple(tasks)

    def ensure_kv_transfer_shutdown(self) -> None:
        pass
