# SPDX-License-Identifier: Apache-2.0
"""
NeuronModelRunner - Model runner implementation for vLLM integration with Neuron

This provides the execution engine that processes SchedulerOutput and returns
ModelRunnerOutput with proper request tracking, output reordering, and persistent
state management.
"""

import logging
import math
import os
import threading
import time
from copy import copy, deepcopy
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from vllm import ModelRegistry
from vllm.v1.attention.backend import AttentionMetadata
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.utils.torch_utils import get_dtype_size
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

from contextlib import contextmanager

from vllm_neuron import envs
from vllm_neuron.utils.neuron_utils import model_forward_context
from vllm_neuron.vllm.worker.neuron_ec_connector_model_runner_mixin import (
    ECLoadFailure,
    NeuronECConnectorModelRunnerMixin,
)
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.interfaces import SupportsMRoPE
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
)
from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from vllm_neuron.vllm.sample.rejection_sampler import RejectionSampler
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer
from vllm_neuron.vllm.platform import SO_DISABLED_MESSAGE
from vllm_neuron.utils.bucket_utils import (
    get_max_num_batched_tokens,
    get_decode_padded_batch_size,
)
from vllm_neuron.utils.spec_decode_utils import (
    extract_next_token_ids,
    replicate_per_seq_rows,
)
from vllm_neuron.accuracy.tensor_replacement import (
    TensorReplacer,
    set_active_context,
)

logger = logging.getLogger(__name__)


NULL_BLOCK_ID = 0
# Read-side sentinel for block tables; write-side padding uses NULL_BLOCK_ID.
PAD_SLOT_ID = -1


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
        model_runner,
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
    top_k = torch.full((num_reqs,), -1, dtype=torch.float32, device=device)
    top_p = torch.ones(num_reqs, dtype=torch.float32, device=device)
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)

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
    logger.debug("On-device sampling params (top_k, top_p, temp): %s", result.tolist())
    return result


# TODO: Inherit from LoRAModelRunnerMixin to support LoRA
class NeuronModelRunner(KVConnectorModelRunnerMixin, NeuronECConnectorModelRunnerMixin):
    """
    Model runner that executes the NeuronModel with proper state management.

    This class processes SchedulerOutput from vLLM and returns ModelRunnerOutput
    with correct request ordering and persistent state tracking.
    """

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

        # TODO: Initialize LORA configuration and manager
        # Why required: Required before LORA adapters can be used
        # Current: No LORA config or manager initialization
        # Target: self.lora_config = vllm_config.lora_config if vllm_config else None; self.lora_manager = None

        self.model: Any | None = None
        self._is_synthetic_model: bool = False
        self._target_tensor_capture = None
        self._draft_tensor_capture = None
        self._vision_tensor_capture = None
        model_config = vllm_config.model_config
        # Pooling/embedding models run forward -> hidden states -> pooler ->
        # embedding. Derive the mode from the resolved runner_type.
        self.is_pooling_model = model_config.runner_type == "pooling"
        self._pooling_num_scheduled_tokens_per_seq: np.ndarray | None = None
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim

        # Extract config from vllm_config
        # For multimodal models (e.g., Qwen3-VL), vocab_size is nested in text_config
        hf_config = vllm_config.model_config.hf_config
        if hasattr(hf_config, "text_config") and hasattr(
            hf_config.text_config, "vocab_size"
        ):
            self.vocab_size = hf_config.text_config.vocab_size
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

        # Vision neuron config for multimodal models (None for text-only)
        vision_config_dict = vllm_config.additional_config.get("vision_neuron_config")
        if vision_config_dict is not None:
            from vllm_neuron.model.neuron_config import VisionNeuronConfig

            self.vision_neuron_config = VisionNeuronConfig.from_dict(vision_config_dict)
            self.vision_neuron_config.resolve_tp_dp(
                vllm_config.parallel_config.world_size
            )
        else:
            self.vision_neuron_config = None

        # Multimodal detection via registry (consistent with GPU model runner)
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )
        if self.supports_mm_inputs:
            logger.info(
                "Multimodal on-device encoder cache path enabled. "
                "Vision embeddings injected during prefill via "
                "vision_embedding_blocks."
            )
        elif self.enable_prompt_embeds:
            logger.info("Prompt embeds path enabled for text model.")
        # Staged prompt_embeds for the current step (None => token-id-only step).
        self._current_inputs_embeds: torch.Tensor | None = None
        self._current_is_token_ids: torch.Tensor | None = None
        self.max_num_batched_tokens = get_max_num_batched_tokens(
            self.max_model_len, vllm_config.scheduler_config.max_num_batched_tokens
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
            SUPPORTED_KV_SEGMENT_SIZES,
            get_default_num_seqs_buckets,
            resolve_num_batched_tokens_buckets,
            resolve_segmented_prefill_config,
            validate_decode_context_length_buckets,
            validate_kv_segment_size_buckets,
            validate_num_seqs_buckets,
        )

        user_set_kv_segment_size_buckets = (
            "kv_segment_size_buckets" in neuron_config_dict
        )
        user_set_num_batched_tokens_buckets = (
            "num_batched_tokens_buckets" in neuron_config_dict
        )

        auto_kv_segment_size_buckets: list[int] | None = None
        auto_num_batched_tokens_buckets: list[int] | None = None
        if not user_set_kv_segment_size_buckets:
            (
                auto_kv_segment_size_buckets,
                auto_num_batched_tokens_buckets,
            ) = resolve_segmented_prefill_config(
                self.max_num_batched_tokens, self.max_model_len
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
        elif auto_num_batched_tokens_buckets is not None:
            logger.info(
                "Segmented prefill auto-enabled; num_batched_tokens_buckets=%s",
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

        # Parse kv_segment_size_buckets: explicit opt-in takes precedence
        # over the auto-enable path.
        if user_set_kv_segment_size_buckets:
            buckets = neuron_config_dict.get("kv_segment_size_buckets")
            # Pass num_batched_tokens_buckets only if user explicitly set it
            explicit_num_batched_tokens_buckets = (
                self.neuron_config.num_batched_tokens_buckets
                if user_set_num_batched_tokens_buckets
                else None
            )
            self.neuron_config.kv_segment_size_buckets = (
                validate_kv_segment_size_buckets(
                    buckets,
                    explicit_num_batched_tokens_buckets,
                )
            )
            kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

            # Auto-set num_batched_tokens_buckets to match if not explicitly set
            if not user_set_num_batched_tokens_buckets:
                self.neuron_config.num_batched_tokens_buckets = (
                    self.neuron_config.kv_segment_size_buckets
                )
                logger.info(
                    "Auto-setting num_batched_tokens_buckets to match "
                    "kv_segment_size_buckets: %s",
                    self.neuron_config.kv_segment_size_buckets,
                )

            if self.max_num_batched_tokens < kv_segment_size:
                logger.warning(
                    "max_num_batched_tokens (%s) < segment size (%s). Adjusting to %s.",
                    self.max_num_batched_tokens,
                    kv_segment_size,
                    kv_segment_size,
                )
                self.max_num_batched_tokens = kv_segment_size

            logger.info(
                "Segmented prefill enabled explicitly with kv_segment_size_buckets: %s",
                self.neuron_config.kv_segment_size_buckets,
            )
        elif auto_kv_segment_size_buckets is not None:
            # Cross-check the resolver's segment size against the user's
            # explicit num_batched_tokens_buckets (if any), using the same
            # validation rule as the explicit opt-in path.
            explicit_num_batched_tokens_buckets = (
                self.neuron_config.num_batched_tokens_buckets
                if user_set_num_batched_tokens_buckets
                else None
            )
            self.neuron_config.kv_segment_size_buckets = (
                validate_kv_segment_size_buckets(
                    auto_kv_segment_size_buckets,
                    explicit_num_batched_tokens_buckets,
                )
            )
            logger.info(
                "Segmented prefill auto-enabled with "
                "kv_segment_size_buckets=%s, num_batched_tokens_buckets=%s",
                self.neuron_config.kv_segment_size_buckets,
                self.neuron_config.num_batched_tokens_buckets,
            )
        else:
            logger.info(
                "Segmented prefill disabled (single-shot prefill; "
                "max_num_batched_tokens == max_model_len)"
            )

        if self.neuron_config.kv_segment_size_buckets is None and getattr(
            vllm_config.cache_config, "enable_prefix_caching", False
        ):
            raise ValueError(
                "Automatic Prefix Caching (APC) requires segmented prefill "
                "to be enabled. Either disable APC with "
                "'--no-enable-prefix-caching' or set "
                "'max_num_batched_tokens' to a supported segmented prefill "
                f"size {sorted(SUPPORTED_KV_SEGMENT_SIZES)} to auto-enable "
                "segmented prefill."
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

        self.use_async_scheduling: bool = vllm_config.scheduler_config.async_scheduling
        if self.use_async_scheduling:
            if not self.on_device_sampling:
                raise RuntimeError(
                    "On-device sampling must be enabled for async execution"
                )
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
        self.drafter = None
        self.is_eagle3_spec = False
        self._draft_token_ids = None
        # Async EAGLE3 draft rows by req_id. Only used at batch-composition
        # changes (e.g. several prefills merging into the first bs-wide decode).
        # Steady-state decode goes through the cross-step future path in
        # async_execution_buffer.
        self._async_spec_transition_full_draft_cache: dict[str, torch.Tensor] = {}

        if self.speculative_config:
            # TODO add more spec decode methods
            if self.speculative_config.method == "eagle3":
                self.is_eagle3_spec = True
                self.drafter = EagleProposer(
                    self.vllm_config, self.device, self.on_device_sampling
                )
                self.rejection_sampler = RejectionSampler()
            else:
                raise ValueError(
                    f"Unsupported speculative decoding method: {self.speculative_config.method}"
                )

        # InputBatch is initialized in initialize_kv_cache once KV cache groups are known
        self.input_batch = None
        self._kv_snapshot_enabled: bool = False
        self._block_table_snapshot: list[torch.Tensor] | None = None
        self._encoder_cache_snapshot_enabled: bool = False
        self._encoder_cache_snapshot: dict[str, Any] | None = None
        # Input-snapshot capture: the lite execute context (one shared capturer,
        # built lazily) installed only around this model's forward when enabled.
        self._snapshot_execute_context = None
        # Caches the current step's _get_dp_padding all-reduce as
        # (reduce_inputs, dp_pad, max_decode_ctx_len), where reduce_inputs is
        # (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode). An
        # async-scheduling re-prep of the same batch reuses the result instead
        # of issuing a second cross-DP reduce (which would desync the
        # host-reduce/EP-forward collective streams across DP ranks and
        # deadlock); reduce_inputs lets the re-prep assert its inputs are
        # unchanged. Overwritten on every primary (non-reuse) prep.
        # Cache tuple: (reduce_inputs, dp_pad, max_decode_ctx_len,
        # group_force_nonspec). reduce_inputs =
        # (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode). The
        # async re-prep reuses dp_pad/max_decode_ctx_len/group_force_nonspec
        # and only re-validates the strip-invariant padded_num_reqs.
        self._dp_padding_cache: tuple[tuple[int, int, bool], int, int, bool] | None = (
            None
        )
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # EPD: Vision encoder only construction gate and PD language-only construction gate.
        # Resolved by set_construction_role below; default False = monolith.
        self.mm_encoder_only: bool = False
        self.mm_language_model_only: bool = False

        # EPD construction-role flags ride on vision_neuron_config (the VE and
        # PD pools always have one; text-only models have none and default to
        # the monolith both-False path). Must resolve before _init_encoder_cache
        # so min_hold_time_ms derives correctly (0 for monolithic, 100ms for EPD).
        if self.vision_neuron_config is not None:
            self.vision_neuron_config.set_construction_role(
                vllm_config.model_config.multimodal_config,
                vllm_config.additional_config,
            )
            self.mm_encoder_only = self.vision_neuron_config.mm_encoder_only
            self.mm_language_model_only = (
                self.vision_neuron_config.mm_language_model_only
            )
        if self.mm_encoder_only:
            logger.info("EPD: mm_encoder_only=True - building vision-only model")
        if self.mm_language_model_only:
            logger.info(
                "EPD: mm_language_model_only=True - building language-only model"
            )

        # On-device encoder cache for multimodal models
        if self.supports_mm_inputs:
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
        self.max_vision_blocks_per_request.
        """
        from vllm.multimodal.encoder_budget import MultiModalBudget
        from vllm_neuron.utils.vision_utils import get_vision_token_merge_factor
        from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

        vnc = self.vision_neuron_config
        model_config = vllm_config.model_config
        merge_factor = get_vision_token_merge_factor(model_config.hf_config)

        # fat_dim = visual_dim * (1 + num_deepstack_levels)
        # This is the embedding width stored per token in the cache buffer.
        hf_config = model_config.hf_config
        vis_config = getattr(hf_config, "vision_config", None)
        if vis_config is not None:
            visual_dim = getattr(
                vis_config, "out_hidden_size", model_config.get_hidden_size()
            )
            deepstack_indexes = (
                getattr(vis_config, "deepstack_visual_indexes", None) or []
            )
            num_deepstack = len(deepstack_indexes)
        else:
            visual_dim = model_config.get_hidden_size()
            num_deepstack = 0
        fat_dim = visual_dim * (1 + num_deepstack)
        assert vnc.vision_attention_block_size % merge_factor == 0, (
            f"vision_attention_block_size ({vnc.vision_attention_block_size}) "
            f"must be divisible by merge_factor ({merge_factor})"
        )
        cache_block_size = vnc.vision_attention_block_size // merge_factor

        # Max vision blocks one request can use (from largest VE bucket)
        max_bucket = max(vnc.num_vision_tokens_buckets)
        assert max_bucket % merge_factor == 0, (
            f"Largest VE bucket ({max_bucket}) must be divisible by "
            f"merge_factor ({merge_factor})"
        )
        self.max_vision_blocks_per_request = math.ceil(
            max_bucket // merge_factor / cache_block_size
        )

        # Derive min_blocks_needed from scheduler's encoder_cache_size:
        #   encoder_cache_size = max(max_num_batched_tokens, max_single_mm_item_tokens)
        # This is the max merged encoder tokens the scheduler can have cached
        # simultaneously (set in vllm.v1.core.encoder_cache_manager).
        # We auto-derive from this value to stay in sync with the scheduler's
        # EncoderCacheManager — both use the same budget so the worker never
        # runs out of blocks for items the scheduler considers cacheable.
        # We divide by cache_block_size to get blocks, plus one scratch block
        # that absorbs VE padding writes and is never allocated to real items.
        #
        # TODO: The scheduler's EncoderCacheManager sizes and evicts based on
        # token counts, unaware of block-level padding waste. This mismatch
        # means the scheduler may cache more items than the block allocator
        # can hold (each item uses a full block regardless of token count).
        # A proper fix requires integrating block layout awareness into the
        # scheduler's EncoderCacheManager. Until then, set vnc.encoder_cache_num_blocks
        # explicitly for workloads with many small images.
        num_blocks = vnc.encoder_cache_num_blocks
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

        if self.max_vision_blocks_per_request > num_blocks - 1:
            raise ValueError(
                f"encoder_cache_num_blocks={num_blocks} is too small: "
                f"a single request needs up to {self.max_vision_blocks_per_request} "
                f"blocks (largest VE bucket={max_bucket}, "
                f"cache_block_size={cache_block_size}), but only "
                f"{num_blocks - 1} allocatable blocks available. "
                f"Increase encoder_cache_num_blocks or reduce "
                f"num_vision_tokens_buckets."
            )

        min_hold_time_ms = vnc.encoder_cache_min_hold_time_ms or 0.0

        self.encoder_cache = EncoderCacheBlocks(
            num_blocks=num_blocks,
            block_size=cache_block_size,
            fat_dim=fat_dim,
            dtype=torch.bfloat16,
            device=self.device,
            min_hold_time_ms=min_hold_time_ms,
        )
        logger.info(
            "On-device encoder cache: num_blocks=%d, block_size=%d, "
            "fat_dim=%d, max_vision_blocks_per_request=%d, buffer_size_mb=%.1f",
            num_blocks,
            cache_block_size,
            fat_dim,
            self.max_vision_blocks_per_request,
            num_blocks * cache_block_size * fat_dim * 2 / (1024 * 1024),
        )

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
            if self.vision_neuron_config is not None:
                # Init multimodal model with vision_neuron_config if provided
                self.model = model_cls.from_configs(
                    hf_config=self.vllm_config.model_config.hf_config,
                    text_neuron_config=self.neuron_config,
                    vision_neuron_config=self.vision_neuron_config,
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

        if self.is_eagle3_spec:
            if supports_eagle3(self.model):
                # First try to get aux_layers from draft model config, then fall back to model's default
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if aux_layers:
                    logger.info(
                        "Using auxiliary layers from speculative config: %s", aux_layers
                    )
                else:
                    aux_layers = self.model.get_eagle3_aux_hidden_state_layers()
                    logger.info(
                        "Using model's default auxiliary layers: %s", aux_layers
                    )
                self.model.set_aux_hidden_state_layers(aux_layers)
            else:
                raise RuntimeError(
                    "Model does not support EAGLE3 interface but EAGLE3 spec decoding was requested"
                )

        self._target_tensor_capture = None
        self._draft_tensor_capture = None
        self._vision_tensor_capture = None

        # Compile model with vllm_neuron backend for Neuron execution
        # This is where torch.compile uses the registered "vllm_neuron" backend
        logger.info("Compiling model with vllm_neuron backend")

        # Disable dynamo cache limit — recompilations are caught at runtime
        # by FailOnRecompileLimitHit, so a static limit adds no value.
        torch._dynamo.config.cache_size_limit = 2**62

        # Build compile options
        self.compile_options = {"alias_meta_to_neuron": True}

        # FIXME: -O1 and mac-threshold are temporary until NKI adds MAC count estimates for kernels.
        # A model can opt out via neuron_config.hlo2tensorizer_options: the
        # threshold only exists for NKI kernels, and it makes neuronx-cc fail
        # codegen on some pure-torch graphs (Qwen3.5's decode graph raises
        # NCC_IBTN006 with it, and compiles cleanly without it).
        override = getattr(self.neuron_config, "hlo2tensorizer_options", None)
        hlo2tensorizer_opts = (
            "--modular-flow-mac-threshold=10" if override is None else override
        )
        # The unsafe fp8 cast flag is only needed on Trn2 where kernels use
        # legacy nl.float8_e4m3 (max=240). Trn3 supports OCP e4m3fn natively.
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        has_fp8 = self.vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
        if not has_fp8 and self.vllm_config.model_config.quantization in ("modelopt",):
            has_fp8 = True
        if not has_fp8:
            has_fp8 = getattr(self.neuron_config, "quantization", None) == "fp8"
        if has_fp8 and get_platform_target() not in ("trn3", "trn3pre"):
            hlo2tensorizer_opts = (
                f"{hlo2tensorizer_opts} --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
            ).strip()
        # vLLM optimization levels map 1:1 onto neuronx-cc optlevels (CHRS-721).
        self.compile_options["compiler_args"] = [
            "--auto-cast=none",
            "--verbose=35",
            f"-O{self.vllm_config.optimization_level.value}",
            # --enable-nested-dynamic-loop: NKI kernels may contain nested
            # dynamic loops (scf.for inside scf.for). After inline_nki_kernel
            # inlines kernel BIR into the parent graph, birverifier rejects the
            # resulting multiple back-edges without this flag.
            "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
        ]
        if hlo2tensorizer_opts:
            self.compile_options["compiler_args"].insert(
                -1, f"--internal-hlo2tensorizer-options={hlo2tensorizer_opts}"
            )
        logger.info(
            "neuronx-cc optlevel -O%s (from vLLM optimization_level)",
            self.vllm_config.optimization_level.value,
        )

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
        from vllm_neuron.envs import get_compile_backend_name

        if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
            logger.debug("Graph capture and parallel compilation is disabled.")
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                self.model,
                backend="neuron_libtorch_graph_capture",
                fullgraph=fullgraph_enabled,
                options=self.compile_options,
            )

        self.model = torch.compile(
            self.model,
            backend=get_compile_backend_name(),
            fullgraph=fullgraph_enabled,
            options=self.compile_options,
        )

        # Compile vision encoder separately (called from embed_multimodal,
        # not from the main text model forward graph).
        if (
            self.vision_neuron_config is not None
            and getattr(self.model, "visual", None) is not None
        ):
            inner_model = getattr(self.model, "_orig_mod", self.model)

            if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
                self.vision_capture_backend = None
            else:
                self.vision_capture_backend = torch.compile(
                    inner_model.visual,
                    backend="neuron_libtorch_graph_capture",
                    fullgraph=fullgraph_enabled,
                    options=self.compile_options,
                )

            inner_model.visual = torch.compile(
                inner_model.visual,
                backend=get_compile_backend_name(),
                fullgraph=fullgraph_enabled,
                options=self.compile_options,
            )
            logger.info("Vision encoder compiled separately")
        else:
            self.vision_capture_backend = None

        logger.info("NeuronModel loading complete (moved to device and compiled)")

        if self.drafter is not None:
            logger.info("Spec decode enabled. Loading draft model ...")
            # TODO: model loading logic could be extracted
            # Pass target model's padded hidden_size to draft model
            target_hidden_size = self.model.config.hidden_size
            self.drafter.load_model(target_hidden_size=target_hidden_size)
            logger.info("Draft model loading complete (moved to device and compiled)")

    def init_tensor_capture(self) -> None:
        """Initialize tensor capture for target, vision, and draft models.

        Must be called after load_model() (models are compiled) and before
        warmup (so dynamo traces hooks and inline capture_tensor() calls).
        """
        from vllm_neuron.accuracy.tensor_capture import TensorRegistry, expand_patterns

        capture_config = self.neuron_config.tensor_capture
        if not capture_config:
            return

        # Split modules into text and vision (visual.* prefix)
        all_modules = (
            expand_patterns(capture_config.modules) if capture_config.modules else []
        )
        text_modules = [m for m in all_modules if not m.startswith("visual.")]
        vision_modules = [
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

        # Vision encoder capture
        self._vision_tensor_capture = None
        if vision_modules and self.vision_neuron_config is not None:
            inner_model = getattr(self.model, "_orig_mod", self.model)
            if getattr(inner_model, "visual", None) is not None:
                self._vision_tensor_capture = self._setup_capture(
                    inner_model.visual, vision_modules, subdirectory="vision"
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
            and prev_future.shape[0] == num_reqs_padded
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
        req_state = self.requests[req_id]
        # Update with new data as needed
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
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.free(mm_hash)

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

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
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
        num_rejected = 0 (no-op). Padded rows are left untouched: they are
        not live, and overwriting their previous sampled-token rows
        wedges the async worker-side sample_tokens path.

        Args:
            prev_future: Device tensor ``[num_reqs_padded, num_spec+1]`` in
                the previous step's batch ordering.
            num_reqs_padded: Padded decode batch size for the current step.

        Returns:
            Device tensor ``[num_reqs_padded, num_spec+1]`` reordered to the
            current batch's compacted slot order.
        """
        prev_req_ids_ordered = self.async_execution_buffer.get("prev_req_ids_ordered")
        curr_req_ids = list(self.input_batch.req_ids[: self.input_batch.num_reqs])
        if prev_req_ids_ordered is None:
            return prev_future

        prev_id_to_idx = {req_id: i for i, req_id in enumerate(prev_req_ids_ordered)}

        perm: list[int] = []
        missing_indices: list[int] = []
        for cur_idx, req_id in enumerate(curr_req_ids):
            old_idx = prev_id_to_idx.get(req_id)
            if old_idx is None:
                # Request wasn't in the previous step (new). Use cur_idx as a
                # placeholder; we'll overwrite with a dummy row below.
                perm.append(cur_idx)
                missing_indices.append(cur_idx)
            else:
                perm.append(old_idx)
        # Padded slots: identity (kept past num_reqs).
        for pad_idx in range(self.input_batch.num_reqs, num_reqs_padded):
            perm.append(pad_idx)

        needs_remap = any(old_idx != cur_idx for cur_idx, old_idx in enumerate(perm))
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

    def _save_block_table_to_device(self) -> None:
        """
        To avoid data dependencies that block async execution, we should store the
        full block table tensor on device instead of updating it in place with commit_block_table
        """
        for block_table in self.input_batch.block_table.block_tables:
            block_table.block_table.gpu = _remap_null_block_to_sentinel(
                block_table.block_table.cpu.to(self.device)
            )
        # Snapshot block tables for KV cache analysis (survives request freeing).
        # Apply remap on CPU tensor directly to avoid device→CPU transfer.
        if self._kv_snapshot_enabled:
            self._block_table_snapshot = [
                _remap_null_block_to_sentinel(bt.block_table.cpu)
                for bt in self.input_batch.block_table.block_tables
            ]

    def _save_slot_mapping_to_device(self) -> None:
        """
        To avoid data dependencies that block async execution, we should store the
        full slot mapping tensor on device instead of updating it in place with commit_slot_mapping
        """
        for block_table in self.input_batch.block_table.block_tables:
            block_table.slot_mapping.gpu = block_table.slot_mapping.cpu.to(self.device)

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput") -> None:
        """Encode uncached multimodal inputs into the on-device encoder cache.

        Allocates cache blocks for each new mm_item, then calls
        embed_multimodal() which has the VE NEFF scatter-write directly
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

        # prompt_embeds modality: pre-computed embeddings that bypass the VE.
        # Stored directly in cache via CPU slice ops (no VE encoding needed).
        # The prefill read path (_gather_mm_embeddings) retrieves them the same
        # way as vision embeddings — block-based lookup is modality-agnostic.
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

        # Filter already-cached items
        uncached = [
            (mm_hash, modality, data)
            for mm_hash, (modality, data) in zip(mm_hashes, mm_kwargs)
            if not self.encoder_cache.contains(mm_hash)
        ]
        if not uncached:
            return

        # Group by modality and batch, then delegate to model.
        # The model handles allocation internally (model-specific token counting).
        uncached_kwargs = [(modality, data) for _, modality, data in uncached]
        uncached_hashes = [mm_hash for mm_hash, _, _ in uncached]

        t0 = time.perf_counter()
        item_cursor = 0
        if self._vision_tensor_capture is not None:
            self._vision_tensor_capture.setup_tensor_capture()
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

        if self._vision_tensor_capture is not None:
            self._vision_tensor_capture.save_tensor_captures(
                positions=torch.tensor([]),
                is_prefill=True,
                req_ids=list(scheduler_output.scheduled_encoder_inputs.keys()),
            )

        logger.debug(
            "[mm_encoder] %d items encoded in %.1fms",
            len(uncached),
            (t_encode - t0) * 1000,
        )

    def _gather_mm_embeddings(
        self,
        num_tokens: int,
        req_ids: list[str],
        scheduler_output: "SchedulerOutput",
        padding_map: dict[str, int],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Gather multimodal embeddings for the current prefill step.

        Analogous to upstream vLLM's GPUModelRunner._gather_mm_embeddings:
        determines which cached vision items overlap the current scheduling
        window and returns them ready for the prefill graph.

        Unlike upstream (which returns variable-length tensor slices for
        CPU-side scatter into inputs_embeds), this returns zero-copy block
        views from the on-device cache buffer + a position map. The prefill
        graph uses torch.stack + index_put_ to merge them into hidden_states.

        Args:
            num_tokens: Total padded token count for this step.
            req_ids: Ordered request IDs in the batch.
            scheduler_output: Scheduler output with scheduling info.
            padding_map: Per-request padding counts.

        Returns:
            (vision_embedding_blocks, vision_positions) where:
            - vision_embedding_blocks: tuple of [block_size, fat_dim] tensors,
              zero-copy views into the cache buffer. Length =
              max_num_vision_blocks (padded with scratch block views).
            - vision_positions: [max_num_vision_blocks, block_size] int64 —
              vision_positions[i, j] = batch position for token j in block i.
              Sentinel value = num_tokens for don't-care positions.
        """
        assert len(req_ids) == 1, (
            f"_gather_mm_embeddings expects single-request prefill but got "
            f"{len(req_ids)} requests. The prefill NEFF input is sized for "
            f"max_vision_blocks_per_request={self.max_vision_blocks_per_request} "
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

        max_num_vision_blocks = self.max_vision_blocks_per_request

        vision_positions = torch.full(
            (max_num_vision_blocks, self.encoder_cache.block_size),
            num_tokens,
            dtype=torch.int64,
            device="cpu",
        )
        block_ids_list: list[int] = []
        block_cursor = 0

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

                    item_token_offset = 0
                    for block_idx, block_id in enumerate(block_ids):
                        if block_cursor >= max_num_vision_blocks:
                            raise RuntimeError(
                                f"Vision block truncation: request has more "
                                f"cached blocks than "
                                f"max_vision_blocks_per_request="
                                f"{max_num_vision_blocks}. This indicates a "
                                f"configuration mismatch between VE bucket "
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
                                vision_positions[block_cursor, lo:hi] = seq_pos[
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
                            vision_positions[block_cursor] = torch.where(
                                in_range, batch_positions, num_tokens
                            )

                        block_cursor += 1
                        item_token_offset += chunk

            req_batch_offset += num_scheduled + padding

        # Pad with scratch block for unused slots.
        while len(block_ids_list) < max_num_vision_blocks:
            block_ids_list.append(self.encoder_cache.scratch_block_id)

        # Build zero-copy views from the cache buffer.
        vision_embedding_blocks = tuple(
            self.encoder_cache.buffer[i] for i in block_ids_list
        )
        return vision_embedding_blocks, vision_positions.to(self.device)

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
        assert (
            scheduler_output.total_num_scheduled_tokens > 0
        )  # this is actual token count, no padding
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # DI mixed-batch + cross-DP spec coordination. With DP+EP the spec vs
        # non-spec decode choice selects which compiled NEFF runs, and that
        # NEFF holds the cross-DP collectives — every rank must dispatch the
        # SAME graph or the collectives mismatch (``status=1100 Collective
        # Operation Pending``). Under DI the ranks fall out of phase as each
        # request's KV transfer completes independently, so on a step some
        # ranks are on their first (non-spec) decode while others speculate.
        #
        # We MAX-reduce each rank's "must run non-spec" vote across DP (folded
        # into the _get_dp_padding reduce below as its 4th element) and, if any
        # rank can't speculate, strip the WHOLE group to the non-spec NEFF.
        # This must happen BEFORE the token tensors are built (so they're sized
        # 1-token-per-req), and therefore before the _get_dp_padding reduce.
        # That ordering is sound because the reduce's other inputs are
        # invariant to the strip: padded_num_reqs buckets purely on num_reqs
        # (get_bucket_for_count) and the strip drops draft *tokens*, not
        # requests; max_query_len only gates decode-vs-prefill (both 1 and
        # 1+num_spec are decode); local_max_decode_ctx_len reads
        # num_computed_tokens. So we run the reduce up front, then strip.
        # padded_num_reqs is strip-invariant (buckets purely on num_reqs;
        # max_query_len only gates decode-vs-prefill, and both 1 and 1+num_spec
        # are decode), so it's identical on the primary prep and the async
        # re-prep — recompute it on both paths.
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
            cached_inputs, dp_pad, max_decode_ctx_len, group_force_nonspec = (
                self._dp_padding_cache
            )
            # The strip-invariant size must match what the primary prep saw.
            assert cached_inputs[0] == padded_num_reqs, (
                "DP-padding reuse invariant violated: primary-prep padded_num_reqs "
                f"{cached_inputs[0]} != re-prep {padded_num_reqs}. The async "
                "re-prep must not change this rank's cross-DP batch size."
            )
            logger.debug("Reusing cached DP padding for async re-prep")
        else:
            # Primary prep: compute this rank's spec/non-spec vote and the
            # decode-ctx input, run the single folded reduce, cache the result,
            # then apply the synced strip — all BEFORE the token tensors below
            # are built, so they're sized 1-token-per-req when we fall back to
            # the non-spec NEFF. See _get_dp_padding /
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
            dp_pad, max_decode_ctx_len, _, group_force_nonspec = self._get_dp_padding(
                padded_num_reqs,
                local_max_decode_ctx_len,
                is_spec_decode=is_spec_decode_step,
                local_force_nonspec=local_force_nonspec,
            )
            # Cache the (strip-invariant) batch size + reduced outputs so the
            # same-step re-prep reuses them without a second reduce.
            self._dp_padding_cache = (
                (padded_num_reqs, local_max_decode_ctx_len, is_spec_decode_step),
                dp_pad,
                max_decode_ctx_len,
                group_force_nonspec,
            )
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
        if self.uses_mrope:
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
                block_table.block_table.cpu,
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

        self._save_slot_mapping_to_device()

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
                # No padding, use actual cumsum directly
                logits_indices = (
                    torch.tensor(cu_num_tokens.tolist(), dtype=torch.long) - 1
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

        # Multimodal path: gather vision embeddings from on-device encoder cache.
        # Only during prefill — decode NEFF has no vision inputs
        # (vision contributions are already in KV cache from prefill).
        mm_kwargs: dict[str, Any] = {}
        is_prefill_step = not self._is_decode()
        if self.supports_mm_inputs and is_prefill_step:
            num_tokens = input_ids.shape[0]
            vision_embedding_blocks, vision_positions = self._gather_mm_embeddings(
                num_tokens, req_ids, scheduler_output, padding_map
            )
            mm_kwargs["vision_embedding_blocks"] = vision_embedding_blocks
            mm_kwargs["vision_positions"] = vision_positions

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
        tokens per request per step; otherwise decode is 1 token per request.

        Matches the `decode_token_threshold` field set on the per-step
        attention metadata by `_build_attention_metadata` whenever all
        scheduled requests carry the full `num_speculative_tokens` drafts
        — the invariant upheld by `NeuronScheduler` (prefill/decode
        separation, no mid-step new-req entries).
        """
        if self.speculative_config is not None:
            return 1 + self.speculative_config.num_speculative_tokens
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
            num_prompt_tokens = req_state.num_prompt_tokens
            # NOTE: Use same prefill detection as has_prefill above (handles prefix caching)
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

    def _compute_swa_num_blocks(self, sliding_window: int, block_size: int) -> int:
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
        dcp_block_size = block_size * max(self._dcp_size, 1)
        full_ctx_blocks = (self.max_model_len + dcp_block_size - 1) // dcp_block_size
        return min(num_swa_blocks, full_ctx_blocks)

    def _compute_swa_decode_tensors(
        self,
        blk_table,
        padded_num_reqs: int,
        sliding_window: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For a SWA decode step, compute:
        - the block_table_tensor trimmed to the window-relevant blocks, so
          the attention kernel reads only ~window/block_size blocks from HBM.
        - the per-seq `swa_kv_pos_offset` tensor (= start_block * block_size),
          which the model subtracts from pos_ids so the causal mask is
          computed in the trimmed frame.
        """
        num_swa_blocks = self._compute_swa_num_blocks(sliding_window, block_size)
        ctx_lens = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu[:padded_num_reqs]
        )
        start_blocks = torch.clamp(ctx_lens // block_size - num_swa_blocks + 1, min=0)
        gather_idx = start_blocks.unsqueeze(1) + torch.arange(
            num_swa_blocks, dtype=start_blocks.dtype
        ).unsqueeze(0)
        gpu_bt = blk_table.get_device_tensor(padded_num_reqs)
        trimmed_bt = gpu_bt.gather(1, gather_idx.long().to(self.device))
        swa_kv_pos_offset = (start_blocks * block_size).to(
            dtype=torch.int32, device=self.device
        )
        return trimmed_bt, swa_kv_pos_offset

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

        needed_seq = max_decode_ctx_len + 1 + max_num_draft_tokens
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
            slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]
            # DCP prefill: prefill now uses the same scheme as decode — it
            # writes only this rank's interleaved token shard (S/DCP) to the
            # per-rank cache. Extract slot_mapping entries at owned positions
            # using the same interleave pattern. Size is always S/DCP
            # (deterministic, no recompile).
            # Slice on CPU (Neuron tensors don't support .contiguous()) then move back.
            if not is_decode and self.cp_world_size > 1:
                # Hybrid KV manager scales block_size per KV group. We need
                # to always use an interleave size equal to block_size
                I = block_size
                W = self.cp_world_size
                R = self.cp_rank
                S = total_num_scheduled_tokens
                ownership_stride = W * I
                if cached_seq_len % ownership_stride != 0:
                    raise ValueError(
                        "DCP prefill requires cached_seq_len to be divisible by "
                        "decode_context_parallel_size * block_size "
                        f"({W} * {I} = {ownership_stride}); got {cached_seq_len}."
                    )
                slot_mapping = (
                    slot_mapping.cpu()
                    .view(S // (W * I), W, I)[:, R, :]
                    .contiguous()
                    .reshape(-1)
                    .to(self.device)
                )
            cached_seq_len_tensor = torch.tensor(
                [[cached_seq_len]], dtype=torch.int32, device=self.device
            )

            # Note there are more data available in self.input_batch you can
            # use in attention_metadata. Now we only use the ones below.
            kv_segment_size = 0
            if self.neuron_config.kv_segment_size_buckets is not None:
                kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

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
                    spec.sliding_window, block_size
                )
                if num_swa_blocks < blk_table_tensor.shape[1]:
                    blk_table_tensor, swa_kv_pos_offset = (
                        self._compute_swa_decode_tensors(
                            blk_table,
                            padded_num_reqs,
                            spec.sliding_window,
                            block_size,
                        )
                    )
                else:
                    # Short sequence: no trimming needed, but always pass
                    # swa_kv_pos_offset to keep the FX graph consistent
                    # with warmup (avoids torch._dynamo recompilation).
                    swa_kv_pos_offset = torch.zeros(
                        padded_num_reqs, dtype=torch.int32, device=self.device
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
                    bucket_idx = torch.arange(bucket_blocks, device=self.device)
                    blk_table_tensor = torch.index_select(
                        blk_table_tensor, 1, bucket_idx
                    )
                swa_kv_pos_offset = torch.zeros(
                    padded_num_reqs, dtype=torch.int32, device=self.device
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
                    spec.sliding_window, block_size
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
            block_table_tensor = (
                torch.arange(max_num_blocks_per_req, dtype=torch.int32)
                .unsqueeze(0)
                .expand(num_reqs, -1)
                .contiguous()
                .to(device)
            )
            # Always provide an untrimmed block_table with the full
            # ``max_model_len`` dim so the on-device spec-decode correction
            # (which needs to compute slot indices into the full KV cache)
            # traces a shape consistent with runtime.
            full_block_table_tensor = (
                torch.arange(full_max_blocks_per_req, dtype=torch.int32)
                .unsqueeze(0)
                .expand(num_reqs, -1)
                .contiguous()
                .to(device)
            )

            # Dummy slot_mapping: [num_tokens] or [num_tokens/DCP] for DCP prefill
            if is_prefill and self.cp_world_size > 1:
                # Hybrid KV manager scales block_size per KV group. We need
                # to always use an interleave size equal to block_size
                I = block_size
                W = self.cp_world_size
                R = self.cp_rank
                slot_mapping = (
                    torch.arange(num_tokens, dtype=torch.int64)
                    .view(num_tokens // (W * I), W, I)[:, R, :]
                    .contiguous()
                    .reshape(-1)
                    .to(device)
                )
            else:
                slot_mapping = torch.arange(num_tokens, dtype=torch.int64).to(device)

            # max_query_len: for prefill = bucket_size, for decode = 1
            max_query_len = num_tokens // num_reqs

            kv_segment_size = 0
            if self.neuron_config.kv_segment_size_buckets is not None:
                kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

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
        if self.supports_mm_inputs:
            max_num_vision_blocks = self.max_vision_blocks_per_request
            warmup_kwargs["vision_embedding_blocks"] = tuple(
                torch.zeros(
                    self.encoder_cache.block_size,
                    self.encoder_cache.fat_dim,
                    dtype=self.encoder_cache.dtype,
                    device=device,
                )
                for _ in range(max_num_vision_blocks)
            )
            warmup_kwargs["vision_positions"] = torch.full(
                (max_num_vision_blocks, self.encoder_cache.block_size),
                num_tokens,
                dtype=torch.int64,
                device=device,
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
                "Capturing EAGLE3 prefill graphs for bucket size: %d", bucket_size
            )
            self.drafter.graph_extract(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=kwargs["attn_metadata"],
                device=device,
            )

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
            logger.info("Warming up EAGLE3 for bucket size: %d", bucket_size)
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
        # vision embeddings are already in KV cache from prefill. Only pass
        # inputs_embeds during decode for non-multimodal prompt_embeds use cases.
        if self.enable_prompt_embeds and self.vision_neuron_config is None:
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

        return decode_warmup_kwargs

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
        try:
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
            try:
                _ = self.capture_backend_model(**kwargs)
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
            logger.info("Capturing EAGLE3 decode graphs for batch size: %d", batch_size)
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
            model_output = self.model(**kwargs)
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
            logger.info("Warming up EAGLE3 for batch size: %d", batch_size)
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
                    self._materialize_pending_async_output("batch composition changed")
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

        # Update cached states (includes sequence ID management)
        with record_function_or_nullcontext("neuron_model_runner: update_states"):
            self._update_states(scheduler_output)

        # Register new requests for tensor replacement prompt matching.
        self._register_requests_for_replacement(scheduler_output)

        # Execute multimodal encoder for scheduled inputs (prefill only).
        # Gather happens later in _prepare_model_input_impl (also prefill-guarded)
        # because it depends on padding_map computed there.
        if self.vision_neuron_config is not None and not self._is_decode():
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
                # Drain the VE NEFF from the NRT execution queue before we return the ready-ack to the router.
                _ = self.encoder_cache.buffer[0, 0, 0].detach().to("cpu")
                self._attach_ec_locator(out, scheduler_output)
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
            original_input_ids = input_ids
            input_ids = self._maybe_swap_async_input_ids(input_ids)
            swap_declined = input_ids is original_input_ids
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
                    logit_mask, input_ids.shape[0]
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
                    logit_mask, input_ids.shape[0]
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
            # Execute model with padded inputs
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
                    )

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )

        # If spec decode enabled, log acceptance stats and propose draft tokens.
        # aux_hidden_states is None when eagle3 is not active.
        if self.is_eagle3_spec and aux_hidden_states is not None:
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
        """True when this step must finish right after the vision encoder.

        A vision-encoder-only pool (mm_encoder_only=True) has no language model
        or sampler, so once _execute_mm_encoder has run we return a token-less
        output instead of falling through to the LM forward.

        That return happens before the language model's cross-DP collectives
        (_coordinate_dp_spec_decode / _get_dp_padding). Those collectives must
        run on every rank or not at all -- if some ranks skip them and others
        don't, the group deadlocks. A vision-only pool runs no language model,
        so they are inactive and skipping them is safe; the assert below
        enforces that and fails loudly rather than risk a silent hang. The
        vision encoder's own collectives are unaffected -- they already ran
        inside _execute_mm_encoder, before this return.
        """
        if not self.mm_encoder_only:
            return False
        if self._is_decode():
            # A vision encoder never decodes; never finish a decode step here.
            return False
        assert not self._dp_collectives_active(), (
            "VE encode-finish returns before the language-model cross-DP "
            "collectives; those collectives are active on this pool, so skipping "
            "them would hang the group. A vision-encoder-only pool must not "
            "enable language-model DP (expert/attention/embedding/mlp/lm_head) "
            "-- this is the LM-DP axis, not vision_dp_size."
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
            NeuronECConnectorOutput,
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
        output.ec_connector_output = NeuronECConnectorOutput(
            finished_sending=None,
            finished_recving=None,
            embed_locator=embed_locator,
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
        connector.register_and_serve(cache.buffer, tp_rank=tp_rank)

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

    def _uses_async_eagle3_transition_cache(self) -> bool:
        """True when async + EAGLE3, i.e. when the transition cache applies."""
        return self.use_async_scheduling and self.is_eagle3_spec

    def _cache_async_spec_transition_drafts(
        self,
        req_ids_output_copy: list[str],
    ) -> None:
        """Stash EAGLE3 draft rows by req_id so we can replay them later.

        Steady state hands draft futures straight from one NEFF to the next —
        previous and current batches match in shape and order, so nothing else
        is needed. At a prefill->decode batch-composition change that breaks:
        the previous future only covers the request that just prefilled, but
        the next decode is bs-wide. The cache fills that gap. It lives in the
        plugin; we don't rely on upstream scheduler draft propagation.
        """
        if not self._uses_async_eagle3_transition_cache():
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
            "Cached async EAGLE3 transition drafts for req_ids=%s full_shape=%s",
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
            self._uses_async_eagle3_transition_cache()
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
            "Missing async EAGLE3 transition drafts for req_ids=%s; "
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
            self._uses_async_eagle3_transition_cache()
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
                "Missing async EAGLE3 transition drafts after spec metadata "
                f"was built for req_ids={req_ids}"
            )

        full_flat, drafts_flat = pieces
        if full_flat.numel() > input_ids.numel():
            raise RuntimeError(
                "Cached async EAGLE3 transition input_ids exceed placeholder "
                f"size: cached={full_flat.numel()} placeholder={input_ids.numel()}"
            )
        if drafts_flat.numel() > spec_decode_metadata.draft_token_ids.numel():
            raise RuntimeError(
                "Cached async EAGLE3 transition draft metadata exceeds "
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
            "Filled async EAGLE3 batch-change inputs from transition cache: "
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

        Capture is opt-in debug, so this is a zero-cost no-op when disabled.
        When enabled, this forward's identity + capture verdict and the lite
        capture hook are set immediately before the forward and cleared
        immediately after — so capture is scoped to this model's executes and
        not, e.g., a draft or vision model's, and a raised forward can't leave a
        stale context tagging the next one.
        """
        if not envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
            yield
            return

        from libtorch_neuronx_lite.compile.execute_context import clear_execute_context

        from vllm_neuron.snapshot.context import clear_current_forward

        try:
            self._publish_snapshot_context(positions, spec_decode_metadata)
            self._set_snapshot_execute_context()
            yield
        finally:
            clear_current_forward()
            clear_execute_context()

    def _set_snapshot_execute_context(self):
        """Set the lite execute context that runs input-snapshot capture.

        The context wraps one shared ``SnapshotCapturer`` (its ``pre_execute`` is
        the ``pre_execute_hook``), built once per process so the capturer's
        per-NEFF call index persists across forwards. It is (re)set before every
        captured forward and cleared after.
        """
        from libtorch_neuronx_lite.compile.execute_context import (
            ExecuteContext,
            set_execute_context,
        )

        if self._snapshot_execute_context is None:
            from vllm_neuron.snapshot.capture import SnapshotCapturer

            self._snapshot_execute_context = ExecuteContext(
                pre_execute_hook=SnapshotCapturer().pre_execute
            )
        set_execute_context(self._snapshot_execute_context)

    def _publish_snapshot_context(self, positions, spec_decode_metadata) -> None:
        """Resolve and publish this forward's input-snapshot context.

        The token/request verdict is the same for every NEFF in a forward, so it
        is evaluated once here — above the compiled boundary, where positions and
        request ids are still semantic — and published for the executables to
        read. Fails loudly: capture is opt-in debug, so a malformed selection or
        a failure computing the verdict aborts the forward rather than silently
        producing no snapshot.
        """
        from vllm_neuron.snapshot.config import get_snapshot_config
        from vllm_neuron.snapshot.context import (
            SnapshotForwardContext,
            next_global_step,
            set_current_forward,
        )

        # Process-wide config, resolved once and shared with the executables'
        # capture specs; the selector is constant for the run.
        selector = get_snapshot_config().selector

        # Decode-vs-prefill from the runner's source of truth (request state,
        # not tensor shapes) so it stays correct under spec decode / prefix
        # caching. On Neuron a batch is wholly one phase.
        is_decode = self._is_decode()

        req_ids = list(self.input_batch.req_ids)
        # The positions host-sync is only worth paying when a token/request
        # rule needs the values; call-index-only capture skips it.
        if selector.has_value_rule():
            pos_list = positions.detach().to("cpu").reshape(-1).tolist()
            spec_window = 1
            spec_cfg = getattr(self.vllm_config, "speculative_config", None)
            if spec_decode_metadata is not None and spec_cfg is not None:
                spec_window = (getattr(spec_cfg, "num_speculative_tokens", 0) or 0) + 1
            capture, matches = selector.evaluate_forward(
                req_ids, pos_list, is_decode, spec_window
            )
        else:
            pos_list = []
            capture, matches = (False, [])

        set_current_forward(
            SnapshotForwardContext(
                global_step=next_global_step(),
                is_prompt=not is_decode,
                capture=capture,
                req_ids=req_ids,
                positions=pos_list,
                matches=matches,
            )
        )

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

        # Multimodal kwargs (vision_embedding_blocks + vision_positions)
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

        with self._snapshot_capture_context(positions, spec_decode_metadata):
            with model_forward_context(self.vllm_config):
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
            if self.is_eagle3_spec:
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
        elif self.is_eagle3_spec:
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

    def _get_dp_padding(
        self,
        num_tokens: int,
        local_max_decode_ctx_len: int = 0,
        is_spec_decode: bool = False,
        local_force_nonspec: bool = False,
    ) -> tuple[int, int, bool, bool]:
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

        Four per-step inputs vary per rank and feed shape decisions; all
        must be reconciled to a single value across the DP group (the last two,
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

        We send all four values (`num_tokens`, `local_max_decode_ctx_len`,
        `is_spec_decode`, `local_force_nonspec`) in a single host-side gloo
        tensor (one RTT instead of one-per-value), since all are needed at the
        same dispatch decision point. Idle dummy ranks contribute `0` for the
        context length and the spec vote; the MAX promotes them to the busy
        rank's values so they dispatch to the same NEFF.

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
        `(0, local_max_decode_ctx_len, False, local_force_nonspec)` so
        callers always get a usable value (the local vote is authoritative
        when there's no group to reconcile with).

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

        Returns:
            `(num_pad_tokens, max_decode_ctx_len, any_rank_spec_decode,
            group_force_nonspec)` — additional rows this rank must pad to match
            the busiest rank, the DP-MAX of all ranks' local context lengths
            for downstream bucket selection, whether any rank is speculating,
            and whether the group must fall back to the non-spec decode NEFF.
        """
        # TODO: Leverage vLLM's forward_context DPMetadata for DP coordination
        # Why: vLLM has built-in DP coordination via forward_context.py that could replace/enhance this logic
        # See vllm/forward_context.py:
        # - DPMetadata.num_tokens_across_dp() for token count synchronization via all_reduce
        # - DPMetadata.should_ubatch_across_dp() for coordinated microbatching decisions for PP cases
        # - Automatic integration via set_forward_context() when parallel_config.data_parallel_size > 1
        if not self._dp_collectives_active():
            return 0, local_max_decode_ctx_len, False, local_force_nonspec

        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group().cpu_group
        t = torch.tensor(
            [
                num_tokens,
                local_max_decode_ctx_len,
                int(is_spec_decode),
                int(local_force_nonspec),
            ],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
        max_tokens_across_dp, max_decode_ctx_len, any_rank_spec, group_force_nonspec = (
            int(t[0]),
            int(t[1]),
            bool(t[2]),
            bool(t[3]),
        )
        num_pad = max_tokens_across_dp - num_tokens

        logger.debug(
            "DP coord: rank=%s local_tokens=%s max_tokens_across_dp=%s num_pad=%s "
            "local_max_decode_ctx_len=%s max_decode_ctx_len=%s "
            "any_rank_spec_decode=%s group_force_nonspec=%s",
            self.vllm_config.parallel_config.data_parallel_rank,
            num_tokens,
            max_tokens_across_dp,
            num_pad,
            local_max_decode_ctx_len,
            max_decode_ctx_len,
            any_rank_spec,
            group_force_nonspec,
        )
        return num_pad, max_decode_ctx_len, any_rank_spec, group_force_nonspec

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
        """
        # Only TP rank 0 writes (avoids duplicates within DP group)
        tp_group = get_tp_group()
        if tp_group and tp_group.rank_in_group != 0:
            return

        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
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

    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        When using Data Parallel (DP), if this rank has no real requests but
        other DP ranks do, this method runs a minimal dummy forward pass to
        keep all ranks synchronized during collective operations.

        This is called automatically by the vLLM engine when DP workload
        imbalance is detected (see vllm/v1/engine/llm_engine.py).

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
        # has no real ctx (local_max_decode_ctx_len=0) and can't itself force
        # non-spec (local_force_nonspec=False); the MAX-reduce promotes us to
        # whatever the busy ranks picked — batch size, decode ctx bucket,
        # whether any rank speculates, and whether the group fell back to
        # non-spec — so we dispatch the same NEFF.
        num_pad, max_decode_ctx_len, any_rank_spec, group_force_nonspec = (
            self._get_dp_padding(
                batch_size,
                local_max_decode_ctx_len=0,
                is_spec_decode=False,
                local_force_nonspec=False,
            )
        )

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
        # match the exact traced graph signature including all kwargs.
        with model_forward_context(self.vllm_config):
            model_output = self.model(**decode_kwargs)
        if self.use_async_scheduling:
            # The dummy batch produces no token consumed by anyone; its only
            # purpose is to make this idle DP rank participate in the
            # cross-DP EP collective inside the NEFF. The backend slices the
            # input-output-aliased KV cache out of the returned tuple
            # (compile/backend.py: `return outputs[0:original_output_count]`),
            # so `model_output` is the only live reference to the NEFF's
            # NrtaFuture; if it is dropped, GC frees the in-flight NRT input
            # slices. Hold a Python reference to keep that invariant without
            # a device->host readback of an output nobody uses (the readback
            # would otherwise land on the critical path between consecutive
            # decode NEFFs). The slices stay alive until the next execute()
            # supersedes the future. Cross-DP step lockstep is provided by
            # the per-step _get_dp_padding reduce and its reuse cache, so the
            # readback is not needed for correctness.
            self._dummy_output_keepalive = model_output

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
            if not self.on_device_sampling:
                logger.info("Using vLLM Sampler")
                sampler_output = self.sampler(
                    logits=model_output_tensor,
                    sampling_metadata=sampling_metadata,
                )
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
        # TODO: now only supports EAGLE speculative decoding
        # Add more when needed
        assert isinstance(self.drafter, EagleProposer)

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
            logitsprocs=None,
            logitsprocs_need_output_token_ids=False,
            is_pooling_model=self.is_pooling_model,
        )

        # Models whose groups have different per-layer layouts on one shared
        # buffer must be page-major. vLLM hands the i-th layer of every KV-cache
        # group the same raw buffer and requires block ``b`` to live inside page
        # ``b`` for all of them; a flat "all of state A, then all of state B"
        # layout only works when a layer owns its buffer outright. On a shared
        # buffer it runs through the other groups' pages, which is silent
        # corruption rather than an error.
        page_major = bool(getattr(self.model, "kv_cache_page_major", False))
        page_bytes = max(
            group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups
        )

        # Initialize the KV Cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            # Two extra pages per buffer, both private to the model because
            # vLLM only knows about ``num_blocks``. The last is the write sink
            # for padded rows; the one before it is the zero page they read, and
            # nothing ever writes it, so it still holds the zeros this
            # allocation puts there. vLLM's own null block cannot serve as the
            # zero page -- warmup leaves data in it.
            size = tensor.size + (2 * page_bytes if page_major else 0)
            raw_tensor = torch.zeros(size, dtype=torch.int8, device=self.device)
            # Case where the KV cache is shared across layers
            for layer_name in tensor.shared_by:
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

                    if page_major:
                        # [pages, 2, kv_heads, block_size, head_size]: natural
                        # strides, and page ``b`` is exactly the bytes vLLM
                        # reserved for block ``b``. K is column 0, V column 1.
                        used = (
                            2
                            * num_kv_heads
                            * block_size
                            * head_size
                            * get_dtype_size(kv_cache_spec.dtype)
                        )
                        if used != page_bytes:
                            raise NotImplementedError(
                                f"page-major attention needs the K/V bytes "
                                f"({used}) to fill the shared page "
                                f"({page_bytes}); reaching only part of a page "
                                f"would need a strided view, and Neuron device "
                                f"tensors reject those"
                            )
                        if k_is_packed:
                            raise NotImplementedError(
                                "page-major layout with an FP8-packed K cache "
                                "is not implemented"
                            )
                        num_pages = raw_tensor.numel() // page_bytes
                        kv_pages = _shared_dtype_view(
                            raw_tensor, kv_cache_spec.dtype
                        ).view(num_pages, 2, num_kv_heads, block_size, head_size)
                        kv_caches[layer_name] = [kv_pages]
                        self._kv_cache_full_tensors[layer_name] = kv_pages
                        continue

                    kv_shape = (2, num_blocks, num_kv_heads, block_size, head_size)
                    typed_tensor = _shared_dtype_view(
                        raw_tensor, kv_cache_spec.dtype
                    ).view(kv_shape)
                    # K and V are backed by the same (2, ...) storage; only
                    # the K slice is reshaped to the packed layout when FP8
                    # packing is enabled (same numel, different view).
                    k_cache = typed_tensor[0]
                    if k_is_packed:
                        k_cache = k_cache.view(k_cache_shape)
                    kv_caches[layer_name] = [k_cache, typed_tensor[1]]
                    # Store the full (2, ...) tensor for DI connector registration;
                    # typed_tensor[0] and typed_tensor[1] are views into it.
                    self._kv_cache_full_tensors[layer_name] = typed_tensor

            # Recurrent-state layers (gated DeltaNet / Mamba).
            #
            # vLLM's own gpu_model_runner strides each state tensor out of the
            # layer's raw buffer with the block dim striding by a whole page, so
            # each block stays page-aligned and one block's states sit together
            # inside its page. **Neuron device tensors reject that**: the runtime
            # raises "Detected non-contiguous slicing for requested Device
            # Tensor" for any view whose stride is not the natural one.
            #
            # Two layouts follow, and which one a model gets is its choice.
            #
            # Default: lay the states out one after another -- every block's conv
            # state, then every block's recurrent state -- which leaves each
            # tensor contiguous. Safe only because nothing outside the model
            # reads them: vLLM's stake in the layout is the *page size*, which is
            # unchanged, so the planner's block accounting still holds, and its
            # per-block state-copy hooks are inactive at
            # ``mamba_cache_mode == "none"``. Revisit if prefix caching over
            # recurrent state is ever enabled -- those hooks are what it needs.
            #
            # ``page_major``: one view per buffer at full page width, states
            # packed inside each page in declaration order. Needed when the
            # buffer is *shared* -- vLLM hands one raw buffer to one layer from
            # each KV-cache group -- because only a full-page-width view of a
            # page-major buffer is contiguous. Two consequences the model must
            # own. One dtype per page, so narrower states are upcast into the
            # widest one's columns rather than ``state_dtypes()`` changing. And
            # block ids, though allocated per group, are global: a page that held
            # another group's float32 state comes back as this group's, so the
            # model must treat any byte it did not write this step as hostile
            # rather than merely stale. Reading such a page back as bf16 yields
            # infinities and NaNs; masking them needs a select, since ``inf * 0``
            # is NaN -- which is why this only shows up above
            # ``max_num_seqs == 1``, where padded rows first appear.
            elif isinstance(kv_cache_spec, MambaSpec):
                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

                    if page_major:
                        # One view, one dtype: a page-major view is only
                        # contiguous at full page width, so every state shares
                        # the widest dtype's element size and the narrower ones
                        # are upcast by the model. The states sit in declaration
                        # order inside the page, and the model recomputes the
                        # same column offsets from its own shapes.
                        elem = max(get_dtype_size(d) for d in kv_cache_spec.dtypes)
                        view_dtype = next(
                            d
                            for d in kv_cache_spec.dtypes
                            if get_dtype_size(d) == elem
                        )
                        used = elem * sum(
                            math.prod(shape) for shape in kv_cache_spec.shapes
                        )
                        if used > page_bytes or page_bytes % elem:
                            raise NotImplementedError(
                                f"page-major recurrent states need {used} bytes "
                                f"inside a {page_bytes}-byte page at {elem}-byte "
                                f"alignment; widen the attention block size"
                            )
                        num_pages = raw_tensor.numel() // page_bytes
                        kv_caches[layer_name] = [
                            _shared_dtype_view(raw_tensor, view_dtype).view(
                                num_pages, page_bytes // elem
                            )
                        ]
                        continue

                    state_tensors = []
                    offset_bytes = 0
                    for shape, dtype in zip(
                        kv_cache_spec.shapes, kv_cache_spec.dtypes
                    ):
                        dtype_size = get_dtype_size(dtype)
                        assert offset_bytes % dtype_size == 0, (
                            f"state offset {offset_bytes} is misaligned for "
                            f"{dtype}; order the shapes widest-dtype-first"
                        )
                        numel = num_blocks * math.prod(shape)
                        start = offset_bytes // dtype_size
                        state_tensors.append(
                            raw_tensor.view(dtype)[start : start + numel].view(
                                num_blocks, *shape
                            )
                        )
                        offset_bytes += numel * dtype_size

                    assert offset_bytes <= raw_tensor.numel()
                    kv_caches[layer_name] = state_tensors

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
                        kv_shape = (2, num_blocks, num_kv_heads, block_size, head_size)
                    else:
                        raise NotImplementedError(
                            f"Unsupported Attention spec type: {type(kv_cache_layer_spec)}"
                        )

                    typed_tensor = _shared_dtype_view(
                        raw_tensor, kv_cache_layer_spec.dtype
                    ).view(kv_shape)
                    kv_caches[layer_name] = [typed_tensor[0], typed_tensor[1]]  # [k, v]
                    self._kv_cache_full_tensors[layer_name] = typed_tensor

            else:
                raise NotImplementedError(
                    f"Unsupported Attention spec type: {type(kv_cache_spec)}"
                )

        # This binds the cache tensors to the model
        self.model.bind_kv_cache(kv_caches)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # This binds the cache tensors to the draft model
            self.drafter.model.bind_kv_cache(kv_caches)
            # validate all draft model layers belong to the same kv cache group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

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
            if layer.sliding_window_size is None:
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

        # Hybrid models (e.g. Qwen3.5: 18 gated-DeltaNet + 6 attention layers)
        # additionally report layers that keep fixed-size recurrent state instead
        # of a KV cache. vLLM puts those in their own KV cache group.
        #
        # block_size and page_size_padded come from cache_config, which vLLM has
        # already filled in for us: ModelConfig.is_hybrid is derived from the HF
        # config, and Platform._align_hybrid_block_size then sets
        # mamba_block_size and mamba_page_size_padded. Do not recompute either --
        # the planner has already sized the pages from those values.
        for rec_layer in getattr(target_kv_spec, "recurrent_layers", ()):
            mamba_block_size = self.vllm_config.cache_config.mamba_block_size
            if mamba_block_size is None:
                raise ValueError(
                    "cache_config.mamba_block_size is unset but the model "
                    f"reports recurrent layer {rec_layer.name!r}. vLLM normally "
                    "sets it for hybrid models; check that ModelConfig.is_hybrid "
                    "is True for this checkpoint."
                )
            all_kv_cache_specs[rec_layer.name] = MambaSpec(
                shapes=rec_layer.shapes,
                dtypes=rec_layer.dtypes,
                block_size=mamba_block_size,
                page_size_padded=(
                    self.vllm_config.cache_config.mamba_page_size_padded
                ),
                mamba_cache_mode=self.vllm_config.cache_config.mamba_cache_mode,
            )

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

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
        encoding when images exceed the max vision bucket). Each step
        adds newly-encoded entries without discarding previous ones,
        so get_encoder_cache() returns all images for the full request.
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
        # encoding where remaining images are encoded in later steps).
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
        """Return encoder cache contents (vision embeddings) as serialized bytes.

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
        if self._vision_tensor_capture is not None:
            self._vision_tensor_capture.enable()
        if self._draft_tensor_capture is not None:
            self._draft_tensor_capture.enable()

    def disable_capture(self) -> None:
        """Disable tensor capture writing to disk."""
        if self._target_tensor_capture is not None:
            self._target_tensor_capture.disable()
        if self._vision_tensor_capture is not None:
            self._vision_tensor_capture.disable()
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
        engine/API layer mounts the matching endpoints (e.g. /v1/embeddings);
        otherwise reports generation.

        Returns:
            Tuple of supported tasks
        """
        if self.vllm_config.model_config.runner_type == "pooling":
            return tuple(self.get_model().pooler.get_supported_tasks())
        return ("generate",)

    def ensure_kv_transfer_shutdown(self) -> None:
        pass
