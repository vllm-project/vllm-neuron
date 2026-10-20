# SPDX-License-Identifier: Apache-2.0
"""
Neuron Scheduler implementation for vLLM

This module provides a custom scheduler optimized for Neuron hardware.
It uses a holdback queue pattern for cleaner request management and
bucket-aware admission control.
"""

import logging
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING


from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_neuron import envs
from vllm_neuron.metrics import (
    NUM_BATCHED_TOKENS_PADDING,
    NUM_SEQS_PADDING,
    SCHEDULER_ENCODER_CACHE_HITS,
    SCHEDULER_ENCODER_CACHE_QUERIES,
)
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig
from vllm_neuron.utils.bucket_utils import (
    get_bucket_for_count,
    get_default_kv_segment_size_buckets,
    get_default_num_seqs_buckets,
    get_max_num_batched_tokens,
    resolve_num_batched_tokens_buckets,
    validate_num_seqs_buckets,
)
from vllm_neuron.vllm.patches.jf_grammar_patch import (
    apply_jf_grammar_rollback_patch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal import MultiModalRegistry
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager

logger = logging.getLogger(__name__)


def _kv_capacity_guidance() -> str:
    """Return capacity guidance for the active KV budget policy."""
    headroom_env = envs.get_kv_headroom_env()
    if headroom_env is not None:
        return f"reduce {headroom_env} or reduce max_model_len."
    return (
        "increase gpu_memory_utilization or "
        "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION, or reduce max_model_len."
    )


class SchedulerState(Enum):
    """Scheduler state for Neuron platform.

    State machine:
        IDLE → ACTIVE_PREFILL → ACTIVE_DECODE → IDLE
                     ↑    ↓           ↑    ↓
                     └────┘           └────┘
                           ↑          ↓
                           └──────────┘

    Transitions:
        - IDLE → ACTIVE_PREFILL: Prefill request arrives
        - ACTIVE_PREFILL → ACTIVE_PREFILL: More segments
        - ACTIVE_PREFILL → ACTIVE_DECODE: Prefill completes
        - ACTIVE_DECODE → ACTIVE_DECODE: Decode continues
        - ACTIVE_DECODE → ACTIVE_PREFILL: Decode has capacity AND prefill waiting
        - ACTIVE_DECODE → IDLE: All decode complete AND no prefill waiting

    States:
        IDLE: No requests in system (waiting=[], running=[])
        ACTIVE_PREFILL: Last schedule() call scheduled prefill.
        This state indicates there are prefill request(s) in waiting or running queue,
        and we don't have a full batch of decode requests
        ACTIVE_DECODE: Last schedule() call scheduled decode,
        This state indicates the running queue only contains decode requests

    """

    IDLE = auto()
    ACTIVE_PREFILL = auto()
    ACTIVE_DECODE = auto()

    def __str__(self) -> str:
        return self.name


class NeuronScheduler(Scheduler):
    """
    Scheduler for Neuron platform with prefill/decode separation.

    This scheduler extends vLLM's Scheduler with Neuron-specific
    constraints:
    - Holdback queue for managing request scheduling constraints
    - Prefill/decode separation (Neuron hardware requirement)
    - Token padding for compiled model bucket sizes
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: "MultiModalRegistry | None" = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Neuron async scheduler.

        Args:
            vllm_config: vLLM configuration object.
            kv_cache_config: KV cache configuration.
            structured_output_manager: Manager for structured outputs.
            block_size: Size of KV cache blocks.
            mm_registry: Multimodal registry (optional).
            include_finished_set: Whether to include finished requests in output.
            log_stats: Whether to log scheduler statistics.
        """
        if mm_registry is None:
            mm_registry = MULTIMODAL_REGISTRY

        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.log_stats = log_stats
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        self.model_name = vllm_config.model_config.model

        self._instrument_encoder_cache_manager()
        self._cap_encoder_budget_to_bucket(vllm_config)

        # TODO: Support additional scheduling policies
        scheduler_policy = vllm_config.scheduler_config.policy
        if scheduler_policy != "fcfs":
            raise ValueError(
                f"NeuronAsyncScheduler currently only supports 'fcfs' scheduling policy, "
                f"got '{scheduler_policy}'."
            )

        # Holdback queue: requests temporarily held back from the base scheduler
        # to enforce Neuron-specific constraints
        self.holdback_queue: deque[Request] = deque()

        # The async-KV last-token-recompute reclassify (see schedule()) is only
        # needed for the isolated-decode bench connector, which reports the whole
        # prompt as external KV. Regular DI (NixlConnector) does not need it, so
        # gate on the specific connector name rather than "any connector".
        kv_transfer_config = vllm_config.kv_transfer_config
        self._is_decode_bench_connector = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "NeuronDecodeBenchConnector"
        )

        # Current scheduler state (updated after each schedule() call)
        self._state: SchedulerState = SchedulerState.IDLE
        self._kv_exhaustion_warned: bool = False

        # Prefill-async pipeline (pure kv_producer only -- prefill-only, decode is
        # remote; kv_both is excluded because it decodes locally). A do_remote_decode
        # producer whose prefill is fully dispatched is parked in this send-lane:
        # out of the batch-1 compute slot -- so schedule() can admit the next
        # prefill -- but still holding its KV blocks until the NIXL handoff lands.
        # Needed because under async scheduling schedule(N+1) runs before
        # update_from_output(N), so base's remove-on-stop would free the slot a
        # step too late. Stays empty on non-producer engines, so the send-lane
        # touch points below are all no-ops there.
        self._pure_producer = (
            kv_transfer_config is not None
            and kv_transfer_config.is_kv_producer
            and not kv_transfer_config.is_kv_consumer
        )
        self._pending_kv_send: set[str] = set()
        # Keep the running decodes when every admissible waiter is a pure asynchronous KV pull.
        # Branch 3.2 hides all running decodes so a waiter can prefill, but a pure pull needs no
        # forward pass: the base scheduler parks it in WAITING_FOR_REMOTE_KVS with zero tokens.
        # Hiding then costs the ready decodes a whole step and buys nothing.
        #
        # Only a consumer receives such a pull, so gate on the role here; a per-step gate below
        # decides whether the actual waiters qualify. ``is_kv_consumer`` is also true for
        # kv_both, which produces as well and can hold a local prefill that does need the
        # holdback, so exclude a producer too and keep this to a pure decode pool.
        self._skip_decode_holdback = (
            kv_transfer_config is not None
            and kv_transfer_config.is_kv_consumer
            and not kv_transfer_config.is_kv_producer
        )
        logger.info(
            "Decode holdback skip: %s. A pure KV consumer keeps the running decodes when "
            "every admissible waiter is an asynchronous KV pull.",
            "enabled" if self._skip_decode_holdback else "disabled",
        )
        # One-shot flag: warn once if a producer ever arrives with max_tokens>1
        # (a P/D-contract breach the send-lane must not park -- see
        # _is_dispatched_producer).
        self._warned_multitoken_producer: bool = False

        # Jump-forward: tokens identified post-decode, dispatched on next step
        self._pending_jf_tokens: dict[str, list[int]] = {}
        self._jf_total_injections: int = 0
        self._jf_total_tokens_injected: int = 0

        # Get max_jump_forward_tokens from config
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        parsed_neuron_config = NeuronConfig.from_dict(neuron_config)
        self.max_jump_forward_tokens: int = parsed_neuron_config.max_jump_forward_tokens
        # Long-prompt routing: requests with longer prompts decode without
        # speculation. See _route_long_prompts_off_spec.
        self.spec_decode_max_prompt_len: int | None = (
            parsed_neuron_config.spec_decode_max_prompt_len
        )

        #  Roll back so upstream base update_from_output can re-accept [jf.., bonus].
        if self.max_jump_forward_tokens > 0:
            apply_jf_grammar_rollback_patch(
                self.structured_output_manager, self.max_jump_forward_tokens
            )

        max_num_batched_tokens = get_max_num_batched_tokens(
            vllm_config.scheduler_config.max_num_batched_tokens, self.max_model_len
        )

        # The two axes are independent, but only when the user says so. Resolve
        # the KV segment default first, then collapse the Q buckets onto it when
        # both are defaulted. The model runner does the same, and it compiles
        # the prefill targets. The scheduler must select the same bucket list,
        # or it dispatches a bucket that the runner never compiled. DCP
        # alignment still applies to whichever bucket list is selected.
        user_set_num_batched_tokens_buckets = (
            "num_batched_tokens_buckets" in neuron_config
        )
        user_set_kv_segment_size_buckets = "kv_segment_size_buckets" in neuron_config
        default_kv_segment_size_buckets = get_default_kv_segment_size_buckets(
            max_num_batched_tokens, self.max_model_len
        )
        auto_num_batched_tokens_buckets = (
            default_kv_segment_size_buckets
            if not user_set_kv_segment_size_buckets
            else None
        )

        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        dcp_stride = (
            dcp_size * vllm_config.cache_config.block_size if dcp_size > 1 else 1
        )
        self.num_batched_tokens_buckets = resolve_num_batched_tokens_buckets(
            max_num_batched_tokens,
            configured_buckets=neuron_config.get("num_batched_tokens_buckets"),
            use_configured_buckets=user_set_num_batched_tokens_buckets,
            auto_buckets=auto_num_batched_tokens_buckets,
            dcp_stride=dcp_stride,
        )

        if user_set_num_batched_tokens_buckets:
            logger.info(
                "Using num_batched_tokens_buckets from neuron_config: %s",
                self.num_batched_tokens_buckets,
            )
        else:
            logger.info(
                "Using default num_batched_tokens_buckets: %s",
                self.num_batched_tokens_buckets,
            )

        # Prefill batch size limit (currently 1, may expand in future)
        self.max_prefills_per_batch: int = 1

        # Parse num_seqs_buckets for decode batch size buckets
        if "num_seqs_buckets" in neuron_config:
            self.num_seqs_buckets = validate_num_seqs_buckets(
                neuron_config["num_seqs_buckets"],
                self.max_num_seqs,
            )
        else:
            self.num_seqs_buckets = get_default_num_seqs_buckets(self.max_num_seqs)

        # Pre-initialize metric labels so they appear at /metrics immediately
        for bucket_size in self.num_batched_tokens_buckets:
            bucket_name = f"prefill_s{bucket_size}"
            NUM_BATCHED_TOKENS_PADDING.labels(
                model_name=self.model_name, bucket_name=bucket_name
            )
        for bucket_size in self.num_seqs_buckets:
            bucket_name = f"decode_b{bucket_size}"
            NUM_SEQS_PADDING.labels(model_name=self.model_name, bucket_name=bucket_name)

        # Padding statistics
        self.total_padding_tokens: int = 0
        self.total_scheduled_tokens: int = 0

        # Warn if max_num_seqs exceeds KV cache capacity.
        # Use the same calculation as upstream vLLM (accounts for HMA groups,
        # sliding window, etc.) so the number matches the "Maximum concurrency"
        # log line from vllm/v1/core/kv_cache_utils.py.
        if kv_cache_config.kv_cache_groups:
            from vllm.v1.core.kv_cache_utils import (
                get_max_concurrency_for_kv_cache_config,
            )

            max_concurrency = get_max_concurrency_for_kv_cache_config(
                vllm_config, kv_cache_config
            )
            max_concurrent = int(max_concurrency)
        else:
            # Fallback for attention-free models or empty groups
            total_kv_tokens = kv_cache_config.num_blocks * block_size
            max_concurrent = (
                total_kv_tokens // self.max_model_len if self.max_model_len > 0 else 0
            )

        self._max_kv_concurrent = (
            max_concurrent if max_concurrent > 0 else self.max_num_seqs
        )

        if 0 < max_concurrent < self.max_num_seqs:
            logger.warning(
                "max_num_seqs (%d) exceeds worst-case KV cache capacity (%d "
                "concurrent requests at max_model_len=%d, assuming no prefix "
                "cache hits). Effective concurrency will be capped at %d to "
                "prevent KV cache exhaustion and request preemption. To "
                "increase concurrency, %s",
                self.max_num_seqs,
                max_concurrent,
                self.max_model_len,
                max_concurrent,
                _kv_capacity_guidance(),
            )

        logger.info("Initialized NeuronAsyncScheduler for Neuron platform")
        logger.info("Max prefills per batch: %d", self.max_prefills_per_batch)

        # Maps mm_hash -> encoder-cache locator dict, carrying the locator from
        # the worker's ModelRunnerOutput (read in update_from_output) to the
        # response (emitted in _free_request). See those methods.
        #
        # TODO(vLLM PR #40695): this map and the update_from_output / _free_request
        # overrides below are an interim shim. When #40695 (ec_transfer_params on
        # EngineCoreOutput + scheduler-driven ec_connector.request_finished) lands
        # in the pinned vLLM, delete all three and use the stock path.
        self._ec_locator_by_mm_hash: dict[str, dict] = {}

    def update_from_output(self, scheduler_output, model_runner_output):
        """Process output. Additionally, check if we can advance
        reqs with JF tokens in next iteration.
        """
        ec_out = getattr(model_runner_output, "ec_connector_output", None)
        worker_meta = getattr(ec_out, "ec_connector_worker_meta", None)
        embed_locator = getattr(worker_meta, "embed_locator", None)
        if embed_locator:
            self._ec_locator_by_mm_hash.update(embed_locator)

        # Rollback grammar by accepted JF tokens so we don't double accept.
        jf_decode_tokens = getattr(scheduler_output, "_jf_decode_tokens", None)
        if jf_decode_tokens:
            for req_id, jf_tokens in jf_decode_tokens.items():
                if not jf_tokens:
                    continue
                request = self.requests.get(req_id)
                if request is None:
                    continue
                if not self.structured_output_manager.should_advance(request):
                    continue
                so_req = request.structured_output_request
                grammar = so_req.grammar if so_req is not None else None
                if grammar is not None:
                    grammar.rollback(len(jf_tokens))

        result = super().update_from_output(scheduler_output, model_runner_output)

        # Prefill-async pipeline: retire producers from the send-lane once their
        # KV send completes. super() above frees a finished-sending req and deletes
        # it from self.requests (lease expiry surfaces the same way), so
        # intersecting the lane with self.requests retires both normal and
        # aborted/expired sends in one shot.
        if self._pending_kv_send:
            self._pending_kv_send.intersection_update(self.requests)

        # Drop locators whose cache blocks the engine freed this step.
        freed = getattr(scheduler_output, "free_encoder_mm_hashes", None)
        if freed:
            for mm_hash in freed:
                self._ec_locator_by_mm_hash.pop(mm_hash, None)

        # Check if we can advance req with JF tokens
        if self.max_jump_forward_tokens > 0:
            tokenizer = getattr(self.structured_output_manager, "tokenizer", None)
            if tokenizer:
                for req_id in scheduler_output.num_scheduled_tokens:
                    # Skip if already has pending JF or is a JF step itself
                    if req_id in self._pending_jf_tokens:
                        continue
                    if getattr(scheduler_output, "_jf_decode_tokens", None):
                        if req_id in scheduler_output._jf_decode_tokens:
                            continue
                    request = self.requests.get(req_id)
                    if request is None or request.is_finished():
                        continue
                    if request.num_computed_tokens < request.num_prompt_tokens:
                        continue
                    if not self.structured_output_manager.should_advance(request):
                        continue
                    struct_output_request = request.structured_output_request
                    if struct_output_request is None:
                        continue
                    grammar = struct_output_request.grammar
                    if grammar is None or grammar.is_terminated():
                        continue

                    jf_tokens = self._compute_jump_forward_tokens(
                        grammar, tokenizer, list(request.output_token_ids)
                    )
                    if jf_tokens:
                        self._pending_jf_tokens[req_id] = jf_tokens

        return result

    def add_request(self, request: Request) -> None:
        """Reject requests when we have logprobs and JF enabled."""
        if self.max_jump_forward_tokens > 0 and request.use_structured_output:
            sampling_params = request.sampling_params
            if sampling_params is not None and (
                sampling_params.logprobs is not None
                or sampling_params.prompt_logprobs is not None
            ):
                raise ValueError(
                    "Structured-output requests cannot request logprobs while "
                    "jump-forward decoding is enabled (max_jump_forward_tokens="
                    f"{self.max_jump_forward_tokens})."
                )
        super().add_request(request)

    def _free_request(self, request, delay_free_blocks: bool = False):
        """Attach this request's EC locators onto kv_transfer_params for the PD.

        The base _free_request returns a (kv_transfer_params, ec_transfer_params)
        tuple that the scheduler unpacks onto EngineCoreOutput; we merge this
        request's stashed locators under an "ec_locator" key in kv_transfer_params
        so they ride that already-wired field to the router/PD. Borrowing the
        field is not using the KV connector slot.

        TODO(vLLM PR #40695): remove this override (see __init__).
        """
        # clear jf tokens at req_id for removed request.
        self._pending_jf_tokens.pop(request.request_id, None)

        kv_xfer_params, ec_xfer_params = super()._free_request(
            request, delay_free_blocks
        )
        if not self._ec_locator_by_mm_hash:
            return kv_xfer_params, ec_xfer_params
        # KNOWN LIMITATION (prototype): the emitted locator is not guaranteed
        # fresh at consume time -- the referenced VE block may be evicted/reused
        # before the PD pulls it (no transfer-completion pin; we rely on the VE
        # cache's min_hold_time window). The prod fix (stamp the block with
        # mm_hash so the PD validates the embedding, plus pin until consumed) is
        # planned for next milestone.
        ec_locator = {}
        for feature in getattr(request, "mm_features", None) or []:
            mm_hash = getattr(feature, "identifier", None)
            if mm_hash is not None and mm_hash in self._ec_locator_by_mm_hash:
                ec_locator[mm_hash] = self._ec_locator_by_mm_hash[mm_hash]
        if not ec_locator:
            return kv_xfer_params, ec_xfer_params
        params = dict(kv_xfer_params) if kv_xfer_params else {}
        params["ec_locator"] = ec_locator
        return params, ec_xfer_params

    def _instrument_encoder_cache_manager(self) -> None:
        """Wrap ``EncoderCacheManager.check_and_update_cache`` to emit metrics.

        This is the scheduler-level encoder cache boundary: the base
        ``_try_schedule_encoder_inputs`` calls ``check_and_update_cache(request,
        i)`` for each mm item, and a ``True`` return means the item was already
        encoded so the scheduler skips re-scheduling the encoder — a
        genuine encoder-cache hit (unlike the worker/device boundary, where the
        scheduler has already filtered hits out, so hits there are ~always 0).

        We wrap the bound method on the inherited manager instance rather than
        subclassing so we don't fork upstream scheduling logic. Guarded so a
        vLLM change that removes/renames the method degrades to a no-op instead
        of crashing startup.

        NOTE: coupled to upstream ``vllm.v1.core.sched.scheduler.Scheduler.
        _try_schedule_encoder_inputs`` calling ``check_and_update_cache``; if
        that entry point moves, these counters silently stop incrementing.
        """
        ecm = getattr(self, "encoder_cache_manager", None)
        orig = getattr(ecm, "check_and_update_cache", None)
        if orig is None:
            logger.warning(
                "EncoderCacheManager.check_and_update_cache not found; "
                "scheduler encoder cache metrics disabled."
            )
            return

        def _counting_check(request: "Request", input_id: int) -> bool:
            SCHEDULER_ENCODER_CACHE_QUERIES.inc()
            hit = orig(request, input_id)
            if hit:
                SCHEDULER_ENCODER_CACHE_HITS.inc()
            return hit

        ecm.check_and_update_cache = _counting_check
        logger.info("Instrumented scheduler encoder cache hit/query metrics.")

    def _cap_encoder_budget_to_bucket(self, vllm_config: "VllmConfig") -> None:
        """Limit encoder inputs per step so they fit one compiled encoder bucket.

        The encoder is compiled for fixed bucket sizes. If the scheduler sends
        more than the largest bucket can hold, embed_multimodal overflows.
        Capping the budget makes the base scheduler defer the excess to later
        prefill steps automatically.

        Budget (max_num_encoder_input_tokens) is embed (post-merge) token space.
        The bucket's capacity in that space is modality-specific, so the model
        computes it (SupportsEncoderBudget.get_max_encoder_embed_budget) — vision
        and audio state their buckets in different units.
        """
        from vllm_neuron.utils.mm_utils import get_max_encoder_embed_budget

        encoder_config_dict = vllm_config.additional_config.get("encoder_neuron_config")
        if not encoder_config_dict or self.max_num_encoder_input_tokens <= 0:
            return

        encoder_neuron_config = EncoderNeuronConfig.from_dict(encoder_config_dict)
        cap = get_max_encoder_embed_budget(
            vllm_config.model_config.hf_config, encoder_neuron_config
        )
        if cap is None:
            return
        cap = max(1, cap)

        if self.max_num_encoder_input_tokens > cap:
            logger.info(
                "Capping encoder_compute_budget from %d to %d embeds (max_bucket=%d)",
                self.max_num_encoder_input_tokens,
                cap,
                max(encoder_neuron_config.num_encoder_input_buckets),
            )
            self.max_num_encoder_input_tokens = cap

    @property
    def state(self) -> SchedulerState:
        """Get the current scheduler state.

        State is updated at the end of each schedule() call based on
        the scheduler output, using num_computed_tokens vs num_prompt_tokens to
        determine prefill vs decode phase.

        Returns:
            IDLE: No requests in system (waiting=[], running=[])
            ACTIVE_PREFILL: Last schedule() call scheduled prefill.
            This state indicates there are prefill request(s) in waiting or running queue,
            and we don't have a full batch of decode requests
            ACTIVE_DECODE: Last schedule() call scheduled decode,
            This state indicates the running queue only contains decode requests
        """
        return self._state

    @property
    def at_capacity(self) -> bool:
        """Check if the running queue is at capacity. The running queue
        may contain either prefill or decode requests.

        When at capacity, new prefill requests are blocked until decode
        requests complete and free up slots.

        Returns:
            True if running requests >= max_num_seqs, False otherwise.
        """
        return len(self.running) >= self.max_num_seqs

    @property
    def has_prefill_in_running(self) -> bool:
        """Check if any requests in the running queue are prefills

        When there are ongoing prefill requests in the running queue,
        new prefill requests are blocks and ongoing decode requests
        are held back from the base scheduler.

        Returns:
            True if any requests in the running queue is in prefill
            phase, False otherwise.
        """
        return any(self._is_prefill_request(req) for req in self.running)

    def _admissible_waiters_are_async_kv_pulls(self) -> bool:
        """Report whether every waiter the base scheduler could admit is a pure KV pull.

        A pure pull schedules no local prefill work. The holdback exists so a prefill can own the
        batch alone, so skipping it is only safe when the admitted request runs no forward pass.

        ``load_kv_async`` decides that, but the base scheduler computes it from
        ``connector.get_num_new_matched_tokens`` after this point, so read the request's declared
        intent instead: both NIXL schedulers return ``load_async=True`` exactly when
        ``do_remote_prefill`` is set and the remaining pull count is above zero. The one-token
        margin covers the connector's sliding-window decode path, which decrements that count and
        falls back at zero.

        Conservative by design: any doubt gives False and the caller keeps the holdback, which is
        always correct and only slower.

        Returns:
            True when every admissible waiter is a pure asynchronous KV pull.
        """
        if not self.waiting:
            return False
        for request in self.waiting:
            transfer_params = request.kv_transfer_params
            if not transfer_params or not transfer_params.get(
                "do_remote_prefill", False
            ):
                return False
            # Count the tokens the same way the connector does, from the token ids rather than
            # from num_prompt_tokens. A prompt_embeds request has no token ids, so
            # num_prompt_tokens is the embed count while the connector sees zero and returns
            # load_async=False, which is a real prefill and does need the holdback.
            prompt_token_ids = request.prompt_token_ids
            if not prompt_token_ids:
                return False
            if len(prompt_token_ids) - request.num_computed_tokens <= 1:
                return False
        return True

    def _is_prefill_request(self, request: Request) -> bool:
        """Check if a request is in prefill phase.

        A request is in prefill if it hasn't finished processing all prompt tokens.
        This is independent of which queue (waiting/running) it's in.

        Args:
            request: The request to check.

        Returns:
            True if request is in prefill phase, False if in decode phase.
        """
        return request.num_computed_tokens < request.num_prompt_tokens

    def _kv_in_use(self) -> int:
        """Requests currently pinning KV blocks against the worst-case cap
        (_max_kv_concurrent): those in the running batch plus any producers
        parked in the send-lane awaiting their NIXL handoff."""
        return len(self.running) + len(self._pending_kv_send)

    def _is_dispatched_producer(self, request: Request) -> bool:
        """A do_remote_decode producer past its (final) prefill chunk and safe to
        park: it holds KV blocks awaiting the async send but no longer needs the
        compute slot. Chunk-safe (False until the last prefill chunk is dispatched).

        Guard: a DI producer hands off after one prefill token, so the proxy sends
        max_tokens=1 (which stops it this step). A max_tokens>1 producer would NOT
        stop -- parking it strands it out of self.running forever (nothing
        re-admits it), leaking the send-lane and eventually wedging admission,
        which is worse than base (base just decodes it locally). So it is left in
        running for base to finish, and the contract breach is logged once.
        """
        kvt = request.kv_transfer_params
        if (
            kvt is None
            or not kvt.get("do_remote_decode", False)
            or self._is_prefill_request(request)
        ):
            return False
        if request.max_tokens > 1:
            if not self._warned_multitoken_producer:
                self._warned_multitoken_producer = True
                logger.warning(
                    "do_remote_decode producer request %s has max_tokens=%d (>1); "
                    "the disaggregation contract expects max_tokens=1 on the prefill "
                    "node. Not parking it in the send-lane (leaving it for base to "
                    "finish locally) to avoid a permanent send-lane leak. Check the "
                    "P/D proxy's max_tokens.",
                    request.request_id,
                    request.max_tokens,
                )
            return False
        return True

    def _release_dispatched_producers(self) -> None:
        """Move fully-dispatched producers from the compute slot to the send-lane.

        Called at the end of schedule() (num_computed already advanced), so a
        producer dispatched this step frees the batch-1 slot for the next
        prefill's admission next step, while its KV blocks stay pinned (connector
        delay-free) for the in-flight NIXL send. It stays in self.requests, so
        base update_from_output still finishes it normally.
        """
        for request in list(self.running):
            if self._is_dispatched_producer(request):
                self.running.remove(request)
                self._pending_kv_send.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        """Count send-lane producers as unfinished: their KV transfer is still in
        flight. Base counts only waiting + running; the lane is empty off a
        pure-producer engine, so this is a no-op there."""
        return super().get_num_unfinished_requests() + len(self._pending_kv_send)

    def _update_state(self, scheduler_output: "SchedulerOutput") -> None:
        """Update state based on scheduler output after scheduling.

        State is determined by checking if any scheduled request is in prefill
        phase using the SchedulerOutput's num_computed_tokens. This is different
        from the request.num_computed_tokens which gets incremented before
        going through model runner.

        Args:
            scheduler_output: The output from schedule().
        """
        old_state = self._state

        if len(self.waiting) == 0 and len(self.running) == 0:
            self._state = SchedulerState.IDLE
        elif scheduler_output.num_scheduled_tokens:
            # Check if any scheduled request is in prefill phase
            # Note: Need to use SchedulerOutput data which has pre-increment num_computed_tokens
            # The request.num_computed_tokens will already be incremented at this point
            has_prefill = False

            # Check new requests (NewRequestData has num_computed_tokens)
            for new_req in scheduler_output.scheduled_new_reqs:
                request = self.requests.get(new_req.req_id)
                if request is not None:
                    if new_req.num_computed_tokens < request.num_prompt_tokens:
                        has_prefill = True
                        break

            # Check cached requests if no prefill found yet
            if not has_prefill and scheduler_output.scheduled_cached_reqs is not None:
                cached_reqs = scheduler_output.scheduled_cached_reqs
                for i, req_id in enumerate(cached_reqs.req_ids):
                    request = self.requests.get(req_id)
                    if request is not None:
                        num_computed = cached_reqs.num_computed_tokens[i]
                        if num_computed < request.num_prompt_tokens:
                            has_prefill = True
                            break

            if has_prefill:
                self._state = SchedulerState.ACTIVE_PREFILL
            else:
                self._state = SchedulerState.ACTIVE_DECODE
        # else: keep previous state if nothing was scheduled

        if self._state != old_state:
            logger.debug(
                "State transition: %s -> %s",
                old_state,
                self._state,
            )

    def _fits_in_bucket(self, num_tokens: int) -> bool:
        """Check if token count fits within available buckets.

        Args:
            num_tokens: Number of tokens to check.

        Returns:
            True if tokens fit in any bucket, False otherwise.
        """
        if not self.num_batched_tokens_buckets:
            return True  # No buckets configured, allow all
        return num_tokens <= self.num_batched_tokens_buckets[-1]

    def _calculate_padded_count(self, token_count: int) -> int:
        """Calculate padded token count using bucket list.

        Finds the smallest bucket that can accommodate the token count.

        Args:
            token_count: Original number of tokens.

        Returns:
            Padded token count (from bucket list).
        """
        if token_count == 0:
            logger.warning("Request has zero tokens, skipping padding")
            return 0

        return get_bucket_for_count(token_count, self.num_batched_tokens_buckets)

    def can_schedule(self, request: Request) -> bool:
        """Admission control for scheduling requests.

        Determines if a new request can be scheduled based on:
        - Scheduler state
        - Decode batch capacity
        - Prefill batch capacity (max 1 prefill at a time)
        - KV cache concurrency cap (worst-case max_model_len per request)

        Note: We don't check bucket fit here because chunked prefill allows
        long prompts to be processed in multiple iterations. Each chunk will
        fit in a bucket. The max_model_len limit is enforced by vLLM.

        Args:
            request: The request to check.

        Returns:
            True if the new request can be scheduled, False otherwise.
        """
        # Scheduler is already running at capacity
        if self.at_capacity:
            return False

        # There is already an active prefill request in the running queue
        if self.has_prefill_in_running:
            return False

        # Check prefill batch capacity (only 1 prefill at a time)
        if len(self.waiting) >= self.max_prefills_per_batch:
            return False

        # Prevent livelock: don't admit a new prefill if running requests
        # already saturate KV cache capacity at worst-case (max_model_len).
        # This is conservative — it doesn't account for prefix cache sharing
        # — but guarantees no decode-decode preemption.
        if self._kv_in_use() >= self._max_kv_concurrent:
            if not self._kv_exhaustion_warned:
                logger.warning(
                    "Prefill deferred: %d requests already holding KV at the "
                    "cache concurrency limit (%d, assuming max_model_len=%d "
                    "per request). Waiting for decode requests to complete. "
                    "To increase concurrency, %s",
                    self._kv_in_use(),
                    self._max_kv_concurrent,
                    self.max_model_len,
                    _kv_capacity_guidance(),
                )
                self._kv_exhaustion_warned = True
            else:
                logger.debug(
                    "Prefill deferred: %d holding KV >= limit %d.",
                    self._kv_in_use(),
                    self._max_kv_concurrent,
                )
            return False

        return True

    @staticmethod
    def _is_structured_output_grammar_waiter(request: Request) -> bool:
        return request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    def _has_ready_structured_output_waiter(self) -> bool:
        """Return whether skipped_waiting has an SO request ready to prefill."""
        for req in self.skipped_waiting:
            if not self._is_structured_output_grammar_waiter(req):
                continue
            structured_output_req = req.structured_output_request
            if structured_output_req and structured_output_req.grammar:
                return True
        return False

    def _has_structured_output_waiter(self) -> bool:
        """Return whether skipped_waiting contains any SO grammar request."""
        return any(
            self._is_structured_output_grammar_waiter(req)
            for req in self.skipped_waiting
        )

    def _should_skip_prefilling_padding_in_di(
        self, scheduler_output: "SchedulerOutput", req_id: str
    ) -> bool:
        """Skip prefill padding on the decode node in disaggregated inference.

        In DI, the decode node receives KV transfers from the prefill node.
        Padding should only be applied on the prefill node, not the decode node.

        Args:
            scheduler_output: The scheduler output to check.
            req_id: The request ID to check.

        Returns:
            True if padding should be skipped (decode node in DI), False otherwise.
        """
        request = self.requests.get(req_id)
        if request is None:
            return False
        kv_transfer_params = request.kv_transfer_params
        return (
            scheduler_output.kv_connector_metadata is not None
            and kv_transfer_params is not None
            and not kv_transfer_params.get("do_remote_decode", False)
        )

    def _apply_padding_and_log_stats(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "SchedulerOutput":
        """Apply padding to scheduler output for the prefill request.

        Since max_prefills_per_batch=1, there's at most one prefill request.
        We determine prefill status from SchedulerOutput data (which has
        pre-increment num_computed_tokens) rather than request state.

        Populates num_scheduled_tokens_padded for the prefill request while
        keeping num_scheduled_tokens as actual (unpadded) token counts.

        Args:
            scheduler_output: Original scheduler output.

        Returns:
            Modified scheduler output with padded token counts.
        """
        # Initialize padded counts dictionary if needed
        # Use getattr/setattr since num_scheduled_tokens_padded is not part of
        # the base vLLM SchedulerOutput dataclass
        if (
            not hasattr(scheduler_output, "num_scheduled_tokens_padded")
            or getattr(scheduler_output, "num_scheduled_tokens_padded") is None
        ):
            scheduler_output.num_scheduled_tokens_padded = {}

        # Build a map of req_id -> num_computed_tokens from SchedulerOutput
        # This has the pre-increment values (before _update_after_schedule)
        req_num_computed: dict[str, int] = {}

        # From new requests
        for new_req in scheduler_output.scheduled_new_reqs:
            req_num_computed[new_req.req_id] = new_req.num_computed_tokens

        # From cached requests
        if scheduler_output.scheduled_cached_reqs is not None:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(cached_reqs.req_ids):
                req_num_computed[req_id] = cached_reqs.num_computed_tokens[i]

        prefill_count = 0
        decode_count = 0
        total_padding_this_batch = 0

        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            # Use num_computed_tokens from SchedulerOutput (pre-increment value)
            num_computed = req_num_computed.get(req_id, request.num_computed_tokens)
            is_prefill = num_computed < request.num_prompt_tokens

            # A last-token re-compute (the prompt is already available; only
            # the final token is re-computed for sampling) is decode-shaped
            # work: classified as prefill it either mismatches the decode NEFF
            # shape on idle DP ranks or trips the separation guard below.
            # Width 1 is reclassified only for the decode-bench connector
            # (segmented prefill / regular DI 1-token re-computes stay
            # prefill). Width 1 + num_spec_tokens is reclassified for any
            # config: vLLM pads the re-compute to spec-decode width only when
            # co-scheduled with running decodes, so the prefill classification
            # always violates the guard.
            if is_prefill and num_computed == request.num_prompt_tokens - 1:
                if (num_tokens == 1 and self._is_decode_bench_connector) or (
                    self.num_spec_tokens > 0 and num_tokens == 1 + self.num_spec_tokens
                ):
                    is_prefill = False
                    logger.debug(
                        "Last-token re-compute for request %s: treating as "
                        "decode (num_tokens=%d, num_computed=%d)",
                        req_id,
                        num_tokens,
                        num_computed,
                    )

            if is_prefill:
                # Skip padding on decode node in DI setting
                if self._should_skip_prefilling_padding_in_di(scheduler_output, req_id):
                    decode_count += 1
                    scheduler_output.num_scheduled_tokens_padded[req_id] = num_tokens
                    logger.debug(
                        "DI decode node: skipping padding for prefill request %s",
                        req_id,
                    )
                    continue

                prefill_count += 1
                padded_tokens = self._calculate_padded_count(num_tokens)
                padding_added = padded_tokens - num_tokens

                # Store padded count separately
                scheduler_output.num_scheduled_tokens_padded[req_id] = padded_tokens

                # Observe padded token count metric
                bucket_name = f"prefill_s{padded_tokens}"
                NUM_BATCHED_TOKENS_PADDING.labels(
                    model_name=self.model_name,
                    bucket_name=bucket_name,
                ).observe(padded_tokens)

                # Track statistics
                self.total_padding_tokens += padding_added
                self.total_scheduled_tokens += num_tokens
                total_padding_this_batch += padding_added

                # Log padding info
                if num_tokens > 0:
                    overhead_pct = (padding_added / num_tokens) * 100
                    logger.debug(
                        "PADDING: Prefill request %s: %d -> %d tokens "
                        "(+%d padding, %.1f%% overhead)",
                        req_id,
                        num_tokens,
                        padded_tokens,
                        padding_added,
                        overhead_pct,
                    )
            else:
                decode_count += 1
                scheduler_output.num_scheduled_tokens_padded[req_id] = num_tokens
                logger.debug(
                    "Decode request %s: %d tokens (no padding)", req_id, num_tokens
                )

        # Log batch summary
        if prefill_count > 0 and total_padding_this_batch > 0:
            original_total = sum(
                num_tokens
                for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
                if self.requests.get(req_id)
                and req_num_computed.get(req_id, 0)
                < self.requests[req_id].num_prompt_tokens
            )
            if original_total > 0:
                batch_overhead_pct = (total_padding_this_batch / original_total) * 100
                logger.debug(
                    "Batch padding summary: %d prefill reqs, "
                    "%d padding tokens (%.1f%% overhead)",
                    prefill_count,
                    total_padding_this_batch,
                    batch_overhead_pct,
                )

        # Observe padded batch size metric for decode
        if decode_count > 0:
            padded_batch_size = get_bucket_for_count(
                decode_count, self.num_seqs_buckets
            )
            bucket_name = f"decode_b{padded_batch_size}"
            NUM_SEQS_PADDING.labels(
                model_name=self.model_name,
                bucket_name=bucket_name,
            ).observe(decode_count)

        # Safety check: Neuron requires prefill and decode to be separate
        if prefill_count > 0 and decode_count > 0:
            raise RuntimeError(
                f"Neuron constraint violated: Mixed prefill ({prefill_count}) "
                f"and decode ({decode_count}) in the same batch. This indicates a bug "
                "in the scheduling logic."
            )

        # Log cumulative statistics periodically
        if self.log_stats and self.total_scheduled_tokens > 0:
            overhead = (self.total_padding_tokens / self.total_scheduled_tokens) * 100
            logger.debug(
                "Cumulative padding overhead: %d / %d = %.2f%%",
                self.total_padding_tokens,
                self.total_scheduled_tokens,
                overhead,
            )

        return scheduler_output

    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Schedule requests with prefill/decode separation and bucket padding.

        Flow:
        1. Move all waiting requests to holdback queue
        2. Selectively restore based on can_schedule() (includes KV capacity check)
        3. Separate prefill and decode based on state
        4. Delegate to parent scheduler
        5. Restore queues
        6. Apply bucket padding to prefill request
        7. Attach grammar bitmask for structured outputs
        8. Update state

        Args:
            throttle_prefills: DP prefill-balancing hint from the engine core;
                forwarded unchanged to the base scheduler.

        Returns:
            SchedulerOutput with padded token counts for prefill requests.
        """
        logger.debug(
            "schedule() called in state %s (waiting=%d, skipped_waiting=%d, "
            "running=%d)",
            self._state,
            len(self.waiting),
            len(self.skipped_waiting),
            len(self.running),
        )

        # Step 0: Long-prompt routing (no-op unless spec_decode_max_prompt_len is set)
        self._route_long_prompts_off_spec()

        # Step 1: Move all waiting to holdback queue (preserving priority)
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Step 2: Selectively restore based on can_schedule()
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Stop to preserve priority order
                break

        # Step 3: Separate prefill/decode. Upstream can promote ready
        # structured-output grammar requests from skipped_waiting in this step.
        has_ready_structured_output_waiting = self._has_ready_structured_output_waiter()
        has_structured_output_waiting = self._has_structured_output_waiter()
        has_prefill_waiting = bool(self.waiting) or has_ready_structured_output_waiting
        can_start_prefill = (
            has_prefill_waiting
            and not self.at_capacity
            and not self.has_prefill_in_running
            and self._kv_in_use() < self._max_kv_concurrent
        )
        running_holdback: list[Request] = []
        max_num_running_reqs_override: int | None = None
        if self.has_prefill_in_running:
            # 3.1 Ongoing prefill segments in running queue - hide decode requests
            running_holdback = [
                req for req in self.running if not self._is_prefill_request(req)
            ]
            self.running = [
                req for req in self.running if self._is_prefill_request(req)
            ]
            if has_structured_output_waiting:
                max_num_running_reqs_override = len(self.running)
            logger.debug(
                "Scheduling prefill step: keeping %d prefill in running, "
                "hiding %d decode, %d in waiting, %d in skipped_waiting, "
                "%d in holdback",
                len(self.running),
                len(running_holdback),
                len(self.waiting),
                len(self.skipped_waiting),
                len(self.holdback_queue),
            )
        elif (
            can_start_prefill
            and self._skip_decode_holdback
            and not has_structured_output_waiting
            and self._admissible_waiters_are_async_kv_pulls()
        ):
            # Take no holdback: every admissible waiter is a pure asynchronous KV pull, which
            # the base scheduler parks with zero tokens, so hiding the decodes would cost them
            # a step for nothing. ``running_holdback`` keeps its empty default, so the capacity
            # decrement is a no-op and the restore concatenates an empty list.
            logger.debug(
                "Scheduling decode step with an async KV pull: %d decode request(s), "
                "%d waiting, no holdback",
                len(self.running),
                len(self.waiting),
            )
        elif can_start_prefill:
            # 3.2 New requests waiting for prefill - hide decode requests
            running_holdback = self.running
            self.running = []
            if has_structured_output_waiting:
                available_slots = max(
                    0, self.max_num_running_reqs - len(running_holdback)
                )
                max_num_running_reqs_override = min(
                    self.max_prefills_per_batch,
                    available_slots,
                )
            logger.debug(
                "Scheduling prefill step: %s request(s), "
                "%s skipped request(s), holding back %s waiting, %s running, "
                "temporary max running=%s",
                len(self.waiting),
                len(self.skipped_waiting),
                len(self.holdback_queue),
                len(running_holdback),
                max_num_running_reqs_override,
            )
        else:
            # 3.3 No prefill requests - no need to hide decodes
            if (
                has_ready_structured_output_waiting
                and len(self.running) < self.max_num_running_reqs
            ):
                # Prevent ready SO waiters from admitting a prefill into a
                # decode batch after the Neuron pre-check.
                max_num_running_reqs_override = len(self.running)
            logger.debug(
                "Scheduling decode step: scheduling %d decode request(s), "
                "waiting=%d, skipped_waiting=%d, prefill_waiting=%s, "
                "structured_output_waiting=%s, temporary max running=%s",
                len(self.running),
                len(self.waiting),
                len(self.skipped_waiting),
                has_prefill_waiting,
                has_structured_output_waiting,
                max_num_running_reqs_override,
            )

        # Step 3.5: Cap the structured-output grammar waiters the base scheduler
        # can see to ensure <= 1 prefill per batch.
        #
        # Gated on has_structured_output_waiting (computed above): when there are
        # no SO grammar waiters this is a no-op, so we skip the drain/re-add
        # entirely and leave the common non-SO path (and skipped_waiting order)
        # untouched.
        skipped_holdback: list[Request] = []
        if has_structured_output_waiting:
            # ``not self.running``: only keep an SO waiter visible when the batch
            # would be all-prefill. If decodes are in flight (the step-3 decode
            # branch leaves them in ``running``), admitting a grammar waiter that
            # becomes ready mid-call would mix prefill+decode in one batch, which
            # Neuron forbids. In that case defer every grammar waiter to a later,
            # dedicated prefill step -- mirroring the ready-at-entry guard in step 3.
            can_admit = (
                not self.running
                and len(self.waiting) < self.max_prefills_per_batch
                and not self.at_capacity
                and not self.has_prefill_in_running
                and len(self.running) < self._max_kv_concurrent
            )
            kept_one = False
            deferred: list[Request] = []
            # pop_request()/add_request() honor the queue's scheduling policy
            # (FCFS: FIFO == popleft/append; PRIORITY: by (priority, arrival_time)).
            # Popping then re-adding the survivors preserves that policy order, so
            # the first grammar waiter we keep visible is the policy-first one.
            while self.skipped_waiting:
                req = self.skipped_waiting.pop_request()
                if not self._is_structured_output_grammar_waiter(req):
                    deferred.append(req)
                    continue
                if can_admit and not kept_one:
                    kept_one = True
                    deferred.append(req)
                    continue
                skipped_holdback.append(req)
            # Preserve queue order for everything left visible.
            for req in deferred:
                self.skipped_waiting.add_request(req)

        # Step 4: Delegate to parent scheduler (sync or async)
        original_max_num_running_reqs = self.max_num_running_reqs
        if max_num_running_reqs_override is not None:
            self.max_num_running_reqs = max_num_running_reqs_override
        else:
            # Reduce capacity by hidden requests so base scheduler doesn't
            # over-admit. Without this, the base scheduler sees self.running
            # == [] and admits up to max_num_running_reqs from waiting; step 5
            # then restores the hidden decodes and the total exceeds the cap,
            # tripping `assert len(self.running) <= self.max_num_running_reqs`.
            # finally: max_num_running_reqs persists on the instance, so a
            # raise from _call_base_schedule must not leave it permanently
            # decremented.
            hidden_count = len(running_holdback)
            if hidden_count > 0:
                self.max_num_running_reqs -= hidden_count
        try:
            scheduler_output = self._call_base_schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = original_max_num_running_reqs

        # Step 5: Restore holdbacks
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())
        # Restore the structured-output grammar waiters held back in Step 3.5.
        for req in skipped_holdback:
            self.skipped_waiting.add_request(req)

        # Check if we can add JF tokens to requests (assuming we're in decode phase).
        is_jf_compatible = (
            not scheduler_output.scheduled_new_reqs and self.max_jump_forward_tokens > 0
        )
        if is_jf_compatible:
            self._inject_jump_forward_tokens(scheduler_output)

        # Step 6: Apply padding to prefill request
        scheduler_output = self._apply_padding_and_log_stats(scheduler_output)

        # Step 7: Attach grammar bitmask for structured outputs
        # Rows = logit rows (SO reqs + spec), NOT prompt tokens / prefill bucket
        # TODO: Consider upstreaming via SamplingMetadata instead of SchedulerOutput
        grammar_output = self.get_grammar_bitmask(scheduler_output)
        if grammar_output is not None and grammar_output.grammar_bitmask is not None:
            scheduler_output._grammar_bitmask = grammar_output.grammar_bitmask
            scheduler_output._structured_output_request_ids = (
                grammar_output.structured_output_request_ids
            )
            logger.debug(
                "[SO] scheduler: attached bitmask ids=%s bitmask_shape=%s",
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask.shape,
            )

        # Step 8: Update states
        self._update_state(scheduler_output)

        # Prefill-async pipeline: a producer whose prefill is now fully dispatched
        # no longer needs the batch-1 compute slot -- park it in the send-lane so
        # the next schedule() can admit the next prefill during this one's device
        # exec and NIXL send.
        if self._pure_producer:
            self._release_dispatched_producers()

        return scheduler_output

    def _is_long_prompt(self, request: Request) -> bool:
        """True when the request's prompt exceeds spec_decode_max_prompt_len."""
        limit = self.spec_decode_max_prompt_len
        return limit is not None and request.num_prompt_tokens > limit

    def _is_spec_routed_off(self, request: Request) -> bool:
        """True when long-prompt routing has turned speculation off for the request."""
        if getattr(request, "_spec_routed_off", False):
            return True
        if self._is_long_prompt(request):
            request._spec_routed_off = True
            return True
        return False

    def _route_long_prompts_off_spec(self) -> None:
        """Run long-prompt requests, and the decodes that share their batch, without drafts.

        A draft model's acceptance usually falls on long contexts, so for a long
        prompt the verification costs more than the drafts save. A request whose
        prompt is longer than ``spec_decode_max_prompt_len`` therefore decodes
        without speculation.

        The compiled decode graphs need the same draft count for every request
        in a batch, and a batch that mixes requests with and without drafts
        gives wrong logits. So while a long-prompt request is running, every
        running request also stops speculating. The flag is sticky: a request
        never goes back to speculation, which avoids a non-spec to spec
        transition in the middle of a request, and a request that joins while
        any routed request runs is routed too, so a decode batch never mixes the
        two. Requests that arrive after the routed ones finish speculate as usual.

        This runs before the base scheduler reads ``spec_token_ids``, so no
        draft tokens are scheduled for a routed request. The model runner then
        takes the existing spec to non-spec transition, the same path it uses
        near ``max_model_len``.
        """
        if self.spec_decode_max_prompt_len is None or self.num_spec_tokens <= 0:
            return
        if not any([self._is_spec_routed_off(req) for req in self.running]):
            return
        for request in self.running:
            request._spec_routed_off = True
            request.spec_token_ids = []

    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Delegate to base vllm Scheduler. Override for async behavior."""
        return Scheduler.schedule(self, throttle_prefills)

    def _inject_jump_forward_tokens(self, scheduler_output: "SchedulerOutput") -> None:
        """Update requests with JF tokens and advance grammar FSM so
        that we sample the next token correctly with the JF NEFF."""
        if not self._pending_jf_tokens:
            return

        # Reset this request's JF tokens to empty
        scheduler_output._jf_decode_tokens = {}

        for req_id in list(scheduler_output.num_scheduled_tokens.keys()):
            if req_id not in self._pending_jf_tokens:
                continue

            # Take the min(len(jf_tokens[req]), self.max_jump_forward_tokens)
            # JF tokens for this req.
            jf_tokens = self._pending_jf_tokens.pop(req_id)[
                : self.max_jump_forward_tokens
            ]
            if not jf_tokens:
                continue

            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            # Check if we can fit JF tokens in remaining slots,
            # after accounting for the to-be-sampled token.
            allowed = self._jf_slots_available(request)
            if allowed <= 0:
                continue
            if allowed < len(jf_tokens):
                logger.info(
                    "JF: req=%s trimming jump span %d -> %d to fit KV/"
                    "max_model_len headroom.",
                    req_id,
                    len(jf_tokens),
                    allowed,
                )
                jf_tokens = jf_tokens[:allowed]

            # Advance the grammar FSM by the JF tokens
            # so we can properly sample the next token
            if self.structured_output_manager.should_advance(request):
                so_req = request.structured_output_request
                grammar = so_req.grammar if so_req is not None else None
                if grammar is not None and not grammar.is_terminated():
                    prev_processed = grammar.num_processed_tokens
                    if not grammar.accept_tokens(req_id, jf_tokens):
                        # rollback on prev accepted so we don't mess up future FSM bitmasks
                        accepted = grammar.num_processed_tokens - prev_processed
                        if accepted > 0:
                            grammar.rollback(accepted)
                        logger.error(
                            "JF: grammar rejected its own jump tokens %s for "
                            "req=%s — skipping JF injection this step.",
                            jf_tokens,
                            req_id,
                        )
                        continue

            # Set the JF tokens we must processed for this request.
            # Advance request params such as (total) scheduled tokens
            # and compute tokens.
            scheduler_output._jf_decode_tokens[req_id] = jf_tokens
            scheduler_output.num_scheduled_tokens[req_id] += len(jf_tokens)
            scheduler_output.total_num_scheduled_tokens += len(jf_tokens)
            request.num_computed_tokens += len(jf_tokens)

            # JF statistics logging
            self._jf_total_injections += 1
            self._jf_total_tokens_injected += len(jf_tokens)
            avg = self._jf_total_tokens_injected / self._jf_total_injections
            logger.info(
                "Jump-forward: req=%s injecting %d tokens (avg=%.1f, total_inj=%d)",
                req_id,
                len(jf_tokens),
                avg,
                self._jf_total_injections,
            )

    def _compute_jump_forward_tokens(
        self, grammar, tokenizer, output_token_ids: list[int]
    ) -> list[int]:
        """Tokenize the jump forward string returned by xgrammar."""
        matcher = grammar.matcher
        jf_string = matcher.find_jump_forward_string()
        if not jf_string:
            return []

        old_decoded = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        new_decoded = old_decoded + jf_string
        new_output_ids = tokenizer.encode(new_decoded, add_special_tokens=False)

        if len(new_output_ids) <= len(output_token_ids):
            return []

        # If the old and new prefix (after retokenization) have the same token
        # sequence, then we can just accept the newly generated tokens.
        # If not, naively tokenize the JF string.
        if new_output_ids[: len(output_token_ids)] != output_token_ids:
            token_ids = tokenizer.encode(jf_string, add_special_tokens=False)
        else:
            token_ids = new_output_ids[len(output_token_ids) :]

        return token_ids

    def _jf_slots_available(self, request) -> int:
        """Compute the number of slots we can use for JF tokens."""
        num_computed = request.num_computed_tokens
        managers = self.kv_cache_manager.coordinator.single_type_managers

        group_headroom = None
        for manager in managers:
            blocks = manager.req_to_blocks.get(request.request_id)
            if not blocks:
                # No blocks tracked for this group → no tail to ride in.
                return 0
            # -1 for the actual sampled token
            headroom = len(blocks) * manager.block_size - num_computed - 1
            if group_headroom is None or headroom < group_headroom:
                group_headroom = headroom

        if group_headroom is None:
            group_headroom = 0

        # Reserve one token of max_model_len headroom for
        # the truly-sampled token.
        model_len_headroom = self.max_model_len - num_computed - 1
        return max(0, min(group_headroom, model_len_headroom))


class NeuronAsyncScheduler(NeuronScheduler, AsyncScheduler):
    """
    Async Neuron scheduler combining:
    - NeuronScheduler: Neuron-specific scheduling (holdback, prefill/decode separation, padding)
    - AsyncScheduler: Output placeholders, async token handling
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: "MultiModalRegistry | None" = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

    def _call_base_schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Override to use AsyncScheduler.schedule() instead of Scheduler.schedule().

        NeuronScheduler.schedule() calls this method to delegate to the base scheduler.
        For async scheduling, we need AsyncScheduler's behavior.
        """
        return AsyncScheduler.schedule(self, throttle_prefills)

    def _update_after_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        """Override to suppress spec-token placeholders near max_model_len.

        ``AsyncScheduler._update_after_schedule`` unconditionally sets
        ``request.spec_token_ids = [-1] * num_spec_tokens`` so the next
        scheduling step reserves ``1 + num_spec`` slots per request. Near
        ``max_model_len``, the scheduler's own trim logic
        (``num_new = min(1 + num_spec, max_model_len - 1 - num_computed)``)
        then produces partial shapes (e.g. 3 or 2 tokens instead of 4).
        Those partial shapes trigger a recompile against the target NEFF
        that was only warmed for the full ``1 + num_spec`` bucket.

        On Neuron the draft NEFF is a single compiled graph with fixed
        input shape, so we can't vary draft count per request. The clean
        transition is: near max_model_len, stop proposing drafts entirely
        and fall through to the non-spec target NEFF (which is warmed at
        ``decode_b{bs}_s1``). We signal "no drafts next step" by clearing
        ``spec_token_ids`` instead of setting placeholders, whenever the
        next step's full spec bucket wouldn't fit within max_model_len.

        Engine-core-side note: ``post_step`` only calls
        ``update_draft_token_ids`` when async scheduling is disabled
        (see ``vllm/v1/engine/core.py``). So the worker's
        ``take_draft_token_ids`` value is never propagated to the scheduler
        in async mode; the placeholder must be gated here instead.
        """
        # Advance num_computed_tokens via the plain Scheduler path, skipping
        # AsyncScheduler's unconditional placeholder injection.
        Scheduler._update_after_schedule(self, scheduler_output)

        # Re-apply AsyncScheduler's bookkeeping. The placeholder-count
        # increment (``+= 1 + cur_num_spec_tokens``) is required for both
        # spec and non-spec async paths — ``_update_request_with_output``
        # decrements it once per emitted token, so skipping the increment
        # would assert-fail. The spec-only logic (sticky disable near
        # max_model_len, ``spec_token_ids`` placeholder injection) is
        # gated on ``num_spec_tokens > 0``.
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Scheduler trims when `num_new = min(1 + num_spec, max_model_len - 1 - num_computed)`
        # falls below `1 + num_spec`. To avoid trim: `num_computed <= max_model_len - 2 - num_spec`.
        # Note: `num_computed` can TEMPORARILY go back down when
        # `update_from_output` applies rejection correction from an
        # in-flight step. That can make a previously-disabled spec
        # placeholder re-enable, producing positions that regress into
        # slots already written by later steps. We keep a sticky
        # ``_async_spec_disabled`` flag per request to prevent that
        # oscillation: once disabled, stays disabled for the rest of the
        # request's lifetime.
        # Extra margin (another 1+num_spec) so we transition to non-spec
        # before the last "safe" spec step. In async mode, the spec→non-spec
        # transition step fires the fallback path in the worker because the
        # previous step's sampled-token future has shape [bs, num_spec+1]
        # while the current non-spec input_ids is [bs]. That fallback
        # materializes async output and rebuilds inputs — but reads
        # input_ids from positions still reflecting the scheduler's
        # OPTIMISTIC num_computed_tokens (off by the previous step's
        # rejections). Transitioning earlier avoids the last spec step,
        # which tends to have heavy rejection near max_model_len.
        is_spec = self.num_spec_tokens > 0
        safe_num_computed = (
            self.max_model_len - 3 - 2 * self.num_spec_tokens if is_spec else 0
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            if not is_spec:
                continue
            # Sticky: once disabled for this request, stay disabled.
            disabled = getattr(request, "_async_spec_disabled", False)
            if not disabled and request.num_computed_tokens > safe_num_computed:
                disabled = True
                request._async_spec_disabled = True
            if disabled or (
                self.spec_decode_max_prompt_len is not None
                and self._is_spec_routed_off(request)
            ):
                request.spec_token_ids = []
            else:
                request.spec_token_ids = self._spec_token_placeholders
