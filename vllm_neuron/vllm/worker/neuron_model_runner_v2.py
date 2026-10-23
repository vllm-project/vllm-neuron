# SPDX-License-Identifier: Apache-2.0
"""Provide the standalone V2 Neuron model runner.

Own one :class:`NeuronRequestState` and implement model loading, KV-cache
setup, warmup, execution, asynchronous output, and scheduler-driven request
lifecycle without importing or subclassing the V1 runner.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import time
from dataclasses import fields as dc_fields

from typing import NamedTuple

import numpy as np
import torch

from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

from vllm_neuron import envs
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.snapshot.runner_hook import (
    SnapshotRunnerHook,
    check_v2_capture_config,
)
from vllm_neuron.utils.lnc import get_explicit_compiler_lnc_arg
from vllm_neuron.utils.neuron_utils import model_forward_context

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists, ModelRunnerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)

from .neuron_forward_adapter import (
    NeuronForwardAdapter,
    build_attention_metadata,
)
from .neuron_input_build import (
    NULL_BLOCK_ID,
    NeuronModelInput,
    build_flat_positions,
    build_kernel_block_rows,
    build_prefill_linear_idx,
    build_slot_mapping,
    build_swa_kernel_block_rows,
    compute_swa_num_blocks,
    gather_block_rows_raw,
)
from .neuron_request_state import NO_LOGPROBS, NeuronRequestState

logger = logging.getLogger(__name__)


class _DebugLogitRow(NamedTuple):
    logit_row: int
    batch_idx: int
    position_id: int


class _PendingDebugLogits(NamedTuple):
    # Counts debug captures, not model steps; steps without captured rows do
    # not advance it.
    capture_index: int
    rows: tuple[_DebugLogitRow, ...]
    # The requests that write their last allowed logit in this capture. The
    # writer adds one done file for each. Build the list here, not in the
    # writer: an async read can run later, after the request leaves the state.
    final_req_ids: tuple[str, ...]


class _PendingStep(NamedTuple):
    """Carry output-formatting state from execution to sampling.

    Apply the committed state bundle during execution rather than carrying it
    here. Keep host validity for sampled rows and a mutable one-shot completion
    fence for the quarantine ledger.

    Attributes:
        req_ids: Submitted request IDs in output order.
        sampled_future: Device future containing sampled tokens.
        valid_rows: Host flags for real sampled rows. Mark padding and
            nonfinal prefill chunks false.
        completed: Mutable completion fence set after output materialization.
        debug_logits: Optional deferred debug-logit output.
        kv_connector_output: Optional KV-connector output for this step, captured
            at ``post_forward`` time. Carried per step (not on the runner) so an
            overlapped next step cannot overwrite it before this step's response
            is formatted.
    """

    req_ids: list[str]
    sampled_future: torch.Tensor
    valid_rows: tuple[bool, ...]
    completed: dict
    debug_logits: _PendingDebugLogits | None
    gathered_logits_future: torch.Tensor | None
    num_logprobs_by_row: tuple[int, ...]
    kv_connector_output: object | None


def _trim_sampled(sampled_tokens, valid_rows, num_submitted):
    """Trim padded sampled rows into per-request token lists.

    Emit ``[token]`` for each valid submitted row and ``[]`` for a submitted
    nonfinal prefill row. Drop rows beyond ``num_submitted``.

    Args:
        sampled_tokens (Sequence[int]): Materialized sampled rows.
        valid_rows (Sequence[bool]): Host validity flag for each graph row.
        num_submitted (int): Number of real submitted request rows.

    Returns:
        list[list[int]]: One sampled-token list per submitted request.
    """
    return [
        [int(sampled_tokens[i])] if valid_rows[i] else [] for i in range(num_submitted)
    ]


def _build_sampled_token_logprobs(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    output_row_indices: list[int],
    max_num_logprobs: int,
    num_output_rows: int,
    logprobs_mode: str,
) -> LogprobsLists:
    """Build sampled-token log-probability results in output order.

    Args:
        logits (torch.Tensor): Logits for rows that requested log probabilities.
        sampled_token_ids (torch.Tensor): Sampled token ID for each row in
            ``logits``.
        output_row_indices (list[int]): Output positions corresponding to rows
            in ``logits``.
        max_num_logprobs (int): Maximum number of candidates requested per row.
        num_output_rows (int): Number of request rows in the output.
        logprobs_mode (str): Requested log-probability format.

    Returns:
        LogprobsLists: Log-probability lists aligned with the output rows.

    Raises:
        RuntimeError: If ``logprobs_mode`` is unsupported.
    """
    from vllm.v1.sample.sampler import Sampler

    token_scores = logits.to(torch.float32)
    if logprobs_mode == "raw_logprobs":
        token_scores = Sampler.compute_logprobs(token_scores)
    elif logprobs_mode != "raw_logits":
        raise RuntimeError(
            f"MRv2 sampled-token scores do not support {logprobs_mode!r}"
        )

    gathered_token_scores = Sampler.gather_logprobs(
        token_scores,
        max_num_logprobs,
        token_ids=sampled_token_ids,
    ).tolists()
    output_width = max_num_logprobs + 1
    output_token_ids = np.zeros(
        (num_output_rows, output_width),
        dtype=gathered_token_scores.logprob_token_ids.dtype,
    )
    output_token_scores = np.zeros(
        (num_output_rows, output_width),
        dtype=gathered_token_scores.logprobs.dtype,
    )
    sampled_token_ranks = np.zeros(
        num_output_rows,
        dtype=gathered_token_scores.sampled_token_ranks.dtype,
    )
    for gathered_row, output_row in enumerate(output_row_indices):
        output_token_ids[output_row] = gathered_token_scores.logprob_token_ids[
            gathered_row
        ]
        output_token_scores[output_row] = gathered_token_scores.logprobs[gathered_row]
        sampled_token_ranks[output_row] = gathered_token_scores.sampled_token_ranks[
            gathered_row
        ]
    return LogprobsLists(
        output_token_ids,
        output_token_scores,
        sampled_token_ranks,
        None,
    )


class _AsyncNeuronModelRunnerOutput(AsyncModelRunnerOutput):
    """Defer materialization of an asynchronous sampled-token future.

    Let ``get_output`` perform the only device-to-host synchronization, trim
    rows with host validity, set the completion fence, and return the response
    output. Cache the result so each instance reads its future once.
    """

    def __init__(self, runner, pending: _PendingStep) -> None:
        self._runner = runner
        self._pending = pending
        self._lock = threading.Lock()
        self._output: ModelRunnerOutput | None = None

    def get_output(self) -> ModelRunnerOutput:
        # The output-rank async-copy thread may already be materializing this future when the worker
        # main thread reaches a new-request join and drains it on every TP rank. Serialize those callers
        # and cache the host result so the device future is read exactly once.
        with self._lock:
            if self._output is None:
                pending = self._pending
                sampled_tokens = pending.sampled_future.reshape(-1).cpu().tolist()
                token_lists = _trim_sampled(
                    sampled_tokens,
                    pending.valid_rows,
                    len(pending.req_ids),
                )
                logprobs = self._runner._materialize_gathered_logits(
                    pending, token_lists
                )
                pending.completed["done"] = (
                    True  # completion fence (future materialized)
                )
                self._output = self._runner.get_output(
                    pending.req_ids,
                    token_lists,
                    logprobs=logprobs,
                    kv_connector_output=pending.kv_connector_output,
                )
            assert self._output is not None, (
                "async output materialization completed without producing "
                "a ModelRunnerOutput"
            )
            return self._output


class NeuronModelRunnerV2:
    """Run a slot-stable scheduler lifecycle with one request-state owner."""

    def __init__(self, vllm_config, device) -> None:
        """Initialize standalone runner state from the vLLM configuration.

        Read dimensions and feature settings directly without sharing the V1
        runner's construction path.

        Args:
            vllm_config (VllmConfig): Engine and model configuration.
            device (torch.device): Device used for model execution and state.
        """
        self.vllm_config = vllm_config
        self.device = device

        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config

        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = model_config.max_model_len
        self.use_async_scheduling = getattr(scheduler_config, "async_scheduling", True)

        self.max_num_batched_tokens = min(
            scheduler_config.max_num_batched_tokens, self.max_model_len
        )
        self.neuron_config = self._build_neuron_config()
        self.num_batched_tokens_buckets = list(
            self.neuron_config.num_batched_tokens_buckets
        )
        self.num_seqs_buckets = list(self.neuron_config.num_seqs_buckets)
        self.prefill_token_capacity = max(self.num_batched_tokens_buckets)
        self.decode_row_capacity = max(self.num_seqs_buckets)
        segment_buckets = self.neuron_config.kv_segment_size_buckets
        self.kv_segment_size = segment_buckets[0] if segment_buckets else 0

        # Placeholder geometry only: vLLM applies the backend block-size default after constructing
        # the worker's model runner. initialize_kv_cache replaces this from the validated per-group
        # KV specs before any request is admitted.
        block_sizes = getattr(cache_config, "block_sizes", None)
        if not block_sizes:
            block_sizes = [cache_config.block_size]
        self.block_sizes = list(block_sizes)
        # Per-group trimmed block-row width for sliding-window groups; replaced
        # from the validated group specs in initialize_kv_cache.
        self._swa_num_blocks: list[int | None] = [None] * len(self.block_sizes)

        # Worker-facing attribute surface.
        # The model is published only after load_model() fully succeeds.
        self.model = None
        # First cell is dense/generate/non-spec; these features route to V1, so the runner exposes the
        # attributes the worker reads unconditionally as their "off" values.
        self.speculative_config = None
        self.drafter = None
        self.encoder_neuron_config = None
        # Async-scheduling stats.
        self._async_steps = 0
        self._sync_fallback_steps = 0
        self._debug_logits_dir = self.neuron_config.debug_logits_dir
        self._emit_debug_logits = self._debug_logits_dir is not None
        self._debug_logits_capture_index = 0
        self._logprobs_mode = getattr(model_config, "logprobs_mode", "raw_logprobs")
        # Match MRv1 migration semantics: _build_neuron_config inherits the
        # vLLM model value when Neuron max_logprobs is not explicitly set.
        self._logprobs_enabled = self.neuron_config.max_logprobs != 0
        # Input-snapshot capture: the shared runner hook (one capturer, built
        # lazily) installed only around serve-time forwards when enabled.
        # Fails fast on configs the V2 capture wiring cannot honor (e.g. a
        # speculative draft window) instead of silently skewing selection.
        check_v2_capture_config(vllm_config)
        self._snapshot_hook = SnapshotRunnerHook()
        # KV-cache placeholders published by initialize_kv_cache.
        self._kv_cache_config = None
        self.kv_caches: dict = {}
        # Block-table snapshot for KV-cache analysis (accuracy debugger). The
        # first get_block_tables RPC enables snapshotting; execute_model then
        # keeps a slot-indexed host copy of the block rows each step so the
        # tables survive request freeing. Zero overhead until enabled.
        self._kv_snapshot_enabled: bool = False
        self._block_table_snapshot: list | None = None
        # Composed KV connector (DI). Defaults to the shared no-op; initialize_kv_cache
        # replaces it with an ActiveKVConnector when a KV-transfer group is present (DI on). Reused
        # verbatim from upstream MRv2. The no-op makes the non-DI path inert.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        self.request_state = NeuronRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            device=self.device,
            block_sizes=self.block_sizes,
        )
        # A bounded list of in-flight submitted graphs, each an
        # ``(is_complete, touched_slots)`` pair, plus the set of slots freed while an in-flight graph
        # may still touch them. A quarantined slot is withheld from ``free_indices`` until every
        # touching graph completes. No per-slot generation state is needed.
        self._inflight_graphs: list[tuple[object, set[int]]] = []
        self._quarantined_slots: set[int] = set()
        # The compiled forward adapter plus the one in-flight
        # step's output-formatting state (execute_model -> sample_tokens).
        self._forward_adapter = None
        self._pending_step: _PendingStep | None = None
        # Retained on EVERY TP rank. The executor consumes the wrapper only on output_rank, but a later
        # new-request prefill may not depend on the previous decode future. At that join boundary every
        # rank materializes this same previous output before admission, completing its lazy TP sampling
        # collective without adding D2H to steady decode.
        self._last_async_output: _AsyncNeuronModelRunnerOutput | None = None
        # Keepalive for the most recent forward output (sampled + aliased bundle) on EVERY TP rank. The
        # adapter returns an NrtaFuture whose in-flight NRT input slices must remain alive. This protects
        # the future's lifetime only; the join-boundary drain above provides cross-rank execution ordering.
        self._submitted_forward_keepalive = None
        # Threading sampled tokens through a non-aliased output forces every TP
        # rank to finish the current sampling collective before the next decode
        # begins, preventing collective-order hangs. When new requests are
        # admitted, refresh it from request state so requests with cached context
        # have the token needed to decode immediately.
        self._next_token_by_slot = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int32, device=self.device
        )
        # Public attributes the worker reads unconditionally (warmup path / EC
        # path). The legacy device backend publishes the adapter capture
        # sidecar here; CPU compile and native compilation leave it unset.
        self.capture_backend_model = None
        self.encoder_capture_backend = None
        self.encoder_cache = None
        # Set True by load_model for the synthetic (off-device test) model, which is never compiled.
        self._is_synthetic_model = False
        # Holds the latest idle-rank DP dummy output future so GC cannot free
        # its in-flight NRT input slices (see execute_dummy_batch).
        self._dp_dummy_output_keepalive = None
        # Private state tensors for DP dummies (see _dp_dummy_state_tensors).
        self._dp_dummy_state = None
        # torch.compile options shared by graph capture, parallel compilation,
        # and the serving adapter. alias_meta_to_neuron threads the aliased
        # state-bundle outputs back onto their inputs (the store-version chain).
        self.compile_options = {"alias_meta_to_neuron": True}
        logger.info("Using the standalone V2 Neuron model runner.")

    def _is_cpu_mode(self) -> bool:
        """Return whether the runner executes on CPU."""
        return envs.VLLM_NEURON_CPU_MODE or (
            self.device.type == "cpu" and not envs.VLLM_NEURON_CPU_COMPILE
        )

    # ------------------------------------------------------------------
    # Worker-facing attribute surface.
    # ------------------------------------------------------------------
    @property
    def is_pooling_model(self) -> bool:
        # A property (not a method) so attribute-style consumers see False, matching V1.
        return False

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig info as a serializable dict for KV reconstruction.

        Consumed by the accuracy-analysis pipeline
        (``vllm_neuron.accuracy.kv_cache_analysis``) over collective_rpc.

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
        if self._kv_cache_config is None:
            return {}

        groups = []
        for g in self._kv_cache_config.kv_cache_groups:
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
            "num_blocks": self._kv_cache_config.num_blocks,
            "groups": groups,
            "tp_size": parallel_config.tensor_parallel_size,
            "total_num_kv_heads": model_config.get_total_num_kv_heads(),
            "dcp_size": parallel_config.decode_context_parallel_size,
            "pcp_size": parallel_config.prefill_context_parallel_size,
        }

    def get_kv_caches(self) -> dict[str, bytes]:
        """Return per-layer KV cache tensors, serialized as bytes.

        Each layer serializes to ``torch.save({"k": ..., "v": ...})`` bytes
        with CPU tensors, keyed by the model layer name
        (``layers.{i}.self_attn``), the format the accuracy-analysis
        pipeline's TP-shard reconstruction and HF golden comparison load.

        Example:
            >>> kv = model_runner.get_kv_caches()
        """
        result = {}
        for layer_name, entry in self.kv_caches.items():
            if len(entry) != 2:
                # Key-only caches (MLA) are unsupported here; skip rather
                # than emit a half pair the analysis pipeline cannot load.
                continue
            k_cache, v_cache = entry
            buf = io.BytesIO()
            torch.save({"k": k_cache.detach().cpu(), "v": v_cache.detach().cpu()}, buf)
            result[layer_name] = buf.getvalue()
        return result

    # Tensor capture is a debug/accuracy tool unused by the first cell. Keep the
    # worker-facing hooks as no-ops.
    def enable_capture(self) -> None:
        pass

    def disable_capture(self) -> None:
        pass

    def init_tensor_capture(self) -> None:
        pass

    def init_tensor_replacement(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Worker-facing hooks for denylisted features.
    # ------------------------------------------------------------------
    # The worker calls these unconditionally (shutdown / post-warmup) or as collective_rpc targets. The
    # underlying features (KV transfer/DI, EC connector, speculative decoding) are denylisted by the V2
    # selector, so on the dense serve path there is nothing to do -- these are no-ops. They exist so the
    # worker never AttributeErrors on a symbol the V1 runner had.
    def ensure_kv_transfer_shutdown(self) -> None:
        pass  # no KV-transfer group in the first cell (DI denylisted)

    def maybe_register_ec_cache(self) -> None:
        pass  # no EC connector in the first cell (EPD/EC denylisted)

    def take_draft_token_ids(self):
        return None  # no speculative decoding in the first cell (spec denylisted)

    # DI/EPD snapshot + EC accuracy RPCs: never on the dense serve path. RAISE (not empty data that could
    # look valid) so an out-of-cell collective_rpc fails loud with a clear message instead of corrupting.
    def _out_of_cell(self, name: str):
        raise NotImplementedError(
            f"{name} is not supported by the V2 standalone runner (route the config to V1)"
        )

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

        Return the live tables while requests hold blocks, otherwise the last
        snapshot taken during a forward pass (see ``enable_kv_snapshot``).

        Rows are slot-indexed ``[max_num_reqs, width]``: each request holds a
        stable slot (lowest free first), so the i-th admitted request of a
        fresh batch is row i -- the row order the analysis pipeline indexes
        with ``seq_idx``.

        Example:
            >>> tables = model_runner.get_block_tables()
        """
        rs = self.request_state
        if rs.has_allocated_blocks():
            tables = [rs.block_rows_snapshot(g) for g in range(len(self.block_sizes))]
        elif self._block_table_snapshot is not None:
            # Blocks freed — use the snapshot from the last forward pass.
            tables = self._block_table_snapshot
        else:
            return []

        result = []
        for table in tables:
            buf = io.BytesIO()
            torch.save(torch.from_numpy(np.ascontiguousarray(table)), buf)
            result.append(buf.getvalue())
        return result

    def clear_kv_snapshot(self) -> None:
        """Release block-table snapshot memory and disable snapshotting.

        Call after KV cache analysis is complete to free memory and restore
        zero-overhead production behavior.

        Example:
            >>> model_runner.clear_kv_snapshot()
        """
        self._kv_snapshot_enabled = False
        self._block_table_snapshot = None

    def get_block_table_info(self):
        self._out_of_cell("get_block_table_info")

    def get_encoder_cache(self):
        self._out_of_cell("get_encoder_cache")

    def enable_encoder_cache_snapshot(self):
        self._out_of_cell("enable_encoder_cache_snapshot")

    def clear_encoder_cache_snapshot(self):
        self._out_of_cell("clear_encoder_cache_snapshot")

    # ------------------------------------------------------------------
    # Model load and KV-cache specification.
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the dense generate Neuron model. Publishes ``self.model`` ONLY on full success.

        Reimplements V1's load path minimally: resolve the model class, build it on ``meta`` from the
        HF + Neuron configs, load weights (or compile-time constants during CPU compilation), and move
        to device. The first cell is dense/generate/non-spec, so the LoRA / multimodal /
        pooling-convert / EPD-footprint / tensor-capture branches are dropped (those configs route to
        V1). A synthetic model short-circuits weight loading for off-device
        execution.
        """
        from vllm import ModelRegistry

        model_config = self.vllm_config.model_config
        if model_config.runner_type != "generate":
            raise ValueError(
                "V2 standalone runner is generate-only (pooling routes to V1); got "
                f"runner_type={model_config.runner_type!r}"
            )
        # Clear any previously published model and adapter. A failed reload
        # must not leave stale state.
        self.model = None
        self._forward_adapter = None
        self._pending_step = None
        self._submitted_forward_keepalive = None
        self.capture_backend_model = None

        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture, model_config=model_config
        )

        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        if issubclass(model_cls, SyntheticNeuronModel):
            # Synthetic model: no weights (off-device test path). The flag gates _compile_forward_adapter
            # to the uncompiled (eager) branch.
            self._is_synthetic_model = True
            self.model = model_cls.from_configs(model_config.hf_config)
        else:
            import torch as _torch

            with _torch.device("meta"):
                model = model_cls.from_configs(
                    model_config.hf_config, self.neuron_config
                )
            if envs.VLLM_NEURON_CPU_COMPILE:
                if hasattr(model, "load_weights_lite"):
                    model.load_weights_lite(
                        model_config.model,
                        _torch.device("cpu"),
                        self.vllm_config.load_config.download_dir,
                    )
                model = model.to("meta")
            else:
                logger.info("Loading weights from %s", model_config.model)
                weight_load_start_time = time.perf_counter()
                model.load_weights(
                    model_config.model,
                    self.device,
                    self.vllm_config.load_config.download_dir,
                )
                logger.info(
                    "Weight loading completed in %.2f seconds",
                    time.perf_counter() - weight_load_start_time,
                )
                logger.info("Moving model to device: %s", self.device)
                model = model.to(self.device)

                model_name = model_config.model
                MODEL_LOAD_TIME.labels(model_name=model_name).set(
                    time.perf_counter() - weight_load_start_time,
                )
                MODEL_LOAD_SIZE.labels(model_name=model_name).set(
                    sum(p.nbytes for p in model.parameters()),
                )
            # Publish only after the selected load path succeeds (no half-loaded self.model).
            self.model = model

            # Stamp each parallel group's rank_in_group as an on-device tensor
            # (SPMD): collective helpers (MoE all-to-all, routed-token permute)
            # read rank_in_group_tensor during tracing so the graph is
            # rank-invariant. Held to keep the tensors alive. Mirrors V1.
            from vllm_neuron.parallel.neuron_parallel_state import (
                stamp_spmd_rank_tensors,
            )

            self._spmd_rank_tensors = stamp_spmd_rank_tensors(self.device)

            hlo2tensorizer_opts = "--modular-flow-mac-threshold=10"
            from libtorch_neuronx_lite.compile.platform import get_platform_target

            has_fp8 = self.vllm_config.cache_config.cache_dtype in (
                "fp8",
                "fp8_e4m3",
            )
            if not has_fp8 and self.vllm_config.model_config.quantization in (
                "modelopt",
            ):
                has_fp8 = True
            if not has_fp8:
                has_fp8 = getattr(self.neuron_config, "quantization", None) == "fp8"
            if has_fp8 and get_platform_target() not in ("trn3", "trn3pre"):
                hlo2tensorizer_opts += " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
            self.compile_options["compiler_args"] = [
                "--auto-cast=none",
                "--verbose=35",
                f"-O{self.vllm_config.optimization_level.value}",
                f"--internal-hlo2tensorizer-options={hlo2tensorizer_opts}",
                "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
            ]
            logger.info(
                "neuronx-cc optlevel -O%s (from vLLM optimization_level)",
                self.vllm_config.optimization_level.value,
            )
            # Compile for the configured LNC degree, and only when it differs from
            # the compiler default (mirrors V1); see NeuronModelRunner.load_model.
            lnc_arg = get_explicit_compiler_lnc_arg()
            if lnc_arg is not None:
                self.compile_options["compiler_args"].append(lnc_arg)
                logger.info("Compiling for the configured LNC degree: %s", lnc_arg)

            # Per-graph compiler args (mirrors NeuronModelRunner.load_model): a
            # model's compiler_args_for(graph_type, example_inputs, base) can
            # raise -O per graph (e.g. decode/vision -O3). An explicit caller
            # optimization level wins, so enable the policy only when the level
            # was not set explicitly. resolve_graph_compiler_args() reads these.
            base_model_for_opt = getattr(self.model, "_orig_mod", self.model)
            opt_level_explicit = getattr(
                self.vllm_config, "_neuron_optimization_level_explicit", False
            )
            policy = getattr(base_model_for_opt, "compiler_args_for", None)
            per_graph = not opt_level_explicit and policy is not None
            self._per_graph_compiler_args = per_graph
            self._compiler_args_policy = policy if per_graph else None
            if per_graph:
                base_model_for_opt._neuron_base_compiler_args = list(
                    self.compile_options["compiler_args"]
                )
            logger.info(
                "per-graph compiler-args policy=%s", "on" if per_graph else "off"
            )

        # Establish the compiled-adapter postcondition here (mirrors MRv1, which compiles at the end of
        # load_model). The native-fork worker re-decorates the adapter once after the package flush;
        # otherwise warmup/execute rely on this. Eager (CPU-sim / synthetic) publishes the uncompiled
        # adapter; device torch.compiles it.
        # If compilation fails, roll BOTH model and adapter aliases back to None so load_model is
        # all-or-nothing (no published model with a missing/stale adapter).
        try:
            self._compile_forward_adapter()
        except Exception:
            self.model = None
            self._forward_adapter = None
            self._pending_step = None
            self._submitted_forward_keepalive = None
            self.capture_backend_model = None
            raise

    def state_pool_bytes(self) -> int:
        """HBM of dedicated recurrent-state pools: none, V2 serves no recurrent-state models."""
        return 0

    def warm_up_hybrid_graphs(self) -> None:
        """No recurrent-state helper graphs on V2 (see the V1 runner)."""

    def get_kv_cache_spec(self) -> dict:
        """Build per-layer KV-cache specifications.

        Support homogeneous full attention and sliding-window layers with a
        context-parallel degree of one. Windowed layers produce
        ``SlidingWindowSpec`` so the hybrid KV cache manager groups them
        separately from full-attention layers.

        Returns:
            A mapping from model layer name to its KV cache spec.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype

        if self.model is None:
            raise RuntimeError("load_model() must run before get_kv_cache_spec()")
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config
        )
        specs: dict = {}
        for layer in self.model.get_kv_spec().layers:
            if getattr(layer, "sliding_window_size", None) is not None:
                # SlidingWindowSpec (not FullAttentionSpec) so the hybrid KV
                # cache manager creates a separate group whose blocks outside
                # the window are freed; the input builder trims that group's
                # kernel block rows to the window. Mirrors the V1 runner.
                specs[layer.name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                )
                continue
            specs[layer.name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=layer.num_kv_heads,
                head_size=layer.head_size,
                dtype=kv_cache_dtype,
                sliding_window=None,
                attention_chunk_size=getattr(layer, "chunk_size", None),
            )
        return specs

    def _build_neuron_config(self):
        """Build the Neuron model configuration and resolve graph buckets.

        Returns:
            NeuronConfig: The resolved Neuron model configuration.
        """
        from vllm_neuron.model.neuron_config import NeuronConfig

        raw = self._raw_neuron_config()
        neuron_cfg_dict = dict(raw)
        neuron_cfg_dict.setdefault(
            "max_logprobs",
            getattr(self.vllm_config.model_config, "max_logprobs", 0),
        )
        from vllm_neuron.utils.bucket_utils import (
            SUPPORTED_KV_SEGMENT_SIZES,
            get_default_kv_segment_size_buckets,
            get_default_num_seqs_buckets,
            resolve_num_batched_tokens_buckets,
            validate_kv_segment_size_buckets,
            validate_num_seqs_buckets,
        )

        user_set_prefill_buckets = "num_batched_tokens_buckets" in neuron_cfg_dict
        user_set_segment_buckets = "kv_segment_size_buckets" in neuron_cfg_dict

        # Resolve the Q axis independently from the KV segment axis. Explicit
        # buckets keep their configured values; otherwise use the standard
        # power-of-two prefill buckets.
        prefill_buckets = resolve_num_batched_tokens_buckets(
            self.max_num_batched_tokens,
            configured_buckets=neuron_cfg_dict.get("num_batched_tokens_buckets"),
            use_configured_buckets=user_set_prefill_buckets,
        )

        if "num_seqs_buckets" in neuron_cfg_dict:
            decode_buckets = validate_num_seqs_buckets(
                neuron_cfg_dict["num_seqs_buckets"], self.max_num_reqs
            )
        else:
            decode_buckets = get_default_num_seqs_buckets(self.max_num_reqs)

        # Non-packed FP8 KV splits the two paths on purpose, matching V1:
        # an explicit request raises (unsatisfiable, and the fix is one line),
        # while a defaulted one falls back to full prefill -- auto-enabling
        # segmented prefill into a hard warmup failure is our bug, not the
        # user's. That auto-routing was the #1904 regression.
        cache_dtype = getattr(self.vllm_config.cache_config, "cache_dtype", None)
        kv_is_fp8 = isinstance(cache_dtype, str) and cache_dtype.startswith("fp8")
        non_packed_fp8_kv = kv_is_fp8 and not neuron_cfg_dict.get(
            "fp8_packed_kv", False
        )

        if user_set_segment_buckets:
            segment_buckets = validate_kv_segment_size_buckets(
                neuron_cfg_dict["kv_segment_size_buckets"]
            )
            if non_packed_fp8_kv:
                raise ValueError(
                    f"kv_segment_size_buckets={segment_buckets} requests "
                    "segmented prefill, but the segmented attention kernel "
                    "cannot read a non-packed FP8 K cache (cache_dtype="
                    f"{cache_dtype!r}, neuron_config.fp8_packed_kv is unset). "
                    "Set 'fp8_packed_kv': true in neuron_config to use "
                    "segmented prefill with an FP8 KV cache, or drop "
                    "kv_segment_size_buckets to run full prefill, which reads "
                    "dense BF16 K/V."
                )
        elif non_packed_fp8_kv:
            segment_buckets = None
            logger.warning(
                "Not defaulting segmented prefill on: KV cache dtype %s is FP8 "
                "without the packed layout (neuron_config.fp8_packed_kv is "
                "unset), which the segmented attention kernel cannot read. "
                "Using full prefill. Set 'fp8_packed_kv': true to enable "
                "segmented prefill for an FP8 KV cache.",
                cache_dtype,
            )
        else:
            # Keep the default segment axis to one value. The automatic Q
            # buckets may create off-diagonal pairs, but the segment axis does
            # not multiply them further.
            segment_buckets = get_default_kv_segment_size_buckets(
                self.max_num_batched_tokens, self.max_model_len
            )

        if segment_buckets is None and getattr(
            self.vllm_config.cache_config, "enable_prefix_caching", False
        ):
            raise ValueError(
                "Automatic Prefix Caching (APC) requires segmented prefill "
                "to be enabled. Either disable APC with "
                "'--no-enable-prefix-caching' or set "
                "'max_num_batched_tokens' to a supported segmented prefill "
                f"size {sorted(SUPPORTED_KV_SEGMENT_SIZES)} to auto-enable "
                "segmented prefill."
            )

        neuron_cfg_dict["num_batched_tokens_buckets"] = prefill_buckets
        neuron_cfg_dict["num_seqs_buckets"] = decode_buckets
        neuron_cfg_dict["kv_segment_size_buckets"] = segment_buckets
        neuron_cfg_dict["decode_context_length_buckets"] = None
        return NeuronConfig.from_dict(neuron_cfg_dict)

    def _raw_neuron_config(self) -> dict:
        additional = getattr(self.vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return {}
        raw = additional.get("neuron_config")
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _select_bucket(count: int, buckets: list[int], name: str) -> int:
        """Return the smallest compiled bucket that fits ``count``."""
        if count > buckets[-1]:
            raise ValueError(
                f"{name}={count} exceeds the largest configured bucket "
                f"{buckets[-1]} ({buckets})"
            )
        from vllm_neuron.utils.bucket_utils import get_bucket_for_count

        return get_bucket_for_count(count, buckets)

    # ------------------------------------------------------------------
    # KV-cache allocation: one-shot, preflight-validated, publish-last.
    # ------------------------------------------------------------------
    def initialize_kv_cache(self, kv_cache_config) -> dict[str, list[torch.Tensor]]:
        """Allocate + bind the device KV cache from a grouped config, atomically.

        A ONE-SHOT transaction: validate the whole config on the host BEFORE any allocation, then
        allocate ONE zeroed int8 backing buffer and slice each :class:`KVCacheTensor` layer out of it
        (layers whose regions coincide share one slice),
        reinterpret each into the ``(2, num_blocks, num_kv_heads, block_size, head_size)`` K/V layout
        that the attention kernels expect, bind them to the loaded model, and publish the config +
        caches LAST. Publish-last is the failure-atomicity mechanism (matches load_model): an OOM or a
        bind failure raises before ``self._kv_cache_config`` / ``self.kv_caches`` change, so a failed
        init leaves the runner at its placeholders and a retry is permitted. A SUCCESSFUL init may not
        be repeated (the worker inits exactly once); a second call raises.

        The retry-safe guarantee is limited: it does not cover a partial ``bind_kv_cache``. The whole
        host preflight (including the pooled ``(dtype, kv_shape)`` check) runs before any state
        mutation, but the block sizes are configured and the model is bound before publication, so a
        caller that retries after a bind failure must reload the model first.

        Supports ``FullAttentionSpec`` and ``SlidingWindowSpec`` groups, and
        ``UniformTypeKVCacheSpecs`` that hold per-layer specs of those types.
        Dense, non-speculative KV transfer is supported. Speculative decoding,
        chunked attention, hybrid/Mamba caches, and FP8-packed configurations
        route to MRv1.

        Args:
            kv_cache_config: Grouped full-attention KV-cache configuration.

        Returns:
            ``{layer_name: [k_cache, v_cache]}`` -- both views of one ``(2, ...)`` buffer per layer.
        """
        from copy import deepcopy

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
            UniformTypeKVCacheSpecs,
        )

        if self._kv_cache_config is not None:
            raise RuntimeError(
                "initialize_kv_cache called twice; the standalone V2 runner initializes the KV "
                "cache exactly once"
            )
        if self.model is None:
            raise RuntimeError("load_model() must run before initialize_kv_cache()")
        if getattr(self.neuron_config, "fp8_packed_kv", False):
            raise NotImplementedError(
                "V2 standalone runner does not support the packed FP8 KV layout (fp8_packed_kv); "
                "such configs must route to V1"
            )

        # --- preflight validation (host-only; ENTIRELY before any allocation) ---
        # Resolve each group into per-layer specs. Reject unsupported spec types
        # and chunked attention, and collect the sliding window per group.
        spec_by_layer: dict[str, FullAttentionSpec] = {}
        group_of_layer: dict[str, int] = {}
        group_layers: list[str] = []
        resolved_block_sizes: list[int] = []
        group_sliding_windows: list[int | None] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            # A UniformTypeKVCacheSpecs group (the DI/KV-transfer path) carries
            # one spec per layer; any other group spec applies to every layer.
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_names = set(group.layer_names)
                spec_names = set(group_spec.kv_cache_specs)
                if group_names != spec_names:
                    raise ValueError(
                        "UniformTypeKVCacheSpecs layer-name mismatch: "
                        f"group={sorted(group_names)}, specs={sorted(spec_names)}"
                    )
                layer_specs = group_spec.kv_cache_specs
            else:
                layer_specs = {name: group_spec for name in group.layer_names}
            # The block-table trim keys off the group spec TYPE, not the
            # sliding_window field. A FullAttentionSpec that carries a window
            # (an HMA-off merged group) stays full-width. Only a
            # SlidingWindowSpec group opts into the trim.
            group_sliding_windows.append(
                group_spec.sliding_window
                if isinstance(group_spec, SlidingWindowSpec)
                else None
            )
            resolved_block_sizes.append(group_spec.block_size)
            for name in group.layer_names:
                if name in spec_by_layer:
                    raise ValueError(
                        f"layer {name!r} appears in more than one KV cache group"
                    )
                layer_spec = layer_specs[name]
                if not isinstance(layer_spec, (FullAttentionSpec, SlidingWindowSpec)):
                    raise NotImplementedError(
                        "V2 standalone runner supports only FullAttentionSpec and "
                        "SlidingWindowSpec KV layers, directly or through "
                        f"UniformTypeKVCacheSpecs; got {type(layer_spec).__name__} "
                        f"for layer {name!r}"
                    )
                if getattr(layer_spec, "attention_chunk_size", None) is not None:
                    raise NotImplementedError(
                        "V2 standalone runner does not support chunked attention "
                        f"(attention_chunk_size={layer_spec.attention_chunk_size}) "
                        f"for layer {name!r}; such configs must route to V1"
                    )
                if layer_spec.block_size != group_spec.block_size:
                    raise ValueError(
                        f"KV block-size mismatch for layer {name!r}: "
                        f"group={group_spec.block_size}, layer={layer_spec.block_size}"
                    )
                spec_by_layer[name] = layer_spec
                group_of_layer[name] = group_idx
                group_layers.append(name)

        # three-way layer-name bijection: groups == KV tensors == model spec, no dups/extras.
        tensors = kv_cache_config.kv_cache_tensors
        num_blocks = kv_cache_config.num_blocks
        for tensor in tensors:
            layer_span = num_blocks * tensor.block_stride
            if tensor.layer_stride != layer_span:
                raise NotImplementedError(
                    "V2 standalone runner needs a layer-outermost KV layout "
                    f"(layer_stride={tensor.layer_stride}, "
                    f"num_blocks*block_stride={layer_span})"
                )
            if not tensor.layers:
                raise ValueError("KVCacheTensor.layers is empty")

        # Layers whose regions start at the same byte alias one buffer. Cache groups
        # overlay each other, so co-tenants are the layers at equal
        # (offset + layer_index * layer_stride) across tensors.
        cotenants: dict[int, list[str]] = {}
        for tensor in tensors:
            for layer_index, name in enumerate(tensor.layers):
                start = tensor.offset + layer_index * tensor.layer_stride
                cotenants.setdefault(start, []).append(name)

        pooled_layers: list[str] = []
        for names in cotenants.values():
            # Two layers of the SAME group must not share one buffer: they use one block table, so a
            # shared buffer would alias their KV and corrupt state. Sharing is only valid across groups.
            seen_groups: set[int] = set()
            for name in names:
                g = group_of_layer.get(name)
                if g is not None and g in seen_groups:
                    raise ValueError(
                        f"KV buffer shared by two layers of the same group ({name!r}); "
                        "same-group layers must not share one KV buffer"
                    )
                seen_groups.add(g)
            pooled_layers.extend(names)
        if len(pooled_layers) != len(set(pooled_layers)):
            raise ValueError("a layer appears in more than one KV cache tensor")
        model_layers = {layer.name for layer in self.model.get_kv_spec().layers}
        group_set = set(group_layers)
        pooled_set = set(pooled_layers)
        if not (group_set == pooled_set == model_layers):
            raise ValueError(
                "KV cache layer-name mismatch: "
                f"groups={sorted(group_set)}, tensors={sorted(pooled_set)}, "
                f"model={sorted(model_layers)}"
            )

        # per-layer page footprint must equal the tensor's block stride; config.num_blocks
        # stays authoritative for the block count (checked per layer, not just layers[0],
        # so a mismatched co-tenant fails here, before allocation).
        for tensor in tensors:
            for name in tensor.layers:
                page = spec_by_layer[name].page_size_bytes
                if page != tensor.block_stride:
                    raise ValueError(
                        f"layer {name!r} page_size_bytes {page} does not match KV tensor "
                        f"block_stride {tensor.block_stride}; the scheduler block count "
                        f"({num_blocks}) is authoritative"
                    )

        # Pooled layers must share an identical (dtype, kv_shape). The allocation
        # below reinterprets each raw buffer ONCE per (id(raw), dtype, kv_shape) so
        # co-tenants receive the exact same K and V objects; two layers with equal
        # page footprint but a different head layout (e.g. a swapped
        # num_kv_heads/head_size split) would otherwise produce distinct slice
        # objects and silently break native slot-sized KV lowering. Reject here --
        # in the host-only preflight, BEFORE any state mutation or allocation.
        for names in cotenants.values():
            signatures = {
                name: (
                    spec_by_layer[name].dtype,
                    spec_by_layer[name].num_kv_heads,
                    spec_by_layer[name].block_size,
                    spec_by_layer[name].head_size,
                )
                for name in names
            }
            if len(set(signatures.values())) > 1:
                raise ValueError(
                    "layers pooled on one KV buffer must share the same "
                    "(dtype, num_kv_heads, block_size, head_size) so they can reuse "
                    f"exact K/V view objects; got mismatched signatures {signatures}"
                )

        # This is the first authoritative block geometry the worker receives. The constructor ran
        # before the executor applied Neuron's backend default, so configure the host block rows from
        # the same validated group specs that size the scheduler and device KV cache.
        self.request_state.configure_block_sizes(resolved_block_sizes)
        self.block_sizes = list(resolved_block_sizes)
        # Trimmed kernel block-row width per group: None for full-attention
        # groups, the window-derived width for sliding-window groups. Computed
        # once here so runtime and synthetic warmup inputs agree by
        # construction on the compiled block-table shape.
        self._swa_num_blocks = [
            (
                compute_swa_num_blocks(window, size, self.max_model_len)
                if window is not None
                else None
            )
            for window, size in zip(group_sliding_windows, resolved_block_sizes)
        ]

        # --- allocate (validation passed) ---
        kv_cache_config = deepcopy(kv_cache_config)
        sizes = {t.size for t in kv_cache_config.kv_cache_tensors}
        if len(sizes) != 1:
            raise ValueError(
                f"every KVCacheTensor must share one backing allocation; got sizes {sizes}"
            )
        backing = torch.zeros(sizes.pop(), dtype=torch.int8, device=self.device)
        raw_by_layer: dict[str, torch.Tensor] = {}
        for start, names in cotenants.items():
            span = spec_by_layer[names[0]].page_size_bytes * num_blocks
            raw = backing[start : start + span]
            for name in names:
                raw_by_layer[name] = raw

        # Reuse ONE raw.view(dtype) per (raw, dtype) so layers pooled on one buffer share a single
        # reinterpret (the AOTAutograd merge_view_inputs alias-merge; V1's closure).
        dtype_views: dict[tuple[int, torch.dtype], torch.Tensor] = {}

        def shared_dtype_view(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            key = (id(raw), dtype)
            view = dtype_views.get(key)
            if view is None:
                view = raw.view(dtype)
                dtype_views[key] = view
            return view

        # Reuse EXACT slice objects for pooled layers (#3325). Re-slicing shared
        # storage per layer gives Dynamo overlapping inputs that AOTAutograd merges
        # into a whole-buffer synthetic base, widening a slot-sized KV update into a
        # full-buffer write. The pooled-signature preflight above guarantees every
        # co-tenant resolves to the same (dtype, kv_shape) key, so one (K, V) pair
        # backs every layer on a shared buffer. id(raw) is a safe key: raw_by_layer,
        # dtype_views, and kv_slices all hold references for the whole build.
        # (k, v, full): the k and v binding views, plus the whole (2, ...) tensor.
        # The model binds [k, v]. The KV connector (DI) registers the full tensor.
        kv_slices: dict[
            tuple[int, torch.dtype, tuple[int, ...]],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

        def shared_kv_slices(
            raw: torch.Tensor,
            dtype: torch.dtype,
            kv_shape: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            key = (id(raw), dtype, kv_shape)
            slices = kv_slices.get(key)
            if slices is None:
                typed = shared_dtype_view(raw, dtype).view(kv_shape)
                slices = (typed[0], typed[1], typed)
                kv_slices[key] = slices
            return slices

        kv_caches: dict[str, list[torch.Tensor]] = {}
        # Full per-layer (2, ...) tensors for KV-connector registration (DI). The connector splits K/V
        # itself, so it wants the whole tensor, not the [k, v] model-binding view (mirrors MRv1's
        # _kv_cache_full_tensors / get_kv_cache_view_for_connector_registration).
        kv_cache_full_tensors: dict[str, torch.Tensor] = {}
        for group in kv_cache_config.kv_cache_groups:
            for name in group.layer_names:
                spec = spec_by_layer[name]
                kv_shape = (
                    2,
                    num_blocks,
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_size,
                )
                k_cache, v_cache, typed = shared_kv_slices(
                    raw_by_layer[name], spec.dtype, kv_shape
                )
                kv_caches[name] = [k_cache, v_cache]
                kv_cache_full_tensors[name] = typed

        # --- bind + publish (last) ---
        self.model.bind_kv_cache(kv_caches)
        self._kv_cache_config = kv_cache_config
        self.kv_caches = kv_caches
        # Compose the KV connector AFTER publish (DI). No-op when no KV-transfer group is
        # present; otherwise ActiveKVConnector registers the full (2, ...) views. Reused verbatim from
        # upstream MRv2 -- the connector is device-neutral and set_host_xfer_buffer_ops self-neuters on
        # Neuron (device-to-device transfer).
        self.kv_connector = get_kv_connector(self.vllm_config, kv_cache_full_tensors)
        return kv_caches

    # ------------------------------------------------------------------
    # Execute and sample: async two-phase loop.
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _serve_forward_guard(self):
        """Apply the serving guard around forward calls.

        Yields:
            None: Control while the serving guard is active.
        """
        if self._is_cpu_mode() or self._is_synthetic_model:
            yield
            return
        with model_forward_context(self.vllm_config):
            yield

    def resolve_graph_compiler_args(self, graph_type):
        """Return the concrete compiler arguments for ``graph_type``.

        Mirrors ``NeuronModelRunner.resolve_graph_compiler_args``. Read the
        base arguments from the compile options and delegate to the model's
        ``compiler_args_for`` policy. Return ``None`` when per-graph tuning is
        off (an explicit optimization level, or a model with no policy) or the
        graph type is ``None``, so the caller leaves the base arguments
        untouched.
        """
        if not getattr(self, "_per_graph_compiler_args", False) or graph_type is None:
            return None
        base = list((self.compile_options or {}).get("compiler_args") or [])
        return self._compiler_args_policy(graph_type, None, base)

    def _run_forward_adapter(self, model_input):
        """Invoke the serving adapter with one fixed-shape model input."""
        return self._forward_adapter(**self._build_forward_adapter_kwargs(model_input))

    def _build_forward_adapter_kwargs(
        self,
        model_input: NeuronModelInput,
        *,
        state_device: torch.device | None = None,
        state_tensors: dict | None = None,
    ) -> dict:
        """Build arguments for one forward-adapter call.

        Args:
            model_input: Fixed-shape operands for one step.
            state_device: Device for state-shaped placeholder tensors. When
                omitted, use the live request-state tensors.
            state_tensors: Named replacement state tensors. Idle-rank DP
                dummies pass their persistent private set so the adapter's
                unconditional in-place state writes (token_ids.index_put_,
                count updates) never alias live request state.

        Returns:
            Arguments accepted by :class:`NeuronForwardAdapter`.
        """
        max_query_len = model_input.padded_num_tokens if model_input.is_prefill else 1
        cached_seq_len = (
            torch.tensor(
                [[model_input.cached_seq_len]],
                dtype=torch.int32,
                device=model_input.positions.device,
            )
            if model_input.kv_segment_size
            else None
        )
        attn_metadata = build_attention_metadata(
            self._kv_cache_config.kv_cache_groups,
            model_input.block_tables,
            model_input.slot_mapping,
            max_query_len=max_query_len,
            decode_token_threshold=1,
            cached_seq_len=cached_seq_len,
            kv_segment_size=model_input.kv_segment_size,
            slot_mappings=model_input.slot_mappings,
            swa_kv_pos_offsets=model_input.swa_kv_pos_offsets,
        )
        rs = self.request_state

        def state_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
            if state_tensors is not None:
                return state_tensors[name]
            if state_device is None:
                return tensor
            return torch.empty(tensor.shape, dtype=tensor.dtype, device=state_device)

        rank = (
            self._rank_tensor()
            if state_device is None
            else torch.empty((1,), dtype=torch.int32, device=state_device)
        )
        return {
            "positions": model_input.positions,
            "attn_metadata": attn_metadata,
            "sampling_positions": model_input.sampling_positions,
            "row_to_slot": model_input.row_to_slot,
            "input_valid_mask": model_input.input_valid_mask,
            "slot_active_row": model_input.slot_active_row,
            "query_len_by_slot": model_input.query_len_by_slot,
            "prefill_len": model_input.prefill_len,
            "token_store": state_tensor("token_store", rs.token_ids),
            "num_computed_tokens": state_tensor(
                "num_computed_tokens", rs.num_computed_tokens
            ),
            "total_len": state_tensor("total_len", rs.total_len),
            "last_sampled_tokens": state_tensor(
                "last_sampled_tokens", rs.last_sampled_tokens
            ),
            "sampling_params": state_tensor("sampling_params", rs.sampling_params),
            "rank": rank,
            "linear_idx": model_input.linear_idx,
            "next_token_by_slot": state_tensor(
                "next_token_by_slot", self._next_token_by_slot
            ),
        }

    def _record_neff_execution(self, model_input: NeuronModelInput) -> None:
        """Record one dispatch in the compatible execution metric bucket.

        Args:
            model_input: Model input that identifies the graph bucket.
        """
        if model_input.is_prefill:
            bucket_name = f"prefill_s{model_input.padded_num_tokens}"
            if model_input.kv_segment_size:
                bucket_name += f"_kv{model_input.kv_segment_size}"
        else:
            bucket_name = f"decode_b{model_input.padded_num_reqs}"

        NEFF_EXECUTION_COUNT.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).inc()

    def execute_model(self, scheduler_output, kv_caches=None):
        """Submit one model step and save its pending output.

        Complete any required async join, update request state, and submit the
        model graph. Store the returned state tensors immediately so the next
        graph depends on them without a device-to-host wait. ``sample_tokens``
        formats the pending output.
        """
        if self._forward_adapter is None:
            raise RuntimeError(
                "The compiled forward adapter is not initialized. Run "
                "load_model() and warmup before execute_model()."
            )

        assert (
            not scheduler_output.scheduled_new_reqs
            or scheduler_output.total_num_scheduled_tokens > 0
        ), (
            "Admission must have model work in the same step. "
            "The forward fence covers the admission write."
        )

        self._drain_async_output_for_new_request_join(scheduler_output)

        # Reap completed graphs before allocation. update_states then frees
        # finished/preempted slots (quarantine-aware) and
        # admits new requests.
        self.reap_quarantined()
        with record_function_or_nullcontext("neuron_model_runner: update_states"):
            self.update_states(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens == 0:
            # A KV-only DI step still drains the connector's finished sends/receives. The no-op
            # connector returns EMPTY_MODEL_RUNNER_OUTPUT, so the non-DI path is unchanged.
            return self.kv_connector.no_forward(scheduler_output)

        # Snapshot slot-indexed block rows for KV-cache analysis so the tables
        # survive request freeing (armed by the enable_kv_snapshot RPC; no-op
        # otherwise). Taken after update_states so this step's admissions and
        # frees are reflected, and only on a step that runs the model: a
        # finished-only step (zero scheduled tokens) frees its slots in
        # update_states, and snapshotting there would overwrite the last
        # forward's valid tables with cleared (-1) rows before generate()
        # returns to the debugger.
        if self._kv_snapshot_enabled:
            self._block_table_snapshot = [
                self.request_state.block_rows_snapshot(g)
                for g in range(len(self.block_sizes))
            ]

        with record_function_or_nullcontext("neuron_model_runner: prepare_input"):
            model_input = self._prepare_model_input(scheduler_output)
        # The submitted rows this step runs (scheduler order): scope the input-snapshot
        # capture now; associate output + quarantine after the forward.
        active_req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs] + list(
            scheduler_output.scheduled_cached_reqs.req_ids
        )
        # Serve-time forward under the V1-parity no-recompile guard (fail_on_recompile): a Dynamo retrace
        # at serve time raises rather than silently recompiling. The snapshot capture scope (opt-in
        # debug, no-op when disabled) wraps only serve forwards -- never warmup -- so the capturer's
        # per-NEFF call index counts post-warmup calls, matching V1.
        # spec_window is left at its default of 1: the V2 policy lists
        # speculative decoding as unsupported and routes any config with a
        # speculative_config to V1, so no draft window exists on this path.
        # check_v2_capture_config (called at init) rejects capture+speculative
        # loudly, so removing that policy gate cannot silently skew token-rule
        # matching here — this call must then pass the real window.
        # pre_forward binds connector metadata + starts the async KV load; ActiveKVConnector.pre_forward
        # sets its own vLLM forward context when none is active (kv_connector.py:71-75), so no wrapper is
        # needed here (model_forward_context is a Dynamo stance, not the vLLM forward context). The
        # except path clears connector metadata if the forward raises after binding (no-op on the no-op
        # connector, which never binds).
        with self._snapshot_hook.capture_context(
            req_ids=active_req_ids,
            positions=model_input.positions,
            is_decode=not model_input.is_prefill,
        ):
            with self._serve_forward_guard():
                self.kv_connector.pre_forward(scheduler_output)
                try:
                    forward_output = self._run_forward_adapter(model_input)
                except Exception:
                    # Cleanup only: a failed forward has no response on which to
                    # publish the connector output, so discard it deliberately.
                    self.kv_connector.post_forward(
                        scheduler_output.finished_req_ids, wait_for_save=False
                    )
                    raise
        self._record_neff_execution(model_input)
        if self._emit_debug_logits or self._logprobs_enabled:
            if len(forward_output) != 4:
                raise RuntimeError(
                    "MRv2 expected 4 forward outputs with gathered logits enabled "
                    f"but got {len(forward_output)}; a recompile is required "
                    "(or set neuron_config.max_logprobs=0)."
                )
            sampled, bundle, next_token_by_slot, gathered_logits_future = forward_output
        else:
            sampled, bundle, next_token_by_slot = forward_output
            gathered_logits_future = None
        # Save the non-aliased next-token output for the next decode. Every TP
        # rank consumes it, which keeps sampling collectives in order.
        self._next_token_by_slot = next_token_by_slot
        # Keep this rank's forward output (its NrtaFuture) alive until the NEXT forward supersedes it, so
        # its in-flight NRT input slices are not GC-freed mid-collective on non-driver ranks (the async
        # TP hang). Replacing here (after the new forward is submitted) is safe: the prior step's future
        # is only released once this step's is in flight. No readback (no-D2H).
        self._submitted_forward_keepalive = forward_output

        # Store the state returned by this forward before the next step.
        self.apply_committed_state(*bundle)

        # Post-forward KV-connector work (DI): wait_for_save + finished sends/receives. Runs on EVERY
        # rank (execute_model runs on all ranks, unlike get_output). Carried on the pending step -- not
        # self -- so an overlapped next step cannot overwrite it before this step's response is built.
        # No-op connector returns None. active_req_ids is already built before the forward (snapshot
        # capture scope), so this path does not recompute it.
        # Ordering dependency: Neuron NIXL does not stage through a host buffer, so wait_for_save is a
        # no-op and the decode peer pulls KV. The sampled-token readback in sample_tokens (or async
        # get_output) completes this graph before kv_connector_output can be returned to the scheduler.
        kv_connector_output = self.kv_connector.post_forward(
            scheduler_output.finished_req_ids
        )
        debug_logits = self._prepare_pending_debug_logits(
            model_input,
            active_req_ids,
        )
        touched_slots = {int(s) for s in model_input.idx_mapping_np.tolist()}
        completed = {"done": False}
        self.note_graph_submitted(lambda: completed["done"], touched_slots)

        if self._logprobs_enabled:
            num_logprobs_by_row = tuple(
                int(self.request_state.num_logprobs[int(slot)])
                for slot in model_input.idx_mapping_np
            )
        else:
            num_logprobs_by_row = ()

        self._pending_step = _PendingStep(
            req_ids=list(active_req_ids),
            sampled_future=sampled,
            valid_rows=model_input.valid_rows,
            completed=completed,
            debug_logits=debug_logits,
            gathered_logits_future=gathered_logits_future,
            num_logprobs_by_row=num_logprobs_by_row,
            kv_connector_output=kv_connector_output,
        )
        return None

    def sample_tokens(self, grammar_output=None):
        """Format the pending step's output (phase 2). The state bundle was already applied in phase 1.

        Trims padded / non-final-prefill rows from the output using the HOST ``valid_rows`` (never a
        device read of a mask). Sync: materialize the sampled future to CPU now, set the completion
        fence, return the ``ModelRunnerOutput``. Async: return an :class:`AsyncModelRunnerOutput` whose
        ``get_output`` materializes on readback and sets the fence.
        """
        with record_function_or_nullcontext("neuron_model_runner: sample"):
            pending = self._pending_step
            self._pending_step = None
            if pending is None:
                raise RuntimeError(
                    "sample_tokens called without a pending execute_model step"
                )
            if not self.use_async_scheduling:
                sampled_tokens = pending.sampled_future.reshape(-1).cpu().tolist()
                token_lists = _trim_sampled(
                    sampled_tokens,
                    pending.valid_rows,
                    len(pending.req_ids),
                )

        with record_function_or_nullcontext("neuron_model_runner: postprocess"):
            if self.use_async_scheduling:
                self._async_steps += 1
                output = _AsyncNeuronModelRunnerOutput(self, pending)
                self._last_async_output = output
                return output

            self._sync_fallback_steps += 1
            logprobs = self._materialize_gathered_logits(pending, token_lists)
            pending.completed["done"] = True  # completion fence (output materialized)
            return self.get_output(
                pending.req_ids,
                token_lists,
                logprobs=logprobs,
                kv_connector_output=pending.kv_connector_output,
            )

    def _prepare_pending_debug_logits(
        self,
        model_input: NeuronModelInput,
        req_ids: list[str],
    ) -> _PendingDebugLogits | None:
        """Snapshot the exact active rows and positions for deferred publication."""
        if not self._emit_debug_logits:
            return None

        assert len(req_ids) == model_input.num_active
        assert len(model_input.valid_rows) == model_input.padded_sample_rows
        assert len(model_input.sample_position_ids) == model_input.padded_sample_rows
        assert not any(model_input.valid_rows[model_input.num_active :])

        rows = []
        final_req_ids = []
        for row, req_id in enumerate(req_ids):
            if not model_input.valid_rows[row]:
                continue
            slot = int(model_input.idx_mapping_np[row])
            position_id = model_input.sample_position_ids[row]
            # max_seq_len is prompt_len + max_tokens, so the last in-budget logit
            # sits at max_seq_len - 2. It predicts output token max_tokens - 1.
            last_position_id = int(self.request_state.max_seq_len[slot]) - 2
            if position_id > last_position_id:
                continue
            if position_id == last_position_id:
                final_req_ids.append(req_id)
            batch_idx = row
            parts = req_id.split("-")
            for index in (-2, -1):
                try:
                    batch_idx = int(parts[index])
                    break
                except (IndexError, ValueError):
                    pass
            rows.append(
                _DebugLogitRow(
                    logit_row=row,
                    batch_idx=batch_idx,
                    position_id=position_id,
                )
            )

        if not rows:
            return None
        capture_index = self._debug_logits_capture_index
        self._debug_logits_capture_index += 1
        return _PendingDebugLogits(capture_index, tuple(rows), tuple(final_req_ids))

    def _write_pending_debug_logits(
        self,
        pending: _PendingDebugLogits,
        gathered_logits_cpu: torch.Tensor,
    ) -> None:
        """Write pending debug logits to the configured output directory."""
        if not self._is_tp_rank_zero():
            return

        gathered_logits = gathered_logits_cpu.to(torch.float32).numpy()
        debug_records = [
            (row.batch_idx, row.position_id, gathered_logits[row.logit_row])
            for row in pending.rows
        ]
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        # data_parallel_index, NOT data_parallel_rank: vLLM resets
        # data_parallel_rank to 0 for dense (non-MoE) DP engines (each is an
        # independent DP=1 server), so under replica DP every engine would
        # write the same step_*_dp0.bin and clobber its peers.
        # data_parallel_index reflects the true engine index for both dense
        # and MoE engines (same rule as core assignment and CCOM rendezvous
        # in neuron_worker.py). Only a real int is trusted; anything else
        # falls back to data_parallel_rank.
        dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if not isinstance(dp_rank, int) or isinstance(dp_rank, bool):
            dp_rank = getattr(parallel_config, "data_parallel_rank", 0)

        logit_rows = [
            np.asarray(record[2], dtype=np.float32).reshape(-1)
            for record in debug_records
        ]
        vocab_size = logit_rows[0].size
        if any(row.size != vocab_size for row in logit_rows):
            raise ValueError("all debug-logit rows must have the same vocabulary size")

        os.makedirs(self._debug_logits_dir, exist_ok=True)
        tmp_path = os.path.join(
            self._debug_logits_dir,
            f"step_{pending.capture_index}_dp{dp_rank}.tmp",
        )
        final_path = os.path.join(
            self._debug_logits_dir,
            f"step_{pending.capture_index}_dp{dp_rank}.bin",
        )
        with open(tmp_path, "wb") as file:
            np.asarray([len(debug_records), vocab_size], dtype=np.int64).tofile(file)
            for (batch_idx, position_id, _), row in zip(debug_records, logit_rows):
                np.asarray([batch_idx, position_id], dtype=np.int64).tofile(file)
                row.tofile(file)
        os.rename(tmp_path, final_path)

        # Rename each sentinel AFTER the step file it closes. A reader that sees
        # the sentinel then knows every step file of that request is on disk.
        for req_id in pending.final_req_ids:
            self._write_debug_logits_sentinel(
                req_id,
                dp_rank,
                pending.capture_index,
            )

    def _write_debug_logits_sentinel(
        self,
        req_id: str,
        dp_rank: int,
        capture_index: int,
    ) -> None:
        """Write the ``done_<req_id>_dp{rank}`` sentinel of one request.

        The body holds the capture index of the request's last step file. A test
        harness waits for one sentinel per request. This gives it a deterministic
        barrier instead of a race against the writer.

        Args:
            req_id (str): Request ID as the worker sees it.
            dp_rank (int): Data parallel rank that writes the sentinel.
            capture_index (int): Capture index of the request's last step file.

        Example:
            >>> runner._write_debug_logits_sentinel("cmpl-abc-0", 0, 41)
            # writes <debug_logits_dir>/done_cmpl-abc-0_dp0 with body "41"
        """
        # A request ID can hold a path separator, so it cannot go into a filename.
        safe_req_id = req_id.replace("/", "_")
        tmp_path = os.path.join(
            self._debug_logits_dir,
            f"done_{safe_req_id}_dp{dp_rank}.tmp",
        )
        final_path = os.path.join(
            self._debug_logits_dir,
            f"done_{safe_req_id}_dp{dp_rank}",
        )
        with open(tmp_path, "w") as file:
            file.write(str(capture_index))
        os.rename(tmp_path, final_path)

    def _materialize_gathered_logits(
        self,
        pending: _PendingStep,
        token_lists: list[list[int]],
    ) -> LogprobsLists | None:
        """Materialize requested debug logits and log-probability results.

        Args:
            pending (_PendingStep): Output data for the completed model step.
            token_lists (list[list[int]]): Sampled token IDs for each request
                in output order.

        Returns:
            LogprobsLists | None: Log-probability lists aligned with the output
            requests, or ``None`` when no log-probability result is produced.

        Raises:
            RuntimeError: If required token scores are unavailable.
        """
        if not self._is_tp_rank_zero():
            return None

        num_output_rows = len(pending.req_ids)
        logprob_rows = (
            [
                row
                for row in range(num_output_rows)
                if token_lists[row] and pending.num_logprobs_by_row[row] != NO_LOGPROBS
            ]
            if self._logprobs_enabled
            else []
        )

        gathered_logits_cpu = None
        if pending.debug_logits is not None or logprob_rows:
            if pending.gathered_logits_future is None:
                raise RuntimeError(
                    "MRv2 gathered logits were requested but no logits future "
                    "was retained"
                )
            gathered_logits_cpu = pending.gathered_logits_future.cpu()

        if pending.debug_logits is not None:
            assert gathered_logits_cpu is not None
            self._write_pending_debug_logits(pending.debug_logits, gathered_logits_cpu)

        if not logprob_rows:
            return None
        assert gathered_logits_cpu is not None

        vocab_size = gathered_logits_cpu.shape[-1]
        max_num_logprobs = min(
            max(pending.num_logprobs_by_row[row] for row in logprob_rows),
            vocab_size,
        )
        requested_logits = gathered_logits_cpu[logprob_rows]
        sampled_token_ids = torch.tensor(
            [int(token_lists[row][0]) for row in logprob_rows],
            dtype=torch.int64,
        )
        return _build_sampled_token_logprobs(
            logits=requested_logits,
            sampled_token_ids=sampled_token_ids,
            output_row_indices=logprob_rows,
            max_num_logprobs=max_num_logprobs,
            num_output_rows=num_output_rows,
            logprobs_mode=self._logprobs_mode,
        )

    def _is_tp_rank_zero(self) -> bool:
        """Return whether this worker is the sole writer for its TP group."""
        from vllm.distributed.parallel_state import get_tp_group

        try:
            tp_group = get_tp_group()
        except AssertionError:
            if not (self._is_cpu_mode() or self._is_synthetic_model):
                raise
            return True
        return tp_group is None or tp_group.rank_in_group == 0

    def _drain_async_output_for_new_request_join(self, scheduler_output) -> None:
        """Complete the previous lazy TP sampling collective before admitting a new request.

        Steady decode keeps its device-to-device chain and never enters this path. A newly scheduled
        request can instead run a prefill whose prompt inputs do not consume the previous decode's sampled
        future. Since the executor materializes async output only on output_rank, that would strand the
        output rank inside distributed argmax. Materializing the retained wrapper here runs on every TP
        rank because ``execute_model`` is an all-rank RPC.

        The latest materialization also completes every earlier graph in the ordered device chain, so all
        pre-join quarantine records can be retired before admission.
        """
        if (
            not self.use_async_scheduling
            or not scheduler_output.scheduled_new_reqs
            or self._last_async_output is None
        ):
            return

        self._last_async_output.get_output()
        self._last_async_output = None
        self._inflight_graphs.clear()

    def _rank_tensor(self):
        """Build and cache the tensor-parallel rank tensor.

        Use the worker's tensor-parallel local rank for vocabulary-shard
        embedding and sampling offsets. Fall back to rank zero only during
        single-shard CPU simulation.

        Returns:
            torch.Tensor: A cached int32 ``[1]`` rank tensor on the runner
            device.

        Raises:
            AssertionError: If the tensor-parallel group is unavailable during
                device execution.
        """
        cached = getattr(self, "_rank_tensor_cache", None)
        if cached is None:
            from vllm.distributed.parallel_state import get_tp_group

            # get_tp_group() ASSERTS the group is initialized. In CPU-simulation tests it is not, so fall
            # back to rank 0 (single-shard) only there. On the real device the group MUST be initialized;
            # do not swallow the assertion there: a silent rank-0 fallback
            # would corrupt shard-local embedding and token offsets.
            try:
                tp_group = get_tp_group()
                tp_rank = tp_group.rank_in_group if tp_group is not None else 0
            except AssertionError:
                if not (self._is_cpu_mode() or self._is_synthetic_model):
                    raise
                tp_rank = 0
            cached = torch.full((1,), tp_rank, dtype=torch.int32, device=self.device)
            self._rank_tensor_cache = cached
        return cached

    # ------------------------------------------------------------------
    # Compile and warmup.
    # ------------------------------------------------------------------
    def redecorate_compiled_model(self) -> None:
        """Compile the forward adapter again, so it loads the flushed package.

        Examples:
            >>> runner.redecorate_compiled_model()  # doctest: +SKIP
        """
        self._compile_forward_adapter()

    def _compile_forward_adapter(self) -> None:
        """Compile and publish the forward adapter for the loaded model.

        Compile the adapter unless eager mode or the synthetic model is in use.
        When supported, also publish a graph-capture adapter for parallel
        compilation. Debug mode allows graph breaks and disables graph capture.

        Raises:
            RuntimeError: If the model is not loaded or device eager execution
                is requested.
        """
        if self.model is None:
            raise RuntimeError(
                "load_model() must run before _compile_forward_adapter()"
            )
        adapter = NeuronForwardAdapter(
            self.model,
            emit_gathered_logits=(self._emit_debug_logits or self._logprobs_enabled),
        )

        # Match Model Runner V1: configured buckets intentionally specialize
        # the compiled forward, so a static Dynamo cache limit adds no value.
        torch._dynamo.config.cache_size_limit = 2**62

        cpu_mode = self._is_cpu_mode()
        eager_mode = self.vllm_config.model_config.enforce_eager
        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        if eager_mode and not cpu_mode:
            raise RuntimeError(
                "enforce_eager is not supported on Neuron for the V2 runner; "
                "it is permitted only in CPU mode."
            )
        if eager_mode:
            torch.compiler.set_stance("force_eager")
        if debug_mode:
            logger.info(
                "VLLM_NEURON_DEBUG_MODE enabled: fullgraph=False "
                "(allows print statements and graph breaks)"
            )
        if eager_mode or self._is_synthetic_model:
            self._forward_adapter = adapter
            self.capture_backend_model = None
            return

        from libtorch_neuronx_lite import envs as libtorch_envs

        skip_graph_capture_backend = (
            libtorch_envs.NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND
        )
        fullgraph_enabled = not debug_mode
        if (
            cpu_mode
            or envs.is_native_backend()
            or debug_mode
            or skip_graph_capture_backend
        ):
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                adapter,
                backend="neuron_libtorch_graph_capture",
                fullgraph=fullgraph_enabled,
                options=self.compile_options,
            )

        self._forward_adapter = torch.compile(
            adapter,
            backend=envs.get_compile_backend_name(),
            fullgraph=fullgraph_enabled,
            options=self.compile_options,
        )

    def _materialize_output(self, output) -> None:
        """Bring every tensor in a forward output (the ``(sampled, bundle, next_token_by_slot)`` nest) to CPU.

        Async ``self._forward_adapter(...)`` returns device futures; warmup discards the result, so
        without an explicit readback the future never resolves and lingers in the NRT queue. Recurse so
        the sampled tensor, the 4-tuple state bundle, AND the ordinary-future carry are all drained.
        """
        if torch.is_tensor(output):
            output.cpu()
        elif isinstance(output, (tuple, list)):
            for item in output:
                self._materialize_output(item)

    def _warmup(self, model_input) -> None:
        """Run one synthetic input through the adapter.

        Materialize asynchronous output so all warmup futures complete.

        Args:
            model_input (NeuronModelInput): Synthetic fixed-shape graph input.

        Raises:
            RuntimeError: If the adapter or KV cache has not been initialized.
        """
        if self._forward_adapter is None:
            raise RuntimeError(
                "_compile_forward_adapter() must run before warmup (adapter not published)"
            )
        if self._kv_cache_config is None:
            raise RuntimeError("initialize_kv_cache() must run before warmup")
        output = self._run_forward_adapter(model_input)
        if self.use_async_scheduling:
            self._materialize_output(output)

    def warmup_prefill(self, bucket_size: int, kv_segment_size: int = 0) -> None:
        """Compile one configured prefill graph via the synthetic prefill input."""
        self._validate_prefill_compile_target(bucket_size, kv_segment_size)
        compile_start = time.perf_counter()
        self._warmup(self._build_prefill_synthetic_inputs(bucket_size, kv_segment_size))
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"prefill_s{bucket_size}"
        if kv_segment_size > 0:
            bucket_name += f"_kv{kv_segment_size}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)

    def _capture_model_input(self, model_input: NeuronModelInput) -> None:
        """Capture one synthetic input with the graph-capture adapter."""
        if self.capture_backend_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_device = model_input.positions.device
        kwargs = self._build_forward_adapter_kwargs(
            model_input,
            state_device=None if input_device == self.device else input_device,
        )
        try:
            self.capture_backend_model(**kwargs)
        except CaptureComplete:
            pass

    def extract_prefill_graphs(
        self,
        bucket_size: int,
        kv_segment_size: int,
        device: torch.device | None = None,
    ) -> None:
        """Capture one prefill graph in the current process."""
        logger.info(
            "Capturing prefill graphs for: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )
        model_input = self._build_prefill_synthetic_inputs(
            bucket_size,
            kv_segment_size,
            device=device,
        )
        self._capture_model_input(model_input)
        logger.debug(
            "Graph capture for prefill completed: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )

    def extract_decode_graphs(
        self,
        batch_size: int,
        context_len: int = 256,
        ctx_bucket: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Capture one decode graph in the current process."""
        logger.info("Capturing decode graphs for: batch_size=%s", batch_size)
        model_input = self._build_decode_synthetic_inputs(
            batch_size,
            device=device,
        )
        # Scope the decode graph's per-graph compiler args (mirrors V1's
        # _plain_decode_capture); a no-op (None) when per-graph tuning is off.
        with scoped_compiler_args(self.resolve_graph_compiler_args("decode")):
            self._capture_model_input(model_input)
        logger.debug("Graph capture for decode completed: batch_size=%s", batch_size)

    def warmup_decode(
        self, batch_size: int, context_len: int = 0, ctx_bucket=None
    ) -> None:
        """Compile one configured decode graph via the synthetic decode input."""
        self._validate_decode_bucket(batch_size)
        compile_start = time.perf_counter()
        # Scope the decode graph's per-graph compiler args (mirrors V1's
        # _plain_decode_forward); a no-op (None) when per-graph tuning is off.
        with scoped_compiler_args(self.resolve_graph_compiler_args("decode")):
            self._warmup(self._build_decode_synthetic_inputs(batch_size))
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"decode_b{batch_size}"
        if ctx_bucket is not None:
            bucket_name += f"_ctx{ctx_bucket}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)

    def parallel_compile(self) -> None:
        """Compile captured HLOs once across this tensor-parallel group."""
        if self.capture_backend_model is None:
            return

        from vllm.distributed.parallel_state import get_tp_group

        from libtorch_neuronx_lite import envs as libtorch_envs
        from libtorch_neuronx_lite.compile.parallel_compile import (
            parallel_compile as neuron_parallel_compile,
        )

        tp_group = get_tp_group()
        neuron_parallel_compile(
            options=self.compile_options,
            # Graph capture uses rank-scoped cache directories whenever
            # torch.distributed is initialized, including TP=1. Preserve that
            # layout contract by always selecting distributed discovery here.
            rank=tp_group.rank_in_group,
            world_size=tp_group.world_size,
            remote_cache_dir=libtorch_envs.NEURON_LIBTORCH_REMOTE_CACHE,
        )

    def _validate_prefill_compile_target(
        self, bucket_size: int, kv_segment_size: int
    ) -> None:
        config = self.neuron_config
        if bucket_size not in config.num_batched_tokens_buckets:
            raise ValueError(
                f"prefill bucket {bucket_size} is not configured; expected one of "
                f"{config.num_batched_tokens_buckets}"
            )
        segment_buckets = config.kv_segment_size_buckets or [0]
        if kv_segment_size not in segment_buckets:
            raise ValueError(
                f"KV segment size {kv_segment_size} is not configured; expected one of "
                f"{segment_buckets}"
            )

    def _validate_decode_bucket(self, batch_size: int) -> None:
        buckets = self.num_seqs_buckets
        if batch_size not in buckets:
            raise ValueError(
                f"decode batch bucket {batch_size} is not configured; expected one of "
                f"{buckets}"
            )

    def _dp_collectives_active(self) -> bool:
        """Whether cross-DP collectives fire inside this deployment's graphs.

        True only for expert parallelism with data_parallel_size > 1: the MoE
        all-to-all then spans DP ranks and every participant must dispatch a
        NEFF with the same token dimension. Replica DP (dense) runs each
        engine as an independent DP=1 process and never reaches here.
        """
        parallel_config = self.vllm_config.parallel_config
        return getattr(parallel_config, "data_parallel_size", 1) > 1 and getattr(
            parallel_config, "enable_expert_parallel", False
        )

    def _dp_sync_shape(self, prefill_tokens: int, decode_rows: int) -> tuple[int, int]:
        """MAX-reduce the per-step shape vote across the DP group, per phase.

        The vote is a 2-element ``[prefill_tokens, decode_rows]`` tensor: a
        prefilling rank votes ``(tokens, 0)``, a decoding rank ``(0, rows)``,
        an idle rank ``(0, 0)``. The two phases MUST stay in separate channels:
        a single scalar lets a prefill token count reach an idle rank that
        reads it as a decode row count and selects from the wrong bucket list.
        Every rank runs exactly one collective per step, so the reduce pairs
        by order regardless of phase.

        The reduced values are the INPUT to bucket selection; every rank then
        picks the bucket independently, keeping the picker pure (boundaries are
        identical across ranks). CPU-side reduce: the result selects the NEFF
        to dispatch and must materialize as a Python int before any device op.

        On V2 the four extra inputs the V1 runner reduces are structurally
        constant (``decode_context_length_buckets`` forced None; speculative /
        KV transfer / structured outputs route to V1), asserted below so a gate
        change fails loudly instead of silently desyncing the group.
        """
        assert self.neuron_config.decode_context_length_buckets is None, (
            "V2 DP sync assumes decode_context_length_buckets is None; "
            "syncing the decode context bucket across DP ranks is not implemented"
        )
        assert self.speculative_config is None, (
            "V2 DP sync assumes no speculative decoding; the spec/non-spec "
            "NEFF vote is not synchronized across DP ranks"
        )
        if not self._dp_collectives_active():
            return prefill_tokens, decode_rows

        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group().cpu_group
        t = torch.tensor([prefill_tokens, decode_rows], device="cpu", dtype=torch.int32)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
        group_tokens, group_rows = int(t[0]), int(t[1])
        if group_tokens > 0 and group_rows > 0:
            # Hard raise (not assert, which -O strips): a bypassed guard lets
            # an idle rank dispatch a prefill NEFF into a decode rank's
            # all-to-all — a silent shape mismatch.
            raise RuntimeError(
                f"DP ranks disagree on phase this step: prefill_tokens={group_tokens} "
                f"and decode_rows={group_rows} are both nonzero. Mixed prefill/decode "
                "phases across the EP-spanning DP group dispatch NEFFs with different "
                "token dimensions and are not reconciled on V2 (same posture as V1, "
                "where production wide-EP separates prefill and decode fleets via "
                "disaggregated inference)."
            )
        return group_tokens, group_rows

    def _build_dp_dummy_inputs(
        self, *, is_prefill: bool, token_bucket: int, row_bucket: int
    ) -> NeuronModelInput:
        """Build a zero-active-row input for an idle-rank DP dummy dispatch.

        Runtime dummies MUST NOT touch live request state: unlike startup
        warmup, live requests may occupy slots when a dummy fires. An empty
        active set keeps every row padding (``input_valid_mask`` and
        ``valid_rows`` all False) and every slot-mapping entry
        ``NULL_BLOCK_ID``, so KV writes land in the reserved null block and no
        token store is read or written. The graph still dispatches at the
        group's bucket, which is all the cross-DP collective needs.
        """
        empty32 = np.zeros(0, dtype=np.int32)
        empty64 = np.zeros(0, dtype=np.int64)
        return self._assemble_model_input(
            idx_mapping_np=empty32,
            num_computed=empty64,
            query_lens=empty64,
            prefill_lens_active=empty64,
            is_prefill=is_prefill,
            token_bucket=token_bucket,
            row_bucket=row_bucket,
            kv_segment_size=self.kv_segment_size if is_prefill else 0,
        )

    def execute_dummy_batch(self) -> None:
        """Run an idle-rank synthetic dispatch for cross-DP EP synchronization.

        With expert parallelism across DP ranks, the MoE all-to-all inside the
        NEFF spans the DP group: a rank with no scheduled requests must still
        dispatch a graph with the group's token dimension or the busy ranks'
        collectives deadlock. Participate in the shape sync with a zero vote,
        read the group's phase from which channel is nonzero, and dispatch a
        zero-active-row input at the group's bucket (prefill or decode).

        The output is NOT read back: nobody consumes a dummy's tokens, and a
        device-to-host copy would land on the critical path between
        consecutive NEFFs. A Python reference to the returned future is held
        instead (same rationale as the V1 dummy path / #3960): dropping it
        would let GC free the in-flight NRT input slices. The reference is
        superseded by the next dispatch.  Under replica DP (dense) no
        cross-DP collective exists and this remains a no-op.
        """
        if not self._dp_collectives_active():
            return None
        group_tokens, group_rows = self._dp_sync_shape(0, 0)
        if group_tokens > 0:
            token_bucket = self._select_bucket(
                group_tokens,
                self.neuron_config.num_batched_tokens_buckets,
                "dp_dummy_tokens",
            )
            model_input = self._build_dp_dummy_inputs(
                is_prefill=True, token_bucket=token_bucket, row_bucket=1
            )
        elif group_rows > 0:
            row_bucket = self._select_bucket(
                group_rows, self.num_seqs_buckets, "dp_dummy_rows"
            )
            model_input = self._build_dp_dummy_inputs(
                is_prefill=False, token_bucket=row_bucket, row_bucket=row_bucket
            )
        else:
            return None
        with self._serve_forward_guard():
            self._dp_dummy_output_keepalive = self._forward_adapter(
                **self._build_forward_adapter_kwargs(
                    model_input, state_tensors=self._dp_dummy_state_tensors()
                )
            )
        return None

    def _dp_dummy_state_tensors(self) -> dict:
        """Persistent private state tensors for idle-rank DP dummies.

        The forward adapter writes its state operands in place
        unconditionally (``token_ids.index_put_`` writes back the old value
        on non-committing rows), and a dummy future is not tracked in
        ``_inflight_graphs`` and therefore never drained before admission
        copies into request state. Binding live state tensors would be an
        aliased concurrent mutation; the dummy binds this private set
        instead. Allocated once (zeros match request-state startup values)
        and reused so every dummy dispatch aliases the same stable buffers.
        """
        if self._dp_dummy_state is None:
            rs = self.request_state
            self._dp_dummy_state = {
                "token_store": torch.zeros_like(rs.token_ids),
                "num_computed_tokens": torch.zeros_like(rs.num_computed_tokens),
                "total_len": torch.zeros_like(rs.total_len),
                "last_sampled_tokens": torch.zeros_like(rs.last_sampled_tokens),
                "sampling_params": torch.zeros_like(rs.sampling_params),
                "next_token_by_slot": torch.zeros_like(self._next_token_by_slot),
            }
        return self._dp_dummy_state

    # ------------------------------------------------------------------
    # State returned by compiled graphs.
    # ------------------------------------------------------------------
    def apply_committed_state(
        self,
        token_ids,
        num_computed_tokens,
        total_len,
        last_sampled_tokens,
    ) -> None:
        """Store the state tensors returned by the model graph.

        Token history aliases its input, while the smaller state vectors are
        new outputs. Saving those exact tensors makes the next graph depend on
        the current graph.

        The new-request join completes an earlier graph before admission
        updates persistent state.
        """
        rs = self.request_state
        rs.token_ids = token_ids
        rs.num_computed_tokens = num_computed_tokens
        rs.total_len = total_len
        rs.last_sampled_tokens = last_sampled_tokens

    # ------------------------------------------------------------------
    # Quarantine ledger for async recycle safety.
    # ------------------------------------------------------------------
    def note_graph_submitted(self, is_complete, touched_slots: set[int]) -> None:
        """Record an in-flight submitted graph and the slots it may touch.

        Args:
            is_complete: a zero-arg callable returning True once the graph's device execution has
                actually completed (the fence). ``touched_slots`` must CONSERVATIVELY include every
                slot the graph may access.
        """
        self._inflight_graphs.append((is_complete, set(touched_slots)))

    def _slot_has_inflight(self, slot: int) -> bool:
        """Return whether an incomplete graph may still touch a slot.

        Args:
            slot: Stable request slot to inspect.

        Returns:
            ``True`` when an incomplete submitted graph includes the slot.
        """
        return any(
            slot in touched and not is_complete()
            for is_complete, touched in self._inflight_graphs
        )

    def release_or_quarantine_slot(self, slot: int) -> None:
        """Return ``slot`` to the free list, or quarantine it if an in-flight graph still touches it.

        Default-quarantine: a slot freed while a prior producer graph may still write it must NOT be
        reused (a recycle overwrite could land on the stale occupant's in-flight write). Reaped later
        by :meth:`reap_quarantined` once the touching graphs complete.
        """
        rs = self.request_state
        if self._slot_has_inflight(slot):
            if slot in rs.free_indices:
                rs.free_indices.remove(slot)
            self._quarantined_slots.add(slot)
        else:
            self._quarantined_slots.discard(slot)
            if slot not in rs.free_indices:
                rs.free_indices.append(slot)

    def reap_quarantined(self) -> None:
        """Drop completed in-flight graphs, then free any quarantined slot no graph still touches."""
        self._inflight_graphs = [
            (is_complete, touched)
            for is_complete, touched in self._inflight_graphs
            if not is_complete()
        ]
        rs = self.request_state
        # Iterate the set in a canonical order for reproducibility.
        # NOT load-bearing — reap timing is rank-local so freed-slot list order may still differ across
        # ranks; correctness comes from equal free-set MEMBERSHIP at allocation + min() selection.
        for slot in sorted(self._quarantined_slots):
            if not self._slot_has_inflight(slot):
                self._quarantined_slots.discard(slot)
                if slot not in rs.free_indices:
                    rs.free_indices.append(slot)

    # ------------------------------------------------------------------
    # Response-only output formatting.
    # ------------------------------------------------------------------
    def get_output(
        self,
        req_ids: list[str],
        sampled_token_ids: list[list[int]],
        logprobs=None,
        kv_connector_output=None,
    ) -> ModelRunnerOutput:
        """Format this step's sampled tokens into a ``ModelRunnerOutput`` — response only.

        Advances NO device-authoritative state: the token/count/history/last-sampled advance is done
        in the compiled producer epilogue, returned as graph outputs,
        and threaded back via :meth:`apply_committed_state`. This method only packages the response.

        Association is by SUBMITTED row (``req_ids`` in the step's batch order), never the slot's
        current occupant: a recycle between submit and output must not misattribute a token to a
        different request (the cross-request leak). ``req_ids`` and ``sampled_token_ids`` are the rows
        the step ran, so the mapping is positional and independent of the live slot table.
        """
        return ModelRunnerOutput(
            req_ids=list(req_ids),
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[list(toks) for toks in sampled_token_ids],
            logprobs=logprobs,
            kv_connector_output=kv_connector_output,
        )

    # ------------------------------------------------------------------
    # Per-step input preparation.
    # ------------------------------------------------------------------
    def _prepare_model_input(self, scheduler_output) -> NeuronModelInput:
        """Build fixed-shape operands for one scheduled step.

        Resolve active rows in scheduler order from :class:`NeuronRequestState`
        and derive positions, block rows, slot mappings, gather indices,
        sampling operands, and commit operands. Build all row mappings from one
        active request list. Read only host advisory progress and immutable
        prefill lengths so asynchronous device execution remains unsynchronized.

        Args:
            scheduler_output (SchedulerOutput): Step already applied to request
                state by ``update_states``.

        Returns:
            The flat device tensors and host metadata for the step.

        Raises:
            AssertionError: If the scheduled step is empty, inconsistent,
                heterogeneous, over capacity, or otherwise violates the
                supported prefill or decode shape.
        """
        rs = self.request_state

        # 1) Active rows in scheduler order (new first, then cached), one source for every operand.
        active_req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs] + list(
            scheduler_output.scheduled_cached_reqs.req_ids
        )
        num_active = len(active_req_ids)
        assert scheduler_output.total_num_scheduled_tokens > 0, (
            "V2 _prepare_model_input called with an empty step"
        )
        assert num_active > 0, "no active requests but total_num_scheduled_tokens > 0"
        assert num_active <= self.max_num_reqs, (
            f"num_active {num_active} > max_num_reqs {self.max_num_reqs}"
        )
        # The scheduled-token dict must name exactly the active set (no missing / stray / duplicate ids).
        assert len(set(active_req_ids)) == num_active, (
            f"duplicate id in active order {active_req_ids}"
        )
        assert set(active_req_ids) == set(scheduler_output.num_scheduled_tokens), (
            "num_scheduled_tokens keys do not match the active req ids"
        )

        idx_mapping_np = rs.build_idx_mapping(active_req_ids)
        slots = idx_mapping_np.astype(np.int64)

        # 2) Per-row host vectors in active order (advisory count = window start; no device read).
        num_computed = rs.scheduler_num_computed_tokens[slots].astype(np.int64)
        query_lens = np.array(
            [scheduler_output.num_scheduled_tokens[r] for r in active_req_ids],
            dtype=np.int64,
        )
        prefill_lens_active = rs.prefill_len[slots].astype(np.int64)

        # Classify each row by its padded query width: one token indicates
        # decode, while a larger width indicates prefill.
        req_id_to_padded_token_count = scheduler_output.num_scheduled_tokens_padded
        req_ids_missing_padded_token_count = [
            r for r in active_req_ids if r not in req_id_to_padded_token_count
        ]
        if req_ids_missing_padded_token_count:
            raise RuntimeError(
                "V2 runner requires scheduler_output.num_scheduled_tokens_padded "
                "with an entry for every scheduled request; missing "
                f"{req_ids_missing_padded_token_count!r}."
            )
        padded_token_counts = np.array(
            [req_id_to_padded_token_count[req_id] for req_id in active_req_ids],
            dtype=np.int64,
        )
        # Standard prefill uses a multi-token bucket. In DI, the first decode
        # invocation processes the final prompt token, so a single-token input
        # is handled by the decode graph.
        is_prefill = bool(padded_token_counts.max() > 1)
        if is_prefill:
            assert bool((num_computed < prefill_lens_active).all()), (
                "V2 prefill classified (padded>1) but a row has num_computed >= prefill_len: "
                f"num_computed {num_computed.tolist()}, prefill_len {prefill_lens_active.tolist()}"
            )
            if num_active != 1:
                raise RuntimeError(
                    "V2 prefill requires exactly one active request row per "
                    f"scheduler step; got {num_active}"
                )
            assert int(num_computed[0] + query_lens[0]) <= int(
                prefill_lens_active[0]
            ), (
                f"prefill overshoots prefill_len: num_computed {int(num_computed[0])} + query_len "
                f"{int(query_lens[0])} > prefill_len {int(prefill_lens_active[0])}"
            )
            # Sync the token-bucket INPUT across the DP group, then pick
            # locally (see _dp_sync_shape).
            group_tokens, _ = self._dp_sync_shape(
                int(scheduler_output.total_num_scheduled_tokens), 0
            )
            token_bucket = self._select_bucket(
                group_tokens,
                self.neuron_config.num_batched_tokens_buckets,
                "total_num_scheduled_tokens",
            )
            row_bucket = 1
            kv_segment_size = self.kv_segment_size
        else:
            assert bool((num_computed >= prefill_lens_active - 1).all()), (
                "V2 decode classified (padded==1) but a row is mid-prompt "
                "(num_computed < prefill_len - 1): "
                f"num_computed {num_computed.tolist()}, prefill_len {prefill_lens_active.tolist()}"
            )
            assert (query_lens == 1).all(), (
                f"V2 first cell decode is 1 token/row (non-spec); got query_lens {query_lens.tolist()}"
            )
            _, group_rows = self._dp_sync_shape(0, num_active)
            row_bucket = self._select_bucket(
                group_rows,
                self.num_seqs_buckets,
                "num_active",
            )
            token_bucket = row_bucket
            kv_segment_size = 0

        return self._assemble_model_input(
            idx_mapping_np=idx_mapping_np,
            num_computed=num_computed,
            query_lens=query_lens,
            prefill_lens_active=prefill_lens_active,
            is_prefill=is_prefill,
            token_bucket=token_bucket,
            row_bucket=row_bucket,
            kv_segment_size=kv_segment_size,
        )

    def _assemble_model_input(
        self,
        *,
        idx_mapping_np: np.ndarray,
        num_computed: np.ndarray,
        query_lens: np.ndarray,
        prefill_lens_active: np.ndarray,
        is_prefill: bool,
        token_bucket: int,
        row_bucket: int,
        kv_segment_size: int,
        device: torch.device | None = None,
    ) -> NeuronModelInput:
        """Pad host vectors and assemble a fixed-shape model input.

        Share this assembly and upload path between real execution and
        synthetic warmup. Keep slot-keyed operands at ``max_num_reqs`` and size
        token- and row-oriented operands to the selected graph bucket.

        Args:
            idx_mapping_np: Stable slot for each active scheduler row.
            num_computed: Host advisory computed-token count per active row.
            query_lens: Scheduled token count per active row.
            prefill_lens_active: Prefill completion length per active row.
            is_prefill: Whether the graph flavor is prefill.
            token_bucket: Compiled token capacity.
            row_bucket: Compiled row capacity.
            kv_segment_size: Segmented-prefill KV length, or zero.
            device: Optional tensor destination. Defaults to the runner device.

        Returns:
            The fixed-shape device operands and host metadata.

        Raises:
            AssertionError: If active rows, scored tokens, positions, or
                sampling rows exceed the selected capacities.
        """
        rs = self.request_state
        device = self.device if device is None else device
        state_capacity = self.max_num_reqs
        num_active = idx_mapping_np.shape[0]
        slots = idx_mapping_np.astype(np.int64)
        assert num_active <= row_bucket, (
            f"active rows {num_active} exceed selected row bucket {row_bucket}"
        )

        # Graph shapes for this flavor. R = sampled-row count (1 prefill / decode row bucket).
        padded_num_tokens = token_bucket
        padded_num_reqs = row_bucket
        num_sample_rows = row_bucket

        # Flat positions + per-token row index (scheduler order), covering fresh / prefix-cache /
        # chunked windows uniformly.
        positions_flat, req_indices = build_flat_positions(num_computed, query_lens)
        total_scored = positions_flat.shape[0]
        # Token capacity is the flavor's own width (P prefill / B decode) -- NOT P for decode, since a
        # valid config can have B > P (e.g. a small chunked-prefill max_num_batched_tokens with a large
        # max_num_seqs), and a full decode batch has total_scored == B.
        assert total_scored <= padded_num_tokens, (
            f"scored tokens {total_scored} > padded_num_tokens {padded_num_tokens}"
        )
        assert total_scored == 0 or positions_flat.max() < self.max_model_len, (
            f"position >= max_model_len ({self.max_model_len})"
        )

        # Block rows per KV group: raw (zero-padded) feeds the slot mapping; kernel (-1 sentinel) is
        # the attention block table. Both are padded to the flavor's row capacity. Sliding-window
        # groups get window-trimmed kernel rows plus a per-row KV position offset on decode; prefill
        # always uses the full table (KV writes span every scheduled slot and the prefill graph is
        # compiled for the full shape). Slot mappings are per group: each group owns its block table,
        # so KV writes must resolve slots through their own group's (untrimmed) rows.
        block_tables = []
        swa_kv_pos_offsets: list[torch.Tensor | None] = []
        slot_mappings_flat: list[np.ndarray] = []
        for g in range(len(self.block_sizes)):
            swa_width = self._swa_num_blocks[g]
            if swa_width is not None and not is_prefill:
                rows, offsets = build_swa_kernel_block_rows(
                    rs,
                    g,
                    idx_mapping_np,
                    padded_num_reqs,
                    num_computed_np=num_computed,
                    num_swa_blocks=swa_width,
                    block_size=self.block_sizes[g],
                    output_device=device,
                )
            else:
                rows = build_kernel_block_rows(
                    rs,
                    g,
                    idx_mapping_np,
                    padded_num_reqs,
                    output_device=device,
                )
                offsets = None
            block_tables.append(rows)
            swa_kv_pos_offsets.append(offsets)
            raw_g = gather_block_rows_raw(rs, g, idx_mapping_np, padded_num_reqs)
            slot_mappings_flat.append(
                build_slot_mapping(
                    raw_g, positions_flat, req_indices, self.block_sizes[g]
                )
            )

        # Pad positions + slot mapping to the fixed token width (pad positions 0, pad slot NULL_BLOCK_ID).
        positions = np.zeros(padded_num_tokens, dtype=np.int32)
        positions[:total_scored] = positions_flat
        slot_mappings: list[np.ndarray] = []
        for flat in slot_mappings_flat:
            padded = np.full(padded_num_tokens, NULL_BLOCK_ID, dtype=np.int64)
            padded[:total_scored] = flat
            slot_mappings.append(padded)
        slot_mapping = slot_mappings[0]

        # row_to_slot [R] (common): the sampling_params row-select on both flavors + the decode store
        # gather. Pad rows carry the in-bounds slot 0 (never -1), masked out by input_valid_mask.
        row_to_slot = np.zeros(num_sample_rows, dtype=np.int64)
        row_to_slot[:num_active] = slots[:num_active]

        # input_valid_mask marks real tokens (prefill, [P]) / real rows (decode, [B]).
        if is_prefill:
            linear_idx, input_valid_mask_np = build_prefill_linear_idx(
                idx_mapping_np,
                num_computed,
                query_lens,
                self.max_model_len,
                token_bucket,
            )
        else:
            linear_idx = None
            input_valid_mask_np = np.zeros(row_bucket, dtype=bool)
            input_valid_mask_np[:num_active] = True

        # sampling_positions [R] (row -> FLAT hidden-token row): decode is row-aligned arange(B);
        # prefill (R=1) is the single request's last scored token in the flat [P] layout.
        sampling_positions = np.zeros(num_sample_rows, dtype=np.int64)
        if is_prefill:
            sampling_positions[0] = max(total_scored - 1, 0)
        else:
            sampling_positions[:] = np.arange(row_bucket, dtype=np.int64)

        # A non-final chunked-prefill row advances its count but discards its
        # sampled token; decode padding rows do not produce output.
        if is_prefill:
            final_chunk = num_active == 1 and (
                int(num_computed[0] + query_lens[0]) >= int(prefill_lens_active[0])
            )
            valid_rows = (final_chunk,)
        else:
            valid_rows = (True,) * num_active + (False,) * (
                num_sample_rows - num_active
            )

        # Commit operands [max_num_reqs] (slot-keyed, the INVERSE of row_to_slot): route each active slot to its
        # output row + carry its query length; idle slots are -1 / 0. prefill_len is the whole-state
        # host array uploaded as a fixed state-capacity operand (no device prefill_len state).
        slot_active_row = np.full(state_capacity, -1, dtype=np.int32)
        query_len_by_slot = np.zeros(state_capacity, dtype=np.int32)
        for row in range(num_active):
            slot = int(slots[row])
            slot_active_row[slot] = row
            query_len_by_slot[slot] = int(query_lens[row])
        prefill_len_operand = rs.prefill_len.astype(np.int32)  # [B]

        sample_position_ids = tuple(
            int(positions[position]) for position in sampling_positions
        )

        # Segmented (chunked) prefill operands: the cached prefix length for the single active prefill row
        # + the resolved singleton segment size. Decode, non-segmented prefill, and zero-active-row DP
        # dummies pass 0.
        cached_seq_len = (
            int(num_computed[0])
            if (is_prefill and kv_segment_size and num_active)
            else 0
        )
        # APC-without-segmentation guard: a prefix-cache hit (num_computed>0) on SINGLE-SHOT prefill
        # (kv_segment_size==0) would run flash_attention, which cannot attend the cached prefix -> wrong
        # output. On Neuron, prefix caching requires segmented prefill. Fail loud rather than silently
        # dropping the prefix. (Decode reads num_computed>0 legitimately -- this guards prefill only.)
        if (
            is_prefill
            and not kv_segment_size
            and num_active
            and int(num_computed.max()) > 0
        ):
            raise RuntimeError(
                "V2 runner invariant violated: received prefill with "
                "num_computed>0 but no KV segment size. Startup validation "
                "requires segmented prefill when prefix caching is enabled, "
                "and the scheduler must not chunk single-shot prefill."
            )

        def _to_dev(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

        return NeuronModelInput(
            positions=_to_dev(positions),
            block_tables=block_tables,
            slot_mapping=_to_dev(slot_mapping),
            sampling_positions=_to_dev(sampling_positions),
            row_to_slot=_to_dev(row_to_slot),
            input_valid_mask=_to_dev(input_valid_mask_np),
            slot_active_row=_to_dev(slot_active_row),
            query_len_by_slot=_to_dev(query_len_by_slot),
            prefill_len=_to_dev(prefill_len_operand),
            idx_mapping_np=idx_mapping_np,
            sample_position_ids=sample_position_ids,
            is_prefill=is_prefill,
            num_active=num_active,
            padded_num_tokens=padded_num_tokens,
            padded_num_reqs=padded_num_reqs,
            padded_sample_rows=num_sample_rows,
            valid_rows=valid_rows,
            cached_seq_len=cached_seq_len,
            kv_segment_size=kv_segment_size,
            linear_idx=_to_dev(linear_idx) if linear_idx is not None else None,
            slot_mappings=tuple(_to_dev(m) for m in slot_mappings),
            swa_kv_pos_offsets=tuple(swa_kv_pos_offsets),
        )

    # ------------------------------------------------------------------
    # Synthetic input builders used by warmup.
    # ------------------------------------------------------------------
    def _build_prefill_synthetic_inputs(
        self,
        bucket_size: int | None = None,
        kv_segment_size: int | None = None,
        device: torch.device | None = None,
    ) -> NeuronModelInput:
        """Build a synthetic input for one prefill graph shape.

        Fill the selected token bucket with one request in slot zero. Use the
        same assembly path as real execution so warmup matches serving shapes
        and dtypes.

        Args:
            bucket_size: Optional configured prefill token capacity.
            kv_segment_size: Optional configured KV segment size.
            device: Optional tensor destination. Defaults to the runner device.

        Returns:
            Synthetic fixed-shape prefill input.
        """
        if bucket_size is None:
            bucket_size = self.prefill_token_capacity
        if kv_segment_size is None:
            kv_segment_size = self.kv_segment_size
        idx_mapping_np = np.zeros(1, dtype=np.int32)
        num_computed = np.zeros(1, dtype=np.int64)
        query_lens = np.array([bucket_size], dtype=np.int64)
        # prefill_len long enough that the synthetic chunk is "final" (shape is what matters, not value).
        prefill_lens_active = np.array([bucket_size], dtype=np.int64)
        return self._assemble_model_input(
            idx_mapping_np=idx_mapping_np,
            num_computed=num_computed,
            query_lens=query_lens,
            prefill_lens_active=prefill_lens_active,
            is_prefill=True,
            token_bucket=bucket_size,
            row_bucket=1,
            kv_segment_size=kv_segment_size,
            device=device,
        )

    def _build_decode_synthetic_inputs(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> NeuronModelInput:
        """Build a synthetic input for one configured decode graph shape.

        Warmup runs this through the compiled forward to fix the decode NEFF
        shape. Fill every selected bucket row over matching synthetic slots.

        Args:
            batch_size: Optional configured decode row capacity.
            device: Optional tensor destination. Defaults to the runner device.

        Returns:
            Synthetic fixed-shape decode input.
        """
        if batch_size is None:
            batch_size = self.decode_row_capacity
        idx_mapping_np = np.arange(batch_size, dtype=np.int32)
        num_computed = np.zeros(batch_size, dtype=np.int64)
        query_lens = np.ones(batch_size, dtype=np.int64)
        # decode requires num_computed >= prefill_len; make prefill_len 0 so every row reads as decode.
        prefill_lens_active = np.zeros(batch_size, dtype=np.int64)
        return self._assemble_model_input(
            idx_mapping_np=idx_mapping_np,
            num_computed=num_computed,
            query_lens=query_lens,
            prefill_lens_active=prefill_lens_active,
            is_prefill=False,
            token_bucket=batch_size,
            row_bucket=batch_size,
            kv_segment_size=0,
            device=device,
        )

    # ------------------------------------------------------------------
    # scheduler-driven lifecycle (upstream _update_states semantics, no condense())
    # ------------------------------------------------------------------
    def update_states(self, scheduler_output) -> None:
        """Apply one scheduler step to the request state.

        Follow upstream order: free finished and preempted slots, add newly
        scheduled requests, then update running requests. Admission copies one
        state row. Other updates change host metadata and block tables. Model
        execution and output commit happen later.
        """
        rs = self.request_state

        # 1) Free finished slots through the quarantine ledger, withholding
        #    slots from reuse while a producer graph is in flight.
        #    Iterate SORTED (finished_req_ids is an unordered set[str]) purely for a canonical, reproducible
        #    processing order. This is NOT load-bearing for rank correctness: deterministic slot SELECTION
        #    (min() in add_request) makes allocation depend only on the free-set MEMBERSHIP, not on
        #    free_indices list order — and rank-local reap timing can legitimately leave different list
        #    orders across ranks. The correctness invariant is: at ALLOCATION time the free SET is equal on
        #    every rank (the new-request join drains all-rank + reaps before admitting), so min() agrees.
        for req_id in sorted(scheduler_output.finished_req_ids):
            self._remove_and_quarantine(req_id)

        # 2) Free preempted slots (the set may be None when nothing was preempted) -- same routing, sorted.
        for req_id in sorted(scheduler_output.preempted_req_ids or ()):
            self._remove_and_quarantine(req_id)

        # Add newly scheduled requests with their token counts after padding.
        req_id_to_padded_token_count = getattr(
            scheduler_output, "num_scheduled_tokens_padded", None
        )
        for new_req in scheduler_output.scheduled_new_reqs:
            if (
                not isinstance(req_id_to_padded_token_count, dict)
                or new_req.req_id not in req_id_to_padded_token_count
            ):
                raise RuntimeError(
                    "V2 runner requires scheduler_output.num_scheduled_tokens_padded with an entry "
                    f"for every scheduled new request; missing {new_req.req_id!r}"
                )
            self._add_new_request(new_req)

        # 4) Update running requests (advisory count + appended blocks). Under the V2 scheduler,
        #    resumed requests are folded into scheduled_new_reqs with full history and the cached
        #    resumed set is cleared, so a nonempty cached resumed_req_ids is a contract violation.
        #    CachedRequestData's fields are always lists (required dataclass fields), so read them
        #    directly — a `getattr(..., []) or []` would mask a malformed None as an empty step.
        cached = scheduler_output.scheduled_cached_reqs
        assert not cached.resumed_req_ids, (
            f"V2 runner expects cached resumed_req_ids empty (resumed reqs are re-added as new); "
            f"got {cached.resumed_req_ids}"
        )
        req_ids = cached.req_ids
        num_computed = cached.num_computed_tokens
        new_block_ids = cached.new_block_ids
        # The scheduler emits one entry per cached req; mismatched lengths mean a malformed step.
        assert len(num_computed) == len(req_ids), (
            f"cached num_computed_tokens length {len(num_computed)} != req_ids {len(req_ids)}"
        )
        assert len(new_block_ids) == len(req_ids), (
            f"cached new_block_ids length {len(new_block_ids)} != req_ids {len(req_ids)}"
        )
        for i, req_id in enumerate(req_ids):
            rs.update_cached_request(
                req_id,
                scheduler_num_computed_tokens=num_computed[i],
                new_block_ids=new_block_ids[i],
            )

    def _remove_and_quarantine(self, req_id: str) -> None:
        """Free ``req_id``'s slot through the quarantine ledger (no-op if it holds no slot).

        ``remove_request`` returns the slot to ``free_indices``; ``release_or_quarantine_slot`` then
        WITHDRAWS it again if a submitted producer graph may still touch it (default-on quarantine),
        else leaves it free. Routing every removal here is what makes the async recycle safe -- a bare
        ``remove_request`` would immediately re-expose an in-flight slot.
        """
        slot = self.request_state.remove_request(req_id)
        if slot is not None:
            self.release_or_quarantine_slot(slot)

    def _release_resident_for_readmission(self, req_id: str) -> None:
        """Release a validated same-ID resident without exposing an in-flight slot.

        Raise before mutation when the resident slot needs quarantine and no
        other slot is free, preserving the live request on failed readmission.
        """
        rs = self.request_state
        slot = rs.req_id_to_index[req_id]
        if not rs.free_indices and self._slot_has_inflight(slot):
            raise RuntimeError(
                f"cannot readmit {req_id!r}: its slot is in flight and no other slot is free"
            )
        self._remove_and_quarantine(req_id)

    def _add_new_request(self, new_req) -> None:
        """Resolve and admit one new scheduler request.

        Route a still-resident request through slot quarantine before
        readmission. Let :meth:`NeuronRequestState.add_request` validate the
        replacement before mutating state so malformed input cannot destroy a
        live request.

        Args:
            new_req (NewRequestData): Scheduler data for admission or resume.

        Raises:
            AssertionError: If required prompt or prefill token IDs are absent.
        """
        req_id = new_req.req_id
        rs = self.request_state

        # The first cell rejects prompt_embeds, so prompt_token_ids is always a real list here; a
        # None is a contract violation, not an empty prompt.
        prompt_token_ids = new_req.prompt_token_ids
        assert prompt_token_ids is not None, (
            f"V2 runner requires prompt_token_ids on {req_id!r} (prompt_embeds is not supported)"
        )
        # The V2 scheduler ALWAYS supplies prefill_token_ids (req._all_token_ids) for new AND resumed
        # requests. A missing value is a scheduler/config contract violation, not a case to paper
        # over with a prompt fallback — fail loud so the wrong history never reaches the device.
        prefill_token_ids = new_req.prefill_token_ids
        assert prefill_token_ids is not None, (
            f"V2 runner requires prefill_token_ids on every new request; {req_id!r} had None "
            "(is use_v2_model_runner set on the scheduler?)"
        )

        sampling_params = new_req.sampling_params
        max_tokens = getattr(sampling_params, "max_tokens", None) or 0

        num_computed_tokens = new_req.num_computed_tokens

        rs.add_request(
            req_id=req_id,
            prompt_len=len(prompt_token_ids),
            prefill_token_ids=list(prefill_token_ids),
            num_computed_tokens=num_computed_tokens,
            max_tokens=max_tokens,
            block_ids=new_req.block_ids,
            sampling_params=sampling_params,
            release_resident=self._release_resident_for_readmission,
        )
