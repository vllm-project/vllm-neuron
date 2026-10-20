# SPDX-License-Identifier: Apache-2.0
"""DFlash block-parallel speculative draft proposer."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import torch
import torch.nn as nn
from libtorch_neuronx_lite import envs as libtorch_envs
from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_neuron import envs
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from vllm_neuron.model.registry import get_models
from vllm_neuron.utils.neuron_utils import model_forward_context

logger = logging.getLogger(__name__)


class DFlashProposer:
    """DFlash proposer backed by a non-causal block drafter.

    DFlash precomputes draft KV from target auxiliary hidden states, then runs
    one block-parallel query pass to produce all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ) -> None:
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        if self.method != "dflash":
            raise ValueError(
                f"DFlashProposer requires method='dflash'; got {self.method!r}"
            )

        self.device = device
        self.on_device_sampling = on_device_sampling
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        num_verify_tokens = max_num_seqs * (self.num_speculative_tokens + 1)
        # TODO: Remove this guard once the distributed argmax kernel supports
        # more than 256 verification rows.
        assert num_verify_tokens <= 256, (
            "DFlash currently supports at most 256 verification tokens; "
            f"max_num_seqs={max_num_seqs} with "
            f"num_speculative_tokens={self.num_speculative_tokens} would verify "
            f"{num_verify_tokens} tokens at once"
        )

        draft_hf_config = self.draft_model_config.hf_config
        nested_dflash_config = getattr(draft_hf_config, "dflash_config", {}) or {}
        if "block_size" in nested_dflash_config:
            checkpoint_block_size = nested_dflash_config["block_size"]
        elif hasattr(draft_hf_config, "block_size"):
            checkpoint_block_size = draft_hf_config.block_size
        else:
            raise ValueError(
                "DFlash draft config must define block_size either directly "
                "or in dflash_config"
            )

        checkpoint_draft_width = checkpoint_block_size - 1
        if self.num_speculative_tokens != checkpoint_draft_width:
            raise ValueError(
                "DFlash num_speculative_tokens must equal the draft checkpoint width "
                f"(block_size - 1 = {checkpoint_draft_width}); got "
                f"{self.num_speculative_tokens} for block_size={checkpoint_block_size}"
            )

        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.capture_backend_model: nn.Module | None = None

    def _build_noop_async_spec_correction_kwargs(
        self,
        num_tokens: int,
        num_reqs: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build no-op correction tensors for warmup and graph capture."""
        if device is None:
            device = self.device

        prev_sampled_token_ids = torch.zeros(
            num_reqs,
            self.num_speculative_tokens + 1,
            dtype=torch.int32,
            device=device,
        )
        prev_num_draft_tokens = torch.full(
            (num_reqs,),
            self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )

        assert num_reqs > 0
        assert num_tokens % num_reqs == 0
        tokens_per_req = num_tokens // num_reqs
        req_indices_per_token = (
            torch.arange(num_reqs, dtype=torch.int64)
            .repeat_interleave(tokens_per_req)
            .to(device)
        )

        return {
            "prev_sampled_token_ids": prev_sampled_token_ids,
            "prev_num_draft_tokens": prev_num_draft_tokens,
            "req_indices_per_token": req_indices_per_token,
        }

    def load_model(
        self,
        target_hidden_size: int | None = None,
        target_hidden_pad_groups: int = 1,
        compile_options: dict | None = None,
    ) -> None:
        """Load and compile the DFlash draft model.

        Args:
            target_hidden_size: Hidden size emitted by the compiled target model.
            target_hidden_pad_groups: Layout of the target's padded hidden dim.
                DFlash supports only the tail-padded layout (1).
            compile_options: Target model's torch.compile options, so the draft
                NEFF is built with the same compiler args.

        Example:
            >>> proposer.load_model(target_hidden_size=4096)
        """
        if target_hidden_size is None:
            raise ValueError(
                "DFlashProposer requires target_hidden_size because it consumes "
                "target hidden states."
            )
        if target_hidden_pad_groups != 1:
            # A deinterleaved target hidden layout needs the same fc treatment as
            # the Eagle3 draft. Raise rather than warn: the draft would read
            # scrambled hidden states and propose useless tokens, which costs
            # decode time and produces no speedup. Every DFlash target today
            # pads its hidden dim at the tail, so this path is unreachable until
            # somebody adds a target that does not.
            raise NotImplementedError(
                f"The target deinterleaves its padded hidden dim into "
                f"{target_hidden_pad_groups} groups, which the DFlash draft does "
                f"not follow. Add the layout to the draft's fc weight loader, or "
                f"serve this target without DFlash speculation."
            )

        target_text_config = self.vllm_config.model_config.hf_text_config
        target_num_layers = target_text_config.num_hidden_layers
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=target_num_layers,
            target_hidden_size=target_hidden_size,
            compile_options=compile_options,
        )
        logger.info("Completed compilation for DFlash draft model.")

        draft_num_layers = self.draft_model_config.hf_config.num_hidden_layers
        self.attn_layer_names = [
            f"layers.{i}.self_attn"
            for i in range(target_num_layers, target_num_layers + draft_num_layers)
        ]

    def _build_synthetic_inputs(
        self,
        num_tokens: int,
        num_reqs: int,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Build synthetic inputs for DFlash warmup and graph capture."""
        assert self.model is not None
        if device is None:
            device = self.device

        hidden_size = self.model.config.hidden_size
        num_aux = self._num_aux_hidden_state_layers()
        target_token_ids = torch.ones(num_tokens, dtype=torch.int32, device=device)
        target_positions = torch.zeros(num_tokens, dtype=torch.long, device=device)
        target_hidden_states = torch.ones(
            num_tokens,
            hidden_size * num_aux,
            dtype=torch.bfloat16,
            device=device,
        )

        tokens_per_req = num_tokens // num_reqs
        last_token_indices = torch.tensor(
            [(i + 1) * tokens_per_req - 1 for i in range(num_reqs)],
            dtype=torch.long,
            device=device,
        )

        if num_tokens == num_reqs * (1 + self.num_speculative_tokens):
            raw_sampled_cols = self.num_speculative_tokens + 1
        else:
            raw_sampled_cols = 1
        raw_sampled_token_ids = torch.ones(
            num_reqs,
            raw_sampled_cols,
            dtype=torch.int32,
            device=device,
        )
        cur_num_draft_tokens = torch.full(
            (num_reqs,),
            self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )

        return {
            "target_token_ids": target_token_ids,
            "target_positions": target_positions,
            "target_hidden_states": target_hidden_states,
            "last_token_indices": last_token_indices,
            "raw_sampled_token_ids": raw_sampled_token_ids,
            "cur_num_draft_tokens": cur_num_draft_tokens,
            **self._build_noop_async_spec_correction_kwargs(
                num_tokens, num_reqs, device=device
            ),
        }

    def warmup(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """Warm up the compiled DFlash proposal path.

        Args:
            num_tokens: Total synthetic token count.
            num_reqs: Number of synthetic requests.
            attn_metadata: Attention metadata for the warmup bucket.

        Example:
            >>> proposer.warmup(10, 1, attn_metadata)
        """
        assert self.model is not None
        logger.info(
            "DFlash warmup: num_tokens=%d, num_reqs=%d",
            num_tokens,
            num_reqs,
        )

        draft_output = self.propose(
            attn_metadata=attn_metadata,
            is_warmup=True,
            **self._build_synthetic_inputs(num_tokens, num_reqs),
        )

        if self.vllm_config.scheduler_config.async_scheduling:
            for out in draft_output:
                if torch.is_tensor(out):
                    out.cpu()

    def graph_extract(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
        device: torch.device | None = None,
    ) -> None:
        """Extract the DFlash HLO graph with the capture backend.

        Args:
            num_tokens: Total number of tokens.
            num_reqs: Number of requests in the batch.
            attn_metadata: Full attention metadata.
            device: Optional device for synthetic inputs.

        Example:
            >>> proposer.graph_extract(10, 1, attn_metadata)
        """
        if self.capture_backend_model is None:
            logger.debug(
                "DFlash graph extraction skipped (capture_backend_model is None)"
            )
            return

        logger.info(
            "DFlash graph capture: num_tokens=%d, num_reqs=%d",
            num_tokens,
            num_reqs,
        )
        try:
            self.propose(
                attn_metadata=attn_metadata,
                is_warmup=True,
                model_override=self.capture_backend_model,
                **self._build_synthetic_inputs(num_tokens, num_reqs, device=device),
            )
        except CaptureComplete:
            logger.debug(
                "DFlash graph capture completed: num_tokens=%d, num_reqs=%d",
                num_tokens,
                num_reqs,
            )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Validate that all DFlash layers share one KV-cache group."""
        kv_cache_groups: dict[str, int] = {}
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = group_id
        assert (
            len({kv_cache_groups[layer_name] for layer_name in self.attn_layer_names})
            == 1
        ), "All DFlash layers should belong to the same KV-cache group"

    def compile_and_load_draft_model(
        self,
        start_layer_idx: int,
        target_hidden_size: int | None = None,
        compile_options: dict | None = None,
    ) -> nn.Module:
        """Compile and load the DFlash model implementation."""
        draft_model_arch = self.draft_model_config.architecture
        vllm_neuron_models = dict(get_models())
        if draft_model_arch not in vllm_neuron_models:
            raise ValueError(
                f"Draft model architecture '{draft_model_arch}' not found in "
                "vLLM Neuron model registry. Supported architectures: "
                f"{list(vllm_neuron_models.keys())}"
            )
        model_cls = vllm_neuron_models[draft_model_arch]

        draft_hf_config = self.draft_model_config.hf_config
        draft_hf_config.unpadded_hidden_size = draft_hf_config.hidden_size
        if (
            target_hidden_size is not None
            and target_hidden_size != draft_hf_config.hidden_size
        ):
            draft_hf_config.hidden_size = target_hidden_size

        on_device_sampling_config = None
        if self.on_device_sampling:
            on_device_sampling_config = OnDeviceSamplingConfig(all_greedy=True)
        neuron_config = NeuronConfig(
            on_device_sampling_config=on_device_sampling_config,
        )

        with torch.device("meta"):
            self.model = model_cls.from_configs(
                config=draft_hf_config,
                start_layer_idx=start_layer_idx,
                neuron_config=neuron_config,
            )
        self.model.num_speculative_tokens = self.num_speculative_tokens

        load_start = time.perf_counter()
        if not envs.VLLM_NEURON_CPU_COMPILE:
            self.model.load_weights(
                self.speculative_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )
            logger.info("Moving DFlash draft model to device: %s", self.device)
            self.model = self.model.to(self.device)

            draft_model_name = self.speculative_config.model
            MODEL_LOAD_TIME.labels(model_name=draft_model_name).set(
                time.perf_counter() - load_start
            )
            MODEL_LOAD_SIZE.labels(model_name=draft_model_name).set(
                sum(p.nbytes for p in self.model.parameters())
            )
        else:
            if hasattr(self.model, "load_weights_lite"):
                self.model.load_weights_lite(
                    self.speculative_config.model,
                    torch.device("cpu"),
                    self.vllm_config.load_config.download_dir,
                )
                logger.info("Light weight loading complete for DFlash CPU compilation.")
            logger.info(
                "CPU compilation is enabled. Skipping full DFlash weight loading."
            )
            self.model = self.model.to("meta")

        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        fullgraph_enabled = not debug_mode
        compile_backend = envs.get_compile_backend_name()

        cpu_mode = envs.VLLM_NEURON_CPU_MODE
        eager_mode = self.vllm_config.model_config.enforce_eager
        skip_graph_capture_backend = (
            libtorch_envs.NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND
        )
        compile_kwargs: dict = {"fullgraph": fullgraph_enabled}
        if compile_options is not None:
            compile_kwargs["options"] = compile_options

        if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
            logger.debug(
                "DFlash graph capture backend disabled "
                "(eager/cpu/debug/skip_graph_capture_backend)."
            )
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                self.model,
                backend="neuron_libtorch_graph_capture",
                **compile_kwargs,
            )

        logger.info("Compiling DFlash draft model with %s", compile_backend)
        # Kept so the runner can wrap the draft model again after the package flush.
        self.compile_kwargs = {"backend": compile_backend, **compile_kwargs}
        return torch.compile(self.model, **self.compile_kwargs)

    def _target_layer_ids_from_config(self, hf_config: Any) -> list[int]:
        dflash_cfg = getattr(hf_config, "dflash_config", None)
        if isinstance(dflash_cfg, dict) and "target_layer_ids" in dflash_cfg:
            return list(dflash_cfg["target_layer_ids"])
        if hasattr(hf_config, "target_layer_ids"):
            return list(hf_config.target_layer_ids)

        raise ValueError(
            "DFlash draft config must provide target_layer_ids either directly "
            "or under dflash_config"
        )

    def get_target_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Return target layers that the DFlash drafter consumes.

        DFlash checkpoints store HF output-hidden-state layer ids. Neuron target
        capture records pre-layer states, so the runner must request each id + 1.
        For a reduced target used only for profiling, fall back to evenly spaced
        capture layers when the checkpoint's trained layer ids do not fit.
        """
        target_layer_ids = self._target_layer_ids_from_config(
            self.draft_model_config.hf_config
        )
        num_aux_layers = len(target_layer_ids)
        target_text_config = self.vllm_config.model_config.hf_text_config
        target_num_layers = target_text_config.num_hidden_layers

        assert target_num_layers >= num_aux_layers, (
            f"DFlash needs >= {num_aux_layers} target layers to source "
            f"{num_aux_layers} aux hidden states, got {target_num_layers}"
        )

        configured_capture_layers = tuple(t + 1 for t in target_layer_ids)
        if all(0 <= layer < target_num_layers for layer in configured_capture_layers):
            return configured_capture_layers

        step = target_num_layers / num_aux_layers
        fallback_capture_layers = tuple(
            int(step * (layer_idx + 0.5)) for layer_idx in range(num_aux_layers)
        )
        logger.warning(
            "DFlash checkpoint target layers %s map to capture layers %s, but "
            "the target has only %d layers. Using evenly spaced capture layers "
            "%s for profiling only; generation quality is not expected to be "
            "valid. This may indicate that the wrong DFlash checkpoint was "
            "selected for the target model.",
            target_layer_ids,
            configured_capture_layers,
            target_num_layers,
            fallback_capture_layers,
        )
        return fallback_capture_layers

    def _num_aux_hidden_state_layers(self) -> int:
        target_layer_ids = self._target_layer_ids_from_config(
            self.draft_model_config.hf_config
        )
        return len(target_layer_ids)

    def alias_target_weights(self, target_model: nn.Module) -> None:
        """Alias target weights after both models have been loaded."""
        if getattr(self, "_target_weights_aliased", False):
            return
        assert self.model is not None

        base = getattr(self.model, "_orig_mod", self.model)
        assert target_model is not None, (
            "DFlash requires the loaded target model to alias embed_tokens and "
            "lm_head because the draft checkpoint does not ship them."
        )
        base.alias_target_head(target_model)
        self._target_weights_aliased = True
        logger.info("Prepared fused DFlash model forward with target head")

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: AttentionMetadata,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
        cur_num_draft_tokens: torch.Tensor | None = None,
        is_warmup: bool = False,
        model_override: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose DFlash draft tokens for the current target step.

        Args:
            target_token_ids: Unused target ids, kept for proposer interface
                compatibility.
            target_positions: Absolute token positions for this step.
            target_hidden_states: Concatenated DFlash target aux hidden states.
            last_token_indices: Per-request sampled-token indices.
            attn_metadata: Draft attention metadata containing the block table.
            raw_sampled_token_ids: Rejection sampler output or sampled target ids.
            prev_sampled_token_ids: Previous-step rejection sampler output.
            prev_num_draft_tokens: Previous-step proposed draft counts.
            req_indices_per_token: Request index for each target token row.
            cur_num_draft_tokens: Current-step proposed draft counts.
            is_warmup: Whether this call is a compile/warmup invocation.
            model_override: Optional capture-backend model.

        Returns:
            Tuple of ``(draft_token_ids, drafts_only)``. ``draft_token_ids`` has
            shape ``[num_reqs, 1 + num_speculative_tokens]`` and includes the
            bonus token followed by drafts. ``drafts_only`` has shape
            ``[num_reqs, num_speculative_tokens]`` and is used by rejection
            sampling.
        """
        assert self.model is not None
        assert attn_metadata, "DFlash requires draft attention metadata"
        _ = target_token_ids
        model = model_override if model_override is not None else self.model

        batch_size = last_token_indices.shape[0]
        device = target_hidden_states.device
        target_positions = target_positions.to(device)
        last_token_indices = last_token_indices.to(device)
        raw_sampled_token_ids = raw_sampled_token_ids.to(device)

        if prev_sampled_token_ids is None:
            correction_kwargs = self._build_noop_async_spec_correction_kwargs(
                target_hidden_states.shape[0],
                batch_size,
                device=device,
            )
            prev_sampled_token_ids = correction_kwargs["prev_sampled_token_ids"]
            prev_num_draft_tokens = correction_kwargs["prev_num_draft_tokens"]
            req_indices_per_token = correction_kwargs["req_indices_per_token"]
        else:
            assert prev_num_draft_tokens is not None
            assert req_indices_per_token is not None
            prev_sampled_token_ids = prev_sampled_token_ids.to(device)
            prev_num_draft_tokens = prev_num_draft_tokens.to(device)
            req_indices_per_token = req_indices_per_token.to(device)

        if cur_num_draft_tokens is None:
            cur_num_draft_tokens = torch.full(
                (batch_size,),
                self.num_speculative_tokens,
                dtype=torch.int32,
                device=device,
            )
        else:
            cur_num_draft_tokens = cur_num_draft_tokens.to(device)

        base_model = getattr(self.model, "_orig_mod", self.model)
        backbone = base_model.model
        block_len = backbone.config.block_size
        model_forward_start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            model_output = model(
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                last_token_indices=last_token_indices,
                attn_metadata=attn_metadata,
                raw_sampled_token_ids=raw_sampled_token_ids,
                prev_sampled_token_ids=prev_sampled_token_ids,
                prev_num_draft_tokens=prev_num_draft_tokens,
                req_indices_per_token=req_indices_per_token,
                cur_num_draft_tokens=cur_num_draft_tokens,
            )
            draft_token_ids, drafts_only = model_output[:2]
        model_forward_elapsed = time.perf_counter() - model_forward_start

        bucket_name = f"dflash_block{block_len}"
        if is_warmup:
            COMPILATION_TIME.labels(
                model_name=self.speculative_config.model,
                bucket_name=bucket_name,
            ).set(model_forward_elapsed)
        else:
            NEFF_EXECUTION_COUNT.labels(
                model_name=self.speculative_config.model,
                bucket_name=bucket_name,
            ).inc()
        return draft_token_ids, drafts_only
