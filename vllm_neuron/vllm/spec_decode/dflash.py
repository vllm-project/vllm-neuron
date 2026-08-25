# SPDX-License-Identifier: Apache-2.0
"""DFlash parallel draft proposer for the Neuron model runner."""

import contextlib
import time

import torch
from libtorch_neuronx_lite.compile.platform import get_platform_target

from vllm_neuron.metrics import COMPILATION_TIME, NEFF_EXECUTION_COUNT
from vllm_neuron.utils.neuron_utils import model_forward_context

from .eagle import EagleProposer

# Targets whose Neuron implementation captures auxiliary hidden states in the
# layout DFlash drafters consume (pre-layer residuals, concatenated in
# ascending layer order).
SUPPORTED_TARGET_TYPES = frozenset({"gpt_oss", "llama", "qwen3"})


def dflash_target_capture_layer_ids(hf_config) -> tuple[int, ...] | None:
    """Translate DFlash decoder-output IDs to Neuron residual boundaries.

    The z-lab checkpoint IDs are zero-based decoder layer outputs. Neuron's
    target model captures the residual stream immediately before a layer, so
    output of decoder layer ``i`` is the boundary before layer ``i + 1``.
    """
    dflash_config = getattr(hf_config, "dflash_config", None)
    if not isinstance(dflash_config, dict):
        return None
    layer_ids = dflash_config.get("target_layer_ids")
    if not layer_ids or not isinstance(layer_ids, (list, tuple)):
        return None
    return tuple(int(layer_id) + 1 for layer_id in layer_ids)


class DFlashProposer(EagleProposer):
    """Generate all DFlash proposal tokens in one non-causal draft pass."""

    expected_method = "dflash"

    def __init__(self, vllm_config, device, on_device_sampling=True):
        super().__init__(vllm_config, device, on_device_sampling)
        if not on_device_sampling:
            raise ValueError("DFlash on Neuron requires on-device sampling")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "DFlash on Neuron initially supports synchronous scheduling only; "
                "pass --no-async-scheduling"
            )
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "DFlash on Neuron does not yet support chunked prefill; "
                "pass --no-enable-chunked-prefill"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError(
                "DFlash on Neuron does not yet support disaggregated inference"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("DFlash on Neuron does not yet support prefix caching")
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if tp_size not in (4, 8):
            raise ValueError(
                "DFlash on Neuron is validated for tensor_parallel_size 4 "
                f"(single Trn2 chip, LNC=2) or 8; got {tp_size}"
            )
        platform = get_platform_target()
        if platform != "trn2":
            raise ValueError(
                "DFlash support is validated only on Trn2; "
                f"detected platform {platform!r}"
            )
        target_quantization = vllm_config.additional_config.get(
            "neuron_config", {}
        ).get("quantization")
        if target_quantization not in (None, "bf16"):
            raise ValueError(
                f"Trn2 DFlash requires quantization='bf16', got {target_quantization!r}"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Trn2 DFlash requires target dtype=bfloat16")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("DFlash requires a BF16 KV cache")

        target_config = vllm_config.model_config.hf_text_config
        target_type = target_config.model_type
        if target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(
                f"DFlash on Neuron supports targets {sorted(SUPPORTED_TARGET_TYPES)}, "
                f"got {target_type!r}"
            )

        # Validate the drafter against the target it is paired with, rather than
        # against one hard-coded checkpoint. A DFlash drafter is trained for a
        # specific target: it reuses that target's embedding and LM head, and
        # consumes its hidden states, so these must line up exactly.
        draft_config = self.draft_model_config.hf_config
        dflash_config = getattr(draft_config, "dflash_config", None) or {}
        layer_ids = dflash_config.get("target_layer_ids")
        if not layer_ids:
            raise ValueError(
                "DFlash checkpoint config must define dflash_config.target_layer_ids"
            )
        if dflash_config.get("mask_token_id") is None:
            raise ValueError(
                "DFlash checkpoint config must define dflash_config.mask_token_id"
            )

        mismatches = []
        if draft_config.hidden_size != target_config.hidden_size:
            mismatches.append(
                f"hidden_size draft={draft_config.hidden_size} "
                f"target={target_config.hidden_size}"
            )
        if draft_config.vocab_size != target_config.vocab_size:
            mismatches.append(
                f"vocab_size draft={draft_config.vocab_size} "
                f"target={target_config.vocab_size}"
            )
        num_target_layers = getattr(draft_config, "num_target_layers", None)
        if (
            num_target_layers is not None
            and num_target_layers != target_config.num_hidden_layers
        ):
            mismatches.append(
                f"num_target_layers draft={num_target_layers} "
                f"target={target_config.num_hidden_layers}"
            )
        if max(layer_ids) + 1 >= target_config.num_hidden_layers:
            mismatches.append(
                f"target_layer_ids {layer_ids} exceed the target's "
                f"{target_config.num_hidden_layers} layers"
            )
        if mismatches:
            raise ValueError(
                "DFlash drafter does not match this target: " + "; ".join(mismatches)
            )

        # The proposal block size is fixed by training, so num_speculative_tokens
        # is not free: the drafter fills block_size - 1 masked slots per pass.
        expected_spec_tokens = draft_config.block_size - 1
        if self.num_speculative_tokens != expected_spec_tokens:
            raise ValueError(
                f"This DFlash checkpoint has block_size={draft_config.block_size}, "
                f"so it requires num_speculative_tokens={expected_spec_tokens}; "
                f"got {self.num_speculative_tokens}"
            )

    def _build_synthetic_inputs(self, num_tokens, num_reqs, device=None):
        """Warmup inputs for the DFlash draft NEFF.

        Delegates to the EAGLE3 builder for everything shape-critical — in
        particular ``raw_sampled_token_ids``, whose column count decides which
        branch of ``_last_valid_context_indices`` is traced ([bs, 1] after a
        prefill, [bs, 1 + num_spec] in verified decode). Only the one thing
        DFlash genuinely differs on is overridden: the target feature stack is
        one column per captured layer rather than EAGLE3's fixed three.

        The inherited all-zero ``target_positions`` are kept deliberately. The
        proposal block sits at ``target_positions[last_token_indices] + 1``, so
        real positions from the largest prefill bucket would place it at
        ``max_model_len``, one block past the end of the block table, and the
        query slot mapping would index out of bounds. Zeros put the block at
        positions 1..block_size, and warmup only needs the shapes to match.
        """
        assert self.model is not None
        if device is None:
            device = self.device
        kwargs = super()._build_synthetic_inputs(num_tokens, num_reqs, device=device)

        hidden_size = self.model.config.hidden_size
        num_features = len(self.model.target_layer_ids)
        kwargs["target_hidden_states"] = torch.ones(
            num_tokens,
            hidden_size * num_features,
            dtype=torch.bfloat16,
            device=device,
        )
        return kwargs

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        last_token_indices,
        attn_metadata,
        raw_sampled_token_ids,
        prev_sampled_token_ids=None,
        prev_num_draft_tokens=None,
        req_indices_per_token=None,
        is_warmup=False,
        model_override=None,
    ):
        """Generate every proposal token for the batch in one draft pass.

        Mirrors EagleProposer.propose in doing no tensor arithmetic of its own:
        inputs are routed to one device and handed straight to the draft NEFF.
        Everything that derives the proposal block lives in
        DFlashDraftModel.forward so it is traced into that NEFF rather than
        executed as eager XLA ops between the two executables.

        DFlash is a parallel drafter, so unlike EAGLE3 there is no recurrent
        loop here: the draft cost per step is one forward, not num_spec.
        """
        del target_token_ids, prev_sampled_token_ids, prev_num_draft_tokens
        del req_indices_per_token
        assert self.model is not None
        model = model_override if model_override is not None else self.model

        # Same device policy as EagleProposer: meta-device callers (parallel
        # trace children) must stay on meta, everything else is promoted to the
        # Neuron device the compiled graph expects.
        target_device = (
            torch.device("meta")
            if target_positions.device.type == "meta"
            else self.device
        )

        target_positions = target_positions.to(target_device)
        target_hidden_states = target_hidden_states.to(target_device)
        last_token_indices = last_token_indices.to(target_device)
        raw_sampled_token_ids = raw_sampled_token_ids.to(target_device)

        first_meta = attn_metadata[self.attn_layer_names[0]]
        context_slot_mapping = first_meta["slot_mapping"].to(target_device)
        batch_size = last_token_indices.shape[0]

        draft_metadata = {name: attn_metadata[name] for name in self.attn_layer_names}
        start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            drafts_only = model(
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                last_token_indices=last_token_indices,
                raw_sampled_token_ids=raw_sampled_token_ids,
                context_slot_mapping=context_slot_mapping,
                attn_metadata=draft_metadata,
                rank=self.rank_tensor.to(target_device),
            )
        elapsed = time.perf_counter() - start
        bucket_name = f"dflash_b{batch_size}_q{self.num_speculative_tokens + 1}"
        model_name = self.speculative_config.model
        if is_warmup:
            COMPILATION_TIME.labels(model_name=model_name, bucket_name=bucket_name).set(
                elapsed
            )
        else:
            NEFF_EXECUTION_COUNT.labels(
                model_name=model_name, bucket_name=bucket_name
            ).inc()

        # EagleProposer returns (bonus + drafts, drafts only). DFlash forbids
        # async scheduling, and only the drafts-only tensor is consumed in sync
        # mode, so the first slot is the same tensor rather than a bonus-padded
        # copy that nothing would read.
        return drafts_only, drafts_only
