# SPDX-License-Identifier: Apache-2.0
"""Tensor replacement: inject HF reference tensors into a model's forward pass.

Replaces specific intermediate values (e.g., MoE router logits) with
HF reference tensors at runtime, driven by the scheduler's metadata
on each forward pass.

Usage::

    from vllm_neuron.accuracy.tensor_replacement import (
        TensorReplacer, set_active_context, get_replacement_tensor,
    )

    # At init: create replacer from HF captures
    replacer = TensorReplacer(reference_captures, prompt_token_ids=prompt_token_ids)

    # Warmup: zero tensors with correct shape for torch.compile tracing
    set_active_context(replacer.warmup_context(num_tokens=256, device=device))
    model(**warmup_kwargs)
    set_active_context(None)

    # Inference: register new request, then build context
    replacer.register_request(req_id, prompt_token_ids)
    set_active_context(replacer.build_context(
        req_ids=["0-abc123"], positions=[0,1,...,1023],
        is_prefill=True, device=device,
    ))
    model(**model_kwargs)
    set_active_context(None)

    # In model code: read from global context
    replacement = get_replacement_tensor("model.layers.0.mlp.router")
    if replacement is not None:
        # shard if needed (e.g. SP slice), then use as router logits override
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _hash_prompt(token_ids: List[int]) -> int:
    """Hash a prompt's token IDs for prefix matching."""
    return hash(tuple(token_ids))


def _flatten_hf_steps(
    hf_tensors: List[Dict[str, List[torch.Tensor]]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Flatten step-wise HF captures into position-indexed tensors.

    Converts {prompt_idx: {module: [step0, step1, ...]}} into
    {prompt_idx: {module: Tensor[total_positions, feature_dim]}}.
    Step 0 is prefill (multi-token), steps 1+ are single-token decodes.
    """
    flat: Dict[int, Dict[str, torch.Tensor]] = {}
    for prompt_idx, modules in enumerate(hf_tensors):
        flat[prompt_idx] = {}
        for module_name, steps in modules.items():
            parts = []
            for i, s in enumerate(steps):
                assert s.dim() == 3 and s.shape[0] == 1, (
                    f"Expected [1, seq_len, E], got {s.shape} at step {i} "
                    f"of {module_name}"
                )
                t = s[0]  # [seq_len, E]
                if i == 0:
                    parts.append(t)
                else:
                    parts.append(t[-1:])
            flat[prompt_idx][module_name] = torch.cat(parts, dim=0)
    return flat


class TensorReplacer:
    """Builds replacement tensors per forward pass from HF reference captures.

    At init, flattens step-wise HF tensors into position-indexed tensors.
    At each forward pass, build_context() indexes into these flat tensors
    using the scheduler's positions and req_ids to produce
    [bucket_size, feature_dim] replacement tensors.

    Args:
        reference_captures: Per-prompt reference tensors in step-wise format:
            List[Dict[module_name, List[step_tensor]]]
            where index = prompt_idx, step 0 = prefill, steps 1+ = decode.
    """

    def __init__(self, reference_captures, prompt_token_ids=None):
        if isinstance(reference_captures, list):
            self._captures = _flatten_hf_steps(reference_captures)
        else:
            self._captures = reference_captures

        if self._captures:
            any_prompt = next(iter(self._captures.values()))
            self._module_names = list(any_prompt.keys())
        else:
            self._module_names = []

        # Prompt hash → prompt_idx map for request matching.
        # Example: prompt_token_ids=[[10,20,30], [40,50]] builds
        # {hash([10,20,30]): 0, hash([40,50]): 1}. When a retry
        # arrives with [10,20,30,77,88], hash(tokens[:3]) matches prompt 0.
        self._prompt_hash_to_idx: Dict[int, int] = {}
        self._prompt_lengths: List[int] = []
        self._prompt_tokens: Dict[int, List[int]] = {}
        if prompt_token_ids:
            for idx, ids in enumerate(prompt_token_ids):
                # Scheduler provides flat [10,20,30] per request.
                # tokenizer([prompt]) returns shape [1, seq_len], so
                # .tolist() gives [[10,20,30]] with a batch dim. Handle both.
                tokens = (
                    ids[0]
                    if isinstance(ids, list) and ids and isinstance(ids[0], list)
                    else ids
                )
                self._prompt_hash_to_idx[_hash_prompt(tokens)] = idx
                self._prompt_lengths.append(len(tokens))
                self._prompt_tokens[idx] = tokens

        # req_id → prompt_idx map, populated by register_request()
        self._req_id_map: Dict[str, int] = {}

        logger.info(
            "TensorReplacer initialized: %d prompts, %d modules",
            len(self._captures),
            len(self._module_names),
        )

    def register_request(self, req_id: str, prompt_token_ids: List[int]) -> None:
        """Map a request ID to its prompt index by matching longest token prefix."""
        if req_id in self._req_id_map:
            return
        if not self._prompt_hash_to_idx:
            raise ValueError(
                "Cannot register request: no prompt_token_ids were provided "
                "to TensorReplacer at init."
            )
        best_match = None
        best_len = -1
        for plen in self._prompt_lengths:
            if plen > len(prompt_token_ids):
                continue
            prefix = prompt_token_ids[:plen]
            h = _hash_prompt(prefix)
            if h in self._prompt_hash_to_idx:
                idx = self._prompt_hash_to_idx[h]
                if prefix == self._prompt_tokens[idx] and plen > best_len:
                    best_match = idx
                    best_len = plen
        if best_match is not None:
            self._req_id_map[req_id] = best_match
            return
        raise KeyError(
            f"No matching prompt for req_id={req_id!r} "
            f"(tried {len(self._prompt_lengths)} prefix lengths)"
        )

    def _resolve_prompt_idx(self, req_id: str) -> int:
        """Look up prompt index for a request ID."""
        if req_id not in self._req_id_map:
            raise KeyError(
                f"Request {req_id!r} not registered. Call register_request() first."
            )
        return self._req_id_map[req_id]

    def build_context(
        self,
        req_ids: List[str],
        positions: List[int],
        is_prefill: bool,
        device: torch.device = torch.device("cpu"),
        tokens_per_req: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Build replacement tensors for one forward pass.

        For prefill: iterates positions in order, indexes into the flat
        HF tensor at each position. First duplicate position marks
        padding start — remaining slots stay zero.

        For decode: iterates req_ids, extracts prompt index from each
        req_id, indexes that prompt's flat HF tensor at the request's
        position. Slots beyond real requests stay zero.

        Args:
            req_ids: Request IDs from the scheduler.
            positions: Token positions from the scheduler (includes padding).
            is_prefill: Whether this is a prefill or decode forward pass.
            device: Device to place tensors on.

        Returns:
            {module_name: Tensor[bucket_size, feature_dim]} for
            set_active_context().
        """
        if not self._captures or not req_ids:
            return {}

        ctx: Dict[str, torch.Tensor] = {}
        bucket_size = len(positions)

        for module_name in self._module_names:
            any_prompt = next(iter(self._captures.values()))
            ref = any_prompt[module_name]
            feature_dim = ref.shape[-1]
            dtype = ref.dtype
            full_tensor = torch.zeros(bucket_size, feature_dim, dtype=dtype)

            if is_prefill:
                req_id = req_ids[0]
                prompt_idx = self._resolve_prompt_idx(req_id)
                if prompt_idx not in self._captures:
                    raise KeyError(
                        f"No HF captures for prompt_idx={prompt_idx} "
                        f"(req_id={req_id!r})"
                    )
                hf_tensor = self._captures[prompt_idx][module_name]
                # Real positions are strictly increasing; first duplicate marks padding.
                seen = set()
                for slot_idx, pos in enumerate(positions):
                    if pos in seen:
                        break
                    seen.add(pos)
                    if pos < hf_tensor.shape[0]:
                        full_tensor[slot_idx] = hf_tensor[pos]
            else:
                # Decode path: positions may have multiple tokens per request
                # (e.g., 1 + num_spec_tokens in speculative decoding).
                # Map each slot to its request via tokens_per_req.
                for slot_idx, pos in enumerate(positions):
                    req_idx = slot_idx // tokens_per_req
                    if req_idx >= len(req_ids):
                        break  # padding slots
                    req_id = req_ids[req_idx]
                    prompt_idx = self._resolve_prompt_idx(req_id)
                    if prompt_idx not in self._captures:
                        raise KeyError(
                            f"No HF captures for prompt_idx={prompt_idx} "
                            f"(req_id={req_id!r})"
                        )
                    hf_tensor = self._captures[prompt_idx][module_name]
                    if pos < hf_tensor.shape[0]:
                        full_tensor[slot_idx] = hf_tensor[pos]

            ctx[module_name] = full_tensor.to(device)

        return ctx

    def warmup_context(
        self,
        num_tokens: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Create zero tensors matching the feature shape of HF captures.

        Args:
            num_tokens: Token count for this warmup bucket.
            device: Device to place tensors on.

        Returns:
            {module_name: Tensor[num_tokens, feature_dim]} of zeros.
        """
        ctx: Dict[str, torch.Tensor] = {}
        if not self._captures:
            return ctx

        any_prompt = next(iter(self._captures.values()))
        for module_name, tensor in any_prompt.items():
            feature_shape = tensor.shape[1:]
            ctx[module_name] = torch.zeros(
                (num_tokens,) + feature_shape,
                dtype=tensor.dtype,
                device=device,
            )
        return ctx


_active_context: Optional[Dict[str, torch.Tensor]] = None


def set_active_context(ctx: Optional[Dict[str, torch.Tensor]]) -> None:
    """Set the active replacement context for the current forward pass."""
    global _active_context
    _active_context = ctx


def get_replacement_tensor(tensor_name: str) -> Optional[torch.Tensor]:
    """Retrieve a replacement tensor from the active context — no-op when replacement is not configured."""
    if _active_context is None:
        return None
    return _active_context.get(tensor_name)
