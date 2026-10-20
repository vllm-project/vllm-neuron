# SPDX-License-Identifier: Apache-2.0
"""
SyntheticNeuronModel — synthetic model for DI framework testing.

Replaces real neural network forward computation with deterministic
KV cache fill (prefill) and byte-exact verification (decode) to test
NIXL transfer correctness without requiring model weights or compilation.

See ``sharding.py`` for the fill/verify logic and sharding semantics.

WARNING: This is for testing only — not for production inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.parallel.neuron_parallel_state import shard_select

SYNTHETIC_MODEL_PATH = str(Path(__file__).parent / "synthetic_config")

TOKEN_SUCCESS = 5  # "KV_TRANSFER_VALID" — KV cache verified after NIXL transfer
TOKEN_FAIL = 6  # "KV_MISMATCH" — KV data mismatch detected
TOKEN_EMPTY = 7  # "KV_NO_BLOCKS" — no blocks to verify
TOKEN_AUTOREGRESSIVE = 8  # "AUTO_DECODE" — autoregressive decode step


@dataclass
class SyntheticConfig:
    """Model geometry for synthetic testing."""

    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    hidden_size: int = 4096
    num_attention_heads: int = 32
    vocab_size: int = 128256
    sliding_window: int | None = None
    swa_layers: list[int] | None = None  # e.g. [0, 2, 4]; None = even layers

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None) -> "SyntheticConfig":
        if hasattr(hf_config, "to_dict"):
            d = hf_config.to_dict()
        elif isinstance(d := hf_config, dict):
            pass
        else:
            d = {}
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in field_names})


class SyntheticNeuronModel(nn.Module):
    """Synthetic model implementing the 5 NeuronModelRunner interfaces.

    WARNING: Not for production inference. Gate behind VLLM_NEURON_SYNTHETIC_MODEL=1.
    """

    is_text_generation_model = True

    def __init__(self, config: SyntheticConfig, **kwargs):
        super().__init__()
        self.config = config
        self._kv_caches: dict[str, list[torch.Tensor]] = {}
        self._last_seq_len: int | None = None

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None) -> "SyntheticNeuronModel":
        return cls(SyntheticConfig.from_configs(hf_config, neuron_config))

    def load_weights(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        download_dir: str | None = None,
    ) -> None:
        print("[SyntheticKV] load_weights — no-op (synthetic model)", flush=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            input_ids.shape[0], self.config.hidden_size, device=input_ids.device
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden_states.shape[0], self.config.vocab_size, device=hidden_states.device
        )

    def get_kv_spec(self) -> KVSpec:
        try:
            world_size = torch.distributed.get_world_size()
        except (RuntimeError, ValueError):
            world_size = 1

        per_rank_kv_heads = max(
            1, self.config.num_key_value_heads // max(world_size, 1)
        )
        layers = []
        for i in range(self.config.num_hidden_layers):
            # Determine which layers use SWA: explicit list or default to even layers
            if self.config.swa_layers is not None:
                is_swa = i in self.config.swa_layers
            else:
                is_swa = i % 2 == 0
            swa = (
                self.config.sliding_window
                if (self.config.sliding_window and is_swa)
                else None
            )
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=per_rank_kv_heads,
                    head_size=self.config.head_dim,
                    dtype=torch.bfloat16,
                    sliding_window_size=swa,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        self._kv_caches = kv_caches
        print(f"[SyntheticKV] bind_kv_cache: {len(kv_caches)} layers", flush=True)
        for name, (k, v) in kv_caches.items():
            print(
                f"[SyntheticKV]   {name}: K={list(k.shape)} V={list(v.shape)} dtype={k.dtype} device={k.device}",
                flush=True,
            )

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_params: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        from vllm_neuron.model.synthetic.sharding import (
            get_kv_heads_for_rank,
        )

        # TODO: ODS detection is incorrect — sampling_params presence doesn't
        # reliably indicate ODS mode. Low harm for now.
        self._ods_mode = sampling_params is not None

        if len(self._kv_caches) == 0:
            raise RuntimeError("The KV cache is not bound. Call bind_kv_cache().")
        if attn_metadata is None or len(attn_metadata) == 0:
            raise RuntimeError(
                "The attention metadata is not available. Provide attn_metadata."
            )

        first_meta = next(iter(attn_metadata.values()))
        # Prefill detection: max_query_len > 1 means multiple tokens in the query.
        # TODO: breaks with speculative decoding where decode sends multiple tokens.
        # Proper fix: scheduler passes is_prefill flag in attn_metadata.
        is_prefill = first_meta.get("max_query_len", 0) > 1

        tp_rank, tp_size = self._get_tp_info(rank)
        head_start, num_heads = get_kv_heads_for_rank(
            tp_rank,
            tp_size,
            self.config.num_key_value_heads,
        )

        if is_prefill:
            from vllm_neuron.model.synthetic.sharding import prefill_fill_kv

            prefill_fill_kv(
                self._kv_caches, positions, attn_metadata, head_start, num_heads
            )
            return self._make_logits(input_ids, TOKEN_EMPTY)

        token_id = self._handle_decode(attn_metadata, head_start, num_heads)
        return self._make_logits(input_ids, token_id)

    # ── Helpers ──────────────────────────────────────────────────

    def _dcp_slice_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Apply DCP interleave slice to positions (mirrors slot_mapping slice in model runner)."""
        from vllm.distributed.parallel_state import (
            get_dcp_group,
        )

        try:
            dcp_group = get_dcp_group()
        except (AssertionError, RuntimeError):
            return positions

        W = dcp_group.world_size
        if W <= 1:
            return positions

        S = positions.shape[0]
        # I = interleave_size = block_size (enforced by DI+DCP constraint)
        # k_cache shape: [num_blocks, num_heads, block_size, head_dim]
        I = next(iter(self._kv_caches.values()))[0].shape[2]

        assert S % (W * I) == 0, (
            f"DCP positions slice requires S={S} divisible by W*I={W}*{I}={W * I}"
        )

        positions_by_rank = positions.view(S // (W * I), W, I)
        return shard_select(positions_by_rank, 1, dcp_group).contiguous().reshape(-1)

    def _get_tp_info(
        self, rank: torch.Tensor | None = None
    ) -> tuple[int | torch.Tensor, int]:
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_world_size,
            )

            tp_size = get_tensor_model_parallel_world_size()

            if rank is not None:
                return get_tp_group().rank_in_group_tensor.to(rank.dtype), tp_size

            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank(), tp_size
        except (RuntimeError, ValueError, ImportError):
            return 0, 1

    def _make_logits(self, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        # ODS mode: return token IDs directly
        if self._ods_mode:
            return torch.full((batch_size,), token_id, dtype=torch.int64, device=device)
        logits = torch.zeros(batch_size, self.config.vocab_size, device=device)
        logits[:, token_id] = 100.0
        return logits

    def _handle_decode(
        self, attn_metadata: dict, head_start: int, num_heads: int
    ) -> int:
        first_meta = next(iter(attn_metadata.values()))
        cached_seq_len = first_meta.get("cached_seq_len")
        num_computed = cached_seq_len.item() if cached_seq_len is not None else 0

        # Detect first decode: when seq_len drops, a new request started.
        # Limitation: breaks with speculative decoding (seq_len jumps by >1).
        # Proper fix requires scheduler to pass is_first_decode flag in attn_metadata.
        if self._last_seq_len is not None and num_computed >= self._last_seq_len:
            self._last_seq_len = num_computed
            return TOKEN_AUTOREGRESSIVE

        self._last_seq_len = num_computed
        return self._verify_kv(attn_metadata, num_computed, head_start, num_heads)

    def _verify_kv(
        self, attn_metadata: dict, num_computed: int, head_start: int, num_heads: int
    ) -> int:
        """Verify KV cache contents match the deterministic fill after NIXL transfer.

        Called on the first decode step for each new request. Uses
        raw_block_table_tensor (the scheduler's original allocation, before any
        neuron-specific transforms) to walk each layer's blocks and compare
        against the deterministic fill formula.

        For SWA layers, only verifies blocks that contain actual transferred
        data by probing each block's first K cache slot (our fill always writes
        values in [1,255], so zero means untransferred).

        Returns TOKEN_SUCCESS if all blocks match, TOKEN_FAIL if any mismatch.
        """
        from vllm_neuron.model.synthetic.sharding import (
            build_block_positions,
            decode_verify_kv_layer,
        )

        errors: list[str] = []
        any_blocks = False

        # Build a map of layer_idx -> sliding_window_size for SWA checks
        kv_spec = self.get_kv_spec()
        layer_swa_map = {
            int(layer.name.split(".")[1]): layer.sliding_window_size
            for layer in kv_spec.layers
        }

        for layer_name, (k_cache, v_cache) in self._kv_caches.items():
            meta = attn_metadata.get(layer_name)
            if meta is None:
                continue
            block_size = meta["block_size"]

            # Use the raw (pre-transform) block table from the scheduler.
            # This avoids any neuron-specific SWA trimming/P_MAX padding.
            raw_bt = meta.get("raw_block_table_tensor")
            if raw_bt is None:
                raw_bt = meta["block_table_tensor"]
            if raw_bt is None or raw_bt.numel() == 0:
                continue

            layer_idx = int(layer_name.split(".")[1])
            sliding_window = layer_swa_map.get(layer_idx)

            # Walk the raw block table by index. The index IS the block ordinal
            # (ordinal N holds gpos [N*bs .. (N+1)*bs)). Probe each block to
            # find which ones have transferred data.
            raw_ids = raw_bt[0].cpu().tolist()
            verified_block_ids = []
            first_valid_index = None
            for idx, bid in enumerate(raw_ids):
                if bid == 0:
                    continue  # null block or unallocated
                # Packed FP8 K (5-D) leaves a trailing size-2 pack axis; probe
                # the first sub-slot. Our fill writes [1,255], so zero means
                # untransferred.
                probe = k_cache[bid, 0, 0, 0]
                if probe.numel() > 1:
                    probe = probe[0]
                if probe.item() != 0.0:
                    verified_block_ids.append(bid)
                    if first_valid_index is None:
                        first_valid_index = idx

            if not verified_block_ids:
                continue
            any_blocks = True

            # seq_start = ordinal of first verified block * block_size
            seq_start = first_valid_index * block_size

            # Sanity check: verify the expected number of blocks were transferred.
            # The window may not start at a block boundary, so we need +1 block
            # to cover the partial start block (matches upstream blocks_per_sw).
            if sliding_window is not None and num_computed > sliding_window:
                expected_blocks = sliding_window // block_size + 1
                assert len(verified_block_ids) == expected_blocks, (
                    f"{layer_name}: expected {expected_blocks} SWA blocks "
                    f"(sw={sliding_window}, bs={block_size}), "
                    f"got {len(verified_block_ids)}"
                )

            print(
                f"[SyntheticKV] verify {layer_name}: {len(verified_block_ids)} blocks, "
                f"block_size={block_size}, num_computed={num_computed}, "
                f"seq_start={seq_start}"
                f"{f', swa={sliding_window}' if sliding_window else ''}",
                flush=True,
            )

            global_positions = build_block_positions(
                num_blocks=len(verified_block_ids),
                block_size=block_size,
                seq_len=num_computed,
                seq_offset=seq_start,
            )

            errors.extend(
                decode_verify_kv_layer(
                    k_cache,
                    v_cache,
                    layer_idx,
                    verified_block_ids,
                    num_computed,
                    block_size,
                    global_positions,
                    head_start,
                    num_heads,
                )
            )

        # Step 7: Report results
        if errors:
            for e in errors[:5]:
                print(f"[SyntheticKV] FAIL: {e}", flush=True)
            if len(errors) > 5:
                print(
                    f"[SyntheticKV] FAIL: ... and {len(errors) - 5} more mismatches",
                    flush=True,
                )
            return TOKEN_FAIL
        if any_blocks:
            print(
                f"[SyntheticKV] OK: {num_computed} computed tokens, {len(self._kv_caches)} layers (heads {head_start}-{head_start + num_heads - 1})",
                flush=True,
            )
            return TOKEN_SUCCESS
        assert False, (
            f"No blocks found in any layer's block_table (num_computed={num_computed}). "
            "This should not happen — decode should always have transferred blocks."
        )
