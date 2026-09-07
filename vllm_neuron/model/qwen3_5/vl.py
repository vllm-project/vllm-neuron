# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 vision-language model: Qwen3.5's hybrid text decoder + Qwen3-VL's tower.

The vision half is **not** a second implementation. Upstream, HF's
``Qwen3_5VisionModel`` is literally ``Qwen3VLVisionModel`` with the deepstack
mergers deleted (see ``transformers/models/qwen3_5/modular_qwen3_5.py``), the
checkpoint's ``model.visual.*`` tensor names are identical, and every attribute
the plugin's Qwen3-VL encoder reads off a vision config exists on
``Qwen3_5VisionConfig``. So this reuses that encoder, its weight loading, its
block-packing/encoder-cache path and its mRoPE helper directly, rather than
copying two thousand lines that would then drift.

Reusing them as plain function references (``embed_multimodal = ...``) is
deliberate: their whole contract is ``self.visual``,
``self.config.vision_config`` and ``self._vision_captures``, which this class
provides. Subclassing the Qwen3-VL model instead would drag in its *text*
decoder, which is exactly the part Qwen3.5 does not share.

What Qwen3.5 changes:

* ``deepstack_visual_indexes`` is empty, so the encoder builds no deepstack
  mergers and the encoder-cache rows are exactly ``out_hidden_size`` wide
  instead of a "fat" concatenation. Both the encoder and the merge helper handle
  that case already; the text model raises if a deepstack tensor ever appears.
* The text decoder is the hybrid DeltaNet/attention stack in ``model.py``.
"""

from __future__ import annotations

import torch

from vllm_neuron.model.interfaces import SupportsVisionWarmup
from vllm_neuron.model.qwen3_vl.model_bf16 import (
    Qwen3VLForConditionalGeneration as _Qwen3VL,
)
from vllm_neuron.model.qwen3_vl.utils.mrope import compute_mrope_positions
from vllm_neuron.model.qwen3_vl.vision_encoder_bf16 import Qwen3VLVisionModel

from .config import Qwen3_5Config
from .model import Qwen3_5ForCausalLM


class Qwen3_5VLForConditionalGeneration(Qwen3_5ForCausalLM, SupportsVisionWarmup):
    """The text-only model plus a vision tower and the multimodal entry points."""

    def __init__(self, config: Qwen3_5Config):
        if config.vision_config is None:
            raise ValueError(
                "Qwen3_5VLForConditionalGeneration needs a vision_config; build "
                "it with Qwen3_5Config.from_configs(..., vision_neuron_config=...) "
                "or use Qwen3_5ForCausalLM for a text-only engine."
            )
        super().__init__(config)

        # Read by the reused ``embed_multimodal`` when tensor capture is enabled.
        self._vision_captures: tuple[torch.Tensor, ...] = ()
        self.visual = Qwen3VLVisionModel(
            config.vision_config, dtype=config.vision_config.torch_dtype
        )

    # ── Vision, reused wholesale from Qwen3-VL ───────────────────────────
    # Encode + allocate + scatter-write into the on-device encoder cache, and the
    # shape-only inputs warmup traces with. Identical pipelines; see the module
    # docstring for why sharing is sound.
    embed_multimodal = _Qwen3VL.embed_multimodal
    build_vision_synthetic_inputs = _Qwen3VL.build_vision_synthetic_inputs

    # ``get_vision_token_merge_factor`` / ``get_max_pixels_token_count`` live on
    # the *factory* in ``factory.py``, because ``vision_utils`` resolves them
    # through the registry and the registry holds the factory. They are not
    # duplicated here.

    # ── mRoPE with real vision grids ─────────────────────────────────────

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        """3D mRoPE positions, with spatial grid positions over vision spans.

        Overrides the text-only version, which rejects multimodal items. The
        helper only reads ``video_token_id``, ``vision_start_token_id``,
        ``vision_end_token_id`` and ``vision_config.spatial_merge_size``, all of
        which ``Qwen3_5Config`` provides.
        """
        return compute_mrope_positions(input_tokens, mm_features, self.config)

    # ── Construction and weights ─────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config,
        text_neuron_config=None,
        vision_neuron_config=None,
    ) -> "Qwen3_5VLForConditionalGeneration":
        return cls(
            Qwen3_5Config.from_configs(
                hf_config,
                text_neuron_config=text_neuron_config,
                vision_neuron_config=vision_neuron_config,
                include_vision=True,
            )
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Text weights through the shared path, then the tower's own loader.

        Order matters: the text load assigns a whole state dict, so the tower has
        to be loaded after it, not before.
        """
        super().load_weights(checkpoint_path, device, cache_dir)
        # The tower shards on the *vision* TP group, so it owns its own loading.
        self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)


__all__ = ["Qwen3_5VLForConditionalGeneration"]
