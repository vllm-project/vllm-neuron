# SPDX-License-Identifier: Apache-2.0
"""Shared, quantization-agnostic Qwen3-VL orchestration.

The BF16 (``model_bf16``) and native-MXFP8 (``model_mxfp8``) Qwen3-VL models
differ only in their per-projection compute graph (attention / MLP / decoder
layer) and in ``load_weights``. Everything else — the top-level multimodal
orchestration (LM-head + sampler wiring, vision encoding into the on-device
cache, KV-spec/bind, M-RoPE position computation, synthetic vision warmup
inputs), the text-backbone ``forward``, and the shared ``__init__`` wiring —
is identical and lives here as two mixins:

  - ``Qwen3VLGenerationMixin`` -> the top-level ``Qwen3VLForConditionalGeneration``
  - ``Qwen3VLTextBackboneForwardMixin`` -> the text backbone ``Qwen3VLTextModel``

Each mixin provides the shared ``forward`` plus an ``_init_*_common`` helper that
builds the shared attributes (``self.language_model``, ``self.lm_head``,
``self.visual``, ``self.tp_group``, ...); each model's ``__init__`` is a thin seam
that calls the helper with its own precision-specific class. The one field that
differs per quantization (the text-model / decoder-layer class) is passed in, so
the attribute contract has a single source of truth. Neither mixin defines
``__init__`` itself, so ``super().__init__()`` in the concrete class still
resolves to ``nn.Module.__init__``; each mixin must precede ``nn.Module`` in the
MRO so its ``forward`` overrides ``nn.Module``'s stub. A ``_check_*_attrs`` hook
run at the end of each ``_init_*_common`` fails loudly at construction if a
required attribute is missing, rather than an opaque ``AttributeError`` later in
``forward``.
"""

from __future__ import annotations

import logging
import math
import time as _time

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.utils import record_function_or_nullcontext
from transformers import PretrainedConfig

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
    EncoderNeuronConfig,
)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import Qwen3VLConfig, Qwen3VLTextConfig
from .vision_encoder_bf16 import Qwen3VLVisionModel
from vllm_neuron.functional.merge_multimodal_embeddings import (
    merge_multimodal_embeddings,
)
from .utils.mrope import compute_mrope_positions
from .utils.vision_block_packing import (
    compute_block_bounds,
    ffd_pack_images,
    scatter_to_blocks,
    select_vision_bucket,
)
from .utils.vision_preprocessing import (
    compute_position_indices_and_weights,
    compute_rotary_pos_emb,
)

logger = logging.getLogger(__name__)


class Qwen3VLTextBackboneForwardMixin:
    """Quantization-agnostic ``forward`` + shared init for the Qwen3-VL text backbone.

    ``forward`` reads ``self.embed_tokens``, ``self.rotary_emb``, ``self.layers``,
    ``self.norm``, ``self.tp_group`` and ``self.rank``. Rather than trust each
    concrete ``__init__`` to build all of these consistently, the concrete
    ``Qwen3VLTextModel.__init__`` in both models is a thin seam that calls
    ``_init_backbone_common`` (below), passing its own decoder-layer class. This
    keeps the attribute contract in one place — a rename here changes both models
    at once — while leaving room to move a field back out if a future quantized
    backbone needs different sharding. Must precede ``nn.Module`` in the MRO so
    this ``forward`` overrides the stub.
    """

    # Attributes this mixin's forward() reads; verified by _check_backbone_attrs.
    _REQUIRED_BACKBONE_ATTRS = (
        "embed_tokens",
        "rotary_emb",
        "layers",
        "norm",
        "tp_group",
        "rank",
    )

    def _init_backbone_common(
        self, config: Qwen3VLTextConfig, decoder_layer_cls: type
    ) -> None:
        """Build the shared backbone attributes (embedding, decoder-layer stack,
        final norm, RoPE, TP group). ``decoder_layer_cls`` is the concrete
        (BF16 or MX) ``Qwen3VLTextDecoderLayer`` — the one field that differs per
        quantization — so the surrounding wiring stays identical across models."""
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                decoder_layer_cls(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # RMSNorm / RoPE are shared leaves (both models import them from model_bf16).
        from .model_bf16 import Qwen3VLTextRMSNorm, Qwen3VLTextRotaryEmbedding

        self.norm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )
        self._check_backbone_attrs()

    def _check_backbone_attrs(self) -> None:
        """Fail loudly at construction if a required forward() attribute is
        missing, instead of an opaque AttributeError deep in forward()."""
        missing = [a for a in self._REQUIRED_BACKBONE_ATTRS if not hasattr(self, a)]
        if missing:
            raise AttributeError(
                f"{type(self).__name__} is missing attributes required by "
                f"Qwen3VLTextBackboneForwardMixin.forward: {missing}. "
                "Its __init__ must call _init_backbone_common (or set these)."
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        # <-- On-device encoder cache inputs (prefill-only)
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional vision inputs.

        Args:
            input_ids: Token IDs.
            positions: Sequential position IDs for attention mask and KV cache.
            attn_metadata: Attention metadata dict.
            rank: TP rank tensor.
            rotary_position_ids: 3D M-RoPE positions [3, T] for RoPE computation.
            encoder_embedding_buffer: On-device encoder cache buffer, shape
                ``[num_blocks, block_size, fat_dim]`` where
                fat_dim = out_hidden_size * (1 + num_deepstack_levels). The graph
                gathers the needed blocks via ``index_select`` for scatter.
            encoder_block_ids: ``[max_num_encoder_blocks]`` int64 block indices to
                gather from the buffer (padded with the scratch block id).
            encoder_positions: Position mapping for on-device cache path.
                Shape ``[max_num_encoder_blocks, block_size]`` int64.
                ``encoder_positions[i, j]`` = batch position for token j in
                block i. Sentinel value = num_tokens (writes to +1 dummy row).
        """
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # Vision embedding merge (prefill only, decode has no vision inputs).
        deepstack_vision_embeds = None
        if (
            is_prefill
            and encoder_embedding_buffer is not None
            and encoder_block_ids is not None
            and encoder_positions is not None
        ):
            hidden_states, deepstack_vision_embeds = merge_multimodal_embeddings(
                hidden_states,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                rank=self.tp_group.rank_in_group_tensor,
            )

        position_embeddings = self.rotary_emb(
            rotary_position_ids, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

            if (
                deepstack_vision_embeds is not None
                and layer_idx < deepstack_vision_embeds.shape[0]
            ):
                hidden_states = hidden_states + deepstack_vision_embeds[layer_idx]

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather to reconstruct full sequence <<<
        if is_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


class Qwen3VLGenerationMixin:
    """Quantization-agnostic orchestration for ``Qwen3VLForConditionalGeneration``.

    Provides the top-level ``forward`` (LM head + sampler), vision encoding
    (``embed_multimodal`` / ``build_encoder_synthetic_inputs``), M-RoPE position
    computation, KV-cache spec/bind, and the ``from_configs`` builder. The concrete
    ``Qwen3VLForConditionalGeneration.__init__`` in both models is a thin seam that
    calls ``_init_generation_common`` (below), passing its own text-model class, so
    the attribute contract lives in one place. ``load_weights`` is intentionally NOT
    here — it is the one orchestration method that differs per quantization. Must
    precede ``nn.Module`` in the MRO so this ``forward`` overrides the stub.
    """

    # Attributes this mixin's methods read; verified by _check_generation_attrs.
    _REQUIRED_GENERATION_ATTRS = (
        "config",
        "text_config",
        "language_model",
        "lm_head",
        "sampler",
        "visual",
        "tp_group",
        "world_size",
        "rank",
        "on_device_sampling_config",
        "_gather_logits",
        "mm_encoder_only",
        "mm_language_model_only",
    )

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Give each graph its own neuronx-cc compiler arguments.

        This is the model's per-graph policy. The runner resolves it eagerly,
        one time per graph, before the cache hash: it passes the graph type, the
        example inputs, and the base argument list, and carries the returned
        list to the compiler (on the trace job for the parallel path, through a
        libtorch scope for the in-process path). Return the full argument list
        for that graph.

        ``base`` holds the arguments at the caller level. Derive the per-graph
        arguments from it. These arguments are a default, not an override: an
        explicit ``--optimization-level`` from the caller wins, so the runner
        does not call this policy and every graph keeps the caller level.

        Qwen3-VL compiles the plain decode graph and the encoder graph at
        ``-O3``. ``-O3`` disables neuronx-cc modular flow. The ``-O1`` default
        enables it, which adds a ``PSEUDO_FUNCTION_RETURN`` /
        ``EVENT_SEMAPHORE`` wrapper on each collective. ``-O3`` improves the
        latency of these graphs. Prefill keeps the base level.
        """
        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        if graph_type in ("decode", "encoder"):
            return with_optimization_level(base, 3)
        # TODO(compiler-args): to tune the jump-forward graph, resolve its
        # warmup and execute sites with graph_type "jf" in neuron_model_runner,
        # then add an `elif graph_type == "jf":` branch here.
        # TODO(compiler-args): to tune the spec-verify (draft-length target)
        # graph, resolve its sites with graph_type "spec_verify", then add an
        # `elif graph_type == "spec_verify":` branch here.
        return base

    def _scoped_compiler_args(self, graph_type):
        """Scope this graph's resolved compiler arguments for an in-process compile.

        Read the base arguments the runner stamped on the model, delegate to
        :meth:`compiler_args_for`, and open a libtorch scope with the result.
        Open a null scope (no override) when the runner did not enable per-graph
        tuning, so the base arguments apply.
        """
        from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

        base = getattr(self, "_neuron_base_compiler_args", None)
        args = None if base is None else self.compiler_args_for(graph_type, None, base)
        return scoped_compiler_args(args)

    def _init_generation_common(
        self, config: Qwen3VLConfig, text_model_cls: type
    ) -> None:
        """Build the shared top-level attributes (vision tower, text backbone, LM
        head, sampler, TP group, EPD role flags). ``text_model_cls`` is the concrete
        (BF16 or MX) ``Qwen3VLTextModel`` — the one field that differs per
        quantization — so the surrounding wiring stays identical across models."""
        # EPD construction-role flags, read off the vision NeuronConfig (both
        # False -> monolith: build both towers). The runner sets them and
        # validates mutual exclusion.
        encoder_neuron_config = (
            config.vision_config.neuron_config if config.vision_config else None
        )
        self.mm_encoder_only = bool(
            encoder_neuron_config and encoder_neuron_config.mm_encoder_only
        )
        self.mm_language_model_only = bool(
            encoder_neuron_config and encoder_neuron_config.mm_language_model_only
        )
        self.config = config
        self.text_config = config.text_config

        # Vision encoder (skipped on a language-only PD pool)
        if not self.mm_language_model_only:
            self.visual = Qwen3VLVisionModel(config.vision_config, dtype=torch.bfloat16)
        else:
            self.visual = None

        if not self.mm_encoder_only:
            self.language_model = text_model_cls(config.text_config)
        else:
            self.language_model = None

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (
            nc is not None and nc.max_logprobs != 0
        ) or debug_logits_enabled

        if not self.mm_encoder_only:
            # >>> PARALLELISM: Column-parallel LM head <<<
            self.lm_head = neuron_nn.ColumnParallelLinear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
                dtype=config.text_config.torch_dtype,
                gather_output=not self.on_device_sampling_config,
                tp_group=self.tp_group.device_group,
                weight_layout="in_out",
            )

            # >>> PARALLELISM: Shard lm_head on the vocab (output) dim <<<
            transposed = self.lm_head.weight_layout == "in_out"
            set_weight_loader(
                self.lm_head.weight,
                sharding_weight_loader_with_padding(
                    shard_dim=1 if transposed else 0,
                    shard_size=self.text_config.vocab_size // self.world_size,
                    num_shards=self.world_size,
                    is_storage_transposed=transposed,
                    pad_dim=0 if transposed else 1,
                    padded_size=self.text_config.hidden_size,
                    unpadded_size=self.text_config.hidden_size,
                ),
            )

            if self.on_device_sampling_config is not None:
                self.sampler = Sampler(
                    self.on_device_sampling_config,
                    process_group=self.tp_group.device_group,
                )
            else:
                self.sampler = None
        else:
            self.lm_head = None
            self.sampler = None

        self._check_generation_attrs()

    def _check_generation_attrs(self) -> None:
        """Fail loudly at construction if a required attribute is missing, instead
        of an opaque AttributeError deep in forward()/embed_multimodal()."""
        missing = [a for a in self._REQUIRED_GENERATION_ATTRS if not hasattr(self, a)]
        if missing:
            raise AttributeError(
                f"{type(self).__name__} is missing attributes required by "
                f"Qwen3VLGenerationMixin: {missing}. Its __init__ must call "
                "_init_generation_common (or set these)."
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            rotary_position_ids=rotary_position_ids,
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Compute 3D M-RoPE positions for the given prompt.

        Called by the model runner during request initialization.

        Returns:
            (positions, delta) where positions is [3, seq_len] int64 tensor
            and delta is the offset for computing decode-phase positions.
        """
        return compute_mrope_positions(input_tokens, mm_features, self.config)

    # ── Vision Encoder ──────────────────────────────────────────────────

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Encode images or videos and allocate+write into the on-device cache.

        Allocates cache blocks for each item based on its merged token count,
        then has the VE NEFF scatter-write directly into the cache buffer.
        No unpack step, no device→host transfer, no per-item split.

        Video reuses the image vision pipeline unchanged; only the kwarg names
        differ (pixel_values_videos / video_grid_thw). The runner supplies one
        modality pair per call, so the video pair is folded onto the image path.

        Args:
            pixel_values: [total_tokens, patch_dim] — flat across all images.
            image_grid_thw: [num_images, 3] — (T, H, W) per image.
            encoder_cache: EncoderCacheBlocks instance (provides buffer,
                allocate, scratch_block_id).
            mm_hashes: Per-item mm_hash identifiers, same order as items in the
                grid. Used for cache allocation.
            pixel_values_videos / video_grid_thw: video equivalents (one grid
                row per video, T frames).
        """
        from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

        # The runner groups multimodal kwargs by modality, so exactly one pair
        # is supplied per call. Enforce it: silently dropping one modality would
        # only surface as wrong output far downstream.
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError(
                "embed_multimodal: cannot supply both pixel_values and "
                "pixel_values_videos in a single call; caller must group by "
                "modality."
            )
        # Video reuses the image path; fold it onto the image kwargs.
        is_video = pixel_values is None and pixel_values_videos is not None
        if is_video:
            pixel_values = pixel_values_videos
            image_grid_thw = video_grid_thw
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "embed_multimodal requires (pixel_values, image_grid_thw) or "
                "(pixel_values_videos, video_grid_thw)."
            )

        cache: EncoderCacheBlocks = encoder_cache
        vc = self.config.vision_config
        encoder_neuron_config = vc.neuron_config

        t_start = _time.perf_counter()

        head_dim = vc.hidden_size // vc.num_heads
        num_grid_per_side = int(vc.num_position_embeddings**0.5)
        merge_factor = vc.spatial_merge_size**2
        configured_block_size = encoder_neuron_config.encoder_attention_block_size
        block_size = configured_block_size
        grid_rows = image_grid_thw.tolist()

        # A video is packed per FRAME, not as one T*H*W item: each [T,H,W] grid
        # row expands to T rows of [1,H,W] so its frames distribute across blocks
        # (and DP ranks) instead of forcing block_size >= the whole video. Rules:
        #   - A frame must fit within one block (per_frame H*W <= block_size) so
        #     its block-local attention stays complete: a frame is never split
        #     across blocks. This is the only floor; frames need NOT tile the
        #     block evenly. (block_size % 128 == 0 is handled separately inside
        #     the vision encoder, which pads the attention sequence.)
        #   - A block holds WHOLE frames plus a trailing pad on non-last blocks;
        #     the cache records each block's real (non-pad) merged token count
        #     (tokens_per_block) so the read path skips the pad. This is what
        #     lets block_size not be a multiple of per_frame.
        #   - group_ids keeps each video's frames in its own block run, so a
        #     block never mixes frames from two videos.
        # Images are unchanged: one grid row per item, each its own group.
        if is_video:
            frames_per_item = image_grid_thw[:, 0]
            grid_for_ve = image_grid_thw.repeat_interleave(frames_per_item, dim=0)
            grid_for_ve[:, 0] = 1
            group_ids = torch.repeat_interleave(
                torch.arange(image_grid_thw.shape[0]), frames_per_item
            ).tolist()
            # Enforce the frame-fits-in-block floor across ALL items before step 1
            # allocates any block (an oversized frame must not leave orphan blocks).
            for t, h, w in grid_rows:
                if h * w > block_size:
                    raise ValueError(
                        f"encoder_attention_block_size={block_size} is smaller "
                        f"than the per-frame token count {h * w} (H*W); a frame "
                        "must fit within one block so its attention is complete. "
                        "Increase the bucket/block size."
                    )
        else:
            grid_for_ve = image_grid_thw
            group_ids = None
            # Enforce the per-item-fits-in-block floor for images too, before
            # allocate() to avoid orphaned blocks.
            for t, h, w in grid_rows:
                if t * h * w > block_size:
                    raise ValueError(
                        f"encoder_attention_block_size={block_size} is smaller "
                        f"than the per-image token count {t * h * w} (T*H*W); an "
                        "image must fit within one block so its attention is complete. "
                        "Increase the bucket/block size."
                    )

        tokens_per_image = grid_for_ve.prod(dim=1).tolist()
        total_tokens = sum(tokens_per_image)

        # 1. Allocate cache blocks per item (one mm_hash each). Images use
        #    allocate's dense default (one run, only the last block partial).
        #    A video packs WHOLE frames per block (see above), so it passes an
        #    explicit tokens_per_block: block j holds min(frames_per_block,
        #    frames_left) frames * per_frame_merged real tokens, then pad.
        cache_block_map: list[list[int]] = []
        for i, (t, h, w) in enumerate(grid_rows):
            raw_tokens = t * h * w
            num_merged = raw_tokens // merge_factor
            if is_video:
                per_frame_raw = h * w
                per_frame_merged = per_frame_raw // merge_factor
                frames_per_block = block_size // per_frame_raw
                num_item_blocks = math.ceil(t / frames_per_block)
                tokens_per_block: list[int] = []
                frames_left = t
                for _ in range(num_item_blocks):
                    k = min(frames_per_block, frames_left)
                    tokens_per_block.append(k * per_frame_merged)
                    frames_left -= k
                block_ids = cache.allocate(mm_hashes[i], tokens_per_block)
            else:
                block_ids = cache.allocate(
                    mm_hashes[i], cache.dense_tokens_per_block(num_merged, block_size)
                )
            cache_block_map.append(block_ids)

        # 2. Bucket selection (CPU). Whole-frame packing can need more blocks than
        #    ceil(total_tokens/block_size), and step 1 already derived the exact
        #    per-video block count, so size the bucket from that to guarantee the
        #    chosen (warmed) bucket and the packer both have enough blocks.
        if group_ids is not None:
            required_blocks = sum(len(blocks) for blocks in cache_block_map)
            bucket_tokens = max(total_tokens, required_blocks * block_size)
        else:
            bucket_tokens = total_tokens
        _bucket, num_blocks = select_vision_bucket(
            bucket_tokens,
            encoder_neuron_config.num_encoder_input_buckets,
            configured_block_size,
            dp_size=encoder_neuron_config.dp_size,
        )

        # 3. FFD packing with one-item-per-block constraint (CPU)
        assignment = ffd_pack_images(
            tokens_per_image,
            block_size,
            num_blocks,
            one_item_per_block=True,
            group_ids=group_ids,
        )

        # 4. CPU preprocessing
        cos, sin = compute_rotary_pos_emb(grid_for_ve, head_dim, vc.spatial_merge_size)
        pos_emb_idx, pos_emb_weight = compute_position_indices_and_weights(
            grid_for_ve, num_grid_per_side, vc.spatial_merge_size
        )
        bound_min, bound_max = compute_block_bounds(
            tokens_per_image, assignment, grid_for_ve
        )

        # 5. Scatter into block layout (CPU)
        packed_pixels = scatter_to_blocks(pixel_values, tokens_per_image, assignment)
        packed_cos = scatter_to_blocks(cos, tokens_per_image, assignment)
        packed_sin = scatter_to_blocks(sin, tokens_per_image, assignment)
        packed_idx = (
            scatter_to_blocks(pos_emb_idx.T, tokens_per_image, assignment)
            .permute(2, 0, 1)
            .contiguous()
        )
        packed_weight = (
            scatter_to_blocks(pos_emb_weight.T, tokens_per_image, assignment)
            .permute(2, 0, 1)
            .contiguous()
        )

        t_preprocess = _time.perf_counter()

        # 6. Build write_block_ids: map each VE output block to a cache block.
        #    Each item/group owns a contiguous run of VE blocks in item order, so
        #    flattening the per-hash cache block ids in order gives the 1:1 VE ->
        #    cache mapping; trailing (unused) VE blocks write to scratch.
        flat_cache_blocks = [b for blocks in cache_block_map for b in blocks]
        write_block_ids_list = flat_cache_blocks + [cache.scratch_block_id] * (
            num_blocks - len(flat_cache_blocks)
        )
        write_block_ids = torch.tensor(write_block_ids_list, dtype=torch.int64)

        # 7. Move to device, dispatch VE in cache-write mode. Scope the "encoder"
        # graph's resolved arguments, so the compiled encoder graph hashes to the
        # same arguments (-O3) as the capture and the warmup.
        with record_function_or_nullcontext("embed_multimodal: device_execution"):
            device = next(self.visual.parameters()).device
            with self._scoped_compiler_args("encoder"):
                visual_output = self.visual(
                    packed_pixels.to(device),
                    packed_idx.to(device),
                    packed_weight.to(device),
                    packed_cos.to(device),
                    packed_sin.to(device),
                    bound_min.to(device),
                    bound_max.to(device),
                    cache.buffer,
                    write_block_ids.to(device),
                )
        t_device = _time.perf_counter()
        fat_tensor = visual_output

        logger.debug(
            "[embed_multimodal] %d images, %d tokens: "
            "cpu_preprocess=%.1fms, device_exec=%.1fms",
            len(tokens_per_image),
            total_tokens,
            (t_preprocess - t_start) * 1000,
            (t_device - t_preprocess) * 1000,
        )

    def get_encoder(self) -> torch.nn.Module | None:
        """Return the vision tower. None on a language-only PD pool."""
        return self.visual

    def build_encoder_synthetic_inputs(
        self,
        bucket: int,
        encoder_neuron_config: EncoderNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Construct shape-only tensors matching the vision encoder forward signature.

        Returns a kwargs dict that can be passed directly to model.visual(**kwargs).
        """
        vc = self.config.vision_config
        block_size = encoder_neuron_config.encoder_attention_block_size
        patch_dim = (
            vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        )
        head_dim = vc.hidden_size // vc.num_heads
        spatial_merge_size = vc.spatial_merge_size

        # Pad num_blocks to be divisible by dp_size for the encoder's DP scatter.
        dp = encoder_neuron_config.dp_size
        num_blocks = math.ceil(math.ceil(bucket / block_size) / dp) * dp

        return {
            "pixel_values": torch.zeros(
                num_blocks, block_size, patch_dim, dtype=torch.bfloat16, device=device
            ),
            "pos_emb_idx": torch.zeros(
                4, num_blocks, block_size, dtype=torch.int32, device=device
            ),
            "pos_emb_weight": torch.zeros(
                4, num_blocks, block_size, dtype=torch.bfloat16, device=device
            ),
            "cos": torch.zeros(
                num_blocks, block_size, head_dim, dtype=torch.float32, device=device
            ),
            "sin": torch.zeros(
                num_blocks, block_size, head_dim, dtype=torch.float32, device=device
            ),
            "bound_min": torch.zeros(
                num_blocks, block_size, 1, dtype=torch.int32, device=device
            ),
            "bound_max": torch.zeros(
                num_blocks, block_size, 1, dtype=torch.int32, device=device
            ),
        }

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ):
        """Build model from HF config and NeuronConfig.

        For now, uses the same NeuronConfig for text. Vision NeuronConfig
        will be added when the vision encoder is implemented.
        """
        config = Qwen3VLConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )
        return cls(config)

    # ── KV Cache Management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def get_kv_spec(self):
        if self.language_model is None:
            return KVSpec(layers=[])
        layers = []
        for i, layer in enumerate(self.language_model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        if self.language_model is None:
            return
        for i, layer in enumerate(self.language_model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]
