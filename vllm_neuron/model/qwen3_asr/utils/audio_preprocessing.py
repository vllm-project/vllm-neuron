# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing functions for the Qwen3-ASR audio encoder.

These functions run on CPU before the compiled audio model. They compute:
- Sinusoids Position Embedding
- The block-packed layout (``AudioBlockAssignment``)
- Attention bounds (bound_min, bound_max)

All functions are pure tensor operations with no learnable parameters.
The actual embedding lookup runs on-device via ``Qwen3ASRAudioPosEmbed``.

The slot unit is one base chunk. Segments (a full window, or a clip tail) pack
into blocks with a deterministic per-clip next-fit, so block assignment is
arithmetic, not a search. A segment never crosses a block boundary.

Reference: HF ``cu_seqlens`` computation (``get_audio_cu_seqlens``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# The sequence-packed audio attention kernel reads the bounds with a per-block
# stride of ``_ATTN_SEQ_ALIGN * ceil(block_size / _ATTN_SEQ_ALIGN)``, but the
# real stride in a ``[num_blocks, block_size, 1]`` tensor is ``block_size``. So
# the attention block size must be a multiple of this value. A non-aligned size
# makes the NEFF bake an out-of-bounds DMA that the runtime rejects at load.
# In qwen3-asr one window has 104 post-CNN tokens, so the smallest usable block
# size is 128.
_ATTN_SEQ_ALIGN = 128

# Kernel constraint in vllm_neuron.functional.attention.attention_cte. The
# attention batch dim is num_blocks * heads_per_rank; above this the call
# silently falls back to the torch implementation.
_MAX_ATTN_BS = 512


def _get_feat_extract_output_lengths(input_lengths, n_window=50):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder

    Matches HF ``modelling_qwen3_asr.py's _get_feat_extract_output_lengths`
    """
    num_mel_frames_per_chunk = n_window * 2
    input_lengths_leave = input_lengths % num_mel_frames_per_chunk
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (input_lengths // num_mel_frames_per_chunk) * 13
    )


def per_chunk_post_cnn_len(n_window: int) -> int:
    """Return the post-CNN time steps for one base chunk (13 for n_window=50).

    The audio encoder reduces ``2 * n_window`` mel frames with three
    (k=3, s=2, p=1) convolution layers. Each layer maps a length ``x`` to
    ``(x - 1) // 2 + 1``.

    Args:
        n_window: base chunk half-size. One base chunk is ``2 * n_window`` mel
            frames.

    Returns:
        The per-chunk post-CNN length. It sizes the sinusoid position-embedding
        buffer (``max_position_embeddings``).
    """
    length = n_window * 2
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def compute_window_tokens(
    max_position_embeddings: int, n_window: int, n_window_infer: int
) -> int:
    """Post-CNN tokens in one full inference window (104 for Qwen3-ASR-1.7B).

    One inference window holds ``n_window_infer / (2 * n_window)`` base chunks,
    and each base chunk gives ``max_position_embeddings`` post-CNN time steps.
    """
    chunks_per_window = n_window_infer // (n_window * 2)
    return max_position_embeddings * chunks_per_window


def resolve_block_geometry(
    block_size: int,
    window_tokens: int,
    n_window: int,
    n_window_infer: int,
) -> tuple[int, int, int]:
    """Validate the audio block layout and return its slot geometry.

    The slot unit is one base chunk (``num_tokens_per_chunk`` post-CNN tokens). A block
    holds ``block_size // num_tokens_per_chunk`` chunk slots. This is the one place that
    checks the block size against the kernel and the window, and checks that the
    window, chunk and inference-window sizes agree.

    Args:
        block_size: static per-block attention sequence length.
        window_tokens: post-CNN tokens in one full inference window (one
            full-window attention segment). A full window must fit in one block.
        n_window: base chunk half-size (num_mel_frames_per_chunk = 2 * n_window mel frames).
        n_window_infer: mel frames in one inference window.

    Returns:
        A ``(slots_per_block, num_tokens_per_chunk, num_mel_frames_per_chunk)`` tuple:
          - slots_per_block: whole chunk slots that fit in one block.
          - num_tokens_per_chunk: post-CNN tokens in one base chunk (the slot size).
          - num_mel_frames_per_chunk: mel frames in one base chunk.

    Raises:
        ValueError: if ``block_size`` is not a multiple of ``_ATTN_SEQ_ALIGN``,
            if it cannot hold one whole window, if ``window_tokens`` is not a
            whole number of base chunks, or if ``n_window_infer`` disagrees with
            the window and chunk sizes.
    """
    num_tokens_per_chunk = per_chunk_post_cnn_len(n_window)
    num_mel_frames_per_chunk = 2 * n_window
    if block_size % _ATTN_SEQ_ALIGN != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of {_ATTN_SEQ_ALIGN}; "
            f"the attention kernel reads the bounds with a per-block stride of "
            f"{_ATTN_SEQ_ALIGN} * ceil(block_size / {_ATTN_SEQ_ALIGN})."
        )
    if block_size < window_tokens:
        raise ValueError(
            f"block_size ({block_size}) must be at least window_tokens "
            f"({window_tokens}); one full-window segment cannot span two "
            f"attention blocks."
        )
    if window_tokens % num_tokens_per_chunk != 0:
        raise ValueError(
            f"window_tokens ({window_tokens}) must be a whole number of base "
            f"chunks (num_tokens_per_chunk={num_tokens_per_chunk})."
        )
    chunks_per_window = window_tokens // num_tokens_per_chunk
    if n_window_infer != chunks_per_window * num_mel_frames_per_chunk:
        raise ValueError(
            f"n_window_infer ({n_window_infer}) must equal chunks_per_window "
            f"({chunks_per_window}) * num_mel_frames_per_chunk ({num_mel_frames_per_chunk}). The window, chunk "
            f"and inference-window sizes are inconsistent."
        )
    return (
        block_size // num_tokens_per_chunk,
        num_tokens_per_chunk,
        num_mel_frames_per_chunk,
    )


def get_audio_window_tokens(hf_config) -> int:
    """Return the post-CNN tokens in one inference window for an HF config.

    This is the hf_config entry point over ``compute_window_tokens``. The
    generic ``EncoderNeuronConfig`` calls it to size the audio encoder budget,
    so it does not need the HF nesting or the CNN reduction. It derives
    ``num_tokens_per_chunk`` from ``n_window`` and does not read
    ``max_position_embeddings``. Some HF audio-config variants do not set it.

    Args:
        hf_config: HuggingFace config with ``thinker_config.audio_config``.

    Returns:
        The post-CNN tokens in one full inference window.

    Raises:
        ValueError: if ``hf_config.thinker_config.audio_config`` is missing.
    """
    audio = getattr(getattr(hf_config, "thinker_config", None), "audio_config", None)
    if audio is None:
        raise ValueError(
            "hf_config.thinker_config.audio_config is missing. Set it, because "
            "the audio window-token budget needs it."
        )
    num_tokens_per_chunk = per_chunk_post_cnn_len(audio.n_window)
    return compute_window_tokens(
        num_tokens_per_chunk, audio.n_window, audio.n_window_infer
    )


def get_max_real_tokens_per_block(hf_config, block_size: int) -> int:
    """Return the real post-CNN tokens that one attention block holds.

    A block holds ``slots_per_block = block_size // num_tokens_per_chunk`` chunk slots.
    Each slot holds ``num_tokens_per_chunk`` real tokens. The block capacity is the
    product ``slots_per_block * num_tokens_per_chunk``. This is the exact per-block
    real-token capacity, not the whole-window lower bound
    ``windows_per_block * window_tokens``. The packer fills the extra slots.

    This is the hf_config entry point over ``resolve_block_geometry``. The
    generic ``EncoderNeuronConfig`` calls it to size the audio encoder budget.

    Args:
        hf_config: HuggingFace config with ``thinker_config.audio_config``.
        block_size: static per-block attention sequence length.

    Returns:
        The real post-CNN tokens in one block.

    Raises:
        ValueError: if ``hf_config.thinker_config.audio_config`` is missing, or
            if the block layout is invalid. See ``resolve_block_geometry``.
    """
    audio = getattr(getattr(hf_config, "thinker_config", None), "audio_config", None)
    if audio is None:
        raise ValueError(
            "hf_config.thinker_config.audio_config is missing. Set it, because "
            "the audio block capacity needs it."
        )
    window_tokens = get_audio_window_tokens(hf_config)
    slots_per_block, num_tokens_per_chunk, _ = resolve_block_geometry(
        block_size, window_tokens, audio.n_window, audio.n_window_infer
    )
    return slots_per_block * num_tokens_per_chunk


@dataclass(frozen=True)
class AudioBlockAssignment:
    """Where each base chunk sits in the block-packed audio layout.

    The slot unit is one base chunk (``num_tokens_per_chunk`` post-CNN tokens). A block
    holds ``slots_per_block = block_size // num_tokens_per_chunk`` chunk slots. Chunk
    slot ``s`` lives in block ``s // slots_per_block`` at block-local token
    offset ``(s % slots_per_block) * num_tokens_per_chunk``.

    Attention runs over segments, not slots. A segment is one inference window
    (``window_tokens`` real tokens = ``window_tokens // num_tokens_per_chunk`` whole
    chunks) or a clip's tail remainder. A segment's chunk slots are contiguous
    and never cross a block boundary, so its real tokens form one
    ``[off, off + len)`` attention span. See ``get_audio_cu_seqlens`` in HF.

    Attributes:
        block_size: static per-block attention sequence length.
        window_tokens: post-CNN tokens in one full inference window (a
            full-window segment).
        num_tokens_per_chunk: post-CNN tokens in one base chunk (the slot size).
        slots_per_block: whole chunk slots that fit in one block.
        num_blocks: attention blocks in this layout, after bucketing and dp
            padding.
        min_blocks: attention blocks the batch needs before bucketing and dp
            padding. num_blocks is always >= min_blocks.
        num_mel_frames_per_chunk: mel frames in one base chunk.
        chunk_lens: real post-CNN tokens in each chunk slot, one entry per slot.
            0 marks a padding slot.
        chunk_source: real base-chunk index feeding each slot. Padding slots
            point at chunk 0; the bounds mask them so they attend to nothing.
        segments: attention segments, one tuple per segment. The length is the
            total segment count over all clips. Each tuple is
            ``(block, slot_start, num_real_tokens)``:
              - ``block``: the attention block that holds the segment,
                ``0 <= block < num_blocks``.
              - ``slot_start``: the block-local chunk slot the segment starts at,
                ``0 <= slot_start < slots_per_block``. The segment spans
                ``ceil(num_real_tokens / num_tokens_per_chunk)`` consecutive slots and
                never crosses a block boundary.
              - ``num_real_tokens``: real post-CNN tokens in the segment,
                ``window_tokens`` for a full window or the remainder for a tail.
        clip_slots: chunk slots owned by each clip, including its own padding
            slots under ``one_item_per_block``.
        one_item_per_block: whether a block holds chunks from only one clip.
            True is required by the on-device encoder cache, which allocates and
            evicts one mm_item per block.
    """

    block_size: int
    window_tokens: int
    num_tokens_per_chunk: int
    slots_per_block: int
    num_blocks: int
    min_blocks: int
    num_mel_frames_per_chunk: int
    chunk_lens: tuple[int, ...]
    chunk_source: tuple[int, ...]
    segments: tuple[tuple[int, int, int], ...]
    clip_slots: tuple[tuple[int, ...], ...]
    one_item_per_block: bool

    @property
    def num_slots(self) -> int:
        """Chunk slots in the layout, including padding slots."""
        return self.num_blocks * self.slots_per_block

    def __post_init__(self):
        if not any(self.chunk_lens):
            raise ValueError(
                "AudioBlockAssignment has no real chunk; chunk_lens is all "
                "zero. Use plan_audio_blocks to build it from at least one clip."
            )

    @property
    def mel_chunks(self) -> int:
        """Real base chunks the mel stream must carry."""
        return max(self.chunk_source[s] for s in self.real_slots) + 1

    @property
    def real_slots(self) -> tuple[int, ...]:
        """Slots that hold a real chunk, in slot order."""
        return tuple(s for s, cl in enumerate(self.chunk_lens) if cl > 0)

    @property
    def feature_length(self) -> int:
        """Mel frames the (compact) encoder input must carry."""
        return self.mel_chunks * self.num_mel_frames_per_chunk

    def chunk_slots(self) -> torch.Tensor:
        """``[num_slots]`` int64 gather index, an encoder graph input.

        The encoder runs ``index_select(0, chunk_slots)`` on its per-chunk conv
        output ``[num_chunks, num_tokens_per_chunk, H]``, so real chunks land in slot
        order. A padding slot points at chunk 0 (a harmless duplicate); the
        bounds give it an empty span so it does not attend.
        """
        return torch.tensor(self.chunk_source, dtype=torch.int64)

    def block_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(bound_min, bound_max)`` each ``[num_blocks, block_size, 1]`` int32.

        Attention is block-local: token ``i`` in block ``b`` attends to
        ``[bound_min[b, i], bound_max[b, i])`` in block-local index space. A
        segment at block-local offset ``off`` with ``L`` real tokens gets
        ``[off, off + L)`` on its real tokens, which is full attention within the
        segment and nothing across segments. Padding tokens keep ``[0, 0)`` and
        attend to nothing.
        """
        bound_min = torch.zeros(self.num_blocks, self.block_size, 1, dtype=torch.int32)
        bound_max = torch.zeros(self.num_blocks, self.block_size, 1, dtype=torch.int32)
        for block, slot_start, seg_len in self.segments:
            offset = slot_start * self.num_tokens_per_chunk
            bound_min[block, offset : offset + seg_len, 0] = offset
            bound_max[block, offset : offset + seg_len, 0] = offset + seg_len
        return bound_min, bound_max

    def unpack_indices(self) -> torch.Tensor:
        """``[total_valid]`` int64 flat index into ``num_blocks * block_size``.

        Use it to gather the real tokens out of the block-packed encoder output,
        in clip then segment then chunk order. This works for every packing mode,
        unlike a per-block prefix slice.
        """
        idx: list[int] = []
        for s, cl in enumerate(self.chunk_lens):
            if cl == 0:
                continue
            b, slot_in_block = divmod(s, self.slots_per_block)
            start = b * self.block_size + slot_in_block * self.num_tokens_per_chunk
            idx.extend(range(start, start + cl))
        return torch.tensor(idx, dtype=torch.int64)


def plan_audio_blocks(
    feature_lens: torch.Tensor | Sequence[int],
    *,
    block_size: int,
    window_tokens: int,
    n_window: int,
    n_window_infer: int,
    num_blocks: int | None = None,
    buckets: Sequence[int] | None = None,
    one_item_per_block: bool = True,
    dp_size: int = 1,
    heads_per_rank: int | None = None,
) -> AudioBlockAssignment:
    """Assign each clip's base chunks to block-packed chunk slots.

    Args:
        feature_lens: ``[num_clips]`` valid mel frames per clip.
        block_size: static per-block attention sequence length.
        window_tokens: post-CNN tokens in one full inference window (segment).
        n_window: base chunk half-size (num_mel_frames_per_chunk = 2 * n_window mel frames).
        n_window_infer: mel frames in one inference window. Checked for
            consistency against window_tokens and num_mel_frames_per_chunk.
        num_blocks: if given, pad the layout up to this many blocks. Extra
            blocks are all padding and attend to nothing. Give num_blocks or
            buckets, not both.
        buckets: if given, pick the smallest bucket that holds the packed need
            and pad the layout up to it. Each bucket is a block count. This
            collapses the two-pass "count then re-plan" into one call.
        one_item_per_block: keep each clip in its own blocks. Required by the
            on-device encoder cache. False packs clips together, which saves
            blocks but cannot be cached.
        dp_size: pad ``num_blocks`` up to a multiple of this.
        heads_per_rank: if given, warn when the attention batch dim exceeds the
            kernel limit.

    Returns:
        The ``AudioBlockAssignment`` for this batch.

    Raises:
        ValueError: on an unusable ``block_size``, a zero-length clip, both
            ``num_blocks`` and ``buckets`` given, a ``num_blocks`` too small to
            hold the batch, or no ``buckets`` entry large enough.
    """
    slots_per_block, num_tokens_per_chunk, num_mel_frames_per_chunk = (
        resolve_block_geometry(block_size, window_tokens, n_window, n_window_infer)
    )

    if isinstance(feature_lens, torch.Tensor):
        lens = [int(v) for v in feature_lens.flatten().tolist()]
    else:
        lens = [int(v) for v in feature_lens]

    # Build the layout block by block. Each block holds up to slots_per_block
    # chunk slots. A segment (a full window or a clip tail) is placed whole and
    # never crosses a block boundary (guaranteed: a full window is
    # window_tokens // num_tokens_per_chunk chunks and block_size >= window_tokens).
    # Under one_item_per_block a clip starts a fresh block, so no block mixes
    # clips.
    blocks: list[
        list[tuple[int, int]]
    ] = []  # per block: (num_mel_frames_per_chunk, chunk_source)
    cur: list[tuple[int, int]] = []
    segments: list[tuple[int, int, int]] = []
    clip_slots: list[tuple[int, ...]] = []
    next_chunk = 0

    def flush() -> None:
        nonlocal cur
        if cur:
            blocks.append(cur)
            cur = []

    for clip, mel_len in enumerate(lens):
        if mel_len <= 0:
            raise ValueError(
                f"feature_lens[{clip}] is {mel_len}; a clip must have at least "
                f"one valid mel frame. A zero-length clip still occupies mel "
                f"frames but produces no chunk, which silently shifts every "
                f"later block."
            )

        aftercnn = int(
            _get_feat_extract_output_lengths(torch.tensor(mel_len), n_window)
        )
        # Segments mirror HF get_audio_cu_seqlens: full windows then a tail.
        num_full, tail = divmod(aftercnn, window_tokens)
        seg_lens = [window_tokens] * num_full + ([tail] if tail else [])

        if one_item_per_block:
            flush()  # start this clip on a fresh block
        clip_start_block = len(blocks)
        clip_slot_ids: list[int] = []

        for seg_len in seg_lens:
            c = math.ceil(
                seg_len / num_tokens_per_chunk
            )  # chunk slots this segment needs
            if len(cur) + c > slots_per_block:
                flush()  # segment does not fit the current block; start a new one
            seg_block = len(blocks)  # cur flushes to this index
            seg_slot_start = len(cur)
            segments.append((seg_block, seg_slot_start, seg_len))
            for k in range(c):
                tok = min(num_tokens_per_chunk, seg_len - k * num_tokens_per_chunk)
                clip_slot_ids.append(seg_block * slots_per_block + seg_slot_start + k)
                cur.append((tok, next_chunk))
                next_chunk += 1

        if one_item_per_block:
            flush()  # this clip owns whole blocks; the next clip starts fresh
            # The clip owns every slot (real + pad) in its whole-block run.
            clip_slots.append(
                tuple(
                    range(
                        clip_start_block * slots_per_block,
                        len(blocks) * slots_per_block,
                    )
                )
            )
        else:
            clip_slots.append(tuple(clip_slot_ids))

    flush()  # flush the last (possibly cross-clip) block

    needed_blocks = len(blocks)
    if num_blocks is not None and buckets is not None:
        raise ValueError(
            "plan_audio_blocks takes num_blocks or buckets, not both. num_blocks "
            "pins the block count; buckets selects the smallest block count that "
            "fits."
        )

    if buckets is not None:
        if not buckets:
            raise ValueError(
                "buckets is empty. Configure num_encoder_input_buckets with at "
                "least one block count for audio."
            )
        resolved = next((b for b in sorted(buckets) if b >= needed_blocks), None)
        if resolved is None:
            raise ValueError(
                f"Audio needs {needed_blocks} blocks, above the largest bucket "
                f"({max(buckets)}). Add a larger num_encoder_input_buckets entry "
                f"instead of dropping audio."
            )
    elif num_blocks is None:
        resolved = needed_blocks
    elif num_blocks < needed_blocks:
        raise ValueError(
            f"num_blocks ({num_blocks}) is too small for this batch, which "
            f"needs {needed_blocks} blocks at block_size={block_size} "
            f"(slots_per_block={slots_per_block}). Pick a larger bucket "
            f"instead of dropping audio."
        )
    else:
        resolved = num_blocks

    if dp_size > 1:
        resolved = math.ceil(resolved / dp_size) * dp_size

    # Pad up to the resolved block count with empty (all-padding) blocks, then
    # flatten to global chunk slots. Slot (block b, local j) has global index
    # b * slots_per_block + j, so the layout matches segments and clip_slots.
    while len(blocks) < resolved:
        blocks.append([])
    chunk_lens: list[int] = []
    chunk_source: list[int] = []
    for block in blocks:
        for tok, src in block:
            chunk_lens.append(tok)
            chunk_source.append(src)
        pad = slots_per_block - len(block)
        chunk_lens.extend([0] * pad)
        chunk_source.extend([0] * pad)

    if heads_per_rank is not None and resolved * heads_per_rank > _MAX_ATTN_BS:
        logger.warning(
            "Audio attention batch dim is %d (num_blocks=%d * heads_per_rank=%d), "
            "above the kernel limit of %d. The attention call falls back to the "
            "torch implementation. Use a larger block size to cut num_blocks.",
            resolved * heads_per_rank,
            resolved,
            heads_per_rank,
            _MAX_ATTN_BS,
        )

    return AudioBlockAssignment(
        block_size=block_size,
        window_tokens=window_tokens,
        num_tokens_per_chunk=num_tokens_per_chunk,
        slots_per_block=slots_per_block,
        num_blocks=resolved,
        min_blocks=needed_blocks,
        num_mel_frames_per_chunk=num_mel_frames_per_chunk,
        chunk_lens=tuple(chunk_lens),
        chunk_source=tuple(chunk_source),
        segments=tuple(segments),
        clip_slots=tuple(clip_slots),
        one_item_per_block=one_item_per_block,
    )
