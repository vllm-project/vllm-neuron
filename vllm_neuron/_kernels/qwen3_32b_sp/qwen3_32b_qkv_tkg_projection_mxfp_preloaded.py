# SPDX-License-Identifier: Apache-2.0
"""Vendored nkilib ``_qkv_tkg_projection_mxfp`` with a preloaded-weights form.

Copied as-is from ``nkilib.core.qkv.qkv_tkg_mx_impl`` (nki 0.7.0b1); the shape
math the sequence-parallel attention composer mirrors is unchanged.

Port notes (vs nkilib ``_qkv_tkg_projection_mxfp``)
---------------------------------------------------
The deliberate changes are marked inline with ``VENDOR CHANGE`` comments:

1. Renamed entry with an optional ``preloaded`` pair
   ``(weights_qtz_sb, weight_scales_sb)`` — the buffers the sequence-parallel
   attention composer allocated and DMA'd BEFORE its collectives (no inline
   marker; it is the signature).
2. When given, the first I block consumes the composer's buffers instead of
   allocating its own (``use_preloaded``; the Qwen3-32B decode shapes give
   exactly one I block and one H block).
3. Under ``use_preloaded`` the weight ``dma_copy`` and the per-quadrant E8M0
   scale scatter are skipped — the composer already issued both.

Delete this file when nkilib grows a preloaded-weights parameter (the same
follow-up as the pre-quantized-activation parameter in
``vllm_neuron/_kernels/qwen3_32b_sp/qwen3_32b_sp_norm_attn.py``).
"""

import math
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from nkilib.core.qkv.qkv_tkg_mx_impl import (
    I_BLOCK_SIZE,
    I_TILE_SIZE,
    WEIGHT_LOAD_BLOCK_SIZE,
    _load_and_broadcast_bias,
    _static_mx_dequantize_from_psum,
)
from nkilib.core.qkv.qkv_tkg_mx_utils import QKV_TKG_MXFP_Config
from nkilib.core.utils.allocator import sizeinbytes
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nkilib.core.utils.tiled_range import TiledRange

P_MAX = 128


def _qkv_tkg_projection_mxfp_preloaded(
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scales_sb: nl.NkiTensor,
    weights_qtz_hbm: nl.NkiTensor,
    weight_scales_hbm: nl.NkiTensor,
    cfg: QKV_TKG_MXFP_Config,
    bias_hbm: Optional[nl.NkiTensor] = None,
    input_scale: Optional[nl.NkiTensor] = None,
    preloaded=None,
) -> nl.NkiTensor:
    """
    QKV MXFP Projection:
        Computes: hidden_qtz_sb @ weights_qtz_hbm.

    Note: This version differs from the current qkv_tkg projection in a few ways:
        (A) hidden_qtz_sb (SBUF) is assumed to be in [H0, H1, BxS] layout (with H1 being the outer dimension in H).
            * This differs from the original [H0, BxS, H1] layout, which won't work with MXFP.
        (B) No Column Tiling: MXFP version cannot have column tiling support (hardware incompatibility)

    Note: Here H_packed stands for the original H // 4.
    Args:
        hidden_qtz_sb (nl.NkiTensor):
            Input hidden states tensor in SBUF (already quantized).
            Dtype: nl.float8_e4m3fn_x4
            Shape:
                [H0, H1_packed_shard, BxS] where H0=128, and H1_packed_shard = H_packed_shard // H0.
                 e.g., H1_packed = H // 512 (or H_packed // 128).
            Indexing: h = h1*H0 + h0, i.e., H_packed is viewed as H1_packed*H0 with H1_packed being the outer dimension.
                Note: This means HBM->SBUF load requires a transpose.
                Torch equivalent of going from [H, BxS] -> [H0, H1, BxS] would be:
                    hidden_sb = input.reshape(H1, H0, BxS).permute(1, 0, 2)

        hidden_scales_sb (nl.NkiTensor):
            Input quantization scales for MXFP in SBUF.
            Dtype:  uint8
            Shape: [H0, H1_packed_shard, BxS],
            Note: Same indexing assumptions as hidden_qtz_sb.

        weights_qtz_hbm (nl.NkiTensor):
            QKV projection weight tensor in HBM (unpacked fp8).
            Dtype: nl.float8_e4m3fn
            Shape: [H_packed, I, 4], can be viewed as [H0, H1_packed, I*4] in fp8.

        weight_scales_hbm (nl.NkiTensor):
            QKV weight quantization scales for MXFP in HBM.
            dtype: uint8
            Shape: [H // 32, I] == [H_packed // 8, I]
            Note: Since weights_qtz_hbm is already quantized, weight scales are 8x times smaller, not 32x.

        cfg (QKV_TKG_MXFP_Config): Kernel configuration (used for sharding info and dtype).

        bias_hbm (Optional[nl.NkiTensor]): [1, I], Optional bias tensor on HBM.

    Returns:
        output_hbm (nl.NkiTensor): [BxS, I], QKV projection output in HBM.
    """

    # Get sharding info from cfg
    num_shards = cfg.num_shards
    shard_id = cfg.shard_id
    hidden_orig_dtype = cfg.hidden_orig_dtype

    # We are deriving dimensions directly from already quantized tensors, hence shapes are 4x smaller already.
    # T_padded from tensor shape is padded to next multiple of 4; BxS is the original token count.
    H0, H1_packed_shard, T_padded = hidden_qtz_sb.shape
    BxS = cfg.B * cfg.S
    Q_WIDTH = 4
    H_packed_shard = H0 * H1_packed_shard
    H_packed, I, _ = weights_qtz_hbm.shape
    # hidden_qtz comes in pre-sliced per program, weights do not. We shard weights at load time with shard_id.
    # This is because we load weights with strided dma_copy pattern which cannot be done on pre-sliced tensor.

    _hbm_weight_dtype = weights_qtz_hbm.dtype
    # Weight and weight scales asserts are done at the top-level of the kernel.
    kernel_assert(
        hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4, but got {hidden_qtz_sb.dtype}.",
    )
    kernel_assert(
        hidden_scales_sb.dtype == nl.uint8,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_scales_sb.dtype == nl.uint8, but got {hidden_scales_sb.dtype}.",
    )

    # Calculate MXFP tiling constants
    NUM_WEIGHT_LOAD_BLOCKS = math.ceil(H_packed_shard / WEIGHT_LOAD_BLOCK_SIZE)
    # NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK is the same constant as in non-MXFP kernel, however we are
    # processing 4x tiles per WEIGHT_LOAD_BLOCK. WEIGHT_LOAD_BLOCK is number of rows of H we
    # load/process at once, e.g., load only [2048, I] and do the compute.
    NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK = math.ceil(WEIGHT_LOAD_BLOCK_SIZE / P_MAX)
    NUM_I_BLOCKS = math.ceil(I / I_BLOCK_SIZE)

    # Allocate output in HBM with physical (unpadded) BxS
    output_hbm = nl.ndarray((BxS, I), dtype=hidden_orig_dtype, buffer=nl.shared_hbm)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg)

    # For STATIC_MX: load and pre-combine per-Q/K/V dequant scales = input_scale * weight_scale
    # weight_scales_hbm: [1, 3] per-Q/K/V, input_scale: [P_MAX, 1] scalar
    # combined: [BxS, 3] — only BxS partitions needed for post-matmul dequant
    if cfg.is_static_quant:
        w_scale_sb = nl.ndarray((T_padded, 3), dtype=nl.float32, buffer=nl.sbuf)
        if weight_scales_hbm.shape[0] == 1:
            nisa.dma_copy(dst=w_scale_sb[0:1, 0:3], src=weight_scales_hbm[0:1, 0:3])
            stream_shuffle_broadcast(w_scale_sb, w_scale_sb)
        else:
            nisa.dma_copy(dst=w_scale_sb, src=weight_scales_hbm[:BxS, 0:3])
        # combined_dequant_scale[P, i] = w_scale[P, i] * input_scale[P, 0]
        nisa.activation(
            dst=w_scale_sb,
            op=nl.copy,
            data=w_scale_sb,
            scale=input_scale[:T_padded, 0:1],
        )
        combined_dequant_scale = w_scale_sb  # [BxS, 3]

    # For ROW_MX: transpose input_dequant_scale from [H0, BxS] to [BxS, 1] for output layout.
    if cfg.is_row_quant:
        """
        input_scale is [H0, BxS, 1] per-token dequant scale from row_quantization.
        All partitions hold the same values after broadcast. Reshape to [H0, BxS],
        take partition 0's [1, BxS] slice, transpose to [BxS, 1] for output layout.
        Weight scale [1, I] is loaded per i_block during dequant.
        """
        input_scale_2d = input_scale.reshape((H0, BxS))
        row_input_dequant_psum = nl.ndarray((BxS, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=row_input_dequant_psum, data=input_scale_2d[0:1, 0:BxS])
        row_input_dequant_sb = nl.ndarray(
            (T_padded, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        if BxS != T_padded:
            nisa.memset(dst=row_input_dequant_sb, value=0)
        nisa.tensor_copy(dst=row_input_dequant_sb[:BxS, :], src=row_input_dequant_psum)

    # QKV Projection loop - process each I_BLOCK_SIZE=4096 block (independent columns accumulated separately)
    for i_block in TiledRange(I, I_BLOCK_SIZE):
        # Allocate qkv_out_sb to store results of current I_BLOCK chunk
        qkv_out_sb = nl.ndarray(
            (T_padded, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf
        )

        """
        SBUF can run out-of-space in unallocated kernels. Add "list block" dimensions here,
        and/or make the kernel allocated. Also for now, SBUF space is approximate, use hardware
        specific SBUF space. None of this should be necessary in unallocated kernels, but it
        seems to be going out-of-space.
        """

        # Allocate weight buffer with double-buffering to overlap DMA with compute (if space permits)
        weight_tile_size = (
            H0
            * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK
            * i_block.size
            * Q_WIDTH
            * sizeinbytes(_hbm_weight_dtype)
        )
        weight_and_weight_scales_tile_size = 2 * weight_tile_size
        # Note: Quick heuristic for now, to be improved later. Since this is unallocated kernel we cannot know exact space taken.
        WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC = 16 * 1024 * 1024
        NUM_W_BUFFERS = (
            2
            if weight_and_weight_scales_tile_size * 2
            <= WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC
            else 1
        )

        """
        If WEIGHT_LOAD_BLOCK_SIZE were maximum of H_packed, then weight_qtz_sb shape would be:
        (H0, NUM_W_BUFFERS, H1_packed = H // 512, i_block_sz * Q_WIDTH) in fp8.
        We use fixed block size, but since H in weights is packed, we process fewer H_BLOCKS
        than in non-quant version.
        """
        # VENDOR CHANGE #2: take the composer's preloaded buffers when given.
        # The composer allocated them with the SAME shape math and issued the
        # load DMAs BEFORE its collectives, so the weights stream under the
        # exchange and the all-gather. Applies to the first I block only; the
        # Qwen3-32B decode shapes give exactly one I block and one H block.
        use_preloaded = preloaded is not None and i_block.index == 0
        if use_preloaded:
            weights_qtz_sb, _preloaded_scales_sb = preloaded
        else:
            # Allocate SBUF in the HBM fp8 dtype for HWDGE type matching,
            # then view-cast to float8_e4m3fn_x4 for nc_matmul_mx.
            weights_qtz_sb = nl.ndarray(
                (
                    H0,
                    NUM_W_BUFFERS,
                    NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK,
                    i_block.size * Q_WIDTH,
                ),
                dtype=_hbm_weight_dtype,
                buffer=nl.sbuf,
            )
        weights_qtz_sb_mx = weights_qtz_sb.view(nl.float8_e4m3fn_x4)

        if cfg.is_static_quant or cfg.is_row_quant:
            # STATIC_MX / ROW_MX: dummy 127 scales — minimal free dim, reused at index 0
            weight_scales_sb = nl.ndarray(
                (H0, 1, 1, I_TILE_SIZE),
                dtype=nl.uint8,
                buffer=nl.sbuf,
            )
            nisa.memset(dst=weight_scales_sb, value=127, engine=nisa.gpsimd_engine)
        elif use_preloaded:
            weight_scales_sb = _preloaded_scales_sb
        else:
            weight_scales_sb = nl.ndarray(
                (H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block.size),
                dtype=nl.uint8,
                buffer=nl.sbuf,
            )

        # Allocate PSUM banks for accumulation (num_i_tiles_per_i_block <= 8)
        mm_result_psum = []
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            psum_tile = nl.ndarray(
                (P_MAX, i_tile.size),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            mm_result_psum.append(psum_tile)

        # Process WEIGHT_LOAD_BLOCK_SIZE at a time, e.g., load [WEIGHT_LOAD_BLOCK_SIZE, I] and do the compute.
        for h_weight_block in TiledRange(H_packed_shard, WEIGHT_LOAD_BLOCK_SIZE):
            # TODO: Remove this once below loops are changed to TiledRange.
            num_128_tiles_in_current_weight_load_block = math.ceil(
                h_weight_block.size / P_MAX
            )

            # Buffer index for double-buffering (cycles 0, 1, 0, 1, ...)
            weight_buffer_idx = h_weight_block.index % NUM_W_BUFFERS

            """
            Load weight tile with single DMA using access patterns.
            Contiguous H sharding: shard 0 gets H[0:H/2], shard 1 gets H[H/2:H].
            HBM source pattern: weights_qtz_hbm is [H_packed, I, 4] fp8 = [H_packed, I*4] flat.
            Offset into this shard's contiguous H portion.
            """
            h_shard_base_offset = shard_id * H_packed_shard * I * Q_WIDTH
            weights_hbm_load_pattern = [
                [I * Q_WIDTH, P_MAX],
                [P_MAX * I * Q_WIDTH, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size * Q_WIDTH],
            ]
            weights_hbm_load_offset = (
                h_shard_base_offset
                + h_weight_block.index * I * Q_WIDTH
                + i_block.start_offset * Q_WIDTH
            )

            # SBUF dest pattern: weights_qtz_sb is [H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block_sz*4] fp8
            weights_sb_load_pattern = [
                [
                    NUM_W_BUFFERS
                    * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK
                    * i_block.size
                    * Q_WIDTH,
                    P_MAX,
                ],
                [i_block.size * Q_WIDTH, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size * Q_WIDTH],
            ]
            weights_sb_load_offset = (
                weight_buffer_idx
                * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK
                * i_block.size
                * Q_WIDTH
            )

            # VENDOR CHANGE #3: the composer already issued this DMA. The
            # preload covers every H block of the first I block, and the
            # composer asserts NUM_WEIGHT_LOAD_BLOCKS == 1.
            if not use_preloaded:
                # Load weights
                nisa.dma_copy(
                    dst=weights_qtz_sb.ap(
                        pattern=weights_sb_load_pattern, offset=weights_sb_load_offset
                    ),
                    src=weights_qtz_hbm.ap(
                        pattern=weights_hbm_load_pattern, offset=weights_hbm_load_offset
                    ),
                )

            """
            Load packed weight scales.
            MX: Load from HBM with quadrant placement for nc_matmul_mx.
            STATIC_MX / ROW_MX: Scales already memset to 127 once at allocation — skip per-block load.
            """
            if not cfg.is_static_quant and not cfg.is_row_quant and not use_preloaded:
                # MX Scale layout constants:
                # Quadrant placement: HBM has 16 contiguous scale rows per H512 tile.
                # SBUF needs them spread across 4 quadrants (4 rows each at offsets 0, 32, 64, 96).
                SCALE_GROUP_SIZE = 32  # One scale per 32 H elements
                SCALES_PER_H512_TILE = (
                    512 // SCALE_GROUP_SIZE
                )  # = 16 scale rows per H512 tile in HBM
                SBUF_QUADRANT_SIZE = 32
                NUM_QUADRANTS = H0 // SBUF_QUADRANT_SIZE  # = 4
                SCALES_PER_QUADRANT = SCALES_PER_H512_TILE // NUM_QUADRANTS  # = 4

                # weight_scales_hbm shape: [H // 32, I] = [H_packed // 8, I]
                # Contiguous sharding: shard's scale rows start at shard_id * (H_packed_shard // 8)
                H_packed_shard_scale_rows = (
                    H_packed_shard // 8
                )  # Number of scale rows for this shard
                scale_shard_base_offset = shard_id * H_packed_shard_scale_rows * I

                # num_128_tiles_in_current_weight_load_block iterations
                # for h_tile_idx_in_block in range(num_128_tiles_in_current_weight_load_block):
                for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                    h_tile_idx_in_h = (
                        h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK
                        + h_tile_in_block.index
                    )

                    for quad_idx in range(NUM_QUADRANTS):
                        # Local offset within this shard's scale rows
                        local_hbm_row_offset = (
                            h_tile_idx_in_h * SCALES_PER_H512_TILE
                            + quad_idx * SCALES_PER_QUADRANT
                        ) * I + i_block.start_offset
                        hbm_row_offset = scale_shard_base_offset + local_hbm_row_offset

                        nisa.dma_copy(
                            dst=weight_scales_sb[
                                quad_idx * SBUF_QUADRANT_SIZE : quad_idx
                                * SBUF_QUADRANT_SIZE
                                + SCALES_PER_QUADRANT,
                                weight_buffer_idx,
                                h_tile_in_block.index,
                                : i_block.size,
                            ],
                            src=weight_scales_hbm.ap(
                                pattern=[[I, SCALES_PER_QUADRANT], [1, i_block.size]],
                                offset=hbm_row_offset,
                                dtype=nl.uint8,
                            ),
                        )

            # Matmul for current weight load
            # num_128_tiles_in_current_weight_load_block iterations
            for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                h_tile_idx_in_h = (
                    h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK
                    + h_tile_in_block.index
                )

                for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
                    # STATIC_MX / ROW_MX: scales are minimal, always index at 0 in free dims
                    if cfg.is_static_quant or cfg.is_row_quant:
                        stat_scale = hidden_scales_sb[0:H0, 0, 0:T_padded]
                        mov_scale = weight_scales_sb[0:H0, 0, 0, 0 : i_tile.size]
                    else:
                        stat_scale = hidden_scales_sb[0:H0, h_tile_idx_in_h, 0:T_padded]
                        mov_scale = weight_scales_sb[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ]
                    nisa.nc_matmul_mx(
                        dst=mm_result_psum[i_tile.index][0:T_padded, 0 : i_tile.size],
                        stationary=hidden_qtz_sb[0:H0, h_tile_idx_in_h, 0:T_padded],
                        # Read from the matmul-side alias (relabeled to
                        # ``nl.float8_e4m3fn_x4``); DMA writes target the
                        # storage tile ``weights_qtz_sb`` typed to the HBM
                        # source dtype to keep HWDGE happy.
                        moving=weights_qtz_sb_mx[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                        stationary_scale=stat_scale,
                        moving_scale=mov_scale,
                    )

        # Copy PSUM results to SBUF.
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            if cfg.is_row_quant:
                # ROW_MX: copy PSUM to SBUF with per-token input dequant scale
                nisa.activation(
                    dst=qkv_out_sb[
                        :T_padded,
                        i_tile.start_offset : i_tile.start_offset + i_tile.size,
                    ],
                    op=nl.copy,
                    data=mm_result_psum[i_tile.index][:T_padded, 0 : i_tile.size],
                    scale=row_input_dequant_sb,
                )
            elif cfg.is_static_quant:
                # STATIC_MX: fuse PSUM→SBUF copy with per-Q/K/V dequant scale multiply
                _static_mx_dequantize_from_psum(
                    dst_sb=qkv_out_sb[
                        :T_padded,
                        i_tile.start_offset : i_tile.start_offset + i_tile.size,
                    ],
                    src_psum=mm_result_psum[i_tile.index][:T_padded, 0 : i_tile.size],
                    dequant_scale_sb=combined_dequant_scale,
                    cfg=cfg,
                    I_start=i_block.start_offset + i_tile.start_offset,
                    I_size=i_tile.size,
                )
            else:
                # MX: optionally fuse bias addition during PSUM copy
                # So we don't double-add bias.
                if cfg.add_bias and cfg.shard_id == 0:
                    nisa.tensor_tensor(
                        dst=qkv_out_sb[
                            :T_padded,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                        data1=mm_result_psum[i_tile.index][:T_padded, 0 : i_tile.size],
                        data2=bias_sb[
                            :T_padded,
                            i_block.start_offset
                            + i_tile.start_offset : i_block.start_offset
                            + i_tile.start_offset
                            + i_tile.size,
                        ],
                        op=nl.add,
                    )
                else:
                    nisa.tensor_copy(
                        dst=qkv_out_sb[
                            :T_padded,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                        src=mm_result_psum[i_tile.index][:T_padded, 0 : i_tile.size],
                    )

        # STATIC_MX: bias after dequant (dequant already fused into PSUM copy above)
        if cfg.is_static_quant and cfg.add_bias and cfg.shard_id == 0:
            nisa.tensor_tensor(
                dst=qkv_out_sb[:T_padded, 0 : i_block.size],
                data1=qkv_out_sb[:T_padded, 0 : i_block.size],
                data2=bias_sb[
                    :T_padded,
                    i_block.start_offset : i_block.start_offset + i_block.size,
                ],
                op=nl.add,
            )

        # ROW_MX: apply per-column weight dequant scale, then bias
        if cfg.is_row_quant:
            # Load w_scale [1, i_block.size] and broadcast to [T_padded, i_block.size]
            row_w_scale_sb = nl.ndarray(
                (T_padded, i_block.size), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=row_w_scale_sb[0:1, 0 : i_block.size],
                src=weight_scales_hbm[
                    0:1, i_block.start_offset : i_block.start_offset + i_block.size
                ],
            )
            stream_shuffle_broadcast(row_w_scale_sb, row_w_scale_sb)
            nisa.tensor_tensor(
                dst=qkv_out_sb[:T_padded, 0 : i_block.size],
                data1=qkv_out_sb[:T_padded, 0 : i_block.size],
                data2=row_w_scale_sb[:T_padded, 0 : i_block.size],
                op=nl.multiply,
            )
            if cfg.add_bias and cfg.shard_id == 0:
                nisa.tensor_tensor(
                    dst=qkv_out_sb[:T_padded, 0 : i_block.size],
                    data1=qkv_out_sb[:T_padded, 0 : i_block.size],
                    data2=bias_sb[
                        :T_padded,
                        i_block.start_offset : i_block.start_offset + i_block.size,
                    ],
                    op=nl.add,
                )

        # Cross-core reduction via sendrecv when LNC > 1
        # Each core has partial sum from its H shard, need to sum across cores
        if num_shards > 1:
            qkv_recv_sb = nl.ndarray(
                (T_padded, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf
            )
            other_core = 1 - shard_id
            nisa.sendrecv(
                src=qkv_out_sb,
                dst=qkv_recv_sb,
                send_to_rank=other_core,
                recv_from_rank=other_core,
                pipe_id=0,
            )
            nisa.tensor_tensor(
                dst=qkv_out_sb, data1=qkv_out_sb, data2=qkv_recv_sb, op=nl.add
            )

        # Store to HBM (only physical BxS rows, discarding padding)
        nisa.dma_copy(
            dst=output_hbm[
                :BxS, i_block.start_offset : i_block.start_offset + i_block.size
            ],
            src=qkv_out_sb[:BxS, 0 : i_block.size],
        )

    return output_hbm
