# SPDX-License-Identifier: Apache-2.0
import torch
import nki

from typing import Optional, Tuple, Union
from torch import Tensor

from nkilib.core.attention.attention_cte import attention_cte
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.hardware_config import get_logical_nc_config

jitted_attention_cte = nki.jit()(attention_cte)

# Match kernel constraints in attention_cte
MAX_BS = 512
MAX_SEQLEN = 131072
MAX_HEAD_DIM = 128


# TODO: Define this NKI attention_cte kernel constraints check in NKILIB
def _can_use_flash_attention_kernel(
    v: Tensor,
) -> bool:
    """
    Check if the NKI attention_cte kernel can be used based on constraints.
    Returns False if any constraint is violated, triggering torch fallback.
    """

    if not can_run_kernel(v):
        return False

    # V is always [B, S, D]
    B, S, D = v.shape

    if B > MAX_BS:
        return False

    if S > MAX_SEQLEN:
        return False

    if D > MAX_HEAD_DIM:
        return False

    return True


def _torch_attention_impl(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    causal_mask: bool = True,
    sliding_window: int = 0,
    sink: Optional[Tensor] = None,
    tp_q: bool = True,
    tp_k: bool = False,
    tp_out: bool = False,
    k_prior: Optional[Tensor] = None,
    v_prior: Optional[Tensor] = None,
    prior_used_len: Optional[Tensor] = None,
    bound_min: Optional[Tensor] = None,
    bound_max: Optional[Tensor] = None,
    cp_offset: int = 0,  # Q position offset for CP causal mask
) -> Tensor:
    """
    PyTorch implementation of attention matching attention_cte kernel behavior.

    Computation: softmax(scale * Q @ K^T + mask) @ V

    Input layouts (controlled by transpose flags):
        q: [B, S_q, D] when tp_q=True, [B, D, S_q] when tp_q=False
        k: [B, S_k, D] when tp_k=True, [B, D, S_k] when tp_k=False
        v: [B, S_k, D] (always)

    Output layout:
        [B, D, S_q] if tp_out=True
        [B, S_q, D] if tp_out=False
    """
    bs = q.shape[0]
    bs_kv = k.shape[0]

    # Get dimensions and prepare Q for matmul [B, S_q, D]
    if tp_q:
        q_for_matmul = q  # [B, S_q, D]
        seqlen_q, d = q.shape[1], q.shape[2]
    else:
        q_for_matmul = q.transpose(-2, -1)  # [B, D, S_q] -> [B, S_q, D]
        d, seqlen_q = q.shape[1], q.shape[2]

    # Prepare K for matmul - need [B, D, S_k] for Q @ K
    if tp_k:
        k_for_matmul = k.transpose(-2, -1)  # [B, S_k, D] -> [B, D, S_k]
        seqlen_k = k.shape[1]
    else:
        k_for_matmul = k  # [B, D, S_k]
        seqlen_k = k.shape[2]

    # Handle prefix caching - concatenate prior K/V
    if k_prior is not None:
        prior_len = int(prior_used_len.item()) if prior_used_len is not None else 0

        if tp_k:
            k_prior_slice = k_prior[:, :prior_len, :].transpose(
                -2, -1
            )  # [B, prior_len, D] -> [B, D, prior_len]
        else:
            k_prior_slice = k_prior[:, :, :prior_len]  # [B, D, prior_len]

        v_prior_slice = v_prior[:, :prior_len, :]  # [B, prior_len, D]

        k_for_matmul = torch.cat(
            [k_prior_slice, k_for_matmul], dim=-1
        )  # [B, D, prior_len + S_k]
        v = torch.cat([v_prior_slice, v], dim=1)  # [B, prior_len + S_k, D]
        seqlen_k_total = prior_len + seqlen_k
    else:
        seqlen_k_total = seqlen_k
        prior_len = 0

    # Handle GQA by expanding K/V
    if bs_kv < bs:
        repeat_factor = bs // bs_kv
        k_for_matmul = k_for_matmul.repeat_interleave(repeat_factor, dim=0)
        v = v.repeat_interleave(repeat_factor, dim=0)

    # Compute attention scores: [B, S_q, D] @ [B, D, S_k] -> [B, S_q, S_k]
    scores = torch.matmul(q_for_matmul, k_for_matmul) * scale

    # Create masks
    if causal_mask:
        # Causal mask: q_pos >= k_pos (adjusted for prefix and CP offset)
        q_pos = (
            torch.arange(seqlen_q, device=scores.device).unsqueeze(1)
            + prior_len
            + cp_offset
        )
        k_pos = torch.arange(seqlen_k_total, device=scores.device).unsqueeze(0)
        mask = q_pos < k_pos  # Upper triangle mask

        # Apply sliding window if specified
        if sliding_window > 0:
            window_mask = q_pos >= k_pos + sliding_window
            mask = mask | window_mask

        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    # Apply per-query bounds on KV range (e.g., vision VIT non-causal use case)
    # bound_min: [B, S_q, 1] inclusive lower bound on KV index
    # bound_max: [B, S_q, 1] exclusive upper bound on KV index
    if bound_min is not None or bound_max is not None:
        assert (bound_min is None) == (bound_max is None), (
            "bound_min and bound_max must both be set or both be None"
        )
        k_pos_all = (
            torch.arange(seqlen_k_total, device=scores.device).unsqueeze(0).unsqueeze(1)
        )  # [1, 1, S_k]
        scores = scores.masked_fill(
            k_pos_all < bound_min, torch.finfo(scores.dtype).min
        )
        scores = scores.masked_fill(
            k_pos_all >= bound_max, torch.finfo(scores.dtype).min
        )

    # Apply attention sink if provided
    if sink is not None:
        sink_expanded = sink.unsqueeze(1).expand(-1, seqlen_q, -1)  # [B, S_q, 1]
        scores = torch.cat([scores, sink_expanded], dim=-1)

    # Softmax along key dimension
    attn_weights = torch.softmax(scores.float(), dim=-1).type_as(q)

    # Remove sink after softmax
    if sink is not None:
        attn_weights = attn_weights[..., :-1]

    # Handle NaN from all -inf rows
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Compute output: [B, S_q, S_k] @ [B, S_k, D] -> [B, S_q, D]
    output = torch.matmul(attn_weights, v)

    # Apply output transpose if requested
    if tp_out:
        output = output.transpose(-2, -1)  # [B, S_q, D] -> [B, D, S_q]

    return output


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Optional[float] = None,
    causal_mask: bool = True,
    sink: Optional[Tensor] = None,
    sliding_window: Optional[int] = None,
    tp_q: bool = True,
    tp_k: bool = False,
    tp_out: bool = False,
    cp_offset: Optional[Tensor] = None,
    global_cp_deg: Optional[int] = None,
    cp_strided_q_slicing: bool = False,
    k_prior: Optional[Tensor] = None,
    v_prior: Optional[Tensor] = None,
    prior_used_len: Optional[Tensor] = None,
    bound_min: Optional[Tensor] = None,
    bound_max: Optional[Tensor] = None,
    allow_vector_exp: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """
    Flash Attention API using attention_cte kernel with automatic PyTorch fallback.

    This implements attention: softmax(scale * Q @ K^T + mask) @ V

    Input Layouts (controlled by transpose flags):
        q: Query tensor
           - [B, S_q, D] when tp_q=True (default)
           - [B, D, S_q] when tp_q=False
        k: Key tensor
           - [B, S_k, D] when tp_k=True
           - [B, D, S_k] when tp_k=False (default)
        v: Value tensor [B, S_k, D] (always this layout)

    Output Layout:
        - [B, D, S_q] if tp_out=True
        - [B, S_q, D] if tp_out=False (default)

    Dimensions:
        B: Batch size (can include num_heads for multi-head attention)
        S_q: Query sequence length
        S_k: Key/Value sequence length
        D: Head dimension (max 128)

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        scale: Scaling factor for attention scores. Default: 1/sqrt(d_head).
               Must be 1.0 when using sliding_window, prefix caching, or context parallel.
        causal_mask: Whether to apply causal masking. Default: True.
                     Required for sliding_window and context parallel.
        sink: Attention sink tensor [B, 1] for streaming/infinite context. Default: None
        sliding_window: Window size for local attention. None or 0 means full attention. Default: None
        tp_q: Query transpose flag. True means Q is [B, S, D]. Default: True
        tp_k: Key transpose flag. True means K is [B, S, D]. Default: False
        tp_out: Output transpose flag. True means output is [B, D, S]. Default: False
        cp_offset: Context parallel offset tensor [1, 1]. Required when global_cp_deg is set.
        global_cp_deg: Context parallel degree (1-32). Default: None (disabled)
        cp_strided_q_slicing: Use strided Q slicing for CP load balancing. Default: False
        k_prior: Prior key tensor for prefix caching (same layout as k)
        v_prior: Prior value tensor for prefix caching [B_kv, S_prior, D]
        prior_used_len: Actual used length in prior [1,]. Required with k_prior/v_prior.
        bound_min: Per-query lower bound (inclusive) of the KV range
                     to attend to, with shape (batch, seqlen_q, 1).
        bound_max: Per-query upper bound (exclusive) of the KV range
                     to attend to, with shape (batch, seqlen_q, 1).
        allow_vector_exp: Allow the Vector Engine exp on trn3. False uses the
                     Scalar Engine exp, which is more accurate. Default: True
    Returns:
        Output tensor with attention results.

    Native GQA Support:
        When batch_size_kv < batch_size (and batch_size % batch_size_kv == 0),
        the kernel handles grouped-query attention natively without K/V replication.

    Kernel Constraints (falls back to PyTorch if violated):
        - All inputs must be 3D tensors
        - Batch size: 1 to 16
        - Q batch size must be multiple of KV batch size (for GQA)
        - Head dimension D: 1 to 128
        - Sequence lengths: 1 to 32K
        - sliding_window/prefix_caching/CP require scale=1.0
        - sliding_window and CP require causal_mask=True
        - CP requires cp_offset tensor with shape [1, 1]
        - Prefix caching requires k_prior, v_prior, and prior_used_len

    Example:
        >>> # Standard attention with [B, S, D] layout
        >>> B, S, D = 8, 1024, 128
        >>> q = torch.randn(B, S, D, dtype=torch.bfloat16, device='neuron:0')
        >>> k = torch.randn(B, D, S, dtype=torch.bfloat16, device='neuron:0')  # tp_k=False default
        >>> v = torch.randn(B, S, D, dtype=torch.bfloat16, device='neuron:0')
        >>> output = flash_attention(q, k, v)  # [B, S, D]

        >>> # GQA with 8 query heads, 2 KV heads
        >>> q = torch.randn(8, S, D, dtype=torch.bfloat16, device='neuron:0')
        >>> k = torch.randn(2, D, S, dtype=torch.bfloat16, device='neuron:0')
        >>> v = torch.randn(2, S, D, dtype=torch.bfloat16, device='neuron:0')
        >>> output = flash_attention(q, k, v)  # [8, S, D]
    """
    if sliding_window is None:
        sliding_window = 0

    # Compute default scale if not provided
    if scale is None:
        d_head = v.shape[2]
        scale = 1.0 / (d_head**0.5)

    can_use_kernel = _can_use_flash_attention_kernel(v=v)

    if can_use_kernel:
        q = q * scale

        wrapped_attention_cte = wrap_nki(jitted_attention_cte)

        return wrapped_attention_cte[get_logical_nc_config()](
            q=q,
            k=k,
            v=v,
            scale=1.0,
            causal_mask=causal_mask,
            k_prior=k_prior,
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            sink=sink,
            sliding_window=sliding_window,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            cache_softmax=False,
            softmax_dtype=torch.float32,
            cp_offset=cp_offset,
            global_cp_deg=global_cp_deg,
            cp_strided_q_slicing=cp_strided_q_slicing,
            bound_min=bound_min,
            bound_max=bound_max,
            allow_vector_exp=allow_vector_exp,
        )
    else:
        return _torch_attention_impl(
            q=q,
            k=k,
            v=v,
            scale=scale,
            causal_mask=causal_mask,
            sliding_window=sliding_window,
            sink=sink,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            k_prior=k_prior,
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            bound_min=bound_min,
            bound_max=bound_max,
            cp_offset=cp_offset[0, 0] if cp_offset is not None else 0,
        )
