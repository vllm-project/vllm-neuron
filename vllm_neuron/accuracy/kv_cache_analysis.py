# SPDX-License-Identifier: Apache-2.0
"""KV Cache analysis for accuracy debugging.

Compares KV caches between an expected reference and a target (actual) model,
optionally with a baseline (ground truth) for three-way comparison. Naming
matches ``logit_validation``:

- ``expected``: reference to compare against (required)
- ``actual``: target under test
- ``baseline``: ground truth for three-way (optional); when provided, all
  errors are relative to baseline since both expected and actual are
  teacher-forced against it

Usage with logit_validation (recommended)::

    from vllm_neuron.accuracy import logit_validation, compare_kv_caches

    def kv_extract_fn(seq_len):
        paged_kv = extract_vllm_kv_caches(llm, kv_config)
        block_tables = extract_vllm_block_tables(llm)
        return reconstruct_contiguous_kv(paged_kv, kv_config, block_tables, seq_len)

    passed, results, merged_kv = logit_validation(
        generate_fn=..., expected_logits=expected_logits,
        baseline_logits=baseline_logits, input_ids=input_ids,
        kv_extract_fn=kv_extract_fn,
    )

    expected_kv = extract_hf_kv_caches_teacher_forced(expected_model, input_ids, tokens)
    result = compare_kv_caches(expected_kv, merged_kv)

    # Three-way with baseline:
    baseline_kv = extract_hf_kv_caches_teacher_forced(baseline_model, input_ids, tokens)
    result = compare_kv_caches(expected_kv, merged_kv, baseline_kv=baseline_kv)
"""

import io
import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import torch

from vllm_neuron.accuracy.utils import natural_sort_key

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from vllm import LLM

logger = logging.getLogger(__name__)


@dataclass
class HeadMetrics:
    """Per-head comparison metrics."""

    k_cos: float  # Cosine similarity (1.0 = identical)
    v_cos: float
    k_linf: float = 0.0  # Relative L-inf: max|diff| / max|ref| (ref = baseline if present, else expected)
    v_linf: float = 0.0
    k_l2: float = 0.0  # Relative L2: ||diff||_2 / ||ref||_2
    v_l2: float = 0.0
    # Three-way fields (base = expected vs baseline, tgt = actual vs baseline)
    base_k_linf: float = 0.0
    base_v_linf: float = 0.0
    base_k_l2: float = 0.0
    base_v_l2: float = 0.0


@dataclass
class TokenKVMetrics:
    """Per-token aggregate metrics across all layers and heads."""

    k_bc: float = 1.0  # BC between baseline K errors and target K errors
    v_bc: float = 1.0  # BC between baseline V errors and target V errors


@dataclass
class TokenKVErrors:
    """Raw errors for a token, used for cross-prompt aggregation."""

    base_k: np.ndarray  # [num_heads * head_dim] baseline K errors
    base_v: np.ndarray  # [num_heads * head_dim] baseline V errors
    tgt_k: np.ndarray  # [num_heads * head_dim] target K errors
    tgt_v: np.ndarray  # [num_heads * head_dim] target V errors


# Result structure: result[token_idx][layer_name][head_idx] -> HeadMetrics
# token_metrics[token_idx] -> TokenKVMetrics (BC across all heads)
KVComparisonResult = List[Dict[str, List[HeadMetrics]]]
# Raw errors: errors[token_idx][layer_name] -> TokenKVErrors
KVRawErrors = List[Dict[str, TokenKVErrors]]


# =============================================================================
# HF EXTRACTION
# =============================================================================


def generate_hf_logits_and_kv(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    num_new_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Generate logits and KV caches from HuggingFace model via autoregressive generation.

    This is the recommended way to get HF's baseline KV cache - it captures KV
    during the same autoregressive generation used for logit_validation.

    Args:
        model: HuggingFace model (AutoModelForCausalLM)
        input_ids: Input token IDs [batch_size, seq_len]
        num_new_tokens: Number of tokens to generate

    Returns:
        Tuple of:
        - logits: [num_new_tokens, batch_size, vocab_size]
        - kv_caches: Dict[layer_name, (k, v)] with shape [batch, heads, total_seq, dim]

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        >>> input_ids = tokenizer(["Hello"], return_tensors="pt")["input_ids"]
        >>> logits, kv = generate_hf_logits_and_kv(model, input_ids, num_new_tokens=16)
    """
    model.eval()
    all_logits = []
    past_kv = None
    current_input_ids = input_ids.clone()

    with torch.inference_mode():
        for step in range(num_new_tokens):
            if past_kv is None:
                outputs = model(current_input_ids, use_cache=True, return_dict=True)
            else:
                outputs = model(
                    current_input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits
            past_kv = outputs.past_key_values
            next_token_logits = logits[:, -1, :]
            all_logits.append(next_token_logits.float())

            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_input_ids = torch.cat([current_input_ids, next_tokens], dim=1)

        # One more forward to get KV for the last generated token
        outputs = model(
            current_input_ids[:, -1:],
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values

    # Convert past_key_values to our format
    kv_caches = {}
    for layer_idx, (k, v) in enumerate(past_kv):
        layer_name = f"layers.{layer_idx}.self_attn"
        kv_caches[layer_name] = (k.cpu().float(), v.cpu().float())

    return torch.stack(all_logits, dim=0), kv_caches


def extract_hf_kv_caches(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    num_new_tokens: int = 1,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract KV caches from HuggingFace model via generation.

    Note: For logit_validation integration, use generate_hf_logits_and_kv instead
    to get both logits and KV from the same generation.

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        >>> input_ids = tokenizer(["Hello"], return_tensors="pt")["input_ids"]
        >>> kv = extract_hf_kv_caches(model, input_ids, num_new_tokens=16)
    """
    _, kv_caches = generate_hf_logits_and_kv(model, input_ids, num_new_tokens)
    return kv_caches


def extract_hf_kv_caches_teacher_forced(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
    return_logits: bool = False,
) -> Union[
    Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
]:
    """
    Extract KV caches from HuggingFace model using teacher forcing.

    Processes the full sequence (prompt + teacher_tokens) in one forward pass,
    ensuring HF computes KV for the exact same tokens as vLLM.

    Args:
        model: HuggingFace model
        input_ids: Prompt tokens [batch_size, prompt_len]
        teacher_tokens: Tokens to force [num_tokens] or [batch_size, num_tokens]
        return_logits: If True, also return logits for the generated positions.

    Returns:
        If return_logits=False: Dict[layer_name, (k_cache, v_cache)]
        If return_logits=True: (logits, kv_caches) where logits is
            [num_tokens, batch_size, vocab_size]

    Example:
        >>> kv = extract_hf_kv_caches_teacher_forced(model, input_ids, tokens)
        >>> logits, kv = extract_hf_kv_caches_teacher_forced(
        ...     model, input_ids, tokens, return_logits=True)
    """
    model.eval()

    if teacher_tokens.dim() == 1:
        teacher_tokens = teacher_tokens.unsqueeze(0).expand(input_ids.shape[0], -1)

    full_sequence = torch.cat([input_ids, teacher_tokens], dim=1)

    with torch.inference_mode():
        outputs = model(full_sequence, use_cache=True, return_dict=True)

    kv_caches = {}
    for layer_idx, (k, v, *_) in enumerate(outputs.past_key_values):
        kv_caches[f"layers.{layer_idx}.self_attn"] = (k.cpu().float(), v.cpu().float())

    if return_logits:
        prompt_len = input_ids.shape[1]
        logits = outputs.logits[:, prompt_len - 1 : -1, :].float().permute(1, 0, 2)
        return logits, kv_caches
    return kv_caches


# =============================================================================
# VLLM EXTRACTION (via collective_rpc)
# =============================================================================


def extract_vllm_kv_caches(
    llm: "LLM",
    kv_cache_config: dict = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract KV caches from vLLM LLM instance.

    Calls get_kv_caches() on each TP worker and reconstructs full KV by
    concatenating shards (or taking rank 0 if replicated). Uses
    total_num_kv_heads vs tp_size to determine sharding mode.

    Args:
        llm: vLLM LLM instance
        kv_cache_config: Config from extract_vllm_kv_cache_config().
            Used to determine TP sharding mode.

    Returns:
        Dict[layer_name, (k_cache, v_cache)]
        Shape: [num_blocks, total_kv_heads, block_size, head_dim]

    Example:
        >>> cfg = extract_vllm_kv_cache_config(llm)
        >>> kv = extract_vllm_kv_caches(llm, cfg)
    """
    per_rank_kv = llm.collective_rpc("get_kv_caches")
    total_num_kv_heads = (
        kv_cache_config.get("total_num_kv_heads") if kv_cache_config else None
    )
    tp_size = kv_cache_config.get("tp_size", 1) if kv_cache_config else 1
    return _reconstruct_from_tp_shards(per_rank_kv, total_num_kv_heads, tp_size)


def extract_vllm_kv_cache_config(llm: "LLM") -> dict:
    """
    Extract KVCacheConfig from vLLM.

    Returns:
        dict with:
        - groups: List of {layer_names, block_size} per KV cache group
        - tp_size: Tensor parallel size
        - total_num_kv_heads: Total KV heads (for TP sharding mode)
        - dcp_size, pcp_size: Context parallel sizes (must be 1)

    Example:
        >>> cfg = extract_vllm_kv_cache_config(llm)
        >>> cfg['tp_size']
        32
    """
    results = llm.collective_rpc("get_kv_cache_config")
    config = results[0] if results else {}

    # Validate CP is not used (not yet supported)
    dcp_size = config.get("dcp_size", 1)
    pcp_size = config.get("pcp_size", 1)
    assert dcp_size == 1 and pcp_size == 1, (
        f"Context parallelism not supported for KV cache analysis. "
        f"Got dcp_size={dcp_size}, pcp_size={pcp_size}."
    )

    return config


def extract_vllm_block_tables(llm: "LLM") -> List[torch.Tensor]:
    """
    Extract block tables from vLLM (one per KV cache group).

    Automatically enables block table snapshotting on first call so that
    subsequent forward passes preserve block tables after request freeing.

    Returns:
        List of tensors, each [batch_size, max_blocks_per_seq]

    Example:
        >>> tables = extract_vllm_block_tables(llm)
    """
    results = llm.collective_rpc("get_block_tables")
    if not results or not results[0]:
        return []

    tables = []
    for data in results[0]:
        with io.BytesIO(data) as buf:
            tables.append(torch.load(buf, weights_only=True))
    return tables


def enable_kv_snapshot(llm: "LLM") -> None:
    """Enable block table snapshotting on all workers.

    Must be called before the generate() whose KV caches you want to
    extract.  Snapshotting preserves block tables during the forward pass
    so they survive request freeing.

    Example:
        >>> enable_kv_snapshot(llm)
        >>> llm.generate(...)
        >>> tables = extract_vllm_block_tables(llm)
    """
    llm.collective_rpc("enable_kv_snapshot")


def cleanup_kv_snapshot(llm: "LLM") -> None:
    """Release block table snapshot memory on all workers.

    Call after KV cache analysis is complete to free memory and restore
    zero-overhead production behavior.

    Example:
        >>> cleanup_kv_snapshot(llm)
    """
    llm.collective_rpc("clear_kv_snapshot")


# =============================================================================
# RECONSTRUCTION
# =============================================================================


def reconstruct_contiguous_kv(
    paged_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_config: dict,
    block_tables: List[torch.Tensor],
    seq_len: int,
    seq_idx: int = 0,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Reconstruct contiguous KV from paged format using KVCacheConfig.

    Model-agnostic: uses KVCacheConfig to determine which layers share
    block tables, their block sizes, and attention type (full vs sliding window).
    For sliding window layers, positions outside the window are filled with NaN.

    Args:
        paged_kv: Dict[layer_name, (k, v)] in paged format
            Shape: [num_blocks, num_kv_heads, block_size, head_dim]
        kv_cache_config: Dict with 'groups' list from extract_vllm_kv_cache_config()
        block_tables: List of block tables, one per KV cache group
        seq_len: Actual sequence length to reconstruct
        seq_idx: Which sequence in the batch (default 0)

    Returns:
        Dict[layer_name, (k, v)] in contiguous format
        Shape: [1, num_kv_heads, seq_len, head_dim]
        Positions outside sliding window are NaN.
    """
    groups = kv_cache_config.get("groups", [])

    # Build layer → group index mapping
    layer_to_group = {
        layer: i for i, g in enumerate(groups) for layer in g["layer_names"]
    }

    result = {}
    for layer_name, (k_paged, v_paged) in paged_kv.items():
        group_idx = layer_to_group.get(layer_name, 0)
        group = groups[group_idx] if groups else {}
        block_size = group.get("block_size", 16)
        if (
            group_idx >= len(block_tables)
            or seq_idx >= block_tables[group_idx].shape[0]
        ):
            raise IndexError(
                f"Block table index out of range for {layer_name}: "
                f"group_idx={group_idx}/{len(block_tables)}, seq_idx={seq_idx}"
            )
        block_table = block_tables[group_idx][seq_idx]
        sliding_window = group.get("sliding_window", None)
        spec_type = group.get("spec_type", "")

        num_heads, head_dim = k_paged.shape[1], k_paged.shape[3]

        # Only apply sliding window truncation when blocks are actually freed
        # (SlidingWindowSpec). FullAttentionSpec with sliding_window set means
        # hybrid allocator is disabled — blocks are kept, KV is valid everywhere.
        blocks_freed = (
            spec_type == "SlidingWindowSpec"
            and sliding_window is not None
            and sliding_window > 0
        )
        if blocks_freed:
            valid_start = max(0, seq_len - sliding_window)
        else:
            valid_start = 0

        # Reconstruct only valid positions
        valid_len = seq_len - valid_start
        valid_start_block = valid_start // block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        valid_num_blocks = num_blocks - valid_start_block

        indices = block_table[valid_start_block:num_blocks].long()
        k_blocks = k_paged[indices]
        v_blocks = v_paged[indices]

        k_valid = k_blocks.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)
        v_valid = v_blocks.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)

        # Trim: the first block may contain positions before valid_start
        trim_start = valid_start - valid_start_block * block_size
        trim_end = trim_start + valid_len
        k_valid = k_valid[:, trim_start:trim_end, :]
        v_valid = v_valid[:, trim_start:trim_end, :]

        # Build full-length output, NaN for positions outside window
        k_cont = torch.full(
            (num_heads, seq_len, head_dim), float("nan"), dtype=k_valid.dtype
        )
        v_cont = torch.full(
            (num_heads, seq_len, head_dim), float("nan"), dtype=v_valid.dtype
        )
        k_cont[:, valid_start:seq_len, :] = k_valid
        v_cont[:, valid_start:seq_len, :] = v_valid

        result[layer_name] = (k_cont.unsqueeze(0), v_cont.unsqueeze(0))

    return result


def reconstruct_contiguous_kv_batch(
    paged_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    kv_cache_config: dict,
    block_tables: List[torch.Tensor],
    seq_len: int,
    batch_size: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Reconstruct contiguous KV for all sequences in a batch.

    Returns:
        Dict[layer_name, (k, v)] with shape [batch_size, num_kv_heads, seq_len, head_dim]
    """
    batch_kvs = [
        reconstruct_contiguous_kv(
            paged_kv, kv_cache_config, block_tables, seq_len, seq_idx
        )
        for seq_idx in range(batch_size)
    ]

    result = {}
    layer_names = list(paged_kv.keys())
    for layer_name in layer_names:
        k_stack = torch.cat([bkv[layer_name][0] for bkv in batch_kvs], dim=0)
        v_stack = torch.cat([bkv[layer_name][1] for bkv in batch_kvs], dim=0)
        result[layer_name] = (k_stack, v_stack)

    return result


def _reconstruct_from_tp_shards(
    per_rank_kv: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    total_num_kv_heads: int = None,
    tp_size: int = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Reconstruct full KV from TP-sharded paged caches.

    Each rank stores shape [num_blocks, heads_per_rank, block_size, head_dim].
    Concat on dim=1 (head) across selected ranks to get total_num_kv_heads.

    Sharding mode (from vLLM's ``max(1, total // tp)`` convention):
    - total_num_kv_heads >= tp_size: partitioned, each rank unique → concat all.
    - total_num_kv_heads < tp_size: replicated, stride by ``tp // total`` to
      pick one rank per replica group, then concat.

    Args:
        per_rank_kv: Per-rank KV dicts from ``collective_rpc("get_kv_caches")``.
        total_num_kv_heads: Model-level total unique KV heads
            (from ``model_config.get_total_num_kv_heads()``).
        tp_size: Tensor parallel size. Defaults to ``len(per_rank_kv)``.

    Returns:
        Dict[layer_name, (k, v)] still in paged layout, heads merged.

    Example:
        >>> cfg = extract_vllm_kv_cache_config(llm)
        >>> per_rank = llm.collective_rpc("get_kv_caches")
        >>> full = _reconstruct_from_tp_shards(per_rank, cfg['total_num_kv_heads'], cfg['tp_size'])
    """

    def load_kv(data):
        if isinstance(data, bytes):
            with io.BytesIO(data) as buf:
                d = torch.load(buf, weights_only=True)
            return d["k"], d["v"]
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            k, v = data[0], data[1]
            if isinstance(k, np.ndarray):
                return torch.from_numpy(k), torch.from_numpy(v)
            return k, v
        raise ValueError(f"Unknown KV format: {type(data)}")

    if tp_size is None:
        tp_size = len(per_rank_kv)

    if tp_size == 1:
        return {name: load_kv(data) for name, data in per_rank_kv[0].items()}

    if total_num_kv_heads is not None and total_num_kv_heads < tp_size:
        stride = tp_size // total_num_kv_heads
        rank_indices = list(range(0, tp_size, stride))
    else:
        rank_indices = list(range(tp_size))

    reconstructed = {}
    for layer_name in per_rank_kv[0].keys():
        k_shards, v_shards = [], []
        for r in rank_indices:
            k, v = load_kv(per_rank_kv[r][layer_name])
            k_shards.append(k)
            v_shards.append(v)
        reconstructed[layer_name] = (
            torch.cat(k_shards, dim=1),
            torch.cat(v_shards, dim=1),
        )

    return reconstructed


# Aliases for backward compatibility
reconstruct_from_sharded = _reconstruct_from_tp_shards


# =============================================================================
# COMPARISON
# =============================================================================


def _compute_bc(errors1: np.ndarray, errors2: np.ndarray, num_bins: int = 100) -> float:
    """Bhattacharyya Coefficient between two error distributions.

    BC=1.0 means identical distributions, BC=0.0 means no overlap.
    """
    if len(errors1) == 0 or len(errors2) == 0:
        return 0.0
    min_val = min(errors1.min(), errors2.min())
    max_val = max(errors1.max(), errors2.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)
    h1, _ = np.histogram(errors1, bins=bins)
    h2, _ = np.histogram(errors2, bins=bins)
    p1 = h1 / (h1.sum() + 1e-10)
    p2 = h2 / (h2.sum() + 1e-10)
    return float(np.clip(np.sum(np.sqrt(p1 * p2)), 0.0, 1.0))


def compare_kv_caches(
    expected_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    actual_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    layers: List[str] = None,
    baseline_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
    return_raw_errors: bool = False,
) -> Union[KVComparisonResult, Tuple[KVComparisonResult, KVRawErrors]]:
    """
    Compare KV caches: token → layer → head.

    Two-way (no baseline): compares actual vs expected.
    Three-way (with baseline): errors are measured against baseline (the
    ground truth used for teacher forcing), and BC measures whether actual's
    error distribution matches expected's.

    Naming matches ``logit_validation``:
    - ``expected_kv``: reference to compare against (required)
    - ``actual_kv``: target under test
    - ``baseline_kv``: ground truth for three-way (optional); when provided,
      all L-inf/L2 errors are relative to baseline, not expected

    All inputs must have the same sequence length per layer. Shape mismatches
    raise ``ValueError`` immediately.

    Args:
        expected_kv: Reference KV caches ``[batch, heads, seq, dim]``.
        actual_kv: Target KV caches ``[batch, heads, seq, dim]``.
        layers: Layers to compare (default: all common layers).
        baseline_kv: Optional ground-truth KV for three-way BC analysis.
        return_raw_errors: If True, also return raw errors for cross-prompt aggregation.

    Returns:
        ``List[Dict[layer, List[HeadMetrics]]]`` indexed by ``[token][layer][head]``.
        If return_raw_errors=True, also returns ``List[Dict[layer, TokenKVErrors]]``.

    Example:
        >>> result = compare_kv_caches(expected_kv, actual_kv)
        >>> result = compare_kv_caches(expected_kv, actual_kv, baseline_kv=fp32_kv)
        >>> result[0]['layers.1.self_attn'][0].k_linf  # token 0, layer 1, head 0
    """
    # When baseline is provided, all errors are relative to it (teacher-forcing ref).
    # Otherwise, expected is the reference.
    has_baseline = baseline_kv is not None
    ref_kv = baseline_kv if has_baseline else expected_kv

    if layers is None:
        all_keys = set(expected_kv.keys()) & set(actual_kv.keys())
        if has_baseline:
            all_keys &= set(baseline_kv.keys())
        layers = sorted(all_keys, key=natural_sort_key)

    # Validate shapes
    for layer in layers:
        r_seq = ref_kv[layer][0].shape[2]
        a_seq = actual_kv[layer][0].shape[2]
        if r_seq != a_seq:
            raise ValueError(
                f"Shape mismatch at {layer}: reference seq_len={r_seq}, "
                f"actual seq_len={a_seq}."
            )
        if has_baseline:
            e_seq = expected_kv[layer][0].shape[2]
            if e_seq != r_seq:
                raise ValueError(
                    f"Shape mismatch at {layer}: baseline seq_len={r_seq}, "
                    f"expected seq_len={e_seq}."
                )

    first = layers[0]
    seq_len = ref_kv[first][0].shape[2] - 1

    result = []
    raw_errors = [] if return_raw_errors else None

    for t in range(seq_len):
        token_data = {}
        token_errors = {} if return_raw_errors else None

        for layer in layers:
            r_k, r_v = ref_kv[layer]
            a_k, a_v = actual_kv[layer]
            num_heads = r_k.shape[1]

            head_metrics = []
            base_k_abs_errs = []
            tgt_k_abs_errs = []
            base_v_abs_errs = []
            tgt_v_abs_errs = []
            for h in range(num_heads):
                rk = r_k[0, h, t, :].float()
                rv = r_v[0, h, t, :].float()
                ak = a_k[0, h, t, :].float()
                av = a_v[0, h, t, :].float()

                rk_max = rk.abs().max().clamp(min=1e-10)
                rv_max = rv.abs().max().clamp(min=1e-10)
                rk_norm = rk.norm().clamp(min=1e-10)
                rv_norm = rv.norm().clamp(min=1e-10)

                k_cos = torch.nn.functional.cosine_similarity(
                    rk.unsqueeze(0), ak.unsqueeze(0)
                ).item()
                v_cos = torch.nn.functional.cosine_similarity(
                    rv.unsqueeze(0), av.unsqueeze(0)
                ).item()
                k_linf = ((rk - ak).abs().max() / rk_max).item()
                v_linf = ((rv - av).abs().max() / rv_max).item()
                k_l2 = ((rk - ak).norm() / rk_norm).item()
                v_l2 = ((rv - av).norm() / rv_norm).item()

                tgt_k_abs_errs.append((rk - ak).abs().numpy())
                tgt_v_abs_errs.append((rv - av).abs().numpy())

                # Three-way: expected vs baseline errors
                base_k_linf = base_v_linf = base_k_l2 = base_v_l2 = 0.0
                if has_baseline:
                    ek = expected_kv[layer][0][0, h, t, :].float()
                    ev = expected_kv[layer][1][0, h, t, :].float()
                    base_k_linf = ((rk - ek).abs().max() / rk_max).item()
                    base_v_linf = ((rv - ev).abs().max() / rv_max).item()
                    base_k_l2 = ((rk - ek).norm() / rk_norm).item()
                    base_v_l2 = ((rv - ev).norm() / rv_norm).item()
                    base_k_abs_errs.append((rk - ek).abs().numpy())
                    base_v_abs_errs.append((rv - ev).abs().numpy())

                head_metrics.append(
                    HeadMetrics(
                        k_cos=k_cos,
                        v_cos=v_cos,
                        k_linf=k_linf,
                        v_linf=v_linf,
                        k_l2=k_l2,
                        v_l2=v_l2,
                        base_k_linf=base_k_linf,
                        base_v_linf=base_v_linf,
                        base_k_l2=base_k_l2,
                        base_v_l2=base_v_l2,
                    )
                )

            # BC across heads for this layer
            if has_baseline and base_k_abs_errs:
                k_bc = _compute_bc(
                    np.concatenate(base_k_abs_errs), np.concatenate(tgt_k_abs_errs)
                )
                v_bc = _compute_bc(
                    np.concatenate(base_v_abs_errs), np.concatenate(tgt_v_abs_errs)
                )
                token_data[f"{layer}._bc"] = TokenKVMetrics(k_bc=k_bc, v_bc=v_bc)

                if return_raw_errors:
                    token_errors[layer] = TokenKVErrors(
                        base_k=np.concatenate(base_k_abs_errs),
                        base_v=np.concatenate(base_v_abs_errs),
                        tgt_k=np.concatenate(tgt_k_abs_errs),
                        tgt_v=np.concatenate(tgt_v_abs_errs),
                    )

            token_data[layer] = head_metrics

        result.append(token_data)
        if return_raw_errors:
            raw_errors.append(token_errors)

    if return_raw_errors:
        return result, raw_errors
    return result


def aggregate_kv_bc_across_prompts(
    all_raw_errors: List[KVRawErrors],
    prompt_lens: List[int],
    layers: List[str] = None,
) -> Dict[int, Dict[str, TokenKVMetrics]]:
    """
    Aggregate KV errors across prompts and compute BC per generation token.

    Aligns by generation token index (0 = first decode token) so prompts
    with different lengths are properly aligned.

    Args:
        all_raw_errors: List of raw errors from each prompt (from compare_kv_caches)
        prompt_lens: List of prompt lengths for each prompt (to find decode start)
        layers: Layers to aggregate (default: all)

    Returns:
        Dict[gen_token_idx, Dict[layer_name, TokenKVMetrics]] with aggregated BC
    """
    if not all_raw_errors or not prompt_lens:
        return {}

    assert len(all_raw_errors) == len(prompt_lens)

    # Find number of generation tokens (decode tokens) per prompt
    gen_token_counts = [
        len(errs) - prompt_len for errs, prompt_len in zip(all_raw_errors, prompt_lens)
    ]
    n_gen_tokens = min(gen_token_counts)
    if n_gen_tokens <= 0:
        return {}

    if layers is None:
        layers = sorted(all_raw_errors[0][0].keys(), key=natural_sort_key)

    result = {}
    for gen_t in range(n_gen_tokens):
        token_bc = {}
        for layer in layers:
            # Collect errors from all prompts at this generation token
            base_k_all, base_v_all = [], []
            tgt_k_all, tgt_v_all = [], []

            for prompt_idx, prompt_errors in enumerate(all_raw_errors):
                # Map generation token to absolute position
                abs_t = prompt_lens[prompt_idx] + gen_t
                if abs_t < len(prompt_errors) and layer in prompt_errors[abs_t]:
                    errs = prompt_errors[abs_t][layer]
                    base_k_all.append(errs.base_k)
                    base_v_all.append(errs.base_v)
                    tgt_k_all.append(errs.tgt_k)
                    tgt_v_all.append(errs.tgt_v)

            if base_k_all and tgt_k_all:
                k_bc = _compute_bc(
                    np.concatenate(base_k_all), np.concatenate(tgt_k_all)
                )
                v_bc = _compute_bc(
                    np.concatenate(base_v_all), np.concatenate(tgt_v_all)
                )
                token_bc[layer] = TokenKVMetrics(k_bc=k_bc, v_bc=v_bc)

        result[gen_t] = token_bc

    return result


def compute_combined_bc_per_layer(
    all_raw_errors: List[KVRawErrors],
    layers: List[str] = None,
) -> Dict[str, TokenKVMetrics]:
    """
    Compute a single combined BC per layer from all raw errors.

    Concatenates all errors across all prompts and all tokens for each layer,
    then computes one BC value.

    Args:
        all_raw_errors: List of raw errors from each prompt
        layers: Layers to compute (default: all)

    Returns:
        Dict[layer_name, TokenKVMetrics] with combined BC per layer
    """
    if not all_raw_errors:
        return {}

    if layers is None:
        layers = sorted(all_raw_errors[0][0].keys(), key=natural_sort_key)

    result = {}
    for layer in layers:
        base_k_all, base_v_all = [], []
        tgt_k_all, tgt_v_all = [], []

        for prompt_errors in all_raw_errors:
            for token_errors in prompt_errors:
                if layer in token_errors:
                    errs = token_errors[layer]
                    base_k_all.append(errs.base_k)
                    base_v_all.append(errs.base_v)
                    tgt_k_all.append(errs.tgt_k)
                    tgt_v_all.append(errs.tgt_v)

        if base_k_all and tgt_k_all:
            k_bc = _compute_bc(np.concatenate(base_k_all), np.concatenate(tgt_k_all))
            v_bc = _compute_bc(np.concatenate(base_v_all), np.concatenate(tgt_v_all))
            result[layer] = TokenKVMetrics(k_bc=k_bc, v_bc=v_bc)

    return result


def print_kv_report(
    result: KVComparisonResult, layers: List[str] = None, max_tokens: int = 10
) -> None:
    """Print per-token, per-layer, per-head comparison.

    Example:
        >>> result = compare_kv_caches(expected_kv, actual_kv)
        >>> print_kv_report(result, max_tokens=5)
    """
    if not result:
        print("No results")
        return

    if layers is None:
        layers = sorted(
            (k for k in result[0].keys() if not k.endswith("._bc")),
            key=natural_sort_key,
        )

    for t, token_data in enumerate(result[:max_tokens]):
        print(f"\n=== Token {t} ===")
        for layer in layers:
            if layer not in token_data:
                continue
            heads = token_data[layer]
            k_cos = [f"{h.k_cos:.3f}" for h in heads]
            v_cos = [f"{h.v_cos:.3f}" for h in heads]
            k_linf = [f"{h.k_linf:.2e}" for h in heads]
            v_linf = [f"{h.v_linf:.2e}" for h in heads]
            k_l2 = [f"{h.k_l2:.2e}" for h in heads]
            v_l2 = [f"{h.v_l2:.2e}" for h in heads]
            print(f"  {layer}:")
            print(
                f"    K cos: [{', '.join(k_cos)}]  L-inf: [{', '.join(k_linf)}]  L2: [{', '.join(k_l2)}]"
            )
            print(
                f"    V cos: [{', '.join(v_cos)}]  L-inf: [{', '.join(v_linf)}]  L2: [{', '.join(v_l2)}]"
            )
            if any(h.base_k_linf > 0 for h in heads):
                bk_linf = [f"{h.base_k_linf:.2e}" for h in heads]
                bv_linf = [f"{h.base_v_linf:.2e}" for h in heads]
                print(
                    f"    Base K L-inf: [{', '.join(bk_linf)}]  Base V L-inf: [{', '.join(bv_linf)}]"
                )
            bc_key = f"{layer}._bc"
            if bc_key in token_data:
                bc = token_data[bc_key]
                print(f"    BC: K={bc.k_bc:.4f}  V={bc.v_bc:.4f}")
