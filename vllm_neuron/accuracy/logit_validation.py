# SPDX-License-Identifier: Apache-2.0
import logging
import math
import os
import collections
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Union, Tuple, List, Optional, Callable, Any, Dict

import torch

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm import LLM

# Enable deterministic algorithms
torch.use_deterministic_algorithms(mode=True, warn_only=True)

import numpy as np

# Import types and utilities
from .constants import DEFAULT_TOLERANCE_MAP, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
from .logit_visualization import visualize_logit_results
from .types import MultiPromptValidationResult

logger = logging.getLogger(__name__)


@dataclass
class _AllcloseResult:
    allclose: bool
    max_rel_error: float


def _cpu_allclose(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8
) -> _AllcloseResult:
    # CPU fallback implementation for neuron_allclose
    abs_diff = torch.abs(actual - expected)
    expected_abs_max = torch.max(torch.abs(expected))
    if torch.is_nonzero(expected_abs_max):
        rel_error = (abs_diff - atol) / expected_abs_max
    else:
        rel_error = torch.full(expected.shape, float("inf"))
    max_rel_error = torch.max(rel_error).item()
    return _AllcloseResult(max_rel_error <= rtol, max_rel_error)


# Default dynamic threshold config for three-way validation
DEFAULT_DYNAMIC_THRESHOLD_CONFIG = {
    "linf_multiplier": 1.5,  # tgt_linf < N * base_linf
    "l2_multiplier": 1.5,  # tgt_l2 < N * base_l2
}

# Default aggregate threshold config for multi-prompt validation
DEFAULT_AGGREGATE_CONFIG = {
    "pp_static_thresholds": [0.03, 0.05],  # max tgt_linf < threshold per prompt
    "pp_linf_multipliers": [1.5, 2.0],  # max tgt_linf < N * max base_linf per prompt
    "pp_l2_multipliers": [1.5, 2.0],  # max tgt_l2 < N * max base_l2 per prompt
    "pp_tok_linf_multipliers": [1.5, 2.0],  # all tokens: tgt_linf < N * base_linf
    "pp_tok_l2_multipliers": [1.5, 2.0],  # all tokens: tgt_l2 < N * base_l2
    "agg_bc_threshold": 0.99,  # BC > threshold for all tokens
    "agg_linf_multipliers": [1.2, 1.5],  # per-token: max_tgt_linf < N * max_base_linf
    "agg_l2_multipliers": [1.2, 1.5],  # per-token: max_tgt_l2 < N * max_base_l2
    "agg_sigma_ratio_threshold": 1.0,  # σ-ratio ≤ threshold passes
}


@dataclass
class ThreeWayTokenMetrics:
    """Metrics for three-way comparison at a single token position.

    Three-way validation compares target (Neuron) against a same-dtype baseline (e.g., BF16),
    using FP32 as the reference. This isolates target-specific errors from dtype-inherent errors.

    Attributes:
        base_linf: L-inf error (max|diff|/max|ref|) between expected and FP32
        tgt_linf: L-inf error between actual (target) and FP32
        base_l2: L2 error (norm(diff)/norm(ref)) between expected and FP32
        tgt_l2: L2 error between actual (target) and FP32
        bc: Bhattacharyya Coefficient - error distribution similarity (1.0 = identical)
        base_errors: Raw absolute errors (expected vs FP32) for cross-prompt BC aggregation
        tgt_errors: Raw absolute errors (actual vs FP32) for cross-prompt BC aggregation
    """

    base_linf: float = 0.0
    tgt_linf: float = 0.0
    base_l2: float = 0.0
    tgt_l2: float = 0.0
    bc: float = 1.0
    base_errors: np.ndarray = None
    tgt_errors: np.ndarray = None


# Summary formatting constants
_SUMMARY_WIDTH = 80
_ERROR_PRECISION = 4


class _Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class _ValidationStatus(Enum):
    """Enum representing different validation status classifications for logit comparison."""

    MATCHED = "matched"
    TOPK_ERRORS = "topk_errors"
    ACCEPTABLE_DIVERGENCE = "acceptable_divergence"
    ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS = "acceptable_divergence_with_topk_errors"
    DIVERGED = "diverged"
    DIVERGED_WITH_TOPK_ERRORS = "diverged_with_topk_errors"

    def get_display_text(self) -> str:
        """Returns the display text for this validation status."""
        display_map = {
            _ValidationStatus.MATCHED: "✓ Matched",
            _ValidationStatus.TOPK_ERRORS: "✗ TopK errors",
            _ValidationStatus.ACCEPTABLE_DIVERGENCE: "✓ Acceptable divergence",
            _ValidationStatus.ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS: "✗ Acceptable divergence + TopK errors",
            _ValidationStatus.DIVERGED: "✗ Diverged",
            _ValidationStatus.DIVERGED_WITH_TOPK_ERRORS: "✗ Diverged + TopK errors",
        }
        return display_map[self]

    def get_color(self) -> str:
        """Returns the appropriate color for this validation status."""
        if self == _ValidationStatus.MATCHED:
            return "GREEN"
        elif self == _ValidationStatus.ACCEPTABLE_DIVERGENCE:
            return "YELLOW"
        else:  # All error states
            return "RED"

    def is_passing(self) -> bool:
        """Returns True if this status represents a passing validation."""
        return self in (
            _ValidationStatus.MATCHED,
            _ValidationStatus.ACCEPTABLE_DIVERGENCE,
        )


def logit_validation(
    input_ids: List[List[int]],
    generate_fn: Callable[
        [torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ],
    expected_logits: torch.Tensor,
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    suppress_passing: bool = True,
    colorize: bool = True,
    baseline_logits: torch.Tensor = None,
    dynamic_threshold_config: dict = None,
    label: str = None,
    test_device: str = "cpu",
    # Enhanced validation parameters
    output_dir: str = "validation_output",
    visualize: bool = False,
    save_logits: bool = True,
    # ULP-aware divergence tolerance
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
    # Multimodal inputs
    multimodal_inputs: Optional[List[dict]] = None,
    # KV cache capture
    kv_extract_fn: Callable[[int], Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Union[
    bool,
    Tuple[bool, List[List[dict]]],
    Tuple[bool, List[List[dict]], Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
]:
    """
    Validates model accuracy by comparing raw prediction scores (logits) against reference values.

    This function performs comprehensive logit-based validation designed specifically for
    hardware-specific model accuracy testing. Unlike token-based validation that only compares
    final predictions, this approach compares the raw prediction scores at each position,
    providing deeper insights into numerical precision differences and model behavior across
    different hardware platforms (CPU, GPU, Neuron).

    What is Logit Matching?
    -----------------------
    Logit matching compares the raw prediction scores (logits) produced by language
    models before any sampling or argmax operations are applied. These logits represent the
    model's confidence in each possible next token as unnormalized log probabilities.

    **Logit Matching vs Token Matching:**

    - **Token Matching**: Only compares the final sampled tokens after argmax operation
        - Example: Comparing "Paris." vs "Paris," as final predictions
        - Limited insight: Only shows the top choice differs (both are correct)
        - Misses subtle differences: Can't detect that logits were [Paris.:15.252, Paris,:15.250] vs [Paris.:15.250, Paris,:15.252]

    - **Logit Matching**: Compares the full probability distribution before sampling
        - Example: Reference [',': 15.250, '.': 15.252, ' and': 12.1] vs Neuron [',': 15.252, '.': 15.250, ' and': 12.0]
        - Rich insight: Shows confidence levels and ranking of all possibilities
        - Detects drift: Can identify when tiny BF16 differences affect token rankings

    **Why Logit Matching is Superior:**

    1. **Captures Full Distribution**: Reveals the model's uncertainty and confidence levels
    2. **Detects Subtle Drift**: Identifies numerical differences before they affect final predictions
    3. **Hardware Validation**: Essential for validating model behavior across different hardware
    4. **Debugging Power**: Helps isolate where models diverge

    Hardware Precision Challenges:
    -----------------------------
    Different hardware platforms compute floating-point math slightly differently due to:

    - **Operation Ordering**: Different platforms may reorder operations for optimization
    - **Parallel Reduction**: Various strategies for summing/reducing across parallel units
    - **Accumulation Effects**: In models with billions of operations, tiny differences
      accumulate through layers, causing numerical drift in final logits

    Teacher Forcing for Fair Comparison:
    -----------------------------------
    To isolate hardware-specific differences at each position, this validation uses
    teacher forcing: even when the model under test would sample a different token
    than the reference, we force it to use the reference token as input for the next position.

    **Without Teacher Forcing:**
    ```
    Given the prompt "The capital of France is", both Reference and Neuron correctly output "Paris".
    However, at the next token position, tiny BF16 precision differences emerge:

    Reference logits: ','=15.250, '.'=15.252 → selects '.' (greedy)
    Neuron logits: ','=15.252, '.'=15.250 → selects ',' (greedy)

    Reference: "The capital of France is Paris. The city has been..."
    Neuron: "The capital of France is Paris, which is known for..."

    → Sequences diverge completely after position 1, remaining logits incomparable
    → Can't determine if position 2+ errors are due to accumulated context differences or precision drift
    ```

    **With Teacher Forcing:**
    ```
    Position 1: Reference wants '.', Neuron wants ',', but we force both to use '.'
    → Both models continue with identical context: "The capital of France is Paris."
    → Can compare logits at positions 2, 3, 4... with same input context
    → Isolates numerical precision drift without cascading context effects
    ```

    **Benefits of Teacher Forcing:**

    1. **Fair Comparison**: Ensures both models process identical context at each step
    2. **Cascading Prevention**: Stops early errors from contaminating later positions
    3. **Comprehensive Coverage**: Validates the entire sequence length, not just until first divergence

    Args:
        input_ids: List of input token sequences for each batch.
            Each inner list represents a sequence of token IDs that serve as the
            initial context for generation.

        generate_fn: Function that takes input_ids as a tensor and returns either:
            - torch.Tensor: Logits tensor of shape (seq_len, batch_size, vocab_size)
            - Tuple[torch.Tensor, torch.Tensor]: (logits, sequences) where sequences
              are the sampled token IDs of shape (batch_size, seq_len)
            This function represents the model under test (e.g., Neuron-compiled model).

        expected_logits: Reference logits tensor of shape
            (seq_len, batch_size, vocab_size) to validate against. These typically
            come from a reference implementation (e.g., CPU/GPU model).

        tol_map: Dictionary mapping top-k values to (atol, rtol) tolerance
            tuples. Keys can be strings representing top-k values or "all" (all tokens).
            Defaults to DEFAULT_TOLERANCE_MAP if None.
            Example: {"all": (1e-5, 0.05), "50": (1e-5, 0.02), "5": (1e-5, 0.01)}

        divergence_difference_tol: Fixed tolerance for divergence difference,
            used only when divergence_n_ulps is None. Measures how much the logit
            for the expected token differs from the maximum logit in the actual output.
            Defaults to DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE (0.001) if None.

        divergence_n_ulps: Number of ULPs (Units in the Last Place) to allow
            when checking divergence difference. When set, the tolerance is
            computed dynamically as ``n_ulps * ULP(max(|logit_a|, |logit_b|))``
            based on the target dtype's precision, replacing the fixed
            divergence_difference_tol. Defaults to 1 (one BF16 step).
            Set to None to fall back to divergence_difference_tol.

        mantissa_bits: Number of mantissa bits in the target dtype for ULP
            calculation. BF16 = 7, FP16 = 10, FP32 = 23. Defaults to 7 (BF16).

        suppress_passing: If True, suppresses display of tokens with matched status
            in the validation results, showing only tokens that require attention
            (errors, divergences, etc.).

        colorize: If True, applies ANSI color codes to the validation output for
            enhanced readability in terminals that support color. If False, outputs
            plain text without color formatting. Defaults to True.

        baseline_logits: Optional FP32 reference logits for three-way comparison.
            When provided, enables three-way validation mode:
            - baseline_logits: FP32 reference (used for teacher forcing)
            - expected_logits: same-dtype baseline (e.g., BF16 on CPU/GPU)
            - actual_logits: target (Neuron, same dtype as expected)
            Teacher forcing uses baseline_logits.argmax() to extend input_ids.
            Shape must match expected_logits: (seq_len, batch_size, vocab_size)

        dynamic_threshold_config: Configuration for dynamic thresholds in three-way mode.
            Keys: "linf_multiplier", "l2_multiplier" (default: 1.5 each)
            Pass criteria: tgt_error < multiplier * base_error

        test_device: Device to run validation on. Either "cpu" or "neuron".
            When "cpu", uses a CPU-based allclose implementation. When "neuron",
            uses neuron_allclose from libtorch_neuronx_lite. Defaults to "cpu".

        output_dir: Directory for visualization outputs.
        visualize: Whether to generate plots and save logits.
        save_logits: Whether to save logit tensors to files.

        multimodal_inputs: Optional list of multimodal input dictionaries, one per
            batch element. Each dict contains modality-specific data (e.g., pixel
            values for images, audio features) that ``generate_fn`` needs alongside
            the token IDs. When provided, ``generate_fn`` is called as
            ``generate_fn(input_ids, multimodal_inputs=multimodal_inputs)``.
            Defaults to None (text-only).

        kv_extract_fn: Optional callback to extract KV caches after each generate step.
            Called with ``kv_extract_fn(seq_len)`` where ``seq_len`` is the current
            total sequence length (prompt + decoded tokens so far) in vLLM's KV cache.
            Should return Dict[layer_name, (k, v)] with contiguous KV tensors
            of shape [batch, heads, seq_len, head_dim]. When provided, the merged
            KV cache is returned as part of the result.

    Returns:
        When kv_extract_fn is None:
            - Two-way mode: bool (passed)
            - Three-way mode: Tuple[bool, List[List[dict]]] (passed, results)
        When kv_extract_fn is provided:
            - Two-way mode: Tuple[bool, Dict] (passed, merged_kv)
            - Three-way mode: Tuple[bool, List[List[dict]], Dict] (passed, results, merged_kv)

    Raises:
        AssertionError: If generate_fn returns logits and sequences with mismatched shapes.
        AssertionError: If baseline_logits shape doesn't match expected_logits.
        KeyError: If expected_logits tensor has incompatible dimensions.

    Examples:
        ```python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1. Load your model
        model_name = 'openlm-research/open_llama_3b'
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # 2. Prepare your input
        prompt = 'I am a fun tutorial.'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # 3. Retrieve your goldens (in a real example, you wouldn't use exactly the
        #    same model in steps 3 and 4)
        generation_result = model.generate(
            input_ids,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        expected_logits = torch.stack(generation_result['scores'])

        # 4. Build your generate function
        def generate_fn(input_ids):
            input_ids = torch.tensor(input_ids)
            generation_result = model.generate(
                input_ids,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            return torch.stack(generation_result['scores'])

        # 5. Validate
        passed = logit_validation(input_ids.tolist(), generate_fn, expected_logits)
        ```

        Custom Tolerance Configuration:
        ```python
        ...
        # Strict validation for critical applications
        strict_tolerances = {
            "all": (1e-6, 0.01),    # Very strict for all tokens
            "50": (1e-6, 0.005),     # Extremely strict for top-50
            "5": (1e-7, 0.001),      # Ultra strict for top-5
            "1": (1e-8, 0.0005),     # Maximum precision for top-1
        }

        passed = logit_validation(
            input_ids=input_ids,
            generate_fn=neuron_generate_fn,
            expected_logits=expected_logits,
            tol_map=strict_tolerances,
            divergence_difference_tol=0.0001  # Very strict divergence tolerance
        )
        ```

        Batch Processing:
        ```python
        ...
        # Multiple sequences in batch
        batch_input_ids = [
            [1, 2, 3, 4],      # Sequence 1
            [5, 6],            # Sequence 2
        ]

        # Expected logits shape: (seq_len=10, batch_size=2, vocab_size=50257)
        expected_logits = torch.randn(10, 2, 50257)

        passed = logit_validation(batch_input_ids, generate_fn, expected_logits)
        ```

        With visualization:
        ```python
        passed = logit_validation(
            input_ids, generate_fn, expected_logits,
            visualize=True, output_dir="my_validation"
        )
        ```

    Note:
        The `generate_fn` should handle teacher forcing internally or return logits
        that correspond to the expected sequence length. The validation will use
        teacher forcing by extending input_ids with reference tokens as needed.

        For optimal performance with large vocabularies, consider using appropriate
        top-k values in tol_map to focus validation on the most relevant tokens.
        save_logits: Whether to save logit tensors to files.
    """
    # Validate required parameters
    if generate_fn is None or expected_logits is None:
        raise ValueError("Must provide both generate_fn and expected_logits")

    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    # Three-way validation setup
    three_way_mode = baseline_logits is not None
    if three_way_mode:
        assert baseline_logits.shape == expected_logits.shape, (
            f"baseline_logits shape {baseline_logits.shape} must match expected_logits shape {expected_logits.shape}"
        )
        if dynamic_threshold_config is None:
            dynamic_threshold_config = DEFAULT_DYNAMIC_THRESHOLD_CONFIG
        # In three-way mode, use baseline for teacher forcing
        teacher_forcing_logits = baseline_logits
    else:
        teacher_forcing_logits = expected_logits

    batch_size = len(input_ids)
    current_output_start_idx = 0
    teacher_forcing_sequences = teacher_forcing_logits.argmax(dim=2).T
    expected_sequence_length = teacher_forcing_sequences.shape[1]

    passed = True
    results = [[] for _ in range(batch_size)]
    actual_logits = None  # Initialize directly as requested

    # KV cache state: incrementally updated after each generate_fn call
    merged_kv = None
    prompt_lens = [len(ids) for ids in input_ids]

    while current_output_start_idx < expected_sequence_length:
        if multimodal_inputs is not None:
            generate_result = generate_fn(
                input_ids, multimodal_inputs=multimodal_inputs
            )
        else:
            generate_result = generate_fn(input_ids)

        client_sampling = False
        if isinstance(generate_result, tuple):
            actual_logits, actual_sequences = generate_result
            assert actual_logits.shape[:-1] == actual_sequences.T.shape, (
                f"Shape mismatch between logits and sequences returned by generate_fn: "
                f"actual_logits.shape={actual_logits.shape}, actual_sequences.shape={actual_sequences.shape}"
            )
            client_sampling = True
        else:
            actual_logits = generate_result
            actual_sequences = actual_logits.argmax(dim=2).T

        # NaN logits argmax to garbage that "diverges" at token 0. In this case, validate the whole step.
        actual_has_nan = torch.isnan(actual_logits).any().item()
        if actual_has_nan:
            logger.warning(
                "NaN in actual logits; validating %d token(s), then aborting "
                "re-prompt loop.",
                actual_logits.shape[0],
            )
            # Validate the whole step so NaN positions show.
            divergence_idx = min(
                current_output_start_idx + actual_logits.shape[0],
                expected_sequence_length,
            )
        else:
            divergence_idx = _get_divergence_idx(
                teacher_forcing_sequences[:, current_output_start_idx:],
                actual_sequences,
            )
            divergence_idx += current_output_start_idx

        # Update KV state with valid portion from this iteration
        if kv_extract_fn is not None:
            # Current KV length = input length + tokens generated this step
            kv_seq_len = len(input_ids[0]) + actual_logits.shape[0]
            kv_caches = kv_extract_fn(kv_seq_len)
            merged_kv = _update_kv_state(
                merged_kv,
                kv_caches,
                prompt_lens=prompt_lens,
                input_lens_this_iter=[len(ids) for ids in input_ids],
                output_start_idx=current_output_start_idx,
                divergence_idx=divergence_idx,
                total_output_len=expected_sequence_length,
            )

        for batch_idx in range(batch_size):
            # Process tokens up to divergence point
            for token_idx in range(divergence_idx - current_output_start_idx):
                actual_token_id = None
                if client_sampling:
                    actual_token_id = actual_sequences[batch_idx, token_idx].item()

                global_token_idx = token_idx + current_output_start_idx

                # Get baseline logits for three-way comparison
                baseline_logits_token = None
                if three_way_mode:
                    baseline_logits_token = baseline_logits[
                        global_token_idx, batch_idx, :
                    ]

                single_token_passed, single_token_results = (
                    _validate_single_token_logits(
                        expected_logits=expected_logits[global_token_idx, batch_idx, :],
                        actual_logits=actual_logits[token_idx, batch_idx, :],
                        tol_map=tol_map,
                        divergence_difference_tol=divergence_difference_tol,
                        remove_shift=True,
                        actual_token_id=actual_token_id,
                        baseline_logits=baseline_logits_token,
                        dynamic_threshold_config=dynamic_threshold_config
                        if three_way_mode
                        else None,
                        divergence_n_ulps=divergence_n_ulps,
                        mantissa_bits=mantissa_bits,
                        test_device=test_device,
                    )
                )

                results[batch_idx].append(single_token_results)
                passed &= single_token_passed

        # Stop after recording this step, skipping re-prompting with NaNs.
        if actual_has_nan:
            passed = False
            break

        # Multimodal teacher-forcing re-entry is supported: the multimodal
        # generate_fn reconstructs the unexpanded prompt from the extended
        # input_ids by collapsing image-pad runs to one placeholder each, so the
        # extend-and-loop path below works for both modalities.
        for i in range(len(input_ids)):
            input_ids[i].extend(
                teacher_forcing_sequences[
                    i, current_output_start_idx:divergence_idx
                ].tolist()
            )

        current_output_start_idx = divergence_idx

    _print_logit_validation_results(
        results,
        tol_map,
        divergence_difference_tol,
        suppress_passing,
        colorize,
        label,
        divergence_n_ulps=divergence_n_ulps,
        mantissa_bits=mantissa_bits,
    )

    # Print three-way metrics if in three-way mode
    if three_way_mode:
        _print_three_way_metrics(results, label)

        # Visualization for three-way mode
        if visualize:
            os.makedirs(output_dir, exist_ok=True)
            visualize_logit_results(results, output_dir, save_logits=save_logits)

            if save_logits:
                torch.save(
                    expected_logits, os.path.join(output_dir, "reference_logits.pt")
                )
                if actual_logits is not None:
                    torch.save(
                        actual_logits, os.path.join(output_dir, "target_logits.pt")
                    )
                if baseline_logits is not None:
                    torch.save(
                        baseline_logits, os.path.join(output_dir, "baseline_logits.pt")
                    )

        if merged_kv is not None:
            return passed, results, merged_kv
        return passed, results

    # Visualization for two-way mode
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        visualize_logit_results(results, output_dir, save_logits=save_logits)

        if save_logits:
            torch.save(expected_logits, os.path.join(output_dir, "reference_logits.pt"))
            if actual_logits is not None:
                torch.save(actual_logits, os.path.join(output_dir, "target_logits.pt"))

    if merged_kv is not None:
        return passed, results, merged_kv
    return passed


def _update_kv_state(
    merged_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    kv_snapshot: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    prompt_lens: List[int],
    input_lens_this_iter: List[int],
    output_start_idx: int,
    divergence_idx: int,
    total_output_len: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Incrementally update merged KV state with valid tokens from a generate step.

    Args:
        merged_kv: Running KV state (None on first call), shape [batch, heads, total_seq, dim]
        kv_snapshot: KV from this generate step, shape [batch, heads, snapshot_seq, dim]
        prompt_lens: Per-sequence prompt lengths
        input_lens_this_iter: Per-sequence input lengths for this iteration
        output_start_idx: Output token index where this generate step started
        divergence_idx: Output token index where divergence occurred
        total_output_len: Total expected output tokens

    Returns:
        Updated merged_kv
    """
    layer_names = list(kv_snapshot.keys())
    first_iteration = merged_kv is None
    batch_size = len(prompt_lens)
    max_prompt_len = max(prompt_lens)
    total_kv_len = max_prompt_len + total_output_len

    # Initialize on first call
    if first_iteration:
        sample_k = kv_snapshot[layer_names[0]][0]
        b, h, _, d = sample_k.shape
        merged_kv = {}
        for name in layer_names:
            merged_kv[name] = (
                torch.zeros(b, h, total_kv_len, d, dtype=sample_k.dtype),
                torch.zeros(b, h, total_kv_len, d, dtype=sample_k.dtype),
            )

    for name in layer_names:
        k_snap, v_snap = kv_snapshot[name]
        k_out, v_out = merged_kv[name]

        for b in range(batch_size):
            plen = prompt_lens[b]
            ilen = input_lens_this_iter[b]
            num_decoded = divergence_idx - output_start_idx
            valid_end_in_kv = ilen + num_decoded

            # First iteration only: copy prompt KV
            if first_iteration:
                k_out[b, :, :plen, :] = k_snap[b, :, :plen, :]
                v_out[b, :, :plen, :] = v_snap[b, :, :plen, :]

            # All iterations: copy decode KV for newly validated tokens
            out_start = plen + output_start_idx
            out_end = plen + divergence_idx
            k_out[b, :, out_start:out_end, :] = k_snap[b, :, ilen:valid_end_in_kv, :]
            v_out[b, :, out_start:out_end, :] = v_snap[b, :, ilen:valid_end_in_kv, :]

    return merged_kv


def _get_divergence_idx(
    expected_sequences: torch.Tensor, actual_sequences: torch.Tensor
) -> int:
    """Get the index of the first divergent token across all batches."""
    min_seq_len = min(actual_sequences.shape[1], expected_sequences.shape[1])
    diff = torch.ne(
        actual_sequences[:, :min_seq_len], expected_sequences[:, :min_seq_len]
    )

    if torch.sum(diff) == 0:
        return min_seq_len
    else:
        return torch.min(torch.nonzero(diff), 0).values[1].item() + 1


def _preprocess_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    remove_shift: bool,
    return_removed_indices: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor],
]:
    """
    This function handles two preprocessing tasks: omitting -inf values and
    removing the shift. -inf values sometimes appear in logits when a token is
    prohibited, and they cause issues in some downstream tasks. This function
    identifies indices at which either the actual or expected logits are -inf
    and omits them. To return the indices of the removed values, enable
    return_removed_indices.

    For instance, the input tensors:
        actual = [1, -inf, 3, 4], expected = [5, 6, 7, -inf]
    would be output as:
        actual = [1, 3], expected = [5, 7]

    This function also optionally finds and removes a constant shift in the
    logits by finding the least-squares approximation for p in the following
    system of linear equations:
        actual_logits = A @ p
    where
        A = [expected_logits | 1]
        p = [slope, shift].T
    In other words, it does a linear regression. Then, it subtracts the shift
    from actual_logits.

    For instance, the input tensors:
        actual = [1, 2, 3, 4], expected = [5, 6, 7, 8]
    would be output as:
        actual = [5, 6, 7, 8], expected = [5, 6, 7, 8], shift = -4
    """
    # Omit indices at which logits are -inf
    vocab_size = len(expected_logits)
    assert vocab_size == len(actual_logits)
    ninf_idxs = torch.nonzero(
        torch.logical_or(
            actual_logits == float("-inf"), expected_logits == float("-inf")
        )
    )
    expected_logits = expected_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    actual_logits = actual_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    shift = 0
    if remove_shift:  # Calculate and remove shift
        A = np.vstack([expected_logits.float(), np.ones(len(expected_logits))]).T
        _, shift = np.linalg.lstsq(A, actual_logits.float(), rcond=None)[0]
        actual_logits -= shift
    if return_removed_indices:
        return expected_logits, actual_logits, shift, ninf_idxs.reshape(-1)
    else:
        return expected_logits, actual_logits, shift


def _validate_top_k_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    top_k: Union[str, int],
    atol: float,
    rtol: float,
    test_device: str = "cpu",
) -> Tuple[bool, float]:
    """Validates logits for top-k tokens using specified tolerances."""
    if top_k != "all":  # filter only the top k most likely tokens
        if isinstance(top_k, str):
            top_k = int(top_k)

        top_k_result = torch.topk(expected_logits, top_k)
        expected_logits = top_k_result.values
        # Select actual logits at the same indices to maintain alignment
        actual_logits = torch.index_select(actual_logits, 0, top_k_result.indices)

        # Mask out positions where actual has -inf (not returned by model)
        valid_mask = ~torch.isinf(actual_logits)
        if valid_mask.sum() == 0:
            return False, float("inf")
        expected_logits = expected_logits[valid_mask]
        actual_logits = actual_logits[valid_mask]

    # Use neuron_allclose on Neuron hardware, CPU fallback only in CPU mode
    if test_device == "cpu":
        result = _cpu_allclose(actual_logits, expected_logits, rtol=rtol, atol=atol)
    else:
        from vllm_neuron.accuracy.testing import _neuron_allclose

        result = _neuron_allclose(actual_logits, expected_logits, rtol=rtol, atol=atol)

    return result.allclose, result.max_rel_error


def _analyze_top_k_logits(
    expected_logits: torch.Tensor, actual_logits: torch.Tensor
) -> dict:
    """Analyze top-k values, indices, and relative errors for logits comparison."""

    def get_top2_values_indices_diff(logits):
        top2_values, top2_indices = torch.topk(logits, 2)
        top1_top2_diff = top2_values[0] - top2_values[1]
        top1_top2_relative_diff = (
            (top1_top2_diff / torch.abs(top2_values[0]))
            if torch.abs(top2_values[0]) > torch.tensor(1e-8)
            else torch.tensor(0.0)
        )
        return top2_values, top2_indices, top1_top2_diff, top1_top2_relative_diff

    (
        expected_top2_values,
        expected_top2_indices,
        expected_top1_top2_diff,
        expected_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(expected_logits)
    (
        actual_top2_values,
        actual_top2_indices,
        actual_top1_top2_diff,
        actual_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(actual_logits)

    def get_relative_error(actual_val, expected_val):
        return (
            torch.abs((actual_val - expected_val) / expected_val)
            if torch.abs(expected_val) > torch.tensor(1e-8)
            else torch.abs(actual_val - expected_val)
        )

    top1_relative_error = get_relative_error(
        actual_top2_values[0], expected_top2_values[0]
    )
    top2_relative_error = get_relative_error(
        actual_top2_values[1], expected_top2_values[1]
    )
    actual_values_with_expected_top1_top2_indices_relative_diff = get_relative_error(
        actual_logits[expected_top2_indices[0]], actual_logits[expected_top2_indices[1]]
    )

    return {
        "expected_top2_values": expected_top2_values,
        "actual_top2_values": actual_top2_values,
        "expected_top2_indices": expected_top2_indices,
        "actual_top2_indices": actual_top2_indices,
        "expected_top1_top2_diff": expected_top1_top2_diff,
        "actual_top1_top2_diff": actual_top1_top2_diff,
        "expected_top1_top2_relative_diff": expected_top1_top2_relative_diff,
        "actual_top1_top2_relative_diff": actual_top1_top2_relative_diff,
        "actual_with_expected_top1_top2_relative_diff": actual_values_with_expected_top1_top2_indices_relative_diff,
        "top1_relative_error": top1_relative_error,
        "top2_relative_error": top2_relative_error,
    }


def _validate_tolerance_levels(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    tol_map: dict,
    test_device: str = "cpu",
) -> Tuple[bool, dict, dict]:
    """Validate logits against tolerance levels for different top-k values."""
    error_map = {k: 0 for k in tol_map.keys()}
    total_errors = collections.defaultdict(dict)
    passed = torch.tensor(True)

    max_abs_expected = torch.max(torch.abs(expected_logits)).item()
    total_errors["all"]["mean_abs_error"] = (
        torch.nn.functional.l1_loss(
            actual_logits, expected_logits, reduction="mean"
        ).item()
        / max_abs_expected
    )
    total_errors["all"]["mean_squared_error"] = torch.nn.functional.mse_loss(
        actual_logits, expected_logits, reduction="mean"
    ).item() / (max_abs_expected**2)

    for top_k, tols in tol_map.items():
        atol, rtol = tols
        in_bounds, error = _validate_top_k_logits(
            expected_logits, actual_logits, top_k, atol, rtol, test_device
        )
        total_errors[top_k]["max_abs_error"] = abs(error)
        total_errors[top_k]["max_squared_error"] = error**2
        passed &= in_bounds
        error_map[top_k] = error

    return passed.item(), error_map, dict(total_errors)


def _compute_ulp(value: float, mantissa_bits: int = 7) -> float:
    """Compute the Unit in the Last Place (ULP) for a floating-point value.

    ULP is the step between two consecutive representable values at a given
    magnitude.  For BF16 (7 mantissa bits) at value 19, ULP = 0.125.

    Args:
        value: The floating-point value (uses its absolute value).
        mantissa_bits: Number of mantissa bits in the target dtype.
            BF16 = 7, FP16 = 10, FP32 = 23.
    """
    import math

    abs_val = abs(value)
    if math.isnan(abs_val) or math.isinf(abs_val):
        return float("nan")
    if abs_val < 1e-30:
        return 0.0
    return 2 ** (math.floor(math.log2(abs_val)) - mantissa_bits)


def _analyze_divergence(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    actual_token_id: Optional[int],
    divergence_difference_tol: float,
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
) -> Tuple[bool, float, bool, float]:
    """Analyze sequence divergence and calculate divergence difference.

    Divergence occurs when the actual model picks a different greedy token than
    the reference.  The divergence difference is the absolute gap between the
    actual logits for the two competing tokens (actual's top-1 vs reference's
    top-1).

    By default, the tolerance is ULP-aware: it is computed dynamically as
    ``n_ulps * ULP(max(|logit_a|, |logit_b|))`` using the target dtype's
    mantissa width (BF16 = 7 bits).  A gap of exactly *n_ulps* ULPs is
    considered in-bounds (``<=``).  This accounts for the fact that BF16 cannot
    distinguish values closer than 1 ULP, so a divergence within that step size
    is expected precision behavior, not a real accuracy issue.

    Set *divergence_n_ulps* to ``None`` to fall back to the fixed
    *divergence_difference_tol*.

    Returns:
        (divergence, divergence_difference, divergence_in_bounds, effective_threshold)
        where *effective_threshold* is the tolerance that was actually applied
        (dynamic ULP-based or static fallback).
    """
    divergence = False
    divergence_difference = 0
    effective_threshold = divergence_difference_tol

    greedy_next_token_id = expected_logits.argmax().item()
    if actual_token_id is None:
        actual_token_id = actual_logits.argmax().item()

    if greedy_next_token_id != actual_token_id:
        divergence = True
        divergence_difference = torch.abs(
            actual_logits[actual_token_id] - actual_logits[greedy_next_token_id]
        )
        if divergence_n_ulps is not None:
            max_abs = max(
                abs(actual_logits[actual_token_id].item()),
                abs(actual_logits[greedy_next_token_id].item()),
            )
            ulp = _compute_ulp(max_abs, mantissa_bits)
            dynamic_tol = divergence_n_ulps * ulp
            effective_threshold = dynamic_tol
            divergence_in_bounds = divergence_difference <= dynamic_tol
        else:
            divergence_in_bounds = divergence_difference <= divergence_difference_tol
    else:
        divergence_in_bounds = True

    return divergence, divergence_difference, divergence_in_bounds, effective_threshold


def _validate_single_token_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    tol_map: dict,
    divergence_difference_tol: float,
    remove_shift: bool,
    actual_token_id: Optional[int] = None,
    baseline_logits: Optional[torch.Tensor] = None,
    dynamic_threshold_config: Optional[dict] = None,
    test_device: str = "cpu",
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
) -> Tuple[bool, dict]:
    """Validates logits for a single token position across all tolerance levels.

    When baseline_logits is provided, performs three-way comparison:
    - baseline_logits: FP32 reference
    - expected_logits: same-dtype baseline (e.g., BF16)
    - actual_logits: target (Neuron)

    Computes L-inf and L2 errors for both expected and actual vs baseline,
    then applies dynamic thresholds: tgt_error < multiplier * base_error
    """
    # Warn on NaN (crashes downstream numerics like _compute_ulp)
    if torch.isnan(actual_logits).any():
        nan_count = torch.isnan(actual_logits).sum().item()
        logger.warning(
            "Actual logits contain %d NaN out of %d. Validation will fail; "
            "downstream metrics may be degenerate.",
            nan_count,
            actual_logits.numel(),
        )

    expected_logits, actual_logits, shift, removed_indices = _preprocess_logits(
        expected_logits,
        actual_logits,
        remove_shift,
        return_removed_indices=True,
    )
    if actual_token_id is not None:
        actual_token_id = _calculate_new_index(actual_token_id, removed_indices)
        assert actual_token_id is not None, "Provided actual token ID has -inf score"

    # Three-way comparison metrics (informational only, does not affect two-way pass/fail)
    three_way_metrics = None
    three_way_passed = None
    if baseline_logits is not None:
        baseline_logits_clean = baseline_logits[
            ~torch.isin(torch.arange(len(baseline_logits)), removed_indices)
        ]
        three_way_metrics = _compute_three_way_metrics(
            baseline_logits_clean, expected_logits, actual_logits
        )
        if dynamic_threshold_config is not None:
            linf_mult = dynamic_threshold_config.get("linf_multiplier", 1.5)
            l2_mult = dynamic_threshold_config.get("l2_multiplier", 1.5)
            linf_pass = (
                three_way_metrics.tgt_linf < linf_mult * three_way_metrics.base_linf
            )
            l2_pass = three_way_metrics.tgt_l2 < l2_mult * three_way_metrics.base_l2
            three_way_passed = linf_pass and l2_pass

    top_k_analysis = _analyze_top_k_logits(expected_logits, actual_logits)

    tolerance_passed, error_map, total_errors = _validate_tolerance_levels(
        expected_logits, actual_logits, tol_map, test_device
    )

    divergence, divergence_difference, divergence_in_bounds, divergence_threshold = (
        _analyze_divergence(
            expected_logits,
            actual_logits,
            actual_token_id,
            divergence_difference_tol,
            divergence_n_ulps=divergence_n_ulps,
            mantissa_bits=mantissa_bits,
        )
    )

    passed = tolerance_passed and divergence_in_bounds

    results = {
        "passed": passed,
        "divergence": divergence,
        "divergence_difference": divergence_difference,
        "divergence_in_bounds": divergence_in_bounds,
        "divergence_threshold": divergence_threshold,
        "total_errors": total_errors,
        "error_map": error_map,
        "shift": shift,
        "expected_logits": expected_logits,
        "actual_logits": actual_logits,
        "expected_top2_values": top_k_analysis["expected_top2_values"].tolist(),
        "actual_top2_values": top_k_analysis["actual_top2_values"].tolist(),
        "expected_top1_top2_diff": top_k_analysis["expected_top1_top2_diff"].item(),
        "actual_top1_top2_diff": top_k_analysis["actual_top1_top2_diff"].item(),
        "expected_top1_top2_relative_diff": top_k_analysis[
            "expected_top1_top2_relative_diff"
        ].item(),
        "actual_top1_top2_relative_diff": top_k_analysis[
            "actual_top1_top2_relative_diff"
        ].item(),
        "actual_with_expected_top1_top2_relative_diff": top_k_analysis[
            "actual_with_expected_top1_top2_relative_diff"
        ].item(),
        "expected_top2_indices": top_k_analysis["expected_top2_indices"].tolist(),
        "actual_top2_indices": top_k_analysis["actual_top2_indices"].tolist(),
        "top1_relative_errors": top_k_analysis["top1_relative_error"],
        "top2_relative_errors": top_k_analysis["top2_relative_error"],
    }

    if three_way_metrics is not None:
        results["three_way_metrics"] = {
            "base_linf": three_way_metrics.base_linf,
            "tgt_linf": three_way_metrics.tgt_linf,
            "base_l2": three_way_metrics.base_l2,
            "tgt_l2": three_way_metrics.tgt_l2,
            "bc": three_way_metrics.bc,
            "base_errors": three_way_metrics.base_errors,
            "tgt_errors": three_way_metrics.tgt_errors,
            "passed": three_way_passed,
        }

    return passed, results


def _compute_bc(errors1: np.ndarray, errors2: np.ndarray, num_bins: int = 100) -> float:
    """Bhattacharyya Coefficient between two error distributions.

    Measures similarity of error distributions. BC=1.0 means identical distributions,
    BC=0.0 means no overlap. Used to compare baseline vs target error patterns.
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


def _compute_three_way_metrics(
    baseline_logits: torch.Tensor,
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
) -> ThreeWayTokenMetrics:
    """Compute three-way comparison metrics.

    Compares expected (same-dtype baseline) and actual (target) against FP32 baseline.
    L-inf/L2 are normalized by FP32 magnitude. BC computed on raw absolute errors.
    """
    # Convert to float and mask out infinities
    fp32 = baseline_logits.float()
    base = expected_logits.float()
    tgt = actual_logits.float()
    mask = ~(torch.isinf(fp32) | torch.isinf(base) | torch.isinf(tgt))

    if mask.sum() == 0:
        return ThreeWayTokenMetrics(
            base_linf=float("inf"),
            tgt_linf=float("inf"),
            base_l2=float("inf"),
            tgt_l2=float("inf"),
            bc=0.0,
        )

    fp32_v, base_v, tgt_v = fp32[mask], base[mask], tgt[mask]
    base_diff = base_v - fp32_v
    tgt_diff = tgt_v - fp32_v

    # L-inf relative error: max|diff| / max|ref|
    fp32_max = torch.max(torch.abs(fp32_v)).clamp(min=1e-10)
    base_linf = (torch.max(torch.abs(base_diff)) / fp32_max).item()
    tgt_linf = (torch.max(torch.abs(tgt_diff)) / fp32_max).item()

    # L2 relative error: norm(diff) / norm(ref)
    fp32_norm = torch.norm(fp32_v).clamp(min=1e-10)
    base_l2 = (torch.norm(base_diff) / fp32_norm).item()
    tgt_l2 = (torch.norm(tgt_diff) / fp32_norm).item()

    # Raw absolute errors (for BC aggregation across prompts)
    base_errors = torch.abs(base_diff).detach().cpu().numpy()
    tgt_errors = torch.abs(tgt_diff).detach().cpu().numpy()

    # BC on raw absolute errors
    bc = _compute_bc(base_errors, tgt_errors)

    return ThreeWayTokenMetrics(
        base_linf=base_linf,
        tgt_linf=tgt_linf,
        base_l2=base_l2,
        tgt_l2=tgt_l2,
        bc=bc,
        base_errors=base_errors,
        tgt_errors=tgt_errors,
    )


def _calculate_new_index(
    original_index: int, removed_indices: torch.Tensor
) -> Optional[int]:
    """Calculate new index after removing elements from a tensor."""
    if original_index in removed_indices:
        return None

    # Count how many elements were removed before this index
    removed_before = 0
    for removed_idx in removed_indices:
        if removed_idx < original_index:
            removed_before += 1

    new_index = original_index - removed_before
    return new_index


def _get_logit_validation_max_topk_error_results_summary(
    results: List[List[dict]],
) -> dict:
    """Extracts maximum errors across all tokens and batches for each top-k value.

    NaN handling: +inf logits (not filtered by _preprocess_logits which only
    removes -inf) produce NaN errors. Since NaN > x is always False in IEEE 754,
    a plain `>` comparison would never update batch_index/token_index, causing a
    KeyError when _format_validation_summary reads them. NaN values are skipped
    here — they are already reported upstream via the NaN logit warning.
    """
    summary = {
        "max_divergence": {"error": -1, "batch_index": -1, "token_index": -1},
        "max_top_k_errors": {},
        "per_batch_max_errors": {},
    }
    for batch_index in range(len(results)):
        batch_max = {}
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            div_diff = token_results["divergence_difference"]
            cur_max = summary["max_divergence"]["error"]
            # Skip NaN — it is reported upstream via the NaN warning
            if not math.isnan(div_diff) and div_diff > cur_max:
                summary["max_divergence"]["error"] = div_diff
                summary["max_divergence"]["batch_index"] = batch_index
                summary["max_divergence"]["token_index"] = token_index
                summary["max_divergence"]["threshold"] = token_results.get(
                    "divergence_threshold"
                )
            for top_k, error in token_results["error_map"].items():
                if top_k not in summary["max_top_k_errors"]:
                    summary["max_top_k_errors"][top_k] = {
                        "error": -1,
                        "batch_index": -1,
                        "token_index": -1,
                    }
                cur_top_k_max = summary["max_top_k_errors"][top_k]["error"]
                if not math.isnan(error) and error > cur_top_k_max:
                    summary["max_top_k_errors"][top_k]["error"] = error
                    summary["max_top_k_errors"][top_k]["batch_index"] = batch_index
                    summary["max_top_k_errors"][top_k]["token_index"] = token_index
                if top_k not in batch_max or error > batch_max[top_k]["error"]:
                    batch_max[top_k] = {"error": error, "token_index": token_index}
        summary["per_batch_max_errors"][batch_index] = batch_max
    return summary


def _get_logit_validation_average_over_tokens_results_summary(
    results: List[List[dict]],
) -> dict:
    """Calculates average errors across all tokens for each error metric."""
    summary = {
        "average_over_tokens": collections.defaultdict(dict),
    }

    count = 0
    total_error_dict = collections.defaultdict(dict)
    for batch_index in range(len(results)):
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            count += 1
            for top_k, top_k_errors in token_results["total_errors"].items():
                for error_key, error in top_k_errors.items():
                    total_error_dict[top_k][error_key] = (
                        total_error_dict.get(top_k, {}).get(error_key, 0) + error
                    )

    if count != 0:
        for top_k, error_dict in total_error_dict.items():
            for error_key, error in error_dict.items():
                summary["average_over_tokens"][top_k][error_key] = error / count

    return summary


def _get_logit_validation_results_summary(results: List[List[dict]]) -> dict:
    """Combines maximum and average error summaries into a single results dictionary."""
    max_error_summary = _get_logit_validation_max_topk_error_results_summary(results)
    average_over_tokens_summary = (
        _get_logit_validation_average_over_tokens_results_summary(results)
    )

    summary = max_error_summary | average_over_tokens_summary

    return summary


def _classify_validation_result(
    token_results: dict, divergence_difference_tol: float, tol_map: dict
) -> _ValidationStatus:
    """
    Classifies validation result status for a single token validation result.

    Args:
        token_results: Dictionary containing validation results for a single token
        divergence_difference_tol: Threshold for acceptable divergence difference
        tol_map: Dictionary mapping top-k values to (atol, rtol) tolerance tuples

    Returns:
        _ValidationStatus enum representing the classification status
    """
    # Check for top-k errors by comparing error_map against tolerances
    has_topk_errors = False
    for top_k, error in token_results["error_map"].items():
        if top_k in tol_map:
            _, rtol = tol_map[top_k]  # Use rtol as the threshold
            if error > rtol:
                has_topk_errors = True
                break

    # Check if there's any divergence
    if not token_results["divergence"]:
        if has_topk_errors:
            return _ValidationStatus.TOPK_ERRORS
        else:
            return _ValidationStatus.MATCHED

    # There is divergence, check if it's within threshold.
    # Prefer pre-computed divergence_in_bounds (set by _analyze_divergence,
    # which may use dynamic ULP-based tolerance).
    if "divergence_in_bounds" in token_results:
        divergence_within_threshold = token_results["divergence_in_bounds"]
    else:
        divergence_within_threshold = (
            token_results["divergence_difference"] <= divergence_difference_tol
        )

    if divergence_within_threshold:
        if has_topk_errors:
            return _ValidationStatus.ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS
        else:
            return _ValidationStatus.ACCEPTABLE_DIVERGENCE
    else:
        if has_topk_errors:
            return _ValidationStatus.DIVERGED_WITH_TOPK_ERRORS
        else:
            return _ValidationStatus.DIVERGED


def _format_error_with_threshold(
    error: float,
    threshold: float,
    colorize: bool = True,
    threshold_label: Optional[str] = None,
) -> str:
    """Format error value with color based on threshold compliance.

    Args:
        threshold_label: If provided, display this label instead of the raw
            numeric threshold (e.g. ``"1 ULP(s)"``).  The numeric value is
            still used for the pass/fail comparison.
    """
    error_exceeds_threshold = error > threshold
    colored_error = _colorize_text(
        f"{error:.{_ERROR_PRECISION}f}",
        "RED" if error_exceeds_threshold else "GREEN",
        colorize,
    )
    display_threshold = (
        threshold_label if threshold_label is not None else str(threshold)
    )
    colored_threshold = f"(threshold: {display_threshold})"
    colored_status_icon = (
        _colorize_text("✗", "RED", colorize)
        if error_exceeds_threshold
        else _colorize_text("✓", "GREEN", colorize)
    )
    return f"{colored_error} {colored_threshold} {colored_status_icon}"


def _calculate_summary_statistics(results: List[List[dict]]) -> dict:
    """Calculate summary statistics from validation results."""
    if not results or not results[0]:
        return {
            "batch_size": 0,
            "total_tokens": 0,
            "passed_tokens": 0,
            "failed_tokens": 0,
            "tokens_per_batch": 0,
        }

    batch_size = len(results)
    total_tokens = sum(len(batch_results) for batch_results in results)
    passed_tokens = sum(
        1
        for batch_results in results
        for token_result in batch_results
        if token_result["passed"]
    )
    failed_tokens = total_tokens - passed_tokens
    tokens_per_batch = len(results[0]) if results and results[0] else 0

    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "passed_tokens": passed_tokens,
        "failed_tokens": failed_tokens,
        "tokens_per_batch": tokens_per_batch,
    }


def _format_summary_header(stats: dict, colorize: bool = True) -> List[str]:
    """Format the summary header section."""
    lines = [_colorize_text("=== VALIDATION SUMMARY ===", "BOLD", colorize)]

    # Show token count and batch configuration
    if stats["tokens_per_batch"] > 0:
        lines.append(
            f"Total Tokens: {stats['total_tokens']} ({stats['tokens_per_batch']} per batch, {stats['batch_size']} batches)"
        )
    else:
        lines.append(f"Total Tokens: {stats['total_tokens']}")

    # Overall status
    overall_status = (
        _colorize_text("PASSED", "GREEN", colorize)
        if stats["failed_tokens"] == 0
        else _colorize_text("FAILED", "RED", colorize)
    )
    lines.extend(
        [
            f"Overall Status: {overall_status}",
            "",
            _colorize_text("Max Errors:", "BOLD", colorize),
        ]
    )

    return lines


def _format_error_metrics_section(
    summary: dict,
    tol_map: dict,
    divergence_difference_tol: float,
    colorize: bool = True,
    divergence_n_ulps: Optional[int] = None,
) -> List[str]:
    """Format the error metrics section with thresholds and coloring."""
    lines = []

    # Add divergence error with threshold
    max_div = summary["max_divergence"]
    if max_div["error"] >= 0:
        # Use the per-token dynamic threshold stored during validation,
        # falling back to the static tolerance for older result dicts.
        effective_tol = max_div.get("threshold") or divergence_difference_tol
        formatted_error = _format_error_with_threshold(
            max_div["error"],
            effective_tol,
            colorize,
            threshold_label=(
                f"{divergence_n_ulps} ULP(s)" if divergence_n_ulps is not None else None
            ),
        )
        lines.append(
            f"- Divergence: {formatted_error} at Batch {max_div['batch_index']} Token {max_div['token_index']}"
        )

    # Add top-k errors with thresholds - use tol_map keys to determine order
    for top_k in tol_map.keys():
        if top_k in summary["max_top_k_errors"]:
            max_error = summary["max_top_k_errors"][top_k]
            threshold = tol_map[top_k][1]  # Use rtol as threshold
            formatted_error = _format_error_with_threshold(
                max_error["error"], threshold, colorize
            )

            k_label = f"K{top_k}" if top_k != "all" else "All"
            lines.append(
                f"- {k_label}: {formatted_error} at Batch {max_error['batch_index']} Token {max_error['token_index']}"
            )

    # Per-batch max errors (only when batch_size > 1)
    per_batch = summary.get("per_batch_max_errors")
    if per_batch and len(per_batch) > 1:
        lines.append("")
        lines.append(_colorize_text("Per-Batch Max Errors:", "BOLD", colorize))
        for batch_idx in sorted(per_batch.keys()):
            batch_errors = per_batch[batch_idx]
            parts = []
            for top_k in tol_map.keys():
                if top_k in batch_errors:
                    k_label = f"K{top_k}" if top_k != "all" else "All"
                    parts.append(f"{k_label}={batch_errors[top_k]['error']:.4f}")
            lines.append(f"  Batch {batch_idx}: {', '.join(parts)}")

    return lines


def _format_token_results_table(
    results: List[List[dict]],
    divergence_difference_tol: float,
    tol_map: dict,
    suppress_passing: bool,
    colorize: bool = True,
) -> str:
    """Formats per-token validation results in a clean table format with automatic color coding."""
    if not results or not results[0]:
        return "No token results to display."

    batch_size = len(results)
    max_tokens = max(len(batch_results) for batch_results in results)

    lines = []
    suppressed_count = 0

    for token_idx in range(max_tokens):
        for batch_idx in range(batch_size):
            if token_idx < len(results[batch_idx]):
                token_results = results[batch_idx][token_idx]

                # Determine status with enhanced classification
                status_enum = _classify_validation_result(
                    token_results, divergence_difference_tol, tol_map
                )

                # Check if we should suppress this result (only suppress MATCHED status)
                if suppress_passing and status_enum == _ValidationStatus.MATCHED:
                    suppressed_count += 1
                    continue

                # Get display text with color
                status = _colorize_text(
                    status_enum.get_display_text(), status_enum.get_color(), colorize
                )

                # Format divergence info
                divergence_info = ""
                if token_results["divergence"]:
                    divergence_info = (
                        f" Δ = {token_results['divergence_difference']:.4f}"
                    )

                # Extract and color-code error values based on thresholds
                def format_error_value(top_k: str, error_value: float) -> str:
                    """Format error value with color based on threshold compliance."""
                    if top_k in tol_map:
                        _, rtol = tol_map[top_k]  # Use rtol as threshold
                        if error_value <= rtol:
                            return _colorize_text(
                                f"{error_value:.4f}", "GREEN", colorize
                            )
                        else:
                            return _colorize_text(f"{error_value:.4f}", "RED", colorize)
                    else:
                        return f"{error_value:.4f}"

                k5_error = token_results["error_map"].get("5", 0.0)
                k50_error = token_results["error_map"].get("50", 0.0)
                k1000_error = token_results["error_map"].get("1000", 0.0)
                all_error = token_results["error_map"].get("all", 0.0)

                k5_formatted = format_error_value("5", k5_error)
                k50_formatted = format_error_value("50", k50_error)
                k1000_formatted = format_error_value("1000", k1000_error)
                all_formatted = format_error_value("all", all_error)

                status_and_divergence = f"{status}{divergence_info}"
                padding_needed = max(
                    0,
                    50
                    - len(
                        status_and_divergence.replace("\033[92m", "")
                        .replace("\033[91m", "")
                        .replace("\033[93m", "")
                        .replace("\033[0m", "")
                    ),
                )
                padding = " " * padding_needed

                line = (
                    f"Batch {batch_idx} Token {token_idx:2d}: {status_and_divergence}{padding} | "
                    f"K5: {k5_formatted}  K50: {k50_formatted}  K1000: {k1000_formatted}  All: {all_formatted}"
                )
                lines.append(line)

                if token_results["divergence"]:
                    lines.append("")
                    lines.append(
                        "⟲ Teacher Forcing Applied: Models diverged but continuing validation with expected tokens"
                    )
                    lines.append("")

    if suppress_passing and suppressed_count > 0:
        if not lines:  # All results were suppressed
            lines.append(f"All {suppressed_count} tokens passed validation")

    return "\n".join(lines)


def _check_for_shifts(results: List[List[dict]]) -> Tuple[bool, float]:
    """Check if any token results had non-zero shifts applied during preprocessing."""
    has_shifts = False
    max_abs_shift = 0.0

    for batch_results in results:
        for token_result in batch_results:
            shift = token_result.get("shift", 0)
            if shift != 0:
                has_shifts = True
                max_abs_shift = max(max_abs_shift, abs(shift))

    return has_shifts, max_abs_shift


def _format_validation_summary(
    results: List[List[dict]],
    tol_map: dict,
    divergence_difference_tol: float,
    colorize: bool = True,
    divergence_n_ulps: Optional[int] = None,
) -> str:
    """Formats a comprehensive validation summary with threshold information and color coding."""
    if not results:
        return "No results to summarize."

    # Calculate summary statistics using helper function
    stats = _calculate_summary_statistics(results)
    if stats["total_tokens"] == 0:
        return "No results to summarize."

    # Get max errors
    summary = _get_logit_validation_results_summary(results)

    lines = _format_summary_header(stats, colorize)
    lines.extend(
        _format_error_metrics_section(
            summary,
            tol_map,
            divergence_difference_tol,
            colorize,
            divergence_n_ulps=divergence_n_ulps,
        )
    )
    has_shifts, max_abs_shift = _check_for_shifts(results)
    if has_shifts:
        lines.append("")
        shift_notice = _colorize_text(
            f"Removed constant shift in logits (max |shift|: {max_abs_shift:.6f})",
            "YELLOW",
            colorize,
        )
        lines.append(shift_notice)
    lines.append("=" * _SUMMARY_WIDTH)

    return "\n".join(lines)


def _format_thresholds_legend(
    tol_map: dict,
    divergence_difference_tol: float,
    colorize: bool = True,
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
) -> str:
    """Format thresholds legend showing divergence and TopK tolerances."""
    if divergence_n_ulps is not None:
        div_label = (
            f"Divergence: {divergence_n_ulps} ULP(s) "
            f"(dynamic, {mantissa_bits} mantissa bits)"
        )
    else:
        div_label = f"Divergence Difference: {divergence_difference_tol}"
    lines = [
        _colorize_text("VALIDATION THRESHOLDS", "BOLD", colorize),
        div_label,
    ]

    topk_tolerances = []
    for top_k, (_, rtol) in tol_map.items():
        if top_k == "all":
            topk_tolerances.append(f"All: {rtol}")
        else:
            topk_tolerances.append(f"K{top_k}: {rtol}")

    lines.append(f"TopK Error Tolerances (rtol): {', '.join(topk_tolerances)}")

    return "\n".join(lines)


def _print_logit_validation_results(
    results: List[List[Any]],
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    suppress_passing: bool = True,
    colorize: bool = True,
    label: str = None,
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
) -> None:
    """Print logit validation results"""
    # Use defaults if not provided
    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    title = (
        f"LOGIT VALIDATION RESULTS ({label})" if label else "LOGIT VALIDATION RESULTS"
    )
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        _format_thresholds_legend(
            tol_map,
            divergence_difference_tol,
            colorize,
            divergence_n_ulps=divergence_n_ulps,
            mantissa_bits=mantissa_bits,
        )
    )
    print()
    print(
        _format_token_results_table(
            results, divergence_difference_tol, tol_map, suppress_passing, colorize
        )
    )
    print()
    print(
        _format_validation_summary(
            results,
            tol_map,
            divergence_difference_tol,
            colorize,
            divergence_n_ulps=divergence_n_ulps,
        )
    )


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check if output is being redirected
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for common terminals that support color
    term = os.environ.get("TERM", "").lower()
    if "color" in term or term in ["xterm", "xterm-256color", "screen", "linux"]:
        return True

    return False


def _colorize_text(text: str, color: str, colorize: bool = True) -> str:
    """Apply color to text if colors are supported by the terminal and colorize is True."""
    if not colorize or not _supports_color():
        return text

    color_code = getattr(_Colors, color.upper(), "")
    if color_code:
        return f"{color_code}{text}{_Colors.RESET}"
    return text


def multi_prompt_logit_validation(
    prompts_input_ids: List[List[List[int]]],
    generate_fn: Callable[
        [torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ],
    prompts_expected_logits: List[torch.Tensor],
    prompts_baseline_logits: List[torch.Tensor] = None,
    aggregate_config: dict = None,
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    suppress_passing: bool = True,
    colorize: bool = True,
    test_device: str = "cpu",
    replicate_to_batch_size: int = None,
    divergence_n_ulps: Optional[int] = 1,
    mantissa_bits: int = 7,
    prompts_multimodal_inputs: Optional[List[List[dict]]] = None,
) -> MultiPromptValidationResult:
    """Validate logits across multiple prompts with aggregate thresholds.

    Runs logit_validation for each prompt and computes cross-prompt aggregate
    metrics. When baseline_logits are provided, enables three-way validation
    mode with per-token BC aggregation across prompts.

    Args:
        prompts_input_ids: List of input_ids per prompt, each as List[List[int]]
            for batch processing.
        generate_fn: Function that generates logits from input_ids.
        prompts_expected_logits: List of expected logits tensors, one per prompt.
            Shape: (seq_len, batch_size, vocab_size).
        prompts_baseline_logits: Optional list of FP32 baseline logits for
            three-way validation. Shape must match prompts_expected_logits.
        aggregate_config: Configuration for aggregate thresholds. See
            DEFAULT_AGGREGATE_CONFIG for available options.
        tol_map: Tolerance map passed to logit_validation.
        divergence_difference_tol: Fixed divergence tolerance, used only when
            divergence_n_ulps is None. Passed to logit_validation.
        divergence_n_ulps: Number of ULPs to allow for divergence difference.
            Defaults to 1 (dynamic, ULP-aware). Set to None to fall back to
            divergence_difference_tol.
        mantissa_bits: Mantissa bits for ULP calculation. Defaults to 7 (BF16).
        suppress_passing: If True, only print failing tokens.
        colorize: If True, colorize output based on thresholds.
        test_device: Device to run validation on. Either "cpu" or "neuron".
            When "cpu", uses a CPU-based allclose implementation. When "neuron",
            uses neuron_allclose from libtorch_neuronx_lite. Defaults to "cpu".
        replicate_to_batch_size: If set, replicates each prompt's input_ids
            and logits to this batch size before validation. This simulates
            batched inference without requiring callers to manually duplicate
            inputs. Defaults to None (no replication).
        prompts_multimodal_inputs: Optional list of multimodal inputs, one per
            prompt. Each element is a list of dicts (one per batch element)
            containing modality-specific data (e.g., pixel values for images).
            When provided, the corresponding entry is passed to
            ``logit_validation`` for each prompt. Defaults to None (text-only).

    Returns:
        MultiPromptValidationResult containing:
            - passed: True if all prompts pass two-way validation
            - per_prompt_results: List of (passed, results) tuples per prompt
            - aggregate_metrics: Cross-prompt aggregate threshold results

    Example:
        >>> prompts_input_ids = [tokenizer([p], return_tensors="pt")["input_ids"].tolist()
        ...                      for p in ["Hello", "World"]]
        >>> prompts_expected = [generate_ref_logits(model, ids) for ids in prompts_input_ids]
        >>> result = multi_prompt_logit_validation(
        ...     prompts_input_ids=prompts_input_ids,
        ...     generate_fn=neuron_generate_fn,
        ...     prompts_expected_logits=prompts_expected,
        ...     replicate_to_batch_size=4,
        ... )
        >>> print(f"Passed: {result.passed}")
    """
    if aggregate_config is None:
        aggregate_config = DEFAULT_AGGREGATE_CONFIG

    n_prompts = len(prompts_input_ids)
    three_way_mode = prompts_baseline_logits is not None

    if three_way_mode:
        assert len(prompts_baseline_logits) == n_prompts

    # Run validation for each prompt
    per_prompt_results = []
    for i in range(n_prompts):
        input_ids_copy = [list(batch) for batch in prompts_input_ids[i]]
        baseline = prompts_baseline_logits[i] if three_way_mode else None
        expected = prompts_expected_logits[i]
        mm_inputs = (
            prompts_multimodal_inputs[i]
            if prompts_multimodal_inputs is not None
            else None
        )

        # Replicate to simulate batched inference (per-prompt to reduce memory)
        if replicate_to_batch_size is not None and replicate_to_batch_size > 1:
            assert expected.shape[1] == 1, (
                f"replicate_to_batch_size requires batch_size=1 in expected_logits, "
                f"got shape {expected.shape}"
            )

            bs = replicate_to_batch_size
            input_ids_copy = [list(ids) for _ in range(bs) for ids in input_ids_copy]
            expected = expected.repeat(1, bs, 1)
            if baseline is not None:
                assert baseline.shape[1] == 1, (
                    f"replicate_to_batch_size requires batch_size=1 in baseline_logits, "
                    f"got shape {baseline.shape}"
                )
                baseline = baseline.repeat(1, bs, 1)
            # Replicate multimodal_inputs to match batch size
            if mm_inputs is not None:
                assert len(mm_inputs) == 1, (
                    f"replicate_to_batch_size requires batch_size=1 in multimodal_inputs, "
                    f"got length {len(mm_inputs)}"
                )
                mm_inputs = mm_inputs * bs

        result = logit_validation(
            input_ids=input_ids_copy,
            generate_fn=generate_fn,
            expected_logits=expected,
            baseline_logits=baseline,
            tol_map=tol_map,
            divergence_difference_tol=divergence_difference_tol,
            suppress_passing=suppress_passing,
            colorize=colorize,
            label=f"Prompt {i + 1}/{n_prompts}",
            test_device=test_device,
            divergence_n_ulps=divergence_n_ulps,
            mantissa_bits=mantissa_bits,
            multimodal_inputs=mm_inputs,
        )

        if three_way_mode:
            passed, results = result
        else:
            passed, results = result, None

        per_prompt_results.append((passed, results))

    # Compute and print aggregate metrics
    aggregate_metrics = {}
    overall_passed = all(p for p, _ in per_prompt_results)

    if three_way_mode:
        aggregate_metrics = _compute_aggregate_metrics(
            per_prompt_results, aggregate_config
        )
        # σ-ratio pass: if aggregate σ-ratio ≤ threshold, target error is
        # within acceptable range of the dtype baseline — pass regardless of
        # two-way results. Also pass if aggregate BC ≥ threshold.
        bc_pass = aggregate_metrics.get("agg_bc") == "PASS"
        sigma_ratio = aggregate_metrics.get("agg_sigma_ratio", float("inf"))
        sigma_thresh = aggregate_config.get("agg_sigma_ratio_threshold", 1.0)
        sigma_pass = sigma_ratio <= sigma_thresh
        overall_passed = overall_passed or sigma_pass or bc_pass
        _print_aggregate_summary(
            per_prompt_results, aggregate_metrics, colorize, aggregate_config
        )

    return MultiPromptValidationResult(
        passed=overall_passed,
        per_prompt_results=per_prompt_results,
        aggregate_metrics=aggregate_metrics,
    )


def _print_three_way_metrics(
    results: List[List[dict]], label: str = "", colorize: bool = True
):
    """Print three-way metrics summary table for a single prompt."""
    if not results or not results[0]:
        return

    print(f"\n{'=' * 80}")
    print(f"THREE-WAY METRICS ({label})")
    print("=" * 80)

    # Print legend
    print("\nMetric Definitions:")
    print("  L-inf: max|error| / max|FP32|  (normalized max error)")
    print("  L2:    norm(error) / norm(FP32) (normalized RMS error)")
    print("  Base:  Same-dtype baseline (e.g., BF16 on CPU) vs FP32")
    print("  Tgt:   Target (Neuron) vs FP32")
    print("  Ratio: Tgt/Base - values near 1.0 mean target matches dtype baseline")
    print(
        "  BC:    Bhattacharyya Coefficient - error distribution similarity (1.0 = identical)"
    )
    print("  Gen Token: Generation step index (0 = first generated token after prompt)")
    print("  Div:   * indicates divergent token (actual argmax != expected argmax)")
    print()

    print(
        f"{'Gen Token':>9} | {'Base L-inf':>10} {'Tgt L-inf':>10} {'Ratio':>7} | {'Base L2':>10} {'Tgt L2':>10} {'Ratio':>7} | {'BC':>7} | Div"
    )
    print("-" * 95)

    linf_thresh = DEFAULT_DYNAMIC_THRESHOLD_CONFIG["linf_multiplier"]
    l2_thresh = DEFAULT_DYNAMIC_THRESHOLD_CONFIG["l2_multiplier"]
    bc_thresh = DEFAULT_AGGREGATE_CONFIG["agg_bc_threshold"]

    for i, token_result in enumerate(results[0]):
        if "three_way_metrics" in token_result:
            m = token_result["three_way_metrics"]
            linf_ratio = (
                m["tgt_linf"] / m["base_linf"] if m["base_linf"] > 0 else float("inf")
            )
            l2_ratio = m["tgt_l2"] / m["base_l2"] if m["base_l2"] > 0 else float("inf")

            linf_color = "GREEN" if linf_ratio < linf_thresh else "RED"
            l2_color = "GREEN" if l2_ratio < l2_thresh else "RED"
            bc_color = "GREEN" if m["bc"] >= bc_thresh else "RED"

            linf_ratio_str = _colorize_text(
                f"{linf_ratio:>6.2f}x", linf_color, colorize
            )
            l2_ratio_str = _colorize_text(f"{l2_ratio:>6.2f}x", l2_color, colorize)
            bc_str = _colorize_text(f"{m['bc']:>7.4f}", bc_color, colorize)
            div_str = "  *" if token_result.get("divergence", False) else ""

            print(
                f"{i:>9} | {m['base_linf']:>10.4f} {m['tgt_linf']:>10.4f} {linf_ratio_str} | "
                f"{m['base_l2']:>10.4f} {m['tgt_l2']:>10.4f} {l2_ratio_str} | {bc_str} |{div_str}"
            )


def _print_aggregate_summary(
    per_prompt_results: List[Tuple[bool, List[List[dict]]]],
    aggregate_metrics: Dict[str, Any],
    colorize: bool = True,
    aggregate_config: dict = None,
):
    """Print aggregate summary across all prompts."""
    print(f"\n{'=' * 80}")
    print("AGGREGATE SUMMARY ACROSS ALL PROMPTS")
    print("=" * 80)

    n_prompts = len(per_prompt_results)
    n_passed = sum(1 for p, _ in per_prompt_results if p)
    print(f"\nTwo-way validation: {n_passed}/{n_prompts} prompts passed")

    # Per generation token cross-prompt summary table
    # TODO: Also print per absolute token index summary (accounting for different prompt lengths)
    per_token_bc = aggregate_metrics.get("per_token_bc", [])
    if per_token_bc:
        print(f"\n{'-' * 80}")
        print("Per Generation Token Cross-Prompt Summary")
        print("-" * 80)
        print(
            "  Token index = generation step (0 = first generated token after prompt)"
        )
        print(
            "  Max errors taken across all prompts for each generation token position"
        )
        print("  BC computed from concatenated error arrays across all prompts")
        print()
        print(
            f"{'Gen Token':>9} | {'Max Base L-inf':>12} {'Max Tgt L-inf':>12} {'Ratio':>7} | "
            f"{'Max Base L2':>11} {'Max Tgt L2':>11} {'Ratio':>7} | {'BC':>7}"
        )
        print("-" * 99)

        # Collect per-token max metrics across prompts
        all_prompt_metrics = []
        for _, results in per_prompt_results:
            if results:
                all_prompt_metrics.append(results)

        linf_thresh = DEFAULT_DYNAMIC_THRESHOLD_CONFIG["linf_multiplier"]
        l2_thresh = DEFAULT_DYNAMIC_THRESHOLD_CONFIG["l2_multiplier"]
        bc_thresh = DEFAULT_AGGREGATE_CONFIG["agg_bc_threshold"]

        n_tokens = len(per_token_bc)
        for tok_idx in range(n_tokens):
            max_base_linf = max(
                pm[0][tok_idx]["three_way_metrics"]["base_linf"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            max_tgt_linf = max(
                pm[0][tok_idx]["three_way_metrics"]["tgt_linf"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            max_base_l2 = max(
                pm[0][tok_idx]["three_way_metrics"]["base_l2"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            max_tgt_l2 = max(
                pm[0][tok_idx]["three_way_metrics"]["tgt_l2"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )

            linf_ratio = (
                max_tgt_linf / max_base_linf if max_base_linf > 0 else float("inf")
            )
            l2_ratio = max_tgt_l2 / max_base_l2 if max_base_l2 > 0 else float("inf")
            bc = per_token_bc[tok_idx]

            linf_color = "GREEN" if linf_ratio < linf_thresh else "RED"
            l2_color = "GREEN" if l2_ratio < l2_thresh else "RED"
            bc_color = "GREEN" if bc >= bc_thresh else "RED"

            linf_ratio_str = _colorize_text(
                f"{linf_ratio:>6.2f}x", linf_color, colorize
            )
            l2_ratio_str = _colorize_text(f"{l2_ratio:>6.2f}x", l2_color, colorize)
            bc_str = _colorize_text(f"{bc:>7.4f}", bc_color, colorize)

            print(
                f"{tok_idx:>9} | {max_base_linf:>12.4f} {max_tgt_linf:>12.4f} {linf_ratio_str} | "
                f"{max_base_l2:>11.4f} {max_tgt_l2:>11.4f} {l2_ratio_str} | {bc_str}"
            )

    # Final aggregate thresholds
    print(f"\n{'=' * 80}")
    print("AGGREGATE THRESHOLD RESULTS (Informational)")
    print("=" * 80)
    print("  pp_*: Per-prompt thresholds (X/N prompts passing)")
    print("  agg_*: Cross-prompt aggregate thresholds")

    # Group metrics by category
    print("\nPer-Prompt Thresholds:")
    for key in sorted(aggregate_metrics.keys()):
        if key.startswith("pp_"):
            print(f"  {key}: {aggregate_metrics[key]}")

    print("\nCross-Prompt Aggregate Thresholds:")
    for key in sorted(aggregate_metrics.keys()):
        if key.startswith("agg_") and key not in ("per_token_bc", "agg_sigma_ratio"):
            status = aggregate_metrics[key]
            color = "GREEN" if status == "PASS" else "RED"
            print(f"  {key}: {_colorize_text(status, color, colorize)}")

    sigma_ratio = aggregate_metrics.get("agg_sigma_ratio", float("inf"))
    if aggregate_config is None:
        aggregate_config = DEFAULT_AGGREGATE_CONFIG
    sigma_thresh = aggregate_config.get("agg_sigma_ratio_threshold", 1.0)
    sigma_color = (
        "GREEN"
        if sigma_ratio <= sigma_thresh
        else ("YELLOW" if sigma_ratio <= 1.5 else "RED")
    )
    print(
        f"  agg_sigma_ratio: {_colorize_text(f'{sigma_ratio:.4f}', sigma_color, colorize)} (≤{sigma_thresh} = pass)"
    )


def _compute_aggregate_metrics(
    per_prompt_results: List[Tuple[bool, List[List[dict]]]],
    config: dict,
) -> Dict[str, Any]:
    """Compute aggregate metrics across all prompts.

    Returns dict with:
    - pp_static_*: per-prompt static threshold results (count passed / total)
    - pp_max_*: per-prompt max-based dynamic threshold results
    - pp_tok_*: per-prompt per-token dynamic threshold results
    - agg_bc: aggregate BC threshold result (PASS/FAIL)
    - agg_linf_*: aggregate per-token linf ratio results
    - agg_l2_*: aggregate per-token l2 ratio results
    """
    metrics = {}
    n_prompts = len(per_prompt_results)

    # Collect three-way metrics from all prompts
    all_prompt_metrics = []  # [prompt][batch][token] -> three_way_metrics
    for _, results in per_prompt_results:
        if results is None:
            continue
        prompt_metrics = []
        for batch_results in results:
            batch_metrics = []
            for token_result in batch_results:
                if "three_way_metrics" in token_result:
                    batch_metrics.append(token_result["three_way_metrics"])
            prompt_metrics.append(batch_metrics)
        all_prompt_metrics.append(prompt_metrics)

    if not all_prompt_metrics:
        return metrics

    # Per-prompt static thresholds: max tgt_linf < threshold
    for thresh in config.get("pp_static_thresholds", []):
        count = 0
        for prompt_metrics in all_prompt_metrics:
            max_tgt_linf = (
                max(m["tgt_linf"] for batch in prompt_metrics for m in batch)
                if prompt_metrics and prompt_metrics[0]
                else 0
            )
            if max_tgt_linf < thresh:
                count += 1
        metrics[f"pp_static_{thresh}"] = f"{count}/{n_prompts}"

    # Per-prompt max-based dynamic thresholds
    for mult in config.get("pp_linf_multipliers", []):
        count = 0
        for prompt_metrics in all_prompt_metrics:
            if not prompt_metrics or not prompt_metrics[0]:
                continue
            max_base = max(m["base_linf"] for batch in prompt_metrics for m in batch)
            max_tgt = max(m["tgt_linf"] for batch in prompt_metrics for m in batch)
            if max_tgt < mult * max_base:
                count += 1
        metrics[f"pp_max_linf_{mult}x"] = f"{count}/{n_prompts}"

    for mult in config.get("pp_l2_multipliers", []):
        count = 0
        for prompt_metrics in all_prompt_metrics:
            if not prompt_metrics or not prompt_metrics[0]:
                continue
            max_base = max(m["base_l2"] for batch in prompt_metrics for m in batch)
            max_tgt = max(m["tgt_l2"] for batch in prompt_metrics for m in batch)
            if max_tgt < mult * max_base:
                count += 1
        metrics[f"pp_max_l2_{mult}x"] = f"{count}/{n_prompts}"

    # Per-prompt per-token thresholds: all tokens must satisfy condition
    for mult in config.get("pp_tok_linf_multipliers", []):
        count = 0
        for prompt_metrics in all_prompt_metrics:
            if not prompt_metrics or not prompt_metrics[0]:
                continue
            all_pass = all(
                m["tgt_linf"] < mult * m["base_linf"]
                for batch in prompt_metrics
                for m in batch
            )
            if all_pass:
                count += 1
        metrics[f"pp_tok_linf_{mult}x"] = f"{count}/{n_prompts}"

    for mult in config.get("pp_tok_l2_multipliers", []):
        count = 0
        for prompt_metrics in all_prompt_metrics:
            if not prompt_metrics or not prompt_metrics[0]:
                continue
            all_pass = all(
                m["tgt_l2"] < mult * m["base_l2"]
                for batch in prompt_metrics
                for m in batch
            )
            if all_pass:
                count += 1
        metrics[f"pp_tok_l2_{mult}x"] = f"{count}/{n_prompts}"

    # Aggregate BC: per-token BC from concatenated errors across all prompts
    # This provides statistically robust BC values by pooling ~vocab_size * n_prompts errors
    bc_thresh = config.get("agg_bc_threshold", 0.95)
    n_tokens = (
        len(all_prompt_metrics[0][0])
        if all_prompt_metrics and all_prompt_metrics[0]
        else 0
    )

    all_bc_pass = True
    per_token_bc = []
    for tok_idx in range(n_tokens):
        # Collect error arrays from all prompts for this token position
        base_arrs = []
        tgt_arrs = []
        for pm in all_prompt_metrics:
            if pm and len(pm[0]) > tok_idx:
                m = pm[0][tok_idx]
                if m.get("base_errors") is not None:
                    base_arrs.append(m["base_errors"])
                if m.get("tgt_errors") is not None:
                    tgt_arrs.append(m["tgt_errors"])

        if base_arrs and tgt_arrs:
            # Concatenate errors across prompts and compute BC
            bc = _compute_bc(np.concatenate(base_arrs), np.concatenate(tgt_arrs))
            per_token_bc.append(bc)
            if bc < bc_thresh:
                all_bc_pass = False
        else:
            per_token_bc.append(0.0)
            all_bc_pass = False

    metrics["agg_bc"] = "PASS" if all_bc_pass else "FAIL"
    metrics["per_token_bc"] = per_token_bc

    # Aggregate σ-ratio: RMS of all target errors / RMS of all baseline errors
    # σ-ratio ≤ 1.0 means target is more accurate than the dtype baseline
    all_base_errors = []
    all_tgt_errors = []
    for pm in all_prompt_metrics:
        for batch in pm:
            for m in batch:
                if m.get("base_errors") is not None:
                    all_base_errors.append(m["base_errors"])
                if m.get("tgt_errors") is not None:
                    all_tgt_errors.append(m["tgt_errors"])

    if all_base_errors and all_tgt_errors:
        cat_base = np.concatenate(all_base_errors)
        cat_tgt = np.concatenate(all_tgt_errors)
        base_rms = np.sqrt(np.mean(cat_base**2))
        tgt_rms = np.sqrt(np.mean(cat_tgt**2))
        sigma_ratio = float(tgt_rms / base_rms) if base_rms > 0 else float("inf")
    else:
        sigma_ratio = float("inf")
    metrics["agg_sigma_ratio"] = sigma_ratio

    # Aggregate per-token ratio thresholds
    # Group metrics by token position across all prompts
    n_tokens = (
        len(all_prompt_metrics[0][0])
        if all_prompt_metrics and all_prompt_metrics[0]
        else 0
    )

    for mult in config.get("agg_linf_multipliers", []):
        all_tokens_pass = True
        for tok_idx in range(n_tokens):
            max_base = max(
                pm[0][tok_idx]["base_linf"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            max_tgt = max(
                pm[0][tok_idx]["tgt_linf"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            if max_tgt >= mult * max_base:
                all_tokens_pass = False
                break
        metrics[f"agg_linf_{mult}x"] = "PASS" if all_tokens_pass else "FAIL"

    for mult in config.get("agg_l2_multipliers", []):
        all_tokens_pass = True
        for tok_idx in range(n_tokens):
            max_base = max(
                pm[0][tok_idx]["base_l2"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            max_tgt = max(
                pm[0][tok_idx]["tgt_l2"]
                for pm in all_prompt_metrics
                if pm and len(pm[0]) > tok_idx
            )
            if max_tgt >= mult * max_base:
                all_tokens_pass = False
                break
        metrics[f"agg_l2_{mult}x"] = "PASS" if all_tokens_pass else "FAIL"

    return metrics


def create_offline_vllm_generate_fn(
    llm: "LLM", output_length: int, label: str = ""
) -> Callable[[list[list[int]]], tuple[torch.Tensor, torch.Tensor]]:
    """Build a ``generate_fn`` that returns dense per-token logits from a vLLM ``LLM``.

    Wraps a running vLLM ``LLM`` in the callable expected by
    :func:`logit_validation` and :func:`multi_prompt_logit_validation`. The
    returned function greedily decodes ``output_length`` tokens per prompt
    (``temperature=0``) with full-vocab logprobs, then packs those logprobs into
    a dense logits tensor so they can be compared token-for-token against a
    HuggingFace reference.

    The ``LLM`` must be created with logprobs enabled (e.g.
    ``max_logprobs=-1`` / ``logprobs_mode="raw_logits"``); positions with no
    logprob are filled with ``-inf``.

    Args:
        llm: A constructed vLLM ``LLM`` instance to generate from.
        output_length: Number of tokens to decode per prompt.
        label: Optional tag; when set, each completion is printed prefixed with
            ``[label]`` for debugging.

    Returns:
        A ``generate_fn(input_ids)`` where ``input_ids`` is a list of tokenized
        prompts. It returns a tuple ``(logits, sampled_token_ids)``:

        - ``logits``: ``torch.Tensor`` of shape ``(output_length, num_prompts, vocab)``.
        - ``sampled_token_ids``: ``torch.Tensor`` of shape ``(num_prompts, output_length)``.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def generate_fn(
        input_ids: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In CPU compile mode, LLM() has already compiled every graph; inference
        # can't run without Neuron devices, so signal compilation-complete instead
        # of calling llm.generate() (the conftest hook then marks the test PASS).
        if envs.VLLM_NEURON_CPU_COMPILE:
            from vllm_neuron.exceptions import CPUCompilationComplete

            raise CPUCompilationComplete(
                "CPU compile mode: offline compilation complete, skipping generate()."
            )
        # Pass input_ids directly to vLLM - no decode round-trip
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
        model_vocab_size = llm.get_tokenizer().vocab_size
        sampling_params = SamplingParams(
            temperature=0, max_tokens=output_length, logprobs=model_vocab_size
        )
        outputs = llm.generate(prompts, sampling_params)

        if label:
            for i, output in enumerate(outputs):
                for completion in output.outputs:
                    print(f"[{label}] Prompt {i}: {completion.text}")

        batch_logits = []
        batch_tokens = []
        for output in outputs:
            sequence_logits = []
            for completion in output.outputs:
                for logprob_dict in completion.logprobs:
                    actual_max_id = max(logprob_dict.keys())
                    token_tensor = torch.full(
                        (max(model_vocab_size, actual_max_id + 1),), -float("inf")
                    )
                    for tid, lp in logprob_dict.items():
                        token_tensor[tid] = lp.logprob
                    sequence_logits.append(token_tensor)
            if sequence_logits:
                batch_logits.append(torch.stack(sequence_logits, dim=0))
            # Extract actual sampled token IDs
            batch_tokens.append(
                torch.tensor(
                    [tid for tid in output.outputs[0].token_ids[:output_length]],
                    dtype=torch.long,
                )
            )
        return torch.stack(batch_logits, dim=1), torch.stack(batch_tokens, dim=0)

    return generate_fn
