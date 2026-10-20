# SPDX-License-Identifier: Apache-2.0
"""Accuracy validation toolkit for vLLM Neuron.

Public API organized by domain:

**Logit validation** (``vllm_neuron.accuracy.logit_validation``):
    E2E logit comparison between vLLM Neuron and HuggingFace reference.

    - :func:`logit_validation` — single-prompt logit comparison
    - :func:`multi_prompt_logit_validation` — multi-prompt with aggregation

**Tensor capture** (``vllm_neuron.accuracy.tensor_capture``):
    Instrument model forward passes to capture intermediate tensors.

    - :class:`ModelCapture` — hook-based tensor capture (no model wrapping)
    - :func:`capture_tensor` — decorator to mark tensors for capture

**Tensor comparison** (``vllm_neuron.accuracy.tensor_compare``):
    Compare captured tensors between implementations.

    - :func:`compare_tensors` — two-way tensor comparison
    - :func:`compare_tensors_three_way` — three-way tensor comparison
    - :func:`compare_captures_two_way` — two-way comparison of captured forward passes
    - :func:`compare_captures_three_way` — three-way comparison of captured forward passes

**KV cache analysis** (``vllm_neuron.accuracy.kv_cache_analysis``):
    Extract and compare KV caches between HF and vLLM.

    - :func:`compare_kv_caches` — compare KV caches with per-head metrics
    - :func:`extract_hf_kv_caches` — extract KV caches from HF model
    - :func:`extract_vllm_kv_caches` — extract KV caches from vLLM

**Encoder cache analysis** (``vllm_neuron.accuracy.encoder_cache_analysis``):
    Extract and compare vision encoder outputs between HF and vLLM.

    - :func:`extract_vllm_encoder_cache` — extract the encoder cache from vLLM
    - :func:`compare_encoder_caches` — compare encoder outputs by mm_hash
    - :func:`compare_encoder_caches_by_index` — compare encoder outputs by order

**Tensor I/O** (``vllm_neuron.accuracy.tensor_io``):
    Read/write captured tensors to disk.

    - :func:`tensor_io_read` — read captured forward pass from disk
    - :func:`tensor_io_write` — write captured forward pass to disk

**Visualization** (``vllm_neuron.accuracy.plotting``,
``vllm_neuron.accuracy.logit_visualization``,
``vllm_neuron.accuracy.kv_cache_visualize``):
    Plotting and HTML report generation for debugging accuracy issues.

**Testing assertions** (``vllm_neuron.accuracy.testing``):
    Two-way and three-way numerical comparison for module-level tests.

    - :func:`assert_close` — assert target ≈ expected (two-way)
    - :func:`assert_close_three_way` — assert target ≈ expected with
      baseline precision floor (three-way)
"""

# ── Logit validation ─────────────────────────────────────────────────────
from vllm_neuron.accuracy.logit_validation import (
    DEFAULT_AGGREGATE_CONFIG,
    DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
    DEFAULT_DYNAMIC_THRESHOLD_CONFIG,
    DEFAULT_TOLERANCE_MAP,
    MultiPromptValidationResult,
    ThreeWayTokenMetrics,
    logit_validation,
    multi_prompt_logit_validation,
)
from vllm_neuron.accuracy.tensor_alignment_utils import (  # noqa: F401
    TAKE_LAST_MODULES,
    align_and_truncate_hidden,
    align_decode_captures,
    count_real_tokens,
    get_seq_dim_size,
    hf_reference_reconstruction,
    slice_token,
)

# ── Tensor capture ───────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_capture import (
    CaptureWriter,
    ModelCapture,
    TensorRegistry,
    capture_tensor,
    expand_patterns,
)

# ── Tensor comparison ────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_compare import (
    AggregateMetrics,
    AlignmentFn,
    ComparisonResult,
    DynamicThresholdConfig,
    ReconstructionFn,
    ThreeWayComparisonResult,
    ThreeWayResults,
    TwoWayResults,
    compare_capture_dirs,
    compare_capture_dirs_three_way,
    compare_captures_three_way,
    compare_captures_two_way,
    compare_tensors,
    compare_tensors_three_way,
    compute_aggregate_metrics,
    print_comparison_report,
    print_three_way_report,
    align_shapes,
)

# ── KV cache analysis ───────────────────────────────────────────────────
from vllm_neuron.accuracy.kv_cache_analysis import (
    compare_kv_caches,
    extract_hf_kv_caches,
    extract_hf_kv_caches_teacher_forced,
    extract_vllm_kv_caches,
    extract_vllm_kv_cache_config,
    extract_vllm_block_tables,
    cleanup_kv_snapshot,
    reconstruct_contiguous_kv,
    reconstruct_contiguous_kv_batch,
    aggregate_kv_bc_across_prompts,
    compute_combined_bc_per_layer,
    generate_hf_logits_and_kv,
    print_kv_report,
)

# ── Encoder cache analysis ──────────────────────────────────────────────
from vllm_neuron.accuracy.encoder_cache_analysis import (
    EncoderCacheMetrics,
    extract_vllm_encoder_cache,
    enable_encoder_cache_snapshot,
    cleanup_encoder_cache_snapshot,
    extract_hf_encoder_outputs,
    compare_encoder_caches,
    compare_encoder_caches_by_index,
    print_encoder_cache_report,
)

# ── Tensor I/O ───────────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_io import (
    read as tensor_io_read,
    write as tensor_io_write,
)

# ── Tensor replacement ───────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_replacement import (
    TensorReplacer,
    set_active_context,
    get_replacement_tensor,
)

# ── Visualization ────────────────────────────────────────────────────────
from vllm_neuron.accuracy.plotting import (
    plot_error_distributions,
    plot_error_qqplot,
    plot_scatter,
    plot_three_way,
)

# ── Testing assertions ───────────────────────────────────────────────────
from vllm_neuron.accuracy.testing import (
    AssertCloseResult,
    CosineThreeWayAssertResult,
    ThreeWayAssertResult,
    assert_close,
    assert_close_three_way,
    assert_pooling_close_three_way,
)

# ── Tensor histogram diagnostics ─────────────────────────────────────────
from vllm_neuron.accuracy.tensor_histogram import TensorHistogram

# ── HF reference model / goldens ─────────────────────────────────────────
from vllm_neuron.accuracy.logit_validation import (
    create_offline_vllm_generate_fn,
)
from vllm_neuron.accuracy.goldens.reference_model import init_hf_model
from vllm_neuron.accuracy.goldens.reference_logits import (
    generate_three_way_reference_logits,
    generate_reference_logits,
    generate_teacher_forced_logits,
    generate_teacher_forced_logits_fp8,
    get_cached_reference_goldens,
)
from vllm_neuron.accuracy.goldens.fp8_kv_golden import (
    hf_simulate_fp8_kv_cache,
    load_kv_scales,
)

__all__ = [
    # Logit validation
    "logit_validation",
    "multi_prompt_logit_validation",
    "DEFAULT_AGGREGATE_CONFIG",
    "DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE",
    "DEFAULT_DYNAMIC_THRESHOLD_CONFIG",
    "DEFAULT_TOLERANCE_MAP",
    "MultiPromptValidationResult",
    "ThreeWayTokenMetrics",
    # Tensor capture
    "ModelCapture",
    "CaptureWriter",
    "TensorRegistry",
    "capture_tensor",
    "expand_patterns",
    # Tensor alignment
    "TAKE_LAST_MODULES",
    "align_and_truncate_hidden",
    "align_decode_captures",
    "count_real_tokens",
    "get_seq_dim_size",
    "hf_reference_reconstruction",
    "slice_token",
    # Tensor comparison
    "compare_tensors",
    "compare_tensors_three_way",
    "compare_capture_dirs",
    "compare_capture_dirs_three_way",
    "compare_captures_two_way",
    "compare_captures_three_way",
    "print_comparison_report",
    "print_three_way_report",
    "compute_aggregate_metrics",
    "align_shapes",
    "AlignmentFn",
    "ReconstructionFn",
    "ComparisonResult",
    "DynamicThresholdConfig",
    "ThreeWayComparisonResult",
    "AggregateMetrics",
    "TwoWayResults",
    "ThreeWayResults",
    # KV cache analysis
    "compare_kv_caches",
    "extract_hf_kv_caches",
    "extract_hf_kv_caches_teacher_forced",
    "extract_vllm_kv_caches",
    "extract_vllm_kv_cache_config",
    "extract_vllm_block_tables",
    "cleanup_kv_snapshot",
    "reconstruct_contiguous_kv",
    "reconstruct_contiguous_kv_batch",
    "aggregate_kv_bc_across_prompts",
    "compute_combined_bc_per_layer",
    "generate_hf_logits_and_kv",
    "print_kv_report",
    # Encoder cache analysis
    "EncoderCacheMetrics",
    "extract_vllm_encoder_cache",
    "enable_encoder_cache_snapshot",
    "cleanup_encoder_cache_snapshot",
    "extract_hf_encoder_outputs",
    "compare_encoder_caches",
    "compare_encoder_caches_by_index",
    "print_encoder_cache_report",
    # Tensor I/O
    "tensor_io_read",
    "tensor_io_write",
    # Tensor replacement
    "TensorReplacer",
    "set_active_context",
    "get_replacement_tensor",
    # Visualization
    "plot_error_distributions",
    "plot_error_qqplot",
    "plot_scatter",
    "plot_three_way",
    # Testing assertions
    "assert_close",
    "assert_close_three_way",
    "assert_pooling_close_three_way",
    "AssertCloseResult",
    "CosineThreeWayAssertResult",
    # Tensor histogram diagnostics
    "TensorHistogram",
    "ThreeWayAssertResult",
    # HF reference model / goldens
    "generate_three_way_reference_logits",
    "create_offline_vllm_generate_fn",
    "get_cached_reference_goldens",
    "generate_reference_logits",
    "generate_teacher_forced_logits",
    "init_hf_model",
    "generate_teacher_forced_logits_fp8",
    "hf_simulate_fp8_kv_cache",
    "load_kv_scales",
]
