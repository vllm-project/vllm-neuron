# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for vLLM-Neuron.

Metrics are defined at module scope so they auto-register with the
``prometheus_client`` global default registry. vLLM serves this registry
at ``/metrics``, so any metric created here appears automatically.

All metrics use the ``vllm_neuron:`` prefix to avoid collisions with
built-in ``vllm:`` metrics.

Usage example::

    from vllm_neuron.metrics import NUM_SEQS_PADDING

    NUM_SEQS_PADDING.labels(
        model_name="meta-llama/Llama-3.1-8B",
        bucket_name="prefill_s128",
    ).observe(4)
"""

from prometheus_client import Counter, Gauge, Histogram

NUM_SEQS_PADDING = Histogram(
    "vllm_neuron:num_seqs_padding",
    "Number of padded batch lines processed by the model.",
    labelnames=["model_name", "bucket_name"],
    buckets=[2**i for i in range(8)],  # [1, 2, 4, 8, 16, 32, 64, 128]
)

NUM_BATCHED_TOKENS_PADDING = Histogram(
    "vllm_neuron:num_batched_tokens_padding",
    "Number of padded sequence length processed by the model.",
    labelnames=["model_name", "bucket_name"],
    buckets=[2**i for i in range(5, 15)],  # [32, 64, 128, ..., 16384]
)

NEFF_EXECUTION_COUNT = Counter(
    "vllm_neuron:neff_execution_count",
    "Number of NEFF executions.",
    labelnames=["model_name", "bucket_name"],
)

COMPILATION_TIME = Gauge(
    "vllm_neuron:compilation_time_seconds",
    "Time spent compiling Neuron graphs (FX trace + neuronxcc compile). Includes compile cache hits.",
    labelnames=["model_name", "bucket_name"],
    multiprocess_mode="max",
)

STARTUP_TIME = Gauge(
    "vllm_neuron:startup_time_seconds",
    "Total server startup time from worker spawn to ready.",
    labelnames=["model_name"],
    multiprocess_mode="max",
)

MODEL_LOAD_TIME = Gauge(
    "vllm_neuron:model_load_time_seconds",
    "Time spent loading model weights to device (host to HBM transfer).",
    labelnames=["model_name"],
    multiprocess_mode="max",
)

MODEL_LOAD_SIZE = Gauge(
    "vllm_neuron:model_load_size_bytes",
    "Size of model weights transferred to device (host to HBM transfer).",
    labelnames=["model_name"],
    multiprocess_mode="sum",
)

DEVICE_ENCODER_CACHE_QUERIES = Counter(
    "vllm_neuron:device_encoder_cache_queries",
    "On-device encoder cache lookups (mm items the worker checked against the "
    "cache). Encoder cache hits = queries - misses.",
)

DEVICE_ENCODER_CACHE_MISSES = Counter(
    "vllm_neuron:device_encoder_cache_misses",
    "On-device encoder cache misses (encoder NEFF execution needed).",
)

DEVICE_ENCODER_CACHE_EVICTIONS = Counter(
    "vllm_neuron:device_encoder_cache_evictions",
    "On-device encoder cache blocks evicted (freed from the device buffer).",
)

# Occupancy gauges — point-in-time state of the on-device encoder cache. Each
# TP/DP worker sets its own value every step (SPMD, so all ranks update in
# lockstep); multiprocess_mode="sum" makes the exported value the fleet-wide
# total across ranks. Per-rank capacity is encoder_cache_num_blocks.
DEVICE_ENCODER_CACHE_ACTIVE_BLOCKS = Gauge(
    "vllm_neuron:device_encoder_cache_active_blocks",
    "On-device encoder cache blocks currently allocated to cached mm items "
    "(summed across ranks).",
    multiprocess_mode="sum",
)

DEVICE_ENCODER_CACHE_FREE_BLOCKS = Gauge(
    "vllm_neuron:device_encoder_cache_free_blocks",
    "On-device encoder cache blocks currently free/allocatable (summed across ranks).",
    multiprocess_mode="sum",
)

DEVICE_ENCODER_CACHE_HELD_BLOCKS = Gauge(
    "vllm_neuron:device_encoder_cache_held_blocks",
    "On-device encoder cache blocks freed but still within the min-hold-time "
    "window (summed across ranks).",
    multiprocess_mode="sum",
)

DEVICE_ENCODER_CACHE_ACTIVE_REAL_TOKENS = Gauge(
    "vllm_neuron:device_encoder_cache_active_real_tokens",
    "Real (non-padding) merged tokens stored across active encoder cache blocks "
    "(summed across ranks). Compare to active_blocks * block_size for padding "
    "waste.",
    multiprocess_mode="sum",
)

SCHEDULER_ENCODER_CACHE_QUERIES = Counter(
    "vllm_neuron:scheduler_encoder_cache_queries",
    "Scheduler EncoderCacheManager lookups (mm items checked before scheduling "
    "the encoder).",
)

SCHEDULER_ENCODER_CACHE_HITS = Counter(
    "vllm_neuron:scheduler_encoder_cache_hits",
    "Scheduler EncoderCacheManager hits (mm item already encoded, so the "
    "scheduler skips re-scheduling the encoder).",
)
