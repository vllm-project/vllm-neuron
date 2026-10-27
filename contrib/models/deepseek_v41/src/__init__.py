# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash on 16 Trainium 2 devices — package setup
============================================================

This port serves the checkpoint from **HBM**: 64 ranks, one per logical
NeuronCore of a trn2.48xlarge, with the routed experts partitioned by id and the
engram tables partitioned by row, so a forward pass reads no file. It descends
from a single-device ancestor that ran 4 ranks on one device and streamed both
weight families off disk every forward. That ancestor is not part of this
contribution, but it is where many of the measurements quoted in these comments
come from, and each such comment says which host it was measured on — worth
knowing before trusting a number here, because several of those premises
inverted when the deployment moved to 16 devices.

    ValueError: The checkpoint you are trying to load has model type
    `deepseek_v41` but Transformers does not recognize this architecture.
"""

from __future__ import annotations

import logging
import os
import sys

from transformers import AutoConfig, PretrainedConfig

from . import config as _config
from .factory import DeepseekV41ForCausalLM

logger = logging.getLogger(__name__)

os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
# The NKI trace cache keys a kernel on its own function source; the megakernels
# fold a hash of their helper modules into their constant flags for the compile
# cache, but the trace cache has no such hook, so it is off (it only costs a
# re-trace when the compile cache misses anyway).
os.environ.setdefault("NKI_DISABLE_TRACE_CACHE", "1")

os.environ.setdefault("VLLM_NEURON_DEBUG_MODE", "1")

os.environ.setdefault("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP", "1")

#: The largest prefill this port can run in one forward on one 24 GiB bank, and
#: a size `SUPPORTED_KV_SEGMENT_SIZES` accepts. See
#: `require_segmented_prefill`; 8192 rather than a smaller segment because a
PREFILL_SEGMENT_TOKENS = 8192

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "5400")

os.environ.setdefault("NEURON_CCOM_NO_INFINITE_TIMEOUT", "0")
os.environ.setdefault("CCOM_NO_INFINITE_TIMEOUT", "0")
os.environ.setdefault("NEURON_CCOM_SOCK_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "7200")
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "7200")
if os.environ.get("NEURON_RT_EXEC_TIMEOUT", "300") == "300":
    os.environ["NEURON_RT_EXEC_TIMEOUT"] = "7200"

_MULTI_RANK_ENV = {
    "NEURON_RT_INSPECT_ENABLE": "0",
    "NEURON_RT_INSPECT_PRECACHE_ENABLE": "0",
    "NEURON_RT_IO_RING_CACHE_SIZE": "1",
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "131072",
    "MALLOC_TOP_PAD_": "131072",
    "NEURON_HOST_MEM_PER_RANK_GIB": "20.0",
}

#: `NEURON_RT_IO_RING_CACHE_SIZE` is iteration 7's addition, and the A/B the
#: previous comment here asked for. `neuron_worker.py:713` raises the runtime's
#: own default of 1 to **32** so the runtime reuses the per-step DMA ring across
#: decode steps "instead of rebuilding it every step (kbl_exec_pre), which
#: otherwise dominates async-decode model-submit time". That is a sound default
#: for a model whose decode step is milliseconds. This port's decode step is
#: ~130 ms (TPOT) / ~370 ms (ITL), so the submit-time win is below measurement
#: noise, while 32 cached DMA rings *per loaded NEFF* are charged against
#: exactly the pool whose exhaustion is the one wedge diagnosed on this host:
#:
#:     TDRV:ddrs_build_ring_from_template  Failed to allocate memory for dma
#:                                         ring for queue qSPIO0_0
#:     TDRV:dmem_buf_copyin                Copy from buffer to memory failed:
#:                                         timeout (ret=-110)
#:     RuntimeError: Failed to schedule neff execution. status=1
#:
#: One rank of 64 ran out of device memory for descriptor rings three seconds
#: after a normal forward, and the other 63 then sat in a collective waiting for
#: it. The pool scales with *graph count*, not batch size, and this model loads
#: ~120 NEFFs.
#:
#: Measured here, `in512_bs1`, 64 ranks, same session, nothing else changed:
#:
#:     ring 32   TTFT 876.5 ms   TPOT 131.5 ms   decode 6.93 tok/s
#:     ring  1   TTFT 883.8 ms   TPOT 130.1 ms   decode 6.99 tok/s
#:
#: i.e. +0.8% / -1.1% / +0.9% — noise in both directions, so the latency the
#: upstream default buys is not present on this port.
#:
#: **The HBM saving is not there either, and that was measured rather than
#: assumed.** The ancestor port recorded ~10 GiB above one rank, which is why
#: this looked worth having. It does not reproduce. `DEEPSEEK_V41_HBM_TRACE=2`
#: logs the Neuron runtime's own `get_vnc_memory_stats()` per layer, and at
#: context 768 the two arms are indistinguishable:
#:
#:     ring  1    14.267 GiB at layer 0 of forward 0, peak 15.318 GiB of 24
#:     ring 32    14.257 GiB at layer 0 of forward 0, peak 15.365 GiB of 24
#:
#: 10 MiB and 47 MiB apart. Either the descriptor-ring pool is charged somewhere
#: `get_vnc_memory_stats()` does not report — it is driver-side, and the wedge
#: signature (`ddrs_build_ring_from_template`) comes from TDRV rather than from
#: the allocator this reads — or the pool simply is not large here.
#:
#: So the case for keeping ring 1 is not "it frees 10 GiB". It is: the latency
#: the upstream default buys is absent on this port (measured above), it is the
#: setting the ancestor port runs and the documented first response to the one
#: wedge diagnosed on this host, and **every measurement that ships was taken
#: with it** — two passing verifier runs, and the `in8192_bs1` cell that had
#: stalled for 1527 s in iteration 6's gate run completing 8/8 in both passes.
#: Whether that last one is *because* of this variable is open:
#: `.autohelix/notes/iter-7.md` §11 states the confound plainly.
#:
#: Above one rank only, as the ancestor had it: at one rank the pool has never
#: run out. `NEURON_RT_IO_RING_CACHE_SIZE=32` in the environment restores the
#: worker default, since every key here is applied with `setdefault` semantics.
#:
#: Two neighbouring knobs the ancestor port tried with it and recorded as
#: **must not touch**: `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (the worker
#: raises it 8 -> 32) and `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`. Setting
#: either to 1 makes the first forward die with `status=7 message=Execution
#: Queue Full` — that queue is not per in-flight request, it is the submission
#: queue for the many NEFFs one forward executes.

#: Ranks at or below which `NEURON_RT_DBG_FORCE_2DEV_PROXY` is applied. It was
#: measured and validated with all ranks on one device; this deployment has 16 in
#: a 4x4 torus, and the knob overrides how a collective routes.
#: `DEEPSEEK_V41_FORCE_2DEV_PROXY=1` applies it anyway.
_2DEV_PROXY_MAX_RANKS = 4

#: `mallopt` parameter numbers from glibc's `malloc.h`, paired with the value the
#: matching `MALLOC_*` environment variable above asks for. Negative by
#: convention: `M_TRIM_THRESHOLD -1`, `M_TOP_PAD -2`, `M_ARENA_MAX -8`.
_MALLOPT_PARAMS = ((-8, 2), (-1, 131072), (-2, 131072))


def apply_multi_rank_host_tuning(world_size: int) -> None:
    """Establish the host-memory environment a >1-rank run of this port needs."""
    if world_size <= 1:
        return
    settings = dict(_MULTI_RANK_ENV)
    if (
        world_size <= _2DEV_PROXY_MAX_RANKS
        or os.environ.get("DEEPSEEK_V41_FORCE_2DEV_PROXY") == "1"
    ):
        settings["NEURON_RT_DBG_FORCE_2DEV_PROXY"] = "1"
    applied = {}
    for key, value in settings.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        for param, value in _MALLOPT_PARAMS:
            libc.mallopt(ctypes.c_int(param), ctypes.c_int(value))
    except OSError as exc:  # pragma: no cover - glibc is always present on Linux
        logger.info("deepseek_v41: mallopt unavailable (%s)", exc)
    if applied:
        logger.info(
            "deepseek_v41: %d rank(s) — applied the measured multi-rank "
            "host environment %s; see __init__.py for what each is worth",
            world_size, applied,
        )


apply_multi_rank_host_tuning(_config.TENSOR_PARALLEL_SIZE)

__all__ = [
    "DeepseekV41ForCausalLM",
    "DeepseekV41HFConfig",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "ISLPY_VALIDATED",
    "PREFILL_SEGMENT_TOKENS",
    "quiet_per_call_logging",
    "register_hf_config",
]


#: Loggers outside this package that emit **once per call** and so scale with
#: the forward, the layer count and the rank count all at once. Quieted to
#: WARNING, which leaves every failure and fallback visible.
_PER_CALL_LOGGERS = ("vllm_neuron.functional.topk",)


def quiet_per_call_logging() -> None:
    """Cut the log down to what a reader can actually use."""
    if os.environ.get("DEEPSEEK_V41_VERBOSE_LOGGING") == "1":
        return
    for name in _PER_CALL_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    package = logging.getLogger(__name__)
    if not package.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - "
                "%(message)s"
            )
        )
        package.addHandler(handler)
        package.propagate = False


def put_bundled_compiler_on_path() -> None:
    """Make `shutil.which("neuronx-cc")` find the compiler in this venv.

        RuntimeError: neuronx-cc compiler binary does not exist
    """
    import shutil

    if shutil.which("neuronx-cc") is not None:
        return
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    if not os.path.isfile(os.path.join(bindir, "neuronx-cc")):
        return
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bindir
    logger.info(
        "deepseek_v41: added %s to PATH so neuronx-cc is found "
        "(the compiler ships in this interpreter's own environment)", bindir,
    )


#: Fields the top-level config mirrors off `text_config`. vLLM and a handful of
#: generic HF helpers read them without descending into the sub-config.
_MIRRORED_FROM_TEXT = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "intermediate_size",
)


class DeepseekV41TextConfig(PretrainedConfig):
    """The `text_config` block: the language model this port actually serves."""

    model_type = "deepseek_v41_text"

    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    moe_intermediate_size: int = 2304
    max_position_embeddings: int = 1_048_576
    rms_norm_eps: float = 1e-20
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = self.moe_intermediate_size
        super().__post_init__(**kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    """The `vision_config` block: recorded, not modelled."""

    model_type = "deepseek_v41_vision"

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14


class DeepseekV41HFConfig(PretrainedConfig):
    """Top level of the checkpoint's nested config: `model_type: deepseek_v41`."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    text_config: dict | PretrainedConfig | None = None
    #: The checkpoint's `vision_config` block, renamed. It is recorded, not
    #: modelled — this port serves the text path only, and the vision tower is
    #: off the `tokens -> logits` path (`factory._validate_config`,
    vision_tower_config: dict | PretrainedConfig | None = None
    #: How the weights are *stored*: `quant_method: fp8` with ue8m0 scales over
    #: 32x32 blocks for attention, `expert_dtype: fp4` for the experts. Carried
    #: through as a plain dict — `weight_loader.py` reads it, and
    quantization_config: dict | None = None
    image_token_id: int = 129_264
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = DeepseekV41TextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = DeepseekV41TextConfig()

        vision = kwargs.pop("vision_config", None)
        if vision is not None and self.vision_tower_config is None:
            self.vision_tower_config = vision
        if isinstance(self.vision_tower_config, dict):
            self.vision_tower_config = DeepseekV41VisionConfig(
                **self.vision_tower_config
            )

        super().__post_init__(**kwargs)

        for field in _MIRRORED_FROM_TEXT:
            if getattr(self, field, None) is None:
                setattr(self, field, getattr(self.text_config, field, None))

    def get_text_config(self, *args, **kwargs) -> PretrainedConfig:
        """The language-model half, which is the half this port serves."""
        return self.text_config


def register_hf_config() -> None:
    """Make `transformers.AutoConfig` recognize this architecture."""
    for config_cls in (
        DeepseekV41TextConfig,
        DeepseekV41VisionConfig,
        DeepseekV41HFConfig,
    ):
        AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)


def register_architecture() -> None:
    """Make vLLM's own `ModelRegistry` accept `DeepseekV41ForCausalLM`.

        Model architectures ['DeepseekV41ForCausalLM'] are not supported for now
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if ARCH in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCH, f"{__name__}.factory:{ARCH}")


def declare_storage_quantization() -> None:
    """Tell the Neuron platform it can serve this checkpoint's fp8 storage.

        fp8 quantization is currently not supported in neuron
    """
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    supported = getattr(platform_cls, "supported_quantization", None)
    if supported is not None and "fp8" not in supported:
        platform_cls.supported_quantization = [*supported, "fp8"]


#: The newest `islpy` neuronx-cc 2.27 can actually call. See
#: `warn_on_unsupported_islpy`.
ISLPY_VALIDATED = (2026, 1)


def warn_on_unsupported_islpy() -> None:
    """Say so at import if the compiler's integer-set binding is too new.

        [INTERNAL_ERROR] [NCC_ISMP902] Simplifier error: is_subset():
        incompatible function arguments. The following argument types are
        supported: -

        VIRTUAL_ENV=.../vllm-neuron/.venv uv pip install islpy==2026.1
    """
    try:
        from importlib.metadata import version

        raw = version("islpy")
        parts = tuple(int(p) for p in raw.split(".")[:2])
    except Exception:  # noqa: BLE001 - diagnostics only, never fatal
        return
    if parts <= ISLPY_VALIDATED:
        return
    logger.warning(
        "deepseek_v41: islpy %s is newer than the %s neuronx-cc "
        "2.27 was validated against. If compiles fail with "
        "'[NCC_ISMP902] Simplifier error: is_subset()', that is why, and it is "
        "not the graph — pin islpy==%s. See "
        "deepseek_v41.warn_on_unsupported_islpy",
        raw, ".".join(str(p) for p in ISLPY_VALIDATED),
        ".".join(str(p) for p in ISLPY_VALIDATED),
    )


def _serves_this_port(vllm_config) -> bool:
    """Is `vllm_config` about to serve this checkpoint?"""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    arches = getattr(model_config, "architectures", None)
    if not arches:
        arches = getattr(
            getattr(model_config, "hf_config", None), "architectures", None
        )
    return bool(arches) and ARCH in arches


def require_segmented_prefill(vllm_config) -> None:
    """Cap the prefill at what one Trainium bank can hold in one forward.

        ValueError: Automatic Prefix Caching (APC) requires segmented prefill
        to be enabled. Either disable APC with '--no-enable-prefix-caching' or
        set 'max_num_batched_tokens' to a supported segmented prefill size
        [512, 1024, 2048, 4096, 8192] to auto-enable segmented prefill.
    """
    if not _serves_this_port(vllm_config):
        return
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = vllm_config.model_config
    if scheduler is None:
        return

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get(
        "neuron_config", {}
    )
    if "kv_segment_size_buckets" in neuron_config:
        return
    if "num_batched_tokens_buckets" in neuron_config:
        return

    batched = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    single_shot = min(batched, max_model_len) if batched else max_model_len
    if single_shot <= PREFILL_SEGMENT_TOKENS:
        return

    scheduler.max_num_batched_tokens = PREFILL_SEGMENT_TOKENS
    if getattr(scheduler, "max_num_partial_prefills", 1) and hasattr(
        scheduler, "long_prefill_token_threshold"
    ):
        threshold = int(getattr(scheduler, "long_prefill_token_threshold", 0) or 0)
        if threshold > PREFILL_SEGMENT_TOKENS:
            scheduler.long_prefill_token_threshold = PREFILL_SEGMENT_TOKENS
    logger.info(
        "deepseek_v41: max_num_batched_tokens %d -> %d for a "
        "max_model_len of %d. A prefill longer than %d tokens does not fit one "
        "NeuronCore's bank in a single forward, so it is segmented; see "
        "deepseek_v41.require_segmented_prefill",
        batched, PREFILL_SEGMENT_TOKENS, max_model_len, PREFILL_SEGMENT_TOKENS,
    )


#: The prefill segment a *multi-rank* run uses. `PREFILL_SEGMENT_TOKENS` above
#: is the ceiling for a single rank; this is a clamp that only applies above one
#: rank, and it is the balance point between two costs that pull in opposite
#: directions. Larger segments mean fewer forwards for the same prompt, and a
#: forward is where the cost is: ~43 world-wide all-reduces and ~120 graph
#: executions, neither of which gets cheaper per token when the segment is
#: small. Smaller segments mean smaller prefill graphs, so less compile time
#: and less device memory per graph.
#:
#: Which term wins is measurable, and measuring it needs two segment sizes:
#: at a fixed segment the number of forwards is itself proportional to the
#: prompt, so TTFT against input length cannot tell a per-forward cost from a
#: per-token one. Measured on `in1024_bs1`, one cell each at 512 and 1024:
#: TTFT p50 1924.7 ms -> 1503.3 ms, which fits
#: `ceil(tokens/segment) * 422 ms + tokens * 1.06 ms` across 503, 1009 and 2027
#: input tokens to within 1%. So 1024 is worth ~22% of TTFT here and 2048 more
#: again, against first-time compiles of a wider prefill graph.
#:
#: **1024 since iter-2**, at every context the deployment uses. The win above is
#: the whole reason, and the cost that argued for waiting has been paid:
#: correctness at this segment was validated by a full verifier run on the final
#: tree (`##verifier[passed=1]`, judge 4/4), and the 1024-row prefill graphs were
#: compiled by hand, one bench cell per context, before the default moved — so
#: the gate's own bench finds them in the shared NEFF cache rather than building
#: them inside a cell timeout, which is the way `bench` fails. See
#: `.autohelix/notes/iter-2.md` §3.
#:
#: Measured here, each cell at both segments, same build:
#:
#:     in1024_bs1   TTFT p50  1916.0 ->  1499.8 ms (-21.7%),  1945 -> 1523 /1k
#:     in2048_bs1   TTFT p50  3956.0 ->  3065.9 ms (-22.5%),  1998 -> 1549 /1k
#:     in8192_bs1   TTFT p50 20296.0 -> 14028.9 ms (-30.9%),  2551 -> 1764 /1k
#:     in512_bs1    unchanged — 503 tokens is one forward at either segment
#:
#: Two segments on one cell separate the two terms, and they say something
#: iter-1's single fit could not: **`F` is per-context, not a constant.**
#: `4F + n·p = 3956.0` with `2F + n·p = 3065.9` gives `F = 445 ms` at the 2304
#: context; `16F' + n'·p' = 20296.0` with `8F' + n'·p' = 14028.9` gives
#: **`F' = 783 ms`** at 8448. The per-token term moves the other way:
#: 1.073 ms/token at 2304 against ~0.95 at 8448.
#:
#: **This paragraph used to attribute `F` to the compressed-KV read, whose
#: ceiling is pinned to `max_model_len` (`model._COMPRESS_READ_CEILING`). That
#: attribution is false and has been removed.** Iteration 9 deleted 34 of the
#: 38 compressed-KV gathers a sequence a forward — 2.6 GB of indirect reads out
#: of an `in8192_bs1` cell — and all four batch-1 cells came back within ±1.3%,
#: two in each direction. See `.autohelix/notes/iter-9.md` §6.
#:
#: Iteration 9 also re-ran this experiment one octave up (1024 vs 2048) and got
#: 325 ms at 2304 and 357 ms at 8448 for the same quantity, against the 445 and
#: 783 above. So **`F` is segment-dependent as well as context-dependent**: the
#: marginal value of eliminating a forward roughly halves with each doubling of
#: the segment, which means part of what both fits call `F` is per-token work
#: that gets cheaper in wider tiles. Quote a number from here only with the
#: segment pair it was measured from, and do not read either as an operator's
#: cost — §9c of those notes.
#:
#: None of this carries the TTFT bar, and the arithmetic says why rather than
#: hoping: every cell lands near 1550-1760 ms per 1k against a 1500 bar, and
#: `1024·p ≈ 1099 ms` of that is the per-token term alone. The segment is close
#: to spent as a lever — see `.autohelix/notes/iter-2.md` §1a and §5 for the
#: joint `(F, p)` target that would clear it.
#: **2048 since 2026-10-20, measured on the integrated tree** rather than
#: inferred from the fit above. Iteration 9 ran the 1024-vs-2048 experiment and
#: left the default at 1024 because the wider segment flipped no cell *on its
#: own*; composed with the rest of this integration it is simply faster, and a
#: cell that gains margin is worth having even when it was already passing:
#:
#:     in2048_bs1   2924.3 -> 2600.6 ms  (-11.1%),  1477 -> 1314 /1k
#:     in8192_bs1  13514.6 -> 12420.0 ms  (-8.1%),  1699 -> 1561 /1k
#:
#: Both cells reported `warmed: true`, so neither figure carries compile time.
#: Only these two move: the clamp below applies when `max_model_len > segment`,
#: so contexts 768 and 1280 are one forward at either setting.
#:
#: **4096 after measuring it**, which is one notch past where iteration 9's
#: extrapolation said the win was. That extrapolation (`in8192_bs1` at
#: 1448-1497 per 1k) did **not** hold: 4096 measured 12220.8 ms = 1536 per 1k,
#: only -1.6% against 2048's 12420, where the fit predicted -6.5%. So the
#: segment is close to spent as a lever after all, and `in8192_bs1` stays over
#: the bar. It is still the better of the two settings on both cells it
#: touches, and its graphs are now warm, so it ships:
#:
#:     in2048_bs1  2924.3 -> 2600.6 (seg 2048) -> 2544.4 ms  = 1285 /1k
#:     in8192_bs1 13514.6 -> 12420.0 (seg 2048) -> 12220.8 ms = 1536 /1k
#:
#: Contexts 768 and 1280 are one forward at any of these settings. Note 4096
#: also stops the clamp applying at context 2304 (2304 < 4096), which leaves
#: `max_num_batched_tokens` to vLLM — measured harmless here, the bucket comes
#: out `[2048]` either way.
MULTI_RANK_PREFILL_SEGMENT_TOKENS = int(
    os.environ.get("DEEPSEEK_V41_PREFILL_SEGMENT_TOKENS", 4096)
)

#: Above this `max_model_len` the segment falls back to 512. **Effectively off**
#: — every context the verifier and the bench matrix use (608, 2104, 768, 1280,
#: 2304, 8448) has had its 1024-row prefill graphs compiled by hand, so there is
#: no context left for it to protect.
#:
#: It exists, and is an environment variable, because of the hazard it was
#: briefly set to 4096 for: a wider segment needs a 1024-row prefill graph *per
#: context*, a fresh one measured 885 s to build (iter-1 §4c), and a bench cell
#: that spends its *measured* pass compiling reports
#: `completed < num_prompts` — which fails the `bench` constraint for every
#: other cell too, not just its own. So if a new context appears that nobody has
#: warmed, `DEEPSEEK_V41_WIDE_SEGMENT_MAX_MODEL_LEN=4096` is the one-variable
#: way to keep it on the graphs that exist while it is warmed separately.
_WIDE_SEGMENT_MAX_MODEL_LEN = int(
    os.environ.get("DEEPSEEK_V41_WIDE_SEGMENT_MAX_MODEL_LEN", 1 << 24)
)

#: The value, and the whole argument for it, live in `config.py` beside
#: `TENSOR_PARALLEL_SIZE` and `ATTENTION_TENSOR_PARALLEL_SIZE`: those three
#: are the port's tunable parallelism declarations, so they belong in one
from .config import MAX_NUM_SEQS as MULTI_RANK_MAX_NUM_SEQS  # noqa: E402


def require_multi_rank_budget(vllm_config) -> None:
    """Declare the prefill segment and batch size this port serves above one rank.

      * `max_num_batched_tokens` down to `MULTI_RANK_PREFILL_SEGMENT_TOKENS`,
        which is a compile-budget declaration;
      * `max_num_seqs` down to `MULTI_RANK_MAX_NUM_SEQS`, which is a
        host-memory one.
    """
    if not _serves_this_port(vllm_config):
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if parallel is None or scheduler is None or model_config is None:
        return
    world_size = int(getattr(parallel, "world_size", 1) or 1)
    if world_size <= 1:
        return

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get(
        "neuron_config", {}
    )
    operator_chose_buckets = (
        "kv_segment_size_buckets" in neuron_config
        or "num_batched_tokens_buckets" in neuron_config
    )

    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    batched = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
    segment = MULTI_RANK_PREFILL_SEGMENT_TOKENS
    if max_model_len > _WIDE_SEGMENT_MAX_MODEL_LEN and segment > 512:
        segment = 512
        logger.info(
            "deepseek_v41: max_model_len %d is above %d, so the prefill segment "
            "stays at %d rather than %d — a wider segment needs a 1024-row "
            "prefill graph for this context and none has been compiled; see "
            "deepseek_v41._WIDE_SEGMENT_MAX_MODEL_LEN",
            max_model_len, _WIDE_SEGMENT_MAX_MODEL_LEN, segment,
            MULTI_RANK_PREFILL_SEGMENT_TOKENS,
        )
    if (
        not operator_chose_buckets
        and batched > segment
        and max_model_len > segment
    ):
        scheduler.max_num_batched_tokens = segment
        threshold = int(getattr(scheduler, "long_prefill_token_threshold", 0) or 0)
        if threshold > segment:
            scheduler.long_prefill_token_threshold = segment
        logger.info(
            "deepseek_v41: %d rank(s) — max_num_batched_tokens %d -> %d. "
            "A segment's row count is a prefill graph's size, and at %d ranks every "
            "sharded shape is a fresh compile; see "
            "deepseek_v41.MULTI_RANK_PREFILL_SEGMENT_TOKENS",
            world_size, batched, segment, world_size,
        )

    seqs = int(getattr(scheduler, "max_num_seqs", 0) or 0)
    if (
        MULTI_RANK_MAX_NUM_SEQS > 1
        and seqs > 1
        and "num_seqs_buckets" not in neuron_config
    ):
        # **One** decode batch size, not the backend's default ladder of
        # powers of two. A decode graph here is keyed by the span count and is
        # 40 layers deep, so each bucket is its own full compile of the model;
        # `get_decode_padded_batch_size` would otherwise hand out 1, 2, 4, 8,
        # 16 and `max_num_seqs`, and the model would compile all six as the
        # live count drifted down through them. With one bucket every decode
        # step presents `max_num_seqs` spans — `model._bucket_spans` keeps the
        # padding rather than trimming it — so there is exactly one decode
        # graph per context for the whole run.
        #
        # What it costs is the attention on the padded rows when the batch is
        # not full; what it buys is that a concurrent cell stops compiling.
        # Measured 2026-10-20 without it: an `in1024_bs32` cell answered 2 of
        # 32 requests in 23 minutes, spending both the 32-span and the 30-span
        # step on `neuronx-cc`. The MoE does not pay for the padding at all
        # (`model._MOE_LIVE_ROWS`), and at `max_num_seqs=1` — every batch-1
        # bench cell — this does not fire.
        additional = getattr(vllm_config, "additional_config", None)
        if additional is not None:
            buckets = dict(additional.get("neuron_config", {}) or {})
            buckets["num_seqs_buckets"] = [min(seqs, MULTI_RANK_MAX_NUM_SEQS)]
            additional["neuron_config"] = buckets
            logger.info(
                "deepseek_v41: %d rank(s) — num_seqs_buckets %s, so the "
                "decode batch always pads to one compiled span count; see "
                "deepseek_v41.require_multi_rank_budget",
                world_size, buckets["num_seqs_buckets"],
            )
    if seqs > MULTI_RANK_MAX_NUM_SEQS:
        scheduler.max_num_seqs = MULTI_RANK_MAX_NUM_SEQS
        logger.info(
            "deepseek_v41: %d rank(s) — max_num_seqs %d -> %d. More "
            "concurrent sequences key more graph shapes and each loaded graph holds "
            "host memory for its code, DMA rings and scratchpad in every one of the "
            "%d worker processes. A run above 1 has stalled on this host; see "
            "config.MAX_NUM_SEQS. Every request is still served, one at a time.",
            world_size, seqs, MULTI_RANK_MAX_NUM_SEQS, world_size,
        )


#: Above this many KV slots the ceiling below declines to act, for the reason
#: `model._cap_kv_cache_blocks` gives: `num_gpu_blocks_override` is applied in
#: either direction, so a configuration whose working set is genuinely large
KV_CEILING_MAX_SLOTS = 131072


def require_attention_group(vllm_config) -> None:
    """Ask the backend for the 8-rank group the attention reduction needs."""
    if not _serves_this_port(vllm_config):
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None:
        return
    world_size = int(getattr(parallel, "world_size", 1) or 1)
    from .config import ATTENTION_TENSOR_PARALLEL_SIZE

    if world_size <= ATTENTION_TENSOR_PARALLEL_SIZE:
        return
    if getattr(parallel, "enable_expert_parallel", False):
        logger.info(
            "deepseek_v41: expert parallelism is already enabled; leaving the "
            "parallel mesh alone"
        )
        return

    additional = getattr(vllm_config, "additional_config", None)
    if additional is None:
        return
    neuron_config = dict(additional.get("neuron_config", {}) or {})
    if "ep_degree" in neuron_config:
        return

    ep_degree = world_size // ATTENTION_TENSOR_PARALLEL_SIZE
    parallel.enable_expert_parallel = True
    neuron_config["ep_degree"] = ep_degree
    additional["neuron_config"] = neuron_config
    logger.info(
        "deepseek_v41: ep_degree=%d so the EP-TP groups are %d consecutive "
        "ranks — the attention's %d-way split replicated %d times, which makes "
        "an EP-TP all-reduce exactly one of each partial instead of a %d-rank "
        "collective summing each eight times; see "
        "deepseek_v41.require_attention_group",
        ep_degree, ATTENTION_TENSOR_PARALLEL_SIZE,
        ATTENTION_TENSOR_PARALLEL_SIZE, ep_degree, world_size,
    )


def require_kv_cache_ceiling(vllm_config) -> None:
    """Cap the KV cache where the engine actually sizes it.

        capping the KV cache at 1184 blocks of 32 (37888 slots)   <- the worker
        GPU KV cache size: 91,098 tokens                          <- EngineCore
        Maximum concurrency for 9,424 tokens per request: 9.67x
    """
    if not _serves_this_port(vllm_config):
        return
    cache = getattr(vllm_config, "cache_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if cache is None or scheduler is None or model_config is None:
        return
    if getattr(cache, "num_gpu_blocks_override", None) is not None:
        return

    block = int(getattr(cache, "block_size", None) or 32)
    if not getattr(cache, "user_specified_block_size", False):
        block = 32
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    max_num_seqs = int(getattr(scheduler, "max_num_seqs", 0) or 0)
    if max_model_len <= 0 or max_num_seqs <= 0:
        return
    per_sequence = -(-max_model_len // block) + 1
    blocks = per_sequence * max_num_seqs
    if blocks * block > KV_CEILING_MAX_SLOTS:
        logger.info(
            "deepseek_v41: leaving the KV cache to vLLM — this "
            "configuration wants %d slots (%d per sequence x %d sequences), "
            "which is not obviously oversized",
            blocks * block, per_sequence, max_num_seqs,
        )
        return
    cache.num_gpu_blocks_override = blocks
    logger.info(
        "deepseek_v41: KV cache ceiling %d blocks of %d (%d slots) — "
        "%d per sequence x %d sequences, one full context each. Declared here "
        "rather than from load_weights because the engine sizes the cache in "
        "the EngineCore process, which a worker's mutation does not reach; see "
        "deepseek_v41.require_kv_cache_ceiling",
        blocks, block, blocks * block, per_sequence, max_num_seqs,
    )


#: Set to "1" to serve this checkpoint without speculative decoding. Only for
#: the A/B that proves speculation changes nothing: the same prompts with it on
#: and off have to produce identical tokens, and the only way to show that is to
NO_SPEC_DECODE_ENV = "DEEPSEEK_V41_NO_SPEC_DECODE"


def require_dspark_speculation(vllm_config) -> None:
    """Give the engine the speculative config that turns the MTP draft into a
    proposal.
    """
    if not _serves_this_port(vllm_config):
        return
    if getattr(vllm_config, "speculative_config", None) is not None:
        logger.info(
            "deepseek_v41: a speculative config is already set "
            "(method=%s); leaving it alone",
            getattr(vllm_config.speculative_config, "method", "?"),
        )
        return
    if os.environ.get(NO_SPEC_DECODE_ENV) == "1":
        logger.warning(
            "deepseek_v41: %s=1, so the mtp.* stages' draft is "
            "computed and not proposed. This is the control arm of the "
            "speculation A/B, not a configuration to serve in.",
            NO_SPEC_DECODE_ENV,
        )
        return

    from .config import DeepseekV41Config
    from .drafter import METHOD

    num_speculative_tokens = int(DeepseekV41Config.dspark_block_size)
    try:
        from vllm_neuron.vllm.spec_decode import attach_speculative_config
    except ImportError as exc:  # noqa: BLE001 - reported at error level, never silent
        logger.error(
            "deepseek_v41: this build of vllm_neuron.vllm.spec_decode "
            "has no attach_speculative_config (%s), so the DSpark draft cannot "
            "be proposed to the scheduler and inference will decode one token "
            "at a time. Everything else is unaffected.",
            exc,
        )
        return
    scheduler = getattr(vllm_config, "scheduler_config", None)
    if scheduler is not None and getattr(scheduler, "async_scheduling", False):
        scheduler.async_scheduling = False
        logger.info(
            "deepseek_v41: async scheduling off — the DSpark draft is "
            "proposed synchronously; see "
            "deepseek_v41.require_dspark_speculation"
        )

    # One RPC a step (vllm_neuron's `NeuronModelRunner._one_rpc_step`): the engine's
    # `sample_tokens` and `take_draft_token_ids` RPCs to the 64 workers cost ~1 ms each
    # of a ~50 ms bs1 step in which the device waits for the host. Set here, before the
    # engine core and the workers are spawned (they inherit it).
    # `DEEPSEEK_V41_ONE_RPC_STEP=0` keeps the three RPCs.
    if os.environ.get("DEEPSEEK_V41_ONE_RPC_STEP", "1") not in ("", "0"):
        os.environ.setdefault("VLLM_NEURON_ONE_RPC_STEP", "1")
    spec = attach_speculative_config(
        vllm_config, METHOD, num_speculative_tokens
    )
    logger.info(
        "deepseek_v41: speculative decoding through the mtp.* stages "
        "— method=%s, %d draft tokens per step, drafted by DSparkProposer from "
        "the stages this forward already runs; see "
        "deepseek_v41.require_dspark_speculation",
        getattr(spec, "method", METHOD), num_speculative_tokens,
    )


def declare_prefill_requirements() -> None:
    """Install `require_segmented_prefill` on the platform's config hook."""
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    if getattr(platform_cls, "_deepseek_v41_prefill_hook", False):
        return
    original = platform_cls.check_and_update_config

    def check_and_update_config(cls, vllm_config) -> None:
        apply_multi_rank_host_tuning(vllm_config.parallel_config.world_size)
        require_attention_group(vllm_config)
        original(vllm_config)
        require_multi_rank_budget(vllm_config)
        require_segmented_prefill(vllm_config)
        require_kv_cache_ceiling(vllm_config)
        require_dspark_speculation(vllm_config)

    platform_cls.check_and_update_config = classmethod(check_and_update_config)
    platform_cls._deepseek_v41_prefill_hook = True


def register_draft_proposer() -> None:
    """Put `DSparkProposer` in `vllm_neuron.vllm.spec_decode.PROPOSERS`."""
    try:
        from vllm_neuron.vllm.spec_decode import register_proposer

        from .drafter import METHOD, DSparkProposer
    except ImportError as exc:  # noqa: BLE001 - reported, never silent
        logger.warning(
            "deepseek_v41: DSpark proposer not registered (%s). Speculative "
            "decoding through the mtp.* stages is unavailable; everything else is "
            "unaffected.",
            exc,
        )
        return
    register_proposer(METHOD, DSparkProposer)


#: The checkpoint's `architectures[0]`, and the key both registries use.
ARCH = "DeepseekV41ForCausalLM"

quiet_per_call_logging()
put_bundled_compiler_on_path()
warn_on_unsupported_islpy()
register_hf_config()
register_architecture()
declare_storage_quantization()
declare_prefill_requirements()
register_draft_proposer()
