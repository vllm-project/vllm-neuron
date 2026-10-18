# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash on 16 Trainium 2 devices — package setup
============================================================

This port serves the checkpoint from **HBM**: 64 ranks, one per logical
NeuronCore of a trn2.48xlarge, with the routed experts partitioned by id and the
engram tables partitioned by row, so a forward pass reads no file. Its ancestor
(`vllm_neuron/model/deepseek_v41_timeshare`, still in-tree and still passing its
own verifier) ran 4 ranks on one device and streamed both families off disk every
forward; that is where most of the measurements quoted in these comments come
from, and each says so.

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
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "131072",
    "MALLOC_TOP_PAD_": "131072",
    "NEURON_HOST_MEM_PER_RANK_GIB": "20.0",
}

#: `NEURON_RT_IO_RING_CACHE_SIZE` is deliberately absent above: left at the
#: worker's own default of 32. Set it to 1 to restore the single-device port's
#: override for an A/B.

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
#: Left at 512 because that is the value every passing run on this host was
#: measured with, and because raising it re-keys every prefill graph: the win
#: is real but it cannot by itself carry `performance-targets`, whose decode
#: bar needs concurrency. Raise it together with `config.MAX_NUM_SEQS`, after a
#: verifier run at the new value. See `.autohelix/notes/iter-1.md` §4b-4d.
MULTI_RANK_PREFILL_SEGMENT_TOKENS = int(
    os.environ.get("DEEPSEEK_V41_PREFILL_SEGMENT_TOKENS", 512)
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
