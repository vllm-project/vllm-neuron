# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) port — package setup
====================================================

Two jobs, both done at import time:

1. Export `DeepseekV41ForCausalLM` so `vllm_neuron.model.registry` can map the
   checkpoint's `architectures[0]` onto this port.
2. Teach `transformers` what `model_type: "deepseek_v41"` is.

Why (2) lives here, and why it has to:

vLLM builds `ModelConfig` *before* it ever consults
`vllm_neuron.model.registry`, and `ModelConfig` resolves the checkpoint through
`transformers.AutoConfig`. `deepseek_v41` is a brand-new architecture that
upstream `transformers` (5.19) has never seen, so that resolution raised

    ValueError: The checkpoint you are trying to load has model type
    `deepseek_v41` but Transformers does not recognize this architecture.

and the engine died in `vllm.engine.arg_utils.create_model_config`, before the
platform plugin, the registry, the model class, or any device existed. Every
downstream failure was this one.

The import ordering that makes registering here sufficient: the verifier runs as
`python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.run_verifier`, and
`runpy` imports each parent package before the target module — so this file
executes before `run_verifier.main()` reaches `from vllm import LLM`. In worker
subprocesses the same registration arrives via
`neuron_worker -> vllm_neuron.model.registry -> this package`, and the config
object itself unpickles by importing this module, which re-runs the
registration. Nothing has to be registered twice to be safe: `AutoConfig`'s
`exist_ok=True` makes it idempotent.

The three config classes below are *transformers* config objects — plumbing, so
vLLM can read a shape off the checkpoint. The port's own architectural config,
which is the one `check_config` holds against the official release, stays in
`config.py`; `DeepseekV41Config.from_configs` consumes either one.
"""

from __future__ import annotations

import logging
import os
import sys

from transformers import AutoConfig, PretrainedConfig

from .factory import DeepseekV41ForCausalLM

logger = logging.getLogger(__name__)

# trn2.3xlarge has one Trainium device and no EFA adapter, so
# `NeuronWorker._set_efa_affinity` cannot resolve
# /sys/bus/pci/devices/<bdf>/infiniband and every worker dies with
#
#     RuntimeError: No EFA device found at ... This is expected on instances
#     without EFA ... set NEURON_SKIP_EFA_AFFINITY=1 to skip it
#
# which is the remedy its own message names: EFA affinity pins worker threads
# to the NUMA node nearest the network adapter, a CPU placement optimization
# with no bearing on correctness or on which device the math runs. `setdefault`
# so an operator who exports it (either way) still wins, and set here rather
# than later because workers inherit this process's environment when they are
# spawned.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

# A timeshared model's `forward` streams weights off disk between compute
# steps, so it is `@torch._dynamo.disable`d and the per-step compute is
# compiled underneath it (see `model.DeepseekV41ForCausalLM.forward`). Dynamo
# refuses to inline a disabled function under `fullgraph=True` —
#
#     Unsupported: Skip inlining torch.compiler.disable()d function
#
# and `NeuronModelRunner.load_model` sets `fullgraph = not
# VLLM_NEURON_DEBUG_MODE`, so this variable is how a model asks for the
# `fullgraph=False` outer compile it needs.
#
# **The name is misleading and worth stating plainly: it does not move anything
# to the host and it relaxes no numerics.** Its only two effects in this repo
# are `fullgraph` here and whether the graph-capture backend is constructed
# (`envs.py`, `neuron_model_runner.py:1431-1442`, `spec_decode/eagle.py:376`).
# Every compute step still compiles to a NEFF and runs on the device; what
# changes is that dynamo is allowed to end a graph at the host boundaries
# instead of raising.
os.environ.setdefault("VLLM_NEURON_DEBUG_MODE", "1")

# Warmup is a cheap pre-compile for an ordinary model and a 251 GiB disk read
# for this one. `NeuronModelRunner` warms up by running a prefill at the
# batched-token bucket; at 512 tokens that routes to most of the 384 experts in
# all 37 MoE layers, so it would stream the whole expert population off disk
# *before serving anything* — ~230 s at the measured 1.1 GB/s — and then throw
# the activations away.
#
# Skipping it loses nothing but the pre-compile: the graphs warmup would have
# built are built by the first real request instead, out of the same
# `~/.cache/neuron_libtorch` compilation cache, and the first request's own
# weight streaming is work it had to do anyway.
os.environ.setdefault("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP", "1")

#: The largest prefill this port can run in one forward on one 24 GiB bank, and
#: a size `SUPPORTED_KV_SEGMENT_SIZES` accepts. See
#: `require_segmented_prefill`; 8192 rather than a smaller segment because a
#: segment is a full pass over the streamed expert population, so halving it
#: doubles the disk traffic for the same prompt — and because 1024 and 2048
#: both run into a `neuronx-cc` `Tensorizer/MaskPropagation` pass that takes
#: minutes per graph on this model's shapes (`.autohelix/notes/iter-5.md` §5e).
PREFILL_SEGMENT_TOKENS = 8192

# vLLM's multi-process executor gives each `execute_model` RPC
# `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, **300 by default**, and raises
#
#     TimeoutError: RPC call to sample_tokens timed out.
#
# when a worker has not answered by then. For an ordinary model 300 s is
# generous. For this one a single forward streams the routed experts of 37 MoE
# layers off disk — ~230 s at the measured 1.1 GB/s before any compilation —
# and the first forward at a new shape also builds its graphs. Measured at
# `--tensor-parallel-size 4`: all four ranks reached layer 39 of the first
# prefill and the executor killed the engine anyway
# (`.autohelix/notes/iter-5.md` §10h).
#
# It only matters above TP=1, because TP=1 uses the in-process `uni` executor
# and never sends an RPC — which is why four iterations at TP=1 never saw it.
# A timeout is not a correctness knob: nothing about what is computed changes,
# only how long the parent waits before declaring the worker dead.
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "5400")

__all__ = [
    "DeepseekV41ForCausalLM",
    "DeepseekV41HFConfig",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "ISLPY_VALIDATED",
    "PREFILL_SEGMENT_TOKENS",
    "register_hf_config",
]


def put_bundled_compiler_on_path() -> None:
    """Make `shutil.which("neuronx-cc")` find the compiler in this venv.

    `libtorch_neuronx_lite.compile.backend` locates the compiler with
    `shutil.which("neuronx-cc")` and raises

        RuntimeError: neuronx-cc compiler binary does not exist

    when it is absent, so **every graph this port has not already cached fails
    to build** — the engine comes up, the weights load, and the first forward
    dies in the backend.

    `neuronx-cc` is a console script installed by the `neuronx-cc` wheel, so it
    lands in the environment's own `bin/` — the directory `sys.executable` is
    in — and is on `PATH` only for a shell that activated the venv. The
    documented invocations in `docs/` all run inside an activated environment;
    an invocation that names the interpreter by path instead, which is how this
    port's own verifier is launched, inherits a `PATH` without it. That is a
    launcher difference with nothing to do with the model, and it costs a whole
    device run to discover, so it is repaired here rather than relied on.

    Appended rather than prepended, and only when `which` cannot already find
    it: an operator with a deliberately chosen compiler on `PATH` keeps it.
    """
    import shutil

    if shutil.which("neuronx-cc") is not None:
        return
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    if not os.path.isfile(os.path.join(bindir, "neuronx-cc")):
        return
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bindir
    logger.info(
        "deepseek_v41_timeshare: added %s to PATH so neuronx-cc is found "
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
    """The `text_config` block: the language model this port actually serves.

    Written in the transformers-5 style — annotated class attributes, no
    `__init__` — because `PreTrainedConfig.__init_subclass__` turns every
    subclass into a `kw_only` dataclass. Two consequences worth knowing:
    annotating a class attribute makes it a dataclass *field* (so a mutable
    default raises), and a subclass without its own `__init__` gets wrapped to
    accept arbitrary keywords, which is what lets the rest of the checkpoint's
    keys (`compress_ratios`, the `engram_*` family, the `dspark_*` family, ...)
    land as plain attributes instead of a `TypeError`.

    Defaults are the official release's, so `DeepseekV41TextConfig()` describes
    the real model rather than a toy.
    """

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
    # Absent from this checkpoint: the FFN is MoE, so the per-expert width is
    # `moe_intermediate_size`. vLLM's memory profiling reads
    # `intermediate_size` unconditionally, so `__post_init__` mirrors it rather
    # than leaving an AttributeError lying around.
    intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = self.moe_intermediate_size
        super().__post_init__(**kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    """The `vision_config` block: recorded, not modelled.

    The official config always carries one, so it has to parse; this port serves
    the text path only (see `factory._validate_config`), and the vision tower is
    off the `tokens -> logits` path.
    """

    model_type = "deepseek_v41_vision"

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14


class DeepseekV41HFConfig(PretrainedConfig):
    """Top level of the checkpoint's nested config: `model_type: deepseek_v41`.

    `sub_configs` is what lets `transformers` build the two blocks above out of
    the nested JSON, and what makes `get_text_config()` return the text block —
    which is where vLLM looks for `hidden_size`, `num_hidden_layers` and the
    rest.
    """

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    text_config: dict | PretrainedConfig | None = None
    #: The checkpoint's `vision_config` block, renamed. It is recorded, not
    #: modelled — this port serves the text path only, and the vision tower is
    #: off the `tokens -> logits` path (`factory._validate_config`,
    #: `config.py`'s module docstring, and `floorplan/`, which treats placing
    #: the tower as an error).
    #:
    #: The rename is load-bearing, not cosmetic. `NeuronPlatform` keys its whole
    #: vision path off `hasattr(hf_config, "vision_config")`
    #: (`_resolve_vision_auto_config`: "Only called for models with
    #: hf_config.vision_config"), which populates
    #: `additional_config["vision_neuron_config"]`, which makes
    #: `NeuronModelRunner.load_model` take its multimodal branch and call
    #: `from_configs(hf_config=..., text_neuron_config=...,
    #: vision_neuron_config=...)`. Serving this checkpoint as the causal LM it
    #: is means not advertising a vision tower that this port does not build.
    vision_tower_config: dict | PretrainedConfig | None = None
    #: How the weights are *stored*: `quant_method: fp8` with ue8m0 scales over
    #: 32x32 blocks for attention, `expert_dtype: fp4` for the experts. Carried
    #: through as a plain dict — `weight_loader.py` reads it, and
    #: `transformers` leaves a dict alone instead of resolving it against its
    #: own quantizer registry, which has no entry for this layout.
    quantization_config: dict | None = None
    image_token_id: int = 129_264
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = DeepseekV41TextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = DeepseekV41TextConfig()

        # `vision_config` as the checkpoint spells it, moved to the name above.
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
    """Make `transformers.AutoConfig` recognize this architecture.

    Idempotent: `exist_ok=True` means a second call (worker subprocess, config
    unpickle, a re-import under a different parent) is a no-op rather than a
    `ValueError`.
    """
    for config_cls in (
        DeepseekV41TextConfig,
        DeepseekV41VisionConfig,
        DeepseekV41HFConfig,
    ):
        AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)


def register_architecture() -> None:
    """Make vLLM's own `ModelRegistry` accept `DeepseekV41ForCausalLM`.

    `ModelConfig` validates `architectures[0]` against `vllm.ModelRegistry`
    before any Neuron code runs, and raises

        Model architectures ['DeepseekV41ForCausalLM'] are not supported for now

    if it is absent. `vllm_neuron.model.registry` is *not* enough on its own:
    `NeuronWorker` copies it into vLLM's registry, but that happens in the
    worker, long after the main process validated `ModelConfig`. Every other
    architecture this backend serves (Llama, Qwen3, GptOss) happens to also be
    in vLLM's in-tree list, so the gap only shows up for a model vLLM has never
    heard of.

    `ARCH` is the checkpoint's `architectures[0]`; the target is
    `factory.DeepseekV41ForCausalLM`, matching `vllm_neuron.model.registry`, so
    both registries resolve to the same class.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if ARCH in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCH, f"{__name__}.factory:{ARCH}")


def declare_storage_quantization() -> None:
    """Tell the Neuron platform it can serve this checkpoint's fp8 storage.

    `ModelConfig._verify_quantization` reads the checkpoint's
    `quantization_config.quant_method` — `"fp8"` here — and hands it to
    `NeuronPlatform.verify_quantization`, whose `supported_quantization` list is
    `neuron_quant`, `compressed-tensors`, `modelopt`. So engine construction
    died on

        fp8 quantization is currently not supported in neuron

    Note what that list means: it names the *vLLM quantization backends* the
    Neuron platform knows how to drive. There is no vLLM backend for this
    checkpoint's layout — fp8 weights with ue8m0 scales over 32x32 blocks,
    experts in fp4 — and there should not be one, because this port does the
    dequantization itself: `weight_loader.py` on the staging path, and the
    fp8/fp4 NKI kernels under `kernels/`. Adding `"fp8"` here says "this port
    handles fp8-stored weights", which is what is true. It does not make vLLM
    apply any quantization machinery to these weights: `neuron_config`'s own
    `quantization` field stays `None`, so nothing in the Neuron weight path is
    rerouted, and `_validate_quantization_config` only constrains
    `compressed-tensors` checkpoints.

    Setting the checkpoint's own `quantization_config` aside instead does *not*
    work, and that dead end is worth recording:
    `vllm.transformers_utils.config.get_config` re-reads `config.json` off disk
    and force-assigns `config.quantization_config = <raw dict>` after
    `AutoConfig` has returned, so the value cannot be influenced from the
    config class at all.

    The list belongs to `vllm_neuron/vllm/platform.py`, which this port may not
    edit, so the declaration is made from here — the port's own setup module —
    rather than by widening a file outside its scope.
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

    `neuronx-cc`'s `IntegerSetAnalysis` is a wrapper over **islpy**, and its
    `Simplifier.optimizeMemSet` pass calls `is_subset()` on every graph whose
    simplification reaches a zero-fill — which, in this model, is most of them.
    With `islpy 2026.2.2` installed, that call does not match any overload the
    binding exposes and the compiler aborts:

        [INTERNAL_ERROR] [NCC_ISMP902] Simplifier error: is_subset():
        incompatible function arguments. The following argument types are
        supported: -

    The empty list of supported types is the tell: it is a pybind11 overload
    error, not anything about the graph. The same HLO, the same compiler build
    and the same NumPy compile to a NEFF against `islpy 2026.1`.

    **This is worth a line of log because of how it fails.** The error names a
    compiler pass and arrives one graph at a time, so it reads as a
    shape-specific compiler bug in the model — two iterations of this port
    chased it as one. The remedy is a pin:

        VIRTUAL_ENV=.../vllm-neuron/.venv uv pip install islpy==2026.1

    (that venv is managed by `uv`; it has no `bin/pip` and `python -m pip`
    fails.) A warning rather than a raise: the version comparison is against
    what this port was validated with, not a known-bad list, so a newer islpy
    that fixes the binding should not be fatal.
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
        "deepseek_v41_timeshare: islpy %s is newer than the %s neuronx-cc "
        "2.27 was validated against. If compiles fail with "
        "'[NCC_ISMP902] Simplifier error: is_subset()', that is why, and it is "
        "not the graph — pin islpy==%s. See "
        "deepseek_v41_timeshare.warn_on_unsupported_islpy",
        raw, ".".join(str(p) for p in ISLPY_VALIDATED),
        ".".join(str(p) for p in ISLPY_VALIDATED),
    )


def _serves_this_port(vllm_config) -> bool:
    """Is `vllm_config` about to serve this checkpoint?

    Every hook below has to be inert for the other architectures this backend
    serves, which share the same platform object.
    """
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

    **What this is.** `PREFILL_SEGMENT_TOKENS` is a property of the model: a
    prefill forward keeps a `[tokens, hc_mult, hidden]` stream, the attention
    scores over the sparse window and the MoE's staged experts live at the same
    time, and at 9424 tokens in one shot that does not fit in the 24 GiB a
    NeuronCore has. A longer prompt is served by *segmenting* the prefill, which
    is the whole timeshare idea applied along the token axis instead of the
    weight axis, and it is what this port is built for — `PagedSpan` addresses
    every cache by absolute position precisely so a prompt can arrive in pieces.

    So this declares the largest prefill the model can run, by lowering
    `scheduler_config.max_num_batched_tokens` to it. Lowering that below
    `max_model_len` is what `resolve_segmented_prefill_config` reads as "chunked
    prefill", and 8192 is one of the sizes it accepts, so the engine
    auto-enables segmented prefill at `kv_segment_size_buckets=[8192]` through
    its own supported path. Nothing is bypassed and no check is relaxed: the
    engine ends up in the configuration `docs/guides/features-guide.md`
    recommends for long contexts (`--max-num-batched-tokens 8192`).

    **Why it has to be here.** `max_num_batched_tokens` is an
    engine-construction argument, settled before any model class exists — the
    same reason the verifier has to read `config.TENSOR_PARALLEL_SIZE` rather
    than letting the model pick its own world size. A caller who does not pass
    it gets `max_num_batched_tokens == max_model_len`, which is a single-shot
    prefill, and on this build a single-shot prefill with vLLM's default
    Automatic Prefix Caching is rejected outright in
    `NeuronModelRunner.__init__`:

        ValueError: Automatic Prefix Caching (APC) requires segmented prefill
        to be enabled. Either disable APC with '--no-enable-prefix-caching' or
        set 'max_num_batched_tokens' to a supported segmented prefill size
        [512, 1024, 2048, 4096, 8192] to auto-enable segmented prefill.

    That raise is in the model *runner*'s constructor, which runs before
    `load_model`, so the hooks this port already uses to reach engine
    configuration (`model._cap_kv_cache_blocks`, from `load_weights`) are all
    downstream of it. The only place a model package gets to speak before it is
    the platform's `check_and_update_config`, which is where vLLM platforms
    normally clamp exactly this field — and which lives in
    `vllm_neuron/vllm/platform.py`, a file this port may not edit. So the
    declaration is made from the port's own setup module, the same way
    `declare_storage_quantization` widens the platform's quantization list.

    **It only ever lowers, and never contradicts the operator.** If a caller
    passed `--max-num-batched-tokens`, or set `kv_segment_size_buckets`
    themselves, or asked for a prefill this port can already run in one shot,
    this does nothing.
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
    # The prefill the engine would actually run in one forward. Below the
    # segment size there is nothing to segment, and a caller who already named
    # a chunk size is already on the segmented path.
    single_shot = min(batched, max_model_len) if batched else max_model_len
    if single_shot <= PREFILL_SEGMENT_TOKENS:
        return

    scheduler.max_num_batched_tokens = PREFILL_SEGMENT_TOKENS
    if getattr(scheduler, "max_num_partial_prefills", 1) and hasattr(
        scheduler, "long_prefill_token_threshold"
    ):
        # Keep vLLM's own long-prefill accounting consistent with the new
        # budget; it is derived from `max_num_batched_tokens` when unset and a
        # stale larger value would let one request claim the whole budget it no
        # longer has.
        threshold = int(getattr(scheduler, "long_prefill_token_threshold", 0) or 0)
        if threshold > PREFILL_SEGMENT_TOKENS:
            scheduler.long_prefill_token_threshold = PREFILL_SEGMENT_TOKENS
    logger.info(
        "deepseek_v41_timeshare: max_num_batched_tokens %d -> %d for a "
        "max_model_len of %d. A prefill longer than %d tokens does not fit one "
        "NeuronCore's bank in a single forward, so it is segmented; see "
        "deepseek_v41_timeshare.require_segmented_prefill",
        batched, PREFILL_SEGMENT_TOKENS, max_model_len, PREFILL_SEGMENT_TOKENS,
    )


#: Above this many KV slots the ceiling below declines to act, for the reason
#: `model._cap_kv_cache_blocks` gives: `num_gpu_blocks_override` is applied in
#: either direction, so a configuration whose working set is genuinely large
#: must be left to vLLM's memory-based sizing rather than handed a number that
#: does not fit. Kept equal to `model._KV_CAP_MAX_SLOTS`.
KV_CEILING_MAX_SLOTS = 131072


def require_kv_cache_ceiling(vllm_config) -> None:
    """Cap the KV cache where the engine actually sizes it.

    `model._cap_kv_cache_blocks` has the arithmetic and the argument for it:
    vLLM sizes the cache from the memory free when it profiles, which is after
    the weights load and *before* any of this port's ~120 NEFFs are loaded, so
    it hands out a cache many times larger than the scheduler can fill and the
    graphs then have nowhere to go. The ceiling is one full `max_model_len` per
    sequence times the running batch — a ceiling, not a reduction below what
    correctness needs, since one sequence cannot reference more slots than its
    own context.

    **What is new here is the process it runs in, and it was a silent bug.**
    `_cap_kv_cache_blocks` runs from `load_weights`, i.e. in a *worker*, and
    sets `cache_config.num_gpu_blocks_override` on the worker's own
    `VllmConfig`. The code that reads that field — `get_kv_cache_config`,
    `vllm/v1/core/kv_cache_utils.py` — runs in the **EngineCore** process. At
    one rank those are the same process, because `world_size == 1` selects the
    in-process `uni` executor, so the mutation is visible and the cap works. At
    four ranks the workers are separate processes with unpickled copies, so the
    mutation goes nowhere and **the cap silently stops applying**. Measured on
    this host, a four-rank run of the verifier's own arguments:

        capping the KV cache at 1184 blocks of 32 (37888 slots)   <- the worker
        GPU KV cache size: 91,098 tokens                          <- EngineCore
        Maximum concurrency for 9,424 tokens per request: 9.67x

    — no `Overriding num_gpu_blocks=...` line at all, and a device-tensor
    census at layer 4 finding 43 cache tensors of `(2, 2523, 1, 32, 512)` where
    the cap had named 1184 blocks. That is 6.6 GiB of KV plus 1.1 GiB of
    index/window buffers against the 3.1 + 0.5 GiB the cap intends: **4.1 GiB
    of a 24 GiB bank**, which is most of what sharding the weights across four
    ranks had just freed.

    So the ceiling is declared here instead, from
    `NeuronPlatform.check_and_update_config`, which runs once in the process
    that builds the `VllmConfig` — before EngineCore, before the workers, and
    before anything reads the field. Workers get it by unpickling.
    `_cap_kv_cache_blocks` stays where it is and becomes a no-op when this has
    acted (it returns early if the override is already set), so a path that
    reaches `load_weights` without this hook still gets capped.

    Only ever lowers, only for this port, and inert if the operator set
    `num_gpu_blocks_override` themselves.
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

    # The block size the cache is **laid out** with, which is not necessarily
    # the one `cache_config` carries now: `update_block_size_for_backend` runs
    # later and settles on 32 unless the operator named one. Dividing by a
    # stale 16 is what made the worker-side cap realize half the reduction it
    # named, before iteration 5 fixed it; the same rule is applied here.
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
            "deepseek_v41_timeshare: leaving the KV cache to vLLM — this "
            "configuration wants %d slots (%d per sequence x %d sequences), "
            "which is not obviously oversized",
            blocks * block, per_sequence, max_num_seqs,
        )
        return
    cache.num_gpu_blocks_override = blocks
    logger.info(
        "deepseek_v41_timeshare: KV cache ceiling %d blocks of %d (%d slots) — "
        "%d per sequence x %d sequences, one full context each. Declared here "
        "rather than from load_weights because the engine sizes the cache in "
        "the EngineCore process, which a worker's mutation does not reach; see "
        "deepseek_v41_timeshare.require_kv_cache_ceiling",
        blocks, block, blocks * block, per_sequence, max_num_seqs,
    )


def declare_prefill_requirements() -> None:
    """Install `require_segmented_prefill` on the platform's config hook.

    `NeuronPlatform.check_and_update_config` is the one point where a vLLM
    platform adjusts a resolved `VllmConfig`, and it runs in the process that
    builds the config — before the engine core, the workers, or the model
    class. This wraps it rather than replacing it: the platform's own work runs
    first, unchanged, and the port's requirement is applied to the result.

    Idempotent, because this module is imported in the parent process (through
    `runpy`, ahead of `from vllm import LLM`), again when a config unpickles,
    and again in every worker through `vllm_neuron.model.registry`.
    """
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    if getattr(platform_cls, "_deepseek_v41_prefill_hook", False):
        return
    original = platform_cls.check_and_update_config

    def check_and_update_config(cls, vllm_config) -> None:
        original(vllm_config)
        require_segmented_prefill(vllm_config)
        require_kv_cache_ceiling(vllm_config)

    platform_cls.check_and_update_config = classmethod(check_and_update_config)
    platform_cls._deepseek_v41_prefill_hook = True


def register_draft_proposer() -> None:
    """Put `DSparkProposer` in `vllm_neuron.vllm.spec_decode.PROPOSERS`.

    So that `speculative_config.method == "deepseek_mtp"` builds this model's own MTP
    stages as the drafter instead of looking for a second checkpoint. Registering is
    cheap and side-effect free — `NeuronModelRunner` only reaches for the registry when
    a speculative config names the method — so it is done unconditionally here, where
    the rest of the port's hooks are installed.

    **This does not yet make inference speculative, and `drafter.py`'s module docstring
    says why**: in this build the only path that calls `propose` is gated on
    `is_eagle3_spec` (`neuron_model_runner.py:5915`) and asserts
    `isinstance(self.drafter, EagleProposer)` (`:8325`), both in a directory the loop
    may not edit. Until those two lines admit a registered proposer, supplying a
    `speculative_config` would build a proposer nothing calls — so
    `require_kv_cache_ceiling`'s neighbour hook deliberately does not supply one.

    The import is local and guarded: `drafter` imports `config`, and this module is
    imported from `vllm_neuron.model.registry` during platform start-up, so an import
    error here would take registration down with it — which costs the port everything,
    for a feature nothing can call yet.
    """
    try:
        from vllm_neuron.vllm.spec_decode import register_proposer

        from .drafter import METHOD, DSparkProposer
    except ImportError as exc:  # noqa: BLE001 - reported, never silent
        logger.warning(
            "deepseek_v41_timeshare: DSpark proposer not registered (%s). Speculative "
            "decoding through the mtp.* stages is unavailable; everything else is "
            "unaffected.",
            exc,
        )
        return
    register_proposer(METHOD, DSparkProposer)


#: The checkpoint's `architectures[0]`, and the key both registries use.
ARCH = "DeepseekV41ForCausalLM"

put_bundled_compiler_on_path()
warn_on_unsupported_islpy()
register_hf_config()
register_architecture()
declare_storage_quantization()
declare_prefill_requirements()
register_draft_proposer()
