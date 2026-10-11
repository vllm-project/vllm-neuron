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

import os

from transformers import AutoConfig, PretrainedConfig

from .factory import DeepseekV41ForCausalLM

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

__all__ = [
    "DeepseekV41ForCausalLM",
    "DeepseekV41HFConfig",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "register_hf_config",
]


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


#: The checkpoint's `architectures[0]`, and the key both registries use.
ARCH = "DeepseekV41ForCausalLM"

register_hf_config()
register_architecture()
declare_storage_quantization()
