# SPDX-License-Identifier: Apache-2.0
"""
Llama3 quantization abstraction.

Parses the HuggingFace ``quantization_config`` (attached to ``hf_config`` by
vLLM's ``transformers_utils.config.get_config``) and produces a
per-module-queryable :class:`QuantizationSpec` that modeling code can consult
to pick weight dtypes, scale handling, and kernel calls.

Design
------
The public API on :class:`QuantizationSpec` is intentionally producer-agnostic:

- ``linear_scheme``   : scheme applied to quantizable linear modules
- ``kv_cache_scheme`` : scheme applied to the KV cache
- ``get_scheme(layer_index, prefix)`` : per-module lookup with a uniform
  signature, so modeling code never needs to know which upstream producer
  built the spec (ModelOpt, compressed-tensors, …).

Today :meth:`get_scheme` unconditionally returns :attr:`linear_scheme`. The
parameters are kept in the signature so call sites in the modeling code can
be wired up without further churn when richer dispatch lands.

Currently only **NVIDIA ModelOpt static FP8** (weight + activation) is
supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Any


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------
class QuantScheme(str, Enum):
    """Quantization scheme applied to a single module / tensor.

    Values are stable strings so they are safe to log, serialize, and compare.
    """

    #: No quantization; tensors stay in the model's compute dtype.
    NONE = "none"

    #: NVIDIA ModelOpt static FP8 (float8_e4m3fn) with per-tensor weight and
    #: activation scales. Applies to linear layers; for KV caches it means
    #: the cache is stored in FP8 with static scales.
    FP8_STATIC_PER_TENSOR = "fp8_static_per_tensor"


# Schemes that KV caches are permitted to use.
_VALID_KV_CACHE_SCHEMES: frozenset[QuantScheme] = frozenset(
    {QuantScheme.NONE, QuantScheme.FP8_STATIC_PER_TENSOR}
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantizationSpec:
    """Per-module-queryable view of a model's quantization configuration.

    A ``None``-valued ``LlamaConfig.quant_spec`` means "not quantized". When
    a spec is present, modeling code should query it uniformly via
    :meth:`get_scheme` regardless of which upstream producer built it.

    Attributes:
        linear_scheme:
            Scheme applied to quantizable linear modules.

            TODO(quant, mixed-precision): today this is a single scheme for
            the entire model. Extend to per-layer / per-module mixed
            precision (e.g. different schemes per transformer block, or
            keeping specific projections in bf16) by promoting this to a
            richer representation and/or moving the decision into
            :meth:`get_scheme`. Call sites that use ``linear_scheme``
            directly (instead of ``get_scheme``) will need to be revisited
            when that happens.
        kv_cache_scheme:
            Scheme applied to the KV cache.
    """

    linear_scheme: QuantScheme
    kv_cache_scheme: QuantScheme

    # ------------------------------------------------------------------
    # Uniform per-module query
    # ------------------------------------------------------------------
    def get_scheme(
        self,
        layer_index: int | None,
        prefix: str,
    ) -> QuantScheme:
        """Return the scheme applied to the module at ``(layer_index, prefix)``.

        Args:
            layer_index: Transformer-block index (0-based) for modules that
                live inside a block (e.g. ``self_attn.q_proj``). ``None``
                for modules outside any block (e.g. ``lm_head``,
                ``embed_tokens``).
            prefix: Qualified module name. Either the full dotted path
                (``"model.layers.7.self_attn.q_proj"``) or a leaf-style
                name (``"q_proj"``, ``"lm_head"``) is accepted.

        Returns:
            The :class:`QuantScheme` to apply to this module. Today this is
            always :attr:`linear_scheme`.

        Notes:
            TODO(quant, mixed-precision): extend dispatch to honour
            per-layer or per-module overrides (ModelOpt ``exclude_modules``
            patterns, compressed-tensors ``config_groups``, layer-range
            carve-outs, …). The signature is fixed now so call sites can be
            wired up without further churn.
        """
        del layer_index, prefix  # reserved for future per-module dispatch
        return self.linear_scheme

    # ------------------------------------------------------------------
    # Construction from HuggingFace ``quantization_config``
    # ------------------------------------------------------------------
    @classmethod
    def from_hf_quantization_config(
        cls, quantization_config: dict[str, Any] | None
    ) -> "QuantizationSpec | None":
        """Parse a HuggingFace ``quantization_config`` dict.

        Returns ``None`` when ``quantization_config`` is ``None`` or falsy
        (i.e. the checkpoint is not quantized).

        Raises:
            ValueError: when the config is a quantized format that is
                recognized but not supported, or when required fields are
                missing / malformed. Users should see a clear error instead
                of a silent fallback to bf16.
        """
        if not quantization_config:
            return None
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "Expected quantization_config to be a dict, got "
                f"{type(quantization_config).__name__}."
            )

        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if not quant_method:
            # A quantization_config that is present but carries no (or an empty)
            # quant_method means "not quantized" — treat it as the bf16 weight
            # path rather than an error. This happens for EAGLE3 draft heads
            # (e.g. GPT-OSS's Llama3-based draft) served under a target
            # --hf-overrides '{"quantization_config": {}}': the empty override
            # can reach the draft config as a dict without a method.
            return None
        if quant_method == "modelopt":
            return _parse_modelopt(quantization_config)
        if quant_method == "compressed-tensors":
            # KV-cache-only quantization, wired in vllm_neuron/vllm/platform.py.
            # Pass through to the bf16 weight path here.
            return None

        raise ValueError(
            f"Unsupported quantization_config.quant_method={quant_method!r}. "
            "Llama3 currently supports: 'modelopt', 'compressed-tensors'."
        )

    # ------------------------------------------------------------------
    # Defensive invariants
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.kv_cache_scheme not in _VALID_KV_CACHE_SCHEMES:
            raise ValueError(
                f"Unsupported kv_cache_scheme={self.kv_cache_scheme!r}; "
                f"expected one of {sorted(s.value for s in _VALID_KV_CACHE_SCHEMES)}."
            )


# ---------------------------------------------------------------------------
# ModelOpt-specific parsing (private)
# ---------------------------------------------------------------------------
def _parse_modelopt(quantization_config: dict[str, Any]) -> QuantizationSpec:
    """Parse a ModelOpt-format ``quantization_config``.

    Accepted shape (both legacy flat and current nested forms are tolerated,
    matching vLLM's ``_extract_modelopt_quant_algo``)::

        {
          "quant_method": "modelopt",
          "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["lm_head", ...]
          }
        }

    ``exclude_modules`` is validated (it must be a list containing
    ``"lm_head"`` literally or via a wildcard that matches it) to catch
    obviously-unsupported checkpoints, but its contents are not yet stored
    on the spec — per-module dispatch is deferred. See
    :meth:`QuantizationSpec.get_scheme`.
    """
    inner = quantization_config.get("quantization")
    algo_src: dict[str, Any] = inner if isinstance(inner, dict) else quantization_config

    quant_algo = str(algo_src.get("quant_algo", "")).upper()
    kv_cache_algo = str(algo_src.get("kv_cache_quant_algo", "")).upper()
    exclude_modules_raw = algo_src.get("exclude_modules") or []

    if quant_algo != "FP8":
        raise ValueError(
            "Llama3 currently only supports ModelOpt quant_algo='FP8', "
            f"got quant_algo={quant_algo!r}."
        )
    if kv_cache_algo != "FP8":
        raise ValueError(
            "Llama3 currently requires ModelOpt kv_cache_quant_algo='FP8', "
            f"got kv_cache_quant_algo={kv_cache_algo!r}."
        )
    if not isinstance(exclude_modules_raw, (list, tuple)):
        raise ValueError(
            "ModelOpt quantization_config.exclude_modules must be a list, "
            f"got {type(exclude_modules_raw).__name__}."
        )

    patterns = tuple(str(m) for m in exclude_modules_raw)
    if not any(p == "lm_head" or fnmatch("lm_head", p) for p in patterns):
        raise ValueError(
            "ModelOpt FP8 checkpoints for Llama3 must list 'lm_head' in "
            f"exclude_modules; got exclude_modules={list(patterns)!r}."
        )

    return QuantizationSpec(
        linear_scheme=QuantScheme.FP8_STATIC_PER_TENSOR,
        kv_cache_scheme=QuantScheme.FP8_STATIC_PER_TENSOR,
    )


# ---------------------------------------------------------------------------
# Module-class dispatch
# ---------------------------------------------------------------------------
def resolve_attention_mlp_classes(
    quant_spec: "QuantizationSpec | None",
    layer_index: int,
) -> tuple[type, type]:
    """Return ``(AttentionCls, MLPCls)`` for the decoder layer at ``layer_index``.

    Per-module class dispatch lives here — next to :class:`QuantizationSpec`
    — so every scheme-to-implementation mapping is in one place. The
    decoder layer is the sole consumer; it instantiates the returned
    classes with the existing ``(config, layer_idx=...)`` /
    ``(config,)`` constructors.

    Args:
        quant_spec: Result of :meth:`QuantizationSpec.from_hf_quantization_config`.
            ``None`` means the model is not quantized — the bf16 classes
            from :mod:`vllm_neuron.model.llama3.model` are returned and
            the caller path is byte-identical to the pre-quantization
            code.
        layer_index: Zero-based transformer-block index. Passed through
            to :meth:`QuantizationSpec.get_scheme` so per-layer overrides
            land here automatically when that lookup becomes richer.

    Returns:
        A ``(AttentionCls, MLPCls)`` pair of :class:`torch.nn.Module`
        subclasses. Neither is instantiated; the caller owns
        construction.

    Raises:
        NotImplementedError: when the resolved scheme has no attention /
            MLP implementation wired up (e.g. ``FP8_STATIC_PER_TENSOR``
            for a future variant we haven't authored modules for yet).

    Notes:
        TODO(quant, mixed-precision): today the attention and MLP share
        a single scheme. If / when attention and MLP diverge (e.g.
        attention bf16, MLP fp8), call :meth:`QuantizationSpec.get_scheme`
        twice with distinct prefixes here.
    """
    # Import lazily to avoid pulling the modeling code into every importer
    # of this module (parsing quantization_config doesn't need torch.nn).
    from . import model as _bf16

    if quant_spec is None:
        return _bf16.LlamaAttention, _bf16.LlamaMLP

    # Single lookup today; kept here (rather than two) so the common case
    # doesn't pay for an extra dict hit, and so the intent ("one scheme
    # decides both attention and MLP") is explicit.
    scheme = quant_spec.get_scheme(layer_index, prefix="self_attn")

    if scheme is QuantScheme.NONE:
        return _bf16.LlamaAttention, _bf16.LlamaMLP

    if scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
        # Trn3 has STATIC_MX kernels (4x prefill speedup) — route to the
        # MX-specific module. Trn2 stays on the legacy STATIC fp8 module.
        import os

        from libtorch_neuronx_lite.compile.platform import get_platform_target

        if (
            get_platform_target() in ("trn3", "trn3pre")
            and os.environ.get("VLLM_NEURON_FORCE_STATIC_FP8") != "1"
        ):
            from . import model_mx_fp8 as _fp8
        else:
            from . import model_static_fp8 as _fp8

        return _fp8.LlamaAttention, _fp8.LlamaMLP

    raise NotImplementedError(
        f"No attention/MLP implementation registered for scheme={scheme!r} "
        "in resolve_attention_mlp_classes. Add a branch here when the "
        "corresponding model_<scheme>.py module lands."
    )
