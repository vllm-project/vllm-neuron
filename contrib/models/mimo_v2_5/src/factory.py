# SPDX-License-Identifier: Apache-2.0
"""Factory for MiMo-V2.5 implementation selection.

This is the class registered with vLLM's ``ModelRegistry`` for the
``MiMoV2ForCausalLM`` architecture (see ``plugin.py`` and the README). It keeps
its module-level imports light so that ``plugin.register`` and vLLM's lazy
model inspection can import it without pulling in NKI or the model code.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class MiMoV2ForCausalLM(nn.Module):
    """Validates the config and selects a MiMo-V2.5 implementation.

    Serves MiMo-V2.5 and MiMo-V2.5-Pro (same architecture string, different
    config). Extends ``nn.Module`` so vLLM's ``ModelRegistry`` accepts the
    entry. Both checkpoints store their matmul weights as 128x128-blockwise
    FP8-e4m3; the weight loaders dequantize attention, the dense MLP, the
    embedding and the LM head to BF16 host-side at load time (see
    ``weight_loaders_bf16``). The routed experts are BF16 too unless the
    expert-dtype mode resolves to fp8 (``MIMO_V2_5_EXPERT_DTYPE``; ``auto``
    picks fp8 for Pro), in which case they are requantized to per-channel
    Trn2-range fp8 and run in torch. ``neuron_config.quantization`` does not
    select any of this: ``None``/``"bf16"`` and ``"fp8"`` build the same model.
    Prefer ``"bf16"``: on Trn2 the 0.29 runner adds
    ``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`` to the compiler flags when
    ``quantization == "fp8"``; the fp8 experts are stored inside the 240 range
    where e4m3fn and Trn2's e4m3 share bit patterns, so they do not need it.
    """

    # vLLM inspects the REGISTERED class for its text-generation protocol
    # (``is_text_generation_model``): ``__init__`` must take ``vllm_config``,
    # and the class needs ``embed_input_ids``, ``compute_logits`` and a
    # ``forward`` with ``input_ids``/``positions``. Without them ``vllm serve``
    # rejects the architecture ("does not support `--runner generate`"). The
    # Neuron runner never calls ``__init__``, ``embed_input_ids`` or
    # ``compute_logits``; it builds the model with ``from_configs``.

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        if hf_config is None:
            raise TypeError(f"build {type(self).__name__} with from_configs()")
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        return self._model.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self._model.lm_head(hidden_states)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: object | None = None,
        vision_neuron_config: object | None = None,
    ) -> nn.Module:
        """Create the model from configs, returning the implementation directly.

        Accepts the multimodal call signature (``text_neuron_config=`` plus
        ``encoder_neuron_config=``) as well as the positional text-only one, and
        ignores the encoder half. That is forced by this checkpoint:

        ``configuration_mimo_v2.py`` sets ``self.vision_config`` unconditionally,
        so ``hasattr(hf_config, "vision_config")`` is True unless the attribute is
        deleted. That hasattr is what makes ``NeuronPlatform.check_and_update_config``
        synthesize an ``encoder_neuron_config``, which in turn makes
        ``NeuronModelRunner.load_model`` take the multimodal
        ``from_configs(hf_config=..., text_neuron_config=..., encoder_neuron_config=...)``
        branch. The offline script deletes the attribute with a callable
        ``hf_overrides``; ``vllm serve`` only accepts a JSON dict, and a
        None-valued key still satisfies hasattr. Absorbing the kwargs here is the
        one fix both entrypoints share, and it needs no change to the platform.

        On 0.21 the runner spelled the kwarg ``vision_neuron_config``; 0.29
        renamed it to ``encoder_neuron_config``. Both are accepted.

        Ignoring the encoder config is sound because this port implements the
        TEXT decoder only: the vision/audio towers are never built and their
        checkpoint tensors are never referenced, and since this factory does not
        implement ``SupportsMultiModal``, vLLM builds no ``multimodal_config``
        for it, so no request can carry multimodal input.
        """
        del encoder_neuron_config, vision_neuron_config
        return cls._select_implementation(
            hf_config,
            neuron_config if neuron_config is not None else text_neuron_config,
        )

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import MiMoV2ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"MiMo-V2.5 supports quantization=None/'bf16'/'fp8' (the FP8 "
                f"checkpoint is converted by the weight loaders; the expert "
                f"dtype is MIMO_V2_5_EXPERT_DTYPE), got "
                f"{quantization!r}."
            )

        # Platform detection raises on a non-Neuron host unless
        # NEURON_PLATFORM_TARGET_OVERRIDE is set; treat that as "unknown" rather
        # than an error so config-only construction still works off-instance.
        try:
            from libtorch_neuronx_lite.compile.platform import get_platform_target

            platform = get_platform_target()
        except Exception:
            platform = None
        if platform in ("trn1", "trn1n", "inf2"):
            raise ValueError(
                f"MiMo-V2.5 needs 24 GB+ per device for its "
                f"{getattr(hf_config, 'n_routed_experts', '?')} routed experts; "
                f"{platform} has 16 GB. Use trn2 or trn3."
            )

        # Attention is eager because the 192-wide Q/K heads exceed every fused
        # attention kernel's 128 cap -- a perf note, not a correctness issue.
        # What IS a correctness trap is a malformed hybrid pattern: a wrong
        # length would silently turn SWA layers into full-attention ones.
        pattern = getattr(hf_config, "hybrid_layer_pattern", None)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if (
            pattern is not None
            and num_layers is not None
            and len(pattern) != num_layers
        ):
            raise ValueError(
                f"hybrid_layer_pattern has {len(pattern)} entries but "
                f"num_hidden_layers={num_layers}."
            )
