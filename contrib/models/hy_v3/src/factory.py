# SPDX-License-Identifier: Apache-2.0
"""Factory for Hy3 (HYV3) model selection and config validation."""

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class HYV3ForCausalLM(nn.Module):
    """Factory that validates config and selects the Hy3 implementation.

    Only the static per-tensor FP8 checkpoint format (tencent/Hy3-FP8) is
    implemented today.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    # vLLM inspects the registered class in every process (including the API
    # server, which never builds the Neuron model) and requires the
    # text-generation interface (``**kwargs`` above accepts its ``vllm_config``
    # keyword). The Neuron runner builds the model through ``from_configs`` and
    # calls ``forward`` only, so these two exist for that check (same as
    # ``SyntheticNeuronModel``).
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Hy3 on Neuron embeds tokens inside forward().")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("Hy3 on Neuron computes logits inside forward().")

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import HYV3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        from vllm.config import get_current_vllm_config

        quant = getattr(hf_config, "quantization_config", None) or {}
        if (
            quant.get("quant_method") != "fp8"
            or quant.get("activation_scheme") != "static"
        ):
            raise ValueError(
                "Hy3 on Neuron currently supports only the static per-tensor FP8 "
                "checkpoint (e.g. tencent/Hy3-FP8); got quantization_config="
                f"{quant!r}."
            )

        vllm_config = get_current_vllm_config()
        parallel = vllm_config.parallel_config
        tp = parallel.tensor_parallel_size

        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise ValueError(
                "Hy3 supports a BF16 KV cache or an FP8 (E4M3) KV cache with a "
                f"calibrated KV scale profile; got kv_cache_dtype={cache_dtype!r}."
            )
        if cache_dtype in ("fp8", "fp8_e4m3"):
            # The checkpoint ships no KV scales: they come from a calibrated
            # profile (HY3_KV_SCALE_PROFILE), loaded and validated by the model.
            packed = bool(neuron_config is not None and neuron_config.fp8_packed_kv)
            dcp = getattr(parallel, "decode_context_parallel_size", 1)
            prefix = bool(vllm_config.cache_config.enable_prefix_caching)
            if (dcp > 1 or prefix) and not packed:
                raise ValueError(
                    "Hy3 FP8 KV with prefix caching or DCP reads the cache in the "
                    "segmented prefill kernels, which need the packed FP8 K layout: "
                    'set "fp8_packed_kv": true in neuron_config.'
                )
            if packed and vllm_config.cache_config.block_size % 2 != 0:
                raise ValueError("The packed FP8 K cache needs an even --block-size.")
        if parallel.data_parallel_size != 1:
            raise ValueError("Hy3 does not support data parallelism yet.")
        if vllm_config.speculative_config is not None:
            raise ValueError("Hy3 does not support speculative decoding (MTP) yet.")
        dcp = getattr(parallel, "decode_context_parallel_size", 1)
        if dcp > 1:
            # DCP shards the sequence across the ranks that hold the same
            # replicated KV head (consecutive TP ranks).
            num_kv = hf_config.num_key_value_heads
            if tp <= num_kv or (tp // num_kv) % dcp != 0:
                raise ValueError(
                    f"decode_context_parallel_size={dcp} must divide "
                    f"tensor_parallel_size / num_key_value_heads = {tp}/{num_kv}."
                )
            interleave = getattr(parallel, "cp_kv_cache_interleave_size", 1)
            block_size = vllm_config.cache_config.block_size
            if interleave != block_size:
                raise ValueError(
                    "Hy3 DCP requires --cp-kv-cache-interleave-size equal to "
                    f"--block-size ({block_size}); got {interleave}."
                )
        if neuron_config is not None:
            for name in (
                "attention_dp_size",
                "mlp_dp_size",
                "embedding_dp_size",
                "lm_head_dp_size",
            ):
                if getattr(neuron_config, name, 1) not in (None, 1):
                    raise ValueError(f"Hy3 does not support {name} > 1 yet.")

        num_heads = hf_config.num_attention_heads
        num_kv = hf_config.num_key_value_heads
        if num_heads % tp != 0 or (tp % num_kv != 0 and num_kv % tp != 0):
            raise ValueError(
                f"tensor_parallel_size={tp} must divide num_attention_heads="
                f"{num_heads} and be a divisor or multiple of num_key_value_heads={num_kv}."
            )

        # Routed experts: the FP8 MoE kernels need >= 128 intermediate columns
        # per rank (moe_cte silently falls back to an unquantized torch path
        # below that), so large TP requires expert parallelism.
        if parallel.enable_expert_parallel:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_degree

            ep_degree = get_neuron_ep_degree()
        else:
            ep_degree = 1
        expert_tp = tp // ep_degree
        i_per_rank = hf_config.moe_intermediate_size // expert_tp
        if hf_config.num_experts % ep_degree != 0:
            raise ValueError(
                f"num_experts={hf_config.num_experts} must be divisible by "
                f"ep_degree={ep_degree}."
            )
        if (
            hf_config.moe_intermediate_size % expert_tp != 0
            or i_per_rank < 128
            or i_per_rank % 16 != 0
        ):
            raise ValueError(
                f"Expert intermediate size per rank is {i_per_rank} "
                f"(moe_intermediate_size={hf_config.moe_intermediate_size}, "
                f"tensor_parallel_size={tp}, ep_degree={ep_degree}); it must be "
                ">= 128 and a multiple of 16. Use --enable-expert-parallel "
                "(full EP is the default)."
            )
