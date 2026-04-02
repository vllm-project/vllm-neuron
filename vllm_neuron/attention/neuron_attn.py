# SPDX-License-Identifier: Apache-2.0
from vllm.attention.backends.abstract import AttentionBackend


class NeuronAttentionBackend(AttentionBackend):
    """
    Neuron-specific attention backend implementation for vLLM.

    This backend provides the attention interface required by vLLM while
    integrating with NeuronX Distributed Inference for optimized
    attention computation on AWS Neuron devices.
    """

    @staticmethod
    def get_name() -> str:
        return "NEURON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["NeuronAttentionBackend"]:
        return NeuronAttentionBackend

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)
