# SPDX-License-Identifier: Apache-2.0
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


@register_backend(AttentionBackendEnum.CUSTOM)
class NeuronAttentionBackend(AttentionBackend):
    """
    The class implements AttentionBackend for Neuron.

    Key Notes:
    1/ When integrated with vLLM Neuron, this class is intentionally a read-only, stateless specification.
    Its sole purpose is to describe the attention capabilities supported by Neuron, enabling vLLM
    features to query its APIs and derive the correct Neuron-specific configuration. The actual attention
    implementation is provided by vLLM Neuron as part of the modeling code. This separation of responsibilities
    is a deliberate design choice as we want to keep vLLM Neuron independent of serving frameworks.

    2/ For vLLM Native, this design is expected to evolve. In that case, a full attention module will
    likely be implemented directly via NeuronAttentionBackend and integrated with the native vLLM modeling code.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_non_causal(cls) -> bool:
        """The Neuron model path supports decoder-style bidirectional attention."""
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # HND layout
        return 2, num_blocks, num_kv_heads, block_size, head_size
