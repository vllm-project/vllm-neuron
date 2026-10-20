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

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Return the shape of ONE layer's K (or V) paged cache, blocks first.

        vLLM's NIXL connector calls this to check the KV cache layout before it registers
        the caches, and it asserts the leading dimension is the block count. Without the
        method the abstract base raises a bare ``NotImplementedError`` inside the connector,
        and a disaggregated decode role dies at engine start with an empty error message.

        This backend binds K and V as SEPARATE per-layer tensors
        (``kv_caches[layer] = [k, v]``), each ``[num_blocks, num_kv_heads, block_size,
        head_size]``, so the layout is already blocks first. The K/V-outer grouping that the
        GK path allocates is a property of the BACKING buffer, not of the per-layer view the
        connector registers.

        Args:
            num_blocks: Blocks in the paged pool.
            block_size: Tokens in one block.
            num_kv_heads: KV heads for one rank.
            head_size: Head dimension.
            cache_dtype_str: Present for the base-class signature. The shape does not
                depend on it; the runner applies the packed-FP8 K view instead.

        Returns:
            ``(num_blocks, num_kv_heads, block_size, head_size)``.

        Example:
            >>> NeuronAttentionBackend.get_kv_cache_shape(4, 128, 8, 64)
            (4, 8, 128, 64)
        """
        return (num_blocks, num_kv_heads, block_size, head_size)
