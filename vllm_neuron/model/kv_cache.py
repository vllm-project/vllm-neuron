# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


@dataclass
class RecurrentLayerSpec:
    """Cache specification for a layer that keeps recurrent state, not a KV cache.

    Linear-attention layers (gated DeltaNet, Mamba) hold a **fixed-size** state
    per sequence however long that sequence is, so they have no block table, no
    per-token growth and no context length. ``LayerSpec`` cannot describe them:
    ``num_kv_heads``/``head_size`` are meaningless here, and what matters instead
    is the concrete state tensor shapes.

    Shapes are per-rank and given in vLLM's own order -- for gated DeltaNet that
    is ``(conv_state, recurrent_state)``. Take them from
    ``MambaStateShapeCalculator`` rather than deriving them: vLLM sizes the state
    pages from the same helper, so a divergent layout aliases memory instead of
    raising.

    Attributes:
        name: Layer name, matching the key vLLM uses for its cache tensor.
        shapes: One shape per state tensor, per rank.
        dtypes: One dtype per state tensor, same order as ``shapes``.
    """

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
    recurrent_layers: list[RecurrentLayerSpec] = field(default_factory=list)
