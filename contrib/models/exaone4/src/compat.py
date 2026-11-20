# SPDX-License-Identifier: Apache-2.0
"""Helpers this model needs that vLLM Neuron 0.29.0.1.2.0 does not ship.

They are copied unchanged from a later vLLM Neuron release, so the model
runs on the 0.29 plugin as installed.
"""

import logging

import torch

from vllm_neuron.model.kv_cache import paged_slot_indices

logger = logging.getLogger(__name__)


def hoist_rope_parameters(
    config_dict: dict, supported_rope_types: tuple[str, ...] = ("default",)
) -> dict:
    """Return ``config_dict`` with ``rope_parameters`` unpacked to the pre-v5 layout.

    transformers >= 5 keeps all RoPE settings under ``rope_parameters``.
    Configs that filter the HF dict to their dataclass fields (``rope_theta``,
    ``rope_scaling``) would otherwise drop them and fall back to the field
    defaults. ``rope_theta`` becomes a top-level key and the remaining
    settings (``rope_type``, ``factor``, ...) become ``rope_scaling`` when the
    type is not ``default``. Existing top-level keys win. Logs a warning if the
    RoPE type is not in ``supported_rope_types``.

    Example::

        >>> d = hoist_rope_parameters({"rope_parameters": {"rope_theta": 1e8}})
        >>> d["rope_theta"]
        100000000.0
        >>> p = {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0}
        >>> hoist_rope_parameters({"rope_parameters": p}, ("yarn",))["rope_scaling"]
        {'rope_type': 'yarn', 'factor': 4.0}
    """
    rope_parameters = config_dict.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        return config_dict
    out = dict(config_dict)
    if "rope_theta" in rope_parameters:
        out.setdefault("rope_theta", rope_parameters["rope_theta"])
    scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
    rope_type = scaling.get("rope_type", "default")
    if rope_type != "default" and out.get("rope_scaling") is None:
        out["rope_scaling"] = scaling
    if rope_type not in supported_rope_types:
        logger.warning(
            "rope_type=%r is not implemented for this model; using default RoPE",
            rope_type,
        )
    return out


def build_per_head_block_table(
    block_table_2d: torch.Tensor, kv_heads: int
) -> torch.Tensor:
    """Build a per-head 3D block table from a 2D base table.

    The kernel folds the 4D KV cache into a ``[num_blocks * kv_heads]`` pool.
    Each ``(batch, head)`` entry is the global pool index ``base * kv_heads + h``.
    Preserve ``-1`` (inactive) entries unchanged.

    Expects a 2D ``[B, num_blocks]`` table with ``-1`` as the inactive sentinel.
    Returns a ``[B, kv_heads, num_blocks]`` table.

    Example:
        >>> import torch
        >>> t = torch.tensor([[0, 1, -1]])
        >>> build_per_head_block_table(t, kv_heads=2)
        tensor([[[ 0,  2, -1],
                 [ 1,  3, -1]]])
    """
    assert block_table_2d.dim() == 2, (
        f"build_per_head_block_table expects a 2D [B, num_blocks] table, got "
        f"{block_table_2d.dim()}D {tuple(block_table_2d.shape)}"
    )
    base_3d = block_table_2d.unsqueeze(1)  # [B, 1, num_blocks]
    kv_h = torch.arange(
        kv_heads, dtype=block_table_2d.dtype, device=block_table_2d.device
    ).view(1, kv_heads, 1)
    shifted = base_3d * kv_heads + kv_h  # [B, kv_heads, num_blocks]
    return torch.where(base_3d == -1, base_3d, shifted).contiguous()


def scatter_new_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    nkh: int,
    head_dim: int,
) -> None:
    """Scatter new K/V tokens into the paged KV cache.

    The fused decode kernel returns ``K_new [d_head, B, kv_heads, S]``
    and ``V_new [B, kv_heads, S, d_head]``. When kv_heads == 1, the kernel
    drops K's head axis to 3D. Callers must re-insert the singleton head axis
    before calling here. This function always receives the 4D layout.

    The function reorders both tensors to head-major ``[kv_heads, B*S, d_head]``.
    ``slot_mapping`` is ``[B*S]`` in (batch, token) order. Row ``i`` of the
    reordered K/V is head ``i // (B*S)`` and token ``i % (B*S)``. The function
    writes that row to ``cache[block_of(token), head, pos_of(token)]``.

    Example:
        >>> import torch
        >>> # 1 block of size 4, 1 kv-head, d_head=2, B*S=2 new tokens
        >>> k_cache = torch.zeros(1, 1, 4, 2)
        >>> v_cache = torch.zeros(1, 1, 4, 2)
        >>> K_new = torch.ones(2, 1, 1, 2)   # [d_head, B, kv, S]
        >>> V_new = torch.ones(1, 1, 2, 2)   # [B, kv, S, d_head]
        >>> slots = torch.tensor([0, 1])
        >>> scatter_new_kv(k_cache, v_cache, K_new, V_new, slots, 4, 1, 2)
    """
    assert K_new.dim() == 4 and V_new.dim() == 4, (
        f"scatter_new_kv expects 4D per-head K_new/V_new from the GQA kernel; "
        f"got K_new.dim()={K_new.dim()}, V_new.dim()={V_new.dim()}"
    )
    block_indices, position_indices = paged_slot_indices(
        slot_mapping, block_size, k_cache.shape[0]
    )

    # num_tokens = B*S from the kernel tensors (K_new: [d_head, B, kv, S]). The
    # downstream index_put_ enumeration assumes one slot per (batch, token), so
    # slot_mapping must be exactly the B*S decode write set; the assert enforces
    # that LENGTH contract. Padded *values* inside that length are safe —
    # paged_slot_indices redirects them to the null block — but a padded decode
    # batch that changes the length would still need the scatter reworked.
    _, B, _, S = K_new.shape
    num_tokens = B * S
    assert slot_mapping.shape[0] == num_tokens, (
        f"slot_mapping length ({slot_mapping.shape[0]}) must equal B*S ({num_tokens}) "
        f"from K_new {tuple(K_new.shape)}"
    )

    k_new = K_new.permute(2, 1, 3, 0).reshape(nkh, num_tokens, head_dim)
    v_new = V_new.permute(1, 0, 2, 3).reshape(nkh, num_tokens, head_dim)
    k_new_flat = k_new.reshape(-1, head_dim)
    v_new_flat = v_new.reshape(-1, head_dim)

    head_indices_for_put = torch.arange(
        nkh, dtype=torch.long, device=slot_mapping.device
    ).repeat_interleave(num_tokens)
    block_indices_for_put = block_indices.repeat(nkh)
    position_indices_for_put = position_indices.repeat(nkh)

    k_cache.index_put_(
        (block_indices_for_put, head_indices_for_put, position_indices_for_put),
        k_new_flat.to(k_cache.dtype),
    )
    v_cache.index_put_(
        (block_indices_for_put, head_indices_for_put, position_indices_for_put),
        v_new_flat.to(v_cache.dtype),
    )
