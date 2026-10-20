# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_WEIGHT_LOADER_ATTR = "weight_loader"


@dataclass
class ShardSpec:
    """Specification for sharding a tensor along a dimension.

    Args:
        dim: Dimension to shard along
        size: Size of each shard
        num_shards: Total number of shards (typically TP world size)
    """

    dim: int
    size: int
    num_shards: int


@dataclass
class SafetensorsWeightLoader:
    """Defines how to load checkpoint tensor(s) from a Safetensor slice into a parameter.

    SafetensorsWeightLoader allows customizing how checkpoint tensors are transformed before
    being loaded into a model parameter. This is useful for tensor parallelism
    (sharding), weight fusion (e.g., fused QKV), key remapping from checkpoint to
    parameter name, and custom transformations to support arbitrary layout requirements.

    Args:
        transform: Function (slices, rank) -> tensor. Receives a list of PySafeSlice
            objects (one per checkpoint key from mappings) and the current rank.
            Returns the final tensor to load into the parameter.
            If None, loads the first slice as-is (identity transform).

    Example:
        Column-parallel sharding (shard along dim 0)::
            SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * 64:(rank + 1) * 64, :])

        Row-parallel sharding (shard along dim 1)::
            SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:, rank * 64:(rank + 1) * 64])

        Fusing multiple checkpoint tensors::
            SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat([s[:] for s in slices], dim=0))
    """

    transform: Callable[[list["PySafeSlice"], int], torch.Tensor] = None

    def load(self, slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        """Load and transform checkpoint slices into a tensor.

        Args:
            slices: List of PySafeSlice objects from the checkpoint.
            rank: Current tensor parallel rank.

        Returns:
            Transformed tensor ready to be loaded into the parameter.
        """
        if self.transform:
            result = self.transform(slices, rank)
        else:
            # By default, load first slice as-is
            assert len(slices) == 1, (
                f"SafetensorsWeightLoader without transform should only take in a single slice but got {len(slices)}"
            )
            result = slices[0][:]

        return result.contiguous()


def set_weight_loader(param: nn.Parameter, loader: SafetensorsWeightLoader) -> None:
    """Attach a SafetensorsWeightLoader to a parameter.

    Args:
        param: The parameter to attach the loader to.
        loader: The SafetensorsWeightLoader defining how to load this parameter.

    Example::
        param = torch.nn.Parameter(torch.empty(256, 512))
        set_weight_loader(param, SafetensorsWeightLoader(...))
    """
    setattr(param, _WEIGHT_LOADER_ATTR, loader)


def get_weight_loader(param: nn.Parameter) -> SafetensorsWeightLoader:
    """Retrieve the SafetensorsWeightLoader attached to a parameter.

    Args:
        param: The parameter to get the loader from.

    Returns:
        The attached SafetensorsWeightLoader, or a default SafetensorsWeightLoader (identity transform) if none.
    """
    return getattr(param, _WEIGHT_LOADER_ATTR, SafetensorsWeightLoader())


def with_rank_override(
    loader: SafetensorsWeightLoader, rank: int
) -> SafetensorsWeightLoader:
    """Wrap a loader to use a specific rank instead of the passed rank.

    This is useful when using multiple tensor parallel groups where the global
    TP rank differs from the rank within a specific group. For example, with
    attention data parallelism, attention weights are sharded across a subset
    of ranks (e.g., 8 ranks per attention TP group), but the weight loading
    system passes the global TP rank (e.g., 0-15). This wrapper lets you
    specify the local rank within the group.

    Args:
        loader: The base loader to wrap.
        rank: The rank to use for loading (typically the local rank within a group).

    Returns:
        A new SafetensorsWeightLoader that uses the specified rank.

    Example::
        base_loader = sharding_weight_loader(shard_dim=0, shard_size=64, num_shards=8)
        # Load shard for local attention TP rank, not global TP rank
        loader = with_rank_override(base_loader, rank=attn_tp_rank)
    """
    return SafetensorsWeightLoader(
        transform=lambda slices, _: loader.load(slices, rank)
    )


##############################
# Common transform factories #
##############################


def sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    pad_shard: bool = False,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that shards a tensor along a dimension.

    Args:
        shard_dim: Dimension to shard along in the final parameter shape.
        shard_size: Size of each shard along shard_dim.
        num_shards: Total number of shards (typically TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed
            (swaps dim 0 and 1) relative to the parameter layout.
        pad_shard: If True, zero-pad the returned shard to exactly shard_size
            when the checkpoint slice is smaller (e.g. last rank when
            total_dim % num_shards != 0). Defaults to False.

    Note:
        If using is_storage_transposed, we require checkpoint weight to be 2-dimensional.
        Please write your own weight loader if you need more complex functionality.

    Returns:
        SafetensorsWeightLoader configured for sharding.

    Example:
        param = torch.nn.Parameter(...)

        Column-parallel linear (shard output dim):
            # Checkpoint has [hidden_size, input_size], we want [hidden_size // tp, input_size]
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=hidden_size // tp_size,
                num_shards=tp_size
            ))

        Linear that is a transpose of checkpoint (shard input dim):
            # Checkpoint has [output_size, hidden_size], we want [hidden_size // tp, output_size]
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=hidden_size // tp_size,
                num_shards=tp_size,
                is_storage_transposed=True,
            ))

        Last rank needs padding (caller does not pre-zero the buffer):
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=shard_size,
                num_shards=tp_size,
                pad_shard=True,
            ))
    """

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "sharding_weight_loader() only supports a single tensor"
        )

        slice_obj = slices[0]

        if is_storage_transposed:
            assert len(slice_obj.get_shape()) <= 2, (
                "sharding_weight_loader() does not support transposed storage for tensors with more than 2 dims"
            )

        # Determine which dim to slice in checkpoint
        storage_shard_dim = shard_dim
        if is_storage_transposed and shard_dim in (0, 1):
            storage_shard_dim = 1 - shard_dim

        # Compute slice range
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Build slice tuple
        slices = [slice(None)] * len(slice_obj.get_shape())
        slices[storage_shard_dim] = slice(start_idx, end_idx)
        result = slice_obj[tuple(slices)]

        if is_storage_transposed:
            result = result.T

        # Optionally pad to shard_size along shard_dim if the checkpoint
        # dimension is shorter (e.g. last rank when total_dim % num_shards != 0).
        if pad_shard:
            target_shape = list(result.shape)
            target_shape[shard_dim] = shard_size
            result = pad_to_shape(result, tuple(target_shape))

        return result

    return SafetensorsWeightLoader(transform=transform)


def sharding_weight_loader_with_padding(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    pad_dim: int | None = None,
    padded_size: int | None = None,
    unpadded_size: int | None = None,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that shards and optionally pads a tensor.

    This loader combines sharding (for tensor parallelism) with optional padding
    (for models that pad hidden dimensions for hardware alignment).

    Args:
        shard_dim: Dimension to shard along in the final parameter shape.
        shard_size: Size of each shard along shard_dim.
        num_shards: Total number of shards (typically TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed.
        pad_dim: Dimension to pad after sharding (0 or 1). None means no padding.
        padded_size: Target size after padding.
        unpadded_size: Original size before padding.

    Returns:
        SafetensorsWeightLoader configured for sharding with optional padding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "sharding_weight_loader_with_padding() only supports a single tensor"
        )

        slice_obj = slices[0]

        if is_storage_transposed:
            assert len(slice_obj.get_shape()) <= 2, (
                "sharding_weight_loader_with_padding() does not support transposed storage for tensors with more than 2 dims"
            )

        # Determine which dim to slice in checkpoint
        storage_shard_dim = shard_dim
        if is_storage_transposed and shard_dim in (0, 1):
            storage_shard_dim = 1 - shard_dim

        # Compute slice range
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Build slice tuple
        sl = [slice(None)] * len(slice_obj.get_shape())
        sl[storage_shard_dim] = slice(start_idx, end_idx)
        result = slice_obj[tuple(sl)]

        if is_storage_transposed:
            result = result.T

        # Apply padding if needed
        if (
            pad_dim is not None
            and padded_size
            and unpadded_size
            and padded_size != unpadded_size
        ):
            pad_amount = padded_size - unpadded_size
            if pad_dim == 0:
                result = torch.nn.functional.pad(result, (0, 0, 0, pad_amount))
            else:  # pad_dim == 1
                result = torch.nn.functional.pad(result, (0, pad_amount))

        return result

    return SafetensorsWeightLoader(transform=transform)


def last_dim_padding_weight_loader(padded_size: int) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that pads a tensor's last dimension.

    This loader is useful for models that pad hidden dimensions for hardware alignment.
    It handles the case where no padding is needed (padded_size <= tensor size).

    Args:
        padded_size: Target size for the last dimension.

    Returns:
        SafetensorsWeightLoader that pads the last dimension.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        tensor = slices[0][:]
        pad_amount = padded_size - tensor.shape[-1]
        if pad_amount <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_amount))

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    num_kv_replicas: int = 1,
    padded_hidden_size: int | None = None,
    unpadded_hidden_size: int | None = None,
    attention_dp_rank: int = 0,
    attention_dp_size: int = 1,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that fuses Q, K, V weights with per-tensor sharding.

    This loader expects 3 checkpoint tensors (Q, K, V) specified via mappings,
    shards each according to rank, and concatenates them into a single fused tensor.

    Args:
        q_size: Size of Q projection per shard.
        kv_size: Size of K (and V) projection per shard.
        shard_dim: Dimension to shard along in the final parameter shape.
        num_shards: Total number of shards (TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed.
        num_kv_replicas: Number of KV replicas. For example with 8 KV heads and TP 16,
            we typically give rank 0 and 1 the 1st KV head, and so on.
        padded_hidden_size: Target hidden size after padding (for models with padded dimensions).
        unpadded_hidden_size: Original hidden size before padding.
        attention_dp_rank: This rank's position within its attention DP column group.
            Default 0 (no attention DP).
        attention_dp_size: Dependent DP degree. Default 1 (disabled).
            When > 1, Q is sharded across TP * attention_dp using an effective rank.
        kv_sharded_across_attention_dp: Whether KV weights are also sharded across
            attention DP (True when num_kv_heads > TP and divisible by TP * attention_dp).

    Returns:
        SafetensorsWeightLoader configured for fused QKV loading.

    Example:
        Fused QKV for attention with GQA::
            set_weight_loader(param, fused_qkv_weight_loader(
                q_size=num_heads * head_dim // tp_size,
                kv_size=num_kv_heads * head_dim // tp_size,
                shard_dim=0,
                num_shards=tp_size,
            ))
    """
    assert shard_dim == 0 or shard_dim == 1, (
        f"Shard dim must be 0 or 1, got {shard_dim}"
    )

    if is_storage_transposed:
        storage_shard_dim = 1 - shard_dim
    else:
        storage_shard_dim = shard_dim

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        assert len(slices) == 3, (
            "fused_qkv_weight_loader expects [Q, K, V] slices in order"
        )
        assert len(slices[0].get_shape()) == 2, "Q slice must be 2 dimensional"
        assert len(slices[0].get_shape()) == 2, "K slice must be 2 dimensional"
        assert len(slices[0].get_shape()) == 2, "V slice must be 2 dimensional"

        # With attention DP, Q uses effective rank across TP * attention_dp
        local_rank = rank % num_shards
        if attention_dp_size > 1:
            q_rank = attention_dp_rank + local_rank * attention_dp_size
            if kv_sharded_across_attention_dp:
                kv_rank = q_rank
            else:
                kv_rank = local_rank // num_kv_replicas
        else:
            q_rank = local_rank
            kv_rank = local_rank // num_kv_replicas

        qkv_tensors = []
        for slice_obj, shard_size, shard_rank in zip(
            slices, [q_size, kv_size, kv_size], [q_rank, kv_rank, kv_rank]
        ):
            start_idx = shard_rank * shard_size
            end_idx = start_idx + shard_size

            sl = [slice(None)] * len(slice_obj.get_shape())
            sl[storage_shard_dim] = slice(start_idx, end_idx)
            tensor = slice_obj[tuple(sl)]
            qkv_tensors.append(tensor)

        # Transpose each tensor first if needed
        if is_storage_transposed:
            qkv_tensors = [t.T for t in qkv_tensors]

        # Fused QKV concatenates along shard dimension (e.g. we shard Q, K, and V weights)
        result = torch.cat(qkv_tensors, dim=shard_dim)

        # Apply padding if needed (pad the hidden dimension, which is dim 0 after transpose)
        if (
            padded_hidden_size
            and unpadded_hidden_size
            and padded_hidden_size != unpadded_hidden_size
        ):
            pad_amount = padded_hidden_size - unpadded_hidden_size
            # After transpose, result is [hidden, qkv_size], pad first dim
            result = torch.nn.functional.pad(result, (0, 0, 0, pad_amount))

        return result

    return SafetensorsWeightLoader(transform=transform)


def scaled_bias_loader(
    scale: float,
    padded_size: int | None = None,
) -> SafetensorsWeightLoader:
    """Create a weight loader that scales a bias tensor (for all-reduce averaging).

    This is used for row-parallel biases where each rank contributes a partial sum
    that gets all-reduced. The bias must be divided by the number of ranks to produce
    the correct result after summation.

    Args:
        scale: Factor to divide by (typically world_size for row-parallel biases).
        padded_size: If provided, pad the last dimension to this size.

    Returns:
        SafetensorsWeightLoader that divides the tensor by scale and optionally pads.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        tensor = slices[0][:] / scale

        if padded_size is not None:
            pad_amount = padded_size - tensor.shape[-1]
            if pad_amount > 0:
                tensor = torch.nn.functional.pad(tensor, (0, pad_amount))

        return tensor

    return SafetensorsWeightLoader(transform=transform)


def _contiguous_expert_bounds(local_expert_indices: list) -> tuple[int, int]:
    """Validate a contiguous expert range and return ``(lo, hi)`` inclusive.

    Only contiguous expert ranges (e.g. ``[ep_rank*L, (ep_rank+1)*L)``) are
    supported today; non-contiguous indices are rejected.
    """
    if len(local_expert_indices) == 0:
        raise ValueError("local_expert_indices cannot be empty")
    lo = local_expert_indices[0]
    hi = local_expert_indices[-1]
    if list(local_expert_indices) != list(range(lo, hi + 1)):
        raise ValueError(
            "expert-parallel weight loaders require contiguous "
            f"local_expert_indices, got {local_expert_indices}"
        )
    return lo, hi


class SliceView:
    """Lazy, composable view over a ``PySafeSlice`` (or tensor).

    Presents the checkpoint-slice interface (``get_shape()`` + ``__getitem__``)
    over a sub-region of ``base`` defined by a per-base-dim ``index`` (slices
    keep a dim, ints collapse it). Indexing the view composes the new key into
    that base selection and performs a SINGLE read on ``base``, so a caller
    that restricts one dim (e.g. an expert-parallel range on dim 0) and an
    inner loader that shards a different dim (e.g. a TP shard on dim 1)
    together materialize only the final shape -- no intermediate over-read.

    This exists because ``PySafeSlice`` (a native pyo3 type) supports only a
    single lazy ``__getitem__`` (which materializes a tensor) and cannot be
    subclassed or sub-sliced lazily. ``SliceView`` defers that one read until
    the composed selection is known.

    Supported index forms: ``slice``, ``int``, ``list`` (integer arrays), and
    ``Ellipsis`` -- the forms weight loaders use. ``None``/newaxis is rejected
    (it inserts a dim with no backing data -- a reshape, not a sub-selection;
    ``unsqueeze`` the result instead), as is boolean/tensor advanced indexing.
    Unsupported forms raise so a gap fails loudly rather than silently slicing
    the wrong region.
    """

    def __init__(self, base, index=None):
        self._base = base
        shape = self._base_shape()
        if index is None:
            index = tuple(slice(None) for _ in shape)
        if len(index) != len(shape):
            raise ValueError(
                f"SliceView index length {len(index)} != base ndim {len(shape)}"
            )
        self._index = tuple(self._normalize(sel, n) for sel, n in zip(index, shape))

    def _base_shape(self) -> tuple[int, ...]:
        get_shape = getattr(self._base, "get_shape", None)
        return tuple(get_shape()) if get_shape is not None else tuple(self._base.shape)

    @staticmethod
    def _normalize(sel, n):
        """Resolve a selector against dim length n to concrete form."""
        if isinstance(sel, slice):
            return slice(*sel.indices(n))  # concrete (start, stop, step)
        if isinstance(sel, int):
            return sel + n if sel < 0 else sel
        raise TypeError(f"SliceView selector must be slice or int, got {sel!r}")

    @staticmethod
    def _slice_len(s: slice) -> int:
        return len(range(s.start, s.stop, s.step))

    def get_shape(self) -> tuple[int, ...]:
        """Shape of the view: lengths of the kept (slice) dims, in order."""
        return tuple(
            self._slice_len(sel) for sel in self._index if isinstance(sel, slice)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.get_shape()

    def _expand_ellipsis(self, key: tuple, n_view_dims: int) -> tuple:
        """Replace a single Ellipsis with full slices covering the unspecified
        view dims (numpy/torch semantics)."""
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis == 0:
            return key
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        non_ellipsis = [k for k in key if k is not Ellipsis]
        fill = n_view_dims - len(non_ellipsis)
        if fill < 0:
            raise IndexError("too many indices for SliceView")
        out = []
        for k in key:
            if k is Ellipsis:
                out.extend([slice(None)] * fill)
            else:
                out.append(k)
        return tuple(out)

    def _compose(self, base_sel: slice, key):
        """Compose an incoming key element (view space) with a base slice
        (underlying space), returning a selector in underlying space."""
        n = self._slice_len(base_sel)
        if isinstance(key, slice):
            ks, ke, kstep = key.indices(n)
            new_start = base_sel.start + base_sel.step * ks
            new_step = base_sel.step * kstep
            new_stop = base_sel.start + base_sel.step * ke
            return slice(new_start, new_stop, new_step)
        if isinstance(key, int):
            k = key + n if key < 0 else key
            if not (0 <= k < n):
                raise IndexError(f"index {key} out of range for dim of size {n}")
            return base_sel.start + base_sel.step * k  # int -> collapses dim
        if isinstance(key, list):
            # List of positions (e.g. get_shuffled_shard's scattered indices).
            # Keeps the dim; map each position view->base coords. Identity when
            # base_sel is the full slice; offset/strided otherwise.
            out = []
            for k in key:
                kk = k + n if k < 0 else k
                if not (0 <= kk < n):
                    raise IndexError(f"index {k} out of range for dim of size {n}")
                out.append(base_sel.start + base_sel.step * kk)
            return out
        raise TypeError(
            f"SliceView does not support index element {key!r}; "
            f"only slice / int / list."
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is None for k in key):
            raise TypeError(
                "SliceView does not support newaxis (None); unsqueeze the "
                "materialized result instead"
            )
        view_dims = [d for d, sel in enumerate(self._index) if isinstance(sel, slice)]
        key = self._expand_ellipsis(key, len(view_dims))
        if len(key) > len(view_dims):
            raise IndexError(
                f"too many indices: got {len(key)} for {len(view_dims)} view dims"
            )
        # Pad with full slices for unspecified trailing view dims.
        key = list(key) + [slice(None)] * (len(view_dims) - len(key))

        composed = list(self._index)
        for vd, k in zip(view_dims, key):
            composed[vd] = self._compose(self._index[vd], k)
        return self._base[tuple(composed)]


# ---------------------------------------------------------------------------
# Expert-parallel (EP) weight loaders.
#
# Each rank owns a subset of experts. These wrappers restrict the inner
# loader's INPUT to the local experts *before* it runs, so the per-expert work
# (e.g. MXFP4 dequant) only touches local experts rather than all experts
# followed by discarding the non-local ones. They differ only by how experts
# are laid out in the inner loader's ``slices``.
#
# ``original_loader`` must be constructed for ``num_local_experts`` and must
# define a transform (these wrappers call ``original_loader.transform``
# directly; a bare ``SafetensorsWeightLoader()`` is not supported).
# ---------------------------------------------------------------------------


def expert_parallel_tensor_dim_loader(
    local_expert_indices: list,
    original_loader: SafetensorsWeightLoader,
    expert_dim: int = 0,
) -> SafetensorsWeightLoader:
    """EP loader for inputs that are single tensors with experts on a dim.

    Each ``slices`` entry is one tensor whose ``expert_dim`` indexes experts
    (e.g. GPT-OSS ``[E, ...]`` weight/scale/bias). Wraps each input in a
    :class:`SliceView` restricted to the local expert range, so the inner
    loader's own indexing (e.g. a TP shard on another dim) composes into a
    single read of just the final shape -- the non-local experts and the
    TP-discarded portion are never materialized.
    """
    lo, hi = _contiguous_expert_bounds(local_expert_indices)

    def transform(slices: list, rank: int) -> torch.Tensor:
        local_slices = []
        for s in slices:
            index = [slice(None)] * len(s.get_shape())
            index[expert_dim] = slice(lo, hi + 1)
            local_slices.append(SliceView(s, index=index))
        return original_loader.transform(local_slices, rank)

    return SafetensorsWeightLoader(transform=transform)


def _items_per_expert(slices: list, total_num_experts: int, kind: str) -> int:
    """Derive the per-expert item count (stride) from the full input list.

    ``slices`` (pre-slice) has ``items_per_expert * total_num_experts``
    entries; the stride is recovered as ``len(slices) // total_num_experts``.
    Deriving it (rather than passing it in) handles loaders whose stride is
    only known at load time -- e.g. deepseek gate_up is stride 2 (bf16) or 4
    (fp8).

    ``total_num_experts`` must be the TOTAL expert count (the number the
    checkpoint mapping enumerated), not this rank's local count.
    """
    if total_num_experts <= 0:
        raise ValueError(f"total_num_experts must be positive, got {total_num_experts}")
    if len(slices) % total_num_experts != 0:
        raise ValueError(
            f"{kind} EP loader expects len(slices) divisible by "
            f"total_num_experts={total_num_experts}, got {len(slices)}"
        )
    return len(slices) // total_num_experts


def expert_parallel_grouped_loader(
    local_expert_indices: list,
    original_loader: SafetensorsWeightLoader,
    total_num_experts: int,
) -> SafetensorsWeightLoader:
    """EP loader for a flat list grouped by item across experts.

    ``slices`` is ``[item0_e0..item0_e{E-1}, item1_e0..item1_e{E-1}, ...]``
    with ``items_per_expert`` groups -- e.g. qwen3 fused gate_up
    ``[gate_0..gate_{E-1}, up_0..up_{E-1}]`` (2 groups), or a single
    per-expert weight like down_proj as a list ``[down_0..down_{E-1}]``
    (1 group). The per-group stride is derived from ``total_num_experts``;
    selects ``[lo:hi+1]`` within each group.
    """
    lo, hi = _contiguous_expert_bounds(local_expert_indices)

    def transform(slices: list, rank: int) -> torch.Tensor:
        items_per_expert = _items_per_expert(slices, total_num_experts, "grouped")
        local_slices = []
        for g in range(items_per_expert):
            base = g * total_num_experts
            local_slices.extend(slices[base + lo : base + hi + 1])
        return original_loader.transform(local_slices, rank)

    return SafetensorsWeightLoader(transform=transform)


def expert_parallel_interleaved_loader(
    local_expert_indices: list,
    original_loader: SafetensorsWeightLoader,
    total_num_experts: int,
) -> SafetensorsWeightLoader:
    """EP loader for a flat list interleaved per expert.

    ``slices`` is ``[e0_item0..e0_item{K-1}, e1_item0.., ...]`` with stride
    ``K`` -- e.g. deepseek gate_up bf16 ``[gate_0, up_0, gate_1, up_1, ...]``
    (K=2), or deepseek gate_up fp8 ``[gate_w_0, gate_s_0, up_w_0, up_s_0,
    ...]`` (K=4). The stride is derived from ``total_num_experts``; selects
    the contiguous per-expert block for the local experts.
    """
    lo, hi = _contiguous_expert_bounds(local_expert_indices)

    def transform(slices: list, rank: int) -> torch.Tensor:
        items_per_expert = _items_per_expert(slices, total_num_experts, "interleaved")
        local_slices = slices[lo * items_per_expert : (hi + 1) * items_per_expert]
        return original_loader.transform(local_slices, rank)

    return SafetensorsWeightLoader(transform=transform)


######################################
# Common functions for composability #
######################################


def get_shard(
    slice_obj: "PySafeSlice", dim: int, shard_size: int, num_shards: int, rank: int
) -> torch.Tensor:
    """Extract a shard from a Safetensors PySafeSlice without loading the full tensor.

    Only reads the shard portion from disk, avoiding full tensor materialization.
    Handles unpadded checkpoints where dim size < shard_size * num_shards.

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension to shard along
        shard_size: Size of each shard
        num_shards: Total number of shards
        rank: Current rank (0-indexed)

    Returns:
        Tensor of shape [..., shard_size, ...] on dim, or smaller if checkpoint
        is unpadded. Caller should pad if result.shape[dim] < shard_size.

    Example:
        # slice_obj shape: [8, 119, 256]
        # dim = 1
        # shard_size = 40
        # num_shards =

        # Rank 0: gets [:, 0:40, :]   -> shape [8, 40, 256]
        # Rank 1: gets [:, 40:80, :]  -> shape [8, 40, 256]
        # Rank 2: gets [:, 80:119, :] -> shape [8, 39, 256], needs 1 pad

        tensor = get_shard(slice_obj, dim=1, shard_size=40, rank=2, num_shards=3)
        # tensor.shape[1] == 39, caller pads to 40
    """
    shape = slice_obj.get_shape()

    start = (rank % num_shards) * shard_size
    end = start + shard_size

    slices = [slice(None)] * len(shape)
    slices[dim] = slice(start, end)

    return slice_obj[tuple(slices)]


def get_shuffled_shard(
    slice_obj: "PySafeSlice",
    dim: int,
    shard_size: int,
    rank: int,
    padded_dim_size: int | None = None,
) -> torch.Tensor:
    """Extract a shard equivalent to pad-then-shuffle-then-shard, without materializing the full tensor.

    The shuffle transform: [H] -> [H//4, 4] -> transpose -> [4, H//4] -> [H]
    For H=8: [0,1,2,3,4,5,6,7] -> [0,4,1,5,2,6,3,7]

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension to shuffle and shard
        shard_size: Size of each shard
        rank: Current rank (0-indexed)
        padded_dim_size: Size of dimension after padding (if None, no padding)

    Returns:
        Tensor equivalent to padding, shuffling, then sharding
    """
    shape = slice_obj.get_shape()
    H_orig = shape[dim]
    H = padded_dim_size if padded_dim_size is not None else H_orig
    stride = H // 4

    # Compute original indices for this shard's shuffled positions
    start_pos = rank * shard_size
    positions = torch.arange(shard_size)
    indices = ((start_pos + positions) % stride) * 4 + (
        (start_pos + positions) // stride
    )

    # Separate valid indices (< H_orig) from padding indices (>= H_orig)
    valid_mask = indices < H_orig
    valid_indices = indices[valid_mask]

    # Build output shape
    out_shape = list(shape)
    out_shape[dim] = shard_size
    output = torch.zeros(out_shape, dtype=slice_obj[tuple([0] * len(shape))].dtype)

    if valid_indices.numel() > 0:
        # Fetch valid data
        sl = [slice(None)] * len(shape)
        sl[dim] = valid_indices.tolist()
        valid_data = slice_obj[tuple(sl)]

        # Scatter into output at valid positions
        out_sl = [slice(None)] * len(shape)
        out_sl[dim] = valid_mask.nonzero().squeeze(-1).tolist()
        output[tuple(out_sl)] = valid_data

    return output


def get_shard_deinterleaved(
    slice_obj: "PySafeSlice",
    dim: int,
    shard_size: int,
    num_shards: int,
    rank: int,
) -> torch.Tensor:
    """Extract a shard of interleaved gate/up data, returning [gate_shard, up_shard] concatenated.

    For checkpoint data stored as [g0, u0, g1, u1, g2, u2, g3, u3] along dim,
    each rank gets its portion of both gate and up, concatenated.

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension with interleaved data
        shard_size: Size of each shard (should be 2 * I_per_rank, i.e., gate + up per rank)
        num_shards: Total number of shards (tp_degree)
        rank: Current rank (0-indexed)

    Returns:
        Tensor with [gate_shard, up_shard] concatenated along dim

    Example:
        # slice_obj data along dim: [g0, u0, g1, u1, g2, u2, g3, u3] (interleaved)
        # shard_size=4, num_shards=2

        # Rank 0: gets [g0, g1, u0, u1] (first half of gate + first half of up)
        # Rank 1: gets [g2, g3, u2, u3] (second half of gate + second half of up)
    """
    shape = slice_obj.get_shape()
    ndim = len(shape)

    start = (rank % num_shards) * shard_size
    end = start + shard_size

    # Gate: even indices [start*2 : end*2 : 2]
    sl_gate = [slice(None)] * ndim
    sl_gate[dim] = slice(start, end, 2)
    gate_shard = slice_obj[tuple(sl_gate)]

    # Up: odd indices [start*2 + 1 : end*2 + 1 : 2]
    sl_up = [slice(None)] * ndim
    sl_up[dim] = slice(start + 1, end + 1, 2)
    up_shard = slice_obj[tuple(sl_up)]

    return gate_shard, up_shard


def pad_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Pad tensor to target shape. Pads at the end of each dimension.

    Args:
        tensor: Input tensor (possibly smaller than target on some dims)
        target_shape: Expected shape after padding

    Returns:
        Tensor padded to target_shape, or original if no padding needed.

    Example:
        # After sharding, assume rank X has shape (8, 39) but needs (8, 40)
        tensor = pad_to_shape(tensor, target_shape=(8, 40))
    """
    # Skip padding if shapes already match
    if tensor.shape == target_shape:
        return tensor

    pad = []
    for actual, target in zip(reversed(tensor.shape), reversed(target_shape)):
        pad.extend([0, target - actual])

    return torch.nn.functional.pad(tensor, pad)
