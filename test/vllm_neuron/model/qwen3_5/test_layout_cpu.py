"""CPU tests for the layout helper of the qwen3_5 model.

`_swap_adjacent_dims` exchanges 2 dimensions that are next to each other. It uses a
reshape when one of the 2 dimensions has the length 1, because a transpose makes a
Copy that the compiler backend refuses at a decode batch of 4 or more rows. These
tests show that the reshape gives the same result as the transpose.

Run these tests on a CPU. They need no Neuron device.
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.model_bf16 import _swap_adjacent_dims

# Shapes that a decode step uses. One of the 2 dimensions has the length 1, so the
# reshape path runs. Each entry holds the shape and the first dimension to exchange.
_RESHAPE_CASES = [
    ((4, 1, 6, 128), 1),  # query, key, value before the recurrence
    ((4, 6, 1, 128), 1),  # the output of the recurrence
    ((8, 1, 6, 64), 1),
    ((1, 1, 6, 128), 1),  # a batch of 1 row
    ((4, 1, 6), 1),  # g and beta, which have no last dimension
    ((4, 6, 1), 1),
    ((4, 3, 1, 256), 1),  # the output of the full attention layer
    ((4, 16, 1, 128, 256), 1),  # the gathered key cache, at 1 key head per rank
    ((4, 1, 256), 0),  # key and value of the full attention layer, 1 key head
    ((1, 3, 256), 0),  # query of the full attention layer, a batch of 1 row
]

# Shapes where neither dimension has the length 1, so the helper must take the
# transpose path. A prefill step and a wider rank give these shapes.
_TRANSPOSE_CASES = [
    ((2, 6, 5, 128), 1),
    ((1, 48, 7, 256), 1),
    ((4, 16, 3, 128, 256), 1),  # the gathered key cache, at 3 key heads per rank
    ((4, 3, 256), 0),  # query of the full attention layer, a batch of 4 rows
    ((7, 2, 256), 0),
]

_ALL_CASES = _RESHAPE_CASES + _TRANSPOSE_CASES


def _reference(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Give the result that the helper must match."""
    return tensor.transpose(dim, dim + 1).contiguous()


@pytest.mark.parametrize("shape,dim", _RESHAPE_CASES)
def test_the_helper_matches_a_transpose_at_a_length_of_one(shape, dim):
    # The reshape path runs. Its result must equal the transpose result exactly.
    torch.manual_seed(42)
    tensor = torch.randn(*shape)
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor, dim), _reference(tensor, dim), rtol=0, atol=0
    )


@pytest.mark.parametrize("shape,dim", _TRANSPOSE_CASES)
def test_the_helper_matches_a_transpose_at_a_larger_length(shape, dim):
    # The transpose path runs. The helper must not change a prefill step.
    torch.manual_seed(42)
    tensor = torch.randn(*shape)
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor, dim), _reference(tensor, dim), rtol=0, atol=0
    )


@pytest.mark.parametrize("shape,dim", _ALL_CASES)
def test_the_helper_matches_a_permute_of_the_same_2_dimensions(shape, dim):
    # The full attention layer writes the same exchange as a permute. Show that the
    # helper agrees with the permute form at every shape.
    torch.manual_seed(42)
    tensor = torch.randn(*shape)
    order = list(range(tensor.dim()))
    order[dim], order[dim + 1] = order[dim + 1], order[dim]
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor, dim),
        tensor.permute(*order).contiguous(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("shape,dim", _ALL_CASES)
def test_the_helper_accepts_a_tensor_that_is_not_contiguous(shape, dim):
    # A caller can give a view. A reshape of a view can need a copy, and the result
    # must stay the same. Build the view from a tensor that has the reverse shape.
    # `torch.flip` does not give a view, because it copies the data.
    torch.manual_seed(42)
    base = torch.randn(*reversed(shape))
    tensor = base.permute(*reversed(range(base.dim())))
    assert tuple(tensor.shape) == tuple(shape)
    assert not tensor.is_contiguous()
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor, dim), _reference(tensor, dim), rtol=0, atol=0
    )


def test_the_helper_gives_the_exchanged_shape():
    # The shape contract holds in both directions and at both dimensions.
    assert _swap_adjacent_dims(torch.zeros(4, 1, 6, 128)).shape == (4, 6, 1, 128)
    assert _swap_adjacent_dims(torch.zeros(4, 6, 1, 128)).shape == (4, 1, 6, 128)
    assert _swap_adjacent_dims(torch.zeros(4, 1, 256), dim=0).shape == (1, 4, 256)
    assert _swap_adjacent_dims(torch.zeros(4, 3, 256), dim=0).shape == (3, 4, 256)
    assert _swap_adjacent_dims(torch.zeros(4, 16, 1, 128, 256)).shape == (
        4,
        1,
        16,
        128,
        256,
    )


def test_the_default_dimension_is_one():
    # Most callers exchange dimension 1 and dimension 2, so 1 is the default.
    torch.manual_seed(42)
    tensor = torch.randn(4, 1, 6, 128)
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor), _swap_adjacent_dims(tensor, 1), rtol=0, atol=0
    )


@pytest.mark.parametrize("shape,dim", _TRANSPOSE_CASES)
def test_a_reshape_alone_does_not_replace_the_transpose(shape, dim):
    # This test guards the length check in the helper. A plain reshape gives a
    # different result when neither dimension has the length 1. So the helper must
    # keep the transpose path for these shapes.
    torch.manual_seed(42)
    tensor = torch.randn(*shape)
    swapped = list(shape)
    swapped[dim], swapped[dim + 1] = swapped[dim + 1], swapped[dim]
    assert not torch.equal(tensor.reshape(swapped), _reference(tensor, dim))
    torch.testing.assert_close(
        _swap_adjacent_dims(tensor, dim), _reference(tensor, dim), rtol=0, atol=0
    )
