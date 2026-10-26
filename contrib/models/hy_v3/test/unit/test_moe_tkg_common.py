# SPDX-License-Identifier: Apache-2.0
"""Drift guard for src/kernels/moe_tkg_common.py against the nkilib originals.

    pytest contrib/models/hy_v3/test/unit/test_moe_tkg_common.py
"""

import itertools
import types

import pytest

nl = pytest.importorskip("nki.language")
moe_tkg = pytest.importorskip("nkilib.core.moe.moe_tkg.moe_tkg")
from hy_v3_neuron.kernels.moe_tkg_common import quantization_type  # noqa: E402


def test_quantization_type_matches_nkilib():
    ref = getattr(moe_tkg, "_extract_quantization_type", None)
    if ref is None:
        pytest.skip("nkilib has no _extract_quantization_type")
    w = types.SimpleNamespace(dtype=nl.bfloat16)
    s = types.SimpleNamespace(dtype=nl.float32, shape=(1, 1))
    for args in itertools.product([None, s], repeat=4):
        want, is_mx = ref(w, *args)
        assert not is_mx
        assert quantization_type(w, *args) == want, args
