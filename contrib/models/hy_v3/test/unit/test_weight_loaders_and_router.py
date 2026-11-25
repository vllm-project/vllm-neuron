# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Hy3 FP8 weight transforms and the router.

No Neuron device or checkpoint is needed:

    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hy_v3/test/unit/ -v
"""

import pytest
import torch
from hy_v3_neuron.weight_loaders import (
    FP8_DTYPE,
    PMAX,
    expert_down_loader,
    expert_gate_up_loader,
    expert_scale_loader,
    halve_fp8,
    padded_mlp_loader,
    scalar_scale_loader,
)

TRN2_FP8_MAX = 240.0


class _Slice:
    """Minimal stand-in for safetensors' ``PySafeSlice``."""

    def __init__(self, t: torch.Tensor):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        return self.t[idx]


def _random_fp8(*shape) -> torch.Tensor:
    return (torch.randn(*shape) * 100).clamp(-448, 448).to(FP8_DTYPE)


def test_halve_fp8_is_exact_and_fits_trn2_range():
    w = _random_fp8(256, 256)
    h = halve_fp8(w)
    assert h.dtype == FP8_DTYPE
    assert h.float().abs().max() <= TRN2_FP8_MAX
    # Halving is an exponent shift: exact except for values that become subnormal.
    normal = w.float().abs() >= 2**-5
    torch.testing.assert_close(
        h.float()[normal], w.float()[normal] * 0.5, rtol=0, atol=0
    )


def test_scalar_scale_loader_fused_qkv_and_shared_input_scale():
    scales = [torch.tensor(0.5), torch.tensor(0.25), torch.tensor(0.125)]
    out = scalar_scale_loader(3, cols=3).transform([_Slice(s) for s in scales], 0)
    assert out.shape == (PMAX, 3) and out.dtype == torch.float32
    torch.testing.assert_close(out[0], torch.tensor([1.0, 0.5, 0.25]))

    shared = scalar_scale_loader(2, cols=1).transform(
        [_Slice(torch.tensor(0.5)), _Slice(torch.tensor(0.5))], 0
    )
    assert shared.shape == (PMAX, 1)
    assert shared[0, 0].item() == 1.0
    with pytest.raises(AssertionError):
        scalar_scale_loader(2, cols=1).transform(
            [_Slice(torch.tensor(0.5)), _Slice(torch.tensor(0.25))], 0
        )


@pytest.mark.parametrize("is_down", [False, True])
def test_padded_mlp_loader_shards_and_zero_pads(is_down):
    H, I, shard, padded, rank = 16, 52, 13, 128, 3
    w = _random_fp8(H, I) if is_down else _random_fp8(I, H)
    out = padded_mlp_loader(is_down, shard, padded, rank).transform([_Slice(w)], rank)
    if is_down:  # [H, I] -> [padded, H]
        assert out.shape == (padded, H)
        expect = w[:, rank * shard : (rank + 1) * shard].T
        real, pad = out[:shard], out[shard:]
    else:  # [I, H] -> [H, padded]
        assert out.shape == (H, padded)
        expect = w[rank * shard : (rank + 1) * shard].T
        real, pad = out[:, :shard], out[:, shard:]
    torch.testing.assert_close(real.float(), halve_fp8(expect.contiguous()).float())
    assert pad.float().abs().max() == 0


def test_expert_loaders_layout():
    L, H, I, i_shard, i_rank = 3, 8, 32, 16, 1
    gates = [_random_fp8(I, H) for _ in range(L)]
    ups = [_random_fp8(I, H) for _ in range(L)]
    downs = [_random_fp8(H, I) for _ in range(L)]

    gu = expert_gate_up_loader(L, i_shard, i_rank).transform(
        [_Slice(t) for t in gates + ups], 0
    )
    assert gu.shape == (L, H, 2, i_shard)
    rows = slice(i_rank * i_shard, (i_rank + 1) * i_shard)
    for e in range(L):
        torch.testing.assert_close(
            gu[e, :, 0].float(), halve_fp8(gates[e][rows]).T.float()
        )
        torch.testing.assert_close(
            gu[e, :, 1].float(), halve_fp8(ups[e][rows]).T.float()
        )

    dn = expert_down_loader(L, i_shard, i_rank).transform([_Slice(t) for t in downs], 0)
    assert dn.shape == (L, i_shard, H)
    for e in range(L):
        torch.testing.assert_close(
            dn[e].float(), halve_fp8(downs[e][:, rows].contiguous()).T.float()
        )

    # Scales arrive grouped by projection: [gate_0..gate_{L-1}, up_0..up_{L-1}].
    gate_s = [torch.tensor(float(e + 1)) for e in range(L)]
    up_s = [torch.tensor(float(10 * (e + 1))) for e in range(L)]
    tkg = expert_scale_loader(L, 2, lambda s: s.view(L, 2, 1)).transform(
        [_Slice(t) for t in gate_s + up_s], 0
    )
    for e in range(L):
        assert tkg[e, 0, 0].item() == 2 * (e + 1)  # x2 FP8 compensation
        assert tkg[e, 1, 0].item() == 20 * (e + 1)


def test_router_matches_hf():
    from hy_v3_neuron.model import hy3_route
    from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config as HFConfig
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter

    torch.manual_seed(0)
    E, K, H, T, scale = 32, 4, 64, 16, 2.826
    cfg = HFConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        router_scaling_factor=scale,
    )
    hf = HYV3TopKRouter(cfg)
    hf.weight.data = torch.randn(E, H) * 0.1
    bias = torch.randn(E) * 0.05
    x = torch.randn(T, H, dtype=torch.bfloat16)

    _, hf_w, hf_idx = hf(x, bias)
    aff, idx = hy3_route(x, hf.weight.float(), bias, K, True, scale)

    assert torch.equal(idx.sort(-1).values, hf_idx.sort(-1).values)
    hf_dense = torch.zeros(T, E).scatter(1, hf_idx, hf_w)
    torch.testing.assert_close(aff, hf_dense)
    # The bias only picks experts; the weights come from the unbiased scores.
    torch.testing.assert_close(aff.sum(-1), torch.full((T,), scale))
