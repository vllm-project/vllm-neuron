# SPDX-License-Identifier: Apache-2.0
"""In-place DeltaNet decode kernel against a float64 reference and the batched kernel.

The in-place kernel (`vllm_neuron/nki/deltanet/decode_inplace.py`) reads the state rows
of the batch by slot index and writes the new rows back to the same rows. These tests
check the result of the kernel, the state rows of other requests (they must not
change), the model wrapper and the selection of the kernel.

Run with the NKI simulator on CPU:
    NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        pytest test/vllm_neuron/model/qwen3_5/test_deltanet_decode_inplace_sim.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="Set NKI_SIMULATOR=1 to run the in-place DeltaNet decode kernel in the NKI simulator.",
)

pytest.importorskip("nki")

from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel  # noqa: E402

from vllm_neuron.model.neuron_config import NeuronConfig  # noqa: E402
from vllm_neuron.model.qwen3_5 import model_bf16 as m  # noqa: E402

from .test_deltanet_prefill_v2_sim import (  # noqa: E402
    GROUP,
    NUM_KH,
    _decode_reference,
    _layer_config,
    _NoOpGroup,
    _raw,
    _rel_rms,
)

DIM = 128


def _inplace_inputs(batch, num_rows, gscale=1.0, seed=0):
    """Kernel inputs for batch rows with 6 v-heads, and a state buffer of num_rows rows."""
    gen = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    hv = NUM_KH * GROUP
    rows = batch * hv
    return dict(
        query=torch.randn(rows, DIM, generator=gen).to(bf),
        key=torch.randn(rows, DIM, generator=gen).to(bf),
        value=torch.randn(rows, DIM, generator=gen).to(bf),
        z=torch.randn(rows, DIM, generator=gen).to(bf),
        a=(torch.randn(rows, 1, generator=gen) * gscale).to(bf),
        b=torch.randn(rows, 1, generator=gen).to(bf),
        a_log=torch.rand(hv, generator=gen).repeat(batch).reshape(rows, 1) * 2 - 2,
        dt_bias=torch.rand(hv, generator=gen).repeat(batch).reshape(rows, 1) * 2 - 1,
        norm_weight=(1 + 0.1 * torch.randn(1, DIM, generator=gen)).to(bf),
        state_buf=(torch.randn(num_rows, hv, DIM, DIM, generator=gen) * 0.5).to(bf),
        slot=torch.randperm(num_rows, generator=gen)[:batch]
        .to(torch.int32)
        .reshape(batch, 1),
        identity=torch.eye(128),
    )


def _inplace_reference(x, eps):
    """float64 decode step for the inputs of _inplace_inputs. Returns (out, new state rows)."""
    import torch.nn.functional as F

    hv = NUM_KH * GROUP
    slot = x["slot"].reshape(-1).long()
    d = {k: v.double() for k, v in x.items() if k not in ("slot", "state_buf")}
    state = x["state_buf"][slot].double().reshape(-1, DIM, DIM)
    qn = F.normalize(d["query"], p=2, dim=-1, eps=1e-6) * DIM**-0.5
    kn = F.normalize(d["key"], p=2, dim=-1, eps=1e-6)
    g = -d["a_log"].exp() * F.softplus(d["a"] + d["dt_bias"])
    s = state * g.exp().unsqueeze(-1)
    delta = (d["value"] - (s * kn.unsqueeze(-1)).sum(-2)) * d["b"].sigmoid()
    s = s + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (s * qn.unsqueeze(-1)).sum(-2)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * d["norm_weight"]
    return o * F.silu(d["z"]), s.reshape(-1, hv, DIM, DIM)


@pytest.mark.parametrize(
    "batch,num_rows,grid,gscale",
    (
        (2, 5, 2, 1.0),
        (5, 9, 1, 1.0),
        (24, 30, 2, 12.0),
        # 1 program, 23 rows of 6 v-heads: 2 tiles of 12 and 11 rows.
        (23, 26, 1, 1.0),
    ),
)
def test_inplace_decode_kernel_matches_reference(batch, num_rows, grid, gscale):
    from vllm_neuron.nki.deltanet.decode_inplace import deltanet_decode_inplace

    x = _inplace_inputs(batch, num_rows, gscale)
    state0 = x["state_buf"].clone()
    # The simulator can write the state buffer in place, so take the reference first.
    ref_out, ref_rows = _inplace_reference(x, 1e-6)
    out, new_buf = simulate_nki_kernel(
        _raw(deltanet_decode_inplace), grid, dict(x, norm_eps=1e-6)
    )
    slot = x["slot"].reshape(-1).long()
    assert out.dtype == torch.bfloat16 and new_buf.dtype == torch.bfloat16
    assert new_buf.shape == state0.shape
    # The kernel uses float32 math. Its error is the rounding of the result to bf16.
    assert (
        _rel_rms(out.float(), ref_out)
        <= 1.05 * _rel_rms(ref_out.to(torch.bfloat16).float(), ref_out) + 1e-6
    )
    state_round = _rel_rms(ref_rows.to(torch.bfloat16).double(), ref_rows)
    assert _rel_rms(new_buf[slot].double(), ref_rows) <= 1.05 * state_round + 1e-6
    # The rows of other requests do not change.
    others = torch.ones(num_rows, dtype=torch.bool)
    others[slot] = False
    assert torch.equal(new_buf[others], state0[others])


def _bf16_layer(monkeypatch, num_seqs=8):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    config = _layer_config()
    config.torch_dtype = torch.bfloat16
    config.neuron_config = NeuronConfig(num_seqs_buckets=[num_seqs])
    layer = m.Qwen3_5DeltaNet(config, layer_idx=0)
    torch.manual_seed(42)
    for name, p in layer.named_parameters():
        with torch.no_grad():
            if name == "a_log":
                p.uniform_(-2.0, 0.0)
            elif name == "dt_bias":
                p.uniform_(-1.0, 1.0)
            elif name in ("recurrent_state", "conv_state"):
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.05)
    return layer


@pytest.mark.parametrize("batch", (1, 4))
def test_inplace_decode_step_matches_reference(monkeypatch, batch):
    layer = _bf16_layer(monkeypatch)
    hk, hv = NUM_KH, NUM_KH * GROUP
    torch.manual_seed(11)
    bf = torch.bfloat16
    q = torch.randn(batch, 1, hk, DIM).to(bf)
    k = torch.randn(batch, 1, hk, DIM).to(bf)
    v = torch.randn(batch, 1, hv, DIM).to(bf)
    a = torch.randn(batch, hv).to(bf)
    b = torch.randn(batch, hv).to(bf)
    z = torch.randn(batch, hv * DIM).to(bf)
    slots = torch.tensor([6, 1, 3, 0][:batch])
    state0 = layer.recurrent_state.detach().clone()
    grid = layer._inplace_decode_grid(q)
    assert grid == (2 if batch % 2 == 0 else 1)
    out = layer._nki_inplace_decode_step(q, k, v, a, b, z, slots, grid)
    ref_out, ref_state = _decode_reference(layer, q, k, v, a, b, z, state0[slots])
    new = layer.recurrent_state.detach()
    assert out.shape == (batch, hv * DIM)
    assert _rel_rms(out.float(), ref_out) < 3e-3
    assert _rel_rms(new[slots].double(), ref_state) < 3e-3
    others = torch.ones(new.shape[0], dtype=torch.bool)
    others[slots] = False
    assert torch.equal(new[others], state0[others])


def test_layer_decode_inplace_matches_batched(monkeypatch):
    layer = _bf16_layer(monkeypatch)
    state0 = layer.recurrent_state.detach().clone()
    conv0 = layer.conv_state.detach().clone()
    batch = 5
    slots = torch.tensor([4, 0, 6, 2, 5])
    h = torch.randn(batch, layer.hidden_size).to(torch.bfloat16)
    calls = []
    step = layer._nki_inplace_decode_step

    def counted_step(*args):
        calls.append(args[0].shape[0])
        return step(*args)

    monkeypatch.setattr(layer, "_nki_inplace_decode_step", counted_step)

    def run(kernel):
        with torch.no_grad():
            layer.recurrent_state.copy_(state0)
            layer.conv_state.copy_(conv0)
        layer._dn_decode_kernel = kernel
        out = layer(
            h, torch.full((batch,), 7), is_prefill=False, batch_slot_indices=slots
        )
        return out.detach().float(), layer.recurrent_state.detach().float()

    out1, st1 = run("batched")
    out2, st2 = run("inplace")
    # One call takes the whole batch.
    assert calls == [batch]
    assert torch.isfinite(out2).all() and torch.isfinite(st2).all()
    assert _rel_rms(out2, out1) < 3e-3
    assert _rel_rms(st2[slots], st1[slots]) < 3e-3
    # The rows of other requests do not change.
    others = torch.tensor([1, 3, 7])
    assert torch.equal(st2[others], state0.float()[others])
    assert not torch.equal(st2[slots], state0.float()[slots])


def test_inplace_decode_falls_back_to_batched(monkeypatch):
    """A batch that the in-place kernel cannot take uses the batched kernel."""
    layer = _bf16_layer(monkeypatch)
    batch = 3
    slots = torch.tensor([2, 7, 0])
    h = torch.randn(batch, layer.hidden_size).to(torch.bfloat16)
    # 1 row of 6 v-heads needs 6 x 128 x 2 bytes for each partition. Allow less.
    monkeypatch.setattr(m, "_INPLACE_DECODE_MAX_STATE_BYTES", 1024)
    calls = []
    step = layer._nki_batched_decode_step

    def counted_step(*args):
        calls.append(args[0].shape[0])
        return step(*args)

    monkeypatch.setattr(layer, "_nki_batched_decode_step", counted_step)
    layer._dn_decode_kernel = "inplace"
    out = layer(h, torch.full((batch,), 7), is_prefill=False, batch_slot_indices=slots)
    assert calls == [batch]
    assert torch.isfinite(out.float()).all()


def test_inplace_decode_grid_limits(monkeypatch):
    layer = _bf16_layer(monkeypatch)
    hv = NUM_KH * GROUP

    def grid_for(batch):
        return layer._inplace_decode_grid(torch.zeros(batch, 1, NUM_KH, DIM))

    # A program keeps rows x v-heads x 128 bf16 values on each partition, at most 96 KiB.
    max_rows = m._INPLACE_DECODE_MAX_STATE_BYTES // (hv * DIM * 2)
    assert grid_for(2) == 2
    assert grid_for(3) == 1
    assert grid_for(2 * max_rows) == 2
    assert grid_for(2 * max_rows + 2) == 0
    assert grid_for(max_rows + 1) == 0
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    assert grid_for(2) == 0


def test_inplace_decode_grid_needs_a_bf16_state(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    layer = m.Qwen3_5DeltaNet(_layer_config(), layer_idx=0)
    assert layer.recurrent_state.dtype == torch.float32
    assert layer._inplace_decode_grid(torch.zeros(2, 1, NUM_KH, DIM)) == 0


def test_decode_kernel_setting_accepts_inplace(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL", "inplace")
    layer = m.Qwen3_5DeltaNet(_layer_config(), layer_idx=0)
    assert layer._dn_decode_kernel == "inplace"
