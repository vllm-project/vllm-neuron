# SPDX-License-Identifier: Apache-2.0
"""DeltaNet prefill kernel v2 against v1 and a float64 reference, in the NKI simulator.

The v2 kernel (`vllm_neuron/nki/deltanet/fused_v2.py`) has 2 formulations:

- "v1math" runs the chunk body of v1. Its output must be bit-identical to v1,
  both with repeated q/k (one row for each v-head) and with GQA by index.
- "fused" changes the data flow. Its error against a float64 recurrence must
  stay within 1.5 times the error of v1.

The opt-in bf16 modes change the numerics. These tests only bound their error.

Run with the NKI simulator on CPU:
    NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        pytest test/vllm_neuron/model/qwen3_5/test_deltanet_prefill_v2_sim.py
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="Set NKI_SIMULATOR=1 to run the DeltaNet v2 kernel in the NKI simulator.",
)

pytest.importorskip("nki")

from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel  # noqa: E402

from vllm_neuron.model.qwen3_5 import model_bf16 as m  # noqa: E402
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config  # noqa: E402
from vllm_neuron.nki.deltanet.fused import deltanet_fused_chunked_fwd_multihead  # noqa: E402
from vllm_neuron.nki.deltanet.fused_v2 import deltanet_prefill_v2, make_v2_constants  # noqa: E402

REGIMES = ("random", "gscale12", "parallel", "zerotail")
NUM_KH = 2
GROUP = 3
SEQ = 256


def _raw(kernel):
    return kernel.func if hasattr(kernel, "func") else kernel


def _inputs(regime: str, seed: int = 0):
    """Return q, k ([Hk, S, 128] bf16), v ([Hv, S, 128] bf16), g, beta, state0.

    The regimes follow tests/README_qwen3_5.md section 7.2.
    """
    gen = torch.Generator().manual_seed(seed)
    num_vh = NUM_KH * GROUP
    q = torch.randn(NUM_KH, SEQ, 128, generator=gen)
    k = torch.randn(NUM_KH, SEQ, 128, generator=gen)
    v = torch.randn(num_vh, SEQ, 128, generator=gen)
    g = -torch.rand(num_vh, SEQ, generator=gen) * (
        12.0 if regime == "gscale12" else 0.5
    )
    beta_min = 0.9 if regime == "parallel" else 0.0
    beta = beta_min + torch.rand(num_vh, SEQ, generator=gen) * (1.0 - beta_min)
    if regime == "parallel":
        # Repeated text: near-parallel keys with beta close to 1.
        v = v * 10.0
        period = 20
        for t in (q, k, v):
            base = t[:, :period].clone()
            for i in range(0, SEQ, period):
                t[:, i : i + period] = base[:, : min(period, SEQ - i)]
    if regime == "zerotail":
        n = 36
        q[:, -n:] = 0
        k[:, -n:] = 0
        v[:, -n:] = 0
        g[:, -n:] = 0
        beta[:, -n:] = 0
    state0 = torch.randn(num_vh, 128, 128, generator=gen) * 0.1
    beta = beta.bfloat16().float()
    return (
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        g.reshape(num_vh, SEQ, 1),
        beta.reshape(num_vh, SEQ, 1),
        state0,
    )


def _reference(q, k, v, g, beta, state0):
    """Run the per-token recurrence in float64."""
    qe = q.double().repeat_interleave(GROUP, 0)
    ke = k.double().repeat_interleave(GROUP, 0)
    qe = torch.nn.functional.normalize(qe, dim=-1, eps=1e-6) / (128**0.5)
    ke = torch.nn.functional.normalize(ke, dim=-1, eps=1e-6)
    v = v.double()
    g = g.double()[..., 0]
    beta = beta.double()[..., 0]
    state = state0.double().clone()
    outs = []
    for t in range(SEQ):
        state = state * g[:, t].exp().reshape(-1, 1, 1)
        kv_mem = (state * ke[:, t].unsqueeze(-1)).sum(-2)
        delta = (v[:, t] - kv_mem) * beta[:, t].reshape(-1, 1)
        state = state + ke[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * qe[:, t].unsqueeze(-1)).sum(-2))
    return torch.stack(outs, 1), state


def _run_v1(q, k, v, g, beta, state0, grid):
    """Launch v1 the way the model does: repeated float32 q/k, `grid` heads per call."""
    consts = make_v2_constants()
    qe = q.float().repeat_interleave(GROUP, 0)
    ke = k.float().repeat_interleave(GROUP, 0)
    ve = v.float()
    lower_diag = torch.tril(torch.ones(128, 128))
    outs, states = [], []
    for h in range(0, ve.shape[0], grid):
        sl = slice(h, h + grid)
        o, s = simulate_nki_kernel(
            _raw(deltanet_fused_chunked_fwd_multihead),
            grid,
            dict(
                query=qe[sl].contiguous(),
                key=ke[sl].contiguous(),
                value=ve[sl].contiguous(),
                g_in=g[sl].contiguous(),
                beta_in=beta[sl].contiguous(),
                initial_state=state0[sl].contiguous(),
                lower_mask=torch.from_numpy(consts["lower_mask"]),
                identity=torch.from_numpy(consts["identity"]),
                lower_mask_diag=lower_diag,
                level_masks=torch.from_numpy(consts["level_masks"]),
            ),
        )
        outs.append(o)
        states.append(s)
    return torch.cat(outs), torch.cat(states)


def _run_v2(q, k, v, g, beta, state0, grid, repeat_qk=False, **options):
    consts = make_v2_constants()
    if repeat_qk:
        q = q.repeat_interleave(GROUP, 0)
        k = k.repeat_interleave(GROUP, 0)
    return simulate_nki_kernel(
        _raw(deltanet_prefill_v2),
        grid,
        dict(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            g_in=g.contiguous(),
            beta_in=beta.contiguous(),
            initial_state=state0.contiguous(),
            lower_mask=torch.from_numpy(consts["lower_mask"]),
            identity=torch.from_numpy(consts["identity"]),
            upper_diag_mask=torch.from_numpy(consts["upper_diag_mask"]),
            level_masks=torch.from_numpy(consts["level_masks"]),
            **options,
        ),
    )


def _max_err(x, ref):
    x = x.double()
    assert torch.isfinite(x).all(), "the kernel output holds a NaN or an Inf"
    return (x - ref).abs().max().item()


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("grid", (1, 2))
@pytest.mark.parametrize("repeat_qk", (False, True), ids=("gqa_index", "repeated_qk"))
def test_v1math_is_bit_identical_to_v1(regime, grid, repeat_qk):
    q, k, v, g, beta, state0 = _inputs(regime)
    out1, st1 = _run_v1(q, k, v, g, beta, state0, grid)
    out2, st2 = _run_v2(
        q, k, v, g, beta, state0, grid, repeat_qk=repeat_qk, formulation="v1math"
    )
    assert out2.dtype == torch.float32 and st2.dtype == torch.float32
    assert torch.equal(out2, out1)
    assert torch.equal(st2, st1)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("grid", (1, 2))
@pytest.mark.parametrize("lookahead", (0, 1))
def test_fused_error_is_within_v1_error(regime, grid, lookahead):
    q, k, v, g, beta, state0 = _inputs(regime)
    ref_out, ref_st = _reference(q, k, v, g, beta, state0)
    out1, st1 = _run_v1(q, k, v, g, beta, state0, grid)
    out2, st2 = _run_v2(
        q, k, v, g, beta, state0, grid, formulation="fused", lookahead=lookahead
    )
    e1_out, e1_st = _max_err(out1, ref_out), _max_err(st1, ref_st)
    e2_out, e2_st = _max_err(out2, ref_out), _max_err(st2, ref_st)
    # The 1e-6 term only keeps the bound above 0 when the v1 error is 0.
    assert e2_out <= 1.5 * e1_out + 1e-6, (e2_out, e1_out)
    assert e2_st <= 1.5 * e1_st + 1e-6, (e2_st, e1_st)
    assert (out2 - out1).abs().max().item() <= 2.0 * e1_out + 1e-6
    assert (st2 - st1).abs().max().item() <= 2.0 * e1_st + 1e-6


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize(
    "options",
    (dict(gemm_dtype="bf16"), dict(inverse_dtype="bf16")),
    ids=("gemm_bf16", "inverse_bf16"),
)
def test_bf16_modes_are_bounded(regime, options):
    # These bounds guard against a regression. They are not an accuracy gate:
    # the accuracy of a bf16 mode is measured on the device with MMLU.
    q, k, v, g, beta, state0 = _inputs(regime)
    ref_out, ref_st = _reference(q, k, v, g, beta, state0)
    out, st = _run_v2(q, k, v, g, beta, state0, 2, formulation="fused", **options)
    assert _max_err(out, ref_out) <= 1e-2
    assert _max_err(st, ref_st) <= 5e-2


# ── Model wrapper ─────────────────────────────────────────────────────────────


class _NoOpGroup:
    world_size = 1
    rank_in_group = 0

    @staticmethod
    def all_gather(x, dim=0):
        return x

    @staticmethod
    def reduce_scatter(x, dim=0):
        return x

    @staticmethod
    def all_reduce(x):
        return x


def _layer_config() -> Qwen3_5Config:
    # The kernel needs 128-wide linear heads. 2 k-heads and 6 v-heads give R = 3.
    return Qwen3_5Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "rope_type": "default",
        },
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=NUM_KH,
        linear_num_value_heads=NUM_KH * GROUP,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        torch_dtype=torch.float32,
    )


def _layer(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    layer = m.Qwen3_5DeltaNet(_layer_config(), layer_idx=0)
    torch.manual_seed(42)
    for name, p in layer.named_parameters():
        with torch.no_grad():
            if name in ("recurrent_state", "conv_state"):
                p.zero_()
            elif name == "a_log":
                p.uniform_(-2.0, 0.0)
            elif name == "dt_bias":
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.05)
    return layer


@pytest.mark.parametrize("kgroups_per_launch", (0, 1))
def test_wrapper_v2_v1math_equals_v1(monkeypatch, kgroups_per_launch):
    layer = _layer(monkeypatch)
    layer._dn_formulation = "v1math"
    layer._dn_kgroups_per_launch = kgroups_per_launch
    q, k, v, g, beta, state0 = _inputs("random", seed=3)
    # Model layout: q/k/v are [B, S, H, dim]; g/beta are [B, Hv, S].
    q_bshd = q.float().permute(1, 0, 2).unsqueeze(0)
    k_bshd = k.float().permute(1, 0, 2).unsqueeze(0)
    v_bshd = v.float().permute(1, 0, 2).unsqueeze(0)
    g_bhs = g[..., 0].unsqueeze(0)
    beta_bhs = beta[..., 0].unsqueeze(0)
    state = state0.unsqueeze(0)

    q_rep = q_bshd.repeat_interleave(GROUP, 2).transpose(1, 2)
    k_rep = k_bshd.repeat_interleave(GROUP, 2).transpose(1, 2)
    out1, st1 = layer._nki_chunked_forward(
        q_rep, k_rep, v_bshd.transpose(1, 2), g_bhs, beta_bhs, initial_state=state
    )
    out2, st2 = layer._nki_chunked_forward_v2(
        q_bshd, k_bshd, v_bshd, g_bhs, beta_bhs, initial_state=state
    )
    assert out2.shape == out1.shape and st2.shape == st1.shape
    assert torch.equal(out2, out1)
    assert torch.equal(st2, st1)


def test_layer_forward_v2_v1math_equals_v1(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    layer = _layer(monkeypatch)
    layer._dn_formulation = "v1math"
    T = 256
    torch.manual_seed(5)
    h = torch.randn(T, layer.hidden_size)
    pos = torch.arange(T)
    mask = torch.ones(T, dtype=torch.bfloat16)

    def run(kernel):
        with torch.no_grad():
            layer.recurrent_state.zero_()
            layer.conv_state.zero_()
        layer._dn_prefill_kernel = kernel
        out = layer(
            h,
            pos,
            is_prefill=True,
            batch_slot_indices=torch.tensor([0]),
            deltanet_padding_mask=mask,
        )
        return out.detach().clone(), layer.recurrent_state.detach().clone()

    out1, st1 = run("v1")
    out2, st2 = run("v2")
    assert torch.equal(out2, out1)
    assert torch.equal(st2, st1)


@pytest.mark.parametrize(
    "settings",
    (
        ("v3", "v1math", "fp32", "fp32", 0, 0),
        ("v2", "other", "fp32", "fp32", 0, 0),
        ("v2", "fused", "fp16", "fp32", 0, 0),
        ("v2", "fused", "fp32", "tf32", 0, 0),
        ("v2", "fused", "fp32", "fp32", 2, 0),
        ("v2", "fused", "fp32", "fp32", 0, -1),
        ("v2", "v1math", "bf16", "fp32", 0, 0),
    ),
)
def test_settings_check_rejects_invalid_values(settings):
    with pytest.raises(ValueError):
        m._check_deltanet_prefill_settings(*settings)


def test_settings_check_accepts_valid_values():
    m._check_deltanet_prefill_settings("v1", "v1math", "bf16", "fp32", 0, 0)
    m._check_deltanet_prefill_settings("v2", "fused", "bf16", "bf16", 1, 2)


# ── Decode step: nkilib gdn_tkg ───────────────────────────────────────────────


def _rel_rms(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm()).item()


def _decode_reference(layer, q, k, v, a, b, z, state):
    """float64 decode step of the model math, on the same inputs as the kernel."""
    import torch.nn.functional as F

    B = q.shape[0]
    hv, dim = layer.num_value_heads_per_rank, layer.value_dim
    rep = hv // layer.num_key_heads_per_rank
    # The model repeats each k-head for its R v-heads: v-head h uses k-head h // R.
    qd = q.double().repeat_interleave(rep, 2).transpose(1, 2)
    kd = k.double().repeat_interleave(rep, 2).transpose(1, 2)
    vd = v.double().transpose(1, 2)
    beta = b.double().sigmoid().reshape(B, hv, 1)
    g = -layer.a_log.double().exp() * F.softplus(a.double() + layer.dt_bias.double())
    out, new_state = layer._recurrent_step(
        qd, kd, vd, g.reshape(B, hv, 1), beta, state.double()
    )
    out = out.reshape(B, hv, dim)
    out = (
        out
        * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)
        * layer.group_norm.weight.double()
    )
    out = out * F.silu(z.double().reshape(B, hv, dim))
    return out.reshape(B, hv * dim), new_state


@pytest.mark.parametrize("batch", (1, 2))
def test_tkg_decode_step_matches_reference(monkeypatch, batch):
    layer = _layer(monkeypatch)
    hk, hv, dim = NUM_KH, NUM_KH * GROUP, 128
    torch.manual_seed(11)
    bf = torch.bfloat16
    q = torch.randn(batch, 1, hk, dim).to(bf)
    k = torch.randn(batch, 1, hk, dim).to(bf)
    v = torch.randn(batch, 1, hv, dim).to(bf)
    a = torch.randn(batch, hv).to(bf)
    b = torch.randn(batch, hv).to(bf)
    z = torch.randn(batch, hv * dim).to(bf)
    state = torch.randn(batch, hv, dim, dim) * 0.5
    out, new_state = layer._nki_tkg_decode_step(q, k, v, a, b, z, state)
    ref_out, ref_state = _decode_reference(layer, q, k, v, a, b, z, state)
    assert out.shape == (batch, hv * dim) and new_state.shape == (batch, hv, dim, dim)
    # The kernel uses bf16 matmul operands, so the error is at the bf16 level.
    assert _rel_rms(out.float(), ref_out) < 1e-2
    assert _rel_rms(new_state, ref_state) < 1e-2


def test_layer_decode_tkg_matches_v1(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    config = _layer_config()
    config.torch_dtype = torch.bfloat16
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
    state0 = layer.recurrent_state.detach().clone()
    conv0 = layer.conv_state.detach().clone()
    h = torch.randn(1, layer.hidden_size).to(torch.bfloat16)

    def run(kernel):
        with torch.no_grad():
            layer.recurrent_state.copy_(state0)
            layer.conv_state.copy_(conv0)
        layer._dn_decode_kernel = kernel
        out = layer(
            h, torch.tensor([7]), is_prefill=False, batch_slot_indices=torch.tensor([0])
        )
        return out.detach().float(), layer.recurrent_state.detach().float()

    out1, st1 = run("v1")
    out2, st2 = run("tkg")
    assert torch.isfinite(out2).all() and torch.isfinite(st2).all()
    assert _rel_rms(out2, out1) < 3e-2
    assert _rel_rms(st2, st1) < 3e-2


def test_decode_kernel_setting_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL", "v3")
    with pytest.raises(ValueError):
        m.Qwen3_5DeltaNet(_layer_config(), layer_idx=0)


def test_tkg_decode_kernel_rejects_other_norm_eps(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL", "tkg")
    config = _layer_config()
    config.rms_norm_eps = 1e-5
    with pytest.raises(ValueError):
        m.Qwen3_5DeltaNet(config, layer_idx=0)


# ── Decode step: batched kernel (nki/deltanet/decode_batched.py) ──────────────


def _batched_inputs(rows, state_dtype, gscale=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    dim = 128
    return dict(
        query=torch.randn(rows, dim, generator=gen).to(bf),
        key=torch.randn(rows, dim, generator=gen).to(bf),
        value=torch.randn(rows, dim, generator=gen).to(bf),
        z=torch.randn(rows, dim, generator=gen).to(bf),
        a=torch.randn(rows, 1, generator=gen) * gscale,
        b=torch.randn(rows, 1, generator=gen),
        a_log=torch.rand(rows, 1, generator=gen) * 2 - 2,
        dt_bias=torch.rand(rows, 1, generator=gen) * 2 - 1,
        norm_weight=(1 + 0.1 * torch.randn(1, dim, generator=gen))
        .expand(rows, dim)
        .contiguous(),
        state=(torch.randn(rows, dim, dim, generator=gen) * 0.5).to(state_dtype),
        identity=torch.eye(128),
    )


def _batched_reference(x, eps):
    """float64 decode step of the model math for the inputs of _batched_inputs."""
    import torch.nn.functional as F

    d = {k: v.double() for k, v in x.items()}
    qn = F.normalize(d["query"], p=2, dim=-1, eps=1e-6) * 128**-0.5
    kn = F.normalize(d["key"], p=2, dim=-1, eps=1e-6)
    g = -d["a_log"].exp() * F.softplus(d["a"] + d["dt_bias"])
    s = d["state"] * g.exp().unsqueeze(-1)
    delta = (d["value"] - (s * kn.unsqueeze(-1)).sum(-2)) * d["b"].sigmoid()
    s = s + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (s * qn.unsqueeze(-1)).sum(-2)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * d["norm_weight"]
    return o * F.silu(d["z"]), s


@pytest.mark.parametrize(
    "rows,grid,state_dtype,gscale",
    (
        (6, 1, torch.bfloat16, 1.0),
        (6, 2, torch.bfloat16, 1.0),
        (36, 2, torch.bfloat16, 12.0),
        (72, 1, torch.bfloat16, 1.0),
        (12, 2, torch.float32, 1.0),
    ),
)
def test_batched_decode_kernel_matches_reference(rows, grid, state_dtype, gscale):
    from vllm_neuron.nki.deltanet.decode_batched import deltanet_decode_batched

    x = _batched_inputs(rows, state_dtype, gscale)
    out, state = simulate_nki_kernel(
        _raw(deltanet_decode_batched), grid, dict(x, norm_eps=1e-6)
    )
    ref_out, ref_state = _batched_reference(x, 1e-6)
    assert out.dtype == torch.bfloat16 and state.dtype == state_dtype
    # The kernel uses float32 math. Its error is the rounding of the result to the output dtype.
    assert (
        _rel_rms(out.float(), ref_out)
        <= 1.05 * _rel_rms(ref_out.to(torch.bfloat16).float(), ref_out) + 1e-6
    )
    state_round = _rel_rms(ref_state.to(state_dtype).double(), ref_state)
    assert _rel_rms(state.double(), ref_state) <= 1.05 * state_round + 1e-6


@pytest.mark.parametrize("batch", (1, 2))
def test_batched_decode_step_matches_reference(monkeypatch, batch):
    layer = _layer(monkeypatch)
    hk, hv, dim = NUM_KH, NUM_KH * GROUP, 128
    torch.manual_seed(11)
    bf = torch.bfloat16
    q = torch.randn(batch, 1, hk, dim).to(bf)
    k = torch.randn(batch, 1, hk, dim).to(bf)
    v = torch.randn(batch, 1, hv, dim).to(bf)
    a = torch.randn(batch, hv).to(bf)
    b = torch.randn(batch, hv).to(bf)
    z = torch.randn(batch, hv * dim).to(bf)
    state = (torch.randn(batch, hv, dim, dim) * 0.5).to(bf)
    grid = layer._batched_decode_grid(q)
    assert grid == 2
    out, new_state = layer._nki_batched_decode_step(q, k, v, a, b, z, state, grid)
    ref_out, ref_state = _decode_reference(layer, q, k, v, a, b, z, state)
    assert out.shape == (batch, hv * dim) and new_state.shape == (batch, hv, dim, dim)
    assert new_state.dtype == bf
    assert _rel_rms(out.float(), ref_out) < 3e-3
    assert _rel_rms(new_state, ref_state) < 3e-3


def test_layer_decode_batched_matches_v1(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    config = _layer_config()
    config.torch_dtype = torch.bfloat16
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
    state0 = layer.recurrent_state.detach().clone()
    conv0 = layer.conv_state.detach().clone()
    h = torch.randn(1, layer.hidden_size).to(torch.bfloat16)

    def run(kernel):
        with torch.no_grad():
            layer.recurrent_state.copy_(state0)
            layer.conv_state.copy_(conv0)
        layer._dn_decode_kernel = kernel
        out = layer(
            h, torch.tensor([7]), is_prefill=False, batch_slot_indices=torch.tensor([0])
        )
        return out.detach().float(), layer.recurrent_state.detach().float()

    out1, st1 = run("v1")
    out2, st2 = run("batched")
    assert torch.isfinite(out2).all() and torch.isfinite(st2).all()
    assert not torch.equal(st2, state0.float())
    assert _rel_rms(out2, out1) < 1e-2
    assert _rel_rms(st2, st1) < 1e-2


def test_batched_decode_grid_limits(monkeypatch):
    layer = _layer(monkeypatch)
    hv = NUM_KH * GROUP

    def grid_for(batch):
        return layer._batched_decode_grid(torch.zeros(batch, 1, NUM_KH, 128))

    # 2 programs own at most 2 x 128 rows. 1 program owns at most 128 rows.
    assert grid_for(256 // hv) == 2
    assert grid_for(256 // hv + 1) == 0
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    assert grid_for(1) == 0


def test_batched_decode_groups_split_a_large_batch(monkeypatch):
    layer = _layer(monkeypatch)
    hv = NUM_KH * GROUP
    max_batch = 256 // hv

    def groups_for(batch):
        return layer._batched_decode_groups(torch.zeros(batch, 1, NUM_KH, 128))

    # A batch that one call takes keeps one call, as before.
    assert groups_for(max_batch) == (2, (max_batch,))
    # A larger batch takes more calls. Their batch sizes differ by at most 1.
    assert groups_for(max_batch + 1) == (2, (22, 21))
    assert groups_for(2 * max_batch + 1) == (2, (29, 28, 28))
    for batch in (max_batch + 1, 2 * max_batch + 1, 5 * max_batch):
        grid, sizes = groups_for(batch)
        assert sum(sizes) == batch and max(sizes) * hv <= grid * 128
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    assert groups_for(max_batch + 1) == (0, ())


def test_layer_decode_grouped_calls_match_v1(monkeypatch):
    """A batch above the row limit of one call takes several calls and gives the v1 result."""
    from vllm_neuron.model.neuron_config import NeuronConfig

    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    hv = NUM_KH * GROUP
    # One program takes the rows of one batch row, so one call takes 2 batch rows.
    monkeypatch.setattr(m, "_BATCHED_DECODE_MAX_ROWS", hv)
    config = _layer_config()
    config.torch_dtype = torch.bfloat16
    config.neuron_config = NeuronConfig(num_seqs_buckets=[8])
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
    state0 = layer.recurrent_state.detach().clone()
    conv0 = layer.conv_state.detach().clone()
    batch = 5
    slots = torch.tensor([4, 0, 6, 2, 5])
    h = torch.randn(batch, layer.hidden_size).to(torch.bfloat16)
    calls = []
    step = layer._nki_batched_decode_step

    def counted_step(*args):
        calls.append(args[0].shape[0])
        return step(*args)

    monkeypatch.setattr(layer, "_nki_batched_decode_step", counted_step)

    def run(kernel):
        with torch.no_grad():
            layer.recurrent_state.copy_(state0)
            layer.conv_state.copy_(conv0)
        layer._dn_decode_kernel = kernel
        out = layer(
            h, torch.full((batch,), 7), is_prefill=False, batch_slot_indices=slots
        )
        return out.detach().float(), layer.recurrent_state.detach().float()

    out1, st1 = run("v1")
    out2, st2 = run("batched")
    assert calls == [2, 2, 1]
    assert torch.isfinite(out2).all() and torch.isfinite(st2).all()
    assert _rel_rms(out2, out1) < 1e-2
    assert _rel_rms(st2[slots], st1[slots]) < 1e-2
    # The rows of other requests do not change.
    others = torch.tensor([1, 3, 7])
    assert torch.equal(st2[others], state0.float()[others])
    assert not torch.equal(st2[slots], state0.float()[slots])


def test_batched_decode_kernel_accepts_other_norm_eps(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL", "batched")
    config = _layer_config()
    config.rms_norm_eps = 1e-5
    layer = m.Qwen3_5DeltaNet(config, layer_idx=0)
    assert layer._dn_decode_kernel == "batched"
