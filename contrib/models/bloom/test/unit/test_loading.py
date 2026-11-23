# SPDX-License-Identifier: Apache-2.0
"""Loading tests for Bloom (bigscience/bloomz-560m) in CPU mode.

The tests check:
  1. The registration contract: the checkpoint architecture, the factory that
     ``src/`` exports, the vLLM model protocol of the factory, and the
     ``vllm.general_plugins`` entry point in ``src/plugin.py``.
  2. The config parse and the validation errors.
  3. The checkpoint key coverage of the weight mappings.
  4. The QKV, O and bias loader transforms against an independent build.
  5. The full model load at TP=2 against the checkpoint values.

Run with:
    VLLM_NEURON_CPU_MODE=1 BLOOM_CHECKPOINT_PATH=/path/to/bloomz-560m \
    pytest contrib/models/bloom/test/unit/ -v
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

import bloom_test_utils as U
from _local_src import SRC_PACKAGE_NAME, src_module, src_package

TP_SIZE = 2


@pytest.fixture(scope="module")
def checkpoint():
    from vllm_neuron import envs

    assert envs.is_native_backend(), (
        "The Bloom tests need the native route. Unset VLLM_NEURON_BACKEND or set "
        "it to neuron_native."
    )
    return U.get_checkpoint()


@pytest.fixture(scope="module")
def hf_config_dict(checkpoint):
    return U.load_hf_config_dict(checkpoint)


@pytest.fixture(scope="module")
def checkpoint_keys(checkpoint):
    keys = U.checkpoint_tensor_names(checkpoint)
    assert keys, f"The checkpoint {checkpoint} has no safetensors tensors."
    return keys


# =============================================================================
# 1. Registration contract
# =============================================================================


def test_registration_contract(hf_config_dict, monkeypatch):
    """The architecture string, the exported factory and the plugin entry point.

    A contrib model is not in ``vllm_neuron.model.registry.get_models()``, so
    this checks what a user registers instead: the checkpoint names the
    architecture, ``src/`` exports the factory under that name, the factory
    satisfies vLLM's text-generation model protocol, and the entry point in
    ``src/plugin.py`` registers the installed package's factory under the same
    name. vLLM ships its own GPU ``BloomForCausalLM``, so the test also checks
    that a registration of the factory replaces vLLM's class.
    """
    import vllm
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces_base import (
        is_pooling_model,
        is_text_generation_model,
        is_vllm_model,
    )

    assert U.ARCHITECTURE in hf_config_dict["architectures"]
    factory = getattr(src_package(), U.ARCHITECTURE)
    assert factory is src_module("factory").BloomForCausalLM
    assert factory.__module__ == f"{SRC_PACKAGE_NAME}.factory"

    assert is_vllm_model(factory)
    assert is_text_generation_model(factory)
    assert not is_pooling_model(factory)

    calls = []
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, model_cls: calls.append((arch, model_cls)),
    )
    src_module("plugin").register()
    assert calls == [(U.ARCHITECTURE, "bloom_neuron.factory:BloomForCausalLM")]
    monkeypatch.undo()

    ModelRegistry.register_model(U.ARCHITECTURE, factory)
    assert ModelRegistry._try_load_model_cls(U.ARCHITECTURE) is factory


# =============================================================================
# 2. Config and validation
# =============================================================================


def test_config_and_validation(hf_config_dict):
    from transformers import BloomConfig as HFBloomConfig
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    config_mod = src_module("config")
    factory_mod = src_module("factory")
    model_mod = src_module("model")
    BloomConfig = config_mod.BloomConfig
    BloomForCausalLM = factory_mod.BloomForCausalLM

    config = BloomConfig.from_configs(hf_config_dict)
    assert (
        config.hidden_size == U.hf_config_value(hf_config_dict, "hidden_size") == 1024
    )
    assert (
        config.num_hidden_layers
        == U.hf_config_value(hf_config_dict, "num_hidden_layers")
        == 24
    )
    assert (
        config.num_attention_heads
        == U.hf_config_value(hf_config_dict, "num_attention_heads")
        == 16
    )
    assert config.vocab_size == hf_config_dict["vocab_size"] == 250880
    assert config.head_dim == 64
    assert config.intermediate_size == 4096
    assert config.layer_norm_epsilon == hf_config_dict["layer_norm_epsilon"]
    assert config.tie_word_embeddings is True
    assert config.torch_dtype == config_mod.BLOOM_COMPUTE_DTYPE == torch.bfloat16

    # The checkpoint config passes the factory validation.
    hf_config = HFBloomConfig.from_dict(hf_config_dict)
    BloomForCausalLM._validate_config(hf_config, None)

    bad = HFBloomConfig.from_dict(
        {**hf_config_dict, "apply_residual_connection_post_layernorm": True}
    )
    with pytest.raises(ValueError, match="apply_residual_connection_post_layernorm"):
        BloomForCausalLM._validate_config(bad, None)
    # config.json sets num_attention_heads. HF maps that key onto n_head after
    # __init__, so remove each head alias before the override.
    wide_dict = {
        k: v
        for k, v in hf_config_dict.items()
        if k not in ("n_head", "num_attention_heads")
    }
    wide = HFBloomConfig.from_dict({**wide_dict, "n_embed": 256 * 4, "n_head": 4})
    assert wide.num_attention_heads == 4 and wide.hidden_size == 1024
    with pytest.raises(ValueError, match="head_dim"):
        BloomForCausalLM._validate_config(wide, None)
    # The head_dim bound leaves room for every ALiBi lane in the padded head.
    assert factory_mod.BLOOM_MAX_HEAD_DIM == (
        model_mod.BLOOM_PADDED_HEAD_DIM - model_mod.BLOOM_ALIBI_LANES
    )

    max_exact = factory_mod.BLOOM_MAX_EXACT_POSITION
    factory_mod.validate_max_model_len(2048)
    factory_mod.validate_max_model_len(max_exact)
    with pytest.raises(ValueError, match="max_model_len"):
        factory_mod.validate_max_model_len(max_exact + 1)

    for tp in (1, 2, 4, 8):
        model_mod._validate_tp(config, tp)
    with pytest.raises(ValueError, match="even"):
        model_mod._validate_tp(config, 16)
    with pytest.raises(ValueError, match="not divisible"):
        model_mod._validate_tp(config, 32)

    # The slopes are the same as the HF ALiBi slopes. HF gives slope * pos.
    hf_alibi = build_alibi_tensor(
        torch.ones(1, 2, dtype=torch.long), config.num_attention_heads, torch.float32
    )  # [heads, 1, 2], value at pos 1 is the slope
    slopes = model_mod.build_alibi_slopes(config.num_attention_heads)
    torch.testing.assert_close(slopes, hf_alibi[:, 0, 1], rtol=0, atol=0)

    terms = model_mod.split_slope_terms(slopes / config.head_dim**-0.5)
    assert torch.equal(terms, terms.to(torch.bfloat16).to(torch.float32))
    torch.testing.assert_close(
        terms.sum(-1), slopes / config.head_dim**-0.5, rtol=3e-7, atol=0
    )


def test_gelu_tanh_matches_torch():
    """The traceable GeLU of the torch MLP path matches the torch function."""
    gelu_tanh = src_module("model").gelu_tanh
    torch.manual_seed(U.RANDOM_SEED)
    x = torch.randn(4096, dtype=torch.float32) * 4
    ref = torch.nn.functional.gelu(x, approximate="tanh")
    torch.testing.assert_close(gelu_tanh(x), ref, rtol=1e-6, atol=1e-6)
    xb = x.to(torch.bfloat16)
    got = gelu_tanh(xb)
    assert got.dtype == torch.bfloat16
    # Both compute in float32 and round once, so they differ by at most 1 ULP.
    torch.testing.assert_close(
        got, torch.nn.functional.gelu(xb, approximate="tanh"), rtol=8e-3, atol=1e-5
    )


# =============================================================================
# 3. Checkpoint key coverage
# =============================================================================


def test_checkpoint_mapping_coverage(hf_config_dict, checkpoint_keys):
    config = src_module("config").BloomConfig.from_configs(hf_config_dict)
    BloomForCausalLM = src_module("model").BloomForCausalLM
    prefix = BloomForCausalLM.detect_checkpoint_prefix(checkpoint_keys)
    # checkpoint_mappings reads only the config and the number of layers.
    stub = SimpleNamespace(
        config=config,
        model=SimpleNamespace(layers=range(config.num_hidden_layers)),
    )
    mappings = BloomForCausalLM.checkpoint_mappings(stub, prefix)

    mapped = set()
    for value in mappings.values():
        mapped.update(value if isinstance(value, list) else [value])

    missing = sorted(mapped - checkpoint_keys)
    assert not missing, f"Mapped keys are not in the checkpoint: {missing}"
    unused = sorted(checkpoint_keys - mapped)
    assert not unused, f"Checkpoint keys have no mapping: {unused}"

    # The checkpoint has 5 global tensors and 12 tensors for each layer.
    assert len(checkpoint_keys) == 5 + 12 * config.num_hidden_layers
    # The model has 5 global parameters, 11 for each layer and the tied head.
    # One QKV parameter reads 2 tensors (query_key_value weight and bias).
    assert len(mappings) == 5 + 11 * config.num_hidden_layers + 1
    qkv_entries = [v for v in mappings.values() if isinstance(v, list)]
    assert len(qkv_entries) == config.num_hidden_layers
    assert all(len(v) == 2 for v in qkv_entries)
    assert mappings["lm_head.weight"] == f"{prefix}word_embeddings.weight"


# =============================================================================
# 4. Loader transforms against an independent build
# =============================================================================


def _expected_qkv(weight, bias, rank, num_heads, head_dim, hidden, extra):
    """Build the main block of the QKV parameter with explicit indices."""
    heads_per_rank = num_heads // TP_SIZE
    padded = 128
    out = torch.zeros(hidden + extra, 3 * heads_per_rank * padded)
    for t in range(3):
        for gl in range(heads_per_rank):
            g = rank * heads_per_rank + gl
            rows = slice((g * 3 + t) * head_dim, (g * 3 + t + 1) * head_dim)
            col = t * heads_per_rank * padded + gl * padded
            out[:hidden, col : col + head_dim] = weight[rows, :].T
            out[hidden, col : col + head_dim] = bias[rows]
    return out


def _layer0_tensors(checkpoint):
    prefix = U.checkpoint_prefix(checkpoint)
    keys = {
        "qkv_w": U.layer_key(prefix, "self_attention.query_key_value.weight"),
        "qkv_b": U.layer_key(prefix, "self_attention.query_key_value.bias"),
        "o_w": U.layer_key(prefix, "self_attention.dense.weight"),
        "up_b": U.layer_key(prefix, "mlp.dense_h_to_4h.bias"),
    }
    raw = U.read_checkpoint_tensors(checkpoint, list(keys.values()))
    return {name: raw[key] for name, key in keys.items()}


@pytest.mark.parametrize("rank", [0, 1])
def test_loader_transforms(checkpoint, hf_config_dict, rank):
    model_mod = src_module("model")
    config = src_module("config").BloomConfig.from_configs(hf_config_dict)
    H = config.hidden_size
    hd = config.head_dim
    nh = config.num_attention_heads
    g_rank = nh // TP_SIZE
    scale = hd**-0.5
    # The decode kernel at LNC 2 needs (input width / 128) to be even.
    extra = model_mod.bloom_extra_columns(H)
    assert extra >= 3 and (H + extra) % 256 == 0
    t = _layer0_tensors(checkpoint)

    # -- QKV --
    qkv = model_mod.bloom_qkv_weight_loader(nh, hd, H, TP_SIZE, scale).load(
        [t["qkv_w"], t["qkv_b"]], rank
    )
    assert qkv.shape == (H + extra, 3 * g_rank * 128)
    assert qkv.dtype == torch.bfloat16
    expected = _expected_qkv(
        t["qkv_w"].float(), t["qkv_b"].float(), rank, nh, hd, H, extra
    ).to(torch.bfloat16)
    view = qkv.float().view(H + extra, 3, g_rank, 128)
    exp_view = expected.float().view(H + extra, 3, g_rank, 128)
    # The real head lanes match the checkpoint exactly.
    assert torch.equal(view[..., :hd], exp_view[..., :hd])
    # Rows below H + 3 are the only rows that can hold a non-zero value.
    assert torch.count_nonzero(view[H + 3 :]) == 0
    # V has no extra lanes. Q and K use only lanes hd .. hd + 5.
    assert torch.count_nonzero(view[:, 2, :, hd:]) == 0
    assert torch.count_nonzero(view[:H, :2, :, hd:]) == 0
    assert torch.count_nonzero(view[:, :2, :, hd + 6 :]) == 0
    # K extras: 1 at (H + 1, lanes hd, hd + 2, hd + 4), (H + 2, odd lanes).
    k_extra = view[H : H + 3, 1, :, hd : hd + 6]
    expected_k = torch.zeros(3, 6)
    expected_k[1, 0::2] = 1.0
    expected_k[2, 1::2] = 1.0
    assert torch.equal(k_extra, expected_k.unsqueeze(1).expand(3, g_rank, 6))
    assert torch.count_nonzero(view[H + 1 :, 0]) == 0

    # The ALiBi identity: scale * q_extra . k_extra(p) = slope * p.
    slopes = model_mod.build_alibi_slopes(nh)[rank * g_rank : (rank + 1) * g_rank]
    q_extra = view[H, 0, :, hd : hd + 6]  # [g, 6]
    for p in (0, 1, 63, 64, 1023, 1024, 1535, 2047, 16383):
        x_extra = torch.tensor([1.0, float(p // 64), float(p % 64)])
        k_vec = (x_extra.view(3, 1, 1) * view[H : H + 3, 1, :, hd : hd + 6]).sum(0)
        score = scale * (q_extra * k_vec).sum(-1)
        torch.testing.assert_close(score, slopes * p, rtol=1e-6, atol=1e-4)

    # -- O --
    o_proj = model_mod.bloom_o_proj_weight_loader(nh, hd, H, TP_SIZE).load(
        [t["o_w"]], rank
    )
    assert o_proj.shape == (g_rank * 128, H)
    o_view = o_proj.float().view(g_rank, 128, H)
    o_w = t["o_w"].float()
    for gl in range(g_rank):
        g = rank * g_rank + gl
        exp = o_w[:, g * hd : (g + 1) * hd].T.to(torch.bfloat16).float()
        assert torch.equal(o_view[gl, :hd], exp)
    assert torch.count_nonzero(o_view[:, hd:]) == 0

    # -- Sharded up bias --
    shard = config.intermediate_size // TP_SIZE
    up_b = model_mod.bloom_row_bias_loader(shard, TP_SIZE).load([t["up_b"]], rank)
    assert up_b.shape == (1, shard)
    assert torch.equal(
        up_b[0].float(),
        t["up_b"][rank * shard : (rank + 1) * shard].to(torch.bfloat16).float(),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_loaders_built_on_meta(checkpoint, hf_config_dict, rank):
    """Build the loaders in a meta context, as the model runner does.

    The runner calls from_configs in ``torch.device("meta")`` and loads the
    weights later on CPU. The result must be equal to the CPU-built result.
    """
    model_mod = src_module("model")
    config = src_module("config").BloomConfig.from_configs(hf_config_dict)
    H = config.hidden_size
    hd = config.head_dim
    nh = config.num_attention_heads
    scale = hd**-0.5
    shard = config.intermediate_size // TP_SIZE
    t = _layer0_tensors(checkpoint)

    def build():
        return (
            model_mod.bloom_qkv_weight_loader(nh, hd, H, TP_SIZE, scale),
            model_mod.bloom_o_proj_weight_loader(nh, hd, H, TP_SIZE),
            model_mod.bloom_row_bias_loader(shard, TP_SIZE),
        )

    with torch.device("meta"):
        meta_loaders = build()
    cpu_loaders = build()
    inputs = ([t["qkv_w"], t["qkv_b"]], [t["o_w"]], [t["up_b"]])
    for meta_loader, cpu_loader, slices in zip(meta_loaders, cpu_loaders, inputs):
        got = meta_loader.load(slices, rank)
        want = cpu_loader.load(slices, rank)
        assert not got.is_meta
        assert got.device == want.device
        assert torch.equal(got, want)


# =============================================================================
# 5. Full model load at TP=2 (MPExecutor, CPU)
# =============================================================================


class LoadReport(nn.Module):
    """Return load statistics and parameter samples of one rank."""

    def __init__(self, model: nn.Module, mapping_ok: bool):
        super().__init__()
        self.model = model
        self.mapping_ok = mapping_ok

    def forward(self, _unused):
        params = dict(self.model.named_parameters())
        n_meta = sum(int(p.is_meta) for p in params.values())
        n_non_bf16 = sum(int(p.dtype != torch.bfloat16) for p in params.values())
        n_non_finite = sum(
            int(not bool(torch.isfinite(p).all()))
            for p in params.values()
            if not p.is_meta
        )
        stats = torch.tensor(
            [len(params), n_meta, n_non_bf16, n_non_finite, int(self.mapping_ok)],
            dtype=torch.int64,
        )
        m = self.model.model
        return (
            stats,
            self.model.lm_head.weight[:4, :8].contiguous(),
            m.word_embeddings.weight[:4, :8].contiguous(),
            m.layers[0].self_attn.qkv_proj_weight[:4, :8].contiguous(),
            m.layers[0].self_attn.o_proj_weight[:8, :4].contiguous(),
            m.layers[0].mlp.up_proj_weight[:4, :8].contiguous(),
            m.layers[0].mlp.down_proj_weight[:4, :8].contiguous(),
            m.layers[23].input_layernorm.bias[:8].contiguous(),
            m.ln_f.weight[:8].contiguous(),
        )


def load_full_model(checkpoint, hf_config_dict, device):
    """model_load for MPExecutor: the full Bloom model with real weights."""
    from transformers import BloomConfig as HFBloomConfig

    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    assert dist.get_world_size() == TP_SIZE
    hf_config = HFBloomConfig.from_dict(hf_config_dict)
    factory = src_module("factory").BloomForCausalLM
    model = factory.from_configs(hf_config, NeuronConfig())

    names = SafetensorsCheckpoint(checkpoint, None).get_tensor_names()
    prefix = model.detect_checkpoint_prefix(names)
    mapping_ok = set(model.checkpoint_mappings(prefix)) == {
        n for n, _ in model.named_parameters()
    }

    # Fill each parameter with NaN. A parameter that the load skips then
    # fails the finite check.
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))
    model.load_weights(checkpoint, device=device, cache_dir=None)
    return LoadReport(model, mapping_ok)


@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_full_model_load(checkpoint, hf_config_dict, world_size):
    from vllm_neuron import envs
    from vllm_neuron.utils.executor import MPExecutor

    assert envs.VLLM_NEURON_CPU_MODE, "Run the load test with VLLM_NEURON_CPU_MODE=1."
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(load_full_model, checkpoint, hf_config_dict, "cpu"),
        device="cpu",
    )
    executor.dispatch(torch.zeros(1))
    outputs = executor.collect(timeout=U.COLLECT_TIMEOUT_S)
    assert len(outputs) == world_size

    prefix = U.checkpoint_prefix(checkpoint)
    k = {
        "embed": f"{prefix}word_embeddings.weight",
        "qkv": U.layer_key(prefix, "self_attention.query_key_value.weight"),
        "o": U.layer_key(prefix, "self_attention.dense.weight"),
        "up": U.layer_key(prefix, "mlp.dense_h_to_4h.weight"),
        "down": U.layer_key(prefix, "mlp.dense_4h_to_h.weight"),
        "ln23_b": U.layer_key(prefix, "input_layernorm.bias", 23),
        "ln_f": f"{prefix}ln_f.weight",
    }
    raw = U.read_checkpoint_tensors(checkpoint, list(k.values()))

    def bf(x):
        return x.to(torch.bfloat16).float()

    vocab_shard = hf_config_dict["vocab_size"] // world_size
    hidden_size = U.hf_config_value(hf_config_dict, "hidden_size")
    num_heads = U.hf_config_value(hf_config_dict, "num_attention_heads")
    num_layers = U.hf_config_value(hf_config_dict, "num_hidden_layers")
    inter_shard = 4 * hidden_size // world_size
    heads_per_rank = num_heads // world_size
    hd = hidden_size // num_heads

    for rank, out in enumerate(outputs):
        stats, lm_head, embed, qkv, o_proj, up, down, ln23_b, ln_f = out
        n_params, n_meta, n_non_bf16, n_non_finite, mapping_ok = stats.tolist()
        assert mapping_ok == 1, f"rank {rank}: mappings do not cover the parameters"
        # 5 global parameters, 11 for each layer and the head.
        assert n_params == 5 + 11 * num_layers + 1, (
            f"rank {rank}: {n_params} parameters"
        )
        assert n_meta == 0, f"rank {rank}: {n_meta} parameters stay on meta"
        assert n_non_bf16 == 0, f"rank {rank}: {n_non_bf16} parameters are not bf16"
        assert n_non_finite == 0, f"rank {rank}: {n_non_finite} non-finite parameters"

        v0 = rank * vocab_shard
        exp_embed = bf(raw[k["embed"]][v0 : v0 + 4, :8])
        assert torch.equal(embed.float(), exp_embed), f"rank {rank}: embedding"
        assert torch.equal(lm_head.float(), exp_embed), f"rank {rank}: lm_head"

        # First Q head of the rank: rows (g * 3 + 0) * hd + d of the fused weight.
        g = rank * heads_per_rank
        exp_qkv = bf(raw[k["qkv"]][g * 3 * hd : g * 3 * hd + 8, :4].T)
        assert torch.equal(qkv.float(), exp_qkv), f"rank {rank}: qkv"
        exp_o = bf(raw[k["o"]][:4, g * hd : g * hd + 8].T)
        assert torch.equal(o_proj.float(), exp_o), f"rank {rank}: o_proj"

        i0 = rank * inter_shard
        exp_up = bf(raw[k["up"]][i0 : i0 + 8, :4].T)
        assert torch.equal(up.float(), exp_up), f"rank {rank}: up_proj"
        exp_down = bf(raw[k["down"]][:8, i0 : i0 + 4].T)
        assert torch.equal(down.float(), exp_down), f"rank {rank}: down_proj"

        assert torch.equal(ln23_b.float(), bf(raw[k["ln23_b"]][:8]))
        assert torch.equal(ln_f.float(), bf(raw[k["ln_f"]][:8]))
