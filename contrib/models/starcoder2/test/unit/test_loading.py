# SPDX-License-Identifier: Apache-2.0
"""StarCoder2 loading tests (CPU).

These tests check the loading path of bigcode/starcoder2-3b at the recorded
topology (TP 4):

1. ``test_registration_contract``: the checkpoint's architecture string, the
   factory that ``src/`` exports, the vLLM model protocol of the factory, and
   the ``vllm.general_plugins`` entry point in ``src/plugin.py``.
2. ``test_config_parse``: the config parse from the checkpoint's
   ``config.json``.
3. ``test_factory_validation``: the factory accepts the checkpoint config and
   refuses the configurations that the implementation does not support.
4. ``test_tp_size_checks``: the attention, the MLP and the LM head accept or
   refuse a TP size.
5. ``test_checkpoint_mapping_coverage``: each model parameter has a mapping, and
   the mappings use each checkpoint key exactly one time.
6. ``test_per_rank_shapes``: on each of the 4 ranks, each loader gives a tensor
   of the parameter's shape. The test reads the safetensors header only.
7. ``test_weight_loader_transforms``: the TP 4 loaders give each rank the
   correct part of each tensor. The test uses mock slices.
8. ``test_mlp_fallback_without_torch_gelu``: the torch MLP fallback computes the
   GELU tanh approximation with elementary ops, not with ``F.gelu``.
   ``test_mlp_kernel_routing``: the MLP calls ``NF.mlp`` only for a shape that
   its NKI kernel runs, so ``NF.mlp`` never reaches its own ``F.gelu`` fallback.
9. ``test_full_checkpoint_load``: a 4-rank CPU load of the full model. It needs
   the 12 GB checkpoint in ``STARCODER2_CHECKPOINT_PATH`` and skips without it.

Tests 1 to 6 read ``config.json`` and the safetensors header of the pinned
revision (from ``STARCODER2_CHECKPOINT_PATH``, or from the HuggingFace Hub).
No test needs Neuron hardware. Run with:

    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/starcoder2/test/unit/test_loading.py -v
"""

import copy
from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    get_weight_loader,
    scaled_bias_loader,
    sharding_weight_loader,
)

from starcoder2_test_utils import (
    ARCHITECTURE,
    TP_SIZE,
    checkpoint_tensor_info,
    hf_config,
    patch_tp_group,
    require_checkpoint,
    src_module,
    src_package,
)

HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 12288
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_LAYERS = 30
# Each layer has 12 parameters: 4 attention, 4 LayerNorm and 4 MLP. The
# embedding, the final norm (2) and the LM head add 4 parameters.
NUM_PARAMS = NUM_LAYERS * 12 + 4


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)


@pytest.fixture(scope="module")
def config():
    return hf_config()


class _Slice:
    """Checkpoint slice stub with the safetensors slice interface."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def get_shape(self):
        return list(self.tensor.shape)


def _starcoder2_config(config):
    return src_module("config").Starcoder2Config.from_configs(config, NeuronConfig())


def _mapped_keys(value) -> list[str]:
    return value if isinstance(value, list) else [value]


# =============================================================================
# 1. Registration contract
# =============================================================================


def test_registration_contract(config):
    """The architecture reaches this factory, which satisfies vLLM's protocol."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    Factory = src_package().Starcoder2ForCausalLM
    assert config.architectures == [ARCHITECTURE]
    # A contrib model is deliberately absent from
    # ``vllm_neuron.model.registry.get_models()``. The user registers it, so
    # the test checks the export that ``ModelRegistry.register_model`` gets.
    assert getattr(src_package(), ARCHITECTURE) is Factory
    assert Factory is src_module("factory").Starcoder2ForCausalLM
    assert is_vllm_model(Factory)
    assert is_text_generation_model(Factory)

    # The plugin entry point registers the factory by its installed name.
    # vLLM ships its own entry for this architecture, and register() must
    # replace it. Restore the registry afterwards.
    registry = ModelRegistry.models
    previous = registry.get(ARCHITECTURE)
    try:
        src_module("plugin").register()
        entry = registry[ARCHITECTURE]
        assert entry.module_name == "starcoder2_neuron.factory"
        assert entry.class_name == ARCHITECTURE
    finally:
        if previous is None:
            registry.pop(ARCHITECTURE, None)
        else:
            registry[ARCHITECTURE] = previous


# =============================================================================
# 2. Config parse
# =============================================================================


def test_config_parse(config):
    """The config parse keeps every architecture field of the checkpoint."""
    parsed = _starcoder2_config(config)
    assert parsed.hidden_size == HIDDEN_SIZE
    assert parsed.intermediate_size == INTERMEDIATE_SIZE
    assert parsed.num_hidden_layers == NUM_LAYERS
    assert parsed.num_attention_heads == NUM_Q_HEADS
    assert parsed.num_key_value_heads == NUM_KV_HEADS
    assert parsed.head_dim == HEAD_DIM
    assert parsed.vocab_size == config.vocab_size
    assert parsed.norm_epsilon == config.norm_epsilon
    # transformers 5 keeps rope_theta under rope_parameters. The compat helper
    # hoists it, so the field is not the dataclass default.
    assert parsed.rope_theta == config.rope_parameters["rope_theta"]
    assert parsed.hidden_act == "gelu_pytorch_tanh"
    assert parsed.use_bias is True
    assert parsed.tie_word_embeddings is True
    assert parsed.sliding_window == config.sliding_window
    # The checkpoint is FP32. The implementation always computes in BF16.
    assert parsed.torch_dtype == torch.bfloat16


# =============================================================================
# 3. Factory validation
# =============================================================================


def test_factory_validation(config):
    """The factory accepts the checkpoint config and refuses unsupported ones."""
    Factory = src_package().Starcoder2ForCausalLM

    Factory._validate_config(config, NeuronConfig())

    bad_act = copy.deepcopy(config)
    bad_act.hidden_act = "relu"
    with pytest.raises(ValueError, match="hidden_act"):
        Factory._validate_config(bad_act, None)

    no_bias = copy.deepcopy(config)
    no_bias.use_bias = False
    with pytest.raises(ValueError, match="use_bias"):
        Factory._validate_config(no_bias, None)

    scaled_rope = copy.deepcopy(config)
    scaled_rope.rope_parameters = {
        **config.rope_parameters,
        "rope_type": "linear",
        "factor": 2.0,
    }
    with pytest.raises(ValueError, match="rope_type"):
        Factory._validate_config(scaled_rope, None)

    quantized = NeuronConfig()
    quantized.quantization = "fp8"
    with pytest.raises(ValueError, match="quantization"):
        Factory._validate_config(config, quantized)

    # The implementation does not apply the sliding window, so the factory
    # refuses a max_model_len that is larger than the window.
    def vllm_config_with(max_model_len):
        vllm_config = MagicMock()
        vllm_config.model_config.max_model_len = max_model_len
        return vllm_config

    target = "vllm.config.get_current_vllm_config_or_none"
    with patch(target, return_value=vllm_config_with(config.sliding_window)):
        Factory._validate_config(config, None)
    with patch(target, return_value=vllm_config_with(config.sliding_window + 1)):
        with pytest.raises(ValueError, match="sliding_window"):
            Factory._validate_config(config, None)


# =============================================================================
# 4. TP size checks
# =============================================================================


def test_tp_size_checks(config, monkeypatch):
    """24 query heads and 2 KV heads: TP 1, 2, 4 and 8 build, TP 3 and 16 do not.

    Only TP 4 has a recorded device run. The constructor checks do not prove
    that the decode kernel accepts the head count of another TP size.
    """
    model = src_module("model")
    parsed = _starcoder2_config(config)

    for world_size in (1, 2, 4, 8):
        patch_tp_group(monkeypatch, world_size, 0)
        with torch.device("meta"):
            attn = model.Starcoder2Attention(parsed, layer_idx=0)
            model.Starcoder2MLP(parsed)
        assert attn.num_attention_heads_per_rank == NUM_Q_HEADS // world_size
        if world_size >= NUM_KV_HEADS:
            assert attn.num_key_value_heads_per_rank == 1
            assert attn.num_kv_replicas == world_size // NUM_KV_HEADS
        else:
            assert attn.num_key_value_heads_per_rank == NUM_KV_HEADS
            assert attn.num_kv_replicas == 1
    # TP 3 does not divide the KV heads. TP 16 does not divide the query heads.
    for world_size in (3, 16):
        patch_tp_group(monkeypatch, world_size, 0)
        with pytest.raises(ValueError), torch.device("meta"):
            model.Starcoder2Attention(parsed, layer_idx=0)


# =============================================================================
# 5. Checkpoint mapping coverage
# =============================================================================


def _layer_parameters(config, monkeypatch, world_size: int, rank: int):
    """Build decoder layer 0 on the meta device. Return its named parameters."""
    model = src_module("model")
    patch_tp_group(monkeypatch, world_size, rank)
    with torch.device("meta"):
        layer = model.Starcoder2DecoderLayer(_starcoder2_config(config), layer_idx=0)
    return dict(layer.named_parameters())


def test_checkpoint_mapping_coverage(config, monkeypatch):
    """Each parameter maps to checkpoint keys. Each key is used one time."""
    Model = src_module("model").Starcoder2ForCausalLM
    info = checkpoint_tensor_info()
    assert "lm_head.weight" not in info, "Expected tied weights"
    # The checkpoint stores FP32 tensors. The loader casts them to BF16.
    assert {dtype for _, dtype in info.values()} == {"F32"}

    mappings = Model.checkpoint_mappings(NUM_LAYERS, tie_word_embeddings=True)
    used = [key for value in mappings.values() for key in _mapped_keys(value)]
    # A parameter without a mapping uses its own name as the key.
    unmapped = {"model.embed_tokens.weight", "model.norm.weight", "model.norm.bias"}
    # The tied LM head maps to the embedding key. The embedding reads the same
    # key through its own name, so the key serves both by design.
    assert mappings["lm_head.weight"] == "model.embed_tokens.weight"
    used.remove("model.embed_tokens.weight")
    used_keys = set(used) | unmapped
    assert len(used) == len(set(used)), "a checkpoint key is used twice"
    assert used_keys == set(info), (
        f"not mapped: {sorted(set(info) - used_keys)[:5]}, "
        f"not in checkpoint: {sorted(used_keys - set(info))[:5]}"
    )

    layer_names = list(_layer_parameters(config, monkeypatch, TP_SIZE, 0))
    param_names = {"lm_head.weight"} | unmapped
    for i in range(NUM_LAYERS):
        param_names |= {f"model.layers.{i}.{name}" for name in layer_names}
    assert len(param_names) == NUM_PARAMS
    assert set(mappings) | unmapped == param_names


# =============================================================================
# 6. Per-rank shapes
# =============================================================================


def test_per_rank_shapes(config, monkeypatch):
    """Each rank's loader output has the shape of the rank's parameter."""
    Model = src_module("model").Starcoder2ForCausalLM
    info = checkpoint_tensor_info()
    mappings = Model.checkpoint_mappings(1, tie_word_embeddings=True)
    prefix = "model.layers.0."

    for rank in range(TP_SIZE):
        params = _layer_parameters(config, monkeypatch, TP_SIZE, rank)
        for name, param in params.items():
            keys = _mapped_keys(mappings[prefix + name])
            slices = [_Slice(torch.empty(info[key][0], device="meta")) for key in keys]
            loaded = get_weight_loader(param).load(slices, rank)
            assert tuple(loaded.shape) == tuple(param.shape), (
                f"rank {rank} {name}: loader gives {tuple(loaded.shape)}, "
                f"parameter is {tuple(param.shape)}"
            )

    # The vocab-sharded embedding and the column-parallel LM head both read
    # the embedding tensor, 1/TP of the vocab per rank.
    vocab, hidden = info["model.embed_tokens.weight"][0]
    assert vocab % TP_SIZE == 0
    loader = sharding_weight_loader(
        shard_dim=0,
        shard_size=vocab // TP_SIZE,
        num_shards=TP_SIZE,
        is_storage_transposed=False,
    )
    embed = _Slice(torch.empty(vocab, hidden, device="meta"))
    for rank in range(TP_SIZE):
        assert tuple(loader.load([embed], rank).shape) == (vocab // TP_SIZE, hidden)


# =============================================================================
# 7. Weight loader transforms
# =============================================================================


def test_weight_loader_transforms():
    """The TP 4 loaders give each rank the correct part of each tensor."""
    fused_qkv_bias_loader = src_module("model").fused_qkv_bias_loader

    q_size = NUM_Q_HEADS * HEAD_DIM // TP_SIZE  # 768
    kv_size = HEAD_DIM  # 1 KV head per rank
    num_kv_replicas = TP_SIZE // NUM_KV_HEADS  # 2

    q_b = torch.randn(NUM_Q_HEADS * HEAD_DIM, dtype=torch.float32)
    k_b = torch.randn(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
    v_b = torch.randn(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
    q_w = torch.randn(NUM_Q_HEADS * HEAD_DIM, 64, dtype=torch.float32)
    k_w = torch.randn(NUM_KV_HEADS * HEAD_DIM, 64, dtype=torch.float32)
    v_w = torch.randn(NUM_KV_HEADS * HEAD_DIM, 64, dtype=torch.float32)
    o_b = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
    fc_b = torch.randn(INTERMEDIATE_SIZE, dtype=torch.float32)

    bias_loader = fused_qkv_bias_loader(q_size, kv_size, num_kv_replicas)
    weight_loader = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=TP_SIZE,
        is_storage_transposed=True,
        num_kv_replicas=num_kv_replicas,
    )
    o_bias_loader = scaled_bias_loader(scale=TP_SIZE)
    fc_bias_loader = sharding_weight_loader(
        shard_dim=0,
        shard_size=INTERMEDIATE_SIZE // TP_SIZE,
        num_shards=TP_SIZE,
    )

    for rank in range(TP_SIZE):
        kv_rank = rank // num_kv_replicas
        q_rows = slice(rank * q_size, (rank + 1) * q_size)
        kv_rows = slice(kv_rank * kv_size, (kv_rank + 1) * kv_size)

        bias = bias_loader.transform([_Slice(q_b), _Slice(k_b), _Slice(v_b)], rank)
        expected_bias = torch.cat([q_b[q_rows], k_b[kv_rows], v_b[kv_rows]])
        torch.testing.assert_close(bias, expected_bias, rtol=0, atol=0)

        # The fused weight and the fused bias must use the same column order.
        weight = weight_loader.transform([_Slice(q_w), _Slice(k_w), _Slice(v_w)], rank)
        expected_weight = torch.cat([q_w[q_rows], k_w[kv_rows], v_w[kv_rows]]).T
        torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)

        # The row-parallel bias is divided in FP32, before the cast to BF16.
        o_bias = o_bias_loader.transform([_Slice(o_b)], rank)
        assert o_bias.dtype == torch.float32
        torch.testing.assert_close(o_bias * TP_SIZE, o_b)

        fc_bias = fc_bias_loader.transform([_Slice(fc_b)], rank)
        shard = INTERMEDIATE_SIZE // TP_SIZE
        torch.testing.assert_close(
            fc_bias, fc_b[rank * shard : (rank + 1) * shard], rtol=0, atol=0
        )


# =============================================================================
# 8. MLP torch fallback without torch's GELU
# =============================================================================


def test_mlp_fallback_without_torch_gelu(monkeypatch):
    """The torch MLP fallback matches HF and never calls ``F.gelu``.

    On the Neuron route that is not ``neuron_native``, ``libtorch_neuronx_lite``
    replaces ``torch._C._nn.gelu`` with a function that ``torch.compile``
    cannot trace, so the fallback must not reach it. The device path runs the
    NKI MLP kernel, which computes the activation itself.
    """
    from transformers import Starcoder2Config as HFConfig
    from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2MLP

    model = src_module("model")
    x = torch.linspace(-8.0, 8.0, 4097)
    torch.testing.assert_close(
        model._gelu_tanh(x), F.gelu(x, approximate="tanh"), rtol=1e-6, atol=1e-6
    )

    hidden, intermediate, tokens = 256, 512, 8
    hf_mlp = Starcoder2MLP(
        HFConfig(hidden_size=hidden, intermediate_size=intermediate, use_bias=True)
    ).eval()
    with torch.no_grad():
        for param in hf_mlp.parameters():
            param.copy_(torch.randn_like(param) * 0.05)

    config = model.Starcoder2Config(
        hidden_size=hidden, intermediate_size=intermediate, torch_dtype=torch.float32
    )
    patch_tp_group(monkeypatch, 1, 0)
    mlp = model.Starcoder2MLP(config)
    with torch.no_grad():
        mlp.c_fc_weight.copy_(hf_mlp.c_fc.weight.T)
        mlp.c_fc_bias.copy_(hf_mlp.c_fc.bias)
        mlp.c_proj_weight.copy_(hf_mlp.c_proj.weight.T)
        mlp.c_proj_bias.copy_(hf_mlp.c_proj.bias)

    h = torch.randn(tokens, hidden)
    with torch.no_grad():
        expected = hf_mlp(h)

    def _refuse(*args, **kwargs):
        raise AssertionError("the MLP fallback called torch's gelu")

    monkeypatch.setattr(torch.nn.functional, "gelu", _refuse)
    monkeypatch.setattr(torch._C._nn, "gelu", _refuse)
    with torch.no_grad():
        actual = mlp._torch_mlp(h)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "intermediate_per_rank, tokens, expect_kernel",
    [
        # TP 4: decode and both prefill buckets of the recorded run.
        (INTERMEDIATE_SIZE // 4, 1, True),
        (INTERMEDIATE_SIZE // 4, 256, True),
        (INTERMEDIATE_SIZE // 4, 1024, True),
        # TP 2 and TP 1 prefill: the prefill kernel refuses the shape.
        (INTERMEDIATE_SIZE // 2, 256, False),
        (INTERMEDIATE_SIZE, 1024, False),
        # TP 1 decode: the decode kernel takes it.
        (INTERMEDIATE_SIZE, 1, True),
    ],
)
def test_mlp_kernel_routing(monkeypatch, intermediate_per_rank, tokens, expect_kernel):
    """``NF.mlp`` is called only for a shape that its NKI kernel runs.

    For any other shape ``NF.mlp`` would take its own torch fallback, which
    calls ``F.gelu`` inside the traced graph, so the model uses ``_torch_mlp``
    instead. The test makes the kernel predicate act as on a NeuronCore with
    LNC 2, and builds the MLP at TP 1 with the per-rank intermediate size.
    """
    import importlib

    model = src_module("model")
    functional_mlp = importlib.import_module("vllm_neuron.functional.mlp")
    monkeypatch.setattr(functional_mlp, "can_run_kernel", lambda *a, **k: True)
    monkeypatch.setattr(functional_mlp, "get_logical_nc_config", lambda: 2)

    kernel_calls, fallback_calls = [], []

    def fake_kernel(hidden, *args, **kwargs):
        kernel_calls.append(kwargs["act_fn"])
        return torch.zeros(hidden.shape[0], HIDDEN_SIZE, device="meta")

    def fake_fallback(self, hidden):
        fallback_calls.append(hidden.shape[0])
        return torch.zeros(hidden.shape[0], HIDDEN_SIZE, device="meta")

    monkeypatch.setattr(model.NF, "mlp", fake_kernel)
    monkeypatch.setattr(model.Starcoder2MLP, "_torch_mlp", fake_fallback)

    patch_tp_group(monkeypatch, 1, 0)
    config = model.Starcoder2Config(
        hidden_size=HIDDEN_SIZE, intermediate_size=intermediate_per_rank
    )
    with torch.device("meta"):
        mlp = model.Starcoder2MLP(config)
        hidden = torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    mlp(hidden, is_prefill=tokens > 1)

    if expect_kernel:
        assert kernel_calls == [model.ActFnType.GELU_Tanh_Approx]
        assert fallback_calls == []
    else:
        assert kernel_calls == []
        assert fallback_calls == [tokens]


# =============================================================================
# 9. Full checkpoint load (needs the weights)
# =============================================================================


class _LoadReport(nn.Module):
    """Hold the result of one rank's checkpoint load as a tensor."""

    def __init__(self, report: torch.Tensor):
        super().__init__()
        self.register_buffer("report", report)

    def forward(self, _x):
        return self.report


def _load_full_model(checkpoint: str):
    """Build the full model on one rank and load its checkpoint shard.

    Return a module that reports: the number of parameters, the number of
    missing parameters, the number of shape mismatches, the number of
    non-finite values, and the number of unused checkpoint tensors.
    """
    import torch.distributed as dist
    from transformers import AutoConfig

    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    Factory = src_package().Starcoder2ForCausalLM
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = Factory.from_configs(AutoConfig.from_pretrained(checkpoint), NeuronConfig())
    mappings = model.checkpoint_mappings(
        model.config.num_hidden_layers, model.config.tie_word_embeddings
    )
    result = SafetensorsCheckpoint(checkpoint, None).load_sharded(
        rank, world_size, model, mappings, "cpu", strict=False
    )
    params = dict(model.named_parameters())
    shape_mismatch = sum(
        1
        for name, tensor in result.state_dict.items()
        if tuple(tensor.shape) != tuple(params[name].shape)
    )
    non_finite = sum(
        int((~torch.isfinite(tensor)).sum()) for tensor in result.state_dict.values()
    )
    missing = len(set(params) - set(result.state_dict))
    report = torch.tensor(
        [
            len(params),
            missing,
            shape_mismatch,
            non_finite,
            len(result.unexpected_keys),
        ],
        dtype=torch.int64,
    )
    return _LoadReport(report)


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_full_checkpoint_load(world_size):
    """Each rank loads every parameter of all 30 layers from the checkpoint.

    The test also checks that the model uses every checkpoint tensor, so no
    original weight is dropped.
    """
    from vllm_neuron.utils.executor import MPExecutor

    checkpoint = require_checkpoint()
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(_load_full_model, checkpoint),
        device="cpu",
        load_timeout=1800,
    )
    executor.dispatch(torch.zeros(1))
    reports = executor.collect(timeout=1800)

    assert len(reports) == world_size
    for rank, report in enumerate(reports):
        num_params, missing, shape_mismatch, non_finite, unused = report.tolist()
        assert num_params == NUM_PARAMS, f"rank {rank}"
        assert missing == 0, f"rank {rank}: {missing} parameters are not loaded"
        assert shape_mismatch == 0, f"rank {rank}: {shape_mismatch} shape errors"
        assert non_finite == 0, f"rank {rank}: {non_finite} non-finite values"
        assert unused == 0, f"rank {rank}: {unused} checkpoint tensors are unused"
