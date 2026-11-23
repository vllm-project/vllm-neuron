# SPDX-License-Identifier: Apache-2.0
"""GLM (zai-org/glm-4-9b-chat-hf) CPU tests: registration, loading and components.

These tests run in CPU mode (``VLLM_NEURON_CPU_MODE=1``) and need no Neuron
device.

Coverage:
  - The registration contract: the checkpoint's architecture string, the
    factory that ``src/`` exports, vLLM's model protocol and the plugin entry
    point.
  - The config parse from the checkpoint, and the factory checks.
  - The RoPE load-time permutation against the HF GLM ``apply_rotary_pos_emb``,
    the RoPE frequencies, ``GlmRMSNorm`` against HF, and a compiled rotary
    graph without an in-place write.
  - The fused QKV (weight and bias) loader and the fused gate/up loader at
    TP 16, also for a model built on the meta device.
  - The weight mapping against every tensor in the checkpoint index.
  - The prefill KV write and the segmented-prefill block table with padded
    prompts.
  - A full-model load of all 40 layers at TP 16 through ``load_weights``, then
    a CPU prefill whose logits are compared three-way with the HF FP32 and
    BF16 models. This one needs the full checkpoint (``GLM4_CHECKPOINT_PATH``).

The tests that read the checkpoint metadata download ``config.json``, the
weight index and the tokenizer of the pinned revision (about 20 MB) unless
``GLM4_CHECKPOINT_PATH`` is set.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm/test/unit/ -v
"""

from functools import partial

import pytest
import torch

from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.model.neuron_config import NeuronConfig

from _local_src import src_module, src_package
from glm_test_utils import (
    ARCHITECTURE,
    GOLDEN_VARIANT,
    MODEL_ID,
    MockSlice,
    hf_glm_config,
    load_config_dict,
    load_weight_index,
    metadata_path,
    weights_path,
)

PREFILL_LEN = 64
BLOCK_SIZE = 16

# The recorded topology: TP 16 (2 query heads per rank, the 2 KV heads
# replicated 8 times).
TP = 16
HIDDEN = 4096
HEAD_DIM = 128
ROTARY_DIM = 64
N_Q, N_KV = 32, 2
INTER = 13696


def _glm_config_cls():
    return src_module("config").GlmConfig


def _weight_loaders():
    return src_module("weight_loaders")


# =============================================================================
# Registration, factory and config
# =============================================================================


def test_glm_registration_contract(monkeypatch):
    """The checkpoint's architecture must resolve to this folder's factory.

    A contrib model is not in the plugin's registry. The user registers the
    factory under the checkpoint's architecture string, either from a script
    or through the ``vllm.general_plugins`` entry point in ``src/plugin.py``.
    """
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
        is_vllm_model,
    )

    cfg_dict = load_config_dict(metadata_path())
    assert cfg_dict["architectures"] == [ARCHITECTURE]

    factory_cls = getattr(src_package(), ARCHITECTURE)
    assert factory_cls is src_module("factory").GlmForCausalLM
    assert is_vllm_model(factory_cls)
    assert is_text_generation_model(factory_cls)

    # The entry point registers the installed package name for the same
    # architecture string.
    calls = []

    class _FakeRegistry:
        @staticmethod
        def register_model(arch, target):
            calls.append((arch, target))

    import vllm

    monkeypatch.setattr(vllm, "ModelRegistry", _FakeRegistry)
    plugin = src_module("plugin")
    plugin.register()
    assert calls == [(ARCHITECTURE, "glm_neuron.factory:GlmForCausalLM")]
    assert plugin.ARCHITECTURE == ARCHITECTURE


def test_glm_config_from_checkpoint():
    """GlmConfig must read the checkpoint geometry."""
    GlmConfig = _glm_config_cls()
    cfg_dict = load_config_dict(metadata_path())
    cfg = GlmConfig.from_configs(cfg_dict, NeuronConfig())
    assert cfg.hidden_size == HIDDEN
    assert cfg.intermediate_size == INTER
    assert cfg.num_hidden_layers == 40
    assert cfg.num_attention_heads == N_Q
    assert cfg.num_key_value_heads == N_KV
    assert cfg.head_dim == HEAD_DIM
    assert cfg.rotary_dim == ROTARY_DIM
    assert cfg.vocab_size == 151552
    assert cfg.attention_bias is True
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.bfloat16
    assert cfg.rms_norm_eps == pytest.approx(cfg_dict["rms_norm_eps"])


def test_glm_config_from_hf_config():
    """The transformers config (the object vLLM passes) must parse the same way.

    transformers 5 keeps ``partial_rotary_factor`` and ``rope_theta`` under
    ``rope_parameters`` and writes ``dtype`` instead of ``torch_dtype``.
    """
    GlmConfig = _glm_config_cls()
    path = metadata_path()
    hf_config = hf_glm_config(path)
    cfg = GlmConfig.from_configs(hf_config, NeuronConfig())
    ref = GlmConfig.from_configs(load_config_dict(path), NeuronConfig())
    assert cfg.rotary_dim == ref.rotary_dim == ROTARY_DIM
    assert cfg.rope_theta == ref.rope_theta == 10000.0
    assert cfg.torch_dtype == torch.bfloat16
    assert cfg.head_dim == ref.head_dim


@pytest.mark.parametrize(
    "override, match",
    [
        ({"torch_dtype": "float32"}, "bfloat16"),
        ({"hidden_act": "gelu"}, "silu"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "rope_scaling"),
    ],
)
def test_glm_factory_rejects_unsupported(override, match):
    """The factory must stop at an unsupported config before construction."""
    factory_cls = src_module("factory").GlmForCausalLM

    cfg_dict = dict(load_config_dict(metadata_path()))
    cfg_dict.pop("dtype", None)
    cfg_dict.update(override)
    with pytest.raises(ValueError, match=match):
        factory_cls._validate_config(cfg_dict, NeuronConfig())


# =============================================================================
# RoPE layout and RMSNorm
# =============================================================================


def test_glm_rope_permutation_matches_hf():
    """Permuted contiguous RoPE must equal the permuted HF GLM interleaved RoPE."""
    from transformers.models.glm.modeling_glm import (
        GlmRotaryEmbedding as HFRotary,
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )

    wl = _weight_loaders()
    apply_rotary_emb = src_module("model")._apply_rotary_emb

    torch.manual_seed(42)
    ckpt = metadata_path()
    hf_config = hf_glm_config(ckpt)
    cfg = _glm_config_cls().from_configs(load_config_dict(ckpt), NeuronConfig())
    seq_len, n_heads, head_dim = 96, 4, cfg.head_dim
    positions = torch.arange(seq_len, dtype=torch.int64)

    q = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float32)
    cos_hf, sin_hf = HFRotary(config=hf_config)(q, positions.unsqueeze(0))
    q_hf, k_hf = hf_apply_rotary_pos_emb(q, k, cos_hf, sin_hf)

    # The port: the same permutation on the Q/K head rows, contiguous RoPE over
    # the full head, zero frequencies for the pass-through positions.
    perm = wl.glm_rope_permutation(head_dim, cfg.rotary_dim)
    inv_freq = wl.glm_kernel_inv_freq(head_dim, cfg.rotary_dim, cfg.rope_theta)
    freqs = positions[:, None].float() * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    q_port = apply_rotary_emb(q[..., perm], cos, sin)
    k_port = apply_rotary_emb(k[..., perm], cos, sin)

    torch.testing.assert_close(q_port, q_hf[..., perm], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_port, k_hf[..., perm], rtol=1e-5, atol=1e-5)
    scores_hf = q_hf @ k_hf.transpose(-1, -2)
    scores_port = q_port @ k_port.transpose(-1, -2)
    torch.testing.assert_close(scores_port, scores_hf, rtol=1e-4, atol=1e-4)


def test_glm_model_rotary_matches_reference():
    """GlmRotaryEmbedding must give cos/sin = cat(freqs, freqs) of the kernel inv_freq."""
    GlmRotaryEmbedding = src_module("model").GlmRotaryEmbedding
    wl = _weight_loaders()

    cfg = _glm_config_cls().from_configs(load_config_dict(metadata_path()), None)
    positions = torch.arange(37, dtype=torch.int64)
    cos, sin = GlmRotaryEmbedding(cfg)(
        positions, device=torch.device("cpu"), dtype=torch.float32
    )
    inv_freq = wl.glm_kernel_inv_freq(cfg.head_dim, cfg.rotary_dim, cfg.rope_theta)
    freqs = positions[:, None].float() * inv_freq[None, :]
    assert cos.shape == (37, cfg.head_dim)
    torch.testing.assert_close(cos, torch.cat((freqs, freqs), -1).cos())
    torch.testing.assert_close(sin, torch.cat((freqs, freqs), -1).sin())
    # The pass-through positions must not rotate.
    half_d, half_r = cfg.head_dim // 2, cfg.rotary_dim // 2
    assert torch.all(cos[:, half_r:half_d] == 1.0)
    assert torch.all(sin[:, half_r:half_d] == 0.0)


# The full model makes cos/sin inside the compiled graph. An earlier version of
# ``glm_kernel_inv_freq`` wrote the frequencies into a zero tensor in place, and
# the device then gave NaN logits although the module checks (cos/sin made on
# the host) passed. This test keeps the graph free of in-place writes.


class _GraphCapture:
    """A torch.compile backend that keeps the Dynamo FX graph and runs it eagerly."""

    def __init__(self):
        self.graphs = []

    def __call__(self, gm, example_inputs):
        self.graphs.append(gm)
        return gm.forward


def test_glm_rotary_graph_has_no_in_place_write():
    """The compiled rotary graph must not write into a tensor in place."""
    import operator

    GlmRotaryEmbedding = src_module("model").GlmRotaryEmbedding

    cfg = _glm_config_cls().from_configs(load_config_dict(metadata_path()), None)
    rotary = GlmRotaryEmbedding(cfg)
    capture = _GraphCapture()
    compiled = torch.compile(rotary, fullgraph=True, backend=capture)
    positions = torch.arange(37, dtype=torch.int32)
    cos, sin = compiled(positions, device=torch.device("cpu"), dtype=torch.bfloat16)

    assert capture.graphs, "Dynamo did not capture the rotary graph."
    for gm in capture.graphs:
        for node in gm.graph.nodes:
            if node.op not in ("call_function", "call_method"):
                continue
            target = node.target
            name = getattr(target, "__name__", str(target))
            assert target is not operator.setitem, f"In-place setitem in {node}."
            assert not name.endswith("_") or name.startswith("__"), (
                f"In-place op {name} in the rotary graph."
            )
    ref_cos, ref_sin = rotary(
        positions, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    torch.testing.assert_close(cos, ref_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, ref_sin, rtol=0, atol=0)


def test_glm_kernel_inv_freq_matches_hf():
    """The inverse frequencies must equal the HF GLM values, then zeros."""
    from transformers.models.glm.modeling_glm import GlmRotaryEmbedding as HFRotary

    wl = _weight_loaders()
    ckpt = metadata_path()
    cfg = _glm_config_cls().from_configs(load_config_dict(ckpt), None)
    hf_inv_freq = HFRotary(config=hf_glm_config(ckpt)).inv_freq.float()
    inv_freq = wl.glm_kernel_inv_freq(cfg.head_dim, cfg.rotary_dim, cfg.rope_theta)

    half_d, half_r = cfg.head_dim // 2, cfg.rotary_dim // 2
    assert inv_freq.shape == (half_d,)
    assert inv_freq.dtype == torch.float32
    assert hf_inv_freq.shape == (half_r,)
    torch.testing.assert_close(inv_freq[:half_r], hf_inv_freq, rtol=1e-6, atol=0)
    assert torch.all(inv_freq[half_r:] == 0.0)
    # A full rotary head has no zero frequencies.
    full = wl.glm_kernel_inv_freq(8, 8, 10000.0)
    assert full.shape == (4,) and torch.all(full > 0)


@pytest.mark.parametrize("scale", [1.0, 1e-4, 300.0])
def test_glm_rmsnorm_matches_hf(scale: float):
    """GlmRMSNorm in BF16 must equal the HF GlmRMSNorm, which uses an FP32 variance."""
    from transformers.models.glm.modeling_glm import GlmRMSNorm as HFRMSNorm

    GlmRMSNorm = src_module("model").GlmRMSNorm

    cfg = _glm_config_cls().from_configs(load_config_dict(metadata_path()), None)
    torch.manual_seed(42)
    weight = (1.0 + 0.1 * torch.randn(cfg.hidden_size)).to(torch.bfloat16)
    hidden = (scale * torch.randn(64, cfg.hidden_size)).to(torch.bfloat16)
    # One row of zeros: the eps alone must keep this row finite.
    hidden[3] = 0.0

    norm = GlmRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, torch.bfloat16)
    hf_norm = HFRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
        hf_norm.weight.copy_(weight)
        out = norm(hidden)
        expected = hf_norm(hidden)

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# =============================================================================
# Loaders (synthetic slices, real geometry)
# =============================================================================


@pytest.mark.parametrize("is_bias", [False, True])
@pytest.mark.parametrize("rank", [0, 7, 8, 15])
def test_glm_fused_qkv_loader_tp16(rank, is_bias):
    """The QKV loader must shard heads, replicate KV, permute Q/K and keep V."""
    wl = _weight_loaders()
    torch.manual_seed(42)
    q_size = N_Q * HEAD_DIM // TP
    kv_size = HEAD_DIM
    replicas = TP // N_KV
    shape_q = (N_Q * HEAD_DIM,) if is_bias else (N_Q * HEAD_DIM, HIDDEN)
    shape_kv = (N_KV * HEAD_DIM,) if is_bias else (N_KV * HEAD_DIM, HIDDEN)
    q = torch.randn(*shape_q)
    k = torch.randn(*shape_kv)
    v = torch.randn(*shape_kv)

    loader = wl.glm_fused_qkv_loader(
        q_size=q_size,
        kv_size=kv_size,
        num_shards=TP,
        num_kv_replicas=replicas,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        is_bias=is_bias,
    )
    out = loader.load([MockSlice(q), MockSlice(k), MockSlice(v)], rank)

    perm = wl.glm_rope_permutation(HEAD_DIM, ROTARY_DIM)
    kv_rank = rank // replicas

    def heads(t, start_head, n_heads, permute):
        rows = []
        for h in range(start_head, start_head + n_heads):
            block = t[h * HEAD_DIM : (h + 1) * HEAD_DIM]
            rows.append(block[perm] if permute else block)
        return torch.cat(rows, dim=0)

    exp_q = heads(q, rank * (N_Q // TP), N_Q // TP, True)
    exp_k = heads(k, kv_rank, 1, True)
    exp_v = heads(v, kv_rank, 1, False)
    expected = torch.cat([exp_q, exp_k, exp_v], dim=0)
    if is_bias:
        assert out.shape == (1, q_size + 2 * kv_size)
        torch.testing.assert_close(out[0], expected, rtol=0, atol=0)
    else:
        assert out.shape == (HIDDEN, q_size + 2 * kv_size)
        torch.testing.assert_close(out, expected.T, rtol=0, atol=0)


def _qkv_mock_slices(is_bias: bool) -> list:
    torch.manual_seed(42)
    shape_q = (N_Q * HEAD_DIM,) if is_bias else (N_Q * HEAD_DIM, HIDDEN)
    shape_kv = (N_KV * HEAD_DIM,) if is_bias else (N_KV * HEAD_DIM, HIDDEN)
    tensors = [
        torch.randn(*shape_q, dtype=torch.bfloat16),
        torch.randn(*shape_kv, dtype=torch.bfloat16),
        torch.randn(*shape_kv, dtype=torch.bfloat16),
    ]
    return [MockSlice(t) for t in tensors]


def test_glm_rope_permutation_ignores_meta_device_context():
    """The permutation must be a CPU tensor in a meta device context.

    The model runner builds the model in ``torch.device("meta")``. A meta
    permutation there would make the loaded Q/K/V tensors meta tensors without
    data.
    """
    wl = _weight_loaders()
    with torch.device("meta"):
        perm = wl.glm_rope_permutation(HEAD_DIM, ROTARY_DIM)
    assert perm.device.type == "cpu"
    assert perm.tolist() == wl.glm_rope_permutation(HEAD_DIM, ROTARY_DIM).tolist()


@pytest.mark.parametrize("is_bias", [False, True])
@pytest.mark.parametrize("rank", [0, 15])
def test_glm_fused_qkv_loader_built_in_meta_context(rank, is_bias):
    """A loader made in a meta device context must give the same CPU data."""
    wl = _weight_loaders()
    loader_args = dict(
        q_size=N_Q * HEAD_DIM // TP,
        kv_size=HEAD_DIM,
        num_shards=TP,
        num_kv_replicas=TP // N_KV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        is_bias=is_bias,
    )
    with torch.device("meta"):
        meta_loader = wl.glm_fused_qkv_loader(**loader_args)
    cpu_loader = wl.glm_fused_qkv_loader(**loader_args)

    out = meta_loader.load(_qkv_mock_slices(is_bias), rank)
    expected = cpu_loader.load(_qkv_mock_slices(is_bias), rank)
    assert out.device.type == "cpu"
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


class _FlagReport(torch.nn.Module):
    """Return the flags that model_load made."""

    def __init__(self, flags: torch.Tensor):
        super().__init__()
        self.flags = flags

    def forward(self, x):
        return self.flags


def _load_meta_built_qkv_report(checkpoint_path: str):
    """``model_load``: run the QKV loaders of a meta-built model on each rank.

    Build one GLM layer in a ``torch.device("meta")`` context, as the model
    runner does, and build it again on the CPU. Run the QKV weight loader and
    the QKV bias loader of each model on synthetic slices with the real
    geometry. Return int32 flags: [cpu, finite, equal] for the weight, then for
    the bias.
    """
    import torch.distributed as dist

    from vllm_neuron.utils.weight_loader import get_weight_loader

    GlmForCausalLM = src_module("model").GlmForCausalLM
    neuron_config = NeuronConfig(on_device_sampling_config=None)
    config = _glm_config_cls().from_configs(
        load_config_dict(checkpoint_path), neuron_config
    )
    config.num_hidden_layers = 1

    def qkv_loaders(model) -> list:
        attn = model.model.layers[0].self_attn
        return [
            get_weight_loader(attn.qkv_proj_weight),
            get_weight_loader(attn.qkv_proj_bias),
        ]

    with torch.device("meta"):
        meta_model = GlmForCausalLM(config)
    if meta_model.model.layers[0].self_attn.qkv_proj_weight.device.type != "meta":
        raise RuntimeError("The meta device context did not make meta parameters.")
    meta_loaders = qkv_loaders(meta_model)
    del meta_model
    cpu_loaders = qkv_loaders(GlmForCausalLM(config))

    rank = dist.get_rank()
    flags = []
    for is_bias, meta_loader, cpu_loader in zip(
        (False, True), meta_loaders, cpu_loaders
    ):
        out = meta_loader.load(_qkv_mock_slices(is_bias), rank)
        expected = cpu_loader.load(_qkv_mock_slices(is_bias), rank)
        is_cpu = out.device.type == "cpu"
        finite = is_cpu and bool(torch.isfinite(out.float()).all())
        equal = is_cpu and torch.equal(out, expected)
        flags.extend([int(is_cpu), int(finite), int(equal)])
    return _FlagReport(torch.tensor(flags, dtype=torch.int32))


@pytest.mark.parametrize("world_size", [TP])
def test_glm_meta_built_model_qkv_loaders_give_cpu_data(world_size: int):
    """The QKV loaders of a meta-built model must give the CPU-built data on each rank.

    This is the model build of the model runner, at TP 16. It uses the
    checkpoint config and synthetic checkpoint slices with the real geometry.
    """
    from vllm_neuron.utils.executor import MPExecutor

    ckpt = metadata_path()
    config = _glm_config_cls().from_configs(
        load_config_dict(ckpt), NeuronConfig(on_device_sampling_config=None)
    )
    assert config.hidden_size == HIDDEN
    assert config.num_attention_heads == N_Q
    assert config.num_key_value_heads == N_KV

    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(_load_meta_built_qkv_report, checkpoint_path=ckpt),
        device="cpu",
        load_timeout=1800,
    )
    executor.dispatch(torch.zeros(1, dtype=torch.int32))
    outputs = executor.collect(timeout=1800)
    assert len(outputs) == world_size
    for rank, output in enumerate(outputs):
        flags = torch.as_tensor(output).tolist()
        assert flags == [1, 1, 1, 1, 1, 1], (
            f"Rank {rank} QKV loader flags [cpu, finite, equal] x [weight, bias] "
            f"are {flags}. Expected all ones."
        )


@pytest.mark.parametrize("rank", [0, 9, 15])
def test_glm_gate_up_loader_tp16(rank):
    """The gate/up loader must take rows from the gate half and the up half."""
    wl = _weight_loaders()
    torch.manual_seed(42)
    shard = INTER // TP
    gate_up = torch.randn(2 * INTER, 64)
    gate = wl.glm_gate_up_loader(INTER, shard, TP, is_up=False).load(
        [MockSlice(gate_up)], rank
    )
    up = wl.glm_gate_up_loader(INTER, shard, TP, is_up=True).load(
        [MockSlice(gate_up)], rank
    )
    torch.testing.assert_close(gate, gate_up[rank * shard : (rank + 1) * shard].T)
    torch.testing.assert_close(
        up, gate_up[INTER + rank * shard : INTER + (rank + 1) * shard].T
    )
    with pytest.raises(ValueError, match="gate_up_proj"):
        wl.glm_gate_up_loader(INTER, shard, TP, is_up=False).load(
            [MockSlice(torch.randn(INTER, 64))], rank
        )


# =============================================================================
# Weight mapping coverage against the checkpoint index
# =============================================================================


def test_glm_weight_mapping_covers_checkpoint():
    """The model mapping must use every checkpoint tensor and no other key."""
    GlmForCausalLM = src_module("model").GlmForCausalLM

    ckpt = metadata_path()
    cfg = _glm_config_cls().from_configs(load_config_dict(ckpt), None)
    ckpt_keys = set(load_weight_index(ckpt))

    mappings = GlmForCausalLM.get_weight_mappings(
        cfg.num_hidden_layers, cfg.attention_bias, cfg.tie_word_embeddings
    )
    used = set()
    for value in mappings.values():
        used.update([value] if isinstance(value, str) else value)
    direct = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(cfg.num_hidden_layers):
        direct.add(f"model.layers.{i}.input_layernorm.weight")
        direct.add(f"model.layers.{i}.post_attention_layernorm.weight")
    used |= direct

    assert sorted(ckpt_keys - used) == [], "checkpoint tensors without a parameter"
    assert sorted(used - ckpt_keys) == [], "mapped keys missing in the checkpoint"
    layers = {int(k.split(".")[2]) for k in ckpt_keys if k.startswith("model.layers.")}
    assert layers == set(range(cfg.num_hidden_layers))


# =============================================================================
# Prefill KV write and segmented prefill with padded prompts
# =============================================================================
#
# The runner pads a short prompt to the prefill bucket with the slot
# PAD_SLOT_ID (-1). A raw ``-1 // block_size`` writes outside the KV cache on
# the device, and the segmented-prefill kernel keeps stale SBUF data for an
# unused (-1) block column. Both made NaN logits for short prompts in a padded
# bucket during the port's device runs; a full-bucket prompt passed. These tests
# keep both fixes in place.


def _bare_attention(num_q_heads: int, kv_heads: int, head_dim: int, block_size: int):
    """Return a GlmAttention with only the fields that forward_prefill reads."""
    GlmAttention = src_module("model").GlmAttention

    q_size, kv_size = num_q_heads * head_dim, kv_heads * head_dim
    attn = GlmAttention.__new__(GlmAttention)
    torch.nn.Module.__init__(attn)
    attn.layer_idx = 0
    attn.dtype = torch.float32
    attn.head_dim = head_dim
    attn.scaling = head_dim**-0.5
    attn.world_size = 1
    attn.num_attention_heads_per_rank = num_q_heads
    attn.num_key_value_heads_per_rank = kv_heads
    attn.num_key_value_groups = num_q_heads // kv_heads
    attn.qkv_split_indices = [q_size, q_size + kv_size]
    attn.qkv_proj_weight = None
    attn.qkv_proj_bias = None
    attn.o_proj_weight = None
    attn.k_cache = torch.zeros(4, kv_heads, block_size, head_dim)
    attn.v_cache = torch.zeros(4, kv_heads, block_size, head_dim)
    return attn


def test_glm_prefill_kv_write_sends_pad_slots_to_null_block():
    """A pad slot (-1) must write to the null block and keep the live blocks."""
    GlmAttention = src_module("model").GlmAttention

    torch.manual_seed(42)
    num_blocks, kv_heads, block_size, head_dim = 3, 1, 4, 8
    attn = GlmAttention.__new__(GlmAttention)
    torch.nn.Module.__init__(attn)
    attn.head_dim = head_dim
    attn.num_key_value_heads_per_rank = kv_heads
    attn.k_cache = torch.zeros(num_blocks, kv_heads, block_size, head_dim)
    attn.v_cache = torch.zeros(num_blocks, kv_heads, block_size, head_dim)

    # Two real tokens in the last block (block 2), then two pad tokens.
    slot_mapping = torch.tensor([8, 9, -1, -1], dtype=torch.int64)
    k = torch.randn(kv_heads, 4, head_dim)
    v = torch.randn(kv_heads, 4, head_dim)
    attn._write_prefill_kv(k, v, slot_mapping, block_size)

    for cache, src in ((attn.k_cache, k), (attn.v_cache, v)):
        torch.testing.assert_close(cache[2, 0, 0], src[0, 0])
        torch.testing.assert_close(cache[2, 0, 1], src[0, 1])
        # The pad tokens must not change the last position of the last block.
        assert torch.count_nonzero(cache[2, 0, 2:]) == 0
        # Block 1 has no token, and the pad tokens must not write to it.
        assert torch.count_nonzero(cache[1]) == 0


def _skip_dma_segmented_attention(
    q,
    k_cache,
    v_cache,
    block_tables,
    prior_tokens,
    block_size,
    scale,
    tp_out,
    **_unused,
):
    """Emulate the device segmented prefill kernel for one request, no prior.

    The kernel skips the DMA for a block column of -1 and keeps the stale SBUF
    tile. This emulation fills that tile with NaN, the worst stale value. The
    PV matmul then multiplies each masked weight (0) by V, as the kernel does.
    """
    assert int(prior_tokens.reshape(-1)[0]) == 0
    bt = block_tables[0].to(torch.int64)
    skipped = (bt < 0).view(-1, 1, 1, 1)
    k_blocks = torch.where(skipped, float("nan"), k_cache[bt.clamp_min(0)])
    v_blocks = torch.where(skipped, float("nan"), v_cache[bt.clamp_min(0)])
    num_kv_heads, d_head = k_cache.shape[1], k_cache.shape[3]
    k_seq = k_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, d_head)
    v_seq = v_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, d_head)
    groups = q.shape[0] // num_kv_heads
    k_seq = k_seq.repeat_interleave(groups, dim=0).float()
    v_seq = v_seq.repeat_interleave(groups, dim=0).float()

    seq_q = q.shape[1]
    causal = torch.ones(seq_q, k_seq.shape[1], dtype=torch.bool).tril()
    scores = torch.bmm(q.float(), k_seq.transpose(1, 2)) * scale
    weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    out = torch.bmm(weights, v_seq).to(q.dtype)  # [Nh, S, D]
    return out.transpose(1, 2) if tp_out else out


def _run_padded_segmented_prefill(monkeypatch, segmented_attention, pad_slot: int):
    """Run forward_prefill on a 3-token prompt padded to an 8-token segment."""
    glm_model = src_module("model")

    torch.manual_seed(42)
    num_q_heads, kv_heads, head_dim, block_size = 2, 1, 8, 4
    segment, num_real = 8, 3
    q_size, kv_size = num_q_heads * head_dim, kv_heads * head_dim
    attn = _bare_attention(num_q_heads, kv_heads, head_dim, block_size)

    qkv = torch.randn(1, segment, q_size + 2 * kv_size)

    def fake_qkv_proj(**_kwargs):
        return qkv

    def fake_o_proj(attn_output, _weight, _bias):
        # [1, Nh, Dh, T] -> [1, T, Nh * Dh]
        return attn_output.permute(0, 3, 1, 2).reshape(1, segment, q_size)

    monkeypatch.setattr(glm_model.NF, "qkv_proj", fake_qkv_proj)
    monkeypatch.setattr(glm_model.NF, "o_proj", fake_o_proj)
    monkeypatch.setattr(glm_model.NF, "segmented_attention", segmented_attention)

    # The prompt has 3 tokens in block 1. The runner pads the segment of 8
    # tokens, and it marks the unused block column with -1.
    slot_mapping = torch.tensor([4, 5, 6] + [pad_slot] * (segment - num_real))
    attn_metadata = {
        "layers.0.self_attn": {
            "slot_mapping": slot_mapping,
            "block_size": block_size,
            "block_table_tensor": torch.tensor([[1, -1]], dtype=torch.int32),
            "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32),
            "kv_segment_size": segment,
        }
    }
    hidden = torch.randn(segment, 16)
    ones = torch.ones(segment, head_dim)
    out = attn.forward_prefill(
        hidden, torch.arange(segment), (ones, torch.zeros_like(ones)), attn_metadata
    )

    assert torch.isfinite(out[:num_real]).all(), out[:num_real]
    # The real rows must match causal attention over the real tokens only.
    q, k, v = torch.tensor_split(qkv[0], attn.qkv_split_indices, dim=-1)
    q = q.view(segment, num_q_heads, head_dim).transpose(0, 1)[:, :num_real]
    k = k.view(segment, kv_heads, head_dim).transpose(0, 1)[:, :num_real]
    v = v.view(segment, kv_heads, head_dim).transpose(0, 1)[:, :num_real]
    k = k.repeat_interleave(num_q_heads // kv_heads, dim=0)
    v = v.repeat_interleave(num_q_heads // kv_heads, dim=0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=attn.scaling
    )  # [Nh, L, Dh]
    expected = expected.transpose(0, 1).reshape(num_real, q_size)
    torch.testing.assert_close(out[:num_real], expected)


def test_glm_segmented_prefill_padded_prompt_stays_finite(monkeypatch):
    """A short prompt in a padded segment must give finite real rows.

    The block columns after the prompt are -1, so the device kernel reads
    stale SBUF tiles there.
    """
    _run_padded_segmented_prefill(
        monkeypatch, _skip_dma_segmented_attention, pad_slot=-1
    )


def test_glm_segmented_prefill_block_table_uses_first_block_of_row():
    """Each unused block column must point at the first block of its row.

    The null block (block 0) is not a safe fill: each pad token and the warm-up
    request write to it, so it holds data of other requests.
    """
    segmented_prefill_block_table = src_module("model")._segmented_prefill_block_table

    block_table = torch.tensor(
        [[3, 5, -1, -1], [-1, -1, -1, -1], [7, -1, -1, -1]], dtype=torch.int32
    )
    out = segmented_prefill_block_table(block_table)
    expected = torch.tensor(
        [[3, 5, 3, 3], [0, 0, 0, 0], [7, 7, 7, 7]], dtype=torch.int32
    )
    assert out.dtype == block_table.dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_glm_segmented_prefill_ignores_non_finite_null_block(monkeypatch):
    """A non-finite value in the null block must not change the real rows.

    This is the serving state: the pad tokens write to the null block, which
    then holds data that this request did not write. The test puts NaN in the
    null block and keeps the real blocks finite.
    """

    def nan_null_block_attention(q, k_cache, v_cache, block_tables, **kwargs):
        # Put NaN in the null block after the pad tokens write to it. Each
        # real block keeps its data.
        k_cache[0] = float("nan")
        v_cache[0] = float("nan")
        return _skip_dma_segmented_attention(
            q, k_cache, v_cache, block_tables, **kwargs
        )

    # The server pads the segment with slot 0 (the null block).
    _run_padded_segmented_prefill(monkeypatch, nan_null_block_attention, pad_slot=0)


# =============================================================================
# Full-model load (40 layers, TP 16) and CPU prefill logits
# =============================================================================


class _PrefillLogits(torch.nn.Module):
    """Call the GLM model with prefill inputs. Return the gathered logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, positions, attn_metadata, sampling_positions):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
        )


def _load_full_model(checkpoint_path: str, num_blocks: int, block_size: int):
    """``model_load``: build the full GLM model and load all its weights."""
    GlmForCausalLM = src_module("model").GlmForCausalLM

    device = torch.device("cpu")
    neuron_config = NeuronConfig(on_device_sampling_config=None)
    config = _glm_config_cls().from_configs(
        load_config_dict(checkpoint_path), neuron_config
    )
    model = GlmForCausalLM(config)
    model.load_weights(checkpoint_path, device, None)

    # Make sure that each parameter has real, finite data.
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"Parameter {name} is on the meta device after load.")
        if param.dtype != torch.bfloat16:
            raise RuntimeError(f"Parameter {name} has dtype {param.dtype}.")
        if not torch.isfinite(param.float()).all():
            raise RuntimeError(f"Parameter {name} has a non-finite value.")
    attn0 = model.model.layers[0].self_attn
    if attn0.qkv_proj_bias is None or float(attn0.qkv_proj_bias.abs().sum()) == 0.0:
        raise RuntimeError("The QKV bias of layer 0 did not load.")

    kv_caches = {}
    for spec in model.get_kv_spec().layers:
        shape = (num_blocks, spec.num_kv_heads, block_size, spec.head_size)
        kv_caches[spec.name] = [
            torch.zeros(shape, dtype=spec.dtype),
            torch.zeros(shape, dtype=spec.dtype),
        ]
    model.bind_kv_cache(kv_caches)
    return _PrefillLogits(model.eval())


def _compute_hf_logits(checkpoint_path: str, input_ids: torch.Tensor) -> dict:
    from transformers import AutoModelForCausalLM

    result = {}
    for key, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, dtype=dtype, attn_implementation="eager"
        ).eval()
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits[0]
        result[key] = logits.float().detach()
        del model
    return result


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("world_size", [TP])
def test_glm_full_model_load_and_prefill_logits_cpu(world_size: int):
    """Load all 40 layers at TP 16 on CPU, then compare the prefill logits with HF.

    Needs the full checkpoint (``GLM4_CHECKPOINT_PATH``), about 19 GB, and
    about 100 GB of host memory for the two HF references and the 16 ranks.
    """
    from vllm_neuron.utils.executor import MPExecutor
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    ckpt = weights_path()
    num_layers = load_config_dict(ckpt)["num_hidden_layers"]
    torch.manual_seed(42)
    input_ids = torch.randint(0, 150000, (PREFILL_LEN,), dtype=torch.int64)

    goldens = get_or_compute_goldens(
        key_config={
            "model_id": MODEL_ID,
            "variant": GOLDEN_VARIANT,
            "test": "glm_full_prefill_logits",
            "seq_len": PREFILL_LEN,
            "seed": 42,
        },
        compute_fn=lambda: _compute_hf_logits(ckpt, input_ids),
    )

    num_blocks = PREFILL_LEN // BLOCK_SIZE
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            _load_full_model,
            checkpoint_path=ckpt,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
        ),
        device="cpu",
        load_timeout=3600,
    )
    attn_meta = {
        "max_query_len": PREFILL_LEN,
        "decode_token_threshold": 1,
        "slot_mapping": torch.arange(PREFILL_LEN, dtype=torch.int64),
        "block_size": BLOCK_SIZE,
        "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0),
        "max_blocks_per_seq": num_blocks,
        "cached_seq_len": None,
        "kv_segment_size": None,
    }
    attn_metadata = {
        f"layers.{i}.self_attn": dict(attn_meta) for i in range(num_layers)
    }
    positions = torch.arange(PREFILL_LEN, dtype=torch.int64)
    sampling_positions = torch.arange(PREFILL_LEN, dtype=torch.int64)

    executor.dispatch(input_ids, positions, attn_metadata, sampling_positions)
    outputs = executor.collect(timeout=3600)

    logits = outputs[0].float()
    assert logits.shape == goldens["fp32"].shape
    for i in range(1, len(outputs)):
        torch.testing.assert_close(outputs[i].float(), logits, rtol=0, atol=0)
    assert_close_three_way(
        baseline=goldens["fp32"],
        expected=goldens["bf16"],
        actual=logits,
        name=f"glm_full_prefill_logits/tp{world_size}",
    )
