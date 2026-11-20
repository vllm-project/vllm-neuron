# SPDX-License-Identifier: Apache-2.0
"""HRM-Text loading checks and CPU component checks.

Run these tests with ``VLLM_NEURON_CPU_MODE=1``. They need the checkpoint
(``HRM_TEXT_MODEL_PATH`` or the HF model ID) and no Neuron device.

They load this folder's ``src/`` through ``test/_local_src.py``, so they do not
need the repository root on ``sys.path``.

- ``test_registry_and_config``: registration, config parsing, KV slot plan
  and config validation.
- ``test_full_checkpoint_load_tp2``: load every checkpoint tensor into the full
  model at TP 2. Compare sampled shards with an independent split of the raw
  tensors.
- ``test_decoder_layer_prefill_tp2``: one decoder layer with SP prefill at
  TP 2 against independent reference math.
"""

import functools

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor


def _load_local_src():
    """Load this folder's ``test/_local_src.py`` by file location.

    That helper loads this folder's ``src/`` (and the shared
    ``test/module/common.py``) under module names unique to this contrib model.
    Loading the helper itself by location, under its own unique name, keeps
    every name out of ``sys.path``: two contrib models collected in one pytest
    run cannot resolve to each other's code, and nothing needs the repository
    root.
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "contrib_hrm_text_local_src"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "_local_src.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_local_src = _load_local_src()
_common = _local_src.load_common()

ATTN_LAYER_PREFIX = _common.ATTN_LAYER_PREFIX
BLOCK_SIZE = _common.BLOCK_SIZE
HIDDEN = _common.HIDDEN
SAMPLE_GLOBAL_PARAMS = _common.SAMPLE_GLOBAL_PARAMS
SAMPLE_LAYER_PREFIXES = _common.SAMPLE_LAYER_PREFIXES
VOCAB = _common.VOCAB
expected_rank_shards = _common.expected_rank_shards
layer_raw_names = _common.layer_raw_names
load_decoder_layer_harness = _common.load_decoder_layer_harness
load_full_model_report = _common.load_full_model_report
load_raw_tensors = _common.load_raw_tensors
read_raw_config = _common.read_raw_config
ref_decoder_layer = _common.ref_decoder_layer
resolve_checkpoint = _common.resolve_checkpoint
sample_param_names = _common.sample_param_names

pytestmark = pytest.mark.skipif(
    not envs.VLLM_NEURON_CPU_MODE,
    reason="HRM-Text CPU checks need VLLM_NEURON_CPU_MODE=1.",
)

COLLECT_TIMEOUT_S = 1800.0
LOAD_TIMEOUT_S = 1800.0


def test_registry_and_config(monkeypatch):
    """Check the registration, the config parsing and the KV slot plan."""
    from transformers import AutoConfig

    from vllm import ModelRegistry

    from vllm_neuron.model.neuron_config import NeuronConfig

    Factory = _local_src.load().HrmTextForCausalLM
    HrmTextConfig = _local_src.load("config").HrmTextConfig

    # A contrib model is not in ``vllm_neuron.model.registry``. The user
    # registers the architecture with vLLM (see contrib/README.md, "How do I
    # register a custom model with vLLM?"), so this test registers it the same
    # way and checks that vLLM holds it under the architecture string of the
    # checkpoint. The copy of ``ModelRegistry.models`` keeps the registry of
    # the pytest session unchanged.
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.register_model("HrmTextForCausalLM", Factory)
    assert "HrmTextForCausalLM" in ModelRegistry.models

    checkpoint = resolve_checkpoint()
    raw = read_raw_config(checkpoint)
    assert raw["architectures"] == ["HrmTextForCausalLM"]

    from_raw = HrmTextConfig.from_configs(raw, NeuronConfig())
    from_hf = HrmTextConfig.from_configs(
        AutoConfig.from_pretrained(checkpoint), NeuronConfig()
    )
    for cfg in (from_raw, from_hf):
        assert cfg.num_layers_per_stack == raw["num_hidden_layers"] == 16
        assert cfg.H_cycles == raw["H_cycles"]
        assert cfg.L_cycles == raw["L_cycles"]
        assert cfg.hidden_size == raw["hidden_size"]
        assert cfg.intermediate_size == raw["intermediate_size"]
        assert cfg.num_attention_heads == raw["num_attention_heads"]
        assert cfg.num_key_value_heads == raw["num_key_value_heads"]
        assert cfg.vocab_size == raw["vocab_size"]
        assert cfg.embedding_scale == pytest.approx(raw["embedding_scale"])
        assert cfg.rms_norm_eps == pytest.approx(raw["rms_norm_eps"])
        assert cfg.torch_dtype == torch.bfloat16
        assert cfg.num_kv_slots == 16 * raw["H_cycles"] * (raw["L_cycles"] + 1)

    # The HF reference gives the pass (h, l) the slot offset
    # (h * (L_cycles + 1) + l) * num_layers. l == L_cycles is the H pass.
    cfg = from_raw
    offsets = [
        cfg.cycle_offset(hc, lc)
        for hc in range(cfg.H_cycles)
        for lc in range(cfg.L_cycles + 1)
    ]
    expected = [
        (hc * (raw["L_cycles"] + 1) + lc) * 16
        for hc in range(raw["H_cycles"])
        for lc in range(raw["L_cycles"] + 1)
    ]
    assert offsets == expected
    assert sorted(offsets) == list(range(0, cfg.num_kv_slots, 16))

    with pytest.raises(ValueError, match="bfloat16"):
        Factory._validate_config({**raw, "torch_dtype": "float16"}, NeuronConfig())
    with pytest.raises(ValueError, match="bias"):
        Factory._validate_config({**raw, "attention_bias": True}, NeuronConfig())


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
def test_full_checkpoint_load_tp2(world_size):
    """Load the full checkpoint at TP 2 and compare sampled shards."""
    checkpoint = resolve_checkpoint()
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_full_model_report, checkpoint_path=checkpoint, device="cpu"
        ),
        device="cpu",
        load_timeout=LOAD_TIMEOUT_S,
    )
    executor.dispatch(torch.zeros(1))
    outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
    assert len(outputs) == world_size

    names = sample_param_names()
    raw_names = [n for p in SAMPLE_LAYER_PREFIXES for n in layer_raw_names(p)]
    raw = load_raw_tensors(checkpoint, raw_names + list(SAMPLE_GLOBAL_PARAMS))
    rows = VOCAB // world_size

    for rank, report in enumerate(outputs):
        counters = report[0].tolist()
        (
            not_loaded,
            meta,
            nonfinite,
            shape_mismatch,
            unused,
            kv_layers,
            kv_unique,
            kv_bound,
            _num_params,
        ) = counters
        assert not_loaded == 0, f"rank {rank}: {not_loaded} parameters not loaded"
        assert meta == 0, f"rank {rank}: {meta} tensors on meta"
        assert nonfinite == 0, f"rank {rank}: {nonfinite} non-finite parameters"
        assert shape_mismatch == 0, f"rank {rank}: {shape_mismatch} shape mismatches"
        assert unused == 0, f"rank {rank}: {unused} unused checkpoint tensors"
        assert kv_layers == kv_unique == kv_bound == 128

        expected = {}
        for prefix in SAMPLE_LAYER_PREFIXES:
            expected.update(expected_rank_shards(raw, prefix, rank, world_size))
        expected["model.embed_tokens.weight"] = raw["model.embed_tokens.weight"][
            rank * rows : (rank + 1) * rows
        ]
        expected["lm_head.weight"] = raw["lm_head.weight"][
            rank * rows : (rank + 1) * rows
        ]
        expected["model.z_L_init"] = raw["model.z_L_init"]

        for name, actual in zip(names, report[1:]):
            want = expected[name].to(torch.bfloat16)
            assert tuple(actual.shape) == tuple(want.shape), name
            assert torch.equal(actual.to(torch.bfloat16), want), (
                f"rank {rank}: {name} differs from the raw checkpoint shard"
            )


@pytest.mark.parametrize("world_size", [2], ids=["tp2"])
def test_decoder_layer_prefill_tp2(world_size):
    """Run one decoder layer with SP prefill and compare with reference math."""
    torch.manual_seed(42)
    checkpoint = resolve_checkpoint()
    tokens = 128
    num_blocks = tokens // BLOCK_SIZE
    raw = load_raw_tensors(checkpoint, layer_raw_names(ATTN_LAYER_PREFIX))

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_decoder_layer_harness,
            checkpoint_path=checkpoint,
            num_blocks=num_blocks,
            device="cpu",
        ),
        device="cpu",
        load_timeout=LOAD_TIMEOUT_S,
    )

    positions = torch.arange(tokens, dtype=torch.int32)
    slot_mapping = torch.arange(tokens, dtype=torch.int64)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    baselines, expecteds, actuals = [], [], []
    for _ in range(2):
        x = (torch.randn(tokens, HIDDEN) * 0.5).to(torch.bfloat16)
        executor.dispatch(x, positions, slot_mapping, block_table)
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        actuals.append(torch.cat(outputs, dim=0).float())
        with torch.no_grad():
            baselines.append(
                ref_decoder_layer(
                    x.float(), {k: v.float() for k, v in raw.items()}, ATTN_LAYER_PREFIX
                )
            )
            expecteds.append(ref_decoder_layer(x, raw, ATTN_LAYER_PREFIX).float())

    assert_close_three_way(
        baselines, expecteds, actuals, name="hrm_text_decoder_layer_prefill_cpu"
    )
