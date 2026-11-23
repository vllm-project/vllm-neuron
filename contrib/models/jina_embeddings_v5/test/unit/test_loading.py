# SPDX-License-Identifier: Apache-2.0
"""Jina Embeddings v5 text nano: loading and CPU component tests.

Run in CPU mode. The tests download the pinned revision of the checkpoint
(about 440 MB, without the task adapters) unless ``JINA_V5_CHECKPOINT`` points
at a local copy:

    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/jina_embeddings_v5/test/unit/ -v

Coverage:

- Registration contract (the architecture string of the checkpoint, the
  factory that ``src/`` exports, and the vLLM model protocol), config mapping
  and RoPE normalization.
- Rejection of the configurations that the port does not implement.
- Complete weight coverage: each checkpoint tensor has a mapping, with the
  dtype and the shape that the config gives.
- TP=2 CPU attention (layer 0) and the full 12-layer encoder against the HF
  ``EuroBertModel`` reference, in a padded bucket.
"""

import functools

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import (
    assert_close_three_way,
    assert_pooling_close_three_way,
)
from vllm_neuron.utils.executor import MPExecutor

import jina_v5_test_utils as utils
from _local_src import src_module, src_package
from jina_v5_test_utils import (
    LAYER_IDX,
    build_attn_metadata,
    get_checkpoint,
    hf_last_hidden,
    hf_layer_attention,
    load_config_dict,
    load_hf_encoder,
    padded_positions,
    padded_token_ids,
    pool_last_normalize,
    random_token_ids,
)

ARCHITECTURE = "JinaEmbeddingsV5Model"
MIN_COS = 0.99


@pytest.fixture(autouse=True)
def require_cpu_mode():
    # Fail, and do not skip, so that a wrong CI environment is visible.
    assert envs.VLLM_NEURON_CPU_MODE, (
        "VLLM_NEURON_CPU_MODE is not set. Set VLLM_NEURON_CPU_MODE=1 to run "
        "the CPU tests."
    )


@pytest.fixture(scope="module")
def checkpoint():
    return get_checkpoint()


@pytest.fixture(scope="module")
def config_dict(checkpoint):
    return load_config_dict(checkpoint)


@pytest.fixture(scope="module")
def hf_models(checkpoint):
    return {
        "fp32": load_hf_encoder(checkpoint, torch.float32),
        "bf16": load_hf_encoder(checkpoint, torch.bfloat16),
    }


# =============================================================================
# Loading
# =============================================================================


def test_registry_and_config(config_dict):
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces_base import (
        is_pooling_model,
        is_text_generation_model,
        is_vllm_model,
    )

    JinaEmbeddingsV5Model = src_package().JinaEmbeddingsV5Model
    config_module = src_module("config")
    jina_v5_config_from_hf = config_module.jina_v5_config_from_hf
    normalize_rope_parameters = config_module.normalize_rope_parameters

    # A contrib model is not in the plugin registry. The registration contract
    # is the architecture string of the checkpoint and the factory that src/
    # exports, which the user (or the plugin entry point) registers with vLLM.
    assert config_dict["architectures"] == [ARCHITECTURE]
    assert src_module("factory").JinaEmbeddingsV5Model is JinaEmbeddingsV5Model
    assert is_vllm_model(JinaEmbeddingsV5Model)
    assert is_pooling_model(JinaEmbeddingsV5Model)
    assert not is_text_generation_model(JinaEmbeddingsV5Model)
    assert JinaEmbeddingsV5Model.default_seq_pooling_type == "LAST"
    ModelRegistry.register_model(ARCHITECTURE, JinaEmbeddingsV5Model)
    assert ARCHITECTURE in ModelRegistry.get_supported_archs()

    cfg = jina_v5_config_from_hf(config_dict, None, torch.bfloat16)
    assert cfg.hidden_size == 768
    assert cfg.intermediate_size == 3072
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert cfg.num_key_value_heads == 12
    assert cfg.head_dim == 64
    assert cfg.vocab_size == 128256
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    assert cfg.rope_parameters == {"rope_type": "default", "rope_theta": 1e6}
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.bfloat16

    # The HF rotary module rewrites a null rope_scaling to linear with factor
    # 1.0. That form gives the same frequencies.
    linear = dict(config_dict, rope_scaling={"rope_type": "linear", "factor": 1.0})
    assert normalize_rope_parameters(linear)["rope_type"] == "default"
    for scaling in (
        {"rope_type": "linear", "factor": 2.0},
        {"rope_type": "yarn", "factor": 4.0},
    ):
        with pytest.raises(NotImplementedError):
            normalize_rope_parameters(dict(config_dict, rope_scaling=scaling))


def test_unsupported_configs_raise(config_dict):
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_neuron.model.neuron_config import NeuronConfig

    JinaEmbeddingsV5Model = src_package().JinaEmbeddingsV5Model

    for override in (
        {"is_decoder": True},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"hidden_act": "gelu"},
        {"quantization_config": {"quant_method": "fp8"}},
    ):
        with pytest.raises(NotImplementedError):
            JinaEmbeddingsV5Model._validate_config(
                dict(config_dict, **override), NeuronConfig()
            )

    # The model has no LM head. A non-pooling runner must fail.
    with set_current_vllm_config(VllmConfig()):
        with pytest.raises(ValueError, match="--runner pooling"):
            JinaEmbeddingsV5Model.from_configs(config_dict, NeuronConfig())


def test_checkpoint_weight_coverage(checkpoint, config_dict):
    import glob
    import json
    import os
    import struct

    checkpoint_mappings = src_module("model").checkpoint_mappings

    # Read the safetensors headers directly. This reads no tensor data.
    tensors = {}
    files = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    assert files, f"No safetensors file in {checkpoint}"
    for path in files:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for key, info in header.items():
            if key != "__metadata__":
                tensors[key] = (tuple(info["shape"]), info["dtype"])

    num_layers = config_dict["num_hidden_layers"]
    referenced = set()
    for value in checkpoint_mappings(num_layers, prefix="").values():
        referenced.update(value if isinstance(value, list) else [value])
    assert set(tensors) == referenced, (
        f"Unmapped checkpoint tensors: {sorted(set(tensors) - referenced)[:5]}; "
        f"missing checkpoint tensors: {sorted(referenced - set(tensors))[:5]}"
    )

    h = config_dict["hidden_size"]
    inter = config_dict["intermediate_size"]
    head_dim = h // config_dict["num_attention_heads"]
    kv = config_dict["num_key_value_heads"] * head_dim
    expected = {
        "embed_tokens.weight": (config_dict["vocab_size"], h),
        "norm.weight": (h,),
    }
    for i in range(num_layers):
        expected.update(
            {
                f"layers.{i}.self_attn.q_proj.weight": (h, h),
                f"layers.{i}.self_attn.k_proj.weight": (kv, h),
                f"layers.{i}.self_attn.v_proj.weight": (kv, h),
                f"layers.{i}.self_attn.o_proj.weight": (h, h),
                f"layers.{i}.input_layernorm.weight": (h,),
                f"layers.{i}.post_attention_layernorm.weight": (h,),
                f"layers.{i}.mlp.gate_proj.weight": (inter, h),
                f"layers.{i}.mlp.up_proj.weight": (inter, h),
                f"layers.{i}.mlp.down_proj.weight": (h, inter),
            }
        )
    assert {k: v[0] for k, v in tensors.items()} == expected
    dtypes = {v[1] for v in tensors.values()}
    assert dtypes == {"BF16"}, f"Checkpoint dtypes are {dtypes}, not BF16 only."


# =============================================================================
# CPU components (TP=2, padded bucket)
# =============================================================================

CPU_ATTN_BUCKET = 128
CPU_ATTN_REAL_LEN = 100
CPU_MODEL_BUCKET = 256
CPU_MODEL_REAL_LENS = (200, 37)


@pytest.mark.parametrize("world_size", [2])
def test_cpu_attention_matches_hf(world_size, checkpoint, config_dict, hf_models):
    torch.manual_seed(42)
    hidden = config_dict["hidden_size"]
    positions = padded_positions(CPU_ATTN_REAL_LEN, CPU_ATTN_BUCKET)
    metadata = build_attn_metadata(
        config_dict["num_hidden_layers"], CPU_ATTN_BUCKET, CPU_ATTN_BUCKET
    )

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            utils.load_attention, checkpoint, config_dict, "cpu"
        ),
        device="cpu",
    )

    baseline, expected, actual = [], [], []
    for sample in range(2):
        torch.manual_seed(sample)
        h = torch.randn(CPU_ATTN_BUCKET, hidden, dtype=torch.bfloat16)
        executor.dispatch(h, positions, metadata)
        out = torch.cat(executor.collect(timeout=600), dim=0)
        real = h[:CPU_ATTN_REAL_LEN]
        baseline.append(hf_layer_attention(hf_models["fp32"], real.float(), LAYER_IDX))
        expected.append(hf_layer_attention(hf_models["bf16"], real, LAYER_IDX))
        actual.append(out[:CPU_ATTN_REAL_LEN])

    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"jina_v5/cpu_attention/tp{world_size}",
    )


@pytest.mark.parametrize("world_size", [2])
def test_cpu_full_model_matches_hf(world_size, checkpoint, config_dict, hf_models):
    torch.manual_seed(42)
    metadata = build_attn_metadata(
        config_dict["num_hidden_layers"], CPU_MODEL_BUCKET, CPU_MODEL_BUCKET
    )
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            utils.load_encoder, checkpoint, config_dict, "cpu"
        ),
        device="cpu",
    )

    baseline, expected, actual = [], [], []
    pooled = {"fp32": [], "bf16": [], "port": []}
    for seed, real_len in enumerate(CPU_MODEL_REAL_LENS):
        token_ids = random_token_ids(seed, real_len)
        executor.dispatch(
            padded_token_ids(token_ids, CPU_MODEL_BUCKET),
            padded_positions(real_len, CPU_MODEL_BUCKET),
            metadata,
        )
        # The encoder all-gathers the sequence, so each rank has [T, H].
        out = executor.collect(timeout=1200)[0]
        ref32 = hf_last_hidden(hf_models["fp32"], token_ids)
        ref16 = hf_last_hidden(hf_models["bf16"], token_ids)
        baseline.append(ref32)
        expected.append(ref16)
        actual.append(out[:real_len])
        pooled["fp32"].append(pool_last_normalize(ref32, real_len))
        pooled["bf16"].append(pool_last_normalize(ref16, real_len))
        pooled["port"].append(pool_last_normalize(out, real_len))

    assert_close_three_way(
        baseline=baseline,
        expected=expected,
        actual=actual,
        name=f"jina_v5/cpu_full_model/tp{world_size}",
    )
    assert_pooling_close_three_way(
        baseline=torch.stack(pooled["fp32"]),
        expected=torch.stack(pooled["bf16"]),
        actual=torch.stack(pooled["port"]),
        task="embed",
        name=f"jina_v5/cpu_pooled/tp{world_size}",
        min_cos=MIN_COS,
    )
