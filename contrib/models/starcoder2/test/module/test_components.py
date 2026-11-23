# SPDX-License-Identifier: Apache-2.0
"""StarCoder2 module accuracy tests at TP 4, on one node.

Each test compares three results (``assert_close_three_way``):

- baseline: the HF module in FP32, with the original FP32 checkpoint weights
- expected: the HF module in BF16
- actual:   the vLLM Neuron module in BF16, sharded across 4 TP ranks

The HF modules come from ``transformers`` and load the layer tensors directly
from the checkpoint. They do not use code from ``src/``.

The tests need the checkpoint weights: set ``STARCODER2_CHECKPOINT_PATH`` to a
local copy of the pinned revision, or every test skips.

- CPU mode (``VLLM_NEURON_CPU_MODE=1``): the 4 ranks run on CPU.
  ``test_attention_prefill`` and ``test_mlp_prefill`` run; the two decode
  cases skip.
- Device mode: the 4 ranks run on 4 NeuronCores, and the modules go through
  ``torch.compile``.

Run in CPU mode:

    VLLM_NEURON_CPU_MODE=1 STARCODER2_CHECKPOINT_PATH=/path/to/starcoder2-3b \\
        pytest -v contrib/models/starcoder2/test/module/test_components.py \\
        -k "test_attention_prefill or test_mlp_prefill"
"""

import copy
from functools import partial

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.model.neuron_config import NeuronConfig

from starcoder2_test_utils import (
    TP_SIZE,
    read_checkpoint_tensors,
    require_checkpoint,
    src_module,
)

ATTN_LAYER_IDX = 0
# Use a layer in the middle of the stack for the MLP.
MLP_LAYER_IDX = 15
PREFILL_LEN = 1024
DECODE_TOKENS = 4
BLOCK_SIZE = 32
# The decode kernel needs a block count that is a multiple of LNC x 128.
BLOCKS_ALIGNMENT = 2 * 128
N_SAMPLES = 3
COLLECT_TIMEOUT_S = 3600


def _device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def _make_executor(world_size: int, model_load):
    from vllm_neuron.utils.executor import MPExecutor

    return MPExecutor(
        world_size=world_size,
        model_load=model_load,
        device=_device(),
        load_timeout=1800,
    )


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)


@pytest.fixture(scope="module")
def checkpoint():
    return require_checkpoint()


@pytest.fixture(scope="module")
def hf_config(checkpoint):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint)
    config._attn_implementation = "eager"
    return config


def _num_blocks(total_tokens: int) -> int:
    min_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    return ((min_blocks + BLOCKS_ALIGNMENT - 1) // BLOCKS_ALIGNMENT) * BLOCKS_ALIGNMENT


# =============================================================================
# HF references
# =============================================================================


def _hf_modules(checkpoint, hf_config, kind: str, layer_idx: int):
    """Return the HF module of one layer in FP32 and in BF16."""
    from transformers.models.starcoder2 import modeling_starcoder2 as hf

    prefix = f"model.layers.{layer_idx}." + ("self_attn" if kind == "attn" else "mlp")
    if kind == "attn":
        module = hf.Starcoder2Attention(hf_config, layer_idx=layer_idx)
        names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        module = hf.Starcoder2MLP(hf_config)
        names = ["c_fc", "c_proj"]
    keys = [f"{prefix}.{n}.{p}" for n in names for p in ("weight", "bias")]
    tensors = read_checkpoint_tensors(checkpoint, keys)
    state = {key[len(prefix) + 1 :]: tensor for key, tensor in tensors.items()}
    module.load_state_dict(state, strict=True)
    module_fp32 = module.float().eval()
    module_bf16 = copy.deepcopy(module_fp32).to(torch.bfloat16).eval()
    return module_fp32, module_bf16


def _hf_rope(hf_config, positions: torch.Tensor, dtype: torch.dtype):
    from transformers.models.starcoder2 import modeling_starcoder2 as hf

    rotary = hf.Starcoder2RotaryEmbedding(config=hf_config)
    dummy = torch.zeros(1, positions.shape[0], hf_config.hidden_size, dtype=dtype)
    return rotary(dummy, positions.unsqueeze(0))


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(torch.float32).min),
        diagonal=1,
    )
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


def _hf_attention(checkpoint, hf_config, prompt, decode):
    """Run HF attention on the prompt, then on one decode token.

    Returns:
        Dict with the prefill outputs and the decode outputs in FP32 and BF16.
    """
    from transformers import DynamicCache

    attn_fp32, attn_bf16 = _hf_modules(checkpoint, hf_config, "attn", ATTN_LAYER_IDX)
    seq_len = prompt.shape[0]
    positions = torch.arange(seq_len + 1, dtype=torch.int64)
    result = {}
    for name, attn, dtype in (
        ("fp32", attn_fp32, torch.float32),
        ("bf16", attn_bf16, torch.bfloat16),
    ):
        cos, sin = _hf_rope(hf_config, positions, dtype)
        cache = DynamicCache()
        with torch.no_grad():
            out, _ = attn(
                hidden_states=prompt.to(dtype).unsqueeze(0),
                position_embeddings=(cos[:, :seq_len], sin[:, :seq_len]),
                attention_mask=_causal_mask(seq_len, dtype),
                past_key_values=cache,
                cache_position=positions[:seq_len],
            )
            result[f"prefill_{name}"] = out[0]
            if decode is not None:
                out, _ = attn(
                    hidden_states=decode.to(dtype).view(1, 1, -1),
                    position_embeddings=(cos[:, seq_len:], sin[:, seq_len:]),
                    attention_mask=None,
                    past_key_values=cache,
                    cache_position=positions[seq_len:],
                )
                result[f"decode_{name}"] = out[0]
    return result


# =============================================================================
# vLLM Neuron modules (built in each worker)
# =============================================================================


def _cast_bf16(state_dict: dict) -> dict:
    return {name: tensor.to(torch.bfloat16) for name, tensor in state_dict.items()}


def _load_shards(checkpoint: str, module, mappings: dict, device: str) -> None:
    import torch.distributed as dist

    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    loader = SafetensorsCheckpoint(checkpoint, None)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if envs.VLLM_NEURON_CPU_MODE:
        shards = loader.load_sharded(rank, world_size, module, mappings, device)
    else:
        shards = loader.load_sharded_pipelined(
            rank, world_size, module, mappings, device
        )
    module.load_state_dict(_cast_bf16(shards.state_dict), strict=True, assign=True)


def _compile(module):
    if envs.VLLM_NEURON_CPU_MODE:
        return module
    return torch.compile(
        module, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def _load_attention(checkpoint, hf_config, num_blocks, prefill_len, device):
    import torch.distributed as dist

    model = src_module("model")
    config = src_module("config").Starcoder2Config.from_configs(
        hf_config, NeuronConfig()
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    attn = model.Starcoder2Attention(config, layer_idx=ATTN_LAYER_IDX)
    full = model.Starcoder2ForCausalLM.checkpoint_mappings(ATTN_LAYER_IDX + 1, False)
    prefix = f"model.layers.{ATTN_LAYER_IDX}.self_attn."
    mappings = {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}
    _load_shards(checkpoint, attn, mappings, device)

    cache_shape = (num_blocks, attn.num_key_value_heads_per_rank, BLOCK_SIZE, 128)
    attn.k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    attn.v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    original_forward = attn.forward
    per_rank = prefill_len // world_size

    def forward(hidden_states, positions, position_embeddings, attn_metadata, phase):
        if phase == "prefill":
            # >>> SP: each rank gets its part of the prompt.
            start = rank * per_rank
            hidden_states = hidden_states[start : start + per_rank, :]
        return original_forward(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    attn.forward = forward
    return _compile(attn)


def _load_mlp(checkpoint, hf_config, num_tokens, phase, device):
    import torch.distributed as dist

    model = src_module("model")
    config = src_module("config").Starcoder2Config.from_configs(
        hf_config, NeuronConfig()
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mlp = model.Starcoder2MLP(config)
    prefix = f"model.layers.{MLP_LAYER_IDX}.mlp"
    mappings = {
        f"{proj}_{kind}": f"{prefix}.{proj}.{kind}"
        for proj in ("c_fc", "c_proj")
        for kind in ("weight", "bias")
    }
    _load_shards(checkpoint, mlp, mappings, device)

    original_forward = mlp.forward
    per_rank = num_tokens // world_size

    def forward(hidden_states):
        if phase == "prefill":
            start = rank * per_rank
            return original_forward(
                hidden_states[start : start + per_rank, :], is_prefill=True
            )
        return original_forward(hidden_states, is_prefill=False)

    mlp.forward = forward
    return _compile(mlp)


def _attn_metadata(num_blocks: int, slot_mapping: torch.Tensor, max_query_len: int):
    return {
        f"layers.{ATTN_LAYER_IDX}.self_attn": {
            "max_query_len": max_query_len,
            "decode_token_threshold": 1,
            "slot_mapping": slot_mapping,
            "block_size": BLOCK_SIZE,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32).view(
                1, -1
            ),
            "max_blocks_per_seq": num_blocks,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def _neuron_rope(hf_config, positions: torch.Tensor):
    config = src_module("config").Starcoder2Config.from_configs(
        hf_config, NeuronConfig()
    )
    return src_module("model").Starcoder2RotaryEmbedding(config)(
        positions, device=torch.device("cpu"), dtype=torch.bfloat16
    )


def _run_attention(executor, hf_config, prompt, decode, num_blocks):
    """Run the prefill and then (optionally) one decode step on the workers."""
    seq_len = prompt.shape[0]
    positions = torch.arange(seq_len, dtype=torch.int64)
    executor.dispatch(
        prompt,
        positions,
        _neuron_rope(hf_config, positions),
        _attn_metadata(num_blocks, torch.arange(seq_len, dtype=torch.int64), seq_len),
        "prefill",
    )
    prefill_out = executor.collect(timeout=COLLECT_TIMEOUT_S)
    if decode is None:
        return prefill_out, None

    decode_positions = torch.tensor([seq_len], dtype=torch.int64)
    executor.dispatch(
        decode.view(1, -1),
        decode_positions,
        _neuron_rope(hf_config, decode_positions),
        _attn_metadata(num_blocks, torch.tensor([seq_len], dtype=torch.int64), 1),
        "decode",
    )
    decode_out = executor.collect(timeout=COLLECT_TIMEOUT_S)
    return prefill_out, decode_out


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_attention_prefill(world_size, checkpoint, hf_config):
    """Prefill attention (fused QKV bias, RoPE, GQA, O bias, SP collectives)."""
    num_blocks = _num_blocks(PREFILL_LEN)
    executor = _make_executor(
        world_size,
        partial(
            _load_attention, checkpoint, hf_config, num_blocks, PREFILL_LEN, _device()
        ),
    )
    for sample in range(N_SAMPLES):
        torch.manual_seed(sample)
        prompt = torch.randn(PREFILL_LEN, hf_config.hidden_size, dtype=torch.bfloat16)
        golden = _hf_attention(checkpoint, hf_config, prompt, decode=None)
        prefill_out, _ = _run_attention(executor, hf_config, prompt, None, num_blocks)
        assert_close_three_way(
            baseline=golden["prefill_fp32"],
            expected=golden["prefill_bf16"],
            actual=torch.cat(prefill_out, dim=0),
            name=f"attn_prefill/s{sample}",
        )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_attention_prefill_decode(world_size, checkpoint, hf_config):
    """Prefill writes the paged KV cache. Then the decode kernel reads it.

    The test checks the prefill output and the decode output. A KV layout
    mismatch between the prefill path and the decode kernel shows as a decode
    error only. The case runs on NeuronCores, where the decode step is the
    fused decode kernel. The port ran it only there.
    """
    if envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("Device-only case: the decode step runs the decode kernel.")
    num_blocks = _num_blocks(PREFILL_LEN + 1)
    executor = _make_executor(
        world_size,
        partial(
            _load_attention, checkpoint, hf_config, num_blocks, PREFILL_LEN, _device()
        ),
    )
    for sample in range(N_SAMPLES):
        torch.manual_seed(sample)
        prompt = torch.randn(PREFILL_LEN, hf_config.hidden_size, dtype=torch.bfloat16)
        decode = torch.randn(hf_config.hidden_size, dtype=torch.bfloat16)
        golden = _hf_attention(checkpoint, hf_config, prompt, decode=decode)
        prefill_out, decode_out = _run_attention(
            executor, hf_config, prompt, decode, num_blocks
        )
        assert_close_three_way(
            baseline=golden["prefill_fp32"],
            expected=golden["prefill_bf16"],
            actual=torch.cat(prefill_out, dim=0),
            name=f"attn_prefill/s{sample}",
        )
        # After the all-reduce, each rank holds the full decode output.
        for rank in range(1, len(decode_out)):
            torch.testing.assert_close(decode_out[rank], decode_out[0])
        assert_close_three_way(
            baseline=golden["decode_fp32"],
            expected=golden["decode_bf16"],
            actual=decode_out[0].view(1, -1),
            name=f"attn_decode/s{sample}",
        )


def _run_mlp_case(world_size, checkpoint, hf_config, num_tokens: int, phase: str):
    mlp_fp32, mlp_bf16 = _hf_modules(checkpoint, hf_config, "mlp", MLP_LAYER_IDX)
    executor = _make_executor(
        world_size,
        partial(_load_mlp, checkpoint, hf_config, num_tokens, phase, _device()),
    )
    for sample in range(N_SAMPLES):
        torch.manual_seed(sample)
        h = (torch.rand(num_tokens, hf_config.hidden_size) * 2 - 1).to(torch.bfloat16)
        with torch.no_grad():
            baseline = mlp_fp32(h.float())
            expected = mlp_bf16(h)
        executor.dispatch(h)
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)
        if phase == "prefill":
            actual = torch.cat(outputs, dim=0)
        else:
            for rank in range(1, len(outputs)):
                torch.testing.assert_close(outputs[rank], outputs[0])
            actual = outputs[0]
        assert_close_three_way(
            baseline=baseline,
            expected=expected,
            actual=actual,
            name=f"mlp_{phase}/s{sample}",
        )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_mlp_prefill(world_size, checkpoint, hf_config):
    """Prefill MLP (c_fc bias, GELU tanh, c_proj bias, SP collectives)."""
    _run_mlp_case(world_size, checkpoint, hf_config, PREFILL_LEN, "prefill")


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("world_size", [TP_SIZE])
def test_mlp_decode(world_size, checkpoint, hf_config):
    """Decode MLP for 4 tokens with the TP all-reduce (device-only case)."""
    if envs.VLLM_NEURON_CPU_MODE:
        pytest.skip("Device-only case: the port ran it on NeuronCores only.")
    _run_mlp_case(world_size, checkpoint, hf_config, DECODE_TOKENS, "decode")
