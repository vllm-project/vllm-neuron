# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Nanbeige4.2-3B tests.

The checkpoint is a looped Llama-style decoder. It runs the 22 physical decoder
layers 2 times, with one KV cache for each (loop, layer) pair, and it applies
the final RMSNorm after each loop.

The independent reference in this file uses the upstream ``transformers``
Llama modules. The checkpoint keys use the Llama names, so
``LlamaForCausalLM`` loads every original weight without a name map. The
reference then runs the layer stack ``num_loops`` times. The checkpoint remote
code is a second, optional cross-check (see ``unit/test_nanbeige_cpu.py``).

The test reads the checkpoint path from ``NANBEIGE4_CHECKPOINT_PATH``. The
helpers make sure that the path holds the pinned revision.

This file is the port's own test helper, moved into the contrib folder with the
model. The multi-node server orchestration of the original (the part that drove
the two-node device test) is not part of this copy, because that test does not
ship here; see the model README's Exceptions section.

The unit tests and the module tests both use this helper, so it sits at the
root of the model's test tree instead of next to either of them.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# This helper reads the modeling code that ships in this contrib folder
# (``contrib/models/nanbeige/src/``). It loads that code by file location
# through the local loader, so the folder keeps working when it is copied out
# of the repository. See ``_local_src.py`` next to this file.
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "contrib_nanbeige_local_src", Path(__file__).resolve().parent / "_local_src.py"
)
_local_src = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_local_src)

# Copied from ``test/vllm_neuron/utils/test_prompts.py`` (``PROMPTS_2``), the
# shared prompt list that the port's tests used. contrib/README.md sanctions
# copying a shared test helper into the model folder.
PROMPTS_2 = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

logger = logging.getLogger(__name__)

MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
HF_REVISION = "b82e54bd609793562a75cbf9337970a93369eab5"
CHECKPOINT_ENV = "NANBEIGE4_CHECKPOINT_PATH"

# Modality coverage: the checkpoint is a text-only causal LM. It has no
# vision, audio or other encoder, so the tests cover the text modality only.
SUPPORTED_MODALITIES = ("text",)

NUM_PHYSICAL_LAYERS = 22
NUM_LOOPS = 2
VOCAB_SIZE = 166144
EXPECTED_NUM_CHECKPOINT_KEYS = 201

# Identifier of the reference implementation. It is part of the golden cache
# key, so a change of the reference cannot reuse an old golden.
REFERENCE_IMPL = "transformers_llama_loop_v1"

# The fixed values of the pinned revision. The revision check compares the
# checkpoint config with these values.
EXPECTED_CONFIG = {
    "architectures": ["NanbeigeForCausalLM"],
    "model_type": "nanbeige",
    "hidden_size": 3072,
    "intermediate_size": 10752,
    "num_hidden_layers": NUM_PHYSICAL_LAYERS,
    "num_loops": NUM_LOOPS,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": VOCAB_SIZE,
    "rms_norm_eps": 1e-05,
    "rope_theta": 70000000,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "skip_loop_final_norm": False,
    "torch_dtype": "bfloat16",
}


# =============================================================================
# Checkpoint resolution
# =============================================================================


def get_checkpoint_path() -> str:
    """Return the local checkpoint directory of the pinned revision.

    Raise an error if the environment variable is not set, or if the
    directory does not hold the pinned revision. The test does not download
    the checkpoint.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        raise RuntimeError(
            f"{CHECKPOINT_ENV} is not set. Set it to the local directory of "
            f"{MODEL_ID} at revision {HF_REVISION}."
        )
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(
            f"{path} has no config.json. Set {CHECKPOINT_ENV} to a directory "
            f"that holds the {MODEL_ID} checkpoint."
        )
    if HF_REVISION not in os.path.abspath(path):
        raise RuntimeError(
            f"Checkpoint path {path} does not name revision {HF_REVISION}. "
            f"Use the directory of the pinned revision."
        )
    config = read_raw_config(path)
    mismatched = {
        key: (config.get(key), value)
        for key, value in EXPECTED_CONFIG.items()
        if config.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            f"Checkpoint config at {path} does not match revision "
            f"{HF_REVISION}: {mismatched} (found, expected). Use the pinned "
            f"revision."
        )
    return path


def read_raw_config(checkpoint_path: str) -> dict:
    """Return the raw ``config.json`` of the checkpoint as a dict."""
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        return json.load(f)


def checkpoint_weight_map(checkpoint_path: str) -> dict[str, str]:
    """Return the map from checkpoint key to safetensors file name."""
    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load_checkpoint_tensors(checkpoint_path: str, keys: list[str]) -> dict:
    """Read the given checkpoint tensors on CPU, in their stored dtype."""
    from safetensors import safe_open

    weight_map = checkpoint_weight_map(checkpoint_path)
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for file_name, file_keys in by_file.items():
        file_path = os.path.join(checkpoint_path, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_tokenizer(checkpoint_path: str):
    """Load the checkpoint tokenizer. It adds the BOS token to each prompt."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# Independent reference: upstream transformers Llama modules
# =============================================================================


def build_llama_config(raw_config: dict, dtype: torch.dtype):
    """Build a ``LlamaConfig`` with the geometry of the physical layer stack."""
    from transformers import LlamaConfig

    config = LlamaConfig(
        vocab_size=raw_config["vocab_size"],
        hidden_size=raw_config["hidden_size"],
        intermediate_size=raw_config["intermediate_size"],
        num_hidden_layers=raw_config["num_hidden_layers"],
        num_attention_heads=raw_config["num_attention_heads"],
        num_key_value_heads=raw_config["num_key_value_heads"],
        head_dim=raw_config["head_dim"],
        hidden_act=raw_config["hidden_act"],
        max_position_embeddings=raw_config["max_position_embeddings"],
        rms_norm_eps=raw_config["rms_norm_eps"],
        tie_word_embeddings=raw_config["tie_word_embeddings"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(raw_config["rope_theta"]),
        },
        attention_bias=False,
        mlp_bias=False,
        pad_token_id=raw_config.get("pad_token_id"),
        bos_token_id=raw_config.get("bos_token_id"),
        eos_token_id=raw_config.get("eos_token_id"),
    )
    config._attn_implementation = "eager"
    return config


class RefKVCache:
    """KV cache of one (loop, layer) pair for the reference.

    ``LlamaAttention`` calls only ``update()``. A separate object for each
    pair gives each loop its own KV slots, as in the Nanbeige model.
    """

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
        if self.keys is None:
            self.keys, self.values = key_states, value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[-2]


def causal_additive_mask(
    past_len: int, query_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Return a [1, 1, query_len, past_len + query_len] additive causal mask."""
    total = past_len + query_len
    query_pos = torch.arange(past_len, total).unsqueeze(1)
    key_pos = torch.arange(total).unsqueeze(0)
    mask = torch.zeros(query_len, total, dtype=dtype)
    mask.masked_fill_(key_pos > query_pos, torch.finfo(dtype).min)
    return mask[None, None]


class NanbeigeLoopReference(nn.Module):
    """Looped reference built on an upstream ``LlamaForCausalLM``.

    The forward pass runs the physical layers ``num_loops`` times and applies
    the final norm after each loop, unless ``skip_loop_final_norm`` is True.
    The interface matches the calls of
    ``vllm_neuron.accuracy.goldens.reference_logits``.
    """

    def __init__(self, llama, num_loops: int, skip_loop_final_norm: bool):
        super().__init__()
        self.llama = llama
        self.config = llama.config
        self.num_loops = num_loops
        self.skip_loop_final_norm = skip_loop_final_norm

    def new_cache(self) -> list[list[RefKVCache]]:
        layers = len(self.llama.model.layers)
        return [[RefKVCache() for _ in range(layers)] for _ in range(self.num_loops)]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        model = self.llama.model
        if past_key_values is None:
            past_key_values = self.new_cache()
        batch, query_len = input_ids.shape
        past_len = past_key_values[0][0].get_seq_length()
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + query_len)
        position_ids = cache_position.view(1, -1).expand(batch, -1)

        hidden_states = model.embed_tokens(input_ids)
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        mask = causal_additive_mask(past_len, query_len, hidden_states.dtype)

        for loop_idx in range(self.num_loops):
            for layer_idx, layer in enumerate(model.layers):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values[loop_idx][layer_idx],
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
            if not self.skip_loop_final_norm:
                hidden_states = model.norm(hidden_states)
        if self.skip_loop_final_norm:
            hidden_states = model.norm(hidden_states)

        logits = self.llama.lm_head(hidden_states)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def load_loop_reference(
    model_checkpoint: str, dtype: torch.dtype, config=None
) -> NanbeigeLoopReference:
    """Load the looped Llama reference with every original weight.

    This function has the ``model_loader`` signature of
    ``generate_three_way_reference_logits``. It ignores ``config``, because
    the reference reads the raw checkpoint config.
    """
    from transformers import LlamaForCausalLM

    raw_config = read_raw_config(model_checkpoint)
    llama_config = build_llama_config(raw_config, dtype)
    llama, info = LlamaForCausalLM.from_pretrained(
        model_checkpoint,
        config=llama_config,
        dtype=dtype,
        output_loading_info=True,
    )
    problems = {
        name: keys
        for name, keys in info.items()
        if name in ("missing_keys", "unexpected_keys", "mismatched_keys") and keys
    }
    if problems:
        raise RuntimeError(
            f"The Llama reference did not load the checkpoint exactly: "
            f"{problems}. Check the checkpoint key names."
        )
    llama.eval()
    return NanbeigeLoopReference(
        llama,
        num_loops=raw_config.get("num_loops", 1),
        skip_loop_final_norm=raw_config.get("skip_loop_final_norm", False),
    )


def get_full_model_goldens(
    checkpoint_path: str, tokenizer, prompts: list[str], output_length: int
) -> dict:
    """Return the FP32 and BF16 reference logits through the golden cache.

    Return a dict with ``input_ids``, ``fp32_logits`` and ``dtype_logits``.
    The FP32 pass decodes greedily. The BF16 pass is teacher-forced on the
    FP32 tokens.
    """
    import transformers

    from vllm_neuron.accuracy.goldens.reference_logits import (
        generate_three_way_reference_logits,
    )
    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    key_config = {
        "model": MODEL_ID.replace("/", "_"),
        "revision": HF_REVISION,
        "dtype": "bfloat16",
        "output_length": output_length,
        "prompts": prompts,
        "reference_impl": REFERENCE_IMPL,
    }

    def compute():
        return generate_three_way_reference_logits(
            checkpoint_path,
            torch.bfloat16,
            output_length,
            prompts,
            tokenizer,
            model_loader=load_loop_reference,
        )

    return get_or_compute_goldens(
        key_config,
        compute,
        metadata={
            "framework_version": f"transformers-{transformers.__version__}",
            "reference_impl": REFERENCE_IMPL,
            "modalities": list(SUPPORTED_MODALITIES),
        },
    )


# =============================================================================
# Decoder layer test: one physical layer, called once for each loop
# =============================================================================


def layer_weight_mappings(layer_idx: int) -> dict[str, str | list[str]]:
    """Return the map from ``NanbeigeDecoderLayer`` parameters to checkpoint keys."""
    prefix = f"model.layers.{layer_idx}"
    return {
        "self_attn.qkv_proj_weight": [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ],
        "self_attn.o_proj_weight": f"{prefix}.self_attn.o_proj.weight",
        "input_layernorm.weight": f"{prefix}.input_layernorm.weight",
        "post_attention_layernorm.weight": f"{prefix}.post_attention_layernorm.weight",
        "mlp.gate_proj_weight": f"{prefix}.mlp.gate_proj.weight",
        "mlp.up_proj_weight": f"{prefix}.mlp.up_proj.weight",
        "mlp.down_proj_weight": f"{prefix}.mlp.down_proj.weight",
    }


def make_layer_inputs(layer_idx: int, seq_len: int, hidden_size: int) -> dict:
    """Return seeded inputs for the two loops of one decoder layer.

    Each loop gets a different prefill input and a different decode input. A
    loop that reads the KV cache of the other loop then gives an incorrect
    result.
    """
    generator = torch.Generator().manual_seed(42 + layer_idx)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    return {
        "prefill": [randn(seq_len, hidden_size) for _ in range(NUM_LOOPS)],
        "decode": [randn(1, hidden_size) for _ in range(NUM_LOOPS)],
    }


def compute_layer_goldens(
    checkpoint_path: str, layer_idx: int, inputs: dict, with_decode: bool
) -> dict:
    """Run one upstream ``LlamaDecoderLayer`` in FP32 and in BF16.

    The order of the calls is the order of the Neuron test: prefill loop 0,
    prefill loop 1, then (if ``with_decode``) decode loop 0 and decode loop 1.
    Each loop has its own KV cache.
    """
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    raw_config = read_raw_config(checkpoint_path)
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in checkpoint_weight_map(checkpoint_path) if k.startswith(prefix)]
    state = {
        k[len(prefix) :]: v
        for k, v in load_checkpoint_tensors(checkpoint_path, keys).items()
    }

    results = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        config = build_llama_config(raw_config, dtype)
        layer = LlamaDecoderLayer(config, layer_idx=layer_idx).to(dtype)
        missing, unexpected = layer.load_state_dict(
            {k: v.to(dtype) for k, v in state.items()}, strict=True
        )
        assert not missing and not unexpected
        layer.eval()
        rotary = LlamaRotaryEmbedding(config)
        caches = [RefKVCache() for _ in range(NUM_LOOPS)]

        def run(hidden, start, loop_idx):
            hidden = hidden.to(dtype).unsqueeze(0)
            query_len = hidden.shape[1]
            position_ids = torch.arange(start, start + query_len).unsqueeze(0)
            with torch.no_grad():
                out = layer(
                    hidden,
                    attention_mask=causal_additive_mask(start, query_len, dtype),
                    position_ids=position_ids,
                    past_key_values=caches[loop_idx],
                    use_cache=True,
                    position_embeddings=rotary(hidden, position_ids),
                )
            return out[0].float()

        seq_len = inputs["prefill"][0].shape[0]
        outputs = {
            f"prefill_loop{loop}": run(inputs["prefill"][loop], 0, loop)
            for loop in range(NUM_LOOPS)
        }
        if with_decode:
            for loop in range(NUM_LOOPS):
                outputs[f"decode_loop{loop}"] = run(
                    inputs["decode"][loop], seq_len, loop
                )
        for loop in range(NUM_LOOPS):
            outputs[f"k_cache_loop{loop}"] = caches[loop].keys[0].float()
            outputs[f"v_cache_loop{loop}"] = caches[loop].values[0].float()
        results[name] = outputs
    return results


class _LayerStep(nn.Module):
    """One phase of one loop for a decoder layer. The weights are shared."""

    def __init__(self, layer, loop_idx: int, is_prefill: bool, rank: int, ws: int):
        super().__init__()
        self.layer = layer
        self.loop_idx = loop_idx
        self.is_prefill = is_prefill
        self.rank = rank
        self.ws = ws

    def forward(self, hidden_states, positions, cos, sin, attn_metadata):
        if self.is_prefill:
            # The decoder layer receives the sequence-parallel shard of this rank.
            per_rank = hidden_states.shape[0] // self.ws
            start = self.rank * per_rank
            hidden_states = hidden_states[start : start + per_rank]
        return self.layer(
            hidden_states, positions, (cos, sin), attn_metadata, self.loop_idx
        )


class DecoderLayerRunner(nn.Module):
    """Run one ``NanbeigeDecoderLayer`` for each (loop, phase) on one rank.

    Each (loop, phase) pair has its own compiled step, so the loop index is a
    constant in each graph. ``forward`` is not compiled. With
    ``return_caches``, it also returns the KV caches of each loop.
    """

    def __init__(self, layer, steps: dict):
        super().__init__()
        self.layer = layer
        self.steps = steps

    def forward(
        self,
        hidden_states,
        positions,
        cos,
        sin,
        attn_metadata,
        loop_idx: int,
        is_prefill: bool,
        return_caches: bool = False,
    ):
        out = self.steps[(loop_idx, is_prefill)](
            hidden_states, positions, cos, sin, attn_metadata
        )
        if not return_caches:
            return out
        caches = []
        for loop in range(NUM_LOOPS):
            caches.extend(self.layer.self_attn.caches(loop))
        return (out, *caches)


def load_decoder_layer_runner(
    checkpoint_path: str,
    layer_idx: int,
    num_blocks: int,
    block_size: int,
    device: str,
    compile_steps: bool,
) -> DecoderLayerRunner:
    """Load the TP shard of one decoder layer on this rank.

    This is the ``model_load`` function for the executor. It is a top-level
    function, so ``functools.partial`` of it can be pickled.
    """
    import torch.distributed as dist

    from vllm_neuron import envs
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    NanbeigeConfig = _local_src.load("config").NanbeigeConfig
    NanbeigeDecoderLayer = _local_src.load("model").NanbeigeDecoderLayer

    rank = dist.get_rank()
    ws = dist.get_world_size()
    config = NanbeigeConfig.from_configs(
        read_raw_config(checkpoint_path), NeuronConfig()
    )
    layer = NanbeigeDecoderLayer(config, layer_idx)

    checkpoint = SafetensorsCheckpoint(checkpoint_path, None)
    mappings = layer_weight_mappings(layer_idx)
    if device == "cpu":
        result = checkpoint.load_sharded(rank, ws, layer, mappings, device)
    else:
        result = checkpoint.load_sharded_pipelined(rank, ws, layer, mappings, device)
    # The loader lists each checkpoint key that no mapping uses. For a
    # single-layer load, this list always contains the other layers, the
    # embedding and the language model head. Examine only the keys of this
    # layer. An unused key of this layer is a missing mapping, so it fails.
    layer_prefix = f"model.layers.{layer_idx}."
    layer_unexpected = [
        key for key in result.unexpected_keys if key.startswith(layer_prefix)
    ]
    if result.missing_keys or layer_unexpected:
        raise RuntimeError(
            f"Layer {layer_idx} load is not complete: missing "
            f"{result.missing_keys}, unexpected {layer_unexpected}. "
            f"Make sure that layer_weight_mappings maps each key of the layer."
        )
    layer.load_state_dict(result.state_dict, strict=True, assign=True)

    attn = layer.self_attn
    cache_shape = (
        num_blocks,
        attn.num_key_value_heads_per_rank,
        block_size,
        attn.head_dim,
    )
    for loop in range(config.num_loops):
        setattr(
            attn,
            f"k_cache_{loop}",
            torch.zeros(cache_shape, dtype=config.torch_dtype, device=device),
        )
        setattr(
            attn,
            f"v_cache_{loop}",
            torch.zeros(cache_shape, dtype=config.torch_dtype, device=device),
        )

    steps = {}
    for loop in range(config.num_loops):
        for is_prefill in (True, False):
            step = _LayerStep(layer, loop, is_prefill, rank, ws)
            if compile_steps:
                step = torch.compile(
                    step, fullgraph=True, backend=envs.get_compile_backend_name()
                )
            steps[(loop, is_prefill)] = step
    return DecoderLayerRunner(layer, steps)


def layer_prefill_metadata(
    layer_idx: int, loop_idx: int, seq_len: int, block_size: int, num_blocks: int
) -> dict:
    """Return the prefill attention metadata of one (loop, layer) pair."""
    name = f"layers.{loop_idx * NUM_PHYSICAL_LAYERS + layer_idx}.self_attn"
    return {
        name: {
            "max_query_len": seq_len,
            "decode_token_threshold": 1,
            "slot_mapping": torch.arange(seq_len, dtype=torch.int64),
            "block_size": block_size,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }


def layer_decode_metadata(
    layer_idx: int, loop_idx: int, position: int, block_size: int, num_blocks: int
) -> dict:
    """Return the decode attention metadata of one (loop, layer) pair."""
    name = f"layers.{loop_idx * NUM_PHYSICAL_LAYERS + layer_idx}.self_attn"
    return {
        name: {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": torch.tensor([position], dtype=torch.long),
            "block_size": block_size,
            "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
            "max_blocks_per_seq": num_blocks,
        }
    }


def rope_tables(
    checkpoint_path: str, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the BF16 cos and sin tables of the native rotary embedding."""
    NanbeigeConfig = _local_src.load("config").NanbeigeConfig
    NanbeigeRotaryEmbedding = _local_src.load("model").NanbeigeRotaryEmbedding

    config = NanbeigeConfig.from_configs(read_raw_config(checkpoint_path))
    rotary = NanbeigeRotaryEmbedding(config)
    return rotary(positions, device=torch.device("cpu"), dtype=torch.bfloat16)


# =============================================================================
# Full model test: TP shard of the complete model on one rank
# =============================================================================


class FullModelRunner(nn.Module):
    """Run the full model and return the logits and a load report."""

    def __init__(self, model, load_report: torch.Tensor):
        super().__init__()
        self.model = model
        self.load_report = load_report

    def forward(self, input_ids, positions, attn_metadata, sampling_positions):
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
        )
        return logits, self.load_report


def load_full_model_runner(
    checkpoint_path: str, num_blocks: int, block_size: int, device: str
) -> FullModelRunner:
    """Load the TP shard of the full model through the registry factory.

    The load report holds: [checkpoint keys that the mappings read, missing
    keys, unexpected keys, parameters on meta, parameters with non-finite
    values, model parameters that the load did not set].
    """
    import torch.distributed as dist

    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    NanbeigeForCausalLM = _local_src.load().NanbeigeForCausalLM

    rank = dist.get_rank()
    ws = dist.get_world_size()
    # Without on-device sampling, the model returns the full-vocabulary logits.
    model = NanbeigeForCausalLM.from_configs(
        read_raw_config(checkpoint_path),
        NeuronConfig(on_device_sampling_config=None),
    )
    mappings = model.get_weight_mappings()
    source_keys = set()
    for value in mappings.values():
        source_keys.update([value] if isinstance(value, str) else value)

    checkpoint = SafetensorsCheckpoint(checkpoint_path, None)
    result = checkpoint.load_sharded(rank, ws, model, mappings, device)
    model.load_state_dict(result.state_dict, strict=False, assign=True)

    params = dict(model.named_parameters())
    not_loaded = [name for name in params if name not in result.state_dict]
    on_meta = [name for name, p in params.items() if p.is_meta]
    non_finite = [
        name
        for name, p in params.items()
        if not p.is_meta and not torch.isfinite(p.float()).all()
    ]
    if not_loaded or on_meta or non_finite:
        logger.error(
            "Rank %d load problems: not loaded %s, meta %s, non-finite %s",
            rank,
            not_loaded,
            on_meta,
            non_finite,
        )

    kv_caches = {}
    for spec in model.get_kv_spec().layers:
        shape = (num_blocks, spec.num_kv_heads, block_size, spec.head_size)
        kv_caches[spec.name] = [
            torch.zeros(shape, dtype=spec.dtype, device=device),
            torch.zeros(shape, dtype=spec.dtype, device=device),
        ]
    model.bind_kv_cache(kv_caches)

    report = torch.tensor(
        [
            len(source_keys),
            len(result.missing_keys),
            len(result.unexpected_keys),
            len(on_meta),
            len(non_finite),
            len(not_loaded),
        ],
        dtype=torch.int64,
    )
    return FullModelRunner(model, report)


def full_model_prefill_metadata(seq_len: int, block_size: int, num_blocks: int) -> dict:
    """Return the prefill attention metadata of every (loop, layer) pair."""
    entry = {
        "max_query_len": seq_len,
        "decode_token_threshold": 1,
        "slot_mapping": torch.arange(seq_len, dtype=torch.int64),
        "block_size": block_size,
        "block_table_tensor": torch.arange(num_blocks, dtype=torch.int32)[None],
        "cached_seq_len": None,
        "kv_segment_size": None,
    }
    return {
        f"layers.{i}.self_attn": dict(entry)
        for i in range(NUM_PHYSICAL_LAYERS * NUM_LOOPS)
    }
