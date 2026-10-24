# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash Config
======================================

<-- MODEL-SPECIFIC: every field here is DeepSeek-V4.1-Flash-specific.

  - **Hyper-connections** (`hc_*`): every sublayer reads and writes `hc_mult=4`
    parallel residual streams, mixed by a 20-iteration Sinkhorn normalization.
  - **Engram** (`engram_*`): layers 1 and 14 carry a 16M-entry n-gram embedding
    table, 94.6 GiB each — 3.9x a single 24 GiB HBM bank.
  - **DSpark** (`dspark_*`): layers 37-39 run a Markov/confidence speculative
    head over its own 128-expert pool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

#: Ranks this port launches with: **one per logical NeuronCore of a
#: trn2.48xlarge**, 16 devices x 4 at LNC=2. Yours to tune; not an architectural
#: parameter, so nothing checks it against the official config.
TENSOR_PARALLEL_SIZE = 64

#: The deployment scheme, `DEEPSEEK_V41_SCHEME`: "tp64" (the default: the 64 ranks
#: are one tensor-parallel group) or "tp8pp8" (vLLM pipeline parallelism: 8 stages
#: of 8 tensor-parallel ranks, each stage holding only its own blocks' weights, the
#: scheduler keeping 8 batches in flight). `mk_model.PipePlan` reads the stage layout
#: back from vLLM's parallel groups; this constant only sizes the engine arguments.
SCHEME = os.environ.get("DEEPSEEK_V41_SCHEME", "tp64")
if SCHEME not in ("tp64", "tp8pp8"):
    raise ValueError(f"DEEPSEEK_V41_SCHEME={SCHEME!r}; expected 'tp64' or 'tp8pp8'")
PIPELINE_PARALLEL_SIZE = 8 if SCHEME == "tp8pp8" else 1
if SCHEME == "tp8pp8":
    TENSOR_PARALLEL_SIZE = 64 // PIPELINE_PARALLEL_SIZE

#: Ranks the **attention and the dense projections** are split across. Capped at
#: `o_groups` (8) because `wo_a` is block-diagonal over the groups and
#: `n_local_groups = o_groups // tp` reaches 0 above it; above the cap attention
#: is replicated `world_size // this` times and the head-partial all-reduces are
#: divided back down by `attn_replicas`.
#: 8 is the cap and the default. `DEEPSEEK_V41_ATTENTION_TP` lowers it, which
#: is needed to reach TP=4 x PP=16: a 4-rank tensor-parallel group cannot
#: contain an 8-way attention shard. At 4 each rank holds 2 of the 8 `o_groups`
#: and the head-partial reduction runs over the 4-rank lane group — one device,
#: the cheapest collective this host has — replicated 16x.
ATTENTION_TENSOR_PARALLEL_SIZE = int(
    os.environ.get("DEEPSEEK_V41_ATTENTION_TP", "8") or 8
)

#: Concurrent sequences this port serves above one rank. One because a run above
#: 1 has stalled rather than because 1 is right: `~/ds41_repro_hang.py` is the
#: minimal reproduction and `integrate.yaml`'s `concurrency-32` is the bar.
MAX_NUM_SEQS = int(os.environ.get("DEEPSEEK_V41_MAX_NUM_SEQS", 1))


@dataclass
class DeepseekV41Config:
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    rms_norm_eps: float = 1e-20
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    initializer_range: float = 0.02
    sliding_window: int = 128

    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    qk_rope_head_dim: int = 64

    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10000.0
    rope_scaling: dict | None = field(
        default_factory=lambda: {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        }
    )
    compress_rope_theta: float = 160000.0

    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-06

    engram_layer_ids: list[int] = field(default_factory=lambda: [1, 14])
    engram_num_embeddings: list[int] = field(
        default_factory=lambda: [384_006_168, 384_016_682]
    )
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16_000_000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99_092

    compress_ratios: list[int] = field(
        default_factory=lambda: [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    )
    kv_source_layer_ids: list[int] = field(default_factory=lambda: [2, 8, 14, 20])
    index_source_layer_ids: list[int] = field(
        default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36]
    )
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128_799
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
    image_token_id: int = 129_264

    quant_method: str = "fp8"
    activation_scheme: str = "dynamic"
    weight_block_size: list[int] = field(default_factory=lambda: [32, 32])
    scale_fmt: str = "ue8m0"
    expert_dtype: str = "fp4"
    torch_dtype: torch.dtype = torch.bfloat16

    vision_config: dict | None = None

    neuron_config: NeuronConfig | None = None


    @property
    def rope_factor(self) -> float:
        return float((self.rope_scaling or {}).get("factor", 1.0))

    @property
    def beta_fast(self) -> int:
        return int((self.rope_scaling or {}).get("beta_fast", 32))

    @property
    def beta_slow(self) -> int:
        return int((self.rope_scaling or {}).get("beta_slow", 1))

    @property
    def original_max_position_embeddings(self) -> int:
        return int(
            (self.rope_scaling or {}).get(
                "original_max_position_embeddings", self.max_position_embeddings
            )
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | str | bytes | dict,
        neuron_config: NeuronConfig | None = None,
    ) -> DeepseekV41Config:
        """Build from the official nested HF config, a path to one, or a dict."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                raw = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            raw = hf_config.to_dict()
        else:
            raw = dict(hf_config)

        def _as_dict(value) -> dict:
            if value is None:
                return {}
            if isinstance(value, PretrainedConfig):
                return value.to_dict()
            return dict(value) if isinstance(value, dict) else {}

        text = _as_dict(raw.get("text_config"))
        quant = _as_dict(raw.get("quantization_config"))
        vision = _as_dict(raw.get("vision_config")) or _as_dict(
            raw.get("vision_tower_config")
        )

        for scope in (text, raw):
            params = _as_dict(scope.get("rope_parameters"))
            if not params:
                continue
            scaling = {k: v for k, v in params.items() if k != "rope_theta"}
            scope.setdefault("rope_scaling", scaling)
            if "rope_theta" in params:
                scope.setdefault("rope_theta", params["rope_theta"])

        merged: dict = {**raw, **text, **quant}
        merged.pop("text_config", None)
        merged.pop("quantization_config", None)
        merged.pop("vision_tower_config", None)
        merged.pop("vision_config", None)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in field_names}

        dtype = raw.get("dtype") or raw.get("torch_dtype")
        if isinstance(dtype, str):
            filtered["torch_dtype"] = getattr(torch, dtype)
        elif isinstance(dtype, torch.dtype):
            filtered["torch_dtype"] = dtype

        if vision:
            filtered["vision_config"] = vision
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config

        return cls(**filtered)
