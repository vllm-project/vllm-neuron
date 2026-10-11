# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) Config
======================================

<-- MODEL-SPECIFIC: every field here is DeepSeek-V4.1-Flash-specific.

Ground truth is the official checkpoint config:
`huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json`
(revision `dba1be0a40aa45a94ad051997016db3960a90277`). Field names and values
below are that file's, not the partitioned reference's — the partition artifact
flattens and renames things (`dim`, `n_layers`, `max_seq_len`) and caps
`max_seq_len` at its own trace budget, so it is the wrong ground truth for a
config that has to load a real checkpoint.

That config is **nested**: `text_config`, `vision_config` and
`quantization_config` beside a handful of top-level fields. `from_configs`
descends into `text_config`, since this port serves the text path only.

Shape of the model, for orientation: 40 layers, 64 Q heads at head_dim 512 with
`num_key_value_heads=1` (MLA-style latent KV), hidden 5120, vocab 129280.
Three things make it unlike a Llama-shaped model, and each has its own kernel
under `kernels/`:

  - **Hyper-connections** (`hc_*`): every sublayer reads and writes `hc_mult=4`
    parallel residual streams, mixed by a 20-iteration Sinkhorn normalization.
  - **Engram** (`engram_*`): layers 1 and 14 carry a 16M-entry n-gram embedding
    table, 94.6 GiB each — 3.9x a single 24 GiB HBM bank.
  - **DSpark** (`dspark_*`): layers 37-39 run a Markov/confidence speculative
    head over its own 128-expert pool.

`vision_config` is read but **not modelled**: the vision tower is off the
`tokens -> logits` path and the floorplan treats placing it as an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

#: Tensor-parallel degree the verifier launches with. Yours to tune: not an
#: architectural parameter, so nothing checks it against the official config.
#: Ceiling is 4 (one device, 4 logical NeuronCores at LNC=2).
#:
#: Out here rather than on the model because `tensor_parallel_size` is an
#: engine-construction argument — workers are spawned before this class is
#: instantiated, once per rank. A model cannot choose its own world size, only
#: be told.
#:
#: **4, because HBM is the binding constraint and this is what buys HBM.**
#: One Trainium2 device carries four 24 GiB HBM stacks, one per logical
#: NeuronCore, and a rank addresses only its own stack — so one rank uses 24 of
#: the device's 96 GiB and leaves 72 idle. Four ranks use all four stacks *and*
#: divide the resident weights, measured at **10.60 -> 5.09 GiB a rank**
#: (`.autohelix/notes/iter-5.md` §10h). The 8192-token prefill segment the
#: verifier's longest prompt needs ran all 40 layers at one rank and then could
#: not load the next NEFF with 2.8 GiB left (§10f); the 5.5 GiB a rank this
#: frees is the headroom for it.
#:
#: It was 1 for iterations 1-5, for reasons that the measurements have since
#: overtaken and which are recorded here so the trade is legible:
#:
#:   - "the resident weights fit a single bank whole, so sharding buys no
#:     capacity" — true of the *weights*, and beside the point. What does not
#:     fit is the weights plus the activations of a 43-block stack plus the
#:     loaded NEFFs of its graphs, and sharding the first of those three is how
#:     the other two get room.
#:   - "every graph is compiled once per rank, and compile time is the binding
#:     cost" — still true and still the price: 88 cache keys / 36 compiles at
#:     four ranks against 27 / 7 at one, for the same work (§10e, §10h). The
#:     four ranks compile concurrently, so it costs wall clock rather than
#:     multiplying it.
#:   - "where only an `opt-rank0` kernel variant exists, the 4-rank collective
#:     overhead exceeded the speedup" — that is the brief's guidance on
#:     *throughput*, and it is why the kernels on the live path carry no
#:     collective. It does not speak to capacity, which is what is short here.
#:
#: `weight_loader.py` shards correctly at either value — `verify_shard_plan`
#: reports 0 problems at 1 and at 4 — the collectives `MoE.forward` needs at
#: four ranks go through `tp_all_reduce` / `tp_all_gather_last` in `model.py`,
#: and a four-rank run has been taken through a whole 40-layer prefill (§10h).
#: So this is still reversible by editing the number.
TENSOR_PARALLEL_SIZE = 4


@dataclass
class DeepseekV41Config:
    # ── Core transformer shape (text_config) ──────────────────────────────
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

    # ── Attention: low-rank Q/O projections, grouped output ───────────────
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    qk_rope_head_dim: int = 64

    # ── RoPE (YaRN), flattened out of text_config.rope_scaling ────────────
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

    # ── MoE FFN ───────────────────────────────────────────────────────────
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5

    # ── Hyper-connections ─────────────────────────────────────────────────
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-06

    # ── Engram n-gram memory (layers 1 and 14) ────────────────────────────
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

    # ── Sparse index / KV compression ─────────────────────────────────────
    # `compress_ratios` carries 43 entries: 40 layers + 3 MTP layers.
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

    # ── DSpark speculative decoding (layers 37-39) ────────────────────────
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128_799
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    # ── Top-level checkpoint fields ───────────────────────────────────────
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
    image_token_id: int = 129_264

    # ── Quantization (top-level quantization_config) ──────────────────────
    # The checkpoint stores attention fp8 (ue8m0 scales, 32x32 weight blocks,
    # dynamic activation scaling) and expert weights fp4, while arithmetic runs
    # bf16. `torch_dtype` is the *compute* dtype, which is what every module
    # signature is written against.
    quant_method: str = "fp8"
    activation_scheme: str = "dynamic"
    weight_block_size: list[int] = field(default_factory=lambda: [32, 32])
    scale_fmt: str = "ue8m0"
    expert_dtype: str = "fp4"
    torch_dtype: torch.dtype = torch.bfloat16

    # ── Vision: recorded, NOT modelled (off the tokens -> logits path) ────
    vision_config: dict | None = None

    # ── Framework config ──────────────────────────────────────────────────
    neuron_config: NeuronConfig | None = None

    # ── Derived views over rope_scaling ───────────────────────────────────
    # Read-only, so there is one source of truth per value and the flattened
    # spellings cannot drift from the nested dict.

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
        """Build from the official nested HF config, a path to one, or a dict.

        Flattens `text_config` and `quantization_config` into this dataclass and
        keeps `vision_config` as an opaque dict. Unknown keys are dropped rather
        than raising, matching the Qwen3/Llama3 loaders in this repo, so a
        checkpoint carrying extra fields still loads.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                raw = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            raw = hf_config.to_dict()
        else:
            raw = dict(hf_config)

        def _as_dict(value) -> dict:
            # transformers may hand back either a nested PretrainedConfig or a
            # plain dict depending on how the config was constructed.
            if value is None:
                return {}
            if isinstance(value, PretrainedConfig):
                return value.to_dict()
            return dict(value) if isinstance(value, dict) else {}

        text = _as_dict(raw.get("text_config"))
        # `storage_quantization_config` is where `DeepseekV41HFConfig` parks the
        # checkpoint's `quantization_config` block, so that vLLM does not read
        # it as a request for one of its own quantization backends. Either
        # spelling is accepted: raw checkpoint JSON has the first, a config that
        # went through transformers has the second.
        quant = _as_dict(raw.get("quantization_config"))
        # `vision_tower_config` is where `DeepseekV41HFConfig` parks the
        # checkpoint's `vision_config`, so the Neuron platform does not switch
        # on a vision tower this port does not build. Either spelling is
        # accepted: raw checkpoint JSON has the first, a config that went
        # through transformers has the second.
        vision = _as_dict(raw.get("vision_config")) or _as_dict(
            raw.get("vision_tower_config")
        )

        # transformers 5 standardizes RoPE: it folds `rope_theta` *into*
        # `rope_scaling` and re-exposes the pair as `rope_parameters`, so a
        # config that round-tripped through `PretrainedConfig` no longer spells
        # `rope_theta` or `rope_scaling` at all. Unfold it back, so the values
        # below come from the checkpoint either way instead of silently falling
        # back to these defaults. Checkpoint JSON read straight off disk still
        # has both keys and takes this path as a no-op.
        for scope in (text, raw):
            params = _as_dict(scope.get("rope_parameters"))
            if not params:
                continue
            scaling = {k: v for k, v in params.items() if k != "rope_theta"}
            scope.setdefault("rope_scaling", scaling)
            if "rope_theta" in params:
                scope.setdefault("rope_theta", params["rope_theta"])

        # text_config wins over top level for LM fields; a flat config (no
        # text_config) still works, which is what makes a hand-written dict or
        # the partitioned reference usable in tests.
        merged: dict = {**raw, **text, **quant}
        merged.pop("text_config", None)
        merged.pop("quantization_config", None)
        merged.pop("vision_tower_config", None)
        merged.pop("vision_config", None)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in field_names}

        # The checkpoint spells the compute dtype `dtype` at top level
        # ("bfloat16"); `quant_method`/`expert_dtype` are storage formats and
        # stay strings.
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
