# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass, field
from typing import Any, List, Literal


# TODO: expand support of device sampling configs
class OnDeviceSamplingConfig:
    def __init__(self, **kwargs: Any):
        all_greedy_val = kwargs.pop("all_greedy", False)
        # Handle string "true"/"false" from config
        if isinstance(all_greedy_val, str):
            self.all_greedy = all_greedy_val.lower() in ("true", "1", "yes")
        else:
            self.all_greedy = bool(all_greedy_val)

        self.max_top_k = int(kwargs.pop("max_top_k", 256))

        deterministic_val = kwargs.pop("deterministic", False)
        if isinstance(deterministic_val, str):
            self.deterministic = deterministic_val.lower() in ("true", "1", "yes")
        else:
            self.deterministic = bool(deterministic_val)

        # Data-parallel sampling degree (1 = disabled, >1 = batch sharding)
        self.sampling_dp_degree = int(kwargs.pop("sampling_dp_degree", 1))

        # Debug/validation only: when True, capture the sampling top-k outputs
        # (values, indices) via the tensor-capture facility. No-op unless a
        # tensor_capture config is also active (TensorRegistry must be set); the
        # if-guard folds to a constant at trace time, so the default-False path
        # leaves the compiled graph unchanged.
        capture_topk_val = kwargs.pop("capture_topk", False)
        if isinstance(capture_topk_val, str):
            self.capture_topk = capture_topk_val.lower() in ("true", "1", "yes")
        else:
            self.capture_topk = bool(capture_topk_val)


@dataclass
class TensorCaptureConfig:
    """Configuration for tensor capture debugging.

    Args:
        modules: List of module paths for hook-based capture
            (e.g., ["model.layers.0", "lm_head"])
        capture_dir: Directory to save captured tensors
        capture_filter: Optional list of tensor names for write-time
            filtering. When set, only captured tensors whose names match
            this list are written to disk; all others are silently
            skipped.
    """

    modules: List[str] = field(default_factory=list)
    draft_modules: List[str] = field(default_factory=list)
    capture_dir: str = "/tmp/vllm_neuron_tensor_captures"
    capture_filter: List[str] | None = None


@dataclass
class TensorReplacementConfig:
    """Configuration for tensor replacement.

    Args:
        tensors: Per-prompt reference tensors as a list of
            {module_name: [step_tensors]} dicts, where list index
            is the prompt index, step 0 is prefill, steps 1+ are decode.
        tensors_path: Path to a torch.save'd file containing
            tensors. Used for online serving where tensors cannot
            be passed in-memory via JSON config.
        prompt_token_ids: Per-prompt input token IDs used to match
            incoming requests to the correct prompt captures via
            prefix hashing. Required for multi-prompt validation.
        prompt_token_ids_path: Path to a torch.save'd file containing
            prompt_token_ids. Used for online serving.
    """

    tensors: list = field(default_factory=list, repr=False)
    tensors_path: str | None = None
    prompt_token_ids: list | None = field(default=None, repr=False)
    prompt_token_ids_path: str | None = None

    def resolve(self) -> tuple[list, list | None]:
        """Validate config and load tensors/prompt_token_ids from paths if needed."""
        import torch

        tensors = self.tensors
        if tensors and self.tensors_path:
            raise ValueError(
                "tensor_replacement: specify either 'tensors' or "
                "'tensors_path', not both."
            )
        if not tensors and self.tensors_path:
            tensors = torch.load(self.tensors_path, weights_only=False)
        if not tensors:
            raise ValueError(
                "tensor_replacement: must specify either 'tensors' or "
                "'tensors_path' with valid data."
            )

        prompt_token_ids = self.prompt_token_ids
        if not prompt_token_ids and self.prompt_token_ids_path:
            prompt_token_ids = torch.load(
                self.prompt_token_ids_path, weights_only=False
            )

        return tensors, prompt_token_ids

    def __repr__(self):
        if self.tensors_path:
            return f"TensorReplacementConfig(path={self.tensors_path!r})"
        n_prompts = len(self.tensors)
        n_modules = len(self.tensors[0]) if n_prompts > 0 else 0
        return f"TensorReplacementConfig(prompts={n_prompts}, modules={n_modules})"


@dataclass
class NeuronConfig:
    # TODO: Revisit sharding config consistency between vLLM Neuron and vLLM.
    # In the case of EP, we define an ep_degree argument in NeuronConfig because vLLM doesn't
    # support configuring EP degree.
    ep_degree: int = 1  # Expert parallelism degree
    attention_dp_size: int = (
        1  # KV data parallelism size (decode-only Q/O sharding across DP)
    )
    embedding_dp_size: int = 1  # Shard Embedding across TP * this many DP ranks
    lm_head_dp_size: int = 1  # Shard LM Head across TP * this many DP ranks
    mlp_dp_size: int = 1  # Shard dense MLP across TP * this many DP ranks
    on_device_sampling_config: OnDeviceSamplingConfig | None = field(
        default_factory=OnDeviceSamplingConfig
    )
    # TODO: Remove max_logprobs from NeuronConfig; use vllm_config directly in the model instead
    max_logprobs: int = 0  # 0 = logprobs disabled; -1 = gather all logits for logprobs;
    tensor_capture: TensorCaptureConfig | None = None
    tensor_replacement: TensorReplacementConfig | None = None
    kv_segment_size_buckets: list[int] | None = None
    # Alternative to max_logprobs for test validation. Avoids the serialization
    # inefficiency of vLLM's logprobs pipeline by writing raw logits directly to
    # /dev/shm/{debug_logits_dir}/ with position metadata.
    # Used by e2e logit tests to compare Neuron outputs against golden baselines.
    debug_logits_dir: str | None = None
    # Model quantization type (e.g., "bf16", "mxfp4", "mxfp8")
    quantization: str | None = None
    # all2all communication backend when EP is enabled. vLLM's --all2all-backend
    # Literal does not include 'neuron'.
    all2all_backend: Literal[None, "neuron"] = None
    # Substring-match list of parameter FQNs to KEEP in bf16, i.e. skip during
    # CPU dequant/quantization. A parameter keeps bf16 iff any entry in this
    # list is a substring of its fully-qualified name. None/[] => convert
    # everything the active quantization path supports. Mirrors the NxDI /
    # AWSNeuronEval modules_to_not_convert convention.
    modules_to_not_convert: list[str] | None = None
    # Bucket configurations for scheduling
    num_batched_tokens_buckets: list[int] | None = None
    num_seqs_buckets: list[int] | None = None
    # Decode context-length bucketing. When set, decode is compiled per
    # (batch_bucket, ctx_bucket) pair plus a max_model_len fallback NEFF.
    # See docs/design/vllm/decode-context-length-bucketing.md.
    decode_context_length_buckets: list[int] | None = None
    # Structured outputs are request-level. Keep this disabled by default for
    # SO-off perf mode. Set true for servers that need to accept SO requests.
    enable_structured_outputs: bool = False
    # Opt-in to the swizzled packed FP8 K cache layout. When enabled (and the
    # KV cache dtype is float8_e4m3fn), K is stored as
    # [num_blocks, kv_heads, block_size // 2, head_size, 2] so the attention
    # kernels can bf16-reinterpret + DMA-transpose the K load. Only models wired
    # for the packed layout (currently GPT-OSS and Llama3, incl. Eagle3) support
    # it; left off by default so other models keep the standard 4-D FP8 cache.
    fp8_packed_kv: bool = False
    # Override for the extra ``--internal-hlo2tensorizer-options`` the runner
    # passes to neuronx-cc. ``None`` keeps the default
    # ``--modular-flow-mac-threshold=10``, which exists only because NKI kernels
    # do not yet report MAC counts. Set ``""`` to pass no extra options, which
    # is what a model built entirely from torch ops wants: the threshold makes
    # neuronx-cc fail codegen on some such graphs (Qwen3.5's decode graph hits
    # NCC_IBTN006, a pftranspose whose copy fails backend verification).
    # The fp8 cast flag is still appended when the graph needs it.
    hlo2tensorizer_options: str | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NeuronConfig":
        """Build NeuronConfig from a dictionary (e.g., from additional_config['neuron_config']).

        Args:
            config_dict: Dictionary with neuron configuration options.

        Returns:
            NeuronConfig instance with parsed values.
        """
        # Parse on_device_sampling_config
        on_device_sampling_dict = config_dict.get("on_device_sampling_config")
        if on_device_sampling_dict is not None:
            on_device_sampling_config = OnDeviceSamplingConfig(
                **on_device_sampling_dict
            )
        elif "on_device_sampling_config" in config_dict:
            # Explicitly set to None → disable ODS
            on_device_sampling_config = None
        else:
            # Key absent → use default (ODS enabled)
            on_device_sampling_config = OnDeviceSamplingConfig()

        # Parse tensor_capture config
        tensor_capture = None
        tensor_capture_dict = config_dict.get("tensor_capture")
        if tensor_capture_dict is not None:
            tensor_capture = TensorCaptureConfig(
                modules=tensor_capture_dict.get("modules", []),
                draft_modules=tensor_capture_dict.get("draft_modules", []),
                capture_dir=tensor_capture_dict.get(
                    "capture_dir", "/tmp/vllm_neuron_tensor_captures"
                ),
                capture_filter=tensor_capture_dict.get("capture_filter"),
            )

        # Parse tensor_replacement config
        tensor_replacement = None
        tensor_replacement_value = config_dict.get("tensor_replacement")
        if tensor_replacement_value is not None:
            if isinstance(tensor_replacement_value, TensorReplacementConfig):
                tensor_replacement = tensor_replacement_value
            else:
                tensor_replacement = TensorReplacementConfig(
                    tensors=tensor_replacement_value.get("tensors", []),
                    tensors_path=tensor_replacement_value.get("tensors_path"),
                    prompt_token_ids=tensor_replacement_value.get("prompt_token_ids"),
                    prompt_token_ids_path=tensor_replacement_value.get(
                        "prompt_token_ids_path"
                    ),
                )

        return cls(
            ep_degree=config_dict.get("ep_degree", 1),
            attention_dp_size=config_dict.get("attention_dp_size", 1),
            embedding_dp_size=config_dict.get("embedding_dp_size", 1),
            lm_head_dp_size=config_dict.get("lm_head_dp_size", 1),
            mlp_dp_size=config_dict.get("mlp_dp_size", 1),
            on_device_sampling_config=on_device_sampling_config,
            max_logprobs=config_dict.get("max_logprobs", 0),
            tensor_capture=tensor_capture,
            tensor_replacement=tensor_replacement,
            kv_segment_size_buckets=config_dict.get("kv_segment_size_buckets"),
            debug_logits_dir=config_dict.get("debug_logits_dir"),
            quantization=config_dict.get("quantization"),
            all2all_backend=config_dict.get("all2all_backend"),
            modules_to_not_convert=config_dict.get("modules_to_not_convert"),
            num_batched_tokens_buckets=config_dict.get("num_batched_tokens_buckets"),
            num_seqs_buckets=config_dict.get("num_seqs_buckets"),
            decode_context_length_buckets=config_dict.get(
                "decode_context_length_buckets"
            ),
            enable_structured_outputs=config_dict.get(
                "enable_structured_outputs", False
            ),
            fp8_packed_kv=config_dict.get("fp8_packed_kv", False),
            # Absent -> None -> the runner keeps its default extra options.
            hlo2tensorizer_options=config_dict.get("hlo2tensorizer_options"),
        )

    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.ep_degree, int) or self.ep_degree < 1:
            raise ValueError(
                f"ep_degree must be a positive integer, got {self.ep_degree}"
            )
        if not isinstance(self.attention_dp_size, int) or self.attention_dp_size < 1:
            raise ValueError(
                f"attention_dp_size must be a positive integer, got {self.attention_dp_size}"
            )
        if not isinstance(self.embedding_dp_size, int) or self.embedding_dp_size < 1:
            raise ValueError(
                f"embedding_dp_size must be a positive integer, got {self.embedding_dp_size}"
            )
        if not isinstance(self.lm_head_dp_size, int) or self.lm_head_dp_size < 1:
            raise ValueError(
                f"lm_head_dp_size must be a positive integer, got {self.lm_head_dp_size}"
            )
        if not isinstance(self.mlp_dp_size, int) or self.mlp_dp_size < 1:
            raise ValueError(
                f"mlp_dp_size must be a positive integer, got {self.mlp_dp_size}"
            )
        if not isinstance(self.enable_structured_outputs, bool):
            raise ValueError(
                "enable_structured_outputs must be a boolean, "
                f"got {self.enable_structured_outputs}"
            )


@dataclass
class VisionNeuronConfig:
    """Neuron-specific configuration for vision encoders.

    Args:
        num_vision_tokens_buckets: Bucket sizes for vision token sequences.
            Each bucket defines a compiled graph for that token count.
        vision_attention_block_size: Block size for chunked vision attention.
            Vision sequences are split into blocks of this size for attention.
        tp_size: Tensor parallelism degree for the vision encoder.
            Default 1. When > 1, encoder weights are sharded across tp_size ranks.
        dp_size: Data parallelism degree for the vision encoder.
            Default 1. When > 1, vision blocks are scattered across dp_size ranks.
            Must satisfy tp_size * dp_size == world_size after resolution.
        max_vision_seq_len: Optional cap on the auto-generated max bucket
            size. Limits compilation to only the bucket range the workload
            needs. Ignored when num_vision_tokens_buckets is set explicitly.
        mm_encoder_only: Build vision-only (skip the language model) — the Vision
            Encoder (VE) pool. Sourced from upstream ``--mm-encoder-only``; set by
            the model runner, not parsed from the vision config dict.
        mm_language_model_only: Build language-only (skip the vision tower) — the
            Prefill/Decode (PD) pool. Upstream has no equivalent because its
            disaggregated-encoder consumer loads the full model (vision tower
            built but unused), so this is a plugin flag from ``additional_config``;
            set by the model runner. Mutually exclusive with ``mm_encoder_only``.
        encoder_cache_num_blocks: Override for on-device encoder cache block
            count. None (default): auto-derived from scheduler's
            encoder_cache_size budget in NeuronModelRunner. Set explicitly to
            increase capacity beyond the scheduler budget, e.g. to compensate
            for per-block padding waste or hold-time backlog under high
            concurrency.
            TODO: The scheduler's EncoderCacheManager sizes and evicts based
            on token counts, unaware of block-level padding waste. This
            mismatch means the scheduler may cache more items than the block
            allocator can hold (each item uses a full block regardless of
            token count). A proper fix requires integrating block layout
            awareness into the scheduler's EncoderCacheManager. Until then,
            set this field explicitly for workloads with many small images.
        encoder_cache_min_hold_time_ms: Minimum time (ms) a cache block stays
            allocated after write before it can be freed. Prevents premature
            reuse when a remote reader (DE/RDMA) hasn't finished pulling the
            data. None = auto (0 for monolithic, 100ms for EPD).
    """

    num_vision_tokens_buckets: list[int] | None = None
    vision_attention_block_size: int = 2048
    tp_size: int = 1
    dp_size: int = 1
    # Default: vision_attention_block_size (compile only the minimum
    # single-block graph). Resolved in __post_init__.
    max_vision_seq_len: int | None = None

    # Encoder cache block configs
    # When the cache can only hold one request's embedding, as long as the hold time
    # is less than prefill latency we should not observe any stall in VE.
    # When the cache is big enough to hold multiple requests' embeddings,
    # we should not observe stall as well.
    encoder_cache_num_blocks: int | None = None
    encoder_cache_min_hold_time_ms: float | None = None

    # EPD construction-role flags (both False -> monolith: build both towers).
    mm_encoder_only: bool = False
    mm_language_model_only: bool = False

    # Derived fields — set by resolve_tp_dp
    num_total_vision_attention_blocks: list[int] | None = None

    def resolve_tp_dp(self, world_size: int) -> tuple[int, int]:
        """Resolve vision TP and DP sizes, inferring whichever is at default.

        Resolution rules:
          - Both at default (1) → tp_size=1, dp_size=world_size
          - Only tp_size set (dp_size==1) → dp_size = world_size // tp_size
          - Only dp_size set (tp_size==1) → tp_size = world_size // dp_size
          - Both set (neither is 1) → validate tp_size * dp_size == world_size

        After resolution, self.tp_size and self.dp_size hold concrete values
        and num_total_vision_attention_blocks is recomputed with dp-aligned
        padding.

        Args:
            world_size: Total number of ranks.

        Returns:
            (vision_tp_size, vision_dp_size) tuple.

        Raises:
            ValueError: If constraints are violated.
        """
        tp = self.tp_size
        dp = self.dp_size

        if tp == 1 and dp == 1:
            tp, dp = 1, world_size
        elif tp > 1 and dp == 1:
            if world_size % tp != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"vision tp_size ({tp})"
                )
            dp = world_size // tp
        elif dp > 1 and tp == 1:
            if world_size % dp != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"vision dp_size ({dp})"
                )
            tp = world_size // dp
        else:
            if tp * dp != world_size:
                raise ValueError(
                    f"vision tp_size ({tp}) * dp_size ({dp}) = "
                    f"{tp * dp} must equal world_size ({world_size})"
                )

        self.tp_size = tp
        self.dp_size = dp

        # Recompute block counts with dp-aligned padding
        if self.num_vision_tokens_buckets is not None:
            self.num_total_vision_attention_blocks = [
                math.ceil(math.ceil(bucket / self.vision_attention_block_size) / dp)
                * dp
                for bucket in self.num_vision_tokens_buckets
            ]

        return tp, dp

    def __post_init__(self):
        """Validate fields and derive initial block counts."""
        if self.max_vision_seq_len is None:
            self.max_vision_seq_len = self.vision_attention_block_size
        if self.tp_size < 1:
            raise ValueError(
                f"vision tp_size must be a positive integer, got {self.tp_size}"
            )
        if self.dp_size < 1:
            raise ValueError(
                f"vision dp_size must be a positive integer, got {self.dp_size}"
            )
        if self.max_vision_seq_len < 1:
            raise ValueError(
                f"max_vision_seq_len must be a positive integer, got {self.max_vision_seq_len}"
            )
        if self.num_vision_tokens_buckets is not None:
            self.num_total_vision_attention_blocks = [
                math.ceil(bucket / self.vision_attention_block_size)
                for bucket in self.num_vision_tokens_buckets
            ]
            if self.dp_size > 1:
                self.num_total_vision_attention_blocks = [
                    math.ceil(nb / self.dp_size) * self.dp_size
                    for nb in self.num_total_vision_attention_blocks
                ]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionNeuronConfig":
        """Build VisionNeuronConfig from a dictionary.

        Args:
            config_dict: Dictionary with vision neuron configuration options.

        Returns:
            VisionNeuronConfig instance with parsed values.
        """
        return cls(
            num_vision_tokens_buckets=config_dict.get("num_vision_tokens_buckets"),
            vision_attention_block_size=config_dict.get(
                "vision_attention_block_size", 2048
            ),
            tp_size=config_dict.get("tp_size", 1),
            dp_size=config_dict.get("dp_size", 1),
            max_vision_seq_len=config_dict.get("max_vision_seq_len"),
            encoder_cache_num_blocks=config_dict.get("encoder_cache_num_blocks"),
            encoder_cache_min_hold_time_ms=config_dict.get(
                "encoder_cache_min_hold_time_ms"
            ),
        )

    def set_construction_role(self, multimodal_config, additional_config: dict) -> None:
        """Set the EPD construction-role flags from the runtime configs.

        ``mm_encoder_only`` is an upstream field on
        ``model_config.multimodal_config`` (``None`` for non-multimodal models);
        ``mm_language_model_only`` is a plugin field read from
        ``additional_config``. The two are mutually exclusive — a pool is either
        vision-only (VE) or language-only (PD), not both.
        """
        self.mm_encoder_only = bool(
            multimodal_config is not None and multimodal_config.mm_encoder_only
        )
        self.mm_language_model_only = bool(
            additional_config.get("mm_language_model_only")
        )
        if self.mm_encoder_only and self.mm_language_model_only:
            raise ValueError(
                "mm_encoder_only and mm_language_model_only are mutually "
                "exclusive: a pool is either vision-only (VE) or language-only "
                "(PD), not both."
            )

        # Resolve hold time now that EPD flags are known.
        # Monolithic needs no guard (prefill reads immediately after VE writes
        # on same device). EPD needs 100ms for the remote reader (DE/RDMA) to
        # pull data before blocks are freed.
        if self.encoder_cache_min_hold_time_ms is None:
            is_epd = self.mm_encoder_only or self.mm_language_model_only
            self.encoder_cache_min_hold_time_ms = 100.0 if is_epd else 0.0
