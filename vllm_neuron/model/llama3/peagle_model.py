# SPDX-License-Identifier: Apache-2.0
"""P-EAGLE (Parallel-Drafting EAGLE) model for speculative decoding on Neuron.

P-EAGLE differs from EAGLE3 in how it produces the K draft tokens. Instead of
feeding each draft token back into the next step (EAGLE3's recurrent loop),
every MTP position (2..K) uses the same learned substitutes: a `mask_hidden`
state and the `ptd_token_id` embedding. Because no position depends on the
previous draft token, the K-1 MTP positions run as one causal multi-token block
in a single forward pass (see `_run_mtp_single_pass`), realizing the paper's
parallel drafting.
"""

import logging
import os

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron import functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.eagle3_model import (
    Eagle3LlamaDecoderLayer,
    compute_slot_mapping,
    extract_accepted_tokens,
)
from vllm_neuron.model.llama3.model import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader_with_padding,
)

logger = logging.getLogger(__name__)


def mask_hidden_weight_loader(
    unpadded_hidden_size: int,
    padded_hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader that per-part pads ``mask_hidden`` as the checkpoint is read.

    The checkpoint stores ``mask_hidden`` as ``[1, 1, 3 * unpadded_hidden_size]``
    (three concatenated auxiliary hidden states). When the model pads hidden_size
    (e.g. GPT-OSS 2880 -> 3072), each of the three parts is padded independently
    so the layout still matches ``fc``'s expected input, giving
    ``[1, 1, 3 * padded_hidden_size]``.

    ``mask_hidden`` is an ``nn.Parameter``, so the sharded checkpoint loader picks
    it up from ``named_parameters()`` and would otherwise assign the raw
    ``[1, 1, 3 * unpadded]`` tensor onto the ``[1, 1, 3 * padded]`` parameter.
    Padding here (rather than after ``load_state_dict``) keeps the shapes matching
    on the one code path that actually assigns the weight. Not TP-sharded: every
    rank holds the full tensor, so ``rank`` is unused.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        mask_hidden = slices[0][:]
        if padded_hidden_size == unpadded_hidden_size:
            return mask_hidden
        parts = torch.split(mask_hidden, unpadded_hidden_size, dim=-1)
        pad_size = padded_hidden_size - unpadded_hidden_size
        padded_parts = [torch.nn.functional.pad(p, (0, pad_size)) for p in parts]
        return torch.cat(padded_parts, dim=-1)

    return SafetensorsWeightLoader(transform=transform)


class PEagleLlamaForCausalLM(nn.Module):
    """P-EAGLE draft model: multi-token prediction via learned substitutes.

    - 4 decoder layers (vs EAGLE3's 1), configurable via num_hidden_layers.
    - MTP positions use learned substitutes instead of the previous draft token,
      so they carry no recurrent dependency and run as one causal block:
        - mask_hidden [1, 1, 3*hidden_size]: the substitute hidden state.
        - ptd_token_id: the substitute token id.
    """

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config
        # num_speculative_tokens is set externally by EagleProposer after construction
        self.num_speculative_tokens = getattr(
            config.neuron_config, "speculation_length", None
        )

        on_device_sampling_config = None
        if config.neuron_config is not None:
            on_device_sampling_config = config.neuron_config.on_device_sampling_config
        self.on_device_sampling = on_device_sampling_config is not None

        self._return_logits = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )

        # P-EAGLE specific: ptd_token_id for MTP positions
        self.ptd_token_id = getattr(config, "ptd_token_id", None)
        assert self.ptd_token_id is not None, (
            "ptd_token_id must be set in config for P-EAGLE"
        )

        # Backbone: multi-layer decoder stack.
        # Layer 0 gets [embeds, hidden] concat; layers 1+ get standard input.
        self.model = PEagleLlamaModel(
            config=config,
            start_layer_idx=start_layer_idx,
        )

        # mask_hidden: learned hidden state for MTP positions [1, 1, 3*hidden_size].
        # It's a trained weight in the checkpoint, so it's an nn.Parameter (not a
        # buffer). requires_grad=False since we only run inference. Uses padded
        # hidden_size so it's compatible with the fc layer input. The checkpoint
        # stores it unpadded, so a weight loader per-part pads it during load.
        self.mask_hidden = nn.Parameter(
            torch.zeros(1, 1, config.hidden_size * 3, dtype=config.torch_dtype),
            requires_grad=False,
        )
        set_weight_loader(
            self.mask_hidden,
            mask_hidden_weight_loader(config.unpadded_hidden_size, config.hidden_size),
        )

        lm_head_vocab = config.draft_vocab_size or config.vocab_size
        self.has_vocab_mapping = (
            config.draft_vocab_size is not None
            and config.draft_vocab_size != config.vocab_size
        )
        if self.has_vocab_mapping:
            self.register_buffer(
                "_draft_to_target",
                torch.zeros(config.draft_vocab_size, dtype=torch.int32),
                persistent=False,
            )

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            lm_head_vocab,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling,
        )

        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
            ),
        )

        if self.on_device_sampling:
            self.sampler = Sampler(
                on_device_sampling_config, process_group=get_tp_group().device_group
            )

    def _sample_draft_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.on_device_sampling:
            return self.sampler(
                logits,
                tp_rank=get_tp_group().rank_in_group_tensor,
            ).to(torch.int32)
        return torch.argmax(logits, dim=-1).to(torch.int32)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce K draft tokens for the current speculative-decode step.

        Two phases:
        1. Initial pass (identical to EAGLE3): run the full context to update
           the KV cache and sample the first draft token from real hidden states.
        2. MTP positions 2..K: each uses ptd_token_id + mask_hidden as inputs
           (no dependency on the previous draft token), submitted as a single
           causal multi-token block (see module docstring).

        Returns:
            (stacked_tokens, drafts_only_stacked, stacked_logits)
        """
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        # On-device token correction
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus_token_ids = extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )
        else:
            bonus_token_ids = None

        # Initial pass: project the 3 concatenated aux hidden states down to
        # hidden_size via fc, then run all T context tokens to update the KV cache.
        target_hidden_states = self.model.fc(initial_target_hidden_states)

        hidden_states, _ = self.model(
            input_ids=input_ids,
            positions=positions,
            target_hidden_states=target_hidden_states,
            attn_metadata=attn_metadata,
            rank=rank,
        )

        # Gather per-request states at sampling positions
        hidden_states = hidden_states[sampling_positions]
        cur_positions = positions[sampling_positions]
        batch_size = sampling_positions.shape[0]

        # Sample first draft token (NTP position, from real hidden states, same
        # as EAGLE3). [bs] token ids, [bs, vocab] logits.
        ntp_logits = self.lm_head(hidden_states)
        ntp_token_ids = self._sample_draft_token(ntp_logits)

        # --- P-EAGLE MTP positions 2..K ---
        # Every MTP position uses the same learned substitutes (ptd_token_id +
        # mask_hidden) rather than the previous draft token, so there is no
        # step-to-step data dependency: the K-1 positions run as one causal
        # multi-token block in a single model call.
        mtp_tokens, mtp_logits = self._run_mtp_single_pass(
            cur_positions=cur_positions,
            batch_size=batch_size,
            attn_metadata=attn_metadata,
            rank=rank,
            device=input_ids.device,
        )
        return self._finalize_draft_outputs(
            ntp_token_ids, mtp_tokens, ntp_logits, mtp_logits, bonus_token_ids
        )

    def _run_mtp_single_pass(
        self,
        cur_positions: torch.Tensor,
        batch_size: int,
        attn_metadata: object,
        rank: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute all K-1 MTP positions in ONE model call.

        Because every MTP position consumes the same inputs (``ptd_token_id`` +
        ``mask_hidden``) rather than the previously drafted token, the K-1
        positions are submitted together as a single multi-token block. Attention
        stays causal over the block, so position j still sees positions < j, and
        it builds one graph with one attention op instead of K-1 of them.

        Every shape here is derived from ``batch_size`` and the compile-time
        constant ``num_speculative_tokens``, so nothing is data-dependent.

        Returns:
            ``(mtp_tokens, mtp_logits)`` where ``mtp_tokens`` is ``[bs, K-1]`` and
            ``mtp_logits`` is ``[bs, K-1, vocab]`` (or ``None`` unless logits are
            requested).
        """
        num_mtp = self.num_speculative_tokens - 1

        first_layer_name = f"layers.{self.model.layers[0].layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        block_table_tensor = base_meta["block_table_tensor"]
        block_size = base_meta["block_size"]
        max_blocks_per_seq = base_meta["max_blocks_per_seq"]

        # Positions for the whole MTP block: [bs, num_mtp] -> flattened.
        # offsets are 1..num_mtp relative to the last real token.
        offsets = torch.arange(1, num_mtp + 1, device=device, dtype=cur_positions.dtype)
        block_positions = cur_positions.unsqueeze(1) + offsets.unsqueeze(0)
        flat_positions = block_positions.reshape(-1)  # [bs * num_mtp]

        # Same learned substitute (fc-projected mask_hidden) at every MTP
        # position, expanded to the whole block.
        mask_hidden_combined = self.model.fc(self.mask_hidden.squeeze(0))
        mtp_hidden = mask_hidden_combined.expand(batch_size * num_mtp, -1)
        mtp_input_ids = torch.full(
            (batch_size * num_mtp,),
            self.ptd_token_id,
            dtype=torch.int32,
            device=device,
        )

        # One slot per MTP position, computed from the same block table.
        slot_mapping = compute_slot_mapping(
            flat_positions,
            block_table_tensor.repeat_interleave(num_mtp, dim=0),
            block_size,
        )

        # Classify the block as a multi-token DECODE (max_query_len ==
        # decode_token_threshold == num_mtp), routing to the decode kernel whose
        # causal mask makes position j attend to < j. A prefill classification
        # would instead trigger sequence-parallel token scatter, which asserts
        # since bs * num_mtp is not divisible by tp_size.
        step_attn_metadata = {}
        for layer in self.model.layers:
            ln = f"layers.{layer.layer_idx}.self_attn"
            step_attn_metadata[ln] = {
                "block_table_tensor": block_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": num_mtp,
                "block_size": block_size,
                "max_blocks_per_seq": max_blocks_per_seq,
                "decode_token_threshold": num_mtp,
            }

        mtp_output, _ = self.model(
            input_ids=mtp_input_ids,
            positions=flat_positions,
            target_hidden_states=mtp_hidden,
            attn_metadata=step_attn_metadata,
            rank=rank,
        )

        # One LM-head call and one sampling call over all MTP positions at once
        # (lm_head and the greedy sampler operate on the leading dims), rather
        # than a per-position Python loop. [bs*num_mtp, H] -> logits -> tokens.
        mtp_logits = self.lm_head(mtp_output)  # [bs*num_mtp, vocab]
        mtp_tokens = self._sample_draft_token(mtp_logits)  # [bs*num_mtp]
        mtp_tokens = mtp_tokens.view(batch_size, num_mtp)  # [bs, num_mtp]
        mtp_logits = (
            mtp_logits.view(batch_size, num_mtp, -1) if self._return_logits else None
        )
        return mtp_tokens, mtp_logits

    def _finalize_draft_outputs(
        self,
        ntp_token_ids: torch.Tensor,
        mtp_tokens: torch.Tensor,
        ntp_logits: torch.Tensor,
        mtp_logits: torch.Tensor | None,
        bonus_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Assemble the K draft tokens/logits and map to the target vocab.

        ``ntp_token_ids`` is ``[bs]`` (the NTP/first draft token), ``mtp_tokens``
        is ``[bs, K-1]`` (the MTP block). The draft sequence is the NTP token
        followed by the MTP tokens; ``bonus_token_ids`` (the target's verified
        token) is prepended when present.
        """
        # [bs] + [bs, K-1] -> [bs, K] draft tokens.
        drafts_only_stacked = torch.cat([ntp_token_ids.unsqueeze(1), mtp_tokens], dim=1)
        if bonus_token_ids is not None:
            stacked_tokens = torch.cat(
                [bonus_token_ids.unsqueeze(1), drafts_only_stacked], dim=1
            )
        else:
            stacked_tokens = drafts_only_stacked

        stacked_tokens, drafts_only_stacked = (
            self._map_output_token_ids_to_target_vocab(
                stacked_tokens, drafts_only_stacked, bonus_token_ids
            )
        )

        stacked_logits = None
        if self._return_logits and mtp_logits is not None:
            stacked_logits = torch.cat([ntp_logits.unsqueeze(1), mtp_logits], dim=1)
        return stacked_tokens, drafts_only_stacked, stacked_logits

    def _map_output_token_ids_to_target_vocab(
        self,
        stacked_tokens: torch.Tensor,
        drafts_only_stacked: torch.Tensor,
        bonus_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_vocab_mapping:
            return stacked_tokens, drafts_only_stacked

        mapped_drafts_only = self._draft_to_target[drafts_only_stacked.long()]
        if bonus_token_ids is None:
            return mapped_drafts_only, mapped_drafts_only

        mapped_stacked = torch.cat(
            [bonus_token_ids.unsqueeze(1), mapped_drafts_only], dim=1
        )
        return mapped_stacked, mapped_drafts_only

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ):
        llama_config = LlamaConfig.from_configs(
            hf_config=config, neuron_config=neuron_config
        )
        llama_config.use_draft_mlp_groups = True
        # Pass ptd_token_id from HF config
        llama_config.ptd_token_id = getattr(config, "ptd_token_id", None)
        return cls(
            llama_config,
            start_layer_idx=start_layer_idx,
        )

    def _init_kv_cache_scales(self, device: torch.device) -> None:
        """Initialize KV cache quantization scales for P-EAGLE attention layers."""
        from vllm.config import get_current_vllm_config
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        ones = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
        for layer in self.model.layers:
            attn = layer.self_attn
            if attn.k_scale is None:
                attn.k_scale = ones.clone()
            if attn.v_scale is None:
                attn.v_scale = ones.clone()
            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load P-EAGLE checkpoint with 4 layers + mask_hidden.

        Weight naming follows the standard HF convention:
        - model.layers.{0-3}.self_attn.{q,k,v,o}_proj.weight
        - model.layers.0.hidden_norm.weight (only layer 0)
        - model.embed_tokens.weight
        - model.fc.weight, model.norm.weight
        - lm_head.weight
        - mask_hidden (P-EAGLE specific)
        """
        from safetensors import safe_open

        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        safetensor_files = sorted(
            f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
        )

        # Build weight mappings
        # Checkpoint naming: layer 0 uses "midlayer.*", layers 1+ use "layers.{1,2,3}.*"
        mappings = {}
        mappings["model.embed_tokens.weight"] = "embed_tokens.weight"
        mappings["model.norm.weight"] = "norm.weight"
        mappings["model.fc.weight"] = "fc.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        for layer_id in range(len(self.model.layers)):
            if layer_id == 0:
                ckpt_prefix = "midlayer"
            else:
                ckpt_prefix = f"layers.{layer_id}"

            mappings[f"model.layers.{layer_id}.self_attn.qkv_proj_weight"] = [
                f"{ckpt_prefix}.self_attn.q_proj.weight",
                f"{ckpt_prefix}.self_attn.k_proj.weight",
                f"{ckpt_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"model.layers.{layer_id}.self_attn.o_proj_weight"] = (
                f"{ckpt_prefix}.self_attn.o_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.up_proj_weight"] = (
                f"{ckpt_prefix}.mlp.up_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.gate_proj_weight"] = (
                f"{ckpt_prefix}.mlp.gate_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.down_proj_weight"] = (
                f"{ckpt_prefix}.mlp.down_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.input_layernorm.weight"] = (
                f"{ckpt_prefix}.input_layernorm.weight"
            )
            mappings[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = (
                f"{ckpt_prefix}.post_attention_layernorm.weight"
            )
            # hidden_norm only exists on layer 0 (Eagle3-style midlayer)
            if layer_id == 0:
                mappings[f"model.layers.{layer_id}.hidden_norm.weight"] = (
                    f"{ckpt_prefix}.hidden_norm.weight"
                )

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict
        # mask_hidden is padded by its weight loader during the load above, so it
        # arrives already shaped [1, 1, 3*padded_hidden] and needs no fixup here.
        load_result = self.load_state_dict(
            rank_sharded_checkpoint, strict=False, assign=True
        )
        # Diagnostic: strict=False silently tolerates params the checkpoint
        # mapping failed to populate (they keep their init values), which would
        # be a uniform, non-crashing accuracy loss. Log any such params loudly so
        # the failure mode is visible. ``_draft_to_target`` is a non-persistent
        # buffer that is loaded separately below, so it is expected here.
        missing = [
            k
            for k in getattr(load_result, "missing_keys", [])
            if "_draft_to_target" not in k
        ]
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing:
            logger.warning(
                "P-EAGLE load: %d MISSING params (ran on init values!): %s",
                len(missing),
                missing,
            )
        if unexpected:
            logger.warning(
                "P-EAGLE load: %d UNEXPECTED checkpoint keys: %s",
                len(unexpected),
                unexpected,
            )
        if not missing and not unexpected:
            logger.info("P-EAGLE load: all params loaded (no missing/unexpected keys)")
        logger.info("Loaded mask_hidden with shape %s", self.mask_hidden.shape)

        self._init_kv_cache_scales(device)

        # Load d2t vocab mapping if present
        if self.has_vocab_mapping:
            for f in safetensor_files:
                with safe_open(os.path.join(checkpoint_path, f), framework="pt") as sf:
                    if "d2t" in sf.keys():
                        d2t = sf.get_tensor("d2t")
                        draft_ids = torch.arange(len(d2t))
                        setattr(
                            self,
                            "_draft_to_target",
                            (draft_ids + d2t).to(torch.int32),
                        )

    def get_kv_spec(self):
        """Returns KV cache specification for vLLM integration."""
        layers = []
        for layer in self.model.layers:
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """Binds pre-allocated KV cache tensors to the model layers."""
        for layer in self.model.layers:
            layer_name = f"layers.{layer.layer_idx}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]


class PEagleLlamaModel(nn.Module):
    """P-EAGLE model backbone: multi-layer (default 4) decoder.

    Layer 0 is an Eagle3-style layer (concat [embeds, hidden] -> 2x QKV input).
    Layers 1+ are standard LlamaDecoderLayers (hidden_size QKV input).
    """

    def __init__(
        self,
        config: LlamaConfig,
        start_layer_idx: int,
    ):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=hidden_size,
            dtype=config.torch_dtype,
            tp_group=get_tp_group().device_group,
        )

        # Layer 0: Eagle3-style (2x hidden input to QKV)
        # Layers 1+: standard LlamaDecoderLayer
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer_idx = start_layer_idx + i
            if i == 0:
                self.layers.append(Eagle3LlamaDecoderLayer(config, layer_idx=layer_idx))
            else:
                self.layers.append(LlamaDecoderLayer(config, layer_idx=layer_idx))

        # Setup padding-aware weight loaders when hidden is padded
        if config.hidden_size != config.unpadded_hidden_size:
            from vllm_neuron.model.llama3.eagle3_model import _make_rmsnorm

            # Layer 0 (Eagle3LlamaDecoderLayer): set norm weight loaders
            layer0 = self.layers[0]
            set_weight_loader(
                layer0.input_layernorm.weight,
                last_dim_padding_weight_loader(config.hidden_size),
            )
            set_weight_loader(
                layer0.hidden_norm.weight,
                last_dim_padding_weight_loader(config.hidden_size),
            )
            set_weight_loader(
                layer0.post_attention_layernorm.weight,
                last_dim_padding_weight_loader(config.hidden_size),
            )

            for i, layer in enumerate(self.layers):
                if i == 0:
                    continue  # Structure already set by Eagle3LlamaDecoderLayer

                # Override norms with padding-aware variants
                layer.input_layernorm = _make_rmsnorm(config)
                layer.post_attention_layernorm = _make_rmsnorm(config)

                attn = layer.self_attn
                # QKV: input dim is hidden_size (padded), needs padding on dim 0
                set_weight_loader(
                    attn.qkv_proj_weight,
                    fused_qkv_weight_loader(
                        q_size=attn.q_size,
                        kv_size=attn.kv_size,
                        shard_dim=1,
                        num_shards=attn.world_size,
                        is_storage_transposed=True,
                        num_kv_replicas=attn.num_kv_replicas,
                        padded_hidden_size=config.hidden_size,
                        unpadded_hidden_size=config.unpadded_hidden_size,
                    ),
                )
                # o_proj: pad hidden_size dim (dim 1)
                set_weight_loader(
                    attn.o_proj_weight,
                    sharding_weight_loader_with_padding(
                        shard_dim=0,
                        shard_size=(attn.num_attention_heads * attn.head_dim)
                        // attn.world_size,
                        num_shards=attn.world_size,
                        is_storage_transposed=True,
                        pad_dim=1,
                        padded_size=config.hidden_size,
                        unpadded_size=config.unpadded_hidden_size,
                    ),
                )

                mlp = layer.mlp
                intermediate_per_rank = config.intermediate_size // mlp.world_size
                # gate_proj/up_proj: pad hidden_size dim (dim 0)
                gate_up_loader = sharding_weight_loader_with_padding(
                    shard_dim=1,
                    shard_size=intermediate_per_rank,
                    num_shards=mlp.world_size,
                    is_storage_transposed=True,
                    pad_dim=0,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                )
                set_weight_loader(mlp.gate_proj_weight, gate_up_loader)
                set_weight_loader(mlp.up_proj_weight, gate_up_loader)
                # down_proj: pad hidden_size dim (dim 1)
                set_weight_loader(
                    mlp.down_proj_weight,
                    sharding_weight_loader_with_padding(
                        shard_dim=0,
                        shard_size=intermediate_per_rank,
                        num_shards=mlp.world_size,
                        is_storage_transposed=True,
                        pad_dim=1,
                        padded_size=config.hidden_size,
                        unpadded_size=config.unpadded_hidden_size,
                    ),
                )

                # Norm weight loaders
                set_weight_loader(
                    layer.input_layernorm.weight,
                    last_dim_padding_weight_loader(config.hidden_size),
                )
                set_weight_loader(
                    layer.post_attention_layernorm.weight,
                    last_dim_padding_weight_loader(config.hidden_size),
                )

        self.fc = nn.Linear(
            hidden_size * 3,
            hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )

        # Padding-aware final norm, matching EAGLE3: _make_rmsnorm normalizes
        # over the unpadded portion when hidden is padded, else a plain RMSNorm.
        from vllm_neuron.model.llama3.eagle3_model import _make_rmsnorm

        self.norm = _make_rmsnorm(config)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # Weight loaders for embedding, fc, and norm (handle padding)
        from vllm_neuron.model.llama3.eagle3_model import (
            embedding_sharding_padding_weight_loader,
            fc_interleaved_padding_weight_loader,
        )

        unpadded_hidden_size = config.unpadded_hidden_size

        set_weight_loader(
            self.embed_tokens.weight,
            embedding_sharding_padding_weight_loader(
                vocab_size_per_rank=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                padded_hidden_size=hidden_size,
            ),
        )
        # TODO(satchill): P-EAGLE also reads the target's aux hidden states, so
        # fc and mask_hidden must follow the target's padding layout
        # (config.target_hidden_pad_groups, see to_target_hidden_layout in
        # eagle3_model.py). Under MXFP4 GPT-OSS this loader puts every input
        # feature in the wrong column, which drops the acceptance length to ~1.
        # Left for a separate change: P-EAGLE has no acceptance measurement yet.
        set_weight_loader(
            self.fc.weight,
            fc_interleaved_padding_weight_loader(unpadded_hidden_size, hidden_size),
        )
        set_weight_loader(self.norm.weight, last_dim_padding_weight_loader(hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.LongTensor,
        target_hidden_states: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through all layers.

        Layer 0 receives (embeds, target_hidden_states) like Eagle3.
        Layers 1+ receive hidden_states from previous layer (standard residual).
        """
        first_layer_name = f"layers.{self.layers[0].layer_idx}.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        embeds = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)

        if is_prefill and self.embed_tokens.tp_size > 1:
            target_hidden_states = shard_select(target_hidden_states, 0, get_tp_group())

        position_embeddings = self.rotary_emb(
            positions, device=embeds.device, dtype=embeds.dtype
        )

        # Layer 0: Eagle3-style forward (embeds + hidden concat)
        hidden_states = self.layers[0](
            embeds=embeds,
            hidden_states=target_hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )

        # Layers 1+: standard forward
        for layer in self.layers[1:]:
            hidden_states = layer(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_prenorm = hidden_states
        hidden_states = self.norm(hidden_states)

        if is_prefill and self.embed_tokens.tp_size > 1:
            tp_group = get_tp_group()
            hidden_states = tp_group.all_gather(hidden_states, dim=0)
            hidden_prenorm = tp_group.all_gather(hidden_prenorm, dim=0)

        return hidden_states, hidden_prenorm
