# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) model — STAGE 1b SCAFFOLD
=========================================================

One module, one `forward`. Deliberately NOT the layered
`Model`/`DecoderLayer`/`Attention` structure the onboarding guide shows: at this
stage the only job is to be registrable and to fail in one obvious place, so the
smoke test exercises the wiring (registry -> factory -> config -> instantiation)
rather than the arithmetic.

Every method that needs real work raises `NotImplementedError` naming what is
missing. What is already real: `__init__` and `from_configs`, because the smoke
test has to get far enough to prove registration works.

Implementation order these stubs imply:

  1. `load_weights`  — nothing runs without weights. See `weight_loader.py`.
  2. `get_kv_spec`   — the runtime allocates KV *before* the first forward, so
                       this gates everything.
  3. `forward`       — embedding -> 40 layers -> norm -> LM head -> sampling.
  4. `bind_kv_cache` — trivial once (2) fixes the layer naming.

The kernels this will call are already written and live in `kernels/`:
`hc_expand`, `hc_collapse`, `hc_attn_out`, `engram_short`, `engram_long`,
`moe_short`, `moe_long`, `l2_attn`, `l20_attn`, `l21_attn`, `l24_attn`,
`l3_attn`, `dspark_attn`, `dspark_moe`, `dspark_markov`, `dspark_confhead`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import KVSpec
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import DeepseekV41Config

_STAGE = "DeepSeek-V4.1-Flash timeshare port is at stage 1b (scaffold only)"


class DeepseekV41ForCausalLM(nn.Module):
    """Single-module scaffold for DeepSeek-V4.1-Flash on Trainium.

    Holds the config and nothing else. No parameters are registered, so this is
    cheap to construct and safe to instantiate on CPU — which is what lets the
    smoke test validate registration without touching a device.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config

        # Deliberately absent until the real implementation lands, so an
        # accidental early call fails on a missing attribute rather than
        # silently computing with an empty module list:
        #   self.embed_tokens / self.layers / self.norm / self.lm_head

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
    ) -> DeepseekV41ForCausalLM:
        """Real, not stubbed: the smoke test needs instantiation to succeed."""
        return cls(DeepseekV41Config.from_configs(hf_config, neuron_config))

    # ── Execution ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata: object | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Execute the model: embedding -> layers -> norm -> LM head -> sampling.

        Signature matches the other causal-LM models in this repo, so the
        runtime's call site needs no special-casing for this architecture.

        Returns:
            Logits when on-device sampling is off; otherwise
            `(sampled_tokens, gathered_logits)`.
        """
        raise NotImplementedError(
            f"{_STAGE}. forward() needs: embedding, the 40 decoder layers over "
            "hc_mult=4 hyper-connection streams, final norm, column-parallel LM "
            "head, and on-device sampling. Kernels are in kernels/."
        )

    # ── KV cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """Declare the KV cache shape for each layer.

        Called by the runtime *before* the first forward, so this is the first
        stub that has to go.

        Open question this must answer: the config declares
        `num_key_value_heads=1` at `head_dim=512` — a single latent KV head,
        MLA-style, not GQA groups. On top of that, KV is produced by a
        compression path rooted at `kv_source_layer_ids=[2, 8, 14, 20]` with
        `compress_ratios` varying per layer, so the mapping from 40 layers onto
        `LayerSpec` entries is a design decision, not a transcription.
        """
        raise NotImplementedError(
            f"{_STAGE}. get_kv_spec() must map num_key_value_heads="
            f"{self.config.num_key_value_heads} at head_dim="
            f"{self.config.head_dim}, kv_source_layer_ids="
            f"{self.config.kv_source_layer_ids} and the per-layer "
            "compress_ratios onto LayerSpec entries. This is a design decision "
            "— see floorplan/MODEL.md and references/INDEX.md."
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind externally-allocated KV tensors onto the attention modules."""
        raise NotImplementedError(
            f"{_STAGE}. bind_kv_cache() is blocked on get_kv_spec() fixing the "
            "layer naming scheme."
        )

    # ── Weights ───────────────────────────────────────────────────────────

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Load the sharded checkpoint. Delegates to `weight_loader`."""
        from .weight_loader import load_weights as _load_weights

        _load_weights(self, checkpoint_path, device, cache_dir)
