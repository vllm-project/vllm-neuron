# SPDX-License-Identifier: Apache-2.0
"""MTP (multi-token prediction / next-n) speculative decoding proposer.

MTP checkpoints (DeepSeek-V3 style, e.g. tencent/Hy3) ship ``n_predict``
extra decoder layers after the target's stack. A draft step embeds the next
token, combines it with the target's final hidden state, runs one MTP layer
and samples with the target's lm_head; later draft steps feed the MTP output
back in place of the target hidden state.

That is the same fused-draft contract as EAGLE3 (KV update pass plus an
unrolled recurrent loop in one NEFF, on-device accepted-token extraction), so
this proposer reuses :class:`EagleProposer` and changes only what differs:

* the target hands over one hidden state per token (its last layer's output,
  before the final norm) instead of three concatenated auxiliary layers;
* the draft layer count comes from ``n_predict`` (vLLM keeps the target's
  ``num_hidden_layers`` in the draft config);
* the draft has no embedding or lm_head of its own and aliases the target's.

The draft model class is resolved by architecture (``HYV3MTPModel`` etc.)
through ``vllm_neuron.model.registry``; out-of-tree models register theirs
with :func:`~vllm_neuron.model.registry.register_neuron_model`.

Draft model contract, on top of the EAGLE3 one (``from_configs(config=...,
start_layer_idx=..., neuron_config=...)``, ``forward(input_ids, positions,
initial_target_hidden_states, attn_metadata, sampling_positions, rank,
raw_sampled_token_ids, prev_sampled_token_ids, prev_num_draft_tokens,
req_indices_per_token) -> (tokens [bs, 1+k], drafts [bs, k], logits)``,
``get_kv_spec``, ``bind_kv_cache``, ``load_weights``):

* ``alias_target_weights(target_model)``: share the target's embedding /
  lm_head (and final norm, which the draft applies to the hidden state it
  receives) instead of loading copies.

Target model contract: ``set_aux_hidden_state_layers(layers)`` with the usual
Neuron capture convention (index i = input of layer i), so index
``num_hidden_layers`` is the last layer's output before the final norm. Return
pre-norm states: returning the post-norm states that also feed the lm_head as
a second graph output corrupted the lm_head rows for batch > 1 on Neuron.
"""

import logging

from torch import nn
from vllm.config import VllmConfig

from vllm_neuron.vllm.spec_decode.eagle import EagleProposer

logger = logging.getLogger(__name__)


class MTPProposer(EagleProposer):
    """Proposer for ``--speculative-config '{"method": "mtp", ...}'``."""

    supported_methods = ("mtp",)

    def __init__(self, vllm_config: VllmConfig, device, on_device_sampling=True):
        super().__init__(vllm_config, device, on_device_sampling)
        draft_hf_config = self.draft_model_config.hf_config
        self.num_mtp_layers = getattr(draft_hf_config, "n_predict", None) or getattr(
            draft_hf_config, "num_nextn_predict_layers", 1
        )
        if self.num_speculative_tokens > 1:
            logger.info(
                "MTP with %d speculative tokens reuses the %d MTP layer(s) "
                "recurrently; later draft positions usually accept less often.",
                self.num_speculative_tokens,
                self.num_mtp_layers,
            )

    def _num_draft_layers(self) -> int:
        return self.num_mtp_layers

    def _target_hidden_states_dim(self, hidden_size: int) -> int:
        return hidden_size

    def get_target_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Ask the target for its last layer's output (before the final norm).

        Index ``num_hidden_layers`` is the input of the (non-existent) next
        layer, i.e. the last decoder layer's output.
        """
        return (self.vllm_config.model_config.hf_config.num_hidden_layers,)

    def alias_target_weights(self, target_model: nn.Module) -> None:
        """Point the draft's embedding and lm_head at the target's weights.

        Called by the runner once both models are loaded; later calls are no-ops.
        """
        if getattr(self, "_target_weights_aliased", False):
            return
        assert self.model is not None
        draft = getattr(self.model, "_orig_mod", self.model)
        draft.alias_target_weights(target_model)
        self._target_weights_aliased = True
        logger.info("MTP draft aliases the target's embedding and lm_head")
