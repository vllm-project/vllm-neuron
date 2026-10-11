# SPDX-License-Identifier: Apache-2.0
"""The DSpark proposer: this model's own MTP stages as a speculative drafter. NOT IMPLEMENTED.

The three `mtp.*` stages -- checkpoint layers 40-42, fed by the attention inputs tapped from
layers 37/38/39 -- already run on every forward and already produce a draft. `model.py`'s
forward computes `drafts = [self.forward_spec(...) for ...]` unconditionally, stores them on
`self.last_draft`, and returns `drafts[0]` when `spec_decode_metadata is not None`. What is
missing is the other half: nothing asks for those drafts, so the tokens the model emits come
from the backbone and the draft is computed and thrown away.

It was not missing for lack of trying. `NeuronModelRunner` accepted exactly one speculative
method, `eagle3`, whose `EagleProposer` loads a *separate* draft model from
`speculative_config.draft_model_config`; there is no second model here. The runner now also
accepts a proposer registered by a model package (`vllm_neuron.vllm.spec_decode.PROPOSERS`),
which is what this file is for, and which is why it lives here: `vllm_neuron/vllm/` is outside
the loop's editable scope and this file is inside it.

**What the runner calls.** Duck-typed; `spec_decode/eagle.py` is the reference implementation,
and the consumption side -- `_build_spec_decode_metadata`, the scheduler's
`scheduled_spec_decode_tokens`, `RejectionSampler`, the accepted/rejected bookkeeping -- is
already generic and needs nothing from here.

    __init__(vllm_config, device, on_device_sampling)
    target_model_provider      set by the runner: a callable returning the loaded target model,
                               because the proposer is built before any model exists
    num_speculative_tokens     3 here, one per MTP stage (`config.num_nextn_predict_layers`)
    model                      the module the runner may bind caches on; for an in-model
                               drafter this is the target itself
    load_model(target_hidden_size)   no weights to load -- the stages came with the checkpoint.
                               The place to pick up `target_model_provider()`
    warmup(...)                drive one `propose` with synthetic inputs so its NEFF exists
    graph_extract(...)         register the draft graph for ahead-of-time compilation
    propose(...) -> (draft_token_ids, drafts_only)
    validate_same_kv_cache_group(kv_cache_config)   trivially satisfied: `get_kv_spec` already
                               returns `mtp.{stage}.attn` in the same group as the 40 backbone
                               layers, and `bind_kv_cache` binds them

**Two things that are already done**, so they are not part of this work: the MTP stages own
paged KV (`model.get_kv_spec`), and their weights load from checkpoint layers 40-42
(`add_block(f"mtp.{stage}", ...)` in the shard plan).

**Registering it.** `__init__.py` calls
`vllm_neuron.vllm.spec_decode.register_proposer("deepseek_mtp", DSparkProposer)` at import, the
same hook pattern `require_segmented_prefill` already uses. The runner then builds it for
`speculative_config.method == "deepseek_mtp"`, and the port's `check_and_update_config` hook
can supply that config so the gate's own verifier command exercises it with no change to the
verifier.

**Proving it works**, as opposed to compiling: a run with speculation on should report accepted
draft tokens, and the output must be *identical* to the same prompts without it -- speculative
decoding is an exactness-preserving optimisation, so any divergence is a bug in the proposer,
not an acceptable cost. `last_draft` being non-None already proves the stages execute; it
proves nothing about consumption.
"""

from __future__ import annotations


class DSparkProposer:  # noqa: D101 - not implemented yet; see the module docstring
    def __init__(self, vllm_config, device, on_device_sampling) -> None:  # noqa: ANN001
        raise NotImplementedError(
            "DSparkProposer is a stub. The MTP stages compute a draft on every forward "
            "(model.py's forward_spec) but nothing consumes it; this is the half that would. "
            "See the module docstring for the interface NeuronModelRunner expects."
        )
