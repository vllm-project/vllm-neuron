# Plugin-core patch

This model folder is self-contained: nothing outside `contrib/models/deepseek_v41_trn3/` is modified by the contribution. The few changes the model needs inside the `vllm-neuron` plugin itself ship here as a patch, `deepseek-v41-plugin-core.patch`, with `apply_plugin_core.sh` to apply, check or revert it.

## Base commit

The patch was generated against [`vllm-project/vllm-neuron@fe7bf43`](https://github.com/vllm-project/vllm-neuron/commit/fe7bf43c9ae9d0e2731a0765496186c11e6152bd) (`main`, vllm-neuron 0.29.0.1.2.0). At that commit `vllm_neuron/` is byte-identical to the package installed in the DLAMI venv `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`, so the patch applies cleanly to either. On a later `main` the script falls back to a three-way merge and stops if that leaves conflicts.

## What it changes

7 files under `vllm_neuron/`, +605 / −29. Each change is commented at its site with the reason for it.

| File | Change |
| --- | --- |
| `vllm/spec_decode/__init__.py` | A proposer registry: `register_proposer`, `resolve_proposer`, `attach_speculative_config`. The DSpark MTP stages draft inside the target model, so the model registers its own proposer instead of using the built-in `eagle3` / `dflash` ones. |
| `vllm/patches/in_model_proposer_patch.py` (new) | Lets `SpeculativeConfig` accept an MTP method with no separate draft model, installed only when a proposer is registered. |
| `vllm/worker/neuron_model_runner.py` | Builds a registered proposer; skips the separate-draft-model steps (compile wrapper, capture, KV binding, KV spec) for an in-model drafter; declares `RUNNER_CAPABILITIES` (`device_draft_futures`: the async-spec transition cache for device-side drafts; `host_step_inputs`: host positions, slot mapping and block table for models that plan the step on the host); passes a `sampling_hint` to models that accept one; corrects positions after an empty async step; repairs malformed synchronous draft rows; `VLLM_NEURON_DEBUG_MODE` no longer turns off the graph-capture backend. |
| `vllm/core/scheduler.py` | `VLLM_NEURON_ASYNC_MAX_TOKENS_HOLDBACK`: under async scheduling, holds back a request's first decode while its prefill is in flight, and a lone request's step that may already reach `max_tokens`. |
| `parallel/neuron_parallel_state.py` | The Trn2 8x8 rank mesh is applied on the `trn2` family only. Its device pairs are not linked on Trn3, where the runtime refuses the group. |
| `vllm/patches/dynamo_recompile_reason_patch.py` (new), `__init__.py` | Makes torch 2.13's recompilation-reason report best-effort, so a diagnostic cannot kill the forward with a `KeyError`. |

The benchmark and integration-test results in the model README were measured with this patch applied. Without it, speculative decoding and async scheduling are off, and the configuration is not the tested one.

## Usage

From anywhere:

```bash
contrib/models/deepseek_v41_trn3/plugin-core/apply_plugin_core.sh --check    # applied? would it apply?
contrib/models/deepseek_v41_trn3/plugin-core/apply_plugin_core.sh            # apply to this checkout's vllm_neuron/
contrib/models/deepseek_v41_trn3/plugin-core/apply_plugin_core.sh --revert   # undo
```

By default the script patches the `vllm_neuron/` of the git checkout it lives in. `--target DIR` patches the `vllm_neuron/` package under `DIR` instead, for example an installed copy in a venv's `site-packages`. The script is idempotent: it reports "already applied" and exits 0 when the patch is in place.

Python must import the patched copy. Running from the repository root does that: `python -m ...` and `python -m pytest` put the current directory first on `sys.path`, ahead of the venv's installed `vllm_neuron`, and vLLM's spawned workers inherit that path. Check it from the repository root:

```bash
python -c "import vllm_neuron.vllm.spec_decode as s; print(s.__file__, hasattr(s, 'register_proposer'))"
```

The output should be a path inside the checkout and `True`.
