# Plugin-core fallback — speculative decoding

A **backup copy** of the one plugin change this model needs, for the case where
the upstream change is not in your tree.

## When you need this

You do not, if the upstream change has merged. Check first:

```bash
./apply_plugin_core.sh --check
```

It reports "already in this tree" and exits 0 when the registry is present,
whether that came from upstream or from an earlier run of this script.

## What the change is, and why it is not in `src/`

`contrib/README.md` requires a contributed model to stay inside its own folder,
and this model does: nothing under `src/` touches the plugin. The one thing it
cannot do from inside its folder is make `NeuronModelRunner` build its draft
proposer.

The runner's speculative path is wired to the proposers that ship with the
plugin — the method string is matched against `"eagle3"` and `"dflash"`, the
drafter is asserted to be one of those classes, and the draft path assumes a
*separate* draft model with its own weights and KV caches. DeepSeek-V4.1-Flash's
DSpark MTP stages are **layers of the target**: they share its weights, their KV
caches are already named by the target's `get_kv_spec`, and the draft is produced
inside the target forward. There is no draft checkpoint to load, and no hook to
register through.

So the patch adds a proposer registry — `register_proposer`, `resolve_proposer`,
`attach_speculative_config`, a small vLLM-side patch so `SpeculativeConfig`
validation accepts a method that names no draft model, and the runner
generalisation. 3 files, +378 / −24. It is generic: the built-in eagle3 and
dflash paths are unchanged, and a build with no registered proposer behaves
exactly as before.

## Without it

The model still **serves correctly**. `src/__init__.py` guards both imports, so a
plugin without the registry produces:

```
deepseek_v41: DSpark proposer not registered (cannot import name
  'register_proposer' from 'vllm_neuron.vllm.spec_decode'). Speculative
  decoding through the mtp.* stages is unavailable; everything else is
  unaffected.
```

and decode falls back to one token per step. Accuracy is unchanged — the
integration test passes either way — but you lose the ~49% draft acceptance at 5
draft tokens a step that the benchmark numbers in the model README were measured
with.

## Usage

```bash
cd contrib/models/deepseek_v41/plugin-core

./apply_plugin_core.sh --check     # would it apply? is it already there?
./apply_plugin_core.sh             # apply it
./apply_plugin_core.sh --revert    # undo it
```

The script runs from anywhere, resolves the repository root itself, and is safe
to run twice. It refuses to act on a half-patched tree rather than compounding
the problem.

Then confirm:

```bash
python -c "from vllm_neuron.vllm.spec_decode import register_proposer; print('ok')"
```

## Caveats

- The patch was generated against **`f32079d`**. If the plugin has moved, the
  script falls back to a three-way merge and tells you; if that leaves
  conflicts it stops and says so rather than leaving a broken tree.
- It edits files **outside** this model's folder, which is why it is a
  clearly-labelled fallback in its own directory rather than part of `src/`.
  Prefer the upstream change where you can.
- The same change was developed and hardware-validated on
  `vllm-neuron 0.24.0.1.1.0`, where it applies unmodified. This copy is the
  forward-port to `main`, which has **not** been executed — see the model
  README's compatibility section.
