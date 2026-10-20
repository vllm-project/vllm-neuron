# Contrib Folder Guidelines

`vllm-neuron` is a hardware plugin that runs models with [vLLM](https://docs.vllm.ai/)
on AWS Trainium accelerators. It maintains a `contrib` folder for community
contributions. This folder contains models that community members have created
and tested with vLLM on Neuron. You can run or adapt these models in your own
inference workflows.

## Using a model from the `contrib` folder

> **Note:** These models are community contributions and may have limited
> testing or performance optimizations compared to the officially-supported
> model architectures that ship in the plugin under `vllm_neuron/model/`. See
> the `README.md` in each model folder for details about how the model was
> tested, including the Neuron SDK versions and instance types.

Follow these steps to use a model from the `contrib` folder:

1. Clone this repository.
2. Find the model in `contrib` that you intend to serve.
3. The model's `README.md` gives the required `vllm-neuron` version. Check out the relevant branch if the model needs a specific version.
4. Read the model's `README.md` and `test/integration/test_model.py` to
   understand how to run the model.
5. Download a compatible model checkpoint (for example, from the Hugging Face
   Hub — see the model's README for links).
6. Register the model with vLLM and run inference. See the next section. The
   model's `src/` code goes to a different place for each option, so the section
   tells you where to put it.

### How do I register a custom model with vLLM?

A contrib model consists of a config, a factory, and a model implementation.
These are the same building blocks that the officially-supported models use.
The reference implementations are `vllm_neuron/model/llama3/` (dense) and
`vllm_neuron/model/gpt_oss/` (MoE). The full walkthrough (the model, the runner
interface, weight loading, compilation, accuracy validation, and benchmarking)
is in the [model onboarding guide](../docs/model-dev/onboarding-models.md).

vLLM discovers a model through its `ModelRegistry`. The registry maps an
**architecture string** to a factory class. The string must match the
`architectures` field in the model's `config.json`. The plugin registers the
officially-supported models for you at worker startup, through `get_models()`
in `vllm_neuron/model/registry.py`. Contrib models are **not** in `get_models()`
by design. This keeps them opt-in and community-owned. So you register a contrib
model yourself.

Pick the option that matches how you run vLLM. Option 1 keeps the model opt-in
and needs no change to the plugin's core code. Option 2 adds the model to the
plugin's registry.

#### Option 1 — Offline (the `LLM(...)` API)

Use this option when you run inference from your own Python script. First copy
the model's `src/` code into your application, so you can import the factory
class. Then register the factory before you construct the engine. Your script
runs first, so it registers the architecture before vLLM builds the model.

```python
from vllm import ModelRegistry

from your_package.factory import YourModelForCausalLM

# Register before you construct LLM(...).
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)

from vllm import LLM

llm = LLM(model="/path/to/checkpoint")
```

The contrib model's sample test (`test/integration/test_model.py`) uses this
option.

#### Option 2 — Online (the `vllm serve` CLI)

`vllm serve` starts the engine directly, so there is no script of yours to run
the registration call. Register the model the same way the officially-supported
models are registered. Add it to the plugin's registry. The plugin then
registers the architecture at worker startup, so `vllm serve` finds it with no
extra setup.

This option edits the plugin's core code, so it crosses the "community-owned,
opt-in" boundary. It is also the last step when you promote a contrib model to
an officially-supported one.

1. Move the modeling code from `contrib/models/<model_name>/src/` to
   `vllm_neuron/model/<model_name>/`.
2. Add the architecture to `get_models()` in `vllm_neuron/model/registry.py`.

   ```python
   from .<model_name> import YourModelForCausalLM

   def get_models() -> list[tuple[str, type]]:
       models = [
           # ... existing models ...
           ("YourModelForCausalLM", YourModelForCausalLM),
       ]
       return models
   ```

3. Run the server. The plugin registers the architecture for you.

   ```bash
   vllm serve /path/to/checkpoint
   ```

This registration also covers the offline `LLM(...)` API, so you do not need the
Option 1 call after you register the model here.

For the end-to-end procedure, including compilation, CPU-mode iteration
(`VLLM_NEURON_CPU_MODE=1`), and accuracy validation, follow the
[model onboarding guide](../docs/model-dev/onboarding-models.md). For general
vLLM-on-Neuron usage and `VLLM_NEURON_*` configuration, see the top-level
`README.md` and `AGENTS.md`.

## Contribution guidelines

This section covers guidelines for contributing a model. For detailed guidelines about contributing to the `vllm-neuron` project, refer to [CONTRIBUTING.md](../CONTRIBUTING.md)

### Folder structure

Each model has a README, a source folder, and a test folder. The README
describes the model and the vLLM / Neuron SDK versions it was tested with.

```text
/contrib
  README.md              (this doc)
  /models
    /template                  (example/starting point — copy this)
      README.md
      /src                     (self-contained modeling code)
        __init__.py
        config.py
        factory.py
        model.py
        /kernels                 (optional — any model-specific NKI kernels)
      /test
        /unit                  (optional)
        /module                (optional — per-component accuracy tests)
        /integration
          test_model.py        (accuracy test — serves the model through vLLM)
    /<your_model_name>
      README.md
      /src
        __init__.py
        config.py
        factory.py
        model.py
        /kernels                 (optional)
      /test
        /unit                  (optional)
        /module                (optional)
        /integration
          test_model.py
```

### How do I contribute?

We encourage you to open a pull request to contribute modeling code. Copy the
`contrib/models/template/` folder as your starting point and fill it in. Your
PR should include:

1. **An accuracy test** at `contrib/models/<model_name>/test/integration/test_model.py`
   that validates your model by serving it through vLLM on Neuron and checking
   the generated output (for example, logit validation or dataset evaluation). The template provides a sample you can adapt.

2. **A `README.md`** for your model that includes:
   1. A usage example showing how to serve the model through vLLM with the
      Neuron plugin.
   2. A compatibility matrix showing which Neuron instance types (Trn2, Trn3, etc) and Neuron SDK versions the model was tested with.
   3. Links to compatible checkpoints (for example, on the Hugging Face Hub).
   4. Instructions for running the tests for the model.

3. **(Optional) Unit tests** for your modeling components under
   `contrib/models/<model_name>/test/unit/`. To improve code quality you can add
   classic CPU-mode unit tests (run with `VLLM_NEURON_CPU_MODE=1`) and/or tests
   that run on Neuron instances. See the plugin's `test/README.md` for the
   progressive CPU-to-hardware testing workflow.

4. **(Optional) Module tests** for individual model components under
   `contrib/models/<model_name>/test/module/`. Module tests validate one
   component (attention, MLP, RMSNorm, and so on) in isolation against a
   HuggingFace reference, before you assemble the full model. A component bug
   is much easier to find at this stage than in a full-model run. They run on
   CPU (`VLLM_NEURON_CPU_MODE=1`) or on device. See the
   [module test guidelines](../docs/design/accuracy/module_test_guidelines.md)
   for the three-way accuracy pattern and threshold guidance.

After you open a PR, the Neuron team will review it and request changes as
needed.

### What goes in `src/`?

The modeling code for your model lives in `contrib/models/<model_name>/src/`.
Structure it however makes sense for your model, but the officially-supported models follow a common layout that is a good starting point. The reference implementations are
`vllm_neuron/model/llama3/` (dense) and `vllm_neuron/model/gpt_oss/` (MoE); see
the [model onboarding guide](../docs/model-dev/onboarding-models.md) for what
each file does.

Typical files:

| File | Required? | Purpose |
| --- | --- | --- |
| `__init__.py` | Yes | Marks the folder as a package and exports the factory class. |
| `config.py` | Yes | Model config dataclass with a `from_configs(hf_config, neuron_config)` classmethod that parses the HuggingFace `PretrainedConfig`. |
| `factory.py` | Yes | Factory `nn.Module` registered with vLLM's `ModelRegistry`; validates config and selects the implementation via `from_configs`. |
| `model.py` | Yes | The full model implementation (attention, MLP, backbone, `ForCausalLM` with `forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`). |
| `weight_loaders*.py` | If needed | Checkpoint-to-parameter transforms (sharding, fusion), especially per quantization scheme. |
| `quantization.py` | If applicable | Quantization spec parsing and quantized variant selection. |
| `model_<variant>.py` | If applicable | Additional implementation variants (for example a BF16 vs. FP8/MXFP4 split, as in `gpt_oss/model_bf16.py` and `model_mxfp4.py`). |
| `kernels/` (or `*_kernel.py`) | If you write kernels | Any custom NKI or other device kernels your model needs. See the self-containment rule below. |
| `README.md` | Yes | Per-model architecture and feature-status card (this is separate from the model folder's top-level `README.md` described above). |

#### Modeling code must be self-contained

A contrib model must be **self-contained within its own
`contrib/models/<model_name>/` folder**. This keeps community contributions
isolated, easy to review, and simple to copy into an application.

- **Keep model-specific code local.** Anything written specifically for your
  model, including any custom kernels, must live inside your model folder, not
  in the shared plugin packages.
- **Depend only on stable, shared building blocks.** You may import the plugin's
  general-purpose utilities and operators (for example `vllm_neuron.functional`,
  `vllm_neuron.nn`, `vllm_neuron.model.neuron_config`, weight-loading helpers)
  and vLLM's public APIs. Avoid importing from another model's folder — if two
  contrib models need the same helper, copy it into each, or raise it with the
  Neuron team for promotion into the shared packages.
- **No cross-model or out-of-folder writes.** A contrib model should not modify
  files outside its own folder. The one intended touch-point outside the folder
  is registering the architecture with vLLM's `ModelRegistry`. A user does this
  from their own script or plugin (see "How do I register a custom model with
  vLLM?"), not by editing plugin core code.

### Relevant Documentation

| Docs | What's inside |
|---|---|
| [`docs/model-dev/`](../docs/model-dev/) | Model onboarding, CPU development, etc. |
| [`docs/design/`](../docs/design/) | Architecture deep dives (parallelism, DI, prefix caching, etc.) |
| [`Agentic port`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-agentic-development/developer_guides/neuron-framework-autoport-vllm-neuron.html#neuron-framework-autoport-vllm-neuron) | Guide to use coding agents and skills to port models to `vllm-neuron`. |
