<!--
Use this template for community model contributions to the contrib/ folder.
To select it, append ?template=model_contribution.md to the PR "compare" URL,
or pick it from the template dropdown when opening the PR.

For all other changes (bug fixes, features, CI, etc.), use the default PR
template instead.
-->

## Description

<!-- Provide a brief description of your model contribution. -->

## Model Information

**Model Name:** <!-- e.g., Llama 3.3 -->

**Model Architecture:** <!-- e.g., Decoder-only transformer -->

**Purpose:** <!-- e.g., Text generation, image-to-text, etc. -->

## Checklist

Please ensure your PR includes the following. See
[contrib/README.md](../../contrib/README.md) for detailed guidelines.

### Required components

- [ ] **Accuracy test** at `contrib/models/<model_name>/test/integration/test_model.py`
  - Serves the model through vLLM on Neuron and validates accuracy
    (logit validation, data set evaluation, or equivalent).
- [ ] **`README.md`** for the model, with:
  - [ ] **Usage example** — how to serve the model through vLLM with the Neuron plugin.
  - [ ] **Compatibility matrix** — tested Neuron SDK versions and instance types (like Trn2, Trn3, etc).
  - [ ] **Example checkpoints** — links to compatible checkpoints (e.g., Hugging Face Hub).
  - [ ] **Testing instructions** — command(s) to run the model's test suite.
- [ ] **Source code** under `contrib/models/<model_name>/src/`.

### Optional components

- [ ] **Unit tests** under `contrib/models/<model_name>/test/unit/` (CPU-mode via `VLLM_NEURON_CPU_MODE=1` and/or Neuron-based).
- [ ] **Module tests** under `contrib/models/<model_name>/test/module/` (CPU-mode via `VLLM_NEURON_CPU_MODE=1` and/or Neuron-based).

## Folder structure

Confirm your contribution follows this structure:

```text
/contrib/models/<model_name>/
  README.md
  /src
    <modeling code>
  /test
    /unit           (optional)
    /module        (optional)
    /integration
      test_model.py
```

## Testing

**How did you test this change?**

<!-- Describe your testing approach, instance types used, and configurations. -->

**Test results:**

<!-- Paste relevant test output or link to test runs. -->

## Compatibility

**Tested with:**

- **Neuron SDK version(s):** <!-- e.g., 2.32 -->
- **Instance type(s):** <!-- e.g., Trn2, Trn3 -->
- **Python version:** <!-- e.g., 3.12 -->
- **vLLM Neuron plugin version:** <!-- e.g., 0.24.0.1.1.0 -->

## Additional information

<!-- Known limitations, performance characteristics, or special requirements. -->

---

**By submitting this PR, I confirm that:**

- [ ] I have read and followed the [contributing guidelines](../../contrib/README.md).
- [ ] This is a community contribution and may have limited testing compared to officially-supported models.
- [ ] The code follows best practices and is documented.
- [ ] All required components listed above are included.
