# Contrib Model Template: Llama (Text)

This folder is an example/template for contributing a model to `vllm-neuron`.
Copy this folder to `contrib/models/<your_model_name>/`, replace the modeling
code under `src/`, adapt the sample test under `test/`, and fill in the sections
below for your model.

The example describes serving a Llama text model (Llama 3 collection) with vLLM
on AWS Trainium through the `vllm-neuron` plugin.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level `README.md`
for installation). The plugin auto-registers with vLLM, so once your model
architecture is registered (see
[contrib/README.md](../../README.md) — "How do I use a custom model
with vLLM?"), you can serve and query it like any other vLLM model.

### Serve with the OpenAI-compatible server

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --tensor-parallel-size 32 \
    --max-model-len 8192 \
    --max-num-seqs 1
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# The vllm-neuron plugin registers the Neuron platform automatically.
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    tensor_parallel_size=32,
    max_model_len=8192,
    max_num_seqs=1,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

This matrix shows which Neuron instance types and Neuron SDK versions this model
was tested with. Update it with your own results.

| Instance / SDK version | 2.32        | 2.31 and earlier |
| ---------------------- | ----------- | ---------------- |
| Trn3                   | Not tested  | Not tested       |
| Trn2                   | Working     | Not tested       |

## Example Checkpoints

- <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
- <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>
- <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>

## Testing

The following command runs the integration test, which serves the model through
vLLM on Neuron and validates its accuracy.

```bash
pytest contrib/models/template/test/integration/test_model.py \
    -v -s --timeout=3600 
```

### Optional: unit tests

To run any optional CPU-mode unit tests you add under `test/unit/`:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/template/test/unit/ 
```

### Optional: module tests

Module tests validate one model component (for example attention, MLP, or
RMSNorm) in isolation, before you assemble the full model. They compare the
component output against a HuggingFace reference.

Add module tests under `test/module/`. They run on CPU (no Neuron hardware) or
on device. Run them in CPU mode with:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/template/test/module/ \
    -v -s --timeout=600 
```

For an example, see the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md).
