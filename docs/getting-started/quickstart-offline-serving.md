# Quickstart: Run offline batch inference with vLLM Neuron

<!-- meta: description: Run high-throughput offline batch inference with vLLM on
Neuron on AWS Trainium or Inferentia. -->
<!-- meta: keywords: vLLM, Neuron, offline inference, batch inference, Trainium,
Inferentia, quickstart -->
<!-- meta: date_updated: 2026-05-28 -->
<!-- Content type: procedural-quickstart -->
<!-- Jira: NDOC-181 -->

This quickstart guides you through running offline batch inference with vLLM on
Neuron. When you have completed it, you will have generated text for a batch of
prompts using the `vllm.LLM` Python API.

## Prerequisites

- Completed the [setup guide](setup-guide.md) (vLLM Neuron installed and verified).
- Permission to pull the model you plan to use.
- Familiarity with basic Python.

## Step 1: Run a batch inference job

First, set compilation and execution timeouts. GPT-OSS compilation can exceed
the default 600-second budget:

```bash
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
```

Then run a short Python script that generates completions for three prompts
using GPT-OSS 20B:

```python
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(
        model="openai/gpt-oss-20b",
        tensor_parallel_size=8,
        max_num_seqs=1,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        hf_overrides={"quantization_config": {}},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": "true"},
            },
        },
    )

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, SamplingParams(top_k=10))
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
```

If the script succeeds, you will see each prompt followed by generated text in
the console.

## Common issues

- **Compilation step is slow on first run:** First-run compilation is expected.
  Subsequent runs reuse the cached artifacts.
- **Out-of-memory at startup:** Reduce `max_num_seqs` or `max_model_len`, or use
  a smaller model for the quickstart.
- **Unexpected generation behavior:** Confirm your `SamplingParams` (temperature,
  top_p, max_tokens) match your expectations.
- **Hugging Face model download fails:** Verify your `HF_TOKEN` environment
  variable is set and you have access to the model.

## Clean up

Stop any running scripts with `Ctrl+C`.

## Next steps

- [Quickstart: Online serving](quickstart-online-serving.md) — Serve GPT-OSS 20B
  online with an OpenAI-compatible API.
- [Features guide](../guides/features-guide.md) — Detailed feature coverage.
