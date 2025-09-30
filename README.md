# vLLM Neuron Plugin (Beta)

> **⚠️ Important**: This is beta preview of the vLLM Neuron plugin. For a more stable experience, consider using the [AWS Neuron vllm fork](https://github.com/aws-neuron/upstreaming-to-vllm/releases/tag/2.26.0)

The vLLM Neuron plugin (vllm-neuron) is a backend extension that integrates AWS Neuron accelerators with vLLM. Built on [vLLM's Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
, it enables optimizing existing vLLM workflows on AWS Neuron.

## Feature/Model Support

| Feature/Model | Status | Notes |
|:--------|:------:|-------|
| Prefix Caching | 🟢 |  |
| Eagle Speculation | 🟢 |   |
| Quantization | 🟢 | INT8/FP8 quantization support |
| Chunked Prefill | 🚧 |  |
| Multimodal | 🚧 |  Llama 4 support |
| Llama 3.1/3.3 | 🟢 | 8B, 70B, 405B |
| Llama 4 | 🚧 | Scout, Maverick |
| Qwen 2 | 🟢 | 7B|

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🚧 WIP: Under active development.

## Installation

### Prerequisites

- AWS Neuron SDK 2.26 ([Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.26.0/))
- vLLM v0.10.2 ([vLLM Release Notes](https://github.com/vllm-project/vllm/releases))
- Python 3.8+ (compatible with vLLM requirements)
- Supported AWS instances: Inf2, Trn1/Trn1n, Trn2

### Quick Start Guide

Install this vLLM plugin:

```bash
git clone https://github.com/aws-neuron/private-vllm-neuron.git vllm-neuron
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

That's it! The plugin will automatically install vLLM version pinned and all required dependencies.

### Basic Usage

```python
import os
from vllm import LLM, SamplingParams

# Enable V1 engine
os.environ["VLLM_USE_V1"] = "1"

# Initialize the model
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tensor_parallel_size=32,
    max_num_seqs=4,
    max_model_len=1024,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    additional_config={
        "override_neuron_config": {
            "skip_warmup": True,
        },
    }
)

# Generate text
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.0)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### OpenAI-Compatible API Server

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --tensor-parallel-size 32 \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --port 8000 
```

## Configuration

### Neuron-Specific Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `tensor_parallel_size` | Number of Neuron cores | 1 |
| `max_model_len` | Maximum sequence length | 128 |
| `enable_chunked_prefill` | Enable chunked prefill | False |
| `enable_prefix_caching` | Enable prefix caching | False |
| `additional_config` | Custom Neuron configuration | {} |

### Performance Tuning

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=8192,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    max_num_seqs=64,
    additional_config={"override_neuron_config": "skip_warmup": True},
)
```


## Support

- **Documentation**: [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- **Issues**: [GitHub Issues](https://github.com/vllm-project/vllm-neuron/issues)
- **Community**: [AWS Neuron Forum](https://forums.aws.amazon.com/forum.jspa?forumID=355)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
