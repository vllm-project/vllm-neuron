# vLLM Neuron Plugin

High-performance inference engine for AWS Neuron accelerators, built on vLLM v1 architecture with AWS Neuron SDK 2.26.

## Highlights

- **vLLM V1 Engine Support**: Full compatibility with vLLM's next-generation V1 engine architecture
- **AWS Neuron SDK 2.26**: Built on the latest Neuron SDK with enhanced performance and stability
- **Production Ready**: Optimized for high-throughput inference on AWS Trainium and Inferentia instances
- **Advanced Features**: Chunked prefill, prefix caching, speculative decoding, and LoRA support

## Feature Support

| Feature | Status | Notes |
|:--------|:------:|-------|
| Chunked Prefill | ✅ | Optimized memory usage for long sequences |
| Prefix Caching | ✅ | Accelerated inference for repeated prompts |
| Eagle Speculation | ✅ | Speculative decoding for faster generation |
| Quantization | ✅ | INT8/FP8 quantization support |
| Multimodal | ⚠️ | Beta - Only support Llama 4 |

## Model Support

| Model Family | Status | Supported Sizes | Notes |
|:-------------|:------:|:----------------|-------|
| Llama 3.1/3.3 | ✅ | 8B, 70B, 405B | Full production support |
| Llama 4 | ⚠️ | Scout, Maverick | Beta release |
| Mistral | ✅ | 7B| Optimized for Neuron |
| Mixtral | ✅ | 8x22B | Optimized for Neuron |
| Qwen 2 | ✅ | 7B| Optimized for Neuron |

## Installation

### Prerequisites

- AWS Neuron SDK 2.26 ([Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.26.0/))
- vLLM v0.10.2 ([vLLM Release Notes](https://github.com/vllm-project/vllm/releases))
- Python 3.8+ (compatible with vLLM requirements)
- Supported AWS instances: Inf2, Trn1/Trn1n, Trn2

### Install vLLM

```bash
# Install vLLM from source (v0.10.2+)
git clone -b v0.10.2 https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Install Neuron Plugin

```bash
# Install the Neuron plugin
git clone https://github.com/aws-neuron/vllm-neuron.git
cd vllm-neuron
pip install -e .
```

### Environment Setup

```bash
# Required for V1 engine
export VLLM_USE_V1=1
```

## Quick Start

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
    --port 8000 \
    --disable-log-requests
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

## Advanced Features

### Speculative Decoding (Eagle)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="llama-2/Llama-2-7b-chat-hf",
    tensor_parallel_size=32,
    max_num_seqs=2,
    max_model_len=256,
    enable_prefix_caching=True,
    block_size=32,
    speculative_config={
        "model": "llama-2/EAGLE-llama2-chat-7B",
        "num_speculative_tokens": 5,
        "max_model_len": 256,
        "method": "eagle",
    },
)

prompts = [
    "I believe the meaning of life is",
    "I believe Artificial Intelligence will",
]
sampling_params = SamplingParams(top_k=50, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

## Benchmarking

### Performance Results

| Model | Instance | Throughput (tok/s) | Latency (ms) |
|:------|:---------|:-------------------|:-------------|
| Llama-3.1-8B | trn1.2xlarge |  |  |
| Llama-3.1-70B | trn1.32xlarge |  |  |

## Support

- **Documentation**: [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- **Issues**: [GitHub Issues](https://github.com/aws-neuron/vllm-neuron/issues)
- **Community**: [AWS Neuron Forum](https://forums.aws.amazon.com/forum.jspa?forumID=355)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
