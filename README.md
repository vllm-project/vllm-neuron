# vLLM Neuron Plugin (Beta)

The vLLM Neuron plugin is the recommended serving solution for large language models on AWS Trainium. It extends [vLLM](https://docs.vllm.ai/) with a Neuron backend, providing the same vLLM APIs and configuration you are already familiar with.

> **If you are using vLLM with NxD Inference (0.5.x plugin version),** see the [release-0.5.3 branch](https://github.com/aws-neuron/private-vllm-neuron/tree/release-0.5.3) and the [NxD Inference + vLLM documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html). To migrate to the plugin based on 0.21.0+, see the [migration guide](docs/getting-started/migration-nxdi-to-vllm-neuron.md).

## Installation

```bash
git clone -b release-0.24.0.1.1.0 https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

This installs the vLLM Neuron plugin from source along with vLLM and the required Neuron SDK packages. See [Version](#version) for the full compatibility matrix.

**For detailed environment setup and instance selection**: See the [Setup guide](docs/getting-started/setup-guide.md).

## Quick Start

Run the following command from a shell terminal in the local or instance environment (`vllm` must be in the path) where vLLM is configured:

```bash
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4
```

## Supported Models

Models listed below are tested end-to-end on Neuron hardware with correctness validation and performance benchmarking.

| Family | Model | Type | Instance | Correctness | Perf Test | Perf Tuning |
|---|---|---|---|---|---|---|
| Llama 3 | 1B, 8B, 70B | Text | Trn2, Trn3 | ✅ | ✅ | In progress |
| GPT-OSS | 20B, 120B | Text | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3-VL | 32B (Dense) | Multimodal | Trn2, Trn3 | ✅ | ✅ | In progress |
| Qwen3-Embedding | 8B | Embedding | Trn2, Trn3 | ✅ | ✅ | In progress |

- **Correctness** — Accuracy validation passing (logit comparison, dataset accuracy validation)
- **Perf Test** — Performance benchmark tests tracked across releases
- **Perf Tuning** — Active optimization work being done

> **Note:** This list includes only the models currently supported in the latest plugin version and does not include models under development or on roadmap.

Performance improvements are delivered incrementally with each Neuron release.

## Features

Feature support is at the framework level. See the model cards in [`docs/model-recipes/`](docs/model-recipes/) for per-model feature availability and the [features guide](docs/guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Serving** | Continuous batching | ✅ |
| | OpenAI-compatible APIs | ✅ |
| | Embedding API (`/v1/embeddings`) | ✅ |
| | Streaming | ✅ |
| | Structured outputs (JSON, regex, grammar) | ✅ |
| | Tool calling | ✅ |
| | LoRA adapters | ❌ |
| | Weight reload API | ❌ |
| | Sleep mode (sleep/wake_up API) | ❌ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Sequence parallelism (SP) | ✅ |
| | Data parallelism (DP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Vision encoder parallelism | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ✅ |
| **Performance** | Paged KV cache | ✅ |
| | Prefix caching | ✅ |
| | Segmented prefill | ✅ |
| | Chunked prefill (mixed batching) | ❌ |
| | Disaggregated inference (1P1D, xPyD) | ✅ |
| | Async scheduling | ✅ |
| | On-device sampling (top-k, top-p, temperature) | ✅ |
| | KV cache offloading | ❌ |
| | Disaggregated encoder (EPD) | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | torch.compile (native PyTorch) | ❌ |
| | Compile cache (local/remote) | ✅ |
| | CPU compilation | ✅ |
| **Speculative Decoding** | EAGLE3 | ✅ |
| | DFlash | Preview |
| | MTP | ❌ |
| **Quantization** | BF16 | ✅ |
| | FP8 | ✅ |
| | Quantized KV cache (FP8) | ✅ |
| | MXFP4 weights (Trn3) | ✅ |
| | MXFP8 weights (Trn3) | ✅ |
| **Multimodal Inputs** | Images + text | ✅ |
| | Videos + text | ✅ |
| | Audio | ❌ |
| **Observability** | Production metrics (`/metrics` endpoint) | ✅ |
| | Profiler API | ✅ |

**Status legend:**

- ✅ Supported — integrated and tested for at least one model
- ❌ Not supported — may be considered for future releases

## Documentation

Full documentation sources are found in [`docs/`](docs/). These docs are published to [awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html).

**Deploying supported models:**

| Folder | What's inside |
|---|---|
| [`docs/getting-started/`](docs/getting-started/) | Setup, quickstarts, migration from NxD Inference |
| [`docs/guides/`](docs/guides/) | Features guide, configuration reference, metrics, profiling |
| [`docs/model-recipes/`](docs/model-recipes/) | Per-model deployment recipes |
| [`docs/tutorials/`](docs/tutorials/) | End-to-end walkthroughs (DI, prefix caching, etc.) |

**Implementing new models & optimizing performance:**

| Folder | What's inside |
|---|---|
| [`docs/model-dev/`](docs/model-dev/) | Model onboarding, CPU development, etc. |
| [`docs/design/`](docs/design/) | Architecture deep dives (parallelism, DI, prefix caching, etc.) |

## Version

Starting with 0.21.0, the vLLM Neuron plugin version follows the format `<vLLM version>.<plugin version>` (e.g., `0.21.0.1.0.0` means vLLM 0.21.0, plugin version 1.0.0). Model implementations live directly in the plugin and NxD Inference is no longer a dependency. Customers on 0.5.x can migrate using the [migration guide](docs/getting-started/migration-nxdi-to-vllm-neuron.md).

| vLLM-Neuron Plugin | vLLM Version | Neuron SDK | NxD Inference | Instance Support | Status | Documentation |
|---|---|---|---|---|---|---|
| 0.24.0.1.1.0 (latest) | 0.24.0 | 2.32 | Not required | Trn2, Trn3 | Beta | [vLLM Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html) |
| 0.21.0.1.0.0 | 0.21.0 | 2.31 | Not required | Trn2, Trn3 | Beta | [vLLM Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html) |
| 0.5.3 | 0.16.0 | 2.31 | 0.10.x | Trn1, Trn2, Trn3, Inf2 | Maintenance | [NxDI + vLLM on Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html) |

## Issues

Report bugs or request features: [GitHub Issues](https://github.com/vllm-project/vllm-neuron/issues)

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact <opensource-codeofconduct@amazon.com>.

## License

Apache-2.0
