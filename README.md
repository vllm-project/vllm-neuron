# VllmNeuronPlugin

** Describe VllmNeuronPlugin here **

## Quick Start Guide

Install the public vllm (latest release tag v0.10.0) from source:

```
git clone -b v0.10.0 https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

Install this plugin repo:

```
git clone https://github.com/aws-neuron/private-neuronx-vllm.git
cd private-neuronx-vllm
pip install -e .
```

Call vllm APIs as usual. Neuron plugin discovery takes place automatically.

## Documentation

Generated documentation for the latest released version can be accessed here:
<https://devcentral.amazon.com/ac/brazil/package-master/package/go/documentation?name=VllmNeuronPlugin&interface=1.0&versionSet=live>

## Development

See instructions in DEVELOPMENT.md
