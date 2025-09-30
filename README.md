# VllmNeuronPlugin

** Describe VllmNeuronPlugin here **

## Quick Start Guide

Install this plugin (includes vLLM v0.10.2):

\`\`\`bash
git clone https://github.com/aws-neuron/private-neuronx-vllm.git vllm-neuron
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
\`\`\`

That's it! The plugin will automatically install vLLM version pinned and all required dependencies.

Call vLLM APIs as usual. Neuron plugin discovery takes place automatically.

## Development

See instructions in DEVELOPMENT.md
