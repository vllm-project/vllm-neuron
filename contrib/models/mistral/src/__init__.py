# SPDX-License-Identifier: Apache-2.0
"""Mistral-7B-v0.3 contrib model for vllm-neuron (package ``mistral_neuron``).

Kept import-free: vLLM imports ``mistral_neuron.plugin`` in every process at
startup, and that should not pull in torch or the modeling code. Import
``mistral_neuron.factory`` / ``mistral_neuron.config`` directly instead.
"""
