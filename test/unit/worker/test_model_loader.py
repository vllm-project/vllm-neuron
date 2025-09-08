# test/unit/worker/test_model_loader.py
import pytest
from unittest.mock import Mock
import torch
from transformers import PretrainedConfig
from neuronx_vllm_plugin.worker.neuronx_distributed_model_loader import get_neuron_model

def test_get_neuron_model(mocker):
    scheduler_config = Mock()
    scheduler_config.max_model_len = 2048
    scheduler_config.max_num_seqs = 32
    scheduler_config.chunked_prefill_enabled = False
    scheduler_config.max_num_batched_tokens = 4096

    cache_config = Mock()
    cache_config.block_size = 8
    cache_config.num_gpu_blocks_override = None
    cache_config.enable_prefix_caching = False

    parallel_config = Mock()
    parallel_config.tensor_parallel_size = 1

    model_config = Mock()
    model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"],
        num_key_value_heads=32,
        head_dim=64,
        vocab_size=32000,
        model_type="llama"
    )
    model_config.model = "meta-llama/Llama-2-7b-hf"
    model_config.dtype = torch.float32
    model_config.override_neuron_config = None

    # Mock the model loading
    mock_model = Mock()
    mock_model.config.neuron_config = Mock()
    mock_causal_lm = Mock()
    mock_causal_lm.model = mock_model
    
    mocker.patch('neuronx_vllm_plugin.worker.neuronx_distributed_model_loader.NeuronCausalLM',
                 return_value=mock_causal_lm)

    model = get_neuron_model(
        model_config,
        cache_config,
        parallel_config,
        scheduler_config,
        Mock()
    )
    
    assert model is not None