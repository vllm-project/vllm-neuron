# SPDX-License-Identifier: Apache-2.0
import torch

NEURON_MULTI_MODAL_MODELS = [
    "MllamaForConditionalGeneration", "LlavaForConditionalGeneration",
    "Llama4ForConditionalGeneration"
]

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
