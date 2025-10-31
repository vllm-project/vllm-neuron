# SPDX-License-Identifier: Apache-2.0
import logging

import torch
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from vllm import SamplingParams, TextPrompt

logger = logging.getLogger("test_utils")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_vllm_mm_model_inputs(prompt, single_image, sampling_params,
                             model_name):
    """
    Prepare all inputs for multi-modal generation, including:
      1. put text prompt into instruct chat template
      2. compose single text and single image prompt into vLLM's prompt class
      3. prepare sampling parameters
    """
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)

    if input_image is not None:
        inputs["multi_modal_data"] = {"image": input_image}

    sampling_params = SamplingParams(**sampling_params)
    return inputs, sampling_params


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt.replace("[IMG]", "")
        generated_text = output.outputs[0].text
        logger.info(
            f"\n[Prompt]\n{prompt!r}\n[Generated]\n{generated_text!r}\n")
        assert generated_text.strip(
        ), f"Output was empty for prompt: {prompt!r}"
