# SPDX-License-Identifier: Apache-2.0
import logging
import os
from math import ceil

import pytest
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig
from vllm import LLM, SamplingParams

# Configure logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_inference_tinyllama_chunked_prefill():
    # Set environment variables
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"

    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    short_prompt_list = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    long_prompt = (
        "Four score and seven years past, our forefathers brought forth upon this continent a "
        "new nation, conceived in the sacred principle of liberty, and dedicated to the "
        "profound proposition that all men are created as equals under heaven. "
        "Now we find ourselves engaged in a great civil war, testing whether this nation, or "
        "indeed any nation so conceived and so dedicated to such principles, can long endure "
        "the trials of time and conflict. We meet today upon a great battlefield of that war, "
        "where brave men gave their last full measure of devotion. We have come to dedicate "
        "a portion of this field as a final resting place for those who perished here, that "
        "this nation might live in freedom. "
        "It is altogether fitting and proper that we should do this solemn duty. But in a "
        "larger sense, we cannot dedicate, we cannot consecrate, we cannot truly hallow this "
        "ground. The courageous men, living and dead, who struggled here, have consecrated "
        "it far above our poor power to add or detract from their sacrifice. The world will "
        "little note, nor long remember, what we say here today, but history can never "
        "forget what they did here. "
        "It is for us, the living, rather, to be dedicated here to the unfinished work for "
        "which they gave the last full measure of devotion. It is for us to be here "
        "dedicated to the great task remaining before us - that from these honored dead we "
        "take increased devotion to their cause - that we here highly resolve that these "
        "dead shall not have died in vain. This nation, under God, shall have a new birth "
        "of freedom, and that government of the people, by the people, for the people, "
        "shall not perish from the earth, but endure for generations yet unborn. "
        "Let us go forth from this hallowed ground with renewed purpose, knowing that their "
        "sacrifice lights our path forward toward a more perfect Union.")
    long_prompt_list = [long_prompt for _ in range(20)]

    sampling_params = SamplingParams(max_tokens=30, top_k=1)

    max_num_seqs = 8
    block_size = 32
    max_model_len = 1024
    llm = LLM(model=model_path,
              max_num_seqs=8,
              max_model_len=1024,
              max_num_batched_tokens=256,
              block_size=32,
              tensor_parallel_size=32,
              enable_prefix_caching=False,
              enable_chunked_prefill=True,
              num_gpu_blocks_override=ceil(max_model_len // block_size) *
              max_num_seqs,
              additional_config={
                  "override_neuron_config": {
                      "is_block_kv_layout": True,
                      "sequence_parallel_enabled": True,
                      "on_device_sampling_config": OnDeviceSamplingConfig(),
                      "chunked_prefill_config": {
                          "max_num_seqs": 8,
                          "kernel_q_tile_size": 128,
                          "kernel_kv_tile_size": 4096,
                      },
                      "skip_warmup": True,
                      "save_sharded_checkpoint": True,
                  },
              })

    for prompt_list in [short_prompt_list, long_prompt_list]:
        outputs = llm.generate(prompt_list, sampling_params)

        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logger.info(
                f"\n[Prompt {i + 1}]\n{prompt!r}\n[Generated]\n{generated_text!r}\n"
            )
            assert generated_text.strip(
            ), f"Output {i + 1} was empty for prompt: {prompt!r}"
