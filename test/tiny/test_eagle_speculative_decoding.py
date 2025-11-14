# SPDX-License-Identifier: Apache-2.0
import logging
import os
from test.utils.eagle_nxdi_util import fix_eagle_draft_for_nxdi
from test.utils.fsx_utils.model_path import resolve_model_dir

import pytest
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_eagle():
    target_id = "meta-llama/Llama-2-7b-chat-hf"
    draft_id = "yuhuili/EAGLE-llama2-chat-7B"

    target_model_dir, _ = resolve_model_dir(target_id)
    draft_model_dir, _ = resolve_model_dir(draft_id)

    logger.info("Target model: %s", target_model_dir)
    logger.info("Draft model: %s", draft_model_dir)

    if not os.path.isabs(draft_model_dir):
        logger.info("Downloading models from HuggingFace")
        from huggingface_hub import snapshot_download

        # Download both models
        target_model_dir = snapshot_download(target_id)
        draft_model_dir = snapshot_download(draft_id)

        logger.info("Downloaded target model to: %s", target_model_dir)
        logger.info("Downloaded draft model to: %s", draft_model_dir)

        # Convert EAGLE draft model for NxD compatibility
        logger.info("Converting EAGLE draft model for Neuron compatibility...")
        fix_eagle_draft_for_nxdi(target_model_dir, draft_model_dir)
        logger.info("EAGLE draft model conversion complete")

    llm = LLM(
        model=target_model_dir,
        tensor_parallel_size=32,
        max_num_seqs=2,
        max_model_len=256,
        enable_prefix_caching=True,
        block_size=32,
        speculative_config={
            "model": draft_model_dir,
            "num_speculative_tokens": 5,
            "max_model_len": 256,
            "method": "eagle",
        },
    )

    prompts = [
        "I believe the meaning of life is",
        "I believe Artificial Intelligence will",
    ]
    sampling = SamplingParams(top_k=50, max_tokens=100)

    logger.info("[EAGLE] Running generate()")
    outputs = llm.generate(prompts, sampling)

    for out in outputs:
        text = out.outputs[0].text
        logger.info("Prompt=%r → Text=%r", out.prompt, text)
        assert isinstance(text, str) and len(text) > 0
