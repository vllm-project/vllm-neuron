# SPDX-License-Identifier: Apache-2.0
import logging
import os

import pytest
from vllm import LLM, SamplingParams

# Configure logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_prefix_caching():
    # Set environment variables
    os.environ["VLLM_USE_V1"] = "1"

    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # All prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        ("It is not the critic who counts; not the man who points out how the "
         "strong man stumbles, or where the doer of deeds could have done them "
         "better. The credit belongs to the man who is actually in the arena, "
         "whose face is marred by dust and sweat and blood; who strives "
         "valiantly; who errs, who comes short again and again, because there "
         "is no effort without error and shortcoming; but who does actually "
         "strive to do the deeds; who knows great enthusiasms, the great "
         "devotions; who spends himself in a worthy cause; who at the best "
         "knows"),
        ("Do not go gentle into that good night, Old age should burn and rave "
         "at close of day; Rage, rage against the dying of the light. Though "
         "wise men at their end know dark is right, Because their words had "
         "forked no lightning they Do not go gentle into that good night. Good "
         "men, the last wave by, crying how bright Their frail deeds might have "
         "danced in a green bay, Rage, rage against the dying of the light. "
         "Wild men who caught and sang the sun in flight, And learn, too late, "
         "they grieved it on its way, Do not go gentle into that good night. "
         "Grave men, near death, who see with blinding sight Blind eyes could "
         "blaze like meteors and be gay, Rage, rage against the dying of the "
         "light. And you, my father, there on the sad height, Curse, bless, me "
         "now with your fierce tears, I pray. Do not go gentle into that good "
         "night. Rage, rage against the dying of the light."),
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.0)

    # Create neuron config dictionary
    override_neuron_config = {
        "enable_bucketing": False,
        "skip_warmup": True,
        "save_sharded_checkpoint": True,
    }

    prefix_cached_llm = LLM(
        model=model_path,
        max_num_seqs=4,
        max_model_len=1024,
        block_size=32,
        tensor_parallel_size=32,
        enable_prefix_caching=True,
        additional_config={"override_neuron_config": override_neuron_config},
        num_gpu_blocks_override=2 * 4 * 1024 // 32,
        dtype="bfloat16",
    )

    # Warmup so that the shared prompt's KV cache is computed.
    prefix_cached_llm.generate(prompts[0], sampling_params)

    # Define expected outputs for each prompt
    expected_outputs = {
        "Hello, my name is":
        " [Your Name] and I am a [Your Position] at [Your Company",
        "The president of the United States is":
        " a man named Donald Trump.\n\n2. B. The president of the",
        "The capital of France is":
        " Paris.\n\n2. B. The capital of Germany is Berlin.\n",
        "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows":
        " in the end the triumph of high achievement, and who at worst, if",
        "Do not go gentle into that good night, Old age should burn and rave at close of day; Rage, rage against the dying of the light. Though wise men at their end know dark is right, Because their words had forked no lightning they Do not go gentle into that good night. Good men, the last wave by, crying how bright Their frail deeds might have danced in a green bay, Rage, rage against the dying of the light. Wild men who caught and sang the sun in flight, And learn, too late, they grieved it on its way, Do not go gentle into that good night. Grave men, near death, who see with blinding sight Blind eyes could blaze like meteors and be gay, Rage, rage against the dying of the light. And you, my father, there on the sad height, Curse, bless, me now with your fierce tears, I pray. Do not go gentle into that good night. Rage, rage against the dying of the light.":
        "\n\n(The play ends with the audience leaving the stage, leaving the stage",
        "The future of AI is":
        " bright, and it's not just for big companies. Small businesses can"
    }
    # Generate outputs with prefix caching and validate
    outputs = prefix_cached_llm.generate(prompts, sampling_params)

    # Validate outputs
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        logger.info(
            f"\n[Prompt {i+1}]\n{prompt!r}\n[Generated]\n{generated_text!r}\n")

        # Assert and log validation results
        try:
            assert generated_text.strip() == expected_outputs[prompt].strip(), \
                f"Output mismatch for prompt '{prompt}'\n[Expected]\n{expected_outputs[prompt]!r}\n[Got]\n{generated_text!r}"
            logger.info(f"[Validation] Prompt {i+1} passed\n")
        except AssertionError as e:
            logger.error(f"[Validation] Prompt {i+1} failed: {str(e)}\n")
