# SPDX-License-Identifier: Apache-2.0
import logging
from test.utils.fsx_utils.model_path import resolve_model_dir
from test.utils.mm_utils import get_vllm_mm_model_inputs, print_outputs
from typing import Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger("vllm_integration")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def _init_vllm(
    model_name_or_path: str,
    tp_degree: int,
    batch_size: int,
    max_len: int,
    block_size: int,
    enable_prefix_caching: bool,
    dtype: str,
    override_config: Optional[dict],
) -> LLM:
    model_name_or_path, _ = resolve_model_dir(model_name_or_path)
    logger.info(
        "Initializing vLLM %s (tp=%d, batch=%d, max_len=%d, block_size=%d, enable_prefix_caching=%s, dtype=%s)",
        model_name_or_path,
        tp_degree,
        batch_size,
        max_len,
        block_size,
        enable_prefix_caching,
        dtype,
    )
    return LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        swap_space=0,
        tensor_parallel_size=tp_degree,
        max_num_seqs=batch_size,
        max_model_len=max_len,
        block_size=block_size,
        additional_config={"override_neuron_config": override_config or {}},
    )


def vllm_integ_test_llama4(
    title: str,
    model_name_or_path: str,
    max_model_len: int,
    batch_size: int,
    tp_degree: int,
    block_size: int,
    enable_prefix_caching: bool,
    dtype: str,
    top_k: int = 1,
    override_neuron_config: Optional[dict] = None,
) -> None:
    logger.info("[%s] Starting vLLM offline inference integration test", title)

    from vllm.assets.image import ImageAsset

    # Model Inputs
    PROMPTS = [
        "What is in this image? Tell me a story",
        "What is the recipe of mayonnaise in two sentences",
        "How many animals do you see? What type?",
        "What is the capital of Italy famous for?",
    ]
    IMAGES = [
        ImageAsset("blue_flowers").pil_image, None,
        ImageAsset("bird").pil_image, None
    ]
    SAMPLING_PARAMS = [
        dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256)
        for _ in range(len(PROMPTS))
    ]

    llm = _init_vllm(
        model_name_or_path,
        tp_degree,
        batch_size,
        max_model_len,
        block_size,
        enable_prefix_caching,
        dtype,
        override_neuron_config,
    )

    batched_inputs = []
    batched_sample_params = []
    for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_vllm_mm_model_inputs(pmpt,
                                                           img,
                                                           params,
                                                           model_name="llama4")
        # test batch-size = 1
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)
        batched_sample_params.append(sampling_params)

    # test batch-size = 4
    outputs = llm.generate(batched_inputs, batched_sample_params)
    print_outputs(outputs)


def vllm_integ_test(
    title: str,
    model_name_or_path: str,
    n_positions: int,
    max_batch_size: int,
    tp_degree: int,
    block_size: int,
    enable_prefix_caching: bool,
    dtype: str,
    top_k: int = 1,
    override_neuron_config: Optional[dict] = None,
) -> None:
    logger.info("[%s] Starting vLLM offline inference integration test", title)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = _init_vllm(
        model_name_or_path,
        tp_degree,
        max_batch_size,
        n_positions,
        block_size,
        enable_prefix_caching,
        dtype,
        override_neuron_config,
    )
    outputs = llm.generate(prompts, SamplingParams(top_k=top_k))

    if top_k == 1:
        # Define expected outputs
        expected_outputs = {
            "Hello, my name is":
            " and I'm writing you today to learn more about the 201",
            "The president of the United States is":
            " a man who has been accused of sexual assault by multiple women. He has been",
            "The capital of France is":
            " Paris. It is the most populous city in France and in Europe. It",
            "The future of AI is":
            " in the hands of the people\nBy: TechCrunch\nDecember 0",
        }

        # Validate outputs
        validation_passed = True
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            logger.info(
                f"\n[Prompt]\n{prompt!r}\n[Generated]\n{generated_text!r}\n")

            try:
                assert generated_text.strip(
                ) == expected_outputs[prompt].strip(), (
                    f"Output mismatch for prompt '{prompt}'\n[Expected]\n"
                    f"{expected_outputs[prompt]!r}\n[Got]\n{generated_text!r}")
                logger.info(f"[Validation] Prompt '{prompt}' passed\n")
            except AssertionError as e:
                logger.error(
                    f"[Validation] Prompt '{prompt}' failed: {str(e)}\n")
                validation_passed = False

        if validation_passed:
            logger.info(
                f"[{title}] vLLM offline inference integration test passed")
        else:
            logger.error(
                f"[{title}] vLLM offline inference integration test failed")
            raise AssertionError("Test failed due to output mismatches")
