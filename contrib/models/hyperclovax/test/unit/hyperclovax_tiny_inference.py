# SPDX-License-Identifier: Apache-2.0
"""Inference helper for the tiny HyperCLOVAX E2E tests.

The tests run ``run_inference_child`` in a new spawned process. A spawned
process finds the function by its module name, so the test imports this module
by its plain name (``hyperclovax_tiny_inference``). pytest puts ``test/unit``
on ``sys.path`` and ``test/conftest.py`` puts ``test/`` there; a spawned child
gets the same ``sys.path``, so it imports this module and ``_local_src``.

A spawned child does not inherit the ``ModelRegistry`` of the test process, so
``run_inference`` registers this folder's factory itself, before it builds the
engine. The engine core then runs in the child
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are forked from the
child, so the registration is in effect everywhere the model is built.
"""

import traceback

PROMPT_TOKEN_IDS = list(range(3, 13))
MAX_TOKENS = 3
VOCAB_SIZE = 256
ARCHITECTURE = "HyperCLOVAXForCausalLM"


def run_inference(model_dir: str, tp_size: int) -> list[int]:
    """Generate MAX_TOKENS greedy tokens with vLLM and return the token IDs."""
    from vllm import LLM, ModelRegistry, SamplingParams

    from _local_src import src_package

    # vLLM has its own HyperCLOVAXForCausalLM. This replaces it.
    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_num_seqs=1,
        max_model_len=256,
        block_size=128,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=128,
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [128, 256]}},
    )
    outputs = llm.generate(
        [{"prompt_token_ids": PROMPT_TOKEN_IDS}],
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True),
    )
    assert len(outputs) == 1
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert len(token_ids) == MAX_TOKENS
    assert all(0 <= t < VOCAB_SIZE for t in token_ids)
    return token_ids


def run_inference_child(conn, model_dir: str, tp_size: int) -> None:
    """Run run_inference and send the result or the error text through conn."""
    try:
        conn.send(("ok", run_inference(model_dir, tp_size)))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()
