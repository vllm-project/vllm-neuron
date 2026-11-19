# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Jina Embeddings v5 contrib model.

Checkpoint: jinaai/jina-embeddings-v5-text-nano (base encoder weights, without
the task LoRA adapters).

The test serves the model offline through vLLM on Neuron with tensor
parallelism 2 on one node, asks the engine for embeddings, and compares each
embedding with the HuggingFace ``EuroBertModel`` reference of the checkpoint.

Why this test is not the template's logit test
----------------------------------------------
Jina Embeddings v5 is a bidirectional encoder with no LM head, no sampler and
no decode phase, so there are no per-token logits and no greedy sequence to
validate. The template's logit shape (``get_cached_reference_goldens`` plus
``multi_prompt_logit_validation``, with ``max_logprobs=-1``,
``logprobs_mode="raw_logits"`` and ``on_device_sampling_config={"all_greedy":
True}``) has nothing to act on here. The test keeps the template's three-way
comparison (FP32 HF baseline, BF16 HF expected, BF16 Neuron actual) and uses
the metric that the plugin defines for pooling models:
``assert_pooling_close_three_way(task="embed")``, a per-vector cosine floor.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. The offline
``LLM(...)`` path starts the engine-core and worker processes with the
``fork`` start method, so they inherit the registration. Do not set
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test. ``pip install -e
contrib/models/jina_embeddings_v5`` also registers the architecture, in every
process, through the plugin entry point.

Running
-------
The configuration below needs 2 NeuronCores on one trn2 node:

    pytest contrib/models/jina_embeddings_v5/test/integration/test_model.py \\
        -v -s --timeout=3600

Set ``JINA_V5_CHECKPOINT`` to a local copy of the checkpoint without the
``adapters/`` directory to skip the download.
"""

import pytest
import torch

import jina_v5_test_utils as utils
from _local_src import src_package

ARCHITECTURE = "JinaEmbeddingsV5Model"

# Tensor parallelism of the recorded runs. 12 heads and an intermediate size of
# 3072 give 6 heads and 1536 channels per rank.
TENSOR_PARALLEL_SIZE = 2

# The serving geometry of the port's own serving test: a 1024-token context,
# one 1024-token prefill bucket and up to 4 sequences in a step.
MAX_MODEL_LEN = utils.DEVICE_BUCKET
MAX_NUM_SEQS = 4

# Cosine floor of the port's device and serving tests. The three-way form also
# reports the BF16-HF-against-FP32-HF cosine, so a floor failure separates a
# port defect from the BF16 precision floor itself.
MIN_COS = 0.99

# Text prompts with the "Query: " and "Document: " prefixes of the model card.
TEXT_PROMPTS = [
    "Query: What is the capital of France?",
    "Document: Paris is the capital and the largest city of France. It is on "
    "the Seine river, in the north of the country.",
]

# Deterministic results across runs.
torch.manual_seed(42)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _prompts(case: str, checkpoint: str) -> tuple[list, list[list[int]]]:
    """Return the vLLM prompts of one case and the token IDs of each prompt."""
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt

    if case == "tokens_1024_and_text":
        # Two full-bucket inputs as token IDs, then two short text inputs that
        # vLLM tokenizes. Each input is one prefill in its own step.
        token_ids = [utils.random_token_ids(seed, MAX_MODEL_LEN) for seed in range(2)]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        text_ids = [tokenizer(p)["input_ids"] for p in TEXT_PROMPTS]
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
        return prompts + list(TEXT_PROMPTS), token_ids + text_ids
    if case == "single_token":
        # 1-token inputs take the decode-classified path of the model.
        token_ids = [[t] for t in utils.random_token_ids(7, MAX_NUM_SEQS)]
        return [TokensPrompt(prompt_token_ids=ids) for ids in token_ids], token_ids
    raise ValueError(f"Unknown case {case!r}")


@pytest.mark.integration
@pytest.mark.parametrize("case", ["tokens_1024_and_text", "single_token"])
def test_model_embeddings(case: str) -> None:
    """Serve the contributed model through vLLM and validate its embeddings.

    Args:
        case: The input set. ``tokens_1024_and_text`` has two 1024-token
            inputs and two text inputs. ``single_token`` has four 1-token
            inputs.
    """
    from vllm import LLM
    from vllm.config import PoolerConfig

    from vllm_neuron.accuracy.testing import assert_pooling_close_three_way

    print(f"Testing model with parameters: {case=}")

    _register_model()

    checkpoint = utils.get_checkpoint()
    prompts, token_ids = _prompts(case, checkpoint)

    # The HF reference in FP32 (baseline) and BF16 (expected): the hidden state
    # of the last token, then L2 normalization, which is the pooling of the
    # checkpoint's sentence-transformers module.
    references = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        hf_model = utils.load_hf_encoder(checkpoint, dtype)
        references[name] = torch.stack(
            [
                utils.pool_last_normalize(utils.hf_last_hidden(hf_model, ids), len(ids))
                for ids in token_ids
            ]
        )
        del hf_model

    # The pooling runner is required: the model has no LM head. The model
    # rejects prefix caching and chunked prefill (the attention is
    # bidirectional), so prefix caching is off and the batched-token limit
    # equals the context length. Pooling models do not support async
    # scheduling on Neuron. The checkpoint's ``modules.json`` has only a
    # custom module, so vLLM finds no pooling config in it; the pooler config
    # sets the LAST pooling and the L2 normalization of the checkpoint.
    llm = LLM(
        model=checkpoint,
        runner="pooling",
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
        async_scheduling=False,
        pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=True),
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [MAX_MODEL_LEN],
                "num_seqs_buckets": [MAX_NUM_SEQS],
            }
        },
    )
    try:
        outputs = llm.embed(prompts)
        for out, ids in zip(outputs, token_ids):
            assert list(out.prompt_token_ids) == list(ids), (
                f"vLLM ran {len(out.prompt_token_ids)} prompt tokens, but the "
                f"reference ran {len(ids)}. The tokenizations differ."
            )
        actual = torch.stack(
            [
                torch.tensor(out.outputs.embedding, dtype=torch.float32)
                for out in outputs
            ]
        )
        assert actual.shape == references["fp32"].shape, (
            f"The engine returned {tuple(actual.shape)} embeddings, but the "
            f"reference has {tuple(references['fp32'].shape)}."
        )
        norms = actual.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), (
            f"The embeddings are not L2 normalized: {norms.tolist()}."
        )
        result = assert_pooling_close_three_way(
            baseline=references["fp32"],
            expected=references["bf16"],
            actual=actual,
            task="embed",
            name=f"jina_v5/integration/{case}/tp{TENSOR_PARALLEL_SIZE}",
            min_cos=MIN_COS,
        )
        print(
            f"worst cos {result.min_tgt_cos:.6f} (floor {MIN_COS}), "
            f"mean cos {result.mean_tgt_cos:.6f}, "
            f"BF16-HF worst cos {result.min_base_cos:.6f}"
        )
        assert result.passed, f"Embedding validation failed for {case=}"
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {case=}")
