# SPDX-License-Identifier: Apache-2.0
"""Integration test for the OLMo contrib model, served through vLLM on Neuron.

The test serves ``allenai/OLMo-1B-hf`` with the implementation in
``contrib/models/olmo/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its own
``OlmoForCausalLM`` entry; the registration replaces it.

The registration has to reach vLLM's engine-core and worker processes, and it
reaches them by inheritance: the offline ``LLM(...)`` path starts those
processes with the ``fork`` start method, so they inherit this process's
``sys.modules`` and the populated ``ModelRegistry``. That is the default for a
Python script. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` for this test.
After ``pip install -e contrib/models/olmo`` the package's general-plugin entry
point registers the same factory in every process, so either way works.

Checkpoint
----------
The test reads ``OLMO_CHECKPOINT_PATH`` when it is set (a local copy of the
pinned revision). Otherwise it downloads the pinned revision from the Hugging
Face Hub (about 4.7 GB of FP32 weights).

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    pytest contrib/models/olmo/test/integration/test_model.py \
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence.
``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so that it returns
dense per-token logits, and ``multi_prompt_logit_validation`` compares the two.
It flags any argmax divergence and any top-K logit error outside ``TOL_MAP``.

Padded vocabulary
-----------------
The model's vocabulary (50,304 rows) is larger than its tokenizer (50,280
tokens): ids 50,280 and up are padding rows with no token string. Two things
in this test handle that, and neither changes a logit:

* ``generate_fn`` requests ``logprobs`` equal to the tokenizer's vocab size,
  which here is smaller than the model's vocabulary. That gives per-step rows
  of different widths, none of them as wide as the HF reference.
  ``_ModelVocabLLM`` reports the model's vocab size to ``generate_fn``, so it
  requests every logit and every row is 50,304 wide.
* vLLM 0.29 decodes every logprob id to a string, and the tokenizer returns
  ``None`` for a padding id. The test maps those to ``""`` in the engine's
  tokenizer. The comparison reads the logits, never the strings.
"""

from types import SimpleNamespace

import pytest
import torch

from _local_src import src_package
from olmo_test_utils import HF_REVISION, MODEL_ID, resolve_checkpoint

# Prompts to validate (``PROMPTS_2`` of the plugin's test prompts, the prompts
# of the port's device logit test). The HF reference logits come from these
# same prompts, so no expected text or logits are hard-coded here.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map of the template, which is also the map
# the port's device logit test used. The argmax (divergence) lower bound stays
# strict at 1e-5 — a nonzero divergence is a real correctness bug and must not
# be widened to hide it. The per-K upper bounds bound the allowed top-K logit
# drift.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# The topology of the recorded runs: TP 8. 16 heads give 2 Q and 2 KV heads
# per rank; TP 16 would give one head per rank, which the model rejects.
TENSOR_PARALLEL_SIZE = 8

# Deterministic results across runs.
torch.manual_seed(42)

ARCHITECTURE = "OlmoForCausalLM"


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


class _ModelVocabLLM:
    """Give ``create_offline_vllm_generate_fn`` the model's vocab size.

    ``generate_fn`` reads only ``get_tokenizer().vocab_size`` and
    ``generate(...)``. Reporting the model's vocabulary makes it request the
    logit of every id, padding rows included, so each row has the HF width.
    """

    def __init__(self, llm, vocab_size: int):
        self._llm = llm
        self._vocab_size = vocab_size

    def generate(self, *args, **kwargs):
        return self._llm.generate(*args, **kwargs)

    def get_tokenizer(self):
        return SimpleNamespace(vocab_size=self._vocab_size)


def _map_padding_ids_to_empty_tokens(tokenizer) -> None:
    """Return ``""`` instead of ``None`` for the ids past the tokenizer."""
    convert_ids_to_tokens = tokenizer.convert_ids_to_tokens

    def _convert_ids_to_tokens(ids, *args, **kwargs):
        tokens = convert_ids_to_tokens(ids, *args, **kwargs)
        if isinstance(tokens, list):
            return ["" if token is None else token for token in tokens]
        return "" if tokens is None else tokens

    tokenizer.convert_ids_to_tokens = _convert_ids_to_tokens


# The (batch_size, seq_len) pairs: batch 1 and batch 4 at a 2048-token model
# length (the model's max_position_embeddings). Each pair compiles its own graph.
@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve the contributed model through vLLM and validate its logits.

    Args:
        batch_size: Number of prompts to run in a single batch.
        seq_len: Maximum model (context + generation) length to configure.
    """

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    _register_model()
    model_path = resolve_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint stores FP32 weights and the engine serves BF16, so the
    # same-dtype reference must be BF16.
    model_config = AutoConfig.from_pretrained(model_path)
    model_config.torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=model_path,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant=f"rev-{HF_REVISION[:12]}",
    )

    # Four of these arguments are required by logit validation, not optional
    # tuning. Do not remove them:
    #   * max_logprobs=-1 gathers the full vocabulary instead of a top-K slice.
    #   * logprobs_mode="raw_logits" returns raw logits. Without it vLLM returns
    #     normalized log-probabilities, which do not match the HF reference.
    #   * async_scheduling=False keeps logprobs available. The async path returns
    #     before it computes logprobs, so generate_fn gets no logits.
    #   * on_device_sampling_config={"all_greedy": True} makes sampling greedy
    #     and deterministic, so a rerun compares the same token sequence.
    #
    # ``quantization: bf16`` is model-specific: the OLMo checkpoint stores FP32
    # weights and this implementation computes in BF16.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        block_size=32,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        _map_padding_ids_to_empty_tokens(llm.get_tokenizer())
        generate_fn = create_offline_vllm_generate_fn(
            _ModelVocabLLM(llm, model_config.vocab_size), OUTPUT_LENGTH
        )
        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
        ]
        result = multi_prompt_logit_validation(
            prompts_input_ids=prompts_input_ids,
            generate_fn=generate_fn,
            prompts_expected_logits=cached_goldens["dtype_logits"],
            prompts_baseline_logits=cached_goldens["fp32_logits"],
            colorize=True,
            replicate_to_batch_size=batch_size,
            tol_map=TOL_MAP,
        )
        assert result.passed, (
            f"Logit validation failed for parameters: {batch_size=}, {seq_len=}"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
