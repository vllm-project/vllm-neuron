# SPDX-License-Identifier: Apache-2.0
"""Integration test for the MiniCPM3 contrib model, served through vLLM on Neuron.

The test serves ``openbmb/MiniCPM3-4B`` with the implementation in
``contrib/models/minicpm3/src/`` and compares the raw per-token logits with a
HuggingFace golden reference, as the template in
``contrib/models/template/test/integration/test_model.py`` does.

Registering the model
---------------------
Contrib models are not in ``vllm_neuron/model/registry.py``, so the test
registers the architecture itself with ``ModelRegistry.register_model`` before
it builds the engine (option 1 in ``contrib/README.md``). It loads the factory
straight out of this folder, through ``test/_local_src.py``. vLLM ships its own
``MiniCPM3ForCausalLM``, so the call replaces that entry. If this folder is
also installed with ``pip install -e contrib/models/minicpm3``, the plugin
entry point registers the same factory again in every process, which is
equivalent.

The registration has to reach vLLM's engine-core and worker processes. Without
the installed entry point it reaches them by inheritance: the offline
``LLM(...)`` path starts those processes with the ``fork`` start method, so
they inherit this process's ``sys.modules`` and the populated
``ModelRegistry``. Do not set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` unless the
folder is installed: a spawned child never runs the registration call,
resolves vLLM's own class and then fails on ``from_configs``.

Running
-------
The configuration below needs 8 NeuronCores on one trn2.48xlarge node:

    pytest contrib/models/minicpm3/test/integration/test_model.py \
        -v -s --timeout=7200 > /tmp/test_output.txt 2>&1

Set ``MINICPM3_CHECKPOINT_PATH`` to a local snapshot of the pinned revision
to skip the download (about 8 GB).

Logit validation
----------------
``get_cached_reference_goldens`` computes (and caches) the HF reference logits:
``fp32_logits`` is the FP32 baseline, and ``dtype_logits`` is a same-dtype pass
teacher-forced along the FP32 greedy sequence. The reference is the
``MiniCPM3ForCausalLM`` built into transformers (``native_hf_model_loader``),
not the checkpoint's remote code. ``_full_vocab_generate_fn`` (below) wraps
the vLLM ``LLM`` so that it returns dense per-token logits over the whole
model vocabulary, and ``multi_prompt_logit_validation`` compares the two. It
flags any argmax divergence and any top-K logit error outside ``TOL_MAP``.
"""

import pytest
import torch

from _local_src import src_package
from minicpm3_test_utils import (
    ARCHITECTURE,
    MODEL_ID,
    WORLD_SIZE,
    load_hf_config,
    native_hf_model_loader,
    resolve_checkpoint,
)

# The two prompts of the port's logit test (``PROMPTS_2`` of the plugin's
# shared test prompts, copied here as contrib/README.md allows). The HF
# reference logits come from these same prompts, so no expected text or logits
# are hard-coded here.
PROMPTS = [
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "spanning across nine countries and containing approximately ten percent "
    "of all species on Earth.",
    "The human genome contains approximately three billion base pairs of DNA, "
    "encoding the genetic instructions for building and maintaining a human "
    "organism.",
]

# Number of tokens to decode and validate per prompt, as in the port's logit
# test.
OUTPUT_LENGTH = 16

# The bf16 dense-golden tolerance map of the template (the port's logit test
# used the same map). The argmax (divergence) lower bound stays strict at
# 1e-5 — a nonzero divergence is a real correctness bug and must not be
# widened to hide it. The per-K upper bounds bound the allowed top-K logit
# drift.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Per-prompt checks of the three-way validation. Each one compares the
# largest error of one prompt against 1.5x the largest error of the CPU BF16
# reference for the same prompt.
PER_PROMPT_CHECKS = ("pp_max_linf_1.5x", "pp_max_l2_1.5x")

# 40 heads: TP 8 gives 5 heads on each rank, the degree of the recorded
# single-node device runs.
TENSOR_PARALLEL_SIZE = WORLD_SIZE

# Deterministic results across runs.
torch.manual_seed(42)


def _register_model() -> None:
    """Load the factory from ``src/`` and register it with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, getattr(src_package(), ARCHITECTURE))


def _full_vocab_generate_fn(llm, output_length: int, vocab_size: int):
    """Return a ``generate_fn`` that packs the logits of every vocabulary id.

    This is the plugin's ``create_offline_vllm_generate_fn`` with three changes,
    all needed for this checkpoint:

    * The model scores 73,448 ids, but the tokenizer reports a base
      ``vocab_size`` of 73,440 (73,448 entries with the added tokens). The
      plugin helper requests ``tokenizer.vocab_size`` logprobs and sizes each
      row from the largest id it gets back, so the rows can get different
      lengths and ``torch.stack`` fails before any validation runs. This
      function requests all ``vocab_size`` ids of the model config and builds
      each row with that width, the width of the HF reference.
    * ``detokenize=False``: vLLM would otherwise decode every one of the
      73,448 logprob ids to a string at each step. The comparison reads only
      the logits.
    * ``ignore_eos=True``: the HF reference decodes exactly ``output_length``
      tokens, so the vLLM run must not stop early on an end-of-sequence token.

    Args:
        llm: The running vLLM ``LLM``.
        output_length: The number of tokens to decode for each prompt.
        vocab_size: The vocabulary size of the model (``config.vocab_size``).

    Returns:
        ``generate_fn(input_ids) -> (logits, sampled_token_ids)``. ``logits``
        has the shape ``[output_length, num_prompts, vocab_size]``.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def generate_fn(input_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=output_length,
            logprobs=vocab_size,
            detokenize=False,
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, sampling_params)

        batch_logits = []
        batch_tokens = []
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.logprobs) == output_length, (
                f"Got {len(completion.logprobs)} logprob steps, expected "
                f"{output_length}."
            )
            rows = []
            for step, logprob_dict in enumerate(completion.logprobs):
                assert len(logprob_dict) == vocab_size, (
                    f"Step {step} returned {len(logprob_dict)} logprobs, expected "
                    f"all {vocab_size}. Check max_logprobs=-1."
                )
                row = torch.full((vocab_size,), -float("inf"))
                row[torch.tensor(list(logprob_dict.keys()))] = torch.tensor(
                    [lp.logprob for lp in logprob_dict.values()]
                )
                rows.append(row)
            batch_logits.append(torch.stack(rows, dim=0))
            batch_tokens.append(
                torch.tensor(completion.token_ids[:output_length], dtype=torch.long)
            )
        return torch.stack(batch_logits, dim=1), torch.stack(batch_tokens, dim=0)

    return generate_fn


# The (batch_size, seq_len) pairs: a 2048-token model length, with one
# sequence and with a 4-sequence batch (the batch sizes of the recorded
# serving runs). Each pair compiles its own graphs.
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

    from transformers import AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    _register_model()
    model_path = resolve_checkpoint()

    # Compute (or load from cache) the HuggingFace reference logits for PROMPTS.
    # The checkpoint's config.json maps to remote code written for transformers
    # 4.41, which does not run on transformers 5.17, so the reference is the
    # transformers built-in MiniCPM3ForCausalLM. ``golden_variant`` keeps these
    # goldens apart from any cached with the default (remote-code) loader.
    model_config = load_hf_config(model_path, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key, so it stays the Hub id
    # when the checkpoint is a local snapshot.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_ID,
        model_checkpoint=model_path,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        model_loader=native_hf_model_loader,
        golden_variant="transformers_builtin_minicpm3",
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
    # generation_config="vllm" keeps the checkpoint's generation_config.json
    # (do_sample, temperature 0.8, top_p 0.8) out of the request defaults.
    #
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        generation_config="vllm",
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
        generate_fn = _full_vocab_generate_fn(
            llm, OUTPUT_LENGTH, model_config.vocab_size
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
        n_prompts = len(result.per_prompt_results)
        for check in PER_PROMPT_CHECKS:
            value = result.aggregate_metrics.get(check)
            assert value == f"{n_prompts}/{n_prompts}", (
                f"{check} is {value} for parameters: {batch_size=}, {seq_len=}"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
