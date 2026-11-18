# SPDX-License-Identifier: Apache-2.0
"""Integration accuracy test for Mistral-7B-v0.3, served through vLLM on Neuron.

Serves the model through vLLM (with the ``vllm-neuron`` plugin) and compares
per-token logits against a Hugging Face golden reference, as in the contrib
template. Helpers from ``vllm_neuron.accuracy``:

  * ``get_cached_reference_goldens`` computes (and caches on disk) the HF
    reference logits: ``fp32_logits`` (FP32 baseline) and ``dtype_logits``
    (BF16, teacher-forced along the FP32 greedy sequence).
  * ``create_offline_vllm_generate_fn`` wraps the vLLM ``LLM`` so it returns
    dense per-token logits.
  * ``multi_prompt_logit_validation`` flags any argmax divergence and any
    top-K logit error outside ``TOL_MAP``.

Registration
------------
``MistralForCausalLM`` is registered by this folder's ``vllm.general_plugins``
entry point, so vLLM registers it in every process it starts (front end,
engine core, Neuron workers). Install the folder first:

    pip install -e contrib/models/mistral

Checkpoint
----------
``MISTRAL_MODEL`` must be a local directory without
``consolidated.safetensors`` (the model refuses to load it, see
``_open_checkpoint`` in ``src/model.py``):

    hf download mistralai/Mistral-7B-Instruct-v0.3 \\
        --exclude consolidated.safetensors --local-dir /data/Mistral-7B-Instruct-v0.3

Running
-------
CPU mode, tiny checkpoint (seconds per case):

    python contrib/models/mistral/test/unit/tiny_checkpoint.py /tmp/tiny-mistral
    VLLM_NEURON_CPU_MODE=1 MISTRAL_MODEL=/tmp/tiny-mistral MISTRAL_TP=2 \\
        pytest contrib/models/mistral/test/integration/test_model.py -v -s --timeout=1800

CPU mode, real 7B (needs about 64 GB of host memory, slow):

    VLLM_NEURON_CPU_MODE=1 MISTRAL_MODEL=/data/Mistral-7B-Instruct-v0.3 \\
        pytest contrib/models/mistral/test/integration/test_model.py -v -s --timeout=7200

Trn2 (trn2.3xlarge has 4 NeuronCores at the default LNC=2):

    MISTRAL_MODEL=/data/Mistral-7B-Instruct-v0.3 MISTRAL_TP=4 \\
        pytest contrib/models/mistral/test/integration/test_model.py \\
        -v -s --timeout=3600 > /tmp/test_output.txt 2>&1
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get("MISTRAL_MODEL")
CPU_MODE = os.environ.get("VLLM_NEURON_CPU_MODE") == "1"
TENSOR_PARALLEL_SIZE = int(os.environ.get("MISTRAL_TP", "1" if CPU_MODE else "4"))

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason="set MISTRAL_MODEL to a local Mistral checkpoint directory",
)

# The HF reference logits are computed from these same prompts, so no
# expected text/logits are hard-coded here. The second prompt uses the
# Mistral instruct format.
PROMPTS = [
    "The color of the sky is",
    "[INST] Name three prime numbers. [/INST]",
]

# Number of tokens to decode and validate per prompt.
OUTPUT_LENGTH = 16

# Unchanged from the contrib template. The argmax (divergence) lower bound
# stays strict at 1e-5: a nonzero divergence is a real correctness bug and
# must not be widened to hide it. The per-K upper bounds are the common bf16
# dense-golden convention.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Argmax (divergence) tolerance in BF16 ULPs. 1 on device with real weights
# (zero divergence observed at TP 2 and TP 4). 2 in CPU mode: the random-weight
# tiny checkpoint has near-tied top logits, and at TP 2 each rank rounds its
# partial sums to BF16 before the cross-rank add, which shifts one token by
# 2 ULPs relative to the single-process HF reference (seen at prompt 1, token 7
# with the seed-0 tiny checkpoint; top-K errors stay within TOL_MAP). The
# Muse-Glimmer contrib tiny test uses a logit margin for the same reason.
DIVERGENCE_N_ULPS = 2 if CPU_MODE else 1

# (batch_size, seq_len) pairs. Each pair compiles a separate graph on device.
# CPU mode pads every prefill to seq_len tokens, so it uses shorter lengths.
DEVICE_CASES = [(1, 2048), (4, 2048), (2, 8192)]
CPU_CASES = [(1, 512), (2, 512)]

# Deterministic results across runs.
torch.manual_seed(42)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size, seq_len", CPU_CASES if CPU_MODE else DEVICE_CASES)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve Mistral through vLLM and validate its logits against HF."""

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from mistral_neuron.plugin import register
    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    # The entry point already registers the model in every vLLM process; this
    # call only makes a missing `pip install -e` fail here with a clear error.
    register()

    print(
        f"Testing {MODEL_PATH} with {batch_size=}, {seq_len=}, "
        f"tp={TENSOR_PARALLEL_SIZE}, cpu_mode={CPU_MODE}"
    )

    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    assert model_config.architectures == ["MistralForCausalLM"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ``model_id`` only builds the golden cache key. The local path keeps the
    # tiny and the real checkpoint apart. Delete the cached goldens if you
    # rebuild a checkpoint at the same path with different weights.
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
    )

    # Four of these arguments are required by logit validation (see the
    # contrib template): max_logprobs=-1, logprobs_mode="raw_logits",
    # async_scheduling=False and on_device_sampling_config all_greedy.
    # The two bucket lists pin compilation to the one shape this case uses.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        # Eager mode is only supported (and only needed) in CPU mode.
        enforce_eager=CPU_MODE,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        generate_fn = create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH)
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
            divergence_n_ulps=DIVERGENCE_N_ULPS,
        )
        assert result.passed, (
            f"Logit validation failed for parameters: {batch_size=}, {seq_len=}"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
