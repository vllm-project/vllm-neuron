# SPDX-License-Identifier: Apache-2.0
"""Tiny E2E loading and inference test for HyperCLOVAX in CPU mode.

The test runs the full vLLM pipeline (model load, weight load and cast, forward,
NKI kernel calls in the simulator, sampling and output) on CPU. It uses a
1-layer random-weight checkpoint that ``save_pretrained()`` writes. It does not
need Neuron hardware or the real weights.

The test covers these HyperCLOVAX items:

- A float32 checkpoint, as in the released model. The loader casts it to
  bfloat16. A bfloat16 checkpoint also loads.
- The MuP multipliers, with values that are not 1.
- Peri-layer normalization (post_norm1 and post_norm2).
- GQA with separate q/k/v checkpoint tensors, fused at load.
- The untied LM head.
- The fp32-input O projection of ``src/o_proj.py`` in prefill.

The tiny config keeps the NKI constraints: hidden_size % 256 == 0,
head_dim == 128 (512 / 4) and 2 KV heads (TP=2 gives 1 KV head on each rank).

Each case runs in its own spawned process (see ``_run_inference_in_new_process``),
which registers this folder's factory with vLLM's ``ModelRegistry`` before it
builds the engine (option 1 of ``contrib/README.md``).

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/hyperclovax/test/unit/test_tiny_hyperclovax_e2e.py -v
"""

import multiprocessing
import os

import pytest
import torch
from hyperclovax_test_utils import use_native_hf_classes
from hyperclovax_tiny_inference import MAX_TOKENS, run_inference_child

# The environment the plugin's CPU-mode tests use. The spawned process
# inherits it.
# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the V1 engine core subprocess stops in
#   CPU mode. The in-process mode prevents this, and the TP workers still start.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no HBM, so the calculated
#   budget is almost zero.
# - NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker runs
#   the NKI kernels in the simulator for a known platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NKI_SIMULATOR": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}

# The time limit for one inference run in its own process.
RUN_TIMEOUT_S = 1500
# The time limit for the process to stop after it sends the result.
EXIT_TIMEOUT_S = 120


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set the environment variables for the tiny E2E tests. Restore them after."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.skip("The tiny E2E test runs in CPU mode (VLLM_NEURON_CPU_MODE=1).")
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _tiny_config():
    from transformers import AutoConfig

    config = AutoConfig.for_model(
        "hyperclovax",
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 100000000.0},
        tie_word_embeddings=False,
        attention_multiplier=0.0078125,
        embedding_multiplier=10.0,
        logits_scaling=0.125,
        residual_multiplier=0.5,
        use_post_norm=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return use_native_hf_classes(config)


@pytest.fixture(scope="module", params=["float32", "bfloat16"])
def tiny_hyperclovax_dir(request, tmp_path_factory):
    """Write a random-weight tiny HyperCLOVAX checkpoint in the given dtype."""
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, request.param)
    model_dir = str(tmp_path_factory.mktemp(f"tiny_hyperclovax_{request.param}"))
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(_tiny_config()).to(dtype)
    # Make the post norms different from the input norms, so that a wrong
    # mapping of the norm weights changes the output.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
    model.save_pretrained(model_dir)
    return model_dir


def _run_inference_in_new_process(model_dir: str, tp_size: int) -> list[int]:
    """Run run_inference in a new spawned process and return its token IDs.

    All cases run in one pytest process. A TP=1 run puts the worker and its
    torch.distributed groups in that process, and the groups stay after the
    run. A later case then gets a group that is not registered. A new process
    for each case starts with no group. An assertion failure in the new process
    fails the test.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=run_inference_child, args=(child_conn, model_dir, tp_size)
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(RUN_TIMEOUT_S):
            raise AssertionError(
                f"The TP={tp_size} inference process did not send a result in "
                f"{RUN_TIMEOUT_S} s (exit code {proc.exitcode}). Read the log."
            )
        status, value = parent_conn.recv()
    except EOFError:
        proc.join(EXIT_TIMEOUT_S)
        raise AssertionError(
            f"The TP={tp_size} inference process stopped with exit code "
            f"{proc.exitcode} before it sent a result. Read the log."
        ) from None
    finally:
        parent_conn.close()
        proc.join(EXIT_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
            proc.join()
    if status != "ok":
        raise AssertionError(f"The TP={tp_size} inference process failed:\n{value}")
    token_ids = value
    assert len(token_ids) == MAX_TOKENS
    return token_ids


def test_tp2(tiny_hyperclovax_dir):
    """TP=2: sharded weight load (fused QKV and GQA) and two workers."""
    _run_inference_in_new_process(tiny_hyperclovax_dir, tp_size=2)


def test_tp1(tiny_hyperclovax_dir):
    """TP=1: the full pipeline on one worker."""
    _run_inference_in_new_process(tiny_hyperclovax_dir, tp_size=1)
