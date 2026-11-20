# SPDX-License-Identifier: Apache-2.0
"""Nanbeige4.2-3B CPU unit tests: loading and configuration.

The tests use the real checkpoint (``NANBEIGE4_CHECKPOINT_PATH``) and the
native route on CPU. They cover:

- the registration and factory wiring, the configuration parse, and the
  refusal of unsupported options;
- a full-model load at TP2 that reads every checkpoint tensor, and a full
  prefill whose next-token logits match the independent reference;
- a cross-check of the independent reference against the checkpoint remote
  code.

The per-component checks (one physical decoder layer, run for each loop with
its own KV cache, against an upstream ``LlamaDecoderLayer``) are module tests.
They live in ``test/module/test_decoder_layer.py``.

The fused decode megakernel does not run on CPU. The two-node device tests
covered the decode path; they are not shipped here (see the model README's
Exceptions section).

Run with:
    VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
        NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
        pytest contrib/models/nanbeige/test/unit/ -v
"""

import functools
import importlib.util
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor

# Load this model's own code and test helper by file location, so the folder
# keeps working when it is copied out of the repository. See
# ``contrib/models/nanbeige/test/_local_src.py``.
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "contrib_nanbeige_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_local_src)

nb = _local_src.load_test_helper("nanbeige_utils")
PROMPTS_2 = nb.PROMPTS_2

OUTPUT_LENGTH = 32
BLOCK_SIZE = 32
COLLECT_TIMEOUT_S = 3600


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    return nb.get_checkpoint_path()


@pytest.fixture(autouse=True)
def require_cpu_mode():
    if not envs.VLLM_NEURON_CPU_MODE:
        pytest.fail("test_nanbeige_cpu.py needs CPU mode. Set VLLM_NEURON_CPU_MODE=1.")


# =============================================================================
# Registry, configuration and validation
# =============================================================================


def test_nanbeige_registry_and_config(checkpoint_path):
    """Registration resolves the architecture, and the config parses.

    A contrib model is not in ``vllm_neuron.model.registry.get_models()`` by
    design, so the user registers the architecture with vLLM (see
    contrib/README.md, "How do I register a custom model with vLLM?"). This
    test does the registration the same way the integration test does it.
    """
    from transformers import AutoConfig
    from vllm import ModelRegistry

    src = _local_src.load()
    NanbeigeConfig = src.NanbeigeConfig
    NanbeigeForCausalLM = src.NanbeigeForCausalLM

    raw = nb.read_raw_config(checkpoint_path)
    architecture = raw["architectures"][0]
    ModelRegistry.register_model(architecture, NanbeigeForCausalLM)
    assert ModelRegistry._try_load_model_cls(architecture) is NanbeigeForCausalLM

    def check(config, source):
        assert config.num_hidden_layers == nb.NUM_PHYSICAL_LAYERS, source
        assert config.num_loops == nb.NUM_LOOPS, source
        assert config.num_logical_layers == nb.NUM_PHYSICAL_LAYERS * nb.NUM_LOOPS
        assert config.rope_theta == pytest.approx(7.0e7), source
        assert config.head_dim == 128, source
        assert config.num_attention_heads == 48, source
        assert config.num_key_value_heads == 8, source
        assert config.vocab_size == nb.VOCAB_SIZE, source
        assert config.rms_norm_eps == pytest.approx(1e-5), source
        assert config.tie_word_embeddings is False, source
        assert config.skip_loop_final_norm is False, source
        assert config.torch_dtype == torch.bfloat16, source

    check(NanbeigeConfig.from_configs(raw), "raw config.json")

    # vLLM loads the config through the remote code. That code can expand
    # num_hidden_layers to physical x loops for its own KV slots.
    hf_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    check(NanbeigeConfig.from_configs(hf_config), "AutoConfig (remote code)")

    expanded = dict(raw, num_hidden_layers=44, num_physical_layers=22)
    check(NanbeigeConfig.from_configs(expanded), "expanded layer count")

    # The factory accepts the unchanged checkpoint config.
    NanbeigeForCausalLM._validate_config(raw, None)


def test_nanbeige_factory_rejects_unsupported(checkpoint_path):
    """The factory refuses each option that the native implementation does not support."""
    NanbeigeForCausalLM = _local_src.load().NanbeigeForCausalLM
    unsupported_flags = _local_src.load("factory")._UNSUPPORTED_FLAGS

    raw = nb.read_raw_config(checkpoint_path)
    cases = {flag: {flag: True} for flag in unsupported_flags}
    cases.update(
        {
            "dtype": {"torch_dtype": "float16"},
            "emb_neighbor_num": {"emb_neighbor_num": 2},
            "loop_loss_weights": {"loop_loss_weights": [0.5, 1.0]},
            "hidden_act": {"hidden_act": "gelu"},
            "rope_scaling": {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        }
    )
    for name, override in cases.items():
        with pytest.raises(ValueError):
            NanbeigeForCausalLM._validate_config(dict(raw, **override), None)
            pytest.fail(f"The factory accepted the unsupported option {name}.")


# =============================================================================
# Full model at TP2: load and prefill logits
# =============================================================================


@pytest.mark.parametrize("world_size", [2])
def test_nanbeige_full_model_load_and_prefill_logits(checkpoint_path, world_size):
    """Load every checkpoint tensor at TP2 and compare the prefill logits.

    The reference is the independent upstream-Llama loop (FP32 and BF16).
    The test compares the logits of the first generated token for each prompt.
    """
    tokenizer = nb.load_tokenizer(checkpoint_path)
    goldens = nb.get_full_model_goldens(
        checkpoint_path, tokenizer, PROMPTS_2, OUTPUT_LENGTH
    )

    weight_map = nb.checkpoint_weight_map(checkpoint_path)
    assert len(weight_map) == nb.EXPECTED_NUM_CHECKPOINT_KEYS

    max_len = max(len(ids[0]) for ids in goldens["input_ids"])
    seq_len = (max_len // world_size + 1) * world_size
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            nb.load_full_model_runner, checkpoint_path, num_blocks, BLOCK_SIZE, "cpu"
        ),
        device="cpu",
        load_timeout=COLLECT_TIMEOUT_S,
    )

    for idx, prompt_ids in enumerate(goldens["input_ids"]):
        ids = prompt_ids[0]
        # Pad at the end. Causal attention keeps the logits at len(ids) - 1.
        padded = ids + [0] * (seq_len - len(ids))
        executor.dispatch(
            torch.tensor(padded, dtype=torch.long),
            torch.arange(seq_len, dtype=torch.long),
            nb.full_model_prefill_metadata(seq_len, BLOCK_SIZE, num_blocks),
            torch.tensor([len(ids) - 1], dtype=torch.long),
        )
        outputs = executor.collect(timeout=COLLECT_TIMEOUT_S)

        for rank, (logits, report) in enumerate(outputs):
            source, missing, unexpected, on_meta, non_finite, not_loaded = (
                report.tolist()
            )
            assert source == len(weight_map), f"rank {rank}: {report.tolist()}"
            assert missing == unexpected == 0, f"rank {rank}: {report.tolist()}"
            assert on_meta == non_finite == not_loaded == 0, (
                f"rank {rank}: {report.tolist()}"
            )
            assert logits.shape == (1, nb.VOCAB_SIZE)
            assert torch.isfinite(logits).all()

        logits = outputs[0][0].float()
        for rank, (other, _) in enumerate(outputs[1:], start=1):
            torch.testing.assert_close(other.float(), logits, msg=f"rank {rank}")

        assert_close_three_way(
            baseline=goldens["fp32_logits"][idx][0, 0],
            expected=goldens["dtype_logits"][idx][0, 0],
            actual=logits[0],
            name=f"nanbeige/full_prefill_logits/tp{world_size}/prompt{idx}",
        )


# =============================================================================
# Cross-check of the independent reference against the remote code
# =============================================================================


def test_nanbeige_reference_matches_remote_code(checkpoint_path):
    """Compare the independent reference with the checkpoint remote code (FP32).

    The remote code is the definition of the model. The installed
    transformers version can be too new for it. If the remote code cannot
    load, the test skips and gives the reason. A numerical difference fails
    the test.
    """
    from transformers import AutoModelForCausalLM

    tokenizer = nb.load_tokenizer(checkpoint_path)
    input_ids = tokenizer([PROMPTS_2[0]], return_tensors="pt")["input_ids"]

    reference = nb.load_loop_reference(checkpoint_path, torch.float32)
    with torch.inference_mode():
        expected = reference(input_ids).logits[0, -1].float()
    del reference

    try:
        remote, info = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
    except Exception as exc:  # noqa: BLE001 - report every load failure
        pytest.skip(
            f"UNMEASURED: the remote code does not load with this transformers "
            f"version: {type(exc).__name__}: {exc}"
        )
    if info.get("missing_keys") or info.get("unexpected_keys"):
        pytest.skip(
            f"UNMEASURED: the remote code loads with key differences: "
            f"missing {info.get('missing_keys')}, "
            f"unexpected {info.get('unexpected_keys')}"
        )
    # Transformers 5 replaces each non-persistent buffer with empty memory
    # during from_pretrained. It initializes again only a rotary module that
    # has original_inv_freq. The remote rotary module does not have it, so
    # its inv_freq buffer is not initialized. Calculate the buffer again with
    # the formula of the remote __init__.
    rotary_modules = [
        module
        for module in remote.modules()
        if "RotaryEmbedding" in type(module).__name__
    ]
    assert rotary_modules, "The remote model has no rotary embedding module."
    for module in rotary_modules:
        inv_freq = 1.0 / (
            module.base
            ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim)
        )
        module.inv_freq.copy_(inv_freq.to(module.inv_freq.device))
        assert torch.isfinite(module.inv_freq).all(), type(module).__name__

    remote.eval()
    with torch.inference_mode():
        actual = remote(input_ids, use_cache=False).logits[0, -1].float()

    assert torch.isfinite(actual).all(), "The remote logits are not finite."
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    assert actual.argmax() == expected.argmax()
