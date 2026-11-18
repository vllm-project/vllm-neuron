# SPDX-License-Identifier: Apache-2.0
"""Tiny E2E tests for EXAONE 4.0 in CPU mode.

The tests run the full vLLM pipeline on CPU with a random-weight tiny
checkpoint: model load, weight loading, prefill, decode and sampling. The HF
``Exaone4ForCausalLM`` gives the independent reference.

The tiny config keeps the EXAONE 4.0 specific paths:
- Hybrid layers in the "LLLG" pattern: 3 sliding-window layers with RoPE and
  1 full layer with NoPE.
- Llama3 RoPE scaling with the real checkpoint values.
- Per-head QK-norm before RoPE, and the post-norm decoder layer.
- An untied LM head.
- An MLP intermediate size that needs padding for each rank (544 -> 640 at
  TP2, 1088 -> 1152 at TP1).

The window (4096) is larger than the context of this test. The test thus does
not reach the window boundary. The device DI test also stays below the window.

Accuracy rule (TP2): the test feeds the vLLM tokens back into HF in FP32 and
in BF16 (teacher forcing). The error of vLLM against FP32 must be at most 5x
the BF16 reference error. This is the default ``max_linf_ratio`` of
``assert_close_three_way``. The BF16 reference error has a floor of 1% of the
largest FP32 logit, which is about one BF16 ULP at the logit scale.

The test registers the contrib architecture with vLLM's ``ModelRegistry``
before it builds the engine (contrib/README.md, "How do I register a custom
model with vLLM?"), because a contrib model is not in the plugin's registry
and vLLM ships its own ``Exaone4ForCausalLM``. The engine core runs in this
process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) and the TP workers are forked
from it, so the registration is in effect everywhere the model is built. See
the model README's Usage section for why that does not hold for ``vllm serve``.

Run with:
    VLLM_NEURON_CPU_MODE=1 pytest \
        contrib/models/exaone4/test/unit/test_tiny_exaone4_e2e.py -v
"""

import contextlib
import logging
import os
import re
import sys

import pytest
import torch
from transformers import Exaone4Config, Exaone4ForCausalLM
from vllm import LLM, ModelRegistry, SamplingParams


def _load_local_src():
    """Load this model's ``test/_local_src.py`` helper by file location.

    Loading the helper by location, under a name unique to this model, keeps
    two contrib models collected in one pytest run from resolving each other's
    helper (a plain ``import _local_src`` would return whichever copy Python
    cached first). Nothing is added to ``sys.path``.
    """
    import importlib.util
    from pathlib import Path

    name = "contrib_exaone4_local_src"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "_local_src.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_local_src = _load_local_src()

# The modeling code that ships in this contrib folder (``../../src/``).
NeuronExaone4ForCausalLM = _local_src.load().Exaone4ForCausalLM
# The module object, so the dispatch assertions below can read the log records
# of its logger (``logging.getLogger(__name__)`` inside ``src/model.py``).
neuron_exaone4_model = _local_src.load("model")

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# The port ran these tests with a conftest beside them. A contrib model keeps
# its test tree flat, so the same environment lives here:
# - VLLM_ENABLE_V1_MULTIPROCESSING=0: the V1 engine core subprocess hangs in
#   CPU mode. In-process mode prevents this. TP workers still start safely.
# - VLLM_NEURON_MIN_KV_BUDGET_GIB=0: CPU mode has no real HBM.
# - NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE=trn2: the CPU worker
#   needs a concrete platform target.
_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
    "NKI_SIMULATOR": "1",
    "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
}


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set the env vars for the tiny E2E tests. Restore them at teardown."""
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


NUM_LAYERS = 4
PROMPT = list(range(3, 40))
MAX_TOKENS = 4
TOP_LOGPROBS = 5
LINF_RATIO = 5.0
BASE_FLOOR_FRACTION = 0.01

TINY_EXAONE4_CONFIG = Exaone4Config(
    vocab_size=256,
    hidden_size=512,
    intermediate_size=1088,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_parameters={
        "rope_type": "llama3",
        "rope_theta": 1_000_000.0,
        "factor": 16.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    sliding_window=4096,
    sliding_window_pattern=4,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

_DISPATCH_RE = re.compile(
    r"\[exaone4_dispatch\] phase=(\w+) component=(\w+) layer=(\d+) "
    r"layer_type=(\w+) rank=(\d+) kernel=(True|False)"
)


@pytest.fixture(scope="module")
def tiny_exaone4_dir(tmp_path_factory):
    """Make a random-weight tiny EXAONE 4.0 model in BF16, one time."""
    model_dir = str(tmp_path_factory.mktemp("tiny_exaone4"))
    torch.manual_seed(42)
    model = Exaone4ForCausalLM(TINY_EXAONE4_CONFIG)
    # Random norm weights make the post-norm path sensitive to a wrong norm.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(model_dir)
    assert TINY_EXAONE4_CONFIG.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    return model_dir


def _make_llm(model_dir: str, tp_size: int) -> LLM:
    # vLLM has its own Exaone4ForCausalLM, so the contrib class must replace
    # it in the registry before the engine resolves the architecture.
    ModelRegistry.register_model("Exaone4ForCausalLM", NeuronExaone4ForCausalLM)
    return LLM(
        model=model_dir,
        max_num_seqs=1,
        max_model_len=256,
        block_size=128,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=128,
        max_logprobs=TOP_LOGPROBS,
        logprobs_mode="raw_logits",
        # The async sampler output has no logprobs. Use sync scheduling.
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [128, 256],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )


@contextlib.contextmanager
def _llm(model_dir: str, tp_size: int):
    """Yield an LLM. Shut down its engine core at exit, also after a failure.

    The engine core runs in this process. Its shutdown removes the default
    process group of this process. A failed test keeps the LLM alive through
    the traceback, and the garbage collector can then run the shutdown during
    the next test. An explicit shutdown prevents this.
    """
    llm = _make_llm(model_dir, tp_size)
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def _generate(llm: LLM):
    outputs = llm.generate(
        [{"prompt_token_ids": PROMPT}],
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            ignore_eos=True,
            logprobs=TOP_LOGPROBS,
        ),
    )
    assert len(outputs) == 1
    return outputs[0].outputs[0]


def _hf_teacher_forced_logits(model_dir: str, tokens: list[int], dtype):
    model = Exaone4ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([PROMPT + tokens], dtype=torch.long)
        logits = model(input_ids=ids).logits[0].float()
    # Row i gives the logits that select generated token i.
    start = len(PROMPT) - 1
    return logits[start : start + len(tokens)]


def test_tp2_logits_match_hf(tiny_exaone4_dir):
    """TP2: the vLLM raw logits must match the HF reference (three-way)."""
    with _llm(tiny_exaone4_dir, tp_size=2) as llm:
        completion = _generate(llm)
    tokens = list(completion.token_ids)
    assert len(tokens) == MAX_TOKENS
    assert completion.logprobs is not None and len(completion.logprobs) == len(
        tokens
    ), "vLLM did not return logprobs for each generated token"

    ref_fp32 = _hf_teacher_forced_logits(tiny_exaone4_dir, tokens, torch.float32)
    ref_bf16 = _hf_teacher_forced_logits(tiny_exaone4_dir, tokens, torch.bfloat16)

    base_floor = BASE_FLOOR_FRACTION * ref_fp32.abs().max().item()
    for step, (token, step_logprobs) in enumerate(zip(tokens, completion.logprobs)):
        ids = sorted(step_logprobs)
        assert token in ids
        got = torch.tensor([step_logprobs[i].logprob for i in ids])
        fp32 = ref_fp32[step, ids]
        bf16 = ref_bf16[step, ids]
        tgt_err = (got - fp32).abs().max().item()
        base_err = max((bf16 - fp32).abs().max().item(), base_floor)
        print(
            f"step {step}: token={token} tgt_linf={tgt_err:.5f} "
            f"base_linf={base_err:.5f} ratio={tgt_err / base_err:.2f}"
        )
        assert tgt_err <= LINF_RATIO * base_err, (
            f"Step {step}: the vLLM logit error {tgt_err:.5f} is more than "
            f"{LINF_RATIO}x the BF16 reference error {base_err:.5f}."
        )

        # Greedy token: compare with the FP32 argmax when the margin is
        # larger than the error bound. A smaller margin is a valid tie.
        top2 = torch.topk(ref_fp32[step], 2)
        margin = (top2.values[0] - top2.values[1]).item()
        if margin > 2 * LINF_RATIO * base_err:
            assert token == top2.indices[0].item(), (
                f"Step {step}: vLLM selected token {token}, but the FP32 "
                f"argmax is {top2.indices[0].item()} (margin {margin:.4f})."
            )


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_tp1_runs_neuron_implementation(tiny_exaone4_dir):
    """TP1: the Neuron EXAONE 4.0 class must run both phases on every layer.

    TP1 keeps the worker in this process, so the test can read the dispatch
    lines from the model logger. TP1 also uses 2 KV heads for each rank, the
    per-head block table path. CPU mode runs an accepted NKI kernel in the
    simulator, but the tiny geometry is not the device geometry. Thus this test
    does not check the kernel field. The device DI test checks it.
    """
    logger = logging.getLogger(neuron_exaone4_model.__name__)
    handler = _ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with _llm(tiny_exaone4_dir, tp_size=1) as llm:
            completion = _generate(llm)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert len(completion.token_ids) == MAX_TOKENS

    seen = set()
    for line in handler.lines:
        match = _DISPATCH_RE.search(line)
        if match:
            phase, component, layer, _, rank, _ = match.groups()
            seen.add((phase, component, int(layer), int(rank)))
    assert seen, "No [exaone4_dispatch] line. The Neuron class did not run."

    missing = []
    for layer in range(NUM_LAYERS):
        for component in ("qkv_proj", "o_proj", "mlp"):
            if ("prefill", component, layer, 0) not in seen:
                missing.append(("prefill", component, layer))
        if not any(
            ("prefill", c, layer, 0) in seen
            for c in ("flash_attention", "segmented_attention")
        ):
            missing.append(("prefill", "attention", layer))
        for component in ("attention_decode", "mlp"):
            if ("decode", component, layer, 0) not in seen:
                missing.append(("decode", component, layer))
    for component in ("decode_mask_sliding", "decode_mask_full"):
        if not any(s[0] == "decode" and s[1] == component for s in seen):
            missing.append(("decode", component, None))
    assert not missing, f"Missing dispatch lines: {missing}"
