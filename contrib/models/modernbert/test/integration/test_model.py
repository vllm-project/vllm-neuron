# SPDX-License-Identifier: Apache-2.0
"""Integration test for the contrib ModernBERT model.

Checkpoint: ibm-granite/granite-embedding-small-english-r2.

The test serves the model through vLLM on Neuron with tensor parallelism 2,
asks the engine for embeddings, and compares each embedding with the
HuggingFace ``ModernBertModel`` reference.

Why this test is not the template's logit test
----------------------------------------------
ModernBERT is a bidirectional encoder, and this checkpoint is an embedding
(pooling) model: it has no LM head, no sampler and no decode phase, so there
are no per-token logits and no greedy sequence to validate. The contrib
template's logit shape (``get_cached_reference_goldens`` plus
``multi_prompt_logit_validation``, with ``max_logprobs=-1``,
``logprobs_mode="raw_logits"`` and ``on_device_sampling_config={"all_greedy":
True}``) has nothing to act on here, and forcing it would assert something the
model cannot produce. The test keeps the template's structure and its
three-way comparison (FP32 HF baseline, BF16 HF expected, BF16 Neuron actual)
and swaps the metric for the one the plugin defines for pooling models:
``assert_pooling_close_three_way(task="embed")``, a per-vector cosine floor.
Embeddings are consumed by cosine ranking, so direction is what matters.

Registration
------------
A contrib model is not in ``vllm_neuron/model/registry.py``. This test
registers the architecture with vLLM itself, before it constructs the engine,
which is option 1 of contrib/README.md ("How do I register a custom model with
vLLM?"). It registers the class object rather than a ``"module:Class"`` string,
because the modeling code here is loaded by file location (see the loader
below) and so has no importable dotted path.

The registration has to be in effect in three processes: this one, which
resolves the architecture while it builds the engine config (the pooling runner
comes off the registered class); the engine-core process; and the worker, where
``ModelRegistry.resolve_model_cls`` loads the class
(``neuron_model_runner_v2.py``). The plugin registers its own models inside the
worker (``NeuronWorker.__init__`` iterates ``get_models()``) and a contrib model
is deliberately not in that list, so what carries this registration into the
children is ``fork`` inheritance. Nothing propagates it explicitly:

  * Offline ``LLM(...)``, which is what this test and the README example use,
    runs under the default ``fork`` start method, so the engine-core process
    and the ``MultiprocExecutor`` workers inherit this registry. The pooling
    runner changes nothing here: ``llm.embed`` drives the same engine-core and
    worker processes, and the same ``resolve_model_cls`` call site, as the
    generate path.
  * A vLLM CLI entrypoint (``vllm serve``) starts the engine with ``spawn``
    instead. A spawned child is a fresh interpreter that inherits no registry
    and never runs this call, so neither this class object nor a
    ``"module:Class"`` string registered from a user process reaches it. Use
    option 2 of contrib/README.md for serving.

Geometry
--------
The recorded runs of this port drove TP 2 as one rank on each of 2
trn2.48xlarge nodes. This test builds one offline engine in one process, so
TP 2 needs 2 Neuron cores reachable from that process. See the model README's
"Validation report" for what the recorded evidence covers.

Running
-------
    pytest contrib/models/modernbert/test/integration/test_model.py \
        -v -s --timeout=3600
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Load this folder's own ``src/`` by file location, not through a package path
# rooted at the repository: a contrib model has to keep working when the folder
# is copied into an application. ``test/_local_src.py`` is the loader, and it is
# loaded the same way, under a name no other contrib model can collide with.
_LOADER_NAME = "contrib_modernbert_local_src"
if _LOADER_NAME in sys.modules:
    _local_src = sys.modules[_LOADER_NAME]
else:
    _loader_spec = importlib.util.spec_from_file_location(
        _LOADER_NAME, Path(__file__).resolve().parents[1] / "_local_src.py"
    )
    _local_src = importlib.util.module_from_spec(_loader_spec)
    sys.modules[_LOADER_NAME] = _local_src
    _loader_spec.loader.exec_module(_local_src)

common = _local_src.load_test_module("module/common.py")

# The architecture string of the checkpoint's ``config.json``, and the factory
# class that implements it. The class comes from the file-location loader, so
# it has no importable dotted path and is registered as the class object; see
# "Registration" above for what that means for the engine's child processes.
ARCHITECTURE = "ModernBertModel"
FACTORY = _local_src.load().ModernBertModel

# ibm-granite/granite-embedding-small-english-r2 at the revision the port pins.
MODEL_ID = common.MODEL_ID
REVISION = common.MODEL_REVISION

# Tensor parallelism. The model has 12 heads and an intermediate size of 1536,
# so TP 2 gives 6 heads and 768 channels per rank. TP 2 is the degree the
# recorded runs used.
TP_SIZE = common.TP_SIZE

# Context length of the engine, and the token counts of the inputs. Both come
# from the port's own device and serving tests.
MAX_MODEL_LEN = 2048
SEQ_LEN = 1024

# Cosine floor of the port's device and serving tests. The three-way form also
# reports the BF16-HF-against-FP32-HF cosine, so a floor failure separates a
# port defect from the BF16 precision floor itself.
MIN_COS = 0.99

# Deterministic results across runs.
torch.manual_seed(common.SEED)


@pytest.mark.integration
@pytest.mark.parametrize(
    "lengths",
    [
        # One 1024-token input, then two inputs in one batch. Each case
        # compiles its own graphs, so keep the list short.
        [SEQ_LEN],
        [700, 300],
    ],
    ids=["single_1024", "batch_700_300"],
)
def test_model_embeddings(lengths: list[int]) -> None:
    """Serve ModernBERT through vLLM and validate its embeddings.

    Args:
        lengths: Token count of each input of the batch.
    """

    from vllm import LLM, ModelRegistry
    from vllm.config import PoolerConfig
    from vllm.inputs import TokensPrompt

    from vllm_neuron.accuracy.testing import assert_pooling_close_three_way

    print(f"Testing model with parameters: {lengths=}")

    # Register before the engine is constructed: this process resolves the
    # architecture while it builds the config, and its children inherit the
    # registry (see "Registration" in the module docstring).
    ModelRegistry.register_model(ARCHITECTURE, FACTORY)

    # A local snapshot of the pinned revision (MODERNBERT_CHECKPOINT overrides).
    checkpoint = common.resolve_checkpoint()
    sequences = common.make_token_ids(checkpoint, lengths)

    # The HF reference in FP32 (baseline) and BF16 (expected), CLS-pooled and
    # L2-normalized, which is the pooling the model card specifies.
    hf_fp32, hf_bf16 = common.load_hf_models(checkpoint)
    baseline = torch.stack(
        [common.cls_embedding(common.hf_hidden_states(hf_fp32, s)) for s in sequences]
    )
    expected = torch.stack(
        [common.cls_embedding(common.hf_hidden_states(hf_bf16, s)) for s in sequences]
    )

    # The pooling runner is required: the model rejects the generate runner,
    # prefix caching (the attention is bidirectional) and an FP8 KV cache.
    # The checkpoint's ``modules.json`` has no Normalize module, so vLLM would
    # default to raw CLS vectors; the pooler config sets the CLS pooling and
    # the normalization of the model card, as the port's serving test does.
    # Cosine is scale-invariant, so the normalization does not change the
    # comparison, but the pooling type does.
    llm = LLM(
        model=checkpoint,
        runner="pooling",
        tensor_parallel_size=TP_SIZE,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=len(lengths),
        enable_prefix_caching=False,
        pooler_config=PoolerConfig(seq_pooling_type="CLS", use_activation=True),
    )
    try:
        outputs = llm.embed(
            [TokensPrompt(prompt_token_ids=s.tolist()) for s in sequences]
        )
        actual = torch.stack(
            [
                torch.tensor(out.outputs.embedding, dtype=torch.float32)
                for out in outputs
            ]
        )
        assert actual.shape == baseline.shape, (
            f"The engine returned {tuple(actual.shape)} embeddings, but the "
            f"reference has {tuple(baseline.shape)}."
        )
        tag = "x".join(str(n) for n in lengths)
        result = assert_pooling_close_three_way(
            baseline=baseline,
            expected=expected,
            actual=actual,
            task="embed",
            name=f"modernbert/integration/embed_{tag}",
            min_cos=MIN_COS,
        )
        print(
            f"worst cos {result.min_tgt_cos:.6f} (floor {MIN_COS}), "
            f"mean cos {result.mean_tgt_cos:.6f}, "
            f"BF16-HF worst cos {result.min_base_cos:.6f}"
        )
        assert result.passed, f"Embedding validation failed for {lengths=}"
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {lengths=}")
