# Contrib Model: Jina Embeddings v5 Text Nano (Text Embedding)

Serve the text embedding model `jinaai/jina-embeddings-v5-text-nano` with vLLM
on AWS Trainium through the `vllm-neuron` plugin. The backbone is a 12-layer
EuroBERT bidirectional encoder: RMSNorm, rotate-half RoPE, separate Q/K/V/O
projections without bias, and a SiLU gated MLP. It is a pooling model (the
hidden state of the last token, then L2 normalization), not a causal LM: it
has no LM head, no sampler and no decode phase. The registered architecture is
`JinaEmbeddingsV5Model`, the `architectures` value of the checkpoint.
`src/README.md` describes the implementation.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

> **Base encoder only.** The published checkpoint ships four task LoRA
> adapters (`retrieval`, `text-matching`, `clustering`, `classification`) in
> `adapters/`, and the model card selects one of them for each encode call.
> This implementation serves the **base encoder weights without any adapter**.
> Its embeddings match the HuggingFace `EuroBertModel` of the checkpoint, not
> the task-specific embeddings of the sentence-transformers or `encode()` API
> of the model card. Use it where the base embeddings are what you want, or
> merge the adapter of your task into the base weights before you serve them.
> The implementation rejects a local checkpoint that still has the
> `adapters/` directory, so that this is a decision and not an accident.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/jina_embeddings_v5
```

The install adds a `vllm.general_plugins` entry point that registers
`JinaEmbeddingsV5Model` in every vLLM process (the API server, the engine and
each tensor-parallel worker), so `vllm serve` and `LLM(...)` find it with no
other setup. `pip uninstall vllm-neuron-contrib-jina-embeddings-v5` removes the
registration.

### Checkpoint

Download the pinned revision without the `adapters/` directory (see the note
above) and use the local directory as the model path:

```bash
hf download jinaai/jina-embeddings-v5-text-nano \
    --revision 8a7f00aac812071b69403df470f1038ec85f8925 \
    --exclude "adapters/*" \
    --local-dir /data/jina-embeddings-v5-text-nano
```

The configuration and the reference modeling files of the checkpoint are
remote code, so the commands below pass `--trust-remote-code`.

### Serve with the OpenAI-compatible server

```bash
vllm serve /data/jina-embeddings-v5-text-nano \
    --served-model-name jina-embeddings-v5-text-nano \
    --runner pooling \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}' \
    --host 127.0.0.1 --port 8000
```

Several of these flags are not optional for this model:

- `--runner pooling`: the model has no LM head and rejects the generate
  runner.
- `--no-enable-prefix-caching`, and `--max-num-batched-tokens` not smaller
  than `--max-model-len`: the attention is bidirectional, so a prompt cannot
  be split across steps. The model rejects prefix caching and chunked
  prefill.
- `--no-async-scheduling`: the plugin does not support async scheduling for
  pooling models.
- `--pooler-config`: the checkpoint's `modules.json` has only a custom
  sentence-transformers module, so vLLM finds no pooling config in it. This
  sets the pooling of the checkpoint: the last token, then L2 normalization.
- `--dtype bfloat16`: `config.json` states `float32`, but the weights are
  BF16.

Then request embeddings. Prefix each input with `Query: ` or `Document: `, as
the model card does:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "jina-embeddings-v5-text-nano",
        "input": ["Query: What is the capital of France?",
                  "Document: Paris is the capital of France."]
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM
from vllm.config import PoolerConfig

llm = LLM(
    model="/data/jina-embeddings-v5-text-nano",
    runner="pooling",
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
    pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=True),
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [4],
        }
    },
)

outputs = llm.embed(["Query: What is the capital of France?"])
for output in outputs:
    print(len(output.outputs.embedding))  # 768
```

Without the `pip install`, register the factory yourself before you build the
engine: copy `src/` into your application as a package, then call
`ModelRegistry.register_model("JinaEmbeddingsV5Model", JinaEmbeddingsV5Model)`
with the class from its `factory` module (option 1 in
[contrib/README.md](../../README.md)). That registration reaches the engine
processes only under the default `fork` start method, and it does not reach
`vllm serve`.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / version | SDK 2.33 (vLLM Neuron 0.29.0.1.2.0, vLLM 0.29.0) | Other versions |
| ------------------ | ------------------------------------------------ | -------------- |
| Trn2               | See the validation report (TP 2)                 | Not tested     |
| Trn3               | Not tested                                       | Not tested     |

| Feature | Status |
|---|---|
| BF16, base encoder weights | Implemented |
| Tensor parallelism | TP 2 is the tested degree. The 12 heads and the 3072 intermediate size must divide by the TP degree; the code rejects other degrees |
| Context length | Tested at 1,024 tokens. The checkpoint allows 8,192; longer contexts need a matching `num_batched_tokens_buckets` and are not tested |
| 1-token inputs | Take a separate decode-classified path (attention equals the value vector); covered by the tests |
| Task LoRA adapters | Not supported (see the note at the top) |
| Prefix caching, chunked prefill, FP8 KV cache, quantized checkpoints, attention / MLP data parallelism, decode context parallelism | Rejected by the code |
| Async scheduling | Not supported by the plugin for pooling models |

## Example Checkpoints

- <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano> — the only
  checkpoint tested, at revision `8a7f00aac812071b69403df470f1038ec85f8925`,
  which the tests pin. Serve it without the `adapters/` directory.

Other EuroBERT-based checkpoints are untested. The factory rejects
`is_decoder=True`, attention or MLP bias, an activation other than SiLU, a
quantized checkpoint, and a RoPE scaling that changes the default
frequencies.

## Testing

All tests reach `src/` through `test/_local_src.py`, which loads it by file
location under the unique module name `contrib_jina_embeddings_v5_src`, so the
folder keeps working when it is copied into an application. They share the
port's test helper `test/jina_v5_test_utils.py` (the HF reference, the input
builders and the `MPExecutor` loaders). The tests download the pinned
revision without `adapters/` (about 440 MB) into
`$HF_HOME/contrib_jina_embeddings_v5/`, unless `JINA_V5_CHECKPOINT` points at a
local copy (see [Checkpoint](#checkpoint)). If that copy has an `adapters/`
directory, the tests use a directory of symbolic links to its other entries
under the system temporary directory.

The HF reference is the checkpoint's own `EuroBertModel` (remote code,
eager attention) in FP32 (baseline) and BF16 (expected). Each comparison is
three-way, as the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
describe.

### Integration test

The integration test serves the model offline through vLLM with TP 2 on one
node and compares the embeddings with the HF reference. It needs 2
NeuronCores:

```bash
pytest contrib/models/jina_embeddings_v5/test/integration/test_model.py \
    -v -s --timeout=3600
```

It has two cases: two 1,024-token inputs plus the two text inputs above, and
four 1-token inputs. The gate is
`assert_pooling_close_three_way(task="embed", min_cos=0.99)`, a per-vector
cosine floor against the FP32 reference. It also checks that vLLM tokenizes
the text inputs as the HF tokenizer does and that each embedding has unit
norm.

This is not the template's logit test, because an embedding model produces no
logits: `multi_prompt_logit_validation`, `max_logprobs=-1`,
`logprobs_mode="raw_logits"` and `on_device_sampling_config` have nothing to
act on. The cosine floor is the pooling metric of the plugin.

### Unit tests (CPU mode)

Registration contract and the vLLM model protocol, config mapping, RoPE
normalization, rejection of unsupported configurations, complete weight
coverage of the checkpoint, and TP 2 CPU runs of the layer-0 attention and
of the full 12-layer encoder (hidden states and pooled embeddings) in a
padded bucket:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/jina_embeddings_v5/test/unit/ -v
```

### Module tests (device)

Layer-0 attention prefill, layer-0 MLP prefill, the full encoder prefill at a
1,024-token bucket, and the full encoder on a batch of 1-token inputs, at TP 2
with both ranks on one node. They need 2 NeuronCores and skip in CPU mode:

```bash
pytest contrib/models/jina_embeddings_v5/test/module/ -v -s --timeout=3600
```

To run the port's original topology (1 rank on each of 2 nodes), set
`JINA_V5_NNODES=2` and `NODE_RANK` on both nodes, together with the
rendezvous address of node 0 and `NEURON_RT_ROOT_COMM_ID` that the plugin's
multi-node executor reads.

## Validation report

### Source and environment

<!-- Filled from the device runs on vLLM Neuron 0.29.0.1.2.0 / SDK 2.33. -->

- Plugin: vLLM Neuron 0.29.0.1.2.0 (vLLM 0.29.0), Neuron SDK 2.33.
- Instance: trn2, one node.
- Checkpoint: `jinaai/jina-embeddings-v5-text-nano` at
  `8a7f00aac812071b69403df470f1038ec85f8925`, without `adapters/`.

### Results

| Test | TP | Result |
| ---- | -- | ------ |
| Unit tests, CPU mode (`test/unit/`) | 2 (CPU) | 5 of 5 passed |
| Integration, CPU mode (`test/integration/`) | 2 (CPU) | 2 of 2 passed; worst cosine 0.99987 (1,024 tokens and text), 0.99938 (1 token); floor 0.99 |
| Integration on device (`test/integration/`) | 2 | _To be filled_ |
| Module tests on device (`test/module/`) | 2 | _Not run for this report_ |

### Exceptions

- **No logit validation.** The model produces embeddings, not logits. The
  cosine floor above is its accuracy gate. No retrieval-quality evaluation
  (MTEB or similar) was run.
- **Base encoder only.** The task LoRA adapters are not applied; see the note
  at the top.
- **Dropped: the port's 2-node online serving test.** It used a shared server
  harness and a multi-node launcher outside this folder. The offline
  integration test covers the same inputs (1,024-token token-ID inputs and
  the `Query: ` / `Document: ` text inputs) on one node.
- **Rewritten registry assertion.** The port's CPU test asserted the plugin's
  model registry. A contrib model is not in that registry, so the unit test
  checks the registration contract and the vLLM model protocol instead.
- **Rewritten checkpoint resolution.** The port resolved the checkpoint with
  a helper outside this folder. The tests now download the pinned revision
  without `adapters/`, or use `JINA_V5_CHECKPOINT`.
- **Untested:** TP degrees other than 2, contexts above 1,024 tokens, Trn3,
  and any plugin version other than 0.29.0.1.2.0.
