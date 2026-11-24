# Contrib Model: Qwen3-30B-A3B (Text, MoE)

[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) is a sparse
mixture-of-experts decoder-only language model with 30.5B parameters, of which
about 3.3B are active per token: 48 layers, hidden size 2048, grouped-query
attention with 32 query heads and 4 KV heads (head_dim 128) with per-head
Q/K RMSNorm and RoPE (theta 1e6), and in every layer a sparse MoE block of 128
SiLU-gated experts (expert intermediate size 768) with softmax top-8 routing
and renormalized top-k weights. The vocabulary has 151936 ids and an untied LM
head. This folder serves it (architecture `Qwen3MoeForCausalLM`,
`model_type: qwen3_moe`) with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism and expert parallelism. The
implementation is in `src/`, and [src/README.md](src/README.md) describes the
architecture, the kernel paths and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/qwen3_moe
```

The install adds a `vllm.general_plugins` entry point (`qwen3_moe_neuron`)
that registers `Qwen3MoeForCausalLM` in every vLLM process, so `vllm serve`
and `LLM(...)` with tensor parallelism find it with no other setup. vLLM ships
its own GPU `Qwen3MoeForCausalLM`; this registration replaces it while the
package is installed. `pip uninstall vllm-neuron-contrib-qwen3-moe` removes
it.

The attention, the norms, the RoPE, the embedding and the LM head come from
the Qwen3 dense model that vLLM Neuron 0.29.0.1.2.0 ships
(`vllm_neuron.model.qwen3`); `src/` adds the sparse MoE block, the decoder
layer and the weight loading.

### Serve with the OpenAI-compatible server

The configuration below is the batch-4 configuration of the integration test:
one trn2.48xlarge node, TP 16 on 16 NeuronCores (LNC 2) with expert
parallelism over the same 16 ranks (8 complete experts per rank), a 9216-token
model length (room for an 8192-token prompt and 1024 generated tokens),
2048-token prefill chunks and up to 4 sequences. The first start compiles the
48-layer graphs, which takes longer than the default engine-start (600 s) and
per-graph compile (600 s) timeouts, so raise them:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve Qwen/Qwen3-30B-A3B \
    --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-model-len 9216 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [2048], "kv_segment_size_buckets": [2048], "num_seqs_buckets": [4]}}'
```

`--max-num-batched-tokens 2048` below `--max-model-len 9216` with
`kv_segment_size_buckets` `[2048]` makes prefill run as segmented (chunked)
prefill with one 2048-token graph and a 2048-token KV segment: an 8192-token
prompt takes 4 prefill steps, and each chunk after the first attends to the
K/V of the earlier chunks in the paged KV cache. A shorter prompt pads to the
one 2048-token prefill graph.

`--enable-expert-parallel` is part of the supported configuration. Without
it, each rank holds all 128 experts with a 48-column shard of the expert
intermediate dimension at TP 16; that shard is below the 128-column minimum of
the prefill MoE kernel, so prefill falls back to the PyTorch MoE path and the
model logs a warning. The tests do not run that layout on device.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-30B-A3B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the package installed, nothing else is needed to register the model (set
the three timeouts above in the environment for the first, compiling run):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    tensor_parallel_size=16,
    enable_expert_parallel=True,
    dtype="bfloat16",
    max_model_len=9216,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [2048],
            "kv_segment_size_buckets": [2048],
            "num_seqs_buckets": [4],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your own script before you
construct `LLM(...)` (option 1 in [contrib/README.md](../../README.md)), as
`test/integration/test_model.py` does:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/qwen3_moe/src/, under whatever name you gave it.
from your_package.factory import Qwen3MoeForCausalLM

ModelRegistry.register_model("Qwen3MoeForCausalLM", Qwen3MoeForCausalLM)
```

That registration reaches vLLM's engine-core and worker processes only
because the offline path starts them with the `fork` start method, the default
for a Python script. Under `VLLM_WORKER_MULTIPROC_METHOD=spawn` a child
re-imports from scratch and resolves vLLM's own `Qwen3MoeForCausalLM`
instead; use the package install in that case, and for `vllm serve`.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See
[docs/guides/reference-configuration.md](../../../docs/guides/reference-configuration.md)
for the `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (trn2.48xlarge)   | TP 16 with expert parallelism, one node, 9216-token model length; see the Validation report | Not tested |
| Trn3                   | Not tested | Not tested |

CPU mode (`VLLM_NEURON_CPU_MODE=1`) on vLLM Neuron 0.29.0.1.2.0 (vLLM 0.29.0,
transformers 5.17.0, Python 3.14): the unit tests pass, see
[Testing](#testing).

## Example Checkpoints

- <https://huggingface.co/Qwen/Qwen3-30B-A3B> — the checkpoint the tests run.
  They pin revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

Other BF16 `Qwen3MoeForCausalLM` checkpoints have the same architecture, but
the tests do not run them, and a different expert count, head count or
hidden size changes the parallel layout (`src/README.md` lists the size
checks). The factory rejects a non-BF16 or quantized checkpoint (for example
the FP8 release), dense MLP layers (`mlp_only_layers`, `decoder_sparse_step`
other than 1), `norm_topk_prob: false`, sliding-window attention, an attention
bias, a hidden activation other than SiLU, and a RoPE scaling other than the
default (for example YaRN).

## Testing

The integration test reads the checkpoint from `QWEN3_MOE_CHECKPOINT_PATH`, a
local copy of the pinned revision. Without it, the test downloads the pinned
revision (about 61 GB) from the HuggingFace hub. The unit tests build tiny
random-weight checkpoints and download nothing.

### Integration test

Serves the model through vLLM on Neuron at TP 16 with expert parallelism on
one node, with the offline `LLM(...)` API, in the configuration of the serve
example above (with `max_num_seqs` equal to the batch size): a 9216-token
model length, segmented prefill in 2048-token chunks (one 2048-token prefill
graph, KV segment 2048), BF16 and on-device greedy sampling. The test builds
two prompts from fixed text in the test file, a prose document whose sections
repeat in a fixed order and a Python module of small generated functions, and
cuts each one at a sentence or function boundary to 8000 to 8192 tokens of the
checkpoint's tokenizer. Each prompt runs as 4 prefill chunks. The test
validates the raw logits of 16 decoded tokens per prompt against a
HuggingFace reference (FP32 baseline and BF16 teacher-forced pass), at batch 1
and batch 4 (four copies of each prompt), cases `[1-9216]` and `[4-9216]`.
The tolerance map is the bf16 dense-golden map of the template (argmax bound
1e-5; top-5 0.03, top-50 0.05, top-1000 0.08, all 0.09), and the test also
requires `pp_max_linf_1.5x` and `pp_max_l2_1.5x` on every prompt. It needs 16
NeuronCores:

```bash
QWEN3_MOE_CHECKPOINT_PATH=/path/to/Qwen3-30B-A3B \
    pytest contrib/models/qwen3_moe/test/integration/test_model.py \
    -v -s --timeout=18000
```

A first run computes the HuggingFace references on the host CPU: the FP32
model (about 122 GB of host memory) and then the BF16 model, one at a time,
with SDPA attention. The run then compiles the TP 16 graphs for each case. The
test sets `VLLM_ENGINE_READY_TIMEOUT_S=7200`,
`NEURON_LIBTORCH_COMPILATION_TIMEOUT=3600` and
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` unless they are already set.

The test packs the vLLM logits with its own `_full_vocab_generate_fn` instead
of the plugin's `create_offline_vllm_generate_fn`, because the model scores
151936 ids and the tokenizer has 151669 tokens (see the function's
docstring), and it maps the token strings of the ids past the tokenizer to
`""` so that vLLM can decode their logprobs. It also sets `ignore_eos`,
because the HF reference always decodes 16 tokens. The comparison and the
tolerance map are the plugin's.

### Unit tests

No Neuron hardware and no download are needed:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_moe/test/unit/ -v -s --timeout=3600
```

- `test_loading.py`: the registration contract (architecture string, vLLM's
  model protocol, the plugin entry point), the config parse of the pinned
  `config.json` and the factory checks, the weight mapping of the 48-layer,
  128-expert model against a fingerprint of every tensor name in the pinned
  checkpoint index, the expert gate/up and down loaders with and without the
  expert-parallel selection, and a TP 2 load of a tiny random-weight
  checkpoint through `load_weights` (with EP 2 and without EP) against slices
  that the test takes from the checkpoint.
- `test_moe_block.py`: the sparse MoE block of a tiny random-weight
  checkpoint on 2 ranks, with EP 2 and without EP, for a 256-token prefill
  (with 16 bucket-padding tokens) and a batch-4 decode step, compared
  three-way (FP32 HF, BF16 HF, BF16 `src/`) with the HF
  `Qwen3MoeSparseMoeBlock`.
- `test_tiny_model.py`: the whole model of `src/` on 2 ranks (load, paged KV
  cache, prefill and 3 decode steps through the model's own forward) on a
  2-layer random-weight checkpoint with 2 query heads and 1 KV head per rank,
  as the real model has at TP 16, compared three-way with the HF
  `Qwen3MoeForCausalLM` logits. It runs single-step flash-attention prefill
  with EP 2, and segmented (chunked) prefill of a 160-token prompt in two
  128-token chunks with EP 2 and without EP.

In CPU mode (without `NKI_SIMULATOR=1`) every NKI op runs its PyTorch
fallback, so the unit tests check the model code and the layouts, not the NKI
kernels.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_qwen3_moe_src`, so the folder keeps working when it is copied into an
application. `test/qwen3_moe_test_utils.py` holds the pinned checkpoint
metadata and the tiny-checkpoint helpers, and `test/conftest.py` puts the
`test/` directory on `sys.path` and shuts down the plugin's `MPExecutor`
worker pool at the end of the session.

### Not validated

- Flash-attention (single-step) prefill on device: the device configuration
  uses segmented prefill. CPU mode covers it with a tiny model.
- TP 16 without expert parallelism, and any other TP size, on device (CPU
  mode covers TP 2 with and without EP on tiny models).
- Context beyond 9216 tokens (the checkpoint has 40960 positions).
- Logits beyond the first 16 generated tokens of each prompt.
- Batch sizes other than 1 and 4 on device.
- Prefix caching, speculative decoding, data parallelism and Trn3.

## Validation report

The validation report for the 9216-token configuration (8k-token prompts,
batch 1 and 4, TP 16 with expert parallelism) is pending. It will be filled in
from the device results of `test/integration/test_model.py` and of the serve
example above.
