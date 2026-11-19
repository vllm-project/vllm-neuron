# Contrib Model: Qwen3.8 27B (text)

[Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) is the dense member of the Qwen3.5 family (architecture
`Qwen3_5ForConditionalGeneration`): 64 layers as `[3 gated delta net + 1 gated attention] x 16`, hidden size 5120.
The 48 gated delta net (GDN) layers have 16 key heads and 48 value heads of 128 and a 4-wide causal convolution;
the 16 attention layers have 24 query heads and 4 KV heads of 256, an output gate, zero-centred QK norms and partial
RoPE over 64 of the 256 dims (theta 1e7). The MLP is a dense SwiGLU (intermediate 17408).

This folder serves the text decoder in BF16 with vLLM on AWS Trainium2 through the `vllm-neuron` plugin, on one
`trn2.3xlarge` (TP=8, one rank per physical NeuronCore), with a 32k context, tool calling and reasoning parsing.
The vision tower is not loaded.

> **Note:** Contrib models are community contributions and may have limited testing or performance optimization
> compared to the officially-supported models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Attribution

The NKI kernels in `src/kernels/` are the Qwen3.5 per-layer kernels of
[sglang-neuron](https://github.com/sgl-project/sglang-neuron) (Apache-2.0), `sglang_neuron/models/qwen35/kernels/giga`,
copied without changes; `src/kernels/README.md` lists the files, their hashes and the commit they come from. The
weight packing (`src/weights.py`) is the same repository's Qwen3.8 packing.

## Usage

Use a Neuron environment with `vllm-neuron` 0.29.0.1.2.0, for example the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`, then
install this model. The install adds a `vllm.general_plugins` entry point that registers Qwen3.8 in every vLLM
process:

```bash
pip install -e contrib/models/qwen38
```

The kernels run at logical NeuronCore config 1, one rank per physical core:

```bash
export NEURON_LOGICAL_NC_CONFIG=1
export NEURON_SKIP_EFA_AFFINITY=1
```

### Serve with the OpenAI-compatible server

```bash
VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.22 vllm serve Qwen/Qwen3.8-27B \
    --served-model-name qwen38 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --block-size 2048 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --optimization-level 1 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [2048],
            "num_seqs_buckets": [1, 2, 4],
            "decode_context_length_buckets": [4096, 16384],
            "on_device_sampling_config": {}
        }
    }'
```

Engine start, including compilation, took 750 s on a `trn2.3xlarge` (engine initialization time in the vLLM log).
The KV cache holds 98,304 tokens: three requests at the full 32k context, or more at shorter ones. The settings:

- **`--tensor-parallel-size 8` and LNC 1**: the kernels are built for one rank per physical NeuronCore. Per rank
  the weights are 6.42 GiB of the core's 12 GiB.
- **`--no-enable-prefix-caching`** (required, the model refuses to load otherwise): the GDN state of a request lives
  in its first KV block (`src/README.md`), so two requests must never share a block.
- **`--optimization-level 1`** (required, the model refuses to load otherwise): at the plugin's default level the
  prefill graphs compiled, but their output disagreed with Hugging Face on prompts longer than 512 tokens (the
  first generated token was already different, with margins of 4.5 to 8.6 logits). At `-O1` the same prompts agree.
  The cause in the compiler was not investigated.
- **`--block-size 2048`**: every block also carries a GDN state region of about 18 MiB per rank, so a larger block
  costs less memory per token (`src/README.md`, Memory per rank). The attention kernels read pages of a power of
  two of at least 128 tokens.
- **`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.22`**: at 0.30 and 0.40 the runtime ran out of device memory (a
  512 MiB prefill scratchpad, then the DMA rings of a graph loaded while serving).
- **`--max-num-batched-tokens 2048` and `num_batched_tokens_buckets: [2048]`**: prompts are prefilled in chunks of
  one block; a chunk that continues a prompt starts from the GDN state the previous chunk wrote.
- **`decode_context_length_buckets: [4096, 16384]`**: decode steps of shorter requests read fewer pages.
- **`on_device_sampling_config: {}`**: on-device sampling with per-request temperature, top-p and top-k. With
  `{"all_greedy": true}` every request decodes greedily.
- **`--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3`**: the checkpoint's chat template writes tool
  calls as `<tool_call><function=...><parameter=...>` and reasoning inside `<think>`.

Qwen recommends `temperature=0.6, top_p=0.95, top_k=20` for this model. Greedy decoding can repeat itself inside long
reasoning (see Accuracy).

**A 16k configuration.** `--max-model-len 16384 --block-size 1024`, `num_batched_tokens_buckets: [1024, 2048]`,
`decode_context_length_buckets: [4096]`: 71,680 cache tokens (four requests at 16k), engine start about 7 minutes,
a shorter time to first token for short prompts (Performance).

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen38", "messages": [{"role": "user", "content": "What is the capital of France?"}],
         "max_tokens": 1024, "temperature": 0.6, "top_p": 0.95, "top_k": 20}'
```

`examples/agent_demo.py` runs a small tool-using agent against the server and prints what each step costs.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.3xlarge`, TP=8, LNC=1) | Working | Not tested |
| Trn3 | Not tested | Not tested |

Tested on the Neuron DLC `pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` as shipped
(vLLM 0.29.0, vLLM Neuron 0.29.0.1.2.0, transformers 5.17.0, neuronx-cc 2.28.7631), with only this folder installed.

| Feature | Status |
|---|---|
| Text generation, chunked prefill, 32k context | ✅ |
| Several sequences per server (`--max-num-seqs` 4) | ✅ |
| Tool calling (`qwen3_coder` parser), streaming and not | ✅ |
| Reasoning parsing (`qwen3`) | ✅ |
| On-device sampling (temperature, top-p, top-k) | ✅ |
| Per-request `seed` | ❌ the plugin builds a `torch.Generator` on the Neuron device for it, which the device does not implement, and the engine stops (`neuron_model_runner._update_states`); this is not specific to this model |
| Asynchronous scheduling | ✅ (the model consumes the runner's in-graph reorder arguments) |
| BF16 | ✅ |
| Tensor parallelism | ✅ TP=8 at LNC 1 only |
| Prefix caching | ❌ refused (the GDN state is kept per request, in its first block) |
| Speculative decoding (the checkpoint's MTP head) | ❌ not implemented |
| FP8 weights or KV cache | ❌ not implemented |
| Vision inputs | ❌ |

## Accuracy

**Against Hugging Face.** Greedy tokens from the full model on Neuron were compared with the Hugging Face reference
(transformers 5.17, CPU, BF16), teacher-forced on the Neuron-generated sequence: at each step, whether the Neuron
token is the reference's argmax, and at a disagreement the margin (the reference's best logit minus its logit for
the Neuron token) and the reference's own gap between its first and second token. With the 16k configuration, 24
tokens per prompt of natural text:

| Prompt tokens | Covers | Argmax agreement |
|---|---|---|
| 5 | basic path | 24/24 |
| 867 | one prefill chunk | 24/24 |
| 2165 | two chunks | 24/24 |
| 7779 | four chunks, the long decode graph | 24/24 |
| 14582 | eight chunks | 23/24 (step 8: margin 1.75, the reference's own top-2 gap) |

With block size 512 and eight prompts of 5 to 3000 tokens (32 tokens each), the other disagreements were exact ties
(margins 0.000 and 0.125, equal to the reference's top-2 gaps), and a batch of eight requests gave the same tokens as
the requests one at a time except on one exact tie. Prompts past 14.6k tokens were not compared with the reference:
on CPU, its attention over 28k tokens needs tens of GiB per layer.

**Needle in a haystack**, 32k configuration: a 4-digit code hidden at 10, 50 and 90 percent of filler text, asked for
at the end (chat, greedy). Found 15 of 15, at 3,863, 7,643, 15,308, 22,973 and 28,643 prompt tokens.

**GSM8K**, 16k configuration, five-shot, greedy, the first questions of the test split (the few-shot examples are
its first five lines, the answer is the last number of the output): 0.990 on the first 100 questions, 0.965 on the
first 200, 0 unreadable answers.

**Tool calling**, 32k configuration: three tools (a weather lookup, a shell command, a file read with integer line
arguments), three requests per tool, at the recommended sampling and greedy, streamed and not: 36 of 36 calls named
the expected function with arguments that parsed and held the expected values. In a second turn with the tool's
result, both sampling settings answered from it.

**Greedy repetition.** One chat prompt, decoded greedily, began repeating a list of invented function names in its
reasoning after about 436 tokens. Teacher-forced over the first 800 generated tokens, the Hugging Face reference
agreed with 793; its first disagreement was an exact tie (margin 0.000). The reference therefore predicts the same
repetition from the same context: it comes from greedy decoding of the checkpoint, not from this implementation.

## Performance

Measured on one `trn2.3xlarge` (TP=8, LNC=1, BF16), SDK 2.33, vLLM Neuron 0.29.0.1.2.0, greedy decoding,
asynchronous scheduling, on an otherwise idle host. Prompts are random token ids (no prefix reuse); TPOT is the time
between the first and the last output token divided by the output tokens minus one, with the token count taken from
the response's usage record (a stream chunk can carry several tokens).

**One client**, 128 new tokens, two runs each:

| Prompt tokens | 32k config: TTFT | 32k config: TPOT | 16k config: TTFT | 16k config: TPOT |
|---|---|---|---|---|
| 128 | 0.64 s | 29.2 ms | 0.32 s | 29.0 ms |
| 1024 | | | 0.32 s | 29.0 ms |
| 4096 | 1.28 to 1.29 s | 31.2 to 31.4 ms | 1.19 to 1.20 s | 31.2 ms |
| 8192 | | | 2.37 to 2.38 s | 31.2 ms |
| 15360 to 16384 | 5.08 to 5.11 s | 34.9 to 35.1 ms | 4.46 s | 31.1 to 31.2 ms |
| 30000 | 9.53 s | 34.9 ms | | |

The first decode step of a short prompt waits for a whole 2048-token prefill chunk in the 32k configuration, which
is the difference in TTFT at 128 tokens.

**Several clients**, 1024 prompt tokens and 256 new tokens per client:

| Clients | 32k config: tokens/s | TPOT p50 | 16k config: tokens/s | TPOT p50 |
|---|---|---|---|---|
| 1 | 31.6 | 29.2 ms | 33.1 | 29.0 ms |
| 2 | 50.8 | 35.6 ms | 54.5 | 34.9 ms |
| 3 | 66.3 | 40.4 ms | | |
| 4 | | | 94.2 | 39.5 ms |

**Soak**, 32k configuration: three clients for 45 minutes, each sending requests of 512 to 8192 random prompt tokens
and 256 new tokens, half greedy and half at the recommended sampling. 525 requests completed and none failed; the
server log held no error. Per five-minute window, output throughput was 43.5 to 48.6 tokens/s, TPOT p50 43.9 to
48.9 ms and TPOT p99 56.4 to 61.3 ms, with no trend from the first window to the last.

**A tool-using agent** (`examples/agent_demo.py`, recommended sampling, 32k configuration): the model listed the
files and searched in one step, read two files in the next and answered in the third, with the right fix. With the
short system prompt each step had 437 to 813 prompt tokens and a TTFT of 0.66 to 0.67 s; with the system prompt
padded to about 15k tokens, as a coding agent's instructions would be, each step had a TTFT of 5.10 to 5.11 s,
because every step prefills the whole conversation again (no prefix cache). Decode ran at 32 to 36 tokens/s.

## Example Checkpoints

- <https://huggingface.co/Qwen/Qwen3.8-27B> (Apache-2.0)

## Testing

Install the package first, so that the TP worker processes register the model:

```bash
pip install -e contrib/models/qwen38
```

CPU tests (no Neuron hardware): the config parse, the state cache geometry and the state addressing.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen38/test/unit/ -v
```

The integration test serves the checkpoint on a `trn2.3xlarge` and checks a factual answer, a count the GDN state
must carry across decode steps, greedy determinism, batch invariance, a prompt prefilled in two chunks and a tool
call in the chat template's format:

```bash
NEURON_LOGICAL_NC_CONFIG=1 NEURON_SKIP_EFA_AFFINITY=1 pytest contrib/models/qwen38/test/integration/test_model.py -v -s
```

## TODO

- Prefix caching: keep GDN state checkpoints at chunk ends so a cached prefix can resume. It is what a multi-step
  agent with a long system prompt pays for today (Performance).
- Keep one GDN state region per running request instead of one per block (`src/README.md`, Memory per rank).
- Find why the prefill graphs disagree with the reference when compiled at `-O2`.
- Speculative decoding with the checkpoint's MTP head.
