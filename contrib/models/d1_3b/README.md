# Contrib Model: LiquidAI d1-3B (decision model, LFM2 hybrid)

[d1-3B](https://huggingface.co/LiquidAI/d1-3B) reads a state (text or JSON) and answers
named questions in one forward pass: Choice (one of named options), Yes/No (`noul`,
P(yes)), and Score (an ordered scale). Each answer is a softmax over the options'
tokens at the answer slot, so nothing is decoded. The backbone is LFM2 (30 layers:
22 gated short-conv, 8 full-attention), post-trained from LFM2.5-VL-3B.

> **Note:** This is a community contribution. **Status:** text and image states are
> validated on Trn2 (trn2.3xlarge, TP=4) against the HF fp32 reference. The
> compatibility matrix records exactly what was tested.

## How it maps onto vLLM

- **One request per question.** The prompt for a (state, question) pair is rendered
  by the checkpoint's own code (`runner.SystemOne.render`), sent as token ids with
  `max_tokens=1`, and the raw last-position logits go through the checkpoint's
  readout (`runner.SystemOne._readout`). Questions about the same state share the
  state as a prefix, so prefix caching reuses it.
- **Conv state in the paged KV cache.** Each conv layer registers as a one-head
  attention layer whose cache holds the conv input `B*x` of every token. The conv
  reads the two previous tokens through the block table, so chunked prefill and
  prefix caching stay correct without any change to the plugin.
- **Not a hybrid model to vLLM.** The registered class does not declare vLLM's
  hybrid (mamba) interface, so vLLM manages every layer through the plain KV spec.
  It does declare the multimodal interface, with vLLM's upstream LFM2-VL processor,
  so images are tiled and expanded into tokens exactly as upstream.
- **Images: one 512 px tile per encoder block.** The processor splits a picture into
  tiles plus a thumbnail, each at most 1024 patches. With
  `encoder_attention_block_size=1024` every tile is one encoder block, and after the
  2x2 pixel-unshuffle (at most 256 tokens) one encoder-cache block. The host builds
  the per-tile inputs with data-dependent shapes (resized position embeddings, key
  mask, unshuffle gather index); the device graph runs the 27 SigLIP2 layers and the
  projector at a fixed shape per bucket and writes the encoder cache. The tower is
  replicated on every rank. The runner's `is_embed` path places the tile tokens
  between the tile-marker tokens.

## Usage

Register the architecture before constructing the engine (see
[contrib/README.md](../../README.md), Option 1), then render, run, and read out:

```python
import os

os.environ["OMP_NUM_THREADS"] = "1"  # first, before torch/vLLM: see the fork-safety note
import sys

import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, "contrib/models/d1_3b")  # or copy src/ into your package
from src import register

register()

path = "/path/to/d1-3B"  # snapshot of LiquidAI/d1-3B @ 051bcc46


def remote(module, attr):  # the checkpoint's pure-Python helpers (no model load)
    return sys.modules[get_class_from_dynamic_module(f"{module}.{attr}", path).__module__]


prompt = remote("prompt", "as_question")
tok = AutoTokenizer.from_pretrained(path)
q = prompt.as_question({"type": "noul", "instructions": "Does the customer have a receipt?"})
# Take questions, render and readout from the same prompt module (a second copy of the
# module defines different Question classes).
text = prompt.render(tok, "The customer has a receipt.", q, tok.bos_token or "",
                     prompt.default_lead("lfm2_vl"), prompt.DEFAULT_STATE_STYLE,
                     prompt.DEFAULT_SYSTEM, "desc")  # what SystemOne.render does
ids = tok.encode(text, add_special_tokens=False)

llm = LLM(
    model=path,
    tensor_parallel_size=4,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=1,
    enable_prefix_caching=False,
    max_logprobs=-1,
    logprobs_mode="raw_logits",
    async_scheduling=False,
    additional_config={
        "neuron_config": {"on_device_sampling_config": {"all_greedy": True}},
        # Required even for text-only serving (see below).
        "encoder_neuron_config": {
            "num_encoder_input_buckets": [4096, 12288],
            "encoder_attention_block_size": 1024,
            "encoder_cache_num_blocks": 25,
        },
    },
)
# Ask for the raw logits of the option tokens only. The readout is a softmax over them,
# so the vocabulary normalisation cancels. (logprobs=-1 returns all 128k logits to the
# host instead: ~0.9 s per request.)
opt = sorted({i for g in prompt.readout_ids(tok, q) for i in g})
out = llm.generate(
    [TokensPrompt(prompt_token_ids=ids)],
    SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=opt, logprobs=len(opt)),
)[0]
logits = {t: lp.logprob for t, lp in out.outputs[0].logprobs[0].items()}
# SystemOne's readout (no calibration): [P(yes), P(no)]
print(prompt.readout(tok, q, logits, None))
```

### Images

Add the encoder config and pass pictures with the prompt text. Render the text with
the checkpoint's helpers, as `SystemOne._request` does for one question:
`prefix_text(tokenizer, state, bos, state_style, system, engine._image_markup(n))`
plus `suffix_text(...)`, with each picture passed through `runner.cap_pixels`.

```python
llm = LLM(
    model=path,
    tensor_parallel_size=4,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    max_num_seqs=1,
    enable_prefix_caching=False,
    max_logprobs=-1,
    logprobs_mode="raw_logits",
    async_scheduling=False,
    limit_mm_per_prompt={"image": 1},
    additional_config={
        "neuron_config": {"on_device_sampling_config": {"all_greedy": True}},
        # Raw patches; 1024 per tile. 12288 holds the maximum of 10 tiles + thumbnail.
        "encoder_neuron_config": {
            "num_encoder_input_buckets": [4096, 12288],
            "encoder_attention_block_size": 1024,
            "encoder_cache_num_blocks": 25,
        },
    },
)
out = llm.generate(
    [{"prompt": text, "multi_modal_data": {"image": [picture]}}],
    SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=opt, logprobs=len(opt)),
)[0]
```

`encoder_attention_block_size` must be 1024 (one tile per block). The encoder config
is required for text-only serving too: the registered class is multimodal, so the
runner always builds the encoder, with defaults (2048-patch blocks, no buckets) that
do not fit d1-3B. The model fails fast with a message saying so. Set
`encoder_cache_num_blocks` explicitly (25 = two 12-tile images + a scratch block): the
derived size follows the scheduler's encoder budget, which is only
`max_num_batched_tokens` tokens when that is small (for example 512 with prefix
caching), and the runner then rejects the 12-tile bucket.

**Fork-safety note (offline API):** vLLM forks its engine processes, and the engine
runs the HF image processor on the CPU while it sizes the multimodal budget. If the
parent process has started torch's OpenMP thread pool (for example by running an HF
model on the CPU), the forked engine deadlocks in libgomp. Set the environment
variable `OMP_NUM_THREADS=1` for the serving process (before `LLM(...)` or `vllm
serve`; the forked engine inherits it), and run any CPU reference in a separate
process. `torch.set_num_threads(1)` alone is not enough: the engine resets its own
thread count (measured: it still deadlocked).

**Registration note:** the plugin's multimodal sizing helpers
(`vllm_neuron/utils/mm_utils.py`: merge factor, encoder budget, blocks per request)
look the model class up in the plugin's own `get_models()` list rather than in
vLLM's `ModelRegistry`. `register()` therefore also adds d1-3B to that list (a
process-local patch of `vllm_neuron.model.registry.get_models`). Without it a contrib
vision model gets a merge factor of 1 and an encoder budget of 0. A plugin-side fix
would be for those helpers to resolve the class through vLLM's registry.

### Environment notes for trn2.3xlarge

- `NEURON_SKIP_EFA_AFFINITY=1` (the instance has no EFA device).
- Do not set `enforce_eager` (not supported on Neuron).
- Prefix caching on Neuron needs segmented prefill: set `max_num_batched_tokens`
  to a supported segment size (512, 1024, ...) below `max_model_len`. With
  `max_num_batched_tokens >= max_model_len`, disable prefix caching.
- Pin sampling in every request (`temperature=0.0`); otherwise vLLM applies
  `generation_config.json` defaults.

## Compatibility Matrix

| Instance / SDK version | 2.33 (neuronx-cc 2.28, vllm 0.29.0, vllm-neuron 0.29.0.1.2.0) | earlier |
| ---------------------- | ------------------------------------------------------------- | ------- |
| Trn2 (trn2.3xlarge, LNC=2, TP=4) | Working: text and image states                               | Not tested |
| Trn3                   | Not tested                                                    | Not tested |

Validation runs (`test_model.py`; reference: HF fp32 on CPU; gate: top option matches
and max \|prob diff\| ≤ 0.0197). Text: 6 decisions (choice / yes-no / score over a short
and a ~600-token state). Images: 4 decisions (yes-no and choice over a 384x384 picture,
one tile, and a 1024x768 picture, tiles + thumbnail).

| Date | Runner | Config | Decisions | Max \|prob diff\| | Prefix-cache hits | Wall time (incl. compile) |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-08 | V2 | text, one-shot prefill (before the vision port) | 6/6 | 0.0020 | – | 7 min 43 s |
| 2026-11-08 | V2 | text, prefix caching + 512-token chunks (before the vision port) | 6/6 | 0.0013 | 576 of 603 / 616 | 4 min 10 s |
| 2026-11-11 | V1 | images + text, one-shot prefill | 4/4 images, 6/6 text | 0.00052 images, 0.0020 text | – | 11 min 14 s |
| 2026-11-11 | V1 | text, prefix caching + 512-token chunks | 6/6 | 0.0029 | 576 of 603 / 616 | 4 min 29 s |
| 2026-11-11 (current) | V1 | images + text, fp32 LM head, option-token readout | 4/4 images, 6/6 text | 0.00015 images, 0.0013 text | – | 9 min 48 s |
| 2026-11-11 (current) | V1 | text, prefix caching + 512-token chunks, fp32 LM head, option-token readout | 6/6 | 0.0013 | 576 of 603 / 616 | 4 min 19 s |

The multimodal registration routes the model to the V1 runner (the plugin's V2 runner
does not take multimodal models). The rows marked "current" are the code in this folder.

## Performance

Measured 2026-11-11 on trn2.3xlarge (TP=4, LNC=2, V1 runner, bf16 weights + fp32 LM head),
on the workloads and CPU fp32 golden of the standalone torch-neuronx d1-3B port
(`card_*`: the model card's scenarios). One vLLM request per (state, question), all requests
of a workload submitted together; e2e p50 including rendering, tokenization and the
option-token readout; every shape warmed before timing.

| Workload | Requests | This port | Prefix caching (opt-in) | Standalone port, shipped | Max \|prob diff\| |
| --- | --- | --- | --- | --- | --- |
| card_1q (one question, ~60 tokens) | 1 | 20.4 ms | 23.3 ms | 5.33 ms | 0.0001 |
| card_3q (3 questions, one state) | 3 | 54.6 ms | 59.3 ms | 5.72 ms | 0.0017 |
| card_3k4 (3.4k-token state) | 1 | 90.1 ms | 136 ms | 49.15 ms | 0.0003 |
| card_img384 (384 px image) | 1 | 22.8 ms | 30.5 ms | 16.07 ms | 0.0002 |
| batch8_3q (8 states x 3 questions) | 24 | 432 ms | 423 ms | 23.9 ms | 0.0068 |
| card_pack64 (64 short states) | 64 | 56 states/s | 59 states/s | 1,311 states/s (DP4) | 0.0119 |
| long1k_3q | 3 | 104 ms | 83 ms | – | 0.0009 |
| long4k_3q | 3 | 271 ms | 206 ms | 61.8 ms | 0.0067 |

All 101 decisions (the rows above plus `card_img_cats`) match the fp32 golden's top option; every max \|prob diff\| is below 0.0197.

What the numbers mean:

- **Multi-request workloads are the gap.** vllm-neuron schedules one prefill per step, so
  the questions of a state and the requests of a batch run one after another. The
  standalone port packs them into one device call (shared state read once, many states per
  call); that is where most of its 10-25x lead on `card_3q`, `batch8_3q` and `card_pack64`
  comes from.
- **Single requests** are 1.4-4x slower than the standalone port, which ships NKI kernels
  that run whole layers (activations kept on chip, in-kernel all-reduce) and a
  sequence-parallel MLP for long states. This port uses the plugin's standard functions.
- **Read only the option tokens.** `logprob_token_ids` makes the readout cost the same as the
  forward pass; `logprobs=-1` adds ~0.9 s per request.
- **fp32 LM head (`D1_FP32_LOGITS=1`, default).** With bf16 logits two options of a
  `batch8_3q` question rounded to the same value and the top option flipped (23/24); the
  fp32 head fixes it (24/24) and lowers every max \|prob diff\| (for example `card_3q`
  0.0057 -> 0.0017), for about +1 ms per request and +11% on batches. Set
  `D1_FP32_LOGITS=0` to trade that back.
- **Prefix caching** (with 1024-token chunked prefill) helps long states with several
  questions (`long4k_3q` -24%) and hurts a single long state (`card_3k4` +51%).
- **Length buckets:** the plugin's power-of-two defaults and the standalone port's serving
  buckets (`128,256,384,512,640,1024,2048,4096`) measure within 3% on these workloads.

## Example Checkpoints

- <https://huggingface.co/LiquidAI/d1-3B> (validated revision
  `051bcc464b01b9f92942b364d9586b0ef5912432`)

## Testing

The integration test compares every option probability against the HF model on
CPU in fp32. Gate: the top option matches, and every probability is within 0.0197.

```bash
# One-shot prefill, no prefix caching
NEURON_SKIP_EFA_AFFINITY=1 D1_MODEL_PATH=/path/to/d1-3B \
    pytest contrib/models/d1_3b/test/integration/test_model.py -v -s --timeout=3600

# Prefix caching + 512-token chunked prefill (conv state read back from the cache)
NEURON_SKIP_EFA_AFFINITY=1 D1_MODEL_PATH=/path/to/d1-3B D1_PREFIX_CACHING=1 \
    pytest contrib/models/d1_3b/test/integration/test_model.py -v -s --timeout=3600

# Images too: builds the vision tower, adds 4 image decisions (a 384x384 picture,
# one tile; a 1024x768 picture, tiles + thumbnail)
NEURON_SKIP_EFA_AFFINITY=1 D1_MODEL_PATH=/path/to/d1-3B D1_IMAGES=1 \
    pytest contrib/models/d1_3b/test/integration/test_model.py -v -s --timeout=3600

# CPU module test of the vision tower + projector vs HF (torch + transformers only)
pytest contrib/models/d1_3b/test/module/test_vision.py -v
```

`D1_TP` sets the tensor-parallel degree (default 4). The test computes the fp32 CPU
reference first, in a separate process (`test/integration/_reference.py`), so its
memory is freed before the Neuron load and the vLLM process tree stays free of
OpenMP state (see the fork-safety note).
With `D1_PREFIX_CACHING=1` the test also asserts that later questions about a state
hit the prefix cache, so the conv state is read back from the paged cache.

If you launch the test from a remote command runner (for example SSM Run Command),
detach it (for example with `systemd-run`): the runner's execution timeout kills
child processes, including `nohup` ones, during the first compile.

## Known limitations and next steps

- The vision tower is replicated on every rank (no encoder TP/DP split) and uses
  plain torch ops (no NKI kernels); every rank encodes every tile.
- One image per prompt is tested (`limit_mm_per_prompt={"image": 1}`).
- Throughput: vllm-neuron runs one prefill per step, so requests are not packed the
  way the standalone torch-neuronx port packs many states into one call (see
  Performance).
- Performance work from the standalone port (NKI block kernels, multi-row and
  shared-trunk kernels, sequence-parallel MLP for long states, fp8 MLP) is not included.
