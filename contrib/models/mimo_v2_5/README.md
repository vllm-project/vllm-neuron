# Contrib Model: Xiaomi MiMo-V2.5 (text decoder)

`MiMoV2ForCausalLM` (`model_type="mimo_v2"`) served with vLLM on AWS Trainium
through the `vllm-neuron` plugin. This is the **text decoder only**: the released
checkpoint also carries vision and audio towers and an MTP head, and this port
builds none of them.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested. **This port has not been run on the 0.29 / SDK 2.33
> stack yet.** It was brought up on vllm-neuron 0.21 (PR #40) and ported here
> without hardware.

| | |
|---|---|
| layers | 48: layer 0 dense SwiGLU MLP (16384), layers 1-47 MoE |
| MoE | 256 routed experts of 2048, top-8, sigmoid `noaux_tc` router with a selection-only correction bias |
| attention | hybrid: 9 full-attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47) and 39 sliding-window layers (window 128) |
| heads | 64 Q heads; KV heads 4 (full) / 8 (SWA); Q/K head dim 192, V head dim 128 |
| RoPE | partial (first 64 of 192 dims), base 1e7 on full layers and 1e4 on SWA layers |
| other | learned attention sink per Q head on SWA layers; V scaled by 0.707 before the cache write |
| vocab / hidden | 152576 / 4096 |
| checkpoint | 128x128 blockwise FP8-e4m3 on disk, dequantized to BF16 on the host at load time |
| parallelism | TP + SP + EP (TP=64, EP=64 on one trn2.48xlarge: 4 experts per rank). No attention/MLP/embedding/LM-head DP |

## Layout

```text
contrib/models/mimo_v2_5/
  README.md
  pyproject.toml            installs src/ as the `mimo_v2_5` package + a vllm.general_plugins entry point
  src/
    __init__.py             exports MiMoV2ForCausalLM (the factory) and MiMoV2Config
    config.py               MiMoV2Config.from_configs(hf_config, neuron_config)
    factory.py              the class registered with vLLM; validates and builds the model
    model.py                attention, MLP, MoE, backbone, ForCausalLM (forward, get_kv_spec,
                            bind_kv_cache, load_weights, compiler_args_for)
    weight_loaders_bf16.py  FP8-blockwise -> BF16 loaders, incl. the pre-sharded fused QKV
    plugin.py               register(): ModelRegistry.register_model(...)
    kernels/
      kv_cache_write.py     in-place paged KV scatter (NKI) for the 192-wide eager-attention cache
  scripts/
    run.py                  offline LLM(...) example
    serve_mimo.sh           vllm serve (TP=64/EP=64)
    bench.sh                vllm bench serve against a running server
  test/integration/
    test_model.py           logit validation against a HuggingFace reference
```

## Usage

Install `vllm-neuron` 0.29.0.1.2.0 (vLLM 0.29.0, Neuron SDK 2.33) in your Neuron
environment (see the top-level `README.md`). Then register the architecture. The
two options below match `contrib/README.md` ("How do I register a custom model
with vLLM?"); neither edits a file under `vllm_neuron/`.

Two engine settings are required by this port, not tuning:

* `max_num_batched_tokens == max_model_len` and prefix caching off. Prefill is
  single-shot. On 0.29 a smaller `max_num_batched_tokens` turns segmented prefill
  on by default, which this port does not implement; the factory rejects it at
  load time.
* The config overrides below. They null `quantization_config` (otherwise vLLM
  derives `quantization="fp8"` and the Neuron platform rejects it; the loaders
  dequantize the FP8 tensors themselves and find them by probing for
  `*_scale_inv` keys) and `vision_config` (otherwise vLLM rewrites the
  architecture to `MiMoV2OmniForCausalLM`).

Use `"quantization": "bf16"` in `neuron_config`. `"fp8"` builds the same BF16
model, but on Trn2 the runner then adds an FP8 compiler flag this graph does not
need.

### Serve with the OpenAI-compatible server

`vllm serve` has no user script to run the registration, so install this folder
as a package. Its `vllm.general_plugins` entry point registers the model in every
process vLLM starts:

```bash
pip install -e contrib/models/mimo_v2_5

export NEURON_SKIP_EFA_AFFINITY=1   # needed on a trn2.48xlarge without EFA
vllm serve XiaomiMiMo/MiMo-V2.5 \
    --served-model-name MiMo-V2.5 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 32 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --hf-overrides '{"quantization_config": null, "vision_config": null}' \
    --additional-config '{"neuron_config": {
        "quantization": "bf16",
        "ep_degree": 64,
        "num_batched_tokens_buckets": [1024],
        "num_seqs_buckets": [32],
        "on_device_sampling_config": {"all_greedy": true}},
      "encoder_neuron_config": {"num_encoder_input_buckets": [2048]}}'
```

`scripts/serve_mimo.sh` wraps the same command (`BS=1 SEQ=512 bash
scripts/serve_mimo.sh`). With `vision_config: null` the platform still sees the
attribute and builds an inert encoder config; `MiMoV2ForCausalLM.from_configs`
accepts and ignores it.

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "MiMo-V2.5", "prompt": "The capital of France is", "max_tokens": 32}'
```

`scripts/bench.sh` runs `vllm bench serve` (900 in / 90 out, random dataset)
against the running server at concurrency 1, 16 and 32.

### Offline inference with the Python API

Register the factory class before you construct `LLM` (Option 1). Use the
installed package, or copy `src/` into your application as a package named
`mimo_v2_5`:

```python
from vllm import LLM, ModelRegistry, SamplingParams

from mimo_v2_5 import ARCHITECTURE, MiMoV2ForCausalLM

# Register before you construct LLM(...).
ModelRegistry.register_model(ARCHITECTURE, MiMoV2ForCausalLM)


def text_only(config):
    # delattr, not None: then the platform builds no encoder config at all.
    for key in ("quantization_config", "vision_config"):
        if hasattr(config, key):
            delattr(config, key)
    return config


llm = LLM(
    model="XiaomiMiMo/MiMo-V2.5",
    tensor_parallel_size=64,
    enable_expert_parallel=True,
    max_model_len=512,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    enable_prefix_caching=False,
    trust_remote_code=True,
    hf_overrides=text_only,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 64,
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1],
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)

outputs = llm.generate(
    ["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=32)
)
print(outputs[0].outputs[0].text)
```

`scripts/run.py` is the same thing with arguments (`--max-model-len`,
`--max-num-seqs`, `--chat`, `--prompts`); it registers the model itself and falls
back to importing `../src` when the package is not installed:

```bash
python contrib/models/mimo_v2_5/scripts/run.py --model-checkpoint /path/to/MiMo-V2.5
```

The registration in a script reaches the engine and worker processes only
because vLLM forks them, which is the Neuron default. If you force
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, install the package so the plugin entry
point registers it in each process.

### Diagnostic switches

All default off. Each one swaps in a functionally equivalent but slower path, to
narrow a device fault down to one kernel.

| Variable | Effect |
|---|---|
| `MIMO_V2_5_KV_WRITE_FORCE_TORCH=1` | KV-cache write via `index_put_` instead of the NKI scatter in `kernels/kv_cache_write.py` |
| `MIMO_V2_5_MOE_CTE_FORCE_TORCH=1` | Prefill experts in torch instead of `NF.moe_cte` |
| `MIMO_V2_5_MOE_BLOCK_SIZE=<n>` | Floor of the prefill MoE dispatch block size (default 256; see `_prefill_block_size`) |
| `MIMO_V2_5_DECODE_OPT_LEVEL=<n>` | `-O<n>` for the plain decode graph only, via `compiler_args_for` (ignored when you pass `--optimization-level`) |

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29.0.1.2.0, vLLM 0.29.0) | 2.31 (vllm-neuron 0.21.0.1.0.0, vLLM 0.21.0) |
| ---------------------- | ----------- | ---------------- |
| Trn3                   | Not tested  | Not tested       |
| Trn2                   | Working (greedy smoke + `vllm serve` bench; logit test not run) | Working (PR #40) |

Measured on this port (trn2.48xlarge, SDK 2.33, TP=64/EP=64, BF16, XLA compile
route), `scripts/serve_mimo.sh` at BS=32 / SEQ=1024 and `scripts/bench.sh`: 900
input / 90 output tokens, random dataset, 0 failed requests:

| Concurrency | Output tok/s | Median TTFT | Median TPOT | Median ITL |
|---|---|---|---|---|
| 1 | 18.6 | 213 ms | 52.0 ms | 51.9 ms |
| 16 | 177.1 | 544 ms | 83.4 ms | 51.8 ms |
| 32 | 254.3 | 1027 ms | 111.5 ms | 51.8 ms |

Greedy smoke (`scripts/run.py`, BS=1, `max_model_len` 512) gives the same text with
the NKI KV scatter and with `MIMO_V2_5_KV_WRITE_FORCE_TORCH=1` on three of the four
demo prompts; the raw-completion counting prompt (`'1 2 3 4 5 6 7 8 9'`) degenerates
on both paths, differently (`'.10.10.11.11...'` / `' 10;10;10;...'`). Not yet
explained; the instruction-tuned checkpoint is a poor fit for raw continuations
(see `--chat`), and it was not recorded on 0.21.

Earlier results on the 0.21 / SDK 2.31 stack (trn2.48xlarge, TP=64/EP=64, BF16,
`vllm bench serve` with 900 input / 90 output tokens), from PR #40. They are
**not** numbers for this 0.29 port:

| Metric | Before the KV scatter kernel | With the KV scatter kernel |
|---|---|---|
| TPOT, BS=1, concurrency 1 | 199 ms | 128.5 ms |
| TPOT, BS=32, concurrency 1 | ~199 ms | 147.0 ms |
| Output tok/s, BS=32, concurrency 32 | 102.1 | 126.2 |
| Median TTFT, concurrency 1 / 16 / 32 | 320 / 1028 / 1676 ms | not re-measured |

Correctness on 0.21 was checked with greedy completions and a needle-in-a-haystack
sweep at `max_model_len=8192` (needle recovered at every input length from 64 to
6865 tokens and every depth). It was not checked with the logit-validation test
below, which is new in this port.

## Example Checkpoints

- <https://huggingface.co/XiaomiMiMo/MiMo-V2.5> (FP8 blockwise, ~294 GiB; loads as is)

A text-only server reads only part of the repo. To skip the vision/audio assets:

```bash
hf download XiaomiMiMo/MiMo-V2.5 --local-dir ./MiMo-V2.5 \
    --include 'config.json' 'configuration_mimo_v2.py' 'modeling_mimo_v2.py' \
              'generation_config.json' 'tokenizer*' 'vocab.json' 'merges.txt' \
              'model*.safetensors' 'model.safetensors.index.json'
```

A BF16 re-export with the same tensor names also loads: the loaders take their
BF16 branch when no `*_scale_inv` tensors are present.

## Testing

The integration test serves the model through vLLM on Neuron at TP=64/EP=64 and
compares the per-token logits against a HuggingFace reference. It needs a
trn2.48xlarge. It registers the model itself, so the package does not need to be
installed.

```bash
MIMO_V2_5_CHECKPOINT=/path/to/MiMo-V2.5 \
pytest contrib/models/mimo_v2_5/test/integration/test_model.py \
    -v -s --timeout=7200
```

The first run computes the HF reference logits on the host (an FP32 and a BF16
pass over a ~310B-parameter model, so ~1.25 TB of host RAM and a long wait) and
caches them. If your transformers version refuses to dequantize the FP8
checkpoint on CPU, set `MIMO_V2_5_REFERENCE_CHECKPOINT` to a BF16 re-export. The
tolerance map is the template's starting point and has not been tuned for this
model.

No CPU-mode unit or module tests are included yet.

## Port notes (0.21 -> 0.29)

What changed from PR #40, and why:

* **Self-contained.** The KV scatter kernel moved from
  `vllm_neuron/functional/attention/kv_cache_write.py` into `src/kernels/`. The
  0.21 edits to `functional/__init__.py`, `moe_blockwise.py`, `moe_cte.py` and
  `model/registry.py` are dropped. Registration is the script or the plugin.
* **No `platform.py` change.** On 0.21 the platform sized vision buckets for any
  config with a `vision_config` attribute and aborted text-only servers with
  `max_model_len < 2048`. On 0.29 the auto-sized encoder bucket is capped at
  `min(next_pow2(max_model_len), block_size)`, so that abort no longer happens.
  What remains is an inert `encoder_neuron_config`, which the factory ignores.
  The runner kwarg was renamed from `vision_neuron_config` to
  `encoder_neuron_config`; the factory takes both.
* **Imports.** `can_run_kernel` is now in `vllm_neuron.utils.neuron_utils`,
  `wrap_nki` in `libtorch_neuronx_lite.nki.nki_hop`, `get_platform_target` in
  `libtorch_neuronx_lite.compile.platform`.
* **SPMD rank tensors.** The EP partition index comes from the EP group's
  stamped `rank_in_group_tensor`, as in `gpt_oss`. The 0.21 code derived it from
  the TP-local rank input. The SP slice of prompt embeds uses `shard_select`
  instead of a Python-int `self.rank` slice.
* **Block-table sentinel.** The eager KV gather clamps the null-block sentinel
  (`-1`) to block 0 before indexing.
* **Torch KV-write fallback.** It now redirects padded slots to the null block
  with `paged_slot_indices`. Before, `-1` wrapped to the last block.
* **`compiler_args_for`.** The 0.29 per-graph compiler-argument hook is
  implemented. It returns the runner's arguments unchanged unless
  `MIMO_V2_5_DECODE_OPT_LEVEL` is set.
* **Fail fast.** A segmented-prefill config is rejected in the factory, at load.
* **Diagnostic env vars** were renamed to `MIMO_V2_5_*`. The 0.21
  `VLLM_NEURON_MOE_MAPPING_FORCE_TORCH` switch is gone: it was a one-line edit
  inside the shared `build_blockwise_mapping`, and there is no way to reach that
  branch from outside.
