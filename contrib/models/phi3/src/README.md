# Phi3 (Phi-4-mini-instruct) on vLLM Neuron

This directory holds the native-route (`neuron_native`) implementation of
`Phi3ForCausalLM`. The target checkpoint is `microsoft/Phi-4-mini-instruct`.
The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 32 |
| Hidden size | 3072 |
| Intermediate size | 8192 (SwiGLU) |
| Attention | GQA, 24 Q heads, 8 KV heads, head_dim 128 |
| RoPE | Partial (0.75, 96 dimensions), LongRoPE, original context 4096 |
| Vocabulary | 200064, tied embeddings |
| Dtype | BF16 |

The checkpoint stores fused `qkv_proj` and `gate_up_proj` weights. The weight
loaders split them for each TP rank. The gate projection is the first half of
`gate_up_proj`.

## Parallelism

Use TP=1, 2, 4 or 8. TP must divide both the 24 Q heads and the 8 KV heads,
because `Phi3Attention` takes `num_attention_heads // world_size` and
`num_key_value_heads // world_size` as exact splits:

| TP | Q heads per rank | KV heads per rank |
|----|------------------|-------------------|
| 1  | 24               | 8                 |
| 2  | 12               | 4                 |
| 4  | 6                | 2                 |
| 8  | 3                | 1                 |

No other size works: TP 3, 6, 12 and 24 divide the 24 Q heads but not the 8 KV
heads, and TP 16 and 32 divide neither exactly, so the integer divisions above
drop heads instead of raising. The tests in this folder run TP 4 (unit, module)
and TP 8 (integration).

> **Correction, made when this model was contributed.** This card first said
> "the decode kernel needs an even number of Q heads on each rank, so TP=8 is
> not valid". That claim is wrong. The constraint in the plugin's decode
> wrapper is per-rank `q_heads % kv_heads == 0` (the GQA head-split check in
> `vllm_neuron/functional/attention/attention_decode.py`, which runs only when
> a rank holds more than one KV head); the same divisibility is asserted inside
> the NKI `attention_block_tkg` kernel, whose only parity requirement is on
> `d_head`, not on the Q heads. At TP 4 a rank holds 2 KV heads, so that check
> does read as "an even number of Q heads per rank" — which is where the
> original sentence came from. At TP 8 a rank holds 1 KV head, 3 Q heads
> satisfy it, and nothing in `config.py`, `factory.py` or `model.py` rejects
> TP 8. The model folder's `README.md` keeps both the original claim and the
> evidence, under Exceptions.

- Prefill uses sequence parallelism: each rank holds 1/TP of the tokens.
  Attention and MLP all-gather the tokens and reduce-scatter the result.
- Decode uses the fused attention kernel and the MLP kernel, then an
  all-reduce.
- The embedding and the LM head are sharded on the vocabulary dimension.

## Partial LongRoPE

The NKI kernels rotate the pairs (j, j + 64) of the full 128-dimension head.
HF Phi3 rotates the pairs (i, i + 48) of the first 96 dimensions. The weight
loader permutes the Q and K head dimensions so that the HF pairs go to the
kernel pairs. The 32 dimensions that do not rotate get cos 1 and sin 0. The
permutation does not change Q.K, and V is not permuted.

The long factor applies when a position in the step is 4096 or more. This is
the same selection as HF `longrope_frequency_update`.

## Limits

- The model has no sliding window attention. It accepts a `sliding_window`
  that is not smaller than `max_position_embeddings`. Phi-4-mini sets 262144.
- `inputs_embeds` is not supported. Give `input_ids`.
- The model supports only BF16 weights.

## Example

See the "Usage" section of the model folder's `README.md` for a `vllm serve`
command and an offline `LLM(...)` script, including the
`pip install -e contrib/models/phi3` step that registers the architecture.

## Tests

The tests that ship with this model are in the folder above this one:

- `test/unit/test_phi3_cpu.py`: CPU-mode registration contract, config parse,
  fused weight transforms, full weight coverage, partial LongRoPE, attention
  and full-model logits at TP=4.
- `test/module/test_phi3_device_modules.py`: device module tests at TP=4 —
  flash and segmented prefill attention, decode, the MLP, and a full-model
  prefill that uses the serving pad layout. The full-model case reports the
  first stage that is not finite and compares the logits with HF.
- `test/integration/test_model.py`: serves the model through vLLM at TP=8 and
  compares the raw logits with a HuggingFace golden reference.

Each test file loads this `src/` folder by file location through
`test/_local_src.py`, under the module name `contrib_phi3_src`, so the tests
need neither the repository root nor a `sys.path` entry.

The port also had 2-node serving tests (served logits, and 1024-input /
512-output chat requests on both model runners). They depend on shared test
helpers outside the model folder and are not part of this contribution.
