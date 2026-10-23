# Fixed benchmark data

Real-text prompts, fixed so that every benchmark run sends exactly the same
requests. The decode MoE kernels (`moe_tkg_hits*`) compute only the routed
experts, so their speed depends on the routing, i.e. on the text: vLLM's
`random` dataset (runs of consecutive token ids, meaningless text) does not
time them representatively.

| Path | Content | Used by |
|---|---|---|
| `prompts/in{L}_c{C}.jsonl.gz` | `max(8, 4 x C)` prompts of exactly `L` tokens (Hy3 tokenizer, no chat template); L = 1024 / 4096 / 8192 / 16000 / 32768 / 65536 / 85000, C = 1 / 4 / 8 / 15 | `scripts/bench_fixed.sh "<L...>" "<C...>"` |
| `needle.jsonl.gz` | 12 passkey prompts: 16000 / 32768 / 65536 / 85000 tokens (chat-templated, +-16) x depth 0.1 / 0.5 / 0.9 | `scripts/needle.py run bench_data/needle.jsonl.gz` |

`prompts/` follows sglang's `random` dataset: the first user turn of ShareGPT
V3 conversations (`anon8231489123/ShareGPT_Vicuna_unfiltered`,
`ShareGPT_V3_unfiltered_cleaned_split.json`), truncated to `L` tokens or
repeated and truncated when shorter. Each cell has its own seed
(`L * 1000 + C`), so cells never share prefix-cache entries. Both files are
deterministic; regenerating them with the same tokenizer and ShareGPT file
gives identical bytes:

```bash
python scripts/gen_bench_prompts.py <Hy3-FP8 dir> ShareGPT_V3_unfiltered_cleaned_split.json \
    bench_data/prompts "1024 4096 8192 16000 32768 65536 85000" "1 4 8 15"
python scripts/needle.py gen <Hy3-FP8 dir> bench_data/needle.jsonl.gz 16000 32768 65536 85000
```
