# Fixed benchmark data

Real-text prompts, fixed so that every benchmark run sends exactly the same
requests. The decode MoE kernels (`moe_tkg_hits*`) compute only the routed
experts, so their speed depends on the routing, i.e. on the text: vLLM's
`random` dataset (runs of consecutive token ids, meaningless text) does not
time them representatively.

| Path | Content | Used by |
|---|---|---|
| `prompts/in{L}_c{C}.jsonl.gz` | `max(8, 4 x C)` prompts of exactly `L` tokens (Hy3 tokenizer, no chat template); L = 1024 / 4096 / 8192 / 16000 / 32768 / 65536 / 85000, C = 1 / 4 / 8 / 15 | `scripts/bench_fixed.sh "<L...>" "<C...>"` |
| `prefix/in85000_p78336_c{C}.jsonl.gz` | shared-prefix prompts: 85000 tokens = one 78336-token prefix (92.2%, a multiple of the 256-token DCP=8 cache block) + a distinct suffix; 1 warm-up + 10 x C requests, C = 1 / 4 / 8; line 0 holds the prefix, each request stores only its suffix | `scripts/bench_prefix_fixed.sh "<C...>"` |
| `needle.jsonl.gz` | 12 passkey prompts: 16000 / 32768 / 65536 / 85000 tokens (chat-templated, +-16) x depth 0.1 / 0.5 / 0.9 | `scripts/needle.py run bench_data/needle.jsonl.gz` |

`prompts/` follows sglang's `random` dataset: the first user turn of ShareGPT
V3 conversations (`anon8231489123/ShareGPT_Vicuna_unfiltered`,
`ShareGPT_V3_unfiltered_cleaned_split.json`), truncated to `L` tokens or
repeated and truncated when shorter. Each cell has its own seed
(`L * 1000 + C`), so cells never share prefix-cache entries. All files are
deterministic; regenerating them with the same tokenizer and ShareGPT file
gives identical bytes. `prefix/` uses ShareGPT conversations joined into one
document as the prefix and other conversations' first user turns as suffixes;
every prompt is checked to be exactly 85000 tokens with the first 78336 equal
to the prefix:

```bash
python scripts/gen_bench_prompts.py <Hy3-FP8 dir> ShareGPT_V3_unfiltered_cleaned_split.json \
    bench_data/prompts "1024 4096 8192 16000 32768 65536 85000" "1 4 8 15"
python scripts/gen_prefix_prompts.py <Hy3-FP8 dir> ShareGPT_V3_unfiltered_cleaned_split.json \
    bench_data/prefix 85000 78336 "1 4 8"
python scripts/needle.py gen <Hy3-FP8 dir> bench_data/needle.jsonl.gz 16000 32768 65536 85000
```
