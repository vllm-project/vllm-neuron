# The model: hf:deepseek-ai/DeepSeek-V4.1-Flash

Revision `dba1be0a40aa45a94ad051997016db3960a90277`, 40 layers, stored `bfloat16`.

**271 modules must be placed**, totalling 474.5 GiB of parameters. vision is excluded — it is not on the `tokens -> logits` path — and placing it is an error.

For scale: the instance holds 1.5 TiB of device memory across 64 banks of 24.0 GiB.

## By kind

| kind | modules | parameters | share |
|---|---|---|---|
| mlp | 43 | 277.2 GiB | 58.4% |
| other | 182 | 189.6 GiB | 40.0% |
| attention | 43 | 5.2 GiB | 1.1% |
| lm_head | 1 | 1.2 GiB | 0.3% |
| embed | 1 | 1.2 GiB | 0.3% |
| norm | 1 | 10.0 KiB | 0.0% |

## The largest modules

These are where placement decisions are won and lost.

| module | kind | parameters | activations | splittable along |
|---|---|---|---|---|
| `layers.14.engram` | other | 94.6 GiB | 480.0 MiB | batch, head, hidden, layer, ngram, seq |
| `layers.1.engram` | other | 94.6 GiB | 480.0 MiB | batch, head, hidden, layer, ngram, seq |
| `layers.18.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.33.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.38.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.37.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.28.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.10.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.7.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.6.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.4.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |
| `layers.13.ffn` | mlp | 6.8 GiB | 480.0 MiB | batch, expert, hidden, layer, seq |

## Modules that do not fit one bank

2 module(s) exceed a single 24.0 GiB bank, so each
must be split, tiered, or placed off-HBM. There is no placement that avoids the
question:

- `layers.14.engram` — 94.6 GiB, 3.9x a bank (a 64-way split still leaves 1.5 GiB per unit)
- `layers.1.engram` — 94.6 GiB, 3.9x a bank (a 64-way split still leaves 1.5 GiB per unit)

## Partition dimensions

The closed vocabulary the simulator understands. A split along a dimension the
module does not have is rejected — not as a bad idea but as a meaningless one, since
it would produce shards of size zero.

| dimension | meaning |
|---|---|
| `head` | attention heads |
| `hidden` | the model dim, or an MLP's intermediate dim |
| `expert` | MoE routed experts |
| `seq` | sequence positions (context/sequence parallel) |
| `batch` | batch members (data parallel) — free, no collective |
| `layer` | whole layers, for a module spanning several (pipeline parallel) |
| `vocab` | vocabulary, for embed and lm_head |
| `ngram` | Engram n-gram table rows |

## Data dependencies

Taken from the partition graph: a module consuming a tensor waits for whoever
produces it. You do not declare dependencies — they are the model's — but the
schedule you choose has to be consistent with them, and the `stage` field is how you
say what runs when.

The graph has 272 modules and 273 named tensors, entry `tokens` and output `logits`.

