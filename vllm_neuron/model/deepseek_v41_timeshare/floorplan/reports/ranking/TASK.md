# Rank 3 deployment schemes for DeepSeek V4.1 Flash on a 16-device Trainium2 instance

`schemes/scheme-*.yaml` are 3 candidate deployments of the same model on the same
hardware. All 3 are **feasible**: an automated gate has already confirmed that each one
places every module on the inference path, splits only along dimensions that exist, keeps every
memory tier within capacity at contexts of both 128 and 8192 tokens, and satisfies every data
dependency. You do not need to re-check any of that.

Your job is to decide **which is the best deployment, and why**, from the architecture.

## What you have

| | |
|---|---|
| `schemes/scheme-*.yaml` | the 3 candidates, in arbitrary order |
| `PLATFORM.md` | the hardware: hierarchy, memory tiers, interconnect, and the numbered text constraints |
| `MODEL.md` | the modules, their sizes, their dependencies, and the legal split dimensions |
| `systems/*.yaml` | the platform datasheet the briefs were generated from |
| the model's own source | at the artifact path given to you, `modules/*/source.py` and `vendor/model.py` |

## What you deliberately do not have

No performance predictions. No simulator, no timings, no profile, no measurement of any kind —
not withheld by request but absent from this directory.

This is on purpose. These schemes were found by a search driven by a simulator that is
**uncalibrated**: it was built without access to any measured latency for this model, so its
absolute numbers are unverified. Your independent architectural judgement is worth more than a
review of its output, and it is only independent if you cannot see it.

So do not speculate about what a simulator would say, and do not try to estimate the latencies
yourself and rank by that. Rank by reasoning about the hardware: what each scheme asks the
machine to do, which limit it runs into first, and which one asks for something the machine is
good at.

The scheme names carry no information. `scheme-a` is not the favourite.

## What to write

One file, `REPORT.md`, in readable but technical prose. It has to answer, clearly:

1. **What is the best scheme?** Name it and commit to it. Then rank the other two, with the
   ranking criteria you used and where the schemes are close enough that the order is a
   judgement call.

2. **How does it distribute and deploy the workload?** Walk through it: which modules go where,
   how the model is divided across the devices and their logical NeuronCores, what the schedule
   is, and where the weights live. Someone who has not read the YAML should be able to follow
   the deployment from your description.

3. **Why is it the best?** This is the core of the report. Argue from the hardware — refer to
   specific numbers and constraints in `PLATFORM.md`: bank capacity, link bandwidths, hop
   counts, the engine exclusions, the tier latencies. "Better load balance" is not an argument;
   "the 24 GiB bank limit forces Engram off-HBM, and this scheme is the only one that puts the
   resulting tier crossing off the critical path of layers 1 and 14" is.

   Say what each scheme's *first limiting factor* is, and why the winner's is the least bad.
   Distinguish prefill from decode: they stress completely different things, and a scheme can
   be right for one and wrong for the other.

4. **Where a module is decomposed, how do you get the pieces?** For every module the winning
   scheme splits, give the recipe concretely:
   - which dimension, and what that means for this module's actual computation
   - how the tensors divide — which axis of which weight, and what each shard holds
   - what has to be communicated to put the result back together, and when
   - which level of the hierarchy each piece lands on: a device, a logical NeuronCore, or the
     two physical NeuronCore-v3 inside one

   Someone implementing this should not have to re-derive the partition from the dimension name.

5. **What would you check first?** The parts of your reasoning that rest on the datasheet's
   less certain numbers, or on an assumption the scheme makes that you cannot verify —
   `hit_rate` on a tiered module is the obvious one. Name them.

Be direct and technical. Do not hedge every claim, and do not pad. If two schemes are
substantially the same deployment with a cosmetic difference, say so rather than manufacturing
a distinction.

## Required: state your rankings on their own lines

Somewhere in `REPORT.md`, put the overall ranking in exactly this form, best first:

    RANKING: scheme-b > scheme-a > scheme-c

This is parsed to decide which plan is published as rank 1, 2 and 3. Without it the ordering
has to be guessed from the order your prose happens to mention the schemes in, which for a
report that introduces all three before choosing is simply wrong.

Then give **one ranking per configuration**, in the same form with the configuration in
brackets:

    RANKING[prefill_128_b1]: scheme-a > scheme-c > scheme-b
    RANKING[prefill_128_b4]: scheme-a > scheme-c > scheme-b

...and so on for every configuration listed in section 6 below. Rank each one on its own
merits: these are genuinely different problems, and a scheme that wins overall need not win
everywhere.

## 6. Which scheme wins each configuration, and why

The deployment is measured at every combination of three axes:

    phase             prefill (ingest a prompt) or decode (produce one token)
    context length    128 or 8192 tokens
    batch size        1, 4, 8 or 32 samples

For each, say which scheme you expect to win and give the reason in a sentence or two. You are
not being asked to guess latencies — you are being asked which scheme's structure suits that
regime, and the axes interact in ways that should drive your answer:

- **Batch reverses the pipeline-depth argument.** At batch 1 a decode step has one token in
  flight, so every stage boundary is a bubble and depth is pure cost. At batch 32 there are 32
  tokens to fill it and the same depth is nearly free. A deeply pipelined scheme should lose the
  batch-1 decode points and may win the batch-32 ones.
- **Batch multiplies the KV cache.** 8192 tokens at batch 32 is 32x the KV of batch 1. A scheme
  that spends its HBM on resident weights has less room for it, and one that tiers a table has
  more.
- **Batch changes expert routing.** At batch 1 a decode step touches a few of the 384 routed
  experts; at batch 32 it touches many more, which shifts the balance between expert
  parallelism and replication.
- **Context length decides whether attention or weight movement dominates**, and that in turn
  decides whether a sequence split or a head split is the better cut.

Where two schemes are effectively identical for a configuration, say so and pick either —
manufacturing a distinction is worse than admitting a tie.

The configurations, by name:

    prefill_128_b1
    prefill_128_b4
    prefill_128_b8
    prefill_128_b32
    prefill_8192_b1
    prefill_8192_b4
    prefill_8192_b8
    prefill_8192_b32
    decode_128_b1
    decode_128_b4
    decode_128_b8
    decode_128_b32
    decode_8192_b1
    decode_8192_b4
    decode_8192_b8
    decode_8192_b32
