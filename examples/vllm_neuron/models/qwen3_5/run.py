# SPDX-License-Identifier: Apache-2.0
"""Text-only offline inference for the dense Qwen3.5 checkpoints on Neuron.

Qwen3.5 is a hybrid stack: 18 of 2B's 24 layers (48 of 27B's 64) are gated
DeltaNet -- a linear recurrence with fixed-size state -- and the rest are full
attention. That makes it the first model in this plugin to need two KV cache
groups, one paged and one recurrent.

Usage (2B, on a trn2.3xlarge: 4 logical NeuronCores, so TP=4 is the ceiling):

    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.5-2B

    # 27B needs both of these on a 4-core instance. See README.md.
    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.5-27B \
        --gpu-memory-utilization 0.65 --optlevel 3

    # 35B-A3B (sparse). Needs the memory cap but *not* the higher optlevel, and
    # must keep the mac threshold the dense models suppress -- see
    # mac_threshold_override below.
    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.5-35B-A3B \
        --gpu-memory-utilization 0.72

    # 397B-A17B (sparse), on all 64 logical cores of a trn2.48xlarge. Every one
    # of these is required; README.md, *Expert parallelism*, says why.
    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.5-397B-A17B \
        --tensor-parallel-size 64 --expert-parallel --ep-degree 8 \
        --gpu-memory-utilization 0.7

This demo is text-only. The vision-language path (``model/qwen3_5/vl.py``,
which reuses the Qwen3-VL encoder) is selected by passing a
``vision_neuron_config``; see ``check_generation_vs_hf.py --vl``.
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
]


def mac_threshold_override(model: str) -> dict:
    """``{"hlo2tensorizer_options": ""}`` for a dense checkpoint, ``{}`` for MoE.

    That override suppresses the runner's ``--modular-flow-mac-threshold=10``.
    The dense graphs need it suppressed -- with the flag, neuronx-cc fails
    codegen on their decode graph (NCC_IBTN006 on a pftranspose copy) -- and the
    sparse ones need it kept, because the flag exists for exactly the NKI kernels
    the MoE block calls. Keyed off the checkpoint rather than a command-line flag
    because the reason is structural and a flag would be forgotten.
    """
    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(model).text_config
    if getattr(text_config, "num_experts", None):
        return {}
    return {"hlo2tensorizer_options": ""}


def relax_head_divisibility() -> None:
    """Let the world be wider than the model's attention-head count.

    ``ModelConfig.verify_with_parallel_config`` rejects
    ``total_num_attention_heads % tensor_parallel_size != 0`` outright, so a
    32-head model cannot be given 64 ranks even though nothing about the model
    requires one head per rank. vLLM already replicates in the analogous case
    one line away -- ``get_num_kv_heads`` is ``max(1, total // tp)``, explicitly
    "so each GPU has at least one KV head" -- it is only the *query* head count
    that is checked for exact divisibility.

    Qwen3.5-397B-A17B needs 64 ranks to fit its weights (738 GiB of text-only
    parameters against a hard 23.5 GiB per logical core) and has 32 query heads,
    so this check, not any hardware or model limit, is what stands between the
    checkpoint and the machine. The modules replicate the affected heads across
    ``tp // heads`` consecutive ranks -- see the replication notes in
    ``Qwen3_5Attention.__init__`` and ``Qwen3_5GatedDeltaNet.__init__``; only
    dimensions a row-parallel projection reduces over have to shard cleanly.

    Relaxed only in the direction that is safe: the world must be a whole
    multiple of the head count, which is what makes every rank's shape identical
    to that of a legal narrower world. ``get_num_attention_heads`` would return
    0 here, so it is given the same ``max(1, ...)`` as its kv-head sibling; on
    Neuron nothing reads it (every consumer in this vLLM is a GPU/ROCm/CPU
    attention backend), but returning 0 from a head count is a trap either way.
    """
    from vllm.config.model import ModelConfig

    original_verify = ModelConfig.verify_with_parallel_config

    def verify(self, parallel_config):
        arch = self.model_arch_config
        heads, tp = arch.total_num_attention_heads, parallel_config.tensor_parallel_size
        if heads and tp > heads and tp % heads == 0:
            # Present a count the check accepts, for the duration of the check.
            try:
                arch.total_num_attention_heads = tp
                return original_verify(self, parallel_config)
            finally:
                arch.total_num_attention_heads = heads
        return original_verify(self, parallel_config)

    def get_num_attention_heads(self, parallel_config):
        heads = self.model_arch_config.total_num_attention_heads
        return max(1, heads // parallel_config.tensor_parallel_size)

    ModelConfig.verify_with_parallel_config = verify
    ModelConfig.get_num_attention_heads = get_num_attention_heads


def ep_neuron_config(args) -> dict:
    """``ep_degree`` for the neuron_config, validated.

    vLLM's ``enable_expert_parallel`` is a bool and its degree is the world size,
    so the plugin carries an explicit degree on NeuronConfig instead. Unset, it
    resolves to the world size -- pure EP, ``tp_degree = 1``.
    """
    if args.ep_degree is None:
        return {}
    if not args.expert_parallel:
        raise SystemExit("--ep-degree requires --expert-parallel")
    return {"ep_degree": args.ep_degree}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--prefill-bucket", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    # None keeps vLLM's default. Lower it for large models: the KV budget is
    # (24 GB per logical core * gmu - weights), and the planner fills whatever
    # it is given, which for a hybrid model means recurrent-state blocks far
    # beyond max_num_seqs -- until neuronx-cc rejects the graph.
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    # None keeps the plugin's default (O1). 27B's decode graph needs a higher
    # level: it fails ISA validation at O1 (NCC_IINAR001 on a pftranspose
    # Copy) and compiles at O2. The platform lowers vLLM's default O2 to O1
    # and cannot see an explicit offline O2, so pass 3 -- its own docstring
    # points at O3 as the way to force a higher level.
    parser.add_argument("--optlevel", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="give each rank a disjoint subset of the experts, instead of a "
        "column of every expert. 35B-A3B does not want this: per-rank prefill "
        "work and the number of graphs to compile both grow with the degree "
        "while the footprint does not. It is here because it is the only layout "
        "that can serve 397B-A17B, whose moe_intermediate_size of 1024 caps pure "
        "tensor parallelism at 8 ranks -- the fused decode kernel needs a "
        "multiple of 128 per rank -- far short of the ranks its weights need.",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=None,
        help="expert-parallel degree; requires --expert-parallel. Left unset it "
        "resolves to the world size, i.e. tp_degree=1, which is the most "
        "expensive legal choice: prefer the smallest degree that keeps "
        "moe_intermediate_size/tp_degree a multiple of 128.",
    )
    args = parser.parse_args()

    extra = {}
    if args.gpu_memory_utilization is not None:
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.optlevel is not None:
        from vllm.config.vllm import OptimizationLevel

        extra["optimization_level"] = OptimizationLevel(args.optlevel)
    if args.expert_parallel:
        extra["enable_expert_parallel"] = True
    if args.tensor_parallel_size > 1:
        relax_head_divisibility()

    llm = LLM(
        model=args.model,
        **extra,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_bucket,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        # Prefix caching hard-requires segmented prefill on this plugin, and a
        # DeltaNet layer has no notion of a reusable prefix anyway: its prefill
        # starts from a zero state.
        enable_prefix_caching=False,
        # Text-only demo: refuse image and video items at the frontend rather
        # than silently answering a multimodal request from its text alone.
        # Omitting ``vision_neuron_config`` below is what selects the text-only
        # implementation in ``factory.py``.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.prefill_bucket],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                **mac_threshold_override(args.model),
                **ep_neuron_config(args),
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outputs = llm.generate(PROMPTS, sampling_params)
    for prompt, output in zip(PROMPTS, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
