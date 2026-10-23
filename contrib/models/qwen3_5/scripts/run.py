# SPDX-License-Identifier: Apache-2.0
"""Text-only offline inference for Qwen3.5 (dense and sparse) on Neuron.

Qwen3.5 is a hybrid stack: most layers are gated DeltaNet (a linear recurrence
with a fixed-size state) and the rest are full attention. It needs the
recurrent-state support in vllm-neuron (``KVSpec.state_layers``).

The script registers the architectures itself (Option 1 in
``contrib/README.md``). It imports the installed ``qwen3_5`` package when
``pip install -e contrib/models/qwen3_5`` has been run, and otherwise loads this
folder's ``src/`` under that name.

Usage (2B, on a trn2.3xlarge: 4 logical NeuronCores, so TP=4 is the ceiling):

    python contrib/models/qwen3_5/scripts/run.py \
        --model <path-to-checkpoint>/Qwen3.5-2B

    # 27B on a 4-core instance needs the memory cap. See README.md.
    python contrib/models/qwen3_5/scripts/run.py \
        --model <path-to-checkpoint>/Qwen3.5-27B \
        --gpu-memory-utilization 0.65

    # 35B-A3B (sparse).
    python contrib/models/qwen3_5/scripts/run.py \
        --model <path-to-checkpoint>/Qwen3.5-35B-A3B \
        --gpu-memory-utilization 0.72

    # 397B-A17B (sparse), on all 64 logical cores of a trn2.48xlarge. Every one
    # of these is required; README.md, "Expert parallelism", says why.
    python contrib/models/qwen3_5/scripts/run.py \
        --model <path-to-checkpoint>/Qwen3.5-397B-A17B \
        --tensor-parallel-size 64 --expert-parallel --ep-degree 8 \
        --gpu-memory-utilization 0.7
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Compiling a cold graph can take longer than vLLM's default step timeout.
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
]


def import_qwen3_5():
    """Return the ``qwen3_5`` package: the installed one, or this folder's ``src/``."""
    try:
        import qwen3_5
    except ImportError:
        src = Path(__file__).resolve().parents[1] / "src"
        spec = importlib.util.spec_from_file_location(
            "qwen3_5", src / "__init__.py", submodule_search_locations=[str(src)]
        )
        qwen3_5 = importlib.util.module_from_spec(spec)
        sys.modules["qwen3_5"] = qwen3_5
        spec.loader.exec_module(qwen3_5)
    return qwen3_5


def register_qwen3_5() -> None:
    """Register both Qwen3.5 architectures (contrib/README.md, Option 1).

    Call this before ``LLM(...)``. The engine and worker processes inherit the
    registration when vLLM forks them. If they are spawned instead, install the
    package (``pip install -e contrib/models/qwen3_5``), whose entry point
    registers the architectures in every process.
    """
    from vllm import ModelRegistry

    qwen3_5 = import_qwen3_5()
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration", qwen3_5.Qwen3_5ForConditionalGeneration
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        qwen3_5.Qwen3_5MoeForConditionalGeneration,
    )


def relax_head_divisibility() -> None:
    """Let the world be wider than the model's attention-head count.

    ``ModelConfig.verify_with_parallel_config`` rejects
    ``total_num_attention_heads % tensor_parallel_size != 0``, so a 32-head model
    cannot get 64 ranks even though the model does not need one head per rank.
    vLLM already replicates in the analogous case: ``get_num_kv_heads`` is
    ``max(1, total // tp)``. Only the query head count is checked for exact
    divisibility.

    Qwen3.5-397B-A17B needs 64 ranks to fit its weights (738 GiB of text-only
    parameters against 23.5 GiB per logical core) and has 32 query heads. The
    modules replicate the affected heads across ``tp // heads`` consecutive
    ranks (see ``Qwen3_5Attention.__init__`` and
    ``Qwen3_5GatedDeltaNet.__init__``).

    Relaxed only in the safe direction: the world must be a whole multiple of
    the head count, so every rank's shapes equal those of a legal narrower world.
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
    resolves to the world size: pure EP, ``tp_degree = 1``.
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
    # Defaults to max_model_len: the model does not support chunked prefill
    # (each prefill starts from a zero state and attends only to its own chunk),
    # and the factory refuses a smaller value.
    parser.add_argument("--prefill-bucket", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    # None keeps vLLM's default. Lower it for large models: the KV budget is
    # (24 GB per logical core * gmu - weights), and the planner fills all of it,
    # which for a hybrid model means recurrent-state blocks far beyond
    # max_num_seqs, until neuronx-cc rejects the graph.
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="give each rank a disjoint subset of the experts instead of a "
        "column of every expert. Only 397B-A17B needs this: its "
        "moe_intermediate_size of 1024 caps pure tensor parallelism at 8 ranks.",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=None,
        help="expert-parallel degree; requires --expert-parallel. Prefer the "
        "smallest degree that keeps moe_intermediate_size/tp_degree a multiple "
        "of 128.",
    )
    args = parser.parse_args()
    if args.prefill_bucket is None:
        args.prefill_bucket = args.max_model_len

    register_qwen3_5()

    extra = {}
    if args.gpu_memory_utilization is not None:
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.expert_parallel:
        extra["enable_expert_parallel"] = True
    if args.tensor_parallel_size > 1:
        relax_head_divisibility()

    from vllm import LLM, SamplingParams

    # Do not pass optimization_level: an explicit level disables the model's
    # compiler_args_for, which the dense decode graph needs (see model.py).
    llm = LLM(
        model=args.model,
        **extra,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_bucket,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        # Prefix caching over recurrent state is not supported.
        enable_prefix_caching=False,
        # Text-only: no vision buckets (this port builds no vision tower).
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.prefill_bucket],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
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
