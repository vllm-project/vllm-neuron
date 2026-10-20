#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KV Cache Analysis: two-way or three-way comparison of expected vs actual (vLLM).

Generates expected (and optionally baseline) goldens on CPU, then runs vLLM on
Neuron and compares KV caches token-by-token, layer-by-layer, head-by-head.

Note: async_scheduling must be disabled because logprobs are not returned in
async scheduling mode with on-device sampling.

See also: docs/design/accuracy/kv_cache_analysis_design.md

Naming matches ``logit_validation``:
- expected: reference to compare against (required)
- actual: target under test (vLLM on Neuron)
- baseline: ground truth for three-way BC analysis (optional)

Two-way mode (--two-way): compares expected vs actual using L-inf, L2, cosine.
Three-way mode (default): also generates FP32 baseline and computes BC to check
whether actual's error pattern matches expected's (both relative to baseline).

Outputs per prompt: HTML heatmap + JSON metrics.
Outputs across prompts (three-way only): aggregated BC HTML.

Usage — Two-way (BF16 expected vs vLLM actual):
    NEURON_VISIBLE_DEVICES=0-31 python run_kv_cache_analysis.py \\
        --model meta-llama/Llama-3.3-70B-Instruct --two-way \\
        --tp-size 32 --max-model-len 256 \\
        --output-dir exp_two_way --num-tokens 16 --num-prompts 3

Usage — Three-way (FP32 baseline + BF16 expected vs vLLM actual):
    NEURON_VISIBLE_DEVICES=0-31 python run_kv_cache_analysis.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --tp-size 32 --max-model-len 256 \\
        --output-dir exp_three_way --num-tokens 16 --num-prompts 3

Usage — Load pre-saved goldens (skip CPU generation):
    python run_kv_cache_analysis.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --goldens-path exp_three_way/goldens.pkl \\
        --tp-size 32 --output-dir exp_rerun
"""

import argparse
import gc
import json
import os
import pickle
import torch

# NOTE: import vllm BEFORE transformers to avoid Neuron class double-registration
from vllm import LLM, SamplingParams  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from vllm_neuron.accuracy import logit_validation
from vllm_neuron.accuracy.kv_cache_analysis import (
    compare_kv_caches,
    extract_hf_kv_caches_teacher_forced,
    extract_vllm_kv_caches,
    extract_vllm_kv_cache_config,
    extract_vllm_block_tables,
    reconstruct_contiguous_kv,
    aggregate_kv_bc_across_prompts,
    cleanup_kv_snapshot,
    enable_kv_snapshot,
)
from vllm_neuron.accuracy.kv_cache_visualize import (
    export_html_report as export_kv_html_report,
    export_json as export_kv_json,
    export_aggregated_bc_html,
)

DEFAULT_PROMPTS = [
    "In the field of machine learning,",
    "Once upon a time in a land far away,",
    "The capital of France is",
    "def fibonacci(n):",
    "Explain the theory of relativity in simple terms:",
]


# =============================================================================
# Golden generation
# =============================================================================


def generate_goldens(
    model_id,
    prompts,
    num_tokens,
    baseline_dtype=torch.float32,
    expected_dtype=torch.bfloat16,
):
    """Generate baseline + expected goldens on CPU.

    Two-way (baseline_dtype=None): only generates expected model goldens.
    Three-way: generates both baseline and expected.

    Args:
        model_id: HuggingFace model ID.
        prompts: List of prompt strings.
        num_tokens: Number of tokens to generate.
        baseline_dtype: Dtype for baseline model. None for two-way mode.
        expected_dtype: Dtype for expected model (reference).

    Returns:
        Dict[int, dict] with per-prompt goldens.

    Example:
        >>> goldens = generate_goldens("meta-llama/Llama-3.3-70B-Instruct",
        ...     ["Hello world"], 16)
        >>> goldens = generate_goldens("meta-llama/Llama-3.3-70B-Instruct",
        ...     ["Hello world"], 16, baseline_dtype=None)  # two-way
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    goldens = {}
    three_way = baseline_dtype is not None

    if three_way:
        # Baseline (autoregressive) — ground truth for three-way
        print(f"Generating {baseline_dtype} baseline goldens on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=baseline_dtype,
            device_map="cpu",
            trust_remote_code=True,
        ).to(baseline_dtype)
        model.eval()
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer([prompt], return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, max_new_tokens=num_tokens, do_sample=False
                )
            teacher_tokens = generated[0, input_ids.shape[1] :]
            logits, kv = extract_hf_kv_caches_teacher_forced(
                model, input_ids, teacher_tokens, return_logits=True
            )
            goldens[i] = {
                "prompt": prompt,
                "input_ids": input_ids,
                "baseline_logits": logits,
                "baseline_kv": kv,
                "teacher_tokens": teacher_tokens,
            }
            print(
                f"  prompt {i}: {input_ids.shape[1]} prompt tokens, {num_tokens} generated"
            )
        del model
        gc.collect()

    # Expected (reference for comparison)
    print(f"Generating {expected_dtype} expected goldens...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=expected_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    if three_way:
        # Teacher-forced with baseline's tokens
        for i in goldens:
            logits, kv = extract_hf_kv_caches_teacher_forced(
                model,
                goldens[i]["input_ids"],
                goldens[i]["teacher_tokens"],
                return_logits=True,
            )
            goldens[i]["expected_logits"] = logits
            goldens[i]["expected_kv"] = kv
            print(f"  prompt {i}: expected_logits {logits.shape}")
    else:
        # Two-way: expected is autoregressive (it's the only reference)
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer([prompt], return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, max_new_tokens=num_tokens, do_sample=False
                )
            teacher_tokens = generated[0, input_ids.shape[1] :]
            logits, kv = extract_hf_kv_caches_teacher_forced(
                model, input_ids, teacher_tokens, return_logits=True
            )
            goldens[i] = {
                "prompt": prompt,
                "input_ids": input_ids,
                "expected_logits": logits,
                "expected_kv": kv,
            }
            print(
                f"  prompt {i}: {input_ids.shape[1]} prompt tokens, {num_tokens} generated"
            )

    del model
    gc.collect()
    return goldens


# =============================================================================
# vLLM helpers
# =============================================================================


def create_vllm(
    model_id, tp_size, max_model_len, hf_overrides=None, neuron_config=None
):
    """Create vLLM LLM instance.

    Args:
        model_id: HuggingFace model ID.
        tp_size: Tensor parallel size.
        max_model_len: Maximum model sequence length.
        hf_overrides: Optional HF config overrides dict.
        neuron_config: Optional Neuron config dict.

    Returns:
        LLM instance.

    Example:
        >>> llm = create_vllm("meta-llama/Llama-3.3-70B-Instruct", 32, 256)
    """
    nc = neuron_config or {}
    nc.setdefault("on_device_sampling_config", {})
    nc.setdefault("num_batched_tokens_buckets", [max_model_len])
    nc.setdefault("num_seqs_buckets", [2])

    kwargs = dict(
        model=model_id,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        max_num_seqs=2,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={"neuron_config": nc},
    )
    if hf_overrides:
        kwargs["hf_overrides"] = hf_overrides
    return LLM(**kwargs)


def make_generate_fn(llm, max_tokens):
    """Create generate_fn compatible with ``logit_validation``.

    Args:
        llm: vLLM LLM instance.
        max_tokens: Default max tokens to generate.

    Returns:
        Callable[[List[List[int]]], torch.Tensor] returning logits
        shaped ``[num_tokens, batch, vocab]``.

    Example:
        >>> fn = make_generate_fn(llm, 16)
        >>> logits = fn([[1, 2, 3]])
    """
    vocab = llm.get_tokenizer().vocab_size

    def generate_fn(input_ids_list, max_new_tokens=None):
        n = max_new_tokens or max_tokens
        sp = SamplingParams(temperature=0, max_tokens=n, logprobs=vocab)
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in input_ids_list],
            sp,
        )
        batch_logits = []
        for output in outputs:
            seq_logits = []
            for completion in output.outputs:
                if not completion.logprobs:
                    continue
                for logprob_dict in completion.logprobs:
                    max_id = max(logprob_dict.keys())
                    t = torch.full((max(vocab, max_id + 1),), -float("inf"))
                    for tid, lp in logprob_dict.items():
                        t[tid] = lp.logprob
                    seq_logits.append(t)
            if seq_logits:
                batch_logits.append(torch.stack(seq_logits))
        return torch.stack(batch_logits, dim=1)

    return generate_fn


# =============================================================================
# Per-prompt analysis
# =============================================================================


def run_prompt(llm, kv_config, generate_fn, golden, output_prefix):
    """Run logit validation + KV analysis for one prompt.

    Args:
        llm: vLLM LLM instance.
        kv_config: KV cache config from ``extract_vllm_kv_cache_config``.
        generate_fn: vLLM generate function.
        golden: Dict with baseline/expected logits, KV, input_ids.
        output_prefix: Path prefix for output files (no extension).

    Returns:
        Dict with result metadata, or None on failure.

    Example:
        >>> meta = run_prompt(llm, cfg, gen_fn, goldens[0], "out/kv_prompt_0")
    """
    input_ids = golden["input_ids"]
    prompt_len = input_ids.shape[1]

    def kv_extract_fn(seq_len):
        paged = extract_vllm_kv_caches(llm, kv_config)
        tables = extract_vllm_block_tables(llm)
        return reconstruct_contiguous_kv(paged, kv_config, tables, seq_len)

    result = logit_validation(
        input_ids=input_ids.tolist(),
        generate_fn=generate_fn,
        expected_logits=golden["expected_logits"],
        baseline_logits=golden.get("baseline_logits"),
        kv_extract_fn=kv_extract_fn,
        colorize=True,
        suppress_passing=True,
    )

    # Unpack result
    if isinstance(result, tuple) and len(result) == 3:
        passed, logit_results, vllm_kv = result
    elif isinstance(result, tuple) and len(result) == 2:
        passed, logit_results = result
        vllm_kv = None
    else:
        passed, logit_results, vllm_kv = result, None, None

    # Extract divergence indices
    divergence_indices = []
    if logit_results:
        for tok_idx, tok_result in enumerate(logit_results[0]):
            if tok_result.get("divergence", False):
                divergence_indices.append(prompt_len + tok_idx)
    if divergence_indices:
        print(f"  Divergence at: {divergence_indices}")

    if vllm_kv is None:
        print("  WARNING: no vLLM KV extracted, skipping")
        return None

    baseline_kv = golden.get("baseline_kv")
    expected_kv = golden["expected_kv"]

    kv_result, raw_errors = compare_kv_caches(
        expected_kv,
        vllm_kv,
        baseline_kv=baseline_kv,
        return_raw_errors=True,
    )

    export_kv_html_report(
        kv_result,
        f"{output_prefix}.html",
        prompt_len=prompt_len,
        divergence_indices=divergence_indices or None,
    )
    export_kv_json(kv_result, f"{output_prefix}.json")

    layers = sorted(k for k in kv_result[0] if not k.endswith("._bc"))
    max_k = max(max(h.k_linf for h in t[ly]) for t in kv_result for ly in layers)
    print(f"  tokens={len(kv_result)}, max_K_linf={max_k:.4f}, passed={passed}")

    return {
        "tokens": len(kv_result),
        "max_k_linf": max_k,
        "passed": passed,
        "prompt_len": prompt_len,
        "raw_errors": raw_errors,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Analysis: two-way or three-way comparison on Neuron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--goldens-path",
        type=str,
        default=None,
        help="Load pre-saved goldens.pkl (skip CPU generation)",
    )
    parser.add_argument(
        "--two-way",
        action="store_true",
        help="Two-way mode: skip baseline, compare expected vs actual only",
    )
    parser.add_argument(
        "--hf-overrides",
        type=str,
        default=None,
        help="JSON string for hf_overrides (e.g. '{\"quantization_config\": {}}')",
    )
    parser.add_argument(
        "--neuron-config",
        type=str,
        default=None,
        help="JSON string for neuron_config overrides",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    baseline_dtype = None if args.two_way else torch.float32

    # --- Goldens ---
    goldens_path = args.goldens_path or os.path.join(args.output_dir, "goldens.pkl")
    if os.path.exists(goldens_path):
        print(f"Loading goldens from {goldens_path}")
        with open(goldens_path, "rb") as f:
            goldens = pickle.load(f)
    else:
        goldens = generate_goldens(
            args.model, prompts, args.num_tokens, baseline_dtype=baseline_dtype
        )
        save_path = os.path.join(args.output_dir, "goldens.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(goldens, f)
        print(f"Saved goldens to {save_path}")

    # --- vLLM ---
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else None
    neuron_config = json.loads(args.neuron_config) if args.neuron_config else None

    print(f"\nStarting vLLM (TP={args.tp_size}, max_model_len={args.max_model_len})...")
    llm = create_vllm(
        args.model, args.tp_size, args.max_model_len, hf_overrides, neuron_config
    )
    kv_config = extract_vllm_kv_cache_config(llm)
    generate_fn = make_generate_fn(llm, args.num_tokens)

    # Arm block-table snapshotting BEFORE any generate(): a request's block
    # rows can be freed before generate() returns, and only an armed forward
    # pass keeps the host-side copy that extract_vllm_block_tables reads.
    enable_kv_snapshot(llm)

    # --- Per-prompt analysis ---
    all_meta = []
    all_raw_errors = []
    prompt_lens = []

    for i in sorted(goldens.keys()):
        g = goldens[i]
        print(f"\n{'=' * 60}")
        print(f"Prompt {i}: '{g['prompt'][:50]}...' (len={g['input_ids'].shape[1]})")
        print(f"{'=' * 60}")

        prefix = os.path.join(args.output_dir, f"kv_prompt_{i}")
        meta = run_prompt(llm, kv_config, generate_fn, g, prefix)
        if meta:
            all_meta.append({"idx": i, **meta})
            all_raw_errors.append(meta["raw_errors"])
            prompt_lens.append(meta["prompt_len"])

    # --- Aggregated BC (three-way only) ---
    if not args.two_way and len(all_raw_errors) > 1:
        agg = aggregate_kv_bc_across_prompts(all_raw_errors, prompt_lens)
        export_aggregated_bc_html(
            agg,
            os.path.join(args.output_dir, "aggregated_bc.html"),
            num_prompts=len(all_meta),
        )

    # --- Cleanup ---
    cleanup_kv_snapshot(llm)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in all_meta:
        print(
            f"  prompt_{r['idx']}: tokens={r['tokens']}, "
            f"max_K_linf={r['max_k_linf']:.4f}, passed={r['passed']}"
        )


if __name__ == "__main__":
    main()
