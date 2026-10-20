# SPDX-License-Identifier: Apache-2.0
"""Tensor compare prompt plugin for the accuracy debugger.

Captures intermediate tensors from HF (FP32, BF16) and vLLM Neuron,
then runs three-way comparison using the tensor_compare module with
reconstruction support.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from typing import Callable, List, Optional

import torch

from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.base import (
    PluginContext,
    PromptPlugin,
)

logger = logging.getLogger("accuracy_debugger")

DEFAULT_MAX_TOKENS = 3

# Type for reconstruction functions (see tensor_compare.ReconstructionFn)
ReconstructionFn = Callable[[List[torch.Tensor], str, str, List[int]], torch.Tensor]


class TensorComparePlugin(PromptPlugin):
    name = "tensor_compare"
    needs_shared_llm = False

    def __init__(
        self,
        modules: list[str],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tp_size: int | None = None,
        max_num_seqs: int = 2,
        max_l2_ratio: float = 3.0,
        module_order: list[str] | None = None,
        reconstruction_fn: Optional[ReconstructionFn] = None,
    ):
        self.modules = modules
        self.max_tokens = max_tokens
        self.tp_size = tp_size
        self.max_num_seqs = max_num_seqs
        self.max_l2_ratio = max_l2_ratio
        self.module_order = module_order
        self.reconstruction_fn = reconstruction_fn or _default_reconstruct
        self._hf_fp32_dir: str = ""
        self._hf_bf16_dir: str = ""

    def pre_llm(self, ctx: PluginContext) -> None:
        """Capture HF FP32 and BF16 tensors before LLM creation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from vllm_neuron.accuracy.tensor_capture import (
            CaptureWriter,
            ModelCapture,
            TensorRegistry,
        )

        base_dir = os.path.join(ctx.output_dir, "tensor_compare")
        self._hf_fp32_dir = os.path.join(base_dir, "hf_fp32")
        self._hf_bf16_dir = os.path.join(base_dir, "hf_bf16")

        tokenizer = AutoTokenizer.from_pretrained(
            ctx.model_checkpoint, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for dtype, label, capture_dir in [
            (torch.float32, "FP32", self._hf_fp32_dir),
            (ctx.dtype or torch.bfloat16, "BF16", self._hf_bf16_dir),
        ]:
            if os.path.exists(capture_dir) and os.listdir(capture_dir):
                logger.info("Reusing existing HF %s captures at %s", label, capture_dir)
                continue
            if os.path.exists(capture_dir):
                shutil.rmtree(capture_dir)
            TensorRegistry.reset_instance()

            logger.info("Loading HF %s model for tensor capture...", label)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    ctx.model_checkpoint,
                    torch_dtype=dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            except (ValueError, TypeError):
                # Some architectures (e.g. GPT-OSS) reject SDPA; fall back to eager.
                model = AutoModelForCausalLM.from_pretrained(
                    ctx.model_checkpoint,
                    torch_dtype=dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
            # MXFP4 models dequantize to BF16 regardless of torch_dtype arg.
            # model.to(dtype) doesn't cast 3D expert weight tensors (e.g.
            # GptOssExperts.gate_up_proj), so we cast each parameter directly.
            if any(p.dtype != dtype for p in model.parameters()):
                for p in model.parameters():
                    p.data = p.data.to(dtype)
            capture = ModelCapture(
                model=model, modules=self.modules, capture_dir=capture_dir
            )
            TensorRegistry._instance = capture._registry
            writer = CaptureWriter(capture_dir, dp_rank=0)
            writer.enable()

            for i, prompt in enumerate(ctx.prompts):
                inputs = tokenizer(prompt, return_tensors="pt")
                generated_ids = inputs["input_ids"].clone()
                with torch.inference_mode():
                    for step in range(self.max_tokens):
                        capture._registry.clear()
                        model_output = model(input_ids=generated_ids)
                        captures = capture._registry.get_all_tensors()
                        capture_names = capture._registry.get_all_names()
                        positions = torch.arange(generated_ids.shape[1])
                        if captures:
                            writer.write(
                                tuple(captures),
                                capture_names,
                                req_ids=[str(i)],
                                positions=positions,
                                is_prefill=(step == 0),
                            )
                        next_token = torch.argmax(model_output.logits[0, -1, :]).item()
                        generated_ids = torch.cat(
                            [generated_ids, torch.tensor([[next_token]])], dim=1
                        )

            del model, capture
            gc.collect()
            logger.info("HF %s captures saved to %s", label, capture_dir)

    def run(self, ctx: PluginContext) -> dict:
        from vllm import LLM, SamplingParams

        from vllm_neuron.accuracy.tensor_alignment_utils import (
            align_and_truncate_hidden,
            align_decode_captures,
            hf_reference_reconstruction,
        )
        from vllm_neuron.accuracy.tensor_compare import (
            compare_captures_three_way,
            compute_aggregate_metrics,
            print_three_way_report,
        )
        from vllm_neuron.accuracy.tensor_io import read as tensor_io_read

        base_dir = os.path.join(ctx.output_dir, "tensor_compare")
        neuron_dir = os.path.join(base_dir, "neuron")

        if os.path.exists(neuron_dir):
            shutil.rmtree(neuron_dir)

        tp = self.tp_size or ctx.server_cfg.get("tp_degree", 8)
        max_model_len = ctx.server_cfg.get("max_model_len", 8192)
        additional_config = dict(ctx.server_cfg.get("additional_config", {}))
        neuron_config = dict(additional_config.get("neuron_config", {}))
        neuron_config.setdefault("on_device_sampling_config", {"all_greedy": True})
        neuron_config.setdefault("num_batched_tokens_buckets", [max_model_len])
        neuron_config["num_seqs_buckets"] = [self.max_num_seqs]
        neuron_config["tensor_capture"] = {
            "modules": self.modules,
            "capture_dir": neuron_dir,
        }

        logger.info("Loading Neuron model for tensor capture...")
        llm_kwargs = dict(
            model=ctx.model_checkpoint,
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=max_model_len,
            tensor_parallel_size=tp,
            enable_prefix_caching=False,
            additional_config={"neuron_config": neuron_config},
        )
        hf_overrides = ctx.server_cfg.get("hf_overrides")
        if hf_overrides:
            llm_kwargs["hf_overrides"] = hf_overrides
        llm = LLM(**llm_kwargs)

        params = SamplingParams(temperature=0, max_tokens=self.max_tokens)
        for prompt in ctx.prompts:
            llm.generate([prompt], params)

        del llm
        gc.collect()
        logger.info("Neuron captures saved to %s", neuron_dir)

        # Read and compare
        fp32 = tensor_io_read(self._hf_fp32_dir)
        bf16 = tensor_io_read(self._hf_bf16_dir)
        neuron = tensor_io_read(neuron_dir)

        fp32 = align_decode_captures(fp32, neuron)
        bf16 = align_decode_captures(bf16, neuron)

        three_way_kwargs = dict(
            reference_reconstruction_fn=hf_reference_reconstruction,
            target_reconstruction_fn=self.reconstruction_fn,
            alignment_fn=align_and_truncate_hidden,
            module_order=self.module_order,
        )

        prefill_results = compare_captures_three_way(
            fp32, bf16, neuron, phase="prefill", **three_way_kwargs
        )
        decode_results = compare_captures_three_way(
            fp32, bf16, neuron, phase="decode", **three_way_kwargs
        )

        print_three_way_report(
            prefill_results, label_expected="HF BF16", label_actual="Neuron"
        )
        print_three_way_report(
            decode_results, label_expected="HF BF16", label_actual="Neuron"
        )

        prefill_agg = compute_aggregate_metrics(prefill_results)
        decode_agg = compute_aggregate_metrics(decode_results)

        passed = _evaluate_results(prefill_results, decode_results, self.max_l2_ratio)

        return {
            "passed": passed,
            "prefill_results": prefill_results,
            "decode_results": decode_results,
            "prefill_aggregate": prefill_agg,
            "decode_aggregate": decode_agg,
        }

    def save(self, ctx: PluginContext, results: dict) -> None:
        if not results:
            return

        output_dir = os.path.join(ctx.output_dir, "tensor_compare")
        os.makedirs(output_dir, exist_ok=True)

        summary = {
            "passed": results["passed"],
            "max_l2_ratio_threshold": self.max_l2_ratio,
            "prefill_summary": _summarize_results(results["prefill_results"]),
            "decode_summary": _summarize_results(results["decode_results"]),
        }
        summary_path = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Tensor compare: %s (report: %s)",
            "PASSED" if results["passed"] else "FAILED",
            summary_path,
        )


def _default_reconstruct(
    rank_tensors: List[torch.Tensor],
    module_name: str,
    phase: str = "prefill",
    positions: List[int] = None,
) -> torch.Tensor:
    """Default reconstruction: rank 0 with padding stripped via positions."""
    from vllm_neuron.accuracy.tensor_alignment_utils import count_real_tokens

    tensor = rank_tensors[0]
    if positions:
        real = count_real_tokens(positions)
        if tensor.shape[0] > real:
            return tensor[:real]
    return tensor


def _evaluate_results(
    prefill_results: dict, decode_results: dict, max_l2_ratio: float = 3.0
) -> bool:
    """Evaluate pass/fail based on L2 ratios across all modules.

    Decode steps where the input embedding diverges (bc < 0.5 on
    embed_tokens) indicate a greedy tie-break picked a different token
    than HuggingFace — all subsequent modules/steps diverge by definition.
    These steps are excluded from the assertion.
    """
    for prompt_results in prefill_results.values():
        for step_results in prompt_results.values():
            for r in step_results:
                if not r.shape_match:
                    return False
                if r.l2_ratio > max_l2_ratio:
                    return False

    for prompt_results in decode_results.values():
        for step in sorted(prompt_results.keys(), key=int):
            step_results = prompt_results[step]
            embed_diverged = any(
                r.name.startswith("model_embed_tokens")
                and r.shape_match
                and r.bc < 0.95
                and r.l2_ratio >= 1.0
                for r in step_results
            )
            if embed_diverged:
                logger.info(
                    "Skipping decode step %s and beyond: the token selected "
                    "at the previous step differs between baseline (HF) and "
                    "target (Neuron) due to greedy tie-breaking "
                    "(model_embed_tokens bc < 0.95, l2_ratio >= 1.0)",
                    step,
                )
                break
            for r in step_results:
                if not r.shape_match:
                    return False
                if r.l2_ratio > max_l2_ratio:
                    return False
    return True


def _summarize_results(results: dict) -> dict:
    """Create a JSON-serializable summary of comparison results."""
    summary = {}
    for prompt_key, prompt_results in results.items():
        prompt_summary = {}
        for step, step_results in prompt_results.items():
            step_summary = []
            for r in step_results:
                step_summary.append(
                    {
                        "name": r.name,
                        "linf_ratio": round(r.linf_ratio, 4),
                        "l2_ratio": round(r.l2_ratio, 4),
                        "bc": round(r.bc, 4),
                        "passed": r.passed,
                    }
                )
            prompt_summary[step] = step_summary
        summary[prompt_key] = prompt_summary
    return summary
