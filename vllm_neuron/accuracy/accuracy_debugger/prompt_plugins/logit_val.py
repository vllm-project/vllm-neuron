# SPDX-License-Identifier: Apache-2.0
"""Logit validation prompt plugin."""

from __future__ import annotations

import os

from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.base import (
    PluginContext,
    PromptPlugin,
)


class LogitValPlugin(PromptPlugin):
    name = "logit_val"

    def __init__(self, tol_map: dict | None = None):
        self.tol_map = tol_map

    def pre_llm(self, ctx: PluginContext) -> None:
        if ctx.goldens is None:
            from functools import partial

            from vllm_neuron.accuracy.goldens.reference_logits import (
                generate_three_way_reference_logits,
            )
            from vllm_neuron.accuracy.goldens.reference_model import init_hf_model

            ctx.goldens = generate_three_way_reference_logits(
                ctx.model_checkpoint,
                ctx.dtype,
                ctx.output_length,
                ctx.prompts,
                ctx.tokenizer,
                # Load non-SDPA models (e.g. GPT-OSS) directly via eager; the
                # debugger does not layer chunked attention on top.
                model_loader=partial(init_hf_model, eager_attn_fallback=True),
            )

    def run(self, ctx: PluginContext) -> dict:
        from vllm_neuron.accuracy.logit_validation import (
            multi_prompt_logit_validation,
        )

        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in ctx.goldens["input_ids"]
        ]
        kwargs = {}
        if self.tol_map is not None:
            kwargs["tol_map"] = self.tol_map
        result = multi_prompt_logit_validation(
            prompts_input_ids=prompts_input_ids,
            generate_fn=ctx.generate_fn,
            prompts_expected_logits=ctx.goldens["dtype_logits"],
            prompts_baseline_logits=ctx.goldens["fp32_logits"],
            colorize=True,
            replicate_to_batch_size=ctx.batch_size,
            **kwargs,
        )
        return {
            "passed": result.passed,
            "per_prompt_results": result.per_prompt_results,
        }

    def save(self, ctx: PluginContext, results: dict) -> None:
        from vllm_neuron.accuracy.logit_visualization import visualize_logit_results

        os.makedirs(ctx.output_dir, exist_ok=True)
        for i, (passed, per_prompt_results) in enumerate(results["per_prompt_results"]):
            if per_prompt_results:
                vis_dir = os.path.join(ctx.output_dir, "logit_validation")
                # Only visualize batch 0 — other batches are replicated copies
                batch0_results = [per_prompt_results[0]] if per_prompt_results else []
                visualize_logit_results(batch0_results, vis_dir)
