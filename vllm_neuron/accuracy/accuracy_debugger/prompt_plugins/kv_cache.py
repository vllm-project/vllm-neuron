# SPDX-License-Identifier: Apache-2.0
"""KV cache analysis prompt plugin."""

from __future__ import annotations

import gc
import logging
import os

from vllm_neuron.accuracy.accuracy_debugger.prompt_plugins.base import (
    PluginContext,
    PromptPlugin,
)

logger = logging.getLogger("accuracy_debugger")


DEFAULT_KV_THRESHOLDS = {
    "min_cos_similarity": 0.99,
    "max_linf_multiplier": 2.0,
    "min_bc": 0.90,
}


class KvCachePlugin(PromptPlugin):
    name = "kv_cache"

    def __init__(self, thresholds: dict | None = None):
        self.fp32_kv = None
        self.hf_kv = None
        self.teacher_seq = None
        self.input_ids = None
        self.thresholds = thresholds or DEFAULT_KV_THRESHOLDS

    def pre_llm(self, ctx: PluginContext) -> None:
        """Extract HF KV caches before LLM creation to avoid OOM."""
        import torch

        from vllm_neuron.accuracy.goldens.reference_logits import (
            generate_reference_logits,
        )
        from vllm_neuron.accuracy.goldens.reference_model import init_hf_model
        from vllm_neuron.accuracy.kv_cache_analysis import (
            extract_hf_kv_caches_teacher_forced,
        )

        _ids = ctx.tokenizer(
            ctx.prompts[:1], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        self.input_ids = _ids[0].tolist()

        fp32_model = init_hf_model(
            ctx.model_checkpoint, torch.float32, eager_attn_fallback=True
        )
        baseline = generate_reference_logits(fp32_model, _ids, ctx.output_length)
        self.teacher_seq = baseline.argmax(dim=2).squeeze(1)
        _, self.fp32_kv = extract_hf_kv_caches_teacher_forced(
            fp32_model, _ids, self.teacher_seq, return_logits=True
        )
        del fp32_model

        bf16_model = init_hf_model(
            ctx.model_checkpoint, ctx.dtype, eager_attn_fallback=True
        )
        _, self.hf_kv = extract_hf_kv_caches_teacher_forced(
            bf16_model, _ids, self.teacher_seq, return_logits=True
        )
        del bf16_model, baseline
        gc.collect()
        logger.info("HF KV caches extracted and models freed before LLM creation")

    def run(self, ctx: PluginContext) -> dict:
        if self.fp32_kv is None:
            return {}

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from vllm_neuron.accuracy.kv_cache_analysis import (
            compare_kv_caches,
            enable_kv_snapshot,
            extract_vllm_block_tables,
            extract_vllm_kv_cache_config,
            extract_vllm_kv_caches,
            print_kv_report,
            reconstruct_contiguous_kv,
        )

        enable_kv_snapshot(ctx.llm)

        # Feed the same teacher-forced sequence to vLLM so KV caches are comparable.
        # Use the same input_ids from pre_llm to guarantee identical tokenization.
        input_ids = self.input_ids
        prompt_len = len(input_ids)
        teacher_tokens = self.teacher_seq.squeeze().tolist()
        if not isinstance(teacher_tokens, list):
            teacher_tokens = [teacher_tokens]
        full_seq = list(input_ids) + teacher_tokens

        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        ctx.llm.generate([TokensPrompt(prompt_token_ids=full_seq)], sampling_params)

        kv_config = extract_vllm_kv_cache_config(ctx.llm)
        paged_kv = extract_vllm_kv_caches(ctx.llm, kv_config)
        block_tables = extract_vllm_block_tables(ctx.llm)

        seq_len = prompt_len + ctx.output_length
        vllm_kv = reconstruct_contiguous_kv(paged_kv, kv_config, block_tables, seq_len)

        print("\n" + "=" * 60)
        print("KV CACHE ANALYSIS (Three-Way)")
        print("=" * 60)
        kv_result = compare_kv_caches(
            expected_kv=self.hf_kv, actual_kv=vllm_kv, baseline_kv=self.fp32_kv
        )
        print_kv_report(kv_result, max_tokens=min(8, len(kv_result)))

        passed, failures = self._evaluate(kv_result)

        print("\n" + "-" * 60)
        print(f"KV CACHE RESULT: {'PASSED' if passed else 'FAILED'}")
        if failures:
            print(f"  Failures ({len(failures)}):")
            for f in failures[:10]:
                print(f"    {f}")
            if len(failures) > 10:
                print(f"    ... and {len(failures) - 10} more")
        print("-" * 60)

        return {
            "passed": passed,
            "failures": failures,
            "kv_result": kv_result,
            "prompt_len": prompt_len,
            "vllm_kv": vllm_kv,
        }

    def _evaluate(self, kv_result) -> tuple[bool, list[str]]:
        """Evaluate KV cache results against thresholds."""
        min_cos = self.thresholds["min_cos_similarity"]
        max_linf_mult = self.thresholds["max_linf_multiplier"]
        min_bc = self.thresholds["min_bc"]

        failures = []
        for t, token_data in enumerate(kv_result):
            for layer, heads in token_data.items():
                if layer.endswith("._bc"):
                    bc = heads
                    if bc.k_bc < min_bc:
                        failures.append(
                            f"token {t} {layer}: K BC={bc.k_bc:.4f} < {min_bc}"
                        )
                    if bc.v_bc < min_bc:
                        failures.append(
                            f"token {t} {layer}: V BC={bc.v_bc:.4f} < {min_bc}"
                        )
                    continue

                for h, m in enumerate(heads):
                    if m.k_cos < min_cos:
                        failures.append(
                            f"token {t} {layer} head {h}: "
                            f"K cos={m.k_cos:.4f} < {min_cos}"
                        )
                    if m.v_cos < min_cos:
                        failures.append(
                            f"token {t} {layer} head {h}: "
                            f"V cos={m.v_cos:.4f} < {min_cos}"
                        )
                    if m.base_k_linf > 0 and m.k_linf > max_linf_mult * m.base_k_linf:
                        ratio = m.k_linf / m.base_k_linf
                        failures.append(
                            f"token {t} {layer} head {h}: "
                            f"K linf={m.k_linf:.2e} > "
                            f"{max_linf_mult}x base ({m.base_k_linf:.2e}) "
                            f"[ratio={ratio:.2f}x]"
                        )
                    if m.base_v_linf > 0 and m.v_linf > max_linf_mult * m.base_v_linf:
                        ratio = m.v_linf / m.base_v_linf
                        failures.append(
                            f"token {t} {layer} head {h}: "
                            f"V linf={m.v_linf:.2e} > "
                            f"{max_linf_mult}x base ({m.base_v_linf:.2e}) "
                            f"[ratio={ratio:.2f}x]"
                        )

        return len(failures) == 0, failures

    def save(self, ctx: PluginContext, results: dict) -> None:
        if not results:
            return

        import torch

        from vllm_neuron.accuracy.kv_cache_visualize import (
            export_html_report as export_kv_html,
        )

        kv_dir = os.path.join(ctx.output_dir, "kv_analysis")
        os.makedirs(kv_dir, exist_ok=True)
        export_kv_html(
            results["kv_result"],
            os.path.join(kv_dir, "kv_report.html"),
            prompt_len=results["prompt_len"],
        )
        torch.save(self.fp32_kv, os.path.join(kv_dir, "fp32_kv.pt"))
        torch.save(self.hf_kv, os.path.join(kv_dir, "hf_kv.pt"))
        torch.save(results["vllm_kv"], os.path.join(kv_dir, "vllm_kv.pt"))
