# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Type definitions for vLLM Neuron accuracy validation.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Any, Dict, Optional


@dataclass
class MultiPromptValidationResult:
    """Results from multi-prompt validation with aggregate metrics."""

    passed: bool
    per_prompt_results: List[Tuple[bool, Optional[List[List[dict]]]]]
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Compact repr to avoid dumping massive tensors into pytest assertion messages.

        Without this, the default dataclass repr serializes entire per-token tensors
        (128K+ floats each) into pytest assertion output, producing 164KB+ single log
        lines that overwhelm the container log pipeline and cause log truncation.
        """
        n_prompts = len(self.per_prompt_results)
        n_passed = sum(1 for ok, _ in self.per_prompt_results if ok)
        n_tokens_per_prompt = []
        for _, token_results in self.per_prompt_results:
            if token_results is None:
                n_tokens_per_prompt.append(0)
            elif token_results and isinstance(token_results[0], list):
                # token_results is List[List[dict]] (batches of token results)
                n_tokens_per_prompt.append(sum(len(batch) for batch in token_results))
            else:
                n_tokens_per_prompt.append(len(token_results))
        failed_prompt_indices = [
            i for i, (ok, _) in enumerate(self.per_prompt_results) if not ok
        ]
        parts = [
            f"MultiPromptValidationResult("
            f"passed={self.passed}, "
            f"prompts={n_passed}/{n_prompts} passed, "
            f"tokens_per_prompt={n_tokens_per_prompt}",
        ]
        if failed_prompt_indices:
            parts.append(f", failed_prompts={failed_prompt_indices}")
            # Include compact summary of failed token results
            for idx in failed_prompt_indices:
                _, token_results = self.per_prompt_results[idx]
                if token_results is None:
                    continue
                # Flatten batches if nested: List[List[dict]] -> List[dict]
                flat_tokens = []
                if token_results and isinstance(token_results[0], list):
                    for batch in token_results:
                        flat_tokens.extend(batch)
                else:
                    flat_tokens = token_results
                failed_tokens = [
                    j
                    for j, t in enumerate(flat_tokens)
                    if isinstance(t, dict) and not t.get("passed", True)
                ]
                if failed_tokens:
                    parts.append(
                        f", prompt_{idx}_failed_tokens={failed_tokens[:10]}"
                        f"{'...' if len(failed_tokens) > 10 else ''}"
                    )
        if self.aggregate_metrics:
            # Only show top-level keys, not full tensor values
            parts.append(f", aggregate_keys={list(self.aggregate_metrics.keys())}")
        parts.append(")")
        return "".join(parts)
