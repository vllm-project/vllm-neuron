# SPDX-License-Identifier: Apache-2.0
"""Base class for prompt analysis plugins."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class PluginContext:
    """Shared state passed to every plugin in a prompt analysis run."""

    model_checkpoint: str
    prompts: list[str]
    server_cfg: dict
    output_length: int
    output_dir: str
    # Populated by the orchestrator before plugins run:
    llm: Any = None
    generate_fn: Any = None
    goldens: dict | None = None
    tokenizer: Any = None
    dtype: Any = None
    batch_size: int = 1


class PromptPlugin(abc.ABC):
    """Interface that every prompt-analysis plugin must implement."""

    name: str  # registry key, e.g. "logit_val"
    needs_shared_llm: bool = True  # Set False for plugins that manage their own LLM

    @abc.abstractmethod
    def pre_llm(self, ctx: PluginContext) -> None:
        """Run before the vLLM LLM is created (e.g. extract HF KV caches)."""

    @abc.abstractmethod
    def run(self, ctx: PluginContext) -> dict:
        """Run the analysis. Returns a results dict."""

    @abc.abstractmethod
    def save(self, ctx: PluginContext, results: dict) -> None:
        """Persist artifacts (HTML reports, tensors, etc.)."""
