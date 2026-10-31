# SPDX-License-Identifier: Apache-2.0
"""Verifier for the DeepSeek-V4.1-Flash port.

`run_verifier` is the entry point; `prompts` builds the input set and `judge`
decides whether the completions are reasonable. Nothing is imported eagerly
here — `run_verifier` pulls in vLLM, and importing this package should not.
"""

__all__ = ["judge", "prompts", "run_verifier"]
