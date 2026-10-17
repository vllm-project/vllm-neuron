# SPDX-License-Identifier: Apache-2.0
"""Latency/throughput benchmark for the DeepSeek-V4.1-Flash port.

`run_bench` is the entry point. Nothing is imported eagerly here — it drives
`vllm serve` and `vllm bench serve` as subprocesses, and importing this package
should not pull vLLM in.

Separate from `verifier/` on purpose: the verifier decides whether the port is
*correct* and says nothing about speed; this decides how fast it is and says
nothing about correctness. Both serve the same checkpoint.
"""

__all__ = ["run_bench"]
