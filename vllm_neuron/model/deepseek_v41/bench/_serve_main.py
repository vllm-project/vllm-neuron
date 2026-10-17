# SPDX-License-Identifier: Apache-2.0
"""`vllm serve`, with this port imported first.

**Why this file has to exist.** `vllm serve` (and
`python -m vllm.entrypoints.openai.api_server`) builds `ModelConfig` before it
loads the Neuron platform plugin, the model registry, or anything else from
`vllm_neuron`. `ModelConfig` resolves the checkpoint through
`transformers.AutoConfig`, and `deepseek_v41` is an architecture upstream
`transformers` has never seen, so the server dies on

    pydantic_core._pydantic_core.ValidationError: 1 validation error for
    ModelConfig
      Value error, The checkpoint you are trying to load has model type
      `deepseek_v41` but Transformers does not recognize this architecture.

before a single worker starts. That is the same failure the port's own
`__init__.py` docstring opens with, and the port fixes it by registering the
config classes with `AutoConfig` at import time.

The *verifier* never hits it because it runs as
`python -m vllm_neuron.model.deepseek_v41.verifier.run_verifier`, and `runpy`
imports every parent package before the target module — so the port is imported,
and the registration done, before `from vllm import LLM` executes. A server
launched by name imports nothing of the port, so nothing registers.

Importing the port here restores that ordering, then hands over to vLLM's own
`api_server` through `runpy` so the server behaves exactly as `python -m` would:
same argument parsing, same `__main__` semantics, no reimplementation of an
entry point that is not ours.

Not a workaround for a bug in the port so much as the consequence of a choice it
made: it registers from its own package, and something has to import that
package. Serving this checkpoint by any route that does not import
`vllm_neuron.model.deepseek_v41` will fail the same way, which is worth knowing
beyond this benchmark.
"""

import runpy

# The import *is* the point — it registers `deepseek_v41` with `AutoConfig`,
# puts `DeepseekV41ForCausalLM` in vLLM's `ModelRegistry`, widens the platform's
# supported-quantization list and installs the port's config hooks. Everything
# below depends on it having happened.
#
# It is deliberately at module scope rather than inside the guard: vLLM starts
# EngineCore with `spawn`, and a spawned child re-imports this module to rebuild
# its `__main__`. Importing the port here means the child gets the registration
# too, which is the same reason the port's docstring gives for the worker path.
import vllm_neuron.model.deepseek_v41  # noqa: F401

# **The guard is load-bearing, not boilerplate.** Without it the `runpy` call
# runs in every spawned child as well as the parent, and each child tries to
# start its own server:
#
#     RuntimeError: An attempt has been made to start a new process before the
#     current process has finished its bootstrapping phase. ... you have
#     forgotten to use the proper idiom in the main module
#
# which is the recursion that error exists to name. Guarded, the parent serves
# and the children only pick up the import above.
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
