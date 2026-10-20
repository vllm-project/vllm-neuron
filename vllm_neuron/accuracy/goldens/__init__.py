# SPDX-License-Identifier: Apache-2.0
"""HuggingFace golden-reference builders for accuracy validation.

Grouped by layer into three modules that build HF "golden" reference outputs
used to validate Neuron model accuracy:

- :mod:`~vllm_neuron.accuracy.goldens.reference_model` — HF model **loading**
  (:func:`~vllm_neuron.accuracy.goldens.reference_model.init_hf_model`).
- :mod:`~vllm_neuron.accuracy.goldens.reference_logits` — reference-logit
  **generation** + golden orchestration (``generate_reference_logits``,
  ``generate_teacher_forced_logits``, ``generate_teacher_forced_logits_fp8``,
  ``generate_three_way_reference_logits``).
- :mod:`~vllm_neuron.accuracy.goldens.fp8_kv_golden` — FP8 KV cache **simulation
  mechanism** (``load_kv_scales``, ``hf_simulate_fp8_kv_cache``).

The vLLM ``generate_fn`` adapter (``create_offline_vllm_generate_fn``) that turns a
running ``LLM`` into the *target* logits to compare against lives in
:mod:`vllm_neuron.accuracy.logit_validation` and is also exported here.

The reference loader (:func:`~vllm_neuron.accuracy.goldens.reference_model.init_hf_model`)
is model-agnostic. Callers needing model-specific golden workarounds (e.g. an
LLM-Compressor MXFP8 CPU-dequant bypass or a GPT-OSS chunked-attention patch)
supply a richer loader through the ``model_loader`` parameter of
``generate_three_way_reference_logits``.
"""

from vllm_neuron.accuracy.goldens.fp8_kv_golden import (
    hf_simulate_fp8_kv_cache,
    load_kv_scales,
)
from vllm_neuron.accuracy.logit_validation import (
    create_offline_vllm_generate_fn,
)
from vllm_neuron.accuracy.goldens.reference_logits import (
    generate_three_way_reference_logits,
    generate_reference_logits,
    generate_teacher_forced_logits,
    generate_teacher_forced_logits_fp8,
    get_cached_reference_goldens,
)
from vllm_neuron.accuracy.goldens.reference_model import init_hf_model

__all__ = [
    "generate_three_way_reference_logits",
    "create_offline_vllm_generate_fn",
    "generate_reference_logits",
    "generate_teacher_forced_logits",
    "get_cached_reference_goldens",
    "init_hf_model",
    "generate_teacher_forced_logits_fp8",
    "hf_simulate_fp8_kv_cache",
    "load_kv_scales",
]
