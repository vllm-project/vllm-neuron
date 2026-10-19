# SPDX-License-Identifier: Apache-2.0
"""Reference-logit generation for accuracy validation.

Generate reference logits / goldens from a loaded HuggingFace model. The model
*loading* concern lives in
:mod:`vllm_neuron.accuracy.goldens.reference_model` (``init_hf_model``); the FP8
KV-cache *simulation mechanism* lives in
:mod:`vllm_neuron.accuracy.goldens.fp8_kv_golden`. This module composes both to
produce the fp32 + teacher-forced target-dtype logits used by the accuracy
debugger prompt plugins and the full-model logit tests. The vLLM ``generate_fn``
adapter that produces the *target* logits to compare against lives in
:mod:`vllm_neuron.accuracy.logit_validation`.
"""

import gc
from typing import Any, Callable, Optional

import torch

from vllm_neuron.accuracy.goldens.fp8_kv_golden import (
    hf_simulate_fp8_kv_cache,
    load_kv_scales,
)
from vllm_neuron.accuracy.goldens.reference_model import init_hf_model


def generate_reference_logits(
    model: Any, input_ids: torch.Tensor, output_length: int
) -> torch.Tensor:
    """Autoregressively generate ``output_length`` reference logits from *model*.

    Prefills the prompt once, then decodes one greedily-sampled token at a time
    (argmax), reusing the KV cache. Returns the per-step next-token logits
    stacked as ``(output_length, batch, vocab)``.

    Args:
        model: A HuggingFace causal-LM (e.g. from :func:`init_hf_model`).
        input_ids: Prompt token ids, shape ``(batch, prompt_len)``.
        output_length: Number of tokens to generate.
    """
    all_logits = []
    past_key_values = None
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        # Prefill: process entire prompt, get KV cache
        cache_position = torch.arange(prompt_len, device=input_ids.device)
        outputs = model(
            input_ids,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        all_logits.append(next_token_logits)
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Decode: one token at a time, reusing KV cache
        for step in range(output_length - 1):
            pos = prompt_len + step
            cache_position = torch.tensor([pos], device=input_ids.device)
            outputs = model(
                next_tokens,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            all_logits.append(next_token_logits)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    return torch.stack(all_logits, dim=0)


def generate_teacher_forced_logits(
    model: Any, input_ids: torch.Tensor, teacher_sequence: torch.Tensor
) -> torch.Tensor:
    """Generate logits with teacher forcing along *teacher_sequence*.

    Prefills the prompt, then feeds the fixed *teacher_sequence* tokens (rather
    than the model's own argmax) at each decode step, reusing the KV cache. This
    isolates per-token numerical differences from divergent token selection when
    comparing against a target model. Returns FP32 per-step logits stacked as
    ``(len(teacher_sequence), batch, vocab)``.

    Args:
        model: A HuggingFace causal-LM (e.g. from :func:`init_hf_model`).
        input_ids: Prompt token ids, shape ``(batch, prompt_len)``.
        teacher_sequence: Tokens to force at each decode step.
    """
    all_logits = []
    past_key_values = None
    seq_len = teacher_sequence.shape[0]
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        # Prefill: process entire prompt, get KV cache
        cache_position = torch.arange(prompt_len, device=input_ids.device)
        outputs = model(
            input_ids,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        all_logits.append(next_token_logits.float())

        # Decode: one teacher-forced token at a time, reusing KV cache
        for step in range(1, seq_len):
            next_tokens = teacher_sequence[step - 1].unsqueeze(1)
            pos = prompt_len + step - 1
            cache_position = torch.tensor([pos], device=input_ids.device)
            outputs = model(
                next_tokens,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            all_logits.append(next_token_logits.float())

    return torch.stack(all_logits, dim=0)


def generate_teacher_forced_logits_fp8(
    model: Any, input_ids: torch.Tensor, teacher_sequence: torch.Tensor
) -> torch.Tensor:
    """Teacher-forced logit generation under an FP8-simulated KV cache.

    The FP8 counterpart of :func:`generate_teacher_forced_logits`: it forces the
    fixed ``teacher_sequence`` at each step and reuses the KV cache, but uses an
    explicit ``transformers.DynamicCache`` so that
    :func:`~vllm_neuron.accuracy.goldens.fp8_kv_golden.hf_simulate_fp8_kv_cache`
    (which patches ``DynamicCache.update``) can round-trip the cached K/V through
    FP8. **Must** be called inside that context manager; otherwise it produces a
    plain (non-FP8) teacher-forced pass.

    Args:
        model: A HuggingFace causal-LM (e.g. from :func:`init_hf_model`).
        input_ids: Prompt token ids, shape ``(batch, prompt_len)``.
        teacher_sequence: Tokens to force at each decode step.

    Returns:
        FP32 per-step next-token logits stacked as
        ``(len(teacher_sequence), batch, vocab)``.
    """
    from transformers import DynamicCache

    all_logits = []
    current_input_ids = input_ids.clone()
    past_key_values = DynamicCache()

    with torch.inference_mode():
        for step in range(teacher_sequence.shape[0]):
            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            all_logits.append(next_token_logits.float())
            current_input_ids = teacher_sequence[step].unsqueeze(1)

    return torch.stack(all_logits, dim=0)


def generate_three_way_reference_logits(
    model_checkpoint: str,
    target_dtype: torch.dtype,
    output_length: int,
    prompts: list[str],
    tokenizer: Any,
    kv_cache_dtype: str = "auto",
    quant_scale_paths: Optional[dict] = None,
    config: Any = None,
    model_loader: Callable[..., Any] = init_hf_model,
) -> dict:
    """Compute the two HuggingFace reference logit sets for logit validation.

    Produces both halves of a three-way comparison per prompt: an FP32 baseline
    (loaded and run autoregressively via :func:`generate_reference_logits`) and a
    target-dtype pass that is teacher-forced along the FP32 greedy sequence
    (:func:`generate_teacher_forced_logits`, or the FP8 variant under
    :func:`~vllm_neuron.accuracy.goldens.fp8_kv_golden.hf_simulate_fp8_kv_cache`
    when ``kv_cache_dtype="fp8"``). The FP32 model is loaded and freed before the
    target-dtype model to bound peak CPU memory. These tensors are what the
    ``logit_validation`` / ``multi_prompt_logit_validation`` API consumes as
    ``baseline`` and ``expected`` logits.

    Args:
        model_checkpoint: HuggingFace model id or local path, passed to
            ``model_loader``.
        target_dtype: Target/device dtype whose numerical behavior is being
            validated (e.g. ``torch.bfloat16``); the second pass runs in it.
        output_length: Number of tokens to generate per prompt.
        prompts: List of prompt strings.
        tokenizer: HuggingFace tokenizer used to encode the prompts.
        kv_cache_dtype: ``"auto"`` (default) or ``"fp8"``. ``"fp8"`` runs the
            target-dtype pass under the FP8 KV-cache simulation.
        quant_scale_paths: Optional per-layer K/V scale key templates for the
            FP8 path (see :func:`~vllm_neuron.accuracy.goldens.fp8_kv_golden.load_kv_scales`);
            only used when ``kv_cache_dtype="fp8"``.
        config: Optional ``PretrainedConfig`` override forwarded to the loader
            (e.g. to truncate layers via ``num_hidden_layers``).
        model_loader: Callable ``(model_checkpoint, dtype, config=...) -> nn.Module``
            used to load the reference model. Defaults to :func:`init_hf_model`
            (a clean, override-free HF load). Callers needing model-specific
            workarounds (MXFP8 CPU-dequant, GPT-OSS chunked attention) pass a
            richer loader that wraps :func:`init_hf_model` with those overrides.

    Returns:
        Dict with three parallel lists (one entry per prompt):

        - ``input_ids``: tokenized prompts (list form).
        - ``fp32_logits``: FP32 baseline logits, ``(output_length, batch, vocab)``.
        - ``dtype_logits``: teacher-forced target-dtype logits, same shape.
    """
    use_fp8_kv = kv_cache_dtype == "fp8"

    input_ids_list = []
    fp32_logits_list = []
    dtype_logits_list = []
    teacher_seqs = []

    print("[Goldens] Computing FP32 baseline logits...")
    fp32_model = model_loader(model_checkpoint, torch.float32, config=config)
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        input_ids_list.append(input_ids.tolist())
        print(
            f"[Goldens] Generating FP32 reference for prompt {idx + 1}/{len(prompts)}"
        )
        print(f"[Goldens]   Prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")
        print(f"[Goldens]   Input tokens: {input_ids[0].tolist()[:10]}...")

        fp32_logits = generate_reference_logits(fp32_model, input_ids, output_length)
        fp32_logits_list.append(fp32_logits)
        teacher_seq = fp32_logits.argmax(dim=2)
        teacher_seqs.append(teacher_seq)

        # Debug: show what was generated
        generated_tokens = teacher_seq.squeeze().tolist()
        if not isinstance(generated_tokens, list):
            generated_tokens = [generated_tokens]
        print(f"[Goldens]   Generated tokens: {generated_tokens[:20]}")
        generated_text = tokenizer.decode(generated_tokens)
        print(f"[Goldens]   Generated text: '{generated_text}'")

    del fp32_model
    gc.collect()

    print("[Goldens] Computing dtype baseline logits...")
    dtype_model = model_loader(model_checkpoint, target_dtype, config=config)
    if use_fp8_kv:
        kv_scales = load_kv_scales(
            model_checkpoint,
            dtype_model.config.num_hidden_layers,
            quant_scale_paths=quant_scale_paths,
        )
        print(
            f"[Goldens] FP8 KV cache enabled, loaded scales for {len(kv_scales)} layers"
        )

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        if use_fp8_kv:
            with hf_simulate_fp8_kv_cache(kv_scales):
                dtype_logits = generate_teacher_forced_logits_fp8(
                    dtype_model, input_ids, teacher_seqs[i]
                )
        else:
            dtype_logits = generate_teacher_forced_logits(
                dtype_model, input_ids, teacher_seqs[i]
            )
        dtype_logits_list.append(dtype_logits)
    del dtype_model
    gc.collect()

    return {
        "input_ids": input_ids_list,
        "fp32_logits": fp32_logits_list,
        "dtype_logits": dtype_logits_list,
    }


def get_cached_reference_goldens(
    model_id: str,
    model_checkpoint: str,
    model_config: Any,
    tokenizer: Any,
    prompts: list[str],
    output_length: int,
    compute_fn: Optional[Callable[..., dict]] = None,
    kv_cache_dtype: str = "auto",
    quant_scale_paths: Optional[dict] = None,
    golden_variant: Optional[str] = None,
    model_loader: Callable[..., Any] = init_hf_model,
) -> dict:
    """Get or compute reference goldens through the disk→S3 golden cache.

    This is the canonical "assembler" that glues the golden producer
    (:func:`generate_three_way_reference_logits`) to the two-tier cache
    (:func:`vllm_neuron.utils.golden_cache.get_or_compute_goldens`). It owns the
    ``key_config`` cache-key layout so every consumer hashes goldens the same
    way — divergent keys would silently miss each other's cache. On a cache hit
    it returns the stored tensors; on a miss it computes them once and backfills.

    Args:
        model_id: Model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct"). Used only
            to build the cache key (slashes are replaced with underscores).
        model_checkpoint: Path to the model checkpoint (passed to ``compute_fn``).
        model_config: HuggingFace model config; ``torch_dtype`` selects the
            target dtype (defaults to bfloat16).
        tokenizer: HuggingFace tokenizer.
        prompts: List of prompt strings.
        output_length: Number of tokens to generate.
        compute_fn: Optional custom golden computation with signature
            ``(model_checkpoint, dtype, output_length, prompts, tokenizer) -> dict``.
            If ``None`` (default) :func:`generate_three_way_reference_logits`
            runs, threading ``model_loader``/``kv_cache_dtype``/``quant_scale_paths``.
        kv_cache_dtype: KV cache dtype ("auto" or "fp8"). Included in the cache
            key only when != "auto", so the default key layout stays stable.
        quant_scale_paths: Optional FP8 quantization scale paths; required when
            ``kv_cache_dtype="fp8"``.
        golden_variant: Optional discriminator that namespaces the cache key so
            goldens with incompatible tensor semantics do not collide (e.g. an
            FP8 GPU e4m3 golden whose ``dtype_logits`` is a raw dequant rather
            than a genuine bf16 baseline). Included in the key only when set, so
            the default bf16-baseline key layout stays stable.
        model_loader: Callable used to load the reference model, forwarded to the
            default ``generate_three_way_reference_logits``. Defaults to the clean,
            override-free :func:`init_hf_model`; a caller can supply an
            override-aware loader for model-specific workarounds.

    Returns:
        Dict with keys: ``input_ids``, ``fp32_logits``, ``dtype_logits``.
    """
    import transformers

    from vllm_neuron.utils.golden_cache import get_or_compute_goldens

    resolved_dtype = model_config.torch_dtype or torch.bfloat16

    # Build the cache key. The field set and ordering are a contract shared by
    # every consumer — changing it invalidates existing cached goldens.
    key_config = {
        "model": model_id.replace("/", "_"),
        "dtype": str(resolved_dtype).split(".")[-1],
        "output_length": output_length,
        "prompts": prompts,
        # Every input that affects the computed goldens is part of this key,
        # and the prompts are fixed text with no runtime RNG, so a cached
        # payload stays valid across torch versions. Opting out of
        # torch-version keying lets previously cached goldens keep resolving
        # instead of forcing a multi-hour CPU recompute of the reference
        # model on every torch uplift.
        "torch_independent": True,
    }

    # Include kv_cache_dtype only when non-default, so the default key is stable.
    if kv_cache_dtype != "auto":
        key_config["kv_cache_dtype"] = kv_cache_dtype

    # Namespace incompatible golden formats onto distinct keys. Included only
    # when set, so the default bf16-baseline key is unaffected (see golden_variant).
    if golden_variant:
        key_config["golden_variant"] = golden_variant

    def wrapped_compute_fn():
        if compute_fn is None:
            return generate_three_way_reference_logits(
                model_checkpoint,
                resolved_dtype,
                output_length,
                prompts,
                tokenizer,
                kv_cache_dtype=kv_cache_dtype,
                quant_scale_paths=quant_scale_paths,
                config=model_config,
                model_loader=model_loader,
            )
        # A caller-supplied compute_fn takes only the core args (no FP8 params).
        return compute_fn(
            model_checkpoint, resolved_dtype, output_length, prompts, tokenizer
        )

    return get_or_compute_goldens(
        key_config,
        wrapped_compute_fn,
        metadata={"framework_version": f"transformers-{transformers.__version__}"},
    )
