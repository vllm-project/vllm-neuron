# SPDX-License-Identifier: Apache-2.0
"""Tensor capture for accuracy debugging.

Architecture:
    ModelCapture registers forward hooks on the model and captures tensors
    into a side-channel registry.
Usage:
    from vllm_neuron.accuracy import ModelCapture

    # After compile — no wrapping needed
    compiled = torch.compile(model, backend=envs.get_compile_backend_name(), fullgraph=True)
    capture = ModelCapture(model, ["model.layers.0-31"], capture_dir="/tmp/caps",
                           subdirectory="target")
    capture.register_clear_hook(compiled)  # auto-clear before each forward

    # Bracket the forward:
    capture.setup_tensor_capture()
    output = compiled(**kwargs)
    capture.save_tensor_captures(positions=positions, is_prefill=True, req_ids=[...])
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm_neuron.accuracy.tensor_io import (
    CapturedForwardPass,
    ForwardPassMetadata,
    write as tensor_io_write,
)

logger = logging.getLogger(__name__)


def expand_patterns(patterns: List[str]) -> List[str]:
    """Expand capture patterns supporting ranges and regex.

    Supports:
        - Exact: "model.layers.0"
        - Range: "model.layers.0-31" expands to layers 0 through 31
        - Range with suffix: "model.layers.0-31.self_attn"
        - Regex: Patterns containing regex metacharacters (*, +, ?, etc.)
          are passed through for regex matching in _matches()

    Args:
        patterns: List of module path patterns

    Returns:
        Expanded list of patterns

    Example:
        >>> expand_patterns(["model.layers.0-2", "lm_head"])
        ['model.layers.0', 'model.layers.1', 'model.layers.2', 'lm_head']
    """
    expanded = []
    # Match range anywhere: prefix.N-M or prefix.N-M.suffix
    range_pattern = re.compile(r"^(.+)\.(\d+)-(\d+)(\..*)?$")

    for pattern in patterns:
        match = range_pattern.match(pattern)
        if match:
            prefix, start, end, suffix = match.groups()
            suffix = suffix or ""
            for i in range(int(start), int(end) + 1):
                expanded.append(f"{prefix}.{i}{suffix}")
        else:
            expanded.append(pattern)

    return expanded


def match_modules(model: nn.Module, patterns: List[str]) -> List[str]:
    """Match module names against expanded patterns using pre-compiled regex.

    Args:
        model: The model whose named_modules to search.
        patterns: Raw pattern strings (will be expanded via expand_patterns).

    Returns:
        List of matched module names in discovery order.
    """
    expanded = expand_patterns(patterns)
    compiled = []
    for p in expanded:
        try:
            compiled.append((p, re.compile(p)))
        except re.error:
            compiled.append((p, None))  # fallback to exact match

    matched = []
    seen: set = set()
    for name in dict(model.named_modules()):
        if not name or name in seen:
            continue
        for raw_pattern, regex in compiled:
            if regex is not None:
                if regex.fullmatch(name):
                    matched.append(name)
                    seen.add(name)
                    break
            else:
                if name == raw_pattern:
                    matched.append(name)
                    seen.add(name)
                    break
    return matched


def register_capture_hooks(
    model: nn.Module,
    module_names: List[str],
    registry: "TensorRegistry",
    name_prefix: str = "",
) -> List[Any]:
    """Register forward hooks on named modules to capture their outputs.

    Args:
        model: The model containing the modules.
        module_names: Exact module names to hook (from match_modules).
        registry: TensorRegistry to store captured tensors.
        name_prefix: Optional prefix for capture names.

    Returns:
        List of hook handles for later removal.
    """
    module_dict = dict(model.named_modules())
    hooks = []
    for name in module_names:
        module = module_dict.get(name)
        if module is None:
            logger.warning("Module not found for hook: %s", name)
            continue
        prefix = f"{name_prefix}." if name_prefix else ""
        capture_name = f"{prefix}{name}"

        def _make_hook(cap_name, reg):
            def hook(_mod, _inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    reg.register_module_tensor(cap_name, tensor)

            return hook

        h = module.register_forward_hook(_make_hook(capture_name, registry))
        hooks.append(h)
        logger.info("Registered capture hook for: %s", name)
    return hooks


class TensorRegistry:
    """Registry for captured tensors during torch.compile tracing.

    Two usage modes:

    1. Hook-based captures (ModelCapture): Each ModelCapture creates its own
       TensorRegistry instance. This supports multiple models with
       overlapping module names (e.g., target + draft in speculative decoding).

    2. Manual captures (capture_tensor()): Uses the global singleton via
       get_instance(). For multi-model scenarios, call reset_instance() before
       creating each ModelCapture to avoid stale captures.
    """

    _instance: Optional["TensorRegistry"] = None

    @classmethod
    def get_instance(cls) -> "TensorRegistry":
        """Get global singleton for manual capture_tensor() calls."""
        if cls._instance is None:
            cls._instance = TensorRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the global singleton.

        Call this before creating a new ModelCapture when using manual
        capture_tensor() calls in multi-model scenarios.
        """
        cls._instance = None

    def __init__(self):
        # Use lists instead of OrderedDicts to avoid torch._dynamo errors
        # from dict mutation tracking ("mapping proxy affected by dictionary
        # mutation") introduced in torch 2.9.  Dynamo handles list appends.
        self._module_items: List[Tuple[str, torch.Tensor]] = []
        self._manual_items: List[Tuple[str, torch.Tensor]] = []
        self._enabled = False

    def configure(self, enabled: bool = True):
        self._enabled = enabled

    def clear(self) -> None:
        """Clear all registered tensors for new forward pass."""
        self._module_items.clear()
        self._manual_items.clear()

    def register_module_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._enabled:
            self._module_items.append((name, tensor.clone().detach()))

    def register_manual_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._enabled:
            key = f"manual.{name}"
            existing = {n for n, _ in self._manual_items}
            if key in existing:
                i = 1
                while f"{key}_{i}" in existing:
                    i += 1
                key = f"{key}_{i}"
            self._manual_items.append((key, tensor.clone().detach()))

    def get_all_tensors(self) -> List[torch.Tensor]:
        """Get all tensors in order (module first, then manual)."""
        return [t for _, t in self._module_items] + [t for _, t in self._manual_items]

    def get_all_names(self) -> List[str]:
        """Get all tensor names in order."""
        return [n for n, _ in self._module_items] + [n for n, _ in self._manual_items]

    def get_all(self) -> List[Tuple[str, torch.Tensor]]:
        """Get all (name, tensor) pairs in order."""
        return list(self._module_items) + list(self._manual_items)

    @property
    def enabled(self) -> bool:
        return self._enabled


class CaptureWriter:
    """Writes captured tensors to disk.

    write() persists captured tensors to disk with this replica's local
    scheduler metadata (request IDs, positions, prefill/decode phase) in
    a 4-layer directory layout with per-bucket sequential counters. When
    no tensors are captured, only metadata JSON files are written
    (metadata-only mode for offline HF-to-Neuron conversion). For
    tensors captured after a cross-DP all-gather, the tensor contains
    tokens from all replicas but the metadata only describes this
    replica's portion. Cross-replica metadata can be assembled at read
    time by matching forward passes across dp{N}/ directories by
    dir_name.

    Disk layout::

        capture_dir/
          dp{dp_rank}/
            prefill_s{bucket_size}_{counter}/
              model.layers.0.input_layernorm/
                rank0.pt    # TP-local rank
              {dir_name}_meta.json
            decode_b{batch_size}_{counter}/
              ...
    """

    def __init__(
        self,
        capture_dir: str,
        dp_rank: int = 0,
        tp_rank: int = 0,
        capture_filter: Optional[set] = None,
    ):
        self.capture_dir = capture_dir
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        # Optional write-time filter. When set, only tensors whose names
        # are in this set are written to disk; others are silently skipped.
        # None means write all captured tensors.
        self._capture_filter = capture_filter
        self._counters: Dict[tuple, int] = {}
        self._enabled = False
        os.makedirs(self.capture_dir, exist_ok=True)

    def enable(self) -> None:
        """Enable writing captures to disk."""
        self._enabled = True

    def disable(self) -> None:
        """Disable writing captures to disk."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_dir_name(self, phase: str, size_label: str) -> str:
        key = (phase, size_label)
        count = self._counters.get(key, 0)
        return f"{phase}_{size_label}_{count}"

    def write(
        self,
        captures: Tuple[torch.Tensor, ...],
        capture_names: List[str],
        req_ids: List[str],
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> None:
        """Save captured tensors to disk for one forward pass.

        Args:
            captures: Tensor tuple from extract().
            capture_names: Module names corresponding to each tensor.
            req_ids: vLLM request IDs in this batch, from the Neuron scheduler.
            positions: Token positions tensor. Its shape determines the bucket
                label: s{len} for prefill, b{len} for decode.
            is_prefill: Whether this forward pass is prefill or decode.
        """
        phase = "prefill" if is_prefill else "decode"
        size_label = (
            f"s{positions.shape[0]}" if is_prefill else f"b{positions.shape[0]}"
        )

        dir_name = self._get_dir_name(phase, size_label)

        # Build tensors dict: {name: {rank: tensor}}, applying write-time filter
        tensors = {}
        for name, tensor in zip(capture_names, captures):
            if not isinstance(tensor, torch.Tensor):
                continue
            if self._capture_filter is not None and name not in self._capture_filter:
                continue
            tensors[name] = {self.tp_rank: tensor}

        metadata = ForwardPassMetadata(
            req_ids=req_ids,
            positions=positions.cpu().tolist(),
            dp_rank=self.dp_rank,
        )

        fwd = CapturedForwardPass(
            dir_name=dir_name,
            metadata=metadata,
            tensors=tensors,
        )

        capture_dir = os.path.join(self.capture_dir, f"dp{self.dp_rank}")
        tensor_io_write(capture_dir, [fwd])

        key = (phase, size_label)
        self._counters[key] = self._counters.get(key, 0) + 1

        logger.debug("Saved %d tensors to %s", len(captures), dir_name)


def capture_tensor(name: str, tensor: torch.Tensor) -> None:
    """Manually capture a tensor from within model code.

    Uses the global TensorRegistry singleton. No-op if no ModelCapture
    has activated its registry via setup_tensor_capture().

    Args:
        name: Identifier for the tensor (will be prefixed with "manual.")
        tensor: Tensor to capture
    """
    registry = TensorRegistry._instance
    if registry is not None:
        registry.register_manual_tensor(name, tensor)


class ModelCapture:
    """Tensor capture for any model (target or draft).

    Registers forward hooks on the model to capture module outputs into a
    side-channel registry.

    Supports hook-based captures (module outputs) and inline captures
    (intermediate tensors via capture_tensor() calls in model code).

    Args:
        model: The model (unwrapped, i.e. _orig_mod if compiled).
        modules: Module name patterns to capture.
        capture_dir: Root capture directory. Captures are written to
            {capture_dir}/{subdirectory}/.
        dp_rank: Data parallel rank.
        tp_rank: Tensor parallel rank.
        capture_filter: Optional set of tensor names to write. None means all.
        subdirectory: Subdirectory under capture_dir (e.g. "target", "draft").
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        modules: List[str],
        capture_dir: str,
        dp_rank: int = 0,
        tp_rank: int = 0,
        capture_filter: Optional[set] = None,
        subdirectory: str = "",
    ):
        self._registry = TensorRegistry()
        self._registry.configure(enabled=True)
        self._prev_instance: Optional[TensorRegistry] = None
        self._clear_hook_handle: Optional[Any] = None

        self._capture_names = match_modules(model, modules)
        self._hooks = register_capture_hooks(model, self._capture_names, self._registry)

        write_dir = (
            os.path.join(capture_dir, subdirectory) if subdirectory else capture_dir
        )
        self._writer = CaptureWriter(
            capture_dir=write_dir,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            capture_filter=capture_filter,
        )

        logger.info(
            "ModelCapture initialized: %d modules, capture_dir=%s",
            len(self._capture_names),
            write_dir,
        )

    def enable(self) -> None:
        """Enable writing captures to disk (call after warmup)."""
        self._writer.enable()

    def disable(self) -> None:
        """Disable writing captures to disk."""
        self._writer.disable()

    @property
    def enabled(self) -> bool:
        return self._writer.enabled

    def clear(self) -> None:
        """Clear the registry. Called before compiled forward to avoid Dynamo guard failures."""
        self._registry.clear()

    def register_clear_hook(self, compiled_model: nn.Module) -> None:
        """Register a pre-forward hook on the compiled model to clear the registry.

        A second call moves the hook: it removes the previous one first, so a model
        that is compiled again keeps exactly one clear hook.
        """
        if self._clear_hook_handle is not None:
            self._clear_hook_handle.remove()
        self._clear_hook_handle = compiled_model.register_forward_pre_hook(
            lambda m, args, kwargs: self._registry.clear(),
            with_kwargs=True,
        )

    def setup_tensor_capture(self) -> None:
        """Set this model's registry as active for inline capture_tensor() calls."""
        self._prev_instance = TensorRegistry._instance
        TensorRegistry._instance = self._registry

    def save_tensor_captures(
        self,
        positions: "torch.Tensor",
        is_prefill: bool,
        req_ids: Optional[List[str]] = None,
    ) -> None:
        """Write captured tensors to disk and restore previous registry."""
        TensorRegistry._instance = self._prev_instance
        self._prev_instance = None

        if not self._writer.enabled:
            return

        captures = self._registry.get_all_tensors()
        capture_names = self._registry.get_all_names()
        if not captures:
            return

        # If positions not provided externally, derive from captured tensors.
        if positions.numel() == 0 and captures:
            positions = torch.arange(captures[0].shape[0])

        self._writer.write(
            captures=captures,
            capture_names=capture_names,
            req_ids=req_ids or [],
            positions=positions,
            is_prefill=is_prefill,
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._clear_hook_handle is not None:
            self._clear_hook_handle.remove()
            self._clear_hook_handle = None
