#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a dense Qwen3 compressed-tensors MXFP8 checkpoint from BF16 weights.

The output contract matches the Neuron dense/static-MX loader expectations:

- quantized ``.weight`` tensors are stored as native ``float8_e4m3fn``
- companion ``.weight_scale`` tensors are stored as ``uint8``
- each scale tensor has shape ``[out_features, in_features / 32]``
- ``lm_head`` remains unquantized by default

This is a weight-only conversion flow. No calibration dataset is required.

Requirements:
    pip install llmcompressor transformers safetensors

Examples:
    python examples/vllm_neuron/models/qwen3/build_mxfp8_checkpoint.py
    python examples/vllm_neuron/models/qwen3/build_mxfp8_checkpoint.py \
        --model-checkpoint /models/Qwen3-32B-bf16 \
        --output-dir /models/Qwen3-32B-mxfp8
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open

MXFP8_FORMAT = "mxfp8-quantized"
MX_GROUP_SIZE = 32
_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".weight_scale"
_DEFAULT_IGNORE_MODULES = ("lm_head",)
_DEFAULT_OUTPUT_SUFFIX = "-mxfp8"
_SAFE_WEIGHT_DTYPES = {"F8_E4M3"}
_SAFE_SCALE_DTYPE = "U8"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BF16 -> MXFP8 checkpoint conversion.

    Example:
        >>> args = parse_args()
        >>> isinstance(args.model_checkpoint, str)
        True
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Source BF16 model checkpoint or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for the compressed-tensors MXFP8 checkpoint. "
            "Defaults to <model-name>-mxfp8 in the current working directory."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Dtype used when loading the source BF16 checkpoint.",
    )
    parser.add_argument(
        "--ignore-modules",
        nargs="+",
        default=list(_DEFAULT_IGNORE_MODULES),
        help=(
            "Module names to leave unquantized. `lm_head` should stay BF16 so "
            "the output embedding path matches the existing dense Qwen3 flow."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to HuggingFace `from_pretrained`/`save_pretrained` APIs.",
    )
    return parser.parse_args()


def normalize_ignore_modules(ignore_modules: Sequence[str]) -> list[str]:
    """Return normalized module names for the llmcompressor ignore list.

    Example:
        >>> normalize_ignore_modules(["lm_head", "model.layers.0.mlp.down_proj"])
        ['lm_head', 'model.layers.0.mlp.down_proj']
    """

    normalized = [module.strip() for module in ignore_modules if module.strip()]
    if not normalized:
        raise ValueError("ignore_modules must contain at least one non-empty entry.")
    return normalized


def parse_torch_dtype(dtype_name: str) -> str | torch.dtype:
    """Map a CLI torch dtype value to the corresponding loader argument.

    Example:
        >>> parse_torch_dtype("bfloat16")
        torch.bfloat16
    """

    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def default_output_dir(model_checkpoint: str) -> Path:
    """Return the default MXFP8 output path for a source checkpoint.

    Example:
        >>> str(default_output_dir("Qwen/Qwen3-32B"))
        'Qwen3-32B-mxfp8'
    """

    model_name = model_checkpoint.rstrip("/").split("/")[-1]
    return Path(f"{model_name}{_DEFAULT_OUTPUT_SUFFIX}")


def build_recipe(ignore_modules: Sequence[str]) -> Any:
    """Build the llmcompressor MXFP8 quantization recipe.

    Example:
        >>> recipe = build_recipe(["lm_head"])  # doctest: +SKIP
    """

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "llmcompressor is required to build MXFP8 checkpoints. "
            "Install it with `pip install llmcompressor`."
        ) from exc

    return QuantizationModifier(
        targets="Linear",
        scheme="MXFP8",
        ignore=list(ignore_modules),
    )


def build_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    """Return the checkpoint tensor -> shard mapping for a safetensors checkpoint.

    Example:
        >>> build_weight_map(Path('/tmp/nonexistent'))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return json.load(f)["weight_map"]

    shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(shard_paths) != 1:
        raise FileNotFoundError(
            f"Expected a sharded safetensors index or a single shard in {checkpoint_dir}."
        )

    shard_path = shard_paths[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        return {key: shard_path.name for key in shard.keys()}


def _module_to_scale_key(module_name: str) -> str:
    module_name = (
        module_name[: -len(_WEIGHT_SUFFIX)]
        if module_name.endswith(_WEIGHT_SUFFIX)
        else module_name
    )
    return f"{module_name}{_SCALE_SUFFIX}"


def _get_slice_metadata(shard: Any, key: str) -> tuple[tuple[int, ...], str]:
    tensor_slice = shard.get_slice(key)
    return tuple(tensor_slice.get_shape()), str(tensor_slice.get_dtype())


def verify_mxfp8_checkpoint_contract(
    checkpoint_dir: str | Path,
    *,
    ignore_modules: Sequence[str] = _DEFAULT_IGNORE_MODULES,
) -> None:
    """Validate the saved checkpoint matches the dense Qwen3 MXFP8 loader contract.

    Example:
        >>> verify_mxfp8_checkpoint_contract('/tmp/nonexistent')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """

    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {checkpoint_path}.")

    with config_path.open() as f:
        config = json.load(f)

    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("config.json is missing a dict quantization_config block.")
    if quantization_config.get("quant_method") != "compressed-tensors":
        raise ValueError(
            "Expected quantization_config.quant_method='compressed-tensors'."
        )
    if quantization_config.get("format") != MXFP8_FORMAT:
        raise ValueError(
            f"Expected quantization_config.format={MXFP8_FORMAT!r}, got "
            f"{quantization_config.get('format')!r}."
        )

    weight_map = build_weight_map(checkpoint_path)
    scale_keys = sorted(key for key in weight_map if key.endswith(_SCALE_SUFFIX))
    if not scale_keys:
        raise ValueError("Checkpoint does not contain any .weight_scale tensors.")

    for ignored_module in normalize_ignore_modules(ignore_modules):
        ignored_scale_key = _module_to_scale_key(ignored_module)
        if ignored_scale_key in weight_map:
            raise ValueError(
                f"Ignore module {ignored_module!r} was quantized as {ignored_scale_key!r}."
            )

    shard_names = sorted(set(weight_map.values()))
    with ExitStack() as stack:
        shards = {
            shard_name: stack.enter_context(
                safe_open(
                    str(checkpoint_path / shard_name), framework="pt", device="cpu"
                )
            )
            for shard_name in shard_names
        }

        for scale_key in scale_keys:
            weight_key = f"{scale_key[: -len(_SCALE_SUFFIX)]}{_WEIGHT_SUFFIX}"
            if weight_key not in weight_map:
                raise ValueError(f"Missing weight tensor companion for {scale_key!r}.")

            scale_shape, scale_dtype = _get_slice_metadata(
                shards[weight_map[scale_key]], scale_key
            )
            weight_shape, weight_dtype = _get_slice_metadata(
                shards[weight_map[weight_key]], weight_key
            )

            if len(weight_shape) != 2:
                raise ValueError(
                    f"Expected 2D weight tensor for {weight_key!r}, got {weight_shape}."
                )
            if len(scale_shape) != 2:
                raise ValueError(
                    f"Expected 2D scale tensor for {scale_key!r}, got {scale_shape}."
                )
            if weight_dtype not in _SAFE_WEIGHT_DTYPES:
                raise ValueError(
                    f"Expected {weight_key!r} dtype in {_SAFE_WEIGHT_DTYPES}, got "
                    f"{weight_dtype!r}."
                )
            if scale_dtype != _SAFE_SCALE_DTYPE:
                raise ValueError(
                    f"Expected {scale_key!r} dtype {_SAFE_SCALE_DTYPE!r}, got "
                    f"{scale_dtype!r}."
                )
            if weight_shape[1] % MX_GROUP_SIZE != 0:
                raise ValueError(
                    f"Expected input dim for {weight_key!r} to be a multiple of "
                    f"{MX_GROUP_SIZE}, got {weight_shape[1]}."
                )

            expected_scale_shape = (weight_shape[0], weight_shape[1] // MX_GROUP_SIZE)
            if scale_shape != expected_scale_shape:
                raise ValueError(
                    f"Expected {scale_key!r} shape {expected_scale_shape}, got "
                    f"{scale_shape}."
                )


def build_mxfp8_checkpoint(
    model_checkpoint: str,
    output_dir: str | Path,
    *,
    torch_dtype: str | torch.dtype = "auto",
    ignore_modules: Sequence[str] = _DEFAULT_IGNORE_MODULES,
    trust_remote_code: bool = False,
) -> Path:
    """Convert a BF16 dense Qwen3 checkpoint into compressed-tensors MXFP8.

    Example:
        >>> build_mxfp8_checkpoint('Qwen/Qwen3-32B', '/tmp/qwen3-mxfp8')  # doctest: +SKIP
    """

    try:
        from llmcompressor import oneshot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "llmcompressor is required to build MXFP8 checkpoints. "
            "Install it with `pip install llmcompressor`."
        ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    normalized_ignore_modules = normalize_ignore_modules(ignore_modules)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading BF16 checkpoint from {model_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    print(
        "Quantizing all Linear modules to MXFP8; keeping these modules in BF16: "
        f"{', '.join(normalized_ignore_modules)}"
    )
    oneshot(
        model=model,
        recipe=build_recipe(normalized_ignore_modules),
    )

    print(f"Saving compressed checkpoint to {output_path}")
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    print("Verifying saved MXFP8 checkpoint contract")
    verify_mxfp8_checkpoint_contract(
        output_path,
        ignore_modules=normalized_ignore_modules,
    )
    print(f"Saved validated MXFP8 checkpoint to {output_path}")
    return output_path


def main() -> None:
    """CLI entry point for dense Qwen3 BF16 -> MXFP8 checkpoint generation.

    Example:
        >>> main()  # doctest: +SKIP
    """

    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.model_checkpoint)
    build_mxfp8_checkpoint(
        model_checkpoint=args.model_checkpoint,
        output_dir=output_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        ignore_modules=args.ignore_modules,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
