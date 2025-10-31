# SPDX-License-Identifier: Apache-2.0
"""
Utilities to make an HF EAGLE draft NxDI-compatible by ensuring it has
`lm_head.weight` copied from the target and is stored as a single safetensors
file. Safe to call repeatedly; no-ops when already NxDI-ready.

Use as a library:
    from eagle_nxdi_util import fix_eagle_draft_for_nxdi, is_nxdi_ready
    result = fix_eagle_draft_for_nxdi("/path/to/Target-Llama", "/path/to/EAGLE-Draft")
    print(result)

Or as a script:
    python eagle_nxdi_util.py --target /path/to/Target-Llama --draft /path/to/EAGLE-Draft
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM_HEAD = "lm_head.weight"
SAFETENSORS = "model.safetensors"
SAFETENSORS_INDEX = "model.safetensors.index.json"
PT_BIN = "pytorch_model.bin"

# ----------------------------- helpers -----------------------------


def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def _find_lm_head_from_sharded_safetensors(
        target_dir: Path) -> Tuple[torch.Tensor, str]:
    """Read `lm_head.weight` directly from the shard that contains it."""
    index_path = target_dir / SAFETENSORS_INDEX
    if not _exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r") as f:
        idx = json.load(f)

    weight_map: Dict[str, str] = idx.get("weight_map", {})
    if LM_HEAD not in weight_map:
        raise KeyError(f"{LM_HEAD} not present in {index_path}")

    shard_name = weight_map[LM_HEAD]
    shard_path = target_dir / shard_name
    if not _exists(shard_path):
        raise FileNotFoundError(f"Shard missing: {shard_path}")

    with safe_open(str(shard_path), framework="pt") as f:
        if LM_HEAD not in f.keys():
            raise KeyError(f"{LM_HEAD} not present in shard {shard_path}")
        lm = f.get_tensor(LM_HEAD)

    return lm, f"{index_path.name} -> {shard_name}"


def _find_lm_head_from_single_safetensors(
        target_dir: Path) -> Tuple[torch.Tensor, str]:
    """Open model.safetensors and fetch lm_head.weight."""
    st_path = target_dir / SAFETENSORS
    if not _exists(st_path):
        raise FileNotFoundError(f"{st_path} not found")
    with safe_open(str(st_path), framework="pt") as f:
        if LM_HEAD not in f.keys():
            raise KeyError(f"{LM_HEAD} not present in {st_path}")
        lm = f.get_tensor(LM_HEAD)
    return lm, st_path.name


def _find_lm_head_from_pt_bin(target_dir: Path) -> Tuple[torch.Tensor, str]:
    """Load pytorch_model.bin (legacy) and fetch lm_head.weight."""
    bin_path = target_dir / PT_BIN
    if not _exists(bin_path):
        raise FileNotFoundError(f"{bin_path} not found")
    sd = torch.load(str(bin_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if LM_HEAD not in sd:
        raise KeyError(f"{LM_HEAD} not present in {bin_path}")
    return sd[LM_HEAD].cpu(), bin_path.name


def read_target_lm_head(target_dir: Path) -> Tuple[torch.Tensor, str]:
    """Try sharded safetensors -> single safetensors -> bin (in that order)."""
    if _exists(target_dir / SAFETENSORS_INDEX):
        return _find_lm_head_from_sharded_safetensors(target_dir)
    if _exists(target_dir / SAFETENSORS):
        return _find_lm_head_from_single_safetensors(target_dir)
    if _exists(target_dir / PT_BIN):
        return _find_lm_head_from_pt_bin(target_dir)
    raise FileNotFoundError(
        f"Could not find {SAFETENSORS_INDEX}, {SAFETENSORS}, or {PT_BIN} under {target_dir}"
    )


def draft_has_safetensors_with_lm_head(draft_dir: Path) -> bool:
    """Detect NxDI-ready draft: single safetensors file containing lm_head.weight."""
    st_path = draft_dir / SAFETENSORS
    if not _exists(st_path):
        return False
    try:
        with safe_open(str(st_path), framework="pt") as f:
            return LM_HEAD in f.keys()
    except Exception:
        return False


def ensure_draft_safetensors(draft_dir: Path) -> Path:
    """
    Ensure draft has model.safetensors:
      - If it exists: return the path.
      - If only pytorch_model.bin exists: convert to safetensors and return path.
    """
    st_path = draft_dir / SAFETENSORS
    if _exists(st_path):
        return st_path

    bin_path = draft_dir / PT_BIN
    if _exists(bin_path):
        sd = torch.load(str(bin_path), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        tensors = {k: v.cpu() for k, v in sd.items()}
        save_file(tensors, str(st_path))
        return st_path

    raise FileNotFoundError(
        f"Draft has neither {SAFETENSORS} nor {PT_BIN}: {draft_dir}")


def read_draft_tensors_and_dtype(
        draft_st_path: Path) -> Tuple[Dict[str, torch.Tensor], torch.dtype]:
    """Read all tensors from draft safetensors file; infer a representative dtype."""
    tensors: Dict[str, torch.Tensor] = {}
    rep_dtype = None
    with safe_open(str(draft_st_path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            tensors[k] = t
            if rep_dtype is None:
                rep_dtype = t.dtype
    if rep_dtype is None:
        rep_dtype = torch.float16
    return tensors, rep_dtype


def inject_lm_head_into_draft(
        draft_st_path: Path,
        lm_head: torch.Tensor) -> Tuple[Tuple[int, ...], torch.dtype]:
    """Insert/overwrite lm_head.weight into draft safetensors, casting to draft dtype."""
    tensors, draft_dtype = read_draft_tensors_and_dtype(draft_st_path)
    tensors[LM_HEAD] = lm_head.to(draft_dtype)
    save_file(tensors, str(draft_st_path))
    return tuple(tensors[LM_HEAD].shape), draft_dtype


# ----------------------------- public API -----------------------------


@dataclass
class FixResult:
    nxdi_ready_before: bool
    nxdi_ready_after: bool
    created_safetensors: bool
    injected_lm_head: bool
    target_source: str | None = None
    lm_head_shape: Tuple[int, ...] | None = None
    lm_head_dtype: torch.dtype | None = None

    def __str__(self) -> str:
        lines = [
            f"nxdi_ready_before={self.nxdi_ready_before}",
            f"created_safetensors={self.created_safetensors}",
            f"injected_lm_head={self.injected_lm_head}",
            f"nxdi_ready_after={self.nxdi_ready_after}",
        ]
        if self.target_source:
            lines.append(f"target_source={self.target_source}")
        if self.lm_head_shape:
            lines.append(f"lm_head_shape={self.lm_head_shape}")
        if self.lm_head_dtype:
            lines.append(f"lm_head_dtype={self.lm_head_dtype}")
        return "FixResult(" + ", ".join(lines) + ")"


def is_nxdi_ready(draft_dir: str | Path) -> bool:
    """Return True if draft has model.safetensors with lm_head.weight."""
    return draft_has_safetensors_with_lm_head(Path(draft_dir))


def fix_eagle_draft_for_nxdi(target_dir: str | Path,
                             draft_dir: str | Path) -> FixResult:
    """
    Idempotent fixer:
      - If draft already has model.safetensors with lm_head.weight -> no-op.
      - Else ensure draft has model.safetensors (convert bin if needed).
      - Copy lm_head.weight from target and inject into draft safetensors.
    """
    target_dir = Path(target_dir).expanduser().resolve()
    draft_dir = Path(draft_dir).expanduser().resolve()

    ready_before = draft_has_safetensors_with_lm_head(draft_dir)
    created_safetensors = False
    injected = False
    source = None
    shape = None
    dtype = None

    if not ready_before:
        # Ensure safetensors exists
        st_path = draft_dir / SAFETENSORS
        if not _exists(st_path):
            ensure_draft_safetensors(draft_dir)
            created_safetensors = True

        # Read target lm_head and inject
        lm_head, source = read_target_lm_head(target_dir)
        shape, dtype = inject_lm_head_into_draft(draft_dir / SAFETENSORS,
                                                 lm_head)
        injected = True

    ready_after = draft_has_safetensors_with_lm_head(draft_dir)

    return FixResult(
        nxdi_ready_before=ready_before,
        nxdi_ready_after=ready_after,
        created_safetensors=created_safetensors,
        injected_lm_head=injected,
        target_source=source,
        lm_head_shape=shape,
        lm_head_dtype=dtype,
    )


# ----------------------------- CLI -----------------------------


def _main():
    ap = argparse.ArgumentParser(
        description="Fix HF EAGLE draft for NxDI use (idempotent).")
    ap.add_argument("--target",
                    required=True,
                    help="Path to TARGET model folder")
    ap.add_argument("--draft",
                    required=True,
                    help="Path to EAGLE DRAFT model folder")
    args = ap.parse_args()

    res = fix_eagle_draft_for_nxdi(args.target, args.draft)
    print(res)
    if not res.nxdi_ready_after:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
