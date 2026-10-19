# SPDX-License-Identifier: Apache-2.0
"""Standard interface for reading and writing captured tensors.

Disk layout::

    capture_dir/
      dp{dp_rank}/
        prefill_s2048_0/
          model.layers.0.input_layernorm/
            rank0.pt
          prefill_s2048_0_meta.json
        decode_b2_0/
          ...

Usage::

    from vllm_neuron.accuracy.tensor_io import read, write

    # Write
    write("/tmp/captures/dp0", [fwd])

    # Read all replicas
    all_captures = read("/tmp/captures")
    # all_captures: [CapturedForwardPass, ...]  flat, dp_rank in metadata
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassMetadata:
    """Scheduler metadata for one forward pass on a single DP replica.

    All fields describe this replica's local state only. For tensors
    captured after a cross-DP all-gather (e.g., MoE input in cross-DP
    EP), the tensor contains tokens from all replicas but this metadata
    only describes the local replica's portion. Cross-replica metadata
    can be assembled at read time by matching forward passes across
    dp{N}/ directories by dir_name.

    Attributes:
        req_ids: vLLM request IDs for real requests on this replica.
        positions: Token positions on this replica.
        dp_rank: DP replica index (0 for dp_size=1).
    """

    req_ids: List[str]
    positions: List[int]
    dp_rank: int = 0


@dataclass
class CapturedForwardPass:
    """Standard data contract for one forward pass of captured tensors.

    Attributes:
        dir_name: Directory name, e.g. "prefill_s2048_0" or "decode_b2_3".
        metadata: Scheduler metadata with req_ids, positions, and dp_rank.
        tensors: Captured tensors keyed by module name then TP-local rank.
            {name: {rank: tensor}}
    """

    dir_name: str
    metadata: ForwardPassMetadata
    tensors: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)

    @property
    def is_prefill(self) -> bool:
        return self.dir_name.startswith("prefill_s")


def _read_replica(capture_dir: str) -> List[CapturedForwardPass]:
    """Load captured tensors from one replica's directory.

    Reads all forward-pass directories (prefill_s*/decode_b*), their
    metadata JSON, and per-name per-rank tensor files.

    Args:
        capture_dir: Replica capture directory (e.g., capture_dir/dp0).

    Returns:
        List of CapturedForwardPass sorted by directory name.
    """
    if not os.path.exists(capture_dir):
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    forward_dirs = sorted(
        d
        for d in os.listdir(capture_dir)
        if os.path.isdir(os.path.join(capture_dir, d))
        and (d.startswith("prefill_s") or d.startswith("decode_b"))
    )

    results: List[CapturedForwardPass] = []

    for dir_name in forward_dirs:
        forward_path = os.path.join(capture_dir, dir_name)

        meta_path = os.path.join(forward_path, f"{dir_name}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file missing for {dir_name}: expected {meta_path}"
            )
        with open(meta_path) as f:
            raw_meta = json.load(f)
        metadata = ForwardPassMetadata(
            req_ids=raw_meta["req_ids"],
            positions=raw_meta["positions"],
            dp_rank=raw_meta.get("dp_rank", 0),
        )

        tensors: Dict[str, Dict[int, torch.Tensor]] = {}
        for entry in sorted(os.listdir(forward_path)):
            module_path = os.path.join(forward_path, entry)
            if not os.path.isdir(module_path):
                continue
            rank_tensors: Dict[int, torch.Tensor] = {}
            for pt_file in sorted(os.listdir(module_path)):
                if not pt_file.endswith(".pt"):
                    continue
                rank_str = pt_file.replace("rank", "").replace(".pt", "")
                try:
                    rank = int(rank_str)
                except ValueError:
                    continue
                pt_path = os.path.join(module_path, pt_file)
                rank_tensors[rank] = torch.load(pt_path, weights_only=False)
            if rank_tensors:
                tensors[entry] = rank_tensors

        results.append(CapturedForwardPass(dir_name, metadata, tensors))

    logger.info("Read %d forward passes from %s", len(results), capture_dir)
    return results


def read(capture_dir: str) -> List[CapturedForwardPass]:
    """Load captured tensors from a capture directory.

    Scans for dp{N}/ subdirectories and reads all replicas.

    Args:
        capture_dir: Root capture directory containing dp{N}/ subdirs.

    Returns:
        Flat list of all CapturedForwardPass across all replicas,
        ordered by dp_rank then by dir_name within each replica.

    Raises:
        FileNotFoundError: If capture_dir doesn't exist or has no dp{N}/ subdirs.
    """
    if not os.path.exists(capture_dir):
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    dp_dirs = sorted(
        d
        for d in os.listdir(capture_dir)
        if os.path.isdir(os.path.join(capture_dir, d)) and d.startswith("dp")
    )

    if not dp_dirs:
        raise FileNotFoundError(f"No dp{{N}}/ subdirectories found in {capture_dir}")

    result = []
    for dp_dir in dp_dirs:
        result.extend(_read_replica(os.path.join(capture_dir, dp_dir)))

    logger.info("Read %d forward passes from %s", len(result), capture_dir)
    return result


def write(
    capture_dir: str,
    forward_passes: List[CapturedForwardPass],
) -> None:
    """Write captured tensors and metadata to disk.

    Args:
        capture_dir: Replica capture directory (e.g., capture_dir/dp0).
        forward_passes: List of forward passes to write.
    """
    os.makedirs(capture_dir, exist_ok=True)

    for fwd in forward_passes:
        forward_dir = os.path.join(capture_dir, fwd.dir_name)
        os.makedirs(forward_dir, exist_ok=True)

        for tensor_name, rank_tensors in fwd.tensors.items():
            tensor_dir = os.path.join(forward_dir, tensor_name)
            os.makedirs(tensor_dir, exist_ok=True)
            for r, tensor in rank_tensors.items():
                torch.save(
                    tensor.clone().detach().cpu(),
                    os.path.join(tensor_dir, f"rank{r}.pt"),
                )

        meta_path = os.path.join(forward_dir, f"{fwd.dir_name}_meta.json")
        meta = {
            "req_ids": fwd.metadata.req_ids,
            "positions": fwd.metadata.positions,
            "dp_rank": fwd.metadata.dp_rank,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logger.debug("Wrote %d forward passes to %s", len(forward_passes), capture_dir)
