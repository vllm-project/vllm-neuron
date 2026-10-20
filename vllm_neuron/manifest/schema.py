# SPDX-License-Identifier: Apache-2.0
"""Versioned schema for the compile-artifact manifest.

This module is the single source of truth for the manifest's shape. Each engine
emits one manifest; it lists the graphs the engine compiled (HLO / NEFF) next to
the config that produced them, so a reader can consume one file instead of
crawling the raw compile-cache directories.

The schema is a versioned contract: ``manifest_version`` pins the version and
:func:`parse_manifest_dict` validates and dispatches on it.

Producers build a :class:`Manifest` and serialize it. Consumers parse with
:func:`parse_manifest_dict` (validates and dispatches on version).

Some fields hold data this module does not define (the ``environment`` version
map and the ``neuron_config`` / ``vllm_config`` settings). They are kept as open
``dict`` / ``Any`` so a change to their contents does not force a schema version
bump.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = "1"


class Topology(BaseModel):
    """The parallelism arrangement (how the model is sharded across ranks).

    Every field is optional and defaults to ``None`` ("not set").
    """

    model_config = ConfigDict(extra="forbid")

    world_size: Optional[int] = Field(
        default=None, description="Total ranks in the job (TP*PP*DP)"
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None, description="Tensor-parallel degree"
    )
    pipeline_parallel_size: Optional[int] = Field(
        default=None, description="Pipeline-parallel degree"
    )
    data_parallel_size: Optional[int] = Field(
        default=None, description="Data-parallel degree (engine-level)"
    )
    attention_dp_size: Optional[int] = Field(
        default=None, description="Attention data-parallel degree (Neuron-specific)"
    )
    ep_degree: Optional[int] = Field(default=None, description="Expert-parallel degree")
    decode_context_parallel_size: Optional[int] = Field(
        default=None, description="Decode context-parallel degree"
    )
    has_collectives: Optional[bool] = Field(
        default=None,
        description="True if any parallel degree > 1 (cross-rank communication); "
        "None when no degree is known",
    )


class RunConfig(BaseModel):
    """How this run was compiled/served: shape, target, parallelism, raw configs.

    Captured from the resolved engine config. Every field is optional.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(default=None, description="Model name/path served")
    target: Optional[str] = Field(
        default=None, description="Compiler target, e.g. trn2"
    )
    dtype: Optional[str] = Field(default=None, description="Model dtype")
    max_num_seqs: Optional[int] = Field(
        default=None, description="Max concurrent sequences"
    )
    max_model_len: Optional[int] = Field(
        default=None, description="Max sequence length"
    )
    topology: Topology = Field(
        default_factory=Topology, description="Parallelism arrangement"
    )
    neuron_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Full resolved vLLM-Neuron NeuronConfig",
    )
    vllm_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Full resolved vLLM VllmConfig, JSON-coerced",
    )


class GraphEntry(BaseModel):
    """One compiled graph (HLO / NEFF) the server produced or reused.

    Native populates fewer fields than XLA, so gate on the field, not the route.
    """

    model_config = ConfigDict(extra="forbid")

    graph_hash: str = Field(
        ...,
        description="Graph identity. XLA: the compile-cache dir name. "
        "Native: the ledger's compile key, a bare 64-hex sha256 of the "
        "serialized MLIR. It names no directory and is not the 32-hex "
        "persistent key in neff_path; a lookup table in the NEFF cache dir "
        "maps one to the other. ':'-joined only under the legacy "
        "TORCH_NEURONX_NEFF_KEY_FROM_HLO=0 opt-out",
    )
    hlo_path: Optional[str] = Field(
        default=None,
        description="Archive-relative path to the HLO; None on native, which "
        "writes no graph.hlo into the compile cache",
    )
    neff_path: Optional[str] = Field(
        default=None,
        description="Archive-relative NEFF path; None if no NEFF was produced",
    )
    input_snapshot_paths: list[str] = Field(
        default_factory=list,
        description="Archive-relative paths to this graph's input snapshots "
        "(a graph maps to 0..N snapshots); reserved, currently empty",
    )
    output_snapshot_paths: list[str] = Field(
        default_factory=list,
        description="Archive-relative paths to this graph's output snapshots "
        "(a graph maps to 0..N snapshots); reserved, currently empty",
    )
    compiler_flags: Optional[str] = Field(
        default=None,
        description="neuronx-cc flags, machine-specific paths removed",
    )
    input_shapes: list[str] = Field(
        default_factory=list, description="Ordered input shape strings"
    )
    output_shapes: list[str] = Field(
        default_factory=list,
        description="Ordered output shape strings, one per graph output",
    )
    created_timestamp: Optional[str] = Field(
        default=None, description="ISO 8601 compile timestamp"
    )
    neff_size: Optional[int] = Field(
        default=None, ge=0, description="Compiled NEFF size in bytes"
    )
    ranks: list[int] = Field(
        default_factory=list,
        description="Global ranks (0..world_size-1) that traced this graph",
    )


class ManifestV1(BaseModel):
    """Per-server compile-artifact manifest, schema version 1."""

    # validate_default so the pinned version below is checked against its
    # Literal at construction rather than silently trusted (pydantic skips
    # default validation otherwise).
    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Each versioned model pins its OWN literal version.
    manifest_version: Literal["1"] = Field(
        default="1", description="Manifest schema version"
    )
    graphs: list[GraphEntry] = Field(
        default_factory=list, description="Graphs compiled or reused by the server"
    )
    config: Optional[RunConfig] = Field(
        default=None,
        description="How the run was compiled/served (target, shape, parallelism, "
        "raw configs), from the resolved engine config; None when no capture",
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Installed package versions",
    )


# Alias to the current schema-version model that producers and consumers use.
Manifest = ManifestV1


def parse_manifest_dict(data: dict[str, Any]) -> Manifest:
    """Validate a manifest dict and return the model for its version.

    Raises ``ValueError`` for an unsupported version and pydantic
    ``ValidationError`` for a payload that doesn't match the versioned schema.

    Example:
        >>> parse_manifest_dict(json.load(open("manifest.json")))  # doctest: +SKIP
        ManifestV1(manifest_version='1', ...)
    """
    version = data.get("manifest_version", MANIFEST_SCHEMA_VERSION)
    if version == "1":
        return ManifestV1(**data)
    raise ValueError(f"Unsupported manifest schema version: {version!r}")
