# SPDX-License-Identifier: Apache-2.0
"""Vendored rotational NKI top-k kernel (on-device constant builders).

This subpackage vendors the ``rotational_topk`` ``@nki.jit`` kernel and its
config factories from nkilib ``core/topk`` so vLLM-Neuron can use the on-device
constant-builder version of the kernel WITHOUT waiting for it to land in a
consumable dependency image. (The on-device version compiles on the current
image's neuronx-cc; the nkilib package version that ships it has not yet
propagated through the SDK/alpha-DLC pipeline into our base image.)

Provenance: copied from KaenaNeuronKernelLibrary commit 88ffd98c (which builds
on CR-284250291 "refactor(topk): Replace shared_constant cache with on-device
constant builders"):
  - rotational_topk.py        (only the @nki.jit rotational_topk kernel; the
                               upstream host topk() dispatcher is NOT vendored)
  - rotational_topk_utils.py  (RotationalTopkConfig + create_*_config + the
                               on-device build_rotation_matrix/build_stage_offsets)
  - cascaded_max_utils.py     (predicated_folded_load/unfolded_store, incl. the
                               hardware-DGE load from the same CR)

Stable leaf utils (kernel_assert, kernel_helpers, logging) are imported from
nkilib directly rather than vendored: they are byte-identical between the
image-baseline nkilib and the vendored source for the symbols used here.

SYNC: when the on-device rotational kernel lands in the consumed nkilib image,
this subpackage should be deleted and ``functional/topk.py`` reverted to import
``rotational_topk`` from ``nkilib.core.topk.rotational_topk``.
"""

from .rotational_topk import rotational_topk
from .rotational_topk_utils import (
    create_rotational_topk_config,
    create_topk_config,
)

__all__ = [
    "rotational_topk",
    "create_rotational_topk_config",
    "create_topk_config",
]
