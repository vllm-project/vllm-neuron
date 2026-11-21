# SPDX-License-Identifier: Apache-2.0
"""Keep the signed dtype of an unsigned NKI kernel output on the XLA route.

libtorch-neuronx-lite 2.13.0.1.0.3888 (ns-1117) re-types each output of an NKI
kernel call to the dtype that the kernel declares. torch_xla reports an unsigned
HLO result as the signed torch dtype (U32 -> int32), so the XLA dispatch of the
kernel HOP calls ``out.view(torch.uint32)`` (``nki_hop._as_declared_dtype``).
On an XLA tensor, ``Tensor.view(dtype)`` returns a plain ``TensorImpl`` with the
XLA dispatch key, not an XLA tensor. Every later torch op on it fails in the
FX-to-HLO conversion:

    RuntimeError: Expected XLA tensor. Got: XLAUInt32Type
    While executing %reshape : call_method[target=reshape](args = (%indices, ...))

The rotational and GpSIMD top-k kernels return uint32 indices, and
``functional/topk.py`` reshapes and casts them, so every sampling graph with
on-device top-k fails to capture. The earlier Lite (2.13.0.1.0.1559, vLLM 0.27.1
image) kept the signed dtype. The native route, the default, does not hit the
re-typing; only VLLM_NEURON_BACKEND=vllm_neuron (the XLA route) needs this patch.

The patch keeps the signed dtype for an XLA tensor whose declared dtype is an
unsigned integer of the same width. The bits are the same. A top-k index is
below 2**31, so the value is the same too. Other outputs, and every output off
the XLA route, keep the Lite behavior.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_UNSIGNED = tuple(
    d for d in (getattr(torch, n, None) for n in ("uint16", "uint32", "uint64")) if d
)

_applied = False


def apply_nki_xla_unsigned_output_patch() -> None:
    """Wrap ``nki_hop._as_declared_dtype`` so an XLA output keeps its signed dtype.

    Idempotent. A Lite without the re-typing (no ``_as_declared_dtype``) is left
    as it is.
    """
    global _applied
    if _applied:
        return
    _applied = True
    try:
        from libtorch_neuronx_lite.nki import nki_hop
    except ImportError:
        return
    original = getattr(nki_hop, "_as_declared_dtype", None)
    if original is None:
        return

    def _keep_signed_on_xla(out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if (
            out.device.type == "xla"
            and dtype in _UNSIGNED
            and not out.dtype.is_floating_point
            and out.dtype.itemsize == dtype.itemsize
        ):
            return out
        return original(out, dtype)

    nki_hop._as_declared_dtype = _keep_signed_on_xla
    logger.info(
        "NKI kernel outputs keep the signed dtype on the XLA route "
        "(nki_xla_unsigned_output_patch)."
    )
