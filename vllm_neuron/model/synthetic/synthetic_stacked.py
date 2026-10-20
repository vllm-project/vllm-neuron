# SPDX-License-Identifier: Apache-2.0
"""SyntheticStackedNeuronModel — synthetic model that opts into the GK stacked KV
cache layout, for testing the DI KV-cache adaptor (non-GK prefill -> GK decode).

It subclasses :class:`SyntheticNeuronModel` to reuse the eager/no-weights
plumbing and the deterministic fill/verify KV oracle, and overrides only the KV
geometry (uniform full-attention, unpacked bf16) and the stacked-KV opt-in. This
exercises the per-buffer NIXL registration + transfer without real weights or the
giga-kernel.

Selected under the ``VLLM_NEURON_SYNTHETIC_MODEL=1`` gate via a config.json with
``"architectures": ["SyntheticStackedNeuronModel"]``.

WARNING: testing only. Not for production inference.
"""

from __future__ import annotations

import os
from pathlib import Path

from vllm_neuron.model.synthetic.synthetic import SyntheticNeuronModel

SYNTHETIC_STACKED_MODEL_PATH = str(Path(__file__).parent / "synthetic_stacked_config")


class SyntheticStackedNeuronModel(SyntheticNeuronModel):
    """Synthetic model with the GK stacked KV layout on the decode role.

    Inherits SyntheticNeuronModel's eager/no-weights plumbing, the deterministic
    prefill fill + decode verify oracle, and the KV spec (which produces
    even-SWA / odd-full layers when ``sliding_window`` is set, matching GPT-OSS).
    Overrides only the stacked-KV opt-in.

    KV geometry follows the server config:
      - ``sliding_window`` set (hf_overrides) -> even layers use SWA, so the
        hybrid manager pairs each SWA layer with the next full layer into one
        buffer (the HMA layout the giga-kernel stacks). The stacked allocator
        needs uniform buffer sizes, so pick ``sliding_window >= max_model_len``
        (``_compute_swa_num_blocks`` then clips SWA to the full-context blocks).
      - packed FP8: set the server ``--kv-cache-dtype fp8`` plus
        ``neuron_config.fp8_packed_kv=True`` -> a 5-D packed K cache. The oracle
        indexes the packed layout; byte-exact fill/verify still holds.
    """

    @property
    def use_stacked_kv_cache(self) -> bool:
        """Opt into the stacked KV buffer + per-buffer NIXL registration, gated to
        the decode role via ``VLLM_NEURON_SYNTH_STACKED_DECODE=1`` (the prefill
        server leaves it unset and keeps the native per-layer allocation)."""
        return os.environ.get("VLLM_NEURON_SYNTH_STACKED_DECODE") == "1"

    def set_kv_cache_buffers(self, k_bufs, v_bufs) -> None:
        """GK handoff of the PER-BUFFER KV caches, one base-0 tensor for each HMA group.

        This is a no-op for correctness. The synthetic verify reads the per-layer views that
        ``bind_kv_cache`` published, and those are views of these buffers. The record gives the
        log the geometry of the caches.

        Without this method the per-buffer runner raises ``AttributeError`` on the decode role,
        which appears only as a worker crash at engine start.

        Args:
            k_bufs: One K cache for each HMA group.
            v_bufs: The matching V caches.

        Example:
            >>> model.set_kv_cache_buffers([k0, k1], [v0, v1])  # doctest: +SKIP
        """
        self.K_cache_bufs = list(k_bufs)
        self.V_cache_bufs = list(v_bufs)
        print(
            f"[SyntheticStacked] set_kv_cache_buffers: {len(self.K_cache_bufs)} groups, "
            f"K[0]={tuple(k_bufs[0].shape)} V[0]={tuple(v_bufs[0].shape)} "
            f"dtype={k_bufs[0].dtype} device={k_bufs[0].device}",
            flush=True,
        )
