# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the gated DeltaNet prefill (opt-in, see ``flags.py``).

``nki_deltanet.py`` and ``nki_deltanet_fused.py`` are two variants of the
chunked delta rule. They are off by default: on device both are correct but
slower than the torch path in ``deltanet.py``.
"""
