# SPDX-License-Identifier: Apache-2.0
"""Example kernels demonstrating the in-tree kernel-development pattern.

Category subpackage under ``vllm_neuron/_kernels`` — see the parent package
docstring. Holds small, self-contained kernels (e.g. ``silu_and_mul``) that
exercise the ``nki-library-testing`` framework end-to-end; a real kernel would
live in its own category subpackage. Tests live under ``test/kernels/examples/``.
"""
