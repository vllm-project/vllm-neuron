# SPDX-License-Identifier: Apache-2.0
"""Vendored NKI kernel source used by the vLLM-Neuron functional layer.

Kernels here are copied from nkilib (see each subpackage's __init__ for source
provenance) so vLLM-Neuron can use a kernel version that is not yet available in
the consumed dependency image. Each is a temporary vendor to be removed once the
upstream version lands in-image.
"""
