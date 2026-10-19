# SPDX-License-Identifier: Apache-2.0
"""NKI kernels authored in-tree in vLLM Neuron.

Kernels here are written against the same NKI APIs as kernels in the NKI
kernel library (NKL, ``nkilib``), so they can be tested with NKL's
test framework.

Kernels are organized into category subpackages, mirroring how NKL groups
kernels under ``nkilib/core/<category>/``. Their tests live under the matching
``test/kernels/<category>/`` path.

Each kernel module exposes:

- an ``@nki.jit`` kernel entry point (the kernel under test), and
- a torch reference function with a *matching signature*, returning a dict
  keyed by output name. The NKL test framework enforces the
  signature match, so keep the two in sync.
"""
