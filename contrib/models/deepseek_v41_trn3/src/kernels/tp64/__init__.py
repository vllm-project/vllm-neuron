# SPDX-License-Identifier: Apache-2.0
"""The flat scheme's megakernels (`DEEPSEEK_V41_SCHEME=tp64`: one tensor-parallel
group of 64 ranks): its own copies of the block kernel, the drafter (one launch
per sequence) and the helper modules. The batched drafter's generic body lives in
`../batched`. The per-op kernels one level up are shared."""

from . import draft_kernel, layer_kernel, mk_common, mk_compress, mk_moe, mk_mx  # noqa: F401
