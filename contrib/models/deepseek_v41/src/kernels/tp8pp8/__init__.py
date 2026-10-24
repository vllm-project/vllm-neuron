# SPDX-License-Identifier: Apache-2.0
"""The pipeline scheme's megakernels (`DEEPSEEK_V41_SCHEME=tp8pp8`: 8 stages of 8
ranks): its own copies of the block kernel, the drafter (every sequence of a
step in one launch), the stage-boundary kernel and the helper modules, so
nothing done here can change the flat scheme's path (`../tp64`). The per-op
kernels one level up are shared."""

from . import draft_kernel, layer_kernel, mk_common, mk_compress, mk_moe, mk_mx, pp_kernels  # noqa: F401
