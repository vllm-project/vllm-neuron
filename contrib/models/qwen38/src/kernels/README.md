# The Qwen3.5 per-layer NKI kernels

Copied without changes from [sglang-neuron](https://github.com/sgl-project/sglang-neuron) (Apache-2.0),
`sglang_neuron/models/qwen35/kernels/giga`, at commit `6326e2370c4b6972d4da8500c36f07edb4234c3a` (the Qwen3.8 branch
of that repository). The edits that the copies carry against their own origin are listed in that package's
`PROVENANCE.md`. The only setting this package adds is `QWEN35_GDNP_WIDE=512` (`__init__.py`), the GDN prefill's
widest matmul that a Trainium2 NeuronCore accepts.

| File | Kernel | sha256 (first 16 hex digits) |
| --- | --- | --- |
| `gdn_decode.py` | gdn_decode_tkg: one GDN decode step | `3e45c890af5400d8` |
| `gdn_prefill.py` | gdn_prefill_tkg: one request's GDN prefill chunk | `135ac4eb2a0d86ff` |
| `attn_decode.py` | attn_decode_tkg: gated attention decode over KV pages | `5e2e8582d157ef26` |
| `attn_prefill.py` | attn_prefill_tkg: gated attention prefill chunk, paged | `e0dd1e40c9541b1e` |
| `gdn_decode_torch.py` | torch reference (CPU mode, tests) | `a0089a1a158d1ae2` |
| `attn_decode_torch.py` | torch reference (CPU mode, tests) | `761405a42b32e3d3` |
| `core/attention/attention_cte.py` | library module the attention prefill imports | `bb7df5758f66cc2f` |
| `core/utils/allocator.py` | library module the attention prefill imports | `26058f289b966109` |
| `core/utils/kernel_helpers.py` | library module the attention prefill imports | `88b33d052da679ce` |
| `core/utils/modular_allocator.py` | library module the attention prefill imports | `a05ee9d8505e57b0` |
