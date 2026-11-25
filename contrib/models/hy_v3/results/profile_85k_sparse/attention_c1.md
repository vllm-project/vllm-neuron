
exec 6: attention module 29.97 ms (union)

| source | ms | instructions |
|---|---|---|
| NKI: attention_tkg.py | 24.70 | 373510 |
| NKI: attention_block_tkg.py | 3.23 | 12641 |
| torch: _sparse_decode_blocks | 1.61 | 20454 |
| no source | 1.22 | 7709 |
| NKI: qkv_tkg.py | 1.13 | 8295 |
| NKI: output_projection_tkg.py | 1.01 | 4352 |
| torch: update_k_summary_decode | 0.99 | 5710 |
| NKI: rmsnorm_tkg.py | 0.75 | 2148 |
| NKI: stream_shuffle_broadcast.py | 0.67 | 8976 |
| torch: reduce | 0.28 | 2773 |
| torch: forward_decode | 0.26 | 656 |
| NKI: rope.py | 0.26 | 2254 |
