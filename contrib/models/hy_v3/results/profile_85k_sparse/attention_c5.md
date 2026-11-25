
exec 6: attention module 29.98 ms (union)

| source | ms | instructions |
|---|---|---|
| torch: sparse_select_blocks | 16.19 | 62362 |
| NKI: attention_tkg.py | 7.86 | 367171 |
| NKI: attention_block_tkg.py | 5.07 | 12181 |
| no source | 4.55 | 8736 |
| torch: _sparse_decode_blocks | 2.41 | 27138 |
| torch: update_k_summary_decode | 1.39 | 3430 |
| NKI: output_projection_tkg.py | 1.30 | 4282 |
| NKI: qkv_tkg.py | 1.25 | 8494 |
| NKI: rmsnorm_tkg.py | 0.78 | 2063 |
| NKI: rope.py | 0.37 | 2286 |
| torch: forward_decode | 0.30 | 621 |
| torch: reduce | 0.26 | 2470 |
| torch: hy3_route | 0.25 | 659 |
| NKI: stream_shuffle_broadcast.py | 0.23 | 2576 |
| torch: _rotate_half | 0.11 | 480 |

sparse_select_blocks by line:

| line | ms | code |
|---|---|---|
| 444 | 4.30 | `summ = k_sum[first].to(torch.float32)  # [B, ns, 2, kvh, D]` |
| 465 | 3.57 | `top = torch.topk(score, k_g - 1, dim=1).indices  # [B, k_g - 1]` |
| 469 | 2.55 | `key = torch.sort(torch.where(top < nsv, top, top + ns), dim=1).values` |
| 475 | 2.18 | `compact = t.gather(1, blocks).to(block_table.dtype)` |
| 455 | 1.74 | `bound = torch.einsum("bhqd,bnhd->bnhq", qpos, smax) + torch.einsum("bh` |
| 452 | 1.28 | `smax = torch.where(empty_g, 0.0, smax)` |
| 451 | 1.07 | `smin = torch.where(empty_g, 0.0, smin)` |
| 449 | 0.60 | `empty_g = smax < smin` |
| 450 | 0.59 | `empty = empty_g.any(dim=-1).any(dim=-1)  # [B, ns]` |
| 460 | 0.46 | `valid = idx < nsv` |
