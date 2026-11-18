#: the greedy head reads this rank's head rows K-major, `[dim // 128, 128, part]` (= head_w.T): its weight
#: tiles load with plain DMAs and the GEMM runs in the PE's column-tiling mode (`l1/head/head_kernel.py`
#: detects the 3-d layout; `mk_model._plan_fused` passes the model's copy, `mk_model.head_kmajor`)
HEAD_W_KMAJOR = True
