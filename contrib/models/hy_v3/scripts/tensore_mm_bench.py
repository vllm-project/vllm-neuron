"""One NeuronCore: TensorE cost of the decode MoE matmul shapes, BF16 x FP8 vs FP8 x FP8 (W8A8).

gate/up shape (moe_tkg): stationary = hidden [128, T], moving = FP8 weight [128, 512] -> psum [T, 512]
  gu_tiled  BF16 hidden, column tiling (128 // T matmuls side by side; what moe_tkg does)
  gu_bf16   BF16 hidden, no tiling
  gu_fp8    FP8 hidden (W8A8), no tiling
  gu_dr     FP8 hidden, perf_mode double_row (2 x 128 rows per instruction; no column tiling)
down shape (lhs_rhs_swap): stationary = FP8 weight [128, 128], moving = hidden [128, T]
  dn_bf16 / dn_fp8 / dn_dr  BF16 / FP8 / FP8 double_row moving operand
Each variant is one NKI kernel issuing N matmuls; the cost per 128-row contraction is
(t(N=8192) - t(N=1024)) / 7168. Run with NKI_DISABLE_TRACE_CACHE=1: the NKI trace cache keys
kernels by argument shapes only (a `tag` input gives every variant its own shape as well).

Usage: NKI_DISABLE_TRACE_CACHE=1 T=32 python tensore_mm_bench.py [variant ...]
"""
import os
import sys
import time

os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.envs import get_compile_backend_name

DEV = "neuron:0"
T = int(os.environ.get("T", "32"))
N = 1024
FP8 = nl.float8_e4m3


def make(kind, N):
    @nki.jit
    def k(xb, xf, w, tag):
        # xb: [128, 2, T] bf16, xf: [128, 2, T] fp8, w: [128, 2, 512] fp8 (all HBM)
        out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
        sb_xb = nl.ndarray((128, 2, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        sb_xf = nl.ndarray((128, 2, T), dtype=FP8, buffer=nl.sbuf)
        sb_w = nl.ndarray((128, 2, 512), dtype=FP8, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb_xb, src=xb)
        nisa.dma_copy(dst=sb_xf, src=xf)
        nisa.dma_copy(dst=sb_w, src=w)
        if kind == "gu_tiled":
            # moe_tkg's gate/up: column tiling packs 128 // T matmuls side by side in the PE array
            f = 128 // T
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for i in range(N):
                c = i % f
                nisa.nc_matmul(dst=ps[c * T:(c + 1) * T, 0:512], stationary=sb_xb[0:128, i % 2, 0:T],
                               moving=sb_w[0:128, i % 2, 0:512], tile_position=(0, T * c), tile_size=(128, T))
            res = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=ps)
            nisa.dma_copy(dst=out[0:128, 0:512], src=res)
        elif kind in ("gu_bf16", "gu_fp8", "gu_dr"):
            ps = nl.ndarray((T, 512), dtype=nl.float32, buffer=nl.psum)
            for i in range(N):
                if kind == "gu_bf16":
                    nisa.nc_matmul(dst=ps, stationary=sb_xb[0:128, i % 2, 0:T], moving=sb_w[0:128, i % 2, 0:512])
                elif kind == "gu_fp8":
                    nisa.nc_matmul(dst=ps, stationary=sb_xf[0:128, i % 2, 0:T], moving=sb_w[0:128, i % 2, 0:512])
                elif i % 2 == 0:  # double_row: one instruction contracts 2 x 128 rows
                    nisa.nc_matmul(dst=ps, stationary=sb_xf[0:128, 0:2, 0:T], moving=sb_w[0:128, 0:2, 0:512],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
            res = nl.ndarray((T, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=ps)
            nisa.dma_copy(dst=out[0:T, 0:512], src=res)
        else:
            ps = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for i in range(N):
                if kind == "dn_bf16":
                    nisa.nc_matmul(dst=ps, stationary=sb_w[0:128, i % 2, 0:128], moving=sb_xb[0:128, i % 2, 0:T])
                elif kind == "dn_fp8":
                    nisa.nc_matmul(dst=ps, stationary=sb_w[0:128, i % 2, 0:128], moving=sb_xf[0:128, i % 2, 0:T])
                elif i % 2 == 0:
                    nisa.nc_matmul(dst=ps, stationary=sb_w[0:128, 0:2, 0:128], moving=sb_xf[0:128, 0:2, 0:T],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
            res = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=ps)
            nisa.dma_copy(dst=out[0:128, 0:T], src=res)
        return out
    return k


def timeit(fn, args, n=20):
    c = torch.compile(fn, backend=get_compile_backend_name(), fullgraph=True)
    with torch.no_grad():
        for _ in range(3): c(*args).cpu()
        t0 = time.time()
        for _ in range(n): c(*args).cpu()
    return (time.time() - t0) / n * 1e3


g = torch.Generator().manual_seed(0)
xb = torch.randn(128, 2, T, generator=g).to(torch.bfloat16)
xf = (torch.randn(128, 2, T, generator=g) * 20).to(torch.float8_e4m3fn)
w = (torch.randn(128, 2, 512, generator=g) * 20).to(torch.float8_e4m3fn)
args = [t.to(DEV) for t in (xb, xf, w)]
print(f"T={T}: per 128-row contraction = (t(N=8192) - t(N=1024)) / 7168")
for kind in sys.argv[1:] or ["gu_tiled", "gu_bf16", "gu_fp8", "gu_dr", "dn_bf16", "dn_fp8"]:
    ts = {}
    for n in [int(x) for x in os.environ.get("NS", "1024,8192").split(",")]:
        kern = make(kind, n)
        kid = ["gu_bf16", "gu_fp8", "gu_dr", "dn_bf16", "dn_fp8", "dn_dr", "gu_tiled"].index(kind) + 1
        tag = torch.zeros(kid, n // 1024 + 7 * T, device=DEV)  # distinct shape per variant: no cache can mix them up
        ts[n] = min(timeit(lambda a, b, c, d, kern=kern: wrap_nki(kern)[2](a, b, c, d), args + [tag]) for _ in range(3))
    per = (ts[8192] - ts[1024]) / 7168 * 1e3 if 8192 in ts and 1024 in ts else float("nan")
    print(f"{kind:9s} " + "  ".join(f"N={k}: {v:.3f} ms" for k, v in ts.items()) + f"  -> {per:.3f} us per 128-row contraction", flush=True)
