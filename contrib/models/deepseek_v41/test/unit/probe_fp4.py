from kharness import run_nki, compare  # noqa: E402 - sets the environment first
import torch
import nki, nki.isa as nisa, nki.language as nl
from deepseek_v41 import model as M
from deepseek_v41.kernels.mk_compress import fp4_qdq_tile, _rne, _round_e2m1

@nki.jit
def qdq(x, flags):
    blk, e4m3 = flags
    T, n = x.shape
    out = nl.ndarray((T, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    t = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:T, :], src=x)
    q = fp4_qdq_tile(t, T, n, blk, e4m3)
    nisa.dma_copy(dst=out, src=q[0:T, :])
    return out

@nki.jit
def rne_probe(x):
    T, n = x.shape
    out = nl.ndarray((T, 2 * n), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:T, :], src=x)
    r = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    r[0:T, :] = _rne(t[0:T, :], (T, n))
    nisa.dma_copy(dst=out[0:T, 0:n], src=r[0:T, :])
    e = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    e[0:T, :] = _round_e2m1(t[0:T, :], (T, n))
    nisa.dma_copy(dst=out[0:T, n:2 * n], src=e[0:T, :])
    return out

torch.manual_seed(0)
x = (torch.randn(6, 512) * 0.4).to(torch.bfloat16)
for blk, e4m3 in ((16, True), (32, False)):
    got = run_nki(qdq, x, (blk, int(e4m3)))
    ref = M.fp4_quant_dequant(x, blk, scale_e4m3=e4m3)
    compare(f"fp4 blk={blk} e4m3={e4m3}", got, ref, tol_rel=1e-3)
    print("  sample got:", got[0, :8].tolist(), "\n  sample ref:", ref[0, :8].tolist(), "\n  sample x  :", x[0, :8].tolist())
y = torch.tensor([[0.4, 0.5, 1.5, 2.5, -2.5, 3.5, 1.25, 1.75, 0.75, -0.75, 5.0, 2.0, 0.1, 4.5, 6.0, -1.5]], dtype=torch.float32)
out = run_nki(rne_probe, y)
print("rne  :", out[0, :16].tolist())
print("e2m1 :", out[0, 16:].tolist())
print("want :", torch.round(y[0]).tolist(), M._round_to_bits(y, 1, 0)[0].tolist())
