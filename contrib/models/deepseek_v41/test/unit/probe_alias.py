"""Can a NKI kernel scatter into an *input* HBM tensor in place through wrap_nki?"""
import torch
import nki, nki.isa as nisa, nki.language as nl
from kharness import run_nki, DEVICE, compile_fn

@nki.jit
def scatter_rows(cache, rows, idx):
    """cache[idx[i]] = rows[i] for i < n, in place; returns cache (aliased)."""
    n, d = rows.shape
    sb = nl.ndarray((n, d), dtype=rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb, src=rows)
    ix = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    nisa.dma_copy(dst=cache.ap(pattern=[[d, n], [1, d]], vector_offset=ix, indirect_dim=0), src=sb)
    return cache

@nki.jit
def scatter_rows_out(cache, rows, idx):
    """Same, but into a fresh output copied from cache (no aliasing)."""
    m, d = cache.shape
    n = rows.shape[0]
    out = nl.ndarray((m, d), dtype=cache.dtype, buffer=nl.shared_hbm)
    for r0 in range(0, m, 128):
        nr = min(128, m - r0)
        t = nl.ndarray((128, d), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=cache[r0:r0+nr, :])
        nisa.dma_copy(dst=out[r0:r0+nr, :], src=t[0:nr, :])
    sb = nl.ndarray((n, d), dtype=rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb, src=rows)
    ix = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    nisa.dma_copy(dst=out.ap(pattern=[[d, n], [1, d]], vector_offset=ix, indirect_dim=0), src=sb)
    return out

cache = torch.zeros(256, 64, dtype=torch.bfloat16)
rows = torch.randn(6, 64).to(torch.bfloat16)
idx = torch.tensor([[5], [7], [200], [3], [255], [100]], dtype=torch.int32)
ref = cache.clone(); ref[idx.reshape(-1).long()] = rows

print("=== variant A: return the input (alias)")
try:
    cache_d = cache.to(DEVICE)
    out = run_nki(scatter_rows, cache_d, rows, idx)
    print("returned equal:", torch.equal(out, ref), " input mutated:", torch.equal(cache_d.cpu(), ref))
except Exception as e:
    print("variant A failed:", type(e).__name__, str(e)[:800])
print("=== variant B: fresh output")
out = run_nki(scatter_rows_out, cache, rows, idx)
print("returned equal:", torch.equal(out, ref))
