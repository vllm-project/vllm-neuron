"""`hc_expand` — `HyperConnectExpand`, the head of the mHC rank-4 residual stream: one hidden
state becomes `hc_mult` = 4 copies of itself, plus the one-hot `pre_mix` that makes the first
layer's `hc_pre` return copy 0 unchanged. No arithmetic at all — pure DMA, so both outputs come
out bit-identical to the reference. Whole module on one core; no collective.
"""

import nki
import nki.isa as nisa
import nki.language as nl

TILE = 128

HC = 4

CHUNK = 1024


@nki.jit
def kernel(hidden_states):
    batch, seqlen, dim = hidden_states.shape
    rows = batch * seqlen
    assert dim % CHUNK == 0 and rows % TILE == 0

    copies_hbm = nl.ndarray((batch, seqlen, HC, dim), dtype=hidden_states.dtype,
                            buffer=nl.shared_hbm)
    mix_hbm = nl.ndarray((batch, seqlen, HC), dtype=nl.float32, buffer=nl.shared_hbm)

    x3 = hidden_states.reshape((rows, 1, dim))
    copies = copies_hbm.reshape((rows, HC, dim))
    mix = mix_hbm.reshape((rows, HC))

    n_tiles = rows // TILE
    n_chunks = dim // CHUNK

    for tile_idx in nl.affine_range(n_tiles):
        row0 = tile_idx * TILE

        pm = nl.ndarray((TILE, HC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pm, 0.0)
        nisa.memset(pm[:, 0:1], 1.0)
        nisa.dma_copy(dst=mix[nl.ds(row0, TILE), 0:HC], src=pm,
                      dge_mode=nisa.dge_mode.none)

        for chunk_idx in nl.affine_range(n_chunks):
            col0 = chunk_idx * CHUNK

            nisa.dma_copy(dst=copies[nl.ds(row0, TILE), 0:HC, nl.ds(col0, CHUNK)],
                          src=x3[nl.ds(row0, TILE), 0:1, nl.ds(col0, CHUNK)].broadcast(1, HC),
                          dge_mode=nisa.dge_mode.none)

    return copies_hbm, mix_hbm
