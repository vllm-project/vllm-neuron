"""`hc_collapse` — `HyperConnectCollapse`, the tail of the mHC rank-4 residual stream: the four
hyper-connection copies are mixed back down into the single hidden state the next layer reads.
One fp32 multiply-accumulate over 4 streams, no parameters. Whole module on one core; no
collective.
"""

import nki
import nki.isa as nisa
import nki.language as nl

PMAX = 128

CHUNK = 1024


@nki.jit
def kernel(input_0, input_1):
    batch, n_tok, hc, dim = input_0.shape
    rows = batch * n_tok

    assert dim % CHUNK == 0, "CHUNK must divide dim"
    assert rows % PMAX == 0, "PMAX must divide batch*tokens"

    y_out = nl.ndarray((batch, n_tok, dim), dtype=input_0.dtype, buffer=nl.shared_hbm)

    x3 = input_0.reshape((rows, hc, dim))
    pm2 = input_1.reshape((rows, hc))
    y2 = y_out.reshape((rows, dim))

    for t in nl.affine_range(rows // PMAX):
        t0 = t * PMAX

        pm = nl.load(pm2[t0:t0 + PMAX, :])

        for c in range(dim // CHUNK):
            f0 = c * CHUNK

            acc = nl.ndarray((PMAX, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            xg = nl.ndarray((PMAX, hc, CHUNK), dtype=input_0.dtype, buffer=nl.sbuf)

            nisa.dma_copy(dst=xg, src=x3[t0:t0 + PMAX, 0:hc, f0:f0 + CHUNK],
                          dge_mode=nisa.dge_mode.none)

            for j in range(hc):
                if j == 0:
                    nisa.tensor_scalar(dst=acc, data=xg[:, j, :], op0=nl.multiply,
                                       operand0=pm[:, j:j + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,
                                              operand0=pm[:, j:j + 1],
                                              op1=nl.add, operand1=acc)

            nisa.dma_copy(dst=y2[t0:t0 + PMAX, f0:f0 + CHUNK], src=acc,
                          dge_mode=nisa.dge_mode.none)

    return y_out
