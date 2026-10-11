"""`mtp.2.confidence_head` — `DSparkConfidenceHead.forward` from the frozen partition reference.
Whole module on one core; no collective.
"""

import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128


@nki.jit
def kernel(input_0, input_1, mtp_2_confidence_head_proj_weight):
    weight = mtp_2_confidence_head_proj_weight

    n_tok = input_0.shape[1]
    d_hidden = input_0.shape[2]
    d_embed = input_1.shape[2]
    out_features, d_cat = weight.shape

    assert d_hidden + d_embed == d_cat, (
        f"cat({d_hidden}, {d_embed}) != weight in_features {d_cat}"
    )
    assert out_features == 1, f"expected out_features == 1, got {out_features}"
    assert d_hidden % PMAX == 0 and d_embed % PMAX == 0, (
        f"PMAX ({PMAX}) must divide both {d_hidden} and {d_embed}"
    )

    c_hidden = d_hidden // PMAX
    c_embed = d_embed // PMAX
    c_total = c_hidden + c_embed

    out = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.shared_hbm)

    xt = nl.ndarray((PMAX, n_tok, c_total), dtype=input_0.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=xt[:, :, 0:c_hidden],
        src=input_0.ap(
            pattern=[[c_hidden, PMAX], [d_hidden, n_tok], [1, c_hidden]], offset=0
        ),
        dge_mode=nisa.dge_mode.none,
    )
    nisa.dma_copy(
        dst=xt[:, :, c_hidden:c_total],
        src=input_1.ap(
            pattern=[[c_embed, PMAX], [d_embed, n_tok], [1, c_embed]], offset=0
        ),
        dge_mode=nisa.dge_mode.none,
    )

    w_flat = weight.reshape((d_cat,))
    wt = nl.ndarray((PMAX, c_total), dtype=weight.dtype, buffer=nl.sbuf)
    wt[:, 0:c_hidden] = nl.load(w_flat[0:d_hidden].reshape((PMAX, c_hidden)))
    wt[:, c_hidden:c_total] = nl.load(w_flat[d_hidden:d_cat].reshape((PMAX, c_embed)))

    acc = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.psum)
    for i in range(c_total):
        nisa.nc_matmul(dst=acc, stationary=wt[:, i:i + 1], moving=xt[:, :, i],
                       accumulate=(i > 0))

    res = nl.ndarray((1, n_tok), dtype=nl.float32, buffer=nl.sbuf)
    res[...] = nl.copy(acc)
    nl.store(out, value=res)

    return out
