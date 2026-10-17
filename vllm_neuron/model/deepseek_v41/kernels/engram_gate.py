"""The engram read-out's gate — the normalized stream-against-key dot product,
the signed square root, the sigmoid and the gated add — as one NKI kernel.

`model.Engram.forward` is two pieces: `wkv`, which turns the 24 looked-up n-gram
rows into one key per hc copy plus one shared value and is already
`kernels/blockquant`, and everything after it. This is everything after it.

Derived from `engram_short.py`, which is the shipped 4-rank / 2-core kernel for
`layers.1.engram` and carries this same arithmetic at the bottom (its `gpad`
block). What changes here: the token count is ragged rather than a baked 128,
`dim`/`hc` come from the operand shapes rather than module constants, `wkv` and
the collectives are gone (the GEMM is `blockquant`'s and this port runs one
rank), and `nl.rsqrt` gets the Newton correction `hc_mixes._rms_recip` uses —
the hardware instruction is an approximation and the reference is a torch
`rsqrt`.

Interface
---------
    h      [rows, hc * dim]  bf16   the residual stream, hc-major
    key    [rows, hc * dim]  bf16   `wkv`'s key output, same layout
    value  [rows, dim]       bf16   `wkv`'s value output, shared by the copies
    weight [1, hc * dim]     fp32   `q_weight * k_weight`, flattened
    ->     [rows, hc * dim]  bf16   `h + gate * value`

`dim` is read off `value`, so `hc = (hc * dim) / dim`; `rows` is whatever the
scheduler handed the step.

The arithmetic, per (token, hc copy), transcribed from `Engram.forward`:

    rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
    dot  = sum(h * weight * key) * rstd * dim**-0.5
    gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))
    out  = h + gate * value

Normalized **per (token, copy)** over `dim`, not jointly over the copies, which
is why the copy loop is outside the reduction rather than folded into the free
axis. `sign` is `2 * (dot >= 0) - 1`, so a dot of exactly zero gives `+mag` —
`Engram.forward`'s `torch.where(dot < 0, ...)` convention, not `copysign`'s.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width.
PT = 128
#: `rms_norm_eps`.
EPS = 1e-20
#: `Engram.clamp_value`: the floor under `|dot|` before the square root, so the
#: derivative at zero stays finite. `engram_short.CLAMP`.
CLAMP = 1e-6


def _chunk(dim):
    """Free-axis chunk for the reductions: a divisor of `dim`, small enough that
    one fp32 copy is a few KiB per partition. `hc_mixes._sqchunk`'s rule."""
    for cand in (2048, 1024, 512, 256, 128):
        if dim % cand == 0:
            return cand
    return dim


def _rsqrt(m):
    """`1 / sqrt(m)` with one Newton step: `y * (1.5 - 0.5 * m * y^2)`.

    `nl.rsqrt` on its own is a hardware approximation and the reference is a
    torch `rsqrt`; `hc_mixes._rms_recip` corrects it the same way.
    """
    y = nl.rsqrt(m)
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    return nl.multiply(y, corr)


@nki.jit
def gated_add(h, key, value, weight):
    """The engram gate and its gated add. See the module docstring."""
    rows, hcd = h.shape
    vrows, dim = value.shape
    hc = hcd // dim

    assert key.shape == (rows, hcd), f"key {key.shape} wants ({rows}, {hcd})"
    assert vrows == rows, f"value has {vrows} rows, h has {rows}"
    assert hc * dim == hcd, f"stream width {hcd} is not a multiple of dim={dim}"
    assert weight.shape == (1, hcd), f"weight {weight.shape} wants (1, {hcd})"

    ch = _chunk(dim)
    nchunks = dim // ch
    inv_dim = 1.0 / dim
    dim_rsqrt = float(dim) ** -0.5

    out = nl.ndarray((rows, hcd), dtype=h.dtype, buffer=nl.shared_hbm)

    def body(r0, nr):
        for c in range(hc):
            base = c * dim
            # This copy's stream rows, kept in fp32 because the gated add below
            # needs them again once the gate — which depends on every chunk —
            # is known. At `dim = 5120` that is 20 KiB per partition.
            hf = nl.ndarray((nr, dim), dtype=nl.float32, buffer=nl.sbuf)
            hsq = ksq = dotp = None
            for j in range(nchunks):
                lo = base + j * ch
                hj = nl.load(h[r0 : r0 + nr, lo : lo + ch], dtype=nl.float32)
                nisa.tensor_copy(dst=hf[0:nr, j * ch : (j + 1) * ch], src=hj)
                kj = nl.load(key[r0 : r0 + nr, lo : lo + ch], dtype=nl.float32)
                # A freshly loaded `(1, n)` tile broadcasts; a slice of a staged
                # one does not — see `hc_mixes`' module docstring.
                wj = nl.broadcast_to(nl.load(weight[0:1, lo : lo + ch]), (nr, ch))
                prod = nl.multiply(nl.multiply(hj, wj), kj)
                sh = nl.sum(nl.square(hj), axis=1, keepdims=True)
                sk = nl.sum(nl.square(kj), axis=1, keepdims=True)
                sd = nl.sum(prod, axis=1, keepdims=True)
                hsq = sh if hsq is None else nl.add(hsq, sh)
                ksq = sk if ksq is None else nl.add(ksq, sk)
                dotp = sd if dotp is None else nl.add(dotp, sd)

            rstd = nl.multiply(
                _rsqrt(nl.add(nl.multiply(hsq, inv_dim), EPS)),
                _rsqrt(nl.add(nl.multiply(ksq, inv_dim), EPS)),
            )
            dot = nl.multiply(nl.multiply(dotp, rstd), dim_rsqrt)
            mag = nl.sqrt(nl.maximum(nl.abs(dot), CLAMP))
            sgn = nl.subtract(
                nl.multiply(
                    nl.copy(nl.greater_equal(dot, 0.0), dtype=nl.float32), 2.0
                ),
                1.0,
            )
            gate = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
            gate[...] = nl.sigmoid(nl.multiply(mag, sgn))

            for j in range(nchunks):
                vj = nl.load(value[r0 : r0 + nr, j * ch : (j + 1) * ch],
                             dtype=nl.float32)
                yv = nl.ndarray((nr, ch), dtype=nl.float32, buffer=nl.sbuf)
                # `gate * value` rides the activation's per-partition scale,
                # which is what `engram_short` does with the same operands.
                nisa.activation(dst=yv, op=nl.copy, data=vj, scale=gate)
                nisa.tensor_tensor(
                    dst=yv, data1=yv, op=nl.add,
                    data2=hf[0:nr, j * ch : (j + 1) * ch],
                )
                nl.store(
                    out[r0 : r0 + nr, base + j * ch : base + (j + 1) * ch],
                    value=nl.copy(yv, dtype=h.dtype),
                )

    # The ragged-row pattern every kernel in this directory uses: full tiles in
    # `affine_range`, then one more call with a Python row count for the tail.
    full, tail = rows // PT, rows % PT
    for t in nl.affine_range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return out
