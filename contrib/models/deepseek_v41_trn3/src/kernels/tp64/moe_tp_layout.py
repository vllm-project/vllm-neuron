# SPDX-License-Identifier: Apache-2.0
"""The decode MoE weight layout of the fused decode launches (`fused/<launch>/moe_tp.py`),
made once on the host (`mk_model.moe_tp_map`).

The deployment's layout (the prefill kernels and every step of more than
`mk_model.MOE_TP_MAX_T` rows keep it): the 384 routed experts by id, 6 a rank (`w1` / `w3`
`[6, dim/4, inter]`, `w2` `[6, inter/4, dim]` MXFP4 words + `[.., K/32, N]` scales), the shared
expert and the router replicated. A decode step has 6 rows a sequence: the routed experts then
cost each rank the weight streams of the few experts its rows picked, and the world all-reduce
after the MoE waits for the busiest rank (two or three experts where the median rank has none).

Decode layout (`TP` x `EP`, TP = 8 ranks of a group, EP = 8 groups of 48 experts):

  rank r, group g = r // TP, slot t = r % TP
  routed    the group's 48 experts (`48 g ..`), each rank its hidden columns
            H_t = [288 t, 288 t + 288) of every one: `w13` (gate | up, tile order
            [w1 H[0:256] | w3 H[0:256] | w1 H[256:288] | w3 H[256:288]]) as
            `[49, 128, nkc * 576]` (partition p, chunk kc = contraction row kc*128 + p), its
            scales with the down projection's in one block `[49, 16, nkc * 576 + dim]` (row g:
            scale row kc*16 + g of gate|up at [g, kc*576 + n], down scale row g (< 9; zeros after)
            at [g, nkc*576 + n]), `w2` rows H_t as `[49, 72, dim]`. Expert 48 is zeros: the
            padding slot of the active-expert list.
  shared    72 blocks of 32 hidden columns: ranks 0-7 two (2r, 2r+1), ranks 8-63 one (r + 8);
            core c of a rank takes its c-th block (zeros when it has one): `sh13` `[2, 128,
            nkc * 64]` fp8 words ([w1 32 | w3 32]), scales `[2, 16, nkc * 64]`, `sh2` `[2, 8, dim]`
            + scales `[2, 1, dim]`; its output is the rank's partial (no 1/world scale).
  router    `gate_w_t` as bf16 (the checkpoint's dtype: the fp32 copy is exact).

Every hidden value the decode MoE computes is the deployment's (the same products, SwiGLU and
quantization blocks: a rank's columns are whole 32-column scale blocks); what changes is how
the down projections' fp32 partial sums are split over ranks and cores before the world
all-reduce adds them.
"""

import torch

TP = 8
NKC_ROWS = 128
HID = 288
T0 = 256
SH_BLOCK = 32


def shared_blocks(rank: int, world: int, inter: int) -> tuple:
    """The (core 0, core 1) shared-expert hidden blocks of `rank` (None: no block)."""
    nb = inter // SH_BLOCK
    two = nb - world                     # ranks with two blocks
    assert 0 <= two <= world
    if rank < two:
        return (2 * rank, 2 * rank + 1)
    return (two + rank, None)


def _tile_cols(a: torch.Tensor, b: torch.Tensor, t0: int) -> torch.Tensor:
    """[a[:, :t0] | b[:, :t0] | a[:, t0:] | b[:, t0:]] (gate / up halves per output tile)."""
    return torch.cat([a[:, :t0], b[:, :t0], a[:, t0:], b[:, t0:]], dim=1)


def _chunked(m: torch.Tensor, rows: int) -> torch.Tensor:
    """`[K, N]` -> `[rows, K / rows * N]` with chunk kc of row p at `[p, kc * N ..]`."""
    k, n = m.shape
    return m.reshape(k // rows, rows, n).permute(1, 0, 2).reshape(rows, (k // rows) * n).contiguous()


def group_of(rank: int, n_experts: int, world: int) -> tuple:
    """`(first expert, experts, hidden columns [h0, h1))` of `rank`'s group in the TP x EP layout."""
    groups = world // TP
    ng = n_experts // groups
    g, t = rank // TP, rank % TP
    return g * ng, ng, (t * HID, (t + 1) * HID)


def routed(rank: int, world: int, n_experts: int, get_slice) -> dict:
    """The rank's routed tensors. `get_slice(name, e, h0, h1)` -> expert `e`'s MX tensor `name`
    restricted to hidden columns [h0, h1): `w1` / `w3` `[dim/4, h1-h0]` int16, `s1` / `s3`
    `[dim/32, h1-h0]` uint8, `w2` `[(h1-h0)/4, dim]` int16, `s2` `[(h1-h0)/32, dim]` uint8."""
    e0, ng, (h0, h1) = group_of(rank, n_experts, world)
    assert h1 - h0 == HID
    w13, s13, w2 = [], [], []
    for j in range(ng):
        e = e0 + j
        w13.append(_chunked(_tile_cols(get_slice("w1", e, h0, h1), get_slice("w3", e, h0, h1), T0), NKC_ROWS))
        sg = _chunked(_tile_cols(get_slice("s1", e, h0, h1), get_slice("s3", e, h0, h1), T0), 16)
        sd = get_slice("s2", e, h0, h1)
        sd = torch.cat([sd, sd.new_zeros((16 - sd.shape[0], sd.shape[1]))], dim=0)
        # one scale block a row group: [gate|up scales of every chunk | down scales]
        s13.append(torch.cat([sg, sd], dim=1).contiguous())
        w2.append(get_slice("w2", e, h0, h1).contiguous())
    for lst in (w13, s13, w2):
        lst.append(torch.zeros_like(lst[0]))
    return {"w1": torch.stack(w13), "s1": torch.stack(s13), "w2": torch.stack(w2),
            "s2": torch.zeros((1, 1), dtype=torch.uint8),
            "w3": torch.zeros((1, 1), dtype=w13[0].dtype), "s3": torch.zeros((1, 1), dtype=torch.uint8),
            "expert_lo": torch.tensor([[float(e0)]], dtype=torch.float32)}


def stacked_slicer(get):
    """`get_slice` over deployment-layout stacked tensors: `get(name, owner)` -> rank `owner`'s
    `[n_local, ..]` MX tensor (experts `owner * n_local ..`)."""
    cache = {}

    def stacked(name, owner):
        if (name, owner) not in cache:
            cache[(name, owner)] = get(name, owner)
        return cache[(name, owner)]

    n_local = []

    def get_slice(name, e, h0, h1):
        if not n_local:
            n_local.append(stacked("w1", 0).shape[0])
        owner, i = divmod(e, n_local[0])
        t = stacked(name, owner)[i]
        if name in ("w1", "w3", "s1", "s3"):
            return t[:, h0:h1]
        if name == "w2":
            return t[h0 // 4 : h1 // 4]
        return t[h0 // 32 : h1 // 32]

    return get_slice


def shared(rank: int, world: int, sh_w1, sh_s1, sh_w3, sh_s3, sh_w2, sh_s2) -> dict:
    """The rank's shared-expert slices (fp8 MX words: `sh_w1` `[dim/4, inter]` int32, ...)."""
    inter = sh_w1.shape[-1]
    blocks = shared_blocks(rank, world, inter)
    w13, s13, w2, s2 = [], [], [], []
    for b in blocks:
        if b is None:
            w13.append(torch.zeros_like(w13[0]))
            s13.append(torch.zeros_like(s13[0]))
            w2.append(torch.zeros_like(w2[0]))
            s2.append(torch.zeros_like(s2[0]))
            continue
        c0, c1 = b * SH_BLOCK, (b + 1) * SH_BLOCK
        w13.append(_chunked(torch.cat([sh_w1[:, c0:c1], sh_w3[:, c0:c1]], dim=1), NKC_ROWS))
        s13.append(_chunked(torch.cat([sh_s1[:, c0:c1], sh_s3[:, c0:c1]], dim=1), 16))
        w2.append(sh_w2[c0 // 4 : c1 // 4].contiguous())
        s2.append(sh_s2[c0 // 32 : c1 // 32].contiguous())
    return {"sh_w1": torch.stack(w13), "sh_s1": torch.stack(s13), "sh_w2": torch.stack(w2),
            "sh_s2": torch.stack(s2),
            "sh_w3": torch.zeros((1, 1), dtype=sh_w1.dtype), "sh_s3": torch.zeros((1, 1), dtype=torch.uint8)}


def router(gate_w_t: torch.Tensor) -> torch.Tensor:
    """The router weight as bf16 (exact: the fp32 copy holds bf16 values)."""
    b = gate_w_t.to(torch.bfloat16)
    assert torch.equal(b.float(), gate_w_t.float()), "gate_w_t is not bf16-valued"
    return b.contiguous()
