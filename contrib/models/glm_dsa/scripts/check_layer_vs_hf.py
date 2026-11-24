# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.x decoder layer on CPU: this port vs ``transformers``' ``GlmMoeDsaDecoderLayer``.

(Adapted from ``contrib/models/hy4_preview/scripts/check_layer_vs_hf.py``: GLM has a plain
residual, no sinks or output gate, and transformers' GLM attention is interleaved like the
port, so no RoPE switch is needed.)

Loads one layer's real weights from the converted checkpoint through the
port's own weight loaders (world size 1), runs

* a prefill of ``--tokens`` tokens in chunks of ``--chunk`` (each chunk after
  the first reads its prefix back from the paged cache, as segmented prefill
  does on device), and
* one decode step against that cache,

and compares every row against the HF layer run on all ``tokens + 1`` at once.
With more than ``index_topk`` = 2048 tokens (the default is 2600) the later
rows attend to a DSA *selection*, so this checks the indexer too; that needs a
``"full"`` indexer layer (0, 1, 2, 6, 10, ...), since a shared one would need the
previous full layer's selection. The HF layer gets the
*same* weights: the fp8 experts are dequantised with the converter's per-row
scales, so the comparison isolates the port's arithmetic from the
quantisation.

    VLLM_NEURON_CPU_MODE=1 python check_layer_vs_hf.py \\
        --model <dir>/GLM-5.3-neuron-fp8 --layer 6
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

_VLLM_CONFIG_CTX = None


def init_world_of_one() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    # Entered for the life of the process (initialize_model_parallel reads it),
    # and kept referenced: a collected context manager restores the old config.
    global _VLLM_CONFIG_CTX
    _VLLM_CONFIG_CTX = set_current_vllm_config(VllmConfig())
    _VLLM_CONFIG_CTX.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29533",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1)


class Checkpoint:
    def __init__(self, root: str):
        self.root = Path(root)
        self.weight_map = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {}

    def _handle(self, key: str):
        fname = self.weight_map[key]
        if fname not in self.handles:
            self.handles[fname] = safe_open(self.root / fname, framework="pt")
        return self.handles[fname]

    def slice(self, key: str):
        return self._handle(key).get_slice(key)

    def tensor(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)


def load_port_layer(
    layer, mappings: dict[str, str], prefix: str, ckpt: Checkpoint
) -> None:
    from vllm_neuron.utils.weight_loader import get_weight_loader

    state = {}
    for name, param in layer.named_parameters():
        key = mappings[f"{prefix}.{name}"]
        tensor = get_weight_loader(param).load([ckpt.slice(key)], 0)
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"{name}: loaded {tuple(tensor.shape)}, param {tuple(param.shape)}"
            )
        state[name] = tensor.to(param.dtype)
    layer.load_state_dict(state, strict=True, assign=True)


def hf_layer_state(ckpt: Checkpoint, layer_idx: int) -> dict[str, torch.Tensor]:
    """One layer of the converted checkpoint, in transformers' names, experts bf16."""
    prefix = f"model.layers.{layer_idx}."
    renames = ()
    out = {}
    for key in ckpt.weight_map:
        if not key.startswith(prefix) or key.endswith("_scale") and ".experts." in key:
            continue
        name = "." + key[len(prefix) :]
        tensor = ckpt.tensor(key)
        if name.endswith(".experts.gate_up_proj"):
            scale = ckpt.tensor(key + "_scale")
            tensor = (tensor.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
        elif name.endswith(".experts.down_proj_t"):
            scale = ckpt.tensor(key[: -len("_t")] + "_scale")  # [E, H]
            tensor = (tensor.float().transpose(1, 2) * scale.unsqueeze(-1)).to(
                torch.bfloat16
            )
            name = name[: -len("_t")]
        for old, new in renames:
            name = name.replace(old, new)
        out[name[1:]] = tensor
    return out


def check_query_split(attn, config, world: int = 4, chunk: int = 256) -> None:
    """``_select_prefill`` split over ``world`` emulated ranks == unsplit.

    On device each rank selects for its own slice of the chunk's queries and an
    all-gather assembles them -- in the *collective's* block order, which on
    Trn2 is not tp-rank order. So the emulated all-gather returns the blocks in
    a shuffled order: first every rank runs once and its gather inputs are
    recorded, then every rank runs again and is handed the recorded blocks,
    concatenated in that shuffled order. An implementation that assumed rank
    order would fail here, as the first one did on device.
    """
    order = [*range(world - 2, -1, -1), world - 1]  # 2, 1, 0, 3 for 4

    class _Group:
        rank_in_group = 0

        def __init__(self, recorded=None):
            self.recorded = recorded  # {call index: {rank: tensor}}
            self.calls = 0
            self.rank = 0

        def all_gather(self, t, dim):
            i = self.calls
            self.calls += 1
            if self.recorded is None:
                record.setdefault(i, {})[self.rank] = t
                return torch.cat([t] * world, dim=dim)
            return torch.cat([self.recorded[i][r] for r in order], dim=dim)

    keys = config.index_topk + chunk + 512
    iq = torch.randn(chunk, config.index_n_heads, config.index_head_dim).bfloat16()
    iw = torch.randn(chunk, config.index_n_heads)
    ik = torch.randn(keys, config.index_head_dim).bfloat16()
    # A prefix of keys - chunk keys visible to everyone, then causal.
    prefix = torch.ones(chunk, keys - chunk, dtype=torch.bool)
    visible = torch.cat((prefix, torch.ones(chunk, chunk, dtype=torch.bool).tril()), 1)
    whole = attn._select_prefill(iq, iw, ik, visible, None)

    record: dict = {}
    saved = attn.world_size, attn.tp_group
    try:
        attn.world_size = world
        for replay in (False, True):
            outs = []
            for r in range(world):
                group = _Group(record if replay else None)
                group.rank = r
                attn.tp_group = group
                outs.append(attn._select_prefill(iq, iw, ik, visible, torch.tensor(r)))
    finally:
        attn.world_size, attn.tp_group = saved
    same = all(torch.equal(o, whole) for o in outs)
    print(
        f"  query-split selection, {world} emulated ranks gathered in order "
        f"{order}: every rank == unsplit: {same}"
    )
    if not same:
        raise SystemExit("FAIL: query-split DSA selection differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=2600)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ref-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="dtype the transformers layer runs in. float32 by default: its eager "
        "attention does the softmax in the activation dtype, so a bf16 reference "
        "carries more rounding than the port does.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(64, os.cpu_count() or 1))
    init_world_of_one()

    from glm_dsa.config import glm_config
    from glm_dsa.hy_v4.config import ROPE_STYLE
    from glm_dsa.hy_v4.model import HYV4RotaryEmbedding
    from glm_dsa.model import GLMDecoderLayer, checkpoint_mappings
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf

    hf_config = AutoConfig.from_pretrained(args.model)
    hf_config._attn_implementation = "eager"
    config = glm_config(hf_config)
    ckpt = Checkpoint(args.model)
    li = args.layer
    print(
        f"layer {li}: mlp={config.mlp_layer_types[li]} "
        f"indexer={config.indexer_types[li]} rope={ROPE_STYLE}"
    )

    # ── this port ────────────────────────────────────────────────────────
    port = GLMDecoderLayer(config, li)
    load_port_layer(port, checkpoint_mappings(config), f"model.layers.{li}", ckpt)
    port.eval()

    # ── transformers ─────────────────────────────────────────────────────
    ref_dtype = getattr(torch, args.ref_dtype)
    ref = hf.GlmMoeDsaDecoderLayer(hf_config, li).to(ref_dtype)
    missing, unexpected = ref.load_state_dict(hf_layer_state(ckpt, li), strict=False)
    if missing:
        raise RuntimeError(f"HF layer missing weights: {missing}")
    if ref.self_attn.indexer is not None:
        ref.self_attn.indexer.k_norm.float()
        ref.self_attn.indexer.weights_proj.float()
    if hasattr(ref.mlp, "gate"):
        ref.mlp.gate.float()
    ref.eval()

    h = config.hidden_size
    total = args.tokens + 1
    if config.indexer_types[li] != "full" and total > config.index_topk:
        raise SystemExit(
            f"layer {li} shares its DSA selection; past {config.index_topk} tokens "
            f"it needs the previous full layer's. Pick a full-indexer layer."
        )
    # Inputs with the scale real residual streams have after a few layers.
    streams = (torch.randn(total, h) * 0.5).to(torch.bfloat16)
    positions = torch.arange(total)
    bs = args.block_size
    rope = HYV4RotaryEmbedding(config)

    def run_port(call, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked prefill of ``tokens`` rows, then one decode row, on a fresh cache.

        Blocks 1.. hold the sequence (0 is vLLM's null block); the table is
        padded with -1 as the runner does. The cache is the runner's key-only
        ``[pages, 1, block, 704]`` one.
        """
        n_blocks = (total + bs - 1) // bs
        width = n_blocks + 2
        port.self_attn.kv_pages = torch.zeros(
            n_blocks + 1, 1, bs, port.self_attn.entry_dim, dtype=torch.bfloat16
        )
        table = torch.full((1, width), -1, dtype=torch.long)
        table[0, :n_blocks] = torch.arange(n_blocks) + 1
        slots = positions + bs
        outs, selected = [], None
        steps = [
            (a, min(a + args.chunk, args.tokens))
            for a in range(0, args.tokens, args.chunk)
        ]
        steps.append((args.tokens, total))
        for a, b in steps:
            meta = {
                f"layers.{li}.self_attn": {
                    "slot_mapping": slots[a:b],
                    "block_size": bs,
                    "max_query_len": b - a,
                    "decode_token_threshold": 1,
                    "block_table_tensor": table,
                    "cached_seq_len": torch.tensor([[a]], dtype=torch.int32),
                }
            }
            out, selected = call(
                inputs[a:b], positions[a:b].int(), rope(positions[a:b]), meta
            )
            outs.append(out)
        return torch.cat(outs), selected

    with torch.no_grad():
        rot = hf.GlmMoeDsaRotaryEmbedding(hf_config)
        pos_emb = rot(streams.to(ref_dtype), positions.unsqueeze(0))
        mask = torch.full((total, total), torch.finfo(ref_dtype).min).triu(1)
        mask = mask.to(ref_dtype)[None, None]
        # A shared-indexer layer needs the previous full layer's top-k; below
        # index_topk that is every key, which the causal mask then trims.
        prev = (
            None
            if ref.self_attn.indexer is not None
            else (torch.arange(total, dtype=torch.int32).expand(1, total, total))
        )
        ref_out, _ = ref(
            streams.to(ref_dtype).unsqueeze(0),
            attention_mask=mask,
            position_embeddings=pos_emb,
            position_ids=positions.unsqueeze(0),
            prev_topk_indices=prev,
        )
        ref_out = ref_out[0]

        # The attention sub-block alone, fed the reference's own normed input,
        # to localise a disagreement.
        h_ref = ref.input_layernorm(streams.to(ref_dtype).unsqueeze(0))
        a_ref, _, _ = ref.self_attn(
            h_ref,
            position_embeddings=pos_emb,
            attention_mask=mask,
            position_ids=positions.unsqueeze(0),
            prev_topk_indices=prev,
        )
        a_mine, _ = run_port(
            lambda inp, pos, pe, meta: port.self_attn(inp, pos, pe, meta),
            h_ref[0].to(torch.bfloat16),
        )
        a_rel = (a_mine.float() - a_ref[0].float()).norm(dim=-1) / a_ref[
            0
        ].float().norm(dim=-1)
        print(
            f"  self_attn alone: per-token median {a_rel.median():.2e}, "
            f"rows <= {config.index_topk} {a_rel[: config.index_topk].median():.2e}, "
            f"rows > {config.index_topk} {a_rel[config.index_topk :].median():.2e}"
        )

        # fp32 streams in: the port casts its output to the streams' dtype, and a
        # bf16 round of ~0.5-magnitude streams is ~6% of a layer's update --
        # which would be measured here as the port's error. Everything inside
        # the sub-blocks still runs in bf16.
        out, selected = run_port(
            lambda inp, pos, pe, meta: port(inp.float(), pos, pe, meta),
            streams,
        )
        print(
            f"  decode row selects {int(selected.sum())} of {selected.shape[-1]} keys"
            if selected.dtype == torch.bool
            else f"  decode row selects {selected.shape[-1]} positions"
        )
        if port.self_attn.indexer is not None:
            check_query_split(port.self_attn, config)
        if ref.self_attn.indexer is not None and total > config.index_topk:
            # The same selection from transformers' indexer, for the decode row.
            # Keys are [cache span | the token itself]: cache column j holds
            # position j, the last column is position ``tokens``.
            attn = ref.self_attn
            q_resid = attn.q_a_layernorm(attn.q_a_proj(h_ref))
            ref_idx = attn.indexer(
                h_ref, q_resid, pos_emb, mask[:, 0], positions.unsqueeze(0)
            )[0, -1]
            ref_set = set(ref_idx.tolist())
            cols = selected[0].nonzero().flatten().tolist()
            mine_set = {args.tokens if j == selected.shape[-1] - 1 else j for j in cols}
            print(
                f"  decode row: {len(mine_set & ref_set)} of {len(ref_set)} selected "
                f"keys agree with transformers' indexer"
            )

    def update_error(
        name: str, mine: torch.Tensor, theirs: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        # The layer adds a small update onto large streams, so compare updates:
        # comparing outputs would hide an error in the update behind the residual.
        d_mine, d_ref = mine.float() - inp.float(), theirs.float() - inp.float()
        rel = ((d_mine - d_ref).norm() / d_ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            d_mine.flatten(), d_ref.flatten(), dim=0
        ).item()
        per_token = (d_mine - d_ref).flatten(1).norm(dim=1) / d_ref.flatten(1).norm(
            dim=1
        )
        outliers = (per_token > 5e-2).nonzero().flatten().tolist()
        print(
            f"{name:12s} rel. error of the layer's update {rel:.2e}, cosine {cos:.6f}, "
            f"per-token median {per_token.median():.2e} max {per_token.max():.2e}, "
            f"{len(outliers)} tokens > 5%" + (f": {outliers[:10]}" if outliers else "")
        )
        return per_token

    t, c = args.tokens, min(args.chunk, args.tokens)
    parts = [update_error("chunk 1", out[:c], ref_out[:c], streams[:c])]
    if t > c:
        parts.append(update_error("later chunks", out[c:t], ref_out[c:t], streams[c:t]))
    parts.append(update_error("decode", out[t:], ref_out[t:], streams[t:]))
    # A token whose 8th and 9th expert scores are within rounding of each other
    # can pick a different expert in bf16 than in fp32; that token then differs
    # by tens of percent while every other one agrees to bf16 precision. So the
    # criterion is on the typical token, with a small allowance for such ties.
    per_token = torch.cat(parts)
    ok = per_token.median() < 2e-2 and (per_token > 5e-2).float().mean() <= 0.05
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
