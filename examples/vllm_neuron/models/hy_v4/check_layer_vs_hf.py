# SPDX-License-Identifier: Apache-2.0
"""One Hy4 decoder layer on CPU: this port vs ``transformers``' ``HYV4DecoderLayer``.

Loads one layer's real weights from the converted checkpoint through the
port's own weight loaders (world size 1), runs

* a prefill of ``--tokens`` tokens, and
* one decode step against the paged latent cache that prefill wrote,

and compares both against the HF layer run on the same input (the decode step
against HF's last row of a ``tokens + 1`` prefill). The HF layer gets the
*same* weights: the fp8 experts are dequantised with the converter's per-row
scales, so the comparison isolates the port's arithmetic from the
quantisation.

transformers' attention uses ``rotate_half`` RoPE, so run this with
``HY4_ROPE_STYLE=neox`` for a like-for-like check (the default, interleaved,
is expected to disagree -- that is what the flag is for).

    HY4_ROPE_STYLE=neox VLLM_NEURON_CPU_MODE=1 python check_layer_vs_hf.py \\
        --model /mnt/nvme/models/Hy4-preview-neuron-fp8 --layer 1
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
    renames = (
        (".hc_attn_layer.hc_pre.hc_", ".attn_hc."),
        (".hc_mlp_layer.hc_pre.hc_", ".ffn_hc."),
        (".learnable_sink_param", ".sinks"),
        (".linear_gate.", ".gate_proj."),
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=40)
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

    from transformers import AutoConfig
    from transformers.models.hy_v4 import modeling_hy_v4 as hf

    from vllm_neuron.model.hy_v4.config import ROPE_STYLE, HYV4Config
    from vllm_neuron.model.hy_v4.model import (
        HYV4DecoderLayer,
        HYV4RotaryEmbedding,
        checkpoint_mappings,
    )

    hf_config = AutoConfig.from_pretrained(args.model)
    hf_config._attn_implementation = "eager"
    config = HYV4Config.from_hf(hf_config)
    ckpt = Checkpoint(args.model)
    li = args.layer
    print(
        f"layer {li}: mlp={config.mlp_layer_types[li]} "
        f"indexer={config.indexer_types[li]} rope={ROPE_STYLE}"
    )

    # ── this port ────────────────────────────────────────────────────────
    port = HYV4DecoderLayer(config, li)
    load_port_layer(port, checkpoint_mappings(config), f"model.layers.{li}", ckpt)
    port.eval()

    # ── transformers ─────────────────────────────────────────────────────
    ref_dtype = getattr(torch, args.ref_dtype)
    ref = hf.HYV4DecoderLayer(hf_config, li).to(ref_dtype)
    missing, unexpected = ref.load_state_dict(hf_layer_state(ckpt, li), strict=False)
    if missing:
        raise RuntimeError(f"HF layer missing weights: {missing}")
    for name in ("attn_hc", "ffn_hc"):
        getattr(ref, name).float()
    ref.self_attn.sinks.data = ref.self_attn.sinks.data.float()
    if ref.self_attn.indexer is not None:
        ref.self_attn.indexer.k_norm.float()
        ref.self_attn.indexer.weights_proj.float()
    if hasattr(ref.mlp, "gate"):
        ref.mlp.gate.float()
    ref.eval()

    hc, h = config.hc_mult, config.hidden_size
    total = args.tokens + 1
    # Inputs with the scale real residual streams have after a few layers.
    streams = (torch.randn(total, hc, h) * 0.5).to(torch.bfloat16)
    positions = torch.arange(total)

    with torch.no_grad():
        rot = hf.HYV4RotaryEmbedding(hf_config)
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

        # Sub-block breakdown (prefill rows), to localise a disagreement.
        post_ref, x_ref = ref.attn_hc(streams.to(ref_dtype).unsqueeze(0))  # HF order
        x_mine, post_mine = port.attn_hc(streams)

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm()).item()

        print(
            f"  attn_hc   pre {rel(x_mine, x_ref[0]):.2e}  "
            f"post {rel(post_mine, post_ref[0]):.2e}"
        )
        h_ref = ref.input_layernorm(x_ref)
        a_ref, _, _ = ref.self_attn(
            h_ref,
            position_embeddings=pos_emb,
            attention_mask=mask,
            position_ids=positions.unsqueeze(0),
            prev_topk_indices=prev,
        )
        h_mine = port.input_layernorm(x_mine)
        print(f"  input_ln  {rel(h_mine, h_ref[0]):.2e}")
        attn_meta = {
            f"layers.{li}.self_attn": {
                "slot_mapping": torch.arange(total) + args.block_size,
                "block_size": args.block_size,
                "max_query_len": total,
                "decode_token_threshold": 1,
                "block_table_tensor": torch.zeros(1, 1, dtype=torch.long),
            }
        }
        port.self_attn.kv_pages = torch.zeros(
            (total + args.block_size - 1) // args.block_size + 4,
            2,
            1,
            args.block_size,
            port.self_attn.cache_head_dim,
            dtype=torch.bfloat16,
        )
        a_mine = port.self_attn(
            h_mine, positions.int(), HYV4RotaryEmbedding(config)(positions), attn_meta
        )
        print(f"  self_attn {rel(a_mine, a_ref[0]):.2e}")
        # Same attention, fed the reference's own normed input.
        a_mine2 = port.self_attn(
            h_ref[0].to(torch.bfloat16),
            positions.int(),
            HYV4RotaryEmbedding(config)(positions),
            attn_meta,
        )
        print(f"  self_attn (ref input) {rel(a_mine2, a_ref[0]):.2e}")

        # Paged cache: page-major [pages, 2, 1, block, 288], plus the two
        # private pages the runner appends (zero page, write sink).
        bs = args.block_size
        n_blocks = (total + bs - 1) // bs + 1
        pages = n_blocks + 1 + 2
        port.self_attn.kv_pages = torch.zeros(
            pages, 2, 1, bs, port.self_attn.cache_head_dim, dtype=torch.bfloat16
        )
        # Blocks 1.. hold this sequence (block 0 is vLLM's null block).
        slots = positions + bs
        block_table = (torch.arange(n_blocks) + 1).unsqueeze(0)
        rope = HYV4RotaryEmbedding(config)

        t = args.tokens
        meta = {
            f"layers.{li}.self_attn": {
                "slot_mapping": slots[:t],
                "block_size": bs,
                "max_query_len": t,
                "decode_token_threshold": 1,
                "block_table_tensor": block_table,
            }
        }
        # fp32 streams in: the port casts its output to the streams' dtype, and a
        # bf16 round of ~0.5-magnitude streams is ~6% of a layer's update --
        # which would be measured here as the port's error. Everything inside
        # the sub-blocks still runs in bf16.
        prefill = port(
            streams[:t].float(), positions[:t].int(), rope(positions[:t]), meta
        )

        meta = {
            f"layers.{li}.self_attn": {
                "slot_mapping": slots[t:],
                "block_size": bs,
                "max_query_len": 1,
                "decode_token_threshold": 1,
                "block_table_tensor": block_table,
            }
        }
        decode = port(
            streams[t:].float(), positions[t:].int(), rope(positions[t:]), meta
        )

    def update_error(
        name: str, mine: torch.Tensor, theirs: torch.Tensor, inp: torch.Tensor
    ) -> float:
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
            f"{name:8s} rel. error of the layer's update {rel:.2e}, cosine {cos:.6f}, "
            f"per-token median {per_token.median():.2e} max {per_token.max():.2e}, "
            f"tokens > 5%: {outliers}"
        )
        return per_token

    u1 = update_error("prefill", prefill, ref_out[:t], streams[:t])
    u2 = update_error("decode", decode, ref_out[t:], streams[t:])
    # A token whose 8th and 9th expert scores are within rounding of each other
    # can pick a different expert in bf16 than in fp32; that token then differs
    # by tens of percent while every other one agrees to bf16 precision. So the
    # criterion is on the typical token, with a small allowance for such ties.
    per_token = torch.cat((u1, u2))
    ok = per_token.median() < 2e-2 and (per_token > 5e-2).float().mean() <= 0.05
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
