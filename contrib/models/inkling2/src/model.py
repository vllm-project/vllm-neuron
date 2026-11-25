# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (text, BF16 experts dequantized at load) for vLLM Neuron: the looped decoder on one trn2.3xlarge.

The arithmetic is ``ops.py`` (checked against Hugging Face, ``test/unit``); this module holds one tensor-parallel
rank's weights (``weights.py``), the caches and the vLLM plumbing.

Caches. Every layer visit has its own cache layer: 24 layers plus the 12 loop-core layers twice more, 48 in all
(``Inkling2Config.cache_layers``), each one KV head of 128 a rank. All are declared as full-attention layers of one
cache group, so they share the block table. The 20 sliding-window layers read only the blocks that cover their window
(``ops.window_blocks``) and the attention mask drops the rest; their cache layers still hold the whole context. The
short convolutions' input windows (3 tokens per convolution, 4 convolutions per cache layer) are an extra KV layer,
``state.conv``, addressed by each request's first block and read and written as one block row a request: with the
prefix cache off (``factory.py``) no two live requests share a first block.

A prefill step is one request's chunk; a decode step is one token per row.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape, set_weight_loader

from . import ops
from . import weights as W
from .config import Inkling2Config

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
F32 = torch.float32
STATE_LAYER = "state.conv"


def _nki(kernel):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    return wrap_nki(kernel)[get_logical_nc_config()]


def cache_name(c: int) -> str:
    return f"layers.{c}.self_attn"


class Layer(nn.Module):
    """One model layer's weights on this rank (names as ``ops.decoder_layer`` reads them)."""

    def __init__(self, cfg: Inkling2Config, L: int):
        super().__init__()
        H, d, hq, hkv = cfg.hidden_size, cfg.head_dim, cfg.q_heads_per_rank, cfg.kv_heads_per_rank
        E, I = cfg.num_experts, cfg.inter_per_rank
        P = lambda *shape, dtype=DTYPE: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)  # noqa
        # every matrix [in, out] (ops.py)
        self.wq, self.q_scale = P(H, hq * d), P(hq, d)
        self.wk, self.wv = P(H, hkv * d), P(H, hkv * d)
        self.wo, self.wgate, self.wsink = P(hq * d, H), P(H, hq * d), P(H, hq)
        self.wr, self.r_scale, self.r_proj = P(H, hq * cfg.d_rel), P(hq, cfg.d_rel), P(cfg.d_rel, cfg.rel_extent)
        self.post_norm, self.mlp_query = P(H), P(1, H, dtype=F32)
        self.post_attn_conv, self.post_mlp_conv = P(H, ops.CONV_TAPS, dtype=F32), P(H, ops.CONV_TAPS, dtype=F32)
        self.router, self.router_bias = P(H, E, dtype=F32), P(E, dtype=F32)
        self.gate_up, self.down = P(E, H, 2 * I), P(E, I, H)
        self.full = cfg.is_full(L)
        if L > 0:
            self.attn_query = P(3 if self.full else 1, H, dtype=F32)
        if self.full:
            self.k_static, self.v_static = P(ops.CONV_TAPS, hkv * d), P(ops.CONV_TAPS, hkv * d)
            self.k_wproj, self.v_wproj = P(H, hkv * ops.CONV_TAPS), P(H, hkv * ops.CONV_TAPS)
            self.q_norm, self.k_norm, self.v_norm = P(H), P(H), P(H)
        else:
            self.k_conv = P(hkv * d, ops.CONV_TAPS, dtype=F32)
            self.v_conv = P(hkv * d, ops.CONV_TAPS, dtype=F32)
            self.in_norm = P(H)
        self.loop = cfg.loop_start <= L < cfg.loop_end
        if self.loop:
            for g in ("q", "k", "v", "mlp"):
                setattr(self, f"gate_{g}_down", P(2 * H, cfg.gate_rank))
                setattr(self, f"gate_{g}_forget", P(cfg.gate_rank, H, dtype=F32))
                setattr(self, f"gate_{g}_gain", P(cfg.gate_rank, H, dtype=F32))

    def params(self) -> dict:
        p = {n: t for n, t in self.named_parameters()}
        p.update(getattr(self, "fused", {}))
        if self.loop:
            for g in ("q", "k", "v", "mlp"):
                p[f"gate_{g}"] = (p[f"gate_{g}_down"], p[f"gate_{g}_forget"], p[f"gate_{g}_gain"])
        return p


class StateLayout:
    """The conv windows of one request: per cache layer, keys ``[3, kv_ch]``, values ``[3, kv_ch]``, attention
    output ``[3, H]``, MoE output ``[3, H]`` (bf16, oldest first), cache layers in order; the first half of the
    record in the state layer's K plane and the second half in its V plane, each at the request's first block."""

    def __init__(self, cfg: Inkling2Config, block_size: int):
        kv_ch, H = cfg.kv_heads_per_rank * cfg.head_dim, cfg.hidden_size
        self.widths = (kv_ch, kv_ch, H, H)
        self.per_layer = ops.CONV_WINDOW * sum(self.widths)
        self.total = cfg.cache_layers * self.per_layer
        self.half = -(-self.total // 2)
        self.block_size = block_size
        self.head_size = -(-self.half // block_size)
        self.plane_block = block_size * self.head_size          # elements of one block in one plane

    def offset(self, c: int, slot: int) -> int:
        return c * self.per_layer + ops.CONV_WINDOW * sum(self.widths[:slot])


class PagedIO(ops.LayerIO):
    """``ops.LayerIO`` over vLLM's paged caches for one step: ``B`` rows of ``S`` tokens."""

    def __init__(self, model: "Inkling2ForCausalLM", positions, block_table, slot_mapping, first_blocks, valid,
                 reset, B: int, S: int):
        # padding entries of the block table are -1: read the null block there (its positions are past the
        # request's length, so the causal mask drops them)
        self.m, self.positions, self.slot_mapping = model, positions, slot_mapping
        self.block_table = block_table.to(torch.int64).clamp(min=0)
        self.B, self.S, self.valid = B, S, valid
        # decode rows the runner padded (slot 0, the null block's first slot, as in prefill): the NKI MoE skips
        # their expert loads
        self.valid_rows = (slot_mapping.reshape(B, S)[:, 0] > 0) if S == 1 else None
        L = model.state_layout
        planes = model.state_planes                    # [N, plane_block] each: one row per block
        # whole-block reads and writes (one contiguous row a request): an element-indexed gather of the record
        # costs an indirect copy per element (measured 16.7 ms a row a step)
        self.first = first_blocks.to(torch.int64).clamp(min=0)
        rec = torch.cat([planes[0].index_select(0, self.first)[:, :L.half],
                         planes[1].index_select(0, self.first)[:, :L.half]], dim=1)[:, :L.total]
        if reset is not None:
            rec = rec * reset.to(rec.dtype)
        self.rec = rec
        self.new = {}

    def kv(self, c, k, v, window=None):
        kc, vc = self.m.kv_cache[c]
        n, _, block, d = kc.shape
        blk, pos = paged_slot_indices(self.slot_mapping, block, n)
        kc[blk, 0, pos] = k.transpose(1, 2).reshape(-1, d)
        vc[blk, 0, pos] = v.transpose(1, 2).reshape(-1, d)
        if window is None:
            blocks, first = self.block_table, torch.zeros(self.B, dtype=torch.int64, device=k.device)
        else:
            # a sliding layer reads only the blocks of its window (about S + window keys, not the whole context)
            blocks, first = ops.window_blocks(self.block_table, self.positions[:, 0], self.S, window, block)
        nr = blocks.shape[1]
        k_ctx = kc.index_select(0, blocks.reshape(-1)).view(self.B, 1, nr * block, d)
        v_ctx = vc.index_select(0, blocks.reshape(-1)).view(self.B, 1, nr * block, d)
        kv_pos = first.view(-1, 1) + torch.arange(nr * block, device=k.device).view(1, -1)
        return k_ctx, v_ctx, kv_pos

    def window(self, c, slot):
        L = self.m.state_layout
        o, w = L.offset(c, slot), L.widths[slot]
        return self.rec[:, o:o + ops.CONV_WINDOW * w].view(self.B, ops.CONV_WINDOW, w)

    def put_window(self, c, slot, seq):
        self.new[(c, slot)] = ops.next_window(seq, self.valid).to(DTYPE)

    def moe_decode(self, x, idx, w, gate_up, down):
        """The decode MoE on nkilib's selective-expert kernel with Inkling 2's activation (kernels/moe_decode.py)."""
        if self.m.moe_kernel == "hw":
            from .kernels import moe_decode_hw as K2
            return K2.moe_decode_hw(_nki, x, idx, w, gate_up, down, self.valid_rows)
        if self.m.moe_kernel == "tkg":
            from .kernels import moe_decode as K
            return K.moe_decode(_nki, x, idx, w, gate_up, down)
        return ops.moe_gather(x, idx, w, gate_up, down, self.m.config.gate_cap, self.m.config.up_cap)

    @property
    def attn_decode(self):
        """The NKI decode attention core (kernels/attn_decode.py), opt-in with INKLING2_NKI_ATTN=1."""
        if not self.m.attn_kernel:
            return None
        from .kernels import attn_decode as K4
        return lambda *a: K4.attn_decode(_nki, *a)

    def all_reduce(self, x):
        tp = self.m.tp_group
        return tp.all_reduce(x) if tp.world_size > 1 else x

    def publish(self):
        """Write every new window back to the request's first block (padding rows go to the null block)."""
        L = self.m.state_layout
        parts = [self.new[(c, s)].reshape(self.B, -1) for c in range(self.m.config.cache_layers) for s in range(4)]
        rec = torch.cat(parts, dim=1)
        rec = torch.cat([rec[:, :L.half], rec.new_zeros(self.B, L.plane_block - L.half),
                         rec[:, L.half:], rec.new_zeros(self.B, L.plane_block - (L.total - L.half))], dim=1)
        planes = self.m.state_planes
        planes[0].index_copy_(0, self.first, rec[:, :L.plane_block])
        planes[1].index_copy_(0, self.first, rec[:, L.plane_block:])


class Inkling2ForCausalLM(nn.Module):
    """The text decoder, the LM head and the on-device sampler."""

    supports_async_reorder = True

    def __init__(self, config: Inkling2Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        config.tp_size = self.world_size
        problems = config.problems()
        if problems:
            raise ValueError("the Inkling 2 contrib model cannot serve this config: " + "; ".join(problems))
        self.config = c = config
        self.embed_tokens = VocabDimShardedEmbedding(vocab_size=c.vocab_size, embed_dim=c.hidden_size, dtype=DTYPE,
                                                     tp_group=self.tp_group.device_group)
        P = lambda *shape, dtype=DTYPE: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)  # noqa
        self.embed_norm, self.norm = P(c.hidden_size), P(c.hidden_size)
        self.final_query = P(1, c.hidden_size, dtype=F32)
        self.message_query = P(c.loop_layers, 4, c.hidden_size, dtype=F32)
        self.lm_head = P(c.hidden_size, c.lm_rows_per_rank)
        self.layers = nn.ModuleList([Layer(c, L) for L in range(c.num_layers)])
        nc = c.neuron_config if hasattr(c, "neuron_config") else None
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (nc.max_logprobs != 0 or nc.debug_logits_dir is not None)
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
        # padded LM-head rows (past the unpadded vocab) never win; filled in load_weights
        self.register_buffer("vocab_mask", torch.zeros(c.lm_rows_per_rank, dtype=F32), persistent=False)
        # the NKI decode MoE (kernels/moe_decode.py), opt-in with INKLING2_NKI_MOE=1: alone it is faster than the torch
        # gather (0.267 against 0.339 ms a visit at two rows) but inside the 48-visit decode graph it measured slower
        # (TPOT 59.6 against 42.6 ms): a kernel call per visit costs more than the fused torch ops around it
        import os

        # INKLING2_NKI_MOE: "1" nkilib moe_tkg (kernels/moe_decode.py), "hw" the one-kernel decode MoE with
        # hardware DMA descriptors (kernels/moe_decode_hw.py); unset: torch
        self.moe_kernel = {"1": "tkg", "hw": "hw"}.get(os.environ.get("INKLING2_NKI_MOE", ""), None)
        self.attn_kernel = os.environ.get("INKLING2_NKI_ATTN") == "1"
        if self.moe_kernel == "tkg":
            from .kernels import moe_decode as K

            K.install()
        self.kv_cache = None
        self.state_layout = None
        self.state_planes = None
        self.block_size = None
        self._capture_block_size()

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        cfg = Inkling2Config.from_hf(hf_config)
        cfg.neuron_config = neuron_config
        return cls(cfg)

    def compiler_args_for(self, graph_type, example_inputs, base):
        """neuronx-cc arguments a graph (the runner calls this unless ``--optimization-level`` is given).

        The decode graph compiles at ``-O3``: the ``-O1`` modular flow wraps each collective
        (``PSEUDO_FUNCTION_RETURN`` / ``EVENT_SEMAPHORE``), and a decode step here has 96 all-reduces (two a layer
        visit). ``INKLING2_DECODE_O`` / ``INKLING2_PREFILL_O`` override the levels (prefill keeps the base level)."""
        import os

        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        level = os.environ.get("INKLING2_DECODE_O" if graph_type == "decode" else "INKLING2_PREFILL_O")
        if graph_type == "decode" and level is None:
            level = "3"
        return with_optimization_level(base, int(level)) if level else base

    # -- KV cache --

    def _capture_block_size(self) -> None:
        try:
            from vllm.config import get_current_vllm_config

            self.block_size = int(get_current_vllm_config().cache_config.block_size)
        except Exception:  # noqa: BLE001 (no engine config, e.g. host tests)
            pass

    def get_kv_spec(self) -> KVSpec:
        c = self.config
        if self.block_size is None:
            raise RuntimeError("the Inkling 2 model was built outside a vLLM engine config")
        self.state_layout = StateLayout(c, self.block_size)
        layers = [LayerSpec(name=cache_name(i), num_kv_heads=c.kv_heads_per_rank, head_size=c.head_dim, dtype=DTYPE)
                  for i in range(c.cache_layers)]
        layers.append(LayerSpec(name=STATE_LAYER, num_kv_heads=1, head_size=self.state_layout.head_size, dtype=DTYPE))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict) -> None:
        if self.state_layout is None:
            self.get_kv_spec()
        self.kv_cache = [(kv_caches[cache_name(i)][0], kv_caches[cache_name(i)][1])
                         for i in range(self.config.cache_layers)]
        planes = kv_caches[STATE_LAYER]
        self.state_planes = [planes[0].reshape(planes[0].shape[0], -1), planes[1].reshape(planes[1].shape[0], -1)]
        L = self.state_layout
        if planes[0].shape[-2] * planes[0].shape[-1] != L.plane_block:
            raise ValueError(f"{STATE_LAYER}: cache {tuple(planes[0].shape)}, expected {L.block_size} x "
                             f"{L.head_size} per block")

    # -- forward --

    @torch.no_grad()
    def forward(self, input_ids, positions, inputs_embeds=None, attn_metadata=None, sampling_positions=None,
                sampling_params=None, spec_decode_metadata=None, logit_mask=None, rank=None, **kwargs):
        # Keep this block identical in every model that declares supports_async_reorder (vllm-neuron's
        # test_async_reorder.py compares them).
        async_reorder_perm = kwargs.pop("async_reorder_perm", None)
        async_reorder_future = kwargs.pop("async_reorder_future", None)
        async_reorder_mask = kwargs.pop("async_reorder_mask", None)
        if async_reorder_perm is not None:
            input_ids = torch.where(
                async_reorder_mask,
                input_ids,
                torch.index_select(async_reorder_future, 0, async_reorder_perm),
            )
        if kwargs:
            raise TypeError(f"unexpected forward arguments {sorted(kwargs)}")
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds are not supported")
        if spec_decode_metadata is not None:
            raise ValueError("speculative decoding is not supported")
        c = self.config
        md0 = attn_metadata[cache_name(0)]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]
        positions = positions.to(torch.int64)
        T = input_ids.shape[-1] if input_ids.dim() == 1 else input_ids.numel()
        h = self.embed_tokens(input_ids.reshape(-1), scatter_tokens=False, rank=rank)
        h = ops.rms_norm(h, self.embed_norm, c.eps, DTYPE)
        block_table = md0["block_table_tensor"]
        if is_prefill:
            B, S = 1, T
            block_table = block_table[:1]
            # the chunk's real tokens: the runner pads a chunk with slot 0 (the null block's first slot)
            valid = (md0["slot_mapping"] > 0).sum().to(torch.int64).view(1)
            reset = (md0["cached_seq_len"].reshape(-1)[:1] > 0).view(1, 1)
        else:
            B, S = T, 1
            block_table = block_table[:T]
            valid, reset = 1, None
        io = PagedIO(self, positions.reshape(B, S), block_table, md0["slot_mapping"].reshape(-1),
                     block_table[:, 0], valid, reset, B, S)
        params = {"layers": [l.params() for l in self.layers], "message_query": self.message_query,
                  "final_query": self.final_query, "norm": self.norm}
        h = ops.text_model(params, c, h.view(B, S, -1), io, dense=bool(is_prefill))
        io.publish()

        h = torch.index_select(h.reshape(B * S, -1), 0, sampling_positions)
        logits = ops.softcap((h @ self.lm_head).float(), c.final_cap) + self.vocab_mask
        if self.on_device_sampling_config is None:
            full = self.tp_group.all_gather(logits, dim=1) if self.world_size > 1 else logits
            return full[:, :c.unpadded_vocab_size]
        tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        gathered = None
        if self._gather_logits:
            gathered = self.tp_group.all_gather(logits, dim=1) if self.world_size > 1 else logits
        return tokens, gathered

    # -- weights --

    def _loaders_and_mappings(self, prefix: str = "model.llm.") -> dict:
        c, tp, r_ = self.config, self.world_size, None
        d, hq, hkv, I = c.head_dim, c.q_heads_per_rank, c.kv_heads_per_rank, c.inter_per_rank
        f32 = lambda t: t.float()  # noqa: E731
        T_ = lambda t: t.t().contiguous()  # noqa: E731 (checkpoint [out, in] -> ops.py's [in, out])
        m = {}

        def add(param, keys, fn):
            set_weight_loader(param, SafetensorsWeightLoader(transform=fn))
            m[param._inkling_name] = keys

        for name, p in self.named_parameters():
            p._inkling_name = name
        rows_e = self.embed_tokens.vocab_size_per_rank
        add(self.embed_tokens.weight, f"{prefix}embed.weight",
            lambda s, rank: pad_to_shape(s[0][rank * rows_e:(rank + 1) * rows_e], (rows_e, c.hidden_size)))
        add(self.embed_norm, f"{prefix}embed_norm.weight", lambda s, rank: s[0][:])
        add(self.norm, f"{prefix}norm.weight", lambda s, rank: s[0][:])
        add(self.final_query, f"{prefix}attn_res.output_query", lambda s, rank: f32(s[0][:]).view(1, -1))
        add(self.message_query, [f"{prefix}middle_cycle_cross_pass_queries", f"{prefix}middle_cycle_cross_pass_kv_queries"],
            lambda s, rank: W.message_queries(s[0], s[1]))
        add(self.lm_head, f"{prefix}unembed.weight",
            lambda s, rank: T_(W.lm_head(s[0], rank, c.lm_rows_per_rank, c.unpadded_vocab_size)))
        for L, lay in enumerate(self.layers):
            a = f"{prefix}layers.{L}.attn."
            g = f"{prefix}layers.{L}."
            add(lay.wq, a + "wq_du.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * d)))
            add(lay.q_scale, a + "q_l2_clip.scale", lambda s, rank: W.rows(s[0], rank, hq))
            add(lay.wk, a + "wk_dv.weight", lambda s, rank: T_(W.rows(s[0], rank, hkv * d)))
            add(lay.wv, a + "wv_dv.weight", lambda s, rank: T_(W.rows(s[0], rank, hkv * d)))
            add(lay.wo, a + "wo_ud.weight", lambda s, rank: T_(W.cols(s[0], rank, hq * d)))
            add(lay.wgate, a + "gate_attn.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * d)))
            add(lay.wsink, a + "attn_sink_proj.weight", lambda s, rank: T_(W.rows(s[0], rank, hq)))
            add(lay.wr, a + "wr_du.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * c.d_rel)))
            add(lay.r_scale, a + "rel_l2_clip.scale", lambda s, rank: W.rows(s[0], rank, hq))
            add(lay.r_proj, a + "rel_logits_proj.proj", lambda s, rank: s[0][:])
            add(lay.post_norm, g + "mlp_norm.weight", lambda s, rank: s[0][:])
            add(lay.mlp_query, f"{prefix}attn_res.mlp_queries.{L}", lambda s, rank: f32(s[0][:]).view(1, -1))
            if L > 0:
                n = 3 if lay.full else 1
                add(lay.attn_query, f"{prefix}attn_res.attn_queries.{L - 1}",
                    lambda s, rank, n=n: f32(s[0][:]).view(n, -1))
            add(lay.post_attn_conv, g + "attn_sconv.weight", lambda s, rank: f32(s[0][:]).squeeze(1))
            add(lay.post_mlp_conv, g + "mlp_sconv.weight", lambda s, rank: f32(s[0][:]).squeeze(1))
            add(lay.router, g + "mlp.gate.weight", lambda s, rank: T_(f32(s[0][:])))
            add(lay.router_bias, g + "mlp.gate.bias", lambda s, rank: f32(s[0][:]))
            e = g + "mlp.experts."
            nv = L < 20  # NVFP4 layers carry amax and the outer denominator; MXFP8 ones do not
            keys13 = [e + "w13_weight", e + "w13_weight_scale"] + (
                [e + "w13_weight_amax", e + "w13_weight_outer_scale_denominator"] if nv else [])
            keys2 = [e + "w2_weight", e + "w2_weight_scale"] + (
                [e + "w2_weight_amax", e + "w2_weight_outer_scale_denominator"] if nv else [])
            add(lay.gate_up, keys13, lambda s, rank, nv=nv: W.gate_up(
                s[0], s[1], s[2] if nv else None, s[3] if nv else None, rank, I).transpose(1, 2).contiguous())
            add(lay.down, keys2, lambda s, rank, nv=nv: W.down(
                s[0], s[1], s[2] if nv else None, s[3] if nv else None, rank, I).transpose(1, 2).contiguous())
            if lay.full:
                for x in ("k", "v"):
                    add(getattr(lay, f"{x}_static"), a + f"{x}_sconv.static_w",
                        lambda s, rank: W.cols(s[0], rank, hkv * d))
                    add(getattr(lay, f"{x}_wproj"), a + f"{x}_sconv.weight_proj",
                        lambda s, rank: T_(W.rows(s[0], rank, hkv * ops.CONV_TAPS)))
                    add(getattr(lay, f"{x}_norm"), g + f"{x}_input_norm.weight", lambda s, rank: s[0][:])
                add(lay.q_norm, g + "q_input_norm.weight", lambda s, rank: s[0][:])
            else:
                for x in ("k", "v"):
                    add(getattr(lay, f"{x}_conv"), a + f"{x}_sconv.weight",
                        lambda s, rank: f32(W.rows(s[0], rank, hkv * d)).squeeze(1))
                add(lay.in_norm, g + "attn_norm.weight", lambda s, rank: s[0][:])
            if lay.loop:
                for ours, theirs in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                    cg = g + f"controllers.{theirs}."
                    add(getattr(lay, f"gate_{ours}_down"), cg + "down_weight", lambda s, rank: T_(s[0][:]))
                    add(getattr(lay, f"gate_{ours}_forget"), cg + "forget_weight", lambda s, rank: T_(f32(s[0][:])))
                    add(getattr(lay, f"gate_{ours}_gain"), cg + "gain_weight", lambda s, rank: T_(f32(s[0][:])))
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        """Read this rank's share of every text-decoder tensor (``weights.py``). The vision tower, its projector
        and the speculator's weights are never named, so they are never read."""
        self._capture_block_size()
        mappings = self._loaders_and_mappings()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(self.rank, self.world_size, self, mappings, device)
        state = {k: (v.to(DTYPE) if v.dtype.is_floating_point and v.dtype not in (DTYPE, F32) else v)
                 for k, v in result.state_dict.items()}
        missing = set(dict(self.named_parameters())) - set(state)
        if missing:
            raise ValueError(f"Inkling 2 weights not loaded: {sorted(missing)[:8]} ({len(missing)} in all)")
        self.load_state_dict(state, strict=False, assign=True)
        # stacked projections (ops.fuse_projections): fewer, larger matmuls a visit; built once on the device
        for lay in self.layers:
            f = ops.fuse_projections({n: t for n, t in lay.named_parameters()})
            lay.fused = {k: f[k] for k in ("w_qside", "qside_sizes", "w_kv", "gate_qkv", "kv_conv") if k in f}
        # built here: the runner constructs the module on the meta device
        c = self.config
        mask = torch.zeros(c.lm_rows_per_rank, dtype=F32)
        mask[max(0, c.unpadded_vocab_size - self.rank * c.lm_rows_per_rank):] = torch.finfo(F32).min
        self.vocab_mask = mask.to(device)
