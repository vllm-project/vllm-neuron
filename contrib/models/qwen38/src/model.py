# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 dense hybrid (text, BF16) for vLLM Neuron, over the Qwen3.5 per-layer NKI kernels.

Each decoder layer is the input norm, then a gated delta net (3 of every 4 layers) or a gated attention layer,
then the post-attention norm and a SwiGLU MLP. Both mixers are single fused kernels from ``kernels/``:

  - GDN: ``gdn_decode_tkg`` (one token per row) and ``gdn_prefill_tkg`` (one request's chunk): in_proj, conv and
    SiLU, l2norm and gates, the gated delta rule, the gated RMSNorm and this rank's out_proj slice;
  - attention: ``attn_decode_tkg`` and ``attn_prefill_tkg`` over the paged KV cache: QKV and gate, zero-centred
    QK norm, partial RoPE, the K/V write, attention, the sigmoid gate and this rank's o_proj slice.

Every kernel returns this rank's partial sum of the layer output, so each is followed by an all-reduce, as the MLP
is. The kernels run one rank per physical NeuronCore (LNC1): on a trn2.3xlarge the model is served at TP8.

Attention KV. The kernels read pages ``k_cache_t [N, d, block]`` (K transposed per page) and ``v_cache [N,
block, d]``. Each attention layer's two planes from the runner (``[N, 1, block, d]``) are reinterpreted in that
layout once, outside the graph, and the block table is the page table. A prefill chunk starts on a block boundary
(vLLM chunks are multiples of ``max_num_batched_tokens``, a multiple of the block size).

GDN state: ``state.py``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as nxdi_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape, set_weight_loader

from . import weights as W
from .config import Qwen38Config
from .kernels import gdn_decode_torch as gdn_ref
from .state import StateBank, StateLayout, plane_views

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
STATE_LAYERS = ("state.rec", "state.conv")


def _nki(kernel):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    return wrap_nki(kernel)[get_logical_nc_config()]


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm: ``normed * (1 + w)`` in fp32 (HF ``Qwen3_5RMSNorm``)."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(x.dtype)


class MLP(nn.Module):
    """SwiGLU with the intermediate dimension sharded over TP (``NF.mlp``, the NKI MLP kernel on device)."""

    def __init__(self, config: Qwen38Config, tp_group):
        super().__init__()
        self.tp_group = tp_group
        h, i = config.hidden_size, config.intermediate_per_rank
        self.gate_weight = nn.Parameter(torch.empty(h, i, dtype=DTYPE))
        self.up_weight = nn.Parameter(torch.empty(h, i, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(i, h, dtype=DTYPE))

    def forward(self, x):
        out = NF.mlp(x, self.gate_weight, self.up_weight, self.down_weight)
        if self.tp_group.world_size > 1:
            self.tp_group.all_reduce(out)
        return out


class GatedDeltaNet(nn.Module):
    """One GDN layer on this rank's heads: the fused kernels, or the kernels' torch reference in CPU mode."""

    def __init__(self, config: Qwen38Config, layer_idx: int, gdn_index: int, tp_group):
        super().__init__()
        self.layer_idx, self.gdn_index, self.tp_group = layer_idx, gdn_index, tp_group
        self.vh, self.kh = config.gdn_v_heads, config.gdn_k_heads
        self.head = config.linear_key_head_dim
        width = 2 * self.kh * self.head + 2 * self.vh * self.head + 2 * self.vh
        h = config.hidden_size
        self.w_proj_p = nn.Parameter(torch.empty(self.head, h // self.head, width, dtype=DTYPE))
        self.conv_w = nn.Parameter(torch.empty(config.conv_channels, config.linear_conv_kernel_dim, dtype=DTYPE))
        self.w_out = nn.Parameter(torch.empty(self.vh * self.head, h, dtype=DTYPE))
        self.a_log = nn.Parameter(torch.empty(self.vh, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.vh, dtype=torch.float32))
        self.norm_w = nn.Parameter(torch.ones(self.head, dtype=torch.float32))

    def _reference_operands(self):
        flat = self.w_proj_p.permute(1, 0, 2).reshape(-1, self.w_proj_p.shape[2])
        vh = self.vh
        return {"w_qkvz": flat[:, :flat.shape[1] - 2 * vh], "w_a": flat[:, -2 * vh:-vh], "w_b": flat[:, -vh:]}

    def _ref_step(self, x, conv, rec):
        out, new_conv, new_rec = gdn_ref.gdn_decode_torch(
            x, conv_w=self.conv_w, conv_state=conv, a_log=self.a_log, dt_bias=self.dt_bias, norm_w=self.norm_w,
            rec_state=gdn_ref.state_from_kernel(rec), w_out=self.w_out, **self._reference_operands())
        return out, new_conv, gdn_ref.state_to_kernel(new_rec)

    def _reduce(self, out):
        if self.tp_group.world_size > 1:
            self.tp_group.all_reduce(out)
        return out

    def decode(self, x, bank: StateBank):
        conv, rec = bank.get(self.gdn_index)
        if not can_run_kernel(x):
            out, new_conv, new_rec = self._ref_step(x, conv, rec)
        else:
            from .kernels.gdn_decode import gdn_decode_tkg

            out, new_conv, new_rec = _nki(gdn_decode_tkg)(
                x=x, w_proj_p=self.w_proj_p, conv_w=self.conv_w, conv_state=conv.contiguous(), a_log=self.a_log,
                dt_bias=self.dt_bias, norm_w=self.norm_w, rec_state=rec.contiguous(), w_out=self.w_out)
        bank.put(self.gdn_index, new_conv, new_rec)
        return self._reduce(out)

    def prefill(self, x, bank: StateBank, valid):
        """One request's chunk ``x [L, H]``, the first ``valid`` rows real. The kernel passes the state through the
        padding rows unchanged (``valid_len``)."""
        conv, rec = bank.get(self.gdn_index)
        if not can_run_kernel(x):
            outs = []
            for t in range(x.shape[0]):
                out, new_conv, new_rec = self._ref_step(x[t:t + 1], conv, rec)
                real = t < valid
                conv = torch.where(real, new_conv, conv)
                rec = torch.where(real, new_rec, rec)
                outs.append(out)
            out, new_conv, new_rec = torch.cat(outs, dim=0), conv, rec
        else:
            from .kernels.gdn_prefill import gdn_prefill_tkg

            out, new_conv, new_rec = _nki(gdn_prefill_tkg)(
                x=x, w_proj_p=self.w_proj_p, conv_w=self.conv_w, a_log=self.a_log, dt_bias=self.dt_bias,
                norm_w=self.norm_w, w_out=self.w_out, conv_state=conv.contiguous(), rec_state=rec.contiguous(),
                seq_len=x.shape[0], valid_len=valid.view(1, 1).to(torch.int32))
        bank.put(self.gdn_index, new_conv, new_rec)
        return self._reduce(out)


def _rmsnorm_zero_centred(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float().view(-1))).to(x.dtype)


def _rope(t, cos, sin, rotary_dim):
    tr, tp = t[..., :rotary_dim].float(), t[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat([-tr[..., half:], tr[..., :half]], dim=-1)
    return torch.cat([(tr * cos + rotated * sin).to(t.dtype), tp], dim=-1)


class GatedAttention(nn.Module):
    """One full-attention layer over the paged KV cache: the fused kernels, or a torch path with the same
    semantics in CPU mode."""

    def __init__(self, config: Qwen38Config, layer_idx: int, tp_group):
        super().__init__()
        self.layer_idx, self.tp_group = layer_idx, tp_group
        self.name = f"layers.{layer_idx}.self_attn"
        self.nq, self.nkv = config.kernel_q_heads, config.kv_heads
        self.head = d = config.head_dim
        self.rotary_dim, self.rope_theta = config.rotary_dim, config.rope_theta
        h = config.hidden_size
        self.w_qkv = nn.Parameter(torch.empty(h, (self.nq + 2 * self.nkv) * d, dtype=DTYPE))
        self.w_gate_p = nn.Parameter(torch.empty(128, h // 128, self.nq * d, dtype=DTYPE))
        self.w_o = nn.Parameter(torch.empty(self.nq * d, h, dtype=DTYPE))
        self.q_norm_w = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        self.k_norm_w = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        self.k_cache_t = None  # [N, d, block], bound by bind_kv_cache
        self.v_cache = None  # [N, block, d]
        self.page = 0

    def rope(self, positions):
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32,
                                                      device=positions.device) / self.rotary_dim))
        f = positions.float()[:, None] * inv[None, :]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos(), emb.sin()

    def _reduce(self, out):
        if self.tp_group.world_size > 1:
            self.tp_group.all_reduce(out)
        return out

    def _torch_step(self, x, positions, table, write=None):
        """CPU mode: row ``b`` writes its K and V at ``positions[b]`` of its pages, then attends ``0 ..
        positions[b]``. Same rounding points as the kernels' reference."""
        b, d, nq, pg = x.shape[0], self.head, self.nq, self.page
        q = (x @ self.w_qkv[:, :nq * d]).view(b, nq, d)
        k = (x @ self.w_qkv[:, nq * d:(nq + 1) * d]).view(b, d)
        v = (x @ self.w_qkv[:, (nq + 1) * d:(nq + 2) * d]).view(b, d)
        gate = x @ self.w_gate_p.permute(1, 0, 2).reshape(-1, self.w_gate_p.shape[2])
        q = _rmsnorm_zero_centred(q, self.q_norm_w)
        k = _rmsnorm_zero_centred(k, self.k_norm_w)
        cos, sin = self.rope(positions)
        q = _rope(q, cos[:, None, :], sin[:, None, :], self.rotary_dim)
        k = _rope(k, cos, sin, self.rotary_dim)
        pos = positions.long()
        rows = table.long()
        row = rows.gather(1, (pos // pg).view(-1, 1)).view(-1)
        off = pos % pg
        k_new, v_new = k.to(self.v_cache.dtype), v.to(self.v_cache.dtype)
        if write is not None:
            k_new = torch.where(write[:, None], k_new, self.k_cache_t[row, :, off])
            v_new = torch.where(write[:, None], v_new, self.v_cache[row, off, :])
        self.k_cache_t[row, :, off] = k_new
        self.v_cache[row, off, :] = v_new
        mp = rows.shape[1]
        keys = self.k_cache_t[rows].permute(0, 2, 1, 3).reshape(b, d, mp * pg).float()
        values = self.v_cache[rows].reshape(b, mp * pg, d).float()
        scores = torch.matmul(q.float(), keys) * d ** -0.5
        keep = torch.arange(mp * pg, device=x.device)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), values).to(DTYPE).view(b, nq * d)
        out = (out.float() * torch.sigmoid(gate.float())).to(DTYPE)
        return out @ self.w_o

    def decode(self, x, positions, md):
        # null entries are -1; page 0 (vLLM's null block) is finite and masked
        table = torch.clamp(md["block_table_tensor"], min=0).to(torch.int32)
        if not can_run_kernel(x):
            return self._reduce(self._torch_step(x, positions, table))
        from .kernels.attn_decode import attn_decode_tkg

        cos, sin = self.rope(positions)
        half = self.rotary_dim // 2
        out, _k, _v = _nki(attn_decode_tkg)(
            x=x, w_qkv=self.w_qkv, q_norm_w=self.q_norm_w, k_norm_w=self.k_norm_w, w_gate_p=self.w_gate_p,
            cos=cos[:, :half], sin=sin[:, :half], k_cache_t=self.k_cache_t, v_cache=self.v_cache,
            pos=positions.view(-1, 1).to(torch.int32), w_o=self.w_o, block_table=table)
        return self._reduce(out)

    def prefill(self, x, positions, md, valid):
        """One request's chunk of ``L`` rows (a whole number of blocks), starting on a block boundary."""
        table = torch.clamp(md["block_table_tensor"][:1], min=0).to(torch.int32)
        if not can_run_kernel(x):
            outs = [self._torch_step(x[t:t + 1], positions[t:t + 1], table, write=(t < valid).view(1))
                    for t in range(x.shape[0])]
            return self._reduce(torch.cat(outs, dim=0))
        from .kernels.attn_prefill import attn_prefill_tkg

        pg = self.page
        slots = md["slot_mapping"]
        write_pages = torch.clamp(torch.div(slots[0::pg], pg, rounding_mode="floor"), min=0)
        cos, sin = self.rope(positions)
        half = self.rotary_dim // 2
        out, _k, _v = _nki(attn_prefill_tkg)(
            x=x, w_qkv=self.w_qkv, q_norm_w=self.q_norm_w, k_norm_w=self.k_norm_w, w_gate_p=self.w_gate_p,
            cos=cos[:, :half], sin=sin[:, :half], k_cache_t=self.k_cache_t, v_cache=self.v_cache,
            slot=torch.zeros(1, 1, dtype=torch.int32, device=x.device), w_o=self.w_o,
            write_pages=write_pages.to(torch.int32).view(1, -1), prior_pages=table,
            prior_len=md["cached_seq_len"].view(1, 1).to(torch.int32))
        return self._reduce(out)


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen38Config, layer_idx: int, gdn_index, tp_group):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config, tp_group)
        self.is_gdn = gdn_index is not None
        if self.is_gdn:
            self.linear_attn = GatedDeltaNet(config, layer_idx, gdn_index, tp_group)
        else:
            self.self_attn = GatedAttention(config, layer_idx, tp_group)

    def forward(self, x, run_mixer):
        x = x + run_mixer(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen38ForCausalLM(nn.Module):
    """The text decoder, the LM head and the on-device sampler."""

    # The forward consumes the runner's in-graph reorder arguments (asynchronous scheduling). Without it the
    # runner keeps its host swap path, where we measured whole requests decoding at twice the step time.
    supports_async_reorder = True

    def __init__(self, config: Qwen38Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        if config.tp_size != self.world_size:
            config.tp_size = self.world_size
        problems = config.problems()
        if problems:
            raise ValueError("Qwen3.8 contrib model cannot serve this config: " + "; ".join(problems))
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=config.hidden_size, dtype=DTYPE,
            tp_group=self.tp_group.device_group)
        gdn_of = {g: i for i, g in enumerate(config.gdn_layers)}
        self.layers = nn.ModuleList([
            DecoderLayer(config, i, gdn_of.get(i), self.tp_group) for i in range(config.num_hidden_layers)])
        self.attention_names = [f"layers.{i}.self_attn" for i in config.attention_layers]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (nc.max_logprobs != 0 or nc.debug_logits_dir is not None)
        self.lm_head = nxdi_nn.ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE,
            gather_output=not self.on_device_sampling_config, tp_group=self.tp_group.device_group)
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
        self.state_layout = None
        self.state_planes = None
        self.block_size = None
        self._capture_block_size()

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(Qwen38Config.from_configs(hf_config, neuron_config))

    # -- KV cache --

    def _capture_block_size(self) -> None:
        """vLLM's block size, read while the engine's config is current (model build and weight load)."""
        try:
            from vllm.config import get_current_vllm_config

            self.block_size = int(get_current_vllm_config().cache_config.block_size)
        except Exception:  # noqa: BLE001 (no engine config, e.g. host tests)
            pass

    def _block_size(self) -> int:
        if self.block_size is None:
            raise RuntimeError("the Qwen3.8 model was built outside a vLLM engine config; set block_size")
        return self.block_size

    def get_kv_spec(self) -> KVSpec:
        """The attention layers' K/V, plus the two state layers (``state.py``) in the same cache group."""
        c = self.config
        layers = [LayerSpec(name=n, num_kv_heads=c.kv_heads, head_size=c.head_dim, dtype=DTYPE)
                  for n in self.attention_names]
        self.state_layout = StateLayout(gdn_layers=len(c.gdn_layers), v_heads=c.gdn_v_heads,
                                        conv_channels=c.conv_channels, block_size=self._block_size())
        for name, which in zip(STATE_LAYERS, ("rec", "conv")):
            layers.append(LayerSpec(name=name, num_kv_heads=1, head_size=self.state_layout.head_size(which),
                                    dtype=DTYPE))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict) -> None:
        if self.state_layout is None:
            self.get_kv_spec()
        L = self.state_layout
        for i in self.config.attention_layers:
            attn = self.layers[i].self_attn
            k, v = kv_caches[attn.name][0], kv_caches[attn.name][1]
            n, heads, block, d = (int(s) for s in k.shape)
            if heads != 1 or block != L.block_size:
                raise ValueError(f"{attn.name}: cache {tuple(k.shape)}, expected one head of {L.block_size} tokens")
            if block & (block - 1) or block < 128:
                raise ValueError(f"the attention kernels need a power-of-two block size >= 128, got {block}")
            # reinterpretations of the runner's planes, made once outside the graph; the model keeps no other view
            attn.k_cache_t = k.view(n, d, block)
            attn.v_cache = v.view(n, block, d)
            attn.page = block
        self.state_planes = {w: plane_views(kv_caches[name], w) for name, w in zip(STATE_LAYERS, ("rec", "conv"))}
        for w in ("rec", "conv"):
            L.block_stride(w)  # refuses a block size whose planes are not whole rows

    # -- forward --

    @torch.no_grad()
    def forward(self, input_ids, positions, inputs_embeds=None, attn_metadata=None, sampling_positions=None,
                sampling_params=None, spec_decode_metadata=None, logit_mask=None, rank=None, **kwargs):
        # The runner gives the device future of the previous step, plus a permutation and a
        # mask that it builds on the host from the request ids. A request that continues takes
        # the reordered future. A new request, a resumed request, a row that still prefills,
        # and a padded slot take the host input_ids. The host therefore does not read the
        # sampled tokens of the previous step back.
        #
        # The arguments are absent on a step that does not use the reorder, and this block is
        # then a no-op. Remove all three arguments in every case, so a later consumer of
        # ``**kwargs`` never sees a partial set.
        #
        # Keep this block identical in every model that declares supports_async_reorder.
        # test_async_reorder.py compares them and fails on a divergence.
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
        md0 = attn_metadata[self.attention_names[0]]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]
        positions = positions.to(torch.int32)
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)

        first_blocks = md0["block_table_tensor"][:, 0]
        if is_prefill:
            slots = first_blocks[:1]
            # the chunk's real rows: the runner pads a prefill chunk to its bucket with slot 0 (the null block's
            # first slot), which no request's token ever has
            valid = (md0["slot_mapping"] > 0).sum().to(torch.int32)
            reset = (md0["cached_seq_len"].reshape(-1)[0] > 0)
            bank = StateBank(self.state_layout, self.state_planes, slots, reset=reset)
        else:
            bank = StateBank(self.state_layout, self.state_planes, first_blocks)

        for layer in self.layers:
            if layer.is_gdn:
                m = layer.linear_attn
                run = (lambda h, m=m: m.prefill(h, bank, valid)) if is_prefill else (lambda h, m=m: m.decode(h, bank))
            else:
                m = layer.self_attn
                md = attn_metadata[m.name]
                run = ((lambda h, m=m, md=md: m.prefill(h, positions, md, valid)) if is_prefill
                       else (lambda h, m=m, md=md: m.decode(h, positions, md)))
            hidden = layer(hidden, run)
        bank.publish()

        hidden = self.norm(torch.index_select(hidden, 0, sampling_positions))
        logits = self.lm_head(hidden)
        if self.on_device_sampling_config is None:
            return logits
        tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        gathered = None
        if self._gather_logits:
            gathered = self.tp_group.all_gather(logits, dim=1) if self.world_size > 1 else logits
        return tokens, gathered

    # -- weights --

    def _set_loaders(self) -> None:
        c, tp = self.config, self.world_size
        head, kh, vh = c.linear_key_head_dim, c.linear_num_key_heads, c.linear_num_value_heads
        loader = lambda fn: SafetensorsWeightLoader(transform=fn)  # noqa: E731
        rows = self.embed_tokens.vocab_size_per_rank
        set_weight_loader(self.embed_tokens.weight, loader(
            lambda s, rank: pad_to_shape(s[0][rank * rows:(rank + 1) * rows], (rows, c.hidden_size))))
        for layer in self.layers:
            set_weight_loader(layer.mlp.gate_weight, loader(lambda s, rank: W.column_shard(s[0][:], rank=rank,
                                                                                          tp_size=tp)))
            set_weight_loader(layer.mlp.up_weight, loader(lambda s, rank: W.column_shard(s[0][:], rank=rank,
                                                                                        tp_size=tp)))
            set_weight_loader(layer.mlp.down_weight, loader(lambda s, rank: W.row_shard(s[0][:], rank=rank,
                                                                                       tp_size=tp)))
            if layer.is_gdn:
                g, per = layer.linear_attn, vh // tp
                set_weight_loader(g.w_proj_p, loader(lambda s, rank: W.gdn_projection(
                    s[0][:], s[1][:], s[2][:], s[3][:], rank=rank, tp_size=tp, num_k_heads=kh, num_v_heads=vh,
                    head_dim=head)))
                set_weight_loader(g.conv_w, loader(lambda s, rank: W.gdn_conv(
                    s[0][:], rank=rank, tp_size=tp, num_k_heads=kh, num_v_heads=vh, head_dim=head)))
                set_weight_loader(g.w_out, loader(lambda s, rank: W.gdn_out(
                    s[0][:], rank=rank, tp_size=tp, num_v_heads=vh, head_dim=head)))
                set_weight_loader(g.a_log, loader(lambda s, rank: s[0][rank * per:(rank + 1) * per].float()))
                set_weight_loader(g.dt_bias, loader(lambda s, rank: s[0][rank * per:(rank + 1) * per].float()))
                set_weight_loader(g.norm_w, loader(lambda s, rank: s[0][:].float()))
            else:
                a = layer.self_attn
                nq, nkv, d, pad = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.kernel_q_heads
                qkv = lambda s, rank: W.attention_qkv(  # noqa: E731
                    s[0][:], s[1][:], s[2][:], rank=rank, tp_size=tp, num_q_heads=nq, num_kv_heads=nkv, head_dim=d,
                    padded_q_heads=pad)
                set_weight_loader(a.w_qkv, loader(lambda s, rank, qkv=qkv: qkv(s, rank)[0]))
                set_weight_loader(a.w_gate_p, loader(lambda s, rank, qkv=qkv: W.row_permute(qkv(s, rank)[1])))
                set_weight_loader(a.w_o, loader(lambda s, rank: W.attention_out(
                    s[0][:], rank=rank, tp_size=tp, num_q_heads=nq, head_dim=d, padded_q_heads=pad)))
                set_weight_loader(a.q_norm_w, loader(lambda s, rank: s[0][:].float().view(1, -1)))
                set_weight_loader(a.k_norm_w, loader(lambda s, rank: s[0][:].float().view(1, -1)))

    def _mappings(self, prefix: str) -> dict:
        c = self.config
        m = {"embed_tokens.weight": f"{prefix}embed_tokens.weight", "norm.weight": f"{prefix}norm.weight",
             "lm_head.weight": f"{prefix}embed_tokens.weight" if c.tie_word_embeddings else "lm_head.weight"}
        for i, layer in enumerate(self.layers):
            src, dst = f"{prefix}layers.{i}", f"layers.{i}"
            m[f"{dst}.input_layernorm.weight"] = f"{src}.input_layernorm.weight"
            m[f"{dst}.post_attention_layernorm.weight"] = f"{src}.post_attention_layernorm.weight"
            for p in ("gate", "up", "down"):
                m[f"{dst}.mlp.{p}_weight"] = f"{src}.mlp.{p}_proj.weight"
            if layer.is_gdn:
                a = f"{src}.linear_attn"
                m[f"{dst}.linear_attn.w_proj_p"] = [f"{a}.in_proj_qkv.weight", f"{a}.in_proj_z.weight",
                                                    f"{a}.in_proj_a.weight", f"{a}.in_proj_b.weight"]
                m[f"{dst}.linear_attn.conv_w"] = f"{a}.conv1d.weight"
                m[f"{dst}.linear_attn.w_out"] = f"{a}.out_proj.weight"
                m[f"{dst}.linear_attn.a_log"] = f"{a}.A_log"
                m[f"{dst}.linear_attn.dt_bias"] = f"{a}.dt_bias"
                m[f"{dst}.linear_attn.norm_w"] = f"{a}.norm.weight"
            else:
                s = f"{src}.self_attn"
                qkv = [f"{s}.q_proj.weight", f"{s}.k_proj.weight", f"{s}.v_proj.weight"]
                m[f"{dst}.self_attn.w_qkv"] = list(qkv)
                m[f"{dst}.self_attn.w_gate_p"] = list(qkv)
                m[f"{dst}.self_attn.w_o"] = f"{s}.o_proj.weight"
                m[f"{dst}.self_attn.q_norm_w"] = f"{s}.q_norm.weight"
                m[f"{dst}.self_attn.k_norm_w"] = f"{s}.k_norm.weight"
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        """Read this rank's shard of every parameter in the kernels' layouts (``weights.py``). The checkpoint's
        vision tower (``model.visual.``) is never named, so it is never read."""
        self._capture_block_size()
        self._set_loaders()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(self.rank, self.world_size, self, self._mappings("model.language_model."), device)
        state = {k: (v.to(DTYPE) if v.dtype.is_floating_point and v.dtype not in (DTYPE, torch.float32) else v)
                 for k, v in result.state_dict.items()}
        self.load_state_dict(state, strict=False, assign=True)
