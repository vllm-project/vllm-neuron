# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS giga-kernel (GK) decode assembly for the mxfp4 perf model.

The ``gpt_oss_decode_mega_kernel`` fuses the WHOLE decoder stack into one NEFF
for the decode step, so it replaces ``GptOssModel``'s per-layer decode loop
entirely. This module holds the glue that:

1. builds the leading-``[num_layers, ...]`` stacked weights the kernel indexes
   (``W[i]``). ``build_stacked_weight_specs`` produces one host-side loader per
   stacked weight (from each weight-holder module's explicit ``gk_weight_loaders``),
   so ``load_weights`` loads each stack to the device once (no per-layer copy);
2. builds the TP and EP rank meshes as tuples-of-tuples for the kernel's two
   ``ReplicaGroup`` args;
3. assembles the shared, layer-invariant attention-control tensors (RoPE cos/sin,
   full + trimmed-SWA block tables, active mask, absolute + window-frame
   positions, kv-update index) from the per-layer ``attn_metadata``;
4. calls ``NF.gpt_oss_mega_decode`` and returns the post-stack SP token shard.

Everything here is decode-only and TRN3/MXFP4-only; prefill keeps the standard
per-layer ``model_mxfp4`` path.

Weight layouts the kernel expects (per layer, before the leading L-stack):
  input_layernorm_weight : [1, H]
  W_qkv                  : [H, (q + 2*kv) * d]
  bias_qkv               : [1, (q + 2*kv) * d]
  W_out                  : [q*d, H]
  bias_out               : [1, H]
  sink                   : [q, 1]
  post_attn_layernorm_wt : [1, H]
  router_weights         : [H, E]        (model stores [E, H] -> pass .T)
  router_bias            : [1, E]
  expert_gate_up_weights : model's packed [E_local, 128, 2, H//512, I] (uint16)
  expert_down_weights    : model's packed [E_local, p_I, I//512, H]    (uint16)
  expert_gate_up_scale   : [E_local, 2, 64, H//512, I//4] (uint8), the PACK64
                           layout; the model holds [E_local, 16, 2, H//512, I],
                           so the host remaps it at weight load when the kernel
                           confirms that it reads the packed layout
  expert_down_scale/bias : the model's packed scale/bias tensors, unchanged
  K/V buffers            : group_size per-group base-0 tensors (see the runner's
                           GK initialize_kv_cache + set_kv_cache_buffers)
"""

import logging
import math
from dataclasses import dataclass

import torch

import vllm_neuron.envs as envs
import vllm_neuron.functional as NF
from vllm_neuron.model.gpt_oss.mega_scale_pack64 import (
    kernel_packs_gate_up_scale,
    pack64_gate_up_scale_transform,
)

# The kernel decides the gate/up MX scale layout. The host must write the layout that the
# kernel reads. Ask the kernel, and do not read the environment here: the flag belongs to
# the kernel package. A second definition of truth in this file makes the two ends disagree.
try:
    from nkilib.models.gpt_oss.c128_giga_kernel.core.utils.kernel_helpers import (
        mega_flag_default_on as _mega_flag_default_on,
    )
except ImportError:  # pragma: no cover - no nkilib, or one without the helper
    _mega_flag_default_on = None
from vllm_neuron.model.gpt_oss.weight_loaders_mxfp4 import Q_WIDTH
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

logger = logging.getLogger(__name__)


def _kernel_packs_gate_up_scale() -> bool:
    return kernel_packs_gate_up_scale(_mega_flag_default_on)


# Prefix for the model buffers holding each leading-[L, ...] stacked weight.
# Use buffers, NOT nn.Parameters: the trace fork's meta-swap binds a Parameter
# stack wrong (device out-of-bounds), a buffer correctly.
GK_STACK_ATTR_PREFIX = "_gk_wstack_"

# The stacked-weight keys, in the order the giga-kernel takes them.
GK_STACK_KEYS = (
    "input_layernorm_weight",
    "W_qkv",
    "bias_qkv",
    "W_out",
    "bias_out",
    "sink",
    "post_attn_layernorm_weight",
    "router_weights",
    "router_bias",
    "expert_gate_up_weights",
    "expert_down_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights_scale",
    "expert_gate_up_bias",
    "expert_down_bias",
)

# Maps each stack key to its per-layer path, used to look up that weight's
# checkpoint key(s) per layer in the load mappings.
GK_STACK_PARAM_PATHS = {
    "input_layernorm_weight": "input_layernorm.weight",
    "W_qkv": "self_attn.qkv_proj_weight",
    "bias_qkv": "self_attn.qkv_proj_bias",
    "W_out": "self_attn.o_proj_weight",
    "bias_out": "self_attn.o_proj_bias",
    "sink": "self_attn.sinks",
    "post_attn_layernorm_weight": "mlp.experts.post_attention_layernorm.weight",
    "router_weights": "mlp.experts.router_weight",
    "router_bias": "mlp.experts.router_bias",
    "expert_gate_up_weights": "mlp.experts.gate_up_proj_weight",
    "expert_down_weights": "mlp.experts.down_proj_weight",
    "expert_gate_up_weights_scale": "mlp.experts.gate_up_proj_scale",
    "expert_down_weights_scale": "mlp.experts.down_proj_scale",
    "expert_gate_up_bias": "mlp.experts.gate_up_proj_bias",
    "expert_down_bias": "mlp.experts.down_proj_bias",
}
assert tuple(GK_STACK_PARAM_PATHS.keys()) == GK_STACK_KEYS, (
    "GK_STACK_PARAM_PATHS out of sync with GK_STACK_KEYS"
)


@dataclass
class StackedWeightSpec:
    """One leading-``[L, ...]`` stacked weight to load directly on the host.

    Attributes:
        attr: The attribute name to set on the model. ``decode()`` reads it.
        ckpt_keys: The checkpoint keys of ALL layers for this weight, in order.
        loader: A loader that reads those slices on the host and returns the
            ``[L, ...]`` stack (one host->device copy, no per-layer device copy).
        shape: The placeholder parameter shape (meta). ``load_state_dict`` with
            ``assign=True`` replaces the placeholder with the loaded stack.
        dtype: The parameter dtype.
    """

    attr: str
    ckpt_keys: list
    loader: SafetensorsWeightLoader
    shape: tuple
    dtype: torch.dtype


def _gk_fuse_embed_enabled() -> bool:
    """True when the giga-kernel builds layer 0's input from the embedding weight.
    Resolved once at construction, because a trace cannot read the environment.
    """
    return bool(envs.VLLM_NEURON_MEGA_FUSE_EMBED)


def _gk_fuse_embed_hidden_enabled() -> bool:
    """True when the fused embedding shards by the HIDDEN dim instead of the vocab dim.
    Needs `_gk_fuse_embed_enabled`.
    """
    return bool(envs.VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN)


def _gk_fuse_rope_enabled() -> bool:
    """True when the giga-kernel computes the RoPE cos/sin tables on-chip."""
    return bool(envs.VLLM_NEURON_MEGA_FUSE_ROPE)


def _tp_and_ep_meshes(tp_size: int, ep_size: int) -> tuple[tuple, tuple]:
    """Return (tp_mesh, ep_mesh) tuples-of-tuples of GLOBAL ranks.

    The kernel's collectives are:
      - TP: AllGather / ReduceScatter within each DP replica's contiguous
        ``tp_size`` ranks -> mesh = [[0..tp-1], [tp..2tp-1], ...].
      - EP: the full expert mesh over ALL ``ep_size`` ranks -> one group
        [[0, 1, ..., ep_size-1]] (matches the golden's
        ``ep_group = ReplicaGroup([list(range(RANKS))])``).

    For TP8/DP8/EP64: world = ep_size = 64, tp_size = 8, so tp mesh is 8 groups
    of 8 and ep mesh is one group of 64.
    """
    tp_mesh = tuple(
        tuple(range(d * tp_size, (d + 1) * tp_size)) for d in range(ep_size // tp_size)
    )
    ep_mesh = (tuple(range(ep_size)),)
    return tp_mesh, ep_mesh


class MegaDecodeRunner:
    """Assembles inputs for and invokes the GK whole-stack decode kernel.

    Constructed once by ``GptOssModel`` and attached as ``self._mega_decode``.
    ``load_weights`` loads the leading-``[L, ...]`` stacked weights as buffers via
    ``build_stacked_weight_specs``, which asks each weight-holder module for its
    explicit ``gk_weight_loaders``.
    """

    def __init__(self, model):
        self._model = model
        self._logged_invoke = False  # one-shot "[GK] mega decode invoked" marker

    # ── weight stacking ──────────────────────────────────────────────────
    def _assert_layer_invariant_kv_scale(self) -> None:
        """Check that k_scale and v_scale are the same on every layer.

        The kernel takes ONE layer-invariant k_scale/v_scale (and ``decode()``
        folds softmax_scale from layer 0's k_scale). GPT-OSS direct_cast
        checkpoints have k_scale=v_scale=1 on every layer, so this holds. Fail
        fast here for a future per-layer-scale checkpoint instead of silently
        decoding layers 1..L-1 with the wrong (layer 0's) scale.
        """
        layers = self._model.layers
        k0 = layers[0].self_attn.k_scale_float
        v0 = layers[0].self_attn.v_scale_float
        for i in range(1, len(layers)):
            assert layers[i].self_attn.k_scale_float == k0, (
                "GK decode requires a layer-invariant k_scale; layer "
                f"{i} has {layers[i].self_attn.k_scale_float} != layer 0 {k0}."
            )
            assert layers[i].self_attn.v_scale_float == v0, (
                "GK decode requires a layer-invariant v_scale; layer "
                f"{i} has {layers[i].self_attn.v_scale_float} != layer 0 {v0}."
            )

    def _post_transforms(self) -> dict:
        """Return per-stack-key transforms ``fn(tensor, layer) -> tensor``.

        Each transform takes one layer's loaded tensor and reshapes it into the
        kernel's per-layer layout; ``torch.stack`` then adds the leading
        ``[L, ...]`` axis. Kept separate from the loaders (which handle
        sharding / fusion / MXFP4 tiling) so the kernel-layout rules live in one
        place, used by ``build_stacked_weight_specs``.
        """
        # gate_up bias must reach the kernel in the SAME 5-D tiled layout the
        # per-layer decode uses (see model_mxfp4_perf._run_moe_block_tkg): the
        # kernel forwards expert_gate_up_bias straight into moe_block_tkg, which
        # reshapes it against the packed weight's I tiling.
        e0 = self._model.layers[0].mlp.experts
        n_local_experts = e0.num_local_experts
        i_per_partition = e0.intermediate_size_per_rank
        num_i_tp_blocks = math.ceil(i_per_partition / 512.0)
        i_tp_block_size = i_per_partition // num_i_tp_blocks

        def _tiled_gate_up_bias(t, ly):
            return t.view(
                n_local_experts,
                i_tp_block_size // Q_WIDTH,
                2,
                num_i_tp_blocks,
                Q_WIDTH,
            )

        # Decide ONCE and report the DECISION. A log line written before the choice, or one that
        # names a fixed layout, is identical on both paths and witnesses nothing.
        _packs_scale = _kernel_packs_gate_up_scale()
        logger.info(
            "[GK] gate/up MX scale layout: %s", "PACK64" if _packs_scale else "UNPACKED"
        )
        row = lambda t, ly: t.reshape(1, -1)  # noqa: E731
        keep = lambda t, ly: t  # noqa: E731
        return {
            "input_layernorm_weight": row,
            "W_qkv": keep,
            "bias_qkv": row,
            # W_out folds in 1/v_scale (the FP8 dequant scale); no-op at v_scale=1.
            "W_out": lambda t, ly: t / ly.self_attn.v_scale_float,
            "bias_out": row,
            # sinks: model holds [q]; kernel wants [q, 1]
            "sink": lambda t, ly: t.reshape(-1, 1),
            "post_attn_layernorm_weight": row,
            # router weight: model holds [E, H]; kernel contracts [H, E] -> pass .T
            "router_weights": lambda t, ly: t.T.contiguous(),
            "router_bias": row,
            "expert_gate_up_weights": keep,
            "expert_down_weights": keep,
            # Follow the kernel's gate/up scale layout, which it selects at trace time
            # in core/moe/moe_tkg/gate_up_projection_mx.py and which is packed by
            # default. Writing the layout it does not read is silent garbage.
            "expert_gate_up_weights_scale": (
                pack64_gate_up_scale_transform if _packs_scale else keep
            ),
            "expert_down_weights_scale": keep,
            "expert_gate_up_bias": _tiled_gate_up_bias,
            "expert_down_bias": keep,
        }

    def _collect_weight_loaders(self) -> dict:
        """Aggregate the ``{stack_key: (loader, per_layer_shape, dtype)}`` specs
        the weight-holder modules declare via their ``gk_weight_loaders``.

        The loaders are layer-invariant (they depend on model geometry, not the
        layer index), so layer 0's specs drive every layer's slice.
        """
        l0 = self._model.layers[0]
        loaders = {}
        loaders.update(l0.gk_weight_loaders())  # input_layernorm_weight
        loaders.update(l0.self_attn.gk_weight_loaders())  # qkv/o/sink
        loaders.update(l0.mlp.experts.gk_weight_loaders())  # norm/router/experts
        assert set(loaders) == set(GK_STACK_KEYS), (
            f"gk_weight_loaders keys {sorted(loaders)} != "
            f"GK_STACK_KEYS {sorted(GK_STACK_KEYS)}"
        )
        return loaders

    def build_stacked_weight_specs(self, mappings: dict) -> list:
        """Return one StackedWeightSpec per stacked weight.

        Each module declares its stacked weights' loaders via gk_weight_loaders;
        wrap each to read all layers' checkpoint slices (from ``mappings``) and
        return the [L, ...] stack, loaded to the device once. Stacking at load,
        not inside the traced decode(), keeps a per-step re-gather out of the graph.
        """
        self._assert_layer_invariant_kv_scale()
        L = len(self._model.layers)
        posts = self._post_transforms()
        loaders = self._collect_weight_loaders()

        specs = []
        for key in GK_STACK_KEYS:
            loader, per_layer_shape, dtype = loaders[key]
            post = posts[key]
            path = GK_STACK_PARAM_PATHS[key]

            per_layer_keys = []
            for i in range(L):
                ck = mappings[f"model.layers.{i}.{path}"]
                per_layer_keys.append(ck if isinstance(ck, list) else [ck])
            nkeys = len(per_layer_keys[0])
            assert all(len(k) == nkeys for k in per_layer_keys), (
                f"inconsistent per-layer checkpoint key count for {key}"
            )
            combined_keys = [k for keys in per_layer_keys for k in keys]

            # Placeholder shape/dtype for the meta declaration;
            # load_state_dict(assign=True) replaces it with the loaded stack.
            sample = post(
                torch.empty(per_layer_shape, dtype=dtype, device="meta"),
                self._model.layers[0],
            )
            specs.append(
                StackedWeightSpec(
                    attr=GK_STACK_ATTR_PREFIX + key,
                    ckpt_keys=combined_keys,
                    loader=self._make_stacking_loader(loader, post, nkeys),
                    shape=(L, *sample.shape),
                    dtype=dtype,
                )
            )
        return specs

    def _make_stacking_loader(
        self, loader: SafetensorsWeightLoader, post, nkeys: int
    ) -> SafetensorsWeightLoader:
        """Wrap one layer-invariant loader so it stacks all layers into [L, ...].

        Split the slices into one nkeys-sized group per layer, run the loader on
        each, apply the kernel-layout post transform, and stack.
        """
        layers = self._model.layers
        L = len(layers)

        def transform(slices, rank):
            assert len(slices) == nkeys * L, (
                f"stacking loader expected {nkeys * L} slices, got {len(slices)}"
            )
            per_layer = []
            for i in range(L):
                group = slices[i * nkeys : (i + 1) * nkeys]
                # The loader bakes its own rank (with_rank_override / replicated
                # loaders), so the passed rank is ignored.
                t = loader.load(group, rank)
                per_layer.append(post(t, layers[i]))
            return torch.stack(per_layer, dim=0)

        return SafetensorsWeightLoader(transform=transform)

    # ── shared attention control ─────────────────────────────────────────
    def _shared_attn_control(self, positions, position_embeddings, attn_metadata):
        """Build the layer-invariant attention-control tensors.

        Uses a FULL-attention (odd) layer as the representative for the full
        block table + absolute positions, and a SWA (even) layer for the trimmed
        table + window-frame positions, mirroring the
        golden's shared-control construction. Identifying full vs SWA by parity is
        what the kernel bakes in (even = SWA, odd = full).
        """
        model = self._model
        head_dim = model.layers[0].self_attn.head_dim
        sliding_window = model.config.sliding_window

        full_name = "layers.1.self_attn"  # odd -> full attention
        swa_name = "layers.0.self_attn"  # even -> sliding window

        block_size = attn_metadata[full_name]["block_size"]
        full_bt = attn_metadata[full_name]["block_table_tensor"]
        B_local = full_bt.shape[0]
        S_decode = positions.shape[0] // B_local

        half_d = head_dim // 2
        if position_embeddings is None:
            # FUSED RoPE: the kernel computes cos/sin on-chip, so these tables are placeholders.
            # Their dtype sets the on-chip output dtype, which must match the decode compute dtype.
            _dtype = model.layers[0].self_attn.dtype
            cos_kernel = torch.zeros(
                half_d, B_local, S_decode, dtype=_dtype, device=positions.device
            )
            sin_kernel = torch.zeros(
                half_d, B_local, S_decode, dtype=_dtype, device=positions.device
            )
        else:
            cos, sin = position_embeddings
            cos_kernel = (
                cos[:, :half_d]
                .view(B_local, S_decode, half_d)
                .permute(2, 0, 1)
                .contiguous()
            )
            sin_kernel = (
                sin[:, :half_d]
                .view(B_local, S_decode, half_d)
                .permute(2, 0, 1)
                .contiguous()
            )

        # Absolute positions (full/odd layers).
        pos_ids = positions.view(B_local, S_decode).to(torch.float32)

        # Window-frame positions + trimmed SWA table (sliding/even layers). The
        # runner already trimmed the SWA block table and provides the per-seq
        # swa_kv_pos_offset = start_block * block_size.
        swa_bt = attn_metadata[swa_name]["block_table_tensor"]
        swa_kv_pos_offset = attn_metadata[swa_name].get("swa_kv_pos_offset")
        pos_flat = positions.view(1, B_local * S_decode)
        if swa_kv_pos_offset is not None:
            swa_pos = pos_flat - swa_kv_pos_offset.view(B_local, 1).expand(
                B_local, S_decode
            ).reshape(1, B_local * S_decode)
            swa_pos = torch.clamp(swa_pos, min=0)
        else:
            swa_pos = pos_flat
        pos_ids_swa = swa_pos.view(B_local, S_decode).to(torch.float32)

        # Per-layer SWA window starts, stacked [L, B_local, S_decode]:
        #   sliding (even) -> clamp(pos_swa - window + 1, 0)
        #   full   (odd)   -> zeros (full causal)
        L = len(model.layers)
        if sliding_window is not None:
            start_swa = torch.clamp(pos_ids_swa - sliding_window + 1, min=0)
        else:
            start_swa = torch.zeros_like(pos_ids_swa)
        zero_start = torch.zeros_like(pos_ids_swa)
        swa_start_pos_ids = torch.stack(
            [start_swa if (i % 2 == 0) else zero_start for i in range(L)], dim=0
        )

        # Active-token mask (REQUIRED — the kernel dereferences attention_mask.shape
        # unconditionally). With pos_ids supplied the kernel regenerates the
        # causal/SWA band in-kernel; this is only the active-token overlay:
        # tril(ones(S,S)) expanded to [B,N,S,S] then permuted to [S,B,N,S].
        q_heads = model.layers[0].self_attn.num_attention_heads_per_rank
        _s = S_decode
        _causal = torch.tril(
            torch.ones(_s, _s, dtype=torch.float32, device=positions.device)
        )
        attention_mask = (
            _causal.unsqueeze(0)
            .unsqueeze(0)
            .expand(B_local, q_heads, _s, _s)
            .permute(3, 0, 1, 2)  # [S_tkg, B_local, q_heads, S_decode]
            .contiguous()
            .to(torch.uint8)
        )

        return {
            "cos": cos_kernel,
            "sin": sin_kernel,
            "pos_ids": pos_ids,
            "pos_ids_swa": pos_ids_swa,
            "active_blocks_table": full_bt,
            "active_blocks_table_swa": swa_bt,
            "swa_start_pos_ids": swa_start_pos_ids,
            "attention_mask": attention_mask,
            "block_size": block_size,
            "B_local": B_local,
            "S_decode": S_decode,
        }

    # ── entry point ──────────────────────────────────────────────────────
    def decode(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata,
        input_ids=None,
        rank=None,
        return_sp_shard=False,
    ):
        """Run the fused whole-stack decode.

        ``hidden_states`` is the TP-replicated ``[B_local*S_decode, H]`` decode
        tensor (identical across the TP ranks of a DP replica). The kernel wants
        this rank's SP shard ``[B/(DP*TP), H]`` = this rank's TP slice; it does
        the TP AllGather + EP collectives internally and returns the post-stack
        SP shard, which we AllGather[TP] back to ``[B_local*S_decode, H]``.

        With ``VLLM_NEURON_MEGA_FUSE_EMBED`` on, the caller passes ``input_ids`` and
        ``rank`` and NO ``hidden_states``. The kernel then builds layer 0's input from
        this rank's embedding shard, so the host does not compute the embedding and the
        embedded hidden state does not go to HBM.
        """
        model = self._model
        attn0 = model.layers[0].self_attn
        tp_group = model.tp_group
        tp_size = tp_group.world_size

        experts0 = model.layers[0].mlp.experts
        ep_size = experts0.ep_degree

        # Read the pre-stacked weight buffers directly (no per-step re-gather).
        weights = {k: getattr(model, GK_STACK_ATTR_PREFIX + k) for k in GK_STACK_KEYS}
        ctrl = self._shared_attn_control(positions, position_embeddings, attn_metadata)

        # This assembly maps DP to SEPARATE ranks, not in-rank attention/MLP DP:
        # in-rank DP gathers would change the token layout entering this branch,
        # which x_shard's simple 1/tp_size slice does not model. Fail loudly rather
        # than silently shard incorrectly.
        assert attn0.attention_dp_size == 1, (
            "GK decode does not support attention_dp_size > 1 "
            f"(got {attn0.attention_dp_size}); express DP as separate ranks."
        )
        assert experts0.mlp_dp_size == 1, (
            "GK decode does not support mlp_dp_size > 1 "
            f"(got {experts0.mlp_dp_size}); express DP as separate ranks."
        )
        # Read the switches from the model, which resolved them at construction. An environment
        # read inside the trace is re-evaluated or breaks the graph.
        fuse_embed = model._gk_fuse_embed and input_ids is not None
        fuse_hidden = fuse_embed and model._gk_fuse_embed_hidden

        # FUSED RoPE: pass inv_freq plus the concentration scalar and the kernel builds cos/sin
        # on-chip. This recomputes inv_freq on device; the non-persistent buffer reads as NaN.
        fuse_rope = model._gk_fuse_rope
        inv_freq = None
        rope_concentration = 1.0
        if fuse_rope:
            _half_d = attn0.head_dim // 2
            inv_freq = model.rotary_emb.inv_freq_on(positions.device).reshape(
                _half_d, 1
            )
            rope_concentration = model.rotary_emb.concentration_float

        # embed_*: set on the fused-embedding path only (below).
        embed_weight = None
        embed_ids = None
        embed_x_shard_ids = None
        total_tokens = positions.shape[0]
        assert total_tokens % tp_size == 0, (
            f"GK decode expects B_local*S_decode ({total_tokens}) divisible by "
            f"tp_size ({tp_size})."
        )
        shard = total_tokens // tp_size
        # SPMD: derive this rank's SP token rows from the stamped rank tensor. A host-int start
        # bakes a per-rank constant, which splits the compile cache into one NEFF per rank.
        _sp_start = tp_group.rank_in_group_tensor.to(torch.int64) * shard
        _sp_idx = (
            torch.arange(shard, device=positions.device, dtype=torch.int64) + _sp_start
        )

        if fuse_hidden:
            # FUSED EMBEDDING, HIDDEN-dim shard. embed_tokens holds this rank's [vocab, H/tp]
            # column slice. The kernel gathers it, rebuilds the hidden state and slices this SP shard.
            embed_weight = model.embed_tokens.weight  # [vocab, H/tp_size]
            H = model.config.hidden_size
            embed_ids = input_ids.reshape(total_tokens).to(torch.int32)
            # This rank's token rows in the full [total_tokens, H] hidden state, from the
            # stamped rank tensor (the same SPMD form as the x_shard select).
            embed_x_shard_ids = _sp_idx.to(torch.int32)
            x_full = None  # the kernel forms the full layer-0 input
            x_shard = torch.zeros(
                (shard, H), dtype=attn0.dtype, device=positions.device
            )
        elif fuse_embed:
            # FUSED EMBEDDING, VOCAB-dim shard. Pass the LOCAL vocab indices, -1 outside this
            # rank's shard, and a placeholder x_shard.
            embed_weight = model.embed_tokens.weight
            H = embed_weight.shape[1]
            per_rank = model.embed_tokens.vocab_size_per_rank
            vocab_start = rank * per_rank  # rank is this rank's vocab position
            ids = input_ids.reshape(total_tokens).to(torch.int32)
            local = ids - vocab_start.to(torch.int32)
            in_shard = (local >= 0) & (local < per_rank)
            embed_ids = torch.where(in_shard, local, torch.full_like(local, -1))
            x_full = None  # the kernel forms the full layer-0 input
            x_shard = torch.zeros(
                (shard, H), dtype=attn0.dtype, device=positions.device
            )
        else:
            # hidden_states and positions must describe the SAME local token rows.
            assert hidden_states.shape[0] == positions.shape[0], (
                f"GK decode layout mismatch: hidden_states tokens "
                f"{hidden_states.shape[0]} != positions {positions.shape[0]}."
            )
            _, H = hidden_states.shape
            # x_shard holds this rank's [B/(DP*TP), H] SP slice, which layer 0 still needs.
            x_full = hidden_states.to(attn0.dtype)
            x_shard = x_full.index_select(0, _sp_idx)

        B_local = ctrl["B_local"]
        S_decode = ctrl["S_decode"]
        block_size = ctrl["block_size"]

        # kv_cache_update_idx: physical slot per token (kv_heads==1 form). Full
        # (odd) and SWA (even) layers share one group_size-deep buffer but get
        # DISJOINT block IDs, so each KV group has its own slot_mapping; the
        # kernel selects the write index per parity.
        full_slot = attn_metadata["layers.1.self_attn"]["slot_mapping"]
        swa_slot = attn_metadata["layers.0.self_attn"]["slot_mapping"]
        kv_cache_update_idx = full_slot.view(B_local, S_decode).to(torch.uint32)
        kv_cache_update_idx_swa = swa_slot.view(B_local, S_decode).to(torch.uint32)

        tp_mesh, ep_mesh = _tp_and_ep_meshes(tp_size, ep_size)

        # active-token mask (REQUIRED, built in _shared_attn_control — the kernel
        # dereferences attention_mask.shape unconditionally).
        attention_mask = ctrl["attention_mask"]

        # SPMD: read the [[ep_rank]] kernel input from the EP group's
        # rank_in_group_tensor so it is a graph input, not a baked per-rank
        # constant — see GptOssExperts._ep_rank_id for why. At ep_size == 1 the
        # EP group is never initialized (get_neuron_ep_group() asserts).
        if ep_size > 1:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_group,
            )

            rank_id = (
                get_neuron_ep_group().rank_in_group_tensor.reshape(1, 1).to(torch.int32)
            )
        else:
            # Read the device from positions, not hidden_states, because the fused-embedding path
            # passes no hidden_states.
            rank_id = torch.zeros((1, 1), dtype=torch.int32, device=positions.device)

        # FP8 scale fusing (matches the per-layer attention_decode convention):
        # softmax_scale carries 1/k_scale here; the stacked W_out was pre-divided
        # by v_scale (the W_out transform in _post_transforms). k_scale/v_scale
        # tensors are also passed so the kernel dequants the FP8 cache internally.
        # Today's direct_cast checkpoints have k_scale=v_scale=1, so all folds
        # are no-ops.
        softmax_scale = attn0.scaling / attn0.k_scale_float
        k_scale = attn0.k_scale
        v_scale = attn0.v_scale

        # Confirm ONCE that the GK path (not the per-layer fallback) is running.
        # Guard on is_dynamo_compiling(): a logger.info inside the traced region
        # breaks the parallel-trace fork.
        if not torch.compiler.is_dynamo_compiling() and not self._logged_invoke:
            self._logged_invoke = True
            logger.info(
                "[GK] mega decode invoked: L=%d tp=%d ep=%d B_local=%d "
                "S_decode=%d fp8_packed=%s",
                len(model.layers),
                tp_size,
                ep_size,
                B_local,
                S_decode,
                bool(attn0.fp8_packed),
            )

        # Read the per-group buffers from individual attrs so the fork-child
        # meta-swap tracks them. The meta-swap skips a list attribute.
        _n = model._gk_kv_buf_count
        _k_bufs = [getattr(model, f"K_cache_buf_{i}") for i in range(_n)]
        _v_bufs = [getattr(model, f"V_cache_buf_{i}") for i in range(_n)]

        out_shard = NF.gpt_oss_mega_decode(
            x_shard,
            input_layernorm_weight=weights["input_layernorm_weight"],
            W_qkv=weights["W_qkv"],
            bias_qkv=weights["bias_qkv"],
            W_out=weights["W_out"],
            bias_out=weights["bias_out"],
            sink=weights["sink"],
            K_cache=None,
            V_cache=None,
            k_bufs=_k_bufs,
            v_bufs=_v_bufs,
            post_attn_layernorm_weight=weights["post_attn_layernorm_weight"],
            router_weights=weights["router_weights"],
            router_bias=weights["router_bias"],
            expert_gate_up_weights=weights["expert_gate_up_weights"],
            expert_down_weights=weights["expert_down_weights"],
            expert_gate_up_weights_scale=weights["expert_gate_up_weights_scale"],
            expert_down_weights_scale=weights["expert_down_weights_scale"],
            expert_gate_up_bias=weights["expert_gate_up_bias"],
            expert_down_bias=weights["expert_down_bias"],
            swa_start_pos_ids=ctrl["swa_start_pos_ids"],
            cos=ctrl["cos"],
            sin=ctrl["sin"],
            active_blocks_table=ctrl["active_blocks_table"],
            active_blocks_table_swa=ctrl["active_blocks_table_swa"],
            attention_mask=attention_mask,
            kv_cache_update_idx=kv_cache_update_idx,
            pos_ids=ctrl["pos_ids"],
            pos_ids_swa=ctrl["pos_ids_swa"],
            rank_id=rank_id,
            tp_replica_ranks=tp_mesh,
            ep_replica_ranks=ep_mesh,
            num_layers=len(model.layers),
            tp_size=tp_size,
            ep_size=ep_size,
            # B_attn = sequences per DP replica, S_tkg = decode tokens/seq.
            # tokens-per-replica = B_attn * S_tkg, sharded across TP into x_shard.
            B_attn=B_local,
            S_tkg=S_decode,
            top_k=experts0.num_experts_per_token,
            eps=attn0.rms_norm_eps,
            hidden_actual=attn0.unpadded_hidden_size,
            softmax_scale=softmax_scale,
            gate_clamp_upper_limit=experts0.limit,
            up_clamp_upper_limit=experts0.limit + 1,
            up_clamp_lower_limit=-experts0.limit + 1,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_cache_update_idx_swa=kv_cache_update_idx_swa,
            # Full TP-replicated first-layer input -> kernel skips layer 0's
            # redundant input AllGather.
            x_first_full=x_full,
            # FUSED EMBEDDING: the kernel gathers the embedded hidden state and forms layer 0's
            # input. All None unless the fusion is on. embed_x_shard_ids is hidden-dim shard only.
            embed_weight=embed_weight,
            embed_ids=embed_ids,
            embed_x_shard_ids=embed_x_shard_ids,
            # FUSED RoPE: the kernel computes cos/sin on-chip from pos_ids plus
            # inv_freq. None (the default) keeps the external cos/sin path.
            inv_freq=inv_freq,
            rope_concentration=rope_concentration,
        )
        # Kernel may return (x, K, V) under in-kernel KV update; take the shard.
        if isinstance(out_shard, (tuple, list)):
            out_shard = out_shard[0]

        # SP shard -> TP-replicated [B_local*S_decode, H].
        # return_sp_shard keeps the SP shard instead, for a caller that gathers the rows itself
        # over a wider group. One flat gather over the lm_head TP group replaces two hops.
        if return_sp_shard:
            return out_shard
        if tp_size > 1:
            out_shard = tp_group.all_gather(out_shard, dim=0)
        return out_shard
