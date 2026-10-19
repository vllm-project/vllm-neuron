# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.distributed.parallel_state import get_node_count, get_world_group

from vllm_neuron import envs
from libtorch_neuronx_lite.compile.platform import get_platform_target
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_group,
    get_neuron_ep_tp_group,
    get_neuron_intranode_ep_group,
    get_neuron_internode_ep_group,
    get_node_group,
    shard_select,
)

from ..functional.moe.permute_routed_tokens import _bitcast

logger = init_logger(__name__)

_SUPPORTED_DISPATCH_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16]


class NeuronAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NKI EP kernels, for 2D Torus and NeuronSwitch topologies.

    2D Torus: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
    NeuronSwitch: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

        # Collective topology
        self.is_neuron_switch = envs.VLLM_NEURON_SWITCH_CC
        self.is_hierarchical = get_node_count() > 1 and not self.is_neuron_switch
        self._validate_topology()

        # Dispatch/combine algorithm selection
        self._dispatch_func = (
            self._dispatch_hierarchical
            if self.is_hierarchical
            else self._dispatch_local
        )
        self._combine_func = (
            self._combine_hierarchical if self.is_hierarchical else self._combine_local
        )

        # Collective dims
        self.num_tokens = None
        self.num_local_experts = None
        self.num_experts_per_tok = None
        self.num_dispatch_elements_per_tok = None

        # Dispatch metadata
        self.hierarchical_dispatch_recv_tokens = None  # EFA A2A-v
        self.dispatch_recv_tokens = None  # Intra-server or intra-switch A2A-v
        self.tp_gather_recv_tokens = None  # TP AG-v

        # Dispatch/combine debug data/metadata capture
        # FIXME: refactor debug capture
        self._debug_dispatch_intra_recv_buffer = None
        self._debug_dispatch_inter_recv_buffer = None
        self._debug_dispatch_intra_metadata = None
        self._debug_dispatch_inter_metadata = None
        self._debug_dispatch_tp_gather_metadata = None
        self._debug_dispatch_intra_send_buffer = None
        self._debug_dispatch_inter_send_buffer = None
        self._debug_combine_intra_recv_tokens = None
        self._debug_combine_inter_recv_tokens = None
        self._debug_combine_intra_recv_buffer = None
        self._debug_combine_inter_recv_buffer = None
        self._debug_combine_intra_metadata = None
        self._debug_combine_inter_metadata = None
        self._debug_combine_intra_send_buffer = None
        self._debug_combine_inter_send_buffer = None

    def _validate_topology(self):
        # NeuronSwitch and Hierarchical all2all are mutually exclusive
        assert not (self.is_hierarchical and self.is_neuron_switch), (
            f"NeuronAll2AllManager requires NeuronSwitch (VLLM_NEURON_SWITCH_CC=1) or 2D Torus with multiple servers, but got both: {envs.VLLM_NEURON_SWITCH_CC=}, {get_node_count()=}. Unset VLLM_NEURON_SWITCH_CC to use hierarchical all2all."
        )

        # Topology validation
        if self.is_hierarchical:
            # FIXME: remove this once we get the prod-ready 1rpd A2A-v
            assert os.environ.get("NEURON_RT_A2AV_ONE_RANK_PER_CHIP") == "1", (
                "NEURON_RT_A2AV_ONE_RANK_PER_CHIP=1 must be set for hierarchical all2all"
            )
            # FIXME[P451411063]: remove check once compiler dependency is fixed. Turning off early posting results in an extra inter-node handshake.
            assert os.environ.get("NEURON_RT_DBG_EARLY_POSTING") == "0", (
                "NEURON_RT_DBG_EARLY_POSTING=0 must be set for hierarchical all2all"
            )
        elif self.is_neuron_switch:
            assert "trn3" in get_platform_target(), (
                f"NeuronAll2AllManager with NeuronSwitch requires Trn3+, got {get_platform_target()=}"
            )
        else:
            raise ValueError(
                f"NeuronAll2AllManager requires NeuronSwitch (VLLM_NEURON_SWITCH_CC=1) or 2D Torus with multiple servers. Got {envs.VLLM_NEURON_SWITCH_CC=}, {get_node_count()=}"
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        """
        EP dispatch with All2All-v. Each token is routed to ranks that contain its top-K experts.

        FIXME[pr]: explain fused EP + TP options

        Args:
            hidden_states (torch.Tensor): [T, H] bf16 or fp8 hidden states.
            topk_weights (torch.Tensor): [T, E] bf16 expert affinities, with zeros for non-routed pairs.
            topk_ids (torch.Tensor): [T, K] int32 expert indices.
            is_sequence_parallel (bool): Whether dispatch() is being invoked with sequence parallel sharding.
                Currently, only is_sequence_parallel=True is supported. The token-permute
                derives its per-rank token-id offset from the on-device
                ``get_world_group().rank_in_group_tensor`` (stamped by
                ``stamp_spmd_rank_tensors``), so the traced graph is identical
                across ranks (SPMD).

        Returns:
            torch.Tensor: [T * world_size, num_dispatch_elements_per_tok], where each row is [hidden | affinities | tok_idx].

        Example:
            >>> dispatched = mgr.dispatch(hidden, topk_weights, topk_ids, is_sequence_parallel=True)
            >>> # Output buffer is statically shaped [T * world_size, num_dispatch_elements_per_tok];
            >>> # rows from each src rank are scattered into the buffer at fixed offsets of T * src_rank,
            >>> # i.e. src_rank's contributed rows occupy [src_rank * T : src_rank * T + dispatch_recv_tokens[src_rank]).
            >>> # The remaining rows in each src's slot (up to T) are zero-padded. Per row layout (in dispatch_dtype):
            >>> #   [0 : H)                          -> hidden state
            >>> #   [H : H + n_local_experts)        -> per-local-expert affinities (bf16, bitcast)
            >>> #   [H + n_local_experts : ...)      -> source token index (int32, bitcast)
        """

        assert is_sequence_parallel, (
            f"NeuronAll2AllManager.dispatch() requires is_sequence_parallel=True but got {is_sequence_parallel=}"
        )

        # FIXME[cc bug]: convert all inputs to bf16, since native fp8 collective is inaccurate
        original_dtype = hidden_states.dtype
        hidden_states = _bitcast(hidden_states, torch.bfloat16)

        self._init_dispatch_combine_dims(hidden_states, topk_weights, topk_ids)

        # Execute communication algorithm based on topology
        # FIXME[cc bug]: convert back to original dtype
        return _bitcast(
            self._dispatch_func(
                hidden_states, topk_weights, topk_ids, is_sequence_parallel
            ),
            original_dtype,
        )

    def _init_dispatch_combine_dims(self, hidden_states, topk_weights, topk_ids):
        """Compute T, E, K, dispatch elements per tok for use during dispatch/combine; validate sizes."""
        self.num_tokens, H = hidden_states.shape
        _, self.num_local_experts = topk_weights.shape
        _, self.num_experts_per_tok = topk_ids.shape

        assert self.num_local_experts % get_neuron_ep_group().world_size == 0, (
            f"Expected num_local_experts divisible by all2all group size, got E={self.num_local_experts}, all2all group size={get_neuron_ep_group().world_size}"
        )
        assert topk_weights.dtype == torch.bfloat16, (
            f"all2all only supports topk_weights.dtype == torch.bfloat16, got {topk_weights.dtype=}"
        )
        assert hidden_states.dtype in _SUPPORTED_DISPATCH_DTYPES, (
            f"all2all dispatch only supports hidden_states.dtype in {_SUPPORTED_DISPATCH_DTYPES}, got {hidden_states.dtype=}"
        )

        # adj_factor = 1 when hidden is bf16, 2 when fp8.
        bf16_per_int32 = bf16_bytes = 2
        adj = bf16_bytes // hidden_states.element_size()

        # Hierarchical carries full E affinities (no slicing); local slices to E/EP.
        affinity_cols = (
            self.num_local_experts
            if self.is_hierarchical
            else self.num_local_experts // get_neuron_ep_group().world_size
        )
        self.num_dispatch_elements_per_tok = int(
            H + (affinity_cols + bf16_per_int32) * adj
        )
        # Hierarchical inter-server stage packs an extra 2*K expert-idx cols.
        self.num_dispatch_elements_per_tok_hierarchical = (
            self.num_dispatch_elements_per_tok
            + self.num_experts_per_tok * bf16_per_int32 * adj
        )

    def _dispatch_local(
        self, hidden_states, topk_weights, topk_ids, is_sequence_parallel
    ):
        """
        Input size: [T * K, num_dispatch_elements_per_tok]
        Output size: [T * all2all_group_size, num_dispatch_elements_per_tok]

        permute_routed_tokens reads the per-rank token-id offset from the stamped
        ``get_world_group().rank_in_group_tensor`` (SPMD).
        """

        assert get_neuron_ep_tp_group().world_size == 1, (
            "Fused EP+TP communication not yet supported for NeuronSwitch"
        )

        # Build all2all-v metadata, permute by dst rank
        metadata = NF.build_all2all_dispatch_metadata(
            expert_index=topk_ids,
            num_experts=self.num_local_experts,
            num_elements_per_token=self.num_dispatch_elements_per_tok,
            group=get_neuron_ep_group(),
            recv_displs=None,
        )
        data = NF.permute_routed_tokens(
            hidden_input=hidden_states,
            expert_index=topk_ids,
            expert_affinities_masked=topk_weights,
            group=get_neuron_ep_group(),
            is_sequence_parallel=is_sequence_parallel,
        )

        # Output must be initialized with zeros
        output_recv = torch.zeros(
            (
                int(self.num_tokens * get_neuron_ep_group().world_size),
                int(self.num_dispatch_elements_per_tok),
            ),
            dtype=data.dtype,
            device=data.device,
        )

        # Execute all2all-v. Dispatch collective computes recv_counts for reuse during combine(), and does not use rdispls.
        # FIXME[P451411063]: cc_use_intermediate_io=True is required for the
        # destination memset in all_to_all_v to take effect.
        output_recv, metadata = NF.all_to_all_v(
            input=data,
            output=output_recv,
            group=get_neuron_ep_group(),
            metadata=metadata,
            recv_counts_known=False,
            has_rdispls=False,
            cc_use_intermediate_io=True,
        )

        # Save dispatch metadata, for reuse during combine()
        self.dispatch_recv_tokens = metadata[2, :] // self.num_dispatch_elements_per_tok

        return output_recv

    def _dispatch_hierarchical(
        self, hidden_states, topk_weights, topk_ids, is_sequence_parallel
    ):
        """
        Inter-server input size: []
        Intra-server input size: []
        Output size: []

        NOTE: need to distinguish between whether we are fusing EP+TP or not here

        permute_routed_tokens reads the per-rank token-id offset from the stamped
        ``get_world_group().rank_in_group_tensor`` (SPMD).
        """

        # Groups
        internode_group = get_neuron_internode_ep_group()
        intranode_group = get_neuron_intranode_ep_group()
        tp_group = get_neuron_ep_tp_group()
        assert tp_group.world_size == 4, (
            f"Hierarchical EP+TP communication requires TP=4, got {tp_group.world_size=}"
        )

        # Build all2all-v metadata, permute by dst node
        metadata_hierarchical = NF.build_all2all_dispatch_metadata(
            expert_index=topk_ids,
            num_experts=self.num_local_experts,
            num_elements_per_token=self.num_dispatch_elements_per_tok_hierarchical,
            group=internode_group,
            recv_displs=None,
        )
        # Inter-server stage routes by node; carry full E affinities (no slicing).
        data_hierarchical = NF.permute_routed_tokens(
            hidden_input=hidden_states,
            expert_index=topk_ids,
            expert_affinities_masked=topk_weights,
            group=internode_group,
            is_sequence_parallel=is_sequence_parallel,
            pack_expert_index=True,
            affinity_slice_size=self.num_local_experts,
        )

        # Output must be initialized with zeros
        output_recv_hierarchical = torch.zeros(
            (
                int(self.num_tokens * internode_group.world_size),
                int(self.num_dispatch_elements_per_tok),
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        neg_one_index = _bitcast(
            torch.full(
                (
                    int(self.num_tokens * internode_group.world_size),
                    int(self.num_experts_per_tok),
                ),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            ),
            hidden_states.dtype,
        )

        output_recv_hierarchical = torch.concat(
            [output_recv_hierarchical, neg_one_index], dim=1
        )
        self._debug_dispatch_inter_send_buffer = data_hierarchical.detach().clone()

        # Execute all2all-v. Dispatch collective computes recv_counts for reuse during combine(), and does not use rdispls.
        output_recv_hierarchical, metadata_hierarchical = NF.all_to_all_v(
            input=data_hierarchical,
            output=output_recv_hierarchical,
            group=internode_group,
            metadata=metadata_hierarchical,
            recv_counts_known=False,
            has_rdispls=False,
        )
        self._debug_dispatch_inter_recv_buffer = (
            output_recv_hierarchical.detach().clone()
        )
        self._debug_dispatch_inter_metadata = metadata_hierarchical.detach().clone()

        # Save dispatch metadata, for reuse during combine()
        self.hierarchical_dispatch_recv_tokens = (
            metadata_hierarchical[2, :]
            // self.num_dispatch_elements_per_tok_hierarchical
        )

        # Build all2all-v metadata, permute by dst rank
        expert_index_recv = _bitcast(
            output_recv_hierarchical[
                :, self.num_dispatch_elements_per_tok :
            ].contiguous(),
            torch.int32,
        )
        metadata = NF.build_all2all_dispatch_metadata(
            expert_index=expert_index_recv,
            num_experts=self.num_local_experts,
            num_elements_per_token=self.num_dispatch_elements_per_tok,
            group=intranode_group,
            recv_displs=None,
            num_experts_per_node=self.num_local_experts // get_node_count(),
        )
        data = NF.hierarchical_all2all_dispatch_permute(
            internode_dispatch_output=output_recv_hierarchical[
                :, : self.num_dispatch_elements_per_tok
            ].contiguous(),
            expert_index=expert_index_recv,
            num_experts_per_node=self.num_local_experts // get_node_count(),
            local_ep_group=intranode_group,
        )

        # Output must be initialized with zeros, shape: [T * EP, num_dispatch_elem]
        output_recv = torch.zeros(
            (
                int(self.num_tokens * get_neuron_ep_group().world_size),
                int(self.num_dispatch_elements_per_tok),
            ),
            dtype=data.dtype,
            device=data.device,
        )
        self._debug_dispatch_intra_send_buffer = data.detach().clone()

        # Execute all2all-v. Dispatch collective computes recv_counts for reuse during combine(), and does not use rdispls.
        output_recv, metadata = NF.all_to_all_v(
            input=data,
            output=output_recv,
            group=intranode_group,
            metadata=metadata,
            recv_counts_known=False,
            has_rdispls=False,
        )
        self._debug_dispatch_intra_recv_buffer = output_recv.detach().clone()
        self._debug_dispatch_intra_metadata = metadata.detach().clone()

        # Save dispatch metadata, for reuse during combine()
        self.dispatch_recv_tokens = metadata[2, :] // self.num_dispatch_elements_per_tok

        # Build AG-v metadata, convert sparse A2A-v output to packed buffer of [tok | pad, :] so that AG-v input is contiguous
        tp_metadata = NF.build_all_gatherv_metadata(
            all2all_recv_counts=metadata[2, :],
            group=tp_group,
        )
        output_recv_packed = NF.pack_tokens(output_recv)

        # Output must be initialized with zeros, shape: [T*world, H + E + 2]
        output_final = torch.zeros(
            (
                int(self.num_tokens * get_world_group().world_size),
                int(self.num_dispatch_elements_per_tok),
            ),
            dtype=output_recv_packed.dtype,
            device=output_recv_packed.device,
        )

        # Execute AG-v for TP gather. Dispatch collective computes recv_counts for reuse during combine(), and does not use rdispls.
        output_final, tp_metadata = NF.all_gather_v(
            input=output_recv_packed,
            output=output_final,
            group=tp_group,
            metadata=tp_metadata,
            recv_counts_known=False,
            has_rdispls=False,
        )

        # Save gather metadata, for reuse during combine()
        self.tp_gather_recv_tokens = (
            tp_metadata[2, :] // self.num_dispatch_elements_per_tok
        )
        self._debug_dispatch_tp_gather_metadata = tp_metadata.detach().clone()

        return output_final

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ):
        """
        EP combine with All2All-v.

        Each token is returned to its dispatch source rank, and partial outputs are reduced. Requires dispatch() to have run first to populate collective metadata.

        Args:
            hidden_states (torch.Tensor): [T * all2all_group_size, H + 2] bf16 expert outputs,
                with the last 2 columns containing int32 token indices.
            is_sequence_parallel (bool): Whether dispatch() is being invoked with sequence parallel sharding.
                Currently, only is_sequence_parallel=True is supported. The topk
                reduce derives its per-rank token-id offset from the stamped
                ``get_world_group().rank_in_group_tensor``, and the hierarchical
                combine-reduce from ``get_node_group().rank_in_group_tensor``, so
                the traced graph is identical across ranks (SPMD).
        Returns:
            torch.Tensor: [T, H] bf16.

        Example:
            >>> reduced = mgr.combine(expert_out, is_sequence_parallel=True)
            >>> # Output buffer is statically shaped [T, H] bf16; rows correspond 1:1 to the
            >>> # source-rank tokens that were dispatched.
        """

        assert is_sequence_parallel, (
            f"NeuronAll2AllManager.combine() requires is_sequence_parallel=True but got {is_sequence_parallel=}"
        )

        assert all(
            v is not None
            for v in (
                self.num_tokens,
                self.num_local_experts,
                self.num_experts_per_tok,
                self.num_dispatch_elements_per_tok,
                self.dispatch_recv_tokens,
            )
        ), (
            "NeuronAll2AllManager.combine() requires dispatch() to run first to populate metadata."
        )

        # Execute communication algorithm based on topology
        return self._combine_func(hidden_states, is_sequence_parallel)

    def _combine_local(self, hidden_states, is_sequence_parallel):
        """
        Input size: [T * all2all_group_size, H + 2]
        Output size: [T * all2all_group_size, H + 2]

        topk_reduce derives the per-rank token-id offset (rank * T) from the
        stamped ``get_world_group().rank_in_group_tensor`` (SPMD), keeping the
        traced graph identical across ranks.
        """

        assert get_neuron_ep_tp_group().world_size == 1, (
            "Fused EP+TP communication not yet supported for NeuronSwitch"
        )

        # Compute combine send counts using recv counts saved during dispatch; build all2all-v metadata
        combine_send_counts = (
            self.dispatch_recv_tokens.to(torch.int32) * hidden_states.shape[1]
        )
        metadata = NF.build_all2all_combine_metadata(
            send_counts=combine_send_counts,
            recv_displs=None,
        )

        # Output must be initialized with zeros
        output_recv = torch.zeros(
            (
                self.num_tokens * get_neuron_ep_group().world_size,
                hidden_states.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Execute all2all-v. Combine collective does not compute recv_counts or use rdispls.
        # FIXME[P451411063]: cc_use_intermediate_io=True is required for the
        # destination memset in all_to_all_v to take effect.
        output_recv, metadata = NF.all_to_all_v(
            input=hidden_states,
            output=output_recv,
            group=get_neuron_ep_group(),
            metadata=metadata,
            recv_counts_known=True,
            has_rdispls=False,
            cc_use_intermediate_io=True,
        )

        # Reduce across top-K, return to original token ordering
        output = NF.topk_reduce(
            input=output_recv,
            T=self.num_tokens,
            K=self.num_experts_per_tok,  # TODO: remove usage of K arg; this will be deprecated.
            is_sequence_parallel=is_sequence_parallel,
        )
        return output

    def _combine_hierarchical(self, hidden_states, is_sequence_parallel):
        """
        Intra-server input size:
        Inter-server input size:
        Output size:

        NOTE: need to distinguish between whether we are fusing EP+TP or not here

        The hierarchical combine-reduce derives its per-rank token-id base from
        the stamped ``get_node_group().rank_in_group_tensor`` (SPMD).
        """

        # Groups
        internode_group = get_neuron_internode_ep_group()
        intranode_group = get_neuron_intranode_ep_group()
        tp_group = get_neuron_ep_tp_group()
        assert tp_group.world_size == 4, (
            f"Hierarchical EP+TP communication requires TP=4, got {tp_group.world_size=}"
        )
        assert self.hierarchical_dispatch_recv_tokens is not None, (
            "expected hierarchical_dispatch_recv_tokens populated by hierarchical dispatch, got None"
        )

        # Step 1: TP ReduceScatter
        # TODO: switch dense RS for RS-v
        # We want to reduce over the hidden portion, and preserve the unreduced indices
        input_tokens, H_concat = hidden_states.shape
        local_tokens = input_tokens // tp_group.world_size
        local_indices = shard_select(hidden_states, 0, tp_group)[:, -2:]

        output_tp = tp_group.reduce_scatter(hidden_states[:, : H_concat - 2], dim=0)
        output_tp = torch.concat([output_tp, local_indices], dim=1)

        # Compute combine send counts using recv counts saved during dispatch; build all2all-v metadata
        combine_send_counts = (
            self.dispatch_recv_tokens.to(torch.int32) * output_tp.shape[1]
        )
        metadata = NF.build_all2all_combine_metadata(
            send_counts=combine_send_counts,
            recv_displs=None,
        )

        # Output must be initialized with zeros
        output_recv = torch.zeros(
            (
                self.num_tokens * get_neuron_ep_group().world_size,
                output_tp.shape[1],
            ),
            dtype=output_tp.dtype,
            device=output_tp.device,
        )
        self._debug_combine_intra_send_buffer = output_tp

        # Execute all2all-v. Combine collective does not compute recv_counts or use rdispls.
        output_recv, metadata = NF.all_to_all_v(
            input=output_tp,
            output=output_recv,
            group=intranode_group,
            metadata=metadata,
            recv_counts_known=True,
            has_rdispls=False,
        )
        self._debug_combine_intra_recv_tokens = metadata[2, :] // hidden_states.shape[1]
        self._debug_combine_intra_recv_buffer = output_recv
        self._debug_combine_intra_metadata = metadata

        # Compute combine send counts using recv counts saved during dispatch; build all2all-v metadata
        combine_send_counts = (
            self.hierarchical_dispatch_recv_tokens.to(torch.int32)
            * hidden_states.shape[1]
        )
        metadata_hierarchical = NF.build_all2all_combine_metadata(
            send_counts=combine_send_counts,
            recv_displs=None,
        )
        # Reduce across top-K, permute for inter-server all2all-v.
        # SPMD: token_base_index is a runtime tensor (rank_in_group_tensor * T + 1)
        # read from the stamped node group, threaded end-to-end through
        # hierarchical_all2all_combine_reduce -> _hierarchical_all2all_combine_reduce_nki
        # -> _topk_reduce_nki (which loads it into SBUF and broadcast-adds it),
        # so the traced graph is identical across ranks — no baked
        # get_node_group().rank_in_group constant.
        output_node = NF.hierarchical_all2all_combine_reduce(
            input=output_recv,
            # T_max represents the maximum number of unique global token ids, not the maximum id.
            T_max=self.num_tokens * get_neuron_internode_ep_group().world_size,
            # base index: intra-server rank * T/world + 1 (tok ids are 1-indexed)
            token_base_index=get_node_group().rank_in_group_tensor * self.num_tokens
            + 1,
            # stride: ranks/server
            token_group_stride=get_node_group().world_size,
            # group size: T/world
            token_group_size=self.num_tokens,
        )

        # Output must be initialized with zeros
        output_recv_hierarchical = torch.zeros(
            (
                self.num_tokens * internode_group.world_size,
                hidden_states.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._debug_combine_inter_send_buffer = output_node

        # Execute all2all-v. Combine collective does not compute recv_counts or use rdispls.
        output_recv_hierarchical, metadata_hierarchical = NF.all_to_all_v(
            input=output_node,
            output=output_recv_hierarchical,
            group=internode_group,
            metadata=metadata_hierarchical,
            recv_counts_known=True,
            has_rdispls=False,
        )
        self._debug_combine_inter_recv_tokens = (
            metadata_hierarchical[2, :] // hidden_states.shape[1]
        )
        self._debug_combine_inter_recv_buffer = output_recv_hierarchical
        self._debug_combine_inter_metadata = metadata_hierarchical

        # Reduce across top-K, return to original token ordering
        output = NF.topk_reduce(
            input=output_recv_hierarchical,
            T=self.num_tokens,
            K=self.num_experts_per_tok,  # TODO: remove usage of K arg; this will be deprecated.
            is_sequence_parallel=is_sequence_parallel,
        )

        return output
