# SPDX-License-Identifier: Apache-2.0
import logging

import torch
from torch.distributed import ProcessGroup

from vllm_neuron.functional.sampling import sample
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig

logger = logging.getLogger(__name__)


class Sampler(torch.nn.Module):
    """Sampler for on-device token generation.

    Wraps functional sampling API with stateful configuration. Automatically
    uses data-parallel sampling when initialized via init_sampling_dp_groups().

    Args:
        sampling_config: Configuration with all_greedy, max_top_k, deterministic,
            and sampling_dp_degree settings.
        process_group: TP process group for distributed operations. None for
            single-device inference.

    Example:
        >>> from vllm_neuron.nn import Sampler
        >>> sampler = Sampler(sampling_config, process_group=tp_group)
        >>> tokens = sampler(logits, sampling_params)
    """

    def __init__(
        self,
        sampling_config: "OnDeviceSamplingConfig",  # type: ignore
        process_group: "ProcessGroup | None" = None,  # type: ignore
    ) -> None:
        super().__init__()
        self.all_greedy = sampling_config.all_greedy
        self.max_top_k = sampling_config.max_top_k
        self.deterministic = sampling_config.deterministic
        self.capture_topk = sampling_config.capture_topk
        self.process_group = process_group

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_sampling_all2all_group,
            get_neuron_sampling_dp_degree,
            get_neuron_sampling_gather_group,
        )

        if get_neuron_sampling_dp_degree() > 1:
            self.all2all_group = get_neuron_sampling_all2all_group().device_group
            self.gather_group = get_neuron_sampling_gather_group().device_group
        else:
            self.all2all_group = None
            self.gather_group = None

    def forward(
        self,
        logits: torch.Tensor,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        tp_rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from logits with per-request sampling parameters.

        Automatically uses batch-sharded topk when DP groups are initialized
        and batch size is divisible by dp_degree.

        Args:
            logits: shape [batch_size, vocab_size] - already selected at sampling positions
            sampling_params: shape [batch_size, 3] - stacked [top_k, top_p, temperature].
                If None, uses sample() defaults (top_k=-1, top_p=1.0, temperature=1.0).
            logit_mask: Optional boolean mask for structured output. Shape [batch_size, vocab_size].
                True = token allowed, False = token disallowed.
            tp_rank: TP rank for sharding full-vocab logit_mask to local vocab shard.
                Must be a tensor to avoid rank becoming a traced constant in
                the compiled path.

        Returns:
            Token indices of shape [batch_size].

        Example:
            >>> from vllm_neuron.nn.sampler import Sampler
            >>> from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
            >>> config = OnDeviceSamplingConfig(all_greedy=True, max_top_k=1)
            >>> sampler = Sampler(config)
            >>> logits = torch.randn(2, 1000)  # batch=2, vocab=1000
            >>> tokens = sampler(logits)  # shape [2]
            >>> # With structured output (constrained decoding):
            >>> mask = torch.ones(2, 1000, dtype=torch.bool)
            >>> mask[:, 500:] = False  # Only allow tokens 0-499
            >>> tokens = sampler(logits, logit_mask=mask)
        """
        # Apply grammar mask for structured output (constrained decoding)
        if logit_mask is not None:
            # Shard full-vocab mask to local vocab shard (matches lm_head sharding)
            if (
                self.process_group is not None
                and logit_mask.shape[-1] != logits.shape[-1]
            ):
                if tp_rank is None:
                    raise ValueError(
                        "tp_rank must be provided when logit_mask requires TP sharding "
                        "(mask width != logits width). Pass the model's rank tensor."
                    )
                shard_size = logits.shape[-1]
                full_mask_width = logit_mask.shape[-1]
                if full_mask_width % shard_size != 0:
                    raise ValueError(
                        "Full logit_mask width must be divisible by local shard width. "
                        f"Got full_mask_width={full_mask_width}, shard_size={shard_size}."
                    )

                if not isinstance(tp_rank, torch.Tensor):
                    raise ValueError(
                        "tp_rank must be a torch.Tensor when TP sharding is required."
                    )
                # Skip device= to avoid triggering cache normalization node rewrite
                tp_rank_tensor = tp_rank.to(dtype=torch.int64)
                if tp_rank_tensor.numel() != 1:
                    raise ValueError(
                        "tp_rank tensor must contain exactly one element. "
                        f"Got shape={tuple(tp_rank.shape)}."
                    )
                # Build indices as tensors so torch.compile doesn't extract scalars
                shard_indices = (
                    torch.arange(
                        shard_size,
                        device=logit_mask.device,
                        dtype=torch.int64,
                    )
                    + tp_rank_tensor * shard_size
                )
                logit_mask = torch.index_select(logit_mask, dim=-1, index=shard_indices)
            # Set disallowed tokens to -inf so they can't be sampled
            logits = logits.masked_fill(~logit_mask, float("-inf"))

        if sampling_params is None:
            return sample(
                logits,
                deterministic=self.deterministic,
                all_greedy=self.all_greedy,
                tp_group=self.process_group,
                max_top_k=self.max_top_k,
                all2all_group=self.all2all_group,
                gather_group=self.gather_group,
                tp_rank=tp_rank,
                capture_topk=self.capture_topk,
            )

        top_k = sampling_params[:, 0].to(torch.int32)
        top_p = sampling_params[:, 1]
        temperature = sampling_params[:, 2]

        return sample(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            deterministic=self.deterministic,
            all_greedy=self.all_greedy,
            tp_group=self.process_group,
            max_top_k=self.max_top_k,
            all2all_group=self.all2all_group,
            gather_group=self.gather_group,
            tp_rank=tp_rank,
            capture_topk=self.capture_topk,
        )
