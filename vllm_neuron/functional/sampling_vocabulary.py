# SPDX-License-Identifier: Apache-2.0
"""Static tokenizer restrictions for full or tensor-parallel vocabulary logits."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingVocabulary:
    size: int | None = None
    suppressed: tuple[int, ...] = ()

    def __post_init__(self):
        if self.size is not None and self.size <= 0:
            raise ValueError("sampling_vocab_size must be positive")
        ids = tuple(sorted(set(self.suppressed)))
        if any(token < 0 for token in ids):
            raise ValueError("suppressed_token_ids must be nonnegative")
        if (
            self.size is not None
            and sum(token < self.size for token in ids) == self.size
        ):
            raise ValueError("The sampling vocabulary must contain at least one token")
        object.__setattr__(self, "suppressed", ids)

    def mask(
        self, logits: torch.Tensor, vocab_size: int, tp_rank: torch.Tensor | None
    ) -> torch.Tensor:
        limit = self.size if self.size is not None else vocab_size
        if limit > vocab_size:
            raise ValueError("sampling_vocab_size exceeds the model vocabulary")
        if limit == vocab_size and not self.suppressed:
            return logits
        if self.size is None and sum(i < limit for i in self.suppressed) == limit:
            raise ValueError("The sampling vocabulary must contain at least one token")
        width = logits.shape[-1]
        ids = torch.arange(width, dtype=torch.int32, device=logits.device)
        if width != vocab_size:
            if tp_rank is None or width > vocab_size or vocab_size % width:
                raise ValueError("Expected evenly sharded logits and a TP rank tensor")
            ids = ids + tp_rank * width
        allowed = ids < limit
        for token in self.suppressed:
            allowed = allowed & (ids != token)
        return logits.masked_fill(~allowed, -torch.inf)
