# SPDX-License-Identifier: Apache-2.0
"""EC connector model-runner mixin for the Neuron runner.

Subclasses upstream's ECConnectorModelRunnerMixin and adds the one orchestration
hook from pending upstream PR #40695: the pre-gather completion
barrier (maybe_wait_for_ec_load). The consumer (PD) pulls vision embeddings from
a remote VE asynchronously, so the runner must block until the READ lands before
the prefill gather reads those blocks.

This is a thin convergence shim. When vllm #40695 lands delete this subclass
and have the runner inherit ECConnectorModelRunnerMixin directly.
"""

from __future__ import annotations

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.v1.worker.ec_connector_model_runner_mixin import (
    ECConnectorModelRunnerMixin,
)


class ECLoadFailure(RuntimeError):
    """Raised when a consumer's encoder-cache READ failed to land."""

    def __init__(self, failed_mm_hashes: set[str]):
        self.failed_mm_hashes = failed_mm_hashes
        super().__init__(
            f"Encoder-cache load failed for mm_hashes: {sorted(failed_mm_hashes)}"
        )


class EncoderCacheReadError(RuntimeError):
    """Raised in the monolithic path when a scheduled vision item cannot be read.

    When the scheduler admitted and scheduled the item, its embedding must be resident
    and correctly sized by the time the prefill gather runs. A miss or a size mismatch
    here means a bug (or corruption), not a recoverable transport condition — so it fails
    loud rather than letting the compiled model run on unfilled placeholder positions and
    return a silently wrong answer.

    Args:
        missed: ``(mm_hash, reason)`` pairs for every item that could not be read.
        context: Optional human-readable diagnostic context (req_id, scheduling
            window, cache-state snapshot) appended to the message.
    """

    def __init__(self, missed: list[tuple[str, str]], context: str = ""):
        self.missed = missed
        detail = "; ".join(f"{h}: {reason}" for h, reason in missed)
        msg = f"Encoder-cache read failed for scheduled vision item(s): {detail}"
        if context:
            msg = f"{msg}\n{context}"
        super().__init__(msg)


class NeuronECConnectorModelRunnerMixin(ECConnectorModelRunnerMixin):
    """EC mixin for NeuronModelRunner: inherits upstream's EC hooks, adds the wait.

    Upstream's hooks (save/get-output) are used as-is via the get_ec_transfer()
    global; only the pre-gather wait below is added.
    """

    @staticmethod
    def maybe_wait_for_ec_load() -> set[str]:
        """Block until the consumer's encoder-cache READs land; return failures.

        TODO(epd-transport): drop this override and inherit it once #40695 lands.
        """
        if not has_ec_transfer():
            return set()
        connector = get_ec_transfer()
        return connector.wait_for_load()
