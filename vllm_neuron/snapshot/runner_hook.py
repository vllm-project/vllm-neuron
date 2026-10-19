# SPDX-License-Identifier: Apache-2.0
"""Runner-side wiring for NRT-boundary input-snapshot capture.

Shared by the V1 and V2 model runners. The wiring has two halves, both scoped
to a single model forward:

- publish the forward's identity and capture verdict
  (``SnapshotForwardContext``) for the compiled executables' capture specs to
  read, and
- install the lite ``ExecuteContext`` whose ``pre_execute`` hook copies the
  selected forward's inputs to host.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from vllm_neuron import envs

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from libtorch_neuronx_lite.compile.execute_context import ExecuteContext


def check_v2_capture_config(vllm_config: VllmConfig) -> None:
    """Fail fast when capture is enabled with a config the V2 wiring cannot honor.

    The V2 capture call passes ``spec_window=1``, which is only correct while
    speculative decoding is absent. A config that reaches the V2 runner with a
    draft window would silently skew token-rule matching, so reject it at
    startup instead: whoever removes the upstream routing gate must wire the
    real window here first.
    """
    if not envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
        return
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise NotImplementedError(
            "input-snapshot capture on the V2 runner does not wire the "
            "speculative draft window (spec_window); disable capture or run "
            "without speculative decoding"
        )


class SnapshotRunnerHook:
    """Per-runner holder of the shared capture ``ExecuteContext``.

    The context wraps one shared ``SnapshotCapturer`` (its ``pre_execute`` is
    the ``pre_execute_hook``), built once so the capturer's per-NEFF call
    index persists across forwards. It is (re)set before every captured
    forward and cleared after.
    """

    def __init__(self) -> None:
        self._execute_context: Optional[ExecuteContext] = None

    @contextmanager
    def capture_context(
        self,
        *,
        req_ids: Sequence[str],
        positions: torch.Tensor,
        is_decode: bool,
        spec_window: int = 1,
    ) -> Iterator[None]:
        """Scope input-snapshot capture to one model forward.

        Capture is opt-in debug, so this is a zero-cost no-op when disabled.
        When enabled, this forward's identity + capture verdict and the lite
        capture hook are set immediately before the forward and cleared
        immediately after — so capture is scoped to this model's executes and
        not, e.g., a draft or vision model's, and a raised forward can't leave
        a stale context tagging the next one.

        Args:
            req_ids: Request ids of this forward, in scheduler order.
            positions: Flat positions tensor for this forward.
            is_decode: Whether the batch is wholly decode.
            spec_window: Tokens per request row (draft window + 1); 1 without
                speculative decoding.
        """
        if not envs.VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE:
            yield
            return

        from libtorch_neuronx_lite.compile.execute_context import (
            clear_execute_context,
        )

        from vllm_neuron.snapshot.context import clear_current_forward

        try:
            self._publish_forward(req_ids, positions, is_decode, spec_window)
            self._set_execute_context()
            yield
        finally:
            clear_current_forward()
            clear_execute_context()

    def _publish_forward(
        self,
        req_ids: Sequence[str],
        positions: torch.Tensor,
        is_decode: bool,
        spec_window: int,
    ) -> None:
        """Resolve and publish this forward's input-snapshot context.

        The token/request verdict is the same for every NEFF in a forward, so
        it is evaluated once here — above the compiled boundary, where
        positions and request ids are still semantic — and published for the
        executables to read. Fails loudly: capture is opt-in debug, so a
        malformed selection or a failure computing the verdict aborts the
        forward rather than silently producing no snapshot.
        """
        from vllm_neuron.snapshot.config import get_snapshot_config
        from vllm_neuron.snapshot.context import (
            SnapshotForwardContext,
            next_global_step,
            set_current_forward,
        )

        # Process-wide config, resolved once and shared with the executables'
        # capture specs; the selector is constant for the run.
        selector = get_snapshot_config().selector

        req_ids = list(req_ids)
        # The positions host-sync is only worth paying when a token/request
        # rule needs the values; call-index-only capture skips it.
        if selector.has_value_rule():
            pos_list = positions.detach().to("cpu").reshape(-1).tolist()
            capture, matches = selector.evaluate_forward(
                req_ids, pos_list, is_decode, spec_window
            )
        else:
            pos_list = []
            capture, matches = (False, [])

        set_current_forward(
            SnapshotForwardContext(
                global_step=next_global_step(),
                is_prompt=not is_decode,
                capture=capture,
                req_ids=req_ids,
                positions=pos_list,
                matches=matches,
            )
        )

    def _set_execute_context(self) -> None:
        """Install the lite execute context that runs input-snapshot capture."""
        from libtorch_neuronx_lite.compile.execute_context import (
            ExecuteContext,
            set_execute_context,
        )

        if self._execute_context is None:
            from vllm_neuron.snapshot.capture import SnapshotCapturer

            self._execute_context = ExecuteContext(
                pre_execute_hook=SnapshotCapturer().pre_execute
            )
        set_execute_context(self._execute_context)
