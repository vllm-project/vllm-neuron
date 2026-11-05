# SPDX-License-Identifier: Apache-2.0
"""Neuron weight-transfer engine (trainer side).

Symmetric to :mod:`vllm_neuron.vllm.weight_transfer.neuron_engine`, but lives in
the trainer process. Rank 0 opens the model-update store and drives the
inference side through the :class:`~vllm.distributed.weight_transfer.base.VLLMWeightSyncClient`
control plane; every rank runs its own trainer-side work so a multi-rank
(FSDP) trainer stays aligned.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import ClassVar

import torch
from typing_extensions import Self

from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.logger import init_logger

from vllm_neuron.vllm.weight_transfer.neuron_engine import (
    NeuronWeightTransferInitInfo,
    NeuronWeightTransferUpdateInfo,
)

logger = init_logger(__name__)


@dataclass
class NeuronTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the Neuron weight-transfer backend.

    ``rank`` (from :class:`TrainerInitInfo`) identifies this trainer process;
    rank 0 is the sender. ``backend`` is the factory dispatch key.
    """

    backend: ClassVar[str] = "neuron"

    master_address: str
    master_port: int
    world_size: int


class NeuronTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[NeuronTrainerInitInfo]
):
    """Trainer-side engine: opens the model-update group as rank 0 and
    broadcasts each parameter to the inference workers over the store."""

    init_info_cls = NeuronTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self._group = None

    @classmethod
    def trainer_init(
        cls,
        init_info: NeuronTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError(
                "Neuron trainer weight transfer requires a WeightSource."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
        )
        if not engine.is_sender:
            # Non-sender trainer ranks aren't part of the transfer group and
            # don't drive the inference side; they only replay the source
            # iteration during send_weights to stay in any trainer collective.
            return engine

        from vllm.distributed.utils import StatelessProcessGroup

        # Workers sit at rank_offset 1, after the single trainer sender rank 0.
        worker_init_info = NeuronWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
        )

        # The inference workers block inside init_weight_transfer_engine waiting
        # for the rendezvous, so kick that off on a side thread while we open the
        # trainer endpoint (rank 0); both sides must rendezvous together.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine,
                asdict(worker_init_info),
            )
            engine._group = StatelessProcessGroup.create(
                host=init_info.master_address,
                port=init_info.master_port,
                rank=0,
                world_size=init_info.world_size,
            )
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self) -> None:
        """Push weights to the inference workers and drive the full round trip."""
        assert self.source is not None  # guaranteed by trainer_init / __init__
        source = self.source

        # Metadata is declared without gathering. Only the sender ships it; it
        # defines the per-parameter order the workers receive in.
        meta = source.metadata()

        if not self.is_sender:
            for _ in source:
                pass
            return

        update_info = NeuronWeightTransferUpdateInfo(
            names=[m.name for m in meta],
            dtype_names=[str(m.dtype).split(".")[-1] for m in meta],
            shapes=[list(m.shape) for m in meta],
        )

        self.client.start_weight_update()
        # update_weights (workers receive) must run concurrently with the
        # trainer-side broadcast — both rendezvous inside the same store reads.
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(self.client.update_weights, asdict(update_info))
            # Cheap best-effort: surface an already-failed update_weights now
            # instead of hanging in a broadcast for a peer that will never read.
            if future.done():
                future.result()
            self._broadcast(source, meta)
            future.result()  # surface inference-side errors
        finally:
            # Never wait for the RPC thread here: if the broadcast raised, the
            # workers are still blocked in their matching store reads and will
            # never return, so joining would turn the error into a permanent
            # hang. Let the exception out instead.
            exe.shutdown(wait=False)
        self.client.finish_weight_update()

    def _broadcast(self, source: WeightSource, meta: list[ParamMeta]) -> None:
        """Iterate the source and broadcast each tensor from rank 0.

        The store transport pickles each tensor, so it is materialized to host
        memory first (a Neuron device tensor is not picklable, and a CUDA tensor
        pickles via an implicit host round-trip).
        """
        assert self._group is not None, (
            "trainer_init() must be called before _broadcast()."
        )
        for _name, tensor in self._checked_iter(source, meta):
            host = tensor.detach()
            if host.device.type != "cpu":
                host = host.to("cpu")
            self._group.broadcast(host, src=0)

    @staticmethod
    def _checked_iter(
        source: WeightSource, meta: list[ParamMeta]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the source's pairs, checking each against what the workers were
        told to expect.

        The workers receive in ``meta`` order; if iteration disagrees with it,
        the two sides split the stream differently and the transfer hangs or
        loads garbage. Sender-only: under pipeline parallelism a non-sender's
        yielded tensor is not meaningful.
        """
        sent = 0
        for name, tensor in source:
            if sent >= len(meta):
                raise ValueError(
                    f"WeightSource yielded more parameters than metadata() "
                    f"declared ({len(meta)}); first extra is {name!r}."
                )
            expected = meta[sent]
            if (
                name != expected.name
                or tensor.dtype != expected.dtype
                or tuple(tensor.shape) != expected.shape
            ):
                raise ValueError(
                    "WeightSource metadata() disagrees with iteration at index "
                    f"{sent}: declared {expected.name!r} "
                    f"{expected.dtype} {tuple(expected.shape)}, got {name!r} "
                    f"{tensor.dtype} {tuple(tensor.shape)}. Both channels must "
                    "enumerate the same parameters in the same order."
                )
            sent += 1
            yield name, tensor
        if sent != len(meta):
            raise ValueError(
                f"WeightSource yielded {sent} parameters but metadata() "
                f"declared {len(meta)}; the workers are waiting for the rest."
            )

    def shutdown(self) -> None:
        """Release the model-update group. Default no-op otherwise."""
        self._group = None
