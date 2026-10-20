# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class NeuronDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ):
        # vLLM 0.27 DeviceCommunicatorBase grew global_ranks, global_world_size,
        # and use_all2all; GroupCoordinator now passes use_all2all to every
        # device communicator (vllm/distributed/parallel_state.py). Accept and
        # forward the full base signature. The base's use_all2all decision is
        # then overridden below with Neuron semantics.
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
            use_all2all=use_all2all,
        )

        # Override all2all enablement, since vLLM's --all2all-backend Literal does
        # not include 'neuron'. NeuronAll2AllManager should be mounted to Neuron's
        # custom EP group, not vLLM's default EP group.
        self.is_neuron_ep_communicator = unique_name.startswith("neuron_ep:")
        self.all2all_backend = self._read_neuron_all2all_backend()
        self.use_all2all = (
            self.is_neuron_ep_communicator and self.all2all_backend == "neuron"
        )

        if self.use_all2all:
            from vllm_neuron.parallel.all2all import NeuronAll2AllManager

            self.all2all_manager = NeuronAll2AllManager(self.cpu_group)
            logger.info(
                "Initialized NeuronAll2AllManager with rank %s, local_rank %s",
                self.global_rank,
                self.rank,
            )

    @staticmethod
    def _read_neuron_all2all_backend() -> str | None:
        """Return additional_config.neuron_config.all2all_backend, or None."""
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None or config.additional_config is None:
            return None
        neuron_config = config.additional_config.get("neuron_config", {}) or {}
        return neuron_config.get("all2all_backend")
