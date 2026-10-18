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
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

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

    # ---- point-to-point, for pipeline-parallel activation handoff ----------
    #
    # `GroupCoordinator.send`/`recv` dispatch here, and `send_tensor_dict` /
    # `recv_tensor_dict` reach `torch.distributed.isend`/`irecv` on
    # `self.device_group` directly. Both end up on the same transport, because
    # **every** process group this backend builds — world, TP, PP, EP and the
    # one called `device_group` — is created with `backend = "gloo"`
    # (`neuron_parallel_state.py:510`). gloo moves host tensors only.
    #
    # There is no device-side point-to-point to lower to either: the
    # `_c10d_functional` overrides that make a collective run on the device
    # ship in `libtorch_neuronx_lite/overrides/xla_collectives.py` and cover
    # exactly four ops — `all_reduce`, `all_gather_into_tensor`,
    # `reduce_scatter_tensor`, `all_to_all_single`. Nothing point-to-point.
    #
    # So a device tensor is staged through host DRAM here, and that is the
    # honest cost of the transport: a stage boundary carries the rank-4
    # hyper-connection stream, `hc_mult(4) x hidden(5120)` bf16 = 41 kB a
    # token, and pays a device->host->device round trip for it. It is enough
    # to *measure* a pipeline; it is not where a pipeline should end up. The
    # on-device route is `nki.collectives.collective_permute(srcs, dsts,
    # source_target_pairs)` inside a kernel — the same mechanism
    # `kernels/legacy/*.py` already use for `all_reduce` — which keeps the
    # handoff on the NeuronLink and never touches the host.
    #
    # Nothing calls these yet: `vllm_neuron` has no `IntermediateTensors`, so
    # there is no activation to hand off. They are here so the next step is
    # plumbing rather than transport.

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Send `tensor` to `dst` (a rank *in this group*), blocking.

        Staged through host DRAM when `tensor` is on the device; see above.
        """
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        staged = tensor if tensor.device.type == "cpu" else tensor.cpu()
        torch.distributed.send(staged, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receive a `size`/`dtype` tensor from `src` (a rank in this group).

        Returned on `self.device`, having landed in host DRAM first.
        """
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        staged = torch.empty(size, dtype=dtype, device="cpu")
        torch.distributed.recv(staged, self.ranks[src], self.device_group)
        if self.device is None or self.device.type == "cpu":
            return staged
        return staged.to(self.device)

    @staticmethod
    def _read_neuron_all2all_backend() -> str | None:
        """Return additional_config.neuron_config.all2all_backend, or None."""
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None or config.additional_config is None:
            return None
        neuron_config = config.additional_config.get("neuron_config", {}) or {}
        return neuron_config.get("all2all_backend")
