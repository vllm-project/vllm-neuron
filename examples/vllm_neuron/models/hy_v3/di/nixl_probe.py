"""Probe: can NIXL (LIBFABRIC) register Neuron HBM and transfer between two cores without EFA?"""

import sys, time
import torch
import vllm_neuron  # noqa: F401  (registers neuron device)
from nixl._api import nixl_agent, nixl_agent_config

backend = sys.argv[1] if len(sys.argv) > 1 else "LIBFABRIC"
mem = sys.argv[2] if len(sys.argv) > 2 else "VRAM"
dev_a, dev_b = ("neuron:0", "neuron:1") if mem == "VRAM" else ("cpu", "cpu")

src = torch.arange(1 << 20, dtype=torch.float32).to(dev_a)
dst = torch.zeros(1 << 20, dtype=torch.float32).to(dev_b)
print(
    "tensors on",
    src.device,
    dst.device,
    "ptrs",
    hex(src.data_ptr()),
    hex(dst.data_ptr()),
    flush=True,
)

a = nixl_agent("A", nixl_agent_config(backends=[backend]))
b = nixl_agent("B", nixl_agent_config(backends=[backend]))
print("backends ok", flush=True)
reg_a = a.register_memory(
    [(src.data_ptr(), src.numel() * 4, 0 if mem == "VRAM" else 0, "")], mem
)
reg_b = b.register_memory(
    [(dst.data_ptr(), dst.numel() * 4, 1 if mem == "VRAM" else 0, "")], mem
)
print("registered", flush=True)
a.add_remote_agent(b.get_agent_metadata())
local = a.get_xfer_descs([(src.data_ptr(), src.numel() * 4, 0)], mem)
remote = b.get_xfer_descs(
    [(dst.data_ptr(), dst.numel() * 4, 1 if mem == "VRAM" else 0)], mem
)
h = a.initialize_xfer("WRITE", local, remote, "B", b"done")
st = a.transfer(h)
t0 = time.time()
while a.check_xfer_state(h) not in ("DONE", "ERR") and time.time() - t0 < 30:
    time.sleep(0.01)
print("xfer state", a.check_xfer_state(h), flush=True)
print("match", torch.equal(src.cpu(), dst.cpu()), flush=True)
