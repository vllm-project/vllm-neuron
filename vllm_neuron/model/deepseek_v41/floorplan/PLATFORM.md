# The platform: trn2-16device

trn2-16device: 16 device(s), 64 logical NeuronCore(s) at LNC=2, 24 GiB per bank, 1536 GiB total, 4x4 torus

Generated from `systems/` by `floorplan/driver.py`. The simulator reads the same
file, so this cannot drift from what your floorplan will actually be costed against.

## Addressing

A placement names one or more units as `d<device>.l<logical_nc>`:

- devices `d0` .. `d15`
- logical NeuronCores `l0` .. `l3` per device
- **64 units in total**

Doc terminology, which is worth getting right because the informal usage is inverted:
a *logical NeuronCore* is the group of two, and the two cores inside it are *physical
NeuronCore-v3*. At LNC=2 the runtime dispatches to the pair, so the pair is
the placement grain. `.p0`/`.p1` may be appended to say how the pair divides one
placement's work; it is not a separate level of the hierarchy.

## Per logical NeuronCore

| HBM bank | **24.0 GiB** — not 96 GiB; that is the device's four banks |
|---|---|
| bank bandwidth | 725 GB/s |
| SBUF | 56.0 MiB (the pair's, not each core's) |
| PSUM | 2.0 MiB |
| physical cores | 2 |
| engines | dma, gpsimd, scalar, tensor, vector |

## Peak compute, per device

| dtype | TFLOPS | achieved fraction (probed) |
|---|---|---|
| fp8 | 1299 | 0.181 |
| bf16 | 667 | 0.410 |
| fp32 | 181 | 0.410 |

Divide by four for one logical NeuronCore. The achieved fraction is what the
simulator actually uses; `matmul_small_k` is separate and much lower, which is the
single most important cost-model fact for MoE.

## Memory tiers

Capacity is enforced at every tier, per scope. Nearest first:

| tier | scope | capacity | bandwidth | latency | notes |
|---|---|---|---|---|---|
| `hbm_bank` | logical_nc | 24.0 GiB | 725.0 GB/s | 0.0 us |  |
| `device_hbm` | device | 96.0 GiB | 2900.0 GB/s | 0.0 us |  |
| `peer_hbm` | instance | 1.5 TiB | 1024.0 GB/s | 1.0 us |  |
| `host_dram` | instance | 2.0 TiB | 13.0 GB/s | 2.0 us |  |
| `nvme` | instance | 7.0 TiB | 9.1 GB/s | 29.740476747974753 us | reached through host_dram; 132866 IOPS |
| `ebs` | instance | — | 10 GB/s | — | **load time only** — may not be read during a forward pass |
| `network` | instance | — | 400 GB/s | — | **load time only** — may not be read during a forward pass |

## Interconnect

Topology: **4x4 torus**

Hop counts are not uniform, and a parallel group's shape on the torus is part of
the design:

```
  d0->d0: 0  d0->d1: 1  d0->d2: 2  d0->d3: 1
  d0->d4: 1  d0->d5: 2  d0->d6: 3  d0->d7: 2
  d0->d8: 2  d0->d9: 3  d0->d10: 4  d0->d11: 3
  d0->d12: 1  d0->d13: 2  d0->d14: 3  d0->d15: 2
```

- intra-device (core to core in one device): 803 GB/s, 1.0 us
- inter-device (over the torus): 1024 GB/s, 1.0 us per hop

## Hardware features and constraints

Prose, because these are not numbers. Each is implemented somewhere in `sim/` and
cited by number; several of them decide whole strategies rather than trimming a
percent.

1. GPSIMD and the Tensor Engine cannot access SBUF simultaneously on Trainium 2. A
   gather feeding a matmul therefore serializes against it on the same logical core
   rather than overlapping, which matters most for MoE expert gathers and for Engram
   n-gram lookups.

2. At LNC=2 the two physical NeuronCore-v3 of a logical core share one address space.
   Splitting a module across the pair costs no collective — the halves read each
   other's SBUF-resident results directly — but it does not add SBUF capacity, since
   the 56 MiB is the pair's, not each core's.

3. A logical NeuronCore reaches its own 24 GiB HBM bank at full bandwidth. Reaching
   another bank on the same device stays on-chip but contends for the device's shared
   2.9 TB/s.

4. Collectives run on the 16 CC-Cores and can overlap with compute, but a collective
   and the compute it feeds cannot overlap with *each other* — the dependency is real.
   Overlap has to be declared in the floorplan to be credited.

5. The 4x4 torus is not uniform. A device has 4 nearest neighbours; the farthest device
   is 4 hops away. Tensor parallelism spanning devices that are not neighbours pays the
   hop count on every collective, so a TP group's shape on the torus is part of the
   design, not an afterthought.

6. There is no direct accelerator-to-NVMe path. Every byte from NVMe crosses host DRAM
   and then PCIe.

7. fp8 weights with block scales are the checkpoint's storage format. Compute is bf16.
   A matmul therefore costs a dequantization pass in addition to the GEMM, and the
   weight's HBM footprint is the fp8 size while its SBUF working set during compute is
   larger.

8. Both Engram tables are ~94.56 GiB of embeddings that a token touches only a few rows
   of (`engram_max_ngram_size: 4`, `engram_n_heads: 8`). Residency is a genuine choice:
   in HBM they cost 12% of the instance's memory for an almost entirely cold table, and
   out of HBM every lookup pays a tier crossing on the critical path of layers 1 and 14.

9. The MoE layers route 6 of 384 experts per token. At batch 1 decode, a step touches a
   small and data-dependent subset of 277 GiB of expert weights, so expert placement
   interacts with the all-to-all and with whether experts are resident at all.


## Provenance

Which numbers are measured and which are extrapolated, from `systems/probed.yaml`:

- **extrapolated, not measured:** `links.inter_device.bandwidth_bytes_per_s`
- **extrapolated, not measured:** `links.inter_device.per_hop_latency_us`
- **extrapolated, not measured:** `collectives wider than the 4 logical cores of one device`

this host has one Trainium2 device; every inter-device figure stays at its datasheet value and is an extrapolation

- **`links.intra_device` is derived, not measured** (device HBM bandwidth (2.9e+12 B/s) x measured dma_large_contiguous (0.2770)). NOT MEASURED. NKI 0.6.0 has no collective primitive and the XLA all-reduce fails to compile (NCC_ISMP902), so no intra-device collective can be timed on this host. Bounded by the on-chip path per text constraint 3.

