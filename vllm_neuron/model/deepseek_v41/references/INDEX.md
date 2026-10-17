### GPU References
---
- Original DeepSeek V4.1 Flash Inference Code (DSpark Speculative Decoding logics are excluded): https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference
    - Please only check the part for text generation
    - `generate.py`: entry point, standard autoregressive decoding
    - `kernel.py`: optimized TileLang kernels

- vLLM DeepSeek V4.1 Flash Implementation: https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v41
    - This will include the DSpark Speculative decoding logics.
    - More optimized version on GPU

- Partitioned Reference: https://huggingface.co/datasets/OswaldHe123/deepseek-v4.1-flash-partition
    - Local directory: `/mnt/nvme/partition_artifact` (private repo; needs an HF token)
    - modules/ contain the pytorch reference of each module
    - trace/ contains the traced golden input and output for the text in https://github.com/OswaldHe/TranSonic/tree/main/partition/inputs
      (locally: `/home/ubuntu/TranSonic/partition/inputs`)
    
When there is a bug, consider vLLM > original huggingface code > partition reference for the level of truth.

### Weights
---
Local: `/mnt/nvme/hf` — the whole 475.3 GiB checkpoint, 48 shards, revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. Do not redownload from huggingface.

**It is on the instance store, which does not survive a stop/start.** The root
volume is not an alternative: measured 67.6 MB/s read against the instance
store's 2.6 GB/s, so a checkpoint there would cost ~2 h per full pass. If
`/mnt/nvme` is empty after a restart, remount it and re-pull — `hf download
deepseek-ai/DeepSeek-V4.1-Flash --local-dir /mnt/nvme/hf` took ~3 minutes at
~2.5 GB/s (`~/download_ckpt.sh`, and `~/download_partition.sh` for the
partition reference).

### Past Kernel Optimization Memory
---
- DeepSeek V4.1 Flash Trainium Kernels: https://github.com/OswaldHe/DeepSeek-V4.1-Flash-Trainium
    - Local path: `/home/ubuntu/DeepSeek-V4.1-Flash-Trainium`
    - Each folder is a module
        - `source.py` and `inference.py`: bootstrapped, unoptimized kernel + launcher
        - `opt-*`: optimized kernel folder, contain `source.py` and `inference.py`
        - `opt-rank0`: rank0 subkernel optimized
        - `FEEDBACK.md`: experiences and bug reports
        - `feedback-repro`: minimal reproducible scripts for FEEDBACK.md
        - `reviews-*` and `notes-*`: autohelix notes and reviews at each iteration.

### Local Environment (use uv)
- `/home/ubuntu/vllm-neuron/.venv`: venv for vLLM Neuron — the one this port runs in.
  Python 3.12.3, vllm 0.24.0, vllm-neuron 0.24.0.1.1.0, torch/torch-xla 2.11.0,
  neuronx-cc 2.27.5334.0, nki 0.6.0, transformers 5.19.0, islpy 2026.1 (already
  pinned to the version `warn_on_unsupported_islpy` wants), pyarrow 26.
- `/home/ubuntu/TranSonic/.venv`: the AutoHelix harness (`autohelix` CLI).
- `source ~/ds41-env.sh` first: it carries the checkpoint path, the compile-cache
  and temp directories (both off the slow root volume) and the compile-worker
  counts this host can afford.
- The standalone NKI venv the old host had at `~/workspace/.venv` does not exist
  here. Kernel work runs in the vllm-neuron venv, which has `nki` and `neuronx-cc`.
- Benign warning at every import, do not chase it: `libtorch_neuronx_lite._compiler`
  fails to load with `undefined symbol: PyObject_ClearManagedDict` (its `_C.so` in
  build 2651 is compiled against Python 3.13). It only disables the *vendored*
  in-process compiler; graphs compile through the `neuronx-cc` subprocess, which is
  the path `put_bundled_compiler_on_path` exists to repair. The DLAMI's own blessed
  venv for this release ships build 1284, which has no `_compiler` directory at all,
  so it runs the same path.

### This Host (trn2.48xlarge, measured 2026-10-17)
---
16 Trainium2 devices, 64 logical NeuronCores at LNC=2, 96 GiB HBM a device /
24 GiB a logical core, 1536 GiB of HBM in total; 192 vCPUs, 2 TB RAM.
`floorplan/systems/trn2-16device.yaml` describes it and is the deployment target
the floorplan was already written against.

Four facts that differ from the trn2.3xlarge the port was built on, each of which
invalidates a calibration in the code:

| | trn2.3xlarge (assumed in code) | this host |
|---|---|---|
| devices / logical cores | 1 / 4 | 16 / 64 |
| HBM | 96 GiB | 1536 GiB — the 475 GiB checkpoint fits |
| host `CommitLimit` | 114.8 GiB, `vm.overcommit_memory=2` | 1000 GiB, **`overcommit_memory=0` and the guard file is absent** |
| fast volume | one 430 GiB instance store, partial stage | 1.8 TB at `/mnt/nvme`, whole checkpoint |

### Website/Documentations/Code Repo Index

- vLLM Neuron (vLLM for Trainium devices): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html
    - Serving quickstart: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/quickstart-online-serving.html
    - Profiling guideline: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/how-to-profile-workloads.html
    - Configuration options: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/reference-configuration.html
    - Onboard a new model (IMPORTANT, MUST CHECK): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-dev/onboarding-models.html
    - Model Optimization (no need to worry the vision part in our case): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-dev/optimizing-vlm-models.html
    - Debugger usage: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-dev/debugging.html
    - Debug Accuracy issue (MUST CHECK WHEN ACCURACY ISSUE OCCURS): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-dev/accuracy-debugging-guide.html
    - vLLM neuron repo: https://github.com/vllm-project/vllm-neuron
        - Accuracy debugging utilities: https://github.com/vllm-project/vllm-neuron/tree/release-0.24.0.1.1.0/vllm_neuron/accuracy

- A nice blog about optimizing DeepSeek V4.1 Flash on GPU from vLLM: https://vllm.ai/blog/2026-10-07-deepseek-v41-flash

- AWS Neuron SDK Doc: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html
    - NKI: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html
        - Trainium 2 Design Guide: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html
        - Performance Optimization: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html
            - Can also use `neuron-nki-*` subagents
        - All API references: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/index.html
    - Neuron Runtime (for Host-Device management): https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/index.html

### Kernels -> Reference Naming Correspondance:
- dspark_attn -> layer-40-dsparkattn
- dspark_confhead -> layer-42-dspark-confhead
- dspark_markov -> layer-42-dspark-markovhead
- dspark_moe -> layer-40-moe
- engram_long -> engram-long
- engram_short -> engram-short
- hc_attn_out -> hc-sublayer-out-short
- hc_collapse -> hc-collapse
- hc_expand -> hcexpand
- hc_attn_in_short -> hc-sublayer-in-short
- hc_attn_in_long -> hc-sublater-in-long
- l2_attn -> layer-2-attention-long
- l3_attn -> layer-0-attention-b
- l20_attn -> layer-20-attention-long
- l21_attn -> layer-0-attention-c
- l24_attn -> layer-24-attention-long
- moe_long -> layer-1-moe-long
- moe_short -> layer-1-moe