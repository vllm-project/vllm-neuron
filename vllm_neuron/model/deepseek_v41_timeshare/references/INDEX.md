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
    - Local directory: ~/workspace/partition_artifact
    - modules/ contain the pytorch reference of each module
    - trace/ contains the traced golden input and output for the text in https://github.com/OswaldHe/TranSonic/tree/main/partition/inputs
    
When there is a bug, consider vLLM > original huggingface code > partition reference for the level of truth.

### Weights
---
Local: ~/workspace/partition_artifact/hf. Do not redownload from huggingface.

### Past Kernel Optimization Memory
---
- DeepSeek V4.1 Flash Trainium Kernels: https://github.com/OswaldHe/DeepSeek-V4.1-Flash-Trainium
    - Local path: ~/workspace/DeepSeek-V4.1-Flash-Trainium
    - Each folder is a module
        - `source.py` and `inference.py`: bootstrapped, unoptimized kernel + launcher
        - `opt-*`: optimized kernel folder, contain `source.py` and `inference.py`
        - `opt-rank0`: rank0 subkernel optimized
        - `FEEDBACK.md`: experiences and bug reports
        - `feedback-repro`: minimal reproducible scripts for FEEDBACK.md
        - `reviews-*` and `notes-*`: autohelix notes and reviews at each iteration.

### Local Environment (use uv)
- ~/workspace/.venv: venv for Neuron SDK 2.32.0 for standard NKI development we used previously
- ~/workspace/vllm-neuron/.venv: venv for vLLM Neuron with more components and features

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