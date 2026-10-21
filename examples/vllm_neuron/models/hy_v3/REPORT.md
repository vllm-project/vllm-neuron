# Hy3-FP8 在 vLLM Neuron 上的接入与测试报告

日期：2026-10-16　｜　硬件：trn2.48xlarge（16 个 Trainium2，64 个逻辑 NeuronCore，每核 24 GiB HBM）
模型：tencent/Hy3-FP8（295B MoE，80 层，192 专家 top-8 + 1 共享专家，GQA 64Q/8KV）
代码：`vllm_neuron/model/hy_v3/`；实验脚本与数据：`examples/vllm_neuron/models/hy_v3/`　｜　软件：vLLM 0.24.0、vllm-neuron 0.24.0.1.1.0、neuronx-cc 2.27

## 1. 结论摘要

- **精度**：MATH-500 no_think 96.8%，high reasoning 94.2%（high 模式失分基本都来自 15000 token 输出上限截断）。4 层真实权重模型与 HF fp32 参考逐步对比 logits，cosine ≥ 0.9986。实现没有发现数值问题。
- **延迟**：单请求 decode ≈ 30 ms/token（≈33 tok/s），与 context 长度（1k–16k）基本无关；prefill ≈ 3.7k tok/s（16k 输入 TTFT 4.4 s）。性能数据都已经用独立 seed 重测，排除了 prefix cache 的污染（见第 4 节）。
- **吞吐**：decode 峰值 ≈ 1.7k tok/s（batch 127），但端到端平均输出吞吐只有 ≈ 380 tok/s。原因是 prefill 会阻塞 decode（Neuron 后端不支持混合批处理），而且短 prompt 也要补齐到 2048 的 prefill bucket。
- **容量瓶颈**：TP=64 时 8 个 KV head 每个被复制 8 份，KV 显存的 7/8 是冗余；16k context 下最多只能 15 个并发。
- **单并发优化（第 10 节）**：MoE prefill 改用动态 block kernel，decode 时共享专家切分，再配合低延迟配置，TTFT 降低 51–63%（1k 输入 0.20 s，16k 输入 1.80 s），TPOT 降低 11–16%（25.5–26.9 ms），MATH-500 精度不变。
- **DCP=8（第 8 节）**：KV 容量提升到 8 倍（26 万 → 208 万 token），32k context 下的并发上限从 5 提高到 63；MATH-500 精度不变（96.0%，基线 96.8%，在采样噪声范围内）。
  - 代价：单请求 TPOT 从约 30.6 ms 增加到约 37.2 ms，prefill 慢 25–37%。
  - 收益：decode 为主的负载有实际提升（MATH-500 no_think 总耗时少 26%）。prefill 为主的长 context 负载，在合并部署下总吞吐没有提升，因为 prefill 只能串行执行。要发挥 DCP 的作用，需要配合 P/D 分离。

## 2. 实现方案

| 部分 | 实现 |
|---|---|
| 并行 | Attention / dense MLP 用 TP=64；路由专家用 EP=64（每 rank 3 个专家，I=1536）；共享专家每 rank 存全量权重 |
| Attention | prefill：`NF.qkv_proj`(STATIC FP8) → torch QK-norm + RoPE → flash / segmented attention → `NF.o_proj`(STATIC)；decode：`NF.attention_decode` megakernel（融合 RMSNorm、pre-RoPE QK-norm、kernel 内写 KV cache） |
| Router | 自写 torch 实现：sigmoid 打分；expert_bias 只用于选择专家；top-k 权重重新归一化后 ×2.826，fp32 计算 |
| 路由专家 | prefill：`moe_cte shard_on_i`（per-channel FP8 反量化）；decode：`moe_tkg` STATIC；两者都是 BF16 激活 × FP8 权重 |
| Dense MLP / 共享专家 | `NF.rmsnorm_quant` + `NF.mlp` STATIC；按 kernel 实际走的路径（CTE/TKG，阈值 96 token）选择输入格式 |
| FP8 范围 | TRN2 的 e4m3 上限是 240：权重 ×0.5（对正规数是精确的指数平移），weight scale ×2 补偿；input scale ×2，避免激活在 240 处截断 |
| DCP | 支持 `--decode-context-parallel-size 8`（第 8 节） |
| 未支持 | MTP 投机解码、FP8 KV cache（checkpoint 没有 KV scale）、DP / attention DP（启动时直接报错） |

框架改动：`moe_tkg.py`（允许 fp8 权重）、`platform.py`（加入 `"fp8"` 量化方法）、`registry.py`（注册模型）；DCP 相关：`neuron_parallel_state.py`（新增 `all_contiguous_replica_groups`）、`attention_decode.py` / `attention_segmented_cte.py`（ReplicaGroup 传完整 DCP 分组）。

## 3. 精度验证

### 3.1 逐层 / 逐 token 对比（4 层真实权重模型 vs HF transformers fp32）

| 测试项 | 结果 |
|---|---|
| 短 prompt × 4，greedy 8 token | logits cosine ≥ 0.9986，top-10 重合 9–10/10；2 处 greedy 分歧都是参考结果里的第 2 名（差距 0.05–0.18 nats） |
| 长 prompt 5000 / 3100 token（segmented prefill） | greedy 8/8，cosine ≥ 0.9994 |
| 再次请求，命中 prefix cache（命中 4992 / 3072 token） | greedy 8/8，cosine ≥ 0.9995 |
| batch 128 vs batch 4（decode >96 token 时共享专家走 CTE 路径） | greedy 完全一致，cosine ≥ 0.9994 |
| 单 kernel 微测试（真实权重，rank-0 分片） | QKV / O / MLP / moe_cte / moe_tkg / attention_decode 与 fake-quant 参考的相对误差 ≤ 2e-2，cos ≥ 0.9995 |

### 3.2 MATH-500（HuggingFaceH4/MATH-500，math-verify 判分，temperature 0.9、top_p 1.0）

| 模式 | 准确率 | L1 | L2 | L3 | L4 | L5 | 平均输出 | 截断数 | 用时 |
|---|---|---|---|---|---|---|---|---|---|
| no_think（max 4096） | **96.8%**（484/500） | 97.7 | 98.9 | 99.0 | 96.9 | 93.3 | 638 | 4 | 19 min（21 并发） |
| high（max 15000） | **94.2%**（471/500） | 97.7 | 100.0 | 95.2 | 96.1 | 86.6 | 4877（中位 1826） | 79 | 2.8 h（15 并发） |

- no_think 的 16 道错题：12 道是模型算错，4 道是输出被截断。
- high 的 29 道错题：27 道是被截断、没有写出最终答案；2 道其实答对了，只是答案用了 `\\[2pt]` 矩阵排版，math-verify 解析不了。截断的内容都是连贯的长推理，没有乱码。
- 两种模式合计，492/500 题至少在一种模式下答对。

## 4. 性能测试

工具：`vllm bench serve`，random 数据集，输出固定 256 token（ignore_eos），每组请求数 = 4×并发（至少 8 个）。吞吐单位都是 tok/s。

> **测试方法修正**：最早一轮测试用的是 bench 默认的固定 seed，不同并发档的 prompt 前缀互相重叠，服务器又开着 prefix caching，所以并发 > 1 的 TTFT / 吞吐被 prefix cache 命中拉低，结果偏乐观（例如 16k 输入时，并发 4 的 TTFT 测出来是 3.47 s，干净重测是 10.85 s）。现在每组用 `seed = 输入长度×1000 + 并发` 生成互不重叠的 prompt，配置 B 和配置 C 都已全部重测；旧数据保留在 `results/bench/archive_fixed_seed/`。配置 C 的旧数据实际上几乎没有受影响（见下文）。

公共配置：TP=64，`--enable-expert-parallel`，block_size 32，prefix caching 打开，`num_batched_tokens_buckets = kv_segment_size_buckets = [2048]`，on-device sampling。

| 配置 | max_model_len | max_num_seqs | decode bucket（batch × context） | KV 预算 | KV 容量 | 实际并发上限 |
|---|---|---|---|---|---|---|
| B | 16384 | 32 | [1,4,16,32] × [4096, 8192, 16384] | 0.45 | 260k token | 15 |
| C | 2048 | 128 | [1,4,16,32,64,128] × [1024, 2048] | 0.45 | 260k token | 127 |

### 配置 B（重测，每组独立 seed，数据干净）

| 输入 token | 并发 | 请求数 | TTFT 中位 | TTFT P99 | TPOT 中位 | TPOT P99 | ITL 中位 | 输出吞吐 | 峰值输出 | 总吞吐 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 8 | 0.54 s | 0.55 s | 28.8 ms | 29.3 ms | 28.3 ms | 32 | 36 | 162 |
| 1024 | 4 | 16 | 1.35 s | 2.13 s | 55.2 ms | 60.9 ms | 51.1 ms | 66 | 88 | 330 |
| 1024 | 15 | 60 | 4.16 s | 7.76 s | 56.4 ms | 70.4 ms | 42.1 ms | 207 | 360 | 1036 |
| 4096 | 1 | 8 | 1.07 s | 1.08 s | 30.3 ms | 30.3 ms | 30.3 ms | 29 | 34 | 495 |
| 4096 | 4 | 16 | 2.62 s | 4.18 s | 56.0 ms | 63.4 ms | 48.7 ms | 61 | 84 | 1036 |
| 4096 | 15 | 60 | 8.26 s | 15.44 s | 75.0 ms | 103.3 ms | 46.9 ms | 140 | 330 | 2383 |
| 8192 | 1 | 8 | 2.14 s | 2.16 s | 30.6 ms | 30.7 ms | 30.7 ms | 26 | 33 | 848 |
| 8192 | 4 | 16 | 5.26 s | 8.38 s | 65.4 ms | 79.2 ms | 52.0 ms | 47 | 80 | 1545 |
| 8192 | 15 | 60 | 16.67 s | 31.19 s | 114.8 ms | 171.9 ms | 57.7 ms | 84 | 270 | 2757 |
| 16000 | 1 | 8 | 4.40 s | 4.44 s | 30.6 ms | 30.7 ms | 30.7 ms | 21 | 33 | 1331 |
| 16000 | 4 | 16 | 10.85 s | 17.29 s | 81.1 ms | 108.8 ms | 55.1 ms | 32 | 80 | 2054 |
| 16000 | 15 | 60 | 34.42 s | 64.42 s | 175.6 ms | 293.6 ms | 57.7 ms | 48 | 270 | 3078 |

### 配置 C（已用独立 seed 重测，`results/bench/baseline_C/`）

| 输入 | 并发 | TTFT 中位 | TTFT P99 | TPOT 中位 | ITL 中位 | 输出吞吐 | 峰值输出 | 总吞吐 | 旧数据 TTFT |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 64 | 16.71 s | 32.82 s | 115.4 ms | 51.9 ms | 355 | 1280 | 1065 | 16.75 s |
| 512 | 127 | 32.80 s | 64.57 s | 200.1 ms | 73.3 ms | 388 | 1778 | 1163 | 33.28 s |
| 1024 | 32 | 8.54 s | 16.49 s | 77.7 ms | 46.4 ms | 289 | 704 | 1445 | 8.56 s |
| 1024 | 63 | 16.45 s | 32.32 s | 123.3 ms | 60.8 ms | 337 | 1071 | 1684 | 16.94 s（c64） |
| 1024 | 127 | 32.83 s | 64.61 s | 208.1 ms | 81.2 ms | 378 | 1651 | 1892 | 33.00 s |

最后一列是旧的固定 seed 数据，和重测结果相差不到 3%。原因是配置 C 只有一个 2048 的 prefill bucket，512 / 1024 的 prompt 都被补齐成一个 2048 段。即使命中 prefix cache，剩下的 token 也还要跑一整步 prefill，省不了时间。配置 B 的长 prompt 要分多段，命中的段可以直接跳过，所以那边受污染很明显。

**同并发下基线 vs DCP=8（1024 输入，都是干净数据）**：

| 1024 输入 | 并发 | TTFT 中位 | TPOT 中位 | ITL 中位 | 输出吞吐 | 峰值输出 | 总吞吐 |
|---|---|---|---|---|---|---|---|
| 基线 | 32 | 8.54 s | 77.7 ms | 46.4 ms | 289 | 704 | 1445 |
| DCP=8 | 32 | 11.39 s | 94.9 ms | 53.1 ms | 230 | 608 | 1151 |
| 基线 | 63 | 16.45 s | 123.3 ms | 60.8 ms | 337 | 1071 | 1684 |
| DCP=8 | 63 | 22.01 s | 155.2 ms | 71.5 ms | 262 | 882 | 1309 |


在 KV 容量足够的短 context 场景下，DCP 只带来额外的通信开销：TPOT / ITL 高 15–25%，吞吐低约 20%。

指标说明：TPOT = (端到端时延 − TTFT) / (输出 token 数 − 1)，包含 decode 被其他请求的 prefill 打断的等待时间；ITL 中位数基本等于一次纯 decode step 的耗时。

### 4.1 分析

1. **单请求 decode 受权重读取限制**：TPOT ≈ 30 ms，1k–16k context 下基本一致，attention 不是瓶颈。
2. **decode 随 batch 扩展得不错**：一步的耗时从 28 ms（bs1）增加到 46 ms（bs32）、61 ms（bs64）、73–81 ms（bs127），对应峰值输出吞吐从 36 增加到 1778 tok/s。
3. **prefill 阻塞 decode 是端到端吞吐的主要瓶颈**：Neuron 后端不支持 chunked prefill / 混合批处理，新请求的 prefill 会暂停整个 batch 的 decode。所以高并发下 TPOT 是 ITL 的 2–3 倍，平均吞吐只有峰值的约 1/4，TTFT 随并发线性增长（并发 127 时 33 s）。
4. **短 prompt 有补齐浪费**：只编译了 2048 一个 prefill bucket（prefix caching 要求它和 KV segment 大小一致），1k 输入的 TTFT 有约一半是在算补齐的部分。
5. **长 context 的并发受 KV 容量限制**：16k context 下最多 15 个并发，原因见第 5 节。

## 5. 显存分析（每个逻辑核 24 GiB）

| 内容 | 单核大小 | 说明 |
|---|---|---|
| 权重 | ≈ 6 GiB | 1/64 的专家 ≈ 4.7 GiB + 共享专家全量 ≈ 1.5 GiB |
| KV cache | ≈ 9.7 GiB（预算 0.45） | 每 token 每 rank 40 KB；**8 个 KV head × 8 份复制，整机每 token 2.56 MB，实际只需要 0.33 MB** |
| NEFF 的 DMA ring 等 | 4–6 GiB | 随编译图数量增加；batch 32 / 16k 那个 decode 图单个就约 1 GiB |

**踩过的坑**：KV 预算设成 0.6 时，运行约 40 分钟后 OOM。有些 NEFF 是第一次执行时才加载的，加载时 DMA ring 没有空间可分配。改成 0.45 之后，后续所有测试都没再出问题。

## 6. 发现的框架问题（已经在模型侧绕过）

1. CTE MLP kernel 在 rank 内 intermediate 不是 128 的倍数时，会**静默算错**（I=208 时 cos≈0.7）→ dense 层按 rank 零填充到 256。
2. nkilib `bwmm_shard_on_block` 实际上不读取 FP8 scale → 改用 `shard_on_i`。
3. MLP TKG 路径收到 fp8 输入时编译失败 → 按 token 数选择融合 norm 路径还是 rmsnorm_quant 路径。
4. Neuron 编译缓存在模型代码改动后**不会失效**，需要手动清理 `compile_cache`。
5. 调度器按 `max_model_len` 的最坏情况硬性限制并发；warmup 不会把所有 NEFF 都加载到设备上，所以 KV 预算要预留余量。

## 7. 后续优化（按收益排序）

1. ~~消除 KV 冗余~~：已通过 DCP=8 完成（见第 8 节）。没有用框架的 attention DP，原因是它需要配合 vLLM DP 和跨 DP 的 EP，而框架只在 decode 阶段对齐各 DP 组，存在 prefill/decode 阶段不一致时的死锁风险。
2. **prefill/decode 分离（1P1D）**：消除 prefill 对 decode 的阻塞，让 TPOT 接近 ITL。在 DCP 之后，这是**收益最大**的一项。
3. **多个 prefill bucket / 更大的 block_size**：减少短 prompt 的补齐浪费；block 越大，DMA 描述符越少，KV 能用的显存越多。
4. 共享专家改为 TP 切分（需要先解决 I=24 时的编译问题），每核省约 1.5 GiB。

## 8. DCP=8（decode context parallel）

### 8.1 实现

- **原理**：TP=64 时，持有同一个 KV head 的 8 个相邻 rank 组成一个 DCP 组。组内每个 rank 只保存序列里属于自己的那些 block（以 block_size=32 交错分配）。
  - prefill：用 `segmented_attention_cp` 计算，Q 在组内 all-gather，各 rank 的局部结果用 LSE 校正后合并。
  - decode：走 `attention_decode` 的 DCP 路径，用 `local_filled` 掩掉不属于本 rank 的位置；当前 token 只由拥有它的 rank 计入。
- **代码改动**：
  - `hy_v3/model_fp8.py`：attention 支持 DCP。DCP rank 由运行时的 `rank` 张量计算，不再作为 Python 常量写死。
  - `hy_v3/factory.py`：加 DCP 约束检查。
  - 框架：两个 attention kernel 的 ReplicaGroup 改为传入完整的 DCP 分组（新增 `all_contiguous_replica_groups`）。
- **踩坑**：第一版把 dcp rank 当作常量写进了计算图，每个 rank 都编出不同的图，4 层模型就要编译 **1216 个图**，完整模型要好几个小时。改成张量 + 完整分组后，图的数量降到 **19 个**（和非 DCP 一样）。这个修复对框架里 Llama3 的 DCP 路径同样有效。
- **启动参数**：`--decode-context-parallel-size 8 --cp-kv-cache-interleave-size 32`（要求等于 block_size）。

### 8.2 精度

| 测试 | 结果 |
|---|---|
| 4 层模型 vs HF fp32（长 prompt segmented prefill、prefix cache 命中、短 prompt） | logits cos ≥ 0.9986；唯一一处 greedy 分歧发生在参考 top-2 只差 0.064 nats 的近似平局位置 |
| batch 128 vs batch 4 | greedy 完全一致，cos ≥ 0.9992 |
| **MATH-500 no_think（完整模型）** | **96.0%**（480/500），基线 96.8%。差距在 temperature 0.9 采样的波动范围内（标准差约 0.8%） |

### 8.3 容量

| | 基线 | DCP=8 |
|---|---|---|
| KV cache 容量（KV 预算 0.45） | 260,448 token | **2,083,584 token（×8.0）** |
| 32k context 并发上限 | 5 | **63** |
| 16k context 并发上限 | 15 | 127 |

### 8.4 性能（配置：max_model_len 32768，max_num_seqs 64，每组独立 seed，没有 prefix cache 污染）

| 输入 | 并发 | 请求数 | TTFT 中位 | TTFT P99 | TPOT 中位 | ITL 中位 | 输出吞吐 | 峰值输出 | 总吞吐 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 8 | 0.71 s | 0.72 s | 37.0 ms | 36.6 ms | 25 | 28 | 126 |
| 1024 | 4 | 16 | 1.77 s | 2.84 s | 56.6 ms | 51.6 ms | 63 | 84 | 315 |
| 1024 | 15 | 60 | 5.55 s | 10.38 s | 62.1 ms | 43.1 ms | 180 | 360 | 898 |
| 1024 | 32 | 128 | 11.39 s | 21.99 s | 94.9 ms | 53.1 ms | 230 | 608 | 1151 |
| 1024 | 63 | 252 | 22.01 s | 43.26 s | 155.2 ms | 71.5 ms | 262 | 882 | 1309 |
| 4096 | 1 | 8 | 1.46 s | 1.48 s | 37.2 ms | 36.5 ms | 23 | 28 | 398 |
| 4096 | 4 | 16 | 3.57 s | 5.69 s | 62.8 ms | 53.4 ms | 52 | 80 | 887 |
| 4096 | 15 | 60 | 11.35 s | 21.22 s | 83.5 ms | 44.7 ms | 118 | 345 | 2000 |
| 4096 | 32 | 128 | 23.34 s | 45.21 s | 140.5 ms | 54.6 ms | 138 | 608 | 2354 |
| 4096 | 63 | 252 | 45.19 s | 88.17 s | 243.8 ms | 72.2 ms | 151 | 882 | 2561 |
| 8192 | 1 | 8 | 2.94 s | 2.95 s | 37.3 ms | 37.0 ms | 21 | 28 | 678 |
| 8192 | 4 | 16 | 7.23 s | 11.55 s | 73.1 ms | 54.7 ms | 40 | 76 | 1305 |
| 8192 | 15 | 60 | 22.97 s | 43.01 s | 124.9 ms | 46.1 ms | 70 | 330 | 2312 |
| 16000 | 1 | 8 | 5.89 s | 5.92 s | 37.4 ms | 37.0 ms | 17 | 28 | 1053 |
| 16000 | 4 | 16 | 14.53 s | 23.22 s | 90.0 ms | 54.1 ms | 27 | 76 | 1734 |
| 16000 | 15 | 60 | 46.23 s | 86.58 s | 204.5 ms | 46.2 ms | 39 | 330 | 2479 |
| 16000 | 32 | 128 | 95.29 s | 184.62 s | 407.3 ms | 56.7 ms | 41 | 576 | 2612 |
| 16000 | 63 | 252 | 184.63 s | 363.33 s | 778.8 ms | 77.5 ms | 42 | 819 | 2672 |

**基线 vs DCP=8（同一批 prompt，基线 → DCP；吞吐单位 tok/s；基线在 16k 下最多 15 并发，所以 c32 / c63 只有 DCP 的数据）**：

| 输入 | 并发 | TTFT 中位 | TPOT 中位 | ITL 中位 | 输出吞吐 | 峰值输出 | 总吞吐 |
|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.54 s → **0.71 s** | 28.8 ms → **37.0 ms** | 28.3 ms → **36.6 ms** | 32 → **25** | 36 → **28** | 162 → **126** |
| 1024 | 4 | 1.35 s → **1.77 s** | 55.2 ms → **56.6 ms** | 51.1 ms → **51.6 ms** | 66 → **63** | 88 → **84** | 330 → **315** |
| 1024 | 15 | 4.16 s → **5.55 s** | 56.4 ms → **62.1 ms** | 42.1 ms → **43.1 ms** | 207 → **180** | 360 → **360** | 1036 → **898** |
| 1024 | 32 | — → **11.39 s** | — → **94.9 ms** | — → **53.1 ms** | — → **230** | — → **608** | — → **1151** |
| 1024 | 63 | — → **22.01 s** | — → **155.2 ms** | — → **71.5 ms** | — → **262** | — → **882** | — → **1309** |
| 4096 | 1 | 1.07 s → **1.46 s** | 30.3 ms → **37.2 ms** | 30.3 ms → **36.5 ms** | 29 → **23** | 34 → **28** | 495 → **398** |
| 4096 | 4 | 2.62 s → **3.57 s** | 56.0 ms → **62.8 ms** | 48.7 ms → **53.4 ms** | 61 → **52** | 84 → **80** | 1036 → **887** |
| 4096 | 15 | 8.26 s → **11.35 s** | 75.0 ms → **83.5 ms** | 46.9 ms → **44.7 ms** | 140 → **118** | 330 → **345** | 2383 → **2000** |
| 4096 | 32 | — → **23.34 s** | — → **140.5 ms** | — → **54.6 ms** | — → **138** | — → **608** | — → **2354** |
| 4096 | 63 | — → **45.19 s** | — → **243.8 ms** | — → **72.2 ms** | — → **151** | — → **882** | — → **2561** |
| 8192 | 1 | 2.14 s → **2.94 s** | 30.6 ms → **37.3 ms** | 30.7 ms → **37.0 ms** | 26 → **21** | 33 → **28** | 848 → **678** |
| 8192 | 4 | 5.26 s → **7.23 s** | 65.4 ms → **73.1 ms** | 52.0 ms → **54.7 ms** | 47 → **40** | 80 → **76** | 1545 → **1305** |
| 8192 | 15 | 16.67 s → **22.97 s** | 114.8 ms → **124.9 ms** | 57.7 ms → **46.1 ms** | 84 → **70** | 270 → **330** | 2757 → **2312** |
| 16000 | 1 | 4.40 s → **5.89 s** | 30.6 ms → **37.4 ms** | 30.7 ms → **37.0 ms** | 21 → **17** | 33 → **28** | 1331 → **1053** |
| 16000 | 4 | 10.85 s → **14.53 s** | 81.1 ms → **90.0 ms** | 55.1 ms → **54.1 ms** | 32 → **27** | 80 → **76** | 2054 → **1734** |
| 16000 | 15 | 34.42 s → **46.23 s** | 175.6 ms → **204.5 ms** | 57.7 ms → **46.2 ms** | 48 → **39** | 270 → **330** | 3078 → **2479** |
| 16000 | 32 | — → **95.29 s** | — → **407.3 ms** | — → **56.7 ms** | — → **41** | — → **576** | — → **2612** |
| 16000 | 63 | — → **184.63 s** | — → **778.8 ms** | — → **77.5 ms** | — → **42** | — → **819** | — → **2672** |

### 8.5 分析

1. **同等并发下，DCP 每一项都更慢**：
   - decode 每层多出 Q all-gather、LSE all-gather 和 reduce-scatter，单请求 TPOT 从约 30.6 ms 涨到约 37.2 ms（+22%），和 context 长度无关。
   - prefill 慢了约 25–37%（例如 16k 输入时 TTFT 从 4.40 s 涨到 5.89 s）。
   - 并发 4 和 15 时，ITL 两边差不多；但 TTFT 和 TPOT 都受 prefill 变慢影响，DCP 更差。
2. **DCP 换来的是容量**：KV 容量变成 8 倍，16k context 下可以放下 63 个请求（基线最多 15 个）。decode 的峰值吞吐从 270 提高到 819 tok/s，ITL 只从 57.7 ms 涨到 77.5 ms。
3. **在当前的合并部署方式下，端到端总吞吐没有提升**：Neuron 后端不能把 prefill 和 decode 混在同一步执行，prefill 只能串行跑，DCP 下约 2.8k tok/s，基线约 3.7k tok/s。输入长的时候，总吞吐基本由 prefill 速度决定：16k 输入时，基线 c15 是 3078 tok/s，DCP c63 只有 2672 tok/s。也就是说，并发加上去了，但 prefill 变慢把收益抵消了。
4. **decode 占比高的负载能看到实际收益**：MATH-500 no_think（输入短、输出较长）用 DCP 跑 63 并发，耗时 848 s；基线 21 并发要 1148 s，**总耗时少了 26%**。high reasoning 这种长输出负载收益应该更大，但没有实测。
5. **结论**：DCP 适合 **decode 为主、需要很多长 context 请求同时在跑** 的场景，比如长推理链、长文档多轮对话、P/D 分离部署里的 decode 实例。prefill 为主、低并发、对延迟敏感的场景，不开 DCP 更好。
6. **下一步**：做 P/D 分离，让 DCP 只承担 decode。这样 prefill 变慢和 prefill 阻塞 decode 这两个问题都会消失。另外，可以单独分析 DCP prefill 为什么比基线慢约 30%（`segmented_attention_cp` 的 Q gather、LSE 合并）。

### 8.6 待办

- ~~用相同的带 seed 测试脚本重测基线~~：已完成（配置 B 12 组在 `results/bench/baseline_B/`，配置 C 5 组在 `results/bench/baseline_C/`）。

## 9. P/D 分离可行性调查（单机，未完成）

目标：在一台 trn2.48xlarge 上把 64 个核分成两半，0–31 跑 prefill（TP=32 / EP=32），32–63 跑 decode（TP=32 / EP=32 / DCP=4），通过 NIXL 在两组核之间传 KV cache。这是框架 DI 教程里的单机拓扑。

**显存估算（每核 24 GiB，TP=32）**：路由专家约 9.4 GB + 共享专家 / attention / embedding 约 1.8 GB + NEFF 4–6 GB，剩下约 5 GB 给 KV。decode 端用 DCP=4 时，KV 容量约 50 万 token，可以做。

**结论：这台机器做不了，原因是没有 EFA 网卡，不是代码问题。**

1. pip 装的 `nixl` 是 CUDA 版本（`nixl_cu13`），LIBFABRIC 插件依赖 `libcuda.so.1`，在没有 NVIDIA 驱动的机器上加载失败。
2. 用一个最小的 `libcuda` stub（`di/libcuda_stub.c`，所有接口都返回 `CUDA_ERROR_NO_DEVICE`）补上依赖后，插件能加载。插件本身支持 Neuron 显存（`FI_HMEM_NEURON`）。
3. 插件初始化时需要可以做 RDMA / HMEM 的网络 provider，报错 `Topology discovery failed - no suitable network providers found`。这台实例只有一块普通 ENI，没有 EFA 设备（`/sys/class/infiniband` 为空，`fi_info -p efa` 也没有结果）。`shm` / `tcp` provider 不能注册 Neuron 显存。
4. Neuron 平台的 NIXL 只支持设备到设备直传（`kv_buffer_device="cuda"`，在这里指 Neuron HBM），不支持先拷到 host 再传。

**可选方案**：
- 停机后给实例加 EFA 网卡（trn2.48xlarge 最多支持 16 块，安全组需要放行组内流量），再按官方流程做单机 DI。
- 实现 host 中转的 KV 传输：Neuron 平台要支持 `kv_buffer_device="cpu"` 和显存与 host 之间的块拷贝。代价是一个 16k 请求每个 rank 要拷约 0.6 GB。
- 用两台带 EFA 的实例，各自 TP=64（decode 端加 DCP=8）。

复现：`di/nixl_probe.py`（两核之间用 NIXL 传一块 HBM）。

## 10. 单并发延迟优化

目标是降低单并发下的 TTFT 和 TPOT。先在单核上做了微基准，定位瓶颈（脚本：`kernel_tests/bench_moe_cte.py`、`kernel_tests/bench_decode.py`）。

### 10.1 瓶颈定位

- **TTFT 主要花在 MoE prefill 上**。`moe_cte shard_on_i` 按最坏情况分配 block：每个 rank 有 3 个本地专家，T=2048、top-8 时静态分配了 26 个 block，但实际只用到约 3 个，所以大部分计算都花在补齐上。2048 token 的一步 prefill，光 MoE 就要约 0.42 s（79 层 × 5.35 ms），而 1k 输入的 TTFT 一共才 0.54 s。
- **TPOT（batch 1）主要是每层 kernel 本身的执行时间**，每层约 0.36 ms，× 80 层 ≈ 29 ms。集合通信只占很小一部分。各组成部分的耗时：

  | 组成部分 | 耗时 | 占比 |
  |---|---|---|
  | 路由专家（moe_tkg，3 个本地专家全部加载） | ~0.19 ms | ~50% |
  | attention megakernel | ~0.10 ms | ~28% |
  | 共享专家（全量 I=1536） | ~0.08 ms | ~22% |

- batch 1 时，每个 rank 上平均不到 1 个专家被选中。但 moe_tkg 的 selective 模式和 dynamic 模式在 EP 下都不能跳过没被选中的专家：selective 要固定循环 K=8 次，dynamic 要求每个 rank 只有 1 个本地专家。要做到这一点需要写新 kernel，这一轮没做。

### 10.2 改动（`hy_v3/model_fp8.py`）

1. **MoE prefill 改用 `shard_on_i_hybrid`**（`num_static_block` = 本地专家数），运行时跳过空 block。单核测试：T=2048 时从 5.35 ms 降到 1.92 ms，T=8192 时从 18.68 ms 降到 5.14 ms，数值误差和原来一样。
   - **限制**：它的动态循环和 segmented attention 的 `nl.dynamic_range` 放在同一个计算图里时，neuronx-cc 报 `NCC_ICFG001`（控制流图出现多个出口节点），编译失败。所以只有 prefill 走 flash attention 时才用 hybrid；segmented prefill（开 prefix caching 时）仍然用 `shard_on_i`。两者在语义上互不影响，这是编译器层面的问题，还没有做最小复现去确认。
2. **decode 时共享专家另存一份 8 路切分的权重**（I=192，补齐到 256）。部分结果乘 1/8 后，合并进原有的 MoE all-reduce，不增加通信；prefill 仍然用全量权重。这样每个 rank 每层读取的共享专家权重从约 19 MB 降到约 2.4 MB。
3. **低延迟服务配置**（`serve_latency.sh`）：关闭 prefix caching；prefill bucket 设为 512 到 16384；`max_num_batched_tokens` 为 16384。这样 1k 输入不用补齐到 2048，16k 输入一步就能完成 prefill。

### 10.3 结果（完整模型，单并发，每组 8 个请求，独立 seed）

| 输入 | TTFT：基线 → 优化 | TPOT：基线 → 优化 | ITL：基线 → 优化 |
|---|---|---|---|
| 1024 | 0.54 s → **0.20 s（−63%）** | 28.8 → **25.5 ms（−11%）** | 28.3 → 25.3 ms |
| 4096 | 1.07 s → **0.53 s（−51%）** | 30.3 → **25.5 ms（−16%）** | 30.3 → 25.5 ms |
| 8192 | 2.14 s → **0.97 s（−55%）** | 30.6 → **26.9 ms（−12%）** | 30.7 → 26.9 ms |
| 16000 | 4.40 s → **1.80 s（−59%）** | 30.6 → **26.9 ms（−12%）** | 30.7 → 26.9 ms |

- TTFT 的下降是 hybrid MoE 和更细的 bucket 共同带来的，两者各自的贡献没有拆开测。
- 8k 和 16k 输入时，基线和优化版本用的是同一个 16384 的 decode context bucket，所以这两行 TPOT 降低的约 3.7 ms 可以归功于共享专家切分。1k 和 4k 输入时，低延迟配置的 decode bucket 更小，attention 少读了一些 KV，所以降幅更大。
- 原始数据：`results/bench/opt_latency/`。

### 10.4 精度

- **4 层模型 vs HF fp32**：flash 路径（长 prompt 一步 prefill）greedy 8/8，cos ≥ 0.9995；segmented 路径（含 prefix cache 命中）cos ≥ 0.9988；batch 128 vs batch 4 greedy 一致。新增的几处短 prompt greedy 分歧，参考 top-2 的差距都在 0.007–0.18 nats 之间（4 层模型的输出分布很平）。
- **MATH-500 no_think，完整模型，前 200 题**：**96.5%**（193/200）。同样这 200 题，基线答对 191 题（95.5%）。数据在 `results/math500/math500_no_think_opt_latency_200.jsonl.gz`。

### 10.5 尚未覆盖

- 优化版本只测了单并发。高并发下、以及 decode token 数 > 96 时共享专家走 CTE 路径的情况，只在 4 层模型上验证了数值，完整模型没有测性能。
- 开 prefix caching 时只能拿到 TPOT 的优化。要解决这个问题，可以把 attention 换成静态的"flash attention + 历史 KV（`k_prior` / `v_prior`）"，或者等编译器修复。
- 下一步可以做：QK-norm 和 RoPE 融合进 QKV kernel（降 TTFT）；MTP 投机解码（降 TPOT，收益最大）；为 batch 1 写一个只加载被选中专家的 MoE kernel（降 TPOT）。

## 附：复现

所有路径相对于 `examples/vllm_neuron/models/hy_v3/`。

- 环境：vLLM 0.24 + 本仓库（`pip install -e .`），另外装 neuronx-cc / nki；这台机器没有 EFA，需要设置 `NEURON_SKIP_EFA_AFFINITY=1`（`serve.sh` 已经设好）。
- 低延迟服务（单并发优化后的配置）：`MODEL=<ckpt> ./serve_latency.sh`。
- 服务：`MODEL=<ckpt> [DCP=8] [KV_CAP=0.45] ./serve.sh <max_model_len> <max_num_seqs> '<num_seqs_buckets>' '<decode_ctx_buckets>'`（文件开头注释里有配置 B / C / DCP 对应的参数）。
- 性能：`OUT_DIR=results/bench/<name> MODEL=<ckpt> ./bench_matrix.sh "<输入长度列表>" "<并发列表>" 256`（每组用独立 seed）。
- MATH-500：`python eval_math500.py {no_think|high} out.jsonl --concurrency N --max-tokens M`（需要 `openai`、`math-verify[antlr4_13_2]`、`datasets`）。
- 精度：
  - `make_mini.py <ckpt> <out> 4` 生成 4 层的 fp8 / bf16 小模型；
  - `hf_reference.py <out>/bf16 ref.pt` 生成 HF fp32 参考（可以用 `HY3_PROMPTS` 指定长 prompt）；
  - `compare_hf_logits.py <out>/fp8 ref.pt` 做逐 token 的 logits 对比；
  - `validate_features.py <out>/fp8 ref_long.pt,ref.pt [8]` 验证 segmented prefill、prefix cache 和 batch 128 的一致性（最后一个参数是 DCP 大小）。
- kernel 微测试：`cd kernel_tests && HY3_CKPT=<ckpt> python test_*.py ...`（单个 NeuronCore，使用真实权重分片）。
- checkpoint 的 scale 统计：`inspect_scales.py <ckpt>`。
- 结果数据：
  - `results/bench/{baseline_B,baseline_C,dcp8}/`（干净数据），汇总在 `results/bench/summary.csv`；`results/bench/opt_latency/`（第 10 节的单并发优化结果）；
  - `results/bench/archive_fixed_seed/`（固定 seed 的旧数据，仅作存档）；
  - `results/math500/*.jsonl.gz`（每道题的回答、判分结果和 token 数）。
