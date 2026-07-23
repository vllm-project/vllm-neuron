# Models

Production-ready deployment recipes for specific models on AWS Trainium and Inferentia. Each recipe includes instance sizing, configuration, and performance guidance.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Deploy GPT-OSS
:link: gpt-oss
:link-type: doc

Model recipe for GPT-OSS 20B and 120B (MoE) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-VL 32B
:link: qwen3-vl
:link-type: doc

Model recipe for Qwen3-VL 32B (multimodal) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3 / Qwen3MoE
:link: qwen3
:link-type: doc

Model recipe for Qwen3 dense and Qwen3MoE (text) on Trn2.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

GPT-OSS <gpt-oss>
Qwen3-VL <qwen3-vl>
Qwen3 / Qwen3MoE <qwen3>
:::
