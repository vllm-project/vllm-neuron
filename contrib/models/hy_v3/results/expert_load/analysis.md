# Routed-expert load, scripts/expert_load.py on the expert_load.sh / expert_load_client.py captures

152 prefill passes (685,318 tokens), 152 requests with 19,164 decode tokens, layers 1-79
(ShareGPT prompts of 1k / 4k / 8k / 16k tokens + 40 MATH-500 problems, 128 generated tokens each).

expert popularity per layer: max / mean 8.02 (median over layers, max 12.96); top 10% of experts
take 40.9% of the selections (10% if uniform)

held-out half (placements fitted on the other half of the requests), mean over layers:

| placement | prefill busiest/mean tokens | decode T=1 busiest hits (mean) | decode T=4 busiest hits (mean) | decode T=8 busiest hits (mean) | decode T=32 busiest hits (mean) |
|---|---|---|---|---|---|
| current | 4.756 | 1.271 (0.125) | 2.073 (0.379) | 2.698 (0.790) | 3.000 (1.723) |
| random | 4.748 | 1.267 (0.125) | 2.072 (0.379) | 2.700 (0.790) | 3.000 (1.723) |
| balanced | 4.350 | 1.184 (0.125) | 1.912 (0.379) | 2.556 (0.790) | 2.999 (1.723) |
| anti-coact | 4.326 | 1.175 (0.125) | 1.904 (0.379) | 2.548 (0.790) | 2.999 (1.723) |

prefill busiest/mean tokens by pass size, current -> balanced fitted on all passes (in-sample, an upper bound):
<=512 tokens 5.91 -> 5.99, 1k 4.11 -> 3.76, 4k 4.31 -> 3.61, 8k 4.42 -> 3.69, 16k 4.51 -> 3.80
