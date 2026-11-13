# KV scale profiles

FP8 KV cache scales for Hy3 (`--kv-cache-dtype fp8`, format and loader in
`src/kv_scales.py`). Select one with `HY3_KV_SCALE_PROFILE=<file>`.

| File | Checkpoint | Calibration |
|---|---|---|
| `Hy3-FP8_mtp_neuron.safetensors` (sha256 `3f3dd44755ce5855...`) | tencent/Hy3-FP8, 81 layers: the 80 decoder layers + the MTP layer `model.layers.80` | layers 0-79 copied unchanged from `Hy3-FP8_neuron.safetensors` (so the target's decode NEFFs are the same); layer 80 from `calibrate_kv_scales_neuron.py calibrate --mtp`: the same run and samples with MTP k=1 on (`max_tokens=2`, so the draft passes write its cache). The same run reproduced layers 0-79 exactly (max ratio 1.000). Layer 80: K absmax 8.06, V absmax 4.94 |
| `Hy3-FP8_neuron.safetensors` (sha256 `3c708f14607b1b0d...`) | tencent/Hy3-FP8 (attention fingerprints of all 80 layers) | `scripts/calibrate_kv_scales_neuron.py` on a trn2.48xlarge: the served W8A8 model, BF16 KV cache, config B, prefill-only run of the 75 samples of `kv_calib/full.json` (`bench_data/kv_calib_full_samples.json.gz`, 72,458 tokens); per-layer absmax of the written cache slots, max over the 64 ranks; DEQUANT scale `absmax / 240` |

The metadata holds the per-layer `k_absmax` / `v_absmax`. K absmax runs from
6.5 to 23.6 over the layers, V absmax from 0.13 (layer 0) to 12.9 (layer 66).
On the 4-layer reduced model the llm-compressor calibration
(`scripts/calibrate_kv_scales.py`, the same samples, BF16 weights on CPU)
gives layers 0-3 within 4.5% of these values (three of the eight identical).
With this profile, config B on the full model scores 96.8% on MATH-500
(greedy, no_think; 97.0% with BF16 KV) and gives the same greedy output as
BF16 KV on 16 8192-token prompts (`results/bench/kv_fp8_configB/`).

Use `Hy3-FP8_mtp_neuron.safetensors` with MTP (`--speculative-config
'{"method": "mtp", ...}'`): the MTP draft reads its layer's scales from the
same profile and refuses an 80-layer one. Without MTP both files give the
same scales.
