# SPDX-License-Identifier: Apache-2.0
"""Calibrate the Hy3 FP8 KV scale profile on Neuron, from the BF16 KV cache.

Same statistic and export as calibrate_kv_scales.py (per-layer per-tensor
absmax of K after q/k-norm + RoPE and of V after v_proj; DEQUANT scale
``absmax / 240``; kv_scales.save_profile with the attention fingerprints), but
measured on the served W8A8 model instead of an HF forward of a BF16 copy:

  * the model runs with a BF16 KV cache over the calibration samples, prefill
    only (max_tokens=1). The cache then holds exactly what an FP8 cache
    quantizes;
  * per rank and layer every cache slot gets a signature (sum, abs-sum) before
    and after the run, and the absmax is taken over the changed slots only
    (warm-up leftovers and unused slots of partial blocks never count). The
    samples (72k tokens) fit the 8139-block pool, so none overwrites another;
  * the layer absmax is the max over the 64 ranks.

On a trn2.48xlarge the prefill takes ~16 s (plus the model compile); the CPU
llm-compressor run took 27 min for the 4-layer reduced model, ~9 h projected
for the full one. Layers 0-3 agree with llm-compressor on the reduced model to
<= 4.5% (BF16 vs W8A8 weights).

Usage:
    # 1. token ids of the calibration mix (needs transformers + datasets; the
    #    kv_calib/full.json result is bench_data/kv_calib_full_samples.json.gz)
    python calibrate_kv_scales_neuron.py samples <tokenizer> kv_calib/full.json out.json.gz
    # 2. calibrate (Neuron venv)
    python calibrate_kv_scales_neuron.py calibrate <Hy3-FP8> samples.json.gz kv_scales.safetensors
    # 2b. add the MTP layer (model.layers.80) to an 80-layer profile: the same
    #     run with MTP on (max_tokens=2, so the draft passes run); layers 0-79
    #     are copied from <base profile> unchanged (decode NEFFs embed them)
    python calibrate_kv_scales_neuron.py calibrate --mtp <base profile> <Hy3-FP8> \
        samples.json.gz kv_scales_mtp.safetensors
"""

import gzip
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import kv_scales  # src/kv_scales.py: torch + safetensors only

TARGET_FP8_MAX = 240.0


def write_samples(tokenizer, config, out):
    from transformers import AutoTokenizer

    sys.path.insert(0, str(HERE))
    from calibrate_kv_scales import build_samples

    with open(config) as f:
        cfg = json.load(f)
    samples, info = build_samples(cfg, AutoTokenizer.from_pretrained(tokenizer))
    with (
        open(out, "wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
    ):
        gz.write(
            json.dumps(
                {"config": Path(config).name, "samples": samples, "info": info}
            ).encode()
        )
    print(f"{out}: {len(samples)} samples, {sum(map(len, samples))} tokens")


def _layers(worker):
    """The target's decoder layers, then the MTP draft's layer when MTP is on."""
    model = worker.model_runner.model
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    layers = list(getattr(getattr(model, "model", model), "layers", []))
    drafter = getattr(worker.model_runner, "drafter", None)
    if drafter is not None:
        draft = getattr(drafter.model, "_orig_mod", drafter.model)
        layers += list(draft.layers)
    return layers


def _slot_view(cache):
    """[blocks, kvh, block, D] on device -> [slots, kvh * D] on host."""
    x = cache.detach().cpu()
    return x.transpose(1, 2).reshape(-1, x.shape[1] * x.shape[3])


def _signature(x):
    import torch

    xd = x.to(torch.float64)
    return torch.stack([xd.sum(-1), xd.abs().sum(-1)], -1)


def snapshot(worker):
    worker._kv_calib_before = [
        [
            _signature(_slot_view(c))
            for c in (layer.self_attn.k_cache, layer.self_attn.v_cache)
        ]
        for layer in _layers(worker)
    ]
    return len(worker._kv_calib_before)


def changed_absmax(worker):
    """Per layer [k_absmax, v_absmax, k_slots, v_slots] over the slots written since snapshot()."""
    out = []
    for li, layer in enumerate(_layers(worker)):
        row, counts = [], []
        for ci, cache in enumerate((layer.self_attn.k_cache, layer.self_attn.v_cache)):
            x = _slot_view(cache)
            changed = (_signature(x) != worker._kv_calib_before[li][ci]).any(-1)
            counts.append(int(changed.sum()))
            row.append(float(x[changed].float().abs().max()) if changed.any() else 0.0)
        out.append(row + counts)
    return out


def calibrate(ckpt, samples_path, out, base_profile=None):
    """``base_profile``: MTP mode, see the module docstring (2b)."""
    from vllm import LLM, SamplingParams

    with gzip.open(samples_path, "rt") as f:
        data = json.load(f)
    samples = data["samples"]
    # Config B parameters (scripts/serve.sh 16384 32 with prefill 512..8192),
    # BF16 KV, the 8139-block pool of the full model.
    llm = LLM(
        model=ckpt,
        tensor_parallel_size=64,
        enable_expert_parallel=True,
        max_model_len=16384,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        block_size=32,
        enable_prefix_caching=True,
        num_gpu_blocks_override=8139,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": 1} if base_profile else None
        ),
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
                "kv_segment_size_buckets": [2048],
                "num_seqs_buckets": [1, 4, 16, 32],
                "decode_context_length_buckets": [4096, 8192],
            }
        },
    )
    llm.collective_rpc(snapshot)
    t0 = time.time()
    llm.generate(
        [{"prompt_token_ids": s} for s in samples],
        SamplingParams(max_tokens=2 if base_profile else 1, temperature=0),
    )
    n_tok = sum(map(len, samples))
    print(
        f"prefilled {len(samples)} samples, {n_tok} tokens in {time.time() - t0:.0f}s",
        flush=True,
    )
    per_rank = llm.collective_rpc(changed_absmax)
    n_layers = len(per_rank[0])
    k_abs = [max(r[i][0] for r in per_rank) for i in range(n_layers)]
    v_abs = [max(r[i][1] for r in per_rank) for i in range(n_layers)]
    slots = sorted({tuple(per_rank[0][i][2:]) for i in range(n_layers)})
    print(f"written slots per layer (rank 0, K / V): {slots}")
    if base_profile:
        base_meta, k_abs, v_abs = _with_mtp_layer(base_profile, ckpt, k_abs, v_abs, per_rank)
        n_layers = len(k_abs)
    meta = {
        **({"mtp_layer": f"layer {n_layers - 1} calibrated with MTP k=1, max_tokens=2; "
            f"layers 0-{n_layers - 2} copied from {Path(base_profile).name} "
            f"(sha256 {base_meta['sha256'][:16]})"} if base_profile else {}),
        "target_fp8_max": str(TARGET_FP8_MAX),
        "attn_fingerprints": kv_scales.attn_fingerprints(ckpt, n_layers),
        "method": "neuron: absmax over the BF16 KV cache slots written by a prefill-only run of the "
        "served W8A8 model (TP64, config B), max over ranks; d = absmax / target_fp8_max",
        "calibration": {
            "config": data.get("config"),
            "samples": len(samples),
            "tokens": n_tok,
        },
        "k_absmax": k_abs,
        "v_absmax": v_abs,
    }
    sha = kv_scales.save_profile(
        out,
        [a / TARGET_FP8_MAX for a in k_abs],
        [a / TARGET_FP8_MAX for a in v_abs],
        meta,
    )
    print(f"wrote {out} sha256 {sha}")
    for i in range(n_layers):
        print(f"layer {i:2d} k_absmax {k_abs[i]:9.4f} v_absmax {v_abs[i]:9.4f}")


def _with_mtp_layer(base_profile, ckpt, k_abs, v_abs, per_rank):
    """Base profile's layers 0..L-1 (as absmax) + the measured MTP layer L."""
    n = len(k_abs) - 1
    if per_rank[0][n][2] == 0:
        raise RuntimeError(f"the MTP layer (row {n}) wrote no KV slots; is MTP on?")
    q_k, q_v, base_meta = kv_scales.load_profile(base_profile, ckpt, n, TARGET_FP8_MAX)
    base_k = [TARGET_FP8_MAX / q for q in q_k]
    base_v = [TARGET_FP8_MAX / q for q in q_v]
    rk = max(a / b for a, b in zip(k_abs[:n], base_k))
    rv = max(a / b for a, b in zip(v_abs[:n], base_v))
    print(f"target layers in this run vs {base_profile}: max ratio K {rk:.3f} V {rv:.3f} "
          "(1.0 = same calibration; the base values are kept)")
    return base_meta, base_k + [k_abs[n]], base_v + [v_abs[n]]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "samples" and len(sys.argv) == 5:
        write_samples(*sys.argv[2:5])
    elif cmd == "calibrate" and len(sys.argv) in (5, 7):
        # Callables sent to the workers through collective_rpc are pickled.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
        os.environ.setdefault("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.45")
        if len(sys.argv) == 7:
            if sys.argv[2] != "--mtp":
                sys.exit(__doc__)
            calibrate(*sys.argv[4:7], base_profile=sys.argv[3])
        else:
            calibrate(*sys.argv[2:5])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
