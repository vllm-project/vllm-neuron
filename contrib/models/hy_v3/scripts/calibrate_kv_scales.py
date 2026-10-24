"""Calibrate Hy3 FP8 KV-cache scales with llm-compressor and export a profile.

Runs llm-compressor's ``QuantizationModifier(kv_cache_scheme=...)`` (KV only:
no weight or activation scheme, so the W8A8 checkpoint is untouched) over a
fixed calibration mix, on a BF16 copy of the checkpoint (HF transformers, CPU).
The KV observer is compressed-tensors' attention cache hook, so K is observed
after the per-head q/k-norm and RoPE and V after v_proj -- the values the Neuron
cache stores -- before the GQA repeat.

Export: per layer, the dequant scale for the TARGET FP8 range,
``d = absmax / target_fp8_max`` (Trn2: 240). llm-compressor's own scale is
``absmax / 448`` (torch float8_e4m3fn), held in the model dtype; the exact FP32
absmax is taken from the same observer calls and cross-checked against it.

Usage (calibration venv: llmcompressor 0.14.0.1, transformers 5.17.0):
    python -I calibrate_kv_scales.py --model <bf16 ckpt> --fp8-ckpt <Hy3-FP8> \
        --config kv_calib/full.json --out kv_scales.safetensors [--report r.json]
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import kv_scales  # noqa: E402  (src/kv_scales.py: torch + safetensors only)

TORCH_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, llm-compressor's range


# ── calibration data ──────────────────────────────────────────────────────


def _messages(fmt: str, ex: dict) -> list[dict]:
    if fmt == "messages":
        return [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
    if fmt == "alpaca":
        user = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]
    if fmt == "qa":
        return [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
    raise ValueError(fmt)


def build_samples(cfg: dict, tok) -> tuple[list[list[int]], list[dict]]:
    from datasets import load_dataset

    rng = random.Random(cfg["seed"])
    max_len = cfg["max_seq_len"]
    samples, info = [], []
    spare = {}  # source -> unused rows for packing long samples
    for src in cfg["sources"]:
        ds = load_dataset(src["hf"], src.get("config"), split=src["split"])
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        take, rest = idx[: src["n"]], idx[src["n"] :]
        spare[src["name"]] = (ds, src["format"], rest)
        for i in take:
            text = tok.apply_chat_template(_messages(src["format"], ds[i]), tokenize=False)
            ids = tok(text, add_special_tokens=False).input_ids[:max_len]
            samples.append(ids)
            info.append({"source": src["name"], "row": i, "tokens": len(ids)})
    # Long samples: real conversations packed back to back up to the target length.
    long_cfg = cfg.get("long")
    if long_cfg:
        for j in range(long_cfg["n"]):
            target = long_cfg["lengths"][j % len(long_cfg["lengths"])]
            name = long_cfg["from"][j % len(long_cfg["from"])]
            ds, fmt, rest = spare[name]
            ids, rows = [], []
            while len(ids) < target and rest:
                i = rest.pop()
                text = tok.apply_chat_template(_messages(fmt, ds[i]), tokenize=False)
                ids += tok(text, add_special_tokens=False).input_ids
                rows.append(i)
            samples.append(ids[:target])
            info.append({"source": f"long:{name}", "rows": rows, "tokens": len(ids[:target])})
    return samples, info


# ── exact absmax capture alongside the llm-compressor observers ───────────


class KVStats:
    """Exact FP32 absmax per (layer, k|v) plus a reservoir sample of |x|."""

    def __init__(self, names: dict, sample_per_call: int = 4096, seed: int = 0):
        self.names = names  # id(attention module) -> layer index
        self.amax = {}
        self.count = {}
        self.samples = {}
        self.n = sample_per_call
        self.gen = torch.Generator().manual_seed(seed)

    def observe(self, module, kind: str, x: torch.Tensor):
        layer = self.names[id(module)]
        key = (layer, kind)
        flat = x.detach().reshape(-1).float()
        a = flat.abs().max().item()
        self.amax[key] = max(self.amax.get(key, 0.0), a)
        self.count[key] = self.count.get(key, 0) + flat.numel()
        pick = torch.randint(0, flat.numel(), (self.n,), generator=self.gen)
        self.samples.setdefault(key, []).append(flat[pick].abs())


def install_stat_hooks(stats: KVStats):
    """Wrap the key/value calibration hooks llm-compressor registers."""
    from llmcompressor.modifiers.quantization.quantization import mixin

    orig_k, orig_v = mixin.calibrate_key_hook, mixin.calibrate_value_hook

    def key_hook(module, key_states):
        stats.observe(module, "k", key_states)
        return orig_k(module, key_states)

    def value_hook(module, value_states):
        stats.observe(module, "v", value_states)
        return orig_v(module, value_states)

    mixin.calibrate_key_hook, mixin.calibrate_value_hook = key_hook, value_hook


# ── main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BF16 (dequantized) Hy3 checkpoint")
    ap.add_argument("--fp8-ckpt", required=True, help="original W8A8 checkpoint (fingerprints)")
    ap.add_argument("--config", required=True, help="calibration data config JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--target-fp8-max", type=float, default=240.0, help="Trn2: 240")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    from compressed_tensors.quantization import QuantizationArgs
    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    cfg_text = Path(args.config).read_text()
    cfg = json.loads(cfg_text)
    tok = AutoTokenizer.from_pretrained(args.model)
    samples, sample_info = build_samples(cfg, tok)
    n_tok = sum(len(s) for s in samples)
    print(f"calibration samples {len(samples)}, tokens {n_tok}", flush=True)

    # Load with linearized experts (llm-compressor's MoE path) instead of a
    # post-load 3D -> 2D conversion; routing / numerics are unchanged.
    with load_quantizable_moe():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).eval()
    n_layers = model.config.num_hidden_layers
    attn_layers = {id(layer.self_attn): i for i, layer in enumerate(model.model.layers)}
    print(f"model loaded ({n_layers} layers) in {time.time() - t0:.0f}s", flush=True)
    w_before = {
        n: hashlib.sha256(p.detach().float().contiguous().numpy().tobytes()[:65536]).hexdigest()
        for n, p in list(model.named_parameters())[:8]
    }

    stats = KVStats(attn_layers)
    install_stat_hooks(stats)
    kv_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        dynamic=False,
        symmetric=True,
        observer="static_minmax",  # global absmax (default "minmax" is a moving average)
    )
    # KV only: no config_groups / scheme, so no Linear gets a weight or activation scheme.
    recipe = QuantizationModifier(kv_cache_scheme=kv_args)
    ds = Dataset.from_dict({"input_ids": samples, "attention_mask": [[1] * len(s) for s in samples]})
    t1 = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=len(samples),
        max_seq_length=max(len(s) for s in samples),
        pipeline="basic",
        batch_size=1,
        shuffle_calibration_samples=False,
        # KV statistics only need the routed experts; the default (every expert
        # sees every token, for weight calibration) costs ~num_experts/top_k x.
        moe_calibrate_all_experts=False,
    )
    calib_s = time.time() - t1
    print(f"oneshot done in {calib_s:.0f}s", flush=True)

    # Weights untouched (KV-only recipe).
    params = dict(model.named_parameters())
    for n, h0 in w_before.items():
        h = hashlib.sha256(params[n].detach().float().contiguous().numpy().tobytes()[:65536]).hexdigest()
        assert h == h0, f"llm-compressor modified weight {n}"

    k_d, v_d, rows = [], [], []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        ct_k = float(attn.k_scale.detach().float().reshape(-1)[0])
        ct_v = float(attn.v_scale.detach().float().reshape(-1)[0])
        ak, av = stats.amax[(i, "k")], stats.amax[(i, "v")]
        # llm-compressor's scale (absmax / 448, model dtype) must match the exact absmax.
        for name, ct, a in (("k", ct_k, ak), ("v", ct_v, av)):
            if not math.isclose(ct * TORCH_FP8_MAX, a, rel_tol=1e-2):
                raise RuntimeError(f"layer {i} {name}: llm-compressor scale {ct} vs absmax {a}")
        dk, dv = ak / args.target_fp8_max, av / args.target_fp8_max
        k_d.append(dk)
        v_d.append(dv)
        row = {"layer": i, "k_absmax": ak, "v_absmax": av, "k_dequant": dk, "v_dequant": dv,
               "ct_k_scale": ct_k, "ct_v_scale": ct_v}
        for name, d in (("k", dk), ("v", dv)):
            s = torch.cat(stats.samples[(i, name)])
            row[f"{name}_p99_9"] = torch.quantile(s[:1 << 24], 0.999).item()
            rt = (s / d).clamp(max=args.target_fp8_max).to(torch.float8_e4m3fn).float() * d
            row[f"{name}_rel_rms_err"] = ((rt - s).pow(2).mean().sqrt() / s.pow(2).mean().sqrt()).item()
        rows.append(row)

    fps = kv_scales.attn_fingerprints(args.fp8_ckpt, n_layers)
    meta = {
        "target_fp8_max": str(args.target_fp8_max),
        "kv_dtype": "float8_e4m3fn",
        "stat_position": "K after q/k-norm + RoPE (pre GQA repeat); V = v_proj output",
        "strategy": "per-tensor (one scalar per layer for K, one for V)",
        "observer": "llm-compressor static_minmax (global absmax); exact fp32 absmax exported",
        "formula": "dequant = absmax / target_fp8_max; quant = 1 / dequant",
        "calib_model": str(Path(args.model).resolve()),
        "calib_weights": "BF16 dequantized copy of the W8A8 checkpoint (activations not fake-quantized)",
        "fp8_ckpt": str(Path(args.fp8_ckpt).resolve()),
        "fp8_ckpt_index_sha256": kv_scales.file_sha256(str(Path(args.fp8_ckpt, "model.safetensors.index.json"))),
        "calib_config": cfg,
        "calib_config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "calib_samples": len(samples),
        "calib_tokens": n_tok,
        "versions": {p: md.version(p) for p in ("llmcompressor", "compressed-tensors", "transformers", "torch", "datasets")},
        "attn_fingerprints": fps,
    }
    sha = kv_scales.save_profile(args.out, k_d, v_d, meta)
    print(f"wrote {args.out} sha256 {sha}", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps({
            "profile": args.out, "profile_sha256": sha, "seconds_calibration": calib_s,
            "seconds_total": time.time() - t0, "samples": sample_info, "layers": rows,
            "meta": {k: v for k, v in meta.items() if k != "attn_fingerprints"},
        }, indent=1))


if __name__ == "__main__":
    main()
