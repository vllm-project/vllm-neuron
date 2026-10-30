# SPDX-License-Identifier: Apache-2.0
"""Run the tiny checkpoint through vLLM on Neuron (CPU mode) and print the result as JSON.

Run in its own process by ``test_tiny_e2e.py`` (one engine per process).

    VLLM_NEURON_CPU_MODE=1 python run_tiny.py CKPT --tp 4 --mode text|image
"""

import argparse
import json
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--mode", choices=["text", "image"], default="text")
ap.add_argument("--max-model-len", type=int, default=1024)
ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                help="float32: exact structural check against the HF fp32 goldens")
args = ap.parse_args()
os.environ["MUSE_GLIMMER_COMPUTE_DTYPE"] = args.dtype

from muse_glimmer_neuron import register  # noqa: E402

register()
from vllm import LLM, SamplingParams  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_checkpoint import synth_image  # noqa: E402

gold = torch.load(os.path.join(args.ckpt, "goldens.pt"), weights_only=False)
mml = args.max_model_len
additional = {"neuron_config": {"num_batched_tokens_buckets": sorted({128, mml}), "num_seqs_buckets": [1],
                                "on_device_sampling_config": {"all_greedy": True}}}
kw = {}
if args.mode == "image":
    additional["vision_neuron_config"] = {"vision_attention_block_size": 4096, "encoder_cache_num_blocks": 8,
                                          "num_vision_tokens_buckets": [4096], "tp_size": args.tp}
    kw = dict(limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_image_tokens": 1024})
else:
    kw = dict(skip_tokenizer_init=True)
llm = LLM(model=args.ckpt, tensor_parallel_size=args.tp, max_model_len=mml, max_num_batched_tokens=mml,
          max_num_seqs=1, enforce_eager=True, dtype=args.dtype, enable_prefix_caching=False, num_gpu_blocks_override=256,
          additional_config=additional, **kw)
sp = SamplingParams(max_tokens=gold["new_tokens"], temperature=0.0, ignore_eos=True,
                    detokenize=args.mode == "image")
res = {}
for name, g in gold[args.mode].items():
    if args.mode == "text":
        req = {"prompt_token_ids": g["prompt_token_ids"]}
    else:
        req = {"prompt": g["prompt"], "multi_modal_data": {"image": synth_image(*g["image"])}}
    o = llm.generate([req], sp)[0]
    res[name] = {"prompt_token_ids": list(o.prompt_token_ids), "tokens": list(o.outputs[0].token_ids)}
print("RESULT_JSON " + json.dumps(res), flush=True)
