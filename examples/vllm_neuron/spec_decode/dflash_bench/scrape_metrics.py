#!/usr/bin/env python3
"""Pull the spec-decode counters out of a vLLM server's /metrics endpoint."""

import json
import sys

import requests

KEYS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:num_preemptions",
    "vllm:generation_tokens",
    "vllm:prompt_tokens",
)

text = requests.get(f"{sys.argv[1]}/metrics", timeout=60).text
out = {}
for line in text.splitlines():
    if line.startswith("#") or not line.strip():
        continue
    name = line.split("{")[0].split(" ")[0]
    if not any(name.startswith(k) for k in KEYS):
        continue
    out.setdefault(name, []).append(line)

json.dump(out, sys.stdout, indent=2)
print()
