"""Needle across prefill chunks: a code at the start of a ~6K-token prompt, asked for at the end.

    python needle_check.py <served name> <tokenizer dir> [port]
"""
import json, sys, urllib.request
from transformers import AutoTokenizer

name, tok_dir = sys.argv[1], sys.argv[2]
port = sys.argv[3] if len(sys.argv) > 3 else "8000"
tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
filler = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. " * 400)
ids = tok(filler)["input_ids"][:5800]
filler = tok.decode(ids)
msg = ("Remember this: the secret code is 7342.\n\n" + filler +
       "\n\nWhat is the secret code mentioned at the very beginning? Answer with the number only.")
prompt = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
n = len(tok(prompt)["input_ids"])
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({"model": name, "prompt": prompt, "max_tokens": 400, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
text = json.load(urllib.request.urlopen(req, timeout=1200))["choices"][0]["text"]
print(f"needle prompt {n} tokens -> {text!r}")
print("NEEDLE", "PASS" if "7342" in text else "FAIL")
