# SPDX-License-Identifier: Apache-2.0
"""A small tool-using agent against the served model, printing what each step costs.

The agent answers a question about a small repository held in memory: it can list the files, read one, and search
for a string. Nothing is executed and nothing outside this script is read. Each step prints the prompt tokens the
server saw, the time to the first streamed token, the decode rate and the tool the model called, so the cost of
re-reading the growing history at every step (the model runs without a prefix cache) is visible.

Start the server as in the README, with tool calling and the reasoning parser:

    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3

then run:

    python contrib/models/qwen38/examples/agent_demo.py --port 8000
"""

import argparse
import json
import time
import urllib.request

REPO = {
    "README.md": "# tinystats\nSmall statistics helpers. `mean`, `variance` and `median` live in stats.py.\n",
    "stats.py": (
        "def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
        "def variance(xs):\n    m = mean(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs) - 1\n\n\n"
        "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n"
        "    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2\n"
    ),
    "test_stats.py": (
        "from stats import mean, variance, median\n\n\n"
        "def test_variance():\n    assert variance([1, 2, 3, 4]) == 1.25\n"
    ),
}

TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List the files of the repository",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read one file of the repository",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "Lines of the repository containing a string",
                                      "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                                                     "required": ["text"]}}},
]


def run_tool(name, args):
    if name == "list_files":
        return json.dumps(sorted(REPO))
    if name == "read_file":
        return REPO.get(args.get("path", ""), f"no such file: {args.get('path')}")
    if name == "search":
        hits = [f"{p}:{i + 1}: {line}" for p, body in REPO.items() for i, line in enumerate(body.splitlines())
                if args.get("text", "") in line]
        return "\n".join(hits) or "no match"
    return f"unknown tool {name}"


def step(url, model, messages, sampling):
    """One streamed chat completion: returns (message, prompt tokens, completion tokens, TTFT, decode tok/s)."""
    body = dict({"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 4096, "stream": True,
                 "stream_options": {"include_usage": True}}, **sampling)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    content, calls, usage = "", {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            msg = json.loads(line[6:])
            usage = msg.get("usage") or usage
            for ch in msg.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") \
                        or delta.get("tool_calls"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": None, "arguments": ""})
                    c["id"] = c["id"] or tc.get("id")
                    fn = tc.get("function") or {}
                    c["name"] = fn.get("name") or c["name"]
                    c["arguments"] += fn.get("arguments") or ""
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / (last - first) if first and last and last > first and n > 1 else 0.0
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                 for i, c in sorted(calls.items())]
    return message, usage.get("prompt_tokens", 0), n, (first - t0) if first else 0.0, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--greedy", action="store_true", help="temperature 0 instead of the recommended sampling")
    ap.add_argument("--system-words", type=int, default=0,
                    help="pad the system prompt with this many words of project guidelines, as a coding agent's long "
                         "instructions would")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    sampling = {"temperature": 0} if a.greedy else {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    guidelines = ""
    if a.system_words:
        rule = ("Keep functions small, name things for what they hold, test every public function, prefer the "
                "standard library, and explain each change in the commit message. ")
        words = rule.split()
        guidelines = "\n\nProject guidelines:\n" + " ".join(words[i % len(words)] for i in range(a.system_words))
    messages = [{"role": "system", "content": "You are a coding assistant. Use the tools to inspect the repository "
                                              "before answering." + guidelines},
                {"role": "user", "content": "The variance test fails. Find the bug and tell me the fix."}]
    print("step  prompt_tok  out_tok  TTFT(s)  decode tok/s  action")
    for i in range(a.max_steps):
        msg, ptok, ctok, ttft, rate = step(url, a.model, messages, sampling)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        action = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls) or "answer"
        print(f"{i + 1:4d}  {ptok:10d}  {ctok:7d}  {ttft:7.2f}  {rate:12.1f}  {action}", flush=True)
        if not calls:
            print("\n" + (msg["content"] or "").strip())
            return
        for c in calls:
            args = json.loads(c["function"]["arguments"] or "{}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(c["function"]["name"], args)})
    print("stopped after", a.max_steps, "steps")


if __name__ == "__main__":
    main()
