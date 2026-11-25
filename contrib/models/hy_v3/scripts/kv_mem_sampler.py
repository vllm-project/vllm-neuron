"""Reduce a neuron-monitor JSON stream to one compact memory sample per report.

Usage: neuron-monitor | python kv_mem_sampler.py <out.jsonl>

Each output line: {"t": unix time, "runtimes": n, "device": {category: bytes,
... summed over all runtimes and NeuronCores}, "device_total": bytes,
"host_total": bytes}. Categories are the runtime's own
(neuroncore_memory_usage: constants, model_code, model_shared_scratchpad,
runtime_memory, tensors). kv_report.py takes the per-category maximum over the
run as the peak.
"""

import json
import sys
import time


def reduce(rep):
    device, n, dev_total, host_total = {}, 0, 0, 0
    for rt in rep.get("neuron_runtime_data", []):
        mu = rt.get("report", {}).get("memory_used", {}).get("neuron_runtime_used_bytes", {})
        if not mu:
            continue
        n += 1
        dev_total += mu.get("neuron_device", 0)
        host_total += mu.get("host", 0)
        for core in mu.get("usage_breakdown", {}).get("neuroncore_memory_usage", {}).values():
            for k, v in core.items():
                device[k] = device.get(k, 0) + v
    return {"t": time.time(), "runtimes": n, "device": device,
            "device_total": dev_total, "host_total": host_total}


def main():
    with open(sys.argv[1], "a") as f:
        for line in sys.stdin:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                f.write(json.dumps(reduce(json.loads(line))) + "\n")
                f.flush()
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    main()
