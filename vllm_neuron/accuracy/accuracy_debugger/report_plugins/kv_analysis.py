# SPDX-License-Identifier: Apache-2.0
"""KV cache analysis plugin - compares KV caches between Neuron and HF.

This plugin parses the KV cache three-way comparison output from the
prompt analysis step. The comparison works as follows:

1. Three KV caches are extracted for the same prompt:
   - FP32: HuggingFace model in FP32 (ground truth)
   - BF16: HuggingFace model in BF16 (expected dtype noise)
   - Neuron: vllm-neuron model (target under test)

2. Per-layer, per-head metrics are computed:
   - L-inf: max absolute error (per attention head)
   - L2: L2 norm of error (per attention head)
   - BC (Bhattacharyya Coefficient): statistical overlap between the
     error distributions of (BF16 vs FP32) and (Neuron vs FP32).
     BC=1.0 means identical distributions; BC<0.9 indicates anomalous error.

3. Pass/fail is determined by the minimum BC across all layers and tokens.
   BC >= BC_THRESHOLD (0.9) everywhere means the KV errors are within
   expected BF16 noise. A layer with anomalously low BC suggests a bug
   in that layer's KV write path.
"""

import os
import re
from typing import Dict, Optional

from .base import ReportPlugin, PluginRegistry


@PluginRegistry.register
class KVAnalysisPlugin(ReportPlugin):
    """Plugin for KV cache analysis (step 2).

    Parses the structured log output from kv_cache_analysis.py which prints
    per-token, per-layer metrics in the format:
        === Token N ===
          layers.X.self_attn:
            K cos: [...] L-inf: [...] L2: [...]
            V cos: [...] L-inf: [...] L2: [...]
            BC: K=0.9876 V=0.9543

    Also embeds the interactive plotly heatmap (kv_report.html) generated
    by the KvCachePlugin during prompt analysis.
    """

    name = "kv_analysis"

    # Minimum BC threshold for pass/fail. Layers with BC below this are
    # flagged as potential KV write bugs. 0.9 is conservative — values
    # between 0.85-0.9 are borderline and may be prompt-dependent.
    BC_THRESHOLD = 0.9

    display_name = "KV Cache"
    step_index = 2
    guide_text = """Compares KV caches between Neuron and HF per layer and per head.
<ul>
<li><b>BC ≥ 0.90</b> everywhere → KV error matches BF16 baseline</li>
<li>A single layer with anomalously low BC → bug in that layer's KV write</li>
<li>Error growing across tokens → possible decode KV corruption or block-table bug</li>
<li>Top bar charts show max L-inf and BC per layer; heatmaps show per-head detail</li>
</ul>"""

    def get_log_path(self) -> Optional[str]:
        """Find the log file containing KV analysis output.

        Tries step-specific log (step2_kv_analysis.txt) first, then falls
        back to validation_log.txt which contains all plugin output.
        """
        if not self.prompt_dir:
            return None
        path = os.path.join(self.prompt_dir, f"step{self.step_index}_{self.name}.txt")
        if os.path.isfile(path):
            return path
        fallback = os.path.join(self.prompt_dir, "validation_log.txt")
        return fallback if os.path.isfile(fallback) else None

    def parse_log(self, log_path: str) -> Optional[Dict]:
        """Parse KV cache analysis log into structured per-token, per-layer metrics.

        Returns dict with structure:
            {"tokens": {
                0: {"layers.0.self_attn": {"k_linf": [...], "v_linf": [...],
                                           "k_l2": [...], "v_l2": [...],
                                           "k_bc": 0.98, "v_bc": 0.95}, ...},
                1: {...}, ...
            }}

        The lists (k_linf, v_linf, etc.) contain per-head values.
        k_bc/v_bc are scalar BC values aggregated across heads for that layer.
        """
        if not os.path.isfile(log_path):
            return None
        with open(log_path) as f:
            text = self.strip_ansi(f.read())

        tokens = {}
        current_token = None
        current_layer = None

        for line in text.split("\n"):
            # Match token header: "=== Token N ==="
            tm = re.match(r"=== Token (\d+) ===", line)
            if tm:
                current_token = int(tm.group(1))
                tokens[current_token] = {}
                current_layer = None
                continue
            if current_token is None:
                continue

            # Match layer header: "  layers.X.self_attn:"
            lm = re.match(r"\s+(layers\.\d+\.self_attn):", line)
            if lm:
                current_layer = lm.group(1)
                tokens[current_token][current_layer] = {}
                continue

            if current_layer:
                # Match K metrics: "    K cos: [...] L-inf: [...] L2: [...]"
                km = re.match(
                    r"\s+K cos: \[(.*?)\]\s+L-inf: \[(.*?)\]\s+L2: \[(.*?)\]", line
                )
                if km:
                    tokens[current_token][current_layer]["k_linf"] = [
                        float(x) for x in km.group(2).split(",")
                    ]
                    tokens[current_token][current_layer]["k_l2"] = [
                        float(x) for x in km.group(3).split(",")
                    ]

                # Match V metrics: "    V cos: [...] L-inf: [...] L2: [...]"
                vm = re.match(
                    r"\s+V cos: \[(.*?)\]\s+L-inf: \[(.*?)\]\s+L2: \[(.*?)\]", line
                )
                if vm:
                    tokens[current_token][current_layer]["v_linf"] = [
                        float(x) for x in vm.group(2).split(",")
                    ]
                    tokens[current_token][current_layer]["v_l2"] = [
                        float(x) for x in vm.group(3).split(",")
                    ]

                # Match BC line: "    BC: K=0.9876 V=0.9543"
                # This is the last line per layer, so reset current_layer after parsing
                bm = re.match(r"\s+BC: K=([\d.]+)\s+V=([\d.]+)", line)
                if bm:
                    tokens[current_token][current_layer]["k_bc"] = float(bm.group(1))
                    tokens[current_token][current_layer]["v_bc"] = float(bm.group(2))
                    current_layer = None

        return {"tokens": tokens} if tokens else None

    def load_artifacts(self, artifact_dir: str) -> Optional[Dict]:
        """Load pre-generated HTML artifacts for embedding in the report.

        Looks for kv_analysis/kv_report.html: interactive plotly heatmap
        showing per-layer, per-head BC and L-inf as color-coded matrices.
        """
        result = {}
        search_dirs = [artifact_dir]
        if self.prompt_dir and self.prompt_dir != artifact_dir:
            search_dirs.append(self.prompt_dir)
        for base in search_dirs:
            if not base:
                continue
            kv_report = os.path.join(base, "kv_analysis", "kv_report.html")
            if os.path.isfile(kv_report):
                with open(kv_report) as f:
                    result["kv_report_html"] = self.extract_body(f.read())
                break
        return result or None

    def check_status(self, data: Dict) -> tuple[bool, str]:
        """Determine pass/fail based on minimum BC across all layers and tokens."""
        bcs = []
        for token_data in data.get("tokens", {}).values():
            for layer in token_data.values():
                bcs.append(layer.get("k_bc", 1.0))
                bcs.append(layer.get("v_bc", 1.0))
        min_bc = min(bcs) if bcs else 0
        n_tokens = len(data.get("tokens", {}))
        if min_bc >= self.BC_THRESHOLD:
            return True, f"Min BC: {min_bc:.4f} ({n_tokens} tokens)"
        return False, f"Min BC: {min_bc:.4f} (< {self.BC_THRESHOLD}, {n_tokens} tokens)"

    def build_html(self, data: Dict) -> str:
        """Build HTML fragment for the KV cache section of the report."""
        parts = []

        # Compute and display min BC as overall status indicator
        bcs = []
        for token_data in data.get("tokens", {}).values():
            for layer in token_data.values():
                bcs.append(layer.get("k_bc", 1.0))
                bcs.append(layer.get("v_bc", 1.0))
        if bcs:
            min_bc = min(bcs)
            n_tokens = len(data.get("tokens", {}))
            cls = "pass" if min_bc >= self.BC_THRESHOLD else "fail"
            parts.append(
                f'<p>Min BC: <span class="{cls}">{min_bc:.4f}</span> ({n_tokens} tokens)</p>'
            )

        # Embed the interactive plotly heatmap (generated by KvCachePlugin.save())
        if data.get("kv_report_html"):
            parts.append(data["kv_report_html"])
        elif not bcs:
            parts.append("<p>No KV cache data available.</p>")

        return "\n".join(parts)

    def build_text_summary(self, data: Dict) -> str:
        """Build plain-text summary listing only layers that fail BC threshold."""
        lines = []
        passed, detail = self.check_status(data)
        lines.append(
            f"KV Cache Analysis: {'PASSED' if passed else 'FAILED'} — {detail}"
        )
        tokens = data.get("tokens", {})
        for tok_id, token_data in sorted(tokens.items()):
            for layer, metrics in sorted(token_data.items()):
                k_bc = metrics.get("k_bc", 0)
                v_bc = metrics.get("v_bc", 0)
                if k_bc < self.BC_THRESHOLD or v_bc < self.BC_THRESHOLD:
                    lines.append(
                        f"  Token {tok_id} {layer}: K_BC={k_bc:.4f} V_BC={v_bc:.4f}"
                    )
        return "\n".join(lines)
