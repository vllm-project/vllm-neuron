# SPDX-License-Identifier: Apache-2.0
"""KV Cache visualization: Y=layer, X=token, aggregated over heads."""

import json

import numpy as np
from typing import List, Optional

from vllm_neuron.accuracy.kv_cache_analysis import KVComparisonResult
from vllm_neuron.accuracy.utils import natural_sort_key


def _build_arrays(result: KVComparisonResult):
    """Build [layers, tokens] arrays with max L-inf and L2 over heads."""
    if not result:
        return {}, []
    layers = sorted(
        (k for k in result[0].keys() if not k.endswith("._bc")), key=natural_sort_key
    )
    num_tokens = len(result)
    num_layers = len(layers)

    k_linf = np.zeros((num_layers, num_tokens))
    v_linf = np.zeros((num_layers, num_tokens))
    k_l2 = np.zeros((num_layers, num_tokens))
    v_l2 = np.zeros((num_layers, num_tokens))
    k_bc = np.ones((num_layers, num_tokens))
    v_bc = np.ones((num_layers, num_tokens))
    # Three-way ratio: tgt_error / base_error (actual vs baseline / expected vs baseline)
    k_linf_ratio = np.ones((num_layers, num_tokens))
    v_linf_ratio = np.ones((num_layers, num_tokens))
    k_l2_ratio = np.ones((num_layers, num_tokens))
    v_l2_ratio = np.ones((num_layers, num_tokens))
    has_bc = False

    for t, token_data in enumerate(result):
        for l, layer in enumerate(layers):
            heads = token_data[layer]
            if not heads:
                continue
            k_linf[l, t] = max(h.k_linf for h in heads)
            v_linf[l, t] = max(h.v_linf for h in heads)
            k_l2[l, t] = max(h.k_l2 for h in heads)
            v_l2[l, t] = max(h.v_l2 for h in heads)
            bc_key = f"{layer}._bc"
            if bc_key in token_data:
                has_bc = True
                k_bc[l, t] = token_data[bc_key].k_bc
                v_bc[l, t] = token_data[bc_key].v_bc
                # Ratio: max tgt_linf / max base_linf (clamped to avoid div-by-zero)
                max_base_k = max(h.base_k_linf for h in heads)
                max_base_v = max(h.base_v_linf for h in heads)
                k_linf_ratio[l, t] = k_linf[l, t] / max(max_base_k, 1e-10)
                v_linf_ratio[l, t] = v_linf[l, t] / max(max_base_v, 1e-10)
                max_base_k_l2 = max(h.base_k_l2 for h in heads)
                max_base_v_l2 = max(h.base_v_l2 for h in heads)
                k_l2_ratio[l, t] = k_l2[l, t] / max(max_base_k_l2, 1e-10)
                v_l2_ratio[l, t] = v_l2[l, t] / max(max_base_v_l2, 1e-10)

    arrays = {"k_linf": k_linf, "v_linf": v_linf, "k_l2": k_l2, "v_l2": v_l2}
    if has_bc:
        arrays["k_bc"] = k_bc
        arrays["v_bc"] = v_bc
        arrays["k_linf_ratio"] = k_linf_ratio
        arrays["v_linf_ratio"] = v_linf_ratio
        arrays["k_l2_ratio"] = k_l2_ratio
        arrays["v_l2_ratio"] = v_l2_ratio
    return arrays, layers


def _add_annotations(
    fig, num_tokens, num_layers, prompt_len=None, divergence_indices=None
):
    """Add prefill/decode boundary and divergence markers to a heatmap figure."""

    # Prefill/decode boundary: vertical line at prompt_len - 0.5
    if prompt_len is not None and 0 < prompt_len < num_tokens:
        fig.add_vline(
            x=prompt_len - 0.5,
            line=dict(color="red", width=2, dash="solid"),
            annotation_text="decode →",
            annotation_position="top right",
            annotation=dict(font_size=10, font_color="red"),
        )
        fig.add_vline(
            x=prompt_len - 0.5,
            line=dict(color="red", width=0),
            annotation_text="← prefill",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="red"),
        )

    # Divergence points: vertical dashed lines
    if divergence_indices:
        for d in divergence_indices:
            if 0 <= d < num_tokens:
                fig.add_vline(
                    x=d,
                    line=dict(color="orange", width=2, dash="dash"),
                )


def export_html_report(
    result: KVComparisonResult,
    output_path: str,
    title: str = "KV Cache Analysis",
    zmax: float = 0.2,
    prompt_len: Optional[int] = None,
    divergence_indices: Optional[List[int]] = None,
) -> None:
    """Export HTML with heatmaps: Y=layer, X=token, max L-inf/L2 over heads.

    Args:
        result: KV comparison result
        output_path: Output HTML file path
        title: Report title
        zmax: Max value for color scale
        prompt_len: Number of prompt tokens (draws prefill/decode boundary)
        divergence_indices: Token indices where divergence occurred (draws markers)

    Example:
        >>> result = compare_kv_caches(expected_kv, actual_kv)
        >>> export_html_report(result, "kv_report.html", prompt_len=32)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("pip install plotly")

    if not result:
        print("No results to export")
        return

    arrays, layers = _build_arrays(result)
    num_layers, num_tokens = arrays["k_linf"].shape

    # BC colorscale: focus on 0.8-1.0 range, everything <0.8 is flat red
    bc_colorscale = [
        [0.0, "rgb(215,25,28)"],  # 0.0 - red
        [0.8, "rgb(215,25,28)"],  # 0.8 - still red (flat)
        [0.85, "rgb(253,174,97)"],  # 0.85 - orange
        [0.90, "rgb(255,255,191)"],  # 0.90 - yellow
        [0.95, "rgb(166,217,106)"],  # 0.95 - light green
        [1.0, "rgb(26,150,65)"],  # 1.0 - green
    ]

    def make_heatmap(
        data,
        name,
        colorscale="RdYlGn_r",
        zmin_val=0,
        zmax_val=zmax,
        cbar_title="Deviation",
        cbar_fmt=".1e",
    ):
        fig = go.Figure(
            data=go.Heatmap(
                z=data.tolist(),
                colorbar=dict(title=cbar_title, tickformat=cbar_fmt),
                colorscale=colorscale,
                zmin=zmin_val,
                zmax=zmax_val,
                xgap=1,
                ygap=1,
            )
        )
        _add_annotations(fig, num_tokens, num_layers, prompt_len, divergence_indices)
        fig.update_layout(
            title=name,
            xaxis_title="Token Index",
            yaxis_title="Model Layer",
            height=max(400, num_layers * 20),
            yaxis=dict(dtick=1),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # Build legend text
    legend_parts = [
        f"Tokens: {num_tokens}, Layers: {num_layers}, Heads: {len(result[0][layers[0]])} (max over heads)"
    ]
    if prompt_len is not None:
        legend_parts.append(
            f"Prompt: {prompt_len} tokens, Decode: {num_tokens - prompt_len} tokens"
        )
    if divergence_indices:
        legend_parts.append(f"Divergence at tokens: {divergence_indices}")
    legend_parts.append(
        "<span style='color:red'>Red line</span> = prefill/decode boundary, "
        "<span style='color:orange'>Orange dashed</span> = divergence point"
    )

    bc_html = ""
    if "k_bc" in arrays:
        bc_html = f"""
    <h2>Bhattacharyya Coefficient (BC) - Error Distribution Similarity</h2>
    <p>BC=1.0: baseline and target error distributions are identical. BC→0: no overlap (target has different error pattern).</p>
    <div class="heatmap">{make_heatmap(arrays["k_bc"], "K Cache - BC per Layer", colorscale=bc_colorscale, zmin_val=0, zmax_val=1, cbar_title="BC", cbar_fmt=".2f")}</div>
    <div class="heatmap">{make_heatmap(arrays["v_bc"], "V Cache - BC per Layer", colorscale=bc_colorscale, zmin_val=0, zmax_val=1, cbar_title="BC", cbar_fmt=".2f")}</div>"""

    # Three-way: show error ratios (actual/expected, both vs baseline). Ratio≈1 = same error magnitude.
    # Two-way: show raw L-inf/L2 errors.
    if "k_linf_ratio" in arrays:
        error_html = f"""
    <h2>Error Ratio: actual / expected (both vs baseline)</h2>
    <p>Ratio ≈ 1.0x: actual error matches expected. Ratio &gt;&gt; 1.0x: actual is worse. Green = good, red = bad.</p>
    <div class="heatmap">{make_heatmap(arrays["k_linf_ratio"], "K Cache - L-inf Ratio (actual/expected)", zmin_val=0, zmax_val=3, cbar_title="Ratio", cbar_fmt=".1f")}</div>
    <div class="heatmap">{make_heatmap(arrays["v_linf_ratio"], "V Cache - L-inf Ratio (actual/expected)", zmin_val=0, zmax_val=3, cbar_title="Ratio", cbar_fmt=".1f")}</div>
    <div class="heatmap">{make_heatmap(arrays["k_l2_ratio"], "K Cache - L2 Ratio (actual/expected)", zmin_val=0, zmax_val=3, cbar_title="Ratio", cbar_fmt=".1f")}</div>
    <div class="heatmap">{make_heatmap(arrays["v_l2_ratio"], "V Cache - L2 Ratio (actual/expected)", zmin_val=0, zmax_val=3, cbar_title="Ratio", cbar_fmt=".1f")}</div>"""
    else:
        error_html = f"""
    <div class="heatmap">{make_heatmap(arrays["k_linf"], "K Cache - Max Relative L-inf per Layer")}</div>
    <div class="heatmap">{make_heatmap(arrays["v_linf"], "V Cache - Max Relative L-inf per Layer")}</div>
    <div class="heatmap">{make_heatmap(arrays["k_l2"], "K Cache - Max Relative L2 per Layer")}</div>
    <div class="heatmap">{make_heatmap(arrays["v_l2"], "V Cache - Max Relative L2 per Layer")}</div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body {{ font-family: sans-serif; margin: 20px; }} .heatmap {{ margin: 20px 0; }} .legend {{ background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 10px 0; }}</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="legend">{"<br>".join(legend_parts)}</div>
    {error_html}
    {bc_html}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved: {output_path}")


def export_json(result: KVComparisonResult, output_path: str) -> None:
    """Export to JSON: [token][layer][head] = {k_cos, v_cos, k_linf, v_linf, k_l2, v_l2, ...}.

    Example:
        >>> result = compare_kv_caches(expected_kv, actual_kv)
        >>> export_json(result, "kv_metrics.json")
    """
    data = []
    for token in result:
        tok = {}
        for layer, val in token.items():
            if layer.endswith("._bc"):
                tok[layer] = {"k_bc": val.k_bc, "v_bc": val.v_bc}
            else:
                tok[layer] = [
                    {
                        "k_cos": h.k_cos,
                        "v_cos": h.v_cos,
                        "k_linf": h.k_linf,
                        "v_linf": h.v_linf,
                        "k_l2": h.k_l2,
                        "v_l2": h.v_l2,
                        "base_k_linf": h.base_k_linf,
                        "base_v_linf": h.base_v_linf,
                        "base_k_l2": h.base_k_l2,
                        "base_v_l2": h.base_v_l2,
                    }
                    for h in val
                ]
        data.append(tok)
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved: {output_path}")


def launch_dashboard(
    result: KVComparisonResult,
    port: int = 8050,
    prompt_len: Optional[int] = None,
    divergence_indices: Optional[List[int]] = None,
) -> None:
    """Interactive dashboard with metric/KV toggles, threshold sliders, and annotations.

    Example:
        >>> result = compare_kv_caches(expected_kv, actual_kv)
        >>> launch_dashboard(result, port=8050, prompt_len=32)
    """
    try:
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        from dash import Dash, Input, Output, dcc, html
    except ImportError:
        raise ImportError("pip install dash dash-bootstrap-components")

    if not result:
        print("No results")
        return

    arrays, layers = _build_arrays(result)
    num_layers, num_tokens = arrays["k_linf"].shape

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def transform(x):
        return 10**x

    slider_marks = {e: f"{transform(e):.0e}" for e in range(-5, 1)}

    app.layout = dbc.Container(
        [
            html.H1("KV Cache Analysis"),
            html.P(f"Tokens: {num_tokens}, Layers: {num_layers}"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("K/V"),
                            dcc.RadioItems(
                                id="kv", options=["K", "V"], value="K", inline=True
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Metric"),
                            dcc.RadioItems(
                                id="metric",
                                options=["L-inf", "L2"],
                                value="L-inf",
                                inline=True,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Label(id="low-label"),
                            dcc.Slider(
                                id="low",
                                min=-5,
                                max=0,
                                step=0.1,
                                value=-4,
                                marks=slider_marks,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label(id="high-label"),
                            dcc.Slider(
                                id="high",
                                min=-5,
                                max=0,
                                step=0.1,
                                value=-1,
                                marks=slider_marks,
                            ),
                        ],
                        width=3,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Graph(id="heatmap"),
        ],
        fluid=True,
    )

    @app.callback(
        Output("heatmap", "figure"),
        Output("low-label", "children"),
        Output("high-label", "children"),
        Input("kv", "value"),
        Input("metric", "value"),
        Input("low", "value"),
        Input("high", "value"),
    )
    def update(kv, metric, low, high):
        low_val, high_val = transform(low), transform(high)
        key = f"{kv.lower()}_{'linf' if metric == 'L-inf' else 'l2'}"
        data = arrays[key]
        fig = go.Figure(
            data=go.Heatmap(
                z=data.tolist(),
                colorbar=dict(title="Deviation", tickformat=".1e"),
                colorscale="Viridis",
                zmin=low_val,
                zmax=high_val,
                xgap=1,
                ygap=1,
            )
        )
        _add_annotations(fig, num_tokens, num_layers, prompt_len, divergence_indices)
        fig.update_layout(
            title=f"{kv} Cache - Max Relative {metric} per Layer",
            xaxis_title="Token Index",
            yaxis_title="Model Layer",
            height=max(500, num_layers * 20),
            yaxis=dict(dtick=1),
        )
        return fig, f"Low: {low_val:.1e}", f"High: {high_val:.1e}"

    print(f"Starting dashboard at http://localhost:{port}")
    app.run_server(port=port, debug=False)


def export_aggregated_bc_html(
    aggregated_result: dict,
    output_path: str,
    title: str = "Aggregated KV Cache BC Analysis",
    divergence_counts: dict = None,
    num_prompts: int = None,
) -> None:
    """Export HTML heatmap for aggregated BC results (Y=layer, X=generation token).

    Args:
        aggregated_result: Dict[gen_token_idx, Dict[layer, TokenKVMetrics]] from aggregate_kv_bc_across_prompts()
        output_path: Output HTML file path
        title: Report title
        divergence_counts: Dict[gen_token_idx, int] - number of prompts diverged at each gen token
        num_prompts: Total number of prompts (for divergence percentage)

    Example:
        >>> agg = aggregate_kv_bc_across_prompts(all_raw_errors, prompt_lens)
        >>> export_aggregated_bc_html(agg, "aggregated_bc.html", num_prompts=3)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("pip install plotly")

    if not aggregated_result:
        print("No results to export")
        return

    gen_tokens = sorted(aggregated_result.keys())
    layers = sorted(aggregated_result[gen_tokens[0]].keys(), key=natural_sort_key)
    num_tokens = len(gen_tokens)
    num_layers = len(layers)

    k_bc = np.ones((num_layers, num_tokens))
    v_bc = np.ones((num_layers, num_tokens))

    for t_idx, gen_t in enumerate(gen_tokens):
        token_data = aggregated_result[gen_t]
        for l_idx, layer in enumerate(layers):
            if layer in token_data:
                k_bc[l_idx, t_idx] = token_data[layer].k_bc
                v_bc[l_idx, t_idx] = token_data[layer].v_bc

    # BC colorscale: focus on 0.8-1.0 range, everything <0.8 is flat red
    bc_colorscale = [
        [0.0, "rgb(215,25,28)"],
        [0.8, "rgb(215,25,28)"],
        [0.85, "rgb(253,174,97)"],
        [0.90, "rgb(255,255,191)"],
        [0.95, "rgb(166,217,106)"],
        [1.0, "rgb(26,150,65)"],
    ]

    def make_heatmap(data, name):
        fig = go.Figure(
            data=go.Heatmap(
                z=data.tolist(),
                x=gen_tokens,
                colorbar=dict(title="BC", tickformat=".2f"),
                colorscale=bc_colorscale,
                zmin=0,
                zmax=1,
                xgap=1,
                ygap=1,
            )
        )
        fig.update_layout(
            title=name,
            xaxis_title="Generation Token Index",
            yaxis_title="Model Layer",
            height=max(400, num_layers * 20),
            yaxis=dict(dtick=1),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # Build divergence info string
    div_info = ""
    if divergence_counts:
        div_tokens = sorted(divergence_counts.keys())
        div_parts = [
            f"Gen {t}: {divergence_counts[t]}/{num_prompts or '?'}"
            for t in div_tokens[:10]
        ]
        if len(div_tokens) > 10:
            div_parts.append(f"... ({len(div_tokens)} total)")
        div_info = f"<br><b>Divergences:</b> {', '.join(div_parts)}"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body {{ font-family: sans-serif; margin: 20px; }} .heatmap {{ margin: 20px 0; }} .legend {{ background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 10px 0; }}</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="legend">
        Generation Tokens: {num_tokens}, Layers: {num_layers}, Prompts: {num_prompts or "N/A"}<br>
        BC aggregated across multiple prompts per generation token index.<br>
        <span style='color:green'>Green (BC≈1)</span> = target errors match expected dtype errors.
        <span style='color:red'>Red (BC≈0)</span> = vLLM has different error pattern.
        {div_info}
    </div>
    <div class="heatmap">{make_heatmap(k_bc, "K Cache - Aggregated BC per Layer")}</div>
    <div class="heatmap">{make_heatmap(v_bc, "V Cache - Aggregated BC per Layer")}</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved: {output_path}")
