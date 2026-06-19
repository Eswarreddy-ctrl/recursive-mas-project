"""Plotly chart builders for the benchmark dashboard.

Each function takes a benchmark dict (the BenchmarkResult shape) plus the
speedup/summary structures and returns a Plotly Figure. Charts are styled to
read like an academic benchmark paper: clean grid, muted palette, clear titles.
"""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go

# Academic-paper palette
TRAD_COLOR = "#B23A48"   # muted red — the baseline
R1_COLOR = "#5B8DB8"
R2_COLOR = "#3E6F94"
R3_COLOR = "#1F4E79"     # deep blue — best
RECURSIVE_SEQ = [R1_COLOR, R2_COLOR, R3_COLOR]
GRID = "#E5E5E5"

LABELS = ["Traditional", "R1", "R2", "R3"]


def _base_layout(title: str, xtitle: str, ytitle: str) -> dict:
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="#1A1A1A")),
        xaxis=dict(title=xtitle, showgrid=False, zeroline=False),
        yaxis=dict(title=ytitle, showgrid=True, gridcolor=GRID, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Georgia, serif", size=13, color="#333"),
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=False,
    )


def chart_inference_time(b: Dict) -> go.Figure:
    """Chart 1 — Inference Time Comparison (bar)."""
    y = [b["traditional_time"], b["recursive_r1_time"], b["recursive_r2_time"], b["recursive_r3_time"]]
    fig = go.Figure(
        go.Bar(
            x=LABELS, y=y,
            marker_color=[TRAD_COLOR, *RECURSIVE_SEQ],
            text=[f"{v:.2f}s" for v in y], textposition="outside",
        )
    )
    fig.update_layout(**_base_layout("Inference Time Comparison", "Architecture", "Seconds"))
    return fig


def chart_tokens(b: Dict) -> go.Figure:
    """Chart 2 — Token Consumption (bar)."""
    y = [b["traditional_tokens"], b["recursive_r1_tokens"], b["recursive_r2_tokens"], b["recursive_r3_tokens"]]
    fig = go.Figure(
        go.Bar(
            x=LABELS, y=y,
            marker_color=[TRAD_COLOR, *RECURSIVE_SEQ],
            text=[f"{v:,}" for v in y], textposition="outside",
        )
    )
    fig.update_layout(**_base_layout("Token Consumption", "Architecture", "Total Tokens"))
    return fig


def chart_quality(b: Dict) -> go.Figure:
    """Chart 3 — Quality Score Improvement (line)."""
    y = [b["traditional_quality"], b["recursive_r1_quality"], b["recursive_r2_quality"], b["recursive_r3_quality"]]
    fig = go.Figure(
        go.Scatter(
            x=LABELS, y=y, mode="lines+markers+text",
            line=dict(color=R3_COLOR, width=3),
            marker=dict(size=11, color=[TRAD_COLOR, *RECURSIVE_SEQ]),
            text=[f"{v:.1f}" for v in y], textposition="top center",
        )
    )
    layout = _base_layout("Quality Score Improvement", "Architecture", "Quality (0–10)")
    layout["yaxis"]["range"] = [0, 10.5]
    fig.update_layout(**layout)
    return fig


def chart_cost(b: Dict) -> go.Figure:
    """Chart 4 — Cost Comparison (bar)."""
    y = [b["traditional_cost"], b["recursive_r1_cost"], b["recursive_r2_cost"], b["recursive_r3_cost"]]
    fig = go.Figure(
        go.Bar(
            x=LABELS, y=y,
            marker_color=[TRAD_COLOR, *RECURSIVE_SEQ],
            text=[f"${v:.5f}" for v in y], textposition="outside",
        )
    )
    fig.update_layout(**_base_layout("Cost Comparison", "Architecture", "Estimated Cost (USD)"))
    return fig


def chart_scalability(speedups: List[Dict]) -> go.Figure:
    """Chart 6 — RecursiveMAS Scalability Curve (time saved % vs depth)."""
    depths = [1, 2, 3]
    saved = [s["time_saved_pct"] for s in speedups]
    fig = go.Figure(
        go.Scatter(
            x=depths, y=saved, mode="lines+markers+text",
            line=dict(color=R3_COLOR, width=3, shape="spline"),
            marker=dict(size=12, color=R3_COLOR),
            text=[f"{v:.1f}%" for v in saved], textposition="top center",
            fill="tozeroy", fillcolor="rgba(31,78,121,0.08)",
        )
    )
    layout = _base_layout("RecursiveMAS Scalability Curve", "Recursion Depth", "Time Saved (%)")
    layout["xaxis"]["tickmode"] = "array"
    layout["xaxis"]["tickvals"] = depths
    fig.update_layout(**layout)
    return fig


def chart_quality_vs_latency(b: Dict) -> go.Figure:
    """Chart 7 — Quality vs Latency Tradeoff (scatter)."""
    pts = [
        ("Traditional", b["traditional_time"], b["traditional_quality"], TRAD_COLOR),
        ("R1", b["recursive_r1_time"], b["recursive_r1_quality"], R1_COLOR),
        ("R2", b["recursive_r2_time"], b["recursive_r2_quality"], R2_COLOR),
        ("R3", b["recursive_r3_time"], b["recursive_r3_quality"], R3_COLOR),
    ]
    fig = go.Figure()
    for name, lat, qual, color in pts:
        fig.add_trace(
            go.Scatter(
                x=[lat], y=[qual], mode="markers+text",
                marker=dict(size=18, color=color, line=dict(width=1, color="white")),
                text=[name], textposition="top center", name=name,
            )
        )
    layout = _base_layout("Quality vs Latency Tradeoff", "Latency (seconds)", "Quality (0–10)")
    layout["yaxis"]["range"] = [0, 10.5]
    fig.update_layout(**layout)
    return fig
