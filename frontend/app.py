"""RecursiveMAS Benchmark Dashboard (Streamlit).

A research-paper-styled dashboard comparing Traditional MAS against
RecursiveMAS across three recursion rounds.

Run:
    streamlit run frontend/app.py

It talks to the FastAPI backend at API_URL (default http://localhost:8000).
If the backend is unreachable, it falls back to bundled sample data so the
dashboard is always demonstrable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

from charts.plotly_charts import (
    chart_cost,
    chart_inference_time,
    chart_quality,
    chart_quality_vs_latency,
    chart_scalability,
    chart_tokens,
)
from components.kpi_cards import render_speedup_cards, render_summary_metrics

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_benchmarks.json"

st.set_page_config(page_title="RecursiveMAS Benchmark", layout="wide", page_icon="📊")

# --------------------------------------------------------------------------- #
# Styling — academic paper aesthetic
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .main { background-color: #FBFBF9; }
      h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #1A1A1A; }
      .paper-title { text-align:center; font-size:34px; font-weight:700; margin-bottom:0; }
      .paper-sub { text-align:center; color:#666; font-style:italic; margin-top:4px; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="paper-title">RecursiveMAS vs Traditional MAS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="paper-sub">A Benchmark of Inference Time Speedup Across Three Recursion Rounds</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_live(query: str) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/benchmark", json={"query": query}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Backend unavailable ({e}). Using bundled sample data.")
        return None


def load_sample() -> dict:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sample_to_response(record: dict) -> dict:
    """Adapt a flat sample BenchmarkResult record into a response-like dict."""
    speedups = []
    for i, key in enumerate(["recursive_r1_time", "recursive_r2_time", "recursive_r3_time"], 1):
        rt = max(record[key], 1e-6)
        t = max(record["traditional_time"], 1e-6)
        speedups.append(
            {
                "round_label": f"Round {i}",
                "recursive_time": record[key],
                "speedup": round(t / rt, 2),
                "time_saved_pct": round((1 - rt / t) * 100, 1),
            }
        )
    best_i = min(range(3), key=lambda i: record[f"recursive_r{i+1}_time"])
    best_time = record[f"recursive_r{best_i+1}_time"]
    best_tok = record[f"recursive_r{best_i+1}_tokens"]
    best_cost = record[f"recursive_r{best_i+1}_cost"]
    best_qual = record[f"recursive_r{best_i+1}_quality"]
    t = max(record["traditional_time"], 1e-6)
    summary = {
        "best_speedup": round(t / max(best_time, 1e-6), 2),
        "token_reduction_pct": round((1 - best_tok / max(record["traditional_tokens"], 1)) * 100, 1),
        "cost_reduction_pct": round((1 - best_cost / max(record["traditional_cost"], 1e-9)) * 100, 1),
        "quality_delta_pct": round((best_qual / max(record["traditional_quality"], 1e-6) - 1) * 100, 1),
    }
    summary["narrative"] = (
        f"RecursiveMAS (best round R{best_i+1}) achieved {summary['best_speedup']}x faster "
        f"inference, {summary['token_reduction_pct']}% lower token usage, "
        f"{summary['cost_reduction_pct']}% lower cost, and "
        f"{abs(summary['quality_delta_pct'])}% "
        f"{'higher' if summary['quality_delta_pct'] >= 0 else 'lower'} quality vs Traditional MAS."
    )
    return {"benchmark": record, "speedups": speedups, "summary": summary, "response": None}


# --------------------------------------------------------------------------- #
# Section 1 — Query input
# --------------------------------------------------------------------------- #
st.markdown("## 1. Query")
default_q = "Payment was deducted twice but my order is not confirmed. No reply from support. Please fix urgently."

with st.sidebar:
    st.header("Configuration")
    st.text_input("API URL", value=API_URL, key="api_url_display", disabled=True)
    st.markdown("**Mode**")
    try:
        cfg = requests.get(f"{API_URL}/config", timeout=5).json()
        st.success(f"Backend: {cfg['backend']}")
    except Exception:
        st.warning("Backend offline — sample mode")
    st.markdown("---")
    st.markdown(
        "**Sample queries** can be loaded below if the backend is offline. "
        "Mock figures are simulated from real context sizes."
    )

query = st.text_area("Enter a support query to benchmark:", value=default_q, height=100)
col_run, col_sample = st.columns([1, 1])
run_clicked = col_run.button("▶ Run Benchmark", type="primary", use_container_width=True)
sample_clicked = col_sample.button("📁 Load Sample Result", use_container_width=True)

data = None
if run_clicked and query.strip():
    with st.spinner("Running Traditional MAS + 3 Recursive rounds..."):
        data = run_live(query.strip())
    if data is None:
        data = sample_to_response(load_sample()["results"][0])
elif sample_clicked:
    sample = load_sample()
    options = {r["query"][:70]: r for r in sample["results"]}
    pick = st.selectbox("Choose a sample query:", list(options.keys()))
    data = sample_to_response(options[pick])

if data is None:
    st.info("Enter a query and click **Run Benchmark**, or load a sample result to explore the dashboard.")
    st.stop()

bench = data["benchmark"]
speedups = data["speedups"]
summary = data["summary"]

# --------------------------------------------------------------------------- #
# Section 2 — Generated responses (tabs)
# --------------------------------------------------------------------------- #
st.markdown("## 2. Generated Responses")
tabs = st.tabs(["Traditional MAS", "Recursive R1", "Recursive R2", "Recursive R3"])
resp = data.get("response")
if resp:
    contents = [resp["traditional"]["response"]] + [r["response"] for r in resp["recursive_rounds"]]
    metas = [resp["traditional"]] + resp["recursive_rounds"]
    for tab, content, meta in zip(tabs, contents, metas):
        with tab:
            st.write(content or "_(no text)_")
            st.caption(
                f"⏱ {meta['duration_ms']/1000:.2f}s · 🔢 {meta['total_tokens']:,} tokens · "
                f"📞 {meta['llm_calls']} calls · 💲 ${meta['estimated_cost']:.5f} · "
                f"⭐ quality {meta['quality']['average'] if isinstance(meta['quality'], dict) else 'n/a'}"
            )
else:
    msgs = [
        ("Traditional MAS", "traditional"),
        ("Recursive R1", "recursive_r1"),
        ("Recursive R2", "recursive_r2"),
        ("Recursive R3", "recursive_r3"),
    ]
    for tab, (label, prefix) in zip(tabs, msgs):
        with tab:
            st.caption("Response text only available for live runs; showing metrics for this sample.")
            st.metric("Quality", f"{bench[prefix + '_quality']:.2f}")
            st.caption(
                f"⏱ {bench[prefix + '_time']:.2f}s · 🔢 {bench[prefix + '_tokens']:,} tokens · "
                f"💲 ${bench[prefix + '_cost']:.5f}"
            )

# --------------------------------------------------------------------------- #
# Section 3 — Speedup KPI cards (Chart 5)
# --------------------------------------------------------------------------- #
st.markdown("---")
render_speedup_cards(speedups)

# --------------------------------------------------------------------------- #
# Section 4 — Performance charts
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("## 3. Performance Charts")

r1c, r2c = st.columns(2)
r1c.plotly_chart(chart_inference_time(bench), use_container_width=True)
r2c.plotly_chart(chart_tokens(bench), use_container_width=True)

r3c, r4c = st.columns(2)
r3c.plotly_chart(chart_quality(bench), use_container_width=True)
r4c.plotly_chart(chart_cost(bench), use_container_width=True)

r5c, r6c = st.columns(2)
r5c.plotly_chart(chart_scalability(speedups), use_container_width=True)
r6c.plotly_chart(chart_quality_vs_latency(bench), use_container_width=True)

# --------------------------------------------------------------------------- #
# Section 5 — Research summary
# --------------------------------------------------------------------------- #
st.markdown("---")
render_summary_metrics(summary, bench.get("backend_mode", "mock"))
