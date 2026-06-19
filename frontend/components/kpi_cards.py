"""Reusable Streamlit UI components: speedup KPI cards and summary metrics."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st


def render_speedup_cards(speedups: List[Dict]) -> None:
    """Chart 5 — Inference Time Speedup across rounds, as large KPI cards."""
    st.markdown(
        "<h3 style='font-family:Georgia,serif;'>Inference Time Speedup of "
        "RecursiveMAS Across Three Recursion Rounds</h3>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(speedups))
    palette = ["#5B8DB8", "#3E6F94", "#1F4E79"]
    for col, card, color in zip(cols, speedups, palette):
        with col:
            st.markdown(
                f"""
                <div style="background:{color};border-radius:14px;padding:22px 16px;
                            text-align:center;color:white;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
                    <div style="font-size:14px;opacity:0.85;letter-spacing:1px;">{card['round_label'].upper()}</div>
                    <div style="font-size:46px;font-weight:700;line-height:1.1;margin:6px 0;">{card['speedup']}x</div>
                    <div style="font-size:13px;opacity:0.9;">{card['recursive_time']:.2f}s &middot; {card['time_saved_pct']:.1f}% saved</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_summary_metrics(summary: Dict, backend_mode: str) -> None:
    """Render the auto-generated research summary as KPI strip + narrative."""
    st.markdown("<h3 style='font-family:Georgia,serif;'>Research Summary</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Speedup", f"{summary['best_speedup']}x")
    c2.metric("Token Reduction", f"{summary['token_reduction_pct']}%")
    c3.metric("Cost Reduction", f"{summary['cost_reduction_pct']}%")
    c4.metric("Quality Δ", f"{summary['quality_delta_pct']}%")

    st.info(summary["narrative"])
    badge = "🟢 LIVE API" if backend_mode == "gemini" else "🟡 MOCK MODE (simulated)"
    st.caption(
        f"Backend: {backend_mode}  ·  {badge}  ·  "
        "All figures are computed from measured run data."
    )
