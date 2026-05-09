"""
Streamlit helper components — styled metric cards and table formatting.
"""
import streamlit as st
import pandas as pd


CRITICAL_BG = "#FDECEA"
HEALTHY_BG  = "#EAFAF1"
ROW_STYLES  = {
    "Critical": f"background-color: {CRITICAL_BG};",
    "Healthy":  f"background-color: {HEALTHY_BG};",
}


def health_badge(status: str) -> str:
    colors = {"Healthy": "#2ECC71", "Monitor": "#F39C12", "Critical": "#E74C3C"}
    bg = colors.get(status, "#999")
    return (f'<span style="background:{bg}; color:white; padding:2px 8px; '
            f'border-radius:10px; font-size:0.75rem; font-weight:bold;">'
            f'{status}</span>')


def metric_row(label: str, value: str, delta: str = "", positive: bool = True):
    delta_color = "#2ECC71" if positive else "#E74C3C"
    delta_html = (f' <span style="color:{delta_color}; font-size:0.8rem;">'
                  f'{delta}</span>') if delta else ""
    st.markdown(
        f'<div style="padding:6px 0;">'
        f'<span style="color:#666; font-size:0.82rem;">{label}</span><br>'
        f'<span style="font-size:1.1rem; font-weight:bold;">{value}</span>'
        f'{delta_html}</div>',
        unsafe_allow_html=True)


def style_critical_rows(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    return df.style.set_properties(**{"font-size": "0.82rem"})
