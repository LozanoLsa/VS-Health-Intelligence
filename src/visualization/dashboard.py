"""
Dashboard layout helpers — thin wrappers used by app.py.
"""
import streamlit as st


def set_page_config():
    st.set_page_config(
        page_title="Plant Operational Intelligence",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header():
    st.markdown(
        """
        <div style="text-align:center; padding:8px 0 4px;">
          <h2 style="margin:0; color:#0D1B2A; letter-spacing:1px;">
            Plant Operational Intelligence Dashboard
          </h2>
          <p style="margin:4px 0 0; color:#555; font-size:0.9rem;">
            MTBF &nbsp;&middot;&nbsp; MTTR &nbsp;&middot;&nbsp; Availability
            &nbsp;&middot;&nbsp; Downtime Risk &nbsp;&middot;&nbsp; Spatial Analytics
          </p>
        </div>
        <hr style="border:none; border-top:2px solid #E84C1F; margin:6px 0 12px;">
        """,
        unsafe_allow_html=True,
    )
