"""Streamlit entry — landing page with system status + page navigation."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="TraderWise", layout="wide")
    st.title("TraderWise — Operator Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Status", "OK", delta=None)
    with col2:
        st.metric("Open Sessions", "—")
    with col3:
        st.metric("Today P&L", "—")
    with col4:
        st.metric("Circuit Breaker", "OK")

    st.markdown("---")
    st.info("Use the sidebar to navigate to individual monitoring pages.")


if __name__ == "__main__":
    main()
