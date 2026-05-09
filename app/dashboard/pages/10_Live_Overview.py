"""Live Overview — open sessions, working orders, account snapshot."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Live Overview")
    st.caption("Real-time open sessions, working orders, account equity")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Account Equity", "—")
    with col2:
        st.metric("Open Positions", "—")
    with col3:
        st.metric("Working Orders", "—")
    st.subheader("Open Sessions")
    st.info("Wire to broker adapter + session repository for live data.")


if __name__ == "__main__":
    main()
