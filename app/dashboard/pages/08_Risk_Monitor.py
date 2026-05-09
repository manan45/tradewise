"""Risk Monitor — exposures, correlations, blackouts, breaker history."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Risk Monitor")
    st.caption("Live exposures, circuit breaker state, blackout windows")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Circuit Breaker", "OK")
        st.metric("Daily DD%", "—")
    with col2:
        st.metric("Open Positions", "—")
        st.metric("Portfolio Beta", "—")
    st.subheader("Active Blackouts")
    st.info("Wire to EventBlackout + CircuitBreaker for live data.")


if __name__ == "__main__":
    main()
