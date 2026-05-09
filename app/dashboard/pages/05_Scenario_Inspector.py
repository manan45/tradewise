"""Scenario Inspector — fire rate, win rate, expectancy by scenario+regime."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Scenario Inspector")
    st.caption("Fire rate, win rate, expectancy by scenario and regime")
    scenarios = ["trend_continuation", "mean_reversion", "vol_expansion"]
    scenario = st.selectbox("Scenario", options=scenarios)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fire Rate", "—")
    with col2:
        st.metric("Win Rate", "—")
    with col3:
        st.metric("Avg Expectancy", "—")
    st.info("Wire to session repository to show live stats.")


if __name__ == "__main__":
    main()
