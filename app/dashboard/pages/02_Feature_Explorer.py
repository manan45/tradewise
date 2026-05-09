"""Feature Explorer — pick (name, version), inspect distribution + drift PSI."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Feature Explorer")
    st.caption("Inspect feature distributions and PSI drift")
    st.selectbox("Feature name", options=["realized_vol", "rsi", "atr", "zscore"])
    st.number_input("Version", value=1, min_value=1)
    st.info("Select a feature to view its distribution over a date range.")


if __name__ == "__main__":
    main()
