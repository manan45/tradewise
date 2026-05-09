"""Data Health — staleness, coverage, gap detection per feed."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Data Health")
    st.caption("Freshness and coverage per feed")
    feeds = ["bars (equities)", "bars (futures)", "macro (FRED)", "news", "COT", "VIX term", "earnings calendar"]
    for feed in feeds:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{feed}**")
        with col2:
            st.write("—")
    st.info("Wire to DB to show live freshness.")


if __name__ == "__main__":
    main()
