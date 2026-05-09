"""Trade Journal — chronological list of closed sessions w/ outcomes + tags."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Trade Journal")
    st.caption("Closed sessions with outcomes, P&L, and scenario tags")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol filter", "")
    with col2:
        scenario = st.selectbox("Scenario", ["All", "trend_continuation",
                                              "mean_reversion", "vol_expansion"])
    st.info("Wire to session repository to show closed sessions.")


if __name__ == "__main__":
    main()
