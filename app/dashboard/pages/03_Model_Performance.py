"""Model Performance — calibration curve, Brier, PSI, holdout metrics."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Model Performance")
    st.caption("Calibration, Brier score, PSI drift per model")
    models = ["vol_regime_a", "rv_expansion_b_prime", "direction_c",
              "cross_section_d", "commodity_pairs_e", "meta_labeler"]
    model = st.selectbox("Model", options=models)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Brier Score", "—")
    with col2:
        st.metric("Sharpe (OOF)", "—")
    with col3:
        st.metric("Hit Rate", "—")
    st.info("Wire to model registry to show live metrics.")


if __name__ == "__main__":
    main()
