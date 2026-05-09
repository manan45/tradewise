"""Backtest Viewer — equity curve, drawdowns, trade list, deflated Sharpe."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Backtest Viewer")
    st.caption("Equity curve, drawdowns, trade table, deflated Sharpe")
    st.info("Upload or select a BacktestResult JSON to visualise.")
    uploaded = st.file_uploader("BacktestResult JSON", type=["json"])
    if uploaded:
        import json
        result = json.load(uploaded)
        equity = result.get("equity_curve", [])
        if equity:
            import polars as pl
            df = pl.DataFrame({"date": [e[0] for e in equity],
                               "equity": [e[1] for e in equity]})
            st.line_chart(df.to_pandas().set_index("date"))
        metrics = result.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, f"{v:.4f}")
        trades = result.get("trades", [])
        if trades:
            st.dataframe(trades)


if __name__ == "__main__":
    main()
