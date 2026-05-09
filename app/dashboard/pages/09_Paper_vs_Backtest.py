"""Paper vs Backtest — parity check (BACKTESTING.md §5)."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Paper vs Backtest")
    st.caption("Per-day equity diff: paper trading vs L3 backtest on same stream")
    st.info(
        "Upload a paper equity CSV and a BacktestResult JSON to compare. "
        "Divergence > 1 cent/share indicates a parity bug."
    )
    c1, c2 = st.columns(2)
    with c1:
        paper = st.file_uploader("Paper equity CSV", type=["csv"])
    with c2:
        bt = st.file_uploader("Backtest JSON", type=["json"])
    if paper and bt:
        import json
        import polars as pl
        bt_result = json.load(bt)
        bt_equity = {e[0]: e[1] for e in bt_result.get("equity_curve", [])}
        st.write("Backtest rows:", len(bt_equity))


if __name__ == "__main__":
    main()
