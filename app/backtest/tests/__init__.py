"""Backtest acceptance tests.

Cheaper-model TODO list:
- test_no_lookahead.py: replay a fixed CSV, assert no scenario reads any bar
  with ts > current.
- test_parity.py: run L3 backtest on a deterministic seed, run paper trading
  against the same fixture stream, assert identical equity curve to within
  1e-9 (BACKTESTING.md §5).
- test_survivorship.py: replay 2008 universe; assert delisted tickers do not
  appear in trades after delist date.
- test_friction_sanity.py: zero-friction backtest must beat full-friction
  backtest by exactly the sum of all simulated frictions.
"""
