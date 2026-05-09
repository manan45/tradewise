"""Data pipelines — every external feed gets one module here.

Contract: each pipeline exposes ``async def run(asof: datetime, ...) -> int``
returning the number of rows ingested. They are idempotent (safe to re-run)
and asof-aware so backtests can replay them deterministically.

Scheduling: pipelines are kicked from app/main.py via APScheduler with each
feed's natural cadence (1m for prices, hourly for news, daily for COT, etc.).
"""
