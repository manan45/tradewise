"""Label generators.

`triple_barrier` follows Lopez de Prado: forward-looking PT/SL barriers plus a
vertical (time) barrier; the first barrier hit determines the label. Sample
weights are uniqueness-corrected so overlapping events don't double-count.

`meta_labels` derives the meta-labeler training set: y_meta = 1 iff the base
model's directional call was right AND the trade hit PT before SL.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def triple_barrier(
    prices: pl.DataFrame,
    events: pl.DataFrame,
    pt_mult: float,
    sl_mult: float,
    horizon_bars: int,
    vol: pl.Series,
) -> pl.DataFrame:
    """Returns events with appended columns: t1, ret, label, sample_weight.

    prices: DataFrame with columns 'ts' (datetime), 'close' (float)
    events: DataFrame with column 'ts' (event entry timestamps)
    vol: per-bar volatility (same length / index as prices)
    """
    price_ts = prices["ts"].to_list()
    price_close = prices["close"].to_list()
    ts_to_idx = {ts: i for i, ts in enumerate(price_ts)}
    vol_list = vol.to_list()

    t1_list: list = []
    ret_list: list[float | None] = []
    label_list: list[int | None] = []

    for event_ts in events["ts"].to_list():
        start_idx = ts_to_idx.get(event_ts)
        if start_idx is None:
            t1_list.append(None)
            ret_list.append(None)
            label_list.append(None)
            continue

        entry_price = price_close[start_idx]
        v = vol_list[start_idx] if vol_list[start_idx] is not None else 0.0
        pt = entry_price * (1 + pt_mult * v)
        sl = entry_price * (1 - sl_mult * v)
        end_idx = min(start_idx + horizon_bars, len(price_ts) - 1)

        hit_ts = price_ts[end_idx]
        hit_label = 0  # time barrier
        hit_ret = (price_close[end_idx] - entry_price) / entry_price

        for i in range(start_idx + 1, end_idx + 1):
            c = price_close[i]
            if c >= pt:
                hit_ts = price_ts[i]
                hit_label = 1
                hit_ret = (c - entry_price) / entry_price
                break
            elif c <= sl:
                hit_ts = price_ts[i]
                hit_label = -1
                hit_ret = (c - entry_price) / entry_price
                break

        t1_list.append(hit_ts)
        ret_list.append(hit_ret)
        label_list.append(hit_label)

    result = events.with_columns([
        pl.Series("t1", t1_list),
        pl.Series("ret", ret_list, dtype=pl.Float64),
        pl.Series("label", label_list, dtype=pl.Int8),
    ])

    # Compute sample weights by uniqueness
    t1_series = result["t1"]
    weights = sample_weights_by_uniqueness(result, prices["ts"])
    return result.with_columns(pl.Series("sample_weight", weights))


def meta_labels(
    base_predictions: pl.DataFrame,
    triple_barrier_outcomes: pl.DataFrame,
) -> np.ndarray:
    """1 if base model predicted the correct sign AND PT barrier was hit."""
    outcomes = triple_barrier_outcomes["label"].to_numpy()
    sides = base_predictions["side"].to_numpy()  # +1 or -1
    # correct direction AND profit-take barrier
    return ((sides == np.sign(outcomes)) & (outcomes == 1)).astype(np.int8)


def sample_weights_by_uniqueness(events: pl.DataFrame,
                                 close_index: pl.Series) -> np.ndarray:
    """Down-weight overlapping events (mlfinlab.sampling.get_av_uniqueness)."""
    ts_list = events["ts"].to_list()
    t1_list = events["t1"].to_list()
    idx_ts = close_index.to_list()

    n = len(ts_list)
    weights = np.ones(n, dtype=np.float64)
    for i in range(n):
        t0 = ts_list[i]
        t1 = t1_list[i]
        if t0 is None or t1 is None:
            weights[i] = 0.0
            continue
        # Count how many other events overlap with this event's [t0, t1] window
        overlap_count = sum(
            1 for j in range(n)
            if ts_list[j] is not None and t1_list[j] is not None
            and ts_list[j] <= t1 and t1_list[j] >= t0
        )
        weights[i] = 1.0 / max(overlap_count, 1)

    # Normalise so sum = n (optional per spec "sum-normalisation optional")
    total = weights.sum()
    if total > 0:
        weights = weights / total * n
    return weights
